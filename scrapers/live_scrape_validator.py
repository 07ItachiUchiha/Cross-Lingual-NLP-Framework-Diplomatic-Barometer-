"""Live scraping validator for diplomatically relevant official sources.

This module is intentionally separate from the pre-RAG analysis pipeline.
It validates real-time retrieval capability from selected official sources and
exports a machine-readable run report plus extracted documents.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import re
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options as ChromeOptions
    from selenium.webdriver.chrome.service import Service as ChromeService
except Exception:  # pragma: no cover - optional runtime dependency path
    webdriver = None
    ChromeOptions = None
    ChromeService = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LiveSource:
    name: str
    code: str
    rationale: str
    urls: List[str]
    link_keywords: List[str]
    channel: str = "official"
    max_start_urls: int = 3
    max_candidate_links: int = 30
    max_docs: int = 8
    min_delay_seconds: float = 1.8
    max_delay_seconds: float = 3.8
    allow_cross_domain_pdf: bool = False


class LiveScrapeValidator:
    """Validate and run live scraping across relevant official portals."""

    def __init__(self, output_dir: Optional[str] = None):
        project_root = Path(__file__).resolve().parent.parent
        self.output_dir = Path(output_dir) if output_dir else project_root / "data" / "raw"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.http_cache_db = self.output_dir / "live_scrape_http_cache.sqlite"
        self._init_http_cache()

        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
                ),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "gzip, deflate, br",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
            }
        )

        self.sources = self._build_sources()
        self._browser = None
        self._browser_available = webdriver is not None
        self._browser_init_error = None
        self.primary_source_codes = {"MEA", "MOFA", "EMBJPIN", "PIB", "JETRO"}
        self.blocked_skip_hours = 24

    def _init_http_cache(self) -> None:
        with sqlite3.connect(self.http_cache_db) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS url_cache (
                    url_hash TEXT PRIMARY KEY,
                    url TEXT NOT NULL,
                    html TEXT NOT NULL,
                    status_code INTEGER,
                    fetched_at_utc TEXT NOT NULL,
                    fetch_method TEXT
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_url_cache_time ON url_cache(fetched_at_utc)"
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS blocked_url (
                    url_hash TEXT NOT NULL,
                    source_code TEXT NOT NULL,
                    url TEXT NOT NULL,
                    status_code INTEGER,
                    first_seen_utc TEXT NOT NULL,
                    last_seen_utc TEXT NOT NULL,
                    hits INTEGER NOT NULL,
                    PRIMARY KEY (url_hash, source_code)
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_blocked_url_last_seen ON blocked_url(last_seen_utc)"
            )
            conn.commit()

    def _blocked_cutoff_iso(self, hours: int) -> str:
        cutoff_ts = time.time() - (hours * 3600)
        return datetime.fromtimestamp(cutoff_ts, timezone.utc).isoformat()

    def _should_skip_url(self, url: str, source_code: str) -> bool:
        url_hash = self._hash_url(url)
        cutoff_iso = self._blocked_cutoff_iso(self.blocked_skip_hours)
        with sqlite3.connect(self.http_cache_db) as conn:
            row = conn.execute(
                """
                SELECT last_seen_utc, status_code, hits
                FROM blocked_url
                WHERE url_hash = ? AND source_code = ? AND last_seen_utc >= ?
                """,
                (url_hash, source_code, cutoff_iso),
            ).fetchone()
        return row is not None

    def _record_blocked_url(self, url: str, source_code: str, status_code: Optional[int]) -> None:
        if status_code not in {403, 429}:
            return
        url_hash = self._hash_url(url)
        now_iso = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.http_cache_db) as conn:
            existing = conn.execute(
                "SELECT first_seen_utc, hits FROM blocked_url WHERE url_hash = ? AND source_code = ?",
                (url_hash, source_code),
            ).fetchone()
            if existing:
                first_seen_utc, hits = existing
                conn.execute(
                    """
                    UPDATE blocked_url
                    SET url = ?, status_code = ?, last_seen_utc = ?, hits = ?
                    WHERE url_hash = ? AND source_code = ?
                    """,
                    (url, int(status_code), now_iso, int(hits) + 1, url_hash, source_code),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO blocked_url(url_hash, source_code, url, status_code, first_seen_utc, last_seen_utc, hits)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (url_hash, source_code, url, int(status_code), now_iso, now_iso, 1),
                )
            conn.commit()

    @staticmethod
    def _hash_url(url: str) -> str:
        return hashlib.sha256(url.encode("utf-8")).hexdigest()

    def _cache_get_html(self, url: str, max_age_hours: int = 72) -> Optional[str]:
        cutoff_ts = time.time() - (max_age_hours * 3600)
        cutoff_iso = datetime.fromtimestamp(cutoff_ts, timezone.utc).isoformat()
        url_hash = self._hash_url(url)
        with sqlite3.connect(self.http_cache_db) as conn:
            row = conn.execute(
                "SELECT html FROM url_cache WHERE url_hash = ? AND fetched_at_utc >= ?",
                (url_hash, cutoff_iso),
            ).fetchone()
        if row and row[0]:
            return row[0]
        return None

    def _cache_put_html(self, url: str, html: str, status_code: int, fetch_method: str) -> None:
        if not html:
            return
        with sqlite3.connect(self.http_cache_db) as conn:
            conn.execute(
                """
                INSERT INTO url_cache(url_hash, url, html, status_code, fetched_at_utc, fetch_method)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(url_hash) DO UPDATE SET
                    url = excluded.url,
                    html = excluded.html,
                    status_code = excluded.status_code,
                    fetched_at_utc = excluded.fetched_at_utc,
                    fetch_method = excluded.fetch_method
                """,
                (
                    self._hash_url(url),
                    url,
                    html,
                    status_code,
                    datetime.now(timezone.utc).isoformat(),
                    fetch_method,
                ),
            )
            conn.commit()

    @staticmethod
    def _build_sources() -> List[LiveSource]:
        """Select only sources relevant to India-Japan diplomatic discourse."""
        return [
            LiveSource(
                name="MEA India",
                code="MEA",
                rationale="Primary official source for India's bilateral statements and press releases.",
                urls=[
                    "https://www.mea.gov.in/press-releases.htm",
                    "https://www.mea.gov.in/bilateral-documents.htm?dtl/1/india-japan-relations",
                    "https://www.mea.gov.in/SiteMap.htm",
                    "https://www.mea.gov.in/rss.xml",
                    "https://www.mea.gov.in/",
                ],
                link_keywords=["japan", "joint", "statement", "summit", "bilateral", "release", "pdf"],
                channel="official_rss_index",
                max_start_urls=4,
                max_candidate_links=35,
                max_docs=10,
            ),
            LiveSource(
                name="MOFA Japan",
                code="MOFA",
                rationale="Primary official Japanese foreign affairs source for India-related diplomacy.",
                urls=[
                    "https://www.mofa.go.jp/region/asia-paci/india/index.html",
                    "https://www.mofa.go.jp/press/release/index.html",
                    "https://www.mofa.go.jp/sitemap.xml",
                    "https://www.mofa.go.jp/",
                ],
                link_keywords=["india", "joint", "statement", "summit", "press", "release", "pdf"],
                channel="official_index",
                max_start_urls=3,
                max_candidate_links=30,
                max_docs=8,
                min_delay_seconds=2.0,
                max_delay_seconds=4.2,
            ),
            LiveSource(
                name="Embassy of Japan in India",
                code="EMBJPIN",
                rationale="Official mission-level announcements and bilateral event updates.",
                urls=[
                    "https://www.in.emb-japan.go.jp/",
                    "https://www.in.emb-japan.go.jp/itpr_en/Japan_India_Relations.html",
                    "https://www.in.emb-japan.go.jp/sitemap.xml",
                ],
                link_keywords=["india", "japan", "press", "release", "statement", "event", "pdf"],
                channel="official_index",
                max_start_urls=3,
                max_candidate_links=25,
                max_docs=6,
                min_delay_seconds=2.0,
                max_delay_seconds=4.0,
            ),
            LiveSource(
                name="PIB India",
                code="PIB",
                rationale="Official India government press releases containing diplomatic updates.",
                urls=[
                    "https://pib.gov.in/",
                    "https://pib.gov.in/PressReleasePage.aspx",
                    "https://pib.gov.in/RssMain.aspx",
                ],
                link_keywords=["japan", "bilateral", "joint", "statement", "prime minister", "summit", "release"],
                channel="official_rss_index",
                max_start_urls=3,
                max_candidate_links=30,
                max_docs=8,
            ),
            LiveSource(
                name="JETRO Reports",
                code="JETRO",
                rationale="Official Japan trade/publication source relevant to India-Japan economic diplomacy.",
                urls=[
                    "https://www.jetro.go.jp/en/reports/",
                    "https://www.jetro.go.jp/en/invest/",
                ],
                link_keywords=["india", "japan", "trade", "report", "investment", "pdf"],
                channel="official_reports",
                max_start_urls=2,
                max_candidate_links=20,
                max_docs=8,
            ),
            LiveSource(
                name="MOFA Archived Mirror",
                code="MOFA_ARCHIVE",
                rationale="Archived snapshots for blocked MOFA pages to preserve partial coverage.",
                urls=[
                    "https://web.archive.org/web/20240101000000/https://www.mofa.go.jp/region/asia-paci/india/index.html",
                    "https://web.archive.org/web/20240101000000/https://www.mofa.go.jp/press/release/index.html",
                ],
                link_keywords=["india", "joint", "statement", "press", "release", "pdf"],
                channel="archive_mirror",
                max_start_urls=2,
                max_candidate_links=20,
                max_docs=5,
                min_delay_seconds=1.0,
                max_delay_seconds=2.0,
                allow_cross_domain_pdf=True,
            ),
            LiveSource(
                name="Embassy Archived Mirror",
                code="EMBJPIN_ARCHIVE",
                rationale="Archived snapshots for blocked Embassy pages to preserve partial coverage.",
                urls=[
                    "https://web.archive.org/web/20240101000000/https://www.in.emb-japan.go.jp/",
                    "https://web.archive.org/web/20240101000000/https://www.in.emb-japan.go.jp/itpr_en/Japan_India_Relations.html",
                ],
                link_keywords=["india", "japan", "press", "statement", "event", "pdf"],
                channel="archive_mirror",
                max_start_urls=2,
                max_candidate_links=20,
                max_docs=5,
                min_delay_seconds=1.0,
                max_delay_seconds=2.0,
                allow_cross_domain_pdf=True,
            ),
        ]

    @staticmethod
    def _is_trusted_pdf_domain(url: str) -> bool:
        try:
            netloc = urlparse(url).netloc.lower()
        except Exception:
            return False
        trusted = {
            "www.mofa.go.jp",
            "www.in.emb-japan.go.jp",
            "www.mea.gov.in",
            "pib.gov.in",
            "www.jetro.go.jp",
            "web.archive.org",
        }
        return netloc in trusted

    def _get_with_retry(
        self,
        url: str,
        max_retries: int = 3,
        timeout: int = 20,
        min_delay_seconds: float = 1.8,
        max_delay_seconds: float = 3.8,
    ) -> Tuple[Optional[requests.Response], Optional[str], Optional[int], List[Dict]]:
        """HTTP GET with retry, backoff and SSL fallback for hostile endpoints."""
        last_error = None
        last_status = None
        attempts: List[Dict] = []

        for attempt in range(max_retries):
            if attempt > 0:
                time.sleep(random.uniform(min_delay_seconds, max_delay_seconds))
            try:
                response = self.session.get(url, timeout=timeout, allow_redirects=True)
                last_status = response.status_code
                attempts.append(
                    {
                        "attempt": attempt + 1,
                        "status_code": response.status_code,
                        "method": "http",
                    }
                )
                if response.status_code in (403, 429, 500, 502, 503, 504):
                    raise requests.HTTPError(f"HTTP {response.status_code}")
                response.raise_for_status()
                return response, None, response.status_code, attempts
            except requests.exceptions.SSLError as exc:
                last_error = f"SSL error: {exc}"
                try:
                    response = self.session.get(url, timeout=timeout, allow_redirects=True, verify=False)
                    last_status = response.status_code
                    attempts.append(
                        {
                            "attempt": attempt + 1,
                            "status_code": response.status_code,
                            "method": "ssl_fallback",
                        }
                    )
                    if response.status_code < 400:
                        logger.warning(f"SSL fallback used for {url}")
                        return response, None, response.status_code, attempts
                    last_error = f"HTTP {response.status_code} after SSL fallback"
                except Exception as ssl_exc:
                    last_error = f"SSL fallback failed: {ssl_exc}"
            except Exception as exc:
                last_error = str(exc)

            if attempt < max_retries - 1:
                time.sleep(1.5 ** attempt)

        logger.warning(f"Failed to fetch {url}: {last_error}")
        return None, last_error, last_status, attempts

    def _get_browser(self):
        if not self._browser_available:
            return None
        if self._browser is not None:
            return self._browser

        try:
            options = ChromeOptions()
            options.add_argument("--headless=new")
            options.add_argument("--disable-gpu")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--window-size=1920,1080")
            options.add_argument(
                "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
            )

            service = ChromeService()
            self._browser = webdriver.Chrome(service=service, options=options)
            self._browser.set_page_load_timeout(30)
            return self._browser
        except Exception as exc:
            self._browser_init_error = str(exc)
            self._browser_available = False
            logger.warning(f"Browser fallback unavailable: {exc}")
            return None

    def _get_with_browser(self, url: str, wait_seconds: float = 2.5) -> Optional[str]:
        browser = self._get_browser()
        if browser is None:
            return None

        try:
            browser.get(url)
            time.sleep(wait_seconds)
            return browser.page_source
        except Exception as exc:
            logger.warning(f"Browser fetch failed for {url}: {exc}")
            return None

    def _fetch_html_with_fallback(self, url: str, source: LiveSource, phase: str) -> Tuple[Optional[str], Dict]:
        if self._should_skip_url(url, source.code):
            return None, {
                "url": url,
                "phase": phase,
                "method": "skip_list",
                "from_cache": False,
                "fallback_used": False,
                "status_code": None,
                "error": f"Skipped (recently blocked within {self.blocked_skip_hours}h)",
                "attempts": [],
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }

        cached = self._cache_get_html(url)
        if cached:
            return cached, {
                "url": url,
                "phase": phase,
                "method": "cache",
                "from_cache": True,
                "fallback_used": False,
                "status_code": 200,
                "error": None,
                "attempts": [],
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }

        response, err, status, attempts = self._get_with_retry(
            url,
            max_retries=3,
            timeout=20,
            min_delay_seconds=source.min_delay_seconds,
            max_delay_seconds=source.max_delay_seconds,
        )
        if response is not None:
            html = response.text
            self._cache_put_html(url, html, response.status_code, "http")
            return html, {
                "url": url,
                "phase": phase,
                "method": "http",
                "from_cache": False,
                "fallback_used": False,
                "status_code": response.status_code,
                "error": None,
                "attempts": attempts,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }

        html = self._get_with_browser(url)
        if html:
            logger.info(f"Browser fallback used for {url}")
            self._cache_put_html(url, html, 200, "browser_fallback")
            return html, {
                "url": url,
                "phase": phase,
                "method": "browser_fallback",
                "from_cache": False,
                "fallback_used": True,
                "status_code": 200,
                "error": err,
                "attempts": attempts,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }

        self._record_blocked_url(url, source.code, status)

        return None, {
            "url": url,
            "phase": phase,
            "method": "failed",
            "from_cache": False,
            "fallback_used": False,
            "status_code": status,
            "error": err,
            "attempts": attempts,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }

    @staticmethod
    def _normalize_url(base_url: str, href: str) -> Optional[str]:
        if not href:
            return None
        href = href.strip()
        if href.startswith("javascript:") or href.startswith("mailto:"):
            return None
        normalized = urljoin(base_url, href)
        if normalized.startswith("http://"):
            normalized = "https://" + normalized[len("http://") :]

        lowered = normalized.lower()
        if "web.archive.org" in lowered and "/screenshot/" in lowered:
            return None

        blocked_ext = (
            ".jpg",
            ".jpeg",
            ".png",
            ".gif",
            ".webp",
            ".svg",
            ".css",
            ".js",
            ".ico",
            ".woff",
            ".woff2",
            ".ttf",
            ".eot",
            ".mp4",
            ".mp3",
        )
        if any(lowered.endswith(ext) for ext in blocked_ext):
            return None
        return normalized

    @staticmethod
    def _is_same_domain(url_a: str, url_b: str) -> bool:
        try:
            return urlparse(url_a).netloc == urlparse(url_b).netloc
        except Exception:
            return False

    def _extract_candidate_links(self, source: LiveSource, page_url: str, html: str, max_links: int = 30) -> List[str]:
        soup = BeautifulSoup(html, "html.parser")
        candidates: List[str] = []

        def allow_url(href: str) -> bool:
            if href.lower().endswith(".pdf"):
                if source.allow_cross_domain_pdf and self._is_trusted_pdf_domain(href):
                    return True
                return self._is_same_domain(page_url, href)
            return self._is_same_domain(page_url, href)

        for anchor in soup.find_all("a", href=True):
            href = self._normalize_url(page_url, anchor.get("href"))
            if not href:
                continue

            text = " ".join(anchor.get_text(" ", strip=True).split()).lower()
            combined = f"{href.lower()} {text}"

            if href.lower().endswith(".pdf") and allow_url(href):
                candidates.append(href)
                continue

            if any(keyword in combined for keyword in source.link_keywords):
                if allow_url(href):
                    candidates.append(href)

        for link_tag in soup.find_all("link"):
            href = self._normalize_url(page_url, link_tag.get("href", ""))
            if not href:
                continue
            combined = f"{href.lower()} {str(link_tag)}"
            if any(keyword in combined for keyword in source.link_keywords):
                candidates.append(href)

        # keep order, remove duplicates
        deduped: List[str] = []
        seen = set()
        for link in candidates:
            if link not in seen:
                seen.add(link)
                deduped.append(link)

        deduped.sort(key=lambda item: 0 if item.lower().endswith(".pdf") else 1)

        return deduped[:max_links]

    @staticmethod
    def _extract_text_block(soup: BeautifulSoup) -> str:
        selectors = [
            "article",
            "main",
            "div.article",
            "div.content",
            "div#content",
            "div.main",
            "section",
        ]

        for selector in selectors:
            node = soup.select_one(selector)
            if node:
                text = " ".join(node.get_text(" ", strip=True).split())
                if len(text) > 200:
                    return text

        text = " ".join(soup.get_text(" ", strip=True).split())
        return text[:6000]

    @staticmethod
    def _extract_date(text: str) -> Optional[str]:
        patterns = [
            r"\b(\d{4}-\d{2}-\d{2})\b",
            r"\b(\d{1,2}\s+[A-Za-z]+\s+\d{4})\b",
            r"\b([A-Za-z]+\s+\d{1,2},\s*\d{4})\b",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        return None

    def _extract_document(self, source: LiveSource, url: str) -> Tuple[Optional[Dict], Dict]:
        if url.lower().endswith(".pdf"):
            date = self._extract_date(url) or datetime.now(timezone.utc).date().isoformat()
            filename = url.split("/")[-1] or "document.pdf"
            return (
                {
                    "date": date,
                    "title": filename,
                    "location": "",
                    "signatories": "",
                    "content": f"PDF reference from official source: {url}",
                    "source": source.code,
                    "url": url,
                    "source_name": source.name,
                    "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                },
                {
                    "url": url,
                    "phase": "document",
                    "method": "pdf_link",
                    "from_cache": False,
                    "fallback_used": False,
                    "status_code": 200,
                    "error": None,
                    "attempts": [],
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                },
            )

        html, fetch_event = self._fetch_html_with_fallback(url, source, phase="document")
        if not html:
            return None, fetch_event

        soup = BeautifulSoup(html, "html.parser")
        title_node = soup.find("h1") or soup.find("title")
        title = " ".join(title_node.get_text(" ", strip=True).split()) if title_node else ""
        content = self._extract_text_block(soup)

        if len(content) < 180:
            return None

        raw_date = self._extract_date(html)
        date = raw_date or datetime.now(timezone.utc).date().isoformat()

        return (
            {
                "date": date,
                "title": title[:500] or f"{source.name} document",
                "location": "",
                "signatories": "",
                "content": content,
                "source": source.code,
                "url": url,
                "source_name": source.name,
                "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
            },
            fetch_event,
        )

    def scrape_live(self, max_docs_per_source: int = 8) -> Dict:
        run_started = datetime.now(timezone.utc)
        all_docs: List[Dict] = []
        source_reports: List[Dict] = []
        blocked_sources: List[Dict] = []
        backfill_docs = 0
        global_doc_cap = 120

        for source in self.sources:
            logger.info(f"[LIVE] Source: {source.name}")
            source_doc_count = 0
            source_status = "unreachable"
            source_errors: List[str] = []
            visited_links = 0
            source_started = datetime.now(timezone.utc)
            start_failures = 0
            blocked_failures = 0
            fetch_log: List[Dict] = []
            blocked_urls: List[str] = []
            docs_cap = min(max_docs_per_source, source.max_docs)

            for start_url in source.urls[: source.max_start_urls]:
                if len(all_docs) >= global_doc_cap:
                    break

                start_html, start_event = self._fetch_html_with_fallback(start_url, source, phase="start")
                fetch_log.append(start_event)
                if not start_html:
                    start_failures += 1
                    if start_event.get("status_code") in {403, 429}:
                        blocked_failures += 1
                        blocked_urls.append(start_url)
                    source_errors.append(f"Failed start URL: {start_url} ({start_event.get('error')})")
                    continue

                source_status = "reachable"
                candidates = self._extract_candidate_links(
                    source, start_url, start_html, max_links=source.max_candidate_links
                )
                visited_links += len(candidates)

                for link in candidates:
                    if source_doc_count >= docs_cap or len(all_docs) >= global_doc_cap:
                        break

                    time.sleep(random.uniform(source.min_delay_seconds, source.max_delay_seconds))
                    doc, event = self._extract_document(source, link)
                    fetch_log.append(event)
                    if event.get("status_code") in {403, 429}:
                        blocked_urls.append(link)

                    if doc:
                        all_docs.append(doc)
                        source_doc_count += 1

                if source_doc_count >= docs_cap:
                    break

            if source_doc_count > 0:
                source_status = "ok"
            elif source_status == "reachable":
                source_status = "reachable_no_docs"
            elif blocked_failures > 0:
                source_status = "blocked"

            if source_status == "blocked":
                blocked_sources.append(
                    {
                        "code": source.code,
                        "source": source.name,
                        "blocked_url_count": len(set(blocked_urls)),
                        "blocked_at_utc": datetime.now(timezone.utc).isoformat(),
                    }
                )

            if source.channel == "archive_mirror":
                backfill_docs += source_doc_count

            source_reports.append(
                {
                    "source": source.name,
                    "code": source.code,
                    "channel": source.channel,
                    "rationale": source.rationale,
                    "status": source_status,
                    "documents": source_doc_count,
                    "candidate_links_seen": visited_links,
                    "start_failures": start_failures,
                    "blocked_failures": blocked_failures,
                    "blocked_urls": sorted(list(set(blocked_urls)))[:50],
                    "errors": source_errors,
                    "urls_tested": source.urls,
                    "policy": {
                        "max_start_urls": source.max_start_urls,
                        "max_candidate_links": source.max_candidate_links,
                        "max_docs": docs_cap,
                        "delay_seconds": [source.min_delay_seconds, source.max_delay_seconds],
                    },
                    "fetch_log": fetch_log,
                    "started_at_utc": source_started.isoformat(),
                    "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                }
            )

        output_csv = self.output_dir / f"live_scrape_{run_started.strftime('%Y%m%d_%H%M%S')}.csv"
        output_report = self.output_dir / f"live_scrape_report_{run_started.strftime('%Y%m%d_%H%M%S')}.json"
        blocked_output = self.output_dir / f"blocked_sources_{run_started.strftime('%Y%m%d_%H%M%S')}.json"

        if all_docs:
            dedup_df = pd.DataFrame(all_docs).drop_duplicates(subset=["source", "url"], keep="first")
            dedup_df.to_csv(output_csv, index=False, encoding="utf-8")
            all_docs = dedup_df.to_dict("records")

        successful_primary_sources = sum(
            1
            for r in source_reports
            if r["code"] in self.primary_source_codes and r["status"] == "ok"
        )
        blocked_primary_sources = sum(
            1
            for r in source_reports
            if r["code"] in self.primary_source_codes and r["status"] == "blocked"
        )
        backfill_sources_ok = sum(
            1 for r in source_reports if r["channel"] == "archive_mirror" and r["status"] == "ok"
        )

        pass_flag = (
            len(all_docs) >= 8
            and successful_primary_sources >= 2
            and (blocked_primary_sources == 0 or backfill_sources_ok >= 1)
        )

        report = {
            "run_started_utc": run_started.isoformat(),
            "run_finished_utc": datetime.now(timezone.utc).isoformat(),
            "selected_sources": [
                {
                    "name": s.name,
                    "code": s.code,
                    "rationale": s.rationale,
                    "channel": s.channel,
                    "policy": {
                        "max_start_urls": s.max_start_urls,
                        "max_candidate_links": s.max_candidate_links,
                        "max_docs": s.max_docs,
                        "delay_seconds": [s.min_delay_seconds, s.max_delay_seconds],
                    },
                }
                for s in self.sources
            ],
            "summary": {
                "total_documents": len(all_docs),
                "successful_sources": sum(1 for r in source_reports if r["status"] == "ok"),
                "reachable_sources": sum(1 for r in source_reports if r["status"] in {"ok", "reachable_no_docs"}),
                "failed_sources": sum(1 for r in source_reports if r["status"] in {"unreachable", "blocked"}),
                "blocked_sources": sum(1 for r in source_reports if r["status"] == "blocked"),
                "successful_primary_sources": successful_primary_sources,
                "blocked_primary_sources": blocked_primary_sources,
                "backfill_sources_ok": backfill_sources_ok,
                "backfill_documents": backfill_docs,
            },
            "sources": source_reports,
            "blocked_sources": blocked_sources,
            "outputs": {
                "documents_csv": str(output_csv) if all_docs else None,
                "report_json": str(output_report),
                "blocked_sources_json": str(blocked_output),
                "http_cache_db": str(self.http_cache_db),
            },
            "pass": pass_flag,
        }

        with open(output_report, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)

        with open(blocked_output, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "run_started_utc": run_started.isoformat(),
                    "run_finished_utc": datetime.now(timezone.utc).isoformat(),
                    "blocked_sources": blocked_sources,
                },
                handle,
                indent=2,
            )

        if self._browser is not None:
            try:
                self._browser.quit()
            except Exception:
                pass
            self._browser = None

        return report


if __name__ == "__main__":
    validator = LiveScrapeValidator()
    result = validator.scrape_live()
    print(json.dumps(result["summary"], indent=2))
    print("PASS" if result["pass"] else "FAIL")
