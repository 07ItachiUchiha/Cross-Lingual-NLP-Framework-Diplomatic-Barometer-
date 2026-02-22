"""Resolve URLs for the local corpus titles.

Goal
- Given the local corpus CSV (India-Japan), try to find official URLs for each row.
- Only write URLs when the match is high-confidence to avoid wrong links.

Approach
- Crawl a small set of official index pages (MEA bilateral docs, MOFA Japan-India relations).
- Extract candidate links, fetch pages, and compute title similarity.
- Persist results into the canonical corpus (data/raw/india_japan_documents_canonical.csv).

This module is intentionally conservative: it is better to leave URLs blank than to attach an incorrect link.
"""

from __future__ import annotations

import logging
import random
import re
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
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
except Exception:  # pragma: no cover
    webdriver = None
    ChromeOptions = None
    ChromeService = None

logger = logging.getLogger(__name__)


@dataclass
class SourceSeed:
    code: str
    name: str
    urls: List[str]


class CorpusURLResolver:
    def __init__(self, data_dir: Optional[str] = None):
        project_root = Path(__file__).resolve().parent.parent
        self.data_dir = Path(data_dir) if data_dir else project_root / "data" / "raw"

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

        self._browser = None
        self._browser_available = webdriver is not None

        self.seeds: List[SourceSeed] = [
            SourceSeed(
                code="MEA",
                name="MEA India",
                urls=[
                    "https://www.mea.gov.in/bilateral-documents.htm?dtl/1/india-japan-relations",
                    "https://www.mea.gov.in/bilateral-documents.htm",
                    "https://www.mea.gov.in/press-releases.htm",
                ],
            ),
            SourceSeed(
                code="MOFA",
                name="MOFA Japan",
                urls=[
                    "https://www.mofa.go.jp/region/asia-paci/india/index.html",
                    "https://www.mofa.go.jp/press/release/index.html",
                ],
            ),
        ]

    @staticmethod
    def _norm_title(text: object) -> str:
        s = str(text or "").strip().lower()
        s = re.sub(r"\s+", " ", s)
        return s

    @staticmethod
    def _same_domain(a: str, b: str) -> bool:
        try:
            return urlparse(a).netloc == urlparse(b).netloc
        except Exception:
            return False

    @staticmethod
    def _normalize_url(base_url: str, href: str) -> Optional[str]:
        if not href:
            return None
        href = href.strip()
        if href.startswith("javascript:") or href.startswith("mailto:"):
            return None
        url = urljoin(base_url, href)
        if url.startswith("http://"):
            url = "https://" + url[len("http://") :]

        lowered = url.lower()
        if any(
            lowered.endswith(ext)
            for ext in (
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
        ):
            return None
        return url

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
            self._browser_available = False
            logger.warning(f"Browser fallback unavailable: {exc}")
            return None

    def _fetch_html(self, url: str) -> Optional[str]:
        try:
            r = self.session.get(url, timeout=20, allow_redirects=True)
            if r.status_code < 400:
                return r.text
        except Exception:
            pass

        browser = self._get_browser()
        if browser is None:
            return None
        try:
            browser.get(url)
            time.sleep(random.uniform(2.0, 3.0))
            return browser.page_source
        except Exception:
            return None

    @staticmethod
    def _gdelt_datetime_range_for_year(year: int) -> Optional[Tuple[str, str]]:
        if year < 1900 or year > 2100:
            return None
        # GDELT format: YYYYMMDDHHMMSS
        return (f"{year}0101000000", f"{year}1231235959")

    def _gdelt_search_urls(self, query: str, maxrecords: int = 50, year: Optional[int] = None) -> List[str]:
        """Search GDELT DOC 2.0 for candidate URLs.

        This returns URLs only (discovery), which we later validate by fetching titles
        and applying conservative similarity matching.
        """

        url = "https://api.gdeltproject.org/api/v2/doc/doc"
        params: Dict[str, object] = {
            "query": query,
            "mode": "artlist",
            "format": "json",
            "maxrecords": int(max(1, min(maxrecords, 250))),
            "sort": "datedesc",
        }
        if year is not None:
            rng = self._gdelt_datetime_range_for_year(int(year))
            if rng:
                params["startdatetime"], params["enddatetime"] = rng

        try:
            r = self.session.get(url, params=params, timeout=25)
            if r.status_code >= 400:
                return []
            payload = r.json()
        except Exception:
            return []

        articles = payload.get("articles") if isinstance(payload, dict) else None
        if not isinstance(articles, list):
            return []

        out: List[str] = []
        seen = set()
        for row in articles:
            cand = str((row or {}).get("url") or "").strip()
            if not cand:
                continue
            if cand not in seen:
                seen.add(cand)
                out.append(cand)
        return out

    @staticmethod
    def _host_matches_any(url: str, allowed_domains: List[str]) -> bool:
        try:
            host = urlparse(url).netloc.lower()
        except Exception:
            return False
        for dom in allowed_domains:
            dom = str(dom or "").lower().strip()
            if not dom:
                continue
            if host == dom or host.endswith("." + dom):
                return True
        return False

    def _gdelt_candidates_for_row(self, title: str, year: Optional[int], source: str) -> List[str]:
        # Keep query conservative and short; title may not exist verbatim online.
        # Strategy: quoted title + fallback keywords for India-Japan.
        title_clean = str(title or "").strip()
        if not title_clean:
            return []

        allowed_domains: List[str]
        if source == "MEA":
            allowed_domains = ["mea.gov.in"]
        elif source == "MOFA":
            allowed_domains = ["mofa.go.jp"]
        else:
            return []

        # Try 2 queries: exact phrase, then keyword mix.
        q1 = f'"{title_clean}" (India OR Japanese OR Japan)'

        # keyword mix: keep the first 10 non-trivial words
        words = [w for w in re.split(r"[^A-Za-z0-9]+", title_clean) if len(w) >= 4]
        words = words[:10]
        mix = " ".join(words)
        q2 = f"({mix}) India Japan"

        urls: List[str] = []
        urls.extend(self._gdelt_search_urls(q1, maxrecords=60, year=year))
        if len(urls) < 6:
            urls.extend(self._gdelt_search_urls(q2, maxrecords=60, year=year))

        # Filter to official domains
        filtered: List[str] = []
        for u in urls:
            if self._host_matches_any(u, allowed_domains):
                filtered.append(u)

        # Dedupe while preserving order
        out: List[str] = []
        seen = set()
        for u in filtered:
            if u not in seen:
                seen.add(u)
                out.append(u)
        return out[:40]

    def _extract_candidates(self, seed_url: str, html: str, max_links: int = 120) -> List[str]:
        soup = BeautifulSoup(html, "html.parser")
        candidates: List[str] = []

        for a in soup.find_all("a", href=True):
            href = self._normalize_url(seed_url, a.get("href"))
            if not href:
                continue
            if not self._same_domain(seed_url, href):
                continue
            candidates.append(href)

        # preserve order, dedupe
        out = []
        seen = set()
        for c in candidates:
            if c not in seen:
                seen.add(c)
                out.append(c)
        return out[:max_links]

    def _fetch_page_titles(self, urls: List[str], budget: int = 160) -> List[Tuple[str, str]]:
        results: List[Tuple[str, str]] = []
        for idx, url in enumerate(urls[:budget]):
            html = self._fetch_html(url)
            if not html:
                continue
            soup = BeautifulSoup(html, "html.parser")
            title_node = soup.find("h1") or soup.find("title")
            title = self._norm_title(title_node.get_text(" ", strip=True) if title_node else "")
            if len(title) < 6:
                continue
            results.append((title, url))
        return results

    @staticmethod
    def _best_unique_match(query_title: str, candidates: List[Tuple[str, str]]) -> Tuple[Optional[str], float]:
        best_ratio = 0.0
        second_ratio = 0.0
        best_url: Optional[str] = None

        for cand_title, cand_url in candidates:
            ratio = SequenceMatcher(None, query_title, cand_title).ratio()
            if ratio > best_ratio:
                second_ratio = best_ratio
                best_ratio = ratio
                best_url = cand_url
            elif ratio > second_ratio:
                second_ratio = ratio

        if best_url is None:
            return None, 0.0

        # conservative: only accept a very strong, clearly unique match
        if best_ratio >= 0.93 and (best_ratio - second_ratio) >= 0.03:
            return best_url, best_ratio

        return None, best_ratio

    def resolve(
        self,
        corpus_df: pd.DataFrame,
        max_urls_per_source: int = 220,
        use_gdelt: bool = True,
        max_gdelt_urls_per_row: int = 12,
    ) -> Dict:
        """Return mapping rows and updated corpus df (urls filled when confident)."""
        if corpus_df is None or len(corpus_df) == 0:
            return {"resolved": 0, "attempted": 0, "rows": []}

        corpus = corpus_df.copy()
        if "url" not in corpus.columns:
            corpus["url"] = ""
        corpus["url"] = corpus["url"].fillna("").astype(str).str.strip().replace({"nan": "", "None": "", "none": ""})

        per_source_candidates: Dict[str, List[Tuple[str, str]]] = {}

        for seed in self.seeds:
            seed_candidates: List[str] = []
            for u in seed.urls:
                html = self._fetch_html(u)
                if not html:
                    continue
                seed_candidates.extend(self._extract_candidates(u, html, max_links=max_urls_per_source))

            # fetch titles for candidates
            pages = self._fetch_page_titles(seed_candidates, budget=max_urls_per_source)
            if pages:
                per_source_candidates[seed.code] = pages
                logger.info(f"Resolver candidate pages: {seed.code} -> {len(pages)}")

        attempted = 0
        resolved = 0
        out_rows = []

        gdelt_cache: Dict[Tuple[str, str, str], List[str]] = {}

        for idx, row in corpus.iterrows():
            if str(row.get("url", "")).strip():
                continue

            source = str(row.get("source", "")).upper().strip()
            title = self._norm_title(row.get("title", ""))
            if not title or len(title) < 10:
                continue

            candidates = per_source_candidates.get(source)
            if not candidates:
                continue

            attempted += 1
            url, score = self._best_unique_match(title, candidates)
            if url:
                corpus.at[idx, "url"] = url
                resolved += 1
                out_rows.append({"row_index": int(idx), "source": source, "title": row.get("title", ""), "url": url, "score": score})
                continue

            # Optional second stage: discover likely official URLs via GDELT, then title-validate.
            if not use_gdelt:
                continue

            try:
                year_val = pd.to_datetime(row.get("date"), errors="coerce").year
                year = int(year_val) if year_val and year_val > 0 else None
            except Exception:
                year = None

            cache_key = (source, title, str(year or ""))
            if cache_key in gdelt_cache:
                gdelt_urls = gdelt_cache[cache_key]
            else:
                gdelt_urls = self._gdelt_candidates_for_row(row.get("title", ""), year=year, source=source)
                gdelt_cache[cache_key] = gdelt_urls

            if not gdelt_urls:
                continue

            gdelt_urls = gdelt_urls[: int(max(1, max_gdelt_urls_per_row))]
            gdelt_pages = self._fetch_page_titles(gdelt_urls, budget=len(gdelt_urls))
            if not gdelt_pages:
                continue

            url2, score2 = self._best_unique_match(title, gdelt_pages)
            if url2:
                corpus.at[idx, "url"] = url2
                resolved += 1
                out_rows.append(
                    {
                        "row_index": int(idx),
                        "source": source,
                        "title": row.get("title", ""),
                        "url": url2,
                        "score": score2,
                        "method": "gdelt",
                    }
                )

        return {"resolved": resolved, "attempted": attempted, "rows": out_rows, "updated_df": corpus}

    def close(self) -> None:
        if self._browser is not None:
            try:
                self._browser.quit()
            except Exception:
                pass
            self._browser = None
