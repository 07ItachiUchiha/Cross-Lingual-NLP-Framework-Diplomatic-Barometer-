"""Build a larger India–Japan corpus from official pages (via GDELT discovery).

Why this exists
- The current `india_japan_documents.csv` in the repo has ~51 rows.
- URL fill is low because many of those titles are not present verbatim on current official index pages.
- For policy work you typically want *hundreds* of documents spanning years.

This builder:
- Uses GDELT DOC 2.0 (no key) to discover candidate URLs on official domains (MEA/MOFA).
- Fetches each official page (HTTP; optional Selenium fallback) and extracts title + text.
- Writes a new CSV in data/raw with the same required schema as the pre-RAG pipeline loader.

It is intentionally conservative:
- Only keeps URLs on official domains.
- Skips pages with very small extracted text.
"""

from __future__ import annotations

import logging
import random
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

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
class CorpusBuildConfig:
    query: str = '"India" "Japan" ("joint statement" OR summit OR "press release" OR "foreign minister" OR "prime minister")'
    start_year: int = 2000
    end_year: int = 2026
    max_urls_per_year: int = 80
    max_docs_total: int = 600
    min_content_chars: int = 900
    sleep_seconds_min: float = 0.8
    sleep_seconds_max: float = 1.8
    use_browser_fallback: bool = True


class OfficialCorpusBuilder:
    def __init__(self, output_dir: Optional[str] = None, timeout_seconds: int = 25):
        project_root = Path(__file__).resolve().parent.parent
        self.output_dir = Path(output_dir) if output_dir else project_root / "data" / "raw"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.timeout_seconds = int(timeout_seconds)
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
                ),
                "Accept-Language": "en-US,en;q=0.9",
            }
        )

        self._browser = None
        self._browser_available = webdriver is not None

        self.allowed_domains = {
            "mea.gov.in": "MEA",
            "www.mea.gov.in": "MEA",
            "mofa.go.jp": "MOFA",
            "www.mofa.go.jp": "MOFA",
        }

    def close(self) -> None:
        if self._browser is not None:
            try:
                self._browser.quit()
            except Exception:
                pass
            self._browser = None

    @staticmethod
    def _gdelt_range(year: int) -> Tuple[str, str]:
        return (f"{year}0101000000", f"{year}1231235959")

    def _gdelt_urls_for_year(self, query: str, year: int, maxrecords: int) -> List[Dict[str, Any]]:
        url = "https://api.gdeltproject.org/api/v2/doc/doc"
        startdt, enddt = self._gdelt_range(year)
        params: Dict[str, Any] = {
            "query": query,
            "mode": "artlist",
            "format": "json",
            "maxrecords": int(max(1, min(maxrecords, 250))),
            "sort": "datedesc",
            "startdatetime": startdt,
            "enddatetime": enddt,
        }
        try:
            r = self.session.get(url, params=params, timeout=self.timeout_seconds)
            if r.status_code >= 400:
                return []
            payload = r.json()
        except Exception:
            return []

        articles = payload.get("articles") if isinstance(payload, dict) else None
        if not isinstance(articles, list):
            return []
        return articles

    def _host_source(self, url: str) -> Optional[str]:
        try:
            host = urlparse(url).netloc.lower()
        except Exception:
            return None
        return self.allowed_domains.get(host)

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

    def _fetch_html(self, url: str, use_browser_fallback: bool) -> Optional[str]:
        try:
            r = self.session.get(url, timeout=self.timeout_seconds, allow_redirects=True)
            if r.status_code < 400 and r.text:
                return r.text
        except Exception:
            pass

        if not use_browser_fallback:
            return None

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
    def _extract_title_and_text(html: str) -> Tuple[str, str]:
        soup = BeautifulSoup(html, "html.parser")

        title_node = soup.find("h1") or soup.find("title")
        title = title_node.get_text(" ", strip=True) if title_node else ""
        title = re.sub(r"\s+", " ", title).strip()

        for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
            try:
                tag.decompose()
            except Exception:
                pass

        text = soup.get_text(" ", strip=True)
        text = re.sub(r"\s+", " ", text).strip()
        return title, text

    def build(self, cfg: CorpusBuildConfig) -> Dict[str, Any]:
        run_started = datetime.now(timezone.utc).isoformat()

        urls_seen: Set[str] = set()
        rows: List[Dict[str, Any]] = []

        years = list(range(int(cfg.start_year), int(cfg.end_year) + 1))
        for year in years:
            if len(rows) >= int(cfg.max_docs_total):
                break

            articles = self._gdelt_urls_for_year(cfg.query, year=year, maxrecords=int(cfg.max_urls_per_year))
            for art in articles:
                url = str((art or {}).get("url") or "").strip()
                if not url or url in urls_seen:
                    continue

                source = self._host_source(url)
                if not source:
                    continue

                urls_seen.add(url)

                html = self._fetch_html(url, use_browser_fallback=bool(cfg.use_browser_fallback))
                if not html:
                    continue

                title, text = self._extract_title_and_text(html)
                if len(text) < int(cfg.min_content_chars):
                    continue

                # Use GDELT seendate if available; else fall back to year-01-01
                seendate = (art or {}).get("seendate") or (art or {}).get("datetime") or ""
                date_val = pd.to_datetime(seendate, errors="coerce", utc=True)
                if pd.isna(date_val):
                    date_val = pd.Timestamp(year=year, month=1, day=1, tz="UTC")

                rows.append(
                    {
                        "date": date_val.date().isoformat(),
                        "title": title or str((art or {}).get("title") or "").strip() or url,
                        "location": "",
                        "signatories": "",
                        "content": text[:50000],
                        "source": source,
                        "url": url,
                    }
                )

                if len(rows) >= int(cfg.max_docs_total):
                    break

                time.sleep(random.uniform(cfg.sleep_seconds_min, cfg.sleep_seconds_max))

        df = pd.DataFrame(rows)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"]).copy()
            df["year"] = df["date"].dt.year
            df = df.sort_values(["date", "source", "title"]).reset_index(drop=True)

        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_csv = self.output_dir / f"india_japan_official_corpus_{stamp}.csv"
        report_path = self.output_dir / f"official_corpus_build_report_{stamp}.json"

        report = {
            "run_started_utc": run_started,
            "run_finished_utc": datetime.now(timezone.utc).isoformat(),
            "query": cfg.query,
            "start_year": int(cfg.start_year),
            "end_year": int(cfg.end_year),
            "max_urls_per_year": int(cfg.max_urls_per_year),
            "max_docs_total": int(cfg.max_docs_total),
            "min_content_chars": int(cfg.min_content_chars),
            "use_browser_fallback": bool(cfg.use_browser_fallback),
            "total_urls_seen": int(len(urls_seen)),
            "total_docs_kept": int(len(df)),
            "outputs": {"corpus_csv": str(out_csv), "report_json": str(report_path)},
        }

        if not df.empty:
            df.to_csv(out_csv, index=False, encoding="utf-8")

        with open(report_path, "w", encoding="utf-8") as handle:
            import json

            json.dump(report, handle, indent=2)

        return {"df": df, "report": report}
