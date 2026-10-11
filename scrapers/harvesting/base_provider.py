"""
Base Provider and Candidate Dataclass for Unified Diplomatic Harvester
"""

from __future__ import annotations

import logging
import random
import sqlite3
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


@dataclass
class HarvestedCandidate:
    """Represents a discovered candidate document before or after content fetch."""
    url: str
    title: str
    date: Optional[str] = None
    year: Optional[int] = None
    source: str = "Unknown"
    source_tier: int = 1  # 1: Official Ministry, 2: Wayback Snapshot, 3: Treaty, 4: Defense/Think-Tank
    genre: str = "Joint Statement"
    language: str = "en"
    archive_url: Optional[str] = None
    raw_html: Optional[str] = None
    content: Optional[str] = None
    cleaned: Optional[str] = None
    location: Optional[str] = None
    signatories: Optional[str] = None
    duplicate_cluster_id: Optional[str] = None
    is_canonical: bool = True
    metadata: Dict = field(default_factory=dict)


class BaseDiscoveryProvider(ABC):
    """Abstract base class for all data discovery and extraction providers."""

    def __init__(
        self,
        cache_db_path: Optional[Path] = None,
        timeout_seconds: int = 25,
        min_jitter: float = 0.5,
        max_jitter: float = 1.5,
    ):
        self.timeout_seconds = timeout_seconds
        self.min_jitter = min_jitter
        self.max_jitter = max_jitter

        if cache_db_path is None:
            project_root = Path(__file__).resolve().parent.parent.parent
            cache_db_path = project_root / "data" / "raw" / "harvesting_cache.sqlite"
        self.cache_db_path = Path(cache_db_path)
        self.cache_db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_cache_db()

        self.session = self._setup_session()

    def _init_cache_db(self) -> None:
        """Initialize SQLite persistent HTTP cache."""
        with sqlite3.connect(self.cache_db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS http_cache (
                    url TEXT PRIMARY KEY,
                    status_code INTEGER,
                    content TEXT,
                    headers TEXT,
                    fetched_at_utc TEXT
                )
                """
            )
            conn.commit()

    def _setup_session(self) -> requests.Session:
        """Setup polite requests session with retry strategy."""
        session = requests.Session()
        session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
                ),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,application/json;q=0.8,*/*;q=0.7",
                "Accept-Language": "en-US,en;q=0.9,fr;q=0.8,ja;q=0.7",
            }
        )
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def fetch_url(self, url: str, use_cache: bool = True) -> Optional[str]:
        """Fetch URL with persistent SQLite cache and polite jitter."""
        if use_cache:
            with sqlite3.connect(self.cache_db_path) as conn:
                cur = conn.cursor()
                cur.execute("SELECT content FROM http_cache WHERE url = ?", (url,))
                row = cur.fetchone()
                if row and row[0]:
                    return row[0]

        # Apply polite rate-limiting jitter
        sleep_time = random.uniform(self.min_jitter, self.max_jitter)
        time.sleep(sleep_time)

        try:
            resp = self.session.get(url, timeout=self.timeout_seconds)
            if resp.status_code == 200:
                text = resp.text
                if use_cache and text:
                    with sqlite3.connect(self.cache_db_path) as conn:
                        conn.execute(
                            """
                            INSERT OR REPLACE INTO http_cache (url, status_code, content, fetched_at_utc)
                            VALUES (?, ?, ?, datetime('now'))
                            """,
                            (url, resp.status_code, text),
                        )
                        conn.commit()
                return text
            else:
                logger.warning(f"HTTP {resp.status_code} for {url}")
                return None
        except Exception as e:
            logger.warning(f"Failed to fetch {url}: {e}")
            return None

    def close(self) -> None:
        """Close session and release resources."""
        if hasattr(self, "session") and self.session:
            try:
                self.session.close()
            except Exception:
                pass

    @abstractmethod
    def discover(self, country_pair: Tuple[str, str]) -> List[HarvestedCandidate]:
        """Discover candidate documents for the given bilateral country pair."""
        pass
