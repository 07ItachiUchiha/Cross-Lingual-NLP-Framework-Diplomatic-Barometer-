"""RSS ingestion (pre-RAG).

Fetches items from one or more RSS/Atom feeds and writes a flat CSV plus JSON report.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import requests

from utils.config import RAW_DATA_DIR

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RssFetchConfig:
    urls: tuple[str, ...]
    since_days: int = 14
    max_items: int = 200


def _parse_entry_datetime(entry) -> Optional[datetime]:
    for key in ("published_parsed", "updated_parsed"):
        raw = getattr(entry, key, None)
        if raw:
            try:
                return datetime(*raw[:6], tzinfo=timezone.utc)
            except Exception:
                continue

    for key in ("published", "updated", "pubDate"):
        raw = str(getattr(entry, key, "") or "").strip()
        if not raw:
            continue

        try:
            dt = parsedate_to_datetime(raw)
            if dt is not None:
                return dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except Exception:
            pass

        for fmt in ("%a, %m/%d/%Y - %H:%M", "%m/%d/%Y - %H:%M", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d %H:%M:%S"):
            try:
                dt = datetime.strptime(raw, fmt)
                return dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
            except Exception:
                continue

    return None


def _normalize_feed_url(url: str) -> str:
    u = str(url or "").strip()
    if u.startswith("https://data.gov.in/"):
        return u.replace("https://data.gov.in/", "https://www.data.gov.in/", 1)
    if u.startswith("http://data.gov.in/"):
        return u.replace("http://data.gov.in/", "https://www.data.gov.in/", 1)
    return u


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def fetch_rss(cfg: RssFetchConfig) -> dict:
    try:
        import feedparser  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Missing dependency feedparser. Install with: pip install -r requirements.txt") from exc

    cutoff = datetime.now(timezone.utc) - timedelta(days=int(cfg.since_days))
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept": "application/rss+xml, application/atom+xml, application/xml;q=0.9, text/xml;q=0.8, */*;q=0.1",
    }

    rows = []
    feed_diagnostics = []
    for url in cfg.urls:
        normalized_url = _normalize_feed_url(url)
        response_status = None
        response_url = normalized_url
        error = ""

        try:
            response = requests.get(normalized_url, headers=headers, timeout=30, allow_redirects=True)
            response_status = int(response.status_code)
            response_url = str(response.url)
            parsed = feedparser.parse(response.content)
        except Exception as exc:
            error = str(exc)
            logger.warning("RSS fetch failed for %s: %s", normalized_url, exc)
            feed_diagnostics.append(
                {
                    "feed_url": url,
                    "requested_url": normalized_url,
                    "final_url": response_url,
                    "http_status": response_status,
                    "entries_seen": 0,
                    "error": error,
                }
            )
            continue

        entries = list(parsed.entries or [])
        if getattr(parsed, "bozo", 0) and len(entries) == 0:
            bozo_exc = getattr(parsed, "bozo_exception", None)
            if bozo_exc is not None and not error:
                error = str(bozo_exc)

        feed_diagnostics.append(
            {
                "feed_url": url,
                "requested_url": normalized_url,
                "final_url": response_url,
                "http_status": response_status,
                "entries_seen": int(len(entries)),
                "error": error,
            }
        )

        for e in entries[: int(cfg.max_items)]:
            published = _parse_entry_datetime(e)

            if published and published < cutoff:
                continue

            rows.append(
                {
                    "source_url": url,
                    "title": str(getattr(e, "title", "") or "").strip(),
                    "link": str(getattr(e, "link", "") or "").strip(),
                    "published_utc": published.isoformat() if published else None,
                    "summary": str(getattr(e, "summary", "") or "").strip(),
                }
            )

    df = pd.DataFrame(rows)
    stamp = _utc_stamp()
    out_csv = RAW_DATA_DIR / f"rss_items_{stamp}.csv"
    out_json = RAW_DATA_DIR / f"rss_report_{stamp}.json"

    if len(df) > 0:
        # Basic de-dupe
        df = df.drop_duplicates(subset=["link", "title"], keep="first")
        df.to_csv(out_csv, index=False, encoding="utf-8")
    else:
        # still write an empty CSV with columns for consistency
        pd.DataFrame(columns=["source_url", "title", "link", "published_utc", "summary"]).to_csv(
            out_csv, index=False, encoding="utf-8"
        )

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "feeds": list(cfg.urls),
        "since_days": int(cfg.since_days),
        "items": int(len(df)),
        "feed_diagnostics": feed_diagnostics,
        "outputs": {"items_csv": str(out_csv), "report_json": str(out_json)},
    }

    with open(out_json, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    return report
