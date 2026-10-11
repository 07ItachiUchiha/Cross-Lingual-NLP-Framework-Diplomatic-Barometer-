"""Summarize RSS items into simple counts (pre-RAG).

Takes an RSS items CSV produced by scrapers/rss_ingestion.py and outputs:
- rss_counts_by_year_<stamp>.csv
- rss_counts_by_month_<stamp>.csv
- rss_summary_report_<stamp>.json

This is useful when you want a lightweight "diplomatic activity" proxy (number of
press releases per time bucket) without scraping.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from utils.config import RAW_DATA_DIR


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def summarize_rss_items(csv_path: str | Path) -> dict:
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"RSS items CSV not found: {p}")

    df = pd.read_csv(p)
    if df is None or len(df) == 0:
        # Still write empty outputs for consistency
        stamp = _utc_stamp()
        out_year = RAW_DATA_DIR / f"rss_counts_by_year_{stamp}.csv"
        out_month = RAW_DATA_DIR / f"rss_counts_by_month_{stamp}.csv"
        out_report = RAW_DATA_DIR / f"rss_summary_report_{stamp}.json"

        pd.DataFrame(columns=["year", "items"]).to_csv(out_year, index=False, encoding="utf-8")
        pd.DataFrame(columns=["year_month", "items"]).to_csv(out_month, index=False, encoding="utf-8")

        report = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "input": str(p),
            "items": 0,
            "outputs": {"by_year_csv": str(out_year), "by_month_csv": str(out_month), "report_json": str(out_report)},
        }
        with open(out_report, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        return report

    # Parse datetime
    published_col = "published_utc" if "published_utc" in df.columns else None
    if not published_col:
        raise ValueError("RSS items CSV missing published_utc column")

    dt = pd.to_datetime(df[published_col], errors="coerce", utc=True)
    df = df.copy()
    df["published_dt"] = dt
    df = df[df["published_dt"].notna()].copy()

    df["year"] = df["published_dt"].dt.year.astype(int)
    df["year_month"] = df["published_dt"].dt.strftime("%Y-%m")

    by_year = df.groupby("year").size().reset_index(name="items").sort_values("year")
    by_month = df.groupby("year_month").size().reset_index(name="items").sort_values("year_month")

    stamp = _utc_stamp()
    out_year = RAW_DATA_DIR / f"rss_counts_by_year_{stamp}.csv"
    out_month = RAW_DATA_DIR / f"rss_counts_by_month_{stamp}.csv"
    out_report = RAW_DATA_DIR / f"rss_summary_report_{stamp}.json"

    by_year.to_csv(out_year, index=False, encoding="utf-8")
    by_month.to_csv(out_month, index=False, encoding="utf-8")

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(p),
        "items": int(len(df)),
        "year_span": [int(by_year["year"].min()), int(by_year["year"].max())] if len(by_year) else None,
        "outputs": {"by_year_csv": str(out_year), "by_month_csv": str(out_month), "report_json": str(out_report)},
    }

    with open(out_report, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    return report
