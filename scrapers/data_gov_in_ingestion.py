"""data.gov.in (OGD India) structured ingestion (pre-RAG).

This module fetches dataset records from data.gov.in via their API and writes:
- raw records CSV (data/raw)
- run report JSON (data/raw)

It never hardcodes or logs API keys. Configure via env var:
- DATA_GOV_IN_API_KEY

Typical API pattern:
  https://api.data.gov.in/resource/<resource_id>?api-key=...&format=json&limit=...&offset=...

Note: Exact query/filter parameters vary by dataset; this client supports passing
arbitrary query params.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import pandas as pd
import requests

from utils.config import RAW_DATA_DIR, DATA_GOV_IN_API_KEY

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OGDQuery:
    resource_id: str
    format: str = "json"
    limit: int = 100
    max_records: int = 5000
    offset: int = 0
    extra_params: Tuple[Tuple[str, str], ...] = ()


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _require_key() -> str:
    key = (DATA_GOV_IN_API_KEY or "").strip()
    if not key:
        raise ValueError("DATA_GOV_IN_API_KEY is not set (configure it in .env)")
    return key


def fetch_ogd_records(query: OGDQuery, timeout_s: int = 30) -> Dict:
    """Fetch records with pagination up to max_records."""

    api_key = _require_key()
    resource_id = str(query.resource_id).strip()
    if not resource_id:
        raise ValueError("resource_id is required")

    base_url = f"https://api.data.gov.in/resource/{resource_id}"

    collected = []
    total = None

    limit = max(1, int(query.limit))
    offset = max(0, int(query.offset))
    max_records = max(1, int(query.max_records))

    extra = list(query.extra_params or ())

    while True:
        params: Dict[str, object] = {
            "api-key": api_key,
            "format": query.format,
            "limit": limit,
            "offset": offset,
        }
        for k, v in extra:
            if k and v is not None:
                params[str(k)] = str(v)

        resp = requests.get(base_url, params=params, timeout=timeout_s)
        if resp.status_code != 200:
            raise RuntimeError(f"data.gov.in request failed: HTTP {resp.status_code}: {resp.text[:500]}")

        payload = resp.json()
        records = payload.get("records") or payload.get("data") or []
        if not isinstance(records, list):
            raise RuntimeError("Unexpected API response shape: records is not a list")

        if total is None:
            try:
                total = int(payload.get("total", payload.get("total_records", payload.get("count", 0))) or 0)
            except Exception:
                total = None

        collected.extend(records)
        if len(collected) >= max_records:
            collected = collected[:max_records]
            break

        if len(records) < limit:
            break

        offset += limit

    return {
        "resource_id": resource_id,
        "base_url": base_url,
        "rows": int(len(collected)),
        "reported_total": total,
        "records": collected,
    }


def write_ogd_outputs(result: Dict, tag: str = "ogd") -> Dict:
    stamp = _utc_stamp()
    out_csv = RAW_DATA_DIR / f"data_gov_in_{tag}_{stamp}.csv"
    out_report = RAW_DATA_DIR / f"data_gov_in_{tag}_{stamp}.json"

    records = result.get("records") or []
    df = pd.DataFrame(records)
    df.to_csv(out_csv, index=False, encoding="utf-8")

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "resource_id": result.get("resource_id"),
        "base_url": result.get("base_url"),
        "rows": int(result.get("rows", 0)),
        "reported_total": result.get("reported_total"),
        "outputs": {"records_csv": str(out_csv), "report_json": str(out_report)},
    }

    with open(out_report, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    return report


def fetch_and_write(query: OGDQuery, tag: str = "ogd") -> Dict:
    result = fetch_ogd_records(query)
    return write_ogd_outputs(result, tag=tag)
