"""e-Stat (Japan) structured ingestion (pre-RAG).

Uses e-Stat API v3 style endpoints with appId + statsDataId.
Writes raw rows CSV + run report JSON into data/raw.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests

from utils.config import RAW_DATA_DIR, ESTAT_APP_ID


@dataclass(frozen=True)
class EStatQuery:
    stats_data_id: str
    lang: str = "E"  # E: English, J: Japanese
    limit: int = 10000
    start_position: int = 1


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _require_app_id() -> str:
    app_id = (ESTAT_APP_ID or "").strip()
    if not app_id:
        raise ValueError("ESTAT_APP_ID is not set in .env")
    return app_id


def _extract_value_rows(payload: Dict) -> List[Dict]:
    # Typical shape:
    # GET_STATS_DATA -> STATISTICAL_DATA -> DATA_INF -> VALUE
    root = payload.get("GET_STATS_DATA") or payload.get("GET_STATS_DATAS") or payload
    stat_data = root.get("STATISTICAL_DATA") if isinstance(root, dict) else None
    if not isinstance(stat_data, dict):
        return []
    data_inf = stat_data.get("DATA_INF")
    if not isinstance(data_inf, dict):
        return []
    values = data_inf.get("VALUE", [])
    if isinstance(values, dict):
        values = [values]
    if not isinstance(values, list):
        return []

    rows: List[Dict] = []
    for item in values:
        if not isinstance(item, dict):
            continue
        row = {}
        for k, v in item.items():
            # e-Stat often stores value in "$" and dimensions as @time, @cat01, etc.
            row[str(k).lstrip("@")] = v
        rows.append(row)
    return rows


def fetch_estat_data(query: EStatQuery, timeout_s: int = 60) -> Dict:
    app_id = _require_app_id()
    stats_data_id = str(query.stats_data_id).strip()
    if not stats_data_id:
        raise ValueError("stats_data_id is required")

    # e-Stat API v3 endpoint (JSON)
    url = "https://api.e-stat.go.jp/rest/3.0/app/json/getStatsData"

    params = {
        "appId": app_id,
        "statsDataId": stats_data_id,
        "lang": str(query.lang),
        "limit": int(query.limit),
        "startPosition": int(query.start_position),
    }

    resp = requests.get(url, params=params, timeout=timeout_s)
    if resp.status_code != 200:
        raise RuntimeError(f"e-Stat request failed: HTTP {resp.status_code}: {resp.text[:400]}")

    payload = resp.json()
    rows = _extract_value_rows(payload)

    return {
        "stats_data_id": stats_data_id,
        "rows": len(rows),
        "request_url": url,
        "raw": payload,
        "records": rows,
    }


def write_estat_outputs(result: Dict, tag: str = "estat") -> Dict:
    stamp = _utc_stamp()
    out_csv = RAW_DATA_DIR / f"estat_rows_{tag}_{stamp}.csv"
    out_report = RAW_DATA_DIR / f"estat_report_{tag}_{stamp}.json"

    pd.DataFrame(result.get("records") or []).to_csv(out_csv, index=False, encoding="utf-8")

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "stats_data_id": result.get("stats_data_id"),
        "rows": int(result.get("rows", 0)),
        "outputs": {"rows_csv": str(out_csv), "report_json": str(out_report)},
    }

    with open(out_report, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    return report


def fetch_and_write(query: EStatQuery, tag: str = "estat") -> Dict:
    result = fetch_estat_data(query)
    return write_estat_outputs(result, tag=tag)
