"""UN Comtrade structured ingestion (pre-RAG).

Supports:
- Importing already-downloaded Comtrade CSV/JSON exports
- Fetching Comtrade Final Data via the official `comtradeapicall` client (subscription key)

Outputs:
- Normalized rows (csv) and yearly time series (csv)
- A small JSON run report (no secrets)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from utils.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, COMTRADE_PRIMARY_KEY, COMTRADE_SECONDARY_KEY

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ComtradeQuery:
    typeCode: str = "C"  # goods
    freqCode: str = "A"  # annual
    clCode: str = "HS"  # classification family
    period: str = "2024"  # comma-separated list of years or year-months
    reporterCode: str = "699"  # India
    partnerCode: str = "392"  # Japan
    cmdCode: str = "TOTAL"
    flowCode: str = "M"  # M(import), X(export)
    maxRecords: int = 250000
    breakdownMode: str = "classic"
    includeDesc: bool = True


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def load_comtrade_file(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Comtrade file not found: {p}")

    if p.suffix.lower() == ".json":
        with open(p, "r", encoding="utf-8") as handle:
            obj = json.load(handle)
        if isinstance(obj, dict) and "data" in obj:
            obj = obj["data"]
        if not isinstance(obj, list):
            raise ValueError("Unsupported JSON structure: expected a list of rows")
        return pd.DataFrame(obj)

    # Default: CSV
    # Comtrade exports sometimes include a trailing comma, which can cause pandas to
    # misinterpret the first field as an implicit index and shift all columns.
    return pd.read_csv(p, index_col=False)


def _pick_key(prefer: str = "primary") -> str:
    prefer = (prefer or "primary").strip().lower()
    if prefer == "secondary" and COMTRADE_SECONDARY_KEY:
        return COMTRADE_SECONDARY_KEY
    if COMTRADE_PRIMARY_KEY:
        return COMTRADE_PRIMARY_KEY
    if COMTRADE_SECONDARY_KEY:
        return COMTRADE_SECONDARY_KEY
    return ""


def fetch_comtrade_final_data(query: ComtradeQuery, key_preference: str = "primary") -> pd.DataFrame:
    """Fetch Final Data using the official comtradeapicall library."""

    subscription_key = _pick_key(key_preference)
    if not subscription_key:
        raise ValueError(
            "Comtrade subscription key not configured. Set COMTRADE_PRIMARY_KEY or COMTRADE_SECONDARY_KEY in .env"
        )

    try:
        import comtradeapicall  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency comtradeapicall. Install with: pip install -r requirements.txt"
        ) from exc

    df = comtradeapicall.getFinalData(
        subscription_key,
        typeCode=query.typeCode,
        freqCode=query.freqCode,
        clCode=query.clCode,
        period=query.period,
        reporterCode=query.reporterCode,
        cmdCode=query.cmdCode,
        flowCode=query.flowCode,
        partnerCode=query.partnerCode,
        partner2Code=None,
        customsCode=None,
        motCode=None,
        maxRecords=int(query.maxRecords),
        format_output="JSON",
        aggregateBy=None,
        breakdownMode=query.breakdownMode,
        countOnly=None,
        includeDesc=bool(query.includeDesc),
    )

    if df is None:
        return pd.DataFrame()
    if not isinstance(df, pd.DataFrame):
        # comtradeapicall normally returns a DataFrame; keep a fallback.
        return pd.DataFrame(df)
    return df


def normalize_comtrade_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize various Comtrade shapes into a stable subset."""

    if df is None or len(df) == 0:
        return pd.DataFrame(
            columns=[
                "refYear",
                "period",
                "reporterISO",
                "partnerISO",
                "flowCode",
                "cmdCode",
                "primaryValue",
                "isReported",
                "legacyEstimationFlag",
                "classificationCode",
            ]
        )

    out = df.copy()

    # Best-effort rename (some exports use primaryValue, others use cifvalue/fobvalue)
    if "primaryValue" not in out.columns:
        if "cifvalue" in out.columns:
            out["primaryValue"] = out["cifvalue"]
        elif "fobvalue" in out.columns:
            out["primaryValue"] = out["fobvalue"]

    # Keep only the stable subset if present
    keep = [
        "refYear",
        "period",
        "reporterISO",
        "partnerISO",
        "flowCode",
        "cmdCode",
        "primaryValue",
        "isReported",
        "legacyEstimationFlag",
        "classificationCode",
    ]
    # Ensure unique column labels (pandas returns a DataFrame when selecting a duplicated column name)
    keep_unique = list(dict.fromkeys(keep))
    existing = [c for c in keep_unique if c in out.columns]
    out = out[existing].copy()

    if "refYear" in out.columns:
        out["refYear"] = pd.to_numeric(out["refYear"], errors="coerce").astype("Int64")

    if "primaryValue" in out.columns:
        out["primaryValue"] = pd.to_numeric(out["primaryValue"], errors="coerce")

    # Normalize strings
    for c in ["period", "reporterISO", "partnerISO", "flowCode", "cmdCode", "classificationCode"]:
        if c in out.columns:
            out[c] = out[c].astype(str).str.strip()

    if "isReported" in out.columns:
        # in some exports isReported is boolean; keep as bool where possible
        out["isReported"] = out["isReported"].astype("boolean")

    if "legacyEstimationFlag" in out.columns:
        out["legacyEstimationFlag"] = pd.to_numeric(out["legacyEstimationFlag"], errors="coerce").astype("Int64")

    return out


def aggregate_yearly(df_norm: pd.DataFrame) -> pd.DataFrame:
    if df_norm is None or len(df_norm) == 0:
        return pd.DataFrame(columns=["refYear", "flowCode", "cmdCode", "primaryValue_sum", "rows"])

    working = df_norm.copy()
    if "refYear" not in working.columns:
        # Try to infer from period
        if "period" in working.columns:
            working["refYear"] = pd.to_numeric(working["period"].astype(str).str.slice(0, 4), errors="coerce")

    working["refYear"] = pd.to_numeric(working.get("refYear"), errors="coerce").astype("Int64")
    working["primaryValue"] = pd.to_numeric(working.get("primaryValue"), errors="coerce")

    group_cols = [c for c in ["refYear", "flowCode", "cmdCode"] if c in working.columns]
    if not group_cols:
        group_cols = ["refYear"]

    out = (
        working.dropna(subset=["refYear"])  # type: ignore[arg-type]
        .groupby(group_cols, dropna=False)
        .agg(primaryValue_sum=("primaryValue", "sum"), rows=("primaryValue", "size"))
        .reset_index()
        .sort_values(group_cols)
    )
    return out


def write_outputs(df_norm: pd.DataFrame, yearly_df: pd.DataFrame, tag: str) -> dict:
    stamp = _utc_stamp()
    out_rows = PROCESSED_DATA_DIR / f"comtrade_rows_{tag}_{stamp}.csv"
    out_yearly = PROCESSED_DATA_DIR / f"comtrade_yearly_{tag}_{stamp}.csv"
    out_report = RAW_DATA_DIR / f"comtrade_report_{tag}_{stamp}.json"

    df_norm.to_csv(out_rows, index=False, encoding="utf-8")
    yearly_df.to_csv(out_yearly, index=False, encoding="utf-8")

    estimated_share = None
    if "isReported" in df_norm.columns and len(df_norm):
        try:
            estimated_share = float((~df_norm["isReported"].fillna(False)).mean())
        except Exception:
            estimated_share = None

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "tag": tag,
        "rows": int(len(df_norm)),
        "years": sorted([int(x) for x in df_norm["refYear"].dropna().unique().tolist()]) if "refYear" in df_norm.columns else [],
        "estimated_or_unreported_share": estimated_share,
        "outputs": {
            "rows_csv": str(out_rows),
            "yearly_csv": str(out_yearly),
            "report_json": str(out_report),
        },
    }

    with open(out_report, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    return report


def ingest_from_file(path: str | Path, tag: str = "file") -> dict:
    df = load_comtrade_file(path)
    norm = normalize_comtrade_rows(df)
    yearly = aggregate_yearly(norm)
    return write_outputs(norm, yearly, tag=tag)


def ingest_from_api(query: ComtradeQuery, key_preference: str = "primary", tag: str = "api") -> dict:
    df = fetch_comtrade_final_data(query, key_preference=key_preference)
    norm = normalize_comtrade_rows(df)
    yearly = aggregate_yearly(norm)
    return write_outputs(norm, yearly, tag=tag)
