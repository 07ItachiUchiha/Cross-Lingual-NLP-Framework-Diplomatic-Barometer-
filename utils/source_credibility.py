from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd


BASE_SOURCE_CREDIBILITY: Dict[str, float] = {
    "MEA": 0.95,
    "MOFA": 0.95,
    "EMBJPIN": 0.90,
    "MOFA_ARCHIVE": 0.88,
    "EMBJPIN_ARCHIVE": 0.86,
    "JETRO": 0.82,
}


def _normalize_source(value: str) -> str:
    return str(value).strip().upper() if value is not None else "UNKNOWN"


def _url_boost(url: str) -> float:
    if not isinstance(url, str):
        return 0.0
    u = url.lower().strip()
    if not u:
        return 0.0
    if ".go.jp" in u or ".gov.in" in u:
        return 0.03
    if ".org" in u:
        return 0.01
    return 0.0


def add_source_credibility(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame()

    out = df.copy()
    src = out.get("source", pd.Series(["UNKNOWN"] * len(out), index=out.index))
    out["source_normalized"] = src.astype(str).map(_normalize_source)

    out["source_credibility"] = out["source_normalized"].map(BASE_SOURCE_CREDIBILITY).fillna(0.60)

    if "url" in out.columns:
        out["source_credibility"] = (
            out["source_credibility"]
            + out["url"].astype(str).map(_url_boost)
        ).clip(upper=1.0)

    return out


def summarize_source_credibility(df: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
    scored = add_source_credibility(df)
    if scored is None or len(scored) == 0:
        return pd.DataFrame(), 0.0

    summary = (
        scored.groupby("source_normalized", dropna=False)
        .agg(
            documents=("source_normalized", "count"),
            credibility_score=("source_credibility", "mean"),
        )
        .reset_index()
        .rename(columns={"source_normalized": "source"})
        .sort_values(["credibility_score", "documents"], ascending=[False, False])
    )

    corpus_score = float(scored["source_credibility"].mean())
    return summary, corpus_score


def weighted_focus_metrics(scored_df: pd.DataFrame) -> Dict[str, float]:
    if scored_df is None or len(scored_df) == 0:
        return {
            "weighted_economic_mean": 0.0,
            "weighted_security_mean": 0.0,
            "weighted_gap_security_minus_economic": 0.0,
        }

    if not {"economic_score", "security_score"}.issubset(set(scored_df.columns)):
        return {
            "weighted_economic_mean": 0.0,
            "weighted_security_mean": 0.0,
            "weighted_gap_security_minus_economic": 0.0,
        }

    scored = add_source_credibility(scored_df)
    if len(scored) == 0:
        return {
            "weighted_economic_mean": 0.0,
            "weighted_security_mean": 0.0,
            "weighted_gap_security_minus_economic": 0.0,
        }

    weights = pd.to_numeric(scored["source_credibility"], errors="coerce").fillna(0.60).clip(lower=0.1)
    econ = pd.to_numeric(scored["economic_score"], errors="coerce").fillna(0.0)
    sec = pd.to_numeric(scored["security_score"], errors="coerce").fillna(0.0)

    weighted_econ = float((econ * weights).sum() / weights.sum())
    weighted_sec = float((sec * weights).sum() / weights.sum())

    return {
        "weighted_economic_mean": weighted_econ,
        "weighted_security_mean": weighted_sec,
        "weighted_gap_security_minus_economic": weighted_sec - weighted_econ,
    }
