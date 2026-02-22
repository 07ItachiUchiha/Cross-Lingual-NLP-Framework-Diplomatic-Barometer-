from __future__ import annotations

from typing import Dict

import pandas as pd

from utils.source_credibility import summarize_source_credibility


def _score_sample_size(n_docs: int) -> float:
    return min(25.0, (max(0, n_docs) / 120.0) * 25.0)


def _score_source_diversity(df: pd.DataFrame) -> float:
    if df is None or len(df) == 0 or "source" not in df.columns:
        return 0.0
    unique_sources = int(df["source"].fillna("unknown").astype(str).nunique())
    return min(10.0, (unique_sources / 5.0) * 10.0)


def _score_source_credibility(df: pd.DataFrame) -> float:
    if df is None or len(df) == 0:
        return 0.0
    _, corpus_cred = summarize_source_credibility(df)
    return min(10.0, max(0.0, corpus_cred) * 10.0)


def _score_temporal_coverage(df: pd.DataFrame) -> float:
    if df is None or len(df) == 0 or "year" not in df.columns:
        return 0.0
    unique_years = int(pd.to_numeric(df["year"], errors="coerce").dropna().nunique())
    return min(15.0, (unique_years / 15.0) * 15.0)


def _score_statistical_strength(stats_result: Dict) -> float:
    if not stats_result:
        return 0.0

    p_value = stats_result.get("preferred_p_value")
    effect = abs(float(stats_result.get("effect_size", 0.0)))

    p_score = 0.0
    if p_value is not None:
        p = float(p_value)
        if p < 0.01:
            p_score = 15.0
        elif p < 0.05:
            p_score = 12.0
        elif p < 0.10:
            p_score = 8.0
        else:
            p_score = 3.0

    effect_score = min(10.0, (effect / 0.8) * 10.0)
    return min(25.0, p_score + effect_score)


def _score_external_coverage(external_ctx: Dict) -> float:
    if not external_ctx:
        return 0.0
    keys = ["comtrade_yearly", "estat_rows", "ogd_rows", "rss_year", "external_signals"]
    loaded = sum(1 for key in keys if external_ctx.get(key, {}).get("available"))
    return (loaded / len(keys)) * 15.0


def build_confidence_summary(
    processed_df: pd.DataFrame,
    stats_result: Dict,
    external_ctx: Dict,
) -> Dict:
    n_docs = int(len(processed_df)) if processed_df is not None else 0

    components = {
        "sample_size": _score_sample_size(n_docs),
        "source_diversity": _score_source_diversity(processed_df),
        "source_credibility": _score_source_credibility(processed_df),
        "temporal_coverage": _score_temporal_coverage(processed_df),
        "statistical_strength": _score_statistical_strength(stats_result),
        "external_coverage": _score_external_coverage(external_ctx),
    }

    total_score = round(sum(components.values()), 2)

    if total_score >= 70:
        label = "High"
    elif total_score >= 45:
        label = "Medium"
    else:
        label = "Low"

    return {
        "score": total_score,
        "label": label,
        "components": {k: round(v, 2) for k, v in components.items()},
    }
