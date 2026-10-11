from __future__ import annotations

from typing import Dict, List

import pandas as pd


def evaluate_policy_triggers(
    processed_df: pd.DataFrame,
    report: Dict,
    stats_result: Dict,
    confidence_summary: Dict,
    contradiction_df: pd.DataFrame,
    source_credibility_score: float,
) -> List[Dict[str, str]]:
    triggers: List[Dict[str, str]] = []

    total_docs = int(len(processed_df)) if processed_df is not None else 0
    confidence_score = float(confidence_summary.get("score", 0.0)) if confidence_summary else 0.0
    confidence_label = str(confidence_summary.get("label", "Low")) if confidence_summary else "Low"

    # Trigger 1: Low confidence
    if confidence_score < 45 or confidence_label == "Low":
        triggers.append(
            {
                "trigger": "Low evidence confidence",
                "severity": "High",
                "condition": f"Confidence {confidence_score:.1f}/100 ({confidence_label})",
                "policy_action": "Require analyst review and avoid hard policy claims until additional corroboration is added.",
            }
        )

    # Trigger 2: Security shift without significance
    trend = str(report.get("trend", "")).upper() if report else ""
    significant = bool(stats_result.get("significant")) if stats_result else False
    preferred_p = stats_result.get("preferred_p_value") if stats_result else None
    if trend == "SECURITY" and not significant:
        ptxt = f"p={float(preferred_p):.4g}" if preferred_p is not None else "p unavailable"
        triggers.append(
            {
                "trigger": "Security-leaning trend not statistically strong",
                "severity": "Medium",
                "condition": f"Trend={trend}, significant={significant}, {ptxt}",
                "policy_action": "Treat as watchlist signal; validate with additional sources before escalation.",
            }
        )

    # Trigger 3: Source imbalance
    if processed_df is not None and len(processed_df) > 0 and "source" in processed_df.columns:
        source_counts = processed_df["source"].fillna("unknown").astype(str).value_counts()
        dominant_source = str(source_counts.index[0])
        dominant_ratio = float(source_counts.iloc[0] / len(processed_df))
        if dominant_ratio >= 0.70:
            triggers.append(
                {
                    "trigger": "Source concentration risk",
                    "severity": "Medium",
                    "condition": f"{dominant_source} contributes {dominant_ratio:.0%} of documents",
                    "policy_action": "Increase source diversity before making bilateral policy recommendations.",
                }
            )

    # Trigger 4: Contradiction load
    contradiction_count = int(len(contradiction_df)) if isinstance(contradiction_df, pd.DataFrame) else 0
    if contradiction_count >= 5:
        triggers.append(
            {
                "trigger": "Cross-source contradiction load",
                "severity": "Medium",
                "condition": f"{contradiction_count} contradiction pairs detected",
                "policy_action": "Initiate reconciliation review and annotate conflicting narratives in briefing output.",
            }
        )

    # Trigger 5: Low corpus coverage
    if total_docs < 50:
        triggers.append(
            {
                "trigger": "Low corpus volume",
                "severity": "High",
                "condition": f"Only {total_docs} documents available",
                "policy_action": "Run fresh scraping and official ingestion before strategic interpretation.",
            }
        )

    # Trigger 6: Low source credibility baseline
    if source_credibility_score < 0.75:
        triggers.append(
            {
                "trigger": "Low source credibility baseline",
                "severity": "High",
                "condition": f"Credibility score={source_credibility_score:.3f}",
                "policy_action": "Down-weight weak sources and prioritize official/government-origin evidence.",
            }
        )

    if not triggers:
        triggers.append(
            {
                "trigger": "No active policy alerts",
                "severity": "Info",
                "condition": "All configured thresholds currently pass",
                "policy_action": "Continue routine monitoring and periodic review.",
            }
        )

    return triggers


def triggers_to_dataframe(triggers: List[Dict[str, str]]) -> pd.DataFrame:
    if not triggers:
        return pd.DataFrame(columns=["trigger", "severity", "condition", "policy_action"])
    return pd.DataFrame(triggers)
