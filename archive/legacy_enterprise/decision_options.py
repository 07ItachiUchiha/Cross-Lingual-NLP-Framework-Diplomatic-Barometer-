from __future__ import annotations

from typing import Dict, List

import pandas as pd


def _extract_top_evidence(scored_df: pd.DataFrame, n: int = 3) -> str:
    if scored_df is None or len(scored_df) == 0:
        return "No evidence rows available"

    working = scored_df.copy()
    if {"security_score", "economic_score"}.issubset(set(working.columns)):
        working["security_score"] = pd.to_numeric(working["security_score"], errors="coerce").fillna(0.0)
        working["economic_score"] = pd.to_numeric(working["economic_score"], errors="coerce").fillna(0.0)
        working["shift_signal"] = working["security_score"] - working["economic_score"]
        top = working.sort_values("shift_signal", ascending=False).head(n)
    else:
        top = working.head(n)

    if "title" in top.columns:
        titles = [str(t)[:80] for t in top["title"].fillna("").tolist() if str(t).strip()]
        return " | ".join(titles) if titles else "No evidence titles available"
    return "No evidence titles available"


def build_decision_options(
    trigger_df: pd.DataFrame,
    confidence_summary: Dict,
    report: Dict,
    weighted_focus: Dict,
    contradiction_df: pd.DataFrame,
    scored_df: pd.DataFrame,
) -> List[Dict[str, str]]:
    options: List[Dict[str, str]] = []

    confidence_label = str(confidence_summary.get("label", "Low")) if confidence_summary else "Low"
    confidence_score = float(confidence_summary.get("score", 0.0)) if confidence_summary else 0.0
    trend = str(report.get("trend", "")).upper() if report else ""
    weighted_gap = float(weighted_focus.get("weighted_gap_security_minus_economic", 0.0)) if weighted_focus else 0.0
    contradiction_count = int(len(contradiction_df)) if isinstance(contradiction_df, pd.DataFrame) else 0

    evidence_anchor = _extract_top_evidence(scored_df, n=3)

    has_high_trigger = False
    has_medium_trigger = False
    if isinstance(trigger_df, pd.DataFrame) and len(trigger_df) > 0 and "severity" in trigger_df.columns:
        severities = set(trigger_df["severity"].astype(str).str.lower().tolist())
        has_high_trigger = "high" in severities
        has_medium_trigger = "medium" in severities

    if has_high_trigger or confidence_label == "Low":
        options.append(
            {
                "option": "Risk-Control Posture",
                "when_to_use": "Use when confidence is low or high-severity policy triggers are active.",
                "recommended_action": "Delay high-stakes commitments; run targeted evidence expansion and contradiction review.",
                "risk_level": "High",
                "evidence_anchor": evidence_anchor,
            }
        )

    if trend == "SECURITY" or weighted_gap > 0.02:
        options.append(
            {
                "option": "Security-First Diplomatic Brief",
                "when_to_use": "Use when security-focused signal is persistent in weighted metrics.",
                "recommended_action": "Prioritize strategic and defense dialogue while preserving economic confidence-building channels.",
                "risk_level": "Medium",
                "evidence_anchor": evidence_anchor,
            }
        )

    if trend == "ECONOMIC" or weighted_gap < -0.02:
        options.append(
            {
                "option": "Economy-First Engagement",
                "when_to_use": "Use when economic signal dominates with acceptable confidence.",
                "recommended_action": "Emphasize trade, supply-chain, and technology cooperation; monitor security drift quarterly.",
                "risk_level": "Low",
                "evidence_anchor": evidence_anchor,
            }
        )

    if contradiction_count > 0:
        options.append(
            {
                "option": "Narrative Reconciliation Track",
                "when_to_use": "Use when cross-source contradiction pairs are detected.",
                "recommended_action": "Create an inter-source clarification note and annotate conflicting statements in policy briefs.",
                "risk_level": "Medium",
                "evidence_anchor": f"Contradiction pairs: {contradiction_count}",
            }
        )

    if not options:
        options.append(
            {
                "option": "Steady-State Monitoring",
                "when_to_use": "Use when no active alerts are detected and confidence is moderate/high.",
                "recommended_action": "Maintain baseline tracking; refresh corpus and update trigger review at monthly cadence.",
                "risk_level": "Info",
                "evidence_anchor": f"Confidence {confidence_score:.1f}/100",
            }
        )

    return options


def decision_options_to_dataframe(options: List[Dict[str, str]]) -> pd.DataFrame:
    if not options:
        return pd.DataFrame(columns=["option", "when_to_use", "recommended_action", "risk_level", "evidence_anchor"])
    return pd.DataFrame(options)
