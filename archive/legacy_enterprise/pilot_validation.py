from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd


def _task_catalog() -> List[Dict[str, str]]:
    return [
        {
            "task_id": "T1",
            "role": "Think Tank",
            "task": "Validate whether strategic trend claim is evidence-supported.",
            "success_criteria": "User can identify confidence label, top evidence rows, and final verdict in <=3 minutes.",
        },
        {
            "task_id": "T2",
            "role": "NGO",
            "task": "Find equity-sensitive issues impacting communities.",
            "success_criteria": "User can apply equity filters and identify top 2 impacted groups/regions.",
        },
        {
            "task_id": "T3",
            "role": "Diplomat",
            "task": "Review contradiction scan and source profile before briefing.",
            "success_criteria": "User can identify whether contradictions exist and export evidence table.",
        },
        {
            "task_id": "T4",
            "role": "Policy",
            "task": "Use trigger rules and decision cards to choose response posture.",
            "success_criteria": "User can state selected option and justify with trigger + evidence anchor.",
        },
    ]


def build_pilot_pack(
    processed_df: pd.DataFrame,
    trigger_df: pd.DataFrame,
    decision_options_df: pd.DataFrame,
    confidence_summary: Dict,
    quality_meta: Dict,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    checklist = pd.DataFrame(_task_catalog())

    log_template = pd.DataFrame(
        [
            {
                "session_id": "",
                "role": "",
                "task_id": "",
                "completed": "",
                "time_minutes": "",
                "confidence_user_rating_1_5": "",
                "clarity_user_rating_1_5": "",
                "decision_quality_rating_1_5": "",
                "notes": "",
            }
        ]
    )

    total_docs = int(len(processed_df)) if processed_df is not None else 0
    trigger_count = int(len(trigger_df)) if isinstance(trigger_df, pd.DataFrame) else 0
    options_count = int(len(decision_options_df)) if isinstance(decision_options_df, pd.DataFrame) else 0
    confidence_score = float(confidence_summary.get("score", 0.0)) if confidence_summary else 0.0
    quality_score = float(quality_meta.get("audit_score", 0.0)) if quality_meta else 0.0

    readiness_score = min(
        100.0,
        round(
            (min(total_docs, 150) / 150.0) * 25.0
            + (confidence_score / 100.0) * 30.0
            + (quality_score / 100.0) * 25.0
            + (min(options_count, 4) / 4.0) * 10.0
            + (1.0 if trigger_count >= 0 else 0.0) * 10.0,
            1,
        ),
    )

    if readiness_score >= 75:
        readiness_label = "Ready"
    elif readiness_score >= 50:
        readiness_label = "Partially Ready"
    else:
        readiness_label = "Needs Prep"

    meta = {
        "readiness_score": readiness_score,
        "readiness_label": readiness_label,
        "total_docs": total_docs,
        "trigger_count": trigger_count,
        "decision_options_count": options_count,
        "confidence_score": confidence_score,
        "quality_score": quality_score,
    }

    return checklist, log_template, meta
