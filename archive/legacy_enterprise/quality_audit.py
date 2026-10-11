from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd


def _quarter_label(dt: pd.Timestamp) -> str:
    return f"{dt.year}-Q{((dt.month - 1) // 3) + 1}"


def _safe_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return df[col]
    return pd.Series([None] * len(df), index=df.index)


def run_quarterly_quality_audit(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    if df is None or len(df) == 0:
        empty_summary = pd.DataFrame(columns=["quarter", "documents", "unique_sources", "missing_url_pct", "missing_location_pct", "avg_content_len"])
        empty_issues = pd.DataFrame(columns=["severity", "issue", "detail", "quarter"])
        return empty_summary, empty_issues, {"audit_score": 0.0, "status": "No data"}

    work = df.copy()
    if "date" in work.columns:
        work["date"] = pd.to_datetime(work["date"], errors="coerce")
    else:
        work["date"] = pd.NaT

    if work["date"].notna().any():
        work = work.dropna(subset=["date"]).copy()
    else:
        empty_summary = pd.DataFrame(columns=["quarter", "documents", "unique_sources", "missing_url_pct", "missing_location_pct", "avg_content_len"])
        empty_issues = pd.DataFrame(columns=["severity", "issue", "detail", "quarter"])
        return empty_summary, empty_issues, {"audit_score": 0.0, "status": "No valid dates"}

    work["quarter"] = work["date"].map(_quarter_label)

    source_series = _safe_series(work, "source").fillna("unknown").astype(str)
    url_series = _safe_series(work, "url").fillna("").astype(str).str.strip()
    loc_series = _safe_series(work, "location").fillna("").astype(str).str.strip()

    if "cleaned" in work.columns:
        text_series = work["cleaned"].fillna("").astype(str)
    elif "content" in work.columns:
        text_series = work["content"].fillna("").astype(str)
    else:
        text_series = pd.Series([""] * len(work), index=work.index)

    work["_source"] = source_series
    work["_url_missing"] = (url_series == "")
    work["_loc_missing"] = (loc_series == "")
    work["_content_len"] = text_series.map(len)

    summary = (
        work.groupby("quarter", as_index=False)
        .agg(
            documents=("quarter", "count"),
            unique_sources=("_source", "nunique"),
            missing_url_pct=("_url_missing", "mean"),
            missing_location_pct=("_loc_missing", "mean"),
            avg_content_len=("_content_len", "mean"),
        )
        .sort_values("quarter")
        .reset_index(drop=True)
    )

    summary["missing_url_pct"] = (summary["missing_url_pct"] * 100).round(1)
    summary["missing_location_pct"] = (summary["missing_location_pct"] * 100).round(1)
    summary["avg_content_len"] = summary["avg_content_len"].round(1)

    issues: List[Dict[str, str]] = []
    for _, row in summary.iterrows():
        q = str(row["quarter"])
        docs = int(row["documents"])
        srcs = int(row["unique_sources"])
        miss_url = float(row["missing_url_pct"])
        miss_loc = float(row["missing_location_pct"])
        avg_len = float(row["avg_content_len"])

        if docs < 2:
            issues.append({"severity": "High", "issue": "Sparse quarter coverage", "detail": f"Only {docs} docs in quarter", "quarter": q})
        elif docs < 4:
            issues.append({"severity": "Medium", "issue": "Low quarter coverage", "detail": f"Only {docs} docs in quarter", "quarter": q})

        if srcs < 2:
            issues.append({"severity": "Low", "issue": "Low source diversity", "detail": f"Only {srcs} unique source(s)", "quarter": q})

        if miss_url > 80:
            issues.append({"severity": "Medium", "issue": "High URL missing rate", "detail": f"{miss_url:.1f}% rows missing URL", "quarter": q})
        if miss_loc > 50:
            issues.append({"severity": "Low", "issue": "High location missing rate", "detail": f"{miss_loc:.1f}% rows missing location", "quarter": q})
        if avg_len < 120:
            issues.append({"severity": "Medium", "issue": "Low average content length", "detail": f"Avg content length={avg_len:.1f}", "quarter": q})

    issue_df = pd.DataFrame(issues) if issues else pd.DataFrame(columns=["severity", "issue", "detail", "quarter"])

    # Lightweight score: start 100, subtract weighted penalties
    # Sparse-quarter issues are still flagged, but with moderated penalty weight.
    penalty = 0.0
    for _, issue in issue_df.iterrows():
        issue_name = str(issue.get("issue", "")).strip()
        if issue_name == "Sparse quarter coverage":
            penalty += 3.0
            continue
        if issue_name == "Low quarter coverage":
            penalty += 1.5
            continue

        sev = str(issue.get("severity", "")).lower()
        if sev == "high":
            penalty += 6.0
        elif sev == "medium":
            penalty += 3.0
        elif sev == "low":
            penalty += 1.5

    quarter_count = int(summary["quarter"].nunique()) if len(summary) > 0 else 1
    normalized_penalty = penalty / max(1, quarter_count)
    audit_score = max(0.0, round(100.0 - (normalized_penalty * 10.0), 1))
    if audit_score >= 80:
        status = "Good"
    elif audit_score >= 60:
        status = "Watch"
    else:
        status = "Needs Attention"

    avg_docs_per_quarter = float(summary["documents"].mean()) if len(summary) > 0 else 0.0
    median_docs_per_quarter = float(summary["documents"].median()) if len(summary) > 0 else 0.0

    if avg_docs_per_quarter >= 6 and median_docs_per_quarter >= 4:
        confidence_band = "High"
    elif avg_docs_per_quarter >= 3 and median_docs_per_quarter >= 2:
        confidence_band = "Medium"
    else:
        confidence_band = "Low"

    meta = {
        "audit_score": audit_score,
        "status": status,
        "quarters_analyzed": int(summary["quarter"].nunique()) if len(summary) > 0 else 0,
        "issues_count": int(len(issue_df)),
        "avg_docs_per_quarter": round(avg_docs_per_quarter, 2),
        "median_docs_per_quarter": round(median_docs_per_quarter, 1),
        "confidence_band": confidence_band,
    }

    return summary, issue_df, meta
