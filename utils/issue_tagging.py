from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd


ISSUE_KEYWORDS: Dict[str, List[str]] = {
    "human_rights": ["human rights", "rights", "freedom", "civil society", "inclusion", "dignity"],
    "climate": ["climate", "net zero", "decarbon", "renewable", "green", "emission", "sustainability"],
    "migration": ["migration", "migrant", "mobility", "visa", "diaspora", "consular"],
    "aid": ["aid", "assistance", "grant", "development cooperation", "capacity building", "relief"],
    "security": ["security", "defense", "defence", "military", "strategic", "deterrence", "maritime"],
    "economy": ["trade", "investment", "economy", "economic", "infrastructure", "supply chain", "market"],
    "technology": ["technology", "digital", "cyber", "ai", "semiconductor", "innovation"],
    "health": ["health", "pandemic", "medical", "public health", "vaccine"],
}

EQUITY_REGION_KEYWORDS: Dict[str, List[str]] = {
    "global_south": ["global south", "developing countries", "least developed", "ldc"],
    "indo_pacific": ["indo-pacific", "maritime", "sea lanes", "quad", "asean"],
    "south_asia": ["south asia", "indian ocean", "bay of bengal", "nepal", "bangladesh", "sri lanka"],
    "east_asia": ["east asia", "japan", "korea", "china", "pacific"],
}

EQUITY_COMMUNITY_KEYWORDS: Dict[str, List[str]] = {
    "women_gender": ["women", "gender", "girls", "gender equality"],
    "children_youth": ["children", "youth", "students", "young people"],
    "workers_livelihoods": ["workers", "jobs", "livelihood", "employment", "labor", "labour"],
    "migrants_diaspora": ["migrant", "diaspora", "expatriate", "visa holders", "consular"],
    "vulnerable_groups": ["vulnerable", "marginalized", "marginalised", "displaced", "refugee"],
}


def _detect_tags(text: str) -> Tuple[List[str], Dict[str, int]]:
    t = str(text).lower()
    matched: List[str] = []
    scores: Dict[str, int] = {}

    for tag, kws in ISSUE_KEYWORDS.items():
        score = sum(t.count(kw) for kw in kws)
        scores[tag] = int(score)
        if score > 0:
            matched.append(tag)

    return matched, scores


def _detect_equity_dimension(text: str, keyword_map: Dict[str, List[str]]) -> Tuple[List[str], Dict[str, int]]:
    t = str(text).lower()
    matched: List[str] = []
    scores: Dict[str, int] = {}
    for key, kws in keyword_map.items():
        score = sum(t.count(kw) for kw in kws)
        scores[key] = int(score)
        if score > 0:
            matched.append(key)
    return matched, scores


def add_issue_tags(df: pd.DataFrame, text_column: str = "cleaned") -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame()

    out = df.copy()
    if text_column not in out.columns:
        if "content" in out.columns:
            text_column = "content"
        elif "title" in out.columns:
            text_column = "title"
        else:
            out["issue_tags"] = [[] for _ in range(len(out))]
            out["primary_issue"] = "other"
            return out

    detected = out[text_column].fillna("").astype(str).map(_detect_tags)
    out["issue_tags"] = detected.map(lambda x: x[0])

    score_maps = detected.map(lambda x: x[1])
    for tag in ISSUE_KEYWORDS.keys():
        out[f"issue_{tag}_score"] = score_maps.map(lambda d: int(d.get(tag, 0)))

    def primary_issue(row: pd.Series) -> str:
        best_tag = "other"
        best_score = 0
        for tag in ISSUE_KEYWORDS.keys():
            sc = int(row.get(f"issue_{tag}_score", 0))
            if sc > best_score:
                best_score = sc
                best_tag = tag
        return best_tag

    out["primary_issue"] = out.apply(primary_issue, axis=1)

    equity_regions_detected = out[text_column].fillna("").astype(str).map(
        lambda t: _detect_equity_dimension(t, EQUITY_REGION_KEYWORDS)
    )
    equity_groups_detected = out[text_column].fillna("").astype(str).map(
        lambda t: _detect_equity_dimension(t, EQUITY_COMMUNITY_KEYWORDS)
    )

    out["equity_regions"] = equity_regions_detected.map(lambda x: x[0])
    out["equity_groups"] = equity_groups_detected.map(lambda x: x[0])

    region_scores = equity_regions_detected.map(lambda x: x[1])
    for region in EQUITY_REGION_KEYWORDS.keys():
        out[f"equity_region_{region}_score"] = region_scores.map(lambda d: int(d.get(region, 0)))

    group_scores = equity_groups_detected.map(lambda x: x[1])
    for group in EQUITY_COMMUNITY_KEYWORDS.keys():
        out[f"equity_group_{group}_score"] = group_scores.map(lambda d: int(d.get(group, 0)))

    return out


def summarize_issue_counts(tagged_df: pd.DataFrame) -> pd.DataFrame:
    if tagged_df is None or len(tagged_df) == 0 or "issue_tags" not in tagged_df.columns:
        return pd.DataFrame(columns=["issue", "documents"])

    counts = {tag: 0 for tag in ISSUE_KEYWORDS.keys()}
    for tags in tagged_df["issue_tags"]:
        if isinstance(tags, list):
            for tag in tags:
                if tag in counts:
                    counts[tag] += 1

    out = pd.DataFrame([
        {"issue": issue, "documents": int(doc_count)}
        for issue, doc_count in counts.items()
    ])
    out = out.sort_values("documents", ascending=False).reset_index(drop=True)
    return out


def summarize_issue_trends(tagged_df: pd.DataFrame) -> pd.DataFrame:
    if tagged_df is None or len(tagged_df) == 0:
        return pd.DataFrame(columns=["year", "issue", "documents"])

    if "year" not in tagged_df.columns:
        if "date" in tagged_df.columns:
            tagged_df = tagged_df.copy()
            tagged_df["date"] = pd.to_datetime(tagged_df["date"], errors="coerce")
            tagged_df["year"] = tagged_df["date"].dt.year
        else:
            return pd.DataFrame(columns=["year", "issue", "documents"])

    rows = []
    for _, row in tagged_df.iterrows():
        year = row.get("year")
        tags = row.get("issue_tags", [])
        if pd.isna(year):
            continue
        if isinstance(tags, list):
            for tag in tags:
                rows.append({"year": int(year), "issue": tag, "documents": 1})

    if not rows:
        return pd.DataFrame(columns=["year", "issue", "documents"])

    out = pd.DataFrame(rows).groupby(["year", "issue"], as_index=False)["documents"].sum()
    out = out.sort_values(["year", "documents"], ascending=[True, False]).reset_index(drop=True)
    return out


def summarize_equity_dimensions(tagged_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    region_cols = ["equity_dimension", "documents"]
    group_cols = ["equity_dimension", "documents"]

    if tagged_df is None or len(tagged_df) == 0:
        return pd.DataFrame(columns=region_cols), pd.DataFrame(columns=group_cols)

    region_counts = {k: 0 for k in EQUITY_REGION_KEYWORDS.keys()}
    group_counts = {k: 0 for k in EQUITY_COMMUNITY_KEYWORDS.keys()}

    if "equity_regions" in tagged_df.columns:
        for tags in tagged_df["equity_regions"]:
            if isinstance(tags, list):
                for tag in tags:
                    if tag in region_counts:
                        region_counts[tag] += 1

    if "equity_groups" in tagged_df.columns:
        for tags in tagged_df["equity_groups"]:
            if isinstance(tags, list):
                for tag in tags:
                    if tag in group_counts:
                        group_counts[tag] += 1

    region_df = pd.DataFrame([
        {"equity_dimension": key, "documents": int(value)}
        for key, value in region_counts.items()
    ]).sort_values("documents", ascending=False).reset_index(drop=True)

    group_df = pd.DataFrame([
        {"equity_dimension": key, "documents": int(value)}
        for key, value in group_counts.items()
    ]).sort_values("documents", ascending=False).reset_index(drop=True)

    return region_df, group_df
