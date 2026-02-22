from __future__ import annotations

from typing import Dict, List

import pandas as pd


TOPIC_KEYWORDS: Dict[str, List[str]] = {
    "security": ["security", "defense", "defence", "military", "strategic", "deterrence", "maritime"],
    "economic": ["trade", "investment", "economy", "economic", "infrastructure", "supply chain", "market"],
    "technology": ["technology", "ai", "semiconductor", "digital", "innovation", "cyber"],
    "humanitarian": ["aid", "humanitarian", "health", "education", "people", "development"],
}

SUPPORT_WORDS = {
    "support", "strengthen", "expand", "enhance", "advance", "commit", "cooperate", "partnership", "progress"
}

CAUTION_WORDS = {
    "concern", "risk", "threat", "tension", "dispute", "oppose", "warn", "challenge", "uncertain"
}


def _dominant_topic(text: str) -> str:
    t = str(text).lower()
    best_topic = "other"
    best_score = 0
    for topic, kws in TOPIC_KEYWORDS.items():
        score = sum(t.count(kw) for kw in kws)
        if score > best_score:
            best_score = score
            best_topic = topic
    return best_topic


def _stance(text: str) -> str:
    t = str(text).lower()
    pos = sum(t.count(w) for w in SUPPORT_WORDS)
    neg = sum(t.count(w) for w in CAUTION_WORDS)
    net = pos - neg
    if net >= 2:
        return "supportive"
    if net <= -2:
        return "cautious"
    return "neutral"


def detect_contradictions(df: pd.DataFrame, text_column: str = "cleaned") -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame()

    work = df.copy()

    if text_column not in work.columns:
        if "content" in work.columns:
            text_column = "content"
        elif "title" in work.columns:
            text_column = "title"
        else:
            return pd.DataFrame()

    if "year" not in work.columns:
        if "date" in work.columns:
            work["date"] = pd.to_datetime(work["date"], errors="coerce")
            work["year"] = work["date"].dt.year
        else:
            return pd.DataFrame()

    work["year"] = pd.to_numeric(work["year"], errors="coerce")
    work = work.dropna(subset=["year"]).copy()
    if len(work) == 0:
        return pd.DataFrame()

    work["source"] = work.get("source", pd.Series(["unknown"] * len(work), index=work.index)).fillna("unknown").astype(str)
    work["_text"] = work[text_column].fillna("").astype(str)
    work["topic"] = work["_text"].map(_dominant_topic)
    work["stance"] = work["_text"].map(_stance)

    rows = []
    for (year, topic), grp in work.groupby(["year", "topic"]):
        if topic == "other" or len(grp) < 2:
            continue

        supportive = grp[grp["stance"] == "supportive"]
        cautious = grp[grp["stance"] == "cautious"]

        if len(supportive) == 0 or len(cautious) == 0:
            continue

        supportive_by_source = supportive.groupby("source", as_index=False).first()
        cautious_by_source = cautious.groupby("source", as_index=False).first()

        for _, s_row in supportive_by_source.iterrows():
            for _, c_row in cautious_by_source.iterrows():
                if str(s_row["source"]).strip().upper() == str(c_row["source"]).strip().upper():
                    continue

                rows.append(
                    {
                        "year": int(year),
                        "topic": topic,
                        "source_supportive": s_row["source"],
                        "source_cautious": c_row["source"],
                        "stance_supportive": "supportive",
                        "stance_cautious": "cautious",
                        "title_supportive": str(s_row.get("title", ""))[:120],
                        "title_cautious": str(c_row.get("title", ""))[:120],
                    }
                )

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    out["contradiction_score"] = 1.0
    out = out.sort_values(["year", "topic", "source_supportive", "source_cautious"]).reset_index(drop=True)
    return out
