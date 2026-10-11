"""Binary Economic-vs-Security validation baseline.

This module is intentionally a validation layer, not a replacement for the
project's phrase-aware lexicon scoring. It answers a narrow question:

    If we train a simple supervised text classifier on labeled examples,
    does it broadly agree with the lexicon-based economic/security signal?

The model is deliberately explainable: TF-IDF features + Logistic Regression.
It excludes Cultural/Other labels during binary training/evaluation so mixed
or non-binary material is not forced into an unsuitable category.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline


BINARY_LABELS = ("Economic", "Security")


@dataclass(frozen=True)
class BinaryClassifierConfig:
    """Configuration for the supervised binary baseline."""

    min_samples_per_class: int = 3
    max_features: int = 5000
    ngram_range: Tuple[int, int] = (1, 2)
    random_state: int = 42


def _normalize_label(value: object) -> str:
    text = str(value or "").strip().lower()
    if text.startswith("econ"):
        return "Economic"
    if text.startswith("sec"):
        return "Security"
    return str(value or "").strip()


def make_binary_training_frame(df: pd.DataFrame, text_col: str = "text", label_col: str = "label") -> pd.DataFrame:
    """Return only valid Economic/Security labeled rows."""

    if not isinstance(df, pd.DataFrame) or text_col not in df.columns or label_col not in df.columns:
        return pd.DataFrame(columns=["text", "label"])

    out = df[[text_col, label_col]].copy()
    out.columns = ["text", "label"]
    out["text"] = out["text"].fillna("").astype(str).str.strip()
    out["label"] = out["label"].apply(_normalize_label)
    out = out[out["text"].ne("") & out["label"].isin(BINARY_LABELS)].copy()
    return out.reset_index(drop=True)


def build_binary_classifier(config: Optional[BinaryClassifierConfig] = None) -> Pipeline:
    """Create the explainable TF-IDF + Logistic Regression baseline."""

    cfg = config or BinaryClassifierConfig()
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    max_features=cfg.max_features,
                    ngram_range=cfg.ngram_range,
                    min_df=1,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    solver="liblinear",
                    class_weight="balanced",
                    random_state=cfg.random_state,
                ),
            ),
        ]
    )


def evaluate_binary_classifier(
    labeled_df: pd.DataFrame,
    text_col: str = "text",
    label_col: str = "label",
    config: Optional[BinaryClassifierConfig] = None,
) -> Dict:
    """Cross-validate the binary classifier on Economic/Security labels only."""

    cfg = config or BinaryClassifierConfig()
    train_df = make_binary_training_frame(labeled_df, text_col=text_col, label_col=label_col)
    label_counts = train_df["label"].value_counts().to_dict() if len(train_df) else {}

    if len(train_df) == 0 or any(label_counts.get(label, 0) < cfg.min_samples_per_class for label in BINARY_LABELS):
        return {
            "available": False,
            "method": "TF-IDF + Logistic Regression",
            "role": "validation_baseline_not_primary_model",
            "binary_labels": list(BINARY_LABELS),
            "samples_used": int(len(train_df)),
            "label_counts": {label: int(label_counts.get(label, 0)) for label in BINARY_LABELS},
            "warning": "Not enough Economic/Security labeled examples for binary cross-validation.",
        }

    min_class = min(int(label_counts.get(label, 0)) for label in BINARY_LABELS)
    n_splits = max(2, min(5, min_class))
    model = build_binary_classifier(cfg)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cfg.random_state)

    texts = train_df["text"].tolist()
    gold = train_df["label"].tolist()
    pred = cross_val_predict(model, texts, gold, cv=cv)

    cm = confusion_matrix(gold, pred, labels=list(BINARY_LABELS))

    return {
        "available": True,
        "method": "TF-IDF + Logistic Regression",
        "role": "validation_baseline_not_primary_model",
        "binary_labels": list(BINARY_LABELS),
        "samples_used": int(len(train_df)),
        "label_counts": {label: int(label_counts.get(label, 0)) for label in BINARY_LABELS},
        "cv_folds": int(n_splits),
        "accuracy": float(round(accuracy_score(gold, pred), 4)),
        "macro_f1": float(round(f1_score(gold, pred, labels=list(BINARY_LABELS), average="macro", zero_division=0), 4)),
        "precision_macro": float(round(precision_score(gold, pred, labels=list(BINARY_LABELS), average="macro", zero_division=0), 4)),
        "recall_macro": float(round(recall_score(gold, pred, labels=list(BINARY_LABELS), average="macro", zero_division=0), 4)),
        "confusion_matrix": pd.DataFrame(cm, index=BINARY_LABELS, columns=BINARY_LABELS).to_dict(),
        "interpretation": (
            "This supervised result is a cross-validated sanity check for the Economic-vs-Security framing. "
            "It should be reported alongside, not instead of, the lexicon trend analysis."
        ),
    }


def train_binary_classifier(
    labeled_df: pd.DataFrame,
    text_col: str = "text",
    label_col: str = "label",
    config: Optional[BinaryClassifierConfig] = None,
) -> Tuple[Optional[Pipeline], Dict]:
    """Fit a binary classifier on all valid Economic/Security labeled rows."""

    cfg = config or BinaryClassifierConfig()
    train_df = make_binary_training_frame(labeled_df, text_col=text_col, label_col=label_col)
    label_counts = train_df["label"].value_counts().to_dict() if len(train_df) else {}

    if len(train_df) == 0 or any(label_counts.get(label, 0) < cfg.min_samples_per_class for label in BINARY_LABELS):
        return None, {
            "available": False,
            "samples_used": int(len(train_df)),
            "label_counts": {label: int(label_counts.get(label, 0)) for label in BINARY_LABELS},
            "warning": "Not enough Economic/Security labeled examples for model fitting.",
        }

    model = build_binary_classifier(cfg)
    model.fit(train_df["text"].tolist(), train_df["label"].tolist())
    return model, {
        "available": True,
        "samples_used": int(len(train_df)),
        "label_counts": {label: int(label_counts.get(label, 0)) for label in BINARY_LABELS},
    }


def predict_binary_documents(model: Pipeline, texts: Sequence[str]) -> List[Dict[str, object]]:
    """Predict Economic/Security labels and probabilities for text rows."""

    clean_texts = [str(t or "") for t in texts]
    labels = model.predict(clean_texts)
    probabilities = model.predict_proba(clean_texts)
    classes = list(model.named_steps["clf"].classes_)

    rows: List[Dict[str, object]] = []
    for label, probs in zip(labels, probabilities):
        prob_by_class = {str(cls): float(prob) for cls, prob in zip(classes, probs)}
        rows.append(
            {
                "binary_ml_label": str(label),
                "binary_ml_confidence": float(max(probs)) if len(probs) else 0.0,
                "binary_ml_prob_economic": float(prob_by_class.get("Economic", 0.0)),
                "binary_ml_prob_security": float(prob_by_class.get("Security", 0.0)),
            }
        )
    return rows


def compare_binary_ml_with_lexicon(
    scored_df: pd.DataFrame,
    labeled_df: Optional[pd.DataFrame] = None,
    text_col: str = "cleaned",
    config: Optional[BinaryClassifierConfig] = None,
) -> Dict:
    """Train on labeled examples, predict project docs, and compare with lexicon signal."""

    if not isinstance(scored_df, pd.DataFrame) or len(scored_df) == 0:
        return {"available": False, "warning": "No scored documents available for binary ML comparison."}

    if labeled_df is None:
        from evaluation.golden_dataset import GoldenDataset

        labeled_df = GoldenDataset().to_dataframe()

    eval_report = evaluate_binary_classifier(labeled_df, config=config)
    model, fit_meta = train_binary_classifier(labeled_df, config=config)

    if model is None:
        out = dict(eval_report)
        out["available"] = False
        out["warning"] = fit_meta.get("warning", eval_report.get("warning", "Binary classifier unavailable."))
        return out

    work = scored_df.copy()
    if text_col not in work.columns:
        fallback = "content" if "content" in work.columns else None
        if fallback is None:
            return {"available": False, "warning": f"Missing text column '{text_col}' and no content fallback exists."}
        text_col = fallback

    predictions = pd.DataFrame(predict_binary_documents(model, work[text_col].fillna("").astype(str).tolist()))
    work = pd.concat([work.reset_index(drop=True), predictions], axis=1)

    econ = pd.to_numeric(work.get("economic_score"), errors="coerce").fillna(0.0)
    sec = pd.to_numeric(work.get("security_score"), errors="coerce").fillna(0.0)
    work["lexicon_binary_label"] = np.where(sec > econ, "Security", "Economic")
    work["binary_ml_agrees_with_lexicon"] = work["binary_ml_label"] == work["lexicon_binary_label"]

    agreement = float(work["binary_ml_agrees_with_lexicon"].mean()) if len(work) else 0.0
    disagreement_df = work[~work["binary_ml_agrees_with_lexicon"]].copy()
    keep_cols = [
        c
        for c in [
            "date",
            "year",
            "title",
            "source",
            "economic_score",
            "security_score",
            "lexicon_binary_label",
            "binary_ml_label",
            "binary_ml_confidence",
            "binary_ml_prob_economic",
            "binary_ml_prob_security",
        ]
        if c in disagreement_df.columns
    ]

    return {
        **eval_report,
        "fit_samples_used": int(fit_meta.get("samples_used", 0)),
        "document_comparison": {
            "documents_compared": int(len(work)),
            "agreement_rate": float(round(agreement, 4)),
            "disagreement_count": int(len(disagreement_df)),
            "ml_label_counts": work["binary_ml_label"].value_counts().to_dict(),
            "lexicon_label_counts": work["lexicon_binary_label"].value_counts().to_dict(),
            "disagreement_examples": disagreement_df[keep_cols].head(10).to_dict("records"),
        },
    }


def evaluate_golden_binary_baseline() -> Dict:
    """Convenience wrapper for CLI/API/pipeline use."""

    from evaluation.golden_dataset import GoldenDataset

    return evaluate_binary_classifier(GoldenDataset().to_dataframe())
