"""
Main Streamlit Dashboard Application
Cross-Lingual NLP Framework - India-Japan Strategic Shift Analysis
Enhanced with statistical testing, better caching, and more visualizations
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
from typing import Dict, List, Tuple, Optional
import sys
import os
import logging
import tempfile
import re
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from preprocessing.preprocessor import Preprocessor
from analysis.strategic_shift_enhanced import StrategicShiftAnalyzer
from analysis.tone_analyzer import ToneAnalyzer
from analysis.thematic_clustering import ThematicAnalyzer
from evaluation.binary_classifier import compare_binary_ml_with_lexicon
from utils.country_config import (
    COUNTRIES,
    COUNTRY_PAIRS,
    get_country_name,
    get_ministry_name,
    get_country_pair_label,
)
from utils.evidence_confidence import build_confidence_summary
from utils.source_credibility import summarize_source_credibility, weighted_focus_metrics
from utils.contradiction_detection import detect_contradictions
from utils.issue_tagging import (
    add_issue_tags,
    summarize_issue_counts,
    summarize_issue_trends,
    summarize_equity_dimensions,
)
from dashboard.page_renderers import render_overview_page, render_stats_page

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


_ORIG_ST_DATAFRAME = st.dataframe
_ORIG_ST_PLOTLY_CHART = st.plotly_chart


def _normalize_stretch_width(kwargs: Dict[str, object]) -> Dict[str, object]:
    normalized = dict(kwargs)
    if normalized.get("width") == "stretch":
        normalized.pop("width", None)
        normalized.setdefault("use_container_width", True)
    return normalized


def _dataframe_compat(*args, **kwargs):
    return _ORIG_ST_DATAFRAME(*args, **_normalize_stretch_width(kwargs))


def _plotly_chart_compat(*args, **kwargs):
    return _ORIG_ST_PLOTLY_CHART(*args, **_normalize_stretch_width(kwargs))


st.dataframe = _dataframe_compat
st.plotly_chart = _plotly_chart_compat

MIN_DOCS_TONE_THEME = 15
MIN_DOCS_SIGNIFICANCE = 20
MIN_DOCS_CHART_HARD_GUARD = 10


def _file_signature(path: Path) -> str:
    try:
        if not path.exists():
            return "missing"
        stat = path.stat()
        return f"{path.name}:{stat.st_mtime_ns}:{stat.st_size}"
    except Exception:
        return f"{path.name}:unreadable"


def _corpus_cache_token(country_pair: Optional[Tuple[str, str]] = None) -> str:
    project_root = Path(__file__).resolve().parent.parent
    raw_dir = project_root / "data" / "raw"
    pair = country_pair or ("india", "japan")
    pair_slug = f"{pair[0]}_{pair[1]}"
    canonical = raw_dir / f"{pair_slug}_documents_canonical.csv"
    primary = raw_dir / f"{pair_slug}_documents.csv"
    target = canonical if canonical.exists() else primary
    return _file_signature(target)


def _external_cache_token() -> str:
    return "decoupled"
def _build_india_france_corpus_from_dashboard(
    start_year: int = 2000,
    end_year: int = 2026,
    max_docs: int = 500,
    max_urls_per_year: int = 50,
    min_content_chars: int = 500,
) -> Dict[str, object]:
    from scrapers.harvesting import UnifiedHarvester

    harvester = UnifiedHarvester()
    try:
        df = harvester.harvest_corpus(
            ("india", "france"),
            max_docs_limit=int(max_docs),
            min_chars_limit=int(min_content_chars),
        )
        return {"total_docs_kept": len(df)}
    finally:
        harvester.close()


def _build_india_japan_corpus_from_dashboard(
    start_year: int = 2000,
    end_year: int = 2026,
    max_docs: int = 600,
    max_urls_per_year: int = 80,
    min_content_chars: int = 500,
) -> Dict[str, object]:
    from scrapers.harvesting import UnifiedHarvester

    harvester = UnifiedHarvester()
    try:
        df = harvester.harvest_corpus(
            ("india", "japan"),
            max_docs_limit=int(max_docs),
            min_chars_limit=int(min_content_chars),
        )
        return {"total_docs_kept": len(df)}
    finally:
        harvester.close()


def _in_streamlit_runtime() -> bool:
    """Return True when running under `streamlit run`.

    This avoids emitting Streamlit runtime warnings during bare Python imports.
    """

    try:
        # Calling get_script_run_ctx() outside `streamlit run` logs a warning.
        # Suppress that single warning for bare-import sanity checks.
        ctx_logger = logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context")
        prev_level = ctx_logger.level
        ctx_logger.setLevel(logging.ERROR)

        from streamlit.runtime.scriptrunner import get_script_run_ctx

        ctx = get_script_run_ctx()
        if ctx is not None:
            ctx_logger.setLevel(prev_level)
            return True
        return False
    except Exception:
        return False


# Avoid Streamlit cache warnings on bare import by switching cache decorators to no-ops.
if _in_streamlit_runtime():
    cache_data = st.cache_data
else:

    def cache_data(*_args, **_kwargs):  # type: ignore
        def _decorator(fn):
            return fn

        return _decorator


@cache_data(show_spinner="Loading diplomatic documents...")
def load_data_for_pair(pair_str: str, corpus_token: str = ""):
    """Load and cache diplomatic documents for a specific country pair"""
    logger.info(f"Loading documents for {pair_str} (cache token: {corpus_token})...")
    pair = tuple(pair_str.split('-'))
    
    # Import here to avoid circular imports
    from data_integration import DashboardDataManager
    manager = DashboardDataManager()
    df = manager.get_data_for_pair(pair)
    return df


@cache_data(show_spinner="Preprocessing documents...")
def preprocess_data(_df, pair_key: str = "", corpus_token: str = ""):
    """Preprocess documents with caching"""
    logger.info("Preprocessing documents...")
    preprocessor = Preprocessor()
    processed_df = preprocessor.process_dataframe(_df, content_column='content')
    if isinstance(processed_df, pd.DataFrame):
        processed_df = processed_df.loc[:, ~processed_df.columns.duplicated()].copy()
        if 'cleaned' in processed_df.columns:
            processed_df['cleaned'] = processed_df['cleaned'].fillna('').astype(str)
        if 'title' in processed_df.columns:
            processed_df['title'] = processed_df['title'].fillna('Diplomatic Document').astype(str)
        if 'doc_type' in processed_df.columns:
            processed_df['doc_type'] = processed_df['doc_type'].fillna('Unknown').astype(str)
    return processed_df


@cache_data(show_spinner="Analyzing strategic shifts...")
def perform_strategic_analysis(_df):
    """Cached strategic shift analysis"""
    analyzer = StrategicShiftAnalyzer()
    # Backwards-compatible call: older analyzer versions may not accept group_by_source.
    try:
        report, scored_df, yearly_df = analyzer.generate_shift_report(_df, group_by_source=True)
    except TypeError as e:
        msg = str(e)
        if "unexpected keyword argument" in msg and "group_by_source" in msg:
            report, scored_df, yearly_df = analyzer.generate_shift_report(_df)
        else:
            raise
    return report, scored_df, yearly_df


@cache_data(show_spinner="Running binary validation baseline...")
def perform_binary_validation(_scored_df):
    """Cached supervised Economic-vs-Security validation baseline."""
    return compare_binary_ml_with_lexicon(scored_df=_scored_df, text_col='cleaned')


@cache_data(show_spinner="Analyzing tone & sentiment...")
def perform_tone_analysis(_df):
    """Cached tone analysis - Fixed to handle empty dataframes"""
    if _df is None or len(_df) == 0:
        logger.warning("Empty dataframe passed to tone analyzer")
        return None, pd.DataFrame()
    
    if 'cleaned' not in _df.columns:
        logger.warning("'cleaned' column not found in dataframe")
        return None, pd.DataFrame()
    
    tone_analyzer = ToneAnalyzer()
    tone_df = tone_analyzer.process_dataframe(_df, text_column='cleaned')
    return tone_analyzer, tone_df


@cache_data(show_spinner="Analyzing themes...")
def perform_thematic_analysis(_df):
    """Cached thematic analysis"""
    n_docs = int(len(_df)) if isinstance(_df, pd.DataFrame) else 0
    adaptive_topics = max(2, min(5, int(n_docs ** 0.5))) if n_docs > 0 else 2
    thematic_analyzer = ThematicAnalyzer(n_topics=adaptive_topics)
    analysis, df_themes = thematic_analyzer.analyze_theme_evolution(
        _df,
        text_column='cleaned'
    )
    return thematic_analyzer, analysis, df_themes


@cache_data(show_spinner=False)
def perform_issue_tagging(_df, pair_key: str = "", corpus_token: str = ""):
    tagged = add_issue_tags(_df)
    counts = summarize_issue_counts(tagged)
    trends = summarize_issue_trends(tagged)
    region_equity, group_equity = summarize_equity_dimensions(tagged)
    return tagged, counts, trends, region_equity, group_equity


def build_analysis_bundle(_df: pd.DataFrame, external_ctx: Dict, include_tone_theme: bool = False) -> Dict:
    report, scored_df, yearly_df = perform_strategic_analysis(_df)
    stats_result = calculate_statistical_significance(
        scored_df.get('economic_score', pd.Series(dtype=float)),
        scored_df.get('security_score', pd.Series(dtype=float)),
    )
    source_balance = calculate_source_balance_metrics(scored_df)
    uncertainty_notes = build_uncertainty_notes(_df, stats_result, source_balance)
    confidence_summary = build_confidence_summary(_df, stats_result, external_ctx)
    contradiction_df = detect_contradictions(_df)
    source_cred_df, corpus_credibility_score = summarize_source_credibility(_df)
    weighted_focus = weighted_focus_metrics(scored_df)
    binary_validation = perform_binary_validation(scored_df)
    trigger_rows = []
    trigger_df = pd.DataFrame()
    decision_rows = []
    decision_options_df = pd.DataFrame()
    q_summary_df = pd.DataFrame()
    q_issue_df = pd.DataFrame()
    q_meta = {}

    issue_counts_df = summarize_issue_counts(_df)
    issue_trends_df = summarize_issue_trends(_df)
    if (not isinstance(issue_counts_df, pd.DataFrame) or len(issue_counts_df) == 0) and isinstance(_df, pd.DataFrame):
        try:
            _, issue_counts_df, issue_trends_df, _, _ = perform_issue_tagging(_df)
        except Exception:
            issue_counts_df = pd.DataFrame()
            issue_trends_df = pd.DataFrame()

    bundle = {
        "report": report,
        "scored_df": scored_df,
        "yearly_df": yearly_df,
        "stats_result": stats_result,
        "source_balance": source_balance,
        "uncertainty_notes": uncertainty_notes,
        "confidence_summary": confidence_summary,
        "contradiction_df": contradiction_df,
        "source_cred_df": source_cred_df,
        "corpus_credibility_score": corpus_credibility_score,
        "weighted_focus": weighted_focus,
        "binary_validation": binary_validation,
        "trigger_rows": trigger_rows,
        "trigger_df": trigger_df,
        "decision_rows": decision_rows,
        "decision_options_df": decision_options_df,
        "q_summary_df": q_summary_df,
        "q_issue_df": q_issue_df,
        "q_meta": q_meta,
        "issue_counts_df": issue_counts_df,
        "issue_trends_df": issue_trends_df,
    }

    if include_tone_theme:
        tone_analyzer, tone_df = perform_tone_analysis(_df)
        thematic_analyzer, thematic_analysis, thematic_df = perform_thematic_analysis(_df)
        tone_summary = build_tone_summary(tone_analyzer, tone_df)
        theme_summary = build_theme_summary(thematic_analysis)
        bundle.update(
            {
                "tone_analyzer": tone_analyzer,
                "tone_df": tone_df,
                "tone_summary": tone_summary,
                "thematic_analyzer": thematic_analyzer,
                "thematic_analysis": thematic_analysis,
                "thematic_df": thematic_df,
                "theme_summary": theme_summary,
            }
        )
    else:
        bundle.update(
            {
                "tone_summary": {},
                "theme_summary": {},
            }
        )

    return bundle


def _safe_read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _safe_read_json(path: Path) -> Dict:
    try:
        import json

        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
            return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _pick_latest(project_root: Path, patterns: List[Tuple[str, str]]) -> Tuple[str, Optional[Path]]:
    candidates: List[Tuple[str, Path, float]] = []
    for folder, pattern in patterns:
        search_dir = project_root / folder
        if not search_dir.exists():
            continue
        for f in search_dir.glob(pattern):
            try:
                candidates.append((f.name, f, f.stat().st_mtime))
            except Exception:
                continue

    if not candidates:
        return "", None

    latest = max(candidates, key=lambda x: x[2])
    return latest[0], latest[1]


@cache_data(show_spinner=False)
def load_external_integrations(cache_token: str = "") -> Dict:
    return {}

def calculate_statistical_significance(series1: pd.Series, series2: pd.Series) -> Dict:
    """Calculate paired statistical significance between two aligned series."""
    try:
        tmp = pd.DataFrame({
            "economic": pd.to_numeric(series1, errors="coerce"),
            "security": pd.to_numeric(series2, errors="coerce"),
        }).dropna()

        n = int(len(tmp))
        if n < 3:
            return None

        economic = tmp["economic"]
        security = tmp["security"]
        diff = security - economic

        diff_arr = pd.to_numeric(diff, errors="coerce").dropna().to_numpy(dtype=float)

        t_stat, p_value = stats.ttest_rel(economic, security, nan_policy="omit")
        if pd.isna(t_stat):
            t_stat = 0.0
        if pd.isna(p_value):
            p_value = 1.0

        diff_std = float(diff.std(ddof=1)) if n > 1 else 0.0
        diff_mean = float(diff.mean())
        diff_sem = float(stats.sem(diff, nan_policy="omit")) if n > 1 else 0.0
        zero_diff_ratio = float(np.mean(np.isclose(diff_arr, 0.0, atol=1e-12))) if n > 0 else 0.0

        if n > 1 and diff_sem > 0:
            t_crit = float(stats.t.ppf(0.975, df=n - 1))
            ci_low = diff_mean - (t_crit * diff_sem)
            ci_high = diff_mean + (t_crit * diff_sem)
        else:
            ci_low, ci_high = diff_mean, diff_mean

        normality_p = None
        if 3 <= n <= 5000:
            try:
                normality_p = float(stats.shapiro(diff).pvalue)
            except Exception:
                normality_p = None

        wilcoxon_p = None
        try:
            if n >= 5:
                wilcoxon_p = float(stats.wilcoxon(diff, zero_method="wilcox", alternative="two-sided").pvalue)
        except Exception:
            wilcoxon_p = None

        permutation_p = None
        bootstrap_ci95_low = None
        bootstrap_ci95_high = None
        if n >= 6 and len(diff_arr) == n:
            rng = np.random.default_rng(42)
            draws = 4000
            try:
                signs = rng.choice(np.array([-1.0, 1.0], dtype=float), size=(draws, n), replace=True)
                perm_means = (signs * diff_arr).mean(axis=1)
                permutation_p = float(np.mean(np.abs(perm_means) >= abs(diff_mean)))
            except Exception:
                permutation_p = None

            try:
                boot_idx = rng.integers(0, n, size=(draws, n))
                boot_means = diff_arr[boot_idx].mean(axis=1)
                bootstrap_ci95_low = float(np.quantile(boot_means, 0.025))
                bootstrap_ci95_high = float(np.quantile(boot_means, 0.975))
            except Exception:
                bootstrap_ci95_low = None
                bootstrap_ci95_high = None

        preferred_test = "paired_t"
        preferred_p = float(p_value)
        if (normality_p is not None and normality_p < 0.05) or zero_diff_ratio >= 0.30:
            if wilcoxon_p is not None and zero_diff_ratio < 0.50:
                preferred_test = "wilcoxon"
                preferred_p = float(wilcoxon_p)
            elif permutation_p is not None:
                preferred_test = "permutation_sign_flip"
                preferred_p = float(permutation_p)
        elif permutation_p is not None and n < MIN_DOCS_SIGNIFICANCE:
            preferred_test = "permutation_sign_flip"
            preferred_p = float(permutation_p)

        if not np.isfinite(preferred_p):
            preferred_p = 1.0

        if bootstrap_ci95_low is None:
            bootstrap_ci95_low = float(ci_low)
        if bootstrap_ci95_high is None:
            bootstrap_ci95_high = float(ci_high)

        uncertainty_flags: List[str] = []
        if n < MIN_DOCS_SIGNIFICANCE:
            uncertainty_flags.append("small_sample")
        if zero_diff_ratio >= 0.30:
            uncertainty_flags.append("zero_heavy_differences")
        if normality_p is not None and normality_p < 0.05:
            uncertainty_flags.append("non_normal_differences")

        return {
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'paired_n': n,
            'normality_p': normality_p,
            'wilcoxon_p': wilcoxon_p,
            'permutation_p': permutation_p,
            'preferred_test': preferred_test,
            'preferred_p_value': preferred_p,
            'significant': preferred_p < 0.05,
            'effect_size': (diff_mean / diff_std) if diff_std > 0 else 0.0,
            'mean_difference_security_minus_economic': diff_mean,
            'zero_diff_ratio': zero_diff_ratio,
            'ci95_low': float(ci_low),
            'ci95_high': float(ci_high),
            'bootstrap_ci95_low': float(bootstrap_ci95_low),
            'bootstrap_ci95_high': float(bootstrap_ci95_high),
            'uncertainty_flags': uncertainty_flags,
        }
    except Exception as e:
        logger.error(f"Statistical error: {str(e)}")
        return None


def calculate_source_balance_metrics(scored_df: pd.DataFrame) -> Dict[str, object]:
    if scored_df is None or len(scored_df) == 0:
        return {
            "available": False,
            "dominant_source": "n/a",
            "dominant_share": 0.0,
            "source_count": 0,
            "document_weighted_gap": 0.0,
            "source_balanced_gap": 0.0,
            "gap_delta_abs": 0.0,
            "effective_source_count": 0.0,
            "by_source": pd.DataFrame(),
        }

    required_cols = {"economic_score", "security_score", "source"}
    if not required_cols.issubset(set(scored_df.columns)):
        return {
            "available": False,
            "dominant_source": "n/a",
            "dominant_share": 0.0,
            "source_count": 0,
            "document_weighted_gap": 0.0,
            "source_balanced_gap": 0.0,
            "gap_delta_abs": 0.0,
            "effective_source_count": 0.0,
            "by_source": pd.DataFrame(),
        }

    work = scored_df.copy()
    work["source"] = work["source"].fillna("unknown").astype(str)
    work["economic_score"] = pd.to_numeric(work["economic_score"], errors="coerce")
    work["security_score"] = pd.to_numeric(work["security_score"], errors="coerce")
    work = work.dropna(subset=["economic_score", "security_score"])

    if len(work) == 0:
        return {
            "available": False,
            "dominant_source": "n/a",
            "dominant_share": 0.0,
            "source_count": 0,
            "document_weighted_gap": 0.0,
            "source_balanced_gap": 0.0,
            "gap_delta_abs": 0.0,
            "effective_source_count": 0.0,
            "by_source": pd.DataFrame(),
        }

    work["gap"] = work["security_score"] - work["economic_score"]
    source_stats = (
        work.groupby("source", dropna=False)
        .agg(documents=("source", "count"), mean_gap=("gap", "mean"))
        .reset_index()
        .sort_values("documents", ascending=False)
    )

    total_docs = int(source_stats["documents"].sum())
    source_stats["share"] = source_stats["documents"] / max(1, total_docs)
    source_stats["weighted_gap_contrib"] = source_stats["mean_gap"] * source_stats["share"]

    dominant_row = source_stats.iloc[0]
    dominant_source = str(dominant_row["source"])
    dominant_share = float(dominant_row["share"])
    document_weighted_gap = float(source_stats["weighted_gap_contrib"].sum())
    source_balanced_gap = float(source_stats["mean_gap"].mean())

    source_probs = source_stats["share"].to_numpy(dtype=float)
    entropy_val = float(stats.entropy(source_probs)) if len(source_probs) > 1 else 0.0
    effective_source_count = float(np.exp(entropy_val)) if entropy_val > 0 else float(len(source_stats))

    return {
        "available": True,
        "dominant_source": dominant_source,
        "dominant_share": dominant_share,
        "source_count": int(len(source_stats)),
        "document_weighted_gap": document_weighted_gap,
        "source_balanced_gap": source_balanced_gap,
        "gap_delta_abs": float(abs(document_weighted_gap - source_balanced_gap)),
        "effective_source_count": effective_source_count,
        "by_source": source_stats,
    }


def build_uncertainty_notes(
    processed_df: pd.DataFrame,
    stats_result: Optional[Dict],
    source_balance: Optional[Dict],
) -> List[str]:
    notes: List[str] = []

    doc_count = int(len(processed_df)) if isinstance(processed_df, pd.DataFrame) else 0
    year_count = int(processed_df["year"].nunique()) if isinstance(processed_df, pd.DataFrame) and "year" in processed_df.columns else 0

    if doc_count < MIN_DOCS_SIGNIFICANCE:
        notes.append(
            f"Sample is limited ({doc_count} docs), so inferential claims should be directional rather than definitive."
        )
    if year_count < 4:
        notes.append(
            f"Time coverage is narrow ({year_count} years), so trend stability remains provisional."
        )

    if isinstance(source_balance, dict) and source_balance.get("available"):
        dominant_share = float(source_balance.get("dominant_share", 0.0))
        if dominant_share >= 0.70:
            dominant_source = str(source_balance.get("dominant_source", "dominant source"))
            notes.append(
                f"Source composition is concentrated ({dominant_source} at {dominant_share * 100:.1f}%), which can bias corpus-level averages."
            )
        if float(source_balance.get("gap_delta_abs", 0.0)) >= 0.05:
            notes.append(
                "Document-weighted and source-balanced security-vs-economic gaps diverge materially; report both for robustness."
            )

    if isinstance(stats_result, dict):
        zero_ratio = float(stats_result.get("zero_diff_ratio", 0.0))
        if zero_ratio >= 0.30:
            notes.append(
                f"Paired differences are zero-heavy ({zero_ratio * 100:.1f}%), so nonparametric/permutation inference is emphasized."
            )
        preferred_test = str(stats_result.get("preferred_test", "paired_t"))
        if preferred_test != "paired_t":
            notes.append(
                f"Preferred significance result uses `{preferred_test}` to reduce sensitivity to normality and sparsity assumptions."
            )

    return notes


def build_tone_summary(tone_analyzer: Optional[ToneAnalyzer], tone_df: pd.DataFrame) -> Dict[str, object]:
    if tone_analyzer is None or tone_df is None or len(tone_df) == 0:
        return {
            "available": False,
            "dominant_tone": "n/a",
            "dominant_sentiment": "n/a",
            "avg_urgency": None,
            "tone_distribution": {},
            "sentiment_distribution": {},
        }

    tone_dist = tone_analyzer.get_tone_distribution(tone_df) if hasattr(tone_analyzer, "get_tone_distribution") else {}
    sentiment_dist = tone_analyzer.get_sentiment_distribution(tone_df) if hasattr(tone_analyzer, "get_sentiment_distribution") else {}

    dominant_tone = "n/a"
    if isinstance(tone_dist, dict) and len(tone_dist) > 0:
        dominant_tone = str(max(tone_dist.items(), key=lambda kv: kv[1])[0])

    dominant_sentiment = "n/a"
    if isinstance(sentiment_dist, dict) and len(sentiment_dist) > 0:
        dominant_sentiment = str(max(sentiment_dist.items(), key=lambda kv: kv[1])[0])

    urgency_series = pd.to_numeric(tone_df.get("urgency_score", pd.Series(dtype=float)), errors="coerce").dropna()
    avg_urgency = float(urgency_series.mean()) if len(urgency_series) > 0 else None

    return {
        "available": True,
        "dominant_tone": dominant_tone,
        "dominant_sentiment": dominant_sentiment,
        "avg_urgency": avg_urgency,
        "tone_distribution": tone_dist if isinstance(tone_dist, dict) else {},
        "sentiment_distribution": sentiment_dist if isinstance(sentiment_dist, dict) else {},
    }


def build_theme_summary(thematic_analysis: Dict) -> Dict[str, object]:
    if not isinstance(thematic_analysis, dict):
        return {
            "available": False,
            "topic_count": 0,
            "top_topics": [],
            "unique_keyword_count": 0,
        }

    overall = thematic_analysis.get("overall_themes", {})
    if not isinstance(overall, dict) or len(overall) == 0:
        return {
            "available": False,
            "topic_count": 0,
            "top_topics": [],
            "unique_keyword_count": 0,
        }

    top_topics = []
    keyword_pool: List[str] = []
    for topic_id, words in sorted(overall.items()):
        if not isinstance(words, list):
            continue
        clean_words = [str(word).strip() for word in words if str(word).strip()]
        if not clean_words:
            continue
        keyword_pool.extend(clean_words)
        top_topics.append(
            {
                "topic_id": int(topic_id) if str(topic_id).isdigit() else str(topic_id),
                "keywords": clean_words[:6],
            }
        )

    return {
        "available": len(top_topics) > 0,
        "topic_count": int(len(top_topics)),
        "top_topics": top_topics,
        "unique_keyword_count": int(len(set(keyword_pool))),
    }


def build_overview_verdicts(
    processed_df: pd.DataFrame,
    report: Dict,
    scored_df: pd.DataFrame,
    yearly_df: pd.DataFrame,
    stats_result: Dict,
    source_balance: Dict,
    uncertainty_notes: List[str],
    tone_df: pd.DataFrame,
    thematic_analysis: Dict,
    external_ctx: Dict,
    contradiction_df: pd.DataFrame,
) -> List[Dict[str, str]]:
    verdicts: List[Dict[str, str]] = []

    total_docs = int(len(processed_df)) if processed_df is not None else 0
    year_count = int(processed_df["year"].nunique()) if processed_df is not None and "year" in processed_df.columns else 0
    verdicts.append({
        "area": "1) Overview",
        "verdict": f"Corpus has {total_docs} documents across {year_count} years.",
    })

    econ = float(report.get("overall_economic_avg", 0.0))
    sec = float(report.get("overall_security_avg", 0.0))
    if sec > econ:
        strategic_text = f"Security signal leads by {sec - econ:.4f}."
    elif econ > sec:
        strategic_text = f"Economic signal leads by {econ - sec:.4f}."
    else:
        strategic_text = "Economic and security signals are balanced."
    if isinstance(source_balance, dict) and source_balance.get("available"):
        strategic_text = (
            f"{strategic_text} Source-balanced gap (security minus economic) is "
            f"{float(source_balance.get('source_balanced_gap', 0.0)):.4f} across "
            f"{int(source_balance.get('source_count', 0))} sources."
        )
    verdicts.append({"area": "2) Economic vs Security", "verdict": strategic_text})

    if tone_df is not None and len(tone_df) > 0 and "urgency_score" in tone_df.columns:
        urgency = pd.to_numeric(tone_df["urgency_score"], errors="coerce").dropna()
        tone_text = f"Average urgency is {urgency.mean():.3f}." if len(urgency) > 0 else "Tone signal unavailable."
    else:
        tone_text = "Tone signal unavailable."
    verdicts.append({"area": "3) Tone & Mood", "verdict": tone_text})

    top_theme = None
    if isinstance(thematic_analysis, dict):
        overall = thematic_analysis.get("overall_themes")
        if isinstance(overall, dict) and len(overall) > 0:
            first_key = sorted(overall.keys())[0]
            words = overall.get(first_key, [])
            if words:
                top_theme = ", ".join(words[:3])
    verdicts.append({
        "area": "4) Topics & Themes",
        "verdict": f"Top theme keywords: {top_theme}." if top_theme else "Theme signal available but no top keyword summary.",
    })

    time_text = "Time-machine comparison unavailable."
    if scored_df is not None and len(scored_df) > 0 and "year" in scored_df.columns:
        tmp = scored_df.copy()
        tmp["year"] = pd.to_numeric(tmp["year"], errors="coerce")
        tmp = tmp.dropna(subset=["year"])
        if len(tmp) > 0 and {"security_score", "economic_score"}.issubset(set(tmp.columns)):
            years_sorted = sorted([int(y) for y in tmp["year"].unique()])
            if len(years_sorted) >= 2:
                early_year, late_year = years_sorted[0], years_sorted[-1]
                early = tmp[tmp["year"] == early_year]
                late = tmp[tmp["year"] == late_year]
                early_gap = (pd.to_numeric(early["security_score"], errors="coerce").fillna(0.0) - pd.to_numeric(early["economic_score"], errors="coerce").fillna(0.0)).mean()
                late_gap = (pd.to_numeric(late["security_score"], errors="coerce").fillna(0.0) - pd.to_numeric(late["economic_score"], errors="coerce").fillna(0.0)).mean()
                time_text = f"Security-minus-economic gap moved from {early_gap:.4f} ({early_year}) to {late_gap:.4f} ({late_year})."
    verdicts.append({"area": "5) Year Explorer", "verdict": time_text})

    search_hits = 0
    if processed_df is not None and "cleaned" in processed_df.columns:
        search_hits = int(processed_df["cleaned"].astype(str).str.contains("trade|security|defense|investment", case=False, regex=True).sum())
    verdicts.append({
        "area": "6) Keyword Search",
        "verdict": f"{search_hits} documents match core strategic keywords (trade/security/defense/investment).",
    })

    if stats_result:
        stat_text = (
            f"{stats_result.get('preferred_test')} p={stats_result.get('preferred_p_value', float('nan')):.4g}; "
            f"effect={stats_result.get('effect_size', 0.0):.3f}."
        )
    else:
        stat_text = "Statistical test unavailable due to insufficient paired data."
    verdicts.append({"area": "7) Statistical Checks", "verdict": stat_text})
    contradiction_count = int(len(contradiction_df)) if isinstance(contradiction_df, pd.DataFrame) else 0
    verdicts.append({
        "area": "Cross-Source Contradictions",
        "verdict": f"{contradiction_count} potential contradiction pairs detected across sources.",
    })

    if uncertainty_notes:
        verdicts.append(
            {
                "area": "Uncertainty",
                "verdict": " ".join(uncertainty_notes[:3]),
            }
        )

    return verdicts


def build_claim_traceability(scored_df: pd.DataFrame, yearly_df: pd.DataFrame) -> pd.DataFrame:
    if scored_df is None or len(scored_df) == 0:
        return pd.DataFrame()

    working = scored_df.copy()
    if "date" in working.columns:
        working["date"] = pd.to_datetime(working["date"], errors="coerce")
    if "year" not in working.columns and "date" in working.columns:
        working["year"] = working["date"].dt.year

    if "year" not in working.columns:
        return pd.DataFrame()

    working["year"] = pd.to_numeric(working["year"], errors="coerce")
    working = working.dropna(subset=["year"]).copy()
    if len(working) == 0:
        return pd.DataFrame()

    if {"security_score", "economic_score"}.issubset(set(working.columns)):
        working["security_score"] = pd.to_numeric(working["security_score"], errors="coerce").fillna(0.0)
        working["economic_score"] = pd.to_numeric(working["economic_score"], errors="coerce").fillna(0.0)
        working["shift_signal"] = working["security_score"] - working["economic_score"]
    else:
        working["shift_signal"] = 0.0

    if "title" not in working.columns:
        working["title"] = ""

    rows = []
    for year, grp in working.groupby("year"):
        top = grp.sort_values("shift_signal", ascending=False).head(3)
        titles = " | ".join([str(t)[:60] for t in top["title"].fillna("").tolist() if str(t).strip()])
        rows.append({
            "claim": f"Year {int(year)} strategic signal",
            "year": int(year),
            "evidence_docs": int(len(grp)),
            "avg_shift_signal": float(grp["shift_signal"].mean()),
            "top_supporting_titles": titles,
        })

    trace_df = pd.DataFrame(rows).sort_values("year")
    return trace_df


def create_distribution_plot(data: List, title: str, xlabel: str):
    """Create advanced distribution plot with statistics"""
    mean = pd.Series(data).mean()
    std = pd.Series(data).std()
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=data, nbinsx=30, name='Distribution'))
    fig.add_vline(x=mean, line_dash="dash", line_color="red", annotation_text=f"Mean: {mean:.2f}")
    fig.add_vline(x=mean-std, line_dash="dot", line_color="orange", annotation_text=f"±1 SD")
    fig.add_vline(x=mean+std, line_dash="dot", line_color="orange")
    
    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title="Frequency",
        height=400,
        showlegend=True
    )
    return fig


def get_role_presets(role_profile: str) -> Dict:
    presets = {
        "Think Tank": {
            "description": "Research-first mode: evidence depth, trend quality, and methodological caution.",
            "preferred_pages": ["Overview", "Economic vs Security", "Statistical Checks"],
            "focus_issues": ["security", "economy", "technology"],
        },
        "NGO": {
            "description": "Impact-first mode: equity, communities, and issue distribution visibility.",
            "preferred_pages": ["Overview", "Topics & Themes", "Keyword Search"],
            "focus_issues": ["human_rights", "climate", "migration", "aid", "health"],
        },
        "Diplomat": {
            "description": "Negotiation-first mode: bilateral narratives, contradictions, and signal shifts.",
            "preferred_pages": ["Overview", "Economic vs Security", "Year Explorer"],
            "focus_issues": ["security", "economy", "technology", "migration"],
        },
        "Policy": {
            "description": "Decision-first mode: trigger alerts, option cards, and confidence thresholds.",
            "preferred_pages": ["Overview", "Statistical Checks", "Economic vs Security"],
            "focus_issues": ["security", "economy", "aid", "technology"],
        },
    }
    return presets.get(role_profile, presets["Policy"])


def _format_source_mix(df: pd.DataFrame) -> str:
    if not isinstance(df, pd.DataFrame) or len(df) == 0 or "source" not in df.columns:
        return "n/a"
    counts = df["source"].fillna("unknown").astype(str).value_counts()
    total = max(1, int(counts.sum()))
    parts = [f"{src}:{(int(cnt) / total) * 100:.0f}%" for src, cnt in counts.head(4).items()]
    return ", ".join(parts)


def build_run_metadata(df: pd.DataFrame, filter_steps: List[Dict[str, object]], exploratory_mode: bool) -> Dict[str, str]:
    if not isinstance(df, pd.DataFrame) or len(df) == 0:
        return {
            "active_docs": "0",
            "date_span": "n/a",
            "years": "0",
            "source_mix": "n/a",
            "filters": "n/a",
            "mode": "Exploratory" if exploratory_mode else "Decision",
        }

    parsed_dates = pd.to_datetime(df.get("date", pd.Series(dtype=str)), errors="coerce")
    start = parsed_dates.min()
    end = parsed_dates.max()
    date_span = f"{str(start)[:10]} to {str(end)[:10]}" if pd.notna(start) and pd.notna(end) else "n/a"
    filters = [f"{s.get('step')}:{s.get('after')}" for s in filter_steps if s.get("step") not in ["Initial corpus"]]

    return {
        "active_docs": str(int(len(df))),
        "date_span": date_span,
        "years": str(int(df["year"].nunique())) if "year" in df.columns else "n/a",
        "source_mix": _format_source_mix(df),
        "filters": " | ".join(filters) if filters else "No extra filters",
        "mode": "Exploratory" if exploratory_mode else "Decision",
    }


def render_page_scaffold(question: str, run_meta: Dict[str, str], exploratory_mode: bool):
    st.markdown("#### Question")
    st.write(question)
    if exploratory_mode:
        st.warning(
            "Exploratory only: current document/year coverage is below robust decision threshold. Use cautious language and verify evidence manually.",
            icon="⚠️",
        )
    st.caption(
        f"Run metadata — docs: {run_meta.get('active_docs', 'n/a')}, years: {run_meta.get('years', 'n/a')}, "
        f"span: {run_meta.get('date_span', 'n/a')}, sources: {run_meta.get('source_mix', 'n/a')}, mode: {run_meta.get('mode', 'n/a')}"
    )
    with st.expander("Filter snapshot", expanded=False):
        st.write(run_meta.get("filters", "n/a"))


def render_page_exports(page_slug: str, tables: Dict[str, pd.DataFrame], run_meta: Dict[str, str]):
    export_rows = []
    for key, frame in tables.items():
        if isinstance(frame, pd.DataFrame) and len(frame) > 0:
            tmp = frame.copy()
            tmp.insert(0, "section", key)
            export_rows.append(tmp)

    if export_rows:
        merged = pd.concat(export_rows, ignore_index=True, sort=False)
        st.download_button(
            label=f"Download {page_slug} snapshot (CSV)",
            data=merged.to_csv(index=False, encoding="utf-8").encode("utf-8"),
            file_name=f"{page_slug}_snapshot_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )

    st.caption("PDF export in the sidebar uses the same active filters and corpus token for traceable parity.")


def _extract_context_snippet(text: str, term: str, radius: int = 120, exact_phrase: bool = False) -> str:
    source = str(text or "")
    if not source.strip() or not term:
        return source[: radius * 2]

    flags = re.IGNORECASE
    if exact_phrase:
        pattern = re.escape(term)
    else:
        pattern = rf"\b{re.escape(term)}\b"

    match = re.search(pattern, source, flags)
    if not match:
        return source[: radius * 2]

    start = max(0, match.start() - radius)
    end = min(len(source), match.end() + radius)
    snippet = source[start:end]
    snippet = re.sub(pattern, lambda m: f"**{m.group(0)}**", snippet, flags=flags)
    return ("..." if start > 0 else "") + snippet + ("..." if end < len(source) else "")


def render_data_basis_caption(run_meta: Dict[str, str], chart_label: str, point_count: Optional[int] = None):
    suffix = f", points={point_count}" if point_count is not None else ""
    st.caption(
        f"Data basis ({chart_label}) — docs={run_meta.get('active_docs', 'n/a')}, years={run_meta.get('years', 'n/a')}, "
        f"span={run_meta.get('date_span', 'n/a')}, filters={run_meta.get('filters', 'n/a')}{suffix}."
    )


def compute_chart_adequacy(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    min_years: int = 4,
    min_points: int = 10,
    min_variance: float = 1e-4,
) -> Dict[str, object]:
    if not isinstance(df, pd.DataFrame) or len(df) == 0 or x_col not in df.columns or y_col not in df.columns:
        return {
            "years": 0,
            "points": 0,
            "missing_pct": 100.0,
            "variance": 0.0,
            "low_info": True,
            "reasons": ["insufficient data"],
        }

    working = df.copy()
    missing_mask = working[[x_col, y_col]].isna().any(axis=1)
    missing_pct = float(missing_mask.mean() * 100.0)
    working = working.dropna(subset=[x_col, y_col])

    years = int(pd.to_numeric(working[x_col], errors="coerce").dropna().nunique())
    points = int(len(working))
    variance = float(pd.to_numeric(working[y_col], errors="coerce").dropna().var()) if points > 1 else 0.0

    reasons: List[str] = []
    if years < min_years:
        reasons.append(f"only {years} years")
    if points < min_points:
        reasons.append(f"only {points} points")
    if variance < min_variance:
        reasons.append("near-flat signal")

    return {
        "years": years,
        "points": points,
        "missing_pct": round(missing_pct, 1),
        "variance": variance,
        "low_info": len(reasons) > 0,
        "reasons": reasons,
    }


def render_data_adequacy_badges(label: str, adequacy: Dict[str, object]):
    st.caption(f"Data adequacy — {label}")
    b1, b2, b3, b4 = st.columns(4)
    with b1:
        st.metric("Years", int(adequacy.get("years", 0)))
    with b2:
        st.metric("Points", int(adequacy.get("points", 0)))
    with b3:
        st.metric("Missing %", f"{float(adequacy.get('missing_pct', 0.0)):.1f}")
    with b4:
        st.metric("Variance", f"{float(adequacy.get('variance', 0.0)):.4f}")


def build_role_takeaways(role_profile: str, page_key: str, context: Dict[str, object]) -> List[str]:
    docs = int(context.get("docs", 0) or 0)
    years = int(context.get("years", 0) or 0)
    confidence_label = str(context.get("confidence_label", "Unknown"))
    trend = str(context.get("trend", "Unknown"))
    top_issue = str(context.get("top_issue", "n/a"))

    role_defaults = {
        "Think Tank": [
            f"Method confidence is {confidence_label}; keep claims proportional to evidence depth.",
            f"Coverage spans {years} years and {docs} docs; prioritize comparability checks before publication.",
            f"Primary strategic direction currently reads as {trend} in lexicon signals.",
        ],
        "NGO": [
            f"Most visible issue cluster is {top_issue}; verify community-level implications before advocacy framing.",
            f"Current evidence window includes {docs} docs across {years} years; monitor underrepresented groups.",
            f"Confidence is {confidence_label}; use cautious impact language where evidence is sparse.",
        ],
        "Diplomat": [
            f"Trend signal indicates {trend}; triangulate with contradiction and source-balance checks.",
            f"Operational evidence base is {docs} docs over {years} years.",
            f"Confidence is {confidence_label}; anchor negotiation posture to high-confidence evidence rows.",
        ],
        "Policy": [
            f"Current directional signal is {trend}; tie actions to trigger thresholds.",
            f"Evidence base: {docs} docs across {years} years.",
            f"Confidence is {confidence_label}; prefer low-regret options when confidence is medium/low.",
        ],
    }

    takeaways = role_defaults.get(role_profile, role_defaults["Policy"])
    if page_key == "search":
        query = str(context.get("query", "")).strip()
        matches = int(context.get("matches", 0) or 0)
        takeaways = [
            f"Query '{query}' returned {matches} matching records in the active corpus.",
            "Use source filters and exact/lemma modes to validate recall vs precision.",
            "Escalate only patterns that recur across multiple years or sources.",
        ]
    elif page_key == "stats":
        pval = context.get("p_value", None)
        effect = context.get("effect", None)
        takeaways = [
            f"Preferred test p-value: {pval if pval is not None else 'n/a'}.",
            f"Effect size: {effect if effect is not None else 'n/a'}.",
            f"Confidence label: {confidence_label}. Use this before policy escalation.",
        ]
    return takeaways


def render_top_takeaways(role_profile: str, page_key: str, context: Dict[str, object]):
    st.markdown("#### Top 3 takeaways")
    for item in build_role_takeaways(role_profile, page_key, context)[:3]:
        st.markdown(f"- {item}")


def match_search_docs(df: pd.DataFrame, query: str, mode: str) -> pd.Series:
    query_text = str(query or "").strip().lower()
    if not query_text or not isinstance(df, pd.DataFrame) or len(df) == 0:
        return pd.Series([False] * len(df), index=df.index if isinstance(df, pd.DataFrame) else None)

    cleaned = df.get("cleaned", pd.Series(dtype=str)).astype(str)
    if mode == "exact":
        pattern = rf"\b{re.escape(query_text)}\b"
        return cleaned.str.contains(pattern, case=False, regex=True, na=False)

    if mode == "contains":
        pattern = re.escape(query_text)
        return cleaned.str.contains(pattern, case=False, regex=True, na=False)

    # lemma mode
    query_terms = [t for t in re.split(r"\s+", query_text) if t]

    def _has_lemmas(lemmas):
        if not isinstance(lemmas, list) or not lemmas:
            return False
        lemma_set = {str(x).strip().lower() for x in lemmas if str(x).strip()}
        return all(term in lemma_set for term in query_terms)

    return df.get("lemmas", pd.Series([[]] * len(df), index=df.index)).map(_has_lemmas)


def main():
    """Main dashboard application"""

    logo_icon_path = Path(__file__).resolve().parent.parent / "logo.jpeg"
    page_icon = str(logo_icon_path) if logo_icon_path.exists() else ":bar_chart:"

    # Page configuration (only when running under Streamlit)
    st.set_page_config(
        page_title="Cross-Lingual NLP Framework",
        page_icon=page_icon,
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom styling
    st.markdown(
        """
        <style>
        .main {
            padding-top: 2rem;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 1.5rem;
            border-radius: 0.5rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .stat-box {
            background-color: #e8f4f8;
            padding: 1rem;
            border-left: 4px solid #2E86AB;
            border-radius: 0.25rem;
            margin: 0.5rem 0;
            color: #1a1a1a;
            font-size: 0.95rem;
            line-height: 1.6;
        }
        .stat-box b {
            color: #0d47a1;
            font-weight: 700;
        }
        [data-testid="stMetricValue"] {
            font-size: 1.5rem !important;
            font-weight: bold;
        }
        [data-testid="stMetricLabel"] {
            font-size: 0.9rem;
        }
        h1, h2, h3 {
            color: #2E86AB;
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    
    # Language selection in sidebar
    col1, col2 = st.columns([0.85, 0.15])
    with col2:
        language = st.selectbox(
            "Language",
            ["English", "日本語"],
            index=0,
            key="language_selector"
        )
    
    # Translation dictionary
    translations = {
        "English": {
            "title": "Cross-Lingual NLP Framework",
            "subtitle_template": "{country1}-{country2} Relations: Economic and Security Themes",
            "description_template": "Explore how {country1}-{country2} diplomatic language and themes vary over time across the available documents.",
            "documents_loaded": "Analyzing {count} diplomatic documents ({start}-{end})",
            "processing": "Processing documents...",
            "navigation": "Pages",
            "select_analysis": "Choose a Page:",
            "executive_summary": "Overview",
            "strategic_shift": "Economic vs Security",
            "tone_sentiment": "Tone & Mood",
            "thematic": "Topics & Themes",
            "time_machine": "Year Explorer",
            "search_explore": "Keyword Search",
            "statistical_tests": "Statistical Checks",
            "total_documents": "Total Documents",
            "total_words": "Total Words Analyzed",
            "key_findings": "Key Findings",
            "overall_economic": "Overall Economic Focus",
            "overall_security": "Overall Security Focus",
            "crossover_year": "Crossover Year",
            "security_overtook": "In this dataset, security focus overtook economic focus",
            "strategic_shift_evident": "No single crossover year detected in this dataset",
            "overall_trend": "Overall Trend"
        },
        "日本語": {
            "title": "外交晴雨計",
            "subtitle_template": "{country1}・{country2}関係：経済・安全保障テーマ分析",
            "description_template": "利用可能な文書に基づき、{country1}・{country2}の外交言語とテーマが時系列でどのように変化・分布するかを確認します。",
            "documents_loaded": "{count}の外交文書を分析中({start}年～{end}年)",
            "processing": "文書を処理中...",
            "navigation": "ページ",
            "select_analysis": "ページを選択:",
            "executive_summary": "概要",
            "strategic_shift": "経済vs安全保障",
            "tone_sentiment": "トーン・ムード",
            "thematic": "テーマ分析",
            "time_machine": "年別探索",
            "search_explore": "キーワード検索",
            "statistical_tests": "統計チェック",
            "total_documents": "総文書数",
            "total_words": "分析対象総語数",
            "key_findings": "主要な発見",
            "overall_economic": "経済重視度合い",
            "overall_security": "安全保障重視度合い",
            "crossover_year": "クロスオーバー年",
            "security_overtook": "このデータセットでは、安全保障が経済重視を上回った",
            "strategic_shift_evident": "このデータセットでは明確なクロスオーバー年が見つかりませんでした",
            "overall_trend": "全体的トレンド"
        }
    }
    
    # Get current language translations

    # Get current language translations
    lang = translations[language]
    
    # Sidebar navigation
    st.sidebar.title(lang["navigation"])
    
    country_pairs = COUNTRY_PAIRS if COUNTRY_PAIRS else [('india', 'japan')]
    pair_labels = [get_country_pair_label(pair) for pair in country_pairs]
    default_pair = ('india', 'japan') if ('india', 'japan') in country_pairs else country_pairs[0]
    default_index = country_pairs.index(default_pair)

    selected_pair_label = st.sidebar.selectbox(
        "Country Pair",
        pair_labels,
        index=default_index,
        key="country_pair_selector",
    )
    selected_pair = country_pairs[pair_labels.index(selected_pair_label)]
    selected_pair_key = f"{selected_pair[0]}-{selected_pair[1]}"

    if st.session_state.get("_active_pair_key") != selected_pair_key:
        for key in [
            "doc_type_filter",
            "issue_tags_filter",
            "equity_regions_filter",
            "equity_groups_filter",
            "india_france_build_msg",
            "india_japan_build_msg",
        ]:
            st.session_state.pop(key, None)
        st.session_state["_active_pair_key"] = selected_pair_key
        st.cache_data.clear()
        st.rerun()

    country1_name = get_country_name(selected_pair[0])
    country2_name = get_country_name(selected_pair[1])
    country1_ministry = get_ministry_name(selected_pair[0])
    country2_ministry = get_ministry_name(selected_pair[1])
    country1_flag = COUNTRIES.get(selected_pair[0], {}).get('flag', '🏳️')
    country2_flag = COUNTRIES.get(selected_pair[1], {}).get('flag', '🏳️')
    
    st.sidebar.info(f"{country1_flag} {country1_name} ({country1_ministry}) ↔ {country2_flag} {country2_name} ({country2_ministry})")

    role_profile = "Think Tank"
    role_preset = get_role_presets(role_profile)
    st.sidebar.caption("Research Mode: Academic rigor, statistical confidence, and provenance-first analysis.")
    
    st.sidebar.markdown("---")
    
    # Now display title and introduction with dynamic country names
    st.title(lang["title"])
    st.markdown(f"### {lang['subtitle_template'].format(country1=country1_name, country2=country2_name)}")
    st.markdown(lang["description_template"].format(country1=country1_name, country2=country2_name))
    
    # Load data for the selected country pair
    col_refresh_1, col_refresh_2 = st.sidebar.columns([0.55, 0.45])
    with col_refresh_1:
        if st.button("Refresh from disk", use_container_width=True):
            st.rerun()
    with col_refresh_2:
        st.caption("Re-evaluates tokens")

    if selected_pair == ('india', 'france'):
        st.sidebar.markdown("---")
        st.sidebar.subheader("Corpus Builder")
        st.sidebar.caption("One-click rebuild for India-France corpus (official-source workflow).")
        if st.sidebar.button("Build / Refresh India-France corpus", use_container_width=True):
            with st.spinner("Building India-France corpus..."):
                try:
                    build_report = _build_india_france_corpus_from_dashboard()
                    docs_kept = int(build_report.get("total_docs_kept", 0)) if isinstance(build_report, dict) else 0
                    st.session_state["india_france_build_msg"] = f"India-France corpus rebuilt: {docs_kept} docs."
                    st.cache_data.clear()
                    st.rerun()
                except Exception as e:
                    st.sidebar.error(f"Build failed: {str(e)}")

        if st.session_state.get("india_france_build_msg"):
            st.sidebar.success(str(st.session_state.get("india_france_build_msg")))

    if selected_pair == ('india', 'japan'):
        st.sidebar.markdown("---")
        st.sidebar.subheader("Corpus Builder")
        st.sidebar.caption("One-click rebuild for India-Japan corpus (official-source workflow).")
        if st.sidebar.button("Build / Refresh India-Japan corpus", use_container_width=True):
            with st.spinner("Building India-Japan corpus..."):
                try:
                    build_report = _build_india_japan_corpus_from_dashboard()
                    docs_kept = int(build_report.get("total_docs_kept", 0)) if isinstance(build_report, dict) else 0
                    st.session_state["india_japan_build_msg"] = f"India-Japan corpus rebuilt: {docs_kept} docs."
                    st.cache_data.clear()
                    st.rerun()
                except Exception as e:
                    st.sidebar.error(f"Build failed: {str(e)}")

        if st.session_state.get("india_japan_build_msg"):
            st.sidebar.success(str(st.session_state.get("india_japan_build_msg")))

    pair_str = f"{selected_pair[0]}-{selected_pair[1]}"
    corpus_token = _corpus_cache_token(selected_pair)
    df = load_data_for_pair(pair_str, corpus_token=corpus_token)
    doc_count = len(df)
    year_min = df['year'].min()
    year_max = df['year'].max()
    st.success(f"Analyzing **{doc_count} diplomatic documents** from {year_min} to {year_max}")
    
    # Preprocess data
    with st.spinner(lang["processing"]):
        processed_df = preprocess_data(df, pair_key=pair_str, corpus_token=corpus_token)

    tagged_df, issue_counts_df, issue_trends_df, region_equity_df, group_equity_df = perform_issue_tagging(
        processed_df,
        pair_key=pair_str,
        corpus_token=corpus_token,
    )
    if isinstance(tagged_df, pd.DataFrame) and len(tagged_df) > 0:
        processed_df = tagged_df

    baseline_processed_df = processed_df.copy()
    filter_steps: List[Dict[str, object]] = []
    filter_steps.append({"step": "Initial corpus", "before": int(doc_count), "after": int(doc_count), "detail": "Loaded from cache"})

    if st.sidebar.button("Reset all filters", use_container_width=True):
        for key in [
            "doc_type_filter",
            "issue_tags_filter",
            "equity_regions_filter",
            "equity_groups_filter",
        ]:
            st.session_state.pop(key, None)
        st.rerun()

    # Optional stratification: document type (heuristic, from loader)
    if "doc_type" in processed_df.columns:
        processed_df["doc_type"] = processed_df["doc_type"].fillna("Unknown").astype(str)
        doc_type_options = sorted([str(x) for x in processed_df["doc_type"].unique() if str(x).strip()])
        if not doc_type_options:
            doc_type_options = ["Unknown"]
        before_count = int(len(processed_df))
        selected_doc_types = st.sidebar.multiselect(
            "Document type",
            doc_type_options,
            default=doc_type_options,
            key="doc_type_filter",
        )
        if 0 < len(selected_doc_types) < len(doc_type_options):
            processed_df = processed_df[processed_df["doc_type"].isin(selected_doc_types)].copy()
        elif len(selected_doc_types) == 0:
            processed_df = processed_df.iloc[0:0].copy()

        filter_steps.append(
            {
                "step": "Document type",
                "before": before_count,
                "after": int(len(processed_df)),
                "detail": f"Selected {len(selected_doc_types)}/{len(doc_type_options)}",
            }
        )

    if "issue_tags" in processed_df.columns:
        all_issue_tags = sorted({
            tag
            for tags in processed_df["issue_tags"]
            if isinstance(tags, list)
            for tag in tags
        })
        if all_issue_tags:
            default_issue_tags = all_issue_tags
            before_count = int(len(processed_df))
            selected_issue_tags = st.sidebar.multiselect(
                "Issue tags",
                all_issue_tags,
                default=default_issue_tags,
                key="issue_tags_filter",
            )
            apply_issue_filter = 0 < len(selected_issue_tags) < len(all_issue_tags)
            if apply_issue_filter:
                include_unclassified = st.sidebar.checkbox(
                    "Include documents without tags",
                    value=True,
                    key="include_unclassified_issues",
                    help="Keep general bilateral communiqués that don't match specific keyword tags",
                )
                if include_unclassified:
                    processed_df = processed_df[
                        processed_df["issue_tags"].map(
                            lambda tags: (not isinstance(tags, list) or len(tags) == 0) or any(tag in selected_issue_tags for tag in tags)
                        )
                    ].copy()
                else:
                    processed_df = processed_df[
                        processed_df["issue_tags"].map(
                            lambda tags: isinstance(tags, list) and any(tag in selected_issue_tags for tag in tags)
                        )
                    ].copy()
            elif len(selected_issue_tags) == 0:
                processed_df = processed_df.iloc[0:0].copy()

            filter_steps.append(
                {
                    "step": "Issue tags",
                    "before": before_count,
                    "after": int(len(processed_df)),
                    "detail": f"Selected {len(selected_issue_tags)}/{len(all_issue_tags)}",
                }
            )

    if "equity_regions" in processed_df.columns:
        all_regions = sorted({
            tag
            for tags in processed_df["equity_regions"]
            if isinstance(tags, list)
            for tag in tags
        })
        if all_regions:
            before_count = int(len(processed_df))
            selected_regions = st.sidebar.multiselect(
                "Equity regions",
                all_regions,
                default=all_regions,
                key="equity_regions_filter",
            )
            apply_region_filter = 0 < len(selected_regions) < len(all_regions)
            if apply_region_filter:
                include_unclassified_reg = st.sidebar.checkbox(
                    "Include documents without regional tag",
                    value=True,
                    key="include_unclassified_reg",
                    help="Keep bilateral documents that apply to entire nation rather than specific sub-regions",
                )
                if include_unclassified_reg:
                    processed_df = processed_df[
                        processed_df["equity_regions"].map(
                            lambda tags: (not isinstance(tags, list) or len(tags) == 0) or any(tag in selected_regions for tag in tags)
                        )
                    ].copy()
                else:
                    processed_df = processed_df[
                        processed_df["equity_regions"].map(
                            lambda tags: isinstance(tags, list) and any(tag in selected_regions for tag in tags)
                        )
                    ].copy()
            elif len(selected_regions) == 0:
                processed_df = processed_df.iloc[0:0].copy()

            filter_steps.append(
                {
                    "step": "Equity regions",
                    "before": before_count,
                    "after": int(len(processed_df)),
                    "detail": f"Selected {len(selected_regions)}/{len(all_regions)}",
                }
            )

    if "equity_groups" in processed_df.columns:
        all_groups = sorted({
            tag
            for tags in processed_df["equity_groups"]
            if isinstance(tags, list)
            for tag in tags
        })
        if all_groups:
            before_count = int(len(processed_df))
            selected_groups = st.sidebar.multiselect(
                "Equity communities",
                all_groups,
                default=all_groups,
                key="equity_groups_filter",
            )
            apply_group_filter = 0 < len(selected_groups) < len(all_groups)
            if apply_group_filter:
                include_unclassified_grp = st.sidebar.checkbox(
                    "Include documents without community tag",
                    value=True,
                    key="include_unclassified_grp",
                    help="Keep bilateral documents that apply generally rather than to specific community sub-groups",
                )
                if include_unclassified_grp:
                    processed_df = processed_df[
                        processed_df["equity_groups"].map(
                            lambda tags: (not isinstance(tags, list) or len(tags) == 0) or any(tag in selected_groups for tag in tags)
                        )
                    ].copy()
                else:
                    processed_df = processed_df[
                        processed_df["equity_groups"].map(
                            lambda tags: isinstance(tags, list) and any(tag in selected_groups for tag in tags)
                        )
                    ].copy()
            elif len(selected_groups) == 0:
                processed_df = processed_df.iloc[0:0].copy()

            filter_steps.append(
                {
                    "step": "Equity communities",
                    "before": before_count,
                    "after": int(len(processed_df)),
                    "detail": f"Selected {len(selected_groups)}/{len(all_groups)}",
                }
            )

    active_doc_count = int(len(processed_df))
    st.sidebar.metric(
        "Active docs",
        active_doc_count,
        delta=active_doc_count - int(doc_count),
        help="Document count after sidebar filters are applied.",
    )
    if doc_count > 0 and active_doc_count < max(10, int(doc_count * 0.25)):
        st.sidebar.warning(
            "Current filters are very narrow; statistical, tone, and theme outputs may be unstable.",
            icon="⚠️",
        )

    if active_doc_count == 0:
        st.error("No documents match current filters. Broaden filters in the sidebar.")
        st.stop()

    with st.sidebar.expander("Filter impact log", expanded=False):
        st.dataframe(pd.DataFrame(filter_steps), width='stretch', hide_index=True)

    filter_log_payload = [
        {
            "step": s.get("step"),
            "before": s.get("before"),
            "after": s.get("after"),
            "detail": s.get("detail"),
        }
        for s in filter_steps
    ]
    filter_signature = (int(doc_count), int(active_doc_count), tuple((p["step"], p["before"], p["after"], p["detail"]) for p in filter_log_payload))
    if st.session_state.get("_last_filter_signature") != filter_signature:
        logger.info(
            "Filter pipeline reduced corpus from %s to %s docs. Steps=%s",
            doc_count,
            active_doc_count,
            filter_log_payload,
        )
        st.session_state["_last_filter_signature"] = filter_signature

    years_covered = int(processed_df["year"].nunique()) if "year" in processed_df.columns else 0
    exploratory_mode = bool(active_doc_count < MIN_DOCS_SIGNIFICANCE or years_covered < 5)
    run_meta = build_run_metadata(processed_df, filter_steps, exploratory_mode)

    external_token = _external_cache_token()
    external_ctx = load_external_integrations(cache_token=external_token)
    analysis_bundle_cache: Dict[str, Optional[Dict]] = {"core": None, "full": None}

    def get_analysis_bundle(include_tone_theme: bool = False) -> Dict:
        key = "full" if include_tone_theme else "core"
        if analysis_bundle_cache[key] is None:
            analysis_bundle_cache[key] = build_analysis_bundle(
                processed_df,
                external_ctx,
                include_tone_theme=include_tone_theme,
            )
        return analysis_bundle_cache[key] or {}
    
    st.sidebar.markdown("---")
    
    st.sidebar.markdown("---")

    page_options = [
        lang["executive_summary"],
        lang["strategic_shift"],
        lang["thematic"],
        lang["statistical_tests"],
        lang["search_explore"],
        lang["time_machine"],
        lang["tone_sentiment"],
    ]
    preferred_pages = role_preset.get("preferred_pages", ["Overview"])
    default_page_label = preferred_pages[0] if preferred_pages else "Overview"
    if default_page_label == "Overview":
        default_page = lang["executive_summary"]
    elif default_page_label == "Economic vs Security":
        default_page = lang["strategic_shift"]
    elif default_page_label == "Tone & Mood":
        default_page = lang["tone_sentiment"]
    elif default_page_label == "Topics & Themes":
        default_page = lang["thematic"]
    elif default_page_label == "Year Explorer":
        default_page = lang["time_machine"]
    elif default_page_label == "Keyword Search":
        default_page = lang["search_explore"]
    elif default_page_label == "Statistical Checks":
        default_page = lang["statistical_tests"]
    else:
        default_page = lang["executive_summary"]

    default_page_index = page_options.index(default_page) if default_page in page_options else 0
    
    page = st.sidebar.radio(lang["select_analysis"], page_options, index=default_page_index)
    
    # =====================================================================
    # PAGE 1: EXECUTIVE SUMMARY
    # =====================================================================
    if page == lang["executive_summary"]:
        render_overview_page(
            {
                "st": st,
                "lang": lang,
                "country1_name": country1_name,
                "country2_name": country2_name,
                "run_meta": run_meta,
                "exploratory_mode": exploratory_mode,
                "processed_df": processed_df,
                "role_profile": role_profile,
                "year_min": year_min,
                "year_max": year_max,
                "external_ctx": external_ctx,
                "baseline_processed_df": baseline_processed_df,
                "render_page_scaffold": render_page_scaffold,
                "get_analysis_bundle": get_analysis_bundle,
                "render_top_takeaways": render_top_takeaways,
                "summarize_issue_counts": summarize_issue_counts,
                "summarize_issue_trends": summarize_issue_trends,
                "summarize_equity_dimensions": summarize_equity_dimensions,
                "render_data_basis_caption": render_data_basis_caption,
                "build_claim_traceability": build_claim_traceability,
                "build_overview_verdicts": build_overview_verdicts,
                "render_page_exports": render_page_exports,
            }
        )
    
    # =====================================================================
    # PAGE 2: STRATEGIC SHIFT ANALYSIS
    # =====================================================================
    elif page == lang["strategic_shift"]:
        st.header(f"Economic vs Security: Detailed Breakdown")
        render_page_scaffold(
            question=f"How strongly does the corpus support a shift between economic and security focus for {country1_name}-{country2_name}?",
            run_meta=run_meta,
            exploratory_mode=exploratory_mode,
        )
        st.markdown(f"""
        A closer look at how economic and security-related terms vary across {country1_name}-{country2_name} documents
        over time (based on lexicon scoring).
        """)
        
        report, scored_df, yearly_df = perform_strategic_analysis(processed_df)
        render_top_takeaways(
            role_profile=role_profile,
            page_key="economic_vs_security",
            context={
                "docs": len(processed_df),
                "years": processed_df["year"].nunique() if "year" in processed_df.columns else 0,
                "confidence_label": "n/a",
                "trend": report.get("trend", "Unknown"),
                "top_issue": "n/a",
            },
        )

        # Lightweight data-quality context
        st.subheader("Data quality")
        try:
            docs_per_year = processed_df.groupby('year').size().to_dict() if 'year' in processed_df.columns else {}
            low_volume_years = sorted([int(y) for y, c in docs_per_year.items() if int(c) < 3])
            if low_volume_years:
                st.warning(f"Low-volume years (<3 docs): {low_volume_years[:20]}" + ("…" if len(low_volume_years) > 20 else ""))
            if 'source' in processed_df.columns:
                src_counts = processed_df['source'].value_counts().to_dict()
                total = max(1, int(len(processed_df)))
                dominant_source, dominant_count = max(src_counts.items(), key=lambda kv: kv[1])
                if (dominant_count / total) >= 0.80:
                    st.warning(f"Source imbalance: {dominant_source} is {dominant_count}/{total} documents.")
        except Exception:
            pass
        
        # Display yearly statistics
        st.subheader("Yearly Statistics")
        format_dict = {
            col: '{:.4f}'
            for col in ['economic_score_mean', 'security_score_mean', 'economic_score_std', 'security_score_std']
            if col in yearly_df.columns
        }
        st.dataframe(yearly_df.style.format(format_dict, na_rep='-'), width='stretch')

        st.markdown("#### Signal")
        fig_yearly = go.Figure()
        fig_yearly.add_trace(go.Scatter(
            x=yearly_df['year'],
            y=yearly_df['economic_score_mean'],
            mode='lines+markers+text',
            name='Economic',
            text=yearly_df.get('economic_score_count', pd.Series([None] * len(yearly_df))),
            textposition='top center',
            line=dict(color='#2E86AB', width=3),
        ))
        if 'economic_score_std' in yearly_df.columns:
            fig_yearly.add_trace(go.Scatter(x=yearly_df['year'], y=yearly_df['economic_score_mean'] + yearly_df['economic_score_std'], mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
            fig_yearly.add_trace(go.Scatter(x=yearly_df['year'], y=yearly_df['economic_score_mean'] - yearly_df['economic_score_std'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(46,134,171,0.12)', showlegend=False, hoverinfo='skip'))

        fig_yearly.add_trace(go.Scatter(
            x=yearly_df['year'],
            y=yearly_df['security_score_mean'],
            mode='lines+markers+text',
            name='Security',
            text=yearly_df.get('security_score_count', pd.Series([None] * len(yearly_df))),
            textposition='bottom center',
            line=dict(color='#A23B72', width=3),
        ))
        if 'security_score_std' in yearly_df.columns:
            fig_yearly.add_trace(go.Scatter(x=yearly_df['year'], y=yearly_df['security_score_mean'] + yearly_df['security_score_std'], mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
            fig_yearly.add_trace(go.Scatter(x=yearly_df['year'], y=yearly_df['security_score_mean'] - yearly_df['security_score_std'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(162,59,114,0.12)', showlegend=False, hoverinfo='skip'))

        fig_yearly.update_layout(
            title='Yearly focus signals with ±1 SD uncertainty bands (labels = per-year sample counts)',
            xaxis_title='Year',
            yaxis_title='Score',
            hovermode='x unified',
            height=520,
            template='plotly_white',
        )
        st.plotly_chart(fig_yearly, width='stretch')
        render_data_basis_caption(run_meta, "Yearly focus with uncertainty", point_count=len(yearly_df))

        st.caption("These scores are lexicon-based signals; validate interpretation using the evidence rows below.")

        st.subheader("Binary ML validation baseline")
        binary_validation = perform_binary_validation(scored_df)
        if binary_validation.get("available"):
            doc_cmp = binary_validation.get("document_comparison", {})
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("CV macro F1", binary_validation.get("macro_f1", "n/a"))
            with c2:
                st.metric("CV accuracy", binary_validation.get("accuracy", "n/a"))
            with c3:
                st.metric("Lexicon agreement", doc_cmp.get("agreement_rate", "n/a"))
            with c4:
                st.metric("Disagreements", doc_cmp.get("disagreement_count", "n/a"))

            st.caption(
                "Validation only: this TF-IDF + logistic regression model checks whether a simple supervised "
                "Economic-vs-Security classifier broadly agrees with the lexicon signal. It is not the primary model."
            )

            examples = doc_cmp.get("disagreement_examples", [])
            if examples:
                with st.expander("Review ML-vs-lexicon disagreement examples"):
                    st.dataframe(pd.DataFrame(examples), width='stretch', hide_index=True)
        else:
            st.info(f"Binary validation unavailable: {binary_validation.get('warning', 'not enough labeled data')}")
        
        # Distribution comparison
        col1, col2 = st.columns(2)
        
        with col1:
            fig = create_distribution_plot(
                scored_df['economic_score'].tolist(),
                "Economic Score Distribution",
                "Score"
            )
            st.plotly_chart(fig, width='stretch')
            render_data_basis_caption(run_meta, "Economic distribution", point_count=len(scored_df))
        
        with col2:
            fig = create_distribution_plot(
                scored_df['security_score'].tolist(),
                "Security Score Distribution",
                "Score"
            )
            st.plotly_chart(fig, width='stretch')
            render_data_basis_caption(run_meta, "Security distribution", point_count=len(scored_df))
        
        # Top economic terms
        st.subheader("Top Economic Sector Terms")
        eco_cols = [c for c in ['date', 'title', 'source', 'url', 'economic_score'] if c in scored_df.columns]
        eco_terms = scored_df.nlargest(15, 'economic_score')[eco_cols]
        source_counts = processed_df['source'].value_counts(dropna=False).to_dict() if 'source' in processed_df.columns else {}
        total_docs_for_source = max(1, int(len(processed_df)))

        if isinstance(eco_terms, pd.DataFrame) and len(eco_terms) > 0 and 'source' in eco_terms.columns:
            eco_terms = eco_terms.copy()
            eco_terms['source_share_pct'] = eco_terms['source'].map(lambda s: round((source_counts.get(s, 0) / total_docs_for_source) * 100, 1))
            eco_terms['source_balance_warning'] = eco_terms['source_share_pct'].map(lambda p: '⚠️ High source concentration' if float(p) >= 80 else 'OK')
        
        if len(eco_terms) > 0:
            fig = px.bar(
                eco_terms.reset_index(drop=True),
                y=eco_terms.index,
                x='economic_score',
                orientation='h',
                title='Top 15 Documents by Economic Focus Score',
                labels={'economic_score': 'Score', 'index': 'Rank'}
            )
            st.plotly_chart(fig, width='stretch')
            render_data_basis_caption(run_meta, "Top economic documents", point_count=len(eco_terms))
            eco_display_cols = [c for c in ['date', 'title', 'source', 'source_share_pct', 'source_balance_warning', 'economic_score'] if c in eco_terms.columns]
            st.dataframe(eco_terms[eco_display_cols], width='stretch', hide_index=True)
        
        # Top security terms
        st.subheader("Top Security Sector Terms")
        sec_cols = [c for c in ['date', 'title', 'source', 'url', 'security_score'] if c in scored_df.columns]
        sec_terms = scored_df.nlargest(15, 'security_score')[sec_cols]

        if isinstance(sec_terms, pd.DataFrame) and len(sec_terms) > 0 and 'source' in sec_terms.columns:
            sec_terms = sec_terms.copy()
            sec_terms['source_share_pct'] = sec_terms['source'].map(lambda s: round((source_counts.get(s, 0) / total_docs_for_source) * 100, 1))
            sec_terms['source_balance_warning'] = sec_terms['source_share_pct'].map(lambda p: '⚠️ High source concentration' if float(p) >= 80 else 'OK')
        
        if len(sec_terms) > 0:
            fig = px.bar(
                sec_terms.reset_index(drop=True),
                y=sec_terms.index,
                x='security_score',
                orientation='h',
                title='Top 15 Documents by Security Focus Score',
                color_discrete_sequence=['#A23B72']
            )
            st.plotly_chart(fig, width='stretch')
            render_data_basis_caption(run_meta, "Top security documents", point_count=len(sec_terms))
            sec_display_cols = [c for c in ['date', 'title', 'source', 'source_share_pct', 'source_balance_warning', 'security_score'] if c in sec_terms.columns]
            st.dataframe(sec_terms[sec_display_cols], width='stretch', hide_index=True)

        st.subheader("Cross-source contradiction scan")
        contradiction_df = detect_contradictions(processed_df)
        if isinstance(contradiction_df, pd.DataFrame) and len(contradiction_df) > 0:
            st.warning(f"Detected {len(contradiction_df)} potential contradiction pairs. Review before policy interpretation.")
            st.dataframe(contradiction_df.head(30), width='stretch', hide_index=True)
        else:
            st.info("No cross-source contradictions detected by current heuristic.")

        st.subheader("Evidence (top documents by shift signal)")
        if 'economic_score' in scored_df.columns and 'security_score' in scored_df.columns:
            tmp = scored_df.copy()
            tmp['economic_score'] = pd.to_numeric(tmp['economic_score'], errors='coerce').fillna(0.0)
            tmp['security_score'] = pd.to_numeric(tmp['security_score'], errors='coerce').fillna(0.0)
            tmp['shift_signal'] = tmp['security_score'] - tmp['economic_score']
            ev_cols = [c for c in ['date', 'year', 'title', 'source', 'url', 'economic_score', 'security_score', 'shift_signal'] if c in tmp.columns]
            evidence = tmp.sort_values('shift_signal', ascending=False).head(10)[ev_cols]
            st.dataframe(evidence, width='stretch')

            export_df = evidence.copy()
            export_df["review_decision"] = ""
            export_df["review_notes"] = ""
            st.download_button(
                label="Download evidence (CSV for review)",
                data=export_df.to_csv(index=False, encoding="utf-8").encode("utf-8"),
                file_name=f"evidence_review_shift_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )
        
        st.info("""
        **How to read this page:**
        
        - **Table:** Average economic and security scores per year
        - **Distribution charts:** How scores cluster — most documents lean economic or security?
        - **Top documents:** The strongest examples of each type
        """)

        render_page_exports(
            page_slug="economic_vs_security",
            tables={
                "yearly_stats": yearly_df,
                "top_economic_docs": eco_terms if isinstance(eco_terms, pd.DataFrame) else pd.DataFrame(),
                "top_security_docs": sec_terms if isinstance(sec_terms, pd.DataFrame) else pd.DataFrame(),
            },
            run_meta=run_meta,
        )
    
    # =====================================================================
    # PAGE 3: TONE & SENTIMENT ANALYSIS
    # =====================================================================
    elif page == lang["tone_sentiment"]:
        st.header(f"Tone & Mood Analysis")
        render_page_scaffold(
            question=f"How does tone evolve over time, and which document excerpts substantiate that tone profile?",
            run_meta=run_meta,
            exploratory_mode=exploratory_mode,
        )
        st.markdown(f"How urgent and positive is the language in {country1_name}-{country2_name} communications (heuristic signals)?")

        st.caption("Tone, urgency, and sentiment are heuristic signals; treat them as indicators and verify using underlying texts.")
        
        # Analyze tone
        with st.spinner("Analyzing tone and sentiment..."):
            if len(processed_df) < MIN_DOCS_TONE_THEME:
                st.warning(
                    f"Tone analysis is running on {len(processed_df)} docs; recommended minimum is {MIN_DOCS_TONE_THEME}.",
                    icon="⚠️",
                )
            tone_analyzer, tone_df = perform_tone_analysis(processed_df)

        render_top_takeaways(
            role_profile=role_profile,
            page_key="tone",
            context={
                "docs": len(processed_df),
                "years": processed_df["year"].nunique() if "year" in processed_df.columns else 0,
                "confidence_label": "n/a",
                "trend": "tone trend",
                "top_issue": "n/a",
            },
        )
        
        # Check if analysis was successful
        if tone_analyzer is None or tone_df is None or len(tone_df) == 0:
            st.error("Unable to perform tone analysis. Please check that data has been loaded correctly.")
            st.stop()

        tone_charts_enabled = bool(len(processed_df) >= MIN_DOCS_CHART_HARD_GUARD)
        if not tone_charts_enabled:
            st.warning(
                f"Tone charts are disabled because filtered docs ({len(processed_df)}) are below the hard threshold ({MIN_DOCS_CHART_HARD_GUARD}).",
                icon="⚠️",
            )
        
        # Tone distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Tone Distribution")
            tone_dist = tone_analyzer.get_tone_distribution(tone_df)
            tone_dist_df = pd.DataFrame(
                {
                    "tone": list(tone_dist.keys()),
                    "count": list(tone_dist.values()),
                }
            ).sort_values("count", ascending=False)
            if tone_charts_enabled:
                fig = px.bar(
                    tone_dist_df,
                    x="tone",
                    y="count",
                    title="Distribution of Diplomatic Tones",
                )
                st.plotly_chart(fig, width='stretch')
                render_data_basis_caption(run_meta, "Tone distribution", point_count=len(tone_df))
            else:
                st.dataframe(tone_dist_df, width='stretch', hide_index=True)
        
        with col2:
            st.subheader("Sentiment Distribution")
            sentiment_dist = tone_analyzer.get_sentiment_distribution(tone_df)
            sentiment_dist_df = pd.DataFrame(
                {
                    "sentiment": list(sentiment_dist.keys()),
                    "count": list(sentiment_dist.values()),
                }
            ).sort_values("count", ascending=False)
            if tone_charts_enabled:
                fig = px.bar(
                    sentiment_dist_df,
                    x="sentiment",
                    y="count",
                    title="Distribution of Sentiments",
                    color_discrete_map={
                        'Positive': '#2ecc71',
                        'Neutral': '#95a5a6',
                        'Negative': '#e74c3c'
                    },
                )
                st.plotly_chart(fig, width='stretch')
                render_data_basis_caption(run_meta, "Sentiment distribution", point_count=len(tone_df))
            else:
                st.dataframe(sentiment_dist_df, width='stretch', hide_index=True)
        
        # Yearly tone statistics
        st.subheader("Yearly Tone Trends")
        yearly_tone = tone_analyzer.get_yearly_tone_statistics(tone_df)

        if yearly_tone is None or len(yearly_tone) == 0:
            st.warning("No yearly tone statistics available for the current dataset.")
            st.info("Fallback: Yearly trend visual is hidden because no valid yearly tone rows are available.")
        elif not tone_charts_enabled:
            st.info("Fallback: Yearly tone charts are disabled under the hard minimum-doc threshold.")
            fallback_cols = [c for c in ['year', 'urgency_score_mean', 'sentiment_polarity_mean', 'urgency_score_std', 'sentiment_polarity_std'] if c in yearly_tone.columns]
            if fallback_cols:
                st.dataframe(yearly_tone[fallback_cols], width='stretch', hide_index=True)
        else:
            if len(yearly_tone) < 5:
                st.warning("Yearly tone trends are based on a small number of years; interpret cautiously.")

            yearly_counts = (
                tone_df.assign(year=pd.to_numeric(tone_df['year'], errors='coerce'))
                .dropna(subset=['year'])
                .groupby('year')
                .size()
                .reset_index(name='documents')
            )
            yearly_counts['year'] = yearly_counts['year'].astype(int)

            tone_trend_df = yearly_tone.copy()
            tone_trend_df['year'] = pd.to_numeric(tone_trend_df['year'], errors='coerce').astype('Int64')
            tone_trend_df = tone_trend_df.dropna(subset=['year']).copy()
            tone_trend_df['year'] = tone_trend_df['year'].astype(int)
            tone_trend_df = tone_trend_df.merge(yearly_counts, on='year', how='left')
            tone_trend_df['documents'] = pd.to_numeric(tone_trend_df['documents'], errors='coerce').fillna(0).astype(int)

            adequacy = compute_chart_adequacy(tone_trend_df, 'year', 'urgency_score_mean', min_years=4, min_points=8, min_variance=5e-5)
            render_data_adequacy_badges('Tone yearly trend', adequacy)

            if adequacy.get('low_info', False):
                reasons = ", ".join(adequacy.get('reasons', []))
                st.info(f"Fallback: Tone trend chart hidden due to low-information signal ({reasons}).")
                fallback_cols = [c for c in ['year', 'documents', 'urgency_score_mean', 'sentiment_polarity_mean'] if c in tone_trend_df.columns]
                if fallback_cols:
                    st.dataframe(tone_trend_df[fallback_cols], width='stretch', hide_index=True)
            else:
                tone_trend_df['urgency_ci95'] = 1.96 * (
                    pd.to_numeric(tone_trend_df.get('urgency_score_std', pd.Series([0.0] * len(tone_trend_df))), errors='coerce').fillna(0.0)
                    / tone_trend_df['documents'].clip(lower=1).pow(0.5)
                )
                tone_trend_df['sentiment_ci95'] = 1.96 * (
                    pd.to_numeric(tone_trend_df.get('sentiment_polarity_std', pd.Series([0.0] * len(tone_trend_df))), errors='coerce').fillna(0.0)
                    / tone_trend_df['documents'].clip(lower=1).pow(0.5)
                )

                c_badge_1, c_badge_2, c_badge_3 = st.columns(3)
                with c_badge_1:
                    st.metric("Years in trend", int(tone_trend_df['year'].nunique()))
                with c_badge_2:
                    st.metric("Median docs/year", int(tone_trend_df['documents'].median()) if len(tone_trend_df) > 0 else 0)
                with c_badge_3:
                    st.metric("Min docs/year", int(tone_trend_df['documents'].min()) if len(tone_trend_df) > 0 else 0)

                fig = go.Figure()

                fig.add_trace(go.Scatter(
                    x=tone_trend_df['year'],
                    y=tone_trend_df['urgency_score_mean'] + tone_trend_df['urgency_ci95'],
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo='skip'
                ))
                fig.add_trace(go.Scatter(
                    x=tone_trend_df['year'],
                    y=tone_trend_df['urgency_score_mean'] - tone_trend_df['urgency_ci95'],
                    mode='lines',
                    line=dict(width=0),
                    fill='tonexty',
                    fillcolor='rgba(231,76,60,0.12)',
                    name='Urgency 95% CI',
                    hoverinfo='skip'
                ))
                fig.add_trace(go.Scatter(
                    x=tone_trend_df['year'],
                    y=tone_trend_df['urgency_score_mean'],
                    mode='lines+markers+text',
                    text=tone_trend_df['documents'],
                    textposition='top center',
                    name='Urgency Score',
                    line=dict(color='#e74c3c', width=2),
                    marker=dict(size=6)
                ))

                fig.add_trace(go.Scatter(
                    x=tone_trend_df['year'],
                    y=tone_trend_df['sentiment_polarity_mean'] + tone_trend_df['sentiment_ci95'],
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo='skip'
                ))
                fig.add_trace(go.Scatter(
                    x=tone_trend_df['year'],
                    y=tone_trend_df['sentiment_polarity_mean'] - tone_trend_df['sentiment_ci95'],
                    mode='lines',
                    line=dict(width=0),
                    fill='tonexty',
                    fillcolor='rgba(46,204,113,0.12)',
                    name='Sentiment 95% CI',
                    hoverinfo='skip'
                ))
                fig.add_trace(go.Scatter(
                    x=tone_trend_df['year'],
                    y=tone_trend_df['sentiment_polarity_mean'],
                    mode='lines+markers',
                    name='Sentiment Polarity',
                    line=dict(color='#2ecc71', width=2),
                    marker=dict(size=6)
                ))

                fig.update_layout(
                    title="Tone and Sentiment Trends Over Time (95% confidence bands; labels=docs/year)",
                    xaxis_title="Year",
                    yaxis_title="Score",
                    hovermode='x unified',
                    height=420
                )

                st.plotly_chart(fig, width='stretch')
                render_data_basis_caption(run_meta, "Tone-sentiment yearly trend", point_count=len(tone_trend_df))

        st.markdown("#### Signal")
        st.subheader("Yearly Tone Composition (stacked)")
        if {'year', 'tone_class'}.issubset(set(tone_df.columns)):
            tone_year = (
                tone_df.assign(year=pd.to_numeric(tone_df['year'], errors='coerce'))
                .dropna(subset=['year'])
                .groupby(['year', 'tone_class'])
                .size()
                .reset_index(name='documents')
            )
            if len(tone_year) > 0:
                if tone_charts_enabled:
                    fig_tone_stack = px.bar(
                        tone_year,
                        x='year',
                        y='documents',
                        color='tone_class',
                        barmode='stack',
                        title='Tone class distribution by year',
                    )
                    st.plotly_chart(fig_tone_stack, width='stretch')
                    render_data_basis_caption(run_meta, "Tone stack by year", point_count=len(tone_year))
                else:
                    st.dataframe(tone_year, width='stretch', hide_index=True)

        if {'year', 'sentiment_class'}.issubset(set(tone_df.columns)):
            sentiment_year = (
                tone_df.assign(year=pd.to_numeric(tone_df['year'], errors='coerce'))
                .dropna(subset=['year'])
                .groupby(['year', 'sentiment_class'])
                .size()
                .reset_index(name='documents')
            )
            if len(sentiment_year) > 0:
                if tone_charts_enabled:
                    fig_sent_stack = px.bar(
                        sentiment_year,
                        x='year',
                        y='documents',
                        color='sentiment_class',
                        barmode='stack',
                        title='Sentiment class distribution by year',
                    )
                    st.plotly_chart(fig_sent_stack, width='stretch')
                    render_data_basis_caption(run_meta, "Sentiment stack by year", point_count=len(sentiment_year))
                else:
                    st.dataframe(sentiment_year, width='stretch', hide_index=True)

        st.info("""
        **How to read this page:**
        
        - **Tone:** Formal vs Cordial — how stiff or friendly is the language?
        - **Sentiment:** Positive (praising), Neutral (factual), Negative (concerned)
        - **Urgency:** How pressing are the issues discussed? Higher = more urgent
        - **Trends:** Are conversations becoming more urgent or more positive over time?
        """)

        render_page_exports(
            page_slug="tone_and_mood",
            tables={
                "tone_records": tone_df,
                "yearly_tone": yearly_tone if isinstance(yearly_tone, pd.DataFrame) else pd.DataFrame(),
            },
            run_meta=run_meta,
        )
    
    # =====================================================================
    # PAGE 4: THEMATIC ANALYSIS
    # =====================================================================
    elif page == lang["thematic"]:
        st.header(f"Topics & Themes")
        render_page_scaffold(
            question=f"Which strategic themes are credible, and what exemplar documents support each theme?",
            run_meta=run_meta,
            exploratory_mode=exploratory_mode,
        )
        st.markdown(f"What topics come up most in {country1_name}-{country2_name} diplomacy, and how have they changed?")
        
        with st.spinner("Performing thematic analysis..."):
            if len(processed_df) < MIN_DOCS_TONE_THEME:
                st.warning(
                    f"Thematic analysis is running on {len(processed_df)} docs; recommended minimum is {MIN_DOCS_TONE_THEME}.",
                    icon="⚠️",
                )
            thematic_analyzer, analysis, df_themes = perform_thematic_analysis(processed_df)

        render_top_takeaways(
            role_profile=role_profile,
            page_key="themes",
            context={
                "docs": len(processed_df),
                "years": processed_df["year"].nunique() if "year" in processed_df.columns else 0,
                "confidence_label": "n/a",
                "trend": "theme evolution",
                "top_issue": "n/a",
            },
        )

        thematic_charts_enabled = bool(len(processed_df) >= MIN_DOCS_CHART_HARD_GUARD)
        if not thematic_charts_enabled:
            st.warning(
                f"Theme charts are disabled because filtered docs ({len(processed_df)}) are below the hard threshold ({MIN_DOCS_CHART_HARD_GUARD}).",
                icon="⚠️",
            )
        
        # Display discovered themes in a better format
        st.subheader("Discovered Themes")
        st.markdown(f"*The computer found these topic groups automatically by reading all the documents:*")
        
        # Create better layout for themes
        for i, (topic_id, words) in enumerate(sorted(analysis['overall_themes'].items())):
            # Add visual indicator
            colors = ['🟦', '🟥', '🟩', '🟨', '🟪']
            color = colors[i % 5]
            
            with st.container():
                st.markdown(f"""
                {color} **Theme {topic_id}**
                
                **Keywords:** {', '.join(words[:6])}
                
                *Supporting terms:* {', '.join(words[6:])}
                """)
                st.divider()

        st.subheader("Topic quality diagnostics")
        overall_themes = analysis.get('overall_themes', {}) if isinstance(analysis, dict) else {}
        keyword_pool = [str(w).strip() for words in overall_themes.values() if isinstance(words, list) for w in words if str(w).strip()]
        unique_keywords = len(set(keyword_pool))
        total_keywords = len(keyword_pool)
        lexical_diversity = (unique_keywords / total_keywords) if total_keywords > 0 else 0.0

        theme_balance = 0.0
        if isinstance(df_themes, pd.DataFrame) and len(df_themes) > 0:
            topic_col = "primary_topic" if "primary_topic" in df_themes.columns else None
            if topic_col:
                counts = df_themes[topic_col].value_counts(normalize=True)
                if len(counts) > 1:
                    entropy_val = float(stats.entropy(counts))
                    max_entropy = float(pd.Series([1 / len(counts)] * len(counts)).pipe(lambda s: stats.entropy(s)))
                    theme_balance = (entropy_val / max_entropy) if max_entropy > 0 else 0.0

        q1, q2, q3 = st.columns(3)
        with q1:
            st.metric("Theme count", int(len(overall_themes)) if isinstance(overall_themes, dict) else 0)
        with q2:
            st.metric("Keyword diversity", f"{lexical_diversity:.2f}")
        with q3:
            st.metric("Theme balance", f"{theme_balance:.2f}")

        st.caption("Keyword diversity and theme balance are diagnostic proxies (higher values generally indicate broader, less collapsed topics).")

        st.subheader("Exemplar documents by theme")
        if isinstance(df_themes, pd.DataFrame) and len(df_themes) > 0:
            topic_col = "primary_topic" if "primary_topic" in df_themes.columns else None
            if topic_col:
                exemplar_rows = []
                for topic_id, grp in df_themes.groupby(topic_col):
                    for _, row in grp.head(3).iterrows():
                        exemplar_rows.append({
                            "theme": f"Theme {topic_id}",
                            "date": row.get("date", ""),
                            "source": row.get("source", ""),
                            "title": row.get("title", ""),
                            "excerpt": str(row.get("cleaned", ""))[:220] + ("..." if len(str(row.get("cleaned", ""))) > 220 else ""),
                        })
                if exemplar_rows:
                    st.dataframe(pd.DataFrame(exemplar_rows), width='stretch', hide_index=True)
                else:
                    st.info("No exemplars available for current theme assignments.")
            else:
                st.info("Primary topic assignments are unavailable for exemplar extraction.")
        
        # Topic distribution by year
        st.subheader("How Themes Evolved Over Time")

        try:
            year_span = int(processed_df['year'].nunique()) if 'year' in processed_df.columns else 0
            if year_span and year_span < 5:
                st.warning("Theme evolution is based on a small number of years; treat the chart as exploratory.")
        except Exception:
            pass
        
        # Prepare data for visualization
        topic_year_data = []
        dist = analysis.get('topic_weights_by_year') or analysis.get('topic_distribution_by_year') or {}
        for year, topic_list in dist.items():
            if topic_list:
                for topic_id, prob in topic_list[:3]:
                    topic_year_data.append({'Year': year, 'Theme': f'Theme {topic_id}', 'ThemeWeight': prob})

        if topic_year_data:
            topic_year_df = pd.DataFrame(topic_year_data)
            topic_year_df['Year'] = pd.to_numeric(topic_year_df['Year'], errors='coerce')
            topic_year_df['ThemeWeight'] = pd.to_numeric(topic_year_df['ThemeWeight'], errors='coerce')

            adequacy = compute_chart_adequacy(topic_year_df, 'Year', 'ThemeWeight', min_years=4, min_points=12, min_variance=5e-5)
            render_data_adequacy_badges('Theme evolution', adequacy)

            dominance_avg = 0.0
            if len(topic_year_df) > 0:
                year_sum = topic_year_df.groupby('Year', as_index=False)['ThemeWeight'].sum().rename(columns={'ThemeWeight': 'sum_w'})
                year_max = topic_year_df.groupby('Year', as_index=False)['ThemeWeight'].max().rename(columns={'ThemeWeight': 'max_w'})
                dom = year_max.merge(year_sum, on='Year', how='left')
                dom['dominance'] = dom['max_w'] / dom['sum_w'].replace(0, pd.NA)
                dominance_avg = float(pd.to_numeric(dom['dominance'], errors='coerce').dropna().mean()) if len(dom) > 0 else 0.0

            gate_reasons = list(adequacy.get('reasons', []))
            if dominance_avg > 0.85:
                gate_reasons.append(f"high dominance ({dominance_avg:.2f})")

            if not thematic_charts_enabled:
                gate_reasons.append(f"hard guard: docs < {MIN_DOCS_CHART_HARD_GUARD}")

            if gate_reasons:
                st.info(f"Fallback: Theme evolution charts hidden due to low-information signal ({', '.join(gate_reasons)}).")
                fallback_theme = (
                    topic_year_df.groupby('Theme', as_index=False)['ThemeWeight']
                    .mean()
                    .sort_values('ThemeWeight', ascending=False)
                )
                if len(fallback_theme) > 0:
                    st.dataframe(fallback_theme, width='stretch', hide_index=True)
            else:
                fig = px.bar(
                    topic_year_df,
                    x='Year',
                    y='ThemeWeight',
                    color='Theme',
                    title='Themes by Year (Stacked)',
                    barmode='stack',
                    color_discrete_sequence=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
                )
                fig.update_layout(
                    xaxis_title='Year',
                    yaxis_title='Theme weight (relative)',
                    hovermode='x unified',
                    height=500
                )
                st.plotly_chart(fig, width='stretch')
                render_data_basis_caption(run_meta, "Theme evolution (stacked)", point_count=len(topic_year_df))
                render_data_basis_caption(run_meta, "Theme evolution", point_count=len(topic_year_df))
        else:
            st.info("Fallback: Theme evolution visual unavailable because no valid yearly theme points were produced.")

        # Show themes explanation
        st.info("""
        **How to read this page:**
        
        - **Each theme:** A group of related topics found automatically by the computer
        - **Keywords:** The most distinctive words defining each theme
        - **Bar chart:** Shows which themes dominated each year
        - **Note:** Themes are unsupervised (LDA) groups; treat them as exploratory signals and verify with source text
        """)

        render_page_exports(
            page_slug="topics_and_themes",
            tables={
                "theme_assignments": df_themes if isinstance(df_themes, pd.DataFrame) else pd.DataFrame(),
                "theme_evolution": topic_year_df if 'topic_year_df' in locals() and isinstance(topic_year_df, pd.DataFrame) else pd.DataFrame(),
            },
            run_meta=run_meta,
        )

    # =====================================================================
    # PAGE 5: YEAR EXPLORER
    # =====================================================================
    elif page == lang["time_machine"]:
        st.header(f"Year-by-Year Explorer")
        render_page_scaffold(
            question=f"What changed between years in diplomatic language and focus signals?",
            run_meta=run_meta,
            exploratory_mode=exploratory_mode,
        )
        st.markdown(f"Slide through time to see what {country1_name} and {country2_name} were discussing each year")
        render_top_takeaways(
            role_profile=role_profile,
            page_key="year_explorer",
            context={
                "docs": len(processed_df),
                "years": processed_df["year"].nunique() if "year" in processed_df.columns else 0,
                "confidence_label": "n/a",
                "trend": "year-to-year",
                "top_issue": "n/a",
            },
        )
        
        valid_years = sorted([int(y) for y in processed_df['year'].dropna().unique() if pd.notna(y)])
        if not valid_years:
            st.warning("No valid years available in the current document set.")
            return

        if len(valid_years) == 1:
            selected_year = valid_years[0]
            st.info(f"Only one year available: {selected_year}")
        else:
            selected_year = st.slider(
                "Select a Year",
                min_value=int(min(valid_years)),
                max_value=int(max(valid_years)),
                value=int(valid_years[len(valid_years)//2])
            )

        compare_mode = st.checkbox("Enable year-vs-year compare mode", value=False)
        compare_year = None
        if compare_mode:
            compare_options = [int(y) for y in valid_years if int(y) != int(selected_year)]
            if compare_options:
                compare_year = st.selectbox(
                    "Compare with year",
                    compare_options,
                    index=0,
                )
            else:
                st.caption("No alternate year available for comparison.")
        
        # Get documents from selected year
        year_data = processed_df[processed_df['year'] == selected_year]
        
        st.subheader(f"Year {selected_year}")
        st.info(f"{len(year_data)} documents from this year")
        
        if len(year_data) > 0:
            # Summarize top terms for the selected year
            all_lemmas = []
            for lemmas in year_data['lemmas']:
                if isinstance(lemmas, list):
                    all_lemmas.extend(lemmas)
            
            if all_lemmas:
                top_terms = (
                    pd.Series(all_lemmas)
                    .astype(str)
                    .str.strip()
                    .replace("", pd.NA)
                    .dropna()
                    .value_counts()
                    .head(20)
                    .reset_index()
                )
                top_terms.columns = ["term", "count"]
                fig_terms = px.bar(
                    top_terms.sort_values("count", ascending=True),
                    x="count",
                    y="term",
                    orientation="h",
                    title=f"Top Terms in {selected_year}",
                )
                st.plotly_chart(fig_terms, width='stretch')
                render_data_basis_caption(run_meta, "Top terms (selected year)", point_count=len(top_terms))

            if compare_mode and compare_year is not None:
                compare_data = processed_df[processed_df['year'] == int(compare_year)]
                st.subheader("What changed vs comparison year")
                d1 = int(len(year_data))
                d2 = int(len(compare_data))
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Docs (selected year)", d1)
                with c2:
                    st.metric("Docs (comparison year)", d2)
                with c3:
                    st.metric("Doc delta", d1 - d2)

                try:
                    _, scored_all, _ = perform_strategic_analysis(processed_df)
                    sy = scored_all[scored_all['year'] == int(selected_year)] if 'year' in scored_all.columns else pd.DataFrame()
                    cy = scored_all[scored_all['year'] == int(compare_year)] if 'year' in scored_all.columns else pd.DataFrame()
                    if len(sy) > 0 and len(cy) > 0:
                        selected_gap = float((pd.to_numeric(sy['security_score'], errors='coerce').fillna(0.0) - pd.to_numeric(sy['economic_score'], errors='coerce').fillna(0.0)).mean())
                        compare_gap = float((pd.to_numeric(cy['security_score'], errors='coerce').fillna(0.0) - pd.to_numeric(cy['economic_score'], errors='coerce').fillna(0.0)).mean())
                        st.caption(
                            f"Security-minus-economic shift delta: {selected_gap - compare_gap:+.4f} "
                            f"({selected_year}: {selected_gap:.4f} vs {compare_year}: {compare_gap:.4f})"
                        )
                except Exception:
                    st.caption("Strategic delta unavailable for selected comparison.")
            
            # Show key documents from this year
            st.subheader("Documents from This Year")
            for idx, row in year_data.iterrows():
                title_text = str(row.get('title', 'Document') or 'Document')
                expander_label = f"{title_text[:60]}..." if len(title_text) > 60 else title_text
                with st.expander(expander_label):
                    st.write(f"**Date:** {row.get('date', 'N/A')}")
                    st.write(f"**Location:** {row.get('location', 'N/A')}")
                    content_str = str(row.get('cleaned', '') or '')
                    st.write(f"**Content:** {content_str[:300]}..." if content_str else "**Content:** *(No content preview available)*")
        
        st.info("""
        **How to read this page:**
        
        - **Slider:** Pick any year to see what was happening
        - **Top terms chart:** Longer bars = mentioned more often that year
        - **Documents:** Actual text from that year's diplomatic meetings
        - **Try it:** Compare an early year vs a recent year to see how terms and focus change
        """)

        render_page_exports(
            page_slug="year_explorer",
            tables={
                "selected_year_documents": year_data if isinstance(year_data, pd.DataFrame) else pd.DataFrame(),
            },
            run_meta=run_meta,
        )
    
    # =====================================================================
    # PAGE 6: SEARCH & EXPLORE
    # =====================================================================
    elif page == lang["search_explore"]:
        st.header(f"Keyword Search")
        render_page_scaffold(
            question=f"Where exactly does a keyword/phrase appear, and in which sources, with traceable context?",
            run_meta=run_meta,
            exploratory_mode=exploratory_mode,
        )
        st.markdown(f"Search for any word and see how often it appears from {year_min} to {year_max} in {country1_name}-{country2_name} documents")
        
        # Keyword search
        search_term = st.text_input(
            "Enter a keyword to search",
            value="relations"
        )

        search_mode = st.selectbox("Search mode", ["exact", "contains", "lemma"], index=1)
        source_filter = []
        if "source" in processed_df.columns:
            source_opts = sorted(processed_df["source"].fillna("unknown").astype(str).unique())
            source_filter = st.multiselect("Source filter", source_opts, default=source_opts)
        
        if search_term:
            search_term_lower = search_term.lower()

            if "issue_tags" in processed_df.columns:
                tag_matches = processed_df[
                    processed_df["issue_tags"].map(
                        lambda tags: isinstance(tags, list) and any(search_term_lower in str(tag).lower() for tag in tags)
                    )
                ]
                if len(tag_matches) > 0:
                    st.caption(f"Issue-tag matches: {len(tag_matches)} documents")
            
            # Find documents containing the term
            search_space = processed_df.copy()
            if source_filter and "source" in search_space.columns:
                search_space = search_space[search_space["source"].astype(str).isin(source_filter)].copy()

            matching_mask = match_search_docs(search_space, search_term_lower, search_mode)
            matching_docs = search_space[matching_mask]

            render_top_takeaways(
                role_profile=role_profile,
                page_key="search",
                context={
                    "query": search_term,
                    "matches": int(len(matching_docs)),
                    "docs": len(processed_df),
                    "years": processed_df["year"].nunique() if "year" in processed_df.columns else 0,
                    "confidence_label": "n/a",
                    "trend": "search",
                    "top_issue": "n/a",
                },
            )
            
            if len(matching_docs) > 0:
                # Frequency by year
                year_freq = matching_docs.groupby('year').size()
                
                fig = px.bar(
                    x=year_freq.index,
                    y=year_freq.values,
                    labels={'x': 'Year', 'y': 'Mentions'},
                    title=f"Occurrences of '{search_term}' Over Time"
                )
                st.plotly_chart(fig, width='stretch')
                render_data_basis_caption(run_meta, f"Keyword search ({search_mode})", point_count=len(year_freq))
                
                # Show matching documents
                st.subheader(f"Documents Mentioning '{search_term}'")
                st.info(f"Found in {len(matching_docs)} documents")
                
                for idx, row in matching_docs.head(10).iterrows():
                    with st.expander(f"{row['title']} ({row['year']})"):
                        st.write(f"**Date:** {row['date']}")
                        st.write(f"**Location:** {row['location']}")
                        snippet = _extract_context_snippet(
                            text=str(row.get('cleaned', '')),
                            term=search_term_lower,
                            exact_phrase=(search_mode == "exact"),
                        )
                        st.markdown(snippet)

            else:
                st.warning(f"No documents found containing '{search_term}'")


        st.info("""
        **How to read this page:**
        
        - **Try searching:** "nuclear", "defense", "trade", "quad", "infrastructure"
        - **Rising bar chart:** That topic got more attention over time
        - **Falling bar chart:** That topic faded from discussions
        - **Documents:** Shows real excerpts where your keyword appears
        """)

        render_page_exports(
            page_slug="keyword_search",
            tables={
                "keyword_matches": matching_docs.head(200) if 'matching_docs' in locals() and isinstance(matching_docs, pd.DataFrame) else pd.DataFrame(),
            },
            run_meta=run_meta,
        )
    
    # =====================================================================
    # PAGE 7: STATISTICAL TESTS
    # =====================================================================
    elif page == lang["statistical_tests"]:
        render_stats_page(
            {
                "st": st,
                "country1_name": country1_name,
                "country2_name": country2_name,
                "run_meta": run_meta,
                "exploratory_mode": exploratory_mode,
                "processed_df": processed_df,
                "role_profile": role_profile,
                "external_ctx": external_ctx,
                "render_page_scaffold": render_page_scaffold,
                "get_analysis_bundle": get_analysis_bundle,
                "render_top_takeaways": render_top_takeaways,
                "build_claim_traceability": build_claim_traceability,
                "render_page_exports": render_page_exports,
                "MIN_DOCS_SIGNIFICANCE": MIN_DOCS_SIGNIFICANCE,
            }
        )


if __name__ == "__main__":
    main()
