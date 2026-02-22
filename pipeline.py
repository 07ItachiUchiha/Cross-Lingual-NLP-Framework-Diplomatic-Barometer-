"""Main pipeline — loads data, runs analysis, exports results"""

import sys
import os
from pathlib import Path
import logging
import pandas as pd
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from scrapers.data_loader import DataLoader
from preprocessing.preprocessor import Preprocessor
from analysis.strategic_shift_enhanced import StrategicShiftAnalyzer
from analysis.tone_analyzer import ToneAnalyzer
from analysis.thematic_clustering import ThematicAnalyzer
from utils.helpers import (
    save_analysis_results,
    export_to_excel,
    build_data_provenance_report,
    save_provenance_report,
)
from utils.config import PROCESSED_DATA_DIR, ANALYSIS_RESULTS_JSON, PROVENANCE_REPORT_JSON

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_full_pipeline():
    """Execute the complete analysis pipeline"""
    
    logger.info("=" * 70)
    logger.info("CROSS-LINGUAL NLP FRAMEWORK - FULL ANALYSIS PIPELINE")
    logger.info("=" * 70)
    
    try:
        # Step 1: Load Data
        logger.info("\n[STEP 1] Loading diplomatic documents...")
        loader = DataLoader()
        try:
            canonical_path = loader.write_or_update_canonical_corpus()
            if canonical_path:
                logger.info(f"✓ Canonical corpus updated: {canonical_path}")
        except Exception as exc:
            logger.warning(f"Canonical corpus update skipped: {exc}")
        raw_df = loader.load_combined_data()
        raw_count = len(raw_df)

        dedupe_subset = ["title", "date", "source"]
        df = raw_df.drop_duplicates(subset=dedupe_subset, keep="first").copy()
        duplicates_removed = raw_count - len(df)

        logger.info(f"✓ Loaded {raw_count} documents")
        if duplicates_removed > 0:
            logger.info(f"  Removed {duplicates_removed} duplicate rows using keys {dedupe_subset}")
        logger.info(f"  Date range: {df['date'].min()} to {df['date'].max()}")

        provenance_report = build_data_provenance_report(
            raw_df=raw_df,
            processed_df=df,
            source_path=str(loader.last_loaded_path) if loader.last_loaded_path else "unknown",
            duplicate_rows_removed=duplicates_removed,
            dedupe_subset=dedupe_subset,
            required_columns=sorted(list(DataLoader.REQUIRED_COLUMNS)),
        )
        save_provenance_report(provenance_report, str(PROVENANCE_REPORT_JSON))
        logger.info(f"✓ Data provenance report saved to {PROVENANCE_REPORT_JSON}")
        
        # Step 2: Preprocessing
        logger.info("\n[STEP 2] Preprocessing documents...")
        preprocessor = Preprocessor()
        processed_df = preprocessor.process_dataframe(df, content_column='content')
        logger.info(f"✓ Preprocessing complete")
        logger.info(f"  Documents processed: {len(processed_df)}")
        
        # Step 3: Strategic Shift Analysis
        logger.info("\n[STEP 3] Analyzing strategic shift...")
        shift_analyzer = StrategicShiftAnalyzer()
        shift_report, scored_df, yearly_df = shift_analyzer.generate_shift_report(processed_df)
        logger.info(f"✓ Strategic shift analysis complete")
        logger.info(f"  Crossover year: {shift_report['crossover_year']}")
        logger.info(f"  Overall trend: {shift_report['trend']}")
        
        # Step 4: Tone and Sentiment Analysis
        logger.info("\n[STEP 4] Analyzing tone and sentiment...")
        tone_analyzer = ToneAnalyzer()
        tone_df = tone_analyzer.process_dataframe(processed_df, text_column='cleaned')
        logger.info(f"✓ Tone analysis complete")
        tone_dist = tone_analyzer.get_tone_distribution(tone_df)
        logger.info(f"  Tone distribution: {tone_dist}")
        
        # Step 5: Thematic Analysis
        logger.info("\n[STEP 5] Performing thematic analysis...")
        thematic_analyzer = ThematicAnalyzer(n_topics=5)
        theme_analysis, theme_df = thematic_analyzer.analyze_theme_evolution(
            processed_df,
            text_column='cleaned'
        )
        logger.info(f"✓ Thematic analysis complete")
        logger.info(f"  Topics discovered: {theme_analysis['n_topics']}")
        
        # Step 6: Compile Results
        logger.info("\n[STEP 6] Compiling results...")

        def _normalize_title(value: object) -> str:
            text = str(value or "").strip().lower()
            return " ".join(text.split())

        def _normalize_date(value: object) -> str:
            try:
                dt = pd.to_datetime(value, errors="coerce")
                if pd.isna(dt):
                    return ""
                return dt.date().isoformat()
            except Exception:
                return ""

        def enrich_urls_from_latest_live_scrape(df_in: pd.DataFrame) -> pd.DataFrame:
            """Fill missing/empty url fields using the most recent live scrape CSV if available."""
            if df_in is None or len(df_in) == 0:
                return df_in

            if "url" in df_in.columns and df_in["url"].astype(str).str.strip().ne("").any():
                return df_in

            raw_dir = PROJECT_ROOT / "data" / "raw"
            candidates = sorted(raw_dir.glob("live_scrape_*.csv"), reverse=True)
            if not candidates:
                return df_in

            latest = candidates[0]
            try:
                live_df = pd.read_csv(latest)
            except Exception:
                return df_in

            required = {"date", "title", "source", "url"}
            if not required.issubset(set(live_df.columns)):
                return df_in

            live_df = live_df.copy()
            live_df["_key_date"] = live_df["date"].apply(_normalize_date)
            live_df["_key_title"] = live_df["title"].apply(_normalize_title)
            live_df["_key_source"] = live_df["source"].astype(str).str.upper().str.strip()
            live_df["url"] = live_df["url"].astype(str).str.strip()
            live_df = live_df[live_df["url"].ne("")]

            lookup = {}
            for _, row in live_df.iterrows():
                key = (row.get("_key_date", ""), row.get("_key_title", ""), row.get("_key_source", ""))
                if key not in lookup:
                    lookup[key] = row.get("url", "")

            out = df_in.copy()
            if "url" not in out.columns:
                out["url"] = ""
            out["_key_date"] = out["date"].apply(_normalize_date) if "date" in out.columns else ""
            out["_key_title"] = out["title"].apply(_normalize_title) if "title" in out.columns else ""
            out["_key_source"] = out["source"].astype(str).str.upper().str.strip() if "source" in out.columns else ""

            def fill_url(row):
                current = str(row.get("url", "")).strip()
                if current:
                    return current
                return lookup.get((row.get("_key_date", ""), row.get("_key_title", ""), row.get("_key_source", "")), "")

            out["url"] = out.apply(fill_url, axis=1)
            out = out.drop(columns=["_key_date", "_key_title", "_key_source"], errors="ignore")
            return out

        def build_decision_hygiene(
            raw_df: pd.DataFrame,
            scored_df: pd.DataFrame,
            yearly_df: pd.DataFrame,
            shift_report: dict,
        ) -> dict:
            """Lightweight decision-support metadata for safer policy-facing use.

            This is intentionally rule-based and transparent (not a model confidence score).
            """

            warnings = []
            notes = []

            # Coverage / gaps
            year_min = int(yearly_df["year"].min()) if len(yearly_df) else None
            year_max = int(yearly_df["year"].max()) if len(yearly_df) else None
            if year_min is not None and year_max is not None:
                expected_years = set(range(year_min, year_max + 1))
                present_years = set(int(x) for x in yearly_df["year"].dropna().unique())
                missing_years = sorted(list(expected_years - present_years))
            else:
                missing_years = []

            docs_per_year = (
                raw_df.groupby("year").size().sort_index().to_dict() if "year" in raw_df.columns else {}
            )
            low_volume_years = [
                int(y) for y, c in docs_per_year.items() if isinstance(y, (int, float)) and int(c) < 3
            ]

            # Source balance
            source_counts = raw_df["source"].value_counts(dropna=False).to_dict()
            total_docs = int(len(raw_df))
            dominant_source = None
            dominant_ratio = 0.0
            if total_docs > 0:
                for source, count in source_counts.items():
                    ratio = float(count) / float(total_docs)
                    if ratio > dominant_ratio:
                        dominant_ratio = ratio
                        dominant_source = str(source)

            if total_docs < 30:
                warnings.append(f"Small corpus: only {total_docs} documents. Treat trends as indicative, not definitive.")
            if missing_years:
                warnings.append(f"Coverage gap: missing years with zero documents: {missing_years[:20]}" + ("…" if len(missing_years) > 20 else ""))
            if low_volume_years:
                warnings.append(f"Low-volume years (fewer than 3 docs): {sorted(set(low_volume_years))[:20]}" + ("…" if len(set(low_volume_years)) > 20 else ""))
            if dominant_source is not None and dominant_ratio >= 0.80:
                warnings.append(
                    f"Source imbalance: {dominant_source} is {dominant_ratio:.0%} of documents. Findings may reflect source-specific phrasing."
                )

            crossover_year = shift_report.get("crossover_year")
            if crossover_year is None:
                notes.append("No crossover year detected; the shift may be gradual or the corpus too small in key years.")

            # Evidence table: documents with strongest shift signals
            evidence_rows = []
            if scored_df is not None and len(scored_df) > 0:
                working = enrich_urls_from_latest_live_scrape(scored_df.copy())
                if "date" in working.columns:
                    working["date"] = pd.to_datetime(working["date"], errors="coerce")
                if "year" not in working.columns and "date" in working.columns:
                    working["year"] = working["date"].dt.year

                def safe_num(col: str) -> pd.Series:
                    return pd.to_numeric(working.get(col), errors="coerce").fillna(0.0)

                working["shift_signal"] = safe_num("security_score") - safe_num("economic_score")

                if crossover_year is not None and "year" in working.columns:
                    focus = working[working["year"].isin([int(crossover_year) - 1, int(crossover_year), int(crossover_year) + 1])]
                    if len(focus) >= 5:
                        working = focus

                top = working.sort_values("shift_signal", ascending=False).head(10)
                for _, row in top.iterrows():
                    evidence_rows.append(
                        {
                            "date": str(row.get("date")) if pd.notna(row.get("date")) else "",
                            "year": int(row.get("year")) if pd.notna(row.get("year")) else None,
                            "title": str(row.get("title", ""))[:300],
                            "source": str(row.get("source", "")),
                            "url": str(row.get("url", "")),
                            "economic_score": float(row.get("economic_score", 0.0) or 0.0),
                            "security_score": float(row.get("security_score", 0.0) or 0.0),
                            "shift_signal": float(row.get("shift_signal", 0.0) or 0.0),
                        }
                    )

            if evidence_rows:
                empty_urls = sum(1 for r in evidence_rows if not str(r.get("url", "")).strip())
                if empty_urls / max(1, len(evidence_rows)) >= 0.5:
                    warnings.append(
                        "Many evidence rows have empty URLs. Add URLs to your corpus or run `--scrape-live` to generate matching URL metadata."
                    )

            return {
                "generated_at_utc": datetime.utcnow().isoformat() + "Z",
                "warnings": warnings,
                "notes": notes,
                "coverage": {
                    "total_documents": total_docs,
                    "year_min": year_min,
                    "year_max": year_max,
                    "missing_years": missing_years,
                    "documents_per_year": docs_per_year,
                    "source_counts": source_counts,
                },
                "evidence": {
                    "description": "Top documents most associated with the security-vs-economic shift (rule-based ranking).",
                    "rows": evidence_rows,
                },
            }
        
        full_results = {
            'metadata': {
                'total_documents': len(df),
                'date_range': {
                    'start': str(df['date'].min()),
                    'end': str(df['date'].max())
                },
                'provenance_report': str(PROVENANCE_REPORT_JSON),
                'decision_hygiene': build_decision_hygiene(raw_df=raw_df, scored_df=scored_df, yearly_df=yearly_df, shift_report=shift_report),
            },
            'strategic_shift': {
                'economic_focus_avg': float(shift_report['overall_economic_avg']),
                'security_focus_avg': float(shift_report['overall_security_avg']),
                'crossover_year': shift_report['crossover_year'],
                'trend': shift_report['trend']
            },
            'tone_and_sentiment': {
                'tone_distribution': tone_dist,
                'sentiment_distribution': tone_analyzer.get_sentiment_distribution(tone_df)
            },
            'themes': {
                'n_topics': theme_analysis['n_topics'],
                'overall_themes': {
                    str(k): v for k, v in theme_analysis['overall_themes'].items()
                }
            }
        }
        
        # Save results
        try:
            hygiene = full_results.get('metadata', {}).get('decision_hygiene', {})
            evidence_rows = hygiene.get('evidence', {}).get('rows', [])
            if evidence_rows:
                evidence_df = pd.DataFrame(evidence_rows)
                evidence_df["review_decision"] = ""
                evidence_df["review_notes"] = ""
                stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                evidence_csv = PROCESSED_DATA_DIR / f"evidence_review_{stamp}.csv"
                evidence_xlsx = PROCESSED_DATA_DIR / f"evidence_review_{stamp}.xlsx"
                evidence_df.to_csv(evidence_csv, index=False, encoding='utf-8')
                evidence_df.to_excel(evidence_xlsx, index=False)
                hygiene.setdefault('evidence', {})['exports'] = {
                    'review_csv': str(evidence_csv),
                    'review_excel': str(evidence_xlsx),
                }
        except Exception as exc:
            logger.warning(f"Decision hygiene evidence export failed: {exc}")

        save_analysis_results(full_results, str(ANALYSIS_RESULTS_JSON))
        logger.info(f"✓ Results saved to {ANALYSIS_RESULTS_JSON}")
        
        # Step 7: Export Data
        logger.info("\n[STEP 7] Exporting data files...")
        
        export_dfs = {
            'Processed Documents': processed_df[['date', 'title', 'location', 'cleaned']].head(20),
            'Yearly Analysis': yearly_df,
            'Tone Analysis': tone_analyzer.get_yearly_tone_statistics(tone_df)
        }
        
        excel_path = PROCESSED_DATA_DIR / "diplomatic_barometer_analysis.xlsx"
        export_to_excel(export_dfs, str(excel_path))
        logger.info(f"✓ Data exported to {excel_path}")
        
        # Step 8: Generate Report
        logger.info("\n[STEP 8] Final Summary")
        logger.info("=" * 70)
        logger.info(f"Total Documents Analyzed: {len(df)}")
        logger.info(f"Economic Focus Score: {shift_report['overall_economic_avg']:.4f}")
        logger.info(f"Security Focus Score: {shift_report['overall_security_avg']:.4f}")
        logger.info(f"Strategic Shift Crossover: {shift_report['crossover_year']}")
        logger.info(f"Overall Trend: {shift_report['trend']}")
        logger.info("=" * 70)
        logger.info("✓ PIPELINE COMPLETE")
        
        return {
            'status': 'success',
            'results': full_results,
            'data': {
                'processed_df': processed_df,
                'scored_df': scored_df,
                'tone_df': tone_df,
                'theme_df': theme_df
            }
        }
    
    except Exception as e:
        logger.error(f"\n✗ Pipeline failed: {str(e)}")
        return {
            'status': 'error',
            'error': str(e)
        }


if __name__ == "__main__":
    result = run_full_pipeline()
    
    # Print results to console
    if result['status'] == 'success':
        print("\n" + "=" * 70)
        print("ANALYSIS RESULTS")
        print("=" * 70)
        print(f"Overall Economic Focus: {result['results']['strategic_shift']['economic_focus_avg']:.4f}")
        print(f"Overall Security Focus: {result['results']['strategic_shift']['security_focus_avg']:.4f}")
        print(f"Crossover Year: {result['results']['strategic_shift']['crossover_year']}")
        print(f"Trend: {result['results']['strategic_shift']['trend']}")
        print("=" * 70)
    else:
        print(f"Error: {result['error']}")
