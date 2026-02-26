#!/usr/bin/env python
"""
Geopolitical Insight Engine — Unified Launcher
───────────────────────────────────────────────
Modes:
  python run.py                  # Classic: pipeline → Streamlit dashboard
    python run.py --provenance-only# Generate data provenance report only
    python run.py --pre-rag-smoke-check  # provenance + pipeline + tests
    python run.py --scrape-live    # Validate live scraping from official sources
        python run.py --resolve-corpus-urls  # Attempt to resolve URLs for corpus titles (writes canonical corpus)
  python run.py --api            # Start FastAPI backend only  (port 8000)
  python run.py --full           # Pipeline → API + Dashboard in parallel
  python run.py --ingest         # Ingest documents into RAG vector store
  python run.py --evaluate       # Run golden-dataset F1 + adversarial tests
  python run.py --digest         # Generate weekly digest PDF
  python run.py --cron           # Run cron updater (crawl + ingest new docs)
  python run.py --benchmark      # Run latency benchmarks
"""

import subprocess
import sys
import os
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent

BANNER = """
╔══════════════════════════════════════════════════════════════════════╗
║    Geopolitical Insight Engine                                     ║
║    LLM-Driven Strategic Analysis of Diplomatic Corpora             ║
╚══════════════════════════════════════════════════════════════════════╝
"""

DEFAULT_DIPLOMACY_RSS_FEEDS = [
    "https://data.gov.in/backend/dms/v1/rss.xml",
    "https://www.mea.gov.in/rss.xml",
    "https://www.un.org/sg/en/rss.xml",
    "https://www.gov.uk/government/organisations/foreign-commonwealth-development-office.atom",
]


def check_python():
    if sys.version_info < (3, 9):
        print("Error: Python 3.9+ required")
        sys.exit(1)
    print(f"  Python {sys.version_info.major}.{sys.version_info.minor} detected")

    if sys.version_info >= (3, 13):
        print("  Warning: spaCy has known compatibility issues on Python 3.13+ in this project stack.")
        print("  Recommended pre-RAG fix: use Python 3.11 environment.")
        print("  PowerShell: powershell -ExecutionPolicy Bypass -File scripts/setup_pre_rag_env.ps1")


def check_core_deps():
    """Verify minimum dependencies are installed."""
    missing = []
    for mod in ["streamlit", "pandas", "plotly", "spacy", "fastapi"]:
        try:
            __import__(mod)
        except Exception:
            missing.append(mod)
    if missing:
        print(f"  Missing: {', '.join(missing)}")
        print("  Install with: pip install -r requirements.txt")
        sys.exit(1)
    print("  Core dependencies OK")


def run_pipeline():
    """Execute legacy analysis pipeline."""
    print("\n[PIPELINE] Running analysis pipeline...")
    print("-" * 70)
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "pipeline.py")],
        cwd=str(PROJECT_ROOT),
    )
    if result.returncode != 0:
        print("  Pipeline failed")
        sys.exit(1)
    print("  Pipeline complete")


def run_provenance_only():
    """Generate and print data provenance report without running full analysis."""
    print("\n[PROVENANCE] Generating data provenance report...")
    print("-" * 70)
    try:
        from scrapers.data_loader import DataLoader
        from utils.helpers import build_data_provenance_report, save_provenance_report
        from utils.config import PROVENANCE_REPORT_JSON

        loader = DataLoader()
        raw_df = loader.load_combined_data()

        dedupe_subset = ["title", "date", "source"]
        dedup_df = raw_df.drop_duplicates(subset=dedupe_subset, keep="first").copy()
        duplicates_removed = len(raw_df) - len(dedup_df)

        report = build_data_provenance_report(
            raw_df=raw_df,
            processed_df=dedup_df,
            source_path=str(loader.last_loaded_path) if loader.last_loaded_path else "unknown",
            duplicate_rows_removed=duplicates_removed,
            dedupe_subset=dedupe_subset,
            required_columns=sorted(list(DataLoader.REQUIRED_COLUMNS)),
        )
        save_provenance_report(report, str(PROVENANCE_REPORT_JSON))

        print(f"  Saved: {PROVENANCE_REPORT_JSON}")
        print(f"  Rows: raw={report['row_counts']['raw_loaded']}, dedup={report['row_counts']['after_dedup']}, removed={report['row_counts']['duplicate_rows_removed']}")
        print(f"  Source split: {report.get('source_split', {})}")
        print(f"  Date range: {report.get('date_range', {})}")
    except Exception as e:
        print(f"  Provenance generation failed: {e}")
        sys.exit(1)


def run_pre_rag_smoke_check():
    """Run pre-RAG readiness checks in one command.

    Stages:
      1) Provenance generation
      2) Full pre-RAG pipeline
      3) Baseline test suite
    """
    print("\n[SMOKE-CHECK] Running pre-RAG smoke check...")
    print("-" * 70)

    # Stage 1: provenance
    print("[1/3] Provenance report")
    run_provenance_only()
    print("  ✓ Provenance report generated")

    # Stage 2: pipeline
    print("\n[2/3] Full pre-RAG pipeline")
    pipeline_result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "pipeline.py")],
        cwd=str(PROJECT_ROOT),
    )
    if pipeline_result.returncode != 0:
        print("  ✗ Pipeline failed")
        sys.exit(1)
    print("  ✓ Pipeline completed")

    # Stage 3: tests
    print("\n[3/3] Baseline tests")
    test_result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests", "-q"],
        cwd=str(PROJECT_ROOT),
    )
    if test_result.returncode != 0:
        print("  ✗ Tests failed")
        sys.exit(1)
    print("  ✓ Tests passed")

    print("\n[SMOKE-CHECK] SUCCESS: pre-RAG baseline is ready for presentation.")


def run_scrape_live():
    """Run live scraper validation against selected official sources."""
    print("\n[SCRAPE-LIVE] Running live scrape validation...")
    print("-" * 70)
    try:
        from scrapers.live_scrape_validator import LiveScrapeValidator

        validator = LiveScrapeValidator()
        report = validator.scrape_live()

        summary = report.get("summary", {})
        print(f"  Documents collected: {summary.get('total_documents', 0)}")
        print(f"  Successful sources: {summary.get('successful_sources', 0)}")
        print(f"  Reachable sources: {summary.get('reachable_sources', 0)}")
        print(f"  Failed sources: {summary.get('failed_sources', 0)}")
        print(f"  Report: {report.get('outputs', {}).get('report_json')}")
        if report.get('outputs', {}).get('documents_csv'):
            print(f"  Documents CSV: {report.get('outputs', {}).get('documents_csv')}")

        if not report.get("pass", False):
            print("  ✗ Live scrape validation failed (insufficient collected docs).")
            sys.exit(1)

        print("  ✓ Live scrape validation passed")
    except Exception as e:
        print(f"  Live scrape validation failed: {e}")
        sys.exit(1)


def run_resolve_corpus_urls(use_gdelt: bool = True, gdelt_per_row: int = 12):
    """Resolve URLs for the local corpus titles and persist to canonical corpus."""
    print("\n[RESOLVE-URLS] Resolving URLs for corpus titles...")
    print("-" * 70)
    try:
        from datetime import datetime
        import json
        from scrapers.data_loader import DataLoader
        from scrapers.corpus_url_resolver import CorpusURLResolver

        loader = DataLoader()
        df = loader.load_real_data()  # prefers canonical if present

        resolver = CorpusURLResolver()
        result = resolver.resolve(df, use_gdelt=bool(use_gdelt), max_gdelt_urls_per_row=int(gdelt_per_row))
        resolver.close()

        resolved = int(result.get("resolved", 0))
        attempted = int(result.get("attempted", 0))
        updated_df = result.get("updated_df")

        canonical_path = loader.data_dir / loader.canonical_filename
        if isinstance(updated_df, type(df)) and len(updated_df) > 0:
            updated_df.to_csv(canonical_path, index=False, encoding="utf-8")

        report = {
            "run_started_utc": datetime.utcnow().isoformat() + "Z",
            "attempted": attempted,
            "resolved": resolved,
            "output_canonical_csv": str(canonical_path),
            "rows": result.get("rows", []),
        }
        report_path = loader.data_dir / f"corpus_url_resolution_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)

        print(f"  Attempted: {attempted}")
        print(f"  Resolved:  {resolved}")
        print(f"  Canonical: {canonical_path}")
        print(f"  Report:    {report_path}")

        if resolved == 0:
            print("  Note: No high-confidence matches found. This can happen when corpus titles don't exist verbatim on current index pages.")
    except Exception as e:
        print(f"  URL resolution failed: {e}")
        sys.exit(1)


def run_fetch_external_signals(query: str, language: str = "en", max_per_provider: int = 100, timeframe_hours: int = 168):
    """Fetch optional external signals (news + GDELT) into data/raw."""
    print("\n[EXTERNAL] Fetching external signals (news + GDELT)...")
    print("-" * 70)
    try:
        from scrapers.external_signals import ExternalSignalsFetcher, ExternalSignalsConfig
        from utils.config import get_external_api_status

        status = get_external_api_status()
        configured = [k for k, v in status.items() if v and k.endswith("_configured")]
        print(f"  Providers configured: {', '.join(configured) if configured else 'none (GDELT only)'}")

        fetcher = ExternalSignalsFetcher()
        cfg = ExternalSignalsConfig(
            query=query,
            language=language,
            max_per_provider=int(max_per_provider),
            timeframe_hours=int(timeframe_hours),
        )
        result = fetcher.fetch_all(cfg)
        report = result.get("report", {})
        outputs = report.get("outputs", {})
        print(f"  Total items: {report.get('total_items', 0)}")
        print(f"  Output report: {outputs.get('report_json')}")
        if outputs.get("signals_csv"):
            print(f"  Output CSV: {outputs.get('signals_csv')}")
    except Exception as e:
        print(f"  External fetch failed: {e}")
        sys.exit(1)


def run_import_comtrade(path: str):
    """Import already-downloaded UN Comtrade exports (CSV/JSON) and normalize outputs."""
    print("\n[COMTRADE] Importing Comtrade export file...")
    print("-" * 70)
    try:
        from scrapers.comtrade_ingestion import ingest_from_file

        report = ingest_from_file(path, tag="import")
        outputs = report.get("outputs", {})
        print(f"  Rows: {report.get('rows', 0)}")
        print(f"  Years: {report.get('years', [])}")
        print(f"  Rows CSV: {outputs.get('rows_csv')}")
        print(f"  Yearly CSV: {outputs.get('yearly_csv')}")
        print(f"  Report: {outputs.get('report_json')}")
    except Exception as e:
        print(f"  Comtrade import failed: {e}")
        sys.exit(1)


def run_fetch_comtrade(
    reporter_code: str = "699",
    partner_code: str = "392",
    flow_code: str = "M",
    period: str = "2020,2021,2022,2023,2024",
    cmd_code: str = "TOTAL",
    key_preference: str = "primary",
):
    """Fetch UN Comtrade Final Data via official comtradeapicall client."""
    print("\n[COMTRADE] Fetching Comtrade Final Data via API...")
    print("-" * 70)
    try:
        from scrapers.comtrade_ingestion import ComtradeQuery, ingest_from_api

        q = ComtradeQuery(
            typeCode="C",
            freqCode="A",
            clCode="HS",
            period=str(period),
            reporterCode=str(reporter_code),
            partnerCode=str(partner_code),
            cmdCode=str(cmd_code),
            flowCode=str(flow_code),
        )
        report = ingest_from_api(q, key_preference=key_preference, tag="api")
        outputs = report.get("outputs", {})
        print(f"  Rows: {report.get('rows', 0)}")
        print(f"  Years: {report.get('years', [])}")
        if report.get("estimated_or_unreported_share") is not None:
            print(f"  Unreported/estimated share: {report.get('estimated_or_unreported_share'):.0%}")
        print(f"  Rows CSV: {outputs.get('rows_csv')}")
        print(f"  Yearly CSV: {outputs.get('yearly_csv')}")
        print(f"  Report: {outputs.get('report_json')}")
    except Exception as e:
        print(f"  Comtrade fetch failed: {e}")
        sys.exit(1)


def run_fetch_rss(urls: list[str], since_days: int = 14, max_items: int = 200):
    """Fetch RSS/Atom feeds and persist items into data/raw."""
    print("\n[RSS] Fetching RSS/Atom feeds...")
    print("-" * 70)
    try:
        from scrapers.rss_ingestion import RssFetchConfig, fetch_rss

        cfg = RssFetchConfig(urls=tuple(urls), since_days=int(since_days), max_items=int(max_items))
        report = fetch_rss(cfg)
        outputs = report.get("outputs", {})
        print(f"  Items: {report.get('items', 0)}")
        print(f"  CSV: {outputs.get('items_csv')}")
        print(f"  Report: {outputs.get('report_json')}")
    except Exception as e:
        print(f"  RSS fetch failed: {e}")
        sys.exit(1)


def run_summarize_rss(items_csv: str):
    """Summarize RSS items into counts by year/month."""
    print("\n[RSS] Summarizing RSS items...")
    print("-" * 70)
    try:
        from scrapers.rss_summary import summarize_rss_items

        report = summarize_rss_items(items_csv)
        outputs = report.get("outputs", {})
        print(f"  Items: {report.get('items', 0)}")
        if report.get("year_span"):
            print(f"  Year span: {report.get('year_span')}")
        print(f"  By-year CSV: {outputs.get('by_year_csv')}")
        print(f"  By-month CSV: {outputs.get('by_month_csv')}")
        print(f"  Report: {outputs.get('report_json')}")
    except Exception as e:
        print(f"  RSS summary failed: {e}")
        sys.exit(1)


def run_url_backfill(min_confidence: float = 0.92):
    """Propose URL backfill patch files without mutating canonical corpus."""
    print("\n[URL BACKFILL] Building non-destructive URL patch proposal...")
    print("-" * 70)
    try:
        from utils.url_backfill import BackfillConfig, generate_url_backfill_patch

        report = generate_url_backfill_patch(BackfillConfig(min_confidence=float(min_confidence)))
        outputs = report.get("outputs", {})
        print(f"  Canonical rows missing URL: {report.get('missing_url_rows', 0)}")
        print(f"  Proposed patch rows: {report.get('proposed_patch_rows', 0)}")
        print(f"  Review CSV: {outputs.get('review_csv')}")
        print(f"  Patch CSV: {outputs.get('patch_csv')}")
        print(f"  Preview Canonical CSV: {outputs.get('preview_canonical_csv')}")
        print(f"  Report: {outputs.get('report_json')}")
    except Exception as e:
        print(f"  URL backfill failed: {e}")
        sys.exit(1)


def run_assisted_url_backfill(review_csv: str = "", min_confidence: float = 0.82, max_rows: int = 200):
    """Second-stage assisted resolver using search results, still non-destructive."""
    print("\n[URL BACKFILL][ASSISTED] Resolving candidate links via search...")
    print("-" * 70)
    try:
        from pathlib import Path
        from utils.url_backfill import AssistedBackfillConfig, assist_url_backfill_from_review

        cfg = AssistedBackfillConfig(
            review_csv=Path(review_csv) if str(review_csv).strip() else None,
            min_confidence=float(min_confidence),
            max_rows=int(max_rows),
        )
        report = assist_url_backfill_from_review(cfg)
        outputs = report.get("outputs", {})
        print(f"  Rows scanned: {report.get('rows_scanned', 0)}")
        print(f"  Assisted patch rows: {report.get('assisted_patch_rows', 0)}")
        print(f"  Assisted review CSV: {outputs.get('assisted_review_csv')}")
        print(f"  Assisted patch CSV: {outputs.get('assisted_patch_csv')}")
        print(f"  Assisted preview Canonical CSV: {outputs.get('assisted_preview_canonical_csv')}")
        print(f"  Assisted report: {outputs.get('assisted_report_json')}")
    except Exception as e:
        print(f"  Assisted URL backfill failed: {e}")
        sys.exit(1)


def run_curated_url_backfill(review_csv: str = "", min_confidence: float = 0.86, max_rows: int = 300):
    """Third-stage deterministic resolver using curated official archive seed pages."""
    print("\n[URL BACKFILL][CURATED] Resolving from official archive seed pages...")
    print("-" * 70)
    try:
        from pathlib import Path
        from utils.url_backfill import CuratedArchiveBackfillConfig, backfill_from_curated_archives

        cfg = CuratedArchiveBackfillConfig(
            review_csv=Path(review_csv) if str(review_csv).strip() else None,
            min_confidence=float(min_confidence),
            max_rows=int(max_rows),
        )
        report = backfill_from_curated_archives(cfg)
        outputs = report.get("outputs", {})
        print(f"  Rows scanned: {report.get('rows_scanned', 0)}")
        print(f"  Curated candidates: {report.get('curated_candidate_rows', 0)}")
        print(f"  Curated patch rows: {report.get('curated_patch_rows', 0)}")
        print(f"  Curated review CSV: {outputs.get('curated_review_csv')}")
        print(f"  Curated patch CSV: {outputs.get('curated_patch_csv')}")
        print(f"  Curated preview Canonical CSV: {outputs.get('curated_preview_canonical_csv')}")
        print(f"  Curated report: {outputs.get('curated_report_json')}")
    except Exception as e:
        print(f"  Curated URL backfill failed: {e}")
        sys.exit(1)


def run_build_semi_auto_approval(curated_review_csv: str = "", medium_min_confidence: float = 0.70, recommend_approve_confidence: float = 0.82):
    """Build a pre-filled approval CSV (APPROVE/REJECT) from curated best candidates."""
    print("\n[URL BACKFILL][SEMI-AUTO] Building approval CSV from curated candidates...")
    print("-" * 70)
    try:
        from pathlib import Path
        from utils.url_backfill import SemiAutoApprovalConfig, build_semi_auto_approval_csv

        cfg = SemiAutoApprovalConfig(
            curated_review_csv=Path(curated_review_csv) if str(curated_review_csv).strip() else None,
            medium_min_confidence=float(medium_min_confidence),
            recommend_approve_confidence=float(recommend_approve_confidence),
        )
        report = build_semi_auto_approval_csv(cfg)
        outputs = report.get("outputs", {})
        print(f"  Approval rows: {report.get('approval_rows', 0)}")
        print(f"  Approval CSV: {outputs.get('approval_csv')}")
        print(f"  Report: {outputs.get('report_json')}")
    except Exception as e:
        print(f"  Build semi-auto approval failed: {e}")
        sys.exit(1)


def run_apply_semi_auto_approved_patch(approval_csv: str):
    """Apply APPROVE rows from approval CSV into non-destructive final patch + preview outputs."""
    print("\n[URL BACKFILL][SEMI-AUTO] Promoting APPROVE rows into final patch artifacts...")
    print("-" * 70)
    try:
        from utils.url_backfill import apply_semi_auto_approved_patch

        report = apply_semi_auto_approved_patch(approval_csv=approval_csv)
        outputs = report.get("outputs", {})
        print(f"  Approved rows in input: {report.get('approved_rows', 0)}")
        print(f"  Applied patch rows: {report.get('applied_patch_rows', 0)}")
        print(f"  Final patch CSV: {outputs.get('final_patch_csv')}")
        print(f"  Final preview Canonical CSV: {outputs.get('final_preview_canonical_csv')}")
        print(f"  Report: {outputs.get('report_json')}")
    except Exception as e:
        print(f"  Apply semi-auto patch failed: {e}")
        sys.exit(1)


def run_build_official_corpus(
    start_year: int = 2000,
    end_year: int = 2026,
    max_docs: int = 600,
    max_urls_per_year: int = 80,
    promote_to_canonical: bool = False,
):
    """Build a larger official corpus using GDELT discovery (MEA/MOFA domains)."""
    print("\n[CORPUS] Building official India–Japan corpus via GDELT...")
    print("-" * 70)
    try:
        from scrapers.official_corpus_builder import OfficialCorpusBuilder, CorpusBuildConfig
        from scrapers.data_loader import DataLoader
        import pandas as pd

        builder = OfficialCorpusBuilder()
        cfg = CorpusBuildConfig(
            start_year=int(start_year),
            end_year=int(end_year),
            max_docs_total=int(max_docs),
            max_urls_per_year=int(max_urls_per_year),
        )
        result = builder.build(cfg)
        builder.close()

        report = result.get("report", {})
        outputs = report.get("outputs", {})
        print(f"  Docs kept: {report.get('total_docs_kept', 0)}")
        print(f"  Corpus CSV: {outputs.get('corpus_csv')}")
        print(f"  Report:     {outputs.get('report_json')}")

        if promote_to_canonical and outputs.get("corpus_csv"):
            loader = DataLoader()
            canonical_path = loader.data_dir / loader.canonical_filename
            df = pd.read_csv(outputs.get("corpus_csv"))
            df.to_csv(canonical_path, index=False, encoding="utf-8")
            print(f"  Promoted to canonical: {canonical_path}")
            print("  Note: The pipeline will now prefer this canonical corpus by default.")
    except Exception as e:
        print(f"  Corpus build failed: {e}")
        sys.exit(1)


def run_build_india_france_corpus(
    start_year: int = 2000,
    end_year: int = 2026,
    max_docs: int = 500,
    max_urls_per_year: int = 80,
    min_content_chars: int = 850,
):
    """Build India–France corpus and write to pair-specific raw CSV files."""
    print("\n[CORPUS] Building India–France corpus via GDELT + official-site fallback...")
    print("-" * 70)
    try:
        from scrapers.india_france_corpus_builder import IndiaFranceCorpusBuilder, IndiaFranceBuildConfig

        builder = IndiaFranceCorpusBuilder()
        cfg = IndiaFranceBuildConfig(
            start_year=int(start_year),
            end_year=int(end_year),
            max_docs_total=int(max_docs),
            max_urls_per_year=int(max_urls_per_year),
            min_content_chars=int(min_content_chars),
        )
        result = builder.build(cfg)
        builder.close()

        report = result.get("report", {})
        outputs = report.get("outputs", {})
        print(f"  Docs kept:   {report.get('total_docs_kept', 0)}")
        print(f"  Source split:{report.get('source_split', {})}")
        print(f"  Primary CSV: {outputs.get('primary_csv')}")
        print(f"  Canonical:   {outputs.get('canonical_csv')}")
        print(f"  Report:      {outputs.get('report_json')}")
    except Exception as e:
        print(f"  India-France corpus build failed: {e}")
        sys.exit(1)


def run_api():
    """Start FastAPI backend."""
    print("\n[API] Starting FastAPI backend on http://localhost:8000")
    print("  Docs: http://localhost:8000/docs")
    print("  Press Ctrl+C to stop\n")
    subprocess.run(
        [sys.executable, "-m", "uvicorn", "api.main:app",
         "--host", "0.0.0.0", "--port", "8000", "--reload"],
        cwd=str(PROJECT_ROOT),
    )


def run_dashboard():
    """Start Streamlit dashboard."""
    print("\n[DASHBOARD] Starting Streamlit on http://localhost:8501")
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run",
         str(PROJECT_ROOT / "dashboard" / "app.py")],
        cwd=str(PROJECT_ROOT),
    )


def run_ingest():
    """Ingest documents into RAG vector store."""
    print("\n[INGEST] Loading documents into RAG pipeline...")
    try:
        from rag.rag_pipeline import RAGPipeline
        from scrapers.data_loader import DiplomaticDataLoader

        pipeline = RAGPipeline()
        loader = DiplomaticDataLoader()
        df = loader.load_data()
        stats = pipeline.ingest_dataframe(df)
        print(f"  Ingested {stats.get('documents', 0)} documents, "
              f"{stats.get('chunks_stored', 0)} chunks stored")
    except Exception as e:
        print(f"  Ingest failed: {e}")
        sys.exit(1)


def run_evaluate():
    """Run F1 evaluation + adversarial tests."""
    print("\n[EVALUATE] Running evaluation suite...")

    # Golden dataset F1
    try:
        from evaluation.golden_dataset import GoldenDataset, F1Evaluator
        ds = GoldenDataset()
        evaluator = F1Evaluator()
        # Baseline keyword classifier
        preds, labels = [], []
        for item in ds.get_all():
            text_lower = item["text"].lower()
            if any(w in text_lower for w in ["trade", "economic", "gdp", "investment", "cepa"]):
                preds.append("Economic")
            elif any(w in text_lower for w in ["defense", "military", "security", "naval"]):
                preds.append("Security")
            else:
                preds.append("Cultural")
            labels.append(item["label"])
        result = evaluator.evaluate(labels, preds)
        print(f"  Golden Dataset — Macro F1: {result['macro_f1']:.3f}")
        for cls, metrics in result["per_class"].items():
            print(f"    {cls}: P={metrics['precision']:.3f} R={metrics['recall']:.3f} F1={metrics['f1']:.3f}")
    except Exception as e:
        print(f"  Golden dataset evaluation failed: {e}")

    # Adversarial tests (structure validation)
    try:
        from evaluation.adversarial_tests import AdversarialTester
        tester = AdversarialTester()
        print(f"  Adversarial tests loaded: {len(tester.test_cases)} cases across "
              f"{len(set(t['category'] for t in tester.test_cases))} categories")
    except Exception as e:
        print(f"  Adversarial test validation failed: {e}")


def run_digest():
    """Generate weekly digest PDF."""
    print("\n[DIGEST] Generating weekly digest...")
    try:
        from automation.weekly_digest import WeeklyDigestGenerator
        gen = WeeklyDigestGenerator()
        path = gen.generate()
        print(f"  Digest saved to: {path}")
    except Exception as e:
        print(f"  Digest generation failed: {e}")


def run_cron():
    """Run cron updater (crawl + ingest)."""
    print("\n[CRON] Running scheduled crawl and ingest...")
    try:
        from automation.cron_updater import CronUpdater
        updater = CronUpdater()
        result = updater.run()
        print(f"  Found {result.get('new_docs_found', 0)} new documents, "
              f"ingested {result.get('docs_ingested', 0)}")
    except Exception as e:
        print(f"  Cron update failed: {e}")


def run_benchmark():
    """Run latency benchmarks."""
    print("\n[BENCHMARK] Running latency benchmarks...")
    try:
        from evaluation.latency_tests import LatencyBenchmark
        bench = LatencyBenchmark()
        report = bench.run_full_benchmark()
        bench.print_report(report)
    except Exception as e:
        print(f"  Benchmark failed: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Geopolitical Insight Engine — Unified Launcher"
    )
    parser.add_argument("--api", action="store_true", help="Start FastAPI backend only")
    parser.add_argument("--provenance-only", action="store_true", help="Generate data provenance report only")
    parser.add_argument("--pre-rag-smoke-check", action="store_true", help="Run provenance + pipeline + tests in one command")
    parser.add_argument("--scrape-live", action="store_true", help="Validate live scraping from selected official sources")
    parser.add_argument("--resolve-corpus-urls", action="store_true", help="Resolve official URLs for the local corpus titles")
    parser.add_argument("--no-gdelt", action="store_true", help="Disable GDELT discovery for --resolve-corpus-urls")
    parser.add_argument("--gdelt-per-row", type=int, default=12, help="Max GDELT candidate URLs to test per row during --resolve-corpus-urls")
    parser.add_argument("--fetch-news", action="store_true", help="Fetch external news/event signals (NewsData/NewsAPI/WorldNewsAPI/GDELT) into data/raw")
    parser.add_argument("--external-query", type=str, default="India Japan relations", help="Query used for --fetch-news")
    parser.add_argument("--external-language", type=str, default="en", help="Language for --fetch-news")
    parser.add_argument("--external-max", type=int, default=100, help="Max items per provider for --fetch-news")
    parser.add_argument("--external-timeframe-hours", type=int, default=168, help="Rolling window for providers that support timeframe")
    parser.add_argument(
        "--external-keywords",
        type=str,
        default="geopolitics,diplomacy,treaty,bilateral,summit",
        help="Comma-separated keywords appended to the query (default is geopolitics/diplomacy focused)",
    )
    parser.add_argument(
        "--external-categories",
        type=str,
        default="politics,world,business",
        help="Comma-separated categories (applied where providers support category filters)",
    )
    parser.add_argument(
        "--external-countries",
        type=str,
        default="in,jp",
        help="Comma-separated country codes (applied where providers support country filters; otherwise added as query hints)",
    )

    parser.add_argument("--build-official-corpus", action="store_true", help="Build a larger official India–Japan corpus via GDELT discovery")
    parser.add_argument("--build-india-france-corpus", action="store_true", help="Build India–France corpus into data/raw/india_france_documents.csv")
    parser.add_argument("--corpus-start-year", type=int, default=2000, help="Start year for --build-official-corpus")
    parser.add_argument("--corpus-end-year", type=int, default=2026, help="End year for --build-official-corpus")
    parser.add_argument("--corpus-max-docs", type=int, default=600, help="Max documents to keep for --build-official-corpus")
    parser.add_argument("--corpus-max-urls-per-year", type=int, default=80, help="Max GDELT URLs per year to try")
    parser.add_argument("--corpus-min-content-chars", type=int, default=850, help="Minimum extracted chars per page for corpus builders")
    parser.add_argument("--promote-corpus", action="store_true", help="Overwrite canonical corpus with the newly built official corpus")
    parser.add_argument("--full", action="store_true", help="Pipeline + API + Dashboard")
    parser.add_argument("--ingest", action="store_true", help="Ingest documents into RAG")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation suite")
    parser.add_argument("--digest", action="store_true", help="Generate weekly digest")
    parser.add_argument("--cron", action="store_true", help="Run cron updater")
    parser.add_argument("--benchmark", action="store_true", help="Run latency benchmarks")

    # -----------------------------------------------------------------
    # Structured data ingestion (pre-RAG friendly)
    # -----------------------------------------------------------------
    parser.add_argument("--import-comtrade", type=str, default="", help="Import a downloaded UN Comtrade CSV/JSON export")
    parser.add_argument("--fetch-comtrade", action="store_true", help="Fetch UN Comtrade Final Data via API (requires COMTRADE_* key in .env)")
    parser.add_argument("--comtrade-reporter", type=str, default="699", help="Comtrade reporterCode (default 699=India)")
    parser.add_argument("--comtrade-partner", type=str, default="392", help="Comtrade partnerCode (default 392=Japan)")
    parser.add_argument("--comtrade-flow", type=str, default="M", help="Flow code: M(import) or X(export)")
    parser.add_argument("--comtrade-period", type=str, default="2020,2021,2022,2023,2024", help="Comma-separated years (annual) or periods")
    parser.add_argument("--comtrade-cmd", type=str, default="TOTAL", help="Commodity code (default TOTAL)")
    parser.add_argument("--comtrade-key", type=str, default="primary", help="Key preference: primary or secondary")

    parser.add_argument("--fetch-rss", action="store_true", help="Fetch RSS/Atom feeds into data/raw")
    parser.add_argument(
        "--rss-urls",
        type=str,
        default="",
        help="Comma-separated RSS/Atom feed URLs (optional; built-in diplomacy defaults used if omitted)",
    )
    parser.add_argument("--rss-since-days", type=int, default=14, help="Only keep items newer than N days")
    parser.add_argument("--rss-max-items", type=int, default=200, help="Max items per feed to consider")

    parser.add_argument("--summarize-rss", type=str, default="", help="Summarize an RSS items CSV (from data/raw/rss_items_*.csv)")
    parser.add_argument("--backfill-urls", action="store_true", help="Create non-destructive URL backfill review + patch artifacts")
    parser.add_argument("--backfill-min-confidence", type=float, default=0.92, help="Minimum confidence for patch rows (default 0.92)")
    parser.add_argument("--assist-backfill-urls", action="store_true", help="Second-stage assisted URL resolver using web search over review file")
    parser.add_argument("--assist-backfill-review-csv", type=str, default="", help="Optional path to url_backfill_review_*.csv; latest is used if omitted")
    parser.add_argument("--assist-backfill-min-confidence", type=float, default=0.82, help="Minimum confidence for assisted patch rows (default 0.82)")
    parser.add_argument("--assist-backfill-max-rows", type=int, default=200, help="Max unresolved rows to process in assisted stage")
    parser.add_argument("--curated-backfill-urls", action="store_true", help="Third-stage deterministic URL resolver using curated official archive pages")
    parser.add_argument("--curated-backfill-review-csv", type=str, default="", help="Optional path to review CSV; latest is used if omitted")
    parser.add_argument("--curated-backfill-min-confidence", type=float, default=0.86, help="Minimum confidence for curated patch rows (default 0.86)")
    parser.add_argument("--curated-backfill-max-rows", type=int, default=300, help="Max unresolved rows to process in curated stage")
    parser.add_argument("--build-semi-auto-approval", action="store_true", help="Build pre-filled APPROVE/REJECT approval CSV from curated candidates")
    parser.add_argument("--semi-auto-curated-review-csv", type=str, default="", help="Optional curated review CSV path; latest is used if omitted")
    parser.add_argument("--semi-auto-min-confidence", type=float, default=0.70, help="Minimum confidence to include in approval CSV")
    parser.add_argument("--semi-auto-recommend-approve", type=float, default=0.82, help="Confidence threshold to prefill decision as APPROVE")
    parser.add_argument("--apply-semi-auto-approved", type=str, default="", help="Path to approval CSV; applies APPROVE rows into final patch artifacts")

    parser.add_argument("--fetch-ogd", action="store_true", help="Fetch a data.gov.in dataset by resource id into data/raw")
    parser.add_argument("--ogd-resource-id", type=str, default="", help="data.gov.in resource_id (UUID-like)")
    parser.add_argument("--ogd-limit", type=int, default=100, help="Page size for data.gov.in API")
    parser.add_argument("--ogd-max-records", type=int, default=5000, help="Max records to fetch total")
    parser.add_argument(
        "--ogd-param",
        action="append",
        default=[],
        help="Extra query param, repeatable. Format: key=value (example: filters[state]=Delhi)",
    )

    parser.add_argument("--fetch-estat", action="store_true", help="Fetch e-Stat (Japan) table data into data/raw")
    parser.add_argument("--estat-stats-data-id", type=str, default="", help="e-Stat statsDataId (table ID)")
    parser.add_argument("--estat-lang", type=str, default="E", help="e-Stat response language: E or J")
    parser.add_argument("--estat-limit", type=int, default=10000, help="Max rows requested from e-Stat")
    parser.add_argument("--estat-start-position", type=int, default=1, help="Start position for e-Stat data")

    args = parser.parse_args()

    print(BANNER)
    check_python()

    if args.provenance_only:
        run_provenance_only()
        print("\nDone.")
        return

    if args.pre_rag_smoke_check:
        run_pre_rag_smoke_check()
        print("\nDone.")
        return

    if args.scrape_live:
        run_scrape_live()
        print("\nDone.")
        return

    if args.resolve_corpus_urls:
        run_resolve_corpus_urls(use_gdelt=not args.no_gdelt, gdelt_per_row=args.gdelt_per_row)
        print("\nDone.")
        return

    if args.fetch_news:
        # Ensure .env is loaded before instantiating the fetcher (it reads env vars at init time)
        from utils.config import get_external_api_status
        from scrapers.external_signals import ExternalSignalsConfig, ExternalSignalsFetcher

        status = get_external_api_status()

        # Build config from CLI
        cfg = ExternalSignalsConfig(
            query=args.external_query,
            language=args.external_language,
            max_per_provider=int(args.external_max),
            timeframe_hours=int(args.external_timeframe_hours),
            keywords=[k.strip() for k in str(args.external_keywords).split(",") if k.strip()],
            categories=[c.strip() for c in str(args.external_categories).split(",") if c.strip()],
            countries=[c.strip() for c in str(args.external_countries).split(",") if c.strip()],
        )

        fetcher = ExternalSignalsFetcher()

        print("\n[EXTERNAL] Fetching external signals (news + GDELT)...")
        print("-" * 70)
        configured = [k for k, v in status.items() if v and k.endswith("_configured")]
        print(f"  Providers configured: {', '.join(configured) if configured else 'none (GDELT only)'}")
        result = fetcher.fetch_all(cfg)
        report = result.get("report", {})
        outputs = report.get("outputs", {})
        print(f"  Total items: {report.get('total_items', 0)}")
        print(f"  Output report: {outputs.get('report_json')}")
        if outputs.get("signals_csv"):
            print(f"  Output CSV: {outputs.get('signals_csv')}")
        print("\nDone.")
        return

    if args.build_official_corpus:
        run_build_official_corpus(
            start_year=args.corpus_start_year,
            end_year=args.corpus_end_year,
            max_docs=args.corpus_max_docs,
            max_urls_per_year=args.corpus_max_urls_per_year,
            promote_to_canonical=args.promote_corpus,
        )
        print("\nDone.")
        return

    if args.build_india_france_corpus:
        run_build_india_france_corpus(
            start_year=args.corpus_start_year,
            end_year=args.corpus_end_year,
            max_docs=args.corpus_max_docs,
            max_urls_per_year=args.corpus_max_urls_per_year,
            min_content_chars=args.corpus_min_content_chars,
        )
        print("\nDone.")
        return

    # Structured ingestion modes
    if args.import_comtrade:
        run_import_comtrade(args.import_comtrade)
        print("\nDone.")
        return

    if args.fetch_comtrade:
        run_fetch_comtrade(
            reporter_code=args.comtrade_reporter,
            partner_code=args.comtrade_partner,
            flow_code=args.comtrade_flow,
            period=args.comtrade_period,
            cmd_code=args.comtrade_cmd,
            key_preference=args.comtrade_key,
        )
        print("\nDone.")
        return

    if args.fetch_rss:
        urls = [u.strip() for u in str(args.rss_urls).split(",") if u.strip()]
        if not urls:
            urls = DEFAULT_DIPLOMACY_RSS_FEEDS
            print("\n[RSS] No --rss-urls provided; using resilient diplomacy feed defaults:")
            for u in urls:
                print(f"  - {u}")
        run_fetch_rss(urls=urls, since_days=args.rss_since_days, max_items=args.rss_max_items)
        print("\nDone.")
        return

    if args.summarize_rss:
        run_summarize_rss(args.summarize_rss)
        print("\nDone.")
        return

    if args.backfill_urls:
        run_url_backfill(min_confidence=args.backfill_min_confidence)
        print("\nDone.")
        return

    if args.assist_backfill_urls:
        run_assisted_url_backfill(
            review_csv=args.assist_backfill_review_csv,
            min_confidence=args.assist_backfill_min_confidence,
            max_rows=args.assist_backfill_max_rows,
        )
        print("\nDone.")
        return

    if args.curated_backfill_urls:
        run_curated_url_backfill(
            review_csv=args.curated_backfill_review_csv,
            min_confidence=args.curated_backfill_min_confidence,
            max_rows=args.curated_backfill_max_rows,
        )
        print("\nDone.")
        return

    if args.build_semi_auto_approval:
        run_build_semi_auto_approval(
            curated_review_csv=args.semi_auto_curated_review_csv,
            medium_min_confidence=args.semi_auto_min_confidence,
            recommend_approve_confidence=args.semi_auto_recommend_approve,
        )
        print("\nDone.")
        return

    if args.apply_semi_auto_approved:
        run_apply_semi_auto_approved_patch(approval_csv=args.apply_semi_auto_approved)
        print("\nDone.")
        return

    if args.fetch_ogd:
        if not args.ogd_resource_id:
            print("\n[OGD] Missing --ogd-resource-id")
            sys.exit(1)

        extra = []
        for item in list(args.ogd_param or []):
            if "=" not in str(item):
                continue
            k, v = str(item).split("=", 1)
            k = k.strip()
            v = v.strip()
            if k:
                extra.append((k, v))

        print("\n[OGD] Fetching data.gov.in dataset...")
        print("-" * 70)
        try:
            from scrapers.data_gov_in_ingestion import OGDQuery, fetch_and_write

            q = OGDQuery(
                resource_id=str(args.ogd_resource_id),
                limit=int(args.ogd_limit),
                max_records=int(args.ogd_max_records),
                extra_params=tuple(extra),
            )
            report = fetch_and_write(q, tag="ogd")
            outputs = report.get("outputs", {})
            print(f"  Rows: {report.get('rows', 0)}")
            print(f"  CSV: {outputs.get('records_csv')}")
            print(f"  Report: {outputs.get('report_json')}")
        except Exception as e:
            print(f"  OGD fetch failed: {e}")
            sys.exit(1)

        print("\nDone.")
        return

    if args.fetch_estat:
        if not args.estat_stats_data_id:
            print("\n[e-Stat] Missing --estat-stats-data-id")
            sys.exit(1)

        print("\n[e-Stat] Fetching dataset...")
        print("-" * 70)
        try:
            from scrapers.estat_ingestion import EStatQuery, fetch_and_write

            q = EStatQuery(
                stats_data_id=str(args.estat_stats_data_id),
                lang=str(args.estat_lang),
                limit=int(args.estat_limit),
                start_position=int(args.estat_start_position),
            )
            report = fetch_and_write(q, tag="estat")
            outputs = report.get("outputs", {})
            print(f"  Rows: {report.get('rows', 0)}")
            print(f"  CSV: {outputs.get('rows_csv')}")
            print(f"  Report: {outputs.get('report_json')}")
        except Exception as e:
            print(f"  e-Stat fetch failed: {e}")
            sys.exit(1)

        print("\nDone.")
        return

    check_core_deps()

    if args.api:
        run_api()
    elif args.full:
        run_pipeline()
        # Start API in background, dashboard in foreground
        import threading
        api_thread = threading.Thread(target=run_api, daemon=True)
        api_thread.start()
        run_dashboard()
    elif args.ingest:
        run_ingest()
    elif args.evaluate:
        run_evaluate()
    elif args.digest:
        run_digest()
    elif args.cron:
        run_cron()
    elif args.benchmark:
        run_benchmark()
    else:
        # Default: classic mode (pipeline → dashboard)
        run_pipeline()
        run_dashboard()

    print("\nDone.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
