#!/usr/bin/env python
"""
Geopolitical Insight Engine — Unified Launcher
───────────────────────────────────────────────
Modes:
  python run.py                  # Classic: pipeline → Streamlit dashboard
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


def check_python():
    if sys.version_info < (3, 9):
        print("Error: Python 3.9+ required")
        sys.exit(1)
    print(f"  Python {sys.version_info.major}.{sys.version_info.minor} detected")


def check_core_deps():
    """Verify minimum dependencies are installed."""
    missing = []
    for mod in ["streamlit", "pandas", "plotly", "spacy", "fastapi"]:
        try:
            __import__(mod)
        except ImportError:
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
    parser.add_argument("--full", action="store_true", help="Pipeline + API + Dashboard")
    parser.add_argument("--ingest", action="store_true", help="Ingest documents into RAG")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation suite")
    parser.add_argument("--digest", action="store_true", help="Generate weekly digest")
    parser.add_argument("--cron", action="store_true", help="Run cron updater")
    parser.add_argument("--benchmark", action="store_true", help="Run latency benchmarks")
    args = parser.parse_args()

    print(BANNER)
    check_python()
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
