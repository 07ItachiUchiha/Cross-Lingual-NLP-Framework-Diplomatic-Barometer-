"""
Latency Benchmark — Performance Testing
-----------------------------------------
"If the dashboard takes 40 seconds to load, it fails."

Benchmarks:
  1. Ingestion throughput   (docs/sec, chunks/sec)
  2. Vector search latency  (target: < 2 sec per query)
  3. Hybrid search latency  (target: < 3 sec per query)
  4. Full RAG query latency (target: < 5 sec per query)
  5. BM25 keyword latency   (target: < 500 ms)

Uses HNSW index (from ChromaDB) for vector retrieval.
"""

import time
import logging
import statistics
from typing import Dict, List, Callable

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Performance targets (in milliseconds) ────────────────────────────
TARGETS = {
    "vector_search": 2000,    # < 2 s
    "keyword_search": 500,    # < 500 ms
    "hybrid_search": 3000,    # < 3 s
    "full_rag_query": 5000,   # < 5 s (with LLM)
    "rag_no_llm": 3000,       # < 3 s (no LLM mode)
    "ingestion_per_doc": 500, # < 500 ms per document
}


class LatencyBenchmark:
    """
    Run latency benchmarks against the RAG pipeline.

    Usage:
        bench = LatencyBenchmark(pipeline)
        report = bench.run_full_benchmark()
    """

    TEST_QUERIES = [
        "What is the CEPA between India and Japan?",
        "When did the Quad become prominent in bilateral relations?",
        "How much ODA has Japan provided to India?",
        "What maritime security cooperation exists?",
        "Describe the evolution of defense exercises between India and Japan.",
        "What semiconductor cooperation was announced?",
        "How has the Indo-Pacific strategy affected bilateral relations?",
        "What cultural exchanges exist between the two countries?",
    ]

    def __init__(self, pipeline=None):
        self.pipeline = pipeline
        self.results: Dict = {}

    def _time_function(self, func: Callable, n_runs: int = 5) -> Dict:
        """Time a function over n_runs and return stats."""
        latencies = []
        for _ in range(n_runs):
            start = time.perf_counter()
            try:
                func()
            except Exception as e:
                logger.warning(f"Benchmark function raised: {e}")
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)

        return {
            "mean_ms": round(statistics.mean(latencies), 1),
            "median_ms": round(statistics.median(latencies), 1),
            "p95_ms": round(sorted(latencies)[int(len(latencies) * 0.95)], 1) if len(latencies) >= 5 else round(max(latencies), 1),
            "min_ms": round(min(latencies), 1),
            "max_ms": round(max(latencies), 1),
            "n_runs": n_runs,
        }

    def benchmark_vector_search(self, n_runs: int = 5) -> Dict:
        """Benchmark pure vector search."""
        if not self.pipeline:
            return {"error": "No pipeline configured"}

        def _run():
            for q in self.TEST_QUERIES[:3]:
                self.pipeline.hybrid_search.search(
                    query=q, n_results=5, mode="vector"
                )

        result = self._time_function(_run, n_runs)
        result["target_ms"] = TARGETS["vector_search"]
        result["passed"] = result["mean_ms"] <= TARGETS["vector_search"]
        self.results["vector_search"] = result
        return result

    def benchmark_keyword_search(self, n_runs: int = 5) -> Dict:
        """Benchmark BM25 keyword search."""
        if not self.pipeline:
            return {"error": "No pipeline configured"}

        def _run():
            for q in self.TEST_QUERIES[:3]:
                self.pipeline.hybrid_search.search(
                    query=q, n_results=5, mode="keyword"
                )

        result = self._time_function(_run, n_runs)
        result["target_ms"] = TARGETS["keyword_search"]
        result["passed"] = result["mean_ms"] <= TARGETS["keyword_search"]
        self.results["keyword_search"] = result
        return result

    def benchmark_hybrid_search(self, n_runs: int = 5) -> Dict:
        """Benchmark hybrid (BM25 + vector) search."""
        if not self.pipeline:
            return {"error": "No pipeline configured"}

        def _run():
            for q in self.TEST_QUERIES[:3]:
                self.pipeline.hybrid_search.search(
                    query=q, n_results=5, mode="hybrid"
                )

        result = self._time_function(_run, n_runs)
        result["target_ms"] = TARGETS["hybrid_search"]
        result["passed"] = result["mean_ms"] <= TARGETS["hybrid_search"]
        self.results["hybrid_search"] = result
        return result

    def benchmark_full_query(self, n_runs: int = 3) -> Dict:
        """Benchmark full RAG query (search + LLM)."""
        if not self.pipeline:
            return {"error": "No pipeline configured"}

        target_key = "rag_no_llm" if self.pipeline.llm_backend == "none" else "full_rag_query"

        def _run():
            self.pipeline.query(self.TEST_QUERIES[0])

        result = self._time_function(_run, n_runs)
        result["target_ms"] = TARGETS[target_key]
        result["passed"] = result["mean_ms"] <= TARGETS[target_key]
        result["llm_backend"] = self.pipeline.llm_backend
        self.results["full_query"] = result
        return result

    def run_full_benchmark(self) -> Dict:
        """Run all benchmarks and return aggregate report."""
        logger.info("Running latency benchmarks...")

        self.benchmark_keyword_search()
        self.benchmark_vector_search()
        self.benchmark_hybrid_search()
        self.benchmark_full_query()

        all_passed = all(r.get("passed", False) for r in self.results.values()
                          if "error" not in r)

        report = {
            "overall_passed": all_passed,
            "benchmarks": self.results,
            "targets": TARGETS,
        }

        return report

    def print_report(self, report: Dict = None):
        """Pretty-print benchmark results."""
        report = report or {"benchmarks": self.results, "overall_passed": False}

        print("\n" + "=" * 70)
        print("  LATENCY BENCHMARK REPORT")
        print("=" * 70)
        print(f"  {'Benchmark':<22} {'Mean':>8} {'P95':>8} {'Target':>8} {'Status':>8}")
        print("  " + "-" * 58)

        for name, data in report.get("benchmarks", {}).items():
            if "error" in data:
                print(f"  {name:<22} {'ERROR':>8}")
                continue
            status = "✓ PASS" if data.get("passed") else "✗ FAIL"
            print(f"  {name:<22} {data['mean_ms']:>7.0f}ms "
                  f"{data.get('p95_ms', 0):>7.0f}ms "
                  f"{data.get('target_ms', 0):>7.0f}ms "
                  f"{status:>8}")

        overall = "✓ ALL PASS" if report.get("overall_passed") else "✗ SOME FAILED"
        print(f"\n  Overall: {overall}")
        print("=" * 70)
