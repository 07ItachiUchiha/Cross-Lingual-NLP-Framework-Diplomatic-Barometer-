"""
RAGAS-style Evaluation Framework
----------------------------------
Adapted from the RAGAS (Retrieval Augmented Generation Assessment) methodology.

Metrics:
  1. Context Recall    — Did we retrieve the right documents?
  2. Context Precision — Are the retrieved docs actually relevant?
  3. Faithfulness      — Did the LLM stick to the facts in context?
  4. Answer Relevance  — Is the answer actually addressing the question?

Reference: https://docs.ragas.io/
"""

import re
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGASEvaluator:
    """
    Evaluate RAG pipeline quality using RAGAS-style metrics.

    Usage:
        evaluator = RAGASEvaluator()
        results = evaluator.evaluate_batch(test_cases)
    """

    def __init__(self):
        self.results: List[Dict] = []

    # ── Test case format ─────────────────────────────────────────────
    # Each test case is a dict:
    # {
    #   "question":          str,
    #   "expected_answer":   str,           # ground truth
    #   "expected_doc_ids":  list[str],     # docs that SHOULD be retrieved
    #   "retrieved_chunks":  list[dict],    # actual retrieval results
    #   "generated_answer":  str,           # LLM output
    # }

    # ── Context Recall ───────────────────────────────────────────────
    @staticmethod
    def context_recall(
        expected_doc_ids: List[str],
        retrieved_chunk_ids: List[str],
    ) -> float:
        """
        What fraction of the expected documents were actually retrieved?

        context_recall = |expected ∩ retrieved| / |expected|
        """
        if not expected_doc_ids:
            return 1.0  # nothing expected → trivially complete

        # Match on doc_id prefix (chunks have _chunk_NNN suffix)
        retrieved_docs = {cid.rsplit("_chunk_", 1)[0] for cid in retrieved_chunk_ids}
        expected_set = set(expected_doc_ids)

        hits = expected_set & retrieved_docs
        return len(hits) / len(expected_set)

    # ── Context Precision ────────────────────────────────────────────
    @staticmethod
    def context_precision(
        expected_doc_ids: List[str],
        retrieved_chunk_ids: List[str],
    ) -> float:
        """
        What fraction of retrieved chunks come from expected documents?

        context_precision = |expected ∩ retrieved| / |retrieved|
        """
        if not retrieved_chunk_ids:
            return 0.0

        retrieved_docs = {cid.rsplit("_chunk_", 1)[0] for cid in retrieved_chunk_ids}
        expected_set = set(expected_doc_ids)

        hits = expected_set & retrieved_docs
        return len(hits) / len(retrieved_docs)

    # ── Faithfulness ─────────────────────────────────────────────────
    @staticmethod
    def faithfulness(
        generated_answer: str,
        context_texts: List[str],
    ) -> float:
        """
        Are the claims in the answer supported by the context?

        Heuristic: split answer into sentences, check if key terms from
        each sentence appear in the context.

        Returns a score in [0, 1].
        """
        if not generated_answer.strip() or not context_texts:
            return 0.0

        combined_context = " ".join(context_texts).lower()

        # Split answer into sentences
        sentences = re.split(r'[.!?]+', generated_answer)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

        if not sentences:
            return 1.0  # no substantive claims → trivially faithful

        supported = 0
        for sentence in sentences:
            # Extract key terms (nouns/proper nouns approximated by title-case + long words)
            key_terms = re.findall(r'\b[A-Z][a-z]+\b|\b\w{5,}\b', sentence)
            if not key_terms:
                supported += 1
                continue

            # Count how many key terms appear in context
            matches = sum(1 for term in key_terms if term.lower() in combined_context)
            ratio = matches / len(key_terms)
            if ratio >= 0.5:
                supported += 1

        return supported / len(sentences)

    # ── Answer Relevance ─────────────────────────────────────────────
    @staticmethod
    def answer_relevance(
        question: str,
        generated_answer: str,
    ) -> float:
        """
        Does the answer address the question?

        Heuristic: overlap of question keywords in the answer.
        """
        if not generated_answer.strip():
            return 0.0

        q_tokens = set(re.findall(r'\b\w{3,}\b', question.lower()))
        # Remove stopwords
        stopwords = {"the", "what", "when", "where", "how", "does", "did", "was",
                      "were", "are", "has", "have", "will", "about", "between",
                      "from", "with", "that", "this", "their", "which", "been"}
        q_tokens -= stopwords

        if not q_tokens:
            return 1.0

        a_text = generated_answer.lower()
        matches = sum(1 for t in q_tokens if t in a_text)
        return matches / len(q_tokens)

    # ── Batch evaluation ─────────────────────────────────────────────
    def evaluate_single(self, test_case: Dict) -> Dict:
        """Evaluate one test case and return per-metric scores."""
        retrieved_ids = [
            c.get("chunk_id", "") for c in test_case.get("retrieved_chunks", [])
        ]
        context_texts = [
            c.get("text", "") for c in test_case.get("retrieved_chunks", [])
        ]

        metrics = {
            "question": test_case["question"],
            "context_recall": self.context_recall(
                test_case.get("expected_doc_ids", []),
                retrieved_ids,
            ),
            "context_precision": self.context_precision(
                test_case.get("expected_doc_ids", []),
                retrieved_ids,
            ),
            "faithfulness": self.faithfulness(
                test_case.get("generated_answer", ""),
                context_texts,
            ),
            "answer_relevance": self.answer_relevance(
                test_case["question"],
                test_case.get("generated_answer", ""),
            ),
        }

        # Composite score (simple average)
        metric_values = [v for k, v in metrics.items() if k != "question"]
        metrics["composite_score"] = round(np.mean(metric_values), 4)

        return metrics

    def evaluate_batch(self, test_cases: List[Dict]) -> Dict:
        """Evaluate multiple test cases and return aggregate report."""
        self.results = [self.evaluate_single(tc) for tc in test_cases]

        # Aggregate
        metric_names = ["context_recall", "context_precision", "faithfulness",
                         "answer_relevance", "composite_score"]
        aggregated = {}
        for m in metric_names:
            values = [r[m] for r in self.results]
            aggregated[m] = {
                "mean": round(np.mean(values), 4),
                "std": round(np.std(values), 4),
                "min": round(min(values), 4),
                "max": round(max(values), 4),
            }

        return {
            "n_test_cases": len(test_cases),
            "aggregated_metrics": aggregated,
            "per_question": self.results,
        }

    def print_report(self, report: Dict):
        """Pretty-print a RAGAS evaluation report."""
        print("\n" + "=" * 70)
        print("  RAGAS EVALUATION REPORT")
        print("=" * 70)
        agg = report["aggregated_metrics"]
        print(f"  Test cases: {report['n_test_cases']}")
        print()
        print(f"  {'Metric':<25} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
        print("  " + "-" * 60)
        for metric, vals in agg.items():
            print(f"  {metric:<25} {vals['mean']:>8.4f} {vals['std']:>8.4f} "
                  f"{vals['min']:>8.4f} {vals['max']:>8.4f}")
        print("=" * 70)


# ── Pre-built test cases for the India-Japan corpus ──────────────────
DIPLOMATIC_TEST_CASES = [
    {
        "question": "When did security language overtake economic language in India-Japan statements?",
        "expected_answer": "The crossover roughly happened around 2013-2018.",
        "expected_doc_ids": ["doc_0020", "doc_0025", "doc_0030"],
        "retrieved_chunks": [],
        "generated_answer": "",
    },
    {
        "question": "What is the CEPA between India and Japan?",
        "expected_answer": "The Comprehensive Economic Partnership Agreement covers trade in goods, services, investment, and IP.",
        "expected_doc_ids": ["doc_0005"],
        "retrieved_chunks": [],
        "generated_answer": "",
    },
    {
        "question": "How has the Quad evolved in India-Japan relations?",
        "expected_answer": "The Quad emerged prominently post-2020 with joint naval exercises and Indo-Pacific strategy coordination.",
        "expected_doc_ids": ["doc_0033", "doc_0034", "doc_0037"],
        "retrieved_chunks": [],
        "generated_answer": "",
    },
    {
        "question": "What defence cooperation agreements exist between India and Japan?",
        "expected_answer": "Key agreements include ACSA, 2+2 Dialogue, joint exercises, and defence technology co-development.",
        "expected_doc_ids": ["doc_0031", "doc_0030", "doc_0027"],
        "retrieved_chunks": [],
        "generated_answer": "",
    },
    {
        "question": "What role does ODA play in India-Japan economic relations?",
        "expected_answer": "Japan's ODA includes Yen loans for metro, highway, and port infrastructure projects in India.",
        "expected_doc_ids": ["doc_0000", "doc_0001", "doc_0003"],
        "retrieved_chunks": [],
        "generated_answer": "",
    },
]


if __name__ == "__main__":
    evaluator = RAGASEvaluator()
    # Demo with placeholder results
    for tc in DIPLOMATIC_TEST_CASES:
        tc["generated_answer"] = tc["expected_answer"]
        tc["retrieved_chunks"] = [{"chunk_id": did + "_chunk_000", "text": tc["expected_answer"]}
                                    for did in tc["expected_doc_ids"]]
    report = evaluator.evaluate_batch(DIPLOMATIC_TEST_CASES)
    evaluator.print_report(report)
