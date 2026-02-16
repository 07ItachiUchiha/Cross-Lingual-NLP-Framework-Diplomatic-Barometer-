"""
Hybrid Search Engine — BM25 (Keyword) + Vector (Semantic)
---------------------------------------------------------
Why both?
  - A diplomat searching "Nuclear"  → wants the EXACT word  (BM25 / keyword)
  - A diplomat searching "Strategic deterrent" → wants the CONCEPT (Vector / cosine)

Implementation:
  1. BM25 via rank_bm25 (Okapi BM25 in pure Python)
  2. Vector via ChromaDB cosine similarity
  3. Results fused using Reciprocal Rank Fusion (RRF)

RRF formula:  score(d) = Σ  1 / (k + rank_i(d))   where k = 60 (constant)
"""

import logging
import math
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BM25Index:
    """
    Lightweight Okapi BM25 implementation.

    Falls back to a pure-Python implementation if ``rank_bm25`` is not
    installed — keeping the dependency optional.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_tokens: List[List[str]] = []
        self.doc_ids: List[str] = []
        self.avgdl: float = 0.0
        self.idf: Dict[str, float] = {}
        self.doc_freq: Dict[str, int] = defaultdict(int)
        self.N = 0

    @staticmethod
    def tokenize(text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

    def fit(self, doc_ids: List[str], documents: List[str]):
        """Build the BM25 index from raw texts."""
        self.doc_ids = doc_ids
        self.corpus_tokens = [self.tokenize(d) for d in documents]
        self.N = len(self.corpus_tokens)
        self.avgdl = sum(len(t) for t in self.corpus_tokens) / max(self.N, 1)

        # Document frequencies
        self.doc_freq = defaultdict(int)
        for tokens in self.corpus_tokens:
            seen = set(tokens)
            for tok in seen:
                self.doc_freq[tok] += 1

        # IDF (with +0.5 smoothing to avoid negatives)
        self.idf = {}
        for tok, df in self.doc_freq.items():
            self.idf[tok] = math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)

        logger.info(f"BM25 index built: {self.N} docs, {len(self.idf)} unique terms")

    def score(self, query: str) -> List[Tuple[str, float]]:
        """Return (doc_id, bm25_score) pairs sorted descending."""
        query_tokens = self.tokenize(query)
        scores = []

        for i, doc_tokens in enumerate(self.corpus_tokens):
            s = 0.0
            dl = len(doc_tokens)
            freq = defaultdict(int)
            for t in doc_tokens:
                freq[t] += 1

            for qt in query_tokens:
                if qt not in self.idf:
                    continue
                tf = freq.get(qt, 0)
                numerator = self.idf[qt] * tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1))
                s += numerator / denominator

            scores.append((self.doc_ids[i], s))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores


class HybridSearchEngine:
    """
    Combines BM25 (keyword) and vector (semantic) retrieval with
    Reciprocal Rank Fusion.

    Parameters
    ----------
    vector_store : VectorStore
        The ChromaDB-backed store.
    embedding_manager : EmbeddingManager
        For encoding queries.
    rrf_k : int
        Constant for RRF (default 60 — standard in IR literature).
    keyword_weight : float
        Weight for BM25 leg in [0, 1].  Default 0.4.
    vector_weight : float
        Weight for vector leg in [0, 1].  Default 0.6.
    """

    def __init__(
        self,
        vector_store,
        embedding_manager,
        rrf_k: int = 60,
        keyword_weight: float = 0.4,
        vector_weight: float = 0.6,
    ):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager
        self.rrf_k = rrf_k
        self.keyword_weight = keyword_weight
        self.vector_weight = vector_weight
        self.bm25 = BM25Index()
        self._index_built = False

    def build_bm25_index(self, doc_ids: List[str], documents: List[str]):
        """Build the keyword index.  Call once after ingesting data."""
        self.bm25.fit(doc_ids, documents)
        self._index_built = True

    # ── search ───────────────────────────────────────────────────────
    def search(
        self,
        query: str,
        n_results: int = 10,
        where: Optional[Dict] = None,
        mode: str = "hybrid",
    ) -> List[Dict]:
        """
        Execute a search.

        Parameters
        ----------
        query : str
        n_results : int
        where : dict   ChromaDB metadata filter.
        mode : str      "hybrid" | "keyword" | "vector"

        Returns
        -------
        list[dict] each with keys: chunk_id, text, score, metadata, source_type
        """
        results_map: Dict[str, Dict] = {}

        # ── Keyword leg (BM25) ───────────────────────────────────────
        if mode in ("hybrid", "keyword") and self._index_built:
            bm25_results = self.bm25.score(query)
            for rank, (doc_id, bm25_score) in enumerate(bm25_results[:n_results * 2]):
                rrf_score = self.keyword_weight / (self.rrf_k + rank + 1)
                if doc_id not in results_map:
                    results_map[doc_id] = {
                        "chunk_id": doc_id,
                        "score": 0.0,
                        "bm25_score": bm25_score,
                        "vector_score": 0.0,
                        "source_type": "keyword",
                    }
                results_map[doc_id]["score"] += rrf_score
                results_map[doc_id]["bm25_score"] = bm25_score

        # ── Vector leg (cosine) ──────────────────────────────────────
        if mode in ("hybrid", "vector"):
            query_vec = self.embedding_manager.embed_query(query)
            vec_results = self.vector_store.query(
                query_embedding=query_vec,
                n_results=n_results * 2,
                where=where,
            )
            for rank, (doc_id, text, meta, dist) in enumerate(zip(
                vec_results["ids"],
                vec_results["documents"],
                vec_results["metadatas"],
                vec_results["distances"],
            )):
                rrf_score = self.vector_weight / (self.rrf_k + rank + 1)
                cosine_sim = 1 - dist
                if doc_id not in results_map:
                    results_map[doc_id] = {
                        "chunk_id": doc_id,
                        "score": 0.0,
                        "bm25_score": 0.0,
                        "vector_score": cosine_sim,
                        "source_type": "vector",
                    }
                else:
                    results_map[doc_id]["source_type"] = "hybrid"
                results_map[doc_id]["score"] += rrf_score
                results_map[doc_id]["vector_score"] = cosine_sim
                results_map[doc_id]["text"] = text
                results_map[doc_id]["metadata"] = meta

        # ── Sort by fused score ──────────────────────────────────────
        ranked = sorted(results_map.values(), key=lambda x: x["score"], reverse=True)
        return ranked[:n_results]

    def get_info(self) -> Dict:
        return {
            "bm25_indexed": self._index_built,
            "bm25_docs": self.bm25.N,
            "vector_chunks": self.vector_store.count(),
            "rrf_k": self.rrf_k,
            "keyword_weight": self.keyword_weight,
            "vector_weight": self.vector_weight,
        }
