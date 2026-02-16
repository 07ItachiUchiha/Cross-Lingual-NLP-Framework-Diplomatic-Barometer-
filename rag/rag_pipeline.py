"""
RAG Pipeline — Full Orchestration
-----------------------------------
Chains: User Query → Hybrid Search → Context Assembly → LLM → Citation Validation

Supports:
  - OpenAI GPT-4o  (via API key)
  - Ollama Llama-3  (local, free)
  - "No-LLM" mode  (returns raw retrieved context — for testing / CI)

This is the "Semantic Engine" described in the SDLC.
"""

import os
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    End-to-end Retrieval-Augmented Generation for diplomatic analysis.

    Workflow:
      1. Ingest  → chunk documents, embed, store in ChromaDB
      2. Query   → hybrid search, build prompt, call LLM, validate citations

    Parameters
    ----------
    llm_backend : str
        "openai" | "ollama" | "none"  (default: "none" for safe CI)
    embedding_backend : str
        "openai" | "huggingface" | "sentence-transformers"
    chunk_size : int
    chunk_overlap : int
    """

    def __init__(
        self,
        llm_backend: Optional[str] = None,
        embedding_backend: Optional[str] = None,
        chunk_size: int = 1500,
        chunk_overlap: int = 150,
        persist_dir: Optional[str] = None,
    ):
        from .chunker import DocumentChunker
        from .embeddings import EmbeddingManager
        from .vector_store import VectorStore
        from .hybrid_search import HybridSearchEngine
        from .citation_layer import CitationLayer

        self.llm_backend = (llm_backend or os.getenv("LLM_BACKEND", "none")).lower()

        # Components
        self.chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.embedding_mgr = EmbeddingManager(backend=embedding_backend)

        _persist = persist_dir or str(Path(__file__).parent.parent / "data" / "vector_db")
        self.vector_store = VectorStore(
            persist_directory=_persist,
            embedding_dimension=self.embedding_mgr.dimension,
        )
        self.hybrid_search = HybridSearchEngine(
            vector_store=self.vector_store,
            embedding_manager=self.embedding_mgr,
        )
        self.citation_layer = CitationLayer()

        # LLM client (lazy init)
        self._llm_client = None

        logger.info(
            f"RAGPipeline initialised — LLM: {self.llm_backend}, "
            f"Embeddings: {self.embedding_mgr.backend} ({self.embedding_mgr.dimension}-d)"
        )

    # ══════════════════════════════════════════════════════════════════
    #  INGEST
    # ══════════════════════════════════════════════════════════════════
    def ingest_dataframe(self, df: pd.DataFrame, text_col: str = "content") -> Dict:
        """
        Chunk, embed, and store an entire DataFrame of documents.

        Returns summary stats.
        """
        t0 = time.time()

        # 1. Chunk
        chunks = self.chunker.chunk_dataframe(df, text_col=text_col)
        if not chunks:
            return {"status": "error", "message": "No chunks produced"}

        # 2. Embed
        texts = [c.text for c in chunks]
        embeddings = self.embedding_mgr.embed_texts(texts)

        # 3. Store
        ids = [c.chunk_id for c in chunks]
        metadatas = [c.to_dict() for c in chunks]
        # Remove 'text' from metadata to avoid duplication
        for m in metadatas:
            m.pop("text", None)

        self.vector_store.add_chunks(ids, embeddings, texts, metadatas)

        # 4. Build BM25 index
        self.hybrid_search.build_bm25_index(ids, texts)

        elapsed = time.time() - t0
        stats = {
            "status": "success",
            "documents": len(df),
            "chunks": len(chunks),
            "embedding_dim": self.embedding_mgr.dimension,
            "vector_store_total": self.vector_store.count(),
            "elapsed_seconds": round(elapsed, 2),
        }
        logger.info(f"Ingestion complete: {stats}")
        return stats

    # ══════════════════════════════════════════════════════════════════
    #  QUERY
    # ══════════════════════════════════════════════════════════════════
    def query(
        self,
        question: str,
        n_results: int = 5,
        search_mode: str = "hybrid",
        year_filter: Optional[int] = None,
        source_filter: Optional[str] = None,
    ) -> Dict:
        """
        Answer a question using RAG.

        Parameters
        ----------
        question : str
        n_results : int       Top-k chunks to retrieve.
        search_mode : str     "hybrid" | "keyword" | "vector"
        year_filter : int     Optional — restrict to a specific year.
        source_filter : str   Optional — "MEA" or "MOFA".

        Returns
        -------
        dict with: answer, citations, confidence, search_results, latency_ms
        """
        t0 = time.time()

        # Build metadata filter for ChromaDB
        where = {}
        if year_filter:
            where["year"] = year_filter
        if source_filter:
            where["source"] = source_filter
        where_clause = where if where else None

        # 1. Retrieve
        search_results = self.hybrid_search.search(
            query=question,
            n_results=n_results,
            where=where_clause,
            mode=search_mode,
        )

        # 2. Build grounded prompt
        prompt = self.citation_layer.build_grounded_prompt(question, search_results)

        # 3. Call LLM (or return raw context in no-LLM mode)
        if self.llm_backend == "none":
            raw_answer = self._format_context_only(search_results)
        else:
            raw_answer = self._call_llm(prompt)

        # 4. Validate citations
        cited_answer = self.citation_layer.validate_and_cite(
            raw_answer, question, search_results
        )

        latency = round((time.time() - t0) * 1000, 1)

        return {
            "answer": cited_answer.answer,
            "citations": [
                {
                    "chunk_id": c.chunk_id,
                    "doc_title": c.doc_title,
                    "year": c.year,
                    "source": c.source,
                    "relevance": c.relevance_score,
                    "excerpt": c.excerpt,
                }
                for c in cited_answer.citations
            ],
            "confidence": cited_answer.confidence,
            "is_grounded": cited_answer.is_grounded,
            "warning": cited_answer.warning,
            "search_mode": search_mode,
            "n_results_retrieved": len(search_results),
            "latency_ms": latency,
        }

    # ── LLM call ─────────────────────────────────────────────────────
    def _call_llm(self, prompt: str) -> str:
        """Dispatch to the configured LLM backend."""
        if self.llm_backend == "openai":
            return self._call_openai(prompt)
        elif self.llm_backend == "ollama":
            return self._call_ollama(prompt)
        else:
            return "LLM backend not configured."

    def _call_openai(self, prompt: str) -> str:
        try:
            import openai
            if self._llm_client is None:
                self._llm_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            response = self._llm_client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o"),
                messages=[
                    {"role": "system", "content": "You are a diplomatic document analyst."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=1024,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI call failed: {e}")
            return f"LLM error: {e}"

    def _call_ollama(self, prompt: str) -> str:
        try:
            import requests
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            model = os.getenv("OLLAMA_MODEL", "llama3")
            resp = requests.post(
                f"{base_url}/api/generate",
                json={"model": model, "prompt": prompt, "stream": False},
                timeout=120,
            )
            resp.raise_for_status()
            return resp.json().get("response", "No response from Ollama.")
        except Exception as e:
            logger.error(f"Ollama call failed: {e}")
            return f"LLM error: {e}"

    def _format_context_only(self, results: List[Dict]) -> str:
        """No-LLM mode — returns retrieved context as a formatted answer."""
        if not results:
            return "No relevant documents found."

        lines = ["Based on the diplomatic corpus, the following relevant passages were found:\n"]
        for i, r in enumerate(results, 1):
            meta = r.get("metadata", {})
            title = meta.get("doc_title", "Unknown")
            year = meta.get("year", "N/A")
            source = meta.get("source", "N/A")
            text = r.get("text", "")[:300]
            lines.append(f"[{i}] {title} ({year}, {source})")
            lines.append(f"    {text}...")
            lines.append("")
        return "\n".join(lines)

    # ── classify (for golden dataset evaluation) ─────────────────────
    def classify_text(self, text: str) -> str:
        """
        Classify a paragraph as Economic / Security / Cultural using RAG.
        Used for benchmarking against the golden dataset.
        """
        prompt = f"""Classify the following diplomatic paragraph into EXACTLY ONE of these categories:
- Economic
- Security
- Cultural

Respond with ONLY the category name, nothing else.

Paragraph: {text}

Category:"""

        if self.llm_backend == "none":
            # Fallback: keyword-based classification
            return self._keyword_classify(text)
        return self._call_llm(prompt).strip()

    @staticmethod
    def _keyword_classify(text: str) -> str:
        """Simple keyword classifier for no-LLM mode."""
        text_lower = text.lower()
        econ_kw = {"trade", "investment", "oda", "loan", "infrastructure", "economic",
                    "cepa", "yen", "manufacturing", "tariff", "customs", "export",
                    "import", "startup", "semiconductor", "fintech", "currency"}
        sec_kw = {"security", "defense", "defence", "military", "maritime", "quad",
                   "indo-pacific", "terrorism", "missile", "naval", "cybersecurity",
                   "intelligence", "nuclear", "fighter", "amphibious", "surveillance"}
        cult_kw = {"cultural", "education", "scholarship", "tourism", "heritage",
                    "language", "film", "sports", "buddhist", "yoga", "exchange",
                    "festival", "sister-city", "cinema", "university"}

        econ_score = sum(1 for kw in econ_kw if kw in text_lower)
        sec_score = sum(1 for kw in sec_kw if kw in text_lower)
        cult_score = sum(1 for kw in cult_kw if kw in text_lower)

        if cult_score >= econ_score and cult_score >= sec_score and cult_score > 0:
            return "Cultural"
        elif sec_score >= econ_score:
            return "Security"
        return "Economic"

    # ── info ─────────────────────────────────────────────────────────
    def get_status(self) -> Dict:
        return {
            "llm_backend": self.llm_backend,
            "embedding": self.embedding_mgr.get_info(),
            "vector_store": self.vector_store.get_stats(),
            "hybrid_search": self.hybrid_search.get_info(),
        }


# ── CLI ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    pipe = RAGPipeline(llm_backend="none")
    print(pipe.get_status())
