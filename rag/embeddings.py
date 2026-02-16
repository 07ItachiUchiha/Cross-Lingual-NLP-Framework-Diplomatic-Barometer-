"""
Embedding Manager
-----------------
Generate vector embeddings for document chunks.

Supports multiple backends:
  1. OpenAI  text-embedding-3-small  (1536-d, best quality)
  2. HuggingFace  BGE-M3             (1024-d, free, local)
  3. Sentence-Transformers            (384-d, lightweight fallback)

The active backend is selected via environment variable
``EMBEDDING_BACKEND`` (openai | huggingface | sentence-transformers).
Defaults to sentence-transformers for zero-cost local dev.
"""

import os
import logging
from typing import List, Optional

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Dimension lookup
EMBEDDING_DIMS = {
    "openai": 1536,
    "huggingface": 1024,
    "sentence-transformers": 384,
}


class EmbeddingManager:
    """Unified interface for embedding generation."""

    def __init__(self, backend: Optional[str] = None):
        self.backend = (backend or os.getenv("EMBEDDING_BACKEND", "sentence-transformers")).lower()
        self.model = None
        self.dimension = EMBEDDING_DIMS.get(self.backend, 384)
        self._load_model()

    # ── model loading ────────────────────────────────────────────────
    def _load_model(self):
        if self.backend == "openai":
            self._init_openai()
        elif self.backend == "huggingface":
            self._init_huggingface()
        else:
            self._init_sentence_transformers()

    def _init_openai(self):
        """OpenAI text-embedding-3-small — requires OPENAI_API_KEY."""
        try:
            import openai
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable is not set")
            self.client = openai.OpenAI(api_key=api_key)
            self.model_name = "text-embedding-3-small"
            self.dimension = 1536
            logger.info(f"OpenAI embedding model ready ({self.model_name}, {self.dimension}-d)")
        except ImportError:
            logger.warning("openai package not installed — falling back to sentence-transformers")
            self.backend = "sentence-transformers"
            self._init_sentence_transformers()

    def _init_huggingface(self):
        """HuggingFace BGE-M3 — runs locally, no API key needed."""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer("BAAI/bge-m3")
            self.model_name = "BAAI/bge-m3"
            self.dimension = 1024
            logger.info(f"HuggingFace model ready ({self.model_name}, {self.dimension}-d)")
        except ImportError:
            logger.warning("sentence-transformers not installed — falling back")
            self.backend = "sentence-transformers"
            self._init_sentence_transformers()

    def _init_sentence_transformers(self):
        """Lightweight local model — zero cost fallback."""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            self.model_name = "all-MiniLM-L6-v2"
            self.dimension = 384
            logger.info(f"SentenceTransformers model ready ({self.model_name}, {self.dimension}-d)")
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. "
                "Embeddings will be random vectors (dev mode only)."
            )
            self.model = None
            self.model_name = "random-fallback"
            self.dimension = 384

    # ── embedding generation ─────────────────────────────────────────
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.

        Returns
        -------
        np.ndarray  shape (len(texts), dimension)
        """
        if not texts:
            return np.empty((0, self.dimension))

        if self.backend == "openai":
            return self._embed_openai(texts)
        elif self.model is not None:
            return self._embed_local(texts)
        else:
            # Random fallback for dev
            logger.warning("Using random embeddings — install sentence-transformers for real vectors")
            return np.random.randn(len(texts), self.dimension).astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string. Returns shape (dimension,)."""
        return self.embed_texts([query])[0]

    # ── backend-specific ─────────────────────────────────────────────
    def _embed_openai(self, texts: List[str]) -> np.ndarray:
        response = self.client.embeddings.create(
            model=self.model_name,
            input=texts,
        )
        embeddings = [item.embedding for item in response.data]
        return np.array(embeddings, dtype=np.float32)

    def _embed_local(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

    # ── utility ──────────────────────────────────────────────────────
    def get_info(self) -> dict:
        return {
            "backend": self.backend,
            "model": self.model_name,
            "dimension": self.dimension,
        }


# ── CLI ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    em = EmbeddingManager()
    print(em.get_info())
    vecs = em.embed_texts(["India-Japan trade partnership", "Maritime security in Indo-Pacific"])
    print(f"Shape: {vecs.shape}")
    # Cosine similarity
    cos_sim = np.dot(vecs[0], vecs[1]) / (np.linalg.norm(vecs[0]) * np.linalg.norm(vecs[1]))
    print(f"Cosine similarity: {cos_sim:.4f}")
