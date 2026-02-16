"""
Vector Store — ChromaDB Integration
------------------------------------
Stores document-chunk embeddings in a persistent ChromaDB collection.
Supports add / query / delete operations with full metadata filtering.

Why ChromaDB?
  - Self-hosted (Docker or in-process), no cloud dependency
  - HNSW index for < 2 s retrieval on 10 k chunks
  - Native metadata filtering (year, source, label)
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default persistence directory
DEFAULT_PERSIST_DIR = str(Path(__file__).parent.parent / "data" / "vector_db")


class VectorStore:
    """
    Thin wrapper around ChromaDB for the diplomatic corpus.

    Parameters
    ----------
    collection_name : str
        Name of the ChromaDB collection.
    persist_directory : str
        Disk path for persistent storage.
    embedding_dimension : int
        Expected dimension of stored vectors.
    """

    def __init__(
        self,
        collection_name: str = "diplomatic_chunks",
        persist_directory: str = DEFAULT_PERSIST_DIR,
        embedding_dimension: int = 384,
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_dimension = embedding_dimension
        self.client = None
        self.collection = None
        self._init_store()

    def _init_store(self):
        try:
            import chromadb
            from chromadb.config import Settings

            Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(anonymized_telemetry=False),
            )
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(
                f"ChromaDB collection '{self.collection_name}' ready "
                f"({self.collection.count()} existing chunks)"
            )
        except ImportError:
            logger.warning(
                "chromadb not installed — vector store operating in memory-only mode. "
                "Install with: pip install chromadb"
            )
            self._init_memory_fallback()

    def _init_memory_fallback(self):
        """In-memory fallback when ChromaDB is not available."""
        self._mem_ids: List[str] = []
        self._mem_embeddings: List[np.ndarray] = []
        self._mem_documents: List[str] = []
        self._mem_metadatas: List[Dict] = []
        logger.info("Vector store running in memory-only fallback mode")

    # ── add ──────────────────────────────────────────────────────────
    def add_chunks(
        self,
        ids: List[str],
        embeddings: np.ndarray,
        documents: List[str],
        metadatas: Optional[List[Dict]] = None,
    ):
        """
        Add document chunks to the store.

        Parameters
        ----------
        ids : list[str]         Unique IDs for each chunk.
        embeddings : ndarray    Shape (n, dim).
        documents : list[str]   Raw chunk texts.
        metadatas : list[dict]  Per-chunk metadata (year, source, title …).
        """
        metadatas = metadatas or [{} for _ in ids]

        # Sanitise metadata values — ChromaDB only allows str/int/float/bool
        clean_meta = []
        for m in metadatas:
            clean = {}
            for k, v in m.items():
                if isinstance(v, (str, int, float, bool)):
                    clean[k] = v
                else:
                    clean[k] = str(v)
            clean_meta.append(clean)

        if self.collection is not None:
            # Batch upsert (handles duplicates)
            BATCH = 500
            for i in range(0, len(ids), BATCH):
                self.collection.upsert(
                    ids=ids[i:i + BATCH],
                    embeddings=embeddings[i:i + BATCH].tolist(),
                    documents=documents[i:i + BATCH],
                    metadatas=clean_meta[i:i + BATCH],
                )
            logger.info(f"Upserted {len(ids)} chunks → ChromaDB ({self.collection.count()} total)")
        else:
            self._mem_ids.extend(ids)
            self._mem_embeddings.extend(embeddings)
            self._mem_documents.extend(documents)
            self._mem_metadatas.extend(clean_meta)
            logger.info(f"Added {len(ids)} chunks to memory store ({len(self._mem_ids)} total)")

    # ── query ────────────────────────────────────────────────────────
    def query(
        self,
        query_embedding: np.ndarray,
        n_results: int = 5,
        where: Optional[Dict] = None,
    ) -> Dict:
        """
        Retrieve top-k chunks by cosine similarity.

        Returns
        -------
        dict with keys: ids, documents, metadatas, distances
        """
        if self.collection is not None:
            kwargs = {
                "query_embeddings": [query_embedding.tolist()],
                "n_results": min(n_results, max(self.collection.count(), 1)),
            }
            if where:
                kwargs["where"] = where
            results = self.collection.query(**kwargs)
            return {
                "ids": results["ids"][0] if results["ids"] else [],
                "documents": results["documents"][0] if results["documents"] else [],
                "metadatas": results["metadatas"][0] if results["metadatas"] else [],
                "distances": results["distances"][0] if results["distances"] else [],
            }
        else:
            return self._query_memory(query_embedding, n_results)

    def _query_memory(self, query_embedding: np.ndarray, n_results: int) -> Dict:
        if not self._mem_embeddings:
            return {"ids": [], "documents": [], "metadatas": [], "distances": []}

        db_vecs = np.array(self._mem_embeddings)
        q = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
        norms = np.linalg.norm(db_vecs, axis=1, keepdims=True) + 1e-10
        cos_sims = (db_vecs / norms) @ q
        top_k = np.argsort(cos_sims)[::-1][:n_results]

        return {
            "ids": [self._mem_ids[i] for i in top_k],
            "documents": [self._mem_documents[i] for i in top_k],
            "metadatas": [self._mem_metadatas[i] for i in top_k],
            "distances": [float(1 - cos_sims[i]) for i in top_k],
        }

    # ── utility ──────────────────────────────────────────────────────
    def count(self) -> int:
        if self.collection is not None:
            return self.collection.count()
        return len(self._mem_ids)

    def delete_collection(self):
        if self.client is not None:
            self.client.delete_collection(self.collection_name)
            logger.info(f"Deleted collection '{self.collection_name}'")

    def get_stats(self) -> Dict:
        return {
            "collection": self.collection_name,
            "total_chunks": self.count(),
            "persist_directory": self.persist_directory,
            "backend": "chromadb" if self.collection is not None else "memory",
        }


# ── CLI ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    store = VectorStore()
    print(store.get_stats())
