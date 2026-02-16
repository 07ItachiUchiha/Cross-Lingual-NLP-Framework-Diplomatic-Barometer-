"""
RAG (Retrieval-Augmented Generation) Module
--------------------------------------------
ChromaDB vector store, hybrid search, LLM orchestration with citation
guardrails.  Designed for the "Geopolitical Insight Engine."
"""

from .chunker import DocumentChunker
from .embeddings import EmbeddingManager
from .vector_store import VectorStore
from .hybrid_search import HybridSearchEngine
from .citation_layer import CitationLayer
from .rag_pipeline import RAGPipeline

__all__ = [
    'DocumentChunker', 'EmbeddingManager', 'VectorStore',
    'HybridSearchEngine', 'CitationLayer', 'RAGPipeline'
]
