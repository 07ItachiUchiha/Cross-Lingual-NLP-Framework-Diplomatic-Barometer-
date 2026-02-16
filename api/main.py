"""
FastAPI Backend — Asynchronous REST API
----------------------------------------
Exposes the Geopolitical Insight Engine as a web service.

Endpoints:
  GET  /health                 — health check + system status
  GET  /api/v1/status          — RAG pipeline status
  POST /api/v1/ingest          — ingest documents into vector DB
  POST /api/v1/query           — ask a question (RAG)
  POST /api/v1/search          — hybrid search (no LLM)
  POST /api/v1/classify        — classify a paragraph
  GET  /api/v1/evaluate        — run golden dataset evaluation
  GET  /api/v1/documents       — list loaded documents
  GET  /api/v1/stats           — corpus statistics

Run:
  uvicorn api.main:app --reload --port 8000
"""

import sys
import os
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Pydantic schemas ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, description="The question to ask")
    n_results: int = Field(5, ge=1, le=20)
    search_mode: str = Field("hybrid", pattern="^(hybrid|keyword|vector)$")
    year_filter: Optional[int] = None
    source_filter: Optional[str] = None


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=2)
    n_results: int = Field(10, ge=1, le=50)
    mode: str = Field("hybrid", pattern="^(hybrid|keyword|vector)$")
    year_filter: Optional[int] = None


class ClassifyRequest(BaseModel):
    text: str = Field(..., min_length=10)


class IngestRequest(BaseModel):
    force_reload: bool = Field(False, description="Re-ingest even if data already loaded")


# ── Application factory ──────────────────────────────────────────────

def create_app() -> FastAPI:
    """Factory that builds and wires the FastAPI application."""

    application = FastAPI(
        title="Geopolitical Insight Engine",
        description=(
            "LLM-Driven Strategic Analysis of Diplomatic Corpora. "
            "India-Japan bilateral relations 2000-2025."
        ),
        version="2.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS — allow Streamlit / React frontends
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── shared state ─────────────────────────────────────────────────
    state: Dict = {
        "pipeline": None,
        "df": None,
        "ingested": False,
        "boot_time": time.time(),
    }

    def _get_pipeline():
        if state["pipeline"] is None:
            from rag.rag_pipeline import RAGPipeline
            state["pipeline"] = RAGPipeline()
        return state["pipeline"]

    def _load_data():
        if state["df"] is None:
            from scrapers.data_loader import DataLoader
            loader = DataLoader()
            state["df"] = loader.load_sample_data()
        return state["df"]

    # ── routes ───────────────────────────────────────────────────────

    @application.get("/health")
    async def health():
        return {
            "status": "healthy",
            "uptime_seconds": round(time.time() - state["boot_time"], 1),
            "ingested": state["ingested"],
        }

    @application.get("/api/v1/status")
    async def pipeline_status():
        pipe = _get_pipeline()
        return pipe.get_status()

    @application.post("/api/v1/ingest")
    async def ingest_documents(req: IngestRequest):
        """Chunk, embed, and store the diplomatic corpus into the vector DB."""
        if state["ingested"] and not req.force_reload:
            return {"status": "already_ingested", "message": "Data already loaded. Use force_reload=true to re-ingest."}

        pipe = _get_pipeline()
        df = _load_data()
        result = pipe.ingest_dataframe(df)
        if result["status"] == "success":
            state["ingested"] = True
        return result

    @application.post("/api/v1/query")
    async def ask_question(req: QueryRequest):
        """
        Ask a question — full RAG pipeline.
        Returns answer with citations, confidence, and latency.
        """
        pipe = _get_pipeline()
        if not state["ingested"]:
            # Auto-ingest on first query
            df = _load_data()
            pipe.ingest_dataframe(df)
            state["ingested"] = True

        result = pipe.query(
            question=req.question,
            n_results=req.n_results,
            search_mode=req.search_mode,
            year_filter=req.year_filter,
            source_filter=req.source_filter,
        )
        return result

    @application.post("/api/v1/search")
    async def search_documents(req: SearchRequest):
        """Hybrid search without LLM generation — returns raw chunks."""
        pipe = _get_pipeline()
        if not state["ingested"]:
            df = _load_data()
            pipe.ingest_dataframe(df)
            state["ingested"] = True

        results = pipe.hybrid_search.search(
            query=req.query,
            n_results=req.n_results,
            mode=req.mode,
        )
        return {"query": req.query, "results": results, "count": len(results)}

    @application.post("/api/v1/classify")
    async def classify_paragraph(req: ClassifyRequest):
        """Classify a paragraph as Economic / Security / Cultural."""
        pipe = _get_pipeline()
        label = pipe.classify_text(req.text)
        return {"text": req.text[:100] + "...", "label": label}

    @application.get("/api/v1/evaluate")
    async def run_evaluation():
        """Run golden dataset evaluation and return F1 metrics."""
        from evaluation.golden_dataset import GoldenDataset, F1Evaluator
        pipe = _get_pipeline()

        gold = GoldenDataset()
        predictions = [pipe.classify_text(t) for t in gold.get_texts()]

        evaluator = F1Evaluator(gold.get_labels(), predictions)
        report = evaluator.full_report()
        return report

    @application.get("/api/v1/documents")
    async def list_documents(
        year: Optional[int] = Query(None),
        source: Optional[str] = Query(None),
        limit: int = Query(50, ge=1, le=200),
    ):
        """List documents in the corpus with optional filters."""
        df = _load_data()
        result = df.copy()
        if year:
            result = result[result["year"] == year]
        if source:
            result = result[result["source"] == source]
        records = result.head(limit)[["date", "title", "source", "year", "location"]].to_dict("records")
        return {"documents": records, "total": len(result), "returned": len(records)}

    @application.get("/api/v1/stats")
    async def corpus_stats():
        """Return corpus-level statistics."""
        df = _load_data()
        return {
            "total_documents": len(df),
            "year_range": {"min": int(df["year"].min()), "max": int(df["year"].max())},
            "sources": df["source"].value_counts().to_dict(),
            "documents_per_year": df["year"].value_counts().sort_index().to_dict(),
        }

    return application


# ── module-level app instance (for uvicorn api.main:app) ─────────────
app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
