"""
Cron Updater — Automated Live Feed Ingestion
----------------------------------------------
SDLC Phase 6: "Cron job checks MEA website every Monday at 9 AM,
auto-embeds new statements."

Schedule: Monday 09:00 IST (configurable)
Pipeline:
  1. Crawl MEA/MOFA for new documents since last run
  2. Preprocess & chunk new documents
  3. Generate embeddings & upsert into ChromaDB
  4. Log ingestion summary

Can be invoked:
  - Directly:  python -m automation.cron_updater
  - Via cron:  0 9 * * 1  cd /app && python -m automation.cron_updater
  - Via scheduler integration in pipeline_scheduler.py
"""

import os
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATE_FILE = PROJECT_ROOT / "data" / "cron_state.json"


class CronUpdater:
    """
    Checks MEA/MOFA sources for new documents and ingests them
    into the RAG pipeline.
    """

    def __init__(self, pipeline=None):
        self.pipeline = pipeline
        self.state = self._load_state()

    # ── State persistence ────────────────────────────────────────────
    def _load_state(self) -> dict:
        """Load last-run state from disk."""
        if STATE_FILE.exists():
            try:
                return json.loads(STATE_FILE.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {"last_run": None, "total_docs_ingested": 0, "runs": []}

    def _save_state(self):
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(json.dumps(self.state, indent=2, default=str), encoding="utf-8")

    # ── Crawling ─────────────────────────────────────────────────────
    def _crawl_new_documents(self) -> list:
        """
        Fetch new documents published since last run.
        Returns list of dicts with keys: title, text, date, source, url
        """
        import pandas as pd

        new_docs = []
        last_run = self.state.get("last_run")
        cutoff = (
            datetime.fromisoformat(last_run)
            if last_run
            else datetime.now() - timedelta(days=7)
        )

        # ── Try MEA Enhanced Crawler ─────────────────────────────────
        try:
            from scrapers.mea_crawler_enhanced import MEACrawlerEnhanced
            mea = MEACrawlerEnhanced()
            mea_docs = mea.crawl()
            if isinstance(mea_docs, pd.DataFrame) and not mea_docs.empty:
                for _, row in mea_docs.iterrows():
                    doc_date = pd.to_datetime(row.get("date", ""), errors="coerce")
                    if doc_date and doc_date >= cutoff:
                        new_docs.append({
                            "title": row.get("title", "Untitled"),
                            "text": row.get("text", row.get("content", "")),
                            "date": str(doc_date.date()),
                            "source": "MEA",
                            "url": row.get("url", ""),
                        })
            logger.info(f"MEA crawl returned {len(new_docs)} new documents")
        except Exception as e:
            logger.warning(f"MEA crawl failed: {e}")

        # ── Try MOFA Enhanced Crawler ────────────────────────────────
        try:
            from scrapers.mofa_crawler_enhanced import MOFACrawlerEnhanced
            mofa = MOFACrawlerEnhanced()
            mofa_docs = mofa.crawl()
            mofa_count = 0
            if isinstance(mofa_docs, pd.DataFrame) and not mofa_docs.empty:
                for _, row in mofa_docs.iterrows():
                    doc_date = pd.to_datetime(row.get("date", ""), errors="coerce")
                    if doc_date and doc_date >= cutoff:
                        new_docs.append({
                            "title": row.get("title", "Untitled"),
                            "text": row.get("text", row.get("content", "")),
                            "date": str(doc_date.date()),
                            "source": "MOFA",
                            "url": row.get("url", ""),
                        })
                        mofa_count += 1
            logger.info(f"MOFA crawl returned {mofa_count} new documents")
        except Exception as e:
            logger.warning(f"MOFA crawl failed: {e}")

        return new_docs

    # ── Ingestion ────────────────────────────────────────────────────
    def _ingest_documents(self, docs: list) -> dict:
        """Chunk, embed, and upsert into vector store."""
        if not docs:
            return {"ingested": 0, "chunks": 0}

        if not self.pipeline:
            logger.warning("No RAG pipeline configured — skipping ingestion.")
            return {"ingested": 0, "chunks": 0, "error": "no pipeline"}

        import pandas as pd

        df = pd.DataFrame(docs)
        df.rename(columns={"text": "content"}, inplace=True)
        stats = self.pipeline.ingest_dataframe(df)
        return {
            "ingested": len(docs),
            "chunks": stats.get("chunks_stored", 0),
        }

    # ── Main entry point ─────────────────────────────────────────────
    def run(self) -> dict:
        """Execute a cron update cycle."""
        logger.info("=" * 60)
        logger.info("  CRON UPDATER — Starting scheduled crawl")
        logger.info("=" * 60)

        run_start = datetime.now()

        # Step 1: Crawl
        new_docs = self._crawl_new_documents()
        logger.info(f"Found {len(new_docs)} new documents since last run")

        # Step 2: Ingest
        ingest_stats = self._ingest_documents(new_docs)
        logger.info(f"Ingested {ingest_stats.get('ingested', 0)} docs, "
                     f"{ingest_stats.get('chunks', 0)} chunks")

        # Step 3: Update state
        run_record = {
            "timestamp": run_start.isoformat(),
            "new_docs_found": len(new_docs),
            "docs_ingested": ingest_stats.get("ingested", 0),
            "chunks_created": ingest_stats.get("chunks", 0),
        }
        self.state["last_run"] = run_start.isoformat()
        self.state["total_docs_ingested"] = (
            self.state.get("total_docs_ingested", 0)
            + ingest_stats.get("ingested", 0)
        )
        self.state["runs"].append(run_record)
        self._save_state()

        logger.info("Cron update complete.")
        return run_record


# ── CLI entry point ──────────────────────────────────────────────────
if __name__ == "__main__":
    # Try to initialize RAG pipeline (optional)
    pipeline = None
    try:
        from rag.rag_pipeline import RAGPipeline
        pipeline = RAGPipeline()
        logger.info("RAG pipeline initialized for ingestion")
    except Exception as e:
        logger.warning(f"RAG pipeline not available: {e}")

    updater = CronUpdater(pipeline=pipeline)
    result = updater.run()
    print(json.dumps(result, indent=2, default=str))
