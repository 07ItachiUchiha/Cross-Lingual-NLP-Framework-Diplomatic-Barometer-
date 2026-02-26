# Changelog — Diplomatic Barometer → Geopolitical Insight Engine

## What changed and why

The original project did lexicon-based tone analysis and LDA topic modelling on India-Japan diplomatic documents. It worked, but it was essentially a static NLP pipeline with a Streamlit frontend. The upgrade turns it into a production-grade system backed by retrieval-augmented generation, a proper API layer, automated evaluation, and containerised deployment.

Below is everything that was added or modified, grouped by module.

---

## RAG Pipeline (`rag/`)

This is the core addition. Instead of just running keyword counts and LDA, the system now chunks documents, embeds them, stores vectors in ChromaDB, and retrieves relevant context before generating answers.

### `rag/chunker.py`

Recursive character text splitter. Breaks diplomatic documents into ~1500-character chunks with 150-char overlap. Preserves full provenance — each chunk knows which document it came from, the year, source agency, and exact character offsets. Separators are tuned for diplomatic prose (paragraph breaks → sentence boundaries → clause-level).

### `rag/embeddings.py`

Multi-backend embedding manager. Supports OpenAI (`text-embedding-3-small`), HuggingFace (`BGE-M3`), and sentence-transformers (`all-MiniLM-L6-v2`) as backends. Picks the best available at runtime and falls back gracefully — so the system works in CI environments with no API keys.

### `rag/vector_store.py`

ChromaDB wrapper with HNSW indexing. Handles batch upserts (500 per batch), cosine similarity queries, metadata filtering. If ChromaDB isn't installed (e.g., lightweight dev setup), it falls back to an in-memory store that still supports the same interface.

### `rag/hybrid_search.py`

Combines BM25 keyword search with vector cosine similarity using Reciprocal Rank Fusion (k=60). Default weighting is 40% keyword / 60% vector. Can be switched to keyword-only or vector-only mode. The BM25 implementation is pure Python (Okapi BM25 with IDF smoothing), no external dependencies.

### `rag/citation_layer.py`

Hallucination guardrail. Every claim in a generated answer must trace back to a source chunk. The layer builds grounded prompts that explicitly number context passages, then validates that the answer doesn't fabricate dates, names, or events not present in the retrieved context. Outputs a `CitedAnswer` with confidence score and grounding flag.

### `rag/rag_pipeline.py`

Orchestrator that ties everything together. Two main flows:

- **Ingest**: DataFrame → chunk → embed → store in ChromaDB + build BM25 index
- **Query**: user question → hybrid search → build grounded prompt → call LLM → validate citations

Supports three LLM backends: OpenAI GPT-4o, Ollama (local Llama-3), and a `none` mode that just returns formatted context (useful for CI and testing without API keys).

---

## API Layer (`api/`)

### `api/main.py`

FastAPI backend with 9 endpoints:

| Endpoint            | Method | What it does                                   |
| ------------------- | ------ | ---------------------------------------------- |
| `/health`           | GET    | Liveness check                                 |
| `/api/v1/status`    | GET    | Pipeline status, doc count, LLM backend info   |
| `/api/v1/ingest`    | POST   | Load and embed documents into vector store     |
| `/api/v1/query`     | POST   | RAG query with citations                       |
| `/api/v1/search`    | POST   | Hybrid search without LLM generation           |
| `/api/v1/classify`  | POST   | Classify text as Economic/Security/Cultural    |
| `/api/v1/evaluate`  | GET    | Run golden dataset F1 evaluation               |
| `/api/v1/documents` | GET    | List ingested documents                        |
| `/api/v1/stats`     | GET    | Chunk count, embedding dimensions, index stats |

CORS is enabled. Auto-ingest happens on first query if the store is empty. Pydantic v2 models for request/response validation.

Run with: `uvicorn api.main:app --reload --port 8000`

---

## Evaluation Suite (`evaluation/`)

### `evaluation/golden_dataset.py`

50 manually labelled paragraphs from real India-Japan diplomatic documents — 20 Economic, 20 Security, 10 Cultural. Each entry has the full text, ground-truth label, year, and document title. The `F1Evaluator` class computes per-class precision/recall/F1, macro F1, and a confusion matrix. This is the baseline metric used in CI.

### `evaluation/ragas_eval.py`

RAGAS-style evaluation framework with four metrics:

- Context recall (how much of the ground truth is covered by retrieved context)
- Context precision (how relevant the retrieved chunks are)
- Faithfulness (does the answer stick to the context)
- Answer relevance (does the answer actually address the question)

Includes 5 pre-built diplomatic test cases with expected answers and ground truth context.

### `evaluation/adversarial_tests.py`

15 adversarial test cases across 5 categories: data poisoning, hallucination probing, out-of-scope queries, prompt injection, and bias detection. Each test has an input, expected behaviour (refuse/hedge/stay-grounded), and detection logic based on refusal markers and hallucination indicators. Designed for government-context safety validation.

### `evaluation/latency_tests.py`

Performance benchmarks with configurable targets:

- Vector search: < 2 seconds
- Keyword search: < 500 ms
- Hybrid search: < 3 seconds
- Full RAG query (with LLM): < 5 seconds

Runs each benchmark multiple times and reports mean, median, P95, min, max. Pass/fail against targets.

---

## Automation (`automation/`)

### `automation/cron_updater.py`

Scheduled crawler that checks MEA and MOFA websites for new documents published since the last run. New documents get preprocessed, chunked, embedded, and upserted into ChromaDB automatically. Persists run state to `data/cron_state.json` so it knows the cutoff date for next time.

Intended to run via cron: `0 9 * * 1` (every Monday at 9 AM).

### `automation/weekly_digest.py`

Generates a briefing document summarising the week's activity:

- How many new documents were ingested
- Vocabulary/tone shifts (emerging vs. declining terms)
- Top strategic queries and their RAG answers
- Thematic trends

Outputs PDF via reportlab if available, otherwise falls back to Markdown. Saved to `data/reports/weekly_digest_YYYY-MM-DD.pdf`.

---

## Docker & CI/CD

### `docker/Dockerfile.api`

Multi-stage build for the FastAPI backend. Slim Python 3.11 base, health check on `/health`, runs as non-root. Default env: `LLM_BACKEND=none`, `EMBEDDING_BACKEND=sentence-transformers`.

### `docker/Dockerfile.dashboard`

Multi-stage build for the Streamlit dashboard. Health check on `/_stcore/health`.

### `docker/docker-compose.yml`

Three services:

- `chromadb` — official ChromaDB image, persistent volume, port 8100
- `api` — FastAPI backend, port 8000, depends on ChromaDB health
- `dashboard` — Streamlit, port 8501, depends on API

Start everything: `docker compose up --build`

### `.github/workflows/ci.yml`

GitHub Actions pipeline triggered on push to `main`/`develop` and PRs to `main`:

1. **Lint** — ruff
2. **Test** — pytest with coverage, golden dataset F1 check (must be > 0.3), adversarial test structure validation
3. **Build** — Docker image build + API smoke test (start container, curl `/health`)

---

## Modified Files

### `requirements.txt`

Added: `chromadb`, `sentence-transformers`, `openai`, `fastapi`, `uvicorn[standard]`, `pydantic>=2.7`, `httpx`, `pytest`, `pytest-cov`, `pytest-asyncio`, `ruff`. Existing deps unchanged.

### `run.py`

Rewritten as a unified launcher with argparse. Modes:

```

python run.py              # classic: pipeline → Streamlit
python run.py --api        # FastAPI only
python run.py --full       # pipeline + API + dashboard in parallel
python run.py --ingest     # load docs into RAG vector store
python run.py --evaluate   # run F1 + adversarial evaluation
python run.py --digest     # generate weekly briefing PDF
python run.py --cron       # crawl + ingest new documents
python run.py --benchmark  # latency benchmarks
```

---

## What's not changed

All existing modules (`scrapers/`, `analysis/`, `preprocessing/`, `utils/`, `dashboard/app.py`, `pipeline.py`) are untouched. The new system extends rather than replaces — the original Streamlit dashboard and LDA pipeline still work as before.
