# Geopolitical Insight Engine — Cross-Lingual Diplomatic Barometer

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111.0-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40.0-FF4B4B.svg?logo=streamlit)](https://streamlit.io)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-orange.svg)](https://www.trychroma.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An enterprise-grade, dual-engine NLP and Retrieval-Augmented Generation (RAG) platform built for analyzing structural, thematic, and linguistic shifts in bilateral diplomatic relations over multi-decade horizons.

The primary focus is tracking the historical transformation of **India–Japan bilateral relations (2000–2026)** from an **economic and developmental partnership** (ODA, yen loans, infrastructure, CEPA) toward an active **security and defense alliance** (Quad, 2+2 Ministerial Dialogues, joint naval exercises, Free and Open Indo-Pacific). The framework also supports multi-bilateral scaling, including **India–France** relations.

---

## Architecture Overview

The system combines **rigorous quantitative NLP** (lexicon scoring, statistical testing, and topic modeling) with a **production RAG system** (hybrid search, vector embeddings, grounded citations, and an asynchronous REST API).

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        DATA INGESTION & HARVESTING                     │
│  MEA (India)  •  MOFA (Japan)  •  France MFA  •  Wayback CDX API      │
│  GDELT DOC 2.0 Discovery  •  Treaty Repositories  •  UN Comtrade / RSS │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                     PREPROCESSING & PROVENANCE                         │
│  • spaCy Lemmatization & Compound Phrase Preservation                 │
│  • Deduplication & Provenance Auditing (data_provenance_report.json)  │
└───────────────────┬────────────────────────────────┬───────────────────┘
                    │                                │
                    ▼                                ▼
┌──────────────────────────────────────┐  ┌──────────────────────────────┐
│       QUANTITATIVE NLP ENGINE        │  │     NEURAL RAG PIPELINE      │
│ • Strategic Shift Analyzer           │  │ • Recursive Chunker (1500 ch)│
│   - Multi-Word Lexicon Matching      │  │ • Embeddings (OpenAI/BGE-M3/ │
│   - Length Normalization             │  │   sentence-transformers)     │
│   - Rolling Crossover Detection      │  │ • ChromaDB Vector Store      │
│ • Statistical Validation             │  │ • Pure-Python BM25 Search    │
│   - Paired t-test & Wilcoxon         │  │ • Reciprocal Rank Fusion     │
│   - Permutation & Bootstrap 95% CIs  │  │ • Grounded Citation Layer    │
│   - Cohen's d Effect Size            │  │ • Multi-LLM Backends         │
│ • Thematic Modeling (LDA)            │  │   (OpenAI GPT-4o / Ollama    │
│ • Tone & Urgency (Modal Verbs)       │  │    Llama-3 / Context Mode)   │
└───────────────────┬──────────────────┘  └──────────────┬───────────────┘
                    │                                    │
                    └─────────────────┬──────────────────┘
                                      │
                                      ▼
┌────────────────────────────────────────────────────────────────────────┐
│                             SERVING LAYER                              │
│                                                                        │
│   Streamlit Dashboard (Port 8501)       FastAPI Microservice (Port 8000)│
│   • Executive Summary                   • Asynchronous Endpoints       │
│   • Strategic Shift Trends              • Hybrid / Keyword / Vector    │
│   • Topics & Themes (LDA)               • Grounded RAG with Citations  │
│   • Statistical Checks & CIs            • Text Classification          │
│   • Year Explorer & Keyword Search      • Automated Ingestion & Status │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Core Features & Functionality

### 1. Quantitative Strategic Shift Analysis
* **Domain Lexicons**: Curated, domain-verified lexicons capturing economic cooperation (ODA, yen loans, CEPA, infrastructure) and security alignments (defense, maritime domain awareness, Quad, deterrence, joint exercises).
* **Multi-Word Phrase Matching**: Evaluates compound diplomatic terminology (e.g., *"Free and Open Indo-Pacific"*, *"bullet train"*, *"acquisition and cross-servicing agreement"*) directly on preprocessed text.
* **Rolling Crossover Point Detection**: Identifies structural pivot years where security focus sustainedly eclipses economic focus, applying multi-year persistence checks rather than single-year anomalies.
* **Statistical Rigor**:
  * Paired sample $t$-tests and non-parametric Wilcoxon signed-rank tests.
  * Permutation sign-flip tests for small-sample validation ($N < 20$).
  * 4,000-draw bootstrap 95% confidence intervals on score differences.
  * Cohen's $d$ effect sizes and Shapiro-Wilk normality diagnostics.
* **Source Disaggregation**: Splits trends by source agency (MEA India vs. MOFA Japan) to detect single-actor bias.

### 2. Thematic & Tone Evolution
* **Topic Modeling**: Unsupervised Latent Dirichlet Allocation (LDA) using Scikit-Learn with custom diplomatic stopword filtration to discover persistent and emerging themes across decades.
* **Tone & Urgency Scoring**: TextBlob polarity/subjectivity combined with modal-verb syntax parsing (*"must"*, *"shall"*, *"will"*, *"urgently"*) to classify diplomatic posture into *Cooperative*, *Assertive*, or *Cautious*.

### 3. Production Neural RAG Pipeline (`rag/`)
* **Recursive Chunking** (`rag/chunker.py`): Segments texts into ~1500-character chunks with 150-character overlap, strictly preserving source document ID, agency, year, title, and character offsets.
* **Adaptive Multi-Backend Embeddings** (`rag/embeddings.py`): Automatically selects OpenAI (`text-embedding-3-small`), HuggingFace (`BGE-M3`), or `sentence-transformers` (`all-MiniLM-L6-v2`), with zero-dependency fallbacks for offline testing.
* **Vector Store** (`rag/vector_store.py`): ChromaDB HNSW persistent indexing with batch upserting (500/batch) and in-memory mock fallback.
* **Hybrid Search with RRF** (`rag/hybrid_search.py`): Combines vector cosine similarity with Okapi BM25 keyword matching via Reciprocal Rank Fusion ($k=60$), balancing semantic relevance with precise diplomatic terminology.
* **Citation & Anti-Hallucination Guardrail** (`rag/citation_layer.py`): Validates every generated statement against numbered retrieved passages. Asserts that names, treaties, and dates exist in context before marking responses as grounded.
* **Multi-LLM Support** (`rag/rag_pipeline.py`): Seamless switching between OpenAI GPT-4o, local Ollama (Llama-3), or a zero-API context assembler.

### 4. Asynchronous REST API (`api/`)
* Built with FastAPI and Pydantic v2 schemas.
* Full OpenAPI documentation and interactive testing at `http://localhost:8000/docs`.
* CORS enabled for web frontends. Auto-ingests documents into vector memory on demand.

### 5. Multi-Source Archival Harvesting (`scrapers/`)
* **Unified Harvester** (`scrapers/harvesting/`):
  * **Wayback CDX API**: Historical snapshot resolution of diplomatic portals back to 2000.
  * **Ministry Archives**: Specialized extractors for MEA (India), MOFA (Japan), and France MFA.
  * **GDELT DOC 2.0**: Automated historical discovery of official press releases and joint statements.
  * **Treaty Repositories**: Direct bilateral treaty text ingestion.
* **Non-Destructive URL Backfilling & Provenance**: Auto-resolves source URLs and publishes `data_provenance_report.json` with duplicate-removal keys and source share ratios.

### 6. Automated Evaluation & Governance (`evaluation/`)
* **Golden Dataset Benchmark** (`evaluation/golden_dataset.py`): 50 hand-annotated diplomatic paragraphs evaluated for Macro-F1 across Economic, Security, and Cultural classifications.
* **RAGAS Evaluation** (`evaluation/ragas_eval.py`): Quantitative assessment of Context Recall, Context Precision, Faithfulness, and Answer Relevance.
* **Adversarial Safety Probes** (`evaluation/adversarial_tests.py`): 15 tests covering prompt injection, data poisoning, out-of-scope inquiries, hallucination traps, and bias probing.
* **Latency Benchmarks** (`evaluation/latency_tests.py`): Automated monitoring against P95 latency thresholds for keyword, vector, hybrid, and LLM query passes.

### 7. Automation & Scheduled Reporting (`automation/`)
* **Cron Updater** (`automation/cron_updater.py`): Incremental weekly delta crawler that ingests newly published communiqués and embeds them into ChromaDB.
* **Weekly Briefing Digest** (`automation/weekly_digest.py`): Automated generation of strategic bilateral briefing reports in PDF (ReportLab) or Markdown.

---

## Directory Structure

```text
diplomatic_barometer/
├── analysis/               # Quantitative NLP and statistical modules
│   ├── strategic_shift_enhanced.py # Phrase matching, crossover, bootstrap statistics
│   ├── thematic_clustering.py     # LDA topic discovery and timeline extraction
│   └── tone_analyzer.py           # Polarity, subjectivity, and modal urgency
├── api/                    # FastAPI microservice
│   └── main.py             # 9 async REST endpoints (/query, /search, /classify, etc.)
├── automation/             # Production schedulers and digests
│   ├── cron_updater.py     # Incremental crawler and vector database updater
│   └── weekly_digest.py    # Executive briefing PDF/Markdown generator
├── dashboard/              # Streamlit web application
│   ├── app.py              # Main multi-page interactive UI (port 8501)
│   └── page_renderers.py   # Specialized view renderers (Overview, Stats, Shift)
├── data/                   # Data storage (git-ignored raw/processed data)
│   ├── raw/                # Canonical CSVs, SQLite cache, and harvest snapshots
│   └── processed/          # Provenance reports, analysis JSON, Excel summaries
├── docker/                 # Containerization configs
│   ├── Dockerfile.api      # FastAPI container definition
│   ├── Dockerfile.dashboard# Streamlit container definition
│   └── docker-compose.yml  # Orchestrates ChromaDB, API, and Dashboard
├── evaluation/             # Test benchmarks and safety suites
│   ├── adversarial_tests.py# 15 government-context safety and grounding checks
│   ├── binary_classifier.py# Supervised ML baseline comparing with lexicon scores
│   ├── golden_dataset.py   # 50-paragraph test set and macro-F1 evaluator
│   ├── latency_tests.py    # Latency benchmarks against performance budgets
│   └── ragas_eval.py       # RAGAS metrics (Faithfulness, Relevance, Recall)
├── preprocessing/          # Linguistic normalization
│   └── preprocessor.py     # spaCy lemmatization and diplomatic entity preservation
├── rag/                    # Neural Retrieval-Augmented Generation
│   ├── chunker.py          # Provenance-preserving recursive text splitter
│   ├── citation_layer.py   # Grounding validator and anti-hallucination guardrail
│   ├── embeddings.py       # Multi-backend embeddings (OpenAI, BGE-M3, MiniLM)
│   ├── hybrid_search.py    # Okapi BM25 + Dense vector Reciprocal Rank Fusion
│   ├── rag_pipeline.py     # Ingest and query orchestrator
│   └── vector_store.py     # ChromaDB wrapper with HNSW indexing
├── scrapers/               # Official document crawlers and builders
│   ├── harvesting/         # Unified multi-provider harvester (Wayback, Treaties, etc.)
│   ├── data_loader.py      # Canonical corpus loader and deduplication engine
│   ├── mea_crawler_enhanced.py # India MEA crawler
│   ├── mofa_crawler_enhanced.py# Japan MOFA crawler
│   ├── france_mfa_crawler_enhanced.py # France MFA crawler
│   ├── official_corpus_builder.py # GDELT DOC 2.0 corpus builder
│   └── live_scrape_validator.py # Real-time scraping validator
├── tests/                  # Pytest test suite
│   ├── conftest.py         # Test fixtures and synthetic data
│   ├── test_analysis.py    # Strategic shift, tone, and thematic unit tests
│   ├── test_data_loading.py# Data loader and deduplication checks
│   ├── test_harvesting.py  # Harvester provider integration tests
│   └── test_pipeline.py    # End-to-end pipeline test
├── utils/                  # Domain configuration and helper utilities
│   ├── country_config.py   # Country definitions, flags, and ministry metadata
│   ├── crawler_factory.py  # Dynamic crawler instantiation by country pair
│   ├── helpers.py          # Provenance reporting, JSON, and Excel export helpers
│   ├── contradiction_detection.py # Automated policy contradiction detection
│   └── issue_tagging.py    # Policy domain and regional equity tagging
├── pipeline.py             # Main CLI batch processing execution script
├── pipeline_scheduler.py   # Continuous periodic pipeline runner
├── quickstart.py           # Verification script for configs, factories, and crawlers
├── requirements.txt        # Pinned project dependencies
└── run.py                  # Unified CLI launcher for all project workflows
```

---

## Installation & Setup

### Prerequisites
* **Python 3.11** (Recommended for strict spaCy and C-extension compatibility)
* **Git**

### Step-by-Step Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/07ItachiUchiha/Cross-Lingual-NLP-Framework-Diplomatic-Barometer-.git
   cd diplomatic_barometer
   ```

2. **Create and activate a Python 3.11 virtual environment**:
   ```bash
   # Windows PowerShell:
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1

   # Linux / macOS:
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

4. **Configure Environment Variables** (Optional):
   Create a `.env` file in the project root:
   ```env
   # LLM & Embeddings (optional — falls back to local models or context mode if omitted)
   OPENAI_API_KEY=your_openai_key_here
   LLM_BACKEND=openai          # options: openai, ollama, none
   EMBEDDING_BACKEND=sentence-transformers # options: openai, huggingface, sentence-transformers

   # External Signals / News APIs (optional)
   NEWSDATA_API_KEY=your_key
   NEWSAPI_API_KEY=your_key
   WORLDNEWS_API_KEY=your_key
   ```

5. **Verify Setup**:
   ```bash
   python quickstart.py
   ```

---

## Usage Guide: The Unified Launcher (`run.py`)

All system workflows are orchestrated via the `run.py` unified launcher.

### Primary Operational Modes

| Command | Description |
| :--- | :--- |
| `python run.py` | **Default Mode**: Executes the batch analysis pipeline and launches the Streamlit dashboard on `http://localhost:8501`. |
| `python run.py --api` | Starts the **FastAPI REST backend** on `http://localhost:8000` with Swagger UI at `/docs`. |
| `python run.py --full` | Executes the batch pipeline, starts the FastAPI server in the background, and opens the Streamlit dashboard. |
| `python run.py --ingest` | Chunks and embeds documents into the **ChromaDB vector store** and builds the BM25 index. |
| `python run.py --evaluate` | Executes the **Golden Dataset F1 evaluation** and the **15 adversarial safety checks**. |
| `python run.py --benchmark` | Runs **latency and throughput benchmarks** against search and RAG targets. |
| `python run.py --digest` | Compiles and exports the **Weekly Strategic Briefing Report** to `data/reports/`. |
| `python run.py --cron` | Executes the scheduled incremental crawler and vector store updater. |

### Data Harvesting & Corpus Building

| Command | Description |
| :--- | :--- |
| `python run.py --build-official-corpus --corpus-start-year 2000 --corpus-end-year 2026 --corpus-max-docs 600` | Discovers and extracts official India–Japan MEA/MOFA communiqués via GDELT DOC 2.0. |
| `python run.py --build-india-france-corpus` | Builds the bilateral corpus for India–France into `data/raw/india_france_documents.csv`. |
| `python run.py --unified-harvest --harvest-pair all` | Runs the multi-provider harvester across Wayback CDX, ministry archives, and treaty repos. |
| `python run.py --scrape-live` | Tests live scraping and parsing against official government URLs. |
| `python run.py --provenance-only` | Audits raw data, computes deduplication metrics, and writes `data_provenance_report.json`. |

---

## FastAPI Endpoints Reference

When running `python run.py --api` (or `uvicorn api.main:app --port 8000`), the following REST endpoints are available:

| Endpoint | Method | Description | Sample Payload / Params |
| :--- | :--- | :--- | :--- |
| `/health` | `GET` | Health and liveness status. | *None* |
| `/api/v1/status` | `GET` | Status of RAG pipeline, doc count, and active LLM backend. | *None* |
| `/api/v1/ingest` | `POST` | Ingests and embeds corpus into ChromaDB. | `{"force_reload": false}` |
| `/api/v1/query` | `POST` | Executes grounded RAG question answering with verified citations. | `{"question": "How did defense cooperation evolve after 2014?", "search_mode": "hybrid", "n_results": 5}` |
| `/api/v1/search` | `POST` | Pure hybrid search (BM25 + Dense vector RRF) without LLM call. | `{"query": "acquisition and cross-servicing", "mode": "hybrid", "n_results": 10}` |
| `/api/v1/classify` | `POST` | Classifies text passage into Economic, Security, or Cultural domain. | `{"text": "Both sides agreed to expand naval joint drills in the Malabar exercise."}` |
| `/api/v1/evaluate` | `GET` | Runs golden dataset F1 evaluation and returns class metrics. | *None* |
| `/api/v1/documents`| `GET` | Lists loaded documents with metadata filtering. | `?skip=0&limit=50&year=2023` |
| `/api/v1/stats` | `GET` | Reports total chunks, vector dimensions, and corpus statistics. | *None* |

---

## Interactive Streamlit Dashboard

The dashboard provides an intuitive, interactive analytical workbench featuring 7 specialized views:

1. **Executive Summary**: High-level strategic overview, corpus health metrics, key findings, and policy takeaways.
2. **Economic vs. Security**: Longitudinal time-series curves with 95% confidence bands, crossover points, and source disaggregation (MEA vs. MOFA).
3. **Topics & Themes**: LDA topic distributions, keyword word clouds, and theme evolution across decades.
4. **Statistical Checks**: Rigorous hypothesis testing outputs, paired differences, Cohen's $d$, Wilcoxon $p$-values, bootstrap distributions, and decision hygiene reports.
5. **Time Machine**: Year-by-year document drilldown, primary vocabulary shifts, and key communiqués.
6. **Keyword Search**: In-corpus full-text search with metadata filtering and original text review.
7. **Tone & Mood**: Sentiment polarity, subjectivity, and modal-verb urgency indices over time.

---

## Evaluation & Benchmarks

The project includes automated evaluation harnesses for academic and production reliability:

```bash
# Run pytest unit and integration test suite
python -m pytest tests/ -v

# Run the golden dataset F1 evaluation
python run.py --evaluate

# Run performance and latency benchmarks
python run.py --benchmark
```

* **Golden Dataset F1**: Tested against 50 manually classified paragraphs from official communiqués.
* **Adversarial Safety**: Validates refusal and grounding against 15 test vectors including prompt injections, out-of-corpus queries, and hallucination bait.
* **Latency Budgets**: Asserts performance against defined thresholds (Vector Search < 2s, Hybrid Search < 3s, End-to-End RAG < 5s).

---

## Docker & Container Deployment

A multi-container setup orchestrates ChromaDB, the FastAPI backend, and the Streamlit dashboard:

```bash
cd docker
docker compose up --build
```

### Services Deployed:
* **`chromadb`** on port `8100`: Persistent vector store with health checks.
* **`api`** on port `8000`: FastAPI backend (depends on healthy ChromaDB).
* **`dashboard`** on port `8501`: Streamlit UI connected to the backend.

To stop services:
```bash
docker compose down
```

---

## Continuous Integration (CI/CD)

The GitHub Actions pipeline (`.github/workflows/ci.yml`) runs on all pushes and pull requests to `main` and `develop`:
1. **Linting**: Code quality enforcement via `ruff`.
2. **Testing**: `pytest` execution with code coverage reporting.
3. **Model Evaluation**: Asserts that Golden Dataset Macro-F1 exceeds minimum quality thresholds.
4. **Docker Smoke Test**: Builds container images and executes `/health` verification curl tests.

---

## Data Governance & Reproducibility

* **Provenance Tracking**: Every batch run writes a detailed `data/processed/data_provenance_report.json` logging the exact file source, raw rows, deduplication keys, missing fields, and date ranges.
* **Decision Hygiene**: Transparent rule-based indicators highlight low-volume years ($n < 3$), source imbalances, and coverage gaps to prevent over-interpreting sparse historical data.
* **Sensitive Assets Notice**: API keys, local `.env` files, raw scraped caches (`*.sqlite`), and vector databases are excluded from version control via `.gitignore`.
