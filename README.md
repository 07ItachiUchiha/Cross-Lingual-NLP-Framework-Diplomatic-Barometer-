# Cross-Lingual NLP Framework — Implementation (pre-RAG)

This repository implements the pre-RAG research pipeline used to analyze shifts in India–Japan diplomatic language (economic → security). The README below focuses only on the implementation and analysis components that form the core of the research (before any retrieval-augmented or LLM-based additions).

## Overview

The project ingests official diplomatic documents, performs structured preprocessing and feature extraction, applies unsupervised topic modeling, measures tone and urgency, and validates observed shifts with standard statistical tests. Results are exposed through a Streamlit dashboard for exploration and export.

## Core pipeline

1. Preprocessing — cleaning, tokenization, lemmatization with `spaCy`.
2. Lexicon-based scoring — document-level economic and security scores computed from curated lexicons and normalized by document length.
3. Topic modeling — LDA (sklearn) with bigrams and a tuned stopword list to reveal persistent themes.
4. Tone & urgency analysis — polarity/subjectivity (TextBlob) and modal-verb-based urgency scoring.
5. Statistical testing — t-tests and Cohen's d to quantify whether shifts are statistically meaningful.

## Setup

Use a Python 3.9+ virtual environment and install dependencies:

```bash
cd diplomatic_barometer
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS / Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Run the pipeline and dashboard

Run the end-to-end pipeline and open the dashboard:

```bash
# run the full pipeline and launch dashboard
python run.py

# or run only the dashboard UI
streamlit run dashboard/app.py
```

The dashboard opens at <http://localhost:8501> by default.

## Project layout (implementation-focused)

```
diplomatic_barometer/
├── scrapers/           # data loaders for MEA and MOFA source documents
├── preprocessing/      # cleaning, tokenization, lemmatization
├── analysis/           # scoring, tone analysis, topic modeling, stats
├── dashboard/          # Streamlit UI and export helpers
├── data/               # raw and processed corpora
├── evaluation/         # evaluation helpers (golden dataset conversion, F1 helpers)
├── tests/              # pytest suite for preprocess/analysis/pipeline
└── run.py              # launcher for common workflows
```

## How the analysis works (details)

### Lexicon scoring

Two curated lexicons (Economic, Security) are applied to each document. Counts are normalized by token length and aggregated to yearly or thematic summaries. The lexicons are purposely conservative (terms selected to reduce false positives in diplomatic text).

### Topic modeling (LDA)

Preprocessed texts use bigrams/trigrams where helpful. LDA runs with a default of 5 topics; topic coherence and top-term inspection guide topic count choice. Custom stopwords remove extremely frequent diplomatic boilerplate.

### Tone and urgency

Polarity and subjectivity are estimated via TextBlob. An urgency score derives from modal verbs and imperative constructions (e.g., "must", "shall", "will"). Combining these features yields simple tone classes (Cooperative / Assertive / Cautious).

### Statistical validation

Hypotheses about shifts (e.g., economic score vs. security score over time) are tested with independent t-tests and effect sizes (Cohen's d). Reported p-values use α=0.05 unless otherwise specified.

## Dashboard pages (implementation outputs)

1. Overview — corpus stats and main economic vs security visualization
2. Economic vs Security — yearly breakdown and top category terms
3. Tone & Mood — sentiment/urgency distributions and trends
4. Topics & Themes — LDA results and topic timelines
5. Time Machine — year-based word clouds and most-relevant documents
6. Search — keyword search across the processed corpus
7. Statistical Tests — t-test results, effect sizes, and exportable tables

## Tests

Run the test suite locally:

```bash
python -m pytest tests/ -v
```

The tests cover data loading, preprocessing, analysis modules, and pipeline integration.

## Reproducibility notes

- Ensure `en_core_web_sm` is downloaded for `spaCy` processing.
- Use the same `requirements.txt` to recreate the environment.
- Processed outputs are in `data/processed/` (check `analysis_results.json` for the processed corpus summary).

## Files not to publish

Do NOT publish the following files or directories to a public Git repository. They often contain sensitive data, large binaries, or licensing-restricted model weights. Ensure these are listed in `.gitignore` before pushing.

- `*.env`, `.env.local` — environment variables that may contain API keys and secrets.
- `credentials.json`, `secret-*.json` — stored service credentials or token files.
- `*.pem`, `*.key`, `*.crt` — private certificates and keys.
- `data/raw/`, `data/processed/` (if containing private data) — raw or sensitive datasets.
- `data/vector_db/`, `models/`, `weights/`, `checkpoints/`, `snapshots/` — large model files, embeddings, or database persistence (often very large and proprietary).
- `*.sqlite`, `*.db` — local databases containing PII or internal data.
- `logs/`, `*.log` — runtime logs which may contain sensitive traces.

If you need to share processed outputs or models with collaborators, use a private release, a cloud storage link, or a separate sanitized artifact repository rather than committing secrets or large binaries to the code repo.

## Next steps (research-centric)

- Expand and annotate the Golden Dataset for cross-validation and inter-annotator agreement.
- Validate lexicons against annotated samples and adjust terms to reduce labeling bias.
- If you want, I can prepare a short checklist for publishing (git push commands and .gitignore notes).

---

## Post-research implementation (planned)

After the core research is complete, the following implementation work is planned and will enable the described production features:

- Retrieval-Augmented Pipeline: document chunking, dense embeddings, and a vector store (e.g., ChromaDB) to enable accurate source retrieval for queries.
- LLM Backends: adapters for Gemini/OpenAI or a local runtime (Ollama) to support summarization, QA, and citation-aware responses.
- Hybrid Retrieval: combine BM25 lexical search with dense vector search (RRF/hybrid fusion) to improve recall and precision for short queries.
- Serving API: a FastAPI service exposing ingestion, search, QA, and evaluation endpoints for reproducible experiments and external integration.
- Evaluation & Monitoring: expand the Golden Dataset, add RAGAS-style metrics and adversarial tests, and run latency/throughput benchmarks to validate performance.
- Automation & Deployment: cron-based incremental ingestion, weekly digest generation, Docker images, and CI to make experiments reproducible and shareable.

These additions will produce grounded, explainable answers with provenance, enable fast reproducible comparisons of retrieval/model choices, and provide a stable demo/API for reviewers or publication.

---

If you want any additional edits focused on experimental reproducibility (e.g., pinned package versions or a lightweight Docker reproduction), tell me and I will add them.
