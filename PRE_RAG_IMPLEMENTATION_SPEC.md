# Pre-RAG Baseline Implementation Specification

This document specifies the **current implementation before RAG** and defines what to harden first for research-grade reliability.

Scope: everything in the baseline NLP pipeline (data loading → preprocessing → lexicon/tone/topic analysis → dashboard/reporting), excluding `rag/`, `api/`, and post-RAG orchestration modes.

---

## 1) Baseline objective

The baseline system quantifies diplomatic discourse shifts in India–Japan documents using transparent, reproducible NLP methods:

1. Data loading and caching from local real CSVs (canonical corpus preferred)
2. Text cleaning + tokenization + lemmatization
3. Lexicon-based economic vs security scoring
4. Tone/sentiment/urgency analysis
5. LDA topic modeling and thematic evolution
6. Statistical reporting and dashboard visualization

Primary baseline entrypoint: `pipeline.py`
Primary UI: `dashboard/app.py`

---

## 2) Pre-RAG module map (what is implemented)

### A. Data layer

#### `scrapers/data_loader.py`

- **Class**: `DataLoader`
- **Implemented features**:
  - `load_combined_data(mea_file=None, mofa_file=None)`
    - Loads CSV sources if provided, appends `source` labels.
    - If no split files are provided, loads the primary real corpus via `load_real_data()`.
  - `load_real_data(csv_path=None)`
    - Loads `data/raw/india_japan_documents_canonical.csv` if present, otherwise `data/raw/india_japan_documents.csv`.
    - Enforces required schema and fails fast if the real dataset is missing or invalid.
  - `write_or_update_canonical_corpus()`
    - Creates/updates `data/raw/india_japan_documents_canonical.csv` without modifying the primary file.
    - Conservatively fills missing `url` values using `data/raw/live_scrape_*.csv` when available.
- **Output schema** (baseline expectation):
  - Required: `date`, `title`, `location`, `signatories`, `content`, `source`, `year`
  - Derived (optional): `doc_type` (heuristic label inferred from `title`, used only for stratification/UI)

#### `data_integration.py`

- **Class**: `CountryPairDataLoader`
  - Caching and retrieval flow for country-pair data.
  - `load_country_pair_data(...)` uses cache → crawler attempt → fail fast if no real data exists.
  - Prefers canonical enriched corpus for India–Japan when present.
- **Class**: `DashboardDataManager`
  - In-memory dashboard-side cache by country pair.

### B. Preprocessing

#### `preprocessing/preprocessor.py`

- **Classes**:
  - `TextCleaner` (HTML stripping, boilerplate/header/footer/whitespace normalization)
  - `Tokenizer` (sentence tokenization, word tokenization, stopword filtering, lemmatization, NER)
  - `Preprocessor` (pipeline wrapper)
- **Key function**:
  - `Preprocessor.process_dataframe(df, content_column='content')`
- **Added baseline columns**:
  - `cleaned`, `sentences`, `tokens`, `tokens_no_stopwords`, `lemmas`, `entities`

**Important note (stopwords vs tone):**

- The preprocessor adds diplomatic modal verbs (e.g., `shall`, `will`, `may`, `must`) to stopwords, so they may be removed from `tokens_no_stopwords` / downstream lemma-based features.
- Tone/urgency analysis operates on the **`cleaned` text** (not the lemma list), so urgency scoring still sees modal verbs.
- This split is intentional: lemmas help topics/lexicons; cleaned text preserves surface phrasing for tone heuristics.

### C. Strategic-shift scoring

#### `analysis/strategic_shift_enhanced.py`

- **Classes**:
  - `LexiconDefinitions` (economic + security lexicons)
  - `StrategicShiftAnalyzer`
- **Key functions**:
  - `calculate_category_scores(df)` → adds `economic_score`, `security_score`
  - `aggregate_by_year(df)` → yearly mean/std/min/max/count
  - `identify_crossover_point(yearly_df)`
  - `calculate_statistical_significance(df)` (t-test + effect size)
  - `audit_lexicons()` → overlap/ambiguity summary for the economic vs security lexicons
  - `generate_shift_report(df, group_by_source=False, include_lexicon_audit=True)` → full report dict + scored dataframe + yearly aggregates

### D. Tone/sentiment/urgency

#### `analysis/tone_analyzer.py`

- **Classes**:
  - `UrgencyDetector`
  - `SentimentAnalyzer`
  - `ToneAnalyzer`
- **Key function**:
  - `ToneAnalyzer.process_dataframe(df, text_column='cleaned')`
- **Added columns**:
  - `urgency_score`, `sentiment_polarity`, `sentiment_subjectivity`, `sentiment_class`, `tone_class`

**Implementation note:** polarity prefers VADER (NLTK) when available, with TextBlob retained as a fallback/proxy.

### E. Topic modeling

#### `analysis/thematic_clustering.py`

- **Classes**:
  - `TopicModeler`
  - `ThematicAnalyzer`
- **Key function**:
  - `ThematicAnalyzer.analyze_theme_evolution(df, text_column='cleaned')`
- **Outputs**:
  - `overall_themes`, `topic_weights_by_year` (comparable weights), per-document `primary_topic`

### F. Orchestration and exports

#### `pipeline.py`

- **Function**: `run_full_pipeline()`
- **Pipeline steps implemented**:
  1. Load documents
  2. Preprocess
  3. Strategic shift scoring/report
  4. Tone analysis
  5. Thematic analysis
  6. Compile and save analysis JSON
  7. Export selected sheets to Excel
- **Saved artifacts**:
- Processed JSON summary (`data/processed/analysis_results.json`)
- Data provenance report (`data/processed/data_provenance_report.json`)
- Evidence review exports (`data/processed/evidence_review_*.csv` / `.xlsx`) when enabled by the pipeline
- Spreadsheet export (`data/processed/*.xlsx`)

### G. Dashboard

#### `dashboard/app.py`

- Streamlit interface with cached data loading and analysis execution.
- Includes pages for overview, strategic trend, tone, topics, search, and statistical view.
- Includes PDF export integration (`utils/pdf_report_generator.py`).

### I. Live scraping + URL resolution (optional, pre-RAG helpers)

These modules are **separate from** the baseline pipeline and are only used when explicitly invoked:

- `scrapers/live_scrape_validator.py`
  - Live, official-source scraping validation with browser fallback, caching, budgets, provenance logs.
  - Outputs `data/raw/live_scrape_*.csv` + JSON reports.

- `scrapers/corpus_url_resolver.py`
  - Conservative URL resolver for the local corpus titles.
  - Optionally uses GDELT discovery to find historical MEA/MOFA pages, then title-validates.

- `scrapers/official_corpus_builder.py`
  - Builds a larger real-data official corpus (MEA/MOFA domains) using GDELT discovery.
  - Can promote the result into the canonical corpus.

- `scrapers/external_signals.py`
  - Optional external context signals (NewsData/NewsAPI/WorldNewsAPI + GDELT).
  - Writes `data/raw/external_signals_*.csv` + JSON run reports.

### H. Baseline tests

#### `tests/`

- `test_preprocessing.py`
- `test_analysis.py`
- `test_pipeline.py`
- `test_data_loading.py`
- `conftest.py` fixtures for shared baseline data and analyzers

---

## 3) Pre-RAG execution flow

```text
DataLoader/DataManager
  -> Preprocessor
    -> StrategicShiftAnalyzer
    -> ToneAnalyzer
    -> ThematicAnalyzer
  -> Save JSON/Excel
  -> Dashboard visualization
```

Run baseline:

```bash
python pipeline.py
streamlit run dashboard/app.py
```

Run baseline tests:

```bash
python -m pytest tests -q
```

Integration tests (end-to-end pipeline / filesystem-heavy) are marked as `integration` and are excluded by default via `pytest.ini`.

```bash
python -m pytest -m integration -q
```

---

## 4) Current robustness status (before RAG)

### Strengths

- Clear modular separation in baseline analysis components.
- Reproducible deterministic baseline techniques (lexicon, LDA, classical stats).
- Existing test suite and fixture structure.
- Dashboard already integrated with baseline analyses.

### Known blockers / gaps to fix first

1. **Environment compatibility**
   - Current failures observed with Python 3.14 + spaCy/pydantic-v1 interaction.
   - Recommendation: pin baseline runtime to Python 3.11 for stable thesis runs.

2. **Dataset provenance clarity**
   - Need explicit separation between scraped-real vs built-in sample documents in outputs and docs.

3. **Reproducibility controls**
   - `requirements.txt` is range-based (`>=`) rather than fully pinned.

4. **Evaluation rigor**
   - Need stronger baseline validation report (e.g., stable metrics across random seeds/subsets, error analysis by category).

5. **Dashboard maintainability**
   - `dashboard/app.py` is large and should be split into sections/modules for maintainability.

---

## 5) Hardening plan (do this before RAG)

### Phase A — Environment stabilization

- Lock Python version (recommended: 3.11.x).
- Freeze dependencies (`pip freeze` or curated pinned requirements for baseline).
- Re-run test suite until baseline tests are consistently green.

### Phase B — Data + provenance hardening

- Add explicit `data_source_type` field (`real_scraped`, `cached_real`, `sample_builtin`).
- Add data summary artifact per run:
  - document counts by source/year
  - missing-field counts
  - duplicate removal counts

### Phase C — Analysis validation hardening

- Add deterministic seeds for topic-modeling experiments and report variance.
- Add per-year confidence intervals for key economic/security trends.
- Add a baseline error-analysis report with examples of false positives/false negatives.

### Phase D — Reporting hardening

- Create one reproducible baseline run script for thesis figures.
- Ensure dashboard charts map exactly to saved analysis artifacts.
- Add a “Methods & Limitations” section in report outputs for transparency.

---

## 6) What “robust baseline complete” means

You are ready to proceed to RAG only when all are true:

- [ ] Baseline tests pass in a pinned environment
- [ ] One-command baseline run reproduces same metrics/figures
- [ ] Dataset provenance is explicit in outputs and docs
- [ ] Baseline statistical claims have confidence intervals/significance context
- [ ] Dashboard and exported artifacts are consistent and traceable to pipeline outputs

---

## 7) Research positioning note

For thesis framing, present this baseline as:

- **Stage 1 (Completed first):** transparent, interpretable classical NLP baseline
- **Stage 2 (After baseline hardening):** RAG extension for semantic retrieval and grounded QA

This order strengthens methodological credibility and makes RAG comparisons scientifically meaningful.

---

## 8) Implementation status addendum (Feb 2026)

This addendum records the concrete hardening and reliability work completed after the initial pre-RAG baseline spec.

### A. Corpus integrity and enrichment

- Canonical corpus workflow is active and non-destructive (`write_or_update_canonical_corpus()`), preserving the primary local file while writing enriched canonical outputs.
- Corpus has been expanded beyond the initial baseline snapshot and validated through loader-based checks.
- URL coverage has been improved through staged, review-first backfill utilities (deterministic local match, assisted lookup, curated archive seed, and approval-based promotion).

### B. Dashboard reliability and policy-grade interpretation

- Dashboard refresh logic was hardened to avoid aggressive cache clearing behavior that previously caused transient media-file errors.
- Confidence and evidence-context features were expanded for policy interpretation:
  - confidence summary and component breakdown,
  - source credibility and contradiction checks,
  - trigger-oriented decision support outputs.
- Filter-impact diagnostics and sample-size context were added to reduce misinterpretation risk from sparse slices.

### C. External context integration hardening

- External integrations are loaded through cache-token invalidation keyed to file signatures.
- RSS ingestion was hardened with stronger feed diagnostics and fallback behavior.
- Multi-source external signals are now exposed in both dashboard metrics and report context.

### D. Reporting and visualization upgrades

- PDF reporting now embeds key visualization images generated from analysis payloads.
- Live Geopolitical Pulse was added to dashboard and PDF outputs.
- Pulse representation was redesigned for interpretability:
  - 7-day activity trend (time-series),
  - recency mix (last 24h vs prior 6 days),
  - momentum label (accelerating/stable/cooling).

---

## 9) Evidence log (executed verification)

The following evidence was collected directly from the local project environment (`.venv311`) on 2026-02-20.

### A. Baseline test suite

Command:

```bash
.\.venv311\Scripts\python.exe -m pytest tests -q
```

Observed result:

- `54 passed, 3 deselected in 64.92s`

Interpretation:

- Core pre-RAG baseline test suite is currently green in the recommended Python 3.11 runtime.

### B. Canonical corpus quality snapshot

Command:

```bash
.\.venv311\Scripts\python.exe -c "import pandas as pd; from pathlib import Path; p=Path('data/raw/india_japan_documents_canonical.csv'); df=pd.read_csv(p); print({'file':str(p),'rows':len(df),'cols':len(df.columns),'missing_url':int(df['url'].isna().sum()) if 'url' in df.columns else None,'url_fill_rate':round(float(df['url'].notna().mean()*100),2) if 'url' in df.columns else None,'date_min':str(pd.to_datetime(df['date'],errors='coerce').min()),'date_max':str(pd.to_datetime(df['date'],errors='coerce').max()),'sources':df['source'].fillna('unknown').astype(str).value_counts().to_dict() if 'source' in df.columns else {}})"
```

Observed result:

- file: `data/raw/india_japan_documents_canonical.csv`
- rows: `72`
- columns: `10`
- missing_url: `51`
- url_fill_rate: `29.17%`
- date range: `2000-08-22` to `2026-02-20`
- sources: `MEA=33, MOFA=24, JETRO=8, EMBJPIN=6, MOFA_ARCHIVE=1`

Interpretation:

- Corpus expansion and source diversity are confirmed.
- URL completeness is improved but still a clear hardening target for proposal-grade provenance traceability.

### C. External integration availability snapshot

Command:

```bash
.\.venv311\Scripts\python.exe -c "from dashboard.app import load_external_integrations,_external_cache_token; ctx=load_external_integrations(cache_token=_external_cache_token()); out={k:{'available':bool(v.get('available',False)),'rows':int(v.get('rows',0) or 0),'file':v.get('file_name','')} for k,v in ctx.items() if isinstance(v,dict)}; print(out)"
```

Observed result (rows):

- `comtrade_yearly`: available, rows=`2`
- `estat_rows`: available, rows=`2000`
- `ogd_rows`: available, rows=`5`
- `rss_year`: available, rows=`6`
- `external_signals`: available, rows=`15`
- `rss_report`: available (JSON metadata artifact)

Interpretation:

- External data hooks are active and visible to the dashboard/reporting layer.

---

## 10) Results summary for research proposal

### A. Reliability outcomes achieved

1. **Baseline reproducibility (runtime-specific):** stable test pass in Python 3.11 environment.
2. **Operational robustness:** improved refresh/caching behavior and hardened RSS ingestion.
3. **Traceability improvements:** canonical corpus enrichment workflow and staged URL backfill with human approval.
4. **Decision-support maturity:** confidence-aware analytical outputs and trigger-based framing now integrated.
5. **Reporting completeness:** dashboard and PDF outputs now share aligned visualization and pulse context.

### B. Quantitative snapshot (current)

- Corpus documents: **72**
- Time span: **2000-08-22 to 2026-02-20**
- Core tests: **54 passed**
- External integrations active: **6/6 key loaders available**
- URL completeness in canonical corpus: **29.17% non-missing** (remaining improvement area)

### C. Research significance

The pre-RAG stage now provides a stronger methodological baseline for thesis comparison because:

- core classical NLP outputs are test-validated,
- data provenance workflows are explicit and improving,
- decision-relevant interpretations are confidence-scoped,
- reporting artifacts are presentation-ready for stakeholder review.

---

## 11) Updated readiness against “robust baseline complete”

Status review of Section 6 criteria:

- [x] Baseline tests pass in a pinned-like runtime context (Python 3.11 environment in local validation)
- [ ] One-command baseline run reproduces same metrics/figures (still needs explicit locked-run script + artifact checksum discipline)
- [ ] Dataset provenance is explicit in outputs and docs (partially complete; URL completeness and source-type formalization still pending)
- [ ] Baseline statistical claims have confidence intervals/significance context (partially complete; additional CI standardization still advised)
- [x] Dashboard and exported artifacts are consistent and traceable to pipeline outputs (substantially improved with shared visual payloads and pulse integration)

Recommendation before initiating full RAG comparison chapter:

1. Pin dependency versions for a frozen thesis environment.
2. Add explicit `data_source_type` in canonical outputs.
3. Add a deterministic run script that writes a run-manifest (versions, seed, input file hashes, output hashes).
