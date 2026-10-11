# Archive of Legacy & Non-Research Components

This directory preserves non-core tools, enterprise utilities, and transient run artifacts that were decoupled from the active research pipeline to align strictly with the Master's Thesis Proposal:
*"Temporal Document Analysis with Retrieval-Augmented Generation: A Domain-Specific Long-Form Document Analysis Framework"*.

---

## Contents

### 1. `legacy_scrapers/`
* `external_signals.py` — Multi-provider news aggregator (NewsData, NewsAPI, WorldNewsAPI).
* `comtrade_ingestion.py` — UN Comtrade trade volume ingestion.
* `estat_ingestion.py` — Japanese government statistics (e-Stat) API crawler.
* `data_gov_in_ingestion.py` — Indian open government data (Data.gov.in) API crawler.
* `rss_ingestion.py` & `rss_summary.py` — Real-time RSS feeds ingestion and temporal aggregators.

**Rationale for archiving:** The thesis scope is strictly bounded to official, long-form bilateral diplomatic texts (joint statements, communiqués, declarations). External media and macroeconomic trade statistics introduce confounding variables and noisy text that dilutes the core study of diplomatic language and semantic retrieval.

---

### 2. `legacy_enterprise/`
* `pdf_report_generator.py` — ReportLab executive briefing PDF generator.
* `weekly_digest.py` — Weekly cron digest summarizer.
* `pilot_validation.py` — Pilot user study checklists for NGO/Think-tank user roles.
* `quality_audit.py` — Corporate quarterly data hygiene auditor.
* `policy_triggers.py` & `decision_options.py` — Heuristic decision matrices and trigger rules.
* `url_backfill.py` — Automated URL backfilling and fuzzy patch generator.

**Rationale for archiving:** These modules are SaaS/product-oriented features that add engineering complexity without contributing empirical answers to the thesis research questions (local vs. cloud RAG, temporal consistency, polysemy resolution, and RAGAS benchmarks).

---

### 3. `legacy_data_artifacts/`
* Transient files (`evidence_review_*.csv`, `url_backfill_*.csv`, `comtrade_*.csv`, `estat_*.csv`, `rss_*.csv`, `external_signals_*.csv`, and legacy scraping reports).

**Rationale for archiving:** Keeps `data/raw/` and `data/processed/` clean and focused solely on canonical bilateral corpus files (`india_japan_documents_canonical.csv`, `india_france_documents_canonical.csv`) and validated research exports.
