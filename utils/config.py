"""Project paths and settings.

NOTE: API keys and secrets must be provided via environment variables (optionally via a local
.env file). Never hardcode secrets in source code.
"""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Load environment variables from .env if available
if load_dotenv is not None:
    load_dotenv(PROJECT_ROOT / ".env", override=False)

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# File paths
MEA_DOCUMENTS_CSV = RAW_DATA_DIR / "mea_documents.csv"
MOFA_DOCUMENTS_CSV = RAW_DATA_DIR / "mofa_documents.csv"
PROCESSED_DOCUMENTS_CSV = PROCESSED_DATA_DIR / "processed_documents.csv"
ANALYSIS_RESULTS_JSON = PROCESSED_DATA_DIR / "analysis_results.json"
PROVENANCE_REPORT_JSON = PROCESSED_DATA_DIR / "data_provenance_report.json"

# NLP Configuration
SPACY_MODEL = "en_core_web_sm"
MAX_DOCUMENT_LENGTH = 50000  # Maximum characters per document

# Runtime guidance (pre-RAG baseline)
RECOMMENDED_PYTHON_MAJOR = 3
RECOMMENDED_PYTHON_MINOR = 11

# Analysis Configuration
ECONOMIC_LEXICON_FILE = PROJECT_ROOT / "utils" / "economic_lexicon.txt"
SECURITY_LEXICON_FILE = PROJECT_ROOT / "utils" / "security_lexicon.txt"

# Date range for analysis
START_YEAR = 2000
END_YEAR = 2025

# Topic Modeling Configuration
N_TOPICS = 5
LDA_MAX_ITER = 20

# Web Scraping Configuration
REQUEST_TIMEOUT = 10  # seconds
REQUEST_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
}

# External signals (optional)
# Provide these via environment variables or a local .env file.
NEWSDATA_API_KEY = os.getenv("NEWSDATA_API_KEY", "").strip()
NEWSAPI_API_KEY = os.getenv("NEWSAPI_API_KEY", "").strip()
WORLDNEWS_API_KEY = os.getenv("WORLDNEWS_API_KEY", "").strip()

# UN Comtrade (optional)
COMTRADE_PRIMARY_KEY = os.getenv("COMTRADE_PRIMARY_KEY", "").strip()
COMTRADE_SECONDARY_KEY = os.getenv("COMTRADE_SECONDARY_KEY", "").strip()

# data.gov.in (OGD India) (optional)
DATA_GOV_IN_API_KEY = os.getenv("DATA_GOV_IN_API_KEY", "").strip()

# e-Stat Japan (optional)
ESTAT_APP_ID = os.getenv("ESTAT_APP_ID", "").strip()

# ACLED (optional; user must register)
ACLED_EMAIL = os.getenv("ACLED_EMAIL", "").strip()
ACLED_ACCESS_KEY = os.getenv("ACLED_ACCESS_KEY", "").strip()

# Logging Configuration
LOG_LEVEL = "INFO"
LOG_FILE = PROJECT_ROOT / "logs" / "diplomatic_barometer.log"

# Create necessary directories
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, PROJECT_ROOT / "logs"]:
    directory.mkdir(parents=True, exist_ok=True)


def get_config():
    """Return configuration as dictionary"""
    return {
        'project_root': str(PROJECT_ROOT),
        'data_dir': str(DATA_DIR),
        'raw_data_dir': str(RAW_DATA_DIR),
        'processed_data_dir': str(PROCESSED_DATA_DIR),
        'spacy_model': SPACY_MODEL,
        'start_year': START_YEAR,
        'end_year': END_YEAR,
        'n_topics': N_TOPICS,
        'lda_max_iter': LDA_MAX_ITER
    }


def get_external_api_status() -> dict:
    """Return a non-sensitive view of which optional external integrations are configured."""

    def _set(name: str) -> bool:
        return bool(os.getenv(name, "").strip())

    return {
        "newsdata_configured": _set("NEWSDATA_API_KEY"),
        "newsapi_configured": _set("NEWSAPI_API_KEY"),
        "worldnews_configured": _set("WORLDNEWS_API_KEY"),
        "acled_configured": _set("ACLED_ACCESS_KEY") and _set("ACLED_EMAIL"),
        "gdelt_configured": True,  # no key required
        "comtrade_configured": _set("COMTRADE_PRIMARY_KEY") or _set("COMTRADE_SECONDARY_KEY"),
        "data_gov_in_configured": _set("DATA_GOV_IN_API_KEY"),
        "estat_configured": _set("ESTAT_APP_ID"),
    }


if __name__ == "__main__":
    config = get_config()
    for key, value in config.items():
        print(f"{key}: {value}")
