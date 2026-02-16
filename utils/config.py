"""Project paths and settings"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# File paths
MEA_DOCUMENTS_CSV = RAW_DATA_DIR / "mea_documents.csv"
MOFA_DOCUMENTS_CSV = RAW_DATA_DIR / "mofa_documents.csv"
PROCESSED_DOCUMENTS_CSV = PROCESSED_DATA_DIR / "processed_documents.csv"
ANALYSIS_RESULTS_JSON = PROCESSED_DATA_DIR / "analysis_results.json"

# NLP Configuration
SPACY_MODEL = "en_core_web_sm"
MAX_DOCUMENT_LENGTH = 50000  # Maximum characters per document

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


if __name__ == "__main__":
    config = get_config()
    for key, value in config.items():
        print(f"{key}: {value}")
