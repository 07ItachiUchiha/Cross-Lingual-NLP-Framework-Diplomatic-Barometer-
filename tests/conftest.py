"""Shared pytest fixtures"""

import sys
import os
import pytest
import pandas as pd

# Ensure project root is on path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)


@pytest.fixture
def sample_df():
    """Deterministic synthetic diplomatic dataset for unit tests.

    Unit tests should be runnable in a clean clone without requiring on-disk CSVs.
    """

    topic_buckets = [
        "infrastructure rail connectivity logistics",
        "energy solar hydrogen climate",
        "technology semiconductors digital innovation",
        "trade investment supply chains",
        "maritime security coast guard",
        "defense joint exercise interoperability",
        "education scholarships research",
        "health pandemic vaccine cooperation",
        "finance yen loan development assistance",
        "multilateral diplomacy indo pacific",
    ]

    rows = []
    for year in range(2000, 2025):
        for source in ("MEA", "MOFA"):
            # 2 docs per source per year => 100 docs total (2000–2024 inclusive)
            for i in range(2):
                bucket = topic_buckets[(year + i + (0 if source == "MEA" else 1)) % len(topic_buckets)]
                rows.append(
                    {
                        "date": f"{year}-01-{(i + 1):02d}",
                        "title": f"{source} Document {year}-{i}",
                        "location": "Test Location",
                        "signatories": "Test Signatory",
                        "content": (
                            f"This official document covers {bucket}. "
                            "The countries discussed cooperation, coordination, and dialogue."
                        ),
                        "source": source,
                    }
                )

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["year"] = df["date"].dt.year
    return df


@pytest.fixture
def real_df():
    """Load the real diplomatic dataset from disk (integration fixture)."""
    from scrapers.data_loader import DataLoader
    loader = DataLoader()
    return loader.load_combined_data()


@pytest.fixture
def processed_df(sample_df):
    """Preprocessed version of the sample dataset"""
    from preprocessing.preprocessor import Preprocessor
    preprocessor = Preprocessor()
    return preprocessor.process_dataframe(sample_df, content_column='content')


@pytest.fixture
def shift_analyzer():
    """Strategic shift analyzer (enhanced version)"""
    from analysis.strategic_shift_enhanced import StrategicShiftAnalyzer
    return StrategicShiftAnalyzer()


@pytest.fixture
def tone_analyzer():
    """Tone analyzer instance"""
    from analysis.tone_analyzer import ToneAnalyzer
    return ToneAnalyzer()


@pytest.fixture
def thematic_analyzer():
    """Thematic analyzer instance"""
    from analysis.thematic_clustering import ThematicAnalyzer
    return ThematicAnalyzer(n_topics=5)
