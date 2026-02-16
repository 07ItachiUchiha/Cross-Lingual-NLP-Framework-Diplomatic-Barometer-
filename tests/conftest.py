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
    """Load the full 50-document sample dataset"""
    from scrapers.data_loader import DataLoader
    loader = DataLoader()
    return loader.load_sample_data()


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
