"""
Tests for preprocessing module
"""

import pytest
import pandas as pd


class TestPreprocessor:
    """Test the text preprocessor"""

    def test_import(self):
        from preprocessing.preprocessor import Preprocessor
        preprocessor = Preprocessor()
        assert preprocessor is not None

    def test_process_dataframe(self, sample_df):
        from preprocessing.preprocessor import Preprocessor
        preprocessor = Preprocessor()
        result = preprocessor.process_dataframe(sample_df, content_column='content')
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_df)

    def test_cleaned_column_created(self, processed_df):
        assert 'cleaned' in processed_df.columns

    def test_lemmas_column_created(self, processed_df):
        assert 'lemmas' in processed_df.columns

    def test_cleaned_not_empty(self, processed_df):
        non_empty = processed_df['cleaned'].apply(lambda x: len(str(x).strip()) > 0).sum()
        assert non_empty == len(processed_df), "Some documents have empty cleaned text"

    def test_preserves_original_columns(self, processed_df):
        for col in ['date', 'title', 'location', 'content', 'source', 'year']:
            assert col in processed_df.columns
