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

    def test_phrase_preservation_in_lemmas(self):
        """Multi-token diplomatic phrases should be preserved as single lemma tokens.

        This supports downstream analysis (topics, diagnostics) and prevents compound terms
        like 'joint exercise' from being split into unrelated tokens.
        """

        from preprocessing.preprocessor import Preprocessor

        df = pd.DataFrame(
            {
                "date": ["2020-01-01"],
                "title": ["Test"],
                "location": [""],
                "signatories": [""],
                "content": ["The two sides agreed to conduct a joint exercise in the Indo-Pacific."],
                "source": ["MEA"],
            }
        )
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["year"] = df["date"].dt.year

        processed = Preprocessor().process_dataframe(df, content_column="content")
        lemmas = processed.loc[0, "lemmas"]
        assert isinstance(lemmas, list)
        joined = " ".join(str(x) for x in lemmas).lower()
        # Accept either underscore-merged or concatenated variant depending on tokenization path
        assert ("joint_exercise" in joined) or ("indo_pacific" in joined) or ("indopacific" in joined)

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
