"""Tests for the supervised binary validation baseline."""

import pandas as pd


class TestBinaryClassifier:
    def test_make_binary_training_frame_filters_cultural(self):
        from evaluation.binary_classifier import make_binary_training_frame

        df = pd.DataFrame(
            {
                "text": ["trade and investment", "joint exercise", "cultural exchange"],
                "label": ["Economic", "Security", "Cultural"],
            }
        )

        out = make_binary_training_frame(df)
        assert len(out) == 2
        assert set(out["label"].tolist()) == {"Economic", "Security"}

    def test_evaluate_binary_classifier_on_golden_dataset(self):
        from evaluation.golden_dataset import GoldenDataset
        from evaluation.binary_classifier import evaluate_binary_classifier

        report = evaluate_binary_classifier(GoldenDataset().to_dataframe())
        assert report["available"] is True
        assert report["samples_used"] == 40
        assert report["cv_folds"] >= 2
        assert 0.0 <= report["macro_f1"] <= 1.0

    def test_compare_binary_ml_with_lexicon_returns_document_comparison(self, processed_df, shift_analyzer):
        from evaluation.binary_classifier import compare_binary_ml_with_lexicon

        _, scored_df, _ = shift_analyzer.generate_shift_report(processed_df)
        report = compare_binary_ml_with_lexicon(scored_df, text_col="cleaned")

        assert report["available"] is True
        assert "document_comparison" in report
        assert report["document_comparison"]["documents_compared"] == len(scored_df)
        assert 0.0 <= report["document_comparison"]["agreement_rate"] <= 1.0

    def test_compare_binary_ml_with_lexicon_handles_empty_df(self):
        from evaluation.binary_classifier import compare_binary_ml_with_lexicon

        report = compare_binary_ml_with_lexicon(pd.DataFrame())
        assert report["available"] is False
        assert "warning" in report
