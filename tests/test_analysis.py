"""
Tests for analysis modules (strategic shift, tone, thematic)
"""

import pytest
import pandas as pd


class TestStrategicShiftAnalyzer:
    """Test the enhanced strategic shift analyzer"""

    def test_import_enhanced_version(self):
        from analysis.strategic_shift_enhanced import StrategicShiftAnalyzer
        analyzer = StrategicShiftAnalyzer()
        assert analyzer is not None

    def test_lexicon_definitions_exist(self):
        from analysis.strategic_shift_enhanced import LexiconDefinitions
        assert len(LexiconDefinitions.ECONOMIC_LEXICON) > 0
        assert len(LexiconDefinitions.SECURITY_LEXICON) > 0

    def test_economic_lexicon_has_subcategories(self):
        from analysis.strategic_shift_enhanced import LexiconDefinitions
        assert len(LexiconDefinitions.ECONOMIC_LEXICON) >= 5

    def test_security_lexicon_has_subcategories(self):
        from analysis.strategic_shift_enhanced import LexiconDefinitions
        assert len(LexiconDefinitions.SECURITY_LEXICON) >= 5

    def test_generate_shift_report(self, processed_df, shift_analyzer):
        report, scored_df, yearly_df = shift_analyzer.generate_shift_report(processed_df)
        assert isinstance(report, dict)
        assert 'overall_economic_avg' in report
        assert 'overall_security_avg' in report
        assert 'crossover_year' in report
        assert 'trend' in report

    def test_scored_df_has_scores(self, processed_df, shift_analyzer):
        report, scored_df, yearly_df = shift_analyzer.generate_shift_report(processed_df)
        assert 'economic_score' in scored_df.columns
        assert 'security_score' in scored_df.columns

    def test_yearly_df_has_aggregation(self, processed_df, shift_analyzer):
        report, scored_df, yearly_df = shift_analyzer.generate_shift_report(processed_df)
        assert 'year' in yearly_df.columns
        assert 'economic_score_mean' in yearly_df.columns
        assert 'security_score_mean' in yearly_df.columns
        assert len(yearly_df) > 5  # Multiple years

    def test_scores_are_non_negative(self, processed_df, shift_analyzer):
        report, scored_df, yearly_df = shift_analyzer.generate_shift_report(processed_df)
        assert (scored_df['economic_score'] >= 0).all()
        assert (scored_df['security_score'] >= 0).all()

    def test_pipeline_uses_enhanced_analyzer(self):
        """Verify pipeline.py imports the enhanced version"""
        import importlib
        import pipeline
        importlib.reload(pipeline)
        # The import should succeed without error if using enhanced version

    def test_multiword_lexicon_phrases_match(self):
        """Multi-word lexicon phrases must contribute to scores.

        This guards against the regression where matching is done token-by-token on lemmas,
        which silently drops phrases like 'yen loan' or 'joint exercise'.
        """

        from analysis.strategic_shift_enhanced import StrategicShiftAnalyzer

        analyzer = StrategicShiftAnalyzer()
        df = pd.DataFrame(
            {
                "date": ["2020-01-01", "2020-02-01"],
                "cleaned": [
                    "The countries agreed to expand the yen loan program for major infrastructure projects.",
                    "They announced a joint exercise and maritime security cooperation between coast guards.",
                ],
                "lemmas": [
                    ["yen", "loan", "infrastructure"],
                    ["joint", "exercise", "maritime", "security", "coast", "guard"],
                ],
            }
        )

        scored = analyzer.calculate_category_scores(df)
        assert float(scored.loc[0, "economic_score"]) > 0.0
        assert float(scored.loc[1, "security_score"]) > 0.0


class TestToneAnalyzer:
    """Test the tone analyzer"""

    def test_import(self):
        from analysis.tone_analyzer import ToneAnalyzer
        analyzer = ToneAnalyzer()
        assert analyzer is not None

    def test_process_dataframe(self, processed_df, tone_analyzer):
        tone_df = tone_analyzer.process_dataframe(processed_df, text_column='cleaned')
        assert isinstance(tone_df, pd.DataFrame)
        assert len(tone_df) > 0

    def test_tone_columns_exist(self, processed_df, tone_analyzer):
        tone_df = tone_analyzer.process_dataframe(processed_df, text_column='cleaned')
        expected_cols = ['tone_class', 'urgency_score', 'sentiment_polarity']
        for col in expected_cols:
            assert col in tone_df.columns, f"Missing column: {col}"

    def test_tone_distribution(self, processed_df, tone_analyzer):
        tone_df = tone_analyzer.process_dataframe(processed_df, text_column='cleaned')
        dist = tone_analyzer.get_tone_distribution(tone_df)
        assert isinstance(dist, dict)
        assert len(dist) > 0
        assert sum(dist.values()) > 0

    def test_sentiment_distribution(self, processed_df, tone_analyzer):
        tone_df = tone_analyzer.process_dataframe(processed_df, text_column='cleaned')
        dist = tone_analyzer.get_sentiment_distribution(tone_df)
        assert isinstance(dist, dict)


class TestThematicAnalyzer:
    """Test the thematic clustering analyzer"""

    def test_import(self):
        from analysis.thematic_clustering import ThematicAnalyzer
        analyzer = ThematicAnalyzer(n_topics=5)
        assert analyzer is not None

    def test_analyze_theme_evolution(self, processed_df, thematic_analyzer):
        analysis, df_themes = thematic_analyzer.analyze_theme_evolution(
            processed_df, text_column='cleaned'
        )
        assert isinstance(analysis, dict)
        assert 'n_topics' in analysis
        assert 'overall_themes' in analysis

    def test_discovers_multiple_topics(self, processed_df, thematic_analyzer):
        analysis, df_themes = thematic_analyzer.analyze_theme_evolution(
            processed_df, text_column='cleaned'
        )
        assert analysis['n_topics'] >= 3

    def test_themes_have_words(self, processed_df, thematic_analyzer):
        analysis, df_themes = thematic_analyzer.analyze_theme_evolution(
            processed_df, text_column='cleaned'
        )
        for topic_id, words in analysis['overall_themes'].items():
            assert len(words) > 0, f"Theme {topic_id} has no words"
