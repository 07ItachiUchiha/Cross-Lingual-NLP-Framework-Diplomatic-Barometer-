"""
Tests for pipeline and integration
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestPipeline:
    """Test the main analysis pipeline"""

    def test_pipeline_imports(self):
        """Pipeline should import without errors"""
        import pipeline
        assert hasattr(pipeline, 'run_full_pipeline')

    def test_pipeline_uses_enhanced_analyzer(self):
        """Pipeline must use the enhanced analyzer, not basic"""
        import inspect
        import pipeline
        source = inspect.getsource(pipeline)
        assert 'strategic_shift_enhanced' in source, \
            "pipeline.py should import from strategic_shift_enhanced"

    def test_full_pipeline_runs(self):
        """Full pipeline should complete without errors"""
        from pipeline import run_full_pipeline
        result = run_full_pipeline()
        assert result['status'] == 'success'
        assert 'results' in result
        assert 'data' in result

    def test_pipeline_results_structure(self):
        """Pipeline results should have expected keys"""
        from pipeline import run_full_pipeline
        result = run_full_pipeline()
        results = result['results']
        assert 'metadata' in results
        assert 'strategic_shift' in results
        assert 'tone_and_sentiment' in results
        assert 'themes' in results

    def test_pipeline_50_documents(self):
        """Pipeline should process 50 documents"""
        from pipeline import run_full_pipeline
        result = run_full_pipeline()
        assert result['results']['metadata']['total_documents'] >= 50


class TestDataIntegration:
    """Test the data integration module"""

    def test_dashboard_data_manager(self):
        from data_integration import DashboardDataManager
        manager = DashboardDataManager()
        df = manager.get_data_for_pair(('india', 'japan'))
        assert len(df) > 0

    def test_country_pair_loader(self):
        from data_integration import CountryPairDataLoader
        loader = CountryPairDataLoader()
        df = loader.load_country_pair_data(('india', 'japan'))
        assert len(df) > 0
        assert 'date' in df.columns
        assert 'content' in df.columns


class TestCountryConfig:
    """Test country configuration"""

    def test_countries_config(self):
        from utils.country_config import COUNTRIES
        assert 'india' in COUNTRIES
        assert 'japan' in COUNTRIES

    def test_get_country_name(self):
        from utils.country_config import get_country_name
        assert get_country_name('india') == 'India'
        assert get_country_name('japan') == 'Japan'

    def test_get_ministry_name(self):
        from utils.country_config import get_ministry_name
        assert 'MEA' in get_ministry_name('india') or 'Ministry' in get_ministry_name('india')
        assert 'MOFA' in get_ministry_name('japan') or 'Ministry' in get_ministry_name('japan')


class TestPDFGenerator:
    """Test PDF report generator"""

    def test_import(self):
        from utils.pdf_report_generator import PDFReportGenerator
        gen = PDFReportGenerator()
        assert gen is not None

    def test_reportlab_available(self):
        from utils.pdf_report_generator import REPORTLAB_AVAILABLE
        # Just check the flag exists; it may be True or False depending on install
        assert isinstance(REPORTLAB_AVAILABLE, bool)
