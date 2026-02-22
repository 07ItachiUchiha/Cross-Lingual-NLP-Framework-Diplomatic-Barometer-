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

    @pytest.mark.integration
    def test_full_pipeline_runs(self, tmp_path, monkeypatch):
        """Full pipeline should complete without errors (integration)."""
        import pandas as pd

        raw_dir = tmp_path / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)

        rows = []
        # >= 60 docs to satisfy pipeline's minimum expectations
        for year in range(2000, 2025):
            for source in ("MEA", "MOFA"):
                rows.append(
                    {
                        "date": f"{year}-01-01",
                        "title": f"{source} Joint Statement {year}",
                        "location": "Test Location",
                        "signatories": "Test Signatory",
                        "content": "Yen loan and infrastructure cooperation alongside maritime security and joint exercise.",
                        "source": source,
                    }
                )

        df = pd.DataFrame(rows)
        csv_path = raw_dir / "india_japan_documents.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8")

        monkeypatch.setenv("DIPLOMATIC_BAROMETER_DATA_DIR", str(raw_dir))

        from pipeline import run_full_pipeline
        result = run_full_pipeline()
        assert result['status'] == 'success'
        assert 'results' in result
        assert 'data' in result

    @pytest.mark.integration
    def test_pipeline_results_structure(self, tmp_path, monkeypatch):
        import pandas as pd

        raw_dir = tmp_path / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for year in range(2000, 2025):
            for source in ("MEA", "MOFA"):
                rows.append(
                    {
                        "date": f"{year}-01-01",
                        "title": f"{source} Document {year}",
                        "location": "Test Location",
                        "signatories": "Test Signatory",
                        "content": "Trade and investment with maritime security cooperation.",
                        "source": source,
                    }
                )
        pd.DataFrame(rows).to_csv(raw_dir / "india_japan_documents.csv", index=False, encoding="utf-8")
        monkeypatch.setenv("DIPLOMATIC_BAROMETER_DATA_DIR", str(raw_dir))

        from pipeline import run_full_pipeline
        result = run_full_pipeline()
        results = result['results']
        assert 'metadata' in results
        assert 'strategic_shift' in results
        assert 'tone_and_sentiment' in results
        assert 'themes' in results

    @pytest.mark.integration
    def test_pipeline_50_documents(self, tmp_path, monkeypatch):
        import pandas as pd

        raw_dir = tmp_path / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for year in range(2000, 2025):
            for source in ("MEA", "MOFA"):
                rows.append(
                    {
                        "date": f"{year}-01-01",
                        "title": f"{source} Document {year}",
                        "location": "Test Location",
                        "signatories": "Test Signatory",
                        "content": "Infrastructure projects and maritime security collaboration.",
                        "source": source,
                    }
                )
        pd.DataFrame(rows).to_csv(raw_dir / "india_japan_documents.csv", index=False, encoding="utf-8")
        monkeypatch.setenv("DIPLOMATIC_BAROMETER_DATA_DIR", str(raw_dir))

        from pipeline import run_full_pipeline
        result = run_full_pipeline()
        assert result['results']['metadata']['total_documents'] >= 50


class TestDataIntegration:
    """Test the data integration module"""

    def test_dashboard_data_manager(self, tmp_path, monkeypatch):
        import pandas as pd

        raw_dir = tmp_path / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for year in range(2000, 2025):
            for source in ("MEA", "MOFA"):
                rows.append(
                    {
                        "date": f"{year}-01-01",
                        "title": f"{source} Document {year}",
                        "location": "Test Location",
                        "signatories": "Test Signatory",
                        "content": "Trade and investment with maritime security cooperation.",
                        "source": source,
                    }
                )
        pd.DataFrame(rows).to_csv(raw_dir / "india_japan_documents.csv", index=False, encoding="utf-8")
        monkeypatch.setenv("DIPLOMATIC_BAROMETER_DATA_DIR", str(raw_dir))

        from data_integration import DashboardDataManager
        manager = DashboardDataManager()
        df = manager.get_data_for_pair(('india', 'japan'))
        assert len(df) > 0

    def test_country_pair_loader(self, tmp_path, monkeypatch):
        import pandas as pd

        raw_dir = tmp_path / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for year in range(2000, 2025):
            for source in ("MEA", "MOFA"):
                rows.append(
                    {
                        "date": f"{year}-01-01",
                        "title": f"{source} Document {year}",
                        "location": "Test Location",
                        "signatories": "Test Signatory",
                        "content": "Infrastructure and maritime security cooperation.",
                        "source": source,
                    }
                )
        pd.DataFrame(rows).to_csv(raw_dir / "india_japan_documents.csv", index=False, encoding="utf-8")
        monkeypatch.setenv("DIPLOMATIC_BAROMETER_DATA_DIR", str(raw_dir))

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
