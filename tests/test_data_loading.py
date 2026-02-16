"""
Tests for data loading module
"""

import pytest
import pandas as pd


class TestDataLoader:
    """Test the DataLoader class"""

    def test_load_sample_data_returns_dataframe(self, sample_df):
        assert isinstance(sample_df, pd.DataFrame)

    def test_sample_data_has_50_plus_documents(self, sample_df):
        assert len(sample_df) >= 50, f"Expected 50+ documents, got {len(sample_df)}"

    def test_required_columns_exist(self, sample_df):
        required = ['date', 'title', 'location', 'signatories', 'content', 'source', 'year']
        for col in required:
            assert col in sample_df.columns, f"Missing column: {col}"

    def test_date_range_2000_to_2024(self, sample_df):
        assert sample_df['year'].min() == 2000
        assert sample_df['year'].max() == 2024

    def test_sources_are_mea_and_mofa(self, sample_df):
        sources = set(sample_df['source'].unique())
        assert sources == {'MEA', 'MOFA'}

    def test_both_sources_have_documents(self, sample_df):
        mea_count = len(sample_df[sample_df['source'] == 'MEA'])
        mofa_count = len(sample_df[sample_df['source'] == 'MOFA'])
        assert mea_count >= 10, f"Too few MEA docs: {mea_count}"
        assert mofa_count >= 10, f"Too few MOFA docs: {mofa_count}"

    def test_no_empty_content(self, sample_df):
        empty = sample_df['content'].apply(lambda x: len(str(x).strip()) == 0).sum()
        assert empty == 0, f"Found {empty} documents with empty content"

    def test_no_empty_titles(self, sample_df):
        empty = sample_df['title'].apply(lambda x: len(str(x).strip()) == 0).sum()
        assert empty == 0, f"Found {empty} documents with empty titles"

    def test_dates_are_datetime(self, sample_df):
        assert pd.api.types.is_datetime64_any_dtype(sample_df['date'])

    def test_load_combined_data_fallback(self):
        """When no CSV files exist, load_combined_data should fall back to sample data"""
        from scrapers.data_loader import DataLoader
        loader = DataLoader()
        df = loader.load_combined_data()  # No files → fallback
        assert len(df) >= 50


class TestCrawlers:
    """Test MEA and MOFA crawlers"""

    def test_mea_crawler_imports(self):
        from scrapers.mea_crawler import MEACrawler
        crawler = MEACrawler()
        assert crawler is not None

    def test_mofa_crawler_imports(self):
        from scrapers.mofa_crawler import MOFACrawler
        crawler = MOFACrawler()
        assert crawler is not None

    def test_mea_crawler_fetch_returns_dataframe(self):
        from scrapers.mea_crawler import MEACrawler
        crawler = MEACrawler()
        df = crawler.fetch_all_documents()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_mofa_crawler_fetch_returns_dataframe(self):
        from scrapers.mofa_crawler import MOFACrawler
        crawler = MOFACrawler()
        df = crawler.fetch_all_documents()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_mea_crawler_search(self):
        from scrapers.mea_crawler import MEACrawler
        crawler = MEACrawler()
        docs = crawler.search_bilateral_documents()
        assert isinstance(docs, list)
        assert len(docs) > 0

    def test_mofa_crawler_search(self):
        from scrapers.mofa_crawler import MOFACrawler
        crawler = MOFACrawler()
        docs = crawler.search_bilateral_documents()
        assert isinstance(docs, list)
        assert len(docs) > 0


class TestRealDataLoader:
    """Test the real data loader"""

    def test_load_all_documents(self):
        from utils.real_data_loader import RealDataLoader
        loader = RealDataLoader()
        df = loader.load_all_documents()
        assert isinstance(df, pd.DataFrame)
        assert len(df) >= 50

    def test_load_mea_documents(self):
        from utils.real_data_loader import RealDataLoader
        loader = RealDataLoader()
        df = loader.load_mea_documents()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert all(df['source'] == 'MEA')

    def test_load_mofa_documents(self):
        from utils.real_data_loader import RealDataLoader
        loader = RealDataLoader()
        df = loader.load_mofa_documents()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert all(df['source'] == 'MOFA')
