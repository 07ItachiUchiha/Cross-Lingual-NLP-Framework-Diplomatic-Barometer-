"""Hooks up scraped data to the analysis modules and dashboard"""

import sys
import os
from pathlib import Path
import pandas as pd
import logging
from typing import Tuple, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.country_config import COUNTRY_PAIRS, COUNTRIES
from scrapers.data_loader import DataLoader
from scrapers.mea_crawler import MEACrawler
from scrapers.mofa_crawler import MOFACrawler
from scrapers.france_mfa_crawler_enhanced import FranceMFACrawler


class CountryPairDataLoader:
    """Load data for specific country pairs"""
    
    def __init__(self):
        self.cache_dir = PROJECT_ROOT / 'data' / 'raw'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir = PROJECT_ROOT / 'data' / 'processed'
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def get_cache_path(self, country_pair: Tuple[str, str]) -> Path:
        """Get cache file path for a country pair"""
        pair_name = f"{country_pair[0]}_{country_pair[1]}"
        default_path = self.cache_dir / f"{pair_name}_documents.csv"

        # Prefer canonical enriched corpus when available (keeps original raw file untouched)
        canonical = self.cache_dir / f"{pair_name}_documents_canonical.csv"
        if canonical.exists():
            return canonical

        return default_path
    
    def load_country_pair_data(self,
                               country_pair: Tuple[str, str],
                               use_cache: bool = True,
                               force_refresh: bool = False) -> pd.DataFrame:
        """
        Load data for a country pair
        
        Priority:
        1. Check cache (use if available and use_cache=True)
        2. Try to scrape real data
        3. Fail fast if real data is unavailable
        
        Args:
            country_pair: Tuple of (country1, country2)
            use_cache: Use cached data if available
            force_refresh: Force re-scraping even if cache exists
        
        Returns:
            DataFrame with documents
        """
        cache_path = self.get_cache_path(country_pair)
        
        # Try cache first
        if use_cache and cache_path.exists() and not force_refresh:
            logger.info(f"Loading cached data for {country_pair[0]}-{country_pair[1]}")
            try:
                df = pd.read_csv(cache_path)
                df['date'] = pd.to_datetime(df['date'])
                logger.info(f"✓ Loaded {len(df)} documents from cache")
                return df
            except Exception as e:
                logger.warning(f"Failed to load cache: {str(e)}")
        
        # Try to load real data
        logger.info(f"Attempting to load real data for {country_pair}")
        df = self._scrape_country_pair(country_pair)
        
        if df is not None and len(df) > 0:
            logger.info(f"✓ Scraped {len(df)} documents for {country_pair}")
            # Save to cache
            try:
                df.to_csv(cache_path, index=False)
                logger.info(f"✓ Cached data to {cache_path}")
            except Exception as e:
                logger.warning(f"Failed to cache data: {str(e)}")
            return df
        
        raise RuntimeError(
            f"No real data available for {country_pair}. "
            "Please provide cached CSV or ensure crawlers return real documents."
        )
    
    def _scrape_country_pair(self, country_pair: Tuple[str, str]) -> Optional[pd.DataFrame]:
        """
        Load data for a country pair using crawlers.
        
        For India-Japan: Uses MEA + MOFA crawlers to get the full 50-document dataset.
        """
        try:
            dfs = []
            
            for country in country_pair:
                if country == 'india':
                    logger.info("  Loading India MEA documents...")
                    crawler = MEACrawler()
                    df = crawler.fetch_all_documents()
                    if df is not None and len(df) > 0:
                        dfs.append(df)
                
                elif country == 'japan':
                    logger.info("  Loading Japan MOFA documents...")
                    crawler = MOFACrawler()
                    df = crawler.fetch_all_documents()
                    if df is not None and len(df) > 0:
                        dfs.append(df)

                elif country == 'france':
                    logger.info("  Loading France MEAE documents...")
                    crawler = FranceMFACrawler()
                    df = crawler.fetch_all_documents(country_pair=country_pair)
                    if df is not None and len(df) > 0:
                        dfs.append(df)
            
            if dfs:
                combined_df = pd.concat(dfs, ignore_index=True)
                # Remove duplicates based on title and date
                combined_df = combined_df.drop_duplicates(subset=['title', 'date'], keep='first')
                combined_df['date'] = pd.to_datetime(combined_df['date'])
                if 'year' not in combined_df.columns:
                    combined_df['year'] = combined_df['date'].dt.year
                logger.info(f"Combined {len(combined_df)} unique documents")
                return combined_df
            
            logger.warning("No documents loaded from crawlers")
            return None
        
        except Exception as e:
            logger.error(f"Error loading data for {country_pair}: {str(e)}")
            return None
    
class DashboardDataManager:
    """Manages data loading for the dashboard"""
    
    def __init__(self):
        self.loader = CountryPairDataLoader()
        self.cache = {}
    
    def get_data_for_pair(self, country_pair: Tuple[str, str], force_refresh: bool = False) -> pd.DataFrame:
        """Get data for a country pair with caching"""
        pair_key = f"{country_pair[0]}-{country_pair[1]}"
        
        if pair_key in self.cache and not force_refresh:
            logger.info(f"Using cached data for {pair_key}")
            return self.cache[pair_key]
        
        df = self.loader.load_country_pair_data(country_pair, force_refresh=force_refresh)
        self.cache[pair_key] = df
        return df
    
    def get_all_available_pairs(self) -> list:
        """Get all country pairs with available data"""
        available = []
        for pair in COUNTRY_PAIRS:
            cache_path = self.loader.get_cache_path(pair)
            if cache_path.exists():
                available.append(pair)
        
        if not available:
            logger.info("No cached data found for any country pair")
        
        return available


def test_data_loading():
    """Test data loading for various country pairs"""
    logger.info("="*70)
    logger.info("Testing Country Pair Data Loading")
    logger.info("="*70)
    
    manager = DashboardDataManager()
    
    # Test loading for each country pair
    test_pairs = [
        ('india', 'japan'),
        ('india', 'france'),
    ]
    
    for pair in test_pairs:
        logger.info(f"\nLoading data for {pair[0]}-{pair[1]}...")
        try:
            df = manager.get_data_for_pair(pair)
            logger.info(f"✓ Loaded {len(df)} documents")
            logger.info(f"  Columns: {list(df.columns)}")
            logger.info(f"  Date range: {df['date'].min()} to {df['date'].max()}")
        except Exception as e:
            logger.error(f"✗ Failed to load data: {str(e)}")
    
    logger.info("\n" + "="*70)
    logger.info("Data Loading Test Complete")
    logger.info("="*70)


if __name__ == "__main__":
    test_data_loading()
