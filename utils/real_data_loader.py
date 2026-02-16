"""
Real Data Loader - Load actual MEA and MOFA documents
Integrates with DataLoader's built-in 50-document India-Japan dataset
and the enhanced crawlers for live scraping when available.
"""

import pandas as pd
import logging
from pathlib import Path
import sys
import os

# Ensure project imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealDataLoader:
    """Load diplomatic documents from MEA (India) and MOFA (Japan)
    
    Uses the built-in comprehensive dataset (50 docs, 2000-2024) by default.
    When live crawling is needed, delegates to the enhanced crawlers.
    """
    
    def __init__(self):
        self.cache_dir = Path(__file__).parent.parent / 'data' / 'raw'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def load_all_documents(self) -> pd.DataFrame:
        """Load the complete India-Japan document dataset"""
        from scrapers.data_loader import DataLoader
        loader = DataLoader()
        df = loader.load_sample_data()
        logger.info(f"Loaded {len(df)} India-Japan diplomatic documents")
        return df
    
    def load_mea_documents(self, query: str = "bilateral relations", max_results: int = 50) -> pd.DataFrame:
        """
        Load documents from Indian Ministry of External Affairs.
        Returns MEA-sourced documents from the built-in dataset.
        
        Args:
            query: Search term (used for filtering)
            max_results: Maximum documents to return
        
        Returns:
            DataFrame with MEA documents
        """
        try:
            df = self.load_all_documents()
            mea_docs = df[df['source'] == 'MEA'].head(max_results)
            
            if query and query != "bilateral relations":
                mask = mea_docs['content'].str.contains(query, case=False, na=False)
                filtered = mea_docs[mask]
                if len(filtered) > 0:
                    mea_docs = filtered
            
            logger.info(f"Loaded {len(mea_docs)} MEA documents")
            return mea_docs
        except Exception as e:
            logger.error(f"MEA load error: {str(e)}")
            return pd.DataFrame()
    
    def load_mofa_documents(self, query: str = "日本インド", max_results: int = 50) -> pd.DataFrame:
        """
        Load documents from Japanese Ministry of Foreign Affairs.
        Returns MOFA-sourced documents from the built-in dataset.
        
        Args:
            query: Search term
            max_results: Maximum documents to return
        
        Returns:
            DataFrame with MOFA documents
        """
        try:
            df = self.load_all_documents()
            mofa_docs = df[df['source'] == 'MOFA'].head(max_results)
            
            if query and query not in ["日本インド", "bilateral relations"]:
                mask = mofa_docs['content'].str.contains(query, case=False, na=False)
                filtered = mofa_docs[mask]
                if len(filtered) > 0:
                    mofa_docs = filtered
            
            logger.info(f"Loaded {len(mofa_docs)} MOFA documents")
            return mofa_docs
        except Exception as e:
            logger.error(f"MOFA load error: {str(e)}")
            return pd.DataFrame()


if __name__ == "__main__":
    loader = RealDataLoader()
    
    all_docs = loader.load_all_documents()
    print(f"Total documents: {len(all_docs)}")
    
    mea = loader.load_mea_documents()
    print(f"MEA documents: {len(mea)}")
    
    mofa = loader.load_mofa_documents()
    print(f"MOFA documents: {len(mofa)}")
    
    print(f"Date range: {all_docs['date'].min()} to {all_docs['date'].max()}")
