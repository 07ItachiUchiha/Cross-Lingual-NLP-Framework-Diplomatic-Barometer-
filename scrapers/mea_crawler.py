"""
MEA (Ministry of External Affairs - India) Web Crawler
Fetches India-Japan joint statements and bilateral documents
"""

import pandas as pd
from datetime import datetime
import logging
from typing import List, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MEACrawler:
    """Crawler for Indian Ministry of External Affairs documents.
    
    Loads real MEA documents from the validated corpus CSV.
    """
    
    def __init__(self):
        self.base_url = "https://mea.gov.in"
        self.documents = []
    
    def fetch_all_documents(self) -> pd.DataFrame:
        """Fetch all MEA-sourced India-Japan documents"""
        from scrapers.data_loader import DataLoader
        loader = DataLoader()
        df = loader.load_real_data()
        mea_docs = df[df['source'] == 'MEA'].copy()
        self.documents = mea_docs.to_dict('records')
        logger.info(f"Loaded {len(mea_docs)} MEA documents")
        return mea_docs
    
    def search_bilateral_documents(self, country: str = "Japan",
                                    doc_type: str = "Joint Statements") -> List[Dict]:
        """
        Search for bilateral documents between India and specified country
        
        Args:
            country: Target country (default: Japan)
            doc_type: Document type filter
        
        Returns:
            List of document metadata dicts
        """
        logger.info(f"Searching MEA for {doc_type} between India and {country}...")
        df = self.fetch_all_documents()
        
        if doc_type and doc_type != "Joint Statements":
            mask = df['title'].str.contains(doc_type.replace("s", ""), case=False, na=False)
            filtered = df[mask]
            if len(filtered) > 0:
                df = filtered
        
        self.documents = df.to_dict('records')
        logger.info(f"Found {len(self.documents)} MEA documents")
        return self.documents
    
    def get_document_count(self) -> int:
        """Get total number of loaded documents"""
        return len(self.documents)


def main():
    """Test MEA crawler"""
    crawler = MEACrawler()
    docs = crawler.search_bilateral_documents()
    print(f"Found {len(docs)} MEA documents")
    if docs:
        print(f"Date range: {docs[0].get('date', 'N/A')} to {docs[-1].get('date', 'N/A')}")
        for doc in docs[:3]:
            print(f"  - {doc.get('title', 'No title')}")


if __name__ == "__main__":
    main()
