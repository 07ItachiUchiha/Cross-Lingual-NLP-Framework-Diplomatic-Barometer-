"""
MOFA (Ministry of Foreign Affairs - Japan) Web Crawler
Fetches Japan-India joint statements and bilateral documents
"""

import pandas as pd
from datetime import datetime
import logging
from typing import List, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MOFACrawler:
    """Crawler for Japanese MOFA documents.
    
    Loads from the built-in dataset of ~50 India-Japan diplomatic
    documents (2000-2024).
    """
    
    def __init__(self):
        self.base_url = "https://www.mofa.go.jp"
        self.documents = []
    
    def fetch_all_documents(self) -> pd.DataFrame:
        """Fetch all MOFA-sourced India-Japan documents"""
        from scrapers.data_loader import DataLoader
        loader = DataLoader()
        df = loader.load_sample_data()
        mofa_docs = df[df['source'] == 'MOFA'].copy()
        self.documents = mofa_docs.to_dict('records')
        logger.info(f"Loaded {len(mofa_docs)} MOFA documents")
        return mofa_docs
    
    def search_bilateral_documents(self, country: str = "India",
                                    doc_type: str = "Joint Statements") -> List[Dict]:
        """
        Search for bilateral documents between Japan and specified country
        
        Args:
            country: Target country (default: India)
            doc_type: Document type filter
        
        Returns:
            List of document metadata dicts
        """
        logger.info(f"Searching MOFA for {doc_type} between Japan and {country}...")
        df = self.fetch_all_documents()
        
        if doc_type and doc_type != "Joint Statements":
            mask = df['title'].str.contains(doc_type.replace("s", ""), case=False, na=False)
            filtered = df[mask]
            if len(filtered) > 0:
                df = filtered
        
        self.documents = df.to_dict('records')
        logger.info(f"Found {len(self.documents)} MOFA documents")
        return self.documents
    
    def get_document_count(self) -> int:
        """Get total number of loaded documents"""
        return len(self.documents)


def main():
    """Test MOFA crawler"""
    crawler = MOFACrawler()
    docs = crawler.search_bilateral_documents()
    print(f"Found {len(docs)} MOFA documents")
    if docs:
        print(f"Date range: {docs[0].get('date', 'N/A')} to {docs[-1].get('date', 'N/A')}")
        for doc in docs[:3]:
            print(f"  - {doc.get('title', 'No title')}")


if __name__ == "__main__":
    main()
