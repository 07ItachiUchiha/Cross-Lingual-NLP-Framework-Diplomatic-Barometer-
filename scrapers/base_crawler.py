"""
Base Crawler Class
Generic diplomatic statement scraper for any country ministry
Implements common functionality: retry logic, rate limiting, error handling
"""

from abc import ABC, abstractmethod
import pandas as pd
from typing import List, Dict, Optional, Tuple
import logging
from datetime import datetime, timedelta
import json
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DiplomaticCrawler(ABC):
    """
    Abstract base class for scraping diplomatic statements from government websites.
    
    Subclasses must implement:
    - get_press_releases(): Fetch list of press releases
    - parse_document(): Parse individual documents
    """
    
    def __init__(self, country_code: str, country_pair: Tuple[str, str], max_retries: int = 3):
        """
        Initialize crawler
        
        Args:
            country_code: Country code (e.g., 'india', 'estonia')
            country_pair: Tuple of (country1, country2) for bilateral analysis
            max_retries: Number of retries for failed requests
        """
        from utils.country_config import COUNTRIES
        
        if country_code not in COUNTRIES:
            raise ValueError(f"Unknown country code: {country_code}")
        
        self.country_code = country_code
        self.country_pair = country_pair
        self.country_config = COUNTRIES[country_code]
        self.max_retries = max_retries
        self.session = None
        self.documents: List[Dict] = []
        
        logger.info(f"Initialized {self.country_config['name']} crawler ({self.country_config['ministry_code']})")
    
    @abstractmethod
    def get_press_releases(self, start_year: int = 2000, end_year: int = 2024) -> List[Dict]:
        """
        Fetch list of press releases from ministry website.
        Must be implemented by subclass.
        
        Args:
            start_year: Earliest year to collect
            end_year: Latest year to collect
        
        Returns:
            List of press release metadata (url, title, date, etc.)
        """
        pass
    
    @abstractmethod
    def parse_document(self, doc_id: str, url: str, title: str, date: str, raw_html: str) -> Optional[Dict]:
        """
        Parse single document from HTML.
        Must be implemented by subclass.
        
        Args:
            doc_id: Unique document identifier
            url: URL of document
            title: Document title
            date: Document date
            raw_html: HTML content to parse
        
        Returns:
            Dictionary with parsed content or None if parsing fails
        """
        pass
    
    def setup_session(self):
        """Setup requests session with retry logic"""
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        
        self.session = requests.Session()
        
        retry_strategy = Retry(
            total=self.max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"]
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Set user agent
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        logger.info("Session setup complete with retry strategy")
    
    def fetch_url(self, url: str, timeout: int = 10) -> Optional[str]:
        """
        Fetch URL content with error handling
        
        Args:
            url: URL to fetch
            timeout: Request timeout in seconds
        
        Returns:
            HTML content or None if failed
        """
        if not self.session:
            self.setup_session()
        
        try:
            response = self.session.get(url, timeout=timeout)
            response.raise_for_status()
            return response.text
        except Exception as e:
            logger.error(f"Failed to fetch {url}: {str(e)}")
            return None
    
    def scrape_all(self, start_year: int = 2000, end_year: int = 2024) -> pd.DataFrame:
        """
        Complete scraping workflow
        
        Args:
            start_year: Earliest year
            end_year: Latest year
        
        Returns:
            DataFrame with all scraped documents
        """
        logger.info(f"Starting scrape for {self.country_config['name']} ({start_year}-{end_year})")
        
        try:
            # Step 1: Get list of press releases
            logger.info("Step 1: Fetching press release list...")
            press_releases = self.get_press_releases(start_year, end_year)
            logger.info(f"Found {len(press_releases)} press releases")
            
            # Step 2: Parse each document
            logger.info("Step 2: Parsing documents...")
            for i, release in enumerate(press_releases):
                if i % 10 == 0:
                    logger.info(f"  Processing {i}/{len(press_releases)}...")
                
                # Fetch document HTML
                html = self.fetch_url(release.get('url'))
                if not html:
                    continue
                
                # Parse document
                doc = self.parse_document(
                    doc_id=release.get('doc_id'),
                    url=release.get('url'),
                    title=release.get('title'),
                    date=release.get('date'),
                    raw_html=html
                )
                
                if doc:
                    self.documents.append(doc)
            
            logger.info(f"Successfully parsed {len(self.documents)} documents")
            
            # Step 3: Convert to DataFrame
            df = pd.DataFrame(self.documents)
            
            # Add source information
            df['source'] = self.country_config['ministry_code']
            df['country'] = self.country_code
            df['ministry_name'] = self.country_config['ministry_name']
            df['scrape_date'] = datetime.now().strftime('%Y-%m-%d')
            
            # Ensure required columns
            required_cols = ['date', 'title', 'content', 'url']
            missing = [col for col in required_cols if col not in df.columns]
            if missing:
                logger.warning(f"Missing columns: {missing}")
            
            return df
        
        except Exception as e:
            logger.error(f"Scraping failed: {str(e)}")
            return pd.DataFrame()
    
    def save_to_csv(self, output_path: str) -> bool:
        """
        Save scraped documents to CSV
        
        Args:
            output_path: Path to save CSV
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.documents:
                logger.warning("No documents to save")
                return False
            
            df = pd.DataFrame(self.documents)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            df.to_csv(output_path, index=False, encoding='utf-8')
            logger.info(f"Saved {len(self.documents)} documents to {output_path}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to save CSV: {str(e)}")
            return False
    
    def save_metadata(self, output_path: str) -> bool:
        """Save scraping metadata"""
        try:
            metadata = {
                'country_code': self.country_code,
                'country_name': self.country_config['name'],
                'ministry': self.country_config['ministry_code'],
                'documents_scraped': len(self.documents),
                'scrape_date': datetime.now().isoformat(),
                'country_pair': self.country_pair,
                'date_range': {
                    'earliest': min([doc.get('date') for doc in self.documents]) if self.documents else None,
                    'latest': max([doc.get('date') for doc in self.documents]) if self.documents else None
                }
            }
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Saved metadata to {output_path}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to save metadata: {str(e)}")
            return False
    
    def get_statistics(self) -> Dict:
        """Get statistics about scraped data"""
        if not self.documents:
            return {}
        
        df = pd.DataFrame(self.documents)
        
        return {
            'total_documents': len(self.documents),
            'date_range': {
                'earliest': df['date'].min(),
                'latest': df['date'].max()
            },
            'avg_content_length': df['content'].apply(len).mean(),
            'documents_by_year': df['date'].str[:4].value_counts().to_dict()
        }


if __name__ == "__main__":
    print("Base Crawler Class - Test")
    print("Use this as base class for country-specific crawlers")
    print("Implement: get_press_releases() and parse_document()")
