"""
MEA (Ministry of External Affairs - India) Web Crawler - Enhanced
Scrapes Joint Statements, Joint Declarations, and Vision Statements from 2000-2025
Includes robust error handling, retries, and rate limiting
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import logging
import time
from typing import List, Dict, Optional, Tuple
import json
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RetryConfig:
    """Configuration for retry logic"""
    max_retries = 3
    retry_delay = 2  # seconds
    timeout = 15  # seconds
    backoff_factor = 1.5


class MEACrawler:
    """Enhanced Crawler for Indian Ministry of External Affairs documents"""
    
    def __init__(self, cache_file: Optional[str] = None):
        self.base_url = "https://mea.gov.in"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.documents = []
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        self.cache_file = cache_file
        self.error_log = []
    
    def _retry_request(self, url: str, max_retries: int = 3) -> Optional[requests.Response]:
        """Make HTTP request with retry logic"""
        for attempt in range(max_retries):
            try:
                response = self.session.get(
                    url,
                    timeout=RetryConfig.timeout,
                    verify=True
                )
                response.raise_for_status()
                logger.info(f"Successfully fetched: {url}")
                return response
            
            except requests.exceptions.Timeout:
                logger.warning(f"Timeout attempt {attempt + 1}/{max_retries}: {url}")
                self.error_log.append({
                    'error': 'Timeout',
                    'url': url,
                    'attempt': attempt + 1,
                    'timestamp': datetime.now().isoformat()
                })
            
            except requests.exceptions.ConnectionError:
                logger.warning(f"Connection error attempt {attempt + 1}/{max_retries}: {url}")
                self.error_log.append({
                    'error': 'ConnectionError',
                    'url': url,
                    'attempt': attempt + 1,
                    'timestamp': datetime.now().isoformat()
                })
            
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:  # Rate limited
                    wait_time = RetryConfig.retry_delay * (RetryConfig.backoff_factor ** attempt)
                    logger.warning(f"Rate limited, waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"HTTP error {e.response.status_code}: {url}")
                    self.error_log.append({
                        'error': f'HTTPError {e.response.status_code}',
                        'url': url,
                        'timestamp': datetime.now().isoformat()
                    })
                    return None
            
            except Exception as e:
                logger.error(f"Unexpected error attempt {attempt + 1}/{max_retries}: {str(e)}")
                self.error_log.append({
                    'error': str(e),
                    'url': url,
                    'attempt': attempt + 1,
                    'timestamp': datetime.now().isoformat()
                })
            
            if attempt < max_retries - 1:
                time.sleep(RetryConfig.retry_delay * (RetryConfig.backoff_factor ** attempt))
        
        logger.error(f"Failed to fetch {url} after {max_retries} attempts")
        return None
    
    def search_bilateral_documents(self, country: str = "Japan", 
                                   doc_type: str = "Joint Statements",
                                   start_year: int = 2000,
                                   end_year: int = 2025) -> List[Dict]:
        """
        Search for bilateral documents between India and specified country
        
        Args:
            country: Target country (default: Japan)
            doc_type: Type of document (Joint Statements, Joint Declarations, Vision Statements)
            start_year: Start year for search
            end_year: End year for search
        
        Returns:
            List of document metadata
        """
        logger.info(f"Searching for {doc_type} between India and {country} ({start_year}-{end_year})...")
        
        # Possible search URLs for MEA website (adjust based on actual structure)
        search_patterns = [
            f"{self.base_url}/press-release?title={country}",
            f"{self.base_url}/bilateral/India_{country}",
            f"{self.base_url}/search?q={country}+{doc_type}"
        ]
        
        documents = []
        
        for search_url in search_patterns:
            logger.info(f"Attempting search URL: {search_url}")
            response = self._retry_request(search_url)
            
            if response:
                try:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # Extract document links (adapt selectors based on actual MEA website)
                    links = soup.find_all('a', href=True)
                    logger.info(f"Found {len(links)} links on search page")
                    
                    for link in links[:20]:  # Limit to first 20 results
                        doc_url = link.get('href')
                        if doc_url and any(keyword in doc_url.lower() for keyword in ['joint', 'statement', 'declaration']):
                            if not doc_url.startswith('http'):
                                doc_url = self.base_url + doc_url
                            
                            metadata = self.extract_document_metadata(doc_url)
                            if metadata:
                                documents.append(metadata)
                
                except Exception as e:
                    logger.error(f"Error parsing search results: {str(e)}")
                    self.error_log.append({
                        'error': f'ParseError: {str(e)}',
                        'url': search_url,
                        'timestamp': datetime.now().isoformat()
                    })
        
        logger.info(f"Found {len(documents)} documents")
        return documents
    
    def extract_document_metadata(self, url: str) -> Optional[Dict]:
        """
        Extract metadata from a single MEA document with error handling
        
        Args:
            url: Document URL
        
        Returns:
            Dictionary with date, title, location, signatories, or None if extraction failed
        """
        try:
            response = self._retry_request(url)
            if not response:
                return None
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract metadata (selectors depend on actual MEA website structure)
            metadata = {
                'url': url,
                'title': None,
                'date': None,
                'location': None,
                'signatories': None,
                'content': None,
                'source': 'MEA'
            }
            
            # Try to extract title
            title_tag = soup.find('h1') or soup.find('title')
            if title_tag:
                metadata['title'] = title_tag.get_text(strip=True)
            
            # Try to extract date
            date_patterns = ['published', 'date', 'released']
            for pattern in date_patterns:
                date_tag = soup.find(string=lambda x: x and pattern in x.lower())
                if date_tag:
                    metadata['date'] = date_tag.strip()
                    break
            
            # Extract main content
            content_div = soup.find('div', class_='content') or soup.find('article')
            if content_div:
                metadata['content'] = content_div.get_text(strip=True)[:1000]
            
            logger.info(f"Extracted metadata: {metadata.get('title', 'Unknown')}")
            return metadata if metadata.get('title') else None
        
        except Exception as e:
            logger.error(f"Error extracting metadata from {url}: {str(e)}")
            self.error_log.append({
                'error': f'ExtractionError: {str(e)}',
                'url': url,
                'timestamp': datetime.now().isoformat()
            })
            return None
    
    def fetch_documents_batch(self, urls: List[str]) -> List[Dict]:
        """
        Fetch multiple documents with rate limiting
        
        Args:
            urls: List of document URLs
        
        Returns:
            List of extracted documents
        """
        documents = []
        for i, url in enumerate(urls):
            logger.info(f"Processing document {i + 1}/{len(urls)}")
            metadata = self.extract_document_metadata(url)
            if metadata:
                documents.append(metadata)
            
            # Rate limiting between requests
            if i < len(urls) - 1:
                time.sleep(1)
        
        return documents
    
    def save_to_cache(self, filename: str):
        """Save documents to JSON cache"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump({
                    'documents': self.documents,
                    'timestamp': datetime.now().isoformat(),
                    'error_log': self.error_log
                }, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(self.documents)} documents to {filename}")
        except Exception as e:
            logger.error(f"Error saving cache: {str(e)}")
    
    def load_from_cache(self, filename: str) -> bool:
        """Load documents from JSON cache"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.documents = data.get('documents', [])
                self.error_log = data.get('error_log', [])
            logger.info(f"Loaded {len(self.documents)} documents from cache")
            return True
        except Exception as e:
            logger.error(f"Error loading cache: {str(e)}")
            return False
    
    def get_error_report(self) -> Dict:
        """Get summary of errors encountered"""
        error_types = {}
        for error in self.error_log:
            error_type = error.get('error', 'Unknown')
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        return {
            'total_errors': len(self.error_log),
            'error_breakdown': error_types,
            'documents_successfully_extracted': len(self.documents)
        }


if __name__ == "__main__":
    crawler = MEACrawler(cache_file="mea_cache.json")
    
    # Try to load from cache first
    if not crawler.load_from_cache("mea_cache.json"):
        logger.info("Cache not found or empty, performing live search...")
        docs = crawler.search_bilateral_documents("Japan", "Joint Statements")
        crawler.documents = docs
        crawler.save_to_cache("mea_cache.json")
    
    print(f"\nMEA Crawler Report:")
    print(f"Documents: {len(crawler.documents)}")
    print(f"Errors: {crawler.get_error_report()}")
