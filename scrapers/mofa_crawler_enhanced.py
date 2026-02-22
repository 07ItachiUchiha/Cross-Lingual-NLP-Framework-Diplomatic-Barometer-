"""
MOFA (Ministry of Foreign Affairs - Japan) Web Crawler - Enhanced
Scrapes joint statements and declarations from 2000-2025.
Handles retries and supports both English and Japanese pages.
"""

import hashlib
import random
import re
import sqlite3
import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import logging
import time
from typing import List, Dict, Optional
import json
from pathlib import Path
from urllib.parse import urljoin, urlparse

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options as ChromeOptions
    from selenium.webdriver.chrome.service import Service as ChromeService
except Exception:
    webdriver = None
    ChromeOptions = None
    ChromeService = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MOFACrawler:
    """Enhanced Crawler for Japanese Ministry of Foreign Affairs documents"""
    
    def __init__(self, cache_file: Optional[str] = None):
        self.base_url_en = "https://www.mofa.go.jp"
        self.base_url_ja = "https://www.mofa.go.jp/mofaj"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9,ja;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0'
        }
        self.documents = []
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        self.cache_file = cache_file
        self.error_log = []
        self.request_delay_range = (2.0, 5.0)
        self.timeout_seconds = 25
        project_root = Path(__file__).resolve().parent.parent
        cache_dir = project_root / "data" / "raw"
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.http_cache_db = cache_dir / "mofa_http_cache.sqlite"
        self._init_http_cache()
        self._browser = None
        self._browser_available = webdriver is not None
        self._browser_error: Optional[str] = None

    def _init_http_cache(self) -> None:
        with sqlite3.connect(self.http_cache_db) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS url_cache (
                    url_hash TEXT PRIMARY KEY,
                    url TEXT NOT NULL,
                    status_code INTEGER,
                    html TEXT,
                    fetched_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_url_cache_fetched_at ON url_cache(fetched_at)"
            )
            conn.commit()

    @staticmethod
    def _hash_url(url: str) -> str:
        return hashlib.sha256(url.encode("utf-8")).hexdigest()

    def _cache_get_html(self, url: str, max_age_hours: int = 48) -> Optional[str]:
        cutoff_ts = time.time() - (max_age_hours * 3600)
        cutoff_iso = datetime.fromtimestamp(cutoff_ts).isoformat()
        url_hash = self._hash_url(url)
        with sqlite3.connect(self.http_cache_db) as conn:
            row = conn.execute(
                "SELECT html FROM url_cache WHERE url_hash = ? AND fetched_at >= ?",
                (url_hash, cutoff_iso),
            ).fetchone()
        if row and row[0]:
            return row[0]
        return None

    def _cache_put_html(self, url: str, status_code: int, html: Optional[str]) -> None:
        if not html:
            return
        with sqlite3.connect(self.http_cache_db) as conn:
            conn.execute(
                """
                INSERT INTO url_cache(url_hash, url, status_code, html, fetched_at)
                VALUES(?, ?, ?, ?, ?)
                ON CONFLICT(url_hash) DO UPDATE SET
                    url = excluded.url,
                    status_code = excluded.status_code,
                    html = excluded.html,
                    fetched_at = excluded.fetched_at
                """,
                (self._hash_url(url), url, status_code, html, datetime.now().isoformat()),
            )
            conn.commit()

    def _get_browser(self):
        if not self._browser_available:
            return None
        if self._browser is not None:
            return self._browser
        try:
            options = ChromeOptions()
            options.add_argument("--headless=new")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--disable-gpu")
            options.add_argument("--window-size=1920,1080")
            options.add_argument(
                "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            )
            service = ChromeService()
            self._browser = webdriver.Chrome(service=service, options=options)
            self._browser.set_page_load_timeout(35)
            return self._browser
        except Exception as exc:
            self._browser_error = str(exc)
            self._browser_available = False
            logger.warning(f"Browser fallback unavailable: {exc}")
            return None

    def _fetch_with_browser(self, url: str) -> Optional[str]:
        browser = self._get_browser()
        if browser is None:
            return None
        try:
            browser.get(url)
            time.sleep(random.uniform(2.0, 3.2))
            return browser.page_source
        except Exception as exc:
            logger.warning(f"Browser fallback failed for {url}: {exc}")
            return None

    @staticmethod
    def _normalize_url(base_url: str, href: str) -> Optional[str]:
        if not href:
            return None
        href = href.strip()
        if href.startswith("javascript:") or href.startswith("mailto:"):
            return None
        normalized = urljoin(base_url, href)
        if normalized.startswith("http://www.mofa.go.jp"):
            normalized = normalized.replace("http://", "https://", 1)
        return normalized

    @staticmethod
    def _is_same_domain(url_a: str, url_b: str) -> bool:
        try:
            return urlparse(url_a).netloc == urlparse(url_b).netloc
        except Exception:
            return False

    def _request_with_backoff(self, url: str, max_retries: int = 4) -> Optional[requests.Response]:
        last_error = None
        for attempt in range(max_retries):
            try:
                if attempt == 0:
                    time.sleep(random.uniform(*self.request_delay_range))
                else:
                    time.sleep(random.uniform(3.0, 6.0))

                response = self.session.get(url, timeout=self.timeout_seconds, allow_redirects=True)
                response.encoding = response.apparent_encoding or 'utf-8'

                if response.status_code == 200:
                    logger.info(f"Successfully fetched: {url}")
                    return response

                if response.status_code in {403, 429, 500, 502, 503, 504}:
                    wait_time = min(20.0, (2 ** attempt) + random.uniform(1.5, 4.0))
                    logger.warning(
                        f"HTTP {response.status_code} attempt {attempt + 1}/{max_retries}: {url} (wait {wait_time:.1f}s)"
                    )
                    last_error = f"HTTP {response.status_code}"
                    if response.status_code == 403 and attempt >= 0:
                        self.error_log.append({
                            'error': last_error,
                            'url': url,
                            'attempt': attempt + 1,
                            'timestamp': datetime.now().isoformat()
                        })
                        return None
                    if attempt < max_retries - 1:
                        time.sleep(wait_time)
                        continue
                else:
                    logger.error(f"HTTP error {response.status_code}: {url}")
                    return None

            except requests.exceptions.Timeout:
                last_error = "Timeout"
                logger.warning(f"Timeout attempt {attempt + 1}/{max_retries}: {url}")
            except requests.exceptions.ConnectionError:
                last_error = "ConnectionError"
                logger.warning(f"Connection error attempt {attempt + 1}/{max_retries}: {url}")
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Unexpected request error attempt {attempt + 1}/{max_retries}: {url} -> {e}")

            self.error_log.append({
                'error': last_error or 'Unknown',
                'url': url,
                'attempt': attempt + 1,
                'timestamp': datetime.now().isoformat()
            })

        logger.error(f"Failed to fetch {url} after {max_retries} attempts ({last_error})")
        return None

    def _fetch_html(
        self,
        url: str,
        use_cache: bool = True,
        use_browser_fallback: bool = True,
        max_retries: int = 4,
    ) -> Optional[str]:
        if use_cache:
            cached = self._cache_get_html(url)
            if cached:
                return cached

        response = self._request_with_backoff(url, max_retries=max_retries)
        if response and response.text:
            self._cache_put_html(url, response.status_code, response.text)
            return response.text

        if use_browser_fallback:
            browser_html = self._fetch_with_browser(url)
            if browser_html:
                logger.info(f"Browser fallback used for {url}")
                self._cache_put_html(url, 200, browser_html)
                return browser_html

        return None

    def _extract_candidate_links(self, page_url: str, html: str, max_links: int = 50) -> List[str]:
        country_terms = ['india', 'j_india', 'インド', '日印']
        document_terms = ['joint', 'statement', 'declaration', 'press', 'release', 'summit', 'meeting', '.pdf']
        soup = BeautifulSoup(html, 'html.parser')
        raw_candidates: List[str] = []

        for anchor in soup.find_all('a', href=True):
            href = self._normalize_url(page_url, anchor.get('href'))
            if not href:
                continue
            if '#' in href and href.endswith('#contents'):
                continue
            if not self._is_same_domain(page_url, href):
                continue

            anchor_text = " ".join(anchor.get_text(" ", strip=True).split()).lower()
            combined = f"{href.lower()} {anchor_text}"
            if any(term in combined for term in country_terms) and any(term in combined for term in document_terms):
                raw_candidates.append(href)

        ordered = []
        seen = set()
        for candidate in raw_candidates:
            if candidate not in seen:
                seen.add(candidate)
                ordered.append(candidate)

        ordered.sort(key=lambda link: 0 if link.lower().endswith('.pdf') else 1)
        return ordered[:max_links]

    @staticmethod
    def _extract_date(text: str) -> Optional[str]:
        patterns = [
            r'\b(\d{4}-\d{2}-\d{2})\b',
            r'\b(\d{1,2}\s+[A-Za-z]+\s+\d{4})\b',
            r'\b([A-Za-z]+\s+\d{1,2},\s*\d{4})\b',
            r'\b(\d{4}/\d{2}/\d{2})\b',
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        return None

    def _retry_request(self, url: str, max_retries: int = 4) -> Optional[requests.Response]:
        return self._request_with_backoff(url, max_retries=max_retries)
    
    def _retry_request(self, url: str, max_retries: int = 3) -> Optional[requests.Response]:
        """Make HTTP request with retry logic and Japanese character handling"""
        for attempt in range(max_retries):
            try:
                response = self.session.get(
                    url,
                    timeout=15,
                    verify=True
                )
                response.encoding = response.apparent_encoding or 'utf-8'  # Handle Japanese encoding
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
                if e.response.status_code == 429:
                    wait_time = 2 * (1.5 ** attempt)
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
                time.sleep(2 * (1.5 ** attempt))
        
        logger.error(f"Failed to fetch {url} after {max_retries} attempts")
        return None
    
    def search_bilateral_documents(self, country: str = "India", 
                                   doc_type: str = "Joint Statements",
                                   start_year: int = 2000,
                                   end_year: int = 2025) -> List[Dict]:
        """
        Search for bilateral documents between Japan and specified country
        
        Args:
            country: Target country (default: India)
            doc_type: Type of document (Joint Statements, Joint Declarations, Vision Statements)
            start_year: Start year for search
            end_year: End year for search
        
        Returns:
            List of document metadata
        """
        logger.info(f"Searching for {doc_type} between Japan and {country} ({start_year}-{end_year})...")
        
        # Search patterns for MOFA website (English and Japanese)
        search_patterns = [
            f"{self.base_url_en}/region/asia-paci/india/index.html",
            f"{self.base_url_en}/region/india/index.html",
            f"{self.base_url_en}/press/release_search.html?country={country}",
            f"{self.base_url_en}/press/release/page1e_{country.lower()}.html",
            f"{self.base_url_ja}/area/india/index.html",
        ]
        
        documents = []
        seen_urls = set()
        doc_type_tokens = [t.lower() for t in doc_type.replace('-', ' ').split() if t]
        max_documents = 40
        
        for search_url in search_patterns:
            logger.info(f"Attempting search URL: {search_url}")
            html = self._fetch_html(search_url, max_retries=3)
            
            if html:
                try:
                    candidate_links = self._extract_candidate_links(search_url, html, max_links=60)
                    logger.info(f"Found {len(candidate_links)} candidate links on search page")

                    for doc_url in candidate_links:
                        if len(documents) >= max_documents:
                            break
                        if doc_url in seen_urls:
                            continue
                        seen_urls.add(doc_url)

                        metadata = self.extract_document_metadata(doc_url)
                        if not metadata:
                            continue

                        date_text = str(metadata.get('date') or '')
                        year_match = re.search(r'(19|20)\d{2}', date_text)
                        if year_match:
                            year = int(year_match.group(0))
                            if year < start_year or year > end_year:
                                continue

                        if doc_type_tokens:
                            title = str(metadata.get('title') or '').lower()
                            if not all(token in title for token in doc_type_tokens if token not in {'and', 'of', 'the'}):
                                if not any(token in title for token in ['joint', 'statement', 'declaration', 'release']):
                                    continue

                        documents.append(metadata)

                    if len(documents) >= max_documents:
                        break
                
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
        Extract metadata from a single MOFA document with error handling
        
        Args:
            url: Document URL
        
        Returns:
            Dictionary with date, title, location, signatories, or None if extraction failed
        """
        try:
            language = 'Japanese' if 'mofaj' in url else 'English'

            if url.lower().endswith('.pdf'):
                filename = url.split('/')[-1]
                title = filename.replace('_', ' ').replace('-', ' ')
                return {
                    'url': url,
                    'title': title or 'MOFA PDF document',
                    'date': self._extract_date(url) or datetime.now().date().isoformat(),
                    'location': None,
                    'signatories': None,
                    'content': f'PDF document reference: {url}',
                    'source': 'MOFA',
                    'language': language
                }

            html = self._fetch_html(url, max_retries=2)
            if not html:
                return None

            soup = BeautifulSoup(html, 'html.parser')
            
            metadata = {
                'url': url,
                'title': None,
                'date': None,
                'location': None,
                'signatories': None,
                'content': None,
                'source': 'MOFA',
                'language': 'unknown'
            }
            
            metadata['language'] = language
            
            # Try to extract title
            title_tag = soup.find('h1') or soup.find('title')
            if title_tag:
                metadata['title'] = title_tag.get_text(strip=True)
            
            # Try to extract date
            metadata['date'] = self._extract_date(html) or datetime.now().date().isoformat()
            
            # Extract main content
            content_div = (
                soup.find('article')
                or soup.find('main')
                or soup.find('div', class_=['content', 'main', 'text', 'article'])
            )
            if content_div:
                text = " ".join(content_div.get_text(" ", strip=True).split())
            else:
                text = " ".join(soup.get_text(" ", strip=True).split())
            metadata['content'] = text[:2500] if len(text) > 2500 else text

            if not metadata.get('title') or len(metadata.get('content') or '') < 120:
                return None
            
            logger.info(f"Extracted MOFA metadata: {metadata.get('title', 'Unknown')} ({metadata.get('language')})")
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
                time.sleep(random.uniform(1.8, 3.5))
        
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

    def close(self) -> None:
        if self._browser is not None:
            try:
                self._browser.quit()
            except Exception:
                pass
            self._browser = None


if __name__ == "__main__":
    crawler = MOFACrawler(cache_file="mofa_cache.json")
    
    # Try to load from cache first
    if not crawler.load_from_cache("mofa_cache.json"):
        logger.info("Cache not found or empty, performing live search...")
        docs = crawler.search_bilateral_documents("India", "Joint Statements")
        crawler.documents = docs
        crawler.save_to_cache("mofa_cache.json")
    
    print(f"\nMOFA Crawler Report:")
    print(f"Documents: {len(crawler.documents)}")
    print(f"Errors: {crawler.get_error_report()}")
    crawler.close()
