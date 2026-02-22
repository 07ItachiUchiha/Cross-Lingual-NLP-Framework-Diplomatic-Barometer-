"""Crawler validation script (not a pytest test module).

This file is intended to be run manually (e.g., `python test_crawlers.py`) to
sanity-check crawler imports and the end-to-end pipeline using real data.
"""

# Prevent pytest from collecting this as a test module even though the filename
# matches its default patterns.
__test__ = False

import sys
import os
from pathlib import Path
import logging
from typing import Dict, List, Tuple
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


class CrawlerTestSuite:
    """Test and validate all diplomatic crawlers"""
    
    def __init__(self):
        self.results = {}
        self.crawler_status = {}
        
    def test_imports(self):
        """Test that all crawlers can be imported"""
        logger.info("\n" + "="*70)
        logger.info("TEST 1: Validating Crawler Imports")
        logger.info("="*70)
        
        crawlers = [
            ('scrapers.data_loader', 'DataLoader'),
            ('scrapers.base_crawler', 'DiplomaticCrawler'),
            ('scrapers.mea_crawler', 'MEACrawler'),
            ('scrapers.mofa_crawler', 'MOFACrawler'),
        ]
        
        for module_name, class_name in crawlers:
            try:
                module = __import__(module_name, fromlist=[class_name])
                crawler_class = getattr(module, class_name)
                logger.info(f"✓ {class_name:25s} - Import successful")
                self.crawler_status[class_name] = {'import': 'PASS'}
            except Exception as e:
                logger.error(f"✗ {class_name:25s} - Import FAILED: {str(e)}")
                self.crawler_status[class_name] = {'import': 'FAIL', 'error': str(e)}
        
        return self.crawler_status
    
    def test_instantiation(self):
        """Test that crawlers can be instantiated"""
        logger.info("\n" + "="*70)
        logger.info("TEST 2: Validating Crawler Instantiation")
        logger.info("="*70)
        
        try:
            from scrapers.mea_crawler import MEACrawler
            mea = MEACrawler()
            logger.info(f"✓ MEACrawler instantiated successfully")
            self.crawler_status.setdefault('MEACrawler', {})['instantiation'] = 'PASS'
        except Exception as e:
            logger.error(f"✗ MEACrawler instantiation FAILED: {str(e)}")
            self.crawler_status.setdefault('MEACrawler', {})['instantiation'] = f'FAIL: {str(e)}'
        
        try:
            from scrapers.mofa_crawler import MOFACrawler
            mofa = MOFACrawler()
            logger.info(f"✓ MOFACrawler instantiated successfully")
            self.crawler_status.setdefault('MOFACrawler', {})['instantiation'] = 'PASS'
        except Exception as e:
            logger.error(f"✗ MOFACrawler instantiation FAILED: {str(e)}")
            self.crawler_status.setdefault('MOFACrawler', {})['instantiation'] = f'FAIL: {str(e)}'
    
    def test_config(self):
        """Test country configuration"""
        logger.info("\n" + "="*70)
        logger.info("TEST 3: Validating Country Configuration")
        logger.info("="*70)
        
        try:
            from utils.country_config import (
                COUNTRIES, COUNTRY_PAIRS, get_country_name,
                get_ministry_name, get_country_pair_label
            )
            
            logger.info(f"✓ Countries defined: {len(COUNTRIES)} (India, Japan)")
            for code, country in COUNTRIES.items():
                logger.info(f"  - {country['name']:15s} ({code:10s}): {country['ministry_name']}")
            
            logger.info(f"\n✓ Country pairs defined: {len(COUNTRY_PAIRS)} (India-Japan)")
            for i, pair in enumerate(COUNTRY_PAIRS, 1):
                try:
                    label = get_country_pair_label(pair)
                    logger.info(f"  {i:2d}. {label}")
                except Exception as e:
                    logger.error(f"  {i:2d}. {pair} - ERROR: {str(e)}")
            
            self.crawler_status['country_config'] = 'PASS'
        except Exception as e:
            logger.error(f"✗ Country configuration FAILED: {str(e)}")
            self.crawler_status['country_config'] = f'FAIL: {str(e)}'
    
    def test_data_pipeline(self):
        """Test the complete data pipeline with real data"""
        logger.info("\n" + "="*70)
        logger.info("TEST 4: Validating Data Pipeline (Real Data)")
        logger.info("="*70)
        
        try:
            from scrapers.data_loader import DataLoader
            from preprocessing.preprocessor import Preprocessor
            from analysis.strategic_shift_enhanced import StrategicShiftAnalyzer
            from analysis.tone_analyzer import ToneAnalyzer
            from analysis.thematic_clustering import ThematicAnalyzer
            
            # Load data
            logger.info("\n[Step 1] Loading real data...")
            loader = DataLoader()
            df = loader.load_combined_data()
            logger.info(f"✓ Loaded {len(df)} documents")
            
            # Preprocess
            logger.info("\n[Step 2] Preprocessing...")
            preprocessor = Preprocessor()
            processed_df = preprocessor.process_dataframe(df, content_column='content')
            logger.info(f"✓ Preprocessed {len(processed_df)} documents")
            
            # Check for cleaned column
            if 'cleaned' not in processed_df.columns:
                raise ValueError("'cleaned' column not found after preprocessing")
            
            non_empty = processed_df['cleaned'].apply(lambda x: len(str(x).strip()) > 0).sum()
            logger.info(f"✓ Non-empty cleaned texts: {non_empty}/{len(processed_df)}")
            
            # Strategic Shift
            logger.info("\n[Step 3] Strategic Shift Analysis...")
            shift_analyzer = StrategicShiftAnalyzer()
            report, scored_df, yearly_df = shift_analyzer.generate_shift_report(processed_df)
            logger.info(f"✓ Economic focus avg: {report['overall_economic_avg']:.4f}")
            logger.info(f"✓ Security focus avg: {report['overall_security_avg']:.4f}")
            logger.info(f"✓ Trend: {report['trend']}")
            
            # Tone Analysis
            logger.info("\n[Step 4] Tone Analysis...")
            tone_analyzer = ToneAnalyzer()
            tone_df = tone_analyzer.process_dataframe(processed_df, text_column='cleaned')
            logger.info(f"✓ Tone analysis completed for {len(tone_df)} documents")
            
            # Thematic Analysis
            logger.info("\n[Step 5] Thematic Analysis...")
            theme_analyzer = ThematicAnalyzer(n_topics=5)
            theme_analysis, theme_df = theme_analyzer.analyze_theme_evolution(
                processed_df, text_column='cleaned'
            )
            logger.info(f"✓ Extracted {theme_analysis['n_topics']} topics")
            
            self.crawler_status['data_pipeline'] = 'PASS'
            logger.info("\n✓ FULL PIPELINE SUCCESSFUL")
            
        except Exception as e:
            logger.error(f"✗ Data pipeline FAILED: {str(e)}")
            logger.error(f"  Error type: {type(e).__name__}")
            import traceback
            logger.error(traceback.format_exc())
            self.crawler_status['data_pipeline'] = f'FAIL: {str(e)}'
    
    def print_summary(self):
        """Print test summary"""
        logger.info("\n" + "="*70)
        logger.info("TEST SUMMARY")
        logger.info("="*70)
        
        passed = sum(1 for v in self.crawler_status.values() 
                    if isinstance(v, dict) and any(
                        status == 'PASS' for status in v.values() if isinstance(status, str)
                    ))
        
        logger.info(f"\nTest Results:")
        for crawler, status in self.crawler_status.items():
            if isinstance(status, dict):
                all_pass = all(v == 'PASS' for v in status.values() if isinstance(v, str))
                symbol = "✓" if all_pass else "✗"
                logger.info(f"{symbol} {crawler}: {status}")
            else:
                symbol = "✓" if status == 'PASS' else "✗"
                logger.info(f"{symbol} {crawler}: {status}")
        
        logger.info("\n" + "="*70)
        logger.info("NEXT STEPS:")
        logger.info("="*70)
        logger.info("""
1. REAL CRAWLER TESTING:
   - Test each crawler against actual websites (requires manual verification)
   - Verify HTML parsing works with live data
   - Check date format detection
   - Handle any rate limiting/blocking

2. DEBUG PARSING ISSUES (if found):
   - Check selector names against actual website structure
   - Update regex patterns for date extraction
   - Add fallback selectors

3. DASHBOARD INTEGRATION:
   - Create country-pair specific data loaders
   - Update dashboard to load real data
   - Test all 7 pages with real documents

4. DATA VALIDATION:
   - Check for duplicates
   - Validate date formats
   - Ensure content is meaningful
   - Check for encoding issues
        """)


def main():
    """Run all tests"""
    suite = CrawlerTestSuite()
    
    suite.test_imports()
    suite.test_instantiation()
    suite.test_config()
    suite.test_data_pipeline()
    suite.print_summary()


if __name__ == "__main__":
    main()
