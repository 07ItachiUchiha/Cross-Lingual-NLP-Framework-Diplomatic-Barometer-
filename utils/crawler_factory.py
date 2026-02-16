"""
Crawler Factory
Dynamically instantiate the right crawler for a country pair
"""

from typing import Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_crawler_for_country(country_code: str, country_pair: Tuple[str, str]):
    """
    Get appropriate crawler class for a country.
    
    Args:
        country_code: Country code ('india' or 'japan')
        country_pair: Tuple of (country1, country2)
    
    Returns:
        Instantiated crawler object
    
    Raises:
        ValueError if country not supported
    """
    
    if country_code == 'india':
        from scrapers.mea_crawler_enhanced import MEACrawler
        return MEACrawler()
    
    elif country_code == 'japan':
        from scrapers.mofa_crawler_enhanced import MOFACrawler
        return MOFACrawler()
    
    else:
        raise ValueError(f"No crawler available for country: {country_code}. Only 'india' and 'japan' are supported.")


def scrape_country_pair(country_pair: Tuple[str, str], start_year: int = 2000, end_year: int = 2024):
    """
    Scrape both countries in a pair.
    
    Args:
        country_pair: Tuple of (country1, country2)
        start_year: Earliest year to scrape
        end_year: Latest year to scrape
    
    Returns:
        Combined DataFrame with documents from both countries
    """
    import pandas as pd
    
    country1, country2 = country_pair
    
    logger.info(f"\nScraping country pair: {country1.upper()} / {country2.upper()}")
    logger.info("=" * 60)
    
    dfs = []
    
    # Scrape country 1
    try:
        logger.info(f"\nScraping {country1.title()}...")
        crawler1 = get_crawler_for_country(country1, country_pair)
        df1 = crawler1.scrape_all(start_year, end_year)
        
        if len(df1) > 0:
            dfs.append(df1)
            logger.info(f"Scraped {len(df1)} documents from {country1.title()}")
        else:
            logger.warning(f"No documents scraped from {country1.title()}")
    
    except Exception as e:
        logger.error(f"Error scraping {country1}: {str(e)}")
    
    # Scrape country 2
    try:
        logger.info(f"\nScraping {country2.title()}...")
        crawler2 = get_crawler_for_country(country2, country_pair)
        df2 = crawler2.scrape_all(start_year, end_year)
        
        if len(df2) > 0:
            dfs.append(df2)
            logger.info(f"Scraped {len(df2)} documents from {country2.title()}")
        else:
            logger.warning(f"No documents scraped from {country2.title()}")
    
    except Exception as e:
        logger.error(f"Error scraping {country2}: {str(e)}")
    
    # Combine results
    if dfs:
        combined_df = pd.concat(dfs, ignore_index=True)
        logger.info(f"\nTotal: {len(combined_df)} documents scraped")
        logger.info(f"   Date range: {combined_df['date'].min()} to {combined_df['date'].max()}")
        logger.info(f"   Sources: {combined_df['country'].unique().tolist()}")
        return combined_df
    else:
        logger.error("No documents scraped from either country")
        return pd.DataFrame()


def main():
    """Test crawler factory"""
    print("\n" + "="*70)
    print("Crawler Factory Test")
    print("="*70)
    
    # Test 1: India crawler
    print("\n[TEST 1] Instantiate India MEA Crawler")
    try:
        india_crawler = get_crawler_for_country('india', ('india', 'japan'))
        print(f"  OK: {india_crawler.__class__.__name__}")
    except Exception as e:
        print(f"  FAIL: {e}")
    
    # Test 2: Japan crawler
    print("\n[TEST 2] Instantiate Japan MOFA Crawler")
    try:
        japan_crawler = get_crawler_for_country('japan', ('india', 'japan'))
        print(f"  OK: {japan_crawler.__class__.__name__}")
    except Exception as e:
        print(f"  FAIL: {e}")
    
    # Test 3: Invalid country
    print("\n[TEST 3] Test invalid country")
    try:
        invalid_crawler = get_crawler_for_country('invalid', ('invalid', 'country'))
        print(f"  FAIL: Should have raised error")
    except ValueError as e:
        print(f"  OK: Correctly raised error: {e}")
    
    print("\n" + "="*70)
    print("Done.")
    print("="*70)


if __name__ == "__main__":
    main()
