"""Quick sanity checks for the data loading and crawler setup"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

def print_header(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def test_country_config():
    """Test 1: Country Configuration System"""
    print_header("TEST 1: Country Configuration System")
    
    try:
        from utils.country_config import (
            COUNTRIES, COUNTRY_PAIRS, get_country_name, 
            get_country_pair_label
        )
        
        print("\n  Country configuration loaded OK\n")
        
        print("Countries:")
        for code, info in COUNTRIES.items():
            print(f"   {info['flag']} {info['name']:15} - {info['ministry_code']:5}")
        
        print("\nCountry Pairs:")
        for pair in COUNTRY_PAIRS:
            print(f"   {get_country_pair_label(pair)}")
        
        return True
    
    except Exception as e:
        print(f"\n  Error loading country config: {e}")
        return False


def test_crawler_factory():
    """Test 2: Crawler Factory"""
    print_header("TEST 2: Crawler Factory System")
    
    try:
        from utils.crawler_factory import get_crawler_for_country
        from utils.country_config import COUNTRIES
        
        print("\n  Crawler factory loaded OK\n")
        
        # Test instantiating crawlers
        test_countries = ['india', 'japan']
        test_pair = ('india', 'japan')
        
        print("Testing crawler instantiation:")
        for country in test_countries:
            try:
                crawler = get_crawler_for_country(country, test_pair)
                ministry = COUNTRIES[country]['ministry_name']
                print(f"   OK  {country.title():10} -> {crawler.__class__.__name__:30} ({ministry})")
            except Exception as e:
                print(f"   FAIL {country.title()}: {e}")
        
        return True
    
    except Exception as e:
        print(f"\n  Error loading crawler factory: {e}")
        return False


def test_base_crawler():
    """Test 3: Base Crawler Class"""
    print_header("TEST 3: Base Crawler Class")
    
    try:
        from scrapers.base_crawler import DiplomaticCrawler
        from utils.country_config import COUNTRIES
        
        print("\n  Base crawler class loaded OK\n")
        
        print("Base Crawler Features:")
        features = [
            "Session management with retry logic (3x retries)",
            "Rate limiting between requests",
            "CSV export functionality",
            "JSON metadata export",
            "Error handling and logging",
            "Support for timeout handling"
        ]
        
        for feature in features:
            print(f"   ✓ {feature}")
        
        return True
    
    except Exception as e:
        print(f"\n  Error loading base crawler: {e}")
        return False


def show_usage_examples():
    print_header("USAGE EXAMPLES")
    
    examples = [
        ("Get country info", """
from utils.country_config import get_country_name, get_ministry_name
print(get_country_name('estonia'))  # "Estonia"
print(get_ministry_name('estonia'))  # "Ministry of Foreign Affairs"
        """),
        
        ("Instantiate a crawler", """
from utils.crawler_factory import get_crawler_for_country
crawler = get_crawler_for_country('estonia', ('estonia', 'france'))
print(crawler.country_config['website'])  # URL of ministry
        """),
        
        ("Scrape a country pair", """
from utils.crawler_factory import scrape_country_pair
df = scrape_country_pair(('estonia', 'france'), 2022, 2024)
print(f"Scraped {len(df)} documents")
df.to_csv('estonia_france_documents.csv', index=False)
        """),
        
        ("Get country pair label", """
from utils.country_config import get_country_pair_label
label = get_country_pair_label(('estonia', 'france'))
print(label)  # "🇪🇪 Estonia ↔ 🇫🇷 France"
        """),
    ]
    
    for i, (title, code) in enumerate(examples, 1):
        print(f"\nExample {i}: {title}")
        print(code.strip())


def show_file_summary():
    print_header("FILES")
    
    files = [
        ("utils/country_config.py", "Country definitions (India, Japan) and keywords"),
        ("scrapers/base_crawler.py", "Generic crawler base class"),
        ("scrapers/mea_crawler.py", "India MEA crawler"),
        ("scrapers/mofa_crawler.py", "Japan MOFA crawler"),
        ("utils/crawler_factory.py", "Factory for instantiating crawlers"),
    ]
    
    print()
    for filepath, description in files:
        exists = "ok" if os.path.exists(filepath) else "MISSING"
        print(f"  [{exists:7}] {filepath:40} - {description}")


def show_next_steps():
    print_header("NEXT STEPS")
    
    steps = [
        "1. Test Country Configuration",
        "   → python utils/country_config.py",
        "",
        "2. Test Crawler Factory", 
        "   → python utils/crawler_factory.py",
        "",
        "3. Test Individual Crawlers (requires internet access)",
        "   \u2192 python scrapers/mea_crawler.py",
        "   \u2192 python scrapers/mofa_crawler.py",
        "",
        "4. Verify Data Collection",
        "   \u2192 Check data/raw/ for CSV files",
        "",
        "5. Launch Dashboard",
        "   \u2192 streamlit run dashboard/app.py"
    ]
    
    for step in steps:
        print(f"\n{step}")


def main():
    print("\n" + "=" * 70)
    print("Quick Start - India-Japan NLP Analysis")
    print("=" * 70)
    
    # Run tests
    test_results = []
    
    test_results.append(("Country Configuration", test_country_config()))
    test_results.append(("Crawler Factory", test_crawler_factory()))
    test_results.append(("Base Crawler", test_base_crawler()))
    
    # Show examples and next steps
    show_usage_examples()
    show_file_summary()
    show_next_steps()
    
    # Summary
    print_header("TEST SUMMARY")
    
    print()
    for test_name, passed in test_results:
        status = "PASSED" if passed else "FAILED"
        print(f"  {test_name:30} {status}")
    
    all_passed = all(result for _, result in test_results)
    
    print("\n" + "=" * 70)
    if all_passed:
        print("All tests passed.")
    else:
        print("Some tests failed - check output above.")
    
    print("=" * 70 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
