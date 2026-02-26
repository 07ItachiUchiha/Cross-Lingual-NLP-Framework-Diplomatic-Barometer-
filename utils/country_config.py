"""
Country Configuration System
Defines all countries and their diplomatic ministry details
"""

# Country Definitions
COUNTRIES = {
    'india': {
        'name': 'India',
        'flag': '🇮🇳',
        'ministry_code': 'MEA',
        'ministry_name': 'Ministry of External Affairs',
        'website': 'https://www.mea.gov.in/',
        'press_release_path': '/en/press-releases',
        'language': 'english',
        'region': 'South Asia',
        'data_available_from': 2000
    },
    'japan': {
        'name': 'Japan',
        'flag': '🇯🇵',
        'ministry_code': 'MOFA',
        'ministry_name': 'Ministry of Foreign Affairs',
        'website': 'https://www.mofa.go.jp/en/',
        'press_release_path': '/press/release',
        'language': 'english',
        'region': 'East Asia',
        'data_available_from': 2000
    },
    'france': {
        'name': 'France',
        'flag': '🇫🇷',
        'ministry_code': 'MEAE',
        'ministry_name': 'Ministry for Europe and Foreign Affairs',
        'website': 'https://www.diplomatie.gouv.fr/en/',
        'press_release_path': '/country-files/india/',
        'language': 'english',
        'region': 'Europe',
        'data_available_from': 2000
    },
}


# Valid Country Pairs
# Each pair represents bilateral relations being analyzed
COUNTRY_PAIRS = [
    ('india', 'japan'),      # South Asia ↔ East Asia
    ('india', 'france'),     # South Asia ↔ Europe
]

# Thematic Keywords for Analysis
# These can be customized per country pair or region
ANALYSIS_THEMES = {
    'bilateral_relations': {
        'keywords': [
            'bilateral', 'relations', 'partnership', 'cooperation', 
            'dialogue', 'engagement', 'joint', 'collaboration'
        ],
        'category': 'Diplomatic'
    },
    'trade_economic': {
        'keywords': [
            'trade', 'commerce', 'investment', 'export', 'import',
            'business', 'economic', 'financial', 'goods', 'tariff',
            'market', 'supply chain', 'commerce', 'merchant'
        ],
        'category': 'Economic'
    },
    'defense_security': {
        'keywords': [
            'defense', 'security', 'military', 'strategic', 'armed forces',
            'navy', 'army', 'air force', 'deterrence', 'strategic partnership',
            'defense cooperation', 'security arrangement'
        ],
        'category': 'Security'
    },
    'political_governance': {
        'keywords': [
            'political', 'government', 'parliament', 'elections', 'democracy',
            'governance', 'reform', 'policy', 'legislation'
        ],
        'category': 'Political'
    },
    'cultural_people': {
        'keywords': [
            'culture', 'cultural', 'people-to-people', 'education', 'student',
            'exchange', 'tradition', 'heritage', 'tourism', 'academic',
            'university', 'scholarship', 'artist'
        ],
        'category': 'Cultural'
    },
    'technology_innovation': {
        'keywords': [
            'technology', 'innovation', 'artificial intelligence', 'AI',
            '5G', 'semiconductor', 'digital', 'cyber', 'cybersecurity',
            'startup', 'research', 'development', 'tech'
        ],
        'category': 'Technology'
    },
    'regional_geopolitics': {
        'keywords': [
            'region', 'regional', 'geopolitical', 'indo-pacific', 'asia',
            'europe', 'neighboring', 'territorial', 'border', 'contested',
            'quad', 'aukus', 'eu', 'nato'
        ],
        'category': 'Regional'
    }
}

# Sentiment/Tone Keywords
TONE_INDICATORS = {
    'urgent': [
        'critical', 'urgent', 'immediate', 'emergency', 'crisis',
        'threat', 'urgent need', 'pressing'
    ],
    'positive': [
        'welcome', 'pleased', 'optimistic', 'encouraged', 'positive',
        'progressive', 'strong', 'successful', 'excellent'
    ],
    'negative': [
        'concern', 'concerned', 'worried', 'alarm', 'alarmed',
        'regret', 'regretted', 'opposition', 'against'
    ],
    'formal': [
        'hereby', 'undersigned', 'witnessed', 'solemnly', 'declare',
        'reaffirm', 'commitment', 'obligation'
    ],
    'cordial': [
        'warm', 'friendly', 'cordial', 'amicable', 'goodwill',
        'mutual respect', 'understanding'
    ]
}


def get_country_pair_label(country_pair: tuple) -> str:
    """Get formatted label for country pair"""
    c1, c2 = country_pair
    return f"{COUNTRIES[c1]['flag']} {COUNTRIES[c1]['name']} ↔ {COUNTRIES[c2]['flag']} {COUNTRIES[c2]['name']}"


def get_country_name(country_code: str) -> str:
    """Get full name of country"""
    return COUNTRIES.get(country_code, {}).get('name', country_code.title())


def get_ministry_name(country_code: str) -> str:
    """Get ministry name for country"""
    return COUNTRIES.get(country_code, {}).get('ministry_name', 'Ministry')


def get_ministry_code(country_code: str) -> str:
    """Get ministry code (MEA, MOFA, etc.)"""
    return COUNTRIES.get(country_code, {}).get('ministry_code', country_code.upper())


def validate_country_pair(country_pair: tuple) -> bool:
    """Validate that country pair is in approved list"""
    return country_pair in COUNTRY_PAIRS


def get_valid_countries() -> list:
    """Get list of all valid countries"""
    return list(COUNTRIES.keys())


if __name__ == "__main__":
    # Test the configuration
    print("Country Configuration\n")
    print("=" * 60)
    
    print("\nValid Country Pairs:")
    for pair in COUNTRY_PAIRS:
        print(f"  {get_country_pair_label(pair)}")
    
    print("\nCountries Available:")
    for code in COUNTRIES.keys():
        c = COUNTRIES[code]
        print(f"  {c['flag']} {c['name']:15} - {c['ministry_code']:5} ({c['region']})")
    
    print("\nAnalysis Themes:")
    for theme, details in ANALYSIS_THEMES.items():
        print(f"  • {theme}: {len(details['keywords'])} keywords")
    
    print("\n" + "=" * 60)
    print("Config loaded OK")
