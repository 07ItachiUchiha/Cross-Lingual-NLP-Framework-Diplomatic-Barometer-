"""
Discovery Providers Package
"""

from scrapers.harvesting.providers.wayback_cdx import WaybackCDXProvider
from scrapers.harvesting.providers.ministry_archives import MinistryArchiveProvider
from scrapers.harvesting.providers.treaty_repositories import TreatyRepositoryProvider
from scrapers.harvesting.providers.security_archives import SecurityArchiveProvider

__all__ = [
    "WaybackCDXProvider",
    "MinistryArchiveProvider",
    "TreatyRepositoryProvider",
    "SecurityArchiveProvider",
]
