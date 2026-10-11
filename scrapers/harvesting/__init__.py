"""
Unified Harvesting Package
Multi-source official diplomatic corpus harvester across MEA, MOFA, MEAE,
Wayback Machine CDX API, UNTC/Treaties, and Security/2+2 Archives.
"""

from scrapers.harvesting.base_provider import HarvestedCandidate, BaseDiscoveryProvider
from scrapers.harvesting.deduplication import DeduplicationEngine
from scrapers.harvesting.unified_harvester import UnifiedHarvester

__all__ = [
    "HarvestedCandidate",
    "BaseDiscoveryProvider",
    "DeduplicationEngine",
    "UnifiedHarvester",
]
