"""
Treaty Repositories Discovery Provider
Harvests formal legal treaties and MoUs registered between India-Japan and India-France
from UNTC (treaties.un.org) and MEA Legal & Treaties Division (mea.gov.in/treaty.htm).
"""

from __future__ import annotations

import logging
import re
from typing import List, Tuple
from urllib.parse import quote, urljoin

from bs4 import BeautifulSoup

from scrapers.harvesting.base_provider import BaseDiscoveryProvider, HarvestedCandidate

logger = logging.getLogger(__name__)


class TreatyRepositoryProvider(BaseDiscoveryProvider):
    """Harvests formal bilateral agreements and treaties from UNTC and MEA Treaties database."""

    SEEDS = {
        ("india", "japan"): [
            {
                "url": "https://www.mea.gov.in/treaty.htm?dtl/1/japan",
                "source": "MEA-Treaty",
                "base_url": "https://www.mea.gov.in",
            },
            {
                "url": "https://treaties.un.org/pages/showDetails.aspx?objid=08000002801456bc",
                "source": "UNTC",
                "base_url": "https://treaties.un.org",
            },
        ],
        ("india", "france"): [
            {
                "url": "https://www.mea.gov.in/treaty.htm?dtl/1/france",
                "source": "MEA-Treaty",
                "base_url": "https://www.mea.gov.in",
            },
            {
                "url": "https://treaties.un.org/pages/showDetails.aspx?objid=080000028012e8ad",
                "source": "UNTC",
                "base_url": "https://treaties.un.org",
            },
        ],
    }

    def discover(self, country_pair: Tuple[str, str]) -> List[HarvestedCandidate]:
        """Discover treaties and bilateral agreements."""
        candidates: List[HarvestedCandidate] = []
        seeds = self.SEEDS.get(country_pair, [])
        partner = country_pair[1].lower()

        for seed in seeds:
            html = self.fetch_url(seed["url"], use_cache=True)
            if not html:
                continue

            soup = BeautifulSoup(html, "html.parser")
            for a in soup.find_all("a", href=True):
                href = a["href"].strip()
                title = re.sub(r"\s+", " ", a.get_text()).strip()

                if len(title) < 12 or not href:
                    continue

                full_url = urljoin(seed["base_url"], href)
                t_lower = title.lower()

                if partner not in t_lower and "bilateral" not in t_lower and "agreement" not in t_lower and "convention" not in t_lower:
                    continue

                year_match = re.search(r"\b(20\d\d|19\d\d)\b", title)
                year_val = int(year_match.group(1)) if year_match else None

                candidates.append(
                    HarvestedCandidate(
                        url=full_url,
                        title=title,
                        date=f"{year_val}-01-01" if year_val else None,
                        year=year_val,
                        source=seed["source"],
                        source_tier=3,  # Tier 3: Treaty Repositories
                        genre="Bilateral Treaty/Agreement",
                        language="en",
                        metadata={"provider": "treaty_repositories", "source_db": seed["source"]},
                    )
                )

        logger.info(f"TreatyRepositoryProvider found {len(candidates)} records for {country_pair}")
        return candidates
