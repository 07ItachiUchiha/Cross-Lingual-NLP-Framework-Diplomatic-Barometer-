"""
Security Archives & 2+2 Dialogue Discovery Provider
Harvests official 2+2 Ministerial Dialogues, Defense Declarations, and Joint Working Group Statements
from IDSA (idsa.in) and NIDS (nids.mod.go.jp).
"""

from __future__ import annotations

import logging
import re
from typing import List, Tuple
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from scrapers.harvesting.base_provider import BaseDiscoveryProvider, HarvestedCandidate

logger = logging.getLogger(__name__)


class SecurityArchiveProvider(BaseDiscoveryProvider):
    """Harvests primary defense declarations and 2+2 ministerial communiqués."""

    SEEDS = {
        ("india", "japan"): [
            {
                "url": "https://www.idsa.in/taxonomy/term/japan",
                "source": "IDSA-Primary",
                "base_url": "https://www.idsa.in",
            },
            {
                "url": "https://www.nids.mod.go.jp/english/publication/index.html",
                "source": "NIDS-Defense",
                "base_url": "https://www.nids.mod.go.jp",
            },
        ],
        ("india", "france"): [
            {
                "url": "https://www.idsa.in/taxonomy/term/france",
                "source": "IDSA-Primary",
                "base_url": "https://www.idsa.in",
            },
        ],
    }

    def discover(self, country_pair: Tuple[str, str]) -> List[HarvestedCandidate]:
        """Discover defense statements and 2+2 dialogue communiqués."""
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

                if len(title) < 15 or not href:
                    continue

                full_url = urljoin(seed["base_url"], href)
                t_lower = title.lower()

                # Strictly target primary official texts (2+2, defense dialogue, joint statement, declaration)
                is_defense_doc = any(
                    term in t_lower
                    for term in ["2+2", "defense dialogue", "defence dialogue", "joint declaration", "defense cooperation", "security partnership"]
                )
                if not is_defense_doc:
                    continue

                year_match = re.search(r"\b(20\d\d)\b", title)
                year_val = int(year_match.group(1)) if year_match else None

                candidates.append(
                    HarvestedCandidate(
                        url=full_url,
                        title=title,
                        date=f"{year_val}-01-01" if year_val else None,
                        year=year_val,
                        source=seed["source"],
                        source_tier=4,  # Tier 4: Defense/Security Primary Archive
                        genre="2+2 Defense Dialogue",
                        language="en",
                        metadata={"provider": "security_archives"},
                    )
                )

        logger.info(f"SecurityArchiveProvider found {len(candidates)} records for {country_pair}")
        return candidates
