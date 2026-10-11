"""
Ministry Archives Discovery Provider
Systematically crawls primary bilateral sections of MEA, MOFA, and MEAE.
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional, Tuple
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

from scrapers.harvesting.base_provider import BaseDiscoveryProvider, HarvestedCandidate

logger = logging.getLogger(__name__)


class MinistryArchiveProvider(BaseDiscoveryProvider):
    """Crawls official live ministry bilateral repositories."""

    SEED_CONFIG = {
        ("india", "japan"): [
            {
                "url": "https://www.mofa.go.jp/region/asia-paci/india/index.html",
                "source": "MOFA",
                "language": "en",
                "base_url": "https://www.mofa.go.jp",
            },
            {
                "url": "https://www.mea.gov.in/bilateral-documents.htm?dtl/1/india-japan-relations",
                "source": "MEA",
                "language": "en",
                "base_url": "https://www.mea.gov.in",
            },
            {
                "url": "https://www.in.emb-japan.go.jp/itpr_en/Japan_India_Relations.html",
                "source": "MOFA",
                "language": "en",
                "base_url": "https://www.in.emb-japan.go.jp",
            },
        ],
        ("india", "france"): [
            {
                "url": "https://www.diplomatie.gouv.fr/en/country-files/india/",
                "source": "MEAE",
                "language": "en",
                "base_url": "https://www.diplomatie.gouv.fr",
            },
            {
                "url": "https://www.diplomatie.gouv.fr/fr/dossiers-pays/inde/",
                "source": "MEAE",
                "language": "fr",
                "base_url": "https://www.diplomatie.gouv.fr",
            },
            {
                "url": "https://www.mea.gov.in/bilateral-documents.htm?dtl/1/india-france-relations",
                "source": "MEA",
                "language": "en",
                "base_url": "https://www.mea.gov.in",
            },
        ],
    }

    def _parse_links_from_html(self, html: str, base_url: str, partner_code: str) -> List[Tuple[str, str]]:
        """Extract candidate document links and anchor text."""
        soup = BeautifulSoup(html, "html.parser")
        results = []
        partner_kw = "japan" if partner_code == "japan" else "france"

        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            text = re.sub(r"\s+", " ", a.get_text()).strip()

            if not href or href.startswith("#") or href.startswith("javascript:"):
                continue

            full_url = urljoin(base_url, href)
            url_lower = full_url.lower()
            text_lower = text.lower()

            # Filter relevant documents
            is_substantive = any(
                term in text_lower or term in url_lower
                for term in [
                    "joint", "statement", "declaration", "summit", "partnership",
                    "communique", "mou", "agreement", "treaty", "speech", "visit",
                    "2+2", "ministerial", "dialogue"
                ]
            )

            # Avoid noise like navigation, terms of use, privacy
            is_noise = any(
                n in text_lower or n in url_lower
                for n in ["privacy", "sitemap", "contact", "feedback", "terms of use", "accessibility", "login"]
            )

            if is_substantive and not is_noise and len(text) > 8:
                results.append((full_url, text))

        return results

    def discover(self, country_pair: Tuple[str, str]) -> List[HarvestedCandidate]:
        """Discover live links across official ministerial bilateral trees."""
        candidates: List[HarvestedCandidate] = []
        seeds = self.SEED_CONFIG.get(country_pair, [])
        partner_code = country_pair[1]
        seen_urls = set()

        for seed in seeds:
            html = self.fetch_url(seed["url"], use_cache=True)
            if not html:
                continue

            links = self._parse_links_from_html(html, seed["base_url"], partner_code)
            logger.info(f"Parsed {len(links)} candidate links from {seed['url']}")

            for full_url, title in links:
                if full_url in seen_urls:
                    continue
                seen_urls.add(full_url)

                # Date heuristic from text or URL (e.g. 2024, 2018-05-02)
                year_match = re.search(r"\b(20\d\d)\b", f"{title} {full_url}")
                year_val = int(year_match.group(1)) if year_match else None

                date_match = re.search(r"\b(20\d\d)[-_/](0[1-9]|1[0-2])[-_/](0[1-9]|[12]\d|3[01])\b", f"{title} {full_url}")
                date_val = f"{date_match.group(1)}-{date_match.group(2)}-{date_match.group(3)}" if date_match else (f"{year_val}-01-01" if year_val else None)

                # Classify genre
                t_lower = title.lower()
                if "treaty" in t_lower or "agreement" in t_lower or "mou" in t_lower:
                    genre = "Treaty/Agreement"
                elif "speech" in t_lower or "address" in t_lower or "banquet" in t_lower:
                    genre = "Official Speech"
                elif "2+2" in t_lower or "defense" in t_lower or "security" in t_lower:
                    genre = "Defense/2+2 Declaration"
                elif "summit" in t_lower or "joint" in t_lower or "declaration" in t_lower:
                    genre = "Joint Summit Statement"
                else:
                    genre = "Official Statement"

                cand = HarvestedCandidate(
                    url=full_url,
                    title=title,
                    date=date_val,
                    year=year_val,
                    source=seed["source"],
                    source_tier=1,  # Tier 1: Primary Official Ministry
                    genre=genre,
                    language=seed["language"],
                    metadata={"provider": "ministry_archives", "seed_parent": seed["url"]},
                )
                candidates.append(cand)

        return candidates
