"""
Wayback Machine CDX API Discovery Provider
Recovers official diplomatic summit statements, communiqués, and press releases
from 2000-2015 lost during ministry CMS redesigns.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Dict, List, Optional, Tuple
from urllib.parse import unquote

from scrapers.harvesting.base_provider import BaseDiscoveryProvider, HarvestedCandidate

logger = logging.getLogger(__name__)


class WaybackCDXProvider(BaseDiscoveryProvider):
    """Discovers historical official documents via the Wayback Machine CDX API."""

    CDX_API_URL = "https://web.archive.org/cdx/search/cdx"

    TARGET_PATTERNS = {
        ("india", "japan"): [
            {
                "url_prefix": "mofa.go.jp/region/asia-paci/india/*",
                "source": "MOFA",
                "from_year": "2000",
                "to_year": "2015",
                "language": "en",
            },
            {
                "url_prefix": "mea.gov.in/bilateral-documents.htm*",
                "source": "MEA",
                "from_year": "2000",
                "to_year": "2015",
                "language": "en",
            },
        ],
        ("india", "france"): [
            {
                "url_prefix": "diplomatie.gouv.fr/en/country-files/india/*",
                "source": "MEAE",
                "from_year": "2000",
                "to_year": "2015",
                "language": "en",
            },
            {
                "url_prefix": "diplomatie.gouv.fr/fr/dossiers-pays/inde/*",
                "source": "MEAE",
                "from_year": "2000",
                "to_year": "2015",
                "language": "fr",
            },
            {
                "url_prefix": "mea.gov.in/bilateral-documents.htm*",
                "source": "MEA",
                "from_year": "2000",
                "to_year": "2015",
                "language": "en",
            },
        ],
    }

    def _query_cdx(self, prefix: str, from_yr: str, to_yr: str) -> List[Dict]:
        """Query CDX API for snapshots matching pattern."""
        params = {
            "url": prefix,
            "output": "json",
            "fl": "original,timestamp,statuscode,mimetype",
            "collapse": "urlkey",
            "filter": "statuscode:200",
            "from": from_yr,
            "to": to_yr,
        }
        query_url = f"{self.CDX_API_URL}?url={params['url']}&output=json&fl={params['fl']}&collapse=urlkey&filter=statuscode:200&from={params['from']}&to={params['to']}"
        
        raw_json = self.fetch_url(query_url, use_cache=True)
        if not raw_json:
            return []

        try:
            data = json.loads(raw_json)
            if not isinstance(data, list) or len(data) <= 1:
                return []
            
            headers = data[0]
            records = []
            for row in data[1:]:
                rec = dict(zip(headers, row))
                records.append(rec)
            return records
        except Exception as e:
            logger.warning(f"CDX JSON parsing error for {prefix}: {e}")
            return []

    def discover(self, country_pair: Tuple[str, str]) -> List[HarvestedCandidate]:
        """Discover historical snapshots for country pair."""
        candidates: List[HarvestedCandidate] = []
        patterns = self.TARGET_PATTERNS.get(country_pair, [])
        partner = country_pair[1].lower()

        for pat in patterns:
            records = self._query_cdx(pat["url_prefix"], pat["from_year"], pat["to_year"])
            logger.info(f"CDX returned {len(records)} candidate URLs for {pat['url_prefix']}")

            for rec in records:
                orig_url = rec.get("original", "")
                ts = rec.get("timestamp", "")
                mimetype = str(rec.get("mimetype", "")).lower()

                # Skip binaries, styles, scripts
                if any(ext in orig_url.lower() for ext in [".jpg", ".png", ".gif", ".css", ".js", ".ico"]):
                    continue
                if mimetype and not any(m in mimetype for m in ["html", "text"]):
                    continue

                # Filter relevance
                url_lower = orig_url.lower()
                if "india" not in url_lower and partner not in url_lower:
                    continue

                year_int = int(ts[:4]) if len(ts) >= 4 and ts[:4].isdigit() else None
                date_str = f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}" if len(ts) >= 8 else None

                # Generate direct Wayback archive snapshot URL (using id_ for raw clean HTML)
                archive_snapshot_url = f"https://web.archive.org/web/{ts}id_/{orig_url}"

                # Infer preliminary title from URL path
                slug = orig_url.rstrip("/").split("/")[-1].replace(".html", "").replace(".htm", "")
                title = unquote(slug).replace("-", " ").replace("_", " ").title()
                if len(title) < 5:
                    title = f"{pat['source']} Diplomatic Statement ({year_int or 'Archival'})"

                candidate = HarvestedCandidate(
                    url=orig_url,
                    title=title,
                    date=date_str,
                    year=year_int,
                    source=pat["source"],
                    source_tier=2,  # Tier 2: Wayback Snapshot
                    genre="Summit Statement" if any(k in url_lower for k in ["summit", "joint", "declar"]) else "Press Release",
                    language=pat["language"],
                    archive_url=archive_snapshot_url,
                    metadata={"cdx_timestamp": ts, "provider": "wayback_cdx"},
                )
                candidates.append(candidate)

        return candidates
