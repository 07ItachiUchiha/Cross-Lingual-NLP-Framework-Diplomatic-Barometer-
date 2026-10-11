"""
Unified Multi-Source Diplomatic Harvester
Orchestrates systematic archival crawling, Wayback CDX historical recovery,
UNTC treaty indexing, and defense archives into canonical corpora.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from bs4 import BeautifulSoup

from scrapers.harvesting.base_provider import HarvestedCandidate
from scrapers.harvesting.deduplication import DeduplicationEngine
from scrapers.harvesting.providers.ministry_archives import MinistryArchiveProvider
from scrapers.harvesting.providers.security_archives import SecurityArchiveProvider
from scrapers.harvesting.providers.treaty_repositories import TreatyRepositoryProvider
from scrapers.harvesting.providers.wayback_cdx import WaybackCDXProvider

logger = logging.getLogger(__name__)


class UnifiedHarvester:
    """Master orchestrator for multi-source diplomatic corpus expansion."""

    def __init__(self, output_dir: Optional[Path] = None):
        project_root = Path(__file__).resolve().parent.parent.parent
        self.output_dir = Path(output_dir) if output_dir else project_root / "data" / "raw"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.cache_path = self.output_dir / "harvesting_cache.sqlite"
        self.dedup_engine = DeduplicationEngine(title_similarity_threshold=0.75)

        # Initialize providers
        self.providers = [
            MinistryArchiveProvider(cache_db_path=self.cache_path),
            WaybackCDXProvider(cache_db_path=self.cache_path),
            TreatyRepositoryProvider(cache_db_path=self.cache_path),
            SecurityArchiveProvider(cache_db_path=self.cache_path),
        ]

    def close(self) -> None:
        """Close all provider resources."""
        for p in self.providers:
            if hasattr(p, "close"):
                p.close()

    def _extract_text_and_metadata(self, html: str) -> Tuple[str, str, Optional[str]]:
        """Extract clean body text and identify signatories/location if present."""
        soup = BeautifulSoup(html, "html.parser")

        # Strip non-textual nodes
        for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "aside", "form"]):
            tag.decompose()

        # Locate main container if available
        main_body = (
            soup.find("div", class_=re.compile(r"content|article|statement|document|body|main|press", re.I))
            or soup.find("main")
            or soup.find("body")
            or soup
        )

        paragraphs = [p.get_text().strip() for p in main_body.find_all(["p", "div", "li"]) if len(p.get_text().strip()) > 20]
        raw_text = "\n\n".join(paragraphs) if paragraphs else main_body.get_text()

        # Clean whitespace
        cleaned = re.sub(r"[ \t]+", " ", raw_text)
        cleaned = re.sub(r"\n\s*\n+", "\n\n", cleaned).strip()

        # Heuristic detection for location and signatories
        location = "New Delhi / Partner Capital"
        loc_match = re.search(r"\b(Tokyo|Paris|New Delhi|Osaka|Marseille|Yokohama|Kyoto)\b", cleaned[:500], re.I)
        if loc_match:
            location = loc_match.group(1).title()

        return raw_text, cleaned, location

    def harvest_corpus(
        self,
        country_pair: Tuple[str, str],
        max_docs_limit: int = 600,
        min_chars_limit: int = 500,
    ) -> pd.DataFrame:
        """Run all discovery providers, fetch and clean texts, deduplicate, and persist canonical CSV."""
        country1, country2 = country_pair
        logger.info(f"Initiating Unified Multi-Source Harvest for {country1.title()}–{country2.title()}...")

        all_candidates: List[HarvestedCandidate] = []

        # 1. Discover candidates across all 4 methods
        for provider in self.providers:
            p_name = provider.__class__.__name__
            try:
                logger.info(f"Running discovery via {p_name}...")
                found = provider.discover(country_pair)
                logger.info(f"{p_name} discovered {len(found)} candidate links")
                all_candidates.extend(found)
            except Exception as e:
                logger.warning(f"Provider {p_name} failed: {e}")

        logger.info(f"Total raw candidates discovered across all sources: {len(all_candidates)}")

        # 2. Fetch and extract content for each candidate
        successful_records: List[HarvestedCandidate] = []
        fetcher = self.providers[0]  # Reuse base session & cache

        for idx, cand in enumerate(all_candidates):
            if len(successful_records) >= max_docs_limit:
                logger.info(f"Reached target document limit ({max_docs_limit}).")
                break

            target_fetch_url = cand.archive_url if cand.archive_url else cand.url
            html = fetcher.fetch_url(target_fetch_url, use_cache=True)
            if not html:
                continue

            raw_text, cleaned_text, loc = self._extract_text_and_metadata(html)

            # Filter out pages that did not yield substantial content
            if len(cleaned_text) < min_chars_limit:
                continue

            cand.content = raw_text
            cand.cleaned = cleaned_text
            if not cand.location and loc:
                cand.location = loc

            successful_records.append(cand)

        logger.info(f"Successfully extracted full text for {len(successful_records)} documents.")

        # 3. Deduplicate across repositories and tag canonical instances
        resolved_records = self.dedup_engine.cluster_and_resolve(successful_records)

        # 4. Convert to canonical DataFrame format
        rows = []
        now_iso = datetime.now(timezone.utc).isoformat()

        for r in resolved_records:
            rows.append(
                {
                    "title": r.title,
                    "date": r.date or f"{r.year or '2000'}-01-01",
                    "year": int(r.year) if r.year else 2000,
                    "location": r.location or "Bilateral",
                    "signatories": r.signatories or f"{country1.title()}, {country2.title()}",
                    "content": r.content or "",
                    "cleaned": r.cleaned or "",
                    "source": r.source,
                    "url": r.url,
                    "retrieved_at_utc": now_iso,
                    "genre": r.genre,
                    "language": r.language,
                    "duplicate_cluster_id": r.duplicate_cluster_id,
                    "is_canonical": r.is_canonical,
                }
            )

        df = pd.DataFrame(rows)

        # Target canonical filename matching repository schema
        target_filename = f"{country1.lower()}_{country2.lower()}_documents_canonical.csv"
        target_path = self.output_dir / target_filename

        # If previous canonical file exists, gracefully merge to ensure no historic documents are lost
        if target_path.exists():
            try:
                prev_df = pd.read_csv(target_path)
                logger.info(f"Found existing corpus at {target_path} ({len(prev_df)} rows). Merging...")
                combined = pd.concat([prev_df, df], ignore_index=True)
                # Keep unique by URL or title
                combined = combined.drop_duplicates(subset=["url"], keep="last")
                df = combined
            except Exception as e:
                logger.warning(f"Error merging with existing corpus: {e}")

        # Save canonical dataset
        df.to_csv(target_path, index=False, encoding="utf-8")
        logger.info(f"Persisted canonical dataset with {len(df)} records to {target_path}")

        # Write audit report
        report = {
            "country_pair": country_pair,
            "generated_at_utc": now_iso,
            "total_documents": len(df),
            "canonical_documents": int(df["is_canonical"].sum()) if "is_canonical" in df.columns else len(df),
            "sources": df["source"].value_counts().to_dict() if "source" in df.columns else {},
            "years_span": [int(df["year"].min()), int(df["year"].max())] if "year" in df.columns and len(df) else [],
            "genres": df["genre"].value_counts().to_dict() if "genre" in df.columns else {},
            "languages": df["language"].value_counts().to_dict() if "language" in df.columns else {},
        }
        report_path = self.output_dir / f"unified_harvest_report_{country1}_{country2}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        return df
