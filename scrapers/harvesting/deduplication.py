"""
Deduplication and Provenance Clustering Engine
Groups cross-source instances of the same diplomatic communiqué while selecting
one authoritative canonical version (is_canonical=True) for statistical analysis.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Set, Tuple

from scrapers.harvesting.base_provider import HarvestedCandidate


class DeduplicationEngine:
    """Clusters documents across multiple repositories and tags canonical provenance."""

    def __init__(self, title_similarity_threshold: float = 0.75, date_window_days: int = 3):
        self.title_similarity_threshold = title_similarity_threshold
        self.date_window_days = date_window_days

    @staticmethod
    def _normalize_title(title: str) -> Set[str]:
        """Normalize title into a set of informative tokens."""
        cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", title.lower())
        tokens = cleaned.split()
        stop_words = {
            "the", "and", "of", "in", "on", "to", "for", "with", "between",
            "at", "by", "a", "an", "is", "from", "joint", "statement", "press", "release"
        }
        return {t for t in tokens if len(t) > 2 and t not in stop_words}

    @staticmethod
    def _jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
        """Compute Jaccard token overlap between two token sets."""
        if not set_a or not set_b:
            return 0.0
        intersection = len(set_a.intersection(set_b))
        union = len(set_a.union(set_b))
        return intersection / union if union > 0 else 0.0

    def cluster_and_resolve(self, candidates: List[HarvestedCandidate]) -> List[HarvestedCandidate]:
        """Group candidates into duplicate clusters and tag canonical version."""
        if not candidates:
            return []

        # Sort primarily by date/year to cluster sequentially
        sorted_cands = sorted(
            candidates,
            key=lambda c: (c.year or 9999, c.date or "", c.source_tier),
        )

        clusters: List[List[HarvestedCandidate]] = []

        for cand in sorted_cands:
            cand_tokens = self._normalize_title(cand.title)
            placed = False

            for cluster in clusters:
                rep = cluster[0]
                # Compare year
                same_year = (cand.year and rep.year and cand.year == rep.year) or (not cand.year and not rep.year)
                if not same_year:
                    continue

                rep_tokens = self._normalize_title(rep.title)
                sim = self._jaccard_similarity(cand_tokens, rep_tokens)

                if sim >= self.title_similarity_threshold:
                    cluster.append(cand)
                    placed = True
                    break

            if not placed:
                clusters.append([cand])

        # Assign cluster metadata and select canonical item per cluster
        resolved: List[HarvestedCandidate] = []

        for cluster in clusters:
            # Sort cluster members: lower tier first (Tier 1 > Tier 2 > Tier 3 > Tier 4), then longest content
            sorted_cluster = sorted(
                cluster,
                key=lambda c: (
                    c.source_tier,
                    -len(c.content or c.cleaned or ""),
                )
            )

            # Generate stable cluster ID
            first = sorted_cluster[0]
            norm_title = re.sub(r"[^a-z0-9]", "", first.title.lower())[:30]
            cluster_id = f"clus_{first.year or 'nodate'}_{norm_title}"

            sources_in_cluster = list({c.source for c in sorted_cluster})

            for idx, item in enumerate(sorted_cluster):
                item.duplicate_cluster_id = cluster_id
                item.is_canonical = (idx == 0)
                item.metadata["duplicate_count"] = len(sorted_cluster)
                item.metadata["cluster_sources"] = sources_in_cluster
                resolved.append(item)

        return resolved
