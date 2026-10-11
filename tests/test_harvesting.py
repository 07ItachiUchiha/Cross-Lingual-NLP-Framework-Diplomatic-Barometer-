"""
Unit tests for Unified Multi-Source Harvester Package
"""

import tempfile
from pathlib import Path
import pytest
import pandas as pd

from scrapers.harvesting.base_provider import HarvestedCandidate, BaseDiscoveryProvider
from scrapers.harvesting.deduplication import DeduplicationEngine
from scrapers.harvesting.providers.wayback_cdx import WaybackCDXProvider
from scrapers.harvesting.providers.ministry_archives import MinistryArchiveProvider
from scrapers.harvesting.unified_harvester import UnifiedHarvester


def test_harvested_candidate_defaults():
    cand = HarvestedCandidate(
        url="https://www.mea.gov.in/doc1.htm",
        title="India-Japan Summit Statement",
        year=2008,
        source="MEA",
    )
    assert cand.source_tier == 1
    assert cand.is_canonical is True
    assert cand.language == "en"
    assert cand.genre == "Joint Statement"


def test_deduplication_engine_clustering():
    engine = DeduplicationEngine(title_similarity_threshold=0.7)

    cand1 = HarvestedCandidate(
        url="https://www.mea.gov.in/joint_statement_2006.htm",
        title="Joint Statement Towards India-Japan Strategic and Global Partnership",
        year=2006,
        source="MEA",
        source_tier=1,
        content="Full statement from MEA containing 2000 words.",
    )
    cand2 = HarvestedCandidate(
        url="https://www.mofa.go.jp/region/asia-paci/india/joint0612.html",
        title="Joint Statement Towards Japan-India Strategic and Global Partnership",
        year=2006,
        source="MOFA",
        source_tier=1,
        content="Full statement from MOFA containing 2050 words.",
    )
    cand3 = HarvestedCandidate(
        url="https://web.archive.org/web/20070101id_/http://mofa.go.jp/joint0612.html",
        title="Joint Statement Towards Japan-India Strategic and Global Partnership",
        year=2006,
        source="MOFA",
        source_tier=2,
        content="Archived version with 2040 words.",
    )
    cand4 = HarvestedCandidate(
        url="https://www.mea.gov.in/declaration_2014.htm",
        title="Tokyo Declaration for India-Japan Special Strategic and Global Partnership",
        year=2014,
        source="MEA",
        source_tier=1,
        content="2014 Tokyo declaration text.",
    )

    resolved = engine.cluster_and_resolve([cand1, cand2, cand3, cand4])
    assert len(resolved) == 4

    # Cand 1, 2, 3 should belong to the same cluster
    cluster_2006 = [c for c in resolved if c.year == 2006]
    assert len(cluster_2006) == 3
    cluster_ids = {c.duplicate_cluster_id for c in cluster_2006}
    assert len(cluster_ids) == 1  # All share same cluster ID

    # Exactly one document in the 2006 cluster must be marked canonical
    canonicals_2006 = [c for c in cluster_2006 if c.is_canonical]
    assert len(canonicals_2006) == 1

    # Cand 4 must be its own cluster and canonical
    doc_2014 = [c for c in resolved if c.year == 2014][0]
    assert doc_2014.is_canonical is True
    assert doc_2014.duplicate_cluster_id != cluster_2006[0].duplicate_cluster_id


def test_sqlite_cache_initialization():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        db_path = Path(tmpdir) / "test_cache.sqlite"
        provider = WaybackCDXProvider(cache_db_path=db_path)
        assert db_path.exists()
        assert provider.session is not None
        provider.close()


def test_unified_harvester_text_extraction():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        harvester = UnifiedHarvester(output_dir=Path(tmpdir))
        sample_html = """
        <html>
            <head><title>Joint Declaration</title></head>
            <body>
                <nav><a href="/">Home</a></nav>
                <div class="article-content">
                    <h1>Joint Statement on Bilateral Cooperation</h1>
                    <p>Prime Minister of India and Prime Minister of Japan met in Tokyo today.</p>
                    <p>Both leaders reaffirmed their commitment to defense and economic partnership.</p>
                </div>
                <footer>Copyright 2024</footer>
            </body>
        </html>
        """
        raw_text, cleaned, loc = harvester._extract_text_and_metadata(sample_html)
        assert "Tokyo" in loc
        assert "Home" not in cleaned
        assert "Copyright 2024" not in cleaned
        assert "Prime Minister of India" in cleaned
        harvester.close()
