"""Scheduled corpus refresh workflow for India-Japan and India-France datasets."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Any

from scrapers.india_france_corpus_builder import IndiaFranceCorpusBuilder, IndiaFranceBuildConfig
from scrapers.official_corpus_builder import OfficialCorpusBuilder, CorpusBuildConfig


def refresh_pair_corpora(
    start_year: int = 2000,
    end_year: int = 2026,
    max_docs_india_japan: int = 600,
    max_docs_india_france: int = 500,
    max_urls_per_year: int = 80,
    min_content_chars: int = 850,
) -> Dict[str, Any]:
    started = datetime.now(timezone.utc).isoformat()

    jp_builder = OfficialCorpusBuilder()
    fr_builder = IndiaFranceCorpusBuilder()
    try:
        jp_cfg = CorpusBuildConfig(
            start_year=int(start_year),
            end_year=int(end_year),
            max_docs_total=int(max_docs_india_japan),
            max_urls_per_year=int(max_urls_per_year),
            min_content_chars=int(min_content_chars),
        )
        fr_cfg = IndiaFranceBuildConfig(
            start_year=int(start_year),
            end_year=int(end_year),
            max_docs_total=int(max_docs_india_france),
            max_urls_per_year=int(max_urls_per_year),
            min_content_chars=int(min_content_chars),
        )

        jp_result = jp_builder.build(jp_cfg)
        fr_result = fr_builder.build(fr_cfg)

        jp_report = jp_result.get("report", {}) if isinstance(jp_result, dict) else {}
        fr_report = fr_result.get("report", {}) if isinstance(fr_result, dict) else {}

        return {
            "run_started_utc": started,
            "run_finished_utc": datetime.now(timezone.utc).isoformat(),
            "india_japan": jp_report,
            "india_france": fr_report,
        }
    finally:
        jp_builder.close()
        fr_builder.close()
