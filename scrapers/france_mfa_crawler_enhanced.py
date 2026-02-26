"""
France MFA (MEAE) crawler adapter.

This implementation is intentionally data-loader-backed to stay consistent with
existing pre-RAG corpus workflows. It reads pair-specific corpus CSV files and
filters France-side rows.
"""

import logging
from typing import Dict, List, Optional, Tuple

import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FranceMFACrawler:
    """Crawler adapter for French Ministry for Europe and Foreign Affairs (MEAE)."""

    def __init__(self, cache_file: Optional[str] = None):
        self.base_url = "https://www.diplomatie.gouv.fr/en/"
        self.cache_file = cache_file
        self.documents: List[Dict] = []

    @staticmethod
    def _filter_france_rows(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or len(df) == 0:
            return pd.DataFrame()

        work_df = df.copy()

        source_series = work_df.get("source", pd.Series(index=work_df.index, dtype=object)).astype(str).str.upper().str.strip()
        country_series = work_df.get("country", pd.Series(index=work_df.index, dtype=object)).astype(str).str.lower().str.strip()

        source_markers = {"MEAE", "MFA_FRANCE", "FRANCE_MFA", "FRANCE"}
        source_mask = source_series.isin(source_markers)
        country_mask = country_series.eq("france")

        filtered = work_df[source_mask | country_mask].copy()
        if len(filtered) > 0 and "source" in filtered.columns:
            filtered["source"] = "MEAE"

        return filtered

    def fetch_all_documents(self, country_pair: Tuple[str, str] = ("india", "france")) -> pd.DataFrame:
        """Fetch all MEAE-side documents for the requested country pair from local corpus files."""
        from scrapers.data_loader import DataLoader

        try:
            loader = DataLoader()
            df = loader.load_real_data(country_pair=country_pair)
            france_docs = self._filter_france_rows(df)
            self.documents = france_docs.to_dict("records")
            logger.info(f"Loaded {len(france_docs)} MEAE documents for pair {country_pair[0]}-{country_pair[1]}")
            return france_docs
        except FileNotFoundError:
            logger.warning(
                "No pair-specific corpus found for %s-%s. Expected data/raw/%s_%s_documents.csv",
                country_pair[0],
                country_pair[1],
                country_pair[0],
                country_pair[1],
            )
            return pd.DataFrame()

    def search_bilateral_documents(self, country: str = "India", doc_type: str = "Joint Statements") -> List[Dict]:
        """Search within locally available MEAE corpus rows."""
        logger.info(f"Searching MEAE corpus for {doc_type} between France and {country}...")
        df = self.fetch_all_documents(country_pair=("india", "france"))

        if len(df) == 0:
            self.documents = []
            return self.documents

        if doc_type and doc_type != "Joint Statements" and "title" in df.columns:
            mask = df["title"].astype(str).str.contains(doc_type.replace("s", ""), case=False, na=False)
            filtered = df[mask]
            if len(filtered) > 0:
                df = filtered

        self.documents = df.to_dict("records")
        logger.info(f"Found {len(self.documents)} MEAE documents")
        return self.documents

    def scrape_all(self, start_year: int = 2000, end_year: int = 2024) -> pd.DataFrame:
        """Factory-compatible scrape method with year-window filtering."""
        df = self.fetch_all_documents(country_pair=("india", "france"))
        if len(df) == 0:
            return df

        if "date" in df.columns:
            dates = pd.to_datetime(df["date"], errors="coerce")
            year_mask = dates.dt.year.between(start_year, end_year, inclusive="both")
            filtered = df[year_mask.fillna(False)].copy()
            if "country" not in filtered.columns:
                filtered["country"] = "france"
            return filtered

        if "country" not in df.columns:
            df["country"] = "france"
        return df

    def get_document_count(self) -> int:
        return len(self.documents)
