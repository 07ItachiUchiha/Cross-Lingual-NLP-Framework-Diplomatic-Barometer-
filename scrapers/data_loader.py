"""Real-data loader for pre-RAG baseline analysis.

This module intentionally avoids hardcoded/mock datasets.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple
import re
from difflib import SequenceMatcher
import os

import pandas as pd

from utils.helpers import infer_document_type

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """Load and validate diplomatic documents from real CSV sources only."""

    REQUIRED_COLUMNS = {
        "date",
        "title",
        "location",
        "signatories",
        "content",
        "source",
    }

    def __init__(self, data_dir: Optional[str] = None):
        project_root = Path(__file__).resolve().parent.parent
        default_data_dir = project_root / "data" / "raw"
        env_dir = os.environ.get("DIPLOMATIC_BAROMETER_DATA_DIR")
        resolved = data_dir or env_dir
        self.data_dir = Path(resolved) if resolved else default_data_dir
        self.documents = None
        self.last_loaded_path: Optional[Path] = None

        self.default_country_pair: Tuple[str, str] = ("india", "japan")
        self.primary_filename, self.canonical_filename = self._pair_filenames(self.default_country_pair)

    @staticmethod
    def _pair_slug(country_pair: Tuple[str, str]) -> str:
        return f"{country_pair[0]}_{country_pair[1]}"

    def _pair_filenames(self, country_pair: Tuple[str, str]) -> Tuple[str, str]:
        slug = self._pair_slug(country_pair)
        return f"{slug}_documents.csv", f"{slug}_documents_canonical.csv"

    def get_pair_filenames(self, country_pair: Optional[Tuple[str, str]] = None) -> Tuple[str, str]:
        pair = country_pair or self.default_country_pair
        return self._pair_filenames(pair)

    def _validate_and_normalize(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        missing = self.REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(
                f"{dataset_name} missing required columns: {sorted(missing)}"
            )

        validated_df = df.copy()
        validated_df["date"] = pd.to_datetime(validated_df["date"], errors="coerce")

        if validated_df["date"].isna().any():
            bad_rows = int(validated_df["date"].isna().sum())
            raise ValueError(f"{dataset_name} has {bad_rows} rows with invalid date values")

        validated_df["content"] = validated_df["content"].astype(str).str.strip()
        validated_df = validated_df[validated_df["content"].str.len() > 0]
        if validated_df.empty:
            raise ValueError(f"{dataset_name} has no valid non-empty content rows")

        validated_df["source"] = validated_df["source"].astype(str).str.upper().str.strip()
        validated_df["year"] = validated_df["date"].dt.year

        # Optional derived metadata: document type (heuristic) for stratified analysis.
        if "doc_type" not in validated_df.columns and "title" in validated_df.columns:
            validated_df["doc_type"] = validated_df["title"].apply(infer_document_type)
        elif "doc_type" in validated_df.columns:
            validated_df["doc_type"] = validated_df["doc_type"].fillna("Unknown").astype(str).str.strip()

        return validated_df

    def load_real_data(self, csv_path: Optional[str] = None, country_pair: Optional[Tuple[str, str]] = None) -> pd.DataFrame:
        """Load the primary real dataset CSV from disk.

        Expected default file: data/raw/<country1>_<country2>_documents.csv
        If a canonical enriched corpus exists, it is preferred:
          data/raw/<country1>_<country2>_documents_canonical.csv
        """

        if csv_path:
            path = Path(csv_path)
        else:
            primary_filename, canonical_filename = self.get_pair_filenames(country_pair)
            canonical = self.data_dir / canonical_filename
            primary = self.data_dir / primary_filename
            path = canonical if canonical.exists() else primary

        if not path.exists():
            raise FileNotFoundError(
                f"Real dataset not found at: {path}. Please provide real CSV data."
            )

        df = pd.read_csv(path)
        validated_df = self._validate_and_normalize(df, dataset_name=str(path.name))
        self.documents = validated_df
        self.last_loaded_path = path
        logger.info(f"Loaded {len(validated_df)} real documents from {path}")
        return validated_df

    @staticmethod
    def _norm_title(value: object) -> str:
        text = str(value or "").strip().lower()
        text = re.sub(r"\s+", " ", text)
        return text

    @staticmethod
    def _norm_date(value: object) -> str:
        try:
            dt = pd.to_datetime(value, errors="coerce")
            if pd.isna(dt):
                return ""
            return dt.date().isoformat()
        except Exception:
            return ""

    def write_or_update_canonical_corpus(self, country_pair: Optional[Tuple[str, str]] = None) -> Optional[Path]:
        """Create/update a canonical corpus CSV with enriched URLs.

        - Reads existing canonical corpus if present, otherwise primary corpus.
        - Uses all available live scrape CSVs (data/raw/live_scrape_*.csv) to fill missing URLs.
        - Writes to: data/raw/<country1>_<country2>_documents_canonical.csv

        The original primary corpus is never modified.
        """

        primary_filename, canonical_filename = self.get_pair_filenames(country_pair)
        canonical_path = self.data_dir / canonical_filename
        primary_path = self.data_dir / primary_filename

        base_path = canonical_path if canonical_path.exists() else primary_path
        if not base_path.exists():
            logger.warning(f"Canonical corpus update skipped: missing base corpus at {base_path}")
            return None

        try:
            base_df = pd.read_csv(base_path)
        except Exception as exc:
            logger.warning(f"Canonical corpus update skipped: failed to read base corpus: {exc}")
            return None

        # Ensure schema and normalized date/year
        base_df = self._validate_and_normalize(base_df, dataset_name=str(base_path.name))
        if "url" not in base_df.columns:
            base_df["url"] = ""
        base_df["url"] = (
            base_df["url"]
            .fillna("")
            .astype(str)
            .str.strip()
            .replace({"nan": "", "None": "", "none": ""})
        )

        live_files = sorted(self.data_dir.glob("live_scrape_*.csv"), reverse=True)
        if not live_files:
            # Still write canonical if it didn't exist yet (separate from original)
            if not canonical_path.exists():
                base_df.to_csv(canonical_path, index=False, encoding="utf-8")
                logger.info(f"Wrote canonical corpus (no live scrape URLs available) to {canonical_path}")
                return canonical_path
            return canonical_path

        live_frames = []
        for path in live_files[:20]:  # cap to avoid unbounded growth
            try:
                df = pd.read_csv(path)
            except Exception:
                continue
            if not {"date", "title", "source", "url"}.issubset(set(df.columns)):
                continue
            df = df[["date", "title", "source", "url"]].copy()
            df["url"] = df["url"].astype(str).fillna("").str.strip()
            df = df[df["url"].ne("")]
            if df.empty:
                continue
            live_frames.append(df)

        if not live_frames:
            return canonical_path

        live_df = pd.concat(live_frames, ignore_index=True)
        live_df["_k_date"] = live_df["date"].apply(self._norm_date)
        live_df["_k_title"] = live_df["title"].apply(self._norm_title)
        live_df["_k_source"] = live_df["source"].astype(str).str.upper().str.strip()
        live_df = live_df[live_df["_k_date"].ne("") & live_df["_k_title"].ne("")]

        # Prefer newest files first via earlier concat order + first-seen key
        url_lookup = {}
        for _, row in live_df.iterrows():
            key = (row.get("_k_date", ""), row.get("_k_title", ""), row.get("_k_source", ""))
            if key not in url_lookup:
                url_lookup[key] = row.get("url", "")

        # Second-pass lookup: title+source only (but only if the mapping is unique)
        title_source_candidates: dict[tuple[str, str], list[str]] = {}
        for _, row in live_df.iterrows():
            key2 = (row.get("_k_title", ""), row.get("_k_source", ""))
            url_val = str(row.get("url", "")).strip()
            if not url_val:
                continue
            title_source_candidates.setdefault(key2, []).append(url_val)

        title_source_unique = {
            k: v[0] for k, v in title_source_candidates.items() if len(set(v)) == 1
        }

        # Fuzzy candidates per source for final pass
        per_source_titles: dict[str, list[tuple[str, str]]] = {}
        for _, row in live_df.iterrows():
            src = str(row.get("_k_source", "")).strip()
            ttl = str(row.get("_k_title", "")).strip()
            url_val = str(row.get("url", "")).strip()
            if not src or not ttl or not url_val:
                continue
            per_source_titles.setdefault(src, []).append((ttl, url_val))

        # deterministic ordering
        for src in per_source_titles:
            per_source_titles[src] = sorted(per_source_titles[src], key=lambda x: (x[0], x[1]))

        base_df["_k_date"] = base_df["date"].apply(self._norm_date)
        base_df["_k_title"] = base_df["title"].apply(self._norm_title)
        base_df["_k_source"] = base_df["source"].astype(str).str.upper().str.strip()

        def fill_url(row) -> str:
            cur = str(row.get("url", "")).strip()
            if cur.lower() in {"nan", "none"}:
                cur = ""
            if cur:
                return cur

            key = (row.get("_k_date", ""), row.get("_k_title", ""), row.get("_k_source", ""))
            exact = url_lookup.get(key, "")
            if exact:
                return exact

            key2 = (row.get("_k_title", ""), row.get("_k_source", ""))
            exact2 = title_source_unique.get(key2, "")
            if exact2:
                return exact2

            # Final pass: fuzzy match within same source (only if clearly unique)
            src = str(row.get("_k_source", "")).strip()
            base_title = str(row.get("_k_title", "")).strip()
            if not src or not base_title or len(base_title) < 12:
                return ""

            candidates = per_source_titles.get(src, [])
            if not candidates:
                return ""

            best_ratio = 0.0
            second_ratio = 0.0
            best_url = ""
            for cand_title, cand_url in candidates:
                ratio = SequenceMatcher(None, base_title, cand_title).ratio()
                if ratio > best_ratio:
                    second_ratio = best_ratio
                    best_ratio = ratio
                    best_url = cand_url
                elif ratio > second_ratio:
                    second_ratio = ratio

            # conservative thresholds to avoid wrong links
            if best_ratio >= 0.93 and (best_ratio - second_ratio) >= 0.03:
                return best_url

            return ""

        base_df["url"] = base_df.apply(fill_url, axis=1)
        base_df = base_df.drop(columns=["_k_date", "_k_title", "_k_source"], errors="ignore")

        base_df.to_csv(canonical_path, index=False, encoding="utf-8")
        filled = int((base_df["url"].astype(str).str.strip().ne("")).sum())
        logger.info(f"Wrote canonical corpus to {canonical_path} (rows with url={filled}/{len(base_df)})")
        return canonical_path

    def load_combined_data(
        self,
        mea_file: Optional[str] = None,
        mofa_file: Optional[str] = None,
        country_pair: Optional[Tuple[str, str]] = None,
    ) -> pd.DataFrame:
        """Load real MEA+MOFA data either from split files or primary combined CSV."""
        if not mea_file and not mofa_file:
            return self.load_real_data(country_pair=country_pair)

        dataframes = []
        if mea_file:
            mea_path = Path(mea_file)
            if not mea_path.exists():
                raise FileNotFoundError(f"MEA file not found: {mea_path}")
            mea_df = pd.read_csv(mea_path)
            mea_df["source"] = "MEA"
            dataframes.append(self._validate_and_normalize(mea_df, dataset_name=mea_path.name))

        if mofa_file:
            mofa_path = Path(mofa_file)
            if not mofa_path.exists():
                raise FileNotFoundError(f"MOFA file not found: {mofa_path}")
            mofa_df = pd.read_csv(mofa_path)
            mofa_df["source"] = "MOFA"
            dataframes.append(self._validate_and_normalize(mofa_df, dataset_name=mofa_path.name))

        if not dataframes:
            raise ValueError("No real MEA/MOFA data files were provided")

        combined = pd.concat(dataframes, ignore_index=True)
        combined = combined.drop_duplicates(subset=["title", "date", "source"], keep="first")
        self.documents = combined
        logger.info(f"Loaded {len(combined)} combined real documents")
        return combined


def main():
    """CLI sanity check for real-data loading."""
    loader = DataLoader()
    df = loader.load_real_data()
    print(f"Loaded {len(df)} documents")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Sources: {df['source'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
