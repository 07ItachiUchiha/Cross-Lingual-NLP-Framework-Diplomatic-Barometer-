"""Misc helper functions — saving results, formatting reports, etc."""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
import logging
from datetime import datetime, timezone
import re

logger = logging.getLogger(__name__)


def build_data_provenance_report(
    raw_df: pd.DataFrame,
    processed_df: pd.DataFrame,
    source_path: str,
    duplicate_rows_removed: int,
    dedupe_subset: List[str],
    required_columns: List[str],
) -> Dict[str, Any]:
    """Build a machine-readable provenance snapshot for a pipeline run."""
    report: Dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_source": {
            "path": source_path,
            "format": "csv",
        },
        "row_counts": {
            "raw_loaded": int(len(raw_df)),
            "after_dedup": int(len(processed_df)),
            "duplicate_rows_removed": int(duplicate_rows_removed),
        },
        "deduplication": {
            "enabled": True,
            "subset_columns": dedupe_subset,
        },
        "columns": {
            "required": required_columns,
            "present": sorted(list(raw_df.columns)),
            "missing_required": sorted(list(set(required_columns) - set(raw_df.columns))),
            "missing_value_counts": {
                column: int(raw_df[column].isna().sum()) if column in raw_df.columns else None
                for column in required_columns
            },
        },
    }

    if "source" in processed_df.columns:
        report["source_split"] = {
            str(key): int(value)
            for key, value in processed_df["source"].value_counts().to_dict().items()
        }

    if "doc_type" in processed_df.columns:
        report["doc_type_split"] = {
            str(key): int(value)
            for key, value in processed_df["doc_type"].fillna("Unknown").astype(str).value_counts().to_dict().items()
        }

    if "date" in processed_df.columns:
        parsed_dates = pd.to_datetime(processed_df["date"], errors="coerce")
        report["date_range"] = {
            "start": str(parsed_dates.min()) if parsed_dates.notna().any() else None,
            "end": str(parsed_dates.max()) if parsed_dates.notna().any() else None,
        }

    return report


def infer_document_type(title: object) -> str:
    """Infer a coarse document type label from a document title.

    This is intentionally heuristic and conservative. It never affects which documents
    are loaded (real-data-only), only how they can be stratified in analysis/UI.
    """

    text = str(title or "").strip().lower()
    if not text:
        return "Unknown"

    # Normalize whitespace/punctuation lightly for robust matching.
    normalized = re.sub(r"\s+", " ", text)

    patterns = [
        (r"\bjoint statement\b", "Joint Statement"),
        (r"\bstatement\b", "Statement"),
        (r"\bpress release\b", "Press Release"),
        (r"\bremarks\b|\bspeech\b|\baddress\b", "Remarks/Speech"),
        (r"\bmemorandum of understanding\b|\bmou\b", "MoU"),
        (r"\bagreement\b|\btreaty\b|\bprotocol\b", "Agreement"),
        (r"\bdeclaration\b", "Declaration"),
        (r"\bminutes\b|\bmeeting\b|\bdialogue\b|\bconsultation\b", "Meeting/Dialogue"),
        (r"\boutcome document\b|\boutcomes\b", "Outcome Document"),
        (r"\bfaq\b|\bq\s*&\s*a\b", "Q&A"),
    ]

    for pattern, label in patterns:
        if re.search(pattern, normalized, flags=re.IGNORECASE):
            return label

    return "Other"


def save_provenance_report(report: Dict[str, Any], filepath: str) -> None:
    """Persist provenance report to JSON."""
    try:
        with open(filepath, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, default=str)
        logger.info(f"Saved data provenance report to {filepath}")
    except Exception as exc:
        logger.error(f"Error saving provenance report: {str(exc)}")


def save_analysis_results(results: Dict[str, Any], filepath: str) -> None:
    """Dump results dict to JSON."""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Saved analysis results to {filepath}")
    except Exception as e:
        logger.error(f"Error saving results: {str(e)}")


def load_analysis_results(filepath: str) -> Dict:
    """
    Load analysis results from JSON file
    
    Args:
        filepath: Path to JSON file
    
    Returns:
        Dictionary of analysis results
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            results = json.load(f)
        logger.info(f"Loaded analysis results from {filepath}")
        return results
    except Exception as e:
        logger.error(f"Error loading results: {str(e)}")
        return {}


def export_to_excel(dataframes: Dict[str, pd.DataFrame], filepath: str) -> None:
    """
    Export multiple DataFrames to Excel with multiple sheets
    
    Args:
        dataframes: Dictionary of {sheet_name: DataFrame}
        filepath: Path to save Excel file
    """
    try:
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            for sheet_name, df in dataframes.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        logger.info(f"Exported data to {filepath}")
    except Exception as e:
        logger.error(f"Error exporting to Excel: {str(e)}")


def generate_summary_report(analysis_results: Dict) -> str:
    """
    Generate a text summary report from analysis results
    
    Args:
        analysis_results: Dictionary of analysis results
    
    Returns:
        Formatted summary report
    """
    report = []
    report.append("=" * 70)
    report.append("CROSS-LINGUAL NLP FRAMEWORK - ANALYSIS SUMMARY REPORT")
    report.append("=" * 70)
    report.append("")
    
    if 'overall_trend' in analysis_results:
        report.append(f"Overall Trend: {analysis_results['overall_trend']}")
    
    if 'crossover_year' in analysis_results:
        report.append(f"Strategic Shift Crossover Year: {analysis_results['crossover_year']}")
    
    if 'total_documents' in analysis_results:
        report.append(f"Total Documents Analyzed: {analysis_results['total_documents']}")
    
    report.append("")
    report.append("Key Findings:")
    report.append("-" * 70)
    
    if 'key_findings' in analysis_results:
        for finding in analysis_results['key_findings']:
            report.append(f"  • {finding}")
    
    report.append("")
    report.append("=" * 70)
    
    return "\n".join(report)


def validate_data(df: pd.DataFrame) -> bool:
    """
    Validate data quality
    
    Args:
        df: DataFrame to validate
    
    Returns:
        True if valid, False otherwise
    """
    required_columns = ['date', 'title', 'content']
    
    for col in required_columns:
        if col not in df.columns:
            logger.error(f"Missing required column: {col}")
            return False
    
    if len(df) == 0:
        logger.error("DataFrame is empty")
        return False
    
    # Check for null values in critical columns
    if df['date'].isnull().any() or df['content'].isnull().any():
        logger.warning("Some records have null dates or content")
    
    logger.info(f"Data validation passed: {len(df)} records")
    return True


def get_file_size(filepath: str) -> str:
    """
    Get human-readable file size
    
    Args:
        filepath: Path to file
    
    Returns:
        Human-readable file size
    """
    try:
        size_bytes = Path(filepath).stat().st_size
        
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        
        return f"{size_bytes:.2f} TB"
    except Exception as e:
        logger.error(f"Error getting file size: {str(e)}")
        return "Unknown"


def compare_time_periods(df: pd.DataFrame, 
                         period1_years: List[int],
                         period2_years: List[int]) -> Dict:
    """
    Compare metrics between two time periods
    
    Args:
        df: DataFrame with analysis results
        period1_years: List of years for first period
        period2_years: List of years for second period
    
    Returns:
        Dictionary with comparison metrics
    """
    df['year'] = pd.to_datetime(df['date']).dt.year
    
    p1_data = df[df['year'].isin(period1_years)]
    p2_data = df[df['year'].isin(period2_years)]
    
    comparison = {
        'period_1': {
            'years': period1_years,
            'n_documents': len(p1_data),
            'avg_economic_score': p1_data.get('economic_score', pd.Series([0])).mean(),
            'avg_security_score': p1_data.get('security_score', pd.Series([0])).mean(),
        },
        'period_2': {
            'years': period2_years,
            'n_documents': len(p2_data),
            'avg_economic_score': p2_data.get('economic_score', pd.Series([0])).mean(),
            'avg_security_score': p2_data.get('security_score', pd.Series([0])).mean(),
        }
    }
    
    return comparison


if __name__ == "__main__":
    # Test utility functions
    print("Utility functions loaded successfully")
