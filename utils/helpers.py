"""Misc helper functions — saving results, formatting reports, etc."""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)


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
