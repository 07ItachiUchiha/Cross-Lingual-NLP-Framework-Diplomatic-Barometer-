"""Main pipeline — loads data, runs analysis, exports results"""

import sys
import os
from pathlib import Path
import logging
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from scrapers.data_loader import DataLoader
from preprocessing.preprocessor import Preprocessor
from analysis.strategic_shift_enhanced import StrategicShiftAnalyzer
from analysis.tone_analyzer import ToneAnalyzer
from analysis.thematic_clustering import ThematicAnalyzer
from utils.helpers import save_analysis_results, export_to_excel
from utils.config import PROCESSED_DATA_DIR, ANALYSIS_RESULTS_JSON

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_full_pipeline():
    """Execute the complete analysis pipeline"""
    
    logger.info("=" * 70)
    logger.info("CROSS-LINGUAL NLP FRAMEWORK - FULL ANALYSIS PIPELINE")
    logger.info("=" * 70)
    
    try:
        # Step 1: Load Data
        logger.info("\n[STEP 1] Loading diplomatic documents...")
        loader = DataLoader()
        df = loader.load_sample_data()
        logger.info(f"✓ Loaded {len(df)} documents")
        logger.info(f"  Date range: {df['date'].min()} to {df['date'].max()}")
        
        # Step 2: Preprocessing
        logger.info("\n[STEP 2] Preprocessing documents...")
        preprocessor = Preprocessor()
        processed_df = preprocessor.process_dataframe(df, content_column='content')
        logger.info(f"✓ Preprocessing complete")
        logger.info(f"  Documents processed: {len(processed_df)}")
        
        # Step 3: Strategic Shift Analysis
        logger.info("\n[STEP 3] Analyzing strategic shift...")
        shift_analyzer = StrategicShiftAnalyzer()
        shift_report, scored_df, yearly_df = shift_analyzer.generate_shift_report(processed_df)
        logger.info(f"✓ Strategic shift analysis complete")
        logger.info(f"  Crossover year: {shift_report['crossover_year']}")
        logger.info(f"  Overall trend: {shift_report['trend']}")
        
        # Step 4: Tone and Sentiment Analysis
        logger.info("\n[STEP 4] Analyzing tone and sentiment...")
        tone_analyzer = ToneAnalyzer()
        tone_df = tone_analyzer.process_dataframe(processed_df, text_column='cleaned')
        logger.info(f"✓ Tone analysis complete")
        tone_dist = tone_analyzer.get_tone_distribution(tone_df)
        logger.info(f"  Tone distribution: {tone_dist}")
        
        # Step 5: Thematic Analysis
        logger.info("\n[STEP 5] Performing thematic analysis...")
        thematic_analyzer = ThematicAnalyzer(n_topics=5)
        theme_analysis, theme_df = thematic_analyzer.analyze_theme_evolution(
            processed_df,
            text_column='cleaned'
        )
        logger.info(f"✓ Thematic analysis complete")
        logger.info(f"  Topics discovered: {theme_analysis['n_topics']}")
        
        # Step 6: Compile Results
        logger.info("\n[STEP 6] Compiling results...")
        
        full_results = {
            'metadata': {
                'total_documents': len(df),
                'date_range': {
                    'start': str(df['date'].min()),
                    'end': str(df['date'].max())
                }
            },
            'strategic_shift': {
                'economic_focus_avg': float(shift_report['overall_economic_avg']),
                'security_focus_avg': float(shift_report['overall_security_avg']),
                'crossover_year': shift_report['crossover_year'],
                'trend': shift_report['trend']
            },
            'tone_and_sentiment': {
                'tone_distribution': tone_dist,
                'sentiment_distribution': tone_analyzer.get_sentiment_distribution(tone_df)
            },
            'themes': {
                'n_topics': theme_analysis['n_topics'],
                'overall_themes': {
                    str(k): v for k, v in theme_analysis['overall_themes'].items()
                }
            }
        }
        
        # Save results
        save_analysis_results(full_results, str(ANALYSIS_RESULTS_JSON))
        logger.info(f"✓ Results saved to {ANALYSIS_RESULTS_JSON}")
        
        # Step 7: Export Data
        logger.info("\n[STEP 7] Exporting data files...")
        
        export_dfs = {
            'Processed Documents': processed_df[['date', 'title', 'location', 'cleaned']].head(20),
            'Yearly Analysis': yearly_df,
            'Tone Analysis': tone_analyzer.get_yearly_tone_statistics(tone_df)
        }
        
        excel_path = PROCESSED_DATA_DIR / "diplomatic_barometer_analysis.xlsx"
        export_to_excel(export_dfs, str(excel_path))
        logger.info(f"✓ Data exported to {excel_path}")
        
        # Step 8: Generate Report
        logger.info("\n[STEP 8] Final Summary")
        logger.info("=" * 70)
        logger.info(f"Total Documents Analyzed: {len(df)}")
        logger.info(f"Economic Focus Score: {shift_report['overall_economic_avg']:.4f}")
        logger.info(f"Security Focus Score: {shift_report['overall_security_avg']:.4f}")
        logger.info(f"Strategic Shift Crossover: {shift_report['crossover_year']}")
        logger.info(f"Overall Trend: {shift_report['trend']}")
        logger.info("=" * 70)
        logger.info("✓ PIPELINE COMPLETE")
        
        return {
            'status': 'success',
            'results': full_results,
            'data': {
                'processed_df': processed_df,
                'scored_df': scored_df,
                'tone_df': tone_df,
                'theme_df': theme_df
            }
        }
    
    except Exception as e:
        logger.error(f"\n✗ Pipeline failed: {str(e)}")
        return {
            'status': 'error',
            'error': str(e)
        }


if __name__ == "__main__":
    result = run_full_pipeline()
    
    # Print results to console
    if result['status'] == 'success':
        print("\n" + "=" * 70)
        print("ANALYSIS RESULTS")
        print("=" * 70)
        print(f"Overall Economic Focus: {result['results']['strategic_shift']['economic_focus_avg']:.4f}")
        print(f"Overall Security Focus: {result['results']['strategic_shift']['security_focus_avg']:.4f}")
        print(f"Crossover Year: {result['results']['strategic_shift']['crossover_year']}")
        print(f"Trend: {result['results']['strategic_shift']['trend']}")
        print("=" * 70)
    else:
        print(f"Error: {result['error']}")
