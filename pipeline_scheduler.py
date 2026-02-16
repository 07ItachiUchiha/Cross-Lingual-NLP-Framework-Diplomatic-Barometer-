"""
Scheduled Update Pipeline
Automates data collection, processing, and report generation
Supports scheduled runs and incremental updates
"""

import schedule
import time
import logging
from datetime import datetime
import json
from pathlib import Path
from typing import Dict, List, Optional
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from scrapers.mea_crawler_enhanced import MEACrawler
from scrapers.mofa_crawler_enhanced import MOFACrawler
from scrapers.data_loader import DataLoader
from preprocessing.preprocessor import Preprocessor
from analysis.strategic_shift_enhanced import StrategicShiftAnalyzer
from analysis.tone_analyzer import ToneAnalyzer
from analysis.thematic_clustering import ThematicAnalyzer
from utils.helpers import save_analysis_results

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/pipeline_schedule.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ScheduledUpdatePipeline:
    """Manages scheduled data collection and analysis updates"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config = self._load_config(config_file)
        self.mea_crawler = MEACrawler(cache_file=self.config['mea_cache'])
        self.mofa_crawler = MOFACrawler(cache_file=self.config['mofa_cache'])
        self.data_loader = DataLoader()
        self.preprocessor = Preprocessor()
        self.strategic_analyzer = StrategicShiftAnalyzer()
        self.tone_analyzer = ToneAnalyzer()
        self.thematic_analyzer = ThematicAnalyzer()
        self.run_history = []
        self.last_update = None
    
    def _load_config(self, config_file: Optional[str] = None) -> Dict:
        """Load pipeline configuration"""
        default_config = {
            'mea_cache': 'cache/mea_documents.json',
            'mofa_cache': 'cache/mofa_documents.json',
            'output_dir': 'data/processed',
            'log_dir': 'logs',
            'schedule_frequency': 'daily',  # daily, weekly, monthly
            'max_retries': 3,
            'timeout': 30
        }
        
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    custom_config = json.load(f)
                    default_config.update(custom_config)
                logger.info(f"Loaded configuration from {config_file}")
            except Exception as e:
                logger.warning(f"Could not load config file: {e}. Using defaults.")
        
        return default_config
    
    def collect_data(self) -> Dict:
        """Collect new data from MEA and MOFA websites"""
        logger.info("="*70)
        logger.info("Starting data collection phase")
        logger.info("="*70)
        
        collection_result = {
            'timestamp': datetime.now().isoformat(),
            'mea': {'success': False, 'documents': 0, 'errors': 0},
            'mofa': {'success': False, 'documents': 0, 'errors': 0}
        }
        
        # Collect MEA data
        try:
            logger.info("Fetching MEA documents...")
            mea_docs = self.mea_crawler.search_bilateral_documents(
                country="Japan",
                doc_type="Joint Statements",
                start_year=2020
            )
            self.mea_crawler.documents.extend(mea_docs)
            self.mea_crawler.save_to_cache(self.config['mea_cache'])
            
            collection_result['mea']['success'] = True
            collection_result['mea']['documents'] = len(mea_docs)
            logger.info(f"MEA collection complete: {len(mea_docs)} new documents")
        
        except Exception as e:
            logger.error(f"MEA collection failed: {str(e)}")
            collection_result['mea']['errors'] = 1
        
        # Collect MOFA data
        try:
            logger.info("Fetching MOFA documents...")
            mofa_docs = self.mofa_crawler.search_bilateral_documents(
                country="India",
                doc_type="Joint Statements",
                start_year=2020
            )
            self.mofa_crawler.documents.extend(mofa_docs)
            self.mofa_crawler.save_to_cache(self.config['mofa_cache'])
            
            collection_result['mofa']['success'] = True
            collection_result['mofa']['documents'] = len(mofa_docs)
            logger.info(f"MOFA collection complete: {len(mofa_docs)} new documents")
        
        except Exception as e:
            logger.error(f"MOFA collection failed: {str(e)}")
            collection_result['mofa']['errors'] = 1
        
        return collection_result
    
    def process_data(self) -> Dict:
        """Process collected data"""
        logger.info("="*70)
        logger.info("Starting data processing phase")
        logger.info("="*70)
        
        processing_result = {
            'timestamp': datetime.now().isoformat(),
            'documents_loaded': 0,
            'documents_processed': 0,
            'errors': 0
        }
        
        try:
            # Load combined data
            logger.info("Loading combined data...")
            df = self.data_loader.load_combined_data()
            processing_result['documents_loaded'] = len(df)
            
            # Preprocess
            logger.info("Preprocessing documents...")
            processed_df = self.preprocessor.process_dataframe(df, content_column='content')
            processing_result['documents_processed'] = len(processed_df)
            
            logger.info(f"Processed {len(processed_df)} documents")
            
        except Exception as e:
            logger.error(f"Data processing failed: {str(e)}")
            processing_result['errors'] = 1
        
        return processing_result
    
    def analyze_data(self, processed_df) -> Dict:
        """Perform analysis on processed data"""
        logger.info("="*70)
        logger.info("Starting analysis phase")
        logger.info("="*70)
        
        analysis_result = {
            'timestamp': datetime.now().isoformat(),
            'strategic_shift': None,
            'tone_analysis': None,
            'thematic_analysis': None,
            'errors': 0
        }
        
        try:
            # Strategic shift analysis
            logger.info("Performing strategic shift analysis...")
            shift_report, scored_df, yearly_df = self.strategic_analyzer.generate_shift_report(processed_df)
            analysis_result['strategic_shift'] = shift_report
            
            # Tone analysis
            logger.info("Performing tone analysis...")
            tone_df = self.tone_analyzer.process_dataframe(processed_df, text_column='cleaned')
            analysis_result['tone_analysis'] = {
                'total_documents': len(tone_df),
                'tone_distribution': self.tone_analyzer.get_tone_distribution(tone_df)
            }
            
            # Thematic analysis
            logger.info("Performing thematic analysis...")
            thematic_analysis, df_themes = self.thematic_analyzer.analyze_theme_evolution(
                processed_df,
                text_column='cleaned'
            )
            analysis_result['thematic_analysis'] = {
                'num_topics': len(thematic_analysis['overall_themes']),
                'themes': list(thematic_analysis['overall_themes'].keys())
            }
            
            logger.info("Analysis complete")
        
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            analysis_result['errors'] = 1
        
        return analysis_result
    
    def export_results(self, analysis_result: Dict) -> Dict:
        """Export analysis results"""
        logger.info("="*70)
        logger.info("Starting export phase")
        logger.info("="*70)
        
        export_result = {
            'timestamp': datetime.now().isoformat(),
            'json_file': None,
            'excel_file': None,
            'errors': 0
        }
        
        try:
            # Create output directory if it doesn't exist
            Path(self.config['output_dir']).mkdir(parents=True, exist_ok=True)
            
            # Save JSON
            json_file = os.path.join(
                self.config['output_dir'],
                f"analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            with open(json_file, 'w') as f:
                json.dump(analysis_result, f, indent=2, default=str)
            export_result['json_file'] = json_file
            logger.info(f"Saved JSON to {json_file}")
        
        except Exception as e:
            logger.error(f"JSON export failed: {str(e)}")
            export_result['errors'] += 1
        
        return export_result
    
    def run_full_pipeline(self) -> Dict:
        """Run complete pipeline"""
        logger.info("\n" + "="*70)
        logger.info(f"STARTING FULL PIPELINE RUN - {datetime.now().isoformat()}")
        logger.info("="*70 + "\n")
        
        pipeline_result = {
            'started': datetime.now().isoformat(),
            'status': 'RUNNING',
            'phases': {}
        }
        
        try:
            # Phase 1: Collect data
            pipeline_result['phases']['collection'] = self.collect_data()
            
            # Phase 2: Process data
            pipeline_result['phases']['processing'] = self.process_data()
            
            # Phase 3: Analyze
            pipeline_result['phases']['analysis'] = self.analyze_data(None)  # Would use actual data in production
            
            # Phase 4: Export
            pipeline_result['phases']['export'] = self.export_results(
                pipeline_result['phases']['analysis']
            )
            
            pipeline_result['status'] = 'COMPLETE'
            logger.info("\n" + "="*70)
            logger.info("PIPELINE RUN COMPLETE")
            logger.info("="*70 + "\n")
        
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            pipeline_result['status'] = 'FAILED'
        
        pipeline_result['completed'] = datetime.now().isoformat()
        
        # Record in history
        self.run_history.append(pipeline_result)
        self.last_update = datetime.now()
        
        return pipeline_result
    
    def schedule_runs(self, frequency: str = 'daily', hour: int = 0, minute: int = 0):
        """Schedule automatic pipeline runs"""
        logger.info(f"Scheduling pipeline runs - Frequency: {frequency}")
        
        if frequency == 'daily':
            schedule.every().day.at(f"{hour:02d}:{minute:02d}").do(self.run_full_pipeline)
            logger.info(f"Scheduled daily at {hour:02d}:{minute:02d}")
        
        elif frequency == 'weekly':
            schedule.every().monday.at(f"{hour:02d}:{minute:02d}").do(self.run_full_pipeline)
            logger.info(f"Scheduled weekly (Monday) at {hour:02d}:{minute:02d}")
        
        elif frequency == 'monthly':
            # Run on the first day of month
            logger.info(f"Monthly schedule will be implemented based on specific requirements")
        
        return schedule
    
    def start_scheduler(self):
        """Start the scheduler loop"""
        logger.info("Starting scheduler daemon...")
        logger.info("Press Ctrl+C to stop")
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        
        except KeyboardInterrupt:
            logger.info("Scheduler stopped by user")
    
    def get_run_history(self) -> List[Dict]:
        """Get history of pipeline runs"""
        return self.run_history
    
    def generate_status_report(self) -> Dict:
        """Generate current status report"""
        return {
            'last_update': self.last_update.isoformat() if self.last_update else None,
            'total_runs': len(self.run_history),
            'successful_runs': sum(1 for r in self.run_history if r['status'] == 'COMPLETE'),
            'failed_runs': sum(1 for r in self.run_history if r['status'] == 'FAILED'),
            'mea_documents': len(self.mea_crawler.documents),
            'mofa_documents': len(self.mofa_crawler.documents),
            'recent_runs': self.run_history[-5:] if self.run_history else []
        }


if __name__ == "__main__":
    # Initialize pipeline
    pipeline = ScheduledUpdatePipeline()
    
    # Run immediately
    logger.info("Running initial pipeline...")
    result = pipeline.run_full_pipeline()
    print(f"\nPipeline Result: {json.dumps(result, indent=2, default=str)}")
    
    # Optional: Schedule for later
    # logger.info("Setting up scheduled runs...")
    # pipeline.schedule_runs(frequency='daily', hour=0, minute=0)
    # pipeline.start_scheduler()
