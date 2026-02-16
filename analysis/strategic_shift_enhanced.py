"""Scores each document on economic vs. security focus.
Expanded lexicons (~70 terms each) + t-test / Cohen's d for significance."""

import pandas as pd
import numpy as np
from collections import Counter
from typing import Dict, List, Tuple, Optional
import logging
from scipy import stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LexiconDefinitions:
    """Enhanced pre-defined lexicons for economic and security categories"""
    
    # Era 1: Economic/Aid focus (expanded)
    ECONOMIC_LEXICON = {
        # Traditional ODA and loans
        'oda', 'yen loan', 'concessional loan', 'grant aid', 'technical cooperation',
        'infrastructure', 'investment', 'trade', 'commerce', 'economic',
        'aid', 'loan', 'assistance', 'development', 'cooperation',
        
        # Transport & Regional Connectivity
        'dmic', 'metro', 'railway', 'corridor', 'transport', 'connectivity',
        'hsbc', 'bullet train', 'shinkansen', 'dedicated freight', 'cargo',
        'port', 'airport', 'bridge', 'tunnel', 'expressway',
        
        # Industrial & Manufacturing
        'industrial', 'manufacturing', 'textile', 'factory', 'production',
        'export', 'import', 'commerce', 'goods', 'merchandise',
        'supply chain', 'logistics', 'warehouse', 'customs',
        
        # Technology & Innovation
        'technology transfer', 'joint venture', 'automotive', 'semiconductor',
        'pharmaceutical', 'chemical', 'electronics', 'machinery',
        'it', 'software', 'hardware', 'startup', 'innovation', 'iot',
        
        # Energy & Resources
        'energy', 'oil', 'gas', 'coal', 'renewable', 'solar', 'wind',
        'hydroelectric', 'power', 'electricity', 'nuclear energy',
        'resources', 'minerals', 'mining', 'extraction',
        
        # Trade & Economic Partnership
        'cepa', 'fta', 'trade agreement', 'tariff', 'duty', 'quota',
        'investment promotion', 'business environment', 'finance', 'banking',
        'insurance', 'export credit', 'credit facility',
        
        # Financial & Monetary
        'yen', 'rupee', 'currency', 'forex', 'swap', 'credit', 'debt',
        'fiscal', 'budget', 'subsidy', 'financial', 'monetary', 'economic policy'
    }
    
    # Era 2: Security/Strategy focus (expanded)
    SECURITY_LEXICON = {
        # Regional Strategy & Geopolitics
        'indo-pacific', 'indopacific', 'free and open indo-pacific', 'foip',
        'quad', 'quadrilateral', 'aukus', 'bimstec',
        'regional order', 'power balance', 'geopolitics', 'strategic',
        'deterrence', 'containment', 'balance', 'alliance',
        
        # Military & Defense
        'defense', 'defence', 'military', 'armed forces', 'forces',
        'joint exercise', 'bilateral exercise', 'multilateral exercise',
        'drills', 'wargames', 'maneuvers', 'operations',
        
        # Naval & Maritime Security
        'maritime', 'naval', 'navy', 'coast guard', 'coast guard', 'maritime security',
        'piracy', 'anti-piracy', 'submarine', 'destroyer', 'frigate',
        'battleship', 'carrier', 'aircraft carrier', 'amphibious',
        'patrol', 'surveillance', 'monitoring', 'shipping lanes',
        
        # Air & Space
        'air force', 'fighter jet', 'fighter', 'aircraft', 'drone', 'uav',
        'helicopter', 'air defense', 'missile', 'missile defense',
        'space', 'satellite', 'space agency', 'isro', 'jaxa',
        'launch', 'orbital', 'communication satellite',
        
        # Cybersecurity & Technology
        'cyber', 'cybersecurity', 'cyber attack', 'cyber warfare', 'hacking',
        'data', 'internet security', 'iot security', '5g', 'telecom',
        'encryption', 'artificial intelligence', 'ai', 'machine learning',
        'data analytics', 'intelligence', 'surveillance',
        
        # Counter-Terrorism & Intelligence
        'counter-terrorism', 'terrorism', 'terrorist', 'extremism', 'radicalization',
        'intelligence', 'spy', 'espionage', 'reconnaissance', 'surveillance',
        'monitoring', 'border security', 'coastal security',
        
        # Nuclear & Strategic Weapons
        'nuclear', 'nuclear weapons', 'warhead', 'plutonium', 'uranium',
        'enrichment', 'ballistic', 'icbm', 'sbm', 'air-launched',
        'hypersonic', 'cruise missile', 'strategic stability',
        
        # Military Interoperability & Cooperation
        'interoperability', 'interoperable', 'joint', 'combined',
        'coordination', 'communication', 'logistic', 'supply',
        'base', 'port access', 'visiting forces', 'military personnel',
        
        # Internal Security & Conflict
        'terrorism', 'insurgency', 'insurgent', 'militant', 'rebellion',
        'conflict', 'dispute', 'border', 'boundary', 'territorial',
        'sovereignty', 'jurisdiction', 'autonomy',
        
        # Strategic Competition & Great Power
        'great power', 'competition', 'rivalry', 'china', 'russia',
        'us', 'united states', 'hegemony', 'dominance',
        'sphere of influence', 'cold war', 'arms race',
        'escalation', 'provocation', 'aggression'
    }


class StrategicShiftAnalyzer:
    """Analyze the shift from economic to security focus with enhanced capabilities"""
    
    def __init__(self, custom_econ: Optional[set] = None, custom_sec: Optional[set] = None):
        self.economic_lexicon = custom_econ if custom_econ else LexiconDefinitions.ECONOMIC_LEXICON
        self.security_lexicon = custom_sec if custom_sec else LexiconDefinitions.SECURITY_LEXICON
        logger.info(f"Initialized with {len(self.economic_lexicon)} economic terms and {len(self.security_lexicon)} security terms")
    
    def extract_lemmas_for_document(self, doc: dict) -> List[str]:
        """Extract lemmas from preprocessed document"""
        if isinstance(doc.get('lemmas'), list):
            return doc['lemmas']
        return []
    
    def count_category_matches(self, lemmas: List[str], lexicon: set) -> int:
        """Count matches between lemmas and lexicon"""
        lemmas_lower = [l.lower() for l in lemmas]
        return sum(1 for lemma in lemmas_lower if lemma in lexicon)
    
    def calculate_category_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate economic and security scores for each document
        
        Args:
            df: DataFrame with 'lemmas' column
        
        Returns:
            DataFrame with economic_score and security_score columns
        """
        economic_scores = []
        security_scores = []
        
        for idx, row in df.iterrows():
            lemmas = self.extract_lemmas_for_document(row)
            
            eco_count = self.count_category_matches(lemmas, self.economic_lexicon)
            sec_count = self.count_category_matches(lemmas, self.security_lexicon)
            
            # Normalize by document length
            total = len(lemmas) if len(lemmas) > 0 else 1
            economic_scores.append(eco_count / total)
            security_scores.append(sec_count / total)
        
        df['economic_score'] = economic_scores
        df['security_score'] = security_scores
        
        return df
    
    def aggregate_by_year(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate scores by year with statistical measures
        
        Args:
            df: DataFrame with date, economic_score, security_score
        
        Returns:
            DataFrame with yearly aggregated scores
        """
        df['year'] = pd.to_datetime(df['date']).dt.year
        
        yearly_stats = df.groupby('year').agg({
            'economic_score': ['mean', 'std', 'min', 'max', 'count'],
            'security_score': ['mean', 'std', 'min', 'max', 'count']
        }).round(4)
        
        # Flatten column names
        yearly_stats.columns = ['_'.join(col).strip() for col in yearly_stats.columns.values]
        
        return yearly_stats.reset_index()
    
    def identify_crossover_point(self, yearly_df: pd.DataFrame) -> Optional[int]:
        """
        Identify the year where security focus overtook economic focus
        
        Returns:
            Year of crossover, or None if no crossover
        """
        for idx, row in yearly_df.iterrows():
            if row['security_score_mean'] > row['economic_score_mean']:
                return int(row['year'])
        
        return None
    
    def calculate_statistical_significance(self, df: pd.DataFrame) -> Dict:
        """Calculate statistical significance of the shift"""
        try:
            t_stat, p_value = stats.ttest_ind(df['security_score'], df['economic_score'])
            effect_size = (df['security_score'].mean() - df['economic_score'].mean()) / df['security_score'].std() if df['security_score'].std() > 0 else 0
            
            return {
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'significant': bool(p_value < 0.05),
                'effect_size': float(effect_size)
            }
        except Exception as e:
            logger.error(f"Error calculating significance: {str(e)}")
            return None
    
    def calculate_trend(self, yearly_df: pd.DataFrame) -> Dict:
        """Calculate linear trend for both categories"""
        years = yearly_df['year'].values
        econ = yearly_df['economic_score_mean'].values
        sec = yearly_df['security_score_mean'].values
        
        # Linear regression
        econ_coeffs = np.polyfit(years, econ, 1)
        sec_coeffs = np.polyfit(years, sec, 1)
        
        return {
            'economic_slope': float(econ_coeffs[0]),
            'economic_intercept': float(econ_coeffs[1]),
            'security_slope': float(sec_coeffs[0]),
            'security_intercept': float(sec_coeffs[1]),
            'econ_direction': 'Increasing' if econ_coeffs[0] > 0 else 'Decreasing',
            'sec_direction': 'Increasing' if sec_coeffs[0] > 0 else 'Decreasing'
        }
    
    def generate_shift_report(self, df: pd.DataFrame) -> Tuple[Dict, pd.DataFrame, pd.DataFrame]:
        """Build the shift report with scores, yearly aggregation, and stat tests."""
        df_scored = self.calculate_category_scores(df)
        yearly_df = self.aggregate_by_year(df_scored)
        crossover = self.identify_crossover_point(yearly_df)
        stats_result = self.calculate_statistical_significance(df_scored)
        trend_result = self.calculate_trend(yearly_df)
        
        report = {
            'total_documents': len(df),
            'date_range': (str(df['date'].min()), str(df['date'].max())),
            'overall_economic_avg': float(df_scored['economic_score'].mean()),
            'overall_security_avg': float(df_scored['security_score'].mean()),
            'crossover_year': crossover,
            'trend': 'SECURITY' if df_scored['security_score'].mean() > df_scored['economic_score'].mean() else 'ECONOMIC',
            'statistical_significance': stats_result,
            'trend_analysis': trend_result,
            'yearly_stats': yearly_df.to_dict('records')
        }
        
        logger.info(f"Report generated: Trend={report['trend']}, Crossover={crossover}, Significant={stats_result.get('significant')}")
        
        return report, df_scored, yearly_df
    
    def get_top_terms_by_category(self, df: pd.DataFrame, category: str = 'security', top_n: int = 20) -> List[Tuple[str, int]]:
        """
        Get most frequent terms from a specific category
        
        Args:
            df: DataFrame with lemmas
            category: 'economic' or 'security'
            top_n: Number of top terms to return
        
        Returns:
            List of (term, frequency) tuples
        """
        lexicon = self.security_lexicon if category == 'security' else self.economic_lexicon
        
        term_counts = Counter()
        
        for lemmas in df['lemmas']:
            if isinstance(lemmas, list):
                for lemma in lemmas:
                    lemma_lower = lemma.lower()
                    if lemma_lower in lexicon:
                        term_counts[lemma_lower] += 1
        
        return term_counts.most_common(top_n)
    
    def add_custom_terms(self, category: str, terms: List[str]):
        """Add custom terms to lexicon"""
        terms_set = set(t.lower() for t in terms)
        
        if category.lower() == 'economic':
            self.economic_lexicon.update(terms_set)
            logger.info(f"Added {len(terms_set)} custom economic terms")
        elif category.lower() == 'security':
            self.security_lexicon.update(terms_set)
            logger.info(f"Added {len(terms_set)} custom security terms")
    
    def export_lexicons(self, filename: str = None):
        """Export current lexicons"""
        import json
        lexicons = {
            'economic': sorted(list(self.economic_lexicon)),
            'security': sorted(list(self.security_lexicon))
        }
        
        if filename:
            with open(filename, 'w') as f:
                json.dump(lexicons, f, indent=2)
            logger.info(f"Lexicons exported to {filename}")
        
        return lexicons


def main():
    """Test enhanced strategic shift analyzer"""
    analyzer = StrategicShiftAnalyzer()
    
    # Test with sample data
    sample_df = pd.DataFrame({
        'date': pd.date_range('2000-01-01', periods=10, freq='Y'),
        'lemmas': [
            ['oda', 'yen', 'loan', 'metro', 'infrastructure', 'railway'],
            ['trade', 'investment', 'commerce', 'joint', 'venture', 'export'],
            ['defense', 'military', 'security', 'strategic', 'armed', 'forces'],
            ['quad', 'indo-pacific', 'naval', 'exercise', 'maritime', 'security'],
            ['cyber', '5g', 'space', 'technology', 'satellite', 'surveillance'],
            ['security', 'defense', 'naval', 'deterrence', 'alliance', 'cooperation'],
            ['quad', 'maritime', 'indo-pacific', 'alliance', 'military', 'interoperability'],
            ['military', 'strategic', 'cyber', 'intelligence', 'defense', 'cooperation'],
            ['defense', 'security', 'naval', 'indo-pacific', 'quad', 'deterrence'],
            ['quad', 'strategic', 'security', 'alliance', 'military', 'cooperation']
        ]
    })
    
    report, scored_df, yearly_df = analyzer.generate_shift_report(sample_df)
    
    print("\n" + "="*70)
    print("STRATEGIC SHIFT ANALYSIS REPORT - ENHANCED")
    print("="*70)
    print(f"Total Documents: {report['total_documents']}")
    print(f"Date Range: {report['date_range'][0]} to {report['date_range'][1]}")
    print(f"\nOverall Economic Focus: {report['overall_economic_avg']:.4f}")
    print(f"Overall Security Focus: {report['overall_security_avg']:.4f}")
    print(f"Crossover Year: {report['crossover_year']}")
    print(f"Overall Trend: {report['trend']}")
    
    if report['statistical_significance']:
        print(f"\nStatistical Significance:")
        print(f"  T-Statistic: {report['statistical_significance']['t_statistic']:.4f}")
        print(f"  P-Value: {report['statistical_significance']['p_value']:.6f}")
        print(f"  Significant: {report['statistical_significance']['significant']}")
        print(f"  Effect Size: {report['statistical_significance']['effect_size']:.4f}")
    
    if report['trend_analysis']:
        print(f"\nTrend Analysis:")
        print(f"  Economic Slope: {report['trend_analysis']['economic_slope']:.6f} ({report['trend_analysis']['econ_direction']})")
        print(f"  Security Slope: {report['trend_analysis']['security_slope']:.6f} ({report['trend_analysis']['sec_direction']})")
    
    print(f"\nYearly Statistics:")
    print(yearly_df.to_string())
    print("="*70)


if __name__ == "__main__":
    main()
