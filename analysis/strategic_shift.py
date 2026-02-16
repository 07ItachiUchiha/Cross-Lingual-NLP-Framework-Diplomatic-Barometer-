"""
Strategic Shift Quantifier Module
Analyzes frequency of economic vs. security-related terms over time
"""

import pandas as pd
import numpy as np
from collections import Counter
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LexiconDefinitions:
    """Pre-defined lexicons for economic and security categories"""
    
    # Era 1: Economic/Aid focus
    ECONOMIC_LEXICON = {
        'oda', 'yen loan', 'infrastructure', 'investment', 'trade',
        'dmic', 'metro', 'assistance', 'development', 'economic',
        'aid', 'loan', 'railway', 'corridor', 'industrial',
        'export', 'import', 'commerce', 'business', 'cooperation',
        'manufacturing', 'technology transfer', 'joint venture',
        'automotive', 'semiconductor', 'pharmaceutical'
    }
    
    # Era 2: Security/Strategy focus
    SECURITY_LEXICON = {
        'indo-pacific', 'quad', 'maritime', 'defense', 'cyber',
        'space', '5g', 'security', 'drone', 'interoperability',
        'military', 'strategic', 'deterrence', 'submarine', 'destroyer',
        'fighter', 'coast guard', 'joint exercise', 'alliance',
        'counter-terrorism', 'surveillance', 'intelligence',
        'nuclear', 'missile', 'radar', 'naval', 'aircraft carrier',
        'hyper-sonic', 'artificial intelligence', 'encryption'
    }


class StrategicShiftAnalyzer:
    """Analyze the shift from economic to security focus"""
    
    def __init__(self):
        self.economic_lexicon = LexiconDefinitions.ECONOMIC_LEXICON
        self.security_lexicon = LexiconDefinitions.SECURITY_LEXICON
    
    def extract_lemmas_for_document(self, doc: dict) -> List[str]:
        """Extract lemmas from preprocessed document"""
        if isinstance(doc.get('lemmas'), list):
            return doc['lemmas']
        return []
    
    def count_category_matches(self, lemmas: List[str], lexicon: set) -> int:
        """Count matches between lemmas and lexicon"""
        lemmas_set = set(lemmas)
        return len(lemmas_set.intersection(lexicon))
    
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
        Aggregate scores by year
        
        Args:
            df: DataFrame with date, economic_score, security_score
        
        Returns:
            DataFrame with yearly aggregated scores
        """
        df['year'] = pd.to_datetime(df['date']).dt.year
        
        yearly_stats = df.groupby('year').agg({
            'economic_score': ['mean', 'sum', 'count'],
            'security_score': ['mean', 'sum', 'count']
        }).round(4)
        
        # Flatten column names
        yearly_stats.columns = ['_'.join(col).strip() for col in yearly_stats.columns.values]
        
        return yearly_stats.reset_index()
    
    def identify_crossover_point(self, yearly_df: pd.DataFrame) -> int:
        """
        Identify the year where security focus overtook economic focus
        
        Returns:
            Year of crossover, or None if no crossover
        """
        for idx, row in yearly_df.iterrows():
            if row['security_score_mean'] > row['economic_score_mean']:
                return int(row['year'])
        
        return None
    
    def generate_shift_report(self, df: pd.DataFrame) -> Dict:
        """Generate the full shift report dict."""
        df_scored = self.calculate_category_scores(df)
        yearly_df = self.aggregate_by_year(df_scored)
        crossover = self.identify_crossover_point(yearly_df)
        
        report = {
            'total_documents': len(df),
            'date_range': (df['date'].min(), df['date'].max()),
            'overall_economic_avg': df_scored['economic_score'].mean(),
            'overall_security_avg': df_scored['security_score'].mean(),
            'crossover_year': crossover,
            'yearly_stats': yearly_df.to_dict('records'),
            'trend': 'SECURITY' if df_scored['security_score'].mean() > df_scored['economic_score'].mean() else 'ECONOMIC'
        }
        
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
                    if lemma in lexicon:
                        term_counts[lemma] += 1
        
        return term_counts.most_common(top_n)


def main():
    """Test strategic shift analyzer"""
    analyzer = StrategicShiftAnalyzer()
    
    # Test with sample data
    sample_df = pd.DataFrame({
        'date': pd.date_range('2000-01-01', periods=10, freq='Y'),
        'lemmas': [
            ['oda', 'yen', 'loan', 'metro', 'infrastructure'],
            ['trade', 'investment', 'commerce', 'joint', 'venture'],
            ['defense', 'military', 'security', 'strategic'],
            ['quad', 'indo-pacific', 'naval', 'exercise'],
            ['cyber', '5g', 'space', 'technology'],
            ['security', 'defense', 'naval', 'deterrence'],
            ['quad', 'maritime', 'Indo-Pacific', 'alliance'],
            ['military', 'strategic', 'cyber', 'intelligence'],
            ['defense', 'security', 'naval', 'Indo-Pacific'],
            ['quad', 'strategic', 'security', 'alliance']
        ]
    })
    
    report, scored_df, yearly_df = analyzer.generate_shift_report(sample_df)
    
    print("Strategic Shift Analysis Report")
    print("=" * 50)
    print(f"Total Documents: {report['total_documents']}")
    print(f"Overall Economic Focus: {report['overall_economic_avg']:.4f}")
    print(f"Overall Security Focus: {report['overall_security_avg']:.4f}")
    print(f"Crossover Year: {report['crossover_year']}")
    print(f"Overall Trend: {report['trend']}")
    print("\nYearly Statistics:")
    print(yearly_df)


if __name__ == "__main__":
    main()
