"""Sentiment polarity + urgency scoring for diplomatic text"""

import pandas as pd
from typing import Dict, List
import logging
from textblob import TextBlob
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UrgencyDetector:
    """Detect urgency in diplomatic language using modal verbs"""
    
    # Modal verbs indicating different levels of urgency
    URGENT_MODALS = {'must', 'shall', 'will', 'essential', 'critical', 'urgent'}
    MODERATE_MODALS = {'should', 'ought', 'need', 'important', 'necessary'}
    CAUTIOUS_MODALS = {'may', 'might', 'could', 'consider', 'hope', 'suggest', 'propose'}
    
    @staticmethod
    def extract_modal_verbs(text: str) -> Dict[str, int]:
        """Extract and count modal verbs"""
        text_lower = text.lower()
        
        urgent_count = sum(1 for modal in UrgencyDetector.URGENT_MODALS if modal in text_lower)
        moderate_count = sum(1 for modal in UrgencyDetector.MODERATE_MODALS if modal in text_lower)
        cautious_count = sum(1 for modal in UrgencyDetector.CAUTIOUS_MODALS if modal in text_lower)
        
        return {
            'urgent_modals': urgent_count,
            'moderate_modals': moderate_count,
            'cautious_modals': cautious_count
        }
    
    @staticmethod
    def calculate_urgency_score(text: str) -> float:
        """
        Calculate urgency score (0-1)
        
        Higher score = more urgent language
        """
        modals = UrgencyDetector.extract_modal_verbs(text)
        
        total_modals = sum(modals.values())
        if total_modals == 0:
            return 0.5  # Neutral
        
        # Weighted urgency calculation
        urgency_value = (
            modals['urgent_modals'] * 1.0 +
            modals['moderate_modals'] * 0.6 +
            modals['cautious_modals'] * 0.2
        )
        
        # Normalize to 0-1 range
        urgency_score = min(urgency_value / total_modals, 1.0)
        
        return urgency_score


class SentimentAnalyzer:
    """Analyze sentiment and tone in diplomatic documents"""
    
    @staticmethod
    def analyze_sentiment(text: str) -> Dict:
        """
        Analyze sentiment using TextBlob
        
        Returns:
            Dictionary with polarity and subjectivity scores
        """
        if not isinstance(text, str) or len(text) == 0:
            return {'polarity': 0.0, 'subjectivity': 0.0}
        
        blob = TextBlob(text)
        
        return {
            'polarity': blob.sentiment.polarity,  # -1 (negative) to 1 (positive)
            'subjectivity': blob.sentiment.subjectivity  # 0 (objective) to 1 (subjective)
        }
    
    @staticmethod
    def classify_sentiment(polarity: float) -> str:
        """Classify sentiment based on polarity"""
        if polarity > 0.1:
            return 'Positive'
        elif polarity < -0.1:
            return 'Negative'
        else:
            return 'Neutral'
    
    @staticmethod
    def classify_tone(urgency: float, subjectivity: float) -> str:
        """
        Classify tone based on urgency and subjectivity
        
        Returns:
            Tone classification
        """
        if urgency > 0.7 and subjectivity < 0.3:
            return 'Assertive'
        elif urgency > 0.7:
            return 'Emphatic'
        elif subjectivity > 0.7:
            return 'Expressive'
        elif subjectivity < 0.3:
            return 'Formal'
        else:
            return 'Balanced'


class ToneAnalyzer:
    """Main tone analysis pipeline"""
    
    def __init__(self):
        self.urgency_detector = UrgencyDetector()
        self.sentiment_analyzer = SentimentAnalyzer()
    
    def analyze_document_tone(self, text: str) -> Dict:
        """Run all tone metrics on a single document."""
        urgency_score = self.urgency_detector.calculate_urgency_score(text)
        sentiment = self.sentiment_analyzer.analyze_sentiment(text)
        sentiment_class = self.sentiment_analyzer.classify_sentiment(sentiment['polarity'])
        tone_class = self.sentiment_analyzer.classify_tone(urgency_score, sentiment['subjectivity'])
        
        return {
            'urgency_score': urgency_score,
            'sentiment_polarity': sentiment['polarity'],
            'sentiment_subjectivity': sentiment['subjectivity'],
            'sentiment_class': sentiment_class,
            'tone_class': tone_class
        }
    
    def process_dataframe(self, df: pd.DataFrame, text_column: str = 'cleaned') -> pd.DataFrame:
        """
        Process all documents in a DataFrame
        
        Args:
            df: DataFrame with documents
            text_column: Column name containing text to analyze
        
        Returns:
            DataFrame with added tone columns
        """
        logger.info(f"Analyzing tone for {len(df)} documents...")
        
        if text_column not in df.columns:
            logger.error(f"Column '{text_column}' not found in dataframe. Available columns: {list(df.columns)}")
            return pd.DataFrame()
        
        # Check if column has data
        non_empty_count = df[text_column].apply(lambda x: len(str(x).strip()) > 0).sum()
        logger.info(f"Non-empty {text_column} texts: {non_empty_count}/{len(df)}")
        
        if non_empty_count == 0:
            logger.warning(f"All texts in '{text_column}' column are empty!")
            return pd.DataFrame()
        
        tone_data = []
        for idx, text in enumerate(df[text_column]):
            if (idx + 1) % max(1, len(df) // 5) == 0 or idx == 0:
                logger.info(f"Processed {idx + 1}/{len(df)} documents")
            
            tone_data.append(self.analyze_document_tone(text))
        
        logger.info(f"Processed {len(df)}/{len(df)} documents - Tone analysis complete")
        
        tone_df = pd.DataFrame(tone_data)
        
        # Combine with original dataframe
        result = pd.concat([df.reset_index(drop=True), tone_df], axis=1)
        
        return result
    
    def get_yearly_tone_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get average tone statistics by year
        
        Returns:
            DataFrame with yearly tone statistics
        """
        df['year'] = pd.to_datetime(df['date']).dt.year
        
        yearly_stats = df.groupby('year').agg({
            'urgency_score': ['mean', 'std'],
            'sentiment_polarity': ['mean', 'std'],
            'sentiment_subjectivity': ['mean', 'std']
        }).round(4)
        
        yearly_stats.columns = ['_'.join(col).strip() for col in yearly_stats.columns.values]
        
        return yearly_stats.reset_index()
    
    def get_tone_distribution(self, df: pd.DataFrame) -> Dict:
        """Get distribution of tone classes across all documents"""
        return df['tone_class'].value_counts().to_dict()
    
    def get_sentiment_distribution(self, df: pd.DataFrame) -> Dict:
        """Get distribution of sentiment classes across all documents"""
        return df['sentiment_class'].value_counts().to_dict()


def main():
    """Test tone analyzer"""
    analyzer = ToneAnalyzer()
    
    test_texts = [
        "We must strengthen our strategic partnership. This is essential.",
        "We hope to consider future cooperation in trade and commerce.",
        "We shall enhance defense and military cooperation through joint exercises.",
        "The two nations recognize the importance of economic development."
    ]
    
    print("Tone Analysis Test")
    print("=" * 60)
    
    for text in test_texts:
        result = analyzer.analyze_document_tone(text)
        print(f"\nText: {text[:50]}...")
        print(f"Urgency Score: {result['urgency_score']:.3f}")
        print(f"Sentiment: {result['sentiment_class']} ({result['sentiment_polarity']:.3f})")
        print(f"Tone: {result['tone_class']}")


if __name__ == "__main__":
    main()
