"""LDA-based topic modeling for diplomatic documents"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import logging
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from collections import Counter
from scipy import sparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TopicModeler:
    """Perform topic modeling using LDA"""
    
    def __init__(self, n_topics: int = 5, max_features: int = 1000):
        self.n_topics = n_topics
        self.max_features = max_features
        self.vectorizer = None
        self.lda_model = None
        self.feature_names = None
    
    def prepare_documents(self, texts: List[str]) -> np.ndarray:
        """
        Convert documents to term-frequency matrix
        
        Args:
            texts: List of document texts
        
        Returns:
            Document-term matrix
        """
        # Enhanced stopwords - remove common diplomacy words that don't distinguish topics
        custom_stopwords = {
            'india', 'japan', 'delhi', 'tokyo', 'bilateral', 'statement', 
            'visit', 'dialogue', 'pm', 'minister', 'official', 'say', 'said',
            'would', 'also', 'one', 'new', 'two', 'country', 'countries',
            'discuss', 'discussed', 'discussion', 'meeting', 'met', 'meet',
            'ambassador', 'envoy', 'delegation', 'government', 'foreign',
            'international', 'global', 'world', 'state', 'year',
            'time', 'way', 'people', 'mr', 'ms', 'dr'
        }
        
        # Create a temporary vectorizer to get default English stopwords
        temp_vectorizer = CountVectorizer(stop_words='english')
        english_stops = set(temp_vectorizer.get_stop_words())
        
        # Combine stopwords and convert to list
        all_stopwords = list(english_stops | custom_stopwords)
        
        def _make_vectorizer(max_df_value: float, stop_words_value):
            return CountVectorizer(
                max_features=self.max_features,
                stop_words=stop_words_value,
                min_df=1,
                max_df=max_df_value,
                token_pattern=r'(?u)\b[a-z]{3,}\b',
                ngram_range=(1, 2),
            )

        # Default settings: prune extremely common terms. If the corpus is very uniform,
        # this can remove everything; we retry with max_df=1.0.
        self.vectorizer = _make_vectorizer(0.9, all_stopwords)

        try:
            doc_term_matrix = self.vectorizer.fit_transform(texts)
        except ValueError as exc:
            msg = str(exc)
            if "After pruning, no terms remain" in msg:
                logger.warning("Vectorizer pruned all terms; retrying with max_df=1.0")
                self.vectorizer = _make_vectorizer(1.0, all_stopwords)
                try:
                    doc_term_matrix = self.vectorizer.fit_transform(texts)
                except ValueError:
                    # As a last resort, drop custom stopwords and retry once.
                    logger.warning("Retrying vectorizer with max_df=1.0 and no custom stopwords")
                    self.vectorizer = _make_vectorizer(1.0, 'english')
                    try:
                        doc_term_matrix = self.vectorizer.fit_transform(texts)
                    except ValueError:
                        self.feature_names = np.array([])
                        return sparse.csr_matrix((len(texts), 0), dtype=np.int64)
            else:
                raise

        self.feature_names = self.vectorizer.get_feature_names_out()
        
        logger.info(f"Created document-term matrix: {doc_term_matrix.shape}")
        logger.info(f"Vocabulary size: {len(self.feature_names)}")
        
        return doc_term_matrix
    
    def fit_lda(self, doc_term_matrix: np.ndarray, max_iter: int = 50) -> LatentDirichletAllocation:
        """
        Fit LDA model to documents
        
        Args:
            doc_term_matrix: Document-term matrix from prepare_documents
            max_iter: Maximum iterations for model fitting (default 50 for better convergence)
        
        Returns:
            Fitted LDA model
        """
        self.lda_model = LatentDirichletAllocation(
            n_components=self.n_topics,
            max_iter=max_iter,
            learning_method='online',  # More stable
            learning_offset=50.0,  # Weight for early documents
            random_state=42,
            n_jobs=1,  # Avoid joblib parallelism for stability across environments
            verbose=0,
            topic_word_prior=0.01,  # Lower = more sparse (distinct) topics
            doc_topic_prior=0.01  # Lower = each doc focuses on fewer topics
        )
        
        self.lda_model.fit(doc_term_matrix)
        
        logger.info(f"Fitted LDA model with {self.n_topics} topics using {max_iter} iterations")
        
        return self.lda_model
    
    def get_top_words_per_topic(self, n_words: int = 12) -> Dict[int, List[str]]:
        """
        Get top words for each topic
        
        Args:
            n_words: Number of top words to extract per topic (default 12 for better clarity)
        
        Returns:
            Dictionary mapping topic ID to list of top words
        """
        if self.lda_model is None or self.feature_names is None:
            logger.error("LDA model not fitted yet")
            return {}
        
        topics = {}
        
        for topic_idx, topic in enumerate(self.lda_model.components_):
            top_indices = topic.argsort()[-n_words:][::-1]
            top_words = [self.feature_names[i] for i in top_indices]
            topics[topic_idx] = top_words
        
        return topics
    
    def get_document_topics(self, doc_term_matrix: np.ndarray) -> np.ndarray:
        """
        Get topic distribution for each document
        
        Returns:
            Array of shape (n_documents, n_topics) with topic probabilities
        """
        if self.lda_model is None:
            logger.error("LDA model not fitted yet")
            return None
        
        return self.lda_model.transform(doc_term_matrix)
    
    def get_primary_topic_per_document(self, doc_term_matrix: np.ndarray) -> List[int]:
        """Get the primary (most likely) topic for each document"""
        doc_topics = self.get_document_topics(doc_term_matrix)
        return np.argmax(doc_topics, axis=1).tolist()


class ThematicAnalyzer:
    """Analyze themes over time"""
    
    def __init__(self, n_topics: int = 5):
        self.modeler = TopicModeler(n_topics=n_topics)
    
    def identify_themes_by_year(self, df: pd.DataFrame, 
                               text_column: str = 'cleaned') -> Dict:
        """
        Identify dominant themes for each year
        
        Args:
            df: DataFrame with 'date' and text column
            text_column: Column containing document text
        
        Returns:
            Dictionary with yearly theme analysis
        """
        df['year'] = pd.to_datetime(df['date']).dt.year
        
        yearly_themes = {}
        
        for year in sorted(df['year'].unique()):
            year_data = df[df['year'] == year]
            
            if len(year_data) < 2:
                continue
            
            # Prepare documents for this year
            year_texts = year_data[text_column].fillna('').astype(str).tolist()
            doc_term_matrix = self.modeler.prepare_documents(year_texts)
            
            # Fit LDA for this year
            self.modeler.fit_lda(doc_term_matrix, max_iter=10)
            
            # Get topics
            top_words = self.modeler.get_top_words_per_topic(n_words=5)
            
            yearly_themes[year] = {
                'n_documents': len(year_data),
                'topics': top_words
            }
        
        return yearly_themes
    
    def analyze_theme_evolution(self, df: pd.DataFrame,
                               text_column: str = 'cleaned') -> Dict:
        """
        Analyze how themes evolve over time
        
        Returns:
            Dictionary with theme evolution analysis
        """
        logger.info("Performing thematic analysis...")
        
        # Prepare all documents
        all_texts = df[text_column].fillna('').astype(str).tolist()
        doc_term_matrix = self.modeler.prepare_documents(all_texts)

        # If we cannot build a vocabulary (e.g., extremely uniform text), return a safe empty result.
        try:
            n_features = int(getattr(doc_term_matrix, "shape", (0, 0))[1])
        except Exception:
            n_features = 0

        if n_features == 0:
            df['primary_topic'] = -1
            df['year'] = pd.to_datetime(df['date']).dt.year
            analysis = {
                'n_topics': 0,
                'overall_themes': {},
                'topic_distribution_by_year': {},
                'topic_weights_by_year': {},
                'documents_with_primary_topic': df[['year', 'primary_topic']].to_dict('records'),
                'warning': 'No vocabulary available for topic modeling (corpus may be too uniform or too small).',
            }
            logger.warning(analysis['warning'])
            return analysis, df
        
        # Fit LDA model
        self.modeler.fit_lda(doc_term_matrix)
        
        # Get overall themes
        themes = self.modeler.get_top_words_per_topic(n_words=8)
        
        # Get document-topic assignments (global model)
        doc_topics = self.modeler.get_document_topics(doc_term_matrix)
        primary_topics = np.argmax(doc_topics, axis=1).tolist() if doc_topics is not None else []
        df['primary_topic'] = primary_topics
        
        # Analyze topic distribution by year
        df['year'] = pd.to_datetime(df['date']).dt.year
        
        # Comparable per-year topic weights using the same global topic space.
        # For each year, compute mean topic probability across docs in that year.
        topic_weights_by_year: Dict[int, List[Tuple[int, float]]] = {}
        if doc_topics is not None and len(doc_topics) == len(df):
            df_topics = df[['year']].copy()
            for t in range(self.modeler.n_topics):
                df_topics[f'topic_{t}'] = doc_topics[:, t]
            for year, g in df_topics.groupby('year'):
                weights = []
                for t in range(self.modeler.n_topics):
                    weights.append((int(t), float(pd.to_numeric(g[f'topic_{t}'], errors='coerce').fillna(0.0).mean())))
                weights = sorted(weights, key=lambda x: x[1], reverse=True)
                topic_weights_by_year[int(year)] = weights

        # Backward compatible: topic_distribution_by_year
        # Previously was counts of primary_topic; now return top topics with mean weights.
        yearly_topic_dist: Dict[int, List[Tuple[int, float]]] = {
            int(y): v[: max(1, min(5, self.modeler.n_topics))] for y, v in topic_weights_by_year.items()
        }
        
        analysis = {
            'n_topics': self.modeler.n_topics,
            'overall_themes': themes,
            'topic_distribution_by_year': yearly_topic_dist,
            'topic_weights_by_year': topic_weights_by_year,
            'documents_with_primary_topic': df[['year', 'primary_topic']].to_dict('records')
        }
        
        logger.info("Thematic analysis complete")
        
        return analysis, df


def main():
    """Test thematic analyzer"""
    analyzer = ThematicAnalyzer(n_topics=3)
    
    # Sample documents
    sample_df = pd.DataFrame({
        'date': pd.date_range('2010-01-01', periods=6, freq='Y'),
        'cleaned': [
            'Infrastructure development metro railway dmic investment yen loan',
            'Trade export import commerce economic partnership',
            'Defense military strategic security cooperation',
            'Quad maritime Indo-Pacific naval exercise',
            'Cyber security 5G technology artificial intelligence',
            'Strategic alliance military deterrence naval cooperation'
        ]
    })
    
    analysis, df_result = analyzer.analyze_theme_evolution(sample_df)
    
    print("Thematic Analysis Report")
    print("=" * 60)
    print(f"Number of Topics: {analysis['n_topics']}")
    print("\nOverall Themes:")
    for topic_id, words in analysis['overall_themes'].items():
        print(f"  Topic {topic_id}: {', '.join(words)}")


if __name__ == "__main__":
    main()
