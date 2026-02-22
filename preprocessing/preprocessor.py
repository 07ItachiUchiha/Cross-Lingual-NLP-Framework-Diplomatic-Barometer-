"""
Data Preprocessing Module
Handles text cleaning, tokenization, lemmatization, and entity recognition
"""

import re
import logging
import pandas as pd
from typing import List, Tuple, Optional, Iterable
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk

try:
    import spacy
    from spacy.matcher import PhraseMatcher
    SPACY_IMPORT_ERROR = None
except Exception as exc:
    spacy = None
    PhraseMatcher = None
    SPACY_IMPORT_ERROR = exc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


class TextCleaner:
    """Clean raw diplomatic text"""
    
    @staticmethod
    def remove_html_tags(text: str) -> str:
        """Remove HTML tags from text"""
        clean = re.compile('<.*?>')
        return re.sub(clean, '', text)
    
    @staticmethod
    def remove_boilerplate(text: str) -> str:
        """Remove common diplomatic boilerplate"""
        boilerplate_patterns = [
            r'(?i)honorable|excellency|your (?:excellency|honor)',
            r'(?i)without prejudice',
            r'(?i)sincerely yours',
            r'(?i)regards',
            r'(?i)respectfully'
        ]
        
        for pattern in boilerplate_patterns:
            text = re.sub(pattern, '', text)
        
        return text
    
    @staticmethod
    def remove_headers_footers(text: str) -> str:
        """Remove headers, footers, and page numbers"""
        # Remove page breaks and excessive whitespace
        text = re.sub(r'\n\s*\n', '\n', text)
        text = re.sub(r'Page \d+|p\. \d+|\[\d+\]', '', text)
        return text
    
    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """Normalize whitespace"""
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Apply all cleaning operations"""
        if not isinstance(text, str):
            return ""
        
        text = TextCleaner.remove_html_tags(text)
        text = TextCleaner.remove_boilerplate(text)
        text = TextCleaner.remove_headers_footers(text)
        text = TextCleaner.normalize_whitespace(text)
        
        return text


class Tokenizer:
    """Tokenize and process text"""
    
    def __init__(self, phrase_terms: Optional[Iterable[str]] = None):
        self.nlp = None
        self.phrase_terms = [str(t).strip() for t in (phrase_terms or []) if str(t).strip()]
        self._phrase_matcher = None
        if spacy is None:
            logger.warning(
                "SpaCy import unavailable. Falling back to NLTK tokenization/lemmatization-free mode. "
                f"Reason: {SPACY_IMPORT_ERROR}"
            )
        else:
            try:
                self.nlp = spacy.load("en_core_web_sm")
                self._phrase_matcher = self._build_phrase_matcher(self.phrase_terms)
            except Exception as exc:
                logger.warning(
                    "SpaCy model/runtime not available. Falling back to NLTK tokenization mode. "
                    f"Reason: {exc}. Install model with: python -m spacy download en_core_web_sm"
                )
        
        self.stop_words = set(stopwords.words('english'))
        
        # Add diplomatic-specific stop words
        self.diplomatic_stopwords = {
            'shall', 'will', 'may', 'must', 'considering', 'noting',
            'recognizing', 'reaffirming', 'expressing', 'emphasizing'
        }
        
        self.stop_words.update(self.diplomatic_stopwords)

    @staticmethod
    def _norm(text: str) -> str:
        return re.sub(r"\s+", " ", str(text or "").strip().lower())

    @staticmethod
    def _term_variants(term: str) -> List[str]:
        base = Tokenizer._norm(term)
        if not base:
            return []
        variants = {base, base.replace("-", " ")}
        if " " in base:
            variants.add(base.replace(" ", ""))
        out = []
        seen = set()
        for v in variants:
            v2 = re.sub(r"\s+", " ", v).strip()
            if v2 and v2 not in seen:
                seen.add(v2)
                out.append(v2)
        return out

    def _build_phrase_matcher(self, terms: List[str]):
        if self.nlp is None or PhraseMatcher is None or not terms:
            return None

        matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        patterns = []
        for t in terms:
            for v in self._term_variants(t):
                if " " not in v:
                    continue
                try:
                    patterns.append(self.nlp.make_doc(v))
                except Exception:
                    continue
        if patterns:
            matcher.add("PHRASE", patterns)
        return matcher

    def _merge_phrases(self, doc):
        """Merge matched phrase spans into single tokens with underscore lemma."""
        if self._phrase_matcher is None:
            return doc

        matches = list(self._phrase_matcher(doc))
        if not matches:
            return doc

        spans = sorted({doc[start:end] for _, start, end in matches}, key=lambda s: (-(s.end - s.start), s.start))

        with doc.retokenize() as retok:
            for span in spans:
                if span.start >= span.end:
                    continue
                merged = self._norm(span.text).replace(" ", "_")
                try:
                    retok.merge(span, attrs={"LEMMA": merged, "ORTH": merged})
                except Exception:
                    continue
        return doc
    
    def sentence_tokenize(self, text: str) -> List[str]:
        """Split text into sentences"""
        return sent_tokenize(text)
    
    def word_tokenize(self, text: str) -> List[str]:
        """Tokenize text into words"""
        tokens = word_tokenize(text.lower())
        return tokens
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Remove stopwords from token list"""
        return [token for token in tokens if token not in self.stop_words and len(token) > 2]
    
    def lemmatize(self, text: str) -> List[str]:
        """Lemmatize text using spaCy"""
        if not self.nlp:
            out_text = str(text or "")
            lowered = out_text.lower()
            for t in self.phrase_terms:
                for v in self._term_variants(t):
                    if not v or " " not in v:
                        continue
                    if v in lowered:
                        out_text = re.sub(re.escape(v), v.replace(" ", "_"), out_text, flags=re.IGNORECASE)
                        lowered = out_text.lower()
            return self.word_tokenize(out_text)
        
        doc = self.nlp(str(text or "").lower())
        doc = self._merge_phrases(doc)
        lemmas = [token.lemma_ for token in doc if not token.is_punct]
        return lemmas
    
    def named_entity_recognition(self, text: str) -> List[Tuple[str, str]]:
        """Extract named entities"""
        if not self.nlp:
            return []
        
        doc = self.nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        return entities


class Preprocessor:
    """Main preprocessing pipeline"""
    
    def __init__(self, phrase_terms: Optional[Iterable[str]] = None):
        self.cleaner = TextCleaner()

        # Default phrase terms: strategic-shift lexicons (helps preserve Indo-Pacific, joint exercise, yen loan, etc.)
        if phrase_terms is None:
            try:
                from analysis.strategic_shift_enhanced import LexiconDefinitions

                phrase_terms = sorted(set(LexiconDefinitions.ECONOMIC_LEXICON) | set(LexiconDefinitions.SECURITY_LEXICON))
            except Exception:
                phrase_terms = []

        self.tokenizer = Tokenizer(phrase_terms=phrase_terms)
    
    def preprocess_text(self, text: str) -> dict:
        """
        Full preprocessing pipeline
        
        Returns:
            Dictionary with cleaned text, tokens, lemmas, and entities
        """
        if not isinstance(text, str) or len(text) == 0:
            return {
                'original': text,
                'cleaned': '',
                'sentences': [],
                'tokens': [],
                'tokens_no_stopwords': [],
                'lemmas': [],
                'entities': []
            }
        
        # Clean text
        cleaned = self.cleaner.clean_text(text)
        
        # Sentence tokenization
        sentences = self.tokenizer.sentence_tokenize(cleaned)
        
        # Word tokenization
        tokens = self.tokenizer.word_tokenize(cleaned)
        
        # Remove stopwords
        tokens_filtered = self.tokenizer.remove_stopwords(tokens)
        
        # Lemmatization
        lemmas = self.tokenizer.lemmatize(cleaned)
        
        # Named entity recognition
        entities = self.tokenizer.named_entity_recognition(text)
        
        return {
            'original': text[:100] + '...' if len(text) > 100 else text,
            'cleaned': cleaned,
            'sentences': sentences,
            'tokens': tokens,
            'tokens_no_stopwords': tokens_filtered,
            'lemmas': lemmas,
            'entities': entities
        }
    
    def process_dataframe(self, df: pd.DataFrame, content_column: str = 'content') -> pd.DataFrame:
        """
        Process all documents in a DataFrame
        
        Args:
            df: DataFrame with documents
            content_column: Column name containing text to process
        
        Returns:
            DataFrame with added processed columns
        """
        logger.info(f"Processing {len(df)} documents...")
        
        processed_data = []
        for idx, text in enumerate(df[content_column]):
            if (idx + 1) % max(1, len(df) // 5) == 0 or idx == 0:
                logger.info(f"Processed {idx + 1}/{len(df)} documents")
            
            processed_data.append(self.preprocess_text(text))
        
        logger.info(f"Processed {len(df)}/{len(df)} documents - Processing complete")
        
        processed_df = pd.DataFrame(processed_data)
        
        # Combine with original dataframe
        result = pd.concat([df.reset_index(drop=True), processed_df], axis=1)
        
        return result


def main():
    """Test preprocessing"""
    test_text = """
    The Government of India and the Government of Japan, reaffirming their commitment
    to the Indo-Pacific region and recognizing the importance of maritime security,
    have decided to enhance cooperation in defense and strategic matters. Both nations
    express their dedication to the Quad and multilateral cooperation frameworks.
    """
    
    preprocessor = Preprocessor()
    result = preprocessor.preprocess_text(test_text)
    
    print("Cleaned text:", result['cleaned'][:100])
    print("Tokens:", result['tokens'][:10])
    print("Lemmas:", result['lemmas'][:10])
    print("Entities:", result['entities'])


if __name__ == "__main__":
    main()
