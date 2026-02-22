"""Scores each document on economic vs. security focus.

Important correctness note:
- Lexicons include many multi-word phrases (e.g. "yen loan", "joint exercise").
- Token-by-token lemma matching will never match those phrases.

This module therefore scores primarily against the *cleaned text* using a phrase matcher when possible,
falling back to token n-grams when spaCy is unavailable.
"""

import pandas as pd
import numpy as np
from collections import Counter
from typing import Dict, List, Tuple, Optional
import logging
from scipy import stats
import re

try:
    import spacy
    from spacy.matcher import PhraseMatcher
except Exception:  # pragma: no cover
    spacy = None
    PhraseMatcher = None

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

        self._spacy_nlp = None
        self._econ_matcher = None
        self._sec_matcher = None
        if spacy is not None and PhraseMatcher is not None:
            try:
                # Use a lightweight tokenizer-only pipeline; avoids requiring en_core_web_sm.
                self._spacy_nlp = spacy.blank("en")
                self._econ_matcher = self._build_phrase_matcher(self.economic_lexicon)
                self._sec_matcher = self._build_phrase_matcher(self.security_lexicon)
            except Exception as exc:
                logger.warning(f"Phrase matcher unavailable; falling back to n-gram matching. Reason: {exc}")
                self._spacy_nlp = None
                self._econ_matcher = None
                self._sec_matcher = None

    @staticmethod
    def _normalize_term(term: str) -> str:
        s = str(term or "").strip().lower()
        s = re.sub(r"\s+", " ", s)
        return s

    @staticmethod
    def _term_variants(term: str) -> List[str]:
        """Generate conservative variants for matching hyphenated/compounded terms."""
        base = StrategicShiftAnalyzer._normalize_term(term)
        if not base:
            return []

        variants = {base}

        # Hyphen and dash normalization
        variants.add(base.replace("-", " "))
        variants.add(base.replace("–", " ").replace("—", " "))

        # Collapse spaces for "indo pacific" -> "indopacific"
        if " " in base and len(base.replace(" ", "")) >= 6:
            variants.add(base.replace(" ", ""))

        # De-duplicate whitespace
        out = []
        seen = set()
        for v in variants:
            v2 = re.sub(r"\s+", " ", v).strip()
            if v2 and v2 not in seen:
                seen.add(v2)
                out.append(v2)
        return out

    def _build_phrase_matcher(self, lexicon: set) -> Optional[PhraseMatcher]:
        if self._spacy_nlp is None or PhraseMatcher is None:
            return None
        matcher = PhraseMatcher(self._spacy_nlp.vocab, attr="LOWER")
        patterns = []
        for raw in sorted({self._normalize_term(t) for t in lexicon if str(t).strip()}):
            for v in self._term_variants(raw):
                try:
                    patterns.append(self._spacy_nlp.make_doc(v))
                except Exception:
                    continue
        if patterns:
            matcher.add("LEX", patterns)
        return matcher

    @staticmethod
    def _simple_tokens(text: str) -> List[str]:
        # Keep alphanumerics + hyphen within tokens
        return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", str(text or "").lower())

    def _count_matches_text(self, cleaned_text: str, lexicon: set, matcher: Optional[PhraseMatcher]) -> Tuple[int, int]:
        """Return (match_count, token_count) for cleaned_text against lexicon."""

        text = str(cleaned_text or "").strip()
        if not text:
            return 0, 1

        # Preferred: spaCy PhraseMatcher
        if self._spacy_nlp is not None and matcher is not None:
            doc = self._spacy_nlp(text)
            matches = matcher(doc)
            # Count occurrences (not just unique terms)
            return int(len(matches)), max(1, int(len(doc)))

        # Fallback: n-gram matching over simple tokens
        tokens = self._simple_tokens(text)
        if not tokens:
            return 0, 1

        lex = {self._normalize_term(t) for t in lexicon}
        # include variants so hyphenated forms can match
        lex_expanded = set()
        for t in lex:
            lex_expanded.update(self._term_variants(t))

        count = 0
        # 1-gram
        for tok in tokens:
            if tok in lex_expanded:
                count += 1
        # 2-gram / 3-gram
        for n in (2, 3, 4, 5):
            for i in range(0, len(tokens) - n + 1):
                phrase = " ".join(tokens[i : i + n])
                if phrase in lex_expanded:
                    count += 1

        return int(count), max(1, int(len(tokens)))
    
    def extract_lemmas_for_document(self, doc: dict) -> List[str]:
        """Extract lemmas from preprocessed document"""
        if isinstance(doc.get('lemmas'), list):
            return doc['lemmas']
        return []
    
    def count_category_matches(self, lemmas: List[str], lexicon: set) -> int:
        """Count matches between lemmas and lexicon (single-token fallback).

        This is kept for backwards compatibility but is no longer the primary scoring path,
        because it cannot match multi-word lexicon phrases.
        """

        lemmas_lower = [str(l).lower() for l in (lemmas or [])]
        lex_lower = {self._normalize_term(t) for t in lexicon}
        return sum(1 for lemma in lemmas_lower if lemma in lex_lower)
    
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
        
        for _, row in df.iterrows():
            cleaned = row.get('cleaned') if isinstance(row, dict) else row.get('cleaned', None)
            if cleaned is None or str(cleaned).strip() == "":
                # Fallback to lemma-join for older fixtures
                lemmas = self.extract_lemmas_for_document(row)
                cleaned = " ".join(str(x) for x in (lemmas or []))

            eco_count, token_total = self._count_matches_text(
                cleaned_text=str(cleaned),
                lexicon=self.economic_lexicon,
                matcher=self._econ_matcher,
            )
            sec_count, _ = self._count_matches_text(
                cleaned_text=str(cleaned),
                lexicon=self.security_lexicon,
                matcher=self._sec_matcher,
            )

            # Normalize by token count from the same text used for matching
            economic_scores.append(float(eco_count) / float(token_total))
            security_scores.append(float(sec_count) / float(token_total))
        
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
        """Identify a more robust crossover point.

        Previous behavior (first year with security_mean > economic_mean) is fragile.
        Current behavior:
        - Excludes low-volume years (requires >=3 docs for both series where available)
        - Uses a 3-year rolling mean
        - Requires the crossover to persist for at least 2 consecutive years
        """

        if yearly_df is None or len(yearly_df) == 0:
            return None

        dfy = yearly_df.copy().sort_values("year")
        # counts are produced by aggregate_by_year
        econ_n = pd.to_numeric(dfy.get("economic_score_count"), errors="coerce")
        sec_n = pd.to_numeric(dfy.get("security_score_count"), errors="coerce")

        if econ_n is not None and sec_n is not None:
            eligible = (econ_n.fillna(0) >= 3) & (sec_n.fillna(0) >= 3)
            dfy = dfy[eligible].copy()

        if len(dfy) < 3:
            return None

        econ = pd.to_numeric(dfy["economic_score_mean"], errors="coerce").fillna(0.0)
        sec = pd.to_numeric(dfy["security_score_mean"], errors="coerce").fillna(0.0)

        econ_roll = econ.rolling(window=3, min_periods=2).mean()
        sec_roll = sec.rolling(window=3, min_periods=2).mean()

        # Find first year where sec>econ for 2 consecutive points
        cond = (sec_roll > econ_roll).fillna(False).tolist()
        years = dfy["year"].astype(int).tolist()
        for i in range(0, len(cond) - 1):
            if cond[i] and cond[i + 1]:
                return int(years[i])

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
    
    def audit_lexicons(self) -> Dict:
        """Audit lexicon quality (overlap/ambiguity) using analyzer normalization.

        Returns a dict suitable for embedding in reports/dashboards.
        """

        econ_norm = {self._normalize_term(t) for t in (self.economic_lexicon or set()) if str(t).strip()}
        sec_norm = {self._normalize_term(t) for t in (self.security_lexicon or set()) if str(t).strip()}
        overlap = sorted(list(econ_norm & sec_norm))

        def _multiword(terms: set[str]) -> int:
            return int(sum(1 for t in terms if " " in str(t)))

        report: Dict = {
            "economic_terms": int(len(econ_norm)),
            "security_terms": int(len(sec_norm)),
            "overlap_count": int(len(overlap)),
            "overlap_terms_sample": overlap[:50],
            "economic_multiword": _multiword(econ_norm),
            "security_multiword": _multiword(sec_norm),
        }

        # Simple flags for manual review
        if overlap:
            report["warnings"] = [
                "Lexicon overlap detected: terms appear in both economic and security lexicons after normalization. Review for ambiguity.",
            ]
        else:
            report["warnings"] = []

        return report

    def generate_shift_report(
        self,
        df: pd.DataFrame,
        group_by_source: bool = False,
        include_lexicon_audit: bool = True,
        **_ignored_kwargs,
    ) -> Tuple[Dict, pd.DataFrame, pd.DataFrame]:
        """Build the shift report with scores, yearly aggregation, and stat tests.

        Args:
            df: preprocessed dataframe
            group_by_source: when True, also compute per-source reports (e.g., MEA vs MOFA)
        """
        df_scored = self.calculate_category_scores(df)
        yearly_df = self.aggregate_by_year(df_scored)
        crossover = self.identify_crossover_point(yearly_df)
        stats_result = self.calculate_statistical_significance(df_scored)
        trend_result = self.calculate_trend(yearly_df)

        # Year-over-year deltas and acceleration (based on yearly means)
        yoy = []
        try:
            ydf = yearly_df.copy().sort_values('year')
            econ = pd.to_numeric(ydf['economic_score_mean'], errors='coerce').fillna(0.0).tolist()
            sec = pd.to_numeric(ydf['security_score_mean'], errors='coerce').fillna(0.0).tolist()
            years = ydf['year'].astype(int).tolist()
            for i in range(1, len(years)):
                de = float(econ[i] - econ[i - 1])
                ds = float(sec[i] - sec[i - 1])
                yoy.append({'year': int(years[i]), 'delta_economic': de, 'delta_security': ds})

            # acceleration = delta change from previous delta
            for i in range(1, len(yoy)):
                yoy[i]['accel_economic'] = float(yoy[i]['delta_economic'] - yoy[i - 1]['delta_economic'])
                yoy[i]['accel_security'] = float(yoy[i]['delta_security'] - yoy[i - 1]['delta_security'])
            if yoy:
                yoy[0]['accel_economic'] = 0.0
                yoy[0]['accel_security'] = 0.0
        except Exception:
            yoy = []
        
        report: Dict = {
            'total_documents': len(df),
            'date_range': (str(df['date'].min()), str(df['date'].max())),
            'overall_economic_avg': float(df_scored['economic_score'].mean()),
            'overall_security_avg': float(df_scored['security_score'].mean()),
            'crossover_year': crossover,
            'trend': 'SECURITY' if df_scored['security_score'].mean() > df_scored['economic_score'].mean() else 'ECONOMIC',
            'statistical_significance': stats_result,
            'trend_analysis': trend_result,
            'yearly_stats': yearly_df.to_dict('records'),
            'yearly_change': yoy,
        }

        if include_lexicon_audit:
            try:
                report["lexicon_audit"] = self.audit_lexicons()
            except Exception:
                report["lexicon_audit"] = {"error": "Lexicon audit failed"}

        if group_by_source and 'source' in df_scored.columns:
            by_source: Dict[str, Dict] = {}
            for source, sdf in df_scored.groupby(df_scored['source'].astype(str).str.upper().str.strip()):
                if sdf is None or len(sdf) < 3:
                    continue
                ydf = self.aggregate_by_year(sdf.copy())
                by_source[str(source)] = {
                    'total_documents': int(len(sdf)),
                    'date_range': (str(sdf['date'].min()), str(sdf['date'].max())) if 'date' in sdf.columns else (None, None),
                    'overall_economic_avg': float(pd.to_numeric(sdf['economic_score'], errors='coerce').fillna(0.0).mean()),
                    'overall_security_avg': float(pd.to_numeric(sdf['security_score'], errors='coerce').fillna(0.0).mean()),
                    'crossover_year': self.identify_crossover_point(ydf),
                    'trend': 'SECURITY' if float(pd.to_numeric(sdf['security_score'], errors='coerce').fillna(0.0).mean()) > float(pd.to_numeric(sdf['economic_score'], errors='coerce').fillna(0.0).mean()) else 'ECONOMIC',
                    'yearly_stats': ydf.to_dict('records'),
                }
            if by_source:
                report['by_source'] = by_source
        
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
        matcher = self._sec_matcher if category == 'security' else self._econ_matcher

        term_counts = Counter()

        # Prefer cleaned text so phrases can be counted.
        if df is None or len(df) == 0:
            return []

        for _, row in df.iterrows():
            cleaned = row.get('cleaned') if isinstance(row, dict) else row.get('cleaned', None)
            if cleaned is None or str(cleaned).strip() == "":
                lemmas = row.get('lemmas', [])
                if isinstance(lemmas, list):
                    cleaned = " ".join(str(x) for x in lemmas)
                else:
                    cleaned = str(lemmas or "")

            text = str(cleaned or "")
            if not text.strip():
                continue

            if self._spacy_nlp is not None and matcher is not None:
                doc = self._spacy_nlp(text)
                for _, start, end in matcher(doc):
                    span = doc[start:end].text
                    term_counts[self._normalize_term(span)] += 1
            else:
                # Fallback: count occurrences via n-grams
                tokens = self._simple_tokens(text)
                if not tokens:
                    continue
                lex_norm = {self._normalize_term(t) for t in lexicon}
                lex_expanded = set()
                for t in lex_norm:
                    lex_expanded.update(self._term_variants(t))

                for tok in tokens:
                    if tok in lex_expanded:
                        term_counts[tok] += 1
                for n in (2, 3, 4, 5):
                    for i in range(0, len(tokens) - n + 1):
                        phrase = " ".join(tokens[i : i + n])
                        if phrase in lex_expanded:
                            term_counts[phrase] += 1

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
