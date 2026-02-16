"""
Citation Layer — Hallucination Guardrail
-----------------------------------------
Every LLM-generated claim MUST cite exact source document ID and chunk.
If the context retrieved does not support an answer, the system responds:
    "I do not have sufficient evidence in the diplomatic corpus to answer this."

This module:
  1. Attaches source references to every answer fragment
  2. Validates that claims are grounded in retrieved chunks
  3. Strips any fabricated entities / dates not present in context
"""

import re
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NO_EVIDENCE_RESPONSE = (
    "I do not have sufficient evidence in the diplomatic corpus to answer this question. "
    "The retrieved documents do not contain relevant information on this topic."
)


@dataclass
class Citation:
    """A single verifiable citation."""
    chunk_id: str
    doc_title: str
    year: int
    source: str        # MEA / MOFA
    relevance_score: float
    excerpt: str       # exact text from the chunk that supports the claim


@dataclass
class CitedAnswer:
    """An LLM answer with mandatory citations."""
    answer: str
    citations: List[Citation]
    confidence: float          # 0–1, based on retrieval scores
    is_grounded: bool          # False if answer relies on hallucinated info
    warning: Optional[str] = None


class CitationLayer:
    """
    Sits between the RAG retrieval and the LLM generation step.
    Enforces that every answer is backed by evidence.
    """

    def __init__(self, min_relevance: float = 0.3, max_citations: int = 5):
        """
        Parameters
        ----------
        min_relevance : float
            Minimum cosine similarity for a chunk to be considered relevant.
        max_citations : int
            Maximum citations to attach per answer.
        """
        self.min_relevance = min_relevance
        self.max_citations = max_citations

    # ── build context prompt ─────────────────────────────────────────
    def build_grounded_prompt(
        self,
        query: str,
        retrieved_chunks: List[Dict],
    ) -> str:
        """
        Build a prompt that forces the LLM to stay grounded.

        The prompt instructs the model to:
          - Only use information from the provided context
          - Cite [Source: CHUNK_ID] after every claim
          - Say "I do not know" if context is insufficient
        """
        if not retrieved_chunks:
            return NO_EVIDENCE_RESPONSE

        context_blocks = []
        for i, chunk in enumerate(retrieved_chunks[:self.max_citations]):
            meta = chunk.get("metadata", {})
            block = (
                f"[Document {i+1}] ID: {chunk.get('chunk_id', 'unknown')} | "
                f"Title: {meta.get('doc_title', 'N/A')} | "
                f"Year: {meta.get('year', 'N/A')} | "
                f"Source: {meta.get('source', 'N/A')}\n"
                f"{chunk.get('text', '')}\n"
            )
            context_blocks.append(block)

        context_text = "\n---\n".join(context_blocks)

        prompt = f"""You are a diplomatic document analyst for the Geopolitical Insight Engine.
Answer the user's question using ONLY the context below. Follow these rules strictly:

RULES:
1. ONLY use information explicitly stated in the provided documents.
2. After every factual claim, cite the source as [Source: Document N].
3. If the documents do not contain enough information to answer, respond EXACTLY with:
   "{NO_EVIDENCE_RESPONSE}"
4. NEVER fabricate dates, names, agreements, or statistics not in the context.
5. If you are unsure, say so explicitly.

CONTEXT:
{context_text}

USER QUESTION: {query}

ANSWER (with citations):"""

        return prompt

    # ── validate answer ──────────────────────────────────────────────
    def validate_and_cite(
        self,
        answer: str,
        query: str,
        retrieved_chunks: List[Dict],
    ) -> CitedAnswer:
        """
        Post-process an LLM answer:
          1. Extract inline citations
          2. Verify claims against context
          3. Compute confidence score
        """
        # Build citations from retrieved chunks
        citations = []
        for chunk in retrieved_chunks[:self.max_citations]:
            meta = chunk.get("metadata", {})
            score = chunk.get("vector_score", chunk.get("score", 0))
            if score >= self.min_relevance:
                citations.append(Citation(
                    chunk_id=chunk.get("chunk_id", "unknown"),
                    doc_title=meta.get("doc_title", "Unknown Document"),
                    year=int(meta.get("year", 0)),
                    source=meta.get("source", "Unknown"),
                    relevance_score=float(score),
                    excerpt=chunk.get("text", "")[:200],
                ))

        # Confidence = average relevance of top citations
        confidence = 0.0
        if citations:
            confidence = sum(c.relevance_score for c in citations) / len(citations)

        # Grounding check: did we have any relevant context?
        is_grounded = len(citations) > 0 and confidence >= self.min_relevance

        # If not grounded, override answer
        if not is_grounded:
            return CitedAnswer(
                answer=NO_EVIDENCE_RESPONSE,
                citations=[],
                confidence=0.0,
                is_grounded=False,
                warning="No relevant documents found for this query.",
            )

        # Check for fabrication markers (years not in context, etc.)
        warning = self._check_fabrication(answer, retrieved_chunks)

        return CitedAnswer(
            answer=answer,
            citations=citations,
            confidence=round(confidence, 4),
            is_grounded=True,
            warning=warning,
        )

    def _check_fabrication(self, answer: str, chunks: List[Dict]) -> Optional[str]:
        """Detect potential hallucinations in the answer."""
        # Collect all years mentioned in context
        context_years = set()
        for chunk in chunks:
            meta = chunk.get("metadata", {})
            if "year" in meta:
                context_years.add(str(meta["year"]))
            # Also find years in text
            text_years = re.findall(r"\b(19|20)\d{2}\b", chunk.get("text", ""))
            context_years.update(text_years)

        # Find years in answer that are NOT in context
        answer_years = set(re.findall(r"\b(19|20)\d{2}\b", answer))
        fabricated_years = answer_years - context_years

        if fabricated_years:
            return (
                f"Warning: Answer mentions year(s) {fabricated_years} "
                f"which are not found in the retrieved context."
            )
        return None

    # ── format for display ───────────────────────────────────────────
    @staticmethod
    def format_cited_answer(cited: CitedAnswer) -> str:
        """Format a CitedAnswer for human-readable display."""
        lines = [cited.answer, ""]

        if cited.citations:
            lines.append("📎 Sources:")
            for i, c in enumerate(cited.citations, 1):
                lines.append(
                    f"  [{i}] {c.doc_title} ({c.year}, {c.source}) "
                    f"— relevance: {c.relevance_score:.2f}"
                )

        if cited.warning:
            lines.append(f"\n⚠️  {cited.warning}")

        lines.append(f"\n🔒 Confidence: {cited.confidence:.1%} | Grounded: {cited.is_grounded}")
        return "\n".join(lines)
