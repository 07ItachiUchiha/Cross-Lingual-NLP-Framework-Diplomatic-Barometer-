"""
Recursive Character Text Splitter for Diplomatic Documents
-----------------------------------------------------------
Cannot feed a 20-page PDF to an LLM — must chunk intelligently.

Strategy:
  - chunk_size  = 500 tokens  (fits in context window efficiently)
  - overlap     = 50 tokens   (preserves cross-chunk context)
  - separators  = paragraph > sentence > word  (diplomatic docs are paragraph-heavy)

Each chunk retains metadata: source doc ID, page, year, title.
"""

import re
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass, field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """A single chunk of text with full provenance metadata."""
    chunk_id: str
    text: str
    doc_id: str
    doc_title: str
    year: int
    source: str          # "MEA" or "MOFA"
    chunk_index: int     # position within document
    total_chunks: int    # how many chunks the doc produced
    start_char: int      # character offset in original
    end_char: int
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "doc_id": self.doc_id,
            "doc_title": self.doc_title,
            "year": self.year,
            "source": self.source,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "start_char": self.start_char,
            "end_char": self.end_char,
            **self.metadata,
        }


class DocumentChunker:
    """
    Recursive character text splitter tuned for diplomatic corpora.

    Parameters
    ----------
    chunk_size : int
        Maximum number of characters per chunk (default 1500 ≈ 500 tokens).
    chunk_overlap : int
        Overlap between consecutive chunks in characters (default 150 ≈ 50 tokens).
    separators : list[str]
        Ordered list of separators to try recursively.
    """

    DEFAULT_SEPARATORS = [
        "\n\n",   # paragraph boundary  (strongest signal)
        "\n",     # line break
        ". ",     # sentence boundary
        "; ",     # clause boundary
        ", ",     # phrase boundary
        " ",      # word boundary
    ]

    def __init__(
        self,
        chunk_size: int = 1500,
        chunk_overlap: int = 150,
        separators: Optional[List[str]] = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or self.DEFAULT_SEPARATORS

    # ── public API ───────────────────────────────────────────────────
    def chunk_document(
        self,
        text: str,
        doc_id: str,
        doc_title: str = "",
        year: int = 0,
        source: str = "",
        extra_metadata: Optional[Dict] = None,
    ) -> List[DocumentChunk]:
        """Split a single document into overlapping chunks with metadata."""
        if not text or not text.strip():
            return []

        raw_chunks = self._recursive_split(text, self.separators)
        merged = self._merge_with_overlap(raw_chunks)

        chunks: List[DocumentChunk] = []
        offset = 0
        for i, chunk_text in enumerate(merged):
            start = text.find(chunk_text, offset)
            if start == -1:
                start = offset
            end = start + len(chunk_text)
            offset = max(offset, start)

            chunk = DocumentChunk(
                chunk_id=f"{doc_id}_chunk_{i:03d}",
                text=chunk_text.strip(),
                doc_id=doc_id,
                doc_title=doc_title,
                year=year,
                source=source,
                chunk_index=i,
                total_chunks=len(merged),
                start_char=start,
                end_char=end,
                metadata=extra_metadata or {},
            )
            chunks.append(chunk)

        logger.debug(f"Chunked '{doc_title}' → {len(chunks)} chunks")
        return chunks

    def chunk_dataframe(self, df, text_col="content", id_col="title") -> List[DocumentChunk]:
        """Chunk every row in a DataFrame."""
        all_chunks: List[DocumentChunk] = []
        for idx, row in df.iterrows():
            doc_id = f"doc_{idx:04d}"
            chunks = self.chunk_document(
                text=str(row.get(text_col, "")),
                doc_id=doc_id,
                doc_title=str(row.get(id_col, "")),
                year=int(row["year"]) if "year" in row else 0,
                source=str(row.get("source", "")),
            )
            all_chunks.extend(chunks)
        logger.info(f"Chunked {len(df)} documents → {len(all_chunks)} chunks "
                     f"(avg {len(all_chunks)/max(len(df),1):.1f} chunks/doc)")
        return all_chunks

    # ── internal helpers ─────────────────────────────────────────────
    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        """Try each separator recursively until chunks fit under chunk_size."""
        if not separators:
            # Base case: character-level split
            return [text[i:i + self.chunk_size]
                    for i in range(0, len(text), self.chunk_size)]

        sep = separators[0]
        remaining_seps = separators[1:]

        parts = text.split(sep)
        good, current = [], ""

        for part in parts:
            candidate = (current + sep + part).strip() if current else part.strip()
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    good.append(current)
                # If the part alone exceeds chunk_size, recurse deeper
                if len(part) > self.chunk_size:
                    good.extend(self._recursive_split(part, remaining_seps))
                    current = ""
                else:
                    current = part.strip()

        if current:
            good.append(current)

        return good

    def _merge_with_overlap(self, chunks: List[str]) -> List[str]:
        """Add overlap between consecutive chunks."""
        if len(chunks) <= 1:
            return chunks

        merged = [chunks[0]]
        for i in range(1, len(chunks)):
            prev = chunks[i - 1]
            curr = chunks[i]
            # Prepend tail of previous chunk as overlap
            overlap_text = prev[-self.chunk_overlap:] if len(prev) > self.chunk_overlap else prev
            overlapped = (overlap_text + " " + curr).strip()
            # Trim if exceeds chunk_size
            if len(overlapped) > self.chunk_size:
                overlapped = overlapped[:self.chunk_size]
            merged.append(overlapped)

        return merged

    # ── stats ────────────────────────────────────────────────────────
    def get_chunk_stats(self, chunks: List[DocumentChunk]) -> Dict:
        lengths = [len(c.text) for c in chunks]
        return {
            "total_chunks": len(chunks),
            "avg_length": sum(lengths) / max(len(lengths), 1),
            "min_length": min(lengths) if lengths else 0,
            "max_length": max(lengths) if lengths else 0,
            "unique_docs": len(set(c.doc_id for c in chunks)),
        }


# ── CLI ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample = (
        "India and Japan signed the CEPA agreement covering goods and services. "
        "Bilateral trade exceeded 20 billion dollars.\n\n"
        "The Quad Leaders Summit discussed maritime security and counter-terrorism. "
        "Joint naval exercises were expanded to include all four nations.\n\n"
        "Cultural exchanges included student scholarships and a film festival."
    )
    chunker = DocumentChunker(chunk_size=200, chunk_overlap=30)
    chunks = chunker.chunk_document(sample, doc_id="test_001", doc_title="Test Doc", year=2023, source="MEA")
    for c in chunks:
        print(f"[{c.chunk_id}] ({len(c.text)} chars): {c.text[:80]}...")
