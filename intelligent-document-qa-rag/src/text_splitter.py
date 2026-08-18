"""
text_splitter.py
-----------------
Splits long document text into overlapping chunks that are small
enough to embed and retrieve effectively.

Why chunking matters for RAG:
    Embedding models and LLM context windows both have limits.
    Splitting into overlapping, semantically-coherent chunks lets
    the retriever return focused, relevant passages instead of
    whole documents, which improves both retrieval precision and
    generation quality.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List


@dataclass
class Chunk:
    """A single chunk of text along with metadata about its origin."""
    text: str
    source: str
    chunk_id: int
    start_char: int
    end_char: int
    metadata: dict = field(default_factory=dict)


def _split_into_sentences(text: str) -> List[str]:
    """Lightweight sentence splitter (avoids heavy NLTK downloads at import time)."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    # Split on sentence-ending punctuation followed by whitespace + capital letter/number
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_text(
    text: str,
    source: str = "document",
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> List[Chunk]:
    """
    Split text into overlapping chunks using a sentence-aware sliding window.

    Parameters
    ----------
    text : str
        Full document text.
    source : str
        Identifier of the originating document (e.g. filename).
    chunk_size : int
        Target maximum number of characters per chunk.
    chunk_overlap : int
        Number of characters to overlap between consecutive chunks,
        which preserves context across chunk boundaries.

    Returns
    -------
    list[Chunk]
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    sentences = _split_into_sentences(text)
    chunks: List[Chunk] = []

    current_chunk = ""
    current_start = 0
    char_cursor = 0
    chunk_id = 0

    for sentence in sentences:
        # If adding this sentence would exceed the target size, close the chunk out
        if current_chunk and len(current_chunk) + len(sentence) + 1 > chunk_size:
            chunks.append(
                Chunk(
                    text=current_chunk.strip(),
                    source=source,
                    chunk_id=chunk_id,
                    start_char=current_start,
                    end_char=current_start + len(current_chunk),
                )
            )
            chunk_id += 1

            # Start the next chunk with overlap from the tail of the previous one
            overlap_text = current_chunk[-chunk_overlap:] if chunk_overlap > 0 else ""
            current_start = current_start + len(current_chunk) - len(overlap_text)
            current_chunk = overlap_text + " " + sentence
        else:
            current_chunk = f"{current_chunk} {sentence}".strip()

        char_cursor += len(sentence) + 1

    if current_chunk.strip():
        chunks.append(
            Chunk(
                text=current_chunk.strip(),
                source=source,
                chunk_id=chunk_id,
                start_char=current_start,
                end_char=current_start + len(current_chunk),
            )
        )

    return chunks


def chunk_documents(documents: dict, chunk_size: int = 800, chunk_overlap: int = 150) -> List[Chunk]:
    """
    Chunk a batch of documents.

    Parameters
    ----------
    documents : dict[str, str]
        Mapping of {source_name: full_text}, as returned by
        document_loader.load_documents_from_directory().

    Returns
    -------
    list[Chunk]
        All chunks from all documents, concatenated.
    """
    all_chunks: List[Chunk] = []
    for source, text in documents.items():
        all_chunks.extend(chunk_text(text, source=source, chunk_size=chunk_size, chunk_overlap=chunk_overlap))
    return all_chunks
