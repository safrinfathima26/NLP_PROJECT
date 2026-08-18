"""
embeddings.py
-------------
Wraps a sentence-transformers model to turn text chunks into
dense vector embeddings used for semantic similarity search.

Default model: 'all-MiniLM-L6-v2'
    - 384-dimensional embeddings
    - Small (~80MB), fast on CPU
    - Strong accuracy/speed trade-off for semantic search,
      making it a common default for RAG projects and coursework.
"""

from __future__ import annotations

from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """Thin wrapper around SentenceTransformer with sensible defaults."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        self.embedding_dim = self._model.get_sentence_embedding_dimension()

    def embed_texts(self, texts: List[str], batch_size: int = 32, show_progress: bool = False) -> np.ndarray:
        """
        Embed a list of texts.

        Returns
        -------
        np.ndarray of shape (len(texts), embedding_dim), L2-normalized
        so that dot product similarity == cosine similarity.
        """
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embeddings.astype("float32")

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string. Returns a 1D array of shape (embedding_dim,)."""
        return self.embed_texts([query])[0]
