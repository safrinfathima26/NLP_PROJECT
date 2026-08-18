"""
vector_store.py
----------------
A thin, persistable wrapper around a FAISS index that stores chunk
embeddings alongside their original text/metadata, so retrieval
can return human-readable passages, not just vector IDs.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import List, Tuple, Union

import faiss
import numpy as np

from src.text_splitter import Chunk


class VectorStore:
    """FAISS-backed store of chunk embeddings with metadata."""

    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
        # Inner product on normalized vectors == cosine similarity
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.chunks: List[Chunk] = []

    def add(self, embeddings: np.ndarray, chunks: List[Chunk]) -> None:
        """Add a batch of embeddings and their corresponding chunk objects."""
        if embeddings.shape[0] != len(chunks):
            raise ValueError("Number of embeddings must match number of chunks")
        self.index.add(embeddings)
        self.chunks.extend(chunks)

    def search(self, query_embedding: np.ndarray, top_k: int = 4) -> List[Tuple[Chunk, float]]:
        """
        Search for the top_k most similar chunks to a query embedding.

        Returns
        -------
        list[(Chunk, similarity_score)]
        """
        if self.index.ntotal == 0:
            return []

        query_embedding = query_embedding.reshape(1, -1).astype("float32")
        top_k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self.chunks[idx], float(score)))
        return results

    def save(self, directory: Union[str, Path]) -> None:
        """Persist the FAISS index and chunk metadata to disk."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / "index.faiss"))
        with open(directory / "chunks.pkl", "wb") as f:
            pickle.dump({"chunks": self.chunks, "embedding_dim": self.embedding_dim}, f)

    @classmethod
    def load(cls, directory: Union[str, Path]) -> "VectorStore":
        """Load a previously saved FAISS index and chunk metadata."""
        directory = Path(directory)
        with open(directory / "chunks.pkl", "rb") as f:
            data = pickle.load(f)

        store = cls(embedding_dim=data["embedding_dim"])
        store.index = faiss.read_index(str(directory / "index.faiss"))
        store.chunks = data["chunks"]
        return store

    def __len__(self) -> int:
        return self.index.ntotal
