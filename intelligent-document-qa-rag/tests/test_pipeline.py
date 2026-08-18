"""
test_pipeline.py
-----------------
Basic sanity tests for the core RAG components. These use the
lightweight, dependency-free parts of the pipeline (chunking) plus
a small end-to-end smoke test of embeddings + retrieval, so they
can run reasonably fast without requiring a GPU.

Run with:
    pytest tests/
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.text_splitter import chunk_text


def test_chunk_text_basic():
    text = "This is sentence one. This is sentence two. This is sentence three."
    chunks = chunk_text(text, source="test.txt", chunk_size=40, chunk_overlap=10)
    assert len(chunks) > 0
    for chunk in chunks:
        assert chunk.source == "test.txt"
        assert len(chunk.text) > 0


def test_chunk_text_empty_string():
    chunks = chunk_text("", source="empty.txt")
    assert chunks == []


def test_chunk_overlap_validation():
    import pytest
    with pytest.raises(ValueError):
        chunk_text("some text", chunk_size=100, chunk_overlap=200)


def test_vector_store_search_empty():
    import numpy as np
    from src.vector_store import VectorStore

    store = VectorStore(embedding_dim=8)
    query = np.random.rand(8).astype("float32")
    results = store.search(query, top_k=3)
    assert results == []


def test_vector_store_add_and_search():
    import numpy as np
    from src.text_splitter import Chunk
    from src.vector_store import VectorStore

    store = VectorStore(embedding_dim=4)
    embeddings = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
        dtype="float32",
    )
    chunks = [
        Chunk(text=f"chunk {i}", source="test.txt", chunk_id=i, start_char=0, end_char=10)
        for i in range(3)
    ]
    store.add(embeddings, chunks)
    assert len(store) == 3

    query = np.array([1, 0, 0, 0], dtype="float32")
    results = store.search(query, top_k=1)
    assert len(results) == 1
    assert results[0][0].text == "chunk 0"
