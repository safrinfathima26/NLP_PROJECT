"""
rag_pipeline.py
----------------
Top-level orchestrator that wires together document loading,
chunking, embedding, vector storage, retrieval, and generation
into a single easy-to-use RAG pipeline.

Typical usage
-------------
    from src.rag_pipeline import RAGPipeline

    rag = RAGPipeline(generator_backend="local")
    rag.ingest_directory("data/sample_docs")
    result = rag.ask("What is the main topic of the document?")
    print(result["answer"])
    print(result["sources"])
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

from src.document_loader import load_document, load_documents_from_directory
from src.embeddings import EmbeddingModel
from src.generator import get_generator
from src.text_splitter import chunk_documents, chunk_text
from src.vector_store import VectorStore


class RAGPipeline:
    """
    End-to-end Retrieval-Augmented Generation pipeline for
    question answering over user-supplied documents.
    """

    def __init__(
        self,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        generator_backend: str = "local",
        generator_model_name: Optional[str] = None,
        chunk_size: int = 800,
        chunk_overlap: int = 150,
        top_k: int = 4,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k

        self.embedder = EmbeddingModel(embedding_model_name)
        self.store = VectorStore(embedding_dim=self.embedder.embedding_dim)
        self.generator = get_generator(generator_backend, generator_model_name)

    # ------------------------------------------------------------------ #
    # Ingestion
    # ------------------------------------------------------------------ #
    def ingest_text(self, text: str, source: str = "document") -> int:
        """Ingest a raw text string. Returns number of chunks added."""
        chunks = chunk_text(text, source=source, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        if not chunks:
            return 0
        embeddings = self.embedder.embed_texts([c.text for c in chunks])
        self.store.add(embeddings, chunks)
        return len(chunks)

    def ingest_file(self, file_path: Union[str, Path]) -> int:
        """Ingest a single .pdf, .docx, or .txt file. Returns number of chunks added."""
        text = load_document(file_path)
        return self.ingest_text(text, source=Path(file_path).name)

    def ingest_directory(self, directory: Union[str, Path]) -> int:
        """Ingest every supported document in a directory. Returns total chunks added."""
        documents = load_documents_from_directory(directory)
        chunks = chunk_documents(documents, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        if not chunks:
            return 0
        embeddings = self.embedder.embed_texts([c.text for c in chunks], show_progress=True)
        self.store.add(embeddings, chunks)
        return len(chunks)

    # ------------------------------------------------------------------ #
    # Querying
    # ------------------------------------------------------------------ #
    def ask(self, question: str, top_k: Optional[int] = None) -> dict:
        """
        Answer a question using the ingested documents.

        Returns
        -------
        dict with keys:
            "answer"   : the generated answer string
            "sources"  : list of {"source", "text", "score"} for each
                         retrieved chunk used as context
        """
        if len(self.store) == 0:
            return {
                "answer": "No documents have been ingested yet. Please add documents first.",
                "sources": [],
            }

        k = top_k or self.top_k
        query_embedding = self.embedder.embed_query(question)
        retrieved = self.store.search(query_embedding, top_k=k)

        answer = self.generator.generate(question, retrieved)

        sources = [
            {"source": chunk.source, "text": chunk.text, "score": round(score, 4)}
            for chunk, score in retrieved
        ]
        return {"answer": answer, "sources": sources}

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save_index(self, directory: Union[str, Path]) -> None:
        self.store.save(directory)

    def load_index(self, directory: Union[str, Path]) -> None:
        self.store = VectorStore.load(directory)
