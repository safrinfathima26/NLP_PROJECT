"""
generator.py
------------
Turns a user question + retrieved context chunks into a final
natural-language answer. Two backends are supported:

1. "local"  (default) - runs google/flan-t5-base fully offline via
   HuggingFace transformers. No API key required. Good for demos,
   coursework, and environments without internet access to LLM APIs.

2. "openai" - routes generation through the OpenAI Chat Completions
   API for higher-quality answers. Requires OPENAI_API_KEY to be
   set (e.g. in a .env file). Only used if you explicitly select it.

Both backends share the same prompt-construction logic, so retrieval
quality (not prompting) is the main variable when comparing them.
"""

from __future__ import annotations

import os
from typing import List, Tuple

from src.text_splitter import Chunk

PROMPT_TEMPLATE = """You are a helpful assistant answering questions using ONLY the context provided below.
If the answer cannot be found in the context, say "I could not find this in the provided document(s)."
Do not make up information that is not supported by the context.

Context:
{context}

Question: {question}

Answer:"""


def build_prompt(question: str, retrieved: List[Tuple[Chunk, float]]) -> str:
    """Assemble the RAG prompt from retrieved chunks."""
    context_blocks = []
    for i, (chunk, score) in enumerate(retrieved, start=1):
        context_blocks.append(f"[Source {i}: {chunk.source}]\n{chunk.text}")
    context = "\n\n".join(context_blocks) if context_blocks else "(no context retrieved)"
    return PROMPT_TEMPLATE.format(context=context, question=question)


class LocalGenerator:
    """Local, offline generator using a small HuggingFace seq2seq model."""

    def __init__(self, model_name: str = "google/flan-t5-base"):
        from transformers import pipeline  # imported lazily to keep startup fast

        self.model_name = model_name
        self._pipe = pipeline("text2text-generation", model=model_name, max_new_tokens=256)

    def generate(self, question: str, retrieved: List[Tuple[Chunk, float]]) -> str:
        prompt = build_prompt(question, retrieved)
        output = self._pipe(prompt, do_sample=False)[0]["generated_text"]
        return output.strip()


class OpenAIGenerator:
    """Generator backed by the OpenAI Chat Completions API."""

    def __init__(self, model_name: str = "gpt-4o-mini"):
        from openai import OpenAI  # imported lazily

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Add it to a .env file or your environment "
                "to use the 'openai' generator backend."
            )
        self.model_name = model_name
        self._client = OpenAI(api_key=api_key)

    def generate(self, question: str, retrieved: List[Tuple[Chunk, float]]) -> str:
        prompt = build_prompt(question, retrieved)
        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a precise, grounded document question-answering assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()


def get_generator(backend: str = "local", model_name: str | None = None):
    """
    Factory function to construct a generator by backend name.

    Parameters
    ----------
    backend : "local" | "openai"
    model_name : optional override of the default model for the chosen backend
    """
    if backend == "local":
        return LocalGenerator(model_name or "google/flan-t5-base")
    elif backend == "openai":
        return OpenAIGenerator(model_name or "gpt-4o-mini")
    else:
        raise ValueError(f"Unknown generator backend: {backend}. Use 'local' or 'openai'.")
