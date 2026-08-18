# Project Report

## Intelligent Document Question Answering using NLP and RAG

---

## Abstract

Manually searching long documents to find specific information is slow
and error-prone. This project presents an **Intelligent Document
Question Answering system** that allows users to upload documents and
ask natural-language questions about their content, receiving accurate,
source-grounded answers. The system is built using **Retrieval-Augmented
Generation (RAG)**, a technique that combines dense semantic retrieval
with large language model (LLM) text generation. Documents are parsed,
split into overlapping chunks, embedded using a Sentence-BERT model, and
indexed in a FAISS vector store. At query time, the most semantically
relevant chunks are retrieved and passed as grounded context to a
language model, which generates the final answer along with citations
to the source passages. This approach significantly reduces
hallucination compared to using an LLM alone, and allows the system to
answer questions about private or newly-added documents that were never
part of any model's training data.

---

## 1. Introduction

### 1.1 Motivation

Organizations and individuals accumulate large volumes of unstructured
text — reports, contracts, research papers, manuals, and internal
documentation. Finding a specific answer within these documents
traditionally requires manual reading or keyword-based search, which
often fails when the exact wording of the answer differs from the
wording of the question. Large Language Models (LLMs) can answer
questions fluently, but on their own they suffer from two key
limitations: they cannot access information outside their training
data, and they can "hallucinate" — producing plausible-sounding but
incorrect answers.

### 1.2 Problem Statement

Build a system that allows a user to upload arbitrary documents and ask
natural-language questions about their content, returning answers that
are accurate, grounded in the source material, and traceable back to
the exact passages used to generate them.

### 1.3 Objectives

- Parse and process multiple document formats (PDF, DOCX, TXT)
- Represent document content in a way that supports semantic (meaning-based) search, not just keyword matching
- Retrieve the most relevant passages for a given question
- Generate a natural-language answer grounded in retrieved evidence
- Present the answer along with the supporting source passages
- Provide an accessible interface (web UI) for end users

---

## 2. Literature Survey / Background

### 2.1 Natural Language Processing and Transformers

Modern NLP is built primarily on the **transformer architecture**
(Vaswani et al., 2017), which uses self-attention to model relationships
between all words in a sequence simultaneously. Transformer-based models
such as BERT and its derivatives significantly improved performance on
tasks such as semantic similarity, classification, and question
answering compared to earlier RNN/LSTM-based approaches.

### 2.2 Semantic Search with Sentence Embeddings

**Sentence-BERT** (Reimers & Gurevych, 2019) fine-tunes BERT-style models
to produce sentence-level embeddings such that semantically similar
sentences are close together in vector space, measured via cosine
similarity. This enables **semantic search**: retrieving text based on
meaning rather than exact keyword overlap, which is essential for
natural-language question answering.

### 2.3 Retrieval-Augmented Generation (RAG)

RAG was introduced by Lewis et al. (2020) as a way to combine
parametric knowledge (stored in a language model's weights) with
non-parametric knowledge (retrieved on demand from an external corpus).
Instead of relying solely on what an LLM memorized during training, a
RAG system retrieves relevant documents at inference time and
conditions generation on that retrieved evidence. This has become the
dominant architecture for building question-answering systems over
private or dynamic document collections, because it:

- Reduces hallucination by grounding answers in retrieved text
- Allows knowledge to be updated simply by updating the document index, without retraining the model
- Provides traceability — answers can be linked back to source passages

### 2.4 Vector Databases

Efficient retrieval over large embedding collections requires
approximate nearest-neighbor search structures. **FAISS** (Facebook AI
Similarity Search) is a widely used library for this purpose, offering
exact and approximate similarity search algorithms that scale to
millions of vectors.

---

## 3. System Architecture

The system follows a standard four-stage RAG pipeline:

1. **Ingestion** — Documents (PDF/DOCX/TXT) are parsed into raw text.
2. **Chunking** — Text is split into overlapping, sentence-aware chunks
   (default: 800 characters with 150-character overlap) to fit within
   embedding and generation model context limits while preserving
   local context.
3. **Embedding & Indexing** — Each chunk is converted into a
   384-dimensional dense vector using the `all-MiniLM-L6-v2`
   Sentence-BERT model, and stored in a FAISS `IndexFlatIP` index
   (inner product on L2-normalized vectors, equivalent to cosine
   similarity).
4. **Retrieval & Generation** — At query time, the user's question is
   embedded using the same model, and the top-*k* most similar chunks
   are retrieved. These chunks are combined with the question into a
   structured prompt and passed to a generation model (a local
   `flan-t5-base` model by default, or optionally OpenAI's API) that
   produces the final answer, explicitly instructed to only use the
   provided context.

### 3.1 Module Breakdown

| Module | Responsibility |
|---|---|
| `document_loader.py` | Extracts raw text from PDF, DOCX, TXT files |
| `text_splitter.py` | Splits text into overlapping, sentence-aware chunks |
| `embeddings.py` | Wraps Sentence-BERT to produce normalized embeddings |
| `vector_store.py` | FAISS index wrapper; supports add, search, save, load |
| `generator.py` | Builds the grounded prompt and generates the final answer |
| `rag_pipeline.py` | Orchestrates the full pipeline end-to-end |
| `app.py` | Streamlit-based user interface |

---

## 4. Methodology

### 4.1 Document Chunking Strategy

A naive fixed-length character split can cut sentences in half,
harming both embedding quality and answer coherence. This project uses
a **sentence-aware sliding window**: sentences are grouped into chunks
up to a target size, and each new chunk begins with an overlapping tail
of the previous chunk's text. This overlap preserves context across
chunk boundaries — an important detail if the answer to a question
spans a boundary between two chunks.

### 4.2 Embedding Model Choice

`all-MiniLM-L6-v2` was selected as the default embedding model because
it offers a strong balance between semantic search accuracy and
inference speed on CPU-only hardware, producing compact 384-dimensional
embeddings without requiring a GPU — an important practicality for a
project intended to run in typical academic/development environments.

### 4.3 Retrieval

Cosine similarity search is used to identify the top-*k* chunks most
relevant to a given question. Using inner product search over
L2-normalized vectors, FAISS's `IndexFlatIP` performs this efficiently
via exact search, which is appropriate for small-to-medium document
collections typical of this kind of application.

### 4.4 Grounded Generation

The retrieved chunks are inserted into a structured prompt that
explicitly instructs the language model to answer *only* using the
provided context and to state clearly when the answer cannot be found,
rather than fabricating information. This prompt-level grounding
constraint is a key mechanism by which RAG systems reduce hallucination
relative to standalone LLM usage.

### 4.5 Dual Generation Backend Design

To keep the project runnable without any paid API access, a local,
fully offline generation backend (`flan-t5-base`, via HuggingFace
Transformers) is used by default. An optional OpenAI-backed generator
is also implemented for users who want higher-quality generation and
have API access, demonstrating that the retrieval and generation
components are cleanly decoupled.

---

## 5. Implementation Details

- **Language:** Python 3.10+
- **Embedding model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Vector index:** FAISS `IndexFlatIP`
- **Local generation model:** `google/flan-t5-base` (HuggingFace Transformers)
- **Optional cloud generation:** OpenAI Chat Completions API (`gpt-4o-mini`)
- **Document parsing:** `pypdf` for PDFs, `python-docx` for Word documents
- **User interface:** Streamlit (chat-style interface with file upload and source display)
- **Testing:** `pytest`, covering chunking edge cases and vector store operations

---

## 6. Results and Observations

Using the included sample document (`data/sample_docs/sample_nlp_overview.txt`,
an overview of NLP and RAG concepts), the system was able to correctly
answer direct factual questions such as:

- *"What is Retrieval-Augmented Generation?"*
- *"What are the four stages of a RAG pipeline?"*
- *"What metrics are used to evaluate RAG systems?"*

In each case, the retrieved source chunks correctly matched the
relevant section of the document, and the generated answer stayed
grounded in that content. When asked a question with no answer present
in the ingested document, the system correctly responded that the
information could not be found in the provided document(s), rather
than fabricating an answer — demonstrating the grounding constraint
working as intended.

### 6.1 Qualitative Observations

- Chunk size and overlap noticeably affect retrieval quality: chunks
  that are too large dilute the embedding's semantic focus, while
  chunks that are too small lose surrounding context.
- The local `flan-t5-base` backend is sufficient for concise factual
  questions but produces shorter, less elaborated answers than the
  optional OpenAI backend, illustrating the trade-off between running
  fully offline and using a larger hosted model.

---

## 7. Conclusion

This project demonstrates a complete, working Intelligent Document
Question Answering system built on the Retrieval-Augmented Generation
paradigm. By combining semantic retrieval (Sentence-BERT + FAISS) with
grounded language generation, the system is able to answer
natural-language questions over arbitrary user-supplied documents while
remaining traceable to its source material and resistant to
hallucination on out-of-scope questions. The modular design — separate,
independently testable components for loading, chunking, embedding,
retrieval, and generation — makes the system straightforward to extend,
for example by swapping in a different vector database, adding
re-ranking, or supporting additional document formats.

---

## 8. Future Work

- **Re-ranking:** Add a cross-encoder re-ranking stage after initial
  retrieval to improve precision on ambiguous queries.
- **OCR support:** Handle scanned/image-based PDFs via OCR preprocessing.
- **Conversational memory:** Support multi-turn follow-up questions
  that reference earlier parts of the conversation.
- **Evaluation framework:** Add automated retrieval and answer-quality
  metrics (e.g. precision@k, faithfulness scoring) for systematic
  benchmarking.
- **Scalable vector storage:** Replace the in-memory FAISS index with a
  persistent, multi-user vector database for production deployment.

---

## 9. References

1. Vaswani, A. et al. (2017). *Attention Is All You Need.* NeurIPS.
2. Reimers, N. & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.* EMNLP.
3. Lewis, P. et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* NeurIPS.
4. Johnson, J., Douze, M., & Jégou, H. (2019). *Billion-scale similarity search with GPUs.* IEEE Transactions on Big Data. (FAISS)
5. Chung, H. W. et al. (2022). *Scaling Instruction-Finetuned Language Models.* (FLAN-T5)
