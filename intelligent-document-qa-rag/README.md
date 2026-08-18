# Intelligent Document Question Answering using NLP and RAG

A complete, working system that lets users upload documents (PDF, DOCX,
or TXT) and ask natural-language questions about their content. Answers
are generated using **Retrieval-Augmented Generation (RAG)** — relevant
passages are retrieved from the document using semantic search, and a
language model uses those passages as grounded context to produce an
accurate, source-cited answer.

---

## ✨ Features

- 📄 Upload and parse **PDF, DOCX, and TXT** documents
- ✂️ Sentence-aware **chunking** with configurable overlap
- 🧠 Semantic **embeddings** via Sentence-BERT (`all-MiniLM-L6-v2`)
- 🔍 Fast **vector similarity search** using FAISS
- 🤖 Grounded **answer generation** — local (offline, no API key) or via OpenAI
- 📎 **Source attribution** — every answer shows which document chunks it came from
- 🖥️ Interactive **Streamlit web app** and a **CLI demo**
- 🧪 Unit tests for the core pipeline

---

## 🏗️ Architecture

```
                    ┌─────────────────┐
   User uploads ──▶ │ Document Loader │  (PDF / DOCX / TXT)
                    └────────┬────────┘
                             ▼
                    ┌─────────────────┐
                    │  Text Splitter   │  sentence-aware chunking
                    └────────┬────────┘
                             ▼
                    ┌─────────────────┐
                    │ Embedding Model  │  Sentence-BERT (MiniLM)
                    └────────┬────────┘
                             ▼
                    ┌─────────────────┐
                    │  FAISS Vector    │  stores chunk embeddings
                    │      Store       │
                    └────────┬────────┘
                             ▲
                             │  top-k similarity search
   User question ──▶ embed ─┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  Retrieved       │
                    │  Context Chunks  │
                    └────────┬────────┘
                             ▼
                    ┌─────────────────┐
                    │  LLM Generator   │  local flan-t5 or OpenAI
                    │  (grounded       │
                    │   prompt)        │
                    └────────┬────────┘
                             ▼
                     Final answer + sources
```

This is the standard **RAG (Retrieval-Augmented Generation)** pattern:
retrieve relevant evidence first, then generate an answer conditioned on
that evidence — rather than relying purely on what a language model
memorized during training.

---

## 📁 Project Structure

```
intelligent-document-qa-rag/
├── app.py                     # Streamlit web application
├── cli_demo.py                 # Command-line demo
├── requirements.txt
├── .env.example                 # Template for optional OpenAI API key
├── src/
│   ├── document_loader.py      # PDF / DOCX / TXT parsing
│   ├── text_splitter.py        # Chunking logic
│   ├── embeddings.py           # Sentence-BERT embedding wrapper
│   ├── vector_store.py         # FAISS index wrapper (save/load)
│   ├── generator.py            # Local + OpenAI answer generation
│   └── rag_pipeline.py         # End-to-end pipeline orchestrator
├── data/sample_docs/            # Sample document for quick testing
├── tests/test_pipeline.py       # Unit tests
└── docs/PROJECT_REPORT.md       # Full academic-style project report
```

---

## 🚀 Setup

### 1. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** The first run will download the embedding model
> (`all-MiniLM-L6-v2`, ~80MB) and, if using the local generation
> backend, the `flan-t5-base` model (~250MB). These are cached
> locally afterward.

### 3. (Optional) Configure OpenAI backend

Only needed if you want higher-quality generation via OpenAI instead of
the default local model:

```bash
cp .env.example .env
# then edit .env and add your OPENAI_API_KEY
```

---

## ▶️ Running the Project

### Option A — Web app (recommended)

```bash
streamlit run app.py
```

Then open the URL shown in the terminal (usually `http://localhost:8501`),
upload a document from the sidebar, click **Ingest documents**, and start
asking questions in the chat box.

### Option B — Command line demo

```bash
python cli_demo.py --docs data/sample_docs --backend local
```

This ingests the sample document included in `data/sample_docs/` and
lets you ask questions interactively in the terminal.

### Option C — Use it as a Python library

```python
from src.rag_pipeline import RAGPipeline

rag = RAGPipeline(generator_backend="local")
rag.ingest_directory("data/sample_docs")

result = rag.ask("What is Retrieval-Augmented Generation?")
print(result["answer"])
for src in result["sources"]:
    print(src["source"], src["score"])
```

---

## 🧪 Running Tests

```bash
pytest tests/
```

---

## ⚙️ Configuration Options

All key parameters can be adjusted when constructing `RAGPipeline`:

| Parameter | Default | Description |
|---|---|---|
| `embedding_model_name` | `all-MiniLM-L6-v2` | Sentence-BERT model used for embeddings |
| `generator_backend` | `local` | `local` (flan-t5, offline) or `openai` (needs API key) |
| `chunk_size` | `800` | Max characters per chunk |
| `chunk_overlap` | `150` | Overlap between consecutive chunks |
| `top_k` | `4` | Number of chunks retrieved per question |

---

## 🔧 Possible Extensions

- Swap FAISS for a hosted vector DB (Pinecone, Weaviate, Chroma) for multi-user persistence
- Add re-ranking (e.g. a cross-encoder) after initial retrieval for higher precision
- Add OCR support for scanned PDFs
- Add conversational memory for multi-turn follow-up questions
- Add evaluation metrics (retrieval precision/recall, answer faithfulness)

---

## 📄 License

MIT License — free to use and modify for academic or personal projects.

---

## 📚 Further Reading

See `docs/PROJECT_REPORT.md` for the full write-up: abstract, literature
review, methodology, implementation details, results, and conclusion —
suitable for academic submission.
