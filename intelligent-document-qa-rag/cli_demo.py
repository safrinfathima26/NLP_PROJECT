"""
cli_demo.py
-----------
Simple command-line demo of the RAG pipeline, useful for quick
testing or for running the project in environments without a
browser (e.g. during grading/evaluation).

Usage
-----
    python cli_demo.py --docs data/sample_docs --backend local

Then type questions interactively. Type 'exit' to quit.
"""

import argparse

from src.rag_pipeline import RAGPipeline


def main():
    parser = argparse.ArgumentParser(description="Intelligent Document QA (NLP + RAG) - CLI demo")
    parser.add_argument("--docs", type=str, default="data/sample_docs", help="Directory of documents to ingest")
    parser.add_argument("--backend", type=str, default="local", choices=["local", "openai"], help="Generator backend")
    parser.add_argument("--top-k", type=int, default=4, help="Number of chunks to retrieve per question")
    args = parser.parse_args()

    print(f"\nLoading RAG pipeline (backend={args.backend})... this may take a minute on first run.\n")
    rag = RAGPipeline(generator_backend=args.backend, top_k=args.top_k)

    print(f"Ingesting documents from: {args.docs}")
    num_chunks = rag.ingest_directory(args.docs)
    print(f"Done. {num_chunks} chunks indexed.\n")

    print("Ask questions about the ingested documents. Type 'exit' to quit.\n")
    while True:
        question = input("Q: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue

        result = rag.ask(question)
        print(f"\nA: {result['answer']}\n")
        print("Sources:")
        for i, src in enumerate(result["sources"], start=1):
            print(f"  [{i}] {src['source']} (score={src['score']})")
        print()


if __name__ == "__main__":
    main()
