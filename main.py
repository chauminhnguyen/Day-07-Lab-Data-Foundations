from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.agent import KnowledgeBaseAgent
from src.chunking import FixedSizeChunker, RecursiveChunker, SentenceChunker
from src.embeddings import (
    EMBEDDING_PROVIDER_ENV,
    LOCAL_EMBEDDING_MODEL,
    OPENAI_EMBEDDING_MODEL,
    LocalEmbedder,
    OpenAIEmbedder,
    _mock_embed,
)
from src.models import Document
from src.store import EmbeddingStore

SAMPLE_FILES = [
    "data/alzheimer.md",
    "data/benh-dai.md",
    "data/benh_lao_phoi.md",
    "data/benh-san-day.md",
    "data/benh-tri.md",
]


def load_documents_from_files(file_paths: list[str]) -> list[Document]:
    """Load documents from file paths for the manual demo."""
    allowed_extensions = {".md", ".txt"}
    documents: list[Document] = []

    for raw_path in file_paths:
        path = Path(raw_path)

        if path.suffix.lower() not in allowed_extensions:
            print(f"Skipping unsupported file type: {path} (allowed: .md, .txt)")
            continue

        if not path.exists() or not path.is_file():
            print(f"Skipping missing file: {path}")
            continue

        content = path.read_text(encoding="utf-8")
        documents.append(
            Document(
                id=path.stem,
                content=content,
                metadata={"source": str(path), "extension": path.suffix.lower()},
            )
        )

    return documents


def demo_llm(prompt: str) -> str:
    """A simple mock LLM for manual RAG testing."""
    preview = prompt[:400].replace("\n", " ")
    return f"[DEMO LLM] Generated answer from prompt preview: {preview}..."


def create_chunker(
    chunker_name: str,
    chunk_size: int = 500,
    max_sentences_per_chunk: int = 3,
):
    """Build a chunker instance from a CLI option."""
    chunker_name = chunker_name.lower()
    if chunker_name == "fixed_size":
        return FixedSizeChunker(chunk_size=chunk_size)
    if chunker_name == "sentence":
        return SentenceChunker(max_sentences_per_chunk=max_sentences_per_chunk)
    if chunker_name == "recursive":
        return RecursiveChunker(chunk_size=chunk_size)
    raise ValueError(f"Unknown chunker: {chunker_name}")


def chunk_documents(docs: list[Document], chunker) -> list[Document]:
    """Split documents into smaller chunks before embedding."""
    chunked_docs: list[Document] = []
    for doc in docs:
        chunks = chunker.chunk(doc.content)
        for index, chunk in enumerate(chunks):
            metadata = dict(doc.metadata)
            metadata["chunk_index"] = index
            chunked_docs.append(
                Document(
                    id=f"{doc.id}_{index}",
                    content=chunk,
                    metadata=metadata,
                )
            )
    return chunked_docs


def run_manual_demo(
    question: str | None = None,
    sample_files: list[str] | None = None,
    chunker_name: str = "fixed_size",
    chunk_size: int = 500,
    sentences_per_chunk: int = 3,
) -> int:
    files = sample_files or SAMPLE_FILES
    query = question or "Summarize the key information from the loaded files."

    print("=== Manual File Test ===")
    print("Accepted file types: .md, .txt")
    print("Input file list:")
    for file_path in files:
        print(f"  - {file_path}")

    docs = load_documents_from_files(files)
    if not docs:
        print("\nNo valid input files were loaded.")
        print("Create files matching the sample paths above, then rerun:")
        print("  python3 main.py")
        return 1

    print(f"\nLoaded {len(docs)} documents")
    for doc in docs:
        print(f"  - {doc.id}: {doc.metadata['source']}")

    chunker = create_chunker(
        chunker_name=chunker_name,
        chunk_size=chunk_size,
        max_sentences_per_chunk=sentences_per_chunk,
    )
    docs = chunk_documents(docs, chunker)
    print(f"\nUsing chunker: {chunker_name}"
          f" (chunk_size={chunk_size}, sentences_per_chunk={sentences_per_chunk})")
    print(f"Chunked into {len(docs)} document chunks")

    load_dotenv(override=False)
    provider = os.getenv(EMBEDDING_PROVIDER_ENV, "mock").strip().lower()
    if provider == "local":
        try:
            embedder = LocalEmbedder(model_name=os.getenv("LOCAL_EMBEDDING_MODEL", LOCAL_EMBEDDING_MODEL))
        except Exception:
            embedder = _mock_embed
    elif provider == "openai":
        try:
            embedder = OpenAIEmbedder(model_name=os.getenv("OPENAI_EMBEDDING_MODEL", OPENAI_EMBEDDING_MODEL))
        except Exception:
            embedder = _mock_embed
    else:
        embedder = _mock_embed

    print(f"\nEmbedding backend: {getattr(embedder, '_backend_name', embedder.__class__.__name__)}")

    store = EmbeddingStore(collection_name="manual_test_store", embedding_fn=embedder)
    store.add_documents(docs)

    print(f"\nStored {store.get_collection_size()} documents in EmbeddingStore")
    print("\n=== EmbeddingStore Search Test ===")
    print(f"Query: {query}")
    search_results = store.search(query, top_k=3)
    for index, result in enumerate(search_results, start=1):
        print(f"{index}. score={result['score']:.3f} source={result['metadata'].get('source')}")
        print(f"   content preview: {result['content'][:120].replace(chr(10), ' ')}...")

    print("\n=== KnowledgeBaseAgent Test ===")
    agent = KnowledgeBaseAgent(store=store, llm_fn=demo_llm)
    print(f"Question: {query}")
    print("Agent answer:")
    print(agent.answer(query, top_k=3))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Manual RAG demo with chunker options")
    parser.add_argument(
        "--chunker",
        choices=["fixed_size", "sentence", "recursive"],
        default="fixed_size",
        help="Select the document chunking strategy to apply before embedding.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Maximum chunk size for fixed_size and recursive chunkers.",
    )
    parser.add_argument(
        "--sentences-per-chunk",
        type=int,
        default=3,
        help="Maximum sentences per chunk when using the sentence chunker.",
    )
    parser.add_argument("question", nargs="*", help="Optional question to ask the knowledge base.")
    args = parser.parse_args()

    question = " ".join(args.question).strip() if args.question else None
    return run_manual_demo(
        question=question,
        chunker_name=args.chunker,
        chunk_size=args.chunk_size,
        sentences_per_chunk=args.sentences_per_chunk,
    )


if __name__ == "__main__":
    raise SystemExit(main())
