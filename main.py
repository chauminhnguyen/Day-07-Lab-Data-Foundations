from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
import google.generativeai as genai

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
    """Call Gemini API for RAG testing."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "[GEMINI ERROR] GEMINI_API_KEY not set in environment."

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")  # or another model

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"[GEMINI ERROR] {str(e)}"


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
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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

    logging.info(f"Loaded {len(docs)} documents from {len(files)} files.")

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

    # Chunk Coherence Analysis
    chunk_lengths = [len(doc.content) for doc in docs]
    avg_chunk_length = sum(chunk_lengths) / len(chunk_lengths) if chunk_lengths else 0
    print(f"Chunk Coherence: Avg chunk length = {avg_chunk_length:.1f} chars")
    print(f"  Total chunks: {len(docs)}, Min length: {min(chunk_lengths)}, Max length: {max(chunk_lengths)}")

    logging.info(f"Chunked documents using {chunker_name}: {len(docs)} chunks, avg length {avg_chunk_length:.1f} chars")

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

    logging.info(f"Stored {store.get_collection_size()} documents in EmbeddingStore.")
    print("\n=== EmbeddingStore Search Test ===")
    print(f"Query: {query}")
    search_results = store.search(query, top_k=3)
    for index, result in enumerate(search_results, start=1):
        print(f"{index}. score={result['score']:.3f} source={result['metadata'].get('source')}")
        print(f"   content preview: {result['content'][:120].replace(chr(10), ' ')}...")

    # Retrieval Precision Analysis
    scores = [result['score'] for result in search_results]
    avg_score = 0.0
    if scores:
        avg_score = sum(scores) / len(scores)
        score_range = max(scores) - min(scores)
        print(f"Retrieval Precision: Avg score = {avg_score:.3f}, Score range = {score_range:.3f}")
        print(f"  Score distribution: {', '.join(f'{s:.3f}' for s in scores)}")
        # Note: Top-k relevance would require manual judgment, so we print scores for analysis
    else:
        print("Retrieval Precision: No results found")

    logging.info(f"Search query '{query}': retrieved {len(search_results)} results, avg score {avg_score:.3f}")

    # Metadata Utility Analysis: Compare search vs search_with_filter
    print("\n=== Metadata Utility Test ===")
    filter_metadata = {"extension": ".md"}  # Example filter
    filtered_results = store.search_with_filter(query, top_k=3, metadata_filter=filter_metadata)
    print(f"Filtered search (extension='.md'): {len(filtered_results)} results")
    for index, result in enumerate(filtered_results, start=1):
        print(f"{index}. score={result['score']:.3f} source={result['metadata'].get('source')}")
        print(f"   content preview: {result['content']}")

    if filtered_results:
        filtered_scores = [r['score'] for r in filtered_results]
        filtered_avg = sum(filtered_scores) / len(filtered_scores)
        print(f"Filtered avg score: {filtered_avg:.3f} (vs unfiltered {avg_score:.3f})")
    else:
        print("No filtered results found")
        filtered_avg = 0.0

    avg_score_filtered = filtered_avg if filtered_results else 0.0
    logging.info(f"Filtered search (extension='.md'): {len(filtered_results)} results, avg score {avg_score_filtered:.3f}")

    print("\n=== KnowledgeBaseAgent Test ===")
    agent = KnowledgeBaseAgent(store=store, llm_fn=demo_llm)
    print(f"Question: {query}")
    print("Agent answer:")
    answer = agent.answer(query, top_k=3)
    print(answer)

    # Grounding Quality Analysis: Source Traceability
    print("\nGrounding Quality: Sources used for answer:")
    agent_results = store.search(query, top_k=3)  # Same as agent uses
    for index, result in enumerate(agent_results, start=1):
        print(f"{index}. Source: {result['metadata'].get('source')}, Chunk index: {result['metadata'].get('chunk_index')}")
        print(f"   Content snippet: {result['content'][:100].replace(chr(10), ' ')}...")

    logging.info(f"Agent answered query '{query}' using {len(agent_results)} sources.")


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
