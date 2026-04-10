# -*- coding: utf-8 -*-
"""
Calculate Retrieval Quality (Precision@5) for different chunking strategies.

This script compares three chunking strategies using keyword-based relevance.
"""

import json
import os
import re
from pathlib import Path
from statistics import mean
from typing import List

from src.chunking import (
    FixedSizeChunker,
    SentenceChunker,
    RecursiveChunker,
)
from src.embeddings import LOCAL_EMBEDDING_MODEL, MockEmbedder, LocalEmbedder
from src.models import Document
from src.store import EmbeddingStore


def extract_keywords(text: str) -> List[str]:
    """Extract keywords from text for relevance matching."""
    text = text.lower()
    words = re.findall(r"\b[a-z0-9]+\b", text)
    stopwords = {
        "la", "va", "o", "se", "co", "toi", "tay", "neu", "khi",
        "trong", "tren", "duoi", "giua", "hai", "ba", "bon",
        "nam", "sau", "bay", "tam", "chin", "muoi", "mot", "cai",
        "nay", "do", "ma", "cho", "noi", "den", "cong", "khong"
    }
    keywords = [w for w in words if len(w) >= 2 and w not in stopwords]
    return keywords


def calculate_relevance(chunk_content: str, chunk_id: str, query: str, label: str) -> bool:
    """Determine if chunk is relevant based on keywords from label."""
    chunk_lower = chunk_content.lower()
    # label_keywords = extract_keywords(label)
    # query_keywords = extract_keywords(query)
    print('chunk_id', chunk_id, 'label', label)
    is_relevant = label.lower() in chunk_id.lower() or label.lower() in chunk_content.lower()
    if is_relevant:
        return True
    return False

def load_queries(queries_file: str) -> list:
    """Load benchmark queries from JSON file."""
    with open(queries_file, "r", encoding="utf-8") as f:
        return json.load(f)

def load_documents(data_dir: str) -> list:
    """Load all markdown documents from data directory."""
    documents = []
    data_path = Path(data_dir)

    file_mappings = {
        "alzheimer.md": "https://tamanhhospital.vn/alzheimer/",
        "benh_lao_phoi.md": "https://tamanhhospital.vn/benh-lao-phoi/",
        "benh-dai.md": "https://tamanhhospital.vn/benh-dai/",
        "benh-san-day.md": "https://tamanhhospital.vn/benh-san-day/",
        "benh-tri.md": "https://tamanhhospital.vn/benh-tri/",
    }

    for filename, source_url in file_mappings.items():
        filepath = data_path / filename
        if filepath.exists():
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            doc = Document(
                id=filename.replace(".md", ""),
                content=content,
                metadata={"source": source_url, "category": "medical"},
            )
            documents.append(doc)

    return documents


def evaluate_strategy(
    strategy_name: str,
    chunker,
    documents: list,
    queries: list,
    embedding_fn,
    top_k: int = 3,
) -> dict:
    """Evaluate a chunking strategy using Precision@k metric."""
    
    store = EmbeddingStore(
        collection_name=f"{strategy_name}_collection",
        embedding_fn=embedding_fn,
    )

    all_chunks = []
    for doc in documents:
        chunks = chunker.chunk(doc.content)
        for i, chunk_content in enumerate(chunks):
            chunk_doc = Document(
                id=f"{doc.id}_chunk_{i}",
                content=chunk_content,
                metadata=doc.metadata,
            )
            all_chunks.append(chunk_doc)

    store.add_documents(all_chunks)

    precisions = []
    for query_dict in queries:
        question = query_dict.get("question", "")
        answer = query_dict.get("answer", "")
        label = query_dict.get("label", "")

        results = store.search(question, top_k=top_k)

        relevant_count = 0
        for result in results:
            chunk_content = result.get("content", "")
            chunk_id = result.get("id", "")

            is_relevant = calculate_relevance(chunk_content, chunk_id, question, label)
            if is_relevant:
                relevant_count += 1

        precision_k = relevant_count / top_k if top_k > 0 else 0
        precisions.append(precision_k)

    avg_precision = mean(precisions) if precisions else 0

    total_chunks = len(all_chunks)
    avg_chunk_length = (
        sum(len(chunk.content) for chunk in all_chunks) / total_chunks
        if total_chunks > 0
        else 0
    )

    return {
        "strategy": strategy_name,
        "chunk_count": total_chunks,
        "avg_chunk_length": round(avg_chunk_length, 1),
        "precisions": precisions,
        "avg_precision": round(avg_precision, 4),
        "retrieval_quality": f"{avg_precision * 100:.1f}%",
    }


def main():
    """Main execution function."""
    print("=" * 80)
    print("RETRIEVAL QUALITY EVALUATION")
    print("=" * 80)

    queries_file = "data/queries.json"
    data_dir = "data"
    chunk_size = 500
    embedding_fn = LocalEmbedder(model_name=os.getenv("LOCAL_EMBEDDING_MODEL", LOCAL_EMBEDDING_MODEL))

    print("\n[1] Loading data...")
    queries = load_queries(queries_file)
    documents = load_documents(data_dir)
    print(f"    Loaded {len(queries)} queries and {len(documents)} documents")

    strategies = [
        # ("fixed_size", FixedSizeChunker(chunk_size=chunk_size, overlap=50)),
        ("by_sentences", SentenceChunker(max_sentences_per_chunk=3)),
        ("recursive", RecursiveChunker(chunk_size=chunk_size)),
    ]

    print("\n[2] Evaluating strategies...")
    results = []

    for strategy_name, chunker in strategies:
        print(f"\n    Evaluating {strategy_name}...")
        result = evaluate_strategy(
            strategy_name=strategy_name,
            chunker=chunker,
            documents=documents,
            queries=queries,
            embedding_fn=embedding_fn,
            top_k=5,
        )
        results.append(result)
        print(f"      [OK] {strategy_name}: {result['retrieval_quality']}")

    print("\n" + "=" * 80)
    print("RESULTS TABLE")
    print("=" * 80)

    print("\n| Strategy | Chunk Count | Avg Length | Precision@5 | Retrieval Quality |")
    print("|----------|-------------|------------|-------------|-------------------|")

    for result in results:
        print(
            f"| {result['strategy']:15} | {result['chunk_count']:11} | "
            f"{result['avg_chunk_length']:10.1f} | "
            f"{result['avg_precision']:11.4f} | {result['retrieval_quality']:17} |"
        )

    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    fixed_result = next(r for r in results if r["strategy"] == "fixed_size")
    recursive_result = next(r for r in results if r["strategy"] == "recursive")
    sentence_result = next(r for r in results if r["strategy"] == "by_sentences")

    best_baseline = fixed_result

    print(f"\nBest Baseline Strategy: {best_baseline['strategy'].upper()}")
    print(f"  - Chunk Count: {best_baseline['chunk_count']}")
    print(f"  - Avg Length: {best_baseline['avg_chunk_length']}")
    print(f"  - Retrieval Quality: {best_baseline['retrieval_quality']}")

    print(f"\nYour Strategy: SentenceChunker (by_sentences)")
    print(f"  - Chunk Count: {sentence_result['chunk_count']}")
    print(f"  - Avg Length: {sentence_result['avg_chunk_length']}")
    print(f"  - Retrieval Quality: {sentence_result['retrieval_quality']}")

    improvement = (
        sentence_result["avg_precision"] - best_baseline["avg_precision"]
    ) * 100
    print(f"\nImprovement vs Baseline: {improvement:+.1f}%")

    print("\n" + "=" * 80)
    print("PER-QUERY PRECISION@5")
    print("=" * 80)

    print("\n| Query # | FixedSize | SentenceChunker | RecursiveChunker |")
    print("|---------|-----------|-----------------|-----------------|")

    for i in range(len(queries)):
        fixed_p = fixed_result["precisions"][i]
        sentence_p = sentence_result["precisions"][i]
        recursive_p = recursive_result["precisions"][i]
        print(
            f"|    {i+1:2d}    | {fixed_p:9.3f} | {sentence_p:15.3f} | {recursive_p:17.3f} |"
        )

    print("\n" + "=" * 80)
    print("OUTPUT FOR REPORT.MD")
    print("=" * 80)

    print("\n### So Sanh: Strategy cua toi vs Baseline\n")
    print("| Tai lieu | Strategy | Chunk Count | Avg Length | Retrieval Quality |")
    print("|-----------|----------|-------------|------------|--------------------|")
    print(
        f"| All docs | {best_baseline['strategy']} (baseline) | {best_baseline['chunk_count']} | "
        f"{best_baseline['avg_chunk_length']} | {best_baseline['retrieval_quality']} |"
    )
    print(
        f"| All docs | **SentenceChunker** (cua toi) | {sentence_result['chunk_count']} | "
        f"{sentence_result['avg_chunk_length']} | **{sentence_result['retrieval_quality']}** |"
    )

    return results


if __name__ == "__main__":
    main()
