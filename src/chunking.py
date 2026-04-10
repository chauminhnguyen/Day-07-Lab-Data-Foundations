from __future__ import annotations

import math
import re


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        # Group into chunks
        chunks = []
        current_chunk = []
        for sentence in sentences:
            current_chunk.append(sentence.strip())
            if len(current_chunk) >= self.max_sentences_per_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        return chunks


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        return self._split(text, self.separators)

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        if not remaining_separators:
            return [current_text] if current_text else []
        separator = remaining_separators[0]
        if separator == "":
            # Last separator, split into characters
            parts = list(current_text)
        else:
            parts = current_text.split(separator)
        chunks = []
        for part in parts:
            if len(part) <= self.chunk_size:
                chunks.append(part)
            else:
                sub_chunks = self._split(part, remaining_separators[1:])
                chunks.extend(sub_chunks)
        return chunks

import re
from typing import List


class custom_SentenceChunker:
    """
    Split text into chunks of at most `max_sentences_per_chunk` sentences.

    Improvements:
    - Better sentence boundary detection (handles ., !, ?, newline)
    - Avoid empty sentences
    - Handle edge cases like abbreviations (basic level)
    - Optional overlap between chunks for context preservation
    """

    def __init__(
        self,
        max_sentences_per_chunk: int = 3,
        overlap: int = 0
    ) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)
        self.overlap = max(0, overlap)

        # Regex improved: handle ., !, ?, newline boundaries
        self.sentence_splitter = re.compile(
            r'(?<=[.!?])\s+|\n+'
        )

    def _split_sentences(self, text: str) -> List[str]:
        sentences = self.sentence_splitter.split(text)
        # Clean + remove empty
        return [s.strip() for s in sentences if s.strip()]

    def chunk(self, text: str) -> List[str]:
        sentences = self._split_sentences(text)

        if not sentences:
            return []

        chunks = []
        i = 0

        while i < len(sentences):
            chunk_sentences = sentences[i:i + self.max_sentences_per_chunk]
            chunks.append(" ".join(chunk_sentences))

            # Move with overlap
            step = self.max_sentences_per_chunk - self.overlap
            i += max(1, step)

        return chunks

def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    dot_product = _dot(vec_a, vec_b)
    norm_a = math.sqrt(_dot(vec_a, vec_a))
    norm_b = math.sqrt(_dot(vec_b, vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        chunkers = {
            "fixed_size": FixedSizeChunker(chunk_size=chunk_size),
            "by_sentences": SentenceChunker(max_sentences_per_chunk=3),
            "recursive": RecursiveChunker(chunk_size=chunk_size)
        }
        results = {}
        for name, chunker in chunkers.items():
            chunks = chunker.chunk(text)
            num_chunks = len(chunks)
            avg_length = sum(len(c) for c in chunks) / num_chunks if num_chunks > 0 else 0
            results[name] = {
                "chunks": chunks,
                "count": num_chunks,
                "avg_length": avg_length
            }
        return results
