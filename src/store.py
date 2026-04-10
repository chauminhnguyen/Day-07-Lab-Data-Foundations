from __future__ import annotations

from typing import Any, Callable

from .chunking import _dot, compute_similarity
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    """
    A vector store for text chunks.

    Tries to use ChromaDB if available; falls back to an in-memory store.
    The embedding_fn parameter allows injection of mock embeddings for tests.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._next_index = 0

        try:
            import chromadb  # noqa: F401

            self._client = chromadb.PersistentClient(path="./chroma_db")
            self._collection = self._client.get_or_create_collection(name=collection_name)
            self._use_chroma = True
        except Exception:
            self._use_chroma = False
            self._collection = None

    def _make_record(self, doc: Document) -> dict[str, Any]:
        embedding = self._embedding_fn(doc.content)
        return {
            "id": f"{doc.id}_{self._next_index}",
            "content": doc.content,
            "embedding": embedding,
            "metadata": doc.metadata
        }

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        query_emb = self._embedding_fn(query)
        similarities = []
        for record in records:
            sim = compute_similarity(query_emb, record["embedding"])
            similarities.append((sim, record))
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [{"id": rec["id"], "content": rec["content"], "metadata": rec["metadata"], "score": sim} for sim, rec in similarities[:top_k]]

    def add_documents(self, docs: list[Document]) -> None:
        """
        Embed each document's content and store it.

        For ChromaDB: use collection.add(ids=[...], documents=[...], embeddings=[...])
        For in-memory: append dicts to self._store
        """
        records = []
        ids = []
        documents = []
        embeddings = []
        for doc in docs:
            record = self._make_record(doc)
            records.append(record)
            ids.append(record["id"])
            documents.append(record["content"])
            embeddings.append(record["embedding"])
            self._next_index += 1
        if self._use_chroma:
            self._collection.add(ids=ids, documents=documents, embeddings=embeddings, metadatas=[r["metadata"] for r in records])
        else:
            self._store.extend(records)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Find the top_k most similar documents to query.

        For in-memory: compute dot product of query embedding vs all stored embeddings.
        """
        if self._use_chroma:
            query_emb = self._embedding_fn(query)
            results = self._collection.query(query_embeddings=[query_emb], n_results=top_k)
            return [
                {"id": id_, "content": doc, "metadata": meta, "score": 1 - dist}  # Assuming distance is cosine distance
                for id_, doc, meta, dist in zip(
                    results["ids"][0], results["documents"][0], results["metadatas"][0], results["distances"][0]
                )
            ]
        else:
            query_emb = self._embedding_fn(query)
            similarities = []
            for record in self._store:
                sim = _dot(query_emb, record["embedding"])
                similarities.append((sim, record))
            similarities.sort(key=lambda x: x[0], reverse=True)
            return [{"id": rec["id"], "content": rec["content"], "metadata": rec["metadata"], "score": sim} for sim, rec in similarities[:top_k]]

    def get_collection_size(self) -> int:
        """Return the total number of stored chunks."""
        if self._use_chroma:
            return self._collection.count()
        else:
            return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        """
        Search with optional metadata pre-filtering.

        First filter stored chunks by metadata_filter, then run similarity search.
        """
        if metadata_filter is None:
            return self.search(query, top_k)
        if self._use_chroma:
            query_emb = self._embedding_fn(query)
            results = self._collection.query(query_embeddings=[query_emb], n_results=top_k, where=metadata_filter)
            return [
                {"id": id_, "content": doc, "metadata": meta, "score": 1 - dist}  # Assuming distance is cosine distance
                for id_, doc, meta, dist in zip(
                    results["ids"][0], results["documents"][0], results["metadatas"][0], results["distances"][0]
                )
            ]
        else:
            filtered_records = [r for r in self._store if all(r["metadata"].get(k) == v for k, v in metadata_filter.items())]
            return self._search_records(query, filtered_records, top_k)

    def delete_document(self, doc_id: str) -> bool:
        """
        Remove all chunks belonging to a document.

        Returns True if any chunks were removed, False otherwise.
        """
        if self._use_chroma:
            initial_count = self._collection.count()
            # ChromaDB delete where id starts with doc_id_
            ids_to_delete = [id_ for id_ in self._collection.get()["ids"] if id_.startswith(f"{doc_id}_")]
            if ids_to_delete:
                self._collection.delete(ids=ids_to_delete)
            return self._collection.count() < initial_count
        else:
            initial_len = len(self._store)
            self._store = [r for r in self._store if not r["id"].startswith(f"{doc_id}_")]
            return len(self._store) < initial_len
