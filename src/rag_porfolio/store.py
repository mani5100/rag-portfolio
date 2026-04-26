"""
Embedding and vector store layer.

Responsibilities:
- Wrap OpenAIEmbeddings with error handling (EmbeddingService)
- Provide a Chroma-backed VectorStore with add, search, index-check, and delete
"""

from __future__ import annotations

from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class EmbeddingError(Exception):
    """Raised when an embedding operation fails."""


# ---------------------------------------------------------------------------
# EmbeddingService
# ---------------------------------------------------------------------------


class EmbeddingService:
    """Thin wrapper around OpenAIEmbeddings with safe error handling.

    Parameters
    ----------
    model:
        The OpenAI embedding model to use.
        Defaults to ``"text-embedding-3-small"`` (1536 dimensions).
    """

    def __init__(self, model: str = "text-embedding-3-small") -> None:
        self.model = model
        self._embeddings = OpenAIEmbeddings(model=self.model)

    # ------------------------------------------------------------------
    # Embeddings protocol (satisfies langchain_core.embeddings.Embeddings)
    # ------------------------------------------------------------------

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return embedding vectors for a list of texts.

        Returns ``[]`` immediately for empty input (no API call).
        Wraps any underlying exception in :class:`EmbeddingError`.
        """
        if not texts:
            return []

        try:
            return self._embeddings.embed_documents(texts)
        except Exception as exc:
            raise EmbeddingError(
                f"Failed to embed {len(texts)} document(s): {exc}"
            ) from exc

    def embed_query(self, text: str) -> list[float]:
        """Return an embedding vector for a single query string.

        Wraps any underlying exception in :class:`EmbeddingError`.
        """
        try:
            return self._embeddings.embed_query(text)
        except Exception as exc:
            raise EmbeddingError(
                f"Failed to embed query: {exc}"
            ) from exc


# ---------------------------------------------------------------------------
# VectorStore
# ---------------------------------------------------------------------------


class VectorStore:
    """Chroma-backed vector store with document management helpers.

    Parameters
    ----------
    embedding:
        Any LangChain-compatible Embeddings object.
    persist_directory:
        Directory where Chroma persists data.  Ignored when
        ``chroma_client`` is supplied.
    chroma_client:
        Optional pre-built chromadb client (e.g. ``chromadb.EphemeralClient()``).
        When provided the store operates fully in-memory — useful for unit tests.
    collection_name:
        Chroma collection name.  Defaults to ``"rag_collection"``.
    """

    def __init__(
        self,
        embedding: Embeddings,
        persist_directory: str = "vectorstore/",
        *,
        chroma_client: Any | None = None,
        collection_name: str = "rag_collection",
    ) -> None:
        self._embedding = embedding

        if chroma_client is not None:
            # In-memory / injected client — used in unit tests
            self._store = Chroma(
                client=chroma_client,
                collection_name=collection_name,
                embedding_function=embedding,
            )
        else:
            # Persistent store — used in production and integration tests
            self._store = Chroma(
                collection_name=collection_name,
                embedding_function=embedding,
                persist_directory=persist_directory,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_documents(self, documents: list[Document]) -> None:
        """Add *documents* to the vector store."""
        self._store.add_documents(documents)

    def similarity_search(self, query: str, k: int = 5) -> list[Document]:
        """Return the top-*k* documents most similar to *query*."""
        return self._store.similarity_search(query, k=k)

    def is_document_indexed(self, filename: str) -> bool:
        """Return ``True`` if any document with ``metadata["source"] == filename`` exists."""
        results = self._store.get(where={"source": filename})
        return len(results["ids"]) > 0

    def delete_by_source(self, filename: str) -> None:
        """Delete all documents whose ``metadata["source"] == filename``.

        No-op (does not raise) if no documents match.
        """
        results = self._store.get(where={"source": filename})
        ids_to_delete = results["ids"]
        if ids_to_delete:
            self._store.delete(ids=ids_to_delete)
