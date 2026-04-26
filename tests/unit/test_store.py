"""
Unit tests for store.py — EmbeddingService and VectorStore.

Rules:
- All tests marked @pytest.mark.unit
- Zero real API calls: OpenAIEmbeddings is always patched
- VectorStore uses chromadb.EphemeralClient() for in-memory isolation
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import chromadb
import pytest
from langchain_core.documents import Document
from langchain_core.embeddings.fake import FakeEmbeddings

from rag_porfolio.store import EmbeddingError, EmbeddingService, VectorStore


# ---------------------------------------------------------------------------
# EmbeddingService tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_embedding_service_empty_input_returns_empty():
    """embed_documents([]) must return [] without touching the underlying model."""
    with patch("rag_porfolio.store.OpenAIEmbeddings") as MockEmb:
        service = EmbeddingService()
        result = service.embed_documents([])

    assert result == []
    # The underlying embeddings object must never have been called
    MockEmb.return_value.embed_documents.assert_not_called()


@pytest.mark.unit
def test_embedding_service_wraps_exception_in_embedding_error():
    """Any exception from the underlying model must be re-raised as EmbeddingError."""
    with patch("rag_porfolio.store.OpenAIEmbeddings") as MockEmb:
        MockEmb.return_value.embed_documents.side_effect = RuntimeError("boom")
        service = EmbeddingService()

        with pytest.raises(EmbeddingError):
            service.embed_documents(["hello"])


@pytest.mark.unit
def test_embedding_service_query_wraps_exception_in_embedding_error():
    """embed_query exceptions must also be wrapped in EmbeddingError."""
    with patch("rag_porfolio.store.OpenAIEmbeddings") as MockEmb:
        MockEmb.return_value.embed_query.side_effect = RuntimeError("api down")
        service = EmbeddingService()

        with pytest.raises(EmbeddingError):
            service.embed_query("what is RAG?")


@pytest.mark.unit
def test_embedding_service_default_model_name():
    """Constructor must default to 'text-embedding-3-small'."""
    with patch("rag_porfolio.store.OpenAIEmbeddings"):
        service = EmbeddingService()
    assert service.model == "text-embedding-3-small"


@pytest.mark.unit
def test_embedding_service_custom_model_name():
    """Constructor must accept a custom model name."""
    with patch("rag_porfolio.store.OpenAIEmbeddings"):
        service = EmbeddingService(model="text-embedding-ada-002")
    assert service.model == "text-embedding-ada-002"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_store(**kwargs) -> VectorStore:
    """Return a VectorStore backed by an in-memory EphemeralClient."""
    fake_emb = FakeEmbeddings(size=1536)
    client = chromadb.EphemeralClient()
    return VectorStore(embedding=fake_emb, chroma_client=client, **kwargs)


# ---------------------------------------------------------------------------
# VectorStore tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_vector_store_add_documents_does_not_raise():
    """add_documents() with valid Documents must not raise."""
    store = _make_store()
    docs = [
        Document(page_content="First document", metadata={"source": "a.pdf"}),
        Document(page_content="Second document", metadata={"source": "b.pdf"}),
    ]
    store.add_documents(docs)  # should not raise


@pytest.mark.unit
def test_vector_store_similarity_search_returns_documents():
    """similarity_search() must return a list of Documents, each with content."""
    store = _make_store()
    docs = [
        Document(page_content="Alpha content", metadata={"source": "alpha.pdf"}),
        Document(page_content="Beta content", metadata={"source": "beta.pdf"}),
        Document(page_content="Gamma content", metadata={"source": "gamma.pdf"}),
    ]
    store.add_documents(docs)

    results = store.similarity_search("alpha", k=2)

    assert isinstance(results, list)
    assert len(results) <= 2
    for doc in results:
        assert isinstance(doc, Document)
        assert doc.page_content  # non-empty


@pytest.mark.unit
def test_vector_store_is_document_indexed_true():
    """is_document_indexed() returns True when source exists in the store."""
    store = _make_store()
    store.add_documents([
        Document(page_content="Report content", metadata={"source": "report.pdf"}),
    ])

    assert store.is_document_indexed("report.pdf") is True


@pytest.mark.unit
def test_vector_store_is_document_indexed_false():
    """is_document_indexed() returns False when source is not present."""
    store = _make_store()
    # Never added "ghost.pdf"
    assert store.is_document_indexed("ghost.pdf") is False


@pytest.mark.unit
def test_vector_store_delete_by_source_removes_docs():
    """delete_by_source() removes only the matching source, leaving others intact."""
    store = _make_store()
    store.add_documents([
        Document(page_content="Keep this", metadata={"source": "keep.pdf"}),
        Document(page_content="Delete this", metadata={"source": "delete.pdf"}),
    ])

    store.delete_by_source("delete.pdf")

    assert store.is_document_indexed("delete.pdf") is False
    assert store.is_document_indexed("keep.pdf") is True


@pytest.mark.unit
def test_vector_store_delete_by_source_nonexistent_does_not_raise():
    """delete_by_source() on a source that doesn't exist must not raise."""
    store = _make_store()
    store.delete_by_source("nonexistent.pdf")  # should not raise


@pytest.mark.unit
def test_vector_store_similarity_search_respects_k():
    """similarity_search() must return at most k results."""
    store = _make_store()
    docs = [
        Document(page_content=f"Document number {i}", metadata={"source": f"doc{i}.pdf"})
        for i in range(10)
    ]
    store.add_documents(docs)

    results = store.similarity_search("document", k=3)
    assert len(results) <= 3
