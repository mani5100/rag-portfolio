"""
Integration tests for store.py — requires a real OPENAI_API_KEY.

These tests make real API calls and write to disk (tmp_path).
Run with: uv run pytest tests/integration/test_store_integration.py -v
"""

from __future__ import annotations

import os

import pytest
from langchain_core.documents import Document

from rag_porfolio.store import EmbeddingService, VectorStore

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def require_api_key():
    """Skip every test in this module if OPENAI_API_KEY is not set."""
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set — skipping integration tests")


# ---------------------------------------------------------------------------
# EmbeddingService integration
# ---------------------------------------------------------------------------


def test_real_embeddings_produce_correct_dimension():
    """Real embed_documents() must return 1536-dim non-zero vectors."""
    service = EmbeddingService()  # uses text-embedding-3-small (1536 dims)
    texts = ["hello world", "test document", "another string"]

    vectors = service.embed_documents(texts)

    assert len(vectors) == 3
    for vec in vectors:
        assert len(vec) == 1536, f"Expected 1536 dims, got {len(vec)}"
        assert sum(vec) != 0, "Vector must not be all zeros"


def test_real_embed_query_returns_vector():
    """embed_query() must return a single 1536-dim vector."""
    service = EmbeddingService()
    vec = service.embed_query("what is retrieval augmented generation?")

    assert isinstance(vec, list)
    assert len(vec) == 1536
    assert sum(vec) != 0


# ---------------------------------------------------------------------------
# VectorStore integration — persistence
# ---------------------------------------------------------------------------


def test_chroma_persistence(tmp_path):
    """Documents added to VectorStore must survive across separate instances."""
    persist_dir = str(tmp_path / "chroma")

    # --- Instance 1: add documents ---
    store1 = VectorStore(
        embedding=EmbeddingService(),
        persist_directory=persist_dir,
    )
    store1.add_documents([
        Document(page_content="Persistence test doc A", metadata={"source": "persist_a.pdf"}),
        Document(page_content="Persistence test doc B", metadata={"source": "persist_b.pdf"}),
    ])

    # --- Instance 2: open same directory and verify data is there ---
    store2 = VectorStore(
        embedding=EmbeddingService(),
        persist_directory=persist_dir,
    )

    assert store2.is_document_indexed("persist_a.pdf"), "persist_a.pdf should survive reload"
    assert store2.is_document_indexed("persist_b.pdf"), "persist_b.pdf should survive reload"

    results = store2.similarity_search("Persistence test", k=5)
    assert len(results) >= 1, "Should find at least one document after reload"


def test_chroma_delete_persists(tmp_path):
    """delete_by_source() result must also persist across instances."""
    persist_dir = str(tmp_path / "chroma_del")

    store1 = VectorStore(
        embedding=EmbeddingService(),
        persist_directory=persist_dir,
    )
    store1.add_documents([
        Document(page_content="Will be deleted", metadata={"source": "gone.pdf"}),
        Document(page_content="Will be kept", metadata={"source": "stay.pdf"}),
    ])
    store1.delete_by_source("gone.pdf")

    store2 = VectorStore(
        embedding=EmbeddingService(),
        persist_directory=persist_dir,
    )
    assert store2.is_document_indexed("gone.pdf") is False
    assert store2.is_document_indexed("stay.pdf") is True
