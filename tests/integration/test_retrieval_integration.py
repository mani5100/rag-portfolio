"""
Integration tests for retrieval.py — Retriever and Reranker.

Requirements:
- OPENAI_API_KEY must be set; tests are skipped otherwise.
- Uses a real Chroma store (tmp_path, ephemeral) with real OpenAI embeddings.
- Uses real FlashrankRerank model for reranking.
"""

from __future__ import annotations

import os

import pytest
from langchain_core.documents import Document

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def require_api_key():
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")


def test_reranker_surfaces_correct_document(tmp_path):
    """Index 3 docs with distinct content; query for term only in doc B;
    assert doc B is at position 0 in reranked results."""
    from rag_porfolio.retrieval import Reranker, Retriever
    from rag_porfolio.store import EmbeddingService, VectorStore

    embedding = EmbeddingService()
    store = VectorStore(
        embedding=embedding._embeddings,
        persist_directory=str(tmp_path),
    )

    doc_a = Document(
        page_content="The mitochondria is the powerhouse of the cell.",
        metadata={"source": "bio.txt"},
    )
    doc_b = Document(
        page_content="Quantum entanglement describes correlated quantum states.",
        metadata={"source": "physics.txt"},
    )
    doc_c = Document(
        page_content="The French Revolution began in 1789.",
        metadata={"source": "history.txt"},
    )

    store.add_documents([doc_a, doc_b, doc_c])

    retriever = Retriever(store)
    reranker = Reranker(retriever)

    results = reranker.retrieve_and_rerank("Tell me about quantum physics and entanglement")

    assert len(results) >= 1
    assert results[0].metadata["source"] == "physics.txt"
