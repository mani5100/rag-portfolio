"""
Integration tests for chain.py — RAGChain end-to-end.

Requires OPENAI_API_KEY to be set. Skipped otherwise.
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def require_api_key():
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")


def test_chain_answers_from_document(tmp_path):
    """Index a doc with a known fact; query for it; assert it appears in response."""
    import chromadb
    from langchain_core.documents import Document
    from rag_porfolio.store import EmbeddingService, VectorStore
    from rag_porfolio.retrieval import Retriever, Reranker
    from rag_porfolio.chain import RAGChain

    embedding = EmbeddingService()
    store = VectorStore(embedding=embedding._embeddings, persist_directory=str(tmp_path))
    store.add_documents([
        Document(
            page_content="The Eiffel Tower is located in Paris, France and was built in 1889.",
            metadata={"source": "facts.txt"}
        )
    ])
    retriever = Retriever(store)
    reranker = Reranker(retriever)
    chain = RAGChain(reranker=reranker)

    result = chain.invoke("Where is the Eiffel Tower located?")
    assert "Paris" in result or "France" in result


def test_chain_returns_fallback_when_no_relevant_docs(tmp_path):
    """RAGChain returns the fallback string when the indexed content is unrelated."""
    from langchain_core.documents import Document
    from rag_porfolio.store import EmbeddingService, VectorStore
    from rag_porfolio.retrieval import Retriever, Reranker
    from rag_porfolio.chain import RAGChain

    FALLBACK = "I could not find relevant information in the indexed documents."

    embedding = EmbeddingService()
    store = VectorStore(embedding=embedding._embeddings, persist_directory=str(tmp_path))
    # Index a completely unrelated single doc so retriever returns empty after reranking threshold
    store.add_documents([
        Document(
            page_content="The sky is blue on a clear day.",
            metadata={"source": "sky.txt"}
        )
    ])
    retriever = Retriever(store)
    # Use a Reranker mock that always returns empty to simulate no relevant results
    from unittest.mock import MagicMock
    reranker = MagicMock()
    reranker.retrieve_and_rerank.return_value = []
    chain = RAGChain(reranker=reranker)

    result = chain.invoke("What is the capital of France?")
    assert result == FALLBACK
