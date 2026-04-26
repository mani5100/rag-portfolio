"""
Unit tests for retrieval.py — Retriever and Reranker.

Rules:
- All tests marked @pytest.mark.unit
- Zero real API calls: all external services are mocked
- FlashrankRerank is patched to avoid downloading models
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document


# ---------------------------------------------------------------------------
# Retriever tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_retriever_returns_documents():
    """Retriever.retrieve() must return a list of Documents with count <= 5."""
    from rag_porfolio.retrieval import Retriever

    mock_store = MagicMock()
    mock_store.similarity_search.return_value = [
        Document(page_content=f"Document {i}", metadata={"source": f"doc{i}.txt"})
        for i in range(5)
    ]
    # as_retriever is used internally; patch it to return a retriever that yields known docs
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [
        Document(page_content=f"Document {i}", metadata={"source": f"doc{i}.txt"})
        for i in range(5)
    ]
    mock_store.as_retriever.return_value = mock_retriever
    mock_store._store = MagicMock()
    mock_store._store.as_retriever.return_value = mock_retriever

    retriever = Retriever(mock_store)
    results = retriever.retrieve("what is RAG?")

    assert isinstance(results, list)
    assert len(results) <= 5
    for doc in results:
        assert isinstance(doc, Document)


@pytest.mark.unit
def test_retriever_raises_on_empty_query():
    """Retriever.retrieve() must raise ValueError for empty or whitespace-only queries."""
    from rag_porfolio.retrieval import Retriever

    mock_store = MagicMock()
    retriever = Retriever(mock_store)

    with pytest.raises(ValueError, match="Query must not be empty"):
        retriever.retrieve("")

    with pytest.raises(ValueError, match="Query must not be empty"):
        retriever.retrieve("   ")


@pytest.mark.unit
def test_retriever_no_duplicate_page_content():
    """Retriever.retrieve() must not return two docs with identical page_content."""
    from rag_porfolio.retrieval import Retriever

    mock_store = MagicMock()
    unique_docs = [
        Document(page_content=f"Unique content {i}", metadata={"source": f"doc{i}.txt"})
        for i in range(5)
    ]
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = unique_docs
    mock_store._store = MagicMock()
    mock_store._store.as_retriever.return_value = mock_retriever

    retriever = Retriever(mock_store)
    results = retriever.retrieve("some query")

    contents = [doc.page_content for doc in results]
    assert len(contents) == len(set(contents)), "Duplicate page_content found in results"


# ---------------------------------------------------------------------------
# Reranker tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_reranker_returns_documents():
    """Reranker.retrieve_and_rerank() must return a list of Documents."""
    from rag_porfolio.retrieval import Reranker, Retriever

    input_docs = [
        Document(page_content=f"Doc {i}", metadata={"source": f"doc{i}.txt"})
        for i in range(5)
    ]
    reranked_subset = input_docs[:3]

    mock_retriever = MagicMock(spec=Retriever)
    mock_retriever.retrieve.return_value = input_docs

    mock_compressor = MagicMock()
    mock_compressor.compress_documents.return_value = reranked_subset

    with patch("rag_porfolio.retrieval.FlashrankRerank", return_value=mock_compressor):
        reranker = Reranker(mock_retriever, top_n=5)
        results = reranker.retrieve_and_rerank("query about something")

    assert isinstance(results, list)
    for doc in results:
        assert isinstance(doc, Document)


@pytest.mark.unit
def test_reranker_count_lte_top_n():
    """Reranker.retrieve_and_rerank() must return at most top_n documents."""
    from rag_porfolio.retrieval import Reranker, Retriever

    input_docs = [
        Document(page_content=f"Doc {i}", metadata={"source": f"doc{i}.txt"})
        for i in range(5)
    ]
    reranked_subset = input_docs[:3]

    mock_retriever = MagicMock(spec=Retriever)
    mock_retriever.retrieve.return_value = input_docs

    mock_compressor = MagicMock()
    mock_compressor.compress_documents.return_value = reranked_subset

    with patch("rag_porfolio.retrieval.FlashrankRerank", return_value=mock_compressor):
        reranker = Reranker(mock_retriever, top_n=5)
        results = reranker.retrieve_and_rerank("query about something")

    assert len(results) <= 5


@pytest.mark.unit
def test_reranker_raises_on_empty_query():
    """Reranker.retrieve_and_rerank() must propagate ValueError from Retriever on empty query."""
    from rag_porfolio.retrieval import Reranker, Retriever

    mock_retriever = MagicMock(spec=Retriever)
    mock_retriever.retrieve.side_effect = ValueError("Query must not be empty")

    mock_compressor = MagicMock()

    with patch("rag_porfolio.retrieval.FlashrankRerank", return_value=mock_compressor):
        reranker = Reranker(mock_retriever, top_n=5)

        with pytest.raises(ValueError, match="Query must not be empty"):
            reranker.retrieve_and_rerank("")


@pytest.mark.unit
def test_reranker_output_is_subset_of_input():
    """All docs returned by reranker must have been in the original retrieved set."""
    from rag_porfolio.retrieval import Reranker, Retriever

    input_docs = [
        Document(page_content=f"Known doc {i}", metadata={"source": f"doc{i}.txt"})
        for i in range(5)
    ]
    # Reranker returns 3 of the original 5, in a different order
    reranked_subset = [input_docs[4], input_docs[1], input_docs[2]]

    mock_retriever = MagicMock(spec=Retriever)
    mock_retriever.retrieve.return_value = input_docs

    mock_compressor = MagicMock()
    mock_compressor.compress_documents.return_value = reranked_subset

    with patch("rag_porfolio.retrieval.FlashrankRerank", return_value=mock_compressor):
        reranker = Reranker(mock_retriever, top_n=5)
        results = reranker.retrieve_and_rerank("some query")

    input_contents = {doc.page_content for doc in input_docs}
    for doc in results:
        assert doc.page_content in input_contents, (
            f"Returned doc '{doc.page_content}' was not in the original retrieved set"
        )
