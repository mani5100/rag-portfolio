"""
Retrieval and reranking layer.

Responsibilities:
- Retriever: MMR-based retrieval from a VectorStore (k=5, fetch_k=20, lambda_mult=0.5)
- Reranker: Applies FlashrankRerank on top of Retriever results
"""

from __future__ import annotations

from langchain_community.document_compressors import FlashrankRerank
from langchain_core.documents import Document

from rag_porfolio.store import VectorStore


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------


class Retriever:
    """MMR-based retriever backed by a VectorStore.

    Parameters
    ----------
    vector_store:
        A :class:`~rag_porfolio.store.VectorStore` instance to retrieve from.
    """

    def __init__(self, vector_store: VectorStore) -> None:
        self._vector_store = vector_store
        self._mmr_retriever = vector_store._store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 5,
                "fetch_k": 20,
                "lambda_mult": 0.5,
            },
        )

    def retrieve(self, query: str) -> list[Document]:
        """Return up to 5 diverse documents relevant to *query* using MMR.

        Parameters
        ----------
        query:
            The search query string.

        Returns
        -------
        list[Document]
            Up to 5 Documents with no two having identical ``page_content``.

        Raises
        ------
        ValueError
            If *query* is empty or whitespace-only.
        """
        if not query.strip():
            raise ValueError("Query must not be empty")

        docs = self._mmr_retriever.invoke(query)

        # Deduplicate by page_content (MMR should already prevent this,
        # but guard explicitly as specified)
        seen: set[str] = set()
        unique_docs: list[Document] = []
        for doc in docs:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                unique_docs.append(doc)

        return unique_docs


# ---------------------------------------------------------------------------
# Reranker
# ---------------------------------------------------------------------------


class Reranker:
    """Reranks retrieved documents using FlashrankRerank.

    Parameters
    ----------
    retriever:
        A :class:`Retriever` instance used to fetch candidate documents.
    top_n:
        Maximum number of documents to return after reranking.
        Defaults to ``5``.
    """

    def __init__(self, retriever: Retriever, top_n: int = 5) -> None:
        self._retriever = retriever
        self.top_n = top_n
        self._compressor = FlashrankRerank(
            model="ms-marco-MiniLM-L-12-v2",
            top_n=self.top_n,
        )

    def retrieve_and_rerank(self, query: str) -> list[Document]:
        """Retrieve candidates and rerank them using Flashrank.

        Parameters
        ----------
        query:
            The search query string.

        Returns
        -------
        list[Document]
            Reranked Documents (count <= top_n). All returned documents are a
            subset/reordering of the documents returned by the retriever —
            no new documents are introduced.

        Raises
        ------
        ValueError
            Propagated from :meth:`Retriever.retrieve` when *query* is empty.
        """
        # Raises ValueError on empty query — propagated as-is
        candidates = self._retriever.retrieve(query)

        if not candidates:
            return []

        reranked = self._compressor.compress_documents(candidates, query)
        return list(reranked)
