"""
RAG generation chain.

Responsibilities:
- Build an LCEL chain: retrieval → prompt → LLM → string output
- invoke(): synchronous query → string answer
- astream(): async query → AsyncIterator[str] token stream
- Return a hardcoded fallback when context is empty (no LLM call)
"""

from __future__ import annotations

import os
from typing import AsyncIterator

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class GenerationError(Exception):
    """Raised when the LLM chain fails during generation."""


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FALLBACK = "I could not find relevant information in the indexed documents."

PROMPT_TEMPLATE = """\
You are a helpful assistant. Answer the question using ONLY the provided context.
Always end your response by citing the source document(s) used.
If the context is empty, respond with exactly:
"I could not find relevant information in the indexed documents."

Context:
{context}

Question: {question}

Answer:"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_docs(docs):
    """Join document page_content with double newlines."""
    return "\n\n".join(doc.page_content for doc in docs)


# ---------------------------------------------------------------------------
# RAGChain
# ---------------------------------------------------------------------------


class RAGChain:
    """LCEL-based RAG chain that retrieves, formats, and generates answers.

    Parameters
    ----------
    reranker:
        A :class:`~rag_porfolio.retrieval.Reranker` instance.
    llm:
        Optional LangChain chat model.  Defaults to
        ``ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), streaming=True)``.
    """

    def __init__(self, reranker, llm=None) -> None:
        load_dotenv()
        self._reranker = reranker

        if llm is None:
            llm = ChatOpenAI(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                streaming=True,
            )
        self._llm = llm

        self._prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

        # LCEL chain — used only when context is non-empty
        self._chain = (
            self._prompt
            | self._llm
            | StrOutputParser()
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def invoke(self, query: str) -> str:
        """Run the RAG chain synchronously.

        Parameters
        ----------
        query:
            The user question.

        Returns
        -------
        str
            The generated answer, or the fallback string if no context was found.

        Raises
        ------
        GenerationError
            If the underlying LLM chain raises an unexpected error.
        """
        docs = self._reranker.retrieve_and_rerank(query)

        if not docs:
            return FALLBACK

        context = _format_docs(docs)

        try:
            return self._chain.invoke({"context": context, "question": query})
        except Exception as exc:
            raise GenerationError(f"LLM generation failed: {exc}") from exc

    async def astream(self, query: str) -> AsyncIterator[str]:
        """Run the RAG chain asynchronously, yielding string tokens.

        Parameters
        ----------
        query:
            The user question.

        Yields
        ------
        str
            Individual string tokens from the LLM, or the full fallback string
            as a single chunk when no context is available.

        Raises
        ------
        GenerationError
            If the underlying LLM stream raises an unexpected error.
        """
        docs = self._reranker.retrieve_and_rerank(query)

        if not docs:
            yield FALLBACK
            return

        context = _format_docs(docs)

        try:
            async for chunk in self._chain.astream({"context": context, "question": query}):
                yield chunk
        except Exception as exc:
            raise GenerationError(f"LLM streaming failed: {exc}") from exc
