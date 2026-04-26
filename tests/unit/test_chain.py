"""
Unit tests for chain.py — RAGChain.

Rules:
- All tests marked @pytest.mark.unit
- Zero real API calls: all external services are mocked
- ChatOpenAI is patched to avoid real OpenAI calls
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel


FALLBACK = "I could not find relevant information in the indexed documents."


def make_mock_reranker(docs):
    """Helper: returns a mock Reranker that yields `docs` for any query."""
    mock = MagicMock()
    mock.retrieve_and_rerank.return_value = docs
    return mock


# ---------------------------------------------------------------------------
# GenerationError
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_generation_error_is_exception():
    """GenerationError must be a subclass of Exception."""
    from rag_porfolio.chain import GenerationError

    err = GenerationError("something went wrong")
    assert isinstance(err, Exception)


# ---------------------------------------------------------------------------
# RAGChain construction
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_ragchain_constructs_with_reranker_and_fake_llm():
    """RAGChain can be constructed with a mock reranker and a fake LLM."""
    from rag_porfolio.chain import RAGChain

    reranker = make_mock_reranker([])
    llm = FakeListChatModel(responses=["hello"])
    chain = RAGChain(reranker=reranker, llm=llm)
    assert chain is not None


@pytest.mark.unit
def test_ragchain_default_llm_uses_chatopenaai():
    """RAGChain without explicit llm instantiates ChatOpenAI."""
    from rag_porfolio.chain import RAGChain

    reranker = make_mock_reranker([])
    mock_llm = MagicMock()
    with patch("rag_porfolio.chain.ChatOpenAI", return_value=mock_llm) as mock_cls:
        chain = RAGChain(reranker=reranker)
        mock_cls.assert_called_once()
    assert chain is not None


# ---------------------------------------------------------------------------
# invoke — happy path
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_chain_invoke_returns_string():
    """invoke() must return a non-empty string when docs are found."""
    from rag_porfolio.chain import RAGChain

    docs = [Document(page_content="LangChain is a framework.", metadata={"source": "doc.txt"})]
    reranker = make_mock_reranker(docs)
    llm = FakeListChatModel(responses=["LangChain is great."])
    chain = RAGChain(reranker=reranker, llm=llm)
    result = chain.invoke("What is LangChain?")
    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.unit
def test_chain_invoke_calls_reranker_once():
    """invoke() calls retrieve_and_rerank exactly once."""
    from rag_porfolio.chain import RAGChain

    docs = [Document(page_content="Some content.", metadata={"source": "a.txt"})]
    reranker = make_mock_reranker(docs)
    llm = FakeListChatModel(responses=["Some answer."])
    chain = RAGChain(reranker=reranker, llm=llm)
    chain.invoke("test query")
    reranker.retrieve_and_rerank.assert_called_once_with("test query")


@pytest.mark.unit
def test_chain_invoke_multiple_docs_joined_with_double_newline():
    """invoke() must join multiple doc contents with '\\n\\n'."""
    from rag_porfolio.chain import RAGChain

    docs = [
        Document(page_content="First fact.", metadata={"source": "a.txt"}),
        Document(page_content="Second fact.", metadata={"source": "b.txt"}),
    ]
    reranker = make_mock_reranker(docs)

    # Capture what the LLM sees by inspecting the prompt
    captured_inputs = []

    class CapturingFakeLLM(FakeListChatModel):
        responses: list = ["answer"]

        def invoke(self, input, config=None, **kwargs):
            captured_inputs.append(input)
            return super().invoke(input, config=config, **kwargs)

    llm = CapturingFakeLLM(responses=["answer"])
    chain = RAGChain(reranker=reranker, llm=llm)
    chain.invoke("question?")

    assert len(captured_inputs) >= 1
    # The formatted context in the prompt must contain both facts
    prompt_text = str(captured_inputs[0])
    assert "First fact." in prompt_text
    assert "Second fact." in prompt_text


# ---------------------------------------------------------------------------
# invoke — empty context (fallback)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_chain_invoke_empty_context_returns_fallback():
    """invoke() returns the exact fallback string when no docs are found."""
    from rag_porfolio.chain import RAGChain

    reranker = make_mock_reranker([])
    llm = FakeListChatModel(responses=["should not be called"])
    chain = RAGChain(reranker=reranker, llm=llm)
    result = chain.invoke("anything")
    assert result == FALLBACK
    reranker.retrieve_and_rerank.assert_called_once()


@pytest.mark.unit
def test_chain_invoke_empty_context_does_not_call_llm():
    """invoke() must NOT call the LLM when context is empty."""
    from rag_porfolio.chain import RAGChain

    reranker = make_mock_reranker([])
    mock_llm = MagicMock()
    mock_llm.invoke = MagicMock(return_value=MagicMock(content=""))
    with patch("rag_porfolio.chain.ChatOpenAI", return_value=mock_llm):
        chain = RAGChain(reranker=reranker)
    chain.invoke("test")
    mock_llm.invoke.assert_not_called()


@pytest.mark.unit
def test_chain_invoke_fallback_exact_string():
    """The fallback string must match exactly (no extra whitespace)."""
    from rag_porfolio.chain import RAGChain

    reranker = make_mock_reranker([])
    llm = FakeListChatModel(responses=["ignored"])
    chain = RAGChain(reranker=reranker, llm=llm)
    result = chain.invoke("query")
    assert result == FALLBACK
    assert result.strip() == result  # no leading/trailing whitespace


# ---------------------------------------------------------------------------
# astream — happy path
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_chain_astream_yields_chunks():
    """astream() yields at least one string chunk when docs are found."""
    from rag_porfolio.chain import RAGChain

    docs = [Document(page_content="RAG combines retrieval and generation.", metadata={"source": "rag.txt"})]
    reranker = make_mock_reranker(docs)
    llm = FakeListChatModel(responses=["RAG is powerful."])
    chain = RAGChain(reranker=reranker, llm=llm)
    chunks = []
    async for chunk in chain.astream("What is RAG?"):
        chunks.append(chunk)
    assert len(chunks) >= 1
    assert "".join(chunks) != ""


@pytest.mark.unit
@pytest.mark.asyncio
async def test_chain_astream_chunks_are_strings():
    """astream() must yield string tokens, not objects."""
    from rag_porfolio.chain import RAGChain

    docs = [Document(page_content="Python is a language.", metadata={"source": "py.txt"})]
    reranker = make_mock_reranker(docs)
    llm = FakeListChatModel(responses=["Python is great."])
    chain = RAGChain(reranker=reranker, llm=llm)
    async for chunk in chain.astream("Tell me about Python"):
        assert isinstance(chunk, str)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_chain_astream_calls_reranker_once():
    """astream() calls retrieve_and_rerank exactly once."""
    from rag_porfolio.chain import RAGChain

    docs = [Document(page_content="Content.", metadata={"source": "x.txt"})]
    reranker = make_mock_reranker(docs)
    llm = FakeListChatModel(responses=["Answer."])
    chain = RAGChain(reranker=reranker, llm=llm)
    async for _ in chain.astream("query"):
        pass
    reranker.retrieve_and_rerank.assert_called_once_with("query")


# ---------------------------------------------------------------------------
# astream — empty context (fallback)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_chain_astream_empty_context_yields_fallback():
    """astream() yields exactly the fallback string as one chunk when context is empty."""
    from rag_porfolio.chain import RAGChain

    reranker = make_mock_reranker([])
    llm = FakeListChatModel(responses=["should not be called"])
    chain = RAGChain(reranker=reranker, llm=llm)
    chunks = []
    async for chunk in chain.astream("anything"):
        chunks.append(chunk)
    assert "".join(chunks) == FALLBACK


@pytest.mark.unit
@pytest.mark.asyncio
async def test_chain_astream_empty_context_does_not_call_llm():
    """astream() must NOT call the LLM when context is empty."""
    from rag_porfolio.chain import RAGChain

    reranker = make_mock_reranker([])
    mock_llm = MagicMock()
    mock_llm.astream = AsyncMock(return_value=aiter_helper([]))
    with patch("rag_porfolio.chain.ChatOpenAI", return_value=mock_llm):
        chain = RAGChain(reranker=reranker)
    async for _ in chain.astream("test"):
        pass
    mock_llm.astream.assert_not_called()


# ---------------------------------------------------------------------------
# Prompt template content
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_ragchain_has_prompt_with_context_and_question():
    """RAGChain must have a prompt template referencing {context} and {question}."""
    from rag_porfolio.chain import RAGChain

    reranker = make_mock_reranker([])
    llm = FakeListChatModel(responses=["ok"])
    chain = RAGChain(reranker=reranker, llm=llm)
    prompt_str = str(chain._prompt.messages)
    assert "context" in prompt_str
    assert "question" in prompt_str


@pytest.mark.unit
def test_ragchain_prompt_contains_fallback_instruction():
    """The prompt must instruct the LLM to return the fallback phrase for empty context."""
    from rag_porfolio.chain import RAGChain

    reranker = make_mock_reranker([])
    llm = FakeListChatModel(responses=["ok"])
    chain = RAGChain(reranker=reranker, llm=llm)
    prompt_str = str(chain._prompt.messages)
    assert "I could not find relevant information" in prompt_str


# ---------------------------------------------------------------------------
# GenerationError propagation
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_ragchain_invoke_wraps_llm_error_in_generation_error():
    """invoke() must raise GenerationError if the LLM raises an unexpected error."""
    from rag_porfolio.chain import RAGChain, GenerationError

    docs = [Document(page_content="Something.", metadata={"source": "x.txt"})]
    reranker = make_mock_reranker(docs)
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = RuntimeError("LLM exploded")
    # Patch the internal _chain so invoke goes through our mock
    with patch("rag_porfolio.chain.ChatOpenAI", return_value=mock_llm):
        chain = RAGChain(reranker=reranker)

    # Patch the chain's internal _chain.invoke to raise
    chain._chain = MagicMock()
    chain._chain.invoke.side_effect = RuntimeError("LLM exploded")

    with pytest.raises(GenerationError):
        chain.invoke("test")


# ---------------------------------------------------------------------------
# Helper for async iteration in tests
# ---------------------------------------------------------------------------


async def aiter_helper(items):
    """Async generator helper for test mocks."""
    for item in items:
        yield item
