"""
Unit tests for app.py — FastAPI endpoints.

Rules:
- All tests marked @pytest.mark.unit
- Zero real API calls: all RAG components are mocked
- Uses httpx.AsyncClient with ASGITransport
- Module-level globals in rag_porfolio.app are patched directly
"""

from __future__ import annotations

import io
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient
from langchain_core.documents import Document


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_upload(filename: str, content: bytes = b"hello world content"):
    """Return a files dict suitable for httpx multipart upload."""
    return {"file": (filename, io.BytesIO(content), "application/octet-stream")}


def _fake_indexed_docs(n: int = 2) -> dict:
    """Return a fake _indexed_docs dict with n entries."""
    return {
        f"doc{i}.txt": {
            "filename": f"doc{i}.txt",
            "chunk_count": i + 1,
            "indexed_at": "2024-01-01T00:00:00+00:00",
        }
        for i in range(n)
    }


async def _make_client():
    """Return an AsyncClient pointed at the FastAPI app."""
    from rag_porfolio.app import app

    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


# ---------------------------------------------------------------------------
# Upload tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_upload_valid_txt_returns_200():
    """Uploading a valid .txt file returns 200 with status/filename/chunk_count."""
    from rag_porfolio.app import app

    mock_store = MagicMock()
    mock_store.is_document_indexed.return_value = False
    mock_store.add_documents.return_value = None

    fake_doc = Document(page_content="hello world", metadata={"source": "test.txt"})
    fake_chunks = [fake_doc, fake_doc]

    with (
        patch("rag_porfolio.app._store", mock_store),
        patch("rag_porfolio.app.load_document", return_value=[fake_doc]),
        patch("rag_porfolio.app.chunk_documents", return_value=fake_chunks),
        patch("rag_porfolio.app._save_upload_file", return_value="uploads/test.txt"),
        patch.dict("rag_porfolio.app._indexed_docs", {}, clear=True),
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/upload", files=make_upload("test.txt", b"some text content")
            )

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "indexed"
    assert body["filename"] == "test.txt"
    assert body["chunk_count"] == 2


@pytest.mark.unit
async def test_upload_valid_pdf_returns_200():
    """Uploading a valid .pdf file returns 200 with correct shape."""
    from rag_porfolio.app import app

    mock_store = MagicMock()
    mock_store.is_document_indexed.return_value = False
    mock_store.add_documents.return_value = None

    fake_doc = Document(page_content="pdf content", metadata={"source": "report.pdf"})
    fake_chunks = [fake_doc]

    with (
        patch("rag_porfolio.app._store", mock_store),
        patch("rag_porfolio.app.load_document", return_value=[fake_doc]),
        patch("rag_porfolio.app.chunk_documents", return_value=fake_chunks),
        patch("rag_porfolio.app._save_upload_file", return_value="uploads/report.pdf"),
        patch.dict("rag_porfolio.app._indexed_docs", {}, clear=True),
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/upload", files=make_upload("report.pdf", b"%PDF-1.4 fake content")
            )

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "indexed"
    assert body["filename"] == "report.pdf"
    assert body["chunk_count"] == 1


@pytest.mark.unit
async def test_upload_valid_docx_returns_200():
    """Uploading a valid .docx file returns 200."""
    from rag_porfolio.app import app

    mock_store = MagicMock()
    mock_store.is_document_indexed.return_value = False
    mock_store.add_documents.return_value = None

    fake_doc = Document(page_content="docx content", metadata={"source": "doc.docx"})
    fake_chunks = [fake_doc, fake_doc, fake_doc]

    with (
        patch("rag_porfolio.app._store", mock_store),
        patch("rag_porfolio.app.load_document", return_value=[fake_doc]),
        patch("rag_porfolio.app.chunk_documents", return_value=fake_chunks),
        patch("rag_porfolio.app._save_upload_file", return_value="uploads/doc.docx"),
        patch.dict("rag_porfolio.app._indexed_docs", {}, clear=True),
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/upload", files=make_upload("doc.docx", b"PK fake docx")
            )

    assert resp.status_code == 200
    assert resp.json()["chunk_count"] == 3


@pytest.mark.unit
async def test_upload_unsupported_format_returns_415():
    """Uploading a .xyz file returns 415 with error=unsupported_format."""
    from rag_porfolio.app import app

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/upload", files=make_upload("data.xyz", b"some data")
        )

    assert resp.status_code == 415
    body = resp.json()
    assert body["error"] == "unsupported_format"
    assert "message" in body


@pytest.mark.unit
async def test_upload_unsupported_format_exe_returns_415():
    """Uploading a .exe file returns 415."""
    from rag_porfolio.app import app

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/upload", files=make_upload("program.exe", b"MZ fake binary")
        )

    assert resp.status_code == 415
    assert resp.json()["error"] == "unsupported_format"


@pytest.mark.unit
async def test_upload_too_large_returns_413():
    """Uploading a file > 10 MB returns 413 with error=file_too_large."""
    from rag_porfolio.app import app

    large_content = b"x" * (10 * 1024 * 1024 + 1)  # 10 MB + 1 byte

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/upload", files=make_upload("big.txt", large_content)
        )

    assert resp.status_code == 413
    body = resp.json()
    assert body["error"] == "file_too_large"
    assert "message" in body


@pytest.mark.unit
async def test_upload_collection_full_returns_400():
    """When 20 documents are already indexed, upload returns 400 with error=collection_full."""
    from rag_porfolio.app import app

    full_docs = _fake_indexed_docs(20)

    with patch.dict("rag_porfolio.app._indexed_docs", full_docs, clear=True):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/upload", files=make_upload("new.txt", b"content")
            )

    assert resp.status_code == 400
    body = resp.json()
    assert body["error"] == "collection_full"
    assert "message" in body


@pytest.mark.unit
async def test_upload_duplicate_returns_409():
    """Uploading a file whose name is already indexed returns 409 with error=duplicate."""
    from rag_porfolio.app import app

    existing = {"report.pdf": {"filename": "report.pdf", "chunk_count": 5, "indexed_at": "2024-01-01T00:00:00+00:00"}}
    mock_store = MagicMock()
    mock_store.is_document_indexed.return_value = True

    with (
        patch("rag_porfolio.app._store", mock_store),
        patch.dict("rag_porfolio.app._indexed_docs", existing, clear=True),
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/upload", files=make_upload("report.pdf", b"content")
            )

    assert resp.status_code == 409
    body = resp.json()
    assert body["error"] == "duplicate"
    assert "report.pdf" in body["message"]


@pytest.mark.unit
async def test_upload_duplicate_message_contains_filename():
    """The 409 duplicate message must reference the actual filename."""
    from rag_porfolio.app import app

    filename = "my_document.txt"
    existing = {filename: {"filename": filename, "chunk_count": 2, "indexed_at": "2024-01-01T00:00:00+00:00"}}
    mock_store = MagicMock()
    mock_store.is_document_indexed.return_value = True

    with (
        patch("rag_porfolio.app._store", mock_store),
        patch.dict("rag_porfolio.app._indexed_docs", existing, clear=True),
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/upload", files=make_upload(filename, b"content")
            )

    assert resp.status_code == 409
    assert filename in resp.json()["message"]


@pytest.mark.unit
async def test_upload_indexes_document_in_indexed_docs():
    """After a successful upload, the filename appears in _indexed_docs."""
    from rag_porfolio.app import app
    import rag_porfolio.app as app_module

    mock_store = MagicMock()
    mock_store.is_document_indexed.return_value = False
    mock_store.add_documents.return_value = None

    fake_doc = Document(page_content="hello", metadata={"source": "new.txt"})
    fake_chunks = [fake_doc]

    with (
        patch("rag_porfolio.app._store", mock_store),
        patch("rag_porfolio.app.load_document", return_value=[fake_doc]),
        patch("rag_porfolio.app.chunk_documents", return_value=fake_chunks),
        patch("rag_porfolio.app._save_upload_file", return_value="uploads/new.txt"),
        patch.dict("rag_porfolio.app._indexed_docs", {}, clear=True),
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/upload", files=make_upload("new.txt", b"content")
            )
        # _indexed_docs should now contain the new file
        assert "new.txt" in app_module._indexed_docs

    assert resp.status_code == 200


@pytest.mark.unit
async def test_upload_store_add_documents_called():
    """store.add_documents must be called with the chunks after upload."""
    from rag_porfolio.app import app

    mock_store = MagicMock()
    mock_store.is_document_indexed.return_value = False

    fake_doc = Document(page_content="chunk", metadata={"source": "file.txt"})
    fake_chunks = [fake_doc, fake_doc]

    with (
        patch("rag_porfolio.app._store", mock_store),
        patch("rag_porfolio.app.load_document", return_value=[fake_doc]),
        patch("rag_porfolio.app.chunk_documents", return_value=fake_chunks),
        patch("rag_porfolio.app._save_upload_file", return_value="uploads/file.txt"),
        patch.dict("rag_porfolio.app._indexed_docs", {}, clear=True),
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            await client.post("/upload", files=make_upload("file.txt", b"data"))

    mock_store.add_documents.assert_called_once_with(fake_chunks)


# ---------------------------------------------------------------------------
# Query tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_query_returns_event_stream():
    """GET /query?q=test returns 200 with text/event-stream content-type."""
    from rag_porfolio.app import app

    async def fake_astream(q):
        yield "Hello"
        yield " world"

    mock_chain = MagicMock()
    mock_chain.astream = fake_astream

    with patch("rag_porfolio.app._chain", mock_chain):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/query", params={"q": "test question"})

    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]


@pytest.mark.unit
async def test_query_response_contains_sse_events():
    """GET /query SSE response body has 'data: ' prefixed tokens."""
    from rag_porfolio.app import app

    async def fake_astream(q):
        yield "Hello"
        yield " world"

    mock_chain = MagicMock()
    mock_chain.astream = fake_astream

    with patch("rag_porfolio.app._chain", mock_chain):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/query", params={"q": "test"})

    body = resp.text
    assert "data: Hello" in body
    assert "data:  world" in body or "data: world" in body


@pytest.mark.unit
async def test_query_response_ends_with_done():
    """GET /query SSE response ends with 'data: [DONE]'."""
    from rag_porfolio.app import app

    async def fake_astream(q):
        yield "Token"

    mock_chain = MagicMock()
    mock_chain.astream = fake_astream

    with patch("rag_porfolio.app._chain", mock_chain):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/query", params={"q": "something"})

    assert "data: [DONE]" in resp.text


@pytest.mark.unit
async def test_query_empty_returns_400():
    """GET /query?q= (empty) returns 400 with error=empty_query."""
    from rag_porfolio.app import app

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.get("/query", params={"q": ""})

    assert resp.status_code == 400
    body = resp.json()
    assert body["error"] == "empty_query"


@pytest.mark.unit
async def test_query_whitespace_only_returns_400():
    """GET /query?q=   (whitespace only) returns 400."""
    from rag_porfolio.app import app

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.get("/query", params={"q": "   "})

    assert resp.status_code == 400
    assert resp.json()["error"] == "empty_query"


@pytest.mark.unit
async def test_query_missing_param_returns_422():
    """GET /query without q param returns 422 (FastAPI validation)."""
    from rag_porfolio.app import app

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.get("/query")

    assert resp.status_code == 422


@pytest.mark.unit
async def test_query_calls_chain_astream_with_query():
    """GET /query calls chain.astream with the exact query string."""
    from rag_porfolio.app import app

    received_queries = []

    async def fake_astream(q):
        received_queries.append(q)
        yield "answer"

    mock_chain = MagicMock()
    mock_chain.astream = fake_astream

    with patch("rag_porfolio.app._chain", mock_chain):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            await client.get("/query", params={"q": "what is RAG?"})

    assert received_queries == ["what is RAG?"]


# ---------------------------------------------------------------------------
# Documents list tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_documents_list_returns_200():
    """GET /documents returns 200."""
    from rag_porfolio.app import app

    with patch.dict("rag_porfolio.app._indexed_docs", _fake_indexed_docs(2), clear=True):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/documents")

    assert resp.status_code == 200


@pytest.mark.unit
async def test_documents_list_returns_array():
    """GET /documents returns a JSON array."""
    from rag_porfolio.app import app

    with patch.dict("rag_porfolio.app._indexed_docs", _fake_indexed_docs(2), clear=True):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/documents")

    assert isinstance(resp.json(), list)


@pytest.mark.unit
async def test_documents_list_has_correct_count():
    """GET /documents returns all indexed documents."""
    from rag_porfolio.app import app

    with patch.dict("rag_porfolio.app._indexed_docs", _fake_indexed_docs(3), clear=True):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/documents")

    assert len(resp.json()) == 3


@pytest.mark.unit
async def test_documents_list_items_have_required_fields():
    """Each document in GET /documents has filename, chunk_count, and indexed_at."""
    from rag_porfolio.app import app

    with patch.dict("rag_porfolio.app._indexed_docs", _fake_indexed_docs(2), clear=True):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/documents")

    for item in resp.json():
        assert "filename" in item
        assert "chunk_count" in item
        assert "indexed_at" in item


@pytest.mark.unit
async def test_documents_list_empty_when_no_docs():
    """GET /documents returns empty list when nothing is indexed."""
    from rag_porfolio.app import app

    with patch.dict("rag_porfolio.app._indexed_docs", {}, clear=True):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/documents")

    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.unit
async def test_documents_list_returns_correct_filenames():
    """GET /documents response contains the expected filenames."""
    from rag_porfolio.app import app

    docs = {
        "alpha.pdf": {"filename": "alpha.pdf", "chunk_count": 3, "indexed_at": "2024-01-01T00:00:00+00:00"},
        "beta.txt": {"filename": "beta.txt", "chunk_count": 1, "indexed_at": "2024-01-02T00:00:00+00:00"},
    }

    with patch.dict("rag_porfolio.app._indexed_docs", docs, clear=True):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/documents")

    filenames = {item["filename"] for item in resp.json()}
    assert "alpha.pdf" in filenames
    assert "beta.txt" in filenames


# ---------------------------------------------------------------------------
# Delete tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_delete_existing_returns_200():
    """DELETE /documents/{filename} for existing file returns 200."""
    from rag_porfolio.app import app

    existing = {"report.pdf": {"filename": "report.pdf", "chunk_count": 5, "indexed_at": "2024-01-01T00:00:00+00:00"}}
    mock_store = MagicMock()
    mock_store.delete_by_source.return_value = None

    with (
        patch("rag_porfolio.app._store", mock_store),
        patch.dict("rag_porfolio.app._indexed_docs", existing, clear=True),
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.delete("/documents/report.pdf")

    assert resp.status_code == 200


@pytest.mark.unit
async def test_delete_existing_returns_status_and_filename():
    """DELETE /documents/{filename} returns status=deleted and correct filename."""
    from rag_porfolio.app import app

    existing = {"report.pdf": {"filename": "report.pdf", "chunk_count": 5, "indexed_at": "2024-01-01T00:00:00+00:00"}}
    mock_store = MagicMock()
    mock_store.delete_by_source.return_value = None

    with (
        patch("rag_porfolio.app._store", mock_store),
        patch.dict("rag_porfolio.app._indexed_docs", existing, clear=True),
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.delete("/documents/report.pdf")

    body = resp.json()
    assert body["status"] == "deleted"
    assert body["filename"] == "report.pdf"


@pytest.mark.unit
async def test_delete_calls_store_delete_by_source():
    """DELETE /documents/{filename} calls store.delete_by_source with the filename."""
    from rag_porfolio.app import app

    existing = {"notes.txt": {"filename": "notes.txt", "chunk_count": 2, "indexed_at": "2024-01-01T00:00:00+00:00"}}
    mock_store = MagicMock()
    mock_store.delete_by_source.return_value = None

    with (
        patch("rag_porfolio.app._store", mock_store),
        patch.dict("rag_porfolio.app._indexed_docs", existing, clear=True),
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            await client.delete("/documents/notes.txt")

    mock_store.delete_by_source.assert_called_once_with("notes.txt")


@pytest.mark.unit
async def test_delete_removes_from_indexed_docs():
    """After DELETE, the file is removed from _indexed_docs."""
    from rag_porfolio.app import app
    import rag_porfolio.app as app_module

    existing = {"gone.pdf": {"filename": "gone.pdf", "chunk_count": 1, "indexed_at": "2024-01-01T00:00:00+00:00"}}
    mock_store = MagicMock()
    mock_store.delete_by_source.return_value = None

    with (
        patch("rag_porfolio.app._store", mock_store),
        patch.dict("rag_porfolio.app._indexed_docs", existing, clear=True),
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            await client.delete("/documents/gone.pdf")

        assert "gone.pdf" not in app_module._indexed_docs


@pytest.mark.unit
async def test_delete_nonexistent_returns_404():
    """DELETE /documents/{filename} for missing file returns 404 with error=not_found."""
    from rag_porfolio.app import app

    with patch.dict("rag_porfolio.app._indexed_docs", {}, clear=True):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.delete("/documents/ghost.pdf")

    assert resp.status_code == 404
    body = resp.json()
    assert body["error"] == "not_found"
    assert "ghost.pdf" in body["message"]


@pytest.mark.unit
async def test_delete_nonexistent_message_contains_filename():
    """The 404 message for delete must reference the missing filename."""
    from rag_porfolio.app import app

    with patch.dict("rag_porfolio.app._indexed_docs", {}, clear=True):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.delete("/documents/missing_file.pdf")

    assert "missing_file.pdf" in resp.json()["message"]


@pytest.mark.unit
async def test_delete_does_not_call_store_when_not_found():
    """store.delete_by_source must NOT be called when file is not in _indexed_docs."""
    from rag_porfolio.app import app

    mock_store = MagicMock()

    with (
        patch("rag_porfolio.app._store", mock_store),
        patch.dict("rag_porfolio.app._indexed_docs", {}, clear=True),
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            await client.delete("/documents/nope.txt")

    mock_store.delete_by_source.assert_not_called()


# ---------------------------------------------------------------------------
# Edge / additional coverage
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_upload_response_body_has_all_required_keys():
    """Successful upload response contains status, filename, and chunk_count."""
    from rag_porfolio.app import app

    mock_store = MagicMock()
    mock_store.is_document_indexed.return_value = False
    mock_store.add_documents.return_value = None

    fake_doc = Document(page_content="data", metadata={"source": "file.txt"})

    with (
        patch("rag_porfolio.app._store", mock_store),
        patch("rag_porfolio.app.load_document", return_value=[fake_doc]),
        patch("rag_porfolio.app.chunk_documents", return_value=[fake_doc]),
        patch("rag_porfolio.app._save_upload_file", return_value="uploads/file.txt"),
        patch.dict("rag_porfolio.app._indexed_docs", {}, clear=True),
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post("/upload", files=make_upload("file.txt", b"data"))

    body = resp.json()
    assert "status" in body
    assert "filename" in body
    assert "chunk_count" in body


@pytest.mark.unit
async def test_indexed_docs_entry_has_indexed_at_timestamp():
    """After upload, the _indexed_docs entry includes an ISO8601 indexed_at timestamp."""
    from rag_porfolio.app import app
    import rag_porfolio.app as app_module

    mock_store = MagicMock()
    mock_store.is_document_indexed.return_value = False

    fake_doc = Document(page_content="ts test", metadata={"source": "ts.txt"})

    with (
        patch("rag_porfolio.app._store", mock_store),
        patch("rag_porfolio.app.load_document", return_value=[fake_doc]),
        patch("rag_porfolio.app.chunk_documents", return_value=[fake_doc]),
        patch("rag_porfolio.app._save_upload_file", return_value="uploads/ts.txt"),
        patch.dict("rag_porfolio.app._indexed_docs", {}, clear=True),
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            await client.post("/upload", files=make_upload("ts.txt", b"timestamp test"))

        entry = app_module._indexed_docs.get("ts.txt")

    assert entry is not None
    assert "indexed_at" in entry
    # Should be parseable as a datetime
    datetime.fromisoformat(entry["indexed_at"])


@pytest.mark.unit
async def test_upload_exactly_at_limit_19_docs_succeeds():
    """Uploading when 19 docs exist (one below limit) should succeed, not return 400."""
    from rag_porfolio.app import app

    docs_19 = _fake_indexed_docs(19)
    mock_store = MagicMock()
    mock_store.is_document_indexed.return_value = False

    fake_doc = Document(page_content="doc20", metadata={"source": "new20.txt"})

    with (
        patch("rag_porfolio.app._store", mock_store),
        patch("rag_porfolio.app.load_document", return_value=[fake_doc]),
        patch("rag_porfolio.app.chunk_documents", return_value=[fake_doc]),
        patch("rag_porfolio.app._save_upload_file", return_value="uploads/new20.txt"),
        patch.dict("rag_porfolio.app._indexed_docs", docs_19, clear=True),
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/upload", files=make_upload("new20.txt", b"twentieth document")
            )

    assert resp.status_code == 200


@pytest.mark.unit
async def test_upload_exactly_20_docs_returns_400():
    """Uploading when exactly 20 docs exist returns 400 collection_full."""
    from rag_porfolio.app import app

    full_docs = _fake_indexed_docs(20)

    with patch.dict("rag_porfolio.app._indexed_docs", full_docs, clear=True):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/upload", files=make_upload("overflow.txt", b"too many")
            )

    assert resp.status_code == 400
    assert resp.json()["error"] == "collection_full"


@pytest.mark.unit
async def test_query_sse_event_format():
    """Each SSE event must match the format 'data: {token}\\n\\n'."""
    from rag_porfolio.app import app

    async def fake_astream(q):
        yield "Alpha"
        yield "Beta"

    mock_chain = MagicMock()
    mock_chain.astream = fake_astream

    with patch("rag_porfolio.app._chain", mock_chain):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/query", params={"q": "test"})

    # Each event: "data: {token}\n\n"
    events = [line for line in resp.text.split("\n\n") if line.strip()]
    assert any(e.startswith("data: ") for e in events)
