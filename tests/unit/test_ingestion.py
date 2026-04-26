import pytest
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document

from rag_porfolio.ingestion import (
    load_document,
    chunk_documents,
    UnsupportedFormatError,
    FileTooLargeError,
    IngestionError,
)


# ---------------------------------------------------------------------------
# Loader tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_txt_loader_returns_documents(tmp_path):
    """TextLoader should return non-empty list of Documents with correct source metadata."""
    txt_file = tmp_path / "test.txt"
    txt_file.write_text("Hello, this is a test document with some content.", encoding="utf-8")

    docs = load_document(str(txt_file))

    assert isinstance(docs, list)
    assert len(docs) > 0
    for doc in docs:
        assert isinstance(doc, Document)
        assert doc.page_content.strip() != ""
        assert doc.metadata.get("source") == "test.txt"


@pytest.mark.unit
def test_unsupported_extension_raises(tmp_path):
    """Files with unsupported extensions should raise UnsupportedFormatError."""
    xyz_file = tmp_path / "document.xyz"
    xyz_file.write_text("some content")

    with pytest.raises(UnsupportedFormatError):
        load_document(str(xyz_file))


@pytest.mark.unit
def test_file_too_large_raises(tmp_path, monkeypatch):
    """Files larger than 10MB should raise FileTooLargeError."""
    txt_file = tmp_path / "big.txt"
    txt_file.write_text("small content")

    monkeypatch.setattr(os.path, "getsize", lambda path: 11 * 1024 * 1024)

    with pytest.raises(FileTooLargeError):
        load_document(str(txt_file))


@pytest.mark.unit
def test_pdf_loader_dispatches_correctly(tmp_path, monkeypatch):
    """PDF files should dispatch to PyPDFLoader with extraction_mode='page'."""
    pdf_file = tmp_path / "sample.pdf"
    pdf_file.write_bytes(b"%PDF-1.4 fake pdf content")

    fake_doc = Document(page_content="PDF content here", metadata={})
    mock_loader_instance = MagicMock()
    mock_loader_instance.load.return_value = [fake_doc]
    mock_loader_class = MagicMock(return_value=mock_loader_instance)

    monkeypatch.setattr("rag_porfolio.ingestion.PyPDFLoader", mock_loader_class)

    docs = load_document(str(pdf_file))

    mock_loader_class.assert_called_once_with(str(pdf_file), extraction_mode="page")
    assert len(docs) == 1
    assert docs[0].metadata["source"] == "sample.pdf"


@pytest.mark.unit
def test_docx_loader_dispatches_correctly(tmp_path, monkeypatch):
    """DOCX files should dispatch to UnstructuredWordDocumentLoader with mode='elements'."""
    docx_file = tmp_path / "sample.docx"
    docx_file.write_bytes(b"PK fake docx content")

    fake_doc = Document(page_content="DOCX content here", metadata={})
    mock_loader_instance = MagicMock()
    mock_loader_instance.load.return_value = [fake_doc]
    mock_loader_class = MagicMock(return_value=mock_loader_instance)

    monkeypatch.setattr("rag_porfolio.ingestion.UnstructuredWordDocumentLoader", mock_loader_class)

    docs = load_document(str(docx_file))

    mock_loader_class.assert_called_once_with(str(docx_file), mode="elements")
    assert len(docs) == 1
    assert docs[0].metadata["source"] == "sample.docx"


@pytest.mark.unit
def test_loader_error_wrapped_in_ingestion_error(tmp_path, monkeypatch):
    """Loader exceptions should be wrapped in IngestionError."""
    txt_file = tmp_path / "broken.txt"
    txt_file.write_text("content")

    mock_loader_instance = MagicMock()
    mock_loader_instance.load.side_effect = RuntimeError("disk read failed")
    mock_loader_class = MagicMock(return_value=mock_loader_instance)

    monkeypatch.setattr("rag_porfolio.ingestion.TextLoader", mock_loader_class)

    with pytest.raises(IngestionError):
        load_document(str(txt_file))


@pytest.mark.unit
def test_source_metadata_is_basename_not_full_path(tmp_path, monkeypatch):
    """metadata['source'] must be the basename, not the full path."""
    txt_file = tmp_path / "myfile.txt"
    txt_file.write_text("Some content for testing.")

    fake_doc = Document(page_content="Some content for testing.", metadata={})
    mock_loader_instance = MagicMock()
    mock_loader_instance.load.return_value = [fake_doc]
    mock_loader_class = MagicMock(return_value=mock_loader_instance)

    monkeypatch.setattr("rag_porfolio.ingestion.TextLoader", mock_loader_class)

    docs = load_document(str(txt_file))

    for doc in docs:
        assert doc.metadata["source"] == "myfile.txt"
        assert "/" not in doc.metadata["source"]
        assert "\\" not in doc.metadata["source"]


# ---------------------------------------------------------------------------
# Chunker tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_chunk_documents_returns_chunks():
    """Long documents must be split into multiple chunks."""
    docs = [Document(page_content="word " * 300, metadata={"source": "doc.txt"})]
    chunks = chunk_documents(docs)
    assert len(chunks) > 1


@pytest.mark.unit
def test_chunk_max_size():
    """No chunk should exceed 1500 characters (generous bound above chunk_size=512)."""
    docs = [Document(page_content="word " * 500, metadata={"source": "doc.txt"})]
    chunks = chunk_documents(docs)
    assert all(len(c.page_content) <= 1500 for c in chunks)


@pytest.mark.unit
def test_chunk_inherits_metadata():
    """Every chunk must carry the parent's metadata, including 'source'."""
    docs = [Document(page_content="hello world " * 100, metadata={"source": "myfile.pdf"})]
    chunks = chunk_documents(docs)
    assert all(c.metadata.get("source") == "myfile.pdf" for c in chunks)


@pytest.mark.unit
def test_chunk_empty_input():
    """Empty input list should return empty list without error."""
    assert chunk_documents([]) == []


@pytest.mark.unit
def test_chunk_unicode_does_not_raise():
    """Unicode and emoji content must not raise exceptions."""
    docs = [Document(page_content="こんにちは 🌍 مرحبا " * 50, metadata={"source": "uni.txt"})]
    result = chunk_documents(docs)
    assert isinstance(result, list)


@pytest.mark.unit
def test_chunk_overlap_shares_content():
    """Adjacent chunks should share overlapping content due to chunk_overlap=64."""
    content = "The quick brown fox jumps over the lazy dog. " * 30
    docs = [Document(page_content=content, metadata={"source": "test.txt"})]
    chunks = chunk_documents(docs)
    if len(chunks) >= 2:
        end_of_first = chunks[0].page_content[-30:]
        assert end_of_first in chunks[1].page_content
