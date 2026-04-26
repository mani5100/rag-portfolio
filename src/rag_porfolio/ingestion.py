"""
Document ingestion pipeline.

Responsibilities:
- Load documents from .pdf, .docx, and .txt files
- Enforce a 10 MB file-size limit
- Chunk documents using RecursiveCharacterTextSplitter
"""

from __future__ import annotations

import os
from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class UnsupportedFormatError(Exception):
    """Raised when the file extension is not supported."""


class FileTooLargeError(Exception):
    """Raised when the file exceeds the 10 MB size limit."""


class IngestionError(Exception):
    """Raised when an underlying loader fails during document loading."""


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB in bytes

CHUNK_SIZE = 512
CHUNK_OVERLAP = 64

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_document(file_path: str) -> list[Document]:
    """Load a document from *file_path* and return a list of Document objects.

    Steps:
    1. Check file size — raises FileTooLargeError if > 10 MB.
    2. Dispatch to the correct loader by file extension.
    3. Set metadata["source"] to the basename of the file on every document.
    4. Wrap loader errors in IngestionError.

    Supported extensions: .pdf, .docx, .txt
    """
    # --- Size check ---
    size = os.path.getsize(file_path)
    if size > MAX_FILE_SIZE:
        raise FileTooLargeError(
            f"File '{file_path}' is {size} bytes, which exceeds the 10 MB limit."
        )

    # --- Extension dispatch ---
    ext = Path(file_path).suffix.lower()

    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".docx":
        loader = UnstructuredWordDocumentLoader(file_path, mode="elements")
    elif ext == ".txt":
        loader = TextLoader(file_path)
    else:
        raise UnsupportedFormatError(
            f"File extension '{ext}' is not supported. "
            "Supported formats: .pdf, .docx, .txt"
        )

    # --- Load documents ---
    try:
        documents = loader.load()
    except Exception as exc:
        raise IngestionError(
            f"Failed to load document '{file_path}': {exc}"
        ) from exc

    # --- Attach source metadata (basename only) ---
    basename = os.path.basename(file_path)
    for doc in documents:
        doc.metadata["source"] = basename

    return documents


def chunk_documents(documents: list[Document]) -> list[Document]:
    """Split *documents* into smaller chunks using RecursiveCharacterTextSplitter.

    Parameters
    ----------
    documents:
        List of Document objects to chunk.

    Returns
    -------
    list[Document]
        Chunked documents.  Each chunk inherits the parent's metadata.
        Returns an empty list when *documents* is empty.
    """
    if not documents:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    return splitter.split_documents(documents)
