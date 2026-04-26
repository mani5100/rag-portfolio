"""
FastAPI application for the RAG portfolio.

Endpoints:
- POST /upload         — ingest a document (PDF, DOCX, TXT)
- GET  /query          — stream an answer via SSE
- GET  /documents      — list all indexed documents
- DELETE /documents/{filename} — remove a document
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from rag_porfolio.chain import RAGChain
from rag_porfolio.ingestion import chunk_documents, load_document
from rag_porfolio.retrieval import Reranker, Retriever
from rag_porfolio.store import EmbeddingService, VectorStore

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
MAX_DOCUMENTS = 20
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}
UPLOADS_DIR = "uploads"

# ---------------------------------------------------------------------------
# Module-level state (singletons + document registry)
# ---------------------------------------------------------------------------

_store: VectorStore | None = None
_retriever: Retriever | None = None
_reranker: Reranker | None = None
_chain: RAGChain | None = None

# Maps filename → {"filename": str, "chunk_count": int, "indexed_at": str}
_indexed_docs: dict[str, dict] = {}

# ---------------------------------------------------------------------------
# Lifespan — initialize singletons on startup
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize all RAG components on startup."""
    global _store, _retriever, _reranker, _chain

    load_dotenv()

    embedding = EmbeddingService()
    _store = VectorStore(embedding=embedding)
    _retriever = Retriever(vector_store=_store)
    _reranker = Reranker(retriever=_retriever)
    _chain = RAGChain(reranker=_reranker)

    # Ensure uploads directory exists
    Path(UPLOADS_DIR).mkdir(parents=True, exist_ok=True)

    yield  # application runs here

    # Teardown (nothing needed for now)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="RAG Portfolio API", lifespan=lifespan)

app.mount("/static", StaticFiles(directory="src/rag_porfolio/static"), name="static")
templates = Jinja2Templates(directory="src/rag_porfolio/templates")


@app.get("/", include_in_schema=False)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


async def _save_upload_file(upload_file: UploadFile, dest_dir: str = UPLOADS_DIR) -> str:
    """Write the uploaded file to disk and return the file path."""
    Path(dest_dir).mkdir(parents=True, exist_ok=True)
    dest = os.path.join(dest_dir, upload_file.filename)
    content = await upload_file.read()
    with open(dest, "wb") as f:
        f.write(content)
    # Reset the stream position so callers can read size again if needed
    await upload_file.seek(0)
    return dest


# ---------------------------------------------------------------------------
# POST /upload
# ---------------------------------------------------------------------------


@app.post("/upload")
async def upload_document(file: UploadFile):
    """Upload and index a document.

    Validations (in order):
    1. Extension must be .pdf, .docx, or .txt  → 415
    2. Content length must be ≤ 10 MB           → 413
    3. Collection must have < 20 documents      → 400
    4. Filename must not already be indexed     → 409
    """
    filename = file.filename or ""

    # --- 1. Extension check ---
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return JSONResponse(
            status_code=415,
            content={
                "error": "unsupported_format",
                "message": "Only PDF, DOCX, and TXT files are accepted.",
            },
        )

    # --- 2. Size check (read content into memory) ---
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        return JSONResponse(
            status_code=413,
            content={
                "error": "file_too_large",
                "message": "Maximum file size is 10MB.",
            },
        )
    # Seek back so downstream reads work
    await file.seek(0)

    # --- 3. Collection capacity check ---
    if len(_indexed_docs) >= MAX_DOCUMENTS:
        return JSONResponse(
            status_code=400,
            content={
                "error": "collection_full",
                "message": "Maximum 20 documents reached.",
            },
        )

    # --- 4. Duplicate check ---
    if filename in _indexed_docs or _store.is_document_indexed(filename):
        return JSONResponse(
            status_code=409,
            content={
                "error": "duplicate",
                "message": f"A document named '{filename}' already exists.",
            },
        )

    # --- Save file to disk ---
    filepath = await _save_upload_file(file)

    # --- Ingest pipeline ---
    docs = load_document(filepath)
    chunks = chunk_documents(docs)
    _store.add_documents(chunks)

    # --- Track in registry ---
    _indexed_docs[filename] = {
        "filename": filename,
        "chunk_count": len(chunks),
        "indexed_at": datetime.now(tz=timezone.utc).isoformat(),
    }

    return {
        "status": "indexed",
        "filename": filename,
        "chunk_count": len(chunks),
    }


# ---------------------------------------------------------------------------
# GET /query
# ---------------------------------------------------------------------------


@app.get("/query")
async def query(q: str):
    """Stream an answer to *q* as Server-Sent Events."""
    if not q or not q.strip():
        return JSONResponse(
            status_code=400,
            content={"error": "empty_query"},
        )

    async def event_stream() -> AsyncIterator[str]:
        async for token in _chain.astream(q):
            yield f"data: {token}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# GET /documents
# ---------------------------------------------------------------------------


@app.get("/documents")
async def list_documents():
    """Return a list of all indexed documents."""
    return list(_indexed_docs.values())


# ---------------------------------------------------------------------------
# DELETE /documents/{filename}
# ---------------------------------------------------------------------------


@app.delete("/documents/{filename}")
async def delete_document(filename: str):
    """Delete a document from the index and vector store."""
    if filename not in _indexed_docs:
        return JSONResponse(
            status_code=404,
            content={
                "error": "not_found",
                "message": f"Document '{filename}' not found.",
            },
        )

    _store.delete_by_source(filename)
    del _indexed_docs[filename]

    return {"status": "deleted", "filename": filename}
