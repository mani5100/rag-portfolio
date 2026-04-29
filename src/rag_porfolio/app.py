"""
FastAPI application for the RAG portfolio.

Endpoints:
- POST /upload         — ingest a document (PDF, DOCX, TXT)
- GET  /query          — stream an answer via SSE
- GET  /documents      — list all indexed documents
- DELETE /documents/{filename} — remove a document

Sessions are browser-tab-scoped (X-Session-ID header). All state is
in-memory; nothing is written to disk. Sessions expire after 30 min of
inactivity.
"""

from __future__ import annotations

import asyncio
import tempfile
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator

import chromadb
from dotenv import load_dotenv
from fastapi import FastAPI, Header, Request, UploadFile
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
SESSION_TTL = 1800  # 30 minutes

# ---------------------------------------------------------------------------
# Per-session state
# ---------------------------------------------------------------------------


@dataclass
class Session:
    store: VectorStore
    retriever: Retriever
    reranker: Reranker
    chain: RAGChain
    indexed_docs: dict = field(default_factory=dict)
    last_accessed: float = field(default_factory=time.monotonic)


_embedding: EmbeddingService | None = None
_sessions: dict[str, Session] = {}


def _create_session() -> Session:
    client = chromadb.EphemeralClient()
    store = VectorStore(embedding=_embedding, chroma_client=client)
    retriever = Retriever(vector_store=store)
    reranker = Reranker(retriever=retriever)
    chain = RAGChain(reranker=reranker)
    return Session(store=store, retriever=retriever, reranker=reranker, chain=chain)


def _get_session(session_id: str) -> Session:
    if session_id not in _sessions:
        _sessions[session_id] = _create_session()
    sess = _sessions[session_id]
    sess.last_accessed = time.monotonic()
    return sess


async def _cleanup_sessions() -> None:
    while True:
        await asyncio.sleep(300)  # check every 5 min
        cutoff = time.monotonic() - SESSION_TTL
        expired = [sid for sid, s in list(_sessions.items()) if s.last_accessed < cutoff]
        for sid in expired:
            del _sessions[sid]


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _embedding
    load_dotenv()
    _embedding = EmbeddingService()
    cleanup_task = asyncio.create_task(_cleanup_sessions())
    yield
    cleanup_task.cancel()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="RAG Portfolio API", lifespan=lifespan)

app.mount("/static", StaticFiles(directory="src/rag_porfolio/static"), name="static")
templates = Jinja2Templates(directory="src/rag_porfolio/templates")


@app.get("/", include_in_schema=False)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


# ---------------------------------------------------------------------------
# POST /upload
# ---------------------------------------------------------------------------


@app.post("/upload")
async def upload_document(
    file: UploadFile,
    session_id: str = Header(alias="X-Session-ID", default=""),
):
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

    # --- 2. Size check ---
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        return JSONResponse(
            status_code=413,
            content={
                "error": "file_too_large",
                "message": "Maximum file size is 10MB.",
            },
        )

    sess = _get_session(session_id)

    # --- 3. Collection capacity check ---
    if len(sess.indexed_docs) >= MAX_DOCUMENTS:
        return JSONResponse(
            status_code=400,
            content={
                "error": "collection_full",
                "message": "Maximum 20 documents reached.",
            },
        )

    # --- 4. Duplicate check ---
    if filename in sess.indexed_docs:
        return JSONResponse(
            status_code=409,
            content={
                "error": "duplicate",
                "message": f"A document named '{filename}' already exists.",
            },
        )

    # --- Write to temp file, ingest, then delete ---
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    try:
        docs = load_document(tmp_path)
        for doc in docs:
            doc.metadata["source"] = filename
        chunks = chunk_documents(docs)
        sess.store.add_documents(chunks)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    sess.indexed_docs[filename] = {
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
async def query(
    q: str,
    session_id: str = Header(alias="X-Session-ID", default=""),
):
    """Stream an answer to *q* as Server-Sent Events."""
    if not q or not q.strip():
        return JSONResponse(
            status_code=400,
            content={"error": "empty_query"},
        )

    sess = _get_session(session_id)

    async def event_stream() -> AsyncIterator[str]:
        async for token in sess.chain.astream(q):
            yield f"data: {token}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# GET /documents
# ---------------------------------------------------------------------------


@app.get("/documents")
async def list_documents(
    session_id: str = Header(alias="X-Session-ID", default=""),
):
    """Return a list of all indexed documents."""
    sess = _get_session(session_id)
    return list(sess.indexed_docs.values())


# ---------------------------------------------------------------------------
# DELETE /documents/{filename}
# ---------------------------------------------------------------------------


@app.delete("/documents/{filename}")
async def delete_document(
    filename: str,
    session_id: str = Header(alias="X-Session-ID", default=""),
):
    """Delete a document from the index and vector store."""
    sess = _get_session(session_id)

    if filename not in sess.indexed_docs:
        return JSONResponse(
            status_code=404,
            content={
                "error": "not_found",
                "message": f"Document '{filename}' not found.",
            },
        )

    sess.store.delete_by_source(filename)
    del sess.indexed_docs[filename]

    return {"status": "deleted", "filename": filename}
