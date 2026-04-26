# RAG Portfolio

A production-style Retrieval-Augmented Generation (RAG) application. Upload PDF, DOCX, or TXT documents and ask questions — answers are streamed in real time, grounded exclusively in your uploaded content.

---

## Architecture

```
frontend (Next.js)
    └── POST /upload        → ingest & embed document chunks
    └── GET  /query         → stream answer tokens via SSE
    └── GET  /documents     → list indexed documents
    └── DELETE /documents/{filename}

backend (FastAPI)
    └── ingestion.py        → load + chunk documents
    └── store.py            → ChromaDB vector store + OpenAI embeddings
    └── retrieval.py        → vector search + FlashRank reranker
    └── chain.py            → LCEL RAG chain (retrieve → prompt → GPT-4o-mini)
    └── app.py              → session management, API endpoints
```

Each browser tab gets its own ephemeral session with an isolated vector store. Sessions expire after 30 minutes of inactivity.

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | OpenAI GPT-4o-mini (streaming) |
| Embeddings | OpenAI `text-embedding-3-small` |
| Vector store | ChromaDB (in-memory, per session) |
| Reranker | FlashRank |
| Orchestration | LangChain (LCEL) |
| Backend | FastAPI + Uvicorn |
| Frontend | Next.js 15, TypeScript, Tailwind CSS |
| Testing | pytest, DeepEval, RAGAS |
| Package manager | uv |

---

## Project Structure

```
.
├── src/rag_porfolio/
│   ├── app.py          # FastAPI app, session management, all endpoints
│   ├── chain.py        # LCEL RAG chain with streaming support
│   ├── ingestion.py    # Document loading and chunking
│   ├── retrieval.py    # Vector retriever + FlashRank reranker
│   └── store.py        # ChromaDB wrapper + embedding service
├── frontend/
│   ├── app/            # Next.js App Router pages
│   ├── components/     # Navbar, ChatPanel, UploadPanel, DocumentList
│   └── lib/            # API client (SSE streaming)
├── tests/
│   ├── unit/           # Pure logic tests (no external calls)
│   ├── integration/    # Live API tests (requires OPENAI_API_KEY)
│   └── evaluation/     # DeepEval / RAGAS quality benchmarks
├── pyproject.toml
└── .env.example
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- Node.js 18+
- [uv](https://github.com/astral-sh/uv)
- An OpenAI API key

### 1. Clone and configure

```bash
git clone https://github.com/abdulrehman/rag-portfolio.git
cd rag-portfolio
cp .env.example .env
# Add your OPENAI_API_KEY to .env
```

### 2. Backend

```bash
uv sync
uv run uvicorn rag_porfolio.app:app --reload
```

API available at `http://localhost:8000`.

### 3. Frontend

```bash
cd frontend
npm install
npm run dev
```

UI available at `http://localhost:3000`.

---

## API Reference

### `POST /upload`
Upload and index a document.

- **Accepts:** `multipart/form-data` with a `file` field
- **Header:** `X-Session-ID: <session-id>`
- **Supported formats:** `.pdf`, `.docx`, `.txt` (max 10 MB)
- **Limits:** 20 documents per session

```json
{ "status": "indexed", "filename": "report.pdf", "chunk_count": 42 }
```

### `GET /query?q=<question>`
Stream an answer as Server-Sent Events.

- **Header:** `X-Session-ID: <session-id>`
- **Response:** `text/event-stream` — tokens followed by `data: [DONE]`

### `GET /documents`
List all indexed documents for the session.

### `DELETE /documents/{filename}`
Remove a document from the vector store.

---

## Running Tests

```bash
# Unit tests only (no API key needed)
uv run pytest -m unit

# Integration tests (requires OPENAI_API_KEY)
uv run pytest -m integration

# Evaluation suite (DeepEval / RAGAS)
uv run pytest -m evaluation
```

---

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `OPENAI_API_KEY` | OpenAI API key | required |
| `OPENAI_MODEL` | Chat model to use | `gpt-4o-mini` |
| `OPENAI_EMBEDDING_MODEL` | Embedding model | `text-embedding-3-small` |

---

## Author

**Abdul Rehman** — [m.abdulrehman.shoukat@gmail.com](mailto:m.abdulrehman.shoukat@gmail.com)
