"""
FastAPI backend for the Agentic RAG Assistant.

Wraps the existing retrieval pipeline (chunking, hybrid_search) and the
CodeAgent in agent.py behind a small JSON API so the standalone frontend
in frontend/index.html can talk to it.

Run with:
    uvicorn backend:app --reload --port 8000
"""

import shutil
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

BASE_DIR = Path(__file__).parent
DOCS_DIR = BASE_DIR / "data" / "documents"
DOCS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Agentic RAG Assistant API")

# Wide open for local development: the frontend is a static HTML file
# that may be opened directly (file://) or served from any local port.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Lazy-loaded pipeline pieces
# ============================================================
# These imports are heavy (embedding model, FAISS index, BM25 index, and
# — for ask_question — the local Qwen model), so they only run once,
# on first use, rather than blocking server startup.

_hybrid_search = None
_chunks = None
_ask_question = None


def get_hybrid_search():
    global _hybrid_search
    if _hybrid_search is None:
        from hybrid_search import hybrid_search
        _hybrid_search = hybrid_search
    return _hybrid_search


def get_chunks():
    global _chunks
    if _chunks is None:
        from chunking import chunks
        _chunks = chunks
    return _chunks


def get_ask_question():
    global _ask_question
    if _ask_question is None:
        from agent import ask_question
        _ask_question = ask_question
    return _ask_question


# ============================================================
# Schemas
# ============================================================

class AskRequest(BaseModel):
    question: str


class Source(BaseModel):
    document_id: str
    title: str
    page: Optional[int] = None
    relevance: float


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


class DocumentInfo(BaseModel):
    filename: str
    title: str
    size_bytes: int
    chunk_count: int
    indexed: bool


class StatsResponse(BaseModel):
    document_count: int
    chunk_count: int
    indexed_percent: int
    embedding_dimensions: Optional[int] = None


# ============================================================
# Helpers
# ============================================================

def chunk_counts_by_document() -> dict[str, int]:
    counts: dict[str, int] = {}
    for chunk in get_chunks():
        doc_id = chunk.metadata.get("document_id", "unknown")
        counts[doc_id] = counts.get(doc_id, 0) + 1
    return counts


def documents_info() -> list[DocumentInfo]:
    counts = chunk_counts_by_document()
    docs = []
    for path in sorted(DOCS_DIR.glob("*.pdf")):
        chunk_count = counts.get(path.stem, 0)
        docs.append(
            DocumentInfo(
                filename=path.name,
                title=path.stem.replace("_", " "),
                size_bytes=path.stat().st_size,
                chunk_count=chunk_count,
                indexed=chunk_count > 0,
            )
        )
    return docs


# ============================================================
# Routes
# ============================================================

@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/documents", response_model=list[DocumentInfo])
def list_documents():
    return documents_info()


@app.get("/api/stats", response_model=StatsResponse)
def stats():
    docs = documents_info()
    total_chunks = sum(d.chunk_count for d in docs)
    indexed_docs = [d for d in docs if d.indexed]
    percent = round(100 * len(indexed_docs) / len(docs)) if docs else 0

    dimensions = None
    try:
        from embeddings import embeddings
        dimensions = int(embeddings.shape[1])
    except Exception:
        pass

    return StatsResponse(
        document_count=len(docs),
        chunk_count=total_chunks,
        indexed_percent=percent,
        embedding_dimensions=dimensions,
    )


@app.post("/api/documents/upload", response_model=DocumentInfo)
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    destination = DOCS_DIR / file.filename
    with destination.open("wb") as out:
        shutil.copyfileobj(file.file, out)

    counts = chunk_counts_by_document()
    chunk_count = counts.get(destination.stem, 0)

    return DocumentInfo(
        filename=destination.name,
        title=destination.stem.replace("_", " "),
        size_bytes=destination.stat().st_size,
        chunk_count=chunk_count,
        indexed=chunk_count > 0,
    )


@app.delete("/api/documents/{filename}")
def delete_document(filename: str):
    path = DOCS_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Document not found.")

    counts = chunk_counts_by_document()
    if counts.get(path.stem, 0) > 0:
        raise HTTPException(
            status_code=409,
            detail=(
                "This document is already indexed. Re-run the ingestion "
                "pipeline (chunking + embeddings) to remove it safely."
            ),
        )

    path.unlink()
    return {"status": "deleted", "filename": filename}


@app.post("/api/ask", response_model=AskResponse)
def ask(payload: AskRequest):
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    hybrid_search = get_hybrid_search()
    results = hybrid_search(question, top_k=4)

    sources = [
        Source(
            document_id=result["id"],
            title=result["title"].replace("_", " "),
            page=result.get("page"),
            relevance=round(result["hybrid_score"], 2),
        )
        for result in results
    ]

    ask_question = get_ask_question()
    answer = ask_question(question)

    return AskResponse(answer=answer, sources=sources)