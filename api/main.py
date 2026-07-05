# api/main.py
# FastAPI backend for FinSight RAG
# Exposes two endpoints:
#   POST /query  — takes a question, returns answer + citations
#   GET  /health — returns service status (standard for any deployed API)

import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_groq import ChatGroq

# Add src/ to path so we can import our modules from api/main.py
# Without this, Python can't find ingestion.py, retriever.py etc.
sys.path.append(str(Path(__file__).parent.parent / "src"))

from ingestion import load_and_chunk
from retriever import build_retriever
from reranker import rerank
from pipeline import SYSTEM_PROMPT, build_context_string, query_pipeline

load_dotenv()

# ── FastAPI app instance ──────────────────────────────────────────────────────
app = FastAPI(
    title="FinSight RAG API",
    description="Financial document intelligence system for Infosys annual reports",
    version="1.0.0"
)

# ── Global state — built once at startup, reused for every request ────────────
# This is the key production pattern — don't rebuild indexes on every request
# Build once when the server starts, hold in memory, serve many queries fast
retriever = None
llm = None
faiss_vectorstore = None


@app.on_event("startup")
async def startup_event():
    """
    Runs once when FastAPI server starts.
    Loads PDFs, builds indexes, initialises LLM.
    All subsequent requests reuse these — no rebuilding per request.
    """
    global retriever, llm

    print("FinSight RAG API starting up...")
    print("Loading documents and building indexes — this takes ~60 seconds on first run...")

    # Load chunks
    chunks = load_and_chunk()

    # Build hybrid retriever — FAISS + BM25
    retriever = build_retriever(chunks)

    # Initialise LLM
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY")
    )

    print("FinSight RAG API ready. Accepting requests.")


# ── Request / Response models ─────────────────────────────────────────────────
class QueryRequest(BaseModel):
    """
    What the client sends in the POST /query request body.
    Pydantic automatically validates types and raises 422 if wrong.
    """
    question: str           # the user's question
    top_n: int = 5          # how many chunks to pass to LLM (default 5)


class SourceCitation(BaseModel):
    """
    A single source citation — filename + page number.
    Returned alongside the answer so the client can show references.
    """
    filename: str
    page: int


class QueryResponse(BaseModel):
    """
    What the API returns after processing a query.
    """
    question: str
    answer: str
    sources: list[SourceCitation]   # list of citations
    latency_ms: float               # how long the query took in milliseconds


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
async def health_check():
    """
    Standard health check endpoint.
    Returns 200 if the service is up and indexes are loaded.
    Used by Railway/Render to verify the container is alive.
    """
    return {
        "status": "healthy",
        "retriever_loaded": retriever is not None,
        "llm_loaded": llm is not None
    }


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """
    Main RAG query endpoint.
    Takes a question, runs full pipeline, returns answer + citations + latency.

    Flow:
    request.question → hybrid retrieval → reranking → LLM → response
    """
    # Guard: if startup failed and retriever isn't loaded, return 503
    if retriever is None or llm is None:
        raise HTTPException(
            status_code=503,
            detail="Service not ready — indexes still loading. Try again in 60 seconds."
        )

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # Time the full pipeline
    start_time = time.time()

    # Run the full RAG pipeline
    result = query_pipeline(
        query=request.question,
        retriever=retriever,
        llm=llm,
        top_n=request.top_n
    )

    latency_ms = (time.time() - start_time) * 1000  # convert to milliseconds

    # Build citation objects from (filename, page) tuples
    sources = [
        SourceCitation(filename=filename, page=int(page))
        for filename, page in result["sources"]
    ]

    return QueryResponse(
        question=request.question,
        answer=result["answer"],
        sources=sources,
        latency_ms=round(latency_ms, 2)
    )


# ── Run directly for local development ───────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    # reload=False in production — reload=True only during development
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)