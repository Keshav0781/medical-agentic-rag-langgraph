"""
FastAPI application for medical-agentic-rag-langgraph pipeline.

Exposes modular endpoints for internal pipeline integration
and workflow testing — exactly as used at Siemens Healthineers.

Endpoints:
- GET  /health    — service health check
- POST /search    — RAG question answering
- POST /summarise — document summarisation

"""

import logging
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from configs.config import LOG_LEVEL, validate_config
from src.agents.rag_agent import rag_graph

# ── Logging setup ────────────────────────────────────────────────────────────
logging.basicConfig(level=getattr(logging, LOG_LEVEL))
logger = logging.getLogger(__name__)

# ── Validate config on startup ───────────────────────────────────────────────
validate_config()

# ── FastAPI app ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="Medical R&D RAG Pipeline",
    description="""
    Agentic RAG pipeline for Siemens Healthineers R&D documents.
    Built with LangGraph, Qdrant, and Groq Llama-3.
    Demonstrates enterprise AI engineering patterns.
    """,
    version="1.0.0"
)

# ── CORS middleware ──────────────────────────────────────────────────────────
# Allows frontend UI to call these endpoints
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request/Response models ──────────────────────────────────────────────────
class SearchRequest(BaseModel):
    query: str
    conversation_history: Optional[list] = []


class SearchResponse(BaseModel):
    query: str
    answer: str
    follow_ups: list
    route: str
    chunks_retrieved: int
    chunks_reranked: int
    rewritten_query: str


class SummariseRequest(BaseModel):
    query: str


class SummariseResponse(BaseModel):
    query: str
    answer: str
    follow_ups: list


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health_check():
    """
    Service health check endpoint.
    Called every 30 seconds by monitoring tools.
    """
    return HealthResponse(
        status="healthy",
        service="medical-agentic-rag-langgraph",
        version="1.0.0"
    )


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest):
    """
    RAG question answering endpoint.
    Runs full LangGraph pipeline — guardrail, routing,
    query understanding, retrieval, reranking, generation.
    """
    if not request.query or len(request.query.strip()) == 0:
        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty"
        )

    logger.info(f"Search request: {request.query[:80]}")
    start_time = time.time()

    try:
        result = rag_graph.invoke({
            "query": request.query,
            "rewritten_query": "",
            "route": "",
            "chunks": [],
            "reranked_chunks": [],
            "answer": "",
            "follow_ups": [],
            "retry_count": 0,
            "is_safe": False,
            "error": "",
            "conversation_history": request.conversation_history or []
        })

        latency = round(time.time() - start_time, 2)
        logger.info(f"Search completed in {latency}s")

        return SearchResponse(
            query=request.query,
            answer=result.get("answer", ""),
            follow_ups=result.get("follow_ups", []),
            route=result.get("route", ""),
            chunks_retrieved=len(result.get("chunks", [])),
            chunks_reranked=len(result.get("reranked_chunks", [])),
            rewritten_query=result.get("rewritten_query", "")
        )

    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline error: {str(e)}"
        )


@app.post("/summarise", response_model=SummariseResponse)
def summarise(request: SummariseRequest):
    """
    Document summarisation endpoint.
    Runs map-reduce summarisation on specified document.
    """
    if not request.query or len(request.query.strip()) == 0:
        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty"
        )

    logger.info(f"Summarise request: {request.query[:80]}")
    start_time = time.time()

    try:
        result = rag_graph.invoke({
            "query": request.query,
            "rewritten_query": "",
            "route": "",
            "chunks": [],
            "reranked_chunks": [],
            "answer": "",
            "follow_ups": [],
            "retry_count": 0,
            "is_safe": False,
            "error": "",
            "conversation_history": []
        })

        latency = round(time.time() - start_time, 2)
        logger.info(f"Summarise completed in {latency}s")

        return SummariseResponse(
            query=request.query,
            answer=result.get("answer", ""),
            follow_ups=result.get("follow_ups", [])
        )

    except Exception as e:
        logger.error(f"Summarise failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline error: {str(e)}"
        )