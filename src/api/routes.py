"""
FastAPI Application — REST API for the Car Brochure Chatbot.

Endpoints:
    POST /chat          — Send a message and get a response
    POST /chat/reset    — Reset conversation history
    GET  /health        — Health check
    GET  /stats         — System statistics
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from config import get_settings
from src.chat.engine import ChatEngine

logger = logging.getLogger(__name__)

# ---- Request / Response Models ----


class ChatRequest(BaseModel):
    """Chat request payload."""
    message: str = Field(..., min_length=1, max_length=2000, description="User's question")
    brand_filter: str | None = Field(None, description="Optional: filter by car brand")
    model_filter: str | None = Field(None, description="Optional: filter by car model")


class ChatResponseModel(BaseModel):
    """Chat response payload."""
    answer: str
    sources: list[dict]
    search_mode: str
    query_intent: dict


class HealthResponse(BaseModel):
    status: str
    search_mode: str
    elasticsearch: str
    pinecone: str


# ---- Application ----

# Global chat engine instance
_chat_engine: ChatEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: initialize and cleanup."""
    global _chat_engine
    logger.info("Starting Car Brochure Search System...")
    _chat_engine = ChatEngine()
    logger.info(f"Search mode: {_chat_engine.search_mode}")
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="Car Brochure Search System",
    description=(
        "An intelligent chatbot for car recommendations and specifications. "
        "Powered by Elasticsearch, Pinecone, and Gemini."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


def _get_engine() -> ChatEngine:
    if _chat_engine is None:
        raise HTTPException(status_code=503, detail="Chat engine not initialized")
    return _chat_engine


# ---- Endpoints ----


@app.post("/chat", response_model=ChatResponseModel)
async def chat(request: ChatRequest):
    """Send a message to the car advisor chatbot."""
    engine = _get_engine()
    try:
        response = engine.chat(request.message)
        return ChatResponseModel(
            answer=response.answer,
            sources=response.sources,
            search_mode=response.search_mode,
            query_intent=response.query_intent,
        )
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process query: {str(e)}")


@app.post("/chat/reset")
async def reset_chat():
    """Reset the conversation history."""
    engine = _get_engine()
    engine.reset_conversation()
    return {"message": "Conversation history cleared"}


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check system health and component status."""
    settings = get_settings()

    es_status = "not_configured"
    if settings.search_mode.value in ("elasticsearch", "hybrid"):
        try:
            from elasticsearch import Elasticsearch
            es = Elasticsearch(hosts=[settings.elasticsearch_url])
            es_status = "connected" if es.ping() else "unreachable"
            es.close()
        except Exception:
            es_status = "error"

    pc_status = "not_configured"
    if settings.search_mode.value in ("pinecone", "hybrid"):
        pc_status = "configured" if settings.pinecone_api_key else "no_api_key"

    return HealthResponse(
        status="healthy",
        search_mode=settings.search_mode.value,
        elasticsearch=es_status,
        pinecone=pc_status,
    )


@app.get("/stats")
async def system_stats():
    """Get system statistics."""
    settings = get_settings()
    stats = {
        "search_mode": settings.search_mode.value,
        "top_k": settings.top_k,
        "chunk_size": settings.chunk_size,
    }

    # ES stats
    if settings.search_mode.value in ("elasticsearch", "hybrid"):
        try:
            from src.ingestion.es_indexer import ElasticsearchIndexer
            indexer = ElasticsearchIndexer()
            stats["elasticsearch"] = {
                "index": settings.es_index_name,
                "doc_count": indexer.get_doc_count(),
            }
            indexer.close()
        except Exception as e:
            stats["elasticsearch"] = {"error": str(e)}

    return stats
