"""
RAG-Based Question Answering System
Author: Pradeep Kumar Verma
Description: FastAPI application with document ingestion, embedding, retrieval, and LLM-powered QA.
"""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from app.routers import documents, query, health
from app.services.vector_store_service import VectorStoreService
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

# Rate limiter
limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize services on startup, cleanup on shutdown."""
    logger.info("🚀 Starting RAG QA System...")
    vector_store = VectorStoreService()
    vector_store.initialize()
    app.state.vector_store = vector_store
    logger.info("✅ Vector store initialized.")
    yield
    logger.info("🛑 Shutting down RAG QA System...")
    vector_store.save()
    logger.info("✅ Vector store persisted to disk.")


app = FastAPI(
    title="RAG-Based Question Answering System",
    description=(
        "Upload documents (PDF/TXT) and ask questions powered by "
        "Retrieval-Augmented Generation. Built by Pradeep Kumar Verma."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = round((time.perf_counter() - start) * 1000, 2)
    response.headers["X-Process-Time-Ms"] = str(elapsed)
    logger.info(f"{request.method} {request.url.path} → {response.status_code} [{elapsed}ms]")
    return response


# Routers
app.include_router(health.router, tags=["Health"])
app.include_router(documents.router, prefix="/api/v1/documents", tags=["Documents"])
app.include_router(query.router, prefix="/api/v1/query", tags=["Query"])


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Please check the logs."},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
