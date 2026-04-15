"""
Query router: accepts user questions and returns RAG-generated answers.
Author: Pradeep Kumar Verma
"""

import logging
import time

from fastapi import APIRouter, HTTPException, Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.models.schemas import QueryRequest, QueryResponse
from app.services.llm_service import LLMService
from app.services.metrics_service import metrics_tracker, QueryMetricRecord

logger = logging.getLogger(__name__)
router = APIRouter()
limiter = Limiter(key_func=get_remote_address)

_llm_service = LLMService()


@router.post(
    "/ask",
    response_model=QueryResponse,
    summary="Ask a question answered using your uploaded documents",
)
@limiter.limit("30/minute")
async def ask_question(request: Request, body: QueryRequest):
    """
    Submit a question. The system retrieves relevant document chunks
    and generates an answer using an LLM grounded in that context.

    Rate limit: 30 questions per minute per IP.
    """
    vector_store = request.app.state.vector_store

    if not vector_store.is_ready:
        raise HTTPException(status_code=503, detail="Vector store is not ready yet.")
    if vector_store.total_chunks == 0:
        raise HTTPException(
            status_code=422,
            detail="No documents have been ingested yet. Please upload documents first.",
        )

    total_start = time.perf_counter()

    # ── Retrieval ──────────────────────────────────────────────────────────
    retrieval_start = time.perf_counter()
    retrieved_chunks, avg_sim = vector_store.search(
        query=body.question,
        top_k=body.top_k,
        similarity_threshold=body.similarity_threshold,
        doc_ids=body.doc_ids,
    )
    retrieval_ms = round((time.perf_counter() - retrieval_start) * 1000, 2)

    # ── Generation ────────────────────────────────────────────────────────
    generation_start = time.perf_counter()
    answer, model_used, gen_ms = _llm_service.generate_answer(
        question=body.question,
        retrieved_chunks=retrieved_chunks,
    )
    generation_ms = round((time.perf_counter() - generation_start) * 1000, 2)

    total_ms = round((time.perf_counter() - total_start) * 1000, 2)

    # ── Metrics ───────────────────────────────────────────────────────────
    metrics_tracker.record(
        QueryMetricRecord(
            question=body.question,
            total_latency_ms=total_ms,
            retrieval_latency_ms=retrieval_ms,
            generation_latency_ms=gen_ms,
            avg_similarity_score=avg_sim,
            num_chunks_retrieved=len(retrieved_chunks),
            retrieval_failed=(len(retrieved_chunks) == 0),
        )
    )

    logger.info(
        f"Query answered | retrieval={retrieval_ms}ms | generation={gen_ms}ms "
        f"| total={total_ms}ms | chunks={len(retrieved_chunks)} | avg_sim={avg_sim}"
    )

    return QueryResponse(
        question=body.question,
        answer=answer,
        retrieved_chunks=retrieved_chunks,
        latency_ms=total_ms,
        retrieval_latency_ms=retrieval_ms,
        generation_latency_ms=gen_ms,
        avg_similarity_score=avg_sim,
        model_used=model_used,
    )
