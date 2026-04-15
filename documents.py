"""
Documents router: upload and status endpoints.
Author: Pradeep Kumar Verma
"""

import logging
from typing import List

from fastapi import APIRouter, File, HTTPException, Request, UploadFile, Depends
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.models.schemas import (
    DocumentUploadResponse,
    DocumentStatus,
    DocumentStatusResponse,
    MetricsResponse,
)
from app.services.ingestion_job import submit_ingestion_job, get_job, get_all_jobs
from app.services.metrics_service import metrics_tracker

logger = logging.getLogger(__name__)
router = APIRouter()
limiter = Limiter(key_func=get_remote_address)

ALLOWED_EXTENSIONS = {".pdf", ".txt"}
MAX_FILE_SIZE_MB = 20
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=202,
    summary="Upload a document for RAG ingestion (PDF or TXT)",
)
@limiter.limit("10/minute")
async def upload_document(request: Request, file: UploadFile = File(...)):
    """
    Upload a document. Ingestion runs in the background.
    Poll GET /status/{doc_id} to track progress.

    Rate limit: 10 uploads per minute per IP.
    """
    filename = file.filename or "unknown"
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )

    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(contents) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max allowed: {MAX_FILE_SIZE_MB}MB.",
        )

    vector_store = request.app.state.vector_store
    doc_id = submit_ingestion_job(contents, filename, vector_store)

    return DocumentUploadResponse(
        doc_id=doc_id,
        filename=filename,
        status=DocumentStatus.PENDING,
        message=(
            f"Document '{filename}' accepted. "
            f"Poll GET /api/v1/documents/status/{doc_id} to check progress."
        ),
    )


@router.get(
    "/status/{doc_id}",
    response_model=DocumentStatusResponse,
    summary="Check ingestion status for a document",
)
async def get_document_status(doc_id: str):
    job = get_job(doc_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Document ID '{doc_id}' not found.")
    return DocumentStatusResponse(
        doc_id=job["doc_id"],
        filename=job["filename"],
        status=job["status"],
        total_chunks=job.get("total_chunks"),
        error=job.get("error"),
    )


@router.get(
    "/list",
    response_model=List[DocumentStatusResponse],
    summary="List all uploaded documents and their ingestion status",
)
async def list_documents():
    jobs = get_all_jobs()
    return [
        DocumentStatusResponse(
            doc_id=j["doc_id"],
            filename=j["filename"],
            status=j["status"],
            total_chunks=j.get("total_chunks"),
            error=j.get("error"),
        )
        for j in jobs.values()
    ]


@router.get(
    "/metrics",
    response_model=MetricsResponse,
    summary="System-level retrieval and performance metrics",
)
async def get_metrics(request: Request):
    vs = request.app.state.vector_store
    summary = metrics_tracker.summary()
    return MetricsResponse(
        total_documents=vs.total_documents,
        total_chunks=vs.total_chunks,
        **summary,
    )
