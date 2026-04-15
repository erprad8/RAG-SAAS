"""
Background document ingestion job manager.
Author: Pradeep Kumar Verma

Uses Python's concurrent.futures.ThreadPoolExecutor to run ingestion
asynchronously so the upload endpoint returns immediately (202 Accepted)
and the client can poll /status/{doc_id} to track progress.

Design rationale for background jobs:
- Embedding generation is CPU-bound and can take seconds for large PDFs.
- Blocking the HTTP request thread during embedding would exhaust the
  Uvicorn thread pool under concurrent uploads.
- ThreadPoolExecutor (vs asyncio) is chosen because sentence-transformers
  does not release the GIL consistently; for true parallelism, switch to
  ProcessPoolExecutor or Celery + Redis.
"""

import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Dict

from app.models.schemas import DocumentStatus
from app.services.parser_service import DocumentParser, UnsupportedFormatError
from app.services.chunking_service import ChunkingService
from app.services.vector_store_service import VectorStoreService

logger = logging.getLogger(__name__)

# Job state store (in-memory; swap to Redis for multi-process deployments)
_job_store: Dict[str, dict] = {}

_executor = ThreadPoolExecutor(max_workers=4)


def get_job(doc_id: str) -> dict:
    return _job_store.get(doc_id, {})


def get_all_jobs() -> Dict[str, dict]:
    return dict(_job_store)


def submit_ingestion_job(
    file_bytes: bytes,
    filename: str,
    vector_store: VectorStoreService,
) -> str:
    """
    Accept a document upload and schedule background ingestion.

    Returns the doc_id for status polling.
    """
    doc_id = str(uuid.uuid4())
    _job_store[doc_id] = {
        "doc_id": doc_id,
        "filename": filename,
        "status": DocumentStatus.PENDING,
        "total_chunks": None,
        "error": None,
        "submitted_at": time.time(),
    }

    _executor.submit(_ingest_document, doc_id, file_bytes, filename, vector_store)
    logger.info(f"[{doc_id}] Ingestion job submitted for '{filename}'.")
    return doc_id


def _ingest_document(
    doc_id: str,
    file_bytes: bytes,
    filename: str,
    vector_store: VectorStoreService,
):
    """
    Background task: parse → chunk → embed → index.
    Updates _job_store at each stage so the status endpoint reflects progress.
    """
    _update_job(doc_id, status=DocumentStatus.PROCESSING)
    start = time.perf_counter()

    try:
        # Step 1: Parse
        parser = DocumentParser()
        text_blocks = parser.parse(file_bytes, filename)
        if not text_blocks:
            raise ValueError(f"No text could be extracted from '{filename}'.")

        # Step 2: Chunk
        chunker = ChunkingService()
        chunks = chunker.chunk_document(text_blocks, doc_id)
        logger.info(f"[{doc_id}] Created {len(chunks)} chunks.")

        # Step 3: Embed & Index
        vector_store.add_chunks(chunks)

        elapsed = round((time.perf_counter() - start) * 1000, 2)
        logger.info(f"[{doc_id}] Ingestion complete in {elapsed}ms.")
        _update_job(doc_id, status=DocumentStatus.COMPLETED, total_chunks=len(chunks))

    except UnsupportedFormatError as e:
        logger.warning(f"[{doc_id}] Unsupported format: {e}")
        _update_job(doc_id, status=DocumentStatus.FAILED, error=str(e))

    except Exception as e:
        logger.error(f"[{doc_id}] Ingestion failed: {e}", exc_info=True)
        _update_job(doc_id, status=DocumentStatus.FAILED, error=str(e))


def _update_job(doc_id: str, **kwargs):
    if doc_id in _job_store:
        _job_store[doc_id].update(kwargs)
