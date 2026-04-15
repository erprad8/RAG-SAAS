"""
Pydantic models for request/response validation.
Author: Pradeep Kumar Verma
"""

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field, validator
import uuid


class DocumentStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ChunkMetadata(BaseModel):
    """Metadata stored alongside each text chunk."""
    doc_id: str
    filename: str
    chunk_index: int
    total_chunks: int
    char_start: int
    char_end: int
    page_number: Optional[int] = None


class DocumentChunk(BaseModel):
    """A single text chunk with metadata."""
    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str
    metadata: ChunkMetadata


class DocumentUploadResponse(BaseModel):
    """Response after document upload is accepted."""
    doc_id: str
    filename: str
    status: DocumentStatus
    message: str


class DocumentStatusResponse(BaseModel):
    """Status polling response for background ingestion jobs."""
    doc_id: str
    filename: str
    status: DocumentStatus
    total_chunks: Optional[int] = None
    error: Optional[str] = None


class QueryRequest(BaseModel):
    """User query payload."""
    question: str = Field(..., min_length=3, max_length=1000, description="The question to ask.")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve.")
    similarity_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum cosine similarity score (0.0–1.0) for a chunk to be included.",
    )
    doc_ids: Optional[List[str]] = Field(
        default=None,
        description="Optional list of document IDs to restrict retrieval to.",
    )

    @validator("question")
    def question_must_not_be_blank(cls, v):
        if not v.strip():
            raise ValueError("Question must not be blank.")
        return v.strip()


class RetrievedChunk(BaseModel):
    """A single retrieved chunk with its similarity score."""
    chunk_id: str
    text: str
    similarity_score: float
    filename: str
    chunk_index: int
    page_number: Optional[int] = None


class QueryResponse(BaseModel):
    """Full QA response including retrieved context and generated answer."""
    question: str
    answer: str
    retrieved_chunks: List[RetrievedChunk]
    latency_ms: float
    retrieval_latency_ms: float
    generation_latency_ms: float
    avg_similarity_score: float
    model_used: str


class MetricsResponse(BaseModel):
    """System-level metrics."""
    total_documents: int
    total_chunks: int
    total_queries_served: int
    avg_query_latency_ms: float
    avg_similarity_score: float
    retrieval_failure_rate: float


class HealthResponse(BaseModel):
    status: str
    vector_store_ready: bool
    total_indexed_chunks: int
