"""Health check router. Author: Pradeep Kumar Verma"""
from fastapi import APIRouter, Request
from app.models.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse, summary="System health check")
async def health_check(request: Request):
    vs = request.app.state.vector_store
    return HealthResponse(
        status="ok",
        vector_store_ready=vs.is_ready,
        total_indexed_chunks=vs.total_chunks,
    )
