from fastapi import APIRouter
from app.services.retrieval import retrieve_chunks
from app.services.llm import generate_answer

router = APIRouter()

@router.post("/query")
async def query(q: str):
    chunks = retrieve_chunks(q)
    answer = generate_answer(q, chunks)
    return {"answer": answer}
