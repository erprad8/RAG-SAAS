from fastapi import APIRouter, UploadFile, BackgroundTasks
from app.worker.ingestion import process_document

router = APIRouter()

@router.post("/upload")
async def upload(file: UploadFile, bg_tasks: BackgroundTasks):
    bg_tasks.add_task(process_document, file)
    return {"msg": "Processing started"}
