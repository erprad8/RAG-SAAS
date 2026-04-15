from fastapi import FastAPI
from app.routes import upload, query

app = FastAPI(title="RAG SaaS")

app.include_router(upload.router)
app.include_router(query.router)

@app.get("/")
def home():
    return {"msg": "RAG SaaS is Live 🚀"}
