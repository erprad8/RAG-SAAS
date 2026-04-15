import fitz
from app.services.chunking import chunk_text
from app.services.embedding import embed_chunks
from app.services.retrieval import store_embeddings

def extract_text(file):
    if file.filename.endswith(".pdf"):
        pdf = fitz.open(stream=file.file.read(), filetype="pdf")
        return "".join([page.get_text() for page in pdf])
    else:
        return file.file.read().decode()

def process_document(file):
    text = extract_text(file)
    chunks = chunk_text(text)
    embeddings = embed_chunks(chunks)
    store_embeddings(chunks, embeddings)
