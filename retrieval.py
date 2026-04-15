import faiss
import numpy as np
from app.services.embedding import model

index = faiss.IndexFlatL2(384)
stored_chunks = []

def store_embeddings(chunks, embeddings):
    global stored_chunks
    stored_chunks.extend(chunks)
    index.add(np.array(embeddings))

def retrieve_chunks(query, k=3):
    query_vector = model.encode([query])
    D, I = index.search(np.array(query_vector), k)
    return [stored_chunks[i] for i in I[0]]
