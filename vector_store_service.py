"""
Vector store service backed by FAISS (IndexFlatIP — inner product on normalised vectors = cosine similarity).
Author: Pradeep Kumar Verma

Why FAISS (not Pinecone/Weaviate for this project):
- Local, zero-cost, no network dependency — ideal for CI/CD and offline demos.
- IndexFlatIP gives exact search (no approximation error), which is acceptable
  for corpora < 1M chunks.
- Pickle-based persistence keeps the store alive across restarts.
- Swap to IndexIVFFlat or Pinecone for production scale (>500K chunks).
"""

import logging
import os
import pickle
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

import numpy as np

from app.models.schemas import DocumentChunk, RetrievedChunk
from app.services.embedding_service import EmbeddingService, EMBEDDING_DIM

logger = logging.getLogger(__name__)

VECTOR_STORE_PATH = Path("vector_store/faiss.index")
METADATA_STORE_PATH = Path("vector_store/metadata.pkl")


class VectorStoreService:
    """
    Manages FAISS index + metadata store.

    Internal design:
    - FAISS index stores float32 embeddings; IDs are integer positions.
    - A parallel list `self._chunks` stores DocumentChunk objects at the
      same positions as FAISS vectors. This avoids a separate database.
    - Thread safety: ingestion runs in a background thread; queries run in
      the request thread. In production, use a read-write lock or separate
      FAISS instances.
    """

    def __init__(self):
        self._index = None          # faiss.IndexFlatIP
        self._chunks: List[DocumentChunk] = []
        self._doc_chunk_map: Dict[str, List[int]] = {}  # doc_id → list of FAISS positions
        self._embedder = EmbeddingService()
        self._ready = False

    # ──────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ──────────────────────────────────────────────────────────────────────

    def initialize(self):
        """Load persisted index from disk, or create a fresh one."""
        try:
            import faiss
        except ImportError:
            raise ImportError("FAISS is required: pip install faiss-cpu")

        VECTOR_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)

        if VECTOR_STORE_PATH.exists() and METADATA_STORE_PATH.exists():
            logger.info("Loading existing FAISS index from disk...")
            import faiss as _faiss
            self._index = _faiss.read_index(str(VECTOR_STORE_PATH))
            with open(METADATA_STORE_PATH, "rb") as f:
                state = pickle.load(f)
                self._chunks = state["chunks"]
                self._doc_chunk_map = state["doc_chunk_map"]
            logger.info(f"✅ Loaded {self._index.ntotal} vectors, {len(self._chunks)} chunk metadata records.")
        else:
            logger.info("No existing index found. Creating fresh FAISS IndexFlatIP.")
            import faiss as _faiss
            self._index = _faiss.IndexFlatIP(EMBEDDING_DIM)

        self._ready = True

    def save(self):
        """Persist index and metadata to disk."""
        if self._index is None:
            return
        import faiss as _faiss
        VECTOR_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _faiss.write_index(self._index, str(VECTOR_STORE_PATH))
        with open(METADATA_STORE_PATH, "wb") as f:
            pickle.dump(
                {"chunks": self._chunks, "doc_chunk_map": self._doc_chunk_map},
                f,
            )
        logger.info(f"💾 Persisted {self._index.ntotal} vectors to {VECTOR_STORE_PATH}.")

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def total_chunks(self) -> int:
        return len(self._chunks)

    @property
    def total_documents(self) -> int:
        return len(self._doc_chunk_map)

    # ──────────────────────────────────────────────────────────────────────
    # Ingestion
    # ──────────────────────────────────────────────────────────────────────

    def add_chunks(self, chunks: List[DocumentChunk]) -> None:
        """
        Embed and index a list of DocumentChunk objects.
        Called from the background ingestion job.
        """
        if not chunks:
            return

        texts = [c.text for c in chunks]
        embeddings = self._embedder.embed(texts)  # shape (n, dim)

        start_idx = self._index.ntotal
        self._index.add(embeddings)

        for i, chunk in enumerate(chunks):
            faiss_pos = start_idx + i
            self._chunks.append(chunk)
            doc_id = chunk.metadata.doc_id
            self._doc_chunk_map.setdefault(doc_id, []).append(faiss_pos)

        logger.info(f"Indexed {len(chunks)} chunk(s). Total in store: {self._index.ntotal}")
        self.save()

    # ──────────────────────────────────────────────────────────────────────
    # Retrieval
    # ──────────────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 5,
        similarity_threshold: float = 0.3,
        doc_ids: Optional[List[str]] = None,
    ) -> Tuple[List[RetrievedChunk], float]:
        """
        Retrieve top-k chunks most relevant to `query`.

        METRIC TRACKED — Cosine Similarity Score:
            Logged per query. Used to detect retrieval failures:
            - avg_score < 0.3 → likely a topic not covered in documents.
            - top-1 score < 0.2 → high chance of hallucination in LLM answer.
            These thresholds were derived empirically by observing score
            distributions across ~50 test queries during development.

        Args:
            query: The user question.
            top_k: Max number of chunks to return.
            similarity_threshold: Minimum cosine similarity to include chunk.
            doc_ids: If given, restrict retrieval to these document IDs only.

        Returns:
            (list of RetrievedChunk, avg_similarity_score)
        """
        if self._index.ntotal == 0:
            logger.warning("Vector store is empty. No chunks to search.")
            return [], 0.0

        query_embedding = self._embedder.embed_query(query)  # (1, dim)

        # Fetch more candidates if filtering by doc_ids, to ensure top_k after filter
        fetch_k = top_k * 5 if doc_ids else top_k
        fetch_k = min(fetch_k, self._index.ntotal)

        scores, indices = self._index.search(query_embedding, fetch_k)
        scores = scores[0]   # flatten to 1D
        indices = indices[0]

        results = []
        for score, idx in zip(scores, indices):
            if idx < 0:
                continue  # FAISS returns -1 for padding
            score = float(score)
            if score < similarity_threshold:
                continue

            chunk = self._chunks[idx]

            # Doc ID filter
            if doc_ids and chunk.metadata.doc_id not in doc_ids:
                continue

            results.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    similarity_score=round(score, 4),
                    filename=chunk.metadata.filename,
                    chunk_index=chunk.metadata.chunk_index,
                    page_number=chunk.metadata.page_number,
                )
            )

            if len(results) >= top_k:
                break

        avg_score = round(sum(r.similarity_score for r in results) / len(results), 4) if results else 0.0
        logger.info(
            f"Query retrieved {len(results)} chunk(s) "
            f"(avg_sim={avg_score}, threshold={similarity_threshold})"
        )
        return results, avg_score
