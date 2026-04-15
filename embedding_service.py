"""
Embedding service using sentence-transformers (all-MiniLM-L6-v2).
Author: Pradeep Kumar Verma

Model choice rationale:
- all-MiniLM-L6-v2 is a lightweight (22M param) model optimised for
  semantic similarity tasks.
- 384-dimensional embeddings are compact enough for fast FAISS similarity
  search while capturing rich semantic meaning.
- Runs entirely on CPU with no GPU required — important for local/dev use.
- Apache 2.0 licensed, no API key needed.
"""

import logging
import time
from typing import List, Union

import numpy as np

logger = logging.getLogger(__name__)

MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


class EmbeddingService:
    """Wraps sentence-transformers for text → embedding vector conversion."""

    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self._model = None
        self.embedding_dim = EMBEDDING_DIM

    def _load_model(self):
        """Lazy-load model on first use to avoid startup delay."""
        if self._model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
                logger.info(f"✅ Embedding model '{self.model_name}' loaded.")
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required: pip install sentence-transformers"
                )
        return self._model

    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Embed one or more texts. Returns float32 numpy array of shape
        (n_texts, embedding_dim), L2-normalised for cosine similarity via dot product.

        METRIC TRACKED — Embedding Latency:
            Measured per-batch and logged. Useful for profiling ingestion throughput.
            Typical: ~10ms per chunk on CPU (MiniLM-L6-v2).
        """
        if isinstance(texts, str):
            texts = [texts]

        model = self._load_model()
        start = time.perf_counter()
        embeddings = model.encode(
            texts,
            normalize_embeddings=True,   # L2 normalise → dot product = cosine sim
            show_progress_bar=False,
            batch_size=32,
        )
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        logger.debug(f"Embedded {len(texts)} chunk(s) in {elapsed_ms}ms")

        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query string. Returns shape (1, embedding_dim).
        Kept separate so query-specific preprocessing can be added later
        (e.g., query expansion, HyDE).
        """
        return self.embed([query])
