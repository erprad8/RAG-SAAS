"""
In-memory metrics tracker.
Author: Pradeep Kumar Verma

METRIC TRACKED: Query Latency (end-to-end ms)
─────────────────────────────────────────────
Every query records total latency, retrieval latency, and generation latency
separately. This allows isolating bottlenecks:
  - High retrieval latency → FAISS index needs IVF approximation or sharding.
  - High generation latency → switch to a faster model tier.
  - Low avg_similarity_score → topic not covered; add more documents.

METRIC TRACKED: Retrieval Failure Rate
────────────────────────────────────────
A query is counted as a "retrieval failure" when zero chunks pass the
similarity threshold. Tracking this rate helps set the right threshold:
  - Too high (e.g., 0.7) → many valid queries fail → lower threshold.
  - Too low (e.g., 0.1) → irrelevant chunks retrieved → raise threshold.
"""

import threading
from dataclasses import dataclass, field
from typing import List


@dataclass
class QueryMetricRecord:
    question: str
    total_latency_ms: float
    retrieval_latency_ms: float
    generation_latency_ms: float
    avg_similarity_score: float
    num_chunks_retrieved: int
    retrieval_failed: bool  # True if 0 chunks returned


class MetricsTracker:
    """Thread-safe in-memory metrics store."""

    def __init__(self):
        self._lock = threading.Lock()
        self._records: List[QueryMetricRecord] = []

    def record(self, record: QueryMetricRecord):
        with self._lock:
            self._records.append(record)

    def summary(self) -> dict:
        with self._lock:
            n = len(self._records)
            if n == 0:
                return {
                    "total_queries_served": 0,
                    "avg_query_latency_ms": 0.0,
                    "avg_similarity_score": 0.0,
                    "retrieval_failure_rate": 0.0,
                }
            avg_latency = round(sum(r.total_latency_ms for r in self._records) / n, 2)
            avg_sim = round(sum(r.avg_similarity_score for r in self._records) / n, 4)
            failure_rate = round(sum(1 for r in self._records if r.retrieval_failed) / n, 4)
            return {
                "total_queries_served": n,
                "avg_query_latency_ms": avg_latency,
                "avg_similarity_score": avg_sim,
                "retrieval_failure_rate": failure_rate,
            }


# Singleton instance
metrics_tracker = MetricsTracker()
