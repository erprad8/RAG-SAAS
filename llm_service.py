"""
LLM service: generates answers from retrieved context using Anthropic Claude API.
Author: Pradeep Kumar Verma

Design choices:
- Uses Anthropic Python SDK (easily swappable to OpenAI or local Ollama).
- Prompt is carefully engineered to:
    1. Ground the model strictly in the provided context (reduces hallucination).
    2. Instruct the model to say "I don't know" when context is insufficient.
    3. Include source attribution in the answer.
- Temperature=0 for reproducible, factual answers.
- Fallback: if no API key is set, returns a formatted "context-only" answer
  (useful for testing without an API key).
"""

import logging
import os
import time
from typing import List, Tuple

from app.models.schemas import RetrievedChunk

logger = logging.getLogger(__name__)

MODEL_ID = "claude-3-haiku-20240307"  # Fast, cost-efficient for RAG answer synthesis


PROMPT_TEMPLATE = """\
You are a helpful assistant that answers questions strictly based on the provided context.

CONTEXT:
{context}

QUESTION:
{question}

INSTRUCTIONS:
1. Answer ONLY using information from the context above.
2. If the context does not contain enough information to answer the question, say:
   "I could not find a clear answer to this question in the provided documents."
3. Keep your answer concise and accurate.
4. If relevant, cite the source document (filename and chunk index).

ANSWER:"""


class LLMService:
    def __init__(self):
        self.api_key = os.getenv("ANTHROPIC_API_KEY", "")
        self.model = MODEL_ID
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError("anthropic SDK required: pip install anthropic")
        return self._client

    def generate_answer(
        self,
        question: str,
        retrieved_chunks: List[RetrievedChunk],
    ) -> Tuple[str, str, float]:
        """
        Generate an answer from retrieved context.

        Returns:
            (answer_text, model_id_used, generation_latency_ms)
        """
        if not retrieved_chunks:
            return (
                "No relevant documents were found to answer this question. "
                "Please upload relevant documents first.",
                "no-retrieval",
                0.0,
            )

        context = self._build_context(retrieved_chunks)
        prompt = PROMPT_TEMPLATE.format(context=context, question=question)

        # If no API key configured, return a "context dump" fallback
        if not self.api_key or self.api_key in ("", "YOUR_API_KEY_HERE"):
            logger.warning("No ANTHROPIC_API_KEY set. Returning context-only response.")
            fallback = self._fallback_response(question, retrieved_chunks)
            return fallback, "context-only-fallback", 0.0

        start = time.perf_counter()
        try:
            client = self._get_client()
            response = client.messages.create(
                model=self.model,
                max_tokens=1024,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            answer = response.content[0].text.strip()
            elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
            logger.info(f"LLM generated answer in {elapsed_ms}ms using {self.model}")
            return answer, self.model, elapsed_ms

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return (
                f"Answer generation failed: {str(e)}. "
                f"Top retrieved context: {retrieved_chunks[0].text[:300]}...",
                self.model,
                0.0,
            )

    @staticmethod
    def _build_context(chunks: List[RetrievedChunk]) -> str:
        """Format retrieved chunks into a numbered context block."""
        parts = []
        for i, chunk in enumerate(chunks, start=1):
            source = f"[Source {i}: {chunk.filename}, chunk #{chunk.chunk_index}"
            if chunk.page_number:
                source += f", page {chunk.page_number}"
            source += f", similarity={chunk.similarity_score}]"
            parts.append(f"{source}\n{chunk.text}")
        return "\n\n---\n\n".join(parts)

    @staticmethod
    def _fallback_response(question: str, chunks: List[RetrievedChunk]) -> str:
        """Return a structured fallback answer when no LLM API is available."""
        lines = [
            f"[No LLM API configured. Showing top retrieved context for: '{question}']\n",
        ]
        for i, chunk in enumerate(chunks[:3], start=1):
            lines.append(
                f"Chunk {i} (sim={chunk.similarity_score}, file={chunk.filename}):\n"
                f"{chunk.text[:400]}{'...' if len(chunk.text) > 400 else ''}"
            )
        return "\n\n".join(lines)
