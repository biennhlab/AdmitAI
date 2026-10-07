from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Real

import numpy as np
from sentence_transformers import CrossEncoder

from src.config import settings
from src.ingestion.chunker import Chunk
from src.retrieval.document import chunk_search_text


class Reranker:
    """Rerank candidates while preserving their chunks and retrieval scores."""

    def __init__(self) -> None:
        self.model = CrossEncoder(
            settings.RERANKER_MODEL,
            cache_folder=settings.HUGGINGFACE_CACHE_DIR,
        )

    def rerank(
        self,
        query: str,
        candidates: list[tuple[Chunk, float]],
        top_k: int | None = None,
    ) -> list[tuple[Chunk, float, float]]:
        """Return the best candidates; None uses settings.RERANK_TOP_K.

        Order is reranker score descending, retrieval score descending, then
        chunk_id ascending. Model failures propagate without retrieval fallback.
        """
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if not query.strip():
            raise ValueError("query cannot be empty")
        if top_k is None:
            top_k = settings.RERANK_TOP_K
        if not isinstance(top_k, int) or isinstance(top_k, bool):
            raise TypeError("top_k must be an integer or None")
        if top_k <= 0:
            return []
        if not isinstance(candidates, list):
            raise TypeError("candidates must be a list of (Chunk, retrieval_score) pairs")
        if not candidates:
            return []

        for index, candidate in enumerate(candidates):
            message = f"Candidate {index} must be a (Chunk, finite real retrieval_score) pair"
            if not isinstance(candidate, tuple) or len(candidate) != 2:
                raise TypeError(message)
            chunk, retrieval_score = candidate
            if not isinstance(chunk, Chunk):
                raise TypeError(message)
            if not isinstance(chunk.chunk_id, str) or not chunk.chunk_id.strip():
                raise TypeError(f"Candidate {index} must have a non-empty string chunk_id")
            if chunk.metadata is not None and not isinstance(chunk.metadata, Mapping):
                raise TypeError(f"Candidate {index} metadata must be a mapping or None")
            if (
                not isinstance(retrieval_score, Real)
                or isinstance(retrieval_score, bool)
                or not math.isfinite(retrieval_score)
            ):
                raise TypeError(message)

        pairs = [(query, chunk_search_text(chunk)) for chunk, _ in candidates]
        try:
            predicted = self.model.predict(pairs)
        except Exception as exc:
            raise RuntimeError("Reranker inference failed") from exc

        try:
            scores = np.asarray(predicted)
        except (TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError("Reranker must return one finite score per candidate") from exc
        if scores.shape != (len(candidates),):
            raise RuntimeError("Reranker must return one score per candidate")
        if scores.dtype.kind not in "iuf":
            raise RuntimeError("Reranker must return real numeric scores")
        if not np.isfinite(scores).all():
            raise RuntimeError("Reranker returned non-finite scores")

        results = [
            (chunk, retrieval_score, float(score))
            for (chunk, retrieval_score), score in zip(candidates, scores)
        ]
        results.sort(key=lambda item: (-item[2], -item[1], item[0].chunk_id))
        return results[:top_k]
