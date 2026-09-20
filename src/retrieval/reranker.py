from __future__ import annotations

import math
from typing import Any, TypeAlias

from sentence_transformers import CrossEncoder

from src.config import settings
from src.ingestion.chunker import Chunk


RerankedResult: TypeAlias = tuple[Chunk, float, float]


class Reranker:
    """Rerank retrieval candidates without modifying their chunks or scores."""

    def __init__(
        self,
        model_name: str | None = None,
        batch_size: int | None = None,
        *,
        model: Any | None = None,
    ) -> None:
        resolved_model_name = model_name or settings.RERANKER_MODEL
        if not isinstance(resolved_model_name, str) or not resolved_model_name.strip():
            raise ValueError("Reranker model name cannot be empty")
        self.model_name = resolved_model_name.strip()

        resolved_batch_size = settings.RERANK_BATCH_SIZE if batch_size is None else batch_size
        if not isinstance(resolved_batch_size, int) or isinstance(resolved_batch_size, bool):
            raise TypeError("batch_size must be an integer")
        if resolved_batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")
        self.batch_size = resolved_batch_size

        if model is not None and not callable(getattr(model, "predict", None)):
            raise TypeError("model must provide a predict method")
        # Loading happens once per Reranker instance and never inside rerank().
        self.model = model if model is not None else CrossEncoder(self.model_name)

    def rerank(
        self,
        query: str,
        candidates: list[tuple[Chunk, float]],
        top_k: int | None = None,
    ) -> list[RerankedResult]:
        """Score candidates in one batch and return them in deterministic order."""
        if top_k is not None:
            if not isinstance(top_k, int) or isinstance(top_k, bool):
                raise TypeError("top_k must be an integer or None")
            if top_k <= 0:
                return []
        if not candidates:
            return []
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query cannot be empty")

        pairs: list[tuple[str, str]] = []
        for index, candidate in enumerate(candidates):
            try:
                chunk, _retrieval_score = candidate
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    "Each candidate must be a (Chunk, retrieval_score) pair"
                ) from exc
            if not isinstance(chunk, Chunk):
                raise TypeError(f"candidates[{index}] must contain a Chunk")
            pairs.append((query, chunk.content))

        raw_scores = self.model.predict(pairs, batch_size=self.batch_size)
        try:
            scores = list(raw_scores)
        except TypeError as exc:
            raise ValueError("Reranker model must return one score per candidate") from exc
        if len(scores) != len(candidates):
            raise ValueError(
                "Reranker score count mismatch: "
                f"expected {len(candidates)}, got {len(scores)}"
            )

        results: list[RerankedResult] = []
        for index, ((chunk, retrieval_score), raw_score) in enumerate(zip(candidates, scores)):
            try:
                reranker_score = float(raw_score)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Reranker score at index {index} cannot be converted to float"
                ) from exc
            if not math.isfinite(reranker_score):
                raise ValueError(f"Reranker score at index {index} must be finite")
            try:
                retrieval_sort_score = float(retrieval_score)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Retrieval score at index {index} cannot be converted to float"
                ) from exc
            if not math.isfinite(retrieval_sort_score):
                raise ValueError(f"Retrieval score at index {index} must be finite")
            results.append((chunk, retrieval_score, reranker_score))

        results.sort(key=lambda item: (-item[2], -float(item[1]), item[0].chunk_id))
        return results if top_k is None else results[:top_k]
