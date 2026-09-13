from __future__ import annotations

from collections.abc import Iterable, Sequence
from numbers import Real
from typing import Any

from src.ingestion.chunker import Chunk


RankedItem = tuple[Chunk | str, Real]


def _chunk_id(document: Chunk | str) -> str:
    chunk_id = document.chunk_id if isinstance(document, Chunk) else document
    if not isinstance(chunk_id, str) or not chunk_id.strip():
        raise ValueError("Every ranked document must have a non-empty chunk_id")
    return chunk_id


def reciprocal_rank_fusion(
    result_lists: Iterable[Sequence[RankedItem]],
    k: Real = 60,
) -> list[tuple[str, float]]:
    """Fuse ranked lists with ``1 / (k + rank)`` using one-based ranks."""
    if not isinstance(k, Real) or isinstance(k, bool):
        raise TypeError("k must be a real number")
    if k < 0:
        raise ValueError("k must be greater than or equal to 0")
    if result_lists is None:
        raise TypeError("result_lists cannot be None")

    fused_scores: dict[str, float] = {}
    for result_list in result_lists:
        seen_in_list: set[str] = set()
        for rank, item in enumerate(result_list, start=1):
            try:
                document, _original_score = item
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    "Each result must be a (Chunk or chunk_id, score) pair"
                ) from exc
            chunk_id = _chunk_id(document)
            if chunk_id in seen_in_list:
                continue
            seen_in_list.add(chunk_id)
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + 1.0 / (
                float(k) + rank
            )

    # chunk_id is a stable tie-break independent of retriever/list insertion order.
    return sorted(fused_scores.items(), key=lambda item: (-item[1], item[0]))


class HybridRetriever:
    def __init__(self, dense_search: Any, sparse_search: Any):
        if dense_search is None or not callable(getattr(dense_search, "search", None)):
            raise TypeError("dense_search must provide a search method")
        if sparse_search is None or not callable(getattr(sparse_search, "search", None)):
            raise TypeError("sparse_search must provide a search method")
        self.dense_search = dense_search
        self.sparse_search = sparse_search

    def search(self, query: str, top_k: int = 20) -> list[tuple[Chunk, float]]:
        if not isinstance(top_k, int) or isinstance(top_k, bool):
            raise TypeError("top_k must be an integer")
        if top_k <= 0:
            return []

        # Materialize fresh lists so fusion never mutates retriever-owned results.
        dense_results = list(self.dense_search.search(query, top_k=top_k))
        sparse_results = list(self.sparse_search.search(query, top_k=top_k))

        chunks_by_id: dict[str, Chunk] = {}
        for result in [*dense_results, *sparse_results]:
            try:
                chunk, _score = result
            except (TypeError, ValueError) as exc:
                raise TypeError("Retriever results must be (Chunk, score) pairs") from exc
            if not isinstance(chunk, Chunk):
                raise TypeError("Hybrid retrievers must return Chunk objects")
            chunks_by_id.setdefault(_chunk_id(chunk), chunk)

        fused = reciprocal_rank_fusion([dense_results, sparse_results])
        return [
            (chunks_by_id[chunk_id], score)
            for chunk_id, score in fused[:top_k]
        ]
