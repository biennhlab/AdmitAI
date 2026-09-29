from __future__ import annotations

from collections.abc import Iterable, Sequence
import math
from numbers import Real
import re
from typing import Any

from src.ingestion.chunker import Chunk


RankedItem = tuple[Chunk | str, Real]


def _chunk_id(document: Chunk | str) -> str:
    chunk_id = document.chunk_id if isinstance(document, Chunk) else document
    if not isinstance(chunk_id, str) or not chunk_id.strip():
        raise ValueError("Every ranked document must have a non-empty chunk_id")
    return chunk_id

def _is_coverage_intent(query: str) -> bool:
    query_lower = query.lower()
    phrases = [
        "các ngành", "danh sách", "những ngành nào", "tất cả",
        "có những", "liệt kê", "các phương thức"
    ]
    if any(phrase in query_lower for phrase in phrases):
        return True
        
    keywords = {"liệt kê", "danh sách", "những", "các", "tất cả"}
    query_tokens = set(re.findall(r"\w+", query_lower, flags=re.UNICODE))
    return bool(keywords & query_tokens)


def reciprocal_rank_fusion(
    result_lists: Iterable[Sequence[RankedItem]],
    k: Real = 60,
) -> list[tuple[str, float]]:
    """Fuse ranked lists with ``1 / (k + rank)`` using one-based ranks."""
    if not isinstance(k, Real) or isinstance(k, bool):
        raise TypeError("k must be a real number")
    if not math.isfinite(float(k)):
        raise ValueError("k must be finite")
    if k < 0:
        raise ValueError("k must be greater than or equal to 0")
    if result_lists is None:
        raise TypeError("result_lists cannot be None")

    fused_scores: dict[str, float] = {}
    for result_list in result_lists:
        seen_in_list: set[str] = set()
        unique_rank = 0
        for item in result_list:
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
            unique_rank += 1
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + 1.0 / (
                float(k) + unique_rank
            )

    # chunk_id is a stable tie-break independent of retriever/list insertion order.
    return sorted(fused_scores.items(), key=lambda item: (-item[1], item[0]))


class HybridRetriever:
    def __init__(self, dense_search: Any, sparse_search: Any, parent_chunks: list[dict[str, Any]] | None = None):
        if dense_search is None or not callable(getattr(dense_search, "search", None)):
            raise TypeError("dense_search must provide a search method")
        if sparse_search is None or not callable(getattr(sparse_search, "search", None)):
            raise TypeError("sparse_search must provide a search method")
        self.dense_search = dense_search
        self.sparse_search = sparse_search
        
        self.chunk_by_doc_and_index: dict[tuple[str, int], Chunk] = {}
        self.table_chunks_by_doc_and_index: dict[tuple[str, int], list[Chunk]] = {}
        if hasattr(self.sparse_search, "chunks"):
            for chunk in self.sparse_search.chunks:
                doc_id = chunk.metadata.get("doc_id")
                chunk_index = chunk.metadata.get("chunk_index")
                if doc_id is not None and chunk_index is not None:
                    self.chunk_by_doc_and_index[(str(doc_id), int(chunk_index))] = chunk
                table_index = chunk.metadata.get("table_index")
                if doc_id is not None and table_index is not None:
                    self.table_chunks_by_doc_and_index.setdefault((str(doc_id), int(table_index)), []).append(chunk)
                    
            for key in self.table_chunks_by_doc_and_index:
                self.table_chunks_by_doc_and_index[key].sort(key=lambda c: c.metadata.get("table_part", 0))

        self.parent_by_id: dict[str, Chunk] = {}
        self.parent_by_doc_and_index: dict[tuple[str, int], Chunk] = {}
        self.parent_table_chunks_by_doc_and_index: dict[tuple[str, int], list[Chunk]] = {}
        if parent_chunks:
            self.parent_by_id = {c["chunk_id"]: Chunk(**c) for c in parent_chunks}
            for chunk in self.parent_by_id.values():
                doc_id = chunk.metadata.get("doc_id")
                chunk_index = chunk.metadata.get("chunk_index")
                if doc_id is not None and chunk_index is not None:
                    self.parent_by_doc_and_index[(str(doc_id), int(chunk_index))] = chunk
                table_index = chunk.metadata.get("table_index")
                if doc_id is not None and table_index is not None:
                    self.parent_table_chunks_by_doc_and_index.setdefault((str(doc_id), int(table_index)), []).append(chunk)
                    
            for key in self.parent_table_chunks_by_doc_and_index:
                self.parent_table_chunks_by_doc_and_index[key].sort(key=lambda c: c.metadata.get("table_part", 0))

    def _get_table_parts(self, chunk: Chunk) -> list[Chunk]:
        metadata = chunk.metadata or {}
        doc_id = metadata.get("doc_id")
        table_index = metadata.get("table_index")
        chunk_type = metadata.get("chunk_type")
        
        if not doc_id or table_index is None:
            return []
            
        is_parent = (chunk_type == "parent")
        lookup = self.parent_table_chunks_by_doc_and_index if is_parent else self.table_chunks_by_doc_and_index
        return lookup.get((str(doc_id), int(table_index)), [])

    def _get_neighbors(self, chunk: Chunk, budget: int = 1) -> list[Chunk]:
        neighbors = []
        metadata = chunk.metadata or {}
        doc_id = metadata.get("doc_id")
        chunk_index = metadata.get("chunk_index")
        heading_path = tuple(metadata.get("heading_path") or [])
        chunk_type = metadata.get("chunk_type")

        if not doc_id or chunk_index is None:
            return neighbors
            
        chunk_index = int(chunk_index)
        is_parent = (chunk_type == "parent")
        lookup = self.parent_by_doc_and_index if is_parent else self.chunk_by_doc_and_index

        for offset in range(1, budget + 1):
            for sign in (-1, 1):
                neighbor_idx = chunk_index + sign * offset
                neighbor = lookup.get((str(doc_id), neighbor_idx))
                
                if neighbor:
                    n_heading_path = tuple(neighbor.metadata.get("heading_path") or [])
                    if n_heading_path == heading_path:
                        neighbors.append(neighbor)
                        
        seen = set()
        unique_neighbors = []
        for n in neighbors:
            if n.chunk_id not in seen:
                seen.add(n.chunk_id)
                unique_neighbors.append(n)
                
        return unique_neighbors

    def search(self, query: str, top_k: int = 20) -> list[tuple[Chunk, float]]:
        if not isinstance(top_k, int) or isinstance(top_k, bool):
            raise TypeError("top_k must be an integer")
        if top_k <= 0:
            return []

        from src.config import settings
        
        coverage_mode = _is_coverage_intent(query)
        multiplier = settings.RETRIEVAL_CANDIDATE_MULTIPLIER
        if coverage_mode:
            multiplier *= 2
            
        candidate_k = top_k * multiplier

        # Materialize fresh lists so fusion never mutates retriever-owned results.
        dense_results = list(self.dense_search.search(query, top_k=candidate_k))
        sparse_results = list(self.sparse_search.search(query, top_k=candidate_k))

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
        
        candidates: list[tuple[Chunk, float]] = []
        seen_candidates = set()
        for chunk_id, score in fused:
            chunk = chunks_by_id[chunk_id]
            metadata = chunk.metadata or {}
            parent_id = metadata.get("parent_chunk_id")
            
            if parent_id and parent_id in self.parent_by_id:
                candidate_chunk = self.parent_by_id[parent_id]
            else:
                candidate_chunk = chunk
                
            if candidate_chunk.chunk_id not in seen_candidates:
                seen_candidates.add(candidate_chunk.chunk_id)
                candidates.append((candidate_chunk, score))

        round1: list[tuple[Chunk, float]] = []
        round2: list[tuple[Chunk, float]] = []
        seen_groups = set()
        
        for c, s in candidates:
            metadata = c.metadata or {}
            parent_id = metadata.get("parent_chunk_id")
            doc_id = metadata.get("doc_id")
            heading_path = tuple(metadata.get("heading_path") or [])
            
            if parent_id:
                group_id = f"parent:{parent_id}"
            elif doc_id:
                group_id = f"section:{doc_id}:{heading_path}"
            else:
                group_id = f"chunk:{c.chunk_id}"
                
            if group_id not in seen_groups:
                seen_groups.add(group_id)
                round1.append((c, s))
            else:
                round2.append((c, s))
                
        hydrated_results: list[tuple[Chunk, float]] = []
        seen_chunk_ids: set[str] = set()
        
        # We relax the strict top_k limit slightly to top_k * 3 for coverage_mode
        # to allow the ContextAssembler downstream to enforce character budgets exactly.
        max_results = top_k * 3 if coverage_mode else top_k
        
        def add_candidate(c: Chunk, s: float) -> None:
            if c.chunk_id not in seen_chunk_ids and len(hydrated_results) < max_results:
                metadata = c.metadata or {}
                
                if coverage_mode and metadata.get("table_index") is not None:
                    table_parts = self._get_table_parts(c)
                    if table_parts:
                        for part in table_parts:
                            if len(hydrated_results) >= max_results:
                                break
                            if part.chunk_id not in seen_chunk_ids:
                                seen_chunk_ids.add(part.chunk_id)
                                if part.metadata is None: part.metadata = {}
                                part.metadata["expansion_type"] = "exact" if part.chunk_id == c.chunk_id else "table_sibling"
                                hydrated_results.append((part, s - 1e-6))
                        return
                
                seen_chunk_ids.add(c.chunk_id)
                if c.metadata is None: c.metadata = {}
                c.metadata["expansion_type"] = "exact"
                hydrated_results.append((c, s))
                
                if coverage_mode:
                    neighbors = self._get_neighbors(c, budget=2)
                    for n in neighbors:
                        if len(hydrated_results) >= max_results:
                            break
                        if n.chunk_id not in seen_chunk_ids:
                            seen_chunk_ids.add(n.chunk_id)
                            if n.metadata is None: n.metadata = {}
                            n.metadata["expansion_type"] = "neighbor"
                            hydrated_results.append((n, s - 1e-5))
                            
        for c, s in round1:
            if len(hydrated_results) >= max_results:
                break
            add_candidate(c, s)
            
        for c, s in round2:
            if len(hydrated_results) >= max_results:
                break
            add_candidate(c, s)

        return hydrated_results
