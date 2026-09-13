from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from rank_bm25 import BM25Okapi

from src.ingestion.chunker import Chunk


def tokenize(text: str) -> list[str]:
    """Tokenize consistently using the phase-3 whitespace strategy."""
    return text.casefold().split()


class BM25Search:
    """In-memory BM25 index with idempotent upserts keyed by ``chunk_id``."""

    def __init__(self):
        self.chunks: list[Chunk] = []
        self.tokenized_corpus: list[list[str]] = []
        self.bm25: BM25Okapi | None = None
        self._token_sets: list[set[str]] = []

    @staticmethod
    def _validated_chunks(chunks: Iterable[Chunk]) -> list[Chunk]:
        if chunks is None:
            raise TypeError("chunks cannot be None")
        try:
            values = list(chunks)
        except TypeError as exc:
            raise TypeError("chunks must be an iterable of Chunk objects") from exc

        for index, chunk in enumerate(values):
            if not isinstance(chunk, Chunk):
                raise TypeError(f"chunks[{index}] must be a Chunk")
            if not isinstance(chunk.chunk_id, str) or not chunk.chunk_id.strip():
                raise ValueError(f"chunks[{index}].chunk_id must be a non-empty string")
            if not isinstance(chunk.content, str) or not chunk.content.strip():
                raise ValueError(f"chunks[{index}].content must be a non-empty string")
        return values

    @staticmethod
    def _deduplicate(chunks: Iterable[Chunk]) -> list[Chunk]:
        # Assignment updates the value while retaining the ID's first position.
        # This gives deterministic last-write-wins upserts.
        by_id: dict[str, Chunk] = {}
        for chunk in chunks:
            by_id[chunk.chunk_id] = chunk
        return list(by_id.values())

    def index(self, chunks: list[Chunk]) -> None:
        """Upsert chunks by ID and immediately rebuild the BM25 corpus."""
        incoming = self._validated_chunks(chunks)
        if not incoming:
            return

        # Validate the complete new state before mutating the live index.
        updated = self._deduplicate([*self.chunks, *incoming])
        self._validated_chunks(updated)
        self.chunks = updated
        self.rebuild()

    def search(self, query: str, top_k: int = 20) -> list[tuple[Chunk, float]]:
        """Return BM25-ranked chunks that share at least one query token."""
        if not isinstance(query, str) or not query.strip():
            return []
        if not isinstance(top_k, int) or isinstance(top_k, bool):
            raise TypeError("top_k must be an integer")
        if top_k <= 0 or self.bm25 is None or not self.chunks:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        query_token_set = set(query_tokens)
        matching_indices = [
            index
            for index, document_tokens in enumerate(self._token_sets)
            if query_token_set.intersection(document_tokens)
        ]
        if not matching_indices:
            return []

        scores = np.asarray(self.bm25.get_scores(query_tokens), dtype=float)
        if scores.ndim != 1 or scores.shape[0] != len(self.chunks):
            raise RuntimeError("BM25 scores are not aligned with the indexed chunks")
        if not np.isfinite(scores).all():
            raise RuntimeError("BM25 returned a non-finite relevance score")

        # Corpus order is the deterministic secondary key for equal BM25 scores.
        matching_indices.sort(key=lambda index: (-float(scores[index]), index))
        limit = min(top_k, len(matching_indices))
        return [
            (self.chunks[index], float(scores[index]))
            for index in matching_indices[:limit]
        ]

    def rebuild(self) -> None:
        """Rebuild BM25 after callers add, remove, or replace ``self.chunks``."""
        validated = self._validated_chunks(self.chunks)
        self.chunks = self._deduplicate(validated)
        self.tokenized_corpus = [tokenize(chunk.content) for chunk in self.chunks]
        self._token_sets = [set(tokens) for tokens in self.tokenized_corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus) if self.chunks else None
