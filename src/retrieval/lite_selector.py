from __future__ import annotations

import math
import re
from collections.abc import Mapping
from numbers import Real

from src.ingestion.chunker import Chunk
from src.retrieval.document import chunk_search_text
from src.retrieval.domain_signals import QueryScope, assess, DomainDecision, normalize


_STOPWORDS = {
    "bao",
    "các",
    "cho",
    "có",
    "của",
    "được",
    "hiện",
    "không",
    "là",
    "mình",
    "nào",
    "năm",
    "những",
    "nhiêu",
    "ptit",
    "theo",
    "thì",
    "tôi",
    "trong",
    "và",
    "về",
}


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"\w+", text.casefold(), flags=re.UNICODE)
        if len(token) > 1 and token not in _STOPWORDS
    }


class LiteEvidenceSelector:
    """Deterministically select retrieved evidence without loading a neural model.

    The selector filters scope conflicts and reorders existing candidates. It returns the original chunk
    objects and retrieval scores so provenance and downstream citation identity are
    preserved. Answer generation remains the responsibility of ``RAGChain``.
    """

    @staticmethod
    def select_seeds(query: str, ranked: list[tuple[Chunk, float, float]], top_k: int):
        """Cover explicitly numbered methods for overview questions before hydration.

        Repeated children of one method and navigation links otherwise consume
        every seed. Only already retrieved, scope-eligible evidence is considered;
        no method names, source IDs or expected number of methods are supplied.
        """
        from .program_catalog import catalog_seeds, is_program_overview

        if is_program_overview(query):
            return catalog_seeds(query, ranked, top_k)
        text = normalize(query)
        overview = (
            "phuong thuc" in text
            and re.search(r"\b(?:nhung|cac|nao|liet ke|danh sach|bao nhieu|may)\b", text)
            and not re.search(r"\b(?:diem|dieu kien|sat|act|ielts|toefl)\b", text)
        )
        if not overview or top_k <= 0:
            return ranked[:max(0, top_k)]

        def plain(chunk):
            return normalize(re.sub(r"[*_`\\]", "", chunk.content))

        def methods(chunk):
            return set(re.findall(r"\bphuong thuc\s+(\d+)\s*:\s*xet tuyen\b", plain(chunk)))

        selected = []
        seen = set()
        covered = set()

        def add(item):
            selected.append(item)
            seen.add(id(item[0]))
            covered.update(methods(item[0]))

        overview_source = next((item for item in ranked if methods(item[0]) and re.search(
            r"\b\d+\s+phuong thuc\s+(?:tuyen sinh|xet tuyen)\b", plain(item[0])
        )), None)
        if overview_source is not None:
            add(overview_source)
        for item in ranked:
            if len(selected) >= top_k:
                break
            if methods(item[0]) - covered:
                add(item)
        for item in ranked:
            if len(selected) >= top_k:
                break
            if id(item[0]) not in seen:
                add(item)
        return selected

    @staticmethod
    def explain(query: str, chunk: Chunk) -> DomainDecision:
        """Request-local diagnostics; never mutate shared corpus metadata."""
        return assess(QueryScope.parse(query), chunk)

    @staticmethod
    def _score(query_tokens: set[str], chunk: Chunk) -> float:
        if not query_tokens:
            return 0.0
        document_tokens = _tokens(chunk_search_text(chunk))
        overlap = len(query_tokens & document_tokens)
        coverage = overlap / len(query_tokens)
        precision = overlap / max(1, len(document_tokens))
        return coverage + (0.25 * precision)

    def rerank(
        self,
        query: str,
        candidates: list[tuple[Chunk, float]],
        top_k: int | None = None,
    ) -> list[tuple[Chunk, float, float]]:
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if not query.strip():
            raise ValueError("query cannot be empty")
        if top_k is None:
            from src.config import settings

            top_k = settings.RERANK_TOP_K
        if not isinstance(top_k, int) or isinstance(top_k, bool):
            raise TypeError("top_k must be an integer or None")
        if top_k <= 0 or not candidates:
            return []
        if not isinstance(candidates, list):
            raise TypeError("candidates must be a list of (Chunk, retrieval_score) pairs")

        query_tokens = _tokens(query)
        scope = QueryScope.parse(query)
        from src.config import settings
        results: list[tuple[Chunk, float, float]] = []
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
            if retrieval_score < 0:
                raise ValueError("Lite retrieval scores must be non-negative")
            decision = assess(scope, chunk)
            if not decision.eligible:
                continue
            factor = 1.0 + 2.0 * self._score(query_tokens, chunk)
            for _, value in decision.factors:
                factor *= value
            factor = min(settings.LITE_DOMAIN_MAX_FACTOR, max(settings.LITE_DOMAIN_MIN_FACTOR, factor))
            domain_score = float(retrieval_score) * factor
            if not math.isfinite(domain_score):
                raise ValueError("Lite domain score overflow")
            results.append((chunk, retrieval_score, domain_score))

        results.sort(key=lambda item: (-item[2], -item[1], item[0].chunk_id))
        return results[:top_k]
