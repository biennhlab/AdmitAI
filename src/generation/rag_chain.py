from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import logging
import math
from numbers import Real
import re
import time
import unicodedata
from typing import Any, List, Optional

from ..config import settings
from ..query_transform import AbbreviationNormalizer, QueryRewriter
from ..retrieval.document import chunk_search_text
from ..retrieval.lite_selector import LiteEvidenceSelector
from .prompts import SYSTEM_PROMPT, build_rag_prompt, format_citations
from .grounding import RAGGroundingError, logical_claims, validate_answer
from .diagnostics import generation_diagnostics, record_generation
from .llm_client import LLMAuthenticationError, LLMProviderError


FALLBACK_ANSWER = "Mình chưa tìm thấy thông tin này trong dữ liệu tuyển sinh hiện có."
logger = logging.getLogger(__name__)

# Whole-answer allowlist, not a prefix or intent classifier. Every other lite
# answer still goes through the unchanged factual/relational validator.
_PURE_ABSTENTIONS = frozenset(text.casefold() for text in (
    FALLBACK_ANSWER.removesuffix("."),
    "Mình chưa có đủ thông tin trong dữ liệu tuyển sinh hiện có để trả lời câu hỏi này",
    "Mình chưa có đủ bằng chứng trong dữ liệu tuyển sinh hiện có để trả lời câu hỏi này",
))


def _is_safe_abstention(answer: str, raw_answer: str) -> bool:
    """Allow three exact forms after cleanup, with no extra content or controls.

    NFC, case and whitespace differences and one optional final period are
    allowed. The raw output must be plain text or one balanced answer wrapper;
    hidden/internal blocks, malformed/nested wrappers and extra text fail closed.
    """
    if any(unicodedata.category(char).startswith("C") and char not in "\t\r\n" for char in raw_answer):
        return False
    canonical = " ".join(unicodedata.normalize("NFC", answer).split()).casefold()
    if canonical.removesuffix(".") not in _PURE_ABSTENTIONS:
        return False
    if raw_answer.strip() == answer:
        return True
    wrapper = re.fullmatch(
        r"\s*<(?P<tag>assistant_answer|answer|response|final)>\s*"
        r"(?P<body>.*?)\s*</(?P=tag)>\s*",
        raw_answer, flags=re.IGNORECASE | re.DOTALL,
    )
    return wrapper is not None and wrapper["body"].strip() == answer


class RAGRetrievalError(RuntimeError):
    """The retrieval dependency failed while answering a request."""


class RAGRerankingError(RAGRetrievalError):
    """Reranking failed; generation must not proceed with unvalidated evidence."""


@dataclass
class RAGResponse:
    answer: str
    citations: List[dict[str, Any]]
    route_type: str
    answer_mode: str = "generated"


@dataclass
class _PromptChunk:
    """Chunk context passed to the model with one stable citation marker."""

    chunk_id: str
    content: str
    metadata: dict[str, Any]


@dataclass
class _CitationSource:
    """Evidence for one retrieved chunk and its citation marker."""

    identity: str
    members: list[tuple[Any, float]]
    prompt_chunk: _PromptChunk
    retrieval_score: float


@dataclass
class _GenerationInput:
    sources: list[_CitationSource]
    system_prompt: str
    messages: list[dict[str, str]]


class RAGChain:
    """Retrieve, ground generation, and return citations from real metadata."""

    def __init__(
        self,
        retriever: Any,
        llm_client: Any,
        top_k: int = 5,
        min_score: float | None = None,
        min_lexical_coverage: float = 0.34,
        query_rewriter: QueryRewriter | None = None,
        abbreviation_normalizer: AbbreviationNormalizer | None = None,
        reranker: Any | None = None,
    ):
        self.retriever = retriever
        self.llm_client = llm_client
        self.top_k = top_k
        self.min_score = min_score
        self.min_lexical_coverage = min_lexical_coverage
        self.query_rewriter = query_rewriter
        self.reranker = reranker
        self.abbreviation_normalizer = (
            abbreviation_normalizer or AbbreviationNormalizer()
        )

    def _retrieval_query(self, question: str) -> str:
        """Normalize verified abbreviations, then optionally rewrite for search."""
        normalized = self.abbreviation_normalizer.normalize(question)
        if self.query_rewriter is None:
            return normalized
        return self.query_rewriter.rewrite(normalized)

    @staticmethod
    def _log_timing(stage: str, started_at: float, **fields: int | float) -> None:
        duration_ms = (time.perf_counter() - started_at) * 1000
        suffix = "".join(f" {key}={value}" for key, value in fields.items())
        logger.info("rag_timing stage=%s duration_ms=%.3f%s", stage, duration_ms, suffix)

    @contextmanager
    def _timed_stage(self, stage: str):
        started_at = time.perf_counter()
        try:
            yield
        finally:
            self._log_timing(stage, started_at)

    def _assemble_context(
        self, query: str, retrieved: list[tuple[Any, float]]
    ) -> list[tuple[Any, float]]:
        from .assembler import ContextAssembler

        rerank_started_at = time.perf_counter()
        rerank_input = retrieved
        rerank_output = retrieved
        char_fields: dict[str, int | float] = {}
        lite = isinstance(self.reranker, LiteEvidenceSelector)
        try:
            if self.reranker is not None and retrieved:
                candidate_k = settings.RERANK_CANDIDATE_K
                if candidate_k <= 0:
                    raise ValueError("RERANK_CANDIDATE_K must be greater than 0")
                candidates = retrieved if lite else retrieved[:candidate_k]
                rerank_input = candidates
                rerank_output = []
                candidate_chars = [len(chunk_search_text(chunk)) for chunk, _ in candidates]
                char_fields = {
                    "avg_candidate_chars": round(sum(candidate_chars) / len(candidate_chars), 1),
                    "max_candidate_chars": max(candidate_chars),
                }
                limit = len(candidates) if lite else min(max(0, settings.RERANK_TOP_K), len(candidates))
                reranked = self.reranker.rerank(query, candidates, top_k=limit)
                expected = limit
                if not isinstance(reranked, list) or (len(reranked) > expected if lite else len(reranked) != expected):
                    raise ValueError("Reranker returned an invalid result count")
                # Validate membership with multiplicity: no invented chunks, changed
                # retrieval scores, or repeated results absent from the input.
                available = Counter((id(chunk), score) for chunk, score in candidates)
                previous_rank = None
                for item in reranked:
                    if not isinstance(item, tuple) or len(item) != 3:
                        raise TypeError("Reranker results must be (chunk, retrieval_score, reranker_score)")
                    chunk, retrieval_score, reranker_score = item
                    if any(
                        not isinstance(score, Real) or isinstance(score, bool)
                        or not math.isfinite(score)
                        for score in (retrieval_score, reranker_score)
                    ):
                        raise ValueError("Reranker returned an invalid score")
                    key = (id(chunk), retrieval_score)
                    if not available[key]:
                        raise ValueError("Reranker returned a chunk or retrieval score absent from candidates")
                    available[key] -= 1
                    rank = (-reranker_score, -retrieval_score, chunk.chunk_id)
                    if previous_rank is not None and rank < previous_rank:
                        raise ValueError("Reranker returned results out of order")
                    previous_rank = rank
                rerank_output = [(chunk, score) for chunk, score, _ in reranked]
                if lite:
                    seed_limit = min(max(0, self.top_k), max(0, settings.RERANK_TOP_K))
                    seeds = self.reranker.select_seeds(query, reranked, seed_limit)
                    rerank_output = [(chunk, score) for chunk, score, _ in seeds]
                    from ..retrieval.fusion import HybridRetriever
                    from ..retrieval.domain_signals import QueryScope, expansion_allowed

                    if isinstance(self.retriever, HybridRetriever):
                        scope = QueryScope.parse(query)
                        rerank_output = self.retriever.expand(
                            query, rerank_output, top_k=max(seed_limit, 1),
                            scope_guard=lambda seed, candidate: expansion_allowed(scope, seed, candidate),
                        )
                retrieved = rerank_output
        except Exception as exc:
            raise RAGRerankingError("Reranking dependency failed") from exc
        finally:
            self._log_timing(
                "rerank",
                rerank_started_at,
                input_candidates=len(rerank_input),
                output_candidates=len(rerank_output),
                **char_fields,
            )

        assembled = ContextAssembler(max_chars=settings.LITE_CONTEXT_MAX_CHARS if lite else 12000).assemble(retrieved)
        if self.reranker is not None and not lite:
            # Keep the assembler's deduplication, expansion priorities and budget
            # decisions, but present surviving evidence in reranker order.
            positions: dict[int, int] = {}
            for index, (chunk, _) in enumerate(retrieved):
                positions.setdefault(id(chunk), index)
            assembled.sort(key=lambda item: positions[id(item[0])])
        if lite:
            assembled = assembled[:min(max(0, self.top_k), max(0, settings.RERANK_TOP_K))]
        return assembled

    def _prepare_generation(
        self,
        question: str,
        session_history: Optional[List[dict]],
    ) -> _GenerationInput | None:
        if isinstance(self.reranker, LiteEvidenceSelector):
            from ..retrieval.domain_signals import requires_live_availability
            if requires_live_availability(question):
                return None
        query_started_at = time.perf_counter()
        try:
            retrieval_query = self._retrieval_query(question)
        finally:
            self._log_timing("query_transform", query_started_at)

        retrieval_started_at = time.perf_counter()
        retrieved: list[tuple[Any, float]] = []
        try:
            if isinstance(self.reranker, LiteEvidenceSelector):
                from ..retrieval.fusion import HybridRetriever

                candidate_k = max(self.top_k, settings.LITE_CANDIDATE_K)
                if isinstance(self.retriever, HybridRetriever):
                    retrieved = self.retriever.search_candidates(
                        retrieval_query, top_k=candidate_k, k=settings.LITE_RRF_K,
                        weights=(settings.LITE_DENSE_WEIGHT, settings.LITE_SPARSE_WEIGHT),
                        candidate_limit=settings.LITE_MAX_BRANCH_CANDIDATES,
                    )
                else:
                    retrieved = self.retriever.search(retrieval_query, top_k=candidate_k)
            else:
                retrieved = self.retriever.search(retrieval_query, top_k=self.top_k)
        except Exception as exc:
            raise RAGRetrievalError("Retrieval dependency failed") from exc
        finally:
            self._log_timing("retrieval", retrieval_started_at, candidates=len(retrieved))

        if self.min_score is not None:
            retrieved = [(chunk, score) for chunk, score in retrieved if score >= self.min_score]

        ranking_query = question if isinstance(self.reranker, LiteEvidenceSelector) else retrieval_query
        final_retrieved = self._assemble_context(ranking_query, retrieved)
        if not final_retrieved or not self._has_lexical_evidence(question, final_retrieved):
            return None
        if isinstance(self.reranker, LiteEvidenceSelector):
            from ..retrieval.domain_signals import has_named_program_evidence

            if not has_named_program_evidence(question, [chunk for chunk, _ in final_retrieved]):
                return None

        sources = self._select_citation_sources(
            question, final_retrieved, preserve_order=self.reranker is not None
        )
        if isinstance(self.reranker, LiteEvidenceSelector):
            # Include escaped metadata and document wrappers in the context budget.
            while sources and len(format_citations(
                [source.prompt_chunk for source in sources],
                [source.retrieval_score for source in sources],
            )) > settings.LITE_CONTEXT_MAX_CHARS:
                sources.pop()
            if not sources:
                return None
        chunks = [source.prompt_chunk for source in sources]
        scores = [source.retrieval_score for source in sources]
        context = format_citations(chunks, scores)
        messages: list[dict[str, str]] = []
        if session_history:
            messages.extend(session_history)
        messages.extend(build_rag_prompt(chunks, question))
        return _GenerationInput(
            sources=sources,
            system_prompt=SYSTEM_PROMPT.format(context=context),
            messages=messages,
        )

    @staticmethod
    def _evidence_tokens(text: str) -> set[str]:
        stopwords = {
            "ptit", "có", "không", "là", "bao", "nhiêu", "năm", "trong", "của", "và",
            "được", "những", "nào", "cho", "tôi", "mình", "về", "ở", "theo", "thì",
            "khi", "đến", "tại", "một", "các", "hiện", "nay",
        }
        return {
            token for token in re.findall(r"\w+", text.lower(), flags=re.UNICODE)
            if len(token) > 1 and token not in stopwords and not token.isdigit()
        }

    def _has_lexical_evidence(self, question: str, retrieved: list[tuple[Any, float]]) -> bool:
        if isinstance(self.reranker, LiteEvidenceSelector):
            from ..retrieval.domain_signals import codes

            requested_codes = codes(question)
            evidence_codes = frozenset().union(*(
                codes(chunk_search_text(chunk)) for chunk, _ in retrieved
            ))
            if not requested_codes <= evidence_codes:
                return False
        query_tokens = self._evidence_tokens(question)
        if not query_tokens:
            return False
        searchable = []
        for chunk, _ in retrieved:
            metadata = getattr(chunk, "metadata", {}) or {}
            searchable.extend(
                [
                    str(metadata.get("title") or ""),
                    str(metadata.get("section") or ""),
                    str(getattr(chunk, "content", "")),
                ]
            )
        context_tokens = self._evidence_tokens("\n".join(searchable))
        matches = len(query_tokens & context_tokens)
        required = max(1, math.ceil(len(query_tokens) * self.min_lexical_coverage))
        return matches >= required

    @staticmethod
    def _citation_tokens(text: str) -> set[str]:
        """Tokenize content for citation ranking while retaining years and scores."""
        stopwords = {
            "ptit", "có", "không", "là", "bao", "nhiêu", "năm", "trong", "của", "và",
            "được", "những", "nào", "cho", "tôi", "mình", "về", "ở", "theo", "thì",
            "khi", "đến", "tại", "một", "các", "hiện", "nay",
        }
        without_markers = re.sub(r"\[\d+\]", " ", text.casefold())
        return {
            token
            for token in re.findall(r"\w+", without_markers, flags=re.UNICODE)
            if token not in stopwords and (len(token) > 1 or token.isdigit())
        }

    @staticmethod
    def _freeze_identity_value(value: Any) -> Any:
        if isinstance(value, dict):
            return tuple(
                sorted(
                    (str(key), RAGChain._freeze_identity_value(item))
                    for key, item in value.items()
                )
            )
        if isinstance(value, (list, tuple)):
            return tuple(RAGChain._freeze_identity_value(item) for item in value)
        if isinstance(value, set):
            return tuple(
                sorted(repr(RAGChain._freeze_identity_value(item)) for item in value)
            )
        try:
            hash(value)
        except TypeError:
            return repr(value)
        return value

    @classmethod
    def _provenance_identity(cls, chunk: Any) -> tuple[Any, ...]:
        metadata = getattr(chunk, "metadata", {}) or {}
        source = tuple(
            (key, str(metadata.get(key)).strip().casefold())
            for key in ("doc_id", "source_url", "source_file", "raw_file", "parent_doc_id")
            if metadata.get(key) not in (None, "")
        )
        locator_keys = (
            "page_number",
            "section",
            "heading",
            "heading_path",
            "chunk_index",
            "parent_chunk_index",
            "table_index",
            "table_part",
            "row_start",
            "row_end",
        )
        has_locator = any(
            metadata.get(key) not in (None, "", []) for key in locator_keys
        )
        details = tuple(
            (key, cls._freeze_identity_value(metadata.get(key)))
            for key in (
                *locator_keys,
                "admission_year",
                "academic_year",
                "campus",
                "campus_code",
                "campuses",
                "version",
                "content_hash",
                "updated_at",
                "provenance",
                "title",
                "source",
                "source_type",
                "published_at",
                "retrieved_at",
                "chunk_type",
                "parent_chunk_id",
            )
            if metadata.get(key) not in (None, "", [])
        )
        chunk_id = str(getattr(chunk, "chunk_id", "") or "").strip()
        if source and has_locator:
            return "source", source, details
        if chunk_id:
            return "chunk", source, chunk_id, details
        return ()

    @classmethod
    def _source_identity(cls, chunk: Any) -> str:
        """Return a stable opaque provenance identity without inventing a URL."""
        identity = cls._provenance_identity(chunk)
        if not identity:
            return ""
        return hashlib.sha256(repr(identity).encode("utf-8")).hexdigest()[:24]

    @classmethod
    def _evidence_identity(cls, chunk: Any, position: int) -> str:
        provenance = cls._source_identity(chunk) or f"position:{position}"
        content = " ".join(str(getattr(chunk, "content", "")).split()).casefold()
        return hashlib.sha256(repr((provenance, content)).encode("utf-8")).hexdigest()

    @classmethod
    def _select_citation_sources(
        cls,
        question: str,
        retrieved: list[tuple[Any, float]],
        limit: int | None = None,
        *,
        preserve_order: bool = False,
    ) -> list[_CitationSource]:
        """Keep reranked order or rank by question support for legacy callers."""
        query_tokens = cls._citation_tokens(question)
        grouped: dict[str, list[tuple[Any, float, int]]] = {}
        chunk_id_evidence: dict[str, tuple[str, int]] = {}
        for position, (chunk, score) in enumerate(retrieved):
            identity = cls._evidence_identity(chunk, position)
            chunk_id = str(getattr(chunk, "chunk_id", "") or "").strip()
            if chunk_id:
                previous = chunk_id_evidence.get(chunk_id)
                if previous is not None and previous[0] != identity:
                    collision_id = hashlib.sha256(chunk_id.encode("utf-8")).hexdigest()[:12]
                    logger.warning(
                        "citation_identity_collision kind=chunk_id id_hash=%s first_position=%d position=%d",
                        collision_id,
                        previous[1],
                        position,
                    )
                else:
                    chunk_id_evidence[chunk_id] = (identity, position)
            grouped.setdefault(identity, []).append((chunk, float(score), position))

        ranked_groups: list[tuple[tuple[float, ...], str, list[tuple[Any, float, int]]]] = []
        for identity, members in grouped.items():
            combined_tokens: set[str] = set()
            best_chunk_overlap = 0
            best_score = max(score for _, score, _ in members)
            first_position = min(position for _, _, position in members)
            for chunk, _, _ in members:
                metadata = getattr(chunk, "metadata", {}) or {}
                searchable = "\n".join(
                    (
                        str(metadata.get("title") or metadata.get("source") or ""),
                        str(metadata.get("section") or metadata.get("heading") or ""),
                        str(getattr(chunk, "content", "")),
                    )
                )
                tokens = cls._citation_tokens(searchable)
                combined_tokens.update(tokens)
                best_chunk_overlap = max(best_chunk_overlap, len(query_tokens & tokens))

            source_coverage = len(query_tokens & combined_tokens)
            rank = (
                float(source_coverage),
                float(best_chunk_overlap),
                best_score,
                float(-first_position),
            )
            if preserve_order:
                rank = (float(-first_position),)
            ranked_groups.append((rank, identity, members))

        ranked_groups.sort(key=lambda item: item[0], reverse=True)
        selected: list[_CitationSource] = []
        for _, identity, members in ranked_groups[:limit]:
            if preserve_order:
                representative, representative_score, _ = min(
                    members, key=lambda item: item[2]
                )
            else:
                def member_rank(item: tuple[Any, float, int]) -> tuple[int, float, int]:
                    chunk, score, position = item
                    metadata = getattr(chunk, "metadata", {}) or {}
                    searchable = "\n".join(
                        (
                            str(metadata.get("title") or metadata.get("source") or ""),
                            str(metadata.get("section") or metadata.get("heading") or ""),
                            str(getattr(chunk, "content", "")),
                        )
                    )
                    overlap = len(query_tokens & cls._citation_tokens(searchable))
                    return overlap, score, -position

                representative, representative_score, _ = max(
                    members, key=member_rank
                )
            prompt_chunk = _PromptChunk(
                chunk_id=str(getattr(representative, "chunk_id", "")),
                content=str(getattr(representative, "content", "")),
                metadata=dict(getattr(representative, "metadata", {}) or {}),
            )
            selected.append(
                _CitationSource(
                    identity=identity,
                    members=[(chunk, score) for chunk, score, _ in members],
                    prompt_chunk=prompt_chunk,
                    retrieval_score=representative_score,
                )
            )
        return selected

    @staticmethod
    def _claim_for_marker(answer: str, marker: int) -> str:
        return " ".join(claim.text for claim in logical_claims(answer) if marker in claim.markers)

    @staticmethod
    def _specific_excerpt_tokens(text: str) -> set[str]:
        without_markers = re.sub(r"\[\d+\]", " ", text)
        tokens = re.findall(r"[\w./%+-]+", without_markers, flags=re.UNICODE)
        return {
            token.casefold()
            for token in tokens
            if any(character.isdigit() for character in token)
            or (
                len(token) >= 2
                and any(character.isalpha() for character in token)
                and token.upper() == token
            )
        }

    @classmethod
    def _best_excerpt(cls, content: str, target_text: str, max_chars: int = 360) -> str:
        """Return the source sentence, list item, or table row that best supports the claim."""
        target_tokens = cls._citation_tokens(target_text)
        target_specific = cls._specific_excerpt_tokens(target_text)
        candidates = [
            " ".join(part.split())
            for part in re.split(r"(?<=[.!?;])\s+|\n+", content)
            if part.strip() and not re.fullmatch(r"\s*\|?(?:\s*:?-{3,}:?\s*\|?)+\s*", part)
        ]
        if not candidates:
            return ""

        def candidate_rank(candidate: str) -> tuple[int, int, float, int]:
            tokens = cls._citation_tokens(candidate)
            overlap = len(target_tokens & tokens)
            specific_overlap = len(
                target_specific & cls._specific_excerpt_tokens(candidate)
            )
            density = overlap / max(1, len(tokens))
            return specific_overlap, overlap, density, -abs(len(candidate) - 220)

        excerpt = max(candidates, key=candidate_rank)
        best_specific, best_overlap, _, _ = candidate_rank(excerpt)
        if best_overlap == 0 or (target_specific and best_specific == 0):
            return ""
        if len(excerpt) <= max_chars:
            return excerpt

        folded = excerpt.casefold()
        positions = [folded.find(token) for token in target_tokens if folded.find(token) >= 0]
        anchor = min(positions) if positions else 0
        start = max(0, min(anchor - max_chars // 3, len(excerpt) - max_chars))
        end = min(len(excerpt), start + max_chars)
        trimmed = excerpt[start:end].strip()
        if start > 0:
            trimmed = f"…{trimmed}"
        if end < len(excerpt):
            trimmed = f"{trimmed}…"
        return trimmed

    @classmethod
    def _citation(
        cls,
        chunk: Any,
        score: float,
        *,
        marker: int | None = None,
        snippet: str | None = None,
    ) -> dict[str, Any]:
        metadata = getattr(chunk, "metadata", {}) or {}
        content = getattr(chunk, "content", "")
        citation = {
            "doc_id": str(metadata.get("doc_id") or ""),
            "title": str(metadata.get("title") or metadata.get("source") or f"Chunk {chunk.chunk_id}"),
            "source": str(metadata.get("title") or metadata.get("source") or f"Chunk {chunk.chunk_id}"),
            "source_url": str(metadata.get("source_url") or ""),
            "source_type": str(metadata.get("source_type") or ""),
            "published_at": str(metadata.get("published_at") or ""),
            "retrieved_at": str(metadata.get("retrieved_at") or ""),
            "page": metadata.get("page_number"),
            "section": metadata.get("section"),
            "snippet": snippet if snippet is not None else " ".join(content.split())[:280],
            "score": round(float(score), 6),
            "chunk_id": str(getattr(chunk, "chunk_id", "")),
        }
        if marker is not None:
            citation["marker"] = marker
        return citation

    @classmethod
    def _citations_for_answer(
        cls,
        sources: list[_CitationSource],
        answer: str,
        question: str,
    ) -> list[dict[str, Any]]:
        answer_markers = [int(value) for value in re.findall(r"\[(\d+)\]", answer)]
        valid_markers = {marker for marker in answer_markers if 1 <= marker <= len(sources)}
        invalid_count = sum(marker < 1 or marker > len(sources) for marker in answer_markers)
        if not answer_markers:
            logger.warning(
                "citation_marker_validation status=missing source_count=%d",
                len(sources),
            )
            return []
        if invalid_count:
            logger.warning(
                "citation_marker_validation status=%s source_count=%d valid_count=%d invalid_count=%d",
                "mixed" if valid_markers else "invalid",
                len(sources),
                len(valid_markers),
                invalid_count,
            )
        # Structural heading markers are checked for validity, but a source
        # must support an actual logical claim to appear in the public list.
        used_markers = {marker for claim in logical_claims(answer) for marker in claim.markers}
        markers = sorted(valid_markers & used_markers)
        citations: list[dict[str, Any]] = []
        for marker in markers:
            source = sources[marker - 1]
            claim = cls._claim_for_marker(answer, marker)
            target_text = f"{claim}\n{question}"
            chunk = source.prompt_chunk
            snippet = cls._best_excerpt(str(getattr(chunk, "content", "")), target_text)
            citations.append(
                cls._citation(
                    chunk,
                    source.retrieval_score,
                    marker=marker,
                    snippet=snippet,
                )
            )
        return citations

    @staticmethod
    def _strip_leading_model_blocks(answer: str, *, wait_for_incomplete: bool = False) -> str | None:
        """Remove provider control blocks without altering user-facing Markdown."""
        cleaned = answer.lstrip()
        wrapper_names = "assistant_answer|answer|response|final"
        hidden_names = "thought|analysis|reasoning"
        known_openers = (
            "<assistant_answer>",
            "<answer>",
            "<response>",
            "<final>",
            "<thought>",
            "<analysis>",
            "<reasoning>",
        )

        while cleaned.startswith("<"):
            wrapper = re.match(
                rf"\A<(?:{wrapper_names})>\s*",
                cleaned,
                flags=re.IGNORECASE,
            )
            if wrapper:
                cleaned = cleaned[wrapper.end():].lstrip()
                continue

            hidden = re.match(
                rf"\A<(?P<tag>{hidden_names})>\s*",
                cleaned,
                flags=re.IGNORECASE,
            )
            if hidden:
                closing = re.search(
                    rf"</{re.escape(hidden.group('tag'))}>\s*",
                    cleaned[hidden.end():],
                    flags=re.IGNORECASE,
                )
                if closing is None:
                    return None if wait_for_incomplete else ""
                cleaned = cleaned[hidden.end() + closing.end():].lstrip()
                continue

            if wait_for_incomplete and any(
                opener.startswith(cleaned.casefold()) for opener in known_openers
            ):
                return None
            break

        return cleaned

    @classmethod
    def _clean_answer(cls, answer: str) -> str:
        cleaned = cls._strip_leading_model_blocks(answer)
        if cleaned is None:
            return ""
        wrapper_names = "assistant_answer|answer|response|final"
        cleaned = re.sub(
            rf"\s*</(?:{wrapper_names})>\Z",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        return cleaned.strip()

    @staticmethod
    def _public_answer(answer: str) -> str:
        """Remove internal attribution only after grounding and citation selection.

        Horizontal whitespace before a marker belongs to the marker. Retain
        newlines, list indentation, table delimiters and Markdown hard breaks.
        """
        return re.sub(r"[ \t]*\[\d+\]", "", answer).strip()

    def answer_stream(self, question: str, session_history: Optional[List[dict]] = None) -> Any:
        # Every mode buffers the same one-call generation and grounding path.
        # No provider token or internal marker reaches SSE before validation.
        result = self.answer(question, session_history)
        metadata = {"type": "metadata", "citations": result.citations, "route_type": result.route_type}
        if result.answer_mode != "generated":
            metadata["answer_mode"] = result.answer_mode
        if result.route_type == "out_of_scope":
            yield metadata
        yield {"type": "chunk", "text": result.answer}
        if result.route_type != "out_of_scope":
            yield metadata

    def answer(self, question: str, session_history: Optional[List[dict]] = None) -> RAGResponse:
        with generation_diagnostics(question or ""), self._timed_stage("total"):
            if not question or not question.strip():
                record_generation([], "", "no_generation")
                return RAGResponse(FALLBACK_ANSWER, [], "out_of_scope")

            try:
                prepared = self._prepare_generation(question, session_history)
            except Exception as exc:
                record_generation([], "", "evidence_error", error=exc)
                raise
            if prepared is None:
                record_generation([], "", "no_evidence")
                return RAGResponse(FALLBACK_ANSWER, [], "out_of_scope")

            answer = ""
            classification = "provider_error"
            try:
                with self._timed_stage("generation"):
                    try:
                        raw_answer = self.llm_client.generate(
                            system_prompt=prepared.system_prompt,
                            messages=prepared.messages,
                        )
                    except LLMProviderError as exc:
                        recovered = None if isinstance(exc, LLMAuthenticationError) else self._evidence_recovery(question, prepared)
                        if recovered is not None:
                            record_generation(prepared.sources, "", "provider_error", error=exc)
                            used_markers = " ".join(f"[{c['marker']}]" for c in recovered.citations)
                            record_generation(prepared.sources, used_markers, "verified_evidence_recovery")
                            return recovered
                        raise
                    classification = "invalid_generation"
                    if not isinstance(raw_answer, str):
                        raise RAGGroundingError(
                            "Generation did not return text", reason="invalid_generation",
                            source_count=len(prepared.sources),
                        )
                    answer = self._clean_answer(raw_answer)

                if _is_safe_abstention(answer, raw_answer):
                    record_generation(prepared.sources, answer, "safe_abstention")
                    return RAGResponse(FALLBACK_ANSWER, [], "out_of_scope")
                classification = "factual_candidate"
                if not answer.strip():
                    classification = "empty_generation"
                elif re.match(r"(?:mình )?(?:chưa|không) (?:tìm thấy|có đủ|thể xác minh)", answer.casefold()):
                    classification = "abstention_candidate"
                envelope = re.sub(r"\A\s*<(thought|analysis|reasoning)>.*?</\1>\s*", "", raw_answer,
                                  flags=re.IGNORECASE | re.DOTALL)
                if re.match(r"\s*<(?:assistant_answer|answer|response|final)>", envelope, re.IGNORECASE):
                    if not re.fullmatch(r"\s*<(?P<tag>assistant_answer|answer|response|final)>.*?</(?P=tag)>\s*",
                                        envelope, re.IGNORECASE | re.DOTALL):
                        classification = "invalid_output"
                        raise RAGGroundingError("Malformed answer wrapper", reason="invalid_generation",
                                                source_count=len(prepared.sources))
                # Cleanup may remove known outer wrappers, but residual controls,
                # invented links and separate source footers are not factual text.
                if re.search(
                    r"<[^>]*>|https?://|\[[^\]]*\]\([^)]*\)|(?m:^\s*\[[^\]]+\]:\s*\S)"
                    r"|\\\[\d+\\?\]|\[\d+\\\]"
                    r"|(?im:^\s*(?:assistant|system|analysis|reasoning|thought)\s*:)"
                    r"|(?im:^\s*(?:#+\s*)?(?:nguồn tham khảo|sources|references)\s*:?(?:\s|$))",
                    answer,
                ):
                    classification = "invalid_output"
                    raise RAGGroundingError(
                        "Generation contains internal controls or a source footer/link",
                        reason="invalid_generation", source_count=len(prepared.sources),
                    )
                validate_answer(answer, [source.prompt_chunk for source in prepared.sources])
                citations = self._citations_for_answer(prepared.sources, answer, question)
                record_generation(prepared.sources, answer, classification)
                return RAGResponse(self._public_answer(answer), citations, "general")
            except Exception as exc:
                record_generation(prepared.sources, answer, classification, error=exc)
                if isinstance(exc, RAGGroundingError) and exc.reason == "missing_citation":
                    # Discard the model answer; never attach guessed markers to
                    # its uncited claims. A separate source-only answer must
                    # independently pass the same factual/relational validator.
                    recovered = self._evidence_recovery(question, prepared)
                    if recovered is not None:
                        used_markers = " ".join(f"[{c['marker']}]" for c in recovered.citations)
                        record_generation(prepared.sources, used_markers, "verified_citation_format_recovery")
                        return recovered
                raise

    def _evidence_recovery(self, question: str, prepared: _GenerationInput) -> RAGResponse | None:
        from .evidence_fallback import admission_methods

        try:
            chunks = [source.prompt_chunk for source in prepared.sources]
            answer = admission_methods(question, chunks)
            if answer is None:
                return None
            validate_answer(answer, chunks)
            citations = self._citations_for_answer(prepared.sources, answer, question)
            return RAGResponse(self._public_answer(answer), citations, "general", "evidence")
        except Exception as exc:
            # A failed recovery never masks the original generation failure or
            # permits an unverified excerpt through either transport.
            logger.warning("Evidence recovery declined error_class=%s", type(exc).__name__)
            return None
