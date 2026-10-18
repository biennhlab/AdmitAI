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
from typing import Any, List, Optional

from ..config import settings
from ..query_transform import AbbreviationNormalizer, QueryRewriter
from ..retrieval.document import chunk_search_text
from .prompts import SYSTEM_PROMPT, build_rag_prompt, format_citations


FALLBACK_ANSWER = "Mình chưa tìm thấy thông tin này trong dữ liệu tuyển sinh hiện có."
logger = logging.getLogger(__name__)


class RAGRetrievalError(RuntimeError):
    """The retrieval dependency failed while answering a request."""


class RAGRerankingError(RAGRetrievalError):
    """Reranking failed; generation must not proceed with unvalidated evidence."""


@dataclass
class RAGResponse:
    answer: str
    citations: List[dict[str, Any]]
    route_type: str


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
        try:
            if self.reranker is not None and retrieved:
                candidate_k = settings.RERANK_CANDIDATE_K
                if candidate_k <= 0:
                    raise ValueError("RERANK_CANDIDATE_K must be greater than 0")
                candidates = retrieved[:candidate_k]
                rerank_input = candidates
                rerank_output = []
                candidate_chars = [len(chunk_search_text(chunk)) for chunk, _ in candidates]
                char_fields = {
                    "avg_candidate_chars": round(sum(candidate_chars) / len(candidate_chars), 1),
                    "max_candidate_chars": max(candidate_chars),
                }
                limit = min(max(0, settings.RERANK_TOP_K), len(candidates))
                reranked = self.reranker.rerank(query, candidates, top_k=limit)
                expected = limit
                if not isinstance(reranked, list) or len(reranked) != expected:
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

        assembled = ContextAssembler(max_chars=12000).assemble(retrieved)
        if self.reranker is not None:
            # Keep the assembler's deduplication, expansion priorities and budget
            # decisions, but present surviving evidence in reranker order.
            positions: dict[int, int] = {}
            for index, (chunk, _) in enumerate(retrieved):
                positions.setdefault(id(chunk), index)
            assembled.sort(key=lambda item: positions[id(item[0])])
        return assembled

    def _prepare_generation(
        self,
        question: str,
        session_history: Optional[List[dict]],
    ) -> _GenerationInput | None:
        query_started_at = time.perf_counter()
        try:
            retrieval_query = self._retrieval_query(question)
        finally:
            self._log_timing("query_transform", query_started_at)

        retrieval_started_at = time.perf_counter()
        retrieved: list[tuple[Any, float]] = []
        try:
            retrieved = self.retriever.search(retrieval_query, top_k=self.top_k)
        except Exception as exc:
            raise RAGRetrievalError("Retrieval dependency failed") from exc
        finally:
            self._log_timing("retrieval", retrieval_started_at, candidates=len(retrieved))

        if self.min_score is not None:
            retrieved = [(chunk, score) for chunk, score in retrieved if score >= self.min_score]

        final_retrieved = self._assemble_context(retrieval_query, retrieved)
        if not final_retrieved or not self._has_lexical_evidence(question, final_retrieved):
            return None

        sources = self._select_citation_sources(
            question, final_retrieved, preserve_order=self.reranker is not None
        )
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
        has_locator = any(
            metadata.get(key) not in (None, "", [])
            for key in ("page_number", "section", "heading", "heading_path")
        )
        details = tuple(
            (key, cls._freeze_identity_value(metadata.get(key)))
            for key in (
                "page_number",
                "section",
                "heading",
                "heading_path",
                "admission_year",
                "academic_year",
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
            representative = min(members, key=lambda item: item[2])[0]
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
                    retrieval_score=max(score for _, score, _ in members),
                )
            )
        return selected

    @staticmethod
    def _claim_for_marker(answer: str, marker: int) -> str:
        marker_text = f"[{marker}]"
        units = re.split(r"(?<=[.!?])\s+|\n+", answer)
        matching = [unit.strip() for unit in units if marker_text in unit]
        return " ".join(matching)

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
        markers = sorted(valid_markers)
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

    def answer_stream(self, question: str, session_history: Optional[List[dict]] = None) -> Any:
        with self._timed_stage("total"):
            if not question or not question.strip():
                yield {"type": "metadata", "citations": [], "route_type": "out_of_scope"}
                yield {"type": "chunk", "text": FALLBACK_ANSWER}
                return

            prepared = self._prepare_generation(question, session_history)
            if prepared is None:
                yield {"type": "metadata", "citations": [], "route_type": "out_of_scope"}
                yield {"type": "chunk", "text": FALLBACK_ANSWER}
                return

            accumulator = ""
            answer_parts: list[str] = []
            started_streaming = False
            close_pattern = re.compile(r"</(?:assistant_answer|answer|response|final)>.*", re.IGNORECASE | re.DOTALL)

            with self._timed_stage("generation"):
                generator = self.llm_client.generate_stream(
                    system_prompt=prepared.system_prompt,
                    messages=prepared.messages,
                )
                for chunk_text in generator:
                    accumulator += chunk_text
                    if not started_streaming:
                        visible = self._strip_leading_model_blocks(
                            accumulator,
                            wait_for_incomplete=True,
                        )
                        if visible is None:
                            continue
                        accumulator = visible
                        if not accumulator:
                            continue
                        started_streaming = True

                    if started_streaming:
                        closing = close_pattern.search(accumulator)
                        if closing:
                            emit = accumulator[:closing.start()]
                            accumulator = ""
                            if emit:
                                answer_parts.append(emit)
                                yield {"type": "chunk", "text": emit}
                            break
                        if len(accumulator) > 30:
                            emit = accumulator[:-30]
                            accumulator = accumulator[-30:]
                            answer_parts.append(emit)
                            yield {"type": "chunk", "text": emit}

                if started_streaming and accumulator:
                    accumulator = close_pattern.sub("", accumulator)
                    if accumulator:
                        answer_parts.append(accumulator)
                        yield {"type": "chunk", "text": accumulator}

            citations = self._citations_for_answer(
                prepared.sources, "".join(answer_parts), question
            )
            yield {"type": "metadata", "citations": citations, "route_type": "general"}

    def answer(self, question: str, session_history: Optional[List[dict]] = None) -> RAGResponse:
        with self._timed_stage("total"):
            if not question or not question.strip():
                return RAGResponse(FALLBACK_ANSWER, [], "out_of_scope")

            prepared = self._prepare_generation(question, session_history)
            if prepared is None:
                return RAGResponse(FALLBACK_ANSWER, [], "out_of_scope")

            with self._timed_stage("generation"):
                answer = self._clean_answer(
                    self.llm_client.generate(
                        system_prompt=prepared.system_prompt,
                        messages=prepared.messages,
                    )
                )

            # The marker field maps each returned citation to its numbered document,
            # including when the answer cites only a subset of the context.
            citations = self._citations_for_answer(prepared.sources, answer, question)
            return RAGResponse(answer, citations, "general")
