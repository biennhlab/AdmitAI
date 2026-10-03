from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, List, Optional

from ..query_transform import AbbreviationNormalizer, QueryRewriter
from .prompts import SYSTEM_PROMPT, build_rag_prompt, format_citations


FALLBACK_ANSWER = "Mình chưa tìm thấy thông tin này trong dữ liệu tuyển sinh hiện có."


class RAGRetrievalError(RuntimeError):
    """The retrieval dependency failed while answering a request."""


@dataclass
class RAGResponse:
    answer: str
    citations: List[dict[str, Any]]
    route_type: str


@dataclass
class _PromptChunk:
    """Source-level context passed to the model with one stable citation marker."""

    chunk_id: str
    content: str
    metadata: dict[str, Any]


@dataclass
class _CitationSource:
    """Retrieved chunks that resolve to the same original source."""

    identity: str
    members: list[tuple[Any, float]]
    prompt_chunk: _PromptChunk
    retrieval_score: float


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
    ):
        self.retriever = retriever
        self.llm_client = llm_client
        self.top_k = top_k
        self.min_score = min_score
        self.min_lexical_coverage = min_lexical_coverage
        self.query_rewriter = query_rewriter
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
    def _source_identity(chunk: Any) -> str:
        """Return a stable internal identity without inventing a public URL."""
        metadata = getattr(chunk, "metadata", {}) or {}
        for key in ("source_url", "raw_file", "parent_doc_id", "doc_id", "source_file"):
            value = str(metadata.get(key) or "").strip()
            if value:
                return f"{key}:{value.casefold()}"
        return f"chunk:{getattr(chunk, 'chunk_id', '')}"

    @classmethod
    def _select_citation_sources(
        cls,
        question: str,
        retrieved: list[tuple[Any, float]],
        limit: int = 2,
    ) -> list[_CitationSource]:
        """Rank distinct sources by question support, using retrieval score only as a tie-breaker."""
        query_tokens = cls._citation_tokens(question)
        grouped: dict[str, list[tuple[Any, float, int]]] = {}
        for position, (chunk, score) in enumerate(retrieved):
            identity = cls._source_identity(chunk)
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
            ranked_groups.append((rank, identity, members))

        ranked_groups.sort(key=lambda item: item[0], reverse=True)
        selected: list[_CitationSource] = []
        for _, identity, members in ranked_groups[:limit]:
            members.sort(
                key=lambda item: (
                    len(
                        query_tokens
                        & cls._citation_tokens(
                            "\n".join(
                                (
                                    str((getattr(item[0], "metadata", {}) or {}).get("title") or ""),
                                    str((getattr(item[0], "metadata", {}) or {}).get("section") or ""),
                                    str(getattr(item[0], "content", "")),
                                )
                            )
                        )
                    ),
                    item[1],
                    -item[2],
                ),
                reverse=True,
            )
            representative = members[0][0]
            unique_content: list[str] = []
            seen_content: set[str] = set()
            for chunk, _, _ in members:
                content = str(getattr(chunk, "content", "")).strip()
                normalized = " ".join(content.split()).casefold()
                if content and normalized not in seen_content:
                    seen_content.add(normalized)
                    unique_content.append(content)
            prompt_chunk = _PromptChunk(
                chunk_id=str(getattr(representative, "chunk_id", "")),
                content="\n\n".join(unique_content),
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
        return " ".join(matching) if matching else answer

    @classmethod
    def _best_excerpt(cls, content: str, target_text: str, max_chars: int = 360) -> str:
        """Return the source sentence, list item, or table row that best supports the claim."""
        target_tokens = cls._citation_tokens(target_text)
        candidates = [
            " ".join(part.split())
            for part in re.split(r"(?<=[.!?;])\s+|\n+", content)
            if part.strip() and not re.fullmatch(r"\s*\|?(?:\s*:?-{3,}:?\s*\|?)+\s*", part)
        ]
        if not candidates:
            return ""

        def candidate_rank(candidate: str) -> tuple[int, float, int]:
            tokens = cls._citation_tokens(candidate)
            overlap = len(target_tokens & tokens)
            density = overlap / max(1, len(tokens))
            return overlap, density, -abs(len(candidate) - 220)

        excerpt = max(candidates, key=candidate_rank)
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
        valid_markers = {
            int(value)
            for value in re.findall(r"\[(\d+)\]", answer)
            if 1 <= int(value) <= len(sources)
        }
        markers = sorted(valid_markers) if valid_markers else list(range(1, len(sources) + 1))
        citations: list[dict[str, Any]] = []
        for marker in markers[:2]:
            source = sources[marker - 1]
            claim = cls._claim_for_marker(answer, marker)
            target_text = f"{claim}\n{question}"
            target_tokens = cls._citation_tokens(target_text)
            chunk, score = max(
                source.members,
                key=lambda item: (
                    len(target_tokens & cls._citation_tokens(str(getattr(item[0], "content", "")))),
                    item[1],
                ),
            )
            snippet = cls._best_excerpt(str(getattr(chunk, "content", "")), target_text)
            citations.append(
                cls._citation(chunk, score, marker=marker, snippet=snippet)
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
        if not question or not question.strip():
            yield {"type": "metadata", "citations": [], "route_type": "out_of_scope"}
            yield {"type": "chunk", "text": FALLBACK_ANSWER}
            return

        retrieval_query = self._retrieval_query(question)
        try:
            retrieved = self.retriever.search(retrieval_query, top_k=self.top_k)
        except Exception as exc:
            raise RAGRetrievalError("Retrieval dependency failed") from exc
            
        if self.min_score is not None:
            retrieved = [(chunk, score) for chunk, score in retrieved if score >= self.min_score]
            
        from .assembler import ContextAssembler
        assembler = ContextAssembler(max_chars=12000)
        final_retrieved = assembler.assemble(retrieved)

        if not final_retrieved or not self._has_lexical_evidence(question, final_retrieved):
            yield {"type": "metadata", "citations": [], "route_type": "out_of_scope"}
            yield {"type": "chunk", "text": FALLBACK_ANSWER}
            return

        chunks = [chunk for chunk, _ in final_retrieved]
        sources = self._select_citation_sources(question, final_retrieved)
        chunks = [source.prompt_chunk for source in sources]
        scores = [source.retrieval_score for source in sources]
        context = format_citations(chunks, scores)
        system_prompt = SYSTEM_PROMPT.format(context=context)

        messages: list[dict[str, str]] = []
        if session_history:
            messages.extend(session_history)
        messages.extend(build_rag_prompt(chunks, question))
        
        generator = self.llm_client.generate_stream(system_prompt=system_prompt, messages=messages)
        
        accumulator = ""
        answer_parts: list[str] = []
        started_streaming = False
        close_pattern = re.compile(r"</(?:assistant_answer|answer|response|final)>.*", re.IGNORECASE | re.DOTALL)
        
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
                if len(accumulator) > 30:
                    emit = accumulator[:-30]
                    accumulator = accumulator[-30:]
                    if close_pattern.search(emit):
                        emit = close_pattern.sub("", emit)
                        if emit:
                            answer_parts.append(emit)
                            yield {"type": "chunk", "text": emit}
                        break
                    else:
                        answer_parts.append(emit)
                        yield {"type": "chunk", "text": emit}
                        
        if started_streaming and accumulator:
            accumulator = close_pattern.sub("", accumulator)
            if accumulator:
                answer_parts.append(accumulator)
                yield {"type": "chunk", "text": accumulator}

        citations = self._citations_for_answer(sources, "".join(answer_parts), question)
        yield {"type": "metadata", "citations": citations, "route_type": "general"}

    def answer(self, question: str, session_history: Optional[List[dict]] = None) -> RAGResponse:
        if not question or not question.strip():
            return RAGResponse(FALLBACK_ANSWER, [], "out_of_scope")

        retrieval_query = self._retrieval_query(question)
        try:
            retrieved = self.retriever.search(retrieval_query, top_k=self.top_k)
        except Exception as exc:
            raise RAGRetrievalError("Retrieval dependency failed") from exc
        if self.min_score is not None:
            retrieved = [(chunk, score) for chunk, score in retrieved if score >= self.min_score]
            
        from .assembler import ContextAssembler
        assembler = ContextAssembler(max_chars=12000)
        final_retrieved = assembler.assemble(retrieved)

        if not final_retrieved or not self._has_lexical_evidence(question, final_retrieved):
            return RAGResponse(FALLBACK_ANSWER, [], "out_of_scope")

        sources = self._select_citation_sources(question, final_retrieved)
        chunks = [source.prompt_chunk for source in sources]
        scores = [source.retrieval_score for source in sources]
        context = format_citations(chunks, scores)
        system_prompt = SYSTEM_PROMPT.format(context=context)

        messages: list[dict[str, str]] = []
        if session_history:
            messages.extend(session_history)
        messages.extend(build_rag_prompt(chunks, question))
        answer = self._clean_answer(
            self.llm_client.generate(system_prompt=system_prompt, messages=messages)
        )

        # Keep a one-to-one mapping with the numbered context documents so an
        # answer marker [n] always points to citations[n - 1] in the API payload.
        citations = self._citations_for_answer(sources, answer, question)
        return RAGResponse(answer, citations, "general")
