from __future__ import annotations

from dataclasses import dataclass
import logging
import math
import re
from typing import Any, List, Optional

from src.config import settings

from .assembler import ContextAssembler
from .prompts import SYSTEM_PROMPT, build_rag_prompt, format_citations


FALLBACK_ANSWER = "Mình chưa tìm thấy thông tin này trong dữ liệu tuyển sinh hiện có."


logger = logging.getLogger(__name__)


class RAGRetrievalError(RuntimeError):
    """The retrieval dependency failed while answering a request."""


@dataclass
class RAGResponse:
    answer: str
    citations: List[dict[str, Any]]
    route_type: str


@dataclass(frozen=True)
class RankedEvidence:
    chunk: Any
    retrieval_score: float
    reranker_score: float | None = None

    @property
    def context_score(self) -> float:
        return self.reranker_score if self.reranker_score is not None else self.retrieval_score


class RAGChain:
    """Retrieve, ground generation, and return citations from real metadata."""

    def __init__(
        self,
        retriever: Any,
        llm_client: Any,
        top_k: int = 5,
        min_score: float | None = None,
        min_lexical_coverage: float = 0.34,
        reranker: Any | None = None,
        rerank_top_k: int | None = None,
    ):
        self.retriever = retriever
        self.llm_client = llm_client
        self.top_k = top_k
        self.min_score = min_score
        self.min_lexical_coverage = min_lexical_coverage
        self.reranker = reranker
        self.rerank_top_k = (
            settings.RERANK_TOP_K
            if reranker is not None and rerank_top_k is None
            else rerank_top_k
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

    def _has_lexical_evidence(self, question: str, evidence: list[RankedEvidence]) -> bool:
        query_tokens = self._evidence_tokens(question)
        if not query_tokens:
            return False
        searchable = []
        for item in evidence:
            chunk = item.chunk
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
    def _citation(
        chunk: Any,
        score: float | None,
        *,
        retrieval_score: float | None = None,
        reranker_score: float | None = None,
    ) -> dict[str, Any]:
        metadata = getattr(chunk, "metadata", {}) or {}
        content = getattr(chunk, "content", "")
        snippet = " ".join(content.split())[:280]
        effective_score = (
            reranker_score
            if reranker_score is not None
            else retrieval_score if retrieval_score is not None else score
        )
        return {
            "doc_id": str(metadata.get("doc_id") or ""),
            "title": str(metadata.get("title") or metadata.get("source") or f"Chunk {chunk.chunk_id}"),
            "source": str(metadata.get("title") or metadata.get("source") or f"Chunk {chunk.chunk_id}"),
            "source_url": str(metadata.get("source_url") or ""),
            "source_type": str(metadata.get("source_type") or ""),
            "published_at": str(metadata.get("published_at") or ""),
            "retrieved_at": str(metadata.get("retrieved_at") or ""),
            "page": metadata.get("page_number"),
            "section": metadata.get("section"),
            "snippet": snippet,
            "score": round(float(effective_score), 6) if effective_score is not None else None,
            "chunk_id": str(getattr(chunk, "chunk_id", "")),
        }

    @staticmethod
    def _evidence_identity(item: RankedEvidence) -> tuple[str, str, str, str]:
        metadata = getattr(item.chunk, "metadata", {}) or {}
        doc_id = str(metadata.get("doc_id") or "")
        page_number = metadata.get("page_number")
        page = "" if page_number is None else str(page_number)
        chunk_id = str(getattr(item.chunk, "chunk_id", "") or "")
        if chunk_id:
            return ("chunk", doc_id, page, chunk_id)
        content = str(getattr(item.chunk, "content", ""))
        normalized_content = " ".join(content.split()).casefold()
        return ("content", doc_id, page, normalized_content)

    @classmethod
    def _deduplicate_evidence(cls, evidence: list[RankedEvidence]) -> list[RankedEvidence]:
        deduplicated: list[RankedEvidence] = []
        seen: set[tuple[str, str, str, str]] = set()
        for item in evidence:
            identity = cls._evidence_identity(item)
            if identity in seen:
                continue
            seen.add(identity)
            deduplicated.append(item)
        return deduplicated

    def _retrieve_and_prepare(self, question: str) -> list[RankedEvidence]:
        try:
            retrieved = list(self.retriever.search(question, top_k=self.top_k))
        except Exception as exc:
            raise RAGRetrievalError("Retrieval dependency failed") from exc

        if self.min_score is not None:
            retrieved = [
                (chunk, score)
                for chunk, score in retrieved
                if score >= self.min_score
            ]
        if not retrieved:
            return []

        evidence = [RankedEvidence(chunk, float(score)) for chunk, score in retrieved]
        if self.reranker is not None:
            try:
                reranked = self.reranker.rerank(
                    question,
                    [(item.chunk, item.retrieval_score) for item in evidence],
                    top_k=self.rerank_top_k,
                )
                evidence = [
                    RankedEvidence(chunk, float(retrieval_score), float(reranker_score))
                    for chunk, retrieval_score, reranker_score in reranked
                ]
            except Exception:
                logger.warning(
                    "Reranker failed; continuing with retrieval-ranked evidence",
                    exc_info=True,
                )

        evidence_by_object: dict[int, RankedEvidence] = {}
        assembler_input: list[tuple[Any, float]] = []
        for item in evidence:
            evidence_by_object.setdefault(id(item.chunk), item)
            assembler_input.append((item.chunk, item.context_score))

        assembled = ContextAssembler(max_chars=12000).assemble(assembler_input)
        final_evidence = self._deduplicate_evidence(
            [evidence_by_object[id(chunk)] for chunk, _score in assembled]
        )
        if not final_evidence or not self._has_lexical_evidence(question, final_evidence):
            return []
        return final_evidence

    def _generation_inputs(
        self,
        question: str,
        session_history: Optional[List[dict]],
        evidence: list[RankedEvidence],
    ) -> tuple[str, list[dict[str, str]], list[dict[str, Any]]]:
        chunks = [item.chunk for item in evidence]
        # Ranking scores are internal signals and do not belong in the prompt.
        context = format_citations(chunks)
        system_prompt = SYSTEM_PROMPT.format(context=context)

        messages: list[dict[str, str]] = []
        if session_history:
            messages.extend(session_history)
        messages.extend(build_rag_prompt(chunks, question))

        citations = [
            self._citation(
                item.chunk,
                item.retrieval_score,
                retrieval_score=item.retrieval_score,
                reranker_score=item.reranker_score,
            )
            for item in evidence
        ]
        return system_prompt, messages, citations

    @staticmethod
    def _clean_answer(answer: str) -> str:
        cleaned = answer.strip()
        wrapper_names = "assistant_answer|answer|response|final"
        cleaned = re.sub(
            rf"\A<(?:{wrapper_names})>\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
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

        evidence = self._retrieve_and_prepare(question)
        if not evidence:
            yield {"type": "metadata", "citations": [], "route_type": "out_of_scope"}
            yield {"type": "chunk", "text": FALLBACK_ANSWER}
            return

        system_prompt, messages, citations = self._generation_inputs(
            question, session_history, evidence
        )
        yield {"type": "metadata", "citations": citations, "route_type": "general"}
        
        generator = self.llm_client.generate_stream(system_prompt=system_prompt, messages=messages)
        
        accumulator = ""
        started_streaming = False
        wrapper_pattern = re.compile(r"^(?:<assistant_answer>|<answer>|<response>|<final>)\s*", re.IGNORECASE)
        close_pattern = re.compile(r"</(?:assistant_answer|answer|response|final)>.*", re.IGNORECASE | re.DOTALL)
        
        for chunk_text in generator:
            accumulator += chunk_text
            if not started_streaming:
                if accumulator.startswith("<"):
                    if len(accumulator) > 20 and not wrapper_pattern.match(accumulator):
                        started_streaming = True
                    elif wrapper_pattern.match(accumulator):
                        accumulator = wrapper_pattern.sub("", accumulator)
                        started_streaming = True
                else:
                    started_streaming = True
            
            if started_streaming:
                if len(accumulator) > 30:
                    emit = accumulator[:-30]
                    accumulator = accumulator[-30:]
                    if close_pattern.search(emit):
                        emit = close_pattern.sub("", emit)
                        if emit:
                            yield {"type": "chunk", "text": emit}
                        break
                    else:
                        yield {"type": "chunk", "text": emit}
                        
        if started_streaming and accumulator:
            accumulator = close_pattern.sub("", accumulator)
            if accumulator:
                yield {"type": "chunk", "text": accumulator}

    def answer(self, question: str, session_history: Optional[List[dict]] = None) -> RAGResponse:
        if not question or not question.strip():
            return RAGResponse(FALLBACK_ANSWER, [], "out_of_scope")

        evidence = self._retrieve_and_prepare(question)
        if not evidence:
            return RAGResponse(FALLBACK_ANSWER, [], "out_of_scope")

        system_prompt, messages, citations = self._generation_inputs(
            question, session_history, evidence
        )
        answer = self._clean_answer(
            self.llm_client.generate(system_prompt=system_prompt, messages=messages)
        )

        # Keep a one-to-one mapping with the numbered context documents so an
        # answer marker [n] always points to citations[n - 1] in the API payload.
        return RAGResponse(answer, citations, "general")
