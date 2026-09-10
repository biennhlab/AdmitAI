from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, List, Optional

from .prompts import SYSTEM_PROMPT, build_rag_prompt, format_citations


FALLBACK_ANSWER = (
    "Mình chưa tìm thấy thông tin đủ phù hợp trong dữ liệu tuyển sinh PTIT hiện có để trả lời "
    "câu hỏi này. Bạn vui lòng hỏi cụ thể hơn hoặc liên hệ Ban Tư vấn Tuyển sinh PTIT."
)


@dataclass
class RAGResponse:
    answer: str
    citations: List[dict[str, Any]]
    route_type: str


class RAGChain:
    """Retrieve, ground generation, and return citations from real metadata."""

    def __init__(
        self,
        retriever: Any,
        llm_client: Any,
        top_k: int = 5,
        min_score: float | None = None,
        min_lexical_coverage: float = 0.34,
    ):
        self.retriever = retriever
        self.llm_client = llm_client
        self.top_k = top_k
        self.min_score = min_score
        self.min_lexical_coverage = min_lexical_coverage

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
    def _citation(chunk: Any, score: float) -> dict[str, Any]:
        metadata = getattr(chunk, "metadata", {}) or {}
        content = getattr(chunk, "content", "")
        snippet = " ".join(content.split())[:280]
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
            "score": round(float(score), 6),
            "chunk_id": str(getattr(chunk, "chunk_id", "")),
        }

    @staticmethod
    def _clean_answer(answer: str) -> str:
        cleaned = answer.strip()
        cleaned = re.sub(r"^<assistant_answer>\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*</assistant_answer>$", "", cleaned, flags=re.IGNORECASE)
        return cleaned.strip()

    def answer(self, question: str, session_history: Optional[List[dict]] = None) -> RAGResponse:
        if not question or not question.strip():
            return RAGResponse(FALLBACK_ANSWER, [], "out_of_scope")

        retrieved = self.retriever.search(question, top_k=self.top_k)
        if self.min_score is not None:
            retrieved = [(chunk, score) for chunk, score in retrieved if score >= self.min_score]
        if not retrieved or not self._has_lexical_evidence(question, retrieved):
            return RAGResponse(FALLBACK_ANSWER, [], "out_of_scope")

        chunks = [chunk for chunk, _ in retrieved]
        scores = [float(score) for _, score in retrieved]
        context = format_citations(chunks, scores)
        system_prompt = SYSTEM_PROMPT.format(context=context)

        messages: list[dict[str, str]] = []
        if session_history:
            messages.extend(session_history)
        messages.extend(build_rag_prompt(chunks, question))
        answer = self._clean_answer(
            self.llm_client.generate(system_prompt=system_prompt, messages=messages)
        )

        citations: list[dict[str, Any]] = []
        seen: set[tuple[str, Any, str]] = set()
        for chunk, score in retrieved:
            citation = self._citation(chunk, score)
            identity = (citation["doc_id"], citation["page"], citation["source_url"])
            if identity not in seen:
                seen.add(identity)
                citations.append(citation)
        return RAGResponse(answer, citations, "general")
