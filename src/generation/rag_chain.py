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
        scores = [float(score) for _, score in final_retrieved]
        context = format_citations(chunks, scores)
        system_prompt = SYSTEM_PROMPT.format(context=context)

        messages: list[dict[str, str]] = []
        if session_history:
            messages.extend(session_history)
        messages.extend(build_rag_prompt(chunks, question))
        
        citations = [self._citation(chunk, score) for chunk, score in final_retrieved]
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

        chunks = [chunk for chunk, _ in final_retrieved]
        scores = [float(score) for _, score in final_retrieved]
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
        citations = [self._citation(chunk, score) for chunk, score in final_retrieved]
        return RAGResponse(answer, citations, "general")
