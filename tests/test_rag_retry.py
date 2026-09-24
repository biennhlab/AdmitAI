from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.generation.rag_chain import (
    CORRECTIVE_GENERATION_INSTRUCTION,
    FALLBACK_ANSWER,
    RAGChain,
)
from src.generation.self_rag import RAGAction


class Chunk:
    def __init__(self, chunk_id: str, content: str):
        self.chunk_id = chunk_id
        self.content = content
        self.metadata = {"source": f"{chunk_id}.pdf"}


class Retriever:
    def __init__(self, results_by_query: dict[str, list[tuple[Chunk, float]]]):
        self.results_by_query = results_by_query
        self.calls: list[str] = []

    def search(self, query: str, top_k: int = 5):
        self.calls.append(query)
        return self.results_by_query.get(query, [])[:top_k]


class Checker:
    def __init__(
        self,
        relevance: list[RAGAction | Exception],
        faithfulness: list[RAGAction | Exception],
        events: list[str] | None = None,
    ):
        self.relevance = list(relevance)
        self.faithfulness = list(faithfulness)
        self.relevance_calls: list[tuple[str, str]] = []
        self.faithfulness_calls: list[tuple[str, str, str]] = []
        self.events = events

    @staticmethod
    def _result(value: RAGAction | Exception):
        if isinstance(value, Exception):
            raise value
        return SimpleNamespace(action=value)

    def check_relevance(self, query: str, context: str):
        self.relevance_calls.append((query, context))
        return self._result(self.relevance.pop(0))

    def check_faithfulness(self, query: str, answer: str, context: str):
        self.faithfulness_calls.append((query, answer, context))
        if self.events is not None:
            self.events.append("faithfulness")
        return self._result(self.faithfulness.pop(0))


class Rewriter:
    def __init__(self, rewritten: str):
        self.rewritten = rewritten
        self.calls: list[str] = []

    def rewrite(self, query: str) -> str:
        self.calls.append(query)
        return self.rewritten


def _chain(
    *,
    retriever: Retriever | None = None,
    checker: Checker | None = None,
    rewriter: Rewriter | None = None,
    llm: MagicMock | None = None,
) -> tuple[RAGChain, Retriever, Checker, Rewriter, MagicMock]:
    query = "tuition details"
    evidence = Chunk("tuition", "PTIT tuition details for 2024")
    retriever = retriever or Retriever({query: [(evidence, 0.9)]})
    checker = checker or Checker([RAGAction.ACCEPT], [RAGAction.ACCEPT])
    rewriter = rewriter or Rewriter("PTIT tuition details")
    llm = llm or MagicMock()
    llm.generate.return_value = "grounded answer [1]"
    return (
        RAGChain(
            retriever,
            llm,
            self_rag=checker,
            query_rewriter=rewriter,
            max_relevance_retries=1,
            max_faithfulness_retries=1,
        ),
        retriever,
        checker,
        rewriter,
        llm,
    )


def test_normal_query_has_no_retry() -> None:
    chain, retriever, checker, rewriter, llm = _chain()

    response = chain.answer("tuition details")

    assert response.answer == "grounded answer [1]"
    assert response.route_type == "general"
    assert retriever.calls == ["tuition details"]
    assert len(checker.relevance_calls) == 1
    assert len(checker.faithfulness_calls) == 1
    assert rewriter.calls == []
    assert llm.generate.call_count == 1


def test_irrelevant_first_retrieval_is_rewritten_once() -> None:
    rewritten = "PTIT tuition details"
    first = Chunk("first", "tuition details from an unrelated source")
    second = Chunk("second", "PTIT tuition details and fees")
    retriever = Retriever(
        {
            "tuition details": [(first, 0.8)],
            rewritten: [(second, 0.95)],
        }
    )
    checker = Checker(
        [RAGAction.RETRY, RAGAction.ACCEPT],
        [RAGAction.ACCEPT],
    )
    rewriter = Rewriter(rewritten)
    chain, _, _, _, llm = _chain(
        retriever=retriever,
        checker=checker,
        rewriter=rewriter,
    )

    response = chain.answer("tuition details")

    assert response.answer == "grounded answer [1]"
    assert retriever.calls == ["tuition details", rewritten]
    assert rewriter.calls == ["tuition details"]
    assert [call[0] for call in checker.relevance_calls] == [
        "tuition details",
        "tuition details",
    ]
    assert "second" in checker.relevance_calls[1][1]
    assert llm.generate.call_count == 1


def test_second_relevance_failure_falls_back_before_generation() -> None:
    rewritten = "PTIT tuition details"
    first = Chunk("first", "tuition details old")
    second = Chunk("second", "PTIT tuition details other")
    retriever = Retriever(
        {
            "tuition details": [(first, 0.8)],
            rewritten: [(second, 0.9)],
        }
    )
    checker = Checker([RAGAction.RETRY, RAGAction.RETRY], [])
    chain, _, _, rewriter, llm = _chain(
        retriever=retriever,
        checker=checker,
        rewriter=Rewriter(rewritten),
    )

    response = chain.answer("tuition details")

    assert response.answer == FALLBACK_ANSWER
    assert response.citations == []
    assert retriever.calls == ["tuition details", rewritten]
    assert rewriter.calls == ["tuition details"]
    llm.generate.assert_not_called()


@pytest.mark.parametrize("rewritten", ["", "  ", "tuition details", " TUITION   DETAILS "])
def test_empty_or_unchanged_rewrite_falls_back_without_loop(rewritten: str) -> None:
    checker = Checker([RAGAction.RETRY], [])
    chain, retriever, _, rewriter, llm = _chain(
        checker=checker,
        rewriter=Rewriter(rewritten),
    )

    response = chain.answer("tuition details")

    assert response.answer == FALLBACK_ANSWER
    assert retriever.calls == ["tuition details"]
    assert rewriter.calls == ["tuition details"]
    llm.generate.assert_not_called()


def test_faithful_answer_is_not_regenerated() -> None:
    chain, _, checker, _, llm = _chain()

    response = chain.answer("tuition details")

    assert response.answer == "grounded answer [1]"
    assert len(checker.faithfulness_calls) == 1
    assert llm.generate.call_count == 1


def test_unfaithful_answer_is_regenerated_once_with_corrective_instruction() -> None:
    checker = Checker(
        [RAGAction.ACCEPT],
        [RAGAction.RETRY, RAGAction.ACCEPT],
    )
    llm = MagicMock()
    llm.generate.side_effect = ["hallucinated answer", "corrected answer [1]"]
    chain, _, _, _, _ = _chain(checker=checker, llm=llm)

    response = chain.answer("tuition details")

    assert response.answer == "corrected answer [1]"
    assert llm.generate.call_count == 2
    first_prompt = llm.generate.call_args_list[0].kwargs["system_prompt"]
    corrective_prompt = llm.generate.call_args_list[1].kwargs["system_prompt"]
    assert CORRECTIVE_GENERATION_INSTRUCTION not in first_prompt
    assert CORRECTIVE_GENERATION_INSTRUCTION in corrective_prompt


def test_second_faithfulness_failure_returns_fallback_without_citations() -> None:
    checker = Checker(
        [RAGAction.ACCEPT],
        [RAGAction.RETRY, RAGAction.RETRY],
    )
    llm = MagicMock()
    llm.generate.side_effect = ["first hallucination", "second hallucination"]
    chain, _, _, _, _ = _chain(checker=checker, llm=llm)

    response = chain.answer("tuition details")

    assert response.answer == FALLBACK_ANSWER
    assert response.citations == []
    assert response.route_type == "out_of_scope"
    assert llm.generate.call_count == 2


@pytest.mark.parametrize(
    "relevance",
    [RAGAction.FALLBACK, RuntimeError("checker unavailable")],
)
def test_checker_fallback_or_exception_does_not_retry_blindly(
    relevance: RAGAction | Exception,
) -> None:
    checker = Checker([relevance], [])
    chain, retriever, _, rewriter, llm = _chain(checker=checker)

    response = chain.answer("tuition details")

    assert response.answer == FALLBACK_ANSWER
    assert retriever.calls == ["tuition details"]
    assert rewriter.calls == []
    llm.generate.assert_not_called()


def test_malformed_checker_result_falls_back_immediately() -> None:
    checker = MagicMock()
    checker.check_relevance.return_value = object()
    chain, retriever, _, rewriter, llm = _chain(checker=checker)

    response = chain.answer("tuition details")

    assert response.answer == FALLBACK_ANSWER
    assert retriever.calls == ["tuition details"]
    assert rewriter.calls == []
    llm.generate.assert_not_called()


def test_empty_retrieval_falls_back_before_self_rag_or_generation() -> None:
    retriever = Retriever({"tuition details": []})
    checker = MagicMock()
    chain, _, _, rewriter, llm = _chain(
        retriever=retriever,
        checker=checker,
    )

    response = chain.answer("tuition details")

    assert response.answer == FALLBACK_ANSWER
    checker.check_relevance.assert_not_called()
    checker.check_faithfulness.assert_not_called()
    assert rewriter.calls == []
    llm.generate.assert_not_called()


def test_streaming_buffers_until_faithfulness_accepts() -> None:
    events: list[str] = []
    checker = Checker(
        [RAGAction.ACCEPT],
        [RAGAction.ACCEPT],
        events=events,
    )
    llm = MagicMock()

    def stream(**_kwargs):
        events.append("provider_chunk_1")
        yield "accepted "
        events.append("provider_chunk_2")
        yield "answer [1]"

    llm.generate_stream.side_effect = stream
    chain, _, _, _, _ = _chain(checker=checker, llm=llm)
    output = chain.answer_stream("tuition details")

    metadata = next(output)

    assert events == ["provider_chunk_1", "provider_chunk_2", "faithfulness"]
    assert metadata["type"] == "metadata"
    assert metadata["route_type"] == "general"
    assert next(output) == {"type": "chunk", "text": "accepted answer [1]"}
    with pytest.raises(StopIteration):
        next(output)


def test_streaming_discards_unfaithful_buffer_and_emits_only_regeneration() -> None:
    checker = Checker(
        [RAGAction.ACCEPT],
        [RAGAction.RETRY, RAGAction.ACCEPT],
    )
    llm = MagicMock()
    llm.generate_stream.side_effect = [
        iter(["unverified ", "hallucination"]),
        iter(["verified answer [1]"]),
    ]
    chain, _, _, _, _ = _chain(checker=checker, llm=llm)

    output = list(chain.answer_stream("tuition details"))

    assert output[0]["type"] == "metadata"
    assert output[1] == {"type": "chunk", "text": "verified answer [1]"}
    assert "unverified" not in str(output)
    assert llm.generate_stream.call_count == 2
    assert CORRECTIVE_GENERATION_INSTRUCTION in (
        llm.generate_stream.call_args_list[1].kwargs["system_prompt"]
    )


def test_streaming_failure_emits_only_fallback_metadata_and_answer() -> None:
    checker = Checker(
        [RAGAction.ACCEPT],
        [RAGAction.RETRY, RAGAction.RETRY],
    )
    llm = MagicMock()
    llm.generate_stream.side_effect = [iter(["bad one"]), iter(["bad two"])]
    chain, _, _, _, _ = _chain(checker=checker, llm=llm)

    output = list(chain.answer_stream("tuition details"))

    assert output == [
        {"type": "metadata", "citations": [], "route_type": "out_of_scope"},
        {"type": "chunk", "text": FALLBACK_ANSWER},
    ]


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("max_relevance_retries", 2),
        ("max_faithfulness_retries", -1),
    ],
)
def test_retry_budget_cannot_exceed_phase_policy(keyword: str, value: int) -> None:
    kwargs = {keyword: value}

    with pytest.raises(ValueError, match=keyword):
        RAGChain(Retriever({}), MagicMock(), **kwargs)
