import json
from unittest.mock import MagicMock

import pytest

from src.generation.prompts import (
    FAITHFULNESS_CHECK_PROMPT,
    RELEVANCE_CHECK_PROMPT,
)
from src.generation.self_rag import (
    RAGAction,
    RAGCheckStatus,
    SelfRAG,
)


def _checker(output: str) -> tuple[SelfRAG, MagicMock]:
    llm_client = MagicMock()
    llm_client.generate.return_value = output
    return SelfRAG(llm_client), llm_client


@pytest.mark.parametrize(
    ("verdict", "expected_status", "expected_action"),
    [
        ("pass", RAGCheckStatus.PASS, RAGAction.ACCEPT),
        ("fail", RAGCheckStatus.FAIL, RAGAction.RETRY),
    ],
)
def test_relevance_maps_verdict_to_action(
    verdict: str,
    expected_status: RAGCheckStatus,
    expected_action: RAGAction,
) -> None:
    checker, _ = _checker(json.dumps({"verdict": verdict, "reason": "evidence"}))

    result = checker.check_relevance("What is the tuition?", "Tuition is listed here.")

    assert result.status is expected_status
    assert result.action is expected_action
    assert result.reason == "evidence"
    assert result.error is None


@pytest.mark.parametrize(
    ("verdict", "expected_status", "expected_action"),
    [
        ("pass", RAGCheckStatus.PASS, RAGAction.ACCEPT),
        ("fail", RAGCheckStatus.FAIL, RAGAction.RETRY),
    ],
)
def test_faithfulness_maps_verdict_to_action(
    verdict: str,
    expected_status: RAGCheckStatus,
    expected_action: RAGAction,
) -> None:
    checker, _ = _checker(json.dumps({"verdict": verdict, "reason": "grounded"}))

    result = checker.check_faithfulness(
        "What is the tuition?",
        "The tuition is 10 million VND.",
        "Tuition is 10 million VND.",
    )

    assert result.status is expected_status
    assert result.action is expected_action


@pytest.mark.parametrize(
    "output",
    [
        "not JSON",
        json.dumps({"verdict": "maybe", "reason": "uncertain"}),
    ],
)
def test_malformed_or_invalid_output_falls_back(output: str) -> None:
    checker, _ = _checker(output)

    result = checker.check_relevance("Question", "Context")

    assert result.status is RAGCheckStatus.UNKNOWN
    assert result.action is RAGAction.FALLBACK
    assert result.error


def test_llm_exception_falls_back_without_propagating(caplog) -> None:
    llm_client = MagicMock()
    llm_client.generate.side_effect = RuntimeError("provider unavailable")

    result = SelfRAG(llm_client).check_relevance("Question", "Context")

    assert result.status is RAGCheckStatus.UNKNOWN
    assert result.action is RAGAction.FALLBACK
    assert "provider unavailable" in result.error
    assert "Self-RAG checker LLM call failed" in caplog.text


def test_fenced_json_is_parsed() -> None:
    checker, _ = _checker(
        '```json\n{"verdict": "pass", "reason": "supported"}\n```'
    )

    result = checker.check_relevance("Question", "Context")

    assert result.status is RAGCheckStatus.PASS
    assert result.action is RAGAction.ACCEPT


def test_empty_relevance_context_does_not_call_llm() -> None:
    checker, llm_client = _checker('{"verdict": "pass"}')

    result = checker.check_relevance("Question", "  \n")

    assert result.status is RAGCheckStatus.FAIL
    assert result.action is RAGAction.RETRY
    llm_client.generate.assert_not_called()


@pytest.mark.parametrize(
    ("answer", "context"),
    [
        ("  ", "Context"),
        ("Answer", "\n\t"),
    ],
)
def test_empty_faithfulness_input_does_not_call_llm(
    answer: str,
    context: str,
) -> None:
    checker, llm_client = _checker('{"verdict": "pass"}')

    result = checker.check_faithfulness("Question", answer, context)

    assert result.status is RAGCheckStatus.FAIL
    assert result.action is RAGAction.RETRY
    llm_client.generate.assert_not_called()


@pytest.mark.parametrize("method", ["relevance", "faithfulness"])
def test_blank_query_raises_without_calling_llm(method: str) -> None:
    checker, llm_client = _checker('{"verdict": "pass"}')

    with pytest.raises(ValueError, match="query must not be blank"):
        if method == "relevance":
            checker.check_relevance(" ", "Context")
        else:
            checker.check_faithfulness(" ", "Answer", "Context")

    llm_client.generate.assert_not_called()


def test_relevance_call_uses_contract_and_deterministic_temperature() -> None:
    checker, llm_client = _checker('{"verdict": "pass", "reason": 123}')

    result = checker.check_relevance("Tuition query", "Tuition context")

    llm_client.generate.assert_called_once_with(
        system_prompt=RELEVANCE_CHECK_PROMPT,
        messages=[
            {
                "role": "user",
                "content": (
                    "<query>\nTuition query\n</query>\n\n"
                    "<context>\nTuition context\n</context>"
                ),
            }
        ],
        temperature=0.0,
    )
    assert result.reason == "123"


def test_faithfulness_call_contains_query_context_and_answer() -> None:
    checker, llm_client = _checker('{"verdict": "pass", "reason": null}')

    result = checker.check_faithfulness(
        "Tuition query",
        "Tuition answer",
        "Tuition context",
    )

    llm_client.generate.assert_called_once_with(
        system_prompt=FAITHFULNESS_CHECK_PROMPT,
        messages=[
            {
                "role": "user",
                "content": (
                    "<query>\nTuition query\n</query>\n\n"
                    "<context>\nTuition context\n</context>\n\n"
                    "<answer>\nTuition answer\n</answer>"
                ),
            }
        ],
        temperature=0.0,
    )
    assert result.reason == ""
