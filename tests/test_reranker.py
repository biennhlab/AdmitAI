from __future__ import annotations

import os
import json
from unittest.mock import Mock, patch

import httpx
import pytest
from pydantic import ValidationError

from src.config import Settings
from src.generation.rag_chain import RAGChain
from src.ingestion.chunker import Chunk
from src.retrieval.reranker import (
    Reranker,
    RerankerHTTPError,
    RerankerInputError,
    RerankerResponseError,
    RerankerTimeoutError,
)


def _chunk(chunk_id: str, content: str | None = None) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        content=content or f"content-{chunk_id}",
        metadata={"source": "test.pdf", "nested": {"kept": True}},
    )


def _remote_reranker(
    handler,
    *,
    token: str = "remote-test-token",
    max_retries: int = 1,
    top_k_limit: int = 20,
) -> Reranker:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return Reranker(
        backend="remote",
        remote_url="https://reranker.example.test/inference",
        remote_api_token=token,
        remote_max_retries=max_retries,
        remote_top_k_limit=top_k_limit,
        http_client=client,
    )


def _response_for(
    request: httpx.Request,
    results: list[dict],
    *,
    usage: dict | None = None,
) -> httpx.Response:
    body = {
        "request_id": request.headers["X-Request-ID"],
        "model": "BAAI/bge-reranker-v2-m3",
        "results": results,
    }
    if usage is not None:
        body["usage"] = usage
    return httpx.Response(200, json=body)


def test_rerank_sorts_scores_and_preserves_original_data() -> None:
    chunks = [_chunk("a"), _chunk("b"), _chunk("c")]
    candidates = [(chunks[0], 0.8), (chunks[1], 0.3), (chunks[2], 0.6)]
    metadata_objects = [chunk.metadata for chunk in chunks]
    metadata_values = [dict(chunk.metadata) for chunk in chunks]
    model = Mock()
    model.predict.return_value = [0.2, 0.9, 0.5]

    results = Reranker(model=model, batch_size=8).rerank("admission query", candidates)

    assert [(chunk.chunk_id, retrieval, reranker) for chunk, retrieval, reranker in results] == [
        ("b", 0.3, 0.9),
        ("c", 0.6, 0.5),
        ("a", 0.8, 0.2),
    ]
    assert [chunk.metadata for chunk in chunks] == metadata_values
    assert all(chunk.metadata is original for chunk, original in zip(chunks, metadata_objects))
    model.predict.assert_called_once_with(
        [("admission query", "content-a"), ("admission query", "content-b"), ("admission query", "content-c")],
        batch_size=8,
    )


def test_rerank_top_k_limits_sorted_results() -> None:
    candidates = [(_chunk("a"), 0.1), (_chunk("b"), 0.2), (_chunk("c"), 0.3)]
    model = Mock()
    model.predict.return_value = [0.5, 0.7, 0.6]

    results = Reranker(model=model).rerank("query", candidates, top_k=2)

    assert [chunk.chunk_id for chunk, _, _ in results] == ["b", "c"]


@pytest.mark.parametrize("top_k", [0, -1])
def test_non_positive_top_k_returns_without_model_call(top_k: int) -> None:
    model = Mock()
    reranker = Reranker(model=model)

    assert reranker.rerank("query", [(_chunk("a"), 0.5)], top_k=top_k) == []
    model.predict.assert_not_called()


def test_empty_candidates_returns_without_model_call() -> None:
    model = Mock()
    reranker = Reranker(model=model)

    assert reranker.rerank("query", []) == []
    model.predict.assert_not_called()


@pytest.mark.parametrize("query", ["", "   ", "\n\t"])
def test_blank_query_is_rejected(query: str) -> None:
    model = Mock()
    reranker = Reranker(model=model)

    with pytest.raises(ValueError, match="Query cannot be empty"):
        reranker.rerank(query, [(_chunk("a"), 0.5)])
    model.predict.assert_not_called()


def test_ties_use_retrieval_score_then_chunk_id() -> None:
    candidates = [
        (_chunk("z"), 0.8),
        (_chunk("b"), 0.9),
        (_chunk("a"), 0.9),
    ]
    model = Mock()
    model.predict.return_value = [0.5, 0.5, 0.5]

    results = Reranker(model=model).rerank("query", candidates)

    assert [chunk.chunk_id for chunk, _, _ in results] == ["a", "b", "z"]


def test_invalid_score_count_fails_clearly() -> None:
    model = Mock()
    model.predict.return_value = [0.5]
    candidates = [(_chunk("a"), 0.2), (_chunk("b"), 0.1)]

    with pytest.raises(ValueError, match=r"score count mismatch: expected 2, got 1"):
        Reranker(model=model).rerank("query", candidates)


def test_model_scores_are_converted_to_finite_floats() -> None:
    model = Mock()
    model.predict.return_value = ["0.75"]

    result = Reranker(model=model).rerank("query", [(_chunk("a"), 0.2)])

    assert result[0][2] == 0.75
    assert isinstance(result[0][2], float)

    model.predict.return_value = ["not-a-score"]
    with pytest.raises(ValueError, match="cannot be converted to float"):
        Reranker(model=model).rerank("query", [(_chunk("a"), 0.2)])


def test_model_is_loaded_once_in_constructor_not_during_rerank() -> None:
    model = Mock()
    model.predict.side_effect = [[0.2], [0.3]]
    with patch("src.retrieval.reranker.CrossEncoder", return_value=model) as factory:
        reranker = Reranker(model_name="test-reranker")
        reranker.rerank("first", [(_chunk("a"), 0.1)])
        reranker.rerank("second", [(_chunk("b"), 0.2)])

    factory.assert_called_once_with("test-reranker")
    assert model.predict.call_count == 2


def test_remote_request_contract_mapping_sorting_and_metadata() -> None:
    captured: dict = {}
    chunks = [_chunk("a"), _chunk("b"), _chunk("c")]
    metadata_objects = [chunk.metadata for chunk in chunks]

    def handler(request: httpx.Request) -> httpx.Response:
        captured["headers"] = request.headers
        captured["payload"] = json.loads(request.content)
        return _response_for(
            request,
            [
                {"index": 2, "document_id": "c", "relevance_score": 0.9},
                {"index": 0, "document_id": "a", "relevance_score": 1.2},
            ],
            usage={"latency_ms": 12.5},
        )

    reranker = _remote_reranker(handler)
    results = reranker.rerank(
        "admission query",
        [(chunks[0], 0.8), (chunks[1], 0.7), (chunks[2], 0.6)],
        top_k=2,
    )

    assert captured["payload"] == {
        "query": "admission query",
        "documents": [
            {"id": "a", "text": "content-a"},
            {"id": "b", "text": "content-b"},
            {"id": "c", "text": "content-c"},
        ],
        "top_n": 2,
    }
    assert captured["headers"]["Authorization"] == "Bearer remote-test-token"
    assert captured["headers"]["Content-Type"] == "application/json"
    assert captured["headers"]["Accept"] == "application/json"
    assert captured["headers"]["X-Request-ID"]
    assert [(chunk.chunk_id, retrieval, score) for chunk, retrieval, score in results] == [
        ("a", 0.8, 1.2),
        ("c", 0.6, 0.9),
    ]
    assert all(chunk.metadata is original for chunk, original in zip(chunks, metadata_objects))
    assert reranker.last_usage_latency_ms == 12.5


def test_remote_backend_never_constructs_cross_encoder() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _response_for(
            request,
            [{"index": 0, "document_id": "a", "relevance_score": 0.5}],
        )

    with patch("src.retrieval.reranker.CrossEncoder") as factory:
        reranker = _remote_reranker(handler)
        reranker.rerank("query", [(_chunk("a"), 0.2)])

    factory.assert_not_called()


@pytest.mark.parametrize("status_code", [429, 503, 504])
def test_remote_retries_only_retryable_status_then_succeeds(status_code: int) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(status_code, json={"error": "temporary"})
        return _response_for(
            request,
            [{"index": 0, "document_id": "a", "relevance_score": 0.7}],
        )

    with patch("src.retrieval.reranker.time.sleep") as sleep:
        results = _remote_reranker(handler).rerank("query", [(_chunk("a"), 0.2)])

    assert results[0][2] == 0.7
    assert len(requests) == 2
    sleep.assert_called_once_with(0.5)


def test_remote_retry_stops_after_configured_maximum() -> None:
    request_count = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        return httpx.Response(503, json={"error": "still unavailable"})

    with patch("src.retrieval.reranker.time.sleep") as sleep:
        with pytest.raises(RerankerHTTPError) as raised:
            _remote_reranker(handler, max_retries=1).rerank(
                "query", [(_chunk("a"), 0.2)]
            )

    assert raised.value.status_code == 503
    assert request_count == 2
    sleep.assert_called_once_with(0.5)


@pytest.mark.parametrize("status_code", [400, 401, 403, 413])
def test_remote_does_not_retry_permanent_http_errors(status_code: int) -> None:
    request_count = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        return httpx.Response(status_code, json={"error": "permanent"})

    with patch("src.retrieval.reranker.time.sleep") as sleep:
        with pytest.raises(RerankerHTTPError) as raised:
            _remote_reranker(handler).rerank("query", [(_chunk("a"), 0.2)])

    assert raised.value.status_code == status_code
    assert request_count == 1
    sleep.assert_not_called()


def test_remote_timeout_retries_once_then_raises_typed_error() -> None:
    request_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        raise httpx.ReadTimeout("provider timeout details", request=request)

    with patch("src.retrieval.reranker.time.sleep") as sleep:
        with pytest.raises(RerankerTimeoutError):
            _remote_reranker(handler).rerank("query", [(_chunk("a"), 0.2)])

    assert request_count == 2
    sleep.assert_called_once_with(0.5)


def _valid_remote_body(request_id: str) -> dict:
    return {
        "request_id": request_id,
        "model": "BAAI/bge-reranker-v2-m3",
        "results": [
            {"index": 0, "document_id": "a", "relevance_score": 0.8},
        ],
    }


@pytest.mark.parametrize(
    "mutate",
    [
        lambda body: body.pop("results"),
        lambda body: body.update(
            results=[{"index": 2, "document_id": "a", "relevance_score": 0.8}]
        ),
        lambda body: body.update(
            results=[
                {"index": 0, "document_id": "a", "relevance_score": 0.8},
                {"index": 0, "document_id": "a", "relevance_score": 0.7},
            ]
        ),
        lambda body: body.update(
            results=[{"index": 0, "document_id": "wrong", "relevance_score": 0.8}]
        ),
        lambda body: body.update(
            results=[{"index": 0, "document_id": "a", "relevance_score": "bad"}]
        ),
        lambda body: body.update(
            results=[{"index": 0, "document_id": "a", "relevance_score": float("nan")}]
        ),
        lambda body: body.update(
            results=[{"index": 0, "document_id": "a", "relevance_score": float("inf")}]
        ),
        lambda body: body.update(request_id="wrong-request"),
        lambda body: body.update(model="wrong-model"),
    ],
    ids=[
        "missing-results",
        "index-out-of-range",
        "duplicate-index",
        "document-id-mismatch",
        "non-numeric-score",
        "nan-score",
        "infinite-score",
        "request-id-mismatch",
        "model-mismatch",
    ],
)
def test_remote_rejects_malformed_success_without_retry(mutate) -> None:
    request_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        body = _valid_remote_body(request.headers["X-Request-ID"])
        mutate(body)
        return httpx.Response(
            200,
            content=json.dumps(body, allow_nan=True),
            headers={"Content-Type": "application/json"},
        )

    with pytest.raises(RerankerResponseError):
        _remote_reranker(handler).rerank(
            "query",
            [(_chunk("a"), 0.2), (_chunk("b"), 0.1)],
            top_k=2,
        )

    assert request_count == 1


def test_remote_token_never_appears_in_exception_or_logs(caplog) -> None:
    token = "super-secret-reranker-token"

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text=f"rejected {token}")

    with pytest.raises(RerankerHTTPError) as raised:
        _remote_reranker(handler, token=token).rerank(
            "query", [(_chunk("a"), 0.2)]
        )

    assert token not in str(raised.value)
    assert token not in caplog.text
    assert "request_id=" in caplog.text
    assert "status=401" in caplog.text
    assert "attempt=1" in caplog.text
    assert "retryable=no" in caplog.text
    assert "candidate_count=1" in caplog.text
    assert "endpoint=reranker.example.test" in caplog.text


@pytest.mark.parametrize(
    ("query", "candidates", "message"),
    [
        ("q" * 4_001, [(_chunk("a"), 0.2)], "Query exceeds"),
        (
            "query",
            [(_chunk(str(index)), float(index)) for index in range(21)],
            "Candidate count exceeds",
        ),
        (
            "query",
            [(_chunk("a", "x" * 16_001), 0.2)],
            "Document text",
        ),
    ],
)
def test_remote_contract_limits_fail_before_http(query, candidates, message) -> None:
    handler = Mock()
    reranker = _remote_reranker(handler)

    with pytest.raises(RerankerInputError, match=message):
        reranker.rerank(query, candidates)

    handler.assert_not_called()


def test_remote_settings_require_credentials_and_validate_bounds() -> None:
    local = Settings(_env_file=None)
    assert local.RERANKER_BACKEND == "local"
    assert str(local.RERANKER_REMOTE_API_TOKEN) == ""

    with pytest.raises(ValidationError, match="RERANKER_REMOTE_URL"):
        Settings(_env_file=None, RERANKER_BACKEND="remote")
    with pytest.raises(ValidationError, match="RERANKER_REMOTE_TOP_K_LIMIT"):
        Settings(_env_file=None, RERANKER_REMOTE_TOP_K_LIMIT=21)


def test_remote_failure_integrates_with_rag_retrieval_order_fallback(caplog) -> None:
    chunks = [
        _chunk("a", "tuition details first"),
        _chunk("b", "tuition details second"),
    ]

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "unavailable"})

    class Retriever:
        def search(self, _query, top_k):
            return [(chunks[0], 0.8), (chunks[1], 0.6)][:top_k]

    llm = Mock()
    llm.generate.return_value = "fallback answer"
    with patch("src.retrieval.reranker.time.sleep"):
        response = RAGChain(
            Retriever(), llm, reranker=_remote_reranker(handler), rerank_top_k=2
        ).answer("tuition details")

    assert [citation["chunk_id"] for citation in response.citations] == ["a", "b"]
    prompt = llm.generate.call_args.kwargs["system_prompt"]
    assert prompt.index("tuition details first") < prompt.index("tuition details second")
    assert "continuing with retrieval-ranked evidence" in caplog.text


def test_remote_success_integrates_with_rag_generation_order() -> None:
    chunks = [
        _chunk("a", "tuition details first"),
        _chunk("b", "tuition details second"),
        _chunk("c", "tuition details best"),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return _response_for(
            request,
            [
                {"index": 2, "document_id": "c", "relevance_score": 1.3},
                {"index": 0, "document_id": "a", "relevance_score": 0.7},
            ],
        )

    class Retriever:
        def search(self, _query, top_k):
            return [(chunks[0], 0.8), (chunks[1], 0.7), (chunks[2], 0.6)][:top_k]

    llm = Mock()
    llm.generate.return_value = "reranked answer [1]"
    response = RAGChain(
        Retriever(), llm, reranker=_remote_reranker(handler), rerank_top_k=2
    ).answer("tuition details")

    assert [citation["chunk_id"] for citation in response.citations] == ["c", "a"]
    prompt = llm.generate.call_args.kwargs["system_prompt"]
    assert prompt.index("tuition details best") < prompt.index("tuition details first")


@pytest.mark.skipif(
    os.getenv("RUN_RERANKER_SMOKE") != "1",
    reason="set RUN_RERANKER_SMOKE=1 when the reranker model is locally cached",
)
def test_real_bge_reranker_smoke() -> None:
    reranker = Reranker(model_name="BAAI/bge-reranker-v2-m3", batch_size=2)
    candidates = [
        (_chunk("relevant", "Học phí ngành Công nghệ thông tin là 30 triệu đồng."), 0.4),
        (_chunk("irrelevant", "Khuôn viên trường có nhiều cây xanh."), 0.8),
        (_chunk("other", "Thông tin tuyển sinh được công bố vào tháng sáu."), 0.6),
    ]

    results = reranker.rerank("Học phí ngành Công nghệ thông tin", candidates)

    assert len(results) == len(candidates)
    assert [score for _, _, score in results] == sorted(
        (score for _, _, score in results), reverse=True
    )


@pytest.mark.skipif(
    os.getenv("RUN_REMOTE_RERANKER_SMOKE") != "1",
    reason="set RUN_REMOTE_RERANKER_SMOKE=1 with endpoint credentials",
)
def test_remote_smoke_through_rag_chain() -> None:
    remote_url = os.getenv("RERANKER_REMOTE_URL", "")
    remote_token = os.getenv("RERANKER_REMOTE_API_TOKEN", "")
    assert remote_url and remote_token, (
        "RERANKER_REMOTE_URL and RERANKER_REMOTE_API_TOKEN are required for smoke"
    )

    candidates = [
        (_chunk("tuition", "tuition details for the information technology program"), 0.6),
        (_chunk("campus", "campus trees and sports facilities"), 0.8),
        (_chunk("admission", "admission schedule and tuition details"), 0.7),
    ]

    class Retriever:
        def search(self, _query, top_k):
            return candidates[:top_k]

    reranker = Reranker(
        backend="remote",
        remote_url=remote_url,
        remote_api_token=remote_token,
        remote_timeout_seconds=float(os.getenv("RERANKER_REMOTE_TIMEOUT_SECONDS", "30")),
        remote_max_retries=int(os.getenv("RERANKER_REMOTE_MAX_RETRIES", "1")),
        remote_top_k_limit=int(os.getenv("RERANKER_REMOTE_TOP_K_LIMIT", "20")),
    )
    llm = Mock()
    llm.generate.return_value = "remote smoke answer [1]"
    try:
        response = RAGChain(
            Retriever(), llm, reranker=reranker, rerank_top_k=3
        ).answer("tuition details")
    finally:
        reranker.close()

    assert response.citations
    assert llm.generate.called
