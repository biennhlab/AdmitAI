from __future__ import annotations

import os
from unittest.mock import Mock, patch

import pytest

from src.ingestion.chunker import Chunk
from src.retrieval.reranker import Reranker


def _chunk(chunk_id: str, content: str | None = None) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        content=content or f"content-{chunk_id}",
        metadata={"source": "test.pdf", "nested": {"kept": True}},
    )


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
