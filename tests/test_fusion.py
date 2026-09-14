from __future__ import annotations

import pytest

from src.ingestion.chunker import Chunk
from src.retrieval.fusion import HybridRetriever, reciprocal_rank_fusion


def chunk(chunk_id: str, *, content: str | None = None) -> Chunk:
    return Chunk(chunk_id=chunk_id, content=content or f"content-{chunk_id}")


def ids(results: list[tuple[Chunk, float]]) -> list[str]:
    return [item.chunk_id for item, _score in results]


def test_rrf_uses_one_based_rank_and_promotes_shared_document() -> None:
    fused = reciprocal_rank_fusion(
        [
            [("a", 0.99), ("b", 0.50), ("c", 0.01)],
            [("b", 42.0), ("c", 12.0), ("d", 1.0)],
        ],
        k=60,
    )
    scores = dict(fused)

    assert [chunk_id for chunk_id, _score in fused] == ["b", "c", "a", "d"]
    assert scores == {
        "a": pytest.approx(1 / 61),
        "b": pytest.approx(1 / 62 + 1 / 61),
        "c": pytest.approx(1 / 63 + 1 / 62),
        "d": pytest.approx(1 / 63),
    }


def test_rrf_deduplicates_within_branch_without_consuming_a_rank() -> None:
    shared = chunk("shared")
    fused = reciprocal_rank_fusion(
        [
            [(shared, 1.0), (shared, 0.5), (chunk("other"), 0.4)],
            [(chunk("shared"), 0.9)],
        ]
    )
    scores = dict(fused)

    assert [chunk_id for chunk_id, _score in fused].count("shared") == 1
    assert scores["shared"] == pytest.approx(1 / 61 + 1 / 61)
    # A duplicate result is not a real rank and must not demote the next chunk.
    assert scores["other"] == pytest.approx(1 / 62)


def test_opposite_dense_and_sparse_orders_still_reward_consensus() -> None:
    fused = reciprocal_rank_fusion(
        [
            [("dense-only", 0.99), ("consensus", 0.70), ("tail-a", 0.10)],
            [("sparse-only", 25.0), ("consensus", 4.0), ("tail-b", 1.0)],
        ]
    )

    assert fused[0][0] == "consensus"
    assert dict(fused)["consensus"] > dict(fused)["dense-only"]
    assert dict(fused)["consensus"] > dict(fused)["sparse-only"]


def test_rrf_handles_missing_chunks_and_empty_result_lists() -> None:
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []

    fused = reciprocal_rank_fusion(
        [[("dense", 0.8), ("shared", 0.7)], [("shared", 9.0), ("sparse", 1.0)]]
    )
    assert [chunk_id for chunk_id, _score in fused] == ["shared", "dense", "sparse"]
    assert dict(fused)["dense"] == pytest.approx(1 / 61)
    assert dict(fused)["sparse"] == pytest.approx(1 / 62)


def test_rrf_ties_have_a_stable_chunk_id_tiebreak() -> None:
    first = reciprocal_rank_fusion(
        [[("b", 100.0), ("a", -5.0)], [("a", 0.0), ("b", 0.0)]]
    )
    second = reciprocal_rank_fusion(
        [[("a", 0.0), ("b", 0.0)], [("b", 100.0), ("a", -5.0)]]
    )

    assert first == second
    assert [chunk_id for chunk_id, _score in first] == ["a", "b"]
    assert first[0][1] == pytest.approx(first[1][1])


def test_rrf_rejects_invalid_rank_constant() -> None:
    with pytest.raises(ValueError, match="greater than or equal to 0"):
        reciprocal_rank_fusion([[("a", 1.0)]], k=-1)
    with pytest.raises(ValueError, match="finite"):
        reciprocal_rank_fusion([[("a", 1.0)]], k=float("inf"))


class StubRetriever:
    def __init__(self, results: list[tuple[Chunk, float]]):
        self.results = results
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, top_k: int) -> list[tuple[Chunk, float]]:
        self.calls.append((query, top_k))
        return self.results


def test_hybrid_honors_top_k_calls_both_and_does_not_mutate_inputs() -> None:
    dense_results = [(chunk("dense"), 0.99), (chunk("shared"), 0.80), (chunk("d3"), 0.70)]
    sparse_results = [(chunk("sparse"), 12.0), (chunk("shared"), 8.0), (chunk("s3"), 2.0)]
    dense_before = list(dense_results)
    sparse_before = list(sparse_results)
    dense = StubRetriever(dense_results)
    sparse = StubRetriever(sparse_results)

    results = HybridRetriever(dense, sparse).search("điểm chuẩn", top_k=2)

    assert dense.calls == [("điểm chuẩn", 2)]
    assert sparse.calls == [("điểm chuẩn", 2)]
    assert dense_results == dense_before
    assert sparse_results == sparse_before
    assert ids(results) == ["shared", "dense"]


def test_hybrid_keeps_chunk_and_fused_score_mapping_aligned() -> None:
    dense_shared = chunk("shared", content="dense payload wins deterministically")
    sparse_shared = chunk("shared", content="different sparse payload")
    dense = StubRetriever([(chunk("z"), 0.999), (dense_shared, 0.001)])
    sparse = StubRetriever([(sparse_shared, 999.0), (chunk("a"), 0.001)])

    results = HybridRetriever(dense, sparse).search("query", top_k=10)
    by_id = {item.chunk_id: (item, score) for item, score in results}

    assert ids(results) == ["shared", "z", "a"]
    assert by_id["shared"][0] is dense_shared
    assert by_id["shared"][1] == pytest.approx(1 / 62 + 1 / 61)
    assert by_id["z"][1] == pytest.approx(1 / 61)
    assert by_id["a"][1] == pytest.approx(1 / 62)
    assert all(score not in {0.999, 999.0, 0.001} for _item, score in results)


def test_hybrid_handles_one_or_both_empty_branches_and_deduplicates_by_id() -> None:
    original = chunk("a")
    dense = StubRetriever([])
    sparse = StubRetriever([(original, 3.0), (chunk("a"), 2.0), (chunk("b"), 1.0)])

    results = HybridRetriever(dense, sparse).search("query", top_k=20)
    assert ids(results) == ["a", "b"]
    assert results[0][0] is original

    assert HybridRetriever(StubRetriever([]), StubRetriever([])).search("query") == []


def test_hybrid_non_positive_top_k_short_circuits_both_retrievers() -> None:
    dense = StubRetriever([(chunk("a"), 1.0)])
    sparse = StubRetriever([(chunk("b"), 1.0)])
    retriever = HybridRetriever(dense, sparse)

    assert retriever.search("query", top_k=0) == []
    assert retriever.search("query", top_k=-1) == []
    assert dense.calls == []
    assert sparse.calls == []


def test_hybrid_is_deterministic_across_repeated_calls() -> None:
    dense = StubRetriever([(chunk("b"), 1.0), (chunk("a"), 0.5)])
    sparse = StubRetriever([(chunk("a"), 7.0), (chunk("b"), 1.0)])
    retriever = HybridRetriever(dense, sparse)

    snapshots = [
        [(item.chunk_id, score) for item, score in retriever.search("same query", top_k=2)]
        for _ in range(5)
    ]

    assert snapshots == [snapshots[0]] * 5
