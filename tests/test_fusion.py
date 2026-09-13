import pytest

from src.ingestion.chunker import Chunk
from src.retrieval.fusion import HybridRetriever, reciprocal_rank_fusion


def chunk(chunk_id: str) -> Chunk:
    return Chunk(chunk_id=chunk_id, content=f"content-{chunk_id}")


def test_rrf_uses_one_based_rank_and_promotes_shared_document() -> None:
    fused = reciprocal_rank_fusion(
        [
            [("a", 0.9), ("b", 0.8), ("c", 0.7)],
            [("b", 0.95), ("c", 0.85), ("d", 0.75)],
        ],
        k=60,
    )
    scores = dict(fused)

    assert fused[0][0] == "b"
    assert scores["a"] == pytest.approx(1 / 61)
    assert scores["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert scores["c"] == pytest.approx(1 / 63 + 1 / 62)
    assert [score for _, score in fused] == sorted(scores.values(), reverse=True)


def test_rrf_deduplicates_within_list_but_accumulates_between_lists() -> None:
    shared = chunk("shared")
    fused = reciprocal_rank_fusion(
        [
            [(shared, 1.0), (shared, 0.5), (chunk("other"), 0.4)],
            [(chunk("shared"), 0.9)],
        ]
    )

    assert dict(fused)["shared"] == pytest.approx(1 / 61 + 1 / 61)
    assert [chunk_id for chunk_id, _ in fused].count("shared") == 1


def test_rrf_handles_empty_lists_and_one_retriever_without_results() -> None:
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []
    assert reciprocal_rank_fusion([[], [("a", 3.0), ("b", -9.0)]]) == [
        ("a", pytest.approx(1 / 61)),
        ("b", pytest.approx(1 / 62)),
    ]


def test_rrf_ties_are_deterministic_by_chunk_id() -> None:
    first = reciprocal_rank_fusion(
        [[("b", 1.0), ("a", 0.5)], [("a", 1.0), ("b", 0.5)]]
    )
    second = reciprocal_rank_fusion(
        [[("a", 1.0), ("b", 0.5)], [("b", 1.0), ("a", 0.5)]]
    )
    assert [chunk_id for chunk_id, _ in first] == ["a", "b"]
    assert second == first


class StubRetriever:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def search(self, query, top_k):
        self.calls.append((query, top_k))
        return self.results


def test_hybrid_calls_both_does_not_mutate_and_honors_top_k() -> None:
    shared = chunk("shared")
    dense_results = [(chunk("dense-only"), 0.99), (shared, 0.8), (chunk("d3"), 0.7)]
    sparse_results = [(chunk("sparse-only"), 12.0), (shared, 8.0), (chunk("s3"), 2.0)]
    dense_before = list(dense_results)
    sparse_before = list(sparse_results)
    dense = StubRetriever(dense_results)
    sparse = StubRetriever(sparse_results)

    results = HybridRetriever(dense, sparse).search("điểm chuẩn", top_k=2)

    assert dense.calls == [("điểm chuẩn", 2)]
    assert sparse.calls == [("điểm chuẩn", 2)]
    assert dense_results == dense_before
    assert sparse_results == sparse_before
    assert len(results) == 2
    # Rank 2 in both lists beats either document that is rank 1 in only one.
    assert results[0][0].chunk_id == "shared"


def test_hybrid_handles_one_empty_retriever_and_deduplicates_by_id() -> None:
    original = chunk("a")
    dense = StubRetriever([])
    sparse = StubRetriever([(original, 3.0), (chunk("a"), 2.0), (chunk("b"), 1.0)])

    results = HybridRetriever(dense, sparse).search("query", top_k=20)

    assert [item.chunk_id for item, _ in results] == ["a", "b"]
    assert results[0][0] is original
