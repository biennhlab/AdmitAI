from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict

import pytest

from src.ingestion.chunker import Chunk, combined_chunk
from src.ingestion.parser import Heading, ParsedPage, Table
from src.retrieval.fusion import HybridRetriever, reciprocal_rank_fusion


def chunk(chunk_id: str, *, content: str | None = None, metadata: dict[str, Any] | None = None) -> Chunk:
    return Chunk(chunk_id=chunk_id, content=content or f"content-{chunk_id}", metadata=metadata)


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
    dense_before = deepcopy(dense_results)
    sparse_before = deepcopy(sparse_results)
    dense = StubRetriever(dense_results)
    sparse = StubRetriever(sparse_results)

    results = HybridRetriever(dense, sparse).search("điểm chuẩn", top_k=2)

    from src.config import settings
    expected_k = 2 * settings.RETRIEVAL_CANDIDATE_MULTIPLIER
    assert dense.calls == [("điểm chuẩn", expected_k)]
    assert sparse.calls == [("điểm chuẩn", expected_k)]
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
    assert by_id["shared"][0].content == dense_shared.content
    assert by_id["shared"][0].metadata == {"expansion_type": "exact"}
    assert dense_shared.metadata is None
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
    assert results[0][0].content == original.content
    assert results[0][0].metadata == {"expansion_type": "exact"}
    assert original.metadata is None

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

def test_hybrid_expands_to_parent_and_deduplicates() -> None:
    parent = {"chunk_id": "p1", "content": "parent content", "metadata": {"chunk_type": "parent"}}
    c1 = chunk("c1", metadata={"parent_chunk_id": "p1"})
    c2 = chunk("c2", metadata={"parent_chunk_id": "p1"})
    c3 = chunk("c3") # No parent

    dense = StubRetriever([(c1, 0.99), (c3, 0.80)])
    sparse = StubRetriever([(c2, 10.0), (c3, 5.0)])
    
    retriever = HybridRetriever(dense, sparse, parent_chunks=[parent])
    results = retriever.search("query", top_k=5)
    
    # RRF ranks:
    # c1: dense rank 1 (1/61)
    # c2: sparse rank 1 (1/61)
    # c3: dense rank 2 (1/62), sparse rank 2 (1/62) -> 2/62
    # So both c1 and c2 have same score 1/61, c3 has 2/62 = 1/31 (~0.032). 
    # 1/61 is ~0.016. So c3 has higher RRF score!
    # Wait, c1 is rank 1 in dense. c2 is rank 1 in sparse.
    # Actually fused_scores: c1: 1/61. c2: 1/61. c3: 1/62 + 1/62 = 2/62 (~0.0322).
    # So c3 is first.
    # Then c1 and c2. Both c1 and c2 point to parent "p1".
    # When hydrating, c1 -> p1. seen_chunk_ids has p1.
    # c2 -> p1 (skipped).
    # Expected results: [c3, p1]
    
    assert len(results) == 2
    assert results[0][0].chunk_id == "c3"
    assert results[1][0].chunk_id == "p1"
    assert results[1][0].content == "parent content"

def test_hybrid_coverage_mode_increases_candidate_pool() -> None:
    dense = StubRetriever([])
    sparse = StubRetriever([])
    
    HybridRetriever(dense, sparse).search("danh sách các ngành", top_k=2)
    
    from src.config import settings
    expected_k = 2 * (settings.RETRIEVAL_CANDIDATE_MULTIPLIER * 2)
    assert dense.calls == [("danh sách các ngành", expected_k)]


def test_hybrid_expands_neighbors_for_coverage_mode() -> None:
    # 5 contiguous chunks to test budget=2
    metadata_1 = {"doc_id": "doc1", "chunk_index": 1, "heading_path": ["H1"]}
    metadata_2 = {"doc_id": "doc1", "chunk_index": 2, "heading_path": ["H1"]}
    metadata_3 = {"doc_id": "doc1", "chunk_index": 3, "heading_path": ["H1"]}
    metadata_4 = {"doc_id": "doc1", "chunk_index": 4, "heading_path": ["H1"]}
    metadata_5 = {"doc_id": "doc1", "chunk_index": 5, "heading_path": ["H2"]} # diff heading
    
    c1 = chunk("c1", metadata=metadata_1)
    c2 = chunk("c2", metadata=metadata_2)
    c3 = chunk("c3", metadata=metadata_3)
    c4 = chunk("c4", metadata=metadata_4)
    c5 = chunk("c5", metadata=metadata_5)
    
    dense = StubRetriever([(c3, 0.99)])
    sparse = StubRetriever([])
    
    sparse.chunks = [c1, c2, c3, c4, c5]
    
    retriever = HybridRetriever(dense, sparse)
    
    # Query without coverage intent -> only c3 (factual query)
    results = retriever.search("điểm chuẩn", top_k=5)
    assert len(results) == 1
    assert results[0][0].chunk_id == "c3"
    
    # Query with coverage intent -> expands to c2, c4 (budget=1) and c1 (budget=2). c5 has diff heading so excluded.
    results_enum = retriever.search("danh sách các ngành", top_k=10)
    assert len(results_enum) == 4
    result_ids = [r[0].chunk_id for r in results_enum]
    # c3 is main. Then neighbors offsets: offset=1 (-1=c2, +1=c4), offset=2 (-2=c1)
    assert result_ids == ["c3", "c2", "c4", "c1"]


def test_hybrid_diversity_aware_selection() -> None:
    # 5 chunks from doc1/H1
    d1_c1 = chunk("d1_c1", metadata={"doc_id": "d1", "heading_path": ["H1"]})
    d1_c2 = chunk("d1_c2", metadata={"doc_id": "d1", "heading_path": ["H1"]})
    d1_c3 = chunk("d1_c3", metadata={"doc_id": "d1", "heading_path": ["H1"]})
    d1_c4 = chunk("d1_c4", metadata={"doc_id": "d1", "heading_path": ["H1"]})
    d1_c5 = chunk("d1_c5", metadata={"doc_id": "d1", "heading_path": ["H1"]})
    
    # 2 chunks from doc2/H2 (different section)
    d2_c1 = chunk("d2_c1", metadata={"doc_id": "d2", "heading_path": ["H2"]})
    d2_c2 = chunk("d2_c2", metadata={"doc_id": "d2", "heading_path": ["H2"]})
    
    # Dense scores: d1 chunks are ranked very high, then d2 chunks
    dense = StubRetriever([
        (d1_c1, 0.99),
        (d1_c2, 0.98),
        (d1_c3, 0.97),
        (d1_c4, 0.96),
        (d1_c5, 0.95),
        (d2_c1, 0.50),
        (d2_c2, 0.40)
    ])
    sparse = StubRetriever([])
    
    retriever = HybridRetriever(dense, sparse)
    
    # Search top_k=4. Without diversity it would be [d1_c1, d1_c2, d1_c3, d1_c4]
    # With diversity: Round 1 -> [d1_c1, d2_c1]. Round 2 -> [d1_c2, d1_c3]
    results = retriever.search("query", top_k=4)
    result_ids = [r[0].chunk_id for r in results]
    assert result_ids == ["d1_c1", "d2_c1", "d1_c2", "d1_c3"]


def test_hybrid_table_part_expansion_in_coverage_mode() -> None:
    # 3 parts of the same table
    t_metadata_1 = {"doc_id": "doc1", "table_index": 0, "table_part": 1, "table_part_count": 3}
    t_metadata_2 = {"doc_id": "doc1", "table_index": 0, "table_part": 2, "table_part_count": 3}
    t_metadata_3 = {"doc_id": "doc1", "table_index": 0, "table_part": 3, "table_part_count": 3}
    
    t_part1 = chunk("t_part1", metadata=t_metadata_1)
    t_part2 = chunk("t_part2", metadata=t_metadata_2)
    t_part3 = chunk("t_part3", metadata=t_metadata_3)
    
    # Another normal chunk
    normal = chunk("normal", metadata={"doc_id": "doc1", "chunk_index": 10})
    
    dense = StubRetriever([(t_part2, 0.99), (normal, 0.90), (t_part3, 0.85)])
    sparse = StubRetriever([])
    
    sparse.chunks = [t_part1, t_part2, t_part3, normal]
    
    retriever = HybridRetriever(dense, sparse)
    
    # Query without coverage intent -> narrow mode
    results_narrow = retriever.search("điểm chuẩn", top_k=5)
    result_ids_narrow = [r[0].chunk_id for r in results_narrow]
    # Should only return what was retrieved
    assert result_ids_narrow == ["t_part2", "normal", "t_part3"]
    
    # Query with coverage intent -> coverage mode
    results_coverage = retriever.search("danh sách", top_k=5)
    result_ids_cov = [r[0].chunk_id for r in results_coverage]
    # table part 2 triggers expansion to all parts in order: 1, 2, 3
    # then normal chunk is added.
    # table part 3 is skipped later because it's already in seen_chunk_ids
    assert result_ids_cov == ["t_part1", "t_part2", "t_part3", "normal"]
    
    # Test budget limit (now relaxed in coverage mode for downstream ContextAssembler)
    results_budget = retriever.search("danh sách", top_k=2)
    # max_results = 2 * 3 = 6. All 4 chunks fit into the relaxed buffer.
    assert len(results_budget) == 4
    assert [r[0].chunk_id for r in results_budget] == ["t_part1", "t_part2", "t_part3", "normal"]


@pytest.mark.parametrize("with_prose", [False, True])
def test_combined_table_hits_keep_rows_and_expand_all_parts(with_prose: bool) -> None:
    heading = Heading("Admission scores", 1, 1)
    table = Table([["Major", "Score"], ["CNTT", "27.0"], ["ATTT", "26.5"], ["AI", "28.0"]], 1)
    text = "Admission scores\n" + ("General admission information." if with_prose else "")
    parents = []
    chunks = combined_chunk(
        [ParsedPage(1, text, tables=[table], headings=[heading])],
        max_size=90,
        metadata={"doc_id": "admission-doc"},
        parent_chunks=parents,
    )
    table_parts = [item for item in chunks if item.metadata["chunk_type"] == "table"]
    assert len(table_parts) > 1
    selected = table_parts[-1]
    dense = StubRetriever([(selected, 0.9)])
    sparse = StubRetriever([])
    sparse.chunks = chunks
    corpus_before = deepcopy(chunks)
    retriever = HybridRetriever(dense, sparse, parent_chunks=[asdict(parent) for parent in parents])

    narrow = retriever.search("AI score", top_k=5)
    assert ids(narrow) == [selected.chunk_id]
    assert narrow[0][0].content == selected.content
    assert "| AI | 28.0 |" in narrow[0][0].content
    assert narrow[0][0].metadata["chunk_type"] == "table"

    coverage = retriever.search("danh sách các ngành", top_k=5)
    assert ids(coverage) == [part.chunk_id for part in table_parts]
    assert [item.content for item, _score in coverage] == [part.content for part in table_parts]
    assert chunks == corpus_before
    assert narrow[0][0].metadata["expansion_type"] == "exact"


def test_search_expansion_labels_do_not_mutate_corpus_or_previous_results() -> None:
    first = chunk("first", metadata={"doc_id": "doc", "chunk_index": 0, "heading_path": ["heading"]})
    second = chunk("second", metadata={"doc_id": "doc", "chunk_index": 1, "heading_path": ["heading"]})
    dense = StubRetriever([(first, 1.0)])
    sparse = StubRetriever([])
    sparse.chunks = [first, second]
    corpus_before = deepcopy(sparse.chunks)
    retriever = HybridRetriever(dense, sparse)

    coverage = retriever.search("danh sách các ngành", top_k=2)
    coverage_before = deepcopy(coverage)
    assert [item.metadata["expansion_type"] for item, _score in coverage] == ["exact", "neighbor"]
    dense.results = [(second, 1.0)]
    narrow = retriever.search("score", top_k=2)

    assert narrow[0][0].metadata["expansion_type"] == "exact"
    assert coverage == coverage_before
    assert sparse.chunks == corpus_before
