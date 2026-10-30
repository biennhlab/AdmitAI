from copy import deepcopy

import pytest

from src.generation.assembler import ContextAssembler
from src.ingestion.chunker import Chunk

def test_assembler_prioritizes_and_deduplicates():
    assembler = ContextAssembler(max_chars=100)
    
    # Exact seeds
    c1 = Chunk(
        chunk_id="c1",
        content="hello world",
        metadata={"expansion_type": "exact", "doc_id": "doc", "page_number": 1},
    )
    
    # Siblings
    c2 = Chunk(chunk_id="c2", content="some table part", metadata={"expansion_type": "table_sibling"})
    
    # Neighbors
    c3 = Chunk(chunk_id="c3", content="neighbor context", metadata={"expansion_type": "neighbor"})
    
    # Duplicate of exact seed, but as neighbor
    c4 = Chunk(
        chunk_id="c4",
        content="Hello  world",
        metadata={"expansion_type": "neighbor", "doc_id": "doc", "page_number": 1},
    )
    
    retrieved = [(c3, 0.8), (c4, 0.7), (c1, 0.9), (c2, 0.85)]
    
    final = assembler.assemble(retrieved)
    
    # Priority: exact (c1) -> sibling (c2) -> neighbor (c3)
    # c4 should be dropped as duplicate of c1
    ids = [c.chunk_id for c, _ in final]
    assert ids == ["c1", "c2", "c3"]


@pytest.mark.parametrize(
    "left_metadata,right_metadata",
    [
        ({"doc_id": "a", "page_number": 1}, {"doc_id": "b", "page_number": 1}),
        (
            {"doc_id": "a", "page_number": 1, "admission_year": 2025},
            {"doc_id": "a", "page_number": 1, "admission_year": 2026},
        ),
        (
            {"doc_id": "a", "page_number": 1, "version": "v1"},
            {"doc_id": "a", "page_number": 1, "version": "v2"},
        ),
        ({"doc_id": "a", "page_number": 1}, {"doc_id": "a", "page_number": 2}),
        (
            {"doc_id": "a", "page_number": 1, "heading": "BVH"},
            {"doc_id": "a", "page_number": 1, "heading": "BVS"},
        ),
    ],
)
def test_assembler_preserves_same_text_with_different_provenance(
    left_metadata, right_metadata,
):
    chunks = [
        Chunk("left", "same evidence", left_metadata),
        Chunk("right", " SAME   EVIDENCE ", right_metadata),
    ]

    assembled = ContextAssembler().assemble([(chunks[0], 0.9), (chunks[1], 0.8)])

    assert [chunk.chunk_id for chunk, _ in assembled] == ["left", "right"]


def test_assembler_does_not_guess_identity_when_metadata_is_missing():
    chunks = [Chunk("left", "same evidence"), Chunk("right", "same evidence")]

    assembled = ContextAssembler().assemble([(chunks[0], 0.9), (chunks[1], 0.8)])

    assert [chunk.chunk_id for chunk, _ in assembled] == ["left", "right"]


def test_assembler_same_chunk_id_with_conflicting_evidence_is_not_merged():
    chunks = [
        Chunk("collision", "evidence A", {"source_url": "https://a", "page_number": 1}),
        Chunk("collision", "evidence B", {"source_url": "https://b", "page_number": 1}),
    ]

    assembled = ContextAssembler().assemble([(chunks[0], 0.9), (chunks[1], 0.8)])

    assert [chunk.content for chunk, _ in assembled] == ["evidence A", "evidence B"]


def test_assembler_preserves_identical_table_text_at_different_stable_locations():
    common = {
        "doc_id": "doc",
        "page_number": 4,
        "heading": "Quota",
        "table_index": 2,
        "chunk_type": "table",
    }
    chunks = [
        Chunk("part-1", "| BVS | 300 |", {**common, "table_part": 1, "row_start": 1, "row_end": 1}),
        Chunk("part-2", "| BVS | 300 |", {**common, "table_part": 2, "row_start": 5, "row_end": 5}),
    ]

    assembled = ContextAssembler().assemble([(chunks[0], 0.9), (chunks[1], 0.8)])

    assert [chunk.chunk_id for chunk, _ in assembled] == ["part-1", "part-2"]


def test_assembler_does_not_mutate_chunks_or_metadata():
    chunks = [
        Chunk("a", "alpha", {"doc_id": "doc", "page_number": 1, "heading_path": ["A"]}),
        Chunk("b", "beta", {"expansion_type": "neighbor"}),
    ]
    before = deepcopy(chunks)

    ContextAssembler().assemble([(chunks[0], 0.9), (chunks[1], 0.8)])

    assert chunks == before

def test_assembler_enforces_budget_and_avoids_splitting():
    assembler = ContextAssembler(max_chars=20)
    
    c1 = Chunk(chunk_id="c1", content="exact seed", metadata={"expansion_type": "exact"}) # 10 chars
    c2 = Chunk(chunk_id="c2", content="this sibling is too long", metadata={"expansion_type": "table_sibling"}) # 24 chars
    c3 = Chunk(chunk_id="c3", content="short", metadata={"expansion_type": "neighbor"}) # 5 chars
    
    retrieved = [(c1, 0.9), (c2, 0.8), (c3, 0.7)]
    final = assembler.assemble(retrieved)
    
    ids = [c.chunk_id for c, _ in final]
    # c1 (10) fits. c2 (24) is skipped because 10+24 = 34 > 20. c3 (5) fits because 10+5 = 15 <= 20.
    assert ids == ["c1", "c3"]
