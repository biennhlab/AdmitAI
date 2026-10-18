from copy import deepcopy

import pytest

from src.generation.assembler import ContextAssembler
from src.ingestion.chunker import Chunk

def test_assembler_prioritizes_and_deduplicates():
    assembler = ContextAssembler(max_chars=100)
    
    # Exact seeds
    provenance = {"doc_id": "doc-1", "page_number": 3, "heading": "Admissions"}
    c1 = Chunk(
        chunk_id="c1",
        content="hello world",
        metadata={**provenance, "expansion_type": "exact"},
    )
    
    # Siblings
    c2 = Chunk(chunk_id="c2", content="some table part", metadata={"expansion_type": "table_sibling"})
    
    # Neighbors
    c3 = Chunk(chunk_id="c3", content="neighbor context", metadata={"expansion_type": "neighbor"})
    
    # Duplicate of exact seed, but as neighbor
    c4 = Chunk(
        chunk_id="c4",
        content="Hello  world",
        metadata={**provenance, "expansion_type": "neighbor"},
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
def test_assembler_preserves_same_text_with_distinct_provenance(
    left_metadata, right_metadata,
):
    chunks = [
        Chunk("left", "Same admission evidence", left_metadata),
        Chunk("right", " Same  admission evidence ", right_metadata),
    ]

    assembled = ContextAssembler().assemble([(chunks[0], 0.9), (chunks[1], 0.8)])

    assert [chunk.chunk_id for chunk, _ in assembled] == ["left", "right"]


def test_assembler_missing_source_metadata_uses_distinct_chunk_identity():
    chunks = [Chunk("left", "Same text", {}), Chunk("right", " same  text ", {})]

    assembled = ContextAssembler().assemble([(chunks[0], 0.9), (chunks[1], 0.8)])

    assert [chunk.chunk_id for chunk, _ in assembled] == ["left", "right"]


@pytest.mark.parametrize(
    "right_content,right_metadata",
    [
        ("Same text", {"doc_id": "b", "page_number": 1}),
        ("Conflicting text", {"doc_id": "a", "page_number": 1}),
    ],
)
def test_assembler_same_chunk_id_does_not_hide_conflicting_evidence(
    right_content, right_metadata,
):
    chunks = [
        Chunk("collision", "Same text", {"doc_id": "a", "page_number": 1}),
        Chunk("collision", right_content, right_metadata),
    ]

    assembled = ContextAssembler().assemble([(chunks[0], 0.9), (chunks[1], 0.8)])

    assert [chunk for chunk, _ in assembled] == chunks


def test_assembler_does_not_mutate_chunks_or_metadata():
    chunks = [
        Chunk("a", "Evidence", {"doc_id": "doc", "page_number": 1}),
        Chunk("b", " evidence ", {"doc_id": "doc", "page_number": 1}),
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
