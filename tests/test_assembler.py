from src.generation.assembler import ContextAssembler
from src.ingestion.chunker import Chunk

def test_assembler_prioritizes_and_deduplicates():
    assembler = ContextAssembler(max_chars=100)
    
    # Exact seeds
    c1 = Chunk(chunk_id="c1", content="hello world", metadata={"expansion_type": "exact"})
    
    # Siblings
    c2 = Chunk(chunk_id="c2", content="some table part", metadata={"expansion_type": "table_sibling"})
    
    # Neighbors
    c3 = Chunk(chunk_id="c3", content="neighbor context", metadata={"expansion_type": "neighbor"})
    
    # Duplicate of exact seed, but as neighbor
    c4 = Chunk(chunk_id="c4", content="Hello  world", metadata={"expansion_type": "neighbor"})
    
    retrieved = [(c3, 0.8), (c4, 0.7), (c1, 0.9), (c2, 0.85)]
    
    final = assembler.assemble(retrieved)
    
    # Priority: exact (c1) -> sibling (c2) -> neighbor (c3)
    # c4 should be dropped as duplicate of c1
    ids = [c.chunk_id for c, _ in final]
    assert ids == ["c1", "c2", "c3"]

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
