import pytest
import numpy as np
from unittest.mock import MagicMock

from src.ingestion.chunker import Chunk
from src.retrieval.dense_search import NaiveDenseSearch
from src.retrieval.embedder import Embedder

@pytest.fixture
def dummy_chunks():
    return [
        Chunk(chunk_id="1", content="chunk 1"),
        Chunk(chunk_id="2", content="chunk 2"),
        Chunk(chunk_id="3", content="chunk 3"),
    ]

@pytest.fixture
def mock_embedder():
    embedder = MagicMock(spec=Embedder)
    # Default behavior: all chunks get [1.0, 0.0], query gets [1.0, 0.0]
    embedder.embed.return_value = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    embedder.embed_query.return_value = np.array([1.0, 0.0])
    return embedder


# 1. Cosine similarity
def test_cosine_similarity_basic(mock_embedder):
    search = NaiveDenseSearch(mock_embedder, [])
    
    # Same vector -> ~1.0
    v1 = np.array([1.0, 2.0, 3.0])
    assert search.cosine_similarity(v1, v1) == pytest.approx(1.0)
    
    # Same direction, different scale -> ~1.0
    v2 = np.array([2.0, 4.0, 6.0])
    assert search.cosine_similarity(v1, v2) == pytest.approx(1.0)
    
    # Orthogonal -> ~0.0
    v_ortho = np.array([-2.0, 1.0, 0.0]) # dot(v1, v_ortho) = -2 + 2 = 0
    assert search.cosine_similarity(v1, v_ortho) == pytest.approx(0.0)
    
    # Opposite -> ~-1.0
    v_opp = np.array([-1.0, -2.0, -3.0])
    assert search.cosine_similarity(v1, v_opp) == pytest.approx(-1.0)

def test_cosine_similarity_edge_cases(mock_embedder):
    search = NaiveDenseSearch(mock_embedder, [])
    
    v_normal = np.array([1.0, 0.0])
    v_zero = np.array([0.0, 0.0])
    
    # Zero vector against normal vector
    assert search.cosine_similarity(v_normal, v_zero) == pytest.approx(0.0)
    assert search.cosine_similarity(v_zero, v_normal) == pytest.approx(0.0)
    
    # Zero against zero
    assert search.cosine_similarity(v_zero, v_zero) == pytest.approx(0.0)
    
    # NaN and Inf handling
    v_nan = np.array([np.nan, 1.0])
    assert search.cosine_similarity(v_normal, v_nan) == pytest.approx(0.0)
    
    v_inf = np.array([np.inf, 1.0])
    assert search.cosine_similarity(v_normal, v_inf) == pytest.approx(0.0)

def test_cosine_similarity_dimension_mismatch(mock_embedder):
    search = NaiveDenseSearch(mock_embedder, [])
    v1 = np.array([1.0, 0.0])
    v2 = np.array([1.0, 0.0, 0.0])
    
    # Numpy raises ValueError for dimension mismatch in dot product
    with pytest.raises(ValueError):
        search.cosine_similarity(v1, v2)

# 2. Ranking correctness
def test_ranking_correctness(mock_embedder):
    # Setup embedder to return specific vectors for chunks
    # Chunk 1: [1, 0] (score = 1.0)
    # Chunk 2: [0, 1] (score = 0.0)
    # Chunk 3: [-1, 0] (score = -1.0)
    # Chunk 4: [1, 0.1] (score ~0.99)
    # Chunk 5: [2, 0] (score = 1.0) -> tie with Chunk 1
    
    chunks = [
        Chunk(chunk_id="1", content="C1"),
        Chunk(chunk_id="2", content="C2"),
        Chunk(chunk_id="3", content="C3"),
        Chunk(chunk_id="4", content="C4"),
        Chunk(chunk_id="5", content="C5"),
    ]
    
    chunk_embeddings = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [-1.0, 0.0],
        [1.0, 0.1],
        [2.0, 0.0]
    ])
    mock_embedder.embed.return_value = chunk_embeddings
    
    # Query is [1, 0]
    mock_embedder.embed_query.return_value = np.array([1.0, 0.0])
    
    search = NaiveDenseSearch(mock_embedder, chunks)
    results = search.search("query", top_k=5)
    
    assert len(results) == 5
    
    # Check ranking
    # Top 2 should be C1 and C5 (both score 1.0)
    top_2_ids = {results[0][0].chunk_id, results[1][0].chunk_id}
    assert top_2_ids == {"1", "5"}
    assert results[0][1] == pytest.approx(1.0)
    assert results[1][1] == pytest.approx(1.0)
    
    # 3rd should be C4
    assert results[2][0].chunk_id == "4"
    assert results[2][1] > 0.0 and results[2][1] < 1.0
    
    # 4th should be C2
    assert results[3][0].chunk_id == "2"
    assert results[3][1] == pytest.approx(0.0)
    
    # 5th should be C3
    assert results[4][0].chunk_id == "3"
    assert results[4][1] == pytest.approx(-1.0)

# 3. top_k behavior
def test_top_k_behavior(dummy_chunks, mock_embedder):
    search = NaiveDenseSearch(mock_embedder, dummy_chunks)
    
    # top_k = 1
    res = search.search("query", top_k=1)
    assert len(res) == 1
    
    # top_k = exact length
    res = search.search("query", top_k=len(dummy_chunks))
    assert len(res) == len(dummy_chunks)
    
    # top_k > length
    res = search.search("query", top_k=100)
    assert len(res) == len(dummy_chunks)
    
    # top_k = 0
    res = search.search("query", top_k=0)
    assert len(res) == 0
    
    # top_k < 0
    res = search.search("query", top_k=-5)
    assert len(res) == 0

# 4. Input edge cases
def test_search_empty_chunks(mock_embedder):
    search = NaiveDenseSearch(mock_embedder, [])
    assert search.search("query", top_k=10) == []
    
def test_search_invalid_query(dummy_chunks, mock_embedder):
    search = NaiveDenseSearch(mock_embedder, dummy_chunks)
    
    # NaiveDenseSearch should check query before embedding
    assert search.search("", top_k=10) == []
    assert search.search("   ", top_k=10) == []
    assert search.search(None, top_k=10) == []

def test_search_zero_vector_query(dummy_chunks, mock_embedder):
    # Query embedder returns zero vector
    mock_embedder.embed_query.return_value = np.array([0.0, 0.0])
    search = NaiveDenseSearch(mock_embedder, dummy_chunks)
    
    assert search.search("query", top_k=10) == []

# 5. Embedding integrity & hidden bugs
def test_dimension_mismatch(dummy_chunks, mock_embedder):
    # Chunks are 2D: [1.0, 0.0]
    # Query is 3D: [1.0, 0.0, 0.0]
    mock_embedder.embed_query.return_value = np.array([1.0, 0.0, 0.0])
    
    search = NaiveDenseSearch(mock_embedder, dummy_chunks)
    
    with pytest.raises(ValueError, match="Dimension mismatch"):
        search.search("query", top_k=10)

def test_embed_called_only_once_for_corpus(dummy_chunks, mock_embedder):
    search = NaiveDenseSearch(mock_embedder, dummy_chunks)
    
    # Embed should be called exactly once during init
    mock_embedder.embed.assert_called_once()
    
    # Reset mock to track new calls
    mock_embedder.reset_mock()
    mock_embedder.embed_query.return_value = np.array([1.0, 0.0])
    
    # Multiple searches should NOT re-embed the corpus
    search.search("query 1", top_k=2)
    search.search("query 2", top_k=2)
    
    mock_embedder.embed.assert_not_called()
    assert mock_embedder.embed_query.call_count == 2
