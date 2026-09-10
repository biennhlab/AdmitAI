import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from src.retrieval.embedder import Embedder

@pytest.fixture
def mock_sentence_transformer():
    with patch("src.retrieval.embedder.SentenceTransformer") as mock_st:
        mock_model = MagicMock()
        mock_st.return_value = mock_model
        
        # Default mock encode behavior
        def fake_encode(texts, normalize_embeddings=False):
            if isinstance(texts, str):
                return np.array([1.0, 0.0, 0.0])
            return np.array([[1.0, 0.0, 0.0] for _ in texts])
            
        mock_model.encode.side_effect = fake_encode
        yield mock_st

def test_embedder_initialization(mock_sentence_transformer):
    # Should use config model if not provided
    with patch("src.retrieval.embedder.settings") as mock_settings:
        mock_settings.EMBEDDING_MODEL = "test-model"
        embedder = Embedder()
        mock_sentence_transformer.assert_called_once_with("test-model", local_files_only=True)
        
    # Should use provided model
    mock_sentence_transformer.reset_mock()
    embedder = Embedder(model_name="custom-model")
    mock_sentence_transformer.assert_called_once_with("custom-model", local_files_only=True)

def test_embed_empty_list(mock_sentence_transformer):
    embedder = Embedder("dummy")
    result = embedder.embed([])
    assert isinstance(result, np.ndarray)
    assert result.size == 0
    # Should not call encode on empty list
    embedder.model.encode.assert_not_called()

def test_embed_valid_texts(mock_sentence_transformer):
    embedder = Embedder("dummy")
    texts = ["hello", "world"]
    
    # Setup mock to return specific embeddings
    expected_embeddings = np.array([[1.0, 2.0], [3.0, 4.0]])
    embedder.model.encode.side_effect = lambda t, **kwargs: expected_embeddings
    
    result = embedder.embed(texts)
    
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, expected_embeddings)
    embedder.model.encode.assert_called_once_with(texts, normalize_embeddings=False)

def test_embed_query_valid(mock_sentence_transformer):
    embedder = Embedder("dummy")
    query = "test query"
    
    expected_embedding = np.array([1.0, 2.0])
    embedder.model.encode.side_effect = lambda q, **kwargs: expected_embedding
    
    result = embedder.embed_query(query)
    
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, expected_embedding)
    embedder.model.encode.assert_called_once_with(query, normalize_embeddings=False)

def test_embed_query_invalid(mock_sentence_transformer):
    embedder = Embedder("dummy")
    
    with pytest.raises(ValueError, match="Query cannot be empty"):
        embedder.embed_query("")
        
    with pytest.raises(ValueError, match="Query cannot be empty"):
        embedder.embed_query("   \n")
        
    with pytest.raises(ValueError, match="Query cannot be empty"):
        embedder.embed_query(None)
