import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from src.config import Settings, settings
from src.retrieval.embedder import Embedder


def test_settings_reads_hf_token_from_environment(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_from_environment")

    loaded = Settings(_env_file=None)

    assert loaded.HF_TOKEN == "hf_from_environment"

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
        mock_sentence_transformer.assert_called_once_with(
            "test-model",
            cache_folder=mock_settings.HUGGINGFACE_CACHE_DIR,
            local_files_only=True,
        )
        
    # Should use provided model
    mock_sentence_transformer.reset_mock()
    embedder = Embedder(model_name="custom-model")
    mock_sentence_transformer.assert_called_once_with(
        "custom-model",
        cache_folder=settings.HUGGINGFACE_CACHE_DIR,
        local_files_only=True,
    )

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


def test_model_is_reused_within_process(mock_sentence_transformer):
    first = Embedder("shared-test-model")
    second = Embedder("shared-test-model")

    assert first.model is second.model
    mock_sentence_transformer.assert_called_once_with(
        "shared-test-model",
        cache_folder=settings.HUGGINGFACE_CACHE_DIR,
        local_files_only=True,
    )


def test_remote_model_download_uses_configured_hf_token(mock_sentence_transformer, monkeypatch):
    mock_sentence_transformer.side_effect = [OSError("not cached"), MagicMock()]
    monkeypatch.setattr(settings, "HF_TOKEN", "  hf_test_token  ")

    Embedder("authenticated-download-model")

    assert mock_sentence_transformer.call_args_list[1].kwargs == {
        "cache_folder": settings.HUGGINGFACE_CACHE_DIR,
        "token": "hf_test_token",
    }


def test_remote_model_download_omits_blank_hf_token(mock_sentence_transformer, monkeypatch):
    mock_sentence_transformer.side_effect = [OSError("not cached"), MagicMock()]
    monkeypatch.setattr(settings, "HF_TOKEN", "  ")

    Embedder("anonymous-download-model")

    assert mock_sentence_transformer.call_args_list[1].kwargs == {
        "cache_folder": settings.HUGGINGFACE_CACHE_DIR,
    }


def test_embed_rejects_invalid_items_and_dimension_changes(mock_sentence_transformer):
    embedder = Embedder("validation-test-model")

    with pytest.raises(TypeError, match="single string"):
        embedder.embed("not a list")
    with pytest.raises(ValueError, match=r"texts\[1\]"):
        embedder.embed(["valid", "   "])

    embedder.model.encode.side_effect = lambda value, **kwargs: np.ones((len(value), 2))
    assert embedder.embed(["first"]).shape == (1, 2)
    embedder.model.encode.side_effect = lambda value, **kwargs: np.ones((len(value), 3))
    with pytest.raises(ValueError, match="dimension changed"):
        embedder.embed(["second"])


def test_embed_rejects_non_finite_output(mock_sentence_transformer):
    embedder = Embedder("non-finite-test-model")
    embedder.model.encode.side_effect = lambda value, **kwargs: np.array([[np.nan, 1.0]])
    with pytest.raises(ValueError, match="NaN or infinite"):
        embedder.embed(["xin chào"])
