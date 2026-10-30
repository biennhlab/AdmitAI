"""Offline model resolution must not leak into Hugging Face metadata requests."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from huggingface_hub.errors import LocalEntryNotFoundError
from transformers import AutoTokenizer

from src.config import settings
from src.retrieval.embedder import Embedder
from src.runtime_resources import StartupCheckError


def test_cached_bge_tokenizer_never_requests_hub_metadata(monkeypatch):
    """Exercise the installed tokenizer's high-vocabulary/Mistral detection.

    Only the real tokenizer is loaded; SentenceTransformer weights are mocked.
    The original hub-ID constructor reaches model_info despite local_files_only.
    """
    monkeypatch.setattr(settings, 'MODELS_LOCAL_ONLY', True)
    monkeypatch.setattr(settings, 'HF_TOKEN', 'expired-test-token')
    loaded = []

    def tokenizer_only(path, **kwargs):
        tokenizer = AutoTokenizer.from_pretrained(
            path, cache_dir=kwargs['cache_folder'], local_files_only=True)
        loaded.append((path, tokenizer))
        return SimpleNamespace(get_sentence_embedding_dimension=lambda: 1024)

    with (
        patch('transformers.tokenization_utils_base.is_offline_mode', return_value=False),
        patch('huggingface_hub.model_info', side_effect=RuntimeError('Unexpected Hugging Face metadata request')) as info,
        patch('requests.sessions.Session.request', side_effect=AssertionError('No HTTP allowed')) as http,
        patch('src.retrieval.embedder.SentenceTransformer', side_effect=tokenizer_only),
    ):
        embedder = Embedder('BAAI/bge-m3')
    assert embedder.dimension == 1024
    assert len(loaded) == 1 and Path(loaded[0][0]).is_dir()
    assert loaded[0][1].vocab_size > 100000
    info.assert_not_called()
    http.assert_not_called()


def test_cached_snapshot_resolves_before_model_load_without_credentials(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, 'MODELS_LOCAL_ONLY', True)
    monkeypatch.setattr(settings, 'HF_TOKEN', 'expired-test-token')
    with (
        patch('src.retrieval.embedder.snapshot_download', return_value=str(tmp_path)) as resolve,
        patch('src.retrieval.embedder.SentenceTransformer') as factory,
    ):
        Embedder('cached-model')
    resolve.assert_called_once_with(repo_id='cached-model', cache_dir=settings.HUGGINGFACE_CACHE_DIR,
                                    local_files_only=True)
    factory.assert_called_once_with(str(tmp_path.resolve()), cache_folder=settings.HUGGINGFACE_CACHE_DIR,
                                    local_files_only=True)


def test_explicit_local_directory_does_not_require_hub_resolution(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, 'MODELS_LOCAL_ONLY', True)
    with (
        patch('src.retrieval.embedder.snapshot_download', side_effect=AssertionError('No Hub lookup')) as resolve,
        patch('src.retrieval.embedder.SentenceTransformer') as factory,
    ):
        first, second = Embedder(str(tmp_path)), Embedder(str(tmp_path))
    assert first.model is second.model
    resolve.assert_not_called()
    factory.assert_called_once_with(str(tmp_path.resolve()), cache_folder=settings.HUGGINGFACE_CACHE_DIR,
                                    local_files_only=True)


def test_missing_local_snapshot_fails_before_model_construction(monkeypatch):
    monkeypatch.setattr(settings, 'MODELS_LOCAL_ONLY', True)
    with (
        patch('src.retrieval.embedder.snapshot_download', side_effect=LocalEntryNotFoundError('not cached')) as resolve,
        patch('src.retrieval.embedder.SentenceTransformer') as factory,
    ):
        with pytest.raises(StartupCheckError) as caught:
            Embedder('missing-model')
    assert caught.value.code == 'MODEL_UNAVAILABLE'
    assert resolve.call_count == 1 and resolve.call_args.kwargs['local_files_only'] is True
    factory.assert_not_called()


@pytest.mark.parametrize('error', [OSError('missing weights'), ValueError('broken config'),
                                   AttributeError('incomplete snapshot')])
def test_incomplete_snapshot_never_retries_online(tmp_path, monkeypatch, error):
    monkeypatch.setattr(settings, 'MODELS_LOCAL_ONLY', True)
    with (
        patch('src.retrieval.embedder.snapshot_download', return_value=str(tmp_path)) as resolve,
        patch('src.retrieval.embedder.SentenceTransformer', side_effect=error) as factory,
    ):
        with pytest.raises(StartupCheckError) as caught:
            Embedder('incomplete-model')
    assert caught.value.code == 'MODEL_UNAVAILABLE'
    assert resolve.call_count == factory.call_count == 1


def test_failed_local_load_is_not_cached(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, 'MODELS_LOCAL_ONLY', True)
    restored = SimpleNamespace(get_sentence_embedding_dimension=lambda: 1024)
    with (
        patch('src.retrieval.embedder.snapshot_download', return_value=str(tmp_path)),
        patch('src.retrieval.embedder.SentenceTransformer', side_effect=[OSError('incomplete'), restored]) as factory,
    ):
        with pytest.raises(StartupCheckError):
            Embedder('restored-model')
        first, second = Embedder('restored-model'), Embedder('restored-model')
    assert first.model is second.model is restored
    assert factory.call_count == 2


def test_model_cache_does_not_cross_cache_directories_or_offline_policies(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, 'MODELS_LOCAL_ONLY', True)
    factory = Mock(side_effect=lambda *args, **kwargs: SimpleNamespace(get_sentence_embedding_dimension=lambda: 1024))
    with (
        patch('src.retrieval.embedder.snapshot_download', return_value=str(tmp_path)),
        patch('src.retrieval.embedder.SentenceTransformer', factory),
    ):
        monkeypatch.setattr(settings, 'HUGGINGFACE_CACHE_DIR', str(tmp_path / 'one'))
        first = Embedder('shared-model')
        assert Embedder('shared-model').model is first.model
        monkeypatch.setattr(settings, 'HUGGINGFACE_CACHE_DIR', str(tmp_path / 'two'))
        second = Embedder('shared-model')
        monkeypatch.setattr(settings, 'MODELS_LOCAL_ONLY', False)
        third = Embedder('shared-model')
    assert first.model is not second.model and second.model is not third.model
    assert factory.call_count == 3
