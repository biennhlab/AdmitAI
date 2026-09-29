from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    RateLimitError,
)

from api.main import app, health_check, startup_event
from api.routers import chat
from src.database.connection import get_db
from src.generation.llm_client import (
    LLMAuthenticationError,
    LLMClient,
    LLMConnectionError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from src.generation.rag_chain import RAGChain, RAGResponse
from src.generation.session_memory import SessionMemory


class FakeDB:
    def __init__(self, *, commit_error: Exception | None = None):
        self.commit_error = commit_error
        self.records = []
        self.rollback_called = False

    def add_all(self, records):
        self.records.extend(records)

    async def commit(self):
        if self.commit_error:
            raise self.commit_error

    async def rollback(self):
        self.rollback_called = True


class FakeRAG:
    def __init__(self, result=None, error: Exception | None = None):
        self.result = result or RAGResponse("Câu trả lời [1]", [], "general")
        self.error = error

    def answer(self, question, history):
        if self.error:
            raise self.error
        return self.result

    def answer_stream(self, question, history):
        if self.error:
            raise self.error
        yield {
            "type": "metadata",
            "citations": self.result.citations,
            "route_type": self.result.route_type,
        }
        yield {"type": "chunk", "text": self.result.answer}


@pytest.fixture(autouse=True)
def restore_chat_state():
    original_chain = chat.rag_chain
    original_manifest = chat.rag_manifest
    original_internal_error = chat.rag_initialization_error
    original_public_error = chat.rag_public_error
    original_memory = chat.session_memory
    original_components = app.state.components
    had_qdrant_client = hasattr(app.state, "qdrant_client")
    original_qdrant_client = getattr(app.state, "qdrant_client", None)
    app.state.components = {
        "database": {"ready": True},
        "local_index": {"ready": True},
        "qdrant": {"ready": True},
        "llm": {"configured": True},
    }
    chat.session_memory = SessionMemory()
    yield
    chat.rag_chain = original_chain
    chat.rag_manifest = original_manifest
    chat.rag_initialization_error = original_internal_error
    chat.rag_public_error = original_public_error
    chat.session_memory = original_memory
    app.state.components = original_components
    if had_qdrant_client:
        app.state.qdrant_client = original_qdrant_client
    elif hasattr(app.state, "qdrant_client"):
        del app.state.qdrant_client
    app.dependency_overrides.clear()


async def request(method: str, path: str, *, json=None, db=None):
    fake_db = db or FakeDB()

    async def override_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_db
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.request(method, path, json=json)


def sse_payloads(response: httpx.Response) -> list[dict]:
    return [
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]


def test_initialize_rag_constructs_one_process_wide_reranker() -> None:
    chunk = MagicMock()
    chunk.chunk_id = "chunk-1"
    chunk.content = "tuition details"
    chunk.metadata = {"doc_id": "doc-1", "source": "source.pdf"}
    retriever = MagicMock()
    retriever.search.return_value = [(chunk, 0.8)]
    reranker = MagicMock()
    reranker.rerank.return_value = [(chunk, 0.8, 0.95)]
    llm = MagicMock()
    llm.generate.return_value = "answer [1]"
    manifest = {
        "document_count": 1,
        "chunk_count": 1,
        "embedding_model": "BAAI/bge-m3",
        "retrieval_backend": "test",
    }

    with (
        patch("api.routers.chat.Reranker", return_value=reranker) as reranker_factory,
        patch("api.routers.chat.LLMClient", return_value=llm),
    ):
        chat.initialize_rag(retriever=retriever, manifest=manifest)
        assert chat.rag_chain is not None
        chat.rag_chain.answer("tuition details")
        chat.rag_chain.answer("tuition details")

    reranker_factory.assert_called_once_with(
        model_name=chat.settings.RERANKER_MODEL,
        batch_size=chat.settings.RERANK_BATCH_SIZE,
        backend=chat.settings.RERANKER_BACKEND,
        remote_provider=chat.settings.RERANKER_REMOTE_PROVIDER,
        remote_url=chat.settings.RERANKER_REMOTE_URL,
        remote_api_token=chat.settings.RERANKER_REMOTE_API_TOKEN.get_secret_value(),
        remote_timeout_seconds=chat.settings.RERANKER_REMOTE_TIMEOUT_SECONDS,
        remote_max_retries=chat.settings.RERANKER_REMOTE_MAX_RETRIES,
        remote_top_k_limit=chat.settings.RERANKER_REMOTE_TOP_K_LIMIT,
    )
    assert chat.rag_chain.reranker is reranker
    assert chat.rag_chain.self_rag.llm_client is llm
    assert chat.rag_chain.query_rewriter.llm_client is llm
    assert reranker.rerank.call_count == 2


@pytest.mark.asyncio
async def test_health_reports_each_ready_component_without_secrets():
    chat.rag_chain = object()
    chat.rag_manifest = {
        "retrieval_backend": "hybrid_qdrant_bm25_rrf",
        "document_count": 10,
        "chunk_count": 20,
        "embedding_model": "BAAI/bge-m3",
    }
    chat.rag_public_error = None

    response = await request("GET", "/api/health")
    payload = response.json()

    assert response.status_code == 200
    assert payload["status"] == "healthy"
    assert payload["components"]["database"]["ready"] is True
    assert payload["components"]["local_index"]["ready"] is True
    assert payload["components"]["qdrant"]["ready"] is True
    assert payload["components"]["rag"]["ready"] is True
    assert payload["components"]["rag"]["reranker"] == {
        "backend": chat.settings.RERANKER_BACKEND,
        "ready": True,
    }
    assert payload["components"]["llm"] == {"configured": True}
    assert "api_key" not in str(payload).lower()


@pytest.mark.asyncio
async def test_startup_degrades_cleanly_when_local_index_fails():
    with (
        patch("api.main.init_db", new=AsyncMock()),
        patch("api.main.load_local_index", side_effect=RuntimeError("D:/secret/index")),
        patch("api.main._llm_is_configured", return_value=True),
    ):
        await startup_event()

    payload = await health_check()

    assert payload["status"] == "degraded"
    assert payload["components"]["database"]["ready"] is True
    assert payload["components"]["local_index"]["ready"] is False
    assert payload["components"]["qdrant"]["ready"] is False
    assert payload["components"]["rag"]["ready"] is False
    assert "D:/secret/index" not in str(payload)


@pytest.mark.asyncio
async def test_startup_degrades_cleanly_when_qdrant_fails():
    manifest = {
        "document_count": 1,
        "chunk_count": 1,
        "embedding_model": "BAAI/bge-m3",
    }
    with (
        patch("api.main.init_db", new=AsyncMock()),
        patch("api.main.load_local_index", return_value=([object()], manifest)),
        patch("api.main.load_hybrid_retriever", side_effect=RuntimeError("qdrant secret")),
        patch("api.main._llm_is_configured", return_value=True),
    ):
        await startup_event()

    payload = await health_check()

    assert payload["status"] == "degraded"
    assert payload["components"]["local_index"]["ready"] is True
    assert payload["components"]["qdrant"]["ready"] is False
    assert payload["components"]["rag"]["ready"] is False
    assert "qdrant secret" not in str(payload)


@pytest.mark.asyncio
async def test_health_detects_qdrant_failure_after_startup():
    qdrant = MagicMock()
    qdrant.collection_exists.side_effect = ConnectionError("connection reset")
    app.state.qdrant_client = qdrant
    app.state.components["local_index"]["chunks"] = 3178
    app.state.components["qdrant"] = {"ready": True, "chunks": 3178}
    chat.rag_chain = object()
    chat.rag_manifest = {"embedding_model": "BAAI/bge-m3"}

    response = await request("GET", "/api/health")
    payload = response.json()

    assert payload["status"] == "degraded"
    assert payload["components"]["qdrant"]["ready"] is False
    assert payload["components"]["rag"]["ready"] is False
    assert "connection reset" not in str(payload)


@pytest.mark.asyncio
async def test_chat_returns_safe_503_when_retrieval_fails_after_startup():
    class BrokenRetriever:
        def search(self, _query, top_k):
            raise ConnectionError(f"Qdrant reset with secret; top_k={top_k}")

    chat.rag_chain = RAGChain(
        BrokenRetriever(),
        MagicMock(),
        min_score=None,
    )

    response = await request("POST", "/api/chat", json={"message": "Học phí PTIT là bao nhiêu?"})

    assert response.status_code == 503
    assert response.json() == {
        "detail": {
            "code": "RAG_UNAVAILABLE",
            "message": chat.RAG_UNAVAILABLE_MESSAGE,
        }
    }
    assert "secret" not in response.text


@pytest.mark.asyncio
async def test_chat_returns_public_503_when_rag_is_unavailable():
    chat.mark_rag_unavailable(RuntimeError("D:/secret/index"))

    response = await request("POST", "/api/chat", json={"message": "Học phí?"})

    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "RAG_UNAVAILABLE"
    assert "D:/secret/index" not in response.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "expected_status", "expected_code"),
    [
        (LLMTimeoutError("provider secret"), 504, "LLM_TIMEOUT"),
        (LLMAuthenticationError("provider secret"), 502, "LLM_AUTHENTICATION_FAILED"),
        (LLMConnectionError("provider secret"), 503, "LLM_UNAVAILABLE"),
        (LLMRateLimitError("provider secret"), 503, "LLM_RATE_LIMITED"),
    ],
)
async def test_chat_classifies_provider_errors_and_process_stays_healthy(
    error,
    expected_status,
    expected_code,
):
    chat.rag_chain = FakeRAG(error=error)
    chat.rag_manifest = {"embedding_model": "BAAI/bge-m3"}

    response = await request("POST", "/api/chat", json={"message": "Điểm chuẩn?"})
    health_response = await request("GET", "/api/health")

    assert response.status_code == expected_status
    assert response.json()["detail"]["code"] == expected_code
    assert "provider secret" not in response.text
    assert health_response.status_code == 200
    assert health_response.json()["components"]["rag"]["ready"] is True


@pytest.mark.asyncio
async def test_unexpected_chat_error_returns_safe_500():
    chat.rag_chain = FakeRAG(error=ValueError("LLM_API_KEY=secret"))

    response = await request("POST", "/api/chat", json={"message": "Chỉ tiêu?"})

    assert response.status_code == 500
    assert response.json()["detail"]["code"] == "INTERNAL_ERROR"
    assert "LLM_API_KEY" not in response.text
    assert "secret" not in response.text


@pytest.mark.asyncio
async def test_database_persistence_failure_does_not_discard_generated_answer():
    chat.rag_chain = FakeRAG()
    failing_db = FakeDB(commit_error=RuntimeError("database locked"))

    response = await request(
        "POST",
        "/api/chat",
        json={"message": "Phương thức tuyển sinh?"},
        db=failing_db,
    )

    assert response.status_code == 200
    payloads = sse_payloads(response)
    assert payloads[-1] == {"type": "chunk", "text": "Câu trả lời [1]"}
    assert failing_db.rollback_called is True


def sdk_error(error_type):
    request = httpx.Request("POST", "https://provider.invalid/v1/chat")
    if error_type is APITimeoutError:
        return error_type(request=request)
    if error_type is APIConnectionError:
        return error_type(request=request)
    response = httpx.Response(401 if error_type is AuthenticationError else 429, request=request)
    return error_type("provider details", response=response, body=None)


@pytest.mark.parametrize(
    ("sdk_error_type", "domain_error_type"),
    [
        (APITimeoutError, LLMTimeoutError),
        (AuthenticationError, LLMAuthenticationError),
        (RateLimitError, LLMRateLimitError),
        (APIConnectionError, LLMConnectionError),
    ],
)
def test_llm_client_classifies_openai_compatible_errors(sdk_error_type, domain_error_type):
    with patch("src.generation.llm_client.OpenAI") as openai_class:
        provider = MagicMock()
        provider.chat.completions.create.side_effect = sdk_error(sdk_error_type)
        openai_class.return_value = provider
        client = LLMClient("not-a-real-key", "model", base_url="fake-url", timeout_seconds=2, max_retries=0)

        with pytest.raises(domain_error_type):
            client.generate("system", [{"role": "user", "content": "hello"}])

        openai_class.assert_called_once_with(
            base_url="fake-url",
            api_key="not-a-real-key",
            timeout=2,
            max_retries=0,
        )
