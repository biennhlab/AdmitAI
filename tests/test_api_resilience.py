from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from types import SimpleNamespace
import json as jsonlib

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
from api.schemas import ChatResponse


class FakeDB:
    def __init__(self, *, commit_error: Exception | None = None):
        self.commit_error = commit_error
        self.records = []
        self.rollback_called = False
        self.commit_called = False
        self.operations = []

    async def get(self, model, identity):
        self.operations.append("get")
        return next((record for record in self.records if isinstance(record, model) and record.id == identity), None)

    async def flush(self):
        self.operations.append("flush")

    def add_all(self, records):
        self.operations.append("add")
        self.records.extend(records)

    async def commit(self):
        self.commit_called = True
        self.operations.append("commit")
        if self.commit_error:
            raise self.commit_error

    async def rollback(self):
        self.rollback_called = True
        self.operations.append("rollback")


class FakeRAG:
    def __init__(self, result=None, error: Exception | None = None):
        self.result = result or RAGResponse("Câu trả lời [1]", [], "general")
        self.error = error

    def answer(self, question, history):
        if self.error:
            raise self.error
        return self.result

    def answer_stream(self, question, history):
        result = self.answer(question, history)
        yield {"type": "chunk", "text": result.answer}
        yield {"type": "metadata", "citations": result.citations, "route_type": result.route_type}


@pytest.fixture(autouse=True)
def mock_reranker_model():
    # Health checks may initialize the real chain; keep model I/O out of API tests.
    with patch("src.retrieval.reranker.CrossEncoder") as factory:
        factory.return_value.predict.side_effect = lambda pairs: [0.0] * len(pairs)
        yield factory


@pytest.fixture(autouse=True)
def restore_chat_state(monkeypatch):
    original_chain = chat.rag_chain
    original_manifest = chat.rag_manifest
    original_internal_error = chat.rag_initialization_error
    original_public_error = chat.rag_public_error
    original_memory = chat.session_memory
    original_components = app.state.components
    had_qdrant_client = hasattr(app.state, "qdrant_client")
    original_qdrant_client = getattr(app.state, "qdrant_client", None)
    # Exercise the real health handler with controlled external dependencies.
    index = MagicMock()
    index.__truediv__.return_value.is_file.return_value = True
    monkeypatch.setattr("api.main.Path", MagicMock(return_value=index))
    monkeypatch.setattr("api.main._llm_is_configured", lambda: True)
    connection = AsyncMock()
    engine = MagicMock()
    engine.connect.return_value.__aenter__ = AsyncMock(return_value=connection)
    engine.connect.return_value.__aexit__ = AsyncMock(return_value=False)
    monkeypatch.setattr("api.main.engine", engine)
    manifest = {"document_count": 1, "chunk_count": 0, "embedding_model": chat.settings.EMBEDDING_MODEL}
    monkeypatch.setattr("api.main.load_local_index", lambda: ([], manifest))
    monkeypatch.setattr("api.main.load_hybrid_retriever", MagicMock(side_effect=ConnectionError("test Qdrant offline")))
    qdrant = MagicMock()
    qdrant.collection_exists.return_value = True
    qdrant.count.return_value = SimpleNamespace(count=0)
    app.state.qdrant_client = qdrant
    app.state.components = {
        "database": {"ready": True},
        "local_index": {"ready": True},
        "qdrant": {"ready": True},
        "llm": {"configured": True},
    }
    chat.session_memory = SessionMemory()
    chat.rag_chain = None
    chat.rag_manifest = None
    chat.rag_initialization_error = None
    chat.rag_public_error = None
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


async def request(method: str, path: str, *, json=None, db=None, headers=None):
    fake_db = db or FakeDB()

    async def override_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_db
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.request(method, path, json=json, headers=headers)


def stream_events(response):
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = []
    for frame in response.text.split("\n\n"):
        if not frame:
            continue
        lines = frame.splitlines()
        data = [line[6:] for line in lines if line.startswith("data: ")]
        assert len(data) == 1, f"Invalid SSE framing: {frame!r}"
        events.append(("error" if "event: error" in lines else "message", jsonlib.loads(data[0])))
    return events


def response_payload(response, accept):
    if accept == "application/json":
        assert response.headers["content-type"].startswith("application/json")
        return response.json()
    events = stream_events(response)
    assert all(kind == "message" for kind, _ in events)
    metadata = [data for _, data in events if data["type"] == "metadata"]
    assert len(metadata) == 1
    return {key: value for key, value in {
        **metadata[0], "answer": "".join(data["text"] for _, data in events if data["type"] == "chunk"),
    }.items() if key != "type"}


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
async def test_health_recovery_skips_qdrant_when_llm_is_not_configured():
    app.state.components["qdrant"] = {"ready": False, "error": "offline"}
    app.state.qdrant_client = None
    load_hybrid = MagicMock()

    with (
        patch("api.main._llm_is_configured", return_value=False),
        patch("api.main.load_hybrid_retriever", load_hybrid),
    ):
        payload = await health_check()

    load_hybrid.assert_not_called()
    assert payload["components"]["llm"] == {"configured": False}
    assert payload["components"]["qdrant"]["ready"] is False
    assert payload["components"]["rag"]["ready"] is False


@pytest.mark.asyncio
async def test_health_recovery_closes_unpublished_qdrant_client_on_rag_failure():
    app.state.components["qdrant"] = {"ready": False, "error": "offline"}
    app.state.qdrant_client = None
    candidate = MagicMock()
    retriever = MagicMock()
    manifest = {
        "document_count": 1,
        "chunk_count": 0,
        "embedding_model": chat.settings.EMBEDDING_MODEL,
    }

    with (
        patch("api.main.load_local_index", return_value=([], manifest)),
        patch("api.main.load_hybrid_retriever", return_value=(candidate, retriever, manifest)),
        patch("api.main.chat.initialize_rag", side_effect=RuntimeError("initialization failed")),
    ):
        payload = await health_check()

    candidate.close.assert_called_once_with()
    assert app.state.qdrant_client is None
    assert payload["components"]["qdrant"]["ready"] is False
    assert payload["components"]["rag"]["ready"] is False


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
@pytest.mark.parametrize("accept", ["application/json", "text/event-stream"])
async def test_database_persistence_failure_does_not_discard_generated_answer(accept):
    chat.rag_chain = FakeRAG()
    failing_db = FakeDB(commit_error=RuntimeError("database locked"))

    response = await request(
        "POST",
        "/api/chat",
        json={"message": "Phương thức tuyển sinh?"},
        db=failing_db,
        headers={"Accept": accept},
    )

    assert response.status_code == 200
    assert response_payload(response, accept)["answer"] == "Câu trả lời [1]"
    assert failing_db.commit_called is True
    assert failing_db.operations == ["get", "add", "flush", "add", "commit", "rollback"]
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


def test_queued_exception_logging_preserves_traceback(caplog):
    try:
        raise TypeError("legacy SDK rejected a keyword")
    except TypeError as exc:
        queued_exception = exc

    chat._log_queued_exception("Chat generation failed at the provider", queued_exception)

    assert "Chat generation failed at the provider" in caplog.text
    assert "TypeError: legacy SDK rejected a keyword" in caplog.text
    assert "NoneType: None" not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("accept", ["application/json", "text/event-stream"])
async def test_success_schema_persistence_and_session_history(accept):
    from src.database.models import ChatMessage, ChatSession

    citation = {"source": "Admission", "chunk_id": "chunk-1", "doc_id": "doc-1",
                "marker": 1, "page": 3, "snippet": "Admission details", "score": 0.0123}
    chain = FakeRAG(RAGResponse("Admission details [1]", [citation], "general"))
    chain.answer = MagicMock(wraps=chain.answer)
    chat.rag_chain = chain
    db = FakeDB()
    payloads = []
    for question in ("First question", "Follow-up question"):
        body = {"message": question}
        if payloads:
            body["session_id"] = payloads[0]["session_id"]
        response = await request("POST", "/api/chat", json=body, db=db, headers={"Accept": accept})
        assert response.status_code == 200
        payload = response_payload(response, accept)
        parsed = ChatResponse.model_validate(payload)
        assert parsed.answer == "Admission details [1]"
        assert parsed.citations[0].score == citation["score"]
        assert parsed.citations[0].doc_id == "doc-1"
        assert parsed.citations[0].marker == 1
        payloads.append(payload)
    assert payloads[0]["session_id"] == payloads[1]["session_id"]
    assert chain.answer.call_args_list[0].args[1] == []
    assert chain.answer.call_args_list[1].args[1] == [
        {"role": "user", "content": "First question"},
        {"role": "assistant", "content": "Admission details [1]"},
    ]
    assert len([record for record in db.records if isinstance(record, ChatSession)]) == 1
    messages = [record for record in db.records if isinstance(record, ChatMessage)]
    assert [m.role for m in messages] == ["user", "assistant", "user", "assistant"]
    assert messages[1].citations[0]["doc_id"] == "doc-1"
    assert db.operations == ["get", "add", "flush", "add", "commit", "get", "add", "commit"]


@pytest.mark.asyncio
@pytest.mark.parametrize("accept", [None, "*/*", "text/event-stream", "application/json;q=0",
                                         "application/json;q=0.4, text/event-stream;q=0.9",
                                         "application/json, text/event-stream", "application/json;q=nan"])
async def test_accept_keeps_streaming_default(accept):
    chain = FakeRAG()
    chat.rag_chain = chain
    response = await request("POST", "/api/chat", json={"message": "question"},
                             headers={"Accept": accept} if accept else None)
    assert response_payload(response, None)["answer"] == chain.result.answer


@pytest.mark.asyncio
@pytest.mark.parametrize("accept", ["application/json", "APPLICATION/JSON; charset=utf-8",
                                         "text/event-stream;q=0.1, application/json;q=0.9"])
async def test_explicit_json_negotiation(accept):
    chat.rag_chain = FakeRAG()
    response = await request("POST", "/api/chat", json={"message": "question"}, headers={"Accept": accept})
    assert response.status_code == 200
    assert response_payload(response, "application/json")["answer"] == "Câu trả lời [1]"


@pytest.mark.asyncio
@pytest.mark.parametrize("accept", ["application/json", "text/event-stream"])
@pytest.mark.parametrize("body, expected", [({}, 422), ({"message": None}, 422),
    ({"message": []}, 422), ({"message": 42}, 422), ({"message": ""}, 400),
    ({"message": " \t\n"}, 400), ({"message": "hello", "session_id": {}}, 422)])
async def test_invalid_request_stops_before_generation(accept, body, expected):
    chain = MagicMock(spec=RAGChain)
    chat.rag_chain = chain
    db = FakeDB()
    response = await request("POST", "/api/chat", json=body, db=db, headers={"Accept": accept})
    assert response.status_code == expected
    assert "detail" in response.json()
    chain.answer.assert_not_called()
    chain.answer_stream.assert_not_called()
    assert not db.commit_called
    assert chat.session_memory.sessions == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("accept", ["application/json", "text/event-stream"])
async def test_unknown_session_and_extra_request_field_keep_existing_behavior(accept):
    chat.rag_chain = FakeRAG()
    response = await request("POST", "/api/chat", json={
        "message": "question", "session_id": "unknown-session", "legacy_extra": True,
    }, headers={"Accept": accept})
    payload = response_payload(response, accept)
    assert payload["session_id"] != "unknown-session"
    assert len(chat.session_memory.get_history(payload["session_id"])) == 2


ERROR_CASES = [
    (LLMTimeoutError, 504, "LLM_TIMEOUT"),
    (LLMAuthenticationError, 502, "LLM_AUTHENTICATION_FAILED"),
    (LLMConnectionError, 503, "LLM_UNAVAILABLE"),
    (LLMRateLimitError, 503, "LLM_RATE_LIMITED"),
    (chat.LLMProviderError, 502, "LLM_PROVIDER_ERROR"),
    (chat.RAGRetrievalError, 503, "RAG_UNAVAILABLE"),
    (ValueError, 500, "INTERNAL_ERROR"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("accept", ["application/json", "text/event-stream"])
@pytest.mark.parametrize("error_type, expected_status, code", ERROR_CASES)
async def test_pre_response_errors_preserve_status_code_and_cause(accept, error_type, expected_status, code, caplog):
    error = error_type("private dependency details")
    chat.rag_chain = FakeRAG(error=error)
    db = FakeDB()
    response = await request("POST", "/api/chat", json={"message": "question"}, db=db, headers={"Accept": accept})
    assert response.status_code == expected_status
    assert response.headers["content-type"].startswith("application/json")
    assert set(response.json()) == {"detail"}
    assert set(response.json()["detail"]) == {"code", "message"}
    assert response.json()["detail"]["code"] == code
    assert "private dependency details" not in response.text
    assert any(record.exc_info and record.exc_info[1] is error for record in caplog.records)
    assert not db.commit_called
    assert all(not history for history in chat.session_memory.sessions.values())


@pytest.mark.asyncio
@pytest.mark.parametrize("error_type, expected_status, code", ERROR_CASES)
async def test_midstream_errors_are_sse_and_do_not_persist_partial_answer(error_type, expected_status, code):
    class PartialRAG:
        def answer_stream(self, question, history):
            yield {"type": "chunk", "text": "Partial answer"}
            raise error_type("private dependency details")

    chat.rag_chain = PartialRAG()
    db = FakeDB()
    response = await request("POST", "/api/chat", json={"message": "question"}, db=db)
    events = stream_events(response)
    assert events[0] == ("message", {"type": "chunk", "text": "Partial answer"})
    assert len(events) == 2
    assert events[1][0] == "error"
    assert events[1][1]["code"] == code
    assert "private dependency details" not in response.text
    assert db.records == []
    assert all(not history for history in chat.session_memory.sessions.values())


@pytest.mark.asyncio
@pytest.mark.parametrize("after_first", [False, True])
@pytest.mark.parametrize("event", [None, {}, "text", {"type": "chunk"},
    {"type": "chunk", "text": 10}, {"type": "unknown"},
    {"type": "metadata", "citations": None, "route_type": "general"},
    {"type": "metadata", "citations": [{}], "route_type": "general"},
    {"type": "metadata", "citations": [], "route_type": 10},
    {"type": "metadata", "citations": [{"source": "x", "score": float("nan")}], "route_type": "general"}])
async def test_malformed_stream_events_fail_safely(event, after_first):
    class InvalidRAG:
        def answer_stream(self, question, history):
            if after_first:
                yield {"type": "chunk", "text": "Partial"}
            yield event

    chat.rag_chain = InvalidRAG()
    db = FakeDB()
    response = await request("POST", "/api/chat", json={"message": "question"}, db=db)
    if after_first:
        events = stream_events(response)
        assert events[-1] == ("error", {"code": "INTERNAL_ERROR", "message": chat.INTERNAL_ERROR_MESSAGE})
    else:
        assert response.status_code == 500
        assert response.json()["detail"]["code"] == "INTERNAL_ERROR"
    assert not db.commit_called


@pytest.mark.asyncio
@pytest.mark.parametrize("events", [[], [{"type": "chunk", "text": ""}],
    [{"type": "chunk", "text": "Partial"}],
    [{"type": "metadata", "citations": [], "route_type": "general"}]])
async def test_incomplete_stream_is_not_success(events):
    chat.rag_chain = SimpleNamespace(answer_stream=lambda *args: iter(events))
    db = FakeDB()
    response = await request("POST", "/api/chat", json={"message": "question"}, db=db)
    if events:
        assert stream_events(response)[-1][1]["code"] == "INTERNAL_ERROR"
    else:
        assert response.status_code == 500
    assert not db.commit_called


@pytest.mark.asyncio
@pytest.mark.parametrize("result", [None, {}, SimpleNamespace(answer=""),
    RAGResponse("", [], "general"), RAGResponse("answer", [{}], "general"),
    RAGResponse("answer", [{"source": "x", "score": float("inf")}], "general")])
async def test_invalid_sync_output_returns_safe_500(result):
    chat.rag_chain = SimpleNamespace(answer=lambda *args: result)
    db = FakeDB()
    response = await request("POST", "/api/chat", json={"message": "question"}, db=db,
                             headers={"Accept": "application/json"})
    assert response.status_code == 500
    assert response.json()["detail"]["code"] == "INTERNAL_ERROR"
    assert not db.commit_called


@pytest.mark.asyncio
@pytest.mark.parametrize("accept", ["application/json", "text/event-stream"])
@pytest.mark.parametrize("bad_interface", ["missing", "not_callable", "wrong_arguments"])
async def test_wrong_dependency_interface_is_reported_not_silently_fallback(accept, bad_interface):
    method = "answer" if accept == "application/json" else "answer_stream"
    chain = SimpleNamespace()
    if bad_interface == "not_callable":
        setattr(chain, method, 42)
    elif bad_interface == "wrong_arguments":
        setattr(chain, method, lambda: None)
    chat.rag_chain = chain
    response = await request("POST", "/api/chat", json={"message": "question"}, headers={"Accept": accept})
    assert response.status_code == 500
    assert response.json()["detail"]["code"] == "INTERNAL_ERROR"


@pytest.mark.asyncio
@pytest.mark.parametrize("accept", ["application/json", "text/event-stream"])
async def test_database_rollback_failure_does_not_replace_answer(accept):
    db = FakeDB(commit_error=RuntimeError("commit failed"))
    db.rollback = AsyncMock(side_effect=RuntimeError("rollback failed"))
    chat.rag_chain = FakeRAG()
    response = await request("POST", "/api/chat", json={"message": "question"}, db=db, headers={"Accept": accept})
    assert response_payload(response, accept)["answer"] == "Câu trả lời [1]"
    db.rollback.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("accept", ["application/json", "text/event-stream"])
@pytest.mark.parametrize("enabled", [False, True])
async def test_api_to_real_rag_reranker_assembler_preserves_evidence(accept, enabled, mock_reranker_model, monkeypatch):
    from src.ingestion.chunker import Chunk
    from src.retrieval.reranker import Reranker
    from copy import deepcopy

    monkeypatch.setattr(chat.settings, "RERANK_TOP_K", 2)
    chunks = [Chunk(key, f"admission course {key}", {
        "doc_id": "same-doc", "title": "Admission", "page_number": page,
        "heading": "Courses", "chunk_type": "child", "parent_chunk_id": "parent",
    }) for page, key in enumerate(["a", "b", "c"], 1)]
    candidates = list(zip(chunks, [0.9, 0.8, 0.8]))
    before = deepcopy(candidates)
    retriever = MagicMock()
    retriever.search.return_value = candidates
    llm = MagicMock()
    answer = "admission course [1] [2]" if enabled else "admission course [1] [2] [3]"
    llm.generate.return_value = answer
    llm.generate_stream.return_value = iter([answer[:7], answer[7:]])
    mock_reranker_model.return_value.predict.side_effect = None
    mock_reranker_model.return_value.predict.return_value = [-5.0, 5.0, 5.0]
    chat.rag_chain = RAGChain(retriever, llm, top_k=3, reranker=Reranker() if enabled else None)
    response = await request("POST", "/api/chat", json={"message": "admission course"}, headers={"Accept": accept})
    payload = response_payload(response, accept)
    citations = ChatResponse.model_validate(payload).citations
    assert [c.chunk_id for c in citations] == (["b", "c"] if enabled else ["a", "b", "c"])
    assert [c.page for c in citations] == ([2, 3] if enabled else [1, 2, 3])
    assert [c.score for c in citations] == ([0.8, 0.8] if enabled else [0.9, 0.8, 0.8])
    assert all(c.doc_id == "same-doc" for c in citations)
    assert [c.marker for c in citations] == list(range(1, len(citations) + 1))
    retriever.search.assert_called_once_with("admission course", top_k=3)
    assert candidates == before
    if enabled:
        mock_reranker_model.return_value.predict.assert_called_once()
    else:
        mock_reranker_model.assert_not_called()
