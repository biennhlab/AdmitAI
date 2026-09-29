from __future__ import annotations

import logging
import math
import time
from typing import Any, Protocol, TypeAlias
from urllib.parse import urlsplit
from uuid import uuid4

import httpx

from src.config import settings
from src.ingestion.chunker import Chunk


logger = logging.getLogger(__name__)

RerankedResult: TypeAlias = tuple[Chunk, float, float]

MAX_REMOTE_QUERY_LENGTH = 4_000
MAX_REMOTE_DOCUMENT_ID_LENGTH = 256
MAX_REMOTE_DOCUMENT_TEXT_LENGTH = 16_000
ABSOLUTE_REMOTE_CANDIDATE_LIMIT = 20
RETRYABLE_REMOTE_STATUSES = frozenset({429, 503, 504})


class RerankerError(RuntimeError):
    """Base class for reranker failures that may use retrieval-order fallback."""


class RerankerConfigurationError(RerankerError):
    """The selected reranker backend is not configured correctly."""


class RerankerInputError(RerankerError):
    """The query or candidates violate the remote endpoint contract."""


class RerankerTimeoutError(RerankerError):
    """The remote reranker did not respond before its timeout."""


class RerankerHTTPError(RerankerError):
    """The remote reranker returned an error or could not be reached."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class RerankerResponseError(RerankerError):
    """The remote reranker returned a malformed successful response."""


class _RerankerBackend(Protocol):
    def rerank(
        self,
        query: str,
        candidates: list[tuple[Chunk, float]],
        top_k: int | None = None,
    ) -> list[RerankedResult]: ...

    def close(self) -> None: ...


def CrossEncoder(model_name: str) -> Any:
    """Load sentence-transformers only when the local backend is selected."""
    from sentence_transformers import CrossEncoder as SentenceTransformersCrossEncoder

    return SentenceTransformersCrossEncoder(model_name)


def _validate_candidate_pair(
    candidate: tuple[Chunk, float],
    index: int,
) -> tuple[Chunk, float, float]:
    try:
        chunk, retrieval_score = candidate
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "Each candidate must be a (Chunk, retrieval_score) pair"
        ) from exc
    if not isinstance(chunk, Chunk):
        raise TypeError(f"candidates[{index}] must contain a Chunk")
    try:
        retrieval_sort_score = float(retrieval_score)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Retrieval score at index {index} cannot be converted to float"
        ) from exc
    if not math.isfinite(retrieval_sort_score):
        raise ValueError(f"Retrieval score at index {index} must be finite")
    return chunk, retrieval_score, retrieval_sort_score


class LocalRerankerBackend:
    """Existing sentence-transformers CrossEncoder implementation."""

    def __init__(
        self,
        model_name: str,
        batch_size: int,
        *,
        model: Any | None = None,
    ) -> None:
        if model is not None and not callable(getattr(model, "predict", None)):
            raise TypeError("model must provide a predict method")
        self.model_name = model_name
        self.batch_size = batch_size
        # Loading happens once per local backend instance and never in rerank().
        self.model = model if model is not None else CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        candidates: list[tuple[Chunk, float]],
        top_k: int | None = None,
    ) -> list[RerankedResult]:
        pairs: list[tuple[str, str]] = []
        validated: list[tuple[Chunk, float, float]] = []
        for index, candidate in enumerate(candidates):
            chunk, retrieval_score, retrieval_sort_score = _validate_candidate_pair(
                candidate, index
            )
            validated.append((chunk, retrieval_score, retrieval_sort_score))
            pairs.append((query, chunk.content))

        raw_scores = self.model.predict(pairs, batch_size=self.batch_size)
        try:
            scores = list(raw_scores)
        except TypeError as exc:
            raise ValueError("Reranker model must return one score per candidate") from exc
        if len(scores) != len(candidates):
            raise ValueError(
                "Reranker score count mismatch: "
                f"expected {len(candidates)}, got {len(scores)}"
            )

        results: list[RerankedResult] = []
        for index, ((chunk, retrieval_score, _retrieval_sort_score), raw_score) in enumerate(
            zip(validated, scores)
        ):
            try:
                reranker_score = float(raw_score)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Reranker score at index {index} cannot be converted to float"
                ) from exc
            if not math.isfinite(reranker_score):
                raise ValueError(f"Reranker score at index {index} must be finite")
            results.append((chunk, retrieval_score, reranker_score))

        results.sort(key=lambda item: (-item[2], -float(item[1]), item[0].chunk_id))
        return results if top_k is None else results[:top_k]

    def close(self) -> None:
        return None


class RemoteRerankerBackend:
    """Adapter for the custom Hugging Face reranking endpoint contract."""

    def __init__(
        self,
        *,
        model_name: str,
        provider: str,
        url: str,
        api_token: str,
        timeout_seconds: float,
        max_retries: int,
        top_k_limit: int,
        http_client: httpx.Client | None = None,
    ) -> None:
        if provider != "huggingface_endpoint":
            raise RerankerConfigurationError(
                "remote_provider must be 'huggingface_endpoint'"
            )
        if not isinstance(url, str) or not url.strip():
            raise RerankerConfigurationError("remote_url is required for remote backend")
        parsed_url = urlsplit(url.strip())
        if (
            parsed_url.scheme not in {"http", "https"}
            or not parsed_url.hostname
            or parsed_url.username is not None
            or parsed_url.password is not None
        ):
            raise RerankerConfigurationError(
                "remote_url must be an HTTP(S) URL without embedded credentials"
            )
        try:
            endpoint_port = parsed_url.port
        except ValueError as exc:
            raise RerankerConfigurationError(
                "remote_url contains an invalid port"
            ) from exc
        if not isinstance(api_token, str) or not api_token.strip():
            raise RerankerConfigurationError(
                "remote_api_token is required for remote backend"
            )
        if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)):
            raise RerankerConfigurationError("remote_timeout_seconds must be a number")
        if not math.isfinite(float(timeout_seconds)) or timeout_seconds <= 0:
            raise RerankerConfigurationError(
                "remote_timeout_seconds must be greater than 0"
            )
        if isinstance(max_retries, bool) or not isinstance(max_retries, int):
            raise RerankerConfigurationError("remote_max_retries must be an integer")
        if not 0 <= max_retries <= 5:
            raise RerankerConfigurationError(
                "remote_max_retries must be between 0 and 5"
            )
        if isinstance(top_k_limit, bool) or not isinstance(top_k_limit, int):
            raise RerankerConfigurationError("remote_top_k_limit must be an integer")
        if not 1 <= top_k_limit <= ABSOLUTE_REMOTE_CANDIDATE_LIMIT:
            raise RerankerConfigurationError(
                f"remote_top_k_limit must be between 1 and {ABSOLUTE_REMOTE_CANDIDATE_LIMIT}"
            )

        self.model_name = model_name
        self.provider = provider
        self.url = url.strip()
        self.timeout_seconds = float(timeout_seconds)
        self.max_retries = max_retries
        self.top_k_limit = top_k_limit
        self._api_token = api_token.strip()
        self._endpoint_label = parsed_url.hostname or "remote-reranker"
        if endpoint_port is not None:
            self._endpoint_label = f"{self._endpoint_label}:{endpoint_port}"
        self._owns_client = http_client is None
        self._client = http_client or httpx.Client(timeout=self.timeout_seconds)
        self.last_usage_latency_ms: float | None = None

    def _input_error(self, message: str) -> RerankerInputError:
        return RerankerInputError(message)

    def _build_documents(
        self,
        query: str,
        candidates: list[tuple[Chunk, float]],
    ) -> tuple[list[dict[str, str]], list[tuple[Chunk, float, float]]]:
        if len(query) > MAX_REMOTE_QUERY_LENGTH:
            raise self._input_error(
                f"Query exceeds maximum length of {MAX_REMOTE_QUERY_LENGTH} characters"
            )
        if len(candidates) > self.top_k_limit:
            raise self._input_error(
                f"Candidate count exceeds configured limit of {self.top_k_limit}"
            )

        documents: list[dict[str, str]] = []
        validated: list[tuple[Chunk, float, float]] = []
        for index, candidate in enumerate(candidates):
            try:
                chunk, retrieval_score, retrieval_sort_score = _validate_candidate_pair(
                    candidate, index
                )
            except (TypeError, ValueError) as exc:
                raise self._input_error(str(exc)) from exc

            if not isinstance(chunk.chunk_id, str) or not chunk.chunk_id.strip():
                raise self._input_error(f"Document id at index {index} cannot be empty")
            if len(chunk.chunk_id) > MAX_REMOTE_DOCUMENT_ID_LENGTH:
                raise self._input_error(
                    f"Document id at index {index} exceeds maximum length of "
                    f"{MAX_REMOTE_DOCUMENT_ID_LENGTH} characters"
                )
            if not isinstance(chunk.content, str) or not chunk.content.strip():
                raise self._input_error(f"Document text at index {index} cannot be empty")
            if len(chunk.content) > MAX_REMOTE_DOCUMENT_TEXT_LENGTH:
                raise self._input_error(
                    f"Document text at index {index} exceeds maximum length of "
                    f"{MAX_REMOTE_DOCUMENT_TEXT_LENGTH} characters"
                )
            documents.append({"id": chunk.chunk_id, "text": chunk.content})
            validated.append((chunk, retrieval_score, retrieval_sort_score))
        return documents, validated

    def _log_failure(
        self,
        *,
        request_id: str,
        status: int | str,
        attempt: int,
        retryable: bool,
        candidate_count: int,
    ) -> None:
        logger.warning(
            "Remote reranker failure request_id=%s status=%s attempt=%s "
            "retryable=%s candidate_count=%s endpoint=%s",
            request_id,
            status,
            attempt,
            "yes" if retryable else "no",
            candidate_count,
            self._endpoint_label,
        )

    def _post(
        self,
        *,
        request_id: str,
        payload: dict[str, Any],
        candidate_count: int,
    ) -> tuple[httpx.Response, int]:
        headers = {
            "Authorization": f"Bearer {self._api_token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-Request-ID": request_id,
        }
        total_attempts = self.max_retries + 1
        for attempt_index in range(total_attempts):
            attempt = attempt_index + 1
            try:
                response = self._client.post(self.url, headers=headers, json=payload)
            except httpx.TimeoutException as exc:
                self._log_failure(
                    request_id=request_id,
                    status="timeout",
                    attempt=attempt,
                    retryable=True,
                    candidate_count=candidate_count,
                )
                if attempt_index < self.max_retries:
                    time.sleep(0.5 * (2**attempt_index))
                    continue
                raise RerankerTimeoutError(
                    f"Remote reranker request {request_id} timed out after {attempt} attempt(s)"
                ) from exc
            except httpx.RequestError as exc:
                self._log_failure(
                    request_id=request_id,
                    status="transport_error",
                    attempt=attempt,
                    retryable=False,
                    candidate_count=candidate_count,
                )
                raise RerankerHTTPError(
                    f"Remote reranker request {request_id} could not be completed"
                ) from exc

            if 200 <= response.status_code < 300:
                return response, attempt

            status_retryable = response.status_code in RETRYABLE_REMOTE_STATUSES
            will_retry = status_retryable and attempt_index < self.max_retries
            self._log_failure(
                request_id=request_id,
                status=response.status_code,
                attempt=attempt,
                retryable=status_retryable,
                candidate_count=candidate_count,
            )
            if will_retry:
                time.sleep(0.5 * (2**attempt_index))
                continue
            raise RerankerHTTPError(
                f"Remote reranker request {request_id} failed with HTTP "
                f"{response.status_code} after {attempt} attempt(s)",
                status_code=response.status_code,
            )

        raise AssertionError("Remote reranker retry loop exited unexpectedly")

    def _parse_response(
        self,
        response: httpx.Response,
        validated: list[tuple[Chunk, float, float]],
        *,
        top_n: int,
        request_id: str,
    ) -> list[RerankedResult]:
        try:
            body = response.json()
        except (ValueError, TypeError) as exc:
            raise RerankerResponseError(
                f"Remote reranker request {request_id} returned invalid JSON"
            ) from exc
        if not isinstance(body, dict):
            raise RerankerResponseError(
                f"Remote reranker request {request_id} returned a non-object response"
            )
        if not isinstance(body.get("request_id"), str) or not body["request_id"].strip():
            raise RerankerResponseError("Remote reranker response has invalid request_id")
        if body["request_id"] != request_id:
            raise RerankerResponseError(
                "Remote reranker response request_id does not match the request"
            )
        if not isinstance(body.get("model"), str) or not body["model"].strip():
            raise RerankerResponseError("Remote reranker response has invalid model")
        if body["model"] != self.model_name:
            raise RerankerResponseError(
                "Remote reranker response model does not match the configured model"
            )
        results = body.get("results")
        if not isinstance(results, list):
            raise RerankerResponseError("Remote reranker response is missing results")
        if len(results) > top_n:
            raise RerankerResponseError(
                "Remote reranker returned more results than requested"
            )

        usage = body.get("usage")
        self.last_usage_latency_ms = None
        if isinstance(usage, dict) and "latency_ms" in usage:
            try:
                latency_ms = float(usage["latency_ms"])
            except (TypeError, ValueError) as exc:
                raise RerankerResponseError(
                    "Remote reranker response has invalid usage.latency_ms"
                ) from exc
            if not math.isfinite(latency_ms) or latency_ms < 0:
                raise RerankerResponseError(
                    "Remote reranker response has invalid usage.latency_ms"
                )
            self.last_usage_latency_ms = latency_ms

        seen_indexes: set[int] = set()
        mapped: list[RerankedResult] = []
        for result_position, result in enumerate(results):
            if not isinstance(result, dict):
                raise RerankerResponseError(
                    f"Remote reranker result {result_position} must be an object"
                )
            index = result.get("index")
            if isinstance(index, bool) or not isinstance(index, int):
                raise RerankerResponseError(
                    f"Remote reranker result {result_position} has invalid index"
                )
            if not 0 <= index < len(validated):
                raise RerankerResponseError(
                    f"Remote reranker result {result_position} index is out of range"
                )
            if index in seen_indexes:
                raise RerankerResponseError(
                    f"Remote reranker response contains duplicate index {index}"
                )
            seen_indexes.add(index)

            chunk, retrieval_score, _retrieval_sort_score = validated[index]
            if result.get("document_id") != chunk.chunk_id:
                raise RerankerResponseError(
                    f"Remote reranker result {result_position} document_id does not match index"
                )
            raw_score = result.get("relevance_score")
            if isinstance(raw_score, bool):
                raise RerankerResponseError(
                    f"Remote reranker result {result_position} has invalid relevance_score"
                )
            try:
                reranker_score = float(raw_score)
            except (TypeError, ValueError) as exc:
                raise RerankerResponseError(
                    f"Remote reranker result {result_position} has invalid relevance_score"
                ) from exc
            if not math.isfinite(reranker_score):
                raise RerankerResponseError(
                    f"Remote reranker result {result_position} relevance_score must be finite"
                )
            mapped.append((chunk, retrieval_score, reranker_score))

        mapped.sort(key=lambda item: (-item[2], -float(item[1]), item[0].chunk_id))
        return mapped

    def rerank(
        self,
        query: str,
        candidates: list[tuple[Chunk, float]],
        top_k: int | None = None,
    ) -> list[RerankedResult]:
        documents, validated = self._build_documents(query, candidates)
        top_n = len(documents) if top_k is None else min(top_k, len(documents))

        request_id = str(uuid4())
        payload = {"query": query, "documents": documents, "top_n": top_n}
        response, attempt = self._post(
            request_id=request_id,
            payload=payload,
            candidate_count=len(candidates),
        )
        try:
            return self._parse_response(
                response,
                validated,
                top_n=top_n,
                request_id=request_id,
            )
        except RerankerResponseError:
            self._log_failure(
                request_id=request_id,
                status="malformed_response",
                attempt=attempt,
                retryable=False,
                candidate_count=len(candidates),
            )
            raise

    def close(self) -> None:
        if self._owns_client:
            self._client.close()


class Reranker:
    """Backend-selectable reranker with a stable public interface."""

    def __init__(
        self,
        model_name: str | None = None,
        batch_size: int | None = None,
        *,
        model: Any | None = None,
        backend: str | None = None,
        remote_provider: str | None = None,
        remote_url: str | None = None,
        remote_api_token: str | None = None,
        remote_timeout_seconds: float | None = None,
        remote_max_retries: int | None = None,
        remote_top_k_limit: int | None = None,
        http_client: httpx.Client | None = None,
    ) -> None:
        resolved_model_name = model_name or settings.RERANKER_MODEL
        if not isinstance(resolved_model_name, str) or not resolved_model_name.strip():
            raise ValueError("Reranker model name cannot be empty")
        self.model_name = resolved_model_name.strip()

        resolved_batch_size = settings.RERANK_BATCH_SIZE if batch_size is None else batch_size
        if not isinstance(resolved_batch_size, int) or isinstance(resolved_batch_size, bool):
            raise TypeError("batch_size must be an integer")
        if resolved_batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")
        self.batch_size = resolved_batch_size

        resolved_backend = (backend or settings.RERANKER_BACKEND).strip().lower()
        if resolved_backend not in {"local", "remote"}:
            raise RerankerConfigurationError("backend must be 'local' or 'remote'")
        self.backend_name = resolved_backend

        if resolved_backend == "local":
            if http_client is not None:
                raise RerankerConfigurationError(
                    "http_client can only be used with the remote backend"
                )
            selected_backend: _RerankerBackend = LocalRerankerBackend(
                self.model_name,
                self.batch_size,
                model=model,
            )
            self.model = selected_backend.model
        else:
            if model is not None:
                raise RerankerConfigurationError(
                    "model injection can only be used with the local backend"
                )
            configured_token = settings.RERANKER_REMOTE_API_TOKEN.get_secret_value()
            selected_backend = RemoteRerankerBackend(
                model_name=self.model_name,
                provider=remote_provider or settings.RERANKER_REMOTE_PROVIDER,
                url=remote_url if remote_url is not None else settings.RERANKER_REMOTE_URL,
                api_token=(
                    remote_api_token
                    if remote_api_token is not None
                    else configured_token
                ),
                timeout_seconds=(
                    remote_timeout_seconds
                    if remote_timeout_seconds is not None
                    else settings.RERANKER_REMOTE_TIMEOUT_SECONDS
                ),
                max_retries=(
                    remote_max_retries
                    if remote_max_retries is not None
                    else settings.RERANKER_REMOTE_MAX_RETRIES
                ),
                top_k_limit=(
                    remote_top_k_limit
                    if remote_top_k_limit is not None
                    else settings.RERANKER_REMOTE_TOP_K_LIMIT
                ),
                http_client=http_client,
            )
            self.model = None
        self.backend = selected_backend

    @property
    def last_usage_latency_ms(self) -> float | None:
        return getattr(self.backend, "last_usage_latency_ms", None)

    def rerank(
        self,
        query: str,
        candidates: list[tuple[Chunk, float]],
        top_k: int | None = None,
    ) -> list[RerankedResult]:
        if top_k is not None:
            if not isinstance(top_k, int) or isinstance(top_k, bool):
                raise TypeError("top_k must be an integer or None")
            if top_k <= 0:
                return []
        if not candidates:
            return []
        if not isinstance(query, str) or not query.strip():
            if self.backend_name == "remote":
                raise RerankerInputError("Query cannot be empty")
            raise ValueError("Query cannot be empty")
        return self.backend.rerank(query, candidates, top_k=top_k)

    def close(self) -> None:
        self.backend.close()
