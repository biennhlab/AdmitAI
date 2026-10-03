import logging
import inspect
from typing import Any, Dict, List, Iterator

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    OpenAI,
    RateLimitError,
)

logger = logging.getLogger(__name__)

_MAX_OUTPUT_TOKENS = 1024


class LLMProviderError(RuntimeError):
    """Base error for failures returned by or while reaching the LLM provider."""

    code = "LLM_PROVIDER_ERROR"


class LLMAuthenticationError(LLMProviderError):
    code = "LLM_AUTHENTICATION_FAILED"


class LLMTimeoutError(LLMProviderError):
    code = "LLM_TIMEOUT"


class LLMRateLimitError(LLMProviderError):
    code = "LLM_RATE_LIMITED"


class LLMConnectionError(LLMProviderError):
    code = "LLM_UNAVAILABLE"


class LLMUpstreamError(LLMProviderError):
    code = "LLM_PROVIDER_ERROR"


class LLMClient:
    """Wrapper for Gemini (OpenAI-compatible) API"""

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str,
        timeout_seconds: float = 30.0,
        max_retries: int = 1,
        reasoning_effort: str | None = None,
    ):
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout_seconds,
            max_retries=max_retries,
        )

    @staticmethod
    def _supports_keyword(callable_obj: Any, keyword: str) -> bool:
        """Return whether an SDK callable accepts a keyword directly."""
        try:
            parameters = inspect.signature(callable_obj).parameters.values()
        except (TypeError, ValueError):
            return False
        return any(
            parameter.name == keyword
            or parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters
        )

    def _reasoning_request_options(self) -> dict[str, Any]:
        """Build reasoning options for both old and new OpenAI SDK releases."""
        if not self.reasoning_effort:
            return {}
        create = self.client.chat.completions.create
        if self._supports_keyword(create, "reasoning_effort"):
            return {"reasoning_effort": self.reasoning_effort}
        # openai 1.14 does not type this field yet, but extra_body forwards
        # arbitrary OpenAI-compatible request fields to the provider.
        return {"extra_body": {"reasoning_effort": self.reasoning_effort}}

    def generate(self, system_prompt: str, messages: List[Dict[str, str]], temperature: float = 0.3) -> str:
        """Generate a response using the provided system prompt and message history."""
        try:
            formatted_messages = [{"role": "system", "content": system_prompt}]
            formatted_messages.extend(messages)

            response = self.client.chat.completions.create(
                model=self.model,
                messages=formatted_messages,
                temperature=temperature,
                max_tokens=_MAX_OUTPUT_TOKENS,
                **self._reasoning_request_options(),
            )
            return response.choices[0].message.content
        except AuthenticationError as exc:
            logger.exception("LLM provider authentication failed")
            raise LLMAuthenticationError("LLM provider authentication failed") from exc
        except APITimeoutError as exc:
            logger.exception("LLM provider request timed out")
            raise LLMTimeoutError("LLM provider request timed out") from exc
        except RateLimitError as exc:
            logger.exception("LLM provider rate limit reached")
            raise LLMRateLimitError("LLM provider rate limit reached") from exc
        except APIConnectionError as exc:
            logger.exception("Could not connect to the LLM provider")
            raise LLMConnectionError("Could not connect to the LLM provider") from exc
        except APIStatusError as exc:
            logger.exception("LLM provider returned HTTP %s", exc.status_code)
            raise LLMUpstreamError("LLM provider returned an error") from exc
        except Exception as exc:
            logger.exception("Unexpected LLM provider failure")
            raise LLMUpstreamError("LLM provider returned an unexpected error") from exc

    def generate_stream(self, system_prompt: str, messages: List[Dict[str, str]], temperature: float = 0.3) -> Iterator[str]:
        """Generate a streamed response using the provided system prompt and message history."""
        try:
            formatted_messages = [{"role": "system", "content": system_prompt}]
            formatted_messages.extend(messages)

            response = self.client.chat.completions.create(
                model=self.model,
                messages=formatted_messages,
                temperature=temperature,
                max_tokens=_MAX_OUTPUT_TOKENS,
                stream=True,
                **self._reasoning_request_options(),
            )
            for chunk in response:
                content = chunk.choices[0].delta.content
                if content is not None:
                    yield content
        except AuthenticationError as exc:
            logger.exception("LLM provider authentication failed")
            raise LLMAuthenticationError("LLM provider authentication failed") from exc
        except APITimeoutError as exc:
            logger.exception("LLM provider request timed out")
            raise LLMTimeoutError("LLM provider request timed out") from exc
        except RateLimitError as exc:
            logger.exception("LLM provider rate limit reached")
            raise LLMRateLimitError("LLM provider rate limit reached") from exc
        except APIConnectionError as exc:
            logger.exception("Could not connect to the LLM provider")
            raise LLMConnectionError("Could not connect to the LLM provider") from exc
        except APIStatusError as exc:
            logger.exception("LLM provider returned HTTP %s", exc.status_code)
            raise LLMUpstreamError("LLM provider returned an error") from exc
        except Exception as exc:
            logger.exception("Unexpected LLM provider failure")
            raise LLMUpstreamError("LLM provider returned an unexpected error") from exc

    def generate_with_tools(self, system_prompt: str, messages: List[Dict[str, str]], tools: List[Dict[str, Any]]) -> Any:
        """Not implemented for Phase 1. Will be used in Agentic Router later."""
        raise NotImplementedError("Tool calling not yet implemented for Phase 1")
