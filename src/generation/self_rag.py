"""Deterministic Self-RAG checks for retrieved context and generated answers."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .prompts import FAITHFULNESS_CHECK_PROMPT, RELEVANCE_CHECK_PROMPT


logger = logging.getLogger(__name__)


class RAGCheckStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    UNKNOWN = "unknown"


class RAGAction(str, Enum):
    ACCEPT = "accept"
    RETRY = "retry"
    FALLBACK = "fallback"


@dataclass(frozen=True)
class RAGCheckResult:
    status: RAGCheckStatus
    action: RAGAction
    reason: str = ""
    error: str | None = None


class SelfRAG:
    """Evaluate retrieval relevance and answer faithfulness without retrying."""

    _FENCED_JSON_PATTERN = re.compile(
        r"^```(?:json)?\s*(.*?)\s*```$",
        flags=re.IGNORECASE | re.DOTALL,
    )

    def __init__(self, llm_client: Any):
        self.llm_client = llm_client

    def check_relevance(self, query: str, context: str) -> RAGCheckResult:
        self._validate_query(query)
        if not self._has_text(context):
            return RAGCheckResult(
                status=RAGCheckStatus.FAIL,
                action=RAGAction.RETRY,
                reason="Context is empty.",
            )

        user_input = (
            f"<query>\n{query}\n</query>\n\n"
            f"<context>\n{context}\n</context>"
        )
        return self._run_check(RELEVANCE_CHECK_PROMPT, user_input)

    def check_faithfulness(
        self,
        query: str,
        answer: str,
        context: str,
    ) -> RAGCheckResult:
        self._validate_query(query)
        if not self._has_text(answer):
            return RAGCheckResult(
                status=RAGCheckStatus.FAIL,
                action=RAGAction.RETRY,
                reason="Answer is empty.",
            )
        if not self._has_text(context):
            return RAGCheckResult(
                status=RAGCheckStatus.FAIL,
                action=RAGAction.RETRY,
                reason="Context is empty.",
            )

        user_input = (
            f"<query>\n{query}\n</query>\n\n"
            f"<context>\n{context}\n</context>\n\n"
            f"<answer>\n{answer}\n</answer>"
        )
        return self._run_check(FAITHFULNESS_CHECK_PROMPT, user_input)

    def _run_check(self, system_prompt: str, user_input: str) -> RAGCheckResult:
        try:
            output = self.llm_client.generate(
                system_prompt=system_prompt,
                messages=[{"role": "user", "content": user_input}],
                temperature=0.0,
            )
        except Exception as exc:
            logger.exception("Self-RAG checker LLM call failed")
            return self._unknown_result(f"LLM checker error: {exc}")

        try:
            return self._parse_result(output)
        except Exception as exc:
            logger.warning("Self-RAG checker returned malformed output: %s", exc)
            return self._unknown_result(f"Malformed checker output: {exc}")

    @classmethod
    def _parse_result(cls, output: Any) -> RAGCheckResult:
        if not isinstance(output, str):
            raise TypeError("checker output must be a string")

        payload_text = output.strip()
        fenced_match = cls._FENCED_JSON_PATTERN.fullmatch(payload_text)
        if fenced_match:
            payload_text = fenced_match.group(1).strip()

        payload = json.loads(payload_text)
        if not isinstance(payload, dict):
            raise ValueError("checker output must be a JSON object")

        verdict = payload.get("verdict")
        if verdict not in (RAGCheckStatus.PASS.value, RAGCheckStatus.FAIL.value):
            raise ValueError("verdict must be exactly 'pass' or 'fail'")

        raw_reason = payload.get("reason", "")
        reason = "" if raw_reason is None else str(raw_reason).strip()
        if verdict == RAGCheckStatus.PASS.value:
            return RAGCheckResult(
                status=RAGCheckStatus.PASS,
                action=RAGAction.ACCEPT,
                reason=reason,
            )
        return RAGCheckResult(
            status=RAGCheckStatus.FAIL,
            action=RAGAction.RETRY,
            reason=reason,
        )

    @staticmethod
    def _unknown_result(error: str) -> RAGCheckResult:
        return RAGCheckResult(
            status=RAGCheckStatus.UNKNOWN,
            action=RAGAction.FALLBACK,
            error=error,
        )

    @staticmethod
    def _has_text(value: Any) -> bool:
        return isinstance(value, str) and bool(value.strip())

    @classmethod
    def _validate_query(cls, query: str) -> None:
        if not cls._has_text(query):
            raise ValueError("query must not be blank")
