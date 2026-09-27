from __future__ import annotations

import asyncio
import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from src.config import settings
from src.retrieval import Embedder


RAGAS_METRIC_NAMES = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
)


def _normalized(value: Any) -> str:
    return " ".join(str(value).split()).casefold()


def _safe_error(exc: BaseException) -> str:
    message = f"{type(exc).__name__}: {exc}"
    secret = settings.LLM_API_KEY
    if secret and secret.lower() != "placeholder":
        message = message.replace(secret, "***")
    return message


@dataclass(frozen=True)
class ExpectedSource:
    doc_id: str | None = None
    title: str | None = None
    page: str | int | None = None

    @property
    def is_matchable(self) -> bool:
        return bool(self.doc_id or self.title)


@dataclass(frozen=True)
class GoldenCase:
    id: str
    query_type: str
    question: str
    expected_answer: str | None = None
    expected_facts: list[str] = field(default_factory=list)
    expected_sources: list[ExpectedSource] = field(default_factory=list)
    expect_fallback: bool = False

    @property
    def reference(self) -> str | None:
        if self.expected_answer and self.expected_answer.strip():
            return self.expected_answer.strip()
        if self.expected_facts:
            return "; ".join(self.expected_facts) + "."
        return None


@dataclass
class CaseResult:
    id: str
    query_type: str
    question: str
    status: str
    error: str | None
    answer: str | None
    route_type: str | None
    contexts: list[str]
    citations: list[dict[str, Any]]
    reference: str | None
    metrics: dict[str, float | None]
    metric_errors: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvaluationReport:
    summary: dict[str, int | float | None]
    cases: list[CaseResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "cases": [case.to_dict() for case in self.cases],
        }


class RagasRunner:
    """Evaluate the real production RAG chain with RAGAS and project metrics."""

    def __init__(
        self,
        rag_chain: Any,
        *,
        metrics: Mapping[str, Any] | None = None,
        evaluator_llm: Any | None = None,
        evaluator_embeddings: Any | None = None,
    ) -> None:
        if rag_chain is None or not callable(getattr(rag_chain, "evaluate_trace", None)):
            raise TypeError("rag_chain must provide evaluate_trace(question)")
        self.rag_chain = rag_chain
        if metrics is None:
            self.metrics = self._build_ragas_metrics(
                evaluator_llm=evaluator_llm,
                evaluator_embeddings=evaluator_embeddings,
            )
        else:
            self.metrics = dict(metrics)

    @staticmethod
    def _parse_source(raw: Any, case_index: int, source_index: int) -> ExpectedSource:
        if not isinstance(raw, dict):
            raise ValueError(
                f"cases[{case_index}].expected_sources[{source_index}] must be an object"
            )
        doc_id = raw.get("doc_id")
        title = raw.get("title")
        page = raw.get("page")
        for name, value in (("doc_id", doc_id), ("title", title)):
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(
                    f"cases[{case_index}].expected_sources[{source_index}].{name} "
                    "must be a non-empty string or null"
                )
        if page is not None and (
            isinstance(page, bool) or not isinstance(page, (str, int))
        ):
            raise ValueError(
                f"cases[{case_index}].expected_sources[{source_index}].page "
                "must be a string, integer, or null"
            )
        return ExpectedSource(doc_id=doc_id, title=title, page=page)

    @classmethod
    def _parse_case(cls, raw: Any, index: int) -> GoldenCase:
        if not isinstance(raw, dict):
            raise ValueError(f"cases[{index}] must be an object")
        required = ("id", "query_type", "question")
        for name in required:
            value = raw.get(name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"cases[{index}].{name} must be a non-empty string")

        expected_answer = raw.get("expected_answer")
        if expected_answer is not None and not isinstance(expected_answer, str):
            raise ValueError(f"cases[{index}].expected_answer must be a string or null")

        expected_facts = raw.get("expected_facts", [])
        if not isinstance(expected_facts, list) or any(
            not isinstance(fact, str) or not fact.strip() for fact in expected_facts
        ):
            raise ValueError(
                f"cases[{index}].expected_facts must be a list of non-empty strings"
            )

        expected_sources = raw.get("expected_sources", [])
        if not isinstance(expected_sources, list):
            raise ValueError(f"cases[{index}].expected_sources must be a list")

        expect_fallback = raw.get("expect_fallback", False)
        if not isinstance(expect_fallback, bool):
            raise ValueError(f"cases[{index}].expect_fallback must be a boolean")

        return GoldenCase(
            id=raw["id"].strip(),
            query_type=raw["query_type"].strip(),
            question=raw["question"].strip(),
            expected_answer=expected_answer.strip() if expected_answer else None,
            expected_facts=[fact.strip() for fact in expected_facts],
            expected_sources=[
                cls._parse_source(source, index, source_index)
                for source_index, source in enumerate(expected_sources)
            ],
            expect_fallback=expect_fallback,
        )

    @classmethod
    def load_dataset(cls, path: str | Path) -> list[GoldenCase]:
        dataset_path = Path(path)
        try:
            raw = json.loads(dataset_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ValueError(f"Dataset does not exist: {dataset_path}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"Dataset is not valid JSON: {exc}") from exc
        if not isinstance(raw, list):
            raise ValueError("Dataset root must be a JSON array")
        if not raw:
            raise ValueError("Dataset must contain at least one case")

        cases = [cls._parse_case(item, index) for index, item in enumerate(raw)]
        ids = [case.id for case in cases]
        duplicates = sorted({case_id for case_id in ids if ids.count(case_id) > 1})
        if duplicates:
            raise ValueError(f"Dataset contains duplicate ids: {', '.join(duplicates)}")
        return cases

    def _project_embedder(self) -> Embedder:
        retriever = getattr(self.rag_chain, "retriever", None)
        dense_search = getattr(retriever, "dense_search", None)
        embedder = getattr(dense_search, "embedder", None)
        if embedder is not None and callable(getattr(embedder, "embed_query", None)):
            return embedder
        direct = getattr(retriever, "embedder", None)
        if direct is not None and callable(getattr(direct, "embed_query", None)):
            return direct
        # Embedder has a process-wide model cache, so this remains a single model
        # instance even for retriever implementations that do not expose it.
        return Embedder(model_name=settings.EMBEDDING_MODEL)

    def _build_embedding_adapter(self) -> Any:
        from ragas.embeddings.base import BaseRagasEmbedding

        project_embedder = self._project_embedder()

        class ProjectEmbeddingAdapter(BaseRagasEmbedding):
            def __init__(self, embedder: Embedder) -> None:
                super().__init__()
                self.embedder = embedder

            def embed_text(self, text: str, **kwargs: Any) -> list[float]:
                del kwargs
                return self.embedder.embed_query(text).astype(float).tolist()

            async def aembed_text(self, text: str, **kwargs: Any) -> list[float]:
                return await asyncio.to_thread(self.embed_text, text, **kwargs)

            def embed_texts(
                self, texts: list[str], **kwargs: Any
            ) -> list[list[float]]:
                del kwargs
                return self.embedder.embed(texts).astype(float).tolist()

            async def aembed_texts(
                self, texts: list[str], **kwargs: Any
            ) -> list[list[float]]:
                return await asyncio.to_thread(self.embed_texts, texts, **kwargs)

        return ProjectEmbeddingAdapter(project_embedder)

    def _build_ragas_metrics(
        self,
        *,
        evaluator_llm: Any | None,
        evaluator_embeddings: Any | None,
    ) -> dict[str, Any]:
        os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")
        from openai import AsyncOpenAI
        from ragas.llms import llm_factory
        from ragas.metrics.collections import (
            AnswerRelevancy,
            ContextPrecision,
            ContextRecall,
            Faithfulness,
        )

        if evaluator_llm is None:
            client = AsyncOpenAI(
                api_key=settings.LLM_API_KEY,
                base_url=settings.LLM_BASE_URL,
                timeout=settings.LLM_TIMEOUT_SECONDS,
                max_retries=settings.LLM_MAX_RETRIES,
            )
            evaluator_llm = llm_factory(
                settings.LLM_MODEL,
                provider="openai",
                client=client,
            )
        if evaluator_embeddings is None:
            evaluator_embeddings = self._build_embedding_adapter()

        return {
            "faithfulness": Faithfulness(llm=evaluator_llm),
            "answer_relevancy": AnswerRelevancy(
                llm=evaluator_llm,
                embeddings=evaluator_embeddings,
            ),
            "context_precision": ContextPrecision(llm=evaluator_llm),
            "context_recall": ContextRecall(llm=evaluator_llm),
        }

    @staticmethod
    def _fact_recall(answer: str, expected_facts: list[str]) -> float | None:
        if not expected_facts:
            return None
        normalized_answer = _normalized(answer)
        found = sum(
            1 for fact in expected_facts if _normalized(fact) in normalized_answer
        )
        return found / len(expected_facts)

    @staticmethod
    def _source_matches(expected: ExpectedSource, citation: Mapping[str, Any]) -> bool:
        if expected.doc_id:
            if _normalized(citation.get("doc_id", "")) != _normalized(expected.doc_id):
                return False
        elif expected.title:
            actual_title = _normalized(
                citation.get("title") or citation.get("source") or ""
            )
            wanted_title = _normalized(expected.title)
            if wanted_title not in actual_title and actual_title not in wanted_title:
                return False
        else:
            return False

        if expected.page is not None:
            return _normalized(citation.get("page", "")) == _normalized(expected.page)
        return True

    @classmethod
    def _source_hit(
        cls,
        citations: list[dict[str, Any]],
        expected_sources: list[ExpectedSource],
    ) -> float | None:
        matchable = [source for source in expected_sources if source.is_matchable]
        if not matchable:
            return None
        return float(
            any(
                cls._source_matches(expected, citation)
                for expected in matchable
                for citation in citations
            )
        )

    @staticmethod
    def _score_value(raw: Any) -> float:
        value = getattr(raw, "value", raw)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"RAGAS returned a non-numeric score: {value!r}")
        score = float(value)
        if not math.isfinite(score):
            raise ValueError(f"RAGAS returned a non-finite score: {score}")
        return score

    def _ragas_scores(
        self,
        case: GoldenCase,
        *,
        answer: str,
        contexts: list[str],
    ) -> tuple[dict[str, float | None], dict[str, str]]:
        scores = {name: None for name in RAGAS_METRIC_NAMES}
        errors: dict[str, str] = {}
        reference = case.reference

        if case.expect_fallback:
            reason = "skipped: RAGAS answer metrics are not meaningful for expected fallback"
            return scores, {name: reason for name in RAGAS_METRIC_NAMES}

        arguments: dict[str, dict[str, Any] | None] = {
            "faithfulness": (
                {
                    "user_input": case.question,
                    "response": answer,
                    "retrieved_contexts": contexts,
                }
                if contexts
                else None
            ),
            "answer_relevancy": {
                "user_input": case.question,
                "response": answer,
            },
            "context_precision": (
                {
                    "user_input": case.question,
                    "reference": reference,
                    "retrieved_contexts": contexts,
                }
                if reference and contexts
                else None
            ),
            "context_recall": (
                {
                    "user_input": case.question,
                    "reference": reference,
                    "retrieved_contexts": contexts,
                }
                if reference and contexts
                else None
            ),
        }

        for name in RAGAS_METRIC_NAMES:
            kwargs = arguments[name]
            scorer = self.metrics.get(name)
            if scorer is None:
                errors[name] = "skipped: metric is not configured"
                continue
            if kwargs is None:
                errors[name] = "skipped: required reference or retrieved context is absent"
                continue
            try:
                scores[name] = self._score_value(scorer.score(**kwargs))
            except Exception as exc:
                errors[name] = _safe_error(exc)
        return scores, errors

    def _run_case(self, case: GoldenCase) -> CaseResult:
        started = time.perf_counter()
        base_metrics: dict[str, float | None] = {
            **{name: None for name in RAGAS_METRIC_NAMES},
            "fact_recall": None,
            "source_hit": None,
            "fallback_accuracy": None,
            "latency_ms": None,
        }
        try:
            trace = self.rag_chain.evaluate_trace(case.question)
            latency_ms = (time.perf_counter() - started) * 1000
            answer = str(trace.answer)
            contexts = [str(context) for context in trace.contexts]
            citations = [dict(citation) for citation in trace.citations]
            route_type = str(trace.route_type)
            ragas_scores, metric_errors = self._ragas_scores(
                case,
                answer=answer,
                contexts=contexts,
            )
            base_metrics.update(ragas_scores)
            base_metrics.update(
                {
                    "fact_recall": self._fact_recall(answer, case.expected_facts),
                    "source_hit": self._source_hit(citations, case.expected_sources),
                    "fallback_accuracy": float(
                        case.expect_fallback == (route_type == "out_of_scope")
                    ),
                    "latency_ms": latency_ms,
                }
            )
            return CaseResult(
                id=case.id,
                query_type=case.query_type,
                question=case.question,
                status="ok",
                error=None,
                answer=answer,
                route_type=route_type,
                contexts=contexts,
                citations=citations,
                reference=case.reference,
                metrics=base_metrics,
                metric_errors=metric_errors,
            )
        except Exception as exc:
            base_metrics["latency_ms"] = (time.perf_counter() - started) * 1000
            return CaseResult(
                id=case.id,
                query_type=case.query_type,
                question=case.question,
                status="error",
                error=_safe_error(exc),
                answer=None,
                route_type=None,
                contexts=[],
                citations=[],
                reference=case.reference,
                metrics=base_metrics,
            )

    def run(self, cases: Iterable[GoldenCase]) -> EvaluationReport:
        results = [self._run_case(case) for case in cases]
        return EvaluationReport(summary=self.summarize(results), cases=results)

    @staticmethod
    def _average(results: list[CaseResult], metric: str) -> float | None:
        values = [
            result.metrics[metric]
            for result in results
            if result.metrics.get(metric) is not None
        ]
        if not values:
            return None
        return sum(float(value) for value in values) / len(values)

    @classmethod
    def summarize(cls, results: list[CaseResult]) -> dict[str, int | float | None]:
        evaluated = [result for result in results if result.status == "ok"]
        expected_ragas_scores = sum(
            0 if result.route_type == "out_of_scope" else len(RAGAS_METRIC_NAMES)
            for result in evaluated
        )
        completed_ragas_scores = sum(
            result.metrics.get(metric) is not None
            for result in evaluated
            for metric in RAGAS_METRIC_NAMES
        )
        return {
            "cases": len(results),
            "ok": sum(result.status == "ok" for result in results),
            "errors": sum(result.status == "error" for result in results),
            "ragas_scores_expected": expected_ragas_scores,
            "ragas_scores_completed": completed_ragas_scores,
            "ragas_complete": float(completed_ragas_scores == expected_ragas_scores),
            "faithfulness": cls._average(results, "faithfulness"),
            "answer_relevancy": cls._average(results, "answer_relevancy"),
            "context_precision": cls._average(results, "context_precision"),
            "context_recall": cls._average(results, "context_recall"),
            "fact_recall": cls._average(results, "fact_recall"),
            "source_hit_rate": cls._average(results, "source_hit"),
            "fallback_accuracy": cls._average(results, "fallback_accuracy"),
            "avg_latency_ms": cls._average(results, "latency_ms"),
        }
