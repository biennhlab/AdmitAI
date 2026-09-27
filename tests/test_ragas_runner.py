from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from scripts.run_eval import write_comparison_table
from src.eval import GoldenCase, RagasRunner


class StaticMetric:
    def __init__(self, value: float):
        self.value = value
        self.calls = []

    def score(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(value=self.value)


class FakeChain:
    def evaluate_trace(self, question: str):
        return SimpleNamespace(
            answer="Học phí là 35,9 triệu đồng mỗi năm [1].",
            contexts=["Học phí dự kiến từ 35,9 triệu đồng/năm."],
            citations=[{"doc_id": "tuition-2026", "page": 1}],
            route_type="general",
        )


def test_golden_dataset_has_at_least_thirty_unique_cases():
    cases = RagasRunner.load_dataset("src/eval/golden_dataset.json")

    assert len(cases) >= 30
    assert len({case.id for case in cases}) == len(cases)
    assert all(case.reference or case.expect_fallback for case in cases)


def test_load_dataset_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "duplicate.json"
    case = {"id": "same", "query_type": "factual", "question": "Question?"}
    path.write_text(json.dumps([case, case]), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate ids"):
        RagasRunner.load_dataset(path)


def test_runner_scores_trace_and_project_metrics():
    metrics = {name: StaticMetric(0.75) for name in (
        "faithfulness", "answer_relevancy", "context_precision", "context_recall"
    )}
    runner = RagasRunner(FakeChain(), metrics=metrics)
    case = GoldenCase(
        id="tuition",
        query_type="factual",
        question="Học phí bao nhiêu?",
        expected_answer="Học phí là 35,9 triệu đồng mỗi năm.",
        expected_facts=["35,9 triệu đồng"],
        expected_sources=[RagasRunner._parse_source({"doc_id": "tuition-2026", "page": 1}, 0, 0)],
    )

    report = runner.run([case])

    assert report.summary["errors"] == 0
    assert report.summary["ragas_complete"] == 1.0
    assert report.summary["faithfulness"] == 0.75
    assert report.summary["fact_recall"] == 1.0
    assert report.summary["source_hit_rate"] == 1.0
    assert metrics["faithfulness"].calls[0]["retrieved_contexts"] == [
        "Học phí dự kiến từ 35,9 triệu đồng/năm."
    ]


def test_metric_failure_is_recorded_without_losing_the_case():
    class BrokenMetric:
        def score(self, **kwargs):
            raise RuntimeError("judge unavailable")

    runner = RagasRunner(
        FakeChain(),
        metrics={
            "faithfulness": BrokenMetric(),
            "answer_relevancy": StaticMetric(0.5),
            "context_precision": StaticMetric(0.5),
            "context_recall": StaticMetric(0.5),
        },
    )
    report = runner.run([
        GoldenCase(
            id="case",
            query_type="factual",
            question="Học phí?",
            expected_answer="35,9 triệu đồng",
        )
    ])

    assert report.summary["ok"] == 1
    assert report.summary["ragas_complete"] == 0.0
    assert report.cases[0].metrics["faithfulness"] is None
    assert "judge unavailable" in report.cases[0].metric_errors["faithfulness"]


def test_comparison_table_uses_only_real_report_files(tmp_path):
    report = {
        "summary": {
            "cases": 5,
            "context_precision": 0.8,
            "context_recall": 0.7,
            "faithfulness": 0.9,
            "answer_relevancy": 0.85,
        }
    }
    (tmp_path / "module6_self_rag.json").write_text(json.dumps(report), encoding="utf-8")
    output = tmp_path / "comparison_table.md"

    write_comparison_table(tmp_path, output)

    text = output.read_text(encoding="utf-8")
    assert "M6: Self Rag" in text
    assert "0.9000" in text
    assert "M5:" not in text
