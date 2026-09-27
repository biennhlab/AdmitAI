from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.routers import chat
from src.config import settings
from src.eval import RagasRunner


DEFAULT_DATASET = PROJECT_ROOT / "src" / "eval" / "golden_dataset.json"
RESULT_DIR = PROJECT_ROOT / "eval_results"
MODULE_NAMES = {
    1: "naive_rag",
    2: "advanced_chunking",
    3: "hybrid_search",
    4: "query_transform",
    5: "reranking",
    6: "self_rag",
    7: "agentic",
}
COMPARISON_METRICS = (
    "context_precision",
    "context_recall",
    "faithfulness",
    "answer_relevancy",
)


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return value


def _configured_key() -> bool:
    return settings.LLM_API_KEY.strip().lower() not in {
        "",
        "placeholder",
        "your_gemini_api_key_here",
    }


def _default_output(module: int) -> Path:
    return RESULT_DIR / f"module{module}_{MODULE_NAMES[module]}.json"


def _format_score(value: Any) -> str:
    return "—" if value is None else f"{float(value):.4f}"


def write_comparison_table(result_dir: Path, output: Path) -> None:
    rows: list[tuple[int, str, dict[str, Any]]] = []
    for module, name in MODULE_NAMES.items():
        path = result_dir / f"module{module}_{name}.json"
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            summary = payload["summary"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError):
            continue
        rows.append((module, name, summary))

    lines = [
        "# RAGAS comparison by module",
        "",
        "Only executed reports are listed; missing modules are not estimated.",
        "",
        "| Module | Context Precision | Context Recall | Faithfulness | Answer Relevancy | Cases |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for module, name, summary in rows:
        scores = [_format_score(summary.get(metric)) for metric in COMPARISON_METRICS]
        lines.append(
            f"| M{module}: {name.replace('_', ' ').title()} | "
            + " | ".join(scores)
            + f" | {summary.get('cases', 0)} |"
        )
    if not rows:
        lines.append("| No completed evaluations | — | — | — | — | 0 |")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the production AdmitAI RAG chain against the golden dataset."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--limit", type=_positive_int)
    parser.add_argument("--module", type=int, choices=MODULE_NAMES, default=6)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--comparison-output",
        type=Path,
        default=RESULT_DIR / "comparison_table.md",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not _configured_key():
        print("LLM_API_KEY is not configured; RAG generation and RAGAS require it.", file=sys.stderr)
        return 2

    try:
        cases = RagasRunner.load_dataset(args.dataset)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if args.limit:
        cases = cases[: args.limit]

    try:
        chat.initialize_rag(settings.INDEX_DIR)
        runner = RagasRunner(chat.rag_chain)
        report = runner.run(cases)
    except Exception as exc:
        print(f"Evaluation setup failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    output = args.output or _default_output(args.module)
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    payload = {
        "module": args.module,
        "module_name": MODULE_NAMES[args.module],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(args.dataset.resolve()),
        "dataset_cases": len(cases),
        "config": {
            "embedding_model": settings.EMBEDDING_MODEL,
            "reranker_model": settings.RERANKER_MODEL,
            "llm_model": settings.LLM_MODEL,
            "retrieval_top_k": settings.RETRIEVAL_TOP_K,
            "rerank_top_k": settings.RERANK_TOP_K,
        },
        **report.to_dict(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    comparison_output = args.comparison_output
    if not comparison_output.is_absolute():
        comparison_output = PROJECT_ROOT / comparison_output
    write_comparison_table(output.parent, comparison_output)

    print(json.dumps(report.summary, ensure_ascii=False, indent=2))
    print(f"Detailed report: {output}")
    print(f"Comparison table: {comparison_output}")
    if report.summary["errors"] or not report.summary["ragas_complete"]:
        print("Evaluation is incomplete; inspect case and metric errors in the report.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
