"""Validate the checked-in AdmitAI golden dataset and its corpus evidence."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.local_profile import LOCAL_INDEX_DIR

DEFAULT_DATASET = PROJECT_ROOT / "eval" / "golden_dataset.json"
DEFAULT_MANIFEST = PROJECT_ROOT / LOCAL_INDEX_DIR / "manifest.json"
FALLBACK_ANSWER = "Mình chưa tìm thấy thông tin này trong dữ liệu tuyển sinh hiện có."
REQUIRED_CASE_FIELDS = {
    "id",
    "category",
    "difficulty",
    "question",
    "expected_route",
    "ground_truth",
    "evidence_ids",
    "required_terms",
    "tags",
}
REQUIRED_EVIDENCE_FIELDS = {
    "chunk_id",
    "doc_id",
    "title",
    "page",
    "source_url",
    "quote",
}


def _normalized(text: str) -> str:
    """Normalize Markdown-heavy corpus text for stable evidence comparisons."""
    text = text.replace("\\", "").replace("*", "").replace("_", "").replace("`", "")
    return " ".join(text.casefold().split())


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def validate_dataset(
    dataset: dict[str, Any], manifest: dict[str, Any] | None = None
) -> list[str]:
    errors: list[str] = []
    if dataset.get("schema_version") != 1:
        errors.append("schema_version must be 1")

    corpus = dataset.get("corpus")
    if not isinstance(corpus, dict):
        errors.append("corpus must be an object")
        corpus = {}
    fingerprint = corpus.get("fingerprint")
    if not isinstance(fingerprint, str) or re.fullmatch(r"[0-9a-f]{64}", fingerprint) is None:
        errors.append("corpus.fingerprint must be a lowercase SHA-256 value")

    evidence_catalog = dataset.get("evidence_catalog")
    if not isinstance(evidence_catalog, dict) or not evidence_catalog:
        errors.append("evidence_catalog must be a non-empty object")
        evidence_catalog = {}
    for evidence_id, evidence in evidence_catalog.items():
        prefix = f"evidence_catalog[{evidence_id!r}]"
        if not isinstance(evidence, dict):
            errors.append(f"{prefix} must be an object")
            continue
        missing = REQUIRED_EVIDENCE_FIELDS - evidence.keys()
        if missing:
            errors.append(f"{prefix} missing fields: {sorted(missing)}")
        if evidence.get("chunk_id") != evidence_id:
            errors.append(f"{prefix}.chunk_id must equal its catalog key")
        for field in ("doc_id", "title", "source_url", "quote"):
            if not isinstance(evidence.get(field), str) or not evidence[field].strip():
                errors.append(f"{prefix}.{field} must be a non-empty string")
        if not isinstance(evidence.get("page"), int) or evidence["page"] <= 0:
            errors.append(f"{prefix}.page must be a positive integer")

    cases = dataset.get("cases")
    if not isinstance(cases, list):
        errors.append("cases must be a list")
        cases = []
    if not 30 <= len(cases) <= 50:
        errors.append("cases must contain between 30 and 50 items")

    seen_case_ids: set[str] = set()
    seen_questions: set[str] = set()
    for index, case in enumerate(cases):
        prefix = f"cases[{index}]"
        if not isinstance(case, dict):
            errors.append(f"{prefix} must be an object")
            continue
        missing = REQUIRED_CASE_FIELDS - case.keys()
        if missing:
            errors.append(f"{prefix} missing fields: {sorted(missing)}")
            continue
        case_id = case["id"]
        if not isinstance(case_id, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", case_id):
            errors.append(f"{prefix}.id must be a lowercase slug")
        elif case_id in seen_case_ids:
            errors.append(f"duplicate case id: {case_id}")
        else:
            seen_case_ids.add(case_id)

        question = case["question"]
        normalized_question = _normalized(question) if isinstance(question, str) else ""
        if not normalized_question:
            errors.append(f"{prefix}.question must be a non-empty string")
        elif normalized_question in seen_questions:
            errors.append(f"duplicate question: {question}")
        else:
            seen_questions.add(normalized_question)

        if case["expected_route"] not in {"general", "out_of_scope"}:
            errors.append(f"{prefix}.expected_route is invalid")
        if case["difficulty"] not in {"single_hop", "multi_hop", "negative"}:
            errors.append(f"{prefix}.difficulty is invalid")
        if not isinstance(case["tags"], list) or not all(
            isinstance(tag, str) and tag for tag in case["tags"]
        ):
            errors.append(f"{prefix}.tags must be a list of non-empty strings")

        evidence_ids = case["evidence_ids"]
        required_terms = case["required_terms"]
        if not isinstance(evidence_ids, list) or not all(isinstance(item, str) for item in evidence_ids):
            errors.append(f"{prefix}.evidence_ids must be a list of strings")
            evidence_ids = []
        if len(evidence_ids) != len(set(evidence_ids)):
            errors.append(f"{prefix}.evidence_ids contains duplicates")
        unknown_evidence = sorted(set(evidence_ids) - evidence_catalog.keys())
        if unknown_evidence:
            errors.append(f"{prefix} references unknown evidence: {unknown_evidence}")
        if not isinstance(required_terms, list) or not all(
            isinstance(term, str) and term.strip() for term in required_terms
        ):
            errors.append(f"{prefix}.required_terms must be a list of non-empty strings")
            required_terms = []

        ground_truth = case["ground_truth"]
        if not isinstance(ground_truth, str) or not ground_truth.strip():
            errors.append(f"{prefix}.ground_truth must be a non-empty string")
            ground_truth = ""

        if case["expected_route"] == "general":
            if not evidence_ids:
                errors.append(f"{prefix} general case must reference evidence")
            if not required_terms:
                errors.append(f"{prefix} general case must define required_terms")
            evidence_text = " ".join(
                evidence_catalog[evidence_id]["quote"]
                for evidence_id in evidence_ids
                if evidence_id in evidence_catalog
            )
            normalized_evidence = _normalized(evidence_text)
            normalized_truth = _normalized(ground_truth)
            for term in required_terms:
                normalized_term = _normalized(term)
                if normalized_term not in normalized_evidence:
                    errors.append(f"{prefix} term {term!r} is absent from evidence quotes")
                if normalized_term not in normalized_truth:
                    errors.append(f"{prefix} term {term!r} is absent from ground_truth")
        else:
            if case["difficulty"] != "negative":
                errors.append(f"{prefix} out_of_scope case must have negative difficulty")
            if evidence_ids or required_terms:
                errors.append(f"{prefix} out_of_scope case must not contain evidence or required terms")
            if ground_truth != FALLBACK_ANSWER:
                errors.append(f"{prefix} out_of_scope ground_truth must use the product fallback")

    if manifest is not None:
        if fingerprint != manifest.get("corpus_fingerprint"):
            errors.append("dataset corpus fingerprint differs from index manifest")
        manifest_chunks = {
            chunk["chunk_id"]: chunk
            for chunk in [*manifest.get("chunks", []), *manifest.get("parent_chunks", [])]
        }
        for evidence_id, evidence in evidence_catalog.items():
            prefix = f"evidence_catalog[{evidence_id!r}]"
            chunk = manifest_chunks.get(evidence_id)
            if chunk is None:
                errors.append(f"{prefix} is missing from the index manifest")
                continue
            metadata = chunk.get("metadata") or {}
            for field, metadata_field in (
                ("doc_id", "doc_id"),
                ("title", "title"),
                ("page", "page_number"),
                ("source_url", "source_url"),
            ):
                if evidence[field] != metadata.get(metadata_field):
                    errors.append(f"{prefix}.{field} differs from the index manifest")
            if _normalized(evidence["quote"]) not in _normalized(chunk.get("content", "")):
                errors.append(f"{prefix}.quote is not present in indexed chunk content")

    return errors


def dataset_summary(dataset: dict[str, Any]) -> str:
    cases = dataset["cases"]
    routes = Counter(case["expected_route"] for case in cases)
    difficulties = Counter(case["difficulty"] for case in cases)
    categories = Counter(case["category"] for case in cases)
    return (
        f"valid cases={len(cases)} evidence={len(dataset['evidence_catalog'])} "
        f"routes={dict(routes)} difficulties={dict(difficulties)} categories={dict(categories)}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--schema-only",
        action="store_true",
        help="Validate the checked-in artifact without requiring the local ignored index.",
    )
    args = parser.parse_args()

    dataset = load_json(args.dataset)
    manifest = None if args.schema_only else load_json(args.manifest)
    errors = validate_dataset(dataset, manifest)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(dataset_summary(dataset))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
