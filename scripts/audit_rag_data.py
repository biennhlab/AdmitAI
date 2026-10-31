"""Audit and export RAG inputs without loading a model or changing a live index.

Example: python scripts/audit_rag_data.py --output-dir data/prepared/rag-audit
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion import Chunk, combined_chunk, load_documents
from src.ingestion.chunker import _markdown_row
from src.ingestion.markdown import is_separator, parse_row


def _tokens(text: str) -> set[str]:
    # Markdown emphasis/escapes may split a visible token (S**PT**, 2026\-2027).
    # Compare what a reader sees, retaining full decimal/date values.
    text = re.sub(r"\\([\\`*_{}\[\]()#+\-.!|])", r"\1", text).replace("*", "")
    return set(re.findall(r"\d+(?:[.,:/-]\d+)*|[^\W\d_]+", text.casefold(), re.UNICODE))


def audit_documents(documents, chunks: list[Chunk], parents: list[Chunk]) -> dict:
    """Check source facts, table rows, IDs, and both directions of parent links."""
    by_document = defaultdict(list)
    parent_by_id = {chunk.chunk_id: chunk for chunk in parents}
    errors = []
    for chunk in chunks:
        by_document[chunk.metadata.get("doc_id")].append(chunk)
        parent = parent_by_id.get(chunk.metadata.get("parent_chunk_id"))
        if parent is None or chunk.chunk_id not in parent.metadata.get("child_chunk_ids", []):
            errors.append({"code": "orphan_chunk", "chunk_id": chunk.chunk_id})
    live_ids = {chunk.chunk_id for chunk in chunks}
    for parent in parents:
        children = parent.metadata.get("child_chunk_ids", [])
        if parent.metadata.get("child_count") != len(children) or any(child not in live_ids for child in children):
            errors.append({"code": "invalid_parent_registry", "chunk_id": parent.chunk_id})
    all_ids = [chunk.chunk_id for chunk in [*chunks, *parents]]
    if len(all_ids) != len(set(all_ids)):
        errors.append({"code": "duplicate_chunk_ids"})

    source_table_rows = 0
    preserved_table_rows = 0
    for document in documents:
        evidence = by_document[document.doc_id]
        if not evidence:
            errors.append({"code": "document_without_chunks", "doc_id": document.doc_id})
            continue
        indexes = [chunk.metadata.get("chunk_index") for chunk in evidence]
        if len(indexes) != len(set(indexes)):
            errors.append({"code": "duplicate_document_chunk_indexes", "doc_id": document.doc_id})
        content_tokens = _tokens("\n".join(chunk.content + "\n" + "\n".join(chunk.metadata.get("heading_path", [])) for chunk in evidence))
        source_tokens = _tokens("\n".join(page.text for page in document.pages))
        missing = source_tokens - content_tokens
        if missing:
            errors.append({"code": "source_tokens_missing", "doc_id": document.doc_id, "tokens": sorted(missing)})
        # Check complete data rows (including order and all numeric values),
        # not just vocabulary coverage. Multi-tier headers are flattened.
        table_index = 0
        for page in document.pages:
            for table in page.tables:
                if not table.rows:
                    continue
                parts = [chunk for chunk in evidence if chunk.metadata.get("chunk_type") == "table" and chunk.metadata.get("table_index") == table_index]
                for row_number, row in enumerate(table.rows[table.header_row_count:], start=table.header_row_count):
                    source_table_rows += 1
                    observed = defaultdict(set)
                    for chunk in parts:
                        if not chunk.metadata["row_start"] <= row_number <= chunk.metadata["row_end"]:
                            continue
                        rendered_rows = [parse_row(line) for line in chunk.content.splitlines() if line.startswith("|")]
                        rendered_rows = [values for values in rendered_rows if values is not None and not is_separator(values)]
                        local_index = row_number - chunk.metadata["row_start"] + 1
                        if local_index >= len(rendered_rows):
                            continue
                        columns = chunk.metadata.get("column_indexes", list(range(len(table.rows[0]))))
                        for column, value in zip(columns, rendered_rows[local_index]):
                            observed[column].add(value)
                    if all(str(value).replace("\n", " ").strip() in observed[column] for column, value in enumerate(row)):
                        preserved_table_rows += 1
                    else:
                        errors.append({"code": "table_row_missing", "doc_id": document.doc_id, "table_index": table_index, "row": row})
                table_index += 1
    children = [chunk for chunk in chunks if chunk.metadata.get("chunk_type") == "child"]
    tables = [chunk for chunk in chunks if chunk.metadata.get("chunk_type") == "table"]
    return {
        "documents": len(documents), "chunks": len(chunks), "parents": len(parents),
        "children": len(children), "table_chunks": len(tables),
        "tiny_children_under_80_chars": sum(len(chunk.content) < 80 for chunk in children),
        "oversized_atomic_table_chunks": sum(bool(chunk.metadata.get("oversized_atomic_table")) for chunk in tables),
        "multi_header_table_chunks": sum(chunk.metadata.get("header_row_count", 1) > 1 for chunk in tables),
        "source_table_rows": source_table_rows, "preserved_table_rows": preserved_table_rows,
        "errors": errors,
    }


def evaluate_lexical(chunks: list[Chunk], parents: list[Chunk], dataset: dict, top_k: int = 10) -> dict:
    """Fixed golden questions; no LLM, synthetic embeddings, or model calls.

    This is a sparse retrieval diagnostic, not an end-to-end RAG quality score.
    """
    from src.retrieval import BM25Search, HybridRetriever
    from src.generation.assembler import ContextAssembler

    class EmptyDense:
        def search(self, query, top_k):
            return []

    sparse = BM25Search()
    sparse.index(chunks)
    retriever = HybridRetriever(EmptyDense(), sparse, parent_chunks=[asdict(parent) for parent in parents])
    cases = []
    for case in dataset["cases"]:
        if case["expected_route"] != "general":
            continue
        results = ContextAssembler(max_chars=12000).assemble(retriever.search(case["question"], top_k=top_k))
        text = " ".join(chunk.content for chunk, _score in results).casefold()
        required = case["required_terms"]
        coverage = sum(term.casefold() in text for term in required) / len(required)
        expected_docs = {dataset["evidence_catalog"][key]["doc_id"] for key in case["evidence_ids"]}
        found_docs = {chunk.metadata.get("doc_id") for chunk, _score in results}
        cases.append({"id": case["id"], "term_coverage": coverage, "source_recall": len(expected_docs & found_docs) / len(expected_docs)})
    return {"branch": "BM25 + parent/table expansion", "top_k": top_k, "context_budget_chars": 12000, "cases": cases,
            "mean_term_coverage": sum(case["term_coverage"] for case in cases) / max(1, len(cases)),
            "mean_source_recall": sum(case["source_recall"] for case in cases) / max(1, len(cases))}


def evaluate_product(chunks: list[Chunk], parents: list[Chunk], dataset: dict) -> dict:
    """Use the application's actual evidence-preparation path, without a provider."""
    from src.config import settings
    from src.generation.rag_chain import RAGChain
    from src.retrieval import BM25Search, HybridRetriever
    from src.retrieval.lite_selector import LiteEvidenceSelector

    class EmptyDense:
        def search(self, query, top_k):
            return []

    sparse = BM25Search()
    sparse.index(chunks)
    retriever = HybridRetriever(EmptyDense(), sparse, [asdict(parent) for parent in parents])
    chain = RAGChain(retriever, None, top_k=settings.RETRIEVAL_TOP_K, min_score=None,
                     min_lexical_coverage=settings.RETRIEVAL_MIN_LEXICAL_COVERAGE, reranker=LiteEvidenceSelector())
    cases = []
    for case in dataset["cases"]:
        prepared = chain._prepare_generation(case["question"], None)
        sources = [] if prepared is None else [source.prompt_chunk for source in prepared.sources]
        text = " ".join(chunk.content for chunk in sources).casefold()
        required = case["required_terms"]
        expected_docs = {dataset["evidence_catalog"][key]["doc_id"] for key in case["evidence_ids"]}
        found_docs = {chunk.metadata.get("doc_id") for chunk in sources}
        cases.append({"id": case["id"], "route_correct": (prepared is None) == (case["expected_route"] == "out_of_scope"),
                      "expected_route": case["expected_route"],
                      "term_coverage": sum(term.casefold() in text for term in required) / len(required) if required else None,
                      "source_recall": len(expected_docs & found_docs) / len(expected_docs) if expected_docs else None})
    positive = [case for case in cases if case["expected_route"] == "general"]
    return {"branch": "application evidence preparation with BM25 + domain lite; empty dense branch",
            "provider_requests": 0, "cases": cases,
            "route_accuracy": sum(case["route_correct"] for case in cases) / len(cases),
            "mean_term_coverage": sum(case["term_coverage"] for case in positive) / len(positive),
            "mean_source_recall": sum(case["source_recall"] for case in positive) / len(positive)}


def _write_jsonl(path: Path, values) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def source_coverage_gaps(source_dir: Path) -> dict:
    quality_path = source_dir / "ptit" / "quality_report.json"
    if not quality_path.exists():
        quality_path = source_dir / "quality_report.json"
    if not quality_path.exists():
        return {"status": "not_audited", "ocr_required_page_count": None}
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    missing_pages = quality.get("content_preservation", {}).get("pdf_pages_without_processed_text", [])
    return {"status": "incomplete" if missing_pages else "reported_complete",
            "source_report_checked_at": quality.get("checked_at"),
            "ocr_required_page_count": len(missing_pages), "ocr_required_pages": missing_pages}


def run_audit(data_dir: Path, output_dir: Path, *, child_size: int = 500,
              parent_size: int = 1500, table_size: int = 1200,
              evaluate: bool = False, compare: bool = False, baseline_index: Path | None = None) -> dict:
    loaded = load_documents(data_dir)
    parents = []
    chunks = []
    for document in loaded.documents:
        metadata = {**document.metadata, "doc_id": document.doc_id}
        document_chunks = combined_chunk(document.pages, max_size=table_size, parent_size=parent_size,
                                         child_size=child_size, metadata=metadata, parent_chunks=parents)
        for chunk in document_chunks:
            chunk.metadata["document_chunk_count"] = len(document_chunks)
        chunks.extend(document_chunks)
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "data_dir": str(data_dir.resolve()),
              "parameters": {"child_size": child_size, "parent_size": parent_size, "table_size": table_size},
              "load_errors": loaded.errors, "load_warnings": loaded.warnings,
              "skipped_duplicates": loaded.skipped_duplicates,
              "quality": audit_documents(loaded.documents, chunks, parents)}
    report["source_coverage_gaps"] = source_coverage_gaps(Path(loaded.source_dir))
    report["historical_faq_sources"] = [
        {"doc_id": document.doc_id, "published_at": document.metadata.get("published_at"), "collection_admission_year": document.metadata.get("collection_admission_year"),
         "admission_year": document.metadata.get("admission_year"), "applicability_note": document.metadata.get("applicability_note"),
         "reason": "FAQ publication predates assigned admission year; verify applicability before using current-year policy."}
        for document in loaded.documents
        if "cau-hoi-thuong-gap" in document.doc_id
        and str(document.metadata.get("published_at", ""))[:4].isdigit()
        and document.metadata.get("temporal_scope") == "historical_or_unspecified"
    ]
    report["media_only_program_introductions"] = [
        {"doc_id": document.doc_id, "source_url": document.metadata.get("source_url"),
         "duplicate_sources": document.metadata.get("duplicate_sources", []),
         "reason": "Only generic video-introduction text is available; a programme-specific transcript/profile is still required."}
        for document in loaded.documents
        if "Trà chiều cùng trường P" in document.content and len(document.content) < 800
    ]
    if evaluate or compare:
        dataset = json.loads((PROJECT_ROOT / "eval/golden_dataset.json").read_text(encoding="utf-8"))
        report["lexical_evaluation"] = evaluate_lexical(chunks, parents, dataset)
        report["product_evaluation"] = evaluate_product(chunks, parents, dataset)
        if baseline_index:
            baseline = json.loads((baseline_index / "manifest.json").read_text(encoding="utf-8"))
            baseline_chunks = [Chunk(**item) for item in baseline["chunks"]]
            baseline_parents = [Chunk(**item) for item in baseline.get("parent_chunks", [])]
            report["baseline"] = {"index_dir": str(baseline_index), "chunks": len(baseline_chunks),
                                   "lexical_evaluation": evaluate_lexical(baseline_chunks, baseline_parents, dataset),
                                   "product_evaluation": evaluate_product(baseline_chunks, baseline_parents, dataset)}
        if compare:
            report["profile_comparison"] = []
            for size in (300, 500, 700, 1000):
                trial_chunks, trial_parents = [], []
                for document in loaded.documents:
                    trial_chunks.extend(combined_chunk(document.pages, child_size=size, parent_size=parent_size, max_size=table_size,
                                                        metadata=document.metadata, parent_chunks=trial_parents))
                report["profile_comparison"].append({"child_size": size,
                    "quality": audit_documents(loaded.documents, trial_chunks, trial_parents),
                    "lexical_evaluation": evaluate_lexical(trial_chunks, trial_parents, dataset),
                    "product_evaluation": evaluate_product(trial_chunks, trial_parents, dataset)})
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "chunks.jsonl", (asdict(chunk) for chunk in chunks))
    _write_jsonl(output_dir / "parents.jsonl", (asdict(parent) for parent in parents))
    _write_jsonl(output_dir / "documents.jsonl", ({"doc_id": document.doc_id, "metadata": document.metadata, "content": document.content} for document in loaded.documents))
    from scripts.ingest import DATA_PIPELINE_REVISION, corpus_fingerprint
    import hashlib
    report["prepared_corpus"] = {
        "data_pipeline_revision": DATA_PIPELINE_REVISION,
        "corpus_fingerprint": corpus_fingerprint(loaded.documents, 3, child_size=child_size, parent_size=parent_size, table_size=table_size),
        "artifact_sha256": {name: hashlib.sha256((output_dir / name).read_bytes()).hexdigest()
                            for name in ("chunks.jsonl", "parents.jsonl", "documents.jsonl")},
    }
    (output_dir / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data/processed")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--child-size", type=int, default=500)
    parser.add_argument("--parent-size", type=int, default=1500)
    parser.add_argument("--table-size", type=int, default=1200)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--baseline-index", type=Path)
    args = parser.parse_args()
    report = run_audit(args.data_dir, args.output_dir, child_size=args.child_size, parent_size=args.parent_size,
                       table_size=args.table_size, evaluate=args.evaluate, compare=args.compare, baseline_index=args.baseline_index)
    quality = report["quality"]
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "quality": quality,
                      "load_errors": report["load_errors"], "ocr_required_pages": report.get("source_coverage_gaps", {}).get("ocr_required_page_count", 0)}, ensure_ascii=False, indent=2))
    return int(bool(report["load_errors"] or quality["errors"]))


if __name__ == "__main__":
    raise SystemExit(main())
