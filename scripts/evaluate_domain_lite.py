"""Provider-free evaluation on actual Phase A BM25 retrieval (no model loads).

Compares the pre-change lexical lite pipeline with domain lite. Dense retrieval,
neural inference and generated-answer citation support are deliberately unmeasured.
Golden evidence is used only to score outputs, never to seed retrieval.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import statistics
import sys
import time
import tracemalloc
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import settings
from src.generation.rag_chain import RAGChain
from src.ingestion.chunker import Chunk
from src.retrieval.domain_signals import QueryScope, assess
from src.retrieval.fusion import HybridRetriever
from src.retrieval.lite_selector import LiteEvidenceSelector, _tokens
from src.retrieval.sparse_search import BM25Search

CASE_IDS = (
    "admissions_sat_act_threshold", "cutoff_infosec_compare_campuses",
    "tuition_compare_2026", "negative_veterinary",
    "negative_dorm_availability",
)


def host_memory():
    if sys.platform != "win32":
        return None

    class MemoryStatus(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
            (key, ctypes.c_ulonglong) for key in (
                "total_physical", "available_physical", "total_commit",
                "available_commit", "total_virtual", "available_virtual", "extended",
            )
        ]

    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    return {key + "_gib": round(getattr(status, key) / 1024**3, 3) for key in (
        "total_physical", "available_physical", "total_commit", "available_commit",
    )}


class EmptyDense:
    def search(self, query, top_k):
        return []


class LexicalBaseline:
    """Frozen pre-change lite scoring; no neural backend is instantiated."""
    def rerank(self, query, candidates, top_k):
        tokens = _tokens(query)
        ranked = [(chunk, score, LiteEvidenceSelector._score(tokens, chunk)) for chunk, score in candidates]
        return sorted(ranked, key=lambda item: (-item[2], -item[1], item[0].chunk_id))[:top_k]


def retrieval_metrics(selected, evidence_ids, parent_by_id):
    ids = [chunk.chunk_id for chunk, _ in selected]
    relevant = [bool({evidence_id, parent_by_id.get(evidence_id)} & set(ids)) for evidence_id in evidence_ids]
    ranks = [index for index, cid in enumerate(ids, 1) if any(cid in {eid, parent_by_id.get(eid)} for eid in evidence_ids)]
    return {
        "recall_at_k_exact": len(set(ids) & set(evidence_ids)) / len(evidence_ids) if evidence_ids else None,
        "golden_coverage_with_parent": sum(relevant) / len(evidence_ids) if evidence_ids else None,
        "mrr_with_parent": 1 / min(ranks) if ranks else 0.0 if evidence_ids else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", default="data/index/module3-phase-a-20261019-223230")
    parser.add_argument("--output", default="domain-rrf-results/evaluation.json")
    parser.add_argument("--all-cases", action="store_true")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--dense-qdrant", action="store_true", help="Load local BGE-M3 and use the existing Qdrant dense branch")
    args = parser.parse_args()
    from src.local_profile import LOCAL_INDEX_DIR, LOCAL_COLLECTION
    if (ROOT / args.index_dir).resolve() != (ROOT / LOCAL_INDEX_DIR).resolve():
        parser.error("Only the validated local index is supported")
    if args.runs <= 0:
        parser.error("runs must be positive")
    memory_before = host_memory()
    path = ROOT / args.index_dir / "manifest.json"
    manifest_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("qdrant_collection") != LOCAL_COLLECTION:
        raise RuntimeError("Index/collection identity mismatch")
    dataset_path = ROOT / "eval/golden_dataset.json"
    dataset_hash = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    chunks = [Chunk(**chunk) for chunk in manifest["chunks"]]
    parent_by_id = {chunk.chunk_id: (chunk.metadata or {}).get("parent_chunk_id") for chunk in chunks}
    cases = [case for case in dataset["cases"] if args.all_cases or case["id"] in CASE_IDS]
    sparse = BM25Search()
    sparse.index(chunks)  # Temporary in-memory BM25 only; never persist an index.
    dense = EmptyDense()
    if args.dense_qdrant:
        from qdrant_client import QdrantClient
        from src.retrieval.embedder import Embedder
        from src.retrieval.dense_search import QdrantDenseSearch
        import atexit
        settings.MODELS_LOCAL_ONLY = True
        client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
        atexit.register(client.close)
        collection = manifest["qdrant_collection"]
        if client.count(collection_name=collection, exact=True).count != len(chunks):
            raise RuntimeError("Qdrant/local index counts differ")
        dense = QdrantDenseSearch(client, collection, Embedder())
        dense.validate_ready(manifest["embedding_dimension"])
    retriever = HybridRetriever(dense, sparse, manifest.get("parent_chunks"))
    results = {}
    for mode, selector in (("baseline_lite", LexicalBaseline()), ("domain_lite", LiteEvidenceSelector())):
        chain = RAGChain(
            retriever, None, top_k=settings.RETRIEVAL_TOP_K, min_score=None,
            min_lexical_coverage=settings.RETRIEVAL_MIN_LEXICAL_COVERAGE,
            reranker=selector,
        )
        rows = []
        tracemalloc.start()
        for case in cases:
            durations = []
            for _ in range(args.runs):
                started = time.perf_counter()
                prepared = chain._prepare_generation(case["question"], None)
                durations.append((time.perf_counter() - started) * 1000)
            selected = [] if prepared is None else [(source.prompt_chunk, source.retrieval_score) for source in prepared.sources]
            scope = QueryScope.parse(case["question"])
            decisions = [assess(scope, chunk) for chunk, _ in selected]
            rows.append({
                "case_id": case["id"], "expected_route": case["expected_route"],
                "actual_route": "out_of_scope" if prepared is None else "general",
                "selected_ids": [chunk.chunk_id for chunk, _ in selected],
                **retrieval_metrics(selected, case["evidence_ids"], parent_by_id),
                "wrong_year_rate": sum(any("year" in reason or "academic" in reason for reason in decision.exclusions) for decision in decisions) / len(decisions) if decisions else 0.0,
                "wrong_campus_rate": sum("wrong_campus" in decision.exclusions for decision in decisions) / len(decisions) if decisions else 0.0,
                "ambiguities": {chunk.chunk_id: list(decision.ambiguities) for (chunk, _), decision in zip(selected, decisions) if decision.ambiguities},
                "prompt_chunk_ids_unique": len({chunk.chunk_id for chunk, _ in selected}) == len(selected),
                "generated_citation_support": None,
                "latency_ms_median": round(statistics.median(durations), 3),
            })
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        metric_names = ("recall_at_k_exact", "golden_coverage_with_parent", "mrr_with_parent", "wrong_year_rate", "wrong_campus_rate", "latency_ms_median")
        aggregate = {key: statistics.mean(row[key] for row in rows if row[key] is not None) for key in metric_names}
        negatives = [row for row in rows if row["expected_route"] == "out_of_scope"]
        aggregate["negative_false_positive_rate"] = sum(row["actual_route"] != "out_of_scope" for row in negatives) / len(negatives) if negatives else None
        results[mode] = {"aggregate": aggregate, "python_peak_kib": round(peak / 1024, 2), "cases": rows}
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest_hash
    assert hashlib.sha256(dataset_path.read_bytes()).hexdigest() == dataset_hash
    result = {
        "retrieval_source": "actual_phase_a_dense_qdrant_bm25" if args.dense_qdrant else "actual_phase_a_bm25_only_empty_dense_branch",
        "golden_injected_into_retrieval": False, "provider_requests": 0,
        "api_started": False, "cross_encoder_loaded": False,
        "baseline": "pre-change lexical lite, not neural",
        "searchable_chunks": len(chunks), "case_count": len(cases), "runs": args.runs,
        "manifest_sha256": manifest_hash, "golden_sha256": dataset_hash,
        "memory_before": memory_before, "memory_after": host_memory(),
        "limitations": (["Dense/Qdrant quality not measured"] if not args.dense_qdrant else []) + ["Neural quality not measured", "Generated answer citation support not measured", "tracemalloc excludes model/native and whole-process memory", "Parent-equivalent coverage does not establish claim entailment"],
        "results": results,
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "memory": result["memory_before"], "case_count": len(cases), "results": {mode: value["aggregate"] for mode, value in results.items()}}))


if __name__ == "__main__":
    main()
