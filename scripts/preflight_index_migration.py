"""Read-only safety checks before rebuilding or cutting over a hybrid index."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Callable

from qdrant_client import QdrantClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ingest import build_chunks, build_fingerprint, corpus_fingerprint
from scripts.validate_golden_dataset import DEFAULT_DATASET, load_json, validate_dataset
from src.config import settings
from src.qdrant_integrity import inspect_qdrant_points as _inspect_qdrant_points
from src.runtime_resources import failure_code
from src.ingestion import load_documents
from src.retrieval import QdrantDenseSearch, load_dense_index, read_manifest
from src.retrieval.index_version import INDEX_BUILD_REVISION, manifest_build_stale_reasons


def _directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _existing_ancestor(path: Path) -> Path:
    candidate = path.resolve()
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


def _duplicate_ids(records: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for record in records:
        chunk_id = record.get("chunk_id")
        if not isinstance(chunk_id, str) or not chunk_id:
            duplicates.add("<missing>")
        elif chunk_id in seen:
            duplicates.add(chunk_id)
        else:
            seen.add(chunk_id)
    return sorted(duplicates)


def run_preflight(
    *,
    index_dir: str | Path,
    data_dir: str | Path,
    collection: str,
    dataset_path: str | Path = DEFAULT_DATASET,
    qdrant_client: Any | None = None,
    disk_usage: Callable[[str | Path], Any] = shutil.disk_usage,
) -> dict[str, Any]:
    """Inspect migration inputs without creating, updating, or deleting data."""
    index_path = Path(index_dir)
    report: dict[str, Any] = {
        "read_only": True,
        "index_dir": str(index_path.resolve()),
        "qdrant_collection": collection,
        "errors": [],
        "warnings": [],
        "stale_reasons": [],
    }
    errors: list[str] = report["errors"]

    try:
        load_result = load_documents(data_dir, prefer_processed=True)
        if load_result.errors:
            errors.append(f"source_load_errors:{len(load_result.errors)}")
        expected_corpus = corpus_fingerprint(load_result.documents, 3)
        expected_build = build_fingerprint(3)
        parent_chunks = []
        proposed_chunks, chunk_errors = build_chunks(load_result.documents, 3, parent_chunks)
        if chunk_errors:
            errors.append(f"chunk_errors:{len(chunk_errors)}")
        proposed_records = [
            {"chunk_id": chunk.chunk_id, "content": chunk.content, "metadata": chunk.metadata}
            for chunk in proposed_chunks
        ]
        proposed_parents = [
            {"chunk_id": chunk.chunk_id, "content": chunk.content, "metadata": chunk.metadata}
            for chunk in parent_chunks
        ]
        proposed_duplicates = _duplicate_ids([*proposed_records, *proposed_parents])
        if proposed_duplicates:
            errors.append(f"proposed_duplicate_chunk_ids:{len(proposed_duplicates)}")
        report["proposed"] = {
            "corpus_fingerprint": expected_corpus,
            "build_fingerprint": expected_build,
            "index_build_revision": INDEX_BUILD_REVISION,
            "chunk_count": len(proposed_chunks),
            "parent_chunk_count": len(parent_chunks),
            "unique_id_count": len(
                {item["chunk_id"] for item in [*proposed_records, *proposed_parents]}
            ),
        }
    except Exception as exc:
        errors.append(f"proposed_build_failed:{type(exc).__name__}:{exc}")
        expected_corpus = ""
        expected_build = build_fingerprint(3)
        proposed_records = []
        proposed_parents = []

    manifest = None
    chunks = []
    embeddings = None
    try:
        chunks, embeddings, manifest = load_dense_index(index_path)
        records = list(manifest.get("chunks") or [])
        parents = list(manifest.get("parent_chunks") or [])
        duplicates = _duplicate_ids([*records, *parents])
        if duplicates:
            errors.append(f"local_duplicate_chunk_ids:{len(duplicates)}")
        if int(manifest.get("chunk_count", -1)) != len(chunks):
            errors.append("local_manifest_chunk_count_mismatch")
        if int(manifest.get("parent_chunk_count", -1)) != len(parents):
            errors.append("local_manifest_parent_count_mismatch")
        if int(manifest.get("embedding_dimension", -1)) != int(embeddings.shape[1]):
            errors.append("local_embedding_dimension_mismatch")
        report["local"] = {
            "schema_version": manifest.get("schema_version"),
            "corpus_fingerprint": manifest.get("corpus_fingerprint"),
            "build_fingerprint": manifest.get("build_fingerprint"),
            "index_build_revision": manifest.get("index_build_revision"),
            "chunk_count": len(chunks),
            "parent_chunk_count": len(parents),
            "unique_id_count": len(
                {item.get("chunk_id") for item in [*records, *parents] if item.get("chunk_id")}
            ),
            "embedding_dimension": int(embeddings.shape[1]),
            "embeddings_checksum_valid": True,
        }
        stale_reasons = manifest_build_stale_reasons(
            manifest,
            expected_corpus_fingerprint=expected_corpus,
            expected_build_fingerprint=expected_build,
        )
        report["stale_reasons"] = stale_reasons
    except Exception as exc:
        errors.append(f"local_snapshot_invalid:{type(exc).__name__}:{exc}")
        report["local"] = {"valid": False}

    dataset = None
    try:
        dataset = load_json(Path(dataset_path))
        current_golden_errors = validate_dataset(dataset, manifest) if manifest else []
        if current_golden_errors:
            errors.append(f"current_golden_validation_failed:{len(current_golden_errors)}")

        proposed_manifest = {
            "corpus_fingerprint": expected_corpus,
            "chunks": proposed_records,
            "parent_chunks": proposed_parents,
        }
        proposed_golden_errors = validate_dataset(dataset, proposed_manifest)
        if proposed_golden_errors:
            errors.append(f"proposed_golden_validation_failed:{len(proposed_golden_errors)}")
        golden_ids = set((dataset.get("evidence_catalog") or {}).keys())
        proposed_ids = {
            item["chunk_id"] for item in [*proposed_records, *proposed_parents]
        }
        missing_ids = sorted(golden_ids - proposed_ids)
        if missing_ids:
            errors.append(f"proposed_golden_ids_missing:{len(missing_ids)}")
        report["golden"] = {
            "evidence_count": len(golden_ids),
            "missing_in_proposed": missing_ids,
            "current_validation_errors": current_golden_errors,
            "proposed_validation_errors": proposed_golden_errors,
        }
    except Exception as exc:
        errors.append(f"golden_validation_failed:{type(exc).__name__}:{exc}")

    owns_client = qdrant_client is None
    client = qdrant_client or QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
    try:
        if manifest is None or embeddings is None:
            errors.append("qdrant_check_skipped_without_valid_local_snapshot")
        else:
            dense = QdrantDenseSearch(client, collection, None)
            dense.validate_ready(int(embeddings.shape[1]))
            qdrant_count = int(client.count(collection_name=collection, exact=True).count)
            if qdrant_count != len(chunks):
                errors.append("local_qdrant_count_mismatch")
            collection_matches = manifest.get("qdrant_collection") == collection
            if not collection_matches:
                errors.append("local_qdrant_collection_mismatch")
            point_check = _inspect_qdrant_points(
                client,
                collection,
                list(manifest.get("chunks") or []),
            )
            if not point_check["point_ids_match"]:
                errors.append("local_qdrant_point_id_mismatch")
            if not point_check["payloads_match"]:
                errors.append("local_qdrant_payload_mismatch")
            compatible = (
                qdrant_count == len(chunks)
                and collection_matches
                and point_check["compatible"]
            )
            report["qdrant"] = {
                "exists": True,
                "chunk_count": qdrant_count,
                "embedding_dimension": int(embeddings.shape[1]),
                **point_check,
                "compatible": compatible,
                "error_code": None if compatible else ("COLLECTION_MISMATCH" if not collection_matches else "QDRANT_INDEX_MISMATCH"),
            }
            report["qdrant_error_code"] = report["qdrant"]["error_code"]
    except Exception as exc:
        errors.append(f"qdrant_incompatible:{type(exc).__name__}:{exc}")
        report["qdrant"] = {"compatible": False}
        report["qdrant_error_code"] = failure_code(exc, "qdrant")
    finally:
        if owns_client:
            try:
                client.close()
            except Exception:
                errors.append("qdrant_client_close_failed")
                report["qdrant"] = {"compatible": False}
                report["qdrant_error_code"] = "QDRANT_INSPECTION_FAILED"

    snapshot_bytes = _directory_size(index_path)
    embeddings_bytes = (index_path / "embeddings.npy").stat().st_size if (index_path / "embeddings.npy").exists() else 0
    staging_bytes = snapshot_bytes
    backup_bytes = snapshot_bytes
    required_bytes = staging_bytes + backup_bytes + embeddings_bytes
    usage = disk_usage(_existing_ancestor(index_path))
    free_bytes = int(usage.free)
    if free_bytes < required_bytes:
        errors.append("insufficient_disk_space")
    report["disk"] = {
        "free_bytes": free_bytes,
        "staging_estimate_bytes": staging_bytes,
        "backup_estimate_bytes": backup_bytes,
        "embedding_working_space_bytes": embeddings_bytes,
        "required_bytes": required_bytes,
        "sufficient": free_bytes >= required_bytes,
    }

    report["rebuild_required"] = bool(report["stale_reasons"])
    report["status"] = "blocked" if errors else (
        "ready_for_rebuild" if report["rebuild_required"] else "current"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", default=settings.INDEX_DIR)
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--collection", default=settings.QDRANT_COLLECTION)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    args = parser.parse_args()

    report = run_preflight(
        index_dir=args.index_dir,
        data_dir=args.data_dir,
        collection=args.collection,
        dataset_path=args.dataset,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if report["status"] == "blocked" else 0


if __name__ == "__main__":
    raise SystemExit(main())
