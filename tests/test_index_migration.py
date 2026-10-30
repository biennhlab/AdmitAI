import json
import uuid
from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from qdrant_client import models

from scripts.ingest import (
    build_chunks,
    build_fingerprint,
    corpus_fingerprint,
    load_reusable_snapshot,
)
from scripts.preflight_index_migration import run_preflight
from scripts.validate_golden_dataset import DEFAULT_DATASET, FALLBACK_ANSWER, load_json
from src.ingestion import load_documents
from src.retrieval import save_dense_index
from src.retrieval import QdrantDenseSearch
from src.retrieval.index_version import INDEX_BUILD_REVISION, manifest_build_stale_reasons


class ReadOnlyQdrant:
    def __init__(self, chunks, dimension: int, *, count: int | None = None):
        self.point_count = len(chunks) if count is None else count
        self.dimension = dimension
        self.calls = []
        self.points = []
        for chunk in chunks:
            metadata = json.loads(json.dumps(chunk.metadata, default=str))
            self.points.append(
                SimpleNamespace(
                    id=QdrantDenseSearch._point_id(chunk.chunk_id),
                    payload={
                        **metadata,
                        "chunk_id": chunk.chunk_id,
                        "content": chunk.content,
                        "metadata": metadata,
                    },
                )
            )

    def collection_exists(self, collection):
        self.calls.append(("collection_exists", collection))
        return True

    def get_collection(self, collection):
        self.calls.append(("get_collection", collection))
        vectors = SimpleNamespace(size=self.dimension, distance=models.Distance.COSINE)
        return SimpleNamespace(config=SimpleNamespace(params=SimpleNamespace(vectors=vectors)))

    def count(self, collection_name, exact):
        self.calls.append(("count", collection_name, exact))
        return SimpleNamespace(count=self.point_count)

    def scroll(
        self,
        collection_name,
        limit,
        offset,
        with_payload,
        with_vectors,
    ):
        self.calls.append(
            (
                "scroll",
                collection_name,
                limit,
                offset,
                with_payload,
                with_vectors,
            )
        )
        return self.points, None


class OfflineQdrant:
    def __init__(self):
        self.calls = []

    def collection_exists(self, collection):
        self.calls.append(("collection_exists", collection))
        raise ConnectionError("offline")


def _write_source(root: Path) -> None:
    processed = root / "processed" / "ptit"
    processed.mkdir(parents=True)
    (processed / "source.md").write_text(
        '---\ndoc_id: "migration-doc"\ntitle: "Migration evidence"\n'
        'source_url: "https://example.test/evidence"\npage_number: 1\n'
        'section: "Admissions"\nversion: "1"\n---\n'
        "# Admissions\nPTIT migration evidence remains attributable.",
        encoding="utf-8",
    )


def _dataset_for(chunk, fingerprint: str) -> dict:
    catalog = {
        chunk.chunk_id: {
            "chunk_id": chunk.chunk_id,
            "doc_id": chunk.metadata["doc_id"],
            "title": chunk.metadata["title"],
            "page": chunk.metadata["page_number"],
            "source_url": chunk.metadata["source_url"],
            "quote": chunk.content,
        }
    }
    cases = [
        {
            "id": "migration-evidence",
            "category": "migration",
            "difficulty": "single_hop",
            "question": "Which migration evidence is indexed?",
            "expected_route": "general",
            "ground_truth": "PTIT migration evidence remains attributable.",
            "evidence_ids": [chunk.chunk_id],
            "required_terms": ["migration evidence"],
            "tags": ["migration"],
        }
    ]
    for index in range(29):
        cases.append(
            {
                "id": f"negative-{index}",
                "category": "negative",
                "difficulty": "negative",
                "question": f"Unsupported migration question {index}?",
                "expected_route": "out_of_scope",
                "ground_truth": FALLBACK_ANSWER,
                "evidence_ids": [],
                "required_terms": [],
                "tags": ["negative"],
            }
        )
    return {
        "schema_version": 1,
        "corpus": {"fingerprint": fingerprint},
        "evidence_catalog": catalog,
        "cases": cases,
    }


def _build_snapshot(tmp_path: Path):
    data_dir = tmp_path / "data"
    _write_source(data_dir)
    load_result = load_documents(data_dir / "processed")
    parents = []
    chunks, errors = build_chunks(load_result.documents, 3, parents)
    assert errors == []
    fingerprint = corpus_fingerprint(load_result.documents, 3)
    build = build_fingerprint(3)
    embeddings = np.ones((len(chunks), 4), dtype=np.float32)
    index_dir = tmp_path / "index"
    save_dense_index(
        index_dir,
        chunks,
        embeddings,
        {
            "module": 3,
            "document_count": 1,
            "parent_chunk_count": len(parents),
            "parent_chunks": [
                {"chunk_id": item.chunk_id, "content": item.content, "metadata": item.metadata}
                for item in parents
            ],
            "embedding_model": "BAAI/bge-m3",
            "qdrant_collection": "migration_test",
            "corpus_fingerprint": fingerprint,
            "index_build_revision": INDEX_BUILD_REVISION,
            "build_fingerprint": build,
        },
    )
    dataset_path = tmp_path / "golden.json"
    dataset_path.write_text(
        json.dumps(_dataset_for(chunks[0], fingerprint), ensure_ascii=False),
        encoding="utf-8",
    )
    return data_dir / "processed", index_dir, dataset_path, chunks, fingerprint, build


def test_manifest_staleness_distinguishes_revision_corpus_and_current_build():
    current = {
        "index_build_revision": INDEX_BUILD_REVISION,
        "build_fingerprint": "build-a",
        "corpus_fingerprint": "corpus-a",
    }
    assert manifest_build_stale_reasons(
        current,
        expected_corpus_fingerprint="corpus-a",
        expected_build_fingerprint="build-a",
    ) == []

    assert "index_build_revision_missing" in manifest_build_stale_reasons(
        {"build_fingerprint": "build-a", "corpus_fingerprint": "corpus-a"},
        expected_corpus_fingerprint="corpus-a",
        expected_build_fingerprint="build-a",
    )
    assert "index_build_revision_changed" in manifest_build_stale_reasons(
        {**current, "index_build_revision": "next-revision"},
        expected_corpus_fingerprint="corpus-a",
        expected_build_fingerprint="build-a",
    )
    assert "corpus_fingerprint_changed" in manifest_build_stale_reasons(
        current,
        expected_corpus_fingerprint="corpus-b",
        expected_build_fingerprint="build-a",
    )


def test_metadata_only_corpus_change_invalidates_snapshot_fingerprint():
    original = SimpleNamespace(
        doc_id="doc-1",
        metadata={"source_url": "https://example.test/v1", "content_hash": "same"},
    )
    changed = SimpleNamespace(
        doc_id="doc-1",
        metadata={"source_url": "https://example.test/v2", "content_hash": "same"},
    )

    assert corpus_fingerprint([original], 3) != corpus_fingerprint([changed], 3)


def test_staging_wrong_revision_is_not_reused_and_corrupt_vectors_fail_safe(tmp_path):
    _, index_dir, _, _, fingerprint, build = _build_snapshot(tmp_path)
    manifest_path = index_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["index_build_revision"] = "old-revision"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    assert load_reusable_snapshot(
        index_dir,
        expected_corpus_fingerprint=fingerprint,
        expected_build_fingerprint=build,
    ) is None

    manifest["index_build_revision"] = INDEX_BUILD_REVISION
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with (index_dir / "embeddings.npy").open("ab") as handle:
        handle.write(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        load_reusable_snapshot(
            index_dir,
            expected_corpus_fingerprint=fingerprint,
            expected_build_fingerprint=build,
        )

    manifest_path.write_text("{not-json", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        load_reusable_snapshot(
            index_dir,
            expected_corpus_fingerprint=fingerprint,
            expected_build_fingerprint=build,
        )


def test_read_only_preflight_preserves_golden_ids_and_never_writes_qdrant(tmp_path):
    data_dir, index_dir, dataset_path, chunks, _, _ = _build_snapshot(tmp_path)
    qdrant = ReadOnlyQdrant(chunks, 4)
    DiskUsage = namedtuple("DiskUsage", "total used free")

    report = run_preflight(
        index_dir=index_dir,
        data_dir=data_dir,
        collection="migration_test",
        dataset_path=dataset_path,
        qdrant_client=qdrant,
        disk_usage=lambda _: DiskUsage(10**9, 0, 10**9),
    )

    assert report["status"] == "current"
    assert report["read_only"] is True
    assert report["golden"]["missing_in_proposed"] == []
    assert report["errors"] == []
    assert {call[0] for call in qdrant.calls} == {
        "collection_exists",
        "get_collection",
        "count",
        "scroll",
    }


def test_preflight_blocks_local_qdrant_count_mismatch(tmp_path):
    data_dir, index_dir, dataset_path, chunks, _, _ = _build_snapshot(tmp_path)
    qdrant = ReadOnlyQdrant(chunks, 4, count=len(chunks) + 1)
    DiskUsage = namedtuple("DiskUsage", "total used free")

    report = run_preflight(
        index_dir=index_dir,
        data_dir=data_dir,
        collection="migration_test",
        dataset_path=dataset_path,
        qdrant_client=qdrant,
        disk_usage=lambda _: DiskUsage(10**9, 0, 10**9),
    )

    assert report["status"] == "blocked"
    assert "local_qdrant_count_mismatch" in report["errors"]


def test_preflight_blocks_same_count_with_different_qdrant_point_ids(tmp_path):
    data_dir, index_dir, dataset_path, chunks, _, _ = _build_snapshot(tmp_path)
    qdrant = ReadOnlyQdrant(chunks, 4)
    qdrant.points[0].id = str(uuid.uuid4())
    DiskUsage = namedtuple("DiskUsage", "total used free")

    report = run_preflight(
        index_dir=index_dir,
        data_dir=data_dir,
        collection="migration_test",
        dataset_path=dataset_path,
        qdrant_client=qdrant,
        disk_usage=lambda _: DiskUsage(10**9, 0, 10**9),
    )

    assert report["status"] == "blocked"
    assert report["qdrant"]["chunk_count"] == len(chunks)
    assert report["qdrant"]["point_ids_match"] is False
    assert "local_qdrant_point_id_mismatch" in report["errors"]


def test_preflight_blocks_matching_ids_with_conflicting_source_payload(tmp_path):
    data_dir, index_dir, dataset_path, chunks, _, _ = _build_snapshot(tmp_path)
    qdrant = ReadOnlyQdrant(chunks, 4)
    qdrant.points[0].payload["source_url"] = "https://wrong.example/source"
    qdrant.points[0].payload["metadata"]["source_url"] = "https://wrong.example/source"
    DiskUsage = namedtuple("DiskUsage", "total used free")

    report = run_preflight(
        index_dir=index_dir,
        data_dir=data_dir,
        collection="migration_test",
        dataset_path=dataset_path,
        qdrant_client=qdrant,
        disk_usage=lambda _: DiskUsage(10**9, 0, 10**9),
    )

    assert report["status"] == "blocked"
    assert report["qdrant"]["point_ids_match"] is True
    assert report["qdrant"]["payloads_match"] is False
    assert "local_qdrant_payload_mismatch" in report["errors"]


def test_preflight_blocks_when_qdrant_is_offline(tmp_path):
    data_dir, index_dir, dataset_path, _, _, _ = _build_snapshot(tmp_path)
    qdrant = OfflineQdrant()
    DiskUsage = namedtuple("DiskUsage", "total used free")

    report = run_preflight(
        index_dir=index_dir,
        data_dir=data_dir,
        collection="migration_test",
        dataset_path=dataset_path,
        qdrant_client=qdrant,
        disk_usage=lambda _: DiskUsage(10**9, 0, 10**9),
    )

    assert report["status"] == "blocked"
    assert report["qdrant"] == {"compatible": False}
    assert any(error.startswith("qdrant_incompatible:RuntimeError:") for error in report["errors"])
    assert {call[0] for call in qdrant.calls} == {"collection_exists"}


def test_phase_a_dry_run_counts_and_golden_ids_are_preserved():
    load_result = load_documents("data/processed")
    parents = []
    chunks, errors = build_chunks(load_result.documents, 3, parents)
    golden = load_json(DEFAULT_DATASET)
    proposed_ids = {chunk.chunk_id for chunk in [*chunks, *parents]}

    assert errors == []
    assert len(chunks) == 3185
    assert len(parents) == 1119
    assert set(golden["evidence_catalog"]) <= proposed_ids
    assert corpus_fingerprint(load_result.documents, 3) == golden["corpus"]["fingerprint"]


@pytest.mark.parametrize("failure,code", [
    ("duplicate", "QDRANT_INDEX_MISMATCH"), ("missing", "QDRANT_INDEX_MISMATCH"),
    ("content", "QDRANT_INDEX_MISMATCH"), ("metadata", "QDRANT_INDEX_MISMATCH"),
    ("pagination-error", "QDRANT_INSPECTION_FAILED"), ("repeated-offset", "QDRANT_INSPECTION_FAILED"),
    ("empty-continuation", "QDRANT_INSPECTION_FAILED"),
])
def test_preflight_fails_closed_on_all_incomplete_or_conflicting_point_inspections(tmp_path, failure, code):
    from unittest.mock import Mock

    data_dir, index_dir, dataset, chunks, _, _ = _build_snapshot(tmp_path)
    client = ReadOnlyQdrant(chunks, 4)
    if failure == "duplicate":
        client.points.append(client.points[0])
    elif failure == "missing":
        client.points.clear()
    elif failure == "content":
        client.points[0].payload["content"] = "Unrelated content"
    elif failure == "metadata":
        client.points[0].payload["metadata"]["source_url"] = "https://wrong.test/"
    elif failure == "pagination-error":
        client.scroll = Mock(side_effect=[(client.points, str(uuid.uuid4())), TimeoutError("offline")])
    elif failure == "repeated-offset":
        cursor = str(uuid.uuid4())
        client.scroll = Mock(side_effect=[(client.points, cursor), (client.points, cursor)])
    else:
        client.scroll = Mock(return_value=([], str(uuid.uuid4())))
    Disk = namedtuple("Disk", "total used free")
    report = run_preflight(index_dir=index_dir, data_dir=data_dir, collection="migration_test", dataset_path=dataset,
                           qdrant_client=client, disk_usage=lambda _: Disk(10**9, 0, 10**9))
    assert report["status"] == "blocked" and report["qdrant"]["compatible"] is False
    assert report["qdrant_error_code"] == code


@pytest.mark.parametrize("failure", [False, True])
def test_preflight_owns_and_closes_its_client_on_success_and_inspection_failure(tmp_path, failure):
    from unittest.mock import Mock, patch

    data_dir, index_dir, dataset, chunks, _, _ = _build_snapshot(tmp_path)
    client = ReadOnlyQdrant(chunks, 4)
    client.close = Mock()
    if failure:
        client.scroll = Mock(side_effect=TimeoutError("offline"))
    Disk = namedtuple("Disk", "total used free")
    with patch("scripts.preflight_index_migration.QdrantClient", return_value=client):
        report = run_preflight(index_dir=index_dir, data_dir=data_dir, collection="migration_test", dataset_path=dataset,
                               disk_usage=lambda _: Disk(10**9, 0, 10**9))
    assert report["status"] == ("blocked" if failure else "current")
    client.close.assert_called_once()
