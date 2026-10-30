"""Exercise the ingest CLI without loading models, writing indexes or using Qdrant."""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from scripts import ingest


@pytest.fixture
def protected(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    active = root / "data/index/module3-active"
    active.mkdir(parents=True)
    (active / "manifest.json").write_text('{"module": 3}', encoding="utf-8")
    (active / "embeddings.npy").write_bytes(b"protected vectors")
    monkeypatch.setattr(ingest, "PROJECT_ROOT", root)
    monkeypatch.setattr(ingest, "LOCAL_INDEX_DIR", "data/index/module3-active")
    monkeypatch.setattr(ingest.settings, "INDEX_DIR", str(active))
    return active


@pytest.fixture
def boundaries(monkeypatch):
    """Any work past argument validation must be visible and must fail closed."""
    mocks = {}
    for name in (
        "load_documents", "build_chunks", "Embedder", "embed_chunks", "save_dense_index",
        "load_reusable_snapshot", "QdrantClient", "build_hybrid_indexes",
        "validate_unchanged_hybrid_index",
    ):
        mocks[name] = Mock(side_effect=AssertionError(f"Unexpected ingestion work: {name}"))
        monkeypatch.setattr(ingest, name, mocks[name])
    return mocks


def reject(monkeypatch, capsys, boundaries, argv, message):
    monkeypatch.setattr(sys, "argv", ["ingest.py", *map(str, argv)])
    with pytest.raises(SystemExit) as exc:
        ingest.main()
    assert exc.value.code == 2
    assert message in capsys.readouterr().err
    for mock in boundaries.values():
        mock.assert_not_called()


@pytest.mark.parametrize("argv", [[], ["--force"], ["--module", "1"], ["--module", "2"],
                                  ["--module", "3"], ["--index-dir", "candidate"]])
def test_selection_is_explicit(monkeypatch, capsys, boundaries, argv):
    reject(monkeypatch, capsys, boundaries, argv, "required")


@pytest.mark.parametrize("module", [1, 2, 3])
@pytest.mark.parametrize("force", [False, True])
@pytest.mark.parametrize("alias", ["absolute", "relative", "dotdot", "child", "parent", "case", "extended", "extended-child"])
def test_protected_paths_fail_before_ingestion(
    protected, monkeypatch, capsys, boundaries, module, force, alias,
):
    monkeypatch.chdir(protected.parents[2])
    destinations = {
        "absolute": protected,
        "relative": protected.relative_to(Path.cwd()),
        "dotdot": protected / "../module3-active",
        "child": protected / "candidate",
        "parent": protected.parent,
        "case": str(protected).upper() if os.name == "nt" else str(protected),
        "extended": "\\\\?\\" + str(protected) if os.name == "nt" else str(protected),
        "extended-child": "\\\\?\\" + str(protected / "candidate") if os.name == "nt" else str(protected / "candidate"),
    }
    before = {p.name: p.read_bytes() for p in protected.iterdir()}
    argv = ["--module", module, "--index-dir", destinations[alias]]
    reject(monkeypatch, capsys, boundaries, argv + (["--force"] if force else []), "Protected active")
    assert {p.name: p.read_bytes() for p in protected.iterdir()} == before


def directory_alias(alias, target):
    try:
        alias.symlink_to(target, target_is_directory=True)
    except OSError:
        if os.name != "nt":
            raise
        # Directory junctions work on Windows without symlink privileges.
        result = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(alias), str(target)],
            capture_output=True, text=True, check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("module", [1, 2, 3])
def test_directory_alias_is_protected(protected, tmp_path, monkeypatch, capsys, boundaries, module):
    alias = tmp_path / "alias"
    directory_alias(alias, protected)
    reject(monkeypatch, capsys, boundaries,
           ["--module", module, "--index-dir", alias, "--force"], "Protected active")


@pytest.mark.parametrize("artifact", ["manifest.json", "embeddings.npy", "manifest.json.tmp", ".module3-staging", ".module3-staging/manifest.json.tmp"])
def test_artifact_alias_is_protected(protected, tmp_path, monkeypatch, capsys, boundaries, artifact):
    output = tmp_path / "candidate"
    output.mkdir()
    if artifact == ".module3-staging":
        directory_alias(output / artifact, protected)
    else:
        # Hard links are a separate filesystem identity check from resolve().
        (output / artifact).parent.mkdir(parents=True, exist_ok=True)
        (output / artifact).hardlink_to(protected / "manifest.json")
    reject(monkeypatch, capsys, boundaries,
           ["--module", "1", "--index-dir", output, "--force"], "aliases protected")


def test_link_to_protected_staging_is_rejected(protected, tmp_path, monkeypatch, capsys, boundaries):
    staging = protected / ".module3-staging"
    staging.mkdir()
    (staging / "manifest.json").write_text('{"module": 3}', encoding="utf-8")
    output = tmp_path / "candidate"
    output.mkdir()
    (output / "manifest.json.tmp").hardlink_to(staging / "manifest.json")
    reject(monkeypatch, capsys, boundaries,
           ["--module", "2", "--index-dir", output, "--force"], "aliases protected")


def test_pinned_path_stays_protected_when_runtime_settings_change(
    protected, tmp_path, monkeypatch, capsys, boundaries,
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ingest.settings, "LOCAL_DEV_PROFILE", False)
    monkeypatch.setattr(ingest.settings, "INDEX_DIR", str(tmp_path / "other-active"))
    reject(monkeypatch, capsys, boundaries,
           ["--module", "2", "--index-dir", protected, "--force"], "Protected active")


@pytest.mark.parametrize("module", [1, 2])
def test_other_module3_snapshot_cannot_be_overwritten(
    protected, tmp_path, monkeypatch, capsys, boundaries, module,
):
    output = tmp_path / "another-snapshot"
    output.mkdir()
    (output / "manifest.json").write_text('{"module": 3}', encoding="utf-8")
    reject(monkeypatch, capsys, boundaries,
           ["--module", module, "--index-dir", output, "--force"], "different module")


@pytest.mark.parametrize("collection", [None, "", "  ", ingest.LOCAL_COLLECTION, "runtime-active", " candidate "])
def test_module3_requires_separate_explicit_collection(
    protected, tmp_path, monkeypatch, capsys, boundaries, collection,
):
    monkeypatch.setattr(ingest.settings, "QDRANT_COLLECTION", "runtime-active")
    argv = ["--module", "3", "--index-dir", tmp_path / "new-index", "--force"]
    if collection is not None:
        argv.extend(["--collection", collection])
    reject(monkeypatch, capsys, boundaries, argv, "error:")
    assert not (tmp_path / "new-index").exists()


@pytest.mark.parametrize("destination", ["", "   "])
def test_empty_output_is_rejected(protected, monkeypatch, capsys, boundaries, destination):
    reject(monkeypatch, capsys, boundaries,
           ["--module", "1", "--index-dir", destination], "explicit safe output")


def test_dense_module_rejects_collection(protected, tmp_path, monkeypatch, capsys, boundaries):
    reject(monkeypatch, capsys, boundaries,
           ["--module", "2", "--index-dir", tmp_path / "safe", "--collection", "candidate"],
           "only supported for Module 3")


@pytest.mark.parametrize("module", [1, 2, 3])
def test_safe_existing_candidate_can_be_reused(
    protected, tmp_path, monkeypatch, capsys, boundaries, module,
):
    output = tmp_path / "candidate"
    output.mkdir()
    manifest = {"module": module, "qdrant_collection": "migration-candidate", "chunk_count": 1}
    (output / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    boundaries["load_documents"].side_effect = None
    boundaries["load_documents"].return_value = SimpleNamespace(
        documents=[], source_dir="synthetic", errors=[], skipped_duplicates=[],
    )
    boundaries["load_reusable_snapshot"].side_effect = None
    boundaries["load_reusable_snapshot"].return_value = ([], np.ones((1, 2)), manifest)
    boundaries["validate_unchanged_hybrid_index"].side_effect = None
    boundaries["validate_unchanged_hybrid_index"].return_value = (manifest, 1)
    argv = ["ingest.py", "--module", str(module), "--index-dir", str(output)]
    if module == 3:
        argv.extend(["--collection", "migration-candidate"])
    monkeypatch.setattr(sys, "argv", argv)
    assert ingest.main() == 0
    assert json.loads(capsys.readouterr().out)["status"] == "unchanged"
    for name in ("Embedder", "embed_chunks", "save_dense_index", "QdrantClient", "build_hybrid_indexes"):
        boundaries[name].assert_not_called()
    if module == 3:
        boundaries["validate_unchanged_hybrid_index"].assert_called_once_with(
            output.resolve(), collection="migration-candidate",
        )


def test_candidate_collection_mismatch_fails_before_work(
    protected, tmp_path, monkeypatch, capsys, boundaries,
):
    output = tmp_path / "candidate"
    output.mkdir()
    (output / "manifest.json").write_text(
        '{"module": 3, "qdrant_collection": "other-candidate"}', encoding="utf-8",
    )
    reject(monkeypatch, capsys, boundaries,
           ["--module", "3", "--index-dir", output, "--collection", "migration-candidate", "--force"],
           "different Qdrant collection")


@pytest.mark.parametrize("module", [1, 2, 3])
@pytest.mark.parametrize("force", [False, True])
def test_safe_output_routes_only_to_selected_destination(
    protected, tmp_path, monkeypatch, capsys, boundaries, module, force,
):
    output = tmp_path / f"module{module}-candidate"
    chunk = SimpleNamespace(chunk_id="test-chunk", content="synthetic", metadata={"doc_id": "test-doc"})
    vectors = np.ones((1, 2), dtype=np.float32)
    load_result = SimpleNamespace(documents=[], source_dir="synthetic", errors=[], skipped_duplicates=[])
    boundaries["load_documents"].side_effect = None
    boundaries["load_documents"].return_value = load_result
    boundaries["load_reusable_snapshot"].side_effect = None
    boundaries["load_reusable_snapshot"].return_value = None
    boundaries["build_chunks"].side_effect = None
    boundaries["build_chunks"].return_value = ([chunk], [])
    boundaries["Embedder"].side_effect = None
    boundaries["embed_chunks"].side_effect = None
    boundaries["embed_chunks"].return_value = ([chunk], vectors, [])
    boundaries["save_dense_index"].side_effect = lambda path, chunks, vectors, metadata: {
        **metadata, "chunk_count": len(chunks), "embedding_dimension": 2,
    }
    boundaries["build_hybrid_indexes"].side_effect = None
    boundaries["build_hybrid_indexes"].return_value = (SimpleNamespace(chunks=[chunk]), 1)
    argv = ["ingest.py", "--module", str(module), "--index-dir", str(output)]
    if module == 3:
        argv.extend(["--collection", "migration-candidate"])
    if force:
        argv.append("--force")
    monkeypatch.setattr(sys, "argv", argv)
    assert ingest.main() == 0
    summary = json.loads("{" + capsys.readouterr().out.split("{", 1)[1])
    assert summary["index_dir"] == str(output.resolve())
    writes = boundaries["save_dense_index"].call_args_list
    assert writes[-1].args[0] == output.resolve()
    assert writes[-1].args[3]["module"] == module
    if module == 3:
        assert writes[0].args[0] == output.resolve() / ".module3-staging"
        assert writes[-1].args[3]["qdrant_collection"] == "migration-candidate"
        assert boundaries["build_hybrid_indexes"].call_args.kwargs["collection"] == "migration-candidate"
    else:
        assert len(writes) == 1
        boundaries["build_hybrid_indexes"].assert_not_called()
    boundaries["QdrantClient"].assert_not_called()
    assert not output.exists()


def test_hybrid_helpers_use_selected_collection(monkeypatch):
    client = Mock()
    client.count.return_value.count = 1
    client.collection_exists.return_value = True
    dense = Mock(qdrant_client=client)
    dense._point_id.return_value = "stale-point"
    dense_factory = Mock(return_value=dense)
    monkeypatch.setattr(ingest, "QdrantClient", Mock(return_value=client))
    monkeypatch.setattr(ingest, "QdrantDenseSearch", dense_factory)
    monkeypatch.setattr(ingest, "BM25Search", Mock(return_value=SimpleNamespace(
        chunks=["synthetic"], index=Mock(),
    )))
    chunks = [SimpleNamespace(chunk_id="synthetic")]
    vectors = np.ones((1, 2), dtype=np.float32)
    _, count = ingest.build_hybrid_indexes(
        chunks, vectors, None, {"stale"}, collection="migration-candidate",
    )
    assert count == 1
    assert dense_factory.call_args.args[1] == "migration-candidate"
    client.delete.assert_called_once_with(
        collection_name="migration-candidate", points_selector=["stale-point"], wait=True,
    )
    client.count.assert_called_with(collection_name="migration-candidate", exact=True)

    manifest = {"module": 3}
    monkeypatch.setattr(ingest, "load_dense_index", Mock(return_value=(chunks, vectors, manifest)))
    assert ingest.validate_unchanged_hybrid_index("synthetic", collection="migration-candidate") == (manifest, 1)
    client.collection_exists.assert_called_once_with("migration-candidate")
    client.count.assert_called_with(collection_name="migration-candidate", exact=True)
