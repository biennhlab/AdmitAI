from unittest.mock import MagicMock, patch
import json

import pytest

from scripts import local_runtime
from urllib.error import URLError


@pytest.mark.parametrize("memory", [
    {"available_physical_gib": 0.01, "commit_headroom_gib": 8},
    {"available_physical_gib": 8, "commit_headroom_gib": 0.01},
    {"available_physical_gib": 0.01, "commit_headroom_gib": 0.01},
    None,
])
def test_preflight_records_low_or_unknown_memory_without_blocking(tmp_path, memory):
    cache = tmp_path / "models/huggingface/hub/models--BAAI--bge-m3/snapshots/cached"
    cache.mkdir(parents=True)
    for name in ("config.json", "modules.json", "pytorch_model.bin"):
        (cache / name).write_text("{}")
    security = tmp_path / "security.json"
    security.write_text(json.dumps({"confirmed": True}))
    with (
        patch.object(local_runtime, "ROOT", tmp_path),
        patch.object(local_runtime, "SECURITY", security),
        patch.object(local_runtime, "application_revision", return_value="a" * 64),
        patch.object(local_runtime, "memory_status", return_value=memory),
        patch.object(local_runtime, "validate_snapshot", return_value={"chunks": 3185}),
        patch.object(local_runtime, "validate_collection_integrity", return_value={"compatible": True}),
        patch.object(local_runtime, "api_listeners", return_value=[]),
        patch.object(local_runtime, "request_json", side_effect=[
            {"result": {"status": "green", "points_count": 3185}}, URLError("no API"),
        ]),
    ):
        report = local_runtime.inspect()
    assert report["memory"] == memory
    assert report["blockers"] == []


def test_invalid_preflight_cannot_stop_or_start_an_api(tmp_path):
    with (
        patch.object(local_runtime, "ARTIFACTS", tmp_path),
        patch.object(local_runtime, "SECURITY", tmp_path / "security.json"),
        patch("sys.argv", ["local_runtime.py", "start"]),
        patch.object(local_runtime, "inspect", return_value={"blockers": ["MODEL_UNAVAILABLE"], "memory": {}}),
        patch.object(local_runtime, "stop_verified") as stop,
        patch.object(local_runtime, "launch") as launch,
    ):
        assert local_runtime.main() == 1
    stop.assert_not_called()
    launch.assert_not_called()


def test_stop_rejects_reused_pid_before_termination():
    with (
        patch.object(local_runtime, "process_identity", return_value={"created": 999, "verified_uvicorn": True, "working_directory_matches": True}),
        patch("psutil.Process") as process,
    ):
        with pytest.raises(RuntimeError, match="unverified"):
            local_runtime.stop_verified({"pid": 12, "created": 100})
    process.assert_not_called()


@pytest.mark.parametrize("memory", [
    {"available_physical_gib": 0.01, "commit_headroom_gib": 0.01}, None,
])
def test_start_records_memory_after_stopping_verified_api_and_launches_once(tmp_path, memory):
    existing = {"pid": 10, "created": 100, "verified_uvicorn": True, "working_directory_matches": True, "workers": []}
    events = []
    with (
        patch.object(local_runtime, "ARTIFACTS", tmp_path),
        patch.object(local_runtime, "SECURITY", tmp_path / "security.json"),
        patch("sys.argv", ["local_runtime.py", "start"]),
        patch.object(local_runtime, "inspect", return_value={"blockers": [], "api_listeners": [existing], "memory": memory}),
        patch.object(local_runtime, "api_listeners", return_value=[]),
        patch.object(local_runtime, "stop_verified", side_effect=lambda identity: events.append(("stop", identity["pid"]))),
        patch.object(local_runtime, "memory_status", side_effect=lambda: events.append(("measure", None)) or memory),
        patch.object(local_runtime, "launch", side_effect=lambda: events.append(("launch", None)) or {"pid": 20}) as launch,
        patch.object(local_runtime, "await_ready", return_value={"status": "healthy"}),
    ):
        assert local_runtime.main() == 0
    assert events == [("stop", 10), ("measure", None), ("launch", None), ("measure", None)]
    launch.assert_called_once_with()
    assert json.loads((tmp_path / "preflight.json").read_text())["memory"] == memory
    result = json.loads((tmp_path / "start-result.json").read_text())
    assert result["status"] == "healthy"
    assert result["memory_after_stop"] == memory
    assert result["memory"] == memory


@pytest.mark.parametrize("code", ["MODEL_UNAVAILABLE", "INSUFFICIENT_MEMORY"])
def test_failed_start_stops_only_candidate_and_never_launches_another_release(tmp_path, code):
    candidate = {"pid": 20}
    with (
        patch.object(local_runtime, "ARTIFACTS", tmp_path),
        patch.object(local_runtime, "SECURITY", tmp_path / "security.json"),
        patch("sys.argv", ["local_runtime.py", "start"]),
        patch.object(local_runtime, "inspect", return_value={"blockers": [], "api_listeners": []}),
        patch.object(local_runtime, "api_listeners", return_value=[]),
        patch.object(local_runtime, "memory_status", return_value={"available_physical_gib": 8, "commit_headroom_gib": 8}),
        patch.object(local_runtime, "stop_verified") as stop,
        patch.object(local_runtime, "launch", return_value=candidate) as launch,
        patch.object(local_runtime, "await_ready", side_effect=RuntimeError(f"STARTUP_BLOCKED:{code}")),
    ):
        with pytest.raises(RuntimeError, match=code):
            local_runtime.main()
    launch.assert_called_once_with()
    stop.assert_called_once_with(candidate)
    result = json.loads((tmp_path / "start-result.json").read_text())
    assert result["port_free"] is True
    assert result["cleanup_pending"] is True
    assert result["blocker"] == f"STARTUP_BLOCKED:{code}"


def test_launch_pins_current_source_and_lite_profile_without_exposing_credentials(tmp_path):
    process = MagicMock(pid=42)
    with (
        patch.object(local_runtime, "ARTIFACTS", tmp_path),
        patch.object(local_runtime, "STATE", tmp_path / "state.json"),
        patch.object(local_runtime, "application_revision", return_value="a" * 64),
        patch.object(local_runtime, "launched_worker_identity", return_value={"pid": 42}),
        patch("dotenv.dotenv_values", return_value={"LLM_API_KEY": "test-only-secret", "RERANK_BACKEND": "neural", "INDEX_DIR": "obsolete"}),
        patch.dict(local_runtime.os.environ, {}, clear=True),
        patch.object(local_runtime.subprocess, "Popen", return_value=process) as spawn,
        patch.object(local_runtime.time, "sleep"),
    ):
        local_runtime.launch()
    arguments = spawn.call_args.args[0]
    assert "--reload" not in arguments
    assert arguments[arguments.index("--workers") + 1] == "1"
    assert arguments[arguments.index("--app-dir") + 1] == str(local_runtime.ROOT)
    assert spawn.call_args.kwargs["cwd"] == local_runtime.ROOT
    env = spawn.call_args.kwargs["env"]
    assert env["INDEX_DIR"] == local_runtime.LOCAL_INDEX_DIR
    assert env["QDRANT_COLLECTION"] == local_runtime.LOCAL_COLLECTION
    assert env["LOCAL_DEV_PROFILE"] == "true"
    assert env["RERANK_BACKEND"] == "lite"
    assert env["MODELS_LOCAL_ONLY"] == "true"
    assert env["LLM_API_KEY"] == "test-only-secret"
    assert "test-only-secret" not in (tmp_path / "state.json").read_text()


def test_launch_tracks_verified_venv_worker_instead_of_redirector():
    args = ["venv/python.exe", "-m", "uvicorn", "api.main:app", "--workers", "1"]
    launcher = MagicMock(pid=42)
    launcher.cmdline.return_value = args
    worker = MagicMock(pid=43)
    worker.name.return_value = "python.exe"
    worker.ppid.return_value = 42
    worker.cmdline.return_value = args
    launcher.children.return_value = [worker]
    records = {
        42: {"pid": 42, "created": 100, "verified_uvicorn": True, "working_directory_matches": True, "reloader": False},
        43: {"pid": 43, "created": 101, "verified_uvicorn": True, "working_directory_matches": True, "reloader": False},
    }
    with patch("psutil.Process", return_value=launcher), patch.object(local_runtime, "process_identity", side_effect=lambda pid: records[pid].copy()):
        identity = local_runtime.launched_worker_identity(MagicMock(pid=42))
    assert identity["pid"] == 43
    assert identity["created"] == 101
    assert identity["launcher_pid"] == 42
    assert identity["launcher_created"] == 100


def test_launch_without_redirector_keeps_original_pid():
    launcher = MagicMock(pid=42)
    launcher.children.return_value = []
    identity = {"pid": 42, "created": 100, "verified_uvicorn": True, "working_directory_matches": True, "reloader": False}
    with patch("psutil.Process", return_value=launcher), patch.object(local_runtime, "process_identity", return_value=identity):
        assert local_runtime.launched_worker_identity(MagicMock(pid=42)) == identity


@pytest.mark.parametrize("mismatch", ["arguments", "workspace", "reloader", "parent", "created", "multiple", "unverified"])
def test_launch_rejects_unproven_or_ambiguous_redirected_workers(mismatch):
    launcher = MagicMock(pid=42)
    launcher.cmdline.return_value = ["python", "-m", "uvicorn", "api.main:app"]
    child = MagicMock(pid=43)
    child.name.return_value = "python.exe"
    child.ppid.return_value = 99 if mismatch == "parent" else 42
    child.cmdline.return_value = ["other"] if mismatch == "arguments" else launcher.cmdline.return_value
    launcher.children.return_value = [child, child] if mismatch == "multiple" else [child]
    root = {"pid": 42, "created": 100, "verified_uvicorn": True, "working_directory_matches": True, "reloader": False}
    worker = {"pid": 43, "created": 99 if mismatch == "created" else 101,
              "verified_uvicorn": mismatch != "unverified", "working_directory_matches": mismatch != "workspace",
              "reloader": mismatch == "reloader"}
    with patch("psutil.Process", return_value=launcher), patch.object(local_runtime, "process_identity", side_effect=lambda pid: root.copy() if pid == 42 else worker.copy()):
        with pytest.raises(RuntimeError):
            local_runtime.launched_worker_identity(MagicMock(pid=42))


def test_removed_action_is_rejected():
    with patch("sys.argv", ["local_runtime.py", "recover"]):
        with pytest.raises(SystemExit) as caught:
            local_runtime.main()
    assert caught.value.code == 2
    for removed in ("rollback_status", "ROLLBACK", "OLD_INDEX", "OLD_COLLECTION", "BACKUP", "RECOVERY_WORKDIR"):
        assert not hasattr(local_runtime, removed)


@pytest.mark.parametrize("failure", ["matching", "id", "payload", "missing", "duplicate", "pagination"])
def test_launcher_uses_the_canonical_read_only_integrity_check_before_start(tmp_path, failure):
    from src.qdrant_integrity import expected_payload, point_id

    items = [{"chunk_id": f"pinned-{i}", "content": f"Pinned content {i}", "metadata": {"year": 2026}} for i in range(3185)]
    index = tmp_path / local_runtime.LOCAL_INDEX_DIR
    index.mkdir(parents=True)
    (index / "manifest.json").write_text(json.dumps({"chunks": items}), encoding="utf-8")
    points = [{"id": point_id(item["chunk_id"]), "payload": expected_payload(item)} for item in items]
    if failure == "id":
        points[0]["id"] = point_id("unrelated")
    elif failure == "payload":
        points[0]["payload"]["metadata"]["year"] = 2025
    elif failure == "missing":
        points.pop()
    elif failure == "duplicate":
        points[1] = points[0]
    cache = tmp_path / "models/huggingface/hub/models--BAAI--bge-m3/snapshots/cached"
    cache.mkdir(parents=True)
    for name in ("config.json", "modules.json", "pytorch_model.bin"):
        (cache / name).write_text("{}")
    security = tmp_path / "security.json"
    security.write_text('{"confirmed": true}')
    scrolls = []
    def request(url, *, payload=None):
        if url.endswith("/points/scroll"):
            scrolls.append(payload)
            assert payload["with_payload"] is True and payload["with_vector"] is False
            if failure == "pagination" and payload["offset"] is not None:
                raise TimeoutError("inspection failed")
            start = payload["offset"] or 0
            end = start + payload["limit"]
            return {"result": {"points": points[start:end], "next_page_offset": end if end < len(points) else None}}
        if url.endswith("/api/health"):
            raise URLError("no API")
        return {"result": {"status": "green", "points_count": 3185,
                           "config": {"params": {"vectors": {"size": 1024, "distance": "Cosine"}}}}}
    with (
        patch.object(local_runtime, "ROOT", tmp_path),
        patch.object(local_runtime, "SECURITY", security),
        patch.object(local_runtime, "ARTIFACTS", tmp_path / "artifacts"),
        patch.object(local_runtime, "application_revision", return_value="a" * 64),
        patch.object(local_runtime, "memory_status", return_value=None),
        patch.object(local_runtime, "validate_snapshot", return_value={"chunks": 3185}),
        patch.object(local_runtime, "api_listeners", return_value=[]),
        patch.object(local_runtime, "request_json", side_effect=request),
        patch.object(local_runtime, "stop_verified") as stop,
        patch.object(local_runtime, "launch") as launch,
    ):
        report = local_runtime.inspect()
        if failure == "matching":
            assert report["blockers"] == []
            assert report["qdrant"]["compatible"] is True
            assert len(scrolls) == 13
        else:
            code = "QDRANT_INSPECTION_FAILED" if failure == "pagination" else "QDRANT_INDEX_MISMATCH"
            assert code in report["blockers"]
            assert report["qdrant"]["compatible"] is False
            with patch("sys.argv", ["local_runtime.py", "start"]):
                assert local_runtime.main() == 1
        stop.assert_not_called()
        launch.assert_not_called()


@pytest.mark.parametrize("field,value", [("size", 768), ("distance", "Dot")])
def test_launcher_rejects_wrong_vector_configuration_without_scrolling(field, value):
    from src.runtime_resources import StartupCheckError

    vectors = {"size": 1024, "distance": "Cosine", field: value}
    with patch.object(local_runtime, "request_json") as request:
        with pytest.raises(StartupCheckError) as error:
            local_runtime.validate_collection_integrity({"config": {"params": {"vectors": vectors}}})
    assert error.value.code == "QDRANT_INDEX_MISMATCH"
    request.assert_not_called()
