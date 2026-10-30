"""Single current-source API with diagnostic-only memory measurements.

No model imports are permitted in this control process. Child API credentials
are inherited privately from the environment/.env, never logged or serialized.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import ProxyHandler, Request, build_opener

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.local_profile import LOCAL_COLLECTION, LOCAL_EMBEDDING_MODEL, LOCAL_INDEX_DIR
from src.runtime_resources import StartupCheckError, memory_status
from src.qdrant_integrity import inspect_qdrant_points

ARTIFACTS = ROOT / "unified-local-results"
STATE = ARTIFACTS / "runtime-state.json"
SECURITY = ARTIFACTS / "security-gate.json"
PORT = 8000


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def application_revision():
    paths = sorted((*ROOT.joinpath("api").rglob("*.py"), *ROOT.joinpath("src").rglob("*.py")), key=lambda path: path.relative_to(ROOT).as_posix())
    return hashlib.sha256("\n".join(f"{path.relative_to(ROOT).as_posix()}={sha(path)}" for path in paths).encode()).hexdigest()


def request_json(url, *, payload=None):
    request = url if payload is None else Request(url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"})
    with build_opener(ProxyHandler({})).open(request, timeout=4) as response:
        return json.load(response)


class _ReadOnlyQdrant:
    def scroll(self, *, collection_name, limit, offset, with_payload, with_vectors):
        result = request_json("http://127.0.0.1:6333/collections/" + collection_name + "/points/scroll", payload={
            "limit": limit, "offset": offset, "with_payload": with_payload, "with_vector": with_vectors,
        })["result"]
        return result["points"], result["next_page_offset"]


def validate_collection_integrity(info):
    """Inspect the pinned searchable snapshot without model or SDK imports."""
    vectors = info["config"]["params"]["vectors"]
    if vectors.get("size") != 1024 or vectors.get("distance") != "Cosine":
        raise StartupCheckError("QDRANT_INDEX_MISMATCH", "Qdrant vector configuration differs from the pinned index")
    manifest = json.loads((ROOT / LOCAL_INDEX_DIR / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(manifest.get("chunks"), list) or len(manifest["chunks"]) != info["points_count"]:
        raise StartupCheckError("QDRANT_INDEX_MISMATCH", "Qdrant count differs from the local searchable records")
    return inspect_qdrant_points(_ReadOnlyQdrant(), LOCAL_COLLECTION, manifest["chunks"])


def process_identity(pid):
    import psutil
    process = psutil.Process(pid)
    args = process.cmdline()
    verified = "api.main:app" in args and any("uvicorn" in arg.casefold() for arg in args)
    if "--app-dir" in args:
        index = args.index("--app-dir") + 1
        verified = verified and index < len(args) and Path(args[index]).resolve() == ROOT
    reloader = "--reload" in args
    children = []
    if verified and reloader:
        for child in process.children(recursive=True):
            child_args = child.cmdline()
            owned_worker = child.exe() == process.exe() and Path(child.cwd()).resolve() == ROOT and any(
                "spawn_main" in arg or "resource_tracker" in arg for arg in child_args)
            children.append({"pid": child.pid, "created": child.create_time(), "verified_worker": owned_worker})
    return {"pid": pid, "created": process.create_time(), "verified_uvicorn": verified,
            "reloader": reloader, "executable": process.exe(),
            "working_directory_matches": Path(process.cwd()).resolve() == ROOT,
            "rss_mib": round(process.memory_info().rss / 1024**2, 2), "workers": children}


def api_listeners():
    import psutil
    listeners = []
    for connection in psutil.net_connections(kind="tcp"):
        if connection.status == psutil.CONN_LISTEN and connection.laddr.port in {8000, 8011}:
            pid = connection.pid
            if pid and not any(row["pid"] == pid for row in listeners):
                try:
                    identity = process_identity(pid)
                    identity["port"] = connection.laddr.port
                except (psutil.Error, OSError):
                    identity = {"pid": pid, "port": connection.laddr.port, "verified_uvicorn": False}
                listeners.append(identity)
    return listeners


def validate_snapshot(directory, collection):
    if Path(directory).resolve() != (ROOT / LOCAL_INDEX_DIR).resolve() or collection != LOCAL_COLLECTION:
        raise RuntimeError("INDEX_TARGET_NOT_SUPPORTED")
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    vector_sha = sha(directory / "embeddings.npy")
    if vector_sha != manifest.get("embeddings_sha256"):
        raise RuntimeError("INDEX_CHECKSUM_MISMATCH")
    if manifest.get("qdrant_collection") != collection or manifest.get("embedding_model") != LOCAL_EMBEDDING_MODEL or manifest.get("embedding_dimension") != 1024:
        raise RuntimeError("INDEX_COLLECTION_MODEL_MISMATCH")
    if len(manifest.get("chunks", [])) != manifest.get("chunk_count"):
        raise RuntimeError("INDEX_COUNT_MISMATCH")
    # Import the pure fingerprint module without importing torch/retrieval.
    spec = importlib.util.spec_from_file_location("local_index_version", ROOT / "src/retrieval/index_version.py")
    version = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(version)
    expected = version.index_build_fingerprint(3, embedding_model=LOCAL_EMBEDDING_MODEL,
        chunk_size=500, chunk_overlap=100, table_chunk_size=500, parent_chunk_size=1500, child_chunk_size=300)
    reasons = version.manifest_build_stale_reasons(manifest, expected_corpus_fingerprint=manifest.get("corpus_fingerprint", ""), expected_build_fingerprint=expected)
    if reasons or manifest.get("module") != 3 or manifest["chunk_count"] != 3185:
        raise RuntimeError("INDEX_INCOMPATIBLE:" + ",".join(reasons))
    return {"chunks": manifest["chunk_count"], "collection": collection, "manifest_sha256": sha(manifest_path), "embeddings_sha256": vector_sha}


def inspect():
    report = {"memory": memory_status(), "configuration": {"port": PORT, "workers": 1,
        "index_dir": LOCAL_INDEX_DIR, "collection": LOCAL_COLLECTION, "rerank_backend": "lite",
        "embedding_model": LOCAL_EMBEDDING_MODEL, "models_local_only": True, "query_rewrite_enabled": False},
        "application_revision": application_revision(), "blockers": []}
    try:
        report["index"] = validate_snapshot(ROOT / LOCAL_INDEX_DIR, LOCAL_COLLECTION)
    except Exception as exc:
        report["blockers"].append(str(exc) if isinstance(exc, RuntimeError) else "INDEX_UNAVAILABLE")
    try:
        report["api_listeners"] = api_listeners()
    except Exception:
        report["api_listeners"] = []
        report["blockers"].append("API_IDENTITY_UNVERIFIED")
    try:
        result = request_json("http://127.0.0.1:6333/collections/" + LOCAL_COLLECTION)["result"]
        report["qdrant"] = {"status": result["status"], "points": result["points_count"]}
        if result["status"] != "green" or result["points_count"] != 3185:
            report["blockers"].append("QDRANT_NOT_READY")
        else:
            integrity = validate_collection_integrity(result)
            report["qdrant"].update(integrity)
            if not integrity["compatible"]:
                report["blockers"].append("QDRANT_INDEX_MISMATCH")
    except StartupCheckError as exc:
        report["qdrant"] = {"status": "incompatible", "compatible": False, "error_code": exc.code}
        report["blockers"].append(exc.code)
    except Exception:
        report["qdrant"] = {"status": "unavailable"}
        report["blockers"].append("QDRANT_UNAVAILABLE")
    try:
        response = request_json(f"http://127.0.0.1:{PORT}/api/health")
        report["active_health"] = {"status": response.get("status"), "runtime": response.get("runtime"),
            "components": {key: {name: component[name] for name in ("ready", "chunks", "collection", "rerank_backend") if name in component}
                for key, component in response.get("components", {}).items()}}
    except Exception:
        report["active_health"] = {"status": "unreachable"}
    cache = ROOT / "models/huggingface/hub/models--BAAI--bge-m3/snapshots"
    snapshots = [path for path in cache.glob("*") if (path / "config.json").is_file() and (path / "modules.json").is_file()
                 and any((path / name).is_file() for name in ("model.safetensors", "pytorch_model.bin"))]
    report["local_model_snapshot_present"] = bool(snapshots)
    if not snapshots:
        report["blockers"].append("MODEL_UNAVAILABLE")
    report["security_gate_confirmed"] = SECURITY.exists() and json.loads(SECURITY.read_text()).get("confirmed") is True
    if not report["security_gate_confirmed"]:
        report["blockers"].append("SECURITY_GATE_UNCONFIRMED")
    return report


def stop_verified(identity):
    import psutil
    try:
        current = process_identity(identity["pid"])
    except psutil.NoSuchProcess:
        return
    if current["created"] != identity.get("created") or not current["verified_uvicorn"] or not current["working_directory_matches"]:
        raise RuntimeError("Refusing to stop an unverified, replaced, or reloading process")
    if any(not worker["verified_worker"] for worker in current["workers"]):
        raise RuntimeError("Refusing to stop an unverified child process")
    children = []
    for worker in current["workers"]:
        child = psutil.Process(worker["pid"])
        if child.create_time() != worker["created"]:
            raise RuntimeError("API worker identity changed")
        children.append(child)
    process = psutil.Process(identity["pid"])
    process.terminate()
    for child in children:
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            pass
    process.wait(timeout=15)
    for child in children:
        child.wait(timeout=15)


def launched_worker_identity(process):
    """Resolve a Windows venv redirector's verified Uvicorn child.

    A venv python.exe can be a forwarding process rather than the interpreter
    serving requests. Only a direct child with the same launch arguments and
    workspace can replace its identity; a health-reported PID alone is not proof.
    """
    import psutil

    launcher = psutil.Process(process.pid)
    identity = process_identity(process.pid)
    if not identity["verified_uvicorn"] or not identity["working_directory_matches"] or identity["reloader"]:
        raise RuntimeError("UNVERIFIED_LAUNCH_PROCESS")
    arguments = launcher.cmdline()
    workers = []
    for child in launcher.children():
        if child.name().casefold() == "conhost.exe":
            continue
        candidate = process_identity(child.pid)
        if not candidate["verified_uvicorn"]:
            raise RuntimeError("UNEXPECTED_LAUNCH_CHILD")
        if (not candidate["working_directory_matches"] or candidate["reloader"]
                or child.ppid() != process.pid or candidate["created"] < identity["created"]
                or child.cmdline() != arguments):
            raise RuntimeError("LAUNCH_WORKER_IDENTITY_MISMATCH")
        workers.append(candidate)
    if len(workers) > 1:
        raise RuntimeError("MULTIPLE_LAUNCHED_WORKERS")
    if workers:
        worker = workers[0]
        worker.update(launcher_pid=identity["pid"], launcher_created=identity["created"])
        return worker
    return identity


def launch():
    from dotenv import dotenv_values
    # Credentials are inherited privately and never placed in arguments/state.
    env = {key: value for key, value in dotenv_values(ROOT / ".env").items() if value is not None}
    env.update(os.environ)
    env.update({"LOCAL_DEV_PROFILE": "true",
        "INDEX_DIR": LOCAL_INDEX_DIR, "QDRANT_COLLECTION": LOCAL_COLLECTION,
        "RERANK_BACKEND": "lite", "QUERY_REWRITE_ENABLED": "false",
        "MODELS_LOCAL_ONLY": "true", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "HUGGINGFACE_CACHE_DIR": str(ROOT / "models/huggingface/hub"),
        "ADMITAI_APPLICATION_REVISION": application_revision(),
        "ADMITAI_LAUNCHER": "scripts/start_local.ps1",
        "NO_PROXY": "127.0.0.1,localhost,::1,generativelanguage.googleapis.com",
        "no_proxy": "127.0.0.1,localhost,::1,generativelanguage.googleapis.com"})
    # Pin persisted chunking settings as a unit with the validated snapshot.
    env.update({"EMBEDDING_MODEL": LOCAL_EMBEDDING_MODEL, "CHUNK_SIZE": "500", "CHUNK_OVERLAP": "100", "TABLE_CHUNK_SIZE": "500", "PARENT_CHUNK_SIZE": "1500", "CHILD_CHUNK_SIZE": "300"})
    log = ARTIFACTS / "local-api.log"
    with log.open("ab") as stream:
        process = subprocess.Popen([sys.executable, "-m", "uvicorn", "api.main:app", "--app-dir", str(ROOT), "--host", "127.0.0.1", "--port", str(PORT), "--workers", "1"],
            cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    try:
        time.sleep(1)
        identity = launched_worker_identity(process)
        identity.update(port=PORT, application_revision=env["ADMITAI_APPLICATION_REVISION"])
        STATE.write_text(json.dumps(identity, indent=2), encoding="utf-8")
    except Exception:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=15)
        raise
    return identity


def await_ready(identity):
    import psutil
    deadline = time.monotonic() + 240
    while time.monotonic() < deadline:
        if not psutil.pid_exists(identity["pid"]):
            raise RuntimeError("API_EXITED_DURING_STARTUP")
        if psutil.Process(identity["pid"]).create_time() != identity["created"]:
            raise RuntimeError("API_PID_REUSED")
        try:
            health = request_json(f"http://127.0.0.1:{PORT}/api/health")
            if health.get("status") == "healthy":
                components = health["components"]
                if not all(components.get(name, {}).get("ready") for name in ("database", "local_index", "qdrant", "rag")):
                    raise RuntimeError("RUNTIME_COMPONENT_NOT_READY")
                if components["local_index"].get("chunks") != 3185 or components["qdrant"].get("chunks") != 3185 or components["qdrant"].get("collection") != LOCAL_COLLECTION:
                    raise RuntimeError("RUNTIME_INDEX_IDENTITY_MISMATCH")
                runtime = health.get("runtime", {})
                if runtime.get("pid") != identity["pid"] or runtime.get("application_revision") != identity["application_revision"] or runtime.get("rerank_backend") != "lite" or runtime.get("neural_reranker_module_loaded") is not False or runtime.get("local_dev_profile") is not True:
                    raise RuntimeError("RUNTIME_APPLICATION_IDENTITY_MISMATCH")
                listeners = api_listeners()
                if len(listeners) != 1 or listeners[0]["pid"] != identity["pid"] or listeners[0]["port"] != PORT:
                    raise RuntimeError("RUNTIME_PORT_OWNERSHIP_MISMATCH")
                return health
            code = health.get("runtime", {}).get("startup_error_code")
            if code and health.get("runtime", {}).get("recovery_blocked"):
                raise RuntimeError("STARTUP_BLOCKED:" + code)
        except (URLError, TimeoutError, ConnectionError):
            pass
        print(json.dumps({"event": "waiting_for_readiness", "pid": identity["pid"]}), flush=True)
        time.sleep(5)
    raise RuntimeError("API_READINESS_TIMEOUT")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "start", "stop"))
    parser.add_argument("--security-gate-confirmed", action="store_true")
    args = parser.parse_args()
    ARTIFACTS.mkdir(exist_ok=True)
    if args.security_gate_confirmed:
        SECURITY.write_text(json.dumps({"confirmed": True, "confirmed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "source": "explicit_user_confirmation", "credentials_recorded": False}, indent=2))
    if args.action == "stop":
        if not STATE.exists():
            raise RuntimeError("No recorded launcher PID; refusing to stop an unrelated process")
        identity = json.loads(STATE.read_text())
        if {row["pid"] for row in api_listeners()} - {identity["pid"]}:
            raise RuntimeError("UNEXPECTED_API_LISTENER")
        stop_verified(identity)
        print(json.dumps({"status": "stopped", "pid": identity["pid"]}))
        return 0
    report = inspect()
    (ARTIFACTS / "preflight.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.action == "check":
        print(json.dumps(report, indent=2))
        return 1 if report["blockers"] else 0
    # Validate dependencies before replacing a verified API. Memory is telemetry,
    # including after stop; only an actual allocation failure blocks model loading.
    if report["blockers"]:
        print(json.dumps({"status": "blocked", "blockers": report["blockers"], "memory": report["memory"], "existing_api_untouched": True}))
        return 1
    listeners = report["api_listeners"]
    if len(listeners) > 1 or any(not row["verified_uvicorn"] or not row.get("working_directory_matches") or any(not worker["verified_worker"] for worker in row.get("workers", [])) for row in listeners):
        raise RuntimeError("Cannot establish a single verified API to replace")
    if listeners:
        stop_verified(listeners[0])
    if api_listeners():
        raise RuntimeError("API_PORT_NOT_FREE")
    after_stop = memory_status()
    transition = {"stopped_api_pids": [row["pid"] for row in listeners], "memory_after_stop": after_stop}
    identity = None
    try:
        identity = launch()
        health = await_ready(identity)
        transition.update(status="healthy", identity=identity, memory=memory_status(), health=health)
        (ARTIFACTS / "start-result.json").write_text(json.dumps(transition, indent=2), encoding="utf-8")
        print(json.dumps(transition))
        return 0
    except Exception as exc:
        cleanup_error = None
        try:
            if identity is not None:
                stop_verified(identity)
        except Exception as cleanup_exc:
            cleanup_error = type(cleanup_exc).__name__
        remaining = api_listeners()
        transition.update(status="blocked", blocker=str(exc) if isinstance(exc, RuntimeError) else type(exc).__name__, cleanup_pending=True, port_free=not remaining, candidate_cleanup_error=cleanup_error)
        (ARTIFACTS / "start-result.json").write_text(json.dumps(transition, indent=2), encoding="utf-8")
        raise


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, OSError, ValueError) as exc:
        print(json.dumps({"status": "blocked", "error_type": type(exc).__name__, "reason": str(exc) if isinstance(exc, RuntimeError) else "Local runtime control failed; inspect paths and permissions"}))
        raise SystemExit(1)
