"""Safe, portable persistence for the Phase-1 NumPy dense index."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from src.ingestion.chunker import Chunk

SCHEMA_VERSION = 1
MANIFEST_NAME = "manifest.json"
EMBEDDINGS_NAME = "embeddings.npy"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_dense_index(
    index_dir: str | Path,
    chunks: list[Chunk],
    embeddings: np.ndarray,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    directory = Path(index_dir)
    directory.mkdir(parents=True, exist_ok=True)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(chunks):
        raise ValueError("Embedding matrix must be 2D and aligned with chunks")

    fd, tmp_vector_name = tempfile.mkstemp(prefix="embeddings-", suffix=".npy", dir=directory)
    os.close(fd)
    tmp_vector = Path(tmp_vector_name)
    tmp_manifest = directory / f"{MANIFEST_NAME}.tmp"
    try:
        with tmp_vector.open("wb") as handle:
            np.save(handle, embeddings, allow_pickle=False)
        vector_hash = _sha256_file(tmp_vector)
        payload = {
            **manifest,
            "schema_version": SCHEMA_VERSION,
            "backend": "naive_dense_numpy",
            "document_count": int(manifest.get("document_count", 0)),
            "chunk_count": len(chunks),
            "embedding_dimension": int(embeddings.shape[1]),
            "embeddings_sha256": vector_hash,
            "chunks": [
                {"chunk_id": chunk.chunk_id, "content": chunk.content, "metadata": chunk.metadata}
                for chunk in chunks
            ],
        }
        tmp_manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp_vector, directory / EMBEDDINGS_NAME)
        os.replace(tmp_manifest, directory / MANIFEST_NAME)
        return payload
    finally:
        tmp_vector.unlink(missing_ok=True)
        tmp_manifest.unlink(missing_ok=True)


def read_manifest(index_dir: str | Path) -> dict[str, Any] | None:
    path = Path(index_dir) / MANIFEST_NAME
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_dense_index(index_dir: str | Path) -> tuple[list[Chunk], np.ndarray, dict[str, Any]]:
    directory = Path(index_dir)
    manifest = read_manifest(directory)
    if manifest is None:
        raise FileNotFoundError(f"Dense index manifest not found: {directory / MANIFEST_NAME}")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported dense index schema: {manifest.get('schema_version')}")
    if manifest.get("backend") != "naive_dense_numpy":
        raise ValueError(f"Unexpected index backend: {manifest.get('backend')}")

    vector_path = directory / EMBEDDINGS_NAME
    if not vector_path.exists():
        raise FileNotFoundError(f"Dense index vectors not found: {vector_path}")
    if _sha256_file(vector_path) != manifest.get("embeddings_sha256"):
        raise ValueError("Dense index vector checksum does not match manifest")
    embeddings = np.load(vector_path, allow_pickle=False)
    chunks = [Chunk(**item) for item in manifest.get("chunks", [])]
    if embeddings.ndim != 2 or len(chunks) != embeddings.shape[0]:
        raise ValueError("Dense index chunks and vectors are not aligned")
    return chunks, embeddings, manifest
