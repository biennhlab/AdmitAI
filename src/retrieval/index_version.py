"""Version the inputs that determine a persisted retrieval index build."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping


# Bump this value whenever code can change chunk identity, chunk provenance, or
# another persisted payload without changing the source corpus fingerprint.
INDEX_BUILD_REVISION = "phase-a-provenance-v1"
CHUNKING_STRATEGY_REVISION = "combined-v3-provenance"


def index_build_fingerprint(
    module: int,
    *,
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
    table_chunk_size: int,
    parent_chunk_size: int,
    child_chunk_size: int,
    revision: str = INDEX_BUILD_REVISION,
) -> str:
    """Return a stable fingerprint for code/config that shapes an index."""
    descriptor: dict[str, Any] = {
        "index_build_revision": revision,
        "module": module,
        "embedding_model": embedding_model,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "embedding_input": "title_section_content_v1",
        "chunking_strategy": "naive-v1",
    }
    if module in {2, 3}:
        descriptor.update(
            {
                "chunking_strategy": CHUNKING_STRATEGY_REVISION,
                "table_chunk_size": table_chunk_size,
                "parent_chunk_size": parent_chunk_size,
                "child_chunk_size": child_chunk_size,
            }
        )
    return hashlib.sha256(
        json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def manifest_build_stale_reasons(
    manifest: Mapping[str, Any] | None,
    *,
    expected_corpus_fingerprint: str,
    expected_build_fingerprint: str,
    expected_revision: str = INDEX_BUILD_REVISION,
) -> list[str]:
    """Explain why a manifest cannot be reused for the requested build."""
    if manifest is None:
        return ["manifest_missing"]

    reasons: list[str] = []
    revision = manifest.get("index_build_revision")
    if revision is None:
        reasons.append("index_build_revision_missing")
    elif revision != expected_revision:
        reasons.append("index_build_revision_changed")

    build_fingerprint = manifest.get("build_fingerprint")
    if build_fingerprint is None:
        reasons.append("build_fingerprint_missing")
    elif build_fingerprint != expected_build_fingerprint:
        reasons.append("build_fingerprint_changed")

    if manifest.get("corpus_fingerprint") != expected_corpus_fingerprint:
        reasons.append("corpus_fingerprint_changed")
    return reasons
