"""Canonical read-only Qdrant snapshot checks, with no model/retrieval imports."""
from __future__ import annotations

from collections.abc import Mapping
import json
from typing import Any
import uuid

from src.runtime_resources import StartupCheckError

_POINT_ID_NAMESPACE = uuid.UUID("260f54e7-d506-4f94-a0ac-b6eac955ff1b")


def point_id(chunk_id: str) -> str:
    """Preserve the persisted ingestion UUID mapping exactly."""
    try:
        return str(uuid.UUID(chunk_id))
    except (ValueError, TypeError, AttributeError):
        return str(uuid.uuid5(_POINT_ID_NAMESPACE, chunk_id))


def expected_payload(record: Mapping[str, Any]) -> dict[str, Any]:
    """Use the same root and nested metadata payload as the indexing writer."""
    metadata = json.loads(json.dumps(dict(record.get("metadata") or {}), ensure_ascii=False, default=str))
    return {**metadata, "chunk_id": record["chunk_id"], "content": record["content"], "metadata": metadata}


def _json_payload(value: Any) -> str:
    if not isinstance(value, Mapping):
        raise ValueError("Point payload is not a mapping")
    # Canonical JSON also distinguishes bool/int and string/number mismatches.
    return json.dumps(dict(value), sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _remote_id(value: Any) -> str:
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return str(value)
    if isinstance(value, str):
        return str(uuid.UUID(value))
    raise ValueError("Invalid Qdrant point ID or pagination offset")


def inspect_qdrant_points(client: Any, collection: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    """Scroll IDs/payloads only; never fetch vectors, write, repair, or close.

    A completed inspection returns compatible=False for data mismatches.
    Invalid local records or an unreliable inspection raise a coded failure.
    Callers own client cleanup. The page bound also stops non-repeating cursor
    loops; any valid traversal needs at most one nonempty page per local point.
    """
    expected: dict[str, str] = {}
    try:
        if not isinstance(records, list):
            raise ValueError("Local searchable records must be a list")
        for record in records:
            if not isinstance(record, Mapping) or not isinstance(record.get("chunk_id"), str) or not record["chunk_id"].strip():
                raise ValueError("Local record has no chunk ID")
            if not isinstance(record.get("content"), str) or not record["content"].strip():
                raise ValueError("Local record has no content")
            if not isinstance(record.get("metadata"), Mapping):
                raise ValueError("Local record metadata is invalid")
            identity = point_id(record["chunk_id"])
            if identity in expected:
                raise ValueError("Duplicate local point identity")
            expected[identity] = _json_payload(expected_payload(record))
    except (KeyError, TypeError, ValueError) as exc:
        raise StartupCheckError("INDEX_INVALID", "Local searchable records cannot establish Qdrant integrity") from exc

    seen: set[str] = set()
    unexpected: set[str] = set()
    duplicates: set[str] = set()
    payload_mismatches: set[str] = set()
    seen_offsets: set[str] = set()
    offset = None
    pages = 0
    observed = 0
    complete = False
    try:
        while True:
            pages += 1
            if pages > max(1, len(expected) + 1):
                raise ValueError("Qdrant pagination exceeded its finite traversal bound")
            response = client.scroll(collection_name=collection, limit=256, offset=offset,
                                     with_payload=True, with_vectors=False)
            if not isinstance(response, (tuple, list)) or len(response) != 2:
                raise ValueError("Malformed Qdrant scroll response")
            points, next_offset = response
            if not isinstance(points, (list, tuple)) or len(points) > 256:
                raise ValueError("Malformed Qdrant scroll page")
            if not points and next_offset is not None:
                raise ValueError("Empty Qdrant page has a continuation offset")
            for point in points:
                identity = _remote_id(point["id"] if isinstance(point, Mapping) else point.id)
                observed += 1
                if identity in seen:
                    duplicates.add(identity)
                    continue
                seen.add(identity)
                if identity not in expected:
                    unexpected.add(identity)
                    continue
                payload = point.get("payload") if isinstance(point, Mapping) else point.payload
                if not isinstance(payload, Mapping) or _json_payload(payload) != expected[identity]:
                    payload_mismatches.add(identity)
            if next_offset is None:
                complete = True
                break
            cursor = _remote_id(next_offset)
            if cursor in seen_offsets:
                raise ValueError("Qdrant scroll repeated a pagination offset")
            seen_offsets.add(cursor)
            if len(seen) > len(expected):
                break  # Already incompatible; bound memory for oversized data.
            offset = next_offset
    except Exception as exc:
        raise StartupCheckError("QDRANT_INSPECTION_FAILED", "Could not complete a reliable read-only Qdrant inspection") from exc

    missing = set(expected) - seen
    ids_match = not missing and not unexpected and not duplicates
    compatible = complete and ids_match and not payload_mismatches
    return {
        "point_ids_match": ids_match,
        "payloads_match": not payload_mismatches,
        "compatible": compatible,
        "inspection_complete": complete,
        "error_code": None if compatible else "QDRANT_INDEX_MISMATCH",
        "pages": pages,
        "observed_point_count": observed,
        "missing_point_id_count": len(missing),
        "unexpected_point_id_count": len(unexpected),
        "duplicate_point_id_count": len(duplicates),
        "payload_mismatch_count": len(payload_mismatches),
        "missing_point_id_samples": sorted(missing)[:10],
        "unexpected_point_id_samples": sorted(unexpected)[:10],
        "duplicate_point_id_samples": sorted(duplicates)[:10],
        "payload_mismatch_id_samples": sorted(payload_mismatches)[:10],
    }
