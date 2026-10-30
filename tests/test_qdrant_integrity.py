"""Synthetic, read-only tests for the canonical snapshot comparison."""
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock
import subprocess
import sys
import uuid

import pytest

from src.qdrant_integrity import expected_payload, inspect_qdrant_points, point_id
from src.runtime_resources import StartupCheckError


def records(count=2):
    return [{"chunk_id": f"record-{i}", "content": f"Pinned content {i}",
             "metadata": {"source_url": f"https://example.test/{i}", "year": 2026, "active": True}}
            for i in range(count)]


def points_for(items):
    return [{"id": point_id(item["chunk_id"]),
             "payload": {**deepcopy(item["metadata"]), "chunk_id": item["chunk_id"],
                         "content": item["content"], "metadata": deepcopy(item["metadata"])}}
            for item in items]


class ScrollOnly:
    def __init__(self, points):
        self.points = points
        self.calls = []

    def scroll(self, **kwargs):
        self.calls.append(kwargs)
        start = kwargs["offset"] or 0
        end = start + kwargs["limit"]
        return self.points[start:end], end if end < len(self.points) else None


@pytest.mark.parametrize("count", [0, 1, 256, 257, 3185])
def test_exact_collection_is_fully_scrolled_without_vectors_or_writes(count):
    items = records(count)
    client = ScrollOnly(points_for(items))
    result = inspect_qdrant_points(client, "pinned", items)
    assert result["compatible"] is True and result["inspection_complete"] is True
    assert result["observed_point_count"] == count
    assert len(client.calls) == max(1, (count + 255) // 256)
    assert all(call["collection_name"] == "pinned" and call["with_payload"] is True
               and call["with_vectors"] is False for call in client.calls)


@pytest.mark.parametrize("mismatch", ["id", "content", "root-metadata", "nested-metadata", "missing-metadata", "extra-field", "bool-as-int", "missing", "duplicate"])
def test_counts_cannot_hide_id_payload_missing_or_duplicate_mismatches(mismatch):
    items = records()
    points = points_for(items)
    if mismatch == "id":
        points[0]["id"] = point_id("unrelated")
    elif mismatch == "content":
        points[0]["payload"]["content"] = "Wrong content"
    elif mismatch == "root-metadata":
        points[0]["payload"]["source_url"] = "https://wrong.test/"
    elif mismatch == "nested-metadata":
        points[0]["payload"]["metadata"]["source_url"] = "https://wrong.test/"
    elif mismatch == "missing-metadata":
        del points[0]["payload"]["metadata"]
    elif mismatch == "extra-field":
        points[0]["payload"]["extra"] = "untrusted"
    elif mismatch == "bool-as-int":
        points[0]["payload"]["active"] = 1
    elif mismatch == "missing":
        points.pop()
    else:
        points[1] = deepcopy(points[0])
    result = inspect_qdrant_points(ScrollOnly(points), "pinned", items)
    assert result["compatible"] is False
    assert result["error_code"] == "QDRANT_INDEX_MISMATCH"
    if mismatch == "duplicate":
        assert result["duplicate_point_id_count"] == 1
        assert result["missing_point_id_count"] == 1


@pytest.mark.parametrize("failure", ["early-terminal", "duplicate-page", "repeated-offset", "empty-continuation", "second-page-error", "non-repeating-loop", "malformed-response", "malformed-points", "invalid-offset", "invalid-id", "missing-id"])
def test_incomplete_or_unreliable_scroll_never_passes(failure):
    items = records()
    points = points_for(items)
    cursor = point_id("cursor")
    pages = {
        "early-terminal": [(points[:1], None)],
        "duplicate-page": [(points[:1], cursor), (points[:1], None)],
        "repeated-offset": [(points[:1], cursor), (points[1:], cursor)],
        "empty-continuation": [([], cursor)],
        "second-page-error": [(points[:1], cursor), TimeoutError("private endpoint")],
        "non-repeating-loop": [(points[:1], i) for i in range(1, 6)],
        "malformed-response": [None],
        "malformed-points": [({"id": "not-a-list"}, None)],
        "invalid-offset": [(points[:1], False)],
        "invalid-id": [([{"id": "invalid", "payload": {}}], None)],
        "missing-id": [([{"payload": {}}], None)],
    }[failure]
    client = SimpleNamespace(scroll=Mock(side_effect=pages))
    if failure in {"early-terminal", "duplicate-page"}:
        assert inspect_qdrant_points(client, "pinned", items)["compatible"] is False
    else:
        with pytest.raises(StartupCheckError) as error:
            inspect_qdrant_points(client, "pinned", items)
        assert error.value.code == "QDRANT_INSPECTION_FAILED"
        assert "private endpoint" not in str(error.value)
    assert client.scroll.call_count <= len(items) + 1


@pytest.mark.parametrize("invalid", ["duplicate", "uuid-alias", "missing-id", "blank-content", "invalid-metadata", "not-a-list"])
def test_invalid_local_identity_is_rejected_before_contacting_qdrant(invalid):
    items = records()
    if invalid == "duplicate":
        items[1] = deepcopy(items[0])
    elif invalid == "uuid-alias":
        identity = str(uuid.uuid4())
        items[0]["chunk_id"], items[1]["chunk_id"] = identity, identity.upper()
    elif invalid == "missing-id":
        del items[0]["chunk_id"]
    elif invalid == "blank-content":
        items[0]["content"] = " "
    elif invalid == "invalid-metadata":
        items[0]["metadata"] = None
    else:
        items = tuple(items)
    client = SimpleNamespace(scroll=Mock())
    with pytest.raises(StartupCheckError) as error:
        inspect_qdrant_points(client, "pinned", items)
    assert error.value.code == "INDEX_INVALID"
    client.scroll.assert_not_called()


def test_uuid_case_and_payload_key_order_are_not_false_mismatches():
    items = records(1)
    points = points_for(items)
    points[0]["id"] = points[0]["id"].upper()
    points[0]["payload"] = dict(reversed(list(points[0]["payload"].items())))
    assert inspect_qdrant_points(ScrollOnly(points), "pinned", items)["compatible"] is True
    assert expected_payload(items[0]) == points[0]["payload"]


def test_launcher_and_integrity_imports_do_not_import_model_or_retrieval_modules():
    code = "import sys; from scripts import local_runtime; assert not any(name in sys.modules for name in ('torch', 'sentence_transformers', 'src.retrieval', 'src.retrieval.embedder'))"
    result = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
