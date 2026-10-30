from copy import deepcopy
from pathlib import Path

import pytest

from scripts.validate_golden_dataset import (
    DEFAULT_DATASET,
    DEFAULT_MANIFEST,
    FALLBACK_ANSWER,
    load_json,
    validate_dataset,
)


@pytest.fixture(scope="module")
def golden_dataset():
    return load_json(DEFAULT_DATASET)


def test_checked_in_golden_dataset_schema(golden_dataset):
    assert validate_dataset(golden_dataset) == []


def test_golden_validation_defaults_to_the_supported_current_manifest():
    from src.local_profile import LOCAL_INDEX_DIR
    assert DEFAULT_MANIFEST == Path(__file__).resolve().parents[1] / LOCAL_INDEX_DIR / "manifest.json"


def test_golden_evidence_matches_local_index(golden_dataset):
    if not DEFAULT_MANIFEST.exists():
        pytest.skip("local ignored index is unavailable")

    assert validate_dataset(golden_dataset, load_json(DEFAULT_MANIFEST)) == []


def test_validator_rejects_unknown_evidence(golden_dataset):
    invalid = deepcopy(golden_dataset)
    invalid["cases"][0]["evidence_ids"] = ["missing-chunk"]

    errors = validate_dataset(invalid)

    assert any("references unknown evidence" in error for error in errors)


def test_validator_enforces_out_of_scope_fallback(golden_dataset):
    invalid = deepcopy(golden_dataset)
    negative_case = next(
        case for case in invalid["cases"] if case["expected_route"] == "out_of_scope"
    )
    assert negative_case["ground_truth"] == FALLBACK_ANSWER
    negative_case["ground_truth"] = "Một câu trả lời không có căn cứ."

    errors = validate_dataset(invalid)

    assert any("must use the product fallback" in error for error in errors)
