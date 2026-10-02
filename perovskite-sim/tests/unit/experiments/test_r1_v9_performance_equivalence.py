"""Exact comparison controls, independent of scientific solver executions."""
import hashlib
import json

import pytest

from scripts.check_r1_v9_performance_equivalence import compare, fine_words, FINE_FIELDS, verify_case_manifest


def test_signed_zero_and_one_bit_changes_are_detected():
    assert compare({"a": [0.0]}, {"a": [0.0]})["bit_identical"]
    assert not compare({"a": [-0.0]}, {"a": [0.0]})["bit_identical"]
    assert not compare([1.0], [1.0000000000000002])["bit_identical"]
    assert not compare([1.0], [[1.0]])["bit_identical"]
    assert not compare([True], [1])["bit_identical"]


def test_nonfinite_evidence_is_rejected():
    for number in (float("nan"), float("inf"), -float("inf")):
        with pytest.raises(ValueError, match="nonfinite"):
            compare([number], [number])


def test_fine_field_coverage_requires_all_seventeen_pairs():
    state = {"fields": {name: {"hi": [1.0], "lo": [0.0]} for name in FINE_FIELDS}}
    assert set(fine_words(state)) == FINE_FIELDS
    del state["fields"]["storage"]
    with pytest.raises(ValueError, match="17 fine"):
        fine_words(state)


def test_v9_inventory_includes_nested_manifest_and_rejects_extra_files(tmp_path):
    nested = tmp_path / "IndependentAnalysis/ManifestV1.json"
    nested.parent.mkdir()
    nested.write_text("{}\n")
    entries = {"IndependentAnalysis/ManifestV1.json": {
        "bytes": nested.stat().st_size, "sha256": hashlib.sha256(nested.read_bytes()).hexdigest()}}
    (tmp_path / "ManifestV1.json").write_text(json.dumps(entries))
    assert verify_case_manifest(tmp_path) == entries
    (tmp_path / "unexpected.json").write_text("{}")
    with pytest.raises(ValueError, match="unmanifested"):
        verify_case_manifest(tmp_path)
