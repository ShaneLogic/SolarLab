"""Previously missed time-axis failures retain exact historical conditions."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from perovskite_sim.experiments.one_dimensional_mechanism_r1_failure_registry import (
    CATEGORIES, exact_conditions, historical_observation, load_failure_registry,
    merge_failure_registries,
)

PROJECT = Path(__file__).resolve().parents[3]
V2 = json.loads((PROJECT / "reproducibility/OneDimensionalMechanismR1AdditionalFailuresV2.json").read_text())
ENTRIES = [entry for category in CATEGORIES for entry in V2[category]]


def test_new_registry_retains_fourteen_observed_conditions_without_rewriting_history():
    assert len(ENTRIES) == 14
    assert sum(entry["control"] == "D" for entry in ENTRIES) == 6
    assert [len(V2[key]) for key in CATEGORIES] == [10, 4]
    assert all(entry["confirmation"]["run_class"] == "development" for entry in ENTRIES)
    assert all(entry["observed_source_commit"] == "04170c4efcbcaa023a63fcee1f30bb41cceee0f8" for entry in ENTRIES)
    assert all(entry["evidence_sources"][0]["json_pointer"] for entry in ENTRIES)
    assert all(len(entry["confirmation"]["manifest_sha256"]) == 64 for entry in ENTRIES)
    assert hashlib.sha256((PROJECT / "reproducibility/OneDimensionalMechanismR1DynamicsInputV1.json").read_bytes()).hexdigest() == "3065a31951d4a90e8b0d03d042e3f2c0349cc7d8e792381a275cb05d58a047ca"
    assert hashlib.sha256((PROJECT / "reproducibility/OneDimensionalMechanismR1AdditionalFailuresV1.json").read_bytes()).hexdigest() == "ae2d52cf08a029e0273689d03f16ce8945fdd1ca423582542a49a641d50f9aaa"
    merged = load_failure_registry()
    assert sum(len(merged[key]) for key in CATEGORIES) == 39


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda entry: entry["case_id"])
def test_each_extra_case_matches_only_its_measured_axes_and_reports_source(entry):
    request = {**exact_conditions(entry), "source_commit": "f" * 40}
    result = historical_observation(request, {"message": entry["failure"]})
    assert not result["waives_checks"] and not result["changes_acceptance_thresholds"]
    matches = result["matching_historical_cases"]
    assert len(matches) == 1 and matches[0]["case_id"] == entry["case_id"]
    assert matches[0]["source_relation"] == "different_source"
    assert matches[0]["outcome"] == "historical_signature_recurred"
    assert historical_observation(request, None)["matching_historical_cases"][0]["outcome"] == "previously_failed_case_now_passed"
    assert historical_observation(request, {"message": "new failure"})["matching_historical_cases"][0]["outcome"] == "different_failure_signature"
    request["refinement_substeps"] = [8, 16, 32]
    assert historical_observation(request, None)["matching_historical_cases"] == []
    request = exact_conditions(entry)
    request["times_s"][-1] *= 10
    assert historical_observation(request, None)["matching_historical_cases"] == []


@pytest.mark.parametrize("field", ["intervals", "control", "nonlinear_factor", "amplitude_V", "times_s"])
def test_missing_conditions_are_never_interpreted_as_wildcards(field):
    request = exact_conditions(ENTRIES[0])
    request.pop(field)
    with pytest.raises((KeyError, ValueError)):
        historical_observation(request, None)


def test_duplicate_exact_measurement_and_waiver_are_rejected():
    additional = deepcopy(V2)
    additional["known_nonconvergence"].append({**additional["known_nonconvergence"][0], "case_id": "duplicate"})
    with pytest.raises(ValueError, match="duplicate exact conditions"):
        merge_failure_registries({}, additional)
    additional = deepcopy(V2)
    additional["known_failure_semantics"]["waives_checks"] = True
    with pytest.raises(ValueError, match="cannot waive"):
        merge_failure_registries({}, additional)
