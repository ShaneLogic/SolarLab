"""Freeze measured history, including passes, so transcription loss is visible.

The expected table was independently extracted from both preserved 48-case
review sweeps at 84f9131. It is deliberately separate from the registry under
test. These are historical-data checks, not fresh solver convergence claims.
"""

from itertools import product
import json
from pathlib import Path

import pytest


INPUT = Path(__file__).resolve().parents[3] / "reproducibility/OneDimensionalMechanismR1DynamicsInputV1.json"
STUDY = json.loads(INPUT.read_text())
SOURCE = "3dd7e0c923fce689909ef4e4237abea400f02b4e"
CONFIRMATION_SOURCE = "84f9131123869a8f55a3589d0f3c435c2add8b35"
TIMES = [0.0, 1e-9, 1e-8, 1e-6, 1e-4]
FACTORS = (1.0, 0.1, 0.01, 0.001)

# Factor order is FACTORS. P=passed, I=initial local line-search failure,
# N=finite-step Newton failure, G=physical gate failure after local convergence.
EXPECTED_SHORT_MATRIX = {
    ("A", 16): ("P", "P", "P", "I"),
    ("B", 16): ("P", "P", "P", "I"),
    ("C", 16): ("P", "P", "P", "I"),
    ("D", 16): ("P", "P", "P", "I"),
    ("A", 32): ("P", "P", "P", "I"),
    ("B", 32): ("P", "P", "P", "I"),
    ("C", 32): ("P", "P", "P", "I"),
    ("D", 32): ("P", "P", "N", "I"),
    ("A", 64): ("P", "P", "I", "I"),
    ("B", 64): ("G", "P", "I", "I"),
    ("C", 64): ("P", "P", "I", "I"),
    ("D", 64): ("G", "P", "I", "I"),
}
INITIAL_MESSAGE = "R1 controlled step failed: R1-1 fixed-population local line search stalled"
NEWTON_MESSAGE = (
    "R1 controlled step failed: analytic sparse Newton line search stalled at iteration 4 "
    "with residual 0.0500128, charge closure 2.23617e-19, all-face current closure "
    "3.38803e-12, interface current closure 1.0046e-11, linear backward error 1.24795e-16"
)
CURRENT_REASON = "contact_internal_current_spread_relative_exceeds_limit"
CHARGE_REASON = "charge_balance_normalized_exceeds_limit"
REPORT_IDENTITIES = {
    "R1StageTwoAcceptanceV1/ReviewV1/LaneReportsV1.json":
        "3523000af39ae0ab2f9878a7a83584871a9a4fd942a0d29ae68c1eb52124556b",
    "R1StageTwoAcceptanceV1/ReviewV1/RefutationsV1.json":
        "243491a0a93503d00a3e0672c1fad9f85c6ecb724499a4a9867ae5fe2554d445",
}


def all_entries():
    return [(category, entry)
            for category in ("known_nonconvergence", "known_physical_gate_failures")
            for entry in STUDY[category]]


def matching(control, intervals, factor, amplitude):
    return [(category, entry) for category, entry in all_entries()
            if (entry["control"], entry["intervals"], entry["nonlinear_factor"],
                entry["amplitude_V"]) == (control, intervals, factor, amplitude)]


@pytest.mark.parametrize("control,intervals,factor,expected", [
    (control, intervals, factor, expected)
    for (control, intervals), outcomes in EXPECTED_SHORT_MATRIX.items()
    for factor, expected in zip(FACTORS, outcomes)
])
def test_every_measured_short_matrix_cell_has_exact_historical_classification(
    control, intervals, factor, expected,
):
    found = matching(control, intervals, factor, .005)
    if expected == "P":
        assert found == [], "A measured passing cell must not acquire a historical failure"
        return
    assert len(found) == 1
    category, entry = found[0]
    assert entry["observed_source_commit"] == SOURCE
    assert entry["coverage_id"] == "short_functional_5mV"
    assert entry["times_s"] == TIMES
    assert entry["refinement_substeps"] == [1, 2, 4]
    confirmation = entry["confirmation"]
    assert confirmation["observed_source_commit"] == CONFIRMATION_SOURCE
    if expected == "G":
        assert category == "known_physical_gate_failures"
        assert entry["phase"] == "finite_step"
        assert entry["failure_reasons"] == [CURRENT_REASON]
        message = "R1 row failed physical gates: " + CURRENT_REASON
        counts = (25, 22, 21)
    else:
        assert category == "known_nonconvergence"
        assert "failure_reasons" not in entry
        assert entry["phase"] == ("0+" if expected == "I" else "finite_step")
        message = INITIAL_MESSAGE if expected == "I" else NEWTON_MESSAGE
        counts = (0, 0, 0) if expected == "I" else (7, 5, 5)
    assert entry["failure"] in message
    assert confirmation["failure_message"] == message
    assert tuple(confirmation[key] for key in (
        "persisted_record_count", "persisted_finite_step_count",
        "physical_passed_finite_step_count")) == counts


@pytest.mark.parametrize("control,intervals,amplitude", list(product("ABCD", (32, 64), (.0025, .01))))
def test_supplementary_amplitude_cases_do_not_extrapolate_the_5mv_failure_set(
    control, intervals, amplitude,
):
    found = matching(control, intervals, 1.0, amplitude)
    if not (control in "BD" and intervals == 32 and amplitude == .01):
        assert found == []
        return
    assert len(found) == 1
    category, entry = found[0]
    assert category == "known_physical_gate_failures"
    assert entry["coverage_id"] == "supplementary_amplitudes_2p5mV_10mV"
    assert entry["observed_source_commit"] == SOURCE
    assert entry["times_s"] == TIMES
    assert entry["refinement_substeps"] == [1, 2, 4]
    assert entry["failure_reasons"] == [CHARGE_REASON, CURRENT_REASON]
    assert entry["failure"] == CHARGE_REASON + ", " + CURRENT_REASON
    confirmation = entry["confirmation"]
    assert confirmation["observed_source_commit"] == CONFIRMATION_SOURCE
    assert confirmation["failure_message"] == "R1 row failed physical gates: " + entry["failure"]
    assert tuple(confirmation[key] for key in (
        "persisted_record_count", "persisted_finite_step_count",
        "physical_passed_finite_step_count")) == (25, 22, 21)


@pytest.mark.parametrize("control,intervals,amplitude,metrics", [
    ("B", 64, .005, {"contact_internal_current_spread_relative": 2.1279340250367033e-6}),
    ("D", 64, .005, {"contact_internal_current_spread_relative": 2.1276232966245267e-6}),
    ("B", 32, .01, {"charge_balance_normalized": 1.2362143191941625e-10,
                    "contact_internal_current_spread_relative": 2.0315452264455667e-6}),
    ("D", 32, .01, {"charge_balance_normalized": 1.2361751755518348e-10,
                    "contact_internal_current_spread_relative": 2.03124602443833e-6}),
])
def test_all_physical_failure_reasons_and_measured_values_are_preserved(
    control, intervals, amplitude, metrics,
):
    _, entry = matching(control, intervals, 1.0, amplitude)[0]
    assert entry["phase"] == "finite_step"
    assert entry["observed_record_index"] == 24
    assert entry["observed_time_s"] == 5.05e-7
    assert entry["observed_dt_s"] == 2.475e-7
    assert entry["observed_refinement_substeps"] == 4
    assert set(entry["observed_checks"]) == set(metrics)
    for name, expected in metrics.items():
        check = entry["observed_checks"][name]
        assert check["value"] == expected
        assert check["limit"] == (1e-10 if name == "charge_balance_normalized" else 2e-6)
        assert check["value"] > check["limit"]
        assert check["failure_reason"] == name + "_exceeds_limit"
    if len(metrics) == 1:
        assert entry["observed_metric"] == next(iter(metrics.values()))
        assert entry["limit"] == 2e-6


def test_registry_counts_and_unique_full_conditions_include_every_observed_failure():
    assert len(STUDY["known_nonconvergence"]) == 17
    assert len(STUDY["known_physical_gate_failures"]) == 4
    entries = [entry for _, entry in all_entries()]
    assert len({entry["case_id"] for entry in entries}) == len(entries) == 21
    expected_key_fields = [
        "observed_source_commit", "control", "intervals", "nonlinear_factor",
        "amplitude_V", "times_s", "refinement_substeps",
    ]
    assert STUDY["known_failure_semantics"]["deduplication_key"] == expected_key_fields
    keys = [json.dumps([entry[key] for key in expected_key_fields]) for entry in entries]
    assert len(set(keys)) == len(keys)
    assert sum(entry["amplitude_V"] == .005 for entry in entries) == 19
    assert sum(entry["amplitude_V"] == .01 for entry in entries) == 2


def test_historical_records_remain_bound_to_the_review_documents():
    for _, entry in all_entries():
        assert {item["path"]: item["sha256"] for item in entry["evidence_sources"]} == REPORT_IDENTITIES
        assert all(item["json_pointer"] == "/FailureRegistry" for item in entry["evidence_sources"])
        assert len(bytes.fromhex(entry["confirmation"]["manifest_sha256"])) == 32
    # These eight cases have newly inspected parent bundles, not extrapolated histories.
    parent_confirmed = [entry for _, entry in all_entries() if "parent_confirmation_manifest_sha256" in entry]
    assert len(parent_confirmed) == 8
    assert all(len(bytes.fromhex(entry["parent_confirmation_manifest_sha256"])) == 32
               for entry in parent_confirmed)


@pytest.mark.parametrize("coverage_id,axes,counts", [
    ("short_functional_5mV", ([16, 32, 64], [1., .1, .01, .001], [.005]), (48, 29, 19, 17, 2, 2)),
    ("supplementary_amplitudes_2p5mV_10mV", ([32, 64], [1.], [.0025, .01]), (16, 14, 2, 0, 2, 1)),
])
def test_coverage_is_complete_only_for_explicit_measured_slices(coverage_id, axes, counts):
    coverage = next(item for item in STUDY["known_failure_coverage"] if item["coverage_id"] == coverage_id)
    assert coverage["observed_source_commit"] == CONFIRMATION_SOURCE
    assert coverage["controls"] == list("ABCD")
    assert tuple(coverage[key] for key in ("intervals", "nonlinear_factors", "amplitudes_V")) == axes
    assert coverage["times_s"] == TIMES
    assert coverage["refinement_substeps"] == [1, 2, 4]
    assert tuple(coverage[key] for key in (
        "case_count", "passed_case_count", "failed_case_count", "nonconvergence_case_count",
        "physical_gate_failure_case_count", "independent_sweep_count")) == counts
    actual = [entry for _, entry in all_entries() if entry["coverage_id"] == coverage_id]
    assert len(actual) == coverage["failed_case_count"]


def test_scope_discloses_amplitude_and_quantitative_signature_limitations():
    semantics = STUDY["known_failure_semantics"]
    for axis in ("controls", "grids", "nonlinear factors", "amplitudes", "windows", "time refinements"):
        assert axis in semantics["scope"]
    assert "failure-message recurrence" in semantics["runtime"]
    assert "not runtime numerical comparison" in semantics["runtime"]
    assert "never skip computation or waive a gate" in semantics["runtime"]
