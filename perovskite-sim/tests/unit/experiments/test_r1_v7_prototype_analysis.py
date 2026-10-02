"""Offline analyzer semantics; these fixtures are not scientific evidence."""
from copy import deepcopy
from decimal import Decimal, localcontext
import json

import pytest

from scripts import analyze_r1_v7_prototype as analyzer


def split_state(values):
    state = {}
    with localcontext() as context:
        context.prec = 100
        for name, items in values.items():
            high = [float(item) for item in items]
            low = [float(Decimal(item) - Decimal.from_float(h)) for item, h in zip(items, high)]
            state["precision_" + name + "_hi"] = high
            state["precision_" + name + "_lo"] = low
    return state


@pytest.fixture
def arithmetic_case():
    values = {"phi_V": [Decimal("0.4")] * 3, "dqfn_V": [Decimal(0)] * 3,
              "dqfp_V": [Decimal(0)] * 3, "n_m3": [Decimal("1e15")] * 3,
              "p_m3": [Decimal("2e15")] * 3, "positive_m3": [Decimal("1e20")] * 3,
              "occupancy": [], "trace_potential_V": [], "trace_state_m3": []}
    previous = {"time_s": 0., "substeps": 1, "state": split_state(values)}
    frozen = {"coefficients": {"interface_nodes": [], "D_ion_face_m2_s": ["1", "1"]},
              "constants": {"thermal_voltage_V": "0.025"}}
    coordinate = [0.002, -0.003, 1e-5, -1e-5, 0., 1e-20]
    expected = analyzer.reconstruct_expected(previous["state"], coordinate, frozen)
    row = {"time_s": 1., "substeps": 1, "state": split_state(expected),
           "physics_reconstruction": {"coordinate": coordinate}}
    budget = {"local_state_arithmetic": {"potential_qf_trace_absolute_error_V": 1e-26,
                                         "population_occupancy_relative_error": 1e-27}}
    return previous, row, frozen, budget


def test_actual_words_are_added_before_decimal_reconstruction(arithmetic_case):
    previous, row, frozen, budget = arithmetic_case
    result = analyzer.compare_state_arithmetic(previous, row, frozen, budget)
    assert result["passed"]
    assert row["state"]["precision_phi_V_hi"] == previous["state"]["precision_phi_V_hi"]
    assert row["state"]["precision_phi_V_lo"] != previous["state"]["precision_phi_V_lo"]


def test_previous_state_mispairing_and_erased_low_parts_are_detected(arithmetic_case):
    previous, row, frozen, budget = arithmetic_case
    changed = deepcopy(previous)
    changed["state"]["precision_n_m3_hi"][1] *= 1.001
    assert not analyzer.compare_state_arithmetic(changed, row, frozen, budget)["passed"]
    erased = deepcopy(row)
    erased["state"]["precision_phi_V_lo"][1] = 0.
    assert not analyzer.compare_state_arithmetic(previous, erased, frozen, budget)["passed"]


def test_tier_crossing_and_unknown_boundary_lift_fail_closed(arithmetic_case):
    previous, row, frozen, budget = arithmetic_case
    wrong_tier = {**row, "substeps": 2}
    with pytest.raises(ValueError, match="crosses a tier"):
        analyzer.compare_state_arithmetic(previous, wrong_tier, frozen, budget)
    lifted = deepcopy(row)
    lifted["state"]["precision_phi_V_hi"][-1] += 0.005
    with pytest.raises(ValueError, match="boundary lift"):
        analyzer.compare_state_arithmetic(previous, lifted, frozen, budget)


def test_boundary_is_read_only_from_actual_saved_assembly_words():
    assert analyzer.observed_words({}) == {}
    state = {"precision_boundary_flux_m2_s_hi": [0., 0.], "precision_boundary_flux_m2_s_lo": [0., 0.]}
    assert analyzer.observed_words(state)["boundary_flux_m2_s"] == {"hi": [0., 0.], "lo": [0., 0.]}
    del state["precision_boundary_flux_m2_s_lo"]
    with pytest.raises(ValueError, match="incomplete observed"):
        analyzer.observed_words(state)


def test_manifest_rejects_changes_and_unlisted_evidence(tmp_path):
    (tmp_path / "record.json").write_text("{}")
    analyzer.write(tmp_path / "ManifestV1.json", {"record.json": {"sha256": analyzer.sha(tmp_path / "record.json"), "bytes": 2}})
    assert analyzer.verify_manifest(tmp_path)
    (tmp_path / "extra.json").write_text("{}")
    with pytest.raises(ValueError, match="unmanifested"):
        analyzer.verify_manifest(tmp_path)
    (tmp_path / "extra.json").unlink()
    (tmp_path / "record.json").write_text('{"tampered":true}')
    with pytest.raises(ValueError, match="content mismatch"):
        analyzer.verify_manifest(tmp_path)


def test_manifest_rejects_parent_traversal(tmp_path):
    analyzer.write(tmp_path / "ManifestV1.json", {"../record": {"sha256": "0" * 64, "bytes": 0}})
    with pytest.raises(ValueError, match="unsafe"):
        analyzer.verify_manifest(tmp_path)


def test_original_gate_reports_absolute_and_scale_and_rejects_inconsistent_ratio():
    component = {"maximum_absolute_difference": 1., "normalization_scale": 2.,
                 "relative_error": .5, "normalization_floor": 1., "unit": "m-2 s-1"}
    row = {"physics_reconstruction": {"eliminated_operator_error": .5,
            "eliminated_operator": {"positive_ion_flux": component, "positive_ion_rate": component}}}
    result = analyzer.original_gate(row, 1e-6)
    assert result["complete"] and not result["passed"]
    assert result["components"]["positive_ion_flux"]["absolute"] == "1"
    component["relative_error"] = 0.
    with pytest.raises(ValueError, match="contradicts"):
        analyzer.original_gate(row, 1e-6)


def test_schedule_keeps_each_zero_plus_anchor_separate():
    assert analyzer.expected_schedule({"times_s": [0., 1., 2.], "time_substeps": [1, 2]}) == [
        (1, 0.), (1, 1.), (1, 2.), (2, 0.), (2, .5), (2, 1.), (2, 1.5), (2, 2.)]


def test_a_numerically_passing_prefix_cannot_become_completed(arithmetic_case, tmp_path, monkeypatch):
    previous, row, frozen, budget = arithmetic_case
    run_dir, output = tmp_path / "run", tmp_path / "analysis"
    run_dir.mkdir()
    (run_dir / "AcceptedStepsV1.jsonl").write_text("\n".join(json.dumps(x) for x in (previous, row)) + "\n")
    (run_dir / "ManifestV1.json").write_text("{}")
    request = {"case": {"times_s": [0., 1., 2.], "time_substeps": [1]}, "execution": {"expected_total_rows": 3}}
    summary = {"execution_status": "failed", "four_predicates_passed": False, "extent": {"accepted_rows": 2}}
    budget["comparison"] = {"original_gate": 1e-6}
    identity = {"manifest_sha256": analyzer.sha(run_dir / "ManifestV1.json")}
    monkeypatch.setattr(analyzer, "verify_run_inputs", lambda *_: (request, summary, frozen, budget, identity))
    monkeypatch.setattr(analyzer, "verify_manifest", lambda *_: {})
    monkeypatch.setattr(analyzer, "verify_side", lambda *_: {"qualified": True, "checks": {},
                       "poisson": {"maximum_absolute_C_m2": "0"}, "missing_observed": []})
    monkeypatch.setattr(analyzer, "original_gate", lambda *_: {"passed": True})
    result = analyzer.analyze(run_dir, output)
    assert result["all_observed_local_checks_passed"]
    assert result["prefix_only"] and not result["completed_trajectory_preserved"]
    assert result["initial_anchor_rows"] == 1 and result["state_arithmetic_checked_rows"] == 1
    assert result["P1_qualified"] is False
