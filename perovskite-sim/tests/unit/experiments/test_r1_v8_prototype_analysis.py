"""Identity and fail-closed initial-evidence semantics of the V8 analyzer."""
from copy import deepcopy
import json

import pytest

from scripts import analyze_r1_v8_prototype as analyzer


def budget():
    return {"local_state_arithmetic": {"potential_qf_trace_absolute_error_V": 1e-26,
                                       "population_occupancy_relative_error": 1e-27},
            "independent_side_operator": {"symmetric_relative_target": 1e-10, "floor": 1.,
                                           "precision_stability_target": 1e-20},
            "comparison": {"original_gate": 1e-6}}


def test_original_precision_targets_cannot_be_relaxed_by_new_request_identity():
    analyzer.validate_budget(budget())
    changed = deepcopy(budget())
    changed["independent_side_operator"]["symmetric_relative_target"] = 1e-8
    with pytest.raises(ValueError, match="target changed"):
        analyzer.validate_budget(changed)


def test_absent_initial_context_is_not_counted_as_success():
    result = analyzer.initial_checks({}, {}, {})
    assert not result["qualified"]
    assert not result["phases"]["zero_minus"]["qualified"]
    assert not result["phases"]["zero_plus"]["qualified"]


def test_three_anchor_records_are_not_three_independent_initial_calculations(monkeypatch):
    state = {"precision_phi_V_hi": [0.], "precision_phi_V_lo": [0.]}
    phases = []
    monkeypatch.setattr(analyzer, "verify_initial_arithmetic", lambda *a, **kw: phases.append(kw["phase"]) or {"qualified": True})
    monkeypatch.setattr(analyzer, "state_identity", lambda _: "identity")
    result = analyzer.initial_checks({}, {"state": state}, {"physics_reconstruction": {"initial_state": state},
        "initial_event": {"precision_arithmetic_context": {"zero_minus_state_identity": "identity",
            "zero_plus_state_identity": "identity", "zero_plus_state": state}}})
    assert phases == ["zero_minus", "zero_plus"]
    assert result["qualified"]


def test_a_passing_recorded_prefix_is_never_promoted_to_complete_evidence(tmp_path, monkeypatch):
    source, output = tmp_path/"run", tmp_path/"analysis"
    source.mkdir()
    state = {"precision_phi_V_hi": [0.], "precision_phi_V_lo": [0.]}
    rows = [{"substeps": 1, "time_s": time, "state": state} for time in (0., 1., 2.)]
    (source/"AcceptedStepsV1.jsonl").write_text("".join(json.dumps(row)+"\n" for row in rows))
    monkeypatch.setattr(analyzer, "verify_inputs", lambda *a, **k: (
        {"case": {}}, {"extent": {"accepted_rows": 3}, "execution_status": "failed", "four_predicates_passed": False},
        {}, {}, {}, {"comparison": {"original_gate": 1e-6}}, {"final_source_eligible": True}))
    monkeypatch.setattr(analyzer, "verify_manifest", lambda _: {})
    monkeypatch.setattr(analyzer, "expected_schedule", lambda _: [(1, time) for time in (0.,1.,2.)])
    monkeypatch.setattr(analyzer, "initial_checks", lambda *a: {"qualified": True,
        "zero_plus_fields": analyzer.precision_fields(state)})
    monkeypatch.setattr(analyzer, "verify_two_sides", lambda *a, **k: {"qualified": True,
        "direct": {"checks": {}}, "eliminated": {"checks": {}}})
    monkeypatch.setattr(analyzer, "original_gate", lambda *a: {"passed": True})
    monkeypatch.setattr(analyzer, "compare_state_arithmetic", lambda *a: {"passed": True, "fields": {}})
    result = analyzer.analyze(source, output)
    assert result["observed_local_checks_passed"] and result["prefix_only"]
    assert not result["arithmetic_evidence_subset_qualified"] and not result["P1_qualified"]
    assert result["initial_anchor_rows"] == 1 and result["state_arithmetic_checked_rows"] == 2
    assert result["per_substeps"]["1"]["rows"] == 3
