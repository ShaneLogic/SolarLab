"""Exercise actual V3 runner qualification wiring with explicit synthetic inputs.

Only the saved/independently-checked input boundary is substituted; the runner
methods and C qualification functions under test execute normally. Synthetic
positive cases test the decision path and do not approve any device budget.
"""

import copy
from dataclasses import asdict
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import perovskite_sim
from perovskite_sim.experiments.one_dimensional_mechanism_r1_admittance import R1AdmittanceErrors
from perovskite_sim.experiments.one_dimensional_mechanism_r1_convergence import observation_times, step_current_charge_responses
from perovskite_sim.experiments.one_dimensional_mechanism_r1_qualification import evidence_digest
from perovskite_sim.experiments.one_dimensional_mechanism_r1_study_response import reconstruct_study_response
from tests.unit.experiments.test_r1_response_qualification import pole_record
from tests.unit.experiments.test_one_dimensional_mechanism_r1_study_response import inputs as reconstruction_inputs


@pytest.fixture
def runner():
    # Resolve the runner from the imported package, so the same file can audit
    # an explicitly selected integration checkout without copying shared code.
    path = Path(perovskite_sim.__file__).resolve().parents[1] / "scripts/run_one_dimensional_mechanism_r1_physics_study.py"
    spec = importlib.util.spec_from_file_location("r1_v3_qualification_wiring_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepared_record(n):
    return {"kind": "equilibrium_D", "sha256": f"prepared-N{n}", "intervals": n,
            "reference_sha256": "reference", "source": {"sha256": "source"},
            "state": {"potential_V": [0., .125, 0.], "initial_state_label": n}}


def controlled_record(item, times):
    a, n = item.amplitude_V, item.intervals
    return {"schema": "R1ControlledStepV1", "control_label": "D", "intervals": n,
            "reference_sha256": "reference", "source": {"sha256": "source"},
            "prepared_sha256": f"prepared-N{n}", "amplitude_V": a,
            "times_s": times.tolist(), "voltage_V": [a] * len(times),
            "policy": {"refinement_substeps": list(item.time_substeps)},
            "certificate": {"certified": True},
            "regular_currents": [{"report_contact_current_A_m2": [2*a, 2*a]} for _ in times],
            "accepted_steps": [{"time_s": float(t), "substeps": max(item.time_substeps),
                                "regular_integrated_charge_C_m2": float(2*a*t)} for t in times],
            "initial_event": {"zero_minus": {"voltage_V": 0.}, "zero_plus": {"voltage_V": a},
                              "voltage_jump_V": a, "impulse_charge_C_m2": 0.}}


@pytest.fixture
def amplitude_study(runner, tmp_path):
    study = runner.Study.__new__(runner.Study)
    study.args = SimpleNamespace(window_amplitude=.005, linearity_case=None, plan_only=False)
    study.grids, study.window, study.times = (16, 32, 64), "full", observation_times()
    study.plan, study.output = {}, tmp_path
    study.qualification_inputs = {"current_budgets": {}, "turnover_evidence": {},
                                 "double_domain_evidence": {}, "trusted_evidence": {}}
    prepared = {n: prepared_record(n) for n in study.grids}
    study.prepared = lambda n: SimpleNamespace(sha256=prepared[n]["sha256"], to_dict=lambda: prepared[n])
    study.zero_dc = lambda n, c: SimpleNamespace(evidence={"current_A_m2": np.zeros(2)})
    records = {study.amplitude_key(c): controlled_record(c, study.times)
               for a in (.005, .0025) for c in study.amplitude_cases(a)}
    reads = []

    def saved(key, *, require_scientific=True):
        reads.append((key, require_scientific))
        return records[key]

    study.saved = saved
    study.compare_available = lambda keys, operation, **kwargs: operation()
    return study, records, prepared, reads


def center_keys(study):
    return [study.amplitude_key(next(c for c in study.amplitude_cases(a)
            if c.intervals == 64 and c.time_substeps == (4, 8, 16) and c.nonlinear_factor == .01))
            for a in (.005, .0025)]


def install_budget(study, records, key, *, trusted=True):
    record = records[key]
    response = step_current_charge_responses(record, [0., 0.])[0]
    budget = {"schema": "R1CurrentErrorBudgetV1", "evidence_id": "synthetic-budget:"+key,
              "scope": study.response_scope(record, 64), "classification": "bounded", "units": "A/m2",
              "amplitude_V": record["amplitude_V"], "method": "exact synthetic constant-current oracle",
              "propagation_method": "sum_nonnegative_absolute_bounds", "source_evidence": ["synthetic-oracle"],
              "coordinates": {"time_s": record["times_s"]}, "components": ["left_contact", "right_contact"],
              "response_sha256": evidence_digest(response), "required_terms": ["oracle"],
              "terms_A_m2": {"oracle": np.zeros(response.values.shape).tolist()},
              "review_status": "approved", "review_id": "unit-test-synthetic-proof-not-device-approval"}
    study.qualification_inputs["current_budgets"][key] = budget
    if trusted:
        study.qualification_inputs["trusted_evidence"][budget["evidence_id"]] = evidence_digest(budget)
    return budget


def test_actual_amplitude_runner_keeps_empirical_pass_unqualified_without_independent_budgets(amplitude_study):
    study, _, _, reads = amplitude_study
    result = study.amplitude_record()
    assert result["comparison"]["passed"]
    assert result["coarse_error_evidence"]["axes_passed"] and result["fine_error_evidence"]["axes_passed"]
    assert result["empirical_comparison_classification"] == "estimate_only"
    assert result["qualification"]["status"] == "budget_unqualified"
    assert result["qualification"]["coarse_budget"]["bound_A_m2"] is None
    assert result["qualification"]["fine_budget"]["bound_A_m2"] is None
    assert not result["linearity_certified"] and not result["within_compared_budgets"]
    assert reads and all(require_scientific for _, require_scientific in reads)


@pytest.mark.parametrize("fault", ["untrusted", "wrong_state", "wrong_units", "wrong_response", "estimate_only"])
def test_actual_amplitude_runner_rejects_external_budget_counterexamples(amplitude_study, fault):
    study, records, _, _ = amplitude_study
    a, b = [install_budget(study, records, key, trusted=fault != "untrusted") for key in center_keys(study)]
    if fault == "wrong_state":
        a["scope"]["state_sha256"] = "borrowed-state"
    elif fault == "wrong_units":
        a["units"] = "S/m2"
    elif fault == "wrong_response":
        a["response_sha256"] = "borrowed-response"
    elif fault == "estimate_only":
        a["classification"] = "estimate_only"
    if fault != "untrusted":
        study.qualification_inputs["trusted_evidence"][a["evidence_id"]] = evidence_digest(a)
    result = study.amplitude_record()
    assert result["comparison"]["passed"]
    assert not result["linearity_certified"]
    assert result["qualification"]["status"] == "budget_unqualified"


@pytest.mark.parametrize("index", [0, 50])
@pytest.mark.parametrize("contact", [0, 1])
def test_actual_amplitude_runner_retains_early_and_interior_contact_errors(amplitude_study, index, contact):
    study, records, _, _ = amplitude_study
    for key in center_keys(study):
        install_budget(study, records, key)
    assert study.amplitude_record()["linearity_certified"]
    coarse = center_keys(study)[0]
    old_final = copy.deepcopy(records[coarse]["regular_currents"][-1])
    records[coarse]["regular_currents"][index]["report_contact_current_A_m2"][contact] += .001
    # Rebind an exact synthetic bound to the changed current. The failure must
    # now arise from all-time linearity, not a stale response digest.
    install_budget(study, records, coarse)
    result = study.amplitude_record()
    assert records[coarse]["regular_currents"][-1] == old_final
    assert not result["linearity_certified"]
    assert result["qualification"]["status"] == "response_not_linear"
    failures = result["qualification"]["comparison"]["failures"]
    assert any(f["coordinates"]["time_s"] == study.times[index]
               and f["component"] == ("left_contact", "right_contact")[contact] for f in failures)


def test_scope_binds_checked_initial_preparation_and_not_amplitude_final_states(amplitude_study):
    study, records, prepared, _ = amplitude_study
    a, b = [records[key] for key in center_keys(study)]
    a["output_states"], b["output_states"] = [{"potential_V": [1.]}], [{"potential_V": [999.]}]
    first, second = study.response_scope(a, 64), study.response_scope(b, 64)
    assert first == second
    assert first["state_sha256"] == evidence_digest(prepared[64]["state"])
    prepared[64]["state"]["potential_V"][1] += .1
    assert study.response_scope(a, 64)["state_sha256"] != first["state_sha256"]


def test_planned_diagnostic_amplitude_does_not_scan_and_replace_itself(amplitude_study):
    study, _, _, _ = amplitude_study
    decoy = study.output / "Linearity/Smaller/AttemptV1"
    decoy.mkdir(parents=True)
    (decoy / "CompletionV1.json").write_text('{}')
    study.saved = lambda *a, **k: pytest.fail("fixed diagnostic request scanned a result")
    assert study.qualified_amplitude() == (.005, None)


def test_qualified_amplitude_requires_the_exact_requested_amplitude_and_initial_state(amplitude_study):
    study, records, prepared, _ = amplitude_study
    for key in center_keys(study):
        install_budget(study, records, key)
    report = study.amplitude_record()
    assert report["linearity_certified"]
    key = "Linearity/A0.005_A0.0025"
    records[key] = report
    study.args.linearity_case = key
    with pytest.raises(ValueError, match="requested window amplitude"):
        study.qualified_amplitude()  # Requested .005; proof is for .0025.
    study.args.window_amplitude = .0025
    assert study.qualified_amplitude() == (.0025, key)
    prepared[64]["state"]["potential_V"][1] += .1
    with pytest.raises(ValueError, match="requested window amplitude"):
        study.qualified_amplitude()


def test_planning_with_a_linearity_reference_never_loads_or_qualifies_results(amplitude_study):
    study, _, _, _ = amplitude_study
    study.args.case_filter, study.args.case = "", []
    study.args.first_time_s, study.args.last_time_s = 1e-9, 100.
    study.controls, study.frequencies = ("D",), np.array([0., 1.])
    study.request = {"source": "fixed", "window_amplitude_V": .0025,
                     "linearity_case": "Linearity/A0.005_A0.0025"}
    study.saved = lambda *a, **k: pytest.fail("planning loaded a saved qualification")
    study.prepared = lambda *a, **k: pytest.fail("planning executed preparation")
    plan = study.make_plan(("windows",))
    windows = {key: value for key, value in plan["cases"].items() if key.startswith("Window/")}
    assert len(windows) == 3
    assert all(value["amplitude_V"] == .0025 for value in windows.values())
    assert all("A0.0025" in key for key in windows)


@pytest.mark.parametrize("fault", [None, "missing", "untrusted", "wrong_state", "wrong_ac"])
def test_frequency_runner_passes_only_the_matching_external_evidence(amplitude_study, fault):
    study, records, _, reads = amplitude_study
    ac, turnover = pole_record()
    ac.update({"intervals": 64, "prepared_sha256": "prepared-N64", "reference_sha256": "reference",
               "source": {"sha256": "source"}})
    key = "AC/N64/D"
    records[key] = ac
    turnover.update(scope=study.response_scope(ac, 64), ac_sha256=evidence_digest(ac))
    if fault == "wrong_state":
        turnover["scope"]["state_sha256"] = "other-state"
    elif fault == "wrong_ac":
        turnover["ac_sha256"] = "other-ac"
    if fault != "missing":
        study.qualification_inputs["turnover_evidence"][key] = turnover
    if fault != "untrusted":
        study.qualification_inputs["trusted_evidence"][turnover["evidence_id"]] = evidence_digest(turnover)
    result = study.frequency_record(64)
    assert result["numeric_checks_passed"]
    assert result["device_frequency_window_certified"] is (fault is None)
    assert (key, False) in reads  # Numeric failures remain reportable diagnostics.


def test_frequency_v2_diagnostic_dispatch_does_not_parse_a_summary_as_raw_ac(runner, amplitude_study):
    study, records, _, _ = amplitude_study
    ac, _ = pole_record()
    ac.update({"intervals": 64, "prepared_sha256": "prepared-N64", "reference_sha256": "reference",
               "source": {"sha256": "source"}})
    records["AC/N64/D"] = ac
    result = study.frequency_record(64)
    assert result["schema"] == "R1FrequencyWindowReportV2" and result["numeric_checks_passed"]
    assert not result["device_frequency_window_certified"]
    assert runner.scientific_checks(result) is True  # Scoped numerical diagnostic only.
    result["numeric_checks_passed"] = False
    assert runner.scientific_checks(result) is False


def test_error_dataclass_digest_matches_external_json_and_binds_actual_reconstruction_values():
    step, prepared, dc, ac, _, errors = reconstruction_inputs()
    error_json = asdict(errors)
    assert evidence_digest({"errors": errors}) == evidence_digest({"errors": error_json})
    result = reconstruct_study_response(step, prepared, dc, ac, errors=errors)
    application = result["qualification"]["application"]
    expected = {"errors": error_json, "quadrature_absolute_tolerance_F_m2": 1e-12,
                "quadrature_relative_tolerance": 1e-10}
    assert application["reconstruction_request_sha256"] == evidence_digest(expected)
    changed = R1AdmittanceErrors(**{**error_json, "current_A_m2": np.full(len(step["times_s"])-1, 1e-14)})
    other = reconstruct_study_response(step, prepared, dc, ac, errors=changed)
    assert other["qualification"]["application"]["reconstruction_request_sha256"] != application["reconstruction_request_sha256"]
    assert not result["double_domain_consistent"] and not other["double_domain_consistent"]


def test_actual_reconstruction_runner_does_not_certify_unreviewed_external_error_numbers(runner, monkeypatch):
    step, common, conductance, ac, _, errors = reconstruction_inputs()
    common["kind"], common["state"] = "equilibrium_D", {"potential_V": [0., .125, 0.]}
    step["policy"] = {"refinement_substeps": [1, 2, 4]}
    step["accepted_steps"] = [{"time_s": float(t), "substeps": 4, "state": common["state"]}
                              for t in step["times_s"]]
    prepared = SimpleNamespace(sha256=common["sha256"], to_dict=lambda: common)
    study = runner.Study.__new__(runner.Study)
    study.stack, study.binding, study.window = None, None, "full"
    study.qualified_amplitude = lambda: (.005, None)
    study.prepared = lambda n: prepared
    records = {"Window/Base/N16/A0.005": step, "AC/N16/D": ac,
               "DC/N16/D": {"conductance": conductance}}
    study.saved = lambda key, **kwargs: records[key]
    study.verified_cases = {key: ("digest", {"certified": True, "content_matches_recomputed": True}) for key in records}
    study.compare_available = lambda keys, operation, **kwargs: (operation() if all(k in records for k in keys)
                                      else {"comparison_available": False})
    study.qualification_inputs = {"current_budgets": {}, "turnover_evidence": {}, "trusted_evidence": {},
                                 "double_domain_evidence": {"DoubleDomain/N16/D": {"errors": asdict(errors)}}}
    monkeypatch.setattr(runner, "solve_controlled_dc", lambda *a, **k: SimpleNamespace())
    monkeypatch.setattr(runner, "compare_transient_tail", lambda *a, **k: {"all_observables_agree": True})
    result = study.reconstruction_record(16)
    assert result["qualification"]["missing_prerequisites"]
    assert not result["double_domain_consistent"] and not result["device_double_domain_consistent"]
    expected = {"errors": asdict(errors), "quadrature_absolute_tolerance_F_m2": 1e-12,
                "quadrature_relative_tolerance": 1e-10}
    assert result["qualification"]["application"]["reconstruction_request_sha256"] == evidence_digest(expected)
    assert result["error_estimate_classification"] == "estimate_only"
    assert result["provided_error_estimates"] == {}
