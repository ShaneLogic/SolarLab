"""Negative tests for independently retained research scope, not just file hashes."""
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from perovskite_sim.experiments.one_dimensional_mechanism_r1_study_request import (
    build_execution_plan, canonical_bytes, inventory_coverage, load_execution_plan, load_qualification_inputs,
)


def test_resealed_smaller_inventory_does_not_satisfy_caller_plan(tmp_path):
    cases = {"Good": {"scope": "example"}, "Failed": {"scope": "example"}}
    plan = build_execution_plan({"source": "fixed"}, ["matrix"], {}, cases)
    path = tmp_path / "pre-execution.json"
    path.write_bytes(canonical_bytes(plan))
    anchor = hashlib.sha256(path.read_bytes()).hexdigest()
    frozen = load_execution_plan(path, anchor)
    assert inventory_coverage(frozen, cases, {"Good": cases["Good"]}) == ["Failed"]
    with pytest.raises(ValueError, match="externally anchored"):
        inventory_coverage(frozen, {"Good": cases["Good"]}, {"Good": cases["Good"]})
    altered = build_execution_plan({"source": "fixed"}, ["matrix"], {}, {"Good": cases["Good"]})
    path.write_bytes(canonical_bytes(altered))
    with pytest.raises(ValueError, match="request digest mismatch"):
        load_execution_plan(path, anchor)


def test_actual_case_parameters_and_finest_membership_are_not_self_declared(tmp_path):
    cases = {"Matrix/N64/F0p01": {"scope": "step", "factor": .01},
             "MatrixCompare/finest": {"scope": "comparison", "required_finest_pair": True}}
    plan = build_execution_plan({}, ["matrix", "compare"], {}, cases)
    changed = {"Matrix/N64/F0p01": {"scope": "step", "factor": .1}}
    with pytest.raises(ValueError, match="planned request"):
        inventory_coverage(plan, cases, changed)
    plan["required_finest_cases"] = []
    path = tmp_path / "bad-plan.json"
    path.write_bytes(canonical_bytes(plan))
    with pytest.raises(ValueError, match="required finest"):
        load_execution_plan(path, hashlib.sha256(path.read_bytes()).hexdigest())


def test_runner_plans_exact_case_and_preparation_without_calling_solver(tmp_path, monkeypatch):
    file = Path(__file__).resolve().parents[3] / "scripts/run_one_dimensional_mechanism_r1_physics_study.py"
    spec = importlib.util.spec_from_file_location("r1_external_request_test_runner", file)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    for name in ("prepare_common_state", "run_r1_step", "solve_controlled_dc"):
        monkeypatch.setattr(runner, name, lambda *a, **k: pytest.fail("planning executed physics"))
    study = runner.Study.__new__(runner.Study)
    study.args = SimpleNamespace(case_filter="", case=["Matrix/N256/T4/F0p01"],
                                 first_time_s=1e-9, last_time_s=100.)
    study.grids, study.controls, study.window = (16, 32, 64, 128, 256), ("D", "A", "B", "C"), "functional"
    study.times, study.frequencies = runner.DEFAULT_TIMES_S, np.array([0., 1.])
    study.request = {"source": "fixed", "window_amplitude_V": .005, "linearity_case": None}
    plan = study.make_plan(("matrix",))
    assert set(plan["cases"]) == {"Preparation/N256", "Matrix/N256/T4/F0p01"}
    assert plan["cases"]["Matrix/N256/T4/F0p01"]["time_substeps"] == [4, 8, 16]
    assert not list(tmp_path.iterdir())
    with pytest.raises(ValueError, match="not defined"):
        study.make_plan(("matrix",), {"case_filter": "", "cases": ["Matrix/N999/T4/F0p01"]})


def test_current_comparison_cannot_ignore_early_or_interior_times():
    file = Path(__file__).resolve().parents[3] / "scripts/run_one_dimensional_mechanism_r1_physics_study.py"
    spec = importlib.util.spec_from_file_location("r1_current_construction_test_runner", file)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    def record(currents):
        return {"times_s": [0., 1e-9, 1e-4], "policy": {"refinement_substeps": [1, 2, 4]},
                "regular_currents": [{"report_contact_current_A_m2": [v, v]} for v in currents],
                "accepted_steps": [{"substeps": 4, "time_s": t, "regular_integrated_charge_C_m2": 0.}
                                   for t in (0., 1e-9, 1e-4)],
                "initial_event": {"impulse_charge_C_m2": 0.}}
    baseline = record([0., 0., 0.])
    assert runner.compare_step_current_charge(baseline, baseline, [0., 0.], [0., 0.])["within_compared_budgets"]
    for currents in ([1e-3, 0., 0.], [0., 1e-3, 0.]):
        report = runner.compare_step_current_charge(baseline, record(currents), [0., 0.], [0., 0.])
        assert not report["regular_current"]["passed"]
        assert not report["within_compared_budgets"]
        assert report["integrated_charge"]["passed"]


def test_qualification_inputs_require_an_external_unchanged_anchor(tmp_path):
    result = tmp_path / "results"
    inputs = load_qualification_inputs(None, None, result_directory=result)
    assert inputs["trusted_evidence"] == {}
    path = tmp_path / "reviewed.json"
    path.write_bytes(canonical_bytes(inputs))
    anchor = hashlib.sha256(path.read_bytes()).hexdigest()
    assert load_qualification_inputs(path, anchor, result_directory=result) == inputs
    with pytest.raises(ValueError, match="digest is required"):
        load_qualification_inputs(path, None, result_directory=result)
    inputs["trusted_evidence"]["unreviewed"] = "a" * 64
    path.write_bytes(canonical_bytes(inputs))
    with pytest.raises(ValueError, match="digest mismatch"):
        load_qualification_inputs(path, anchor, result_directory=result)
    result.mkdir()
    nested = result / "self-approved.json"
    nested.write_bytes(canonical_bytes(inputs))
    with pytest.raises(ValueError, match="outside the result"):
        load_qualification_inputs(nested, hashlib.sha256(nested.read_bytes()).hexdigest(), result_directory=result)
