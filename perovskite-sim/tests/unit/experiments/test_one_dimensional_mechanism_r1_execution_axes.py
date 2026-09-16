"""Declared R1-2 settings reach the runner without changing physical policy."""

from dataclasses import asdict
from copy import deepcopy
import hashlib
import importlib.util
from itertools import product
from pathlib import Path

import numpy as np
import pytest
from types import SimpleNamespace

from perovskite_sim.experiments import one_dimensional_mechanism_r1_protocol as protocol
from perovskite_sim.experiments.one_dimensional_mechanism_r1_state import digest
from tests.fixtures.r1_reference import approved_r1_binding


PROJECT = Path(__file__).resolve().parents[3]
SCRIPT = PROJECT / "scripts/run_one_dimensional_mechanism_r1_stage_one.py"
SPEC_TIMES = ((1, 2, 4), (2, 4, 8), (4, 8, 16), (8, 16, 32), (16, 32, 64))
SPEC_TOLERANCES = {
    "storage_relative_tolerance", "carrier_storage_atol_m3", "interface_storage_atol_m2",
    "ion_storage_atol_m3", "poisson_relative_tolerance", "poisson_atol_C_m2",
    "interface_algebraic_relative_tolerance", "interface_potential_atol_V",
    "interface_gauss_atol_C_m2", "interface_flux_atol_m2_s",
}


@pytest.mark.parametrize("steps", SPEC_TIMES)
@pytest.mark.parametrize("factor", (1., .1, .01, .001))
def test_time_and_nonlinear_axes_are_independent_and_keep_all_physical_limits(steps, factor):
    original = asdict(protocol.r1_policy(nonlinear_factor=1.))
    changed = asdict(protocol.r1_policy(nonlinear_factor=factor, time_substeps=steps))
    assert changed.pop("refinement_substeps") == steps
    original.pop("refinement_substeps")
    for key, value in original.items():
        assert changed[key] == (value*factor if key in SPEC_TOLERANCES else value), key
    assert changed["maximum_newton_iterations"] == 100
    assert changed["maximum_line_search_steps"] == 40
    assert changed["maximum_near_acceptance_nonmonotone_steps"] == 2


@pytest.mark.parametrize("steps", [(1, 2), (1, 4, 8), (4, 2, 1), (32, 64, 128),
                                   (0, 1, 2), (1., 2., 4.), (True, 2, 4)])
def test_policy_rejects_undeclared_or_coerced_time_settings(steps):
    with pytest.raises((TypeError, ValueError), match="time"):
        protocol.r1_policy(time_substeps=steps)


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.syspath_prepend(str(PROJECT / "scripts"))
    spec = importlib.util.spec_from_file_location("r1_execution_axes_runner", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def solver_boundary(runner, tmp_path, monkeypatch):
    """Capture real argument/policy flow, then stop before an expensive solve."""
    import threadpoolctl

    reference = tmp_path / "ReferenceBindingV1.json"
    runner.write_json(reference, approved_r1_binding())
    prepared = {"schema": "R1CommonStateV1"}
    prepared["sha256"] = digest(prepared)
    prepared_path = tmp_path / "PreparedStateV1.json"
    runner.write_json(prepared_path, prepared)
    captured = {}

    def stop_at_solver(stack, intervals, binding, prepared, **kwargs):
        captured.update(intervals=intervals, **kwargs)
        raise protocol.R1RunError("bounded test stopped at solver boundary", {
            "certificate": {"certified": False, "reasons": ["test_solver_boundary"]},
        })

    monkeypatch.setattr(protocol, "run_r1_step", stop_at_solver)
    monkeypatch.setattr(runner, "_record_execution_source", lambda output: None)
    monkeypatch.setattr(threadpoolctl, "threadpool_info", lambda: [{"num_threads": 1}])
    arguments = ["step", "--development", "--control", "D", "--reference", str(reference),
                 "--prepared", str(prepared_path), "--output-dir", str(tmp_path / "output")]
    return arguments, captured, tmp_path / "output"


@pytest.mark.parametrize("intervals,steps,factor", list(product(
    (16, 32, 64), ((1, 2, 4), (2, 4, 8), (4, 8, 16)), (1., .1, .01),
)))
def test_all_twenty_seven_base_d_settings_reach_solver_with_full_window(
    runner, solver_boundary, intervals, steps, factor,
):
    arguments, captured, output = solver_boundary
    assert runner.main([*arguments, "--intervals", str(intervals), "--time-substeps",
                        *map(str, steps), "--nonlinear-factor", str(factor), "--window", "full"]) == 1
    assert captured["intervals"] == intervals
    assert captured["policy"] == protocol.r1_policy(nonlinear_factor=factor, time_substeps=steps)
    times = captured["times_s"]
    assert len(times) == 134 and times[0] == 0. and times[1] == 1e-9 and times[-1] == 100.
    np.testing.assert_array_equal(times[1::12],
        [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, .001, .01, .1, 1., 10., 100.])
    saved = runner.read_json(output / "ProtocolV1.json")
    assert saved["stage_scope"] == "R1-2-development"
    assert saved["run_class"] == "development"
    assert saved["times_s"] == times.tolist()
    assert saved["time_substeps"] == saved["policy"]["refinement_substeps"] == list(steps)
    assert saved["observation_window"] == {
        "kind": "full", "first_positive_time_s": 1e-9, "last_time_s": 100.,
        "output_point_count": 134, "logarithmic_intervals_per_decade": 12,
    }
    assert "no three-axis convergence" in saved["numerical_validation_scope"]
    completion = runner.read_json(output / "CompletionV1.json")
    assert completion["status"] == "failed" and completion["stage_scope"] == "R1-2-development"
    assert completion["historical_case_observation"]["matching_historical_cases"] == []
    assert not (output / "StepResultV1.json").exists()


@pytest.mark.parametrize("intervals,steps", [(128, (8, 16, 32)), (256, (16, 32, 64))])
def test_extended_space_time_and_observation_axes_reach_solver(runner, solver_boundary, intervals, steps):
    arguments, captured, output = solver_boundary
    assert runner.main([*arguments, "--intervals", str(intervals), "--time-substeps", *map(str, steps),
                        "--window", "full", "--first-time-s", "1e-12", "--last-time-s", "1e5"]) == 1
    assert captured["intervals"] == intervals and captured["policy"].refinement_substeps == steps
    times = captured["times_s"]
    assert len(times) == 206 and times[0] == 0. and times[1] == 1e-12 and times[-1] == 1e5
    assert runner.read_json(output / "ProtocolV1.json")["observation_window"]["output_point_count"] == 206


def test_short_functional_default_is_preserved(runner, solver_boundary):
    arguments, captured, output = solver_boundary
    assert runner.main(arguments) == 1
    assert captured["intervals"] == 16
    assert captured["policy"].refinement_substeps == (1, 2, 4)
    np.testing.assert_array_equal(captured["times_s"], [0., 1e-9, 1e-8, 1e-6, 1e-4])
    saved = runner.read_json(output / "ProtocolV1.json")
    assert saved["stage_scope"] == "R1-1" and saved["observation_window"]["kind"] == "functional"


@pytest.mark.parametrize("extra", [
    ["--time-substeps", "1", "4", "8"], ["--time-substeps", "32", "64", "128"],
    ["--intervals", "512"], ["--window", "full", "--first-time-s", "2e-9"],
    ["--window", "full", "--last-time-s", "1e6"], ["--last-time-s", "1e2"],
    ["--window", "full", "--first-time-s", "nan"],
])
def test_invalid_execution_axes_fail_before_creating_output(runner, tmp_path, extra):
    output = tmp_path / "not_created"
    with pytest.raises(SystemExit) as error:
        runner.main(["step", "--development", "--reference", "reference.json", "--prepared", "state.json",
                     "--control", "D", "--output-dir", str(output), *extra])
    assert error.value.code == 2 and not output.exists()


@pytest.mark.parametrize("extra", [["--intervals", "128"], ["--time-substeps", "2", "4", "8"],
                                   ["--window", "full"]])
def test_direct_extended_settings_still_require_controlled_execution(runner, tmp_path, monkeypatch, extra):
    output = tmp_path / "not_created"
    monkeypatch.setattr(runner, "_record_execution_source", lambda path: None)
    assert runner.main(["step", "--reference", "reference.json", "--prepared", "state.json",
                        "--control", "D", "--output-dir", str(output), *extra]) == 1
    completion, _ = runner.verify_output(output)
    assert completion["stage_scope"] == "R1-rejected"
    assert completion["run_class"] == "rejected_before_execution"
    assert "controlled -I -S launcher" in completion["failure"]["message"]


@pytest.mark.parametrize("extra", [["--time-substeps", "2", "4", "8"], ["--window", "full"]])
def test_step_only_settings_are_not_silently_ignored_by_zero_check(runner, tmp_path, extra):
    output = tmp_path / "not_created"
    with pytest.raises(SystemExit) as error:
        runner.main(["zero-check", "--development", "--reference", "reference.json", "--prepared", "state.json",
                     "--output-dir", str(output), *extra])
    assert error.value.code == 2 and not output.exists()


SUPPLEMENTAL_CASES = [
    ("B", .0003125, 2.4169778930021016e-6, 27, 4,
     "ed9cb57389f0f2e90b33096ec02c3de6e45275b7a21fa9a5397d6f231a4fa036"),
    ("B", .00015625, 2.3391361940958168e-6, 12, 2,
     "af2b48ba04b721a0dac6a3eb518006d1b643cc09c9d2215b8fae0a4535e088b4"),
    ("D", .0003125, 2.4166196890316515e-6, 27, 4,
     "4097d85e3b200e388b776c0476016cd9dacc1de9508c3ae6b756aeaed11b0600"),
    ("D", .00015625, 2.338788966732179e-6, 12, 2,
     "c488461d2704b9a6314a1b214eadd18bf428a99d62b704956c01b4be53cd6863"),
]


def test_supplement_adds_four_observations_without_mutating_original_twenty_one(runner):
    original = runner.read_json(runner.INPUT_PATH)
    snapshot = deepcopy(original)
    additional = runner.read_json(runner.ADDITIONAL_FAILURES_PATH)
    merged = runner._merge_additional_failures(original, additional)
    assert original == snapshot
    assert len(original["known_nonconvergence"]) + len(original["known_physical_gate_failures"]) == 21
    assert len(additional["known_physical_gate_failures"]) == 4
    assert len(merged["known_nonconvergence"]) + len(merged["known_physical_gate_failures"]) == 25
    assert hashlib.sha256(runner.INPUT_PATH.read_bytes()).hexdigest() == (
        "3065a31951d4a90e8b0d03d042e3f2c0349cc7d8e792381a275cb05d58a047ca"
    )


@pytest.mark.parametrize("control,amplitude,metric,index,substeps,manifest_sha256", SUPPLEMENTAL_CASES)
def test_supplemental_failures_keep_exact_metrics_source_and_match_without_waiver(
    runner, control, amplitude, metric, index, substeps, manifest_sha256,
):
    original = runner.read_json(runner.INPUT_PATH)
    additional = runner.read_json(runner.ADDITIONAL_FAILURES_PATH)
    entries = additional["known_physical_gate_failures"]
    entry, = [entry for entry in entries if entry["control"] == control and entry["amplitude_V"] == amplitude]
    assert entry["intervals"] == 16 and entry["nonlinear_factor"] == 1.
    assert entry["times_s"] == [0., 1e-9, 1e-8, 1e-6, 1e-4]
    assert entry["refinement_substeps"] == [1, 2, 4]
    assert entry["observed_source_commit"] == "97d9ef3141c9fb433d534e8f769223ecfaaff6d2"
    assert entry["observed_record_index"] == index
    assert entry["observed_refinement_substeps"] == substeps
    assert entry["observed_metric"] == metric
    assert entry["limit"] == 2e-6
    assert entry["observed_checks"]["contact_internal_current_spread_relative"]["value"] == metric
    assert entry["observed_physical_record"]["contact_internal_current_spread_relative"] == metric
    assert entry["confirmation"]["manifest_sha256"] == entry["raw_evidence"]["manifest_sha256"] == manifest_sha256
    assert entry["parent_confirmation_manifest_sha256"] == (
        "32a8f1cd184f4656c43e708a6388a6e69a9169a2d831609c80ab0ac6d48b0b6e"
    )
    merged = runner._merge_additional_failures(original, additional)
    args = SimpleNamespace(stage="step", intervals=16, control=control, nonlinear_factor=1.,
                           amplitude=amplitude, resolved_times_s=[0., 1e-9, 1e-8, 1e-6, 1e-4])
    result = runner._historical_case_observation(merged, args, protocol.r1_policy(1.),
                                                {"message": entry["failure"]})
    assert result["waives_checks"] is False
    match, = result["matching_historical_cases"]
    assert match["case_id"] == entry["case_id"] and match["category"] == "known_physical_gate_failures"
    assert match["outcome"] == "historical_signature_recurred"
    # A changed time axis or window is a different case, not a historical waiver.
    assert runner._historical_case_observation(merged, args, protocol.r1_policy(1., time_substeps=(2, 4, 8)),
                                                {"message": entry["failure"]})["matching_historical_cases"] == []


def test_runtime_archives_supplement_and_observes_total_registry_count(runner, solver_boundary):
    arguments, captured, output = solver_boundary
    assert runner.main(arguments) == 1
    assert captured, runner.read_json(output / "FailureV1.json")
    saved = runner.read_json(output / "ProtocolV1.json")
    assert saved["historical_failure_case_count"] == 25
    assert (output / "AdditionalFailuresV1.json").read_bytes() == runner.ADDITIONAL_FAILURES_PATH.read_bytes()
    assert (output / "StudyInputV1.json").read_bytes() == runner.INPUT_PATH.read_bytes()
    assert saved["additional_failures_sha256"] == runner.sha256(runner.ADDITIONAL_FAILURES_PATH)


@pytest.mark.parametrize("waiver", ["waives_checks", "skip_computation", "changes_acceptance_thresholds"])
def test_supplement_cannot_authorize_a_historical_waiver(runner, waiver):
    additional = runner.read_json(runner.ADDITIONAL_FAILURES_PATH)
    additional["known_failure_semantics"][waiver] = True
    with pytest.raises(ValueError, match="cannot waive"):
        runner._merge_additional_failures(runner.read_json(runner.INPUT_PATH), additional)
