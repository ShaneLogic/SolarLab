"""A real cold default curve, or explicitly pinned author evidence without repeats."""

from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from examples.research_1d.ionic_jv import compile_case
from examples.research_1d.ionic_steady import controls
from examples.research_1d.shared_site_ionic_fixture import prepare_confined_shared_site_ion_device
from examples.research_1d.transient import json_value
from solarlab.device.compile_ionic_1d import compile_ionic_device_1d
from solarlab.experiments.ionic_transient import load_ionic_steady_initial
from solarlab.experiments.jv import ionic_steady as driver
from solarlab.solvers.nonlinear.ionic_1d import IonicSteadyOptions, IonicSteadyResult
from solarlab.solvers.nonlinear.newton import IterationRecord, NewtonResult, Termination
from solarlab.solvers.nonlinear.problem import GateOutcome, SolveBudget, state_digest
from solarlab.solvers.nonlinear.steady_1d import SteadyOptions


@pytest.fixture(scope="module", autouse=True)
def blas1():
    with threadpool_limits(limits=1):
        yield


@pytest.fixture(scope="module")
def curve(tmp_path_factory):
    device = compile_case("shared")
    path = os.environ.get("SOLARLAB_IONIC_JV_RESULT")
    if path:
        digest = os.environ["SOLARLAB_IONIC_JV_SHA256"]
        record = driver.read_ionic_steady_jv(path, digest)
    else:
        # One real three-point curve, whose first requested point is cold.
        # No separate baseline solve or external/private artifact is needed.
        result = driver.run_ionic_steady_jv(device, [0.001, 0.002, 0.003], generation_scale=0.0,
            options=driver.IonicSteadyJVOptions(controls(device.ion_node_count)), initial_policy="compiled_cold")
        path = tmp_path_factory.mktemp("ionic_jv") / "Curve.json"
        digest = driver.write_ionic_steady_jv(result, path)
        record = driver.read_ionic_steady_jv(path, digest)
        assert result.completed, [(a.voltage_V, a.termination, a.error) for a in result.failed_attempts]
        for old, saved in zip(result.attempts, record["result"]["attempts"]):
            assert old.solution.state.u.tobytes() == saved["solution"]["state"]["u"].tobytes()
    assert record["completed"]
    assert record["result"]["compiled_identity"] == device.identity
    return device, record, Path(path), digest


def state_from_record(device, state):
    return device.state(state["u"], voltage_V=state["electronic_state"]["voltage_V"])


def restored_point(device, record):
    """Restore an actual cached point for control tests; no nonlinear solve."""
    values = deepcopy(record)
    values["state"] = state_from_record(device, values["state"])
    values["initial_state"] = state_from_record(device, values["initial_state"])
    values["checks"] = tuple(GateOutcome(**item) for item in values["checks"])
    newton = values["newton"]
    newton["gates"] = tuple(GateOutcome(**item) for item in newton["gates"])
    newton["iterations"] = tuple(IterationRecord(**item) for item in newton["iterations"])
    newton["termination"] = Termination(newton["termination"])
    values["newton"] = NewtonResult(**newton)
    return IonicSteadyResult(**values)


@pytest.fixture(scope="module")
def saved_initial(curve, tmp_path_factory):
    device, record, _, _ = curve
    point = record["result"]["attempts"][0]["solution"]
    configured = dict(point["solver_controls"])
    configured["electronic"] = SteadyOptions(**configured["electronic"])
    configured["budget"] = SolveBudget(**configured["budget"])
    settings = IonicSteadyOptions(**configured)
    document = {"summary": {"accepted": True, "compiled_identity": device.identity,
        "voltage_V": 0.001, "generation_scale": 0.0}, "result": point}
    raw = (json.dumps(json_value(document), allow_nan=False) + "\n").encode()
    path = tmp_path_factory.mktemp("ionic_jv_initial") / "RecordedPoint.json"
    path.write_bytes(raw)
    initial = load_ionic_steady_initial(device, path, hashlib.sha256(raw).hexdigest(), controls=settings)
    return initial, driver.IonicSteadyJVOptions(settings)


def test_real_equilibrated_curve_has_all_requested_points_currents_and_unknown_metrics(curve):
    device, record, _, _ = curve
    result = record["result"]
    assert result["experiment_kind"] == "equilibrated_ionic_steady_jv"
    assert result["physical_certificate_status"] == "not_evaluated"
    assert (device.node_count, device.ion_node_count, device.state_size) == (37, 13, 137)
    np.testing.assert_array_equal(result["voltage_V"], [0.001, 0.002, 0.003])
    assert result["solver_calls"] == len(result["attempts"]) == 3
    assert result["current_density_A_m2"].shape == (3, 38)
    np.testing.assert_array_equal(result["output_current_A_m2"],
        -device.electronic_component.voltage_polarity * result["current_density_A_m2"][:, 0])
    assert all(value is None for name, value in result["metrics"].items() if name != "limitation")
    assert result["metrics"]["limitation"] == "zero_voltage_not_bracketed"
    assert result["wall_s"] < result["solver_controls"]["wall_s"]
    for index, attempt in enumerate(result["attempts"]):
        point = attempt["solution"]
        assert attempt["accepted"] and attempt["solver_called"] and point["newton"]["accepted"]
        assert all(item["passed"] for item in point["checks"])
        expected = point["diagnostics"]["electronic_current_density_A_m2"] + point["diagnostics"]["ionic_current_density_A_m2"]
        np.testing.assert_array_equal(result["current_density_A_m2"][index], expected)
        assert np.ptp(expected) <= 1e-8


def test_each_original_species_inventory_flux_and_dropped_row_is_retained(curve):
    device, record, _, _ = curve
    weights = device.ion_geometry.cell_measures_m
    for attempt in record["result"]["attempts"]:
        point, seed = attempt["solution"], attempt["initial_guess"]
        info, state = point["diagnostics"], point["state"]
        occupied = state["positive_ion_density_m3"] + state["negative_ion_density_m3"]
        assert np.all(occupied < device.positive_ion_site_limit_m3)
        for species, block in zip(device.ion_species_names, device.ion_species_slices):
            one = info["ion_species"][species]
            density = state[species + "_ion_density_m3"]
            inventory = math.fsum((weights * density).tolist())
            assert inventory == one["inventory_m2"]
            assert abs(inventory / one["inventory_target_m2"] - 1.0) <= 1e-12
            np.testing.assert_array_equal(one["ion_balance_m2_s"], info["original_residual"][block])
            assert one["original_dropped_balance_m2_s"] == info["original_residual"][one["constraint_row"]]
            assert one["ion_balance_l1_m2_s"] <= 1e6 and one["ion_flux_max_m2_s"] <= 1e6
            assert np.max(abs(density / seed[species + "_ion_density_m3"] - 1.0)) > 1e-4


def test_voltage_reference_guess_rebase_preserves_carriers_ions_and_seed_identity(curve):
    device, record, _, _ = curve
    attempts = record["result"]["attempts"]
    for index in (1, 2):
        before, after = attempts[index - 1]["solution"]["state"], attempts[index]["initial_guess"]
        expected = driver._rebase_guess(device, state_from_record(device, before), attempts[index]["voltage_V"])
        np.testing.assert_array_equal(after["u"], expected.u)
        assert attempts[index]["seed_provenance"]["previous_accepted_point"] == index - 1
        assert attempts[index]["seed_provenance"]["previous_state_sha256"] == state_digest(before["u"])
        np.testing.assert_array_equal(after["electronic_state"]["potential_V"], before["electronic_state"]["potential_V"])
        for name in ("electron_density_m3", "hole_density_m3"):
            np.testing.assert_allclose(after["electronic_state"][name], before["electronic_state"][name], rtol=5e-14)
        for species in device.ion_species_names:
            np.testing.assert_array_equal(after[species + "_ion_density_m3"], before[species + "_ion_density_m3"])


def test_declared_controls_are_preserved_except_decreasing_remaining_wall_cap(curve):
    _, record, _, _ = curve
    declared = record["result"]["solver_controls"]
    previous_cap = min(declared["wall_s"], declared["steady"]["wall_s"])
    for attempt in record["result"]["attempts"]:
        actual = dict(attempt["solution"]["solver_controls"])
        assert 0.0 < actual["wall_s"] <= previous_cap
        previous_cap = actual["wall_s"]
        actual["wall_s"] = declared["steady"]["wall_s"]
        assert actual == declared["steady"]


@pytest.mark.parametrize("voltages", [[], [0.001, 0.001], [0.001, 0.003, 0.002], [np.nan], [np.inf], [1 + 0j], list(range(33))])
def test_invalid_voltage_requests_refuse_before_solving(curve, monkeypatch, voltages):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid requested voltages reached a nonlinear solve")
    monkeypatch.setattr(driver, "solve_ionic_steady_state", forbidden)
    with pytest.raises(ValueError, match="voltage"):
        driver.run_ionic_steady_jv(curve[0], voltages, generation_scale=0.0,
            options=driver.IonicSteadyJVOptions(controls(13)), initial_policy="compiled_cold")


@pytest.mark.parametrize("generation", [-1.0, np.nan, np.inf])
def test_invalid_generation_refuses(curve, generation):
    with pytest.raises(ValueError, match="generation"):
        driver.run_ionic_steady_jv(curve[0], [0.001], generation_scale=generation,
            options=driver.IonicSteadyJVOptions(controls(13)), initial_policy="compiled_cold")


def test_D0_and_hidden_or_mismatched_initial_policies_refuse(curve, saved_initial):
    device, _, _, _ = curve
    initial, options = saved_initial
    zero = compile_ionic_device_1d(prepare_confined_shared_site_ion_device(positive_diffusivity_m2_s=0.0),
        intervals_per_layer=(12, 12, 12), ion_domain_layer="absorber", ion_domain_endpoints=("blocking", "blocking"))
    with pytest.raises(ValueError, match="positive-D"):
        driver.run_ionic_steady_jv(zero, [0.001], generation_scale=0.0, options=options, initial_policy="compiled_cold")
    for policy, supplied, message in (("implicit", None, "initial_policy"), ("saved_accepted", None, "accepted artifact"),
            ("compiled_cold", initial, "hidden supplied"), ("saved_accepted", replace(initial, source_sha256="0" * 64), "SHA-256")):
        with pytest.raises(ValueError, match=message):
            driver.run_ionic_steady_jv(device, [0.001], generation_scale=0.0, options=options,
                                       initial_policy=policy, initial=supplied)


def test_first_failed_point_is_preserved_and_never_seeds_a_later_call(curve, monkeypatch, tmp_path):
    device, record, _, _ = curve
    points = [restored_point(device, a["solution"]) for a in record["result"]["attempts"]]
    seen = []
    def controlled(compiled, *, voltage_V, generation_scale, options, initial_state):
        seen.append((voltage_V, initial_state))
        point = replace(points[len(seen) - 1], initial_state=initial_state, solver_controls=asdict(options))
        if len(seen) == 2:
            return replace(point, accepted=False, termination="injected_failure", error="bounded injected second-point refusal")
        return point
    monkeypatch.setattr(driver, "solve_ionic_steady_state", controlled)
    result = driver.run_ionic_steady_jv(device, [0.001, 0.002, 0.003], generation_scale=0.0,
        options=driver.IonicSteadyJVOptions(controls(13)), initial_policy="compiled_cold")
    assert not result.completed and len(result.attempts) == result.solver_calls == 2
    assert [v for v, _ in seen] == [0.001, 0.002]
    assert result.attempts[0].accepted and not result.attempts[1].accepted
    np.testing.assert_array_equal(result.voltage_V, [0.001])
    np.testing.assert_array_equal(seen[1][1].u, driver._rebase_guess(device, points[0].state, 0.002).u)
    np.testing.assert_array_equal(result.failed_attempts[0].solution.candidate_coordinates, points[1].candidate_coordinates)
    path = tmp_path / "Partial.json"
    digest = driver.write_ionic_steady_jv(result, path)
    saved = driver.read_ionic_steady_jv(path, digest)
    assert not saved["completed"] and saved["result"]["attempts"][-1]["error"] == result.failed_attempts[0].error


def test_wall_exhaustion_retains_unstarted_first_point_without_a_solve(curve, monkeypatch):
    ticks = iter((0.0, 1.0, 31.0, 32.0, 33.0))
    monkeypatch.setattr(driver, "time", SimpleNamespace(monotonic=lambda: next(ticks)))
    result = driver.run_ionic_steady_jv(curve[0], [0.001, 0.002], generation_scale=0.0,
        options=driver.IonicSteadyJVOptions(controls(13)), initial_policy="compiled_cold")
    assert not result.completed and result.solver_calls == 0
    assert result.failed_attempts[0].termination == "wall_budget_exhausted"


def test_source_change_after_call_keeps_physical_acceptance_and_seeding(curve, monkeypatch, tmp_path):
    from solarlab.solvers.nonlinear import ionic_1d
    device, record, _, _ = curve
    points = [restored_point(device, a["solution"]) for a in record["result"]["attempts"][:2]]
    called = []
    changed = tmp_path / "edited_source.py"
    changed.write_text("# source edit after admission\n")
    def controlled(compiled, **kwargs):
        called.append(kwargs["voltage_V"])
        monkeypatch.setattr(ionic_1d, "__file__", str(changed))
        return replace(points[len(called)-1], initial_state=kwargs["initial_state"], solver_controls=asdict(kwargs["options"]))
    monkeypatch.setattr(driver, "solve_ionic_steady_state", controlled)
    result = driver.run_ionic_steady_jv(device, [0.001, 0.002], generation_scale=0.0,
        options=driver.IonicSteadyJVOptions(controls(13)), initial_policy="compiled_cold")
    assert called == [0.001, 0.002] and result.completed
    assert all(a.accepted and a.error is None for a in result.attempts)
    np.testing.assert_array_equal(result.attempts[1].initial_guess.u, driver._rebase_guess(device, points[0].state, .002).u)


def test_common_metrics_are_used_only_with_actual_brackets():
    unknown = driver._metrics(np.array([0.001, 0.002]), np.array([10.0, 9.0]), None)
    assert unknown.open_circuit_voltage_V is None and unknown.efficiency is None
    bracketed = driver._metrics(np.array([0.0, 0.5, 1.0]), np.array([10.0, 5.0, 0.0]), 100.0)
    assert (bracketed.open_circuit_voltage_V, bracketed.fill_factor, bracketed.efficiency) == (1.0, 0.25, 0.025)


def test_saved_reader_is_solver_free_lossless_and_rejects_changed_generation(curve, monkeypatch, tmp_path):
    _, original, path, digest = curve
    def forbidden(*args, **kwargs):
        pytest.fail("saved result reader called a model/solver/admission")
    monkeypatch.setattr(driver, "solve_ionic_steady_state", forbidden)
    monkeypatch.setattr(driver, "_admit_evidence", forbidden)
    monkeypatch.setattr(driver.CompiledIonicDevice1D, "state", forbidden)
    read = driver.read_ionic_steady_jv(path, digest)
    for before, after in zip(original["result"]["attempts"], read["result"]["attempts"]):
        assert before["solution"]["state"]["u"].tobytes() == after["solution"]["state"]["u"].tobytes()
    with pytest.raises(ValueError, match="SHA-256"):
        driver.read_ionic_steady_jv(path, "0" * 64)
    changed = deepcopy(original)
    changed["result"]["generation_scale"] = 1.0
    altered_path = tmp_path / "ChangedGeneration.json"
    raw = (json.dumps(driver._encode(changed), allow_nan=False) + "\n").encode()
    altered_path.write_bytes(raw)
    with pytest.raises(ValueError, match="request identity"):
        driver.read_ionic_steady_jv(altered_path, hashlib.sha256(raw).hexdigest())
    changed["result"]["generation_scale"] = -1.0
    raw = json.dumps(driver._encode(changed), allow_nan=False).encode()
    altered_path.write_bytes(raw)
    with pytest.raises(ValueError, match="generation"):
        driver.read_ionic_steady_jv(altered_path, hashlib.sha256(raw).hexdigest())
