"""Pinned initial admission, exact dynamic controls, motion and partial outcomes."""

from dataclasses import replace
import hashlib
import json
import math

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from examples.research_1d.ionic_multilayer import prepare_confined_ion_device
from examples.research_1d.ionic_residual import prepare_single_ion_device
from examples.research_1d.ionic_steady import controls as steady_controls
from examples.research_1d.ionic_transient import controls, protocol
from examples.research_1d.transient import json_value
from solarlab.device.compile_ionic_1d import compile_ionic_device_1d
from solarlab.equations.ionic_1d import Ionic1DProblem
from solarlab.experiments import ionic_transient as driver
from solarlab.experiments.transient import TransientProtocol
from solarlab.solvers.nonlinear.ionic_1d import solve_ionic_steady_state
from solarlab.solvers.nonlinear.production import HISTORICAL_MAX_ROW_V1, production_convergence_norm
from solarlab.solvers.time.ionic_backward_euler import IonicBackwardEuler1DStep


@pytest.fixture(scope="module", autouse=True)
def blas1():
    with threadpool_limits(limits=1):
        yield


@pytest.fixture(scope="module")
def case(tmp_path_factory):
    device = compile_ionic_device_1d(prepare_single_ion_device(diffusion_only=True), intervals_per_layer=12)
    admitted_controls = steady_controls(device.ion_node_count)
    point = solve_ionic_steady_state(device, voltage_V=0.002, generation_scale=0.0, options=admitted_controls)
    assert point.accepted
    document = {"summary": {"accepted": point.accepted, "compiled_identity": device.identity,
                             "voltage_V": 0.002, "generation_scale": 0.0}, "result": point}
    raw = (json.dumps(json_value(document), allow_nan=False) + "\n").encode()
    path = tmp_path_factory.mktemp("ionic_initial") / "Steady.json"
    path.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    initial = driver.load_ionic_steady_initial(device, path, digest, controls=admitted_controls)
    return device, initial, path, digest


def test_initial_pin_and_steady_controls_are_checked_without_resolving(case):
    device, initial, path, digest = case
    with pytest.raises(ValueError, match="SHA-256"):
        driver.load_ionic_steady_initial(device, path, "0" * 64, controls=initial.steady_controls)
    with pytest.raises(ValueError, match="controls differ"):
        driver.load_ionic_steady_initial(device, path, digest,
                                          controls=replace(initial.steady_controls, wall_s=31.0))
    np.testing.assert_array_equal(initial.state.u, json.loads(path.read_text())["result"]["state"]["u"])


@pytest.mark.parametrize("mutation", ["not_accepted", "wrong_field", "wrong_component", "false_generation",
                                     "contradictory_termination", "contradictory_check"])
def test_rebound_initial_artifacts_fail_closed(case, tmp_path, mutation):
    device, initial, path, _ = case
    record = json.loads(path.read_text())
    if mutation == "not_accepted":
        record["result"]["accepted"] = False
    elif mutation == "wrong_field":
        record["result"]["state"]["positive_ion_density_m3"][2] *= 1.01
    elif mutation == "wrong_component":
        record["result"]["state"]["electronic_state"]["device_sha256"] = "wrong"
    elif mutation == "false_generation":
        record["summary"]["generation_scale"] = 1.0
    elif mutation == "contradictory_termination":
        record["result"]["newton"]["termination"] = "budget_exhausted"
    else:
        check = record["result"]["checks"][0]
        check["value"] = check["limit"] * 2
    raw = json.dumps(record, allow_nan=False).encode()
    target = tmp_path / "Rebound.json"
    target.write_bytes(raw)
    with pytest.raises(ValueError):
        driver.load_ionic_steady_initial(device, target, hashlib.sha256(raw).hexdigest(), controls=initial.steady_controls)


def test_initial_object_and_first_protocol_sample_are_bound(case):
    device, initial, _, _ = case
    altered = initial.state.u.copy()
    altered[-1] += 0.01
    wrong = replace(initial, state=device.state(altered, voltage_V=0.002))
    with pytest.raises(ValueError, match="pinned steady evidence"):
        driver.solve_ionic_transient(device, wrong, protocol(4), options=controls())
    wrong_protocol = TransientProtocol(np.array([0.0, 0.25]), np.array([0.003, 0.004]), np.zeros(2))
    with pytest.raises(ValueError, match="first protocol sample"):
        driver.solve_ionic_transient(device, initial, wrong_protocol, options=controls())
    limited = replace(controls(), electronic=replace(controls().electronic, max_steps=3))
    with pytest.raises(ValueError, match="max_steps"):
        driver.solve_ionic_transient(device, initial, protocol(4), options=limited)


def test_unsupported_internal_policy_or_predictor_is_explicitly_refused():
    configured = controls()
    with pytest.raises(ValueError, match="poisson_contact_balance_v1"):
        replace(configured, electronic=replace(configured.electronic, convergence_policy=HISTORICAL_MAX_ROW_V1))
    with pytest.raises(ValueError, match="previous-state"):
        replace(configured, electronic=replace(configured.electronic, predictor="generation_inventory"))


def test_exact_N_M_control_allocation_and_L1_norm_without_any_steady_row():
    device = compile_ionic_device_1d(prepare_confined_ion_device(), intervals_per_layer=(12, 12, 12),
                                     ion_domain_layer="absorber", ion_domain_endpoints=("blocking", "blocking"))
    settings, norm = driver._newton_settings(device, controls())
    N, M = device.node_count, device.ion_node_count
    assert (N, M, settings.scaling.row_weights.size) == (37, 13, 124)
    np.testing.assert_array_equal(settings.scaling.row_weights[3 * N:], np.full(M, M / 1e6))
    np.testing.assert_array_equal(settings.step_control.max_step[3 * N:], np.ones(M))
    electronic_norm = production_convergence_norm(device.electronic_component,
                                                   convergence_policy=controls().electronic.convergence_policy)
    scaled = np.zeros(device.state_size)
    scaled[-1] = M
    assert norm(scaled) == 1.0
    scaled[-1] = 1.01 * M
    assert norm(scaled) == 1.01
    scaled[:] = np.random.default_rng(314).uniform(-2, 2, device.state_size)
    assert norm(scaled) == max(electronic_norm(scaled[:3 * N]), math.fsum(abs(scaled[3 * N:]).tolist()) / M)
    assert settings.budget == controls().electronic.newton_budget and not settings.gates


def test_driver_passes_every_physical_BE_row_and_J_to_Newton(case, monkeypatch):
    device, initial, _, _ = case
    requested = protocol(4)
    step = IonicBackwardEuler1DStep(Ionic1DProblem(device, requested.voltage_V[1], 0.0), initial.state, 0.25)
    observed = {}

    def probe_then_refuse(problem, seed, settings, **kwargs):
        trial = seed.copy()
        trial[3 * device.node_count:] += 0.001
        observed["trial"] = trial
        observed["rows"] = problem.residual(trial)
        observed["J"] = problem.jacobian(trial)
        raise ValueError("injected refusal after bounded row/J probes")

    monkeypatch.setattr(driver, "solve_newton", probe_then_refuse)
    result = driver.solve_ionic_transient(device, initial, requested, options=controls())
    assert not result.completed and len(result.failed_attempts) == 1 and len(result.states) == 1
    np.testing.assert_array_equal(observed["rows"], step.residual(observed["trial"]))
    np.testing.assert_array_equal(observed["J"].toarray(), step.jacobian(observed["trial"]).toarray())
    np.testing.assert_array_equal(result.attempts[0].candidate_coordinates, observed["trial"])
    assert result.attempts[0].evaluation_counts == {"residual": 1, "jacobian": 1, "accounting": 0}
    assert "injected refusal" in result.attempts[0].error


@pytest.fixture(scope="module")
def history(case):
    device, initial, _, _ = case
    return driver.solve_ionic_transient(device, initial, protocol(4), options=controls())


def test_real_small_history_moves_ions_and_retains_raw_conservative_accounting(case, history):
    device, initial, _, digest = case
    assert history.completed, [(a.termination, a.error, None if a.newton is None else a.newton.message)
                                for a in history.failed_attempts]
    assert history.physical_certificate_status == "not_evaluated"
    assert history.initial_provenance["source_sha256"] == digest
    np.testing.assert_array_equal(history.accepted_times_s, protocol(4).times_s)
    assert len(history.states) == 5 and len(history.attempts) == 4
    assert np.max(abs(history.states[-1].positive_ion_density_m3 / initial.state.positive_ion_density_m3 - 1)) > 1e-5
    inventory0 = math.fsum((device.ion_geometry.cell_measures_m * initial.state.positive_ion_density_m3).tolist())
    integrated_bound = 0.0
    for index, attempt in enumerate(history.attempts, start=1):
        info = attempt.accounting
        dt = attempt.end_s - attempt.start_s
        assert attempt.accepted and info["physical_certificate_status"] == "not_evaluated"
        assert info["backward_euler_residual"].shape == (device.state_size,)
        assert info["ion_balance_m2_s"].shape == (device.ion_node_count,)
        assert info["total_current_density_A_m2"].shape == (device.node_count + 1,)
        assert math.fsum(abs(info["ion_balance_m2_s"]).tolist()) <= 1e6
        np.testing.assert_allclose(info["total_current_density_A_m2"],
            info["conduction_current_density_A_m2"] + info["displacement_current_density_A_m2"], rtol=0, atol=0)
        np.testing.assert_allclose(info["maxwell_cell_jump_A_m2"], info["predicted_maxwell_cell_jump_A_m2"], atol=1e-12, rtol=1e-10)
        assert np.max(abs(info["ion_domain_fluxes_m2_s"])) > 1e6
        integrated_bound += dt * (math.fsum(abs(info["ion_balance_m2_s"]).tolist())
                                  + math.fsum(abs(info["ion_inventory_readback_error_m2_s"]).tolist()))
        inventory = math.fsum((device.ion_geometry.cell_measures_m * history.states[index].positive_ion_density_m3).tolist())
        assert abs(inventory - inventory0) <= integrated_bound + 2 * index * np.spacing(inventory0)
    assert history.evaluation_counts == {key: sum(a.evaluation_counts[key] for a in history.attempts)
                                         for key in ("residual", "jacobian", "accounting")}


def test_first_failure_preserves_the_real_accepted_prefix_without_retry(case, history, monkeypatch):
    device, initial, _, _ = case
    assert history.completed
    calls = []

    def reused_first_then_fail(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            return history.attempts[0].newton
        raise ValueError("injected second-step refusal")

    monkeypatch.setattr(driver, "solve_newton", reused_first_then_fail)
    result = driver.solve_ionic_transient(device, initial, protocol(4), options=controls())
    assert not result.completed and len(calls) == 2
    assert len(result.states) == 2 and len(result.attempts) == 2 and len(result.failed_attempts) == 1
    np.testing.assert_array_equal(result.states[-1].u, history.states[1].u)
    np.testing.assert_array_equal(result.accepted_times_s, np.array([0.0, 0.25]))
    np.testing.assert_array_equal(result.protocol.times_s, protocol(4).times_s)
    assert result.failed_attempts[0].end_s == 0.5 and "second-step refusal" in result.failed_attempts[0].error


def test_one_deadline_is_not_reset_after_the_accepted_prefix(case, history, monkeypatch):
    device, initial, _, _ = case
    assert history.completed
    monkeypatch.setattr(driver, "solve_newton", lambda *args, **kwargs: history.attempts[0].newton)
    ticks = iter([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.1] + [1.2] * 100)
    monkeypatch.setattr(driver.time, "monotonic", lambda: next(ticks))
    bounded = replace(controls(), electronic=replace(controls().electronic, wall_s=1.0))
    result = driver.solve_ionic_transient(device, initial, protocol(4), options=bounded)
    assert not result.completed and len(result.states) == 2
    assert result.failed_attempts[0].termination == "wall_budget_exhausted"
    assert result.failed_attempts[0].start_s == 0.25
    assert result.failed_attempts[0].evaluation_counts == {"residual": 0, "jacobian": 0, "accounting": 0}


def test_nonfinite_accounting_keeps_the_unaccepted_candidate(case, history, monkeypatch):
    device, initial, _, _ = case
    assert history.completed
    monkeypatch.setattr(driver, "solve_newton", lambda *args, **kwargs: history.attempts[0].newton)
    original_accounting = driver.step_accounting

    def nonfinite(step, y):
        result = original_accounting(step, y)
        result["charge_subtraction_roundoff_A_m2"][0] = np.nan
        return result

    monkeypatch.setattr(driver, "step_accounting", nonfinite)
    result = driver.solve_ionic_transient(device, initial, protocol(4), options=controls())
    assert not result.completed and len(result.states) == 1
    assert result.failed_attempts[0].termination == "nonfinite_accounting"
    np.testing.assert_array_equal(result.failed_attempts[0].candidate_coordinates, history.attempts[0].candidate_coordinates)
    assert np.isnan(result.failed_attempts[0].accounting["charge_subtraction_roundoff_A_m2"][0])
