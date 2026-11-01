"""Real electronic/positive-ion protocol adapters, BE parity and BDF tangents."""

from dataclasses import replace
import hashlib
import json

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from examples.research_1d.steady import load_device
from examples.research_1d.transient import controls, json_value
from examples.research_1d.ionic_residual import prepare_single_ion_device
from examples.research_1d.ionic_steady import controls as ionic_steady_controls
from examples.research_1d.ionic_transient import controls as ionic_controls
from solarlab.device.compile_1d import compile_device_1d
from solarlab.device.compile_ionic_1d import compile_ionic_device_1d
from solarlab.equations.dc_1d import DC1DProblem
from solarlab.equations.implementation import ASSEMBLED, LEGACY, equation_implementation
from solarlab.equations.ionic_1d import Ionic1DProblem
from solarlab.experiments.transient import TransientProtocol, solve_transient
from solarlab.experiments.ionic_transient import load_ionic_steady_initial, solve_ionic_transient
from solarlab.solvers.nonlinear.steady_1d import SteadyOptions, solve_steady_state
from solarlab.solvers.nonlinear.ionic_1d import solve_ionic_steady_state
from solarlab.solvers.time.backward_euler import BackwardEuler1DStep
from solarlab.solvers.time.ionic_backward_euler import IonicBackwardEuler1DStep
from solarlab.solvers.time.bdf import BDFOptions, BDFStep, HistoryPoint
from solarlab.solvers.time.bdf_1d import (
    BDFDrive, ElectronicStorage1D, IonicStorage1D, _displacement, bdf_accounting,
)
from solarlab.experiments.bdf import _adapter_step_budget, solve_electronic_bdf, solve_ionic_bdf
from solarlab.experiments.bdf_results import _canonical, _decode, _encode
from support.equation_provenance import counted, one_ulp_residual


@pytest.fixture(scope="module", autouse=True)
def blas1():
    with threadpool_limits(limits=1):
        yield


@pytest.fixture(scope="module")
def electronic():
    device = compile_device_1d(load_device(), intervals_per_layer=12)
    initial_options = SteadyOptions()
    initial = solve_steady_state(device, voltage_V=0.0, generation_scale=1.0, options=initial_options)
    return device, initial, initial_options


@pytest.fixture(scope="module")
def ionic(tmp_path_factory):
    device = compile_ionic_device_1d(prepare_single_ion_device(diffusion_only=True), intervals_per_layer=12)
    steady_options = ionic_steady_controls(device.ion_node_count)
    point = solve_ionic_steady_state(device, voltage_V=.002, generation_scale=0.0, options=steady_options)
    assert point.accepted
    record = {"summary": {"accepted": True, "compiled_identity": device.identity,
                          "voltage_V": .002, "generation_scale": 0.0}, "result": point}
    raw = (json.dumps(json_value(record), allow_nan=False) + "\n").encode()
    path = tmp_path_factory.mktemp("bdf_ionic") / "Steady.json"
    path.write_bytes(raw)
    initial = load_ionic_steady_initial(device, path, hashlib.sha256(raw).hexdigest(), controls=steady_options)
    return device, initial


@pytest.mark.parametrize("voltage", [0.0, .002])
def test_electronic_order1_original_residual_and_J_are_identical(electronic, voltage):
    device, initial, _ = electronic
    old = ElectronicStorage1D(DC1DProblem(device, 0.0, 1.0))
    new = ElectronicStorage1D(DC1DProblem(device, voltage, 1.01))
    history = HistoryPoint(0.0, initial.state.u, old.storage(initial.state.u), old)
    step = BDFStep(new, (history,), 1e-7, 1)
    be = BackwardEuler1DStep(new.problem, initial.state, 1e-7)
    trial = be.initial_coordinates(voltage_lift=True)
    trial[device.node_count:2 * device.node_count] += .01
    np.testing.assert_array_equal(step.residual(trial), be.residual(trial))
    np.testing.assert_array_equal(step.jacobian(trial).toarray(), be.jacobian(trial).toarray())


def test_electronic_fixed_step_BDF1_matches_real_BE_trajectory(electronic):
    device, initial, admission = electronic
    times = np.linspace(0, 4e-7, 5)
    protocol = TransientProtocol(times, np.zeros(5), 1 + .01 * (times / times[-1])**2)
    old = solve_transient(device, initial, protocol, options=controls(), initial_options=admission)
    new = solve_transient(device, initial, protocol, options=controls(), initial_options=admission,
        method="bdf", bdf_options=BDFOptions(1e-7, adaptive=False, max_order=1))
    assert old.completed and new.completed, (new.termination, new.error)
    for state, point in zip(old.states, new.accepted_coordinates):
        np.testing.assert_array_equal(state.u, point)
    for prior, candidate in zip(old.attempts, new.accepted_steps):
        assert candidate.assessment is not None
        for name, value in prior.accounting.items():
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(value, candidate.assessment.accounting[name])


def test_single_ion_fixed_step_BDF1_matches_real_BE_trajectory(ionic):
    device, initial = ionic
    times = np.linspace(0, 1, 5)
    protocol = TransientProtocol(times, .002 + .002 * times, np.zeros(5))
    old = solve_ionic_transient(device, initial, protocol, options=ionic_controls())
    new = solve_ionic_transient(device, initial, protocol, options=ionic_controls(), method="bdf",
        bdf_options=BDFOptions(.25, adaptive=False, max_order=1))
    assert old.completed and new.completed, (new.termination, new.error)
    for state, point in zip(old.states, new.accepted_coordinates):
        np.testing.assert_array_equal(state.u, point)
    for prior, candidate in zip(old.attempts, new.accepted_steps):
        assert candidate.assessment is not None
        for name, value in prior.accounting.items():
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(value, candidate.assessment.accounting[name])


def test_ionic_BDF2_original_rows_directional_J_and_Maxwell_identity(ionic):
    device, initial = ionic
    old0 = IonicStorage1D(Ionic1DProblem(device, .002, 0.0))
    old1 = IonicStorage1D(Ionic1DProblem(device, .003, 0.0))
    new = IonicStorage1D(Ionic1DProblem(device, .004, 0.0))
    u0 = initial.state.u.copy()
    be1 = IonicBackwardEuler1DStep(old1.problem, initial.state, .25)
    u1 = be1.initial_coordinates(voltage_lift=True)
    u1[3 * device.node_count:] += .001
    h0 = HistoryPoint(0.0, u0, old0.storage(u0), old0)
    h1 = HistoryPoint(.25, u1, old1.storage(u1), old1)
    step = BDFStep(new, (h1, h0), .75, 2)
    trial = IonicBackwardEuler1DStep(new.problem, device.state(u1, voltage_V=.003), .5).initial_coordinates(voltage_lift=True)
    trial[3 * device.node_count:] += .002
    direction = np.random.default_rng(590).normal(size=len(trial))
    direction[:device.node_count] *= .01
    eps = 2e-5
    numeric = (step.residual(trial + eps * direction) - step.residual(trial - eps * direction)) / (2 * eps)
    matrix = step.jacobian(trial)
    scale = np.maximum(abs(matrix) @ abs(direction), 1.0)
    assert np.max(abs(numeric - matrix @ direction) / scale) < 3e-8
    data = bdf_accounting(step, trial)
    assert data["bdf_residual"].shape == (device.state_size,)
    current_scale = max(np.max(abs(data["bdf_conduction_current_A_m2"])),
                        np.max(abs(data["bdf_displacement_rate_A_m2"])), 1.0)
    assert np.max(abs(data["maxwell_identity_error_A_m2"])) < 64 * np.finfo(float).eps * current_scale
    endpoint_increment = _displacement(new, trial)[0] - _displacement(old1, u1)[0]
    np.testing.assert_array_equal(data["actual_endpoint_displacement_increment_C_m2"], endpoint_increment)
    np.testing.assert_array_equal(data["bdf_weighted_displacement_increment_C_m2"],
                                  step.dt_s * data["bdf_displacement_rate_A_m2"])
    assert np.max(abs(endpoint_increment - data["bdf_weighted_displacement_increment_C_m2"])) > 1e-12
    np.testing.assert_array_equal(data["interval_displacement_C_m2"],
        data["backward_euler_comparison"]["interval_displacement_C_m2"])
    order1 = BDFStep(new, (h1,), .75, 1)
    read1 = bdf_accounting(order1, trial)
    np.testing.assert_array_equal(read1["interval_displacement_C_m2"],
                                  order1.dt_s * read1["displacement_current_density_A_m2"])


def test_real_electronic_voltage_jump_preserves_inventory_and_restarts(electronic):
    device, initial, admission = electronic
    drive = BDFDrive(np.array([0, 2e-7, 4e-7]), np.array([[0, 1], [0, 1], [.002, 1]]),
                     np.array([[0, 1], [.002, 1], [.002, 1]]))
    protocol = TransientProtocol([0, 2e-7, 4e-7], [0, .002, .002], [1, 1, 1])
    result = solve_transient(device, initial, protocol, options=controls(), initial_options=admission,
        method="bdf", bdf_options=BDFOptions(1e-7, adaptive=False), bdf_drive=drive)
    assert result.completed, (result.termination, result.error)
    assert result.restarts and result.restarts[0].accepted
    assert result.restarts[0].assessment.accounting["relative_inventory_change"] < 1e-11
    assert next(step for step in result.accepted_steps if step.start_s == 2e-7).order == 1


def test_selection_and_unsupported_population_refusals_are_explicit(electronic):
    device, initial, admission = electronic
    protocol = TransientProtocol([0, 1e-7], [0, 0], [1, 1])
    with pytest.raises(ValueError, match="explicit BDFOptions"):
        solve_transient(device, initial, protocol, options=controls(), initial_options=admission, method="bdf")
    with pytest.raises(ValueError, match="unknown"):
        solve_transient(device, initial, protocol, options=controls(), initial_options=admission, method="bogus")


def _numerical_bdf_bytes(result):
    saved = _decode(_encode(result))
    del saved["wall_s"]
    del saved["controls"]["wall_s"]
    del saved["controls"]["initial_admission_wall_s"]
    for attempt in saved["attempts"]:
        del attempt["wall_s"]
    return _canonical(_encode(saved))


@pytest.mark.parametrize("family", ["electronic", "single-ion"])
def test_public_assembled_BDF2_history_is_bitwise_and_uses_only_its_selected_equations(electronic, ionic, family):
    if family == "electronic":
        device, initial, admission = electronic
        request = TransientProtocol([0., 1e-7, 2e-7, 4e-7], [0.] * 4, [1.] * 4)

        def run():
            return solve_transient(device, initial, request, options=controls(), initial_options=admission,
                method="bdf", bdf_options=BDFOptions(1e-7))
    else:
        device, initial = ionic
        request = TransientProtocol([0., .25, .5, 1.], [.002] * 4, [0.] * 4)

        def run():
            return solve_ionic_transient(device, initial, request, options=ionic_controls(), method="bdf",
                bdf_options=BDFOptions(.25))
    results = []
    for implementation in (LEGACY, ASSEMBLED):
        with equation_implementation(implementation), counted() as observed:
            result = run()
        assert result.completed and any(step.order == 2 for step in result.accepted_steps)
        assert (observed.original > 0 and observed.assembled == 0) if implementation == LEGACY else (
            observed.original == 0 and observed.assembled > 0)
        results.append(result)
    assert _numerical_bdf_bytes(results[0]) == _numerical_bdf_bytes(results[1])


def test_public_BDF_does_not_fall_back_after_an_assembled_only_word_change(electronic):
    device, initial, admission = electronic
    request = TransientProtocol([0., 1e-7, 2e-7], [0.] * 3, [1., 1.001, 1.004])

    def run():
        return solve_transient(device, initial, request, options=controls(), initial_options=admission,
            method="bdf", bdf_options=BDFOptions(1e-7, adaptive=False, max_order=1))
    with equation_implementation(LEGACY):
        reference = run()
    with equation_implementation(ASSEMBLED), counted() as observed, one_ulp_residual() as mutations:
        candidate = run()
    assert mutations[0] > 0 and observed.original == 0 and observed.assembled > 0
    assert _numerical_bdf_bytes(reference) != _numerical_bdf_bytes(candidate)


@pytest.mark.parametrize("invalid", [True, 0, -1, 4.0])
def test_explicit_adaptive_work_budget_is_validated_and_keeps_sample_admission(invalid):
    protocol = TransientProtocol([0, 1], [0, 0], [0, 0])
    bdf_options = BDFOptions(.25, max_steps=8)
    with pytest.raises(ValueError, match="explicit positive integer"):
        _adapter_step_budget(protocol, 1, bdf_options, invalid)
    assert _adapter_step_budget(protocol, 1, bdf_options, None) == 1
    assert _adapter_step_budget(protocol, 1, bdf_options, 4) == 4
    assert _adapter_step_budget(protocol, 1, bdf_options, 100) == 8
    extra_samples = TransientProtocol([0, .5, 1], [0, 0, 0], [0, 0, 0])
    with pytest.raises(ValueError, match="sample admission"):
        _adapter_step_budget(extra_samples, 1, bdf_options, 4)


def test_electronic_explicit_work_budget_changes_work_admission_only(electronic):
    device, initial, admission = electronic
    protocol = TransientProtocol([0, 2e-8], [0, 0], [1, 1])
    physical = replace(controls(), max_steps=1)
    bdf_options = BDFOptions(5e-9, adaptive=False, max_order=1, max_steps=8)
    default = solve_electronic_bdf(device, initial, protocol, options=physical,
        initial_options=admission, bdf_options=bdf_options)
    explicit = solve_electronic_bdf(device, initial, protocol, options=physical,
        initial_options=admission, bdf_options=bdf_options, adaptive_max_steps=4)
    assert not default.completed and default.termination == "step_budget_exhausted"
    assert len(default.accepted_steps) == default.controls["max_steps"] == 1
    assert "adapter_adaptive_max_steps" not in default.controls
    assert explicit.completed and len(explicit.accepted_steps) == explicit.controls["max_steps"] == 4
    assert explicit.controls["physical_controls"] == default.controls["physical_controls"]
    assert explicit.controls["adapter_sample_interval_limit"] == 1
    assert explicit.controls["adapter_adaptive_max_steps"] == 4
    for result in (default, explicit):
        for u in result.accepted_coordinates:
            assert u.tobytes() == initial.state.u.tobytes()


def test_single_ion_explicit_work_budget_keeps_the_default_and_physical_checks(ionic):
    device, initial = ionic
    protocol = TransientProtocol([0, .2], [.002, .002], [0, 0])
    original = ionic_controls()
    physical = replace(original, electronic=replace(original.electronic, max_steps=1))
    bdf_options = BDFOptions(.05, adaptive=False, max_order=1, max_steps=8)
    default = solve_ionic_bdf(device, initial, protocol, options=physical, bdf_options=bdf_options)
    explicit = solve_ionic_bdf(device, initial, protocol, options=physical,
        bdf_options=bdf_options, adaptive_max_steps=4)
    assert not default.completed and default.termination == "step_budget_exhausted"
    assert len(default.accepted_steps) == default.controls["max_steps"] == 1
    assert "adapter_adaptive_max_steps" not in default.controls
    assert explicit.completed and len(explicit.accepted_steps) == explicit.controls["max_steps"] == 4
    assert explicit.controls["physical_controls"] == default.controls["physical_controls"]
    assert explicit.controls["adapter_sample_interval_limit"] == 1
    assert explicit.controls["adapter_adaptive_max_steps"] == 4
    assert default.accepted_coordinates[1].tobytes() == explicit.accepted_coordinates[1].tobytes()
    assert all(all(check.passed for check in a.assessment.checks) for a in explicit.accepted_steps)
