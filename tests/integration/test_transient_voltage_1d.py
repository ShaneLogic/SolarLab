"""Variable-bias physical secants, algebraic seeds and bounded real histories."""

from dataclasses import asdict
from decimal import Decimal, localcontext
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from examples.research_1d.steady import load_device
from examples.research_1d.transient import controls, json_value
from examples.research_1d.transient_voltage import voltage_protocol
from solarlab.device.compile_1d import compile_device_1d
from solarlab.equations.dc_1d import DC1DProblem
from solarlab.experiments.transient import TransientProtocol, solve_transient
from solarlab.solvers.nonlinear.steady_1d import SteadyOptions, solve_steady_state
from solarlab.solvers.time import BackwardEuler1DStep, inventory_m2


@pytest.fixture(scope="module", autouse=True)
def blas1():
    with threadpool_limits(limits=1):
        yield


@pytest.fixture(scope="module")
def case():
    device = compile_device_1d(load_device(), intervals_per_layer=12)
    initial_options = SteadyOptions()
    point = solve_steady_state(device, voltage_V=0.1, generation_scale=1.0, options=initial_options)
    return device, point, initial_options


def test_protocol_keeps_every_voltage_sample_and_owns_its_arrays():
    times = np.asarray([0.0, 1e-8, 3e-8])
    voltages = np.asarray([0.1, 0.101, 0.1005])
    p = TransientProtocol(times, voltages, [1.0, 1.0, 1.0])
    times[:] = 5.0
    voltages[:] = 2.0
    np.testing.assert_array_equal(p.times_s, [0.0, 1e-8, 3e-8])
    np.testing.assert_array_equal(p.voltage_V, [0.1, 0.101, 0.1005])
    assert not p.voltage_V.flags.writeable


@pytest.mark.parametrize("voltage_change", [1e-3, -1e-3])
def test_harmonic_lift_preserves_physical_populations_and_contact_constraints(case, voltage_change):
    device, point, _ = case
    target = point.voltage_V + voltage_change
    step = BackwardEuler1DStep(DC1DProblem(device, target, 1.0), point.state, 1e-7)
    before = point.state.u.copy()
    seed = step.initial_coordinates(voltage_lift=True)
    phi, n, p = device.fields(seed, voltage_V=target)
    np.testing.assert_allclose(n, point.state.electron_density_m3, rtol=3e-14)
    np.testing.assert_allclose(p, point.state.hole_density_m3, rtol=3e-14)
    np.testing.assert_allclose(phi[[0, -1]], device.contact_potentials(target), rtol=0, atol=3e-15)
    resistance = 1.0 / device.poisson.face_capacitance_F_m2
    profile = np.r_[0.0, np.cumsum(resistance)] / np.sum(resistance)
    expected = -device.voltage_polarity * voltage_change * profile
    np.testing.assert_allclose(phi - point.state.potential_V, expected, rtol=0, atol=4e-16)
    poisson = device.poisson.jacobian()
    defect = poisson @ (phi - point.state.potential_V)
    rounding_scale = np.finfo(float).eps * (abs(poisson) @ (abs(phi) + abs(point.state.potential_V)))
    assert np.all(abs(defect[1:-1]) <= 64 * rounding_scale[1:-1])
    count = device.node_count
    for carrier, offset in (("n", count), ("p", 2 * count)):
        for side, node in (("left", 0), ("right", count - 1)):
            if device.channel(carrier, side).kind == "fixed_reservoir":
                assert abs(step.residual(seed)[offset + node]) < 3e-14
    np.testing.assert_array_equal(point.state.u, before)
    assert step.previous is point.state


def test_variable_reference_secant_matches_70_digit_physical_difference(case):
    device, point, _ = case
    step = BackwardEuler1DStep(DC1DProblem(device, 0.102, 1.0), point.state, 1e-7)
    u = step.initial_coordinates(voltage_lift=True)
    count = device.node_count
    u[count:2 * count] += 0.013
    u[2 * count:] -= 0.019
    rate = step.inventory_rate_m2_s(u)
    old_inventory = inventory_m2(device, point.state)
    old_ref = device.carrier_reference_energies(point.voltage_V)
    new_ref = device.carrier_reference_energies(step.endpoint.voltage_V)
    with localcontext() as context:
        context.prec = 70
        D = Decimal.from_float
        for species, sign in ((0, 1), (1, -1)):
            offset = (species + 1) * count
            for node in (0, 11, 18, 25, count - 1):
                exponent = sign * ((D(float(u[node])) - D(float(point.state.u[node]))
                                   + D(new_ref[species]) - D(old_ref[species])) / D(device.thermal_voltage_V)
                                  + D(float(u[offset + node])) - D(float(point.state.u[offset + node])))
                expected = D(float(old_inventory[species, node])) * (exponent.exp() - 1) / D(step.dt_s)
                assert rate[species, node] == pytest.approx(float(expected), rel=2e-13)
    # Omitting the reference change or adding a separate voltage-rate term
    # would materially change this physical inventory difference.
    delta = u - point.state.u
    wrong_n = old_inventory[0] * np.expm1(delta[:count] / device.thermal_voltage_V
                                         + delta[count:2 * count]) / step.dt_s
    assert np.max(abs(wrong_n - rate[0]) / np.maximum(abs(rate[0]), 1.0)) > 1.0


def test_variable_bias_jacobian_is_the_new_endpoint_analytic_jacobian(case):
    device, point, _ = case
    step = BackwardEuler1DStep(DC1DProblem(device, 0.101, 1.0), point.state, 1e-7)
    u = step.initial_coordinates(voltage_lift=True)
    direction = np.random.default_rng(72).normal(size=u.size)
    direction[:device.node_count] *= 0.01
    h = 2e-5
    numerical = (step.residual(u + h * direction) - step.residual(u - h * direction)) / (2 * h)
    matrix = step.jacobian(u)
    scale = np.maximum(abs(matrix) @ abs(direction), 1.0)
    assert np.max(abs(numerical - matrix @ direction) / scale) < 3e-8
    np.testing.assert_array_equal(matrix.toarray(), (step.endpoint.jacobian(u)
                                                   + step.endpoint.storage_jacobian(u) / step.dt_s).toarray())


@pytest.fixture(scope="module")
def runs(case):
    device, point, initial_options = case
    return tuple(solve_transient(device, point, voltage_protocol(n), options=controls(), initial_options=initial_options)
                 for n in (4, 8, 16))


def test_real_variable_bias_histories_and_maxwell_refinement(case, runs):
    device, point, _ = case
    for count, result in zip((4, 8, 16), runs):
        assert result.completed, [(a.termination, a.error) for a in result.failed_attempts]
        assert len(result.states) == count + 1 and not result.failed_attempts
        requested = voltage_protocol(count)
        np.testing.assert_array_equal(result.accepted_times_s, requested.times_s)
        np.testing.assert_array_equal([s.voltage_V for s in result.states], requested.voltage_V)
        assert result.states[0] is point.state
        assert result.physical_certificate_status == "not_evaluated"
        assert result.solver_controls["voltage_seed_policy"] == "harmonic_potential_preserve_densities_when_voltage_changes"
        for k, attempt in enumerate(result.attempts, start=1):
            a = attempt.accounting
            assert attempt.voltage_V == requested.voltage_V[k]
            assert attempt.newton.gates == ()
            np.testing.assert_allclose(result.states[k].potential_V[[0, -1]],
                                       device.contact_potentials(requested.voltage_V[k]), atol=1e-11, rtol=0)
            assert a["continuity_defect_bound_A_m2"] <= controls().continuity_tolerance_A_m2
            assert a["maxwell_spread_A_m2"] < 1e-8
            assert np.max(abs(a["maxwell_identity_error_A_m2"])) < 1e-10
        last = result.attempts[-1].accounting
        assert np.max(abs(last["displacement_current_density_A_m2"])) > 0.1
        assert np.max(abs(last["terminal_displacement_current_density_A_m2"])) > 100 * controls().continuity_tolerance_A_m2
        assert np.max(abs(last["dc_residual"][device.node_count:])) > 1e12
    inventories = [inventory_m2(device, r.states[-1]) for r in runs]
    currents = [r.attempts[-1].accounting["terminal_total_current_density_A_m2"] for r in runs]
    for values in (inventories, currents):
        errors = [np.max(abs(a - b)) for a, b in zip(values[:-1], values[1:])]
        assert 0 < errors[1] < errors[0]
        assert 1.8 < errors[0] / errors[1] < 2.2


def test_fixed_bias_seed_branch_is_exactly_unchanged(case):
    device, point, _ = case
    step = BackwardEuler1DStep(DC1DProblem(device, point.voltage_V, 1.01), point.state, 1e-7)
    for predictor in ("previous", "generation_inventory"):
        old = step.initial_coordinates(predictor)
        new = step.initial_coordinates(predictor, voltage_lift=True)
        assert old.tobytes() == new.tobytes()


def test_fixed_bias_history_matches_accepted_archive_exactly():
    path = os.environ.get("SOLARLAB_FIXED_TRANSIENT_RESULT")
    if not path:
        pytest.skip("explicit fixed-bias acceptance artifact required for exact history witness")
    raw = Path(path).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == "c0f4f36928eb3583348c4b7d8e1b66660dedc5968c16bdad748436300240373b"
    old = json.loads(raw)["results"][0]
    device = compile_device_1d(load_device(), intervals_per_layer=12)
    initial_options = SteadyOptions()
    initial = solve_steady_state(device, voltage_V=0.0, generation_scale=1.0, options=initial_options)
    p = TransientProtocol(**old["protocol"])
    result = solve_transient(device, initial, p, options=controls(), initial_options=initial_options)
    assert result.completed
    np.testing.assert_array_equal(result.accepted_times_s, old["accepted_times_s"])
    assert result.solver_controls == old["solver_controls"]
    for new_state, old_state in zip(result.states, old["states"], strict=True):
        for name in ("u", "potential_V", "electron_density_m3", "hole_density_m3"):
            assert getattr(new_state, name).tobytes() == np.asarray(old_state[name], dtype=float).tobytes()
    for new_attempt, old_attempt in zip(result.attempts, old["attempts"], strict=True):
        assert json_value(asdict(new_attempt.newton)) == old_attempt["newton"]
        for name, value in new_attempt.accounting.items():
            old_value = old_attempt["accounting"][name]
            if isinstance(value, np.ndarray):
                assert value.tobytes() == np.asarray(old_value, dtype=value.dtype).tobytes(), name
            else:
                assert value == old_value, name
