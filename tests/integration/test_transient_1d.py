"""Physical BE storage, analytic tangents, accounting and actual small steps."""

from copy import copy
from dataclasses import replace
from decimal import Decimal, localcontext

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from examples.research_1d.steady import load_device
from examples.research_1d.transient import controls
from solarlab.config.resolve_device import resolve_device
from solarlab.device.compile_1d import compile_device_1d
from solarlab.device.inputs import DeviceInput
from solarlab.equations.dc_1d import DC1DProblem
from solarlab.experiments.transient import TransientProtocol, solve_transient
from solarlab.solvers.nonlinear.problem import SolveBudget
from solarlab.solvers.nonlinear.steady_1d import SteadyOptions, solve_equilibrium, solve_steady_state
from solarlab.solvers.time import BackwardEuler1DStep, inventory_m2, step_accounting


@pytest.fixture(scope="module", autouse=True)
def blas1():
    with threadpool_limits(limits=1):
        yield


@pytest.fixture(scope="module")
def case():
    compiled = compile_device_1d(load_device(), intervals_per_layer=12)
    initial_controls = SteadyOptions()
    point = solve_steady_state(compiled, voltage_V=0.0, generation_scale=1.0, options=initial_controls)
    return compiled, point, initial_controls


def protocol(steps=4):
    times = np.linspace(0, 4e-7, steps + 1)
    return TransientProtocol(times, np.zeros(steps + 1), 1.0 + 0.01 * (times / times[-1]) ** 2)


def trial(device, point):
    u = point.state.u.copy()
    count = device.node_count
    u[:count] += np.linspace(-2e-4, 3e-4, count)
    u[count:2 * count] += np.linspace(0.12, 0.24, count)
    u[2 * count:] -= np.linspace(0.21, 0.09, count)
    return u


def test_storage_is_physical_inventory_and_algebraic_rows_are_unchanged(case):
    device, point, _ = case
    step = BackwardEuler1DStep(DC1DProblem(device, 0.0, 1.01), point.state, 1e-7)
    u, count = trial(device, point), device.node_count
    new = device.state(u)
    actual = step.inventory_rate_m2_s(u)
    direct = (inventory_m2(device, new) - inventory_m2(device, point.state)) / step.dt_s
    np.testing.assert_allclose(actual, direct, rtol=2e-13)
    mask = np.ones((2, count), dtype=bool)
    for carrier, species in (("n", 0), ("p", 1)):
        for side, node in (("left", 0), ("right", count - 1)):
            if device.channel(carrier, side).kind == "fixed_reservoir":
                mask[species, node] = False
    expected = step.endpoint.residual(u)
    expected[count:].reshape(2, count)[mask] -= actual[mask]
    np.testing.assert_array_equal(step.residual(u), expected)
    original = step.endpoint.residual(u)
    algebraic = np.r_[np.ones(count, dtype=bool), (~mask).ravel()]
    np.testing.assert_array_equal(step.residual(u)[algebraic], original[algebraic])
    # A DAE call with a coordinate difference is a different method.
    shortcut = step.endpoint.residual(u, (u - point.state.u) / step.dt_s)
    assert np.max(np.abs(step.residual(u) - shortcut)) > 1e15


def test_nonlinear_secant_matches_high_precision_and_resolves_tiny_changes(case):
    device, point, _ = case
    step = BackwardEuler1DStep(DC1DProblem(device, 0.0, 1.0), point.state, 1e-7)
    node = device.node_count - 5
    u = point.state.u.copy()
    u[device.node_count + node] += 2e-12
    dlog = u[device.node_count + node] - point.state.u[device.node_count + node]
    with localcontext() as context:
        context.prec = 70
        reference = (Decimal.from_float(float(inventory_m2(device, point.state)[0, node]))
                     * (Decimal.from_float(float(dlog)).exp() - 1) / Decimal.from_float(step.dt_s))
    assert step.inventory_rate_m2_s(u)[0, node] == pytest.approx(float(reference), rel=3e-16)
    np.testing.assert_array_equal(step.inventory_rate_m2_s(point.state.u), np.zeros((2, device.node_count)))


def test_analytic_step_jacobian_matches_directional_residual(case):
    device, point, _ = case
    step = BackwardEuler1DStep(DC1DProblem(device, 0.0, 1.01), point.state, 1e-7)
    u = trial(device, point)
    direction = np.random.default_rng(23).normal(size=u.size)
    direction[:device.node_count] *= 0.01
    h = 2e-5
    numerical = (step.residual(u + h * direction) - step.residual(u - h * direction)) / (2 * h)
    matrix = step.jacobian(u)
    scale = np.maximum(abs(matrix) @ abs(direction), 1.0)
    assert np.max(abs(numerical - matrix @ direction) / scale) < 3e-8
    np.testing.assert_array_equal(matrix.toarray(), (step.endpoint.jacobian(u)
                                                   + step.endpoint.storage_jacobian(u) / step.dt_s).toarray())


def test_fixed_bias_secant_retains_tiny_potential_increment(case):
    device, point, _ = case
    rebased = device.rebase_coordinates(point.state.u, point.voltage_V, 1.0)
    old = device.state(rebased, voltage_V=1.0)
    step = BackwardEuler1DStep(DC1DProblem(device, 1.0, 1.0), old, 1e-7)
    u = old.u.copy()
    u[0] += 1e-18
    delta_phi = u[0] - old.u[0]
    refs = device.carrier_reference_energies(1.0)
    species = 0 if refs[0] != 0.0 else 1
    sign = 1.0 if species == 0 else -1.0
    # This wrongly associated voltage expression is exactly zero in binary64.
    assert (delta_phi + refs[species]) - refs[species] == 0.0
    expected = inventory_m2(device, old)[species, 0] * np.expm1(sign * delta_phi / device.thermal_voltage_V) / step.dt_s
    actual = step.inventory_rate_m2_s(u)[species, 0]
    assert actual != 0.0
    assert actual == pytest.approx(expected, rel=3e-16, abs=0.0)


def test_voltage_reference_rebase_preserves_the_old_physical_fields(case):
    device, point, _ = case
    step = BackwardEuler1DStep(DC1DProblem(device, 0.2, 1.01), point.state, 1e-7)
    u = step.initial_coordinates()
    phi, n, p = device.fields(u, voltage_V=0.2)
    np.testing.assert_array_equal(phi, point.state.potential_V)
    np.testing.assert_allclose(n, point.state.electron_density_m3, rtol=2e-14)
    np.testing.assert_allclose(p, point.state.hole_density_m3, rtol=2e-14)
    rate = step.inventory_rate_m2_s(u)
    scaled = abs(rate) * step.dt_s / inventory_m2(device, point.state)
    assert np.max(scaled) < 5e-14
    with pytest.raises(ValueError, match="variable-voltage"):
        TransientProtocol([0, 1e-7], [0, 0.2], [1, 1.01])


@pytest.mark.parametrize("member", ["times_s", "voltage_V", "generation_scale"])
def test_complex_protocol_arrays_are_rejected(member):
    fields = dict(times_s=[0, 1], voltage_V=[0, 0], generation_scale=[1, 1])
    fields[member] = np.asarray(fields[member], dtype=complex) + 1j
    with pytest.raises(ValueError, match="real"):
        TransientProtocol(**fields)


@pytest.mark.parametrize("times", [[0, 0], [1, 0], [0], [-1e308, 1e308]])
def test_times_must_have_finite_positive_intervals(times):
    with pytest.raises(ValueError, match="strictly increasing"):
        TransientProtocol(times, np.zeros(len(times)), np.ones(len(times)))


def test_fixed_endpoint_inventory_and_gauss_extension_are_retained(case):
    device, point, _ = case
    step = BackwardEuler1DStep(DC1DProblem(device, 0.0, 1.01), point.state, 1e-7)
    a = step_accounting(step, trial(device, point))
    particles, source, rate = a["particle_fluxes_m2_s"], a["net_pair_source_m2_s"], a["inventory_rate_m2_s"]
    np.testing.assert_allclose(particles[:, 0], particles[:, 1] - source[0] + rate[:, 0], rtol=3e-16)
    np.testing.assert_allclose(particles[:, -1], particles[:, -2] + source[-1] - rate[:, -1], rtol=3e-16)
    assert np.max(abs(rate[:, [0, -1]])) > 0
    assert np.max(abs(a["terminal_conduction_current_density_A_m2"]
                      - step.endpoint.contact_current_density(trial(device, point)))) > 0
    # The predicted jump includes old/new Gauss defects and recorded
    # roundoff from the independently stored charge/inventory subtraction.
    scale = max(np.max(abs(a["conduction_current_density_A_m2"])),
                np.max(abs(a["displacement_current_density_A_m2"])), 1.0)
    assert np.max(abs(a["maxwell_identity_error_A_m2"])) < 64 * np.finfo(float).eps * scale
    np.testing.assert_allclose(a["global_particle_balance_m2_s"], -np.sum(a["carrier_balance_m2_s"], axis=1),
                               rtol=3e-14, atol=1000)


def test_finite_and_blocking_contact_rows_keep_storage():
    prepared = load_device()
    document = prepared.to_input().editing_data()
    document["contacts"] = [dict(id="left", side="left", layer="htl", S_n=0.0, S_p=200.0),
                            dict(id="right", side="right", layer="etl", S_n=200.0, S_p=0.0)]
    prepared = resolve_device(DeviceInput.model_validate(document), prepared.defaults,
                              prepared.resources, sources=prepared.sources)
    device = compile_device_1d(prepared, intervals_per_layer=12)
    point = solve_equilibrium(device)
    step = BackwardEuler1DStep(DC1DProblem(device, 0.0, 1e-4), point.state, 1e-7)
    u = trial(device, point)
    a = step_accounting(step, u)
    assert np.all(a["storage_mask"])
    assert a["particle_fluxes_m2_s"][0, 0] == a["particle_fluxes_m2_s"][1, -1] == 0.0
    count = device.node_count
    for row in (count, 2 * count - 1, 2 * count, 3 * count - 1):
        assert step.endpoint.storage_jacobian(u).getrow(row).nnz > 0
    np.testing.assert_allclose(a["carrier_balance_m2_s"].ravel(), step.residual(u)[count:], rtol=2e-14)


def test_generation_predictor_is_explicit_and_does_not_modify_old_state(case):
    device, point, _ = case
    before = point.state.u.copy()
    step = BackwardEuler1DStep(DC1DProblem(device, 0.0, 1.01), point.state, 1e-7)
    guess = step.initial_coordinates("generation_inventory")
    fields = device.fields(guess)
    mask = step_accounting(step, guess)["storage_mask"]
    for species, density in enumerate((point.state.electron_density_m3, point.state.hole_density_m3)):
        expected = density + np.where(mask[species], step.dt_s * 1.01 * device.generation.source_m3_s, 0)
        np.testing.assert_allclose(fields[species + 1], expected, rtol=3e-14)
    np.testing.assert_array_equal(point.state.u, before)
    assert step.previous is point.state


@pytest.fixture(scope="module")
def refinements(case):
    device, point, initial_controls = case
    return tuple(solve_transient(device, point, protocol(n), options=controls(), initial_options=initial_controls)
                 for n in (4, 8, 16))


def test_real_37_node_time_refinement_and_maxwell_accounting(case, refinements):
    device, point, _ = case
    assert device.node_count == 37
    for steps, result in zip((4, 8, 16), refinements):
        assert result.completed, [(a.termination, a.error, a.newton.message if a.newton else None) for a in result.failed_attempts]
        assert len(result.states) == steps + 1 and not result.failed_attempts
        np.testing.assert_array_equal(result.accepted_times_s, protocol(steps).times_s)
        assert result.states[0] is point.state
        assert result.physical_certificate_status == "not_evaluated"
        for attempt in result.attempts:
            assert attempt.newton.gates == ()  # No inappropriate DC-current gate.
            a = attempt.accounting
            assert a["continuity_defect_bound_A_m2"] <= controls().continuity_tolerance_A_m2
            bound = (a["continuity_defect_bound_A_m2"] + np.sum(abs(a["gauss_defect_rate_A_m2"]))
                     + np.sum(abs(a["charge_subtraction_roundoff_A_m2"])))
            assert a["maxwell_spread_A_m2"] <= bound + 1e-10
            assert np.max(abs(a["maxwell_identity_error_A_m2"])) < 1e-10
    final = [inventory_m2(device, r.states[-1]) for r in refinements]
    differences = [np.max(abs(a-b)) for a, b in zip(final[:-1], final[1:])]
    assert 0 < differences[1] < differences[0]
    assert np.max(abs(final[-1] - inventory_m2(device, point.state))) > 1e6


def test_failed_attempt_keeps_successful_prefix_and_has_no_retry(case):
    device, point, initial_controls = case
    p = TransientProtocol([0, 1e-7, 2e-7, 3e-7], [0, 0, 0, 0], [1, 1, 1.5, 1])
    bounded = replace(controls(), newton_budget=SolveBudget(1, 10, 1, 4))
    result = solve_transient(device, point, p, options=bounded, initial_options=initial_controls)
    assert not result.completed and len(result.attempts) == 2
    np.testing.assert_array_equal(result.accepted_times_s, [0, 1e-7])
    assert len(result.states) == 2 and len(result.failed_attempts) == 1
    failed = result.failed_attempts[0]
    assert failed.end_s == 2e-7 and not failed.newton.accepted
    assert failed.candidate_coordinates.shape == point.state.u.shape


def test_deadline_is_one_budget_and_retains_failed_attempt(case, monkeypatch):
    device, point, initial_controls = case
    ticks = iter(range(1000))
    monkeypatch.setattr("solarlab.experiments.transient.time.monotonic", lambda: next(ticks))
    result = solve_transient(device, point, protocol(), options=replace(controls(), wall_s=0.5),
                             initial_options=initial_controls)
    assert len(result.states) == 1 and len(result.attempts) == 1
    assert result.failed_attempts[0].termination == "wall_budget_exhausted"
    assert result.failed_attempts[0].newton is None


def test_unaccepted_or_foreign_initial_and_overbudget_protocol_are_rejected(case):
    device, point, initial_controls = case
    failed = copy(point)
    object.__setattr__(failed, "newton", replace(point.newton, accepted=False))
    with pytest.raises(ValueError, match="accepted"):
        solve_transient(device, failed, protocol(), options=controls(), initial_options=initial_controls)
    with pytest.raises(ValueError, match="identity"):
        solve_transient(compile_device_1d(load_device(), intervals_per_layer=8), point, protocol(),
                         options=controls(), initial_options=initial_controls)
    with pytest.raises(ValueError, match="max_steps"):
        solve_transient(device, point, protocol(), options=replace(controls(), max_steps=1), initial_options=initial_controls)
