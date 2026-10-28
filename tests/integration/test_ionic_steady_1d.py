"""Actual constrained joint ion/carrier solve and independent conservation checks."""

from dataclasses import replace
import math

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from examples.research_1d.ionic_residual import prepare_single_ion_device
from examples.research_1d.ionic_steady import controls
from solarlab.config.resolve_device import resolve_device
from solarlab.device.compile_ionic_1d import compile_ionic_device_1d
from solarlab.device.inputs import DeviceInput
from solarlab.equations.dc_1d import CapabilityError
from solarlab.equations.ionic_1d import Ionic1DProblem
from solarlab.physics.ions import WHOLE_FLUX_STERIC_V1
from solarlab.solvers.nonlinear.ionic_1d import ConstrainedIonic1DProblem, solve_ionic_steady_state
from solarlab.solvers.nonlinear.problem import SolveBudget
from solarlab.solvers.nonlinear.production import POISSON_CONTACT_BALANCE_V1


@pytest.fixture(scope="module", autouse=True)
def blas1():
    with threadpool_limits(limits=1):
        yield


@pytest.fixture(scope="module")
def case():
    prepared = prepare_single_ion_device(diffusion_only=True)
    device = compile_ionic_device_1d(prepared, intervals_per_layer=12)
    original = Ionic1DProblem(device, voltage_V=0.002)
    problem = ConstrainedIonic1DProblem(original, device.node_count - 1)
    initial = device.initial_state(voltage_V=0.002)
    return prepared, device, original, problem, initial


def trial(case):
    _, device, _, _, initial = case
    count = device.node_count
    position = (device.nodes_m - device.nodes_m[0]) / device.ion_geometry.extent_m
    y = initial.u.copy()
    y[:count] += 0.001 * np.sin(np.pi * position)
    y[count:2 * count] += 0.02 * np.sin(np.pi * position)
    y[2 * count:3 * count] -= 0.03 * np.sin(np.pi * position)
    y[3 * count:] += 0.04 * np.cos(np.pi * position)
    return y


def test_constraint_uses_declared_physical_inventory_and_changes_only_one_row(case):
    prepared, device, original, problem, initial = case
    y = trial(case)
    original_rows = original.residual(y)
    constrained = problem.residual(y)
    row = problem.constraint_row
    other = np.arange(y.size) != row
    np.testing.assert_array_equal(constrained[other], original_rows[other])
    density = device.fields(y, voltage_V=0.002)[3]
    target = math.fsum((device.ion_geometry.cell_measures_m * device.positive_ion_reference_m3).tolist())
    assert problem.inventory_target_m2 == target == initial.domain_inventory_m2
    assert device.original_prepared_sha256 == prepared.content_sha256
    assert constrained[row] == math.fsum((device.ion_geometry.cell_measures_m * density).tolist()) - target
    wrong_integer_inventory = np.dot(device.electronic_component.geometry.cell_measures_m, device.positive_ion_reference_m3)
    assert abs(wrong_integer_inventory - target) / target > 1e-3
    assert abs(math.fsum(original_rows[3 * device.node_count:].tolist())) < 1000.0


def test_constrained_J_has_exact_inventory_row_and_coupled_cross_blocks(case):
    _, device, original, problem, _ = case
    y = trial(case)
    count, row = device.node_count, problem.constraint_row
    original_J, constrained = original.jacobian(y), problem.jacobian(y)
    other = np.arange(y.size) != row
    np.testing.assert_array_equal(constrained.toarray()[other], original_J.toarray()[other])
    np.testing.assert_array_equal(constrained.getrow(row).toarray()[0, :3 * count], np.zeros(3 * count))
    expected = -original.storage_jacobian(y).diagonal()[3 * count:]
    np.testing.assert_array_equal(constrained.getrow(row).toarray()[0, 3 * count:], expected)
    assert constrained[:count, 3 * count:].nnz > 0
    assert constrained[3 * count:, :count].nnz > 0


def test_full_constrained_directional_J_is_independent_of_the_replaced_residual(case):
    _, device, _, problem, _ = case
    y = trial(case)
    direction = np.random.default_rng(815).uniform(-0.1, 0.1, y.size)
    direction[:device.node_count] *= device.electronic_component.thermal_voltage_V
    h = 1e-5
    numerical = (problem.residual(y + h * direction) - problem.residual(y - h * direction)) / (2 * h)
    matrix = problem.jacobian(y)
    scale = np.maximum(abs(matrix) @ abs(direction), np.finfo(float).tiny)
    assert np.max(abs(numerical - matrix @ direction) / scale) < 3e-7


@pytest.fixture(scope="module")
def solved(case):
    _, device, _, _, _ = case
    return solve_ionic_steady_state(device, voltage_V=0.002, generation_scale=0.0,
                                    options=controls(device.node_count))


def test_actual_joint_solution_moves_ions_and_retains_all_original_balances(case, solved):
    _, device, original, problem, initial = case
    assert solved.accepted, (solved.termination, solved.error)
    assert not solved.qualification and solved.newton.accepted
    assert solved.state.u.size == 4 * device.node_count
    assert all(check.passed for check in solved.checks)
    np.testing.assert_array_equal(solved.initial_state.u, initial.u)
    np.testing.assert_array_equal(solved.state.u, solved.newton.state)
    assert np.max(abs(solved.state.positive_ion_density_m3 / device.positive_ion_reference_m3 - 1)) > 0.01
    assert np.max(abs(solved.state.electronic_state.potential_V - initial.electronic_state.potential_V)) > 1e-4
    assert np.all(solved.state.positive_ion_density_m3 > 0)
    assert np.all(solved.state.positive_ion_density_m3 < device.positive_ion_site_limit_m3)
    a = solved.diagnostics
    np.testing.assert_array_equal(a["original_residual"], original.residual(solved.state.u))
    np.testing.assert_array_equal(a["constrained_residual"], problem.residual(solved.state.u))
    assert a["ion_balance_l1_m2_s"] <= 1e6 and a["ion_flux_max_m2_s"] <= 1e6
    assert abs(a["original_residual"][problem.constraint_row]) <= 1e6
    assert a["inventory_relative_error"] <= 1e-12
    assert a["current_spread_A_m2"] <= 1e-8
    assert a["carrier_balance_A_m2"] <= 1e-8
    np.testing.assert_allclose(a["deposited_ion_net_charge_C_m2"], a["physical_ion_net_charge_C_m2"],
                               rtol=5e-15, atol=1e-20)


def test_freezing_ions_does_not_meet_the_same_original_problem(case, solved):
    _, device, original, _, initial = case
    frozen = solved.state.u.copy()
    frozen[3 * device.node_count:] = initial.u[3 * device.node_count:]
    assert np.max(abs(original.ion_fluxes(frozen))) > 1e6
    assert np.max(abs(original.residual(frozen)[:device.node_count])) > 1e-10


def test_wrong_initial_inventory_and_electronic_only_state_are_refused(case):
    _, device, _, _, initial = case
    wrong = initial.u.copy()
    wrong[3 * device.node_count:] += 0.01
    wrong_state = device.state(wrong, voltage_V=0.002)
    with pytest.raises(ValueError, match="prepared-domain inventory"):
        solve_ionic_steady_state(device, voltage_V=0.002, generation_scale=0.0,
                                 options=controls(device.node_count), initial_state=wrong_state)
    with pytest.raises(ValueError, match="full same-device"):
        solve_ionic_steady_state(device, voltage_V=0.002, generation_scale=0.0,
                                 options=controls(device.node_count), initial_state=initial.electronic_state)


def test_mismatched_component_or_policy_binding_is_refused(case):
    _, device, _, _, initial = case
    changed = replace(device, steric_model=WHOLE_FLUX_STERIC_V1)
    with pytest.raises(ValueError, match="provenance"):
        ConstrainedIonic1DProblem(Ionic1DProblem(changed), changed.node_count - 1)
    wrong_component = replace(initial.electronic_state, device_sha256="not-the-compiled-component")
    wrong = replace(initial, electronic_state=wrong_component)
    with pytest.raises(ValueError, match="full same-device"):
        solve_ionic_steady_state(device, voltage_V=0.002, generation_scale=0.0,
                                 options=controls(device.node_count), initial_state=wrong)


def test_zero_diffusion_is_explicitly_underdetermined_for_one_constraint():
    prepared = prepare_single_ion_device(diffusion_only=True)
    data = prepared.to_input().editing_data()
    data["layers"][0]["parameters"]["D_ion"] = 0.0
    changed = resolve_device(DeviceInput.model_validate(data), prepared.defaults,
                              prepared.resources, sources=prepared.sources)
    device = compile_ionic_device_1d(changed, intervals_per_layer=12)
    assert device.reference_inventory_m2 > 0.0
    with pytest.raises(CapabilityError, match="positive-D"):
        solve_ionic_steady_state(device, voltage_V=0.002, generation_scale=0.0, options=controls(device.node_count))


def test_failed_bounded_attempt_is_retained_without_projection_or_retry(case):
    _, device, _, _, initial = case
    bounded = replace(controls(device.node_count), budget=SolveBudget(1, 4, 1, 2))
    result = solve_ionic_steady_state(device, voltage_V=0.002, generation_scale=0.0, options=bounded)
    assert not result.accepted and not result.qualification
    assert result.newton.iteration_count == 1 and result.newton.jacobian_evaluations == 1
    np.testing.assert_array_equal(result.initial_state.u, initial.u)
    np.testing.assert_array_equal(result.candidate_coordinates, result.newton.state)
    assert result.diagnostics is not None and result.state is not None


def test_one_wall_budget_keeps_the_unaccepted_initial_state(case, monkeypatch):
    _, device, _, _, initial = case
    ticks = iter(range(100))
    monkeypatch.setattr("solarlab.solvers.nonlinear.ionic_1d.time.monotonic", lambda: next(ticks))
    bounded = replace(controls(device.node_count), wall_s=0.5)
    result = solve_ionic_steady_state(device, voltage_V=0.002, generation_scale=0.0, options=bounded)
    assert not result.accepted and result.termination == "wall_budget_exhausted"
    assert result.newton is None
    np.testing.assert_array_equal(result.candidate_coordinates, initial.u)


def test_unsupported_convergence_policy_is_not_silently_reported():
    configured = controls(13)
    alternative = replace(configured.electronic, convergence_policy=POISSON_CONTACT_BALANCE_V1)
    with pytest.raises(ValueError, match="only historical_max_row_v1"):
        replace(configured, electronic=alternative)


@pytest.mark.parametrize("field", ["positive_ion_site_limit_m3", "positive_ion_logit_reference"])
def test_rebound_capacity_or_logit_is_rejected_by_the_compiled_boundary(case, field):
    _, device, _, _, _ = case
    changed = getattr(device, field) * 2 if field == "positive_ion_site_limit_m3" else getattr(device, field) + 0.4
    with pytest.raises(ValueError):
        replace(device, **{field: changed})
