"""Actual constrained joint ion/carrier solve and independent conservation checks."""

from dataclasses import replace
import hashlib
import json
import math

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from examples.research_1d.ionic_residual import prepare_single_ion_device
from examples.research_1d.ionic_steady import controls
from solarlab.config.resolve_device import resolve_device
from solarlab.contracts.units import ELEMENTARY_CHARGE_C as Q
from solarlab.device.compile_ionic_1d import compile_ionic_device_1d
from solarlab.device.inputs import DeviceInput
from solarlab.equations.dc_1d import CapabilityError
from solarlab.equations.ionic_1d import Ionic1DProblem
from solarlab.physics.ions import WHOLE_FLUX_STERIC_V1
from solarlab.solvers.nonlinear.ionic_1d import ConstrainedIonic1DProblem, _diagnostics, solve_ionic_steady_state
from solarlab.solvers.nonlinear.problem import SolveBudget
from solarlab.solvers.nonlinear.production import (
    POISSON_CONTACT_BALANCE_V1, production_coordinates, production_residual_scaling,
)


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


@pytest.fixture(scope="module")
def multilayer_case():
    from examples.research_1d.ionic_multilayer import prepare_confined_ion_device

    prepared = prepare_confined_ion_device(diffusion_only=True)
    device = compile_ionic_device_1d(prepared, intervals_per_layer=(12, 12, 12),
                                     ion_domain_layer="absorber",
                                     ion_domain_endpoints=("blocking", "blocking"))
    original = Ionic1DProblem(device, voltage_V=0.002)
    problem = ConstrainedIonic1DProblem(original, device.ion_node_count - 1)
    initial = device.initial_state(voltage_V=0.002)
    return prepared, device, original, problem, initial


def multilayer_trial(case):
    _, device, _, _, initial = case
    count = device.node_count
    position = (device.nodes_m - device.nodes_m[0]) / np.ptp(device.nodes_m)
    ion_position = (device.ion_geometry.nodes_m - device.ion_geometry.nodes_m[0]) / device.ion_geometry.extent_m
    y = initial.u.copy()
    y[:count] += 0.001 * np.sin(np.pi * position)
    y[count:2 * count] += 0.02 * np.sin(np.pi * position)
    y[2 * count:3 * count] -= 0.03 * np.sin(np.pi * position)
    y[3 * count:] += 0.04 * np.cos(np.pi * ion_position)
    return y


def test_multilayer_constraint_preserves_all_other_rows_and_exact_inventory_J(multilayer_case):
    prepared, device, original, problem, initial = multilayer_case
    count, ion_count = device.node_count, device.ion_node_count
    assert (count, ion_count, device.state_size) == (37, 13, 124)
    assert initial.u.shape == (3 * count + ion_count,)
    assert initial.electronic_state.u.shape == (3 * count,)
    np.testing.assert_array_equal(device.ion_node_indices, np.arange(12, 25))
    assert device.original_prepared_sha256 == prepared.content_sha256
    assert problem.constraint_row == 3 * count + ion_count - 1
    with pytest.raises(ValueError, match="ion-domain node"):
        ConstrainedIonic1DProblem(original, count - 1)
    y = multilayer_trial(multilayer_case)
    density = device.fields(y, voltage_V=0.002)[3]
    original_rows, constrained = original.residual(y), problem.residual(y)
    row = problem.constraint_row
    other = np.arange(device.state_size) != row
    np.testing.assert_array_equal(constrained[other], original_rows[other])
    inventory = math.fsum((device.ion_geometry.cell_measures_m * density).tolist())
    assert constrained[row] == inventory - problem.inventory_target_m2
    assert problem.inventory_target_m2 == initial.domain_inventory_m2
    original_J, changed_J = original.jacobian(y), problem.jacobian(y)
    np.testing.assert_array_equal(changed_J.toarray()[other], original_J.toarray()[other])
    np.testing.assert_array_equal(changed_J.getrow(row).toarray()[0, :3 * count], np.zeros(3 * count))
    np.testing.assert_array_equal(changed_J.getrow(row).toarray()[0, 3 * count:],
                                   -original.storage_jacobian(y).diagonal()[3 * count:])
    assert changed_J[:count, 3 * count:].nnz > 0 and changed_J[3 * count:, :count].nnz > 0


def test_multilayer_constrained_directional_J(multilayer_case):
    _, device, _, problem, _ = multilayer_case
    y = multilayer_trial(multilayer_case)
    direction = np.random.default_rng(816).uniform(-0.1, 0.1, device.state_size)
    direction[:device.node_count] *= device.electronic_component.thermal_voltage_V
    h = 1e-5
    numerical = (problem.residual(y + h * direction) - problem.residual(y - h * direction)) / (2 * h)
    matrix = problem.jacobian(y)
    scale = np.maximum(abs(matrix) @ abs(direction), np.finfo(float).tiny)
    assert np.max(abs(numerical - matrix @ direction) / scale) < 3e-7


def test_multilayer_diagnostics_map_charge_and_current_without_endpoint_heuristics(multilayer_case):
    _, device, original, problem, _ = multilayer_case
    count = device.node_count
    y = multilayer_trial(multilayer_case)
    density = device.fields(y, voltage_V=0.002)[3]
    diagnostics = _diagnostics(problem, y)
    physical_domain = Q * device.ion_geometry.cell_measures_m * (density - device.positive_ion_reference_m3)
    physical = np.zeros(count)
    physical[device.ion_node_indices] = physical_domain
    np.testing.assert_array_equal(diagnostics["physical_ion_net_charge_C_m2"], physical)
    np.testing.assert_array_equal(diagnostics["domain_ion_net_charge_C_m2"], physical_domain)
    np.testing.assert_allclose(diagnostics["deposited_ion_net_charge_C_m2"], physical, rtol=5e-13, atol=1e-18)
    assert np.all(abs(device.ion_charge_weight_ratio[[0, -1]] - 0.5) > 0.1)
    current = np.zeros(count + 1)
    current[device.ion_node_indices[:-1] + 1] = Q * original.ion_fluxes(y)
    assert np.max(abs(current)) > Q * 1e6
    np.testing.assert_array_equal(diagnostics["ionic_current_density_A_m2"], current)
    np.testing.assert_array_equal(diagnostics["total_current_density_A_m2"],
                                   diagnostics["electronic_current_density_A_m2"] + current)
    np.testing.assert_array_equal(diagnostics["ion_balance_m2_s"], original.residual(y)[3 * count:])
    assert diagnostics["ion_balance_l1_m2_s"] == math.fsum(abs(original.residual(y)[3 * count:]).tolist())


def test_multilayer_Newton_weights_steps_and_failure_use_N_and_M(multilayer_case, monkeypatch):
    _, device, _, _, initial = multilayer_case
    configured = controls(device.ion_node_count)
    seen = []

    def capture_then_stop(problem, y, settings):
        seen.append((y.copy(), settings))
        raise RuntimeError("intentional stop after recording explicit controls")

    monkeypatch.setattr("solarlab.solvers.nonlinear.ionic_1d.solve_newton", capture_then_stop)
    result = solve_ionic_steady_state(device, voltage_V=0.002, generation_scale=0.0, options=configured)
    assert not result.accepted and result.termination == "evaluation_error"
    assert not result.qualification and len(seen) == 1
    assert result.evaluation_counts == {"residual": 0, "jacobian": 0}
    coordinates, settings = seen[0]
    np.testing.assert_array_equal(coordinates, initial.u)
    np.testing.assert_array_equal(result.candidate_coordinates, initial.u)
    assert settings.scaling.row_weights.shape == (device.state_size,)
    assert settings.step_control.max_step.shape == (device.state_size,)
    electronic = configured.electronic
    np.testing.assert_array_equal(settings.scaling.row_weights[:3 * device.node_count],
        production_residual_scaling(device.electronic_component,
            poisson_tolerance=electronic.poisson_tolerance,
            continuity_tolerance_A_m2=electronic.continuity_tolerance_A_m2,
            contact_log_tolerance=electronic.contact_log_tolerance).row_weights)
    np.testing.assert_array_equal(settings.step_control.max_step[:3 * device.node_count],
        production_coordinates(device.electronic_component, potential_step_V=electronic.potential_step_V,
                                quasi_fermi_step=electronic.quasi_fermi_step).max_step)
    expected = np.full(device.ion_node_count, 2 * device.ion_node_count / configured.ion_balance_tolerance_m2_s)
    expected[-1] = 1 / (configured.inventory_relative_tolerance * device.reference_inventory_m2)
    np.testing.assert_array_equal(settings.scaling.row_weights[3 * device.node_count:], expected)
    np.testing.assert_array_equal(settings.step_control.max_step[3 * device.node_count:],
                                   np.full(device.ion_node_count, configured.ion_logit_step))
    assert settings.budget == configured.budget and settings.tolerance == 1.0


def test_multilayer_rebound_index_manifest_is_refused(multilayer_case):
    _, device, _, _, _ = multilayer_case
    manifest = device.binding_manifest
    manifest["ion_domain"]["electrical_node_indices"][1] += 1
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False)
    with pytest.raises(ValueError):
        changed = replace(device, binding_manifest_json=encoded, sha256=hashlib.sha256(encoded.encode()).hexdigest())
        ConstrainedIonic1DProblem(Ionic1DProblem(changed), device.ion_node_count - 1)


def test_multilayer_zero_D_keeps_inventory_but_refuses_the_steady_closure(multilayer_case):
    prepared, original_device, _, _, _ = multilayer_case
    data = prepared.to_input().editing_data()
    layer = next(layer for layer in data["layers"] if layer["id"] == original_device.domain_layer_id)
    layer["parameters"]["D_ion"] = 0.0
    changed = resolve_device(DeviceInput.model_validate(data), prepared.defaults,
                              prepared.resources, sources=prepared.sources)
    device = compile_ionic_device_1d(changed, intervals_per_layer=(12, 12, 12),
                                     ion_domain_layer="absorber", ion_domain_endpoints=("blocking", "blocking"))
    assert device.reference_inventory_m2 == original_device.reference_inventory_m2
    with pytest.raises(CapabilityError, match="positive-D"):
        solve_ionic_steady_state(device, voltage_V=0.002, generation_scale=0.0,
                                 options=controls(device.ion_node_count))
