"""Separate steady inventories, original dropped rows and one bounded witness."""

from __future__ import annotations

from dataclasses import replace
import math
import os
from pathlib import Path

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from examples.research_1d.ionic_steady import controls
from examples.research_1d.multiple_ionic_domains import DOMAINS, INTERVALS, prepare_multiple_ionic_device
from examples.research_1d.multiple_ionic_steady import prepare_mixed_immobile_device
from solarlab.device.compile_multiple_ionic_1d import compile_multiple_ionic_device_1d
from solarlab.equations.multiple_ionic_1d import MultipleIonic1DProblem
from solarlab.equations.ionic_1d import Ionic1DProblem
from solarlab.solvers.nonlinear.ionic_1d import ConstrainedIonic1DProblem
from solarlab.solvers.nonlinear.multiple_ionic_1d import (
    ConstrainedMultipleIonic1DProblem, _settings, multiple_ionic_checks, multiple_ionic_diagnostics,
    solve_multiple_ionic_steady_state,
)


@pytest.fixture(scope="module", autouse=True)
def blas1():
    with threadpool_limits(limits=1):
        yield


def case(kind="positive", *, counts=INTERVALS, right_immobile=False):
    device = compile_multiple_ionic_device_1d(
        prepare_multiple_ionic_device(kind=kind, right_immobile=right_immobile),
        intervals_per_layer=counts, ion_domains=DOMAINS)
    options = controls(device.domains[0].ion_node_count)
    original = MultipleIonic1DProblem(device, voltage_V=0.002, generation_scale=0.0)
    return device, options, original


@pytest.mark.parametrize("kind", ["positive", "independent", "shared"])
def test_only_each_own_inventory_row_changes_and_full_storage_gradient_is_retained(kind):
    device, options, original = case(kind)
    constrained = ConstrainedMultipleIonic1DProblem(original, options.inventory_row_node)
    y = device.initial_state(voltage_V=0.002).u.copy()
    y[3 * device.node_count:] = np.linspace(-0.11, 0.12, device.state_size - 3 * device.node_count)
    F, C, J, K = original.residual(y), constrained.residual(y), constrained.jacobian(y), original.storage_jacobian(y)
    changed_rows = []
    for domain in device.domains:
        name = domain.domain_layer_id
        populations = device.fields_by_domain(y, voltage_V=0.002)[3][name]
        for index, species in enumerate(domain.ion_species_names):
            block, row = device.ion_species_slices[name][species], constrained.constraint_rows[name][species]
            expected = math.fsum((domain.ion_geometry.cell_measures_m
                                 * (populations[index] - getattr(domain, species + "_ion_reference_m3"))).tolist())
            assert C[row] == expected
            np.testing.assert_array_equal(J.getrow(row).toarray().ravel(), -np.asarray(K[block, :].sum(axis=0)).ravel())
            other = device.domain_slices[device.domain_ids[1] if name == device.domain_ids[0] else device.domain_ids[0]]
            assert J[row, other].nnz == 0
            changed_rows.append(row)
    mask = np.ones(device.state_size, dtype=bool)
    mask[changed_rows] = False
    np.testing.assert_array_equal(C[mask], F[mask])
    np.testing.assert_array_equal(J[mask, :].toarray(), original.jacobian(y)[mask, :].toarray())


@pytest.mark.parametrize("kind", ["positive", "independent", "shared"])
def test_complete_constrained_directional_check(kind):
    device, options, original = case(kind, counts=(2, 3, 2, 4, 2))
    problem = ConstrainedMultipleIonic1DProblem(original, options.inventory_row_node)
    y = device.initial_state(voltage_V=0.002).u.copy()
    y[3 * device.node_count:] = np.linspace(-0.08, 0.1, device.state_size - 3 * device.node_count)
    rng = np.random.default_rng(148894)
    direction = rng.uniform(-0.1, 0.1, device.state_size)
    direction[:device.node_count] *= device.electronic_component.thermal_voltage_V
    J = problem.jacobian(y)
    h = 1e-5
    finite = (problem.residual(y + h * direction) - problem.residual(y - h * direction)) / (2 * h)
    error = float(np.max(abs(finite - J @ direction) / np.maximum(abs(J) @ abs(direction), np.finfo(float).tiny)))
    print(f"two-domain constrained {kind}: directional error {error:.9g}")
    assert error < 2e-7


def test_original_per_domain_scaling_and_common_row_node_validation():
    device, options, original = case("independent", counts=(2, 3, 2, 4, 2))
    constrained = ConstrainedMultipleIonic1DProblem(original, 3)
    settings = _settings(constrained, options)
    assert settings.tolerance == 1.0 and not settings.reuse_jacobian
    assert settings.budget == options.budget
    for domain in device.domains:
        for species, block in device.ion_species_slices[domain.domain_layer_id].items():
            row = constrained.constraint_rows[domain.domain_layer_id][species]
            for node in range(block.start, block.stop):
                expected = (1 / (1e-12 * device.reference_inventories_m2[domain.domain_layer_id][species])
                            if node == row else 2 * domain.ion_node_count / 1e6)
                assert settings.scaling.row_weights[node] == expected
                assert settings.step_control.max_step[node] == 1.0
    for node in (True, -1, 4):
        with pytest.raises(ValueError):
            ConstrainedMultipleIonic1DProblem(original, node)


def test_combined_inventory_cancellation_is_rejected_before_newton(monkeypatch):
    device, options, _ = case()
    y = device.initial_state(voltage_V=0.002).u.copy()
    left, right = device.domains
    left_block = device.ion_species_slices[left.domain_layer_id]["positive"]
    y[left_block] = 0.1
    current_left = device.state(y, voltage_V=0.002).domain_inventories_m2[left.domain_layer_id]["positive"]
    total = sum(values["positive"] for values in device.reference_inventories_m2.values())
    density = (total - current_left) / math.fsum(right.ion_geometry.cell_measures_m.tolist())
    capacity, reference = right.positive_ion_site_limit_m3[0], right.positive_ion_reference_m3[0]
    y[device.ion_species_slices[right.domain_layer_id]["positive"]] = (
        math.log(density / (capacity - density)) - math.log(reference / (capacity - reference)))
    state = device.state(y, voltage_V=0.002)
    assert math.isclose(sum(v["positive"] for v in state.domain_inventories_m2.values()), total, rel_tol=1e-12)
    monkeypatch.setattr("solarlab.solvers.nonlinear.multiple_ionic_1d.solve_newton",
                        lambda *args, **kwargs: pytest.fail("invalid inventory reached Newton"))
    with pytest.raises(ValueError, match="each mobile domain"):
        solve_multiple_ionic_steady_state(device, voltage_V=0.002, generation_scale=0.0, options=options, initial_state=state)


def test_every_original_dropped_balance_is_checked(monkeypatch):
    device, options, original = case()
    problem = ConstrainedMultipleIonic1DProblem(original, options.inventory_row_node)
    y = device.initial_state(voltage_V=0.002).u.copy()
    retained = original.residual(y)
    retained[3 * device.node_count:] = 0.0
    row = problem.constraint_rows["absorber_left"]["positive"]
    retained[row] = 2e6
    monkeypatch.setattr(MultipleIonic1DProblem, "residual", lambda self, point, *args: retained.copy())
    assert problem.residual(y)[row] == 0.0
    checks = {check.name: check for check in multiple_ionic_checks(problem, y, options)}
    assert not checks["absorber_left.positive.ion_balance_l1_m2_s"].passed
    assert checks["absorber_right.positive.ion_balance_l1_m2_s"].passed
    data = multiple_ionic_diagnostics(problem, y)
    assert data["domains"]["absorber_left"]["species"]["positive"]["dropped_balance_m2_s"] == 2e6


@pytest.mark.parametrize("kind", ["positive", "independent", "shared"])
def test_immobile_physical_rows_match_accepted_local_closure_and_retain_storage(kind):
    device, options, original = case(kind, right_immobile=True)
    y = device.initial_state(voltage_V=0.002).u.copy()
    y[device.domain_slices["absorber_right"]] = np.linspace(-0.1, 0.1, device.domains[1].state_size - 3 * device.node_count)
    assert original.storage_jacobian(y)[device.domain_slices["absorber_right"], :].nnz > 0
    constrained = ConstrainedMultipleIonic1DProblem(original, options.inventory_row_node)
    domain = device.domains[1]
    local = ConstrainedIonic1DProblem(Ionic1DProblem(domain, 0.002, 0.0), options.inventory_row_node)
    local_y = device.local_coordinates(y, domain.domain_layer_id)
    np.testing.assert_array_equal(constrained.residual(y)[device.domain_slices[domain.domain_layer_id]],
                                  local.residual(local_y)[3 * device.node_count:])
    K, J = original.storage_jacobian(y), constrained.jacobian(y)
    for rows in constrained.profile_constraint_rows[domain.domain_layer_id].values():
        for row in rows:
            np.testing.assert_array_equal(J.getrow(row).toarray(), -K.getrow(row).toarray())
    np.testing.assert_array_equal(original.residual(y)[device.domain_slices[domain.domain_layer_id]],
                                  np.zeros(domain.state_size - 3 * device.node_count))
    assert constrained.constraint_rows["absorber_right"] == {}
    checks = {check.name: check for check in multiple_ionic_checks(constrained, y, options)}
    assert not checks["absorber_right.positive.profile_relative_error"].passed


def test_shared_partner_moves_with_exact_physical_frozen_target_and_full_cross_gradient():
    device = compile_multiple_ionic_device_1d(prepare_mixed_immobile_device(),
        intervals_per_layer=INTERVALS, ion_domains=DOMAINS)
    original = MultipleIonic1DProblem(device, 0.002, 0.0)
    constrained = ConstrainedMultipleIonic1DProblem(original, 3)
    y = device.initial_state(voltage_V=0.002).u.copy()
    domain = device.domains[1]
    plus, minus = device.ion_species_slices["absorber_right"].values()
    capacity = domain.positive_ion_site_limit_m3
    positive = domain.positive_ion_reference_m3 * 1.1
    negative = domain.negative_ion_reference_m3
    vacancy = 1.0 - (positive + negative) / capacity
    y[plus] = np.log(positive / capacity) - np.log(vacancy) - domain.positive_ion_logit_reference
    y[minus] = np.log(negative / capacity) - np.log(vacancy) - domain.negative_ion_logit_reference
    values = device.fields_by_domain(y, voltage_V=0.002)[3]["absorber_right"]
    np.testing.assert_allclose(values[1], negative, rtol=4 * np.finfo(float).eps, atol=0.0)
    assert np.any(y[minus] != 0.0) and np.any(y[plus] != 0.0)
    assert constrained.jacobian(y)[minus, plus].nnz == domain.ion_node_count
    np.testing.assert_array_equal(original.residual(y)[minus], np.zeros(domain.ion_node_count))
    wrong = y.copy()
    wrong[minus] = 0.0
    assert np.max(abs(constrained.residual(wrong)[minus])) > 1e5
    fixed = constrained.residual(y)[minus]
    assert np.max(abs(fixed) / (domain.ion_geometry.cell_measures_m * negative)) < 1e-12


@pytest.mark.parametrize("kind", ["positive", "independent", "shared"])
def test_mixed_full_directional_and_local_profile_scaling(kind):
    device, options, original = case(kind, right_immobile=True)
    problem = ConstrainedMultipleIonic1DProblem(original, 3)
    y = device.initial_state(voltage_V=0.002).u.copy()
    y[3 * device.node_count:] = np.linspace(-0.05, 0.07, device.state_size - 3 * device.node_count)
    rng = np.random.default_rng(9908)
    direction = rng.uniform(-0.1, 0.1, y.size)
    direction[:device.node_count] *= device.electronic_component.thermal_voltage_V
    J = problem.jacobian(y)
    # The D0 row's weak directional component otherwise loses precision
    # when differencing physical populations. The error criterion is unchanged.
    h = 1e-4
    finite = (problem.residual(y + h * direction) - problem.residual(y - h * direction)) / (2 * h)
    error = float(np.max(abs(finite - J @ direction) / np.maximum(abs(J) @ abs(direction), np.finfo(float).tiny)))
    print(f"two-domain mixed {kind}: directional error {error:.9g}")
    assert error < 2e-7
    settings = _settings(problem, options)
    domain = device.domains[1]
    for species, block in device.ion_species_slices["absorber_right"].items():
        np.testing.assert_array_equal(settings.scaling.row_weights[block],
            1 / (options.inventory_relative_tolerance * domain.ion_geometry.cell_measures_m
                 * getattr(domain, species + "_ion_reference_m3")))


def test_actual_mixed_shared_immobile_regression(tmp_path):
    from examples.research_1d.multiple_ionic_steady import run_case
    output = Path(os.environ.get("SOLARLAB_MULTIPLE_D0_EVIDENCE", str(tmp_path / "MixedSteady.json")))
    result = run_case(output, mixed_immobile=True)
    assert result.accepted, (result.termination, result.error)
    assert result.state is not None and result.diagnostics is not None and not result.qualification
    negative = result.diagnostics["domains"]["absorber_right"]["species"]["negative"]
    assert negative["profile_relative_error"] <= 1e-12
    np.testing.assert_array_equal(negative["ion_balance_m2_s"], np.zeros(4))
    np.testing.assert_array_equal(negative["ion_flux_m2_s"], np.zeros(5))
    assert all(check.passed for check in result.checks)
    right = result.diagnostics["domains"]["absorber_right"]["species"]["positive"]
    assert right["maximum_relative_change_from_reference"] > 0.0
    assert len(result.solver_controls["profile_constraint_rows"]["absorber_right"]["negative"]) == 4


def test_original_equilibrium_gate_names_and_unbound_state_refusal():
    device, options, _ = case()
    y = device.initial_state(voltage_V=0.0).u
    original = MultipleIonic1DProblem(device, voltage_V=0.0, generation_scale=0.0)
    names = {check.name for check in multiple_ionic_checks(ConstrainedMultipleIonic1DProblem(original, 3), y, options)}
    assert {"poisson_balance", "continuity_current_bound", "all_face_current_spread",
            "zero_equilibrium_carrier_current", "contact_thermodynamics", "fixed_contact_log_max"} <= names
    wrong = replace(device.initial_state(voltage_V=0.002), device_sha256="unbound")
    with pytest.raises(ValueError):
        solve_multiple_ionic_steady_state(device, voltage_V=0.002, generation_scale=0.0, options=options, initial_state=wrong)


def test_actual_cold_two_domain_regression(tmp_path):
    """Run once locally; its saved raw JSON also serves read-only final checks."""
    from examples.research_1d.multiple_ionic_steady import run_case
    output = Path(os.environ.get("SOLARLAB_MULTIPLE_STEADY_EVIDENCE", str(tmp_path / "Steady.json")))
    result = run_case(output)
    assert result.accepted, (result.termination, result.error)
    assert result.newton is not None and result.newton.accepted and all(check.passed for check in result.checks)
    assert not result.qualification
    data = result.diagnostics
    assert data is not None and result.state is not None
    assert set(data["domains"]) == {"absorber_left", "absorber_right"}
    for name, record in data["domains"].items():
        species = record["species"]["positive"]
        assert species["inventory_relative_error"] <= 1e-12
        assert species["ion_balance_l1_m2_s"] <= 1e6 and species["ion_flux_max_m2_s"] <= 1e6
        assert abs(species["dropped_balance_m2_s"]) <= 1e6
        assert species["minimum_density_m3"] > 0.0 and species["maximum_site_fraction"] < 1.0
        assert result.state.domain_inventories_m2[name]["positive"] > 0.0
    device, _, original = case()
    J = original.jacobian(result.state.u)
    for block in device.domain_slices.values():
        assert J[:device.node_count, block].nnz > 0 and J[block, :device.node_count].nnz > 0
    assert data["carrier_balance_A_m2"] <= 1e-8 and data["current_spread_A_m2"] <= 1e-8
    np.testing.assert_array_equal(data["ionic_current_density_A_m2"][[0, 1, 2, 6, 7, 11, 12, 13]], np.zeros(8))
