"""Declared per-cell immobile profiles, full gradients and native storage."""

from __future__ import annotations

from dataclasses import replace
import math
from types import SimpleNamespace

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from examples.research_1d.immobile_ions import prepare_immobile_ion_device
from examples.research_1d.ionic_multilayer import prepare_confined_ion_device, prepare_confined_dual_ion_device
from examples.research_1d.shared_site_ionic_fixture import prepare_confined_shared_site_ion_device
from solarlab.device.compile_ionic_1d import compile_ionic_device_1d
from solarlab.equations.dc_1d import CapabilityError
from solarlab.equations.ionic_1d import Ionic1DProblem
from solarlab.equations.ionic_closure import (
    declared_immobile_profiles, immobile_profile_binding, ion_mobility_partition,
    ionic_closure_residual, ionic_closure_jacobian,
)
from solarlab.solvers.nonlinear.ionic_1d import ConstrainedIonic1DProblem, ionic_steady_control_record, checks as ionic_checks, diagnostics as ionic_diagnostics
from solarlab.equations.ionic_small_signal import build_ionic_small_signal
from examples.research_1d.ionic_steady import controls


@pytest.fixture(scope="module", autouse=True)
def blas1():
    with threadpool_limits(limits=1):
        yield


def compile_case(prepared, single=False):
    if single:
        return compile_ionic_device_1d(prepared, intervals_per_layer=4)
    return compile_ionic_device_1d(prepared, intervals_per_layer=(4, 4, 4),
                                   ion_domain_layer="absorber", ion_domain_endpoints=("blocking", "blocking"))


@pytest.fixture(scope="module", params=["shared_mixed", "shared_full", "single"])
def case(request):
    device = compile_case(prepare_immobile_ion_device(request.param), request.param == "single")
    problem = Ionic1DProblem(device, voltage_V=0.002)
    y = device.initial_state(voltage_V=0.002).u.copy()
    for index, block in enumerate(device.ion_species_slices):
        y[block] = np.linspace(-0.1, 0.08, device.ion_node_count) + index * 0.03
    return device, problem, y


def test_exact_partition_and_declared_profile_do_not_depend_on_guess(case):
    device, problem, y = case
    mobile, immobile = ion_mobility_partition(device)
    assert immobile
    profiles = declared_immobile_profiles(device)
    binding = immobile_profile_binding(device)
    assert binding is not None and binding["compiled_identity"] == device.identity
    assert binding["rule"] == "declared_per_cell_zero_diffusivity_profile_v1"
    unchanged = immobile_profile_binding(device)
    for species, density, block in zip(device.ion_species_names, device.fields_by_species(y, voltage_V=0.002)[3], device.ion_species_slices):
        if species in immobile:
            reference = getattr(device, species + "_ion_reference_m3")
            np.testing.assert_array_equal(profiles[species], reference)
            assert np.any(density != reference)
            np.testing.assert_array_equal(problem.residual(y)[block], np.zeros(device.ion_node_count))
            with pytest.raises(ValueError):
                profiles[species].setflags(write=True)
    y[device.ion_species_slices[-1]] += 0.01
    assert immobile_profile_binding(device) == unchanged
    assert set(mobile).isdisjoint(immobile)


def test_every_immobile_cell_has_full_physical_gradient(case):
    device, problem, y = case
    profiles = declared_immobile_profiles(device)
    original, constrained = problem.residual(y), ionic_closure_residual(problem, y, 0)
    J, K = ionic_closure_jacobian(problem, y, 0), problem.storage_jacobian(y)
    np.testing.assert_array_equal(constrained[:3 * device.node_count], original[:3 * device.node_count])
    for species, density, block in zip(device.ion_species_names, device.fields_by_species(y, voltage_V=0.002)[3], device.ion_species_slices):
        if species in profiles:
            np.testing.assert_array_equal(constrained[block], device.ion_geometry.cell_measures_m * (density - profiles[species]))
            np.testing.assert_array_equal(J[block, :].toarray(), -K[block, :].toarray())
            if device.ion_site_contract == "shared_finite_sites":
                other = device.ion_species_slices[1 - device.ion_species_names.index(species)]
                assert J[block, other].nnz == device.ion_node_count
        else:
            inventory = math.fsum((device.ion_geometry.cell_measures_m * density).tolist())
            target = math.fsum((device.ion_geometry.cell_measures_m * getattr(device, species + "_ion_reference_m3")).tolist())
            assert constrained[block.start] == inventory - target
            np.testing.assert_array_equal(constrained[block.start + 1:block.stop], original[block.start + 1:block.stop])


def test_wired_steady_profile_metadata_checks_and_AC_full_local_rows(case):
    device, problem, y = case
    steady = ConstrainedIonic1DProblem(problem, 0)
    np.testing.assert_array_equal(steady.residual(y), ionic_closure_residual(problem, y, 0))
    np.testing.assert_array_equal(steady.jacobian(y).toarray(), ionic_closure_jacobian(problem, y, 0).toarray())
    options = controls(device.ion_node_count)
    record = ionic_steady_control_record(device, options)
    assert record["ion_closure"] == immobile_profile_binding(device)
    checked = ionic_checks(ionic_diagnostics(steady, y), options)
    for species in steady.immobile_species:
        item = next(item for item in checked if item.name == species + "_immobile_profile_relative_error")
        assert not item.passed and item.limit == options.inventory_relative_tolerance
    state = device.state(y, voltage_V=problem.voltage_V)
    operator = build_ionic_small_signal(device, state)
    zero = operator.system(0.0, voltage_phasor_V=2e-6 + 1e-6j)
    finite = operator.system(1.0, voltage_phasor_V=2e-6 + 1e-6j)
    K = operator.storage_jacobian
    for species, block in zip(device.ion_species_names, device.ion_species_slices):
        if species in steady.immobile_species:
            assert zero.immobile_constraint_rows[species] == tuple(range(block.start, block.stop))
            np.testing.assert_array_equal(zero.jacobian[block, :].toarray(), -K[block, :].toarray())
            assert zero.storage_jacobian[block, :].nnz == 0
            assert np.count_nonzero(zero.rhs[block]) == np.count_nonzero(zero.rhs_rate[block]) == 0
            np.testing.assert_array_equal(finite.storage_jacobian[block, :].toarray(), K[block, :].toarray())
        else:
            assert species in zero.constraint_rows
    assert zero.original_rows_enforced == tuple(range(device.state_size))


def test_complete_closure_has_independent_represented_directional_derivative(case):
    device, problem, y = case
    rng = np.random.default_rng(40781)
    d = rng.uniform(-0.1, 0.1, device.state_size)
    d[:device.node_count] *= device.electronic_component.thermal_voltage_V
    h = 5e-5
    plus, minus = y + h * d, y - h * d
    represented = (plus - minus) / (2 * h)
    finite = (ionic_closure_residual(problem, plus, 0) - ionic_closure_residual(problem, minus, 0)) / (2 * h)
    J = ionic_closure_jacobian(problem, y, 0)
    scale = abs(J) @ abs(d)
    error = float(np.max(abs(finite - J @ represented) / np.maximum(scale, np.finfo(float).tiny)))
    print(f"immobile closure {device.ion_species_names}: full directional {error:.9g}")
    assert error < 2e-7


def test_native_time_storage_and_all_zero_D_balances_remain_present(case):
    device, problem, y = case
    dot = np.zeros(device.state_size)
    dot[3 * device.node_count:] = np.linspace(-0.2, 0.15, device.state_size - 3 * device.node_count)
    K = problem.storage_jacobian(y)
    native = problem.residual(y, dot)
    for species, block in zip(device.ion_species_names, device.ion_species_slices):
        if species in declared_immobile_profiles(device):
            np.testing.assert_allclose(native[block], (K @ dot)[block], rtol=2e-13, atol=0.0)
            assert np.max(abs(native[block])) > 0.0
            assert np.all(K.diagonal()[block] < 0.0)


@pytest.mark.parametrize("factory", [prepare_confined_ion_device, prepare_confined_dual_ion_device, prepare_confined_shared_site_ion_device])
def test_positive_D_original_residual_and_J_are_identical(factory):
    device = compile_case(factory())
    problem = Ionic1DProblem(device, voltage_V=0.002)
    original = ConstrainedIonic1DProblem(problem, 0)
    y = device.initial_state(voltage_V=0.002).u.copy()
    for block in device.ion_species_slices:
        y[block] = np.linspace(-0.05, 0.03, device.ion_node_count)
    assert declared_immobile_profiles(device) == {} and immobile_profile_binding(device) is None
    np.testing.assert_array_equal(ionic_closure_residual(problem, y, 0), original.residual(y))
    np.testing.assert_array_equal(ionic_closure_jacobian(problem, y, 0).toarray(), original.jacobian(y).toarray())


def test_global_inventory_cannot_hide_a_wrong_local_immobile_profile():
    device = compile_case(prepare_immobile_ion_device("shared_mixed"))
    y = device.initial_state(voltage_V=0.002).u.copy()
    P, N = device.positive_ion_reference_m3.copy(), device.negative_ion_reference_m3.copy()
    widths = device.ion_geometry.cell_measures_m
    change = np.zeros(device.ion_node_count)
    change[1] = 0.005 * N[1]
    change[-2] = -widths[1] * change[1] / widths[-2]
    N += change
    vacancy = device.positive_ion_site_limit_m3 - P - N
    plus, minus = device.ion_species_slices
    y[plus] = np.log(P / vacancy) - device.positive_ion_logit_reference
    y[minus] = np.log(N / vacancy) - device.negative_ion_logit_reference
    physical = device.fields_by_species(y, voltage_V=0.002)[3][1]
    global_error = abs(math.fsum((widths * (physical - device.negative_ion_reference_m3)).tolist()))
    assert global_error < 1e-12 * device.negative_reference_inventory_m2
    local = ionic_closure_residual(Ionic1DProblem(device, 0.002), y, 0)[minus]
    assert np.max(abs(local)) > 1e-4 * device.negative_reference_inventory_m2


def test_disconnected_face_pattern_remains_explicitly_unsupported():
    device = compile_case(prepare_confined_ion_device())
    faces = device.positive_ion_diffusivity_faces_m2_s.copy()
    faces[1] = 0.0
    changed = replace(device, positive_ion_diffusivity_faces_m2_s=faces)
    with pytest.raises(CapabilityError, match="disconnected"):
        ion_mobility_partition(changed)


def test_AC_guard_rejects_equal_opposite_local_immobile_motion_without_a_solve():
    from solarlab.experiments.ionic_impedance import _checks as ac_checks, IonicImpedanceOptions
    device = compile_case(prepare_immobile_ion_device("shared_mixed"))
    perturbation = np.zeros(device.ion_node_count, dtype=complex)
    widths = device.ion_geometry.cell_measures_m
    perturbation[1] = 1e-6 * device.negative_ion_reference_m3[1]
    perturbation[-2] = -widths[1] * perturbation[1] / widths[-2]
    # Adversarial guard inputs, not a manufactured accepted frequency state.
    observed = {"total_current_density_A_m2": np.zeros(device.node_count + 1, dtype=complex),
                "original_componentwise_backward_error": 0.0,
                "ion_density_perturbation_m3": {"positive": np.zeros(device.ion_node_count, dtype=complex),
                                               "negative": perturbation},
                "immobile_density_perturbation_relative_per_V": {"negative": 0.0}}
    diagnostics = ac_checks(device, observed, SimpleNamespace(componentwise_backward_error=0.0),
                            1.0 + 0.0j, "mobile", IonicImpedanceOptions())
    global_check = next(item for item in diagnostics if item.name == "negative_inventory_relative_per_V")
    local_check = next(item for item in diagnostics if item.name == "negative_immobile_density_relative_per_V")
    assert global_check.passed and not local_check.passed
    assert local_check.limit == 1e-12 and local_check.value >= 1e-6
