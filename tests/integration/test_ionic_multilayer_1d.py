"""Real confined-domain N/M residuals, mapped charge, storage and declarations."""

from __future__ import annotations

from dataclasses import replace
import math

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from examples.research_1d.ionic_multilayer import prepare_confined_ion_device
from examples.research_1d.ionic_residual import prepare_single_ion_device
from solarlab.config.resolve_device import resolve_device
from solarlab.device.compile_ionic_1d import compile_ionic_device_1d
from solarlab.device.inputs import DeviceInput
from solarlab.equations.dc_1d import CapabilityError
from solarlab.equations.ionic_1d import Ionic1DProblem
from solarlab.physics.ions import ion_face_flux, single_site_density

Q = 1.602176634e-19


@pytest.fixture(scope="module", autouse=True)
def blas1():
    with threadpool_limits(limits=1):
        yield


def compile_confined(prepared=None, counts=(12, 12, 12)):
    return compile_ionic_device_1d(
        prepare_confined_ion_device() if prepared is None else prepared,
        intervals_per_layer=counts, ion_domain_layer="absorber",
        ion_domain_endpoints=("blocking", "blocking"),
    )


@pytest.fixture(scope="module")
def case():
    device = compile_confined()
    problem = Ionic1DProblem(device, voltage_V=0.002)
    y = device.initial_state(voltage_V=0.002).u.copy()
    y[3 * device.node_count:] = np.linspace(-0.15, 0.12, device.ion_node_count)
    return device, problem, y


def test_real_N_M_map_contains_both_planes_and_exact_physical_inventory(case):
    device, _, y = case
    assert (device.node_count, device.ion_node_count, device.state_size) == (37, 13, 124)
    np.testing.assert_array_equal(device.ion_node_indices, np.arange(12, 25))
    np.testing.assert_array_equal(device.ion_geometry.nodes_m, device.nodes_m[device.ion_node_indices])
    planes = device.electronic_component.geometry.interfaces
    assert device.ion_node_indices[0] == planes[0].shared_node
    assert device.ion_node_indices[-1] == planes[1].shared_node
    assert math.isclose(device.reference_inventory_m2, 3e17, rel_tol=3e-16)
    state = device.state(y, voltage_V=0.002)
    assert state.u.shape == (124,) and state.electronic_state.u.shape == (111,)
    assert state.positive_ion_density_m3.shape == (13,)
    assert state.device_sha256 == device.identity
    np.testing.assert_array_equal(device.binding_manifest["ion_domain"]["electrical_node_indices"], device.ion_node_indices)
    with pytest.raises(ValueError):
        device.ion_node_indices.setflags(write=True)


@pytest.mark.parametrize("counts", [(12, 12, 12), (5, 11, 7)])
def test_interface_half_cells_use_inside_spacing_not_a_fixed_half_ratio(counts):
    device = compile_confined(counts=counts)
    indices = device.ion_node_indices
    x = device.nodes_m
    expected_left = (x[indices[1]] - x[indices[0]]) / (x[indices[1]] - x[indices[0] - 1])
    expected_right = (x[indices[-1]] - x[indices[-2]]) / (x[indices[-1] + 1] - x[indices[-2]])
    np.testing.assert_allclose(device.ion_charge_weight_ratio[[0, -1]], [expected_left, expected_right], rtol=2e-14)
    assert np.all(device.ion_charge_weight_ratio[[0, -1]] != 0.5)
    np.testing.assert_array_equal(device.ion_charge_weight_ratio[1:-1], np.ones(device.ion_node_count - 2))


def test_charge_oracle_scatter_and_poisson_include_both_ion_domain_endpoints(case):
    device, problem, y = case
    N, idx = device.node_count, device.ion_node_indices
    _, n, p, P = device.fields(y, voltage_V=0.002)
    physical = np.zeros(N)
    physical[idx] = Q * device.ion_geometry.cell_measures_m * (P - device.positive_ion_reference_m3)
    ledger = device.charge_ledger(n, p, P)
    deposited = Q * device.electronic_component.geometry.cell_measures_m * ledger.entry("positive_ion").net_population()
    np.testing.assert_allclose(deposited, physical, rtol=5e-15, atol=1e-20)
    np.testing.assert_array_equal(deposited[:idx[0]], np.zeros(idx[0]))
    np.testing.assert_array_equal(deposited[idx[-1] + 1:], np.zeros(N - idx[-1] - 1))
    difference = problem.residual(y)[:N] - problem.electronic_problem.residual(y[:3 * N])[:N]
    np.testing.assert_allclose(difference, physical, rtol=1e-13, atol=1e-18)
    assert abs(difference[idx[0]]) > 0.0 and abs(difference[idx[-1]]) > 0.0
    J = problem.jacobian(y)
    assert J[idx[0], 3 * N] > 0.0 and J[idx[-1], device.state_size - 1] > 0.0


def test_domain_flux_storage_and_global_current_embedding_are_conservative(case):
    device, problem, y = case
    N, M, idx = device.node_count, device.ion_node_count, device.ion_node_indices
    flux = problem.ion_fluxes(y)
    expected = np.zeros(N - 1)
    expected[idx[:-1]] = flux
    np.testing.assert_array_equal(problem.ion_fluxes_on_electronic_faces(y), expected)
    balance = -np.diff(np.r_[0.0, flux, 0.0])
    np.testing.assert_array_equal(problem.residual(y)[3 * N:], balance)
    bound = 4 * np.finfo(float).eps * math.fsum(abs(flux).tolist())
    assert abs(math.fsum(balance.tolist())) <= bound
    _, _, _, P = device.fields(y, voltage_V=0.002)
    partial = P * (1.0 - P / device.positive_ion_site_limit_m3)
    K = problem.storage_jacobian(y)
    np.testing.assert_array_equal(K.diagonal()[3 * N:], -device.ion_geometry.cell_measures_m * partial)
    assert K.shape == (3 * N + M, 3 * N + M)
    assert K.diagonal()[3 * N] < 0.0 and K.diagonal()[-1] < 0.0
    outside = np.setdiff1d(np.arange(N), idx)
    assert problem.jacobian(y)[3 * N:, :N][:, outside].nnz == 0
    assert problem.jacobian(y)[3 * N:, N:3 * N].nnz == 0


@pytest.mark.parametrize("alpha", [0.0, 2.5])
def test_full_mapped_Fy_and_nonlinear_storage_have_independent_directional_check(case, alpha):
    device, problem, y = case
    rng = np.random.default_rng(40837)
    direction = rng.uniform(-0.1, 0.1, device.state_size)
    derivative = rng.uniform(-0.2, 0.2, device.state_size)
    direction[:device.node_count] *= device.electronic_component.thermal_voltage_V
    derivative[:device.node_count] *= device.electronic_component.thermal_voltage_V
    J = problem.jacobian(y, derivative, alpha=alpha)
    h = 1e-5
    finite = (problem.residual(y + h * direction, derivative + alpha * h * direction)
              - problem.residual(y - h * direction, derivative - alpha * h * direction)) / (2 * h)
    conditioning = abs(J) @ abs(direction)
    error = float(np.max(abs(finite - J @ direction) / np.maximum(conditioning, np.finfo(float).tiny)))
    print(f"3N+M alpha={alpha:g}: complete directional error {error:.9g}")
    assert error < 2e-7


@pytest.mark.parametrize("field", ["positive_ion_reference_m3", "positive_ion_site_limit_m3", "positive_ion_logit_reference"])
def test_original_binding_refusals_are_retained_on_the_M_node_domain(case, field):
    device, _, _ = case
    value = getattr(device, field)
    with pytest.raises(ValueError, match="ionic binding"):
        replace(device, **{field: value + 0.4 if field == "positive_ion_logit_reference" else value * 2})


def test_missing_or_transmissive_domain_declarations_refuse():
    prepared = prepare_confined_ion_device()
    with pytest.raises(CapabilityError, match="explicit ion domain"):
        compile_ionic_device_1d(prepared)
    with pytest.raises(CapabilityError, match="explicit blocking endpoints"):
        compile_ionic_device_1d(prepared, ion_domain_layer="absorber")
    with pytest.raises(CapabilityError, match="interface exchange"):
        compile_ionic_device_1d(prepared, ion_domain_layer="absorber", ion_domain_endpoints=("transmissive", "blocking"))


@pytest.mark.parametrize("parameter,value", [("P0", 1e21), ("D_ion", 1e-16), ("P0_neg", 1e21)])
def test_outside_or_additional_ionic_physics_is_not_projected_away(parameter, value):
    prepared = prepare_confined_ion_device()
    data = prepared.to_input().editing_data()
    data["layers"][0]["parameters"][parameter] = value
    changed = resolve_device(DeviceInput.model_validate(data), prepared.defaults, prepared.resources, sources=prepared.sources)
    with pytest.raises(CapabilityError, match="outside-domain|negative/dual"):
        compile_confined(changed)


def test_disconnected_or_rebound_domain_maps_and_wrong_state_shapes_refuse(case):
    device, _, y = case
    wrong = device.ion_node_indices.copy()
    wrong[5] += 1
    with pytest.raises(ValueError, match="contiguous"):
        replace(device, ion_node_indices=wrong)
    with pytest.raises(ValueError, match="coincide|binding"):
        replace(device, ion_node_indices=device.ion_node_indices + 1)
    with pytest.raises(ValueError, match="3.N.*M"):
        device.state(np.r_[y, 0.0], voltage_V=0.002)


def test_one_single_layer_primitive_regression_keeps_accepted_physical_values():
    device = compile_ionic_device_1d(prepare_single_ion_device(diffusion_only=True), intervals_per_layer=12)
    N = device.node_count
    y = device.initial_state().u.copy()
    x = (device.nodes_m - device.nodes_m[0]) / device.ion_geometry.extent_m
    y[:N] += 0.002 * np.sin(np.pi * x)
    y[3 * N:] = 0.1 * np.cos(np.pi * x)
    P = single_site_density(device.positive_ion_reference_m3, device.positive_ion_site_limit_m3,
                            device.positive_ion_logit_reference, y[3 * N:])
    expected = ion_face_flux(y[:N], P, np.diff(device.nodes_m), device.positive_ion_diffusivity_faces_m2_s,
                             device.electronic_component.thermal_voltage_V, device.positive_ion_site_limit_m3[:-1],
                             steric_diffusion_only=True, P_lim_node=device.positive_ion_site_limit_m3)
    np.testing.assert_array_equal(Ionic1DProblem(device).ion_fluxes(y), expected)
    assert device.reference_inventory_m2 == 3e17 and device.state_size == 4 * N
    np.testing.assert_array_equal(device.ion_node_indices, np.arange(N))
