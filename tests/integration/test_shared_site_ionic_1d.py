"""Actual shared-site F/J/storage, physical paired secants and old-model limits."""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal, localcontext
import math

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from examples.research_1d.shared_site_ionic_fixture import prepare_confined_shared_site_ion_device
from solarlab.config.resolve_device import resolve_device
from solarlab.device.compile_ionic_1d import compile_ionic_device_1d
from solarlab.device.inputs import DeviceInput
from solarlab.equations.dc_1d import CapabilityError
from solarlab.equations.ionic_1d import Ionic1DProblem
from solarlab.physics.ions import ion_face_flux, shared_site_densities
from solarlab.physics.ions.shared_site import (
    shared_site_coordinate_jacobian_m3, shared_site_coordinate_hessian_m3,
    shared_site_secant_m3,
)

Q = 1.602176634e-19


@pytest.fixture(scope="module", autouse=True)
def blas1():
    with threadpool_limits(limits=1):
        yield


def compile_domain(prepared):
    return compile_ionic_device_1d(prepared, intervals_per_layer=(12, 12, 12),
                                   ion_domain_layer="absorber", ion_domain_endpoints=("blocking", "blocking"))


@pytest.fixture(scope="module")
def device():
    return compile_domain(prepare_confined_shared_site_ion_device())


@pytest.fixture(scope="module", params=["normal", "high", "clipped"])
def case(device, request):
    y = device.initial_state(voltage_V=0.002).u.copy()
    plus, minus = device.ion_species_slices
    if request.param == "normal":
        y[plus] = np.linspace(-0.15, 0.12, 13)
        y[minus] = np.linspace(0.12, -0.09, 13)
    else:
        p, n = (0.58, 0.40) if request.param == "high" else (0.6, 0.3999995)
        vacancy = 1.0 - p - n
        y[plus] = math.log(p / vacancy) - device.positive_ion_logit_reference
        y[minus] = math.log(n / vacancy) - device.negative_ion_logit_reference
    return device, Ionic1DProblem(device, voltage_V=0.002), y


def test_compiled_shared_identity_domain_and_physical_map_are_complete(device):
    assert (device.node_count, device.ion_node_count, device.state_size) == (37, 13, 137)
    assert device.ion_site_contract == "shared_finite_sites"
    np.testing.assert_array_equal(device.ion_node_indices, np.arange(12, 25))
    np.testing.assert_array_equal(device.positive_ion_site_limit_m3, device.negative_ion_site_limit_m3)
    manifest = device.binding_manifest
    assert manifest["ion_model"]["site_contract"] == "shared_finite_sites"
    assert manifest["ion_model"]["coordinate_map"] == "historical_shared_site_softmax_v1"
    vacancy = 1.0 - (device.positive_ion_reference_m3 + device.negative_ion_reference_m3) / device.positive_ion_site_limit_m3
    np.testing.assert_array_equal(device.positive_ion_logit_reference,
                                  np.log(device.positive_ion_reference_m3 / device.positive_ion_site_limit_m3) - np.log(vacancy))
    assert np.all(vacancy > 0.0)
    state = device.initial_state(voltage_V=0.002)
    assert set(state.domain_inventories_m2) == {"positive", "negative"}
    assert all(value > 0.0 for value in state.domain_inventories_m2.values())


def test_high_occupancy_uses_original_softmax_and_historical_partner_law(case):
    device, problem, y = case
    phi, _, _, populations = device.fields_by_species(y, voltage_V=0.002)
    plus, minus = device.ion_species_slices
    expected = shared_site_densities(device.positive_ion_site_limit_m3,
                                    device.positive_ion_logit_reference, device.negative_ion_logit_reference,
                                    y[plus], y[minus])
    for actual, original in zip(populations, expected):
        np.testing.assert_array_equal(actual, original)
    assert np.all(populations[0] + populations[1] < device.positive_ion_site_limit_m3)
    for index, species in enumerate(device.ion_species_names):
        flux = ion_face_flux(phi[device.ion_node_indices], populations[index], np.diff(device.ion_geometry.nodes_m),
                             getattr(device, species + "_ion_diffusivity_faces_m2_s"),
                             device.electronic_component.thermal_voltage_V, device.positive_ion_site_limit_m3[:-1],
                             steric_diffusion_only=True, P_lim_node=device.positive_ion_site_limit_m3,
                             P_other_node=populations[1 - index], drift_sign=1.0 if index == 0 else -1.0)
        np.testing.assert_array_equal(problem.ion_fluxes(y, species=species), flux)


def test_all_particle_rows_signed_current_and_half_cell_charge_are_conservative(case):
    device, problem, y = case
    N, idx = device.node_count, device.ion_node_indices
    _, n, p, populations = device.fields_by_species(y, voltage_V=0.002)
    residual = problem.residual(y)
    for species, block in zip(device.ion_species_names, device.ion_species_slices):
        flux = problem.ion_fluxes(y, species=species)
        np.testing.assert_array_equal(residual[block], -np.diff(np.r_[0.0, flux, 0.0]))
        assert abs(math.fsum(residual[block].tolist())) <= 4 * np.finfo(float).eps * math.fsum(abs(flux).tolist())
    projected_balance = np.zeros(N)
    plus, minus = device.ion_species_slices
    projected_balance[idx] = Q * (residual[plus] - residual[minus])
    current = problem.ion_current_density_A_m2(y)
    scale = max(float(np.max(abs(current))), np.finfo(float).tiny)
    np.testing.assert_allclose(-np.diff(np.r_[0.0, current, 0.0]), projected_balance,
                               rtol=0.0, atol=16 * np.finfo(float).eps * scale)
    ledger = device.charge_ledger(n, p, populations[0], populations[1])
    for species, density in zip(device.ion_species_names, populations):
        entry = ledger.entry(species + "_ion")
        reference = getattr(device, species + "_ion_reference_m3")
        physical = np.zeros(N)
        physical[idx] = entry.charge_number * Q * device.ion_geometry.cell_measures_m * (density - reference)
        actual = entry.charge_number * Q * device.electronic_component.geometry.cell_measures_m * entry.net_population()
        np.testing.assert_allclose(actual, physical, rtol=5e-15, atol=1e-20)
    J = problem.jacobian(y)
    for local, node in enumerate(idx):
        assert J[node, plus.start + local] > 0.0 and J[node, minus.start + local] < 0.0


def test_full_mass_hessian_storage_and_inventory_gradients_include_partner_columns(case):
    device, problem, y = case
    N, blocks = device.node_count, device.ion_species_slices
    A = device.ion_coordinate_jacobian_m3(y, voltage_V=0.002)
    H = device.ion_coordinate_hessian_m3(y, voltage_V=0.002)
    assert A.shape == (13, 2, 2) and H.shape == (13, 2, 2, 2)
    assert np.all(A[:, 0, 1] < 0.0) and np.all(A[:, 1, 0] < 0.0)
    assert np.all(np.linalg.eigvalsh(A) > 0.0)
    dot = np.linspace(-0.2, 0.15, device.state_size)
    dot[:N] *= device.electronic_component.thermal_voltage_V
    base, dynamic, K = problem.residual(y), problem.residual(y, dot), problem.storage_jacobian(y)
    np.testing.assert_array_equal(dynamic[N:3 * N], problem.electronic_problem.residual(y[:3 * N], dot[:3 * N])[N:3 * N])
    for species, block in enumerate(blocks):
        rate = A[:, species, 0] * dot[blocks[0]] + A[:, species, 1] * dot[blocks[1]]
        np.testing.assert_array_equal(dynamic[block], base[block] - device.ion_geometry.cell_measures_m * rate)
        assert problem.inventory_rate_m2_s(y, dot, species=device.ion_species_names[species]) == math.fsum((device.ion_geometry.cell_measures_m * rate).tolist())
        inventory_gradient = -np.asarray(K[block, :].sum(axis=0)).ravel()
        for coordinate, coordinate_block in enumerate(blocks):
            expected = -device.ion_geometry.cell_measures_m * A[:, species, coordinate]
            np.testing.assert_array_equal(K[block, coordinate_block].toarray(), np.diag(expected))
            np.testing.assert_array_equal(inventory_gradient[coordinate_block], -expected)
    assert K[blocks[0], blocks[1]].nnz == K[blocks[1], blocks[0]].nnz == 13


@pytest.mark.parametrize("alpha", [0.0, 2.5])
def test_every_original_Fy_and_storage_row_matches_represented_directional_inputs(case, alpha):
    device, problem, y = case
    rng = np.random.default_rng(40881)
    direction = rng.uniform(-0.1, 0.1, device.state_size)
    dot = rng.uniform(-0.2, 0.2, device.state_size)
    direction[:37] *= device.electronic_component.thermal_voltage_V
    dot[:37] *= device.electronic_component.thermal_voltage_V
    h = 1e-5
    yp, ym = y + h * direction, y - h * direction
    dp, dm = dot + alpha * h * direction, dot - alpha * h * direction
    # The intended tiny boundary direction rounds when added to order-one phi.
    # Account for actual input increments, retaining every row and old scale.
    represented_y = (yp - ym) / (2 * h)
    represented_dot = (dp - dm) / (2 * h)
    J, K = problem.jacobian(y, dot, alpha=alpha), problem.storage_jacobian(y)
    numerical = (problem.residual(yp, dp) - problem.residual(ym, dm)) / (2 * h)
    analytic = J @ represented_y + K @ (represented_dot - alpha * represented_y)
    original_scale = abs(J) @ abs(direction)
    error = float(np.max(abs(analytic - numerical) / np.maximum(original_scale, np.finfo(float).tiny)))
    print(f"shared occupancy={float(np.max(sum(device.fields_by_species(y, voltage_V=0.002)[3]) / device.positive_ion_site_limit_m3)):.9g} alpha={alpha:g}: full represented FD {error:.9g}")
    assert error < 2e-7
    np.testing.assert_array_equal(J.toarray(), (problem.jacobian(y, dot) + alpha * K).toarray())
    for row in (0, 36):
        assert J[row, row] == 1.0 and J[row, :].nnz == 1 and K[row, :].nnz == 0


def decimal_density(reference_plus, reference_minus, plus, minus, capacity):
    wplus, wminus = (reference_plus + plus).exp(), (reference_minus + minus).exp()
    denominator = Decimal(1) + wplus + wminus
    return capacity * wplus / denominator, capacity * wminus / denominator


@pytest.mark.parametrize("motion", ["finite", "common_tiny", "opposite_tiny", "partner_tiny"])
def test_paired_secant_matches_independent_100_digit_physical_difference(motion):
    capacity = 2e24
    refp, refn = np.log(0.5) - np.log(0.2), np.log(0.3) - np.log(0.2)
    oldp, oldn = 0.4, -0.3
    newp, newn = oldp, oldn
    if motion == "finite":
        newp += 0.08
        newn -= 0.06
    elif motion == "common_tiny":
        newp, newn = np.nextafter(oldp, np.inf), np.nextafter(oldn, np.inf)
    elif motion == "opposite_tiny":
        newp, newn = np.nextafter(oldp, np.inf), np.nextafter(oldn, -np.inf)
    else:
        newn = np.nextafter(oldn, np.inf)
    actual = shared_site_secant_m3(*(np.array([value]) for value in (capacity, refp, refn, oldp, oldn, newp, newn)))
    with localcontext() as context:
        context.prec = 100
        values = [Decimal.from_float(float(value)) for value in (capacity, refp, refn, oldp, oldn, newp, newn)]
        C, rp, rn, ap, an, bp, bn = values
        before, after = decimal_density(rp, rn, ap, an, C), decimal_density(rp, rn, bp, bn, C)
        expected = [float(after[index] - before[index]) for index in range(2)]
    for observed, oracle in zip(actual, expected):
        np.testing.assert_allclose(observed[0], oracle, rtol=2e-13, atol=0.0)
        assert observed[0] != 0.0
    if motion == "partner_tiny":
        assert actual[0][0] < 0.0 and actual[1][0] > 0.0


def test_zero_partner_primitive_reduces_to_independent_logistic_limit():
    C, P = np.array([2e24]), np.array([7e23])
    A = shared_site_coordinate_jacobian_m3(P, np.zeros(1), C)
    H = shared_site_coordinate_hessian_m3(P, np.zeros(1), C)
    np.testing.assert_array_equal(A[:, 0, 0], P * (1 - P / C))
    np.testing.assert_allclose(H[:, 0, 0, 0], P * (1 - P / C) * (1 - 2 * P / C), rtol=2e-13)
    assert np.count_nonzero(A) == 1 and np.count_nonzero(H) == 1


def test_zero_negative_diffusivity_still_has_coupled_physical_storage():
    device = compile_domain(prepare_confined_shared_site_ion_device(negative_diffusivity_m2_s=0.0))
    problem, y = Ionic1DProblem(device), device.initial_state().u
    plus, minus = device.ion_species_slices
    np.testing.assert_array_equal(problem.ion_fluxes(y, species="negative"), np.zeros(12))
    K = problem.storage_jacobian(y)
    assert K[minus, plus].nnz == 13 and np.all(K.diagonal()[minus] < 0.0)
    dot = np.zeros(device.state_size)
    dot[plus] = 0.1
    assert problem.inventory_rate_m2_s(y, dot, species="negative") < 0.0
    assert np.all(problem.residual(y, dot)[minus] > 0.0)


def test_shared_source_capacity_flags_and_saturation_refuse_explicitly(device):
    original = prepare_confined_shared_site_ion_device()
    for name, value, message in (("P_lim_neg", 3e24, "common"), ("P0_neg", 1e24, "summed occupancy")):
        data = original.to_input().editing_data()
        next(layer for layer in data["layers"] if layer["id"] == "absorber")["parameters"][name] = value
        prepared = resolve_device(DeviceInput.model_validate(data), original.defaults, original.resources, sources=original.sources)
        with pytest.raises(CapabilityError, match=message):
            compile_domain(prepared)
    with pytest.raises(ValueError, match="ionic binding"):
        replace(device, ion_site_contract="independent_finite_sites")
    saturated = device.initial_state().u.copy()
    saturated[3 * device.node_count:] = 1000.0
    with pytest.raises(ValueError, match="saturated|vacancy"):
        device.fields_by_species(saturated)
    with pytest.raises(CapabilityError, match="interface exchange"):
        compile_ionic_device_1d(original, ion_domain_layer="absorber", ion_domain_endpoints=("blocking", "transmissive"))
