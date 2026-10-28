"""Bounded 37/13 independent-site assembly, signed charge and physical storage."""

from __future__ import annotations

from dataclasses import replace
import json
import math

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from examples.research_1d.ionic_multilayer import (
    prepare_confined_dual_ion_device, prepare_confined_ion_device,
)
from solarlab.config.resolve_device import resolve_device
from solarlab.device.compile_ionic_1d import compile_ionic_device_1d
from solarlab.device.defaults import DefaultCatalog
from solarlab.device.inputs import DeviceInput
from solarlab.equations.dc_1d import CapabilityError
from solarlab.equations.ionic_1d import Ionic1DProblem
from solarlab.materials.source import SourceDocument
from solarlab.physics.band_structure.temperature import ion_diffusivity_at_temperature
from solarlab.physics.ions import ion_face_flux, single_site_density

Q = 1.602176634e-19


@pytest.fixture(scope="module", autouse=True)
def blas1():
    with threadpool_limits(limits=1):
        yield


def compile_domain(prepared):
    return compile_ionic_device_1d(
        prepared, intervals_per_layer=(12, 12, 12), ion_domain_layer="absorber",
        ion_domain_endpoints=("blocking", "blocking"),
    )


@pytest.fixture(scope="module", params=[True, False], ids=["diffusion_only", "whole_flux"])
def case(request):
    device = compile_domain(prepare_confined_dual_ion_device(diffusion_only=request.param))
    problem = Ionic1DProblem(device, voltage_V=0.002)
    y = device.initial_state(voltage_V=0.002).u.copy()
    y[device.ion_species_slices[0]] = np.linspace(-0.15, 0.12, device.ion_node_count)
    y[device.ion_species_slices[1]] = np.linspace(0.12, -0.09, device.ion_node_count)
    return device, problem, y


def test_actual_3N_2M_layout_keeps_both_bound_populations_and_inventories(case):
    device, _, y = case
    assert (device.node_count, device.ion_node_count, device.state_size) == (37, 13, 137)
    assert device.ion_species_names == ("positive", "negative") and device.ion_species_count == 2
    assert device.ion_species_slices == (slice(111, 124), slice(124, 137))
    np.testing.assert_array_equal(device.ion_node_indices, np.arange(12, 25))
    state = device.state(y, voltage_V=0.002)
    assert state.device_sha256 == device.identity and state.electronic_state.u.shape == (111,)
    assert state.positive_ion_density_m3.shape == state.negative_ion_density_m3.shape == (13,)
    assert np.all(state.positive_ion_density_m3 > 0.0) and np.all(state.negative_ion_density_m3 > 0.0)
    for species, density in zip(device.ion_species_names, device.fields_by_species(y, voltage_V=0.002)[3]):
        expected = math.fsum((device.ion_geometry.cell_measures_m * density).tolist())
        assert state.domain_inventories_m2[species] == expected
    reference_inventory = math.fsum((device.ion_geometry.cell_measures_m * np.full(device.ion_node_count, 1e24)).tolist())
    assert device.reference_inventory_m2 == reference_inventory
    assert math.isclose(device.negative_reference_inventory_m2, 1.8e17, rel_tol=3e-16)
    for species, raw_D in (("positive", 1e-16), ("negative", 5e-17)):
        expected_D = ion_diffusivity_at_temperature(raw_D, device.electronic_component.parameters.temperature_K, 0.58)
        np.testing.assert_array_equal(getattr(device, species + "_ion_diffusivity_faces_m2_s"), np.full(12, expected_D))
    assert state.domain_inventory_m2 != state.negative_domain_inventory_m2
    manifest = device.binding_manifest
    assert manifest["state_layout"]["size"] == 137
    assert manifest["ion_model"]["site_contract"] == "independent_finite_sites"
    with pytest.raises(ValueError):
        state.negative_ion_density_m3.setflags(write=True)
    with pytest.raises(ValueError, match="fields_by_species"):
        device.fields(y, voltage_V=0.002)
    with pytest.raises(ValueError, match="negative population"):
        device.charge_ledger(state.electronic_state.electron_density_m3,
                            state.electronic_state.hole_density_m3, state.positive_ion_density_m3)


def test_uniform_population_drift_has_independent_opposite_sign_analytic_limit(case):
    device, problem, _ = case
    cold = device.initial_state(voltage_V=0.002)
    phi, _, _, populations = device.fields_by_species(cold.u, voltage_V=0.002)
    dx = np.diff(device.ion_geometry.nodes_m)
    gradient = np.diff(phi[device.ion_node_indices]) / dx
    for species, sign, density in zip(device.ion_species_names, (1.0, -1.0), populations):
        capacity = getattr(device, species + "_ion_site_limit_m3")
        diffusion = getattr(device, species + "_ion_diffusivity_faces_m2_s")
        crowding = 1.0 if device.steric_model == "historical_diffusion_only_steric_v1" else 1.0 / (1.0 - density[:-1] / capacity[:-1])
        expected = -sign * diffusion * density[:-1] * gradient / device.electronic_component.thermal_voltage_V * crowding
        actual = problem.ion_fluxes(cold.u, species=species)
        np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=0.0)
        assert np.all(actual != 0.0)
    assert np.all(problem.ion_fluxes(cold.u) * problem.ion_fluxes(cold.u, species="negative") < 0.0)


def test_each_original_particle_law_and_signed_current_deposition_is_conservative(case):
    device, problem, y = case
    phi, _, _, populations = device.fields_by_species(y, voltage_V=0.002)
    full_fluxes = []
    residual = problem.residual(y)
    for species, sign, density, block in zip(device.ion_species_names, (1.0, -1.0), populations, device.ion_species_slices):
        capacity = getattr(device, species + "_ion_site_limit_m3")
        expected = ion_face_flux(phi[device.ion_node_indices], density, np.diff(device.ion_geometry.nodes_m),
                                 getattr(device, species + "_ion_diffusivity_faces_m2_s"),
                                 device.electronic_component.thermal_voltage_V, capacity[:-1],
                                 steric_diffusion_only=device.steric_model == "historical_diffusion_only_steric_v1",
                                 P_lim_node=capacity, drift_sign=sign)
        np.testing.assert_array_equal(problem.ion_fluxes(y, species=species), expected)
        mapped = np.zeros(device.node_count - 1)
        mapped[device.ion_node_indices[:-1]] = expected
        np.testing.assert_array_equal(problem.ion_fluxes_on_electronic_faces(y, species=species), mapped)
        full_fluxes.append(np.r_[0.0, mapped, 0.0])
        balance = -np.diff(np.r_[0.0, expected, 0.0])
        np.testing.assert_array_equal(residual[block], balance)
        assert abs(math.fsum(balance.tolist())) <= 4 * np.finfo(float).eps * math.fsum(abs(expected).tolist())
    actual_current = problem.ion_current_density_A_m2(y)
    np.testing.assert_array_equal(actual_current, Q * (full_fluxes[0][1:-1] - full_fluxes[1][1:-1]))
    charge_balance = np.zeros(device.node_count)
    charge_balance[device.ion_node_indices] = Q * (residual[device.ion_species_slices[0]] - residual[device.ion_species_slices[1]])
    reconstructed = -np.diff(np.r_[0.0, actual_current, 0.0])
    bound = 16 * np.finfo(float).eps * max(float(np.max(abs(actual_current))), np.finfo(float).tiny)
    np.testing.assert_allclose(reconstructed, charge_balance, rtol=0.0, atol=bound)


def test_half_cell_charge_backgrounds_and_Poisson_columns_have_opposite_signs(case):
    device, problem, y = case
    N, idx = device.node_count, device.ion_node_indices
    _, n, p, populations = device.fields_by_species(y, voltage_V=0.002)
    positive, negative = populations
    ledger = device.charge_ledger(n, p, positive, negative)
    assert ledger.entry("positive_ion").charge_number == 1
    assert ledger.entry("negative_ion").charge_number == -1
    assert ledger.entry("negative_ion").group != ledger.entry("positive_ion").group
    for species, density, reference in zip(device.ion_species_names, populations,
                                          (device.positive_ion_reference_m3, device.negative_ion_reference_m3)):
        entry = ledger.entry(species + "_ion")
        physical = np.zeros(N)
        physical[idx] = entry.charge_number * Q * device.ion_geometry.cell_measures_m * (density - reference)
        projected = entry.charge_number * Q * device.electronic_component.geometry.cell_measures_m * entry.net_population()
        np.testing.assert_allclose(projected, physical, rtol=5e-15, atol=1e-20)
        np.testing.assert_array_equal(entry.background_m3, device.project_ion_density(reference))
    expected_charge = np.zeros(N)
    expected_charge[idx] = Q * device.ion_geometry.cell_measures_m * (
        (positive - device.positive_ion_reference_m3) - (negative - device.negative_ion_reference_m3)
    )
    difference = problem.residual(y)[:N] - problem.electronic_problem.residual(y[:3 * N])[:N]
    np.testing.assert_allclose(difference, expected_charge, rtol=1e-13, atol=1e-18)
    J = problem.jacobian(y)
    for local, node in enumerate(idx):
        assert J[node, 3 * N + local] > 0.0
        assert J[node, 3 * N + device.ion_node_count + local] < 0.0


def test_storage_and_nonzero_ydot_keep_independent_complete_M_blocks(case):
    device, problem, y = case
    N = device.node_count
    dot = np.linspace(-0.2, 0.15, device.state_size)
    dot[:N] *= device.electronic_component.thermal_voltage_V
    K, J = problem.storage_jacobian(y), problem.jacobian(y, dot)
    base_J = problem.jacobian(y)
    base_R, dynamic_R = problem.residual(y), problem.residual(y, dot)
    # Compare complete electronic rows directly: subtracting their large DC
    # transport totals need not resolve a much smaller storage increment.
    np.testing.assert_array_equal(dynamic_R[:N], base_R[:N])
    np.testing.assert_array_equal(dynamic_R[N:3 * N], problem.electronic_problem.residual(y[:3 * N], dot[:3 * N])[N:3 * N])
    for species, density, block in zip(device.ion_species_names, device.fields_by_species(y, voltage_V=0.002)[3], device.ion_species_slices):
        capacity = getattr(device, species + "_ion_site_limit_m3")
        partial = density * (1.0 - density / capacity)
        expected_storage = -device.ion_geometry.cell_measures_m * partial
        np.testing.assert_array_equal(K.diagonal()[block], expected_storage)
        np.testing.assert_array_equal(dynamic_R[block], base_R[block] - device.ion_geometry.cell_measures_m * partial * dot[block])
        second = partial * (1.0 - 2.0 * density / capacity)
        expected_second = -device.ion_geometry.cell_measures_m * second * dot[block]
        # Check the assembled column before a second lossy subtraction of
        # large transport derivatives; full independent FD checks follow.
        np.testing.assert_array_equal(J.diagonal()[block], base_J.diagonal()[block] + expected_second)
        expected_rate = math.fsum((-expected_storage * dot[block]).tolist())
        assert problem.inventory_rate_m2_s(y, dot, species=species) == expected_rate
        assert J[block, N:3 * N].nnz == 0
    positive, negative = device.ion_species_slices
    assert J[positive, negative].nnz == J[negative, positive].nnz == 0
    assert K[positive, negative].nnz == K[negative, positive].nnz == 0
    assert np.all(K.diagonal()[3 * N:] < 0.0)


@pytest.mark.parametrize("alpha", [0.0, 2.5])
def test_full_dual_Fy_and_storage_have_independent_directional_check(case, alpha):
    device, problem, y = case
    rng = np.random.default_rng(40872)
    direction = rng.uniform(-0.1, 0.1, device.state_size)
    derivative = rng.uniform(-0.2, 0.2, device.state_size)
    direction[:device.node_count] *= device.electronic_component.thermal_voltage_V
    derivative[:device.node_count] *= device.electronic_component.thermal_voltage_V
    J = problem.jacobian(y, derivative, alpha=alpha)
    h = 1e-5
    numerical = (problem.residual(y + h * direction, derivative + alpha * h * direction)
                 - problem.residual(y - h * direction, derivative - alpha * h * direction)) / (2 * h)
    scale = abs(J) @ abs(direction)
    error = float(np.max(abs(numerical - J @ direction) / np.maximum(scale, np.finfo(float).tiny)))
    print(f"37/13 dual {device.steric_model} alpha={alpha:g}: directional error {error:.9g}")
    assert error < 2e-7


@pytest.mark.parametrize("positive_D,negative_D", [(0.0, 5e-17), (1e-16, 0.0), (0.0, 0.0)])
def test_zero_diffusivity_retains_population_charge_and_dynamic_storage(positive_D, negative_D):
    device = compile_domain(prepare_confined_dual_ion_device(
        positive_diffusivity_m2_s=positive_D, negative_diffusivity_m2_s=negative_D,
    ))
    problem = Ionic1DProblem(device, voltage_V=0.002)
    y = device.initial_state(voltage_V=0.002).u.copy()
    dot = np.zeros(device.state_size)
    dot[3 * device.node_count:] = 0.01
    assert device.state_size == 137
    K = problem.storage_jacobian(y)
    for species, diffusion, block in zip(device.ion_species_names, (positive_D, negative_D), device.ion_species_slices):
        assert np.all(K.diagonal()[block] < 0.0)
        assert device.state(y, voltage_V=0.002).domain_inventories_m2[species] > 0.0
        assert problem.inventory_rate_m2_s(y, dot, species=species) > 0.0
        if diffusion == 0.0:
            np.testing.assert_array_equal(problem.ion_fluxes(y, species=species), np.zeros(device.ion_node_count - 1))
            np.testing.assert_array_equal(problem.residual(y)[block], np.zeros(device.ion_node_count))
            np.testing.assert_array_equal(problem.residual(y, dot)[block], (K @ dot)[block])
            assert problem.jacobian(y)[block, :].nnz == 0


@pytest.mark.parametrize("field", ["negative_ion_reference_m3", "negative_ion_site_limit_m3", "negative_ion_logit_reference", "negative_ion_diffusivity_faces_m2_s"])
def test_negative_arrays_cannot_be_rebound_or_dropped_from_source(case, field):
    device, _, _ = case
    value = getattr(device, field)
    with pytest.raises(ValueError, match="ionic binding"):
        replace(device, **{field: value + 0.4 if field == "negative_ion_logit_reference" else value * 2.0})
    with pytest.raises(ValueError, match="cannot be stripped"):
        replace(device, negative_ion_reference_m3=np.empty(0), negative_ion_site_limit_m3=np.empty(0),
                negative_ion_logit_reference=np.empty(0), negative_ion_diffusivity_faces_m2_s=np.empty(0))


def test_explicit_domain_shared_sites_and_missing_negative_reference_refuse():
    original = prepare_confined_dual_ion_device()
    for path, value, message in (("outside", 1e21, "outside-domain"),
                                 ("shared", True, "shared-site"),
                                 ("reference", 0.0, "negative-ion reference")):
        data = original.to_input().editing_data()
        if path == "outside":
            data["layers"][0]["parameters"]["P0_neg"] = value
        elif path == "shared":
            data["settings"]["ion_steric_shared_site"] = value
        else:
            next(layer for layer in data["layers"] if layer["id"] == "absorber")["parameters"]["P0_neg"] = value
        changed = resolve_device(DeviceInput.model_validate(data), original.defaults, original.resources, sources=original.sources)
        with pytest.raises(CapabilityError, match=message):
            compile_domain(changed)
    with pytest.raises(CapabilityError, match="interface exchange"):
        compile_ionic_device_1d(original, ion_domain_layer="absorber", ion_domain_endpoints=("blocking", "transmissive"))


def test_positive_only_151ee28_identity_fields_and_old_state_shape_remain_compatible():
    prepared = prepare_confined_ion_device()
    # The example binds its absolute source path. Use a stable logical source
    # for this frozen 151ee28 identity so checkout location is not a test input.
    source = SourceDocument("research_1d/device.json", prepared.sources[0].content)
    defaults = DefaultCatalog.from_mapping({
        **json.loads(source.content)["default_catalog"],
        "evidence": [[source.id, source.sha256]],
    })
    stable = resolve_device(prepared.to_input(), defaults, prepared.resources, sources=(source,))
    device = compile_domain(stable)
    assert device.identity == "29f79d5b84cd31409010ebcf85798965bfcfe52bbd310106e66f87343730cea3"
    assert device.identity != compile_domain(prepared).identity
    assert device.ion_species_names == ("positive",) and device.state_size == 124
    y = device.initial_state(voltage_V=0.002).u.copy()
    y[3 * device.node_count:] = np.linspace(-0.15, 0.12, device.ion_node_count)
    expected = single_site_density(device.positive_ion_reference_m3, device.positive_ion_site_limit_m3,
                                   device.positive_ion_logit_reference, y[3 * device.node_count:])
    np.testing.assert_array_equal(device.fields(y, voltage_V=0.002)[3], expected)
    old = device.state(y, voltage_V=0.002)
    assert old.negative_ion_density_m3.size == 0 and old.negative_domain_inventory_m2 == 0.0
    assert old.domain_inventories_m2 == {"positive": old.domain_inventory_m2}
    assert "ion_species" not in device.binding_manifest["state_layout"]
    dual = compile_domain(prepare_confined_dual_ion_device())
    assert dual.identity != old.device_sha256
    with pytest.raises(ValueError, match="3.N.*M"):
        dual.state(old.u, voltage_V=old.voltage_V)
    with pytest.raises(ValueError, match="not assembled"):
        Ionic1DProblem(device).ion_fluxes(y, species="negative")
