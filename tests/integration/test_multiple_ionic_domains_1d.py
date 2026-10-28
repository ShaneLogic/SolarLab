"""Bounded actual two-domain assembly, independent physical oracles and refusals."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from examples.research_1d.ionic_multilayer import prepare_confined_dual_ion_device, prepare_confined_ion_device
from examples.research_1d.multiple_ionic_domains import DOMAINS, INTERVALS, prepare_multiple_ionic_device
from examples.research_1d.shared_site_ionic_fixture import prepare_confined_shared_site_ion_device
from solarlab.config.resolve_device import resolve_device
from solarlab.contracts.units import ELEMENTARY_CHARGE_C as Q
from solarlab.device.compile_ionic_1d import compile_ionic_device_1d
from solarlab.device.compile_multiple_ionic_1d import compile_multiple_ionic_device_1d
from solarlab.device.inputs import DeviceInput
from solarlab.equations.dc_1d import CapabilityError
from solarlab.equations.ionic_1d import Ionic1DProblem
from solarlab.equations.multiple_ionic_1d import MultipleIonic1DProblem
from solarlab.physics.ions import ion_face_flux


@pytest.fixture(scope="module", autouse=True)
def blas1():
    with threadpool_limits(limits=1):
        yield


def compile_case(kind="positive", *, right_immobile=False, counts=INTERVALS):
    return compile_multiple_ionic_device_1d(
        prepare_multiple_ionic_device(kind=kind, right_immobile=right_immobile),
        intervals_per_layer=counts, ion_domains=DOMAINS)


def trial(device, *, high_occupancy=False):
    y = device.initial_state(voltage_V=0.002).u.copy()
    for domain in device.domains:
        for index, block in enumerate(device.ion_species_slices[domain.domain_layer_id].values()):
            y[block] = (2.6 if high_occupancy else 0.0) + np.linspace(-0.13 + 0.03 * index, 0.12, domain.ion_node_count)
    return y


@pytest.mark.parametrize("kind", ["positive", "independent", "shared"])
@pytest.mark.parametrize("right_immobile", [False, True])
def test_source_bound_maps_complete_state_and_distinct_reference_inventories(kind, right_immobile):
    device = compile_case(kind, right_immobile=right_immobile)
    assert device.node_count == 13
    assert device.state_size == (47 if kind == "positive" else 55)
    np.testing.assert_array_equal(device.domains[0].ion_node_indices, [2, 3, 4, 5])
    np.testing.assert_array_equal(device.domains[1].ion_node_indices, [7, 8, 9, 10])
    assert all(domain.electronic_component is device.electronic_component for domain in device.domains)
    state = device.initial_state(voltage_V=0.002)
    device.validate_state(state)
    assert state.domain_ids == ("absorber_left", "absorber_right")
    assert state.device_sha256 == device.identity
    assert math.isclose(device.reference_inventories_m2["absorber_left"]["positive"], 1.2e17, rel_tol=3e-16)
    assert math.isclose(device.reference_inventories_m2["absorber_right"]["positive"], 1.26e17, rel_tol=3e-16)
    for domain, local in zip(device.domains, state.domain_component_states):
        original_cold = domain.initial_state(voltage_V=0.002)
        for species in domain.ion_species_names:
            np.testing.assert_array_equal(getattr(local, species + "_ion_density_m3"),
                                          getattr(original_cold, species + "_ion_density_m3"))
            np.testing.assert_allclose(getattr(local, species + "_ion_density_m3"),
                                       getattr(domain, species + "_ion_reference_m3"),
                                       rtol=4 * np.finfo(float).eps, atol=0.0)
        np.testing.assert_array_equal(domain.ion_geometry.nodes_m, device.nodes_m[domain.ion_node_indices])
    assert len(device.binding_manifest["original_ionic_inputs"]) == 5
    if right_immobile:
        assert np.all(device.domains[1].positive_ion_diffusivity_faces_m2_s == 0.0)
    with pytest.raises(ValueError):
        state.u.setflags(write=True)


def test_unequal_domain_node_counts_are_bound_to_the_full_sparse_shape():
    device = compile_case("independent", counts=(2, 3, 2, 4, 2))
    assert (device.node_count, device.domains[0].ion_node_count, device.domains[1].ion_node_count, device.state_size) == (14, 4, 5, 60)
    problem = MultipleIonic1DProblem(device, voltage_V=0.002)
    y = trial(device)
    assert problem.residual(y).shape == (60,)
    assert problem.jacobian(y).shape == problem.storage_jacobian(y).shape == (60, 60)
    device.validate_state(device.state(y, voltage_V=0.002))


@pytest.mark.parametrize("kind", ["positive", "independent", "shared"])
def test_physical_charge_oracle_each_half_cell_background_and_signed_poisson(kind):
    device = compile_case(kind)
    problem, y = MultipleIonic1DProblem(device, voltage_V=0.002), trial(device)
    _, n, p, populations = device.fields_by_domain(y, voltage_V=0.002)
    ledger = device.charge_ledger(n, p, populations)
    physical = np.zeros(device.node_count)
    for domain in device.domains:
        for index, species in enumerate(domain.ion_species_names):
            sign = 1.0 if species == "positive" else -1.0
            expected = np.zeros(device.node_count)
            expected[domain.ion_node_indices] = sign * Q * domain.ion_geometry.cell_measures_m * (
                populations[domain.domain_layer_id][index] - getattr(domain, species + "_ion_reference_m3"))
            deposited = sign * Q * device.electronic_component.geometry.cell_measures_m * ledger.entry(
                species + "_ion." + domain.domain_layer_id).net_population()
            np.testing.assert_allclose(deposited, expected, rtol=8e-15, atol=1e-20)
            physical += expected
            assert deposited[domain.ion_node_indices[0]] != 0.0
            assert deposited[domain.ion_node_indices[-1]] != 0.0
    electronic = problem.electronic_problem.residual(y[:3 * device.node_count])
    np.testing.assert_allclose(problem.residual(y)[:device.node_count] - electronic[:device.node_count],
                               physical, rtol=2e-12, atol=2e-18)
    assert physical[6] == 0.0


@pytest.mark.parametrize("kind", ["positive", "independent", "shared"])
def test_original_face_law_every_native_balance_and_exact_zero_gap_end_flux(kind):
    device = compile_case(kind)
    problem = MultipleIonic1DProblem(device, voltage_V=0.002)
    y, current = trial(device), np.zeros(device.node_count - 1)
    F, J = problem.residual(y), problem.jacobian(y)
    for domain in device.domains:
        local = device.local_coordinates(y, domain.domain_layer_id)
        phi, _, _, populations = domain.fields_by_species(local, voltage_V=0.002)
        for index, species in enumerate(domain.ion_species_names):
            sign = 1.0 if species == "positive" else -1.0
            capacity = getattr(domain, species + "_ion_site_limit_m3")
            oracle = ion_face_flux(phi[domain.ion_node_indices], populations[index], np.diff(domain.ion_geometry.nodes_m),
                getattr(domain, species + "_ion_diffusivity_faces_m2_s"), domain.electronic_component.thermal_voltage_V,
                capacity[:-1], steric_diffusion_only=True, P_lim_node=capacity,
                P_other_node=populations[1 - index] if kind == "shared" else None, drift_sign=sign)
            mapped = np.zeros(device.node_count - 1)
            mapped[domain.ion_node_indices[:-1]] = oracle
            np.testing.assert_array_equal(problem.ion_fluxes_on_electronic_faces(y, domain_id=domain.domain_layer_id, species=species), mapped)
            block = device.ion_species_slices[domain.domain_layer_id][species]
            balance = -np.diff(np.r_[0.0, oracle, 0.0])
            np.testing.assert_array_equal(F[block], balance)
            assert abs(math.fsum(balance.tolist())) <= 4 * np.finfo(float).eps * math.fsum(abs(oracle).tolist())
            assert J[block, device.node_count:3 * device.node_count].nnz == 0
            current += sign * Q * mapped
    np.testing.assert_array_equal(problem.ion_current_density_A_m2(y), current)
    np.testing.assert_array_equal(current[[0, 1, 5, 6, 10, 11]], np.zeros(6))
    left, right = device.domain_slices.values()
    assert J[left, right].nnz == J[right, left].nnz == 0


@pytest.mark.parametrize("kind", ["positive", "independent", "shared"])
def test_independent_inventories_and_full_gradients_include_only_their_own_domain(kind):
    device = compile_case(kind)
    problem, y = MultipleIonic1DProblem(device, voltage_V=0.002), trial(device)
    before = device.state(y, voltage_V=0.002).domain_inventories_m2
    left, right = device.domain_ids
    first = device.ion_species_slices[left]["positive"]
    changed = y.copy()
    changed[first] += 0.1
    after = device.state(changed, voltage_V=0.002).domain_inventories_m2
    assert after[right] == before[right]
    assert after[left]["positive"] != before[left]["positive"]
    gradients = problem.inventory_gradients(y)
    rng = np.random.default_rng(4301)
    direction = rng.uniform(-0.1, 0.1, device.state_size)
    h = 1e-4
    plus = device.state(y + h * direction, voltage_V=0.002).domain_inventories_m2
    minus = device.state(y - h * direction, voltage_V=0.002).domain_inventories_m2
    for name, species_values in gradients.items():
        outside = device.domain_slices[right if name == left else left]
        for species, gradient in species_values.items():
            assert np.all(gradient[outside] == 0.0) and np.all(gradient[:3 * device.node_count] == 0.0)
            finite = (plus[name][species] - minus[name][species]) / (2 * h)
            assert math.isclose(float(gradient @ direction), finite, rel_tol=2e-7)


@pytest.mark.parametrize("kind", ["positive", "independent", "shared"])
@pytest.mark.parametrize("alpha", [0.0, 2.5])
def test_complete_Fy_and_storage_curvature_against_directional_difference(kind, alpha):
    device = compile_case(kind)
    problem, y = MultipleIonic1DProblem(device, voltage_V=0.002), trial(device, high_occupancy=kind == "shared")
    rng = np.random.default_rng(430108)
    direction, derivative = rng.uniform(-0.1, 0.1, device.state_size), rng.uniform(-0.2, 0.2, device.state_size)
    direction[:device.node_count] *= device.electronic_component.thermal_voltage_V
    derivative[:device.node_count] *= device.electronic_component.thermal_voltage_V
    J = problem.jacobian(y, derivative, alpha=alpha)
    h = 1e-5
    finite = (problem.residual(y + h * direction, derivative + alpha * h * direction)
              - problem.residual(y - h * direction, derivative - alpha * h * direction)) / (2 * h)
    error = float(np.max(abs(finite - J @ direction) / np.maximum(abs(J) @ abs(direction), np.finfo(float).tiny)))
    print(f"two domains {kind} alpha={alpha}: complete directional error {error:.9g}")
    assert error < 2e-7
    if kind == "shared":
        populations = device.fields_by_domain(y, voltage_V=0.002)[3]
        for domain in device.domains:
            assert np.all(sum(populations[domain.domain_layer_id]) / domain.positive_ion_site_limit_m3 > 0.8)


@pytest.mark.parametrize("kind", ["positive", "independent", "shared"])
def test_physical_storage_and_immobile_local_rows_are_retained(kind):
    device = compile_case(kind, right_immobile=True)
    problem, y = MultipleIonic1DProblem(device, voltage_V=0.002), trial(device)
    rng = np.random.default_rng(1084301)
    direction = rng.uniform(-0.1, 0.1, device.state_size)
    direction[:device.node_count] *= device.electronic_component.thermal_voltage_V
    K = problem.storage_jacobian(y)
    h = 1e-5
    finite = (problem.physical_storage(y + h * direction) - problem.physical_storage(y - h * direction)) / (2 * h)
    error = float(np.max(abs(finite + K @ direction) / np.maximum(abs(K) @ abs(direction), np.finfo(float).tiny)))
    assert error < 2e-7
    right = device.domains[1]
    derivative = np.zeros(device.state_size)
    derivative[device.domain_slices[right.domain_layer_id]] = 0.2
    F = problem.residual(y, derivative)
    for species, block in device.ion_species_slices[right.domain_layer_id].items():
        np.testing.assert_array_equal(problem.ion_fluxes(y, domain_id=right.domain_layer_id, species=species),
                                      np.zeros(right.ion_node_count - 1))
        np.testing.assert_allclose(F[block], (K @ derivative)[block], rtol=5e-15, atol=0.0)
        assert np.all(K.diagonal()[block] < 0.0)
    for row in (0, device.node_count - 1, device.node_count, 2 * device.node_count - 1, 2 * device.node_count, 3 * device.node_count - 1):
        assert K.getrow(row).nnz == 0


@pytest.mark.parametrize("declarations", [
    (), DOMAINS[:1], DOMAINS + (("htl", ("blocking", "blocking")),),
    DOMAINS[::-1], (DOMAINS[0], DOMAINS[0]),
    (("missing", ("blocking", "blocking")), DOMAINS[1]),
    (("absorber_left", ("transmissive", "blocking")), DOMAINS[1]),
    (("absorber_left", ("blocking", "blocking")), ("inactive_gap", ("blocking", "blocking"))),
])
def test_absent_undeclared_exchange_touching_or_reordered_domains_refuse(declarations):
    with pytest.raises(CapabilityError):
        compile_multiple_ionic_device_1d(prepare_multiple_ionic_device(), intervals_per_layer=INTERVALS, ion_domains=declarations)


def test_undeclared_population_is_not_stripped_and_single_domain_api_is_unchanged():
    prepared = prepare_multiple_ionic_device()
    data = prepared.to_input().editing_data()
    data["layers"][2]["parameters"]["P0"] = 1e22
    changed = resolve_device(DeviceInput.model_validate(data), prepared.defaults, prepared.resources, sources=prepared.sources)
    with pytest.raises(CapabilityError, match="undeclared"):
        compile_multiple_ionic_device_1d(changed, intervals_per_layer=INTERVALS, ion_domains=DOMAINS)
    with pytest.raises(CapabilityError, match="outside-domain"):
        compile_ionic_device_1d(prepared, intervals_per_layer=INTERVALS, ion_domain_layer="absorber_left", ion_domain_endpoints=("blocking", "blocking"))


def test_scalar_state_shape_identity_and_layout_validation():
    device = compile_case()
    y, state = trial(device), device.initial_state(voltage_V=0.002)
    for value in (True, float("nan"), float("inf"), "0.002"):
        with pytest.raises(ValueError):
            MultipleIonic1DProblem(device, voltage_V=value)
    with pytest.raises(ValueError):
        MultipleIonic1DProblem(device, generation_scale=-1.0)
    with pytest.raises(ValueError):
        device.state(np.r_[y, 0.0], voltage_V=0.002)
    with pytest.raises(ValueError, match="real numeric"):
        device.state(y.astype(complex) + 1j, voltage_V=0.002)
    with pytest.raises(ValueError):
        device.validate_state(replace(state, device_sha256="other"))
    with pytest.raises(ValueError):
        device.validate_state(replace(state, domain_ids=state.domain_ids[::-1]))
    first, second = state.domain_component_states
    wrong_fields = replace(first, positive_ion_density_m3=first.positive_ion_density_m3 * 1.1,
                           domain_inventory_m2=first.domain_inventory_m2 * 1.1)
    with pytest.raises(ValueError, match="physical inventories"):
        device.validate_state(replace(state, domain_component_states=(wrong_fields, second)))
    manifest = device.binding_manifest
    manifest["ionic_state_blocks"][0]["stop"] += 1
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False)
    with pytest.raises(ValueError, match="state block"):
        replace(device, binding_manifest_json=encoded, sha256=hashlib.sha256(encoded.encode()).hexdigest())
    with pytest.raises(ValueError):
        compile_multiple_ionic_device_1d(prepare_multiple_ionic_device(), intervals_per_layer=True, ion_domains=DOMAINS)


@pytest.mark.parametrize("kind", ["positive", "independent", "shared"])
@pytest.mark.parametrize("immobile", [False, True])
def test_new_composition_does_not_mutate_old_single_domain_identity_F_J_K_or_profile(kind, immobile):
    factory = {"positive": prepare_confined_ion_device, "independent": prepare_confined_dual_ion_device,
               "shared": prepare_confined_shared_site_ion_device}[kind]
    prepared = factory()
    if immobile:
        data = prepared.to_input().editing_data()
        next(layer for layer in data["layers"] if layer["id"] == "absorber")["parameters"].update(D_ion=0.0, D_ion_neg=0.0)
        prepared = resolve_device(DeviceInput.model_validate(data), prepared.defaults, prepared.resources, sources=prepared.sources)
    def old():
        return compile_ionic_device_1d(prepared, intervals_per_layer=(2, 3, 2),
                                      ion_domain_layer="absorber", ion_domain_endpoints=("blocking", "blocking"))
    before = old()
    u = before.initial_state(voltage_V=0.002).u.copy()
    u[3 * before.node_count:] = np.linspace(-0.1, 0.1, u.size - 3 * before.node_count)
    problem = Ionic1DProblem(before, voltage_V=0.002)
    F, J, K = problem.residual(u), problem.jacobian(u).toarray(), problem.storage_jacobian(u).toarray()
    compile_case(kind, right_immobile=immobile)
    after = old()
    assert before.identity == after.identity and before.binding_manifest_json == after.binding_manifest_json
    replay = Ionic1DProblem(after, voltage_V=0.002)
    np.testing.assert_array_equal(replay.residual(u), F)
    np.testing.assert_array_equal(replay.jacobian(u).toarray(), J)
    np.testing.assert_array_equal(replay.storage_jacobian(u).toarray(), K)
    for species in after.ion_species_names:
        np.testing.assert_array_equal(getattr(before, species + "_ion_reference_m3"), getattr(after, species + "_ion_reference_m3"))


@pytest.mark.parametrize("kind", ["positive", "independent"])
def test_whole_flux_original_flag_has_actual_two_domain_directional_parity(kind):
    prepared = prepare_multiple_ionic_device(kind=kind)
    data = prepared.to_input().editing_data()
    data["settings"]["ion_steric_diffusion_only"] = False
    prepared = resolve_device(DeviceInput.model_validate(data), prepared.defaults, prepared.resources, sources=prepared.sources)
    device = compile_multiple_ionic_device_1d(prepared, intervals_per_layer=INTERVALS, ion_domains=DOMAINS)
    problem, y = MultipleIonic1DProblem(device, voltage_V=0.002), trial(device)
    rng = np.random.default_rng(430109)
    direction, derivative = rng.uniform(-0.1, 0.1, device.state_size), rng.uniform(-0.2, 0.2, device.state_size)
    direction[:device.node_count] *= device.electronic_component.thermal_voltage_V
    derivative[:device.node_count] *= device.electronic_component.thermal_voltage_V
    J = problem.jacobian(y, derivative, alpha=2.5)
    h = 1e-5
    finite = (problem.residual(y + h * direction, derivative + 2.5 * h * direction)
              - problem.residual(y - h * direction, derivative - 2.5 * h * direction)) / (2 * h)
    error = float(np.max(abs(finite - J @ direction) / np.maximum(abs(J) @ abs(direction), np.finfo(float).tiny)))
    print(f"two domains whole-flux {kind}: complete directional error {error:.9g}")
    assert error < 2e-7
    assert all(domain.binding_manifest["original_ionic_settings"]["ion_steric_diffusion_only"] is False
               for domain in device.domains)
