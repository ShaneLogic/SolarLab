"""Bounded real prepared-ion assembly, inventory, storage and independent tangents."""

from __future__ import annotations

from dataclasses import replace
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from examples.research_1d.ionic_residual import prepare_single_ion_device
from examples.research_1d.steady import load_device
from solarlab.config.resolve_device import resolve_device
from solarlab.device.compile_ionic_1d import compile_ionic_device_1d
from solarlab.device.inputs import DeviceInput
from solarlab.equations.dc_1d import CapabilityError
from solarlab.equations.ionic_1d import Ionic1DProblem
from solarlab.mesh.geometry import HALF_ENDPOINT_V1, INTEGER_ENDPOINT_V1
from solarlab.physics.band_structure.temperature import ion_diffusivity_at_temperature

Q = 1.602176634e-19
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module", autouse=True)
def blas1():
    with threadpool_limits(limits=1):
        yield


def changed(prepared, *, parameters=None, settings=None, defect_model=None):
    data = prepared.to_input().editing_data()
    data["layers"][0]["parameters"].update(parameters or {})
    data["settings"].update(settings or {})
    if defect_model is not None:
        data["layers"][0]["defect_model"] = defect_model
        data["layers"][0]["defect_schema_version"] = "solarlab-explicit-bulk-defects-v1"
        data["layers"][0]["bulk_defects"] = [{
            "id": "neutral_probe", "name": "Neutral probe", "distribution": {
                "kind": "single_level", "normalization": "integrated_total", "total_density_m3": 1e20,
                "center_eV_above_vb": 0.4, "width_convention": "not_applicable",
            },
            "charge_transition": "neutral", "neutral_reference": "all_occupancies", "degeneracy": 1.0,
            "kinetics": {"sigma_n_m2": 1e-19, "sigma_p_m2": 1e-19,
                         "thermal_velocity_n_m_s": 1e5, "thermal_velocity_p_m_s": 1e5},
        }]
    return resolve_device(DeviceInput.model_validate(data), prepared.defaults, prepared.resources, sources=prepared.sources)


@pytest.fixture(scope="module", params=[False, True], ids=["whole_flux", "diffusion_only"])
def case(request):
    prepared = prepare_single_ion_device(diffusion_only=request.param)
    compiled = compile_ionic_device_1d(prepared, intervals_per_layer=12)
    problem = Ionic1DProblem(compiled)
    y = compiled.initial_state().u.copy()
    count = compiled.node_count
    position = (compiled.nodes_m - compiled.nodes_m[0]) / compiled.ion_geometry.extent_m
    y[:count] += 0.002 * np.sin(np.pi * position)
    y[count:2 * count] = 0.05 * np.sin(2 * np.pi * position)
    y[2 * count:3 * count] = 0.03 * np.cos(np.pi * position)
    y[3 * count:] = 0.1 * np.cos(np.pi * position)
    return prepared, compiled, problem, y


def test_original_ionic_inputs_domain_and_physical_inventory_are_bound(case):
    prepared, compiled, _, _ = case
    state = compiled.initial_state()
    assert compiled.original_prepared_sha256 == prepared.content_sha256
    assert compiled.identity != compiled.electronic_component.identity
    assert "ion_free" not in compiled.capabilities
    assert compiled.ion_geometry.name == HALF_ENDPOINT_V1
    assert compiled.ion_geometry.geometric_cells
    assert compiled.electronic_component.geometry.name == INTEGER_ENDPOINT_V1
    np.testing.assert_array_equal(state.positive_ion_density_m3, compiled.positive_ion_reference_m3)
    thickness = prepared.layers[0].thickness
    assert math.isclose(compiled.reference_inventory_m2, 1e24 * thickness, rel_tol=3e-16)
    assert compiled.domain_bounds_m == (0.0, thickness)
    assert state.device_sha256 == compiled.identity
    assert state.domain_inventory_m2 == compiled.reference_inventory_m2
    manifest = compiled.binding_manifest
    assert manifest["original_ionic_inputs"][0]["P0"] == 1e24
    assert manifest["original_ionic_inputs"][0]["E_a_ion"] == 0.58
    assert manifest["electronic_component"]["projection"]
    # The separately named component has no ion population; the full state does.
    assert np.all(compiled.electronic_component.parameters.ion_diffusivity_m2_s == 0.0)
    assert prepared.content_sha256 == compiled.original_prepared_sha256


def test_uniform_reference_is_compensated_without_stripping_population(case):
    _, compiled, problem, _ = case
    state = compiled.initial_state()
    count = compiled.node_count
    result = problem.residual(state.u)
    electronic = problem.electronic_problem.residual(state.u[:3 * count])
    np.testing.assert_array_equal(result[:3 * count], electronic)
    np.testing.assert_array_equal(result[3 * count:], np.zeros(count))
    assert compiled.charge_ledger(state.electronic_state.electron_density_m3,
                                 state.electronic_state.hole_density_m3,
                                 state.positive_ion_density_m3).entry("positive_ion").background_m3 is not None


def test_ion_charge_changes_poisson_through_the_same_ledger(case):
    _, compiled, problem, y = case
    count = compiled.node_count
    phi, n, p, density = compiled.fields(y)
    ledger = compiled.charge_ledger(n, p, density)
    left, right = compiled.electronic_component.contact_potentials(0.0)
    expected = compiled.electronic_component.poisson.residual(phi, ledger.space_charge_C_m3(), left, right)
    actual = problem.residual(y)[:count]
    np.testing.assert_array_equal(actual, expected)
    original = problem.electronic_problem.residual(y[:3 * count])[:count]
    charge_change = Q * compiled.electronic_component.poisson.charge_jacobian() * (density - compiled.positive_ion_reference_m3)
    np.testing.assert_allclose(actual - original, charge_change, rtol=2e-14, atol=1e-20)
    assert np.max(abs(charge_change)) > 0.0
    assert ledger.space_charge_derivative("positive_ion") == Q


def test_ion_ledger_deposition_matches_physical_storage_at_both_contacts(case):
    _, compiled, _, y = case
    _, n, p, density = compiled.fields(y)
    ledger = compiled.charge_ledger(n, p, density)
    deposited = compiled.electronic_component.geometry.cell_measures_m * Q * ledger.entry("positive_ion").net_population()
    physical = compiled.ion_geometry.cell_measures_m * Q * (density - compiled.positive_ion_reference_m3)
    np.testing.assert_allclose(deposited, physical, rtol=3e-16, atol=0.0)
    assert compiled.ion_charge_weight_ratio[0] == 0.5
    assert compiled.ion_charge_weight_ratio[-1] == 0.5
    np.testing.assert_array_equal(compiled.ion_charge_weight_ratio[1:-1], np.ones(compiled.node_count - 2))
    np.testing.assert_array_equal(compiled.binding_manifest["ion_charge_deposition"]["physical_density_ratio"],
                                  compiled.ion_charge_weight_ratio)


def test_conservative_faces_and_physical_endpoint_storage(case):
    _, compiled, problem, y = case
    count = compiled.node_count
    flux = problem.ion_fluxes(y)
    balance = -np.diff(np.r_[0.0, flux, 0.0])
    np.testing.assert_array_equal(problem.residual(y)[3 * count:], balance)
    bound = 4 * np.finfo(float).eps * math.fsum(abs(flux).tolist())
    assert abs(math.fsum(balance.tolist())) <= bound
    _, _, _, density = compiled.fields(y)
    partial = density * (1.0 - density / compiled.positive_ion_site_limit_m3)
    K = problem.storage_jacobian(y)
    np.testing.assert_array_equal(K.diagonal()[3 * count:], -compiled.ion_geometry.cell_measures_m * partial)
    assert K[:count].nnz == 0
    assert K[3 * count:, :3 * count].nnz == 0
    assert K.diagonal()[3 * count] < 0.0 and K.diagonal()[-1] < 0.0
    derivative = np.zeros(4 * count)
    # Instantaneous ion-only rate from the original balance, not a timestep or projection.
    derivative[3 * count:] = balance / (compiled.ion_geometry.cell_measures_m * partial)
    np.testing.assert_allclose(problem.residual(y, derivative)[3 * count:], 0.0, rtol=0.0, atol=bound)
    assert abs(problem.inventory_rate_m2_s(y, derivative)) <= 2 * bound
    np.testing.assert_allclose(problem.residual(y, derivative) - problem.residual(y), K @ derivative,
                               rtol=2e-14, atol=1e-20)


@pytest.mark.parametrize("alpha", [0.0, 2.5])
def test_full_Fy_and_storage_chain_have_independent_directional_check(case, alpha):
    _, compiled, problem, y = case
    count = compiled.node_count
    rng = np.random.default_rng(40708)
    direction = rng.uniform(-0.1, 0.1, 4 * count)
    direction[:count] *= compiled.electronic_component.thermal_voltage_V
    derivative = rng.uniform(-0.2, 0.2, 4 * count)
    derivative[:count] *= compiled.electronic_component.thermal_voltage_V
    J = problem.jacobian(y, derivative, alpha=alpha)
    step = 1e-5
    expected = (problem.residual(y + step * direction, derivative + alpha * step * direction)
                - problem.residual(y - step * direction, derivative - alpha * step * direction)) / (2 * step)
    actual = J @ direction
    conditioning = abs(J) @ abs(direction)
    error = np.max(abs(actual - expected) / np.maximum(conditioning, np.finfo(float).tiny))
    print(f"{compiled.steric_model}, alpha={alpha:g}: full directional error {float(error):.9g}")
    assert error < 2e-7


def test_zero_diffusion_retains_population_background_and_storage():
    prepared = changed(prepare_single_ion_device(), parameters={"D_ion": 0.0})
    compiled = compile_ionic_device_1d(prepared, intervals_per_layer=6)
    problem = Ionic1DProblem(compiled)
    y = compiled.initial_state().u.copy()
    y[3 * compiled.node_count:] = 0.2
    np.testing.assert_array_equal(problem.ion_fluxes(y), np.zeros(compiled.node_count - 1))
    np.testing.assert_array_equal(problem.residual(y)[3 * compiled.node_count:], np.zeros(compiled.node_count))
    assert np.max(abs(problem.residual(y)[1:compiled.node_count - 1])) > 0.0
    assert np.all(problem.storage_jacobian(y).diagonal()[3 * compiled.node_count:] < 0.0)
    assert compiled.reference_inventory_m2 > 0.0


def test_configured_activation_energy_and_inactive_capacity_are_not_lost():
    prepared = changed(prepare_single_ion_device(), parameters={"E_a_ion": 0.35}, settings={"T": 330.0})
    compiled = compile_ionic_device_1d(prepared, intervals_per_layer=6)
    expected = ion_diffusivity_at_temperature(1e-16, 330.0, 0.35)
    np.testing.assert_array_equal(compiled.positive_ion_diffusivity_faces_m2_s,
                                  np.full(compiled.node_count - 1, expected))
    altered = compile_ionic_device_1d(changed(prepared, parameters={"P_lim_neg": 2e28}), intervals_per_layer=6)
    assert altered.identity != compiled.identity
    np.testing.assert_array_equal(altered.positive_ion_reference_m3, compiled.positive_ion_reference_m3)


@pytest.mark.parametrize("field", [
    "positive_ion_reference_m3", "positive_ion_site_limit_m3", "positive_ion_logit_reference",
])
def test_original_binding_rejects_reference_capacity_and_logit_counterexamples(field):
    compiled = compile_ionic_device_1d(prepare_single_ion_device(diffusion_only=True), intervals_per_layer=12)
    value = getattr(compiled, field)
    changed_value = value + 0.4 if field == "positive_ion_logit_reference" else 2 * value
    with pytest.raises(ValueError, match="ionic binding"):
        replace(compiled, **{field: changed_value})


@pytest.mark.parametrize("parameters,settings,message", [
    ({"D_ion_neg": 1e-17}, {}, "negative-ion reference"),
    ({}, {"ion_steric_shared_site": True}, "shared-site"),
    ({"P0": 0.0}, {}, "strictly positive"),
    ({"P0": 1e27}, {}, "unsaturated"),
])
def test_unassembled_ionic_capabilities_refuse_explicitly(parameters, settings, message):
    prepared = changed(prepare_single_ion_device(), parameters=parameters, settings=settings)
    with pytest.raises(CapabilityError, match=message):
        compile_ionic_device_1d(prepared, intervals_per_layer=6)


def test_configured_negative_population_survives_zero_mobility_with_physical_storage():
    prepared = changed(prepare_single_ion_device(), parameters={"P0_neg": 1e23, "D_ion_neg": 0.0})
    compiled = compile_ionic_device_1d(prepared, intervals_per_layer=6)
    state = compiled.initial_state()
    problem = Ionic1DProblem(compiled)
    negative = compiled.ion_species_slices[1]
    assert compiled.ion_species_count == 2 and compiled.state_size == 5 * compiled.node_count
    np.testing.assert_array_equal(state.negative_ion_density_m3, np.full(compiled.ion_node_count, 1e23))
    assert state.negative_domain_inventory_m2 > 0.0
    np.testing.assert_array_equal(problem.ion_fluxes(state.u, species="negative"), np.zeros(compiled.ion_node_count - 1))
    storage = problem.storage_jacobian(state.u)
    assert np.all(storage.diagonal()[negative] < 0.0)
    derivative = np.zeros(compiled.state_size)
    derivative[negative] = 0.1
    np.testing.assert_array_equal(problem.residual(state.u, derivative)[negative], (storage @ derivative)[negative])


def test_multilayer_domains_and_electronic_defects_are_not_silently_dropped():
    with pytest.raises(CapabilityError, match="one electrical layer"):
        compile_ionic_device_1d(load_device(), intervals_per_layer=6)
    prepared = changed(prepare_single_ion_device(), defect_model="explicit_quasi_steady")
    compiled = compile_ionic_device_1d(prepared, intervals_per_layer=6)
    reaction = compiled.electronic_component.recombination
    model = reaction.explicit_defects
    assert model is not None
    assert model.species[0].identifier == prepared.layers[0].id + "/neutral_probe"
    assert np.all(model.owned_nodes)
    # Electronic projection must retain the configured neutral rate, not just
    # accept its declaration while silently using lifetime SRH.
    evaluation = reaction.evaluate(np.full(compiled.node_count, 1e20),
                                   np.full(compiled.node_count, 2e20))
    assert evaluation.explicit is not None
    assert np.all(evaluation.explicit.total_rate_m3_s > 0.0)
    np.testing.assert_array_equal(evaluation.srh.rate, evaluation.explicit.total_rate_m3_s)


@pytest.mark.parametrize("diffusion_only", [False, True])
def test_historical_regularization_kinks_and_site_saturation_are_refused(diffusion_only):
    prepared = changed(prepare_single_ion_device(diffusion_only=diffusion_only),
                       parameters={"P0": 0.999999, "P_lim": 1.0})
    compiled = compile_ionic_device_1d(prepared, intervals_per_layer=6)
    problem = Ionic1DProblem(compiled)
    y = compiled.initial_state().u
    assert np.all(np.isfinite(problem.residual(y)))
    with pytest.raises(ValueError, match="regularization kink"):
        problem.jacobian(y)
    saturated = y.copy()
    saturated[3 * compiled.node_count:] = 1000.0
    with pytest.raises(ValueError, match="saturated"):
        problem.residual(saturated)


def test_executable_case_has_no_legacy_import_or_execution():
    script = """
import importlib.abc, sys
class RefuseLegacy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'perovskite_sim' or fullname.startswith('perovskite_sim.'):
            raise AssertionError('legacy import forbidden: ' + fullname)
sys.meta_path.insert(0, RefuseLegacy())
from examples.research_1d.ionic_residual import main
main()
"""
    result = subprocess.run([sys.executable, "-c", script], cwd=ROOT, env={
        **os.environ, "PYTHONDONTWRITEBYTECODE": "1", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
    }, capture_output=True, text=True, check=False, timeout=30)
    assert result.returncode == 0, result.stderr
    assert '"node_count": 13' in result.stdout
