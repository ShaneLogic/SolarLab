"""Real production 1D DC, independent tangents, and capability boundaries."""

from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from examples.research_1d.steady import SaturatingRecombination, load_device
from solarlab.config.resolve_device import resolve_device
from solarlab.device.compile_1d import compile_device_1d
from solarlab.device.inputs import DeviceInput
from solarlab.equations.dc_1d import CapabilityError, DC1DProblem
from solarlab.materials.resources import ResourceLibrary, ResourceTable
from solarlab.materials.source import SourceDocument
from solarlab.mesh import INTEGER_ENDPOINT_V1, RIGHT_LAYER_WINS
from solarlab.solvers.nonlinear.newton import Termination
from solarlab.solvers.nonlinear.production import (
    production_coordinates,
    production_residual_scaling,
    production_steady_linearization,
    production_steady_problem,
)
from solarlab.solvers.nonlinear import steady_1d as steady_solver
from solarlab.solvers.nonlinear.steady_1d import (
    SteadyOptions, SteadyStateError, solve_equilibrium, solve_steady_state,
)

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module", autouse=True)
def blas1():
    with threadpool_limits(limits=1):
        yield


@pytest.fixture(scope="module")
def prepared():
    return load_device()


@pytest.fixture(scope="module")
def device(prepared):
    return compile_device_1d(prepared)


@pytest.fixture(scope="module")
def dark(device):
    return solve_equilibrium(device)


@pytest.fixture(scope="module")
def light(device):
    return solve_steady_state(device)


def edited(prepared, document):
    return resolve_device(
        DeviceInput.model_validate(document), prepared.defaults,
        prepared.resources, sources=prepared.sources,
    )


def test_compiles_real_prepared_configuration_and_named_geometry(prepared, device):
    assert device.node_count == 73
    assert device.geometry.name == INTEGER_ENDPOINT_V1
    assert device.geometry.material_owner_convention == RIGHT_LAYER_WINS
    assert len(device.identity) == 64
    assert device.identity != prepared.content_sha256
    assert device.nodes_m[0] == 0.0
    assert device.nodes_m[-1] == pytest.approx(sum(layer.thickness for layer in prepared.layers))
    widths = device.geometry.cell_measures_m
    spacing = np.diff(device.nodes_m)
    assert widths[0] == spacing[0]
    assert widths[-1] == spacing[-1]
    assert np.sum(widths) > device.geometry.extent_m
    for plane in device.geometry.interfaces:
        assert plane.shared_node is not None
    first_interface = device.geometry.interfaces[0].shared_node
    assert device.parameters.band_structure.physical_band_gap_eV[first_interface] == 1.6
    assert device.parameters.conduction_dos_m3[first_interface] == 1.0e25
    assert device.recombination.coefficients.tau_n_s.shape == (device.node_count,)
    np.testing.assert_allclose(
        device.generation.source_m3_s * widths, device.generation.absorbed_m2_s,
        rtol=3e-16, atol=0.0,
    )
    assert device.generation.areal_generation_m2_s <= prepared.values["settings"]["Phi"]


def test_real_dark_equilibrium_has_common_fermi_and_zero_every_face(device, dark):
    problem = DC1DProblem(device)
    assert dark.converged
    assert dark.density_basin_initializations == 0
    assert dark.density_basin_stages == ()
    assert dark.newton.termination == Termination.CONVERGED
    assert all(result.accepted for result in dark.preparation)
    assert all(gate.passed for gate in dark.physical_gates)
    np.testing.assert_array_equal(problem.fluxes(dark.state.u).electron.flux, 0.0)
    np.testing.assert_array_equal(problem.fluxes(dark.state.u).hole.flux, 0.0)
    np.testing.assert_array_equal(dark.contact_current_density_A_m2, 0.0)
    np.testing.assert_allclose(
        dark.state.electron_density_m3 * dark.state.hole_density_m3,
        device.recombination.coefficients.ni_sq_m6, rtol=4e-14, atol=0.0,
    )
    assert dark.state.voltage_V == 0.0
    assert dark.state.electron_density_m3[-1] == device.channel("n", "right").reservoir_density_m3
    assert dark.state.hole_density_m3[0] == device.channel("p", "left").reservoir_density_m3


def test_real_illumination_obeys_current_continuity_and_photon_budget(device, light):
    output = -device.voltage_polarity * light.current_density_A_m2
    assert light.converged
    assert light.generation_scale == 1.0
    assert light.density_basin_initializations == 5
    assert light.density_basin_stages == tuple(
        (0.0, value) for value in (1e-8, 1e-6, 1e-4, 1e-2, 1.0)
    )
    assert all(point.accepted for point in light.preparation)
    assert np.min(output) > 0.0
    assert np.max(output) <= device.generation_limit_A_m2 * (1.0 + 1e-10)
    assert np.ptp(light.current_density_A_m2) < 1e-8
    np.testing.assert_allclose(
        light.contact_current_density_A_m2,
        light.current_density_A_m2[[0, -1]], rtol=0.0, atol=1e-8,
    )
    limits = {gate.name: gate for gate in light.physical_gates}
    assert limits["poisson_balance"].value < 1e-11
    assert limits["continuity_current_bound"].value < 1e-8
    assert limits["all_face_current_spread"].value < 1e-8


def test_refined_fixed_reservoir_device_starts_from_dark_with_unchanged_gates(prepared):
    compiled = compile_device_1d(prepared, intervals_per_layer=96)
    assert compiled.node_count == 289
    assert all(channel.kind == "fixed_reservoir" for channel in compiled.channels)
    result = solve_steady_state(compiled)
    assert result.converged
    assert result.density_basin_initializations == 5
    assert all(gate.passed for gate in result.physical_gates)
    currents = np.r_[result.current_density_A_m2, result.contact_current_density_A_m2]
    assert np.ptp(currents) < 1e-8
    output = -compiled.voltage_polarity * currents
    assert 0.0 < np.min(output) <= np.max(output) <= compiled.generation_limit_A_m2


@pytest.mark.parametrize("state_kind", ["dark", "light", "storage"])
def test_independent_directional_fy_in_physical_states(device, dark, light, state_kind):
    rng = np.random.default_rng(20261028)
    count = device.node_count
    if state_kind == "light":
        u = light.state.u.copy()
        problem = DC1DProblem(device, generation_scale=1.0)
        udot = None
    elif state_kind == "dark":
        u = dark.state.u.copy()
        u[:count] += 2e-4 * rng.normal(size=count)
        u[count:] += 0.03 * rng.normal(size=2 * count)
        # A large constant G on depleted dark rows can bury a finite difference.
        problem = DC1DProblem(device, generation_scale=0.0)
        udot = None
    else:
        u = dark.state.u.copy()
        problem = DC1DProblem(device)
        udot = 1e4 * rng.normal(size=3 * count)
    matrix = problem.jacobian(u, udot)
    absolute = abs(matrix)
    for _ in range(4):
        direction = np.r_[0.02 * rng.normal(size=count), rng.normal(size=2 * count)]
        step = 1e-5
        numerical = (
            problem.residual(u + step * direction, udot)
            - problem.residual(u - step * direction, udot)
        ) / (2 * step)
        analytical = matrix @ direction
        scale = np.maximum(absolute @ np.abs(direction), 1e-20)
        assert np.max(np.abs(numerical - analytical) / scale) < 2e-8
    assert matrix.nnz < 30 * count


def test_independent_storage_and_combined_implicit_tangent(device, dark):
    count = device.node_count
    rng = np.random.default_rng(31)
    u = dark.state.u
    problem = DC1DProblem(device)
    direction = np.r_[0.02 * rng.normal(size=count), rng.normal(size=2 * count)]
    storage = problem.storage_jacobian(u)
    numerical = (problem.residual(u, direction) - problem.residual(u, -direction)) / 2
    scale = np.maximum(abs(storage) @ abs(direction), 1e-20)
    assert np.max(abs(numerical - storage @ direction) / scale) < 2e-12
    udot = 1e4 * rng.normal(size=3 * count)
    alpha, step = 1e4, 1e-5
    combined = problem.jacobian(u, udot, alpha=alpha)
    numerical = (
        problem.residual(u + step * direction, udot + alpha * step * direction)
        - problem.residual(u - step * direction, udot - alpha * step * direction)
    ) / (2 * step)
    scale = np.maximum(abs(combined) @ abs(direction), 1e-20)
    assert np.max(abs(numerical - combined @ direction) / scale) < 2e-8
    assert storage[:count].nnz == 0
    for offset in (count, 2 * count):
        assert storage[offset].nnz == 0
        assert storage[offset + count - 1].nnz == 0


def test_real_biased_warm_start_preserves_physical_state_and_contact_precision(device, light):
    rebased = device.rebase_coordinates(light.state.u, 0.0, 0.05)
    _, n, p = device.fields(rebased, voltage_V=0.05)
    np.testing.assert_allclose(n, light.state.electron_density_m3, rtol=3e-14, atol=0.0)
    np.testing.assert_allclose(p, light.state.hole_density_m3, rtol=3e-14, atol=0.0)
    biased = solve_steady_state(device, initial_state=light.state, voltage_V=0.05)
    assert biased.converged
    assert biased.state.voltage_V == 0.05
    assert biased.voltage_V == 0.05
    assert biased.density_basin_initializations == 0
    assert biased.density_basin_stages == ()
    assert np.ptp(biased.current_density_A_m2) < 1e-8
    assert biased.state.electron_density_m3[-1] == device.channel("n", "right").reservoir_density_m3


def test_reversed_junction_and_potential_gauge(prepared):
    for reverse, gauge in ((True, 0.0), (False, 0.4)):
        document = prepared.to_input().editing_data()
        if reverse:
            document["layers"] = list(reversed(document["layers"]))
        document["settings"]["phi_left"] = gauge
        compiled = compile_device_1d(edited(prepared, document))
        assert compiled.voltage_polarity == (-1.0 if reverse else 1.0)
        result = solve_steady_state(compiled)
        assert result.density_basin_initializations == 5
        assert all(point.accepted for point in result.preparation)
        assert np.min(-compiled.voltage_polarity * result.current_density_A_m2) > 0.0
        assert np.ptp(result.current_density_A_m2) < 1e-8
        assert result.state.potential_V[0] == pytest.approx(gauge, abs=1e-13)
        biased = solve_steady_state(compiled, initial_state=result.state, voltage_V=0.05)
        assert biased.converged
        assert np.ptp(biased.current_density_A_m2) < 1e-8


@pytest.mark.parametrize("contact_kind", ["selective", "robin"])
def test_production_finite_and_blocking_contact_channels(prepared, contact_kind):
    document = prepared.to_input().editing_data()
    if contact_kind == "selective":
        document["contacts"] = [
            dict(id="left", side="left", layer="htl", S_n=0.0, S_p=200.0),
            dict(id="right", side="right", layer="etl", S_n=200.0, S_p=0.0),
        ]
    else:
        document["settings"]["flat_band_contacts"] = True
    compiled = compile_device_1d(edited(prepared, document))
    equilibrium = solve_equilibrium(compiled)
    result = solve_steady_state(compiled)
    assert equilibrium.converged and result.converged
    assert result.density_basin_initializations == 5
    assert result.density_basin_stages == tuple(
        (0.0, value) for value in (1e-8, 1e-6, 1e-4, 1e-2, 1.0)
    )
    assert np.ptp(np.r_[result.current_density_A_m2, result.contact_current_density_A_m2]) < 1e-8
    assert all(gate.passed for gate in result.physical_gates)
    rng = np.random.default_rng(44)
    direction = np.r_[0.02 * rng.normal(size=compiled.node_count), rng.normal(size=2 * compiled.node_count)]
    problem = DC1DProblem(compiled, generation_scale=1.0)
    matrix = problem.jacobian(result.state.u)
    step = 1e-5
    numerical = (
        problem.residual(result.state.u + step * direction)
        - problem.residual(result.state.u - step * direction)
    ) / (2 * step)
    scale = np.maximum(abs(matrix) @ abs(direction), 1e-20)
    assert np.max(abs(numerical - matrix @ direction) / scale) < 2e-8
    if contact_kind == "selective":
        assert compiled.channel("n", "left").kind == "blocking"
        assert compiled.channel("p", "right").kind == "blocking"


def test_failed_light_stage_preserves_failure_and_never_advances(prepared, light, monkeypatch):
    document = prepared.to_input().editing_data()
    document["settings"]["flat_band_contacts"] = True
    compiled = compile_device_1d(edited(prepared, document))
    failed = replace(
        light.newton, state=np.full(light.state.u.shape, np.nan), accepted=False,
        termination=Termination.LINE_SEARCH_FAILED, message="injected failed light stage",
    )
    original_point = steady_solver._solve_point
    original_seed = steady_solver.density_basin_seed
    seed_inputs = []
    accepted = []

    def record_seed(problem, u):
        seed_inputs.append((problem.generation_scale, np.array(u, copy=True)))
        return original_seed(problem, u)

    def stop_second_light(device, u, voltage_V, generation_scale, options, **kwargs):
        if generation_scale == 1e-6:
            raise SteadyStateError("injected second light stage", failed)
        problem, result = original_point(device, u, voltage_V, generation_scale, options, **kwargs)
        accepted.append(result)
        return problem, result

    monkeypatch.setattr(steady_solver, 'density_basin_seed', record_seed)
    monkeypatch.setattr(steady_solver, "_solve_point", stop_second_light)
    with pytest.raises(SteadyStateError) as failure:
        solve_steady_state(compiled)
    assert failure.value.result is failed
    assert not failed.accepted
    assert [generation for generation, _ in seed_inputs] == [1e-8, 1e-6]
    assert all(point.accepted for point in accepted)
    np.testing.assert_array_equal(seed_inputs[-1][1], accepted[-1].state)
    assert all(np.all(np.isfinite(u)) for _, u in seed_inputs)


def test_local_model_injection_has_independent_partials_and_real_solution(prepared, device, light):
    coefficient = np.zeros(device.node_count)
    coefficient[device.parameters.layer_masks[1]] = 5e-15
    model = SaturatingRecombination(coefficient, device.parameters.intrinsic_product_m6, 1e21)
    n, p = light.state.electron_density_m3, light.state.hole_density_m3
    rate = model.evaluate(n, p)
    for name, tangent in (("n", rate.d_rate_dn * n), ("p", rate.d_rate_dp * p)):
        h = 1e-30
        numerical = model.evaluate(n + 1j * h * n if name == "n" else n, p + 1j * h * p if name == "p" else p).rate.imag / h
        np.testing.assert_allclose(numerical, tangent, rtol=2e-13, atol=1e-30)
    compiled = compile_device_1d(prepared, additional_recombination=model)
    assert compiled.identity != device.identity
    result = solve_steady_state(compiled)
    assert result.converged
    assert np.ptp(result.current_density_A_m2) < 1e-8
    assert abs(result.current_density_A_m2[0] - light.current_density_A_m2[0]) > 1e-7
    with pytest.raises(ValueError, match="same compiled device"):
        solve_steady_state(compiled, initial_state=light.state)


@pytest.mark.parametrize("feature", [
    "ions", "fermi_dirac", "dopant_ionization", "band_gap_narrowing",
    "field_mobility", "interface_recombination", "optics", "manual_contacts",
    "ion_steric", "qualified_solver_policy",
])
def test_unsupported_enabled_physics_fails_before_execution(prepared, feature):
    document = prepared.to_input().editing_data()
    parameters = document["layers"][1]["parameters"]
    if feature == "ions":
        parameters["D_ion"] = 1e-15
    elif feature == "fermi_dirac":
        parameters["carrier_statistics"] = "fermi_dirac"
    elif feature == "dopant_ionization":
        parameters["dopant_ionization_model"] = "discrete_level"
    elif feature == "band_gap_narrowing":
        parameters["band_gap_narrowing_model"] = "slotboom"
    elif feature == "field_mobility":
        parameters["pf_gamma_n"] = 1e-4
    elif feature == "interface_recombination":
        document["interfaces"] = [dict(id="interface", left="htl", right="absorber", v_n=1.0, v_p=1.0)]
    elif feature == "optics":
        parameters["n_optical"] = 2.0
    elif feature == "ion_steric":
        document["settings"]["ion_steric_shared_site"] = True
    elif feature == "qualified_solver_policy":
        document["settings"]["jv_solver_policy"] = "cancellation_safe_qf_required"
    else:
        document["settings"]["built_in_potential_mode"] = "legacy_manual"
    with pytest.raises(CapabilityError, match="unsupported_classical_1d_dc"):
        compile_device_1d(edited(prepared, document))


def test_inconsistent_dos_intrinsic_product_is_not_silently_replaced(prepared):
    document = prepared.to_input().editing_data()
    document["layers"][1]["parameters"]["ni"] *= 2.0
    with pytest.raises(CapabilityError, match="inconsistent_intrinsic_product"):
        compile_device_1d(edited(prepared, document))


def test_fully_blocking_device_requires_an_inventory_constraint(prepared):
    document = prepared.to_input().editing_data()
    document["contacts"] = [
        dict(id="left", side="left", layer="htl", S_n=0.0, S_p=0.0),
        dict(id="right", side="right", layer="etl", S_n=0.0, S_p=0.0),
    ]
    with pytest.raises(CapabilityError, match="fully_blocking_contacts"):
        compile_device_1d(edited(prepared, document))


def test_fixed_generation_resource_is_consumed_without_interpolation(prepared, device, light):
    document = prepared.to_input().editing_data()
    document["fixed_generation"] = "fixed"
    content = json.dumps({
        "coordinate_m": device.nodes_m.tolist(),
        "generation_m3_s": device.generation.source_m3_s.tolist(),
    }).encode()
    table = ResourceTable("fixed", "fixed_generation", SourceDocument("fixed_generation.json", content))
    configured = resolve_device(
        DeviceInput.model_validate(document), prepared.defaults,
        ResourceLibrary((table,)), sources=prepared.sources,
    )
    compiled = compile_device_1d(configured)
    assert compiled.generation.method == "fixed"
    np.testing.assert_array_equal(compiled.generation.source_m3_s, device.generation.source_m3_s)
    result = solve_steady_state(compiled)
    np.testing.assert_allclose(result.current_density_A_m2, light.current_density_A_m2, rtol=0.0, atol=1e-8)
    with pytest.raises(CapabilityError, match="resource coordinates must exactly match"):
        compile_device_1d(configured, intervals_per_layer=12)


@pytest.mark.parametrize("binding_kind", ["temperature", "grading", "doping_profile"])
def test_static_parameter_binding_has_real_dc_solution(prepared, binding_kind):
    document = prepared.to_input().editing_data()
    if binding_kind == "temperature":
        document["settings"]["T"] = 330.0
    elif binding_kind == "grading":
        document["settings"]["band_grading"] = True
        document["layers"][1]["parameters"].update(Eg_back=1.7, chi_back=3.95)
    else:
        document["layers"][1]["parameters"].update(
            N_D_bulk=1e22, doping_profile_shape="gaussian", doping_decay_length=8e-8,
        )
    compiled = compile_device_1d(edited(prepared, document))
    result = solve_steady_state(compiled)
    assert result.converged and all(gate.passed for gate in result.physical_gates)
    assert np.ptp(result.current_density_A_m2) < 1e-8
    if binding_kind == "temperature":
        assert compiled.parameters.temperature_K == 330.0
    elif binding_kind == "grading":
        assert compiled.parameters.layer_states[1].graded
    else:
        assert np.ptp(compiled.parameters.donor_density_m3[compiled.parameters.layer_masks[1]]) > 0.0


def test_concrete_production_adapters_and_honest_budget_failure(device):
    problem = production_steady_problem(device)
    u = np.r_[device.reference_potential_V, np.zeros(2 * device.node_count)]
    assert isinstance(problem, DC1DProblem)
    assert production_steady_linearization(problem, u).shape == (u.size, u.size)
    assert production_residual_scaling(device).row_weights.shape == u.shape
    assert production_coordinates(device).max_step.shape == u.shape
    with pytest.raises(SteadyStateError) as failure:
        solve_steady_state(device, options=SteadyOptions(max_iterations=1))
    assert not failure.value.result.accepted
    assert failure.value.result.termination == Termination.BUDGET_EXHAUSTED


def test_fresh_process_real_solve_blocks_test_and_legacy_imports():
    script = """
import importlib.abc, json, pathlib, sys
class BlockImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'perovskite_sim','backend','tests','p0403_dc_scaffold','dc_device_support'}:
            raise RuntimeError('forbidden production import: ' + fullname)
sys.meta_path.insert(0, BlockImports())
from solarlab.config.resolve_device import resolve_device
from solarlab.device.defaults import DefaultCatalog
from solarlab.device.inputs import DeviceInput
from solarlab.device.compile_1d import compile_device_1d
from solarlab.materials.resources import ResourceLibrary
from solarlab.materials.source import SourceDocument
from solarlab.solvers.nonlinear.steady_1d import solve_equilibrium, solve_steady_state
source=SourceDocument(sys.argv[1], pathlib.Path(sys.argv[1]).read_bytes())
data=json.loads(source.content)
catalog=DefaultCatalog.from_mapping({**data['default_catalog'],'evidence':[[source.id,source.sha256]]})
prepared=resolve_device(DeviceInput.model_validate(data['device']),catalog,ResourceLibrary(),sources=(source,))
device=compile_device_1d(prepared)
dark=solve_equilibrium(device)
light=solve_steady_state(device)
print(json.dumps({'dark':dark.converged,'light':light.converged,'current':float(light.current_density_A_m2[0])}))
"""
    environment = {
        **os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
    }
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, str(ROOT / "examples/research_1d/device.json")],
        env=environment, capture_output=True, text=True, timeout=60, check=False,
    )
    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["dark"] and summary["light"]
    assert summary["current"] < 0.0
