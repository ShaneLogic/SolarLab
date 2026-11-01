"""Real connected 2D algebra, independent geometric/flux oracles; no Newton."""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
import json
import sys
from typing import Any

import numpy as np
import pytest
from scipy.sparse import csc_matrix, coo_matrix

from examples.research_1d.steady import load_device
from examples.research_2d import dc_assembly as example
from solarlab.config.resolve_device import resolve_device
from solarlab.contracts import AREA, LENGTH
from solarlab.contracts.units import ELEMENTARY_CHARGE_C as Q
from solarlab.device.compile_2d import compile_device_2d
from solarlab.device.inputs import DeviceInput
from solarlab.device.resolved import PreparedDevice
from solarlab.discretization.face_average import VACUUM_PERMITTIVITY_F_M as EPS
from solarlab.equations.dc_1d import CapabilityError, DC1DProblem
from solarlab.equations.dc_2d import CompiledDevice2D, DC2DProblem
from solarlab.experiments.two_dimensional.inputs import JV2DInput
from solarlab.mesh.geometry_2d import RectangularGeometry2D


@pytest.fixture(scope="module")
def prepared() -> PreparedDevice:
    return load_device()


@pytest.fixture(scope="module")
def device(prepared: PreparedDevice) -> CompiledDevice2D:
    return compile_device_2d(prepared, example.spatial_input())


def reprepare(prepared: PreparedDevice, edit: Any) -> PreparedDevice:
    data = prepared.to_input().editing_data()
    edit(data)
    return resolve_device(DeviceInput.model_validate(data), prepared.defaults, prepared.resources, sources=prepared.sources)


def test_physical_areas_faces_and_nonuniform_lateral_geometry(device: CompiledDevice2D) -> None:
    g = RectangularGeometry2D(np.array([0, .13, .4, 1.0]) * 1e-6, device.geometry.y_geometry)
    c = g.contract_geometry
    assert c.volume_unit == AREA and c.face_unit == LENGTH
    assert g.shape == (13, 4)
    y = g.y_geometry.nodes_m
    # Exact rational midpoints avoid subtractive loss at the far, small half
    # cell. This is an independent geometry oracle, not a new endpoint rule.
    def exact_widths(nodes: np.ndarray) -> list[Fraction]:
        values = [Fraction.from_float(float(value)) for value in nodes]
        faces = [values[0], *((left + right) / 2 for left, right in zip(values, values[1:])), values[-1]]
        return [right - left for left, right in zip(faces, faces[1:])]
    expected_areas = np.array([[float(wy * wx) for wx in exact_widths(g.x_m)] for wy in exact_widths(y)])
    np.testing.assert_allclose(c.volumes.reshape(g.shape), expected_areas, rtol=8*np.finfo(float).eps, atol=0)
    np.testing.assert_allclose(c.volumes.sum(), (y[-1] - y[0]) * (g.x_m[-1] - g.x_m[0]), rtol=1e-15)
    left, right = c.pairs.T
    assert np.all(right[:g.x_face_count] - left[:g.x_face_count] == 1)
    assert np.all(left[:g.x_face_count] // g.nx == right[:g.x_face_count] // g.nx)
    assert np.all(right[g.x_face_count:] - left[g.x_face_count:] == g.nx)
    assert not c.volumes.flags.writeable and not c.pairs.flags.writeable


def test_uniform_fields_reduce_to_1d_residual_jacobian_and_current(device: CompiledDevice2D) -> None:
    d, g = device, device.geometry
    u1, u2 = example.coordinates(d, lateral=False)
    p1 = DC1DProblem(d.column, example.VOLTAGE_V, 1.0)
    p2 = DC2DProblem(d, example.VOLTAGE_V, 1.0)
    weights = np.tile(g.x_widths_m, g.ny)
    weights[g.contact_nodes] = 1.0  # Dirichlet/log-density constraints are not weighted.
    weights = np.tile(weights, 3)
    f1, f2 = p1.residual(u1), p2.residual(u2) / weights
    for actual, expected in zip(f2.reshape(3, -1), d.extrude_coordinates(f1).reshape(3, -1)):
        assert np.max(abs(actual - expected)) < 1e-13 * np.max(abs(expected))
    rows = np.arange(3 * d.node_count)
    cols = np.repeat(np.arange(3 * g.ny), g.nx)
    lift = coo_matrix((np.ones(rows.size), (rows, cols)), shape=(rows.size, 3 * g.ny)).tocsc()
    actual_j = (p2.jacobian(u2) @ lift).toarray() / weights[:, None]
    expected_j = (lift @ p1.jacobian(u1)).toarray()
    row_scale = np.maximum(np.max(abs(expected_j), axis=1), 1e-30)
    assert np.max(np.max(abs(actual_j - expected_j), axis=1) / row_scale) < 1e-13
    np.testing.assert_allclose(p2.terminal_current_density_A_m2(u2), p1.contact_current_density(u1), rtol=2e-14)
    fluxes = p2.fluxes(u2)
    assert not np.any(fluxes.electron.flux[:g.x_face_count])
    assert not np.any(fluxes.hole.flux[:g.x_face_count])
    assert not np.any(p2.displacement(u2)[:g.x_face_count])


def test_lateral_perturbation_has_real_neighbor_coupling(device: CompiledDevice2D, monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("2D assembly delegated to independent 1D columns")
    monkeypatch.setattr(DC1DProblem, "residual", forbidden)
    monkeypatch.setattr(DC1DProblem, "jacobian", forbidden)
    _, u = example.coordinates(device, lateral=True)
    problem = DC2DProblem(device, example.VOLTAGE_V, 1.0)
    assert np.all(np.isfinite(problem.residual(u)))
    fluxes = problem.fluxes(u)
    g, count = device.geometry, device.node_count
    assert np.max(abs(fluxes.electron.flux[:g.x_face_count])) > 1e20
    assert np.max(abs(fluxes.hole.flux[:g.x_face_count])) > 1e20
    assert np.max(abs(problem.displacement(u)[:g.x_face_count])) > 1e-8
    matrix = problem.jacobian(u)
    center = 6 * g.nx + 2
    for neighbor in (center - 1, center + 1, center - g.nx, center + g.nx):
        assert matrix[center, neighbor] > 0.0
        assert matrix[count + center, count + neighbor] != 0.0
        assert matrix[2 * count + center, 2 * count + neighbor] != 0.0


@pytest.mark.parametrize("potential_only", [False, True])
def test_full_directional_jacobian_uses_existing_1d_criterion(device: CompiledDevice2D, potential_only: bool) -> None:
    _, u = example.coordinates(device, lateral=True)
    problem = DC2DProblem(device, example.VOLTAGE_V, 0.0)
    matrix = problem.jacobian(u)
    np.testing.assert_array_equal(matrix.toarray(), DC2DProblem(device, example.VOLTAGE_V, 1.0).jacobian(u).toarray())
    count = device.node_count
    rng = np.random.default_rng(772)
    direction = np.r_[.02 * rng.normal(size=count), rng.normal(size=2 * count)]
    if potential_only:
        direction[count:] = 0.0
    h = example.DIRECTIONAL_STEP
    numerical = (problem.residual(u - 2*h*direction) - 8*problem.residual(u - h*direction)
                 + 8*problem.residual(u + h*direction) - problem.residual(u + 2*h*direction)) / (12*h)
    scale = np.maximum(abs(matrix) @ abs(direction), 1e-20)
    assert np.max(abs(numerical - matrix @ direction) / scale) < 2e-8


def test_gauss_matches_independent_two_axis_stencil(device: CompiledDevice2D) -> None:
    d, g = device, device.geometry
    _, u = example.coordinates(d, lateral=True)
    problem = DC2DProblem(d, example.VOLTAGE_V, 1.0)
    phi, n, p = d.fields(u, voltage_V=example.VOLTAGE_V)
    rho = d.charge_ledger(n, p).space_charge_C_m3().reshape(g.shape)
    phi = phi.reshape(g.shape)
    y = d.column.nodes_m
    wx, wy = g.x_widths_m, g.y_geometry.cell_measures_m
    eps = d.relative_permittivity_y
    expected = np.zeros(g.shape)
    for j in range(1, g.ny - 1):
        for i in range(g.nx):
            value = rho[j, i] * wx[i] * wy[j]
            for neighbor in (i - 1, i + 1):
                if 0 <= neighbor < g.nx:
                    value += EPS * eps[j] * wy[j] * (phi[j, neighbor] - phi[j, i]) / abs(g.x_m[neighbor] - g.x_m[i])
            for neighbor in (j - 1, j + 1):
                harmonic = 2 * eps[j] * eps[neighbor] / (eps[j] + eps[neighbor])
                value += EPS * harmonic * wx[i] * (phi[neighbor, i] - phi[j, i]) / abs(y[neighbor] - y[j])
            expected[j, i] = value
    actual = problem.residual(u)[:d.node_count].reshape(g.shape)
    np.testing.assert_allclose(actual[1:-1], expected[1:-1], rtol=1e-12, atol=1e-24)


def test_generation_control_volume_budget_and_carrier_source(device: CompiledDevice2D) -> None:
    d, g = device, device.geometry
    absorbed = d.column.generation.areal_generation_m2_s
    np.testing.assert_allclose(d.generation_limit_A_m, Q * absorbed * g.lateral_length_m, rtol=1e-15)
    np.testing.assert_allclose(d.generation_y.absorbed_m2_s, d.column.generation.absorbed_m2_s, rtol=0, atol=0)
    np.testing.assert_array_equal(d.generation_y.source_m3_s[1:-1], d.column.generation.source_m3_s[1:-1])
    np.testing.assert_array_equal(d.generation_y.source_m3_s[[0, -1]], 2 * d.column.generation.source_m3_s[[0, -1]])
    _, u = example.coordinates(d, lateral=True)
    dark = DC2DProblem(d, example.VOLTAGE_V, 0.0).residual(u)
    lit = DC2DProblem(d, example.VOLTAGE_V, 1.0).residual(u)
    expected = d.generation_source_m3_s * g.contract_geometry.volumes
    expected[g.contact_nodes] = 0.0
    for offset in (d.node_count, 2*d.node_count):
        np.testing.assert_allclose(lit[offset:offset+d.node_count] - dark[offset:offset+d.node_count], expected, rtol=3e-15, atol=1.0)
    np.testing.assert_array_equal(lit[:d.node_count], dark[:d.node_count])


def test_particle_balance_and_terminal_normalization(device: CompiledDevice2D) -> None:
    d, g = device, device.geometry
    _, u = example.coordinates(d, lateral=True)
    # Perturb a contact node too, so missing lateral contact-cell inflow is detectable.
    u[1] += .001
    u[d.node_count + 1] += .02
    problem = DC2DProblem(d, example.VOLTAGE_V, 1.0)
    fluxes = problem.fluxes(u)
    _, n, p = d.fields(u, voltage_V=example.VOLTAGE_V)
    source = (d.generation_source_m3_s - problem.recombination_rate(n, p).rate) * g.contract_geometry.volumes
    left, right = g.contract_geometry.pairs.T
    for flux, contact in zip((fluxes.electron.flux, fluxes.hole.flux), problem.contact_particle_fluxes(u)):
        physical = source.copy()
        weighted = flux * g.contract_geometry.face_measures
        np.add.at(physical, left, -weighted)
        np.add.at(physical, right, weighted)
        physical[:g.nx] += contact[0] * g.x_widths_m
        physical[-g.nx:] -= contact[1] * g.x_widths_m
        scale = max(np.max(abs(weighted)), np.max(abs(source)), 1.0)
        assert np.max(abs(physical[g.contact_nodes])) / scale < 2e-15
    np.testing.assert_allclose(problem.terminal_current_A_m(u) / g.lateral_length_m, problem.terminal_current_density_A_m2(u), rtol=0, atol=0)
    # Pair generation cancels between signed species; closed lateral faces leave
    # the summed terminal currents equal to their nearest y-face sums even off steady state.
    np.testing.assert_allclose(problem.terminal_current_A_m(u), problem.cross_section_current_A_m(u)[[0, -1]], rtol=2e-14)


def test_width_changes_integrated_current_but_not_uniform_density(prepared: PreparedDevice, device: CompiledDevice2D) -> None:
    wide = compile_device_2d(prepared, JV2DInput(kind="jv_2d", lateral_length="2 um", Nx=4, Ny_per_layer=4, lateral_bc="neumann"))
    u1, uniform = example.coordinates(device, lateral=False)
    wide_u = wide.extrude_coordinates(u1)
    narrow_problem, wide_problem = DC2DProblem(device, example.VOLTAGE_V), DC2DProblem(wide, example.VOLTAGE_V)
    np.testing.assert_allclose(wide_problem.terminal_current_A_m(wide_u), 2 * narrow_problem.terminal_current_A_m(uniform), rtol=1e-14)
    np.testing.assert_allclose(wide_problem.terminal_current_density_A_m2(wide_u), narrow_problem.terminal_current_density_A_m2(uniform), rtol=1e-14)
    assert wide.identity != device.identity and wide.geometry.sha256 != device.geometry.sha256


@pytest.mark.parametrize("field,value", [("lateral_bc", "periodic"), ("lateral_length", None), ("Ny_per_layer", 1), ("settle_t", 1e-6)])
def test_unbound_spatial_modes_are_rejected(prepared: PreparedDevice, field: str, value: Any) -> None:
    data = example.spatial_input().editing_data()
    data[field] = value
    with pytest.raises(ValueError):
        compile_device_2d(prepared, JV2DInput.model_validate(data))


@pytest.mark.parametrize("field,value", [("P0", 1e20), ("v_sat_n", 1e5), ("pf_gamma_p", 1e-4), ("n_optical", 2.0)])
def test_unbound_device_physics_is_rejected(prepared: PreparedDevice, field: str, value: float) -> None:
    updated = reprepare(prepared, lambda data: data["layers"][1]["parameters"].update({field: value}))
    with pytest.raises(CapabilityError):
        compile_device_2d(updated, example.spatial_input())


def test_incoherent_optics_is_rejected_before_column_compilation(prepared: PreparedDevice, monkeypatch: pytest.MonkeyPatch) -> None:
    updated = reprepare(prepared, lambda data: data["layers"][1]["parameters"].update(incoherent=True))
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("unsupported incoherent declaration reached column compilation")
    monkeypatch.setattr("solarlab.device.compile_2d.compile_device_1d", forbidden)
    with pytest.raises(CapabilityError, match=r"layers\.absorber\.incoherent"):
        compile_device_2d(updated, example.spatial_input())


def test_identity_tracks_declared_source_and_state_is_immutable(prepared: PreparedDevice, device: CompiledDevice2D) -> None:
    updated = reprepare(prepared, lambda data: data["settings"].update(Phi=1e21))
    changed = compile_device_2d(updated, example.spatial_input())
    assert changed.identity != device.identity
    np.testing.assert_allclose(changed.generation_source_m3_s, device.generation_source_m3_s / 2, rtol=0, atol=0)
    _, u = example.coordinates(device, lateral=True)
    state = device.state(u, voltage_V=example.VOLTAGE_V)
    assert state.potential_V.shape == device.geometry.shape
    with pytest.raises(ValueError):
        state.potential_V[0, 0] = 1.0
    with pytest.raises(ValueError):
        replace(state, electron_density_m3=-state.electron_density_m3)


def test_runnable_example_and_lossless_solver_free_read(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "assembly.json"
    monkeypatch.setattr(sys, "argv", ["dc_assembly", "--output", str(path)])
    assert example.main() == 0
    value = json.loads(path.read_text())
    summary = value["summary"]
    assert summary["newton_calls"] == 0 and not summary["accepted_steady_state"] and not summary["qualification"]
    assert summary["lateral_electron_flux_max_m2_s"] > 0.0 and summary["lateral_hole_flux_max_m2_s"] > 0.0
    sparse = value["jacobian_csc"]
    matrix = csc_matrix((sparse["data"], sparse["indices"], sparse["indptr"]), shape=tuple(sparse["shape"]))
    np.testing.assert_array_equal(matrix @ np.asarray(value["direction"]), value["analytic_direction"])
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        example.main()
    assert path.read_bytes() == original
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("saved read performed assembly")
    monkeypatch.setattr(example, "compile_device_2d", forbidden)
    monkeypatch.setattr(sys, "argv", ["dc_assembly", "--read", str(path)])
    assert example.main() == 0
