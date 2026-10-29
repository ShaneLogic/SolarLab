"""2D Newton adapter algebra/units/failure checks and saved native readback."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from examples.research_1d.steady import load_device
from examples.research_2d import dc_assembly, steady as example
from solarlab.device.compile_2d import compile_device_2d
from solarlab.equations.dc_2d import DC2DProblem
from solarlab.experiments.transient_results import _encode
from solarlab.experiments.two_dimensional.inputs import JV2DInput
from solarlab.solvers.nonlinear import steady_2d as solver
from solarlab.solvers.nonlinear.newton import NewtonResult, Termination
from solarlab.solvers.nonlinear.problem import GateOutcome, state_digest

Q = 1.602176634e-19


@pytest.fixture(scope="module", autouse=True)
def blas1():
    with threadpool_limits(limits=1):
        yield


@pytest.fixture(scope="module")
def case():
    device = compile_device_2d(load_device(), dc_assembly.spatial_input())
    _, u = dc_assembly.coordinates(device, lateral=True)
    return device, u, DC2DProblem(device, .002, 1.0), solver.Steady2DOptions()


def test_full195_row_scaling_units_and_original_step_budget(case):
    device, u, problem, options = case
    assert device.geometry.shape == (13, 5) and device.node_count == 65
    settings, norm = solver._settings(device, options, lambda x: GateOutcome("probe", True, 0.0, 1.0))
    assert norm is None and settings.scaling.row_weights.shape == (195,)
    np.testing.assert_array_equal(settings.step_control.max_step[:65], .1)
    np.testing.assert_array_equal(settings.step_control.max_step[65:], 2.0)
    assert settings.budget == options.electronic.budget()
    contacts = device.geometry.contact_nodes
    for offset in (65, 130):
        np.testing.assert_array_equal(settings.scaling.row_weights[offset + contacts], 1.0 / 1e-11)
        interior = np.setdiff1d(np.arange(65), contacts)
        expected = 130 * Q / (device.geometry.lateral_length_m * 1e-8)
        np.testing.assert_array_equal(settings.scaling.row_weights[offset + interior], expected)
    data = solver._diagnostics(problem, u, options)
    residual = problem.residual(u)[65:].reshape(2, 65).copy()
    residual[:, contacts] = 0.0
    assert data["carrier_balance_bound_A_m2"] == Q * np.sum(abs(residual)) / device.geometry.lateral_length_m
    assert data["original_rows"] == tuple(range(195))
    assert not all(check.passed for check in solver._checks(data, options))


def test_gauss_scale_uses_area_and_both_axis_face_lengths(case):
    device, _u, _problem, _options = case
    g = device.geometry.contract_geometry
    reference = max(abs(device.column.built_in_potential_V), device.thermal_voltage_V)
    density = device.extrude(np.maximum(np.maximum(device.column.parameters.acceptor_density_m3,
        device.column.parameters.donor_density_m3), 1.0))
    expected = []
    for node in range(device.node_count):
        incident = np.any(g.pairs == node, axis=1)
        charge_scale = Q * density[node] * g.volumes[node]
        capacitance_scale = reference * np.sum(device.face_capacitance_F_m2[incident] * g.face_measures[incident])
        expected.append(1.0 / (charge_scale + capacitance_scale))
    expected = np.asarray(expected)
    expected[device.geometry.contact_nodes] = 1.0 / reference
    np.testing.assert_allclose(solver._poisson_weights(device), expected, rtol=4*np.finfo(float).eps, atol=0.0)
    center = 6 * device.geometry.nx + 2
    assert np.sum(np.any(g.pairs[:device.geometry.x_face_count] == center, axis=1)) == 2
    assert np.sum(np.any(g.pairs[device.geometry.x_face_count:] == center, axis=1)) == 2


def test_terminal_source_and_per_cell_physical_balances_are_conservative(case):
    device, u, problem, options = case
    # A lateral contact perturbation exercises the physical boundary-cell
    # inflow term even though this deliberately fails reservoir constraints.
    u = u.copy()
    u[1] += .001
    u[device.node_count + 1] += .02
    data = solver._diagnostics(problem, u, options)
    contacts = device.geometry.contact_nodes
    interior = np.setdiff1d(np.arange(device.node_count), contacts)
    physical = data["physical_carrier_balance_per_depth_m_s"]
    rows = problem.residual(u)[device.node_count:].reshape(2, device.node_count)
    np.testing.assert_allclose(physical[:, interior], rows[:, interior], rtol=2e-14, atol=1.0)
    scale = max(np.max(abs(physical)), np.max(abs(data["net_pair_source_per_depth_m_s"])), 1.0)
    assert np.max(abs(physical[:, contacts])) / scale < 2e-14
    np.testing.assert_allclose(np.sum(physical, axis=1), data["source_terminal_particle_balance_per_depth_m_s"], rtol=2e-13, atol=1.0)
    np.testing.assert_array_equal(data["terminal_current_per_depth_A_m"] / device.geometry.lateral_length_m,
                                  data["terminal_mean_current_density_A_m2"])
    assert not np.any(data["lateral_boundary_particle_flux_m2_s"])
    assert not np.any(data["lateral_boundary_displacement_C_m2"])


def test_width_normalizes_real2d_carrier_and_current_bounds(case):
    device, _u, _problem, options = case
    wide = compile_device_2d(load_device(), JV2DInput(kind="jv_2d", lateral_length="2 um", Nx=4,
                                                    Ny_per_layer=4, lateral_bc="neumann"))
    column, u = dc_assembly.coordinates(device, lateral=False)
    a = solver._diagnostics(DC2DProblem(device, .002, 1.0), u, options)
    b = solver._diagnostics(DC2DProblem(wide, .002, 1.0), wide.extrude_coordinates(column), options)
    for key in ("carrier_balance_bound_A_m2", "source_terminal_current_bound_A_m2", "cross_section_terminal_spread_A_m2"):
        np.testing.assert_allclose(a[key], b[key], rtol=2e-13, atol=1e-12)
    np.testing.assert_allclose(b["terminal_current_per_depth_A_m"], 2 * a["terminal_current_per_depth_A_m"], rtol=2e-14)


def test_lateral_seed_changes_true_neighbors_and_keeps_contacts(case):
    device, _u, _problem, _options = case
    column, uniform = dc_assembly.coordinates(device, lateral=False)
    state = example.lateral_initial(device, column, voltage_V=.002)
    ids = device.geometry.contact_nodes
    for offset in (0, device.node_count, 2 * device.node_count):
        np.testing.assert_array_equal(state.u[offset + ids], uniform[offset + ids])
    problem = DC2DProblem(device, .002, 1.0)
    fluxes = problem.fluxes(state.u)
    assert np.any(fluxes.electron.flux[:device.geometry.x_face_count] != 0.0)
    assert np.any(fluxes.hole.flux[:device.geometry.x_face_count] != 0.0)
    matrix = problem.jacobian(state.u)
    center = 6 * device.geometry.nx + 2
    assert matrix[center, center - 1] != 0 and matrix[center, center + 1] != 0


def test_failed_newton_retains_original2d_evidence_and_bad_initial_refused(case, monkeypatch):
    device, u, problem, options = case
    def failed(p, initial, settings, **kwargs):
        np.testing.assert_array_equal(p.residual(initial), problem.residual(initial))
        np.testing.assert_array_equal(p.jacobian(initial).toarray(), problem.jacobian(initial).toarray())
        return NewtonResult(initial, False, Termination.BUDGET_EXHAUSTED, 2.0, (), (), 1, 1,
                            state_digest(initial), "synthetic no-solve budget probe")
    monkeypatch.setattr(solver, "solve_newton", failed)
    state = device.state(u, voltage_V=.002)
    result = solver.solve_steady_state_2d(device, voltage_V=.002, generation_scale=1.0,
        options=options, initial_state=state)
    assert not result.accepted and result.termination == "budget_exhausted"
    assert result.newton is not None and result.diagnostics is not None
    assert result.diagnostics["original_rows"] == tuple(range(195))
    assert result.state is not None and result.state.potential_V.shape == (13, 5)
    with pytest.raises(ValueError, match="same device"):
        solver.solve_steady_state_2d(device, voltage_V=.002, generation_scale=1.0,
            initial_state=replace(state, device_sha256="wrong"))


def test_saved_failed_path_reader_is_solver_free(case, monkeypatch, tmp_path):
    device, _u, _problem, _options = case
    record = {"schema": example.SCHEMA, "qualification": False, "completed": False,
        "geometry": {"shape_Ny_Nx": (13, 5)}, "requested_targets": [{"voltage_V": .002, "generation_scale": 0.0}],
        "attempts": [{"target": {"voltage_V": .002, "generation_scale": 0.0}, "accepted": False,
                      "termination": "synthetic_failure", "error": "fixture only"}], "compiled_identity": device.identity}
    path = tmp_path / "Failed.json"
    path.write_text(json.dumps(_encode(record)) + "\n")
    def forbidden(*args, **kwargs):
        raise AssertionError("saved reader compiled or solved")
    monkeypatch.setattr(example, "compile_device_2d", forbidden)
    monkeypatch.setattr(example, "solve_steady_state", forbidden)
    monkeypatch.setattr(example, "solve_steady_state_2d", forbidden)
    value = example.read_result(path, hashlib.sha256(path.read_bytes()).hexdigest())
    assert not value["completed"] and value["attempts"][0]["termination"] == "synthetic_failure"
    with pytest.raises(ValueError, match="SHA-256"):
        example.read_result(path, "wrong")


def test_illumination_completion_uses_cold_seed_without_replaying_dark(case, monkeypatch, tmp_path):
    """Synthetic branch-selection fixture; no seed or2D Newton is executed."""
    device, u, _problem, _options = case
    state = device.state(u, voltage_V=.002)
    fake_prior = {"compiled_identity": device.identity, "attempts": [
        {"target": {"voltage_V": .002, "generation_scale": 0.0}, "accepted": True,
         "result_2d": {"state": {"u": state.u, "potential_V": state.potential_V,
             "electron_density_m3": state.electron_density_m3, "hole_density_m3": state.hole_density_m3}}},
        {"target": {"voltage_V": .002, "generation_scale": 1.0}, "accepted": False,
         "error": "synthetic warm-seed failure"}]}
    monkeypatch.setattr(example, "compile_device_2d", lambda *args: device)
    monkeypatch.setattr(example, "read_result", lambda *args: fake_prior)
    monkeypatch.setattr(example, "_checks", lambda *args: (GateOutcome("synthetic_branch_check", True, 0.0, 1.0),))
    calls = []
    def seed_probe(*args, **kwargs):
        calls.append(kwargs)
        raise RuntimeError("synthetic cold-seed entry observed; no solve")
    monkeypatch.setattr(example, "solve_steady_state", seed_probe)
    monkeypatch.setattr(example, "solve_steady_state_2d", lambda *args, **kwargs: pytest.fail("unexpected2D solve"))
    result = example.run_case(tmp_path / "Branch.json", illumination_only=True,
        prior_result=tmp_path / "Prior.json", prior_sha256="synthetic")
    assert len(calls) == 1 and calls[0]["generation_scale"] == 1.0 and calls[0]["initial_state"] is None
    assert result["requested_targets"] == [{"voltage_V": .002, "generation_scale": 1.0}]
    assert result["auxiliary_1d_seed_calls"] == 1 and result["full_2d_newton_calls"] == 0
    assert result["preserved_prior_path"]["dark_revalidated_without_newton"]


def test_saved_native_195_coordinate_path_has_original_physical_gates(monkeypatch):
    path, sha = os.environ.get("SOLARLAB_STEADY2D_RESULT"), os.environ.get("SOLARLAB_STEADY2D_SHA256")
    if path is None or sha is None:
        pytest.skip("bind the sole saved native path; never repeat its solve")
    def forbidden(*args, **kwargs):
        raise AssertionError("saved evidence review attempted a new physics solve")
    monkeypatch.setattr(example, "compile_device_2d", forbidden)
    monkeypatch.setattr(example, "solve_steady_state", forbidden)
    monkeypatch.setattr(example, "solve_steady_state_2d", forbidden)
    record = example.read_result(Path(path), sha)
    assert record["completed"] and len(record["attempts"]) in (1, 2)
    assert record["full_2d_newton_calls"] == record["auxiliary_1d_seed_calls"] == len(record["attempts"])
    if len(record["attempts"]) == 1:
        prior = record["preserved_prior_path"]
        assert prior["source_sha256"] == "f9048faec6ba7699859e41f1e0276ffd5bd6689787ecea64b22f08941795083b"
        assert prior["dark_revalidated_without_newton"]
        assert prior["accepted_dark_2d_attempt"]["accepted"]
        assert not prior["first_failed_illumination_attempt"]["accepted"]
    for attempt in record["attempts"]:
        result = attempt["result_2d"]
        assert attempt["accepted"] and result["newton"]["accepted"]
        assert all(check["passed"] for check in result["checks"])
        assert result["state"]["u"].shape == (195,)
        assert result["state"]["potential_V"].shape == (13, 5)
        assert np.any(result["initial_diagnostics"]["lateral_current_density_A_m2"] != 0.0)
        assert result["initial_provenance"]["auxiliary_seed_is_2d_completion"] is False
