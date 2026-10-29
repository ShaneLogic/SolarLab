"""Independent channel/area/balance oracles for 2D contacts; no Newton runs."""

from __future__ import annotations

from dataclasses import asdict, replace
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy.sparse import coo_matrix

from examples.research_1d.steady import load_device
from examples.research_1d.transient import json_value
from examples.research_2d import contacts as example
from examples.research_2d.dc_assembly import DIRECTIONAL_RTOL, DIRECTIONAL_STEP, coordinates, spatial_input
from solarlab.contracts import AREA, LENGTH, PARTICLE, SECOND
from solarlab.contracts.units import ELEMENTARY_CHARGE_C as Q
from solarlab.device.compile_2d import compile_device_2d
from solarlab.device.resolved import PreparedDevice
from solarlab.equations.dc_1d import CapabilityError, DC1DProblem
from solarlab.equations.dc_2d import CompiledDevice2D, DC2DProblem
from solarlab.experiments.two_dimensional.inputs import JV2DInput
from solarlab.physics.contacts.exchange import legacy_selective_contact_current
from solarlab.solvers.nonlinear import steady_2d as solver
from solarlab.solvers.nonlinear.problem import GateOutcome
from solarlab.solvers.nonlinear.production import HISTORICAL_MAX_ROW_V1, POISSON_CONTACT_BALANCE_V1
from solarlab.solvers.nonlinear.steady_1d import SteadyOptions

ORDER = (("n", "left"), ("p", "left"), ("n", "right"), ("p", "right"))


@pytest.fixture(scope="module")
def prepared() -> PreparedDevice:
    return load_device()


def compiled(prepared: PreparedDevice, velocities: tuple[float | None, ...]) -> CompiledDevice2D:
    assert len(velocities) == 4
    declared = example.contact_device((velocities[0], velocities[1], velocities[2], velocities[3]), prepared=prepared)
    return compile_device_2d(declared, spatial_input())


@pytest.fixture(scope="module")
def mixed(prepared: PreparedDevice) -> CompiledDevice2D:
    return compiled(prepared, example.VELOCITIES)


def nodes_for(device: CompiledDevice2D, side: str) -> np.ndarray:
    return device.geometry.contact_nodes.reshape(2, device.geometry.nx)[0 if side == "left" else 1]


@pytest.mark.parametrize("channel_index", range(4))
@pytest.mark.parametrize("velocity", [0.0, 731.5])
def test_four_channels_are_independent_and_bind_identity(prepared, channel_index, velocity):
    velocities = [None] * 4
    velocities[channel_index] = velocity
    fixed = compile_device_2d(prepared, spatial_input())
    active = compiled(prepared, tuple(velocities))
    assert tuple((c.carrier, c.side) for c in active.column.channels) == ORDER
    assert tuple(c.velocity_m_s for c in active.column.channels) == tuple(velocities)
    assert active.identity != fixed.identity
    mask = active.fixed_carrier_mask
    expected = np.zeros(mask.shape, dtype=bool)
    for index, (carrier, side) in enumerate(ORDER):
        if index != channel_index:
            expected[0 if carrier == "n" else 1, nodes_for(active, side)] = True
    np.testing.assert_array_equal(mask, expected)
    assert not mask.flags.writeable
    for name in ("reference_electron_density_m3", "reference_hole_density_m3", "reference_potential_V"):
        np.testing.assert_array_equal(getattr(active.column, name), getattr(fixed.column, name))
    assert active.column.contact_potentials(.002) == fixed.column.contact_potentials(.002)
    np.testing.assert_array_equal(active.generation_source_m3_s, fixed.generation_source_m3_s)


@pytest.mark.parametrize("channel_index", range(4))
def test_contact_flux_sign_units_and_exact_density_jacobian(prepared, channel_index):
    velocities = [None] * 4
    velocities[channel_index] = 65536.0
    active = compiled(prepared, tuple(velocities))
    velocities[channel_index] = 0.0
    blocking = compiled(prepared, tuple(velocities))
    p, zero = DC2DProblem(active, .002, 1.0), DC2DProblem(blocking, .002, 1.0)
    u = example.algebra_coordinates(active)
    count = active.node_count
    channel = active.column.channels[channel_index]
    carrier, side = channel.carrier, channel.side
    nodes = nodes_for(active, side)
    offset, sign, charge = (count, 1.0, -Q) if carrier == "n" else (2 * count, -1.0, Q)
    reference = active.reference_potentials(.002)[0 if carrier == "n" else 1]
    exponent = np.linspace(-.2, .3, nodes.size)
    u[offset + nodes] = sign * exponent - (u[nodes] - reference[nodes]) / active.thermal_voltage_V
    density = active.fields(u, voltage_V=.002)[1 if carrier == "n" else 2][nodes]
    exchange, slope = p._contact_exchange(u, channel, nodes)
    assert slope == channel.outward_normal * 65536.0
    legacy = legacy_selective_contact_current(density, channel.reservoir_density_m3, 65536.0, carrier=carrier, side=side)
    np.testing.assert_allclose(charge * exchange, legacy, rtol=32*np.finfo(float).eps, atol=0.0)
    # Independently subtract blocking from finite exchange: only contact
    # inflow and its two local density-chain entries may remain.
    delta = p.residual(u) - zero.residual(u)
    expected = np.zeros_like(delta)
    expected[offset + nodes] = -65536.0 * channel.reservoir_density_m3 * np.expm1(exponent) * active.geometry.x_widths_m
    allowance = 16*np.finfo(float).eps * (abs(p.residual(u)) + abs(zero.residual(u)) + abs(expected))
    assert np.all(abs(delta - expected) <= allowance)
    a, b = p.jacobian(u).toarray(), zero.jacobian(u).toarray()
    target = np.zeros_like(a)
    for node, width, c in zip(nodes, active.geometry.x_widths_m, density):
        target[offset + node, node] = -65536.0 * width * sign * c / active.thermal_voltage_V
        target[offset + node, offset + node] = -65536.0 * width * sign * c
    allowance_j = 16*np.finfo(float).eps * (abs(a) + abs(b) + abs(target))
    assert np.all(abs(a - b - target) <= allowance_j)
    assert active.geometry.contract_geometry.volume_unit == AREA
    assert active.geometry.contract_geometry.face_unit == LENGTH
    assert next(e for e in active.layout.equations if e.id == "electron").unit == PARTICLE / (LENGTH * SECOND)


@pytest.mark.parametrize("channel_index", range(4))
def test_zero_velocity_is_blocking_and_near_reservoir_excess_is_not_rounded_away(prepared, channel_index):
    velocities = [None] * 4
    velocities[channel_index] = 1024.0
    d = compiled(prepared, tuple(velocities))
    problem = DC2DProblem(d, 0.0, 0.0)
    u = np.r_[d.extrude(d.column.reference_potential_V), np.zeros(2 * d.node_count)]
    channel = d.column.channels[channel_index]
    nodes = nodes_for(d, channel.side)
    sign, offset = (1.0, d.node_count) if channel.carrier == "n" else (-1.0, 2 * d.node_count)
    tiny = 2.0**-55
    u[offset + nodes] = sign * tiny
    flux, slope = problem._contact_exchange(u, channel, nodes)
    assert np.exp(tiny) == 1.0 and np.all(flux != 0.0)
    np.testing.assert_array_equal(flux, slope * channel.reservoir_density_m3 * np.expm1(tiny))
    one = DC1DProblem(d.column, 0.0, 0.0)
    column_u = u.reshape(3, d.geometry.ny, d.geometry.nx)[:, :, 0].ravel()
    expected, expected_slope = one._contact_exchange(column_u, channel, 0 if channel.side == "left" else d.geometry.ny - 1)
    np.testing.assert_array_equal(flux, expected)
    assert slope == expected_slope
    blocked, blocked_slope = problem._contact_exchange(u, replace(channel, velocity_m_s=0.0), nodes)
    assert blocked_slope == 0.0 and not np.any(blocked)


@pytest.mark.parametrize("channel_index", range(4))
def test_finite_exchange_has_the_compatible_fixed_reservoir_limit(prepared, channel_index):
    d = compiled(prepared, (None, None, None, None))
    problem = DC2DProblem(d, 0.0, 0.0)
    channel = d.column.channels[channel_index]
    nodes = nodes_for(d, channel.side)
    sign, offset = (1.0, d.node_count) if channel.carrier == "n" else (-1.0, 2 * d.node_count)
    target = channel.outward_normal * channel.reservoir_density_m3 * .01
    constraints = []
    for velocity in (1.0, 1024.0, 1048576.0):
        u = np.r_[d.extrude(d.column.reference_potential_V), np.zeros(2 * d.node_count)]
        u[offset + nodes] = sign * math.log1p(.01 / velocity)
        flux, _ = problem._contact_exchange(u, replace(channel, velocity_m_s=velocity), nodes)
        np.testing.assert_allclose(flux, target, rtol=4*np.finfo(float).eps, atol=0.0)
        constraints.append(float(np.max(abs(problem.residual(u)[offset + nodes]))))
    assert constraints[0] > constraints[1] > constraints[2] > 0.0
    assert constraints[2] < constraints[0] / 1e6


@pytest.mark.parametrize("potential_only", [False, True])
def test_full_contact_jacobian_with_lateral_and_endpoint_perturbations(mixed, potential_only):
    p = DC2DProblem(mixed, .002, 0.0)
    u = example.algebra_coordinates(mixed)
    matrix = p.jacobian(u)
    np.testing.assert_array_equal(matrix.toarray(), DC2DProblem(mixed, .002, 1.0).jacobian(u).toarray())
    rng = np.random.default_rng(772)
    v = np.r_[.02 * rng.normal(size=mixed.node_count), rng.normal(size=2 * mixed.node_count)]
    if potential_only:
        v[mixed.node_count:] = 0.0
    h = DIRECTIONAL_STEP
    numerical = (p.residual(u - 2*h*v) - 8*p.residual(u - h*v) + 8*p.residual(u + h*v) - p.residual(u + 2*h*v)) / (12*h)
    scale = np.maximum(abs(matrix) @ abs(v), 1e-20)
    assert np.max(abs(numerical - matrix @ v) / scale) < DIRECTIONAL_RTOL
    corner = mixed.node_count  # n at the blocking left/lateral corner.
    assert matrix[corner, mixed.node_count + 1] != 0.0
    assert matrix[corner, mixed.node_count + mixed.geometry.nx] != 0.0


@pytest.mark.parametrize("velocities", [example.VELOCITIES, (10.0, 20.0, 30.0, 40.0), (None, 0.0, None, 0.0)])
def test_uniform_reduction_with_explicit_half_endpoint_source_correction(prepared, velocities):
    d = compiled(prepared, velocities)
    g, ny, N = d.geometry, d.geometry.ny, d.node_count
    u1, u2 = coordinates(d, lateral=False)
    one, two = DC1DProblem(d.column, .002, 1.0), DC2DProblem(d, .002, 1.0)
    f1, j1 = one.residual(u1).reshape(3, ny), one.jacobian(u1).toarray()
    _, n, p = d.column.fields(u1, voltage_V=.002)
    rate = one.recombination_rate(n, p)
    w1, w2 = d.column.geometry.cell_measures_m, g.y_geometry.cell_measures_m
    source_delta = (d.generation_y.source_m3_s - rate.rate) * w2 - (d.column.generation.source_m3_s - rate.rate) * w1
    assert rate.d_rate_dn is not None and rate.d_rate_dp is not None
    dn, dp = np.asarray(rate.d_rate_dn) * n, -np.asarray(rate.d_rate_dp) * p
    current_correction = np.zeros(2)
    for carrier_index, carrier in enumerate(("n", "p")):
        for side_index, (side, node) in enumerate((("left", 0), ("right", ny - 1))):
            if d.column.channel(carrier, side).kind == "fixed_reservoir":
                charge = -Q if carrier == "n" else Q
                current_correction[side_index] += charge * (-1.0 if side == "left" else 1.0) * source_delta[node]
                continue
            row = (carrier_index + 1) * ny + node
            f1[carrier_index + 1, node] += source_delta[node]
            j1[row, node] += (w1[node] - w2[node]) * (dn[node] + dp[node]) / d.thermal_voltage_V
            j1[row, ny + node] += (w1[node] - w2[node]) * dn[node]
            j1[row, 2 * ny + node] += (w1[node] - w2[node]) * dp[node]
    weights = np.tile(np.tile(g.x_widths_m, ny), (3, 1))
    weights[0, g.contact_nodes] = 1.0
    weights[1:][d.fixed_carrier_mask] = 1.0
    actual = two.residual(u2) / weights.ravel()
    expected = d.extrude_coordinates(f1.ravel())
    block_scale = np.repeat(np.max(abs(expected.reshape(3, N)), axis=1), N)
    assert np.max(abs(actual - expected) / np.maximum(block_scale, 1e-30)) < 1e-13
    rows, columns = np.arange(3 * N), np.repeat(np.arange(3 * ny), g.nx)
    lift = coo_matrix((np.ones(rows.size), (rows, columns)), shape=(3 * N, 3 * ny)).tocsc()
    actual_j = (two.jacobian(u2) @ lift).toarray() / weights.ravel()[:, None]
    expected_j = lift @ j1
    row_scale = np.maximum(np.max(abs(expected_j), axis=1), 1e-30)
    assert np.max(np.max(abs(actual_j - expected_j), axis=1) / row_scale) < 1e-13
    np.testing.assert_allclose(two.terminal_current_density_A_m2(u2), one.contact_current_density(u1) + current_correction,
                               rtol=2e-13, atol=0.0)
    assert not np.any(two.fluxes(u2).electron.flux[:g.x_face_count])
    assert not np.any(two.fluxes(u2).hole.flux[:g.x_face_count])


def test_physical_terminal_and_full_row_balances_keep_failed_finite_contacts(mixed):
    d, g = mixed, mixed.geometry
    problem = DC2DProblem(d, .002, 1.0)
    u = example.algebra_coordinates(d)
    _, n, p = d.fields(u, voltage_V=.002)
    source = (d.generation_source_m3_s - problem.recombination_rate(n, p, derivatives=False).rate) * g.contract_geometry.volumes
    contacts = np.stack(problem.contact_particle_fluxes(u))
    expected = np.tile(source, (2, 1))
    fluxes = problem.fluxes(u)
    for field, flux, terminal in zip(expected, (fluxes.electron.flux, fluxes.hole.flux), contacts):
        # Preserve the declared negative-then-positive accumulation order;
        # interleaving those additions tests a different rounding sequence.
        for (left, right), measure, value in zip(g.contract_geometry.pairs, g.contract_geometry.face_measures, flux):
            field[left] -= measure * value
        for (left, right), measure, value in zip(g.contract_geometry.pairs, g.contract_geometry.face_measures, flux):
            field[right] += measure * value
        field[:g.nx] += g.x_widths_m * terminal[0]
        field[-g.nx:] -= g.x_widths_m * terminal[1]
    residual = problem.residual(u)[d.node_count:].reshape(2, d.node_count)
    np.testing.assert_allclose(expected[~d.fixed_carrier_mask], residual[~d.fixed_carrier_mask], rtol=2e-14, atol=1.0)
    data = solver._diagnostics(problem, u, solver.Steady2DOptions())
    np.testing.assert_array_equal(data["physical_carrier_balance_per_depth_m_s"], expected)
    physical_sum = np.sum(expected, axis=1)
    np.testing.assert_allclose(physical_sum, data["source_terminal_particle_balance_per_depth_m_s"], rtol=2e-13, atol=1.0)
    rows = residual.copy()
    rows[d.fixed_carrier_mask] = 0.0
    assert data["carrier_balance_bound_A_m2"] == Q * np.sum(abs(rows)) / g.lateral_length_m
    contact_rows = np.zeros_like(rows, dtype=bool)
    contact_rows[:, g.contact_nodes] = True
    assert np.any(expected[contact_rows & ~d.fixed_carrier_mask] != 0.0)
    assert not all(check.passed for check in solver._checks(data, solver.Steady2DOptions()))
    expected_current = np.sum((-Q * contacts[0] + Q * contacts[1]) * g.x_widths_m, axis=1)
    np.testing.assert_array_equal(problem.terminal_current_A_m(u), expected_current)
    np.testing.assert_array_equal(problem.terminal_current_density_A_m2(u), expected_current / g.lateral_length_m)
    assert not np.any(contacts[0, 0])  # A blocking channel is not inferred from its failed row.


@pytest.mark.parametrize("policy", [HISTORICAL_MAX_ROW_V1, POISSON_CONTACT_BALANCE_V1])
@pytest.mark.parametrize("velocities", [example.VELOCITIES, (10.0, 20.0, 30.0, 40.0)])
def test_only_real_fixed_rows_get_log_scaling_and_gate_exclusion(prepared, policy, velocities):
    d = compiled(prepared, velocities)
    options = solver.Steady2DOptions(SteadyOptions(convergence_policy=policy))
    settings, norm = solver._settings(d, options, lambda u: GateOutcome("unused", True, 0.0, 1.0))
    mask = d.fixed_carrier_mask
    expected_weight = 2 * d.node_count * Q / (d.geometry.lateral_length_m * options.electronic.continuity_tolerance_A_m2)
    weights = settings.scaling.row_weights[d.node_count:].reshape(mask.shape)
    np.testing.assert_array_equal(weights[~mask], expected_weight)
    np.testing.assert_array_equal(weights[mask], 1.0 / options.electronic.contact_log_tolerance)
    assert settings.budget == SteadyOptions().budget()
    u = example.algebra_coordinates(d)
    data = solver._diagnostics(DC2DProblem(d, .002, 1.0), u, options)
    if not np.any(mask):
        assert data["contact_log_max"] == 0.0
    assert data["contact_log_max"] == np.max(abs(DC2DProblem(d, .002, 1.0).residual(u)[d.node_count:].reshape(mask.shape)[mask]), initial=0.0)
    if norm is not None:
        scaled = np.arange(3 * d.node_count, dtype=float) / 10
        fixed = np.r_[np.ones(d.node_count, dtype=bool), mask.ravel()]
        assert norm(scaled) == max(np.max(abs(scaled[fixed])), np.sum(abs(scaled[~fixed])) / (2*d.node_count))


def test_cold_boundary_imposition_preserves_finite_unknowns_without_running_newton(mixed, monkeypatch):
    captured = []
    def no_solve(problem, initial, settings, **kwargs):
        captured.append(initial.copy())
        raise RuntimeError("contact boundary probe; no Newton executed")
    monkeypatch.setattr(solver, "solve_newton", no_solve)
    result = solver.solve_steady_state_2d(mixed, voltage_V=.002, generation_scale=1.0)
    assert len(captured) == 1 and not result.accepted and result.termination == "evaluation_error"
    u, N = captured[0], mixed.node_count
    np.testing.assert_array_equal(u[N:].reshape(2, N)[~mixed.fixed_carrier_mask], 0.0)
    assert np.all(result.initial_diagnostics["original_rows"] == np.arange(3*N))
    assert "finite/blocking" in result.solver_controls["row_units"]["contact_carrier"]
    assert result.solver_controls["numerical"] == asdict(solver.Steady2DOptions())
    references = mixed.reference_potentials(.002)
    for offset, reference, mask in zip((N, 2*N), references, mixed.fixed_carrier_mask):
        np.testing.assert_array_equal(u[offset:offset+N][mask], (reference[mask] - u[:N][mask]) / mixed.thermal_voltage_V)


@pytest.mark.parametrize("extra", [{"lateral_bc": "periodic"}, {"ion_dynamics": "single_mobile"},
                                  {"interface_srh": "two_sided_cross_node"}])
def test_unbound_contact_combinations_remain_refused(prepared, extra):
    spatial = spatial_input().editing_data()
    spatial.update(extra)
    with pytest.raises(CapabilityError):
        compile_device_2d(example.contact_device(prepared=prepared), JV2DInput.model_validate(spatial))


def test_fully_blocking_inventory_remains_explicitly_unbound(prepared):
    with pytest.raises(CapabilityError, match="fully_blocking_contacts"):
        compiled(prepared, (0.0, 0.0, 0.0, 0.0))


def fixed_snapshot(d: CompiledDevice2D, voltage: float, generation: float, lateral: bool, policy: str) -> dict[str, Any]:
    """Shared capture format for the exact committed baseline, with no solve."""
    _, u = coordinates(d, lateral=lateral)
    problem = DC2DProblem(d, voltage, generation)
    matrix = problem.jacobian(u)
    options = solver.Steady2DOptions(SteadyOptions(convergence_policy=policy))
    settings, _ = solver._settings(d, options, lambda values: GateOutcome("unused", True, 0.0, 1.0))
    return {"identity": d.identity, "column_identity": d.column.identity, "geometry_identity": d.geometry.sha256,
        "generation_identity": d.generation_y.binding_sha256, "u": u, "fields": np.stack(d.fields(u, voltage_V=voltage)),
        "residual": problem.residual(u), "jacobian_data": matrix.data, "jacobian_indices": matrix.indices,
        "jacobian_indptr": matrix.indptr, "flux_n": problem.fluxes(u).electron.flux,
        "flux_p": problem.fluxes(u).hole.flux, "displacement": problem.displacement(u),
        "terminal_particles": np.stack(problem.contact_particle_fluxes(u)), "terminal_A_m": problem.terminal_current_A_m(u),
        "current_A_m2": problem.current_density(u), "sections_A_m": problem.cross_section_current_A_m(u),
        "weights": settings.scaling.row_weights, "diagnostics": solver._diagnostics(problem, u, options)}


def test_exact_default_against_pinned_committed_source(monkeypatch):
    name = os.environ.get("SOLARLAB_CONTACTS_BASELINE")
    if name is None:
        pytest.skip("explicit committed-source algebra artifact required; no native replay")
    path = Path(name)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == os.environ["SOLARLAB_CONTACTS_BASELINE_SHA256"]
    baseline = json.loads(path.read_bytes())
    assert baseline["commit"] == "b819a2696971e568d360a80beb78ede565f671d6"
    prepared = load_device(Path(baseline["input_path"]))
    d = compile_device_2d(prepared, spatial_input())
    def forbidden(*args, **kwargs):
        raise AssertionError("default parity attempted a Newton solve")
    monkeypatch.setattr(solver, "solve_newton", forbidden)
    for record in baseline["cases"]:
        values = fixed_snapshot(d, **record["arguments"])
        # Hex strings preserve signed zeros and every finite binary64 bit.
        def exact(value):
            if isinstance(value, np.ndarray):
                return {"dtype": value.dtype.str, "shape": list(value.shape), "bytes_hex": value.tobytes().hex()}
            if isinstance(value, np.floating):
                return float(value).hex()
            if isinstance(value, float):
                return value.hex()
            if isinstance(value, dict):
                return {key: exact(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [exact(item) for item in value]
            return value
        assert exact(values) == record["exact"]


def test_example_and_saved_reader_do_not_solve(monkeypatch, tmp_path):
    monkeypatch.setattr(solver, "solve_newton", lambda *args, **kwargs: pytest.fail("unexpected Newton"))
    result = example.assemble()
    assert result["summary"]["newton_calls"] == 0 and not result["summary"]["qualification"]
    assert result["summary"]["directional_error"] < DIRECTIONAL_RTOL
    assert result["witness_proposal"]["newton_calls"] == 1 and not result["witness_proposal"]["scheduled"]
    path = tmp_path/"Contacts.json"
    path.write_text(json.dumps(json_value(result), allow_nan=False))
    pin = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(example, "compile_device_2d", lambda *args, **kwargs: pytest.fail("reader compiled"))
    assert example.read_result(path, pin)["summary"] == json_value(result["summary"])
    with pytest.raises(ValueError, match="SHA-256"):
        example.read_result(path, "wrong")


def saved_contact_archive() -> Path:
    value = os.environ.get("SOLARLAB_CONTACTS_ARCHIVE")
    if value is None:
        pytest.skip("explicit saved contact evidence required; no native replay")
    return Path(value)


def test_saved_density_preparation_column_chain_and_unchanged_coordinates():
    path = saved_contact_archive()/"DensityPreparation.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == "2052fabd3c3b2d8059ce4d27731cb32a5e609cf271da3ebe4390e18cbb5d4a76"
    data = json.loads(path.read_bytes())
    matrix = np.asarray(data["blocking_physical_jacobian"])
    increment = np.asarray(data["blocking_density_increment_m3"])
    before = np.asarray(data["blocking_residual_before"])
    bound = 16*np.finfo(float).eps * (abs(matrix) @ abs(increment) + abs(before))
    assert np.all(abs(matrix @ increment + before) <= bound)
    assert np.all(np.asarray(data["blocking_proposed_density_m3"]) > 0.0)
    active = np.asarray(data["blocking_rows"])
    kinetic = np.asarray(data["finite_matched_coordinates"])
    final = np.asarray(data["state"]["u"])
    other = np.setdiff1d(np.arange(195), active)
    np.testing.assert_array_equal(final[other], kinetic[other])
    np.testing.assert_array_equal(final[active], kinetic[active] + np.log1p(increment / np.asarray(data["blocking_density_m3"])))
    assert data["contact_density_linear_solves"] == data["full_direction_linear_solves"] == 1
    assert data["nonlinear_solves"] == 0 and data["accepted_for_contact_model"] is False
    assert data["full_step_can_progress"] and data["full_step_fraction"] > .1


@pytest.mark.parametrize("name,pin,accepted", [
    ("Witness01", "55c7035923bf2d50c5e9071c839adbcdb45f03156973bb0e4be390fe12b611f9", False),
    ("Witness02", "71f934d32b356b545d5f55f5671a7016848b06147efc9d92534f13008a468bf5", False),
    ("Witness03", "ee2519253d4677798dc2974694a9cb3a9a64379502937d12576dd17fe0244d51", True),
])
def test_saved_contact_success_and_failures_never_resolve(name, pin, accepted, monkeypatch, tmp_path):
    path = saved_contact_archive()/name/"Result.json"
    def forbidden(*args, **kwargs):
        raise AssertionError("saved contact reader attempted calculation")
    monkeypatch.setattr(example, "prepare_seed", forbidden)
    monkeypatch.setattr(example, "compile_device_2d", forbidden)
    monkeypatch.setattr(example, "solve_steady_state_2d", forbidden)
    value = example.read_result(path, pin)
    assert value["accepted"] is accepted and value["qualification"] is False
    assert value["result"]["termination"] == ("converged" if accepted else "stagnated")
    assert value["newton_calls"] == 1 and value["auxiliary_calls"] == 0
    if accepted:
        assert all(c["passed"] and c["value"] <= c["limit"] for c in value["result"]["checks"])
    else:
        bad = deepcopy(value)
        bad["accepted"] = True
        target = tmp_path/"Overclaimed.json"
        target.write_text(json.dumps(bad))
        with pytest.raises(ValueError, match="overclaimed"):
            example.read_result(target, hashlib.sha256(target.read_bytes()).hexdigest())


@pytest.mark.parametrize("stage", ["finite", "blocking"])
def test_nonpositive_preparation_is_rejected_without_fallback(stage, monkeypatch):
    archive = saved_contact_archive()
    path = archive.parent/"DCSteadyV1/Consumers/IlluminatedCompletion/Steady.json"
    pin = "af7dcf6b7f6adad7254666971c87c33357f7b91600ac5a2ca27e62b638784c05"
    old = example.read_fixed_result(path, pin)
    if stage == "finite":
        changed = deepcopy(old)
        flux = np.asarray(changed["attempts"][0]["result_2d"]["diagnostics"]["terminal_particle_flux_density_m2_s"]).copy()
        flux[0, 1] = -2 * 1e5 * 1e24
        changed["attempts"][0]["result_2d"]["diagnostics"]["terminal_particle_flux_density_m2_s"] = flux
        monkeypatch.setattr(example, "read_fixed_result", lambda *args: changed)
        monkeypatch.setattr(np.linalg, "solve", lambda *args: pytest.fail("nonpositive finite seed reached block solve"))
    else:
        density = np.asarray(json.loads((archive/"DensityPreparation.json").read_bytes())["blocking_density_m3"])
        monkeypatch.setattr(np.linalg, "solve", lambda *args: -2*density)
    monkeypatch.setattr(example, "_Factor", lambda *args: pytest.fail("nonpositive seed reached full direction"))
    monkeypatch.setattr(example, "solve_steady_state_2d", lambda *args, **kwargs: pytest.fail("negative seed reached Newton"))
    with pytest.raises(ValueError, match="positive|positive/finite|finite/positive"):
        example.prepare_seed(path, pin)


def test_runnable_solve_uses_one_explicit_prepared_state_and_refuses_overwrite(mixed, monkeypatch, tmp_path):
    data = json.loads((saved_contact_archive()/"DensityPreparation.json").read_bytes())
    state = mixed.state(np.asarray(data["state"]["u"]), voltage_V=.002)
    monkeypatch.setattr(example, "prepare_seed", lambda *args: (mixed, state, data))
    calls = []
    def stop(device, **kwargs):
        calls.append(kwargs)
        assert device is mixed and kwargs["initial_state"] is state
        assert kwargs["options"] == solver.Steady2DOptions(wall_s=25.0)
        assert kwargs["initial_provenance"]["accepted_for_contact_model"] is False
        raise RuntimeError("one-call wiring fixture; no Newton executed")
    monkeypatch.setattr(example, "solve_steady_state_2d", stop)
    output = tmp_path/"Result.json"
    with pytest.raises(RuntimeError, match="wiring fixture"):
        example.solve_from_seed(output, tmp_path/"Seed.json", "fixture")
    assert len(calls) == 1
    output.write_text("preserved")
    with pytest.raises(FileExistsError):
        example.solve_from_seed(output, tmp_path/"Seed.json", "fixture")
    assert len(calls) == 1 and output.read_text() == "preserved"
