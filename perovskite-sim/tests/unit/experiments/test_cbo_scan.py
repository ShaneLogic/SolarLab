"""Continuation and validity contracts for the interface CBO scan."""
from __future__ import annotations

import dataclasses
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

import perovskite_sim.experiments.cbo_scan as cbo
from perovskite_sim.experiments.jv_sweep import JVMetrics
from perovskite_sim.experiments.quasi_fermi_steady_state import (
    QuasiFermiSteadyStateError,
)


class _FakeStack(float):
    V_bi = 1.3


def _install_fake_scan(monkeypatch, short_circuit):
    monkeypatch.setattr(cbo, "build_electrical_grid", lambda stack, N: np.arange(3.0))
    monkeypatch.setattr(
        cbo,
        "_stack_at_cbo",
        lambda baseline, delta, *, boundary_policy: _FakeStack(delta),
    )
    monkeypatch.setattr(cbo, "solve_quasi_fermi_steady_state", short_circuit)

    def fake_jv(grid, stack, voltages, **kwargs):
        delta = float(stack)
        drop = 0.15 if delta >= 0.3 else 0.0
        voltage_array = np.asarray(voltages, dtype=float)
        return SimpleNamespace(
            certified=True,
            metrics_certified=True,
            voltages_V=voltage_array,
            currents_A_m2=np.linspace(100.0, -1.0, len(voltage_array)),
            points=tuple(
                SimpleNamespace(certified=True) for _ in voltage_array
            ),
            metrics=JVMetrics(
                V_oc=1.0,
                J_sc=100.0 * (1.0 - drop),
                FF=0.8 * (1.0 - drop),
                PCE=0.2 * (1.0 - drop),
            ),
        )

    monkeypatch.setattr(cbo, "solve_quasi_fermi_jv_sweep", fake_jv)


def _state(delta: float, current: float):
    return SimpleNamespace(
        delta=delta,
        current_A_m2=current,
        certified=True,
        face_current_spread_A_m2=1.0e-8,
        interface_local_residual=1.0e-10,
    )


def test_scan_inserts_bridge_points_and_bounds_metric_onsets(monkeypatch):
    def fake_short(grid, stack, *, initial_state=None, **kwargs):
        delta = float(stack)
        if initial_state is not None and abs(delta - initial_state.delta) > 0.11:
            raise QuasiFermiSteadyStateError("outside continuation basin")
        return _state(delta, 100.0 if delta < 0.3 else 85.0)

    _install_fake_scan(monkeypatch, fake_short)
    result = cbo.solve_interface_cbo_scan(
        object(),
        np.array([0.0, 0.4]),
        voltages_V=np.array([0.0, 1.0]),
        minimum_delta_step_eV=1.0e-3,
    )

    assert result.complete
    assert result.certified
    assert result.N_grid == 30
    assert result.calculate_jv_metrics
    assert not result.sync_vbi
    assert result.boundary_policy == cbo.FIXED_CONTACTS
    assert result.interface_topology == cbo.DEDUPLICATED_QSS
    assert result.qf_coordinate_system == "edge_drop"
    assert result.grid_node_count == 3
    assert result.grid_interval_count == 2
    assert result.reference_grid_warm_start_failures == 0
    assert result.reference_grid_cold_recoveries == 0
    assert result.reference_grid_predictor_recoveries == 0
    assert np.array_equal(result.voltages_V, np.array([0.0, 1.0]))
    assert len(result.short_circuit_trace) > 2
    assert any(not sample.requested for sample in result.short_circuit_trace)
    intervals = {interval.metric: interval for interval in result.critical_intervals}
    assert intervals["Jsc"].resolved
    assert intervals["Jsc"].lower_delta_ec_eV < 0.3
    assert intervals["Jsc"].upper_delta_ec_eV >= 0.3
    assert intervals["FF"].resolved
    assert intervals["PCE"].resolved


def test_scan_returns_a_bracket_at_the_bulk_statistics_limit(monkeypatch):
    def fake_short(grid, stack, *, initial_state, **kwargs):
        delta = float(stack)
        if delta > 0.413:
            raise QuasiFermiSteadyStateError(
                "carrier log-density is outside the audited exponential range"
            )
        return _state(delta, 100.0)

    _install_fake_scan(monkeypatch, fake_short)
    result = cbo.solve_interface_cbo_scan(
        object(),
        np.array([0.0, 0.5]),
        voltages_V=np.array([0.0, 1.0]),
        minimum_delta_step_eV=5.0e-4,
    )

    assert not result.complete
    assert result.certified
    assert len(result.points) == 1
    assert len(result.terminations) == 1
    termination = result.terminations[0]
    assert termination.direction == "positive"
    assert termination.last_certified_delta_ec_eV <= 0.413
    assert termination.first_failed_delta_ec_eV > 0.413
    assert (
        termination.first_failed_delta_ec_eV
        - termination.last_certified_delta_ec_eV
        <= 5.0e-4
    )


def test_short_circuit_only_mode_leaves_ff_and_pce_unresolved(monkeypatch):
    def fake_short(grid, stack, *, initial_state, **kwargs):
        delta = float(stack)
        return _state(delta, 100.0 - 20.0 * delta)

    _install_fake_scan(monkeypatch, fake_short)
    result = cbo.solve_interface_cbo_scan(
        object(),
        np.array([0.0, 0.4]),
        calculate_jv_metrics=False,
    )

    assert result.complete
    assert result.certified
    assert not result.calculate_jv_metrics
    assert all(point.jv is None for point in result.points)
    intervals = {interval.metric: interval for interval in result.critical_intervals}
    assert intervals["Jsc"].resolved
    assert not intervals["FF"].resolved
    assert not intervals["PCE"].resolved


def test_scan_recertifies_a_cross_grid_reference_seed(monkeypatch):
    calls = []

    def fake_short(grid, stack, *, initial_state, **kwargs):
        calls.append((grid.copy(), initial_state, kwargs.get("initial_state_grid")))
        return _state(float(stack), 100.0)

    _install_fake_scan(monkeypatch, fake_short)
    seed = _state(0.0, 100.0)
    seed_grid = np.array([0.0, 0.5, 1.0])
    monkeypatch.setattr(
        cbo,
        "build_electrical_grid",
        lambda stack, N: np.linspace(0.0, 1.0, 5),
    )

    result = cbo.solve_interface_cbo_scan(
        object(),
        np.array([0.0]),
        calculate_jv_metrics=False,
        reference_initial_state=seed,
        reference_initial_state_grid=seed_grid,
    )

    assert result.certified
    assert result.reference_grid_warm_starts == 1
    assert calls[0][1] is seed
    assert calls[0][2] is seed_grid


def test_failed_cross_grid_seed_recovers_with_certified_target_cold_start(
    monkeypatch,
):
    calls = []
    seed = _state(0.0, 100.0)

    def fake_short(grid, stack, *, initial_state=None, **kwargs):
        calls.append(initial_state)
        if initial_state is seed:
            raise QuasiFermiSteadyStateError("cross-grid predictor missed basin")
        return _state(float(stack), 100.0)

    _install_fake_scan(monkeypatch, fake_short)
    result = cbo.solve_interface_cbo_scan(
        object(),
        np.array([0.0]),
        calculate_jv_metrics=False,
        reference_initial_state=seed,
        reference_initial_state_grid=np.array([0.0, 0.5, 1.0]),
    )

    assert result.certified
    assert result.reference_grid_warm_starts == 0
    assert result.reference_grid_warm_start_failures == 1
    assert result.reference_grid_cold_recoveries == 1
    assert result.reference_grid_predictor_recoveries == 0
    assert calls[:2] == [seed, None]


def test_reference_cold_failure_recovers_through_coarse_certified_basin(
    monkeypatch,
):
    calls = []
    baseline = SimpleNamespace(
        V_bi=1.3,
        grid_interval_weights=(1.0, 2.0, 1.0),
        grid_alphas=(3.0, 3.0, 3.0),
    )

    monkeypatch.setattr(
        cbo,
        "build_electrical_grid",
        lambda stack, N: np.linspace(0.0, 1.0, N + 1),
    )
    monkeypatch.setattr(cbo, "electrical_layers", lambda stack: (1, 2, 3))
    monkeypatch.setattr(
        cbo,
        "_stack_at_cbo",
        lambda baseline, delta, *, boundary_policy: _FakeStack(delta),
    )

    def fake_short(grid, stack, *, initial_state=None, **kwargs):
        intervals = len(grid) - 1
        calls.append((intervals, initial_state))
        if intervals == 50 and initial_state is None:
            raise QuasiFermiSteadyStateError("target-grid cold basin failed")
        state = _state(float(stack), 100.0)
        state.source_intervals = intervals
        if intervals == 50:
            assert initial_state.source_intervals == 40
        return state

    monkeypatch.setattr(cbo, "solve_quasi_fermi_steady_state", fake_short)
    result = cbo.solve_interface_cbo_scan(
        baseline,
        np.array([0.0]),
        N_grid=50,
        calculate_jv_metrics=False,
    )

    assert result.certified
    assert result.reference_grid_warm_starts == 1
    assert result.reference_grid_predictor_recoveries == 1
    assert [item[0] for item in calls] == [50, 40, 50]


def test_reference_grid_requires_a_reference_state():
    with pytest.raises(ValueError, match="reference_initial_state_grid requires"):
        cbo.solve_interface_cbo_scan(
            object(),
            np.array([0.0]),
            reference_initial_state_grid=np.array([0.0, 1.0]),
        )


def test_two_sided_scan_reduces_grid_and_propagates_topology(monkeypatch):
    short_calls = []
    jv_calls = []

    def fake_short(grid, stack, *, initial_state=None, **kwargs):
        short_calls.append((grid.copy(), kwargs))
        return _state(float(stack), 100.0)

    _install_fake_scan(monkeypatch, fake_short)
    monkeypatch.setattr(
        cbo,
        "build_two_sided_trace_grid",
        lambda grid, stack: np.array([grid[0], grid[-1]]),
    )

    def fake_jv(grid, stack, voltages, **kwargs):
        jv_calls.append((grid.copy(), kwargs))
        voltage_array = np.asarray(voltages, dtype=float)
        return SimpleNamespace(
            certified=True,
            metrics_certified=True,
            voltages_V=voltage_array,
            currents_A_m2=np.linspace(100.0, -1.0, len(voltage_array)),
            points=tuple(
                SimpleNamespace(certified=True) for _ in voltage_array
            ),
            metrics=JVMetrics(V_oc=1.0, J_sc=100.0, FF=0.8, PCE=0.2),
        )

    monkeypatch.setattr(cbo, "solve_quasi_fermi_jv_sweep", fake_jv)
    result = cbo.solve_interface_cbo_scan(
        object(),
        np.array([0.0]),
        voltages_V=np.array([0.0, 1.0]),
        interface_transport_model=cbo.FERMI_DIRAC_RICHARDSON,
        interface_topology=cbo.TWO_SIDED_TRACE,
    )

    assert result.interface_topology == cbo.TWO_SIDED_TRACE
    assert result.grid_node_count == 2
    assert result.grid_interval_count == 1
    assert short_calls[0][1]["interface_topology"] == cbo.TWO_SIDED_TRACE
    assert jv_calls[0][1]["interface_topology"] == cbo.TWO_SIDED_TRACE
    np.testing.assert_array_equal(short_calls[0][0], np.array([0.0, 2.0]))
    np.testing.assert_array_equal(jv_calls[0][0], np.array([0.0, 2.0]))


def test_two_sided_scan_rejects_non_fd_transport_before_grid_build():
    with pytest.raises(ValueError, match="currently requires"):
        cbo.solve_interface_cbo_scan(
            object(),
            np.array([0.0]),
            interface_topology=cbo.TWO_SIDED_TRACE,
        )


def test_two_sided_scan_rejects_shared_node_despike_protocol():
    stack = SimpleNamespace(het_recomb_despike=0.53)
    with pytest.raises(ValueError, match="explicitly set"):
        cbo.solve_interface_cbo_scan(
            stack,
            np.array([0.0]),
            interface_transport_model=cbo.FERMI_DIRAC_RICHARDSON,
            interface_topology=cbo.TWO_SIDED_TRACE,
        )


@pytest.mark.parametrize(
    "values, message",
    [
        (np.array([0.1, 0.2]), "reference_delta"),
        (np.array([0.0, 0.0]), "strictly increasing"),
    ],
)
def test_scan_rejects_ambiguous_axes(values, message):
    with pytest.raises(ValueError, match=message):
        cbo.solve_interface_cbo_scan(object(), values)


@pytest.mark.parametrize("N_grid", [0, -1, 3.5, True])
def test_scan_rejects_invalid_grid_size(N_grid):
    with pytest.raises(ValueError, match="N_grid"):
        cbo.solve_interface_cbo_scan(object(), np.array([0.0]), N_grid=N_grid)


def test_scan_records_recomputed_built_in_policy(monkeypatch):
    def fake_short(grid, stack, *, initial_state, **kwargs):
        return _state(float(stack), 100.0)

    _install_fake_scan(monkeypatch, fake_short)
    result = cbo.solve_interface_cbo_scan(
        object(),
        np.array([0.0]),
        boundary_policy=cbo.RECOMPUTED_BUILT_IN,
        calculate_jv_metrics=False,
    )

    assert result.sync_vbi
    assert result.boundary_policy == cbo.RECOMPUTED_BUILT_IN


def test_nested_voltage_ladder_solves_only_the_finest_branch(monkeypatch):
    def fake_short(grid, stack, *, initial_state, **kwargs):
        return _state(float(stack), 100.0)

    _install_fake_scan(monkeypatch, fake_short)
    monkeypatch.setattr(cbo, "thermodynamic_voc_ceiling", lambda stack: 2.0)
    calls = []

    def fake_jv(grid, stack, voltages, **kwargs):
        calls.append((np.asarray(voltages).copy(), kwargs))
        currents = 100.0 * (1.0 - np.asarray(voltages))
        points = tuple(
            SimpleNamespace(certified=True) for _ in np.asarray(voltages)
        )
        metrics = cbo.compute_metrics(
            np.asarray(voltages),
            currents,
            V_oc_max=2.0,
            validity=[True] * len(points),
        )
        return SimpleNamespace(
            voltages_V=np.asarray(voltages),
            currents_A_m2=currents,
            points=points,
            metrics=metrics,
            certified=True,
            metrics_certified=True,
        )

    monkeypatch.setattr(cbo, "solve_quasi_fermi_jv_sweep", fake_jv)
    voltage_grids = tuple(
        np.linspace(0.0, 1.2, count) for count in (5, 9, 17)
    )

    result = cbo.solve_interface_cbo_scan(
        object(),
        np.array([0.0]),
        voltage_grids_V=voltage_grids,
    )

    assert len(calls) == 1
    assert len(calls[0][0]) == 17
    assert calls[0][1]["stop_after_voc"] is True
    assert calls[0][1]["voc_stop_grid_V"] == pytest.approx(voltage_grids[0])
    assert [
        sample.voltage_point_count
        for sample in result.points[0].voltage_grid_metrics
    ] == [5, 9, 17]
    assert result.certified


def test_scan_rejects_non_nested_voltage_grids(monkeypatch):
    def fake_short(grid, stack, *, initial_state, **kwargs):
        return _state(float(stack), 100.0)

    _install_fake_scan(monkeypatch, fake_short)

    with pytest.raises(ValueError, match="strict subset"):
        cbo.solve_interface_cbo_scan(
            object(),
            np.array([0.0]),
            voltage_grids_V=(
                np.array([0.0, 0.5, 1.0]),
                np.array([0.0, 0.4, 0.8, 1.0]),
            ),
        )


def _with_voltage_metric_ladder(base, *, voc, ff, pce):
    counts = (29, 57, 113)
    samples = tuple(
        cbo.CBOJVMetricsGridSample(
            voltage_point_count=count,
            voltage_interval_count=count - 1,
            metrics=JVMetrics(
                V_oc=voc[index],
                J_sc=100.0,
                FF=ff[index],
                PCE=pce[index],
            ),
            certified=True,
        )
        for index, count in enumerate(counts)
    )
    point = dataclasses.replace(
        base.points[0],
        voltage_grid_metrics=samples,
    )
    return dataclasses.replace(
        base,
        points=(point,),
        voltage_grids_V=tuple(
            np.linspace(0.0, 1.4, count) for count in counts
        ),
    )


def test_voltage_grid_certificate_accepts_contracting_metric_changes(monkeypatch):
    def fake_short(grid, stack, *, initial_state, **kwargs):
        return _state(float(stack), 100.0)

    _install_fake_scan(monkeypatch, fake_short)
    base = cbo.solve_interface_cbo_scan(object(), np.array([0.0]))
    ladder = _with_voltage_metric_ladder(
        base,
        voc=(1.05, 1.068, 1.0695),
        ff=(0.894, 0.896, 0.8966),
        pce=(0.220, 0.2228, 0.2230),
    )

    certificate = cbo.certify_cbo_voltage_grid_convergence(ladder)

    assert certificate.certified
    assert certificate.voltage_interval_counts == (28, 56, 112)
    point = certificate.points[0]
    assert point.final_voc_change_V == pytest.approx(1.5e-3)
    assert point.final_ff_change == pytest.approx(6.0e-4)
    assert point.final_pce_change == pytest.approx(2.0e-4)


def test_voltage_grid_certificate_rejects_aliasing_plateau(monkeypatch):
    def fake_short(grid, stack, *, initial_state, **kwargs):
        return _state(float(stack), 100.0)

    _install_fake_scan(monkeypatch, fake_short)
    base = cbo.solve_interface_cbo_scan(object(), np.array([0.0]))
    ladder = _with_voltage_metric_ladder(
        base,
        voc=(1.069, 1.069, 1.069),
        ff=(0.896, 0.896, 0.896),
        pce=(0.2228, 0.2228, 0.2230),
    )

    certificate = cbo.certify_cbo_voltage_grid_convergence(ladder)

    assert not certificate.certified
    assert certificate.points[0].successive_pce_change_ratios == (math.inf,)
    assert "do not contract" in " ".join(certificate.points[0].reasons)


def test_voltage_grid_certificate_ignores_sub_tolerance_ratio_noise(monkeypatch):
    def fake_short(grid, stack, *, initial_state, **kwargs):
        return _state(float(stack), 100.0)

    _install_fake_scan(monkeypatch, fake_short)
    base = cbo.solve_interface_cbo_scan(object(), np.array([0.0]))
    ladder = _with_voltage_metric_ladder(
        base,
        voc=(1.069, 1.069, 1.069),
        ff=(0.896, 0.896, 0.896),
        pce=(0.22280, 0.22281, 0.22283),
    )

    certificate = cbo.certify_cbo_voltage_grid_convergence(ladder)

    assert certificate.points[0].successive_pce_change_ratios == pytest.approx(
        (2.0,)
    )
    assert certificate.certified


def test_grid_certificate_uses_actual_intervals_and_union_envelope(monkeypatch):
    def fake_short(grid, stack, *, initial_state, **kwargs):
        delta = float(stack)
        return _state(delta, 100.0 if delta < 0.3 else 85.0)

    _install_fake_scan(monkeypatch, fake_short)
    base = cbo.solve_interface_cbo_scan(
        object(),
        np.array([0.0, 0.4]),
        calculate_jv_metrics=False,
        minimum_delta_step_eV=1.0e-3,
    )
    ladder = tuple(
        dataclasses.replace(
            base,
            N_grid=requested,
            grid_node_count=intervals + 1,
            grid_interval_count=intervals,
        )
        for requested, intervals in ((12, 10), (24, 20), (48, 40))
    )

    certificate = cbo.certify_cbo_grid_convergence(
        ladder,
        maximum_envelope_width_eV=2.0e-3,
    )

    assert certificate.certified
    assert certificate.grid_interval_counts == (10, 20, 40)
    assert certificate.envelope_width_eV <= 2.0e-3
    assert certificate.successive_midpoint_shifts_eV == pytest.approx((0.0, 0.0))
    assert certificate.successive_shift_ratios == (0.0,)
    assert certificate.reference_relative_spread == 0.0


def test_grid_certificate_rejects_noncontracting_critical_drift(monkeypatch):
    def fake_short(grid, stack, *, initial_state, **kwargs):
        delta = float(stack)
        return _state(delta, 100.0 if delta < 0.3 else 85.0)

    _install_fake_scan(monkeypatch, fake_short)
    base = cbo.solve_interface_cbo_scan(
        object(),
        np.array([0.0, 0.4]),
        calculate_jv_metrics=False,
        minimum_delta_step_eV=1.0e-3,
    )
    ladder = []
    for intervals, midpoint in ((10, 0.400), (20, 0.395), (40, 0.390)):
        critical = tuple(
            dataclasses.replace(
                item,
                lower_delta_ec_eV=midpoint - 2.0e-4,
                upper_delta_ec_eV=midpoint + 2.0e-4,
            )
            if item.metric == "Jsc"
            else item
            for item in base.critical_intervals
        )
        ladder.append(
            dataclasses.replace(
                base,
                grid_interval_count=intervals,
                critical_intervals=critical,
            )
        )

    certificate = cbo.certify_cbo_grid_convergence(
        ladder,
        maximum_envelope_width_eV=2.0e-2,
    )

    assert not certificate.certified
    assert certificate.successive_shift_ratios == pytest.approx((1.0,))
    assert "do not contract" in " ".join(certificate.reasons)


def test_grid_certificate_rejects_mixed_mesh_protocols(monkeypatch):
    def fake_short(grid, stack, *, initial_state, **kwargs):
        delta = float(stack)
        return _state(delta, 100.0 if delta < 0.3 else 85.0)

    _install_fake_scan(monkeypatch, fake_short)
    base = cbo.solve_interface_cbo_scan(
        object(),
        np.array([0.0, 0.4]),
        calculate_jv_metrics=False,
        minimum_delta_step_eV=1.0e-3,
    )
    ladder = (
        dataclasses.replace(base, grid_interval_count=10),
        dataclasses.replace(base, grid_interval_count=20),
        dataclasses.replace(
            base,
            grid_interval_count=40,
            grid_alphas=(4.0, 4.0, 4.0),
        ),
    )

    certificate = cbo.certify_cbo_grid_convergence(ladder)

    assert not certificate.certified
    assert "do not share one physical protocol" in " ".join(
        certificate.reasons
    )


def test_grid_certificate_rejects_mixed_qf_coordinates(monkeypatch):
    def fake_short(grid, stack, *, initial_state, **kwargs):
        delta = float(stack)
        return _state(delta, 100.0 if delta < 0.3 else 85.0)

    _install_fake_scan(monkeypatch, fake_short)
    base = cbo.solve_interface_cbo_scan(
        object(),
        np.array([0.0, 0.4]),
        calculate_jv_metrics=False,
        minimum_delta_step_eV=1.0e-3,
    )
    ladder = (
        dataclasses.replace(base, grid_interval_count=10),
        dataclasses.replace(base, grid_interval_count=20),
        dataclasses.replace(
            base,
            grid_interval_count=40,
            qf_coordinate_system="nodal_increment",
        ),
    )

    certificate = cbo.certify_cbo_grid_convergence(ladder)

    assert not certificate.certified
    assert "do not share one physical protocol" in " ".join(
        certificate.reasons
    )


def test_grid_certificate_rejects_mixed_interface_topologies(monkeypatch):
    def fake_short(grid, stack, *, initial_state, **kwargs):
        delta = float(stack)
        return _state(delta, 100.0 if delta < 0.3 else 85.0)

    _install_fake_scan(monkeypatch, fake_short)
    base = cbo.solve_interface_cbo_scan(
        object(),
        np.array([0.0, 0.4]),
        calculate_jv_metrics=False,
        minimum_delta_step_eV=1.0e-3,
    )
    ladder = (
        dataclasses.replace(base, grid_interval_count=10),
        dataclasses.replace(base, grid_interval_count=20),
        dataclasses.replace(
            base,
            grid_interval_count=40,
            interface_topology=cbo.TWO_SIDED_TRACE,
        ),
    )

    certificate = cbo.certify_cbo_grid_convergence(ladder)

    assert not certificate.certified
    assert "do not share one physical protocol" in " ".join(
        certificate.reasons
    )


def test_grid_certificate_rejects_mixed_despike_protocols(monkeypatch):
    def fake_short(grid, stack, *, initial_state, **kwargs):
        delta = float(stack)
        return _state(delta, 100.0 if delta < 0.3 else 85.0)

    _install_fake_scan(monkeypatch, fake_short)
    base = cbo.solve_interface_cbo_scan(
        object(),
        np.array([0.0, 0.4]),
        calculate_jv_metrics=False,
        minimum_delta_step_eV=1.0e-3,
    )
    ladder = (
        dataclasses.replace(base, grid_interval_count=10),
        dataclasses.replace(base, grid_interval_count=20),
        dataclasses.replace(
            base,
            grid_interval_count=40,
            heterojunction_recombination_despike=0.53,
        ),
    )

    certificate = cbo.certify_cbo_grid_convergence(ladder)

    assert not certificate.certified
    assert "do not share one physical protocol" in " ".join(
        certificate.reasons
    )


def test_statistics_certificate_rejects_degenerate_boltzmann_states(monkeypatch):
    def fake_short(grid, stack, *, initial_state, **kwargs):
        return _state(float(stack), 100.0)

    _install_fake_scan(monkeypatch, fake_short)
    base = cbo.solve_interface_cbo_scan(
        object(),
        np.array([0.0]),
        calculate_jv_metrics=False,
    )
    populated_trace = tuple(
        dataclasses.replace(sample, interface_max_state_to_dos=0.2)
        for sample in base.short_circuit_trace
    )
    boltzmann = dataclasses.replace(
        base,
        interface_transport_model="scaps_thermionic",
        short_circuit_trace=populated_trace,
    )
    fermi = dataclasses.replace(
        base,
        interface_transport_model="fermi_richardson",
        short_circuit_trace=tuple(
            dataclasses.replace(sample, interface_max_state_to_dos=0.9)
            for sample in base.short_circuit_trace
        ),
    )
    fermi_dirac = dataclasses.replace(
        base,
        interface_transport_model="fermi_dirac_richardson",
        short_circuit_trace=tuple(
            dataclasses.replace(sample, interface_max_state_to_dos=3.0)
            for sample in base.short_circuit_trace
        ),
    )

    boltzmann_certificate = cbo.certify_cbo_statistics_validity(boltzmann)
    fermi_certificate = cbo.certify_cbo_statistics_validity(fermi)
    fermi_dirac_certificate = cbo.certify_cbo_statistics_validity(fermi_dirac)

    assert not boltzmann_certificate.certified
    assert boltzmann_certificate.allowed_state_to_dos == 0.1
    assert fermi_certificate.certified
    assert fermi_certificate.allowed_state_to_dos == 1.0
    assert fermi_dirac_certificate.certified
    assert fermi_dirac_certificate.allowed_state_to_dos is None
    assert fermi_dirac_certificate.maximum_reduced_fermi_level == pytest.approx(
        2.11839,
        rel=1.0e-4,
    )


def test_external_scaps_certificate_checks_normalized_trend_and_hash(
    monkeypatch,
    tmp_path,
):
    def fake_short(grid, stack, *, initial_state, **kwargs):
        delta = float(stack)
        return _state(delta, 100.0 if delta < 0.41 else 10.0)

    _install_fake_scan(monkeypatch, fake_short)
    result = cbo.solve_interface_cbo_scan(
        object(),
        np.array([0.0, 0.4, 0.41]),
        calculate_jv_metrics=False,
    )
    reference = {
        "schema": "solarlab.scaps_cbo_reference",
        "schema_version": "1.0",
        "source_xlsx": "partner.xlsx",
        "source_pdf": "partner.pdf",
        "extracted_at": "2026-01-01",
        "cbo_validation": {
            "solver": "SCAPS-1D",
            "solver_version": "3.3.11",
            "delta_ec_convention": "chi_absorber - chi_etl",
            "swept_parameter": "etl_electron_affinity",
            "boundary_policy": "fixed_contacts",
            "reference_delta_ec_eV": 0.0,
            "temperature_K": 300.0,
            "illumination": "AM1.5G",
            "independently_generated": True,
            "interpolated": False,
            "source_export_sha256": "a" * 64,
            "source_deck_sha256": "b" * 64,
            "parameter_manifest_sha256": "c" * 64,
        },
        "sweeps": {
            "CHI_ETL": {
                "n_points": 3,
                "points": [
                    {"x": 0.0, "Jsc_mA_cm2": 10.0},
                    {"x": 0.4, "Jsc_mA_cm2": 10.0},
                    {"x": 0.41, "Jsc_mA_cm2": 1.0},
                ]
            }
        },
    }
    path = tmp_path / "scaps_reference.json"
    path.write_text(json.dumps(reference), encoding="utf-8")

    validation = cbo.compare_cbo_scan_to_scaps_reference(result, path)

    assert validation.certified
    assert validation.reference_audit.certified
    assert len(validation.reference_sha256) == 64
    certificate = validation.certificates[0]
    assert certificate.max_normalized_error == pytest.approx(0.0)
    assert certificate.critical_interval_distance_eV == 0.0
    assert certificate.reference_critical_interval_width_eV == pytest.approx(
        0.01
    )


def test_external_certificate_rejects_sparse_unaudited_reference(
    monkeypatch,
    tmp_path,
):
    def fake_short(grid, stack, *, initial_state, **kwargs):
        delta = float(stack)
        return _state(delta, 100.0 if delta < 0.5 else 10.0)

    _install_fake_scan(monkeypatch, fake_short)
    result = cbo.solve_interface_cbo_scan(
        object(),
        np.array([0.0, 0.4, 0.5]),
        calculate_jv_metrics=False,
    )
    reference = {
        "sweeps": {
            "CHI_ETL": {
                "n_points": 3,
                "points": [
                    {"x": 0.0, "Jsc_mA_cm2": 10.0},
                    {"x": 0.4, "Jsc_mA_cm2": 10.0},
                    {"x": 0.5, "Jsc_mA_cm2": 1.0},
                ],
            }
        }
    }
    path = tmp_path / "legacy-reference.json"
    path.write_text(json.dumps(reference), encoding="utf-8")

    validation = cbo.compare_cbo_scan_to_scaps_reference(result, path)

    assert not validation.certified
    assert not validation.reference_audit.certified
    assert "lacks cbo_validation" in " ".join(
        validation.reference_audit.reasons
    )
    certificate = validation.certificates[0]
    assert not certificate.certified
    assert certificate.reference_critical_interval_width_eV == pytest.approx(
        0.1
    )
    assert "critical-interval width" in " ".join(certificate.reasons)
