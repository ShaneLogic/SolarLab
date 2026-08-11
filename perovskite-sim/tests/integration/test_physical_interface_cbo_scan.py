"""Slow device-level gate for the physical ETL/PVK interface response."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from perovskite_sim.experiments.cbo_scan import (
    RECOMPUTED_BUILT_IN,
    solve_interface_cbo_scan,
)
from perovskite_sim.scaps_compat import load_scaps_yaml


@pytest.fixture(scope="module")
def physical_cbo_scan():
    stack = load_scaps_yaml("configs/scaps_mirror_v2.yaml")
    return solve_interface_cbo_scan(
        stack,
        np.array([0.0, 0.4, 0.5]),
        voltages_V=np.linspace(0.0, 1.4, 15),
        N_grid=20,
        boundary_policy=RECOMPUTED_BUILT_IN,
    )


@pytest.fixture(scope="module")
def edge_qf_n60_scan():
    stack = load_scaps_yaml("configs/scaps_mirror_v2.yaml")
    stack = replace(
        stack,
        grid_interval_weights=(1.0, 2.0, 1.0),
        grid_alphas=(3.0, 3.25, 3.0),
    )
    return solve_interface_cbo_scan(
        stack,
        np.array([0.0, 0.4]),
        N_grid=60,
        boundary_policy="fixed_contacts",
        calculate_jv_metrics=False,
        interface_transport_model="fermi_richardson",
        minimum_delta_step_eV=5.0e-4,
        maximum_delta_step_eV=5.0e-2,
    )


@pytest.fixture(scope="module")
def two_sided_fd_n30_scan():
    stack = load_scaps_yaml("configs/scaps_mirror_v2.yaml")
    stack = replace(stack, het_recomb_despike=0.0)
    return solve_interface_cbo_scan(
        stack,
        np.array([0.0, 0.4, 0.5]),
        N_grid=30,
        boundary_policy="fixed_contacts",
        calculate_jv_metrics=False,
        interface_transport_model="fermi_dirac_richardson",
        interface_topology="two_sided_trace",
        minimum_delta_step_eV=5.0e-4,
        maximum_delta_step_eV=5.0e-2,
    )


@pytest.mark.slow
def test_physical_interface_cbo_onset_and_jv_are_certified(physical_cbo_scan):
    result = physical_cbo_scan
    assert result.certified
    points = {point.delta_ec_eV: point for point in result.points}
    assert set(points) == {0.0, 0.4}
    assert points[0.0].metrics.J_sc == pytest.approx(234.17, rel=3.0e-3)
    assert points[0.4].metrics.J_sc < 210.0
    assert points[0.4].metrics.FF < 0.65
    assert points[0.4].metrics.PCE < 0.12
    jsc_interval = next(
        interval
        for interval in result.critical_intervals
        if interval.metric == "Jsc"
    )
    assert jsc_interval.resolved
    assert 0.385 < jsc_interval.lower_delta_ec_eV < 0.4
    assert 0.39 < jsc_interval.upper_delta_ec_eV <= 0.4


@pytest.mark.slow
def test_bulk_statistics_endpoint_is_reported_not_extrapolated(physical_cbo_scan):
    result = physical_cbo_scan
    assert not result.complete
    assert len(result.terminations) == 1
    termination = result.terminations[0]
    assert termination.direction == "positive"
    assert 0.41 < termination.last_certified_delta_ec_eV < 0.42
    assert 0.41 < termination.first_failed_delta_ec_eV < 0.42
    assert (
        termination.first_failed_delta_ec_eV
        - termination.last_certified_delta_ec_eV
        <= result.minimum_delta_step_eV
    )
    assert "audited exponential range" in termination.reason


@pytest.mark.slow
def test_edge_qf_coordinates_close_the_n60_residual_floor(edge_qf_n60_scan):
    result = edge_qf_n60_scan
    assert result.complete
    assert result.certified
    assert result.qf_coordinate_system == "edge_drop"
    assert result.reference_grid_predictor_recoveries == 0
    interval = next(
        item for item in result.critical_intervals if item.metric == "Jsc"
    )
    assert interval.resolved
    assert 0.3795 < interval.lower_delta_ec_eV < 0.3805
    assert 0.3800 < interval.upper_delta_ec_eV < 0.3810
    assert max(
        point.short_circuit_state.electron_continuity_bound_A_m2
        for point in result.points
    ) < 1.0e-4
    assert max(
        sample.face_current_spread_A_m2
        for sample in result.short_circuit_trace
    ) < 1.0e-4


@pytest.mark.slow
def test_two_sided_fd_topology_closes_full_device_cbo_scan(
    two_sided_fd_n30_scan,
):
    result = two_sided_fd_n30_scan
    assert result.complete
    assert result.certified
    assert result.interface_topology == "two_sided_trace"
    assert result.heterojunction_recombination_despike == 0.0
    assert result.grid_interval_count == 28
    interval = next(
        item for item in result.critical_intervals if item.metric == "Jsc"
    )
    assert interval.resolved
    assert 0.398 < interval.lower_delta_ec_eV < 0.400
    assert 0.398 < interval.upper_delta_ec_eV < 0.400
    assert max(
        sample.face_current_spread_A_m2
        for sample in result.short_circuit_trace
    ) < 1.0e-4
    assert max(
        sample.interface_local_residual
        for sample in result.short_circuit_trace
    ) < 1.0e-7
