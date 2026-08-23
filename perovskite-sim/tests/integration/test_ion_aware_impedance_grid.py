from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import perovskite_sim.experiments.ion_aware_impedance_grid as grid_module
from perovskite_sim.experiments.ion_aware_dc import (
    build_ion_aware_dc_protocol,
    solve_ion_aware_dc,
)
from perovskite_sim.experiments.jv_sweep import build_electrical_grid
from perovskite_sim.models.config_loader import load_device_from_yaml


@pytest.fixture(scope="module")
def resolved_grid_ladder():
    stack = load_device_from_yaml("configs/ionmonger_benchmark.yaml")
    dc_protocol = build_ion_aware_dc_protocol(
        stack,
        V_dc=0.9,
        illuminated=True,
    )
    states = tuple(
        solve_ion_aware_dc(
            build_electrical_grid(stack, nominal_grid),
            stack,
            dc_protocol,
        )
        for nominal_grid in (60, 90, 120)
    )
    protocol = grid_module.build_ion_aware_impedance_grid_protocol(
        states,
        np.logspace(-6, 1, 29),
    )
    result = grid_module.run_ion_aware_impedance_grid_ladder(
        protocol,
        dc_states=states,
        require_frequency_window_certificate=True,
    )
    return states, protocol, result


def test_resolved_ionmonger_grid_ladder_passes_each_frequency(
    resolved_grid_ladder,
):
    states, protocol, result = resolved_grid_ladder
    certificate = result.certificate

    assert protocol.grid_node_counts == (61, 91, 121)
    assert len(set(protocol.grid_sha256s)) == 3
    assert len(set(item.protocol_hash for item in protocol.impedance_protocols)) == 3
    assert all(state.numerically_certified for state in states)
    assert certificate.numerically_certified
    assert not certificate.thermodynamically_certified
    assert certificate.frequency_window_certified
    assert not certificate.certified
    assert certificate.numerical_reasons == ()
    assert certificate.reasons == ("contact_thermodynamics_not_certified",)
    assert (
        certificate.finest_pair_max_impedance_magnitude_relative_change
        < protocol.max_grid_impedance_magnitude_relative_change
    )
    assert (
        certificate.finest_pair_max_impedance_phase_change_deg
        < protocol.max_grid_impedance_phase_change_deg
    )
    assert len(certificate.frequency_point_certificates) == 29
    assert all(
        item.numerically_certified
        and len(item.grid_refinement_assessments) == 2
        and item.grid_refinement_assessments[-1].passed
        for item in certificate.frequency_point_certificates
    )
    assert all(
        len(grid_result.certificate.frequency_point_certificates) == 29
        for grid_result in result.grid_results
    )
    assert all(
        grid_result.frequency_window.max_observed_sampling_gap_decades
        <= protocol.impedance_protocols[0].max_frequency_sampling_gap_decades
        for grid_result in result.grid_results
    )


def test_grid_ladder_rejects_coordinate_replacement_before_ac_execution(
    resolved_grid_ladder,
):
    states, protocol, _result = resolved_grid_ladder
    changed_grid = states[1].x.copy()
    changed_grid[1] = np.nextafter(changed_grid[1], changed_grid[2])
    stale = replace(states[1], x=changed_grid)

    with pytest.raises(
        grid_module.IonAwareImpedanceGridCapabilityError,
        match="coordinates do not match",
    ):
        grid_module.run_ion_aware_impedance_grid_ladder(
            protocol,
            dc_states=(states[0], stale, states[2]),
        )


def test_grid_ladder_rejects_state_or_stack_replacement_before_ac_execution(
    resolved_grid_ladder,
):
    states, protocol, _result = resolved_grid_ladder
    changed_y = states[1].y.copy()
    changed_y[1] = np.nextafter(changed_y[1], np.inf)
    stale_state = replace(states[1], y=changed_y)
    changed_stack = replace(
        states[1],
        stack=replace(states[1].stack, Phi=states[1].stack.Phi * 1.01),
    )

    with pytest.raises(
        grid_module.IonAwareImpedanceGridCapabilityError,
        match="bound impedance protocol",
    ):
        grid_module.run_ion_aware_impedance_grid_ladder(
            protocol,
            dc_states=(states[0], stale_state, states[2]),
        )
    with pytest.raises(
        grid_module.IonAwareImpedanceGridCapabilityError,
        match="same device stack",
    ):
        grid_module.run_ion_aware_impedance_grid_ladder(
            protocol,
            dc_states=(states[0], changed_stack, states[2]),
        )


def test_same_results_fail_a_tighter_preregistered_grid_gate(
    resolved_grid_ladder,
):
    _states, protocol, result = resolved_grid_ladder
    strict = replace(
        protocol,
        max_grid_impedance_magnitude_relative_change=1.0e-5,
        max_grid_impedance_phase_change_deg=1.0e-5,
    )

    certificate = grid_module._grid_certificate(
        strict,
        result.grid_results,
    )

    assert not certificate.numerically_certified
    assert certificate.numerical_reasons == ("grid_refinement_not_converged",)
    assert any(
        not item.numerically_certified
        for item in certificate.frequency_point_certificates
    )
