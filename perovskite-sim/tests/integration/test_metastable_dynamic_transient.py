"""D7-E4 fully dynamic metastable configuration transient."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from perovskite_sim.experiments.metastable_preparation import (
    MetastablePreparationError,
    prepare_metastable_configuration,
)
from perovskite_sim.experiments.metastable_transient import (
    MetastableTransientError,
    run_metastable_configuration_transient,
)
from perovskite_sim.physics.metastable_defect_closure import (
    evaluate_metastable_configuration_closure,
)
from perovskite_sim.physics.metastable_dynamic_state import (
    MetastableDynamicStateError,
    advance_metastable_configuration,
    configuration_from_logit,
    configuration_logit,
)

from .test_metastable_preparation_and_frozen_measurement import (  # noqa: F401
    GAP_EV,
    NC_M3,
    NV_M3,
    _definition,
    _grid,
    _protocol,
    _stack,
)


@pytest.fixture(scope="module")
def prepared():
    grid = _grid()
    stack = _stack()
    frozen = prepare_metastable_configuration(
        grid,
        stack,
        _definition(),
        _protocol(),
        layer_name="meta_absorber",
    )
    return grid, stack, frozen


def _relaxation_time(frozen, temperature_K: float) -> np.ndarray:
    count = frozen.grid_m.size
    mask = frozen.active_nodes
    closure = evaluate_metastable_configuration_closure(
        np.asarray(frozen.preparation_state.y[:count], dtype=float)[mask],
        np.asarray(frozen.preparation_state.y[count : 2 * count], dtype=float)[mask],
        frozen.definition,
        band_gap_eV=GAP_EV,
        effective_conduction_dos_m3=NC_M3,
        effective_valence_dos_m3=NV_M3,
        temperature_K=temperature_K,
    )
    return 1.0 / (
        np.asarray(closure.donor_to_acceptor_rate_s1)
        + np.asarray(closure.acceptor_to_donor_rate_s1)
    )


def test_slow_limit_leaves_the_prepared_configuration_in_place(prepared):
    grid, stack, frozen = prepared
    tau = _relaxation_time(frozen, float(stack.T))
    dt = float(np.min(tau)) * 1.0e-4
    times = np.array([0.0, dt, 2.0 * dt, 3.0 * dt])

    result = run_metastable_configuration_transient(
        grid,
        stack,
        frozen,
        times,
        V_app=0.0,
        illuminated=False,
    )

    assert result.certificate.certified
    assert result.certificate.reasons == ()
    assert result.certificate.maximum_step_over_relaxation_time < 1.0e-3
    moved = np.max(np.abs(result.donor_fraction[-1] - result.donor_fraction[0]))
    assert moved < 1.0e-3
    # It has not simply reached the stationary point by another route.
    assert (
        np.max(np.abs(result.donor_fraction[-1] - result.stationary_fraction[-1]))
        > 1.0e-3
    )


def test_fast_limit_lands_on_the_measurement_stationary_configuration(prepared):
    grid, stack, frozen = prepared
    tau = _relaxation_time(frozen, float(stack.T))
    dt = float(np.max(tau)) * 50.0
    times = np.array([0.0, dt, 2.0 * dt])

    # A single step far beyond the relaxation time is a limit probe, not a
    # resolved trajectory, so the splitting gate is expected to fire and the
    # certificate is read rather than required.
    result = run_metastable_configuration_transient(
        grid,
        stack,
        frozen,
        times,
        V_app=0.0,
        illuminated=False,
        require_certificate=False,
    )

    assert result.certificate.maximum_step_over_relaxation_time > 10.0
    assert result.certificate.reasons == ("operator_splitting_not_resolved",)
    np.testing.assert_allclose(
        result.donor_fraction[-1],
        result.stationary_fraction[-1],
        rtol=0.0,
        atol=1.0e-15,
    )
    assert np.max(np.abs(result.donor_fraction[-1] - result.donor_fraction[0])) > 1.0e-3


def test_resolved_trace_is_certified_and_relaxes_monotonically(prepared):
    grid, stack, frozen = prepared
    tau = _relaxation_time(frozen, float(stack.T))
    dt = float(np.min(tau)) * 0.05
    times = np.arange(5, dtype=float) * dt

    result = run_metastable_configuration_transient(
        grid,
        stack,
        frozen,
        times,
        V_app=0.0,
        illuminated=False,
    )

    assert result.certificate.certified
    assert result.certificate.step_count == times.size
    distance = np.max(
        np.abs(result.donor_fraction - result.stationary_fraction[0]),
        axis=1,
    )
    # Every step moves toward the stationary configuration.
    assert np.all(np.diff(distance) < 0.0)


def test_charge_transfer_identity_holds_to_machine_precision(prepared):
    """Each converted defect moves exactly two elementary charges."""
    grid, stack, frozen = prepared
    tau = _relaxation_time(frozen, float(stack.T))
    times = np.array([0.0, 0.05, 0.1]) * float(np.min(tau))

    result = run_metastable_configuration_transient(
        grid,
        stack,
        frozen,
        times,
        V_app=0.0,
        illuminated=False,
    )

    assert result.certificate.maximum_charge_transfer_relative_error < 1.0e-12
    assert result.configuration_charge_transfer_C_m2.size == times.size - 1
    assert np.all(np.isfinite(result.configuration_charge_transfer_C_m2))


def test_dynamic_state_stays_bounded_without_clipping(prepared):
    grid, stack, frozen = prepared
    tau = _relaxation_time(frozen, float(stack.T))
    times = np.array([0.0, 1.0, 10.0, 100.0]) * float(np.max(tau))

    result = run_metastable_configuration_transient(
        grid,
        stack,
        frozen,
        times,
        V_app=0.0,
        illuminated=False,
        require_certificate=False,
    )

    assert result.certificate.clipping_used is False
    assert result.certificate.minimum_fraction > 0.0
    assert result.certificate.maximum_fraction < 1.0
    assert np.all(np.isfinite(result.donor_fraction))


def test_analytic_step_is_the_exact_two_state_solution(prepared):
    """The step is the closed-form solution, so it is exact for fixed rates."""
    _grid_m, stack, frozen = prepared
    count = frozen.grid_m.size
    mask = frozen.active_nodes
    n = np.asarray(frozen.preparation_state.y[:count], dtype=float)[mask]
    p = np.asarray(frozen.preparation_state.y[count : 2 * count], dtype=float)[mask]
    tau = _relaxation_time(frozen, float(stack.T))
    dt = float(np.min(tau))

    step = advance_metastable_configuration(
        frozen.donor_fraction,
        n,
        p,
        frozen.definition,
        dt,
        band_gap_eV=GAP_EV,
        effective_conduction_dos_m3=NC_M3,
        effective_valence_dos_m3=NV_M3,
        temperature_K=float(stack.T),
    )

    expected = step.stationary_fraction + (
        frozen.donor_fraction - step.stationary_fraction
    ) * np.exp(-step.relaxation_rate_s1 * dt)
    np.testing.assert_allclose(
        step.donor_fraction,
        expected,
        rtol=0.0,
        atol=1.0e-16,
    )
    # Two half steps must equal one whole step at fixed carriers.
    half = advance_metastable_configuration(
        frozen.donor_fraction,
        n,
        p,
        frozen.definition,
        0.5 * dt,
        band_gap_eV=GAP_EV,
        effective_conduction_dos_m3=NC_M3,
        effective_valence_dos_m3=NV_M3,
        temperature_K=float(stack.T),
    )
    whole = advance_metastable_configuration(
        half.donor_fraction,
        n,
        p,
        frozen.definition,
        0.5 * dt,
        band_gap_eV=GAP_EV,
        effective_conduction_dos_m3=NC_M3,
        effective_valence_dos_m3=NV_M3,
        temperature_K=float(stack.T),
    )
    np.testing.assert_allclose(
        whole.donor_fraction,
        step.donor_fraction,
        rtol=1.0e-14,
        atol=0.0,
    )


def test_logit_coordinate_rejects_the_closed_interval():
    with pytest.raises(MetastableDynamicStateError, match="strictly inside"):
        configuration_logit(np.array([0.0]))
    with pytest.raises(MetastableDynamicStateError, match="strictly inside"):
        configuration_logit(np.array([1.0]))
    with pytest.raises(MetastableDynamicStateError, match="saturated"):
        configuration_from_logit(np.array([1.0e4]))
    round_trip = configuration_from_logit(configuration_logit(np.array([0.25, 0.75])))
    np.testing.assert_allclose(round_trip, [0.25, 0.75], rtol=1.0e-15, atol=0.0)


def test_transient_rejects_a_mismatched_grid_or_temperature(prepared):
    grid, stack, frozen = prepared
    times = np.array([0.0, 1.0])

    with pytest.raises(MetastablePreparationError, match="grid does not match"):
        run_metastable_configuration_transient(
            _grid(16),
            stack,
            frozen,
            times,
        )

    scaled = replace(stack, T=350.0, mode="full")
    with pytest.raises(MetastableTransientError, match="LEGACY tier"):
        run_metastable_configuration_transient(grid, scaled, frozen, times)
