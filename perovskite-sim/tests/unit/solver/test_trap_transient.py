from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from perovskite_sim.models.defects import ACCEPTOR, DONOR
from perovskite_sim.physics.trap_kinetics import (
    TrapReservoirKinetics,
    TrapReservoirState,
)
from perovskite_sim.physics.trap_transient import TrapTransientError
from perovskite_sim.solver.trap_transient import (
    LocalTrapTransientCertificationError,
    LocalTrapTransientPolicy,
    solve_local_trap_transient,
)


def _kinetics(scale: float = 1.0) -> TrapReservoirKinetics:
    return TrapReservoirKinetics(
        identifier=f"solver/transient/{scale:.1e}",
        electron_capture_coefficients_m3_s=scale * np.array([2.0e-14]),
        hole_capture_coefficients_m3_s=scale * np.array([5.0e-15]),
        electron_reference_densities_m3=np.array([1.0e14]),
        hole_reference_densities_m3=np.array([2.0e15]),
    )


def _state() -> TrapReservoirState:
    return TrapReservoirState(
        electron_densities_m3=np.array([3.0e20]),
        hole_densities_m3=np.array([4.0e16]),
    )


def test_local_logit_solver_certifies_against_closed_form_and_charge_balance():
    result = solve_local_trap_transient(
        _kinetics(),
        _state(),
        0.13,
        np.linspace(0.0, 2.0e-6, 41),
        charge_transition=ACCEPTOR,
    )

    assert result.certified
    assert result.analytic_jacobian_used
    assert result.state_coordinate == "logit"
    assert result.maximum_closed_form_absolute_error < 1.0e-8
    assert result.maximum_charge_balance_relative_error == 0.0
    assert np.all((result.occupancy > 0.0) & (result.occupancy < 1.0))
    assert result.nfev > 0
    assert result.njev > 0


def test_acceptor_and_donor_have_identical_occupancy_but_distinct_charge():
    arguments = (
        _kinetics(),
        _state(),
        0.29,
        np.linspace(0.0, 8.0e-7, 21),
    )
    acceptor = solve_local_trap_transient(
        *arguments,
        charge_transition=ACCEPTOR,
    )
    donor = solve_local_trap_transient(
        *arguments,
        charge_transition=DONOR,
    )

    np.testing.assert_array_equal(acceptor.occupancy, donor.occupancy)
    np.testing.assert_array_equal(
        acceptor.trap_charge_rate_C_s,
        donor.trap_charge_rate_C_s,
    )
    assert np.all(acceptor.trap_charge_C < 0.0)
    assert np.all(donor.trap_charge_C > 0.0)


def test_fine_policy_reduces_the_closed_form_error():
    times = np.linspace(0.0, 2.0e-6, 37)
    coarse = solve_local_trap_transient(
        _kinetics(),
        _state(),
        0.21,
        times,
        charge_transition=ACCEPTOR,
        policy=LocalTrapTransientPolicy(
            rtol=1.0e-3,
            atol_logit=1.0e-5,
            max_closed_form_absolute_error=1.0e-3,
        ),
    )
    fine = solve_local_trap_transient(
        _kinetics(),
        _state(),
        0.21,
        times,
        charge_transition=ACCEPTOR,
        policy=LocalTrapTransientPolicy(
            rtol=1.0e-9,
            atol_logit=1.0e-11,
            max_closed_form_absolute_error=1.0e-7,
        ),
    )

    assert fine.maximum_closed_form_absolute_error < (
        coarse.maximum_closed_form_absolute_error
    )


def test_fast_and_slow_trap_limits_recover_qss_and_frozen_occupancy():
    initial = 0.19
    times = np.linspace(0.0, 1.0e-5, 31)
    fast = solve_local_trap_transient(
        _kinetics(),
        _state(),
        initial,
        times,
        charge_transition=ACCEPTOR,
    )
    slow = solve_local_trap_transient(
        _kinetics(1.0e-12),
        _state(),
        initial,
        times,
        charge_transition=ACCEPTOR,
    )

    assert abs(fast.occupancy[-1] - fast.exact_trace.quasi_steady_occupancy) < 1.0e-8
    assert abs(slow.occupancy[-1] - initial) < 1.0e-9


def test_overstrict_closed_form_gate_fails_closed_but_diagnostic_result_is_kept():
    policy = LocalTrapTransientPolicy(
        rtol=1.0e-5,
        atol_logit=1.0e-7,
        max_closed_form_absolute_error=1.0e-30,
    )
    arguments = (
        _kinetics(),
        _state(),
        0.23,
        np.linspace(0.0, 2.0e-6, 19),
    )
    with pytest.raises(
        LocalTrapTransientCertificationError,
        match="did not certify",
    ):
        solve_local_trap_transient(
            *arguments,
            charge_transition=ACCEPTOR,
            policy=policy,
        )
    diagnostic = solve_local_trap_transient(
        *arguments,
        charge_transition=ACCEPTOR,
        policy=policy,
        require_certified=False,
    )
    assert not diagnostic.certified
    assert diagnostic.maximum_closed_form_absolute_error > 1.0e-30


@pytest.mark.parametrize(
    "change",
    [
        {"rtol": 0.0},
        {"atol_logit": np.nan},
        {"max_step_s": -1.0},
        {"method": "RK45"},
    ],
)
def test_local_policy_rejects_non_strict_or_unsupported_values(change):
    with pytest.raises(ValueError):
        LocalTrapTransientPolicy(**change)


def test_local_solver_rejects_occupancy_endpoints_without_clipping():
    with pytest.raises(TrapTransientError, match="strictly inside"):
        solve_local_trap_transient(
            _kinetics(),
            _state(),
            0.0,
            np.linspace(0.0, 1.0e-6, 5),
            charge_transition=ACCEPTOR,
        )


def test_result_arrays_are_immutable():
    result = solve_local_trap_transient(
        _kinetics(),
        _state(),
        0.31,
        np.linspace(0.0, 1.0e-6, 11),
        charge_transition=DONOR,
    )
    with pytest.raises(ValueError):
        result.occupancy[0] = 0.5
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.certified = False
