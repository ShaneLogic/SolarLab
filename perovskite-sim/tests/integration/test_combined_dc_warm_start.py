"""Warm DC recovery preserves physical gates and the evaluation budget."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import OptimizeResult

from perovskite_sim.experiments import defect_ion_combined_impedance as combined
from perovskite_sim.experiments.interface_defect_ion_transient import (
    InterfaceDefectIonTransientError,
    InterfaceDefectIonTransientPolicy,
    run_interface_defect_ion_device_transient,
)
from perovskite_sim.experiments.one_dimensional_mechanism_r1 import (
    build_r1_material,
    solve_r1_dc,
)
from perovskite_sim.models.config_loader import load_device_from_yaml
from tests.integration.test_interface_defect_ion_transient import (
    _interface_grid,
    _stack,
)
from tests.integration.test_one_dimensional_mechanism_r1 import FIXTURE, binding


@pytest.fixture(scope="module")
def r1_case():
    stack = load_device_from_yaml(FIXTURE)
    grid, material = build_r1_material(stack, 4)
    return stack, grid, material, binding(stack)


def _run_r1(case, *, policy=None):
    stack, grid, material, reference = case
    return run_interface_defect_ion_device_transient(
        grid,
        stack,
        [0.0, 1.0e-8],
        [0.0, 0.005],
        mat=material,
        research_binding=reference,
        policy=policy,
    )


def _stalled_result(fun, initial, *, nfev, success=True):
    """Model optimizer termination without modifying any physical residual."""
    coordinate = np.asarray(initial, dtype=float).copy()
    return OptimizeResult(
        x=coordinate,
        fun=fun(coordinate),
        nfev=nfev,
        success=success,
    )


def test_four_interval_public_dc_matches_independent_fresh_solve(r1_case):
    result = _run_r1(r1_case)
    stack, _grid, _material, reference = r1_case
    _fresh_grid, _fresh_material, fresh, _local = solve_r1_dc(
        stack, 4, np.asarray(reference["f_ref"])
    )

    assert result.certificate.certified
    assert result.dc_state.certificate.certified
    assert fresh.certificate.certified
    assert result.dc_state.certificate.maximum_normalized_residual <= 1.0e-8
    np.testing.assert_array_equal(
        result.dark_reference.equilibrium_occupancy, reference["f_ref"]
    )
    for field in (
        "positive_ion_density_m3",
        "electron_density_m3",
        "hole_density_m3",
        "interface_occupancy",
    ):
        np.testing.assert_allclose(
            getattr(result.dc_state, field),
            getattr(fresh, field),
            rtol=1.0e-9,
            atol=0.0,
        )
    np.testing.assert_allclose(
        result.dc_state.potential_V, fresh.potential_V, rtol=0.0, atol=1.0e-11
    )


def test_accepted_legacy_warm_start_uses_one_attempt(monkeypatch):
    original = combined.least_squares
    calls = []

    def observed(fun, initial, **kwargs):
        solution = original(fun, initial, **kwargs)
        calls.append(solution)
        return solution

    monkeypatch.setattr(combined, "least_squares", observed)
    stack = _stack()
    result = run_interface_defect_ion_device_transient(
        _interface_grid(stack), stack, [0.0, 1.0e-8], [0.0, 0.005]
    )

    assert result.certificate.certified
    # One charge-off solve and one already acceptable warm operating point.
    assert len(calls) == 2
    assert result.dark_reference.dc_state.certificate.optimizer_nfev == calls[0].nfev
    assert result.dc_state.certificate.optimizer_nfev == calls[1].nfev


def test_stalled_warm_start_retries_with_remaining_budget(monkeypatch, r1_case):
    original = combined.least_squares
    calls = []
    budget = 40
    stalled_nfev = 3

    def observed(fun, initial, **kwargs):
        call = {"initial": initial.copy(), "budget": kwargs["max_nfev"]}
        calls.append(call)
        if len(calls) == 2:
            solution = _stalled_result(fun, initial, nfev=stalled_nfev)
            assert np.max(np.abs(solution.fun)) > 1.0e-8
        else:
            solution = original(fun, initial, **kwargs)
        call["solution"] = solution
        return solution

    monkeypatch.setattr(combined, "least_squares", observed)
    result = _run_r1(
        r1_case, policy=InterfaceDefectIonTransientPolicy(dc_max_nfev=budget)
    )

    assert result.certificate.certified
    assert len(calls) == 3
    assert calls[1]["budget"] == budget
    assert calls[2]["budget"] == budget - stalled_nfev
    np.testing.assert_array_equal(calls[2]["initial"], 0.0)
    assert result.dc_state.certificate.optimizer_nfev == (
        stalled_nfev + calls[2]["solution"].nfev
    )
    assert result.dc_state.certificate.optimizer_nfev <= budget
    assert result.dc_state.certificate.maximum_normalized_residual <= 1.0e-8


@pytest.mark.parametrize("exhausted", [True, False])
def test_stalled_recovery_fails_closed_and_accounts_for_attempts(
    monkeypatch, r1_case, exhausted
):
    original = combined.least_squares
    budgets = []
    budget = 10
    stalled_nfev = budget if exhausted else 3
    retry_nfev = 2
    initial_residual = None

    def stalled(fun, initial, **kwargs):
        nonlocal initial_residual
        budgets.append(kwargs["max_nfev"])
        if len(budgets) == 1:
            return original(fun, initial, **kwargs)
        if len(budgets) == 2:
            solution = _stalled_result(fun, initial, nfev=stalled_nfev)
            initial_residual = float(np.max(np.abs(solution.fun)))
            assert initial_residual > 1.0e-8
            return solution
        assert len(budgets) == 3, "recovery must be bounded to one retry"
        np.testing.assert_array_equal(initial, 0.0)
        return _stalled_result(fun, initial, nfev=retry_nfev, success=False)

    monkeypatch.setattr(combined, "least_squares", stalled)
    with pytest.raises(
        InterfaceDefectIonTransientError, match="operating point is not certified"
    ) as error:
        _run_r1(r1_case, policy=InterfaceDefectIonTransientPolicy(dc_max_nfev=budget))

    certificate = error.value.result.certificate
    assert not certificate.certified
    assert "joint_dc_residual_exceeds_limit" in certificate.reasons
    assert certificate.maximum_normalized_residual == initial_residual
    assert budgets == ([budget, budget] if exhausted else [budget, budget, budget - 3])
    assert certificate.optimizer_nfev == stalled_nfev + (
        0 if exhausted else retry_nfev
    )
    assert certificate.optimizer_nfev <= budget
