"""Auxiliary startup failures retain trial records and the original budgets."""

from dataclasses import replace

import numpy as np
import pytest

from solarlab.solvers.nonlinear.newton import NewtonSettings, Termination
from solarlab.solvers.nonlinear.problem import ResidualScaling, SolveBudget, StepControl
from solarlab.solvers.time import bdf


class ConstantInventory:
    def residual(self, u):
        return np.zeros_like(u)

    def jacobian(self, u):
        return np.zeros((len(u), len(u)))

    def storage(self, u):
        return u.copy()

    def storage_tangent(self, u):
        return np.eye(len(u))


def settings(step):
    return NewtonSettings(ResidualScaling(np.ones(1)), 1e-12, StepControl(np.ones(1)),
                          SolveBudget(20, 100, 20, 10), (), False)


def inject_failure(monkeypatch, failed_call=None):
    original = bdf.solve_newton
    calls = []

    def wrapped(problem, seed, control, **kwargs):
        result = original(problem, seed, control, **kwargs)
        if len(calls) + 1 == failed_call or (failed_call is None and len(calls) % 2 == 1):
            result = replace(result, accepted=False, termination=Termination.LINE_SEARCH_FAILED,
                             state=np.array([999.0]), message="injected auxiliary nonlinear failure")
        calls.append(result)
        return result

    monkeypatch.setattr(bdf, "solve_newton", wrapped)
    return calls


@pytest.mark.parametrize("failed_call", [2, 3])
def test_each_half_step_failure_is_a_recorded_rejection_without_accepting_auxiliary_history(monkeypatch, failed_call):
    calls = inject_failure(monkeypatch, failed_call)
    problem = ConstantInventory()
    result = bdf.solve_bdf(lambda t, side: problem, np.ones(1), (0.0, 1.0),
        options=bdf.BDFOptions(.5), settings=settings)
    assert result.completed, (result.termination, result.error)
    rejected = result.rejected_steps
    assert len(rejected) == 1
    failure = rejected[0]
    assert failure.start_s == 0 and failure.end_s == .5
    assert failure.termination == "startup_newton_line_search_failed"
    assert len(failure.auxiliary_newton) == failed_call - 1
    assert failure.newton.accepted and not failure.auxiliary_newton[-1].accepted
    np.testing.assert_array_equal(failure.auxiliary_newton[-1].state, [999.0])
    assert result.accepted_steps[0].start_s == 0 and result.accepted_steps[0].end_s == .25
    for u in result.accepted_coordinates:
        np.testing.assert_array_equal(u, [1.0])
    recorded = [value for a in result.attempts for value in (a.newton, *a.auxiliary_newton) if value is not None]
    assert len(recorded) == result.nonlinear_calls == len(calls)
    assert result.evaluation_counts["residual"] == sum(n.residual_evaluations for n in recorded)
    assert result.evaluation_counts["jacobian"] == sum(n.jacobian_evaluations for n in recorded)


@pytest.mark.parametrize("max_attempts, consecutive, expected_trials, termination", [
    (20000, 2, 3, "rejection_budget_exhausted"),
    (1, 12, 1, "step_budget_exhausted"),
])
def test_repeated_auxiliary_failure_obeys_unchanged_trial_and_rejection_budgets(
    monkeypatch, max_attempts, consecutive, expected_trials, termination,
):
    calls = inject_failure(monkeypatch)
    problem = ConstantInventory()
    result = bdf.solve_bdf(lambda t, side: problem, np.ones(1), (0.0, 1.0),
        options=bdf.BDFOptions(.5, max_attempts=max_attempts, max_consecutive_rejections=consecutive),
        settings=settings)
    assert not result.completed and result.termination == termination
    assert len(result.attempts) == len(result.rejected_steps) == expected_trials
    assert result.nonlinear_calls == len(calls) == 2 * expected_trials
    np.testing.assert_array_equal(result.accepted_times_s, [0.0])
    assert all(a.start_s == 0 and a.termination == "startup_newton_line_search_failed" for a in result.attempts)
    assert result.controls["max_attempts"] == max_attempts
    assert result.controls["max_consecutive_rejections"] == consecutive
