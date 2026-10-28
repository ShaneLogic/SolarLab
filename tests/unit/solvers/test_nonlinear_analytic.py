"""Unified DC nonlinear layer on analytic problems (N class, V1 P04-04 'analytic basis').

Quadratic convergence, damping of overshooting steps, direction-preserving
step limits, feasible bounds, Jacobian reuse with rebuild, explicit failure
reasons and budgets, gate (wrong-branch) rejection, bounded continuation with
traceable starts, bounded PTC, domain inventories, frozen fields, the DC port
constraint and the fail-closed P04-B switch points.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.sparse import csc_matrix

from solarlab.discretization import bernoulli
from solarlab.solvers.nonlinear import production
from solarlab.solvers.nonlinear.constraints import (
    TRANSIENT_CONDUCTION_CURRENT, DcPortProblem, DomainInventory, FrozenVariables,
    InventoryConstrainedProblem, PortConstraintError, augmented_start,
)
from solarlab.solvers.nonlinear.continuation import (
    PREVIOUS, SECANT, ContinuationSettings, solve_continuation,
)
from solarlab.solvers.nonlinear.newton import NewtonSettings, Termination, gate_from, solve_newton
from solarlab.solvers.nonlinear.problem import (
    CallableProblem, ResidualScaling, SolveBudget, StepControl, state_digest,
)
from solarlab.solvers.nonlinear.ptc import PtcResult, PtcSettings, solve_ptc

EPS = float(np.finfo(float).eps)


def settings(size, *, tol=1e-12, max_step=math.inf, lower=None, upper=None, gates=(),
             reuse=False, iterations=60, backtracks=30, weights=None):
    return NewtonSettings(
        ResidualScaling(np.ones(size) if weights is None else weights), tol,
        StepControl(np.full(size, max_step), lower, upper),
        SolveBudget(iterations, 100_000, 1_000, backtracks), tuple(gates), reuse)


def problem(F, J):
    return CallableProblem(lambda u: np.atleast_1d(F(u)), lambda u: np.atleast_2d(J(u)))


def coupled(root):
    A = np.array([[4.0, 1.0, 0.0], [1.0, 3.0, 0.5], [0.0, 0.5, 2.0]])

    def F(u):
        d = u - root
        return A @ d + 0.5 * d ** 2 + np.sin(d)

    def J(u):
        d = u - root
        return A + np.diag(d + np.cos(d))
    return problem(F, J)


def test_newton_converges_quadratically_on_a_smooth_system():
    root = np.array([0.3, -1.2, 2.5])
    result = solve_newton(coupled(root), root + np.array([0.4, -0.3, 0.5]), settings(3))
    assert result.accepted and result.termination is Termination.CONVERGED
    np.testing.assert_allclose(result.state, root, rtol=0, atol=8 * EPS * 3)
    norms = [record.scaled_residual for record in result.iterations]
    pairs = [(a, b) for a, b in zip(norms[:-1], norms[1:]) if 1e-12 < a < 1e-2 and b > 0]
    assert pairs and all(b <= 20.0 * a * a for a, b in pairs)
    assert all(record.backtracks == 0 for record in result.iterations[1:])


def test_damping_controls_an_overshooting_newton_step():
    arctan = problem(np.arctan, lambda u: 1.0 / (1.0 + u * u))
    damped = solve_newton(arctan, np.array([10.0]), settings(1))
    assert damped.accepted and abs(damped.state[0]) < 1e-14
    assert damped.iterations[0].backtracks > 0
    undamped = solve_newton(arctan, np.array([10.0]), settings(1, backtracks=0))
    assert not undamped.accepted and undamped.termination is Termination.LINE_SEARCH_FAILED


def test_step_limit_shortens_the_whole_step_and_keeps_its_direction():
    target = np.array([3.0, 6.0])
    linear = problem(lambda u: u - target, lambda u: np.eye(2))
    result = solve_newton(linear, np.zeros(2), settings(2, max_step=1.0))
    assert result.accepted and result.iteration_count == 6
    np.testing.assert_allclose([r.damping for r in result.iterations],
                               [1 / 6, 1 / 5, 1 / 4, 1 / 3, 1 / 2, 1.0], rtol=1e-14)
    np.testing.assert_allclose(result.state, target, rtol=0, atol=8 * EPS)


def test_feasible_bounds_keep_every_trial_inside_the_domain():
    calls = []

    def F(u):
        calls.append(float(u[0]))
        return np.log(u) + 18.420680743952367      # root 1e-8, undefined for u <= 0

    result = solve_newton(problem(F, lambda u: 1.0 / u), np.array([1.0]),
                          settings(1, lower=np.zeros(1), iterations=200))
    assert result.accepted and result.state[0] == pytest.approx(1e-8, rel=1e-12)
    assert min(calls) > 0.0
    infeasible = solve_newton(problem(F, lambda u: 1.0 / u), np.array([-1.0]),
                              settings(1, lower=np.zeros(1)))
    assert infeasible.termination is Termination.INFEASIBLE_START


def test_jacobian_reuse_rebuilds_a_stale_factor_and_matches_full_newton():
    root = np.array([0.3, -1.2, 2.5])
    start = root + np.array([1.5, -1.0, 1.2])
    full = solve_newton(coupled(root), start, settings(3))
    reused = solve_newton(coupled(root), start, settings(3, reuse=True))
    assert full.accepted and reused.accepted
    # Both stop at |F| <= 1e-12; the states then agree within tol / sigma_min(J) < 1e-12.
    np.testing.assert_allclose(reused.state, full.state, rtol=0, atol=1e-12)
    assert reused.jacobian_evaluations < full.jacobian_evaluations
    assert reused.jacobian_evaluations > 1                    # a stale factor was rebuilt
    assert any(not record.fresh_jacobian for record in reused.iterations)


def test_failures_are_explicit_and_bounded():
    arctan = problem(np.arctan, lambda u: 1.0 / (1.0 + u * u))
    budget = solve_newton(arctan, np.array([10.0]), settings(1, iterations=2))
    assert budget.termination is Termination.BUDGET_EXHAUSTED and "budget" in budget.message
    singular = problem(lambda u: np.array([u[0] ** 2 - 4.0, u[0] - 2.0]),
                       lambda u: np.array([[2.0 * u[0], 0.0], [1.0, 0.0]]))
    result = solve_newton(singular, np.array([3.0, 1.0]), settings(2))
    assert result.termination is Termination.SINGULAR_JACOBIAN and not result.accepted
    nan_start = solve_newton(problem(lambda u: np.sqrt(u - 1.0), lambda u: 1.0),
                             np.array([0.0]), settings(1))
    assert nan_start.termination is Termination.NONFINITE_RESIDUAL


def test_nonfinite_trials_are_backtracked_not_accepted():
    def F(u):    # defined only below 5; the first full Newton step lands at 10
        return np.where(u < 5.0, u - 2.0 + 0.0 * np.sqrt(5.0 - u), np.nan)

    result = solve_newton(problem(F, lambda u: 1.0), np.array([-6.0]), settings(1))
    assert result.accepted and result.state[0] == pytest.approx(2.0, abs=1e-14)


def test_a_converged_residual_on_the_wrong_branch_is_rejected_by_its_gate():
    square = problem(lambda u: u * u - 4.0, lambda u: 2.0 * u)
    positive = gate_from("positive_root", lambda u: -float(u[0]), 0.0, "physical root is u>0")
    wrong = solve_newton(square, np.array([-3.0]), settings(1, gates=[positive]))
    assert wrong.termination is Termination.GATE_REJECTED and not wrong.accepted
    assert wrong.scaled_residual <= 1e-12 and wrong.state[0] == pytest.approx(-2.0)
    assert "positive_root" in wrong.message and not wrong.gates[0].passed
    right = solve_newton(square, np.array([3.0]), settings(1, gates=[positive]))
    assert right.accepted and right.gates[0].passed


def test_controls_have_no_defaults_and_validate_their_inputs():
    with pytest.raises(TypeError):
        NewtonSettings(ResidualScaling(np.ones(2)), 1e-10)          # type: ignore[call-arg]
    with pytest.raises(ValueError):
        ResidualScaling(np.array([1.0, 0.0]))
    with pytest.raises(ValueError):
        StepControl(np.ones(2), lower=np.zeros(3))
    with pytest.raises(ValueError):
        settings(2, tol=0.0)
    # Acceptance follows the caller's weights: a heavier row needs a smaller residual.
    linear = problem(lambda u: u - 1.0 + 1e-9, lambda u: np.eye(1))
    loose = solve_newton(linear, np.array([1.0]), settings(1, tol=1e-8, iterations=0 + 1))
    assert loose.accepted and loose.iteration_count == 0
    heavy = solve_newton(linear, np.array([1.0]), settings(1, tol=1e-8, weights=np.array([1e3])))
    assert heavy.accepted and heavy.iteration_count == 1


def test_sparse_jacobians_are_supported():
    root = np.array([0.3, -1.2, 2.5])
    dense = coupled(root)
    sparse = CallableProblem(dense.residual, lambda u: csc_matrix(dense.jacobian(u)))
    a = solve_newton(dense, root + 0.4, settings(3))
    b = solve_newton(sparse, root + 0.4, settings(3))
    assert a.accepted and b.accepted
    np.testing.assert_allclose(a.state, b.state, rtol=0, atol=8 * EPS * 3)


def arctan_family(s):
    return problem(lambda u: np.arctan(u - s), lambda u: 1.0 / (1.0 + (u - s) ** 2))


def undamped_newton(target_problem, guess):
    return solve_newton(target_problem, guess, settings(1, backtracks=0, iterations=40))


def continuation(max_step=4.0, min_step=0.05, cuts=6, attempts=40, predictor=PREVIOUS):
    return ContinuationSettings(max_step, min_step, cuts, attempts, predictor, 2.0)


def test_continuation_recovers_by_cutting_and_never_seeds_from_a_failure():
    result = solve_continuation(arctan_family, np.array([0.0]), 0.0, [4.0], undamped_newton,
                                continuation())
    assert result.completed and result.failure is None
    assert result.points[-1].parameter == 4.0 and result.points[-1].requested
    assert any(not attempt.accepted for attempt in result.attempts)
    accepted_digests = {state_digest(point.state) for point in result.points}
    for attempt in result.attempts[1:]:
        # Every attempt was built from an accepted state and (previous predictor)
        # started exactly there.
        assert attempt.base_digest in accepted_digests
        assert attempt.start_digest == attempt.base_digest
    for point in result.points:
        assert point.result.accepted


def test_continuation_failure_is_bounded_and_diagnosed():
    result = solve_continuation(arctan_family, np.array([0.0]), 0.0, [4.0], undamped_newton,
                                continuation(min_step=2.0, cuts=1))
    assert not result.completed and "consecutive cuts" in result.failure
    assert result.points[-1].parameter == 0.0
    assert all(attempt.termination is not Termination.CONVERGED
               for attempt in result.attempts[1:])
    budget = solve_continuation(arctan_family, np.array([0.0]), 0.0, [4.0], undamped_newton,
                                continuation(attempts=2))
    assert not budget.completed and "attempt budget" in budget.failure


def test_secant_predictor_only_uses_accepted_states():
    family = lambda s: problem(lambda u: u - s * s, lambda u: np.eye(1))  # noqa: E731
    result = solve_continuation(family, np.array([0.0]), 0.0, [0.5, 1.0, 1.5], undamped_newton,
                                continuation(max_step=0.5, predictor=SECANT))
    assert result.completed
    secant = [attempt for attempt in result.attempts if attempt.predictor == SECANT]
    assert secant and all(attempt.start_digest != attempt.base_digest for attempt in secant)
    np.testing.assert_allclose([p.state[0] for p in result.requested_points()],
                               [0.0, 0.25, 1.0, 2.25], rtol=0, atol=1e-14)


def test_staged_preparation_relaxes_only_intermediate_stages():
    family = lambda s: problem(lambda u: np.arctan(u - s) + 1e-11, lambda u: 1.0 / (  # noqa: E731
        1.0 + (u - s) ** 2))
    stages = [1e-6, 1e-3, 1.0]

    def newton(target, guess):
        return solve_newton(target, guess, settings(1, tol=1e-12, iterations=40))

    result = solve_continuation(family, np.array([0.0]), 0.0, stages, newton,
                                continuation(max_step=1.0))
    assert result.completed
    assert [point.parameter for point in result.requested_points()] == [0.0, *stages]
    assert result.points[-1].result.scaled_residual <= 1e-12


def cubic():
    return problem(lambda u: u ** 3 - 2.0 * u + 2.0, lambda u: 3.0 * u * u - 2.0)


def ptc_settings(steps=200, max_dtau=1e6):
    return PtcSettings(np.ones(1), 0.1, 1e-8, max_dtau, steps, 1e-3, ResidualScaling(np.ones(1)),
                       StepControl(np.full(1, math.inf)))


def test_ptc_reaches_a_steady_state_that_newton_misses_and_records_a_numerical_path():
    newton = solve_newton(cubic(), np.array([0.0]), settings(1, iterations=40))
    assert not newton.accepted          # the merit has a local minimum near u = 0.8
    result = solve_ptc(cubic(), np.array([0.0]), ptc_settings(),
                       lambda target, guess: solve_newton(target, guess, settings(1)))
    assert result.accepted and result.newton is not None and result.newton.accepted
    root = result.state[0]
    assert root ** 3 - 2.0 * root + 2.0 == pytest.approx(0.0, abs=1e-12)
    assert result.pseudo_history and result.physical_history is False
    with pytest.raises(ValueError, match="never a physical"):
        PtcResult(True, "converged", None, (), "", "", physical_history=True)
    bounded = solve_ptc(cubic(), np.array([0.0]), ptc_settings(steps=3, max_dtau=0.1),
                        lambda target, guess: solve_newton(target, guess, settings(1)))
    assert not bounded.accepted and bounded.termination == "budget_exhausted"


def ion_problem(potential, blocking_faces, V_T=0.025852, k=1.0):
    """Steady single positive species in a fixed potential; u = ln c."""
    size = potential.size
    D = (potential[1:] - potential[:-1]) / V_T
    kf = np.where(np.isin(np.arange(size - 1), blocking_faces), 0.0, k)

    def flux(c):
        return kf * (bernoulli(D) * c[:-1] - bernoulli(-D) * c[1:])

    def residual(u):
        c = np.exp(u)
        F = flux(c)
        r = np.zeros(size)
        r[:-1] -= F
        r[1:] += F
        return r

    def jacobian(u):
        c = np.exp(u)
        J = np.zeros((size, size))
        left, right = kf * bernoulli(D) * c[:-1], -kf * bernoulli(-D) * c[1:]
        faces = np.arange(size - 1)
        for sign, cell in ((-1.0, faces), (1.0, faces + 1)):
            J[cell, faces] += sign * left
            J[cell, faces + 1] += sign * right
        return J
    return CallableProblem(residual, jacobian)


def test_domain_inventories_close_the_rank_deficient_ion_steady_state():
    x = np.linspace(0.0, 1.0, 13)
    potential = 0.1 * np.sin(3.0 * x) + 0.05 * x
    base = ion_problem(potential, blocking_faces=[6])     # face 6 splits two closed domains
    weights = np.full(13, 1.0 / 12)
    domains = (DomainInventory("left", np.arange(0, 7), weights[:7], 3.0e23, 3),
               DomainInventory("right", np.arange(7, 13), weights[7:], 1.0e23, 9))
    constrained = InventoryConstrainedProblem(base, domains, np.exp, np.exp)
    u0 = np.full(13, math.log(2.0e24))
    plain = solve_newton(base, u0, settings(13))
    assert not plain.accepted                    # singular: profile scale is undetermined
    result = solve_newton(constrained, u0, settings(13, weights=np.full(13, 1e-24)))
    assert result.accepted
    c = np.exp(result.state)
    for domain in domains:
        boltzmann = np.exp(-potential[domain.indices] / 0.025852)
        expected = domain.inventory * boltzmann / np.dot(domain.weights, boltzmann)
        np.testing.assert_allclose(c[domain.indices], expected, rtol=1e-12)
    for name, value in constrained.inventories(result.state).items():
        target = {"left": 3.0e23, "right": 1.0e23}[name]
        assert value == pytest.approx(target, rel=1e-13)
    with pytest.raises(ValueError, match="overlap"):
        InventoryConstrainedProblem(base, (domains[0], DomainInventory(
            "x", np.arange(5, 9), weights[5:9], 1e23, 6)), np.exp, np.exp)
    with pytest.raises(ValueError, match="belong to the domain"):
        DomainInventory("bad", np.arange(0, 3), weights[:3], 1e23, 5)


def test_frozen_variables_keep_the_complete_profile_bitwise():
    x = np.linspace(0.0, 1.0, 13)
    potential = 0.1 * np.sin(3.0 * x)
    base = ion_problem(potential, blocking_faces=[6])
    frozen_profile = np.log(np.geomspace(1e22, 5e23, 6))
    weights = np.full(13, 1.0 / 12)
    left = InventoryConstrainedProblem(
        base, (DomainInventory("left", np.arange(0, 7), weights[:7], 3.0e23, 3),),
        np.exp, np.exp)
    frozen = FrozenVariables(left, np.arange(7, 13), frozen_profile, 13)
    result = solve_newton(frozen, np.full(7, math.log(1e24)),
                          settings(7, weights=np.full(7, 1e-24)))
    assert result.accepted
    full = frozen.expand(result.state)
    assert full[7:].tobytes() == frozen_profile.tobytes()
    assert frozen.restrict(full).tobytes() == result.state.tobytes()


def diode(J0=1e-9, J_ph=200.0, V_T=0.025852):
    def problem_at(V):
        return CallableProblem(lambda u: u - V, lambda u: np.eye(1))

    def current(u, V):
        return J0 * math.expm1(u[0] / V_T) - J_ph

    return DcPortProblem(
        problem_at, lambda u, V: np.array([-1.0]), current,
        lambda u, V: np.array([J0 * math.exp(u[0] / V_T) / V_T]), lambda u, V: 0.0,
        current_scale=J_ph)


def test_direct_voc_closes_on_the_dc_terminal_current():
    port = diode()
    result = solve_newton(port, augmented_start(np.array([0.5]), 0.5),
                          settings(2, max_step=0.1, iterations=80))
    assert result.accepted
    expected = 0.025852 * math.log1p(200.0 / 1e-9)
    assert result.state[-1] == pytest.approx(expected, rel=1e-13)
    with pytest.raises(PortConstraintError, match="TPV"):
        DcPortProblem(port.problem_at, port.dresidual_dV, port.terminal_current,
                      port.terminal_current_gradient, port.terminal_current_dV, 1.0,
                      kind=TRANSIENT_CONDUCTION_CURRENT)
    with pytest.raises(PortConstraintError, match="unknown"):
        DcPortProblem(port.problem_at, port.dresidual_dV, port.terminal_current,
                      port.terminal_current_gradient, port.terminal_current_dV, 1.0,
                      kind="median_current")


@pytest.mark.parametrize("name", [
    "production_steady_problem",
    "production_steady_linearization",
    "production_residual_scaling",
    "production_coordinates",
])
def test_production_bindings_require_an_explicit_compiled_problem(name):
    # Positive physical bindings are exercised by integration/test_core_1d.py;
    # calling them without a compiled device/problem cannot select a fallback.
    with pytest.raises(TypeError, match="required"):
        getattr(production, name)()
