"""Actual DC brackets, direct V_oc states, and bounded recovery evidence."""

from __future__ import annotations

from copy import copy
from dataclasses import asdict, replace
import math

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from examples.research_1d.steady import load_device
from solarlab.config.resolve_device import resolve_device
from solarlab.device.compile_1d import compile_device_1d
from solarlab.device.inputs import DeviceInput
from solarlab.equations.dc_1d import DC1DProblem
from solarlab.experiments.jv import operating_points as points
from solarlab.experiments.jv.steady import run_steady_jv
from solarlab.solvers.nonlinear.newton import NewtonResult, Termination
from solarlab.solvers.nonlinear.problem import state_digest
from solarlab.solvers.nonlinear.production import HISTORICAL_MAX_ROW_V1, POISSON_CONTACT_BALANCE_V1
from solarlab.solvers.nonlinear.steady_1d import (
    SteadyOptions,
    SteadyStateError,
    solve_steady_state,
)


@pytest.fixture(scope="module", autouse=True)
def blas1():
    with threadpool_limits(limits=1):
        yield


@pytest.fixture(scope="module", params=[12, 24], ids=["37_nodes", "73_nodes"])
def actual_bracket(request):
    compiled = compile_device_1d(load_device(), intervals_per_layer=request.param)
    # The actual known sign interval; only two operating points are requested.
    # Smaller preparation steps change no final residual/current acceptance gate.
    controls = SteadyOptions(max_bias_step_V=0.05)
    bracket = run_steady_jv(compiled, [1.2, 1.225], options=controls)
    assert bracket.output_current_A_m2[0] > 0.0
    assert bracket.output_current_A_m2[1] < 0.0
    return compiled, bracket, controls


@pytest.fixture(scope="module")
def direct_result(actual_bracket):
    compiled, bracket, _ = actual_bracket
    return points.solve_direct_voc(compiled, bracket)


def test_real_direct_voc_is_a_solved_point_in_an_actual_bracket(actual_bracket, direct_result):
    compiled, bracket, controls = actual_bracket
    result = direct_result
    assert result.compiled_identity == compiled.identity
    assert result.voltage_polarity == compiled.voltage_polarity
    assert result.solver_controls == asdict(controls)
    assert result.interval_width_V <= 1e-6
    assert result.lower.voltage_V <= result.voltage_V <= result.upper.voltage_V
    assert result.initial_endpoints[0] is bracket.solutions[0]
    assert result.initial_endpoints[1] is bracket.solutions[1]
    assert any(result.solution is point for point in result.solutions)
    assert 0 < result.solve_calls <= 64
    for point in result.solutions:
        assert point.converged and point.state.device_sha256 == compiled.identity
        assert all(gate.passed for gate in point.physical_gates)
    for endpoint in (result.lower, result.upper):
        measured = DC1DProblem(compiled, endpoint.voltage_V, bracket.generation_scale).current_density(endpoint.state.u)
        np.testing.assert_array_equal(measured, endpoint.current_density_A_m2)
    left = -compiled.voltage_polarity * result.lower.current_density_A_m2[0]
    right = -compiled.voltage_polarity * result.upper.current_density_A_m2[0]
    assert left == 0.0 or right == 0.0 or np.sign(left) != np.sign(right)
    assert result.output_current_A_m2 == -compiled.voltage_polarity * result.solution.current_density_A_m2[0]


def test_independent_finer_solved_endpoints_enclose_the_same_root(actual_bracket, direct_result):
    compiled, bracket, controls = actual_bracket
    lower, upper = direct_result.lower, direct_result.upper
    left_sign = np.sign(-compiled.voltage_polarity * lower.current_density_A_m2[0])
    while upper.voltage_V - lower.voltage_V > 1e-7:
        voltage = (lower.voltage_V + upper.voltage_V) / 2
        seed = min((lower, upper), key=lambda point: abs(point.voltage_V - voltage))
        point = solve_steady_state(
            compiled, initial_state=seed.state, voltage_V=voltage,
            generation_scale=bracket.generation_scale, options=controls,
        )
        sign = np.sign(-compiled.voltage_polarity * point.current_density_A_m2[0])
        if sign == 0:
            lower = upper = point
        elif sign == left_sign:
            lower = point
        else:
            upper = point
    assert direct_result.lower.voltage_V <= lower.voltage_V
    assert upper.voltage_V <= direct_result.upper.voltage_V
    assert abs(direct_result.voltage_V - (lower.voltage_V + upper.voltage_V) / 2) <= direct_result.interval_width_V
    assert all(gate.passed for endpoint in (lower, upper) for gate in endpoint.physical_gates)


def test_descending_actual_bracket_has_the_same_direct_root(actual_bracket, direct_result):
    compiled, bracket, _ = actual_bracket
    reversed_bracket = replace(
        bracket, voltage_V=bracket.voltage_V[::-1],
        current_density_A_m2=bracket.current_density_A_m2[::-1],
        output_current_A_m2=bracket.output_current_A_m2[::-1],
        solutions=bracket.solutions[::-1],
    )
    result = points.solve_direct_voc(compiled, reversed_bracket)
    assert result.interval_V == direct_result.interval_V
    assert result.voltage_V == direct_result.voltage_V


def test_forged_current_identity_or_controls_are_not_an_actual_bracket(actual_bracket):
    compiled, bracket, controls = actual_bracket
    forged = replace(bracket, output_current_A_m2=np.array([1.0, -1.0]))
    with pytest.raises(points.VocBracketError, match="actual solved outputs"):
        points.solve_direct_voc(compiled, forged)
    with pytest.raises(points.VocBracketError, match="same compiled device"):
        points.solve_direct_voc(compiled, replace(bracket, compiled_identity="0" * 64))
    with pytest.raises(points.VocBracketError, match="retain"):
        points.solve_direct_voc(compiled, bracket, options=replace(controls, max_iterations=2))


def test_solve_budget_failure_preserves_accepted_endpoints(actual_bracket):
    compiled, bracket, _ = actual_bracket
    with pytest.raises(points.DirectVocError, match="solve-call budget") as error:
        points.solve_direct_voc(compiled, bracket, max_solve_calls=1)
    result = error.value
    assert result.solve_calls == 1
    assert len(result.solutions) == len(bracket.solutions) + 1
    assert result.lower.converged and result.upper.converged
    assert not result.failed_attempts


def _injected_failure(seed):
    # A deliberately unusable failed state makes any unsafe seed reuse visible.
    failed = NewtonResult(
        np.full(seed.u.shape, np.nan), False, Termination.LINE_SEARCH_FAILED,
        math.inf, (), (), 1, 1, state_digest(seed.u), "injected control-path failure",
    )
    return SteadyStateError("test step", failed)


def test_failed_midpoint_uses_finer_same_equation_and_never_failed_state(actual_bracket, monkeypatch):
    compiled, bracket, _ = actual_bracket
    target = (bracket.voltage_V[0] + bracket.voltage_V[1]) / 2
    large_distance = 0.4 * float(np.ptp(bracket.voltage_V))
    genuine = points.solve_steady_state
    attempts = []

    def controlled(device, *, initial_state, voltage_V, generation_scale, options):
        assert np.all(np.isfinite(initial_state.u))
        attempts.append((voltage_V, initial_state))
        if voltage_V == target and abs(initial_state.voltage_V - target) > large_distance:
            raise _injected_failure(initial_state)
        return genuine(
            device, initial_state=initial_state, voltage_V=voltage_V,
            generation_scale=generation_scale, options=options,
        )

    monkeypatch.setattr(points, "solve_steady_state", controlled)
    result = points.solve_direct_voc(compiled, bracket)
    assert result.interval_width_V <= 1e-6
    assert result.solve_calls == len(attempts)
    assert result.failed_attempts
    failure = result.failed_attempts[0]
    assert failure.voltage_V == target
    assert failure.subdivisions == 0
    assert not failure.newton_result.accepted
    assert np.all(np.isnan(failure.newton_result.state))
    assert all(any(seed is point.state for point in result.solutions) for _, seed in attempts)
    # Each seed is the nearest previously accepted physical point, not a failed iterate.
    assert abs(attempts[1][1].voltage_V - attempts[1][0]) < abs(failure.seed_voltage_V - target)


def test_persistent_step_failures_are_bounded_and_all_recorded(actual_bracket, monkeypatch):
    compiled, bracket, _ = actual_bracket
    attempts = []

    def failing(device, *, initial_state, voltage_V, generation_scale, options):
        assert np.all(np.isfinite(initial_state.u))
        attempts.append(voltage_V)
        raise _injected_failure(initial_state)

    monkeypatch.setattr(points, "solve_steady_state", failing)
    with pytest.raises(points.DirectVocError, match="budget") as error:
        points.solve_direct_voc(compiled, bracket, max_solve_calls=3)
    assert error.value.solve_calls == len(attempts) == 3
    assert len(error.value.failed_attempts) == 3
    assert len(error.value.solutions) == len(bracket.solutions)
    assert all(not failure.newton_result.accepted for failure in error.value.failed_attempts)


@pytest.mark.parametrize("current, expected", [
    ([1.0, -1.0], (0, 1)), ([-1.0, 1.0], (0, 1)),
    ([1.0, 0.0, -1.0], (1, 1)),
])
def test_unique_root_selection_policy(current, expected):
    assert points._unique_root_indices(np.array(current)) == expected


@pytest.mark.parametrize("current", [[1.0, 2.0], [1.0, -1.0, 1.0], [1.0, 0.0, 0.0, -1.0]])
def test_missing_or_multiple_roots_are_explicit_scope(current):
    with pytest.raises(points.VocBracketError):
        points._unique_root_indices(np.array(current))


@pytest.mark.parametrize("controls", [
    {"voltage_tolerance_V": 0.0}, {"voltage_tolerance_V": math.nan},
    {"max_solve_calls": 0}, {"max_continuation_subdivisions": 1},
])
def test_invalid_root_controls_are_rejected(actual_bracket, controls):
    compiled, bracket, _ = actual_bracket
    with pytest.raises(ValueError):
        points.solve_direct_voc(compiled, bracket, **controls)


def test_reversed_junction_retains_the_terminal_current_polarity():
    prepared = load_device()
    data = prepared.to_input().editing_data()
    data["layers"] = list(reversed(data["layers"]))
    reverse = resolve_device(DeviceInput.model_validate(data), prepared.defaults, prepared.resources, sources=prepared.sources)
    compiled = compile_device_1d(reverse, intervals_per_layer=12)
    bracket = run_steady_jv(compiled, [1.2, 1.225], options=SteadyOptions(max_bias_step_V=0.05))
    result = points.solve_direct_voc(compiled, bracket)
    assert compiled.voltage_polarity == -1.0
    assert result.interval_width_V <= 1e-6
    assert result.output_current_A_m2 == result.solution.current_density_A_m2[0]


def test_exact_solved_zero_is_an_endpoint_without_interpolated_state():
    compiled = compile_device_1d(load_device(), intervals_per_layer=12)
    bracket = run_steady_jv(compiled, [0.0, 0.05], generation_scale=0.0)
    result = points.solve_direct_voc(compiled, bracket)
    assert result.solution is bracket.solutions[0]
    assert result.voltage_V == result.interval_width_V == 0.0
    assert result.solve_calls == 0


def test_opt_in_balance_policy_is_recorded_and_retained_by_direct_voc():
    compiled = compile_device_1d(load_device(), intervals_per_layer=12)
    controls = SteadyOptions(
        max_bias_step_V=0.05, convergence_policy=POISSON_CONTACT_BALANCE_V1,
    )
    # A fresh cold solve prepares this bracket; no saved failed state is used.
    bracket = run_steady_jv(compiled, [1.2, 1.225], options=controls)
    assert bracket.solver_controls["convergence_policy"] == POISSON_CONTACT_BALANCE_V1
    result = points.solve_direct_voc(compiled, bracket)
    assert result.solver_controls == asdict(controls)
    assert result.interval_width_V <= 1e-6
    assert all(gate.passed for point in result.solutions for gate in point.physical_gates)
    for point in result.solutions:
        points._verify_point(compiled, point, bracket.generation_scale, controls)
    with pytest.raises(points.VocBracketError, match="retain"):
        points.solve_direct_voc(
            compiled, bracket, options=replace(controls, convergence_policy=HISTORICAL_MAX_ROW_V1),
        )


def test_old_control_records_default_to_historical_row_policy(actual_bracket):
    compiled, bracket, controls = actual_bracket
    old_record = dict(bracket.solver_controls)
    assert old_record.pop("convergence_policy") == HISTORICAL_MAX_ROW_V1
    restored = SteadyOptions(**old_record)
    assert restored == controls
    points._verify_point(compiled, bracket.solutions[0], bracket.generation_scale, restored)


@pytest.mark.parametrize("policy", [HISTORICAL_MAX_ROW_V1, POISSON_CONTACT_BALANCE_V1])
def test_changing_policy_never_resigns_a_failed_saved_point(actual_bracket, policy):
    compiled, bracket, controls = actual_bracket
    failed_point = copy(bracket.solutions[0])
    failure = _injected_failure(failed_point.state)
    # Represent a failed saved record without changing any accepted fixture.
    object.__setattr__(failed_point, "newton", failure.result)
    with pytest.raises(points.VocBracketError, match="accepted steady solve"):
        points._verify_point(
            compiled, failed_point, bracket.generation_scale,
            replace(controls, convergence_policy=policy),
        )
    assert not failure.result.accepted
    assert all(point.converged for point in bracket.solutions)


@pytest.mark.parametrize("policy", [HISTORICAL_MAX_ROW_V1, POISSON_CONTACT_BALANCE_V1])
def test_saved_points_recompute_gates_independently_of_stored_passes(actual_bracket, monkeypatch, policy):
    compiled, bracket, controls = actual_bracket
    point = bracket.solutions[0]
    genuine = points._physical_gates
    calls = []

    def guarded(problem, options, *, equilibrium):
        gates = genuine(problem, options, equilibrium=equilibrium)

        def rejected(u):
            calls.append(u)
            original = gates[0](u)
            return replace(original, passed=False, message="injected independent gate rejection")

        return (rejected, *gates[1:])

    monkeypatch.setattr(points, "_physical_gates", guarded)
    assert all(gate.passed for gate in point.physical_gates)
    with pytest.raises(points.VocBracketError, match="recomputed production physical gate"):
        points._verify_point(
            compiled, point, bracket.generation_scale, replace(controls, convergence_policy=policy),
        )
    assert calls and point.converged
