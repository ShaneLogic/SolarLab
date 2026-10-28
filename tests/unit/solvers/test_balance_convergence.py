"""Opt-in carrier-balance convergence with unchanged merit and final gates."""

from __future__ import annotations

from dataclasses import asdict
import math

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from examples.research_1d.steady import load_device
from solarlab.config.resolve_device import resolve_device
from solarlab.device.compile_1d import compile_device_1d
from solarlab.device.inputs import DeviceInput
from solarlab.solvers.nonlinear.newton import NewtonSettings, Termination, gate_from, solve_newton
from solarlab.solvers.nonlinear.problem import CallableProblem, ResidualScaling, SolveBudget, StepControl
from solarlab.solvers.nonlinear.production import (
    HISTORICAL_MAX_ROW_V1, POISSON_CONTACT_BALANCE_V1,
    production_convergence_norm, production_residual_scaling,
)
from solarlab.solvers.nonlinear.steady_1d import SteadyOptions


@pytest.fixture(scope="module", autouse=True)
def blas1():
    with threadpool_limits(limits=1):
        yield


@pytest.fixture(scope="module")
def compiled():
    return compile_device_1d(load_device(), intervals_per_layer=12)


@pytest.fixture(scope="module", params=["robin", "selective"])
def finite_contacts(request):
    prepared = load_device()
    document = prepared.to_input().editing_data()
    if request.param == "robin":
        document["settings"]["flat_band_contacts"] = True
    else:
        document["contacts"] = [
            dict(id="left", side="left", layer="htl", S_n=0.0, S_p=200.0),
            dict(id="right", side="right", layer="etl", S_n=200.0, S_p=0.0),
        ]
    prepared = resolve_device(
        DeviceInput.model_validate(document), prepared.defaults,
        prepared.resources, sources=prepared.sources,
    )
    return compile_device_1d(prepared, intervals_per_layer=12)


def balance_norm(device):
    return production_convergence_norm(device, convergence_policy=POISSON_CONTACT_BALANCE_V1)


def fixed_rows(device):
    count = device.node_count
    rows = []
    for carrier, offset in (("n", count), ("p", 2 * count)):
        for side, node in (("left", 0), ("right", count - 1)):
            if device.channel(carrier, side).kind == "fixed_reservoir":
                rows.append(offset + node)
    return rows


def test_default_policy_and_every_existing_limit_are_preserved(compiled):
    original = SteadyOptions()
    explicit = SteadyOptions(convergence_policy=POISSON_CONTACT_BALANCE_V1)
    assert original.convergence_policy == HISTORICAL_MAX_ROW_V1
    assert production_convergence_norm(compiled) is None
    old_controls, new_controls = asdict(original), asdict(explicit)
    assert old_controls.pop("convergence_policy") == HISTORICAL_MAX_ROW_V1
    assert new_controls.pop("convergence_policy") == POISSON_CONTACT_BALANCE_V1
    assert old_controls == new_controls
    assert original.budget() == explicit.budget()


@pytest.mark.parametrize("row_kind", ["poisson_contact", "poisson_interior", "n_left", "n_right", "p_left", "p_right"])
def test_poisson_and_fixed_contact_rows_are_never_diluted(compiled, row_kind):
    count = compiled.node_count
    row = {
        "poisson_contact": 0, "poisson_interior": count // 2,
        "n_left": count, "n_right": 2 * count - 1,
        "p_left": 2 * count, "p_right": 3 * count - 1,
    }[row_kind]
    residual = np.zeros(3 * count)
    residual[row] = 1.01
    assert balance_norm(compiled)(residual) == 1.01
    assert balance_norm(compiled)(residual) > 1.0


def test_both_carriers_share_one_absolute_balance_budget_without_cancellation(compiled):
    count = compiled.node_count
    residual = np.zeros(3 * count)
    residual[count:2 * count] = 1.1 * 2 * count / (2 * count - len(fixed_rows(compiled)))
    residual[2 * count:] = -residual[count:2 * count]
    residual[fixed_rows(compiled)] = 0.0
    assert np.sum(residual) == pytest.approx(0.0, abs=1e-13)
    measured = balance_norm(compiled)(residual)
    assert measured == pytest.approx(1.1, rel=2e-15)
    assert measured > 1.0
    # Each species separately fits one tolerance, but their joint total fails.
    assert np.sum(abs(residual[count:2 * count])) / (2 * count) < 1.0
    assert np.sum(abs(residual[2 * count:])) / (2 * count) < 1.0


def test_fixed_algebraic_contacts_are_excluded_only_from_the_charge_sum(compiled):
    count = compiled.node_count
    residual = np.zeros(3 * count)
    residual[count:] = 0.99 * 2 * count / (2 * count - len(fixed_rows(compiled)))
    residual[fixed_rows(compiled)] = 0.75
    measured = balance_norm(compiled)(residual)
    assert measured == pytest.approx(0.99, rel=2e-15)
    assert measured < 1.0
    assert np.sum(abs(residual[count:])) / (2 * count) > 1.0


@pytest.mark.parametrize("carrier, side", [("n", "left"), ("n", "right"), ("p", "left"), ("p", "right")])
def test_finite_and_blocking_boundary_cells_remain_in_balance_sum(finite_contacts, carrier, side):
    device = finite_contacts
    count = device.node_count
    assert device.channel(carrier, side).kind != "fixed_reservoir"
    row = (count if carrier == "n" else 2 * count) + (0 if side == "left" else count - 1)
    raw = np.zeros(3 * count)
    raw[row] = 1.01 * SteadyOptions().continuity_tolerance_A_m2 / 1.602176634e-19
    scaled = production_residual_scaling(device).scaled(raw)
    measured = balance_norm(device)(scaled)
    assert measured == pytest.approx(1.01, rel=2e-15)
    assert measured > 1.0


def test_single_overbudget_row_can_meet_the_global_physical_budget(compiled):
    count = compiled.node_count
    residual = np.zeros(3 * count)
    residual[count + count // 2] = 1.25
    assert np.max(abs(residual)) > 1.0
    assert balance_norm(compiled)(residual) == 1.25 / (2 * count)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_nonfinite_rows_fail_closed_in_every_block(compiled, value):
    for row in (0, compiled.node_count, compiled.node_count + 2, 2 * compiled.node_count + 2):
        residual = np.zeros(3 * compiled.node_count)
        residual[row] = value
        assert balance_norm(compiled)(residual) == math.inf


def controls(*, gates=()):
    return NewtonSettings(
        ResidualScaling(np.array([1.0, 3.0])), 1e-12,
        StepControl(np.full(2, math.inf)), SolveBudget(60, 10000, 1000, 30),
        tuple(gates), True,
    )


def coupled_problem(calls):
    root = np.array([0.3, -1.2])
    matrix = np.array([[4.0, 1.0], [1.0, 3.0]])

    def residual(u):
        calls.append(u.copy())
        delta = u - root
        return matrix @ delta + 0.5 * delta**2 + np.sin(delta)

    def jacobian(u):
        delta = u - root
        return matrix + np.diag(delta + np.cos(delta))

    return CallableProblem(residual, jacobian)


def test_explicit_norm_cannot_mutate_merit_or_default_newton_trace():
    original_calls, callback_calls = [], []
    start = np.array([1.8, -2.2])
    original = solve_newton(coupled_problem(original_calls), start, controls())

    def norm(values):
        measured = float(np.max(abs(values)))
        values[:] = math.nan
        return measured

    explicit = solve_newton(coupled_problem(callback_calls), start, controls(), convergence_norm=norm)
    assert original.accepted and explicit.accepted
    assert original.iterations == explicit.iterations
    assert original.residual_evaluations == explicit.residual_evaluations
    assert original.jacobian_evaluations == explicit.jacobian_evaluations
    assert original.state.tobytes() == explicit.state.tobytes()
    assert len(original_calls) == len(callback_calls)
    assert all(a.tobytes() == b.tobytes() for a, b in zip(original_calls, callback_calls))


def test_different_stopping_norm_preserves_armijo_trials_and_row_max_refresh():
    original_calls, callback_calls = [], []
    start = np.array([1.8, -2.2])
    original = solve_newton(coupled_problem(original_calls), start, controls())
    explicit = solve_newton(
        coupled_problem(callback_calls), start, controls(),
        convergence_norm=lambda values: float(np.sum(abs(values))),
    )
    assert original.accepted and explicit.accepted
    assert len(callback_calls) >= len(original_calls)
    assert all(a.tobytes() == b.tobytes() for a, b in zip(original_calls, callback_calls))
    raw = coupled_problem([]).residual(explicit.state)
    assert explicit.scaled_residual == float(np.sum(abs(controls().scaling.scaled(raw))))


def test_final_physical_gate_cannot_be_bypassed_by_a_norm_callback():
    gate = gate_from("wrong_branch", lambda u: 2.0, 1.0)
    result = solve_newton(
        coupled_problem([]), np.array([1.8, -2.2]), controls(gates=(gate,)),
        convergence_norm=lambda values: 0.0,
    )
    assert not result.accepted
    assert result.termination == Termination.GATE_REJECTED
    assert not result.gates[0].passed


@pytest.mark.parametrize("value", [math.nan, math.inf, -1.0])
def test_invalid_norm_callback_outputs_cannot_accept(value):
    with pytest.raises(ValueError, match="finite nonnegative"):
        solve_newton(coupled_problem([]), np.zeros(2), controls(), convergence_norm=lambda values: value)


def test_unknown_policy_is_not_an_implicit_new_route(compiled):
    with pytest.raises(ValueError, match="unknown"):
        SteadyOptions(convergence_policy="unknown")
    with pytest.raises(ValueError, match="unknown"):
        production_convergence_norm(compiled, convergence_policy="unknown")
