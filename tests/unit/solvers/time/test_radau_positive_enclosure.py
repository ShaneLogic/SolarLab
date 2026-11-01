"""Branch and positive-tube proofs on explicit polynomials; no time solves.

Integration conditions: optional, unqualified reference infrastructure for
restricted declared E/I inputs and analytic unit fixtures only. Independence
is static-only: no BDF stepping, history, interpolation or accounting is used.
Retained failed attempts stay failures and qualification flags stay false.
Any future nonzero initial uncertainty must be expressed in the selected
weighted log norm; a numerical trajectory alone does not qualify a reference.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from fractions import Fraction
from pathlib import Path
import runpy
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

pytest.importorskip("flint")
from flint import arb, ctx, fmpq

from solarlab.solvers.time.radau_enclosure import (
    NonlinearInventoryEnclosure,
    PolynomialPath,
    bernoulli_branch_enclosure,
    positive_polynomial_tube,
)
import solarlab.solvers.time.radau_enclosure as source


@pytest.fixture(scope="module", autouse=True)
def one_thread() -> Iterator[None]:
    before = ctx.threads
    ctx.threads = 1
    with threadpool_limits(limits=1), ctx.workprec(192):
        yield
    ctx.threads = before


def q(value: float | int | Fraction) -> Any:
    f = Fraction(value)
    return arb(fmpq(f.numerator, f.denominator))


@pytest.mark.parametrize("historical", [False, True])
def test_bernoulli_switch_encloses_both_retained_branches(historical: bool) -> None:
    limit = q(1e-8)
    whole = (-2 * limit).union(2 * limit)
    envelope = bernoulli_branch_enclosure(whole, historical=historical)
    assert envelope.retained_error_upper > 0
    assert envelope.retained_error_upper < q(1e-32)
    for multiplier in (-2, -1, -0.5, 0, 0.5, 1, 2):
        x = limit * q(multiplier)
        original = source._bernoulli(source._Jet(x), historical=historical).value
        assert envelope.retained_value.contains(original)
    assert envelope.analytic_derivative.contains(q(Fraction(-1, 2)))


def test_small_branch_error_is_nonzero_and_included() -> None:
    x = q(1e-9)
    envelope = bernoulli_branch_enclosure(x)
    quadratic = 1 - x / 2 + x * x / 12
    difference = abs(quadratic - envelope.analytic_value)
    assert difference > 0
    assert envelope.retained_error_upper >= difference.upper()
    assert envelope.retained_value.contains(quadratic)


def test_historical_700_cutoff_is_included() -> None:
    envelope = bernoulli_branch_enclosure(arb(699).union(arb(701)), historical=True)
    assert envelope.retained_value.contains(0)
    for x in (699, 700, 701):
        actual = source._bernoulli(source._Jet(arb(x)), historical=True).value
        assert envelope.retained_value.contains(actual)
    assert envelope.retained_error_upper > 0


def test_wide_zero_crossing_analytic_envelope() -> None:
    value = bernoulli_branch_enclosure(arb(-5).union(arb(5)))
    assert value.analytic_value.contains(1)
    assert value.analytic_derivative.contains(q(Fraction(-1, 2)))
    # Arb's union may round the radius outward; test containment, not inward
    # equality with the exact mathematical range [-1, 0].
    assert value.analytic_derivative.contains(-1)
    assert value.analytic_derivative.contains(0)
    for x in (-4, 4):
        assert value.analytic_value.contains(arb(x) / arb(x).expm1())
        slope = (arb(x).expm1() - x * arb(x).exp()) / arb(x).expm1() ** 2
        assert value.analytic_derivative.contains(slope)


class Oracle:
    branch_aware = True
    size = 1
    scale = np.ones(1)

    def __init__(self, *, constant: float | None = None, forcing_error: float = 0.0):
        self.constant = constant
        self.forcing_error = forcing_error
        self.evaluations = 0
        self.positive = True

    def evaluate(
        self,
        left: Fraction,
        right: Fraction,
        inventories: tuple[Any, ...],
        *,
        derivatives: bool = True,
    ) -> Any:
        self.evaluations += 1
        z = inventories[0]
        self.positive = self.positive and bool(z > 0)
        f = -z * z if self.constant is None else q(self.constant)
        jac = -2 * z if self.constant is None else arb(0)
        return SimpleNamespace(
            rhs=(f,),
            scaled_jacobian=((jac,),) if derivatives else ((),),
            coordinates=(z, z),
            densities=(z,),
            kernel_error_bound=(q(self.forcing_error),),
        )


def path(coefficients: list[float], end: float) -> PolynomialPath:
    high = np.asarray(coefficients).reshape(1, 4, 1)
    endpoint = float(sum((Fraction(x) for x in coefficients), Fraction(0)))
    return PolynomialPath(
        np.array([[0.0, end]]),
        high,
        np.zeros_like(high),
        np.array([coefficients[0]]),
        np.zeros(1),
        np.array([coefficients[0]] * 2),
        np.array([[endpoint] * 2]),
        np.ones(1),
    )


def test_positive_tube_keeps_tiny_carriers_without_clipping() -> None:
    tiny = 1e-30
    duration = 0.125
    data = path([tiny, tiny * duration, 0.0, 0.0], duration)
    field = Oracle(constant=tiny)
    result = positive_polynomial_tube(
        cast(NonlinearInventoryEnclosure, field),
        data,
        initial_log_error=0.0,
        log_radius=1e-6,
        subdivisions=2,
    )
    assert result.tube_proved and result.error_upper == 0.0
    assert result.log_error_upper == 0.0 and field.positive
    assert not result.reference_qualified


def test_explicit_polynomial_logistic_bound_against_exact_solution(
    record_property: Callable[[str, object], None],
) -> None:
    T = 1 / 16
    data = path([1.0, -T, T * T, -(T**3)], T)
    result = positive_polynomial_tube(
        cast(NonlinearInventoryEnclosure, Oracle()),
        data,
        initial_log_error=0.0,
        log_radius=0.02,
        subdivisions=2,
    )
    assert result.tube_proved and result.error_upper is not None
    observed = Fraction(0)
    for k in range(65):
        t = Fraction(k, 64) * Fraction(T)
        p = 1 - t + t * t - t**3
        observed = max(observed, abs(p - 1 / (1 + t)))
    assert Fraction(result.error_upper) >= observed
    record_property("polynomial_error_bound", result.error_upper)
    record_property("observed_exact_polynomial_error", float(observed))
    record_property("log_error_bound", result.log_error_upper)
    failed = positive_polynomial_tube(
        cast(NonlinearInventoryEnclosure, Oracle()),
        data,
        initial_log_error=0.0,
        log_radius=1e-12,
        subdivisions=2,
    )
    assert not failed.tube_proved and failed.error_upper is None


def test_retained_kernel_gap_is_propagated_as_positive_forcing() -> None:
    duration = 0.125
    epsilon = 0.001
    data = path([1.0, 0.0, 0.0, 0.0], duration)
    field = Oracle(constant=0.0, forcing_error=epsilon)
    result = positive_polynomial_tube(
        cast(NonlinearInventoryEnclosure, field),
        data,
        initial_log_error=0.0,
        log_radius=0.01,
        subdivisions=2,
    )
    assert result.tube_proved and result.error_upper is not None
    assert Fraction(result.error_upper) >= Fraction(epsilon) * Fraction(duration)
    assert all(
        row["retained_kernel_perturbation_upper"] >= epsilon for row in result.slabs
    )


def test_nonpositive_center_and_missing_kernel_error_are_refused() -> None:
    data = path([1.0, -2.0, 0.0, 0.0], 1.0)
    result = positive_polynomial_tube(
        cast(NonlinearInventoryEnclosure, Oracle(constant=-2.0)),
        data,
        initial_log_error=0.0,
        log_radius=0.01,
    )
    assert (
        not result.tube_proved
        and result.reason == "saved_polynomial_range_not_positive"
    )
    field = Oracle(constant=0.0, forcing_error=-1.0)
    result = positive_polynomial_tube(
        cast(NonlinearInventoryEnclosure, field),
        path([1.0, 0.0, 0.0, 0.0], 0.125),
        initial_log_error=0.0,
        log_radius=0.01,
    )
    assert (
        not result.tube_proved
        and result.reason == "missing_or_invalid_retained_kernel_error"
    )


def test_saved_reader_binds_inventories_and_retains_reconstructed_coordinates(
    tmp_path: Path,
) -> None:
    reader = runpy.run_path(str(Path(__file__).with_name("radau_enclose_saved.py")))
    grid = np.linspace(0.0, 1.0, 9)
    initial_u = np.array([1.0, np.nextafter(1.0, 2.0)])
    high = np.tile(np.array([1.0, 0.0, 0.0, 0.0])[None, :, None], (8, 1, 1))
    history = tmp_path / "saved.npz"
    np.savez_compressed(
        history,
        step_times=np.stack((grid[:-1], grid[1:]), axis=1),
        coefficients_high=high,
        coefficients_low=np.zeros_like(high),
        initial_inventory_high=np.ones(1),
        initial_inventory_low=np.zeros(1),
        initial_u=initial_u,
        endpoint_u=np.tile(initial_u, (8, 1)),
        component_scale=np.ones(1),
    )
    case = {
        "initial_inventory_high": [1.0],
        "initial_inventory_low": [0.0],
        "component_scale": [1.0],
        "initial": {"u": [1.0, 1.0]},
        "short_window_s": [0.0, 1.0],
    }
    data = reader["load_path"](history, case)
    assert data.initial_u.tobytes() == initial_u.tobytes()
    assert data.initial_high.tobytes() == np.ones(1).tobytes()
    case["initial_inventory_high"] = [np.nextafter(1.0, 2.0)]
    with pytest.raises(ValueError, match="saved_history_packet_mismatch"):
        reader["load_path"](history, case)
    with pytest.raises(ValueError, match="saved_source_hash_mismatch"):
        reader["require_hash"](history, "0" * 64)
