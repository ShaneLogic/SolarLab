"""Verified diagonal norms on explicit interval matrices; no time solves.

Integration conditions: optional, unqualified reference infrastructure for
restricted declared E/I inputs and analytic unit fixtures only. Independence
is static-only: no BDF stepping, history, interpolation or accounting is used.
Retained failed attempts stay failures and qualification flags stay false.
Any future nonzero initial uncertainty must be expressed in the selected
weighted log norm; a numerical trajectory alone does not qualify a reference.
"""

from __future__ import annotations

from collections.abc import Iterator
from fractions import Fraction
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
    weighted_logarithmic_norm_upper,
    weighted_positive_polynomial_tube,
)


@pytest.fixture(scope="module", autouse=True)
def one_thread() -> Iterator[None]:
    previous = ctx.threads
    ctx.threads = 1
    with threadpool_limits(limits=1), ctx.workprec(192):
        yield
    ctx.threads = previous


def q(value: float | int) -> Any:
    f = Fraction(value)
    return arb(fmpq(f.numerator, f.denominator))


def test_diagonal_scaling_of_known_metzler_matrix() -> None:
    matrix = ((arb(-4), arb(16)), (q(0.25), arb(-4)))
    assert weighted_logarithmic_norm_upper(matrix, [1.0, 1.0]) >= 12
    # D=diag(1,8) gives row sums -2 exactly and the leading eigenvalue is -2.
    bound = weighted_logarithmic_norm_upper(matrix, [1.0, 0.125])
    assert -2 <= bound < -1.999999999999


def test_interval_norm_covers_unsampled_matrix_values() -> None:
    matrix = (
        (arb(-2), arb(0).union(arb(300))),
        (arb(0).union(q(0.03)), arb(-2)),
    )
    weights = [1.0, 0.01]
    bound = weighted_logarithmic_norm_upper(matrix, weights)
    assert bound >= 1
    for upper, lower in ((0.0, 0.0), (150.0, 0.015), (300.0, 0.03)):
        actual = weighted_logarithmic_norm_upper(
            ((arb(-2), q(upper)), (q(lower), arb(-2))), weights
        )
        assert bound >= actual
    sampled_center = weighted_logarithmic_norm_upper(
        ((arb(-2), arb(150)), (q(0.015), arb(-2))), weights
    )
    assert sampled_center < 0  # A point-only certificate would be misleading.


@pytest.mark.parametrize("weights", [[0.0, 1.0], [-1.0, 1.0], [np.nan, 1.0], [1.0]])
def test_invalid_diagonal_weights_are_refused(weights: list[float]) -> None:
    with pytest.raises(ValueError, match="invalid_diagonal_norm_inputs"):
        weighted_logarithmic_norm_upper(((arb(1), arb(0)), (arb(0), arb(1))), weights)


class LinearField:
    """G(z)=A(z-1); constant polynomial z=1 is the exact solution."""

    branch_aware = True
    scale = np.ones(2)
    size = 2

    def __init__(self, diagonal: float):
        self.matrix = ((q(diagonal), arb(16)), (q(0.25), q(diagonal)))
        self.evaluations = 0

    def evaluate(
        self,
        left: Fraction,
        right: Fraction,
        inventories: tuple[Any, ...],
        *,
        derivatives: bool = True,
    ) -> Any:
        self.evaluations += 1
        return SimpleNamespace(
            rhs=tuple(
                sum((a * (z - 1) for a, z in zip(row, inventories)), arb(0))
                for row in self.matrix
            ),
            scaled_jacobian=self.matrix if derivatives else ((), ()),
            coordinates=inventories,
            densities=inventories,
            kernel_error_bound=(arb(0), arb(0)),
        )


def constant_path(end: float) -> PolynomialPath:
    coefficients = np.zeros((1, 4, 2))
    coefficients[0, 0] = 1.0
    return PolynomialPath(
        np.array([[0.0, end]]),
        coefficients,
        np.zeros_like(coefficients),
        np.ones(2),
        np.zeros(2),
        np.ones(2),
        np.ones((1, 2)),
        np.ones(2),
    )


def test_verified_weight_selection_enables_full_positive_tube() -> None:
    result = weighted_positive_polynomial_tube(
        cast(NonlinearInventoryEnclosure, LinearField(-4.0)),
        constant_path(16.0),
        initial_log_error=0.0,
        log_radius=1e-6,
    )
    uniform = next(c for c in result.candidates if c["candidate"] == "uniform")
    assert uniform["worst_exponent_upper"] > 50
    assert result.enclosed_slabs == 2
    assert result.best_exponent_upper is not None and result.best_exponent_upper < 0
    assert max(result.selected_weights) == 1 and min(result.selected_weights) > 0
    assert result.tube_bound is not None and result.tube_bound.tube_proved
    assert result.tube_bound.error_upper == 0.0
    assert not result.reference_qualified
    assert not result.comparison_matrices.flags.writeable


def test_diagonal_scaling_does_not_bypass_propagation_cap() -> None:
    field = LinearField(1.0)
    result = weighted_positive_polynomial_tube(
        cast(NonlinearInventoryEnclosure, field),
        constant_path(40.0),
        initial_log_error=0.0,
        log_radius=1e-6,
    )
    assert result.best_exponent_upper is not None and result.best_exponent_upper > 50
    assert result.reason == "weighted_log_norm_propagation_exponent_exceeds_50"
    assert result.tube_bound is None and field.evaluations == 2
    selected = next(
        c for c in result.candidates if c["candidate"] == result.selected_candidate
    )
    assert selected["global_comparison_abscissa_lower"] > 2.9
    assert (
        selected["global_comparison_abscissa_upper"]
        >= selected["global_comparison_abscissa_lower"]
    )
