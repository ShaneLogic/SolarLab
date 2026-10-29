"""Split-reference arithmetic and retained constraints; no device time runs.

Integration conditions: optional, unqualified reference infrastructure for
restricted declared E/I inputs and analytic unit fixtures only. Independence
is static-only: no BDF stepping, history, interpolation or accounting is used.
Retained failed attempts stay failures and qualification flags stay false.
Any future nonzero initial uncertainty must be expressed in the selected
weighted log norm; a numerical trajectory alone does not qualify a reference.
"""

from __future__ import annotations

import ast
from collections.abc import Callable, Iterator
from fractions import Fraction
import inspect
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from scipy.sparse import csc_matrix
from threadpoolctl import threadpool_limits

pytest.importorskip("flint")
from flint import arb, ctx, fmpq

from solarlab.contracts.units import ONE
from solarlab.equations.assembly.assembler import AssembledSystem
from solarlab.equations.assembly.layout import Block, NodeSet, SystemLayout
from solarlab.solvers.time.radau_enclosure import (
    EnclosureBlocked,
    NonlinearInventoryEnclosure,
)
import solarlab.solvers.time.radau_enclosure as enclosure_source
from solarlab.solvers.time.radau_reference import (
    InventoryDAE,
    RadauControls,
    RadauFailure,
    integrate_radau,
)
import solarlab.solvers.time.radau_reference as stage1
from solarlab.solvers.time.radau_split import (
    SplitInventory,
    SplitRadauFailure,
    SplitRadauModel,
    integrate_split_radau,
)
import solarlab.solvers.time.radau_split as split


@pytest.fixture(scope="module", autouse=True)
def one_thread() -> Iterator[None]:
    old = ctx.threads
    ctx.threads = 1
    with threadpool_limits(limits=1), ctx.workprec(192):
        yield
    ctx.threads = old


def q(value: float | int | Fraction) -> Any:
    f = Fraction(value)
    return arb(fmpq(f.numerator, f.denominator))


class AffineEnclosure:
    size = 1
    scale = np.ones(1)

    def __init__(self, rate: float, source: float, reference: float):
        self.rate, self.source, self.reference = rate, source, reference
        self.evaluations = 0
        self.fail_after: Fraction | None = None

    def evaluate(
        self,
        left: Fraction,
        right: Fraction,
        inventories: tuple[Any, ...],
        *,
        derivatives: bool = True,
    ) -> Any:
        self.evaluations += 1
        if self.fail_after is not None and right > self.fail_after:
            raise EnclosureBlocked("test_enclosure_stopped")
        z = inventories[0]
        return SimpleNamespace(
            rhs=(q(self.rate) * (z - q(self.reference)) + q(self.source),),
            scaled_jacobian=((q(self.rate),),) if derivatives else ((),),
            coordinates=(z, z),
            densities=(z,),
        )


def model(rate: float, source: float = 0.0, reference: float = 0.0) -> SplitRadauModel:
    layout = SystemLayout(
        (NodeSet("one", 1),),
        (Block("constraint", "one", ONE, ONE), Block("inventory", "one", ONE, ONE)),
    )

    def evaluate(time_s: float, u: Any) -> AssembledSystem:
        F = np.array([u[0] - u[1], rate * (u[1] - reference) + source])
        return AssembledSystem(
            layout,
            True,
            F,
            F.copy(),
            csc_matrix([[1.0, -1.0], [0.0, rate]]),
            np.array([0.0, u[1]]),
            csc_matrix([[0.0, 0.0], [0.0, 1.0]]),
        )

    static = InventoryDAE(layout, evaluate, (1,), np.ones(1), np.ones(1))
    return SplitRadauModel(
        static,
        cast(NonlinearInventoryEnclosure, AffineEnclosure(rate, source, reference)),
    )


def test_original_tableau_and_controls_are_unchanged() -> None:
    for name in ("_A", "_C", "_BASIS"):
        assert getattr(split, name).tobytes() == getattr(stage1, name).tobytes()
    assert RadauControls().scaled_newton_tolerance == 1e-12


def test_split_reference_and_scaled_departure_preserve_subulp_words() -> None:
    high, low = np.ones(1), np.array([2.0**-60])
    parts = SplitInventory(high, low)
    high[0], low[0] = 2.0, 0.0
    assert not parts.high.flags.writeable and not parts.low.flags.writeable
    assert parts.values()[0].contains(q(Fraction(1) + Fraction(1, 2**60)))
    shifted = parts.shifted(np.array([2.0**-80]), np.ones(1))[0]
    assert shifted.contains(q(Fraction(1) + Fraction(1, 2**60) + Fraction(1, 2**80)))
    assert not shifted.contains(parts.values()[0])
    encoded, error = SplitInventory.from_values((shifted,))
    assert error[0] == 0 and encoded.values()[0].contains(shifted)


def test_stiff_subulp_update_completes_without_relaxing_tolerance(
    record_property: Callable[[str, object], None],
) -> None:
    bound_model = model(-1e12, 1e-5, 1.0)
    grid = np.linspace(0.0, 0.1, 5)
    with pytest.raises(RadauFailure, match="radau_stage_line_search"):
        integrate_radau(bound_model.static, [1.0, 1.0], grid)
    result = integrate_split_radau(bound_model, [1.0, 1.0], grid)
    assert len(result.steps) == 4 and not result.qualified
    assert result.steps[-1].endpoint.inventory.high[0] == 1.0
    assert result.steps[-1].endpoint.inventory.low[0] > 0.0
    assert result.counters.changed_departures_with_unchanged_high_word > 0
    maximum = max(s.scaled_stage_residual_upper for s in result.steps)
    assert maximum <= RadauControls().scaled_newton_tolerance
    for point in (
        result.initial,
        *(p for s in result.steps for p in (*s.stages, s.endpoint)),
        *(result.sample(float(t)) for t in np.linspace(0.0, 0.1, 9)),
    ):
        assert np.array_equal(point.algebraic_residual, np.zeros(1))
        assert (
            point.scaled_reconstruction_residual
            <= RadauControls().scaled_newton_tolerance
        )
    record_property("max_scaled_stage_residual", maximum)
    record_property(
        "endpoint_inventory_high", result.steps[-1].endpoint.inventory.high[0]
    )
    record_property(
        "endpoint_inventory_low", result.steps[-1].endpoint.inventory.low[0]
    )
    record_property(
        "updates_lost_by_unsplit_high_word",
        result.counters.changed_departures_with_unchanged_high_word,
    )


def test_split_endpoint_order_and_bound_against_exact_RC(
    record_property: Callable[[str, object], None],
) -> None:
    errors, trajectories = [], []
    for count in (4, 8, 16):
        trajectory = integrate_split_radau(
            model(-2.0), [1.0, 1.0], np.linspace(0.0, 1.0, count + 1)
        )
        trajectories.append(trajectory)
        errors.append(
            float(abs(trajectory.steps[-1].polynomial(1.0)[0] - q(-2).exp()).mid())
        )
    orders = np.log2(np.array(errors[:-1]) / errors[1:])
    assert np.all((orders > 4.8) & (orders < 5.2))
    finest = trajectories[-1]
    proof = finest.bound(initial_error=0.0, tube_radius=0.1, subdivisions=2)
    assert (
        proof.tube_proved
        and proof.error_upper is not None
        and not proof.reference_qualified
    )
    observed = arb(0)
    for t in np.linspace(0.0, 1.0, 65):
        step = next(s for s in finest.steps if s.left_s <= t <= s.right_s)
        error = abs(step.polynomial(float(t))[0] - (q(-2) * q(float(t))).exp())
        observed = observed.union(error)
    assert q(proof.error_upper) >= observed.upper()
    record_property("endpoint_errors", errors)
    record_property("endpoint_orders", orders.tolist())
    record_property("polynomial_error_bound", proof.error_upper)
    record_property("observed_polynomial_error_upper", split._upper(observed.upper()))


def test_low_polynomial_word_and_initial_word_are_used_in_tube_proof() -> None:
    parts = SplitInventory(np.ones(1), np.array([2.0**-80]))
    trajectory = integrate_split_radau(
        model(0.0), [1.0, 1.0], [0.0, 0.1], initial_inventory=parts
    )
    assert trajectory.steps[0].coefficients_low[0, 0] == 2.0**-80
    bound = trajectory.bound(initial_error=0.0, tube_radius=0.01)
    assert bound.tube_proved and bound.error_upper == 0.0
    assert bound.reconstruction_error_upper[0] >= 2.0**-80
    high = np.zeros((4, 1))
    high[0, 0] = 1.0
    low = np.zeros((4, 1))
    low[0, 0] = 2.0**-80
    low[3, 0] = 2.0**-100
    value = enclosure_source._polynomial_range(
        high, Fraction(1), Fraction(1), correction=low
    )[0]
    derivative = enclosure_source._polynomial_range(
        high, Fraction(1), Fraction(1), derivative=True, correction=low
    )[0]
    assert value.contains(q(Fraction(1) + Fraction(1, 2**80) + Fraction(1, 2**100)))
    assert derivative.contains(q(Fraction(3, 2**100)))


def test_failure_keeps_exact_start_departures_and_completed_prefix() -> None:
    bound_model = model(-2.0)
    cast(AffineEnclosure, bound_model.enclosure).fail_after = Fraction(1, 16)
    with pytest.raises(SplitRadauFailure) as info:
        integrate_split_radau(bound_model, [1.0, 1.0], [0.0, 0.0625, 0.125])
    error = info.value
    assert error.failed_step == 1 and len(error.completed_steps) == 1
    assert (
        error.step_start.high.tobytes()
        == error.completed_steps[0].endpoint.inventory.high.tobytes()
    )
    assert (
        error.step_start.low.tobytes()
        == error.completed_steps[0].endpoint.inventory.low.tobytes()
    )
    assert error.trial_departures is not None
    assert not error.trial_departures.flags.writeable and error.counters is not None


def test_budget_stop_keeps_prefix_without_inventing_a_trial() -> None:
    bound_model = model(-2.0)
    original = bound_model.enclosure.evaluate

    def expired(
        left: Fraction,
        right: Fraction,
        inventories: tuple[Any, ...],
        *,
        derivatives: bool = True,
    ) -> Any:
        if right > Fraction(1, 16):
            raise TimeoutError("test deadline")
        return original(left, right, inventories, derivatives=derivatives)

    bound_model.enclosure.evaluate = expired  # type: ignore[method-assign]
    with pytest.raises(SplitRadauFailure) as info:
        integrate_split_radau(bound_model, [1.0, 1.0], [0.0, 0.0625, 0.125])
    assert info.value.code == "radau_split_budget_exhausted"
    assert info.value.failed_step == 1 and len(info.value.completed_steps) == 1
    assert info.value.trial_departures is None


def test_split_route_has_no_BDF_BE_or_storage_increment_calls() -> None:
    tree = ast.parse(inspect.getsource(split))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert "bdf" not in (node.module or "") and "backward_euler" not in (
                node.module or ""
            )
        if isinstance(node, ast.Attribute):
            assert node.attr not in {
                "increment",
                "storage_increment",
                "finite_site_secant_m3",
            }
