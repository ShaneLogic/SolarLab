"""Nonlinear enclosure tests; short real-device integrations are a separate runner.

Integration conditions: optional, unqualified reference infrastructure for
restricted declared E/I inputs and analytic unit fixtures only. Independence
is static-only: no BDF stepping, history, interpolation or accounting is used.
Retained failed attempts stay failures and qualification flags stay false.
Any future nonzero initial uncertainty must be expressed in the selected
weighted log norm; a numerical trajectory alone does not qualify a reference.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
import ast
from dataclasses import replace
from fractions import Fraction
import inspect
from typing import Any, cast

import numpy as np
import pytest
from scipy.sparse import csc_matrix
from threadpoolctl import threadpool_limits

pytest.importorskip("flint")
from flint import arb, ctx

from examples.research_1d.ionic_multilayer import prepare_confined_ion_device
from examples.research_1d.steady import load_device
from solarlab.contracts.units import ONE
from solarlab.device.compile_1d import compile_device_1d
from solarlab.device.compile_ionic_1d import compile_ionic_device_1d
from solarlab.equations.assembly.assembler import AssembledSystem, Assembler
import solarlab.equations.assembly.ionic_1d as ionic_assembly
from solarlab.equations.assembly.classical_1d import AssembledDC1DProblem
from solarlab.equations.assembly.ionic_1d import AssembledIonic1DProblem
from solarlab.equations.assembly.layout import Block, NodeSet, SystemLayout
import solarlab.solvers.time.radau_enclosure as enclosure_module
from solarlab.solvers.time.radau_enclosure import (
    EnclosureBlocked,
    NonlinearInventoryEnclosure,
    PWLDrive,
    enclosed_inventory_dae,
    nonlinear_tube_bound,
)
from solarlab.solvers.time.radau_reference import (
    InventoryDAE,
    RadauControls,
    RadauCounters,
    integrate_radau,
)
import solarlab.solvers.time.ionic_backward_euler as ionic_be
import solarlab.solvers.time.radau_reference as radau_source


@pytest.fixture(scope="module", autouse=True)
def one_thread() -> Iterator[None]:
    old_threads = ctx.threads
    ctx.threads = 1
    with threadpool_limits(limits=1), ctx.workprec(192):
        yield
    ctx.threads = old_threads


def ball(value: float | int | Fraction) -> Any:
    return enclosure_module._q(value)


def test_exact_polynomial_derivative_does_not_round_the_integer_product() -> None:
    p = np.array([[0.0], [0.0], [0.0], [np.nextafter(1.0, np.inf)]])
    value = enclosure_module._polynomial_range(
        p, Fraction(1), Fraction(1), derivative=True
    )[0]
    exact = ball(Fraction(float(p[3, 0])) * 3)
    assert value.contains(exact)
    assert not value.contains(ball(float(p[3, 0] * 3)))


def test_maximum_uses_enclosed_endpoints_not_float_ranking() -> None:
    larger = ball(Fraction(1) + Fraction(1, 2**100))
    assert float(larger) == 1.0
    maximum = enclosure_module._maximum([arb(1), larger])
    assert maximum >= larger


@pytest.mark.parametrize("historical", [False, True])
@pytest.mark.parametrize("x", [-0.3, 0.3, 1e-9])
def test_bernoulli_interval_derivative_matches_exact_formula(
    x: float, historical: bool
) -> None:
    q = ball(x)
    jet = enclosure_module._Jet(q, (arb(1),))
    value = enclosure_module._bernoulli(jet, historical=historical)
    if abs(x) < 1e-8:
        exact, derivative = 1 - q / 2 + q * q / 12, -ball(Fraction(1, 2)) + q / 6
    else:
        exact = q / q.expm1()
        derivative = (q.expm1() - q * q.exp()) / (q.expm1() ** 2)
    assert value.value.overlaps(exact)
    assert value.gradient[0].overlaps(derivative)


def test_non_lipschitz_approximation_branch_is_refused() -> None:
    cut = ball(1e-8)
    value = (cut - ball(1e-12)).union(cut + ball(1e-12))
    with pytest.raises(EnclosureBlocked, match="bernoulli_value_branch_crossing"):
        enclosure_module._bernoulli(enclosure_module._Jet(value, (arb(1),)))


def test_unimplemented_carrier_extreme_product_branch_is_refused() -> None:
    jet = enclosure_module._Jet
    with pytest.raises(EnclosureBlocked, match="carrier_extreme_product_branch"):
        enclosure_module._carrier_flux(
            jet(arb(1)), jet(arb(1)), jet(arb(501)), jet(arb(0)), jet(arb(1))
        )


def test_frozen_PWL_interpolation_and_knot_refusal() -> None:
    drive = PWLDrive((0.0, 0.5, 1.0), (0.0, 2.0, 3.0), (1.0, 2.0, 1.0))
    V, g = drive.enclose(Fraction(1, 8), Fraction(1, 4))
    assert V.contains(ball(Fraction(1, 2))) and V.contains(arb(1))
    assert g.contains(ball(Fraction(5, 4))) and g.contains(ball(Fraction(3, 2)))
    with pytest.raises(EnclosureBlocked, match="drive_knot"):
        drive.enclose(Fraction(1, 4), Fraction(3, 4))


@pytest.mark.parametrize("ionic", [False, True])
def test_actual_E_I_nonlinear_value_and_jacobian_enclosures(ionic: bool) -> None:
    if ionic:
        ion = compile_ionic_device_1d(
            prepare_confined_ion_device(),
            intervals_per_layer=(4, 4, 4),
            ion_domain_layer="absorber",
            ion_domain_endpoints=("blocking", "blocking"),
        )
        problem: Any = AssembledIonic1DProblem(ion)
        e = ion.electronic_component
        u = np.r_[
            e.reference_potential_V, np.zeros(2 * e.node_count + ion.ion_node_count)
        ]
    else:
        e = compile_device_1d(load_device(), intervals_per_layer=(4, 4, 4))
        problem = AssembledDC1DProblem(e)
        u = np.r_[e.reference_potential_V, np.zeros(2 * e.node_count)]
    mask = problem.binding.layout.constraint_mask()
    mask[problem.binding.layout.rows("potential")] = True
    z = problem.storage(u)[~mask]
    drive = PWLDrive((0.0, 1.0), (0.0, 0.002), (1.0, 1.0))
    enclosure = NonlinearInventoryEnclosure(problem, drive, z)
    epsilon = ball(Fraction(1, 10**8))
    boxes = tuple(
        (ball(value) * (1 - epsilon)).union(ball(value) * (1 + epsilon)) for value in z
    )
    result = enclosure.evaluate(Fraction(0), Fraction(0), boxes)
    assert len(result.rhs) == z.size and len(result.coordinates) == u.size
    # Mean-value check against independent high-precision directional secants
    # across actual physical-inventory boxes; no binary64 Jacobian is assumed.
    for column in (0, z.size // 2, z.size - 1):
        plus, minus = [ball(value) for value in z], [ball(value) for value in z]
        delta = epsilon * ball(z[column]) / 2
        plus[column] += delta
        minus[column] -= delta
        a = enclosure.evaluate(Fraction(0), Fraction(0), tuple(plus), derivatives=False)
        b = enclosure.evaluate(
            Fraction(0), Fraction(0), tuple(minus), derivatives=False
        )
        for row, (left, right) in enumerate(zip(a.rhs, b.rhs)):
            secant = (left - right) / epsilon / ball(z[row])
            assert result.scaled_jacobian[row][column].overlaps(secant), (
                ionic,
                row,
                column,
            )
    if ionic:
        assert (
            np.max(ion.positive_ion_reference_m3 / ion.positive_ion_site_limit_m3)
            < 0.999999
        )


def test_enclosed_point_evaluation_has_no_reconstruction_tolerance_dead_zone() -> None:
    e = compile_device_1d(load_device(), intervals_per_layer=(4, 4, 4))
    problem = AssembledDC1DProblem(e)
    u = np.r_[e.reference_potential_V, np.zeros(2 * e.node_count)]
    mask = problem.binding.layout.constraint_mask()
    mask[problem.binding.layout.rows("potential")] = True
    z = problem.storage(u)[~mask]
    drive = PWLDrive((0.0, 1.0), (0.0, 0.0), (1.0, 1.0))
    model = enclosed_inventory_dae(
        problem, drive, inventory_scale=z, algebraic_scale=np.ones(e.node_count + 4)
    )
    controls, counters = RadauControls(), RadauCounters()
    point = model.reconstruct(0.0, z, u, controls, counters)
    changed = z.copy()
    changed[0] *= 1 + 1e-13
    moved = model.reconstruct(0.0, changed, point.u, controls, counters)
    assert np.max(np.abs((changed - z) / z)) < controls.scaled_newton_tolerance
    assert np.any(point.rhs != moved.rhs)
    assert moved.scaled_reconstruction_residual <= controls.scaled_newton_tolerance


def test_transitive_BE_secant_is_unreachable_on_reference_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Static reachability: the only secant call is inside IonStorage.increment;
    # the only dispatch to a storage increment is Assembler.storage_increment.
    tree = ast.parse(inspect.getsource(ionic_assembly))
    sites = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            for method in node.body:
                if isinstance(method, ast.FunctionDef) and any(
                    isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Name)
                    and call.func.id == "finite_site_secant_m3"
                    for call in ast.walk(method)
                ):
                    sites.append((node.name, method.name))
    assert sites == [("IonStorage", "increment")]
    methods = ast.parse(inspect.getsource(Assembler)).body[0]
    assert isinstance(methods, ast.ClassDef)
    dispatches = [
        method.name
        for method in methods.body
        if isinstance(method, ast.FunctionDef)
        and any(
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "increment"
            for call in ast.walk(method)
        )
    ]
    assert dispatches == ["storage_increment"]
    reference_tree = ast.parse(inspect.getsource(radau_source))
    assert not any(
        isinstance(node, ast.Attribute)
        and node.attr in {"increment", "storage_increment"}
        for node in ast.walk(reference_tree)
    )
    calls = []

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        calls.append("BE_secant_or_increment")
        raise AssertionError("reference attempted to use a BE secant/increment")

    monkeypatch.setattr(ionic_assembly, "finite_site_secant_m3", forbidden)
    monkeypatch.setattr(ionic_be, "finite_site_secant_m3", forbidden)
    monkeypatch.setattr(ionic_assembly.IonStorage, "increment", forbidden)
    monkeypatch.setattr(Assembler, "storage_increment", forbidden)
    ion = compile_ionic_device_1d(
        prepare_confined_ion_device(),
        intervals_per_layer=(4, 4, 4),
        ion_domain_layer="absorber",
        ion_domain_endpoints=("blocking", "blocking"),
    )
    problem = AssembledIonic1DProblem(ion)
    e = ion.electronic_component
    u = np.r_[e.reference_potential_V, np.zeros(2 * e.node_count + ion.ion_node_count)]
    mask = problem.binding.layout.constraint_mask()
    mask[problem.binding.layout.rows("potential")] = True
    z = problem.storage(u)[~mask]
    drive = PWLDrive((0.0, 1.0), (0.0, 0.002), (0.0, 0.0))
    for factory in (radau_source.assembled_inventory_dae, enclosed_inventory_dae):
        model = factory(
            problem, drive, inventory_scale=z, algebraic_scale=np.ones(e.node_count + 4)
        )
        model.assembled(0.0, u, RadauCounters())
        model.reconstruct(0.0, z, u, RadauControls(), RadauCounters(), derivatives=True)
    assert not calls


def test_nonlinear_tube_against_exact_logistic_decay(
    record_property: Callable[[str, object], None],
) -> None:
    layout = SystemLayout(
        (NodeSet("one", 1),), (Block("a", "one", ONE, ONE), Block("z", "one", ONE, ONE))
    )

    def evaluate(t: float, u: Any) -> AssembledSystem:
        z = np.exp(u[1])
        F = np.array([u[0] - z, -z * z])
        return AssembledSystem(
            layout,
            True,
            F,
            F.copy(),
            csc_matrix([[1, -z], [0, -2 * z * z]]),
            np.array([0.0, z]),
            csc_matrix([[0, 0], [0, z]]),
        )

    model = InventoryDAE(layout, evaluate, (1,), np.ones(1), np.ones(1))
    result = integrate_radau(
        model,
        [1.0, 0.0],
        np.linspace(0.0, 0.1, 9),
        controls=RadauControls(scaled_newton_tolerance=1e-13),
    )

    class ExactLogistic:
        size = 1
        scale = np.ones(1)
        evaluations = 0

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
            return enclosure_module._Evaluation(
                (-z * z,), ((-2 * z,),) if derivatives else ((),), (z, z.log()), (z,)
            )

    bound = nonlinear_tube_bound(
        cast(NonlinearInventoryEnclosure, ExactLogistic()),
        result,
        initial_error=0.0,
        tube_radius=0.01,
        subdivisions=2,
    )
    assert (
        bound.tube_proved
        and bound.error_upper is not None
        and not bound.reference_qualified
    )
    assert bound.status == "reference_blocked"
    observed, floating_observed, output_roundoff = Fraction(0), Fraction(0), Fraction(0)
    for t in np.linspace(0.0, 0.1, 65):
        step = next(s for s in result.steps if s.left_s <= t <= s.right_s)
        theta = (Fraction(float(t)) - Fraction(step.left_s)) / (
            Fraction(step.right_s) - Fraction(step.left_s)
        )
        polynomial = sum(
            (Fraction(float(step.coefficients[k, 0])) * theta**k for k in range(4)),
            Fraction(0),
        )
        exact = 1 / (1 + Fraction(float(t)))
        floating = Fraction(float(result.sample(float(t)).inventory[0]))
        observed = max(observed, abs(polynomial - exact))
        floating_observed = max(floating_observed, abs(floating - exact))
        output_roundoff = max(output_roundoff, abs(floating - polynomial))
    assert Fraction(bound.error_upper) >= observed
    assert Fraction(bound.error_upper) + output_roundoff >= floating_observed
    record_property("polynomial_error_bound", bound.error_upper)
    record_property("observed_polynomial_error", float(observed))
    record_property("observed_floating_output_error", float(floating_observed))
    record_property(
        "sampled_output_roundoff_upper", enclosure_module._upper(ball(output_roundoff))
    )
    record_property("completed_slabs", bound.completed_slabs)
    record_property("scope_status", bound.status)
    blocked = nonlinear_tube_bound(
        cast(NonlinearInventoryEnclosure, ExactLogistic()),
        result,
        initial_error=0.0,
        tube_radius=1e-14,
        subdivisions=2,
    )
    assert blocked.status == "reference_blocked" and blocked.error_upper is None
    for altered, reason in (
        (replace(result, steps=()), "empty_trajectory_has_no_time_tube"),
        (
            replace(
                result, steps=(replace(result.steps[0], coefficients=np.zeros((4, 2))),)
            ),
            "trajectory_polynomial_shape_mismatch",
        ),
        (
            replace(
                result,
                steps=(
                    replace(
                        result.steps[0],
                        endpoint=replace(result.steps[0].endpoint, time_s=1.0),
                    ),
                ),
            ),
            "trajectory_endpoint_mismatch",
        ),
    ):
        invalid = nonlinear_tube_bound(
            cast(NonlinearInventoryEnclosure, ExactLogistic()),
            altered,
            initial_error=0.0,
            tube_radius=0.01,
        )
        assert not invalid.tube_proved and invalid.error_upper is None
        assert invalid.reason == reason
