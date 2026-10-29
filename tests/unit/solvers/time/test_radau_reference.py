"""Unit-only independent Radau tests; no admission protocol or native campaign.

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
from dataclasses import replace
from decimal import Decimal, localcontext
import inspect
import math
from pathlib import Path
import subprocess
import sys
from textwrap import dedent
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray
import pytest
from scipy.sparse import csc_matrix
from threadpoolctl import threadpool_limits

from examples.research_1d.ionic_multilayer import prepare_confined_ion_device
from examples.research_1d.steady import load_device
from solarlab.contracts.units import ONE
from solarlab.device.compile_1d import compile_device_1d
from solarlab.device.compile_ionic_1d import compile_ionic_device_1d
from solarlab.equations.assembly.assembler import AssembledSystem
from solarlab.equations.assembly.classical_1d import AssembledDC1DProblem
from solarlab.equations.assembly.ionic_1d import AssembledIonic1DProblem
from solarlab.equations.assembly.layout import Block, ConstraintRows, NodeSet, SystemLayout
from solarlab.equations.assembly.terms import OperatingInputs
import solarlab.solvers.time.radau_reference as module
from solarlab.solvers.time.radau_reference import (
    InventoryDAE, RadauControls, RadauCounters, RadauFailure, RadauTrajectory,
    assembled_inventory_dae, integrate_radau, linear_tube_bound,
)


def test_optional_arb_modules_import_without_python_flint() -> None:
    """Core imports stay usable; only an explicit Arb operation needs the extra."""
    source_root = Path(module.__file__).resolve().parents[3]
    program = dedent(
        """
        import importlib.abc
        import sys

        class AbsentFlint(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "flint" or fullname.startswith("flint."):
                    raise ModuleNotFoundError("flint deliberately unavailable", name="flint")
                return None

        sys.meta_path.insert(0, AbsentFlint())
        sys.path.insert(0, sys.argv[1])
        import solarlab
        from solarlab.solvers.time import radau_reference, radau_enclosure, radau_split

        assert "flint" not in sys.modules
        assert radau_reference.RadauControls().scaled_newton_tolerance == 1e-12
        parts = radau_split.SplitInventory([1.0], [0.0])
        for operation in (parts.values, lambda: radau_enclosure.bernoulli_branch_enclosure(0.0)):
            try:
                operation()
            except ImportError as error:
                assert "optional python-flint backend" in str(error)
                assert "solarlab[arb]" in str(error)
            else:
                raise AssertionError("Arb operation did not report its missing dependency")
        assert "flint" not in sys.modules
        """
    )
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", program, str(source_root)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr

Vector: TypeAlias = NDArray[np.float64]
CONTROLS = RadauControls(scaled_newton_tolerance=2.0e-14)


@pytest.fixture(scope="module", autouse=True)
def blas_one() -> Iterator[None]:
    with threadpool_limits(limits=1):
        yield


def affine_dae(matrix: Vector, inventories: Vector, weights: Vector | None = None,
               forcing: Vector | None = None, *, coordinate_shift_per_s: float = 0.0) -> tuple[InventoryDAE, Vector]:
    """Exact affine inventory dynamics behind nonlinear log-density coordinates.

    Two algebraic rows mimic Poisson response and a time-varying reservoir:
    a0=sum((j+1) z_j)+sin(t), a1=1+cos(t); z_j=W_j exp(u_j).
    These constraints cannot be obtained by interpolating endpoint a values.
    """
    n = inventories.size
    W = np.ones(n) if weights is None else weights
    b = np.zeros(n) if forcing is None else forcing
    coupling = np.arange(1, n + 1, dtype=float)
    layout = SystemLayout((NodeSet("algebraic", 2), NodeSet("cells", n)),
                          (Block("a", "algebraic", ONE, ONE), Block("log_density", "cells", ONE, ONE)),
                          (ConstraintRows("a", (0, 1), "algebraic.constraints", ONE),))

    def evaluate(time_s: float, u: Vector) -> AssembledSystem:
        with np.errstate(over="raise", invalid="raise"):
            z = W * np.exp(u[2:] + coordinate_shift_per_s * time_s)
        F = np.r_[u[0] - coupling @ z - math.sin(time_s),
                  u[1] - 1.0 - math.cos(time_s), matrix @ z + b]
        J = np.zeros((n + 2, n + 2))
        J[0, 0], J[1, 1] = 1.0, 1.0
        J[0, 2:] = -coupling * z
        J[2:, 2:] = matrix * z[None, :]
        Su = np.zeros_like(J)
        Su[2:, 2:] = np.diag(z)
        return AssembledSystem(layout, True, F, F.copy(), csc_matrix(J), np.r_[np.zeros(2), z], csc_matrix(Su))

    model = InventoryDAE(layout, evaluate, tuple(range(2, n + 2)), np.ones(n), np.ones(2))
    # Deliberately inconsistent algebraic seed; initial inventories remain fixed.
    return model, np.r_[0.0, 0.0, np.log(inventories / W)]


def decay(time_s: float, rate: float = -2.0) -> float:
    """Independent 80-digit analytic exponential, not a time integrator."""
    with localcontext() as context:
        context.prec = 80
        return float((Decimal.from_float(rate) * Decimal.from_float(time_s)).exp())


def trajectory(rate: float = -2.0, steps: int = 16) -> RadauTrajectory:
    model, u = affine_dae(np.array([[rate]]), np.array([1.0]), np.array([2.0]))
    return integrate_radau(model, u, np.linspace(0.0, 1.0, steps + 1), controls=CONTROLS)


@pytest.mark.parametrize("kind", ["rc", "diffusion"])
def test_fifth_order_at_endpoints(kind: str, record_property: Callable[[str, object], None]) -> None:
    if kind == "rc":
        A, z0, weights = np.array([[-2.0]]), np.array([1.0]), np.array([2.0])
        exact = np.array([decay(1.0)])
    else:
        # Two closed cells with unequal CV measures and one diffusive face.
        # F=[g(n1-n0), -g(n1-n0)], eigenvalues 0 and -2 s^-1.
        weights = np.array([0.25, 0.75])
        A = 0.375 * np.array([[-1 / weights[0], 1 / weights[1]],
                             [1 / weights[0], -1 / weights[1]]])
        z0 = np.array([0.5, 0.5])
        exact = weights + np.array([0.25, -0.25]) * decay(1.0)
    errors = []
    for count in (4, 8, 16):
        model, u = affine_dae(A, z0, weights)
        result = integrate_radau(model, u, np.linspace(0, 1, count + 1), controls=CONTROLS)
        errors.append(float(np.max(np.abs(result.steps[-1].endpoint.inventory - exact))))
        if kind == "diffusion":
            assert all(abs(float(np.sum(step.endpoint.inventory)) - 1.0) < 3e-14 for step in result.steps)
    orders = np.log2(np.asarray(errors[:-1]) / errors[1:])
    assert np.all((orders > 4.7) & (orders < 5.4)), (errors, orders)
    record_property("radau_order", {"case": kind, "step_counts": [4, 8, 16],
                                    "errors": errors, "observed_orders": orders.tolist()})


def test_every_stage_and_output_reconstructs_constraints_and_inventories() -> None:
    result = trajectory(steps=5)
    history = [step.coefficients.tobytes() for step in result.steps]
    stage_words = [[point.u.tobytes() for point in step.stages] for step in result.steps]
    outputs = [0.0, 0.037, 0.11, 0.401, 0.713, 0.92, 1.0]
    points = [point for step in result.steps for point in (*step.stages, step.endpoint)]
    points += [result.sample(t) for t in outputs]
    for point in points:
        value = result.model.evaluate(point.time_s, point.u)
        assert value.S is not None
        np.testing.assert_allclose(value.F[:2], 0.0, atol=3e-14, rtol=0)
        np.testing.assert_allclose(value.S[2:], point.inventory, atol=3e-14, rtol=0)
        assert point.scaled_reconstruction_residual <= CONTROLS.scaled_newton_tolerance
    assert all(np.max(np.abs(step.stage_residual)) <= CONTROLS.scaled_newton_tolerance for step in result.steps)
    assert result.counters.output_reconstructions == len(outputs)
    assert history == [step.coefficients.tobytes() for step in result.steps]
    assert stage_words == [[point.u.tobytes() for point in step.stages] for step in result.steps]
    with pytest.raises(ValueError):
        result.steps[0].coefficients[0, 0] = 1.0
    assert not result.qualified


def test_elimination_jacobian_and_full_row_rate_mapping() -> None:
    A = np.array([[-2.0, 0.5], [0.25, -1.0]])
    z = np.array([1.2, 0.7])
    shift = 0.4
    model, u = affine_dae(A, z, coordinate_shift_per_s=shift)
    counters = RadauCounters()
    point = model.reconstruct(0.2, z, u, CONTROLS, counters, derivatives=True)
    np.testing.assert_allclose(point.rhs_jacobian, A, atol=3e-15, rtol=0)
    for column in range(2):
        delta = np.zeros(2)
        delta[column] = 1e-5
        plus = model.reconstruct(0.2, z + delta, point.u, CONTROLS, counters).rhs
        minus = model.reconstruct(0.2, z - delta, point.u, CONTROLS, counters).rhs
        np.testing.assert_allclose((plus - minus) / (2e-5), A[:, column], atol=4e-11, rtol=0)
    system = model.evaluate(0.2, point.u)
    assert system.J is not None and system.S_u is not None
    B = np.vstack((system.S_u[2:, :].toarray(), system.J[:2, :].toarray()))
    # Voltage-referenced coordinates can give S explicit time dependence.
    S_t = np.r_[np.zeros(2), shift * z]
    g_t = np.array([-np.arange(1, 3) @ (shift * z) - math.cos(0.2), math.sin(0.2)])
    udot = np.linalg.solve(B, np.r_[point.rhs - S_t[2:], -g_t])
    np.testing.assert_allclose(system.S_u[2:, :] @ udot + S_t[2:], system.F[2:], atol=1e-15, rtol=0)
    np.testing.assert_allclose(system.J[:2, :] @ udot + g_t, 0.0, atol=1e-15, rtol=0)
    np.testing.assert_allclose(system.F - system.S_u @ udot - S_t, 0.0, atol=3e-14, rtol=0)


@pytest.mark.parametrize("rate", [-2.0, 0.0, 2.0])
def test_exact_rational_tube_bounds_analytic_solution(rate: float, record_property: Callable[[str, object], None]) -> None:
    result = trajectory(rate)
    times = np.linspace(0, 1, 97).tolist()
    certificate = linear_tube_bound(result, [[rate]], [1.0], component_scale=[1.0],
                                    initial_radius=[0.0], tube_radius=0.01,
                                    reference_budget=0.01, output_times_s=times)
    actual = float(max(abs(result.sample(t).inventory[0] - decay(t, rate)) for t in times))
    assert actual <= certificate.scaled_error_upper
    assert certificate.scaled_error_upper < 0.01
    assert certificate.tube_validated and certificate.within_budget
    assert certificate.log_norm_upper == rate
    assert certificate.full_state_reconstruction_bound is None
    assert not certificate.reference_qualified
    record_property("radau_tube", {"rate": rate, "sampled_exact_error": actual,
                                   "whole_interval_error_upper": certificate.scaled_error_upper,
                                   "max_defect_upper": max(certificate.defect_upper),
                                   "tube_validated": certificate.tube_validated,
                                   "inventory_budget_met": certificate.within_budget,
                                   "device_reference_qualified": False})


def test_loose_or_unproved_device_bound_never_qualifies() -> None:
    result = trajectory()
    args = dict(component_scale=[1.0], initial_radius=[0.0])
    loose = linear_tube_bound(result, [[-2.0]], [1.0], tube_radius=0.01, reference_budget=1e-15, **args)
    outside = linear_tube_bound(result, [[-2.0]], [1.0], tube_radius=1e-15, reference_budget=0.01, **args)
    assert loose.tube_validated and not loose.within_budget and loose.status == "reference_budget_exceeded"
    assert not outside.tube_validated and not outside.within_budget and outside.status == "tube_radius_exceeded"
    assert not loose.reference_qualified and loose.full_state_reconstruction_bound is None
    with pytest.raises(ValueError, match="invalid_affine_data"):
        linear_tube_bound(result, [[float("nan")]], [1.0], tube_radius=1, reference_budget=1, **args)


def test_initial_uncertainty_scaling_and_stored_polynomial_jump_are_carried() -> None:
    result = trajectory()
    uncertain = linear_tube_bound(result, [[-2.0]], [1.0], component_scale=[1.0], initial_radius=[1e-4],
                                  tube_radius=0.01, reference_budget=0.01)
    scaled = linear_tube_bound(result, [[-2.0]], [1.0], component_scale=[10.0], initial_radius=[1e-4],
                               tube_radius=0.01, reference_budget=0.01)
    assert uncertain.scaled_error_upper >= 1e-4
    assert scaled.scaled_error_upper == pytest.approx(uncertain.scaled_error_upper / 10, rel=3e-15)
    p = result.steps[3].coefficients.copy()
    p[0, 0] += 1e-3
    changed = replace(result.steps[3], coefficients=p)
    shifted = replace(result, steps=result.steps[:3] + (changed,) + result.steps[4:])
    jump = linear_tube_bound(shifted, [[-2.0]], [1.0], component_scale=[1.0], initial_radius=[0.0],
                             tube_radius=0.01, reference_budget=0.01)
    assert jump.interval_error_upper[3] >= 1e-3


@pytest.mark.parametrize("ionic", [False, True])
def test_public_assembled_E_I_partition_and_reconstruction(ionic: bool) -> None:
    # 13 electronic / 5 ionic nodes, never the 37/73-node admission runs.
    if ionic:
        device = compile_ionic_device_1d(prepare_confined_ion_device(), intervals_per_layer=(4, 4, 4),
                                        ion_domain_layer="absorber", ion_domain_endpoints=("blocking", "blocking"))
        problem: AssembledDC1DProblem | AssembledIonic1DProblem = AssembledIonic1DProblem(device)
        u = device.initial_state().u
    else:
        electronic = compile_device_1d(load_device(), intervals_per_layer=(4, 4, 4))
        problem = AssembledDC1DProblem(electronic)
        left, right = electronic.contact_potentials(0.0)
        position = (electronic.nodes_m - electronic.nodes_m[0]) / electronic.geometry.extent_m
        u = np.r_[left + (right - left) * position, np.zeros(2 * electronic.node_count)]
    layout = problem.binding.layout
    raw = problem.assemble(u, storage=True)
    mask = layout.constraint_mask()
    mask[layout.rows("potential")] = True
    rows = np.flatnonzero(~mask)
    assert raw.S is not None
    scale = np.maximum(np.abs(raw.S[rows]), 1.0)  # Numerical row scaling, particles/m2.
    model = assembled_inventory_dae(problem, lambda t: OperatingInputs(1e-5 * t, 0.0),
                                    inventory_scale=scale, algebraic_scale=np.ones(np.count_nonzero(mask)))
    assert model.differential_rows == tuple(rows)
    assert len(model.algebraic_rows) == 13 + 4
    if ionic:
        assert len(model.differential_rows) == 2 * 11 + 5
        assert all(row in model.differential_rows for row in range(39, 44))
        assert model.inventory_capacity is not None
        np.testing.assert_array_equal(model.inventory_capacity[-5:],
                                       device.ion_geometry.cell_measures_m * device.positive_ion_site_limit_m3)
    counters = RadauCounters()
    copied = model.assembled(0.0, u, counters)
    for name in ("F", "S"):
        assert getattr(raw, name).tobytes() == getattr(copied, name).tobytes()
    for name in ("J", "S_u"):
        first, second = getattr(raw, name), getattr(copied, name)
        assert (first.data.tobytes(), first.indices.tobytes(), first.indptr.tobytes()) == \
               (second.data.tobytes(), second.indices.tobytes(), second.indptr.tobytes())
    controls = RadauControls(scaled_newton_tolerance=2e-12)
    point = model.reconstruct(0.0, raw.S[rows], u, controls, counters)
    moved = model.reconstruct(1.0, point.inventory, point.u, controls, counters)
    assert moved.scaled_reconstruction_residual <= controls.scaled_newton_tolerance
    assert np.max(np.abs(moved.inventory_residual / scale)) <= controls.scaled_newton_tolerance
    assert np.max(np.abs(moved.algebraic_residual)) <= controls.scaled_newton_tolerance


def test_failed_step_retains_prefix_and_original_cause() -> None:
    model, u = affine_dae(np.array([[-2.0]]), np.array([1.0]))
    original = model.evaluate

    def stopped(time_s: float, value: Vector) -> AssembledSystem:
        if time_s > 0.5:
            raise FloatingPointError("deliberate_unit_failure")
        return original(time_s, value)

    with pytest.raises(RadauFailure) as caught:
        integrate_radau(replace(model, evaluate=stopped), u, [0.0, 0.25, 0.5, 0.75], controls=CONTROLS)
    assert caught.value.failed_step == 2
    assert len(caught.value.completed_steps) == 2
    assert caught.value.__cause__ is not None
    assert str(caught.value.__cause__) == "deliberate_unit_failure"


def test_segment_bridge_holds_inventories_when_the_coordinate_reference_changes() -> None:
    A = np.array([[-2.0]])
    model, u = affine_dae(A, np.array([1.0]))
    left = integrate_radau(model, u, [0.0, 0.25, 0.5], controls=CONTROLS)
    held = left.steps[-1].endpoint.inventory
    after, _ = affine_dae(A, held, coordinate_shift_per_s=1.0)
    right = integrate_radau(after, left.steps[-1].endpoint.u, [0.5, 0.75, 1.0],
                            controls=CONTROLS, initial_inventory=held)
    assert right.initial.inventory.tobytes() == held.tobytes()
    assert right.initial.scaled_reconstruction_residual <= CONTROLS.scaled_newton_tolerance
    np.testing.assert_allclose(right.initial.u[2:], left.steps[-1].endpoint.u[2:] - 0.5,
                               atol=3e-14, rtol=0)
    assert not right.qualified  # No terminal impulse or event certificate is implied.


def test_no_time_route_imports_and_finite_work_controls() -> None:
    tree = ast.parse(inspect.getsource(module))
    imported = [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert not any(name.startswith("solarlab.solvers") for name in imported)
    assert not any("bdf" in name or "backward_euler" in name for name in imported)
    model, u = affine_dae(np.array([[-2.0]]), np.array([1.0]))
    with pytest.raises(ValueError, match="increasing_grid"):
        integrate_radau(model, u, [0, 0])
    with pytest.raises(ValueError, match="step_cap"):
        integrate_radau(model, u, [0, 0.5, 1], controls=RadauControls(max_steps=1))
    with pytest.raises(ValueError, match="positive_row_scales"):
        replace(model, inventory_scale=np.zeros(1))
    assert np.all(u == np.r_[0.0, 0.0, 0.0])


@pytest.mark.parametrize("target", [-1e-100, 0.0, 1.0, np.nextafter(1.0, np.inf)])
def test_newton_tolerance_cannot_bypass_physical_inventory_domain(target: float) -> None:
    model, u = affine_dae(np.array([[-2.0]]), np.array([0.5]))
    bounded = replace(model, inventory_capacity=np.ones(1))
    counters = RadauCounters()
    with pytest.raises(FloatingPointError, match="inventory_outside_physical_domain"):
        bounded.reconstruct(0.0, [target], u, CONTROLS, counters)
    assert counters.assembly_evaluations == 0


def test_collapsed_stage_clocks_are_refused_with_retained_failure() -> None:
    model, u = affine_dae(np.array([[-2.0]]), np.array([1.0]))
    with pytest.raises(RadauFailure, match="stage_clock_resolution") as caught:
        integrate_radau(model, u, [1e16, np.nextafter(1e16, np.inf)], controls=CONTROLS)
    assert caught.value.failed_step == 0 and not caught.value.completed_steps
