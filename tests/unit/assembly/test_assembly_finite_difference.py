"""Finite-difference checks of the assembled Jacobian and storage tangent.

The small research mesh (14 nodes) is evaluated at accepted dark and
illuminated points and at seeded perturbations, for fixed-reservoir,
finite-rate/blocking and field-dependent-transport bindings. Each column is
a fourth-order Richardson central difference of ``F`` (and ``S``) along one
coordinate. An entry error is normalized by its row's largest Jacobian
entry, the scale of the cancellation inside one balance row.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray
import pytest
from threadpoolctl import threadpool_limits

from examples.research_1d.field_transport import prepare_case as field_transport_case
from examples.research_1d.steady import load_device
from solarlab.config.resolve_device import resolve_device
from solarlab.device.compile_1d import compile_device_1d
from solarlab.device.inputs import DeviceInput
from solarlab.equations.assembly.classical_1d import AssembledDC1DProblem, classical_dc_1d
from solarlab.equations.dc_1d import CompiledDevice1D
from solarlab.solvers.nonlinear.steady_1d import solve_equilibrium, solve_steady_state

Vector: TypeAlias = NDArray[np.float64]
Matrix: TypeAlias = NDArray[np.float64]
BLOCKS = ("potential", "electron", "hole")
# Richardson-extrapolated central differences of smooth rows reach ~1e-9 of
# the row scale here; a missing or wrong-sign tangent term is O(1).
ROW_TOLERANCE = 1.0e-6


@pytest.fixture(scope="module", autouse=True)
def blas1() -> Iterator[None]:
    with threadpool_limits(limits=1):
        yield


def small(kind: str) -> CompiledDevice1D:
    if kind == "ct":
        return compile_device_1d(field_transport_case(), intervals_per_layer=(4, 4, 4))
    prepared = load_device()
    document = prepared.to_input().editing_data()
    if kind == "robin":
        document["settings"]["flat_band_contacts"] = True
    elif kind == "selective":
        document["contacts"] = [dict(id="left", side="left", layer="htl", S_n=0.0, S_p=200.0),
                                dict(id="right", side="right", layer="etl", S_n=200.0, S_p=0.0)]
    if kind != "fixed":
        prepared = resolve_device(DeviceInput.model_validate(document), prepared.defaults,
                                  prepared.resources, sources=prepared.sources)
    return compile_device_1d(prepared, intervals_per_layer=4)


def richardson(function: Callable[[Vector], Vector], u: Vector, steps: Vector) -> Matrix:
    """Fourth-order central-difference Jacobian, one coordinate at a time."""
    columns = []
    for index, step in enumerate(steps):
        def central(h: float) -> Vector:
            plus, minus = u.copy(), u.copy()
            plus[index] += h
            minus[index] -= h
            result: Vector = (function(plus) - function(minus)) / (2.0 * h)
            return result
        columns.append((4.0 * central(0.5 * step) - central(step)) / 3.0)
    return np.column_stack(columns)


def row_errors(exact: Matrix, estimate: Matrix, count: int) -> dict[tuple[str, str], float]:
    """Max |exact - estimate| per block pair, normalized by each row's largest entry."""
    scale = np.max(np.abs(exact), axis=1)
    scale[scale == 0.0] = 1.0
    normalized = np.abs(exact - estimate) / scale[:, None]
    return {(row, column): float(np.max(normalized[i * count:(i + 1) * count, j * count:(j + 1) * count]))
            for i, row in enumerate(BLOCKS) for j, column in enumerate(BLOCKS)}


def states(device: CompiledDevice1D) -> list[tuple[str, Vector, float, float]]:
    dark = solve_equilibrium(device)
    light = solve_steady_state(device, voltage_V=0.4, generation_scale=1.0)
    rng = np.random.default_rng(20261029)
    count = device.node_count
    scale = np.r_[np.full(count, device.thermal_voltage_V), np.ones(2 * count)]
    return [
        ("dark_V0", dark.state.u, 0.0, 0.0),
        ("light_V0.4", light.state.u, 0.4, 1.0),
        ("dark_perturbed", dark.state.u + 0.05 * scale * rng.standard_normal(3 * count), 0.0, 0.0),
        ("light_perturbed", light.state.u + 0.05 * scale * rng.standard_normal(3 * count), 0.4, 1.0),
    ]


@pytest.mark.parametrize("kind", ["fixed", "robin", "selective", "ct"])
def test_jacobian_and_storage_tangent_match_finite_differences(kind: str) -> None:
    device = small(kind)
    system = classical_dc_1d(device)
    count = device.node_count
    steps = np.r_[np.full(count, 1.0e-5 * device.thermal_voltage_V), np.full(2 * count, 1.0e-5)]
    worst: dict[tuple[str, str], float] = {}
    for _label, u, voltage, light in states(device):
        problem = AssembledDC1DProblem(device, voltage, light, system)
        J = problem.jacobian(u).toarray()
        FD = richardson(problem.residual, u, steps)
        S_u = problem.storage_tangent(u).toarray()
        FD_S = richardson(problem.storage, u, steps)
        for errors in (row_errors(J, FD, count), row_errors(S_u, FD_S, count)):
            for key, value in errors.items():
                worst[key] = max(worst.get(key, 0.0), value)
        assert np.all(np.isfinite(FD))
    assert max(worst.values()) < ROW_TOLERANCE, worst


def test_storage_increment_is_the_exact_storage_difference() -> None:
    """The cancellation-free increment equals S(u) - S(u_old) and has slope S_u."""
    device = small("fixed")
    system = classical_dc_1d(device)
    old = solve_steady_state(device, voltage_V=0.4, generation_scale=1.0)
    old_problem = AssembledDC1DProblem(device, 0.4, 1.0, system)
    problem = old_problem.at(0.45, 1.0)
    rng = np.random.default_rng(7)
    u = device.rebase_coordinates(old.state.u, 0.4, 0.45) + 1.0e-3 * rng.standard_normal(old.state.u.size)
    increment = problem.storage_increment(u, old_problem, old.state.u)
    difference = problem.storage(u) - old_problem.storage(old.state.u)
    scale = np.maximum(np.abs(problem.storage(u)), np.abs(old_problem.storage(old.state.u)))
    # The subtraction inherits the rounding of each density's exponent
    # (|exponent| <= ~50, so <= ~1.1e-14 of the inventory); the increment
    # forms the exponent difference once and does not cancel.
    np.testing.assert_array_less(np.abs(increment - difference), 1.0e-13 * scale + 1e-300)
    count = device.node_count
    steps = np.r_[np.full(count, 1.0e-5 * device.thermal_voltage_V), np.full(2 * count, 1.0e-5)]
    slope = richardson(lambda v: problem.storage_increment(v, old_problem, old.state.u), u, steps)
    errors = row_errors(problem.storage_tangent(u).toarray(), slope, count)
    assert max(errors.values()) < ROW_TOLERANCE, errors
