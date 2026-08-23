"""Time-discrete reference solver for the research DAE backbone.

This module is intentionally disconnected from production experiment routes.
It supplies a conservative backward-Euler reference for the narrow
no-ion/no-interface DAE capability in :mod:`perovskite_sim.solver.dae`.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Literal

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import LinearOperator, onenormest, splu

from perovskite_sim.constants import Q
from perovskite_sim.solver.dae import (
    DAEConsistentInitialCondition,
    DAEResidualReport,
    NoIonNoInterfaceDAE,
    build_consistent_initial_condition,
    finite_difference_state_jacobian,
    project_algebraic_state,
)
from perovskite_sim.solver.dae_jacobian import build_structured_state_jacobian


class DAEIntegrationError(RuntimeError):
    """A research DAE time step failed its nonlinear certificate."""

    def __init__(
        self,
        message: str,
        *,
        step_index: int,
        time_s: float,
        residual_norm: float,
    ) -> None:
        self.step_index = int(step_index)
        self.time_s = float(time_s)
        self.residual_norm = float(residual_norm)
        super().__init__(
            f"{message} at DAE step {self.step_index}, "
            f"t={self.time_s:.9g} s, residual={self.residual_norm:.6g}"
        )


def _readonly_f64(value: object, *, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite with shape {shape}")
    result = np.array(array, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _max_abs(value: np.ndarray) -> float:
    return float(np.max(np.abs(value), initial=0.0))


def _transient_sha256(
    model: NoIonNoInterfaceDAE,
    time_s: np.ndarray,
    coordinates: np.ndarray,
    physical_states: np.ndarray,
    potentials_V: np.ndarray,
) -> str:
    digest = hashlib.sha256(b"no-ion-no-interface-dae-be-v1")
    for value in (
        model.grid_m,
        time_s,
        coordinates,
        physical_states,
        potentials_V,
    ):
        array = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class DAETimeStepReport:
    """Nonlinear and conservation evidence for one accepted BE step."""

    step_index: int
    time_start_s: float
    time_end_s: float
    dt_s: float
    nonlinear_iterations: int
    residual_evaluations: int
    jacobian_evaluations: int
    line_search_backtracks: int
    log_step_scalings: int
    max_scaled_jacobian_condition: float
    residual_report: DAEResidualReport
    electron_balance_defect_A_m2: float
    hole_balance_defect_A_m2: float


@dataclass(frozen=True, slots=True)
class DAETransientResult:
    """Certified trajectory from the research backward-Euler reference."""

    time_s: np.ndarray
    coordinates: np.ndarray
    physical_states: np.ndarray
    potentials_V: np.ndarray
    step_reports: tuple[DAETimeStepReport, ...]
    success: bool
    method: str
    jacobian_mode: str
    total_nonlinear_iterations: int
    total_residual_evaluations: int
    total_jacobian_evaluations: int
    max_normalized_differential_residual: float
    max_normalized_algebraic_residual: float
    max_electron_balance_defect_A_m2: float
    max_hole_balance_defect_A_m2: float
    trajectory_sha256: str


def _backward_euler_derivative(
    model: NoIonNoInterfaceDAE,
    coordinate: np.ndarray,
    previous_coordinate: np.ndarray,
    dt_s: float,
) -> np.ndarray:
    """Return qdot so ``density*qdot == (density-new - old)/dt``."""
    layout = model.layout
    count = layout.node_count
    derivative = np.zeros(layout.size, dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        electron = -np.expm1(
            previous_coordinate[1 : count - 1]
            - coordinate[1 : count - 1]
        ) / dt_s
        hole = -np.expm1(
            previous_coordinate[count + 1 : 2 * count - 1]
            - coordinate[count + 1 : 2 * count - 1]
        ) / dt_s
    if not np.all(np.isfinite(electron)) or not np.all(np.isfinite(hole)):
        raise ValueError("backward-Euler density ratio overflowed")
    derivative[1 : count - 1] = electron
    derivative[count + 1 : 2 * count - 1] = hole
    return derivative


def _backward_euler_derivative_chain(
    model: NoIonNoInterfaceDAE,
    coordinate: np.ndarray,
    previous_coordinate: np.ndarray,
    dt_s: float,
) -> np.ndarray:
    """Return the diagonal ``d(qdot_BE)/dq_new``."""
    layout = model.layout
    count = layout.node_count
    diagonal = np.zeros(layout.size, dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        diagonal[1 : count - 1] = np.exp(
            previous_coordinate[1 : count - 1]
            - coordinate[1 : count - 1]
        ) / dt_s
        diagonal[count + 1 : 2 * count - 1] = np.exp(
            previous_coordinate[count + 1 : 2 * count - 1]
            - coordinate[count + 1 : 2 * count - 1]
        ) / dt_s
    if not np.all(np.isfinite(diagonal)):
        raise ValueError("backward-Euler derivative chain overflowed")
    return diagonal


def _validate_time_grid(time_s: np.ndarray) -> np.ndarray:
    value = np.asarray(time_s, dtype=float)
    if (
        value.ndim != 1
        or value.size < 2
        or not np.all(np.isfinite(value))
        or value[0] < 0.0
        or np.any(np.diff(value) <= 0.0)
    ):
        raise ValueError(
            "time_s must be a finite, non-negative, strictly increasing vector"
        )
    return value


def _validate_initial_condition(
    model: NoIonNoInterfaceDAE,
    initial: DAEConsistentInitialCondition,
    residual_tolerance: float,
) -> None:
    layout = model.layout
    report = model.residual_report(initial.coordinate, initial.derivative)
    expected_state = model.packed_physical_state(initial.coordinate)
    _n, _p, expected_potential = model.physical_fields(initial.coordinate)
    if not initial.certified:
        raise ValueError("initial DAE condition is not certified")
    if report.max_normalized_residual > residual_tolerance:
        raise ValueError(
            "initial DAE condition exceeds the requested residual tolerance"
        )
    if (
        np.asarray(initial.coordinate).shape != (layout.size,)
        or np.asarray(initial.derivative).shape != (layout.size,)
        or not np.array_equal(initial.physical_state, expected_state)
        or not np.array_equal(initial.potential_V, expected_potential)
    ):
        raise ValueError("initial DAE condition does not belong to this model")


def run_backward_euler_reference(
    model: NoIonNoInterfaceDAE,
    time_s: np.ndarray,
    *,
    initial: DAEConsistentInitialCondition | None = None,
    residual_tolerance: float = 1.0e-9,
    max_newton_iterations: int = 16,
    max_line_search_backtracks: int = 12,
    max_log_density_update: float = 2.0,
    finite_difference_relative_step: float = 1.0e-6,
    jacobian_mode: Literal[
        "dense_central",
        "structured_analytic",
    ] = "dense_central",
) -> DAETransientResult:
    """Integrate the narrow research DAE with conservative backward Euler.

    The nonlinear Jacobian is a deliberately transparent dense reference:
    central differences for differential state rows, exact algebraic state
    rows, and exact chain-rule coupling through ``dF/d(qdot)``. It establishes
    correctness evidence; it is not the planned sparse production algorithm.
    """
    time = _validate_time_grid(time_s)
    scalar_controls = {
        "residual_tolerance": residual_tolerance,
        "max_log_density_update": max_log_density_update,
        "finite_difference_relative_step": finite_difference_relative_step,
    }
    for name, value in scalar_controls.items():
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if not isinstance(max_newton_iterations, int) or max_newton_iterations < 1:
        raise ValueError("max_newton_iterations must be a positive integer")
    if (
        not isinstance(max_line_search_backtracks, int)
        or max_line_search_backtracks < 0
    ):
        raise ValueError(
            "max_line_search_backtracks must be a non-negative integer"
        )
    if jacobian_mode not in ("dense_central", "structured_analytic"):
        raise ValueError(
            "jacobian_mode must be 'dense_central' or 'structured_analytic'"
        )

    initial_state = (
        build_consistent_initial_condition(
            model,
            residual_tolerance=residual_tolerance,
        )
        if initial is None
        else initial
    )
    _validate_initial_condition(model, initial_state, residual_tolerance)

    layout = model.layout
    count = layout.node_count
    coordinates = [np.array(initial_state.coordinate, dtype=float, copy=True)]
    physical_states = [
        np.array(initial_state.physical_state, dtype=float, copy=True)
    ]
    potentials = [np.array(initial_state.potential_V, dtype=float, copy=True)]
    reports: list[DAETimeStepReport] = []
    previous_derivative = np.array(initial_state.derivative, dtype=float, copy=True)

    for step_index, (time_start, time_end) in enumerate(
        zip(time[:-1], time[1:]),
        start=1,
    ):
        dt_s = float(time_end - time_start)
        previous = coordinates[-1]
        prediction_delta = dt_s * previous_derivative
        predictor_carrier_step = _max_abs(prediction_delta[: 2 * count])
        predictor_scaled = predictor_carrier_step > max_log_density_update
        if predictor_scaled:
            prediction_delta *= max_log_density_update / predictor_carrier_step
        prediction = previous + prediction_delta
        prediction[layout.potential_slice] = previous[layout.potential_slice]
        residual_evaluations = 0
        jacobian_evaluations = 0
        line_search_backtracks = 0
        log_step_scalings = int(predictor_scaled)
        worst_condition = 0.0
        try:
            coordinate = project_algebraic_state(model, prediction)
        except ValueError as exc:
            raise DAEIntegrationError(
                f"algebraic predictor projection failed: {exc}",
                step_index=step_index,
                time_s=float(time_end),
                residual_norm=float("inf"),
            ) from exc

        def evaluate(
            candidate: np.ndarray,
        ) -> tuple[np.ndarray, DAEResidualReport]:
            nonlocal residual_evaluations
            derivative = _backward_euler_derivative(
                model,
                candidate,
                previous,
                dt_s,
            )
            report = model.residual_report(candidate, derivative)
            residual_evaluations += 1
            return derivative, report

        try:
            derivative, report = evaluate(coordinate)
        except ValueError as exc:
            raise DAEIntegrationError(
                str(exc),
                step_index=step_index,
                time_s=float(time_end),
                residual_norm=float("inf"),
            ) from exc
        residual_norm = report.max_normalized_residual
        nonlinear_iterations = 0

        while residual_norm > residual_tolerance:
            if nonlinear_iterations >= max_newton_iterations:
                raise DAEIntegrationError(
                    "Newton iterations exhausted",
                    step_index=step_index,
                    time_s=float(time_end),
                    residual_norm=residual_norm,
                )
            try:
                derivative_chain = _backward_euler_derivative_chain(
                    model,
                    coordinate,
                    previous,
                    dt_s,
                )
                if jacobian_mode == "dense_central":
                    state_jacobian = finite_difference_state_jacobian(
                        model,
                        coordinate,
                        derivative,
                        relative_step=finite_difference_relative_step,
                    )
                    residual_evaluations += 2 * layout.size
                    exact_algebraic = model.algebraic_state_jacobian(coordinate)
                    state_jacobian[layout.algebraic_mask] = exact_algebraic[
                        layout.algebraic_mask
                    ]
                    jacobian = state_jacobian + np.diag(
                        model.derivative_jacobian_diagonal(coordinate)
                        * derivative_chain
                    )
                    condition = float(np.linalg.cond(jacobian))
                    if not np.isfinite(condition):
                        raise np.linalg.LinAlgError(
                            "non-finite Jacobian condition"
                        )
                    delta = np.linalg.solve(
                        jacobian,
                        -report.normalized_residual,
                    )
                else:
                    state_jacobian = build_structured_state_jacobian(
                        model,
                        coordinate,
                        derivative,
                    ).matrix
                    jacobian = (
                        state_jacobian
                        + diags(
                            model.derivative_jacobian_diagonal(coordinate)
                            * derivative_chain
                        )
                    ).tocsc()
                    factor = splu(jacobian)
                    inverse = LinearOperator(
                        jacobian.shape,
                        matvec=factor.solve,
                        rmatvec=lambda value: factor.solve(value, trans="T"),
                        dtype=float,
                    )
                    condition = float(
                        onenormest(jacobian) * onenormest(inverse)
                    )
                    if not np.isfinite(condition):
                        raise np.linalg.LinAlgError(
                            "non-finite sparse Jacobian condition estimate"
                        )
                    delta = factor.solve(-report.normalized_residual)
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                raise DAEIntegrationError(
                    f"Jacobian solve failed: {exc}",
                    step_index=step_index,
                    time_s=float(time_end),
                    residual_norm=residual_norm,
                ) from exc
            jacobian_evaluations += 1
            nonlinear_iterations += 1
            worst_condition = max(worst_condition, condition)

            carrier_step = _max_abs(delta[: 2 * count])
            if carrier_step > max_log_density_update:
                delta *= max_log_density_update / carrier_step
                log_step_scalings += 1

            accepted = False
            for backtrack in range(max_line_search_backtracks + 1):
                alpha = 0.5**backtrack
                candidate = coordinate + alpha * delta
                try:
                    candidate_derivative, candidate_report = evaluate(candidate)
                    candidate_norm = candidate_report.max_normalized_residual
                except ValueError:
                    candidate_norm = float("inf")
                if (
                    candidate_norm <= residual_tolerance
                    or candidate_norm <= residual_norm * (1.0 - 1.0e-4 * alpha)
                ):
                    coordinate = candidate
                    derivative = candidate_derivative
                    report = candidate_report
                    residual_norm = candidate_norm
                    line_search_backtracks += backtrack
                    accepted = True
                    break
            if not accepted:
                raise DAEIntegrationError(
                    "Newton line search stalled",
                    step_index=step_index,
                    time_s=float(time_end),
                    residual_norm=residual_norm,
                )

        electron_balance = Q * abs(
            float(
                np.dot(
                    report.electron_rate_residual_m3_s,
                    model.material.poisson_factor.h_cell,
                )
            )
        )
        hole_balance = Q * abs(
            float(
                np.dot(
                    report.hole_rate_residual_m3_s,
                    model.material.poisson_factor.h_cell,
                )
            )
        )
        reports.append(
            DAETimeStepReport(
                step_index=step_index,
                time_start_s=float(time_start),
                time_end_s=float(time_end),
                dt_s=dt_s,
                nonlinear_iterations=nonlinear_iterations,
                residual_evaluations=residual_evaluations,
                jacobian_evaluations=jacobian_evaluations,
                line_search_backtracks=line_search_backtracks,
                log_step_scalings=log_step_scalings,
                max_scaled_jacobian_condition=worst_condition,
                residual_report=report,
                electron_balance_defect_A_m2=electron_balance,
                hole_balance_defect_A_m2=hole_balance,
            )
        )
        coordinates.append(np.array(coordinate, copy=True))
        physical_states.append(model.packed_physical_state(coordinate))
        _n, _p, potential = model.physical_fields(coordinate)
        potentials.append(np.array(potential, copy=True))
        previous_derivative = derivative

    coordinate_array = np.stack(coordinates)
    physical_array = np.stack(physical_states)
    potential_array = np.stack(potentials)
    time_ro = _readonly_f64(time, shape=(time.size,), name="DAE time grid")
    coordinate_ro = _readonly_f64(
        coordinate_array,
        shape=(time.size, layout.size),
        name="DAE coordinates",
    )
    physical_ro = _readonly_f64(
        physical_array,
        shape=(time.size, 3 * count),
        name="DAE physical states",
    )
    potential_ro = _readonly_f64(
        potential_array,
        shape=(time.size, count),
        name="DAE potentials",
    )
    report_tuple = tuple(reports)
    return DAETransientResult(
        time_s=time_ro,
        coordinates=coordinate_ro,
        physical_states=physical_ro,
        potentials_V=potential_ro,
        step_reports=report_tuple,
        success=True,
        method=(
            "physical-density-backward-euler/dense-hybrid-newton-v1"
            if jacobian_mode == "dense_central"
            else "physical-density-backward-euler/sparse-analytic-newton-v1"
        ),
        jacobian_mode=jacobian_mode,
        total_nonlinear_iterations=sum(
            item.nonlinear_iterations for item in report_tuple
        ),
        total_residual_evaluations=sum(
            item.residual_evaluations for item in report_tuple
        ),
        total_jacobian_evaluations=sum(
            item.jacobian_evaluations for item in report_tuple
        ),
        max_normalized_differential_residual=max(
            (
                item.residual_report.max_normalized_differential_residual
                for item in report_tuple
            ),
            default=0.0,
        ),
        max_normalized_algebraic_residual=max(
            (
                item.residual_report.max_normalized_algebraic_residual
                for item in report_tuple
            ),
            default=0.0,
        ),
        max_electron_balance_defect_A_m2=max(
            (item.electron_balance_defect_A_m2 for item in report_tuple),
            default=0.0,
        ),
        max_hole_balance_defect_A_m2=max(
            (item.hole_balance_defect_A_m2 for item in report_tuple),
            default=0.0,
        ),
        trajectory_sha256=_transient_sha256(
            model,
            time_ro,
            coordinate_ro,
            physical_ro,
            potential_ro,
        ),
    )


__all__ = [
    "DAEIntegrationError",
    "DAETimeStepReport",
    "DAETransientResult",
    "run_backward_euler_reference",
]
