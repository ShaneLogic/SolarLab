"""Backward-Euler references for the algebraic interface-state DAE."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Literal

import numpy as np
from scipy.sparse.linalg import LinearOperator, onenormest, splu

from perovskite_sim.constants import Q
from perovskite_sim.solver.dae_interface_jacobian import (
    build_algebraic_interface_structured_backward_euler_jacobian,
)
from perovskite_sim.solver.dae_interface_states import (
    AlgebraicInterfaceDAEConsistentInitialCondition,
    AlgebraicInterfaceDAEResidualReport,
    AlgebraicInterfaceStateDAE,
    build_algebraic_interface_consistent_initial_condition,
    finite_difference_state_jacobian,
    project_algebraic_interface_state,
)


class AlgebraicInterfaceDAEIntegrationError(RuntimeError):
    """A research algebraic-interface DAE step failed its certificate."""

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
            f"{message} at algebraic-interface DAE step {self.step_index}, "
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


def _trajectory_sha256(
    model: AlgebraicInterfaceStateDAE,
    time_s: np.ndarray,
    coordinates: np.ndarray,
    physical_states: np.ndarray,
    interface_states_m3: np.ndarray,
    potentials_V: np.ndarray,
) -> str:
    digest = hashlib.sha256(b"algebraic-interface-state-dae-be-v1")
    for value in (
        model.grid_m,
        time_s,
        coordinates,
        physical_states,
        interface_states_m3,
        potentials_V,
    ):
        array = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class AlgebraicInterfaceDAETimeStepReport:
    """Nonlinear, balance, and interface evidence for one accepted BE step."""

    step_index: int
    time_start_s: float
    time_end_s: float
    dt_s: float
    nonlinear_iterations: int
    residual_evaluations: int
    jacobian_evaluations: int
    line_search_backtracks: int
    coordinate_step_scalings: int
    max_scaled_jacobian_condition: float
    residual_report: AlgebraicInterfaceDAEResidualReport
    electron_balance_defect_A_m2: float
    hole_balance_defect_A_m2: float
    interface_state_balance_m2_s: float


@dataclass(frozen=True, slots=True)
class AlgebraicInterfaceDAETransientResult:
    """Certified trajectory for the research algebraic-interface DAE slice."""

    time_s: np.ndarray
    coordinates: np.ndarray
    physical_states: np.ndarray
    interface_states_m3: np.ndarray
    potentials_V: np.ndarray
    step_reports: tuple[AlgebraicInterfaceDAETimeStepReport, ...]
    success: bool
    method: str
    jacobian_mode: str
    total_nonlinear_iterations: int
    total_residual_evaluations: int
    total_jacobian_evaluations: int
    max_normalized_carrier_residual: float
    max_normalized_interface_residual: float
    max_normalized_algebraic_residual: float
    max_electron_balance_defect_A_m2: float
    max_hole_balance_defect_A_m2: float
    max_interface_state_balance_m2_s: float
    trajectory_sha256: str


def _backward_euler_derivative(
    model: AlgebraicInterfaceStateDAE,
    coordinate: np.ndarray,
    previous_coordinate: np.ndarray,
    dt_s: float,
) -> np.ndarray:
    """Return qdot giving exact physical carrier-density BE storage."""
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
        raise ValueError("backward-Euler carrier density ratio overflowed")
    derivative[1 : count - 1] = electron
    derivative[count + 1 : 2 * count - 1] = hole
    return derivative


def _backward_euler_derivative_chain(
    model: AlgebraicInterfaceStateDAE,
    coordinate: np.ndarray,
    previous_coordinate: np.ndarray,
    dt_s: float,
) -> np.ndarray:
    """Return diagonal ``d(qdot_BE)/dq_new`` for carrier coordinates."""
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
    model: AlgebraicInterfaceStateDAE,
    initial: AlgebraicInterfaceDAEConsistentInitialCondition,
    residual_tolerance: float,
) -> None:
    layout = model.layout
    report = model.residual_report(initial.coordinate, initial.derivative)
    expected_state = model.packed_physical_state(initial.coordinate)
    _n, _p, expected_interface, expected_potential = model.physical_fields(
        initial.coordinate
    )
    if not initial.certified:
        raise ValueError("initial algebraic-interface DAE condition is not certified")
    if report.max_normalized_residual > residual_tolerance:
        raise ValueError(
            "initial algebraic-interface DAE condition exceeds residual tolerance"
        )
    if (
        np.asarray(initial.coordinate).shape != (layout.size,)
        or np.asarray(initial.derivative).shape != (layout.size,)
        or not np.array_equal(initial.physical_state, expected_state)
        or not np.array_equal(initial.interface_state_m3, expected_interface)
        or not np.array_equal(initial.potential_V, expected_potential)
    ):
        raise ValueError(
            "initial algebraic-interface DAE condition does not belong to model"
        )


def run_algebraic_interface_backward_euler_reference(
    model: AlgebraicInterfaceStateDAE,
    time_s: np.ndarray,
    *,
    initial: AlgebraicInterfaceDAEConsistentInitialCondition | None = None,
    residual_tolerance: float = 1.0e-9,
    max_newton_iterations: int = 20,
    max_line_search_backtracks: int = 12,
    max_log_density_update: float = 2.0,
    max_interface_logit_update: float = 2.0,
    finite_difference_relative_step: float = 1.0e-6,
    jacobian_mode: Literal[
        "dense_central",
        "structured_analytic",
    ] = "dense_central",
) -> AlgebraicInterfaceDAETransientResult:
    """Integrate the research slice with physical-density backward Euler."""
    time = _validate_time_grid(time_s)
    scalar_controls = {
        "residual_tolerance": residual_tolerance,
        "max_log_density_update": max_log_density_update,
        "max_interface_logit_update": max_interface_logit_update,
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
        build_algebraic_interface_consistent_initial_condition(
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
    interface_states = [
        np.array(initial_state.interface_state_m3, dtype=float, copy=True)
    ]
    potentials = [np.array(initial_state.potential_V, dtype=float, copy=True)]
    reports: list[AlgebraicInterfaceDAETimeStepReport] = []
    previous_derivative = np.array(initial_state.derivative, dtype=float, copy=True)

    for step_index, (time_start, time_end) in enumerate(
        zip(time[:-1], time[1:]),
        start=1,
    ):
        dt_s = float(time_end - time_start)
        previous = coordinates[-1]
        prediction_delta = dt_s * previous_derivative
        carrier_prediction = _max_abs(prediction_delta[: 2 * count])
        predictor_scaled = carrier_prediction > max_log_density_update
        if predictor_scaled:
            prediction_delta *= max_log_density_update / carrier_prediction
        prediction = previous + prediction_delta
        prediction[layout.interface_slice] = previous[layout.interface_slice]
        prediction[layout.potential_slice] = previous[layout.potential_slice]
        residual_evaluations = 0
        jacobian_evaluations = 0
        line_search_backtracks = 0
        coordinate_step_scalings = int(predictor_scaled)
        worst_condition = 0.0
        try:
            coordinate = project_algebraic_interface_state(model, prediction)
        except (ValueError, RuntimeError) as exc:
            raise AlgebraicInterfaceDAEIntegrationError(
                f"algebraic predictor projection failed: {exc}",
                step_index=step_index,
                time_s=float(time_end),
                residual_norm=float("inf"),
            ) from exc

        def evaluate(
            candidate: np.ndarray,
        ) -> tuple[np.ndarray, AlgebraicInterfaceDAEResidualReport]:
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
            raise AlgebraicInterfaceDAEIntegrationError(
                str(exc),
                step_index=step_index,
                time_s=float(time_end),
                residual_norm=float("inf"),
            ) from exc
        residual_norm = report.max_normalized_residual
        nonlinear_iterations = 0

        while residual_norm > residual_tolerance:
            if nonlinear_iterations >= max_newton_iterations:
                raise AlgebraicInterfaceDAEIntegrationError(
                    "Newton iterations exhausted",
                    step_index=step_index,
                    time_s=float(time_end),
                    residual_norm=residual_norm,
                )
            try:
                if jacobian_mode == "dense_central":
                    derivative_chain = _backward_euler_derivative_chain(
                        model,
                        coordinate,
                        previous,
                        dt_s,
                    )
                    state_jacobian = finite_difference_state_jacobian(
                        model,
                        coordinate,
                        derivative,
                        relative_step=finite_difference_relative_step,
                    )
                    residual_evaluations += 2 * layout.size
                    exact_rows = model.boundary_poisson_state_jacobian(coordinate)
                    exact_mask = np.zeros(layout.size, dtype=bool)
                    exact_mask[[0, count - 1, count, 2 * count - 1]] = True
                    exact_mask[layout.potential_slice] = True
                    state_jacobian[exact_mask] = exact_rows[exact_mask]
                    derivative_diagonal = np.diag(
                        model.derivative_jacobian(coordinate)
                    )
                    jacobian = state_jacobian + np.diag(
                        derivative_diagonal * derivative_chain
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
                    jacobian = (
                        build_algebraic_interface_structured_backward_euler_jacobian(
                            model,
                            coordinate,
                            dt_s,
                        ).matrix.tocsc()
                    )
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
                raise AlgebraicInterfaceDAEIntegrationError(
                    f"Jacobian solve failed: {exc}",
                    step_index=step_index,
                    time_s=float(time_end),
                    residual_norm=residual_norm,
                ) from exc
            jacobian_evaluations += 1
            nonlinear_iterations += 1
            worst_condition = max(worst_condition, condition)

            carrier_step = _max_abs(delta[: 2 * count])
            interface_step = _max_abs(delta[layout.interface_slice])
            scale = 1.0
            if carrier_step > max_log_density_update:
                scale = min(scale, max_log_density_update / carrier_step)
            if interface_step > max_interface_logit_update:
                scale = min(scale, max_interface_logit_update / interface_step)
            if scale < 1.0:
                delta *= scale
                coordinate_step_scalings += 1

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
                    or candidate_norm
                    <= residual_norm * (1.0 - 1.0e-4 * alpha)
                ):
                    coordinate = candidate
                    derivative = candidate_derivative
                    report = candidate_report
                    residual_norm = candidate_norm
                    line_search_backtracks += backtrack
                    accepted = True
                    break
            if not accepted:
                raise AlgebraicInterfaceDAEIntegrationError(
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
        interface_balance = _max_abs(
            report.interface_state_flux_residual_m2_s
        )
        reports.append(
            AlgebraicInterfaceDAETimeStepReport(
                step_index=step_index,
                time_start_s=float(time_start),
                time_end_s=float(time_end),
                dt_s=dt_s,
                nonlinear_iterations=nonlinear_iterations,
                residual_evaluations=residual_evaluations,
                jacobian_evaluations=jacobian_evaluations,
                line_search_backtracks=line_search_backtracks,
                coordinate_step_scalings=coordinate_step_scalings,
                max_scaled_jacobian_condition=worst_condition,
                residual_report=report,
                electron_balance_defect_A_m2=electron_balance,
                hole_balance_defect_A_m2=hole_balance,
                interface_state_balance_m2_s=interface_balance,
            )
        )
        coordinates.append(np.array(coordinate, copy=True))
        physical_states.append(model.packed_physical_state(coordinate))
        _n, _p, interface_state, potential = model.physical_fields(coordinate)
        interface_states.append(np.array(interface_state, copy=True))
        potentials.append(np.array(potential, copy=True))
        previous_derivative = derivative

    coordinate_array = np.stack(coordinates)
    physical_array = np.stack(physical_states)
    interface_array = np.stack(interface_states)
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
    interface_ro = _readonly_f64(
        interface_array,
        shape=(time.size, layout.interface_state_count),
        name="DAE interface states",
    )
    potential_ro = _readonly_f64(
        potential_array,
        shape=(time.size, count),
        name="DAE potentials",
    )
    report_tuple = tuple(reports)
    return AlgebraicInterfaceDAETransientResult(
        time_s=time_ro,
        coordinates=coordinate_ro,
        physical_states=physical_ro,
        interface_states_m3=interface_ro,
        potentials_V=potential_ro,
        step_reports=report_tuple,
        success=True,
        method=(
            "physical-density-backward-euler/dense-central-newton-v1"
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
        max_normalized_carrier_residual=max(
            (
                item.residual_report.max_normalized_carrier_residual
                for item in report_tuple
            ),
            default=0.0,
        ),
        max_normalized_interface_residual=max(
            (
                item.residual_report.max_normalized_interface_residual
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
        max_interface_state_balance_m2_s=max(
            (item.interface_state_balance_m2_s for item in report_tuple),
            default=0.0,
        ),
        trajectory_sha256=_trajectory_sha256(
            model,
            time_ro,
            coordinate_ro,
            physical_ro,
            interface_ro,
            potential_ro,
        ),
    )


__all__ = [
    "AlgebraicInterfaceDAEIntegrationError",
    "AlgebraicInterfaceDAETimeStepReport",
    "AlgebraicInterfaceDAETransientResult",
    "run_algebraic_interface_backward_euler_reference",
]
