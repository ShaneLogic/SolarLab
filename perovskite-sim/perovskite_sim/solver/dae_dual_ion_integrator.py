"""Conservative time reference for the dual-mobile-ion research DAE."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np

from perovskite_sim.constants import Q
from perovskite_sim.physics.generation import dual_cell_integral
from perovskite_sim.solver.dae_dual_ions import (
    DualIonDAE,
    DualIonDAEConsistentInitialCondition,
    DualIonDAEResidualReport,
    build_dual_ion_consistent_initial_condition,
    project_dual_ion_algebraic_state,
)
from perovskite_sim.solver.dae_integrator import DAEIntegrationError


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
    model: DualIonDAE,
    time_s: np.ndarray,
    coordinates: np.ndarray,
    physical_states: np.ndarray,
    potentials_V: np.ndarray,
) -> str:
    digest = hashlib.sha256(b"dual-mobile-ion-dae-be-v1")
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


def _independent_logistic_density_difference_m3(
    new_density_m3: np.ndarray,
    old_density_m3: np.ndarray,
    limit_m3: np.ndarray,
    coordinate_delta: np.ndarray,
) -> np.ndarray:
    theta = new_density_m3 / limit_m3
    old_theta = old_density_m3 / limit_m3
    positive = (
        theta
        * (1.0 - old_theta)
        * (-np.expm1(-np.maximum(coordinate_delta, 0.0)))
    )
    negative = (
        -old_theta
        * (1.0 - theta)
        * (-np.expm1(np.minimum(coordinate_delta, 0.0)))
    )
    return limit_m3 * np.where(coordinate_delta >= 0.0, positive, negative)


def dual_ion_density_difference_m3(
    model: DualIonDAE,
    coordinate: np.ndarray,
    previous_coordinate: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return stable physical ``(Delta P+, Delta P-)`` for one BE step."""
    layout = model.layout
    value = np.asarray(coordinate, dtype=float)
    previous = np.asarray(previous_coordinate, dtype=float)
    if value.shape != (layout.size,) or previous.shape != (layout.size,):
        raise ValueError("coordinates must match the dual-ion DAE layout")
    _n, _p, positive, negative, _phi = model.physical_fields(value)
    _old_n, _old_p, old_positive, old_negative, _old_phi = model.physical_fields(
        previous
    )
    positive_delta = (
        value[layout.positive_ion_slice]
        - previous[layout.positive_ion_slice]
    )
    negative_delta = (
        value[layout.negative_ion_slice]
        - previous[layout.negative_ion_slice]
    )
    if layout.shared_site:
        limit = layout.positive_ion_site_limit_m3
        old_positive_fraction = old_positive / limit
        old_negative_fraction = old_negative / limit
        with np.errstate(over="ignore", invalid="ignore"):
            positive_excess = np.expm1(positive_delta)
            negative_excess = np.expm1(negative_delta)
        denominator = (
            1.0
            + old_positive_fraction * positive_excess
            + old_negative_fraction * negative_excess
        )
        positive_difference = (
            limit
            * old_positive_fraction
            * (
                (1.0 - old_positive_fraction) * positive_excess
                - old_negative_fraction * negative_excess
            )
            / denominator
        )
        negative_difference = (
            limit
            * old_negative_fraction
            * (
                (1.0 - old_negative_fraction) * negative_excess
                - old_positive_fraction * positive_excess
            )
            / denominator
        )
    else:
        positive_difference = _independent_logistic_density_difference_m3(
            positive,
            old_positive,
            layout.positive_ion_site_limit_m3,
            positive_delta,
        )
        negative_difference = _independent_logistic_density_difference_m3(
            negative,
            old_negative,
            layout.negative_ion_site_limit_m3,
            negative_delta,
        )
    arrays = (positive_difference, negative_difference)
    if any(
        value.shape != (layout.node_count,) or not np.all(np.isfinite(value))
        for value in arrays
    ):
        raise ValueError("dual-ion density difference is non-finite")
    return positive_difference, negative_difference


@dataclass(frozen=True, slots=True)
class DualIonDAETimeStepReport:
    """Nonlinear, local-balance, and inventory evidence for one step."""

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
    residual_report: DualIonDAEResidualReport
    electron_balance_defect_A_m2: float
    hole_balance_defect_A_m2: float
    positive_ion_balance_defect_m2_s: float
    negative_ion_balance_defect_m2_s: float
    positive_ion_rhs_inventory_rate_m2_s: float
    negative_ion_rhs_inventory_rate_m2_s: float
    positive_ion_inventory_m2: float
    negative_ion_inventory_m2: float
    positive_ion_inventory_change_m2: float
    negative_ion_inventory_change_m2: float
    minimum_site_vacancy_fraction: float


@dataclass(frozen=True, slots=True)
class DualIonDAETransientResult:
    """Certified trajectory from the dual-ion dense BE reference."""

    time_s: np.ndarray
    coordinates: np.ndarray
    physical_states: np.ndarray
    potentials_V: np.ndarray
    step_reports: tuple[DualIonDAETimeStepReport, ...]
    success: bool
    method: str
    jacobian_mode: str
    total_nonlinear_iterations: int
    total_residual_evaluations: int
    total_jacobian_evaluations: int
    max_normalized_carrier_residual: float
    max_normalized_positive_ion_residual: float
    max_normalized_negative_ion_residual: float
    max_normalized_differential_residual: float
    max_normalized_algebraic_residual: float
    max_electron_balance_defect_A_m2: float
    max_hole_balance_defect_A_m2: float
    max_positive_ion_balance_defect_m2_s: float
    max_negative_ion_balance_defect_m2_s: float
    max_positive_ion_rhs_inventory_rate_m2_s: float
    max_negative_ion_rhs_inventory_rate_m2_s: float
    initial_positive_ion_inventory_m2: float
    terminal_positive_ion_inventory_m2: float
    initial_negative_ion_inventory_m2: float
    terminal_negative_ion_inventory_m2: float
    max_relative_positive_ion_inventory_drift: float
    max_relative_negative_ion_inventory_drift: float
    minimum_site_vacancy_fraction: float
    trajectory_sha256: str


def dual_ion_backward_euler_derivative(
    model: DualIonDAE,
    coordinate: np.ndarray,
    previous_coordinate: np.ndarray,
    dt_s: float,
) -> np.ndarray:
    """Map exact physical-density BE storage into DAE coordinate rates."""
    if not np.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError("dt_s must be finite and positive")
    layout = model.layout
    value = np.asarray(coordinate, dtype=float)
    previous = np.asarray(previous_coordinate, dtype=float)
    if value.shape != (layout.size,) or previous.shape != (layout.size,):
        raise ValueError("BE coordinates must match the dual-ion DAE layout")
    count = layout.node_count
    derivative = np.zeros(layout.size, dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        derivative[1 : count - 1] = -np.expm1(
            previous[1 : count - 1] - value[1 : count - 1]
        ) / dt_s
        derivative[count + 1 : 2 * count - 1] = -np.expm1(
            previous[count + 1 : 2 * count - 1]
            - value[count + 1 : 2 * count - 1]
        ) / dt_s
    positive_difference, negative_difference = dual_ion_density_difference_m3(
        model,
        value,
        previous,
    )
    physical_rate = np.stack(
        (positive_difference, negative_difference),
        axis=1,
    ) / dt_s
    try:
        ion_rate = np.linalg.solve(
            model.ion_coordinate_jacobian_m3(value),
            physical_rate[..., None],
        )[..., 0]
    except np.linalg.LinAlgError as exc:
        raise ValueError("dual-ion BE coordinate mass matrix is singular") from exc
    derivative[layout.positive_ion_slice] = ion_rate[:, 0]
    derivative[layout.negative_ion_slice] = ion_rate[:, 1]
    if not np.all(np.isfinite(derivative)):
        raise ValueError("dual-ion backward-Euler coordinate rate is non-finite")
    return derivative


def finite_difference_dual_ion_backward_euler_jacobian(
    model: DualIonDAE,
    coordinate: np.ndarray,
    previous_coordinate: np.ndarray,
    dt_s: float,
    *,
    relative_step: float = 1.0e-6,
) -> np.ndarray:
    """Independent central Jacobian of the complete time-discrete residual."""
    if not np.isfinite(relative_step) or relative_step <= 0.0:
        raise ValueError("relative_step must be finite and positive")
    value = np.asarray(coordinate, dtype=float)
    previous = np.asarray(previous_coordinate, dtype=float)
    layout = model.layout
    if value.shape != (layout.size,) or previous.shape != (layout.size,):
        raise ValueError("BE coordinates must match the dual-ion DAE layout")
    result = np.empty((layout.size, layout.size), dtype=float)
    potential_start = layout.potential_slice.start
    assert potential_start is not None
    for column in range(layout.size):
        scale = 1.0 if column < potential_start else layout.potential_scale_V
        step = relative_step * max(abs(value[column]), scale)
        plus = value.copy()
        minus = value.copy()
        plus[column] += step
        minus[column] -= step
        plus_rate = dual_ion_backward_euler_derivative(
            model,
            plus,
            previous,
            dt_s,
        )
        minus_rate = dual_ion_backward_euler_derivative(
            model,
            minus,
            previous,
            dt_s,
        )
        result[:, column] = (
            model.residual(plus, plus_rate) - model.residual(minus, minus_rate)
        ) / (2.0 * step)
    return result


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
    model: DualIonDAE,
    initial: DualIonDAEConsistentInitialCondition,
    residual_tolerance: float,
) -> None:
    layout = model.layout
    report = model.residual_report(initial.coordinate, initial.derivative)
    expected_state = model.packed_physical_state(initial.coordinate)
    _n, _p, _positive, _negative, expected_potential = model.physical_fields(
        initial.coordinate
    )
    if not initial.certified:
        raise ValueError("initial dual-ion DAE condition is not certified")
    if report.max_normalized_residual > residual_tolerance:
        raise ValueError(
            "initial dual-ion DAE condition exceeds the residual tolerance"
        )
    if (
        np.asarray(initial.coordinate).shape != (layout.size,)
        or np.asarray(initial.derivative).shape != (layout.size,)
        or not np.array_equal(initial.physical_state, expected_state)
        or not np.array_equal(initial.potential_V, expected_potential)
    ):
        raise ValueError("initial dual-ion DAE condition does not belong to model")


def _step_scale(
    delta: np.ndarray,
    model: DualIonDAE,
    *,
    max_log_density_update: float,
    max_ion_coordinate_update: float,
) -> float:
    count = model.layout.node_count
    carrier_step = _max_abs(delta[: 2 * count])
    ion_step = max(
        _max_abs(delta[model.layout.positive_ion_slice]),
        _max_abs(delta[model.layout.negative_ion_slice]),
    )
    scale = 1.0
    if carrier_step > max_log_density_update:
        scale = min(scale, max_log_density_update / carrier_step)
    if ion_step > max_ion_coordinate_update:
        scale = min(scale, max_ion_coordinate_update / ion_step)
    return scale


def _minimum_site_vacancy_fraction(
    model: DualIonDAE,
    positive_ion: np.ndarray,
    negative_ion: np.ndarray,
) -> float:
    layout = model.layout
    if layout.shared_site:
        vacancy = 1.0 - (
            positive_ion + negative_ion
        ) / layout.positive_ion_site_limit_m3
    else:
        vacancy = np.minimum(
            1.0 - positive_ion / layout.positive_ion_site_limit_m3,
            1.0 - negative_ion / layout.negative_ion_site_limit_m3,
        )
    result = float(np.min(vacancy))
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError("dual-ion site vacancy fraction is not positive")
    return result


def run_dual_ion_backward_euler_reference(
    model: DualIonDAE,
    time_s: np.ndarray,
    *,
    initial: DualIonDAEConsistentInitialCondition | None = None,
    residual_tolerance: float = 1.0e-9,
    max_newton_iterations: int = 20,
    max_line_search_backtracks: int = 12,
    max_log_density_update: float = 2.0,
    max_ion_coordinate_update: float = 2.0,
    finite_difference_relative_step: float = 1.0e-6,
) -> DualIonDAETransientResult:
    """Integrate the narrow dual-ion DAE with dense conservative BE."""
    time = _validate_time_grid(time_s)
    scalar_controls = {
        "residual_tolerance": residual_tolerance,
        "max_log_density_update": max_log_density_update,
        "max_ion_coordinate_update": max_ion_coordinate_update,
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

    initial_state = (
        build_dual_ion_consistent_initial_condition(
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
    reports: list[DualIonDAETimeStepReport] = []
    previous_derivative = np.array(initial_state.derivative, dtype=float, copy=True)
    initial_fields = model.physical_fields(initial_state.coordinate)
    initial_positive_inventory = dual_cell_integral(model.grid_m, initial_fields[2])
    initial_negative_inventory = dual_cell_integral(model.grid_m, initial_fields[3])
    positive_inventory_scale = max(
        abs(initial_positive_inventory),
        np.finfo(float).tiny,
    )
    negative_inventory_scale = max(
        abs(initial_negative_inventory),
        np.finfo(float).tiny,
    )
    positive_inventory_drifts = [0.0]
    negative_inventory_drifts = [0.0]
    vacancy_fractions = [
        _minimum_site_vacancy_fraction(model, initial_fields[2], initial_fields[3])
    ]

    for step_index, (time_start, time_end) in enumerate(
        zip(time[:-1], time[1:]),
        start=1,
    ):
        dt_s = float(time_end - time_start)
        previous = coordinates[-1]
        prediction_delta = dt_s * previous_derivative
        prediction_delta[layout.potential_slice] = 0.0
        predictor_scale = _step_scale(
            prediction_delta,
            model,
            max_log_density_update=max_log_density_update,
            max_ion_coordinate_update=max_ion_coordinate_update,
        )
        prediction = previous + predictor_scale * prediction_delta
        residual_evaluations = 0
        jacobian_evaluations = 0
        line_search_backtracks = 0
        coordinate_step_scalings = int(predictor_scale < 1.0)
        worst_condition = 0.0
        try:
            coordinate = project_dual_ion_algebraic_state(model, prediction)
        except ValueError as exc:
            raise DAEIntegrationError(
                f"dual-ion algebraic predictor projection failed: {exc}",
                step_index=step_index,
                time_s=float(time_end),
                residual_norm=float("inf"),
            ) from exc

        def evaluate(
            candidate: np.ndarray,
        ) -> tuple[np.ndarray, DualIonDAEResidualReport]:
            nonlocal residual_evaluations
            derivative = dual_ion_backward_euler_derivative(
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
                    "dual-ion Newton iterations exhausted",
                    step_index=step_index,
                    time_s=float(time_end),
                    residual_norm=residual_norm,
                )
            try:
                jacobian = finite_difference_dual_ion_backward_euler_jacobian(
                    model,
                    coordinate,
                    previous,
                    dt_s,
                    relative_step=finite_difference_relative_step,
                )
                residual_evaluations += 2 * layout.size
                exact_algebraic = model.algebraic_state_jacobian(coordinate)
                jacobian[layout.algebraic_mask] = exact_algebraic[
                    layout.algebraic_mask
                ]
                condition = float(np.linalg.cond(jacobian))
                if not np.isfinite(condition):
                    raise np.linalg.LinAlgError("non-finite Jacobian condition")
                delta = np.linalg.solve(
                    jacobian,
                    -report.normalized_residual,
                )
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                raise DAEIntegrationError(
                    f"dual-ion Jacobian solve failed: {exc}",
                    step_index=step_index,
                    time_s=float(time_end),
                    residual_norm=residual_norm,
                ) from exc
            jacobian_evaluations += 1
            nonlinear_iterations += 1
            worst_condition = max(worst_condition, condition)

            update_scale = _step_scale(
                delta,
                model,
                max_log_density_update=max_log_density_update,
                max_ion_coordinate_update=max_ion_coordinate_update,
            )
            if update_scale < 1.0:
                delta *= update_scale
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
                    "dual-ion Newton line search stalled",
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
        _n, _p, positive_ion, negative_ion, potential = model.physical_fields(
            coordinate
        )
        previous_fields = model.physical_fields(previous)
        positive_inventory = dual_cell_integral(model.grid_m, positive_ion)
        negative_inventory = dual_cell_integral(model.grid_m, negative_ion)
        previous_positive_inventory = dual_cell_integral(
            model.grid_m,
            previous_fields[2],
        )
        previous_negative_inventory = dual_cell_integral(
            model.grid_m,
            previous_fields[3],
        )
        positive_inventory_drifts.append(
            abs(positive_inventory - initial_positive_inventory)
            / positive_inventory_scale
        )
        negative_inventory_drifts.append(
            abs(negative_inventory - initial_negative_inventory)
            / negative_inventory_scale
        )
        vacancy_fraction = _minimum_site_vacancy_fraction(
            model,
            positive_ion,
            negative_ion,
        )
        vacancy_fractions.append(vacancy_fraction)
        reports.append(
            DualIonDAETimeStepReport(
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
                positive_ion_balance_defect_m2_s=abs(
                    report.positive_ion_inventory_residual_m2_s
                ),
                negative_ion_balance_defect_m2_s=abs(
                    report.negative_ion_inventory_residual_m2_s
                ),
                positive_ion_rhs_inventory_rate_m2_s=abs(
                    report.positive_ion_rhs_inventory_rate_m2_s
                ),
                negative_ion_rhs_inventory_rate_m2_s=abs(
                    report.negative_ion_rhs_inventory_rate_m2_s
                ),
                positive_ion_inventory_m2=positive_inventory,
                negative_ion_inventory_m2=negative_inventory,
                positive_ion_inventory_change_m2=(
                    positive_inventory - previous_positive_inventory
                ),
                negative_ion_inventory_change_m2=(
                    negative_inventory - previous_negative_inventory
                ),
                minimum_site_vacancy_fraction=vacancy_fraction,
            )
        )
        coordinates.append(np.array(coordinate, copy=True))
        physical_states.append(model.packed_physical_state(coordinate))
        potentials.append(np.array(potential, copy=True))
        previous_derivative = derivative

    coordinate_array = np.stack(coordinates)
    physical_array = np.stack(physical_states)
    potential_array = np.stack(potentials)
    time_ro = _readonly_f64(time, shape=(time.size,), name="dual-ion DAE time grid")
    coordinate_ro = _readonly_f64(
        coordinate_array,
        shape=(time.size, layout.size),
        name="dual-ion DAE coordinates",
    )
    physical_ro = _readonly_f64(
        physical_array,
        shape=(time.size, 4 * count),
        name="dual-ion DAE physical states",
    )
    potential_ro = _readonly_f64(
        potential_array,
        shape=(time.size, count),
        name="dual-ion DAE potentials",
    )
    report_tuple = tuple(reports)
    terminal_positive_inventory = report_tuple[-1].positive_ion_inventory_m2
    terminal_negative_inventory = report_tuple[-1].negative_ion_inventory_m2
    return DualIonDAETransientResult(
        time_s=time_ro,
        coordinates=coordinate_ro,
        physical_states=physical_ro,
        potentials_V=potential_ro,
        step_reports=report_tuple,
        success=True,
        method="physical-density-backward-euler/dual-ion-dense-central-newton-v1",
        jacobian_mode="dense_central",
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
        max_normalized_positive_ion_residual=max(
            (
                item.residual_report.max_normalized_positive_ion_residual
                for item in report_tuple
            ),
            default=0.0,
        ),
        max_normalized_negative_ion_residual=max(
            (
                item.residual_report.max_normalized_negative_ion_residual
                for item in report_tuple
            ),
            default=0.0,
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
        max_positive_ion_balance_defect_m2_s=max(
            (item.positive_ion_balance_defect_m2_s for item in report_tuple),
            default=0.0,
        ),
        max_negative_ion_balance_defect_m2_s=max(
            (item.negative_ion_balance_defect_m2_s for item in report_tuple),
            default=0.0,
        ),
        max_positive_ion_rhs_inventory_rate_m2_s=max(
            (
                item.positive_ion_rhs_inventory_rate_m2_s
                for item in report_tuple
            ),
            default=0.0,
        ),
        max_negative_ion_rhs_inventory_rate_m2_s=max(
            (
                item.negative_ion_rhs_inventory_rate_m2_s
                for item in report_tuple
            ),
            default=0.0,
        ),
        initial_positive_ion_inventory_m2=initial_positive_inventory,
        terminal_positive_ion_inventory_m2=terminal_positive_inventory,
        initial_negative_ion_inventory_m2=initial_negative_inventory,
        terminal_negative_ion_inventory_m2=terminal_negative_inventory,
        max_relative_positive_ion_inventory_drift=max(positive_inventory_drifts),
        max_relative_negative_ion_inventory_drift=max(negative_inventory_drifts),
        minimum_site_vacancy_fraction=min(vacancy_fractions),
        trajectory_sha256=_trajectory_sha256(
            model,
            time_ro,
            coordinate_ro,
            physical_ro,
            potential_ro,
        ),
    )


__all__ = [
    "DualIonDAETimeStepReport",
    "DualIonDAETransientResult",
    "dual_ion_backward_euler_derivative",
    "dual_ion_density_difference_m3",
    "finite_difference_dual_ion_backward_euler_jacobian",
    "run_dual_ion_backward_euler_reference",
]
