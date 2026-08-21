"""Frequency-domain linearization for residual-certified dynamic models.

The model callback exposes four quantities at one operating point:

``storage``
    Differential state whose time derivative appears in the conservation law.
``rate``
    Right-hand side for that stored state.
``conduction_current_faces``
    Terminal-current contribution in the caller's measurement convention.
``displacement_charge_faces``
    Electric-displacement contribution whose time derivative is current.

For a perturbation ``u_hat * exp(i omega t)`` and voltage amplitude ``V_hat``,
central finite differences form

``(i omega M - A) u_hat = (b - i omega m_V) V_hat``.

The implementation is intentionally independent of density, quasi-Fermi, or
ionic coordinates. Device-specific adapters own those choices and their
capability gates.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.linalg.lapack import zgesvx


class SmallSignalLinearizationError(RuntimeError):
    """The operating point could not produce a finite small-signal result."""


@dataclass(frozen=True)
class SmallSignalCurrentComponent:
    """Named all-face conduction-current contribution at one evaluation."""

    name: str
    current_faces: np.ndarray


@dataclass(frozen=True)
class SmallSignalEvaluation:
    """Model quantities evaluated at one state coordinate and bias."""

    storage: np.ndarray
    rate: np.ndarray
    conduction_current_faces: np.ndarray
    displacement_charge_faces: np.ndarray
    current_components: tuple[SmallSignalCurrentComponent, ...] = ()


@dataclass(frozen=True)
class FrequencyDomainCurrentComponent:
    """Named complex all-face current response per applied volt."""

    name: str
    admittance_faces: np.ndarray
    current_jacobian: np.ndarray
    voltage_derivative: np.ndarray


@dataclass(frozen=True)
class FrequencyDomainResult:
    """Complex admittance/impedance and all-face conservation diagnostics."""

    frequencies: np.ndarray
    impedance: np.ndarray
    admittance: np.ndarray
    admittance_faces: np.ndarray
    state_response: np.ndarray
    storage_response: np.ndarray
    max_relative_face_spread: np.ndarray
    reciprocal_condition: np.ndarray
    backward_error: np.ndarray
    conduction_admittance_faces: np.ndarray
    displacement_admittance_faces: np.ndarray
    current_components: tuple[FrequencyDomainCurrentComponent, ...]
    storage_at_operating_point: np.ndarray
    mass_matrix: np.ndarray
    rate_jacobian: np.ndarray
    conduction_current_jacobian: np.ndarray
    displacement_charge_jacobian: np.ndarray
    storage_voltage_derivative: np.ndarray
    rate_voltage_derivative: np.ndarray
    conduction_current_voltage_derivative: np.ndarray
    displacement_charge_voltage_derivative: np.ndarray
    state_step: float
    voltage_step: float


EvaluationCallback = Callable[[np.ndarray, float], SmallSignalEvaluation]
ProgressCallback = Callable[[str, int, int, str], None]


def _as_finite_vector(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise SmallSignalLinearizationError(
            f"{name} must be a non-empty finite one-dimensional array"
        )
    return array


def _validate_evaluation(
    value: SmallSignalEvaluation,
    *,
    state_size: int,
    face_count: int | None = None,
    component_names: tuple[str, ...] | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[str, ...],
    np.ndarray,
]:
    storage = _as_finite_vector(value.storage, "storage")
    rate = _as_finite_vector(value.rate, "rate")
    conduction = _as_finite_vector(
        value.conduction_current_faces,
        "conduction_current_faces",
    )
    displacement = _as_finite_vector(
        value.displacement_charge_faces,
        "displacement_charge_faces",
    )
    if storage.size != state_size or rate.shape != storage.shape:
        raise SmallSignalLinearizationError(
            "storage and rate must match the linearized state-coordinate size"
        )
    if displacement.shape != conduction.shape:
        raise SmallSignalLinearizationError(
            "conduction current and displacement charge must share face shape"
        )
    if face_count is not None and conduction.size != face_count:
        raise SmallSignalLinearizationError(
            "model callback changed its number of current faces"
        )
    names: list[str] = []
    components: list[np.ndarray] = []
    for component in value.current_components:
        if not isinstance(component, SmallSignalCurrentComponent):
            raise SmallSignalLinearizationError(
                "current_components must contain SmallSignalCurrentComponent values"
            )
        if not isinstance(component.name, str) or not component.name.strip():
            raise SmallSignalLinearizationError(
                "current component names must be non-empty strings"
            )
        names.append(component.name)
        component_faces = _as_finite_vector(
            component.current_faces,
            f"current_components[{component.name!r}]",
        )
        if component_faces.shape != conduction.shape:
            raise SmallSignalLinearizationError(
                "current component faces must match conduction current faces"
            )
        components.append(component_faces)
    normalized_names = tuple(names)
    if len(set(normalized_names)) != len(normalized_names):
        raise SmallSignalLinearizationError(
            "current component names must be unique"
        )
    if component_names is not None and normalized_names != component_names:
        raise SmallSignalLinearizationError(
            "model callback changed its current component names"
        )
    component_array = (
        np.stack(components, axis=0)
        if components
        else np.empty((0, conduction.size), dtype=float)
    )
    if components:
        component_sum = np.sum(component_array, axis=0)
        scale = np.maximum(
            np.maximum(np.abs(conduction), np.sum(np.abs(component_array), axis=0)),
            np.finfo(float).tiny,
        )
        if np.any(np.abs(component_sum - conduction) > 1.0e-12 * scale):
            raise SmallSignalLinearizationError(
                "named current components must sum to conduction_current_faces"
            )
    return (
        storage,
        rate,
        conduction,
        displacement,
        normalized_names,
        component_array,
    )


def solve_frequency_domain(
    evaluate: EvaluationCallback,
    state_coordinate: np.ndarray,
    V_dc: float,
    frequencies: np.ndarray,
    *,
    state_step: float = 1.0e-5,
    voltage_step: float = 1.0e-5,
    face_weights: np.ndarray | None = None,
    progress: ProgressCallback | None = None,
) -> FrequencyDomainResult:
    """Linearize a dynamic model and solve its complex frequency response.

    ``state_coordinate`` may use any dimensionless or dimensional variables,
    provided ``storage`` and ``rate`` describe the same physical conserved
    quantities. Central differences are used for both state and voltage
    derivatives. Rows are scaled by the operating-point storage magnitude
    before the complex solve; this changes conditioning, not the equations.
    """
    coordinate = _as_finite_vector(state_coordinate, "state_coordinate")
    omega_hz = _as_finite_vector(frequencies, "frequencies")
    if np.any(omega_hz <= 0.0):
        raise ValueError("frequencies must be positive")
    if not np.isfinite(V_dc):
        raise ValueError("V_dc must be finite")
    for name, value in (
        ("state_step", state_step),
        ("voltage_step", voltage_step),
    ):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")

    base = evaluate(coordinate, float(V_dc))
    (
        storage_0,
        rate_0,
        current_0,
        displacement_0,
        component_names,
        component_current_0,
    ) = _validate_evaluation(
        base,
        state_size=coordinate.size,
    )
    del rate_0, current_0, displacement_0, component_current_0
    state_size = coordinate.size
    face_count = base.conduction_current_faces.size

    mass = np.empty((state_size, state_size), dtype=float)
    rate_jacobian = np.empty_like(mass)
    current_jacobian = np.empty((face_count, state_size), dtype=float)
    displacement_jacobian = np.empty_like(current_jacobian)
    component_current_jacobian = np.empty(
        (len(component_names), face_count, state_size), dtype=float
    )
    denominator = 2.0 * float(state_step)
    for column in range(state_size):
        plus_coordinate = coordinate.copy()
        minus_coordinate = coordinate.copy()
        plus_coordinate[column] += state_step
        minus_coordinate[column] -= state_step
        plus = _validate_evaluation(
            evaluate(plus_coordinate, float(V_dc)),
            state_size=state_size,
            face_count=face_count,
            component_names=component_names,
        )
        minus = _validate_evaluation(
            evaluate(minus_coordinate, float(V_dc)),
            state_size=state_size,
            face_count=face_count,
            component_names=component_names,
        )
        mass[:, column] = (plus[0] - minus[0]) / denominator
        rate_jacobian[:, column] = (plus[1] - minus[1]) / denominator
        current_jacobian[:, column] = (plus[2] - minus[2]) / denominator
        displacement_jacobian[:, column] = (
            plus[3] - minus[3]
        ) / denominator
        component_current_jacobian[:, :, column] = (
            plus[5] - minus[5]
        ) / denominator
        if progress is not None:
            progress(
                "small_signal_linearization",
                column + 1,
                state_size,
                f"column {column + 1}/{state_size}",
            )

    plus_voltage = _validate_evaluation(
        evaluate(coordinate, float(V_dc + voltage_step)),
        state_size=state_size,
        face_count=face_count,
        component_names=component_names,
    )
    minus_voltage = _validate_evaluation(
        evaluate(coordinate, float(V_dc - voltage_step)),
        state_size=state_size,
        face_count=face_count,
        component_names=component_names,
    )
    voltage_denominator = 2.0 * float(voltage_step)
    storage_voltage = (plus_voltage[0] - minus_voltage[0]) / voltage_denominator
    rate_voltage = (plus_voltage[1] - minus_voltage[1]) / voltage_denominator
    current_voltage = (plus_voltage[2] - minus_voltage[2]) / voltage_denominator
    displacement_voltage = (
        plus_voltage[3] - minus_voltage[3]
    ) / voltage_denominator
    component_current_voltage = (
        plus_voltage[5] - minus_voltage[5]
    ) / voltage_denominator

    # A supplied decomposition is the authoritative discrete definition of
    # conduction current. Summing its differentiated channels avoids a
    # second, differently ordered subtraction of the same large currents,
    # which otherwise leaves avoidable cancellation noise in the total.
    if component_names:
        current_jacobian = np.sum(component_current_jacobian, axis=0)
        current_voltage = np.sum(component_current_voltage, axis=0)

    arrays = (
        mass,
        rate_jacobian,
        current_jacobian,
        displacement_jacobian,
        storage_voltage,
        rate_voltage,
        current_voltage,
        displacement_voltage,
        component_current_jacobian,
        component_current_voltage,
    )
    if any(not np.all(np.isfinite(array)) for array in arrays):
        raise SmallSignalLinearizationError(
            "finite differences produced a non-finite linearized operator"
        )

    if face_weights is None:
        weights = np.full(face_count, 1.0 / face_count, dtype=float)
    else:
        weights = np.asarray(face_weights, dtype=float)
        if (
            weights.shape != (face_count,)
            or not np.all(np.isfinite(weights))
            or np.any(weights < 0.0)
            or float(np.sum(weights)) <= 0.0
        ):
            raise ValueError(
                "face_weights must be finite, non-negative, and match current faces"
            )
        weights = weights / float(np.sum(weights))

    row_scale = np.maximum(np.abs(storage_0), 1.0)
    mass_scaled = mass / row_scale[:, None]
    rate_scaled = rate_jacobian / row_scale[:, None]
    storage_voltage_scaled = storage_voltage / row_scale
    rate_voltage_scaled = rate_voltage / row_scale

    admittance_faces = np.empty((omega_hz.size, face_count), dtype=complex)
    conduction_admittance_faces = np.empty_like(admittance_faces)
    displacement_admittance_faces = np.empty_like(admittance_faces)
    component_admittance_faces = np.empty(
        (len(component_names), omega_hz.size, face_count), dtype=complex
    )
    state_response = np.empty((omega_hz.size, state_size), dtype=complex)
    storage_response = np.empty_like(state_response)
    reciprocal_condition = np.empty(omega_hz.size, dtype=float)
    backward_error = np.empty(omega_hz.size, dtype=float)
    for index, frequency in enumerate(omega_hz):
        angular_frequency = 2.0 * np.pi * float(frequency)
        operator = 1j * angular_frequency * mass_scaled - rate_scaled
        forcing = rate_voltage_scaled - (
            1j * angular_frequency * storage_voltage_scaled
        )
        lapack = zgesvx(
            np.asarray(operator, dtype=complex, order="F"),
            np.asarray(forcing[:, None], dtype=complex, order="F"),
        )
        solution = lapack[7]
        rcond = float(lapack[8])
        info = int(lapack[11])
        if info < 0:
            raise SmallSignalLinearizationError(
                "LAPACK rejected the frequency-domain operator at "
                f"f={frequency:.6g} Hz"
            )
        if 0 < info <= state_size:
            raise SmallSignalLinearizationError(
                "singular frequency-domain operator at "
                f"f={frequency:.6g} Hz"
            )
        # zgesvx uses info=n+1 when the reciprocal condition estimate falls
        # below machine precision. A small residual cannot certify forward
        # accuracy in that regime, so fail closed instead of returning it.
        if info == state_size + 1 or rcond <= np.finfo(float).eps:
            raise SmallSignalLinearizationError(
                "numerically singular frequency-domain operator at "
                f"f={frequency:.6g} Hz"
            )
        response = solution[:, 0]
        if not np.all(np.isfinite(response)):
            raise SmallSignalLinearizationError(
                f"non-finite state response at f={frequency:.6g} Hz"
            )
        residual = forcing - operator @ response
        residual_scale = np.abs(operator) @ np.abs(response) + np.abs(forcing)
        berr = float(
            np.max(
                np.abs(residual)
                / np.maximum(residual_scale, np.finfo(float).tiny)
            )
        )
        if not np.isfinite(rcond) or not np.isfinite(berr):
            raise SmallSignalLinearizationError(
                f"invalid linear-solve diagnostics at f={frequency:.6g} Hz"
            )
        reciprocal_condition[index] = rcond
        backward_error[index] = berr
        state_response[index] = response
        storage_response[index] = mass @ response + storage_voltage
        if component_names:
            component_admittance_faces[:, index, :] = (
                np.einsum("cfs,s->cf", component_current_jacobian, response)
                + component_current_voltage
            )
            conduction = np.sum(component_admittance_faces[:, index, :], axis=0)
        else:
            conduction = current_jacobian @ response + current_voltage
        displacement = (
            displacement_jacobian @ response + displacement_voltage
        )
        conduction_admittance_faces[index] = conduction
        displacement_admittance_faces[index] = (
            1j * angular_frequency * displacement
        )
        admittance_faces[index] = (
            conduction_admittance_faces[index]
            + displacement_admittance_faces[index]
        )

    admittance = admittance_faces @ weights
    if np.any(~np.isfinite(admittance)) or np.any(np.abs(admittance) == 0.0):
        raise SmallSignalLinearizationError(
            "frequency-domain solve produced zero or non-finite admittance"
        )
    impedance = 1.0 / admittance
    spread = np.max(
        np.abs(admittance_faces - admittance[:, None]),
        axis=1,
    ) / np.maximum(np.abs(admittance), np.finfo(float).tiny)
    return FrequencyDomainResult(
        frequencies=omega_hz.copy(),
        impedance=impedance,
        admittance=admittance,
        admittance_faces=admittance_faces,
        state_response=state_response,
        storage_response=storage_response,
        max_relative_face_spread=spread,
        reciprocal_condition=reciprocal_condition,
        backward_error=backward_error,
        conduction_admittance_faces=conduction_admittance_faces,
        displacement_admittance_faces=displacement_admittance_faces,
        current_components=tuple(
            FrequencyDomainCurrentComponent(
                name=name,
                admittance_faces=component_admittance_faces[index].copy(),
                current_jacobian=component_current_jacobian[index].copy(),
                voltage_derivative=component_current_voltage[index].copy(),
            )
            for index, name in enumerate(component_names)
        ),
        storage_at_operating_point=storage_0.copy(),
        mass_matrix=mass,
        rate_jacobian=rate_jacobian,
        conduction_current_jacobian=current_jacobian,
        displacement_charge_jacobian=displacement_jacobian,
        storage_voltage_derivative=storage_voltage,
        rate_voltage_derivative=rate_voltage,
        conduction_current_voltage_derivative=current_voltage,
        displacement_charge_voltage_derivative=displacement_voltage,
        state_step=float(state_step),
        voltage_step=float(voltage_step),
    )


__all__ = [
    "FrequencyDomainResult",
    "FrequencyDomainCurrentComponent",
    "SmallSignalEvaluation",
    "SmallSignalCurrentComponent",
    "SmallSignalLinearizationError",
    "solve_frequency_domain",
]
