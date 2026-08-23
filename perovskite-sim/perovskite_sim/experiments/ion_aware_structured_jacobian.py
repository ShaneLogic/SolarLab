"""Structured Jacobian comparison for ion-aware impedance.

The full nonlinear reference lane re-solves Poisson at every central-
difference stencil.  This module independently differentiates the exact
discrete Poisson solve and the carrier/ion Scharfetter-Gummel face currents.
Bulk SRH, radiative, and Auger recombination are also analytic.  Interface
SRH is analytic for defect-free single-node sampling and for declared-defect
cross-node sampling whose no-generation clamp is proven inactive. Smooth,
unclipped Boltzmann interface-plane projection is included. Finite-rate
selective outer contacts have an analytic local rate block. Unsupported
interface closures remain central differences at a frozen potential carrying
the implicit Poisson sensitivity. It remains a comparison scaffold, not yet a
fully analytic production operator.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from numbers import Real
from typing import Any, Literal, Mapping, Self

import numpy as np

from perovskite_sim.constants import EPS_0, Q
from perovskite_sim.experiments.ion_aware_analytic_reaction import (
    IonAwareAnalyticBulkReactionLinearization,
    IonAwareAnalyticContactLinearization,
    IonAwareAnalyticInterfaceReactionLinearization,
    apply_analytic_bulk_reaction_linearization,
    apply_analytic_contact_linearization,
    apply_analytic_interface_reaction_linearization,
    build_ion_aware_analytic_bulk_reaction_linearization,
    build_ion_aware_analytic_contact_linearization,
    build_ion_aware_analytic_interface_reaction_linearization,
)
from perovskite_sim.experiments.ion_aware_analytic_transport import (
    IonAwareAnalyticTransportLinearization,
    apply_analytic_transport_linearization,
    build_ion_aware_analytic_transport_linearization,
)
from perovskite_sim.experiments.ion_aware_dc import IonAwareDCResult
from perovskite_sim.experiments.ion_aware_impedance import (
    IonAwareImpedanceProtocol,
    IonAwareImpedanceResult,
    IonAwareStateCoordinateLayout,
    ProgressCallback,
    _build_reference_evaluator,
    _physical_state,
    _state_coordinate_layout,
    _stencil_occupancy_admissible,
    run_ion_aware_impedance,
)
from perovskite_sim.experiments.jv_sweep import (
    compute_current_components,
    compute_ionic_current_components,
    extract_spatial_snapshot,
)
from perovskite_sim.models.device import DeviceStack
from perovskite_sim.physics.poisson import solve_poisson_prefactored
from perovskite_sim.solver.mol import (
    MaterialArrays,
    _harmonic_face_average,
    assemble_rhs,
    build_material_arrays,
)
from perovskite_sim.solver.small_signal import (
    FrequencyDomainResult,
    SmallSignalCurrentComponent,
    SmallSignalEvaluation,
    solve_frequency_domain,
)


ION_AWARE_STRUCTURED_JACOBIAN_PROTOCOL_SCHEMA = (
    "ion-aware-structured-jacobian-protocol-v8"
)


class IonAwareStructuredJacobianCapabilityError(ValueError):
    """The supplied reference contract cannot enter the structured lane."""


class IonAwareStructuredJacobianCertificationError(RuntimeError):
    """A finite structured operator failed one or more comparison gates."""

    def __init__(
        self,
        message: str,
        result: "IonAwareStructuredJacobianResult",
    ) -> None:
        self.result = result
        super().__init__(message)


def _finite(value: object, field: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{field} must be a real number")
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def _positive(value: object, field: str) -> float:
    number = _finite(value, field)
    if number <= 0.0:
        raise ValueError(f"{field} must be positive")
    return number


def _sha256(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True, slots=True)
class IonAwareStructuredJacobianProtocol:
    """Immutable operator-comparison method and acceptance contract."""

    impedance_protocol_sha256: str
    minimum_state_step: float
    maximum_state_step: float
    target_potential_step_V: float
    voltage_step: float
    max_nonsmooth_field_stencil_fraction: float = 0.1
    column_relevance_floor_relative: float = 1.0e-4
    max_poisson_backward_error: float = 2.0e-12
    max_group_normalized_column_error: float = 1.0e-6
    max_mass_matrix_column_relative_error: float = 2.0e-7
    max_storage_voltage_relative_error: float = 1.0e-12
    max_rate_jacobian_column_relative_error: float = 5.0e-5
    max_rate_voltage_relative_error: float = 5.0e-5
    max_conduction_jacobian_column_relative_error: float = 1.0e-4
    max_conduction_voltage_relative_error: float = 5.0e-5
    max_displacement_jacobian_column_relative_error: float = 1.0e-5
    max_displacement_voltage_relative_error: float = 1.0e-8
    max_component_jacobian_column_relative_error: float = 1.0e-4
    max_component_voltage_relative_error: float = 5.0e-5
    max_analytic_transport_jacobian_column_relative_error: float = 5.0e-6
    max_analytic_transport_voltage_relative_error: float = 5.0e-6
    max_analytic_field_mobility_derivative_relative_error: float = 5.0e-6
    max_analytic_bulk_reaction_jacobian_column_relative_error: float = 5.0e-6
    max_analytic_interface_reaction_jacobian_column_relative_error: float = (
        5.0e-6
    )
    max_analytic_contact_jacobian_column_relative_error: float = 5.0e-6
    max_impedance_magnitude_relative_error: float = 1.0e-4
    max_impedance_phase_error_deg: float = 1.0e-3
    poisson_linearization: Literal["exact_discrete_implicit"] = (
        "exact_discrete_implicit"
    )
    transport_linearization: Literal[
        "analytic_sg_field_mobility_transport"
    ] = (
        "analytic_sg_field_mobility_transport"
    )
    reaction_linearization: Literal[
        "analytic_bulk_local_cross_node_projected_interface_selective_contact"
    ] = (
        "analytic_bulk_local_cross_node_projected_interface_selective_contact"
    )
    interface_clamp_linearization: Literal[
        "positive_branch_stencil_certified"
    ] = "positive_branch_stencil_certified"
    interface_projection_linearization: Literal[
        "smooth_unclipped_boltzmann"
    ] = "smooth_unclipped_boltzmann"
    rate_row_scaling: Literal["operating_storage"] = "operating_storage"
    column_grouping: Literal["species_blocks"] = "species_blocks"
    schema_version: Literal["ion-aware-structured-jacobian-protocol-v8"] = (
        ION_AWARE_STRUCTURED_JACOBIAN_PROTOCOL_SCHEMA
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "impedance_protocol_sha256",
            _sha256(
                self.impedance_protocol_sha256,
                "impedance_protocol_sha256",
            ),
        )
        for name in (
            "minimum_state_step",
            "maximum_state_step",
            "target_potential_step_V",
            "voltage_step",
            "max_nonsmooth_field_stencil_fraction",
            "column_relevance_floor_relative",
            "max_poisson_backward_error",
            "max_group_normalized_column_error",
            "max_mass_matrix_column_relative_error",
            "max_storage_voltage_relative_error",
            "max_rate_jacobian_column_relative_error",
            "max_rate_voltage_relative_error",
            "max_conduction_jacobian_column_relative_error",
            "max_conduction_voltage_relative_error",
            "max_displacement_jacobian_column_relative_error",
            "max_displacement_voltage_relative_error",
            "max_component_jacobian_column_relative_error",
            "max_component_voltage_relative_error",
            "max_analytic_transport_jacobian_column_relative_error",
            "max_analytic_transport_voltage_relative_error",
            "max_analytic_field_mobility_derivative_relative_error",
            "max_analytic_bulk_reaction_jacobian_column_relative_error",
            "max_analytic_interface_reaction_jacobian_column_relative_error",
            "max_analytic_contact_jacobian_column_relative_error",
            "max_impedance_magnitude_relative_error",
            "max_impedance_phase_error_deg",
        ):
            object.__setattr__(self, name, _positive(getattr(self, name), name))
        if self.column_relevance_floor_relative > 1.0:
            raise ValueError("column_relevance_floor_relative cannot exceed one")
        if self.max_nonsmooth_field_stencil_fraction > 0.25:
            raise ValueError(
                "max_nonsmooth_field_stencil_fraction cannot exceed 0.25"
            )
        if self.maximum_state_step < self.minimum_state_step:
            raise ValueError("maximum_state_step must not be below minimum_state_step")
        if self.maximum_state_step > 1.0e-2:
            raise ValueError("maximum_state_step cannot exceed 1e-2")
        if self.poisson_linearization != "exact_discrete_implicit":
            raise ValueError("unsupported Poisson linearization")
        if self.transport_linearization != (
            "analytic_sg_field_mobility_transport"
        ):
            raise ValueError("unsupported transport linearization")
        if self.reaction_linearization != (
            "analytic_bulk_local_cross_node_projected_interface_selective_contact"
        ):
            raise ValueError("unsupported reaction linearization")
        if self.interface_clamp_linearization != (
            "positive_branch_stencil_certified"
        ):
            raise ValueError("unsupported interface clamp linearization")
        if self.interface_projection_linearization != (
            "smooth_unclipped_boltzmann"
        ):
            raise ValueError("unsupported interface projection linearization")
        if self.rate_row_scaling != "operating_storage":
            raise ValueError("unsupported structured rate row scaling")
        if self.column_grouping != "species_blocks":
            raise ValueError("unsupported structured column grouping")
        if self.schema_version != ION_AWARE_STRUCTURED_JACOBIAN_PROTOCOL_SCHEMA:
            raise ValueError("unsupported structured Jacobian protocol schema")

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def canonical_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )

    @property
    def protocol_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("ascii")).hexdigest()

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        if not isinstance(payload, Mapping):
            raise TypeError("structured Jacobian protocol must be a mapping")
        expected = {field.name for field in dataclasses.fields(cls)}
        actual = set(payload)
        if actual != expected:
            raise ValueError(
                "structured Jacobian protocol keys do not match schema; "
                f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
            )
        return cls(**dict(payload))

    @classmethod
    def from_json(cls, payload: str) -> Self:
        parsed = json.loads(payload)
        if not isinstance(parsed, Mapping):
            raise TypeError("structured Jacobian protocol JSON must contain an object")
        return cls.from_dict(parsed)


def build_ion_aware_structured_jacobian_protocol(
    impedance_protocol: IonAwareImpedanceProtocol,
    **overrides: float,
) -> IonAwareStructuredJacobianProtocol:
    """Bind a structured comparison to the final reference FD level."""
    if not isinstance(impedance_protocol, IonAwareImpedanceProtocol):
        raise TypeError("impedance_protocol must be an IonAwareImpedanceProtocol")
    values: dict[str, Any] = {
        "impedance_protocol_sha256": impedance_protocol.protocol_hash,
        "minimum_state_step": (
            impedance_protocol.state_step
            * impedance_protocol.refinement_factors[-1]
        ),
        "maximum_state_step": 1.0e-3,
        "target_potential_step_V": 1.0e-9,
        "voltage_step": impedance_protocol.voltage_step,
    }
    values.update(overrides)
    return IonAwareStructuredJacobianProtocol(**values)


@dataclass(frozen=True, slots=True)
class PoissonImplicitSensitivity:
    """Exact derivative of the discrete eliminated Poisson solution."""

    potential_at_operating_point_V: np.ndarray
    potential_state_jacobian_V: np.ndarray
    potential_voltage_derivative: np.ndarray
    state_steps: np.ndarray
    max_componentwise_backward_error: float
    max_nonsmooth_state_field_stencil_fraction: float
    max_nonsmooth_voltage_field_stencil_fraction: float


@dataclass(frozen=True, slots=True)
class MatrixColumnComparison:
    name: str
    relative_error_by_column: np.ndarray
    group_normalized_error_by_column: np.ndarray
    resolved_by_column: np.ndarray
    max_relative_error: float
    max_group_normalized_error: float
    worst_column: int
    bounded_weak_columns: tuple[int, ...]
    absolute_bounded_columns: tuple[int, ...]
    failed_columns: tuple[int, ...]
    limit: float
    group_normalized_limit: float
    passed: bool


@dataclass(frozen=True, slots=True)
class VectorComparison:
    name: str
    relative_error: float
    limit: float
    passed: bool


@dataclass(frozen=True, slots=True)
class CurrentComponentComparison:
    name: str
    jacobian: MatrixColumnComparison
    voltage_derivative: VectorComparison


@dataclass(frozen=True, slots=True)
class IonAwareStructuredJacobianCertificate:
    numerically_certified: bool
    thermodynamically_certified: bool
    certified: bool
    max_poisson_backward_error: float
    mass_matrix: MatrixColumnComparison
    storage_voltage_derivative: VectorComparison
    rate_jacobian: MatrixColumnComparison
    rate_voltage_derivative: VectorComparison
    conduction_jacobian: MatrixColumnComparison
    conduction_voltage_derivative: VectorComparison
    displacement_jacobian: MatrixColumnComparison
    displacement_voltage_derivative: VectorComparison
    current_components: tuple[CurrentComponentComparison, ...]
    analytic_transport_conduction_jacobian: MatrixColumnComparison
    analytic_transport_conduction_voltage_derivative: VectorComparison
    analytic_transport_components: tuple[CurrentComponentComparison, ...]
    analytic_electron_field_mobility_derivative: VectorComparison
    analytic_hole_field_mobility_derivative: VectorComparison
    analytic_bulk_reaction_rate_jacobian: MatrixColumnComparison
    analytic_bulk_reaction_rate_voltage_derivative: VectorComparison
    analytic_interface_reaction_rate_jacobian: MatrixColumnComparison
    analytic_interface_reaction_rate_voltage_derivative: VectorComparison
    analytic_contact_rate_jacobian: MatrixColumnComparison
    analytic_contact_rate_voltage_derivative: VectorComparison
    max_impedance_magnitude_relative_error: float
    max_impedance_phase_error_deg: float
    max_structured_face_spread: float
    max_structured_backward_error: float
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class IonAwareStructuredJacobianResult:
    reference: IonAwareImpedanceResult
    operator_reference: FrequencyDomainResult
    frozen_phi_finite_difference: FrequencyDomainResult
    structured: FrequencyDomainResult
    analytic_transport: IonAwareAnalyticTransportLinearization
    analytic_bulk_reaction: IonAwareAnalyticBulkReactionLinearization
    analytic_interface_reaction: IonAwareAnalyticInterfaceReactionLinearization
    analytic_contact: IonAwareAnalyticContactLinearization
    poisson_sensitivity: PoissonImplicitSensitivity
    protocol: IonAwareStructuredJacobianProtocol
    certificate: IonAwareStructuredJacobianCertificate

    @property
    def protocol_hash(self) -> str:
        return self.protocol.protocol_hash


def _matrix_column_comparison(
    name: str,
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    limit: float,
    group_normalized_limit: float,
    column_groups: tuple[slice, ...],
    column_relevance_floor_relative: float,
) -> MatrixColumnComparison:
    expected = np.asarray(reference)
    actual = np.asarray(candidate)
    if (
        expected.ndim != 2
        or expected.shape != actual.shape
        or expected.shape[1] == 0
        or not np.all(np.isfinite(expected))
        or not np.all(np.isfinite(actual))
    ):
        raise IonAwareStructuredJacobianCapabilityError(
            f"{name} matrices must be finite, non-empty, and shape matched"
        )
    column_magnitude = np.maximum(
        np.max(np.abs(expected), axis=0),
        np.max(np.abs(actual), axis=0),
    )
    resolved = np.zeros(expected.shape[1], dtype=bool)
    covered = np.zeros(expected.shape[1], dtype=bool)
    group_normalized_errors = np.zeros(expected.shape[1], dtype=float)
    absolute_difference = np.max(np.abs(actual - expected), axis=0)
    for group in column_groups:
        indices = np.arange(expected.shape[1])[group]
        if indices.size == 0:
            continue
        covered[indices] = True
        group_scale = float(np.max(column_magnitude[indices]))
        if group_scale > 0.0:
            resolved[indices] = column_magnitude[indices] >= (
                group_scale * column_relevance_floor_relative
            )
            group_normalized_errors[indices] = (
                absolute_difference[indices] / group_scale
            )
    if not np.all(covered):
        raise IonAwareStructuredJacobianCapabilityError(
            f"{name} column groups do not cover every state coordinate"
        )
    errors = np.divide(
        absolute_difference,
        column_magnitude,
        out=np.zeros(expected.shape[1], dtype=float),
        where=column_magnitude > np.finfo(float).tiny,
    )
    resolved_indices = np.flatnonzero(resolved)
    if resolved_indices.size:
        local_worst = int(np.argmax(errors[resolved_indices]))
        worst = int(resolved_indices[local_worst])
        maximum = float(errors[worst])
    else:
        worst = -1
        maximum = 0.0
    relative_passed = errors <= limit
    absolute_passed = group_normalized_errors <= group_normalized_limit
    passed_by_column = relative_passed | absolute_passed
    failed_columns = tuple(np.flatnonzero(~passed_by_column).tolist())
    return MatrixColumnComparison(
        name=name,
        relative_error_by_column=errors,
        group_normalized_error_by_column=group_normalized_errors,
        resolved_by_column=resolved,
        max_relative_error=maximum,
        max_group_normalized_error=float(np.max(group_normalized_errors)),
        worst_column=worst,
        bounded_weak_columns=tuple(np.flatnonzero(~resolved).tolist()),
        absolute_bounded_columns=tuple(
            np.flatnonzero(~relative_passed & absolute_passed).tolist()
        ),
        failed_columns=failed_columns,
        limit=limit,
        group_normalized_limit=group_normalized_limit,
        passed=not failed_columns,
    )


def _vector_comparison(
    name: str,
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    limit: float,
) -> VectorComparison:
    expected = np.asarray(reference)
    actual = np.asarray(candidate)
    if (
        expected.ndim != 1
        or expected.shape != actual.shape
        or expected.size == 0
        or not np.all(np.isfinite(expected))
        or not np.all(np.isfinite(actual))
    ):
        raise IonAwareStructuredJacobianCapabilityError(
            f"{name} vectors must be finite, non-empty, and shape matched"
        )
    scale = max(
        float(np.max(np.abs(expected))),
        float(np.max(np.abs(actual))),
        np.finfo(float).tiny,
    )
    error = float(np.max(np.abs(actual - expected)) / scale)
    return VectorComparison(
        name=name,
        relative_error=error,
        limit=limit,
        passed=error <= limit,
    )


def _poisson_componentwise_backward_error(
    material: MaterialArrays,
    potential: np.ndarray,
    rho_derivative: np.ndarray,
) -> float:
    factor = material.poisson_factor
    left = factor.C[:-1] * (potential[1:-1] - potential[:-2])
    right = factor.C[1:] * (potential[2:] - potential[1:-1])
    right_hand_side = -rho_derivative[1:-1] * factor.h_cell
    residual = right_hand_side - (right - left)
    scale = np.abs(right) + np.abs(left) + np.abs(right_hand_side)
    return float(
        np.max(
            np.abs(residual)
            / np.maximum(scale, np.finfo(float).tiny)
        )
    )


def _build_poisson_implicit_sensitivity(
    x: np.ndarray,
    stack: DeviceStack,
    dc_state: IonAwareDCResult,
    material: MaterialArrays,
    layout: IonAwareStateCoordinateLayout,
    protocol: IonAwareStructuredJacobianProtocol,
    *,
    progress: ProgressCallback | None,
) -> PoissonImplicitSensitivity:
    base_state = np.asarray(dc_state.y, dtype=float)
    base_phi = extract_spatial_snapshot(
        x,
        base_state,
        stack,
        dc_state.protocol.V_dc,
        mat=material,
    ).phi
    sensitivity = np.empty((x.size, layout.size), dtype=float)
    backward_errors = np.empty(layout.size + 1, dtype=float)
    charge_sign = (-1.0, 1.0, 1.0, -1.0)
    for column, state_index in enumerate(layout.state_indices):
        block, node = divmod(state_index, x.size)
        if block >= len(charge_sign):
            raise IonAwareStructuredJacobianCapabilityError(
                "structured Poisson sensitivity received an unknown state block"
            )
        rho_derivative = np.zeros(x.size, dtype=float)
        rho_derivative[node] = charge_sign[block] * Q * base_state[state_index]
        sensitivity[:, column] = solve_poisson_prefactored(
            material.poisson_factor,
            rho_derivative,
            phi_left=0.0,
            phi_right=0.0,
        )
        backward_errors[column] = _poisson_componentwise_backward_error(
            material,
            sensitivity[:, column],
            rho_derivative,
        )
        if progress is not None:
            progress(
                "structured_poisson_sensitivity",
                column + 1,
                layout.size,
                f"column {column + 1}/{layout.size}",
            )
    voltage_derivative = solve_poisson_prefactored(
        material.poisson_factor,
        np.zeros(x.size, dtype=float),
        phi_left=0.0,
        phi_right=-float(material.junction_polarity),
    )
    backward_errors[-1] = _poisson_componentwise_backward_error(
        material,
        voltage_derivative,
        np.zeros(x.size, dtype=float),
    )
    arrays = (base_phi, sensitivity, voltage_derivative)
    if any(not np.all(np.isfinite(array)) for array in arrays):
        raise IonAwareStructuredJacobianCapabilityError(
            "structured Poisson sensitivity produced non-finite values"
        )
    potential_sensitivity = np.max(np.abs(sensitivity), axis=0)
    requested_steps = np.divide(
        protocol.target_potential_step_V,
        potential_sensitivity,
        out=np.full(layout.size, protocol.maximum_state_step, dtype=float),
        where=potential_sensitivity > 0.0,
    )
    state_steps = np.clip(
        requested_steps,
        protocol.minimum_state_step,
        protocol.maximum_state_step,
    )
    max_state_field_fraction = 0.0
    max_voltage_field_fraction = 0.0
    if material.has_field_mobility:
        parameter_arrays = (
            material.v_sat_n_face,
            material.v_sat_p_face,
            material.ct_beta_n_face,
            material.ct_beta_p_face,
            material.pf_gamma_n_face,
            material.pf_gamma_p_face,
        )
        if any(value is None for value in parameter_arrays):
            raise IonAwareStructuredJacobianCapabilityError(
                "field-mobility material arrays are incomplete"
            )
        arrays = tuple(np.asarray(value, dtype=float) for value in parameter_arrays)
        face_shape = (x.size - 1,)
        if any(
            value.shape != face_shape or not np.all(np.isfinite(value))
            for value in arrays
        ):
            raise IonAwareStructuredJacobianCapabilityError(
                "field-mobility material arrays must be finite and face matched"
            )
        (
            v_sat_n,
            v_sat_p,
            beta_n,
            beta_p,
            gamma_n,
            gamma_p,
        ) = arrays
        nonsmooth_faces = (
            (gamma_n != 0.0)
            | (gamma_p != 0.0)
            | ((v_sat_n > 0.0) & (beta_n > 0.0) & (beta_n <= 1.0))
            | ((v_sat_p > 0.0) & (beta_p > 0.0) & (beta_p <= 1.0))
        )
        if np.any(nonsmooth_faces):
            spacing = np.diff(x)
            operating_field = -np.diff(base_phi) / spacing
            if (
                not np.all(np.isfinite(operating_field))
                or np.any(operating_field[nonsmooth_faces] == 0.0)
            ):
                raise IonAwareStructuredJacobianCapabilityError(
                    "a non-smooth field-mobility face is at zero electric field"
                )
            field_sensitivity = -np.diff(sensitivity, axis=0) / spacing[:, None]
            selected_sensitivity = np.abs(field_sensitivity[nonsmooth_faces])
            selected_field = np.abs(operating_field[nonsmooth_faces])
            allowed_field_change = (
                protocol.max_nonsmooth_field_stencil_fraction
                * selected_field[:, None]
            )
            per_face_caps = np.divide(
                allowed_field_change,
                selected_sensitivity,
                out=np.full_like(selected_sensitivity, np.inf),
                where=selected_sensitivity > 0.0,
            )
            state_caps = np.min(per_face_caps, axis=0)
            if np.any(
                state_caps
                < protocol.minimum_state_step
                * (1.0 - 16.0 * np.finfo(float).eps)
            ):
                raise IonAwareStructuredJacobianCapabilityError(
                    "the minimum state step crosses a non-smooth "
                    "field-mobility zero-field surface"
                )
            state_steps = np.minimum(state_steps, state_caps)
            state_field_fraction = np.divide(
                selected_sensitivity * state_steps[None, :],
                selected_field[:, None],
            )
            max_state_field_fraction = float(np.max(state_field_fraction))

            voltage_field_sensitivity = (
                -np.diff(voltage_derivative) / spacing
            )
            voltage_field_fraction = np.abs(
                voltage_field_sensitivity[nonsmooth_faces]
                * protocol.voltage_step
            ) / selected_field
            max_voltage_field_fraction = float(np.max(voltage_field_fraction))
            if max_voltage_field_fraction > (
                protocol.max_nonsmooth_field_stencil_fraction
                * (1.0 + 16.0 * np.finfo(float).eps)
            ):
                raise IonAwareStructuredJacobianCapabilityError(
                    "the voltage step crosses a non-smooth field-mobility "
                    "zero-field surface"
                )
    return PoissonImplicitSensitivity(
        potential_at_operating_point_V=np.asarray(base_phi, dtype=float).copy(),
        potential_state_jacobian_V=sensitivity,
        potential_voltage_derivative=voltage_derivative,
        state_steps=state_steps,
        max_componentwise_backward_error=float(np.max(backward_errors)),
        max_nonsmooth_state_field_stencil_fraction=max_state_field_fraction,
        max_nonsmooth_voltage_field_stencil_fraction=max_voltage_field_fraction,
    )


def _structured_evaluator(
    x: np.ndarray,
    stack: DeviceStack,
    dc_state: IonAwareDCResult,
    impedance_protocol: IonAwareImpedanceProtocol,
    material: MaterialArrays,
    layout: IonAwareStateCoordinateLayout,
    poisson: PoissonImplicitSensitivity,
):
    base_state = np.asarray(dc_state.y, dtype=float)
    state_indices = np.asarray(layout.state_indices, dtype=int)
    base_storage = base_state[state_indices]
    eps_face = EPS_0 * _harmonic_face_average(material.eps_r)
    polarity = float(material.junction_polarity)

    def evaluate(coordinate: np.ndarray, voltage: float) -> SmallSignalEvaluation:
        scaled_coordinate = np.asarray(coordinate, dtype=float)
        if (
            scaled_coordinate.shape != (layout.size,)
            or not np.all(np.isfinite(scaled_coordinate))
        ):
            raise IonAwareStructuredJacobianCapabilityError(
                "structured state coordinate is non-finite or shape mismatched"
            )
        increments = poisson.state_steps * scaled_coordinate
        physical = _physical_state(increments, base_state, layout)
        phi = (
            poisson.potential_at_operating_point_V
            + poisson.potential_state_jacobian_V @ increments
            + poisson.potential_voltage_derivative
            * (float(voltage) - impedance_protocol.V_dc)
        )
        rate = assemble_rhs(
            0.0,
            physical,
            x,
            stack,
            material,
            illuminated=impedance_protocol.illuminated,
            V_app=voltage,
            phi_frozen=phi,
        )
        current = compute_current_components(
            x,
            physical,
            stack,
            voltage,
            mat=material,
            phi_frozen=phi,
        )
        ionic = compute_ionic_current_components(
            x,
            physical,
            stack,
            voltage,
            mat=material,
            phi_frozen=phi,
        )
        components = [
            SmallSignalCurrentComponent(
                "electron",
                -np.asarray(current.J_n, dtype=float),
            ),
            SmallSignalCurrentComponent(
                "hole",
                -np.asarray(current.J_p, dtype=float),
            ),
            SmallSignalCurrentComponent(
                "positive_ion",
                -np.asarray(ionic.J_positive, dtype=float),
            ),
        ]
        if ionic.J_negative is not None:
            components.append(
                SmallSignalCurrentComponent(
                    "negative_ion",
                    -np.asarray(ionic.J_negative, dtype=float),
                )
            )
        conduction = sum(
            (component.current_faces for component in components),
            start=np.zeros(x.size - 1, dtype=float),
        )
        field = -np.diff(phi) / np.diff(x)
        displacement_charge = polarity * eps_face * field
        return SmallSignalEvaluation(
            # The exact tangent of y_dc*exp(u) is y_dc.  Returning its affine
            # form makes the structured mass block analytic instead of
            # reintroducing a finite-difference approximation to storage.
            storage=base_storage * (1.0 + increments),
            rate=np.asarray(rate, dtype=float)[state_indices],
            conduction_current_faces=conduction,
            displacement_charge_faces=displacement_charge,
            current_components=tuple(components),
        )

    return evaluate


def _component_comparisons(
    reference: FrequencyDomainResult,
    structured: FrequencyDomainResult,
    protocol: IonAwareStructuredJacobianProtocol,
    column_groups: tuple[slice, ...],
    *,
    name_prefix: str = "",
    jacobian_limit: float | None = None,
    voltage_limit: float | None = None,
) -> tuple[CurrentComponentComparison, ...]:
    expected = {component.name: component for component in reference.current_components}
    actual = {component.name: component for component in structured.current_components}
    if set(expected) != set(actual):
        raise IonAwareStructuredJacobianCapabilityError(
            "reference and structured current component names differ"
        )
    return tuple(
        CurrentComponentComparison(
            name=name,
            jacobian=_matrix_column_comparison(
                f"{name_prefix}{name}_current_jacobian",
                expected[name].current_jacobian,
                actual[name].current_jacobian,
                limit=(
                    protocol.max_component_jacobian_column_relative_error
                    if jacobian_limit is None
                    else jacobian_limit
                ),
                group_normalized_limit=(
                    protocol.max_group_normalized_column_error
                ),
                column_groups=column_groups,
                column_relevance_floor_relative=(
                    protocol.column_relevance_floor_relative
                ),
            ),
            voltage_derivative=_vector_comparison(
                f"{name_prefix}{name}_current_voltage_derivative",
                expected[name].voltage_derivative,
                actual[name].voltage_derivative,
                limit=(
                    protocol.max_component_voltage_relative_error
                    if voltage_limit is None
                    else voltage_limit
                ),
            ),
        )
        for name in expected
    )


def run_ion_aware_structured_jacobian_comparison(
    x: np.ndarray,
    stack: DeviceStack,
    impedance_protocol: IonAwareImpedanceProtocol,
    structured_protocol: IonAwareStructuredJacobianProtocol,
    *,
    dc_state: IonAwareDCResult,
    mat: MaterialArrays | None = None,
    require_numerical_certificate: bool = True,
    require_contact_certificate: bool = False,
    progress: ProgressCallback | None = None,
) -> IonAwareStructuredJacobianResult:
    """Regenerate the FD reference and compare a structured operator to it."""
    if not isinstance(impedance_protocol, IonAwareImpedanceProtocol):
        raise TypeError("impedance_protocol must be an IonAwareImpedanceProtocol")
    if not isinstance(structured_protocol, IonAwareStructuredJacobianProtocol):
        raise TypeError(
            "structured_protocol must be an IonAwareStructuredJacobianProtocol"
        )
    if structured_protocol.impedance_protocol_sha256 != (
        impedance_protocol.protocol_hash
    ):
        raise IonAwareStructuredJacobianCapabilityError(
            "structured protocol does not match the impedance protocol hash"
        )
    grid = np.asarray(x, dtype=float)
    material = build_material_arrays(grid, stack) if mat is None else mat
    reference = run_ion_aware_impedance(
        grid,
        stack,
        impedance_protocol,
        dc_state=dc_state,
        mat=material,
        require_numerical_certificate=True,
        require_contact_certificate=require_contact_certificate,
        progress=progress,
    )
    layout = _state_coordinate_layout(material, grid.size)
    if layout != reference.coordinate_layout:
        raise IonAwareStructuredJacobianCapabilityError(
            "structured and reference state-coordinate layouts differ"
        )
    poisson = _build_poisson_implicit_sensitivity(
        grid,
        stack,
        dc_state,
        material,
        layout,
        structured_protocol,
        progress=progress,
    )
    if not _stencil_occupancy_admissible(
        np.asarray(dc_state.y, dtype=float),
        layout,
        material,
        float(np.max(poisson.state_steps)),
    ):
        raise IonAwareStructuredJacobianCapabilityError(
            "an adaptive structured stencil crosses the ion site-occupancy limit"
        )
    full_reference_evaluate = _build_reference_evaluator(
        grid,
        stack,
        impedance_protocol,
        material,
        layout,
        np.asarray(dc_state.y, dtype=float),
    )

    def adaptive_reference_evaluate(
        coordinate: np.ndarray,
        voltage: float,
    ) -> SmallSignalEvaluation:
        return full_reference_evaluate(poisson.state_steps * coordinate, voltage)

    coordinate = np.zeros(layout.size, dtype=float)
    frequencies = np.asarray(impedance_protocol.frequencies_Hz, dtype=float)
    face_weights = np.diff(grid) / float(grid[-1] - grid[0])
    operator_reference = solve_frequency_domain(
        adaptive_reference_evaluate,
        coordinate,
        impedance_protocol.V_dc,
        frequencies,
        state_step=1.0,
        voltage_step=structured_protocol.voltage_step,
        face_weights=face_weights,
        progress=progress,
    )
    evaluate = _structured_evaluator(
        grid,
        stack,
        dc_state,
        impedance_protocol,
        material,
        layout,
        poisson,
    )
    frozen_phi_finite_difference = solve_frequency_domain(
        evaluate,
        coordinate,
        impedance_protocol.V_dc,
        frequencies,
        state_step=1.0,
        voltage_step=structured_protocol.voltage_step,
        face_weights=face_weights,
        progress=progress,
    )
    try:
        analytic_transport = build_ion_aware_analytic_transport_linearization(
            grid,
            stack,
            np.asarray(dc_state.y, dtype=float),
            impedance_protocol.V_dc,
            material,
            layout,
            potential_at_operating_point_V=(
                poisson.potential_at_operating_point_V
            ),
            potential_state_jacobian_V=(
                poisson.potential_state_jacobian_V
            ),
            potential_voltage_derivative=(
                poisson.potential_voltage_derivative
            ),
            state_steps=poisson.state_steps,
        )
        analytic_transport_result = apply_analytic_transport_linearization(
            frozen_phi_finite_difference,
            analytic_transport,
            material,
            layout,
            V_dc=impedance_protocol.V_dc,
            face_weights=face_weights,
            progress=progress,
        )
        analytic_bulk_reaction = (
            build_ion_aware_analytic_bulk_reaction_linearization(
                grid,
                stack,
                np.asarray(dc_state.y, dtype=float),
                impedance_protocol.V_dc,
                material,
                layout,
                potential_at_operating_point_V=(
                    poisson.potential_at_operating_point_V
                ),
                state_steps=poisson.state_steps,
            )
        )
        analytic_bulk_reaction_result = (
            apply_analytic_bulk_reaction_linearization(
                analytic_transport_result,
                analytic_bulk_reaction,
                layout,
                V_dc=impedance_protocol.V_dc,
                face_weights=face_weights,
                progress=progress,
            )
        )
        analytic_interface_reaction = (
            build_ion_aware_analytic_interface_reaction_linearization(
                grid,
                stack,
                np.asarray(dc_state.y, dtype=float),
                impedance_protocol.V_dc,
                material,
                layout,
                potential_at_operating_point_V=(
                    poisson.potential_at_operating_point_V
                ),
                potential_state_jacobian_V=(
                    poisson.potential_state_jacobian_V
                ),
                potential_voltage_derivative=(
                    poisson.potential_voltage_derivative
                ),
                state_steps=poisson.state_steps,
                voltage_step=structured_protocol.voltage_step,
            )
        )
        analytic_interface_reaction_result = (
            apply_analytic_interface_reaction_linearization(
                analytic_bulk_reaction_result,
                analytic_interface_reaction,
                layout,
                V_dc=impedance_protocol.V_dc,
                face_weights=face_weights,
                progress=progress,
            )
        )
        analytic_contact = build_ion_aware_analytic_contact_linearization(
            grid,
            stack,
            np.asarray(dc_state.y, dtype=float),
            impedance_protocol.V_dc,
            material,
            layout,
            potential_at_operating_point_V=(
                poisson.potential_at_operating_point_V
            ),
            state_steps=poisson.state_steps,
        )
        structured = apply_analytic_contact_linearization(
            analytic_interface_reaction_result,
            analytic_contact,
            layout,
            V_dc=impedance_protocol.V_dc,
            face_weights=face_weights,
            progress=progress,
        )
    except ValueError as exc:
        raise IonAwareStructuredJacobianCapabilityError(
            f"analytic structured operator assembly failed: {exc}"
        ) from exc
    expected = operator_reference
    column_groups = tuple(
        layout.coordinate_slice(species)
        for species in (
            "electron",
            "hole",
            "positive_ion",
            "negative_ion",
        )
        if layout.coordinate_slice(species).stop
        > layout.coordinate_slice(species).start
    )
    row_scale = np.maximum(
        np.abs(expected.storage_at_operating_point),
        1.0,
    )
    mass = _matrix_column_comparison(
        "mass_matrix",
        expected.mass_matrix / row_scale[:, None],
        structured.mass_matrix / row_scale[:, None],
        limit=structured_protocol.max_mass_matrix_column_relative_error,
        group_normalized_limit=(
            structured_protocol.max_group_normalized_column_error
        ),
        column_groups=column_groups,
        column_relevance_floor_relative=(
            structured_protocol.column_relevance_floor_relative
        ),
    )
    storage_voltage = _vector_comparison(
        "storage_voltage_derivative",
        expected.storage_voltage_derivative / row_scale,
        structured.storage_voltage_derivative / row_scale,
        limit=structured_protocol.max_storage_voltage_relative_error,
    )
    rate = _matrix_column_comparison(
        "rate_jacobian",
        expected.rate_jacobian / row_scale[:, None],
        structured.rate_jacobian / row_scale[:, None],
        limit=structured_protocol.max_rate_jacobian_column_relative_error,
        group_normalized_limit=(
            structured_protocol.max_group_normalized_column_error
        ),
        column_groups=column_groups,
        column_relevance_floor_relative=(
            structured_protocol.column_relevance_floor_relative
        ),
    )
    rate_voltage = _vector_comparison(
        "rate_voltage_derivative",
        expected.rate_voltage_derivative / row_scale,
        structured.rate_voltage_derivative / row_scale,
        limit=structured_protocol.max_rate_voltage_relative_error,
    )
    conduction = _matrix_column_comparison(
        "conduction_current_jacobian",
        expected.conduction_current_jacobian,
        structured.conduction_current_jacobian,
        limit=structured_protocol.max_conduction_jacobian_column_relative_error,
        group_normalized_limit=(
            structured_protocol.max_group_normalized_column_error
        ),
        column_groups=column_groups,
        column_relevance_floor_relative=(
            structured_protocol.column_relevance_floor_relative
        ),
    )
    conduction_voltage = _vector_comparison(
        "conduction_current_voltage_derivative",
        expected.conduction_current_voltage_derivative,
        structured.conduction_current_voltage_derivative,
        limit=structured_protocol.max_conduction_voltage_relative_error,
    )
    displacement = _matrix_column_comparison(
        "displacement_charge_jacobian",
        expected.displacement_charge_jacobian,
        structured.displacement_charge_jacobian,
        limit=structured_protocol.max_displacement_jacobian_column_relative_error,
        group_normalized_limit=(
            structured_protocol.max_group_normalized_column_error
        ),
        column_groups=column_groups,
        column_relevance_floor_relative=(
            structured_protocol.column_relevance_floor_relative
        ),
    )
    displacement_voltage = _vector_comparison(
        "displacement_charge_voltage_derivative",
        expected.displacement_charge_voltage_derivative,
        structured.displacement_charge_voltage_derivative,
        limit=structured_protocol.max_displacement_voltage_relative_error,
    )
    components = _component_comparisons(
        expected,
        structured,
        structured_protocol,
        column_groups,
    )
    analytic_transport_conduction = _matrix_column_comparison(
        "analytic_transport_conduction_current_jacobian",
        frozen_phi_finite_difference.conduction_current_jacobian,
        structured.conduction_current_jacobian,
        limit=(
            structured_protocol
            .max_analytic_transport_jacobian_column_relative_error
        ),
        group_normalized_limit=(
            structured_protocol.max_group_normalized_column_error
        ),
        column_groups=column_groups,
        column_relevance_floor_relative=(
            structured_protocol.column_relevance_floor_relative
        ),
    )
    analytic_transport_voltage = _vector_comparison(
        "analytic_transport_conduction_current_voltage_derivative",
        frozen_phi_finite_difference.conduction_current_voltage_derivative,
        structured.conduction_current_voltage_derivative,
        limit=(
            structured_protocol.max_analytic_transport_voltage_relative_error
        ),
    )
    analytic_transport_components = _component_comparisons(
        frozen_phi_finite_difference,
        structured,
        structured_protocol,
        column_groups,
        name_prefix="analytic_transport_",
        jacobian_limit=(
            structured_protocol
            .max_analytic_transport_jacobian_column_relative_error
        ),
        voltage_limit=(
            structured_protocol.max_analytic_transport_voltage_relative_error
        ),
    )
    field_mobility = analytic_transport.field_mobility
    analytic_electron_field_mobility = _vector_comparison(
        "analytic_electron_field_mobility_derivative",
        field_mobility.electron_finite_difference_derivative_m3_V2_s,
        field_mobility.electron_field_derivative_m3_V2_s,
        limit=(
            structured_protocol
            .max_analytic_field_mobility_derivative_relative_error
        ),
    )
    analytic_hole_field_mobility = _vector_comparison(
        "analytic_hole_field_mobility_derivative",
        field_mobility.hole_finite_difference_derivative_m3_V2_s,
        field_mobility.hole_field_derivative_m3_V2_s,
        limit=(
            structured_protocol
            .max_analytic_field_mobility_derivative_relative_error
        ),
    )
    analytic_bulk_reaction_rate = _matrix_column_comparison(
        "analytic_bulk_reaction_rate_jacobian",
        analytic_bulk_reaction.finite_difference_rate_jacobian
        / row_scale[:, None],
        analytic_bulk_reaction.rate_jacobian / row_scale[:, None],
        limit=(
            structured_protocol
            .max_analytic_bulk_reaction_jacobian_column_relative_error
        ),
        group_normalized_limit=(
            structured_protocol.max_group_normalized_column_error
        ),
        column_groups=column_groups,
        column_relevance_floor_relative=(
            structured_protocol.column_relevance_floor_relative
        ),
    )
    analytic_bulk_reaction_voltage = _vector_comparison(
        "analytic_bulk_reaction_rate_voltage_derivative",
        analytic_bulk_reaction.finite_difference_rate_voltage_derivative,
        analytic_bulk_reaction.rate_voltage_derivative,
        limit=structured_protocol.max_rate_voltage_relative_error,
    )
    analytic_interface_reaction_rate = _matrix_column_comparison(
        "analytic_interface_reaction_rate_jacobian",
        analytic_interface_reaction.complex_step_rate_jacobian
        / row_scale[:, None],
        analytic_interface_reaction.rate_jacobian / row_scale[:, None],
        limit=(
            structured_protocol
            .max_analytic_interface_reaction_jacobian_column_relative_error
        ),
        group_normalized_limit=(
            structured_protocol.max_group_normalized_column_error
        ),
        column_groups=column_groups,
        column_relevance_floor_relative=(
            structured_protocol.column_relevance_floor_relative
        ),
    )
    analytic_interface_reaction_voltage = _vector_comparison(
        "analytic_interface_reaction_rate_voltage_derivative",
        analytic_interface_reaction.complex_step_rate_voltage_derivative,
        analytic_interface_reaction.rate_voltage_derivative,
        limit=structured_protocol.max_rate_voltage_relative_error,
    )
    analytic_contact_rate = _matrix_column_comparison(
        "analytic_contact_rate_jacobian",
        analytic_contact.finite_difference_rate_jacobian
        / row_scale[:, None],
        analytic_contact.rate_jacobian / row_scale[:, None],
        limit=(
            structured_protocol
            .max_analytic_contact_jacobian_column_relative_error
        ),
        group_normalized_limit=(
            structured_protocol.max_group_normalized_column_error
        ),
        column_groups=column_groups,
        column_relevance_floor_relative=(
            structured_protocol.column_relevance_floor_relative
        ),
    )
    analytic_contact_voltage = _vector_comparison(
        "analytic_contact_rate_voltage_derivative",
        analytic_contact.finite_difference_rate_voltage_derivative,
        analytic_contact.rate_voltage_derivative,
        limit=structured_protocol.max_rate_voltage_relative_error,
    )
    response_reference = reference.reference_linearization
    impedance_scale = np.maximum(
        np.maximum(
            np.abs(response_reference.impedance),
            np.abs(structured.impedance),
        ),
        np.finfo(float).tiny,
    )
    impedance_magnitude_error = float(
        np.max(
            np.abs(
                np.abs(structured.impedance)
                - np.abs(response_reference.impedance)
            )
            / impedance_scale
        )
    )
    impedance_phase_error = float(
        np.max(
            np.abs(
                np.angle(
                    structured.impedance / response_reference.impedance,
                    deg=True,
                )
            )
        )
    )
    structured_spread = float(np.max(structured.max_relative_face_spread))
    structured_backward = float(np.max(structured.backward_error))
    reasons: list[str] = []
    comparisons = (
        mass,
        storage_voltage,
        rate,
        rate_voltage,
        conduction,
        conduction_voltage,
        displacement,
        displacement_voltage,
        analytic_transport_conduction,
        analytic_transport_voltage,
        analytic_electron_field_mobility,
        analytic_hole_field_mobility,
        analytic_bulk_reaction_rate,
        analytic_bulk_reaction_voltage,
        analytic_interface_reaction_rate,
        analytic_interface_reaction_voltage,
        analytic_contact_rate,
        analytic_contact_voltage,
    )
    reasons.extend(
        f"{comparison.name}_exceeds_limit"
        for comparison in comparisons
        if not comparison.passed
    )
    if poisson.max_componentwise_backward_error > (
        structured_protocol.max_poisson_backward_error
    ):
        reasons.append("poisson_implicit_backward_error_exceeds_limit")
    for component in components:
        if not component.jacobian.passed:
            reasons.append(f"{component.jacobian.name}_exceeds_limit")
        if not component.voltage_derivative.passed:
            reasons.append(
                f"{component.voltage_derivative.name}_exceeds_limit"
            )
    for component in analytic_transport_components:
        if not component.jacobian.passed:
            reasons.append(f"{component.jacobian.name}_exceeds_limit")
        if not component.voltage_derivative.passed:
            reasons.append(
                f"{component.voltage_derivative.name}_exceeds_limit"
            )
    if impedance_magnitude_error > (
        structured_protocol.max_impedance_magnitude_relative_error
    ):
        reasons.append("impedance_magnitude_comparison_exceeds_limit")
    if impedance_phase_error > structured_protocol.max_impedance_phase_error_deg:
        reasons.append("impedance_phase_comparison_exceeds_limit")
    if structured_spread > impedance_protocol.max_relative_face_spread:
        reasons.append("structured_all_face_admittance_spread_exceeds_limit")
    if structured_backward > impedance_protocol.max_backward_error:
        reasons.append("structured_backward_error_exceeds_limit")
    numerical = not reasons
    thermodynamic = reference.certificate.thermodynamically_certified
    certificate = IonAwareStructuredJacobianCertificate(
        numerically_certified=numerical,
        thermodynamically_certified=thermodynamic,
        certified=numerical and thermodynamic,
        max_poisson_backward_error=poisson.max_componentwise_backward_error,
        mass_matrix=mass,
        storage_voltage_derivative=storage_voltage,
        rate_jacobian=rate,
        rate_voltage_derivative=rate_voltage,
        conduction_jacobian=conduction,
        conduction_voltage_derivative=conduction_voltage,
        displacement_jacobian=displacement,
        displacement_voltage_derivative=displacement_voltage,
        current_components=components,
        analytic_transport_conduction_jacobian=(
            analytic_transport_conduction
        ),
        analytic_transport_conduction_voltage_derivative=(
            analytic_transport_voltage
        ),
        analytic_transport_components=analytic_transport_components,
        analytic_electron_field_mobility_derivative=(
            analytic_electron_field_mobility
        ),
        analytic_hole_field_mobility_derivative=(
            analytic_hole_field_mobility
        ),
        analytic_bulk_reaction_rate_jacobian=analytic_bulk_reaction_rate,
        analytic_bulk_reaction_rate_voltage_derivative=(
            analytic_bulk_reaction_voltage
        ),
        analytic_interface_reaction_rate_jacobian=(
            analytic_interface_reaction_rate
        ),
        analytic_interface_reaction_rate_voltage_derivative=(
            analytic_interface_reaction_voltage
        ),
        analytic_contact_rate_jacobian=analytic_contact_rate,
        analytic_contact_rate_voltage_derivative=analytic_contact_voltage,
        max_impedance_magnitude_relative_error=impedance_magnitude_error,
        max_impedance_phase_error_deg=impedance_phase_error,
        max_structured_face_spread=structured_spread,
        max_structured_backward_error=structured_backward,
        reasons=tuple(reasons),
    )
    result = IonAwareStructuredJacobianResult(
        reference=reference,
        operator_reference=operator_reference,
        frozen_phi_finite_difference=frozen_phi_finite_difference,
        structured=structured,
        analytic_transport=analytic_transport,
        analytic_bulk_reaction=analytic_bulk_reaction,
        analytic_interface_reaction=analytic_interface_reaction,
        analytic_contact=analytic_contact,
        poisson_sensitivity=poisson,
        protocol=structured_protocol,
        certificate=certificate,
    )
    if require_numerical_certificate and not numerical:
        raise IonAwareStructuredJacobianCertificationError(
            "ion-aware structured Jacobian comparison failed: "
            + ", ".join(reasons),
            result,
        )
    return result


__all__ = [
    "ION_AWARE_STRUCTURED_JACOBIAN_PROTOCOL_SCHEMA",
    "CurrentComponentComparison",
    "IonAwareStructuredJacobianCapabilityError",
    "IonAwareStructuredJacobianCertificate",
    "IonAwareStructuredJacobianCertificationError",
    "IonAwareStructuredJacobianProtocol",
    "IonAwareStructuredJacobianResult",
    "MatrixColumnComparison",
    "PoissonImplicitSensitivity",
    "VectorComparison",
    "build_ion_aware_structured_jacobian_protocol",
    "run_ion_aware_structured_jacobian_comparison",
]
