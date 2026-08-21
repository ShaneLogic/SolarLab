"""Analytic bulk-recombination block for ion-aware impedance."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from perovskite_sim.experiments.ion_aware_impedance import (
    IonAwareStateCoordinateLayout,
)
from perovskite_sim.experiments.jv_sweep import _state_fields
from perovskite_sim.models.device import DeviceStack
from perovskite_sim.physics.recombination import (
    bulk_srh_denominator,
    total_recombination,
    total_recombination_derivatives,
)
from perovskite_sim.solver.mol import MaterialArrays
from perovskite_sim.solver.small_signal import (
    FrequencyDomainResult,
    ProgressCallback,
    SmallSignalCurrentComponent,
    SmallSignalEvaluation,
    solve_frequency_domain,
)


class IonAwareAnalyticReactionCapabilityError(ValueError):
    """An active reaction closure has no declared analytic tangent."""


@dataclass(frozen=True, slots=True)
class IonAwareAnalyticBulkReactionLinearization:
    """Bulk SRH, radiative, and Auger rate derivatives."""

    recombination_rate_m3_s: np.ndarray
    electron_density_derivative_s1: np.ndarray
    hole_density_derivative_s1: np.ndarray
    rate_jacobian: np.ndarray
    finite_difference_rate_jacobian: np.ndarray
    rate_voltage_derivative: np.ndarray


def _node_recombination_rate(
    material: MaterialArrays,
    node: int,
    n: float,
    p: float,
) -> float:
    return float(
        total_recombination(
            np.asarray(n),
            np.asarray(p),
            float(material.ni_sq[node]),
            float(material.tau_n[node]),
            float(material.tau_p[node]),
            float(material.n1[node]),
            float(material.p1[node]),
            float(material.B_rad[node]),
            float(material.C_n[node]),
            float(material.C_p[node]),
        )
    )


def build_ion_aware_analytic_bulk_reaction_linearization(
    x: np.ndarray,
    stack: DeviceStack,
    base_state: np.ndarray,
    V_dc: float,
    material: MaterialArrays,
    layout: IonAwareStateCoordinateLayout,
    *,
    potential_at_operating_point_V: np.ndarray,
    state_steps: np.ndarray,
) -> IonAwareAnalyticBulkReactionLinearization:
    """Assemble the local bulk-recombination tangent in scaled log coordinates."""

    grid = np.asarray(x, dtype=float)
    state = np.asarray(base_state, dtype=float)
    potential = np.asarray(potential_at_operating_point_V, dtype=float)
    steps = np.asarray(state_steps, dtype=float)
    expected_state_size = (4 if material.has_dual_ions else 3) * grid.size
    if (
        grid.ndim != 1
        or grid.size < 3
        or np.any(np.diff(grid) <= 0.0)
        or not np.all(np.isfinite(grid))
        or layout.n_nodes != grid.size
        or state.shape != (expected_state_size,)
        or not np.all(np.isfinite(state))
        or potential.shape != grid.shape
        or not np.all(np.isfinite(potential))
        or steps.shape != (layout.size,)
        or not np.all(np.isfinite(steps))
        or np.any(steps <= 0.0)
        or not np.isfinite(V_dc)
        or not np.isfinite(material.het_recomb_despike)
    ):
        raise IonAwareAnalyticReactionCapabilityError(
            "analytic bulk-reaction inputs must be finite and shape matched"
        )
    if material.has_radiative_reabsorption:
        raise IonAwareAnalyticReactionCapabilityError(
            "self-consistent radiative reabsorption has no analytic nonlocal tangent"
        )
    if material.het_recomb_despike > 0.0 and material.het_recomb_nodes:
        raise IonAwareAnalyticReactionCapabilityError(
            "heterojunction recombination de-spike has no analytic cross-node tangent"
        )

    n, p, _phi, _state_vector = _state_fields(
        grid,
        state,
        stack,
        V_dc,
        material,
        phi_frozen=potential,
    )
    denominator = bulk_srh_denominator(
        n,
        p,
        material.tau_n,
        material.tau_p,
        material.n1,
        material.p1,
    )
    if not np.all(np.isfinite(denominator)) or np.any(denominator <= 0.0):
        raise IonAwareAnalyticReactionCapabilityError(
            "bulk SRH denominator must be finite and positive"
        )
    derivatives = total_recombination_derivatives(
        n,
        p,
        material.ni_sq,
        material.tau_n,
        material.tau_p,
        material.n1,
        material.p1,
        material.B_rad,
        material.C_n,
        material.C_p,
    )
    derivative_arrays = (
        derivatives.rate,
        derivatives.electron_density_derivative,
        derivatives.hole_density_derivative,
    )
    if any(
        np.asarray(value).shape != grid.shape
        or not np.all(np.isfinite(value))
        for value in derivative_arrays
    ):
        raise IonAwareAnalyticReactionCapabilityError(
            "analytic bulk-reaction formula produced a non-finite block"
        )

    analytic = np.zeros((layout.size, layout.size), dtype=float)
    finite_difference = np.zeros_like(analytic)
    row_by_state_index = {
        state_index: row for row, state_index in enumerate(layout.state_indices)
    }
    target_rows = tuple(
        (
            row_by_state_index.get(node),
            row_by_state_index.get(grid.size + node),
        )
        for node in range(grid.size)
    )

    for species, density, density_derivative in (
        ("electron", n, derivatives.electron_density_derivative),
        ("hole", p, derivatives.hole_density_derivative),
    ):
        coordinate_slice = layout.coordinate_slice(species)
        columns = np.arange(layout.size)[coordinate_slice]
        nodes = layout.node_indices(species)
        if columns.size != nodes.size:
            raise IonAwareAnalyticReactionCapabilityError(
                "reaction state layout and carrier coordinates are inconsistent"
            )
        for column, node in zip(columns, nodes, strict=True):
            step = float(steps[column])
            tangent = float(density[node]) * step
            analytic_recombination = float(density_derivative[node]) * tangent
            n_plus = float(n[node])
            n_minus = n_plus
            p_plus = float(p[node])
            p_minus = p_plus
            if species == "electron":
                n_plus *= float(np.exp(step))
                n_minus *= float(np.exp(-step))
            else:
                p_plus *= float(np.exp(step))
                p_minus *= float(np.exp(-step))
            finite_recombination = 0.5 * (
                _node_recombination_rate(material, int(node), n_plus, p_plus)
                - _node_recombination_rate(material, int(node), n_minus, p_minus)
            )
            for row in target_rows[int(node)]:
                if row is not None:
                    analytic[row, column] = -analytic_recombination
                    finite_difference[row, column] = -finite_recombination

    voltage_derivative = np.zeros(layout.size, dtype=float)
    arrays = (analytic, finite_difference, voltage_derivative)
    if any(not np.all(np.isfinite(value)) for value in arrays):
        raise IonAwareAnalyticReactionCapabilityError(
            "analytic bulk-reaction assembly produced a non-finite operator"
        )
    return IonAwareAnalyticBulkReactionLinearization(
        recombination_rate_m3_s=np.asarray(derivatives.rate, dtype=float),
        electron_density_derivative_s1=np.asarray(
            derivatives.electron_density_derivative,
            dtype=float,
        ),
        hole_density_derivative_s1=np.asarray(
            derivatives.hole_density_derivative,
            dtype=float,
        ),
        rate_jacobian=analytic,
        finite_difference_rate_jacobian=finite_difference,
        rate_voltage_derivative=voltage_derivative,
    )


def apply_analytic_bulk_reaction_linearization(
    reference: FrequencyDomainResult,
    analytic: IonAwareAnalyticBulkReactionLinearization,
    layout: IonAwareStateCoordinateLayout,
    *,
    V_dc: float,
    face_weights: np.ndarray,
    progress: ProgressCallback | None = None,
) -> FrequencyDomainResult:
    """Replace only the bulk-recombination part of a linearized rate block."""

    state_shape = (layout.size, layout.size)
    weights = np.asarray(face_weights, dtype=float)
    face_count = reference.admittance_faces.shape[1]
    reaction_arrays = (
        analytic.rate_jacobian,
        analytic.finite_difference_rate_jacobian,
        analytic.rate_voltage_derivative,
    )
    if (
        reference.rate_jacobian.shape != state_shape
        or analytic.rate_jacobian.shape != state_shape
        or analytic.finite_difference_rate_jacobian.shape != state_shape
        or analytic.rate_voltage_derivative.shape != (layout.size,)
        or weights.shape != (face_count,)
        or not np.all(np.isfinite(weights))
        or np.any(weights < 0.0)
        or float(np.sum(weights)) <= 0.0
        or not np.isfinite(V_dc)
        or any(not np.all(np.isfinite(value)) for value in reaction_arrays)
    ):
        raise IonAwareAnalyticReactionCapabilityError(
            "bulk-reaction correction inputs must be finite and shape matched"
        )
    rate_jacobian = (
        reference.rate_jacobian
        - analytic.finite_difference_rate_jacobian
        + analytic.rate_jacobian
    )
    rate_voltage = (
        reference.rate_voltage_derivative + analytic.rate_voltage_derivative
    )

    def evaluate(coordinate: np.ndarray, voltage: float) -> SmallSignalEvaluation:
        state_coordinate = np.asarray(coordinate, dtype=float)
        voltage_increment = float(voltage) - V_dc
        components = tuple(
            SmallSignalCurrentComponent(
                component.name,
                component.current_jacobian @ state_coordinate
                + component.voltage_derivative * voltage_increment,
            )
            for component in reference.current_components
        )
        conduction = sum(
            (component.current_faces for component in components),
            start=np.zeros(face_count, dtype=float),
        )
        return SmallSignalEvaluation(
            storage=(
                reference.storage_at_operating_point
                + reference.mass_matrix @ state_coordinate
                + reference.storage_voltage_derivative * voltage_increment
            ),
            rate=(
                rate_jacobian @ state_coordinate
                + rate_voltage * voltage_increment
            ),
            conduction_current_faces=conduction,
            displacement_charge_faces=(
                reference.displacement_charge_jacobian @ state_coordinate
                + reference.displacement_charge_voltage_derivative
                * voltage_increment
            ),
            current_components=components,
        )

    return solve_frequency_domain(
        evaluate,
        np.zeros(layout.size, dtype=float),
        V_dc,
        reference.frequencies,
        state_step=reference.state_step,
        voltage_step=reference.voltage_step,
        face_weights=weights,
        progress=progress,
    )


__all__ = [
    "IonAwareAnalyticBulkReactionLinearization",
    "IonAwareAnalyticReactionCapabilityError",
    "apply_analytic_bulk_reaction_linearization",
    "build_ion_aware_analytic_bulk_reaction_linearization",
]
