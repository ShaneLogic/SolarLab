"""Analytic Scharfetter-Gummel current blocks for ion-aware impedance."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from perovskite_sim.constants import Q
from perovskite_sim.discretization.fe_operators import (
    ScharfetterGummelFaceJacobian,
    sg_fluxes_n_jacobian,
    sg_fluxes_p_jacobian,
    thermionic_emission_flux,
)
from perovskite_sim.experiments.ion_aware_impedance import (
    IonAwareStateCoordinateLayout,
)
from perovskite_sim.experiments.jv_sweep import _state_fields
from perovskite_sim.models.device import DeviceStack
from perovskite_sim.physics.ion_migration import (
    IonFaceFluxJacobian,
    ion_face_flux_jacobian,
)
from perovskite_sim.solver.mol import MaterialArrays
from perovskite_sim.solver.small_signal import (
    FrequencyDomainResult,
    ProgressCallback,
    SmallSignalCurrentComponent,
    SmallSignalEvaluation,
    solve_frequency_domain,
)


class IonAwareAnalyticTransportCapabilityError(ValueError):
    """The active transport closure has no declared analytic tangent."""


@dataclass(frozen=True, slots=True)
class AnalyticCurrentComponentLinearization:
    """One all-face current component and its small-signal derivatives."""

    name: str
    current_faces: np.ndarray
    current_jacobian: np.ndarray
    voltage_derivative: np.ndarray


@dataclass(frozen=True, slots=True)
class IonAwareAnalyticTransportLinearization:
    """Analytic carrier and ion current blocks in scaled log coordinates."""

    current_components: tuple[AnalyticCurrentComponentLinearization, ...]
    conduction_current_faces: np.ndarray
    conduction_current_jacobian: np.ndarray
    conduction_current_voltage_derivative: np.ndarray


def _face_density_of_states(
    density_of_states: np.ndarray | None,
    face: int,
) -> float | None:
    if density_of_states is None:
        return None
    left = float(density_of_states[face])
    right = float(density_of_states[face + 1])
    if not (
        np.isfinite(left)
        and np.isfinite(right)
        and left > 0.0
        and right > 0.0
    ):
        return None
    return float(np.sqrt(left * right))


def _require_inactive_thermionic_caps(
    material: MaterialArrays,
    n: np.ndarray,
    p: np.ndarray,
    electron_flux: np.ndarray,
    hole_flux: np.ndarray,
) -> None:
    if material.iface_qss_exclusive_transport:
        raise IonAwareAnalyticTransportCapabilityError(
            "exclusive interface transport has no analytic SG rate block"
        )
    if material.te_softness > 0.0:
        raise IonAwareAnalyticTransportCapabilityError(
            "smoothed thermionic-cap derivatives are not implemented"
        )
    if not material.interface_faces:
        return
    if material.A_star_n is None or material.A_star_p is None:
        raise IonAwareAnalyticTransportCapabilityError(
            "thermionic-cap material arrays are incomplete"
        )
    chi = material.chi if material.chi_phys is None else material.chi_phys
    Eg = material.Eg if material.Eg_phys is None else material.Eg_phys
    for face in material.interface_faces:
        electron_offset = float(chi[face] - chi[face + 1])
        if abs(electron_offset) > 0.05:
            electron_bound = thermionic_emission_flux(
                float(n[face]),
                float(n[face + 1]),
                electron_offset,
                material.T_device,
                float(material.A_star_n[face]),
                N_dos=(
                    _face_density_of_states(material.N_C_node, face)
                    if material.te_physical_norm
                    else None
                ),
            )
            if abs(electron_flux[face]) * 1.01 >= abs(electron_bound):
                raise IonAwareAnalyticTransportCapabilityError(
                    "electron thermionic cap is active or within one percent "
                    "of its switching surface"
                )
        hole_offset = float(
            (chi[face] + Eg[face])
            - (chi[face + 1] + Eg[face + 1])
        )
        if abs(hole_offset) > 0.05:
            hole_bound = thermionic_emission_flux(
                float(p[face]),
                float(p[face + 1]),
                hole_offset,
                material.T_device,
                float(material.A_star_p[face]),
                N_dos=(
                    _face_density_of_states(material.N_V_node, face)
                    if material.te_physical_norm
                    else None
                ),
            )
            if abs(hole_flux[face]) * 1.01 >= abs(hole_bound):
                raise IonAwareAnalyticTransportCapabilityError(
                    "hole thermionic cap is active or within one percent "
                    "of its switching surface"
                )


def _add_direct_density_derivatives(
    matrix: np.ndarray,
    *,
    density: np.ndarray,
    left_derivative: np.ndarray,
    right_derivative: np.ndarray,
    coordinate_slice: slice,
    nodes: np.ndarray,
    state_steps: np.ndarray,
) -> None:
    columns = np.arange(matrix.shape[1])[coordinate_slice]
    if columns.size != nodes.size:
        raise IonAwareAnalyticTransportCapabilityError(
            "state layout and species coordinate slice are inconsistent"
        )
    for column, node in zip(columns, nodes, strict=True):
        density_tangent = density[node] * state_steps[column]
        if node < density.size - 1:
            matrix[node, column] += left_derivative[node] * density_tangent
        if node > 0:
            matrix[node - 1, column] += (
                right_derivative[node - 1] * density_tangent
            )


def _face_component(
    *,
    name: str,
    local: ScharfetterGummelFaceJacobian | IonFaceFluxJacobian,
    current_factor: float,
    density: np.ndarray,
    species: str,
    layout: IonAwareStateCoordinateLayout,
    state_steps: np.ndarray,
    potential_coordinate_jacobian: np.ndarray,
    potential_voltage_derivative: np.ndarray,
    partner_density: np.ndarray | None = None,
    partner_species: str | None = None,
) -> AnalyticCurrentComponentLinearization:
    matrix = (
        local.potential_left_derivative[:, None]
        * potential_coordinate_jacobian[:-1]
        + local.potential_right_derivative[:, None]
        * potential_coordinate_jacobian[1:]
    )
    _add_direct_density_derivatives(
        matrix,
        density=density,
        left_derivative=local.density_left_derivative,
        right_derivative=local.density_right_derivative,
        coordinate_slice=layout.coordinate_slice(species),
        nodes=layout.node_indices(species),
        state_steps=state_steps,
    )
    if partner_density is not None and partner_species is not None:
        if not isinstance(local, IonFaceFluxJacobian):
            raise IonAwareAnalyticTransportCapabilityError(
                "only ionic transport can declare a partner-density tangent"
            )
        _add_direct_density_derivatives(
            matrix,
            density=partner_density,
            left_derivative=local.partner_left_derivative,
            right_derivative=local.partner_right_derivative,
            coordinate_slice=layout.coordinate_slice(partner_species),
            nodes=layout.node_indices(partner_species),
            state_steps=state_steps,
        )
    voltage = (
        local.potential_left_derivative * potential_voltage_derivative[:-1]
        + local.potential_right_derivative * potential_voltage_derivative[1:]
    )
    return AnalyticCurrentComponentLinearization(
        name=name,
        current_faces=current_factor * np.asarray(local.flux, dtype=float),
        current_jacobian=current_factor * matrix,
        voltage_derivative=current_factor * voltage,
    )


def build_ion_aware_analytic_transport_linearization(
    x: np.ndarray,
    stack: DeviceStack,
    base_state: np.ndarray,
    V_dc: float,
    material: MaterialArrays,
    layout: IonAwareStateCoordinateLayout,
    *,
    potential_at_operating_point_V: np.ndarray,
    potential_state_jacobian_V: np.ndarray,
    potential_voltage_derivative: np.ndarray,
    state_steps: np.ndarray,
) -> IonAwareAnalyticTransportLinearization:
    """Assemble analytic all-face current derivatives at one DC state.

    The returned columns use the same scaled log coordinates as the structured
    comparison: ``u_j = state_steps[j] * coordinate_j``.  Electrostatic chain
    terms therefore use the exact discrete Poisson sensitivity multiplied by
    the same per-column step.
    """
    grid = np.asarray(x, dtype=float)
    potential = np.asarray(potential_at_operating_point_V, dtype=float)
    sensitivity = np.asarray(potential_state_jacobian_V, dtype=float)
    voltage_sensitivity = np.asarray(potential_voltage_derivative, dtype=float)
    steps = np.asarray(state_steps, dtype=float)
    state = np.asarray(base_state, dtype=float)
    if (
        grid.ndim != 1
        or grid.size < 3
        or np.any(np.diff(grid) <= 0.0)
        or not np.all(np.isfinite(grid))
        or potential.shape != grid.shape
        or voltage_sensitivity.shape != grid.shape
        or sensitivity.shape != (grid.size, layout.size)
        or steps.shape != (layout.size,)
        or not np.all(np.isfinite(potential))
        or not np.all(np.isfinite(voltage_sensitivity))
        or not np.all(np.isfinite(sensitivity))
        or not np.all(np.isfinite(steps))
        or np.any(steps <= 0.0)
        or not np.isfinite(V_dc)
    ):
        raise IonAwareAnalyticTransportCapabilityError(
            "analytic transport inputs must be finite and shape matched"
        )
    expected_state_size = (4 if material.has_dual_ions else 3) * grid.size
    if state.shape != (expected_state_size,) or not np.all(np.isfinite(state)):
        raise IonAwareAnalyticTransportCapabilityError(
            "analytic transport requires the packed bulk ion-aware state"
        )
    if material.N_iface_state:
        raise IonAwareAnalyticTransportCapabilityError(
            "dynamic interface-state transport has no analytic current block"
        )
    if material.has_field_mobility:
        raise IonAwareAnalyticTransportCapabilityError(
            "field-dependent mobility derivatives are not implemented"
        )

    n, p, _phi, state_vector = _state_fields(
        grid,
        state,
        stack,
        V_dc,
        material,
        phi_frozen=potential,
    )
    spacing = np.diff(grid)
    potential_coordinate_jacobian = sensitivity * steps[None, :]
    polarity = float(material.junction_polarity)
    electron_local = sg_fluxes_n_jacobian(
        potential + material.chi,
        n,
        spacing,
        material.D_n_face,
        material.V_T_device,
    )
    hole_local = sg_fluxes_p_jacobian(
        potential + material.chi + material.Eg,
        p,
        spacing,
        material.D_p_face,
        material.V_T_device,
    )
    _require_inactive_thermionic_caps(
        material,
        n,
        p,
        electron_local.flux,
        hole_local.flux,
    )
    components = [
        _face_component(
            name="electron",
            local=electron_local,
            current_factor=polarity,
            density=n,
            species="electron",
            layout=layout,
            state_steps=steps,
            potential_coordinate_jacobian=potential_coordinate_jacobian,
            potential_voltage_derivative=voltage_sensitivity,
        ),
        _face_component(
            name="hole",
            local=hole_local,
            current_factor=polarity,
            density=p,
            species="hole",
            layout=layout,
            state_steps=steps,
            potential_coordinate_jacobian=potential_coordinate_jacobian,
            potential_voltage_derivative=voltage_sensitivity,
        ),
    ]

    shared_site = (
        material.ion_steric_diffusion_only
        and material.ion_steric_shared_site
        and material.has_dual_ions
        and state_vector.P_neg is not None
    )
    positive_local = ion_face_flux_jacobian(
        potential,
        state_vector.P,
        spacing,
        material.D_ion_face,
        material.V_T_device,
        material.P_lim_face,
        steric_diffusion_only=material.ion_steric_diffusion_only,
        P_lim_node=material.P_lim_node,
        P_other_node=(state_vector.P_neg if shared_site else None),
        drift_sign=1.0,
    )
    if not np.all(positive_local.differentiable_faces):
        raise IonAwareAnalyticTransportCapabilityError(
            "positive-ion operating point touches a steric clipping kink"
        )
    components.append(
        _face_component(
            name="positive_ion",
            local=positive_local,
            current_factor=polarity * Q,
            density=state_vector.P,
            species="positive_ion",
            layout=layout,
            state_steps=steps,
            potential_coordinate_jacobian=potential_coordinate_jacobian,
            potential_voltage_derivative=voltage_sensitivity,
            partner_density=(state_vector.P_neg if shared_site else None),
            partner_species=("negative_ion" if shared_site else None),
        )
    )
    if material.has_dual_ions and state_vector.P_neg is not None:
        if (
            material.D_ion_neg_face is None
            or material.P_lim_neg_face is None
            or material.P_lim_neg_node is None
        ):
            raise IonAwareAnalyticTransportCapabilityError(
                "dual-ion material arrays are incomplete"
            )
        negative_local = ion_face_flux_jacobian(
            potential,
            state_vector.P_neg,
            spacing,
            material.D_ion_neg_face,
            material.V_T_device,
            material.P_lim_neg_face,
            steric_diffusion_only=material.ion_steric_diffusion_only,
            P_lim_node=material.P_lim_neg_node,
            P_other_node=(state_vector.P if shared_site else None),
            drift_sign=-1.0,
        )
        if not np.all(negative_local.differentiable_faces):
            raise IonAwareAnalyticTransportCapabilityError(
                "negative-ion operating point touches a steric clipping kink"
            )
        components.append(
            _face_component(
                name="negative_ion",
                local=negative_local,
                current_factor=-polarity * Q,
                density=state_vector.P_neg,
                species="negative_ion",
                layout=layout,
                state_steps=steps,
                potential_coordinate_jacobian=potential_coordinate_jacobian,
                potential_voltage_derivative=voltage_sensitivity,
                partner_density=(state_vector.P if shared_site else None),
                partner_species=("positive_ion" if shared_site else None),
            )
        )

    component_tuple = tuple(components)
    faces = sum(
        (component.current_faces for component in component_tuple),
        start=np.zeros(grid.size - 1, dtype=float),
    )
    jacobian = sum(
        (component.current_jacobian for component in component_tuple),
        start=np.zeros((grid.size - 1, layout.size), dtype=float),
    )
    voltage = sum(
        (component.voltage_derivative for component in component_tuple),
        start=np.zeros(grid.size - 1, dtype=float),
    )
    arrays = (faces, jacobian, voltage)
    if any(not np.all(np.isfinite(array)) for array in arrays):
        raise IonAwareAnalyticTransportCapabilityError(
            "analytic current assembly produced a non-finite block"
        )
    return IonAwareAnalyticTransportLinearization(
        current_components=component_tuple,
        conduction_current_faces=faces,
        conduction_current_jacobian=jacobian,
        conduction_current_voltage_derivative=voltage,
    )


def apply_analytic_transport_linearization(
    reference: FrequencyDomainResult,
    analytic: IonAwareAnalyticTransportLinearization,
    material: MaterialArrays,
    layout: IonAwareStateCoordinateLayout,
    *,
    V_dc: float,
    face_weights: np.ndarray,
    progress: ProgressCallback | None = None,
) -> FrequencyDomainResult:
    """Replace SG currents and their matching continuity-divergence blocks.

    Reaction, generation, interface recombination, and contact terms remain
    the independently evaluated frozen-potential finite differences in
    ``reference``.  The correction converts only the carrier/ion face-current
    truncation error into the corresponding conservative rate correction,
    keeping the frequency-domain continuity and Ampere rows mutually closed.
    """
    expected_names = tuple(component.name for component in reference.current_components)
    analytic_names = tuple(component.name for component in analytic.current_components)
    if expected_names != analytic_names:
        raise IonAwareAnalyticTransportCapabilityError(
            "analytic and finite-difference current component names differ"
        )
    weights = np.asarray(face_weights, dtype=float)
    face_count = reference.admittance_faces.shape[1]
    if (
        weights.shape != (face_count,)
        or not np.all(np.isfinite(weights))
        or np.any(weights < 0.0)
        or float(np.sum(weights)) <= 0.0
    ):
        raise IonAwareAnalyticTransportCapabilityError(
            "face weights must be finite, non-negative, and shape matched"
        )
    if (
        reference.rate_jacobian.shape != (layout.size, layout.size)
        or material.dx_cell.shape != (layout.n_nodes,)
        or not np.isfinite(V_dc)
        or material.junction_polarity not in (-1.0, 1.0)
    ):
        raise IonAwareAnalyticTransportCapabilityError(
            "finite-difference rate block and transport layout are inconsistent"
        )
    finite_components = {
        component.name: component for component in reference.current_components
    }
    analytic_components = {
        component.name: component for component in analytic.current_components
    }
    rate_jacobian = reference.rate_jacobian.copy()
    rate_voltage = reference.rate_voltage_derivative.copy()
    row_by_state_index = {
        state_index: row for row, state_index in enumerate(layout.state_indices)
    }
    signed_divergence = {
        "electron": 1.0,
        "hole": -1.0,
        "positive_ion": -1.0,
        "negative_ion": 1.0,
    }
    state_indices = {
        "electron": layout.electron_state_indices,
        "hole": layout.hole_state_indices,
        "positive_ion": layout.positive_ion_state_indices,
        "negative_ion": layout.negative_ion_state_indices,
    }
    charge_scale = float(material.junction_polarity) * Q
    for name in expected_names:
        current_difference = (
            analytic_components[name].current_jacobian
            - finite_components[name].current_jacobian
        )
        voltage_difference = (
            analytic_components[name].voltage_derivative
            - finite_components[name].voltage_derivative
        )
        coefficient = signed_divergence[name] / charge_scale
        for state_index in state_indices[name]:
            node = state_index % layout.n_nodes
            row = row_by_state_index[state_index]
            right = (
                current_difference[node]
                if node < layout.n_nodes - 1
                else 0.0
            )
            left = current_difference[node - 1] if node > 0 else 0.0
            rate_jacobian[row] += (
                coefficient * (right - left) / material.dx_cell[node]
            )
            right_voltage = (
                voltage_difference[node]
                if node < layout.n_nodes - 1
                else 0.0
            )
            left_voltage = voltage_difference[node - 1] if node > 0 else 0.0
            rate_voltage[row] += (
                coefficient
                * (right_voltage - left_voltage)
                / material.dx_cell[node]
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
            for component in analytic.current_components
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
    "AnalyticCurrentComponentLinearization",
    "IonAwareAnalyticTransportCapabilityError",
    "IonAwareAnalyticTransportLinearization",
    "apply_analytic_transport_linearization",
    "build_ion_aware_analytic_transport_linearization",
]
