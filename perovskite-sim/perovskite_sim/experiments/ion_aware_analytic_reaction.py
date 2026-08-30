"""Analytic local rate blocks for ion-aware impedance."""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from perovskite_sim.constants import Q
from perovskite_sim.experiments.ion_aware_impedance import (
    IonAwareStateCoordinateLayout,
)
from perovskite_sim.experiments.jv_sweep import _state_fields
from perovskite_sim.models.device import (
    DeviceStack,
    electrical_interface_defects,
    electrical_interfaces,
)
from perovskite_sim.physics.contacts import selective_contact_flux
from perovskite_sim.physics.recombination import (
    bulk_recombination_denominators,
    interface_recombination,
    interface_recombination_derivatives,
    interface_srh_denominator,
    total_recombination_at_node,
    total_recombination_derivatives,
)
from perovskite_sim.solver.mol import (
    MaterialArrays,
    _IFACE_PROJ_EXP_CAP,
    _QSS_V_TH_MS,
    _qss_interface_R,
)
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
    finite_difference_rate_voltage_derivative: np.ndarray


@dataclass(frozen=True, slots=True)
class IonAwareAnalyticInterfaceReactionLinearization:
    """Certified smooth interface-SRH and implicit QSS derivatives."""

    interface_nodes: tuple[int, ...]
    electron_evaluation_nodes: tuple[int, ...]
    hole_evaluation_nodes: tuple[int, ...]
    cross_node_interface_indices: tuple[int, ...]
    projected_interface_indices: tuple[int, ...]
    shared_occupancy_interface_indices: tuple[int, ...]
    two_sided_interface_indices: tuple[int, ...]
    qss_interface_indices: tuple[int, ...]
    minimum_cross_node_clamp_margin_m2_s: float | None
    minimum_projection_exponent_cap_margin: float | None
    minimum_shared_density_floor_margin_m3: float | None
    minimum_two_sided_clamp_margin_m2_s: float | None
    minimum_two_sided_density_floor_margin_m3: float | None
    qss_transport_velocity_m_s: float | None
    minimum_qss_supply_rate_margin_m2_s: float | None
    minimum_qss_root_headroom_m3: float | None
    maximum_qss_root_relative_residual: float | None
    surface_recombination_rate_m2_s: np.ndarray
    electron_density_derivative_m_s: np.ndarray
    hole_density_derivative_m_s: np.ndarray
    mirror_surface_recombination_rate_m2_s: np.ndarray
    mirror_electron_density_derivative_m_s: np.ndarray
    mirror_hole_density_derivative_m_s: np.ndarray
    rate_jacobian: np.ndarray
    finite_difference_rate_jacobian: np.ndarray
    complex_step_rate_jacobian: np.ndarray
    rate_voltage_derivative: np.ndarray
    finite_difference_rate_voltage_derivative: np.ndarray
    complex_step_rate_voltage_derivative: np.ndarray


@dataclass(frozen=True, slots=True)
class IonAwareAnalyticContactLinearization:
    """Finite-rate outer-contact contributions to carrier rate rows."""

    active_channels: tuple[str, ...]
    boundary_state_indices: tuple[int, ...]
    relaxation_rate_s1: np.ndarray
    rate_at_operating_point_m3_s: np.ndarray
    rate_jacobian: np.ndarray
    finite_difference_rate_jacobian: np.ndarray
    rate_voltage_derivative: np.ndarray
    finite_difference_rate_voltage_derivative: np.ndarray


@dataclass(frozen=True, slots=True)
class _QSSInteriorEvaluation:
    rate_m2_s: float
    supply_rate_m2_s: float
    depletion_m3: float
    root_headroom_m3: float
    relative_root_residual: float
    electron_density_derivative_m_s: float
    hole_density_derivative_m_s: float
    reference_derivative_m4_s: float


def _evaluate_qss_interior_rate(
    n_supply: float,
    p_supply: float,
    ni_sq_reference: float,
    n1: float,
    p1: float,
    v_n: float,
    v_p: float,
    transport_velocity_m_s: float,
    *,
    max_relative_root_residual: float = 1.0e-6,
) -> _QSSInteriorEvaluation:
    supply_rate = float(
        interface_recombination(
            n_supply,
            p_supply,
            ni_sq_reference,
            n1,
            p1,
            v_n,
            v_p,
        )
    )
    if not np.isfinite(supply_rate) or supply_rate <= 0.0:
        raise IonAwareAnalyticReactionCapabilityError(
            "QSS interface supply rate must remain strictly positive"
        )
    upper = min(n_supply, p_supply) * (1.0 - 1.0e-9)
    if not np.isfinite(upper) or upper <= 0.0:
        raise IonAwareAnalyticReactionCapabilityError(
            "QSS interface root bracket must be finite and positive"
        )
    upper_residual = transport_velocity_m_s * upper - float(
        interface_recombination(
            n_supply - upper,
            p_supply - upper,
            ni_sq_reference,
            n1,
            p1,
            v_n,
            v_p,
        )
    )
    if not np.isfinite(upper_residual) or upper_residual <= 0.0:
        raise IonAwareAnalyticReactionCapabilityError(
            "QSS interface transport-limited branch must be inactive"
        )
    rate = float(
        _qss_interface_R(
            n_supply,
            p_supply,
            ni_sq_reference,
            n1,
            p1,
            v_n,
            v_p,
            transport_velocity_m_s,
        )
    )
    depletion = rate / transport_velocity_m_s
    headroom = min(depletion, upper - depletion)
    if (
        not np.isfinite(rate)
        or not np.isfinite(depletion)
        or not np.isfinite(headroom)
        or rate <= 0.0
        or headroom <= 0.0
    ):
        raise IonAwareAnalyticReactionCapabilityError(
            "QSS interface root must lie strictly inside its bracket"
        )
    depleted_n = n_supply - depletion
    depleted_p = p_supply - depletion
    inner_rate = float(
        interface_recombination(
            depleted_n,
            depleted_p,
            ni_sq_reference,
            n1,
            p1,
            v_n,
            v_p,
        )
    )
    residual_scale = max(abs(rate), abs(inner_rate), np.finfo(float).tiny)
    relative_residual = abs(rate - inner_rate) / residual_scale
    if (
        not np.isfinite(relative_residual)
        or relative_residual > max_relative_root_residual
    ):
        raise IonAwareAnalyticReactionCapabilityError(
            "QSS interface production root is not residual-resolved"
        )
    inner_denominator = float(
        interface_srh_denominator(
            depleted_n,
            depleted_p,
            n1,
            p1,
            v_n,
            v_p,
        )
    )
    inner_derivatives = interface_recombination_derivatives(
        depleted_n,
        depleted_p,
        ni_sq_reference,
        n1,
        p1,
        v_n,
        v_p,
    )
    implicit_denominator = transport_velocity_m_s + float(
        inner_derivatives.electron_density_derivative
        + inner_derivatives.hole_density_derivative
    )
    if (
        not np.isfinite(inner_denominator)
        or inner_denominator <= 0.0
        or not np.isfinite(implicit_denominator)
        or implicit_denominator <= 0.0
    ):
        raise IonAwareAnalyticReactionCapabilityError(
            "QSS interface implicit derivative denominator must be positive"
        )
    derivative_scale = transport_velocity_m_s / implicit_denominator
    return _QSSInteriorEvaluation(
        rate_m2_s=rate,
        supply_rate_m2_s=supply_rate,
        depletion_m3=depletion,
        root_headroom_m3=headroom,
        relative_root_residual=relative_residual,
        electron_density_derivative_m_s=(
            derivative_scale
            * float(inner_derivatives.electron_density_derivative)
        ),
        hole_density_derivative_m_s=(
            derivative_scale
            * float(inner_derivatives.hole_density_derivative)
        ),
        reference_derivative_m4_s=(
            -derivative_scale / inner_denominator
        ),
    )


def _complex_qss_interior_rate(
    n_supply: complex,
    p_supply: complex,
    ni_sq_reference: complex,
    n1: float,
    p1: float,
    v_n: float,
    v_p: float,
    transport_velocity_m_s: float,
    depletion_at_operating_point_m3: float,
) -> complex:
    depletion = complex(depletion_at_operating_point_m3)
    for _ in range(6):
        derivatives = interface_recombination_derivatives(
            n_supply - depletion,
            p_supply - depletion,
            ni_sq_reference,
            n1,
            p1,
            v_n,
            v_p,
        )
        residual = transport_velocity_m_s * depletion - complex(
            derivatives.rate
        )
        derivative = transport_velocity_m_s + complex(
            derivatives.electron_density_derivative
            + derivatives.hole_density_derivative
        )
        if not np.isfinite(derivative) or abs(derivative) <= 0.0:
            raise IonAwareAnalyticReactionCapabilityError(
                "complex QSS interface Newton derivative is singular"
            )
        depletion -= residual / derivative
    rate = transport_velocity_m_s * depletion
    inner_rate = complex(
        interface_recombination(
            n_supply - depletion,
            p_supply - depletion,
            ni_sq_reference,
            n1,
            p1,
            v_n,
            v_p,
        )
    )
    residual = rate - inner_rate
    residual_scale = max(
        abs(rate),
        abs(inner_rate),
        np.finfo(float).tiny,
    )
    if (
        not np.isfinite(rate)
        or not np.isfinite(residual)
        or abs(residual) / residual_scale > 1.0e-12
    ):
        raise IonAwareAnalyticReactionCapabilityError(
            "complex QSS interface reference did not converge"
        )
    return rate


def _node_recombination_rate(
    material: MaterialArrays,
    node: int,
    n: float,
    p: float,
) -> float:
    # This lane forwards only the neutral inventory, so a charged or
    # multivalent explicit model would be silently replaced by the
    # effective-lifetime law. Those closures are certified only on the guarded
    # QF/DC lane; refuse rather than substitute different physics.
    if material.monovalent_bulk_defects is not None:
        raise IonAwareAnalyticReactionCapabilityError(
            "charged explicit bulk defects are closed only by the QF/DC lane"
        )
    if material.multivalent_bulk_defects is not None:
        raise IonAwareAnalyticReactionCapabilityError(
            "multivalent bulk defects are closed only by the QF/DC lane"
        )
    return total_recombination_at_node(
        n,
        p,
        float(material.ni_sq[node]),
        float(material.tau_n[node]),
        float(material.tau_p[node]),
        float(material.n1[node]),
        float(material.p1[node]),
        float(material.B_rad[node]),
        float(material.C_n[node]),
        float(material.C_p[node]),
        node=node,
        neutral_bulk_defects=material.neutral_bulk_defects,
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
    denominator = bulk_recombination_denominators(
        n,
        p,
        material.tau_n,
        material.tau_p,
        material.n1,
        material.p1,
        neutral_bulk_defects=material.neutral_bulk_defects,
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
        neutral_bulk_defects=material.neutral_bulk_defects,
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
        finite_difference_rate_voltage_derivative=np.zeros_like(
            voltage_derivative
        ),
    )


def build_ion_aware_analytic_interface_reaction_linearization(
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
    voltage_step: float,
    qss_transport_velocity_m_s: float = _QSS_V_TH_MS,
    max_qss_root_relative_residual: float = 1.0e-6,
) -> IonAwareAnalyticInterfaceReactionLinearization:
    """Assemble smooth interface-SRH tangents in log coordinates."""

    grid = np.asarray(x, dtype=float)
    state = np.asarray(base_state, dtype=float)
    potential = np.asarray(potential_at_operating_point_V, dtype=float)
    potential_jacobian = np.asarray(potential_state_jacobian_V, dtype=float)
    potential_voltage = np.asarray(potential_voltage_derivative, dtype=float)
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
        or potential_jacobian.shape != (grid.size, layout.size)
        or not np.all(np.isfinite(potential_jacobian))
        or potential_voltage.shape != grid.shape
        or not np.all(np.isfinite(potential_voltage))
        or steps.shape != (layout.size,)
        or not np.all(np.isfinite(steps))
        or np.any(steps <= 0.0)
        or not np.isfinite(V_dc)
        or not np.isfinite(voltage_step)
        or voltage_step <= 0.0
        or not np.isfinite(qss_transport_velocity_m_s)
        or qss_transport_velocity_m_s <= 0.0
        or not np.isfinite(max_qss_root_relative_residual)
        or max_qss_root_relative_residual <= 0.0
    ):
        raise IonAwareAnalyticReactionCapabilityError(
            "analytic interface-reaction inputs must be finite and shape matched"
        )
    if material.N_iface_state:
        raise IonAwareAnalyticReactionCapabilityError(
            "dynamic interface-plane states have no analytic reaction tangent"
        )
    if material.iface_qss_exclusive_transport:
        raise IonAwareAnalyticReactionCapabilityError(
            "exclusive interface transport has no analytic reaction tangent"
        )
    active_closures = tuple(
        name
        for name, active in (
            ("interface-plane closure", material.iface_plane_closure),
        )
        if active
    )
    if active_closures:
        raise IonAwareAnalyticReactionCapabilityError(
            f"{active_closures[0]} has no declared analytic tangent"
        )
    qss_active = os.environ.get("SOLARLAB_IFACE_QSS", "") == "1"
    if qss_active and qss_transport_velocity_m_s != _QSS_V_TH_MS:
        raise IonAwareAnalyticReactionCapabilityError(
            "QSS interface protocol velocity does not match production"
        )
    projection_active = bool(material.iface_plane_projection)
    shared_occupancy_active = bool(material.iface_shared_occ)
    two_sided_active = bool(material.iface_two_sided)
    thermal_voltage = float(material.V_T_device)
    if not np.isfinite(thermal_voltage) or thermal_voltage <= 0.0:
        raise IonAwareAnalyticReactionCapabilityError(
            "interface projection requires a finite positive thermal voltage"
        )

    interfaces = electrical_interfaces(stack)
    defects = electrical_interface_defects(stack)
    nodes = tuple(int(node) for node in material.interface_nodes)
    count = len(nodes)
    if len(interfaces) != count or len(defects) != count:
        raise IonAwareAnalyticReactionCapabilityError(
            "electrical interface topology and material nodes are not aligned"
        )
    if any(node < 0 or node >= grid.size for node in nodes):
        raise IonAwareAnalyticReactionCapabilityError(
            "interface node lies outside the electrical grid"
        )
    required_arrays = (
        ("interface_n1", material.interface_n1),
        ("interface_p1", material.interface_p1),
    )
    if any(len(values) != count for _name, values in required_arrays):
        raise IonAwareAnalyticReactionCapabilityError(
            "interface SRH parameter arrays are not topology aligned"
        )

    def optional_values(
        name: str,
        values: tuple[float, ...] | tuple[int, ...],
        defaults: tuple[float, ...] | tuple[int, ...],
    ) -> tuple[float, ...] | tuple[int, ...]:
        if not values:
            return defaults
        if len(values) != count:
            raise IonAwareAnalyticReactionCapabilityError(
                f"{name} is not topology aligned"
            )
        return values

    eval_n = tuple(
        int(value)
        for value in optional_values(
            "interface_eval_node_n",
            material.interface_eval_node_n,
            nodes,
        )
    )
    eval_p = tuple(
        int(value)
        for value in optional_values(
            "interface_eval_node_p",
            material.interface_eval_node_p,
            nodes,
        )
    )
    if any(
        value < 0 or value >= grid.size
        for value in (*eval_n, *eval_p)
    ):
        raise IonAwareAnalyticReactionCapabilityError(
            "interface evaluation node lies outside the electrical grid"
        )
    cross_node_indices: list[int] = []
    for index, (node, n_node, p_node, defect) in enumerate(
        zip(nodes, eval_n, eval_p, defects, strict=True)
    ):
        if defect is None and (n_node != node or p_node != node):
            raise IonAwareAnalyticReactionCapabilityError(
                "cross-node interface sampling requires a declared InterfaceDefect"
            )
        if defect is not None and (n_node == node or p_node == node):
            raise IonAwareAnalyticReactionCapabilityError(
                "declared InterfaceDefect is not represented by cross-node "
                "material sampling"
            )
        if defect is not None:
            cross_node_indices.append(index)
    if (
        cross_node_indices
        and os.environ.get("SOLARLAB_IFACE_ALLOW_GEN", "") == "1"
        and (not qss_active or shared_occupancy_active)
    ):
        raise IonAwareAnalyticReactionCapabilityError(
            "clamp-inactive cross-node interface tangents require "
            "SOLARLAB_IFACE_ALLOW_GEN to remain disabled"
        )
    ni_sq_eff = tuple(
        float(value)
        for value in optional_values(
            "interface_ni_sq_eff",
            material.interface_ni_sq_eff,
            tuple(float(material.ni_sq[node]) for node in nodes),
        )
    )
    calibration = tuple(
        float(value)
        for value in optional_values(
            "interface_calibration_factor",
            material.interface_calibration_factor,
            (1.0,) * count,
        )
    )
    if (
        not np.all(np.isfinite(ni_sq_eff))
        or np.any(np.asarray(ni_sq_eff) < 0.0)
        or not np.all(np.isfinite(calibration))
        or np.any(np.asarray(calibration) < 0.0)
    ):
        raise IonAwareAnalyticReactionCapabilityError(
            "interface references and calibration factors must be finite and nonnegative"
        )
    shared_arrays = (
        material.interface_n1_L,
        material.interface_p1_L,
        material.interface_n1_R,
        material.interface_p1_R,
        material.interface_n_L_eq,
        material.interface_p_L_eq,
        material.interface_n_R_eq,
        material.interface_p_R_eq,
    )
    if shared_occupancy_active and any(
        len(values) != count for values in shared_arrays
    ):
        raise IonAwareAnalyticReactionCapabilityError(
            "shared-occupancy interface arrays are not topology aligned"
        )
    if shared_occupancy_active:
        shared_values = np.asarray(
            [value for values in shared_arrays for value in values],
            dtype=float,
        )
        if (
            not np.all(np.isfinite(shared_values))
            or np.any(shared_values < 0.0)
        ):
            raise IonAwareAnalyticReactionCapabilityError(
                "shared-occupancy interface arrays must be finite and nonnegative"
            )
    two_sided_arrays = (
        material.interface_n_L_eq,
        material.interface_p_R_eq,
    )
    if two_sided_active and any(
        len(values) != count for values in two_sided_arrays
    ):
        raise IonAwareAnalyticReactionCapabilityError(
            "two-sided mirror interface arrays are not topology aligned"
        )
    if two_sided_active:
        two_sided_values = np.asarray(
            [value for values in two_sided_arrays for value in values],
            dtype=float,
        )
        if (
            not np.all(np.isfinite(two_sided_values))
            or np.any(two_sided_values < 0.0)
        ):
            raise IonAwareAnalyticReactionCapabilityError(
                "two-sided mirror interface arrays must be finite and nonnegative"
            )

    n, p, _phi, _state_vector = _state_fields(
        grid,
        state,
        stack,
        V_dc,
        material,
        phi_frozen=potential,
    )
    analytic = np.zeros((layout.size, layout.size), dtype=float)
    finite_difference = np.zeros_like(analytic)
    complex_step = np.zeros_like(analytic)
    voltage_derivative = np.zeros(layout.size, dtype=float)
    finite_difference_voltage = np.zeros(layout.size, dtype=float)
    complex_step_voltage = np.zeros(layout.size, dtype=float)
    surface_rate = np.zeros(count, dtype=float)
    electron_derivative = np.zeros(count, dtype=float)
    hole_derivative = np.zeros(count, dtype=float)
    mirror_surface_rate = np.zeros(count, dtype=float)
    mirror_electron_derivative = np.zeros(count, dtype=float)
    mirror_hole_derivative = np.zeros(count, dtype=float)
    minimum_cross_node_clamp_margin = np.inf
    clamp_certified_indices: list[int] = []
    projected_indices: list[int] = []
    minimum_projection_cap_margin = np.inf
    shared_occupancy_indices: list[int] = []
    minimum_shared_density_floor_margin = np.inf
    two_sided_indices: list[int] = []
    minimum_two_sided_clamp_margin = np.inf
    minimum_two_sided_density_floor_margin = np.inf
    qss_indices: list[int] = []
    qss_evaluations: list[_QSSInteriorEvaluation] = []
    row_by_state_index = {
        state_index: row
        for row, state_index in enumerate(layout.state_indices)
    }

    for index, (node, velocities) in enumerate(
        zip(nodes, interfaces, strict=True)
    ):
        dx_cell = float(material.dx_cell[node])
        electron_node = eval_n[index]
        hole_node = eval_p[index]
        is_cross_node = index in cross_node_indices
        is_shared_occupancy = shared_occupancy_active and is_cross_node
        if is_shared_occupancy:
            electron_components = (
                float(n[hole_node]),
                float(n[electron_node]),
            )
            hole_components = (
                float(p[hole_node]),
                float(p[electron_node]),
            )
            density_components = (*electron_components, *hole_components)
            if (
                not np.all(np.isfinite(density_components))
                or np.any(np.asarray(density_components) <= 0.0)
            ):
                raise IonAwareAnalyticReactionCapabilityError(
                    "shared-occupancy density floors must be inactive"
                )
            n_value = float(sum(electron_components))
            p_value = float(sum(hole_components))
            n1 = float(
                material.interface_n1_L[index]
                + material.interface_n1_R[index]
            )
            p1 = float(
                material.interface_p1_L[index]
                + material.interface_p1_R[index]
            )
            ni_reference = float(
                (
                    material.interface_n_L_eq[index]
                    + material.interface_n_R_eq[index]
                )
                * (
                    material.interface_p_L_eq[index]
                    + material.interface_p_R_eq[index]
                )
            )
            shared_occupancy_indices.append(index)
            minimum_shared_density_floor_margin = min(
                minimum_shared_density_floor_margin,
                *density_components,
            )
        else:
            electron_components = (float(n[electron_node]),)
            hole_components = (float(p[hole_node]),)
            n_value = electron_components[0]
            p_value = hole_components[0]
            n1 = float(material.interface_n1[index])
            p1 = float(material.interface_p1[index])
            ni_reference = ni_sq_eff[index]
        v_n = float(velocities[0]) * calibration[index]
        v_p = float(velocities[1]) * calibration[index]
        qss_branch = qss_active and not is_shared_occupancy
        is_qss = qss_branch and v_n > 0.0 and v_p > 0.0
        if qss_branch:
            qss_indices.append(index)
        is_projected = (
            (projection_active or qss_branch)
            and is_cross_node
            and not is_shared_occupancy
        )
        projection_log_n = 0.0
        projection_log_p = 0.0
        if is_projected:
            projection_log_n = (
                float(potential[node]) - float(potential[electron_node])
            ) / thermal_voltage
            projection_log_p = (
                float(potential[hole_node]) - float(potential[node])
            ) / thermal_voltage
            projection_margin = _IFACE_PROJ_EXP_CAP - max(
                abs(projection_log_n),
                abs(projection_log_p),
            )
            if not np.isfinite(projection_margin) or projection_margin <= 0.0:
                raise IonAwareAnalyticReactionCapabilityError(
                    "interface projection exponent cap is active at the "
                    "operating point"
                )
            projected_indices.append(index)
            minimum_projection_cap_margin = min(
                minimum_projection_cap_margin,
                projection_margin,
            )

        projection_factor_n = float(np.exp(projection_log_n))
        projection_factor_p = float(np.exp(projection_log_p))
        projected_n = n_value * projection_factor_n
        projected_p = p_value * projection_factor_p
        projected_ni_sq = (
            ni_reference * projection_factor_n * projection_factor_p
        )
        scalars = (
            dx_cell,
            n_value,
            p_value,
            projected_n,
            projected_p,
            projected_ni_sq,
            n1,
            p1,
            v_n,
            v_p,
        )
        if (
            not np.all(np.isfinite(scalars))
            or dx_cell <= 0.0
            or n_value <= 0.0
            or p_value <= 0.0
            or n1 < 0.0
            or p1 < 0.0
            or v_n < 0.0
            or v_p < 0.0
        ):
            raise IonAwareAnalyticReactionCapabilityError(
                "interface SRH inputs must be finite and physically admissible"
            )
        qss_base_evaluation: _QSSInteriorEvaluation | None = None
        if is_qss:
            qss_base_evaluation = _evaluate_qss_interior_rate(
                projected_n,
                projected_p,
                projected_ni_sq,
                n1,
                p1,
                v_n,
                v_p,
                qss_transport_velocity_m_s,
                max_relative_root_residual=(
                    max_qss_root_relative_residual
                ),
            )
            qss_evaluations.append(qss_base_evaluation)
            rate_value = qss_base_evaluation.rate_m2_s
            rate_electron_derivative = (
                qss_base_evaluation.electron_density_derivative_m_s
            )
            rate_hole_derivative = (
                qss_base_evaluation.hole_density_derivative_m_s
            )
            rate_reference_derivative = (
                qss_base_evaluation.reference_derivative_m4_s
            )
        else:
            if v_n > 0.0 and v_p > 0.0:
                denominator = interface_srh_denominator(
                    projected_n,
                    projected_p,
                    n1,
                    p1,
                    v_n,
                    v_p,
                )
                if not np.isfinite(denominator) or denominator <= 0.0:
                    raise IonAwareAnalyticReactionCapabilityError(
                        "interface SRH denominator must be finite and positive"
                    )
            else:
                denominator = np.inf
            derivatives = interface_recombination_derivatives(
                projected_n,
                projected_p,
                projected_ni_sq,
                n1,
                p1,
                v_n,
                v_p,
            )
            rate_value = float(derivatives.rate)
            rate_electron_derivative = float(
                derivatives.electron_density_derivative
            )
            rate_hole_derivative = float(
                derivatives.hole_density_derivative
            )
            rate_reference_derivative = -1.0 / denominator
        local_values = (
            rate_value,
            rate_electron_derivative,
            rate_hole_derivative,
            rate_reference_derivative,
        )
        if any(not np.all(np.isfinite(value)) for value in local_values):
            raise IonAwareAnalyticReactionCapabilityError(
                "analytic interface-reaction formula produced a non-finite block"
            )
        if is_cross_node and not qss_branch:
            if rate_value <= 0.0:
                raise IonAwareAnalyticReactionCapabilityError(
                    "cross-node interface clamp must be inactive at the "
                    "operating point (raw R_s > 0)"
                )
            clamp_certified_indices.append(index)
            minimum_cross_node_clamp_margin = min(
                minimum_cross_node_clamp_margin,
                rate_value,
            )
        surface_rate[index] = rate_value
        electron_derivative[index] = (
            rate_electron_derivative * projection_factor_n
        )
        hole_derivative[index] = (
            rate_hole_derivative * projection_factor_p
        )
        target_rows = tuple(
            row_by_state_index.get(state_index)
            for state_index in (node, grid.size + node)
        )

        def projected_rate(
            electron_density: float,
            hole_density: float,
            electron_log_factor: float,
            hole_log_factor: float,
        ) -> float:
            factor_n = float(np.exp(electron_log_factor))
            factor_p = float(np.exp(hole_log_factor))
            projected_electron = electron_density * factor_n
            projected_hole = hole_density * factor_p
            projected_reference = ni_reference * factor_n * factor_p
            if is_qss:
                evaluation = _evaluate_qss_interior_rate(
                    projected_electron,
                    projected_hole,
                    projected_reference,
                    n1,
                    p1,
                    v_n,
                    v_p,
                    qss_transport_velocity_m_s,
                    max_relative_root_residual=(
                        max_qss_root_relative_residual
                    ),
                )
                qss_evaluations.append(evaluation)
                return evaluation.rate_m2_s
            if qss_branch:
                return 0.0
            return interface_recombination(
                projected_electron,
                projected_hole,
                projected_reference,
                n1,
                p1,
                v_n,
                v_p,
            )

        def complex_projected_rate(
            electron_density: complex,
            hole_density: complex,
            electron_log_factor: complex,
            hole_log_factor: complex,
        ) -> complex:
            factor_n = np.exp(electron_log_factor)
            factor_p = np.exp(hole_log_factor)
            projected_electron = electron_density * factor_n
            projected_hole = hole_density * factor_p
            projected_reference = ni_reference * factor_n * factor_p
            if is_qss:
                if qss_base_evaluation is None:
                    raise IonAwareAnalyticReactionCapabilityError(
                        "QSS interface operating-point root is unavailable"
                    )
                return _complex_qss_interior_rate(
                    projected_electron,
                    projected_hole,
                    projected_reference,
                    n1,
                    p1,
                    v_n,
                    v_p,
                    qss_transport_velocity_m_s,
                    qss_base_evaluation.depletion_m3,
                )
            if qss_branch:
                return 0.0j
            return complex(
                interface_recombination(
                    projected_electron,
                    projected_hole,
                    projected_reference,
                    n1,
                    p1,
                    v_n,
                    v_p,
                )
            )

        if is_shared_occupancy:
            electron_component_by_state_index = {
                hole_node: electron_components[0],
                electron_node: electron_components[1],
            }
            hole_component_by_state_index = {
                grid.size + hole_node: hole_components[0],
                grid.size + electron_node: hole_components[1],
            }
        else:
            electron_component_by_state_index = {
                electron_node: electron_components[0],
            }
            hole_component_by_state_index = {
                grid.size + hole_node: hole_components[0],
            }

        for column, state_index in enumerate(layout.state_indices):
            step = float(steps[column])
            direct_n_component = float(
                electron_component_by_state_index.get(state_index, 0.0)
            )
            direct_p_component = float(
                hole_component_by_state_index.get(state_index, 0.0)
            )
            direct_n_increment = direct_n_component * step
            direct_p_increment = direct_p_component * step
            if is_shared_occupancy and (
                direct_n_component > 0.0 or direct_p_component > 0.0
            ):
                minimum_shared_density_floor_margin = min(
                    minimum_shared_density_floor_margin,
                    *(
                        value
                        for value in (
                            direct_n_component * float(np.exp(-step)),
                            direct_p_component * float(np.exp(-step)),
                        )
                        if value > 0.0
                    ),
                )
            projection_delta_n = 0.0
            projection_delta_p = 0.0
            if is_projected:
                potential_increment = potential_jacobian[:, column] * step
                projection_delta_n = (
                    float(potential_increment[node])
                    - float(potential_increment[electron_node])
                ) / thermal_voltage
                projection_delta_p = (
                    float(potential_increment[hole_node])
                    - float(potential_increment[node])
                ) / thermal_voltage
            reference_log_increment = projection_delta_n + projection_delta_p
            if (
                direct_n_increment == 0.0
                and direct_p_increment == 0.0
                and projection_delta_n == 0.0
                and projection_delta_p == 0.0
                and reference_log_increment == 0.0
            ):
                continue
            if is_projected:
                projection_margin = _IFACE_PROJ_EXP_CAP - max(
                    abs(projection_log_n + projection_delta_n),
                    abs(projection_log_n - projection_delta_n),
                    abs(projection_log_p + projection_delta_p),
                    abs(projection_log_p - projection_delta_p),
                )
                if (
                    not np.isfinite(projection_margin)
                    or projection_margin <= 0.0
                ):
                    raise IonAwareAnalyticReactionCapabilityError(
                        "interface projection state stencil reaches the "
                        "exponent cap"
                    )
                minimum_projection_cap_margin = min(
                    minimum_projection_cap_margin,
                    projection_margin,
                )
            projected_n_increment = projection_factor_n * (
                direct_n_increment + n_value * projection_delta_n
            )
            projected_p_increment = projection_factor_p * (
                direct_p_increment + p_value * projection_delta_p
            )
            analytic_recombination = (
                rate_electron_derivative * projected_n_increment
                + rate_hole_derivative * projected_p_increment
                + rate_reference_derivative
                * projected_ni_sq
                * reference_log_increment
            ) / dx_cell
            n_plus = n_value + direct_n_component * float(np.expm1(step))
            n_minus = n_value + direct_n_component * float(np.expm1(-step))
            p_plus = p_value + direct_p_component * float(np.expm1(step))
            p_minus = p_value + direct_p_component * float(np.expm1(-step))
            rate_plus = projected_rate(
                n_plus,
                p_plus,
                projection_log_n + projection_delta_n,
                projection_log_p + projection_delta_p,
            )
            rate_minus = projected_rate(
                n_minus,
                p_minus,
                projection_log_n - projection_delta_n,
                projection_log_p - projection_delta_p,
            )
            if is_cross_node and not qss_branch:
                stencil_minimum = min(float(rate_plus), float(rate_minus))
                if not np.isfinite(stencil_minimum) or stencil_minimum <= 0.0:
                    raise IonAwareAnalyticReactionCapabilityError(
                        "cross-node interface central stencil crosses the "
                        "inactive clamp branch"
                    )
                minimum_cross_node_clamp_margin = min(
                    minimum_cross_node_clamp_margin,
                    stencil_minimum,
                )
            finite_recombination = 0.5 * (
                rate_plus - rate_minus
            ) / dx_cell
            complex_epsilon = 1.0e-30
            complex_rate = complex_projected_rate(
                complex(
                    n_value
                    + direct_n_component
                    * np.expm1(1j * step * complex_epsilon)
                ),
                complex(
                    p_value
                    + direct_p_component
                    * np.expm1(1j * step * complex_epsilon)
                ),
                complex(
                    projection_log_n
                    + 1j * projection_delta_n * complex_epsilon
                ),
                complex(
                    projection_log_p
                    + 1j * projection_delta_p * complex_epsilon
                ),
            )
            complex_recombination = (
                float(np.imag(complex_rate)) / complex_epsilon / dx_cell
            )
            for row in target_rows:
                if row is not None:
                    analytic[row, column] -= analytic_recombination
                    finite_difference[row, column] -= finite_recombination
                    complex_step[row, column] -= complex_recombination

        if is_projected:
            projection_voltage_n = (
                float(potential_voltage[node])
                - float(potential_voltage[electron_node])
            ) / thermal_voltage
            projection_voltage_p = (
                float(potential_voltage[hole_node])
                - float(potential_voltage[node])
            ) / thermal_voltage
            voltage_projection_step_n = projection_voltage_n * voltage_step
            voltage_projection_step_p = projection_voltage_p * voltage_step
            projection_margin = _IFACE_PROJ_EXP_CAP - max(
                abs(projection_log_n + voltage_projection_step_n),
                abs(projection_log_n - voltage_projection_step_n),
                abs(projection_log_p + voltage_projection_step_p),
                abs(projection_log_p - voltage_projection_step_p),
            )
            if not np.isfinite(projection_margin) or projection_margin <= 0.0:
                raise IonAwareAnalyticReactionCapabilityError(
                    "interface projection voltage stencil reaches the "
                    "exponent cap"
                )
            minimum_projection_cap_margin = min(
                minimum_projection_cap_margin,
                projection_margin,
            )
            projection_voltage_reference = (
                projection_voltage_n + projection_voltage_p
            )
            analytic_voltage_recombination = (
                rate_electron_derivative
                * projected_n
                * projection_voltage_n
                + rate_hole_derivative
                * projected_p
                * projection_voltage_p
                + rate_reference_derivative
                * projected_ni_sq
                * projection_voltage_reference
            ) / dx_cell
            rate_voltage_plus = projected_rate(
                n_value,
                p_value,
                projection_log_n + voltage_projection_step_n,
                projection_log_p + voltage_projection_step_p,
            )
            rate_voltage_minus = projected_rate(
                n_value,
                p_value,
                projection_log_n - voltage_projection_step_n,
                projection_log_p - voltage_projection_step_p,
            )
            if is_cross_node and not qss_branch:
                stencil_minimum = min(
                    float(rate_voltage_plus),
                    float(rate_voltage_minus),
                )
                if not np.isfinite(stencil_minimum) or stencil_minimum <= 0.0:
                    raise IonAwareAnalyticReactionCapabilityError(
                        "cross-node interface voltage stencil crosses the "
                        "inactive clamp branch"
                    )
                minimum_cross_node_clamp_margin = min(
                    minimum_cross_node_clamp_margin,
                    stencil_minimum,
                )
            finite_voltage_recombination = (
                float(rate_voltage_plus) - float(rate_voltage_minus)
            ) / (2.0 * voltage_step * dx_cell)
            complex_epsilon = 1.0e-30
            complex_voltage_rate = complex_projected_rate(
                complex(n_value),
                complex(p_value),
                complex(
                    projection_log_n
                    + 1j * projection_voltage_n * complex_epsilon
                ),
                complex(
                    projection_log_p
                    + 1j * projection_voltage_p * complex_epsilon
                ),
            )
            complex_voltage_recombination = (
                float(np.imag(complex_voltage_rate))
                / complex_epsilon
                / dx_cell
            )
            for row in target_rows:
                if row is not None:
                    voltage_derivative[row] -= analytic_voltage_recombination
                    finite_difference_voltage[row] -= (
                        finite_voltage_recombination
                    )
                    complex_step_voltage[row] -= (
                        complex_voltage_recombination
                    )

        is_two_sided = (
            two_sided_active
            and is_cross_node
            and not is_shared_occupancy
            and not qss_branch
        )
        if is_two_sided:
            mirror_n_value = float(n[hole_node])
            mirror_p_value = float(p[electron_node])
            if (
                not np.isfinite(mirror_n_value)
                or not np.isfinite(mirror_p_value)
                or mirror_n_value <= 0.0
                or mirror_p_value <= 0.0
            ):
                raise IonAwareAnalyticReactionCapabilityError(
                    "two-sided mirror density floors must be inactive"
                )
            mirror_ni_reference = float(
                material.interface_n_L_eq[index]
                * material.interface_p_R_eq[index]
            )
            mirror_projection_log_n = 0.0
            mirror_projection_log_p = 0.0
            mirror_projected = projection_active
            if mirror_projected:
                mirror_projection_log_n = (
                    float(potential[node]) - float(potential[hole_node])
                ) / thermal_voltage
                mirror_projection_log_p = (
                    float(potential[electron_node]) - float(potential[node])
                ) / thermal_voltage
                projection_margin = _IFACE_PROJ_EXP_CAP - max(
                    abs(mirror_projection_log_n),
                    abs(mirror_projection_log_p),
                )
                if (
                    not np.isfinite(projection_margin)
                    or projection_margin <= 0.0
                ):
                    raise IonAwareAnalyticReactionCapabilityError(
                        "two-sided mirror projection exponent cap is active "
                        "at the operating point"
                    )
                minimum_projection_cap_margin = min(
                    minimum_projection_cap_margin,
                    projection_margin,
                )

            mirror_factor_n = float(np.exp(mirror_projection_log_n))
            mirror_factor_p = float(np.exp(mirror_projection_log_p))
            mirror_projected_n = mirror_n_value * mirror_factor_n
            mirror_projected_p = mirror_p_value * mirror_factor_p
            mirror_projected_ni_sq = (
                mirror_ni_reference * mirror_factor_n * mirror_factor_p
            )
            mirror_scalars = (
                mirror_projected_n,
                mirror_projected_p,
                mirror_projected_ni_sq,
            )
            if (
                not np.all(np.isfinite(mirror_scalars))
                or mirror_projected_n <= 0.0
                or mirror_projected_p <= 0.0
                or mirror_projected_ni_sq < 0.0
            ):
                raise IonAwareAnalyticReactionCapabilityError(
                    "two-sided mirror inputs must be finite and physically admissible"
                )
            if v_n > 0.0 and v_p > 0.0:
                mirror_denominator = interface_srh_denominator(
                    mirror_projected_n,
                    mirror_projected_p,
                    n1,
                    p1,
                    v_n,
                    v_p,
                )
                if (
                    not np.isfinite(mirror_denominator)
                    or mirror_denominator <= 0.0
                ):
                    raise IonAwareAnalyticReactionCapabilityError(
                        "two-sided mirror SRH denominator must be finite and positive"
                    )
            else:
                mirror_denominator = np.inf
            mirror_derivatives = interface_recombination_derivatives(
                mirror_projected_n,
                mirror_projected_p,
                mirror_projected_ni_sq,
                n1,
                p1,
                v_n,
                v_p,
            )
            mirror_values = (
                mirror_derivatives.rate,
                mirror_derivatives.electron_density_derivative,
                mirror_derivatives.hole_density_derivative,
            )
            if any(
                not np.all(np.isfinite(value)) for value in mirror_values
            ):
                raise IonAwareAnalyticReactionCapabilityError(
                    "analytic two-sided mirror formula produced a non-finite block"
                )
            if float(mirror_derivatives.rate) <= 0.0:
                raise IonAwareAnalyticReactionCapabilityError(
                    "two-sided mirror clamp must be inactive at the operating "
                    "point (raw R_B > 0)"
                )
            two_sided_indices.append(index)
            minimum_two_sided_clamp_margin = min(
                minimum_two_sided_clamp_margin,
                float(mirror_derivatives.rate),
            )
            minimum_two_sided_density_floor_margin = min(
                minimum_two_sided_density_floor_margin,
                mirror_n_value,
                mirror_p_value,
            )
            mirror_surface_rate[index] = float(mirror_derivatives.rate)
            mirror_electron_derivative[index] = float(
                mirror_derivatives.electron_density_derivative
            ) * mirror_factor_n
            mirror_hole_derivative[index] = float(
                mirror_derivatives.hole_density_derivative
            ) * mirror_factor_p

            def mirror_rate(
                electron_density: complex | float,
                hole_density: complex | float,
                electron_log_factor: complex | float,
                hole_log_factor: complex | float,
            ) -> complex | float:
                factor_n = np.exp(electron_log_factor)
                factor_p = np.exp(hole_log_factor)
                return interface_recombination(
                    electron_density * factor_n,
                    hole_density * factor_p,
                    mirror_ni_reference * factor_n * factor_p,
                    n1,
                    p1,
                    v_n,
                    v_p,
                )

            for column, state_index in enumerate(layout.state_indices):
                step = float(steps[column])
                direct_n_component = (
                    mirror_n_value if state_index == hole_node else 0.0
                )
                direct_p_component = (
                    mirror_p_value
                    if state_index == grid.size + electron_node
                    else 0.0
                )
                direct_n_increment = direct_n_component * step
                direct_p_increment = direct_p_component * step
                if direct_n_component > 0.0 or direct_p_component > 0.0:
                    positive_lower_stencils = tuple(
                        value
                        for value in (
                            direct_n_component * float(np.exp(-step)),
                            direct_p_component * float(np.exp(-step)),
                        )
                        if value > 0.0
                    )
                    minimum_two_sided_density_floor_margin = min(
                        minimum_two_sided_density_floor_margin,
                        *positive_lower_stencils,
                    )
                mirror_projection_delta_n = 0.0
                mirror_projection_delta_p = 0.0
                if mirror_projected:
                    potential_increment = potential_jacobian[:, column] * step
                    mirror_projection_delta_n = (
                        float(potential_increment[node])
                        - float(potential_increment[hole_node])
                    ) / thermal_voltage
                    mirror_projection_delta_p = (
                        float(potential_increment[electron_node])
                        - float(potential_increment[node])
                    ) / thermal_voltage
                mirror_reference_log_increment = (
                    mirror_projection_delta_n + mirror_projection_delta_p
                )
                if (
                    direct_n_increment == 0.0
                    and direct_p_increment == 0.0
                    and mirror_projection_delta_n == 0.0
                    and mirror_projection_delta_p == 0.0
                ):
                    continue
                if mirror_projected:
                    projection_margin = _IFACE_PROJ_EXP_CAP - max(
                        abs(
                            mirror_projection_log_n
                            + mirror_projection_delta_n
                        ),
                        abs(
                            mirror_projection_log_n
                            - mirror_projection_delta_n
                        ),
                        abs(
                            mirror_projection_log_p
                            + mirror_projection_delta_p
                        ),
                        abs(
                            mirror_projection_log_p
                            - mirror_projection_delta_p
                        ),
                    )
                    if (
                        not np.isfinite(projection_margin)
                        or projection_margin <= 0.0
                    ):
                        raise IonAwareAnalyticReactionCapabilityError(
                            "two-sided mirror projection state stencil "
                            "reaches the exponent cap"
                        )
                    minimum_projection_cap_margin = min(
                        minimum_projection_cap_margin,
                        projection_margin,
                    )
                mirror_projected_n_increment = mirror_factor_n * (
                    direct_n_increment
                    + mirror_n_value * mirror_projection_delta_n
                )
                mirror_projected_p_increment = mirror_factor_p * (
                    direct_p_increment
                    + mirror_p_value * mirror_projection_delta_p
                )
                analytic_mirror = (
                    float(mirror_derivatives.electron_density_derivative)
                    * mirror_projected_n_increment
                    + float(mirror_derivatives.hole_density_derivative)
                    * mirror_projected_p_increment
                    - mirror_projected_ni_sq
                    * mirror_reference_log_increment
                    / mirror_denominator
                ) / dx_cell
                mirror_n_plus = mirror_n_value + direct_n_component * float(
                    np.expm1(step)
                )
                mirror_n_minus = mirror_n_value + direct_n_component * float(
                    np.expm1(-step)
                )
                mirror_p_plus = mirror_p_value + direct_p_component * float(
                    np.expm1(step)
                )
                mirror_p_minus = mirror_p_value + direct_p_component * float(
                    np.expm1(-step)
                )
                mirror_rate_plus = mirror_rate(
                    mirror_n_plus,
                    mirror_p_plus,
                    mirror_projection_log_n + mirror_projection_delta_n,
                    mirror_projection_log_p + mirror_projection_delta_p,
                )
                mirror_rate_minus = mirror_rate(
                    mirror_n_minus,
                    mirror_p_minus,
                    mirror_projection_log_n - mirror_projection_delta_n,
                    mirror_projection_log_p - mirror_projection_delta_p,
                )
                mirror_stencil_minimum = min(
                    float(mirror_rate_plus),
                    float(mirror_rate_minus),
                )
                if (
                    not np.isfinite(mirror_stencil_minimum)
                    or mirror_stencil_minimum <= 0.0
                ):
                    raise IonAwareAnalyticReactionCapabilityError(
                        "two-sided mirror central stencil crosses the "
                        "inactive clamp branch"
                    )
                minimum_two_sided_clamp_margin = min(
                    minimum_two_sided_clamp_margin,
                    mirror_stencil_minimum,
                )
                finite_mirror = 0.5 * (
                    mirror_rate_plus - mirror_rate_minus
                ) / dx_cell
                complex_epsilon = 1.0e-30
                complex_mirror_rate = mirror_rate(
                    mirror_n_value
                    + direct_n_component
                    * np.expm1(1j * step * complex_epsilon),
                    mirror_p_value
                    + direct_p_component
                    * np.expm1(1j * step * complex_epsilon),
                    mirror_projection_log_n
                    + 1j * mirror_projection_delta_n * complex_epsilon,
                    mirror_projection_log_p
                    + 1j * mirror_projection_delta_p * complex_epsilon,
                )
                complex_mirror = (
                    float(np.imag(complex_mirror_rate))
                    / complex_epsilon
                    / dx_cell
                )
                for row in target_rows:
                    if row is not None:
                        analytic[row, column] -= analytic_mirror
                        finite_difference[row, column] -= finite_mirror
                        complex_step[row, column] -= complex_mirror

            if mirror_projected:
                mirror_projection_voltage_n = (
                    float(potential_voltage[node])
                    - float(potential_voltage[hole_node])
                ) / thermal_voltage
                mirror_projection_voltage_p = (
                    float(potential_voltage[electron_node])
                    - float(potential_voltage[node])
                ) / thermal_voltage
                mirror_voltage_step_n = (
                    mirror_projection_voltage_n * voltage_step
                )
                mirror_voltage_step_p = (
                    mirror_projection_voltage_p * voltage_step
                )
                projection_margin = _IFACE_PROJ_EXP_CAP - max(
                    abs(mirror_projection_log_n + mirror_voltage_step_n),
                    abs(mirror_projection_log_n - mirror_voltage_step_n),
                    abs(mirror_projection_log_p + mirror_voltage_step_p),
                    abs(mirror_projection_log_p - mirror_voltage_step_p),
                )
                if (
                    not np.isfinite(projection_margin)
                    or projection_margin <= 0.0
                ):
                    raise IonAwareAnalyticReactionCapabilityError(
                        "two-sided mirror projection voltage stencil reaches "
                        "the exponent cap"
                    )
                minimum_projection_cap_margin = min(
                    minimum_projection_cap_margin,
                    projection_margin,
                )
                mirror_voltage_reference = (
                    mirror_projection_voltage_n
                    + mirror_projection_voltage_p
                )
                analytic_voltage_mirror = (
                    float(mirror_derivatives.electron_density_derivative)
                    * mirror_projected_n
                    * mirror_projection_voltage_n
                    + float(mirror_derivatives.hole_density_derivative)
                    * mirror_projected_p
                    * mirror_projection_voltage_p
                    - mirror_projected_ni_sq
                    * mirror_voltage_reference
                    / mirror_denominator
                ) / dx_cell
                mirror_voltage_plus = mirror_rate(
                    mirror_n_value,
                    mirror_p_value,
                    mirror_projection_log_n + mirror_voltage_step_n,
                    mirror_projection_log_p + mirror_voltage_step_p,
                )
                mirror_voltage_minus = mirror_rate(
                    mirror_n_value,
                    mirror_p_value,
                    mirror_projection_log_n - mirror_voltage_step_n,
                    mirror_projection_log_p - mirror_voltage_step_p,
                )
                mirror_voltage_minimum = min(
                    float(mirror_voltage_plus),
                    float(mirror_voltage_minus),
                )
                if (
                    not np.isfinite(mirror_voltage_minimum)
                    or mirror_voltage_minimum <= 0.0
                ):
                    raise IonAwareAnalyticReactionCapabilityError(
                        "two-sided mirror voltage stencil crosses the "
                        "inactive clamp branch"
                    )
                minimum_two_sided_clamp_margin = min(
                    minimum_two_sided_clamp_margin,
                    mirror_voltage_minimum,
                )
                finite_voltage_mirror = (
                    float(mirror_voltage_plus)
                    - float(mirror_voltage_minus)
                ) / (2.0 * voltage_step * dx_cell)
                complex_epsilon = 1.0e-30
                complex_voltage_mirror_rate = mirror_rate(
                    mirror_n_value,
                    mirror_p_value,
                    mirror_projection_log_n
                    + 1j
                    * mirror_projection_voltage_n
                    * complex_epsilon,
                    mirror_projection_log_p
                    + 1j
                    * mirror_projection_voltage_p
                    * complex_epsilon,
                )
                complex_voltage_mirror = (
                    float(np.imag(complex_voltage_mirror_rate))
                    / complex_epsilon
                    / dx_cell
                )
                for row in target_rows:
                    if row is not None:
                        voltage_derivative[row] -= analytic_voltage_mirror
                        finite_difference_voltage[row] -= finite_voltage_mirror
                        complex_step_voltage[row] -= complex_voltage_mirror

    arrays = (
        surface_rate,
        electron_derivative,
        hole_derivative,
        mirror_surface_rate,
        mirror_electron_derivative,
        mirror_hole_derivative,
        analytic,
        finite_difference,
        complex_step,
        voltage_derivative,
        finite_difference_voltage,
        complex_step_voltage,
    )
    if any(not np.all(np.isfinite(value)) for value in arrays):
        raise IonAwareAnalyticReactionCapabilityError(
            "analytic interface-reaction assembly produced a non-finite operator"
        )
    return IonAwareAnalyticInterfaceReactionLinearization(
        interface_nodes=nodes,
        electron_evaluation_nodes=eval_n,
        hole_evaluation_nodes=eval_p,
        cross_node_interface_indices=tuple(cross_node_indices),
        projected_interface_indices=tuple(projected_indices),
        shared_occupancy_interface_indices=tuple(shared_occupancy_indices),
        two_sided_interface_indices=tuple(two_sided_indices),
        qss_interface_indices=tuple(qss_indices),
        minimum_cross_node_clamp_margin_m2_s=(
            float(minimum_cross_node_clamp_margin)
            if clamp_certified_indices
            else None
        ),
        minimum_projection_exponent_cap_margin=(
            float(minimum_projection_cap_margin)
            if projected_indices
            else None
        ),
        minimum_shared_density_floor_margin_m3=(
            float(minimum_shared_density_floor_margin)
            if shared_occupancy_indices
            else None
        ),
        minimum_two_sided_clamp_margin_m2_s=(
            float(minimum_two_sided_clamp_margin)
            if two_sided_indices
            else None
        ),
        minimum_two_sided_density_floor_margin_m3=(
            float(minimum_two_sided_density_floor_margin)
            if two_sided_indices
            else None
        ),
        qss_transport_velocity_m_s=(
            float(qss_transport_velocity_m_s) if qss_indices else None
        ),
        minimum_qss_supply_rate_margin_m2_s=(
            min(item.supply_rate_m2_s for item in qss_evaluations)
            if qss_evaluations
            else None
        ),
        minimum_qss_root_headroom_m3=(
            min(item.root_headroom_m3 for item in qss_evaluations)
            if qss_evaluations
            else None
        ),
        maximum_qss_root_relative_residual=(
            max(item.relative_root_residual for item in qss_evaluations)
            if qss_evaluations
            else None
        ),
        surface_recombination_rate_m2_s=surface_rate,
        electron_density_derivative_m_s=electron_derivative,
        hole_density_derivative_m_s=hole_derivative,
        mirror_surface_recombination_rate_m2_s=mirror_surface_rate,
        mirror_electron_density_derivative_m_s=mirror_electron_derivative,
        mirror_hole_density_derivative_m_s=mirror_hole_derivative,
        rate_jacobian=analytic,
        finite_difference_rate_jacobian=finite_difference,
        complex_step_rate_jacobian=complex_step,
        rate_voltage_derivative=voltage_derivative,
        finite_difference_rate_voltage_derivative=(
            finite_difference_voltage
        ),
        complex_step_rate_voltage_derivative=complex_step_voltage,
    )


def build_ion_aware_analytic_contact_linearization(
    x: np.ndarray,
    stack: DeviceStack,
    base_state: np.ndarray,
    V_dc: float,
    material: MaterialArrays,
    layout: IonAwareStateCoordinateLayout,
    *,
    potential_at_operating_point_V: np.ndarray,
    state_steps: np.ndarray,
) -> IonAwareAnalyticContactLinearization:
    """Assemble finite-rate outer-contact tangents in scaled log coordinates.

    The cached contact reservoirs and surface velocities are independent of the
    applied voltage.  For every active carrier/side channel, the production
    Robin flux is linear in its boundary density.  Its contribution to the
    corresponding continuity row therefore has the same local derivative on
    both sides and for both carrier signs: ``-S / dx_cell``.

    The implementation keeps the full sign chain instead of inserting that
    simplified result directly.  The independent central stencil calls the
    production :func:`selective_contact_flux`, so a side or carrier sign error
    cannot be hidden by using the analytic formula twice.
    """

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
        or material.dx_cell.shape != grid.shape
        or not np.all(np.isfinite(material.dx_cell))
        or np.any(material.dx_cell <= 0.0)
    ):
        raise IonAwareAnalyticReactionCapabilityError(
            "analytic contact inputs must be finite and shape matched"
        )

    n, p, _phi, _state_vector = _state_fields(
        grid,
        state,
        stack,
        V_dc,
        material,
        phi_frozen=potential,
    )
    analytic = np.zeros((layout.size, layout.size), dtype=float)
    finite_difference = np.zeros_like(analytic)
    voltage_derivative = np.zeros(layout.size, dtype=float)
    coordinate_by_state_index = {
        state_index: column
        for column, state_index in enumerate(layout.state_indices)
    }
    row_by_state_index = {
        state_index: row
        for row, state_index in enumerate(layout.state_indices)
    }

    channel_specs = (
        (
            "electron_left",
            "n",
            "left",
            0,
            0,
            material.S_n_L,
            float(n[0]),
            float(material.n_L),
        ),
        (
            "electron_right",
            "n",
            "right",
            grid.size - 1,
            grid.size - 1,
            material.S_n_R,
            float(n[-1]),
            float(material.n_R),
        ),
        (
            "hole_left",
            "p",
            "left",
            0,
            grid.size,
            material.S_p_L,
            float(p[0]),
            float(material.p_L),
        ),
        (
            "hole_right",
            "p",
            "right",
            grid.size - 1,
            2 * grid.size - 1,
            material.S_p_R,
            float(p[-1]),
            float(material.p_R),
        ),
    )
    active_channels: list[str] = []
    boundary_state_indices: list[int] = []
    relaxation_rates: list[float] = []
    operating_rates: list[float] = []
    for (
        name,
        carrier,
        side,
        node,
        state_index,
        surface_velocity,
        density,
        density_eq,
    ) in channel_specs:
        if not material.has_selective_contacts or surface_velocity is None:
            continue
        surface_velocity = float(surface_velocity)
        scalars = (surface_velocity, density, density_eq)
        if (
            not np.all(np.isfinite(scalars))
            or surface_velocity < 0.0
            or density <= 0.0
            or density_eq <= 0.0
        ):
            raise IonAwareAnalyticReactionCapabilityError(
                "selective-contact inputs must be finite and physically admissible"
            )
        column = coordinate_by_state_index.get(state_index)
        row = row_by_state_index.get(state_index)
        if column is None or row is None:
            raise IonAwareAnalyticReactionCapabilityError(
                "active selective-contact boundary is absent from the state layout"
            )
        step = float(steps[column])
        width = float(material.dx_cell[node])
        flux_sign = {
            ("n", "left"): 1.0,
            ("n", "right"): -1.0,
            ("p", "left"): -1.0,
            ("p", "right"): 1.0,
        }[(carrier, side)]
        continuity_sign = {
            ("n", "left"): -1.0,
            ("n", "right"): 1.0,
            ("p", "left"): 1.0,
            ("p", "right"): -1.0,
        }[(carrier, side)]
        rate_from_flux = continuity_sign / (Q * width)
        flux_density_derivative = flux_sign * Q * surface_velocity
        analytic[row, column] = (
            rate_from_flux * flux_density_derivative * density * step
        )

        flux = selective_contact_flux(
            density,
            density_eq,
            surface_velocity,
            carrier=carrier,
            side=side,
        )
        flux_plus = selective_contact_flux(
            density * float(np.exp(step)),
            density_eq,
            surface_velocity,
            carrier=carrier,
            side=side,
        )
        flux_minus = selective_contact_flux(
            density * float(np.exp(-step)),
            density_eq,
            surface_velocity,
            carrier=carrier,
            side=side,
        )
        finite_difference[row, column] = 0.5 * rate_from_flux * (
            float(flux_plus) - float(flux_minus)
        )
        active_channels.append(name)
        boundary_state_indices.append(state_index)
        relaxation_rates.append(surface_velocity / width)
        operating_rates.append(rate_from_flux * float(flux))

    arrays = (
        analytic,
        finite_difference,
        voltage_derivative,
        np.asarray(relaxation_rates, dtype=float),
        np.asarray(operating_rates, dtype=float),
    )
    if any(not np.all(np.isfinite(value)) for value in arrays):
        raise IonAwareAnalyticReactionCapabilityError(
            "analytic contact assembly produced a non-finite operator"
        )
    return IonAwareAnalyticContactLinearization(
        active_channels=tuple(active_channels),
        boundary_state_indices=tuple(boundary_state_indices),
        relaxation_rate_s1=np.asarray(relaxation_rates, dtype=float),
        rate_at_operating_point_m3_s=np.asarray(operating_rates, dtype=float),
        rate_jacobian=analytic,
        finite_difference_rate_jacobian=finite_difference,
        rate_voltage_derivative=voltage_derivative,
        finite_difference_rate_voltage_derivative=np.zeros_like(
            voltage_derivative
        ),
    )


def _apply_analytic_reaction_linearization(
    reference: FrequencyDomainResult,
    analytic: (
        IonAwareAnalyticBulkReactionLinearization
        | IonAwareAnalyticInterfaceReactionLinearization
        | IonAwareAnalyticContactLinearization
    ),
    layout: IonAwareStateCoordinateLayout,
    *,
    block_name: str,
    V_dc: float,
    face_weights: np.ndarray,
    progress: ProgressCallback | None = None,
) -> FrequencyDomainResult:
    """Replace one independently validated reaction contribution."""

    state_shape = (layout.size, layout.size)
    weights = np.asarray(face_weights, dtype=float)
    face_count = reference.admittance_faces.shape[1]
    reaction_arrays = (
        analytic.rate_jacobian,
        analytic.finite_difference_rate_jacobian,
        analytic.rate_voltage_derivative,
        analytic.finite_difference_rate_voltage_derivative,
    )
    if (
        reference.rate_jacobian.shape != state_shape
        or analytic.rate_jacobian.shape != state_shape
        or analytic.finite_difference_rate_jacobian.shape != state_shape
        or analytic.rate_voltage_derivative.shape != (layout.size,)
        or analytic.finite_difference_rate_voltage_derivative.shape
        != (layout.size,)
        or weights.shape != (face_count,)
        or not np.all(np.isfinite(weights))
        or np.any(weights < 0.0)
        or float(np.sum(weights)) <= 0.0
        or not np.isfinite(V_dc)
        or any(not np.all(np.isfinite(value)) for value in reaction_arrays)
    ):
        raise IonAwareAnalyticReactionCapabilityError(
            f"{block_name} correction inputs must be finite and shape matched"
        )
    rate_jacobian = (
        reference.rate_jacobian
        - analytic.finite_difference_rate_jacobian
        + analytic.rate_jacobian
    )
    rate_voltage = (
        reference.rate_voltage_derivative
        - analytic.finite_difference_rate_voltage_derivative
        + analytic.rate_voltage_derivative
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


def apply_analytic_bulk_reaction_linearization(
    reference: FrequencyDomainResult,
    analytic: IonAwareAnalyticBulkReactionLinearization,
    layout: IonAwareStateCoordinateLayout,
    *,
    V_dc: float,
    face_weights: np.ndarray,
    progress: ProgressCallback | None = None,
) -> FrequencyDomainResult:
    """Replace only the bulk-recombination contribution in the rate block."""

    return _apply_analytic_reaction_linearization(
        reference,
        analytic,
        layout,
        block_name="bulk-reaction",
        V_dc=V_dc,
        face_weights=face_weights,
        progress=progress,
    )


def apply_analytic_interface_reaction_linearization(
    reference: FrequencyDomainResult,
    analytic: IonAwareAnalyticInterfaceReactionLinearization,
    layout: IonAwareStateCoordinateLayout,
    *,
    V_dc: float,
    face_weights: np.ndarray,
    progress: ProgressCallback | None = None,
) -> FrequencyDomainResult:
    """Replace only the certified local interface-SRH rate contribution."""

    return _apply_analytic_reaction_linearization(
        reference,
        analytic,
        layout,
        block_name="interface-reaction",
        V_dc=V_dc,
        face_weights=face_weights,
        progress=progress,
    )


def apply_analytic_contact_linearization(
    reference: FrequencyDomainResult,
    analytic: IonAwareAnalyticContactLinearization,
    layout: IonAwareStateCoordinateLayout,
    *,
    V_dc: float,
    face_weights: np.ndarray,
    progress: ProgressCallback | None = None,
) -> FrequencyDomainResult:
    """Replace only the independently validated selective-contact rate block."""

    return _apply_analytic_reaction_linearization(
        reference,
        analytic,
        layout,
        block_name="selective-contact",
        V_dc=V_dc,
        face_weights=face_weights,
        progress=progress,
    )


__all__ = [
    "IonAwareAnalyticBulkReactionLinearization",
    "IonAwareAnalyticContactLinearization",
    "IonAwareAnalyticInterfaceReactionLinearization",
    "IonAwareAnalyticReactionCapabilityError",
    "apply_analytic_bulk_reaction_linearization",
    "apply_analytic_contact_linearization",
    "apply_analytic_interface_reaction_linearization",
    "build_ion_aware_analytic_bulk_reaction_linearization",
    "build_ion_aware_analytic_contact_linearization",
    "build_ion_aware_analytic_interface_reaction_linearization",
]
