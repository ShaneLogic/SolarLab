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
    bulk_srh_denominator,
    interface_recombination,
    interface_recombination_derivatives,
    interface_srh_denominator,
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


@dataclass(frozen=True, slots=True)
class IonAwareAnalyticInterfaceReactionLinearization:
    """Defect-free, single-node interface SRH rate derivatives."""

    interface_nodes: tuple[int, ...]
    surface_recombination_rate_m2_s: np.ndarray
    electron_density_derivative_m_s: np.ndarray
    hole_density_derivative_m_s: np.ndarray
    rate_jacobian: np.ndarray
    finite_difference_rate_jacobian: np.ndarray
    complex_step_rate_jacobian: np.ndarray
    rate_voltage_derivative: np.ndarray


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


def build_ion_aware_analytic_interface_reaction_linearization(
    x: np.ndarray,
    stack: DeviceStack,
    base_state: np.ndarray,
    V_dc: float,
    material: MaterialArrays,
    layout: IonAwareStateCoordinateLayout,
    *,
    potential_at_operating_point_V: np.ndarray,
    state_steps: np.ndarray,
) -> IonAwareAnalyticInterfaceReactionLinearization:
    """Assemble smooth, local interface-SRH tangents in log coordinates."""

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
            ("shared-occupancy interface closure", material.iface_shared_occ),
            ("interface-plane projection", material.iface_plane_projection),
            ("two-sided interface closure", material.iface_two_sided),
        )
        if active
    )
    if active_closures:
        raise IonAwareAnalyticReactionCapabilityError(
            f"{active_closures[0]} has no declared analytic tangent"
        )
    if os.environ.get("SOLARLAB_IFACE_QSS", "") == "1":
        raise IonAwareAnalyticReactionCapabilityError(
            "QSS interface-plane root solve has no declared analytic tangent"
        )

    interfaces = electrical_interfaces(stack)
    defects = electrical_interface_defects(stack)
    nodes = tuple(int(node) for node in material.interface_nodes)
    count = len(nodes)
    if len(interfaces) != count or len(defects) != count:
        raise IonAwareAnalyticReactionCapabilityError(
            "electrical interface topology and material nodes are not aligned"
        )
    if any(defect is not None for defect in defects):
        raise IonAwareAnalyticReactionCapabilityError(
            "declared interface defects require an unsupported interface tangent"
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
    if eval_n != nodes or eval_p != nodes:
        raise IonAwareAnalyticReactionCapabilityError(
            "cross-node interface sampling has no declared analytic tangent"
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
    surface_rate = np.zeros(count, dtype=float)
    electron_derivative = np.zeros(count, dtype=float)
    hole_derivative = np.zeros(count, dtype=float)
    coordinate_by_state_index = {
        state_index: column
        for column, state_index in enumerate(layout.state_indices)
    }
    row_by_state_index = {
        state_index: row
        for row, state_index in enumerate(layout.state_indices)
    }

    for index, (node, velocities) in enumerate(
        zip(nodes, interfaces, strict=True)
    ):
        dx_cell = float(material.dx_cell[node])
        n_value = float(n[node])
        p_value = float(p[node])
        n1 = float(material.interface_n1[index])
        p1 = float(material.interface_p1[index])
        v_n = float(velocities[0]) * calibration[index]
        v_p = float(velocities[1]) * calibration[index]
        scalars = (dx_cell, n_value, p_value, n1, p1, v_n, v_p)
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
                "local interface SRH inputs must be finite and physically admissible"
            )
        if v_n > 0.0 and v_p > 0.0:
            denominator = interface_srh_denominator(
                n_value,
                p_value,
                n1,
                p1,
                v_n,
                v_p,
            )
            if not np.isfinite(denominator) or denominator <= 0.0:
                raise IonAwareAnalyticReactionCapabilityError(
                    "interface SRH denominator must be finite and positive"
                )
        derivatives = interface_recombination_derivatives(
            n_value,
            p_value,
            ni_sq_eff[index],
            n1,
            p1,
            v_n,
            v_p,
        )
        local_values = (
            derivatives.rate,
            derivatives.electron_density_derivative,
            derivatives.hole_density_derivative,
        )
        if any(not np.all(np.isfinite(value)) for value in local_values):
            raise IonAwareAnalyticReactionCapabilityError(
                "analytic interface-reaction formula produced a non-finite block"
            )
        surface_rate[index] = float(derivatives.rate)
        electron_derivative[index] = float(
            derivatives.electron_density_derivative
        )
        hole_derivative[index] = float(derivatives.hole_density_derivative)
        target_rows = tuple(
            row_by_state_index.get(state_index)
            for state_index in (node, grid.size + node)
        )
        for species, state_index, density, density_derivative in (
            ("electron", node, n_value, electron_derivative[index]),
            ("hole", grid.size + node, p_value, hole_derivative[index]),
        ):
            column = coordinate_by_state_index.get(state_index)
            if column is None:
                continue
            step = float(steps[column])
            analytic_recombination = (
                density_derivative * density * step / dx_cell
            )
            n_plus = n_value
            n_minus = n_value
            p_plus = p_value
            p_minus = p_value
            if species == "electron":
                n_plus *= float(np.exp(step))
                n_minus *= float(np.exp(-step))
            else:
                p_plus *= float(np.exp(step))
                p_minus *= float(np.exp(-step))
            finite_recombination = 0.5 * (
                interface_recombination(
                    n_plus,
                    p_plus,
                    ni_sq_eff[index],
                    n1,
                    p1,
                    v_n,
                    v_p,
                )
                - interface_recombination(
                    n_minus,
                    p_minus,
                    ni_sq_eff[index],
                    n1,
                    p1,
                    v_n,
                    v_p,
                )
            ) / dx_cell
            complex_epsilon = 1.0e-30
            if species == "electron":
                complex_rate = interface_recombination(
                    n_value + 1j * n_value * complex_epsilon,
                    p_value,
                    ni_sq_eff[index],
                    n1,
                    p1,
                    v_n,
                    v_p,
                )
            else:
                complex_rate = interface_recombination(
                    n_value,
                    p_value + 1j * p_value * complex_epsilon,
                    ni_sq_eff[index],
                    n1,
                    p1,
                    v_n,
                    v_p,
                )
            complex_recombination = (
                float(np.imag(complex_rate))
                / complex_epsilon
                * step
                / dx_cell
            )
            for row in target_rows:
                if row is not None:
                    analytic[row, column] -= analytic_recombination
                    finite_difference[row, column] -= finite_recombination
                    complex_step[row, column] -= complex_recombination

    arrays = (
        surface_rate,
        electron_derivative,
        hole_derivative,
        analytic,
        finite_difference,
        complex_step,
        voltage_derivative,
    )
    if any(not np.all(np.isfinite(value)) for value in arrays):
        raise IonAwareAnalyticReactionCapabilityError(
            "analytic interface-reaction assembly produced a non-finite operator"
        )
    return IonAwareAnalyticInterfaceReactionLinearization(
        interface_nodes=nodes,
        surface_recombination_rate_m2_s=surface_rate,
        electron_density_derivative_m_s=electron_derivative,
        hole_density_derivative_m_s=hole_derivative,
        rate_jacobian=analytic,
        finite_difference_rate_jacobian=finite_difference,
        complex_step_rate_jacobian=complex_step,
        rate_voltage_derivative=voltage_derivative,
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
            f"{block_name} correction inputs must be finite and shape matched"
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
