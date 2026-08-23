"""Research DAE combining one positive ion with algebraic interface states.

This topology is deliberately separate from the already certified single-ion
and ion-free algebraic-interface slices.  It covers two electrical layers, one
physical interface, a positive mobile ion at every bulk node, four uncharged
Fermi-Richardson interface trace densities, blocking ion boundaries, and ohmic
carrier contacts.  Production transient routes remain unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np
from scipy.special import expit

from perovskite_sim.constants import Q
from perovskite_sim.models.device import (
    DeviceStack,
    electrical_interface_defects,
    electrical_interfaces,
    electrical_layers,
)
from perovskite_sim.physics.generation import dual_cell_integral
from perovskite_sim.physics.interface_plane import (
    FERMI_RICHARDSON,
    InterfaceStateQSSResult,
    compute_interface_srh_occupancy_on_state,
    compute_interface_te_fluxes_live,
    solve_interface_states_live_qss,
)
from perovskite_sim.physics.poisson import solve_poisson_prefactored
from perovskite_sim.solver.dae import DAECapabilityError
from perovskite_sim.solver.dae_interface_states import (
    prepare_algebraic_interface_material,
)
from perovskite_sim.solver.mol import (
    MaterialArrays,
    StateVec,
    assemble_rhs,
    poisson_right_boundary,
)


def _readonly_f64(value: object, *, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite with shape {shape}")
    result = np.array(array, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _state_sha256(label: str, *arrays: np.ndarray) -> str:
    digest = hashlib.sha256(label.encode("ascii"))
    for value in arrays:
        array = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _bounded_density(
    reference_m3: np.ndarray,
    capacity_m3: np.ndarray,
    logit_reference: np.ndarray,
    coordinate: np.ndarray,
    *,
    label: str,
) -> np.ndarray:
    """Map a shifted logit to density without losing the zero increment."""
    occupation = expit(logit_reference + coordinate)
    reference_occupation = reference_m3 / capacity_m3
    positive_delta = (
        occupation
        * (1.0 - reference_occupation)
        * (-np.expm1(-np.maximum(coordinate, 0.0)))
    )
    negative_delta = (
        -reference_occupation
        * (1.0 - occupation)
        * (-np.expm1(np.minimum(coordinate, 0.0)))
    )
    density = reference_m3 + capacity_m3 * np.where(
        coordinate >= 0.0,
        positive_delta,
        negative_delta,
    )
    if np.any(occupation <= 0.0) or np.any(occupation >= 1.0):
        raise ValueError(f"{label} logit coordinate saturated")
    if not np.all(np.isfinite(density)):
        raise ValueError(f"{label} coordinate mapping is non-finite")
    return density


def _logit_coordinates(
    state_m3: np.ndarray,
    capacity_m3: np.ndarray,
    logit_reference: np.ndarray,
    *,
    label: str,
) -> np.ndarray:
    state = np.asarray(state_m3, dtype=float)
    if state.shape != capacity_m3.shape or not np.all(np.isfinite(state)):
        raise ValueError(f"{label} state must be finite and layout-sized")
    occupation = state / capacity_m3
    if np.any(occupation <= 0.0) or np.any(occupation >= 1.0):
        raise ValueError(f"{label} state must lie strictly inside capacity bounds")
    return np.log(occupation) - np.log1p(-occupation) - logit_reference


def _interface_capacity_m3(material: MaterialArrays) -> np.ndarray:
    values: list[float] = []
    for interface_node in material.interface_nodes:
        right = int(interface_node)
        left = right - 1
        values.extend(
            (
                float(material.N_C_physical[right]),
                float(material.N_V_physical[right]),
                float(material.N_C_physical[left]),
                float(material.N_V_physical[left]),
            )
        )
    return np.asarray(values, dtype=float)


@dataclass(frozen=True, slots=True)
class IonInterfaceDAELayout:
    """Coordinate layout for ``(log n, log p, ion logit, interface logits, phi)``."""

    node_count: int
    interface_count: int
    electron_reference_m3: np.ndarray
    hole_reference_m3: np.ndarray
    positive_ion_reference_m3: np.ndarray
    positive_ion_site_limit_m3: np.ndarray
    positive_ion_logit_reference: np.ndarray
    interface_reference_m3: np.ndarray
    interface_capacity_m3: np.ndarray
    interface_logit_reference: np.ndarray
    electron_rate_scale_m3_s: np.ndarray
    hole_rate_scale_m3_s: np.ndarray
    positive_ion_rate_scale_m3_s: np.ndarray
    interface_flux_scale_m2_s: np.ndarray
    poisson_scale_C_m2: np.ndarray
    potential_scale_V: float
    differential_mask: np.ndarray
    algebraic_mask: np.ndarray

    @property
    def interface_state_count(self) -> int:
        return 4 * self.interface_count

    @property
    def size(self) -> int:
        return 4 * self.node_count + self.interface_state_count

    @property
    def electron_slice(self) -> slice:
        return slice(0, self.node_count)

    @property
    def hole_slice(self) -> slice:
        return slice(self.node_count, 2 * self.node_count)

    @property
    def positive_ion_slice(self) -> slice:
        return slice(2 * self.node_count, 3 * self.node_count)

    @property
    def interface_slice(self) -> slice:
        start = 3 * self.node_count
        return slice(start, start + self.interface_state_count)

    @property
    def potential_slice(self) -> slice:
        start = 3 * self.node_count + self.interface_state_count
        return slice(start, start + self.node_count)


@dataclass(frozen=True, slots=True)
class IonInterfaceDAEResidualReport:
    """Separated carrier, ion, interface, Poisson, and inventory evidence."""

    normalized_residual: np.ndarray
    electron_rate_residual_m3_s: np.ndarray
    hole_rate_residual_m3_s: np.ndarray
    positive_ion_rate_residual_m3_s: np.ndarray
    interface_state_flux_residual_m2_s: np.ndarray
    interface_bulk_flux_m2_s: np.ndarray
    interface_cross_flux_m2_s: np.ndarray
    poisson_residual_C_m2: np.ndarray
    carrier_boundary_residual_log: np.ndarray
    potential_boundary_residual_V: np.ndarray
    positive_ion_inventory_residual_m2_s: float
    positive_ion_rhs_inventory_rate_m2_s: float
    max_normalized_carrier_residual: float
    max_normalized_positive_ion_residual: float
    max_normalized_interface_residual: float
    max_normalized_differential_residual: float
    max_normalized_algebraic_residual: float
    max_normalized_residual: float


@dataclass(frozen=True, slots=True)
class IonInterfaceDAEConsistentInitialCondition:
    """Reproducible state/derivative pair satisfying the combined DAE rows."""

    coordinate: np.ndarray
    derivative: np.ndarray
    physical_state: np.ndarray
    interface_state_m3: np.ndarray
    potential_V: np.ndarray
    report: IonInterfaceDAEResidualReport
    certified: bool
    state_sha256: str


@dataclass(frozen=True, slots=True)
class SingleIonAlgebraicInterfaceDAE:
    """Two-layer DAE retaining one positive ion and four interface traces."""

    grid_m: np.ndarray
    stack: DeviceStack
    material: MaterialArrays
    layout: IonInterfaceDAELayout
    V_app_V: float
    illuminated: bool
    interface_residual_tolerance: float

    def physical_fields(
        self,
        coordinate: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        value = np.asarray(coordinate, dtype=float)
        if value.shape != (self.layout.size,) or not np.all(np.isfinite(value)):
            raise ValueError(
                "ion-interface DAE coordinate must be finite and layout-sized"
            )
        layout = self.layout
        with np.errstate(over="ignore", invalid="ignore"):
            n = layout.electron_reference_m3 * np.exp(value[layout.electron_slice])
            p = layout.hole_reference_m3 * np.exp(value[layout.hole_slice])
        positive_ion = _bounded_density(
            layout.positive_ion_reference_m3,
            layout.positive_ion_site_limit_m3,
            layout.positive_ion_logit_reference,
            value[layout.positive_ion_slice],
            label="positive-ion",
        )
        interface_state = _bounded_density(
            layout.interface_reference_m3,
            layout.interface_capacity_m3,
            layout.interface_logit_reference,
            value[layout.interface_slice],
            label="interface-state",
        )
        phi = np.asarray(value[layout.potential_slice], dtype=float)
        if not np.all(np.isfinite(n)) or not np.all(np.isfinite(p)):
            raise ValueError("ion-interface DAE log-density coordinate overflowed")
        return n, p, positive_ion, interface_state, phi

    def positive_ion_coordinate_derivative_m3(
        self,
        coordinate: np.ndarray,
    ) -> np.ndarray:
        _n, _p, positive_ion, _interface, _phi = self.physical_fields(coordinate)
        occupation = positive_ion / self.layout.positive_ion_site_limit_m3
        return positive_ion * (1.0 - occupation)

    def interface_coordinate_jacobian_m3(
        self,
        coordinate: np.ndarray,
    ) -> np.ndarray:
        _n, _p, _ion, interface_state, _phi = self.physical_fields(coordinate)
        occupation = interface_state / self.layout.interface_capacity_m3
        return interface_state * (1.0 - occupation)

    def interface_coordinates_from_state(self, state_m3: np.ndarray) -> np.ndarray:
        return _logit_coordinates(
            state_m3,
            self.layout.interface_capacity_m3,
            self.layout.interface_logit_reference,
            label="interface-state",
        )

    def packed_physical_state(self, coordinate: np.ndarray) -> np.ndarray:
        n, p, positive_ion, _interface, _phi = self.physical_fields(coordinate)
        return StateVec.pack(n, p, positive_ion)

    def interface_response(
        self,
        n: np.ndarray,
        p: np.ndarray,
        interface_state: np.ndarray,
        phi: np.ndarray,
    ) -> InterfaceStateQSSResult:
        """Evaluate the explicit interface balances without nested elimination."""
        material = self.material
        bulk_flux = compute_interface_te_fluxes_live(
            material,
            interface_state,
            n,
            p,
            phi,
            v_th_eff=material.iface_state_v_th,
            v_cross_eff=0.0,
            V_app=self.V_app_V,
            interface_transport_model=material.iface_qss_transport_model,
        )
        thermionic_flux = compute_interface_te_fluxes_live(
            material,
            interface_state,
            n,
            p,
            phi,
            v_th_eff=material.iface_state_v_th,
            v_cross_eff=(
                material.iface_state_v_th * material.iface_qss_cross_transmission
            ),
            V_app=self.V_app_V,
            interface_transport_model=material.iface_qss_transport_model,
        )
        cross_flux = thermionic_flux - bulk_flux
        srh_flux = compute_interface_srh_occupancy_on_state(
            interface_state,
            self.stack,
            material,
        )
        state_flux = thermionic_flux + srh_flux
        normalized = float(
            np.max(
                np.abs(state_flux / self.layout.interface_flux_scale_m2_s),
                initial=0.0,
            )
        )
        return InterfaceStateQSSResult(
            state_m3=np.asarray(interface_state, dtype=float),
            bulk_flux_m2_s=np.asarray(bulk_flux, dtype=float),
            cross_flux_m2_s=np.asarray(cross_flux, dtype=float),
            state_flux_m2_s=np.asarray(state_flux, dtype=float),
            normalized_residual=normalized,
            evaluations=0,
            transport_model=material.iface_qss_transport_model,
        )

    def compatible_derivative(self, coordinate: np.ndarray) -> np.ndarray:
        """Return coordinate rates satisfying every differential row."""
        layout = self.layout
        n, p, _positive_ion, interface_state, phi = self.physical_fields(coordinate)
        response = self.interface_response(n, p, interface_state, phi)
        rhs = StateVec.unpack(
            assemble_rhs(
                0.0,
                self.packed_physical_state(coordinate),
                self.grid_m,
                self.stack,
                self.material,
                illuminated=self.illuminated,
                V_app=self.V_app_V,
                phi_frozen=phi,
                interface_qss_result=response,
            ),
            layout.node_count,
        )
        derivative = np.zeros(layout.size, dtype=float)
        derivative[1 : layout.node_count - 1] = rhs.n[1:-1] / n[1:-1]
        derivative[layout.node_count + 1 : 2 * layout.node_count - 1] = (
            rhs.p[1:-1] / p[1:-1]
        )
        derivative[layout.positive_ion_slice] = (
            rhs.P / self.positive_ion_coordinate_derivative_m3(coordinate)
        )
        if not np.all(np.isfinite(derivative)):
            raise ValueError("ion-interface compatible derivative is non-finite")
        return derivative

    def residual_report(
        self,
        coordinate: np.ndarray,
        derivative: np.ndarray,
    ) -> IonInterfaceDAEResidualReport:
        layout = self.layout
        rate = np.asarray(derivative, dtype=float)
        if rate.shape != (layout.size,) or not np.all(np.isfinite(rate)):
            raise ValueError(
                "ion-interface DAE derivative must be finite and layout-sized"
            )
        n, p, positive_ion, interface_state, phi = self.physical_fields(coordinate)
        response = self.interface_response(n, p, interface_state, phi)
        rhs = StateVec.unpack(
            assemble_rhs(
                0.0,
                StateVec.pack(n, p, positive_ion),
                self.grid_m,
                self.stack,
                self.material,
                illuminated=self.illuminated,
                V_app=self.V_app_V,
                phi_frozen=phi,
                interface_qss_result=response,
            ),
            layout.node_count,
        )

        electron_rate = n * rate[layout.electron_slice] - rhs.n
        hole_rate = p * rate[layout.hole_slice] - rhs.p
        positive_ion_rate = (
            self.positive_ion_coordinate_derivative_m3(coordinate)
            * rate[layout.positive_ion_slice]
            - rhs.P
        )
        normalized = np.zeros(layout.size, dtype=float)
        interior = slice(1, -1)
        normalized[1 : layout.node_count - 1] = (
            electron_rate[interior] / layout.electron_rate_scale_m3_s[interior]
        )
        normalized[layout.node_count + 1 : 2 * layout.node_count - 1] = (
            hole_rate[interior] / layout.hole_rate_scale_m3_s[interior]
        )
        normalized[layout.positive_ion_slice] = (
            positive_ion_rate / layout.positive_ion_rate_scale_m3_s
        )

        boundary_nodes = np.array([0, layout.node_count - 1], dtype=int)
        electron_boundary = np.log(
            n[boundary_nodes]
            / np.array([self.material.n_L, self.material.n_R], dtype=float)
        )
        hole_boundary = np.log(
            p[boundary_nodes]
            / np.array([self.material.p_L, self.material.p_R], dtype=float)
        )
        normalized[0] = electron_boundary[0]
        normalized[layout.node_count - 1] = electron_boundary[1]
        normalized[layout.node_count] = hole_boundary[0]
        normalized[2 * layout.node_count - 1] = hole_boundary[1]
        normalized[layout.interface_slice] = (
            response.state_flux_m2_s / layout.interface_flux_scale_m2_s
        )

        rho = Q * (
            p
            - n
            + positive_ion
            - self.material.P_ion0
            + self.material.N_D
            - self.material.N_A
        )
        capacitance = self.material.poisson_factor.C
        widths = self.material.poisson_factor.h_cell
        poisson = (
            capacitance[1:] * (phi[2:] - phi[1:-1])
            - capacitance[:-1] * (phi[1:-1] - phi[:-2])
            + rho[1:-1] * widths
        )
        potential_boundary = np.array(
            (
                phi[0],
                phi[-1] - poisson_right_boundary(self.material, self.V_app_V),
            ),
            dtype=float,
        )
        potential_rows = normalized[layout.potential_slice]
        potential_rows[0] = potential_boundary[0] / layout.potential_scale_V
        potential_rows[-1] = potential_boundary[1] / layout.potential_scale_V
        potential_rows[1:-1] = poisson / layout.poisson_scale_C_m2

        carrier_rows = np.concatenate(
            (
                normalized[1 : layout.node_count - 1],
                normalized[layout.node_count + 1 : 2 * layout.node_count - 1],
            )
        )
        differential = normalized[layout.differential_mask]
        algebraic = normalized[layout.algebraic_mask]
        return IonInterfaceDAEResidualReport(
            normalized_residual=_readonly_f64(
                normalized,
                shape=(layout.size,),
                name="ion-interface normalized residual",
            ),
            electron_rate_residual_m3_s=_readonly_f64(
                electron_rate[interior],
                shape=(layout.node_count - 2,),
                name="electron rate residual",
            ),
            hole_rate_residual_m3_s=_readonly_f64(
                hole_rate[interior],
                shape=(layout.node_count - 2,),
                name="hole rate residual",
            ),
            positive_ion_rate_residual_m3_s=_readonly_f64(
                positive_ion_rate,
                shape=(layout.node_count,),
                name="positive-ion rate residual",
            ),
            interface_state_flux_residual_m2_s=_readonly_f64(
                response.state_flux_m2_s,
                shape=(layout.interface_state_count,),
                name="interface-state flux residual",
            ),
            interface_bulk_flux_m2_s=_readonly_f64(
                response.bulk_flux_m2_s,
                shape=(layout.interface_state_count,),
                name="interface bulk flux",
            ),
            interface_cross_flux_m2_s=_readonly_f64(
                response.cross_flux_m2_s,
                shape=(layout.interface_state_count,),
                name="interface cross flux",
            ),
            poisson_residual_C_m2=_readonly_f64(
                poisson,
                shape=(layout.node_count - 2,),
                name="Poisson residual",
            ),
            carrier_boundary_residual_log=_readonly_f64(
                np.concatenate((electron_boundary, hole_boundary)),
                shape=(4,),
                name="carrier boundary residual",
            ),
            potential_boundary_residual_V=_readonly_f64(
                potential_boundary,
                shape=(2,),
                name="potential boundary residual",
            ),
            positive_ion_inventory_residual_m2_s=dual_cell_integral(
                self.grid_m,
                positive_ion_rate,
            ),
            positive_ion_rhs_inventory_rate_m2_s=dual_cell_integral(
                self.grid_m,
                rhs.P,
            ),
            max_normalized_carrier_residual=float(
                np.max(np.abs(carrier_rows), initial=0.0)
            ),
            max_normalized_positive_ion_residual=float(
                np.max(
                    np.abs(normalized[layout.positive_ion_slice]),
                    initial=0.0,
                )
            ),
            max_normalized_interface_residual=float(
                np.max(
                    np.abs(normalized[layout.interface_slice]),
                    initial=0.0,
                )
            ),
            max_normalized_differential_residual=float(
                np.max(np.abs(differential), initial=0.0)
            ),
            max_normalized_algebraic_residual=float(
                np.max(np.abs(algebraic), initial=0.0)
            ),
            max_normalized_residual=float(np.max(np.abs(normalized), initial=0.0)),
        )

    def residual(self, coordinate: np.ndarray, derivative: np.ndarray) -> np.ndarray:
        return self.residual_report(coordinate, derivative).normalized_residual

    def derivative_jacobian_diagonal(self, coordinate: np.ndarray) -> np.ndarray:
        layout = self.layout
        n, p, _ion, _interface, _phi = self.physical_fields(coordinate)
        result = np.zeros(layout.size, dtype=float)
        interior = np.arange(1, layout.node_count - 1)
        result[interior] = n[interior] / layout.electron_rate_scale_m3_s[interior]
        result[layout.node_count + interior] = (
            p[interior] / layout.hole_rate_scale_m3_s[interior]
        )
        result[layout.positive_ion_slice] = (
            self.positive_ion_coordinate_derivative_m3(coordinate)
            / layout.positive_ion_rate_scale_m3_s
        )
        return result

    def derivative_jacobian(self, coordinate: np.ndarray) -> np.ndarray:
        return np.diag(self.derivative_jacobian_diagonal(coordinate))

    def boundary_poisson_state_jacobian(self, coordinate: np.ndarray) -> np.ndarray:
        """Return exact carrier-boundary and ion-aware Poisson rows."""
        n, p, _ion, _interface, _phi = self.physical_fields(coordinate)
        ion_slope = self.positive_ion_coordinate_derivative_m3(coordinate)
        layout = self.layout
        count = layout.node_count
        result = np.zeros((layout.size, layout.size), dtype=float)
        for index in (0, count - 1, count, 2 * count - 1):
            result[index, index] = 1.0

        potential_offset = layout.potential_slice.start
        assert potential_offset is not None
        result[potential_offset, potential_offset] = 1.0 / layout.potential_scale_V
        result[-1, -1] = 1.0 / layout.potential_scale_V
        capacitance = self.material.poisson_factor.C
        widths = self.material.poisson_factor.h_cell
        for local, node in enumerate(range(1, count - 1)):
            row = potential_offset + node
            scale = layout.poisson_scale_C_m2[local]
            result[row, node] = -Q * n[node] * widths[local] / scale
            result[row, count + node] = Q * p[node] * widths[local] / scale
            result[row, 2 * count + node] = Q * ion_slope[node] * widths[local] / scale
            result[row, potential_offset + node - 1] = capacitance[node - 1] / scale
            result[row, potential_offset + node] = (
                -(capacitance[node - 1] + capacitance[node]) / scale
            )
            result[row, potential_offset + node + 1] = capacitance[node] / scale
        return result


def _validate_capability(
    stack: DeviceStack,
    material: MaterialArrays,
    packed_state: np.ndarray,
    node_count: int,
) -> StateVec:
    violations: list[str] = []
    if len(electrical_layers(stack)) != 2:
        violations.append("exactly two electrical layers are required")
    if len(material.interface_nodes) != 1:
        violations.append("exactly one physical interface is required")
    if len(electrical_interfaces(stack)) != 1:
        violations.append("exactly one interface SRH velocity pair is required")
    elif any(
        not np.isfinite(value) or value <= 0.0
        for value in electrical_interfaces(stack)[0]
    ):
        violations.append("both interface capture velocities must be positive")
    if any(defect is not None for defect in electrical_interface_defects(stack)):
        violations.append("InterfaceDefect is not supported")
    if stack.interface_charge_closure != "off" or material.iface_state_charge != 0.0:
        violations.append("interface charge closure must be off")
    if material.N_iface_state:
        violations.append("dynamic interface states are not supported")
    if not material.iface_qss_exclusive_transport:
        violations.append("explicit algebraic interface transport is required")
    if not material.iface_state_physical_offsets:
        violations.append("physical interface offsets are required")
    if material.iface_qss_two_sided_trace:
        violations.append("two-sided trace geometry is not supported")
    if material.iface_qss_transport_model != FERMI_RICHARDSON:
        violations.append("fermi_richardson interface transport is required")
    if material.has_selective_contacts:
        violations.append("selective contacts are not supported")
    if material.has_dual_ions:
        violations.append("dual mobile ions are not supported")
    if np.any(material.D_ion_node <= 0.0):
        violations.append("one positive mobile ion must be active at every node")
    if np.any(material.P_ion0 <= 0.0) or np.any(material.P_lim_node <= material.P_ion0):
        violations.append("positive-ion reference must lie inside site limits")
    if material.has_field_mobility:
        violations.append("field-dependent mobility is not supported")
    if material.has_radiative_reabsorption:
        violations.append("self-consistent photon recycling is not supported")
    if material.het_recomb_despike != 0.0:
        violations.append("heterojunction recombination de-spiking is not supported")
    if material.iface_plane_projection or material.iface_two_sided:
        violations.append("legacy interface projection modifiers are not supported")
    if material.interface_nodes and (
        tuple(material.interface_eval_node_n) != tuple(material.interface_nodes)
        or tuple(material.interface_eval_node_p) != tuple(material.interface_nodes)
    ):
        violations.append("cross-node carrier sampling is not supported")
    if violations:
        raise DAECapabilityError("; ".join(violations))

    state = np.asarray(packed_state, dtype=float)
    if state.shape != (3 * node_count,) or not np.all(np.isfinite(state)):
        raise ValueError("reference_state must be a finite single-ion-layout vector")
    unpacked = StateVec.unpack(state, node_count)
    if np.any(unpacked.n <= 0.0) or np.any(unpacked.p <= 0.0):
        raise ValueError("reference carrier densities must be strictly positive")
    if np.any(unpacked.P <= 0.0) or np.any(unpacked.P >= material.P_lim_node):
        raise ValueError("reference positive-ion density must be inside site limits")
    return unpacked


def build_single_ion_algebraic_interface_dae(
    grid_m: np.ndarray,
    stack: DeviceStack,
    reference_state: np.ndarray,
    *,
    V_app_V: float = 0.0,
    illuminated: bool = False,
    carrier_reference_time_s: float = 1.0e-7,
    ion_reference_time_s: float = 1.0,
    interface_velocity_m_s: float = 1.0e5,
    cross_transmission: float = 1.0,
    interface_residual_tolerance: float = 1.0e-9,
    material: MaterialArrays | None = None,
) -> SingleIonAlgebraicInterfaceDAE:
    """Build the first combined ion/interface DAE capability slice."""
    grid = np.asarray(grid_m, dtype=float)
    if (
        grid.ndim != 1
        or grid.size < 5
        or not np.all(np.isfinite(grid))
        or np.any(np.diff(grid) <= 0.0)
    ):
        raise ValueError("grid_m must be finite and strictly increasing")
    if not np.isfinite(V_app_V):
        raise ValueError("V_app_V must be finite")
    for name, value in (
        ("carrier_reference_time_s", carrier_reference_time_s),
        ("ion_reference_time_s", ion_reference_time_s),
        ("interface_residual_tolerance", interface_residual_tolerance),
    ):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    prepared = prepare_algebraic_interface_material(
        grid,
        stack,
        material=material,
        interface_velocity_m_s=interface_velocity_m_s,
        cross_transmission=cross_transmission,
    )
    if prepared.poisson_factor.N != grid.size:
        raise ValueError("material Poisson factor does not match the DAE grid")
    state = _validate_capability(stack, prepared, reference_state, grid.size)
    n = np.array(state.n, copy=True)
    p = np.array(state.p, copy=True)
    positive_ion = np.array(state.P, copy=True)
    n[[0, -1]] = (prepared.n_L, prepared.n_R)
    p[[0, -1]] = (prepared.p_L, prepared.p_R)

    rho = Q * (p - n + positive_ion - prepared.P_ion0 + prepared.N_D - prepared.N_A)
    phi = solve_poisson_prefactored(
        prepared.poisson_factor,
        rho,
        phi_left=0.0,
        phi_right=poisson_right_boundary(prepared, V_app_V),
    )
    qss = solve_interface_states_live_qss(
        prepared,
        stack,
        n,
        p,
        phi,
        V_app=V_app_V,
        v_th_eff=prepared.iface_state_v_th,
        cross_transmission=prepared.iface_qss_cross_transmission,
        interface_transport_model=prepared.iface_qss_transport_model,
        residual_tolerance=interface_residual_tolerance,
        fail_on_residual=True,
    )
    interface_reference = np.asarray(qss.state_m3, dtype=float)
    interface_capacity = _interface_capacity_m3(prepared)
    if (
        interface_reference.shape != interface_capacity.shape
        or not np.all(np.isfinite(interface_capacity))
        or np.any(interface_capacity <= 0.0)
        or np.any(interface_reference <= 0.0)
        or np.any(interface_reference >= interface_capacity)
    ):
        raise DAECapabilityError(
            "interface QSS reference must lie strictly inside physical DOS bounds"
        )
    interface_occupation = interface_reference / interface_capacity
    interface_logit_reference = np.log(interface_occupation) - np.log1p(
        -interface_occupation
    )
    ion_occupation = positive_ion / prepared.P_lim_node
    ion_logit_reference = np.log(ion_occupation) - np.log1p(-ion_occupation)

    potential_scale = max(float(prepared.V_T_device), 1.0e-3)
    charge_scale = (
        Q
        * (
            np.abs(n[1:-1])
            + np.abs(p[1:-1])
            + np.abs(prepared.N_D[1:-1])
            + np.abs(prepared.N_A[1:-1])
            + np.abs(prepared.P_ion0[1:-1])
        )
        * prepared.poisson_factor.h_cell
    )
    dielectric_scale = (
        prepared.poisson_factor.C[:-1] + prepared.poisson_factor.C[1:]
    ) * potential_scale
    poisson_scale = np.maximum.reduce(
        (
            charge_scale,
            dielectric_scale,
            np.full(grid.size - 2, np.finfo(float).tiny),
        )
    )
    interface_flux_scale = np.maximum.reduce(
        (
            prepared.iface_state_v_th * interface_reference,
            np.abs(qss.bulk_flux_m2_s),
            np.abs(qss.cross_flux_m2_s),
            np.ones_like(interface_reference),
        )
    )

    size = 4 * grid.size + interface_reference.size
    differential_mask = np.zeros(size, dtype=bool)
    differential_mask[1 : grid.size - 1] = True
    differential_mask[grid.size + 1 : 2 * grid.size - 1] = True
    differential_mask[2 * grid.size : 3 * grid.size] = True
    algebraic_mask = ~differential_mask
    for array in (differential_mask, algebraic_mask):
        array.setflags(write=False)
    layout = IonInterfaceDAELayout(
        node_count=grid.size,
        interface_count=len(prepared.interface_nodes),
        electron_reference_m3=_readonly_f64(
            n, shape=(grid.size,), name="electron reference"
        ),
        hole_reference_m3=_readonly_f64(p, shape=(grid.size,), name="hole reference"),
        positive_ion_reference_m3=_readonly_f64(
            positive_ion,
            shape=(grid.size,),
            name="positive-ion reference",
        ),
        positive_ion_site_limit_m3=_readonly_f64(
            prepared.P_lim_node,
            shape=(grid.size,),
            name="positive-ion site limit",
        ),
        positive_ion_logit_reference=_readonly_f64(
            ion_logit_reference,
            shape=(grid.size,),
            name="positive-ion logit reference",
        ),
        interface_reference_m3=_readonly_f64(
            interface_reference,
            shape=(interface_reference.size,),
            name="interface-state reference",
        ),
        interface_capacity_m3=_readonly_f64(
            interface_capacity,
            shape=(interface_capacity.size,),
            name="interface-state capacity",
        ),
        interface_logit_reference=_readonly_f64(
            interface_logit_reference,
            shape=(interface_reference.size,),
            name="interface-state logit reference",
        ),
        electron_rate_scale_m3_s=_readonly_f64(
            np.maximum(n / carrier_reference_time_s, 1.0),
            shape=(grid.size,),
            name="electron rate scale",
        ),
        hole_rate_scale_m3_s=_readonly_f64(
            np.maximum(p / carrier_reference_time_s, 1.0),
            shape=(grid.size,),
            name="hole rate scale",
        ),
        positive_ion_rate_scale_m3_s=_readonly_f64(
            np.maximum(positive_ion / ion_reference_time_s, 1.0),
            shape=(grid.size,),
            name="positive-ion rate scale",
        ),
        interface_flux_scale_m2_s=_readonly_f64(
            interface_flux_scale,
            shape=(interface_reference.size,),
            name="interface flux scale",
        ),
        poisson_scale_C_m2=_readonly_f64(
            poisson_scale,
            shape=(grid.size - 2,),
            name="Poisson scale",
        ),
        potential_scale_V=potential_scale,
        differential_mask=differential_mask,
        algebraic_mask=algebraic_mask,
    )
    return SingleIonAlgebraicInterfaceDAE(
        grid_m=_readonly_f64(grid, shape=(grid.size,), name="grid"),
        stack=stack,
        material=prepared,
        layout=layout,
        V_app_V=float(V_app_V),
        illuminated=bool(illuminated),
        interface_residual_tolerance=float(interface_residual_tolerance),
    )


def project_single_ion_algebraic_interface_state(
    model: SingleIonAlgebraicInterfaceDAE,
    coordinate: np.ndarray,
) -> np.ndarray:
    """Pin reservoirs, solve ion-aware Poisson, then eliminate trace states."""
    value = np.asarray(coordinate, dtype=float)
    if value.shape != (model.layout.size,) or not np.all(np.isfinite(value)):
        raise ValueError("coordinate must be finite and layout-sized")
    result = np.array(value, copy=True)
    layout = model.layout
    count = layout.node_count
    result[0] = 0.0
    result[count - 1] = 0.0
    result[count] = 0.0
    result[2 * count - 1] = 0.0
    n, p, positive_ion, _interface, _phi = model.physical_fields(result)
    rho = Q * (
        p
        - n
        + positive_ion
        - model.material.P_ion0
        + model.material.N_D
        - model.material.N_A
    )
    phi = solve_poisson_prefactored(
        model.material.poisson_factor,
        rho,
        phi_left=0.0,
        phi_right=poisson_right_boundary(model.material, model.V_app_V),
    )
    result[layout.potential_slice] = phi
    qss = solve_interface_states_live_qss(
        model.material,
        model.stack,
        n,
        p,
        phi,
        V_app=model.V_app_V,
        v_th_eff=model.material.iface_state_v_th,
        cross_transmission=model.material.iface_qss_cross_transmission,
        interface_transport_model=model.material.iface_qss_transport_model,
        residual_tolerance=model.interface_residual_tolerance,
        fail_on_residual=True,
    )
    result[layout.interface_slice] = model.interface_coordinates_from_state(
        qss.state_m3
    )
    return result


def compatible_derivative(
    model: SingleIonAlgebraicInterfaceDAE,
    coordinate: np.ndarray,
) -> np.ndarray:
    return model.compatible_derivative(coordinate)


def build_single_ion_algebraic_interface_consistent_initial_condition(
    model: SingleIonAlgebraicInterfaceDAE,
    *,
    residual_tolerance: float = 1.0e-9,
) -> IonInterfaceDAEConsistentInitialCondition:
    """Construct deterministic ion/interface/Poisson consistent initial data."""
    if not np.isfinite(residual_tolerance) or residual_tolerance <= 0.0:
        raise ValueError("residual_tolerance must be finite and positive")
    coordinate = project_single_ion_algebraic_interface_state(
        model,
        np.zeros(model.layout.size, dtype=float),
    )
    derivative = model.compatible_derivative(coordinate)
    n, p, positive_ion, interface_state, phi = model.physical_fields(coordinate)
    packed = StateVec.pack(n, p, positive_ion)
    report = model.residual_report(coordinate, derivative)
    coordinate_ro = _readonly_f64(
        coordinate, shape=(model.layout.size,), name="consistent coordinate"
    )
    derivative_ro = _readonly_f64(
        derivative, shape=(model.layout.size,), name="consistent derivative"
    )
    packed_ro = _readonly_f64(
        packed,
        shape=(3 * model.layout.node_count,),
        name="consistent physical state",
    )
    interface_ro = _readonly_f64(
        interface_state,
        shape=(model.layout.interface_state_count,),
        name="consistent interface state",
    )
    potential_ro = _readonly_f64(
        phi,
        shape=(model.layout.node_count,),
        name="consistent potential",
    )
    return IonInterfaceDAEConsistentInitialCondition(
        coordinate=coordinate_ro,
        derivative=derivative_ro,
        physical_state=packed_ro,
        interface_state_m3=interface_ro,
        potential_V=potential_ro,
        report=report,
        certified=bool(report.max_normalized_residual <= residual_tolerance),
        state_sha256=_state_sha256(
            "single-ion-algebraic-interface-dae-initial-v1",
            model.grid_m,
            coordinate_ro,
            derivative_ro,
            packed_ro,
            interface_ro,
            potential_ro,
        ),
    )


def finite_difference_state_jacobian(
    model: SingleIonAlgebraicInterfaceDAE,
    coordinate: np.ndarray,
    derivative: np.ndarray,
    *,
    relative_step: float = 1.0e-6,
) -> np.ndarray:
    """Independent central reference for combined-topology ``dF/dq``."""
    if not np.isfinite(relative_step) or relative_step <= 0.0:
        raise ValueError("relative_step must be finite and positive")
    value = np.asarray(coordinate, dtype=float)
    if value.shape != (model.layout.size,):
        raise ValueError("coordinate does not match the ion-interface layout")
    result = np.empty((model.layout.size, model.layout.size), dtype=float)
    potential_start = model.layout.potential_slice.start
    assert potential_start is not None
    for column in range(model.layout.size):
        scale = 1.0 if column < potential_start else model.layout.potential_scale_V
        step = relative_step * max(abs(value[column]), scale)
        plus = value.copy()
        minus = value.copy()
        plus[column] += step
        minus[column] -= step
        result[:, column] = (
            model.residual(plus, derivative) - model.residual(minus, derivative)
        ) / (2.0 * step)
    return result


def finite_difference_derivative_jacobian(
    model: SingleIonAlgebraicInterfaceDAE,
    coordinate: np.ndarray,
    derivative: np.ndarray,
    *,
    relative_step: float = 1.0e-6,
) -> np.ndarray:
    """Independent central reference for combined-topology ``dF/d(qdot)``."""
    if not np.isfinite(relative_step) or relative_step <= 0.0:
        raise ValueError("relative_step must be finite and positive")
    rate = np.asarray(derivative, dtype=float)
    if rate.shape != (model.layout.size,):
        raise ValueError("derivative does not match the ion-interface layout")
    layout = model.layout
    ion_slope = model.positive_ion_coordinate_derivative_m3(coordinate)
    result = np.empty((layout.size, layout.size), dtype=float)
    for column in range(layout.size):
        if column < layout.node_count:
            scale = (
                layout.electron_rate_scale_m3_s[column]
                / layout.electron_reference_m3[column]
            )
        elif column < 2 * layout.node_count:
            local = column - layout.node_count
            scale = layout.hole_rate_scale_m3_s[local] / layout.hole_reference_m3[local]
        elif column < 3 * layout.node_count:
            local = column - 2 * layout.node_count
            scale = layout.positive_ion_rate_scale_m3_s[local] / ion_slope[local]
        else:
            scale = 1.0
        step = relative_step * max(abs(rate[column]), scale)
        plus = rate.copy()
        minus = rate.copy()
        plus[column] += step
        minus[column] -= step
        result[:, column] = (
            model.residual(coordinate, plus) - model.residual(coordinate, minus)
        ) / (2.0 * step)
    return result


__all__ = [
    "IonInterfaceDAEConsistentInitialCondition",
    "IonInterfaceDAELayout",
    "IonInterfaceDAEResidualReport",
    "SingleIonAlgebraicInterfaceDAE",
    "build_single_ion_algebraic_interface_consistent_initial_condition",
    "build_single_ion_algebraic_interface_dae",
    "compatible_derivative",
    "finite_difference_derivative_jacobian",
    "finite_difference_state_jacobian",
    "project_single_ion_algebraic_interface_state",
]
