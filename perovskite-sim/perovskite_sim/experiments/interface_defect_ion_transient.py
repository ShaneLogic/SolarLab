"""Research-only two-sided interface-defect/mobile-ion transient.

The adapter adds conservative positive/negative ion storage to the D6-E2
index-1 interface DAE. Shared areal trap occupancy, six local algebraic trace
variables per interface, mobile ions, carriers, and Poisson are advanced in one
analytic sparse Newton system.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from types import SimpleNamespace

import numpy as np
from scipy import sparse

from perovskite_sim.constants import Q
from perovskite_sim.experiments.defect_ion_combined_impedance import (
    CombinedDCState,
    CombinedIonLayout,
    DefectIonCombinedError,
    _build_ion_layout,
    _component_inventories,
    _DCSolveContext,
    _ion_fields,
    _maximum_component_inventory_error,
    _solve_combined_dc,
)
from perovskite_sim.experiments.interface_defect_transient import (
    InterfaceDefectTransientError,
    InterfaceDefectTransientPolicy,
    _InterfaceTransientSystem,
    _RIGHT_FIRST,
    _integrate_trace,
    _readonly,
    _symmetric_relative_error,
    _validate_trace_inputs,
)
from perovskite_sim.experiments.quasi_fermi_steady_state import (
    _QuasiFermiSystem,
    _build_qf_material,
    _prepare_two_sided_material,
    _require_supported,
    _research_charge_off_stack,
)
from perovskite_sim.models.device import DeviceStack
from perovskite_sim.physics.contacts import (
    ContactThermodynamicError,
    require_contact_thermodynamic_certificate,
)
from perovskite_sim.physics.defect_distributions import (
    DEFAULT_DEFECT_ENERGY_QUADRATURE_ORDER,
)
from perovskite_sim.physics.interface_plane import FERMI_DIRAC_RICHARDSON
from perovskite_sim.physics.ion_migration import ion_face_flux_jacobian
from perovskite_sim.physics.dynamic_storage import log_density_increment
from perovskite_sim.physics.two_sided_interface import (
    TWO_SIDED_TRACE,
    _material_two_sided_interface_problem,
    solve_electrostatic_traces,
    solve_material_two_sided_interfaces_qss,
)
from perovskite_sim.solver.mol import MaterialArrays


INTERFACE_DEFECT_ION_TRANSIENT_SCOPE = (
    "research_two_sided_interface_defect_mobile_ion_transient_only"
)
INTERFACE_DEFECT_ION_TRANSIENT_VERSION = "interface-defect-ion-transient-v2"


class InterfaceDefectIonTransientError(InterfaceDefectTransientError):
    """The requested joint interface-defect/mobile-ion trace failed closed."""


class InterfaceDefectIonTransientCertificationError(InterfaceDefectIonTransientError):
    """A finite joint trace failed one or more evidence gates."""

    def __init__(
        self,
        message: str,
        result: "InterfaceDefectIonTransientResult",
    ) -> None:
        self.result = result
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class InterfaceIonDarkReference:
    """Microscopic interface binding plus the ion-equilibrated dark state."""

    equilibrium_occupancy: np.ndarray
    trap_density_m2: np.ndarray
    capture_velocities_m_s: np.ndarray
    interface_defect_document_sha256: tuple[str, ...]
    interface_transmission: float
    dc_state: CombinedDCState

    def __post_init__(self) -> None:
        occupancy = _readonly(self.equilibrium_occupancy)
        density = _readonly(self.trap_density_m2)
        velocities = _readonly(self.capture_velocities_m_s)
        count = occupancy.size
        if (
            occupancy.shape != (count,)
            or count == 0
            or density.shape != (count,)
            or velocities.shape != (count, 2)
            or not np.all(np.isfinite(occupancy))
            or not np.all(np.isfinite(density))
            or not np.all(np.isfinite(velocities))
            or np.any((occupancy <= 0.0) | (occupancy >= 1.0))
            or np.any(density <= 0.0)
            or np.any(velocities < 0.0)
        ):
            raise InterfaceDefectIonTransientError(
                "combined interface dark reference is invalid"
            )
        documents = tuple(str(value) for value in self.interface_defect_document_sha256)
        if len(documents) != count or any(
            len(value) != 64 or any(c not in "0123456789abcdef" for c in value)
            for value in documents
        ):
            raise InterfaceDefectIonTransientError(
                "interface document hashes are invalid"
            )
        transmission = float(self.interface_transmission)
        if not math.isfinite(transmission) or transmission < 0.0:
            raise InterfaceDefectIonTransientError(
                "interface transmission must be finite and non-negative"
            )
        if not self.dc_state.certificate.certified:
            raise InterfaceDefectIonTransientError(
                "combined interface dark reference must be DC certified"
            )
        object.__setattr__(self, "equilibrium_occupancy", occupancy)
        object.__setattr__(self, "trap_density_m2", density)
        object.__setattr__(self, "capture_velocities_m_s", velocities)
        object.__setattr__(self, "interface_defect_document_sha256", documents)
        object.__setattr__(self, "interface_transmission", transmission)


@dataclass(frozen=True, slots=True)
class InterfaceDefectIonTransientPolicy(InterfaceDefectTransientPolicy):
    ion_storage_atol_m3: float = 1.0
    maximum_eliminated_operator_relative_error: float = 1.0e-6
    maximum_ion_inventory_relative_drift: float = 1.0e-9
    maximum_current_decomposition_relative_error: float = 1.0e-14
    site_occupancy_ceiling: float = 0.999
    maximum_dc_normalized_residual: float = 1.0e-8
    maximum_dc_continuity_bound_A_m2: float = 1.0e-4
    maximum_dc_ionic_face_current_A_m2: float = 1.0e-6
    maximum_dc_inventory_error: float = 1.0e-10
    maximum_dc_poisson_residual: float = 1.0e-8
    maximum_dc_face_current_spread_A_m2: float = 1.0e-4
    dc_max_nfev: int = 1000

    def __post_init__(self) -> None:
        InterfaceDefectTransientPolicy.__post_init__(self)
        names = (
            "ion_storage_atol_m3",
            "maximum_ion_inventory_relative_drift",
            "maximum_current_decomposition_relative_error",
            "site_occupancy_ceiling",
            "maximum_dc_normalized_residual",
            "maximum_dc_continuity_bound_A_m2",
            "maximum_dc_ionic_face_current_A_m2",
            "maximum_dc_inventory_error",
            "maximum_dc_poisson_residual",
            "maximum_dc_face_current_spread_A_m2",
        )
        for name in names:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)
        if self.site_occupancy_ceiling >= 0.999999:
            raise ValueError(
                "site_occupancy_ceiling must remain below the ion-flux clip"
            )
        count = int(self.dc_max_nfev)
        if count <= 0:
            raise ValueError("dc_max_nfev must be positive")
        object.__setattr__(self, "dc_max_nfev", count)


@dataclass(frozen=True, slots=True)
class InterfaceDefectIonTransientCertificate:
    dc_operating_point_certified: bool
    dark_reference_certified: bool
    microscopic_binding_certified: bool
    qss_embedding_relative_error: float
    maximum_scaled_nonlinear_residual: float
    maximum_poisson_residual_C_m2: float
    maximum_local_carrier_normalized_residual: float
    maximum_local_gauss_normalized_residual: float
    maximum_analytic_jacobian_column_relative_error: float
    maximum_charge_balance_absolute_error_A_m2: float
    maximum_charge_balance_relative_error: float
    maximum_all_face_current_spread_relative: float
    maximum_two_sided_interface_total_current_relative_error: float
    maximum_eliminated_operator_relative_error: float
    eliminated_operator_components: tuple[tuple[str, float], ...]
    maximum_positive_ion_inventory_relative_drift: float
    maximum_negative_ion_inventory_relative_drift: float
    maximum_ion_inventory_relative_drift: float
    maximum_site_occupancy_fraction: float
    maximum_current_decomposition_relative_error: float
    maximum_refinement_state_change: float
    maximum_refinement_current_relative_change: float
    near_acceptance_nonmonotone_step_count: int
    analytic_jacobian_nnz: int
    dense_jacobian_entries: int
    sparse_linear_solver_used: bool
    clipping_used: bool
    certified: bool
    reasons: tuple[str, ...]
    scope: str = INTERFACE_DEFECT_ION_TRANSIENT_SCOPE
    version: str = INTERFACE_DEFECT_ION_TRANSIENT_VERSION


@dataclass(frozen=True, slots=True, eq=False)
class InterfaceDefectIonTransientResult:
    times_s: np.ndarray
    voltage_V: np.ndarray
    electron_density_m3: np.ndarray
    hole_density_m3: np.ndarray
    interface_occupancy: np.ndarray
    interface_quasi_steady_occupancy: np.ndarray
    positive_ion_density_m3: np.ndarray
    negative_ion_density_m3: np.ndarray | None
    electrostatic_potential_V: np.ndarray
    interface_trace_potential_V: np.ndarray
    interface_trace_state_m3: np.ndarray
    interface_sheet_charge_C_m2: np.ndarray
    electron_capture_flux_m2_s: np.ndarray
    hole_capture_flux_m2_s: np.ndarray
    electron_bulk_flux_m2_s: np.ndarray
    hole_bulk_flux_m2_s: np.ndarray
    carrier_conduction_current_faces_A_m2: np.ndarray
    positive_ion_current_faces_A_m2: np.ndarray
    negative_ion_current_faces_A_m2: np.ndarray | None
    conduction_current_faces_A_m2: np.ndarray
    displacement_current_faces_A_m2: np.ndarray
    total_current_faces_A_m2: np.ndarray
    interface_conduction_current_A_m2: np.ndarray
    interface_displacement_current_A_m2: np.ndarray
    interface_total_current_A_m2: np.ndarray
    integrated_free_interface_ion_charge_C_m2: np.ndarray
    positive_ion_component_inventory_m2: np.ndarray
    negative_ion_component_inventory_m2: np.ndarray | None
    newton_iterations: np.ndarray
    ion_layout: CombinedIonLayout
    dc_state: CombinedDCState
    dark_reference: InterfaceIonDarkReference
    policy: InterfaceDefectIonTransientPolicy
    certificate: InterfaceDefectIonTransientCertificate
    state_coordinate: str = "qf_interface_logit_log_ion_potential_local_algebraic"
    time_discretization: str = "backward_euler_index_1_dae"
    scope: str = INTERFACE_DEFECT_ION_TRANSIENT_SCOPE
    version: str = INTERFACE_DEFECT_ION_TRANSIENT_VERSION

    def __post_init__(self) -> None:
        times = _readonly(self.times_s)
        voltage = _readonly(self.voltage_V)
        if (
            times.ndim != 1
            or times.size < 2
            or not np.all(np.isfinite(times))
            or np.any(np.diff(times) <= 0.0)
            or voltage.shape != times.shape
            or not np.all(np.isfinite(voltage))
        ):
            raise InterfaceDefectIonTransientError("times/voltage trace is invalid")
        object.__setattr__(self, "times_s", times)
        object.__setattr__(self, "voltage_V", voltage)
        points, nodes = self.electron_density_m3.shape
        interfaces = self.dark_reference.trap_density_m2.size
        faces = nodes - 1
        shapes = {
            "electron_density_m3": (points, nodes),
            "hole_density_m3": (points, nodes),
            "interface_occupancy": (points, interfaces),
            "interface_quasi_steady_occupancy": (points, interfaces),
            "positive_ion_density_m3": (points, nodes),
            "electrostatic_potential_V": (points, nodes),
            "interface_trace_potential_V": (points, interfaces, 2),
            "interface_trace_state_m3": (points, interfaces, 4),
            "interface_sheet_charge_C_m2": (points, interfaces),
            "electron_capture_flux_m2_s": (points, interfaces, 2),
            "hole_capture_flux_m2_s": (points, interfaces, 2),
            "electron_bulk_flux_m2_s": (points, interfaces, 2),
            "hole_bulk_flux_m2_s": (points, interfaces, 2),
            "carrier_conduction_current_faces_A_m2": (points, faces),
            "positive_ion_current_faces_A_m2": (points, faces),
            "conduction_current_faces_A_m2": (points, faces),
            "displacement_current_faces_A_m2": (points, faces),
            "total_current_faces_A_m2": (points, faces),
            "interface_conduction_current_A_m2": (points, interfaces, 2),
            "interface_displacement_current_A_m2": (points, interfaces, 2),
            "interface_total_current_A_m2": (points, interfaces, 2),
            "integrated_free_interface_ion_charge_C_m2": (points,),
            "positive_ion_component_inventory_m2": (
                points,
                len(self.ion_layout.positive_components),
            ),
        }
        for name, shape in shapes.items():
            values = _readonly(getattr(self, name))
            if values.shape != shape or not np.all(np.isfinite(values)):
                raise InterfaceDefectIonTransientError(f"{name} is invalid")
            object.__setattr__(self, name, values)
        optional = (
            ("negative_ion_density_m3", self.negative_ion_density_m3, (points, nodes)),
            (
                "negative_ion_current_faces_A_m2",
                self.negative_ion_current_faces_A_m2,
                (points, faces),
            ),
            (
                "negative_ion_component_inventory_m2",
                self.negative_ion_component_inventory_m2,
                (points, len(self.ion_layout.negative_components)),
            ),
        )
        for name, value, shape in optional:
            if value is None:
                if self.ion_layout.negative_size:
                    raise InterfaceDefectIonTransientError(
                        f"{name} is required for active negative ions"
                    )
                continue
            values = _readonly(value)
            if values.shape != shape or not np.all(np.isfinite(values)):
                raise InterfaceDefectIonTransientError(f"{name} is invalid")
            object.__setattr__(self, name, values)
        if np.any(self.electron_density_m3 <= 0.0) or np.any(
            self.hole_density_m3 <= 0.0
        ):
            raise InterfaceDefectIonTransientError("carrier densities must be positive")
        if np.any(
            (self.interface_occupancy <= 0.0) | (self.interface_occupancy >= 1.0)
        ):
            raise InterfaceDefectIonTransientError(
                "interface occupancy must remain inside (0, 1)"
            )
        if np.any(self.interface_trace_state_m3 <= 0.0):
            raise InterfaceDefectIonTransientError(
                "interface trace densities must remain positive"
            )
        iterations = _readonly(self.newton_iterations, dtype=np.int64)
        if iterations.shape != (points,) or np.any(iterations < 0):
            raise InterfaceDefectIonTransientError("newton_iterations is invalid")
        object.__setattr__(self, "newton_iterations", iterations)
        if self.state_coordinate != (
            "qf_interface_logit_log_ion_potential_local_algebraic"
        ):
            raise InterfaceDefectIonTransientError("unexpected state coordinate")


@dataclass(slots=True)
class _InterfaceIonDeviceState:
    coordinate: np.ndarray
    dqfn: np.ndarray
    dqfp: np.ndarray
    n: np.ndarray
    p: np.ndarray
    occupancy: np.ndarray
    positive: np.ndarray
    negative: np.ndarray | None
    phi: np.ndarray
    local: tuple
    storage: np.ndarray
    rate: np.ndarray
    positive_rate: np.ndarray
    negative_rate: np.ndarray | None
    current_n: np.ndarray
    current_p: np.ndarray
    positive_flux: np.ndarray
    negative_flux: np.ndarray | None
    carrier_conduction: np.ndarray
    positive_current: np.ndarray
    negative_current: np.ndarray | None
    conduction: np.ndarray
    sheet_charge: np.ndarray
    poisson_residual: np.ndarray
    local_residual: np.ndarray
    storage_jacobian: sparse.csr_matrix
    rate_jacobian: sparse.csr_matrix
    poisson_jacobian: sparse.csr_matrix
    local_jacobian: sparse.csr_matrix


class _InterfaceIonTransientSystem(_InterfaceTransientSystem):
    def __init__(
        self,
        grid: np.ndarray,
        stack: DeviceStack,
        material: MaterialArrays,
        dc_state: CombinedDCState,
        qf_system: _QuasiFermiSystem,
        dark_reference: InterfaceIonDarkReference,
        occupancy_reference: np.ndarray,
        dynamic_dc: object,
        ion_layout: CombinedIonLayout,
        *,
        voltage: float,
        illuminated: bool,
        site_occupancy_ceiling: float,
    ) -> None:
        proxy = SimpleNamespace(
            V_app=float(voltage),
            electron_quasi_fermi_reference_V=np.asarray(qf_system.qfn0),
            hole_quasi_fermi_reference_V=np.asarray(qf_system.qfp0),
            electron_quasi_fermi_increment_V=np.asarray(
                dc_state.electron_qf_increment_V
            ),
            hole_quasi_fermi_increment_V=np.asarray(dc_state.hole_qf_increment_V),
        )
        super().__init__(
            grid,
            stack,
            material,
            proxy,
            dark_reference,
            occupancy_reference,
            dynamic_dc,
            illuminated=illuminated,
        )
        self.ion_layout = ion_layout
        self.positive_nodes = np.asarray(ion_layout.positive_nodes, dtype=int)
        self.negative_nodes = np.asarray(ion_layout.negative_nodes, dtype=int)
        self.positive_slice = slice(
            self.trap_slice.stop,
            self.trap_slice.stop + ion_layout.positive_size,
        )
        self.negative_slice = slice(
            self.positive_slice.stop,
            self.positive_slice.stop + ion_layout.negative_size,
        )
        self.potential_slice = slice(
            self.negative_slice.stop,
            self.negative_slice.stop + self.interior_count,
        )
        self.local_slice = slice(
            self.potential_slice.stop,
            self.potential_slice.stop + 6 * self.interface_count,
        )
        self.dimension = self.local_slice.stop
        self.reference_positive = np.asarray(
            dc_state.positive_ion_density_m3,
            dtype=float,
        )
        self.reference_negative = (
            None
            if dc_state.negative_ion_density_m3 is None
            else np.asarray(dc_state.negative_ion_density_m3, dtype=float)
        )
        self.site_occupancy_ceiling = float(site_occupancy_ceiling)
        self.positive_targets = _component_inventories(
            self.reference_positive,
            ion_layout.positive_components,
            self.widths,
        )
        self.negative_targets = (
            np.empty(0, dtype=float)
            if self.reference_negative is None
            else _component_inventories(
                self.reference_negative,
                ion_layout.negative_components,
                self.widths,
            )
        )
        self._maximum_eliminated_operator_components: dict[str, float] = {}

    def _site_fraction(
        self,
        positive: np.ndarray,
        negative: np.ndarray | None,
        *,
        reject: bool,
    ) -> float:
        values: list[np.ndarray] = []
        shared = bool(
            self.material.ion_steric_diffusion_only
            and self.material.ion_steric_shared_site
            and negative is not None
        )
        if self.positive_nodes.size:
            total = positive + negative if shared and negative is not None else positive
            limit = np.asarray(self.material.P_lim_node, dtype=float)
            values.append(total[self.positive_nodes] / limit[self.positive_nodes])
        if self.negative_nodes.size:
            if negative is None:
                raise InterfaceDefectIonTransientError("negative-ion block is missing")
            total = positive + negative if shared else negative
            limit = np.asarray(self.material.P_lim_neg_node, dtype=float)
            values.append(total[self.negative_nodes] / limit[self.negative_nodes])
        maximum = max(
            (float(np.max(item)) for item in values if item.size),
            default=0.0,
        )
        if not math.isfinite(maximum) or maximum < 0.0:
            raise InterfaceDefectIonTransientError("ion site occupancy is invalid")
        if reject and maximum >= self.site_occupancy_ceiling:
            raise InterfaceDefectIonTransientError(
                "ion coordinate reached the declared pre-clipping site ceiling"
            )
        return maximum

    def _ion_coordinates(
        self,
        coordinate: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        values = np.asarray(coordinate, dtype=float)
        positive = self.reference_positive.copy()
        if self.positive_nodes.size:
            with np.errstate(over="ignore", invalid="ignore"):
                positive[self.positive_nodes] *= np.exp(values[self.positive_slice])
            if not np.all(np.isfinite(positive[self.positive_nodes])) or np.any(
                positive[self.positive_nodes] <= 0.0
            ):
                raise InterfaceDefectIonTransientError(
                    "positive-ion coordinate overflowed"
                )
        negative = None
        if self.reference_negative is not None:
            negative = self.reference_negative.copy()
            if self.negative_nodes.size:
                with np.errstate(over="ignore", invalid="ignore"):
                    negative[self.negative_nodes] *= np.exp(values[self.negative_slice])
                if not np.all(np.isfinite(negative[self.negative_nodes])) or np.any(
                    negative[self.negative_nodes] <= 0.0
                ):
                    raise InterfaceDefectIonTransientError(
                        "negative-ion coordinate overflowed"
                    )
        self._site_fraction(positive, negative, reject=True)
        return positive, negative

    @staticmethod
    def _density_jacobian(
        density: np.ndarray,
        nodes: np.ndarray,
        block: slice,
        dimension: int,
    ) -> sparse.csr_matrix:
        matrix = sparse.lil_matrix((density.size, dimension))
        for local, node in enumerate(nodes):
            matrix[int(node), block.start + local] = density[int(node)]
        return matrix.tocsr()

    def _ion_jacobians(
        self,
        phi: np.ndarray,
        positive: np.ndarray,
        negative: np.ndarray | None,
    ):
        positive_density = self._density_jacobian(
            positive,
            self.positive_nodes,
            self.positive_slice,
            self.dimension,
        )
        negative_density = sparse.csr_matrix((self.node_count, self.dimension))
        if negative is not None:
            negative_density = self._density_jacobian(
                negative,
                self.negative_nodes,
                self.negative_slice,
                self.dimension,
            )
        phi_jacobian = sparse.lil_matrix((self.node_count, self.dimension))
        for local, node in enumerate(range(1, self.node_count - 1)):
            phi_jacobian[node, self.potential_slice.start + local] = (
                self.thermal_voltage
            )
        phi_jacobian = phi_jacobian.tocsr()
        shared = bool(
            self.material.ion_steric_diffusion_only
            and self.material.ion_steric_shared_site
            and negative is not None
        )

        def face_matrix(local, own, partner):
            return (
                sparse.diags(local.density_left_derivative) @ own[:-1]
                + sparse.diags(local.density_right_derivative) @ own[1:]
                + sparse.diags(local.partner_left_derivative) @ partner[:-1]
                + sparse.diags(local.partner_right_derivative) @ partner[1:]
                + sparse.diags(local.potential_left_derivative) @ phi_jacobian[:-1]
                + sparse.diags(local.potential_right_derivative) @ phi_jacobian[1:]
            ).tocsr()

        positive_local = ion_face_flux_jacobian(
            phi,
            positive,
            np.diff(self.grid),
            self.material.D_ion_face,
            self.thermal_voltage,
            self.material.P_lim_face,
            steric_diffusion_only=self.material.ion_steric_diffusion_only,
            P_lim_node=self.material.P_lim_node,
            P_other_node=negative if shared else None,
            drift_sign=1.0,
        )
        if np.any(
            (np.asarray(self.material.D_ion_face) > 0.0)
            & ~positive_local.differentiable_faces
        ):
            raise InterfaceDefectIonTransientError(
                "positive-ion flux reached a non-differentiable face"
            )
        positive_flux = face_matrix(
            positive_local,
            positive_density,
            negative_density,
        )
        negative_flux = sparse.csr_matrix((self.node_count - 1, self.dimension))
        if negative is not None:
            negative_local = ion_face_flux_jacobian(
                phi,
                negative,
                np.diff(self.grid),
                self.material.D_ion_neg_face,
                self.thermal_voltage,
                self.material.P_lim_neg_face,
                steric_diffusion_only=self.material.ion_steric_diffusion_only,
                P_lim_node=self.material.P_lim_neg_node,
                P_other_node=positive if shared else None,
                drift_sign=-1.0,
            )
            if np.any(
                (np.asarray(self.material.D_ion_neg_face) > 0.0)
                & ~negative_local.differentiable_faces
            ):
                raise InterfaceDefectIonTransientError(
                    "negative-ion flux reached a non-differentiable face"
                )
            negative_flux = face_matrix(
                negative_local,
                negative_density,
                positive_density,
            )
        inverse_volume = sparse.diags(1.0 / self.widths)
        positive_rate = (inverse_volume @ self._divergence @ positive_flux).tocsr()
        negative_rate = (inverse_volume @ self._divergence @ negative_flux).tocsr()
        return (
            positive_density,
            negative_density,
            positive_rate,
            negative_rate,
        )

    def evaluate(
        self,
        coordinate: np.ndarray,
        voltage: float,
    ) -> _InterfaceIonDeviceState:
        (
            dqfn,
            dqfp,
            phi,
            n,
            p,
            occupancy,
            trace_potential,
            trace_log_state,
        ) = super()._coordinates(coordinate, voltage)
        positive, negative = self._ion_coordinates(coordinate)
        local, interface_qss = self._local_states(
            n,
            p,
            phi,
            occupancy,
            trace_potential,
            trace_log_state,
        )
        source = self._source(n, p, phi, voltage, interface_qss)
        transport_n, transport_p, current_n, current_p = self._currents(
            dqfn,
            dqfp,
            phi,
            n,
            p,
            local,
        )
        rate_n = source[: self.node_count] + (self._divergence @ transport_n) / (
            Q * self.widths
        )
        rate_p = source[self.node_count :] - (self._divergence @ transport_p) / (
            Q * self.widths
        )
        positive_rate, negative_rate, positive_flux, negative_flux = _ion_fields(
            self.grid,
            self.material,
            positive,
            negative,
            phi,
        )
        capture = np.asarray([item.tangent.balance.capture_flux_m2_s for item in local])
        trap_rate = capture[:, [0, 2]].sum(axis=1) - capture[:, [1, 3]].sum(axis=1)
        storage_parts = [
            n[1:-1],
            p[1:-1],
            self.trap_density * occupancy,
            positive[self.positive_nodes],
        ]
        rate_parts = [
            rate_n[1:-1],
            rate_p[1:-1],
            trap_rate,
            positive_rate[self.positive_nodes],
        ]
        if self.negative_nodes.size:
            if negative is None or negative_rate is None:
                raise InterfaceDefectIonTransientError(
                    "active negative-ion storage block is missing"
                )
            storage_parts.append(negative[self.negative_nodes])
            rate_parts.append(negative_rate[self.negative_nodes])
        storage = np.concatenate(storage_parts)
        rate = np.concatenate(rate_parts)
        rho, _ = self.system._bulk_space_charge_and_tangent(
            n,
            p,
            positive_ion_density_m3=positive,
            negative_ion_density_m3=negative,
        )
        factor = self.material.poisson_factor
        poisson = self._poisson_laplacian @ phi + rho[1:-1] * factor.h_cell
        sheet_charge = np.asarray([item.sheet_charge_C_m2 for item in local])
        for index, (left, right) in enumerate(zip(self.left_nodes, self.right_nodes)):
            weight_left, weight_right = self._sheet_weights(index)
            poisson[left - 1] += weight_left * sheet_charge[index]
            poisson[right - 1] += weight_right * sheet_charge[index]
        local_residual = np.concatenate(
            [
                np.r_[item.electrostatic_residual, item.tangent.balance.residual_m2_s]
                for item in local
            ]
        )
        base_storage, base_rate, base_poisson, local_jacobian = super()._jacobians(
            phi,
            n,
            p,
            occupancy,
            local,
        )
        (
            positive_density_jacobian,
            negative_density_jacobian,
            positive_rate_jacobian,
            negative_rate_jacobian,
        ) = self._ion_jacobians(phi, positive, negative)
        storage_blocks = [base_storage, positive_density_jacobian[self.positive_nodes]]
        rate_blocks = [base_rate, positive_rate_jacobian[self.positive_nodes]]
        if self.negative_nodes.size:
            storage_blocks.append(negative_density_jacobian[self.negative_nodes])
            rate_blocks.append(negative_rate_jacobian[self.negative_nodes])
        storage_jacobian = sparse.vstack(storage_blocks, format="csr")
        rate_jacobian = sparse.vstack(rate_blocks, format="csr")
        ion_charge_jacobian = Q * (
            positive_density_jacobian - negative_density_jacobian
        )
        poisson_jacobian = (
            base_poisson + sparse.diags(factor.h_cell) @ ion_charge_jacobian[1:-1]
        ).tocsr()
        carrier_conduction = self.polarity * (current_n + current_p)
        positive_current = self.polarity * Q * positive_flux
        negative_current = (
            None if negative_flux is None else -self.polarity * Q * negative_flux
        )
        conduction = carrier_conduction + positive_current
        if negative_current is not None:
            conduction = conduction + negative_current
        arrays = (storage, rate, poisson, local_residual, conduction)
        if any(not np.all(np.isfinite(item)) for item in arrays):
            raise InterfaceDefectIonTransientError(
                "joint interface/ion operator produced a non-finite value"
            )
        return _InterfaceIonDeviceState(
            coordinate=np.asarray(coordinate, dtype=float).copy(),
            dqfn=dqfn,
            dqfp=dqfp,
            n=n,
            p=p,
            occupancy=occupancy,
            positive=positive,
            negative=negative,
            phi=phi,
            local=local,
            storage=storage,
            rate=rate,
            positive_rate=positive_rate,
            negative_rate=negative_rate,
            current_n=current_n,
            current_p=current_p,
            positive_flux=positive_flux,
            negative_flux=negative_flux,
            carrier_conduction=carrier_conduction,
            positive_current=positive_current,
            negative_current=negative_current,
            conduction=conduction,
            sheet_charge=sheet_charge,
            poisson_residual=np.asarray(poisson),
            local_residual=local_residual,
            storage_jacobian=storage_jacobian,
            rate_jacobian=rate_jacobian,
            poisson_jacobian=poisson_jacobian,
            local_jacobian=local_jacobian,
        )

    def interface_current_sides(
        self,
        state: _InterfaceIonDeviceState,
        previous: _InterfaceIonDeviceState | None = None,
        dt: float | None = None,
    ):
        conduction, displacement, _ = super().interface_current_sides(
            state,
            previous,
            dt,
        )
        ionic = state.positive_current.copy()
        if state.negative_current is not None:
            ionic += state.negative_current
        for index, face in enumerate(self.interface_faces):
            conduction[index] += ionic[face]
        return conduction, displacement, conduction + displacement

    def storage_scale(
        self,
        previous_storage: np.ndarray,
        previous_state: _InterfaceIonDeviceState,
        dt: float,
        policy: InterfaceDefectIonTransientPolicy,
    ) -> np.ndarray:
        reference = np.r_[
            self.reference_n[1:-1],
            self.reference_p[1:-1],
            self.trap_density * self.reference_occupancy,
            self.reference_positive[self.positive_nodes],
            (
                np.empty(0)
                if self.reference_negative is None
                else self.reference_negative[self.negative_nodes]
            ),
        ]
        absolute = np.r_[
            np.full(2 * self.interior_count, policy.carrier_storage_atol_m3),
            np.full(self.interface_count, policy.interface_storage_atol_m2),
            np.full(self.ion_layout.size, policy.ion_storage_atol_m3),
        ]
        scale = absolute + policy.storage_relative_tolerance * np.maximum(
            np.abs(previous_storage),
            np.abs(reference),
        )
        rounding = 256.0 * np.finfo(float).eps
        coordinate_resolution = rounding * np.maximum(
            np.abs(previous_state.coordinate),
            1.0,
        )
        resolvable_storage = (
            abs(previous_state.storage_jacobian)
            + dt * abs(previous_state.rate_jacobian)
        ) @ coordinate_resolution
        scale += np.asarray(resolvable_storage).reshape(-1)
        for local, node in enumerate(range(1, self.node_count - 1)):
            scale[local] += (
                dt
                * rounding
                * max(
                    abs(previous_state.current_n[node - 1]),
                    abs(previous_state.current_n[node]),
                )
                / (Q * self.widths[node])
            )
            scale[self.interior_count + local] += (
                dt
                * rounding
                * max(
                    abs(previous_state.current_p[node - 1]),
                    abs(previous_state.current_p[node]),
                )
                / (Q * self.widths[node])
            )
        return scale

    def storage_increment(
        self,
        state: _InterfaceIonDeviceState,
        previous: _InterfaceIonDeviceState,
    ) -> np.ndarray:
        base = super().storage_increment(state, previous)
        coordinate_increment = state.coordinate - previous.coordinate
        blocks = [base]
        if self.positive_nodes.size:
            blocks.append(
                log_density_increment(
                    previous.positive[self.positive_nodes],
                    coordinate_increment[self.positive_slice],
                )
            )
        if self.negative_nodes.size:
            if previous.negative is None:
                raise InterfaceDefectIonTransientError("negative-ion block is missing")
            blocks.append(
                log_density_increment(
                    previous.negative[self.negative_nodes],
                    coordinate_increment[self.negative_slice],
                )
            )
        return np.concatenate(blocks)

    def integrated_charge(self, state: _InterfaceIonDeviceState) -> float:
        charge = float(
            np.sum(Q * (state.p[1:-1] - state.n[1:-1]) * self.widths[1:-1])
            + np.sum(state.sheet_charge)
        )
        charge += Q * float(
            np.sum(
                (state.positive[1:-1] - self.material.P_ion0[1:-1]) * self.widths[1:-1]
            )
        )
        if state.negative is not None:
            charge -= Q * float(
                np.sum(
                    (state.negative[1:-1] - self.material.P_ion0_neg[1:-1])
                    * self.widths[1:-1]
                )
            )
        return charge

    def integrated_charge_increment(
        self,
        state: _InterfaceIonDeviceState,
        previous: _InterfaceIonDeviceState,
    ) -> float:
        charge = super().integrated_charge_increment(state, previous)
        coordinate_increment = state.coordinate - previous.coordinate
        if self.positive_nodes.size:
            values = log_density_increment(
                previous.positive[self.positive_nodes],
                coordinate_increment[self.positive_slice],
            )
            interior = (self.positive_nodes > 0) & (
                self.positive_nodes < self.node_count - 1
            )
            charge += Q * float(
                np.sum(values[interior] * self.widths[self.positive_nodes[interior]])
            )
        if self.negative_nodes.size:
            if previous.negative is None:
                raise InterfaceDefectIonTransientError("negative-ion block is missing")
            values = log_density_increment(
                previous.negative[self.negative_nodes],
                coordinate_increment[self.negative_slice],
            )
            interior = (self.negative_nodes > 0) & (
                self.negative_nodes < self.node_count - 1
            )
            charge -= Q * float(
                np.sum(values[interior] * self.widths[self.negative_nodes[interior]])
            )
        return charge

    @staticmethod
    def _relative(left: np.ndarray, right: np.ndarray, floor: float) -> float:
        scale = max(
            float(np.max(np.abs(left), initial=0.0)),
            float(np.max(np.abs(right), initial=0.0)),
            floor,
        )
        return float(np.max(np.abs(left - right), initial=0.0)) / scale

    def eliminated_operator_error(
        self,
        state: _InterfaceIonDeviceState,
        voltage: float,
    ) -> float:
        eliminated = self.system.evaluate_quasi_fermi_increments_defect_ion_combined(
            state.dqfn,
            state.dqfp,
            1.0 if self.illuminated else 0.0,
            positive_ion_density_m3=state.positive,
            negative_ion_density_m3=state.negative,
            dynamic_interface_occupancy=state.occupancy,
            V_app=float(voltage),
        )
        fixed = eliminated.interface_charge_dynamic
        if fixed is None or fixed.qss.capture_flux_m2_s is None:
            raise InterfaceDefectIonTransientError(
                "eliminated comparison lost fixed-interface evidence"
            )
        positive_rate, negative_rate, positive_flux, negative_flux = _ion_fields(
            self.grid,
            self.material,
            state.positive,
            state.negative,
            eliminated.phi,
        )
        current_scale = max(
            float(np.max(np.abs(state.current_n))),
            float(np.max(np.abs(state.current_p))),
            1.0,
        )
        rate_scale = max(
            current_scale
            / (Q * float(np.min(self.widths[1:-1])))
            * math.sqrt(np.finfo(float).eps),
            1.0,
        )
        ion_rate_scale = max(
            float(np.max(np.abs(state.positive_rate), initial=0.0)),
            (
                0.0
                if state.negative_rate is None
                else float(np.max(np.abs(state.negative_rate), initial=0.0))
            ),
            1.0,
        )
        canonical_state = np.asarray([item.state_m3 for item in state.local])
        eliminated_state = np.asarray(fixed.qss.state_m3).reshape(-1, 4)[
            :,
            _RIGHT_FIRST,
        ]
        canonical_capture = np.asarray(
            [item.tangent.balance.capture_flux_m2_s for item in state.local]
        )
        eliminated_capture = np.asarray(fixed.qss.capture_flux_m2_s).reshape(-1, 4)[
            :,
            _RIGHT_FIRST,
        ]
        trace_shift = np.empty((self.interface_count, 2))
        for index, item in enumerate(state.local):
            geometry, _physics, bulk = _material_two_sided_interface_problem(
                self.material,
                self.stack,
                state.n,
                state.p,
                state.phi,
                index,
                cross_transmission=self.dark_reference.interface_transmission,
            )
            off = solve_electrostatic_traces(geometry, bulk)
            trace_shift[index] = item.trace_potential - np.array(
                [off.phi_left_V, off.phi_right_V]
            )
        values = {
            "electron_density": self._relative(
                state.n, eliminated.y[: self.node_count], 1.0
            ),
            "hole_density": self._relative(
                state.p,
                eliminated.y[self.node_count : 2 * self.node_count],
                1.0,
            ),
            "potential": self._relative(
                state.phi, eliminated.phi, self.thermal_voltage
            ),
            "electron_rate": self._relative(
                state.rate[: self.interior_count],
                eliminated.rate_n[1:-1],
                rate_scale,
            ),
            "hole_rate": self._relative(
                state.rate[self.interior_count : 2 * self.interior_count],
                eliminated.rate_p[1:-1],
                rate_scale,
            ),
            "electron_current": self._relative(
                state.current_n, eliminated.current_n, current_scale
            ),
            "hole_current": self._relative(
                state.current_p, eliminated.current_p, current_scale
            ),
            "local_state": self._relative(canonical_state, eliminated_state, 1.0),
            "capture_flux": self._relative(canonical_capture, eliminated_capture, 1.0),
            "sheet_charge": self._relative(
                state.sheet_charge,
                np.asarray(fixed.incremental_sheet_charge_C_m2),
                Q * float(np.max(self.trap_density)),
            ),
            "trace_potential": self._relative(
                trace_shift,
                np.asarray(fixed.trace_potential_shift_V),
                self.thermal_voltage,
            ),
            "positive_ion_rate": self._relative(
                state.positive_rate, positive_rate, ion_rate_scale
            ),
            "positive_ion_flux": self._relative(
                state.positive_flux, positive_flux, 1.0
            ),
        }
        if state.negative is not None:
            if (
                state.negative_rate is None
                or state.negative_flux is None
                or negative_rate is None
                or negative_flux is None
            ):
                raise InterfaceDefectIonTransientError(
                    "eliminated comparison lost negative ions"
                )
            values.update(
                negative_ion_rate=self._relative(
                    state.negative_rate,
                    negative_rate,
                    ion_rate_scale,
                ),
                negative_ion_flux=self._relative(
                    state.negative_flux, negative_flux, 1.0
                ),
            )
        for name, value in values.items():
            self._maximum_eliminated_operator_components[name] = max(
                value,
                self._maximum_eliminated_operator_components.get(name, 0.0),
            )
        return max(values.values())


def _refinement_changes(coarse, fine, system: _InterfaceIonTransientSystem):
    n_coarse = np.asarray([state.n for state in coarse.states])
    n_fine = np.asarray([state.n for state in fine.states])
    p_coarse = np.asarray([state.p for state in coarse.states])
    p_fine = np.asarray([state.p for state in fine.states])
    f_coarse = np.asarray([state.occupancy for state in coarse.states])
    f_fine = np.asarray([state.occupancy for state in fine.states])
    phi_coarse = np.asarray([state.phi for state in coarse.states])
    phi_fine = np.asarray([state.phi for state in fine.states])
    local_coarse = np.asarray(
        [[item.state_m3 for item in state.local] for state in coarse.states]
    )
    local_fine = np.asarray(
        [[item.state_m3 for item in state.local] for state in fine.states]
    )
    state_change = max(
        float(np.max(np.abs(np.log(n_coarse / n_fine)))),
        float(np.max(np.abs(np.log(p_coarse / p_fine)))),
        float(np.max(np.abs(np.log(local_coarse / local_fine)))),
        float(np.max(np.abs(f_coarse - f_fine))),
        float(np.max(np.abs(phi_coarse - phi_fine)))
        / max(float(np.ptp(phi_fine)), 0.025),
    )
    if system.positive_nodes.size:
        positive_coarse = np.asarray([state.positive for state in coarse.states])
        positive_fine = np.asarray([state.positive for state in fine.states])
        state_change = max(
            state_change,
            float(
                np.max(
                    np.abs(
                        np.log(
                            positive_coarse[:, system.positive_nodes]
                            / positive_fine[:, system.positive_nodes]
                        )
                    )
                )
            ),
        )
    if system.negative_nodes.size:
        negative_coarse = np.asarray([state.negative for state in coarse.states])
        negative_fine = np.asarray([state.negative for state in fine.states])
        state_change = max(
            state_change,
            float(
                np.max(
                    np.abs(
                        np.log(
                            negative_coarse[:, system.negative_nodes]
                            / negative_fine[:, system.negative_nodes]
                        )
                    )
                )
            ),
        )
    current_scale = max(float(np.max(np.abs(fine.total_current[1:]))), 1.0)
    current_change = (
        float(np.max(np.abs(coarse.total_current[1:] - fine.total_current[1:])))
        / current_scale
    )
    return state_change, current_change


def _inventory_trace(states, layout, widths, *, positive):
    components = layout.positive_components if positive else layout.negative_components
    return np.asarray(
        [
            _component_inventories(
                state.positive if positive else state.negative,
                components,
                widths,
            )
            for state in states
        ],
        dtype=float,
    )


def _inventory_drift(trace: np.ndarray, target: np.ndarray) -> float:
    if trace.shape[1] == 0:
        return 0.0
    return max(_maximum_component_inventory_error(values, target) for values in trace)


def _embedding_error(reference: object, dynamic: object, material: MaterialArrays):
    current_scale = max(
        float(np.max(np.abs(reference.current_n + reference.current_p))),
        1.0,
    )
    rate_scale = max(
        current_scale / (Q * float(np.min(np.asarray(material.dx_cell)[1:-1]))),
        1.0,
    )
    values = (
        _symmetric_relative_error(reference.y, dynamic.y),
        _symmetric_relative_error(reference.phi, dynamic.phi, material.V_T_device),
        _symmetric_relative_error(reference.rate_n, dynamic.rate_n, rate_scale),
        _symmetric_relative_error(reference.rate_p, dynamic.rate_p, rate_scale),
        _symmetric_relative_error(
            reference.current_n, dynamic.current_n, current_scale
        ),
        _symmetric_relative_error(
            reference.current_p, dynamic.current_p, current_scale
        ),
    )
    return max(values)


def run_interface_defect_ion_device_transient(
    x: np.ndarray,
    stack: DeviceStack,
    times_s: object,
    voltage_V: object,
    *,
    illuminated: bool = False,
    mat: MaterialArrays | None = None,
    policy: InterfaceDefectIonTransientPolicy | None = None,
    require_certificate: bool = True,
) -> InterfaceDefectIonTransientResult:
    """Integrate a shared-occupancy two-sided interface/mobile-ion DAE."""
    grid = np.asarray(x, dtype=float)
    if (
        grid.ndim != 1
        or grid.size < 4
        or not np.all(np.isfinite(grid))
        or np.any(np.diff(grid) <= 0.0)
    ):
        raise ValueError("x must be finite, increasing, and contain >=4 nodes")
    times, voltage = _validate_trace_inputs(times_s, voltage_V)
    if not isinstance(illuminated, (bool, np.bool_)):
        raise TypeError("illuminated must be boolean")
    if not isinstance(require_certificate, (bool, np.bool_)):
        raise TypeError("require_certificate must be boolean")
    resolved_policy = policy or InterfaceDefectIonTransientPolicy()
    if not isinstance(resolved_policy, InterfaceDefectIonTransientPolicy):
        raise TypeError("policy must be an InterfaceDefectIonTransientPolicy or None")
    try:
        charge_off_stack, microscopic = _research_charge_off_stack(stack)
    except (TypeError, ValueError) as exc:
        raise InterfaceDefectIonTransientError(
            f"interface/ion transient needs a microscopic interface: {exc}"
        ) from exc
    if not microscopic.documents:
        raise InterfaceDefectIonTransientError(
            "interface/ion transient requires an explicit interface defect"
        )
    material = mat
    if material is None:
        material = _build_qf_material(
            grid,
            charge_off_stack,
            defect_energy_quadrature_order=DEFAULT_DEFECT_ENERGY_QUADRATURE_ORDER,
        )
        material = _prepare_two_sided_material(grid, charge_off_stack, material)
        material = replace(
            material,
            N_iface_state=0,
            iface_state_v_th=1.0e5,
            iface_state_live_proj=True,
            iface_state_shared_occ=True,
            iface_state_physical_offsets=True,
            iface_qss_exclusive_transport=True,
            iface_qss_cross_transmission=1.0,
            iface_qss_transport_model=FERMI_DIRAC_RICHARDSON,
            iface_qss_allow_inexact_inner=True,
        )
    try:
        _require_supported(
            material,
            interface_boundary=True,
            interface_topology=TWO_SIDED_TRACE,
            allow_mobile_ions=True,
        )
        contact = require_contact_thermodynamic_certificate(charge_off_stack, material)
        ion_layout = _build_ion_layout(material)
    except (DefectIonCombinedError, ContactThermodynamicError, ValueError) as exc:
        raise InterfaceDefectIonTransientError(
            f"interface/ion material contract failed: {exc}"
        ) from exc
    if material.monovalent_bulk_defects is not None:
        raise InterfaceDefectIonTransientError(
            "D6-E3b excludes simultaneous bulk and interface defects"
        )
    widths = np.asarray(material.dx_cell, dtype=float)
    positive_target = _component_inventories(
        np.asarray(material.P_ion0, dtype=float),
        ion_layout.positive_components,
        widths,
    )
    negative_target = np.empty(0, dtype=float)
    if material.has_dual_ions:
        negative_target = _component_inventories(
            np.asarray(material.P_ion0_neg, dtype=float),
            ion_layout.negative_components,
            widths,
        )
    charge_off_system = _QuasiFermiSystem(
        grid,
        charge_off_stack,
        material,
        0.0,
        interface_boundary=True,
        interface_topology=TWO_SIDED_TRACE,
        interface_transmission=1.0,
        interface_transport_model=FERMI_DIRAC_RICHARDSON,
        poisson_tolerance_V=1.0e-13,
        poisson_max_iterations=100,
    )
    dark_context = _DCSolveContext(
        grid,
        charge_off_stack,
        material,
        charge_off_system,
        ion_layout,
        0.0,
        0.0,
        positive_target,
        negative_target,
        contact,
    )
    dark_state, dark_value = _solve_combined_dc(
        dark_context,
        maximum_normalized_residual=resolved_policy.maximum_dc_normalized_residual,
        maximum_continuity_bound_A_m2=(
            resolved_policy.maximum_dc_continuity_bound_A_m2
        ),
        maximum_ionic_face_current_A_m2=(
            resolved_policy.maximum_dc_ionic_face_current_A_m2
        ),
        maximum_inventory_error=resolved_policy.maximum_dc_inventory_error,
        maximum_poisson_residual=resolved_policy.maximum_dc_poisson_residual,
        maximum_face_current_spread_A_m2=(
            resolved_policy.maximum_dc_face_current_spread_A_m2
        ),
        max_nfev=resolved_policy.dc_max_nfev,
    )
    if not dark_state.certificate.certified:
        raise InterfaceDefectIonTransientError(
            "combined dark state is not certified: "
            + ", ".join(dark_state.certificate.reasons)
        )
    local_dark = solve_material_two_sided_interfaces_qss(
        material,
        charge_off_stack,
        dark_value.y[: grid.size],
        dark_value.y[grid.size : 2 * grid.size],
        dark_value.phi,
        cross_transmission=1.0,
        interface_transport_model=FERMI_DIRAC_RICHARDSON,
        fail_on_residual=True,
    )
    dark_reference = InterfaceIonDarkReference(
        equilibrium_occupancy=np.asarray(local_dark.occupancy),
        trap_density_m2=np.asarray(microscopic.trap_density_m2),
        capture_velocities_m_s=np.asarray(microscopic.capture_velocities_m_s),
        interface_defect_document_sha256=microscopic.document_sha256,
        interface_transmission=1.0,
        dc_state=dark_state,
    )
    qf_system = _QuasiFermiSystem(
        grid,
        charge_off_stack,
        material,
        float(voltage[0]),
        interface_boundary=True,
        interface_topology=TWO_SIDED_TRACE,
        interface_transmission=1.0,
        interface_transport_model=FERMI_DIRAC_RICHARDSON,
        interface_charge_reference_occupancy=dark_reference.equilibrium_occupancy,
        interface_charge_trap_density_m2=dark_reference.trap_density_m2,
        poisson_tolerance_V=1.0e-13,
        poisson_max_iterations=100,
    )
    context = _DCSolveContext(
        grid,
        charge_off_stack,
        material,
        qf_system,
        ion_layout,
        1.0 if illuminated else 0.0,
        float(voltage[0]),
        positive_target,
        negative_target,
        contact,
    )
    dc_state, qss_dc = _solve_combined_dc(
        context,
        initial_state=(dark_state if voltage[0] == 0.0 and not illuminated else None),
        maximum_normalized_residual=resolved_policy.maximum_dc_normalized_residual,
        maximum_continuity_bound_A_m2=(
            resolved_policy.maximum_dc_continuity_bound_A_m2
        ),
        maximum_ionic_face_current_A_m2=(
            resolved_policy.maximum_dc_ionic_face_current_A_m2
        ),
        maximum_inventory_error=resolved_policy.maximum_dc_inventory_error,
        maximum_poisson_residual=resolved_policy.maximum_dc_poisson_residual,
        maximum_face_current_spread_A_m2=(
            resolved_policy.maximum_dc_face_current_spread_A_m2
        ),
        max_nfev=resolved_policy.dc_max_nfev,
    )
    if not dc_state.certificate.certified:
        raise InterfaceDefectIonTransientError(
            "combined operating point is not certified: "
            + ", ".join(dc_state.certificate.reasons)
        )
    if qss_dc.interface_charge_qss is None:
        raise InterfaceDefectIonTransientError(
            "combined operating point lost interface occupancy"
        )
    occupancy = np.asarray(qss_dc.interface_charge_qss.qss.occupancy, dtype=float)
    dynamic_dc = qf_system.evaluate_quasi_fermi_increments_defect_ion_combined(
        np.asarray(dc_state.electron_qf_increment_V),
        np.asarray(dc_state.hole_qf_increment_V),
        1.0 if illuminated else 0.0,
        positive_ion_density_m3=np.asarray(dc_state.positive_ion_density_m3),
        negative_ion_density_m3=(
            None
            if dc_state.negative_ion_density_m3 is None
            else np.asarray(dc_state.negative_ion_density_m3)
        ),
        dynamic_interface_occupancy=occupancy,
        V_app=float(voltage[0]),
    )
    qss_embedding_error = _embedding_error(qss_dc, dynamic_dc, material)
    system = _InterfaceIonTransientSystem(
        grid,
        charge_off_stack,
        material,
        dc_state,
        qf_system,
        dark_reference,
        occupancy,
        dynamic_dc,
        ion_layout,
        voltage=float(voltage[0]),
        illuminated=bool(illuminated),
        site_occupancy_ceiling=resolved_policy.site_occupancy_ceiling,
    )
    try:
        levels = tuple(
            _integrate_trace(system, times, voltage, substeps, resolved_policy)
            for substeps in resolved_policy.refinement_substeps
        )
    except InterfaceDefectTransientError as exc:
        raise InterfaceDefectIonTransientError(
            f"joint interface/ion transient solve failed: {exc}"
        ) from exc
    changes = tuple(
        _refinement_changes(coarse, fine, system)
        for coarse, fine in zip(levels, levels[1:])
    )
    refinement_state, refinement_current = changes[-1]
    final = levels[-1]
    positive_inventory = _inventory_trace(
        final.states,
        ion_layout,
        widths,
        positive=True,
    )
    negative_inventory = (
        None
        if not ion_layout.negative_size
        else _inventory_trace(
            final.states,
            ion_layout,
            widths,
            positive=False,
        )
    )
    positive_drift = _inventory_drift(positive_inventory, system.positive_targets)
    negative_drift = (
        0.0
        if negative_inventory is None
        else _inventory_drift(negative_inventory, system.negative_targets)
    )
    inventory_drift = max(positive_drift, negative_drift)
    site_fraction = max(
        system._site_fraction(state.positive, state.negative, reject=False)
        for state in final.states
    )
    decomposition_error = 0.0
    for state in final.states:
        expected = state.carrier_conduction + state.positive_current
        if state.negative_current is not None:
            expected += state.negative_current
        scale = max(float(np.max(np.abs(expected))), 1.0)
        decomposition_error = max(
            decomposition_error,
            float(np.max(np.abs(expected - state.conduction))) / scale,
        )
    reasons: list[str] = []
    gates = (
        (
            "nonlinear_residual_exceeds_limit",
            final.maximum_scaled_residual,
            resolved_policy.maximum_scaled_nonlinear_residual,
        ),
        (
            "qss_embedding_failed",
            qss_embedding_error,
            resolved_policy.maximum_eliminated_operator_relative_error,
        ),
        (
            "local_interface_carrier_balance_failed",
            final.maximum_local_carrier_residual,
            resolved_policy.maximum_local_carrier_normalized_residual,
        ),
        (
            "local_interface_gauss_balance_failed",
            final.maximum_local_gauss_residual,
            resolved_policy.maximum_local_gauss_normalized_residual,
        ),
        (
            "analytic_jacobian_check_failed",
            final.maximum_jacobian_error,
            resolved_policy.maximum_jacobian_column_relative_error,
        ),
        (
            "carrier_interface_ion_charge_balance_failed",
            final.maximum_charge_balance_error,
            resolved_policy.maximum_charge_balance_relative_error,
        ),
        (
            "all_face_total_current_closure_failed",
            final.maximum_face_spread,
            resolved_policy.maximum_all_face_current_spread_relative,
        ),
        (
            "two_sided_interface_total_current_closure_failed",
            final.maximum_interface_current_error,
            resolved_policy.maximum_two_sided_interface_total_current_relative_error,
        ),
        (
            "eliminated_qf_ion_operator_mismatch",
            final.maximum_operator_error,
            resolved_policy.maximum_eliminated_operator_relative_error,
        ),
        (
            "ion_inventory_drift_exceeds_limit",
            inventory_drift,
            resolved_policy.maximum_ion_inventory_relative_drift,
        ),
        (
            "current_decomposition_failed",
            decomposition_error,
            resolved_policy.maximum_current_decomposition_relative_error,
        ),
        (
            "time_refinement_state_not_converged",
            refinement_state,
            resolved_policy.maximum_refinement_state_change,
        ),
        (
            "time_refinement_current_not_converged",
            refinement_current,
            resolved_policy.maximum_refinement_current_relative_change,
        ),
    )
    for reason, value, limit in gates:
        if not math.isfinite(value) or value > limit:
            reasons.append(reason)
    dense_entries = system.dimension * system.dimension
    if final.maximum_nnz >= dense_entries:
        reasons.append("analytic_jacobian_not_sparse")
    certificate = InterfaceDefectIonTransientCertificate(
        dc_operating_point_certified=True,
        dark_reference_certified=True,
        microscopic_binding_certified=True,
        qss_embedding_relative_error=qss_embedding_error,
        maximum_scaled_nonlinear_residual=final.maximum_scaled_residual,
        maximum_poisson_residual_C_m2=final.maximum_poisson_residual,
        maximum_local_carrier_normalized_residual=(
            final.maximum_local_carrier_residual
        ),
        maximum_local_gauss_normalized_residual=final.maximum_local_gauss_residual,
        maximum_analytic_jacobian_column_relative_error=(final.maximum_jacobian_error),
        maximum_charge_balance_absolute_error_A_m2=(
            final.maximum_charge_balance_absolute_error
        ),
        maximum_charge_balance_relative_error=final.maximum_charge_balance_error,
        maximum_all_face_current_spread_relative=final.maximum_face_spread,
        maximum_two_sided_interface_total_current_relative_error=(
            final.maximum_interface_current_error
        ),
        maximum_eliminated_operator_relative_error=final.maximum_operator_error,
        eliminated_operator_components=tuple(
            sorted(system._maximum_eliminated_operator_components.items())
        ),
        maximum_positive_ion_inventory_relative_drift=positive_drift,
        maximum_negative_ion_inventory_relative_drift=negative_drift,
        maximum_ion_inventory_relative_drift=inventory_drift,
        maximum_site_occupancy_fraction=site_fraction,
        maximum_current_decomposition_relative_error=decomposition_error,
        maximum_refinement_state_change=refinement_state,
        maximum_refinement_current_relative_change=refinement_current,
        near_acceptance_nonmonotone_step_count=sum(
            level.near_acceptance_nonmonotone_step_count for level in levels
        ),
        analytic_jacobian_nnz=final.maximum_nnz,
        dense_jacobian_entries=dense_entries,
        sparse_linear_solver_used=True,
        clipping_used=False,
        certified=not reasons,
        reasons=tuple(reasons),
    )
    capture = np.asarray(
        [
            [item.tangent.balance.capture_flux_m2_s for item in state.local]
            for state in final.states
        ]
    )
    bulk_flux = np.asarray(
        [
            [item.tangent.balance.bulk_flux_m2_s for item in state.local]
            for state in final.states
        ]
    )
    result = InterfaceDefectIonTransientResult(
        times_s=times,
        voltage_V=voltage,
        electron_density_m3=np.asarray([state.n for state in final.states]),
        hole_density_m3=np.asarray([state.p for state in final.states]),
        interface_occupancy=np.asarray([state.occupancy for state in final.states]),
        interface_quasi_steady_occupancy=np.asarray(
            [
                [item.quasi_steady_occupancy for item in state.local]
                for state in final.states
            ]
        ),
        positive_ion_density_m3=np.asarray([state.positive for state in final.states]),
        negative_ion_density_m3=(
            None
            if final.states[0].negative is None
            else np.asarray([state.negative for state in final.states])
        ),
        electrostatic_potential_V=np.asarray([state.phi for state in final.states]),
        interface_trace_potential_V=np.asarray(
            [[item.trace_potential for item in state.local] for state in final.states]
        ),
        interface_trace_state_m3=np.asarray(
            [[item.state_m3 for item in state.local] for state in final.states]
        ),
        interface_sheet_charge_C_m2=np.asarray(
            [state.sheet_charge for state in final.states]
        ),
        electron_capture_flux_m2_s=capture[:, :, [0, 2]],
        hole_capture_flux_m2_s=capture[:, :, [1, 3]],
        electron_bulk_flux_m2_s=bulk_flux[:, :, [0, 2]],
        hole_bulk_flux_m2_s=bulk_flux[:, :, [1, 3]],
        carrier_conduction_current_faces_A_m2=np.asarray(
            [state.carrier_conduction for state in final.states]
        ),
        positive_ion_current_faces_A_m2=np.asarray(
            [state.positive_current for state in final.states]
        ),
        negative_ion_current_faces_A_m2=(
            None
            if final.states[0].negative_current is None
            else np.asarray([state.negative_current for state in final.states])
        ),
        conduction_current_faces_A_m2=np.asarray(
            [state.conduction for state in final.states]
        ),
        displacement_current_faces_A_m2=final.displacement,
        total_current_faces_A_m2=final.total_current,
        interface_conduction_current_A_m2=final.interface_conduction,
        interface_displacement_current_A_m2=final.interface_displacement,
        interface_total_current_A_m2=final.interface_total_current,
        integrated_free_interface_ion_charge_C_m2=final.integrated_charge,
        positive_ion_component_inventory_m2=positive_inventory,
        negative_ion_component_inventory_m2=negative_inventory,
        newton_iterations=final.iterations,
        ion_layout=ion_layout,
        dc_state=dc_state,
        dark_reference=dark_reference,
        policy=resolved_policy,
        certificate=certificate,
    )
    if require_certificate and not certificate.certified:
        raise InterfaceDefectIonTransientCertificationError(
            "interface defect/ion transient did not certify: "
            + ", ".join(certificate.reasons),
            result,
        )
    return result


__all__ = [
    "INTERFACE_DEFECT_ION_TRANSIENT_SCOPE",
    "INTERFACE_DEFECT_ION_TRANSIENT_VERSION",
    "InterfaceDefectIonTransientCertificate",
    "InterfaceDefectIonTransientCertificationError",
    "InterfaceDefectIonTransientError",
    "InterfaceDefectIonTransientPolicy",
    "InterfaceDefectIonTransientResult",
    "InterfaceIonDarkReference",
    "run_interface_defect_ion_device_transient",
]
