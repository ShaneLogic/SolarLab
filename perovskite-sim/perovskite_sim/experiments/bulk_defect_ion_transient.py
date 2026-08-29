"""Research-only transient coupling bulk-trap occupancy and mobile ions.

This D6-E3a adapter extends the sparse index-1 DAE used by the bulk-defect
transient with conservative positive/negative ion storage.  Carrier, trap,
ion, and electrostatic states are solved in one Newton system; no separately
computed transient traces are added after the solve.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
from scipy import sparse

from perovskite_sim.constants import Q
from perovskite_sim.experiments.bulk_defect_transient import (
    BulkDefectTransientError,
    BulkDefectTransientPolicy,
    _BulkTransientSystem,
    _integrate_trace,
    _readonly,
    _validate_trace_inputs,
)
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
from perovskite_sim.experiments.quasi_fermi_steady_state import (
    _QuasiFermiSystem,
    _build_qf_material,
    _require_material_defect_contract,
    _require_supported,
)
from perovskite_sim.models.defects import NEUTRAL, SINGLE_LEVEL
from perovskite_sim.models.device import DeviceStack, electrical_interface_defects
from perovskite_sim.physics.contacts import (
    ContactThermodynamicError,
    require_contact_thermodynamic_certificate,
)
from perovskite_sim.physics.defect_distributions import (
    DEFAULT_DEFECT_ENERGY_QUADRATURE_ORDER,
)
from perovskite_sim.physics.dynamic_defect_state import (
    DynamicBulkTrapEvaluation,
    DynamicBulkTrapLayout,
    bulk_trap_charge_density,
    compile_dynamic_bulk_trap_layout,
    quasi_steady_bulk_trap_occupancy,
)
from perovskite_sim.physics.ion_migration import ion_face_flux_jacobian
from perovskite_sim.solver.mol import MaterialArrays


BULK_DEFECT_ION_TRANSIENT_SCOPE = (
    "research_bulk_dynamic_defect_mobile_ion_device_transient_only"
)
BULK_DEFECT_ION_TRANSIENT_VERSION = "bulk-defect-ion-device-transient-v1"


class BulkDefectIonTransientError(BulkDefectTransientError):
    """The requested joint bulk-defect/mobile-ion transient failed closed."""


class BulkDefectIonTransientCertificationError(BulkDefectIonTransientError):
    """A finite joint trace failed one or more declared evidence gates."""

    def __init__(self, message: str, result: "BulkDefectIonTransientResult") -> None:
        self.result = result
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class BulkDefectIonTransientPolicy(BulkDefectTransientPolicy):
    """Nonlinear, DC, inventory, and refinement gates for D6-E3a."""

    ion_storage_atol_m3: float = 1.0
    maximum_scaled_nonlinear_residual: float = 5.0e-2
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
        BulkDefectTransientPolicy.__post_init__(self)
        positive = (
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
        for name in positive:
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
class BulkDefectIonTransientCertificate:
    dc_operating_point_certified: bool
    qss_embedding_relative_error: float
    maximum_scaled_nonlinear_residual: float
    maximum_poisson_residual_C_m2: float
    maximum_analytic_jacobian_column_relative_error: float
    maximum_charge_balance_absolute_error_A_m2: float
    maximum_charge_balance_relative_error: float
    maximum_all_face_current_spread_relative: float
    maximum_eliminated_operator_relative_error: float
    eliminated_operator_components: tuple[tuple[str, float], ...]
    maximum_positive_ion_inventory_relative_drift: float
    maximum_negative_ion_inventory_relative_drift: float
    maximum_ion_inventory_relative_drift: float
    maximum_site_occupancy_fraction: float
    maximum_current_decomposition_relative_error: float
    maximum_refinement_state_change: float
    maximum_refinement_current_relative_change: float
    analytic_jacobian_nnz: int
    dense_jacobian_entries: int
    sparse_linear_solver_used: bool
    clipping_used: bool
    certified: bool
    reasons: tuple[str, ...]
    scope: str = BULK_DEFECT_ION_TRANSIENT_SCOPE
    version: str = BULK_DEFECT_ION_TRANSIENT_VERSION


@dataclass(frozen=True, slots=True, eq=False)
class BulkDefectIonTransientResult:
    times_s: np.ndarray
    voltage_V: np.ndarray
    electron_density_m3: np.ndarray
    hole_density_m3: np.ndarray
    trap_occupancy: np.ndarray
    positive_ion_density_m3: np.ndarray
    negative_ion_density_m3: np.ndarray | None
    electrostatic_potential_V: np.ndarray
    trap_charge_density_C_m3: np.ndarray
    carrier_conduction_current_faces_A_m2: np.ndarray
    positive_ion_current_faces_A_m2: np.ndarray
    negative_ion_current_faces_A_m2: np.ndarray | None
    conduction_current_faces_A_m2: np.ndarray
    displacement_current_faces_A_m2: np.ndarray
    total_current_faces_A_m2: np.ndarray
    integrated_free_trap_ion_charge_C_m2: np.ndarray
    positive_ion_component_inventory_m2: np.ndarray
    negative_ion_component_inventory_m2: np.ndarray | None
    newton_iterations: np.ndarray
    trap_layout: DynamicBulkTrapLayout
    ion_layout: CombinedIonLayout
    dc_state: CombinedDCState
    policy: BulkDefectIonTransientPolicy
    certificate: BulkDefectIonTransientCertificate
    state_coordinate: str = "qf_log_trap_log_ion_potential"
    time_discretization: str = "backward_euler_index_1_dae"
    scope: str = BULK_DEFECT_ION_TRANSIENT_SCOPE
    version: str = BULK_DEFECT_ION_TRANSIENT_VERSION

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
            raise BulkDefectIonTransientError("times/voltage trace is invalid")
        object.__setattr__(self, "times_s", times)
        object.__setattr__(self, "voltage_V", voltage)
        points = times.size
        nodes = self.trap_layout.node_count
        faces = nodes - 1
        shapes = {
            "electron_density_m3": (points, nodes),
            "hole_density_m3": (points, nodes),
            "trap_occupancy": (points, self.trap_layout.size),
            "positive_ion_density_m3": (points, nodes),
            "electrostatic_potential_V": (points, nodes),
            "trap_charge_density_C_m3": (points, nodes),
            "carrier_conduction_current_faces_A_m2": (points, faces),
            "positive_ion_current_faces_A_m2": (points, faces),
            "conduction_current_faces_A_m2": (points, faces),
            "displacement_current_faces_A_m2": (points, faces),
            "total_current_faces_A_m2": (points, faces),
            "integrated_free_trap_ion_charge_C_m2": (points,),
            "positive_ion_component_inventory_m2": (
                points,
                len(self.ion_layout.positive_components),
            ),
        }
        for name, shape in shapes.items():
            values = _readonly(getattr(self, name))
            if values.shape != shape or not np.all(np.isfinite(values)):
                raise BulkDefectIonTransientError(f"{name} is invalid")
            object.__setattr__(self, name, values)
        optional = (
            (
                "negative_ion_density_m3",
                self.negative_ion_density_m3,
                (points, nodes),
            ),
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
                    raise BulkDefectIonTransientError(
                        f"{name} is required for an active negative-ion block"
                    )
                continue
            values = _readonly(value)
            if values.shape != shape or not np.all(np.isfinite(values)):
                raise BulkDefectIonTransientError(f"{name} is invalid")
            object.__setattr__(self, name, values)
        if np.any(self.electron_density_m3 <= 0.0) or np.any(
            self.hole_density_m3 <= 0.0
        ):
            raise BulkDefectIonTransientError("carrier densities must remain positive")
        if np.any((self.trap_occupancy <= 0.0) | (self.trap_occupancy >= 1.0)):
            raise BulkDefectIonTransientError(
                "trap occupancy must remain strictly inside (0, 1)"
            )
        positive_nodes = np.asarray(self.ion_layout.positive_nodes, dtype=int)
        if positive_nodes.size and np.any(
            self.positive_ion_density_m3[:, positive_nodes] <= 0.0
        ):
            raise BulkDefectIonTransientError(
                "active positive-ion densities must remain positive"
            )
        if self.ion_layout.negative_size:
            negative_nodes = np.asarray(self.ion_layout.negative_nodes, dtype=int)
            assert self.negative_ion_density_m3 is not None
            if np.any(self.negative_ion_density_m3[:, negative_nodes] <= 0.0):
                raise BulkDefectIonTransientError(
                    "active negative-ion densities must remain positive"
                )
        iterations = _readonly(self.newton_iterations, dtype=np.int64)
        if iterations.shape != (points,) or np.any(iterations < 0):
            raise BulkDefectIonTransientError("newton_iterations is invalid")
        object.__setattr__(self, "newton_iterations", iterations)
        if self.state_coordinate != "qf_log_trap_log_ion_potential":
            raise BulkDefectIonTransientError("unexpected state coordinate")
        if self.time_discretization != "backward_euler_index_1_dae":
            raise BulkDefectIonTransientError("unexpected time discretization")


@dataclass(slots=True)
class _BulkIonDeviceState:
    coordinate: np.ndarray
    dqfn: np.ndarray
    dqfp: np.ndarray
    n: np.ndarray
    p: np.ndarray
    occupancy: np.ndarray
    positive: np.ndarray
    negative: np.ndarray | None
    phi: np.ndarray
    dynamic: DynamicBulkTrapEvaluation
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
    trap_charge: np.ndarray
    poisson_residual: np.ndarray
    storage_jacobian: sparse.csr_matrix
    rate_jacobian: sparse.csr_matrix
    poisson_jacobian: sparse.csr_matrix


class _BulkIonTransientSystem(_BulkTransientSystem):
    def __init__(
        self,
        grid: np.ndarray,
        stack: DeviceStack,
        material: MaterialArrays,
        dc_state: CombinedDCState,
        qf_system: _QuasiFermiSystem,
        layout: DynamicBulkTrapLayout,
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
            layout,
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
        self.dimension = self.potential_slice.stop
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
                raise BulkDefectIonTransientError(
                    "positive-ion log coordinate overflowed or became non-positive"
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
                    raise BulkDefectIonTransientError(
                        "negative-ion log coordinate overflowed or became non-positive"
                    )
        self._site_fraction(positive, negative, reject=True)
        return positive, negative

    def _site_fraction(
        self,
        positive: np.ndarray,
        negative: np.ndarray | None,
        *,
        reject: bool,
    ) -> float:
        fractions: list[np.ndarray] = []
        shared = bool(
            self.material.ion_steric_diffusion_only
            and self.material.ion_steric_shared_site
            and negative is not None
        )
        if self.positive_nodes.size:
            total = positive + negative if shared and negative is not None else positive
            limit = np.asarray(self.material.P_lim_node, dtype=float)
            fractions.append(total[self.positive_nodes] / limit[self.positive_nodes])
        if self.negative_nodes.size:
            if negative is None:
                raise BulkDefectIonTransientError("negative-ion block is missing")
            total = positive + negative if shared else negative
            limit = np.asarray(self.material.P_lim_neg_node, dtype=float)
            fractions.append(total[self.negative_nodes] / limit[self.negative_nodes])
        maximum = max(
            (float(np.max(values)) for values in fractions if values.size),
            default=0.0,
        )
        if not math.isfinite(maximum) or maximum < 0.0:
            raise BulkDefectIonTransientError("ion site occupancy is non-finite")
        if reject and maximum >= self.site_occupancy_ceiling:
            raise BulkDefectIonTransientError(
                "ion coordinate reached the declared pre-clipping site ceiling"
            )
        return maximum

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
    ) -> tuple[
        sparse.csr_matrix,
        sparse.csr_matrix,
        sparse.csr_matrix,
        sparse.csr_matrix,
        sparse.csr_matrix,
        sparse.csr_matrix,
    ]:
        positive_density_jacobian = self._density_jacobian(
            positive,
            self.positive_nodes,
            self.positive_slice,
            self.dimension,
        )
        negative_density_jacobian = sparse.csr_matrix((self.node_count, self.dimension))
        if negative is not None:
            negative_density_jacobian = self._density_jacobian(
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

        def face_matrix(local, own, partner) -> sparse.csr_matrix:
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
        positive_active_faces = np.asarray(self.material.D_ion_face) > 0.0
        if np.any(positive_active_faces & ~positive_local.differentiable_faces):
            raise BulkDefectIonTransientError(
                "positive-ion flux reached a non-differentiable steric face"
            )
        positive_flux_jacobian = face_matrix(
            positive_local,
            positive_density_jacobian,
            negative_density_jacobian,
        )
        negative_flux_jacobian = sparse.csr_matrix(
            (self.node_count - 1, self.dimension)
        )
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
            negative_active_faces = np.asarray(self.material.D_ion_neg_face) > 0.0
            if np.any(negative_active_faces & ~negative_local.differentiable_faces):
                raise BulkDefectIonTransientError(
                    "negative-ion flux reached a non-differentiable steric face"
                )
            negative_flux_jacobian = face_matrix(
                negative_local,
                negative_density_jacobian,
                positive_density_jacobian,
            )
        inverse_volume = sparse.diags(1.0 / self.widths)
        positive_rate_jacobian = (
            inverse_volume @ self._divergence @ positive_flux_jacobian
        ).tocsr()
        negative_rate_jacobian = (
            inverse_volume @ self._divergence @ negative_flux_jacobian
        ).tocsr()
        return (
            positive_density_jacobian,
            negative_density_jacobian,
            positive_rate_jacobian,
            negative_rate_jacobian,
            positive_flux_jacobian,
            negative_flux_jacobian,
        )

    def evaluate(self, coordinate: np.ndarray, voltage: float) -> _BulkIonDeviceState:
        dqfn, dqfp, phi, n, p, occupancy = super()._coordinates(
            coordinate,
            voltage,
        )
        positive, negative = self._ion_coordinates(coordinate)
        source, dynamic = self._source_and_dynamic(n, p, phi, occupancy, voltage)
        current_n, current_p = self._currents(dqfn, dqfp, phi, n, p)
        divergence_n = self._divergence @ current_n
        divergence_p = self._divergence @ current_p
        rate_n = source[: self.node_count] + divergence_n / (Q * self.widths)
        rate_p = source[self.node_count :] - divergence_p / (Q * self.widths)
        positive_rate, negative_rate, positive_flux, negative_flux = _ion_fields(
            self.grid,
            self.material,
            positive,
            negative,
            phi,
        )
        storage_parts = [
            n[1:-1],
            p[1:-1],
            dynamic.occupied_storage_m3,
            positive[self.positive_nodes],
        ]
        rate_parts = [
            rate_n[1:-1],
            rate_p[1:-1],
            dynamic.trap_storage_rate_m3_s,
            positive_rate[self.positive_nodes],
        ]
        if self.negative_nodes.size:
            if negative is None or negative_rate is None:
                raise BulkDefectIonTransientError(
                    "active negative-ion storage/rate block is missing"
                )
            storage_parts.append(negative[self.negative_nodes])
            rate_parts.append(negative_rate[self.negative_nodes])
        storage = np.concatenate(storage_parts)
        rate = np.concatenate(rate_parts)
        trap_charge = bulk_trap_charge_density(occupancy, self.layout)
        rho, _ = self.system._bulk_space_charge_and_tangent(
            n,
            p,
            dynamic_bulk_charge_density_C_m3=trap_charge,
            positive_ion_density_m3=positive,
            negative_ion_density_m3=negative,
        )
        factor = self.material.poisson_factor
        poisson = self._poisson_laplacian @ phi + rho[1:-1] * factor.h_cell

        base_storage_jacobian, base_rate_jacobian, base_poisson_jacobian = (
            super()._jacobians(phi, n, p, occupancy)
        )
        (
            positive_density_jacobian,
            negative_density_jacobian,
            positive_rate_jacobian,
            negative_rate_jacobian,
            _positive_flux_jacobian,
            _negative_flux_jacobian,
        ) = self._ion_jacobians(phi, positive, negative)
        storage_blocks = [
            base_storage_jacobian,
            positive_density_jacobian[self.positive_nodes],
        ]
        rate_blocks = [
            base_rate_jacobian,
            positive_rate_jacobian[self.positive_nodes],
        ]
        if self.negative_nodes.size:
            storage_blocks.append(negative_density_jacobian[self.negative_nodes])
            rate_blocks.append(negative_rate_jacobian[self.negative_nodes])
        storage_jacobian = sparse.vstack(storage_blocks, format="csr")
        rate_jacobian = sparse.vstack(rate_blocks, format="csr")
        ion_charge_jacobian = Q * (
            positive_density_jacobian - negative_density_jacobian
        )
        poisson_jacobian = (
            base_poisson_jacobian
            + sparse.diags(factor.h_cell) @ ion_charge_jacobian[1:-1]
        ).tocsr()

        carrier_conduction = self.polarity * (current_n + current_p)
        positive_current = self.polarity * Q * positive_flux
        negative_current = (
            None if negative_flux is None else -self.polarity * Q * negative_flux
        )
        conduction = carrier_conduction + positive_current
        if negative_current is not None:
            conduction = conduction + negative_current
        arrays = (storage, rate, poisson, conduction)
        if any(not np.all(np.isfinite(value)) for value in arrays):
            raise BulkDefectIonTransientError(
                "joint transient operator produced a non-finite value"
            )
        return _BulkIonDeviceState(
            coordinate=np.asarray(coordinate, dtype=float).copy(),
            dqfn=dqfn,
            dqfp=dqfp,
            n=n,
            p=p,
            occupancy=np.asarray(occupancy),
            positive=positive,
            negative=negative,
            phi=phi,
            dynamic=dynamic,
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
            trap_charge=np.asarray(trap_charge),
            poisson_residual=np.asarray(poisson),
            storage_jacobian=storage_jacobian,
            rate_jacobian=rate_jacobian,
            poisson_jacobian=poisson_jacobian,
        )

    def storage_scale(
        self,
        previous_storage: np.ndarray,
        previous_state: _BulkIonDeviceState,
        dt: float,
        policy: BulkDefectIonTransientPolicy,
    ) -> np.ndarray:
        reference = np.r_[
            self.reference_n[1:-1],
            self.reference_p[1:-1],
            self.layout.population_density_m3 * self.reference_occupancy,
            self.reference_positive[self.positive_nodes],
            (
                np.empty(0)
                if self.reference_negative is None
                else self.reference_negative[self.negative_nodes]
            ),
        ]
        absolute = np.r_[
            np.full(2 * self.interior_count, policy.carrier_storage_atol_m3),
            np.full(self.trap_count, policy.trap_storage_atol_m3),
            np.full(self.ion_layout.size, policy.ion_storage_atol_m3),
        ]
        scale = absolute + policy.storage_relative_tolerance * np.maximum(
            np.abs(previous_storage),
            np.abs(reference),
        )
        rounding = 256.0 * np.finfo(float).eps
        for local, node in enumerate(range(1, self.node_count - 1)):
            electron_face_scale = max(
                abs(previous_state.current_n[node - 1]),
                abs(previous_state.current_n[node]),
            )
            hole_face_scale = max(
                abs(previous_state.current_p[node - 1]),
                abs(previous_state.current_p[node]),
            )
            scale[local] += (
                dt * rounding * electron_face_scale / (Q * self.widths[node])
            )
            scale[self.interior_count + local] += (
                dt * rounding * hole_face_scale / (Q * self.widths[node])
            )

        ion_offset = 2 * self.interior_count + self.trap_count
        for local, node in enumerate(self.positive_nodes):
            left = abs(previous_state.positive_flux[node - 1]) if node > 0 else 0.0
            right = (
                abs(previous_state.positive_flux[node])
                if node < self.node_count - 1
                else 0.0
            )
            scale[ion_offset + local] += (
                dt * rounding * max(left, right) / self.widths[node]
            )
        ion_offset += self.positive_nodes.size
        if self.negative_nodes.size:
            assert previous_state.negative_flux is not None
            for local, node in enumerate(self.negative_nodes):
                left = abs(previous_state.negative_flux[node - 1]) if node > 0 else 0.0
                right = (
                    abs(previous_state.negative_flux[node])
                    if node < self.node_count - 1
                    else 0.0
                )
                scale[ion_offset + local] += (
                    dt * rounding * max(left, right) / self.widths[node]
                )
        return scale

    def integrated_charge(self, state: _BulkIonDeviceState) -> float:
        carrier_trap = Q * (state.p - state.n) + state.trap_charge
        charge = float(np.sum(carrier_trap[1:-1] * self.widths[1:-1]))
        charge += Q * float(
            np.sum((state.positive - self.material.P_ion0) * self.widths)
        )
        if state.negative is not None:
            charge -= Q * float(
                np.sum((state.negative - self.material.P_ion0_neg) * self.widths)
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
        state: _BulkIonDeviceState,
        voltage: float,
    ) -> float:
        eliminated = self.system.evaluate_quasi_fermi_increments_defect_ion_combined(
            state.dqfn,
            state.dqfp,
            1.0 if self.illuminated else 0.0,
            positive_ion_density_m3=state.positive,
            negative_ion_density_m3=state.negative,
            dynamic_bulk_layout=self.layout,
            dynamic_bulk_occupancy=state.occupancy,
            dynamic_bulk_reference_n=self.reference_n,
            dynamic_bulk_reference_p=self.reference_p,
            dynamic_bulk_reference_occupancy=self.reference_occupancy,
            V_app=float(voltage),
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
            float(np.max(np.abs(state.positive_current))),
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
                raise BulkDefectIonTransientError(
                    "eliminated operator lost the negative-ion block"
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


def _refinement_changes(coarse, fine, system: _BulkIonTransientSystem):
    n_coarse = np.asarray([state.n for state in coarse.states])
    n_fine = np.asarray([state.n for state in fine.states])
    p_coarse = np.asarray([state.p for state in coarse.states])
    p_fine = np.asarray([state.p for state in fine.states])
    f_coarse = np.asarray([state.occupancy for state in coarse.states])
    f_fine = np.asarray([state.occupancy for state in fine.states])
    phi_coarse = np.asarray([state.phi for state in coarse.states])
    phi_fine = np.asarray([state.phi for state in fine.states])
    state_change = max(
        float(np.max(np.abs(np.log(n_coarse / n_fine)))),
        float(np.max(np.abs(np.log(p_coarse / p_fine)))),
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


def _inventory_trace(
    states: tuple[_BulkIonDeviceState, ...],
    layout: CombinedIonLayout,
    widths: np.ndarray,
    *,
    positive: bool,
) -> np.ndarray:
    components = layout.positive_components if positive else layout.negative_components
    values = []
    for state in states:
        density = state.positive if positive else state.negative
        if density is None:
            values.append(np.empty(0, dtype=float))
        else:
            values.append(_component_inventories(density, components, widths))
    return np.asarray(values, dtype=float)


def _inventory_drift(trace: np.ndarray, target: np.ndarray) -> float:
    if trace.shape[1] == 0:
        return 0.0
    return max(_maximum_component_inventory_error(values, target) for values in trace)


def _embedding_error(
    reference: object, dynamic: object, material: MaterialArrays
) -> float:
    current_scale = max(
        float(np.max(np.abs(reference.current_n + reference.current_p))),
        1.0,
    )
    rate_scale = max(
        current_scale / (Q * float(np.min(np.asarray(material.dx_cell)[1:-1]))),
        1.0,
    )

    def relative(left: np.ndarray, right: np.ndarray, floor: float) -> float:
        scale = max(
            float(np.max(np.abs(left), initial=0.0)),
            float(np.max(np.abs(right), initial=0.0)),
            floor,
        )
        return float(np.max(np.abs(left - right), initial=0.0)) / scale

    return max(
        relative(reference.y, dynamic.y, 1.0),
        relative(reference.phi, dynamic.phi, material.V_T_device),
        relative(reference.rate_n, dynamic.rate_n, rate_scale),
        relative(reference.rate_p, dynamic.rate_p, rate_scale),
        relative(reference.current_n, dynamic.current_n, current_scale),
        relative(reference.current_p, dynamic.current_p, current_scale),
    )


def run_bulk_defect_ion_device_transient(
    x: np.ndarray,
    stack: DeviceStack,
    times_s: object,
    voltage_V: object,
    *,
    illuminated: bool = False,
    mat: MaterialArrays | None = None,
    policy: BulkDefectIonTransientPolicy | None = None,
    require_certificate: bool = True,
) -> BulkDefectIonTransientResult:
    """Integrate one bulk-defect/mobile-ion transient in a shared sparse DAE."""
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
    resolved_policy = policy or BulkDefectIonTransientPolicy()
    if not isinstance(resolved_policy, BulkDefectIonTransientPolicy):
        raise TypeError("policy must be a BulkDefectIonTransientPolicy or None")
    if any(defect is not None for defect in electrical_interface_defects(stack)):
        raise BulkDefectIonTransientError(
            "D6-E3a excludes interface defects until the E3b two-sided DAE"
        )
    try:
        material = (
            _build_qf_material(
                grid,
                stack,
                defect_energy_quadrature_order=(DEFAULT_DEFECT_ENERGY_QUADRATURE_ORDER),
            )
            if mat is None
            else mat
        )
        _require_material_defect_contract(
            stack,
            material,
            defect_energy_quadrature_order=(DEFAULT_DEFECT_ENERGY_QUADRATURE_ORDER),
        )
        _require_supported(
            material,
            allow_charged_bulk_defects=True,
            allow_mobile_ions=True,
        )
        contact = require_contact_thermodynamic_certificate(stack, material)
        ion_layout = _build_ion_layout(material)
    except (DefectIonCombinedError, ContactThermodynamicError, ValueError) as exc:
        raise BulkDefectIonTransientError(
            f"bulk defect/ion transient material contract failed: {exc}"
        ) from exc
    model = material.monovalent_bulk_defects
    if model is None:
        raise BulkDefectIonTransientError(
            "bulk defect/ion transient requires an explicit bulk defect"
        )
    if model.has_distributed_species or model.has_spatial_profiles:
        raise BulkDefectIonTransientError(
            "D6-E3a supports only non-spatial single-level bulk defects"
        )
    if any(
        source.distribution.kind != SINGLE_LEVEL
        for region in model.regions
        for source in region.species
    ):
        raise BulkDefectIonTransientError(
            "D6-E3a supports only single-level bulk defect species"
        )
    if any(
        source.charge_transition == NEUTRAL
        for region in model.regions
        for source in region.species
    ):
        raise BulkDefectIonTransientError(
            "D6-E3a charge-coupled transient rejects neutral transitions"
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
    qf_system = _QuasiFermiSystem(
        grid,
        stack,
        material,
        float(voltage[0]),
        poisson_tolerance_V=1.0e-13,
        poisson_max_iterations=100,
    )
    context = _DCSolveContext(
        grid,
        stack,
        material,
        qf_system,
        ion_layout,
        1.0 if illuminated else 0.0,
        float(voltage[0]),
        positive_target,
        negative_target,
        contact,
    )
    try:
        dc_state, qss_dc = _solve_combined_dc(
            context,
            maximum_normalized_residual=(
                resolved_policy.maximum_dc_normalized_residual
            ),
            maximum_continuity_bound_A_m2=(
                resolved_policy.maximum_dc_continuity_bound_A_m2
            ),
            maximum_ionic_face_current_A_m2=(
                resolved_policy.maximum_dc_ionic_face_current_A_m2
            ),
            maximum_inventory_error=resolved_policy.maximum_dc_inventory_error,
            maximum_poisson_residual=(resolved_policy.maximum_dc_poisson_residual),
            maximum_face_current_spread_A_m2=(
                resolved_policy.maximum_dc_face_current_spread_A_m2
            ),
            max_nfev=resolved_policy.dc_max_nfev,
        )
    except (DefectIonCombinedError, ValueError) as exc:
        raise BulkDefectIonTransientError(
            f"joint defect/ion DC solve failed: {exc}"
        ) from exc
    if not dc_state.certificate.certified:
        raise BulkDefectIonTransientError(
            "joint defect/ion DC state is not certified: "
            + ", ".join(dc_state.certificate.reasons)
        )

    dynamic_mask = np.ones(grid.size, dtype=bool)
    dynamic_mask[[0, -1]] = False
    trap_layout = compile_dynamic_bulk_trap_layout(
        model,
        dynamic_node_mask=dynamic_mask,
    )
    occupancy = quasi_steady_bulk_trap_occupancy(
        np.asarray(dc_state.electron_density_m3),
        np.asarray(dc_state.hole_density_m3),
        trap_layout,
    )
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
        dynamic_bulk_layout=trap_layout,
        dynamic_bulk_occupancy=occupancy,
        dynamic_bulk_reference_n=np.asarray(dc_state.electron_density_m3),
        dynamic_bulk_reference_p=np.asarray(dc_state.hole_density_m3),
        dynamic_bulk_reference_occupancy=occupancy,
        V_app=float(voltage[0]),
    )
    qss_embedding_error = _embedding_error(qss_dc, dynamic_dc, material)
    system = _BulkIonTransientSystem(
        grid,
        stack,
        material,
        dc_state,
        qf_system,
        trap_layout,
        occupancy,
        dynamic_dc,
        ion_layout,
        voltage=float(voltage[0]),
        illuminated=bool(illuminated),
        site_occupancy_ceiling=resolved_policy.site_occupancy_ceiling,
    )
    try:
        levels = tuple(
            _integrate_trace(
                system,
                times,
                voltage,
                substeps,
                resolved_policy,
            )
            for substeps in resolved_policy.refinement_substeps
        )
    except BulkDefectTransientError as exc:
        raise BulkDefectIonTransientError(
            f"joint defect/ion transient solve failed: {exc}"
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
    positive_inventory_drift = _inventory_drift(
        positive_inventory,
        system.positive_targets,
    )
    negative_inventory_drift = (
        0.0
        if negative_inventory is None
        else _inventory_drift(negative_inventory, system.negative_targets)
    )
    inventory_drift = max(positive_inventory_drift, negative_inventory_drift)
    maximum_site_fraction = max(
        system._site_fraction(state.positive, state.negative, reject=False)
        for state in final.states
    )
    decomposition_error = 0.0
    for state in final.states:
        expected = state.carrier_conduction + state.positive_current
        if state.negative_current is not None:
            expected = expected + state.negative_current
        scale = max(
            float(np.max(np.abs(expected), initial=0.0)),
            float(np.max(np.abs(state.conduction), initial=0.0)),
            1.0,
        )
        decomposition_error = max(
            decomposition_error,
            float(np.max(np.abs(expected - state.conduction), initial=0.0)) / scale,
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
            "analytic_jacobian_check_failed",
            final.maximum_jacobian_error,
            resolved_policy.maximum_jacobian_column_relative_error,
        ),
        (
            "carrier_trap_ion_charge_balance_failed",
            final.maximum_charge_balance_error,
            resolved_policy.maximum_charge_balance_relative_error,
        ),
        (
            "all_face_total_current_closure_failed",
            final.maximum_face_spread,
            resolved_policy.maximum_all_face_current_spread_relative,
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
    certificate = BulkDefectIonTransientCertificate(
        dc_operating_point_certified=True,
        qss_embedding_relative_error=qss_embedding_error,
        maximum_scaled_nonlinear_residual=final.maximum_scaled_residual,
        maximum_poisson_residual_C_m2=final.maximum_poisson_residual,
        maximum_analytic_jacobian_column_relative_error=(final.maximum_jacobian_error),
        maximum_charge_balance_absolute_error_A_m2=(
            final.maximum_charge_balance_absolute_error
        ),
        maximum_charge_balance_relative_error=final.maximum_charge_balance_error,
        maximum_all_face_current_spread_relative=final.maximum_face_spread,
        maximum_eliminated_operator_relative_error=final.maximum_operator_error,
        eliminated_operator_components=tuple(
            sorted(system._maximum_eliminated_operator_components.items())
        ),
        maximum_positive_ion_inventory_relative_drift=positive_inventory_drift,
        maximum_negative_ion_inventory_relative_drift=negative_inventory_drift,
        maximum_ion_inventory_relative_drift=inventory_drift,
        maximum_site_occupancy_fraction=maximum_site_fraction,
        maximum_current_decomposition_relative_error=decomposition_error,
        maximum_refinement_state_change=refinement_state,
        maximum_refinement_current_relative_change=refinement_current,
        analytic_jacobian_nnz=final.maximum_nnz,
        dense_jacobian_entries=dense_entries,
        sparse_linear_solver_used=True,
        clipping_used=False,
        certified=not reasons,
        reasons=tuple(reasons),
    )
    result = BulkDefectIonTransientResult(
        times_s=times,
        voltage_V=voltage,
        electron_density_m3=np.asarray([state.n for state in final.states]),
        hole_density_m3=np.asarray([state.p for state in final.states]),
        trap_occupancy=np.asarray([state.occupancy for state in final.states]),
        positive_ion_density_m3=np.asarray([state.positive for state in final.states]),
        negative_ion_density_m3=(
            None
            if final.states[0].negative is None
            else np.asarray([state.negative for state in final.states])
        ),
        electrostatic_potential_V=np.asarray([state.phi for state in final.states]),
        trap_charge_density_C_m3=np.asarray(
            [state.trap_charge for state in final.states]
        ),
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
        integrated_free_trap_ion_charge_C_m2=final.integrated_charge,
        positive_ion_component_inventory_m2=positive_inventory,
        negative_ion_component_inventory_m2=negative_inventory,
        newton_iterations=final.iterations,
        trap_layout=trap_layout,
        ion_layout=ion_layout,
        dc_state=dc_state,
        policy=resolved_policy,
        certificate=certificate,
    )
    if require_certificate and not certificate.certified:
        raise BulkDefectIonTransientCertificationError(
            "bulk defect/ion device transient did not certify: "
            + ", ".join(certificate.reasons),
            result,
        )
    return result


__all__ = [
    "BULK_DEFECT_ION_TRANSIENT_SCOPE",
    "BULK_DEFECT_ION_TRANSIENT_VERSION",
    "BulkDefectIonTransientCertificate",
    "BulkDefectIonTransientCertificationError",
    "BulkDefectIonTransientError",
    "BulkDefectIonTransientPolicy",
    "BulkDefectIonTransientResult",
    "run_bulk_defect_ion_device_transient",
]
