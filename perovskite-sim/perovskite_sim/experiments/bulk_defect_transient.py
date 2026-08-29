"""Research-only device transient with explicit monovalent bulk occupancy.

The legacy method-of-lines state does not contain a shared trap occupancy and
therefore cannot carry charged explicit defects through Poisson consistently.
This module keeps a separate index-1 DAE in quasi-Fermi, trap-logit, and
electrostatic-potential coordinates.  Carrier and occupied-trap populations
are the differential storage rows; Poisson is retained as an algebraic row so
the exact Jacobian remains sparse.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
import warnings

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import MatrixRankWarning, spsolve

from perovskite_sim.constants import EPS_0, Q
from perovskite_sim.discretization.fe_operators import (
    bernoulli,
    sg_fluxes_n_jacobian,
    sg_fluxes_p_jacobian,
)
from perovskite_sim.experiments.quasi_fermi_impedance import (
    _contact_quasi_fermi_increments,
)
from perovskite_sim.experiments.quasi_fermi_steady_state import (
    QuasiFermiSteadyStateError,
    QuasiFermiSteadyStateResult,
    _QuasiFermiSystem,
    _require_material_defect_contract,
    _require_supported,
    solve_quasi_fermi_steady_state,
)
from perovskite_sim.models.defects import (
    NEUTRAL,
    SINGLE_LEVEL,
    ExplicitDefectCapabilityError,
)
from perovskite_sim.models.device import DeviceStack
from perovskite_sim.physics.defect_distributions import (
    DEFAULT_DEFECT_ENERGY_QUADRATURE_ORDER,
)
from perovskite_sim.physics.dynamic_defect_state import (
    DynamicBulkTrapEvaluation,
    DynamicBulkTrapLayout,
    bulk_trap_charge_density,
    compile_dynamic_bulk_trap_layout,
    evaluate_dynamic_bulk_traps_about_qss,
    occupancy_from_logit_increment,
    occupancy_logit,
    quasi_steady_bulk_trap_occupancy,
)
from perovskite_sim.physics.recombination import (
    srh_recombination,
    srh_recombination_derivatives,
    total_recombination_derivatives,
)
from perovskite_sim.solver.mol import (
    EXPLICIT_DEFECT_CHARGE_QF_DC,
    MaterialArrays,
    _harmonic_face_average,
    assemble_rhs,
    build_material_arrays,
    poisson_right_boundary,
)


BULK_DEFECT_TRANSIENT_SCOPE = "research_bulk_dynamic_defect_device_transient_only"
BULK_DEFECT_TRANSIENT_VERSION = "bulk-dynamic-defect-device-transient-v1"


class BulkDefectTransientError(RuntimeError):
    """The requested dynamic-defect device transient failed closed."""


class BulkDefectTransientCertificationError(BulkDefectTransientError):
    """A finite trace failed one or more declared evidence gates."""

    def __init__(self, message: str, result: "BulkDefectTransientResult") -> None:
        self.result = result
        super().__init__(message)


def _readonly(value: object, *, dtype: object = float) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _finite_positive(name: str, value: object) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a real number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real number") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _validate_refinement_substeps(value: object) -> tuple[int, ...]:
    try:
        result = tuple(int(item) for item in value)
    except TypeError as exc:
        raise TypeError("refinement_substeps must be iterable") from exc
    if (
        len(result) < 2
        or any(item <= 0 for item in result)
        or any(right <= left for left, right in zip(result, result[1:]))
        or any(right % left != 0 for left, right in zip(result, result[1:]))
    ):
        raise ValueError(
            "refinement_substeps must contain at least two increasing nested levels"
        )
    return result


@dataclass(frozen=True, slots=True)
class BulkDefectTransientPolicy:
    """Nonlinear, refinement, and conservation gates for the D6-E1 lane."""

    storage_relative_tolerance: float = 1.0e-9
    carrier_storage_atol_m3: float = 1.0
    trap_storage_atol_m3: float = 1.0
    poisson_relative_tolerance: float = 1.0e-10
    poisson_atol_C_m2: float = 1.0e-18
    maximum_scaled_nonlinear_residual: float = 5.0e-1
    maximum_newton_iterations: int = 30
    maximum_line_search_steps: int = 18
    jacobian_check_step: float = 1.0e-6
    maximum_jacobian_column_relative_error: float = 2.0e-4
    refinement_substeps: tuple[int, ...] = (1, 2, 4)
    maximum_refinement_state_change: float = 2.0e-2
    maximum_refinement_current_relative_change: float = 5.0e-2
    maximum_charge_balance_relative_error: float = 1.0e-10
    maximum_all_face_current_spread_relative: float = 1.0e-6
    maximum_eliminated_operator_relative_error: float = 2.0e-7

    def __post_init__(self) -> None:
        for name in (
            "storage_relative_tolerance",
            "carrier_storage_atol_m3",
            "trap_storage_atol_m3",
            "poisson_relative_tolerance",
            "poisson_atol_C_m2",
            "maximum_scaled_nonlinear_residual",
            "jacobian_check_step",
            "maximum_jacobian_column_relative_error",
            "maximum_refinement_state_change",
            "maximum_refinement_current_relative_change",
            "maximum_charge_balance_relative_error",
            "maximum_all_face_current_spread_relative",
            "maximum_eliminated_operator_relative_error",
        ):
            object.__setattr__(self, name, _finite_positive(name, getattr(self, name)))
        for name in ("maximum_newton_iterations", "maximum_line_search_steps"):
            value = int(getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "refinement_substeps",
            _validate_refinement_substeps(self.refinement_substeps),
        )


@dataclass(frozen=True, slots=True)
class BulkDefectTransientCertificate:
    """Numerical and physical evidence for one returned transient trace."""

    dc_operating_point_certified: bool
    qss_embedding_relative_error: float
    maximum_scaled_nonlinear_residual: float
    maximum_poisson_residual_C_m2: float
    maximum_analytic_jacobian_column_relative_error: float
    maximum_charge_balance_absolute_error_A_m2: float
    maximum_charge_balance_relative_error: float
    maximum_all_face_current_spread_relative: float
    maximum_eliminated_operator_relative_error: float
    maximum_refinement_state_change: float
    maximum_refinement_current_relative_change: float
    analytic_jacobian_nnz: int
    dense_jacobian_entries: int
    sparse_linear_solver_used: bool
    clipping_used: bool
    certified: bool
    reasons: tuple[str, ...]
    scope: str = BULK_DEFECT_TRANSIENT_SCOPE
    version: str = BULK_DEFECT_TRANSIENT_VERSION


@dataclass(frozen=True, slots=True, eq=False)
class BulkDefectTransientResult:
    """Immutable physical trace from the finest nested time grid."""

    times_s: np.ndarray
    voltage_V: np.ndarray
    electron_density_m3: np.ndarray
    hole_density_m3: np.ndarray
    trap_occupancy: np.ndarray
    electrostatic_potential_V: np.ndarray
    trap_charge_density_C_m3: np.ndarray
    conduction_current_faces_A_m2: np.ndarray
    displacement_current_faces_A_m2: np.ndarray
    total_current_faces_A_m2: np.ndarray
    integrated_free_and_trap_charge_C_m2: np.ndarray
    newton_iterations: np.ndarray
    layout: DynamicBulkTrapLayout
    dc_state: QuasiFermiSteadyStateResult
    policy: BulkDefectTransientPolicy
    certificate: BulkDefectTransientCertificate
    state_coordinate: str = "qf_log_trap_logit_potential"
    time_discretization: str = "backward_euler_index_1_dae"
    scope: str = BULK_DEFECT_TRANSIENT_SCOPE
    version: str = BULK_DEFECT_TRANSIENT_VERSION

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
            raise BulkDefectTransientError("times/voltage trace is invalid")
        object.__setattr__(self, "times_s", times)
        object.__setattr__(self, "voltage_V", voltage)
        point_count = times.size
        node_count = self.layout.node_count
        shapes = {
            "electron_density_m3": (point_count, node_count),
            "hole_density_m3": (point_count, node_count),
            "trap_occupancy": (point_count, self.layout.size),
            "electrostatic_potential_V": (point_count, node_count),
            "trap_charge_density_C_m3": (point_count, node_count),
            "conduction_current_faces_A_m2": (point_count, node_count - 1),
            "displacement_current_faces_A_m2": (point_count, node_count - 1),
            "total_current_faces_A_m2": (point_count, node_count - 1),
            "integrated_free_and_trap_charge_C_m2": (point_count,),
        }
        for name, shape in shapes.items():
            values = _readonly(getattr(self, name))
            if values.shape != shape or not np.all(np.isfinite(values)):
                raise BulkDefectTransientError(f"{name} is invalid")
            object.__setattr__(self, name, values)
        if np.any(self.electron_density_m3 <= 0.0) or np.any(
            self.hole_density_m3 <= 0.0
        ):
            raise BulkDefectTransientError("carrier densities must remain positive")
        if np.any((self.trap_occupancy <= 0.0) | (self.trap_occupancy >= 1.0)):
            raise BulkDefectTransientError("trap occupancy must remain inside (0, 1)")
        iterations = _readonly(self.newton_iterations, dtype=np.int64)
        if iterations.shape != (point_count,) or np.any(iterations < 0):
            raise BulkDefectTransientError("newton_iterations is invalid")
        object.__setattr__(self, "newton_iterations", iterations)
        if self.state_coordinate != "qf_log_trap_logit_potential":
            raise BulkDefectTransientError("unexpected device transient coordinate")
        if self.time_discretization != "backward_euler_index_1_dae":
            raise BulkDefectTransientError("unexpected time discretization")


@dataclass(slots=True)
class _DeviceState:
    coordinate: np.ndarray
    dqfn: np.ndarray
    dqfp: np.ndarray
    n: np.ndarray
    p: np.ndarray
    occupancy: np.ndarray
    phi: np.ndarray
    dynamic: DynamicBulkTrapEvaluation
    storage: np.ndarray
    rate: np.ndarray
    current_n: np.ndarray
    current_p: np.ndarray
    conduction: np.ndarray
    trap_charge: np.ndarray
    poisson_residual: np.ndarray
    storage_jacobian: sparse.csr_matrix
    rate_jacobian: sparse.csr_matrix
    poisson_jacobian: sparse.csr_matrix


@dataclass(slots=True)
class _Trace:
    times: np.ndarray
    voltage: np.ndarray
    coordinates: np.ndarray
    states: tuple[_DeviceState, ...]
    displacement: np.ndarray
    total_current: np.ndarray
    integrated_charge: np.ndarray
    iterations: np.ndarray
    maximum_scaled_residual: float
    maximum_poisson_residual: float
    maximum_jacobian_error: float
    maximum_charge_balance_absolute_error: float
    maximum_charge_balance_error: float
    maximum_face_spread: float
    maximum_operator_error: float
    maximum_nnz: int


class _BulkTransientSystem:
    def __init__(
        self,
        grid: np.ndarray,
        stack: DeviceStack,
        material: MaterialArrays,
        dc_state: QuasiFermiSteadyStateResult,
        layout: DynamicBulkTrapLayout,
        occupancy_reference: np.ndarray,
        dynamic_dc,
        *,
        illuminated: bool,
    ) -> None:
        self.grid = grid
        self.stack = stack
        self.material = material
        self.dc_state = dc_state
        self.layout = layout
        self.illuminated = bool(illuminated)
        self.node_count = grid.size
        self.interior_count = grid.size - 2
        self.trap_count = layout.size
        self.dimension = 3 * self.interior_count + self.trap_count
        self.electron_slice = slice(0, self.interior_count)
        self.hole_slice = slice(self.interior_count, 2 * self.interior_count)
        self.trap_slice = slice(
            2 * self.interior_count,
            2 * self.interior_count + self.trap_count,
        )
        self.potential_slice = slice(
            self.trap_slice.stop,
            self.trap_slice.stop + self.interior_count,
        )
        self.system = _QuasiFermiSystem(
            grid,
            stack,
            material,
            float(dc_state.V_app),
            poisson_tolerance_V=1.0e-13,
            poisson_max_iterations=100,
        )
        self.qfn_reference = np.asarray(
            dc_state.electron_quasi_fermi_reference_V,
            dtype=float,
        )
        self.qfp_reference = np.asarray(
            dc_state.hole_quasi_fermi_reference_V,
            dtype=float,
        )
        self.dqfn_dc = np.asarray(
            dc_state.electron_quasi_fermi_increment_V,
            dtype=float,
        )
        self.dqfp_dc = np.asarray(
            dc_state.hole_quasi_fermi_increment_V,
            dtype=float,
        )
        if not (
            np.array_equal(self.qfn_reference, self.system.qfn0)
            and np.array_equal(self.qfp_reference, self.system.qfp0)
        ):
            raise BulkDefectTransientError(
                "DC QF reference does not match the transient operator"
            )
        self.reference_n = np.asarray(dynamic_dc.y[: self.node_count], dtype=float)
        self.reference_p = np.asarray(
            dynamic_dc.y[self.node_count : 2 * self.node_count],
            dtype=float,
        )
        self.reference_phi = np.asarray(dynamic_dc.phi, dtype=float)
        self.reference_occupancy = np.asarray(occupancy_reference, dtype=float)
        self.reference_logit = np.asarray(
            occupancy_logit(occupancy_reference, layout),
            dtype=float,
        )
        self.thermal_voltage = float(material.V_T_device)
        self.polarity = float(material.junction_polarity)
        self.eps_face = EPS_0 * _harmonic_face_average(material.eps_r)
        self.widths = np.asarray(material.dx_cell, dtype=float)
        self.face_weights = np.diff(grid) / float(grid[-1] - grid[0])
        self._divergence = self._build_divergence_matrix()
        self._poisson_laplacian = self._build_poisson_laplacian()

    def _build_divergence_matrix(self) -> sparse.csr_matrix:
        rows: list[int] = []
        columns: list[int] = []
        values: list[float] = []
        for face in range(self.node_count - 1):
            rows.extend((face, face + 1))
            columns.extend((face, face))
            values.extend((1.0, -1.0))
        # QF uses diff([0, J, 0]): +J on the left node and -J on right.
        return sparse.csr_matrix(
            (values, (rows, columns)),
            shape=(self.node_count, self.node_count - 1),
        )

    def _build_poisson_laplacian(self) -> sparse.csr_matrix:
        factor = self.material.poisson_factor
        matrix = sparse.lil_matrix((self.interior_count, self.node_count))
        for row, node in enumerate(range(1, self.node_count - 1)):
            matrix[row, node - 1] = factor.C[node - 1]
            matrix[row, node] = -(factor.C[node - 1] + factor.C[node])
            matrix[row, node + 1] = factor.C[node]
        return matrix.tocsr()

    def initial_coordinate(self) -> np.ndarray:
        return np.zeros(self.dimension, dtype=float)

    def _coordinates(
        self,
        coordinate: np.ndarray,
        voltage: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        values = np.asarray(coordinate, dtype=float)
        if values.shape != (self.dimension,) or not np.all(np.isfinite(values)):
            raise BulkDefectTransientError(
                f"device coordinate must be a finite vector of length {self.dimension}"
            )
        dqfn = self.dqfn_dc.copy()
        dqfp = self.dqfp_dc.copy()
        dqfn[1:-1] += self.thermal_voltage * values[self.electron_slice]
        dqfp[1:-1] += self.thermal_voltage * values[self.hole_slice]
        _contact_quasi_fermi_increments(
            dqfn,
            dqfp,
            self.qfn_reference,
            self.qfp_reference,
            self.material,
            float(voltage),
        )
        phi = self.reference_phi.copy()
        phi[0] = 0.0
        phi[-1] = poisson_right_boundary(self.material, float(voltage))
        phi[1:-1] += self.thermal_voltage * values[self.potential_slice]
        dphi = phi - self.system.phi0
        log_n = self.system.log_n0 + (dqfn + dphi) / self.thermal_voltage
        log_p = self.system.log_p0 + (dqfp - dphi) / self.thermal_voltage
        limit = math.log(np.finfo(float).max)
        if np.any(log_n > limit) or np.any(log_p > limit):
            raise BulkDefectTransientError("carrier coordinate overflow")
        n = np.exp(log_n)
        p = np.exp(log_p)
        if (
            not np.all(np.isfinite(n))
            or not np.all(np.isfinite(p))
            or np.any(n <= 0.0)
            or np.any(p <= 0.0)
        ):
            raise BulkDefectTransientError(
                "carrier coordinate produced non-positive or non-finite density"
            )
        occupancy = occupancy_from_logit_increment(
            self.reference_logit,
            values[self.trap_slice],
            self.layout,
        )
        return dqfn, dqfp, phi, n, p, occupancy

    def _source_and_dynamic(
        self,
        n: np.ndarray,
        p: np.ndarray,
        phi: np.ndarray,
        occupancy: np.ndarray,
        voltage: float,
    ) -> tuple[np.ndarray, DynamicBulkTrapEvaluation]:
        y = self.system.base.copy()
        y[: self.node_count] = n
        y[self.node_count : 2 * self.node_count] = p
        source = assemble_rhs(
            0.0,
            y,
            self.grid,
            self.stack,
            self.system.dynamic_bulk_source_mat,
            illuminated=False,
            V_app=float(voltage),
            phi_frozen=phi,
        )[: 2 * self.node_count]
        dynamic = evaluate_dynamic_bulk_traps_about_qss(
            n,
            p,
            occupancy,
            self.layout,
            reference_electron_density_m3=self.reference_n,
            reference_hole_density_m3=self.reference_p,
            reference_occupancy=self.reference_occupancy,
        )
        legacy_srh = srh_recombination(
            n,
            p,
            self.material.ni_sq,
            self.material.tau_n,
            self.material.tau_p,
            self.material.n1,
            self.material.p1,
        )
        dynamic_nodes = np.zeros(self.node_count, dtype=bool)
        dynamic_nodes[self.layout.device_node_indices] = True
        source_n = source[: self.node_count]
        source_p = source[self.node_count :]
        source_n[dynamic_nodes] += legacy_srh[dynamic_nodes]
        source_p[dynamic_nodes] += legacy_srh[dynamic_nodes]
        source_n -= dynamic.total_electron_capture_rate_m3_s
        source_p -= dynamic.total_hole_capture_rate_m3_s
        source += (1.0 if self.illuminated else 0.0) * self.system.generation
        return source, dynamic

    def _currents(
        self,
        dqfn: np.ndarray,
        dqfp: np.ndarray,
        phi: np.ndarray,
        n: np.ndarray,
        p: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        psi_n = phi + self.material.chi
        psi_p = phi + self.material.chi + self.material.Eg
        xi_n = np.diff(psi_n) / self.thermal_voltage
        xi_p = np.diff(psi_p) / self.thermal_voltage
        delta_n = self.system.reference_edge_drop_n + (
            np.diff(dqfn) / self.thermal_voltage
        )
        delta_p = -(
            self.system.reference_edge_drop_p + np.diff(dqfp) / self.thermal_voltage
        )
        current_n = (
            Q
            * self.material.D_n_face
            / np.diff(self.grid)
            * self.system._stable_difference(
                bernoulli(xi_n) * n[1:],
                bernoulli(-xi_n) * n[:-1],
                delta_n,
            )
        )
        current_p = (
            Q
            * self.material.D_p_face
            / np.diff(self.grid)
            * self.system._stable_difference(
                bernoulli(xi_p) * p[:-1],
                bernoulli(-xi_p) * p[1:],
                delta_p,
            )
        )
        return current_n, current_p

    def evaluate(self, coordinate: np.ndarray, voltage: float) -> _DeviceState:
        dqfn, dqfp, phi, n, p, occupancy = self._coordinates(coordinate, voltage)
        source, dynamic = self._source_and_dynamic(
            n,
            p,
            phi,
            occupancy,
            voltage,
        )
        current_n, current_p = self._currents(dqfn, dqfp, phi, n, p)
        divergence_n = self._divergence @ current_n
        divergence_p = self._divergence @ current_p
        rate_n = source[: self.node_count] + divergence_n / (Q * self.widths)
        rate_p = source[self.node_count :] - divergence_p / (Q * self.widths)
        storage = np.r_[
            n[1:-1],
            p[1:-1],
            dynamic.occupied_storage_m3,
        ]
        rate = np.r_[
            rate_n[1:-1],
            rate_p[1:-1],
            dynamic.trap_storage_rate_m3_s,
        ]
        trap_charge = bulk_trap_charge_density(occupancy, self.layout)
        rho, _ = self.system._bulk_space_charge_and_tangent(
            n,
            p,
            dynamic_bulk_charge_density_C_m3=trap_charge,
        )
        factor = self.material.poisson_factor
        poisson = self._poisson_laplacian @ phi + rho[1:-1] * factor.h_cell
        storage_jacobian, rate_jacobian, poisson_jacobian = self._jacobians(
            phi,
            n,
            p,
            occupancy,
        )
        conduction = self.polarity * (current_n + current_p)
        arrays = (storage, rate, poisson, conduction)
        if any(not np.all(np.isfinite(value)) for value in arrays):
            raise BulkDefectTransientError(
                "device transient operator produced a non-finite value"
            )
        return _DeviceState(
            coordinate=np.asarray(coordinate, dtype=float).copy(),
            dqfn=dqfn,
            dqfp=dqfp,
            n=n,
            p=p,
            occupancy=np.asarray(occupancy),
            phi=phi,
            dynamic=dynamic,
            storage=storage,
            rate=rate,
            current_n=current_n,
            current_p=current_p,
            conduction=conduction,
            trap_charge=np.asarray(trap_charge),
            poisson_residual=np.asarray(poisson),
            storage_jacobian=storage_jacobian,
            rate_jacobian=rate_jacobian,
            poisson_jacobian=poisson_jacobian,
        )

    def _jacobians(
        self,
        phi: np.ndarray,
        n: np.ndarray,
        p: np.ndarray,
        occupancy: np.ndarray,
    ) -> tuple[sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix]:
        dimension = self.dimension
        n_jacobian = sparse.lil_matrix((self.node_count, dimension))
        p_jacobian = sparse.lil_matrix((self.node_count, dimension))
        phi_jacobian = sparse.lil_matrix((self.node_count, dimension))
        for local, node in enumerate(range(1, self.node_count - 1)):
            n_jacobian[node, self.electron_slice.start + local] = n[node]
            n_jacobian[node, self.potential_slice.start + local] = n[node]
            p_jacobian[node, self.hole_slice.start + local] = p[node]
            p_jacobian[node, self.potential_slice.start + local] = -p[node]
            phi_jacobian[
                node,
                self.potential_slice.start + local,
            ] = self.thermal_voltage
        n_jacobian = n_jacobian.tocsr()
        p_jacobian = p_jacobian.tocsr()
        phi_jacobian = phi_jacobian.tocsr()

        occupancy_tangent = occupancy * (1.0 - occupancy)
        occupancy_jacobian = sparse.lil_matrix((self.trap_count, dimension))
        for state_index, tangent in enumerate(occupancy_tangent):
            occupancy_jacobian[
                state_index,
                self.trap_slice.start + state_index,
            ] = tangent
        occupancy_jacobian = occupancy_jacobian.tocsr()
        trap_storage_jacobian = (
            sparse.diags(self.layout.population_density_m3) @ occupancy_jacobian
        )
        storage_jacobian = sparse.vstack(
            (
                n_jacobian[1:-1],
                p_jacobian[1:-1],
                trap_storage_jacobian,
            ),
            format="csr",
        )

        recombination = total_recombination_derivatives(
            n,
            p,
            self.material.ni_sq,
            self.material.tau_n,
            self.material.tau_p,
            self.material.n1,
            self.material.p1,
            self.material.B_rad,
            self.material.C_n,
            self.material.C_p,
            neutral_bulk_defects=self.system.dynamic_bulk_source_mat.neutral_bulk_defects,
        )
        legacy = srh_recombination_derivatives(
            n,
            p,
            self.material.ni_sq,
            self.material.tau_n,
            self.material.tau_p,
            self.material.n1,
            self.material.p1,
        )
        source_n_n = -np.asarray(recombination.electron_density_derivative)
        source_n_p = -np.asarray(recombination.hole_density_derivative)
        source_p_n = source_n_n.copy()
        source_p_p = source_n_p.copy()
        dynamic_nodes = np.zeros(self.node_count, dtype=bool)
        dynamic_nodes[self.layout.device_node_indices] = True
        source_n_n[dynamic_nodes] += np.asarray(legacy.electron_density_derivative)[
            dynamic_nodes
        ]
        source_n_p[dynamic_nodes] += np.asarray(legacy.hole_density_derivative)[
            dynamic_nodes
        ]
        source_p_n[dynamic_nodes] += np.asarray(legacy.electron_density_derivative)[
            dynamic_nodes
        ]
        source_p_p[dynamic_nodes] += np.asarray(legacy.hole_density_derivative)[
            dynamic_nodes
        ]

        source_n_jacobian = (
            sparse.diags(source_n_n) @ n_jacobian
            + sparse.diags(source_n_p) @ p_jacobian
        ).tolil()
        source_p_jacobian = (
            sparse.diags(source_p_n) @ n_jacobian
            + sparse.diags(source_p_p) @ p_jacobian
        ).tolil()
        trap_rate_jacobian = sparse.lil_matrix((self.trap_count, dimension))
        nodes = self.layout.device_node_indices
        population = self.layout.population_density_m3
        capture_n = self.layout.capture_n_m3_s
        capture_p = self.layout.capture_p_m3_s
        for state_index, node in enumerate(nodes):
            electron_n = (
                population[state_index]
                * capture_n[state_index]
                * (1.0 - occupancy[state_index])
            )
            electron_f = (
                -population[state_index]
                * capture_n[state_index]
                * (n[node] + self.layout.n1_m3[state_index])
            )
            hole_p = (
                population[state_index]
                * capture_p[state_index]
                * occupancy[state_index]
            )
            hole_f = (
                population[state_index]
                * capture_p[state_index]
                * (p[node] + self.layout.p1_m3[state_index])
            )
            source_n_jacobian[node] -= electron_n * n_jacobian.getrow(node)
            source_n_jacobian[
                node,
                self.trap_slice.start + state_index,
            ] -= electron_f * occupancy_tangent[state_index]
            source_p_jacobian[node] -= hole_p * p_jacobian.getrow(node)
            source_p_jacobian[
                node,
                self.trap_slice.start + state_index,
            ] -= hole_f * occupancy_tangent[state_index]
            trap_rate_jacobian[state_index] += electron_n * n_jacobian.getrow(node)
            trap_rate_jacobian[state_index] -= hole_p * p_jacobian.getrow(node)
            trap_rate_jacobian[
                state_index,
                self.trap_slice.start + state_index,
            ] += (electron_f - hole_f) * occupancy_tangent[state_index]

        electron_local = sg_fluxes_n_jacobian(
            phi + self.material.chi,
            n,
            np.diff(self.grid),
            self.material.D_n_face,
            self.thermal_voltage,
        )
        hole_local = sg_fluxes_p_jacobian(
            phi + self.material.chi + self.material.Eg,
            p,
            np.diff(self.grid),
            self.material.D_p_face,
            self.thermal_voltage,
        )

        def current_jacobian(local, density_jacobian) -> sparse.csr_matrix:
            return (
                sparse.diags(local.density_left_derivative) @ density_jacobian[:-1]
                + sparse.diags(local.density_right_derivative) @ density_jacobian[1:]
                + sparse.diags(local.potential_left_derivative) @ phi_jacobian[:-1]
                + sparse.diags(local.potential_right_derivative) @ phi_jacobian[1:]
            ).tocsr()

        current_n_jacobian = current_jacobian(electron_local, n_jacobian)
        current_p_jacobian = current_jacobian(hole_local, p_jacobian)
        inverse_charge_volume = sparse.diags(1.0 / (Q * self.widths))
        rate_n_jacobian = source_n_jacobian.tocsr() + (
            inverse_charge_volume @ self._divergence @ current_n_jacobian
        )
        rate_p_jacobian = source_p_jacobian.tocsr() - (
            inverse_charge_volume @ self._divergence @ current_p_jacobian
        )
        rate_jacobian = sparse.vstack(
            (
                rate_n_jacobian[1:-1],
                rate_p_jacobian[1:-1],
                trap_rate_jacobian.tocsr(),
            ),
            format="csr",
        )

        trap_charge_jacobian = -Q * sparse.diags(population) @ occupancy_jacobian
        charge_jacobian = Q * (p_jacobian - n_jacobian)
        trap_charge_nodes = sparse.lil_matrix((self.node_count, dimension))
        for state_index, node in enumerate(nodes):
            trap_charge_nodes[node] += trap_charge_jacobian.getrow(state_index)
        charge_jacobian += trap_charge_nodes.tocsr()
        poisson_jacobian = (
            self._poisson_laplacian @ phi_jacobian
            + sparse.diags(self.material.poisson_factor.h_cell) @ charge_jacobian[1:-1]
        ).tocsr()
        return storage_jacobian, rate_jacobian, poisson_jacobian

    def residual_and_jacobian(
        self,
        coordinate: np.ndarray,
        voltage: float,
        previous_storage: np.ndarray,
        dt: float,
        storage_scale: np.ndarray,
        poisson_scale: np.ndarray,
    ) -> tuple[np.ndarray, sparse.csr_matrix, _DeviceState]:
        state = self.evaluate(coordinate, voltage)
        storage_residual = state.storage - previous_storage - dt * state.rate
        residual = np.r_[
            storage_residual / storage_scale,
            state.poisson_residual / poisson_scale,
        ]
        jacobian = sparse.vstack(
            (
                sparse.diags(1.0 / storage_scale)
                @ (state.storage_jacobian - dt * state.rate_jacobian),
                sparse.diags(1.0 / poisson_scale) @ state.poisson_jacobian,
            ),
            format="csr",
        )
        return residual, jacobian, state

    def storage_scale(
        self,
        previous_storage: np.ndarray,
        previous_state: _DeviceState,
        dt: float,
        policy: BulkDefectTransientPolicy,
    ) -> np.ndarray:
        reference = np.r_[
            self.reference_n[1:-1],
            self.reference_p[1:-1],
            self.layout.population_density_m3 * self.reference_occupancy,
        ]
        absolute = np.r_[
            np.full(2 * self.interior_count, policy.carrier_storage_atol_m3),
            np.full(self.trap_count, policy.trap_storage_atol_m3),
        ]
        scale = absolute + policy.storage_relative_tolerance * np.maximum(
            np.abs(previous_storage),
            np.abs(reference),
        )
        # At a DC root the two adjacent face currents can be O(1-100 A/m2)
        # while their conservative divergence is at the floating-point noise
        # floor.  Convert that explicitly bounded subtraction floor into the
        # carrier storage units used by the backward-Euler residual.  Trap
        # storage is local and does not receive this transport allowance.
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
        return scale

    def poisson_scale(self, policy: BulkDefectTransientPolicy) -> np.ndarray:
        factor = self.material.poisson_factor
        reference = (factor.C[:-1] + factor.C[1:]) * self.thermal_voltage
        return policy.poisson_atol_C_m2 + policy.poisson_relative_tolerance * reference

    def integrated_charge(self, state: _DeviceState) -> float:
        rho = Q * (state.p - state.n) + state.trap_charge
        return float(np.sum(rho[1:-1] * self.widths[1:-1]))

    def eliminated_operator_error(
        self,
        state: _DeviceState,
        voltage: float,
    ) -> float:
        eliminated = self.system.evaluate_quasi_fermi_increments_dynamic_bulk(
            state.dqfn,
            state.dqfp,
            self.layout,
            state.occupancy,
            1.0 if self.illuminated else 0.0,
            V_app=float(voltage),
            reference_electron_density_m3=self.reference_n,
            reference_hole_density_m3=self.reference_p,
            reference_occupancy=self.reference_occupancy,
        )

        def relative(left: np.ndarray, right: np.ndarray, floor: float) -> float:
            scale = max(
                float(np.max(np.abs(left))),
                float(np.max(np.abs(right))),
                floor,
            )
            return float(np.max(np.abs(left - right))) / scale

        current_scale = max(
            float(np.max(np.abs(state.current_n))),
            float(np.max(np.abs(state.current_p))),
            float(np.max(np.abs(eliminated.current_n))),
            float(np.max(np.abs(eliminated.current_p))),
            1.0,
        )
        optical_rate_scale = abs(float(self.stack.Phi)) / float(
            self.grid[-1] - self.grid[0]
        )
        transport_rate_scale = current_scale / (Q * float(np.min(self.widths[1:-1])))
        rate_scale = max(
            optical_rate_scale,
            transport_rate_scale * math.sqrt(np.finfo(float).eps),
            1.0,
        )

        values = (
            relative(state.n, eliminated.y[: self.node_count], 1.0),
            relative(
                state.p,
                eliminated.y[self.node_count : 2 * self.node_count],
                1.0,
            ),
            relative(state.phi, eliminated.phi, self.thermal_voltage),
            relative(
                state.rate[: self.interior_count],
                eliminated.rate_n[1:-1],
                rate_scale,
            ),
            relative(
                state.rate[self.interior_count : 2 * self.interior_count],
                eliminated.rate_p[1:-1],
                rate_scale,
            ),
            relative(state.current_n, eliminated.current_n, current_scale),
            relative(state.current_p, eliminated.current_p, current_scale),
        )
        return max(values)


def _validate_trace_inputs(
    times_s: object,
    voltage_V: object,
) -> tuple[np.ndarray, np.ndarray]:
    times = np.asarray(times_s, dtype=float)
    voltage = np.asarray(voltage_V, dtype=float)
    if (
        times.ndim != 1
        or times.size < 2
        or not np.all(np.isfinite(times))
        or np.any(np.diff(times) <= 0.0)
        or voltage.shape != times.shape
        or not np.all(np.isfinite(voltage))
    ):
        raise ValueError(
            "times_s and voltage_V must be finite aligned vectors with increasing time"
        )
    return times, voltage


def _jacobian_error(
    system: _BulkTransientSystem,
    coordinate: np.ndarray,
    voltage: float,
    previous_storage: np.ndarray,
    dt: float,
    storage_scale: np.ndarray,
    poisson_scale: np.ndarray,
    analytic: sparse.csr_matrix,
    step: float,
) -> float:
    columns: list[float] = []
    analytic_dense = analytic.toarray()
    for column in range(system.dimension):
        plus = coordinate.copy()
        minus = coordinate.copy()
        plus[column] += step
        minus[column] -= step
        plus_residual, _, _ = system.residual_and_jacobian(
            plus,
            voltage,
            previous_storage,
            dt,
            storage_scale,
            poisson_scale,
        )
        minus_residual, _, _ = system.residual_and_jacobian(
            minus,
            voltage,
            previous_storage,
            dt,
            storage_scale,
            poisson_scale,
        )
        finite = (plus_residual - minus_residual) / (2.0 * step)
        expected = analytic_dense[:, column]
        scale = max(float(np.linalg.norm(expected)), 1.0e-12)
        columns.append(float(np.linalg.norm(finite - expected)) / scale)
    return max(columns, default=0.0)


def _solve_step(
    system: _BulkTransientSystem,
    coordinate: np.ndarray,
    previous: _DeviceState,
    voltage: float,
    dt: float,
    policy: BulkDefectTransientPolicy,
    *,
    check_jacobian: bool,
) -> tuple[_DeviceState, int, float, float, int]:
    storage_scale = system.storage_scale(
        previous.storage,
        previous,
        dt,
        policy,
    )
    poisson_scale = system.poisson_scale(policy)
    trial = np.asarray(coordinate, dtype=float).copy()
    maximum_jacobian_error = 0.0
    maximum_nnz = 0
    for iteration in range(1, policy.maximum_newton_iterations + 1):
        residual, jacobian, state = system.residual_and_jacobian(
            trial,
            voltage,
            previous.storage,
            dt,
            storage_scale,
            poisson_scale,
        )
        norm = float(np.max(np.abs(residual)))
        maximum_nnz = max(maximum_nnz, int(jacobian.nnz))
        if norm <= policy.maximum_scaled_nonlinear_residual:
            if check_jacobian:
                maximum_jacobian_error = _jacobian_error(
                    system,
                    trial,
                    voltage,
                    previous.storage,
                    dt,
                    storage_scale,
                    poisson_scale,
                    jacobian,
                    policy.jacobian_check_step,
                )
            return state, iteration - 1, norm, maximum_jacobian_error, maximum_nnz
        with warnings.catch_warnings():
            warnings.simplefilter("error", MatrixRankWarning)
            try:
                step = np.asarray(spsolve(jacobian, -residual), dtype=float)
            except (MatrixRankWarning, RuntimeError, ValueError) as exc:
                raise BulkDefectTransientError(
                    f"analytic sparse Newton solve failed: {exc}"
                ) from exc
        if step.shape != trial.shape or not np.all(np.isfinite(step)):
            raise BulkDefectTransientError(
                "analytic sparse Newton solve returned a non-finite step"
            )
        accepted = False
        damping = 1.0
        for _ in range(policy.maximum_line_search_steps):
            candidate = trial + damping * step
            try:
                candidate_residual, _, _ = system.residual_and_jacobian(
                    candidate,
                    voltage,
                    previous.storage,
                    dt,
                    storage_scale,
                    poisson_scale,
                )
            except (BulkDefectTransientError, ValueError, FloatingPointError):
                damping *= 0.5
                continue
            candidate_norm = float(np.max(np.abs(candidate_residual)))
            if candidate_norm < norm * (1.0 - 1.0e-4 * damping):
                trial = candidate
                accepted = True
                break
            damping *= 0.5
        if not accepted:
            raise BulkDefectTransientError(
                f"analytic sparse Newton line search stalled at residual {norm:.6g}"
            )
    raise BulkDefectTransientError(
        f"analytic sparse Newton exceeded {policy.maximum_newton_iterations} iterations"
    )


def _integrate_trace(
    system: _BulkTransientSystem,
    times: np.ndarray,
    voltage: np.ndarray,
    substeps: int,
    policy: BulkDefectTransientPolicy,
) -> _Trace:
    coordinate = system.initial_coordinate()
    initial = system.evaluate(coordinate, float(voltage[0]))
    states: list[_DeviceState] = [initial]
    coordinates = [coordinate.copy()]
    displacement = [np.zeros(system.node_count - 1, dtype=float)]
    total_current = [initial.conduction.copy()]
    integrated_charge = [system.integrated_charge(initial)]
    iterations = [0]
    maximum_scaled_residual = 0.0
    maximum_poisson = float(np.max(np.abs(initial.poisson_residual)))
    maximum_jacobian_error = 0.0
    maximum_charge_absolute_error = 0.0
    maximum_charge_error = 0.0
    maximum_face_spread = 0.0
    maximum_operator_error = system.eliminated_operator_error(
        initial,
        float(voltage[0]),
    )
    maximum_nnz = 0
    previous = initial
    previous_phi = initial.phi
    for point in range(1, times.size):
        interval = float(times[point] - times[point - 1])
        dt = interval / substeps
        target_voltage = float(voltage[point])
        final_displacement = np.zeros(system.node_count - 1, dtype=float)
        final_total = previous.conduction.copy()
        point_iterations = 0
        for local_step in range(substeps):
            state, count, residual, jacobian_error, nnz = _solve_step(
                system,
                coordinate,
                previous,
                target_voltage,
                dt,
                policy,
                check_jacobian=(point == 1 and local_step == 0),
            )
            point_iterations += count
            coordinate = state.coordinate.copy()
            field = -np.diff(state.phi) / np.diff(system.grid)
            previous_field = -np.diff(previous_phi) / np.diff(system.grid)
            final_displacement = (
                system.polarity * system.eps_face * (field - previous_field) / dt
            )
            final_total = state.conduction + final_displacement
            charge = system.integrated_charge(state)
            storage_rate = (
                system.polarity * (charge - integrated_charge[-1]) / dt
                if local_step == 0
                else system.polarity
                * (charge - system.integrated_charge(previous))
                / dt
            )
            boundary_rate = state.conduction[0] - state.conduction[-1]
            charge_scale = max(
                abs(storage_rate),
                abs(boundary_rate),
                float(np.max(np.abs(state.conduction))),
                1.0,
            )
            charge_absolute_error = abs(storage_rate - boundary_rate)
            charge_error = charge_absolute_error / charge_scale
            face_scale = max(float(np.max(np.abs(final_total))), 1.0e-20)
            face_spread = float(np.ptp(final_total)) / face_scale
            maximum_scaled_residual = max(maximum_scaled_residual, residual)
            maximum_poisson = max(
                maximum_poisson,
                float(np.max(np.abs(state.poisson_residual))),
            )
            maximum_jacobian_error = max(
                maximum_jacobian_error,
                jacobian_error,
            )
            maximum_charge_absolute_error = max(
                maximum_charge_absolute_error,
                charge_absolute_error,
            )
            maximum_charge_error = max(maximum_charge_error, charge_error)
            maximum_face_spread = max(maximum_face_spread, face_spread)
            maximum_operator_error = max(
                maximum_operator_error,
                system.eliminated_operator_error(state, target_voltage),
            )
            maximum_nnz = max(maximum_nnz, nnz)
            previous = state
            previous_phi = state.phi
        states.append(previous)
        coordinates.append(coordinate.copy())
        displacement.append(final_displacement)
        total_current.append(final_total)
        integrated_charge.append(system.integrated_charge(previous))
        iterations.append(point_iterations)
    return _Trace(
        times=times,
        voltage=voltage,
        coordinates=np.asarray(coordinates),
        states=tuple(states),
        displacement=np.asarray(displacement),
        total_current=np.asarray(total_current),
        integrated_charge=np.asarray(integrated_charge),
        iterations=np.asarray(iterations),
        maximum_scaled_residual=maximum_scaled_residual,
        maximum_poisson_residual=maximum_poisson,
        maximum_jacobian_error=maximum_jacobian_error,
        maximum_charge_balance_absolute_error=maximum_charge_absolute_error,
        maximum_charge_balance_error=maximum_charge_error,
        maximum_face_spread=maximum_face_spread,
        maximum_operator_error=maximum_operator_error,
        maximum_nnz=maximum_nnz,
    )


def _refinement_changes(coarse: _Trace, fine: _Trace) -> tuple[float, float]:
    n_coarse = np.asarray([state.n for state in coarse.states])
    n_fine = np.asarray([state.n for state in fine.states])
    p_coarse = np.asarray([state.p for state in coarse.states])
    p_fine = np.asarray([state.p for state in fine.states])
    f_coarse = np.asarray([state.occupancy for state in coarse.states])
    f_fine = np.asarray([state.occupancy for state in fine.states])
    phi_coarse = np.asarray([state.phi for state in coarse.states])
    phi_fine = np.asarray([state.phi for state in fine.states])
    carrier_change = max(
        float(np.max(np.abs(np.log(n_coarse / n_fine)))),
        float(np.max(np.abs(np.log(p_coarse / p_fine)))),
    )
    occupancy_change = float(np.max(np.abs(f_coarse - f_fine)))
    potential_scale = max(float(np.ptp(phi_fine)), 0.025)
    potential_change = float(np.max(np.abs(phi_coarse - phi_fine))) / potential_scale
    state_change = max(carrier_change, occupancy_change, potential_change)
    current_scale = max(float(np.max(np.abs(fine.total_current[1:]))), 1.0)
    current_change = (
        float(np.max(np.abs(coarse.total_current[1:] - fine.total_current[1:])))
        / current_scale
    )
    return state_change, current_change


def run_bulk_defect_device_transient(
    x: np.ndarray,
    stack: DeviceStack,
    times_s: object,
    voltage_V: object,
    *,
    illuminated: bool = False,
    mat: MaterialArrays | None = None,
    dc_state: QuasiFermiSteadyStateResult | None = None,
    policy: BulkDefectTransientPolicy | None = None,
    require_certificate: bool = True,
) -> BulkDefectTransientResult:
    """Integrate a bounded, charged single-level bulk-trap device transient.

    ``voltage_V[0]`` defines the residual-certified DC reference.  Every later
    value is applied as a right-continuous step-and-hold over its preceding
    time interval.  This narrow D6-E1 route deliberately rejects distributed
    or spatially graded defects, interface states, mobile ions, and all legacy
    MoL state reinterpretation.
    """
    grid = np.asarray(x, dtype=float)
    if (
        grid.ndim != 1
        or grid.size < 4
        or not np.all(np.isfinite(grid))
        or np.any(np.diff(grid) <= 0.0)
    ):
        raise ValueError("x must be a finite increasing grid with at least four nodes")
    times, voltage = _validate_trace_inputs(times_s, voltage_V)
    if not isinstance(illuminated, (bool, np.bool_)):
        raise TypeError("illuminated must be boolean")
    if not isinstance(require_certificate, (bool, np.bool_)):
        raise TypeError("require_certificate must be boolean")
    resolved_policy = policy or BulkDefectTransientPolicy()
    if not isinstance(resolved_policy, BulkDefectTransientPolicy):
        raise TypeError("policy must be a BulkDefectTransientPolicy or None")
    try:
        material = (
            build_material_arrays(
                grid,
                stack,
                explicit_defect_charge_closure=EXPLICIT_DEFECT_CHARGE_QF_DC,
            )
            if mat is None
            else mat
        )
        _require_material_defect_contract(
            stack,
            material,
            defect_energy_quadrature_order=(DEFAULT_DEFECT_ENERGY_QUADRATURE_ORDER),
        )
        _require_supported(material, allow_charged_bulk_defects=True)
    except ExplicitDefectCapabilityError as exc:
        raise BulkDefectTransientError(
            f"bulk defect transient requires a supported explicit-defect model: {exc}"
        ) from exc
    except QuasiFermiSteadyStateError as exc:
        raise BulkDefectTransientError(
            f"bulk defect transient material contract failed: {exc}"
        ) from exc
    model = material.monovalent_bulk_defects
    if model is None:
        raise BulkDefectTransientError(
            "bulk defect transient requires a compiled explicit-defect model"
        )
    if model.has_distributed_species or model.has_spatial_profiles:
        raise BulkDefectTransientError(
            "D6-E1 supports only non-spatial single-level bulk defects"
        )
    if any(
        source.distribution.kind != SINGLE_LEVEL
        for region in model.regions
        for source in region.species
    ):
        raise BulkDefectTransientError(
            "D6-E1 supports only single-level bulk defect species"
        )
    if any(
        source.charge_transition == NEUTRAL
        for region in model.regions
        for source in region.species
    ):
        raise BulkDefectTransientError(
            "D6-E1 charge-coupled transient rejects neutral transitions"
        )

    operating_point = dc_state
    if operating_point is None:
        try:
            operating_point = solve_quasi_fermi_steady_state(
                grid,
                stack,
                V_app=float(voltage[0]),
                illuminated=bool(illuminated),
                mat=material,
            )
        except QuasiFermiSteadyStateError as exc:
            raise BulkDefectTransientError(
                f"bulk defect transient could not certify its DC state: {exc}"
            ) from exc
    if not operating_point.certified:
        raise BulkDefectTransientError(
            "bulk defect transient requires a certified QF DC state"
        )
    if operating_point.contact_thermodynamic_status != "certified":
        raise BulkDefectTransientError(
            "bulk defect transient requires contact-thermodynamic certification"
        )
    if not np.isclose(
        operating_point.V_app,
        voltage[0],
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise BulkDefectTransientError("DC-state voltage does not match voltage_V[0]")
    if bool(operating_point.illuminated) != bool(illuminated):
        raise BulkDefectTransientError("DC-state illumination does not match")
    if (
        operating_point.bulk_defect_diagnostics is None
        or operating_point.bulk_defect_diagnostics.model_identity_sha256
        != model.identity_sha256
    ):
        raise BulkDefectTransientError(
            "DC-state defect model identity does not match the transient material"
        )
    dynamic_mask = np.ones(grid.size, dtype=bool)
    dynamic_mask[[0, -1]] = False
    layout = compile_dynamic_bulk_trap_layout(
        model,
        dynamic_node_mask=dynamic_mask,
    )
    qf_system = _QuasiFermiSystem(
        grid,
        stack,
        material,
        float(voltage[0]),
        poisson_tolerance_V=1.0e-13,
        poisson_max_iterations=100,
    )
    qss = qf_system.evaluate_quasi_fermi_increments(
        np.asarray(operating_point.electron_quasi_fermi_increment_V),
        np.asarray(operating_point.hole_quasi_fermi_increment_V),
        1.0 if illuminated else 0.0,
        V_app=float(voltage[0]),
    )
    occupancy = quasi_steady_bulk_trap_occupancy(
        qss.y[: grid.size],
        qss.y[grid.size : 2 * grid.size],
        layout,
    )
    dynamic_dc = qf_system.evaluate_quasi_fermi_increments_dynamic_bulk(
        np.asarray(operating_point.electron_quasi_fermi_increment_V),
        np.asarray(operating_point.hole_quasi_fermi_increment_V),
        layout,
        occupancy,
        1.0 if illuminated else 0.0,
        V_app=float(voltage[0]),
        reference_electron_density_m3=qss.y[: grid.size],
        reference_hole_density_m3=qss.y[grid.size : 2 * grid.size],
        reference_occupancy=occupancy,
    )
    current_scale = max(
        float(np.max(np.abs(qss.current_n + qss.current_p))),
        abs(Q * float(stack.Phi)),
        1.0,
    )
    rate_difference_A_m2 = Q * float(
        np.sum(
            (
                np.abs(dynamic_dc.rate_n - qss.rate_n)
                + np.abs(dynamic_dc.rate_p - qss.rate_p)
            )
            * np.asarray(material.dx_cell)
        )
    )
    qss_embedding_error = max(
        float(np.max(np.abs(dynamic_dc.phi - qss.phi))) / material.V_T_device,
        rate_difference_A_m2 / current_scale,
        float(np.max(np.abs(dynamic_dc.current_n - qss.current_n))) / current_scale,
        float(np.max(np.abs(dynamic_dc.current_p - qss.current_p))) / current_scale,
    )
    system = _BulkTransientSystem(
        grid,
        stack,
        material,
        operating_point,
        layout,
        occupancy,
        dynamic_dc,
        illuminated=bool(illuminated),
    )
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
    changes = tuple(
        _refinement_changes(coarse, fine) for coarse, fine in zip(levels, levels[1:])
    )
    refinement_state = changes[-1][0]
    refinement_current = changes[-1][1]
    final = levels[-1]
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
            "carrier_trap_charge_balance_failed",
            final.maximum_charge_balance_error,
            resolved_policy.maximum_charge_balance_relative_error,
        ),
        (
            "all_face_total_current_closure_failed",
            final.maximum_face_spread,
            resolved_policy.maximum_all_face_current_spread_relative,
        ),
        (
            "eliminated_qf_operator_mismatch",
            final.maximum_operator_error,
            resolved_policy.maximum_eliminated_operator_relative_error,
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
        if not np.isfinite(value) or value > limit:
            reasons.append(reason)
    dense_entries = system.dimension * system.dimension
    if final.maximum_nnz >= dense_entries:
        reasons.append("analytic_jacobian_not_sparse")
    certificate = BulkDefectTransientCertificate(
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
        maximum_refinement_state_change=refinement_state,
        maximum_refinement_current_relative_change=refinement_current,
        analytic_jacobian_nnz=final.maximum_nnz,
        dense_jacobian_entries=dense_entries,
        sparse_linear_solver_used=True,
        clipping_used=False,
        certified=not reasons,
        reasons=tuple(reasons),
    )
    result = BulkDefectTransientResult(
        times_s=times,
        voltage_V=voltage,
        electron_density_m3=np.asarray([state.n for state in final.states]),
        hole_density_m3=np.asarray([state.p for state in final.states]),
        trap_occupancy=np.asarray([state.occupancy for state in final.states]),
        electrostatic_potential_V=np.asarray([state.phi for state in final.states]),
        trap_charge_density_C_m3=np.asarray(
            [state.trap_charge for state in final.states]
        ),
        conduction_current_faces_A_m2=np.asarray(
            [state.conduction for state in final.states]
        ),
        displacement_current_faces_A_m2=final.displacement,
        total_current_faces_A_m2=final.total_current,
        integrated_free_and_trap_charge_C_m2=final.integrated_charge,
        newton_iterations=final.iterations,
        layout=layout,
        dc_state=operating_point,
        policy=resolved_policy,
        certificate=certificate,
    )
    if require_certificate and not certificate.certified:
        raise BulkDefectTransientCertificationError(
            "bulk defect device transient did not certify: "
            + ", ".join(certificate.reasons),
            result,
        )
    return result


__all__ = [
    "BULK_DEFECT_TRANSIENT_SCOPE",
    "BULK_DEFECT_TRANSIENT_VERSION",
    "BulkDefectTransientCertificate",
    "BulkDefectTransientCertificationError",
    "BulkDefectTransientError",
    "BulkDefectTransientPolicy",
    "BulkDefectTransientResult",
    "run_bulk_defect_device_transient",
]
