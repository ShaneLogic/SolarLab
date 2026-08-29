"""Research-only transient for dynamic two-sided interface occupancy.

The bulk carrier quasi-Fermi potentials, shared interface trap logits, and
bulk electrostatic potential are coupled to six algebraic trace variables per
physical interface.  Retaining those local variables avoids a nested nonlinear
solve inside every device Newton evaluation and preserves an analytic sparse
index-1 DAE Jacobian.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import math
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
    EquilibriumReferencedInterfaceChargeDarkReference,
    QuasiFermiSteadyStateError,
    QuasiFermiSteadyStateResult,
    _QuasiFermiSystem,
    _build_qf_material,
    _prepare_two_sided_material,
    _require_supported,
    _research_array_sha256,
    _research_charge_off_stack,
    build_equilibrium_referenced_interface_charge_dark_reference,
    solve_equilibrium_referenced_interface_charge_steady_state,
)
from perovskite_sim.models.device import DeviceStack
from perovskite_sim.physics.defect_distributions import (
    DEFAULT_DEFECT_ENERGY_QUADRATURE_ORDER,
)
from perovskite_sim.physics.interface_plane import FERMI_DIRAC_RICHARDSON
from perovskite_sim.physics.recombination import total_recombination_derivatives
from perovskite_sim.physics.two_sided_interface import (
    TWO_SIDED_TRACE,
    EquilibriumReferencedSheetCharge,
    FixedOccupancyCarrierTangent,
    InterfaceTracePotentials,
    TwoSidedMaterialQSSResult,
    _material_two_sided_interface_problem,
    electrostatic_trace_residual_and_jacobian,
    fixed_occupancy_carrier_tangent,
    shared_trap_occupancy,
    solve_electrostatic_traces,
    solve_fixed_occupancy_two_sided_interface,
)
from perovskite_sim.solver.mol import (
    MaterialArrays,
    _harmonic_face_average,
    assemble_rhs,
    poisson_right_boundary,
)


INTERFACE_DEFECT_TRANSIENT_SCOPE = (
    "research_two_sided_dynamic_interface_defect_device_transient_only"
)
INTERFACE_DEFECT_TRANSIENT_VERSION = "two-sided-dynamic-interface-transient-v1"
_RIGHT_FIRST = np.array([2, 3, 0, 1])


class InterfaceDefectTransientError(RuntimeError):
    """The dynamic two-sided-interface transient failed closed."""


class InterfaceDefectTransientCertificationError(InterfaceDefectTransientError):
    """A finite transient trace failed one or more declared evidence gates."""

    def __init__(
        self,
        message: str,
        result: "InterfaceDefectTransientResult",
    ) -> None:
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


def _occupancy_logit(occupancy: np.ndarray) -> np.ndarray:
    values = np.asarray(occupancy, dtype=float)
    if (
        values.ndim != 1
        or not np.all(np.isfinite(values))
        or np.any((values <= 0.0) | (values >= 1.0))
    ):
        raise InterfaceDefectTransientError(
            "dynamic interface occupancy must lie strictly inside (0, 1)"
        )
    return np.log(values) - np.log1p(-values)


def _occupancy_from_increment(
    reference_logit: np.ndarray,
    increment: np.ndarray,
) -> np.ndarray:
    values = np.asarray(reference_logit, dtype=float) + np.asarray(
        increment,
        dtype=float,
    )
    if not np.all(np.isfinite(values)):
        raise InterfaceDefectTransientError(
            "interface occupancy coordinates are non-finite"
        )
    occupancy = np.empty_like(values)
    positive = values >= 0.0
    occupancy[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    occupancy[~positive] = exponential / (1.0 + exponential)
    if np.any((occupancy <= 0.0) | (occupancy >= 1.0)):
        raise InterfaceDefectTransientError(
            "interface occupancy logit saturated outside resolvable bounds"
        )
    return occupancy


@dataclass(frozen=True, slots=True)
class InterfaceDefectTransientPolicy:
    """Nonlinear, local-algebraic, refinement, and conservation gates."""

    storage_relative_tolerance: float = 1.0e-9
    carrier_storage_atol_m3: float = 1.0
    interface_storage_atol_m2: float = 1.0
    poisson_relative_tolerance: float = 1.0e-10
    poisson_atol_C_m2: float = 1.0e-18
    interface_algebraic_relative_tolerance: float = 1.0e-10
    interface_potential_atol_V: float = 1.0e-14
    interface_gauss_atol_C_m2: float = 1.0e-18
    interface_flux_atol_m2_s: float = 1.0
    maximum_scaled_nonlinear_residual: float = 5.0e-2
    maximum_newton_iterations: int = 35
    maximum_line_search_steps: int = 20
    jacobian_check_step: float = 1.0e-6
    maximum_jacobian_column_relative_error: float = 3.0e-4
    refinement_substeps: tuple[int, ...] = (1, 2, 4)
    maximum_refinement_state_change: float = 2.0e-2
    maximum_refinement_current_relative_change: float = 5.0e-2
    maximum_charge_balance_relative_error: float = 1.0e-10
    maximum_all_face_current_spread_relative: float = 2.0e-6
    maximum_two_sided_interface_total_current_relative_error: float = 2.0e-6
    maximum_eliminated_operator_relative_error: float = 3.0e-7
    maximum_local_carrier_normalized_residual: float = 1.0e-7
    maximum_local_gauss_normalized_residual: float = 1.0e-7

    def __post_init__(self) -> None:
        for name in (
            "storage_relative_tolerance",
            "carrier_storage_atol_m3",
            "interface_storage_atol_m2",
            "poisson_relative_tolerance",
            "poisson_atol_C_m2",
            "interface_algebraic_relative_tolerance",
            "interface_potential_atol_V",
            "interface_gauss_atol_C_m2",
            "interface_flux_atol_m2_s",
            "maximum_scaled_nonlinear_residual",
            "jacobian_check_step",
            "maximum_jacobian_column_relative_error",
            "maximum_refinement_state_change",
            "maximum_refinement_current_relative_change",
            "maximum_charge_balance_relative_error",
            "maximum_all_face_current_spread_relative",
            "maximum_two_sided_interface_total_current_relative_error",
            "maximum_eliminated_operator_relative_error",
            "maximum_local_carrier_normalized_residual",
            "maximum_local_gauss_normalized_residual",
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
class InterfaceDefectTransientCertificate:
    """Numerical and physical evidence for one returned interface trace."""

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
    maximum_refinement_state_change: float
    maximum_refinement_current_relative_change: float
    analytic_jacobian_nnz: int
    dense_jacobian_entries: int
    sparse_linear_solver_used: bool
    clipping_used: bool
    certified: bool
    reasons: tuple[str, ...]
    scope: str = INTERFACE_DEFECT_TRANSIENT_SCOPE
    version: str = INTERFACE_DEFECT_TRANSIENT_VERSION


@dataclass(frozen=True, slots=True, eq=False)
class InterfaceDefectTransientResult:
    """Immutable physical trace from the finest nested time grid."""

    times_s: np.ndarray
    voltage_V: np.ndarray
    electron_density_m3: np.ndarray
    hole_density_m3: np.ndarray
    interface_occupancy: np.ndarray
    interface_quasi_steady_occupancy: np.ndarray
    electrostatic_potential_V: np.ndarray
    interface_trace_potential_V: np.ndarray
    interface_trace_state_m3: np.ndarray
    interface_sheet_charge_C_m2: np.ndarray
    electron_capture_flux_m2_s: np.ndarray
    hole_capture_flux_m2_s: np.ndarray
    electron_bulk_flux_m2_s: np.ndarray
    hole_bulk_flux_m2_s: np.ndarray
    conduction_current_faces_A_m2: np.ndarray
    displacement_current_faces_A_m2: np.ndarray
    total_current_faces_A_m2: np.ndarray
    interface_conduction_current_A_m2: np.ndarray
    interface_displacement_current_A_m2: np.ndarray
    interface_total_current_A_m2: np.ndarray
    integrated_free_and_interface_charge_C_m2: np.ndarray
    newton_iterations: np.ndarray
    dc_state: QuasiFermiSteadyStateResult
    dark_reference: EquilibriumReferencedInterfaceChargeDarkReference
    policy: InterfaceDefectTransientPolicy
    certificate: InterfaceDefectTransientCertificate
    state_coordinate: str = "qf_interface_logit_potential_local_algebraic"
    time_discretization: str = "backward_euler_index_1_dae"
    scope: str = INTERFACE_DEFECT_TRANSIENT_SCOPE
    version: str = INTERFACE_DEFECT_TRANSIENT_VERSION

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
            raise InterfaceDefectTransientError("times/voltage trace is invalid")
        object.__setattr__(self, "times_s", times)
        object.__setattr__(self, "voltage_V", voltage)
        point_count = times.size
        node_count = self.electron_density_m3.shape[1]
        interface_count = len(self.dark_reference.trap_density_m2)
        shapes = {
            "electron_density_m3": (point_count, node_count),
            "hole_density_m3": (point_count, node_count),
            "interface_occupancy": (point_count, interface_count),
            "interface_quasi_steady_occupancy": (point_count, interface_count),
            "electrostatic_potential_V": (point_count, node_count),
            "interface_trace_potential_V": (point_count, interface_count, 2),
            "interface_trace_state_m3": (point_count, interface_count, 4),
            "interface_sheet_charge_C_m2": (point_count, interface_count),
            "electron_capture_flux_m2_s": (point_count, interface_count, 2),
            "hole_capture_flux_m2_s": (point_count, interface_count, 2),
            "electron_bulk_flux_m2_s": (point_count, interface_count, 2),
            "hole_bulk_flux_m2_s": (point_count, interface_count, 2),
            "conduction_current_faces_A_m2": (point_count, node_count - 1),
            "displacement_current_faces_A_m2": (point_count, node_count - 1),
            "total_current_faces_A_m2": (point_count, node_count - 1),
            "interface_conduction_current_A_m2": (
                point_count,
                interface_count,
                2,
            ),
            "interface_displacement_current_A_m2": (
                point_count,
                interface_count,
                2,
            ),
            "interface_total_current_A_m2": (point_count, interface_count, 2),
            "integrated_free_and_interface_charge_C_m2": (point_count,),
        }
        for name, shape in shapes.items():
            values = _readonly(getattr(self, name))
            if values.shape != shape or not np.all(np.isfinite(values)):
                raise InterfaceDefectTransientError(f"{name} is invalid")
            object.__setattr__(self, name, values)
        if np.any(self.electron_density_m3 <= 0.0) or np.any(
            self.hole_density_m3 <= 0.0
        ):
            raise InterfaceDefectTransientError(
                "carrier densities must remain positive"
            )
        if np.any(
            (self.interface_occupancy <= 0.0) | (self.interface_occupancy >= 1.0)
        ):
            raise InterfaceDefectTransientError(
                "interface occupancy must remain strictly inside (0, 1)"
            )
        if np.any(
            (self.interface_quasi_steady_occupancy <= 0.0)
            | (self.interface_quasi_steady_occupancy >= 1.0)
        ):
            raise InterfaceDefectTransientError(
                "quasi-steady interface occupancy must remain inside (0, 1)"
            )
        if np.any(self.interface_trace_state_m3 <= 0.0):
            raise InterfaceDefectTransientError(
                "interface trace carrier densities must remain positive"
            )
        iterations = _readonly(self.newton_iterations, dtype=np.int64)
        if iterations.shape != (point_count,) or np.any(iterations < 0):
            raise InterfaceDefectTransientError("newton_iterations is invalid")
        object.__setattr__(self, "newton_iterations", iterations)
        if self.state_coordinate != "qf_interface_logit_potential_local_algebraic":
            raise InterfaceDefectTransientError(
                "unexpected device transient coordinate"
            )
        if self.time_discretization != "backward_euler_index_1_dae":
            raise InterfaceDefectTransientError("unexpected time discretization")


@dataclass(slots=True)
class _LocalState:
    trace_potential: np.ndarray
    log_state: np.ndarray
    state_m3: np.ndarray
    quasi_steady_occupancy: float
    sheet_charge_C_m2: float
    electrostatic_residual: np.ndarray
    tangent: FixedOccupancyCarrierTangent


@dataclass(slots=True)
class _DeviceState:
    coordinate: np.ndarray
    dqfn: np.ndarray
    dqfp: np.ndarray
    n: np.ndarray
    p: np.ndarray
    occupancy: np.ndarray
    phi: np.ndarray
    local: tuple[_LocalState, ...]
    storage: np.ndarray
    rate: np.ndarray
    current_n: np.ndarray
    current_p: np.ndarray
    conduction: np.ndarray
    sheet_charge: np.ndarray
    poisson_residual: np.ndarray
    local_residual: np.ndarray
    storage_jacobian: sparse.csr_matrix
    rate_jacobian: sparse.csr_matrix
    poisson_jacobian: sparse.csr_matrix
    local_jacobian: sparse.csr_matrix


@dataclass(slots=True)
class _Trace:
    times: np.ndarray
    voltage: np.ndarray
    coordinates: np.ndarray
    states: tuple[_DeviceState, ...]
    displacement: np.ndarray
    total_current: np.ndarray
    interface_conduction: np.ndarray
    interface_displacement: np.ndarray
    interface_total_current: np.ndarray
    integrated_charge: np.ndarray
    iterations: np.ndarray
    maximum_scaled_residual: float
    maximum_poisson_residual: float
    maximum_local_carrier_residual: float
    maximum_local_gauss_residual: float
    maximum_jacobian_error: float
    maximum_charge_balance_absolute_error: float
    maximum_charge_balance_error: float
    maximum_face_spread: float
    maximum_interface_current_error: float
    maximum_operator_error: float
    maximum_nnz: int


class _InterfaceTransientSystem:
    def __init__(
        self,
        grid: np.ndarray,
        stack: DeviceStack,
        material: MaterialArrays,
        dc_state: QuasiFermiSteadyStateResult,
        dark_reference: EquilibriumReferencedInterfaceChargeDarkReference,
        occupancy_reference: np.ndarray,
        dynamic_dc,
        *,
        illuminated: bool,
    ) -> None:
        self.grid = grid
        self.stack = stack
        self.material = material
        self.dc_state = dc_state
        self.dark_reference = dark_reference
        self.illuminated = bool(illuminated)
        self.node_count = grid.size
        self.interior_count = grid.size - 2
        self.interface_count = len(dark_reference.trap_density_m2)
        self.trap_density = np.asarray(dark_reference.trap_density_m2, dtype=float)
        self.equilibrium_occupancy = np.asarray(
            dark_reference.equilibrium_occupancy,
            dtype=float,
        )
        self.dimension = 3 * self.interior_count + 7 * self.interface_count
        self.electron_slice = slice(0, self.interior_count)
        self.hole_slice = slice(self.interior_count, 2 * self.interior_count)
        self.trap_slice = slice(
            2 * self.interior_count,
            2 * self.interior_count + self.interface_count,
        )
        self.potential_slice = slice(
            self.trap_slice.stop,
            self.trap_slice.stop + self.interior_count,
        )
        self.local_slice = slice(
            self.potential_slice.stop,
            self.potential_slice.stop + 6 * self.interface_count,
        )
        self.system = _QuasiFermiSystem(
            grid,
            stack,
            material,
            float(dc_state.V_app),
            interface_boundary=True,
            interface_topology=TWO_SIDED_TRACE,
            interface_transmission=dark_reference.interface_transmission,
            interface_transport_model=FERMI_DIRAC_RICHARDSON,
            interface_charge_reference_occupancy=self.equilibrium_occupancy,
            interface_charge_trap_density_m2=self.trap_density,
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
            raise InterfaceDefectTransientError(
                "DC QF reference does not match the transient operator"
            )
        self.reference_n = np.asarray(dynamic_dc.y[: self.node_count], dtype=float)
        self.reference_p = np.asarray(
            dynamic_dc.y[self.node_count : 2 * self.node_count],
            dtype=float,
        )
        self.reference_phi = np.asarray(dynamic_dc.phi, dtype=float)
        self.reference_occupancy = np.asarray(occupancy_reference, dtype=float)
        self.reference_logit = _occupancy_logit(self.reference_occupancy)
        self.thermal_voltage = float(material.V_T_device)
        self.polarity = float(material.junction_polarity)
        self.eps_face = EPS_0 * _harmonic_face_average(material.eps_r)
        self.widths = np.asarray(material.dx_cell, dtype=float)
        self.interface_faces = tuple(
            int(value) for value in material.iface_qss_interface_faces
        )
        self.left_nodes = tuple(int(value) for value in material.iface_qss_left_nodes)
        self.right_nodes = tuple(int(value) for value in material.iface_qss_right_nodes)
        if not (
            len(self.interface_faces)
            == len(self.left_nodes)
            == len(self.right_nodes)
            == self.interface_count
        ):
            raise InterfaceDefectTransientError(
                "two-sided interface topology is not aligned with trap populations"
            )
        if any(
            left <= 0
            or right >= self.node_count - 1
            or right != left + 1
            or face != left
            for left, right, face in zip(
                self.left_nodes,
                self.right_nodes,
                self.interface_faces,
            )
        ):
            raise InterfaceDefectTransientError(
                "two-sided transient requires adjacent interior bulk reservoirs"
            )
        self._divergence = self._build_divergence_matrix()
        self._poisson_laplacian = self._build_poisson_laplacian()
        self.reference_trace_potential = np.empty((self.interface_count, 2))
        self.reference_trace_log_state = np.empty((self.interface_count, 4))
        self.reference_local_scale = np.empty((self.interface_count, 6))
        aggregate = dynamic_dc.interface_charge_dynamic
        if aggregate is None or aggregate.qss.capture_flux_m2_s is None:
            raise InterfaceDefectTransientError(
                "DC embedding lacks fixed-occupancy interface evidence"
            )
        for index in range(self.interface_count):
            geometry, physics, bulk = _material_two_sided_interface_problem(
                material,
                stack,
                self.reference_n,
                self.reference_p,
                self.reference_phi,
                index,
                cross_transmission=dark_reference.interface_transmission,
            )
            potential_jump = float(geometry.potential_jump_right_minus_left_V)
            barrier_steps = (
                float(physics.conduction_band_step_eV) - potential_jump,
                float(physics.hole_transport_step_eV) + potential_jump,
            )
            clamp_tolerance = (
                128.0
                * np.finfo(float).eps
                * max(
                    1.0,
                    abs(float(physics.conduction_band_step_eV)),
                    abs(float(physics.hole_transport_step_eV)),
                    abs(potential_jump),
                )
            )
            if any(abs(value) <= clamp_tolerance for value in barrier_steps):
                raise InterfaceDefectTransientError(
                    "analytic interface transient excludes a cross-node barrier "
                    "clamp switching boundary"
                )
            base = 4 * index
            canonical_seed = np.asarray(
                aggregate.qss.state_m3[base : base + 4],
                dtype=float,
            )[_RIGHT_FIRST]
            local = solve_fixed_occupancy_two_sided_interface(
                geometry,
                physics,
                bulk,
                self.reference_occupancy[index],
                EquilibriumReferencedSheetCharge(
                    self.equilibrium_occupancy[index],
                    self.trap_density[index],
                ),
                initial_state_m3=canonical_seed,
            )
            trace = local.qss.potentials
            reference_balance = fixed_occupancy_carrier_tangent(
                np.log(local.qss.state_m3),
                replace(
                    geometry,
                    fixed_sheet_charge_C_m2=(
                        float(geometry.fixed_sheet_charge_C_m2)
                        + float(local.incremental_sheet_charge_C_m2)
                    ),
                ),
                physics,
                bulk,
                self.reference_occupancy[index],
                trace,
            ).balance
            self.reference_trace_potential[index] = (
                trace.phi_left_V,
                trace.phi_right_V,
            )
            self.reference_trace_log_state[index] = np.log(local.qss.state_m3)
            capacitance = EPS_0 * (
                geometry.eps_r_left / geometry.left_distance_m
                + geometry.eps_r_right / geometry.right_distance_m
            )
            electron_scale = max(
                abs(local.qss.bulk_flux_m2_s[0]),
                abs(local.qss.bulk_flux_m2_s[2]),
                abs(local.qss.capture_flux_m2_s[0]),
                abs(local.qss.capture_flux_m2_s[2]),
                reference_balance.one_way_cross_scale_m2_s[0],
                abs(reference_balance.jacobian_log_state_m2_s[0, 0])
                * np.finfo(float).eps,
                1.0,
            )
            hole_scale = max(
                abs(local.qss.bulk_flux_m2_s[1]),
                abs(local.qss.bulk_flux_m2_s[3]),
                abs(local.qss.capture_flux_m2_s[1]),
                abs(local.qss.capture_flux_m2_s[3]),
                reference_balance.one_way_cross_scale_m2_s[1],
                abs(reference_balance.jacobian_log_state_m2_s[1, 1])
                * np.finfo(float).eps,
                1.0,
            )
            self.reference_local_scale[index] = (
                self.thermal_voltage,
                max(
                    capacitance * self.thermal_voltage,
                    Q * self.trap_density[index],
                    np.finfo(float).tiny,
                ),
                electron_scale,
                hole_scale,
                electron_scale,
                hole_scale,
            )

    def _build_divergence_matrix(self) -> sparse.csr_matrix:
        rows: list[int] = []
        columns: list[int] = []
        values: list[float] = []
        for face in range(self.node_count - 1):
            rows.extend((face, face + 1))
            columns.extend((face, face))
            values.extend((1.0, -1.0))
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

    def _local_block_slice(self, index: int) -> slice:
        start = self.local_slice.start + 6 * index
        return slice(start, start + 6)

    def _coordinates(
        self,
        coordinate: np.ndarray,
        voltage: float,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        values = np.asarray(coordinate, dtype=float)
        if values.shape != (self.dimension,) or not np.all(np.isfinite(values)):
            raise InterfaceDefectTransientError(
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
            raise InterfaceDefectTransientError("carrier coordinate overflow")
        n = np.exp(log_n)
        p = np.exp(log_p)
        if (
            not np.all(np.isfinite(n))
            or not np.all(np.isfinite(p))
            or np.any(n <= 0.0)
            or np.any(p <= 0.0)
        ):
            raise InterfaceDefectTransientError(
                "carrier coordinate produced non-positive or non-finite density"
            )
        occupancy = _occupancy_from_increment(
            self.reference_logit,
            values[self.trap_slice],
        )
        trace_potential = np.empty((self.interface_count, 2))
        trace_log_state = np.empty((self.interface_count, 4))
        trace_state = np.empty((self.interface_count, 4))
        for index in range(self.interface_count):
            block = values[self._local_block_slice(index)]
            trace_potential[index] = self.reference_trace_potential[index] + (
                self.thermal_voltage * block[:2]
            )
            trace_log_state[index] = self.reference_trace_log_state[index] + block[2:]
            if np.any(trace_log_state[index] > limit):
                raise InterfaceDefectTransientError(
                    "interface trace-density coordinate overflow"
                )
            trace_state[index] = np.exp(trace_log_state[index])
        if not np.all(np.isfinite(trace_state)) or np.any(trace_state <= 0.0):
            raise InterfaceDefectTransientError(
                "interface trace coordinate produced non-positive density"
            )
        return (
            dqfn,
            dqfp,
            phi,
            n,
            p,
            occupancy,
            trace_potential,
            trace_log_state,
        )

    def _sheet_weights(self, index: int) -> tuple[float, float]:
        left = self.left_nodes[index]
        right = self.right_nodes[index]
        capacitance_left = (
            EPS_0
            * float(self.material.eps_r[left])
            / float(self.material.iface_qss_left_distances_m[index])
        )
        capacitance_right = (
            EPS_0
            * float(self.material.eps_r[right])
            / float(self.material.iface_qss_right_distances_m[index])
        )
        total = capacitance_left + capacitance_right
        return capacitance_left / total, capacitance_right / total

    def _local_states(
        self,
        n: np.ndarray,
        p: np.ndarray,
        phi: np.ndarray,
        occupancy: np.ndarray,
        trace_potential: np.ndarray,
        trace_log_state: np.ndarray,
    ) -> tuple[tuple[_LocalState, ...], TwoSidedMaterialQSSResult]:
        states: list[_LocalState] = []
        size = 4 * self.interface_count
        state = np.empty(size)
        bulk_flux = np.empty(size)
        cross_flux = np.empty(size)
        capture_flux = np.empty(size)
        residual = np.empty(size)
        maximum_residual = 0.0
        for index in range(self.interface_count):
            geometry, physics, bulk = _material_two_sided_interface_problem(
                self.material,
                self.stack,
                n,
                p,
                phi,
                index,
                cross_transmission=self.dark_reference.interface_transmission,
            )
            sheet_charge = (
                -Q
                * self.trap_density[index]
                * (occupancy[index] - self.equilibrium_occupancy[index])
            )
            charged_geometry = replace(
                geometry,
                fixed_sheet_charge_C_m2=(
                    float(geometry.fixed_sheet_charge_C_m2) + sheet_charge
                ),
            )
            traces = InterfaceTracePotentials(*trace_potential[index])
            electrostatic, _ = electrostatic_trace_residual_and_jacobian(
                trace_potential[index],
                charged_geometry,
                bulk,
            )
            tangent = fixed_occupancy_carrier_tangent(
                trace_log_state[index],
                charged_geometry,
                physics,
                bulk,
                occupancy[index],
                traces,
            )
            balance = tangent.balance
            base = 4 * index
            state[base : base + 4] = balance.state_m3[_RIGHT_FIRST]
            bulk_flux[base : base + 4] = balance.bulk_flux_m2_s[_RIGHT_FIRST]
            cross_flux[base : base + 4] = balance.cross_flux_m2_s[_RIGHT_FIRST]
            capture_flux[base : base + 4] = balance.capture_flux_m2_s[_RIGHT_FIRST]
            residual[base : base + 4] = balance.residual_m2_s[_RIGHT_FIRST]
            maximum_residual = max(
                maximum_residual,
                float(
                    np.max(
                        np.abs(balance.residual_m2_s)
                        / self.reference_local_scale[index, 2:]
                    )
                ),
            )
            states.append(
                _LocalState(
                    trace_potential=np.asarray(trace_potential[index]).copy(),
                    log_state=np.asarray(trace_log_state[index]).copy(),
                    state_m3=np.asarray(balance.state_m3).copy(),
                    quasi_steady_occupancy=shared_trap_occupancy(
                        balance.state_m3,
                        physics,
                    ),
                    sheet_charge_C_m2=float(sheet_charge),
                    electrostatic_residual=np.asarray(electrostatic).copy(),
                    tangent=tangent,
                )
            )
        return tuple(states), TwoSidedMaterialQSSResult(
            state_m3=state,
            bulk_flux_m2_s=bulk_flux,
            cross_flux_m2_s=cross_flux,
            state_flux_m2_s=residual,
            normalized_residual=maximum_residual,
            evaluations=0,
            transport_model=FERMI_DIRAC_RICHARDSON,
            capture_flux_m2_s=capture_flux,
            occupancy=np.asarray(occupancy).copy(),
        )

    def _source(
        self,
        n: np.ndarray,
        p: np.ndarray,
        phi: np.ndarray,
        voltage: float,
        interface_qss: TwoSidedMaterialQSSResult,
    ) -> np.ndarray:
        y = self.system.base.copy()
        y[: self.node_count] = n
        y[self.node_count : 2 * self.node_count] = p
        source = assemble_rhs(
            0.0,
            y,
            self.grid,
            self.stack,
            self.system.source_mat,
            illuminated=False,
            V_app=float(voltage),
            phi_frozen=phi,
            interface_qss_result=interface_qss,
        )[: 2 * self.node_count]
        source += (1.0 if self.illuminated else 0.0) * self.system.generation
        return source

    def _currents(
        self,
        dqfn: np.ndarray,
        dqfp: np.ndarray,
        phi: np.ndarray,
        n: np.ndarray,
        p: np.ndarray,
        local: tuple[_LocalState, ...],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
        transport_n = current_n.copy()
        transport_p = current_p.copy()
        reported_n = current_n.copy()
        reported_p = current_p.copy()
        for index, face in enumerate(self.interface_faces):
            balance = local[index].tangent.balance
            transport_n[face] = 0.0
            transport_p[face] = 0.0
            reported_n[face] = -Q * balance.bulk_flux_m2_s[0]
            reported_p[face] = Q * balance.bulk_flux_m2_s[1]
        return transport_n, transport_p, reported_n, reported_p

    def interface_current_sides(
        self,
        state: _DeviceState,
        previous: _DeviceState | None = None,
        dt: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return left/right conduction, displacement, and total currents."""
        conduction = np.empty((self.interface_count, 2), dtype=float)
        displacement = np.zeros((self.interface_count, 2), dtype=float)
        if (previous is None) != (dt is None):
            raise InterfaceDefectTransientError(
                "previous state and dt must be supplied together"
            )
        if dt is not None and (not np.isfinite(dt) or dt <= 0.0):
            raise InterfaceDefectTransientError("interface current dt is invalid")
        for index, (left, right, item) in enumerate(
            zip(self.left_nodes, self.right_nodes, state.local)
        ):
            flux = item.tangent.balance.bulk_flux_m2_s
            conduction[index] = (
                self.polarity
                * Q
                * np.array(
                    [-flux[0] + flux[1], flux[2] - flux[3]],
                    dtype=float,
                )
            )
            if previous is not None and dt is not None:
                capacitance_left = (
                    EPS_0
                    * float(self.material.eps_r[left])
                    / float(self.material.iface_qss_left_distances_m[index])
                )
                capacitance_right = (
                    EPS_0
                    * float(self.material.eps_r[right])
                    / float(self.material.iface_qss_right_distances_m[index])
                )
                trace_increment = (
                    item.trace_potential - previous.local[index].trace_potential
                )
                bulk_increment = np.array(
                    [
                        state.phi[left] - previous.phi[left],
                        state.phi[right] - previous.phi[right],
                    ]
                )
                drop_increment = trace_increment - bulk_increment
                displacement[index] = self.polarity * np.array(
                    [
                        -capacitance_left * drop_increment[0] / dt,
                        capacitance_right * drop_increment[1] / dt,
                    ],
                    dtype=float,
                )
        return conduction, displacement, conduction + displacement

    def evaluate(self, coordinate: np.ndarray, voltage: float) -> _DeviceState:
        (
            dqfn,
            dqfp,
            phi,
            n,
            p,
            occupancy,
            trace_potential,
            trace_log_state,
        ) = self._coordinates(coordinate, voltage)
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
        divergence_n = self._divergence @ transport_n
        divergence_p = self._divergence @ transport_p
        rate_n = source[: self.node_count] + divergence_n / (Q * self.widths)
        rate_p = source[self.node_count :] - divergence_p / (Q * self.widths)
        capture = np.asarray([item.tangent.balance.capture_flux_m2_s for item in local])
        trap_rate = capture[:, [0, 2]].sum(axis=1) - capture[:, [1, 3]].sum(axis=1)
        storage = np.r_[
            n[1:-1],
            p[1:-1],
            self.trap_density * occupancy,
        ]
        rate = np.r_[rate_n[1:-1], rate_p[1:-1], trap_rate]
        rho, _ = self.system._bulk_space_charge_and_tangent(n, p)
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
        (
            storage_jacobian,
            rate_jacobian,
            poisson_jacobian,
            local_jacobian,
        ) = self._jacobians(phi, n, p, occupancy, local)
        conduction = self.polarity * (current_n + current_p)
        arrays = (storage, rate, poisson, local_residual, conduction)
        if any(not np.all(np.isfinite(value)) for value in arrays):
            raise InterfaceDefectTransientError(
                "interface transient operator produced a non-finite value"
            )
        return _DeviceState(
            coordinate=np.asarray(coordinate, dtype=float).copy(),
            dqfn=dqfn,
            dqfp=dqfp,
            n=n,
            p=p,
            occupancy=occupancy,
            phi=phi,
            local=local,
            storage=storage,
            rate=rate,
            current_n=current_n,
            current_p=current_p,
            conduction=conduction,
            sheet_charge=sheet_charge,
            poisson_residual=np.asarray(poisson),
            local_residual=local_residual,
            storage_jacobian=storage_jacobian,
            rate_jacobian=rate_jacobian,
            poisson_jacobian=poisson_jacobian,
            local_jacobian=local_jacobian,
        )

    def _bulk_coordinate_chain(
        self,
        left: int,
        right: int,
    ) -> sparse.csr_matrix:
        chain = sparse.lil_matrix((6, self.dimension))
        for side, node in enumerate((left, right)):
            local = node - 1
            phi_row = side
            n_row = 2 + 2 * side
            p_row = n_row + 1
            electron_column = self.electron_slice.start + local
            hole_column = self.hole_slice.start + local
            potential_column = self.potential_slice.start + local
            chain[phi_row, potential_column] = self.thermal_voltage
            chain[n_row, electron_column] = 1.0
            chain[n_row, potential_column] = 1.0
            chain[p_row, hole_column] = 1.0
            chain[p_row, potential_column] = -1.0
        return chain.tocsr()

    def _jacobians(
        self,
        phi: np.ndarray,
        n: np.ndarray,
        p: np.ndarray,
        occupancy: np.ndarray,
        local_states: tuple[_LocalState, ...],
    ) -> tuple[
        sparse.csr_matrix,
        sparse.csr_matrix,
        sparse.csr_matrix,
        sparse.csr_matrix,
    ]:
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
        occupancy_jacobian = sparse.lil_matrix((self.interface_count, dimension))
        for index, tangent in enumerate(occupancy_tangent):
            occupancy_jacobian[index, self.trap_slice.start + index] = tangent
        occupancy_jacobian = occupancy_jacobian.tocsr()
        storage_jacobian = sparse.vstack(
            (
                n_jacobian[1:-1],
                p_jacobian[1:-1],
                sparse.diags(self.trap_density) @ occupancy_jacobian,
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
            neutral_bulk_defects=self.system.source_mat.neutral_bulk_defects,
        )
        source_n_jacobian = (
            sparse.diags(-np.asarray(recombination.electron_density_derivative))
            @ n_jacobian
            + sparse.diags(-np.asarray(recombination.hole_density_derivative))
            @ p_jacobian
        ).tolil()
        source_p_jacobian = source_n_jacobian.copy()
        trap_rate_jacobian = sparse.lil_matrix((self.interface_count, dimension))
        local_jacobian = sparse.lil_matrix((6 * self.interface_count, dimension))
        for index, (left, right, item) in enumerate(
            zip(self.left_nodes, self.right_nodes, local_states)
        ):
            tangent = item.tangent
            balance = tangent.balance
            block = self._local_block_slice(index)
            trace_columns = slice(block.start, block.start + 2)
            state_columns = slice(block.start + 2, block.stop)
            bulk_chain = self._bulk_coordinate_chain(left, right)

            bulk_flux_jacobian = (
                sparse.csr_matrix(tangent.bulk_flux_jacobian_bulk_coordinates)
                @ bulk_chain
            ).tolil()
            bulk_flux_jacobian[:, trace_columns] += (
                tangent.bulk_flux_jacobian_trace_potential_m2_s_V * self.thermal_voltage
            )
            bulk_flux_jacobian[:, state_columns] += (
                tangent.bulk_flux_jacobian_log_state_m2_s
            )
            source_n_jacobian[left] -= bulk_flux_jacobian.getrow(0) / self.widths[left]
            source_p_jacobian[left] -= bulk_flux_jacobian.getrow(1) / self.widths[left]
            source_n_jacobian[right] -= (
                bulk_flux_jacobian.getrow(2) / self.widths[right]
            )
            source_p_jacobian[right] -= (
                bulk_flux_jacobian.getrow(3) / self.widths[right]
            )

            capture_jacobian = sparse.lil_matrix((4, dimension))
            capture_jacobian[:, state_columns] = (
                tangent.capture_flux_jacobian_log_state_m2_s
            )
            capture_jacobian[:, self.trap_slice.start + index] = (
                tangent.capture_flux_occupancy_derivative_m2_s
                * occupancy_tangent[index]
            )[:, np.newaxis]
            trap_rate_jacobian[index] = (
                capture_jacobian.getrow(0)
                + capture_jacobian.getrow(2)
                - capture_jacobian.getrow(1)
                - capture_jacobian.getrow(3)
            )

            row = 6 * index
            geometry, _physics, _bulk = _material_two_sided_interface_problem(
                self.material,
                self.stack,
                n,
                p,
                phi,
                index,
                cross_transmission=self.dark_reference.interface_transmission,
            )
            _, electrostatic_trace_jacobian = electrostatic_trace_residual_and_jacobian(
                item.trace_potential,
                replace(
                    geometry,
                    fixed_sheet_charge_C_m2=(
                        geometry.fixed_sheet_charge_C_m2 + item.sheet_charge_C_m2
                    ),
                ),
                _bulk,
            )
            local_jacobian[row : row + 2, trace_columns] = (
                electrostatic_trace_jacobian * self.thermal_voltage
            )
            capacitance_left = EPS_0 * geometry.eps_r_left / geometry.left_distance_m
            capacitance_right = EPS_0 * geometry.eps_r_right / geometry.right_distance_m
            local_jacobian[
                row + 1,
                self.potential_slice.start + left - 1,
            ] = -capacitance_left * self.thermal_voltage
            local_jacobian[
                row + 1,
                self.potential_slice.start + right - 1,
            ] = -capacitance_right * self.thermal_voltage
            local_jacobian[
                row + 1,
                self.trap_slice.start + index,
            ] = Q * self.trap_density[index] * occupancy_tangent[index]

            carrier_jacobian = (
                sparse.csr_matrix(balance.jacobian_bulk_coordinates) @ bulk_chain
            ).tolil()
            carrier_jacobian[:, trace_columns] += (
                balance.jacobian_trace_potential_m2_s_V * self.thermal_voltage
            )
            carrier_jacobian[:, state_columns] += balance.jacobian_log_state_m2_s
            carrier_jacobian[:, self.trap_slice.start + index] += (
                tangent.residual_occupancy_derivative_m2_s * occupancy_tangent[index]
            )[:, np.newaxis]
            local_jacobian[row + 2 : row + 6] = carrier_jacobian

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
            result = (
                sparse.diags(local.density_left_derivative) @ density_jacobian[:-1]
                + sparse.diags(local.density_right_derivative) @ density_jacobian[1:]
                + sparse.diags(local.potential_left_derivative) @ phi_jacobian[:-1]
                + sparse.diags(local.potential_right_derivative) @ phi_jacobian[1:]
            ).tolil()
            for face in self.interface_faces:
                result[face] = 0.0
            return result.tocsr()

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

        charge_jacobian = Q * (p_jacobian - n_jacobian)
        poisson_jacobian = (
            self._poisson_laplacian @ phi_jacobian
            + sparse.diags(self.material.poisson_factor.h_cell) @ charge_jacobian[1:-1]
        ).tolil()
        for index, (left, right) in enumerate(zip(self.left_nodes, self.right_nodes)):
            weight_left, weight_right = self._sheet_weights(index)
            derivative = -Q * self.trap_density[index] * occupancy_tangent[index]
            column = self.trap_slice.start + index
            poisson_jacobian[left - 1, column] += weight_left * derivative
            poisson_jacobian[right - 1, column] += weight_right * derivative
        return (
            storage_jacobian,
            rate_jacobian,
            poisson_jacobian.tocsr(),
            local_jacobian.tocsr(),
        )

    def local_algebraic_scale(
        self,
        policy: InterfaceDefectTransientPolicy,
    ) -> np.ndarray:
        absolute = np.tile(
            np.array(
                [
                    policy.interface_potential_atol_V,
                    policy.interface_gauss_atol_C_m2,
                    policy.interface_flux_atol_m2_s,
                    policy.interface_flux_atol_m2_s,
                    policy.interface_flux_atol_m2_s,
                    policy.interface_flux_atol_m2_s,
                ]
            ),
            (self.interface_count, 1),
        )
        return (
            absolute
            + policy.interface_algebraic_relative_tolerance * self.reference_local_scale
        ).reshape(-1)

    def residual_and_jacobian(
        self,
        coordinate: np.ndarray,
        voltage: float,
        previous_storage: np.ndarray,
        dt: float,
        storage_scale: np.ndarray,
        poisson_scale: np.ndarray,
        local_scale: np.ndarray,
    ) -> tuple[np.ndarray, sparse.csr_matrix, _DeviceState]:
        state = self.evaluate(coordinate, voltage)
        storage_residual = state.storage - previous_storage - dt * state.rate
        residual = np.r_[
            storage_residual / storage_scale,
            state.poisson_residual / poisson_scale,
            state.local_residual / local_scale,
        ]
        jacobian = sparse.vstack(
            (
                sparse.diags(1.0 / storage_scale)
                @ (state.storage_jacobian - dt * state.rate_jacobian),
                sparse.diags(1.0 / poisson_scale) @ state.poisson_jacobian,
                sparse.diags(1.0 / local_scale) @ state.local_jacobian,
            ),
            format="csr",
        )
        return residual, jacobian, state

    def storage_scale(
        self,
        previous_storage: np.ndarray,
        previous_state: _DeviceState,
        dt: float,
        policy: InterfaceDefectTransientPolicy,
    ) -> np.ndarray:
        reference = np.r_[
            self.reference_n[1:-1],
            self.reference_p[1:-1],
            self.trap_density * self.reference_occupancy,
        ]
        absolute = np.r_[
            np.full(2 * self.interior_count, policy.carrier_storage_atol_m3),
            np.full(self.interface_count, policy.interface_storage_atol_m2),
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
        scale += np.asarray(resolvable_storage, dtype=float).reshape(-1)
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

    def poisson_scale(self, policy: InterfaceDefectTransientPolicy) -> np.ndarray:
        factor = self.material.poisson_factor
        reference = (factor.C[:-1] + factor.C[1:]) * self.thermal_voltage
        return policy.poisson_atol_C_m2 + policy.poisson_relative_tolerance * reference

    def local_normalized_residuals(self, state: _DeviceState) -> tuple[float, float]:
        carrier = 0.0
        gauss = 0.0
        for index, item in enumerate(state.local):
            carrier = max(
                carrier,
                float(
                    np.max(
                        np.abs(item.tangent.balance.residual_m2_s)
                        / self.reference_local_scale[index, 2:]
                    )
                ),
            )
            gauss = max(
                gauss,
                abs(float(item.electrostatic_residual[1]))
                / self.reference_local_scale[index, 1],
            )
        return carrier, gauss

    def integrated_charge(self, state: _DeviceState) -> float:
        free = Q * (state.p - state.n)
        return float(
            np.sum(free[1:-1] * self.widths[1:-1]) + np.sum(state.sheet_charge)
        )

    def eliminated_operator_error(
        self,
        state: _DeviceState,
        voltage: float,
    ) -> float:
        eliminated = self.system.evaluate_quasi_fermi_increments_dynamic_interface(
            state.dqfn,
            state.dqfp,
            state.occupancy,
            1.0 if self.illuminated else 0.0,
            V_app=float(voltage),
        )
        fixed = eliminated.interface_charge_dynamic
        if fixed is None or fixed.qss.capture_flux_m2_s is None:
            raise InterfaceDefectTransientError(
                "eliminated comparison lost its fixed-occupancy evidence"
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
            relative(canonical_state, eliminated_state, 1.0),
            relative(canonical_capture, eliminated_capture, 1.0),
            relative(
                state.sheet_charge,
                np.asarray(fixed.incremental_sheet_charge_C_m2),
                Q * float(np.max(self.trap_density)),
            ),
            relative(
                trace_shift,
                np.asarray(fixed.trace_potential_shift_V),
                self.thermal_voltage,
            ),
        )
        return max(values)


def _jacobian_error(
    system: _InterfaceTransientSystem,
    coordinate: np.ndarray,
    voltage: float,
    previous_storage: np.ndarray,
    dt: float,
    storage_scale: np.ndarray,
    poisson_scale: np.ndarray,
    local_scale: np.ndarray,
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
            local_scale,
        )
        minus_residual, _, _ = system.residual_and_jacobian(
            minus,
            voltage,
            previous_storage,
            dt,
            storage_scale,
            poisson_scale,
            local_scale,
        )
        finite = (plus_residual - minus_residual) / (2.0 * step)
        expected = analytic_dense[:, column]
        scale = max(float(np.linalg.norm(expected)), 1.0e-12)
        columns.append(float(np.linalg.norm(finite - expected)) / scale)
    return max(columns, default=0.0)


def _solve_step(
    system: _InterfaceTransientSystem,
    coordinate: np.ndarray,
    previous: _DeviceState,
    voltage: float,
    dt: float,
    policy: InterfaceDefectTransientPolicy,
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
    local_scale = system.local_algebraic_scale(policy)
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
            local_scale,
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
                    local_scale,
                    jacobian,
                    policy.jacobian_check_step,
                )
            return state, iteration - 1, norm, maximum_jacobian_error, maximum_nnz
        with warnings.catch_warnings():
            warnings.simplefilter("error", MatrixRankWarning)
            try:
                step = np.asarray(spsolve(jacobian, -residual), dtype=float)
            except (MatrixRankWarning, RuntimeError, ValueError) as exc:
                raise InterfaceDefectTransientError(
                    f"analytic sparse Newton solve failed: {exc}"
                ) from exc
        if step.shape != trial.shape or not np.all(np.isfinite(step)):
            raise InterfaceDefectTransientError(
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
                    local_scale,
                )
            except (InterfaceDefectTransientError, ValueError, FloatingPointError):
                damping *= 0.5
                continue
            candidate_norm = float(np.max(np.abs(candidate_residual)))
            if candidate_norm < norm * (1.0 - 1.0e-4 * damping):
                trial = candidate
                accepted = True
                break
            damping *= 0.5
        if not accepted:
            raise InterfaceDefectTransientError(
                f"analytic sparse Newton line search stalled at residual {norm:.6g}"
            )
    raise InterfaceDefectTransientError(
        f"analytic sparse Newton exceeded {policy.maximum_newton_iterations} iterations"
    )


def _integrate_trace(
    system: _InterfaceTransientSystem,
    times: np.ndarray,
    voltage: np.ndarray,
    substeps: int,
    policy: InterfaceDefectTransientPolicy,
) -> _Trace:
    coordinate = system.initial_coordinate()
    initial = system.evaluate(coordinate, float(voltage[0]))
    states: list[_DeviceState] = [initial]
    coordinates = [coordinate.copy()]
    displacement = [np.zeros(system.node_count - 1, dtype=float)]
    total_current = [initial.conduction.copy()]
    (
        initial_interface_conduction,
        initial_interface_displacement,
        initial_interface_total,
    ) = system.interface_current_sides(initial)
    interface_conduction = [initial_interface_conduction]
    interface_displacement = [initial_interface_displacement]
    interface_total_current = [initial_interface_total]
    integrated_charge = [system.integrated_charge(initial)]
    iterations = [0]
    local_carrier, local_gauss = system.local_normalized_residuals(initial)
    maximum_scaled_residual = 0.0
    maximum_poisson = float(np.max(np.abs(initial.poisson_residual)))
    maximum_local_carrier = local_carrier
    maximum_local_gauss = local_gauss
    maximum_jacobian_error = 0.0
    maximum_charge_absolute_error = 0.0
    maximum_charge_error = 0.0
    maximum_face_spread = 0.0
    maximum_interface_current_error = 0.0
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
        final_interface_conduction = initial_interface_conduction
        final_interface_displacement = initial_interface_displacement
        final_interface_total = initial_interface_total
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
            field_increment = -np.diff(state.phi - previous_phi) / np.diff(system.grid)
            final_displacement = (
                system.polarity * system.eps_face * field_increment / dt
            )
            (
                final_interface_conduction,
                final_interface_displacement,
                final_interface_total,
            ) = system.interface_current_sides(state, previous, dt)
            for index, face in enumerate(system.interface_faces):
                final_displacement[face] = final_interface_displacement[index, 0]
            final_total = state.conduction + final_displacement
            charge = system.integrated_charge(state)
            previous_charge = (
                integrated_charge[-1]
                if local_step == 0
                else system.integrated_charge(previous)
            )
            storage_rate = system.polarity * (charge - previous_charge) / dt
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
            interface_scale = np.maximum(
                np.max(np.abs(final_interface_total), axis=1),
                1.0e-20,
            )
            interface_current_error = float(
                np.max(
                    np.abs(final_interface_total[:, 0] - final_interface_total[:, 1])
                    / interface_scale
                )
            )
            carrier_residual, gauss_residual = system.local_normalized_residuals(state)
            maximum_scaled_residual = max(maximum_scaled_residual, residual)
            maximum_poisson = max(
                maximum_poisson,
                float(np.max(np.abs(state.poisson_residual))),
            )
            maximum_local_carrier = max(
                maximum_local_carrier,
                carrier_residual,
            )
            maximum_local_gauss = max(maximum_local_gauss, gauss_residual)
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
            maximum_interface_current_error = max(
                maximum_interface_current_error,
                interface_current_error,
            )
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
        interface_conduction.append(final_interface_conduction)
        interface_displacement.append(final_interface_displacement)
        interface_total_current.append(final_interface_total)
        integrated_charge.append(system.integrated_charge(previous))
        iterations.append(point_iterations)
    return _Trace(
        times=times,
        voltage=voltage,
        coordinates=np.asarray(coordinates),
        states=tuple(states),
        displacement=np.asarray(displacement),
        total_current=np.asarray(total_current),
        interface_conduction=np.asarray(interface_conduction),
        interface_displacement=np.asarray(interface_displacement),
        interface_total_current=np.asarray(interface_total_current),
        integrated_charge=np.asarray(integrated_charge),
        iterations=np.asarray(iterations),
        maximum_scaled_residual=maximum_scaled_residual,
        maximum_poisson_residual=maximum_poisson,
        maximum_local_carrier_residual=maximum_local_carrier,
        maximum_local_gauss_residual=maximum_local_gauss,
        maximum_jacobian_error=maximum_jacobian_error,
        maximum_charge_balance_absolute_error=maximum_charge_absolute_error,
        maximum_charge_balance_error=maximum_charge_error,
        maximum_face_spread=maximum_face_spread,
        maximum_interface_current_error=maximum_interface_current_error,
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
    trace_coarse = np.asarray(
        [[item.state_m3 for item in state.local] for state in coarse.states]
    )
    trace_fine = np.asarray(
        [[item.state_m3 for item in state.local] for state in fine.states]
    )
    trace_phi_coarse = np.asarray(
        [[item.trace_potential for item in state.local] for state in coarse.states]
    )
    trace_phi_fine = np.asarray(
        [[item.trace_potential for item in state.local] for state in fine.states]
    )
    carrier_change = max(
        float(np.max(np.abs(np.log(n_coarse / n_fine)))),
        float(np.max(np.abs(np.log(p_coarse / p_fine)))),
        float(np.max(np.abs(np.log(trace_coarse / trace_fine)))),
    )
    occupancy_change = float(np.max(np.abs(f_coarse - f_fine)))
    potential_scale = max(float(np.ptp(phi_fine)), 0.025)
    potential_change = max(
        float(np.max(np.abs(phi_coarse - phi_fine))) / potential_scale,
        float(np.max(np.abs(trace_phi_coarse - trace_phi_fine))) / potential_scale,
    )
    state_change = max(carrier_change, occupancy_change, potential_change)
    current_scale = max(float(np.max(np.abs(fine.total_current[1:]))), 1.0)
    current_change = (
        float(np.max(np.abs(coarse.total_current[1:] - fine.total_current[1:])))
        / current_scale
    )
    return state_change, current_change


def _symmetric_relative_error(left: object, right: object, floor: float = 1.0) -> float:
    left_values = np.asarray(left, dtype=float)
    right_values = np.asarray(right, dtype=float)
    scale = max(
        float(np.max(np.abs(left_values))),
        float(np.max(np.abs(right_values))),
        float(floor),
    )
    return float(np.max(np.abs(left_values - right_values))) / scale


def run_interface_defect_device_transient(
    x: np.ndarray,
    stack: DeviceStack,
    times_s: object,
    voltage_V: object,
    *,
    illuminated: bool = False,
    dark_reference: EquilibriumReferencedInterfaceChargeDarkReference | None = None,
    mat: MaterialArrays | None = None,
    dc_state: QuasiFermiSteadyStateResult | None = None,
    policy: InterfaceDefectTransientPolicy | None = None,
    require_certificate: bool = True,
) -> InterfaceDefectTransientResult:
    """Integrate a bounded shared-occupancy two-sided-interface transient.

    ``voltage_V[0]`` defines the residual-certified DC reference. Every later
    value is applied as a right-continuous step-and-hold over its preceding time
    interval. The local electrostatic and carrier trace states are zero-volume
    algebraic unknowns; only bulk carriers and the areal trap population carry
    time derivatives.
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
    resolved_policy = policy or InterfaceDefectTransientPolicy()
    if not isinstance(resolved_policy, InterfaceDefectTransientPolicy):
        raise TypeError("policy must be an InterfaceDefectTransientPolicy or None")

    try:
        charge_off_stack, microscopic_contract = _research_charge_off_stack(stack)
    except (TypeError, ValueError) as exc:
        raise InterfaceDefectTransientError(
            f"interface transient requires a microscopic charged-interface stack: {exc}"
        ) from exc
    interface_count = len(microscopic_contract.documents)
    if interface_count == 0:
        raise InterfaceDefectTransientError(
            "interface transient requires at least one microscopic interface defect"
        )
    reference = dark_reference
    if reference is None:
        try:
            reference = build_equilibrium_referenced_interface_charge_dark_reference(
                grid,
                stack,
                require_contact_certificate=True,
            )
        except (QuasiFermiSteadyStateError, TypeError, ValueError) as exc:
            raise InterfaceDefectTransientError(
                f"interface transient could not build its dark reference: {exc}"
            ) from exc
    expected_grid_sha = _research_array_sha256("interface-charge-grid-v1", grid)
    expected_stack_sha = hashlib.sha256(repr(stack).encode("utf-8")).hexdigest()
    expected_dark_sha = _research_array_sha256(
        "interface-charge-dark-state-v2",
        grid,
        reference.dark_state.y,
        reference.dark_state.phi,
        reference.dark_state.electron_quasi_fermi_potential_V,
        reference.dark_state.hole_quasi_fermi_potential_V,
        np.asarray(reference.equilibrium_occupancy),
        np.asarray(microscopic_contract.trap_density_m2),
        np.asarray(microscopic_contract.capture_velocities_m_s),
        np.asarray([float(reference.interface_transmission)]),
    )
    dark_reference_certified = bool(
        reference.dark_state.certified
        and reference.dark_state.contact_thermodynamic_status == "certified"
        and reference.grid_sha256 == expected_grid_sha
        and reference.stack_sha256 == expected_stack_sha
        and reference.dark_state_sha256 == expected_dark_sha
    )
    microscopic_binding_certified = bool(
        reference.interface_defect_document_sha256
        == microscopic_contract.document_sha256
        and reference.capture_velocities_m_s
        == microscopic_contract.capture_velocities_m_s
        and reference.trap_density_m2 == microscopic_contract.trap_density_m2
        and len(reference.equilibrium_occupancy) == interface_count
    )
    if not dark_reference_certified or not microscopic_binding_certified:
        raise InterfaceDefectTransientError(
            "dark-reference provenance or microscopic interface binding is invalid"
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
            iface_qss_cross_transmission=float(reference.interface_transmission),
            iface_qss_transport_model=FERMI_DIRAC_RICHARDSON,
            iface_qss_allow_inexact_inner=True,
        )
    try:
        _require_supported(
            material,
            interface_boundary=True,
            interface_topology=TWO_SIDED_TRACE,
        )
    except QuasiFermiSteadyStateError as exc:
        raise InterfaceDefectTransientError(
            f"interface transient material capability gate failed: {exc}"
        ) from exc
    if material.monovalent_bulk_defects is not None:
        raise InterfaceDefectTransientError(
            "D6-E2 excludes combined bulk and interface defects"
        )
    if len(material.iface_qss_interface_faces) != interface_count:
        raise InterfaceDefectTransientError(
            "prepared two-sided geometry does not match microscopic interfaces"
        )

    operating_point = dc_state
    if operating_point is None:
        try:
            operating_point = (
                solve_equilibrium_referenced_interface_charge_steady_state(
                    grid,
                    stack,
                    float(voltage[0]),
                    dark_reference=reference,
                    illuminated=bool(illuminated),
                    require_contact_certificate=True,
                )
            )
        except (QuasiFermiSteadyStateError, TypeError, ValueError) as exc:
            raise InterfaceDefectTransientError(
                f"interface transient could not certify its DC state: {exc}"
            ) from exc
    if not operating_point.certified:
        raise InterfaceDefectTransientError(
            "interface transient requires a certified QF DC state"
        )
    if operating_point.contact_thermodynamic_status != "certified":
        raise InterfaceDefectTransientError(
            "interface transient requires contact-thermodynamic certification"
        )
    if not np.isclose(
        operating_point.V_app,
        voltage[0],
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise InterfaceDefectTransientError(
            "DC-state voltage does not match voltage_V[0]"
        )
    if bool(operating_point.illuminated) != bool(illuminated):
        raise InterfaceDefectTransientError("DC-state illumination does not match")
    if (
        not operating_point.interface_boundary
        or operating_point.interface_topology != TWO_SIDED_TRACE
        or operating_point.interface_transport_model != FERMI_DIRAC_RICHARDSON
        or operating_point.interface_charge_closure != "equilibrium_referenced"
    ):
        raise InterfaceDefectTransientError(
            "DC state lacks the charged two-sided-interface contract"
        )
    provenance_pairs = (
        (
            operating_point.interface_charge_reference_grid_sha256,
            reference.grid_sha256,
        ),
        (
            operating_point.interface_charge_reference_stack_sha256,
            reference.stack_sha256,
        ),
        (
            operating_point.interface_charge_reference_dark_state_sha256,
            reference.dark_state_sha256,
        ),
    )
    if any(actual != expected for actual, expected in provenance_pairs):
        raise InterfaceDefectTransientError(
            "DC-state interface-charge provenance does not match the dark reference"
        )

    qf_system = _QuasiFermiSystem(
        grid,
        charge_off_stack,
        material,
        float(voltage[0]),
        interface_boundary=True,
        interface_topology=TWO_SIDED_TRACE,
        interface_transmission=reference.interface_transmission,
        interface_transport_model=FERMI_DIRAC_RICHARDSON,
        interface_charge_reference_occupancy=np.asarray(
            reference.equilibrium_occupancy,
            dtype=float,
        ),
        interface_charge_trap_density_m2=np.asarray(
            reference.trap_density_m2,
            dtype=float,
        ),
        poisson_tolerance_V=1.0e-13,
        poisson_max_iterations=100,
    )
    dqfn_dc = np.asarray(operating_point.electron_quasi_fermi_increment_V)
    dqfp_dc = np.asarray(operating_point.hole_quasi_fermi_increment_V)
    illumination_fraction = 1.0 if illuminated else 0.0
    qss_dc = qf_system._evaluate_increments(
        dqfn_dc,
        dqfp_dc,
        illumination_fraction,
        V_app=float(voltage[0]),
    )
    if qss_dc.interface_charge_qss is None:
        raise InterfaceDefectTransientError(
            "fresh DC evaluation lacks charged interface evidence"
        )
    occupancy_dc = np.asarray(qss_dc.interface_charge_qss.qss.occupancy, dtype=float)
    _occupancy_logit(occupancy_dc)
    dynamic_dc = qf_system.evaluate_quasi_fermi_increments_dynamic_interface(
        dqfn_dc,
        dqfp_dc,
        occupancy_dc,
        illumination_fraction,
        V_app=float(voltage[0]),
    )
    fixed_dc = dynamic_dc.interface_charge_dynamic
    if fixed_dc is None or fixed_dc.qss.capture_flux_m2_s is None:
        raise InterfaceDefectTransientError(
            "fixed-occupancy DC embedding lacks local interface evidence"
        )
    widths = np.asarray(material.dx_cell, dtype=float)
    current_scale = max(
        float(np.max(np.abs(qss_dc.current_n + qss_dc.current_p))),
        abs(Q * float(stack.Phi)),
        1.0,
    )
    operating_y = np.asarray(operating_point.y, dtype=float)[: 2 * grid.size]
    operator_match = max(
        _symmetric_relative_error(operating_y, qss_dc.y[: 2 * grid.size]),
        float(np.max(np.abs(np.asarray(operating_point.phi) - qss_dc.phi)))
        / material.V_T_device,
        float(
            np.max(
                np.abs(
                    np.asarray(operating_point.electron_face_current_A_m2)
                    - qss_dc.current_n
                )
            )
        )
        / current_scale,
        float(
            np.max(
                np.abs(
                    np.asarray(operating_point.hole_face_current_A_m2)
                    - qss_dc.current_p
                )
            )
        )
        / current_scale,
        _symmetric_relative_error(
            np.asarray(operating_point.interface_occupancy),
            occupancy_dc,
        ),
    )
    rate_difference_A_m2 = Q * float(
        np.sum(
            (
                np.abs(dynamic_dc.rate_n - qss_dc.rate_n)
                + np.abs(dynamic_dc.rate_p - qss_dc.rate_p)
            )
            * widths
        )
    )
    qss_embedding_error = max(
        operator_match,
        float(np.max(np.abs(dynamic_dc.phi - qss_dc.phi))) / material.V_T_device,
        rate_difference_A_m2 / current_scale,
        float(np.max(np.abs(dynamic_dc.current_n - qss_dc.current_n))) / current_scale,
        float(np.max(np.abs(dynamic_dc.current_p - qss_dc.current_p))) / current_scale,
        _symmetric_relative_error(
            fixed_dc.qss.state_m3,
            qss_dc.interface_charge_qss.qss.state_m3,
        ),
        _symmetric_relative_error(
            fixed_dc.incremental_sheet_charge_C_m2,
            qss_dc.interface_charge_qss.incremental_sheet_charge_C_m2,
            Q * float(np.max(np.asarray(reference.trap_density_m2))),
        ),
    )
    system = _InterfaceTransientSystem(
        grid,
        charge_off_stack,
        material,
        operating_point,
        reference,
        occupancy_dc,
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
            "carrier_interface_charge_balance_failed",
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
    certificate = InterfaceDefectTransientCertificate(
        dc_operating_point_certified=True,
        dark_reference_certified=dark_reference_certified,
        microscopic_binding_certified=microscopic_binding_certified,
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
        maximum_refinement_state_change=refinement_state,
        maximum_refinement_current_relative_change=refinement_current,
        analytic_jacobian_nnz=final.maximum_nnz,
        dense_jacobian_entries=dense_entries,
        sparse_linear_solver_used=True,
        clipping_used=False,
        certified=not reasons,
        reasons=tuple(reasons),
    )
    local_states = np.asarray(
        [[item.state_m3 for item in state.local] for state in final.states]
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
    result = InterfaceDefectTransientResult(
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
        electrostatic_potential_V=np.asarray([state.phi for state in final.states]),
        interface_trace_potential_V=np.asarray(
            [[item.trace_potential for item in state.local] for state in final.states]
        ),
        interface_trace_state_m3=local_states,
        interface_sheet_charge_C_m2=np.asarray(
            [state.sheet_charge for state in final.states]
        ),
        electron_capture_flux_m2_s=capture[:, :, [0, 2]],
        hole_capture_flux_m2_s=capture[:, :, [1, 3]],
        electron_bulk_flux_m2_s=bulk_flux[:, :, [0, 2]],
        hole_bulk_flux_m2_s=bulk_flux[:, :, [1, 3]],
        conduction_current_faces_A_m2=np.asarray(
            [state.conduction for state in final.states]
        ),
        displacement_current_faces_A_m2=final.displacement,
        total_current_faces_A_m2=final.total_current,
        interface_conduction_current_A_m2=final.interface_conduction,
        interface_displacement_current_A_m2=final.interface_displacement,
        interface_total_current_A_m2=final.interface_total_current,
        integrated_free_and_interface_charge_C_m2=final.integrated_charge,
        newton_iterations=final.iterations,
        dc_state=operating_point,
        dark_reference=reference,
        policy=resolved_policy,
        certificate=certificate,
    )
    if require_certificate and not certificate.certified:
        raise InterfaceDefectTransientCertificationError(
            "interface defect device transient did not certify: "
            + ", ".join(certificate.reasons),
            result,
        )
    return result


__all__ = [
    "INTERFACE_DEFECT_TRANSIENT_SCOPE",
    "INTERFACE_DEFECT_TRANSIENT_VERSION",
    "InterfaceDefectTransientCertificate",
    "InterfaceDefectTransientCertificationError",
    "InterfaceDefectTransientError",
    "InterfaceDefectTransientPolicy",
    "InterfaceDefectTransientResult",
    "run_interface_defect_device_transient",
]
