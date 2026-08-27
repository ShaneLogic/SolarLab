"""Opt-in quasi-Fermi-potential steady-state solver.

This solver is deliberately narrower than the general transient and steady-
state drivers. It targets cancellation-sensitive steady states and, through
an explicit opt-in, abrupt heterointerfaces represented by a locally
eliminated interface plane.

The nonlinear carrier unknowns are electron and hole quasi-Fermi-potential
increments.  Electrostatic potential is eliminated by an accurately converged
Poisson-Boltzmann solve at every residual evaluation.  Face currents are then
evaluated with ``expm1`` identities in terms of quasi-Fermi differences.  The
result is returned only after independent physical certificates pass; a Newton
termination condition alone is never treated as a steady-state certificate.

The solver is available through the explicit ``quasi_fermi`` J-V driver.
Unsupported non-local or contact models fail before Newton starts; the
interface-plane boundary remains default-off.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from typing import Literal

import numpy as np
from scipy.linalg import solve_banded

from perovskite_sim.constants import EPS_0, Q
from perovskite_sim.discretization.fe_operators import (
    bernoulli,
)
from perovskite_sim.discretization.grid import Layer, multilayer_grid
from perovskite_sim.experiments.jv_sweep import (
    JVMetrics,
    compute_metrics,
    thermodynamic_voc_ceiling,
)
from perovskite_sim.models.device import DeviceStack, electrical_layers
from perovskite_sim.models.defects import ACCEPTOR, DONOR, EXPLICIT_QUASI_STEADY
from perovskite_sim.physics.contacts import (
    ContactThermodynamicCertificate,
    ContactThermodynamicError,
    require_contact_thermodynamic_certificate,
)
from perovskite_sim.physics.defect_closure import (
    MonovalentBulkDefectEvaluation,
    evaluate_monovalent_bulk_defects,
    solve_monovalent_defect_charge_neutrality,
)
from perovskite_sim.physics.interface_plane import (
    FERMI_DIRAC_RICHARDSON,
    FERMI_RICHARDSON,
    validate_interface_transport_model,
)
from perovskite_sim.physics.poisson import factor_poisson_from_finite_volume
from perovskite_sim.physics.two_sided_interface import (
    DEDUPLICATED_QSS,
    EquilibriumReferencedMaterialQSSResult,
    TWO_SIDED_TRACE,
    build_two_sided_interface_stencils,
    remove_shared_interface_nodes,
    validate_interface_topology,
)
from perovskite_sim.solver.mol import (
    EXPLICIT_DEFECT_CHARGE_QF_DC,
    MaterialArrays,
    StateVec,
    _charge_density,
    assemble_rhs,
    build_material_arrays,
    poisson_right_boundary,
)
from perovskite_sim.solver.newton import solve_equilibrium


DEFAULT_ILLUMINATION_STEPS = (
    0.0,
    1.0e-14,
    1.0e-12,
    1.0e-10,
    1.0e-8,
    1.0e-6,
    1.0e-5,
    1.0e-4,
    1.0e-3,
    1.0e-2,
    1.0e-1,
    1.0,
)

INTERFACE_FALLBACK_ILLUMINATION_STEPS = (
    0.0,
    1.0e-14,
    1.0e-13,
    1.0e-12,
    1.0e-11,
    1.0e-10,
    1.0e-9,
    1.0e-8,
    1.0e-7,
    1.0e-6,
    1.0e-5,
    1.0e-4,
    1.0e-3,
    1.0e-2,
    1.0e-1,
    1.0,
)

_MAX_ABS_LOG_DENSITY = 100.0
_INTERFACE_NUMERICAL_RESIDUAL_FLOOR = 1.0e-7
_RESEARCH_INTERFACE_CHARGE_TOKEN = object()


class QuasiFermiSteadyStateError(RuntimeError):
    """The guarded QF solve is unsupported or lacks a physical certificate."""


@dataclass(frozen=True)
class QuasiFermiSteadyStateResult:
    """Certified state and cancellation-safe current/QF diagnostics.

    The optional reference/increment arrays preserve QF differences that can
    be smaller than the resolution of the corresponding absolute potential.
    """

    y: np.ndarray
    phi: np.ndarray
    electron_quasi_fermi_potential_V: np.ndarray
    hole_quasi_fermi_potential_V: np.ndarray
    electron_face_current_A_m2: np.ndarray
    hole_face_current_A_m2: np.ndarray
    total_face_current_A_m2: np.ndarray
    electron_rate_per_s: np.ndarray
    hole_rate_per_s: np.ndarray
    current_A_m2: float
    face_current_spread_A_m2: float
    electron_continuity_bound_A_m2: float
    hole_continuity_bound_A_m2: float
    max_normalized_cell_residual: float
    poisson_residual: float
    poisson_residual_C_m2: float
    illumination_steps: tuple[float, ...]
    newton_iterations: int
    residual_evaluations: int
    V_app: float = 0.0
    illuminated: bool = True
    certified: bool = True
    electron_quasi_fermi_reference_V: np.ndarray | None = None
    hole_quasi_fermi_reference_V: np.ndarray | None = None
    electron_quasi_fermi_increment_V: np.ndarray | None = None
    hole_quasi_fermi_increment_V: np.ndarray | None = None
    electron_quasi_fermi_edge_drop_V: np.ndarray | None = None
    hole_quasi_fermi_edge_drop_V: np.ndarray | None = None
    interface_boundary: bool = False
    interface_transmission: float = 1.0
    interface_transport_model: str = FERMI_RICHARDSON
    interface_topology: str = DEDUPLICATED_QSS
    interface_faces: tuple[int, ...] = ()
    interface_basin_initializations: int = 0
    interface_basin_predictor_failures: int = 0
    interface_basin_predictor_regrids: int = 0
    initial_state_regrids: int = 0
    qf_coordinate_system: str = "nodal_increment"
    edge_coordinate_predictor_used: bool = False
    edge_coordinate_predictor_iterations: int = 0
    interface_local_residual: float = 0.0
    interface_max_state_to_dos: float = 0.0
    numerical_residual_limit: float = 1.0e-10
    interface_charge_closure: str = "off"
    interface_equilibrium_occupancy: tuple[float, ...] = ()
    interface_occupancy: tuple[float, ...] = ()
    interface_incremental_sheet_charge_C_m2: tuple[float, ...] = ()
    interface_trace_potential_shift_V: tuple[tuple[float, float], ...] = ()
    interface_normalized_gauss_residual: tuple[float, ...] = ()
    interface_scaled_local_jacobian_condition: tuple[float, ...] = ()
    bulk_defect_diagnostics: MonovalentBulkDefectEvaluation | None = None
    contact_thermodynamic_status: str | None = None
    contact_fermi_level_span_eV: float | None = None


@dataclass(frozen=True)
class EquilibriumReferencedInterfaceChargeDarkReference:
    """Certified charge-off dark state used by the charged research lane."""

    dark_state: QuasiFermiSteadyStateResult
    equilibrium_occupancy: tuple[float, ...]
    trap_density_m2: tuple[float, ...]
    interface_transmission: float
    grid_sha256: str
    stack_sha256: str
    dark_state_sha256: str


@dataclass(frozen=True)
class QuasiFermiJVSweepResult:
    """Illuminated QF states and extracted metrics on one voltage grid."""

    voltages_V: np.ndarray
    currents_A_m2: np.ndarray
    points: tuple[QuasiFermiSteadyStateResult, ...]
    metrics: JVMetrics
    continuation_bridge_count: int = 0
    minimum_voltage_step_V: float | None = None
    mpp_interpolation: str = "sampled"
    nodal_predictor_fallback_attempts: int = 0
    nodal_predictor_fallback_failures: int = 0

    @property
    def certified(self) -> bool:
        """Whether every retained voltage point has a physical certificate."""
        voltages = np.asarray(self.voltages_V, dtype=float)
        currents = np.asarray(self.currents_A_m2, dtype=float)
        return bool(
            voltages.ndim == 1
            and currents.ndim == 1
            and voltages.shape == currents.shape
            and len(self.points) == voltages.size
            and voltages.size > 0
            and np.all(np.isfinite(voltages))
            and np.all(np.isfinite(currents))
            and all(point.certified for point in self.points)
        )

    @property
    def metrics_certified(self) -> bool:
        """Whether point certificates also span a resolved open circuit."""
        return self.certified and self.metrics.voc_bracketed


@dataclass(frozen=True)
class _Evaluation:
    residual: np.ndarray
    y: np.ndarray
    phi: np.ndarray
    rate_n: np.ndarray
    rate_p: np.ndarray
    current_n: np.ndarray
    current_p: np.ndarray
    poisson_residual: float
    poisson_residual_C_m2: float
    interface_charge_qss: EquilibriumReferencedMaterialQSSResult | None = None


def _pin_mask(node_count: int) -> np.ndarray:
    pin = np.zeros(2 * node_count, dtype=bool)
    pin[[0, node_count - 1, node_count, 2 * node_count - 1]] = True
    return pin


def _density_from_log(log_density: np.ndarray, *, context: str) -> np.ndarray:
    """Exponentiate only inside the audited, unclipped density domain."""
    values = np.asarray(log_density, dtype=float)
    if (
        not np.all(np.isfinite(values))
        or np.any(np.abs(values) > _MAX_ABS_LOG_DENSITY)
    ):
        raise QuasiFermiSteadyStateError(
            f"{context} log-density is outside the audited exponential range "
            f"[-{_MAX_ABS_LOG_DENSITY:g}, {_MAX_ABS_LOG_DENSITY:g}]"
        )
    return np.exp(values)


def _regrid_edge_drops(
    source_grid: np.ndarray,
    target_grid: np.ndarray,
    source_edge_drops: np.ndarray,
) -> np.ndarray:
    """Conservatively map exact face drops through a piecewise-linear QF field."""
    source = np.asarray(source_grid, dtype=float)
    target = np.asarray(target_grid, dtype=float)
    drops = np.asarray(source_edge_drops, dtype=float)
    for name, values in (("source_grid", source), ("target_grid", target)):
        if (
            values.ndim != 1
            or values.size < 2
            or not np.all(np.isfinite(values))
            or np.any(np.diff(values) <= 0.0)
        ):
            raise ValueError(f"{name} must be finite and strictly increasing")
    if drops.shape != (source.size - 1,) or not np.all(np.isfinite(drops)):
        raise ValueError(
            "source_edge_drops must be finite and match source-grid faces"
        )
    endpoint_tolerance = 1.0e-12 * max(
        1.0,
        abs(float(target[-1] - target[0])),
    )
    if (
        abs(float(source[0] - target[0])) > endpoint_tolerance
        or abs(float(source[-1] - target[-1])) > endpoint_tolerance
    ):
        raise ValueError("source and target grids must share endpoints")
    if np.array_equal(source, target):
        return drops.copy()

    coordinate_tolerance = 32.0 * np.finfo(float).eps * max(
        abs(float(source[0])),
        abs(float(source[-1])),
        abs(float(target[0])),
        abs(float(target[-1])),
        abs(float(target[-1] - target[0])),
        np.finfo(float).tiny,
    )
    mapped = np.zeros(target.size - 1, dtype=float)
    source_face = 0
    for target_face, (left, right) in enumerate(
        zip(target[:-1], target[1:])
    ):
        while (
            source_face < source.size - 2
            and source[source_face + 1] <= left + coordinate_tolerance
        ):
            source_face += 1
        cursor = float(left)
        face = source_face
        while cursor < right - coordinate_tolerance:
            if face >= source.size - 1:
                raise ValueError("target grid extends beyond source-grid support")
            overlap_right = min(float(right), float(source[face + 1]))
            if overlap_right <= cursor:
                face += 1
                continue
            mapped[target_face] += drops[face] * (
                (overlap_right - cursor)
                / (source[face + 1] - source[face])
            )
            cursor = overlap_right
            if cursor >= source[face + 1] - coordinate_tolerance:
                face += 1
        source_face = min(face, source.size - 2)

    # The overlap loop is conservative analytically. Put accumulated floating-
    # point summation error on the closing contact face so the total QF drop
    # remains identical to the certified source state.
    mapped[-1] += float(np.sum(drops) - np.sum(mapped))
    return mapped


def _abrupt_interface_faces(mat: MaterialArrays) -> tuple[int, ...]:
    """Map each interface node to its left/right finite-volume face."""
    faces = tuple(int(node) - 1 for node in mat.interface_nodes)
    face_count = len(mat.D_n_face)
    if any(face < 0 or face >= face_count for face in faces):
        raise QuasiFermiSteadyStateError(
            "heterointerface node does not map to an internal transport face"
        )
    if len(set(faces)) != len(faces):
        raise QuasiFermiSteadyStateError(
            "multiple heterointerfaces map to the same transport face"
        )
    return faces


def _interface_positions(stack: DeviceStack) -> tuple[float, ...]:
    layers = electrical_layers(stack)
    return tuple(
        float(value)
        for value in np.cumsum(
            [layer.thickness for layer in layers[:-1]],
            dtype=float,
        )
    )


def build_two_sided_trace_grid(
    x: np.ndarray,
    stack: DeviceStack,
) -> np.ndarray:
    """Remove only shared material-boundary nodes for two-sided trace QSS."""
    return remove_shared_interface_nodes(x, _interface_positions(stack))


def _prepare_two_sided_material(
    x: np.ndarray,
    stack: DeviceStack,
    mat: MaterialArrays,
) -> MaterialArrays:
    """Attach exact interface finite-volume geometry to material arrays."""
    positions = _interface_positions(stack)
    stencils = build_two_sided_interface_stencils(x, positions)
    if any(stencil.shared_boundary_node is not None for stencil in stencils):
        raise QuasiFermiSteadyStateError(
            "two_sided_trace requires a grid without shared interface nodes; "
            "call build_two_sided_trace_grid first"
        )
    if len(stencils) != len(mat.interface_V_partition_2):
        raise QuasiFermiSteadyStateError(
            "two-sided grid interfaces are not aligned with material caches"
        )
    if mat.poisson_factor is None:
        raise QuasiFermiSteadyStateError("material Poisson factor is unavailable")
    cell_widths = np.asarray(mat.dx_cell, dtype=float).copy()
    capacitance = np.asarray(mat.poisson_factor.C, dtype=float).copy()
    faces: list[int] = []
    left_nodes: list[int] = []
    right_nodes: list[int] = []
    left_distances: list[float] = []
    right_distances: list[float] = []
    for stencil in stencils:
        left = int(stencil.left_bulk_node)
        right = int(stencil.right_bulk_node)
        if right != left + 1:
            raise QuasiFermiSteadyStateError(
                "two-sided interface bulk nodes must share one grid face"
            )
        face_width = float(x[right] - x[left])
        half_width = 0.5 * face_width
        cell_widths[left] += float(stencil.left_distance_m) - half_width
        cell_widths[right] += float(stencil.right_distance_m) - half_width
        capacitance[left] = EPS_0 / (
            float(stencil.left_distance_m) / float(mat.eps_r[left])
            + float(stencil.right_distance_m) / float(mat.eps_r[right])
        )
        faces.append(left)
        left_nodes.append(left)
        right_nodes.append(right)
        left_distances.append(float(stencil.left_distance_m))
        right_distances.append(float(stencil.right_distance_m))
    if np.any(~np.isfinite(cell_widths)) or np.any(cell_widths <= 0.0):
        raise QuasiFermiSteadyStateError(
            "two-sided interface produced an invalid carrier control volume"
        )
    poisson_factor = factor_poisson_from_finite_volume(
        capacitance,
        cell_widths[1:-1],
    )
    return replace(
        mat,
        dx_cell=cell_widths,
        poisson_factor=poisson_factor,
        interface_faces=(),
        iface_qss_two_sided_trace=True,
        iface_qss_interface_faces=tuple(faces),
        iface_qss_left_nodes=tuple(left_nodes),
        iface_qss_right_nodes=tuple(right_nodes),
        iface_qss_interface_positions_m=positions,
        iface_qss_left_distances_m=tuple(left_distances),
        iface_qss_right_distances_m=tuple(right_distances),
    )


def _stack_has_charged_explicit_defects(stack: DeviceStack) -> bool:
    return any(
        species.charge_transition in {ACCEPTOR, DONOR}
        for layer in electrical_layers(stack)
        for species in layer.params.bulk_defects
    )


def _build_qf_material(
    grid: np.ndarray,
    stack: DeviceStack,
) -> MaterialArrays:
    if _stack_has_charged_explicit_defects(stack):
        return build_material_arrays(
            grid,
            stack,
            explicit_defect_charge_closure=EXPLICIT_DEFECT_CHARGE_QF_DC,
        )
    return build_material_arrays(grid, stack)


def _require_material_defect_contract(
    stack: DeviceStack,
    mat: MaterialArrays,
) -> None:
    expected = tuple(
        (
            f"layer[{index}]/{layer.name}",
            layer.params.defect_document.sha256,
        )
        for index, layer in enumerate(electrical_layers(stack))
        if layer.params.defect_document is not None
        and layer.params.defect_model == EXPLICIT_QUASI_STEADY
    )
    model = mat.monovalent_bulk_defects
    if _stack_has_charged_explicit_defects(stack):
        if model is None:
            raise QuasiFermiSteadyStateError(
                "charged explicit-defect stack requires a qf_dc material cache"
            )
        actual = tuple(
            (region.identifier, region.document_sha256)
            for region in model.regions
        )
        if actual != expected:
            raise QuasiFermiSteadyStateError(
                "qf_dc material cache does not match the stack defect documents"
            )
    elif model is not None:
        raise QuasiFermiSteadyStateError(
            "qf_dc material cache supplied for a stack without charged defects"
        )


def _require_supported(
    mat: MaterialArrays,
    *,
    interface_boundary: bool = False,
    interface_topology: str = DEDUPLICATED_QSS,
    allow_charged_bulk_defects: bool = False,
) -> None:
    unsupported: list[str] = []
    if mat.monovalent_bulk_defects is not None:
        if not allow_charged_bulk_defects:
            unsupported.append("charged explicit bulk defects outside QF/DC")
    if getattr(mat, "has_dual_ions", False):
        unsupported.append("dual ions")
    if np.any(np.asarray(mat.D_ion_face, dtype=float) != 0.0):
        unsupported.append("mobile ions")
    if np.any(np.asarray(mat.P_ion0, dtype=float) != 0.0):
        unsupported.append("nonzero ionic background")
    if mat.N_iface_state != 0:
        unsupported.append("interface-plane states/charge")
    if mat.has_selective_contacts:
        unsupported.append("selective contacts")
    if mat.has_field_mobility:
        unsupported.append("field-dependent mobility")
    if mat.has_radiative_reabsorption:
        unsupported.append("non-local photon recycling")
    if mat.interface_faces and not interface_boundary:
        unsupported.append("thermionic interface flux")
    has_cross_node_recombination = any(
        (mat.interface_eval_node_n[k] != node)
        or (mat.interface_eval_node_p[k] != node)
        for k, node in enumerate(mat.interface_nodes)
        if k < len(mat.interface_eval_node_n)
        and k < len(mat.interface_eval_node_p)
    )
    if has_cross_node_recombination and not interface_boundary:
        unsupported.append("cross-node interface recombination")
    if interface_boundary:
        interface_faces = (
            tuple(int(value) for value in mat.iface_qss_interface_faces)
            if interface_topology == TWO_SIDED_TRACE
            else _abrupt_interface_faces(mat)
        )
        if not mat.iface_qss_exclusive_transport:
            unsupported.append("local interface-state QSS elimination")
        if not mat.iface_state_physical_offsets:
            unsupported.append("physical interface band-offset convention")
        N_C = getattr(mat, "N_C_physical", None)
        N_V = getattr(mat, "N_V_physical", None)
        if N_C is None or N_V is None:
            unsupported.append("interface effective-density-of-states data")
        else:
            endpoint_pairs = (
                zip(mat.iface_qss_left_nodes, mat.iface_qss_right_nodes)
                if interface_topology == TWO_SIDED_TRACE
                else ((face, face + 1) for face in interface_faces)
            )
            endpoint_values = np.asarray(
                [
                    value
                    for left, right in endpoint_pairs
                    for value in (
                        N_C[left],
                        N_C[right],
                        N_V[left],
                        N_V[right],
                    )
                ],
                dtype=float,
            )
            if np.any(~np.isfinite(endpoint_values)) or np.any(
                endpoint_values <= 0.0
            ):
                unsupported.append("finite positive interface Nc/Nv values")
        if interface_topology == TWO_SIDED_TRACE:
            if not mat.iface_qss_two_sided_trace:
                unsupported.append("prepared two-sided interface geometry")
            if mat.het_recomb_despike > 0.0:
                unsupported.append("two-sided topology with recombination de-spike")
    if unsupported:
        raise QuasiFermiSteadyStateError(
            "quasi-Fermi steady-state solver does not support "
            + ", ".join(unsupported)
        )


def _validate_illumination_steps(
    illuminated: bool,
    values: tuple[float, ...],
) -> tuple[float, ...]:
    if not illuminated:
        return (0.0,)
    stages = tuple(float(value) for value in values)
    if not stages or stages[-1] != 1.0:
        raise ValueError("illuminated continuation must end at 1.0")
    if any(not np.isfinite(value) or not 0.0 <= value <= 1.0 for value in stages):
        raise ValueError("illumination continuation values must lie in [0, 1]")
    if any(right <= left for left, right in zip(stages[:-1], stages[1:])):
        raise ValueError("illumination continuation values must strictly increase")
    return stages


def _defect_aware_neutral_carriers(
    mat: MaterialArrays,
) -> tuple[np.ndarray, np.ndarray]:
    """Return local dark-neutral carriers including compiled defect charge."""

    net = np.asarray(mat.N_D - mat.N_A, dtype=float)
    intrinsic_product = np.asarray(mat.ni_sq, dtype=float)
    discriminant = np.sqrt(net**2 + 4.0 * intrinsic_product)
    electron_majority = 0.5 * (net + discriminant)
    hole_majority = 0.5 * (-net + discriminant)
    with np.errstate(divide="ignore", invalid="ignore"):
        electron = np.where(
            net >= 0.0,
            electron_majority,
            intrinsic_product / hole_majority,
        )
        hole = np.where(
            net >= 0.0,
            intrinsic_product / electron_majority,
            hole_majority,
        )
    model = mat.monovalent_bulk_defects
    if model is not None:
        for region in model.regions:
            node_indices = np.flatnonzero(region.active_nodes)
            doping_pairs = np.column_stack(
                (mat.N_A[node_indices], mat.N_D[node_indices])
            )
            unique_pairs, inverse = np.unique(
                doping_pairs,
                axis=0,
                return_inverse=True,
            )
            local_n = np.empty(node_indices.size, dtype=float)
            local_p = np.empty(node_indices.size, dtype=float)
            for pair_index, (acceptors, donors) in enumerate(unique_pairs):
                closure = solve_monovalent_defect_charge_neutrality(
                    temperature_K=region.temperature_K,
                    band_gap_eV=region.band_gap_eV,
                    effective_conduction_dos_m3=(
                        region.effective_conduction_dos_m3
                    ),
                    effective_valence_dos_m3=(
                        region.effective_valence_dos_m3
                    ),
                    acceptor_density_m3=float(acceptors),
                    donor_density_m3=float(donors),
                    species=region.species,
                ).neutrality
                selected = inverse == pair_index
                local_n[selected] = closure.electron_density_m3
                local_p[selected] = closure.hole_density_m3
            electron[node_indices] = local_n
            hole[node_indices] = local_p
    electron = np.maximum(electron, 1.0)
    hole = np.maximum(hole, 1.0)
    electron[[0, -1]] = (mat.n_L, mat.n_R)
    hole[[0, -1]] = (mat.p_L, mat.p_R)
    return electron, hole


def _transport_balanced_seed(
    x: np.ndarray,
    stack: DeviceStack,
    mat: MaterialArrays,
    V_app: float,
    *,
    poisson_tolerance_V: float,
    poisson_max_iterations: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a Poisson-consistent seed with resistance-weighted QF drops."""
    node_count = len(x)
    if mat.monovalent_bulk_defects is None:
        y_neutral = solve_equilibrium(x, stack)
        n_neutral = np.maximum(y_neutral[:node_count], 1.0)
        p_neutral = np.maximum(y_neutral[node_count : 2 * node_count], 1.0)
    else:
        n_neutral, p_neutral = _defect_aware_neutral_carriers(mat)
    dx = np.diff(x)
    thermal_voltage = mat.V_T_device
    phi_right = poisson_right_boundary(mat, V_app)

    def invariant_profile(
        density: np.ndarray,
        diffusivity: np.ndarray,
        left: float,
        right: float,
    ) -> np.ndarray:
        conductance = diffusivity * np.sqrt(density[:-1] * density[1:])
        resistance = dx / np.maximum(conductance, np.finfo(float).tiny)
        cumulative = np.concatenate(([0.0], np.cumsum(resistance)))
        if not np.isfinite(cumulative[-1]) or cumulative[-1] <= 0.0:
            raise QuasiFermiSteadyStateError(
                "cannot construct a finite transport-balanced seed"
            )
        return left + (right - left) * cumulative / cumulative[-1]

    u_n = invariant_profile(
        n_neutral,
        mat.D_n_face,
        np.log(mat.n_L) - mat.chi[0] / thermal_voltage,
        np.log(mat.n_R)
        - (phi_right + mat.chi[-1]) / thermal_voltage,
    )
    u_p = invariant_profile(
        p_neutral,
        mat.D_p_face,
        np.log(mat.p_L) + (mat.chi[0] + mat.Eg[0]) / thermal_voltage,
        np.log(mat.p_R)
        + (phi_right + mat.chi[-1] + mat.Eg[-1]) / thermal_voltage,
    )

    phi = np.linspace(0.0, phi_right, node_count)
    factor = mat.poisson_factor
    for _ in range(poisson_max_iterations):
        n = _density_from_log(
            u_n + (phi + mat.chi) / thermal_voltage,
            context="transport-balanced electron seed",
        )
        p = _density_from_log(
            u_p - (phi + mat.chi + mat.Eg) / thermal_voltage,
            context="transport-balanced hole seed",
        )
        rho = Q * (p - n + mat.N_D - mat.N_A)
        defect_charge_derivative = np.zeros(node_count, dtype=float)
        if mat.monovalent_bulk_defects is not None:
            defect_evaluation = evaluate_monovalent_bulk_defects(
                n,
                p,
                mat.monovalent_bulk_defects,
            )
            rho = rho + defect_evaluation.total_charge_density_C_m3
            defect_charge_derivative = (
                defect_evaluation.total_charge_derivative_fixed_qf_C_m3_V
            )
        residual = (
            factor.C[:-1] * (phi[:-2] - phi[1:-1])
            + factor.C[1:] * (phi[2:] - phi[1:-1])
            + rho[1:-1] * factor.h_cell
        )
        banded = np.zeros((3, node_count - 2), dtype=float)
        banded[0, 1:] = factor.C[1:-1]
        banded[1] = -(
            factor.C[:-1] + factor.C[1:]
        ) + (
            -Q * (n[1:-1] + p[1:-1]) / thermal_voltage
            + defect_charge_derivative[1:-1]
        ) * factor.h_cell
        banded[2, :-1] = factor.C[1:-1]
        step = solve_banded((1, 1), banded, -residual)
        damping = min(1.0, 0.05 / max(float(np.max(np.abs(step))), np.finfo(float).tiny))
        phi[1:-1] += damping * step
        if float(np.max(np.abs(damping * step))) < poisson_tolerance_V:
            break
    else:
        raise QuasiFermiSteadyStateError(
            "transport-balanced seed Poisson iteration did not converge"
        )

    n = _density_from_log(
        u_n + (phi + mat.chi) / thermal_voltage,
        context="transport-balanced electron seed",
    )
    p = _density_from_log(
        u_p - (phi + mat.chi + mat.Eg) / thermal_voltage,
        context="transport-balanced hole seed",
    )
    n[[0, -1]] = (mat.n_L, mat.n_R)
    p[[0, -1]] = (mat.p_L, mat.p_R)
    return StateVec.pack(n, p, mat.P_ion0.copy()), phi


class _QuasiFermiSystem:
    def __init__(
        self,
        x: np.ndarray,
        stack: DeviceStack,
        mat: MaterialArrays,
        V_app: float,
        *,
        interface_boundary: bool = False,
        interface_topology: str = DEDUPLICATED_QSS,
        interface_transmission: float = 1.0,
        interface_transport_model: str = FERMI_RICHARDSON,
        interface_charge_reference_occupancy: np.ndarray | None = None,
        interface_charge_trap_density_m2: np.ndarray | None = None,
        poisson_tolerance_V: float,
        poisson_max_iterations: int,
    ) -> None:
        self.x = x
        self.stack = stack
        self.mat = mat
        self.V_app = V_app
        self.node_count = len(x)
        self.poisson_tolerance_V = poisson_tolerance_V
        self.poisson_max_iterations = poisson_max_iterations
        self.interface_boundary = bool(interface_boundary)
        self.interface_topology = validate_interface_topology(interface_topology)
        self.interface_transmission = float(interface_transmission)
        self.interface_transport_model = validate_interface_transport_model(
            interface_transport_model
        )
        self.interface_faces = (
            (
                tuple(int(value) for value in mat.iface_qss_interface_faces)
                if self.interface_topology == TWO_SIDED_TRACE
                else _abrupt_interface_faces(mat)
            )
            if self.interface_boundary
            else ()
        )
        references = (
            None
            if interface_charge_reference_occupancy is None
            else np.asarray(interface_charge_reference_occupancy, dtype=float)
        )
        densities = (
            None
            if interface_charge_trap_density_m2 is None
            else np.asarray(interface_charge_trap_density_m2, dtype=float)
        )
        if (references is None) != (densities is None):
            raise ValueError(
                "interface charge reference occupancies and trap densities "
                "must be supplied together"
            )
        if references is not None:
            if not self.interface_boundary or self.interface_topology != TWO_SIDED_TRACE:
                raise ValueError(
                    "equilibrium-referenced interface charge requires the "
                    "two-sided interface boundary"
                )
            expected = (len(self.interface_faces),)
            if references.shape != expected or densities.shape != expected:
                raise ValueError(
                    "interface charge arrays must match physical interfaces"
                )
            if (
                not np.all(np.isfinite(references))
                or np.any((references < 0.0) | (references > 1.0))
                or not np.all(np.isfinite(densities))
                or np.any(densities < 0.0)
            ):
                raise ValueError(
                    "interface charge references/densities must be physical"
                )
        self.interface_charge_reference_occupancy = references
        self.interface_charge_trap_density_m2 = densities
        self.base, self.phi0 = _transport_balanced_seed(
            x,
            stack,
            mat,
            V_app,
            poisson_tolerance_V=poisson_tolerance_V,
            poisson_max_iterations=poisson_max_iterations,
        )
        n0 = np.maximum(self.base[: self.node_count], 1.0)
        p0 = np.maximum(self.base[self.node_count : 2 * self.node_count], 1.0)
        self.log_n0 = np.log(n0)
        self.log_p0 = np.log(p0)
        self.thermal_voltage = mat.V_T_device
        self.qfn0 = self.thermal_voltage * self.log_n0 - (self.phi0 + mat.chi)
        self.qfp0 = self.thermal_voltage * self.log_p0 + (
            self.phi0 + mat.chi + mat.Eg
        )
        self.reference_edge_drop_n = np.diff(self.qfn0) / self.thermal_voltage
        self.reference_edge_drop_p = np.diff(self.qfp0) / self.thermal_voltage
        self.dx = np.diff(x)
        self.current_scale = max(abs(Q * float(stack.Phi)), 1.0)
        self.pin = _pin_mask(self.node_count)
        self.evaluation_count = 0

        # Source-only assembly retains generation, bulk recombination, and
        # local interface recombination without first forming the ordinary SG
        # transport divergence.  This avoids subtracting two O(1e12 A/m2)
        # fluxes before inserting the cancellation-safe QF current.
        self.source_mat = replace(
            mat,
            D_n_face=np.zeros_like(mat.D_n_face),
            D_p_face=np.zeros_like(mat.D_p_face),
        )

        dark = assemble_rhs(
            0.0,
            self.base,
            x,
            stack,
            self.source_mat,
            illuminated=False,
            V_app=V_app,
            phi_frozen=self.phi0,
        )[: 2 * self.node_count]
        light = assemble_rhs(
            0.0,
            self.base,
            x,
            stack,
            self.source_mat,
            illuminated=True,
            V_app=V_app,
            phi_frozen=self.phi0,
        )[: 2 * self.node_count]
        self.generation = light - dark

    @staticmethod
    def _stable_difference(
        a: np.ndarray,
        b: np.ndarray,
        delta: np.ndarray,
    ) -> np.ndarray:
        """Evaluate ``a - b`` from their known logarithmic ratio.

        Direct subtraction loses the terminal current when both SG legs are
        around 1e12 A/m2 in a highly doped emitter.  ``delta`` is the relevant
        quasi-Fermi-potential difference divided by thermal voltage.
        """
        out = np.empty_like(delta)
        positive = delta >= 0.0
        out[positive] = a[positive] * (-np.expm1(-delta[positive]))
        out[~positive] = b[~positive] * np.expm1(delta[~positive])
        return out

    def _bulk_space_charge_and_tangent(
        self,
        n: np.ndarray,
        p: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate carrier/dopant/ion and explicit-defect Poisson terms."""

        rho = _charge_density(
            p,
            n,
            self.base[2 * self.node_count : 3 * self.node_count],
            self.mat.P_ion0,
            self.mat.N_A,
            self.mat.N_D,
        )
        derivative = -Q * (n + p) / self.thermal_voltage
        model = self.mat.monovalent_bulk_defects
        if model is not None:
            defect = evaluate_monovalent_bulk_defects(n, p, model)
            rho = rho + defect.total_charge_density_C_m3
            derivative = (
                derivative
                + defect.total_charge_derivative_fixed_qf_C_m3_V
            )
        return rho, derivative

    def _add_interface_sheet_charge_to_poisson(
        self,
        raw: np.ndarray,
        charged_qss: EquilibriumReferencedMaterialQSSResult,
        banded: np.ndarray | None = None,
    ) -> None:
        left_nodes = tuple(int(value) for value in self.mat.iface_qss_left_nodes)
        right_nodes = tuple(int(value) for value in self.mat.iface_qss_right_nodes)
        left_distances = tuple(
            float(value) for value in self.mat.iface_qss_left_distances_m
        )
        right_distances = tuple(
            float(value) for value in self.mat.iface_qss_right_distances_m
        )
        for index, (left, right) in enumerate(zip(left_nodes, right_nodes)):
            if left <= 0 or right >= self.node_count - 1 or right != left + 1:
                raise QuasiFermiSteadyStateError(
                    "charged two-sided interfaces require adjacent interior nodes"
                )
            capacitance_left = (
                EPS_0
                * float(self.mat.eps_r[left])
                / left_distances[index]
            )
            capacitance_right = (
                EPS_0
                * float(self.mat.eps_r[right])
                / right_distances[index]
            )
            capacitance_sum = capacitance_left + capacitance_right
            weight_left = capacitance_left / capacitance_sum
            weight_right = capacitance_right / capacitance_sum
            sheet_charge = charged_qss.incremental_sheet_charge_C_m2[index]
            raw[left - 1] += weight_left * sheet_charge
            raw[right - 1] += weight_right * sheet_charge
            if banded is None:
                continue
            derivative_left, derivative_right = (
                charged_qss.sheet_charge_jacobian_bulk_phi_C_m2_V[index]
            )
            left_row = left - 1
            right_row = right - 1
            banded[1, left_row] += weight_left * derivative_left
            banded[0, right_row] += weight_left * derivative_right
            banded[2, left_row] += weight_right * derivative_left
            banded[1, right_row] += weight_right * derivative_right

    def _evaluate_charged_poisson_system(
        self,
        phi: np.ndarray,
        dqfn: np.ndarray,
        dqfp: np.ndarray,
        *,
        interface_seed: np.ndarray | None = None,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        EquilibriumReferencedMaterialQSSResult,
    ]:
        from perovskite_sim.physics.two_sided_interface import (
            solve_material_equilibrium_referenced_two_sided_interfaces_qss,
        )

        if (
            self.interface_charge_reference_occupancy is None
            or self.interface_charge_trap_density_m2 is None
        ):
            raise QuasiFermiSteadyStateError(
                "charged Poisson evaluation requires an explicit dark reference"
            )
        dphi = phi - self.phi0
        n = _density_from_log(
            self.log_n0 + (dqfn + dphi) / self.thermal_voltage,
            context="charged electron Poisson-Boltzmann evaluation",
        )
        p = _density_from_log(
            self.log_p0 + (dqfp - dphi) / self.thermal_voltage,
            context="charged hole Poisson-Boltzmann evaluation",
        )
        rho, charge_derivative = self._bulk_space_charge_and_tangent(n, p)
        charged_qss = solve_material_equilibrium_referenced_two_sided_interfaces_qss(
            self.mat,
            self.stack,
            n,
            p,
            phi,
            equilibrium_occupancy=self.interface_charge_reference_occupancy,
            trap_density_m2=self.interface_charge_trap_density_m2,
            cross_transmission=self.interface_transmission,
            interface_transport_model=self.interface_transport_model,
            initial_state_m3=interface_seed,
            # Trial bulk states need a finite IFT elimination, while the
            # converged outer state is residual-certified below.
            fail_on_residual=False,
        )
        factor = self.mat.poisson_factor
        raw = (
            factor.C[:-1] * (phi[:-2] - phi[1:-1])
            + factor.C[1:] * (phi[2:] - phi[1:-1])
            + rho[1:-1] * factor.h_cell
        )
        banded = np.zeros((3, self.node_count - 2), dtype=float)
        banded[0, 1:] = factor.C[1:-1]
        banded[1] = -(
            factor.C[:-1] + factor.C[1:]
        ) + charge_derivative[1:-1] * factor.h_cell
        banded[2, :-1] = factor.C[1:-1]
        self._add_interface_sheet_charge_to_poisson(raw, charged_qss, banded)
        return raw, banded, n, p, charged_qss

    def _solve_charged_poisson(
        self,
        dqfn: np.ndarray,
        dqfp: np.ndarray,
        *,
        V_app: float | None = None,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        float,
        float,
        EquilibriumReferencedMaterialQSSResult,
    ]:
        if (
            self.interface_charge_reference_occupancy is None
            or self.interface_charge_trap_density_m2 is None
        ):
            raise QuasiFermiSteadyStateError(
                "charged Poisson solve requires an explicit dark reference"
            )
        phi = self.phi0.copy()
        voltage = self.V_app if V_app is None else float(V_app)
        phi[0] = 0.0
        phi[-1] = poisson_right_boundary(self.mat, voltage)
        factor = self.mat.poisson_factor
        interface_seed: np.ndarray | None = None
        charged_qss: EquilibriumReferencedMaterialQSSResult | None = None
        for _ in range(self.poisson_max_iterations):
            raw, banded, _n, _p, charged_qss = (
                self._evaluate_charged_poisson_system(
                    phi,
                    dqfn,
                    dqfp,
                    interface_seed=interface_seed,
                )
            )
            interface_seed = charged_qss.qss.state_m3
            step = solve_banded((1, 1), banded, -raw)
            damping = min(
                1.0,
                0.05 / max(float(np.max(np.abs(step))), np.finfo(float).tiny),
            )
            phi[1:-1] += damping * step
            if float(np.max(np.abs(damping * step))) < self.poisson_tolerance_V:
                break
        else:
            raise QuasiFermiSteadyStateError(
                "charged eliminated Poisson-Boltzmann solve did not converge"
            )

        raw, _banded, n, p, charged_qss = (
            self._evaluate_charged_poisson_system(
                phi,
                dqfn,
                dqfp,
                interface_seed=interface_seed,
            )
        )
        scale = (factor.C[:-1] + factor.C[1:]) * self.thermal_voltage
        return (
            phi,
            n,
            p,
            float(np.max(np.abs(raw / scale))),
            float(np.max(np.abs(raw))),
            charged_qss,
        )

    def _solve_poisson(
        self,
        dqfn: np.ndarray,
        dqfp: np.ndarray,
        *,
        V_app: float | None = None,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        float,
        float,
        EquilibriumReferencedMaterialQSSResult | None,
    ]:
        if self.interface_charge_reference_occupancy is not None:
            return self._solve_charged_poisson(dqfn, dqfp, V_app=V_app)
        phi = self.phi0.copy()
        voltage = self.V_app if V_app is None else float(V_app)
        phi[0] = 0.0
        phi[-1] = poisson_right_boundary(self.mat, voltage)
        factor = self.mat.poisson_factor
        for _ in range(self.poisson_max_iterations):
            dphi = phi - self.phi0
            n = _density_from_log(
                self.log_n0 + (dqfn + dphi) / self.thermal_voltage,
                context="electron Poisson-Boltzmann iterate",
            )
            p = _density_from_log(
                self.log_p0 + (dqfp - dphi) / self.thermal_voltage,
                context="hole Poisson-Boltzmann iterate",
            )
            rho, charge_derivative = self._bulk_space_charge_and_tangent(n, p)
            raw = (
                factor.C[:-1] * (phi[:-2] - phi[1:-1])
                + factor.C[1:] * (phi[2:] - phi[1:-1])
                + rho[1:-1] * factor.h_cell
            )
            banded = np.zeros((3, self.node_count - 2), dtype=float)
            banded[0, 1:] = factor.C[1:-1]
            banded[1] = -(
                factor.C[:-1] + factor.C[1:]
            ) + charge_derivative[1:-1] * factor.h_cell
            banded[2, :-1] = factor.C[1:-1]
            step = solve_banded((1, 1), banded, -raw)
            damping = min(
                1.0,
                0.05 / max(float(np.max(np.abs(step))), np.finfo(float).tiny),
            )
            phi[1:-1] += damping * step
            if float(np.max(np.abs(damping * step))) < self.poisson_tolerance_V:
                break
        else:
            raise QuasiFermiSteadyStateError(
                "eliminated Poisson-Boltzmann solve did not converge"
            )

        dphi = phi - self.phi0
        n = _density_from_log(
            self.log_n0 + (dqfn + dphi) / self.thermal_voltage,
            context="electron Poisson-Boltzmann solution",
        )
        p = _density_from_log(
            self.log_p0 + (dqfp - dphi) / self.thermal_voltage,
            context="hole Poisson-Boltzmann solution",
        )
        rho, _charge_derivative = self._bulk_space_charge_and_tangent(n, p)
        raw = (
            factor.C[:-1] * (phi[:-2] - phi[1:-1])
            + factor.C[1:] * (phi[2:] - phi[1:-1])
            + rho[1:-1] * factor.h_cell
        )
        scale = (factor.C[:-1] + factor.C[1:]) * self.thermal_voltage
        return (
            phi,
            n,
            p,
            float(np.max(np.abs(raw / scale))),
            float(np.max(np.abs(raw))),
            None,
        )

    def _evaluate_increments(
        self,
        dqfn: np.ndarray,
        dqfp: np.ndarray,
        illumination_fraction: float,
        *,
        V_app: float | None = None,
        edge_increment_n: np.ndarray | None = None,
        edge_increment_p: np.ndarray | None = None,
    ) -> _Evaluation:
        self.evaluation_count += 1
        dqfn_arr = np.asarray(dqfn, dtype=float)
        dqfp_arr = np.asarray(dqfp, dtype=float)
        if dqfn_arr.shape != (self.node_count,) or dqfp_arr.shape != (
            self.node_count,
        ):
            raise ValueError(
                "quasi-Fermi increment arrays must match the electrical grid"
            )
        exact_edges = edge_increment_n is not None or edge_increment_p is not None
        if exact_edges:
            if edge_increment_n is None or edge_increment_p is None:
                raise ValueError(
                    "electron and hole edge increments must be supplied together"
                )
            edge_n = np.asarray(edge_increment_n, dtype=float)
            edge_p = np.asarray(edge_increment_p, dtype=float)
            expected = (self.node_count - 1,)
            if edge_n.shape != expected or edge_p.shape != expected:
                raise ValueError(
                    "quasi-Fermi edge increments must match the grid faces"
                )
            if not np.all(np.isfinite(edge_n)) or not np.all(np.isfinite(edge_p)):
                raise ValueError("quasi-Fermi edge increments must be finite")
        voltage = self.V_app if V_app is None else float(V_app)
        phi, n, p, poisson_scaled, poisson_raw, charged_qss = self._solve_poisson(
            dqfn_arr,
            dqfp_arr,
            V_app=voltage,
        )

        y = self.base.copy()
        y[: self.node_count] = n
        y[self.node_count : 2 * self.node_count] = p
        interface_qss = None
        if self.interface_boundary:
            if self.interface_topology == TWO_SIDED_TRACE:
                if charged_qss is not None:
                    interface_qss = charged_qss.qss
                else:
                    from perovskite_sim.physics.two_sided_interface import (
                        solve_material_two_sided_interfaces_qss,
                    )

                    interface_qss = solve_material_two_sided_interfaces_qss(
                        self.mat,
                        self.stack,
                        n,
                        p,
                        phi,
                        cross_transmission=self.interface_transmission,
                        interface_transport_model=self.interface_transport_model,
                        fail_on_residual=False,
                    )
            else:
                from perovskite_sim.physics.interface_plane import (
                    solve_interface_states_live_qss,
                )

                interface_qss = solve_interface_states_live_qss(
                    self.mat,
                    self.stack,
                    n,
                    p,
                    phi,
                    V_app=voltage,
                    v_th_eff=self.mat.iface_state_v_th,
                    cross_transmission=self.interface_transmission,
                    interface_transport_model=self.interface_transport_model,
                    fail_on_residual=False,
                )
        source = assemble_rhs(
            0.0,
            y,
            self.x,
            self.stack,
            self.source_mat,
            illuminated=False,
            V_app=voltage,
            phi_frozen=phi,
            interface_qss_result=interface_qss,
        )[: 2 * self.node_count]
        source += float(illumination_fraction) * self.generation

        psi_n = phi + self.mat.chi
        psi_p = phi + self.mat.chi + self.mat.Eg
        xi_n = np.diff(psi_n) / self.thermal_voltage
        xi_p = np.diff(psi_p) / self.thermal_voltage
        # Keep the reference and increment differences separate. Near the DC
        # root, forming an absolute QF potential and subtracting it again loses
        # Newton-scale increments at highly doped contacts.
        if exact_edges:
            increment_n = edge_n
            increment_p = edge_p
        else:
            increment_n = np.diff(dqfn_arr) / self.thermal_voltage
            increment_p = np.diff(dqfp_arr) / self.thermal_voltage
        delta_n = self.reference_edge_drop_n + increment_n
        delta_p = -(self.reference_edge_drop_p + increment_p)
        current_n = Q * self.mat.D_n_face / self.dx * self._stable_difference(
            bernoulli(xi_n) * n[1:],
            bernoulli(-xi_n) * n[:-1],
            delta_n,
        )
        current_p = Q * self.mat.D_p_face / self.dx * self._stable_difference(
            bernoulli(xi_p) * p[:-1],
            bernoulli(-xi_p) * p[1:],
            delta_p,
        )

        interface_face_currents: list[tuple[int, float, float]] = []
        if self.interface_boundary:
            if interface_qss is None:
                raise QuasiFermiSteadyStateError(
                    "interface QSS result was not assembled"
                )
            current_n = current_n.copy()
            current_p = current_p.copy()
            for interface_index, face in enumerate(self.interface_faces):
                base = 4 * interface_index
                # Report the current on the left reservoir side of the
                # zero-thickness plane. State-to-state cross flux is an
                # internal variable when interface SRH captures electrons and
                # holes from different sides; treating it as the observable
                # face current leaves the local capture contribution out. The
                # conservative bulk flux is the physical limiting current and
                # matches the adjacent SG segment at a certified root.
                interface_face_currents.append(
                    (
                        face,
                        -Q * interface_qss.bulk_flux_m2_s[base + 2],
                        Q * interface_qss.bulk_flux_m2_s[base + 3],
                    )
                )
                # The locally eliminated bulk-to-plane drains already enter
                # ``source``. Keep the placeholder face at zero in the outer
                # divergence so cross-interface transfer is not counted twice.
                current_n[face] = 0.0
                current_p[face] = 0.0

        rate_n = source[: self.node_count] + np.diff(
            np.r_[0.0, current_n, 0.0]
        ) / (Q * self.mat.dx_cell)
        rate_p = source[self.node_count :] - np.diff(
            np.r_[0.0, current_p, 0.0]
        ) / (Q * self.mat.dx_cell)
        residual = np.r_[
            Q * rate_n * self.mat.dx_cell / self.current_scale,
            Q * rate_p * self.mat.dx_cell / self.current_scale,
        ]
        residual[self.pin] = 0.0
        reported_current_n = current_n
        reported_current_p = current_p
        if interface_face_currents:
            reported_current_n = current_n.copy()
            reported_current_p = current_p.copy()
            for face, electron_current, hole_current in interface_face_currents:
                reported_current_n[face] = electron_current
                reported_current_p[face] = hole_current
        return _Evaluation(
            residual=residual,
            y=y,
            phi=phi,
            rate_n=rate_n,
            rate_p=rate_p,
            current_n=reported_current_n,
            current_p=reported_current_p,
            poisson_residual=poisson_scaled,
            poisson_residual_C_m2=poisson_raw,
            interface_charge_qss=charged_qss,
        )

    def evaluate_quasi_fermi(
        self,
        qfn: np.ndarray,
        qfp: np.ndarray,
        illumination_fraction: float,
        *,
        V_app: float | None = None,
    ) -> _Evaluation:
        """Evaluate stable rates/currents at absolute quasi-Fermi potentials.

        This compatibility interface is suitable when the requested QF
        differences are resolvable in the absolute representation. Sensitive
        small-signal paths should use ``evaluate_quasi_fermi_increments``.
        """
        qfn_arr = np.asarray(qfn, dtype=float)
        qfp_arr = np.asarray(qfp, dtype=float)
        if qfn_arr.shape != (self.node_count,) or qfp_arr.shape != (
            self.node_count,
        ):
            raise ValueError("quasi-Fermi arrays must match the electrical grid")
        return self._evaluate_increments(
            qfn_arr - self.qfn0,
            qfp_arr - self.qfp0,
            illumination_fraction,
            V_app=V_app,
        )

    def evaluate_quasi_fermi_increments(
        self,
        dqfn: np.ndarray,
        dqfp: np.ndarray,
        illumination_fraction: float,
        *,
        V_app: float | None = None,
    ) -> _Evaluation:
        """Evaluate QF increments without collapsing them into absolute values."""
        return self._evaluate_increments(
            dqfn,
            dqfp,
            illumination_fraction,
            V_app=V_app,
        )

    def evaluate(self, z: np.ndarray, illumination_fraction: float) -> _Evaluation:
        z_arr = np.asarray(z, dtype=float)
        physical = self._evaluate_increments(
            self.thermal_voltage * z_arr[: self.node_count],
            self.thermal_voltage * z_arr[self.node_count :],
            illumination_fraction,
            V_app=self.V_app,
        )
        residual = physical.residual.copy()
        residual[self.pin] = z_arr[self.pin]
        return replace(physical, residual=residual)

    @staticmethod
    def _expand_independent_edge_drops(
        independent: np.ndarray,
    ) -> np.ndarray:
        """Append the contact-closing edge drop without nodal subtraction."""
        values = np.asarray(independent, dtype=float)
        if values.ndim != 1 or not np.all(np.isfinite(values)):
            raise ValueError("independent edge drops must be a finite vector")
        return np.r_[values, -float(np.sum(values))]

    def edge_coordinates_to_increments(
        self,
        edge_coordinates: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Map independent face drops to pinned nodal QF increments.

        Each carrier has N-2 independent face drops. The final face drop
        closes the path so both contact increments remain exactly zero. Face
        drops are returned separately because reconstructing them later by
        nodal subtraction would lose sub-ULP drops in conductive layers.
        """
        coordinates = np.asarray(edge_coordinates, dtype=float)
        per_carrier = self.node_count - 2
        if coordinates.shape != (2 * per_carrier,):
            raise ValueError(
                "edge-coordinate vector must contain N-2 drops per carrier"
            )
        edge_n = self._expand_independent_edge_drops(
            coordinates[:per_carrier]
        )
        edge_p = self._expand_independent_edge_drops(
            coordinates[per_carrier:]
        )
        z_n = np.r_[0.0, np.cumsum(edge_n[:-1]), 0.0]
        z_p = np.r_[0.0, np.cumsum(edge_p[:-1]), 0.0]
        return z_n, z_p, edge_n, edge_p

    def evaluate_edge_coordinates(
        self,
        edge_coordinates: np.ndarray,
        illumination_fraction: float,
        *,
        physical_residual: bool = False,
    ) -> _Evaluation:
        """Evaluate the interface solve in cancellation-safe face coordinates."""
        z_n, z_p, edge_n, edge_p = self.edge_coordinates_to_increments(
            edge_coordinates
        )
        physical = self._evaluate_increments(
            self.thermal_voltage * z_n,
            self.thermal_voltage * z_p,
            illumination_fraction,
            V_app=self.V_app,
            edge_increment_n=edge_n,
            edge_increment_p=edge_p,
        )
        if physical_residual:
            return physical
        interior_residual = np.r_[
            physical.residual[1 : self.node_count - 1],
            physical.residual[
                self.node_count + 1 : 2 * self.node_count - 1
            ],
        ]
        return replace(physical, residual=interior_residual)


def _solve_newton_stage(
    system: _QuasiFermiSystem,
    z0: np.ndarray,
    illumination_fraction: float,
    *,
    finite_difference_step: float,
    residual_tolerance: float,
    max_iterations: int,
    rank_deficient: bool = False,
    near_tolerance_stagnation_factor: float = 1.0,
    edge_coordinates: bool = False,
) -> tuple[np.ndarray, int]:
    z = np.asarray(z0, dtype=float).copy()
    size = z.size
    evaluate = (
        system.evaluate_edge_coordinates
        if edge_coordinates
        else system.evaluate
    )

    for iteration in range(max_iterations + 1):
        residual = evaluate(z, illumination_fraction).residual
        max_residual = float(np.max(np.abs(residual)))
        if max_residual < residual_tolerance:
            return z, iteration
        if iteration == max_iterations:
            break

        jacobian = np.empty((size, size), dtype=float)
        for column in range(size):
            trial = z.copy()
            trial[column] += finite_difference_step
            jacobian[:, column] = (
                evaluate(trial, illumination_fraction).residual - residual
            ) / finite_difference_step
        if rank_deficient:
            column_scale = np.max(np.abs(jacobian), axis=0)
            # Equilibrate before deciding numerical rank. A depleted
            # minority-carrier column may be 30 decades below a majority
            # column yet remain the only control direction for its local
            # generation residual.
            active = column_scale > 1.0e-30
            if not np.any(active):
                raise QuasiFermiSteadyStateError(
                    "QF interface Jacobian has no active transport columns"
                )
            equilibrated = jacobian[:, active] / column_scale[active]
            scaled_step, *_ = np.linalg.lstsq(
                equilibrated,
                -residual,
                rcond=1.0e-14,
            )
            step = np.zeros(size, dtype=float)
            step[active] = scaled_step / column_scale[active]
        else:
            try:
                step = np.linalg.solve(jacobian, -residual)
            except np.linalg.LinAlgError as exc:
                raise QuasiFermiSteadyStateError(
                    "singular QF Newton Jacobian at "
                    f"illumination={illumination_fraction:g}"
                ) from exc
        step = np.clip(step, -5.0, 5.0)
        norm = float(np.linalg.norm(residual))
        for line_search_iteration in range(30):
            damping = 0.5**line_search_iteration
            candidate = z + damping * step
            try:
                candidate_norm = float(
                    np.linalg.norm(
                        evaluate(candidate, illumination_fraction).residual
                    )
                )
            except (RuntimeError, QuasiFermiSteadyStateError, ValueError):
                if rank_deficient:
                    continue
                raise
            if candidate_norm < norm * (1.0 - 1.0e-4 * damping):
                z = candidate
                break
        else:
            if (
                near_tolerance_stagnation_factor > 1.0
                and max_residual
                <= near_tolerance_stagnation_factor * residual_tolerance
            ):
                return z, iteration
            raise QuasiFermiSteadyStateError(
                "QF Newton line search failed at "
                f"illumination={illumination_fraction:g}, "
                f"max normalized residual={max_residual:.6g}"
            )
    raise QuasiFermiSteadyStateError(
        "QF Newton iteration limit reached at "
        f"illumination={illumination_fraction:g}, "
        f"max normalized residual={max_residual:.6g}"
    )


def solve_quasi_fermi_steady_state(
    x: np.ndarray,
    stack: DeviceStack,
    V_app: float = 0.0,
    *,
    illuminated: bool = True,
    mat: MaterialArrays | None = None,
    interface_boundary: bool = False,
    interface_topology: str = DEDUPLICATED_QSS,
    interface_transmission: float = 1.0,
    interface_transport_model: str = FERMI_RICHARDSON,
    initial_state: QuasiFermiSteadyStateResult | None = None,
    initial_state_grid: np.ndarray | None = None,
    force_nodal_coordinate_predictor: bool = False,
    illumination_steps: tuple[float, ...] = DEFAULT_ILLUMINATION_STEPS,
    finite_difference_step: float = 1.0e-5,
    newton_residual_tolerance: float = 1.0e-10,
    max_newton_iterations: int = 30,
    poisson_tolerance_V: float = 1.0e-13,
    poisson_max_iterations: int = 100,
    continuity_tolerance_A_m2: float = 1.0e-4,
    current_spread_tolerance_A_m2: float = 1.0e-4,
    poisson_residual_tolerance: float = 1.0e-8,
    _research_interface_charge_reference_occupancy: np.ndarray | None = None,
    _research_interface_charge_trap_density_m2: np.ndarray | None = None,
    _research_interface_charge_token: object | None = None,
) -> QuasiFermiSteadyStateResult:
    """Solve and certify the guarded local QF steady-state problem.

    ``interface_boundary=True`` replaces every internal layer-boundary SG
    face with a locally eliminated thermionic/interface-SRH closure.
    ``interface_topology='two_sided_trace'`` additionally requires a grid
    prepared by :func:`build_two_sided_trace_grid` and uses strict per-material
    reservoirs plus exact interface finite-volume geometry. Both are opt-in;
    the default guarded homojunction behavior is unchanged.

    ``continuity_tolerance_A_m2`` bounds the integrated absolute continuity
    defect of each carrier over unpinned nodes.  ``current_spread_tolerance``
    gates the peak-to-peak cancellation-safe total face current.  These are
    independent physical gates in addition to Newton's normalized cell
    residual tolerance. ``initial_state`` may warm-start a nearby voltage,
    but only when it already carries a physical certificate; the new voltage
    is still solved and certified independently.
    """
    grid = np.asarray(x, dtype=float)
    if grid.ndim != 1 or len(grid) < 3 or np.any(np.diff(grid) <= 0.0):
        raise ValueError("x must be a strictly increasing one-dimensional grid")
    if not np.isfinite(V_app):
        raise ValueError("V_app must be finite")
    if not isinstance(force_nodal_coordinate_predictor, (bool, np.bool_)):
        raise TypeError("force_nodal_coordinate_predictor must be boolean")
    positive_controls = {
        "finite_difference_step": finite_difference_step,
        "newton_residual_tolerance": newton_residual_tolerance,
        "poisson_tolerance_V": poisson_tolerance_V,
        "continuity_tolerance_A_m2": continuity_tolerance_A_m2,
        "current_spread_tolerance_A_m2": current_spread_tolerance_A_m2,
        "poisson_residual_tolerance": poisson_residual_tolerance,
    }
    for name, value in positive_controls.items():
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if max_newton_iterations <= 0 or poisson_max_iterations <= 0:
        raise ValueError("iteration limits must be positive")
    if (
        not np.isfinite(interface_transmission)
        or interface_transmission <= 0.0
        or interface_transmission > 1.0
    ):
        raise ValueError("interface_transmission must lie in (0, 1]")
    transport_model = validate_interface_transport_model(
        interface_transport_model
    )
    topology = validate_interface_topology(interface_topology)
    if topology == TWO_SIDED_TRACE:
        if not interface_boundary:
            raise ValueError("two_sided_trace requires interface_boundary=True")
        if transport_model != FERMI_DIRAC_RICHARDSON:
            raise ValueError(
                "two_sided_trace currently requires "
                f"interface_transport_model={FERMI_DIRAC_RICHARDSON!r}"
            )
    research_charge = _research_interface_charge_reference_occupancy is not None
    if research_charge != (_research_interface_charge_trap_density_m2 is not None):
        raise ValueError(
            "research interface-charge reference and density must be supplied together"
        )
    if research_charge and _research_interface_charge_token is not (
        _RESEARCH_INTERFACE_CHARGE_TOKEN
    ):
        raise ValueError(
            "research interface charge must be entered through the certified "
            "dark-reference API"
        )
    if not research_charge and _research_interface_charge_token is not None:
        raise ValueError("research interface-charge token has no charge payload")
    if research_charge and (
        not interface_boundary or topology != TWO_SIDED_TRACE
    ):
        raise ValueError(
            "research interface charge requires interface_boundary=True and "
            "interface_topology='two_sided_trace'"
        )

    stages = _validate_illumination_steps(illuminated, illumination_steps)
    input_material = _build_qf_material(grid, stack) if mat is None else mat
    if len(input_material.eps_r) != len(grid):
        raise ValueError("mat arrays must match the supplied electrical grid")
    _require_material_defect_contract(stack, input_material)
    material = input_material
    if topology == TWO_SIDED_TRACE:
        material = _prepare_two_sided_material(grid, stack, material)
    if interface_boundary:
        material = replace(
            material,
            N_iface_state=0,
            iface_state_v_th=1.0e5,
            iface_state_live_proj=True,
            iface_state_shared_occ=True,
            iface_state_physical_offsets=True,
            iface_qss_exclusive_transport=True,
            iface_qss_cross_transmission=float(interface_transmission),
            iface_qss_transport_model=transport_model,
            iface_qss_allow_inexact_inner=True,
        )
    _require_supported(
        material,
        interface_boundary=interface_boundary,
        interface_topology=topology,
        allow_charged_bulk_defects=True,
    )
    contact_certificate: ContactThermodynamicCertificate | None = None
    if material.monovalent_bulk_defects is not None:
        try:
            contact_certificate = require_contact_thermodynamic_certificate(
                stack,
                material,
            )
        except ContactThermodynamicError as exc:
            raise QuasiFermiSteadyStateError(
                "charged explicit defects require a certified contact "
                f"thermodynamic reference: {exc}"
            ) from exc
    system = _QuasiFermiSystem(
        grid,
        stack,
        material,
        float(V_app),
        interface_boundary=interface_boundary,
        interface_topology=topology,
        interface_transmission=interface_transmission,
        interface_transport_model=transport_model,
        interface_charge_reference_occupancy=(
            _research_interface_charge_reference_occupancy
        ),
        interface_charge_trap_density_m2=(
            _research_interface_charge_trap_density_m2
        ),
        poisson_tolerance_V=poisson_tolerance_V,
        poisson_max_iterations=poisson_max_iterations,
    )
    z = np.zeros(2 * len(grid), dtype=float)
    edge_coordinate_seed: np.ndarray | None = None
    initial_state_regrids = 0
    if initial_state is None and initial_state_grid is not None:
        raise ValueError("initial_state_grid requires initial_state")
    if initial_state is not None:
        if not initial_state.certified:
            raise ValueError("initial_state must carry a physical certificate")
        if initial_state.interface_boundary != bool(interface_boundary):
            raise ValueError(
                "initial_state must use the same interface-boundary model"
            )
        if (
            interface_boundary
            and initial_state.interface_topology != topology
        ):
            raise ValueError(
                "initial_state must use the same interface topology"
            )
        if interface_boundary and not np.isclose(
            initial_state.interface_transmission,
            interface_transmission,
            rtol=0.0,
            atol=0.0,
        ):
            raise ValueError(
                "initial_state must use the same interface transmission"
            )
        if (
            interface_boundary
            and initial_state.interface_transport_model != transport_model
        ):
            raise ValueError(
                "initial_state must use the same interface transport model"
            )
        qfn = np.asarray(initial_state.electron_quasi_fermi_potential_V, dtype=float)
        qfp = np.asarray(initial_state.hole_quasi_fermi_potential_V, dtype=float)
        warm_grid = (
            grid
            if initial_state_grid is None
            else np.asarray(initial_state_grid, dtype=float)
        )
        if (
            warm_grid.ndim != 1
            or warm_grid.size < 2
            or not np.all(np.isfinite(warm_grid))
            or np.any(np.diff(warm_grid) <= 0.0)
        ):
            raise ValueError(
                "initial_state_grid must be finite and strictly increasing"
            )
        if qfn.shape != warm_grid.shape or qfp.shape != warm_grid.shape:
            raise ValueError(
                "initial_state quasi-Fermi arrays must match initial_state_grid"
            )
        if not np.all(np.isfinite(qfn)) or not np.all(np.isfinite(qfp)):
            raise ValueError("initial_state quasi-Fermi arrays must be finite")
        regrid = not np.array_equal(warm_grid, grid)
        if regrid:
            endpoint_tolerance = 1.0e-12 * max(
                1.0,
                abs(float(grid[-1] - grid[0])),
            )
            if (
                abs(float(warm_grid[0] - grid[0])) > endpoint_tolerance
                or abs(float(warm_grid[-1] - grid[-1])) > endpoint_tolerance
            ):
                raise ValueError(
                    "initial_state_grid and target grid must share endpoints"
                )
            qfn = np.interp(grid, warm_grid, qfn)
            qfp = np.interp(grid, warm_grid, qfp)
            initial_state_regrids = 1
        initial_qfn_reference = initial_state.electron_quasi_fermi_reference_V
        initial_qfp_reference = initial_state.hole_quasi_fermi_reference_V
        initial_dqfn = initial_state.electron_quasi_fermi_increment_V
        initial_dqfp = initial_state.hole_quasi_fermi_increment_V
        split_qf = (
            initial_qfn_reference,
            initial_qfp_reference,
            initial_dqfn,
            initial_dqfp,
        )
        split_qf_present = tuple(value is not None for value in split_qf)
        if any(split_qf_present) and not all(split_qf_present):
            raise ValueError(
                "initial_state must provide all QF reference/increment arrays "
                "or none of them"
            )
        if all(split_qf_present):
            qfn_reference_arr = np.asarray(initial_qfn_reference, dtype=float)
            qfp_reference_arr = np.asarray(initial_qfp_reference, dtype=float)
            dqfn_arr = np.asarray(initial_dqfn, dtype=float)
            dqfp_arr = np.asarray(initial_dqfp, dtype=float)
            compensated = (
                qfn_reference_arr,
                qfp_reference_arr,
                dqfn_arr,
                dqfp_arr,
            )
            if any(
                value.shape != warm_grid.shape or not np.all(np.isfinite(value))
                for value in compensated
            ):
                raise ValueError(
                    "initial_state QF reference/increment arrays must be finite "
                    "and match initial_state_grid"
                )
            if regrid:
                qfn_reference_arr = np.interp(
                    grid,
                    warm_grid,
                    qfn_reference_arr,
                )
                qfp_reference_arr = np.interp(
                    grid,
                    warm_grid,
                    qfp_reference_arr,
                )
                dqfn_arr = np.interp(grid, warm_grid, dqfn_arr)
                dqfp_arr = np.interp(grid, warm_grid, dqfp_arr)
            z = np.r_[
                (qfn_reference_arr - system.qfn0 + dqfn_arr)
                / system.thermal_voltage,
                (qfp_reference_arr - system.qfp0 + dqfp_arr)
                / system.thermal_voltage,
            ]
        else:
            z = np.r_[
                (qfn - system.qfn0) / system.thermal_voltage,
                (qfp - system.qfp0) / system.thermal_voltage,
            ]
        z[system.pin] = 0.0
        exact_edge_drops = (
            getattr(
                initial_state,
                "electron_quasi_fermi_edge_drop_V",
                None,
            ),
            getattr(
                initial_state,
                "hole_quasi_fermi_edge_drop_V",
                None,
            ),
        )
        exact_edge_present = tuple(
            value is not None for value in exact_edge_drops
        )
        if any(exact_edge_present) and not all(exact_edge_present):
            raise ValueError(
                "initial_state must provide both electron and hole exact "
                "QF edge-drop arrays or neither"
            )
        if all(exact_edge_present):
            electron_edge_drop = np.asarray(
                exact_edge_drops[0],
                dtype=float,
            )
            hole_edge_drop = np.asarray(
                exact_edge_drops[1],
                dtype=float,
            )
            expected_edge_shape = (warm_grid.size - 1,)
            if (
                electron_edge_drop.shape != expected_edge_shape
                or hole_edge_drop.shape != expected_edge_shape
                or not np.all(np.isfinite(electron_edge_drop))
                or not np.all(np.isfinite(hole_edge_drop))
            ):
                raise ValueError(
                    "initial_state exact QF edge drops must be finite and "
                    "match initial_state_grid faces"
                )
            if interface_boundary:
                if regrid:
                    electron_edge_drop = _regrid_edge_drops(
                        warm_grid,
                        grid,
                        electron_edge_drop,
                    )
                    hole_edge_drop = _regrid_edge_drops(
                        warm_grid,
                        grid,
                        hole_edge_drop,
                    )
                increment_n = (
                    electron_edge_drop / system.thermal_voltage
                    - system.reference_edge_drop_n
                )
                increment_p = (
                    hole_edge_drop / system.thermal_voltage
                    - system.reference_edge_drop_p
                )
                edge_coordinate_seed = np.r_[
                    increment_n[:-1],
                    increment_p[:-1],
                ]
    interface_basin_initializations = 0
    interface_basin_predictor_failures = 0
    interface_basin_predictor_regrids = 0
    if (
        interface_boundary
        and illuminated
        and initial_state is None
        and input_material.monovalent_bulk_defects is None
    ):
        # The dark-to-light QF path becomes rank-deficient when a blocking
        # interface depletes a minority carrier below finite-difference
        # sensitivity. Use the established density-form SG solution only as a
        # one-sun basin predictor, then solve and certify the exclusive QSS
        # equations below. No predictor current or residual is returned.
        from perovskite_sim.experiments.steady_state import (
            _phi_from_y,
            solve_steady_state,
        )

        seed_material = replace(
            input_material,
            N_iface_state=0,
            te_softness=0.02,
            iface_state_physical_offsets=False,
            iface_qss_exclusive_transport=False,
        )
        density_seed = None
        phi_seed = None
        predictor_grid = grid
        predictor_material = seed_material
        try:
            density_seed = solve_steady_state(
                grid,
                stack,
                float(V_app),
                illuminated=True,
                mat=seed_material,
                max_newton=60,
            )
            phi_seed = _phi_from_y(
                grid,
                seed_material,
                density_seed.y,
                float(V_app),
            )
        except (RuntimeError, ValueError):
            interface_basin_predictor_failures = 1
            configured_alphas = tuple(
                float(value) for value in stack.grid_alphas
            )
            if configured_alphas and max(configured_alphas) > 3.0:
                layers = electrical_layers(stack)
                boundaries = np.cumsum(
                    [layer.thickness for layer in layers[:-1]],
                    dtype=float,
                )
                interface_indices = [
                    int(np.argmin(np.abs(grid - boundary)))
                    for boundary in boundaries
                ]
                edge_indices = [0, *interface_indices, len(grid) - 1]
                interval_counts = [
                    right - left
                    for left, right in zip(
                        edge_indices[:-1],
                        edge_indices[1:],
                    )
                ]
                predictor_grid = multilayer_grid(
                    [
                        Layer(layer.thickness, count)
                        for layer, count in zip(layers, interval_counts)
                    ],
                    alpha=tuple(min(value, 3.0) for value in configured_alphas),
                )
                if topology == TWO_SIDED_TRACE:
                    predictor_grid = build_two_sided_trace_grid(
                        predictor_grid,
                        stack,
                    )
                predictor_material = build_material_arrays(
                    predictor_grid,
                    stack,
                )
                predictor_material = replace(
                    predictor_material,
                    N_iface_state=0,
                    te_softness=0.02,
                    iface_state_physical_offsets=False,
                    iface_qss_exclusive_transport=False,
                )
                try:
                    density_seed = solve_steady_state(
                        predictor_grid,
                        stack,
                        float(V_app),
                        illuminated=True,
                        mat=predictor_material,
                        max_newton=60,
                    )
                    phi_seed = _phi_from_y(
                        predictor_grid,
                        predictor_material,
                        density_seed.y,
                        float(V_app),
                    )
                    interface_basin_predictor_regrids = 1
                except (RuntimeError, ValueError):
                    density_seed = None
                    phi_seed = None
            if density_seed is None:
                # Both density predictors are accelerators, not part of the
                # physical model. The dark QF seed remains a final fallback.
                stages = INTERFACE_FALLBACK_ILLUMINATION_STEPS
        if density_seed is not None and phi_seed is not None:
            node_count = len(predictor_grid)
            n_seed = np.maximum(density_seed.y[:node_count], 1.0)
            p_seed = np.maximum(
                density_seed.y[node_count : 2 * node_count],
                1.0,
            )
            qfn_seed = predictor_material.V_T_device * np.log(n_seed) - (
                phi_seed + predictor_material.chi
            )
            qfp_seed = predictor_material.V_T_device * np.log(p_seed) + (
                phi_seed + predictor_material.chi + predictor_material.Eg
            )
            if not np.array_equal(predictor_grid, grid):
                qfn_seed = np.interp(grid, predictor_grid, qfn_seed)
                qfp_seed = np.interp(grid, predictor_grid, qfp_seed)
            z = np.r_[
                (qfn_seed - system.qfn0) / system.thermal_voltage,
                (qfp_seed - system.qfp0) / system.thermal_voltage,
            ]
            z[system.pin] = 0.0
            stages = (1.0,)
            interface_basin_initializations = 1
    nonlinear_coordinates = z
    using_edge_coordinates = False
    numerical_residual_limit = (
        max(
            newton_residual_tolerance,
            _INTERFACE_NUMERICAL_RESIDUAL_FLOOR,
        )
        if interface_boundary
        else newton_residual_tolerance
    )
    total_iterations = 0
    edge_coordinate_predictor_iterations = 0
    charged_edge_continuation = bool(
        research_charge and edge_coordinate_seed is not None
    )
    # Exact face drops already provide a cancellation-safe same-grid voltage
    # seed. Nodal prediction is needed only after regridding or for legacy
    # states that do not carry those drops.
    edge_coordinate_predictor_required = bool(
        interface_boundary
        and initial_state is not None
        and stages == (1.0,)
        and (
            force_nodal_coordinate_predictor
            or initial_state_regrids
            or edge_coordinate_seed is None
        )
    )
    if edge_coordinate_predictor_required:
        nonlinear_coordinates, predictor_iterations = _solve_newton_stage(
            system,
            nonlinear_coordinates,
            1.0,
            finite_difference_step=finite_difference_step,
            residual_tolerance=numerical_residual_limit,
            max_iterations=max_newton_iterations,
            rank_deficient=True,
        )
        total_iterations += predictor_iterations
        edge_coordinate_predictor_iterations = predictor_iterations
        # The predictor exists only to locate the target-grid basin. Its
        # nodal face differences are converted below and the returned state is
        # always solved and certified again in edge-drop coordinates.
        edge_coordinate_seed = None
    for stage_index, fraction in enumerate(stages):
        use_edge_coordinates = (
            interface_boundary
            and (
                charged_edge_continuation
                or stage_index == len(stages) - 1
            )
        )
        if use_edge_coordinates and not using_edge_coordinates:
            per_carrier = len(grid) - 2
            if stage_index == 0 and edge_coordinate_seed is not None:
                nonlinear_coordinates = edge_coordinate_seed
            else:
                nonlinear_coordinates = np.r_[
                    np.diff(nonlinear_coordinates[: len(grid)])[
                        :per_carrier
                    ],
                    np.diff(nonlinear_coordinates[len(grid) :])[
                        :per_carrier
                    ],
                ]
            using_edge_coordinates = True
        stage_tolerance = numerical_residual_limit
        nonlinear_coordinates, iterations = _solve_newton_stage(
            system,
            nonlinear_coordinates,
            fraction,
            finite_difference_step=finite_difference_step,
            residual_tolerance=stage_tolerance,
            max_iterations=max_newton_iterations,
            rank_deficient=interface_boundary,
            near_tolerance_stagnation_factor=(
                2.0
                if interface_boundary and stage_index < len(stages) - 1
                else 1.0
            ),
            edge_coordinates=use_edge_coordinates,
        )
        total_iterations += iterations

    edge_n: np.ndarray | None = None
    edge_p: np.ndarray | None = None
    if interface_boundary:
        z_n, z_p, edge_n, edge_p = system.edge_coordinates_to_increments(
            nonlinear_coordinates
        )
        z = np.r_[z_n, z_p]
        final = system.evaluate_edge_coordinates(
            nonlinear_coordinates,
            stages[-1],
            physical_residual=True,
        )
    else:
        z = nonlinear_coordinates
        final = system.evaluate(z, stages[-1])
    bulk_defect_diagnostics = None
    if material.monovalent_bulk_defects is not None:
        bulk_defect_diagnostics = evaluate_monovalent_bulk_defects(
            final.y[: len(grid)],
            final.y[len(grid) : 2 * len(grid)],
            material.monovalent_bulk_defects,
        )
    interface_local_residual = 0.0
    interface_max_state_to_dos = 0.0
    interface_certificate = None
    if interface_boundary:
        try:
            if topology == TWO_SIDED_TRACE:
                if final.interface_charge_qss is not None:
                    interface_certificate = final.interface_charge_qss.qss
                else:
                    from perovskite_sim.physics.two_sided_interface import (
                        solve_material_two_sided_interfaces_qss,
                    )

                    interface_certificate = solve_material_two_sided_interfaces_qss(
                        material,
                        stack,
                        final.y[: len(grid)],
                        final.y[len(grid) : 2 * len(grid)],
                        final.phi,
                        cross_transmission=interface_transmission,
                        interface_transport_model=transport_model,
                        residual_tolerance=1.0e-7,
                        fail_on_residual=True,
                    )
            else:
                from perovskite_sim.physics.interface_plane import (
                    solve_interface_states_live_qss,
                )

                interface_certificate = solve_interface_states_live_qss(
                    material,
                    stack,
                    final.y[: len(grid)],
                    final.y[len(grid) : 2 * len(grid)],
                    final.phi,
                    V_app=float(V_app),
                    v_th_eff=material.iface_state_v_th,
                    cross_transmission=interface_transmission,
                    interface_transport_model=transport_model,
                    residual_tolerance=1.0e-7,
                    fail_on_residual=True,
                )
        except RuntimeError as exc:
            raise QuasiFermiSteadyStateError(
                "final interface-state QSS certificate failed"
            ) from exc
        interface_local_residual = interface_certificate.normalized_residual
        capacities: list[float] = []
        endpoint_pairs = (
            zip(material.iface_qss_left_nodes, material.iface_qss_right_nodes)
            if topology == TWO_SIDED_TRACE
            else (
                (int(interface_node) - 1, int(interface_node))
                for interface_node in material.interface_nodes
            )
        )
        for left, right in endpoint_pairs:
            capacities.extend(
                (
                    float(material.N_C_physical[right]),
                    float(material.N_V_physical[right]),
                    float(material.N_C_physical[left]),
                    float(material.N_V_physical[left]),
                )
            )
        interface_max_state_to_dos = float(
            np.max(
                interface_certificate.state_m3
                / np.asarray(capacities, dtype=float)
            )
        )
    interior = np.ones(len(grid), dtype=bool)
    interior[[0, -1]] = False
    electron_bound = float(
        Q * np.sum(np.abs(final.rate_n[interior]) * material.dx_cell[interior])
    )
    hole_bound = float(
        Q * np.sum(np.abs(final.rate_p[interior]) * material.dx_cell[interior])
    )
    max_normalized = float(np.max(np.abs(final.residual)))
    total_faces = -float(material.junction_polarity) * (
        final.current_n + final.current_p
    )
    # The first and last faces enter the first and last solved interior
    # control-volume equations, so terminal faces are part of the certificate.
    face_spread = float(np.ptp(total_faces))
    current = float(total_faces[0])

    diagnostics = {
        "max normalized cell residual": (
            max_normalized,
            numerical_residual_limit,
        ),
        "electron continuity bound [A/m2]": (
            electron_bound,
            continuity_tolerance_A_m2,
        ),
        "hole continuity bound [A/m2]": (
            hole_bound,
            continuity_tolerance_A_m2,
        ),
        "face-current spread [A/m2]": (
            face_spread,
            current_spread_tolerance_A_m2,
        ),
        "normalized Poisson residual": (
            final.poisson_residual,
            poisson_residual_tolerance,
        ),
    }
    if interface_boundary:
        diagnostics["local interface-state residual"] = (
            interface_local_residual,
            1.0e-7,
        )
    if final.interface_charge_qss is not None:
        diagnostics["charged interface Gauss residual"] = (
            float(np.max(final.interface_charge_qss.normalized_gauss_residual)),
            1.0e-7,
        )
    failures = [
        f"{name}={value:.6g} > {limit:.6g}"
        for name, (value, limit) in diagnostics.items()
        if not np.isfinite(value) or value > limit
    ]
    arrays = (
        final.y,
        final.phi,
        final.current_n,
        final.current_p,
        total_faces,
        final.rate_n,
        final.rate_p,
        z,
    )
    if any(not np.all(np.isfinite(value)) for value in arrays):
        failures.append("result contains non-finite state or current values")
    if final.interface_charge_qss is not None:
        charge_arrays = (
            final.interface_charge_qss.incremental_sheet_charge_C_m2,
            final.interface_charge_qss.trace_potential_shift_V,
            final.interface_charge_qss.normalized_gauss_residual,
            final.interface_charge_qss.scaled_local_jacobian_condition,
        )
        if any(not np.all(np.isfinite(value)) for value in charge_arrays):
            failures.append("interface-charge evidence contains non-finite values")
    if bulk_defect_diagnostics is not None:
        defect_arrays = (
            bulk_defect_diagnostics.occupancy,
            bulk_defect_diagnostics.kinetic_denominator_s1,
            bulk_defect_diagnostics.charge_density_C_m3,
            bulk_defect_diagnostics.recombination_rate_m3_s,
            bulk_defect_diagnostics.total_charge_density_C_m3,
            bulk_defect_diagnostics.total_recombination_rate_m3_s,
        )
        if any(not np.all(np.isfinite(value)) for value in defect_arrays):
            failures.append("bulk-defect evidence contains non-finite values")
        if (
            bulk_defect_diagnostics.minimum_occupancy < 0.0
            or bulk_defect_diagnostics.maximum_occupancy > 1.0
            or bulk_defect_diagnostics.minimum_kinetic_denominator_s1 <= 0.0
        ):
            failures.append("bulk-defect occupancy or kinetic denominator is invalid")
    if failures:
        raise QuasiFermiSteadyStateError(
            "QF Newton terminated without a physical certificate: "
            + "; ".join(failures)
        )

    dqfn = system.thermal_voltage * z[: len(grid)]
    dqfp = system.thermal_voltage * z[len(grid) :]
    return QuasiFermiSteadyStateResult(
        y=final.y,
        phi=final.phi,
        electron_quasi_fermi_potential_V=system.qfn0 + dqfn,
        hole_quasi_fermi_potential_V=system.qfp0 + dqfp,
        electron_face_current_A_m2=final.current_n,
        hole_face_current_A_m2=final.current_p,
        total_face_current_A_m2=total_faces,
        electron_rate_per_s=final.rate_n,
        hole_rate_per_s=final.rate_p,
        current_A_m2=current,
        face_current_spread_A_m2=face_spread,
        electron_continuity_bound_A_m2=electron_bound,
        hole_continuity_bound_A_m2=hole_bound,
        max_normalized_cell_residual=max_normalized,
        poisson_residual=final.poisson_residual,
        poisson_residual_C_m2=final.poisson_residual_C_m2,
        illumination_steps=stages,
        newton_iterations=total_iterations,
        residual_evaluations=system.evaluation_count,
        V_app=float(V_app),
        illuminated=bool(illuminated),
        electron_quasi_fermi_reference_V=system.qfn0.copy(),
        hole_quasi_fermi_reference_V=system.qfp0.copy(),
        electron_quasi_fermi_increment_V=dqfn.copy(),
        hole_quasi_fermi_increment_V=dqfp.copy(),
        electron_quasi_fermi_edge_drop_V=(
            None
            if edge_n is None
            else system.thermal_voltage
            * (system.reference_edge_drop_n + edge_n)
        ),
        hole_quasi_fermi_edge_drop_V=(
            None
            if edge_p is None
            else system.thermal_voltage
            * (system.reference_edge_drop_p + edge_p)
        ),
        interface_boundary=bool(interface_boundary),
        interface_transmission=float(interface_transmission),
        interface_transport_model=transport_model,
        interface_topology=topology,
        interface_faces=system.interface_faces,
        interface_basin_initializations=interface_basin_initializations,
        interface_basin_predictor_failures=interface_basin_predictor_failures,
        interface_basin_predictor_regrids=interface_basin_predictor_regrids,
        initial_state_regrids=initial_state_regrids,
        qf_coordinate_system=(
            "edge_drop" if interface_boundary else "nodal_increment"
        ),
        edge_coordinate_predictor_used=edge_coordinate_predictor_required,
        edge_coordinate_predictor_iterations=(
            edge_coordinate_predictor_iterations
        ),
        interface_local_residual=interface_local_residual,
        interface_max_state_to_dos=interface_max_state_to_dos,
        numerical_residual_limit=numerical_residual_limit,
        interface_charge_closure=(
            "equilibrium_referenced"
            if final.interface_charge_qss is not None
            else "off"
        ),
        interface_equilibrium_occupancy=(
            ()
            if final.interface_charge_qss is None
            else tuple(
                float(value)
                for value in final.interface_charge_qss.equilibrium_occupancy
            )
        ),
        interface_occupancy=(
            ()
            if interface_certificate is None
            or getattr(interface_certificate, "occupancy", None) is None
            else tuple(
                float(value)
                for value in getattr(interface_certificate, "occupancy")
            )
        ),
        interface_incremental_sheet_charge_C_m2=(
            ()
            if final.interface_charge_qss is None
            else tuple(
                float(value)
                for value in (
                    final.interface_charge_qss.incremental_sheet_charge_C_m2
                )
            )
        ),
        interface_trace_potential_shift_V=(
            ()
            if final.interface_charge_qss is None
            else tuple(
                (float(values[0]), float(values[1]))
                for values in final.interface_charge_qss.trace_potential_shift_V
            )
        ),
        interface_normalized_gauss_residual=(
            ()
            if final.interface_charge_qss is None
            else tuple(
                float(value)
                for value in final.interface_charge_qss.normalized_gauss_residual
            )
        ),
        interface_scaled_local_jacobian_condition=(
            ()
            if final.interface_charge_qss is None
            else tuple(
                float(value)
                for value in (
                    final.interface_charge_qss.scaled_local_jacobian_condition
                )
            )
        ),
        bulk_defect_diagnostics=bulk_defect_diagnostics,
        contact_thermodynamic_status=(
            None if contact_certificate is None else contact_certificate.status
        ),
        contact_fermi_level_span_eV=(
            None
            if contact_certificate is None
            else contact_certificate.fermi_level_span_eV
        ),
    )


def _research_array_sha256(label: str, *arrays: np.ndarray) -> str:
    digest = hashlib.sha256(label.encode("ascii"))
    for value in arrays:
        array = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _require_equilibrium_referenced_research_stack(stack: DeviceStack) -> None:
    if stack.interface_charge_closure != "equilibrium_referenced":
        raise ValueError(
            "interface-charge research requires "
            "interface_charge_closure='equilibrium_referenced'"
        )
    if not stack.interface_charge_rebaseline_acknowledged:
        raise ValueError(
            "interface-charge research requires the rebaseline acknowledgement"
        )


def _research_charge_off_stack(stack: DeviceStack) -> DeviceStack:
    _require_equilibrium_referenced_research_stack(stack)
    return replace(stack, interface_charge_closure="off")


def build_equilibrium_referenced_interface_charge_dark_reference(
    x: np.ndarray,
    stack: DeviceStack,
    *,
    interface_transmission: float = 1.0,
    **solver_controls,
) -> EquilibriumReferencedInterfaceChargeDarkReference:
    """Build the certified charge-off dark reference for the research lane."""
    from perovskite_sim.models.device import electrical_interface_defects

    reserved = {
        "V_app",
        "illuminated",
        "mat",
        "interface_boundary",
        "interface_topology",
        "interface_transmission",
        "interface_transport_model",
        "initial_state",
        "initial_state_grid",
        "_research_interface_charge_reference_occupancy",
        "_research_interface_charge_trap_density_m2",
        "_research_interface_charge_token",
    }
    overlap = reserved.intersection(solver_controls)
    if overlap:
        raise ValueError(
            "dark-reference builder owns solver controls: "
            + ", ".join(sorted(overlap))
        )
    grid = np.asarray(x, dtype=float)
    charge_off_stack = _research_charge_off_stack(stack)
    dark = solve_quasi_fermi_steady_state(
        grid,
        charge_off_stack,
        0.0,
        illuminated=False,
        interface_boundary=True,
        interface_topology=TWO_SIDED_TRACE,
        interface_transmission=interface_transmission,
        interface_transport_model=FERMI_DIRAC_RICHARDSON,
        **solver_controls,
    )
    if not dark.certified or not dark.interface_occupancy:
        raise QuasiFermiSteadyStateError(
            "charge-off dark reference lacks interface occupancy evidence"
        )
    defects = electrical_interface_defects(stack)
    if len(defects) != len(dark.interface_occupancy):
        raise QuasiFermiSteadyStateError(
            "interface defects are not aligned with the dark reference"
        )
    if any(defect is None or float(defect.N_t_cm2) <= 0.0 for defect in defects):
        raise QuasiFermiSteadyStateError(
            "equilibrium-referenced charge requires one positive-Nt "
            "InterfaceDefect per physical interface"
        )
    trap_density = tuple(float(defect.N_t_cm2) * 1.0e4 for defect in defects)
    equilibrium_occupancy = tuple(float(value) for value in dark.interface_occupancy)
    grid_sha256 = _research_array_sha256("interface-charge-grid-v1", grid)
    stack_sha256 = hashlib.sha256(repr(stack).encode("utf-8")).hexdigest()
    dark_state_sha256 = _research_array_sha256(
        "interface-charge-dark-state-v1",
        grid,
        dark.y,
        dark.phi,
        dark.electron_quasi_fermi_potential_V,
        dark.hole_quasi_fermi_potential_V,
        np.asarray(equilibrium_occupancy),
    )
    return EquilibriumReferencedInterfaceChargeDarkReference(
        dark_state=dark,
        equilibrium_occupancy=equilibrium_occupancy,
        trap_density_m2=trap_density,
        interface_transmission=float(interface_transmission),
        grid_sha256=grid_sha256,
        stack_sha256=stack_sha256,
        dark_state_sha256=dark_state_sha256,
    )


def solve_equilibrium_referenced_interface_charge_steady_state(
    x: np.ndarray,
    stack: DeviceStack,
    V_app: float,
    *,
    dark_reference: EquilibriumReferencedInterfaceChargeDarkReference,
    illuminated: bool = True,
    **solver_controls,
) -> QuasiFermiSteadyStateResult:
    """Solve one self-consistent charged QF state from a certified dark anchor."""
    reserved = {
        "mat",
        "interface_boundary",
        "interface_topology",
        "interface_transmission",
        "interface_transport_model",
        "initial_state",
        "initial_state_grid",
        "_research_interface_charge_reference_occupancy",
        "_research_interface_charge_trap_density_m2",
        "_research_interface_charge_token",
    }
    overlap = reserved.intersection(solver_controls)
    if overlap:
        raise ValueError(
            "charged research solver owns solver controls: "
            + ", ".join(sorted(overlap))
        )
    grid = np.asarray(x, dtype=float)
    charge_off_stack = _research_charge_off_stack(stack)
    if _research_array_sha256("interface-charge-grid-v1", grid) != (
        dark_reference.grid_sha256
    ):
        raise ValueError("dark reference grid does not match the charged solve")
    stack_sha256 = hashlib.sha256(repr(stack).encode("utf-8")).hexdigest()
    if stack_sha256 != dark_reference.stack_sha256:
        raise ValueError("dark reference stack does not match the charged solve")
    if not dark_reference.dark_state.certified:
        raise ValueError("dark reference state must remain certified")
    current_dark_state_sha256 = _research_array_sha256(
        "interface-charge-dark-state-v1",
        grid,
        dark_reference.dark_state.y,
        dark_reference.dark_state.phi,
        dark_reference.dark_state.electron_quasi_fermi_potential_V,
        dark_reference.dark_state.hole_quasi_fermi_potential_V,
        np.asarray(dark_reference.equilibrium_occupancy),
    )
    if current_dark_state_sha256 != dark_reference.dark_state_sha256:
        raise ValueError("dark reference state content hash does not match")
    interface_count = len(dark_reference.equilibrium_occupancy)
    if interface_count == 0 or len(dark_reference.trap_density_m2) != interface_count:
        raise ValueError("dark reference interface arrays are empty or misaligned")
    if float(V_app) == 0.0 and not illuminated:
        zeros = tuple(0.0 for _ in range(interface_count))
        trace_zeros = tuple((0.0, 0.0) for _ in range(interface_count))
        return replace(
            dark_reference.dark_state,
            interface_charge_closure="equilibrium_referenced",
            interface_equilibrium_occupancy=dark_reference.equilibrium_occupancy,
            interface_occupancy=dark_reference.equilibrium_occupancy,
            interface_incremental_sheet_charge_C_m2=zeros,
            interface_trace_potential_shift_V=trace_zeros,
            interface_normalized_gauss_residual=zeros,
            interface_scaled_local_jacobian_condition=zeros,
        )
    result = solve_quasi_fermi_steady_state(
        grid,
        charge_off_stack,
        float(V_app),
        illuminated=illuminated,
        interface_boundary=True,
        interface_topology=TWO_SIDED_TRACE,
        interface_transmission=dark_reference.interface_transmission,
        interface_transport_model=FERMI_DIRAC_RICHARDSON,
        initial_state=dark_reference.dark_state,
        _research_interface_charge_reference_occupancy=np.asarray(
            dark_reference.equilibrium_occupancy,
            dtype=float,
        ),
        _research_interface_charge_trap_density_m2=np.asarray(
            dark_reference.trap_density_m2,
            dtype=float,
        ),
        _research_interface_charge_token=_RESEARCH_INTERFACE_CHARGE_TOKEN,
        **solver_controls,
    )
    if result.interface_charge_closure != "equilibrium_referenced":
        raise QuasiFermiSteadyStateError(
            "charged solve returned without interface-charge evidence"
        )
    charge = np.asarray(result.interface_incremental_sheet_charge_C_m2)
    density = np.asarray(dark_reference.trap_density_m2)
    if charge.shape != density.shape or np.any(np.abs(charge) > Q * density):
        raise QuasiFermiSteadyStateError(
            "charged solve violated the one-electron-per-trap bound"
        )
    return result


def solve_quasi_fermi_jv_sweep(
    x: np.ndarray,
    stack: DeviceStack,
    voltages_V: np.ndarray,
    *,
    mat: MaterialArrays | None = None,
    interface_boundary: bool = False,
    interface_topology: str = DEDUPLICATED_QSS,
    interface_transmission: float = 1.0,
    interface_transport_model: str = FERMI_RICHARDSON,
    initial_short_circuit_state: QuasiFermiSteadyStateResult | None = None,
    P_in_W_m2: float = 1000.0,
    illumination_steps: tuple[float, ...] = DEFAULT_ILLUMINATION_STEPS,
    finite_difference_step: float = 1.0e-5,
    newton_residual_tolerance: float = 1.0e-10,
    max_newton_iterations: int = 30,
    poisson_tolerance_V: float = 1.0e-13,
    poisson_max_iterations: int = 100,
    continuity_tolerance_A_m2: float = 1.0e-4,
    current_spread_tolerance_A_m2: float = 1.0e-4,
    poisson_residual_tolerance: float = 1.0e-8,
    stop_after_voc: bool = False,
    voc_stop_grid_V: np.ndarray | None = None,
    minimum_voltage_step_V: float | None = None,
    max_voltage_bridge_points: int = 256,
    mpp_interpolation: Literal["sampled", "local_quadratic"] = "sampled",
) -> QuasiFermiJVSweepResult:
    """Solve a strictly increasing illuminated J-V grid by QF continuation.

    The first voltage uses the full illumination continuation unless
    ``initial_short_circuit_state`` supplies a certified one-sun basin. That
    state is re-solved against the current stack before it is retained, which
    permits parameter continuation without trusting a certificate from a
    different parameter point. Each later voltage maps the preceding
    certified QF potentials onto its own contact boundary problem and solves
    directly at one sun. No point is retained if its local physical
    certificate fails. With ``stop_after_voc=True``, the sweep stops
    immediately after the first certified current sign change; the retained
    0-to-Voc arc still determines all J-V figures of merit.

    Voltage bisection is default-off. Setting ``minimum_voltage_step_V`` opts
    into certified continuation bridge points when a requested voltage leaves
    Newton's basin. Bridge states warm-start the next attempt but are not added
    to the requested J-V sampling grid; their count and minimum allowed step
    are retained in the result for audit.

    ``voc_stop_grid_V`` may declare a nested coarse sampling grid. After the
    first current sign change, continuation then stops at the next voltage on
    that grid so every nested metric extraction retains its own Voc bracket.
    """
    voltages = np.asarray(voltages_V, dtype=float)
    if (
        voltages.ndim != 1
        or voltages.size < 2
        or not np.all(np.isfinite(voltages))
        or np.any(np.diff(voltages) <= 0.0)
    ):
        raise ValueError("voltages_V must be finite and strictly increasing")
    if voltages[0] != 0.0:
        raise ValueError("voltages_V must start at 0 V for Jsc extraction")
    if not np.isfinite(P_in_W_m2) or P_in_W_m2 <= 0.0:
        raise ValueError("P_in_W_m2 must be finite and positive")
    if mpp_interpolation not in ("sampled", "local_quadratic"):
        raise ValueError(
            "mpp_interpolation must be 'sampled' or 'local_quadratic'"
        )
    stop_grid = None
    if voc_stop_grid_V is not None:
        if not stop_after_voc:
            raise ValueError("voc_stop_grid_V requires stop_after_voc=True")
        stop_grid = np.asarray(voc_stop_grid_V, dtype=float)
        if (
            stop_grid.ndim != 1
            or stop_grid.size < 2
            or not np.all(np.isfinite(stop_grid))
            or np.any(np.diff(stop_grid) <= 0.0)
        ):
            raise ValueError(
                "voc_stop_grid_V must be finite and strictly increasing"
            )
        for stop_voltage in stop_grid:
            if not np.any(
                np.isclose(
                    voltages,
                    stop_voltage,
                    rtol=0.0,
                    atol=1.0e-12,
                )
            ):
                raise ValueError(
                    "voc_stop_grid_V must be a subset of voltages_V"
                )
    if minimum_voltage_step_V is not None and (
        not np.isfinite(minimum_voltage_step_V)
        or minimum_voltage_step_V <= 0.0
    ):
        raise ValueError(
            "minimum_voltage_step_V must be finite and positive when enabled"
        )
    if (
        isinstance(max_voltage_bridge_points, (bool, np.bool_))
        or not isinstance(max_voltage_bridge_points, (int, np.integer))
        or int(max_voltage_bridge_points) <= 0
    ):
        raise ValueError("max_voltage_bridge_points must be a positive integer")

    grid = np.asarray(x, dtype=float)
    material = _build_qf_material(grid, stack) if mat is None else mat
    _require_material_defect_contract(stack, material)
    common = dict(
        illuminated=True,
        mat=material,
        interface_boundary=interface_boundary,
        interface_topology=interface_topology,
        interface_transmission=interface_transmission,
        interface_transport_model=interface_transport_model,
        finite_difference_step=finite_difference_step,
        newton_residual_tolerance=newton_residual_tolerance,
        max_newton_iterations=max_newton_iterations,
        poisson_tolerance_V=poisson_tolerance_V,
        poisson_max_iterations=poisson_max_iterations,
        continuity_tolerance_A_m2=continuity_tolerance_A_m2,
        current_spread_tolerance_A_m2=current_spread_tolerance_A_m2,
        poisson_residual_tolerance=poisson_residual_tolerance,
    )
    if initial_short_circuit_state is not None:
        if not initial_short_circuit_state.certified:
            raise ValueError(
                "initial_short_circuit_state must carry a physical certificate"
            )
        if not np.isclose(
            initial_short_circuit_state.V_app,
            0.0,
            rtol=0.0,
            atol=1.0e-15,
        ):
            raise ValueError("initial_short_circuit_state must be at 0 V")

    points: list[QuasiFermiSteadyStateResult] = []
    previous = initial_short_circuit_state
    previous_voltage: float | None = None
    bridge_count = 0
    voc_crossed = False
    nodal_predictor_fallback_attempts = 0
    nodal_predictor_fallback_failures = 0

    def solve_at_voltage(
        voltage: float,
        initial_state: QuasiFermiSteadyStateResult | None,
        *,
        stages: tuple[float, ...],
        force_nodal_predictor: bool = False,
    ) -> QuasiFermiSteadyStateResult:
        return solve_quasi_fermi_steady_state(
            grid,
            stack,
            V_app=float(voltage),
            initial_state=initial_state,
            force_nodal_coordinate_predictor=force_nodal_predictor,
            illumination_steps=stages,
            **common,
        )

    def advance_voltage(
        left_voltage: float,
        left_state: QuasiFermiSteadyStateResult,
        right_voltage: float,
    ) -> QuasiFermiSteadyStateResult:
        nonlocal bridge_count
        nonlocal nodal_predictor_fallback_attempts
        nonlocal nodal_predictor_fallback_failures
        try:
            return solve_at_voltage(
                right_voltage,
                left_state,
                stages=(1.0,),
            )
        except QuasiFermiSteadyStateError as direct_exc:
            exc = direct_exc
            if interface_boundary:
                nodal_predictor_fallback_attempts += 1
                try:
                    return solve_at_voltage(
                        right_voltage,
                        left_state,
                        stages=(1.0,),
                        force_nodal_predictor=True,
                    )
                except QuasiFermiSteadyStateError as fallback_exc:
                    nodal_predictor_fallback_failures += 1
                    exc = fallback_exc
            span = right_voltage - left_voltage
            if minimum_voltage_step_V is None:
                raise QuasiFermiSteadyStateError(
                    "J-V voltage continuation failed in interval "
                    f"[{left_voltage:.9g}, {right_voltage:.9g}] V: {exc}"
                ) from exc
            if span <= minimum_voltage_step_V * (1.0 + 1.0e-12):
                raise QuasiFermiSteadyStateError(
                    "J-V voltage continuation failed at the minimum interval "
                    f"[{left_voltage:.9g}, {right_voltage:.9g}] V: {exc}"
                ) from exc
            if bridge_count >= max_voltage_bridge_points:
                raise QuasiFermiSteadyStateError(
                    "J-V voltage continuation exceeded "
                    f"{max_voltage_bridge_points} bridge points"
                ) from exc
            midpoint = 0.5 * (left_voltage + right_voltage)
            bridge_count += 1
            midpoint_state = advance_voltage(
                left_voltage,
                left_state,
                midpoint,
            )
            return advance_voltage(
                midpoint,
                midpoint_state,
                right_voltage,
            )

    for index, voltage in enumerate(voltages):
        first_stages = (
            (1.0,)
            if initial_short_circuit_state is not None
            else illumination_steps
        )
        if index == 0:
            point = solve_at_voltage(
                float(voltage),
                previous,
                stages=first_stages,
            )
        else:
            assert previous is not None and previous_voltage is not None
            point = advance_voltage(
                previous_voltage,
                previous,
                float(voltage),
            )
        points.append(point)
        previous = point
        previous_voltage = float(voltage)
        if (
            len(points) >= 2
            and points[-2].current_A_m2 > 0.0 >= point.current_A_m2
        ):
            voc_crossed = True
        if stop_after_voc and voc_crossed:
            on_stop_grid = stop_grid is None or np.any(
                np.isclose(
                    stop_grid,
                    voltage,
                    rtol=0.0,
                    atol=1.0e-12,
                )
            )
            if on_stop_grid:
                break

    retained_voltages = voltages[: len(points)]
    currents = np.asarray([point.current_A_m2 for point in points], dtype=float)
    metrics = compute_metrics(
        retained_voltages,
        currents,
        P_in=P_in_W_m2,
        V_oc_max=thermodynamic_voc_ceiling(stack),
        validity=[point.certified for point in points],
        mpp_interpolation=mpp_interpolation,
    )
    return QuasiFermiJVSweepResult(
        voltages_V=retained_voltages.copy(),
        currents_A_m2=currents,
        points=tuple(points),
        metrics=metrics,
        continuation_bridge_count=bridge_count,
        minimum_voltage_step_V=(
            None
            if minimum_voltage_step_V is None
            else float(minimum_voltage_step_V)
        ),
        mpp_interpolation=mpp_interpolation,
        nodal_predictor_fallback_attempts=(
            nodal_predictor_fallback_attempts
        ),
        nodal_predictor_fallback_failures=nodal_predictor_fallback_failures,
    )


__all__ = [
    "DEFAULT_ILLUMINATION_STEPS",
    "EquilibriumReferencedInterfaceChargeDarkReference",
    "QuasiFermiSteadyStateError",
    "QuasiFermiJVSweepResult",
    "QuasiFermiSteadyStateResult",
    "build_equilibrium_referenced_interface_charge_dark_reference",
    "solve_equilibrium_referenced_interface_charge_steady_state",
    "solve_quasi_fermi_jv_sweep",
    "solve_quasi_fermi_steady_state",
]
