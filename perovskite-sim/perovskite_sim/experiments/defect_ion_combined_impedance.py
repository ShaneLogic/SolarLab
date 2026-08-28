"""Research-only joint dynamic-defect and mobile-ion device impedance.

The adapter closes carrier, defect, and ion dynamics about one nonlinear DC
state.  It deliberately does not add independently computed spectra: every
finite-difference stencil re-solves the same eliminated Poisson problem with
bulk trap charge, two-sided interface sheet charge, and positive/negative ion
space charge present together.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, replace
from typing import Callable, Literal

import numpy as np
from scipy.optimize import least_squares

from perovskite_sim.constants import EPS_0, Q
from perovskite_sim.experiments.impedance_frequency import (
    FrequencyWindowAssessment,
    assess_impedance_frequency_window,
)
from perovskite_sim.experiments.quasi_fermi_impedance import (
    MAX_LINEAR_PERTURBATION_V,
    _contact_quasi_fermi_increments,
)
from perovskite_sim.experiments.quasi_fermi_steady_state import (
    _QuasiFermiSystem,
    _build_qf_material,
    _prepare_two_sided_material,
    _require_material_defect_contract,
    _require_supported,
    _research_charge_off_stack,
)
from perovskite_sim.models.defects import NEUTRAL
from perovskite_sim.models.device import DeviceStack
from perovskite_sim.physics.defect_distributions import (
    DEFAULT_DEFECT_ENERGY_QUADRATURE_ORDER,
    validate_defect_energy_quadrature_order,
)
from perovskite_sim.physics.dynamic_defect_state import (
    DynamicBulkTrapLayout,
    compile_dynamic_bulk_trap_layout,
    evaluate_dynamic_bulk_traps,
    evaluate_dynamic_bulk_traps_about_qss,
    occupancy_from_logit_increment,
    occupancy_logit,
    quasi_steady_bulk_trap_occupancy,
)
from perovskite_sim.physics.interface_plane import FERMI_DIRAC_RICHARDSON
from perovskite_sim.physics.ion_migration import (
    ion_continuity_rhs,
    ion_continuity_rhs_neg,
    ion_face_flux,
)
from perovskite_sim.physics.contacts import (
    ContactThermodynamicCertificate,
    ContactThermodynamicError,
    require_contact_thermodynamic_certificate,
)
from perovskite_sim.physics.two_sided_interface import (
    TWO_SIDED_TRACE,
    FixedOccupancyMaterialInterfaceResult,
    solve_material_two_sided_interfaces_qss,
)
from perovskite_sim.solver.mol import (
    MaterialArrays,
    _harmonic_face_average,
)
from perovskite_sim.solver.small_signal import (
    FrequencyDomainResult,
    SmallSignalCurrentComponent,
    SmallSignalEvaluation,
    SmallSignalLinearizationError,
    solve_frequency_domain,
)


DEFECT_ION_COMBINED_SCOPE = "research_defect_ion_combined_device_ac_only"
DEFECT_ION_COMBINED_VERSION = "defect-ion-combined-device-ac-v1"
DEFAULT_REFINEMENT_FACTORS = (1.0, 0.5, 0.25)
ProgressCallback = Callable[[str, int, int, str], None]
Capability = Literal[
    "bulk_defect_plus_ions",
    "interface_defect_plus_ions",
    "bulk_interface_defect_plus_ions",
]
InterfaceCurrentObservation = Literal[
    "ordinary_finite_volume_faces",
    "symmetric_adjacent_physical_faces",
]


class DefectIonCombinedError(SmallSignalLinearizationError):
    """The research-only combined defect/ion contract failed closed."""


class DefectIonCombinedCertificationError(DefectIonCombinedError):
    """A finite combined response failed one or more declared gates."""

    def __init__(self, message: str, result: "DefectIonCombinedResult") -> None:
        self.result = result
        super().__init__(message)


def _readonly(value: object, *, dtype: object) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _relative_error(left: object, right: object) -> float:
    a = np.asarray(left)
    b = np.asarray(right)
    scale = np.maximum(np.abs(a), np.abs(b))
    floor = max(float(np.max(scale)) * 1.0e-15, np.finfo(float).tiny)
    return float(np.max(np.abs(a - b) / np.maximum(scale, floor)))


def _validate_frequencies(value: object) -> np.ndarray:
    frequencies = np.asarray(value, dtype=float)
    if (
        frequencies.ndim != 1
        or frequencies.size < 3
        or not np.all(np.isfinite(frequencies))
        or np.any(frequencies <= 0.0)
        or np.any(np.diff(frequencies) <= 0.0)
    ):
        raise ValueError(
            "frequencies_Hz must contain at least three finite, positive, "
            "strictly increasing values"
        )
    return frequencies


def _validate_refinement_factors(value: object) -> tuple[float, ...]:
    factors = tuple(float(item) for item in value)
    if (
        len(factors) < 3
        or factors[0] != 1.0
        or not all(math.isfinite(item) and item > 0.0 for item in factors)
        or any(right >= left for left, right in zip(factors, factors[1:]))
    ):
        raise ValueError(
            "refinement_factors must start at 1 and contain at least three "
            "strictly decreasing positive values"
        )
    return factors


def _occupancy_logit(values: np.ndarray) -> np.ndarray:
    occupancy = np.asarray(values, dtype=float)
    if (
        occupancy.ndim != 1
        or occupancy.size == 0
        or not np.all(np.isfinite(occupancy))
        or np.any((occupancy <= 0.0) | (occupancy >= 1.0))
    ):
        raise DefectIonCombinedError(
            "interface occupancies must lie strictly inside (0, 1)"
        )
    return np.log(occupancy) - np.log1p(-occupancy)


def _occupancy_from_increment(
    reference_logit: np.ndarray,
    increment: np.ndarray,
) -> np.ndarray:
    delta = np.asarray(increment, dtype=float)
    if delta.shape != reference_logit.shape or not np.all(np.isfinite(delta)):
        raise DefectIonCombinedError("interface occupancy coordinate is invalid")
    logits = reference_logit + delta
    result = np.empty_like(logits)
    positive = logits >= 0.0
    result[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exponential = np.exp(logits[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    if np.any((result <= 0.0) | (result >= 1.0)):
        raise DefectIonCombinedError("interface occupancy coordinate saturated")
    return result


def _contiguous_components(nodes: np.ndarray) -> tuple[np.ndarray, ...]:
    if nodes.size == 0:
        return ()
    return tuple(
        np.asarray(group, dtype=int)
        for group in np.split(nodes, np.flatnonzero(np.diff(nodes) > 1) + 1)
    )


@dataclass(frozen=True, slots=True)
class CombinedIonLayout:
    positive_nodes: tuple[int, ...]
    negative_nodes: tuple[int, ...]
    positive_components: tuple[tuple[int, ...], ...]
    negative_components: tuple[tuple[int, ...], ...]

    @property
    def positive_size(self) -> int:
        return len(self.positive_nodes)

    @property
    def negative_size(self) -> int:
        return len(self.negative_nodes)

    @property
    def size(self) -> int:
        return self.positive_size + self.negative_size


@dataclass(frozen=True, slots=True)
class CombinedStateLayout:
    interior_count: int
    ion_layout: CombinedIonLayout
    bulk_trap_layout: DynamicBulkTrapLayout | None
    interface_count: int

    @property
    def electron_slice(self) -> slice:
        return slice(0, self.interior_count)

    @property
    def hole_slice(self) -> slice:
        return slice(self.interior_count, 2 * self.interior_count)

    @property
    def positive_ion_slice(self) -> slice:
        start = 2 * self.interior_count
        return slice(start, start + self.ion_layout.positive_size)

    @property
    def negative_ion_slice(self) -> slice:
        start = self.positive_ion_slice.stop
        return slice(start, start + self.ion_layout.negative_size)

    @property
    def bulk_trap_slice(self) -> slice:
        start = self.negative_ion_slice.stop
        count = 0 if self.bulk_trap_layout is None else self.bulk_trap_layout.size
        return slice(start, start + count)

    @property
    def interface_trap_slice(self) -> slice:
        start = self.bulk_trap_slice.stop
        return slice(start, start + self.interface_count)

    @property
    def size(self) -> int:
        return self.interface_trap_slice.stop

    @property
    def qss_size(self) -> int:
        return self.negative_ion_slice.stop


@dataclass(frozen=True, slots=True)
class CombinedDCCertificate:
    maximum_normalized_residual: float
    electron_continuity_bound_A_m2: float
    hole_continuity_bound_A_m2: float
    maximum_ion_electrochemical_residual: float
    maximum_ionic_face_current_A_m2: float
    maximum_ion_inventory_relative_error: float
    face_current_spread_A_m2: float
    poisson_residual: float
    maximum_bulk_trap_balance_relative_error: float
    maximum_interface_residual: float
    maximum_interface_gauss_residual: float
    contact_thermodynamics: ContactThermodynamicCertificate
    optimizer_success: bool
    optimizer_nfev: int
    certified: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CombinedDCState:
    electron_qf_increment_V: np.ndarray
    hole_qf_increment_V: np.ndarray
    positive_ion_density_m3: np.ndarray
    negative_ion_density_m3: np.ndarray | None
    electron_density_m3: np.ndarray
    hole_density_m3: np.ndarray
    potential_V: np.ndarray
    bulk_trap_occupancy: np.ndarray | None
    interface_occupancy: np.ndarray | None
    certificate: CombinedDCCertificate
    state_sha256: str


@dataclass(frozen=True, slots=True)
class CombinedFrequencyWindow:
    ionic: FrequencyWindowAssessment
    minimum_trap_relaxation_frequency_Hz: float
    maximum_trap_relaxation_frequency_Hz: float
    trap_low_frequency_limit_covered: bool
    trap_high_frequency_limit_covered: bool
    every_trap_relaxation_frequency_bracketed: bool
    sampling_density_certified: bool
    certified: bool


@dataclass(frozen=True, slots=True)
class DefectIonCombinedCertificate:
    capability: Capability
    dc_operating_point_certified: bool
    qss_embedding_relative_error: float
    maximum_bulk_trap_balance_relative_error: float
    maximum_interface_trap_balance_relative_error: float
    maximum_all_face_admittance_spread: float
    maximum_linear_solve_backward_error: float
    minimum_reciprocal_condition: float
    maximum_refinement_relative_change: float
    maximum_ion_inventory_response_relative: float
    maximum_current_decomposition_relative_error: float
    low_frequency_qss_relative_error: float
    high_frequency_frozen_relative_error: float
    frequency_window: CombinedFrequencyWindow
    certified: bool
    reasons: tuple[str, ...]
    scope: str = DEFECT_ION_COMBINED_SCOPE
    version: str = DEFECT_ION_COMBINED_VERSION


@dataclass(frozen=True, slots=True)
class DefectIonCombinedResult:
    frequencies_Hz: np.ndarray
    impedance_ohm_m2: np.ndarray
    admittance_S_m2: np.ndarray
    admittance_faces_S_m2: np.ndarray
    electron_admittance_faces_S_m2: np.ndarray
    hole_admittance_faces_S_m2: np.ndarray
    positive_ion_admittance_faces_S_m2: np.ndarray
    negative_ion_admittance_faces_S_m2: np.ndarray | None
    displacement_admittance_faces_S_m2: np.ndarray
    electron_storage_response_F_m2: np.ndarray
    hole_storage_response_F_m2: np.ndarray
    positive_ion_storage_response_F_m2: np.ndarray
    negative_ion_storage_response_F_m2: np.ndarray | None
    bulk_trap_charge_storage_response_F_m2: np.ndarray | None
    interface_sheet_charge_storage_response_F_m2: np.ndarray | None
    bulk_trap_occupancy_response_per_V: np.ndarray | None
    interface_occupancy_response_per_V: np.ndarray | None
    state_response_per_V: np.ndarray
    layout: CombinedStateLayout
    dc_state: CombinedDCState
    reference_linearization: FrequencyDomainResult
    reference_linearizations: tuple[FrequencyDomainResult, ...]
    qss_reference_admittance_S_m2: complex
    frozen_reference_admittance_S_m2: complex
    interface_current_observation: InterfaceCurrentObservation
    refinement_factors: tuple[float, ...]
    refinement_relative_changes: tuple[float, ...]
    certificate: DefectIonCombinedCertificate
    scope: str = DEFECT_ION_COMBINED_SCOPE
    version: str = DEFECT_ION_COMBINED_VERSION


def _build_ion_layout(material: MaterialArrays) -> CombinedIonLayout:
    positive = np.flatnonzero(
        (np.asarray(material.P_ion0, dtype=float) > 0.0)
        & (np.asarray(material.D_ion_node, dtype=float) > 0.0)
    )
    negative = np.array([], dtype=int)
    if material.has_dual_ions:
        negative = np.flatnonzero(
            (np.asarray(material.P_ion0_neg, dtype=float) > 0.0)
            & (np.asarray(material.D_ion_neg_node, dtype=float) > 0.0)
        )
    if positive.size == 0 and negative.size == 0:
        raise DefectIonCombinedError(
            "combined defect/ion impedance requires an active mobile-ion species"
        )
    return CombinedIonLayout(
        positive_nodes=tuple(int(value) for value in positive),
        negative_nodes=tuple(int(value) for value in negative),
        positive_components=tuple(
            tuple(int(value) for value in component)
            for component in _contiguous_components(positive)
        ),
        negative_components=tuple(
            tuple(int(value) for value in component)
            for component in _contiguous_components(negative)
        ),
    )


def _ion_fields(
    grid: np.ndarray,
    material: MaterialArrays,
    positive: np.ndarray,
    negative: np.ndarray | None,
    phi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray | None]:
    shared = bool(
        material.ion_steric_diffusion_only
        and material.ion_steric_shared_site
        and material.has_dual_ions
        and negative is not None
    )
    positive_rate = ion_continuity_rhs(
        grid,
        phi,
        positive,
        material.D_ion_face,
        material.V_T_device,
        material.P_lim_face,
        steric_diffusion_only=material.ion_steric_diffusion_only,
        P_lim_node=material.P_lim_node,
        P_other_node=negative if shared else None,
    )
    dx = np.diff(grid)
    positive_flux = ion_face_flux(
        phi,
        positive,
        dx,
        material.D_ion_face,
        material.V_T_device,
        material.P_lim_face,
        steric_diffusion_only=material.ion_steric_diffusion_only,
        P_lim_node=material.P_lim_node,
        P_other_node=negative if shared else None,
        drift_sign=1.0,
    )
    negative_rate = None
    negative_flux = None
    if negative is not None:
        negative_rate = ion_continuity_rhs_neg(
            grid,
            phi,
            negative,
            material.D_ion_neg_face,
            material.V_T_device,
            material.P_lim_neg_face,
            steric_diffusion_only=material.ion_steric_diffusion_only,
            P_lim_node=material.P_lim_neg_node,
            P_other_node=positive if shared else None,
        )
        negative_flux = ion_face_flux(
            phi,
            negative,
            dx,
            material.D_ion_neg_face,
            material.V_T_device,
            material.P_lim_neg_face,
            steric_diffusion_only=material.ion_steric_diffusion_only,
            P_lim_node=material.P_lim_neg_node,
            P_other_node=positive if shared else None,
            drift_sign=-1.0,
        )
    return positive_rate, negative_rate, positive_flux, negative_flux


def _chemical_potential(
    density: np.ndarray,
    other: np.ndarray | None,
    phi: np.ndarray,
    nodes: np.ndarray,
    material: MaterialArrays,
    *,
    positive: bool,
) -> np.ndarray:
    result = np.log(density[nodes])
    if material.ion_steric_diffusion_only:
        limit = np.asarray(
            material.P_lim_node if positive else material.P_lim_neg_node,
            dtype=float,
        )[nodes]
        total = density[nodes]
        if material.ion_steric_shared_site and other is not None:
            total = total + other[nodes]
        theta = total / limit
        if np.any((theta < 0.0) | (theta >= 1.0)):
            raise DefectIonCombinedError("ion coordinate crossed a site limit")
        result = result - np.log1p(-theta)
    sign = 1.0 if positive else -1.0
    return result + sign * phi[nodes] / material.V_T_device


def _ion_equilibrium_residual(
    density: np.ndarray,
    other: np.ndarray | None,
    phi: np.ndarray,
    components: tuple[tuple[int, ...], ...],
    target: np.ndarray,
    material: MaterialArrays,
    *,
    positive: bool,
) -> np.ndarray:
    widths = np.asarray(material.dx_cell, dtype=float)
    parts: list[np.ndarray] = []
    for index, component in enumerate(components):
        nodes = np.asarray(component, dtype=int)
        mu = _chemical_potential(
            density,
            other,
            phi,
            nodes,
            material,
            positive=positive,
        )
        inventory = float(density[nodes] @ widths[nodes])
        parts.append(np.r_[np.diff(mu), (inventory - target[index]) / target[index]])
    return np.concatenate(parts) if parts else np.empty(0, dtype=float)


def _component_inventories(
    density: np.ndarray,
    components: tuple[tuple[int, ...], ...],
    widths: np.ndarray,
) -> np.ndarray:
    return np.asarray(
        [
            density[np.asarray(nodes, dtype=int)] @ widths[np.asarray(nodes, dtype=int)]
            for nodes in components
        ],
        dtype=float,
    )


def _maximum_component_inventory_error(
    inventory: np.ndarray,
    target: np.ndarray,
) -> float:
    values = np.asarray(inventory, dtype=float)
    references = np.asarray(target, dtype=float)
    if values.shape != references.shape:
        raise DefectIonCombinedError("ion inventory and target shapes differ")
    if values.size == 0:
        return 0.0
    if (
        not np.all(np.isfinite(values))
        or not np.all(np.isfinite(references))
        or np.any(references <= 0.0)
    ):
        raise DefectIonCombinedError(
            "active ion inventories must be finite and positive"
        )
    return float(np.max(np.abs(values - references) / references))


def _state_sha256(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256(b"defect-ion-combined-dc-v1\0")
    for array in arrays:
        values = np.ascontiguousarray(np.asarray(array, dtype="<f8"))
        digest.update(np.asarray(values.shape, dtype="<i8").tobytes())
        digest.update(values.tobytes())
    return digest.hexdigest()


@dataclass(slots=True)
class _DCSolveContext:
    grid: np.ndarray
    stack: DeviceStack
    material: MaterialArrays
    system: _QuasiFermiSystem
    ion_layout: CombinedIonLayout
    illumination_fraction: float
    V_dc: float
    positive_target: np.ndarray
    negative_target: np.ndarray
    contact_thermodynamics: ContactThermodynamicCertificate


def _solve_combined_dc(
    context: _DCSolveContext,
    *,
    initial_state: CombinedDCState | None = None,
    maximum_normalized_residual: float,
    maximum_continuity_bound_A_m2: float,
    maximum_ionic_face_current_A_m2: float,
    maximum_inventory_error: float,
    maximum_poisson_residual: float,
    maximum_face_current_spread_A_m2: float,
    max_nfev: int,
) -> tuple[CombinedDCState, object]:
    grid = context.grid
    material = context.material
    system = context.system
    ion_layout = context.ion_layout
    count = grid.size
    interior = count - 2
    positive_nodes = np.asarray(ion_layout.positive_nodes, dtype=int)
    negative_nodes = np.asarray(ion_layout.negative_nodes, dtype=int)
    positive_reference = (
        np.asarray(initial_state.positive_ion_density_m3, dtype=float).copy()
        if initial_state is not None
        else np.asarray(material.P_ion0, dtype=float).copy()
    )
    negative_reference = None
    if material.has_dual_ions:
        negative_reference = (
            np.asarray(initial_state.negative_ion_density_m3, dtype=float).copy()
            if initial_state is not None
            else np.asarray(material.P_ion0_neg, dtype=float).copy()
        )
    initial = np.zeros(2 * interior + ion_layout.size, dtype=float)
    if initial_state is not None:
        initial[:interior] = (
            np.asarray(initial_state.electron_qf_increment_V)[1:-1]
            / material.V_T_device
        )
        initial[interior : 2 * interior] = (
            np.asarray(initial_state.hole_qf_increment_V)[1:-1] / material.V_T_device
        )

    def unpack(coordinate: np.ndarray):
        values = np.asarray(coordinate, dtype=float)
        dqfn = np.zeros(count, dtype=float)
        dqfp = np.zeros(count, dtype=float)
        dqfn[1:-1] = material.V_T_device * values[:interior]
        dqfp[1:-1] = material.V_T_device * values[interior : 2 * interior]
        _contact_quasi_fermi_increments(
            dqfn,
            dqfp,
            system.qfn0,
            system.qfp0,
            material,
            context.V_dc,
        )
        cursor = 2 * interior
        positive_density = positive_reference.copy()
        positive_density[positive_nodes] *= np.exp(
            values[cursor : cursor + positive_nodes.size]
        )
        cursor += positive_nodes.size
        negative_density = None
        if negative_reference is not None:
            negative_density = negative_reference.copy()
            negative_density[negative_nodes] *= np.exp(
                values[cursor : cursor + negative_nodes.size]
            )
        return dqfn, dqfp, positive_density, negative_density

    def evaluate(coordinate: np.ndarray):
        dqfn, dqfp, positive_density, negative_density = unpack(coordinate)
        value = system.evaluate_quasi_fermi_increments_defect_ion_combined(
            dqfn,
            dqfp,
            context.illumination_fraction,
            positive_ion_density_m3=positive_density,
            negative_ion_density_m3=negative_density,
            V_app=context.V_dc,
        )
        return value, dqfn, dqfp, positive_density, negative_density

    def residual(coordinate: np.ndarray) -> np.ndarray:
        try:
            value, _dqfn, _dqfp, positive_density, negative_density = evaluate(
                coordinate
            )
            carrier = np.r_[
                value.residual[1 : count - 1],
                value.residual[count + 1 : 2 * count - 1],
            ]
            positive_residual = _ion_equilibrium_residual(
                positive_density,
                negative_density,
                value.phi,
                ion_layout.positive_components,
                context.positive_target,
                material,
                positive=True,
            )
            negative_residual = (
                np.empty(0, dtype=float)
                if negative_density is None
                else _ion_equilibrium_residual(
                    negative_density,
                    positive_density,
                    value.phi,
                    ion_layout.negative_components,
                    context.negative_target,
                    material,
                    positive=False,
                )
            )
            result = np.r_[carrier, positive_residual, negative_residual]
            if not np.all(np.isfinite(result)):
                raise DefectIonCombinedError("joint DC residual became non-finite")
            return result
        except (ValueError, FloatingPointError, DefectIonCombinedError):
            return np.full(initial.size, 1.0e6, dtype=float)

    solution = least_squares(
        residual,
        initial,
        xtol=1.0e-12,
        ftol=1.0e-12,
        gtol=1.0e-12,
        max_nfev=int(max_nfev),
        x_scale="jac",
    )
    value, dqfn, dqfp, positive_density, negative_density = evaluate(solution.x)
    positive_rate, negative_rate, positive_flux, negative_flux = _ion_fields(
        grid,
        material,
        positive_density,
        negative_density,
        value.phi,
    )
    positive_mu_residual = _ion_equilibrium_residual(
        positive_density,
        negative_density,
        value.phi,
        ion_layout.positive_components,
        context.positive_target,
        material,
        positive=True,
    )
    negative_mu_residual = (
        np.empty(0, dtype=float)
        if negative_density is None
        else _ion_equilibrium_residual(
            negative_density,
            positive_density,
            value.phi,
            ion_layout.negative_components,
            context.negative_target,
            material,
            positive=False,
        )
    )
    widths = np.asarray(material.dx_cell, dtype=float)
    mask = np.ones(count, dtype=bool)
    mask[[0, -1]] = False
    electron_bound = float(Q * np.sum(np.abs(value.rate_n[mask]) * widths[mask]))
    hole_bound = float(Q * np.sum(np.abs(value.rate_p[mask]) * widths[mask]))
    ionic_current = Q * positive_flux
    if negative_flux is not None:
        ionic_current = ionic_current - Q * negative_flux
    maximum_ionic_current = float(np.max(np.abs(ionic_current)))
    total_current = value.current_n + value.current_p + ionic_current
    face_spread = float(np.ptp(total_current))
    positive_inventory = _component_inventories(
        positive_density,
        ion_layout.positive_components,
        widths,
    )
    positive_inventory_error = _maximum_component_inventory_error(
        positive_inventory,
        context.positive_target,
    )
    negative_inventory_error = 0.0
    if ion_layout.negative_size:
        if negative_density is None:
            raise DefectIonCombinedError("active negative-ion DC block is missing")
        negative_inventory = _component_inventories(
            negative_density,
            ion_layout.negative_components,
            widths,
        )
        negative_inventory_error = _maximum_component_inventory_error(
            negative_inventory,
            context.negative_target,
        )
    inventory_error = float(max(positive_inventory_error, negative_inventory_error))
    ion_mu_residual = float(
        max(
            np.max(np.abs(positive_mu_residual), initial=0.0),
            np.max(np.abs(negative_mu_residual), initial=0.0),
        )
    )
    bulk_occupancy = None
    bulk_balance = 0.0
    model = material.monovalent_bulk_defects
    if model is not None:
        dynamic_mask = np.ones(count, dtype=bool)
        dynamic_mask[[0, -1]] = False
        bulk_layout = compile_dynamic_bulk_trap_layout(
            model,
            dynamic_node_mask=dynamic_mask,
        )
        bulk_occupancy = quasi_steady_bulk_trap_occupancy(
            value.y[:count],
            value.y[count : 2 * count],
            bulk_layout,
        )
        bulk_dynamic = evaluate_dynamic_bulk_traps(
            value.y[:count],
            value.y[count : 2 * count],
            bulk_occupancy,
            bulk_layout,
        )
        bulk_balance = bulk_dynamic.maximum_local_charge_balance_relative_error
    interface_occupancy = None
    interface_residual = 0.0
    interface_gauss = 0.0
    if value.interface_charge_qss is not None:
        interface_occupancy = np.asarray(
            value.interface_charge_qss.qss.occupancy,
            dtype=float,
        )
        interface_residual = float(value.interface_charge_qss.qss.normalized_residual)
        interface_gauss = float(
            np.max(value.interface_charge_qss.normalized_gauss_residual)
        )
    maximum_residual_value = float(np.max(np.abs(residual(solution.x))))
    reasons: list[str] = []
    gates = (
        (
            context.contact_thermodynamics.certified,
            "contact_thermodynamics_not_certified",
        ),
        (solution.success, "joint_dc_optimizer_failed"),
        (
            maximum_residual_value <= maximum_normalized_residual,
            "joint_dc_residual_exceeds_limit",
        ),
        (
            electron_bound <= maximum_continuity_bound_A_m2,
            "electron_continuity_bound_exceeds_limit",
        ),
        (
            hole_bound <= maximum_continuity_bound_A_m2,
            "hole_continuity_bound_exceeds_limit",
        ),
        (
            maximum_ionic_current <= maximum_ionic_face_current_A_m2,
            "ionic_face_current_exceeds_limit",
        ),
        (
            inventory_error <= maximum_inventory_error,
            "ion_inventory_error_exceeds_limit",
        ),
        (
            value.poisson_residual <= maximum_poisson_residual,
            "poisson_residual_exceeds_limit",
        ),
        (
            face_spread <= maximum_face_current_spread_A_m2,
            "dc_face_current_spread_exceeds_limit",
        ),
    )
    reasons.extend(reason for passed, reason in gates if not passed)
    certificate = CombinedDCCertificate(
        maximum_normalized_residual=maximum_residual_value,
        electron_continuity_bound_A_m2=electron_bound,
        hole_continuity_bound_A_m2=hole_bound,
        maximum_ion_electrochemical_residual=ion_mu_residual,
        maximum_ionic_face_current_A_m2=maximum_ionic_current,
        maximum_ion_inventory_relative_error=inventory_error,
        face_current_spread_A_m2=face_spread,
        poisson_residual=float(value.poisson_residual),
        maximum_bulk_trap_balance_relative_error=float(bulk_balance),
        maximum_interface_residual=interface_residual,
        maximum_interface_gauss_residual=interface_gauss,
        contact_thermodynamics=context.contact_thermodynamics,
        optimizer_success=bool(solution.success),
        optimizer_nfev=int(solution.nfev),
        certified=not reasons,
        reasons=tuple(reasons),
    )
    state_arrays = [dqfn, dqfp, positive_density]
    if negative_density is not None:
        state_arrays.append(negative_density)
    state = CombinedDCState(
        electron_qf_increment_V=_readonly(dqfn, dtype=float),
        hole_qf_increment_V=_readonly(dqfp, dtype=float),
        positive_ion_density_m3=_readonly(positive_density, dtype=float),
        negative_ion_density_m3=(
            None
            if negative_density is None
            else _readonly(negative_density, dtype=float)
        ),
        electron_density_m3=_readonly(value.y[:count], dtype=float),
        hole_density_m3=_readonly(value.y[count : 2 * count], dtype=float),
        potential_V=_readonly(value.phi, dtype=float),
        bulk_trap_occupancy=(
            None if bulk_occupancy is None else _readonly(bulk_occupancy, dtype=float)
        ),
        interface_occupancy=(
            None
            if interface_occupancy is None
            else _readonly(interface_occupancy, dtype=float)
        ),
        certificate=certificate,
        state_sha256=_state_sha256(*state_arrays),
    )
    return state, value


def _interface_relaxation_rate(
    dynamic: FixedOccupancyMaterialInterfaceResult,
    material: MaterialArrays,
    trap_density_m2: np.ndarray,
    capture_velocities_m_s: np.ndarray,
) -> np.ndarray:
    state = np.asarray(dynamic.qss.state_m3).reshape(-1, 4)[:, [2, 3, 0, 1]]
    return (
        capture_velocities_m_s[:, 0]
        * (
            state[:, 0]
            + state[:, 2]
            + np.asarray(material.interface_n1_L)
            + np.asarray(material.interface_n1_R)
        )
        + capture_velocities_m_s[:, 1]
        * (
            state[:, 1]
            + state[:, 3]
            + np.asarray(material.interface_p1_L)
            + np.asarray(material.interface_p1_R)
        )
    ) / trap_density_m2


def run_defect_ion_combined_impedance(
    x: np.ndarray,
    stack: DeviceStack,
    frequencies_Hz: object,
    *,
    V_dc: float = 0.0,
    illuminated: bool = False,
    delta_V: float = 0.01,
    state_step: float = 1.0e-5,
    voltage_step: float = 1.0e-5,
    refinement_factors: object = DEFAULT_REFINEMENT_FACTORS,
    defect_energy_quadrature_order: int = DEFAULT_DEFECT_ENERGY_QUADRATURE_ORDER,
    frequency_branch_margin_decades: float = 1.0,
    maximum_frequency_sampling_gap_decades: float = 0.5,
    maximum_dc_normalized_residual: float = 1.0e-8,
    maximum_dc_continuity_bound_A_m2: float = 1.0e-4,
    maximum_dc_ionic_face_current_A_m2: float = 1.0e-6,
    maximum_dc_inventory_error: float = 1.0e-10,
    maximum_dc_poisson_residual: float = 1.0e-8,
    maximum_dc_face_current_spread_A_m2: float = 1.0e-4,
    maximum_qss_embedding_relative_error: float = 1.0e-8,
    maximum_trap_balance_relative_error: float = 1.0e-3,
    maximum_all_face_admittance_spread: float = 5.0e-4,
    maximum_linear_solve_backward_error: float = 1.0e-10,
    maximum_refinement_relative_change: float = 2.0e-3,
    maximum_ion_inventory_response_relative: float = 1.0e-8,
    maximum_current_decomposition_relative_error: float = 1.0e-7,
    maximum_limit_relative_error: float = 3.0e-2,
    dc_max_nfev: int = 1000,
    require_certificate: bool = True,
    progress: ProgressCallback | None = None,
) -> DefectIonCombinedResult:
    """Solve the joint bulk/interface defect and mobile-ion AC operator."""
    grid = np.asarray(x, dtype=float)
    if (
        grid.ndim != 1
        or grid.size < 3
        or not np.all(np.isfinite(grid))
        or np.any(np.diff(grid) <= 0.0)
    ):
        raise ValueError("x must be a finite, strictly increasing one-dimensional grid")
    frequencies = _validate_frequencies(frequencies_Hz)
    factors = _validate_refinement_factors(refinement_factors)
    if not np.isfinite(V_dc):
        raise ValueError("V_dc must be finite")
    if not np.isfinite(delta_V) or not 0.0 < delta_V < MAX_LINEAR_PERTURBATION_V:
        raise ValueError("delta_V must be positive and below the 20 mV limit")
    scalar_values = {
        "state_step": state_step,
        "voltage_step": voltage_step,
        "frequency_branch_margin_decades": frequency_branch_margin_decades,
        "maximum_frequency_sampling_gap_decades": (
            maximum_frequency_sampling_gap_decades
        ),
        "maximum_dc_normalized_residual": maximum_dc_normalized_residual,
        "maximum_dc_continuity_bound_A_m2": maximum_dc_continuity_bound_A_m2,
        "maximum_dc_ionic_face_current_A_m2": maximum_dc_ionic_face_current_A_m2,
        "maximum_dc_inventory_error": maximum_dc_inventory_error,
        "maximum_dc_poisson_residual": maximum_dc_poisson_residual,
        "maximum_dc_face_current_spread_A_m2": maximum_dc_face_current_spread_A_m2,
        "maximum_qss_embedding_relative_error": maximum_qss_embedding_relative_error,
        "maximum_trap_balance_relative_error": maximum_trap_balance_relative_error,
        "maximum_all_face_admittance_spread": maximum_all_face_admittance_spread,
        "maximum_linear_solve_backward_error": maximum_linear_solve_backward_error,
        "maximum_refinement_relative_change": maximum_refinement_relative_change,
        "maximum_ion_inventory_response_relative": (
            maximum_ion_inventory_response_relative
        ),
        "maximum_current_decomposition_relative_error": (
            maximum_current_decomposition_relative_error
        ),
        "maximum_limit_relative_error": maximum_limit_relative_error,
    }
    if any(not np.isfinite(value) or value <= 0.0 for value in scalar_values.values()):
        raise ValueError("all numerical gates and perturbation steps must be positive")
    energy_order = validate_defect_energy_quadrature_order(
        defect_energy_quadrature_order
    )

    has_interface = bool(stack.interface_defects)
    microscopic_contract = None
    working_stack = stack
    if has_interface:
        working_stack, microscopic_contract = _research_charge_off_stack(stack)
    material = _build_qf_material(
        grid,
        working_stack,
        defect_energy_quadrature_order=energy_order,
    )
    if has_interface:
        material = _prepare_two_sided_material(grid, working_stack, material)
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
    _require_material_defect_contract(
        working_stack,
        material,
        defect_energy_quadrature_order=energy_order,
    )
    _require_supported(
        material,
        interface_boundary=has_interface,
        interface_topology=TWO_SIDED_TRACE if has_interface else "deduplicated_qss",
        allow_charged_bulk_defects=True,
        allow_mobile_ions=True,
    )
    try:
        contact_thermodynamics = require_contact_thermodynamic_certificate(
            working_stack,
            material,
        )
    except ContactThermodynamicError as exc:
        raise DefectIonCombinedError(
            "combined defect/ion impedance requires a certified contact "
            f"thermodynamic reference: {exc}"
        ) from exc
    has_bulk = material.monovalent_bulk_defects is not None
    if not has_bulk and not has_interface:
        raise DefectIonCombinedError(
            "combined impedance requires an explicit bulk or interface defect"
        )
    capability: Capability
    if has_bulk and has_interface:
        capability = "bulk_interface_defect_plus_ions"
    elif has_bulk:
        capability = "bulk_defect_plus_ions"
    else:
        capability = "interface_defect_plus_ions"

    ion_layout = _build_ion_layout(material)
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

    interface_reference = None
    interface_trap_density = None
    capture_velocities = None
    dark_state = None
    if has_interface:
        charge_off_system = _QuasiFermiSystem(
            grid,
            working_stack,
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
            working_stack,
            material,
            charge_off_system,
            ion_layout,
            0.0,
            0.0,
            positive_target,
            negative_target,
            contact_thermodynamics,
        )
        dark_state, dark_value = _solve_combined_dc(
            dark_context,
            maximum_normalized_residual=maximum_dc_normalized_residual,
            maximum_continuity_bound_A_m2=maximum_dc_continuity_bound_A_m2,
            maximum_ionic_face_current_A_m2=maximum_dc_ionic_face_current_A_m2,
            maximum_inventory_error=maximum_dc_inventory_error,
            maximum_poisson_residual=maximum_dc_poisson_residual,
            maximum_face_current_spread_A_m2=maximum_dc_face_current_spread_A_m2,
            max_nfev=dc_max_nfev,
        )
        if not dark_state.certificate.certified:
            raise DefectIonCombinedError(
                "combined dark reference is not residual certified: "
                + ", ".join(dark_state.certificate.reasons)
            )
        qss = solve_material_two_sided_interfaces_qss(
            material,
            working_stack,
            dark_value.y[: grid.size],
            dark_value.y[grid.size : 2 * grid.size],
            dark_value.phi,
            cross_transmission=1.0,
            interface_transport_model=FERMI_DIRAC_RICHARDSON,
            fail_on_residual=True,
        )
        interface_reference = np.asarray(qss.occupancy, dtype=float)
        interface_trap_density = np.asarray(
            microscopic_contract.trap_density_m2,
            dtype=float,
        )
        capture_velocities = np.asarray(
            microscopic_contract.capture_velocities_m_s,
            dtype=float,
        )

    system = _QuasiFermiSystem(
        grid,
        working_stack,
        material,
        float(V_dc),
        interface_boundary=has_interface,
        interface_topology=TWO_SIDED_TRACE if has_interface else "deduplicated_qss",
        interface_transmission=1.0,
        interface_transport_model=FERMI_DIRAC_RICHARDSON,
        interface_charge_reference_occupancy=interface_reference,
        interface_charge_trap_density_m2=interface_trap_density,
        poisson_tolerance_V=1.0e-13,
        poisson_max_iterations=100,
    )
    context = _DCSolveContext(
        grid,
        working_stack,
        material,
        system,
        ion_layout,
        1.0 if illuminated else 0.0,
        float(V_dc),
        positive_target,
        negative_target,
        contact_thermodynamics,
    )
    dc_state, qss_dc = _solve_combined_dc(
        context,
        initial_state=(
            dark_state if has_interface and V_dc == 0.0 and not illuminated else None
        ),
        maximum_normalized_residual=maximum_dc_normalized_residual,
        maximum_continuity_bound_A_m2=maximum_dc_continuity_bound_A_m2,
        maximum_ionic_face_current_A_m2=maximum_dc_ionic_face_current_A_m2,
        maximum_inventory_error=maximum_dc_inventory_error,
        maximum_poisson_residual=maximum_dc_poisson_residual,
        maximum_face_current_spread_A_m2=maximum_dc_face_current_spread_A_m2,
        max_nfev=dc_max_nfev,
    )
    if not dc_state.certificate.certified:
        raise DefectIonCombinedError(
            "combined DC operating point is not certified: "
            + ", ".join(dc_state.certificate.reasons)
        )

    bulk_layout = None
    bulk_occupancy_dc = None
    bulk_reference_logit = None
    bulk_dynamic_dc = None
    if has_bulk:
        dynamic_mask = np.ones(grid.size, dtype=bool)
        dynamic_mask[[0, -1]] = False
        bulk_layout = compile_dynamic_bulk_trap_layout(
            material.monovalent_bulk_defects,
            dynamic_node_mask=dynamic_mask,
        )
        bulk_occupancy_dc = quasi_steady_bulk_trap_occupancy(
            qss_dc.y[: grid.size],
            qss_dc.y[grid.size : 2 * grid.size],
            bulk_layout,
        )
        bulk_reference_logit = occupancy_logit(bulk_occupancy_dc, bulk_layout)
    interface_occupancy_dc = (
        None
        if qss_dc.interface_charge_qss is None
        else np.asarray(qss_dc.interface_charge_qss.qss.occupancy, dtype=float)
    )
    interface_reference_logit = (
        None
        if interface_occupancy_dc is None
        else _occupancy_logit(interface_occupancy_dc)
    )
    interface_count = (
        0 if interface_occupancy_dc is None else interface_occupancy_dc.size
    )
    layout = CombinedStateLayout(
        interior_count=grid.size - 2,
        ion_layout=ion_layout,
        bulk_trap_layout=bulk_layout,
        interface_count=interface_count,
    )
    dqfn_dc = np.asarray(dc_state.electron_qf_increment_V, dtype=float)
    dqfp_dc = np.asarray(dc_state.hole_qf_increment_V, dtype=float)
    positive_dc = np.asarray(dc_state.positive_ion_density_m3, dtype=float)
    negative_dc = (
        None
        if dc_state.negative_ion_density_m3 is None
        else np.asarray(dc_state.negative_ion_density_m3, dtype=float)
    )
    reference_n = np.asarray(dc_state.electron_density_m3, dtype=float)
    reference_p = np.asarray(dc_state.hole_density_m3, dtype=float)
    edge_dc_n = np.diff(dqfn_dc) / material.V_T_device
    edge_dc_p = np.diff(dqfp_dc) / material.V_T_device

    def physical_coordinates(coordinate: np.ndarray, voltage: float):
        values = np.asarray(coordinate, dtype=float)
        if values.shape != (layout.size,) or not np.all(np.isfinite(values)):
            raise DefectIonCombinedError("combined AC coordinate is invalid")
        dqfn = dqfn_dc.copy()
        dqfp = dqfp_dc.copy()
        dqfn[1:-1] += material.V_T_device * values[layout.electron_slice]
        dqfp[1:-1] += material.V_T_device * values[layout.hole_slice]
        _contact_quasi_fermi_increments(
            dqfn,
            dqfp,
            system.qfn0,
            system.qfp0,
            material,
            voltage,
        )
        edge_n = edge_dc_n + np.diff(dqfn - dqfn_dc) / material.V_T_device
        edge_p = edge_dc_p + np.diff(dqfp - dqfp_dc) / material.V_T_device
        positive_density = positive_dc.copy()
        positive_density[np.asarray(ion_layout.positive_nodes, dtype=int)] *= np.exp(
            values[layout.positive_ion_slice]
        )
        negative_density = None
        if negative_dc is not None:
            negative_density = negative_dc.copy()
            negative_density[np.asarray(ion_layout.negative_nodes, dtype=int)] *= (
                np.exp(values[layout.negative_ion_slice])
            )
        bulk_occupancy = None
        if bulk_layout is not None:
            bulk_occupancy = occupancy_from_logit_increment(
                bulk_reference_logit,
                values[layout.bulk_trap_slice],
                bulk_layout,
            )
        interface_occupancy = None
        if interface_reference_logit is not None:
            interface_occupancy = _occupancy_from_increment(
                interface_reference_logit,
                values[layout.interface_trap_slice],
            )
        return (
            dqfn,
            dqfp,
            edge_n,
            edge_p,
            positive_density,
            negative_density,
            bulk_occupancy,
            interface_occupancy,
        )

    local_interface_residuals: list[float] = []
    local_interface_gauss: list[float] = []

    def dynamic_value(coordinate: np.ndarray, voltage: float):
        (
            dqfn,
            dqfp,
            edge_n,
            edge_p,
            positive_density,
            negative_density,
            bulk_occupancy,
            interface_occupancy,
        ) = physical_coordinates(coordinate, voltage)
        value = system.evaluate_quasi_fermi_increments_defect_ion_combined(
            dqfn,
            dqfp,
            1.0 if illuminated else 0.0,
            positive_ion_density_m3=positive_density,
            negative_ion_density_m3=negative_density,
            dynamic_bulk_layout=bulk_layout,
            dynamic_bulk_occupancy=bulk_occupancy,
            dynamic_bulk_reference_n=reference_n if bulk_layout is not None else None,
            dynamic_bulk_reference_p=reference_p if bulk_layout is not None else None,
            dynamic_bulk_reference_occupancy=(
                bulk_occupancy_dc if bulk_layout is not None else None
            ),
            dynamic_interface_occupancy=interface_occupancy,
            V_app=voltage,
            edge_increment_n=edge_n,
            edge_increment_p=edge_p,
        )
        bulk_dynamic = None
        if bulk_layout is not None:
            bulk_dynamic = evaluate_dynamic_bulk_traps_about_qss(
                value.y[: grid.size],
                value.y[grid.size : 2 * grid.size],
                bulk_occupancy,
                bulk_layout,
                reference_electron_density_m3=reference_n,
                reference_hole_density_m3=reference_p,
                reference_occupancy=bulk_occupancy_dc,
            )
        interface_dynamic = value.interface_charge_dynamic
        if interface_count:
            if interface_dynamic is None:
                raise DefectIonCombinedError(
                    "combined evaluation lost fixed-interface evidence"
                )
            local_interface_residuals.append(
                float(interface_dynamic.qss.normalized_residual)
            )
            local_interface_gauss.append(
                float(np.max(interface_dynamic.normalized_gauss_residual))
            )
        return (
            value,
            positive_density,
            negative_density,
            bulk_dynamic,
            interface_dynamic,
        )

    def small_signal_evaluation(
        coordinate: np.ndarray,
        voltage: float,
        *,
        trap_storage: bool,
        qss_traps: bool = False,
    ) -> SmallSignalEvaluation:
        if qss_traps:
            values = np.asarray(coordinate, dtype=float)
            full = np.zeros(layout.size, dtype=float)
            full[: layout.qss_size] = values
            (
                dqfn,
                dqfp,
                edge_n,
                edge_p,
                positive_density,
                negative_density,
                _bulk,
                _interface,
            ) = physical_coordinates(full, voltage)
            value = system.evaluate_quasi_fermi_increments_defect_ion_combined(
                dqfn,
                dqfp,
                1.0 if illuminated else 0.0,
                positive_ion_density_m3=positive_density,
                negative_ion_density_m3=negative_density,
                V_app=voltage,
                edge_increment_n=edge_n,
                edge_increment_p=edge_p,
            )
            bulk_dynamic = None
            interface_dynamic = None
        else:
            (
                value,
                positive_density,
                negative_density,
                bulk_dynamic,
                interface_dynamic,
            ) = dynamic_value(coordinate, voltage)
        positive_rate, negative_rate, positive_flux, negative_flux = _ion_fields(
            grid,
            material,
            positive_density,
            negative_density,
            value.phi,
        )
        storage_parts = [
            value.y[1 : grid.size - 1],
            value.y[grid.size + 1 : 2 * grid.size - 1],
        ]
        rate_parts = [value.rate_n[1:-1], value.rate_p[1:-1]]
        positive_nodes = np.asarray(ion_layout.positive_nodes, dtype=int)
        negative_nodes = np.asarray(ion_layout.negative_nodes, dtype=int)
        storage_parts.append(positive_density[positive_nodes])
        rate_parts.append(positive_rate[positive_nodes])
        if ion_layout.negative_size:
            if negative_density is None or negative_rate is None:
                raise DefectIonCombinedError(
                    "active negative-ion AC storage/rate block is missing"
                )
            storage_parts.append(negative_density[negative_nodes])
            rate_parts.append(negative_rate[negative_nodes])
        if trap_storage and bulk_dynamic is not None:
            storage_parts.append(bulk_dynamic.occupied_storage_m3)
            rate_parts.append(bulk_dynamic.trap_storage_rate_m3_s)
        if trap_storage and interface_dynamic is not None:
            capture = np.asarray(interface_dynamic.qss.capture_flux_m2_s).reshape(-1, 4)
            interface_rate = capture[:, [0, 2]].sum(axis=1) - capture[:, [1, 3]].sum(
                axis=1
            )
            storage_parts.append(interface_trap_density * interface_dynamic.occupancy)
            rate_parts.append(interface_rate)
        polarity = float(material.junction_polarity)
        electron = polarity * np.asarray(value.current_n, dtype=float)
        hole = polarity * np.asarray(value.current_p, dtype=float)
        positive_current = polarity * Q * positive_flux
        negative_current = (
            None if negative_flux is None else -polarity * Q * negative_flux
        )
        eps_face = EPS_0 * _harmonic_face_average(material.eps_r)
        electric_field = -np.diff(value.phi) / np.diff(grid)
        displacement_charge = polarity * eps_face * electric_field
        # A zero-thickness interface replaces one ordinary bulk face.  Once a
        # volumetric dynamic defect also occupies either adjacent control
        # volume, the local plane flux is no longer the same observation as a
        # face-centred external current.  Reconstruct that diagnostic face
        # symmetrically from its two physical neighbours; four-leg capture and
        # Gauss balance remain certified separately on the local plane.
        for face in material.iface_qss_interface_faces:
            face = int(face)
            if face <= 0 or face >= electron.size - 1:
                raise DefectIonCombinedError(
                    "two-sided interface current reconstruction lacks neighbours"
                )
            for channel in (electron, hole, positive_current, displacement_charge):
                channel[face] = 0.5 * (channel[face - 1] + channel[face + 1])
            if negative_current is not None:
                negative_current[face] = 0.5 * (
                    negative_current[face - 1] + negative_current[face + 1]
                )
        components = [
            SmallSignalCurrentComponent("electron", electron),
            SmallSignalCurrentComponent("hole", hole),
            SmallSignalCurrentComponent("positive_ion", positive_current),
        ]
        if ion_layout.negative_size:
            if negative_current is None:
                raise DefectIonCombinedError(
                    "active negative-ion AC current block is missing"
                )
            components.append(
                SmallSignalCurrentComponent("negative_ion", negative_current)
            )
        conduction = sum(
            (component.current_faces for component in components),
            start=np.zeros_like(electron),
        )
        return SmallSignalEvaluation(
            storage=np.concatenate(storage_parts),
            rate=np.concatenate(rate_parts),
            conduction_current_faces=conduction,
            displacement_charge_faces=displacement_charge,
            current_components=tuple(components),
        )

    coordinate = np.zeros(layout.size, dtype=float)
    dynamic_dc_value = dynamic_value(coordinate, float(V_dc))
    bulk_dynamic_dc = dynamic_dc_value[3]
    interface_dynamic_dc = dynamic_dc_value[4]
    dc_current_scale = max(
        float(np.max(np.abs(qss_dc.current_n + qss_dc.current_p))),
        abs(Q * float(working_stack.Phi)),
        1.0,
    )
    qss_embedding = max(
        float(np.max(np.abs(dynamic_dc_value[0].phi - qss_dc.phi)))
        / material.V_T_device,
        Q
        * float(
            np.sum(
                (
                    np.abs(dynamic_dc_value[0].rate_n - qss_dc.rate_n)
                    + np.abs(dynamic_dc_value[0].rate_p - qss_dc.rate_p)
                )
                * widths
            )
        )
        / dc_current_scale,
        float(np.max(np.abs(dynamic_dc_value[0].current_n - qss_dc.current_n)))
        / dc_current_scale,
        float(np.max(np.abs(dynamic_dc_value[0].current_p - qss_dc.current_p)))
        / dc_current_scale,
    )

    face_weights = np.diff(grid) / float(grid[-1] - grid[0])
    levels: list[FrequencyDomainResult] = []
    for index, factor in enumerate(factors):
        if progress is not None:
            progress(
                "defect_ion_combined_refinement",
                index,
                len(factors),
                f"finite-difference factor {factor:g}",
            )
        levels.append(
            solve_frequency_domain(
                lambda state, voltage: small_signal_evaluation(
                    state,
                    voltage,
                    trap_storage=True,
                ),
                coordinate,
                float(V_dc),
                frequencies,
                state_step=state_step * factor,
                voltage_step=voltage_step * factor,
                face_weights=face_weights,
                progress=progress,
            )
        )
    final = levels[-1]
    refinement_changes = tuple(
        _relative_error(coarse.admittance_faces, fine.admittance_faces)
        for coarse, fine in zip(levels, levels[1:])
    )
    qss_reference = solve_frequency_domain(
        lambda state, voltage: small_signal_evaluation(
            state,
            voltage,
            trap_storage=False,
            qss_traps=True,
        ),
        np.zeros(layout.qss_size),
        float(V_dc),
        np.asarray([frequencies[0]]),
        state_step=state_step * factors[-1],
        voltage_step=voltage_step * factors[-1],
        face_weights=face_weights,
    )

    def frozen_evaluate(state: np.ndarray, voltage: float) -> SmallSignalEvaluation:
        full = np.zeros(layout.size, dtype=float)
        full[: layout.qss_size] = np.asarray(state, dtype=float)
        return small_signal_evaluation(full, voltage, trap_storage=False)

    frozen_reference = solve_frequency_domain(
        frozen_evaluate,
        np.zeros(layout.qss_size),
        float(V_dc),
        np.asarray([frequencies[-1]]),
        state_step=state_step * factors[-1],
        voltage_step=voltage_step * factors[-1],
        face_weights=face_weights,
    )
    low_error = _relative_error(
        final.admittance_faces[0],
        qss_reference.admittance_faces[0],
    )
    high_error = _relative_error(
        final.admittance_faces[-1],
        frozen_reference.admittance_faces[0],
    )

    component_map = {
        component.name: component.admittance_faces
        for component in final.current_components
    }
    component_sum = np.sum(np.stack(tuple(component_map.values())), axis=0)
    component_scale = np.maximum(
        np.maximum(
            np.abs(final.conduction_admittance_faces),
            sum(
                (np.abs(value) for value in component_map.values()),
                start=np.zeros_like(final.conduction_admittance_faces.real),
            ),
        ),
        np.finfo(float).tiny,
    )
    decomposition_error = float(
        np.max(
            np.abs(final.conduction_admittance_faces - component_sum) / component_scale
        )
    )
    positive_nodes = np.asarray(ion_layout.positive_nodes, dtype=int)
    negative_nodes = np.asarray(ion_layout.negative_nodes, dtype=int)
    positive_response = final.storage_response[:, layout.positive_ion_slice]
    positive_inventory = positive_response @ widths[positive_nodes]
    positive_scale = np.abs(positive_response) @ widths[positive_nodes]
    positive_ratio = np.divide(
        np.abs(positive_inventory),
        positive_scale,
        out=np.zeros_like(positive_scale, dtype=float),
        where=positive_scale > np.finfo(float).tiny,
    )
    negative_response = None
    negative_ratio = np.zeros_like(positive_ratio)
    if ion_layout.negative_size:
        negative_response = final.storage_response[:, layout.negative_ion_slice]
        negative_inventory = negative_response @ widths[negative_nodes]
        negative_scale = np.abs(negative_response) @ widths[negative_nodes]
        negative_ratio = np.divide(
            np.abs(negative_inventory),
            negative_scale,
            out=np.zeros_like(negative_scale, dtype=float),
            where=negative_scale > np.finfo(float).tiny,
        )
    inventory_response = float(max(np.max(positive_ratio), np.max(negative_ratio)))

    bulk_balance = 0.0
    interface_balance = 0.0
    if bulk_layout is not None or interface_count:

        def diagnostic(values: np.ndarray, voltage: float) -> np.ndarray:
            result = dynamic_value(values, voltage)
            parts: list[np.ndarray] = []
            if result[3] is not None:
                parts.extend(
                    [
                        result[3].electron_capture_rate_m3_s,
                        result[3].hole_capture_rate_m3_s,
                    ]
                )
            if result[4] is not None:
                capture = np.asarray(result[4].qss.capture_flux_m2_s).reshape(-1, 4)
                parts.extend(
                    [capture[:, [0, 2]].sum(axis=1), capture[:, [1, 3]].sum(axis=1)]
                )
            return np.concatenate(parts)

        step = state_step * factors[-1]
        jacobian = np.empty((diagnostic(coordinate, float(V_dc)).size, layout.size))
        for column in range(layout.size):
            plus = coordinate.copy()
            minus = coordinate.copy()
            plus[column] += step
            minus[column] -= step
            jacobian[:, column] = (
                diagnostic(plus, float(V_dc)) - diagnostic(minus, float(V_dc))
            ) / (2.0 * step)
        vstep = voltage_step * factors[-1]
        voltage_derivative = (
            diagnostic(coordinate, float(V_dc + vstep))
            - diagnostic(coordinate, float(V_dc - vstep))
        ) / (2.0 * vstep)
        response = final.state_response @ jacobian.T + voltage_derivative
        cursor = 0
        omega = 2.0 * np.pi * frequencies
        if bulk_layout is not None:
            size = bulk_layout.size
            electron_capture = response[:, cursor : cursor + size]
            cursor += size
            hole_capture = response[:, cursor : cursor + size]
            cursor += size
            occupied = final.storage_response[:, layout.bulk_trap_slice]
            residual_value = (
                electron_capture - hole_capture - 1j * omega[:, None] * occupied
            )
            scale = (
                np.abs(electron_capture)
                + np.abs(hole_capture)
                + np.abs(1j * omega[:, None] * occupied)
            )
            bulk_balance = float(
                np.max(
                    np.divide(
                        np.abs(residual_value),
                        scale,
                        out=np.zeros_like(scale, dtype=float),
                        where=scale > 0.0,
                    )
                )
            )
        if interface_count:
            electron_capture = response[:, cursor : cursor + interface_count]
            cursor += interface_count
            hole_capture = response[:, cursor : cursor + interface_count]
            occupied = final.storage_response[:, layout.interface_trap_slice]
            residual_value = (
                electron_capture - hole_capture - 1j * omega[:, None] * occupied
            )
            scale = (
                np.abs(electron_capture)
                + np.abs(hole_capture)
                + np.abs(1j * omega[:, None] * occupied)
            )
            interface_balance = float(
                np.max(
                    np.divide(
                        np.abs(residual_value),
                        scale,
                        out=np.zeros_like(scale, dtype=float),
                        where=scale > 0.0,
                    )
                )
            )

    trap_rates: list[np.ndarray] = []
    if bulk_dynamic_dc is not None:
        trap_rates.append(np.asarray(bulk_dynamic_dc.relaxation_rate_s1, dtype=float))
    if interface_dynamic_dc is not None:
        trap_rates.append(
            _interface_relaxation_rate(
                interface_dynamic_dc,
                material,
                interface_trap_density,
                capture_velocities,
            )
        )
    all_trap_rates = np.concatenate(trap_rates)
    trap_frequencies = all_trap_rates / (2.0 * np.pi)
    margin = 10.0**frequency_branch_margin_decades
    gap = float(np.max(np.diff(np.log10(frequencies))))
    trap_min = float(np.min(trap_frequencies))
    trap_max = float(np.max(trap_frequencies))
    trap_low = bool(frequencies[0] <= trap_min / margin)
    trap_high = bool(frequencies[-1] >= trap_max * margin)
    trap_bracketed = bool(frequencies[0] < trap_min and frequencies[-1] > trap_max)
    sampling = bool(gap <= maximum_frequency_sampling_gap_decades)
    ionic_window = assess_impedance_frequency_window(
        grid,
        material,
        frequencies,
        branch_margin_decades=frequency_branch_margin_decades,
        max_sampling_gap_decades=maximum_frequency_sampling_gap_decades,
    )
    frequency_window = CombinedFrequencyWindow(
        ionic=ionic_window,
        minimum_trap_relaxation_frequency_Hz=trap_min,
        maximum_trap_relaxation_frequency_Hz=trap_max,
        trap_low_frequency_limit_covered=trap_low,
        trap_high_frequency_limit_covered=trap_high,
        every_trap_relaxation_frequency_bracketed=trap_bracketed,
        sampling_density_certified=sampling,
        certified=bool(
            trap_low
            and trap_high
            and trap_bracketed
            and sampling
            and ionic_window.ionic_branch_covered is True
        ),
    )
    max_spread = float(np.max(final.max_relative_face_spread))
    max_backward = float(np.max(final.backward_error))
    min_rcond = float(np.min(final.reciprocal_condition))
    max_refinement = max(refinement_changes, default=0.0)
    reasons: list[str] = []
    gates = (
        (dc_state.certificate.certified, "combined_dc_not_certified"),
        (
            qss_embedding <= maximum_qss_embedding_relative_error,
            "qss_embedding_mismatch",
        ),
        (
            bulk_balance <= maximum_trap_balance_relative_error,
            "bulk_trap_balance_exceeds_limit",
        ),
        (
            interface_balance <= maximum_trap_balance_relative_error,
            "interface_trap_balance_exceeds_limit",
        ),
        (
            max_spread <= maximum_all_face_admittance_spread,
            "all_face_admittance_spread_exceeds_limit",
        ),
        (
            max_backward <= maximum_linear_solve_backward_error,
            "linear_backward_error_exceeds_limit",
        ),
        (
            max_refinement <= maximum_refinement_relative_change,
            "finite_difference_refinement_not_converged",
        ),
        (
            inventory_response <= maximum_ion_inventory_response_relative,
            "blocking_ion_inventory_response_exceeds_limit",
        ),
        (
            decomposition_error <= maximum_current_decomposition_relative_error,
            "current_decomposition_exceeds_limit",
        ),
        (low_error <= maximum_limit_relative_error, "low_frequency_qss_limit_failed"),
        (
            high_error <= maximum_limit_relative_error,
            "high_frequency_frozen_limit_failed",
        ),
        (frequency_window.certified, "combined_frequency_window_not_covered"),
    )
    reasons.extend(reason for passed, reason in gates if not passed)
    if local_interface_residuals and max(local_interface_residuals) > 1.0e-7:
        reasons.append("local_interface_residual_exceeds_limit")
    if local_interface_gauss and max(local_interface_gauss) > 1.0e-7:
        reasons.append("local_interface_gauss_residual_exceeds_limit")
    certificate = DefectIonCombinedCertificate(
        capability=capability,
        dc_operating_point_certified=dc_state.certificate.certified,
        qss_embedding_relative_error=qss_embedding,
        maximum_bulk_trap_balance_relative_error=bulk_balance,
        maximum_interface_trap_balance_relative_error=interface_balance,
        maximum_all_face_admittance_spread=max_spread,
        maximum_linear_solve_backward_error=max_backward,
        minimum_reciprocal_condition=min_rcond,
        maximum_refinement_relative_change=max_refinement,
        maximum_ion_inventory_response_relative=inventory_response,
        maximum_current_decomposition_relative_error=decomposition_error,
        low_frequency_qss_relative_error=low_error,
        high_frequency_frozen_relative_error=high_error,
        frequency_window=frequency_window,
        certified=not reasons,
        reasons=tuple(dict.fromkeys(reasons)),
    )
    bulk_occupied = (
        None
        if bulk_layout is None
        else final.storage_response[:, layout.bulk_trap_slice]
    )
    interface_occupied = (
        None
        if interface_count == 0
        else final.storage_response[:, layout.interface_trap_slice]
    )
    bulk_charge = None
    bulk_occupancy_response = None
    if bulk_occupied is not None:
        charged = np.asarray(bulk_layout.charge_transitions) != NEUTRAL
        bulk_charge = -Q * (
            bulk_occupied[:, charged] @ widths[bulk_layout.device_node_indices[charged]]
        )
        bulk_occupancy_response = (
            bulk_occupied / bulk_layout.population_density_m3[np.newaxis, :]
        )
    interface_charge = (
        None if interface_occupied is None else -Q * np.sum(interface_occupied, axis=1)
    )
    interface_occupancy_response = (
        None
        if interface_occupied is None
        else interface_occupied / interface_trap_density[np.newaxis, :]
    )
    electron_storage = Q * (
        final.storage_response[:, layout.electron_slice] @ widths[1:-1]
    )
    hole_storage = Q * (final.storage_response[:, layout.hole_slice] @ widths[1:-1])
    positive_storage = Q * (positive_response @ widths[positive_nodes])
    negative_storage = (
        None
        if negative_response is None
        else Q * (negative_response @ widths[negative_nodes])
    )
    result = DefectIonCombinedResult(
        frequencies_Hz=_readonly(frequencies, dtype=float),
        impedance_ohm_m2=_readonly(final.impedance, dtype=complex),
        admittance_S_m2=_readonly(final.admittance, dtype=complex),
        admittance_faces_S_m2=_readonly(final.admittance_faces, dtype=complex),
        electron_admittance_faces_S_m2=_readonly(
            component_map["electron"], dtype=complex
        ),
        hole_admittance_faces_S_m2=_readonly(component_map["hole"], dtype=complex),
        positive_ion_admittance_faces_S_m2=_readonly(
            component_map["positive_ion"], dtype=complex
        ),
        negative_ion_admittance_faces_S_m2=(
            None
            if "negative_ion" not in component_map
            else _readonly(component_map["negative_ion"], dtype=complex)
        ),
        displacement_admittance_faces_S_m2=_readonly(
            final.displacement_admittance_faces, dtype=complex
        ),
        electron_storage_response_F_m2=_readonly(electron_storage, dtype=complex),
        hole_storage_response_F_m2=_readonly(hole_storage, dtype=complex),
        positive_ion_storage_response_F_m2=_readonly(positive_storage, dtype=complex),
        negative_ion_storage_response_F_m2=(
            None
            if negative_storage is None
            else _readonly(negative_storage, dtype=complex)
        ),
        bulk_trap_charge_storage_response_F_m2=(
            None if bulk_charge is None else _readonly(bulk_charge, dtype=complex)
        ),
        interface_sheet_charge_storage_response_F_m2=(
            None
            if interface_charge is None
            else _readonly(interface_charge, dtype=complex)
        ),
        bulk_trap_occupancy_response_per_V=(
            None
            if bulk_occupancy_response is None
            else _readonly(bulk_occupancy_response, dtype=complex)
        ),
        interface_occupancy_response_per_V=(
            None
            if interface_occupancy_response is None
            else _readonly(interface_occupancy_response, dtype=complex)
        ),
        state_response_per_V=_readonly(final.state_response, dtype=complex),
        layout=layout,
        dc_state=dc_state,
        reference_linearization=final,
        reference_linearizations=tuple(levels),
        qss_reference_admittance_S_m2=complex(qss_reference.admittance[0]),
        frozen_reference_admittance_S_m2=complex(frozen_reference.admittance[0]),
        interface_current_observation=(
            "symmetric_adjacent_physical_faces"
            if has_interface
            else "ordinary_finite_volume_faces"
        ),
        refinement_factors=factors,
        refinement_relative_changes=refinement_changes,
        certificate=certificate,
    )
    if require_certificate and not certificate.certified:
        raise DefectIonCombinedCertificationError(
            "combined defect/ion impedance certificate failed: "
            + ", ".join(certificate.reasons),
            result,
        )
    return result


__all__ = [
    "DEFECT_ION_COMBINED_SCOPE",
    "DEFECT_ION_COMBINED_VERSION",
    "CombinedDCCertificate",
    "CombinedDCState",
    "CombinedFrequencyWindow",
    "CombinedIonLayout",
    "CombinedStateLayout",
    "DefectIonCombinedCertificate",
    "DefectIonCombinedCertificationError",
    "DefectIonCombinedError",
    "DefectIonCombinedResult",
    "InterfaceCurrentObservation",
    "run_defect_ion_combined_impedance",
]
