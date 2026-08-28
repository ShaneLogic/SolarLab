"""Research device AC closure for one dynamic trap per physical interface.

Each physical two-sided interface owns one shared electron occupancy. The
occupancy drives four microscopic capture legs, an equilibrium-referenced
sheet charge, and one conserved areal-population row. Local carrier and
electrostatic traces are eliminated again at every nonlinear evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import math
from typing import Callable

import numpy as np

from perovskite_sim.constants import EPS_0, Q
from perovskite_sim.experiments.quasi_fermi_impedance import (
    MAX_LINEAR_PERTURBATION_V,
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
from perovskite_sim.physics.two_sided_interface import TWO_SIDED_TRACE
from perovskite_sim.solver.mol import _harmonic_face_average
from perovskite_sim.solver.small_signal import (
    FrequencyDomainResult,
    SmallSignalCurrentComponent,
    SmallSignalEvaluation,
    SmallSignalLinearizationError,
    solve_frequency_domain,
)


INTERFACE_DEFECT_DEVICE_AC_SCOPE = (
    "research_two_sided_dynamic_interface_defect_device_ac_only"
)
INTERFACE_DEFECT_DEVICE_AC_VERSION = "two-sided-dynamic-interface-device-ac-v1"
DEFAULT_REFINEMENT_FACTORS = (1.0, 0.5, 0.25)
ProgressCallback = Callable[[str, int, int, str], None]


class InterfaceDefectDeviceACError(SmallSignalLinearizationError):
    """The dynamic-interface device AC contract failed closed."""


class InterfaceDefectDeviceACCertificationError(InterfaceDefectDeviceACError):
    """A finite response failed one or more declared evidence gates."""

    def __init__(self, message: str, result: "InterfaceDefectDeviceACResult") -> None:
        self.result = result
        super().__init__(message)


def _readonly(value: object, *, dtype: object) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _symmetric_relative_error(left: object, right: object) -> float:
    left_values = np.asarray(left)
    right_values = np.asarray(right)
    scale = np.maximum(np.abs(left_values), np.abs(right_values))
    floor = max(float(np.max(scale)) * 1.0e-15, np.finfo(float).tiny)
    return float(np.max(np.abs(left_values - right_values) / np.maximum(scale, floor)))


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
    try:
        factors = tuple(float(item) for item in value)
    except TypeError as exc:
        raise TypeError("refinement_factors must be iterable") from exc
    if (
        len(factors) < 2
        or not all(math.isfinite(item) and item > 0.0 for item in factors)
        or any(right >= left for left, right in zip(factors, factors[1:]))
    ):
        raise ValueError(
            "refinement_factors must contain at least two positive decreasing values"
        )
    return factors


def _occupancy_logit(occupancy: np.ndarray) -> np.ndarray:
    values = np.asarray(occupancy, dtype=float)
    if (
        values.ndim != 1
        or values.size == 0
        or not np.all(np.isfinite(values))
        or np.any((values <= 0.0) | (values >= 1.0))
    ):
        raise InterfaceDefectDeviceACError(
            "DC interface occupancies must lie strictly inside (0, 1)"
        )
    return np.log(values) - np.log1p(-values)


def _occupancy_from_increment(
    reference_logit: np.ndarray, increment: np.ndarray
) -> np.ndarray:
    coordinates = np.asarray(increment, dtype=float)
    if coordinates.shape != reference_logit.shape or not np.all(
        np.isfinite(coordinates)
    ):
        raise ValueError("interface occupancy coordinates are invalid")
    logits = reference_logit + coordinates
    result = np.empty_like(logits)
    positive = logits >= 0.0
    result[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exponential = np.exp(logits[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    if np.any((result <= 0.0) | (result >= 1.0)):
        raise InterfaceDefectDeviceACError(
            "interface occupancy perturbation reached a saturated floating-point bound"
        )
    return result


@dataclass(frozen=True, slots=True)
class InterfaceDefectFrequencyWindow:
    """Coverage of all device-level interface-trap relaxation corners."""

    minimum_relaxation_frequency_Hz: float
    maximum_relaxation_frequency_Hz: float
    requested_minimum_frequency_Hz: float
    requested_maximum_frequency_Hz: float
    maximum_sampling_gap_decades: float
    low_frequency_limit_covered: bool
    high_frequency_limit_covered: bool
    every_relaxation_frequency_bracketed: bool
    sampling_density_certified: bool
    certified: bool


@dataclass(frozen=True, slots=True)
class InterfaceDefectDeviceACCertificate:
    """Provenance, DC, conservation, numerical, and asymptotic evidence."""

    dark_reference_certified: bool
    microscopic_binding_certified: bool
    dc_operating_point_certified: bool
    dc_state_operator_match_error: float
    dc_maximum_normalized_residual: float
    dc_electron_continuity_bound_A_m2: float
    dc_hole_continuity_bound_A_m2: float
    dc_face_current_spread_A_m2: float
    dc_poisson_residual: float
    qss_embedding_normalized_error: float
    maximum_local_interface_residual: float
    maximum_local_gauss_residual: float
    maximum_local_trap_balance_relative_error: float
    maximum_all_face_admittance_spread: float
    maximum_linear_solve_backward_error: float
    minimum_reciprocal_condition: float
    maximum_refinement_relative_change: float
    low_frequency_qss_relative_error: float
    high_frequency_frozen_relative_error: float
    frequency_window: InterfaceDefectFrequencyWindow
    certified: bool
    reasons: tuple[str, ...]
    scope: str = INTERFACE_DEFECT_DEVICE_AC_SCOPE
    version: str = INTERFACE_DEFECT_DEVICE_AC_VERSION


@dataclass(frozen=True, slots=True)
class InterfaceDefectDeviceACResult:
    """Dynamic two-sided-interface admittance and resolved state channels."""

    frequencies_Hz: np.ndarray
    impedance_ohm_m2: np.ndarray
    admittance_S_m2: np.ndarray
    admittance_faces_S_m2: np.ndarray
    electron_conduction_admittance_faces_S_m2: np.ndarray
    hole_conduction_admittance_faces_S_m2: np.ndarray
    displacement_admittance_faces_S_m2: np.ndarray
    electron_storage_response_F_m2: np.ndarray
    hole_storage_response_F_m2: np.ndarray
    interface_sheet_charge_storage_response_F_m2: np.ndarray
    interface_occupied_population_response_m2_V: np.ndarray
    interface_occupancy_response_per_V: np.ndarray
    interface_trace_state_response_m3_V: np.ndarray
    electron_capture_response_m2_s_V: np.ndarray
    hole_capture_response_m2_s_V: np.ndarray
    qss_reference_admittance_S_m2: complex
    frozen_reference_admittance_S_m2: complex
    refinement_factors: tuple[float, ...]
    refinement_relative_changes: tuple[float, ...]
    dark_reference: EquilibriumReferencedInterfaceChargeDarkReference
    dc_state: QuasiFermiSteadyStateResult
    certificate: InterfaceDefectDeviceACCertificate
    scope: str = INTERFACE_DEFECT_DEVICE_AC_SCOPE
    version: str = INTERFACE_DEFECT_DEVICE_AC_VERSION


def _assess_frequency_window(
    frequencies: np.ndarray,
    relaxation_rate_s1: np.ndarray,
    *,
    branch_margin_decades: float,
    maximum_sampling_gap_decades: float,
) -> InterfaceDefectFrequencyWindow:
    corners = np.asarray(relaxation_rate_s1, dtype=float) / (2.0 * np.pi)
    if (
        corners.ndim != 1
        or corners.size == 0
        or not np.all(np.isfinite(corners))
        or np.any(corners <= 0.0)
    ):
        raise InterfaceDefectDeviceACError(
            "interface-trap relaxation frequencies must be finite and positive"
        )
    margin = 10.0 ** float(branch_margin_decades)
    gap = float(np.max(np.diff(np.log10(frequencies))))
    minimum = float(np.min(corners))
    maximum = float(np.max(corners))
    low = bool(frequencies[0] <= minimum / margin)
    high = bool(frequencies[-1] >= maximum * margin)
    bracketed = bool(frequencies[0] < minimum and frequencies[-1] > maximum)
    sampled = bool(gap <= maximum_sampling_gap_decades)
    return InterfaceDefectFrequencyWindow(
        minimum_relaxation_frequency_Hz=minimum,
        maximum_relaxation_frequency_Hz=maximum,
        requested_minimum_frequency_Hz=float(frequencies[0]),
        requested_maximum_frequency_Hz=float(frequencies[-1]),
        maximum_sampling_gap_decades=gap,
        low_frequency_limit_covered=low,
        high_frequency_limit_covered=high,
        every_relaxation_frequency_bracketed=bracketed,
        sampling_density_certified=sampled,
        certified=bool(low and high and bracketed and sampled),
    )


def _current_evaluation(
    value,
    *,
    grid: np.ndarray,
    polarity: float,
    eps_face: np.ndarray,
    trap_density_m2: np.ndarray,
    trap_storage: bool,
) -> SmallSignalEvaluation:
    carrier_storage = np.r_[
        value.y[1 : grid.size - 1],
        value.y[grid.size + 1 : 2 * grid.size - 1],
    ]
    carrier_rate = np.r_[value.rate_n[1:-1], value.rate_p[1:-1]]
    if trap_storage:
        dynamic = value.interface_charge_dynamic
        if dynamic is None or dynamic.qss.capture_flux_m2_s is None:
            raise InterfaceDefectDeviceACError(
                "dynamic interface capture/storage evidence is unavailable"
            )
        capture = np.asarray(dynamic.qss.capture_flux_m2_s).reshape(-1, 4)
        trap_rate = capture[:, [0, 2]].sum(axis=1) - capture[:, [1, 3]].sum(axis=1)
        storage = np.r_[carrier_storage, trap_density_m2 * dynamic.occupancy]
        rate = np.r_[carrier_rate, trap_rate]
    else:
        storage = carrier_storage
        rate = carrier_rate
    electron = polarity * np.asarray(value.current_n, dtype=float)
    hole = polarity * np.asarray(value.current_p, dtype=float)
    electric_field = -np.diff(value.phi) / np.diff(grid)
    displacement_charge = polarity * eps_face * electric_field
    return SmallSignalEvaluation(
        storage=storage,
        rate=rate,
        conduction_current_faces=electron + hole,
        displacement_charge_faces=displacement_charge,
        current_components=(
            SmallSignalCurrentComponent("electron", electron),
            SmallSignalCurrentComponent("hole", hole),
        ),
    )


def run_interface_defect_device_impedance(
    x: np.ndarray,
    stack: DeviceStack,
    frequencies_Hz: object,
    *,
    V_dc: float = 0.0,
    delta_V: float = 0.01,
    illuminated: bool = False,
    dark_reference: EquilibriumReferencedInterfaceChargeDarkReference | None = None,
    dc_state: QuasiFermiSteadyStateResult | None = None,
    state_step: float = 1.0e-5,
    voltage_step: float = 1.0e-5,
    refinement_factors: object = DEFAULT_REFINEMENT_FACTORS,
    frequency_branch_margin_decades: float = 2.0,
    maximum_frequency_sampling_gap_decades: float = 0.5,
    maximum_dc_operator_match_error: float = 1.0e-10,
    maximum_dc_normalized_residual: float = 1.0e-7,
    maximum_dc_continuity_bound_A_m2: float = 1.0e-4,
    maximum_dc_face_current_spread_A_m2: float = 1.0e-4,
    maximum_dc_poisson_residual: float = 1.0e-8,
    maximum_qss_embedding_normalized_error: float = 1.0e-9,
    maximum_local_interface_residual: float = 1.0e-7,
    maximum_local_gauss_residual: float = 1.0e-7,
    maximum_local_trap_balance_relative_error: float = 1.0e-4,
    maximum_all_face_admittance_spread: float = 5.0e-4,
    maximum_linear_solve_backward_error: float = 1.0e-10,
    maximum_refinement_relative_change: float = 2.0e-3,
    maximum_limit_relative_error: float = 3.0e-2,
    require_certificate: bool = True,
    progress: ProgressCallback | None = None,
) -> InterfaceDefectDeviceACResult:
    """Solve and certify dynamic two-sided-interface device impedance."""
    grid = np.asarray(x, dtype=float)
    if grid.ndim != 1 or grid.size < 3 or np.any(np.diff(grid) <= 0.0):
        raise ValueError("x must be a strictly increasing one-dimensional grid")
    frequencies = _validate_frequencies(frequencies_Hz)
    factors = _validate_refinement_factors(refinement_factors)
    if not np.isfinite(delta_V) or not 0.0 < delta_V < MAX_LINEAR_PERTURBATION_V:
        raise ValueError("delta_V must be positive and below the 20 mV limit")
    if not np.isfinite(V_dc):
        raise ValueError("V_dc must be finite")
    scalar_limits = {
        "state_step": state_step,
        "voltage_step": voltage_step,
        "frequency_branch_margin_decades": frequency_branch_margin_decades,
        "maximum_frequency_sampling_gap_decades": (
            maximum_frequency_sampling_gap_decades
        ),
        "maximum_dc_operator_match_error": maximum_dc_operator_match_error,
        "maximum_dc_normalized_residual": maximum_dc_normalized_residual,
        "maximum_dc_continuity_bound_A_m2": maximum_dc_continuity_bound_A_m2,
        "maximum_dc_face_current_spread_A_m2": maximum_dc_face_current_spread_A_m2,
        "maximum_dc_poisson_residual": maximum_dc_poisson_residual,
        "maximum_qss_embedding_normalized_error": (
            maximum_qss_embedding_normalized_error
        ),
        "maximum_local_interface_residual": maximum_local_interface_residual,
        "maximum_local_gauss_residual": maximum_local_gauss_residual,
        "maximum_local_trap_balance_relative_error": (
            maximum_local_trap_balance_relative_error
        ),
        "maximum_all_face_admittance_spread": maximum_all_face_admittance_spread,
        "maximum_linear_solve_backward_error": maximum_linear_solve_backward_error,
        "maximum_refinement_relative_change": maximum_refinement_relative_change,
        "maximum_limit_relative_error": maximum_limit_relative_error,
    }
    for name, value in scalar_limits.items():
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")

    try:
        charge_off_stack, microscopic_contract = _research_charge_off_stack(stack)
    except (TypeError, ValueError) as exc:
        raise InterfaceDefectDeviceACError(
            f"interface device AC requires a microscopic charged-interface stack: {exc}"
        ) from exc
    interface_count = len(microscopic_contract.documents)
    if interface_count == 0:
        raise InterfaceDefectDeviceACError(
            "interface device AC requires at least one microscopic interface defect"
        )
    reference = dark_reference
    if reference is None:
        try:
            reference = build_equilibrium_referenced_interface_charge_dark_reference(
                grid,
                stack,
            )
        except (QuasiFermiSteadyStateError, TypeError, ValueError) as exc:
            raise InterfaceDefectDeviceACError(
                f"interface device AC could not build its dark reference: {exc}"
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
        raise InterfaceDefectDeviceACError(
            "dark-reference provenance or microscopic interface binding is invalid"
        )

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
        raise InterfaceDefectDeviceACError(
            f"interface-only dynamic AC capability gate failed: {exc}"
        ) from exc
    if material.monovalent_bulk_defects is not None:
        raise InterfaceDefectDeviceACError(
            "combined bulk and interface defects are reserved for D5-E2c"
        )

    operating_point = dc_state
    if operating_point is None:
        try:
            operating_point = (
                solve_equilibrium_referenced_interface_charge_steady_state(
                    grid,
                    stack,
                    float(V_dc),
                    dark_reference=reference,
                    illuminated=illuminated,
                )
            )
        except (QuasiFermiSteadyStateError, TypeError, ValueError) as exc:
            raise InterfaceDefectDeviceACError(
                f"interface device AC could not certify its DC state: {exc}"
            ) from exc
    if not operating_point.certified:
        raise InterfaceDefectDeviceACError("device AC requires a certified QF DC state")
    if not np.isclose(operating_point.V_app, V_dc, rtol=0.0, atol=1.0e-12):
        raise InterfaceDefectDeviceACError("DC-state voltage does not match V_dc")
    if bool(operating_point.illuminated) != bool(illuminated):
        raise InterfaceDefectDeviceACError("DC-state illumination does not match")
    if (
        not operating_point.interface_boundary
        or operating_point.interface_topology != TWO_SIDED_TRACE
        or operating_point.interface_transport_model != FERMI_DIRAC_RICHARDSON
        or operating_point.interface_charge_closure != "equilibrium_referenced"
    ):
        raise InterfaceDefectDeviceACError(
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
        raise InterfaceDefectDeviceACError(
            "DC-state interface-charge provenance does not match the dark reference"
        )

    system = _QuasiFermiSystem(
        grid,
        charge_off_stack,
        material,
        float(V_dc),
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
    interior_count = grid.size - 2
    thermal_voltage = float(material.V_T_device)
    qfn_reference = np.asarray(
        operating_point.electron_quasi_fermi_reference_V,
        dtype=float,
    )
    qfp_reference = np.asarray(
        operating_point.hole_quasi_fermi_reference_V,
        dtype=float,
    )
    dqfn_dc = np.asarray(
        operating_point.electron_quasi_fermi_increment_V,
        dtype=float,
    )
    dqfp_dc = np.asarray(
        operating_point.hole_quasi_fermi_increment_V,
        dtype=float,
    )
    edge_drop_n = np.asarray(
        operating_point.electron_quasi_fermi_edge_drop_V,
        dtype=float,
    )
    edge_drop_p = np.asarray(
        operating_point.hole_quasi_fermi_edge_drop_V,
        dtype=float,
    )
    qf_arrays = (qfn_reference, qfp_reference, dqfn_dc, dqfp_dc)
    if any(
        values.shape != grid.shape or not np.all(np.isfinite(values))
        for values in qf_arrays
    ):
        raise InterfaceDefectDeviceACError("DC state has invalid QF reference arrays")
    if (
        edge_drop_n.shape != (grid.size - 1,)
        or edge_drop_p.shape != (grid.size - 1,)
        or not np.all(np.isfinite(edge_drop_n))
        or not np.all(np.isfinite(edge_drop_p))
    ):
        raise InterfaceDefectDeviceACError(
            "DC state lacks cancellation-safe QF face-drop evidence"
        )
    if not (
        np.array_equal(qfn_reference, system.qfn0)
        and np.array_equal(qfp_reference, system.qfp0)
    ):
        raise InterfaceDefectDeviceACError(
            "DC QF reference does not match the interface dynamic operator"
        )
    edge_increment_dc_n = edge_drop_n / thermal_voltage - system.reference_edge_drop_n
    edge_increment_dc_p = edge_drop_p / thermal_voltage - system.reference_edge_drop_p

    def qf_coordinates(
        carrier_coordinate: np.ndarray,
        voltage: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        values = np.asarray(carrier_coordinate, dtype=float)
        if values.shape != (2 * interior_count,):
            raise ValueError("carrier coordinate does not match interior QF states")
        dqfn = dqfn_dc.copy()
        dqfp = dqfp_dc.copy()
        dqfn[1:-1] += thermal_voltage * values[:interior_count]
        dqfp[1:-1] += thermal_voltage * values[interior_count:]
        _contact_quasi_fermi_increments(
            dqfn,
            dqfp,
            qfn_reference,
            qfp_reference,
            material,
            voltage,
        )
        edge_n = edge_increment_dc_n + np.diff(dqfn - dqfn_dc) / thermal_voltage
        edge_p = edge_increment_dc_p + np.diff(dqfp - dqfp_dc) / thermal_voltage
        return dqfn, dqfp, edge_n, edge_p

    illumination_fraction = 1.0 if illuminated else 0.0
    qss_dc_value = system._evaluate_increments(
        dqfn_dc,
        dqfp_dc,
        illumination_fraction,
        V_app=float(V_dc),
        edge_increment_n=edge_increment_dc_n,
        edge_increment_p=edge_increment_dc_p,
    )
    if qss_dc_value.interface_charge_qss is None:
        raise InterfaceDefectDeviceACError(
            "fresh DC evaluation lacks charged interface evidence"
        )
    interior_mask = np.ones(grid.size, dtype=bool)
    interior_mask[[0, -1]] = False
    current_scale = max(
        float(np.max(np.abs(qss_dc_value.current_n + qss_dc_value.current_p))),
        abs(Q * float(stack.Phi)),
        1.0,
    )
    operating_y = np.asarray(operating_point.y, dtype=float)[: 2 * grid.size]
    operator_match = max(
        _symmetric_relative_error(operating_y, qss_dc_value.y[: 2 * grid.size]),
        float(np.max(np.abs(np.asarray(operating_point.phi) - qss_dc_value.phi)))
        / thermal_voltage,
        float(
            np.max(
                np.abs(
                    np.asarray(operating_point.electron_face_current_A_m2)
                    - qss_dc_value.current_n
                )
            )
        )
        / current_scale,
        float(
            np.max(
                np.abs(
                    np.asarray(operating_point.hole_face_current_A_m2)
                    - qss_dc_value.current_p
                )
            )
        )
        / current_scale,
        _symmetric_relative_error(
            np.asarray(operating_point.interface_occupancy),
            qss_dc_value.interface_charge_qss.qss.occupancy,
        ),
    )
    dc_maximum_normalized_residual = float(np.max(np.abs(qss_dc_value.residual)))
    widths = np.asarray(material.dx_cell, dtype=float)
    dc_electron_continuity_bound = float(
        Q * np.sum(np.abs(qss_dc_value.rate_n[interior_mask]) * widths[interior_mask])
    )
    dc_hole_continuity_bound = float(
        Q * np.sum(np.abs(qss_dc_value.rate_p[interior_mask]) * widths[interior_mask])
    )
    dc_face_current_spread = float(
        np.ptp(
            -float(material.junction_polarity)
            * (qss_dc_value.current_n + qss_dc_value.current_p)
        )
    )
    dc_poisson_residual = float(qss_dc_value.poisson_residual)
    dc_gates = {
        "operator match": (operator_match, maximum_dc_operator_match_error),
        "normalized residual": (
            dc_maximum_normalized_residual,
            maximum_dc_normalized_residual,
        ),
        "electron continuity bound": (
            dc_electron_continuity_bound,
            maximum_dc_continuity_bound_A_m2,
        ),
        "hole continuity bound": (
            dc_hole_continuity_bound,
            maximum_dc_continuity_bound_A_m2,
        ),
        "face-current spread": (
            dc_face_current_spread,
            maximum_dc_face_current_spread_A_m2,
        ),
        "Poisson residual": (dc_poisson_residual, maximum_dc_poisson_residual),
    }
    dc_failures = [
        f"{name}={value:.6g} > {limit:.6g}"
        for name, (value, limit) in dc_gates.items()
        if not np.isfinite(value) or value > limit
    ]
    if dc_failures:
        raise InterfaceDefectDeviceACError(
            "DC state is not certified on the interface AC operator: "
            + "; ".join(dc_failures)
        )

    occupancy_dc = np.asarray(
        qss_dc_value.interface_charge_qss.qss.occupancy,
        dtype=float,
    )
    trap_density = np.asarray(reference.trap_density_m2, dtype=float)
    reference_logit = _occupancy_logit(occupancy_dc)
    dynamic_dc_value = system.evaluate_quasi_fermi_increments_dynamic_interface(
        dqfn_dc,
        dqfp_dc,
        occupancy_dc,
        illumination_fraction,
        V_app=float(V_dc),
        edge_increment_n=edge_increment_dc_n,
        edge_increment_p=edge_increment_dc_p,
    )
    dynamic_dc = dynamic_dc_value.interface_charge_dynamic
    if dynamic_dc is None or dynamic_dc.qss.capture_flux_m2_s is None:
        raise InterfaceDefectDeviceACError(
            "fixed-occupancy DC embedding lacks local interface evidence"
        )
    qss_interface = qss_dc_value.interface_charge_qss
    qss_embedding_error = max(
        float(np.max(np.abs(dynamic_dc_value.phi - qss_dc_value.phi)))
        / thermal_voltage,
        Q
        * float(
            np.sum(
                (
                    np.abs(dynamic_dc_value.rate_n - qss_dc_value.rate_n)
                    + np.abs(dynamic_dc_value.rate_p - qss_dc_value.rate_p)
                )
                * widths
            )
        )
        / current_scale,
        float(np.max(np.abs(dynamic_dc_value.current_n - qss_dc_value.current_n)))
        / current_scale,
        float(np.max(np.abs(dynamic_dc_value.current_p - qss_dc_value.current_p)))
        / current_scale,
        _symmetric_relative_error(dynamic_dc.qss.state_m3, qss_interface.qss.state_m3),
        _symmetric_relative_error(
            dynamic_dc.incremental_sheet_charge_C_m2,
            qss_interface.incremental_sheet_charge_C_m2,
        ),
    )
    local_interface_residuals = [float(dynamic_dc.qss.normalized_residual)]
    local_gauss_residuals = [float(np.max(dynamic_dc.normalized_gauss_residual))]

    eps_face = EPS_0 * _harmonic_face_average(material.eps_r)
    polarity = float(material.junction_polarity)

    def dynamic_value(coordinate: np.ndarray, voltage: float):
        values = np.asarray(coordinate, dtype=float)
        expected = 2 * interior_count + interface_count
        if values.shape != (expected,):
            raise ValueError("dynamic interface coordinate has the wrong size")
        occupancy = _occupancy_from_increment(
            reference_logit,
            values[2 * interior_count :],
        )
        dqfn, dqfp, edge_n, edge_p = qf_coordinates(
            values[: 2 * interior_count],
            voltage,
        )
        result = system.evaluate_quasi_fermi_increments_dynamic_interface(
            dqfn,
            dqfp,
            occupancy,
            illumination_fraction,
            V_app=voltage,
            edge_increment_n=edge_n,
            edge_increment_p=edge_p,
        )
        dynamic = result.interface_charge_dynamic
        if dynamic is None:
            raise InterfaceDefectDeviceACError(
                "dynamic interface evaluation lost its fixed-occupancy evidence"
            )
        local_interface_residuals.append(float(dynamic.qss.normalized_residual))
        local_gauss_residuals.append(float(np.max(dynamic.normalized_gauss_residual)))
        return result, occupancy

    def dynamic_evaluate(
        coordinate: np.ndarray,
        voltage: float,
    ) -> SmallSignalEvaluation:
        value, _occupancy = dynamic_value(coordinate, voltage)
        return _current_evaluation(
            value,
            grid=grid,
            polarity=polarity,
            eps_face=eps_face,
            trap_density_m2=trap_density,
            trap_storage=True,
        )

    def qss_evaluate(
        coordinate: np.ndarray,
        voltage: float,
    ) -> SmallSignalEvaluation:
        dqfn, dqfp, edge_n, edge_p = qf_coordinates(coordinate, voltage)
        value = system._evaluate_increments(
            dqfn,
            dqfp,
            illumination_fraction,
            V_app=voltage,
            edge_increment_n=edge_n,
            edge_increment_p=edge_p,
        )
        return _current_evaluation(
            value,
            grid=grid,
            polarity=polarity,
            eps_face=eps_face,
            trap_density_m2=trap_density,
            trap_storage=False,
        )

    def frozen_evaluate(
        coordinate: np.ndarray,
        voltage: float,
    ) -> SmallSignalEvaluation:
        dqfn, dqfp, edge_n, edge_p = qf_coordinates(coordinate, voltage)
        value = system.evaluate_quasi_fermi_increments_dynamic_interface(
            dqfn,
            dqfp,
            occupancy_dc,
            illumination_fraction,
            V_app=voltage,
            edge_increment_n=edge_n,
            edge_increment_p=edge_p,
        )
        return _current_evaluation(
            value,
            grid=grid,
            polarity=polarity,
            eps_face=eps_face,
            trap_density_m2=trap_density,
            trap_storage=False,
        )

    face_weights = np.diff(grid) / float(grid[-1] - grid[0])
    coordinate = np.zeros(2 * interior_count + interface_count, dtype=float)
    levels: list[FrequencyDomainResult] = []
    for level, factor in enumerate(factors):
        if progress is not None:
            progress(
                "interface_defect_device_ac_refinement",
                level,
                len(factors),
                f"finite-difference factor {factor:g}",
            )
        levels.append(
            solve_frequency_domain(
                dynamic_evaluate,
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
        _symmetric_relative_error(coarse.admittance_faces, fine.admittance_faces)
        for coarse, fine in zip(levels, levels[1:])
    )
    reference_frequencies = np.array([frequencies[0], frequencies[-1]])
    qss_reference = solve_frequency_domain(
        qss_evaluate,
        np.zeros(2 * interior_count),
        float(V_dc),
        reference_frequencies,
        state_step=state_step * factors[-1],
        voltage_step=voltage_step * factors[-1],
        face_weights=face_weights,
    )
    frozen_reference = solve_frequency_domain(
        frozen_evaluate,
        np.zeros(2 * interior_count),
        float(V_dc),
        reference_frequencies,
        state_step=state_step * factors[-1],
        voltage_step=voltage_step * factors[-1],
        face_weights=face_weights,
    )
    low_error = _symmetric_relative_error(
        final.admittance_faces[0],
        qss_reference.admittance_faces[0],
    )
    high_error = _symmetric_relative_error(
        final.admittance_faces[-1],
        frozen_reference.admittance_faces[-1],
    )

    interior_widths = widths[1:-1]
    electron_storage = Q * (
        final.storage_response[:, :interior_count] @ interior_widths
    )
    hole_storage = Q * (
        final.storage_response[:, interior_count : 2 * interior_count] @ interior_widths
    )
    occupied_response = final.storage_response[:, 2 * interior_count :]
    occupancy_response = occupied_response / trap_density[np.newaxis, :]
    sheet_charge_storage = -Q * occupied_response

    def diagnostic_vector(values: np.ndarray, voltage: float) -> np.ndarray:
        evaluation, _occupancy = dynamic_value(values, voltage)
        dynamic = evaluation.interface_charge_dynamic
        if dynamic is None or dynamic.qss.capture_flux_m2_s is None:
            raise InterfaceDefectDeviceACError(
                "interface diagnostic evaluation lacks local state/capture data"
            )
        right_first_state = np.asarray(dynamic.qss.state_m3).reshape(
            interface_count,
            4,
        )
        right_first_capture = np.asarray(dynamic.qss.capture_flux_m2_s).reshape(
            interface_count,
            4,
        )
        canonical_order = np.array([2, 3, 0, 1])
        return np.r_[
            right_first_state[:, canonical_order].ravel(),
            right_first_capture[:, canonical_order].ravel(),
        ]

    diagnostic_size = 8 * interface_count
    diagnostic_jacobian = np.empty((diagnostic_size, coordinate.size), dtype=float)
    diagnostic_step = state_step * factors[-1]
    for column in range(coordinate.size):
        plus = coordinate.copy()
        minus = coordinate.copy()
        plus[column] += diagnostic_step
        minus[column] -= diagnostic_step
        diagnostic_jacobian[:, column] = (
            diagnostic_vector(plus, float(V_dc)) - diagnostic_vector(minus, float(V_dc))
        ) / (2.0 * diagnostic_step)
    diagnostic_voltage_step = voltage_step * factors[-1]
    diagnostic_voltage_derivative = (
        diagnostic_vector(coordinate, float(V_dc + diagnostic_voltage_step))
        - diagnostic_vector(coordinate, float(V_dc - diagnostic_voltage_step))
    ) / (2.0 * diagnostic_voltage_step)
    diagnostic_response = (
        final.state_response @ diagnostic_jacobian.T
        + diagnostic_voltage_derivative[np.newaxis, :]
    )
    trace_response = diagnostic_response[:, : 4 * interface_count].reshape(
        frequencies.size,
        interface_count,
        4,
    )
    capture_response = diagnostic_response[:, 4 * interface_count :].reshape(
        frequencies.size,
        interface_count,
        4,
    )
    electron_capture = capture_response[:, :, [0, 2]]
    hole_capture = capture_response[:, :, [1, 3]]

    omega = 2.0 * np.pi * frequencies
    trap_balance_residual = (
        electron_capture.sum(axis=2)
        - hole_capture.sum(axis=2)
        - 1j * omega[:, np.newaxis] * occupied_response
    )
    trap_balance_scale = (
        np.abs(electron_capture).sum(axis=2)
        + np.abs(hole_capture).sum(axis=2)
        + np.abs(1j * omega[:, np.newaxis] * occupied_response)
    )
    trap_balance_relative = np.divide(
        np.abs(trap_balance_residual),
        trap_balance_scale,
        out=np.zeros_like(trap_balance_scale, dtype=float),
        where=trap_balance_scale > 0.0,
    )
    max_trap_balance = float(np.max(trap_balance_relative))

    canonical_dc_state = np.asarray(dynamic_dc.qss.state_m3).reshape(
        interface_count,
        4,
    )[:, [2, 3, 0, 1]]
    velocities = np.asarray(microscopic_contract.capture_velocities_m_s, dtype=float)
    n1_left = np.asarray(material.interface_n1_L, dtype=float)
    n1_right = np.asarray(material.interface_n1_R, dtype=float)
    p1_left = np.asarray(material.interface_p1_L, dtype=float)
    p1_right = np.asarray(material.interface_p1_R, dtype=float)
    relaxation_rate = (
        velocities[:, 0]
        * (canonical_dc_state[:, 0] + canonical_dc_state[:, 2] + n1_left + n1_right)
        + velocities[:, 1]
        * (canonical_dc_state[:, 1] + canonical_dc_state[:, 3] + p1_left + p1_right)
    ) / trap_density
    window = _assess_frequency_window(
        frequencies,
        relaxation_rate,
        branch_margin_decades=frequency_branch_margin_decades,
        maximum_sampling_gap_decades=maximum_frequency_sampling_gap_decades,
    )
    maximum_local_interface_observed = max(local_interface_residuals)
    maximum_local_gauss_observed = max(local_gauss_residuals)
    max_face_spread = float(np.max(final.max_relative_face_spread))
    max_backward = float(np.max(final.backward_error))
    min_rcond = float(np.min(final.reciprocal_condition))
    max_refinement = max(refinement_changes, default=0.0)
    reasons: list[str] = []
    gates = (
        (
            qss_embedding_error <= maximum_qss_embedding_normalized_error,
            "qss_embedding_mismatch",
        ),
        (
            maximum_local_interface_observed <= maximum_local_interface_residual,
            "local_interface_residual_failed",
        ),
        (
            maximum_local_gauss_observed <= maximum_local_gauss_residual,
            "local_gauss_residual_failed",
        ),
        (
            max_trap_balance <= maximum_local_trap_balance_relative_error,
            "local_trap_balance_failed",
        ),
        (
            max_face_spread <= maximum_all_face_admittance_spread,
            "all_face_current_closure_failed",
        ),
        (
            max_backward <= maximum_linear_solve_backward_error,
            "linear_solve_backward_error_failed",
        ),
        (
            max_refinement <= maximum_refinement_relative_change,
            "finite_difference_refinement_failed",
        ),
        (low_error <= maximum_limit_relative_error, "low_frequency_qss_limit_failed"),
        (
            high_error <= maximum_limit_relative_error,
            "high_frequency_frozen_limit_failed",
        ),
        (window.certified, "trap_frequency_window_incomplete"),
    )
    reasons.extend(reason for passed, reason in gates if not passed)
    certificate = InterfaceDefectDeviceACCertificate(
        dark_reference_certified=dark_reference_certified,
        microscopic_binding_certified=microscopic_binding_certified,
        dc_operating_point_certified=bool(operating_point.certified),
        dc_state_operator_match_error=operator_match,
        dc_maximum_normalized_residual=dc_maximum_normalized_residual,
        dc_electron_continuity_bound_A_m2=dc_electron_continuity_bound,
        dc_hole_continuity_bound_A_m2=dc_hole_continuity_bound,
        dc_face_current_spread_A_m2=dc_face_current_spread,
        dc_poisson_residual=dc_poisson_residual,
        qss_embedding_normalized_error=qss_embedding_error,
        maximum_local_interface_residual=maximum_local_interface_observed,
        maximum_local_gauss_residual=maximum_local_gauss_observed,
        maximum_local_trap_balance_relative_error=max_trap_balance,
        maximum_all_face_admittance_spread=max_face_spread,
        maximum_linear_solve_backward_error=max_backward,
        minimum_reciprocal_condition=min_rcond,
        maximum_refinement_relative_change=max_refinement,
        low_frequency_qss_relative_error=low_error,
        high_frequency_frozen_relative_error=high_error,
        frequency_window=window,
        certified=not reasons,
        reasons=tuple(reasons),
    )
    component_map = {
        component.name: component.admittance_faces
        for component in final.current_components
    }
    if set(component_map) != {"electron", "hole"}:
        raise InterfaceDefectDeviceACError(
            "interface device AC current decomposition is incomplete"
        )
    result = InterfaceDefectDeviceACResult(
        frequencies_Hz=_readonly(final.frequencies, dtype=float),
        impedance_ohm_m2=_readonly(final.impedance, dtype=complex),
        admittance_S_m2=_readonly(final.admittance, dtype=complex),
        admittance_faces_S_m2=_readonly(final.admittance_faces, dtype=complex),
        electron_conduction_admittance_faces_S_m2=_readonly(
            component_map["electron"],
            dtype=complex,
        ),
        hole_conduction_admittance_faces_S_m2=_readonly(
            component_map["hole"],
            dtype=complex,
        ),
        displacement_admittance_faces_S_m2=_readonly(
            final.displacement_admittance_faces,
            dtype=complex,
        ),
        electron_storage_response_F_m2=_readonly(electron_storage, dtype=complex),
        hole_storage_response_F_m2=_readonly(hole_storage, dtype=complex),
        interface_sheet_charge_storage_response_F_m2=_readonly(
            sheet_charge_storage,
            dtype=complex,
        ),
        interface_occupied_population_response_m2_V=_readonly(
            occupied_response,
            dtype=complex,
        ),
        interface_occupancy_response_per_V=_readonly(
            occupancy_response,
            dtype=complex,
        ),
        interface_trace_state_response_m3_V=_readonly(
            trace_response,
            dtype=complex,
        ),
        electron_capture_response_m2_s_V=_readonly(
            electron_capture,
            dtype=complex,
        ),
        hole_capture_response_m2_s_V=_readonly(
            hole_capture,
            dtype=complex,
        ),
        qss_reference_admittance_S_m2=complex(qss_reference.admittance[0]),
        frozen_reference_admittance_S_m2=complex(frozen_reference.admittance[-1]),
        refinement_factors=factors,
        refinement_relative_changes=refinement_changes,
        dark_reference=reference,
        dc_state=operating_point,
        certificate=certificate,
    )
    if require_certificate and not certificate.certified:
        raise InterfaceDefectDeviceACCertificationError(
            "interface defect device AC certificate failed: " + ", ".join(reasons),
            result,
        )
    return result


__all__ = [
    "DEFAULT_REFINEMENT_FACTORS",
    "INTERFACE_DEFECT_DEVICE_AC_SCOPE",
    "INTERFACE_DEFECT_DEVICE_AC_VERSION",
    "InterfaceDefectDeviceACCertificate",
    "InterfaceDefectDeviceACCertificationError",
    "InterfaceDefectDeviceACError",
    "InterfaceDefectDeviceACResult",
    "InterfaceDefectFrequencyWindow",
    "run_interface_defect_device_impedance",
]
