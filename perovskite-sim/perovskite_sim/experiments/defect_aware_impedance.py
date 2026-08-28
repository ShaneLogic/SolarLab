"""Research device AC closure for dynamic monovalent bulk defects.

This adapter augments the cancellation-safe quasi-Fermi device coordinates
with one logit occupancy coordinate per interior spatial/energy defect state.
Trap charge enters the eliminated Poisson solve, electron and hole captures
enter their own continuity equations, and occupied population is a conserved
storage row.  The result remains research-only until the protocol, API, UI,
and clean refinement evidence are completed in D5-E3.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

from perovskite_sim.constants import EPS_0, Q
from perovskite_sim.experiments.quasi_fermi_impedance import (
    MAX_LINEAR_PERTURBATION_V,
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
from perovskite_sim.models.device import DeviceStack
from perovskite_sim.models.defects import ExplicitDefectCapabilityError, NEUTRAL
from perovskite_sim.physics.defect_distributions import (
    DEFAULT_DEFECT_ENERGY_QUADRATURE_ORDER,
    validate_defect_energy_quadrature_order,
)
from perovskite_sim.physics.dynamic_defect_state import (
    DynamicBulkTrapEvaluation,
    DynamicBulkTrapLayout,
    compile_dynamic_bulk_trap_layout,
    evaluate_dynamic_bulk_traps_about_qss,
    occupancy_from_logit_increment,
    occupancy_logit,
    quasi_steady_bulk_trap_occupancy,
)
from perovskite_sim.solver.mol import (
    EXPLICIT_DEFECT_CHARGE_QF_DC,
    MaterialArrays,
    _harmonic_face_average,
    build_material_arrays,
)
from perovskite_sim.solver.small_signal import (
    FrequencyDomainResult,
    SmallSignalCurrentComponent,
    SmallSignalEvaluation,
    SmallSignalLinearizationError,
    solve_frequency_domain,
)


BULK_DEFECT_DEVICE_AC_SCOPE = "research_bulk_dynamic_defect_device_ac_only"
BULK_DEFECT_DEVICE_AC_VERSION = "bulk-dynamic-defect-device-ac-v1"
DEFAULT_REFINEMENT_FACTORS = (1.0, 0.5, 0.25)
ProgressCallback = Callable[[str, int, int, str], None]


class BulkDefectDeviceACError(SmallSignalLinearizationError):
    """The bulk dynamic-defect device AC contract failed closed."""


class BulkDefectDeviceACCertificationError(BulkDefectDeviceACError):
    """A finite device response failed one or more declared evidence gates."""

    def __init__(self, message: str, result: "BulkDefectDeviceACResult") -> None:
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


@dataclass(frozen=True, slots=True)
class BulkDefectFrequencyWindow:
    """Coverage of every compiled trap relaxation corner."""

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
class BulkDefectDeviceACCertificate:
    """Device-level numerical, conservation, and asymptotic evidence."""

    dc_operating_point_certified: bool
    dc_maximum_normalized_residual: float
    dc_electron_continuity_bound_A_m2: float
    dc_hole_continuity_bound_A_m2: float
    dc_face_current_spread_A_m2: float
    dc_poisson_residual: float
    qss_embedding_normalized_error: float
    maximum_local_trap_balance_relative_error: float
    maximum_all_face_admittance_spread: float
    maximum_linear_solve_backward_error: float
    minimum_reciprocal_condition: float
    maximum_refinement_relative_change: float
    low_frequency_qss_relative_error: float
    high_frequency_frozen_relative_error: float
    frequency_window: BulkDefectFrequencyWindow
    certified: bool
    reasons: tuple[str, ...]
    scope: str = BULK_DEFECT_DEVICE_AC_SCOPE
    version: str = BULK_DEFECT_DEVICE_AC_VERSION


@dataclass(frozen=True, slots=True)
class BulkDefectDeviceACResult:
    """Dynamic-defect admittance and state/current decomposition."""

    frequencies_Hz: np.ndarray
    impedance_ohm_m2: np.ndarray
    admittance_S_m2: np.ndarray
    admittance_faces_S_m2: np.ndarray
    electron_conduction_admittance_faces_S_m2: np.ndarray
    hole_conduction_admittance_faces_S_m2: np.ndarray
    displacement_admittance_faces_S_m2: np.ndarray
    electron_storage_response_F_m2: np.ndarray
    hole_storage_response_F_m2: np.ndarray
    trap_charge_storage_response_F_m2: np.ndarray
    trap_occupied_population_response_m2_V: np.ndarray
    trap_occupancy_response_per_V: np.ndarray
    electron_capture_response_m3_s_V: np.ndarray
    hole_capture_response_m3_s_V: np.ndarray
    qss_reference_admittance_S_m2: complex
    frozen_reference_admittance_S_m2: complex
    refinement_factors: tuple[float, ...]
    refinement_relative_changes: tuple[float, ...]
    layout: DynamicBulkTrapLayout
    dc_state: QuasiFermiSteadyStateResult
    certificate: BulkDefectDeviceACCertificate
    scope: str = BULK_DEFECT_DEVICE_AC_SCOPE
    version: str = BULK_DEFECT_DEVICE_AC_VERSION


def _assess_frequency_window(
    frequencies: np.ndarray,
    relaxation_rate_s1: np.ndarray,
    *,
    branch_margin_decades: float,
    maximum_sampling_gap_decades: float,
) -> BulkDefectFrequencyWindow:
    corners = np.asarray(relaxation_rate_s1, dtype=float) / (2.0 * np.pi)
    if (
        corners.ndim != 1
        or corners.size == 0
        or not np.all(np.isfinite(corners))
        or np.any(corners <= 0.0)
    ):
        raise BulkDefectDeviceACError(
            "trap relaxation frequencies must be finite and positive"
        )
    margin = 10.0 ** float(branch_margin_decades)
    gap = float(np.max(np.diff(np.log10(frequencies))))
    minimum = float(np.min(corners))
    maximum = float(np.max(corners))
    low = bool(frequencies[0] <= minimum / margin)
    high = bool(frequencies[-1] >= maximum * margin)
    bracketed = bool(frequencies[0] < minimum and frequencies[-1] > maximum)
    sampled = bool(gap <= maximum_sampling_gap_decades)
    return BulkDefectFrequencyWindow(
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
    dynamic: DynamicBulkTrapEvaluation | None,
    *,
    grid: np.ndarray,
    polarity: float,
    eps_face: np.ndarray,
    trap_storage: bool,
) -> SmallSignalEvaluation:
    interior = grid.size - 2
    carrier_storage = np.r_[
        value.y[1 : grid.size - 1],
        value.y[grid.size + 1 : 2 * grid.size - 1],
    ]
    carrier_rate = np.r_[value.rate_n[1:-1], value.rate_p[1:-1]]
    if trap_storage:
        if dynamic is None:
            raise BulkDefectDeviceACError("dynamic trap storage was not evaluated")
        storage = np.r_[carrier_storage, dynamic.occupied_storage_m3]
        rate = np.r_[carrier_rate, dynamic.trap_storage_rate_m3_s]
    else:
        storage = carrier_storage
        rate = carrier_rate
    if storage.size != 2 * interior + (
        0 if dynamic is None or not trap_storage else dynamic.occupancy.size
    ):
        raise BulkDefectDeviceACError("device AC storage layout is inconsistent")
    electron = polarity * np.asarray(value.current_n, dtype=float)
    hole = polarity * np.asarray(value.current_p, dtype=float)
    conduction = electron + hole
    electric_field = -np.diff(value.phi) / np.diff(grid)
    displacement_charge = polarity * eps_face * electric_field
    return SmallSignalEvaluation(
        storage=storage,
        rate=rate,
        conduction_current_faces=conduction,
        displacement_charge_faces=displacement_charge,
        current_components=(
            SmallSignalCurrentComponent("electron", electron),
            SmallSignalCurrentComponent("hole", hole),
        ),
    )


def run_bulk_defect_device_impedance(
    x: np.ndarray,
    stack: DeviceStack,
    frequencies_Hz: object,
    *,
    V_dc: float = 0.0,
    delta_V: float = 0.01,
    illuminated: bool = False,
    mat: MaterialArrays | None = None,
    dc_state: QuasiFermiSteadyStateResult | None = None,
    defect_energy_quadrature_order: int = DEFAULT_DEFECT_ENERGY_QUADRATURE_ORDER,
    state_step: float = 1.0e-5,
    voltage_step: float = 1.0e-5,
    refinement_factors: object = DEFAULT_REFINEMENT_FACTORS,
    frequency_branch_margin_decades: float = 2.0,
    maximum_frequency_sampling_gap_decades: float = 0.5,
    maximum_dc_normalized_residual: float = 1.0e-10,
    maximum_dc_continuity_bound_A_m2: float = 1.0e-4,
    maximum_dc_face_current_spread_A_m2: float = 1.0e-4,
    maximum_dc_poisson_residual: float = 1.0e-8,
    maximum_qss_embedding_normalized_error: float = 1.0e-10,
    maximum_local_trap_balance_relative_error: float = 1.0e-4,
    maximum_all_face_admittance_spread: float = 5.0e-4,
    maximum_linear_solve_backward_error: float = 1.0e-10,
    maximum_refinement_relative_change: float = 2.0e-3,
    maximum_limit_relative_error: float = 3.0e-2,
    require_certificate: bool = True,
    progress: ProgressCallback | None = None,
) -> BulkDefectDeviceACResult:
    """Solve and certify the research-only bulk dynamic-defect device AC lane."""
    grid = np.asarray(x, dtype=float)
    if grid.ndim != 1 or grid.size < 3 or np.any(np.diff(grid) <= 0.0):
        raise ValueError("x must be a strictly increasing one-dimensional grid")
    frequencies = _validate_frequencies(frequencies_Hz)
    factors = _validate_refinement_factors(refinement_factors)
    if not np.isfinite(delta_V) or not 0.0 < delta_V < MAX_LINEAR_PERTURBATION_V:
        raise ValueError("delta_V must be positive and below the 20 mV limit")
    energy_order = validate_defect_energy_quadrature_order(
        defect_energy_quadrature_order
    )
    scalar_limits = {
        "state_step": state_step,
        "voltage_step": voltage_step,
        "frequency_branch_margin_decades": frequency_branch_margin_decades,
        "maximum_frequency_sampling_gap_decades": (
            maximum_frequency_sampling_gap_decades
        ),
        "maximum_dc_normalized_residual": maximum_dc_normalized_residual,
        "maximum_dc_continuity_bound_A_m2": maximum_dc_continuity_bound_A_m2,
        "maximum_dc_face_current_spread_A_m2": maximum_dc_face_current_spread_A_m2,
        "maximum_dc_poisson_residual": maximum_dc_poisson_residual,
        "maximum_qss_embedding_normalized_error": (
            maximum_qss_embedding_normalized_error
        ),
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
        material = (
            build_material_arrays(
                grid,
                stack,
                explicit_defect_charge_closure=EXPLICIT_DEFECT_CHARGE_QF_DC,
                defect_energy_quadrature_order=energy_order,
            )
            if mat is None
            else mat
        )
    except ExplicitDefectCapabilityError as exc:
        raise BulkDefectDeviceACError(
            f"bulk defect device AC requires a supported explicit-defect model: {exc}"
        ) from exc
    try:
        _require_material_defect_contract(
            stack,
            material,
            defect_energy_quadrature_order=energy_order,
        )
        _require_supported(material, allow_charged_bulk_defects=True)
    except QuasiFermiSteadyStateError as exc:
        raise BulkDefectDeviceACError(
            f"bulk defect device AC material contract failed: {exc}"
        ) from exc
    model = material.monovalent_bulk_defects
    if model is None:
        raise BulkDefectDeviceACError(
            "bulk defect device AC requires a compiled explicit-defect model"
        )
    if model.has_distributed_species and any(
        region.energy_quadrature_order != energy_order
        for region in model.regions
        if region.has_distributed_species
    ):
        raise BulkDefectDeviceACError(
            "material defect energy order does not match the requested AC model"
        )

    operating_point = dc_state
    if operating_point is None:
        try:
            operating_point = solve_quasi_fermi_steady_state(
                grid,
                stack,
                V_app=float(V_dc),
                illuminated=illuminated,
                mat=material,
                defect_energy_quadrature_order=energy_order,
            )
        except QuasiFermiSteadyStateError as exc:
            raise BulkDefectDeviceACError(
                f"bulk defect device AC could not certify its DC state: {exc}"
            ) from exc
    if not operating_point.certified:
        raise BulkDefectDeviceACError("device AC requires a certified QF DC state")
    if not np.isclose(operating_point.V_app, V_dc, rtol=0.0, atol=1.0e-12):
        raise BulkDefectDeviceACError("DC-state voltage does not match V_dc")
    if bool(operating_point.illuminated) != bool(illuminated):
        raise BulkDefectDeviceACError("DC-state illumination does not match")
    if operating_point.bulk_defect_diagnostics is None:
        raise BulkDefectDeviceACError("DC state lacks bulk defect diagnostics")
    if (
        operating_point.bulk_defect_diagnostics.model_identity_sha256
        != model.identity_sha256
    ):
        raise BulkDefectDeviceACError(
            "DC-state defect model identity does not match the AC material"
        )
    if (
        model.has_distributed_species
        and operating_point.defect_energy_quadrature_order != energy_order
    ):
        raise BulkDefectDeviceACError(
            "DC-state defect energy order does not match the AC material"
        )
    if operating_point.contact_thermodynamic_status != "certified":
        raise BulkDefectDeviceACError(
            "device AC requires a contact-thermodynamically certified DC state"
        )

    dynamic_mask = np.ones(grid.size, dtype=bool)
    dynamic_mask[[0, -1]] = False
    layout = compile_dynamic_bulk_trap_layout(
        model,
        dynamic_node_mask=dynamic_mask,
    )
    system = _QuasiFermiSystem(
        grid,
        stack,
        material,
        float(V_dc),
        poisson_tolerance_V=1.0e-13,
        poisson_max_iterations=100,
    )
    interior_count = grid.size - 2
    thermal_voltage = material.V_T_device
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
    qf_arrays = (qfn_reference, qfp_reference, dqfn_dc, dqfp_dc)
    if any(
        value.shape != grid.shape or not np.all(np.isfinite(value))
        for value in qf_arrays
    ):
        raise BulkDefectDeviceACError("DC state has invalid QF reference arrays")
    if not (
        np.array_equal(qfn_reference, system.qfn0)
        and np.array_equal(qfp_reference, system.qfp0)
    ):
        raise BulkDefectDeviceACError(
            "DC QF reference does not match the dynamic device operator"
        )

    def qf_coordinates(
        carrier_coordinate: np.ndarray,
        voltage: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        dqfn = dqfn_dc.copy()
        dqfp = dqfp_dc.copy()
        dqfn[1:-1] += thermal_voltage * carrier_coordinate[:interior_count]
        dqfp[1:-1] += thermal_voltage * carrier_coordinate[interior_count:]
        _contact_quasi_fermi_increments(
            dqfn,
            dqfp,
            qfn_reference,
            qfp_reference,
            material,
            voltage,
        )
        return dqfn, dqfp

    qss_dc_value = system.evaluate_quasi_fermi_increments(
        dqfn_dc,
        dqfp_dc,
        1.0 if illuminated else 0.0,
        V_app=float(V_dc),
    )
    interior = np.ones(grid.size, dtype=bool)
    interior[[0, -1]] = False
    dc_maximum_normalized_residual = float(np.max(np.abs(qss_dc_value.residual)))
    dc_electron_continuity_bound = float(
        Q
        * np.sum(
            np.abs(qss_dc_value.rate_n[interior])
            * np.asarray(material.dx_cell)[interior]
        )
    )
    dc_hole_continuity_bound = float(
        Q
        * np.sum(
            np.abs(qss_dc_value.rate_p[interior])
            * np.asarray(material.dx_cell)[interior]
        )
    )
    dc_face_current_spread = float(
        np.ptp(
            -float(material.junction_polarity)
            * (qss_dc_value.current_n + qss_dc_value.current_p)
        )
    )
    dc_poisson_residual = float(qss_dc_value.poisson_residual)
    dc_gates = {
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
        "Poisson residual": (
            dc_poisson_residual,
            maximum_dc_poisson_residual,
        ),
    }
    dc_failures = [
        f"{name}={value:.6g} > {limit:.6g}"
        for name, (value, limit) in dc_gates.items()
        if not np.isfinite(value) or value > limit
    ]
    if dc_failures:
        raise BulkDefectDeviceACError(
            "DC state is not certified on the AC device operator: "
            + "; ".join(dc_failures)
        )
    occupancy_dc = quasi_steady_bulk_trap_occupancy(
        qss_dc_value.y[: grid.size],
        qss_dc_value.y[grid.size : 2 * grid.size],
        layout,
    )
    reference_n = qss_dc_value.y[: grid.size].copy()
    reference_p = qss_dc_value.y[grid.size : 2 * grid.size].copy()
    reference_logit = occupancy_logit(occupancy_dc, layout)
    dynamic_dc_value = system.evaluate_quasi_fermi_increments_dynamic_bulk(
        dqfn_dc,
        dqfp_dc,
        layout,
        occupancy_dc,
        1.0 if illuminated else 0.0,
        V_app=float(V_dc),
        reference_electron_density_m3=reference_n,
        reference_hole_density_m3=reference_p,
        reference_occupancy=occupancy_dc,
    )
    dynamic_dc = evaluate_dynamic_bulk_traps_about_qss(
        dynamic_dc_value.y[: grid.size],
        dynamic_dc_value.y[grid.size : 2 * grid.size],
        occupancy_dc,
        layout,
        reference_electron_density_m3=reference_n,
        reference_hole_density_m3=reference_p,
        reference_occupancy=occupancy_dc,
    )
    rate_difference = Q * np.sum(
        (
            np.abs(dynamic_dc_value.rate_n - qss_dc_value.rate_n)
            + np.abs(dynamic_dc_value.rate_p - qss_dc_value.rate_p)
        )
        * np.asarray(material.dx_cell)
    )
    current_scale = max(
        float(np.max(np.abs(qss_dc_value.current_n + qss_dc_value.current_p))),
        abs(Q * float(stack.Phi)),
        1.0,
    )
    qss_embedding_error = max(
        float(np.max(np.abs(dynamic_dc_value.phi - qss_dc_value.phi)))
        / material.V_T_device,
        float(rate_difference) / current_scale,
        float(np.max(np.abs(dynamic_dc_value.current_n - qss_dc_value.current_n)))
        / current_scale,
        float(np.max(np.abs(dynamic_dc_value.current_p - qss_dc_value.current_p)))
        / current_scale,
    )

    eps_face = EPS_0 * _harmonic_face_average(material.eps_r)
    polarity = float(material.junction_polarity)

    def dynamic_evaluate(
        coordinate: np.ndarray, voltage: float
    ) -> SmallSignalEvaluation:
        carrier_coordinate = coordinate[: 2 * interior_count]
        occupancy = occupancy_from_logit_increment(
            reference_logit,
            coordinate[2 * interior_count :],
            layout,
        )
        dqfn, dqfp = qf_coordinates(carrier_coordinate, voltage)
        value = system.evaluate_quasi_fermi_increments_dynamic_bulk(
            dqfn,
            dqfp,
            layout,
            occupancy,
            1.0 if illuminated else 0.0,
            V_app=voltage,
            reference_electron_density_m3=reference_n,
            reference_hole_density_m3=reference_p,
            reference_occupancy=occupancy_dc,
        )
        dynamic = evaluate_dynamic_bulk_traps_about_qss(
            value.y[: grid.size],
            value.y[grid.size : 2 * grid.size],
            occupancy,
            layout,
            reference_electron_density_m3=reference_n,
            reference_hole_density_m3=reference_p,
            reference_occupancy=occupancy_dc,
        )
        return _current_evaluation(
            value,
            dynamic,
            grid=grid,
            polarity=polarity,
            eps_face=eps_face,
            trap_storage=True,
        )

    def qss_evaluate(coordinate: np.ndarray, voltage: float) -> SmallSignalEvaluation:
        dqfn, dqfp = qf_coordinates(coordinate, voltage)
        value = system.evaluate_quasi_fermi_increments(
            dqfn,
            dqfp,
            1.0 if illuminated else 0.0,
            V_app=voltage,
        )
        return _current_evaluation(
            value,
            None,
            grid=grid,
            polarity=polarity,
            eps_face=eps_face,
            trap_storage=False,
        )

    def frozen_evaluate(
        coordinate: np.ndarray, voltage: float
    ) -> SmallSignalEvaluation:
        dqfn, dqfp = qf_coordinates(coordinate, voltage)
        value = system.evaluate_quasi_fermi_increments_dynamic_bulk(
            dqfn,
            dqfp,
            layout,
            occupancy_dc,
            1.0 if illuminated else 0.0,
            V_app=voltage,
            reference_electron_density_m3=reference_n,
            reference_hole_density_m3=reference_p,
            reference_occupancy=occupancy_dc,
        )
        return _current_evaluation(
            value,
            None,
            grid=grid,
            polarity=polarity,
            eps_face=eps_face,
            trap_storage=False,
        )

    face_weights = np.diff(grid) / float(grid[-1] - grid[0])
    coordinate = np.zeros(2 * interior_count + layout.size, dtype=float)
    levels: list[FrequencyDomainResult] = []
    for level, factor in enumerate(factors):
        if progress is not None:
            progress(
                "bulk_defect_device_ac_refinement",
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

    widths = np.asarray(material.dx_cell, dtype=float)
    interior_widths = widths[1:-1]
    electron_storage = Q * (
        final.storage_response[:, :interior_count] @ interior_widths
    )
    hole_storage = Q * (
        final.storage_response[:, interior_count : 2 * interior_count] @ interior_widths
    )
    occupied_response = final.storage_response[:, 2 * interior_count :]
    state_widths = widths[layout.device_node_indices]
    occupied_population = occupied_response @ state_widths
    charged = np.asarray(layout.charge_transitions) != NEUTRAL
    trap_charge_storage = -Q * (occupied_response[:, charged] @ state_widths[charged])
    occupancy_response = occupied_response / layout.population_density_m3[np.newaxis, :]

    n_response = np.zeros((frequencies.size, grid.size), dtype=complex)
    p_response = np.zeros_like(n_response)
    n_response[:, 1:-1] = final.storage_response[:, :interior_count]
    p_response[:, 1:-1] = final.storage_response[:, interior_count : 2 * interior_count]
    node = layout.device_node_indices
    n_dc = dynamic_dc_value.y[: grid.size]
    p_dc = dynamic_dc_value.y[grid.size : 2 * grid.size]
    electron_capture = layout.population_density_m3[np.newaxis, :] * (
        layout.capture_n_m3_s[np.newaxis, :]
        * (
            (1.0 - occupancy_dc)[np.newaxis, :] * n_response[:, node]
            - (n_dc[node] + layout.n1_m3)[np.newaxis, :] * occupancy_response
        )
    )
    hole_capture = layout.population_density_m3[np.newaxis, :] * (
        layout.capture_p_m3_s[np.newaxis, :]
        * (
            occupancy_dc[np.newaxis, :] * p_response[:, node]
            + (p_dc[node] + layout.p1_m3)[np.newaxis, :] * occupancy_response
        )
    )
    omega = 2.0 * np.pi * frequencies
    source_count = int(np.max(layout.source_indices)) + 1
    grouped_shape = (frequencies.size, source_count, grid.size)
    grouped_electron_capture = np.zeros(grouped_shape, dtype=complex)
    grouped_hole_capture = np.zeros(grouped_shape, dtype=complex)
    grouped_occupied_response = np.zeros(grouped_shape, dtype=complex)
    for state_index, (source_index, device_node) in enumerate(
        zip(layout.source_indices, layout.device_node_indices, strict=True)
    ):
        grouped_electron_capture[:, source_index, device_node] += electron_capture[
            :, state_index
        ]
        grouped_hole_capture[:, source_index, device_node] += hole_capture[
            :, state_index
        ]
        grouped_occupied_response[:, source_index, device_node] += occupied_response[
            :, state_index
        ]
    trap_balance_residual = (
        grouped_electron_capture
        - grouped_hole_capture
        - 1j * omega[:, np.newaxis, np.newaxis] * grouped_occupied_response
    )
    trap_balance_scale = (
        np.abs(grouped_electron_capture)
        + np.abs(grouped_hole_capture)
        + np.abs(1j * omega[:, np.newaxis, np.newaxis] * grouped_occupied_response)
    )
    trap_balance_relative = np.divide(
        np.abs(trap_balance_residual),
        trap_balance_scale,
        out=np.zeros_like(trap_balance_scale, dtype=float),
        where=trap_balance_scale > 0.0,
    )
    max_trap_balance = float(np.max(trap_balance_relative))
    window = _assess_frequency_window(
        frequencies,
        dynamic_dc.relaxation_rate_s1,
        branch_margin_decades=frequency_branch_margin_decades,
        maximum_sampling_gap_decades=maximum_frequency_sampling_gap_decades,
    )
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
    certificate = BulkDefectDeviceACCertificate(
        dc_operating_point_certified=bool(operating_point.certified),
        dc_maximum_normalized_residual=dc_maximum_normalized_residual,
        dc_electron_continuity_bound_A_m2=dc_electron_continuity_bound,
        dc_hole_continuity_bound_A_m2=dc_hole_continuity_bound,
        dc_face_current_spread_A_m2=dc_face_current_spread,
        dc_poisson_residual=dc_poisson_residual,
        qss_embedding_normalized_error=qss_embedding_error,
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
    result = BulkDefectDeviceACResult(
        frequencies_Hz=_readonly(final.frequencies, dtype=float),
        impedance_ohm_m2=_readonly(final.impedance, dtype=complex),
        admittance_S_m2=_readonly(final.admittance, dtype=complex),
        admittance_faces_S_m2=_readonly(final.admittance_faces, dtype=complex),
        electron_conduction_admittance_faces_S_m2=_readonly(
            component_map["electron"], dtype=complex
        ),
        hole_conduction_admittance_faces_S_m2=_readonly(
            component_map["hole"], dtype=complex
        ),
        displacement_admittance_faces_S_m2=_readonly(
            final.displacement_admittance_faces, dtype=complex
        ),
        electron_storage_response_F_m2=_readonly(electron_storage, dtype=complex),
        hole_storage_response_F_m2=_readonly(hole_storage, dtype=complex),
        trap_charge_storage_response_F_m2=_readonly(trap_charge_storage, dtype=complex),
        trap_occupied_population_response_m2_V=_readonly(
            occupied_population, dtype=complex
        ),
        trap_occupancy_response_per_V=_readonly(occupancy_response, dtype=complex),
        electron_capture_response_m3_s_V=_readonly(electron_capture, dtype=complex),
        hole_capture_response_m3_s_V=_readonly(hole_capture, dtype=complex),
        qss_reference_admittance_S_m2=complex(qss_reference.admittance[0]),
        frozen_reference_admittance_S_m2=complex(frozen_reference.admittance[-1]),
        refinement_factors=factors,
        refinement_relative_changes=refinement_changes,
        layout=layout,
        dc_state=operating_point,
        certificate=certificate,
    )
    if require_certificate and not certificate.certified:
        raise BulkDefectDeviceACCertificationError(
            "bulk defect device AC certificate failed: " + ", ".join(reasons),
            result,
        )
    return result


__all__ = [
    "BULK_DEFECT_DEVICE_AC_SCOPE",
    "BULK_DEFECT_DEVICE_AC_VERSION",
    "BulkDefectDeviceACCertificate",
    "BulkDefectDeviceACCertificationError",
    "BulkDefectDeviceACError",
    "BulkDefectDeviceACResult",
    "BulkDefectFrequencyWindow",
    "DEFAULT_REFINEMENT_FACTORS",
    "run_bulk_defect_device_impedance",
]
