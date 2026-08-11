"""Certified ETL/absorber conduction-band-offset scans.

The physical interface response is stiff with respect to a large band-step
change. This driver therefore continues outward from a declared reference CBO,
inserts temporary bridge points when a requested step leaves Newton's basin,
and retains only states that pass the quasi-Fermi and local-interface
certificates.

The CBO convention matches ``device_parameter_sweep``:

``delta_Ec = chi_absorber - chi_ETL``.

Positive values are electron-extraction spikes at an absorber/ETL boundary.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
from typing import Callable

import numpy as np

from perovskite_sim.experiments.jv_sweep import (
    JVMetrics,
    build_electrical_grid,
    compute_metrics,
    thermodynamic_voc_ceiling,
)
from perovskite_sim.experiments.quasi_fermi_steady_state import (
    QuasiFermiJVSweepResult,
    QuasiFermiSteadyStateError,
    QuasiFermiSteadyStateResult,
    build_two_sided_trace_grid,
    solve_quasi_fermi_jv_sweep,
    solve_quasi_fermi_steady_state,
)
from perovskite_sim.models.device import DeviceStack, electrical_layers
from perovskite_sim.physics.interface_plane import (
    FERMI_DIRAC_RICHARDSON,
    FERMI_RICHARDSON,
    validate_interface_transport_model,
)
from perovskite_sim.physics.fermi_dirac import inverse_fermi_dirac_half
from perovskite_sim.physics.two_sided_interface import (
    DEDUPLICATED_QSS,
    TWO_SIDED_TRACE,
    validate_interface_topology,
)
from perovskite_sim.sweeps.device_parameter_sweep import (
    SweepPoint,
    apply_sweep_point,
)


class InterfaceCBOScanError(RuntimeError):
    """A CBO scan could not preserve a certified continuation path."""


FIXED_CONTACTS = "fixed_contacts"
RECOMPUTED_BUILT_IN = "recomputed_built_in"
CBO_BOUNDARY_POLICIES = (FIXED_CONTACTS, RECOMPUTED_BUILT_IN)


def validate_cbo_boundary_policy(policy: str) -> str:
    """Return a normalized contact/built-in-potential policy."""
    normalized = str(policy).strip().lower()
    if normalized not in CBO_BOUNDARY_POLICIES:
        raise ValueError(
            f"boundary_policy must be one of {CBO_BOUNDARY_POLICIES}; "
            f"got {policy!r}"
        )
    return normalized


@dataclass(frozen=True)
class CBOShortCircuitSample:
    """One certified point on the parameter-continuation trace."""

    delta_ec_eV: float
    current_A_m2: float
    requested: bool
    face_current_spread_A_m2: float
    interface_local_residual: float
    interface_max_state_to_dos: float
    applied_V_bi_V: float


@dataclass(frozen=True)
class CBOCriticalInterval:
    """Sample-bounded onset of a relative metric loss."""

    metric: str
    reference_delta_ec_eV: float
    reference_value: float | None
    relative_drop_fraction: float
    threshold_value: float | None
    lower_delta_ec_eV: float | None
    upper_delta_ec_eV: float | None
    resolved: bool


@dataclass(frozen=True)
class CBOJVMetricsGridSample:
    """J-V metrics extracted from one nested voltage sampling grid."""

    voltage_point_count: int
    voltage_interval_count: int
    metrics: JVMetrics
    certified: bool
    retained_voltage_point_count: int | None = None


@dataclass(frozen=True)
class CBOScanPoint:
    """A requested CBO point with a certified short-circuit state and optional J-V."""

    delta_ec_eV: float
    short_circuit_state: QuasiFermiSteadyStateResult
    jv: QuasiFermiJVSweepResult | None
    voltage_grid_metrics: tuple[CBOJVMetricsGridSample, ...] = ()

    @property
    def metrics(self) -> JVMetrics | None:
        return None if self.jv is None else self.jv.metrics

    @property
    def certified(self) -> bool:
        return bool(
            self.short_circuit_state.certified
            and (
                self.jv is None
                or (
                    self.jv.certified
                    and self.jv.metrics_certified
                    and self.voltage_grid_metrics
                    and all(
                        sample.certified
                        for sample in self.voltage_grid_metrics
                    )
                )
            )
        )


@dataclass(frozen=True)
class CBOScanTermination:
    """A bracket where the declared bulk-statistics envelope ended."""

    direction: str
    last_certified_delta_ec_eV: float
    first_failed_delta_ec_eV: float
    requested_delta_ec_eV: float
    reason: str


@dataclass(frozen=True)
class InterfaceCBOScanResult:
    """Certified CBO results, onset brackets, and any validity endpoints."""

    requested_delta_ec_eV: np.ndarray
    points: tuple[CBOScanPoint, ...]
    short_circuit_trace: tuple[CBOShortCircuitSample, ...]
    critical_intervals: tuple[CBOCriticalInterval, ...]
    terminations: tuple[CBOScanTermination, ...]
    reference_delta_ec_eV: float
    interface_transmission: float
    relative_drop_fraction: float
    minimum_delta_step_eV: float
    maximum_delta_step_eV: float
    minimum_voltage_step_V: float | None
    N_grid: int
    grid_node_count: int
    grid_interval_count: int
    grid_interval_weights: tuple[float, ...]
    grid_alphas: tuple[float, ...]
    reference_grid_warm_starts: int
    reference_grid_warm_start_failures: int
    reference_grid_cold_recoveries: int
    reference_grid_predictor_recoveries: int
    voltages_V: np.ndarray
    calculate_jv_metrics: bool
    boundary_policy: str
    interface_transport_model: str
    interface_topology: str
    heterojunction_recombination_despike: float
    qf_coordinate_system: str
    voltage_grids_V: tuple[np.ndarray, ...]
    mpp_interpolation: str

    @property
    def sync_vbi(self) -> bool:
        """Compatibility view of the explicit boundary policy."""
        return self.boundary_policy == RECOMPUTED_BUILT_IN

    @property
    def complete(self) -> bool:
        return bool(
            not self.terminations
            and len(self.points) == len(self.requested_delta_ec_eV)
        )

    @property
    def certified(self) -> bool:
        return bool(
            self.points
            and all(point.certified for point in self.points)
            and all(
                np.isfinite(sample.current_A_m2)
                and sample.face_current_spread_A_m2 <= 1.0e-4
                and sample.interface_local_residual <= 1.0e-7
                for sample in self.short_circuit_trace
            )
        )


@dataclass(frozen=True)
class CBOGridConvergenceCertificate:
    """Critical-CBO agreement across independently certified grids."""

    metric: str
    grid_interval_counts: tuple[int, ...]
    grid_interval_weights: tuple[float, ...]
    grid_alphas: tuple[float, ...]
    critical_intervals_eV: tuple[tuple[float, float], ...]
    envelope_lower_eV: float | None
    envelope_upper_eV: float | None
    envelope_width_eV: float | None
    maximum_envelope_width_eV: float
    critical_midpoints_eV: tuple[float, ...]
    successive_midpoint_shifts_eV: tuple[float, ...]
    successive_shift_ratios: tuple[float, ...]
    maximum_successive_shift_ratio: float
    reference_values: tuple[float, ...]
    reference_relative_spread: float | None
    maximum_reference_relative_spread: float
    certified: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class CBOVoltageGridPointCertificate:
    """Voltage-sampling convergence of J-V metrics at one CBO."""

    delta_ec_eV: float
    voltage_point_counts: tuple[int, ...]
    retained_voltage_point_counts: tuple[int, ...]
    voc_values_V: tuple[float, ...]
    ff_values: tuple[float, ...]
    pce_values: tuple[float, ...]
    successive_voc_changes_V: tuple[float, ...]
    successive_ff_changes: tuple[float, ...]
    successive_pce_changes: tuple[float, ...]
    successive_voc_change_ratios: tuple[float, ...]
    successive_ff_change_ratios: tuple[float, ...]
    successive_pce_change_ratios: tuple[float, ...]
    final_voc_change_V: float | None
    final_ff_change: float | None
    final_pce_change: float | None
    continuation_bridge_count: int
    minimum_voltage_step_V: float | None
    nodal_predictor_fallback_attempts: int
    nodal_predictor_fallback_failures: int
    certified: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class CBOVoltageGridConvergenceCertificate:
    """Nested voltage-grid convergence across every requested CBO point."""

    sampling_method: str
    mpp_interpolation: str
    voltage_point_counts: tuple[int, ...]
    voltage_interval_counts: tuple[int, ...]
    minimum_voltage_grids: int
    maximum_voc_change_V: float
    maximum_ff_change: float
    maximum_pce_change: float
    maximum_successive_change_ratio: float
    contraction_noise_floor_fraction: float
    points: tuple[CBOVoltageGridPointCertificate, ...]
    certified: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class CBOStatisticsValidityCertificate:
    """Validity of the interface occupation law over one CBO scan."""

    interface_transport_model: str
    maximum_state_to_dos: float | None
    allowed_state_to_dos: float | None
    maximum_reduced_fermi_level: float | None
    certified: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class CBOExternalMetricCertificate:
    """One normalized CBO response checked against an external reference."""

    metric: str
    matched_delta_ec_eV: tuple[float, ...]
    simulated_normalized: tuple[float, ...]
    reference_normalized: tuple[float, ...]
    rms_normalized_error: float | None
    max_normalized_error: float | None
    maximum_normalized_error: float
    simulated_critical_interval_eV: tuple[float, float] | None
    reference_critical_interval_eV: tuple[float, float] | None
    reference_critical_interval_width_eV: float | None
    maximum_reference_critical_interval_width_eV: float
    critical_interval_distance_eV: float | None
    maximum_critical_interval_distance_eV: float
    certified: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class CBOExternalReferenceAudit:
    """Provenance and protocol checks for an independent SCAPS export."""

    schema: str | None
    schema_version: str | None
    solver: str | None
    solver_version: str | None
    delta_ec_convention: str | None
    swept_parameter: str | None
    boundary_policy: str | None
    reference_delta_ec_eV: float | None
    temperature_K: float | None
    illumination: str | None
    independently_generated: bool | None
    interpolated: bool | None
    source_export_sha256: str | None
    source_deck_sha256: str | None
    parameter_manifest_sha256: str | None
    point_count: int
    unique_delta_count: int
    certified: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class CBOExternalValidation:
    """Content-addressed SCAPS comparison with explicit pass criteria."""

    reference_path: str
    reference_sha256: str
    source_xlsx: str | None
    source_pdf: str | None
    extracted_at: str | None
    reference_audit: CBOExternalReferenceAudit
    certificates: tuple[CBOExternalMetricCertificate, ...]

    @property
    def certified(self) -> bool:
        return bool(
            self.reference_audit.certified
            and self.certificates
            and all(certificate.certified for certificate in self.certificates)
        )


def _metric_interval(
    result: InterfaceCBOScanResult,
    metric: str,
) -> CBOCriticalInterval | None:
    return next(
        (
            interval
            for interval in result.critical_intervals
            if interval.metric == metric
        ),
        None,
    )


def certify_cbo_grid_convergence(
    results: tuple[InterfaceCBOScanResult, ...] | list[InterfaceCBOScanResult],
    *,
    metric: str = "Jsc",
    minimum_grids: int = 3,
    maximum_envelope_width_eV: float = 1.0e-2,
    maximum_reference_relative_spread: float = 1.0e-2,
    maximum_successive_shift_ratio: float = 0.9,
) -> CBOGridConvergenceCertificate:
    """Certify a critical CBO only when the full grid ladder agrees.

    The critical interval itself already contains parameter-sampling error.
    The conservative cross-grid result is therefore the union envelope of all
    certified intervals, not a midpoint extrapolation.
    """
    scans = tuple(results)
    if minimum_grids < 2:
        raise ValueError("minimum_grids must be at least two")
    for name, value in (
        ("maximum_envelope_width_eV", maximum_envelope_width_eV),
        (
            "maximum_reference_relative_spread",
            maximum_reference_relative_spread,
        ),
        ("maximum_successive_shift_ratio", maximum_successive_shift_ratio),
    ):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if maximum_successive_shift_ratio >= 1.0:
        raise ValueError("maximum_successive_shift_ratio must be smaller than 1")

    ordered = tuple(sorted(scans, key=lambda scan: scan.grid_interval_count))
    reasons: list[str] = []
    if len(ordered) < minimum_grids:
        reasons.append(
            f"requires at least {minimum_grids} grids; received {len(ordered)}"
        )
    counts = tuple(scan.grid_interval_count for scan in ordered)
    if len(set(counts)) != len(counts):
        reasons.append("actual grid interval counts must be unique")
    if ordered:
        reference = ordered[0]
        for scan in ordered:
            if not scan.complete:
                reasons.append(
                    f"grid {scan.grid_interval_count} did not complete the "
                    "requested CBO axis"
                )
            if not scan.certified:
                reasons.append(
                    f"grid {scan.grid_interval_count} lacks a numerical "
                    "certificate"
                )
            if (
                scan.boundary_policy != reference.boundary_policy
                or scan.interface_transport_model
                != reference.interface_transport_model
                or scan.interface_topology != reference.interface_topology
                or not math.isclose(
                    scan.heterojunction_recombination_despike,
                    reference.heterojunction_recombination_despike,
                    rel_tol=0.0,
                    abs_tol=0.0,
                )
                or not math.isclose(
                    scan.interface_transmission,
                    reference.interface_transmission,
                    rel_tol=0.0,
                    abs_tol=0.0,
                )
                or not math.isclose(
                    scan.relative_drop_fraction,
                    reference.relative_drop_fraction,
                    rel_tol=0.0,
                    abs_tol=0.0,
                )
                or not math.isclose(
                    scan.reference_delta_ec_eV,
                    reference.reference_delta_ec_eV,
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                )
                or scan.grid_interval_weights
                != reference.grid_interval_weights
                or scan.grid_alphas != reference.grid_alphas
                or scan.qf_coordinate_system
                != reference.qf_coordinate_system
            ):
                reasons.append("grid scans do not share one physical protocol")
                break

    critical: list[tuple[float, float]] = []
    reference_values: list[float] = []
    for scan in ordered:
        interval = _metric_interval(scan, metric)
        if (
            interval is None
            or not interval.resolved
            or interval.lower_delta_ec_eV is None
            or interval.upper_delta_ec_eV is None
            or interval.reference_value is None
        ):
            reasons.append(
                f"grid {scan.grid_interval_count} has no resolved {metric} "
                "critical interval"
            )
            continue
        critical.append(
            (interval.lower_delta_ec_eV, interval.upper_delta_ec_eV)
        )
        reference_values.append(interval.reference_value)

    envelope_lower = min((item[0] for item in critical), default=None)
    envelope_upper = max((item[1] for item in critical), default=None)
    envelope_width = (
        None
        if envelope_lower is None or envelope_upper is None
        else envelope_upper - envelope_lower
    )
    if (
        envelope_width is not None
        and envelope_width > maximum_envelope_width_eV
    ):
        reasons.append(
            f"critical-CBO envelope {envelope_width:.6g} eV exceeds "
            f"{maximum_envelope_width_eV:.6g} eV"
        )

    midpoints = tuple(0.5 * (lower + upper) for lower, upper in critical)
    shifts = tuple(
        abs(right - left)
        for left, right in zip(midpoints[:-1], midpoints[1:])
    )
    shift_ratios: list[float] = []
    for previous, current in zip(shifts[:-1], shifts[1:]):
        if previous == 0.0:
            ratio = 0.0 if current == 0.0 else math.inf
        else:
            ratio = current / previous
        shift_ratios.append(ratio)
    if shift_ratios and any(
        ratio > maximum_successive_shift_ratio for ratio in shift_ratios
    ):
        reasons.append(
            "successive critical-CBO shifts do not contract below ratio "
            f"{maximum_successive_shift_ratio:.6g}; observed "
            + ", ".join(f"{ratio:.6g}" for ratio in shift_ratios)
        )

    relative_spread: float | None = None
    if reference_values:
        denominator = abs(reference_values[-1])
        relative_spread = (
            math.inf
            if denominator == 0.0
            else (max(reference_values) - min(reference_values)) / denominator
        )
        if relative_spread > maximum_reference_relative_spread:
            reasons.append(
                f"reference {metric} grid spread {relative_spread:.6g} "
                f"exceeds {maximum_reference_relative_spread:.6g}"
            )

    complete_ladder = (
        len(ordered) >= minimum_grids
        and len(critical) == len(ordered)
        and len(reference_values) == len(ordered)
    )
    return CBOGridConvergenceCertificate(
        metric=metric,
        grid_interval_counts=counts,
        grid_interval_weights=(
            ordered[0].grid_interval_weights if ordered else ()
        ),
        grid_alphas=ordered[0].grid_alphas if ordered else (),
        critical_intervals_eV=tuple(critical),
        envelope_lower_eV=envelope_lower,
        envelope_upper_eV=envelope_upper,
        envelope_width_eV=envelope_width,
        maximum_envelope_width_eV=float(maximum_envelope_width_eV),
        critical_midpoints_eV=midpoints,
        successive_midpoint_shifts_eV=shifts,
        successive_shift_ratios=tuple(shift_ratios),
        maximum_successive_shift_ratio=float(maximum_successive_shift_ratio),
        reference_values=tuple(reference_values),
        reference_relative_spread=relative_spread,
        maximum_reference_relative_spread=float(
            maximum_reference_relative_spread
        ),
        certified=bool(complete_ladder and not reasons),
        reasons=tuple(dict.fromkeys(reasons)),
    )


def _nested_voltage_grids(
    grids: tuple[np.ndarray, ...] | list[np.ndarray],
) -> tuple[np.ndarray, ...]:
    """Validate and copy a coarse-to-fine nested voltage-grid ladder."""
    normalized = tuple(np.asarray(grid, dtype=float) for grid in grids)
    if not normalized:
        raise ValueError("voltage_grids_V must contain at least one grid")
    for grid in normalized:
        if (
            grid.ndim != 1
            or grid.size < 2
            or not np.all(np.isfinite(grid))
            or np.any(np.diff(grid) <= 0.0)
        ):
            raise ValueError(
                "each voltage grid must be finite and strictly increasing"
            )
        if grid[0] != 0.0:
            raise ValueError("each voltage grid must start at 0 V")

    point_counts = tuple(len(grid) for grid in normalized)
    if any(right <= left for left, right in zip(point_counts[:-1], point_counts[1:])):
        raise ValueError(
            "voltage grids must be ordered from coarse to fine with unique "
            "point counts"
        )
    finest_maximum = normalized[-1][-1]
    if any(
        not math.isclose(
            float(grid[-1]),
            float(finest_maximum),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        for grid in normalized[:-1]
    ):
        raise ValueError("nested voltage grids must share one maximum voltage")

    for coarse, fine in zip(normalized[:-1], normalized[1:]):
        for voltage in coarse:
            matches = np.flatnonzero(
                np.isclose(fine, voltage, rtol=0.0, atol=1.0e-12)
            )
            if matches.size != 1:
                raise ValueError(
                    "each voltage grid must be a strict subset of the next grid"
                )
    return tuple(grid.copy() for grid in normalized)


def _successive_absolute_changes(
    values: tuple[float, ...],
) -> tuple[float, ...]:
    return tuple(abs(right - left) for left, right in zip(values[:-1], values[1:]))


def _successive_change_ratios(
    changes: tuple[float, ...],
) -> tuple[float, ...]:
    ratios: list[float] = []
    for previous, current in zip(changes[:-1], changes[1:]):
        if previous == 0.0:
            ratio = 0.0 if current == 0.0 else math.inf
        else:
            ratio = current / previous
        ratios.append(ratio)
    return tuple(ratios)


def certify_cbo_voltage_grid_convergence(
    result: InterfaceCBOScanResult,
    *,
    minimum_voltage_grids: int = 3,
    maximum_voc_change_V: float = 2.0e-3,
    maximum_ff_change: float = 1.0e-3,
    maximum_pce_change: float = 5.0e-4,
    maximum_successive_change_ratio: float = 0.8,
    contraction_noise_floor_fraction: float = 0.1,
) -> CBOVoltageGridConvergenceCertificate:
    """Certify nested voltage sampling for Voc, FF, and PCE at every CBO.

    The finest J-V branch is solved once over the complete voltage window.
    Coarser metrics are then extracted from strict nested subsets of that same
    certified branch. This isolates voltage-sampling error without repeating
    the nonlinear solve at voltage points shared by all grids.
    """
    if minimum_voltage_grids < 3:
        raise ValueError("minimum_voltage_grids must be at least three")
    for name, value in (
        ("maximum_voc_change_V", maximum_voc_change_V),
        ("maximum_ff_change", maximum_ff_change),
        ("maximum_pce_change", maximum_pce_change),
        ("maximum_successive_change_ratio", maximum_successive_change_ratio),
        (
            "contraction_noise_floor_fraction",
            contraction_noise_floor_fraction,
        ),
    ):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if maximum_successive_change_ratio >= 1.0:
        raise ValueError(
            "maximum_successive_change_ratio must be smaller than 1"
        )
    if contraction_noise_floor_fraction >= 1.0:
        raise ValueError(
            "contraction_noise_floor_fraction must be smaller than 1"
        )

    reasons: list[str] = []
    try:
        grids = _nested_voltage_grids(result.voltage_grids_V)
    except ValueError as exc:
        grids = ()
        reasons.append(str(exc))
    point_counts = tuple(len(grid) for grid in grids)
    interval_counts = tuple(count - 1 for count in point_counts)
    if len(grids) < minimum_voltage_grids:
        reasons.append(
            f"requires at least {minimum_voltage_grids} voltage grids; "
            f"received {len(grids)}"
        )
    if not result.calculate_jv_metrics:
        reasons.append("scan did not calculate full J-V metrics")
    if not result.complete:
        reasons.append("scan did not complete the requested CBO axis")
    if not result.certified:
        reasons.append("scan lacks a numerical point certificate")

    point_certificates: list[CBOVoltageGridPointCertificate] = []
    for point in result.points:
        point_reasons: list[str] = []
        samples = point.voltage_grid_metrics
        sample_counts = tuple(sample.voltage_point_count for sample in samples)
        retained_sample_counts = tuple(
            (
                sample.voltage_point_count
                if sample.retained_voltage_point_count is None
                else sample.retained_voltage_point_count
            )
            for sample in samples
        )
        if sample_counts != point_counts:
            point_reasons.append(
                "metric samples do not match the declared voltage-grid ladder"
            )
        if not samples or not all(sample.certified for sample in samples):
            point_reasons.append(
                "one or more voltage-grid metric samples are uncertified"
            )

        voc_values = tuple(float(sample.metrics.V_oc) for sample in samples)
        ff_values = tuple(float(sample.metrics.FF) for sample in samples)
        pce_values = tuple(float(sample.metrics.PCE) for sample in samples)
        if not all(
            np.isfinite(value)
            for values in (voc_values, ff_values, pce_values)
            for value in values
        ):
            point_reasons.append("one or more J-V metrics are not finite")

        voc_changes = _successive_absolute_changes(voc_values)
        ff_changes = _successive_absolute_changes(ff_values)
        pce_changes = _successive_absolute_changes(pce_values)
        voc_ratios = _successive_change_ratios(voc_changes)
        ff_ratios = _successive_change_ratios(ff_changes)
        pce_ratios = _successive_change_ratios(pce_changes)
        final_voc = voc_changes[-1] if voc_changes else None
        final_ff = ff_changes[-1] if ff_changes else None
        final_pce = pce_changes[-1] if pce_changes else None

        for metric, final, limit in (
            ("Voc", final_voc, maximum_voc_change_V),
            ("FF", final_ff, maximum_ff_change),
            ("PCE", final_pce, maximum_pce_change),
        ):
            if final is None:
                point_reasons.append(f"{metric} has no refinement difference")
            elif final > limit:
                point_reasons.append(
                    f"final {metric} change {final:.6g} exceeds {limit:.6g}"
                )
        for metric, ratios, changes, limit in (
            ("Voc", voc_ratios, voc_changes, maximum_voc_change_V),
            ("FF", ff_ratios, ff_changes, maximum_ff_change),
            ("PCE", pce_ratios, pce_changes, maximum_pce_change),
        ):
            material_ratios = tuple(
                ratio
                for ratio, current_change in zip(ratios, changes[1:])
                if current_change
                > contraction_noise_floor_fraction * limit
            )
            if any(
                ratio > maximum_successive_change_ratio
                for ratio in material_ratios
            ):
                point_reasons.append(
                    f"successive {metric} changes do not contract below ratio "
                    f"{maximum_successive_change_ratio:.6g}; observed "
                    + ", ".join(
                        f"{ratio:.6g}" for ratio in material_ratios
                    )
                )

        point_certificate = CBOVoltageGridPointCertificate(
            delta_ec_eV=float(point.delta_ec_eV),
            voltage_point_counts=sample_counts,
            retained_voltage_point_counts=retained_sample_counts,
            voc_values_V=voc_values,
            ff_values=ff_values,
            pce_values=pce_values,
            successive_voc_changes_V=voc_changes,
            successive_ff_changes=ff_changes,
            successive_pce_changes=pce_changes,
            successive_voc_change_ratios=voc_ratios,
            successive_ff_change_ratios=ff_ratios,
            successive_pce_change_ratios=pce_ratios,
            final_voc_change_V=final_voc,
            final_ff_change=final_ff,
            final_pce_change=final_pce,
            continuation_bridge_count=int(
                getattr(point.jv, "continuation_bridge_count", 0)
            ),
            minimum_voltage_step_V=getattr(
                point.jv,
                "minimum_voltage_step_V",
                None,
            ),
            nodal_predictor_fallback_attempts=int(
                getattr(point.jv, "nodal_predictor_fallback_attempts", 0)
            ),
            nodal_predictor_fallback_failures=int(
                getattr(point.jv, "nodal_predictor_fallback_failures", 0)
            ),
            certified=not point_reasons,
            reasons=tuple(dict.fromkeys(point_reasons)),
        )
        point_certificates.append(point_certificate)
        if not point_certificate.certified:
            reasons.append(
                f"CBO {point.delta_ec_eV:+.6g} eV failed voltage-grid "
                "convergence"
            )

    complete_points = bool(
        result.points
        and len(point_certificates) == len(result.points)
        and all(point.certified for point in point_certificates)
    )
    return CBOVoltageGridConvergenceCertificate(
        sampling_method="nested_subsampling_of_finest_certified_jv",
        mpp_interpolation=result.mpp_interpolation,
        voltage_point_counts=point_counts,
        voltage_interval_counts=interval_counts,
        minimum_voltage_grids=int(minimum_voltage_grids),
        maximum_voc_change_V=float(maximum_voc_change_V),
        maximum_ff_change=float(maximum_ff_change),
        maximum_pce_change=float(maximum_pce_change),
        maximum_successive_change_ratio=float(
            maximum_successive_change_ratio
        ),
        contraction_noise_floor_fraction=float(
            contraction_noise_floor_fraction
        ),
        points=tuple(point_certificates),
        certified=bool(complete_points and not reasons),
        reasons=tuple(dict.fromkeys(reasons)),
    )


def certify_cbo_statistics_validity(
    result: InterfaceCBOScanResult,
    *,
    maximum_boltzmann_state_to_dos: float = 0.1,
) -> CBOStatisticsValidityCertificate:
    """Reject Boltzmann interface transport outside its dilute-state regime."""
    if (
        not np.isfinite(maximum_boltzmann_state_to_dos)
        or maximum_boltzmann_state_to_dos <= 0.0
        or maximum_boltzmann_state_to_dos >= 1.0
    ):
        raise ValueError(
            "maximum_boltzmann_state_to_dos must lie strictly between 0 and 1"
        )
    values = tuple(
        float(sample.interface_max_state_to_dos)
        for sample in result.short_circuit_trace
    )
    reasons: list[str] = []
    maximum = max(values) if values else None
    if maximum is None:
        reasons.append("scan contains no interface-state occupation samples")
    elif not np.isfinite(maximum) or maximum < 0.0:
        reasons.append("interface state-to-DOS ratio is not finite and nonnegative")

    is_bounded_fermi = result.interface_transport_model == FERMI_RICHARDSON
    is_fermi_dirac = (
        result.interface_transport_model == FERMI_DIRAC_RICHARDSON
    )
    allowed = (
        None
        if is_fermi_dirac
        else (
            1.0
            if is_bounded_fermi
            else float(maximum_boltzmann_state_to_dos)
        )
    )
    maximum_eta = (
        inverse_fermi_dirac_half(maximum)
        if is_fermi_dirac
        and maximum is not None
        and np.isfinite(maximum)
        and maximum > 0.0
        else None
    )
    if (
        allowed is not None
        and maximum is not None
        and np.isfinite(maximum)
        and maximum > allowed + 1.0e-12
    ):
        regime = "bounded Fermi" if is_bounded_fermi else "dilute Boltzmann"
        reasons.append(
            f"maximum interface state/DOS {maximum:.6g} exceeds the "
            f"{regime} limit {allowed:.6g}"
        )
    return CBOStatisticsValidityCertificate(
        interface_transport_model=result.interface_transport_model,
        maximum_state_to_dos=maximum,
        allowed_state_to_dos=allowed,
        maximum_reduced_fermi_level=maximum_eta,
        certified=not reasons,
        reasons=tuple(reasons),
    )

@dataclass(frozen=True)
class _ValidityLimit(Exception):
    last_delta_ec_eV: float
    failed_delta_ec_eV: float
    cause: Exception


def _stack_at_cbo(
    baseline: DeviceStack,
    delta_ec_eV: float,
    *,
    boundary_policy: str,
) -> DeviceStack:
    point = SweepPoint(
        point_id=f"interface_cbo_{delta_ec_eV:+.12g}",
        axis="etl_delta_ec",
        label=f"{delta_ec_eV:+.12g} eV",
        updates={"etl_delta_ec_eV": float(delta_ec_eV)},
    )
    policy = validate_cbo_boundary_policy(boundary_policy)
    return apply_sweep_point(
        baseline,
        point,
        sync_vbi=(policy == RECOMPUTED_BUILT_IN),
    )


def _error_chain(exc: BaseException) -> str:
    messages: list[str] = []
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        text = str(current).strip()
        if text and text not in messages:
            messages.append(text)
        current = current.__cause__ or current.__context__
    return ": ".join(messages)


def _is_bulk_statistics_limit(exc: BaseException) -> bool:
    message = _error_chain(exc).lower()
    return bool(
        "log-density is outside the audited exponential range" in message
        or "outside the audited exponential range" in message
    )


def _critical_interval(
    samples: list[tuple[float, float]],
    *,
    metric: str,
    reference_delta_ec_eV: float,
    relative_drop_fraction: float,
) -> CBOCriticalInterval:
    unique = {float(delta): float(value) for delta, value in samples}
    ordered = sorted(unique.items())
    reference_value = next(
        (
            value
            for delta, value in ordered
            if np.isclose(
                delta,
                reference_delta_ec_eV,
                rtol=0.0,
                atol=1.0e-12,
            )
        ),
        None,
    )
    if reference_value is None or not np.isfinite(reference_value):
        return CBOCriticalInterval(
            metric=metric,
            reference_delta_ec_eV=reference_delta_ec_eV,
            reference_value=None,
            relative_drop_fraction=relative_drop_fraction,
            threshold_value=None,
            lower_delta_ec_eV=None,
            upper_delta_ec_eV=None,
            resolved=False,
        )

    threshold = reference_value * (1.0 - relative_drop_fraction)
    positive_branch = [
        (delta, value)
        for delta, value in ordered
        if delta >= reference_delta_ec_eV - 1.0e-12
    ]
    previous_delta: float | None = None
    for delta, value in positive_branch:
        if value <= threshold and previous_delta is not None:
            return CBOCriticalInterval(
                metric=metric,
                reference_delta_ec_eV=reference_delta_ec_eV,
                reference_value=reference_value,
                relative_drop_fraction=relative_drop_fraction,
                threshold_value=threshold,
                lower_delta_ec_eV=previous_delta,
                upper_delta_ec_eV=delta,
                resolved=True,
            )
        if value > threshold:
            previous_delta = delta

    return CBOCriticalInterval(
        metric=metric,
        reference_delta_ec_eV=reference_delta_ec_eV,
        reference_value=reference_value,
        relative_drop_fraction=relative_drop_fraction,
        threshold_value=threshold,
        lower_delta_ec_eV=previous_delta,
        upper_delta_ec_eV=None,
        resolved=False,
    )


def _simulation_metric_samples(
    result: InterfaceCBOScanResult,
    metric: str,
) -> list[tuple[float, float]]:
    if metric == "Jsc":
        return [
            (
                point.delta_ec_eV,
                (
                    point.short_circuit_state.current_A_m2
                    if point.metrics is None
                    else point.metrics.J_sc
                ),
            )
            for point in result.points
        ]
    attribute = {"FF": "FF", "PCE": "PCE", "Voc": "V_oc"}.get(metric)
    if attribute is None:
        raise ValueError("metric must be one of Jsc, FF, PCE, or Voc")
    return [
        (point.delta_ec_eV, float(getattr(point.metrics, attribute)))
        for point in result.points
        if point.metrics is not None
    ]


def _reference_metric_samples(
    points: list[dict],
    metric: str,
) -> list[tuple[float, float]]:
    key_and_scale = {
        "Jsc": ("Jsc_mA_cm2", 10.0),
        "FF": ("FF_percent", 1.0e-2),
        "PCE": ("PCE_percent", 1.0e-2),
        "Voc": ("Voc_V", 1.0),
    }
    if metric not in key_and_scale:
        raise ValueError("metric must be one of Jsc, FF, PCE, or Voc")
    key, scale = key_and_scale[metric]
    return [(float(point["x"]), float(point[key]) * scale) for point in points]


def _interval_pair(
    interval: CBOCriticalInterval | None,
) -> tuple[float, float] | None:
    if (
        interval is None
        or not interval.resolved
        or interval.lower_delta_ec_eV is None
        or interval.upper_delta_ec_eV is None
    ):
        return None
    return interval.lower_delta_ec_eV, interval.upper_delta_ec_eV


def _interval_distance(
    left: tuple[float, float] | None,
    right: tuple[float, float] | None,
) -> float | None:
    if left is None or right is None:
        return None
    if left[1] < right[0]:
        return right[0] - left[1]
    if right[1] < left[0]:
        return left[0] - right[1]
    return 0.0


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(character in "0123456789abcdefABCDEF" for character in value)


def _audit_scaps_cbo_reference(
    payload: dict,
    reference_points: list[dict],
    result: InterfaceCBOScanResult,
) -> CBOExternalReferenceAudit:
    """Fail closed unless the reference identifies one reproducible protocol."""
    protocol = payload.get("cbo_validation")
    reasons: list[str] = []
    if not isinstance(protocol, dict):
        protocol = {}
        reasons.append("reference lacks cbo_validation protocol metadata")

    schema = payload.get("schema")
    schema_version = payload.get("schema_version")
    solver = protocol.get("solver")
    solver_version = protocol.get("solver_version")
    convention = protocol.get("delta_ec_convention")
    swept_parameter = protocol.get("swept_parameter")
    boundary_policy = protocol.get("boundary_policy")
    illumination = protocol.get("illumination")
    independently_generated = protocol.get("independently_generated")
    interpolated = protocol.get("interpolated")
    source_export_sha256 = protocol.get("source_export_sha256")
    source_deck_sha256 = protocol.get("source_deck_sha256")
    parameter_manifest_sha256 = protocol.get("parameter_manifest_sha256")

    if schema != "solarlab.scaps_cbo_reference":
        reasons.append("reference schema must be solarlab.scaps_cbo_reference")
    if schema_version != "1.0":
        reasons.append("reference schema_version must be 1.0")
    if solver != "SCAPS-1D":
        reasons.append("reference solver must be SCAPS-1D")
    if not isinstance(solver_version, str) or not solver_version.strip():
        reasons.append("reference solver_version is missing")
    if convention != "chi_absorber - chi_etl":
        reasons.append(
            "reference delta_ec_convention must be chi_absorber - chi_etl"
        )
    if swept_parameter != "etl_electron_affinity":
        reasons.append(
            "reference swept_parameter must be etl_electron_affinity"
        )
    if boundary_policy != result.boundary_policy:
        reasons.append("reference and SolarLab boundary policies differ")
    if independently_generated is not True:
        reasons.append("reference must declare independently_generated=true")
    if interpolated is not False:
        reasons.append("reference must declare interpolated=false")
    if not isinstance(illumination, str) or not illumination.strip():
        reasons.append("reference illumination is missing")
    for name, value in (
        ("source_export_sha256", source_export_sha256),
        ("source_deck_sha256", source_deck_sha256),
        ("parameter_manifest_sha256", parameter_manifest_sha256),
    ):
        if not _is_sha256(value):
            reasons.append(f"reference {name} is missing or invalid")

    reference_delta: float | None = None
    try:
        reference_delta = float(protocol["reference_delta_ec_eV"])
    except (KeyError, TypeError, ValueError):
        reasons.append("reference reference_delta_ec_eV is missing or invalid")
    if reference_delta is not None and not math.isclose(
        reference_delta,
        result.reference_delta_ec_eV,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        reasons.append("reference and SolarLab reference CBO values differ")

    temperature: float | None = None
    try:
        temperature = float(protocol["temperature_K"])
    except (KeyError, TypeError, ValueError):
        reasons.append("reference temperature_K is missing or invalid")
    if temperature is not None and (
        not np.isfinite(temperature) or temperature <= 0.0
    ):
        reasons.append("reference temperature_K must be finite and positive")

    deltas: list[float] = []
    for point in reference_points:
        try:
            delta = float(point["x"])
        except (KeyError, TypeError, ValueError):
            reasons.append("reference contains an invalid CBO coordinate")
            continue
        if not np.isfinite(delta):
            reasons.append("reference contains a non-finite CBO coordinate")
            continue
        deltas.append(delta)
    unique_delta_count = len(set(deltas))
    if len(deltas) < 2:
        reasons.append("reference requires at least two finite CBO points")
    if unique_delta_count != len(deltas):
        reasons.append("reference CBO coordinates must be unique")
    if any(right <= left for left, right in zip(deltas, deltas[1:])):
        reasons.append("reference CBO coordinates must be strictly increasing")
    declared_count = payload.get("sweeps", {}).get("CHI_ETL", {}).get(
        "n_points"
    )
    if declared_count is None or declared_count != len(reference_points):
        reasons.append("reference CHI_ETL.n_points does not match its points")

    return CBOExternalReferenceAudit(
        schema=schema if isinstance(schema, str) else None,
        schema_version=(
            schema_version if isinstance(schema_version, str) else None
        ),
        solver=solver if isinstance(solver, str) else None,
        solver_version=(
            solver_version if isinstance(solver_version, str) else None
        ),
        delta_ec_convention=(
            convention if isinstance(convention, str) else None
        ),
        swept_parameter=(
            swept_parameter if isinstance(swept_parameter, str) else None
        ),
        boundary_policy=(
            boundary_policy if isinstance(boundary_policy, str) else None
        ),
        reference_delta_ec_eV=reference_delta,
        temperature_K=temperature,
        illumination=illumination if isinstance(illumination, str) else None,
        independently_generated=(
            independently_generated
            if isinstance(independently_generated, bool)
            else None
        ),
        interpolated=interpolated if isinstance(interpolated, bool) else None,
        source_export_sha256=(
            source_export_sha256
            if isinstance(source_export_sha256, str)
            else None
        ),
        source_deck_sha256=(
            source_deck_sha256
            if isinstance(source_deck_sha256, str)
            else None
        ),
        parameter_manifest_sha256=(
            parameter_manifest_sha256
            if isinstance(parameter_manifest_sha256, str)
            else None
        ),
        point_count=len(reference_points),
        unique_delta_count=unique_delta_count,
        certified=not reasons,
        reasons=tuple(dict.fromkeys(reasons)),
    )


def compare_cbo_scan_to_scaps_reference(
    result: InterfaceCBOScanResult,
    reference_path: str | Path,
    *,
    metrics: tuple[str, ...] = ("Jsc",),
    maximum_normalized_error: float = 5.0e-2,
    maximum_critical_interval_distance_eV: float = 2.5e-2,
    maximum_reference_critical_interval_width_eV: float = 2.0e-2,
    minimum_matched_points: int = 3,
) -> CBOExternalValidation:
    """Compare normalized CBO responses with a content-addressed SCAPS file.

    Normalization at the declared reference CBO deliberately separates the
    interface-response trend from an absolute optical-current mismatch.  This
    is a CBO-response parity certificate, not an absolute device calibration.
    """
    path = Path(reference_path)
    raw = path.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    try:
        reference_points = payload["sweeps"]["CHI_ETL"]["points"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "reference must contain sweeps.CHI_ETL.points"
        ) from exc
    if minimum_matched_points < 2:
        raise ValueError("minimum_matched_points must be at least two")
    for name, value in (
        ("maximum_normalized_error", maximum_normalized_error),
        (
            "maximum_critical_interval_distance_eV",
            maximum_critical_interval_distance_eV,
        ),
        (
            "maximum_reference_critical_interval_width_eV",
            maximum_reference_critical_interval_width_eV,
        ),
    ):
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")

    certificates: list[CBOExternalMetricCertificate] = []
    for metric in metrics:
        simulation = _simulation_metric_samples(result, metric)
        reference = _reference_metric_samples(reference_points, metric)
        simulation_by_delta = {delta: value for delta, value in simulation}
        reference_by_delta = {delta: value for delta, value in reference}

        matched: list[tuple[float, float, float]] = []
        for sim_delta, sim_value in simulation_by_delta.items():
            reference_delta = next(
                (
                    delta
                    for delta in reference_by_delta
                    if math.isclose(
                        delta,
                        sim_delta,
                        rel_tol=0.0,
                        abs_tol=1.0e-9,
                    )
                ),
                None,
            )
            if reference_delta is not None:
                matched.append(
                    (
                        sim_delta,
                        sim_value,
                        reference_by_delta[reference_delta],
                    )
                )
        matched.sort()
        sim_reference = next(
            (
                value
                for delta, value in simulation
                if math.isclose(
                    delta,
                    result.reference_delta_ec_eV,
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                )
            ),
            None,
        )
        ext_reference = next(
            (
                value
                for delta, value in reference
                if math.isclose(
                    delta,
                    result.reference_delta_ec_eV,
                    rel_tol=0.0,
                    abs_tol=1.0e-9,
                )
            ),
            None,
        )
        reasons: list[str] = []
        if not result.complete:
            reasons.append("SolarLab scan did not complete its requested axis")
        if not result.certified:
            reasons.append("SolarLab scan lacks a numerical certificate")
        if len(matched) < minimum_matched_points:
            reasons.append(
                f"requires at least {minimum_matched_points} matched points; "
                f"received {len(matched)}"
            )
        normalized_sim: tuple[float, ...] = ()
        normalized_ref: tuple[float, ...] = ()
        rms_error: float | None = None
        max_error: float | None = None
        if (
            sim_reference is None
            or ext_reference is None
            or sim_reference == 0.0
            or ext_reference == 0.0
        ):
            reasons.append("reference CBO is absent or has a zero metric")
        else:
            normalized_sim = tuple(
                value / sim_reference for _delta, value, _ref in matched
            )
            normalized_ref = tuple(
                value / ext_reference for _delta, _sim, value in matched
            )
            errors = np.asarray(normalized_sim) - np.asarray(normalized_ref)
            if errors.size:
                rms_error = float(np.sqrt(np.mean(errors * errors)))
                max_error = float(np.max(np.abs(errors)))
                if max_error > maximum_normalized_error:
                    reasons.append(
                        f"maximum normalized error {max_error:.6g} exceeds "
                        f"{maximum_normalized_error:.6g}"
                    )

        simulated_interval = _interval_pair(_metric_interval(result, metric))
        reference_interval = _interval_pair(
            _critical_interval(
                reference,
                metric=metric,
                reference_delta_ec_eV=result.reference_delta_ec_eV,
                relative_drop_fraction=result.relative_drop_fraction,
            )
        )
        distance = _interval_distance(simulated_interval, reference_interval)
        if distance is None:
            reasons.append("one or both critical intervals are unresolved")
        elif distance > maximum_critical_interval_distance_eV:
            reasons.append(
                f"critical-interval distance {distance:.6g} eV exceeds "
                f"{maximum_critical_interval_distance_eV:.6g} eV"
            )
        reference_interval_width = (
            None
            if reference_interval is None
            else reference_interval[1] - reference_interval[0]
        )
        if (
            reference_interval_width is not None
            and reference_interval_width
            > maximum_reference_critical_interval_width_eV
        ):
            reasons.append(
                "reference critical-interval width "
                f"{reference_interval_width:.6g} eV exceeds "
                f"{maximum_reference_critical_interval_width_eV:.6g} eV"
            )

        certificates.append(
            CBOExternalMetricCertificate(
                metric=metric,
                matched_delta_ec_eV=tuple(item[0] for item in matched),
                simulated_normalized=normalized_sim,
                reference_normalized=normalized_ref,
                rms_normalized_error=rms_error,
                max_normalized_error=max_error,
                maximum_normalized_error=float(maximum_normalized_error),
                simulated_critical_interval_eV=simulated_interval,
                reference_critical_interval_eV=reference_interval,
                reference_critical_interval_width_eV=(
                    reference_interval_width
                ),
                maximum_reference_critical_interval_width_eV=float(
                    maximum_reference_critical_interval_width_eV
                ),
                critical_interval_distance_eV=distance,
                maximum_critical_interval_distance_eV=float(
                    maximum_critical_interval_distance_eV
                ),
                certified=not reasons,
                reasons=tuple(dict.fromkeys(reasons)),
            )
        )

    reference_audit = _audit_scaps_cbo_reference(
        payload,
        reference_points,
        result,
    )
    return CBOExternalValidation(
        reference_path=str(path),
        reference_sha256=hashlib.sha256(raw).hexdigest(),
        source_xlsx=payload.get("source_xlsx"),
        source_pdf=payload.get("source_pdf"),
        extracted_at=payload.get("extracted_at"),
        reference_audit=reference_audit,
        certificates=tuple(certificates),
    )


def solve_interface_cbo_scan(
    baseline_stack: DeviceStack,
    delta_ec_eV: np.ndarray,
    *,
    voltages_V: np.ndarray | None = None,
    voltage_grids_V: tuple[np.ndarray, ...] | list[np.ndarray] | None = None,
    N_grid: int = 30,
    reference_delta_ec_eV: float = 0.0,
    relative_drop_fraction: float = 0.01,
    minimum_delta_step_eV: float = 5.0e-4,
    maximum_delta_step_eV: float = 5.0e-2,
    minimum_voltage_step_V: float | None = None,
    mpp_interpolation: str = "sampled",
    max_bridge_points: int = 256,
    interface_transmission: float = 1.0,
    interface_transport_model: str = FERMI_RICHARDSON,
    interface_topology: str = DEDUPLICATED_QSS,
    boundary_policy: str = FIXED_CONTACTS,
    calculate_jv_metrics: bool = True,
    reference_initial_state: QuasiFermiSteadyStateResult | None = None,
    reference_initial_state_grid: np.ndarray | None = None,
    progress: Callable[[str, int, int, str], None] | None = None,
) -> InterfaceCBOScanResult:
    """Run a certified physical-interface CBO scan.

    Requested values may span both cliff and spike sides. The reference point
    is solved first; two independent continuations then walk outward so a
    difficult point on one side cannot contaminate the other branch. Temporary
    bridge points are included in ``short_circuit_trace`` and sharpen the Jsc
    onset bracket, but full J-V metrics are calculated only at requested CBOs.
    Set ``calculate_jv_metrics=False`` for a faster certified Jsc/grid study;
    the FF and PCE intervals are then returned unresolved.

    ``voltage_grids_V`` enables a coarse-to-fine nested voltage ladder. The
    complete finest J-V branch is solved once and the coarser metrics are
    extracted from strict subsets of its certified voltage points. Supplying
    both ``voltages_V`` and ``voltage_grids_V`` is rejected as ambiguous.

    ``boundary_policy='fixed_contacts'`` holds the configured electrostatic
    contact boundary fixed while varying only the ETL affinity. The
    ``'recomputed_built_in'`` alternative also changes ``V_bi`` and therefore
    represents a different physical experiment.

    ``interface_topology='two_sided_trace'`` removes shared material-boundary
    nodes and currently requires ``fermi_dirac_richardson``. A nonzero legacy
    ``het_recomb_despike`` is rejected because its target shared node no longer
    exists; callers must set it to zero explicitly and record that protocol
    change.

    A certified reference state from a coarser grid may be supplied with its
    source grid. It is used only as a nonlinear initial guess; the reference
    point is solved and certified again on this scan's target grid.

    A failure caused by the outer solver's audited Boltzmann log-density range
    is returned as a bracketed ``termination``. Other failures raise
    ``InterfaceCBOScanError`` because they are numerical failures, not a
    physical CBO limit.
    """
    requested = np.asarray(delta_ec_eV, dtype=float)
    if reference_initial_state is None and reference_initial_state_grid is not None:
        raise ValueError(
            "reference_initial_state_grid requires reference_initial_state"
        )
    if (
        requested.ndim != 1
        or requested.size == 0
        or not np.all(np.isfinite(requested))
        or np.any(np.diff(requested) <= 0.0)
    ):
        raise ValueError("delta_ec_eV must be finite and strictly increasing")
    reference_matches = np.flatnonzero(
        np.isclose(
            requested,
            reference_delta_ec_eV,
            rtol=0.0,
            atol=1.0e-12,
        )
    )
    if reference_matches.size != 1:
        raise ValueError(
            "delta_ec_eV must contain reference_delta_ec_eV exactly once"
        )
    if not np.isfinite(relative_drop_fraction) or not (
        0.0 < relative_drop_fraction < 1.0
    ):
        raise ValueError("relative_drop_fraction must lie in (0, 1)")
    if not np.isfinite(minimum_delta_step_eV) or minimum_delta_step_eV <= 0.0:
        raise ValueError("minimum_delta_step_eV must be finite and positive")
    if (
        not np.isfinite(maximum_delta_step_eV)
        or maximum_delta_step_eV < minimum_delta_step_eV
    ):
        raise ValueError(
            "maximum_delta_step_eV must be finite and no smaller than "
            "minimum_delta_step_eV"
        )
    if minimum_voltage_step_V is not None and (
        not np.isfinite(minimum_voltage_step_V)
        or minimum_voltage_step_V <= 0.0
    ):
        raise ValueError(
            "minimum_voltage_step_V must be finite and positive when enabled"
        )
    if mpp_interpolation not in ("sampled", "local_quadratic"):
        raise ValueError(
            "mpp_interpolation must be 'sampled' or 'local_quadratic'"
        )
    if max_bridge_points <= 0:
        raise ValueError("max_bridge_points must be positive")
    if (
        isinstance(N_grid, (bool, np.bool_))
        or not isinstance(N_grid, (int, np.integer))
        or int(N_grid) <= 0
    ):
        raise ValueError("N_grid must be a positive integer")
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
    if (
        topology == TWO_SIDED_TRACE
        and transport_model != FERMI_DIRAC_RICHARDSON
    ):
        raise ValueError(
            "two_sided_trace currently requires "
            f"interface_transport_model={FERMI_DIRAC_RICHARDSON!r}"
        )
    legacy_despike = float(
        getattr(baseline_stack, "het_recomb_despike", 0.0)
    )
    if topology == TWO_SIDED_TRACE and legacy_despike > 0.0:
        raise ValueError(
            "two_sided_trace removes the shared heterointerface node and is "
            "incompatible with its bulk-recombination de-spike; explicitly "
            "set stack.het_recomb_despike=0.0 for this physical protocol"
        )
    resolved_boundary_policy = validate_cbo_boundary_policy(boundary_policy)
    if voltages_V is not None and voltage_grids_V is not None:
        raise ValueError(
            "use either voltages_V or voltage_grids_V, not both"
        )
    if voltage_grids_V is not None and not calculate_jv_metrics:
        raise ValueError(
            "voltage_grids_V requires calculate_jv_metrics=True"
        )
    if voltage_grids_V is None:
        voltage_grid = (
            np.linspace(0.0, 1.4, 29)
            if voltages_V is None
            else np.asarray(voltages_V, dtype=float)
        )
        validated_single_grid = _nested_voltage_grids((voltage_grid,))
        resolved_voltage_grids = (
            validated_single_grid if calculate_jv_metrics else ()
        )
        voltages = validated_single_grid[-1]
    else:
        resolved_voltage_grids = _nested_voltage_grids(voltage_grids_V)
        voltages = resolved_voltage_grids[-1]

    grid_points = int(N_grid)
    grid = build_electrical_grid(baseline_stack, grid_points)
    if topology == TWO_SIDED_TRACE:
        grid = build_two_sided_trace_grid(grid, baseline_stack)
    target_values = [float(value) for value in requested]
    short_circuit_trace: list[CBOShortCircuitSample] = []
    short_states_by_delta: dict[float, QuasiFermiSteadyStateResult] = {}
    points_by_delta: dict[float, CBOScanPoint] = {}
    terminations: list[CBOScanTermination] = []
    bridge_count = 0
    reference_grid_warm_starts = 0
    reference_grid_warm_start_failures = 0
    reference_grid_cold_recoveries = 0
    reference_grid_predictor_recoveries = 0

    def notify(stage: str, current: int, message: str) -> None:
        if progress is not None:
            progress(stage, current, len(target_values), message)

    def solve_short_circuit(
        target_delta: float,
        initial_state: QuasiFermiSteadyStateResult | None,
        *,
        requested_point: bool,
        initial_state_grid: np.ndarray | None = None,
    ) -> QuasiFermiSteadyStateResult:
        stack = _stack_at_cbo(
            baseline_stack,
            target_delta,
            boundary_policy=resolved_boundary_policy,
        )
        solve_kwargs = {}
        if initial_state is not None:
            solve_kwargs["illumination_steps"] = (1.0,)
        if initial_state_grid is not None:
            solve_kwargs["initial_state_grid"] = initial_state_grid
        state = solve_quasi_fermi_steady_state(
            grid,
            stack,
            V_app=0.0,
            illuminated=True,
            interface_boundary=True,
            interface_topology=topology,
            interface_transmission=interface_transmission,
            interface_transport_model=transport_model,
            initial_state=initial_state,
            **solve_kwargs,
        )
        short_circuit_trace.append(
            CBOShortCircuitSample(
                delta_ec_eV=target_delta,
                current_A_m2=state.current_A_m2,
                requested=requested_point,
                face_current_spread_A_m2=state.face_current_spread_A_m2,
                interface_local_residual=state.interface_local_residual,
                interface_max_state_to_dos=float(
                    getattr(state, "interface_max_state_to_dos", 0.0)
                ),
                applied_V_bi_V=float(stack.V_bi),
            )
        )
        short_states_by_delta[target_delta] = state
        return state

    def solve_full_point(
        target_delta: float,
        short_circuit_state: QuasiFermiSteadyStateResult,
    ) -> CBOScanPoint:
        if not calculate_jv_metrics:
            return CBOScanPoint(
                delta_ec_eV=target_delta,
                short_circuit_state=short_circuit_state,
                jv=None,
                voltage_grid_metrics=(),
            )
        stack = _stack_at_cbo(
            baseline_stack,
            target_delta,
            boundary_policy=resolved_boundary_policy,
        )
        jv = solve_quasi_fermi_jv_sweep(
            grid,
            stack,
            voltages,
            interface_boundary=True,
            interface_topology=topology,
            interface_transmission=interface_transmission,
            interface_transport_model=transport_model,
            initial_short_circuit_state=short_circuit_state,
            stop_after_voc=True,
            voc_stop_grid_V=(
                resolved_voltage_grids[0]
                if len(resolved_voltage_grids) > 1
                else None
            ),
            minimum_voltage_step_V=minimum_voltage_step_V,
            mpp_interpolation=mpp_interpolation,
        )
        if not jv.metrics_certified:
            raise InterfaceCBOScanError(
                f"CBO {target_delta:+.6g} eV did not bracket a certified Voc"
            )
        if len(resolved_voltage_grids) == 1:
            only_grid = resolved_voltage_grids[0]
            metric_samples = [
                CBOJVMetricsGridSample(
                    voltage_point_count=len(only_grid),
                    voltage_interval_count=len(only_grid) - 1,
                    metrics=jv.metrics,
                    certified=jv.metrics_certified,
                    retained_voltage_point_count=len(jv.voltages_V),
                )
            ]
        else:
            metric_samples = []
        for voltage_grid in resolved_voltage_grids[:-1]:
            retained_grid = voltage_grid[
                voltage_grid <= jv.voltages_V[-1] + 1.0e-12
            ]
            source_indices: list[int] = []
            for voltage in retained_grid:
                matches = np.flatnonzero(
                    np.isclose(
                        jv.voltages_V,
                        voltage,
                        rtol=0.0,
                        atol=1.0e-12,
                    )
                )
                if matches.size != 1:
                    raise InterfaceCBOScanError(
                        "finest J-V branch did not retain the complete nested "
                        f"voltage grid at CBO {target_delta:+.6g} eV"
                    )
                source_indices.append(int(matches[0]))
            selected = np.asarray(source_indices, dtype=int)
            validity = tuple(jv.points[index].certified for index in selected)
            metrics = compute_metrics(
                jv.voltages_V[selected],
                jv.currents_A_m2[selected],
                P_in=1000.0,
                V_oc_max=thermodynamic_voc_ceiling(stack),
                validity=validity,
                mpp_interpolation=mpp_interpolation,
            )
            sample = CBOJVMetricsGridSample(
                voltage_point_count=len(voltage_grid),
                voltage_interval_count=len(voltage_grid) - 1,
                metrics=metrics,
                certified=bool(all(validity) and metrics.voc_bracketed),
                retained_voltage_point_count=len(retained_grid),
            )
            if not sample.certified:
                raise InterfaceCBOScanError(
                    f"CBO {target_delta:+.6g} eV did not bracket certified "
                    f"metrics on the {len(voltage_grid)}-point voltage grid"
                )
            metric_samples.append(sample)
        if len(resolved_voltage_grids) > 1:
            finest_grid = resolved_voltage_grids[-1]
            metric_samples.append(
                CBOJVMetricsGridSample(
                    voltage_point_count=len(finest_grid),
                    voltage_interval_count=len(finest_grid) - 1,
                    metrics=jv.metrics,
                    certified=jv.metrics_certified,
                    retained_voltage_point_count=len(jv.voltages_V),
                )
            )
        return CBOScanPoint(
            delta_ec_eV=target_delta,
            short_circuit_state=short_circuit_state,
            jv=jv,
            voltage_grid_metrics=tuple(metric_samples),
        )

    def advance(
        left_delta: float,
        left_state: QuasiFermiSteadyStateResult,
        right_delta: float,
        *,
        requested_point: bool,
    ) -> QuasiFermiSteadyStateResult:
        nonlocal bridge_count
        span = abs(right_delta - left_delta)
        if span > maximum_delta_step_eV * (1.0 + 1.0e-12):
            if bridge_count >= max_bridge_points:
                raise InterfaceCBOScanError(
                    f"CBO continuation exceeded {max_bridge_points} bridge points"
                )
            direction = 1.0 if right_delta > left_delta else -1.0
            bridge_delta = left_delta + direction * maximum_delta_step_eV
            bridge_count += 1
            bridge_state = advance(
                left_delta,
                left_state,
                bridge_delta,
                requested_point=False,
            )
            return advance(
                bridge_delta,
                bridge_state,
                right_delta,
                requested_point=requested_point,
            )
        try:
            return solve_short_circuit(
                right_delta,
                left_state,
                requested_point=requested_point,
            )
        except InterfaceCBOScanError:
            raise
        except (QuasiFermiSteadyStateError, RuntimeError, ValueError) as exc:
            if span <= minimum_delta_step_eV:
                if _is_bulk_statistics_limit(exc):
                    raise _ValidityLimit(left_delta, right_delta, exc) from exc
                raise InterfaceCBOScanError(
                    "CBO continuation failed inside interval "
                    f"[{left_delta:+.9g}, {right_delta:+.9g}] eV: "
                    f"{_error_chain(exc)}"
                ) from exc
            if bridge_count >= max_bridge_points:
                raise InterfaceCBOScanError(
                    f"CBO continuation exceeded {max_bridge_points} bridge points"
                ) from exc
            midpoint = 0.5 * (left_delta + right_delta)
            bridge_count += 1
            middle_state = advance(
                left_delta,
                left_state,
                midpoint,
                requested_point=False,
            )
            return advance(
                midpoint,
                middle_state,
                right_delta,
                requested_point=requested_point,
            )

    reference = float(requested[reference_matches[0]])
    notify("cbo", 0, f"Solving reference CBO {reference:+.6g} eV")
    try:
        reference_state = solve_short_circuit(
            reference,
            reference_initial_state,
            requested_point=True,
            initial_state_grid=reference_initial_state_grid,
        )
        if (
            reference_initial_state is not None
            and reference_initial_state_grid is not None
            and not np.array_equal(
                np.asarray(reference_initial_state_grid, dtype=float),
                grid,
            )
        ):
            reference_grid_warm_starts = 1
    except (QuasiFermiSteadyStateError, RuntimeError, ValueError) as exc:
        recovery_errors: list[BaseException] = [exc]
        recovered = False
        if reference_initial_state is not None:
            reference_grid_warm_start_failures = 1
            try:
                reference_state = solve_short_circuit(
                    reference,
                    None,
                    requested_point=True,
                )
                reference_grid_cold_recoveries = 1
                recovered = True
            except (
                QuasiFermiSteadyStateError,
                RuntimeError,
                ValueError,
            ) as cold_recovery:
                recovery_errors.append(cold_recovery)
        predictor_protocols: list[tuple[DeviceStack, int, str]] = []
        if not recovered:
            configured_alphas = tuple(
                float(value)
                for value in getattr(baseline_stack, "grid_alphas", ())
            )
            if configured_alphas and max(configured_alphas) > 3.0:
                predictor_protocols.append(
                    (
                        replace(
                            baseline_stack,
                            grid_alphas=tuple(
                                min(value, 3.0) for value in configured_alphas
                            ),
                        ),
                        grid_points,
                        "reduced clustering",
                    )
                )
            layer_count = len(electrical_layers(baseline_stack))
            coarse_grid_points = max(
                2 * layer_count,
                int(np.floor(0.8 * grid_points)),
            )
            if coarse_grid_points < grid_points:
                predictor_protocols.append(
                    (
                        baseline_stack,
                        coarse_grid_points,
                        "coarse-grid basin",
                    )
                )

        for predictor_stack, predictor_size, _label in predictor_protocols:
            predictor_grid = build_electrical_grid(
                predictor_stack,
                predictor_size,
            )
            if topology == TWO_SIDED_TRACE:
                predictor_grid = build_two_sided_trace_grid(
                    predictor_grid,
                    predictor_stack,
                )
            predictor_point_stack = _stack_at_cbo(
                predictor_stack,
                reference,
                boundary_policy=resolved_boundary_policy,
            )
            try:
                predictor_state = solve_quasi_fermi_steady_state(
                    predictor_grid,
                    predictor_point_stack,
                    V_app=0.0,
                    illuminated=True,
                    interface_boundary=True,
                    interface_topology=topology,
                    interface_transmission=interface_transmission,
                    interface_transport_model=transport_model,
                )
                reference_state = solve_short_circuit(
                    reference,
                    predictor_state,
                    requested_point=True,
                    initial_state_grid=predictor_grid,
                )
                reference_grid_warm_starts = 1
                reference_grid_predictor_recoveries = 1
                recovered = True
                break
            except (
                QuasiFermiSteadyStateError,
                RuntimeError,
                ValueError,
            ) as recovery:
                recovery_errors.append(recovery)
        if not recovered:
            raise InterfaceCBOScanError(
                f"reference CBO {reference:+.6g} eV failed after "
                "cross-grid recovery: "
                + ": ".join(
                    _error_chain(error) for error in recovery_errors
                )
            ) from recovery_errors[-1]
    reference_point = solve_full_point(reference, reference_state)
    points_by_delta[reference] = reference_point

    branches = (
        ("negative", sorted((value for value in target_values if value < reference), reverse=True)),
        ("positive", sorted(value for value in target_values if value > reference)),
    )
    completed_requested = 1
    for direction, branch in branches:
        current_delta = reference
        current_state = reference_state
        for target_delta in branch:
            notify(
                "cbo",
                completed_requested,
                f"Solving CBO {target_delta:+.6g} eV",
            )
            try:
                target_state = advance(
                    current_delta,
                    current_state,
                    target_delta,
                    requested_point=True,
                )
                target_point = solve_full_point(target_delta, target_state)
            except _ValidityLimit as limit:
                terminations.append(
                    CBOScanTermination(
                        direction=direction,
                        last_certified_delta_ec_eV=limit.last_delta_ec_eV,
                        first_failed_delta_ec_eV=limit.failed_delta_ec_eV,
                        requested_delta_ec_eV=target_delta,
                        reason=_error_chain(limit.cause),
                    )
                )
                break
            except InterfaceCBOScanError:
                raise
            except (QuasiFermiSteadyStateError, RuntimeError, ValueError) as exc:
                if _is_bulk_statistics_limit(exc):
                    terminations.append(
                        CBOScanTermination(
                            direction=direction,
                            last_certified_delta_ec_eV=current_delta,
                            first_failed_delta_ec_eV=target_delta,
                            requested_delta_ec_eV=target_delta,
                            reason=_error_chain(exc),
                        )
                    )
                    break
                raise InterfaceCBOScanError(
                    f"full J-V failed at CBO {target_delta:+.6g} eV: "
                    f"{_error_chain(exc)}"
                ) from exc
            points_by_delta[target_delta] = target_point
            current_delta = target_delta
            current_state = target_state
            completed_requested += 1

    # Refine the Jsc threshold itself, rather than treating continuation bridge
    # points as an accidental critical-value mesh.  This keeps the reported
    # interval width tied to ``minimum_delta_step_eV`` even when every coarse
    # requested point solves on the first attempt.
    provisional_jsc = _critical_interval(
        [
            (sample.delta_ec_eV, sample.current_A_m2)
            for sample in short_circuit_trace
        ],
        metric="Jsc",
        reference_delta_ec_eV=reference,
        relative_drop_fraction=relative_drop_fraction,
    )
    if (
        provisional_jsc.resolved
        and provisional_jsc.lower_delta_ec_eV is not None
        and provisional_jsc.upper_delta_ec_eV is not None
        and provisional_jsc.threshold_value is not None
    ):
        lower = provisional_jsc.lower_delta_ec_eV
        upper = provisional_jsc.upper_delta_ec_eV
        lower_state = short_states_by_delta[lower]
        while upper - lower > minimum_delta_step_eV * (1.0 + 1.0e-12):
            midpoint = 0.5 * (lower + upper)
            midpoint_state = advance(
                lower,
                lower_state,
                midpoint,
                requested_point=False,
            )
            if midpoint_state.current_A_m2 <= provisional_jsc.threshold_value:
                upper = midpoint
            else:
                lower = midpoint
                lower_state = midpoint_state

    points = tuple(points_by_delta[delta] for delta in sorted(points_by_delta))
    trace = tuple(
        sorted(
            short_circuit_trace,
            key=lambda sample: sample.delta_ec_eV,
        )
    )
    jsc_samples = [
        (sample.delta_ec_eV, sample.current_A_m2)
        for sample in trace
    ]
    metric_samples = {
        "FF": [
            (point.delta_ec_eV, point.metrics.FF)
            for point in points
            if point.metrics is not None
        ],
        "PCE": [
            (point.delta_ec_eV, point.metrics.PCE)
            for point in points
            if point.metrics is not None
        ],
    }
    critical_intervals = (
        _critical_interval(
            jsc_samples,
            metric="Jsc",
            reference_delta_ec_eV=reference,
            relative_drop_fraction=relative_drop_fraction,
        ),
        _critical_interval(
            metric_samples["FF"],
            metric="FF",
            reference_delta_ec_eV=reference,
            relative_drop_fraction=relative_drop_fraction,
        ),
        _critical_interval(
            metric_samples["PCE"],
            metric="PCE",
            reference_delta_ec_eV=reference,
            relative_drop_fraction=relative_drop_fraction,
        ),
    )
    notify("cbo", completed_requested, "CBO scan complete")
    return InterfaceCBOScanResult(
        requested_delta_ec_eV=requested.copy(),
        points=points,
        short_circuit_trace=trace,
        critical_intervals=critical_intervals,
        terminations=tuple(terminations),
        reference_delta_ec_eV=reference,
        interface_transmission=float(interface_transmission),
        relative_drop_fraction=float(relative_drop_fraction),
        minimum_delta_step_eV=float(minimum_delta_step_eV),
        maximum_delta_step_eV=float(maximum_delta_step_eV),
        minimum_voltage_step_V=(
            None
            if minimum_voltage_step_V is None
            else float(minimum_voltage_step_V)
        ),
        N_grid=grid_points,
        grid_node_count=len(grid),
        grid_interval_count=len(grid) - 1,
        grid_interval_weights=tuple(
            getattr(baseline_stack, "grid_interval_weights", ())
        ),
        grid_alphas=tuple(getattr(baseline_stack, "grid_alphas", ())),
        reference_grid_warm_starts=reference_grid_warm_starts,
        reference_grid_warm_start_failures=(
            reference_grid_warm_start_failures
        ),
        reference_grid_cold_recoveries=reference_grid_cold_recoveries,
        reference_grid_predictor_recoveries=(
            reference_grid_predictor_recoveries
        ),
        voltages_V=voltages.copy(),
        calculate_jv_metrics=bool(calculate_jv_metrics),
        boundary_policy=resolved_boundary_policy,
        interface_transport_model=transport_model,
        interface_topology=topology,
        heterojunction_recombination_despike=legacy_despike,
        qf_coordinate_system=str(
            getattr(reference_state, "qf_coordinate_system", "edge_drop")
        ),
        voltage_grids_V=tuple(
            voltage_grid.copy() for voltage_grid in resolved_voltage_grids
        ),
        mpp_interpolation=mpp_interpolation,
    )


__all__ = [
    "CBO_BOUNDARY_POLICIES",
    "CBOCriticalInterval",
    "CBOExternalMetricCertificate",
    "CBOExternalReferenceAudit",
    "CBOExternalValidation",
    "CBOGridConvergenceCertificate",
    "CBOJVMetricsGridSample",
    "CBOStatisticsValidityCertificate",
    "CBOVoltageGridConvergenceCertificate",
    "CBOVoltageGridPointCertificate",
    "CBOScanPoint",
    "CBOScanTermination",
    "CBOShortCircuitSample",
    "FIXED_CONTACTS",
    "InterfaceCBOScanError",
    "InterfaceCBOScanResult",
    "RECOMPUTED_BUILT_IN",
    "certify_cbo_grid_convergence",
    "certify_cbo_statistics_validity",
    "certify_cbo_voltage_grid_convergence",
    "compare_cbo_scan_to_scaps_reference",
    "solve_interface_cbo_scan",
    "validate_cbo_boundary_policy",
]
