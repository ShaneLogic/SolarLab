"""Repository-native executors for registered Phase-1 refinement lanes.

These adapters are deliberately thin.  They translate a frozen lane contract
into existing solver calls and return raw observables plus independent quality
metrics; certification remains owned by :mod:`numerical_certificate`.
"""

from __future__ import annotations

import dataclasses
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from perovskite_sim.constants import Q
from perovskite_sim.discretization.grid import (
    Layer,
    require_thick_layer_interface_resolution,
)
from perovskite_sim.experiments.jv_sweep import (
    _compute_current_ss_with_spread,
    build_jv_experiment_protocol,
    build_electrical_grid,
    compute_current_components,
    compute_metrics,
    run_jv_sweep,
)
from perovskite_sim.experiments.impedance import build_impedance_experiment_protocol
from perovskite_sim.experiments.ion_aware_dc import (
    build_ion_aware_dc_protocol,
    ion_aware_dc_state_sha256,
    solve_ion_aware_dc,
)
from perovskite_sim.experiments.ion_aware_impedance import (
    build_ion_aware_impedance_protocol,
    run_ion_aware_impedance,
)
from perovskite_sim.experiments.ion_aware_impedance_grid import (
    ion_aware_impedance_grid_sha256,
)
from perovskite_sim.experiments.mott_schottky import _fit_mott_schottky
from perovskite_sim.experiments.quasi_fermi_steady_state import (
    DEFAULT_ILLUMINATION_STEPS,
    _prepare_two_sided_material,
    build_equilibrium_referenced_interface_charge_dark_reference,
    build_two_sided_trace_grid,
    solve_equilibrium_referenced_interface_charge_steady_state,
    solve_quasi_fermi_jv_sweep,
    solve_quasi_fermi_steady_state,
)
from perovskite_sim.experiments.quasi_fermi_impedance import (
    run_quasi_fermi_impedance,
)
from perovskite_sim.experiments.steady_state import (
    solve_steady_state,
)
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.models.device import (
    electrical_interface_defects,
    electrical_interfaces,
    electrical_layers,
)
from perovskite_sim.physics.contacts import (
    require_contact_thermodynamic_certificate,
)
from perovskite_sim.physics.interface_plane import FERMI_DIRAC_RICHARDSON
from perovskite_sim.physics.two_sided_interface import (
    TWO_SIDED_TRACE,
    solve_material_two_sided_interfaces_qss,
)
from perovskite_sim.physics.generation import dual_cell_integral
from perovskite_sim.scaps_compat.loader import load_scaps_yaml
from perovskite_sim.solver.illuminated_ss import solve_illuminated_ss
from perovskite_sim.solver.mol import (
    StateVec,
    build_material_arrays,
)
from perovskite_sim.solver.numerical_diagnostics import (
    NumericalDiagnosticsReport,
)
from perovskite_sim.solver.tolerances import ComponentwiseAtol
from perovskite_sim.twod.experiments.jv_sweep_2d import (
    compute_terminal_current_2d,
    extract_snapshot_2d,
)
from perovskite_sim.twod.grid_2d import build_grid_2d
from perovskite_sim.twod.microstructure import Microstructure
from perovskite_sim.twod.solver_2d import (
    _charge_density_2d,
    build_material_arrays_2d,
    run_transient_2d,
)

from .numerical_certificate import LaneDefinition, MatrixPoint, content_sha256
from .refinement_runner import CellMeasurement


def _option(
    options: dict[str, Any],
    name: str,
    expected_type: type,
    default: Any,
) -> Any:
    value = options.get(name, default)
    if (
        expected_type is float
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
    ):
        return float(value)
    if expected_type is int and isinstance(value, int) and not isinstance(value, bool):
        return value
    if expected_type is bool and isinstance(value, bool):
        return value
    if expected_type is str and isinstance(value, str):
        return value
    raise ValueError(f"lane option {name!r} must be {expected_type.__name__}")


def _load_stack(lane: LaneDefinition, root: Path):
    loader = _option(lane.options, "config_loader", str, "standard")
    path = root / lane.config_path
    if loader == "standard":
        return load_device_from_yaml(path)
    if loader == "scaps":
        return load_scaps_yaml(path)
    raise ValueError(f"unsupported config_loader {loader!r}")


def _componentwise_policy(
    options: dict[str, Any],
    tolerance_factor: float,
) -> ComponentwiseAtol:
    return ComponentwiseAtol(
        carrier_fraction=_option(options, "carrier_atol_fraction", float, 1.0e-12),
        ion_fraction=_option(options, "ion_atol_fraction", float, 1.0e-12),
        interface_fraction=_option(options, "interface_atol_fraction", float, 1.0e-12),
        minimum_atol=_option(options, "minimum_atol", float, 1.0e-6),
    ).refined(tolerance_factor)


def _protocol_metadata(protocol: dict[str, Any]) -> dict[str, Any]:
    """Attach one canonical protocol document and its content hash."""
    schema = protocol.get("schema_version")
    if not isinstance(schema, str) or not schema:
        raise ValueError("refinement protocol requires a string schema_version")
    return {
        "protocol": protocol,
        "protocol_hash": content_sha256(protocol),
        "protocol_schema": schema,
    }


def _steady_jv_numerical_protocol(
    voltages: np.ndarray,
    *,
    adapter: str,
    illuminated: bool,
    base_residual_tolerance_per_s: float,
    base_log_step_tolerance: float,
    base_stall_tolerance_per_s: float,
    max_continuity_current_error_A_m2: float,
) -> dict[str, Any]:
    """Describe a numerical steady ladder without inventing experiment time."""
    return {
        "adapter": adapter,
        "continuation": {
            "first_point": "solver_default_seed",
            "subsequent_points": "previous_residual_certified_state",
        },
        "illumination": "baseline" if illuminated else "dark",
        "measurement": "steady_state_jv_ladder",
        "ordering": "ascending",
        "sampling_voltage_V": voltages.tolist(),
        "schema_version": "numerical-refinement-execution-protocol-v1",
        "settle": {
            "base_log_step_tolerance": base_log_step_tolerance,
            "base_residual_tolerance_per_s": base_residual_tolerance_per_s,
            "base_stall_tolerance_per_s": base_stall_tolerance_per_s,
            "max_continuity_current_error_A_m2": (
                max_continuity_current_error_A_m2
            ),
            "refinement_factor_source": "matrix.tolerance_factor",
            "type": "residual_certified",
        },
    }


def run_frozen_ion_steady_jv(
    lane: LaneDefinition,
    point: MatrixPoint,
    project_root: Path,
) -> CellMeasurement:
    """Run a strict carrier steady-state J-V lane with frozen ionic blocks."""
    options = lane.options
    stack = _load_stack(lane, project_root)
    x = build_electrical_grid(stack, point.grid)
    v_max = _option(options, "V_max_V", float, 1.5)
    n_points = _option(options, "voltage_points", int, 31)
    illuminated = _option(options, "illuminated", bool, True)
    iface_states = _option(options, "interface_states", bool, False)
    base_tol = _option(options, "base_residual_tolerance_per_s", float, 1.0e-6)
    base_step_tol = _option(options, "base_log_step_tolerance", float, 1.0e-8)
    base_accept = _option(options, "base_stall_tolerance_per_s", float, 0.5)
    max_current_error = _option(
        options,
        "max_continuity_current_error_A_m2",
        float,
        0.1,
    )
    voltages = np.linspace(0.0, v_max, n_points)
    numerical_protocol = _steady_jv_numerical_protocol(
        voltages,
        adapter="frozen-ion-residual-certified-steady-jv",
        illuminated=illuminated,
        base_residual_tolerance_per_s=base_tol,
        base_log_step_tolerance=base_step_tol,
        base_stall_tolerance_per_s=base_accept,
        max_continuity_current_error_A_m2=max_current_error,
    )
    currents: list[float] = []
    residuals: list[float] = []
    continuity_bounds: list[float] = []
    current_spreads: list[float] = []
    acceptances: list[str] = []
    iterations: list[int] = []
    previous_state = None

    for voltage in voltages:
        result = solve_steady_state(
            x,
            stack,
            float(voltage),
            illuminated=illuminated,
            y0=previous_state,
            tol=base_tol * point.tolerance_factor,
            tol_step=base_step_tol * point.tolerance_factor,
            tol_accept=base_accept * point.tolerance_factor,
            max_continuity_current_error=max_current_error,
            iface_states=iface_states,
            relative_log_variables=previous_state is not None,
        )
        previous_state = result.y
        current, spread = _compute_current_ss_with_spread(
            x,
            result.y,
            stack,
            float(voltage),
        )
        currents.append(current)
        residuals.append(result.residual)
        continuity_bounds.append(result.continuity_current_bound)
        current_spreads.append(spread)
        acceptances.append(result.acceptance)
        iterations.append(result.iterations)

    current_array = np.asarray(currents, dtype=float)
    metrics = compute_metrics(voltages, current_array)
    current_scale = max(abs(metrics.J_sc), 1.0e-30)
    return CellMeasurement.from_mapping(
        {
            "observables": {
                "ff": metrics.FF,
                "jsc_A_m2": metrics.J_sc,
                "jv_normalized": current_array / current_scale,
                "pce_percent": metrics.PCE,
                "voc_V": metrics.V_oc,
            },
            "quality": {
                "all_points_residual_converged": float(
                    all(item == "residual_converged" for item in acceptances)
                ),
                "max_continuity_bound_A_m2": max(continuity_bounds),
                "max_current_spread_A_m2": max(current_spreads),
                "max_residual_per_s": max(residuals),
                "voc_bracketed": float(metrics.voc_bracketed),
            },
            "units": {
                "jsc_A_m2": "A m-2",
                "pce_percent": "%",
                "voc_V": "V",
                "max_continuity_bound_A_m2": "A m-2",
                "max_current_spread_A_m2": "A m-2",
                "max_residual_per_s": "s-1",
            },
            "metadata": {
                **_protocol_metadata(numerical_protocol),
                "actual_intervals": len(x) - 1,
                "acceptance": acceptances,
                "newton_iterations": iterations,
                "voltage_grid_V": voltages.tolist(),
            },
        }
    )


def run_mobile_ion_transient_jv(
    lane: LaneDefinition,
    point: MatrixPoint,
    project_root: Path,
) -> CellMeasurement:
    """Run a coupled transient J-V lane with componentwise tolerance scaling."""
    options = lane.options
    stack = _load_stack(lane, project_root)
    v_max = _option(options, "V_max_V", float, 1.5)
    n_points = _option(options, "voltage_points", int, 20)
    v_rate = _option(options, "scan_rate_V_s", float, 5.0)
    rtol = _option(options, "rtol", float, 1.0e-4)
    policy = _componentwise_policy(options, point.tolerance_factor)
    experiment_protocol = build_jv_experiment_protocol(
        stack,
        v_rate=v_rate,
        n_points=n_points,
        V_max=v_max,
        illuminated=True,
        implicit_legacy_protocol=False,
    )
    result = run_jv_sweep(
        stack,
        N_grid=point.grid,
        v_rate=v_rate,
        n_points=n_points,
        rtol=rtol,
        atol=policy,  # runtime accepts the explicit componentwise policy
        V_max=v_max,
        save_snapshots=True,
        certification_mode="strict",
        experiment_protocol=experiment_protocol,
        protocol_mode="research_strict",
        collect_numerical_diagnostics=True,
    )
    resolved_protocol = getattr(result, "protocol", experiment_protocol)
    if (
        resolved_protocol is None
        or resolved_protocol.protocol_hash != experiment_protocol.protocol_hash
    ):
        raise RuntimeError("transient J-V returned a different experiment protocol")
    protocol_document = {
        "experiment_protocol": resolved_protocol.to_dict(),
        "schema_version": "experiment-protocol-v1",
    }
    snapshots = tuple(result.snapshots_fwd or ()) + tuple(result.snapshots_rev or ())
    if not snapshots:
        raise RuntimeError("transient J-V did not return ion-inventory snapshots")
    inventories = np.asarray(
        [dual_cell_integral(snapshot.x, snapshot.P) for snapshot in snapshots],
        dtype=float,
    )
    inventory_scale = max(abs(float(inventories[0])), 1.0e-30)
    normalized_inventory = inventories / inventory_scale
    inventory_drift = float(np.max(np.abs(normalized_inventory - 1.0)))
    terminal_trace = np.concatenate((result.J_fwd, result.J_rev))
    current_scale = max(abs(result.metrics_rev.J_sc), 1.0e-30)
    all_statuses = tuple(result.status_fwd or ()) + tuple(result.status_rev or ())
    reports: list[NumericalDiagnosticsReport] = []
    initial_diagnostics = getattr(result, "initial_numerical_diagnostics", None)
    initial_report = getattr(initial_diagnostics, "numerical_diagnostics", None)
    diagnostics_complete = (
        len(all_statuses) == terminal_trace.size
        and isinstance(initial_report, NumericalDiagnosticsReport)
    )
    if isinstance(initial_report, NumericalDiagnosticsReport):
        reports.append(initial_report)
    for status in all_statuses:
        accepted = tuple(getattr(status, "numerical_diagnostics", ()))
        if not bool(getattr(status, "valid", False)) or not accepted:
            diagnostics_complete = False
            continue
        for segment in accepted:
            report = getattr(segment, "report", None)
            if not isinstance(report, NumericalDiagnosticsReport):
                diagnostics_complete = False
                continue
            reports.append(report)
    diagnostics_complete = bool(
        diagnostics_complete
        and reports
        and all(
            report.solver_success is True
            and report.final_minimum_density_m3 is not None
            for report in reports
        )
    )

    def _finite_minimum(values: list[float | None]) -> float | None:
        finite = [
            float(value)
            for value in values
            if value is not None and np.isfinite(value)
        ]
        return min(finite) if finite else None

    terminal_carrier_minimum = _finite_minimum(
        [
            value
            for report in reports
            if report.final_minimum_density_m3 is not None
            for value in (
                report.final_minimum_density_m3.n,
                report.final_minimum_density_m3.p,
            )
        ]
    )
    terminal_positive_ion_minimum = _finite_minimum(
        [
            report.final_minimum_density_m3.positive_ion_active
            for report in reports
            if report.final_minimum_density_m3 is not None
        ]
    )
    terminal_negative_ion_minimum = _finite_minimum(
        [
            report.final_minimum_density_m3.negative_ion_active
            for report in reports
            if report.final_minimum_density_m3 is not None
        ]
    )
    terminal_interface_state_minimum = _finite_minimum(
        [
            report.final_minimum_density_m3.interface_state
            for report in reports
            if report.final_minimum_density_m3 is not None
        ]
    )
    active_terminal_minima = [
        terminal_carrier_minimum,
        terminal_positive_ion_minimum,
    ]
    active_terminal_minima.extend(
        value
        for value in (
            terminal_negative_ion_minimum,
            terminal_interface_state_minimum,
        )
        if value is not None
    )
    terminal_densities_positive = bool(
        diagnostics_complete
        and reports
        and all(value is not None and value > 0.0 for value in active_terminal_minima)
    )
    minimum_bulk_srh = _finite_minimum(
        [report.minimum_bulk_srh_denominator_s_m3 for report in reports]
    )
    minimum_interface_srh = _finite_minimum(
        [report.minimum_interface_srh_denominator_s_m4 for report in reports]
    )
    nonfinite_trial_evaluations = sum(
        report.nonfinite_trial_evaluations for report in reports
    )
    nonfinite_rhs_evaluations = sum(
        report.nonfinite_rhs_evaluations for report in reports
    )
    negative_trial_evaluations = sum(
        report.negative_trial_evaluations for report in reports
    )
    negative_trial_entries = {
        name: sum(
            getattr(report.negative_trial_entries, name) for report in reports
        )
        for name in (
            "n",
            "p",
            "positive_ion",
            "negative_ion",
            "interface_state",
        )
    }
    nfev_values = [
        value
        for status in all_statuses
        for value in (getattr(status, "nfev", None),)
        if isinstance(value, int)
    ]
    return CellMeasurement.from_mapping(
        {
            "observables": {
                "hysteresis_index": result.hysteresis_index,
                "positive_ion_inventory_relative_trace": normalized_inventory,
                "terminal_current_normalized_trace": terminal_trace / current_scale,
                "voc_reverse_V": result.metrics_rev.V_oc,
            },
            "quality": {
                "bulk_srh_denominator_positive": float(
                    minimum_bulk_srh is not None and minimum_bulk_srh > 0.0
                ),
                "diagnostics_complete": float(diagnostics_complete and bool(reports)),
                "interface_srh_denominator_positive": float(
                    minimum_interface_srh is not None
                    and minimum_interface_srh > 0.0
                ),
                "jv_certified": float(result.certified),
                "max_positive_ion_inventory_relative_drift": inventory_drift,
                "nonfinite_rhs_evaluations": float(nonfinite_rhs_evaluations),
                "nonfinite_trial_evaluations": float(
                    nonfinite_trial_evaluations
                ),
                "terminal_densities_positive": float(
                    terminal_densities_positive
                ),
                "voc_bracketed": float(result.metrics_rev.voc_bracketed),
                "zero_floor_diagnostics_pass": float(
                    diagnostics_complete
                    and all(report.would_pass_strict for report in reports)
                ),
            },
            "units": {
                "voc_reverse_V": "V",
            },
            "metadata": {
                **_protocol_metadata(protocol_document),
                "accepted_solver_segment_count": (
                    len(reports)
                    - int(isinstance(initial_report, NumericalDiagnosticsReport))
                ),
                "initial_preconditioner_diagnostics_present": bool(
                    isinstance(initial_report, NumericalDiagnosticsReport)
                ),
                "initial_preconditioner_nfev": getattr(
                    initial_diagnostics, "nfev", None
                ),
                "initial_preconditioner_njev": getattr(
                    initial_diagnostics, "njev", None
                ),
                "initial_preconditioner_nlu": getattr(
                    initial_diagnostics, "nlu", None
                ),
                "minimum_bulk_srh_denominator_s_m3": minimum_bulk_srh,
                "minimum_interface_srh_denominator_s_m4": minimum_interface_srh,
                "minimum_terminal_carrier_density_m3": terminal_carrier_minimum,
                "minimum_terminal_interface_state_density_m3": (
                    terminal_interface_state_minimum
                ),
                "minimum_terminal_negative_ion_density_m3": (
                    terminal_negative_ion_minimum
                ),
                "minimum_terminal_positive_ion_density_m3": (
                    terminal_positive_ion_minimum
                ),
                "negative_trial_entries": negative_trial_entries,
                "negative_trial_evaluations": negative_trial_evaluations,
                "nfev_reported_sum": sum(nfev_values) if nfev_values else None,
                "reverse_voltage_grid_V": result.V_rev.tolist(),
                "forward_voltage_grid_V": result.V_fwd.tolist(),
            },
        }
    )


def _numeric_array_option(
    options: dict[str, Any],
    name: str,
    default: tuple[float, ...],
) -> np.ndarray:
    raw = options.get(name, default)
    try:
        values = np.asarray(raw, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"lane option {name!r} must be a numeric array") from exc
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError(f"lane option {name!r} must be a finite 1-D array")
    return values


def _complex_metadata(values: np.ndarray) -> list[dict[str, float]]:
    """Encode a finite complex trace without relying on JSON coercion."""
    trace = np.asarray(values, dtype=complex)
    if trace.ndim != 1 or not np.all(np.isfinite(trace)):
        raise ValueError("complex metadata trace must be finite and one-dimensional")
    return [
        {"real": float(value.real), "imag": float(value.imag)}
        for value in trace
    ]


def run_ion_aware_dc_operating_point(
    lane: LaneDefinition,
    point: MatrixPoint,
    project_root: Path,
) -> CellMeasurement:
    """Certify one fixed-bias mobile-ion DC state and its refinement outputs."""
    options = lane.options
    stack = _load_stack(lane, project_root)
    grid = build_electrical_grid(stack, point.grid)
    material = build_material_arrays(grid, stack)
    voltage = _option(options, "V_dc_V", float, 0.9)
    illuminated = _option(options, "illuminated", bool, True)
    rtol = _option(options, "rtol", float, 1.0e-4)
    end_times = tuple(
        float(value)
        for value in _numeric_array_option(
            options,
            "settle_end_times_s",
            (1.0e-3, 1.0e-2, 1.0e-1, 1.0, 10.0, 32.0, 64.0, 128.0),
        )
    )
    required_passes = _option(options, "required_consecutive_passes", int, 2)
    max_carrier_rate = _option(
        options, "max_carrier_area_rate_A_m2", float, 1.0e-1
    )
    max_ion_rate = _option(options, "max_ion_area_rate_A_m2", float, 1.0e-6)
    max_ion_current = _option(
        options, "max_ionic_face_current_A_m2", float, 1.0e-6
    )
    max_current_spread = _option(
        options, "max_dc_face_current_spread_A_m2", float, 1.0e-1
    )
    max_inventory_drift = _option(
        options, "max_ion_inventory_relative_drift", float, 1.0e-10
    )
    max_nfev = _option(options, "max_nfev_per_attempt", int, 20_000)
    base_policy = ComponentwiseAtol(
        carrier_fraction=_option(
            options, "carrier_atol_fraction", float, 1.0e-12
        ),
        ion_fraction=_option(options, "ion_atol_fraction", float, 1.0e-12),
        interface_fraction=_option(
            options, "interface_atol_fraction", float, 1.0e-12
        ),
        minimum_atol=_option(options, "minimum_atol", float, 1.0e-6),
    )
    protocol = build_ion_aware_dc_protocol(
        stack,
        V_dc=voltage,
        illuminated=illuminated,
        settle_end_times_s=end_times,
        required_consecutive_passes=required_passes,
        max_carrier_area_rate_A_m2=max_carrier_rate,
        max_ion_area_rate_A_m2=max_ion_rate,
        max_ionic_face_current_A_m2=max_ion_current,
        max_dc_face_current_spread_A_m2=max_current_spread,
        max_ion_inventory_relative_drift=max_inventory_drift,
    )
    numerical_protocol = {
        "dc_protocol": protocol.to_dict(),
        "numerical_controls": {
            "atol_policy": {
                "carrier_fraction": base_policy.carrier_fraction,
                "interface_fraction": base_policy.interface_fraction,
                "ion_fraction": base_policy.ion_fraction,
                "minimum_atol": base_policy.minimum_atol,
                "refinement_factor_source": "matrix.tolerance_factor",
            },
            "grid_source": "matrix.grid",
            "max_nfev_per_attempt": max_nfev,
            "method_ladder": ["Radau", "BDF"],
            "rtol": rtol,
        },
        "schema_version": "ion-aware-dc-execution-protocol-v1",
    }
    result = solve_ion_aware_dc(
        grid,
        stack,
        protocol,
        mat=material,
        rtol=rtol,
        atol=base_policy.refined(point.tolerance_factor),
        method_ladder=("Radau", "BDF"),
        max_nfev_per_attempt=max_nfev,
        require_numerical_certificate=False,
        require_contact_certificate=False,
    )
    certificate = result.state_certificate
    reports = tuple(step.numerical_diagnostics for step in result.steps)
    attempts = tuple(
        attempt for step in result.steps for attempt in step.attempts
    )
    attempt_reports = tuple(
        attempt.numerical_diagnostics
        for attempt in attempts
        if attempt.numerical_diagnostics is not None
    )
    candidate_steps = result.steps[-required_passes:]
    diagnostics_complete = bool(
        reports
        and len(attempt_reports) == len(attempts)
        and all(
            report.solver_success is True
            and report.final_minimum_density_m3 is not None
            and report.minimum_bulk_srh_denominator_s_m3 is not None
            for report in reports
        )
    )
    terminal_densities_positive = all(
        value is None or value > protocol.terminal_density_floor_m3
        for value in (
            certificate.minimum_electron_density_m3,
            certificate.minimum_hole_density_m3,
            certificate.minimum_positive_ion_density_m3,
            certificate.minimum_negative_ion_density_m3,
        )
    )
    contact = certificate.contact_thermodynamics
    bulk_srh_denominators = tuple(
        report.minimum_bulk_srh_denominator_s_m3
        for report in reports
        if report.minimum_bulk_srh_denominator_s_m3 is not None
    )
    minimum_bulk_srh = (
        min(bulk_srh_denominators) if bulk_srh_denominators else None
    )
    negative_trials = sum(
        report.negative_trial_evaluations for report in attempt_reports
    )
    return CellMeasurement.from_mapping(
        {
            "observables": {
                "dc_current_density_A_m2": certificate.dc_current_density_A_m2,
                "maximum_site_occupancy_fraction": (
                    certificate.maximum_site_occupancy_fraction
                ),
                "positive_ion_centroid_fraction": (
                    certificate.positive_ion_inventory.terminal_centroid_fraction
                ),
            },
            "quality": {
                "candidate_diagnostics_pass": float(
                    len(candidate_steps) == required_passes
                    and all(step.diagnostics_passed for step in candidate_steps)
                ),
                "contact_not_inconsistent": float(contact.status != "inconsistent"),
                "diagnostics_complete": float(diagnostics_complete),
                "max_carrier_area_rate_A_m2": (
                    certificate.carrier_area_rate_A_m2
                ),
                "max_dc_face_current_spread_A_m2": (
                    certificate.dc_face_current_spread_A_m2
                ),
                "max_ion_area_rate_A_m2": certificate.ion_area_rate_A_m2,
                "max_ion_inventory_relative_drift": (
                    certificate.max_ion_inventory_relative_drift
                ),
                "max_ionic_face_current_A_m2": (
                    certificate.max_ionic_face_current_A_m2
                ),
                "nonfinite_rhs_evaluations": float(
                    sum(
                        report.nonfinite_rhs_evaluations
                        for report in attempt_reports
                    )
                ),
                "nonfinite_trial_evaluations": float(
                    sum(
                        report.nonfinite_trial_evaluations
                        for report in attempt_reports
                    )
                ),
                "required_consecutive_passes_met": float(
                    result.numerically_certified
                ),
                "site_occupancy_admissible": float(
                    certificate.maximum_site_occupancy_fraction <= 1.0 + 1.0e-8
                ),
                "terminal_densities_positive": float(
                    terminal_densities_positive
                ),
            },
            "units": {
                "dc_current_density_A_m2": "A m-2",
                "max_carrier_area_rate_A_m2": "A m-2",
                "max_dc_face_current_spread_A_m2": "A m-2",
                "max_ion_area_rate_A_m2": "A m-2",
                "max_ionic_face_current_A_m2": "A m-2",
            },
            "metadata": {
                **_protocol_metadata(numerical_protocol),
                "actual_intervals": len(grid) - 1,
                "contact_thermodynamics": dataclasses.asdict(contact),
                "accepted_methods": [step.accepted_method for step in result.steps],
                "failed_attempt_count": sum(not attempt.success for attempt in attempts),
                "minimum_bulk_srh_denominator_s_m3": minimum_bulk_srh,
                "negative_trial_evaluations": negative_trials,
                "nfev_reported_sum": sum(
                    step.nfev for step in result.steps if step.nfev is not None
                ),
                "numerically_certified": result.numerically_certified,
                "thermodynamically_certified": (
                    result.thermodynamically_certified
                ),
                "total_settle_time_s": result.total_settle_time_s,
                "used_settle_end_times_s": [
                    step.target_time_s for step in result.steps
                ],
            },
        }
    )


def run_ion_aware_impedance_frequency_domain(
    lane: LaneDefinition,
    point: MatrixPoint,
    project_root: Path,
) -> CellMeasurement:
    """Run one cell of the ion-aware impedance grid/stencil matrix."""
    options = lane.options
    stack = _load_stack(lane, project_root)
    grid = build_electrical_grid(stack, point.grid)
    material = build_material_arrays(grid, stack)

    voltage = _option(options, "V_dc_V", float, 0.9)
    illuminated = _option(options, "illuminated", bool, True)
    dc_rtol = _option(options, "dc_rtol", float, 1.0e-4)
    end_times = tuple(
        float(value)
        for value in _numeric_array_option(
            options,
            "settle_end_times_s",
            (1.0e-3, 1.0e-2, 1.0e-1, 1.0, 10.0, 32.0, 64.0, 128.0),
        )
    )
    required_passes = _option(options, "required_consecutive_passes", int, 2)
    max_carrier_rate = _option(
        options, "max_carrier_area_rate_A_m2", float, 1.0e-1
    )
    max_ion_rate = _option(options, "max_ion_area_rate_A_m2", float, 1.0e-6)
    max_ion_current = _option(
        options, "max_ionic_face_current_A_m2", float, 1.0e-6
    )
    max_dc_spread = _option(
        options, "max_dc_face_current_spread_A_m2", float, 1.0e-1
    )
    max_dc_inventory_drift = _option(
        options, "max_dc_ion_inventory_relative_drift", float, 1.0e-10
    )
    max_nfev = _option(options, "max_nfev_per_attempt", int, 20_000)
    dc_atol = ComponentwiseAtol(
        carrier_fraction=_option(
            options, "carrier_atol_fraction", float, 1.0e-12
        ),
        ion_fraction=_option(options, "ion_atol_fraction", float, 1.0e-12),
        interface_fraction=_option(
            options, "interface_atol_fraction", float, 1.0e-12
        ),
        minimum_atol=_option(options, "minimum_atol", float, 1.0e-6),
    )
    dc_protocol = build_ion_aware_dc_protocol(
        stack,
        V_dc=voltage,
        illuminated=illuminated,
        settle_end_times_s=end_times,
        required_consecutive_passes=required_passes,
        max_carrier_area_rate_A_m2=max_carrier_rate,
        max_ion_area_rate_A_m2=max_ion_rate,
        max_ionic_face_current_A_m2=max_ion_current,
        max_dc_face_current_spread_A_m2=max_dc_spread,
        max_ion_inventory_relative_drift=max_dc_inventory_drift,
    )
    dc_result = solve_ion_aware_dc(
        grid,
        stack,
        dc_protocol,
        mat=material,
        rtol=dc_rtol,
        atol=dc_atol,
        method_ladder=("Radau", "BDF"),
        max_nfev_per_attempt=max_nfev,
        require_numerical_certificate=False,
        require_contact_certificate=False,
    )

    frequency_min = _option(options, "frequency_min_Hz", float, 1.0e-6)
    frequency_max = _option(options, "frequency_max_Hz", float, 10.0)
    frequency_count = _option(options, "frequency_count", int, 29)
    spacing = _option(options, "frequency_spacing", str, "logspace")
    if frequency_min <= 0.0 or frequency_max <= frequency_min:
        raise ValueError("frequency bounds must be positive and increasing")
    if frequency_count < 2:
        raise ValueError("frequency_count must be at least two")
    if spacing != "logspace":
        raise ValueError("ion-aware impedance matrix requires logspace frequencies")
    frequencies = np.logspace(
        np.log10(frequency_min),
        np.log10(frequency_max),
        frequency_count,
    )

    delta_voltage = _option(options, "delta_V", float, 0.01)
    base_state_step = _option(options, "base_state_step", float, 1.0e-5)
    base_voltage_step = _option(options, "base_voltage_step", float, 1.0e-5)
    refinement_factors = tuple(
        float(value)
        for value in _numeric_array_option(
            options, "internal_refinement_factors", (1.0, 0.5, 0.25)
        )
    )
    max_face_spread = _option(
        options, "max_relative_face_spread", float, 5.0e-4
    )
    max_backward = _option(options, "max_backward_error", float, 1.0e-10)
    max_stencil_magnitude = _option(
        options,
        "max_impedance_magnitude_relative_change",
        float,
        1.0e-2,
    )
    max_stencil_phase = _option(
        options, "max_impedance_phase_change_deg", float, 0.5
    )
    max_mass = _option(
        options, "max_mass_matrix_relative_error", float, 1.0e-8
    )
    max_inventory_response = _option(
        options, "max_ion_inventory_response_relative", float, 1.0e-8
    )
    max_decomposition = _option(
        options, "max_current_decomposition_relative_error", float, 1.0e-7
    )
    branch_margin = _option(
        options, "frequency_branch_margin_decades", float, 1.0
    )
    max_frequency_gap = _option(
        options, "max_frequency_sampling_gap_decades", float, 0.5
    )
    impedance_protocol = build_ion_aware_impedance_protocol(
        dc_result,
        frequencies,
        delta_V=delta_voltage,
        state_step=base_state_step * point.tolerance_factor,
        voltage_step=base_voltage_step * point.tolerance_factor,
        refinement_factors=refinement_factors,
        max_relative_face_spread=max_face_spread,
        max_backward_error=max_backward,
        max_impedance_magnitude_relative_change=max_stencil_magnitude,
        max_impedance_phase_change_deg=max_stencil_phase,
        max_mass_matrix_relative_error=max_mass,
        max_ion_inventory_response_relative=max_inventory_response,
        max_current_decomposition_relative_error=max_decomposition,
        frequency_branch_margin_decades=branch_margin,
        max_frequency_sampling_gap_decades=max_frequency_gap,
    )
    impedance_result = run_ion_aware_impedance(
        grid,
        stack,
        impedance_protocol,
        dc_state=dc_result,
        require_numerical_certificate=False,
        require_contact_certificate=False,
        require_frequency_window_certificate=False,
    )

    certificate = impedance_result.certificate
    dc_certificate = dc_result.state_certificate
    contact = dc_certificate.contact_thermodynamics
    observed_frequency_gap = (
        impedance_result.frequency_window.max_observed_sampling_gap_decades
    )
    if observed_frequency_gap is None:
        observed_frequency_gap = np.finfo(float).max
    mass_matrix_error = max(
        certificate.max_mass_diagonal_relative_error,
        certificate.max_mass_off_diagonal_relative,
    )
    numerical_protocol = {
        "acceptance": {
            "matrix_observables": {
                gate.metric: gate.to_dict() for gate in lane.observables
            },
            "per_cell_quality": {
                gate.metric: gate.to_dict() for gate in lane.quality_gates
            },
        },
        "adapter": "ionmonger-ion-aware-impedance-grid-stencil-matrix",
        "dc_protocol": dc_protocol.to_dict(),
        "frequency_request": {
            "count": frequency_count,
            "maximum_Hz": frequency_max,
            "minimum_Hz": frequency_min,
            "spacing": spacing,
        },
        "numerical_controls": {
            "dc_atol_policy": dataclasses.asdict(dc_atol),
            "dc_atol_refinement_factor": 1.0,
            "dc_rtol": dc_rtol,
            "finite_difference_factor_source": "matrix.tolerance_factor",
            "grid_source": "matrix.grid",
            "impedance_base_state_step": base_state_step,
            "impedance_base_voltage_step": base_voltage_step,
            "impedance_internal_refinement_factors": list(refinement_factors),
            "max_nfev_per_attempt": max_nfev,
            "method_ladder": ["Radau", "BDF"],
        },
        "schema_version": "ion-aware-impedance-refinement-execution-protocol-v1",
    }
    frequency_evidence = [
        {
            "backward_error": item.backward_error,
            "current_decomposition_relative_error": (
                item.current_decomposition_relative_error
            ),
            "electron_storage_response_F_m2": {
                "real": float(item.electron_storage_response_F_m2.real),
                "imag": float(item.electron_storage_response_F_m2.imag),
            },
            "frequency_Hz": item.frequency_Hz,
            "hole_storage_response_F_m2": {
                "real": float(item.hole_storage_response_F_m2.real),
                "imag": float(item.hole_storage_response_F_m2.imag),
            },
            "max_relative_face_spread": item.max_relative_face_spread,
            "negative_ion_inventory_response_relative": (
                item.negative_ion_inventory_response_relative
            ),
            "negative_ion_storage_response_F_m2": (
                None
                if item.negative_ion_storage_response_F_m2 is None
                else {
                    "real": float(
                        item.negative_ion_storage_response_F_m2.real
                    ),
                    "imag": float(
                        item.negative_ion_storage_response_F_m2.imag
                    ),
                }
            ),
            "net_charge_storage_response_F_m2": {
                "real": float(item.net_charge_storage_response_F_m2.real),
                "imag": float(item.net_charge_storage_response_F_m2.imag),
            },
            "numerically_certified": item.numerically_certified,
            "perturbation_assessments": [
                dataclasses.asdict(assessment)
                for assessment in item.perturbation_assessments
            ],
            "positive_ion_inventory_response_relative": (
                item.positive_ion_inventory_response_relative
            ),
            "positive_ion_storage_response_F_m2": {
                "real": float(item.positive_ion_storage_response_F_m2.real),
                "imag": float(item.positive_ion_storage_response_F_m2.imag),
            },
            "reasons": list(item.reasons),
            "reciprocal_condition": item.reciprocal_condition,
        }
        for item in certificate.frequency_point_certificates
    ]
    return CellMeasurement.from_mapping(
        {
            "observables": {
                "impedance_magnitude_ohm_m2": np.abs(impedance_result.Z),
                "impedance_phase_deg": np.angle(impedance_result.Z, deg=True),
            },
            "quality": {
                "all_frequency_points_certified": float(
                    all(
                        item.numerically_certified
                        for item in certificate.frequency_point_certificates
                    )
                ),
                "contact_not_inconsistent": float(contact.status != "inconsistent"),
                "dc_numerically_certified": float(dc_result.numerically_certified),
                "frequency_window_certified": float(
                    certificate.frequency_window_certified
                ),
                "impedance_numerically_certified": float(
                    certificate.numerically_certified
                ),
                "max_backward_error": certificate.max_backward_error,
                "max_current_decomposition_relative_error": (
                    certificate.max_current_decomposition_relative_error
                ),
                "max_dc_ion_inventory_relative_drift": (
                    dc_certificate.max_ion_inventory_relative_drift
                ),
                "max_frequency_sampling_gap_decades": observed_frequency_gap,
                "max_ion_inventory_response_relative": (
                    certificate.max_ion_inventory_response_relative
                ),
                "max_mass_matrix_relative_error": mass_matrix_error,
                "max_relative_face_spread": certificate.max_relative_face_spread,
                "site_occupancy_admissible": float(
                    dc_certificate.maximum_site_occupancy_fraction <= 1.0 + 1.0e-8
                ),
            },
            "units": {
                "impedance_magnitude_ohm_m2": "ohm m2",
                "impedance_phase_deg": "deg",
            },
            "metadata": {
                **_protocol_metadata(numerical_protocol),
                "actual_intervals": len(grid) - 1,
                "actual_nodes": len(grid),
                "contact_thermodynamics": dataclasses.asdict(contact),
                "dc_numerically_certified": dc_result.numerically_certified,
                "dc_protocol_hash": dc_result.protocol_hash,
                "dc_state_hash": ion_aware_dc_state_sha256(dc_result.y),
                "external_finite_difference_step_factor": (
                    point.tolerance_factor
                ),
                "frequency_evidence": frequency_evidence,
                "frequency_window": dataclasses.asdict(
                    impedance_result.frequency_window
                ),
                "grid_sha256": ion_aware_impedance_grid_sha256(grid),
                "impedance_numerically_certified": (
                    certificate.numerically_certified
                ),
                "impedance_protocol": impedance_protocol.to_dict(),
                "impedance_protocol_hash": impedance_result.protocol_hash,
                "perturbation_assessments": [
                    dataclasses.asdict(item)
                    for item in certificate.perturbation_assessments
                ],
                "raw_impedance_ohm_m2": _complex_metadata(impedance_result.Z),
                "thermodynamically_certified": (
                    certificate.thermodynamically_certified
                ),
            },
        }
    )


def run_csi_qf_frequency_domain(
    lane: LaneDefinition,
    point: MatrixPoint,
    project_root: Path,
) -> CellMeasurement:
    """Run the registered c-Si QF C-V and finite-difference-step ladder."""
    options = lane.options
    stack = _load_stack(lane, project_root)
    grid = build_electrical_grid(stack, point.grid)
    require_thick_layer_interface_resolution(
        grid,
        stack,
        N_grid=point.grid,
    )
    material = build_material_arrays(grid, stack)
    biases = _numeric_array_option(
        options,
        "biases_V",
        (-0.3, -0.2, -0.1, 0.0, 0.1, 0.2),
    )
    frequencies = _numeric_array_option(
        options,
        "frequencies_Hz",
        (1.0e4, 1.0e5, 1.0e6),
    )
    delta_v = _option(options, "nominal_delta_V", float, 0.01)
    base_state_step = _option(options, "base_state_difference_step", float, 1.0e-5)
    base_voltage_step = _option(
        options,
        "base_voltage_difference_step_V",
        float,
        1.0e-5,
    )
    eps_r = _option(options, "mott_schottky_eps_r", float, 11.7)
    temperature = _option(options, "temperature_K", float, 300.0)
    if not np.isclose(float(stack.T), temperature, rtol=0.0, atol=1.0e-12):
        raise ValueError(
            "Mott-Schottky fit temperature must match the experiment protocol"
        )
    experiment_protocols = tuple(
        build_impedance_experiment_protocol(
            stack,
            frequencies,
            V_dc=float(voltage),
            delta_V=delta_v,
            illuminated=False,
            method="qf_frequency_ion_free",
            implicit_legacy_protocol=False,
        )
        for voltage in biases
    )
    capacitance = np.empty((frequencies.size, biases.size), dtype=float)
    spreads = np.empty_like(capacitance)
    backward_errors = np.empty_like(capacitance)
    reciprocal_conditions = np.empty_like(capacitance)
    dc_certified: list[bool] = []
    dc_residuals: list[float] = []

    for column, voltage in enumerate(biases):
        response = run_quasi_fermi_impedance(
            grid,
            stack,
            frequencies,
            V_dc=float(voltage),
            delta_V=delta_v,
            illuminated=False,
            mat=material,
            state_step=base_state_step * point.tolerance_factor,
            voltage_step=base_voltage_step * point.tolerance_factor,
        )
        admittance = 1.0 / response.Z
        capacitance_column = admittance.imag / (2.0 * np.pi * frequencies)
        if not np.all(np.isfinite(capacitance_column)) or np.any(
            capacitance_column <= 0.0
        ):
            raise ValueError(
                "cSi QF capacitance must be finite and positive at "
                f"V_dc={float(voltage):.6g} V"
            )
        capacitance[:, column] = capacitance_column
        spreads[:, column] = response.max_relative_face_spread
        backward_errors[:, column] = response.backward_error
        reciprocal_conditions[:, column] = response.reciprocal_condition
        dc_certified.append(bool(response.dc_state.certified))
        dc_residuals.append(response.dc_state.max_normalized_cell_residual)

    central_index = int(np.argmin(np.abs(frequencies - 1.0e5)))
    central_curve = capacitance[central_index]
    intercept, effective_doping, fit_lo, fit_hi = _fit_mott_schottky(
        biases,
        central_curve,
        eps_r=eps_r,
        T=temperature,
    )
    return CellMeasurement.from_mapping(
        {
            "observables": {
                "cv_curve_F_m2": central_curve,
                "effective_doping_m3": effective_doping,
                "mott_intercept_V": intercept,
            },
            "quality": {
                "dc_residual_certified": float(all(dc_certified)),
                "max_backward_error": float(np.max(backward_errors)),
                "max_relative_face_spread": float(np.max(spreads)),
            },
            "units": {
                "cv_curve_F_m2": "F m-2",
                "effective_doping_m3": "m-3",
                "mott_intercept_V": "V",
            },
            "metadata": {
                **_protocol_metadata(
                    {
                        "adapter": "csi-qf-frequency-bias-ladder",
                        "bias_order_V": biases.tolist(),
                        "experiments": [
                            protocol.to_dict() for protocol in experiment_protocols
                        ],
                        "schema_version": ("numerical-refinement-protocol-bundle-v1"),
                    }
                ),
                "actual_intervals": len(grid) - 1,
                "biases_V": biases.tolist(),
                "capacitance_all_frequencies_F_m2": capacitance.tolist(),
                "dc_max_normalized_cell_residual": dc_residuals,
                "finite_difference_state_step": (
                    base_state_step * point.tolerance_factor
                ),
                "finite_difference_voltage_step_V": (
                    base_voltage_step * point.tolerance_factor
                ),
                "fit_window_V": [fit_lo, fit_hi],
                "frequencies_Hz": frequencies.tolist(),
                "minimum_reciprocal_condition": float(np.min(reciprocal_conditions)),
            },
        }
    )


def _freeze_ions(stack):
    return replace(
        stack,
        layers=tuple(
            replace(layer, params=replace(layer.params, D_ion=0.0, D_ion_neg=0.0))
            for layer in stack.layers
        ),
    )


def _one_d_conduction_current_from_snapshots(result, stack) -> np.ndarray:
    """Return contact-face Jn + Jp from the saved 1-D forward snapshots."""
    voltages = np.asarray(result.V_fwd, dtype=float)
    snapshots = result.snapshots_fwd
    if snapshots is None or len(snapshots) != voltages.size:
        raise RuntimeError(
            "1-D uniform-limit comparison requires one saved snapshot per voltage"
        )
    if not snapshots:
        raise RuntimeError("1-D uniform-limit comparison received no snapshots")

    grid = np.asarray(snapshots[0].x, dtype=float)
    if grid.ndim != 1 or grid.size < 2 or not np.all(np.isfinite(grid)):
        raise RuntimeError("1-D uniform-limit snapshots carry an invalid grid")
    material = build_material_arrays(grid, stack)
    currents: list[float] = []
    for voltage, snapshot in zip(voltages, snapshots, strict=True):
        snapshot_grid = np.asarray(snapshot.x, dtype=float)
        if snapshot_grid.shape != grid.shape or not np.array_equal(snapshot_grid, grid):
            raise RuntimeError("1-D forward snapshots do not share one fixed grid")
        if not np.isclose(
            float(snapshot.V_app),
            float(voltage),
            rtol=0.0,
            atol=1.0e-14,
        ):
            raise RuntimeError("1-D snapshot voltage does not match the J-V ladder")
        state = StateVec.pack(
            np.asarray(snapshot.n, dtype=float),
            np.asarray(snapshot.p, dtype=float),
            np.asarray(snapshot.P, dtype=float),
        )
        components = compute_current_components(
            grid,
            state,
            stack,
            float(voltage),
            mat=material,
        )
        conduction_faces = np.asarray(components.J_n, dtype=float) + np.asarray(
            components.J_p,
            dtype=float,
        )
        if conduction_faces.ndim != 1 or conduction_faces.size == 0:
            raise RuntimeError("1-D carrier-conduction current has no contact face")
        terminal_current = float(conduction_faces[0])
        if not np.isfinite(terminal_current):
            raise RuntimeError("1-D carrier-conduction current is non-finite")
        currents.append(terminal_current)
    return np.asarray(currents, dtype=float)


def _normalized_current_parity(
    two_d_current: np.ndarray,
    one_d_current: np.ndarray,
    *,
    scale: float,
) -> tuple[np.ndarray, float]:
    """Return the signed normalized trace error and its absolute envelope."""
    two_d = np.asarray(two_d_current, dtype=float)
    one_d = np.asarray(one_d_current, dtype=float)
    if two_d.shape != one_d.shape or two_d.ndim != 1 or two_d.size == 0:
        raise ValueError("1-D and 2-D current traces must be nonempty and shape-matched")
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("current parity scale must be finite and positive")
    difference = (two_d - one_d) / scale
    if not np.all(np.isfinite(difference)):
        raise ValueError("normalized 1-D/2-D current parity is non-finite")
    return difference, float(np.max(np.abs(difference)))


def _max_lateral_carrier_variation_relative(snapshot) -> float:
    """Return the worst independently normalized lateral variation of n or p."""

    def relative_variation(field: np.ndarray) -> float:
        values = np.asarray(field, dtype=float)
        if values.ndim != 2 or values.size == 0 or not np.all(np.isfinite(values)):
            raise ValueError("2-D carrier snapshot must be a finite nonempty matrix")
        reference = values[:, [0]]
        return float(np.max(np.abs(values - reference))) / max(
            float(np.max(np.abs(reference))),
            1.0e-30,
        )

    return max(relative_variation(snapshot.n), relative_variation(snapshot.p))


def _settle_2d_with_tolerance(
    state: np.ndarray,
    material,
    *,
    voltage: float,
    settle_time: float,
    rtol: float,
    atol: ComponentwiseAtol,
    max_nfev: int,
    max_bisect: int,
) -> np.ndarray:
    try:
        return run_transient_2d(
            state,
            material,
            V_app=voltage,
            t_end=settle_time,
            max_step=settle_time / 50.0,
            rtol=rtol,
            atol=atol,
            max_nfev=max_nfev,
        )
    except RuntimeError:
        if max_bisect == 0:
            raise
    half = 0.5 * settle_time
    midpoint = _settle_2d_with_tolerance(
        state,
        material,
        voltage=voltage,
        settle_time=half,
        rtol=rtol,
        atol=atol,
        max_nfev=max_nfev,
        max_bisect=max_bisect - 1,
    )
    return _settle_2d_with_tolerance(
        midpoint,
        material,
        voltage=voltage,
        settle_time=half,
        rtol=rtol,
        atol=atol,
        max_nfev=max_nfev,
        max_bisect=max_bisect - 1,
    )


def _poisson_relative_residual(snapshot, material) -> float:
    factor = material.poisson_factor
    phi = np.asarray(snapshot.phi, dtype=float)
    rho = _charge_density_2d(snapshot.n, snapshot.p, material)
    grid = factor.grid
    dx = np.diff(grid.x)
    dy = np.diff(grid.y)
    hx = np.empty(grid.Nx)
    hx[0] = dx[0] / 2.0
    hx[-1] = dx[-1] / 2.0
    hx[1:-1] = 0.5 * (dx[:-1] + dx[1:])
    if factor.lateral_bc == "periodic":
        hx[0] = hx[-1] = 0.5 * (dx[0] + dx[-1])
    worst = 0.0
    for j in range(1, grid.Ny - 1):
        hy = 0.5 * (dy[j - 1] + dy[j])
        for i in range(grid.Nx):
            centre = phi[j, i]
            terms = [
                factor.C_y[j - 1, i] * hx[i] * (phi[j - 1, i] - centre),
                factor.C_y[j, i] * hx[i] * (phi[j + 1, i] - centre),
            ]
            if factor.lateral_bc == "periodic":
                left = (i - 1) % grid.Nx
                right = (i + 1) % grid.Nx
                terms.extend(
                    (
                        factor.C_x[j, left] * hy * (phi[j, left] - centre),
                        factor.C_x[j, i] * hy * (phi[j, right] - centre),
                    )
                )
            else:
                if i > 0:
                    terms.append(factor.C_x[j, i - 1] * hy * (phi[j, i - 1] - centre))
                if i < grid.Nx - 1:
                    terms.append(factor.C_x[j, i] * hy * (phi[j, i + 1] - centre))
            charge = rho[j, i] * factor.cell_area[j - 1, i]
            residual = sum(terms) + charge
            scale = max(sum(abs(term) for term in terms) + abs(charge), 1.0e-30)
            worst = max(worst, abs(float(residual)) / scale)
    return worst


def run_twod_uniform_limit(
    lane: LaneDefinition,
    point: MatrixPoint,
    project_root: Path,
) -> CellMeasurement:
    """Run matched frozen-ion 1-D/2-D J-V grids with one tolerance policy."""
    options = lane.options
    stack = _freeze_ions(_load_stack(lane, project_root))
    electrical = electrical_layers(stack)
    base_vertical = _option(options, "base_vertical_intervals_per_layer", int, 5)
    base_lateral = _option(options, "base_lateral_intervals", int, 4)
    if base_vertical < 1 or base_lateral < 1:
        raise ValueError("matched base x/y interval counts must be positive")
    refinement = point.grid
    ny_per_layer = base_vertical * refinement
    nx_intervals = base_lateral * refinement
    one_d_n_grid = 1 + ny_per_layer * len(electrical)
    lateral_length = _option(options, "lateral_length_m", float, 500.0e-9)
    voltage_max = _option(options, "V_max_V", float, 1.2)
    voltage_step = _option(options, "V_step_V", float, 0.1)
    settle_time = _option(options, "settle_time_s", float, 1.0e-3)
    rtol = _option(options, "rtol", float, 1.0e-6)
    max_nfev = _option(options, "max_nfev", int, 200_000)
    max_bisect = _option(options, "max_bisect", int, 6)
    policy = _componentwise_policy(options, point.tolerance_factor)
    voltages = np.arange(0.0, voltage_max + voltage_step / 2.0, voltage_step)
    matched_v_rate = voltage_step / settle_time
    one_d_protocol = build_jv_experiment_protocol(
        stack,
        v_rate=matched_v_rate,
        n_points=len(voltages),
        V_max=voltage_max,
        illuminated=True,
        implicit_legacy_protocol=False,
    )
    two_d_protocol = {
        "adapter": "uniform-2d-ascending-jv",
        "illumination": "baseline",
        "initial_state": {
            "pre_bias_V": 0.0,
            "source": "one_dimensional_illuminated_seed_broadcast_laterally",
            "preconditioning_time_s": 1.0e-3,
        },
        "measurement": "finite_time_jv_ladder",
        "ordering": "ascending",
        "sampling_voltage_V": voltages.tolist(),
        "schema_version": "numerical-refinement-execution-protocol-v1",
        "settle_time_per_voltage_s": settle_time,
    }
    protocol_bundle = {
        "comparison": "two_d_ascending_against_one_d_forward_carrier_conduction",
        "current_composition": {
            "compared": "electron_plus_hole_conduction",
            "excluded": ["ionic", "displacement"],
            "one_d_source": "saved_forward_snapshots_contact_face",
            "two_d_source": "terminal_contact_carrier_flux",
        },
        "one_d": {
            "experiment_protocol": one_d_protocol.to_dict(),
            "selected_branch": "forward",
        },
        "schema_version": "numerical-refinement-protocol-bundle-v1",
        "two_d": two_d_protocol,
    }

    one_d = run_jv_sweep(
        stack,
        N_grid=one_d_n_grid,
        v_rate=matched_v_rate,
        n_points=len(voltages),
        rtol=rtol,
        atol=policy,
        V_max=voltage_max,
        illuminated=True,
        certification_mode="strict",
        experiment_protocol=one_d_protocol,
        protocol_mode="research_strict",
        save_snapshots=True,
    )
    resolved_one_d_protocol = getattr(one_d, "protocol", one_d_protocol)
    if (
        resolved_one_d_protocol is None
        or resolved_one_d_protocol.protocol_hash != one_d_protocol.protocol_hash
    ):
        raise RuntimeError("1-D J-V returned a different experiment protocol")
    protocol_bundle["one_d"]["experiment_protocol"] = resolved_one_d_protocol.to_dict()
    one_d_voltage = np.asarray(one_d.V_fwd, dtype=float)
    if not np.allclose(one_d_voltage, voltages, rtol=0.0, atol=1.0e-14):
        raise RuntimeError("1-D and 2-D voltage grids are not identical")
    one_d_current = _one_d_conduction_current_from_snapshots(one_d, stack)
    one_d_metrics = compute_metrics(one_d_voltage, one_d_current)

    grid = build_grid_2d(
        [Layer(layer.thickness, ny_per_layer) for layer in electrical],
        lateral_length=lateral_length,
        Nx=nx_intervals,
        lateral_uniform=True,
    )
    seed_1d = solve_illuminated_ss(
        grid.y,
        stack,
        V_app=0.0,
        t_settle=1.0e-3,
        rtol=rtol,
        atol=policy,
    )
    seed_state = StateVec.unpack(seed_1d, len(grid.y))
    material = build_material_arrays_2d(
        grid,
        stack,
        Microstructure(),
        lateral_bc=_option(options, "lateral_boundary", str, "periodic"),
        P_ion_static_1d=seed_state.P,
    )
    if material.has_radiative_reabsorption_2d:
        raise RuntimeError(
            "the Phase-1 uniform adapter does not silently change the "
            "radiative-reabsorption recovery protocol"
        )
    state = np.concatenate(
        (
            np.broadcast_to(seed_state.n[:, None], (grid.Ny, grid.Nx)).ravel(),
            np.broadcast_to(seed_state.p[:, None], (grid.Ny, grid.Nx)).ravel(),
        )
    )
    snapshots = []
    currents = []
    for voltage in voltages:
        state = _settle_2d_with_tolerance(
            state,
            material,
            voltage=float(voltage),
            settle_time=settle_time,
            rtol=rtol,
            atol=policy,
            max_nfev=max_nfev,
            max_bisect=max_bisect,
        )
        snapshot = extract_snapshot_2d(state, material, V_app=float(voltage))
        snapshots.append(snapshot)
        currents.append(compute_terminal_current_2d(snapshot))
    two_d_current_raw = np.asarray(currents, dtype=float)
    two_d_current = (
        -two_d_current_raw if two_d_current_raw[0] < 0.0 else two_d_current_raw
    )
    metrics = compute_metrics(voltages, two_d_current)
    current_scale = max(abs(one_d_metrics.J_sc), 1.0e-30)
    trace_difference, max_abs_parity = _normalized_current_parity(
        two_d_current,
        one_d_current,
        scale=current_scale,
    )

    poisson_residuals = [
        _poisson_relative_residual(snapshot, material) for snapshot in snapshots
    ]
    current_spreads = []
    lateral_variations = []
    for snapshot in snapshots:
        vertical = np.mean(snapshot.Jy_n + snapshot.Jy_p, axis=1)
        scale = max(float(np.max(np.abs(vertical))), current_scale, 1.0e-30)
        current_spreads.append(float(np.ptp(vertical)) / scale)
        lateral_variations.append(_max_lateral_carrier_variation_relative(snapshot))
    return CellMeasurement.from_mapping(
        {
            "observables": {
                "jsc_2d_A_m2": metrics.J_sc,
                "jv_2d_to_1d_normalized_difference": trace_difference,
                "voc_2d_V": metrics.V_oc,
            },
            "quality": {
                "frozen_ion_scope_declared": 1.0,
                "max_abs_2d_to_1d_normalized_difference": max_abs_parity,
                "max_charge_residual_relative": max(poisson_residuals),
                "max_current_conservation_relative": max(current_spreads),
                "max_lateral_carrier_variation_relative": max(lateral_variations),
                "one_d_jv_certified": float(one_d.certified),
                "voc_bracketed": float(metrics.voc_bracketed),
            },
            "units": {
                "jsc_2d_A_m2": "A m-2",
                "voc_2d_V": "V",
            },
            "metadata": {
                **_protocol_metadata(protocol_bundle),
                "Nx_intervals": nx_intervals,
                "Ny_intervals": len(grid.y) - 1,
                "matched_refinement_multiplier": refinement,
                "one_d_intervals": one_d_n_grid - 1,
                "one_d_certified": one_d.certified,
                "voltage_grid_V": voltages.tolist(),
            },
        }
    )


def _interface_charge_off_protocol(
    voltages: np.ndarray,
    *,
    base_finite_difference_step: float,
    base_newton_residual_tolerance: float,
    max_newton_iterations: int,
    base_poisson_tolerance_V: float,
    poisson_max_iterations: int,
    continuity_tolerance_A_m2: float,
    current_spread_tolerance_A_m2: float,
    poisson_residual_tolerance: float,
    interface_qss_residual_tolerance: float,
) -> dict[str, Any]:
    """Describe the charge-off QF reference without embedding a matrix rung."""
    return {
        "acceptance": {
            "continuity_tolerance_A_m2": continuity_tolerance_A_m2,
            "current_spread_tolerance_A_m2": current_spread_tolerance_A_m2,
            "interface_qss_residual_tolerance": (interface_qss_residual_tolerance),
            "poisson_residual_tolerance": poisson_residual_tolerance,
            "require_contact_thermodynamic_certificate": True,
            "require_voc_bracket": True,
        },
        "adapter": "interface-charge-off-two-sided-qf-jv",
        "continuation": {
            "first_point": "dark_then_registered_illumination_ladder",
            "illumination_steps": list(DEFAULT_ILLUMINATION_STEPS),
            "subsequent_points": "previous_certified_qf_state",
            "voltage_bridge": "disabled_fail_closed",
        },
        "dark_reference": {
            "illumination": "dark",
            "occupancy": "shared_two_sided_interface_trap",
            "voltage_V": 0.0,
        },
        "interface": {
            "charge_closure": "off",
            "cross_transmission": 1.0,
            "rebaseline_acknowledged": True,
            "topology": TWO_SIDED_TRACE,
            "transport_model": FERMI_DIRAC_RICHARDSON,
        },
        "measurement": "steady_state_jv_and_interface_capture_flux",
        "sampling_voltage_V": voltages.tolist(),
        "schema_version": "interface-charge-off-reference-protocol-v1",
        "solver": {
            "base_finite_difference_step": base_finite_difference_step,
            "base_newton_residual_tolerance": (base_newton_residual_tolerance),
            "base_poisson_tolerance_V": base_poisson_tolerance_V,
            "max_newton_iterations": max_newton_iterations,
            "poisson_max_iterations": poisson_max_iterations,
            "refinement_factor_source": "matrix.tolerance_factor",
            "refinement_mapping": {
                "finite_difference_step": "base*sqrt(factor)",
                "newton_residual_tolerance": "base*factor",
                "poisson_tolerance_V": "base*factor",
            },
            "type": "quasi_fermi_residual_certified",
        },
    }


def _two_sided_interface_evidence(
    result: Any,
    *,
    interface_count: int,
) -> tuple[list[float], float, float, np.ndarray]:
    """Extract signed recombination and independent local balance defects."""
    capture = np.asarray(result.capture_flux_m2_s, dtype=float)
    residual = np.asarray(result.state_flux_m2_s, dtype=float)
    occupancy = np.asarray(result.occupancy, dtype=float)
    expected = 4 * interface_count
    if capture.shape != (expected,) or residual.shape != (expected,):
        raise RuntimeError("two-sided QSS result has an invalid trace layout")
    if occupancy.shape != (interface_count,):
        raise RuntimeError("two-sided QSS result has an invalid occupancy layout")
    if not all(
        np.all(np.isfinite(values)) for values in (capture, residual, occupancy)
    ):
        raise RuntimeError("two-sided QSS evidence contains non-finite values")
    if np.any((occupancy < 0.0) | (occupancy > 1.0)):
        raise RuntimeError("two-sided QSS occupancy left [0, 1]")

    fluxes: list[float] = []
    maximum_carrier_balance = 0.0
    for index in range(interface_count):
        base = 4 * index
        electron_capture = float(capture[base] + capture[base + 2])
        hole_capture = float(capture[base + 1] + capture[base + 3])
        fluxes.append(Q * electron_capture)
        maximum_carrier_balance = max(
            maximum_carrier_balance,
            Q * abs(electron_capture - hole_capture),
        )
    maximum_state_residual = float(Q * np.max(np.abs(residual)))
    return fluxes, maximum_carrier_balance, maximum_state_residual, occupancy


def run_interface_recombination_charge_off(
    lane: LaneDefinition,
    point: MatrixPoint,
    project_root: Path,
) -> CellMeasurement:
    """Certify the uncalibrated two-sided QF reference with trap charge off."""
    options = lane.options
    stack = _load_stack(lane, project_root)
    if stack.interface_charge_closure != "off":
        raise RuntimeError("charge-off reference requires charge_closure='off'")
    if not stack.interface_charge_rebaseline_acknowledged:
        raise RuntimeError("charge-off reference requires explicit rebaseline intent")
    if stack.het_recomb_despike != 0.0:
        raise RuntimeError("charge-off reference forbids recombination de-spiking")
    if stack.flat_band_contacts or stack.flat_band_metal_contacts:
        raise RuntimeError("charge-off reference forbids calibrated contact floors")
    if stack.contact_phi_B_eV != 0.0:
        raise RuntimeError("charge-off reference forbids a calibrated contact barrier")
    defects = tuple(defect for defect in stack.interface_defects if defect is not None)
    if not defects:
        raise RuntimeError("charge-off reference resolved no interface defects")
    if any(
        defect.calibration_factor != 1.0 or defect.iface_state_calibration_factor != 1.0
        for defect in defects
    ):
        raise RuntimeError("charge-off reference requires unity calibration factors")

    shared_grid = build_electrical_grid(stack, point.grid)
    grid = build_two_sided_trace_grid(shared_grid, stack)
    base_material = build_material_arrays(grid, stack)
    if base_material.iface_state_charge != 0.0:
        raise RuntimeError(
            "interface electrostatic trap charge must remain disabled in this lane"
        )
    contact_certificate = require_contact_thermodynamic_certificate(
        stack,
        base_material,
    )
    material = _prepare_two_sided_material(grid, stack, base_material)
    interface_count = len(material.iface_qss_left_nodes)
    if interface_count == 0 or interface_count != len(defects):
        raise RuntimeError("two-sided topology is not aligned with interface defects")

    voltage_max = _option(options, "V_max_V", float, 1.2)
    voltage_points = _option(options, "voltage_points", int, 25)
    base_fd_step = _option(
        options,
        "base_finite_difference_step",
        float,
        1.0e-5,
    )
    base_newton_tol = _option(
        options,
        "base_newton_residual_tolerance",
        float,
        4.0e-7,
    )
    max_newton = _option(options, "max_newton_iterations", int, 60)
    base_poisson_tol = _option(
        options,
        "base_poisson_tolerance_V",
        float,
        1.0e-12,
    )
    poisson_max = _option(options, "poisson_max_iterations", int, 100)
    continuity_tol = _option(
        options,
        "continuity_tolerance_A_m2",
        float,
        1.0e-4,
    )
    spread_tol = _option(
        options,
        "current_spread_tolerance_A_m2",
        float,
        1.0e-4,
    )
    poisson_residual_tol = _option(
        options,
        "poisson_residual_tolerance",
        float,
        1.0e-8,
    )
    interface_qss_tol = _option(
        options,
        "interface_qss_residual_tolerance",
        float,
        1.0e-7,
    )
    voltages = np.linspace(0.0, voltage_max, voltage_points)
    numerical_protocol = _interface_charge_off_protocol(
        voltages,
        base_finite_difference_step=base_fd_step,
        base_newton_residual_tolerance=base_newton_tol,
        max_newton_iterations=max_newton,
        base_poisson_tolerance_V=base_poisson_tol,
        poisson_max_iterations=poisson_max,
        continuity_tolerance_A_m2=continuity_tol,
        current_spread_tolerance_A_m2=spread_tol,
        poisson_residual_tolerance=poisson_residual_tol,
        interface_qss_residual_tolerance=interface_qss_tol,
    )
    factor = point.tolerance_factor
    solve_controls = {
        "interface_boundary": True,
        "interface_topology": TWO_SIDED_TRACE,
        "interface_transmission": 1.0,
        "interface_transport_model": FERMI_DIRAC_RICHARDSON,
        "finite_difference_step": base_fd_step * np.sqrt(factor),
        "newton_residual_tolerance": base_newton_tol * factor,
        "max_newton_iterations": max_newton,
        "poisson_tolerance_V": base_poisson_tol * factor,
        "poisson_max_iterations": poisson_max,
        "continuity_tolerance_A_m2": continuity_tol,
        "current_spread_tolerance_A_m2": spread_tol,
        "poisson_residual_tolerance": poisson_residual_tol,
    }

    dark = solve_quasi_fermi_steady_state(
        grid,
        stack,
        0.0,
        illuminated=False,
        mat=base_material,
        **solve_controls,
    )
    dark_qss = solve_material_two_sided_interfaces_qss(
        material,
        stack,
        dark.y[: len(grid)],
        dark.y[len(grid) : 2 * len(grid)],
        dark.phi,
        cross_transmission=1.0,
        interface_transport_model=FERMI_DIRAC_RICHARDSON,
        residual_tolerance=interface_qss_tol,
        fail_on_residual=True,
    )
    dark_flux, dark_balance, dark_state_residual, f_eq = _two_sided_interface_evidence(
        dark_qss,
        interface_count=interface_count,
    )
    if not dark.certified:
        raise RuntimeError("dark reference lacks a QF physical certificate")

    sweep = solve_quasi_fermi_jv_sweep(
        grid,
        stack,
        voltages,
        mat=base_material,
        **solve_controls,
    )
    if len(sweep.points) != len(voltages):
        raise RuntimeError("QF sweep did not retain every registered voltage")

    interface_fluxes: list[float] = []
    carrier_balance_defects = [dark_balance]
    state_residuals = [dark_state_residual]
    occupancies = [f_eq]
    local_residuals = [dark.interface_local_residual]
    for result in sweep.points:
        if result.interface_topology != TWO_SIDED_TRACE:
            raise RuntimeError("QF point did not retain two-sided topology")
        qss = solve_material_two_sided_interfaces_qss(
            material,
            stack,
            result.y[: len(grid)],
            result.y[len(grid) : 2 * len(grid)],
            result.phi,
            cross_transmission=1.0,
            interface_transport_model=FERMI_DIRAC_RICHARDSON,
            residual_tolerance=interface_qss_tol,
            fail_on_residual=True,
        )
        flux, balance, state_residual, occupancy = _two_sided_interface_evidence(
            qss, interface_count=interface_count
        )
        interface_fluxes.extend(flux)
        carrier_balance_defects.append(balance)
        state_residuals.append(state_residual)
        occupancies.append(occupancy)
        local_residuals.append(result.interface_local_residual)

    current_array = np.asarray(sweep.currents_A_m2, dtype=float)
    current_scale = max(abs(sweep.metrics.J_sc), 1.0e-30)
    all_points = (dark, *sweep.points)
    occupancy_array = np.concatenate(occupancies)
    calibration_unity = all(
        defect.calibration_factor == 1.0
        and defect.iface_state_calibration_factor == 1.0
        for defect in defects
    )
    return CellMeasurement.from_mapping(
        {
            "observables": {
                "interface_flux_A_m2": interface_fluxes,
                "jv_normalized": current_array / current_scale,
                "voc_V": sweep.metrics.V_oc,
            },
            "quality": {
                "all_points_certified": float(dark.certified and sweep.certified),
                "calibration_factors_unity": float(calibration_unity),
                "contact_thermodynamics_certified": float(
                    contact_certificate.certified
                ),
                "dark_reference_certified": float(dark.certified),
                "max_continuity_bound_A_m2": max(
                    max(
                        result.electron_continuity_bound_A_m2,
                        result.hole_continuity_bound_A_m2,
                    )
                    for result in all_points
                ),
                "max_current_spread_A_m2": max(
                    result.face_current_spread_A_m2 for result in all_points
                ),
                "max_interface_carrier_balance_A_m2": max(carrier_balance_defects),
                "max_interface_local_residual": max(local_residuals),
                "max_interface_state_residual_A_m2": max(state_residuals),
                "max_normalized_cell_residual": max(
                    result.max_normalized_cell_residual for result in all_points
                ),
                "max_poisson_residual": max(
                    result.poisson_residual for result in all_points
                ),
                "occupancy_bounded": float(
                    np.all((occupancy_array >= 0.0) & (occupancy_array <= 1.0))
                ),
                "rebaseline_acknowledged": float(
                    stack.interface_charge_rebaseline_acknowledged
                ),
                "trap_electrostatic_charge_enabled": float(
                    base_material.iface_state_charge != 0.0
                ),
                "two_sided_topology_active": float(
                    dark.interface_topology == TWO_SIDED_TRACE
                    and all(
                        result.interface_topology == TWO_SIDED_TRACE
                        for result in sweep.points
                    )
                ),
                "voc_bracketed": float(sweep.metrics.voc_bracketed),
            },
            "units": {
                "interface_flux_A_m2": "A m-2",
                "voc_V": "V",
                "max_continuity_bound_A_m2": "A m-2",
                "max_current_spread_A_m2": "A m-2",
                "max_interface_carrier_balance_A_m2": "A m-2",
                "max_interface_state_residual_A_m2": "A m-2",
            },
            "metadata": {
                **_protocol_metadata(numerical_protocol),
                "actual_intervals": len(grid) - 1,
                "contact_thermodynamics": dataclasses.asdict(contact_certificate),
                "dark_interface_flux_A_m2": dark_flux,
                "dark_reference_occupancy": f_eq.tolist(),
                "dark_reference_state_sha256": content_sha256(
                    {
                        "occupancy": f_eq.tolist(),
                        "phi_V": dark.phi.tolist(),
                        "state": dark.y.tolist(),
                    }
                ),
                "interface_count": interface_count,
                "interface_flux_layout": "voltage_major_interface_minor",
                "source_grid_intervals": point.grid,
                "tolerance_controls": {
                    "finite_difference_step": solve_controls["finite_difference_step"],
                    "newton_residual_tolerance": solve_controls[
                        "newton_residual_tolerance"
                    ],
                    "poisson_tolerance_V": solve_controls["poisson_tolerance_V"],
                },
                "voltage_grid_V": voltages.tolist(),
            },
        }
    )


def _interface_charge_research_protocol(
    *,
    bias_voltage_V: float,
    illuminated_voltage_V: float,
    base_finite_difference_step: float,
    base_newton_residual_tolerance: float,
    max_newton_iterations: int,
    base_poisson_tolerance_V: float,
    poisson_max_iterations: int,
    continuity_tolerance_A_m2: float,
    current_spread_tolerance_A_m2: float,
    poisson_residual_tolerance: float,
) -> dict[str, Any]:
    """Describe the charged matrix without embedding its grid or tolerance rung."""
    return {
        "acceptance": {
            "continuity_tolerance_A_m2": continuity_tolerance_A_m2,
            "current_spread_tolerance_A_m2": current_spread_tolerance_A_m2,
            "local_interface_residual_limit": 1.0e-7,
            "normalized_gauss_residual_limit": 1.0e-10,
            "poisson_residual_tolerance": poisson_residual_tolerance,
            "require_contact_thermodynamic_certificate": True,
            "require_dark_charge_off_bit_identity": True,
        },
        "adapter": "equilibrium-referenced-interface-charge-two-sided-qf",
        "dark_reference": {
            "charge_closure": "off",
            "illumination": "dark",
            "occupancy": "shared_two_sided_interface_trap",
            "voltage_V": 0.0,
        },
        "interface": {
            "charge_closure": "equilibrium_referenced",
            "charge_law": "-q*N_t*(f-f_eq)",
            "cross_transmission": 1.0,
            "topology": TWO_SIDED_TRACE,
            "transport_model": FERMI_DIRAC_RICHARDSON,
        },
        "measurement": "charged_steady_state_bias_and_light",
        "schema_version": "interface-charge-research-protocol-v1",
        "solver": {
            "base_finite_difference_step": base_finite_difference_step,
            "base_newton_residual_tolerance": base_newton_residual_tolerance,
            "base_poisson_tolerance_V": base_poisson_tolerance_V,
            "illumination_steps": list(DEFAULT_ILLUMINATION_STEPS),
            "max_newton_iterations": max_newton_iterations,
            "poisson_max_iterations": poisson_max_iterations,
            "refinement_factor_source": "matrix.tolerance_factor",
            "refinement_mapping": {
                "finite_difference_step": "base*sqrt(factor)",
                "newton_residual_tolerance": "base*factor",
                "poisson_tolerance_V": "base*factor",
            },
            "type": "quasi_fermi_residual_certified_with_local_ift",
        },
        "targets": [
            {
                "illuminated": False,
                "label": "dark_bias",
                "voltage_V": bias_voltage_V,
            },
            {
                "illuminated": True,
                "label": "illuminated_operating_point",
                "voltage_V": illuminated_voltage_V,
            },
        ],
    }


def _dark_charge_reference_arrays_are_bit_identical(reference, charged_dark) -> bool:
    fields = (
        "y",
        "phi",
        "electron_quasi_fermi_potential_V",
        "hole_quasi_fermi_potential_V",
        "electron_face_current_A_m2",
        "hole_face_current_A_m2",
        "total_face_current_A_m2",
        "electron_rate_per_s",
        "hole_rate_per_s",
    )
    return all(
        np.array_equal(getattr(reference.dark_state, name), getattr(charged_dark, name))
        for name in fields
    )


def run_equilibrium_referenced_interface_charge(
    lane: LaneDefinition,
    point: MatrixPoint,
    project_root: Path,
) -> CellMeasurement:
    """Certify self-consistent equilibrium-referenced interface charge."""
    options = lane.options
    stack = _load_stack(lane, project_root)
    if stack.interface_charge_closure != "equilibrium_referenced":
        raise RuntimeError(
            "charged reference requires charge_closure='equilibrium_referenced'"
        )
    if not stack.interface_charge_rebaseline_acknowledged:
        raise RuntimeError("charged reference requires explicit rebaseline intent")
    if stack.het_recomb_despike != 0.0:
        raise RuntimeError("charged reference forbids recombination de-spiking")
    if stack.flat_band_contacts or stack.flat_band_metal_contacts:
        raise RuntimeError("charged reference forbids calibrated contact floors")
    if stack.contact_phi_B_eV != 0.0:
        raise RuntimeError("charged reference forbids a calibrated contact barrier")
    defects = electrical_interface_defects(stack)
    if not defects or any(defect is None for defect in defects):
        raise RuntimeError(
            "charged reference requires one interface defect per physical interface"
        )
    if any(
        defect.N_t_cm2 <= 0.0
        or defect.calibration_factor != 1.0
        or defect.iface_state_calibration_factor != 1.0
        for defect in defects
        if defect is not None
    ):
        raise RuntimeError(
            "charged reference requires positive trap densities and unity calibration"
        )

    shared_grid = build_electrical_grid(stack, point.grid)
    grid = build_two_sided_trace_grid(shared_grid, stack)
    charge_off_stack = replace(stack, interface_charge_closure="off")
    base_material = build_material_arrays(grid, charge_off_stack)
    if base_material.iface_state_charge != 0.0:
        raise RuntimeError("legacy shared-node interface charge must remain zero")
    contact_certificate = require_contact_thermodynamic_certificate(
        charge_off_stack,
        base_material,
    )

    bias_voltage = _option(options, "bias_voltage_V", float, 0.05)
    illuminated_voltage = _option(
        options,
        "illuminated_voltage_V",
        float,
        0.0,
    )
    base_fd_step = _option(
        options,
        "base_finite_difference_step",
        float,
        1.0e-5,
    )
    base_newton_tol = _option(
        options,
        "base_newton_residual_tolerance",
        float,
        4.0e-7,
    )
    max_newton = _option(options, "max_newton_iterations", int, 60)
    base_poisson_tol = _option(
        options,
        "base_poisson_tolerance_V",
        float,
        1.0e-12,
    )
    poisson_max = _option(options, "poisson_max_iterations", int, 100)
    continuity_tol = _option(
        options,
        "continuity_tolerance_A_m2",
        float,
        1.0e-4,
    )
    spread_tol = _option(
        options,
        "current_spread_tolerance_A_m2",
        float,
        1.0e-4,
    )
    poisson_residual_tol = _option(
        options,
        "poisson_residual_tolerance",
        float,
        1.0e-8,
    )
    protocol = _interface_charge_research_protocol(
        bias_voltage_V=bias_voltage,
        illuminated_voltage_V=illuminated_voltage,
        base_finite_difference_step=base_fd_step,
        base_newton_residual_tolerance=base_newton_tol,
        max_newton_iterations=max_newton,
        base_poisson_tolerance_V=base_poisson_tol,
        poisson_max_iterations=poisson_max,
        continuity_tolerance_A_m2=continuity_tol,
        current_spread_tolerance_A_m2=spread_tol,
        poisson_residual_tolerance=poisson_residual_tol,
    )
    factor = point.tolerance_factor
    solve_controls = {
        "finite_difference_step": base_fd_step * np.sqrt(factor),
        "newton_residual_tolerance": base_newton_tol * factor,
        "max_newton_iterations": max_newton,
        "poisson_tolerance_V": base_poisson_tol * factor,
        "poisson_max_iterations": poisson_max,
        "continuity_tolerance_A_m2": continuity_tol,
        "current_spread_tolerance_A_m2": spread_tol,
        "poisson_residual_tolerance": poisson_residual_tol,
    }

    reference = build_equilibrium_referenced_interface_charge_dark_reference(
        grid,
        stack,
        interface_transmission=1.0,
        **solve_controls,
    )
    charged_dark = solve_equilibrium_referenced_interface_charge_steady_state(
        grid,
        stack,
        0.0,
        dark_reference=reference,
        illuminated=False,
        **solve_controls,
    )
    target_specs = (
        ("dark_bias", bias_voltage, False),
        ("illuminated_operating_point", illuminated_voltage, True),
    )
    target_results = tuple(
        solve_equilibrium_referenced_interface_charge_steady_state(
            grid,
            stack,
            voltage,
            dark_reference=reference,
            illuminated=illuminated,
            **solve_controls,
        )
        for _label, voltage, illuminated in target_specs
    )

    interface_count = len(reference.equilibrium_occupancy)
    if interface_count == 0 or interface_count != len(defects):
        raise RuntimeError("charged interface evidence is not defect-aligned")
    expected_trace_shape = (interface_count, 2)
    for result in target_results:
        shapes = (
            np.asarray(result.interface_equilibrium_occupancy).shape,
            np.asarray(result.interface_occupancy).shape,
            np.asarray(result.interface_incremental_sheet_charge_C_m2).shape,
            np.asarray(result.interface_normalized_gauss_residual).shape,
            np.asarray(result.interface_scaled_local_jacobian_condition).shape,
        )
        if any(shape != (interface_count,) for shape in shapes):
            raise RuntimeError("charged interface result has a misaligned vector")
        if np.asarray(result.interface_trace_potential_shift_V).shape != (
            expected_trace_shape
        ):
            raise RuntimeError("charged interface result has a misaligned trace shift")

    equilibrium = np.asarray(reference.equilibrium_occupancy, dtype=float)
    trap_density = np.asarray(reference.trap_density_m2, dtype=float)
    occupancy = np.asarray(
        [result.interface_occupancy for result in target_results],
        dtype=float,
    )
    sheet_charge = np.asarray(
        [result.interface_incremental_sheet_charge_C_m2 for result in target_results],
        dtype=float,
    )
    trace_shift = np.asarray(
        [result.interface_trace_potential_shift_V for result in target_results],
        dtype=float,
    )
    expected_charge = -Q * trap_density[np.newaxis, :] * (
        occupancy - equilibrium[np.newaxis, :]
    )
    charge_law_consistent = bool(
        np.allclose(sheet_charge, expected_charge, rtol=1.0e-12, atol=0.0)
    )
    charge_fraction = np.abs(sheet_charge) / (Q * trap_density[np.newaxis, :])
    all_states = (reference.dark_state, charged_dark, *target_results)
    all_occupancy = np.r_[equilibrium, occupancy.ravel()]
    dark_arrays_identical = _dark_charge_reference_arrays_are_bit_identical(
        reference,
        charged_dark,
    )
    dark_charge = np.asarray(
        charged_dark.interface_incremental_sheet_charge_C_m2,
        dtype=float,
    )
    dark_trace_shift = np.asarray(
        charged_dark.interface_trace_potential_shift_V,
        dtype=float,
    )
    interface_pairs = electrical_interfaces(stack)
    target_evidence = []
    for (label, voltage, illuminated), result in zip(
        target_specs,
        target_results,
    ):
        target_evidence.append(
            {
                "current_A_m2": result.current_A_m2,
                "equilibrium_occupancy": list(result.interface_equilibrium_occupancy),
                "illuminated": illuminated,
                "incremental_sheet_charge_C_m2": list(
                    result.interface_incremental_sheet_charge_C_m2
                ),
                "local_interface_residual": result.interface_local_residual,
                "normalized_gauss_residual": list(
                    result.interface_normalized_gauss_residual
                ),
                "occupancy": list(result.interface_occupancy),
                "scaled_local_jacobian_condition": list(
                    result.interface_scaled_local_jacobian_condition
                ),
                "state_sha256": content_sha256(
                    {
                        "phi_V": result.phi.tolist(),
                        "state": result.y.tolist(),
                    }
                ),
                "target": label,
                "trace_potential_shift_V": [
                    list(values)
                    for values in result.interface_trace_potential_shift_V
                ],
                "voltage_V": voltage,
            }
        )

    return CellMeasurement.from_mapping(
        {
            "observables": {
                "charged_current_density_A_m2": [
                    result.current_A_m2 for result in target_results
                ],
                "interface_occupancy": occupancy.ravel(),
                "interface_sheet_charge_C_m2": sheet_charge.ravel(),
                "interface_trace_potential_shift_V": trace_shift.ravel(),
            },
            "quality": {
                "all_points_certified": float(
                    all(result.certified for result in all_states)
                ),
                "calibration_factors_unity": float(
                    all(
                        defect is not None
                        and defect.calibration_factor == 1.0
                        and defect.iface_state_calibration_factor == 1.0
                        for defect in defects
                    )
                ),
                "charge_law_consistent": float(charge_law_consistent),
                "contact_thermodynamics_certified": float(
                    contact_certificate.certified
                ),
                "dark_charge_off_bit_identical": float(dark_arrays_identical),
                "dark_incremental_charge_zero_C_m2": float(
                    np.max(np.abs(dark_charge))
                ),
                "dark_reference_certified": float(reference.dark_state.certified),
                "dark_reference_hash_verified": 1.0,
                "dark_trace_shift_zero_V": float(
                    np.max(np.abs(dark_trace_shift))
                ),
                "interface_evidence_aligned": 1.0,
                "max_charge_fraction_of_one_electron": float(
                    np.max(charge_fraction)
                ),
                "max_continuity_bound_A_m2": max(
                    max(
                        result.electron_continuity_bound_A_m2,
                        result.hole_continuity_bound_A_m2,
                    )
                    for result in all_states
                ),
                "max_current_spread_A_m2": max(
                    result.face_current_spread_A_m2 for result in all_states
                ),
                "max_interface_local_residual": max(
                    result.interface_local_residual for result in all_states
                ),
                "max_normalized_cell_residual": max(
                    result.max_normalized_cell_residual for result in all_states
                ),
                "max_normalized_gauss_residual": float(
                    np.max(
                        np.abs(
                            [
                                result.interface_normalized_gauss_residual
                                for result in target_results
                            ]
                        )
                    )
                ),
                "max_poisson_residual": max(
                    result.poisson_residual for result in all_states
                ),
                "max_scaled_local_jacobian_condition": float(
                    np.max(
                        [
                            result.interface_scaled_local_jacobian_condition
                            for result in target_results
                        ]
                    )
                ),
                "occupancy_bounded": float(
                    np.all((all_occupancy >= 0.0) & (all_occupancy <= 1.0))
                ),
                "rebaseline_acknowledged": float(
                    stack.interface_charge_rebaseline_acknowledged
                ),
                "research_charge_closure_active": float(
                    all(
                        result.interface_charge_closure
                        == "equilibrium_referenced"
                        for result in (charged_dark, *target_results)
                    )
                ),
                "two_sided_topology_active": float(
                    all(
                        result.interface_topology == TWO_SIDED_TRACE
                        for result in all_states
                    )
                ),
            },
            "units": {
                "charged_current_density_A_m2": "A m-2",
                "dark_incremental_charge_zero_C_m2": "C m-2",
                "dark_trace_shift_zero_V": "V",
                "interface_sheet_charge_C_m2": "C m-2",
                "interface_trace_potential_shift_V": "V",
                "max_continuity_bound_A_m2": "A m-2",
                "max_current_spread_A_m2": "A m-2",
            },
            "metadata": {
                **_protocol_metadata(protocol),
                "actual_intervals": len(grid) - 1,
                "contact_thermodynamics": dataclasses.asdict(contact_certificate),
                "dark_reference": {
                    "dark_state_sha256": reference.dark_state_sha256,
                    "equilibrium_occupancy": list(reference.equilibrium_occupancy),
                    "grid_sha256": reference.grid_sha256,
                    "stack_sha256": reference.stack_sha256,
                    "trap_density_m2": list(reference.trap_density_m2),
                },
                "interfaces": [
                    {
                        "capture_velocity_n_m_s": pair[0],
                        "capture_velocity_p_m_s": pair[1],
                        "charge_character": "equilibrium_increment_character_independent",
                        "energy_reference": "below_local_conduction_band",
                        "trap_energy_eV": defect.E_t_eV,
                        "trap_density_m2": defect.N_t_cm2 * 1.0e4,
                    }
                    for pair, defect in zip(interface_pairs, defects)
                    if defect is not None
                ],
                "source_grid_intervals": point.grid,
                "target_evidence": target_evidence,
                "target_layout": "target_major_interface_minor",
                "tolerance_controls": {
                    "finite_difference_step": solve_controls[
                        "finite_difference_step"
                    ],
                    "newton_residual_tolerance": solve_controls[
                        "newton_residual_tolerance"
                    ],
                    "poisson_tolerance_V": solve_controls[
                        "poisson_tolerance_V"
                    ],
                },
                "trace_shift_layout": "target_major_interface_minor_side_minor",
            },
        }
    )
