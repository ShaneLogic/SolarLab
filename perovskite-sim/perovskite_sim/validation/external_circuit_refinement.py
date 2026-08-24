"""Numerical refinement adapter for the area-normalized external DC circuit."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from perovskite_sim.experiments.external_circuit import (
    ExternalCircuitProtocol,
    apply_external_circuit,
)
from perovskite_sim.experiments.jv_sweep import (
    JVResult,
    build_jv_experiment_protocol,
    run_jv_sweep,
)
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.solver.tolerances import ComponentwiseAtol

from .dae_refinement import (
    _finite_option,
    _integer_option,
    _protocol_metadata,
    _string_option,
)
from .numerical_certificate import LaneDefinition, MatrixPoint
from .refinement_runner import CellMeasurement


def _componentwise_policy(
    options: dict[str, Any],
    tolerance_factor: float,
) -> ComponentwiseAtol:
    return ComponentwiseAtol(
        carrier_fraction=_finite_option(
            options,
            "carrier_atol_fraction",
            1.0e-12,
        ),
        ion_fraction=_finite_option(
            options,
            "ion_atol_fraction",
            1.0e-12,
        ),
        interface_fraction=_finite_option(
            options,
            "interface_atol_fraction",
            1.0e-12,
        ),
        minimum_atol=_finite_option(options, "minimum_atol", 1.0e-6),
    ).refined(tolerance_factor)


def _study_protocol(
    lane: LaneDefinition,
    *,
    source_experiment_protocol: dict[str, Any],
    voltage_points: int,
    voltage_max_V: float,
    scan_rate_V_s: float,
    rtol: float,
    current_normalization_A_m2: float,
    circuit_protocol: ExternalCircuitProtocol,
    incident_power_W_m2: float,
) -> dict[str, Any]:
    return {
        "circuit": {
            "mapping": (
                "J_terminal=J_device-V_junction/R_shunt;"
                "V_terminal=V_junction-J_terminal*R_series"
            ),
            "protocol": circuit_protocol.to_dict(),
        },
        "intrinsic_jv": {
            "certification_mode": "strict",
            "experiment_protocol": source_experiment_protocol,
            "grid_parameter": lane.grid_parameter,
            "numerical_diagnostics": "accepted_path_collected",
            "rtol": rtol,
            "tolerance_parameter": lane.tolerance_parameter,
        },
        "matrix": {
            "grid_values": list(lane.grid_values),
            "tolerance_factors": list(lane.tolerance_factors),
        },
        "normalization": {
            "terminal_current_A_m2": current_normalization_A_m2,
        },
        "operating_protocol": {
            "illumination": "stack_baseline_generation",
            "incident_power_W_m2": incident_power_W_m2,
            "scan_rate_V_s": scan_rate_V_s,
            "voltage_max_V": voltage_max_V,
            "voltage_points_per_branch": voltage_points,
        },
        "schema_version": "external-series-shunt-dc-refinement-protocol-v1",
    }


def _zero_coupling_exact(source: JVResult) -> bool:
    zero = apply_external_circuit(source, ExternalCircuitProtocol())
    return (
        zero.certified
        and np.array_equal(zero.forward.terminal_voltage_V, source.V_fwd)
        and np.array_equal(zero.forward.terminal_current_A_m2, source.J_fwd)
        and np.array_equal(zero.reverse.terminal_voltage_V, source.V_rev)
        and np.array_equal(zero.reverse.terminal_current_A_m2, source.J_rev)
        and zero.metrics_fwd == source.metrics_fwd
        and zero.metrics_rev == source.metrics_rev
        and zero.hysteresis_index == source.hysteresis_index
    )


def _pce_loss_fraction(source: float, terminal: float) -> float:
    if not np.isfinite(source) or source <= 0.0:
        raise RuntimeError("intrinsic PCE must be finite and positive")
    value = 1.0 - float(terminal) / float(source)
    if not np.isfinite(value):
        raise RuntimeError("external-circuit PCE loss is non-finite")
    return value


def run_external_series_shunt_dc_refinement(
    lane: LaneDefinition,
    point: MatrixPoint,
    project_root: Path,
) -> CellMeasurement:
    """Run one intrinsic-JV plus external-circuit refinement matrix cell."""

    options = lane.options
    if _string_option(options, "config_loader", "standard") != "standard":
        raise ValueError("the external-circuit lane requires config_loader='standard'")
    if options.get("require_protocol") is not True:
        raise ValueError("the external-circuit lane requires an explicit protocol")

    voltage_max_V = _finite_option(options, "V_max_V", 1.2)
    voltage_points = _integer_option(options, "voltage_points", 12, minimum=3)
    scan_rate_V_s = _finite_option(options, "scan_rate_V_s", 20.0)
    rtol = _finite_option(options, "rtol", 1.0e-4)
    series_resistance = _finite_option(
        options,
        "series_resistance_ohm_m2",
        2.0e-4,
        positive=False,
    )
    shunt_resistance = _finite_option(
        options,
        "shunt_resistance_ohm_m2",
        0.2,
    )
    incident_power = _finite_option(options, "incident_power_W_m2", 1000.0)
    current_normalization = _finite_option(
        options,
        "current_normalization_A_m2",
        250.0,
    )
    circuit_protocol = ExternalCircuitProtocol(
        series_resistance_ohm_m2=series_resistance,
        shunt_resistance_ohm_m2=shunt_resistance,
    )
    if circuit_protocol.zero_coupling:
        raise ValueError("the refinement circuit must contain nonzero parasitics")

    stack = load_device_from_yaml(project_root / lane.config_path)
    experiment_protocol = build_jv_experiment_protocol(
        stack,
        v_rate=scan_rate_V_s,
        n_points=voltage_points,
        V_max=voltage_max_V,
        illuminated=True,
        implicit_legacy_protocol=False,
    )
    study_protocol = _study_protocol(
        lane,
        source_experiment_protocol=experiment_protocol.to_dict(),
        voltage_points=voltage_points,
        voltage_max_V=voltage_max_V,
        scan_rate_V_s=scan_rate_V_s,
        rtol=rtol,
        current_normalization_A_m2=current_normalization,
        circuit_protocol=circuit_protocol,
        incident_power_W_m2=incident_power,
    )

    source = run_jv_sweep(
        stack,
        N_grid=point.grid,
        v_rate=scan_rate_V_s,
        n_points=voltage_points,
        rtol=rtol,
        atol=_componentwise_policy(options, point.tolerance_factor),
        V_max=voltage_max_V,
        illuminated=True,
        certification_mode="strict",
        experiment_protocol=experiment_protocol,
        protocol_mode="research_strict",
        collect_numerical_diagnostics=True,
    )
    if not source.certified:
        raise RuntimeError("intrinsic J-V source is not certified")
    if source.protocol is None or source.protocol.sha256 != experiment_protocol.sha256:
        raise RuntimeError("intrinsic J-V returned a different experiment protocol")
    for name in ("V_fwd", "J_fwd", "V_rev", "J_rev"):
        values = np.asarray(getattr(source, name), dtype=float)
        if values.shape != (voltage_points,) or not np.all(np.isfinite(values)):
            raise RuntimeError(f"intrinsic J-V {name} violates the point contract")

    terminal = apply_external_circuit(
        source,
        circuit_protocol,
        incident_power_W_m2=incident_power,
    )
    if not terminal.certified:
        raise RuntimeError("external-circuit mapping is not certified")
    if terminal.source_experiment_protocol_sha256 != experiment_protocol.sha256:
        raise RuntimeError("external-circuit mapping lost the source protocol hash")

    branches = (terminal.forward, terminal.reverse)
    terminal_current = np.concatenate(
        tuple(branch.terminal_current_A_m2 for branch in branches)
    )
    terminal_voltage = np.concatenate(
        tuple(branch.terminal_voltage_V for branch in branches)
    )
    series_drop = np.concatenate(
        tuple(branch.series_voltage_drop_V for branch in branches)
    )
    terminal_metrics = (terminal.metrics_fwd, terminal.metrics_rev)
    source_metrics = (source.metrics_fwd, source.metrics_rev)
    losses = np.asarray(
        [
            _pce_loss_fraction(intrinsic.PCE, mapped.PCE)
            for intrinsic, mapped in zip(source_metrics, terminal_metrics)
        ],
        dtype=float,
    )
    current_balance = max(branch.max_current_balance_error_A_m2 for branch in branches)
    voltage_balance = max(branch.max_voltage_balance_error_V for branch in branches)
    monotonic = (
        terminal.forward.orientation == "ascending"
        and terminal.reverse.orientation == "descending"
        and np.all(np.diff(terminal.forward.terminal_voltage_V) > 0.0)
        and np.all(np.diff(terminal.reverse.terminal_voltage_V) < 0.0)
    )

    return CellMeasurement.from_mapping(
        {
            "observables": {
                "series_voltage_drop_V": series_drop,
                "terminal_current_normalized_trace": (
                    terminal_current / current_normalization
                ),
                "terminal_ff": [metric.FF for metric in terminal_metrics],
                "terminal_jsc_A_m2": [metric.J_sc for metric in terminal_metrics],
                "terminal_pce_percent": [metric.PCE for metric in terminal_metrics],
                "terminal_voc_V": [metric.V_oc for metric in terminal_metrics],
                "terminal_voltage_trace_V": terminal_voltage,
            },
            "quality": {
                "circuit_protocol_hash_verified": float(
                    terminal.circuit_protocol_sha256 == circuit_protocol.sha256
                ),
                "external_circuit_certified": float(terminal.certified),
                "intrinsic_jv_certified": float(source.certified),
                "max_current_balance_error_A_m2": current_balance,
                "max_pce_loss_fraction": float(np.max(losses)),
                "max_voltage_balance_error_V": voltage_balance,
                "min_pce_loss_fraction": float(np.min(losses)),
                "peak_series_voltage_drop_V": float(np.max(np.abs(series_drop))),
                "peak_shunt_current_A_m2": float(
                    max(
                        np.max(np.abs(branch.shunt_current_A_m2)) for branch in branches
                    )
                ),
                "source_experiment_protocol_verified": float(
                    terminal.source_experiment_protocol_sha256
                    == experiment_protocol.sha256
                ),
                "terminal_branches_monotonic": float(monotonic),
                "terminal_voc_bracketed": float(
                    all(metric.voc_bracketed for metric in terminal_metrics)
                ),
                "voltage_points_per_branch_completed": float(voltage_points),
                "zero_coupling_exact": float(_zero_coupling_exact(source)),
            },
            "units": {
                "max_current_balance_error_A_m2": "A m-2",
                "max_voltage_balance_error_V": "V",
                "peak_series_voltage_drop_V": "V",
                "peak_shunt_current_A_m2": "A m-2",
                "series_voltage_drop_V": "V",
                "terminal_jsc_A_m2": "A m-2",
                "terminal_pce_percent": "%",
                "terminal_voc_V": "V",
                "terminal_voltage_trace_V": "V",
            },
            "metadata": {
                **_protocol_metadata(study_protocol),
                "actual": {
                    "circuit_protocol_sha256": circuit_protocol.sha256,
                    "grid": point.grid,
                    "mapping_sha256": terminal.mapping_sha256,
                    "source_experiment_protocol_sha256": experiment_protocol.sha256,
                    "source_result_sha256": terminal.source_result_sha256,
                    "tolerance_factor": point.tolerance_factor,
                },
            },
        }
    )


__all__ = ["run_external_series_shunt_dc_refinement"]


def _measurement_metric(
    measurement: CellMeasurement,
    name: str,
    *,
    quality: bool = False,
) -> np.ndarray:
    metrics = measurement.quality if quality else measurement.observables
    try:
        metric = next(item for item in metrics if item.name == name)
    except StopIteration as exc:
        raise RuntimeError(f"base external-circuit measurement lacks {name!r}") from exc
    values = np.asarray(metric.values, dtype=float)
    return values.reshape(metric.shape) if metric.shape else values


def run_external_series_shunt_dc_operating_quadrant_refinement(
    lane: LaneDefinition,
    point: MatrixPoint,
    project_root: Path,
) -> CellMeasurement:
    """Compare the terminal power-producing quadrant on fixed V/Voc nodes."""

    base = run_external_series_shunt_dc_refinement(lane, point, project_root)
    voltage_points = _integer_option(
        lane.options,
        "voltage_points",
        12,
        minimum=3,
    )
    quadrant_points = _integer_option(
        lane.options,
        "quadrant_voltage_fraction_points",
        21,
        minimum=3,
    )
    terminal_voltage = _measurement_metric(
        base,
        "terminal_voltage_trace_V",
    )
    normalized_current = _measurement_metric(
        base,
        "terminal_current_normalized_trace",
    )
    terminal_voc = _measurement_metric(base, "terminal_voc_V")
    terminal_jsc = _measurement_metric(base, "terminal_jsc_A_m2")
    current_normalization = _finite_option(
        lane.options,
        "current_normalization_A_m2",
        250.0,
    )
    if (
        terminal_voltage.shape != (2 * voltage_points,)
        or normalized_current.shape != terminal_voltage.shape
        or terminal_voc.shape != (2,)
        or terminal_jsc.shape != (2,)
    ):
        raise RuntimeError("base external-circuit trace violates the v2 shape contract")
    if np.any(terminal_voc <= 0.0) or np.any(terminal_jsc <= 0.0):
        raise RuntimeError(
            "terminal Voc and Jsc must be positive in the power quadrant"
        )

    voltage_fraction = np.linspace(0.0, 1.0, quadrant_points)
    normalized_quadrants: list[np.ndarray] = []
    for branch_index in range(2):
        start = branch_index * voltage_points
        stop = start + voltage_points
        branch_voltage_fraction = (
            terminal_voltage[start:stop] / terminal_voc[branch_index]
        )
        branch_current_fraction = (
            normalized_current[start:stop]
            * current_normalization
            / terminal_jsc[branch_index]
        )
        order = np.argsort(branch_voltage_fraction)
        sorted_voltage = branch_voltage_fraction[order]
        sorted_current = branch_current_fraction[order]
        if (
            not np.all(np.isfinite(sorted_voltage))
            or not np.all(np.isfinite(sorted_current))
            or np.any(np.diff(sorted_voltage) <= 0.0)
            or sorted_voltage[0] > 0.0
            or sorted_voltage[-1] < 1.0
        ):
            raise RuntimeError(
                "terminal branch does not cover the normalized power quadrant"
            )
        normalized_quadrants.append(
            np.interp(voltage_fraction, sorted_voltage, sorted_current)
        )
    normalized_trace = np.concatenate(normalized_quadrants)
    if not np.all(np.isfinite(normalized_trace)):
        raise RuntimeError("normalized terminal power-quadrant trace is non-finite")

    metadata = json.loads(base.metadata_json)
    base_protocol_hash = metadata["protocol_hash"]
    protocol = metadata["protocol"]
    protocol["schema_version"] = (
        "external-series-shunt-dc-operating-quadrant-refinement-protocol-v2"
    )
    protocol["observable_sampling"] = {
        "branch_order": ["forward", "reverse"],
        "current_coordinate": "J_terminal/Jsc_terminal",
        "interpolation": "piecewise_linear",
        "scope": "terminal_power_producing_quadrant",
        "voltage_coordinate": "V_terminal/Voc_terminal",
        "voltage_fraction_points": voltage_fraction.tolist(),
    }
    actual = metadata["actual"]
    actual["base_full_trace_protocol_hash"] = base_protocol_hash

    observables = {
        name: _measurement_metric(base, name)
        for name in (
            "terminal_ff",
            "terminal_jsc_A_m2",
            "terminal_pce_percent",
            "terminal_voc_V",
        )
    }
    observables["terminal_power_quadrant_normalized_trace"] = normalized_trace
    quality = {item.name: item.values[0] for item in base.quality}
    quality.update(
        {
            "power_quadrant_interpolation_verified": 1.0,
            "terminal_quadrant_points_completed": float(2 * quadrant_points),
        }
    )
    units = {
        item.name: item.units
        for item in (*base.observables, *base.quality)
        if item.name in observables or item.name in quality
    }
    return CellMeasurement.from_mapping(
        {
            "observables": observables,
            "quality": quality,
            "units": units,
            "metadata": {
                **_protocol_metadata(protocol),
                "actual": actual,
            },
        }
    )


__all__.append("run_external_series_shunt_dc_operating_quadrant_refinement")
