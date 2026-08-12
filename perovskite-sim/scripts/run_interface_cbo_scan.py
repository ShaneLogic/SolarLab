#!/usr/bin/env python
"""Run the certified quasi-Fermi physical-interface CBO scan."""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np

from perovskite_sim.experiments.cbo_scan import (
    ADAPTIVE_JV_METRICS,
    CBO_BOUNDARY_POLICIES,
    CBOVoltageGridConvergencePolicy,
    FIXED_CONTACTS,
    InterfaceCBOScanError,
    certify_cbo_grid_convergence,
    certify_cbo_statistics_validity,
    certify_cbo_voltage_grid_convergence,
    compare_cbo_scan_to_scaps_reference,
    solve_interface_cbo_scan,
)
from perovskite_sim.experiments.jv_sweep import build_electrical_grid
from perovskite_sim.experiments.quasi_fermi_steady_state import (
    build_two_sided_trace_grid,
)
from perovskite_sim.models.device import electrical_layers
from perovskite_sim.physics.interface_plane import (
    FERMI_DIRAC_RICHARDSON,
    INTERFACE_TRANSPORT_MODELS,
    SCAPS_THERMIONIC,
)
from perovskite_sim.physics.two_sided_interface import (
    DEDUPLICATED_QSS,
    INTERFACE_TOPOLOGIES,
    TWO_SIDED_TRACE,
)
from perovskite_sim.scaps_compat import load_scaps_yaml


def _axis(start: float, stop: float, step: float) -> np.ndarray:
    if step <= 0.0 or stop < start:
        raise ValueError("CBO range requires stop >= start and step > 0")
    count = int(np.floor((stop - start) / step + 1.0e-12))
    values = start + step * np.arange(count + 1, dtype=float)
    if values[-1] < stop - 1.0e-12:
        values = np.r_[values, stop]
    if not np.any(np.isclose(values, 0.0, rtol=0.0, atol=1.0e-12)):
        values = np.r_[values, 0.0]
    return np.unique(np.round(values, 12))


def _build_scan_grid(stack, N_grid: int, interface_topology: str) -> np.ndarray:
    grid = build_electrical_grid(stack, N_grid)
    if interface_topology == TWO_SIDED_TRACE:
        grid = build_two_sided_trace_grid(grid, stack)
    return grid


def _summary(result) -> dict:
    return {
        "schema": "solarlab.interface_cbo_scan",
        "schema_version": "1.7",
        "complete": result.complete,
        "certified": result.certified,
        "settings": {
            "requested_delta_ec_eV": result.requested_delta_ec_eV.tolist(),
            "reference_delta_ec_eV": result.reference_delta_ec_eV,
            "N_grid": result.N_grid,
            "grid_node_count": result.grid_node_count,
            "grid_interval_count": result.grid_interval_count,
            "grid_interval_weights": list(result.grid_interval_weights),
            "grid_alphas": list(result.grid_alphas),
            "reference_grid_warm_starts": result.reference_grid_warm_starts,
            "reference_grid_warm_start_failures": (
                result.reference_grid_warm_start_failures
            ),
            "reference_grid_cold_recoveries": (
                result.reference_grid_cold_recoveries
            ),
            "reference_grid_predictor_recoveries": (
                result.reference_grid_predictor_recoveries
            ),
            "voltages_V": result.voltages_V.tolist(),
            "voltage_grid_point_counts": [
                len(grid) for grid in result.voltage_grids_V
            ],
            "voltage_grid_interval_counts": [
                len(grid) - 1 for grid in result.voltage_grids_V
            ],
            "voltage_refinement_grid_point_counts": [
                len(grid) for grid in result.voltage_refinement_grids_V
            ],
            "voltage_refinement_grid_interval_counts": [
                len(grid) - 1
                for grid in result.voltage_refinement_grids_V
            ],
            "voltage_grid_convergence_policy": dataclasses.asdict(
                result.voltage_grid_convergence_policy
            ),
            "voltage_sampling_method": (
                "point_local_nested_refinement"
                if result.voltage_refinement_grids_V
                else (
                    "nested_subsampling_of_finest_certified_jv"
                    if len(result.voltage_grids_V) > 1
                    else "single_voltage_grid"
                )
            ),
            "mpp_interpolation": result.mpp_interpolation,
            "calculate_jv_metrics": result.calculate_jv_metrics,
            "sync_vbi": result.sync_vbi,
            "boundary_policy": result.boundary_policy,
            "interface_transport_model": result.interface_transport_model,
            "interface_topology": result.interface_topology,
            "heterojunction_recombination_despike": (
                result.heterojunction_recombination_despike
            ),
            "qf_coordinate_system": result.qf_coordinate_system,
            "interface_transmission": result.interface_transmission,
            "relative_drop_fraction": result.relative_drop_fraction,
            "minimum_delta_step_eV": result.minimum_delta_step_eV,
            "maximum_delta_step_eV": result.maximum_delta_step_eV,
            "minimum_voltage_step_V": result.minimum_voltage_step_V,
            "adaptive_full_jv_metrics": list(result.adaptive_jv_metrics),
        },
        "points": [
            {
                "delta_ec_eV": point.delta_ec_eV,
                "requested": point.requested,
                "refinement_metrics": list(point.refinement_metrics),
                "voltage_grid_refined": point.voltage_grid_refined,
                "initial_voltage_grid_reasons": list(
                    point.initial_voltage_grid_reasons
                ),
                "certified": point.certified,
                "metrics": (
                    None
                    if point.metrics is None
                    else dataclasses.asdict(point.metrics)
                ),
                "voltage_grid_metrics": [
                    {
                        "voltage_point_count": sample.voltage_point_count,
                        "voltage_interval_count": (
                            sample.voltage_interval_count
                        ),
                        "retained_voltage_point_count": (
                            sample.retained_voltage_point_count
                        ),
                        "metrics": dataclasses.asdict(sample.metrics),
                        "certified": sample.certified,
                    }
                    for sample in point.voltage_grid_metrics
                ],
                "jv_voltage_continuation": (
                    None
                    if point.jv is None
                    else {
                        "bridge_count": getattr(
                            point.jv,
                            "continuation_bridge_count",
                            0,
                        ),
                        "minimum_voltage_step_V": getattr(
                            point.jv,
                            "minimum_voltage_step_V",
                            None,
                        ),
                        "nodal_predictor_fallback_attempts": getattr(
                            point.jv,
                            "nodal_predictor_fallback_attempts",
                            0,
                        ),
                        "nodal_predictor_fallback_failures": getattr(
                            point.jv,
                            "nodal_predictor_fallback_failures",
                            0,
                        ),
                    }
                ),
                "short_circuit_certificate": {
                    "face_current_spread_A_m2": (
                        point.short_circuit_state.face_current_spread_A_m2
                    ),
                    "electron_continuity_bound_A_m2": (
                        point.short_circuit_state.electron_continuity_bound_A_m2
                    ),
                    "hole_continuity_bound_A_m2": (
                        point.short_circuit_state.hole_continuity_bound_A_m2
                    ),
                    "poisson_residual": point.short_circuit_state.poisson_residual,
                    "interface_local_residual": (
                        point.short_circuit_state.interface_local_residual
                    ),
                    "interface_topology": (
                        point.short_circuit_state.interface_topology
                    ),
                    "qf_coordinate_system": (
                        point.short_circuit_state.qf_coordinate_system
                    ),
                    "edge_coordinate_predictor_used": (
                        point.short_circuit_state.edge_coordinate_predictor_used
                    ),
                    "edge_coordinate_predictor_iterations": (
                        point.short_circuit_state
                        .edge_coordinate_predictor_iterations
                    ),
                },
            }
            for point in result.points
        ],
        "short_circuit_trace": [
            dataclasses.asdict(sample) for sample in result.short_circuit_trace
        ],
        "critical_intervals": [
            dataclasses.asdict(interval) for interval in result.critical_intervals
        ],
        "metric_refinement_trace": [
            dataclasses.asdict(step) for step in result.metric_refinement_trace
        ],
        "terminations": [
            dataclasses.asdict(termination) for termination in result.terminations
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/scaps_mirror_v2.yaml"),
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--delta-min", type=float, default=-1.0)
    parser.add_argument("--delta-max", type=float, default=0.5)
    parser.add_argument("--delta-step", type=float, default=0.1)
    parser.add_argument("--N-grid", type=int, default=30)
    parser.add_argument(
        "--grid-alpha",
        type=float,
        help=(
            "uniform per-layer tanh concentration for an interface-refined "
            "grid; equal interval weights are used when the config has none"
        ),
    )
    parser.add_argument(
        "--grid-alphas",
        type=float,
        nargs="+",
        help="per-electrical-layer tanh concentrations in stack order",
    )
    parser.add_argument(
        "--grid-interval-weights",
        type=float,
        nargs="+",
        help="per-electrical-layer interval allocation weights in stack order",
    )
    parser.add_argument(
        "--grid-ladder",
        type=int,
        nargs="+",
        help="run at least three requested grids and issue a CBO grid certificate",
    )
    parser.add_argument("--n-voltages", type=int, default=29)
    parser.add_argument(
        "--voltage-grid-ladder",
        type=int,
        nargs="+",
        help=(
            "at least three nested voltage point counts, for example "
            "29 57 113; only the finest complete J-V branch is solved"
        ),
    )
    parser.add_argument(
        "--voltage-refinement-grid-ladder",
        type=int,
        nargs="+",
        help=(
            "point-local fallback ladder; it must reuse the two finest base "
            "grids and add one finer nested grid, for example 225 449 897"
        ),
    )
    parser.add_argument(
        "--adaptive-full-jv-metrics",
        nargs="+",
        choices=ADAPTIVE_JV_METRICS,
        default=(),
        help=(
            "adaptively refine selected FF/PCE CBO onsets on every spatial "
            "grid; requires both grid ladders"
        ),
    )
    parser.add_argument("--V-max", type=float, default=1.4)
    parser.add_argument(
        "--mpp-interpolation",
        choices=("sampled", "local_quadratic"),
        default="local_quadratic",
        help="MPP extraction mode for CBO J-V metrics",
    )
    parser.add_argument("--relative-drop", type=float, default=0.01)
    parser.add_argument("--interface-transmission", type=float, default=1.0)
    parser.add_argument(
        "--transmission-continuation",
        type=float,
        nargs="+",
        help=(
            "first-grid reference-state continuation path; the final value "
            "must equal interface-transmission"
        ),
    )
    parser.add_argument(
        "--interface-transport-model",
        choices=INTERFACE_TRANSPORT_MODELS,
        default=SCAPS_THERMIONIC,
    )
    parser.add_argument(
        "--interface-topology",
        choices=INTERFACE_TOPOLOGIES,
        default=DEDUPLICATED_QSS,
        help="opt-in interface grid topology; legacy topology remains default",
    )
    parser.add_argument(
        "--disable-legacy-heterojunction-despike",
        action="store_true",
        help=(
            "explicitly set het_recomb_despike=0 for a two-sided run; the "
            "old correction targets a shared boundary node that this topology "
            "removes"
        ),
    )
    parser.add_argument("--minimum-delta-step", type=float, default=5.0e-4)
    parser.add_argument("--maximum-delta-step", type=float, default=5.0e-2)
    parser.add_argument(
        "--minimum-voltage-step",
        type=float,
        help=(
            "explicitly enable J-V voltage-bisection warm starts down to this "
            "step; default is disabled"
        ),
    )
    parser.add_argument(
        "--boundary-policy",
        choices=CBO_BOUNDARY_POLICIES,
        default=FIXED_CONTACTS,
    )
    parser.add_argument(
        "--maximum-grid-envelope-eV",
        type=float,
        default=1.0e-2,
    )
    parser.add_argument(
        "--maximum-successive-shift-ratio",
        type=float,
        default=0.9,
        help="maximum allowed ratio between consecutive critical-CBO shifts",
    )
    parser.add_argument(
        "--maximum-voc-change-mV",
        type=float,
        default=2.0,
        help="maximum Voc change between the two finest voltage grids",
    )
    parser.add_argument(
        "--maximum-ff-change",
        type=float,
        default=1.0e-3,
        help="maximum absolute FF change between the two finest grids",
    )
    parser.add_argument(
        "--maximum-pce-change",
        type=float,
        default=5.0e-4,
        help="maximum absolute PCE change between the two finest grids",
    )
    parser.add_argument(
        "--maximum-voltage-successive-change-ratio",
        type=float,
        default=0.8,
        help="maximum contraction ratio for successive metric changes",
    )
    parser.add_argument(
        "--voltage-contraction-noise-floor-fraction",
        type=float,
        default=0.1,
        help=(
            "apply the contraction-ratio gate only above this fraction of "
            "each metric's absolute convergence tolerance"
        ),
    )
    parser.add_argument(
        "--maximum-boltzmann-state-to-dos",
        type=float,
        default=0.1,
        help="dilute-statistics validity ceiling for Boltzmann interface models",
    )
    parser.add_argument(
        "--scaps-reference",
        type=Path,
        help="content-addressed scaps_reference.json used for external comparison",
    )
    parser.add_argument(
        "--maximum-reference-critical-interval-width-eV",
        type=float,
        default=2.0e-2,
        help="maximum accepted SCAPS bracket width around the metric onset",
    )
    parser.add_argument(
        "--external-metrics",
        nargs="+",
        choices=("Jsc", "FF", "PCE"),
        default=("Jsc",),
    )
    parser.add_argument(
        "--short-circuit-only",
        action="store_true",
        help="skip full J-V curves and report only the certified Jsc onset",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    adaptive_full_jv_metrics = tuple(args.adaptive_full_jv_metrics)
    if (
        args.interface_topology == TWO_SIDED_TRACE
        and args.interface_transport_model != FERMI_DIRAC_RICHARDSON
    ):
        raise ValueError(
            "two_sided_trace requires --interface-transport-model "
            f"{FERMI_DIRAC_RICHARDSON}"
        )
    transmission_continuation = tuple(
        float(value) for value in (args.transmission_continuation or ())
    )
    if transmission_continuation:
        if any(
            not np.isfinite(value) or value <= 0.0 or value > 1.0
            for value in transmission_continuation
        ):
            raise ValueError("transmission-continuation values must lie in (0, 1]")
        if not np.isclose(
            transmission_continuation[-1],
            args.interface_transmission,
            rtol=0.0,
            atol=0.0,
        ):
            raise ValueError(
                "transmission-continuation must end at interface-transmission"
            )
    stack = load_scaps_yaml(args.config)
    input_het_recomb_despike = float(
        getattr(stack, "het_recomb_despike", 0.0)
    )
    if args.disable_legacy_heterojunction_despike:
        stack = dataclasses.replace(stack, het_recomb_despike=0.0)
    effective_het_recomb_despike = float(
        getattr(stack, "het_recomb_despike", 0.0)
    )
    if (
        args.interface_topology == TWO_SIDED_TRACE
        and effective_het_recomb_despike > 0.0
    ):
        raise ValueError(
            "two_sided_trace removes the shared recombination node; pass "
            "--disable-legacy-heterojunction-despike to acknowledge the "
            "protocol change"
        )
    if args.grid_alpha is not None and args.grid_alphas is not None:
        raise ValueError("use either grid_alpha or grid_alphas, not both")
    layer_count = len(electrical_layers(stack))
    override_alphas: tuple[float, ...] | None = None
    if args.grid_alphas is not None:
        override_alphas = tuple(float(value) for value in args.grid_alphas)
        if len(override_alphas) != layer_count:
            raise ValueError(
                "grid_alphas must contain one value per electrical layer"
            )
    elif args.grid_alpha is not None:
        override_alphas = (float(args.grid_alpha),) * layer_count
    if override_alphas is not None:
        if any(not np.isfinite(value) or value <= 0.0 for value in override_alphas):
            raise ValueError("grid alphas must be finite and positive")
        weights = (
            tuple(float(value) for value in args.grid_interval_weights)
            if args.grid_interval_weights is not None
            else stack.grid_interval_weights or (1.0,) * layer_count
        )
        if len(weights) != layer_count:
            raise ValueError(
                "grid_interval_weights must contain one value per "
                "electrical layer"
            )
        if any(not np.isfinite(value) or value <= 0.0 for value in weights):
            raise ValueError("grid interval weights must be finite and positive")
        stack = dataclasses.replace(
            stack,
            grid_interval_weights=tuple(weights),
            grid_alphas=override_alphas,
        )
    elif args.grid_interval_weights is not None:
        raise ValueError("grid_interval_weights requires a grid alpha override")
    values = _axis(args.delta_min, args.delta_max, args.delta_step)
    if not np.isfinite(args.V_max) or args.V_max <= 0.0:
        raise ValueError("V-max must be finite and positive")
    voltage_point_counts = tuple(args.voltage_grid_ladder or ())
    voltage_refinement_point_counts = tuple(
        args.voltage_refinement_grid_ladder or ()
    )
    if voltage_refinement_point_counts and not voltage_point_counts:
        raise ValueError(
            "voltage-refinement-grid-ladder requires voltage-grid-ladder"
        )
    if voltage_point_counts:
        if args.short_circuit_only:
            raise ValueError(
                "voltage-grid-ladder is incompatible with short-circuit-only"
            )
        if len(voltage_point_counts) < 3:
            raise ValueError(
                "voltage-grid-ladder requires at least three point counts"
            )
        if any(count < 2 for count in voltage_point_counts):
            raise ValueError("voltage grid point counts must be at least two")
        if any(
            right <= left
            for left, right in zip(
                voltage_point_counts[:-1], voltage_point_counts[1:]
            )
        ):
            raise ValueError(
                "voltage-grid-ladder must have unique increasing point counts"
            )
        if any(
            (right - 1) % (left - 1) != 0
            for left, right in zip(
                voltage_point_counts[:-1], voltage_point_counts[1:]
            )
        ):
            raise ValueError(
                "uniform voltage-grid-ladder intervals must be exactly nested"
            )
        voltage_grids = tuple(
            np.linspace(0.0, args.V_max, count)
            for count in voltage_point_counts
        )
        voltages = voltage_grids[-1]
    else:
        if args.n_voltages < 2:
            raise ValueError("n-voltages must be at least two")
        voltages = np.linspace(0.0, args.V_max, args.n_voltages)
        voltage_grids = None

    if voltage_refinement_point_counts:
        if args.short_circuit_only:
            raise ValueError(
                "voltage-refinement-grid-ladder is incompatible with "
                "short-circuit-only"
            )
        if len(voltage_refinement_point_counts) < 3:
            raise ValueError(
                "voltage-refinement-grid-ladder requires at least three "
                "point counts"
            )
        if any(count < 2 for count in voltage_refinement_point_counts):
            raise ValueError(
                "voltage refinement grid point counts must be at least two"
            )
        if any(
            right <= left
            for left, right in zip(
                voltage_refinement_point_counts[:-1],
                voltage_refinement_point_counts[1:],
            )
        ):
            raise ValueError(
                "voltage-refinement-grid-ladder must have unique increasing "
                "point counts"
            )
        if any(
            (right - 1) % (left - 1) != 0
            for left, right in zip(
                voltage_refinement_point_counts[:-1],
                voltage_refinement_point_counts[1:],
            )
        ):
            raise ValueError(
                "uniform voltage-refinement-grid-ladder intervals must be "
                "exactly nested"
            )
        if (
            len(voltage_point_counts) != 3
            or len(voltage_refinement_point_counts) != 3
            or voltage_refinement_point_counts[:-1]
            != voltage_point_counts[1:]
        ):
            raise ValueError(
                "point-local voltage refinement requires base [a,b,c] and "
                "fallback [b,c,d] ladders"
            )
        voltage_refinement_grids = tuple(
            np.linspace(0.0, args.V_max, count)
            for count in voltage_refinement_point_counts
        )
    else:
        voltage_refinement_grids = None

    voltage_grid_convergence_policy = CBOVoltageGridConvergencePolicy(
        maximum_voc_change_V=args.maximum_voc_change_mV * 1.0e-3,
        maximum_ff_change=args.maximum_ff_change,
        maximum_pce_change=args.maximum_pce_change,
        maximum_successive_change_ratio=(
            args.maximum_voltage_successive_change_ratio
        ),
        contraction_noise_floor_fraction=(
            args.voltage_contraction_noise_floor_fraction
        ),
    )

    requested_grids = tuple(sorted(args.grid_ladder or (args.N_grid,)))
    if adaptive_full_jv_metrics:
        if args.short_circuit_only:
            raise ValueError(
                "adaptive-full-jv-metrics is incompatible with "
                "short-circuit-only"
            )
        if voltage_grids is None:
            raise ValueError(
                "adaptive-full-jv-metrics requires --voltage-grid-ladder"
            )
        if (
            args.grid_ladder is None
            or len(requested_grids) < 3
            or len(set(requested_grids)) != len(requested_grids)
        ):
            raise ValueError(
                "adaptive-full-jv-metrics requires at least three unique "
                "--grid-ladder values"
            )
    results = []
    voltage_certificates_by_result = {}

    def voltage_certificate_for(result):
        key = id(result)
        certificate = voltage_certificates_by_result.get(key)
        if certificate is None:
            certificate = certify_cbo_voltage_grid_convergence(
                result,
                maximum_voc_change_V=(
                    args.maximum_voc_change_mV * 1.0e-3
                ),
                maximum_ff_change=args.maximum_ff_change,
                maximum_pce_change=args.maximum_pce_change,
                maximum_successive_change_ratio=(
                    args.maximum_voltage_successive_change_ratio
                ),
                contraction_noise_floor_fraction=(
                    args.voltage_contraction_noise_floor_fraction
                ),
            )
            voltage_certificates_by_result[key] = certificate
        return certificate

    def build_payload(failure: dict | None = None):
        if results:
            finest = max(results, key=lambda item: item.grid_interval_count)
            payload = _summary(finest)
            payload["numerical_certified"] = finest.certified
            acceptance = finest.certified and failure is None
        else:
            finest = None
            payload = {
                "schema": "solarlab.interface_cbo_scan",
                "schema_version": "1.7",
                "complete": False,
                "certified": False,
                "settings": {
                    "requested_delta_ec_eV": values.tolist(),
                    "requested_grids": list(requested_grids),
                    "boundary_policy": args.boundary_policy,
                    "interface_transport_model": (
                        args.interface_transport_model
                    ),
                    "interface_topology": args.interface_topology,
                    "transmission_continuation": list(
                        transmission_continuation
                    ),
                    "voltage_grid_point_counts": list(
                        voltage_point_counts
                    ),
                    "voltage_refinement_grid_point_counts": list(
                        voltage_refinement_point_counts
                    ),
                    "voltage_grid_convergence_policy": dataclasses.asdict(
                        voltage_grid_convergence_policy
                    ),
                    "minimum_voltage_step_V": args.minimum_voltage_step,
                    "mpp_interpolation": args.mpp_interpolation,
                    "adaptive_full_jv_metrics": list(
                        adaptive_full_jv_metrics
                    ),
                },
            }
            acceptance = False

        payload["settings"]["requested_grids"] = list(requested_grids)
        payload["settings"]["transmission_continuation"] = list(
            transmission_continuation
        )
        payload["settings"]["requested_voltage_grid_point_counts"] = list(
            voltage_point_counts
        )
        payload["settings"][
            "requested_voltage_refinement_grid_point_counts"
        ] = list(voltage_refinement_point_counts)
        payload["settings"]["voltage_grid_convergence_policy"] = (
            dataclasses.asdict(voltage_grid_convergence_policy)
        )
        payload["settings"]["minimum_voltage_step_V"] = (
            args.minimum_voltage_step
        )
        payload["settings"]["mpp_interpolation"] = args.mpp_interpolation
        payload["settings"]["adaptive_full_jv_metrics"] = list(
            adaptive_full_jv_metrics
        )
        payload["settings"]["input_heterojunction_recombination_despike"] = (
            input_het_recomb_despike
        )
        payload["settings"]["heterojunction_recombination_despike"] = (
            effective_het_recomb_despike
        )
        payload["settings"][
            "legacy_heterojunction_despike_explicitly_disabled"
        ] = bool(args.disable_legacy_heterojunction_despike)

        statistics_certificates = tuple(
            certify_cbo_statistics_validity(
                item,
                maximum_boltzmann_state_to_dos=(
                    args.maximum_boltzmann_state_to_dos
                ),
            )
            for item in results
        )
        statistics_certified = bool(
            statistics_certificates
            and all(
                certificate.certified
                for certificate in statistics_certificates
            )
        )
        payload["statistics_validity"] = {
            "certificates": [
                dataclasses.asdict(certificate)
                for certificate in statistics_certificates
            ],
            "certified": statistics_certified,
        }
        acceptance = acceptance and statistics_certified

        if args.grid_ladder and results:
            grid_metrics = ("Jsc", *adaptive_full_jv_metrics)
            grid_certificates = {
                metric: certify_cbo_grid_convergence(
                    results,
                    metric=metric,
                    maximum_envelope_width_eV=(
                        args.maximum_grid_envelope_eV
                    ),
                    maximum_successive_shift_ratio=(
                        args.maximum_successive_shift_ratio
                    ),
                )
                for metric in grid_metrics
            }
            payload["grid_convergence"] = dataclasses.asdict(
                grid_certificates["Jsc"]
            )
            grid_metrics_certified = all(
                certificate.certified
                for certificate in grid_certificates.values()
            )
            payload["metric_grid_convergence"] = {
                "certificates": {
                    metric: dataclasses.asdict(certificate)
                    for metric, certificate in grid_certificates.items()
                },
                "certified": grid_metrics_certified,
            }
            payload["grid_runs"] = [_summary(item) for item in results]
            acceptance = acceptance and grid_metrics_certified
        if not args.short_circuit_only:
            if voltage_grids is not None and finest is not None:
                voltage_results = (
                    tuple(results)
                    if adaptive_full_jv_metrics
                    else (finest,)
                )
                voltage_certificates = tuple(
                    voltage_certificate_for(item)
                    for item in voltage_results
                )
                voltage_certificate = voltage_certificates[-1]
                payload["voltage_grid_convergence"] = dataclasses.asdict(
                    voltage_certificate
                )
                spatial_voltage_certified = all(
                    certificate.certified
                    for certificate in voltage_certificates
                )
                payload["spatial_voltage_grid_convergence"] = {
                    "certificates": [
                        {
                            "grid_interval_count": item.grid_interval_count,
                            "certificate": dataclasses.asdict(certificate),
                        }
                        for item, certificate in zip(
                            voltage_results,
                            voltage_certificates,
                        )
                    ],
                    "certified": spatial_voltage_certified,
                }
                acceptance = acceptance and spatial_voltage_certified
            else:
                payload["voltage_grid_convergence"] = {
                    "sampling_method": "single_voltage_grid",
                    "certified": False,
                    "reasons": [
                        "full J-V metrics require --voltage-grid-ladder "
                        "for top-level certification"
                    ],
                }
                payload["spatial_voltage_grid_convergence"] = {
                    "certificates": [],
                    "certified": False,
                    "reasons": [
                        "full J-V metrics require --voltage-grid-ladder"
                    ],
                }
                acceptance = False
        if args.scaps_reference is not None and finest is not None:
            external = compare_cbo_scan_to_scaps_reference(
                finest,
                args.scaps_reference,
                metrics=tuple(args.external_metrics),
                maximum_reference_critical_interval_width_eV=(
                    args.maximum_reference_critical_interval_width_eV
                ),
            )
            payload["external_validation"] = dataclasses.asdict(external)
            payload["external_validation"]["certified"] = external.certified
            acceptance = acceptance and external.certified
        if failure is not None:
            payload["complete"] = False
            payload["grid_failure"] = failure
            acceptance = False
        payload["certified"] = bool(acceptance)
        return payload, bool(acceptance), finest

    reference_initial_state = None
    reference_initial_state_grid = None
    for requested_grid in requested_grids:
        calculate_full_jv = bool(
            not args.short_circuit_only
            and (
                adaptive_full_jv_metrics
                or
                voltage_grids is None
                or requested_grid == requested_grids[-1]
            )
        )

        def progress(stage: str, current: int, total: int, message: str) -> None:
            print(
                f"[N={requested_grid}:{stage}] {current}/{total} {message}",
                flush=True,
            )

        try:
            if not results and transmission_continuation:
                continuation_state = None
                continuation_grid = _build_scan_grid(
                    stack,
                    requested_grid,
                    args.interface_topology,
                )
                for transmission in transmission_continuation:
                    predictor = (
                        None
                        if continuation_state is None
                        else dataclasses.replace(
                            continuation_state,
                            interface_transmission=transmission,
                        )
                    )
                    print(
                        f"[N={requested_grid}:transmission] "
                        f"T={transmission:g}",
                        flush=True,
                    )
                    continuation_result = solve_interface_cbo_scan(
                        stack,
                        np.array([0.0]),
                        N_grid=requested_grid,
                        relative_drop_fraction=args.relative_drop,
                        minimum_delta_step_eV=args.minimum_delta_step,
                        maximum_delta_step_eV=args.maximum_delta_step,
                        minimum_voltage_step_V=args.minimum_voltage_step,
                        mpp_interpolation=args.mpp_interpolation,
                        interface_transmission=transmission,
                        interface_transport_model=(
                            args.interface_transport_model
                        ),
                        interface_topology=args.interface_topology,
                        boundary_policy=args.boundary_policy,
                        calculate_jv_metrics=False,
                        reference_initial_state=predictor,
                        reference_initial_state_grid=(
                            continuation_grid if predictor is not None else None
                        ),
                    )
                    continuation_point = next(
                        point
                        for point in continuation_result.points
                        if np.isclose(
                            point.delta_ec_eV,
                            continuation_result.reference_delta_ec_eV,
                            rtol=0.0,
                            atol=1.0e-12,
                        )
                    )
                    continuation_state = (
                        continuation_point.short_circuit_state
                    )
                reference_initial_state = continuation_state
                reference_initial_state_grid = continuation_grid
            result = solve_interface_cbo_scan(
                stack,
                values,
                voltages_V=(
                    voltages
                    if calculate_full_jv and voltage_grids is None
                    else None
                ),
                voltage_grids_V=(
                    voltage_grids if calculate_full_jv else None
                ),
                voltage_refinement_grids_V=(
                    voltage_refinement_grids
                    if calculate_full_jv
                    else None
                ),
                voltage_grid_convergence_policy=(
                    voltage_grid_convergence_policy
                ),
                N_grid=requested_grid,
                relative_drop_fraction=args.relative_drop,
                minimum_delta_step_eV=args.minimum_delta_step,
                maximum_delta_step_eV=args.maximum_delta_step,
                minimum_voltage_step_V=args.minimum_voltage_step,
                mpp_interpolation=args.mpp_interpolation,
                interface_transmission=args.interface_transmission,
                interface_transport_model=args.interface_transport_model,
                interface_topology=args.interface_topology,
                boundary_policy=args.boundary_policy,
                calculate_jv_metrics=calculate_full_jv,
                adaptive_jv_metrics=(
                    adaptive_full_jv_metrics if calculate_full_jv else ()
                ),
                reference_initial_state=reference_initial_state,
                reference_initial_state_grid=reference_initial_state_grid,
                progress=progress,
            )
        except InterfaceCBOScanError as exc:
            failure = {
                "requested_grid": requested_grid,
                "error_type": type(exc).__name__,
                "message": str(exc),
            }
            payload, _, _ = build_payload(failure)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(
                json.dumps(payload, indent=2),
                encoding="utf-8",
            )
            print(f"grid failure at N={requested_grid}: {exc}")
            print(f"partial JSON: {args.out}")
            return 1
        results.append(result)
        if adaptive_full_jv_metrics and voltage_grids is not None:
            voltage_certificate = voltage_certificate_for(result)
            if not voltage_certificate.certified:
                failure = {
                    "requested_grid": requested_grid,
                    "error_type": "CBOVoltageGridConvergenceError",
                    "message": "; ".join(voltage_certificate.reasons),
                }
                payload, _, _ = build_payload(failure)
                args.out.parent.mkdir(parents=True, exist_ok=True)
                args.out.write_text(
                    json.dumps(payload, indent=2),
                    encoding="utf-8",
                )
                print(
                    f"voltage-grid failure at N={requested_grid}: "
                    f"{failure['message']}"
                )
                print(f"partial JSON: {args.out}")
                return 1
        reference_point = next(
            point
            for point in result.points
            if np.isclose(
                point.delta_ec_eV,
                result.reference_delta_ec_eV,
                rtol=0.0,
                atol=1.0e-12,
            )
        )
        reference_initial_state = reference_point.short_circuit_state
        reference_initial_state_grid = _build_scan_grid(
            stack,
            requested_grid,
            args.interface_topology,
        )
    payload, acceptance, result = build_payload()
    assert result is not None
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(
        f"accepted={acceptance} "
        f"numerical_certified={result.certified} "
        f"complete={result.complete}"
    )
    print(
        f"boundary={result.boundary_policy} "
        f"transport={result.interface_transport_model} "
        f"topology={result.interface_topology} "
        f"actual_grid={result.grid_interval_count} intervals"
    )
    for interval in result.critical_intervals:
        print(
            f"{interval.metric}: "
            f"[{interval.lower_delta_ec_eV}, {interval.upper_delta_ec_eV}] eV"
        )
    for termination in result.terminations:
        print(
            "validity endpoint: "
            f"[{termination.last_certified_delta_ec_eV}, "
            f"{termination.first_failed_delta_ec_eV}] eV"
        )
    print(f"JSON: {args.out}")
    return 0 if acceptance else 1


if __name__ == "__main__":
    raise SystemExit(main())
