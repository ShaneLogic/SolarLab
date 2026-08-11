#!/usr/bin/env python
"""Run fail-closed CBO model/transmission sensitivity scans."""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np

from perovskite_sim.experiments.cbo_scan import (
    CBO_BOUNDARY_POLICIES,
    FIXED_CONTACTS,
    InterfaceCBOScanError,
    certify_cbo_statistics_validity,
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


def _requested_trace(result) -> list[dict]:
    return [
        dataclasses.asdict(sample)
        for sample in result.short_circuit_trace
        if sample.requested
    ]


def _external_validation_payload(external) -> dict | None:
    if external is None:
        return None
    payload = dataclasses.asdict(external)
    payload["certified"] = bool(external.certified)
    return payload


def _classify_single_grid_candidate(
    *,
    numerical_certified: bool,
    statistics_certified: bool,
    external_certified: bool | None,
) -> dict:
    """Classify a screening run without overstating single-grid evidence."""
    reasons: list[str] = []
    if not numerical_certified:
        reasons.append("single-grid numerical certificate failed")
    if not statistics_certified:
        reasons.append("interface statistics certificate failed")
    if external_certified is False:
        reasons.append("external comparison certificate failed")
    screened = not reasons
    reasons.append(
        "grid convergence is not evaluated by the sensitivity runner; "
        "run_interface_cbo_scan.py --grid-ladder is required"
    )
    return {
        "single_grid_screen_passed": screened,
        "certified": False,
        "certification_reasons": reasons,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/scaps_mirror_v2.yaml"),
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=INTERFACE_TRANSPORT_MODELS)
    parser.add_argument(
        "--interface-topology",
        choices=INTERFACE_TOPOLOGIES,
        default=DEDUPLICATED_QSS,
    )
    parser.add_argument(
        "--disable-legacy-heterojunction-despike",
        action="store_true",
    )
    parser.add_argument(
        "--transmissions",
        nargs="+",
        type=float,
        default=(1.0, 0.3, 0.1, 0.03, 0.01),
    )
    parser.add_argument("--delta-min", type=float, default=0.0)
    parser.add_argument("--delta-max", type=float, default=0.5)
    parser.add_argument("--delta-step", type=float, default=0.1)
    parser.add_argument("--N-grid", type=int, default=20)
    parser.add_argument("--grid-alphas", type=float, nargs="+")
    parser.add_argument("--grid-interval-weights", type=float, nargs="+")
    parser.add_argument("--relative-drop", type=float, default=0.01)
    parser.add_argument("--minimum-delta-step", type=float, default=5.0e-4)
    parser.add_argument("--maximum-delta-step", type=float, default=5.0e-2)
    parser.add_argument(
        "--boundary-policy",
        choices=CBO_BOUNDARY_POLICIES,
        default=FIXED_CONTACTS,
    )
    parser.add_argument("--scaps-reference", type=Path)
    parser.add_argument(
        "--maximum-boltzmann-state-to-dos",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--maximum-reference-critical-interval-width-eV",
        type=float,
        default=0.02,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    default_models = (
        (FERMI_DIRAC_RICHARDSON,)
        if args.interface_topology == TWO_SIDED_TRACE
        else INTERFACE_TRANSPORT_MODELS
    )
    models = tuple(args.models or default_models)
    if args.interface_topology == TWO_SIDED_TRACE and any(
        model != FERMI_DIRAC_RICHARDSON for model in models
    ):
        raise ValueError(
            "two_sided_trace sensitivity runs currently support only "
            f"{FERMI_DIRAC_RICHARDSON}"
        )
    transmissions = tuple(float(value) for value in args.transmissions)
    if any(
        not np.isfinite(value) or value <= 0.0 or value > 1.0
        for value in transmissions
    ):
        raise ValueError("transmissions must lie in (0, 1]")
    if len(set(transmissions)) != len(transmissions):
        raise ValueError("transmissions must be unique")

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
    layer_count = len(electrical_layers(stack))
    if args.grid_alphas is not None:
        alphas = tuple(float(value) for value in args.grid_alphas)
        if len(alphas) != layer_count:
            raise ValueError("grid-alphas requires one value per layer")
        weights = tuple(
            float(value)
            for value in (
                args.grid_interval_weights
                or stack.grid_interval_weights
                or (1.0,) * layer_count
            )
        )
        if len(weights) != layer_count:
            raise ValueError("grid-interval-weights requires one value per layer")
        if any(
            not np.isfinite(value) or value <= 0.0
            for value in alphas + weights
        ):
            raise ValueError("grid controls must be finite and positive")
        stack = dataclasses.replace(
            stack,
            grid_alphas=alphas,
            grid_interval_weights=weights,
        )
    elif args.grid_interval_weights is not None:
        raise ValueError("grid-interval-weights requires grid-alphas")

    values = _axis(args.delta_min, args.delta_max, args.delta_step)
    grid = build_electrical_grid(stack, args.N_grid)
    if args.interface_topology == TWO_SIDED_TRACE:
        grid = build_two_sided_trace_grid(grid, stack)
    runs: list[dict] = []
    complete_runs = 0
    for model in models:
        previous_reference = None
        for transmission in transmissions:
            predictor_used = previous_reference is not None
            predictor = (
                None
                if previous_reference is None
                else dataclasses.replace(
                    previous_reference,
                    interface_transmission=transmission,
                )
            )
            print(
                f"[{model}:T={transmission:g}] "
                f"predictor={'yes' if predictor_used else 'no'}",
                flush=True,
            )
            try:
                result = solve_interface_cbo_scan(
                    stack,
                    values,
                    N_grid=args.N_grid,
                    relative_drop_fraction=args.relative_drop,
                    minimum_delta_step_eV=args.minimum_delta_step,
                    maximum_delta_step_eV=args.maximum_delta_step,
                    interface_transmission=transmission,
                    interface_transport_model=model,
                    interface_topology=args.interface_topology,
                    boundary_policy=args.boundary_policy,
                    calculate_jv_metrics=False,
                    reference_initial_state=predictor,
                    reference_initial_state_grid=(
                        grid if predictor is not None else None
                    ),
                )
            except InterfaceCBOScanError as exc:
                classification = _classify_single_grid_candidate(
                    numerical_certified=False,
                    statistics_certified=False,
                    external_certified=None,
                )
                runs.append(
                    {
                        "interface_transport_model": model,
                        "interface_topology": args.interface_topology,
                        "interface_transmission": transmission,
                        "transmission_predictor_used": predictor_used,
                        "complete": False,
                        "numerical_certified": False,
                        **classification,
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    }
                )
                continue

            complete_runs += 1
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
            previous_reference = reference_point.short_circuit_state
            statistics = certify_cbo_statistics_validity(
                result,
                maximum_boltzmann_state_to_dos=(
                    args.maximum_boltzmann_state_to_dos
                ),
            )
            external = (
                None
                if args.scaps_reference is None
                else compare_cbo_scan_to_scaps_reference(
                    result,
                    args.scaps_reference,
                    maximum_reference_critical_interval_width_eV=(
                        args.maximum_reference_critical_interval_width_eV
                    ),
                )
            )
            classification = _classify_single_grid_candidate(
                numerical_certified=bool(result.complete and result.certified),
                statistics_certified=statistics.certified,
                external_certified=(
                    None if external is None else external.certified
                ),
            )
            runs.append(
                {
                    "interface_transport_model": model,
                    "interface_topology": args.interface_topology,
                    "interface_transmission": transmission,
                    "transmission_predictor_used": predictor_used,
                    "complete": result.complete,
                    "numerical_certified": result.certified,
                    **classification,
                    "critical_intervals": [
                        dataclasses.asdict(interval)
                        for interval in result.critical_intervals
                    ],
                    "requested_trace": _requested_trace(result),
                    "statistics_validity": dataclasses.asdict(statistics),
                    "external_validation": _external_validation_payload(
                        external
                    ),
                }
            )

    payload = {
        "schema": "solarlab.interface_cbo_sensitivity",
        "schema_version": "1.2",
        "certification_scope": "single_grid_model_form_screening_only",
        "requires_grid_ladder": True,
        "settings": {
            "config": str(args.config),
            "models": list(models),
            "interface_topology": args.interface_topology,
            "transmissions": list(transmissions),
            "requested_delta_ec_eV": values.tolist(),
            "N_grid": args.N_grid,
            "grid_interval_count": len(grid) - 1,
            "grid_alphas": list(stack.grid_alphas or ()),
            "grid_interval_weights": list(stack.grid_interval_weights or ()),
            "boundary_policy": args.boundary_policy,
            "input_heterojunction_recombination_despike": (
                input_het_recomb_despike
            ),
            "heterojunction_recombination_despike": (
                effective_het_recomb_despike
            ),
            "legacy_heterojunction_despike_explicitly_disabled": bool(
                args.disable_legacy_heterojunction_despike
            ),
            "scaps_reference": (
                None if args.scaps_reference is None else str(args.scaps_reference)
            ),
        },
        "runs": runs,
        "complete_runs": complete_runs,
        "single_grid_screened_candidates": sum(
            bool(run["single_grid_screen_passed"]) for run in runs
        ),
        "certified_candidates": 0,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0 if complete_runs else 1


if __name__ == "__main__":
    raise SystemExit(main())
