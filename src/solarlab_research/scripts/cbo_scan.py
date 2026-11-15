"""CLI entrypoint for public 1D CBO scans (`python -m solarlab_research.scripts.cbo_scan`)."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
import sys
from typing import Any

from solarlab_research.cbo_scan import (
    CBO_BOUNDARY_POLICIES,
    CBOScanConfig,
    RECOMPUTED_BUILT_IN,
    default_cbo_prepared_device,
    run_cbo_scan,
)

__all__ = ["build_parser", "main"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m solarlab_research.scripts.cbo_scan",
        description="Run a public 1D conduction/valence-band-offset (CBO) scan.",
    )
    parser.add_argument(
        "--delta-ec",
        type=float,
        nargs="+",
        default=[-0.1, 0.0, 0.1, 0.2],
        help="Requested delta_ec_eV = chi_absorber - chi_transport points (eV).",
    )
    parser.add_argument(
        "--boundary-policy",
        choices=CBO_BOUNDARY_POLICIES,
        default=RECOMPUTED_BUILT_IN,
        help="Contact boundary policy across the band-offset sweep.",
    )
    parser.add_argument(
        "--intervals",
        type=int,
        nargs="+",
        default=[6, 12, 6],
        help="Intervals per layer for the 3-layer HTL/absorber/ETL stack.",
    )
    parser.add_argument(
        "--voltage-points",
        type=int,
        default=13,
        help="Number of bias points between 0 V and V_upper.",
    )
    parser.add_argument(
        "--v-upper",
        type=float,
        default=1.125,
        help="Upper bias voltage (V) for dark/illuminated JV scan.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    intervals = tuple(int(n) for n in args.intervals)
    device = default_cbo_prepared_device()
    voltages = tuple(
        args.v_upper * idx / max(int(args.voltage_points) - 1, 1)
        for idx in range(int(args.voltage_points))
    )
    config = CBOScanConfig(
        delta_ec_eV=tuple(float(v) for v in args.delta_ec),
        intervals_per_layer=intervals,
        voltages_V=voltages,
        boundary_policy=args.boundary_policy,
    )
    result = run_cbo_scan(device, config)
    payload: dict[str, Any] = {
        "schema": "solarlab_research.cbo_scan_cli.v1",
        "boundary_policy": result.boundary_policy,
        "pinned_models": [f"{ref.id}@{ref.version}" for ref in result.pinned_models],
        "complete": result.complete,
        "termination_count": len(result.terminations),
        "points": [
            {
                "delta_ec_eV": pt.delta_ec_eV,
                "applied_V_bi_V": pt.applied_V_bi_V,
                "Jsc": None if pt.metrics is None else pt.metrics.short_circuit_current_A_m2,
                "Voc": None if pt.metrics is None else pt.metrics.open_circuit_voltage_V,
                "FF": None if pt.metrics is None else pt.metrics.fill_factor,
                "PCE": None if pt.metrics is None else pt.metrics.efficiency,
            }
            for pt in result.points
            if pt.requested
        ],
        "critical_intervals": [
            {
                "metric": ci.metric,
                "lower_delta_ec_eV": ci.lower_delta_ec_eV,
                "upper_delta_ec_eV": ci.upper_delta_ec_eV,
                "resolved": ci.resolved,
            }
            for ci in result.critical_intervals
        ],
    }
    sys.stdout.write(json.dumps(payload, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
