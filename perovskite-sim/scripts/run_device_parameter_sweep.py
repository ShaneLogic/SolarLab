#!/usr/bin/env python
"""Run controlled SolarLab device-parameter sweeps."""

from __future__ import annotations

import argparse
from pathlib import Path

from perovskite_sim.sweeps.device_parameter_sweep import (
    make_coupled_points,
    make_defect_matrix_points,
    make_full_one_factor_points,
    make_pilot_points,
    run_sweep,
    write_results_csv,
    write_results_json,
    write_summary_plots,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/solarscale_nip_band_aligned.yaml"),
        help="Baseline YAML config. Prefer a band-aligned config with chi/Eg on every electrical layer.",
    )
    parser.add_argument(
        "--preset",
        choices=("pilot", "coupled", "defect-matrix", "full-one-factor"),
        default="pilot",
        help="Sweep point set to run.",
    )
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--max-points", type=int, default=None)
    parser.add_argument("--N-grid", type=int, default=30)
    parser.add_argument("--n-points", type=int, default=8)
    parser.add_argument("--v-rate", type=float, default=5.0)
    parser.add_argument("--V-max", type=float, default=None)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument(
        "--keep-config-vbi",
        action="store_true",
        help="Do not sync device.V_bi to stack.compute_V_bi() after parameter changes.",
    )
    parser.add_argument(
        "--etl-delta-step",
        type=float,
        default=0.1,
        help="Step size for the full-one-factor ETL DeltaEc range.",
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Write simple one-factor trend PNGs next to the JSON/CSV outputs.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    points = _points_for_preset(args.preset, etl_delta_step=args.etl_delta_step)
    results = run_sweep(
        args.config,
        points,
        N_grid=args.N_grid,
        n_points=args.n_points,
        v_rate=args.v_rate,
        V_max=args.V_max,
        rtol=args.rtol,
        atol=args.atol,
        max_points=args.max_points,
        sync_vbi=not args.keep_config_vbi,
    )
    json_path = args.out_dir / "device_parameter_sweep.json"
    csv_path = args.out_dir / "device_parameter_sweep.csv"
    write_results_json(results, json_path)
    write_results_csv(results, csv_path)

    plot_paths: list[str] = []
    if args.plots:
        plot_paths = write_summary_plots(results, args.out_dir / "figures")

    summary = results["summary"]
    print(
        f"Executed {results['settings']['executed_points']} / "
        f"{results['settings']['requested_points']} points: "
        f"{summary['succeeded']} succeeded, {summary['failed']} failed"
    )
    print(f"JSON: {json_path}")
    print(f"CSV:  {csv_path}")
    if plot_paths:
        print(f"Plots: {args.out_dir / 'figures'} ({len(plot_paths)} files)")
    return 1 if summary["failed"] else 0


def _points_for_preset(preset: str, *, etl_delta_step: float):
    if preset == "pilot":
        return make_pilot_points()
    if preset == "coupled":
        return make_coupled_points()
    if preset == "defect-matrix":
        return make_defect_matrix_points()
    if preset == "full-one-factor":
        return make_full_one_factor_points(etl_delta_step=etl_delta_step)
    raise ValueError(f"unknown preset {preset!r}")


if __name__ == "__main__":
    raise SystemExit(main())
