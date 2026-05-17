#!/usr/bin/env python
"""Plan or run first-pass SolarLab screening from SolarScale records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from perovskite_sim.screening.solarscale import (
    generate_solarlab_inputs,
    plan_solarlab_import,
    run_smoke_device_results,
    write_device_results,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True, type=Path, help="SolarScale material_records.json")
    parser.add_argument(
        "--policy",
        "--import-policy",
        choices=("production", "exploratory"),
        default="exploratory",
        dest="policy",
        help="exploratory accepts phonon/promising records; production requires promising records",
    )
    parser.add_argument(
        "--base-config",
        "--template",
        type=Path,
        default=Path("configs/nip_MAPbI3.yaml"),
        dest="base_config",
        help="SolarLab YAML template used for contacts/transport layers",
    )
    parser.add_argument("--top-n", "--limit", type=int, default=None, dest="top_n")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write only an import plan JSON; do not generate YAML configs or run JV",
    )
    parser.add_argument(
        "--activate-bandgap",
        action="store_true",
        help=(
            "Map band_gap_hse_ev into absorber Eg. This requires a fully "
            "band-aligned template with chi and positive Eg on every electrical layer."
        ),
    )
    parser.add_argument(
        "--run-smoke",
        action="store_true",
        help="After generating YAML configs, run a tiny JV smoke check on the top candidate",
    )
    parser.add_argument("--smoke-n-grid", type=int, default=12)
    parser.add_argument("--smoke-n-points", type=int, default=4)
    parser.add_argument("--smoke-v-rate", type=float, default=5.0)
    parser.add_argument("--smoke-v-max", type=float, default=0.2)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        plan = plan_solarlab_import(
            args.records,
            template_path=args.base_config,
            import_policy=args.policy,
            limit=args.top_n,
            include_configs=False,
            activate_bandgap=args.activate_bandgap,
        )
        plan_path = args.out_dir / "screening_plan.json"
        plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8")
        print(
            f"Selected {plan['selected_count']} candidates; skipped {plan['skipped_count']}; "
            f"plan: {plan_path}"
        )
        _print_dry_run_summary(plan)
        return 0

    manifest = generate_solarlab_inputs(
        args.records,
        template_path=args.base_config,
        out_dir=args.out_dir,
        limit=args.top_n,
        import_policy=args.policy,
        activate_bandgap=args.activate_bandgap,
    )
    print(
        f"Generated {len(manifest['generated'])} SolarLab configs; "
        f"skipped {len(manifest['skipped'])}; manifest: {args.out_dir / 'manifest.json'}"
    )
    if args.run_smoke and manifest["generated"]:
        device_results = run_smoke_device_results(
            manifest,
            N_grid=args.smoke_n_grid,
            n_points=args.smoke_n_points,
            v_rate=args.smoke_v_rate,
            V_max=args.smoke_v_max,
            max_configs=1,
        )
        smoke_path = args.out_dir / "smoke_jv.json"
        smoke_record = device_results["records"][0] if device_results["records"] else {}
        smoke_payload = smoke_record.get("smoke_result") or {
            "simulation_status": smoke_record.get("simulation_status"),
            "error": smoke_record.get("error"),
        }
        smoke_path.write_text(json.dumps(smoke_payload, indent=2, sort_keys=True), encoding="utf-8")
        print(f"Smoke JV complete: {smoke_path}")
        device_json_path = args.out_dir / "device_results.json"
        device_csv_path = args.out_dir / "device_results.csv"
        write_device_results(
            device_results,
            json_path=device_json_path,
            csv_path=device_csv_path,
        )
        print(f"Device results complete: {device_json_path}; {device_csv_path}")
        failed = device_results.get("summary", {}).get("status_counts", {}).get("failed", 0)
        if failed:
            return 1
    return 0


def _print_dry_run_summary(plan: dict) -> None:
    summary = plan.get("summary", {})
    readiness = summary.get("readiness_distribution", {})
    gate_summary = summary.get("gate_summary", {})
    skipped_reasons = summary.get("skipped_reason_counts", {})
    top_selected = summary.get("top_selected_candidates", [])

    if readiness:
        print("Readiness distribution:")
        for name, count in readiness.items():
            print(f"  {name}: {count}")
    totals = gate_summary.get("totals", {})
    if totals:
        print(
            "Gate totals: "
            f"pass={totals.get('pass', 0)}, "
            f"fail={totals.get('fail', 0)}, "
            f"missing={totals.get('missing', 0)}, "
            f"unknown={totals.get('unknown', 0)}"
        )
    if top_selected:
        print("Top selected candidates:")
        for item in top_selected[:10]:
            source = item.get("ranking_score_source") or "none"
            score = item.get("ranking_score")
            score_text = "n/a" if score is None else f"{score:.6g}"
            print(
                f"  #{item.get('rank')} {item.get('material_id')} "
                f"readiness={item.get('readiness')} score={score_text} source={source}"
            )
    if skipped_reasons:
        print("Skipped reasons:")
        for reason, count in skipped_reasons.items():
            print(f"  {count}x {reason}")


if __name__ == "__main__":
    raise SystemExit(main())
