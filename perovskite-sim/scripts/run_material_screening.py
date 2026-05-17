#!/usr/bin/env python
"""Plan or run first-pass SolarLab screening from SolarScale records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from perovskite_sim.screening.solarscale import (
    generate_solarlab_inputs,
    plan_solarlab_import,
    run_smoke_jv,
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
        )
        plan_path = args.out_dir / "screening_plan.json"
        plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8")
        print(
            f"Selected {plan['selected_count']} candidates; skipped {plan['skipped_count']}; "
            f"plan: {plan_path}"
        )
        return 0

    manifest = generate_solarlab_inputs(
        args.records,
        template_path=args.base_config,
        out_dir=args.out_dir,
        limit=args.top_n,
        import_policy=args.policy,
    )
    print(
        f"Generated {len(manifest['generated'])} SolarLab configs; "
        f"skipped {len(manifest['skipped'])}; manifest: {args.out_dir / 'manifest.json'}"
    )
    if args.run_smoke and manifest["generated"]:
        config_path = manifest["generated"][0]["config_path"]
        smoke = run_smoke_jv(
            config_path,
            N_grid=args.smoke_n_grid,
            n_points=args.smoke_n_points,
            v_rate=args.smoke_v_rate,
            V_max=args.smoke_v_max,
        )
        smoke_path = args.out_dir / "smoke_jv.json"
        smoke_path.write_text(json.dumps(smoke, indent=2, sort_keys=True), encoding="utf-8")
        print(f"Smoke JV complete: {smoke_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
