#!/usr/bin/env python
"""Generate SolarLab YAML inputs from SolarScale MaterialRecord JSON."""

from __future__ import annotations

import argparse
from pathlib import Path

from perovskite_sim.screening.solarscale import generate_solarlab_inputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True, type=Path, help="SolarScale material_records.json")
    parser.add_argument(
        "--template",
        type=Path,
        default=Path("configs/nip_MAPbI3.yaml"),
        help="SolarLab YAML template used for contacts/transport layers",
    )
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory")
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of configs")
    parser.add_argument(
        "--sweep-policy",
        choices=("quick", "exploratory", "production"),
        default="quick",
        help="Device-level unknown sweep grid recorded in the generated manifest.",
    )
    parser.add_argument(
        "--import-policy",
        choices=("production", "exploratory"),
        default="production",
        help="production requires promising records; exploratory also accepts phonon-ready records",
    )
    parser.add_argument(
        "--activate-bandgap",
        action="store_true",
        help=(
            "Map band_gap_hse_ev into absorber Eg. This requires a fully "
            "band-aligned template with chi and positive Eg on every electrical layer."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = generate_solarlab_inputs(
        args.records,
        template_path=args.template,
        out_dir=args.out_dir,
        limit=args.limit,
        import_policy=args.import_policy,
        sweep_policy=args.sweep_policy,
        activate_bandgap=args.activate_bandgap,
    )
    print(
        f"Generated {len(manifest['generated'])} SolarLab configs; "
        f"skipped {len(manifest['skipped'])}; manifest: {args.out_dir / 'manifest.json'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
