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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = generate_solarlab_inputs(
        args.records,
        template_path=args.template,
        out_dir=args.out_dir,
        limit=args.limit,
    )
    print(
        f"Generated {len(manifest['generated'])} SolarLab configs; "
        f"skipped {len(manifest['skipped'])}; manifest: {args.out_dir / 'manifest.json'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
