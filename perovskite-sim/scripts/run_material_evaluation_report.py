#!/usr/bin/env python
"""Generate a SolarScale x SolarLab material evaluation report pack."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from perovskite_sim.screening.evaluation_report import run_material_evaluation_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Generated SolarLab YAML config")
    parser.add_argument("--material-record", required=True, type=Path, help="SolarScale material_records.json")
    parser.add_argument("--device-results", required=True, type=Path, help="SolarLab device_results or sweep_device_results JSON")
    parser.add_argument("--out-dir", required=True, type=Path, help="Report output directory")
    parser.add_argument("--material-id", default=None, help="Override material id when records contain multiple materials")
    parser.add_argument(
        "--profile",
        choices=("smoke", "diagnostic", "production"),
        default="diagnostic",
        help="Report quality profile; production enforces physics gates.",
    )
    parser.add_argument("--quick", action="store_true", help="Compatibility alias for --profile smoke")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = run_material_evaluation_report(
        config_path=args.config,
        material_record_path=args.material_record,
        device_results_path=args.device_results,
        out_dir=args.out_dir,
        quick=args.quick,
        profile=args.profile,
        material_id=args.material_id,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
