"""CLI entrypoint for R1 short-window verification (`python -m solarlab_research.scripts.r1_short_window`)."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path
import sys
from typing import Any

from solarlab_research.r1.contracts import R1_CONTROLS
from solarlab_research.r1.controls import verify_r1_controls_abcd
from solarlab_research.r1.provenance import (
    load_verified_release_bundle,
    synthetic_research_release_bundle,
)
from solarlab_research.r1.short_window import (
    evaluate_r1_witness_minimal_input,
    run_r1_short_window_pairing,
)

__all__ = ["build_parser", "main"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m solarlab_research.scripts.r1_short_window",
        description=(
            "Run conservative 4/8/16 short-window baseline-vs-compensated R1 public-interface "
            "verification without claiming R1-L2 (100 s) or R1-L3 device qualification."
        ),
    )
    parser.add_argument(
        "--control",
        choices=tuple(R1_CONTROLS),
        default="D",
        help="R1 control identifier A:(0,0), B:(1,0), C:(0,1), or D:(1,1).",
    )
    parser.add_argument(
        "--intervals",
        type=int,
        nargs=2,
        default=[4, 3],
        help="Intervals per layer for the 2-layer absorber/ETL R1 stack.",
    )
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=None,
        help="Optional materialized ReleaseBundle directory.",
    )
    parser.add_argument(
        "--expected-bundle-sha256",
        type=str,
        default=None,
        help="Expected SHA-256 hex digest when --bundle-dir is supplied.",
    )
    parser.add_argument(
        "--check-controls",
        action="store_true",
        help="Also run and verify controls A-D 0- anchor replication and suppression.",
    )
    parser.add_argument(
        "--witness",
        type=Path,
        default=None,
        help="Optional path to R1FailureWitnessV1.json for minimal 183/309 public check.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.bundle_dir is not None:
        if not args.expected_bundle_sha256:
            raise ValueError("--expected-bundle-sha256 is required with --bundle-dir")
        bundle = load_verified_release_bundle(
            args.bundle_dir,
            expected_bundle_sha256=args.expected_bundle_sha256,
        )
    else:
        bundle = synthetic_research_release_bundle()

    intervals = (int(args.intervals[0]), int(args.intervals[1]))
    pairing = run_r1_short_window_pairing(
        bundle=bundle,
        control=str(args.control),
        intervals_per_layer=intervals,
    )
    payload: dict[str, Any] = {
        "schema": "solarlab_research.r1_short_window_cli.v1",
        "control": pairing.request.control,
        "baseline_passed": pairing.baseline.certificate.passed,
        "compensated_passed": pairing.compensated.certificate.passed,
        "baseline_rows": pairing.baseline.accepted_rows,
        "compensated_rows": pairing.compensated.accepted_rows,
        "sub_ulp_precision_ratio": pairing.sub_ulp_precision_ratio,
        "deterministic_replay_match": pairing.replay_receipt.deterministic_match,
        "request_sha256": pairing.replay_receipt.request_sha256,
        "release_bundle_sha256": pairing.replay_receipt.release_bundle_sha256,
        "qualification_level": pairing.qualification.qualification_level,
        "r1_l2_certified": pairing.qualification.r1_l2_certified,
        "r1_l3_device_qualified": pairing.qualification.r1_l3_device_qualified,
    }
    if args.check_controls:
        suite = verify_r1_controls_abcd(intervals_per_layer=intervals)
        payload["controls_abcd_passed"] = suite.passed
    if args.witness is not None:
        witness_eval = evaluate_r1_witness_minimal_input(args.witness, intervals_per_layer=intervals)
        payload["witness_minimal_evaluation"] = {
            "public_step183_accepted": witness_eval.public_step183_accepted,
            "public_step309_accepted": witness_eval.public_step309_accepted,
            "historical_step309_status": witness_eval.historical_step309_status,
            "relabelled": witness_eval.relabelled,
        }
    sys.stdout.write(json.dumps(payload, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
