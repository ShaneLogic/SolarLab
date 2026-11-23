"""CLI entrypoint for R1 source & historical artifact verification (`python -m solarlab_research.scripts.r1_verify_sources`)."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path
import sys
from typing import Any

from solarlab_research.r1.provenance import (
    load_verified_release_bundle,
    read_historical_r1_artifact,
    verify_historical_r1_failure_witness,
    verify_r1_release_source,
)

__all__ = ["build_parser", "main"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m solarlab_research.scripts.r1_verify_sources",
        description=(
            "Verify historical R1 artifacts against their original recorded sources "
            "without relabeling, and/or verify an immutable ReleaseBundle directory."
        ),
    )
    parser.add_argument(
        "--historical",
        type=Path,
        action="append",
        default=[],
        help="Path to a historical R1 JSON artifact (R1CommonStateV1/V2 or R1FailureWitnessV1).",
    )
    parser.add_argument(
        "--witness",
        type=Path,
        default=None,
        help="Path to R1FailureWitnessV1.json for explicit witness audit.",
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
        help="Expected SHA-256 hex digest of the ReleaseBundle manifest.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    historical_records: list[dict[str, Any]] = []
    for path in args.historical:
        summary = read_historical_r1_artifact(path)
        historical_records.append({
            "path": str(path),
            "schema": summary.schema,
            "artifact_sha256": summary.artifact_sha256,
            "original_source_id": summary.original_source_id,
            "original_source_sha256": summary.original_source_sha256,
            "relabelled": summary.relabelled,
        })

    payload: dict[str, Any] = {
        "schema": "solarlab_research.r1_verify_sources_cli.v1",
        "historical_artifacts": historical_records,
    }

    if args.witness is not None:
        audit = verify_historical_r1_failure_witness(args.witness)
        payload["witness_audit"] = {
            "fixture_sha256": audit.fixture_sha256,
            "source_commit": audit.source_commit,
            "step183_accepted_status": audit.step183_accepted_status,
            "step183_failed_status": audit.step183_failed_status,
            "step309_original_status": audit.step309_original_status,
            "relabelled": audit.relabelled,
        }

    if args.bundle_dir is not None:
        if not args.expected_bundle_sha256:
            raise ValueError("--expected-bundle-sha256 is required with --bundle-dir")
        bundle = load_verified_release_bundle(
            args.bundle_dir,
            expected_bundle_sha256=args.expected_bundle_sha256,
        )
        binding = verify_r1_release_source(
            bundle,
            expected_bundle_sha256=args.expected_bundle_sha256,
            inputs={"cli": "r1_verify_sources"},
        )
        payload["release_bundle"] = {
            "build_source_commit": binding.build_source_commit,
            "profile": binding.profile,
            "bundle_sha256": binding.bundle_sha256,
            "scientific_qualification": binding.scientific_qualification,
        }

    sys.stdout.write(json.dumps(payload, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
