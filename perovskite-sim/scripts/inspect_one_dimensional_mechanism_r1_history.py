#!/usr/bin/env python3
"""Inspect a caller-selected frozen historical producer; never grant V3 acceptance."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from perovskite_sim.experiments.one_dimensional_mechanism_r1_history import inspect_historical_evidence


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path, help="existing historical evidence directory")
    parser.add_argument("--producer-project", required=True, type=Path, help="caller-selected clean frozen checkout")
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--prepared-manifest-sha256")
    parser.add_argument("--python-executable", required=True, type=Path)
    parser.add_argument("--dependency-path", required=True, type=Path)
    parser.add_argument("--kind", choices=("stage-one", "physics-study"), default="stage-one")
    parser.add_argument("--timeout-s", type=float, default=300.)
    args = parser.parse_args(argv)
    if not 0 < args.timeout_s <= 3600:
        parser.error("--timeout-s must be positive and at most 3600 seconds")
    try:
        report = inspect_historical_evidence(args.output_dir, expected_source_commit=args.source_commit,
            producer_project=args.producer_project, expected_manifest_sha256=args.manifest_sha256,
            python_executable=args.python_executable, dependency_path=args.dependency_path,
            expected_prepared_manifest_sha256=args.prepared_manifest_sha256, kind=args.kind,
            timeout_s=args.timeout_s)
    except (ValueError, OSError) as exc:
        print("HISTORICAL INSPECTION STOPPED: " + str(exc), file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
