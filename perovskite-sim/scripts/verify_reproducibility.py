#!/usr/bin/env python3
"""Validate the historical P0 source and 52-preset benchmark matrix.

Requires the checkout before the 2026-09-05 preset deletion. For the current
scope, run tests/reproducibility/test_research_presets.py instead.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from perovskite_sim.reproducibility import validate_matrix, verify_baseline  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default="p0-certified-2026-08-01")
    parser.add_argument(
        "--check-p0-worktree", action="store_true",
        help="also require current P0-owned files to equal the frozen snapshot",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = {
        "baseline": verify_baseline(
            args.baseline, check_worktree=args.check_p0_worktree,
        ),
        "matrix": validate_matrix(),
    }
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        baseline = report["baseline"]
        matrix = report["matrix"]
        print(
            f"baseline {baseline['baseline_id']}: reconstructed from "
            f"{baseline['base_commit']}, {baseline['checked_files']} frozen files "
            f"matched; {baseline['worktree_checked_files']} worktree files checked"
        )
        print(
            f"matrix: {matrix['configs']} configs, "
            f"{matrix['resources']} resources, {matrix['benchmarks']} benchmarks, "
            f"schemas={matrix['schemas']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
