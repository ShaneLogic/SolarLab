#!/usr/bin/env python3
"""Run or resume one pre-registered Phase-1 numerical refinement lane."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from perovskite_sim.validation.numerical_certificate import (
    NumericalCertificateError,
    load_refinement_registry,
)
from perovskite_sim.validation.refinement_runner import (
    DEFAULT_OUTPUT_ROOT,
    RefinementRunnerError,
    load_executor,
    plan_refinement,
    run_refinement,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = ROOT / "reproducibility/numerical_refinement_registry.yaml"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Execute a content-addressed grid-by-tolerance lane. Existing "
            "cell, manifest, and certificate artifacts are never overwritten."
        )
    )
    parser.add_argument("lane", nargs="?", help="pre-registered lane ID")
    parser.add_argument(
        "--registry",
        type=Path,
        default=DEFAULT_REGISTRY,
        help="lane registry YAML",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="new result namespace; historical references are rejected",
    )
    parser.add_argument(
        "--executor",
        help="explicit module:function adapter for a lane with no registered adapter",
    )
    parser.add_argument(
        "--max-cells",
        type=int,
        help="execute at most this many new cells, leaving a resumable failed certificate",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="retry failed cells while retaining their immutable prior artifacts",
    )
    parser.add_argument(
        "--list-lanes",
        action="store_true",
        help="list lane IDs, adapters, and matrix sizes without executing",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and print the content-addressed run plan without writing",
    )
    parser.add_argument(
        "--allow-noncertified-exit-zero",
        action="store_true",
        help="return zero for partial/failed diagnostic runs; status remains explicit",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    registry_path = args.registry
    if not registry_path.is_absolute():
        registry_path = ROOT / registry_path
    try:
        registry = load_refinement_registry(
            registry_path,
            project_root=ROOT,
        )
        if args.list_lanes:
            listing = [
                {
                    "executor": lane.executor,
                    "lane_id": lane.lane_id,
                    "matrix_cells": len(lane.matrix_points),
                }
                for lane in registry.lanes
            ]
            print(json.dumps(listing, indent=2, sort_keys=True))
            return 0
        if not args.lane:
            raise RefinementRunnerError("lane is required unless --list-lanes is used")
        lane = registry.lane(args.lane)
        executor_spec = args.executor or lane.executor
        if executor_spec is None:
            raise RefinementRunnerError(
                f"lane {lane.lane_id!r} has no registered adapter; pass "
                "--executor module:function without changing its pre-registered gates"
            )
        if (
            args.executor is not None
            and lane.executor is not None
            and args.executor != lane.executor
        ):
            raise RefinementRunnerError(
                "a registered adapter cannot be replaced under the same immutable lane ID"
            )
        executor = load_executor(executor_spec)
        if args.dry_run:
            plan = plan_refinement(
                lane,
                executor,
                project_root=ROOT,
                output_root=args.output_root,
                executor_id=executor_spec,
            )
            summary = {
                **plan.to_dict(),
                "max_new_cells": args.max_cells,
                "mode": "dry-run",
                "retry_failed": args.retry_failed,
                "writes_performed": False,
            }
            print(json.dumps(summary, indent=2, sort_keys=True))
            return 0
        outcome = run_refinement(
            lane,
            executor,
            project_root=ROOT,
            output_root=args.output_root,
            executor_id=executor_spec,
            max_cells=args.max_cells,
            retry_failed=args.retry_failed,
        )
    except (KeyError, NumericalCertificateError, RefinementRunnerError) as exc:
        print(f"numerical refinement error: {exc}", file=sys.stderr)
        return 4

    summary = {
        "certificate": str(outcome.certificate_path),
        "certificate_sha256": outcome.certificate.certificate_sha256,
        "executed_cells": outcome.executed_cells,
        "failed_cells": list(outcome.certificate.failed_cells),
        "manifest": str(outcome.manifest_path),
        "missing_cells": list(outcome.certificate.missing_cells),
        "reused_cells": outcome.reused_cells,
        "run_directory": str(outcome.run_directory),
        "run_id": outcome.certificate.run_id,
        "status": outcome.certificate.status,
        "unconverged_dimensions": list(outcome.certificate.unconverged_dimensions),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.allow_noncertified_exit_zero or outcome.certificate.status == "certified":
        return 0
    return 2 if outcome.certificate.status == "partial" else 3


if __name__ == "__main__":
    raise SystemExit(main())
