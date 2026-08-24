#!/usr/bin/env python3
"""Run the synthetic production-formula interface-SRH identifiability slice."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from perovskite_sim.experiments.identifiability import (  # noqa: E402
    OBSERVABLE_FAMILIES,
    PARAMETER_NAMES,
    IdentifiabilityError,
    InterfaceSRHIdentifiabilityProtocol,
    build_interface_srh_identifiability_protocol,
    run_interface_srh_identifiability,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        help="strict protocol JSON; cannot be combined with builder overrides",
    )
    parser.add_argument(
        "--observable-family",
        choices=OBSERVABLE_FAMILIES,
    )
    parser.add_argument(
        "--estimated-parameters",
        nargs="+",
        choices=PARAMETER_NAMES,
    )
    parser.add_argument("--carrier-condition-count", type=int)
    parser.add_argument("--finite-difference-step-log10", type=float)
    parser.add_argument("--synthetic-noise-sigma-multiplier", type=float)
    parser.add_argument("--noise-seed", type=int)
    return parser


def _builder_overrides(args: argparse.Namespace) -> dict:
    values = {
        "observable_family": args.observable_family,
        "estimated_parameters": args.estimated_parameters,
        "carrier_condition_count": args.carrier_condition_count,
        "finite_difference_step_log10": args.finite_difference_step_log10,
        "synthetic_noise_sigma_multiplier": args.synthetic_noise_sigma_multiplier,
        "noise_seed": args.noise_seed,
    }
    return {key: value for key, value in values.items() if value is not None}


def _load_protocol(args: argparse.Namespace) -> InterfaceSRHIdentifiabilityProtocol:
    overrides = _builder_overrides(args)
    if args.protocol is not None:
        if overrides:
            raise ValueError(
                "--protocol cannot be combined with protocol builder overrides"
            )
        payload = json.loads(args.protocol.read_text(encoding="utf-8"))
        return InterfaceSRHIdentifiabilityProtocol.from_dict(payload)
    return build_interface_srh_identifiability_protocol(**overrides)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        protocol = _load_protocol(args)
        result = run_interface_srh_identifiability(protocol)
        payload = {
            "analysis_type": "synthetic_interface_srh_identifiability",
            "result": result.to_dict(),
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    except (
        IdentifiabilityError,
        json.JSONDecodeError,
        OSError,
        TypeError,
        ValueError,
    ) as exc:
        print(f"identifiability error: {exc}", file=sys.stderr)
        return 4

    summary = {
        "analysis_certified": result.analysis_certified,
        "mapping_sha256": result.mapping_sha256,
        "numerical_rank": result.numerical_rank,
        "output": str(args.out),
        "parameter_count": len(result.protocol.estimated_parameters),
        "parameters_identifiable": result.parameters_identifiable,
        "protocol_sha256": result.protocol_sha256,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if result.analysis_certified else 2


if __name__ == "__main__":
    raise SystemExit(main())
