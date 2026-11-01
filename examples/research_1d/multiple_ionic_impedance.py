"""One two-domain AC sweep from checksum-bound accepted DC; no DC solve.

    .venv/bin/python -m examples.research_1d.multiple_ionic_impedance solve \
        --initial-result Steady.json --initial-sha256 SHA --output AC.json
    .venv/bin/python -m examples.research_1d.multiple_ionic_impedance read \
        --result AC.json --sha256 SHA

The initial scope is the declared 13-electrical/4+4-positive-ion dark device.
Both ion domains are explicitly blocking and keep their independent physical
inventories. Phasors use exp(+i omega t), with no scientific qualification.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from examples.research_1d.ionic_steady import controls as steady_controls
from examples.research_1d.multiple_ionic_domains import DOMAINS, INTERVALS, prepare_multiple_ionic_device
from solarlab.device.compile_multiple_ionic_1d import compile_multiple_ionic_device_1d
from solarlab.experiments.ionic_impedance import IonicImpedanceOptions
from solarlab.experiments.multiple_ionic_impedance import (
    MultipleIonicImpedanceResult, read_multiple_ionic_impedance_result,
    solve_multiple_ionic_impedance, write_multiple_ionic_impedance_result,
)
from solarlab.experiments.multiple_ionic_transient import load_multiple_ionic_steady_initial


def summary(record: dict[str, Any]) -> dict[str, Any]:
    result = record["result"]
    return {"completed": record["completed"], "compiled_identity": result["compiled_identity"],
        "initial_provenance": {key: result["initial_provenance"][key] for key in ("source_path", "source_sha256", "state_sha256")},
        "linear_solver_calls": result["linear_solver_calls"], "wall_s": result["wall_s"],
        "points": [{"frequency_Hz": attempt["frequency_Hz"], "accepted": attempt["accepted"],
            "termination": attempt["termination"], "error": attempt["error"],
            "checks": attempt["checks"]} for attempt in result["attempts"]],
        "physical_certificate_status": result["physical_certificate_status"], "qualification": False}


def run_case(initial_path: Path, initial_sha256: str, output: Path,
             *, frequencies_Hz: tuple[float, ...] = (0.0, .01, 1.0, 1000.0),
             voltage_phasor_V: complex = 2e-6 + 1e-6j) -> MultipleIonicImpedanceResult:
    if output.exists():
        raise FileExistsError("preserve the original AC attempt; output already exists")
    device = compile_multiple_ionic_device_1d(prepare_multiple_ionic_device(),
        intervals_per_layer=INTERVALS, ion_domains=DOMAINS)
    initial = load_multiple_ionic_steady_initial(device, initial_path, initial_sha256, controls=steady_controls(4))
    result = solve_multiple_ionic_impedance(device, initial, frequencies_Hz,
        voltage_phasor_V=voltage_phasor_V, options=IonicImpedanceOptions())
    output.parent.mkdir(parents=True, exist_ok=True)
    digest = write_multiple_ionic_impedance_result(result, output)
    print(json.dumps({"result_path": str(output.resolve()), "result_sha256": digest,
        **summary(read_multiple_ionic_impedance_result(output, digest))}, indent=2, allow_nan=False), flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    solve = commands.add_parser("solve")
    solve.add_argument("--initial-result", type=Path, required=True)
    solve.add_argument("--initial-sha256", required=True)
    solve.add_argument("--output", type=Path, required=True)
    solve.add_argument("--frequencies", type=float, nargs="+", default=(0.0, .01, 1.0, 1000.0))
    solve.add_argument("--voltage-phasor", type=complex, default=2e-6 + 1e-6j)
    read = commands.add_parser("read")
    read.add_argument("--result", type=Path, required=True)
    read.add_argument("--sha256", required=True)
    args = parser.parse_args()
    if args.command == "read":
        record = read_multiple_ionic_impedance_result(args.result, args.sha256)
        print(json.dumps(summary(record), indent=2, allow_nan=False))
        return 0 if record["completed"] else 2
    result = run_case(args.initial_result, args.initial_sha256, args.output,
                      frequencies_Hz=tuple(args.frequencies), voltage_phasor_V=args.voltage_phasor)
    return 0 if result.completed else 2


if __name__ == "__main__":
    raise SystemExit(main())
