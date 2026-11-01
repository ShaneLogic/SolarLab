"""One bounded no-grain2D J-V wave from an accepted saved illuminated state.

    .venv/bin/python -m examples.research_2d.jv solve --initial-result Steady.json \
        --initial-sha256 SHA --output JV.json
    .venv/bin/python -m examples.research_2d.jv read --result JV.json --sha256 SHA

The51 points are steady voltage continuation, not a transient or hysteresis
experiment. PV metrics are sampled/interpolated. Efficiency requires explicit
--incident-power-W-m2; no irradiance is inferred from the generation profile.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from examples.research_1d.steady import load_device
from examples.research_2d.dc_assembly import spatial_input
from solarlab.device.compile_2d import compile_device_2d
from solarlab.experiments.jv.steady_2d import (
    SteadyJV2DAttempt, SteadyJV2DOptions, SteadyJV2DResult, load_steady_jv_2d_initial,
    continue_steady_jv_2d, read_steady_jv_2d, run_steady_jv_2d, write_steady_jv_2d,
)


def summary(record: dict[str, Any]) -> dict[str, Any]:
    result = record["result"]
    linked = record.get("continuation")
    offset = 0 if linked is None else linked["tail_index_offset"]
    return {"completed": record["completed"], "requested_points": len(result["requested_voltage_V"]) if linked is None
            else len(linked["original_requested_voltage_V"]),
        "accepted_points": len(result["voltage_V"]) + offset, "unattempted_points": len(record["unattempted_voltage_V"]),
        "metrics": result["metrics"] if linked is None else linked["metrics"],
        "incident_power_W_m2": result["incident_power_W_m2"],
        "compiled_identity": result["compiled_identity"], "solver_calls": result["solver_calls"] if linked is None
            else linked["total_actual_solver_calls"], "new_tail_solver_calls": None if linked is None else result["solver_calls"],
        "preserved_prefix_failure": None if linked is None else linked["preserved_prefix_failure"],
        "preserved_earlier_failure": None if linked is None else linked.get("preserved_earlier_failure"),
        "wall_s": result["wall_s"], "first_failure": next(({
            "index": item["index"] + offset, "voltage_V": item["voltage_V"], "termination": item["termination"], "error": item["error"]}
            for item in result["attempts"] if not item["accepted"]), None), "qualification": False}


def run_case(initial_path: Path, initial_sha256: str, output: Path,
             *, incident_power_W_m2: float | None = None,
             absolute_deadline_s: float | None = None) -> SteadyJV2DResult:
    if output.exists():
        raise FileExistsError("preserve the original2D J-V wave; output already exists")
    device = compile_device_2d(load_device(), spatial_input())
    controls = SteadyJV2DOptions()
    initial = load_steady_jv_2d_initial(device, initial_path, initial_sha256, options=controls.steady)
    def progress(attempt: SteadyJV2DAttempt) -> None:
        if attempt.index == 2 or not attempt.accepted:
            print(json.dumps({"same_wave_checkpoint": attempt.index + 1, "voltage_V": attempt.voltage_V,
                "accepted": attempt.accepted, "termination": attempt.termination, "error": attempt.error}), flush=True)
    result = run_steady_jv_2d(device, initial, np.linspace(0.0, 1.25, 51), options=controls,
        incident_power_W_m2=incident_power_W_m2, on_point=progress,
        checkpoint_directory=output.with_name(output.stem + ".checkpoints"),
        absolute_deadline_s=absolute_deadline_s)
    output.parent.mkdir(parents=True, exist_ok=True)
    digest = write_steady_jv_2d(result, output)
    print(json.dumps({"result_path": str(output.resolve()), "result_sha256": digest,
        **summary(read_steady_jv_2d(output, digest))}, indent=2, allow_nan=False), flush=True)
    return result


def continue_case(prefix_path: Path, prefix_sha256: str, output: Path,
                  *, absolute_deadline_s: float | None = None) -> SteadyJV2DResult:
    """One explicitly authorized remaining tail, preserving its failed prefix."""
    if output.exists():
        raise FileExistsError("preserve the existing2D tail wave")
    device = compile_device_2d(load_device(), spatial_input())
    result = continue_steady_jv_2d(device, prefix_path, prefix_sha256,
        checkpoint_directory=output.with_name(output.stem + ".checkpoints"),
        absolute_deadline_s=absolute_deadline_s)
    output.parent.mkdir(parents=True, exist_ok=True)
    digest = write_steady_jv_2d(result, output)
    print(json.dumps({"result_path": str(output.resolve()), "result_sha256": digest,
        **summary(read_steady_jv_2d(output, digest))}, indent=2, allow_nan=False), flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    solve = commands.add_parser("solve")
    solve.add_argument("--initial-result", type=Path, required=True)
    solve.add_argument("--initial-sha256", required=True)
    solve.add_argument("--output", type=Path, required=True)
    solve.add_argument("--incident-power-W-m2", type=float)
    solve.add_argument("--absolute-deadline-s", type=float)
    tail = commands.add_parser("continue", help="explicit recovery of an unchanged accepted wall-stopped prefix")
    tail.add_argument("--prefix-result", type=Path, required=True)
    tail.add_argument("--prefix-sha256", required=True)
    tail.add_argument("--output", type=Path, required=True)
    tail.add_argument("--absolute-deadline-s", type=float)
    read = commands.add_parser("read")
    read.add_argument("--result", type=Path, required=True)
    read.add_argument("--sha256", required=True)
    args = parser.parse_args()
    if args.command == "read":
        record = read_steady_jv_2d(args.result, args.sha256)
        print(json.dumps(summary(record), indent=2, allow_nan=False))
        return 0 if record["completed"] else 2
    if args.command == "continue":
        result = continue_case(args.prefix_result, args.prefix_sha256, args.output,
                               absolute_deadline_s=args.absolute_deadline_s)
        return 0 if result.completed else 2
    result = run_case(args.initial_result, args.initial_sha256, args.output,
                      incident_power_W_m2=args.incident_power_W_m2,
                      absolute_deadline_s=args.absolute_deadline_s)
    return 0 if result.completed else 2


if __name__ == "__main__":
    raise SystemExit(main())
