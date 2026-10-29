"""One bounded neutral-grain2D J-V curve using the shared accepted-state driver.

    .venv/bin/python -m examples.research_2d.grain_jv solve --initial-result GrainSteady.json \
        --initial-sha256 SHA --output JV.json --baseline-result NoGrainJV.json --baseline-sha256 SHA
    .venv/bin/python -m examples.research_2d.grain_jv read --result JV.json --sha256 SHA

The existing0.4um center/0.3um width and0.5/0.25us lifetimes are illustrative
declared inputs. PV quantities are sampled/interpolated; irradiance is not
inferred. The saved no-grain curve is read only and no prior point is replayed.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from examples.research_1d.steady import load_device
from examples.research_2d.grains import spatial_input, run_case as prepare_grain_initial
from examples.research_2d.jv import summary as shared_summary
from solarlab.device.compile_2d import compile_device_2d
from solarlab.experiments.jv.steady_2d import (
    SteadyJV2DAttempt, SteadyJV2DOptions, SteadyJV2DResult, load_steady_jv_2d_initial,
    read_steady_jv_2d, run_steady_jv_2d, write_steady_jv_2d,
)
from solarlab.experiments.transient_results import _encode


def summary(record: dict[str, Any]) -> dict[str, Any]:
    model = record["result"]["geometry"].get("grain_model")
    if model is None:
        raise ValueError("this example requires explicit saved neutral-grain provenance")
    return {**shared_summary(record), "grain_model_id": model["model_id"],
        "grain_declarations": [r["declaration"] for r in model["regions"]],
        "lateral_current_max_A_m2": max((float(np.max(abs(a["solution"]["diagnostics"]["lateral_current_density_A_m2"])))
            for a in record["result"]["attempts"] if a["accepted"]), default=None)}


def compare_saved(grain_path: Path, grain_sha256: str, baseline_path: Path,
                   baseline_sha256: str) -> dict[str, Any]:
    """Compare only hash-bound actual accepted samples; compile/solve nothing."""
    grain, baseline = read_steady_jv_2d(grain_path, grain_sha256), read_steady_jv_2d(baseline_path, baseline_sha256)
    g, b = grain["result"], baseline["result"]
    if "grain_model" not in g["geometry"] or "grain_model" in b["geometry"]:
        raise ValueError("comparison requires explicit grain and no-grain curves")
    if (g["geometry"]["identity"] != b["geometry"]["identity"] or g["generation_scale"] != b["generation_scale"]
            or g["lateral_width_m"] != b["lateral_width_m"] or g["voltage_polarity"] != b["voltage_polarity"]):
        raise ValueError("comparison geometry, generation, sign or per-depth width differs")
    linked = baseline.get("continuation")
    bv = b["voltage_V"] if linked is None else linked["combined_voltage_V"]
    bj = b["output_current_A_m2"] if linked is None else linked["combined_output_current_A_m2"]
    bm = b["metrics"] if linked is None else linked["metrics"]
    common, gi, bi = np.intersect1d(g["voltage_V"], bv, return_indices=True)
    keys = ("short_circuit_current_A_m2", "open_circuit_voltage_V", "maximum_power_W_m2", "fill_factor")
    return {"grain_artifact": {"path": str(grain_path.resolve()), "sha256": grain_sha256},
        "no_grain_artifact": {"path": str(baseline_path.resolve()), "sha256": baseline_sha256},
        "grain_compiled_identity": g["compiled_identity"], "no_grain_compiled_identity": b["compiled_identity"],
        "grain_model": g["geometry"]["grain_model"], "common_accepted_voltage_V": common,
        "grain_output_current_A_m2": g["output_current_A_m2"][gi], "no_grain_output_current_A_m2": bj[bi],
        "delta_current_A_m2": g["output_current_A_m2"][gi] - bj[bi],
        "grain_metrics": g["metrics"], "no_grain_metrics": bm,
        "metric_deltas": {key: None if g["metrics"][key] is None or bm[key] is None else g["metrics"][key] - bm[key]
                          for key in keys}, "new_numerical_solves": 0, "irradiance_W_m2": None,
        "limitation": "illustrative source-bound comparison; partial accepted grids retained; no convergence or causal qualification",
        "qualification": False}


def run_case(initial_path: Path | None, initial_sha256: str | None, output: Path, *,
             baseline_path: Path | None = None, baseline_sha256: str | None = None,
             absolute_deadline_s: float | None = None) -> SteadyJV2DResult:
    if output.exists():
        raise FileExistsError("preserve the original grain curve")
    began = time.monotonic()
    device = compile_device_2d(load_device(), spatial_input())
    controls = SteadyJV2DOptions(wall_s=80.0)
    startup: dict[str, Any] = {"kind": "exact_saved_same_source_initial", "auxiliary_1d_seed_calls": 0,
                               "full_2d_initial_newton_calls": 0}
    if initial_path is None:
        if initial_sha256 is not None:
            raise ValueError("an initial SHA requires an explicit initial path")
        initial_path = output.with_name(output.stem + ".initial.json")
        startup_began = time.monotonic()
        prepared = prepare_grain_initial(initial_path)
        startup = {"kind": "fresh_illuminated_experiment_startup", "path": str(initial_path.resolve()),
            "auxiliary_1d_seed_calls": prepared["auxiliary_1d_seed_calls"],
            "full_2d_initial_newton_calls": prepared["full_2d_newton_calls"],
            "auxiliary_seed_is_2d_completion": False,
            "started_monotonic_s": startup_began, "completed_monotonic_s": time.monotonic(),
            "wall_s": time.monotonic() - startup_began}
        if not prepared["completed"]:
            raise RuntimeError("fresh grain2D startup failed; preserve the original initial receipt and stop")
        initial_sha256 = hashlib.sha256(initial_path.read_bytes()).hexdigest()
    if initial_sha256 is None:
        raise ValueError("saved initial requires its explicit SHA-256")
    initial = load_steady_jv_2d_initial(device, initial_path, initial_sha256, options=controls.steady)
    def progress(attempt: SteadyJV2DAttempt) -> None:
        if attempt.index == 2 or (attempt.index + 1) % 10 == 0 or not attempt.accepted:
            print(json.dumps({"same_wave_point": attempt.index + 1, "voltage_V": attempt.voltage_V,
                "accepted": attempt.accepted, "termination": attempt.termination}), flush=True)
    curve_began = time.monotonic()
    result = run_steady_jv_2d(device, initial, np.linspace(0.0, 1.25, 51), options=controls,
        on_point=progress, checkpoint_directory=output.with_name(output.stem + ".checkpoints"),
        absolute_deadline_s=absolute_deadline_s)
    curve_completed = time.monotonic()
    output.parent.mkdir(parents=True, exist_ok=True)
    digest = write_steady_jv_2d(result, output)
    record = read_steady_jv_2d(output, digest)
    if baseline_path is not None:
        if baseline_sha256 is None:
            raise ValueError("saved comparison requires its explicit baseline SHA-256")
        comparison = compare_saved(output, digest, baseline_path, baseline_sha256)
        path = output.with_name(output.stem + ".comparison.json")
        with path.open("x") as stream:
            json.dump(_encode(comparison), stream, indent=2, allow_nan=False)
    print(json.dumps({"result_path": str(output.resolve()), "result_sha256": digest,
        "startup": startup, "controls": asdict(controls), **summary(record)}, indent=2, allow_nan=False), flush=True)
    with output.with_name(output.stem + ".timing.json").open("x") as stream:
        json.dump({"startup": startup, "curve_started_monotonic_s": curve_began,
            "curve_completed_monotonic_s": curve_completed, "curve_call_wall_s": curve_completed - curve_began,
            "driver_owned_wall_s": result.wall_s, "example_whole_wall_s": time.monotonic() - began,
            "absolute_compute_deadline_s": absolute_deadline_s, "qualification": False}, stream, indent=2, allow_nan=False)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    solve = commands.add_parser("solve")
    solve.add_argument("--initial-result", type=Path)
    solve.add_argument("--initial-sha256")
    solve.add_argument("--output", type=Path, required=True)
    solve.add_argument("--baseline-result", type=Path)
    solve.add_argument("--baseline-sha256")
    solve.add_argument("--absolute-deadline-s", type=float)
    read = commands.add_parser("read")
    read.add_argument("--result", type=Path, required=True)
    read.add_argument("--sha256", required=True)
    args = parser.parse_args()
    if args.command == "read":
        record = read_steady_jv_2d(args.result, args.sha256)
        print(json.dumps(summary(record), indent=2, allow_nan=False))
        return 0 if record["completed"] else 2
    result = run_case(args.initial_result, args.initial_sha256, args.output,
        baseline_path=args.baseline_result, baseline_sha256=args.baseline_sha256,
        absolute_deadline_s=args.absolute_deadline_s)
    return 0 if result.completed else 2


if __name__ == "__main__":
    raise SystemExit(main())
