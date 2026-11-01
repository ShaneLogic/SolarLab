"""Bounded illuminated shared-site ionic J-V samples and explicit continuations.

Supply an accepted G1 same-device steady artifact explicitly:
    python -m examples.research_1d.ionic_pv solve \
      --initial-result ACCEPTED.json --initial-sha256 SHA --output CURVE.json

``solve`` runs one fixed 51-point grid, 0 to 1.25 V in 25 mV increments.
For historical reproduction, ``coarse-probe`` uses the original 17-point grid;
``continue`` and ``tail`` use accepted parents at 0.2 V and 1.2 V.
These are research samples, without mesh/grid qualification.
Irradiance is undeclared; generation_scale=1 does not supply an incident power.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from examples.research_1d.ionic_jv import compile_case
from examples.research_1d.ionic_steady import controls
from examples.research_1d.transient import json_value
from solarlab.equations.ionic_1d import CompiledIonicDevice1D
from solarlab.experiments.ionic_transient import IonicTransientInitial, load_ionic_steady_initial
from solarlab.experiments.jv.steady import steady_jv_metrics
from solarlab.experiments.transient_results import _encode
from solarlab.solvers.nonlinear.ionic_1d import IonicSteadyOptions
from solarlab.solvers.nonlinear.problem import SolveBudget
from solarlab.solvers.nonlinear.steady_1d import SteadyOptions
from solarlab.experiments.jv.ionic_steady import (
    IonicSteadyJVOptions, read_ionic_steady_jv, run_ionic_steady_jv, write_ionic_steady_jv,
)

VOLTAGES_V = tuple(tick / 40.0 for tick in range(51))
COARSE_PROBE_VOLTAGES_V = (0.0, 0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5,
                         0.6, 0.7, 0.8, 0.9, 1.0, 1.05, 1.1, 1.2)
GENERATION_SCALE = 1.0
WAVE_WALL_S = 25.0
CONTINUATION_VOLTAGES_V = tuple(tick / 40.0 for tick in range(9, 49))
TAIL_VOLTAGES_V = (1.225, 1.25)


def export_parent_initial(
    device: CompiledIonicDevice1D, path: Path, sha256: str, index: int, output: Path,
    *, expected_voltage_V: float = 0.2,
) -> tuple[IonicTransientInitial, dict[str, Any], str]:
    """Export a genuine accepted point, retaining its actual effective controls."""
    record = read_ionic_steady_jv(path, sha256)
    parent = record["result"]
    if (not 0 <= index < len(parent["attempts"]) or not parent["attempts"][index]["accepted"]
            or index != len(parent["voltage_V"]) - 1 or parent["voltage_V"][-1] != expected_voltage_V
            or parent["compiled_identity"] != device.identity or parent["generation_scale"] != 1.0):
        raise ValueError(f"continuation requires the last accepted same-device G1 point at {expected_voltage_V} V")
    point = parent["attempts"][index]["solution"]
    config = dict(point["solver_controls"])
    canonical = controls(device.ion_node_count)
    if ({**config, "wall_s": canonical.wall_s} != asdict(canonical)
            or not 0 < config["wall_s"] <= canonical.wall_s):
        raise ValueError("the recorded physical controls differ from this example's original controls")
    config["electronic"] = SteadyOptions(**config["electronic"])
    config["budget"] = SolveBudget(**config["budget"])
    settings = IonicSteadyOptions(**config)
    document = {"summary": {"accepted": True, "compiled_identity": device.identity,
        "voltage_V": expected_voltage_V, "generation_scale": 1.0, "parent_curve_path": str(path.resolve()),
        "parent_curve_sha256": sha256, "parent_attempt_index": index,
        "initial_policy": "exact_accepted_parent_point", "qualification": False}, "result": point}
    raw = (json.dumps(json_value(document), indent=2, allow_nan=False) + "\n").encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(raw)
    digest = hashlib.sha256(raw).hexdigest()
    initial = load_ionic_steady_initial(device, output, digest, controls=settings)
    if _encode(initial.state) != _encode(point["state"]):
        raise ValueError("export changed the accepted parent state words")
    return initial, record, digest


def read_joined_pv(
    parent_path: Path, parent_sha256: str, path: Path, sha256: str, index: int,
) -> dict[str, Any]:
    """Join verified accepted samples, retaining the separate failed parent wave."""
    parent_record = read_ionic_steady_jv(parent_path, parent_sha256)
    continued_record = read_ionic_steady_jv(path, sha256)
    parent, continued = parent_record["result"], continued_record["result"]
    if not 0 <= index < len(parent["attempts"]) or not parent["attempts"][index]["accepted"]:
        raise ValueError("joined data requires an accepted parent point")
    initial = continued["initial_provenance"]
    seed_raw = Path(initial["source_path"]).read_bytes()
    if hashlib.sha256(seed_raw).hexdigest() != initial["source_sha256"]:
        raise ValueError("continuation initial artifact SHA-256 mismatch")
    seed = json.loads(seed_raw)
    point = parent["attempts"][index]["solution"]
    if (seed["summary"]["parent_curve_sha256"] != parent_sha256
            or seed["summary"]["parent_attempt_index"] != index
            or index != len(parent["voltage_V"]) - 1
            or _encode(initial["state"]) != _encode(point["state"])
            or any(parent[key] != continued[key] for key in
                   ("compiled_identity", "source_identity", "generation_scale", "incident_power_W_m2"))
            or continued["requested_voltage_V"][0] <= parent["voltage_V"][-1]
            or continued["solver_controls"]["steady"] != point["solver_controls"]):
        raise ValueError("continuation parent/state/source/controls binding mismatch")
    voltage = np.r_[parent["voltage_V"], continued["voltage_V"]]
    current = np.r_[parent["output_current_A_m2"], continued["output_current_A_m2"]]
    metrics = asdict(steady_jv_metrics(voltage, current, incident_power_W_m2=parent["incident_power_W_m2"]))
    return {"kind": "joined_accepted_steady_samples", "parent_curve_sha256": parent_sha256,
        "continuation_curve_sha256": sha256, "parent_attempt_index": index,
        "initial_export_sha256": initial["source_sha256"],
        "parent_completed": parent_record["completed"], "continuation_completed": continued_record["completed"],
        "parent_first_failure": pv_summary(parent_record)["first_failure"],
        "continuation_first_failure": pv_summary(continued_record)["first_failure"],
        "accepted_voltage_V": voltage.tolist(), "output_current_A_m2": current.tolist(), "metrics": metrics,
        "metric_basis": "piecewise-linear interpolation of linked accepted samples only",
        "solver_calls_including_preserved_failures": parent["solver_calls"] + continued["solver_calls"],
        "physical_certificate_status": "not_evaluated", "voltage_grid_qualification": "not_evaluated"}


def pv_summary(record: dict[str, Any]) -> dict[str, Any]:
    """Summarize an already verified record; use accepted data only."""
    result = record["result"]
    voltage = np.asarray(result["voltage_V"])
    current = np.asarray(result["output_current_A_m2"])
    metrics = result["metrics"]
    at_zero = np.flatnonzero(voltage == 0.0)
    jsc_sample = None if at_zero.size == 0 else float(current[at_zero[0]])
    order = np.argsort(voltage)
    v, j = voltage[order], current[order]
    crossings = np.flatnonzero((v[:-1] >= 0.0) & (j[:-1] > 0.0) & (j[1:] <= 0.0))
    bracket = None
    if metrics["open_circuit_voltage_V"] is not None and crossings.size:
        index = int(crossings[0])
        bracket = {"voltage_V": [float(v[index]), float(v[index + 1])],
                   "output_current_A_m2": [float(j[index]), float(j[index + 1])]}
    failed = [attempt for attempt in result["attempts"] if not attempt["accepted"]]
    first_failure = None
    if failed:
        attempt = failed[0]
        solution = attempt["solution"]
        newton = None if solution is None else solution["newton"]
        first_failure = {"index": attempt["index"], "voltage_V": attempt["voltage_V"],
            "termination": attempt["termination"], "error": attempt["error"],
            "solver_called": attempt["solver_called"],
            "evaluation_counts": None if solution is None else solution["evaluation_counts"],
            "iterations": None if newton is None else len(newton["iterations"]),
            "newton_message": None if newton is None else newton["message"]}
    return {"completed": record["completed"], "requested_voltage_V": result["requested_voltage_V"].tolist(),
        "accepted_voltage_V": voltage.tolist(), "output_current_A_m2": current.tolist(),
        "sampled_short_circuit_current_A_m2": jsc_sample,
        "open_circuit_bracket": bracket, "open_circuit_bracket_established": bracket is not None,
        "metrics": metrics, "metric_basis": "piecewise-linear interpolation of accepted voltage samples",
        "incident_power_W_m2": result["incident_power_W_m2"],
        "efficiency_status": ("unknown_incident_power" if result["incident_power_W_m2"] is None
                              else "unavailable_without_bracket" if metrics["efficiency"] is None
                              else "piecewise_linear_voltage_grid"),
        "first_failure": first_failure, "solver_calls": result["solver_calls"],
        "evaluation_counts": result["evaluation_counts"], "wall_s": result["wall_s"],
        "generation_scale": result["generation_scale"], "compiled_identity": result["compiled_identity"],
        "source_identity": result["source_identity"], "controls_sha256": result["controls_sha256"],
        "request_sha256": result["request_sha256"],
        "physical_certificate_status": result["physical_certificate_status"],
        "voltage_grid_qualification": "not_evaluated", "mesh_qualification": "not_evaluated"}


def read_three_wave_pv(
    coarse_path: Path, coarse_sha256: str, fine_path: Path, fine_sha256: str,
    tail_path: Path, tail_sha256: str,
) -> dict[str, Any]:
    """Read the two explicit parent links; preserve all three wave outcomes."""
    coarse_fine = read_joined_pv(coarse_path, coarse_sha256, fine_path, fine_sha256, 5)
    fine_tail = read_joined_pv(fine_path, fine_sha256, tail_path, tail_sha256, 39)
    tail = read_ionic_steady_jv(tail_path, tail_sha256)["result"]
    voltage = np.r_[coarse_fine["accepted_voltage_V"], tail["voltage_V"]]
    current = np.r_[coarse_fine["output_current_A_m2"], tail["output_current_A_m2"]]
    metrics = asdict(steady_jv_metrics(voltage, current, incident_power_W_m2=tail["incident_power_W_m2"]))
    crossings = np.flatnonzero((voltage[:-1] >= 0.0) & (current[:-1] > 0.0) & (current[1:] <= 0.0))
    bracket = None
    if metrics["open_circuit_voltage_V"] is not None and crossings.size:
        index = int(crossings[0])
        bracket = {"voltage_V": voltage[index:index + 2].tolist(),
                   "output_current_A_m2": current[index:index + 2].tolist()}
    return {"kind": "three_wave_accepted_steady_samples", "coarse_curve_sha256": coarse_sha256,
        "fine_curve_sha256": fine_sha256, "tail_curve_sha256": tail_sha256,
        "fine_parent_attempt_index": 5, "tail_parent_attempt_index": 39,
        "fine_initial_sha256": coarse_fine["initial_export_sha256"],
        "tail_initial_sha256": fine_tail["initial_export_sha256"],
        "coarse_completed": coarse_fine["parent_completed"],
        "fine_completed": coarse_fine["continuation_completed"],
        "tail_completed": fine_tail["continuation_completed"],
        "coarse_first_failure": coarse_fine["parent_first_failure"],
        "fine_first_failure": fine_tail["parent_first_failure"],
        "tail_first_failure": fine_tail["continuation_first_failure"],
        "accepted_voltage_V": voltage.tolist(), "output_current_A_m2": current.tolist(),
        "open_circuit_bracket": bracket, "open_circuit_bracket_established": bracket is not None,
        "metrics": metrics, "metric_basis": "piecewise-linear interpolation of linked accepted samples only",
        "incident_power_W_m2": tail["incident_power_W_m2"],
        "efficiency_status": "unknown_incident_power" if tail["incident_power_W_m2"] is None else "sampled_grid",
        "solver_calls_including_preserved_failures":
            coarse_fine["solver_calls_including_preserved_failures"] + tail["solver_calls"],
        "physical_certificate_status": "not_evaluated", "voltage_grid_qualification": "not_evaluated",
        "mesh_qualification": "not_evaluated"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (("solve", "one 51-point curve from an accepted illuminated initial"),
                            ("coarse-probe", "historical 17-point coarse probe")):
        solve = commands.add_parser(name, help=help_text)
        solve.add_argument("--initial-result", type=Path, required=True)
        solve.add_argument("--initial-sha256", required=True)
        solve.add_argument("--output", type=Path, required=True)
    for name, help_text in (("continue", "forty 25 mV steps from an accepted 0.2 V point"),
                            ("tail", "two 25 mV steps from an accepted 1.2 V point")):
        continuation = commands.add_parser(name, help=help_text)
        continuation.add_argument("--parent-result", type=Path, required=True)
        continuation.add_argument("--parent-sha256", required=True)
        continuation.add_argument("--parent-index", type=int, required=True)
        continuation.add_argument("--initial-output", type=Path, required=True)
        continuation.add_argument("--output", type=Path, required=True)
    read = commands.add_parser("read")
    read.add_argument("--result", type=Path, required=True)
    read.add_argument("--sha256", required=True)
    read.add_argument("--parent-result", type=Path)
    read.add_argument("--parent-sha256")
    read.add_argument("--parent-index", type=int)
    read.add_argument("--coarse-result", type=Path)
    read.add_argument("--coarse-sha256")
    args = parser.parse_args()
    if args.command == "read":
        value = pv_summary(read_ionic_steady_jv(args.result, args.sha256))
        if args.parent_result or args.parent_sha256 or args.parent_index is not None:
            if not (args.parent_result and args.parent_sha256 and args.parent_index is not None):
                parser.error("joined read requires parent-result, parent-sha256 and parent-index")
            value = read_joined_pv(args.parent_result, args.parent_sha256, args.result, args.sha256, args.parent_index)
        if args.coarse_result or args.coarse_sha256:
            if not (args.coarse_result and args.coarse_sha256 and args.parent_result
                    and args.parent_sha256 and args.parent_index == 39):
                parser.error("three-wave read requires coarse result/hash and fine parent at index39")
            value = read_three_wave_pv(args.coarse_result, args.coarse_sha256,
                args.parent_result, args.parent_sha256, args.result, args.sha256)
        print(json.dumps(value, indent=2, allow_nan=False))
        return 0
    if args.output.exists():
        raise FileExistsError("preserve the previous wave; choose an unused output path")
    device = compile_case("shared")
    voltages: tuple[float, ...]
    if args.command in ("continue", "tail"):
        initial, _, _ = export_parent_initial(device, args.parent_result, args.parent_sha256,
            args.parent_index, args.initial_output, expected_voltage_V=0.2 if args.command == "continue" else 1.2)
        declared = initial.steady_controls
        voltages = CONTINUATION_VOLTAGES_V if args.command == "continue" else TAIL_VOLTAGES_V
    else:
        declared = controls(device.ion_node_count)
        initial = load_ionic_steady_initial(device, args.initial_result, args.initial_sha256, controls=declared)
        voltages = VOLTAGES_V if args.command == "solve" else COARSE_PROBE_VOLTAGES_V
    if initial.generation_scale != GENERATION_SCALE:
        raise ValueError("the PV curve requires an already accepted G1 initial state")
    result = run_ionic_steady_jv(device, voltages, generation_scale=GENERATION_SCALE,
        options=IonicSteadyJVOptions(declared, max_points=len(voltages), wall_s=WAVE_WALL_S),
        initial_policy="saved_accepted", initial=initial, incident_power_W_m2=None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    digest = write_ionic_steady_jv(result, args.output)
    value = {"result_path": str(args.output.resolve()), "result_sha256": digest,
        **pv_summary(read_ionic_steady_jv(args.output, digest))}
    if args.command in ("continue", "tail"):
        value["joined"] = read_joined_pv(args.parent_result, args.parent_sha256, args.output, digest, args.parent_index)
    print(json.dumps(value, indent=2, allow_nan=False), flush=True)
    return 0 if result.completed else 2


if __name__ == "__main__":
    raise SystemExit(main())
