"""Resume the pinned 1153-node illuminated tail without replaying its prefix.

    python -m examples.research_1d.voc_resume --input-record Refinement384.json \
        --input-states Refinement384States.npz --output Resume.json
    python -m examples.research_1d.voc_resume --read Resume.json

The saved state is a numerical seed until the current coupled solver accepts
it. Every subsequent seed is an accepted current-source result. After the
first physical sign bracket, actual bisection solves use the existing direct
Voc voltage resolution. Neither result establishes mesh/grid qualification.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shlex
import signal
import subprocess
import time
from types import FrameType
from typing import Any

import numpy as np

from examples.research_1d.steady import load_device
from examples.research_1d.transient import json_value
from solarlab.device.compile_1d import compile_device_1d
from solarlab.equations.dc_1d import CompiledDevice1D, DC1DProblem, State1D
from solarlab.solvers.nonlinear.problem import state_digest
from solarlab.solvers.nonlinear.production import production_convergence_norm, production_residual_scaling
from solarlab.solvers.nonlinear.steady_1d import SteadyOptions, SteadyState1D, SteadyStateError, physical_gates, solve_steady_state

SCHEMA = "solarlab.fine_mesh_voc_resume.v1"
RECORD_SHA256 = "9823ea99e7178039210f40be5314a733710431ad0d50e8ba7631dee39e0a2ee8"
STATES_SHA256 = "10dbffdc58d2df20cd5802de5df179950830eacfdb83057796746b3c672ca8c9"
TAIL_VOLTAGES_V = tuple(index / 80 for index in range(91, 101))
SOFT_WALL_SECONDS = 84.0
VOLTAGE_TOLERANCE_V = 1e-6  # Existing solve_direct_voc default; Root confirmed.
MAX_BISECTION_CALLS = 64  # Existing direct-Voc call cap; first failure still stops.


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def original_processes(record_path: Path) -> list[dict[str, Any]]:
    """Inspect only matching original scripts; expose no unrelated arguments."""
    balance_root = record_path.parent.parent
    targets = (str(balance_root / "Review/Refine.py"), str(balance_root.parent / "RepairedV1/Measure.py"))
    matches = []
    for line in subprocess.check_output(["/bin/ps", "-axww", "-o", "pid=,etime=,command="], text=True).splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) != 3:
            continue
        try:
            args = shlex.split(parts[2])
        except ValueError:
            continue
        if args and Path(args[0]).name.startswith("python"):
            for target in targets:
                if any(target in arg for arg in args[1:]):
                    matches.append({"pid": int(parts[0]), "elapsed": parts[1], "target": target})
    return matches


def load_seed(record_path: Path, states_path: Path) -> tuple[dict[str, Any], CompiledDevice1D, State1D, SteadyOptions]:
    """Validate exact supplied evidence before constructing a numerical seed."""
    if file_sha256(record_path) != RECORD_SHA256 or file_sha256(states_path) != STATES_SHA256:
        raise ValueError("resume input hash mismatch")
    original = json.loads(record_path.read_text())
    controls = SteadyOptions(**original["production_controls"])
    if controls.convergence_policy != "poisson_contact_balance_v1":
        raise ValueError("the pinned balance-norm policy must be retained")
    device = compile_device_1d(load_device(), intervals_per_layer=original["intervals_per_layer"])
    with np.load(states_path, allow_pickle=False) as archive:
        coordinates = archive["last_accepted_u"].copy()
        nodes = archive["nodes_m"].copy()
    last = original["points"][-1]
    if device.node_count != 1153 or not np.array_equal(device.nodes_m, nodes):
        raise ValueError("saved nodes do not match the current 1153-node compilation")
    if device.identity != original["compiled_identity"]:
        raise ValueError("current compiled physical identity differs from the saved device")
    if state_digest(coordinates) != last["state_u_sha256"]:
        raise ValueError("last accepted coordinate digest mismatch")
    return original, device, device.state(coordinates, voltage_V=last["voltage_V"]), controls


def physical_evidence(device: CompiledDevice1D, state: State1D, controls: SteadyOptions) -> dict[str, Any]:
    problem = DC1DProblem(device, state.voltage_V, 1.0)
    residual = problem.residual(state.u)
    scaling = production_residual_scaling(device, poisson_tolerance=controls.poisson_tolerance,
                                         continuity_tolerance_A_m2=controls.continuity_tolerance_A_m2,
                                         contact_log_tolerance=controls.contact_log_tolerance)
    norm = production_convergence_norm(device, convergence_policy=controls.convergence_policy)
    assert norm is not None
    value = norm(scaling.scaled(residual))
    gates = [asdict(gate(state.u.copy())) for gate in physical_gates(problem, controls, equilibrium=False)]
    contacts = problem.contact_current_density(state.u)
    current = np.r_[contacts[:1], problem.current_density(state.u), contacts[1:]]
    output = -device.voltage_polarity * current
    return {"convergence_norm": value, "physical_gates": gates,
            "checks_pass": value <= 1.0 and all(gate["passed"] for gate in gates),
            "Jout_A_m2": float(output[0]), "Jout_interval_A_m2": [float(np.min(output)), float(np.max(output))],
            "original_residual": residual, "all_face_contact_current_A_m2": current}


def read_result(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text())
    if not isinstance(result, dict) or result.get("schema") != SCHEMA:
        raise ValueError("not a fine-mesh Voc resume result")
    return result


def run(record_path: Path, states_path: Path, output_path: Path) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError("preserve the original resume attempt")
    processes = original_processes(record_path)
    if processes:
        raise RuntimeError("the original refinement process is still present")
    started = time.monotonic()
    summary: dict[str, Any] = {"status": "starting", "accepted_tail_points": 0, "solve_calls": 0,
                               "current_source_readmitted": False, "qualification": False,
                               "original_processes": processes, "root_refinement_performed": False,
                               "accepted_bisection_points": 0, "voltage_tolerance_V": VOLTAGE_TOLERANCE_V}
    payload: dict[str, Any] = {
        "schema": SCHEMA, "summary": summary,
        "input_record": {"path": str(record_path.resolve()), "sha256": file_sha256(record_path)},
        "input_states": {"path": str(states_path.resolve()), "sha256": file_sha256(states_path)},
        "protocol": {"name": "saved_state_readmission_12p5mV_tail_then_bisection_v1",
                     "requested_tail_voltages_V": TAIL_VOLTAGES_V, "generation_scale": 1.0,
                     "stop": "direct_voc_width_target_or_first_failed_point",
                     "voltage_tolerance_V": VOLTAGE_TOLERANCE_V,
                     "max_bisection_calls": MAX_BISECTION_CALLS,
                     "soft_wall_seconds": SOFT_WALL_SECONDS, "prefix_replayed": False,
                     "failed_candidates_used_as_seeds": False},
        "attempts": [], "accepted_prefix": [], "first_failure": None,
        "coarse_voc_bracket": None, "voc_bracket": None,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.touch(exist_ok=False)

    def checkpoint() -> None:
        summary["elapsed_seconds"] = time.monotonic() - started
        raw = json.dumps(json_value(payload), indent=2, allow_nan=False) + "\n"
        temporary = output_path.with_suffix(output_path.suffix + ".pending")
        temporary.write_text(raw)
        temporary.replace(output_path)

    interrupted: dict[str, Any] = {}

    def timeout(signum: int, frame: FrameType | None) -> None:
        while frame is not None:
            if frame.f_globals.get("__name__") == "solarlab.solvers.nonlinear.newton":
                local = frame.f_locals
                if isinstance(local.get("u"), np.ndarray):
                    interrupted.update(state=local["u"].copy(), iteration=local.get("iteration"),
                                       counts=dict(local.get("counts", {})), convergence_norm=local.get("norm"))
                    break
            frame = frame.f_back
        raise TimeoutError("the single resume wave reached its 84-second soft wall")

    active: dict[str, Any] | None = None
    previous_handler = signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, SOFT_WALL_SECONDS)
    try:
        original, device, seed, controls = load_seed(record_path, states_path)
        summary.update(node_count=device.node_count, compiled_identity=device.identity,
                       saved_voltage_V=seed.voltage_V, convergence_policy=controls.convergence_policy)
        payload.update(controls=asdict(controls), initial_seed=seed,
                       seed_current_checks=physical_evidence(device, seed, controls),
                       original_evidence={"status": original["status"], "accepted_points": original["accepted_points"],
                                          "solve_calls": original["solve_calls"], "failed_attempts": original["failed_attempts"],
                                          "source_pins": original["source_pins_before"]})
        checkpoint()
        accepted: dict[float, SteadyState1D] = {}

        def attempt(voltage: float, phase: str, initial: State1D) -> dict[str, Any] | None:
            nonlocal active
            call = summary["solve_calls"] + 1
            active = {"call": call, "phase": phase,
                      "voltage_V": voltage, "seed_voltage_V": initial.voltage_V,
                      "seed_u_sha256": state_digest(initial.u),
                      "seed_status": "historical_numerical_seed" if phase == "readmission" else "current_source_accepted",
                      "status": "started"}
            payload["attempts"].append(active)
            summary["solve_calls"] = call
            checkpoint()
            before = time.monotonic()
            try:
                result = solve_steady_state(device, initial_state=initial, voltage_V=voltage,
                                            generation_scale=1.0, options=controls)
            except SteadyStateError as error:
                active.update(status="failed", elapsed_seconds=time.monotonic() - before,
                              stage=error.stage, error=str(error), failed_newton=error.result)
                candidate = device.state(error.result.state, voltage_V=voltage)
                active["failed_candidate_current_checks"] = physical_evidence(device, candidate, controls)
                payload["first_failure"] = {"call": call, "voltage_V": voltage, "error": str(error)}
                summary["status"] = "failed_" + phase
                return None
            checks = physical_evidence(device, result.state, controls)
            active.update(result=result, current_checks=checks, elapsed_seconds=time.monotonic() - before)
            if not result.converged or not checks["checks_pass"]:
                raise RuntimeError("the current-source result failed its original physical recheck")
            active["status"] = "accepted"
            point = {"voltage_V": voltage, "Jout_A_m2": checks["Jout_A_m2"],
                     "Jout_interval_A_m2": checks["Jout_interval_A_m2"],
                     "state_u_sha256": state_digest(result.state.u), "call": call}
            payload["accepted_prefix"].append(point)
            accepted[voltage] = result
            summary["current_source_readmitted"] = True
            if phase in ("tail", "bisection"):
                summary["accepted_" + phase + "_points"] += 1
            summary["last_accepted_voltage_V"] = voltage
            return point

        last_positive: dict[str, Any] | None = None
        for index, voltage in enumerate((seed.voltage_V, *TAIL_VOLTAGES_V)):
            point = attempt(voltage, "readmission" if index == 0 else "tail", seed)
            if point is None:
                break
            seed = accepted[voltage].state
            if last_positive is not None and point["Jout_interval_A_m2"][1] < 0.0:
                left_v, left_j = last_positive["voltage_V"], last_positive["Jout_A_m2"]
                interpolation = left_v + left_j * (voltage - left_v) / (left_j - point["Jout_A_m2"])
                payload["coarse_voc_bracket"] = {"left": last_positive, "right": point,
                                                  "width_V": voltage - left_v, "linear_interpolated_V": interpolation}
                payload["voc_bracket"] = {"left": last_positive, "right": point, "width_V": voltage - left_v}
                summary.update(status="voc_bracketed", coarse_voc_linear_interpolated_V=interpolation,
                               voc_bracket_width_V=voltage - left_v)
                break
            if point["Jout_interval_A_m2"][0] > 0.0:
                last_positive = point
            checkpoint()
        else:
            summary["status"] = "tail_complete_without_bracket"
        if payload["coarse_voc_bracket"] is not None and payload["first_failure"] is None:
            summary["root_refinement_performed"] = True
            for _ in range(MAX_BISECTION_CALLS):
                bracket = payload["voc_bracket"]
                lower, upper = bracket["left"], bracket["right"]
                if bracket["width_V"] <= VOLTAGE_TOLERANCE_V:
                    estimate = min((lower, upper), key=lambda point: abs(point["Jout_A_m2"]))
                    payload["direct_voc_solution"] = estimate
                    summary.update(status="voc_resolved", voc_solution_V=estimate["voltage_V"],
                                   voc_solution_Jout_A_m2=estimate["Jout_A_m2"],
                                   voc_bracket_width_V=bracket["width_V"])
                    break
                target = lower["voltage_V"] + .5 * bracket["width_V"]
                if target in (lower["voltage_V"], upper["voltage_V"]):
                    raise RuntimeError("binary64 cannot narrow the solved voltage bracket")
                nearest = min(accepted.values(), key=lambda point: (abs(point.voltage_V - target), point.voltage_V))
                point = attempt(target, "bisection", nearest.state)
                if point is None:
                    break
                if point["Jout_interval_A_m2"][0] > 0.0:
                    lower = point
                elif point["Jout_interval_A_m2"][1] < 0.0:
                    upper = point
                else:
                    summary["status"] = "unresolved_midpoint_sign"
                    payload["first_failure"] = {"call": point["call"], "voltage_V": target,
                                                "type": "UnresolvedCurrentSign",
                                                "error": "accepted midpoint has an all-face interval containing zero",
                                                "Jout_interval_A_m2": point["Jout_interval_A_m2"]}
                    break  # Keep the last strict bracket and the accepted midpoint.
                payload["voc_bracket"] = {"left": lower, "right": upper,
                                           "width_V": upper["voltage_V"] - lower["voltage_V"]}
                summary["voc_bracket_width_V"] = payload["voc_bracket"]["width_V"]
                checkpoint()
            else:
                summary["status"] = "incomplete_bisection_call_cap"
    except Exception as error:
        summary["status"] = "incomplete_timeout" if isinstance(error, TimeoutError) else "failed"
        failure = {"type": type(error).__name__, "error": str(error), "interrupted_newton": interrupted}
        if active is not None:
            active.update(status="failed", failure=failure)
            failure.update(call=active["call"], voltage_V=active["voltage_V"])
        if payload["first_failure"] is None:
            payload["first_failure"] = failure
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)
        checkpoint()
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path)
    mode.add_argument("--read", type=Path)
    parser.add_argument("--input-record", type=Path)
    parser.add_argument("--input-states", type=Path)
    args = parser.parse_args()
    if args.read is not None:
        print(json.dumps(read_result(args.read)["summary"], indent=2))
        return 0
    if args.input_record is None or args.input_states is None:
        parser.error("--input-record and --input-states are required for the one resume attempt")
    value = run(args.input_record, args.input_states, args.output)
    print(json.dumps({"result_sha256": file_sha256(args.output), **value["summary"]}, indent=2), flush=True)
    return 0 if value["summary"]["status"] == "voc_resolved" else 2


if __name__ == "__main__":
    raise SystemExit(main())
