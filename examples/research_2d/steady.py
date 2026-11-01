"""One bounded genuine2D dark/illuminated path and solver-free saved reading.

    .venv/bin/python -m examples.research_2d.steady --output Steady2D.json
    .venv/bin/python -m examples.research_2d.steady --read Steady2D.json --sha256 SHA

Each explicitly counted newcore1D solve supplies only a numerical seed. The
seed is perturbed laterally before solving all195 connected2D equations.
Scientific qualification remains false; every first failure is retained.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from examples.research_1d.steady import load_device
from examples.research_2d.dc_assembly import spatial_input
from solarlab.device.compile_2d import compile_device_2d
from solarlab.equations.dc_2d import CompiledDevice2D, State2D
from solarlab.experiments.transient_results import _decode, _encode
from solarlab.solvers.nonlinear.steady_1d import SteadyOptions, SteadyStateError, solve_steady_state
from solarlab.solvers.nonlinear.steady_2d import Steady2DOptions, checks as steady_checks, diagnostics as steady_diagnostics, solve_steady_state_2d
from solarlab.equations.dc_2d import DC2DProblem

SCHEMA = "solarlab.classical_2d_steady.research.v1"


def lateral_initial(device: CompiledDevice2D, column_u: np.ndarray, *, voltage_V: float) -> State2D:
    g, N = device.geometry, device.node_count
    y = (device.column.nodes_m - device.column.nodes_m[0]) / np.ptp(device.column.nodes_m)
    x = (g.x_m - g.x_m[0]) / g.lateral_length_m
    wave = np.sin(np.pi * y)
    wave[[0, -1]] = 0.0
    u = device.extrude_coordinates(column_u)
    u[:N] += (.0007 * wave[:, None] * np.cos(np.pi * x)[None, :]).ravel()
    u[N:2 * N] += (.013 * wave[:, None] * np.sin(np.pi * x)[None, :]).ravel()
    u[2 * N:] += (-.011 * wave[:, None] * np.cos(2 * np.pi * x)[None, :]).ravel()
    return device.state(u, voltage_V=voltage_V)


def read_result(path: Path, sha256: str) -> dict[str, Any]:
    """Check raw identity/prefix/fields without a compiler, model or solver call."""
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha256:
        raise ValueError("2D steady artifact SHA-256 mismatch")
    value = _decode(json.loads(raw))
    if value.get("schema") != SCHEMA or value.get("qualification") is not False:
        raise ValueError("unsupported2D steady saved result")
    targets, attempts = value["requested_targets"], value["attempts"]
    shape = tuple(value["geometry"]["shape_Ny_Nx"])
    count = int(np.prod(shape))
    if len(attempts) > len(targets):
        raise ValueError("attempts exceed the explicit requested2D path")
    for index, attempt in enumerate(attempts):
        if attempt["target"] != targets[index] or (not attempt["accepted"] and index != len(attempts) - 1):
            raise ValueError("2D results must retain the complete prefix and first failure")
        if attempt["accepted"]:
            result = attempt["result_2d"]
            state, newton, data = result["state"], result["newton"], result["diagnostics"]
            if (result["accepted"] is not True or newton["accepted"] is not True
                    or result["termination"] != "converged" or not result["checks"]
                    or any(item["passed"] is not True or item["value"] > item["limit"] for item in result["checks"])
                    or result["compiled_identity"] != value["compiled_identity"]
                    or state["device_sha256"] != value["compiled_identity"]
                    or not np.array_equal(state["u"], newton["state"])
                    or not np.array_equal(state["u"], result["candidate_coordinates"])
                    or tuple(data["original_rows"]) != tuple(range(3 * count))):
                raise ValueError("accepted2D result lost original Newton/field/source evidence")
            for name in ("potential_V", "electron_density_m3", "hole_density_m3"):
                fields = np.asarray(state[name])
                if fields.shape != shape or not np.all(np.isfinite(fields)):
                    raise ValueError("accepted2D fields have invalid shape or values")
            if result["physical_certificate_status"] != "not_evaluated":
                raise ValueError("saved2D data cannot infer scientific qualification")
    completed = len(attempts) == len(targets) and all(item["accepted"] for item in attempts)
    if value["completed"] is not completed:
        raise ValueError("2D completion contradicts its accepted/failed prefix")
    return value


def run_case(output: Path, *, illumination_only: bool = False,
             prior_result: Path | None = None, prior_sha256: str | None = None) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError("preserve the first2D attempt; output already exists")
    started = time.monotonic()
    controls = Steady2DOptions()
    deadline = started + controls.wall_s
    prepared, spatial = load_device(), spatial_input()
    device = compile_device_2d(prepared, spatial)
    g = device.geometry
    prior = None
    if illumination_only:
        if prior_result is None or prior_sha256 is None:
            raise ValueError("illumination-only completion requires the exact preserved dark/failure artifact")
        prior = read_result(prior_result, prior_sha256)
        if (prior["compiled_identity"] != device.identity or not prior["attempts"]
                or prior["attempts"][0]["target"] != {"voltage_V": .002, "generation_scale": 0.0}
                or prior["attempts"][0]["accepted"] is not True):
            raise ValueError("prior artifact must preserve the accepted same-device dark2D point")
        saved = prior["attempts"][0]["result_2d"]["state"]
        dark_state = device.state(saved["u"], voltage_V=.002)
        if any(not np.array_equal(saved[name], getattr(dark_state, name))
               for name in ("potential_V", "electron_density_m3", "hole_density_m3")):
            raise ValueError("prior dark fields disagree with their exact coordinates")
        checks = steady_checks(steady_diagnostics(DC2DProblem(device, .002, 0.0), dark_state.u, controls), controls)
        if not all(check.passed for check in checks):
            raise ValueError("preserved dark2D point fails current original physical revalidation")
    targets = [{"voltage_V": .002, "generation_scale": scale}
               for scale in ((1.0,) if illumination_only else (0.0, 1.0))]
    attempts = []
    seed_calls, coupled_calls = 0, 0
    for target in targets:
        seed = None
        result = None
        try:
            if time.monotonic() >= deadline:
                raise TimeoutError("whole2D path budget exhausted before preparation")
            seed_calls += 1
            seed = solve_steady_state(device.column, voltage_V=target["voltage_V"],
                generation_scale=target["generation_scale"], initial_state=None,
                options=SteadyOptions())
            initial = lateral_initial(device, seed.state.u, voltage_V=target["voltage_V"])
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise TimeoutError("whole2D path budget exhausted during numerical seed preparation")
            result = solve_steady_state_2d(device, **target, options=replace(controls, wall_s=remaining),
                initial_state=initial, initial_provenance={"kind": "lateral perturbation of auxiliary1D numerical seed",
                    "auxiliary_seed_is_2d_completion": False, "auxiliary_1d_compiled_identity": device.column.identity,
                    "auxiliary_1d_state_sha256": hashlib.sha256(seed.state.u.tobytes()).hexdigest(),
                    "perturbation": {"phi_V": .0007, "z_n": .013, "z_p": .011,
                        "shape": "stack sine times lateral sine/cosine; contact perturbations exactly zero"}})
            coupled_calls += result.solver_controls["full_2d_newton_calls"]
            accepted, error = result.accepted, result.error
            termination = result.termination
        except Exception as exc:
            accepted, termination = False, "preparation_or_evaluation_failed"
            error = f"{type(exc).__name__}: {str(exc)[:600]}"
            failed_seed = exc.result if isinstance(exc, SteadyStateError) else None
        else:
            failed_seed = None
        attempts.append({"target": target, "accepted": accepted, "termination": termination, "error": error,
            "auxiliary_1d_seed": seed, "failed_auxiliary_seed_newton": failed_seed,
            "result_2d": result, "auxiliary_seed_is_2d_completion": False})
        if not accepted:
            break
    record = {"schema": SCHEMA, "completed": len(attempts) == len(targets) and all(item["accepted"] for item in attempts),
        "requested_targets": targets, "attempts": attempts, "compiled_identity": device.identity,
        "prepared_identity": prepared.content_sha256, "prepared_declaration": prepared.to_mapping(),
        "spatial_declaration": spatial.editing_data(), "controls": asdict(controls),
        "geometry": {"identity": g.sha256, "shape_Ny_Nx": g.shape, "x_m": g.x_m,
            "y_m": device.column.nodes_m, "cell_areas_m2": g.contract_geometry.volumes,
            "face_lengths_m": g.contract_geometry.face_measures, "face_pairs": g.contract_geometry.pairs,
            "face_spacing_m": g.face_spacing_m, "lateral_width_m": g.lateral_length_m},
        "auxiliary_1d_seed_calls": seed_calls, "full_2d_newton_calls": coupled_calls,
        "wall_s": time.monotonic() - started, "qualification": False}
    if prior is not None:
        record["preserved_prior_path"] = {"source_path": str(prior_result), "source_sha256": prior_sha256,
            "accepted_dark_2d_attempt": prior["attempts"][0],
            "first_failed_illumination_attempt": prior["attempts"][1] if len(prior["attempts"]) > 1 else None,
            "dark_revalidated_without_newton": True}
    output.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(_encode(record), indent=2, allow_nan=False) + "\n").encode()
    with output.open("xb") as stream:
        stream.write(raw)
    print(json.dumps({"result_path": str(output.resolve()), "result_sha256": hashlib.sha256(raw).hexdigest(),
        "completed": record["completed"], "shape": g.shape, "unknowns": 3 * device.node_count,
        "auxiliary_1d_seed_calls": seed_calls, "full_2d_newton_calls": coupled_calls,
        "points": [{"target": item["target"], "accepted": item["accepted"], "termination": item["termination"],
            "error": item["error"]} for item in attempts], "wall_s": record["wall_s"], "qualification": False}, indent=2), flush=True)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path)
    mode.add_argument("--read", type=Path)
    parser.add_argument("--sha256")
    parser.add_argument("--illumination-only", action="store_true")
    parser.add_argument("--prior-result", type=Path)
    parser.add_argument("--prior-sha256")
    args = parser.parse_args()
    if args.read is not None:
        if args.sha256 is None:
            parser.error("--read requires --sha256")
        record = read_result(args.read, args.sha256)
        print(json.dumps({"completed": record["completed"], "targets": record["requested_targets"],
            "accepted": [item["accepted"] for item in record["attempts"]], "qualification": False}, indent=2))
    else:
        record = run_case(args.output, illumination_only=args.illumination_only,
                          prior_result=args.prior_result, prior_sha256=args.prior_sha256)
    return 0 if record["completed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
