"""One illustrative neutral grain-region steady point, with lossless saved read.

    .venv/bin/python -m examples.research_2d.grains --algebra Algebra.json
    .venv/bin/python -m examples.research_2d.grains --output GrainSteady.json
    .venv/bin/python -m examples.research_2d.grains --read GrainSteady.json --sha256 SHA

The declared lifetimes and geometry are illustrative research inputs. A
single cold new-core 1D calculation is numerical preparation only; only the
connected 2D Newton result can establish the requested steady point.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

from examples.research_1d.steady import load_device
from examples.research_2d.dc_assembly import coordinates, spatial_input as uniform_spatial_input
from examples.research_2d.steady import lateral_initial
from solarlab.device.compile_2d import compile_device_2d
from solarlab.device.grain_2d import GrainRecombination2D
from solarlab.device.resolved import PreparedDevice
from solarlab.equations.dc_2d import CompiledDevice2D, DC2DProblem
from solarlab.experiments.transient_results import _decode, _encode
from solarlab.experiments.two_dimensional.inputs import JV2DInput
from solarlab.solvers.nonlinear.steady_1d import SteadyOptions, SteadyStateError, solve_steady_state
from solarlab.solvers.nonlinear.steady_2d import Steady2DOptions, solve_steady_state_2d

SCHEMA = "solarlab.neutral_grain_2d.research.v1"
TARGET = {"voltage_V": .002, "generation_scale": 1.0}


def spatial_input() -> JV2DInput:
    return JV2DInput.model_validate({**uniform_spatial_input().editing_data(), "microstructure": {
        "grain_boundaries": [{"id": "absorber_grain", "layer_ids": ["absorber"],
            "x_position": "0.4 um", "width": "0.3 um", "tau_n": "0.5 us", "tau_p": "0.25 us"}]}})


def source_pins() -> dict[str, str]:
    """Record the actual imported core plus the three research consumers."""
    result = {}
    for name, module in tuple(sys.modules.items()):
        if not (name == "solarlab" or name.startswith("solarlab.") or name in {
                __name__, "examples.research_1d.steady", "examples.research_2d.steady",
                "examples.research_2d.dc_assembly"}):
            continue
        filename = getattr(module, "__file__", None)
        if filename is not None:
            result[name] = hashlib.sha256(Path(filename).read_bytes()).hexdigest()
    return result


def binding_record(d: CompiledDevice2D, prepared: PreparedDevice) -> dict[str, Any]:
    binding = d.grain_recombination
    if not isinstance(binding, GrainRecombination2D):
        raise ValueError("this example requires its explicit neutral grain inventory")
    g = d.geometry
    return {"model_id": binding.model_id, "geometry_sha256": binding.geometry_sha256,
        "effective_origin": binding.effective_origin, "grain_row_ownership": "left_layer_wins",
        "material_ownership": "right_layer_wins", "rate_unit": "m^-3 s^-1", "partial_unit": "s^-1",
        "rule": "R_total += f * (R_grain_SRH - R_bulk_SRH); identical correction to dR/dn and dR/dp",
        "no_grain_charge_or_extra_state": True, "effective_lifetime_painting": False,
        "regions": [{"declaration": r.declaration.normalized_data(), "x_overlap_fraction": r.x_overlap_fraction,
            "y_mask": r.y_mask, "fractions": r.fractions.reshape(g.shape),
            "represented_width_m": r.represented_width_m, "represented_height_m": r.represented_height_m,
            "represented_area_m2": r.represented_area_m2,
            "nominal_selected_layer_thickness_m": sum(layer.thickness for layer in prepared.layers
                if layer.id in r.declaration.layer_ids)} for r in binding.regions],
        "bulk_coefficients": asdict(d.recombination.coefficients),
        "column_projection": None if d.column_projection_json is None else json.loads(d.column_projection_json)}


def _base_record(d: CompiledDevice2D, prepared: PreparedDevice, spatial: JV2DInput) -> dict[str, Any]:
    g = d.geometry
    return {"schema": SCHEMA, "target": TARGET, "qualification": False,
        "input_interpretation": "illustrative declared neutral volumetric SRH bands; no mesh or scientific qualification",
        "compiled_identity": d.identity, "prepared_identity": prepared.content_sha256,
        "prepared_declaration": prepared.to_mapping(), "spatial_declaration": spatial.editing_data(),
        "auxiliary_column_identity": d.column.identity, "grain_binding": binding_record(d, prepared),
        "geometry": {"sha256": g.sha256, "shape_Ny_Nx": g.shape, "x_m": g.x_m, "y_m": d.column.nodes_m,
            "x_widths_m": g.x_widths_m, "y_widths_m": g.y_geometry.cell_measures_m,
            "cell_areas_m2": g.contract_geometry.volumes.reshape(g.shape),
            "lateral_width_m": g.lateral_length_m, "face_pairs": g.contract_geometry.pairs,
            "face_lengths_m": g.contract_geometry.face_measures, "face_spacing_m": g.face_spacing_m},
        "generation_source_m3_s": d.generation_source_m3_s.reshape(g.shape),
        "source_sha256": source_pins()}


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(_encode(value), indent=2, allow_nan=False) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(raw)
    print(json.dumps({"path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest(),
                      "completed": value.get("completed", False), "qualification": False}), flush=True)


def algebra(output: Path, *, config: Path | None = None) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError("preserve original evidence; output already exists")
    prepared, spatial = load_device(config), spatial_input()
    d = compile_device_2d(prepared, spatial)
    record = _base_record(d, prepared, spatial)
    _, u = coordinates(d, lateral=True)
    problem = DC2DProblem(d, .002, 0.0)
    matrix = problem.jacobian(u)
    rng = np.random.default_rng(772)
    direction = np.r_[.02*rng.normal(size=d.node_count), rng.normal(size=2*d.node_count)]
    h = 2e-4
    numeric = (problem.residual(u-2*h*direction)-8*problem.residual(u-h*direction)
               + 8*problem.residual(u+h*direction)-problem.residual(u+2*h*direction))/(12*h)
    scale = np.maximum(abs(matrix) @ abs(direction), 1e-20)
    _, n, p = d.fields(u, voltage_V=.002)
    record.update(mode="algebra", completed=False, accepted=False, newton_calls=0,
        state=d.state(u, voltage_V=.002), residual=problem.residual(u),
        jacobian_csc={"data": matrix.data, "indices": matrix.indices, "indptr": matrix.indptr, "shape": matrix.shape},
        recombination=asdict(problem.recombination_rate(n, p)), bulk_recombination=asdict(d.recombination.evaluate(n, p).total),
        direction=direction, numeric_direction=numeric, analytic_direction=matrix@direction,
        directional_scale=scale, directional_step=h, directional_criterion=2e-8,
        directional_relative_error=float(np.max(abs(numeric-matrix@direction)/scale)))
    _write(output, record)
    return record


def run_case(output: Path, *, config: Path | None = None) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError("preserve the first grain steady attempt; output already exists")
    started = time.monotonic()
    controls = Steady2DOptions()
    prepared, spatial = load_device(config), spatial_input()
    d = compile_device_2d(prepared, spatial)
    record = _base_record(d, prepared, spatial)
    seed, result, failed_seed, error = None, None, None, None
    seed_calls, coupled_calls = 0, 0
    try:
        seed_calls += 1
        seed = solve_steady_state(d.column, **TARGET, initial_state=None, options=SteadyOptions())
        initial = lateral_initial(d, seed.state.u, voltage_V=TARGET["voltage_V"])
        remaining = controls.wall_s - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError("whole grain witness budget exhausted in cold numerical preparation")
        result = solve_steady_state_2d(d, **TARGET, options=replace(controls, wall_s=remaining), initial_state=initial,
            initial_provenance={"kind": "existing cold auxiliary1D preparation plus explicit lateral perturbation",
                "auxiliary_seed_is_2d_completion": False, "auxiliary_1d_compiled_identity": d.column.identity,
                "auxiliary_1d_state_sha256": hashlib.sha256(seed.state.u.tobytes()).hexdigest(),
                "grain_model_id": record["grain_binding"]["model_id"],
                "perturbation": {"phi_V": .0007, "z_n": .013, "z_p": .011,
                    "shape": "stack sine times lateral sine/cosine; contact perturbations exactly zero"}})
        coupled_calls = result.solver_controls["full_2d_newton_calls"]
        after = source_pins()
        record["source_sha256_after"] = after
        if any(after.get(name) != pin for name, pin in record["source_sha256"].items()):
            raise RuntimeError("imported physical source changed during the grain witness")
        error = result.error
    except Exception as exc:
        error = f"{type(exc).__name__}: {str(exc)[:600]}"
        failed_seed = exc.result if isinstance(exc, SteadyStateError) else None
    accepted = result is not None and result.accepted and error is None
    record.update(mode="steady", completed=accepted, accepted=accepted,
        termination="preparation_or_source_failed" if result is None or error is not None else result.termination,
        error=error, result_2d=result, auxiliary_1d_seed=seed, failed_auxiliary_seed_newton=failed_seed,
        auxiliary_1d_seed_calls=seed_calls, full_2d_newton_calls=coupled_calls,
        auxiliary_seed_is_2d_completion=False, controls=asdict(controls), wall_s=time.monotonic()-started)
    if result is not None and result.state is not None:
        problem = DC2DProblem(d, **TARGET)
        _, n, p = d.fields(result.state.u, voltage_V=TARGET["voltage_V"])
        record["recombination"] = asdict(problem.recombination_rate(n, p))
        record["bulk_recombination"] = asdict(d.recombination.evaluate(n, p).total)
    _write(output, record)
    return record


def read_result(path: Path, sha256: str) -> dict[str, Any]:
    """Validate saved identities, representation and results without any solve."""
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha256:
        raise ValueError("grain2D saved SHA-256 mismatch")
    value = _decode(json.loads(raw))
    if value.get("schema") != SCHEMA or value.get("qualification") is not False:
        raise ValueError("unsupported grain2D saved result")
    g, binding = value["geometry"], value["grain_binding"]
    shape = tuple(g["shape_Ny_Nx"])
    if binding["geometry_sha256"] != g["sha256"] or not binding["regions"]:
        raise ValueError("missing exact grain geometry binding")
    for r in binding["regions"]:
        f = np.asarray(r["fractions"])
        expected = np.asarray(r["y_mask"])[:, None] * np.asarray(r["x_overlap_fraction"])[None, :]
        if (f.shape != shape or not np.array_equal(f, expected) or not np.all(np.isfinite(f))
                or np.any(f < 0) or np.any(f > 1)
                or float(f.ravel() @ np.asarray(g["cell_areas_m2"]).ravel()) != r["represented_area_m2"]):
            raise ValueError("saved grain mask/fractions/represented area disagree")
    if value["mode"] == "steady":
        result = value["result_2d"]
        completed = result is not None and result["accepted"] is True and value["error"] is None
        if value["completed"] is not completed or value["accepted"] is not completed:
            raise ValueError("grain completion disagrees with the retained attempt")
        if result is not None:
            if (result["compiled_identity"] != value["compiled_identity"]
                    or result["physical_certificate_status"] != "not_evaluated"
                    or result["operating_voltage_V"] != value["target"]["voltage_V"]
                    or result["generation_scale"] != value["target"]["generation_scale"]):
                raise ValueError("grain2D result lost original source or operating point")
        if completed:
            state, newton, data = result["state"], result["newton"], result["diagnostics"]
            if (state["device_sha256"] != value["compiled_identity"] or newton["accepted"] is not True
                    or not np.array_equal(state["u"], newton["state"])
                    or not np.array_equal(state["u"], result["candidate_coordinates"])
                    or tuple(data["original_rows"]) != tuple(range(3*int(np.prod(shape))))
                    or not result["checks"] or any(check["passed"] is not True or check["value"] > check["limit"] for check in result["checks"])):
                raise ValueError("accepted grain2D result lost original full-row Newton/gate evidence")
            for name in ("potential_V", "electron_density_m3", "hole_density_m3"):
                fields = np.asarray(state[name])
                if fields.shape != shape or not np.all(np.isfinite(fields)) or (name != "potential_V" and np.any(fields <= 0)):
                    raise ValueError("saved2D physical fields invalid")
    elif value["mode"] != "algebra" or value["completed"] or value["newton_calls"] != 0:
        raise ValueError("algebra evidence cannot claim a steady result")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--algebra", type=Path)
    mode.add_argument("--output", type=Path)
    mode.add_argument("--read", type=Path)
    parser.add_argument("--sha256")
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    if args.read is not None:
        if args.sha256 is None:
            parser.error("--read requires --sha256")
        record = read_result(args.read, args.sha256)
        print(json.dumps({"mode": record["mode"], "completed": record["completed"],
                          "compiled_identity": record["compiled_identity"], "qualification": False}))
    elif args.algebra is not None:
        record = algebra(args.algebra, config=args.config)
    else:
        record = run_case(args.output, config=args.config)
    return 0 if record["mode"] == "algebra" or record["completed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
