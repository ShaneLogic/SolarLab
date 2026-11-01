"""Normal-face CT2D arithmetic and one bounded13x5 operating point.

    .venv/bin/python -m examples.research_2d.field_transport --output Algebra.json
    .venv/bin/python -m examples.research_2d.field_transport --steady-output Steady.json

The scalar law is reused with x/y normal components and legacy face averages.
The steady mode requires the explicitly composed compiler/DC binding; the
arithmetic mode and saved reader perform no operating-point solve.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

from examples.research_1d.steady import load_device
from examples.research_2d.dc_assembly import coordinates, spatial_input
from solarlab.device.compile_2d import compile_device_2d
from solarlab.config.resolve_device import resolve_device
from solarlab.device.inputs import DeviceInput
from solarlab.device.resolved import PreparedDevice
from solarlab.equations.dc_2d import DC2DProblem
from solarlab.equations.field_transport_2d import FieldTransport2D, bind_field_transport_2d
from solarlab.experiments.transient_results import _decode, _encode
from solarlab.solvers.nonlinear.steady_2d import Steady2DOptions, solve_steady_state_2d
from solarlab.solvers.nonlinear.problem import state_digest


def prepare_case(config: Path | None = None) -> PreparedDevice:
    """Declare only the accepted smoothCT parameters; retain physical defaults."""
    prepared = load_device(config)
    data = prepared.to_input().editing_data()
    data["layers"][1]["parameters"].update(v_sat_n=1e4, v_sat_p=1.5e4, ct_beta_n=2.0, ct_beta_p=2.0)
    return resolve_device(DeviceInput.model_validate(data), prepared.defaults, prepared.resources, sources=prepared.sources)


def native_case(output: Path, *, config: Path | None = None) -> dict[str, Any]:
    """One25s dark2mV connected2D solve; no numerical preparation/fallback."""
    if output.exists():
        raise FileExistsError("preserve the firstCT2D operating-point attempt")
    prepared = prepare_case(config)
    d = compile_device_2d(prepared, spatial_input())
    model: FieldTransport2D | None = getattr(d, "field_transport", None)
    if model is None:
        raise ValueError("native CT2D requires the explicit composed compiler/transport binding")
    phi = d.extrude(d.column.reference_potential_V)
    lower, upper = d.column.contact_potentials(.002)
    phi[:d.geometry.nx], phi[-d.geometry.nx:] = lower, upper
    z = []
    for reference in d.reference_potentials(.002):
        value = np.zeros(d.node_count)
        value[d.geometry.contact_nodes] = (reference[d.geometry.contact_nodes] - phi[d.geometry.contact_nodes]) / d.thermal_voltage_V
        z.append(value)
    stack = np.sin(np.linspace(0.0, np.pi, d.geometry.ny))
    stack[[0, -1]] = 0.0
    lateral = np.sin(np.linspace(0.0, np.pi, d.geometry.nx))
    pattern = (stack[:, None] * lateral[None, :]).ravel()
    u = np.r_[phi + .0007 * pattern, z[0] + .013 * pattern, z[1] - .011 * pattern]
    initial = d.state(u, voltage_V=.002)
    options = Steady2DOptions(wall_s=25.0)
    result = solve_steady_state_2d(d, voltage_V=.002, generation_scale=0.0, options=options,
        initial_state=initial, initial_provenance={"kind": "compiler physical reference plus explicit lateral perturbation",
            "new_auxiliary_or_prior_solve_calls": 0, "phi_perturbation_V": .0007,
            "z_n_perturbation": .013, "z_p_perturbation": -.011, "initial_digest": state_digest(u)})
    field = model.evaluate(result.candidate_coordinates[:d.node_count])
    problem = DC2DProblem(d, .002, 0.0)
    record = {"schema": "solarlab.normal_face_ct_2d.steady.v1", "completed": result.accepted,
        "result_2d": result, "initial_state": initial, "initial_coordinates": u,
        "compiled_identity": d.identity, "geometry_identity": d.geometry.sha256, "geometry_shape": d.geometry.shape,
        "field_transport_identity": model.identity_sha256, "field_measure": model.field_measure,
        "configuration_sources": [{"id": source.id, "sha256": source.sha256} for source in prepared.sources],
        "base_electron_D_m2_s": d.electron_diffusivity_faces_m2_s, "base_hole_D_m2_s": d.hole_diffusivity_faces_m2_s,
        "electron_parameters": asdict(model.electron), "hole_parameters": asdict(model.hole),
        "field_evaluation": asdict(field), "candidate_original_residual": problem.residual(result.candidate_coordinates),
        "face_pairs": d.geometry.contract_geometry.pairs, "face_spacing_m": d.geometry.face_spacing_m,
        "face_lengths_m": d.geometry.contract_geometry.face_measures, "cell_areas_m2": d.geometry.contract_geometry.volumes,
        "lateral_width_m": d.geometry.lateral_length_m, "options": asdict(options),
        "new_auxiliary_or_prior_solve_calls": 0, "qualification": False}
    output.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(_encode(record), indent=2, allow_nan=False) + "\n").encode()
    with output.open("xb") as stream:
        stream.write(raw)
    print(json.dumps({"path": str(output.resolve()), "sha256": hashlib.sha256(raw).hexdigest(),
        "completed": result.accepted, "newton_calls": result.solver_controls["full_2d_newton_calls"],
        "qualification": False}, indent=2))
    return record


def algebra(output: Path, *, config: Path | None = None) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError("preserve the original field2D arithmetic receipt")
    d = compile_device_2d(load_device(config), spatial_input())
    g, count = d.geometry, d.node_count
    profile = np.zeros(g.ny)
    profile[4:9] = 1.0
    v_n, v_p = d.extrude(1e4 * profile), d.extrude(1.5e4 * profile)
    beta = np.full(count, 2.0)
    model = bind_field_transport_2d(g, d.electron_diffusivity_faces_m2_s,
        d.hole_diffusivity_faces_m2_s, d.thermal_voltage_V, v_sat_n=v_n, v_sat_p=v_p,
        ct_beta_n=beta, ct_beta_p=beta, pf_gamma_n=np.zeros(count), pf_gamma_p=np.zeros(count))
    assert model is not None
    _, u = coordinates(d, lateral=True)
    u[:count] += np.tile(np.linspace(0.0, .08, g.nx), g.ny)
    phi, n, p = d.fields(u, voltage_V=.002)
    psi_n, psi_p = d.band_structure.transport_potentials(phi)
    left, right = g.contract_geometry.pairs.T
    fluxes = model.fluxes(phi, n, p, psi_n, psi_p,
        electron_electrochemical_step=u[count + right] - u[count + left],
        hole_electrochemical_step=-(u[2 * count + right] - u[2 * count + left]))
    evaluation = model.evaluate(phi)
    source = {}
    for name in ("solarlab.equations.field_transport_2d", "solarlab.physics.transport.field_mobility",
                 "solarlab.physics.transport.electrochemical_flux", "solarlab.mesh.geometry_2d"):
        filename = sys.modules[name].__file__
        if filename is None:
            raise ValueError("actual arithmetic source binding required")
        source[name] = hashlib.sha256(Path(filename).read_bytes()).hexdigest()
    record = {"schema": "solarlab.normal_face_ct_2d.algebra.v1", "completed": False,
        "algebra_completed": True, "operating_point_accepted": False, "newton_calls": 0,
        "geometry_identity": g.sha256, "geometry_shape": g.shape, "coordinate_count": 3 * count,
        "constant_compiled_identity": d.identity, "field_transport_identity": model.identity_sha256,
        "field_measure": model.field_measure, "scope": "smoothCT beta>1; PFoff; nonperiodic normal components",
        "shared_compiler_dc_integration": "pending", "base_electron_D_m2_s": model.base_electron_diffusivity_m2_s,
        "base_hole_D_m2_s": model.base_hole_diffusivity_m2_s, "electron_parameters": asdict(model.electron),
        "hole_parameters": asdict(model.hole), "evaluation": asdict(evaluation), "fluxes": asdict(fluxes),
        "x_face_count": g.x_face_count, "face_pairs": g.contract_geometry.pairs,
        "face_spacing_m": g.face_spacing_m, "face_lengths_m": g.contract_geometry.face_measures,
        "cell_areas_m2": g.contract_geometry.volumes, "source_sha256": source, "qualification": False}
    output.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(_encode(record), indent=2, allow_nan=False) + "\n").encode()
    with output.open("xb") as stream:
        stream.write(raw)
    print(json.dumps({"path": str(output.resolve()), "sha256": hashlib.sha256(raw).hexdigest(),
        "algebra_completed": True, "newton_calls": 0, "qualification": False}, indent=2))
    return record


def read_result(path: Path, sha256: str) -> dict[str, Any]:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha256:
        raise ValueError("field2D arithmetic SHA mismatch")
    record = _decode(json.loads(raw))
    if record["qualification"]:
        raise ValueError("invalid solver-free field2D arithmetic receipt")
    if record["schema"] == "solarlab.normal_face_ct_2d.steady.v1":
        result = record["result_2d"]
        if record["completed"] is not result["accepted"] or result["compiled_identity"] != record["compiled_identity"]:
            raise ValueError("CT2D saved acceptance/model identity mismatch")
        if record["completed"] and (result["newton"]["accepted"] is not True or
                tuple(result["diagnostics"]["original_rows"]) != tuple(range(195)) or
                not all(value["passed"] for value in result["checks"])):
            raise ValueError("CT2D saved original rows/gates are incomplete")
    elif record["schema"] != "solarlab.normal_face_ct_2d.algebra.v1" or record["newton_calls"] != 0:
        raise ValueError("invalid solver-free field2D arithmetic receipt")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path)
    mode.add_argument("--steady-output", type=Path)
    mode.add_argument("--read", type=Path)
    parser.add_argument("--sha256")
    parser.add_argument("--config", type=Path, help="explicit content-bound device/catalog JSON")
    args = parser.parse_args()
    if args.read is not None:
        if args.sha256 is None:
            parser.error("--read requires --sha256")
        record = read_result(args.read, args.sha256)
        print(json.dumps({"schema": record["schema"], "completed": record.get("completed", False), "qualification": False}))
    elif args.steady_output is not None:
        return 0 if native_case(args.steady_output, config=args.config)["completed"] else 2
    else:
        algebra(args.output, config=args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
