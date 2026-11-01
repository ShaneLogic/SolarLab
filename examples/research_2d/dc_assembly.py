"""Inspect genuine 2D DC equations and analytic tangents without a Newton run.

From the repository root:
    python -m examples.research_2d.dc_assembly --output assembly.json
    python -m examples.research_2d.dc_assembly --read assembly.json

The finite states below are algebra probes, not accepted steady states.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from examples.research_1d.steady import load_device
from examples.research_1d.transient import json_value
from solarlab.device.compile_2d import compile_device_2d
from solarlab.equations.dc_1d import DC1DProblem
from solarlab.equations.dc_2d import CompiledDevice2D, DC2DProblem
from solarlab.experiments.two_dimensional.inputs import JV2DInput

SCHEMA = "solarlab.classical_2d_dc.algebra.v1"
VOLTAGE_V = .002
DIRECTIONAL_STEP = 2e-4
DIRECTIONAL_RTOL = 2e-8  # Existing 1D field-transport directional-J criterion.


def spatial_input() -> JV2DInput:
    return JV2DInput(kind="jv_2d", lateral_length="1 um", Nx=4, Ny_per_layer=4, lateral_bc="neumann")


def coordinates(device: CompiledDevice2D, *, lateral: bool) -> tuple[np.ndarray, np.ndarray]:
    """Explicit admissible algebra states; no equilibrium/steady initialization."""
    g = device.geometry
    y = device.column.nodes_m / device.column.nodes_m[-1]
    x = g.x_m / g.lateral_length_m
    bottom, top = device.column.contact_potentials(VOLTAGE_V)
    wave = np.sin(np.pi * y)
    column = np.r_[bottom + (top - bottom) * y + .001 * wave, .03 * wave, -.02 * wave]
    result = device.extrude_coordinates(column)
    if lateral:
        count = device.node_count
        result[:count] += (.0007 * wave[:, None] * np.cos(np.pi * x)[None, :]).ravel()
        result[count:2 * count] += (.013 * wave[:, None] * np.sin(np.pi * x)[None, :]).ravel()
        result[2 * count:] += (-.011 * wave[:, None] * np.cos(2 * np.pi * x)[None, :]).ravel()
    return column, result


def assemble() -> dict[str, Any]:
    prepared = load_device()
    spatial = spatial_input()
    d = compile_device_2d(prepared, spatial)
    g, count = d.geometry, d.node_count
    u1, uniform = coordinates(d, lateral=False)
    _, u = coordinates(d, lateral=True)
    illuminated = DC2DProblem(d, VOLTAGE_V, 1.0)
    matrix = illuminated.jacobian(u)
    # G is state-independent. As in the accepted 1D test, remove it only from
    # finite differences to avoid losing small derivatives to a large constant.
    derivative_problem = DC2DProblem(d, VOLTAGE_V, 0.0)
    rng = np.random.default_rng(772)
    direction = np.r_[.02 * rng.normal(size=count), rng.normal(size=2 * count)]
    h = DIRECTIONAL_STEP
    numerical = (derivative_problem.residual(u - 2 * h * direction)
                 - 8 * derivative_problem.residual(u - h * direction)
                 + 8 * derivative_problem.residual(u + h * direction)
                 - derivative_problem.residual(u + 2 * h * direction)) / (12 * h)
    scale = np.maximum(abs(matrix) @ abs(direction), 1e-20)
    directional_error = np.abs(numerical - matrix @ direction) / scale
    row_widths = np.tile(g.x_widths_m, g.ny)
    row_widths[g.contact_nodes] = 1.0
    actual = (illuminated.residual(uniform) / np.tile(row_widths, 3)).reshape(3, count)
    column_rows = DC1DProblem(d.column, VOLTAGE_V, 1.0).residual(u1)
    reference = d.extrude_coordinates(column_rows).reshape(3, count)
    reduction = [float(np.max(abs(a - b)) / max(float(np.max(abs(b))), 1e-300)) for a, b in zip(actual, reference)]
    fluxes = illuminated.fluxes(u)
    summary = {
        "shape_Ny_Nx": g.shape, "unknowns": 3 * count, "x_faces": g.x_face_count,
        "y_faces": g.face_spacing_m.size - g.x_face_count, "jacobian_nnz": matrix.nnz,
        "prepared_identity": prepared.content_sha256, "compiled_identity": d.identity,
        "geometry_identity": g.sha256, "generation_identity": d.generation_y.binding_sha256,
        "control_area_m2": float(np.sum(g.contract_geometry.volumes)),
        "physical_area_m2": g.lateral_length_m * float(d.column.nodes_m[-1] - d.column.nodes_m[0]),
        "generation_limit_A_per_m_depth": d.generation_limit_A_m,
        "generation_limit_A_m2": d.generation_limit_A_m / g.lateral_length_m,
        "uniform_reduction_relative_by_block": reduction,
        "directional_error_by_block": [float(np.max(v)) for v in directional_error.reshape(3, count)],
        "directional_step": DIRECTIONAL_STEP, "directional_rtol": DIRECTIONAL_RTOL,
        "illuminated_jacobian_identical": bool(np.array_equal(matrix.toarray(), derivative_problem.jacobian(u).toarray())),
        "lateral_electron_flux_max_m2_s": float(np.max(abs(fluxes.electron.flux[:g.x_face_count]))),
        "lateral_hole_flux_max_m2_s": float(np.max(abs(fluxes.hole.flux[:g.x_face_count]))),
        "lateral_displacement_max_C_m2": float(np.max(abs(illuminated.displacement(u)[:g.x_face_count]))),
        "lateral_boundary_condition": "zero_normal_displacement_and_particle_flux",
        "newton_calls": 0, "accepted_steady_state": False, "qualification": False,
    }
    return {"schema": SCHEMA, "summary": summary, "spatial_declaration": spatial.editing_data(),
            "device_declaration": prepared.to_mapping(), "state": d.state(u, voltage_V=VOLTAGE_V),
            "column_coordinates": u1, "uniform_coordinates": uniform,
            "uniform_residual": illuminated.residual(uniform), "residual": illuminated.residual(u),
            "jacobian_csc": {"data": matrix.data, "indices": matrix.indices, "indptr": matrix.indptr, "shape": matrix.shape},
            "direction": direction, "analytic_direction": matrix @ direction, "numerical_direction": numerical,
            "directional_scale": scale, "x_m": g.x_m, "y_m": d.column.nodes_m,
            "cell_areas_m2": g.contract_geometry.volumes, "face_lengths_m": g.contract_geometry.face_measures,
            "face_pairs": g.contract_geometry.pairs, "face_spacing_m": g.face_spacing_m,
            "face_displacement_C_m2": illuminated.displacement(u), "carrier_fluxes": asdict(fluxes),
            "generation_source_m3_s": d.generation_source_m3_s,
            "source_profile_y": d.generation_y,
            "terminal_current_A_per_m_depth": illuminated.terminal_current_A_m(u),
            "terminal_mean_current_A_m2": illuminated.terminal_current_density_A_m2(u),
            "cross_section_current_A_per_m_depth": illuminated.cross_section_current_A_m(u)}


def read_result(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict) or value.get("schema") != SCHEMA:
        raise ValueError("not a 2D DC algebra result")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path)
    mode.add_argument("--read", type=Path)
    args = parser.parse_args()
    if args.read is not None:
        print(json.dumps(read_result(args.read)["summary"], indent=2))
        return 0
    if args.output.exists():
        raise FileExistsError("output exists; preserve the original algebra evidence")
    value = assemble()
    raw = (json.dumps(json_value(value), indent=2, allow_nan=False) + "\n").encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write(raw)
    print(json.dumps({"result_path": str(args.output.resolve()), "sha256": hashlib.sha256(raw).hexdigest(), **value["summary"]}, indent=2))
    return 0 if max(value["summary"]["directional_error_by_block"]) < DIRECTIONAL_RTOL else 2


if __name__ == "__main__":
    raise SystemExit(main())
