"""Actual uniform-column spectral optical source on13x5 half-height cells.

    .venv/bin/python -m examples.research_2d.optical_generation --output Optical.json
    .venv/bin/python -m examples.research_2d.optical_generation --read Optical.json --sha256 SHA

The supplied seven-point spectrum and n-k table reuse the illustrative1D
declaration. This constructs an optical2D source, without a carrier solve,
lateral electromagnetic propagation, irradiance/PCE or scientific qualification.
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

from examples.research_1d.optical_generation import prepare_case as prepare_1d_case
from solarlab.contracts.units import ELEMENTARY_CHARGE_C as Q
from solarlab.device.optical_generation import PHOTON_BUDGET_RTOL
from solarlab.device.optical_generation_2d import bind_optical_generation_2d
from solarlab.device.resolved import PreparedDevice
from solarlab.experiments.transient_results import _decode, _encode
from solarlab.mesh import multilayer_grid
from solarlab.mesh.geometry import half_endpoint_geometry
from solarlab.mesh.geometry_2d import RectangularGeometry2D

SCHEMA = "solarlab.optical_source_2d.research.v1"


def research_geometry(prepared: PreparedDevice) -> RectangularGeometry2D:
    layers = tuple(layer for layer in prepared.layers if layer.role != "substrate")
    grid = {entry["layer"]: entry for entry in prepared.to_mapping()["electrical_grid"]}
    thicknesses = [layer.thickness for layer in layers]
    nodes = multilayer_grid([(value, 4) for value in thicknesses],
                           alpha=[grid.get(layer.id, {}).get("alpha", 3.0) for layer in layers])
    return RectangularGeometry2D(np.linspace(0.0, 1e-6, 5),
        half_endpoint_geometry(nodes, np.cumsum(thicknesses)[:-1]))


def source_hashes() -> dict[str, str]:
    names = (__name__, "examples.research_1d.optical_generation", "solarlab.device.optical_generation_2d",
        "solarlab.device.optical_generation", "solarlab.physics.generation.optical_binding",
        "solarlab.physics.generation.transfer_matrix", "solarlab.physics.generation.spectrum",
        "solarlab.physics.generation.profile", "solarlab.physics.generation._validation",
        "solarlab.materials.resources", "solarlab.mesh.grid", "solarlab.mesh.geometry", "solarlab.mesh.geometry_2d")
    result = {}
    for name in names:
        filename = sys.modules[name].__file__
        if filename is None:
            raise ValueError("actual optical source files are required")
        result[name] = hashlib.sha256(Path(filename).read_bytes()).hexdigest()
    return result


def construct(output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError("preserve the first optical2D source receipt")
    prepared = prepare_1d_case()
    geometry = research_geometry(prepared)
    optical = bind_optical_generation_2d(prepared, geometry)
    optical.require_inputs(prepared, geometry)
    binding = asdict(optical.optical_y.binding)
    binding["physical_layer_slices"] = [{"name": name, "start": part.start, "stop": part.stop}
                                       for name, part in optical.optical_y.binding.physical_layer_slices]
    budget = optical.optical_y.budget
    summary = {"geometry_shape": geometry.shape, "source_cell_count": geometry.node_count,
        "prepared_identity": prepared.content_sha256, "geometry_identity": geometry.sha256,
        "optical2d_identity": optical.identity_sha256, "incident_photons_m2_s": budget.incident_m2_s,
        "reflected_photons_m2_s": budget.reflected_m2_s, "transmitted_photons_m2_s": budget.transmitted_m2_s,
        "electrical_absorbed_photons_m2_s": budget.electrical_absorbed_m2_s,
        "parasitic_absorbed_photons_m2_s": budget.parasitic_absorbed_m2_s,
        "particle_generation_per_depth_m_s": optical.particle_generation_per_depth_m_s,
        "generation_current_limit_A_m": optical.generation_current_limit_A_m,
        "generation_current_limit_A_m2": optical.generation_current_limit_A_m2,
        "photon_budget_roundoff_rtol": PHOTON_BUDGET_RTOL,
        "source_constructed": True, "coupled_2d_operating_point_accepted": False,
        "nonlinear_solve_calls": 0, "qualification": False}
    record = {"schema": SCHEMA, "summary": summary, "source_m3_s": optical.source_m3_s,
        "cell_generation_per_depth_m_s": optical.cell_generation_per_depth_m_s,
        "generation_y": optical.optical_y.profile, "photon_budget": budget, "optical_binding": binding,
        "x_m": geometry.x_m, "y_m": geometry.y_geometry.nodes_m,
        "x_cell_widths_m": geometry.x_widths_m, "y_cell_faces_m": geometry.y_geometry.cell_faces_m,
        "y_cell_widths_m": geometry.y_geometry.cell_measures_m, "cell_areas_m2": geometry.contract_geometry.volumes,
        "lateral_width_m": geometry.lateral_length_m, "prepared_declaration": prepared.to_mapping(),
        "input_resources": [{"name": item.name, "kind": item.kind, "source_id": item.source.id,
            "source_sha256": item.source.sha256, "content_sha256": item.content_sha256,
            "source_utf8": item.source.content.decode()} for item in prepared.resources.tables],
        "source_sha256": source_hashes(), "scope": "coherent normal/front, unit-index media, uniform lateral extrusion",
        "physical_certificate_status": "not_evaluated", "qualification": False}
    output.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(_encode(record), indent=2, allow_nan=False) + "\n").encode()
    with output.open("xb") as stream:
        stream.write(raw)
    print(json.dumps({"path": str(output.resolve()), "sha256": hashlib.sha256(raw).hexdigest(), **summary}, indent=2))
    return record


def read_result(path: Path, sha256: str) -> dict[str, Any]:
    """Validate the saved photon/unit relations; do not recompute TMM or DC."""
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha256:
        raise ValueError("optical2D source SHA mismatch")
    record = _decode(json.loads(raw))
    if record["schema"] != SCHEMA or record["qualification"] or record["summary"]["qualification"]:
        raise ValueError("invalid optical2D source receipt")
    summary, budget, binding = record["summary"], record["photon_budget"], record["optical_binding"]
    if not summary["source_constructed"] or summary["coupled_2d_operating_point_accepted"] or summary["nonlinear_solve_calls"] != 0:
        raise ValueError("source construction is distinct from coupled2D completion")
    if binding["incidence"] != "front" or binding["ambient_index"] != 1.0 or binding["exit_index"] != 1.0:
        raise ValueError("optical2D receipt has unsupported incidence/bounding media")
    source, areas, photons = (record[name] for name in ("source_m3_s", "cell_areas_m2", "cell_generation_per_depth_m_s"))
    ny, nx = len(record["y_m"]), len(record["x_m"])
    if source.shape != (ny * nx,) or areas.shape != source.shape or photons.shape != source.shape:
        raise ValueError("optical2D cells are incomplete")
    if not np.all(np.isfinite(source)) or np.any(source < 0.0) or np.any(areas <= 0.0):
        raise ValueError("invalid physical optical2D source/areas")
    np.testing.assert_array_equal(source.reshape(ny, nx), np.repeat(record["generation_y"]["source_m3_s"][:, None], nx, axis=1))
    np.testing.assert_allclose(source * areas, photons, rtol=PHOTON_BUDGET_RTOL, atol=0.0)
    np.testing.assert_allclose(areas.reshape(ny, nx), record["y_cell_widths_m"][:, None] * record["x_cell_widths_m"][None, :], rtol=0.0, atol=0.0)
    wavelength_m = budget["wavelengths_nm"] * 1e-9
    incident = float(np.trapezoid(budget["incident_spectral_m2_s_m"], x=wavelength_m))
    electrical_mask = np.asarray(budget["layer_electrical"], dtype=bool)
    electrical = float(np.trapezoid(budget["incident_spectral_m2_s_m"] * np.sum(budget["layer_absorptance"][electrical_mask], axis=0), x=wavelength_m))
    closure = 1.0 - budget["reflectance"] - budget["transmittance"] - np.sum(budget["layer_absorptance"], axis=0) - budget["dropped_multipass"]
    if np.max(abs(closure)) > PHOTON_BUDGET_RTOL:
        raise ValueError("saved spectral photon accounting does not close")
    width = float(record["x_m"][-1] - record["x_m"][0])
    tolerance = PHOTON_BUDGET_RTOL * incident
    if width != record["lateral_width_m"] or abs(float(np.sum(source * areas)) / width - electrical) > tolerance:
        raise ValueError("saved physical2D source does not close the electrical photon budget")
    if abs(summary["generation_current_limit_A_m2"] - Q * float(np.sum(source * areas)) / width) > Q * tolerance:
        raise ValueError("saved optical2D current normalization is inconsistent")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path)
    mode.add_argument("--read", type=Path)
    parser.add_argument("--sha256")
    args = parser.parse_args()
    if args.read is not None:
        if args.sha256 is None:
            parser.error("--read requires --sha256")
        record = read_result(args.read, args.sha256)
        print(json.dumps(record["summary"], indent=2))
    else:
        construct(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
