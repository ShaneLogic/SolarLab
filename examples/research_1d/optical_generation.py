"""Illustrative supplied spectral/n-k data driving one new-core steady state.

Run from the repository root:
    python -m examples.research_1d.optical_generation --output optical.json
    python -m examples.research_1d.optical_generation --read optical.json

The seven-point photon spectrum and constant n-k table are illustrative
declarations, not an AM1.5 spectrum or calibrated optical material data.
The first optical binding fixes front normal incidence and unit-index bounding
media; scalar Phi is explicitly zero and does not scale the supplied spectrum.
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
from examples.research_1d.transient import json_value
from solarlab.config.resolve_device import resolve_device
from solarlab.device.compile_1d import compile_device_1d
from solarlab.device.inputs import DeviceInput
from solarlab.device.optical_generation import PHOTON_BUDGET_RTOL, bind_optical_generation_1d
from solarlab.device.resolved import PreparedDevice
from solarlab.equations.dc_1d import DC1DProblem
from solarlab.materials.resources import ResourceLibrary, ResourceTable
from solarlab.materials.source import SourceDocument
from solarlab.solvers.nonlinear.steady_1d import SteadyOptions, SteadyStateError, solve_steady_state

INTERVALS = (12, 12, 12)
VOLTAGE_V = .002
GENERATION_SCALE = 1.0
SCHEMA = "solarlab.optical_generation_1d.research.v1"


def prepare_case() -> PreparedDevice:
    base = load_device()
    wavelengths = (400, 450, 500, 550, 600, 650, 700)
    photon_csv = "wavelength_nm,spectral_flux_m-2_s-1_m-1\n" + "".join(f"{wavelength},6e27\n" for wavelength in wavelengths)
    nk_csv = "wavelength_nm,n,k\n" + "".join(f"{wavelength},2.5,0.1\n" for wavelength in wavelengths)
    tables = (
        ResourceTable("illustrative_photons", "spectrum", SourceDocument("research_1d.illustrative_photons.csv", photon_csv.encode())),
        ResourceTable("illustrative_absorber_nk", "nk", SourceDocument("research_1d.illustrative_absorber_nk.csv", nk_csv.encode())),
    )
    declaration = base.to_input().editing_data()
    declaration["spectrum"] = tables[0].name
    declaration["settings"]["Phi"] = 0.0
    for layer in declaration["layers"]:
        parameters = layer["parameters"]
        parameters.update(alpha=0.0, incoherent=False)
        if layer["id"] == "absorber":
            parameters.update(optical_material=tables[1].name, n_optical=None)
        else:
            parameters.update(optical_material=None, n_optical=2.0)
    raw = json.dumps(declaration, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    source = SourceDocument("research_1d.optical_generation.declaration", raw)
    return resolve_device(DeviceInput.model_validate(declaration), base.defaults, ResourceLibrary(tables),
                          sources=(*base.sources, source))


def read_result(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict) or value.get("schema") != SCHEMA:
        raise ValueError("not an optical-generation research result")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path)
    mode.add_argument("--read", type=Path)
    args = parser.parse_args()
    if args.read is not None:
        print(json.dumps(read_result(args.read)["summary"], indent=2), flush=True)
        return 0
    if args.output.exists():
        raise FileExistsError("output exists; preserve the original attempt")
    prepared = prepare_case()
    device = compile_device_1d(prepared, intervals_per_layer=INTERVALS)
    optical = bind_optical_generation_1d(prepared, device.geometry)
    device.generation.require_inputs(optical.binding.sha256, device.geometry)
    controls = SteadyOptions()
    budget = optical.budget
    summary: dict[str, Any] = {
        "node_count": device.node_count, "prepared_identity": prepared.content_sha256,
        "compiled_identity": device.identity, "optical_binding_identity": optical.binding.sha256,
        "generation_method": device.generation.method,
        "optical_data_scope": "illustrative_declared_spectrum_and_indices",
        "incident_photons_m2_s": budget.incident_m2_s,
        "reflected_photons_m2_s": budget.reflected_m2_s,
        "transmitted_photons_m2_s": budget.transmitted_m2_s,
        "electrical_absorbed_photons_m2_s": budget.electrical_absorbed_m2_s,
        "parasitic_absorbed_photons_m2_s": budget.parasitic_absorbed_m2_s,
        "photon_budget_roundoff_rtol": PHOTON_BUDGET_RTOL,
        "generation_current_limit_A_m2": device.generation_limit_A_m2,
        "voltage_V": VOLTAGE_V, "generation_scale": GENERATION_SCALE,
        "controls": asdict(controls), "physical_certificate_status": "not_evaluated", "qualification": False,
    }
    optical_record = asdict(optical.binding)
    optical_record["physical_layer_slices"] = [
        {"name": name, "start": part.start, "stop": part.stop}
        for name, part in optical.binding.physical_layer_slices
    ]
    payload: dict[str, Any] = {
        "schema": SCHEMA, "summary": summary, "generation": device.generation,
        "optical_binding": optical_record, "photon_budget": budget,
        "prepared_declaration": prepared.to_mapping(),
        "input_resources": [{"name": table.name, "kind": table.kind, "source_id": table.source.id,
                              "source_sha256": table.source.sha256, "content_sha256": table.content_sha256,
                              "source_utf8": table.source.content.decode()} for table in prepared.resources.tables],
    }
    started = time.monotonic()
    try:
        result = solve_steady_state(device, voltage_V=VOLTAGE_V, generation_scale=GENERATION_SCALE, options=controls)
    except SteadyStateError as error:
        summary.update(accepted=False, failure_stage=error.stage, error=str(error))
        payload["failed_newton"] = error.result
        exit_code = 2
    else:
        summary.update(accepted=result.converged,
                       output_current_A_m2=float(-device.voltage_polarity * result.contact_current_density_A_m2[0]),
                       current_spread_A_m2=float(np.ptp(np.r_[result.current_density_A_m2, result.contact_current_density_A_m2])))
        payload.update(result=result, original_residual=DC1DProblem(device, VOLTAGE_V, GENERATION_SCALE).residual(result.state.u))
        exit_code = 0
    summary["solve_wall_s"] = time.monotonic() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(json_value(payload), indent=2, allow_nan=False) + "\n").encode()
    with args.output.open("xb") as stream:
        stream.write(raw)
    print(json.dumps({"result_path": str(args.output.resolve()), "result_sha256": hashlib.sha256(raw).hexdigest(),
                      **summary}, indent=2), flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
