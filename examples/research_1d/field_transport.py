"""One illuminated new-core steady solve with smooth CT velocity saturation.

Run from the repository root:
    python -m examples.research_1d.field_transport --output field_transport.json
    python -m examples.research_1d.field_transport --read field_transport.json

The default adds absorber-only CT to the existing ion-free device.json. An
explicit --config supplies its own complete declaration. PF is unsupported.
This bounded research example does not establish scientific qualification.
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
from solarlab.device.resolved import PreparedDevice
from solarlab.equations.dc_1d import DC1DProblem
from solarlab.materials.source import SourceDocument
from solarlab.solvers.nonlinear.steady_1d import (
    SteadyOptions, SteadyStateError, solve_steady_state,
)

VOLTAGE_V = 0.002
GENERATION_SCALE = 1.0
INTERVALS = (12, 12, 12)
SCHEMA = "solarlab.field_transport_1d.research.v1"


def prepare_case(config: Path | None = None) -> PreparedDevice:
    if config is not None:
        return load_device(config)
    base = load_device()
    declaration = base.to_input().editing_data()
    absorber = next(layer for layer in declaration["layers"] if layer["id"] == "absorber")
    absorber["parameters"].update(v_sat_n=1.0e4, v_sat_p=1.5e4, ct_beta_n=2.0, ct_beta_p=2.0)
    raw = json.dumps(declaration, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    source = SourceDocument("research_1d.field_transport.declaration", raw)
    return resolve_device(DeviceInput.model_validate(declaration), base.defaults, base.resources,
                          sources=(*base.sources, source))


def read_result(path: Path) -> dict[str, Any]:
    """Read the saved research result without importing a solver state."""
    value = json.loads(path.read_text())
    if not isinstance(value, dict) or value.get("schema") != SCHEMA:
        raise ValueError("not a field-transport research result")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path)
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--output", type=Path)
    destination.add_argument("--read", type=Path)
    args = parser.parse_args()
    if args.read is not None:
        print(json.dumps(read_result(args.read)["summary"], indent=2), flush=True)
        return 0
    if args.output.exists():
        raise FileExistsError("the output already exists; retain the original attempt")
    prepared = prepare_case(args.config)
    device = compile_device_1d(prepared, intervals_per_layer=INTERVALS)
    model = device.field_transport
    if model is None:
        raise ValueError("this example requires nonzero configured CT saturation")
    controls = SteadyOptions()
    summary: dict[str, Any] = {
        "node_count": device.node_count, "prepared_identity": prepared.content_sha256,
        "compiled_identity": device.identity, "transport_identity": model.identity_sha256,
        "source_documents": [{"id": source.id, "sha256": source.sha256} for source in prepared.sources],
        "field_measure": model.field_measure,
        "voltage_V": VOLTAGE_V, "generation_scale": GENERATION_SCALE,
        "controls": asdict(controls), "physical_certificate_status": "not_evaluated",
        "qualification": False,
    }
    payload: dict[str, Any] = {"schema": SCHEMA, "summary": summary,
                              "binding": asdict(model), "prepared_declaration": prepared.to_mapping()}
    started = time.monotonic()
    try:
        result = solve_steady_state(device, voltage_V=VOLTAGE_V,
                                    generation_scale=GENERATION_SCALE, options=controls)
    except SteadyStateError as error:
        summary.update(accepted=False, failure_stage=error.stage, error=str(error))
        payload["failed_newton"] = error.result
        exit_code = 2
    else:
        state = result.state
        transport = model.evaluate(state.potential_V)
        problem = DC1DProblem(device, VOLTAGE_V, GENERATION_SCALE)
        summary.update(accepted=result.converged,
                       current_spread_A_m2=float(np.ptp(np.r_[result.current_density_A_m2,
                                                             result.contact_current_density_A_m2])),
                       output_current_A_m2=float(-device.voltage_polarity * result.contact_current_density_A_m2[0]),
                       minimum_electron_diffusivity_ratio=float(np.min(transport.electron_diffusivity_m2_s /
                                                                       device.electron_diffusivity_faces_m2_s)),
                       minimum_hole_diffusivity_ratio=float(np.min(transport.hole_diffusivity_m2_s /
                                                                   device.hole_diffusivity_faces_m2_s)),
                       maximum_abs_electron_dD_dE=float(np.max(np.abs(transport.electron_d_diffusivity_d_field))),
                       maximum_abs_hole_dD_dE=float(np.max(np.abs(transport.hole_d_diffusivity_d_field))))
        payload.update(result=result, transport=transport, original_residual=problem.residual(state.u))
        exit_code = 0
    summary["solve_wall_s"] = time.monotonic() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(json_value(payload), indent=2, allow_nan=False) + "\n").encode()
    with args.output.open("xb") as stream:
        stream.write(raw)
    print(json.dumps({"result_path": str(args.output.resolve()),
                      "result_sha256": hashlib.sha256(raw).hexdigest(), **summary}, indent=2), flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
