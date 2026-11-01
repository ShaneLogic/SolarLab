"""One illuminated new-core steady solve with an explicit neutral bulk defect.

Run from the repository root:
    python -m examples.research_1d.neutral_defects --output neutral.json
    python -m examples.research_1d.neutral_defects --unequal-dos --output neutral_unequal_dos.json

The default declaration adds one absorber species to the existing device.json
fixture. An explicit --config supplies its own complete validated declaration.
This research result does not establish final scientific qualification.
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
from solarlab.device.neutral_defects import LITERAL_ABOVE_VB_DOS_REFERENCE_V1
from solarlab.device.resolved import PreparedDevice
from solarlab.equations.dc_1d import DC1DProblem
from solarlab.materials.source import SourceDocument
from solarlab.solvers.nonlinear.steady_1d import (
    SteadyOptions, SteadyStateError, solve_steady_state,
)

VOLTAGE_V = 0.002
GENERATION_SCALE = 1.0
INTERVALS = (12, 12, 12)


def defect_declaration() -> dict[str, Any]:
    """Physical parameters in the existing validated single-level schema."""
    return {
        "id": "neutral_1", "name": "Neutral absorber level",
        "distribution": {"kind": "single_level", "normalization": "integrated_total",
                         "total_density_m3": 1.0e20, "center_eV_above_vb": 0.75},
        "charge_transition": "neutral", "neutral_reference": "all_occupancies",
        "kinetics": {"sigma_n_m2": 1.0e-19, "sigma_p_m2": 2.0e-19,
                     "thermal_velocity_n_m_s": 1.0e5, "thermal_velocity_p_m_s": 1.0e5},
        "degeneracy": 1.0,
    }


def prepare_case(config: Path | None = None) -> PreparedDevice:
    if config is not None:
        return load_device(config)
    base = load_device()
    declaration = base.to_input().editing_data()
    absorber = next(layer for layer in declaration["layers"] if layer["id"] == "absorber")
    absorber.update(defect_schema_version="solarlab-explicit-bulk-defects-v1",
                    defect_model="explicit_quasi_steady", bulk_defects=[defect_declaration()])
    raw = json.dumps(declaration, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    source = SourceDocument("research_1d.neutral_defects.declaration", raw)
    return resolve_device(DeviceInput.model_validate(declaration), base.defaults, base.resources,
                          sources=(*base.sources, source))


def prepare_unequal_dos_case() -> PreparedDevice:
    """Explicit physical variant: double absorber Nv, with consistent ni²."""
    base = prepare_case()
    declaration = base.to_input().editing_data()
    absorber = next(layer for layer in declaration["layers"] if layer["id"] == "absorber")
    parameters = absorber["parameters"]
    parameters["Nv300"] *= 2.0
    for name in ("ni", "n1", "p1"):
        parameters[name] *= 2.0**0.5
    raw = json.dumps(declaration, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    source = SourceDocument("research_1d.neutral_defects.unequal_dos.declaration", raw)
    return resolve_device(DeviceInput.model_validate(declaration), base.defaults, base.resources,
                          sources=(*base.sources, source))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--unequal-dos", action="store_true",
                        help="use the explicit default-case variant with doubled absorber Nv")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("the output already exists; retain the original attempt")
    if args.unequal_dos and args.config is not None:
        raise ValueError("--unequal-dos selects the declared default variant; use --config for a complete custom input")
    prepared = prepare_unequal_dos_case() if args.unequal_dos else prepare_case(args.config)
    device = compile_device_1d(prepared, intervals_per_layer=INTERVALS)
    model = device.recombination.explicit_defects
    if model is None:
        raise ValueError("this example requires a configured explicit neutral inventory")
    controls = SteadyOptions()
    unequal = model.owned_nodes & (device.parameters.conduction_dos_m3 != device.parameters.valence_dos_m3)
    summary: dict[str, Any] = {
        "node_count": device.node_count, "prepared_identity": prepared.content_sha256,
        "compiled_identity": device.identity, "defect_model_identity": model.identity_sha256,
        "defect_reference_binding": (LITERAL_ABOVE_VB_DOS_REFERENCE_V1 if np.any(unequal)
                                     else "historical_def1_equal_dos"),
        "unequal_dos_owned_node_count": int(np.count_nonzero(unequal)),
        "source_documents": [{"id": source.id, "sha256": source.sha256} for source in prepared.sources],
        "voltage_V": VOLTAGE_V, "generation_scale": GENERATION_SCALE,
        "controls": asdict(controls), "physical_certificate_status": "not_evaluated",
        "qualification": False,
    }
    payload: dict[str, Any] = {"summary": summary, "binding": asdict(model),
                              "prepared_declaration": prepared.to_mapping(),
                              "physical_reference_inputs": {
                                  "conduction_dos_m3": device.parameters.conduction_dos_m3,
                                  "valence_dos_m3": device.parameters.valence_dos_m3,
                                  "band_gap_eV": device.parameters.band_structure.physical_band_gap_eV,
                                  "ni_sq_m6": device.parameters.intrinsic_product_m6,
                                  "thermal_voltage_V": device.thermal_voltage_V,
                              }}
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
        evaluation = device.recombination.evaluate(state.electron_density_m3, state.hole_density_m3)
        explicit = evaluation.explicit
        assert explicit is not None
        problem = DC1DProblem(device, VOLTAGE_V, GENERATION_SCALE)
        summary.update(accepted=result.converged,
                       current_spread_A_m2=float(np.ptp(np.r_[result.current_density_A_m2,
                                                             result.contact_current_density_A_m2])),
                       output_current_A_m2=float(-device.voltage_polarity * result.contact_current_density_A_m2[0]),
                       explicit_owned_node_count=int(np.count_nonzero(model.owned_nodes)),
                       maximum_explicit_rate_m3_s=float(np.max(np.abs(explicit.total_rate_m3_s))),
                       maximum_explicit_d_rate_dn_s1=float(np.max(np.abs(explicit.total_d_rate_dn_s1))),
                       maximum_explicit_d_rate_dp_s1=float(np.max(np.abs(explicit.total_d_rate_dp_s1))),
                       trap_charge_density_C_m3=None)
        payload.update(result=result, recombination=evaluation,
                       original_residual=problem.residual(state.u),
                       charge_density_C_m3=device.charge_ledger(state.electron_density_m3,
                                                                state.hole_density_m3).space_charge_C_m3())
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
