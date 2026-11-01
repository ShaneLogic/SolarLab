"""Smooth CT AC from checksum-bound accepted DC, without a new DC solve.

    .venv/bin/python -m examples.research_1d.field_transport_ac solve \
        --kind ion-free --initial-result Steady.json --initial-sha256 SHA \
        --output AC.json
    .venv/bin/python -m examples.research_1d.field_transport_ac read \
        --result AC.json --sha256 SHA

Ionic inputs require an accepted result for prepare_ionic_case(kind), with
the original explicit blocking absorber domain. A 1 V phasor denotes a
derivative; the default small complex normalization tests the same tangent.
This research example does not establish scientific qualification.
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

from examples.research_1d.field_transport import INTERVALS, prepare_case
from examples.research_1d.ionic_multilayer import (
    prepare_confined_dual_ion_device, prepare_confined_ion_device,
)
from examples.research_1d.ionic_steady import controls as ionic_steady_controls
from examples.research_1d.shared_site_ionic_fixture import prepare_confined_shared_site_ion_device
from solarlab.config.resolve_device import resolve_device
from solarlab.device.compile_1d import compile_device_1d
from solarlab.device.compile_ionic_1d import compile_ionic_device_1d
from solarlab.device.inputs import DeviceInput
from solarlab.device.resolved import PreparedDevice
from solarlab.equations.dc_1d import CompiledDevice1D
from solarlab.experiments.impedance import SmallSignalResult1D, solve_impedance
from solarlab.experiments.ionic_impedance import (
    IonicImpedanceOptions, IonicImpedanceResult, _decoded, _encoded, solve_ionic_impedance,
)
from solarlab.experiments.ionic_transient import load_ionic_steady_initial
from solarlab.materials.source import SourceDocument
from solarlab.solvers.linear import FrequencyOptions
from solarlab.solvers.nonlinear.newton import IterationRecord, NewtonResult, Termination
from solarlab.solvers.nonlinear.problem import GateOutcome
from solarlab.solvers.nonlinear.steady_1d import SteadyOptions, SteadyState1D

SCHEMA = "solarlab.field_transport_ac.research.v1"


def prepare_ionic_case(kind: str = "shared") -> PreparedDevice:
    """Original ionic fixture and backgrounds plus absorber-only smooth CT."""
    factories = {"positive": prepare_confined_ion_device,
                 "independent": prepare_confined_dual_ion_device,
                 "shared": prepare_confined_shared_site_ion_device}
    if kind not in factories:
        raise ValueError("ionic kind must be positive, independent or shared")
    base = factories[kind]()
    declaration = base.to_input().editing_data()
    absorber = next(layer for layer in declaration["layers"] if layer["id"] == "absorber")
    absorber["parameters"].update(v_sat_n=1e4, v_sat_p=1.5e4, ct_beta_n=2.0, ct_beta_p=2.0)
    raw = json.dumps(declaration, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    source = SourceDocument("research_1d.field_transport_ionic.declaration", raw)
    return resolve_device(DeviceInput.model_validate(declaration), base.defaults, base.resources,
                          sources=(*base.sources, source))


def _newton(saved: dict[str, Any]) -> NewtonResult:
    return NewtonResult(
        np.asarray(saved["state"], dtype=float), saved["accepted"], Termination(saved["termination"]),
        saved["scaled_residual"], tuple(GateOutcome(**gate) for gate in saved["gates"]),
        tuple(IterationRecord(**item) for item in saved["iterations"]),
        saved["residual_evaluations"], saved["jacobian_evaluations"], saved["start_digest"], saved["message"],
    )


def load_ion_free_initial(path: Path, sha256: str) -> tuple[CompiledDevice1D, SteadyState1D, SteadyOptions]:
    """Restore the full pinned CT example result; AC rechecks its physical gates."""
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha256:
        raise ValueError("initial DC artifact SHA-256 mismatch")
    document = json.loads(raw)
    if document.get("schema") != "solarlab.field_transport_1d.research.v1":
        raise ValueError("initial must be an actual field-transport DC result")
    summary, saved = document["summary"], document["result"]
    if (summary["accepted"] is not True or saved["newton"]["accepted"] is not True
            or saved["newton"]["termination"] != "converged"):
        raise ValueError("initial must retain an accepted DC result")
    prepared = prepare_case()
    compiled = compile_device_1d(prepared, intervals_per_layer=INTERVALS)
    transport = compiled.field_transport
    if (transport is None or compiled.identity != summary["compiled_identity"]
            or prepared.content_sha256 != summary["prepared_identity"]
            or transport.identity_sha256 != summary["transport_identity"]
            or prepared.to_mapping() != document["prepared_declaration"]):
        raise ValueError("initial configuration/compiled/transport identity differs")
    state = compiled.state(saved["state"]["u"], voltage_V=summary["voltage_V"])
    if (saved["state"]["device_sha256"] != compiled.identity
            or saved["state"]["voltage_V"] != state.voltage_V
            or saved["voltage_V"] != summary["voltage_V"]
            or saved["generation_scale"] != summary["generation_scale"]
            or any(not np.array_equal(saved["state"][name], getattr(state, name))
                   for name in ("potential_V", "electron_density_m3", "hole_density_m3"))):
        raise ValueError("initial DC physical fields or voltage disagree with coordinates")
    point = SteadyState1D(
        state, saved["current_density_A_m2"], saved["contact_current_density_A_m2"],
        saved["voltage_V"], saved["generation_scale"], _newton(saved["newton"]),
        tuple(_newton(item) for item in saved["preparation"]),
        saved["density_basin_initializations"], tuple(saved["density_basin_stages"]),
    )
    return compiled, point, SteadyOptions(**summary["controls"])


def read_result(path: Path, sha256: str) -> dict[str, Any]:
    """Checksum-checked saved reading; does not rebuild or solve a device."""
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha256:
        raise ValueError("CT AC artifact SHA-256 mismatch")
    record = _decoded(json.loads(raw))
    if not isinstance(record, dict) or record.get("schema") != SCHEMA:
        raise ValueError("not a CT AC research result")
    return record


def run_response(kind: str, initial: Path, sha256: str, output: Path,
                 frequencies_Hz: tuple[float, ...], *, voltage_phasor_V: complex = 2e-6 + 1e-6j,
                 ion_mode: str = "mobile") -> dict[str, Any]:
    """One native AC sweep; retain the first failure and never run a DC solve."""
    if output.exists():
        raise FileExistsError("preserve the original AC attempt; output already exists")
    summary: dict[str, Any] = {"kind": kind, "requested_frequencies_Hz": frequencies_Hz,
        "initial_path": str(initial.resolve()), "initial_sha256": sha256,
        "voltage_phasor_V": voltage_phasor_V, "ion_mode": ion_mode,
        "physical_certificate_status": "not_evaluated", "qualification": False}
    record: dict[str, Any] = {"schema": SCHEMA, "summary": summary}
    started = time.monotonic()
    result: SmallSignalResult1D | IonicImpedanceResult
    try:
        if kind == "ion-free":
            compiled, point, controls = load_ion_free_initial(initial, sha256)
            summary.update(compiled_identity=compiled.identity, linear_controls=asdict(FrequencyOptions()))
            result = solve_impedance(compiled, point, frequencies_Hz,
                voltage_phasor_V=voltage_phasor_V, steady_options=controls, linear_options=FrequencyOptions())
            summary.update(accepted=True, maxwell_spread_A_m2_V=result.maxwell_spread_A_m2_V,
                           admittance_density_S_m2=result.admittance_density_S_m2)
        else:
            ionic = compile_ionic_device_1d(prepare_ionic_case(kind), intervals_per_layer=INTERVALS,
                ion_domain_layer="absorber", ion_domain_endpoints=("blocking", "blocking"))
            source = load_ionic_steady_initial(ionic, initial, sha256,
                controls=ionic_steady_controls(ionic.ion_node_count))
            summary.update(compiled_identity=ionic.identity, linear_controls=asdict(IonicImpedanceOptions()))
            result = solve_ionic_impedance(ionic, source, frequencies_Hz,
                voltage_phasor_V=voltage_phasor_V, ion_mode=ion_mode, options=IonicImpedanceOptions())
            summary.update(accepted=result.completed, accepted_frequencies_Hz=[
                item.frequency_Hz for item in result.attempts if item.accepted],
                failed_frequencies_Hz=[item.frequency_Hz for item in result.failed_attempts])
        record["result"] = result
    except (ValueError, ArithmeticError, RuntimeError) as error:
        summary.update(accepted=False, failure_type=type(error).__name__, error=str(error))
    summary["wall_s"] = time.monotonic() - started
    output.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(_encoded(record), indent=2, allow_nan=False) + "\n").encode()
    with output.open("xb") as stream:
        stream.write(raw)
    print(json.dumps(_encoded({"result_path": str(output.resolve()),
        "result_sha256": hashlib.sha256(raw).hexdigest(), **summary}), indent=2), flush=True)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    solve = commands.add_parser("solve")
    solve.add_argument("--kind", choices=("ion-free", "positive", "independent", "shared"), required=True)
    solve.add_argument("--initial-result", type=Path, required=True)
    solve.add_argument("--initial-sha256", required=True)
    solve.add_argument("--output", type=Path, required=True)
    solve.add_argument("--frequencies", nargs="+", type=float, default=(0.0, 1.0, 1000.0))
    solve.add_argument("--voltage-phasor", type=complex, default=2e-6 + 1e-6j)
    solve.add_argument("--ion-mode", choices=("mobile", "frozen"), default="mobile")
    read = commands.add_parser("read")
    read.add_argument("--result", type=Path, required=True)
    read.add_argument("--sha256", required=True)
    args = parser.parse_args()
    if args.command == "read":
        print(json.dumps(_encoded(read_result(args.result, args.sha256)["summary"]), indent=2))
        return 0
    record = run_response(args.kind, args.initial_result, args.initial_sha256, args.output,
                          tuple(args.frequencies), voltage_phasor_V=args.voltage_phasor, ion_mode=args.ion_mode)
    return 0 if record["summary"]["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
