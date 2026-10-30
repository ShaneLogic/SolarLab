"""One positive-D two-domain dark cold steady witness, preserving first failure."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path

from examples.research_1d.ionic_steady import controls
from examples.research_1d.multiple_ionic_domains import DOMAINS, INTERVALS, prepare_multiple_ionic_device
from solarlab.config.resolve_device import resolve_device
from solarlab.device.compile_multiple_ionic_1d import compile_multiple_ionic_device_1d
from solarlab.device.inputs import DeviceInput
from solarlab.device.resolved import PreparedDevice
from solarlab.equations.implementation import implementation_source_modules
from solarlab.experiments.transient_results import _encode
from solarlab.solvers.nonlinear.multiple_ionic_1d import MultipleIonicSteadyResult, solve_multiple_ionic_steady_state


def prepare_mixed_immobile_device() -> PreparedDevice:
    """One right shared-site D0 population; every other species stays mobile."""
    prepared = prepare_multiple_ionic_device(kind="shared")
    data = prepared.to_input().editing_data()
    next(layer for layer in data["layers"] if layer["id"] == "absorber_right")["parameters"]["D_ion_neg"] = 0.0
    return resolve_device(DeviceInput.model_validate(data), prepared.defaults, prepared.resources, sources=prepared.sources)


def run_case(output: Path, *, mixed_immobile: bool = False) -> MultipleIonicSteadyResult:
    if output.exists():
        raise FileExistsError("preserve the existing attempt; no silent retry")
    device = compile_multiple_ionic_device_1d(prepare_mixed_immobile_device() if mixed_immobile else prepare_multiple_ionic_device(),
        intervals_per_layer=INTERVALS, ion_domains=DOMAINS)
    result = solve_multiple_ionic_steady_state(device, voltage_V=0.002, generation_scale=0.0,
                                              options=controls(device.domains[0].ion_node_count))
    names = ("solarlab.device.compile_multiple_ionic_1d", "solarlab.equations.multiple_ionic_1d",
        "solarlab.equations.ionic_1d", "solarlab.solvers.nonlinear.multiple_ionic_1d",
        "solarlab.solvers.nonlinear.newton", "solarlab.solvers.nonlinear.production",
        "solarlab.solvers.nonlinear.steady_1d") + implementation_source_modules("ionic")
    sources = {}
    for name in names:
        filename = importlib.import_module(name).__file__
        if filename is None:
            raise ValueError("steady evidence requires actual bound source files")
        sources[name] = hashlib.sha256(Path(filename).read_bytes()).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    raw = {"schema": "solarlab.multiple_ionic_steady.v1", "result": result,
           "source_sha256": sources, "qualification": False}
    output.write_text(json.dumps(_encode(raw), indent=2, allow_nan=False) + "\n")
    print(json.dumps({"accepted": result.accepted, "termination": result.termination,
        "compiled_identity": result.compiled_identity, "original_prepared_identity": result.original_prepared_identity,
        "shape": [device.node_count, *[domain.ion_node_count for domain in device.domains], device.state_size],
        "wall_s": result.wall_s, "iterations": None if result.newton is None else result.newton.iteration_count,
        "evaluation_counts": result.evaluation_counts, "checks": _encode(result.checks),
        "result_path": str(output), "result_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "error": result.error, "qualification": False}, indent=2, allow_nan=False), flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mixed-immobile", action="store_true")
    args = parser.parse_args()
    result = run_case(args.output, mixed_immobile=args.mixed_immobile)
    return 0 if result.accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
