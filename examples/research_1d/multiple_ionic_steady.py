"""One positive-D two-domain dark cold steady witness, preserving first failure."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path

from examples.research_1d.ionic_steady import controls
from examples.research_1d.multiple_ionic_domains import DOMAINS, INTERVALS, prepare_multiple_ionic_device
from solarlab.device.compile_multiple_ionic_1d import compile_multiple_ionic_device_1d
from solarlab.experiments.transient_results import _encode
from solarlab.solvers.nonlinear.multiple_ionic_1d import MultipleIonicSteadyResult, solve_multiple_ionic_steady_state


def run_case(output: Path) -> MultipleIonicSteadyResult:
    if output.exists():
        raise FileExistsError("preserve the existing attempt; no silent retry")
    device = compile_multiple_ionic_device_1d(prepare_multiple_ionic_device(),
        intervals_per_layer=INTERVALS, ion_domains=DOMAINS)
    result = solve_multiple_ionic_steady_state(device, voltage_V=0.002, generation_scale=0.0,
                                              options=controls(device.domains[0].ion_node_count))
    names = ("solarlab.device.compile_multiple_ionic_1d", "solarlab.equations.multiple_ionic_1d",
        "solarlab.equations.ionic_1d", "solarlab.solvers.nonlinear.multiple_ionic_1d",
        "solarlab.solvers.nonlinear.newton", "solarlab.solvers.nonlinear.production",
        "solarlab.solvers.nonlinear.steady_1d")
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
    result = run_case(parser.parse_args().output)
    return 0 if result.accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
