"""One authorized warm initial for the positive/right-D0 BE parity case.

The already accepted positive-case coordinates supply only a numerical
guess. Right-D0 coordinates are set to the declared zero profile before one
current-source steady solve under the original controls. No failed state,
different shared-site fixture, gate change or retry is used.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import runpy
from typing import Any

import numpy as np

from solarlab.experiments.transient_results import _decode, _encode
from solarlab.solvers.nonlinear.multiple_ionic_1d import solve_multiple_ionic_steady_state
from support.assembly_references import REFERENCE_FACTORIES

POSITIVE_SHA256 = "880b6437ac39e49c8e39d0d3fd557374343dcca3f053cd257db121d6b5b7f856"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("preserve the one warm preparation attempt")
    args.output.mkdir(parents=True)
    raw = args.positive.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == POSITIVE_SHA256
    saved = _decode(json.loads(raw))
    assert saved["factory"] == "multiple_be:positive"
    before = saved["steady_initial"]
    assert before["accepted"] is True and before["newton"]["accepted"] is True
    assert all(item["passed"] for item in before["checks"])
    positive = REFERENCE_FACTORIES["multiple_be:positive"]().device
    device = REFERENCE_FACTORIES["multiple_be:immobile"]().device
    assert before["compiled_identity"] == positive.identity
    assert device.state_size == positive.state_size
    assert np.array_equal(device.electronic_component.nodes_m, positive.electronic_component.nodes_m)
    u = np.array(before["state"]["u"], copy=True)
    domain = next(item for item in device.domains if item.domain_layer_id == "absorber_right")
    assert domain.ion_species_names == ("positive",)
    assert np.all(domain.positive_ion_diffusivity_faces_m2_s == 0.0)
    profile = device.ion_species_slices["absorber_right"]["positive"]
    u[profile] = 0.0
    guess = device.state(u, voltage_V=.002)
    helpers = runpy.run_path("tests/unit/assembly/test_assembly_ionic_drivers.py")
    controls = helpers["steady_controls"](device.domains[0].ion_node_count)
    receipt: dict[str, Any] = {"schema": "solarlab.ionic_assembly.immobile_warm_preparation.v1",
        "coordinator_approval": "msg_b678da9af1a1: one warm preparation, then one BE pair only if accepted; no retries",
        "source_positive_sha256": POSITIVE_SHA256, "source_positive_compiled_identity": positive.identity,
        "current_immobile_compiled_identity": device.identity, "source_role": "numerical guess only",
        "historical_acceptance_transferred": False, "right_D0_profile_coordinate": 0.0,
        "guess": guess, "controls": asdict(controls), "cold_preparation_calls": 0,
        "warm_preparation_calls": 0, "qualification": False}
    with (args.output / "Started.json").open("x") as stream:
        json.dump(_encode(receipt), stream, indent=2, allow_nan=False)
        stream.write("\n")
    result = solve_multiple_ionic_steady_state(device, voltage_V=.002, generation_scale=0.0,
        options=controls, initial_state=guess)
    receipt.update(result=result, warm_preparation_calls=1)
    with (args.output / "Preparation.json").open("x") as stream:
        json.dump(_encode(receipt), stream, indent=2, allow_nan=False)
        stream.write("\n")
    summary = {"accepted": result.accepted, "termination": result.termination,
        "every_original_gate_passed": bool(result.checks and all(item.passed for item in result.checks)),
        "newton_iterations": None if result.newton is None else len(result.newton.iterations),
        "wall_s": result.wall_s, "warm_preparation_calls": 1, "cold_preparation_calls": 0,
        "preparation_sha256": hashlib.sha256((args.output / "Preparation.json").read_bytes()).hexdigest(),
        "error": result.error, "qualification": False}
    if result.accepted:
        path, digest = helpers["evidence"](device, result, args.output)
        summary.update(accepted_initial_path=str(path.resolve()), accepted_initial_sha256=digest)
    with (args.output / "Summary.json").open("x") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(summary, indent=2))
    return 0 if result.accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
