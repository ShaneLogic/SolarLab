"""Fresh direct illuminated initialization from a pinned accepted dark state.

    python -m examples.research_1d.ionic_initialization solve \
      --policy direct_density_seed_v1 --initial-result DARK.json \
      --initial-sha256 SHA --voltage 0.002 --generation-scale 1 --output INIT.json

The direct policy performs one positive carrier-density preparation and one
original full coupled ionic solve. Explicit --density-seed-result/sha256 flags
reuse a recorded preparation instead. No failing auxiliary solve is required.
Use an outer process supervisor for a hard deadline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from examples.research_1d.ionic_steady import controls
from examples.research_1d.shared_site_ionic_fixture import prepare_confined_shared_site_ion_device
from examples.research_1d.transient import json_value
from solarlab.device.compile_ionic_1d import compile_ionic_device_1d
from solarlab.equations.ionic_1d import CompiledIonicDevice1D
from solarlab.experiments.ionic_initialization import (
    POLICY, DIRECT_POLICY, IonicInitializationOptions, initialize_ionic_steady_state, load_ionic_density_seed,
    read_ionic_initialization, write_ionic_initialization,
)
from solarlab.experiments.ionic_transient import load_ionic_steady_initial


def compile_shared_case() -> CompiledIonicDevice1D:
    return compile_ionic_device_1d(prepare_confined_shared_site_ion_device(),
        intervals_per_layer=(12, 12, 12), ion_domain_layer="absorber",
        ion_domain_endpoints=("blocking", "blocking"))


def summary(record: dict[str, Any]) -> dict[str, Any]:
    result = record["result"]
    coupled = result["coupled_result"]
    initial = result["initial"]["state"]
    item = {"accepted": result["accepted"], "termination": result["termination"],
        "voltage_V": result["voltage_V"], "generation_scale": result["generation_scale"],
        "policy": result["solver_controls"]["policy"], "compiled_identity": result["compiled_identity"],
        "initial_sha256": result["initial"]["source_sha256"], "component_identity": result["component_identity"],
        "evaluation_counts": result["evaluation_counts"], "wall_s": result["wall_s"], "error": result["error"],
        "stages": [{"name": x["name"], "accepted": x["accepted"]} for x in result["stages"]],
        "qualification": result["qualification"], "physical_certificate_status": result["physical_certificate_status"]}
    if result.get("density_seed_provenance") is not None:
        item["density_seed_mode"] = result["density_seed_provenance"].get("mode", "saved")
    if result["coupled_initial_state"] is not None:
        seeded = result["coupled_initial_state"]
        item["seed_ion_profiles_unchanged"] = all(np.array_equal(
            initial[name + "_ion_density_m3"], seeded[name + "_ion_density_m3"]) for name in ("positive", "negative"))
        item["seed_ion_inventories_unchanged"] = (
            initial["domain_inventory_m2"] == seeded["domain_inventory_m2"]
            and initial["negative_domain_inventory_m2"] == seeded["negative_domain_inventory_m2"])
    if coupled is not None:
        item["checks"] = coupled["checks"]
        item["newton_iterations"] = None if coupled["newton"] is None else len(coupled["newton"]["iterations"])
        if coupled["diagnostics"] is not None:
            diagnostics = coupled["diagnostics"]
            item["all_face_current_A_m2"] = diagnostics["total_current_density_A_m2"].tolist()
            item["inventories"] = {name: {key: values[key] for key in
                ("inventory_m2", "inventory_relative_error", "ion_balance_l1_m2_s", "ion_flux_max_m2_s",
                 "original_dropped_balance_m2_s")} for name, values in diagnostics.get("ion_species", {}).items()}
    return item


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    solve = commands.add_parser("solve")
    solve.add_argument("--policy", choices=(POLICY, DIRECT_POLICY), required=True)
    solve.add_argument("--density-seed-result", type=Path)
    solve.add_argument("--density-seed-sha256")
    solve.add_argument("--initial-result", type=Path, required=True)
    solve.add_argument("--initial-sha256", required=True)
    solve.add_argument("--voltage", type=float, required=True)
    solve.add_argument("--generation-scale", type=float, required=True)
    solve.add_argument("--wall-s", type=float, default=25.0)
    solve.add_argument("--output", type=Path, required=True)
    solve.add_argument("--accepted-steady-output", type=Path,
                       help="also export the accepted full solve in the existing initial-artifact format")
    read = commands.add_parser("read")
    read.add_argument("--result", type=Path, required=True)
    read.add_argument("--sha256", required=True)
    args = parser.parse_args()
    if args.command == "read":
        print(json.dumps(summary(read_ionic_initialization(args.result, args.sha256)), indent=2, allow_nan=False))
        return 0
    if args.policy == DIRECT_POLICY:
        if bool(args.density_seed_result) != bool(args.density_seed_sha256):
            parser.error("recorded reuse requires both density-seed-result and density-seed-sha256")
    elif args.density_seed_result or args.density_seed_sha256:
        parser.error("electronic_density_seed_v1 does not accept a recorded density seed")
    if args.output.exists() or (args.accepted_steady_output and args.accepted_steady_output.exists()):
        raise FileExistsError("preserve previous attempts; choose unused output paths")
    device = compile_shared_case()
    declared = controls(device.ion_node_count)
    initial = load_ionic_steady_initial(device, args.initial_result, args.initial_sha256, controls=declared)
    result = initialize_ionic_steady_state(device, initial=initial,
        voltage_V=args.voltage, generation_scale=args.generation_scale,
        options=IonicInitializationOptions(declared, args.policy, args.wall_s),
        density_seed_artifact=(None if args.density_seed_result is None else load_ionic_density_seed(
            args.density_seed_result, args.density_seed_sha256)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    digest = write_ionic_initialization(result, args.output)
    printed = {"result_path": str(args.output.resolve()), "result_sha256": digest,
               **summary(read_ionic_initialization(args.output, digest))}
    if result.accepted and args.accepted_steady_output is not None:
        solution = result.coupled_result
        evidence = {"summary": {"accepted": True, "compiled_identity": result.compiled_identity,
            "voltage_V": result.voltage_V, "generation_scale": result.generation_scale,
            "initialization_policy": args.policy, "initialization_result_path": str(args.output.resolve()),
            "initialization_result_sha256": digest, "original_initial_sha256": initial.source_sha256,
            "qualification": False}, "result": solution}
        raw = (json.dumps(json_value(evidence), indent=2, allow_nan=False) + "\n").encode()
        args.accepted_steady_output.parent.mkdir(parents=True, exist_ok=True)
        with args.accepted_steady_output.open("xb") as stream:
            stream.write(raw)
        printed["accepted_steady_path"] = str(args.accepted_steady_output.resolve())
        printed["accepted_steady_sha256"] = hashlib.sha256(raw).hexdigest()
    print(json.dumps(json_value(printed), indent=2, allow_nan=False), flush=True)
    return 0 if result.accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
