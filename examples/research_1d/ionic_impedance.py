"""Ionic AC from an existing accepted steady artifact; no steady solve here.

For a new shared-site input, run the existing example once:
    python -m examples.research_1d.shared_site_ionic --output-directory /tmp/shared-input
Hash its Steady.json with sha256sum or shasum -a 256, then pass the actual hash:
    python -m examples.research_1d.ionic_impedance solve --kind shared \
        --initial-result /tmp/shared-input/Steady.json --initial-sha256 HASH \
        --output /tmp/shared-ac.json
Use --ion-mode frozen for an explicit frozen response about the same ionic
charge profile, or --voltage-phasor 2e-6+1e-6j for a linear normalization.
The read subcommand requires only the result and its hash and never solves.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from examples.research_1d.ionic_multilayer import prepare_confined_ion_device, prepare_confined_dual_ion_device
from examples.research_1d.ionic_steady import controls as steady_controls
from examples.research_1d.shared_site_ionic_fixture import prepare_confined_shared_site_ion_device
from solarlab.device.compile_ionic_1d import compile_ionic_device_1d
from solarlab.equations.ionic_1d import CompiledIonicDevice1D
from solarlab.experiments.ionic_impedance import (
    IonicImpedanceOptions, IonicImpedanceResult, read_ionic_impedance_result,
    solve_ionic_impedance, write_ionic_impedance_result,
)
from solarlab.experiments.ionic_transient import IonicTransientInitial, load_ionic_steady_initial


def compile_case(kind: str) -> CompiledIonicDevice1D:
    factories = {"shared": prepare_confined_shared_site_ion_device,
                 "independent": prepare_confined_dual_ion_device, "positive": prepare_confined_ion_device}
    if kind not in factories:
        raise ValueError("kind must be shared, independent or positive")
    return compile_ionic_device_1d(factories[kind](), intervals_per_layer=(12, 12, 12),
        ion_domain_layer="absorber", ion_domain_endpoints=("blocking", "blocking"))


def load_case(kind: str, path: str | Path, sha256: str) -> tuple[CompiledIonicDevice1D, IonicTransientInitial]:
    compiled = compile_case(kind)
    initial = load_ionic_steady_initial(compiled, path, sha256, controls=steady_controls(compiled.ion_node_count))
    return compiled, initial


def load_manifest_case(manifest_path: str | Path, kind: str) -> tuple[CompiledIonicDevice1D, IonicTransientInitial]:
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest["schema"] != "solarlab.ionic_ac_inputs.v1":
        raise ValueError("unsupported ionic AC initial manifest")
    config = Path(manifest["configuration_path"])
    actual_config = Path(__file__).with_name("device.json").resolve()
    if (config.resolve() != actual_config
            or hashlib.sha256(config.read_bytes()).hexdigest() != manifest["configuration_sha256"]):
        raise ValueError("configuration identity/path differs from the pinned steady source")
    item = manifest["cases"][kind]
    compiled, initial = load_case(kind, item["steady_path"], item["steady_sha256"])
    if compiled.identity != item["compiled_identity"]:
        raise ValueError("compiled identity differs from the saved steady manifest")
    return compiled, initial


def summary(record: dict[str, Any]) -> dict[str, Any]:
    """Summarize serialized responses without calling a model or solver."""
    result = record["result"]
    points = []
    for attempt in result["attempts"]:
        observed = attempt["observation"]
        entry = {"frequency_Hz": attempt["frequency_Hz"], "accepted": attempt["accepted"],
                 "termination": attempt["termination"], "error": attempt["error"],
                 "checks": attempt["checks"]}
        if observed is not None:
            admittance = complex(observed["admittance_density_S_m2"])
            entry["admittance_density_S_m2"] = {"real": admittance.real, "imag": admittance.imag}
            impedance = None if admittance == 0.0 else 1.0 / admittance
            entry["impedance_area_ohm_m2"] = (None if impedance is None
                else {"real": impedance.real, "imag": impedance.imag})
            entry["Maxwell_spread_A_m2_V"] = observed.get("consumer_Maxwell_spread_A_m2_V")
            entry["inventory_relative_per_V"] = observed.get("consumer_inventory_relative_per_V")
        points.append(entry)
    return {"completed": record["completed"], "ion_mode": result["ion_mode"], "points": points,
            "compiled_identity": result["compiled_identity"], "initial_provenance": result["initial_provenance"],
            "linear_solver_calls": result["linear_solver_calls"], "wall_s": result["wall_s"],
            "physical_certificate_status": result["physical_certificate_status"]}


def run_response(
    compiled: CompiledIonicDevice1D, initial: IonicTransientInitial, output: Path,
    *, frequencies_Hz: tuple[float, ...] = (0.0, 0.01, 1.0, 1000.0),
    ion_mode: str = "mobile", voltage_phasor_V: complex = 1.0,
) -> IonicImpedanceResult:
    if output.exists():
        raise FileExistsError("preserve the existing AC attempt; choose an unused output path")
    result = solve_ionic_impedance(compiled, initial, frequencies_Hz, ion_mode=ion_mode,
        voltage_phasor_V=voltage_phasor_V, options=IonicImpedanceOptions())
    output.parent.mkdir(parents=True, exist_ok=True)
    digest = write_ionic_impedance_result(result, output)
    print(json.dumps({"result_path": str(output.resolve()), "result_sha256": digest,
        **summary(read_ionic_impedance_result(output, digest))}, indent=2, allow_nan=False), flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    solve = commands.add_parser("solve", help="one AC sweep from an existing accepted steady artifact")
    solve.add_argument("--kind", choices=("shared", "independent", "positive"), default="shared")
    solve.add_argument("--initial-result", type=Path, required=True)
    solve.add_argument("--initial-sha256", required=True)
    solve.add_argument("--frequencies", type=float, nargs="+", default=[0.0, 0.01, 1.0, 1000.0])
    solve.add_argument("--ion-mode", choices=("mobile", "frozen"), default="mobile")
    solve.add_argument("--voltage-phasor", type=complex, default=1.0 + 0.0j)
    solve.add_argument("--output", type=Path, required=True)
    read = commands.add_parser("read", help="checksum-checked saved result reading without solving")
    read.add_argument("--result", type=Path, required=True)
    read.add_argument("--sha256", required=True)
    arguments = parser.parse_args()
    if arguments.command == "read":
        print(json.dumps(summary(read_ionic_impedance_result(arguments.result, arguments.sha256)), indent=2, allow_nan=False))
        return 0
    compiled, initial = load_case(arguments.kind, arguments.initial_result, arguments.initial_sha256)
    result = run_response(compiled, initial, arguments.output, frequencies_Hz=tuple(arguments.frequencies),
        ion_mode=arguments.ion_mode, voltage_phasor_V=arguments.voltage_phasor)
    return 0 if result.completed else 2


if __name__ == "__main__":
    raise SystemExit(main())
