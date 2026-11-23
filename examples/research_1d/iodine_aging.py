"""Bitton & Tessler (2023) 1D iodine aging workflow: stress, recovery, and J-V probes.

Demonstrates coupled 1D ``(phi, n, p, I^-, I^0, I_2)`` reaction-diffusion-aging
comparing:
1. Closed-boundary reversible light-induced degradation & dark self-healing
   (``mol_escape_velocity_m_s = 0``);
2. Open-boundary irreversible volatile ``I_2`` outgassing degradation
   (``mol_escape_velocity_m_s > 0``);
with non-destructive frozen-chemistry J-V probes and checksum-verified bundle
readback. Every output explicitly retains ``scientific_qualification = False``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from solarlab.config.resolve_device import resolve_device
from solarlab.device.compile_iodine_1d import (
    bitton_tessler_2023_table_s1_chemistry,
    compile_iodine_device_1d,
)
from solarlab.device.defaults import DefaultCatalog
from solarlab.device.inputs import DeviceInput
from solarlab.device.resolved import PreparedDevice
from solarlab.experiments.iodine_aging import (
    IodineAgingProtocol,
    IodineProbeSpec,
    StressRecoveryStage,
    load_iodine_aging_bundle,
    run_iodine_aging_protocol,
    save_iodine_aging_bundle,
)
from solarlab.materials.resources import ResourceLibrary
from solarlab.materials.source import SourceDocument

REPO_ROOT = Path(__file__).resolve().parents[2]


def prepare_bitton_tessler_stack() -> PreparedDevice:
    """Resolve the 3-layer HBL / Perovskite / EBL stack from SI Table S1."""
    base_path = REPO_ROOT / "examples" / "research_1d" / "device.json"
    source = SourceDocument(str(base_path), base_path.read_bytes())
    doc = json.loads(source.content)
    catalog = {
        **doc["default_catalog"],
        "evidence": [[source.id, source.sha256]],
    }
    q_c = 1.602176634e-19
    kb_j_k = 1.380649e-23
    vt = (kb_j_k * 300.0) / q_c
    eps0 = 8.8541878128e-12
    mu_bl = 1.0e-3 * 1.0e-4
    b_langevin = q_c * (2.0 * mu_bl) / (eps0 * 3.0)
    ni_bl = float(np.exp(0.5 * (np.log(1.0e27) + np.log(1.0e27) - 2.5 / vt)))
    ni_pvk = float(np.exp(0.5 * (np.log(7.0e24) + np.log(2.5e24) - 1.6 / vt)))

    device_dict = {
        "schema_version": "solarlab.device-preparation.v1",
        "id": "bitton_tessler_2023_stack",
        "name": "Bitton & Tessler 2023 HBL/PVK/EBL stack",
        "source_format": "canonical",
        "settings": {
            "built_in_potential_mode": "semiconductor_work_function",
            "dos_band_potentials": True,
            "T": 300.0,
            "Phi": 2.5e21,
        },
        "layers": [
            {
                "id": "hbl",
                "name": "HBL",
                "role": "ETL",
                "thickness": 50.0e-9,
                "parameters": {
                    "chi": 4.3,
                    "Eg": 2.5,
                    "Nc300": 1.0e27,
                    "Nv300": 1.0e27,
                    "eps_r": 3.0,
                    "N_D": 2.0e22,
                    "N_A": 0.0,
                    "ni": ni_bl,
                    "n1": ni_bl,
                    "p1": ni_bl,
                    "tau_n": 1.0,
                    "tau_p": 1.0,
                    "B_rad": b_langevin,
                    "C_n": 0.0,
                    "C_p": 0.0,
                    "mu_n": mu_bl,
                    "mu_p": mu_bl,
                    "alpha": 0.0,
                },
            },
            {
                "id": "absorber",
                "name": "Perovskite",
                "role": "absorber",
                "thickness": 500.0e-9,
                "parameters": {
                    "chi": 4.3,
                    "Eg": 1.6,
                    "Nc300": 7.0e24,
                    "Nv300": 2.5e24,
                    "eps_r": 50.0,
                    "N_D": 0.0,
                    "N_A": 0.0,
                    "ni": ni_pvk,
                    "n1": ni_pvk,
                    "p1": ni_pvk,
                    "tau_n": 1.0e-5,
                    "tau_p": 1.0e-5,
                    "B_rad": 1.0e-10 * 1.0e-6,
                    "C_n": 0.0,
                    "C_p": 0.0,
                    "mu_n": 2.0e-4,
                    "mu_p": 2.0e-4,
                    "alpha": 1.0e7,
                },
            },
            {
                "id": "ebl",
                "name": "EBL",
                "role": "HTL",
                "thickness": 50.0e-9,
                "parameters": {
                    "chi": 3.4,
                    "Eg": 2.5,
                    "Nc300": 1.0e27,
                    "Nv300": 1.0e27,
                    "eps_r": 3.0,
                    "N_D": 0.0,
                    "N_A": 2.0e22,
                    "ni": ni_bl,
                    "n1": ni_bl,
                    "p1": ni_bl,
                    "tau_n": 1.0,
                    "tau_p": 1.0,
                    "B_rad": b_langevin,
                    "C_n": 0.0,
                    "C_p": 0.0,
                    "mu_n": mu_bl,
                    "mu_p": mu_bl,
                    "alpha": 0.0,
                },
            },
        ],
    }
    return resolve_device(
        DeviceInput.model_validate(device_dict),
        DefaultCatalog.from_mapping(catalog),
        ResourceLibrary(),
        sources=(source,),
    )


def run_comparison(output_dir: Path) -> dict[str, object]:
    """Run closed self-healing vs open I_2 outgassing aging comparison."""
    if output_dir.exists():
        raise FileExistsError(f"Choose a fresh output directory: {output_dir}")

    prepared = prepare_bitton_tessler_stack()
    protocol = IodineAgingProtocol(
        stages=(
            StressRecoveryStage(
                name="light_stress_0p9V",
                duration_s=100.0,
                voltage_V=0.9,
                generation_scale=1.0,
                max_step_s=25.0,
            ),
            StressRecoveryStage(
                name="dark_recovery_0V",
                duration_s=300.0,
                voltage_V=0.0,
                generation_scale=0.0,
                max_step_s=50.0,
            ),
        ),
        checkpoint_times_s=(0.0, 100.0, 400.0),
        probe=IodineProbeSpec(
            mode="frozen_chemistry_jv",
            voltages_V=tuple(float(v) for v in np.linspace(0.0, 1.32, 16)),
            generation_scale=1.0,
        ),
    )

    closed_chem = bitton_tessler_2023_table_s1_chemistry(
        domain_mode="absorber_blocking",
        rate_mode="smoluchowski_capture_radius",
        perovskite_ka_cm3_s=5.0e-11,
        right_escape_m_s=0.0,
    )
    open_chem = bitton_tessler_2023_table_s1_chemistry(
        domain_mode="absorber_blocking",
        rate_mode="smoluchowski_capture_radius",
        perovskite_ka_cm3_s=5.0e-11,
        right_escape_m_s=5.0e-2,
    )
    closed_dev = compile_iodine_device_1d(
        prepared, closed_chem, intervals_per_layer=(6, 20, 6)
    )
    open_dev = compile_iodine_device_1d(
        prepared, open_chem, intervals_per_layer=(6, 20, 6)
    )

    closed_res = run_iodine_aging_protocol(closed_dev, protocol)
    open_res = run_iodine_aging_protocol(open_dev, protocol)

    closed_manifest = save_iodine_aging_bundle(output_dir / "closed", closed_res)
    open_manifest = save_iodine_aging_bundle(output_dir / "open", open_res)

    # Verify checksum-authenticated readback
    _ = load_iodine_aging_bundle(output_dir / "closed")
    _ = load_iodine_aging_bundle(output_dir / "open")

    summary: dict[str, object] = {
        "scientific_qualification": False,
        "protocol_sha256": protocol.sha256,
        "closed": {
            "device_sha256": closed_dev.identity,
            "max_relative_conservation_error": closed_res.max_relative_conservation_error,
            "final_in_device_iodine_m2": closed_res.checkpoints[-1].in_device_iodine_m2,
            "final_escaped_iodine_m2": (
                closed_res.checkpoints[-1].cumulative_escaped_iodine_atoms_m2
            ),
            "arrays_npz_sha256": closed_manifest["arrays_npz_sha256"],
        },
        "open": {
            "device_sha256": open_dev.identity,
            "max_relative_conservation_error": open_res.max_relative_conservation_error,
            "final_in_device_iodine_m2": open_res.checkpoints[-1].in_device_iodine_m2,
            "final_escaped_iodine_m2": (
                open_res.checkpoints[-1].cumulative_escaped_iodine_atoms_m2
            ),
            "arrays_npz_sha256": open_manifest["arrays_npz_sha256"],
        },
    }
    (output_dir / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Fresh directory to write closed/ and open/ iodine aging bundles.",
    )
    args = parser.parse_args()
    summary = run_comparison(args.output)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
