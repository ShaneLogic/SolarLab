from __future__ import annotations

import json
from pathlib import Path

import pytest

from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.screening.solarscale import generate_solarlab_inputs


def _prop(value, kind="computed"):
    return {"value": value, "unit": "", "provenance": {"kind": kind, "source": "test"}}


def test_generate_solarlab_inputs_filters_and_loads_generated_config(tmp_path: Path):
    records_path = tmp_path / "material_records.json"
    records_path.write_text(
        json.dumps(
            {
                "schema": "solarscale.material_records",
                "schema_version": "0.1",
                "records": [
                    {
                        "material_id": "mp-ready",
                        "properties": {
                            "dft_result_available": _prop(True),
                            "band_gap_hse_ev": _prop(1.62),
                            "electron_effective_mass_m0": _prop(0.2),
                            "hole_effective_mass_m0": _prop(0.35),
                            "dielectric_static_avg": _prop(12.0),
                            "electron_mobility_cm2_v_s": _prop(3.0),
                            "hole_mobility_cm2_v_s": _prop(4.0),
                            "carrier_lifetime_s": _prop(None, "swept"),
                        },
                    },
                    {
                        "material_id": "mp-partial",
                        "properties": {
                            "dft_result_available": _prop(False),
                            "band_gap_hse_ev": _prop(None, "missing"),
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    out_dir = tmp_path / "generated"
    manifest = generate_solarlab_inputs(
        records_path,
        template_path="configs/nip_MAPbI3.yaml",
        out_dir=out_dir,
    )

    assert [item["material_id"] for item in manifest["generated"]] == ["mp-ready"]
    assert manifest["skipped"] == [
        {"material_id": "mp-partial", "reason": "dft_result_available is not true"}
    ]

    config_path = Path(manifest["generated"][0]["config_path"])
    stack = load_device_from_yaml(str(config_path))
    absorber = [layer for layer in stack.layers if layer.role == "absorber"][0]

    assert absorber.name == "SolarScale_mp-ready"
    assert absorber.params.Eg == 0.0
    assert manifest["generated"][0]["dft_properties"]["band_gap_hse_ev"] == 1.62
    assert absorber.params.eps_r == 12.0
    assert absorber.params.mu_n == pytest.approx(3.0e-4)
    assert absorber.params.mu_p == pytest.approx(4.0e-4)
    assert absorber.thickness == 300e-9
    assert stack.interfaces == ((0.0, 0.0), (0.0, 0.0))
    assert (out_dir / "manifest.json").exists()
