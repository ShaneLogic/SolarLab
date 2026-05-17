from __future__ import annotations

import json
from pathlib import Path

import pytest

from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.screening.solarscale import generate_solarlab_inputs


def _prop(value, kind="computed"):
    return {"value": value, "unit": "", "provenance": {"kind": kind, "source": "test"}}


def _record(material_id: str, readiness: str, *, with_md: bool) -> dict:
    properties = {
        "dft_result_available": _prop(True),
        "band_gap_hse_ev": _prop(1.62),
        "electron_effective_mass_m0": _prop(0.2),
        "hole_effective_mass_m0": _prop(0.35),
        "dielectric_static_avg": _prop(12.0),
        "electron_mobility_cm2_v_s": _prop(3.0),
        "hole_mobility_cm2_v_s": _prop(4.0),
        "carrier_lifetime_s": _prop(None, "swept"),
    }
    if with_md:
        properties.update(
            {
                "ion_diffusion_coefficient_m2_s": _prop(2e-16),
                "ion_activation_energy_ev": _prop(0.42),
                "md_stable": _prop(True),
            }
        )
    return {
        "material_id": material_id,
        "screening": {
            "readiness": readiness,
            "promising_candidate": readiness == "promising",
        },
        "properties": properties,
    }


def _write_records(tmp_path: Path) -> Path:
    records_path = tmp_path / "material_records.json"
    records_path.write_text(
        json.dumps(
            {
                "schema": "solarscale.material_records",
                "schema_version": "0.2",
                "records": [
                    _record("mp-promising", "promising", with_md=True),
                    _record("mp-phonon", "phonon", with_md=False),
                    {
                        "material_id": "mp-partial",
                        "screening": {"readiness": "incomplete"},
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
    return records_path


def test_production_import_requires_promising_and_maps_md_ion_inputs(tmp_path: Path):
    records_path = _write_records(tmp_path)
    out_dir = tmp_path / "production"
    manifest = generate_solarlab_inputs(
        records_path,
        template_path="configs/nip_MAPbI3.yaml",
        out_dir=out_dir,
        import_policy="production",
    )

    assert [item["material_id"] for item in manifest["generated"]] == ["mp-promising"]
    assert manifest["import_policy"] == "production"
    assert manifest["allowed_readiness"] == ["promising"]
    assert {item["material_id"] for item in manifest["skipped"]} == {"mp-phonon", "mp-partial"}

    config_path = Path(manifest["generated"][0]["config_path"])
    stack = load_device_from_yaml(str(config_path))
    absorber = [layer for layer in stack.layers if layer.role == "absorber"][0]

    assert absorber.name == "SolarScale_mp-promising"
    assert absorber.params.Eg == 0.0
    assert manifest["generated"][0]["dft_properties"]["band_gap_hse_ev"] == 1.62
    assert manifest["generated"][0]["md_properties"]["ion_diffusion_coefficient_m2_s"] == 2e-16
    assert absorber.params.eps_r == 12.0
    assert absorber.params.mu_n == pytest.approx(3.0e-4)
    assert absorber.params.mu_p == pytest.approx(4.0e-4)
    assert absorber.params.D_ion == pytest.approx(2e-16)
    assert absorber.params.E_a_ion == pytest.approx(0.42)
    assert absorber.thickness == 300e-9
    assert stack.interfaces == ((0.0, 0.0), (0.0, 0.0))
    assert stack.compute_V_bi() == stack.V_bi
    assert (out_dir / "manifest.json").exists()


def test_exploratory_import_accepts_phonon_ready_records(tmp_path: Path):
    records_path = _write_records(tmp_path)
    out_dir = tmp_path / "exploratory"
    manifest = generate_solarlab_inputs(
        records_path,
        template_path="configs/nip_MAPbI3.yaml",
        out_dir=out_dir,
        import_policy="exploratory",
    )

    assert [item["material_id"] for item in manifest["generated"]] == ["mp-promising", "mp-phonon"]
    assert manifest["allowed_readiness"] == ["phonon", "promising"]
    phonon_config = out_dir / "mp-phonon.yaml"
    stack = load_device_from_yaml(str(phonon_config))
    absorber = [layer for layer in stack.layers if layer.role == "absorber"][0]
    assert absorber.params.D_ion == pytest.approx(1e-16)
    assert absorber.params.E_a_ion == pytest.approx(0.58)
