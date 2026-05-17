from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.screening.solarscale import (
    generate_solarlab_inputs,
    parse_material_records,
    plan_solarlab_import,
)


def _prop(value, kind="computed", unit=""):
    return {"value": value, "unit": unit, "provenance": {"kind": kind, "source": "test"}}


def _record(
    material_id: str,
    readiness: str,
    *,
    final_score=None,
    ml_score=None,
    with_md: bool = True,
    mobility_kind: str = "computed",
) -> dict:
    properties = {
        "dft_result_available": _prop(True),
        "band_gap_hse_ev": _prop(1.62, unit="eV"),
        "electron_effective_mass_m0": _prop(0.2),
        "hole_effective_mass_m0": _prop(0.35),
        "dielectric_static_avg": _prop(12.0),
        "electron_mobility_cm2_v_s": _prop(3.0, mobility_kind, unit="cm^2/V/s"),
        "hole_mobility_cm2_v_s": _prop(4.0, mobility_kind, unit="cm^2/V/s"),
        "carrier_lifetime_s": _prop(None, "swept", unit="s"),
        "trap_density_m3": _prop(None, "swept", unit="m^-3"),
        "ml_pv_score": _prop(ml_score, "derived", unit="score"),
        "final_fom_score": _prop(final_score, "derived", unit="score"),
        "slme_0p5um": _prop(0.18, "derived"),
    }
    if with_md:
        properties.update(
            {
                "ion_diffusion_coefficient_m2_s": _prop(2e-16, unit="m^2/s"),
                "ion_activation_energy_ev": _prop(0.42, unit="eV"),
                "md_stable": _prop(True),
            }
        )
    return {
        "material_id": material_id,
        "screening": {
            "readiness": readiness,
            "promising_candidate": readiness == "promising",
            "provisional_solarlab_ready": readiness in {"phonon", "promising"},
            "production_solarlab_ready": readiness == "promising",
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
                    _record("mp-ml-only", "promising", final_score=None, ml_score=0.95),
                    _record("mp-best", "promising", final_score=72.0, ml_score=0.20),
                    _record("mp-phonon", "phonon", final_score=12.0, ml_score=0.40, with_md=False),
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


def test_parses_material_records_with_property_provenance(tmp_path: Path):
    records_path = _write_records(tmp_path)
    records = parse_material_records(records_path)

    best = next(record for record in records if record.material_id == "mp-best")
    assert best.properties["band_gap_hse_ev"].value == 1.62
    assert best.properties["band_gap_hse_ev"].provenance.kind == "computed"
    assert best.properties["carrier_lifetime_s"].provenance.kind == "swept"


def test_plan_sorts_by_final_fom_then_ml_score_and_keeps_scores_as_metadata(tmp_path: Path):
    records_path = _write_records(tmp_path)
    plan = plan_solarlab_import(
        records_path,
        template_path="configs/nip_MAPbI3.yaml",
        import_policy="production",
        include_configs=False,
    )

    assert [item["material_id"] for item in plan["selected"]] == ["mp-best", "mp-ml-only"]
    assert plan["selected"][0]["ranking_score_source"] == "final_fom_score"
    assert plan["selected"][1]["ranking_score_source"] == "ml_pv_score"
    assert "absorber.Eg" in plan["selected"][0]["mapped_parameters"]
    assert "final_fom_score imported as ranking metadata only" in plan["selected"][0]["diagnostics"]
    assert "ml_pv_score imported as ranking metadata only" in plan["selected"][0]["diagnostics"]


def test_production_import_requires_promising_and_maps_dft_md_absorber_inputs(tmp_path: Path):
    records_path = _write_records(tmp_path)
    out_dir = tmp_path / "production"
    manifest = generate_solarlab_inputs(
        records_path,
        template_path="configs/nip_MAPbI3.yaml",
        out_dir=out_dir,
        limit=1,
        import_policy="production",
    )

    assert [item["material_id"] for item in manifest["generated"]] == ["mp-best"]
    assert manifest["import_policy"] == "production"
    assert manifest["allowed_readiness"] == ["promising"]
    assert {item["material_id"] for item in manifest["skipped"]} == {
        "mp-ml-only",
        "mp-phonon",
        "mp-partial",
    }

    config_path = Path(manifest["generated"][0]["config_path"])
    stack = load_device_from_yaml(str(config_path))
    absorber = [layer for layer in stack.layers if layer.role == "absorber"][0]
    htl = [layer for layer in stack.layers if layer.role == "HTL"][0]

    assert absorber.name == "SolarScale_mp-best"
    assert absorber.params.Eg == pytest.approx(1.62)
    assert absorber.params.eps_r == 12.0
    assert absorber.params.mu_n == pytest.approx(3.0e-4)
    assert absorber.params.mu_p == pytest.approx(4.0e-4)
    assert absorber.params.D_ion == pytest.approx(2e-16)
    assert absorber.params.E_a_ion == pytest.approx(0.42)
    assert absorber.thickness == pytest.approx(300e-9)
    assert htl.params.eps_r == pytest.approx(3.0)
    assert stack.interfaces == ((0.0, 0.0), (0.0, 0.0))
    assert (out_dir / "manifest.json").exists()


def test_exploratory_import_accepts_phonon_records_and_uses_template_ion_defaults(tmp_path: Path):
    records_path = _write_records(tmp_path)
    out_dir = tmp_path / "exploratory"
    manifest = generate_solarlab_inputs(
        records_path,
        template_path="configs/nip_MAPbI3.yaml",
        out_dir=out_dir,
        import_policy="exploratory",
    )

    assert [item["material_id"] for item in manifest["generated"]] == [
        "mp-best",
        "mp-phonon",
        "mp-ml-only",
    ]
    assert manifest["allowed_readiness"] == ["phonon", "promising"]
    phonon_config = out_dir / "mp-phonon.yaml"
    stack = load_device_from_yaml(str(phonon_config))
    absorber = [layer for layer in stack.layers if layer.role == "absorber"][0]
    assert absorber.params.D_ion == pytest.approx(1e-16)
    assert absorber.params.E_a_ion == pytest.approx(0.58)
    phonon = next(item for item in manifest["generated"] if item["material_id"] == "mp-phonon")
    assert "ion_diffusion_coefficient_m2_s" in phonon["missing_optional"]


def test_swept_mobility_is_not_mapped_as_fixed_physical_input(tmp_path: Path):
    records_path = tmp_path / "records.json"
    records_path.write_text(
        json.dumps(
            {
                "schema": "solarscale.material_records",
                "schema_version": "0.2",
                "records": [
                    _record(
                        "mp-swept-mobility",
                        "promising",
                        final_score=5.0,
                        ml_score=0.4,
                        mobility_kind="swept",
                    )
                ],
            }
        ),
        encoding="utf-8",
    )

    manifest = generate_solarlab_inputs(
        records_path,
        template_path="configs/nip_MAPbI3.yaml",
        out_dir=tmp_path / "out",
        import_policy="production",
    )

    generated = manifest["generated"][0]
    assert "electron_mobility_cm2_v_s" in generated["missing_optional"]
    assert "electron_mobility_cm2_v_s is swept" in " ".join(generated["diagnostics"])
    stack = load_device_from_yaml(generated["config_path"])
    absorber = [layer for layer in stack.layers if layer.role == "absorber"][0]
    assert absorber.params.mu_n == pytest.approx(2e-4)
    assert absorber.params.mu_p == pytest.approx(2e-4)


def test_assumed_property_is_not_accepted_as_fixed_dft_md_input(tmp_path: Path):
    record = _record("mp-assumed-eg", "promising", final_score=5.0, ml_score=0.4)
    record["properties"]["band_gap_hse_ev"] = _prop(1.7, "assumed", unit="eV")
    records_path = tmp_path / "records.json"
    records_path.write_text(
        json.dumps(
            {
                "schema": "solarscale.material_records",
                "schema_version": "0.2",
                "records": [record],
            }
        ),
        encoding="utf-8",
    )

    plan = plan_solarlab_import(
        records_path,
        template_path="configs/nip_MAPbI3.yaml",
        import_policy="production",
        include_configs=False,
    )

    assert plan["selected"] == []
    skipped = plan["skipped"][0]
    assert skipped["material_id"] == "mp-assumed-eg"
    assert skipped["missing_required"] == ["band_gap_hse_ev"]
    assert "provenance kind 'assumed' is not accepted" in " ".join(skipped["diagnostics"])


def test_cli_dry_run_writes_screening_plan(tmp_path: Path):
    records_path = _write_records(tmp_path)
    out_dir = tmp_path / "cli"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_material_screening.py",
            "--records",
            str(records_path),
            "--policy",
            "exploratory",
            "--base-config",
            "configs/nip_MAPbI3.yaml",
            "--top-n",
            "2",
            "--out-dir",
            str(out_dir),
            "--dry-run",
        ],
        check=True,
        cwd=Path(__file__).parents[3],
        text=True,
        capture_output=True,
    )

    plan_path = out_dir / "screening_plan.json"
    assert "Selected 2 candidates" in result.stdout
    assert plan_path.exists()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert [item["material_id"] for item in plan["selected"]] == ["mp-best", "mp-phonon"]
    assert not (out_dir / "manifest.json").exists()
