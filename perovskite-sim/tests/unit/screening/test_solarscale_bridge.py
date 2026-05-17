from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.screening.solarscale import (
    generate_solarlab_inputs,
    parse_material_records,
    plan_solarlab_import,
    run_smoke_device_results,
    write_device_results,
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
            "gates": {
                "electronic": {"passed": True, "band_gap_window_ev": [1.0, 2.0]},
                "phonon": True if readiness in {"phonon", "promising"} else None,
                "md_ion": {"status": "pass" if with_md else "missing"},
                "sustainability": "pass",
            },
            "thresholds": {
                "band_gap_hse_ev": [1.0, 2.0],
                "dielectric_static_avg_min": 5.0,
                "ion_activation_energy_ev_min": 0.25,
            },
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
    assert plan["sweep_policy"] == "quick"
    assert plan["sweep_dimensions"]["total_points"] == 1
    assert plan["selected"][0]["screening_evidence"]["gates"]["electronic"]["passed"] is True
    assert plan["selected"][0]["screening_evidence"]["thresholds"]["band_gap_hse_ev"] == [1.0, 2.0]
    assert "absorber.Eg" not in plan["selected"][0]["mapped_parameters"]
    assert plan["selected"][0]["material_metadata"]["band_gap_hse_ev"] == pytest.approx(1.62)
    assert "band_gap_hse_ev kept as metadata" in " ".join(plan["selected"][0]["diagnostics"])
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
    assert manifest["sweep_policy"] == "quick"
    assert manifest["sweep_dimensions"]["total_points"] == 1
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
    assert absorber.params.Eg == pytest.approx(0.0)
    assert absorber.params.eps_r == 12.0
    assert absorber.params.mu_n == pytest.approx(3.0e-4)
    assert absorber.params.mu_p == pytest.approx(4.0e-4)
    assert absorber.params.D_ion == pytest.approx(2e-16)
    assert absorber.params.E_a_ion == pytest.approx(0.42)
    assert absorber.thickness == pytest.approx(300e-9)
    assert htl.params.eps_r == pytest.approx(3.0)
    assert stack.interfaces == ((1.0, 1.0), (1.0, 1.0))
    assert stack.compute_V_bi() == pytest.approx(stack.V_bi)
    assert manifest["generated"][0]["material_metadata"]["band_gap_hse_ev"] == pytest.approx(1.62)
    assert (out_dir / "manifest.json").exists()


def test_default_generated_yaml_preserves_bandgap_as_metadata_only(tmp_path: Path):
    records_path = _write_records(tmp_path)
    out_dir = tmp_path / "default-bandgap"
    manifest = generate_solarlab_inputs(
        records_path,
        template_path="configs/nip_MAPbI3.yaml",
        out_dir=out_dir,
        limit=1,
        import_policy="production",
    )

    config_path = Path(manifest["generated"][0]["config_path"])
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    absorber = next(layer for layer in config["layers"] if layer["role"] == "absorber")

    assert absorber.get("Eg", 0.0) == pytest.approx(0.0)
    assert config["source"]["activate_bandgap"] is False
    assert config["source"]["schema_version"] == "0.4"
    assert config["source"]["sweep_policy"] == "quick"
    assert config["source"]["sweep_dimensions"]["total_points"] == 1
    assert "absorber.Eg" not in config["source"]["mapped_parameters"]
    assert config["source"]["material_metadata"]["band_gap_hse_ev"] == pytest.approx(1.62)
    assert config["source"]["screening_evidence"]["gates"]["md_ion"]["status"] == "pass"


def test_activate_bandgap_rejects_legacy_template(tmp_path: Path):
    records_path = _write_records(tmp_path)

    with pytest.raises(ValueError, match="fully band-aligned template"):
        generate_solarlab_inputs(
            records_path,
            template_path="configs/nip_MAPbI3.yaml",
            out_dir=tmp_path / "bad-activation",
            limit=1,
            import_policy="production",
            activate_bandgap=True,
        )


def test_activate_bandgap_maps_eg_for_band_aligned_template(tmp_path: Path):
    records_path = _write_records(tmp_path)
    template = yaml.safe_load(Path("configs/nip_MAPbI3.yaml").read_text(encoding="utf-8"))
    for layer in template["layers"]:
        if layer["role"] == "HTL":
            layer["chi"] = 2.2
            layer["Eg"] = 3.0
        elif layer["role"] == "absorber":
            layer["chi"] = 3.9
            layer["Eg"] = 1.55
        elif layer["role"] == "ETL":
            layer["chi"] = 4.0
            layer["Eg"] = 3.2
    template_path = tmp_path / "band_aligned.yaml"
    template_path.write_text(yaml.safe_dump(template, sort_keys=False), encoding="utf-8")

    manifest = generate_solarlab_inputs(
        records_path,
        template_path=template_path,
        out_dir=tmp_path / "activated",
        limit=1,
        import_policy="production",
        activate_bandgap=True,
    )

    config_path = Path(manifest["generated"][0]["config_path"])
    stack = load_device_from_yaml(str(config_path))
    absorber = next(layer for layer in stack.layers if layer.role == "absorber")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert absorber.params.Eg == pytest.approx(1.62)
    assert manifest["activate_bandgap"] is True
    assert manifest["generated"][0]["mapped_parameters"]["absorber.Eg"] == pytest.approx(1.62)
    assert config["source"]["activate_bandgap"] is True
    assert config["source"]["mapped_parameters"]["absorber.Eg"] == pytest.approx(1.62)


def test_solarscale_band_aligned_template_keeps_template_eg_by_default(tmp_path: Path):
    records_path = _write_records(tmp_path)
    manifest = generate_solarlab_inputs(
        records_path,
        template_path="configs/solarscale_nip_band_aligned.yaml",
        out_dir=tmp_path / "solarscale-default",
        limit=1,
        import_policy="production",
    )

    config_path = Path(manifest["generated"][0]["config_path"])
    stack = load_device_from_yaml(str(config_path))
    absorber = next(layer for layer in stack.layers if layer.role == "absorber")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert absorber.params.Eg == pytest.approx(1.6)
    assert config["source"]["activate_bandgap"] is False
    assert "absorber.Eg" not in config["source"]["mapped_parameters"]
    assert config["source"]["material_metadata"]["band_gap_hse_ev"] == pytest.approx(1.62)


def test_activate_bandgap_maps_eg_for_solarscale_band_aligned_template(tmp_path: Path):
    records_path = _write_records(tmp_path)
    manifest = generate_solarlab_inputs(
        records_path,
        template_path="configs/solarscale_nip_band_aligned.yaml",
        out_dir=tmp_path / "solarscale-activated",
        limit=1,
        import_policy="production",
        activate_bandgap=True,
    )

    config_path = Path(manifest["generated"][0]["config_path"])
    stack = load_device_from_yaml(str(config_path))
    absorber = next(layer for layer in stack.layers if layer.role == "absorber")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert absorber.params.Eg == pytest.approx(1.62)
    assert absorber.params.chi == pytest.approx(3.7)
    assert manifest["activate_bandgap"] is True
    assert manifest["generated"][0]["mapped_parameters"]["absorber.Eg"] == pytest.approx(1.62)
    assert manifest["generated"][0]["material_metadata"]["band_gap_hse_ev"] == pytest.approx(1.62)
    assert config["source"]["mapped_parameters"]["absorber.Eg"] == pytest.approx(1.62)


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


def test_plan_preserves_gate_evidence_and_adds_auditable_summary(tmp_path: Path):
    records_path = _write_records(tmp_path)
    plan = plan_solarlab_import(
        records_path,
        template_path="configs/nip_MAPbI3.yaml",
        import_policy="exploratory",
        include_configs=False,
    )

    assert plan["schema_version"] == "0.4"
    assert plan["summary"]["readiness_distribution"] == {
        "incomplete": 1,
        "phonon": 1,
        "promising": 2,
    }
    assert plan["summary"]["gate_summary"]["records_with_gate_data"] == 3
    assert plan["summary"]["gate_summary"]["by_gate"]["electronic"]["pass"] == 3
    assert plan["summary"]["gate_summary"]["by_gate"]["md_ion"]["missing"] == 1
    assert plan["summary"]["skipped_reason_counts"]["screening.readiness='incomplete' not allowed by exploratory policy"] == 1
    assert [item["material_id"] for item in plan["summary"]["top_selected_candidates"][:2]] == [
        "mp-best",
        "mp-phonon",
    ]

    phonon = next(item for item in plan["selected"] if item["material_id"] == "mp-phonon")
    assert phonon["screening_evidence"]["readiness"] == "phonon"
    assert phonon["screening_evidence"]["resolved_readiness"] == "phonon"
    assert phonon["screening_evidence"]["gates"]["md_ion"]["status"] == "missing"
    assert phonon["screening_evidence"]["raw_screening"]["gates"] == phonon["screening_evidence"]["gates"]


def test_sweep_policy_controls_recorded_device_unknown_grid(tmp_path: Path):
    records_path = _write_records(tmp_path)
    exploratory = plan_solarlab_import(
        records_path,
        template_path="configs/nip_MAPbI3.yaml",
        import_policy="exploratory",
        sweep_policy="exploratory",
        include_configs=False,
    )
    production = generate_solarlab_inputs(
        records_path,
        template_path="configs/nip_MAPbI3.yaml",
        out_dir=tmp_path / "production-sweep-policy",
        limit=1,
        import_policy="production",
        sweep_policy="production",
    )

    assert exploratory["sweep_policy"] == "exploratory"
    assert exploratory["sweep_dimensions"]["total_points"] == 243
    assert exploratory["sweep_grid"]["absorber.thickness"] == [300e-9, 500e-9, 800e-9]

    assert production["sweep_policy"] == "production"
    assert production["sweep_dimensions"]["total_points"] == 32
    generated = production["generated"][0]
    assert generated["sweep_parameters"]["absorber.thickness"] == [400e-9, 600e-9]
    config = yaml.safe_load(Path(generated["config_path"]).read_text(encoding="utf-8"))
    assert config["source"]["sweep_policy"] == "production"
    assert config["source"]["sweep_grid"]["absorber.trap_N_t_bulk"] == [1e21, 1e22]


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
            "--sweep-policy",
            "exploratory",
        ],
        check=True,
        cwd=Path(__file__).parents[3],
        text=True,
        capture_output=True,
    )

    plan_path = out_dir / "screening_plan.json"
    assert "Selected 2 candidates" in result.stdout
    assert "Readiness distribution:" in result.stdout
    assert "Gate totals: pass=11, fail=0, missing=1, unknown=0" in result.stdout
    assert "#1 mp-best readiness=promising score=72 source=final_fom_score" in result.stdout
    assert "1x limit reached" in result.stdout
    assert "Sweep policy: exploratory (243 point(s) per candidate recorded)" in result.stdout
    assert plan_path.exists()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert [item["material_id"] for item in plan["selected"]] == ["mp-best", "mp-phonon"]
    assert not (out_dir / "manifest.json").exists()


def test_smoke_device_results_write_json_and_csv(tmp_path: Path):
    records_path = _write_records(tmp_path)
    out_dir = tmp_path / "device-results"
    manifest = generate_solarlab_inputs(
        records_path,
        template_path="configs/nip_MAPbI3.yaml",
        out_dir=out_dir,
        limit=1,
        import_policy="production",
    )

    results = run_smoke_device_results(
        manifest,
        N_grid=6,
        n_points=2,
        V_max=0.05,
        max_configs=1,
    )
    json_path = out_dir / "device_results.json"
    csv_path = out_dir / "device_results.csv"
    write_device_results(results, json_path=json_path, csv_path=csv_path)

    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    assert loaded["schema"] == "solarlab.device_results"
    assert loaded["schema_version"] == "0.1"
    assert loaded["summary"]["status_counts"] == {"completed": 1}
    record = loaded["records"][0]
    assert record["material_id"] == "mp-best"
    assert record["simulation_status"] == "completed"
    assert record["template_path"] == "configs/nip_MAPbI3.yaml"
    assert record["mapped_parameters"]["absorber.eps_r"] == pytest.approx(12.0)
    assert record["screening_evidence"]["gates"]["electronic"]["passed"] is True
    assert record["JV_metrics"]["forward"]["J_sc"] > 0.0
    assert record["error"] is None
    assert "mp-best" in csv_path.read_text(encoding="utf-8")


def test_smoke_device_results_record_failures_without_dropping_metadata(tmp_path: Path):
    records_path = _write_records(tmp_path)
    manifest = generate_solarlab_inputs(
        records_path,
        template_path="configs/nip_MAPbI3.yaml",
        out_dir=tmp_path / "broken-device-results",
        limit=1,
        import_policy="production",
    )
    manifest["generated"][0]["config_path"] = str(tmp_path / "missing.yaml")

    results = run_smoke_device_results(manifest, N_grid=6, n_points=2, V_max=0.05)

    record = results["records"][0]
    assert results["summary"]["status_counts"] == {"failed": 1}
    assert results["summary"]["failed_materials"] == ["mp-best"]
    assert record["material_id"] == "mp-best"
    assert record["simulation_status"] == "failed"
    assert record["mapped_parameters"]["absorber.eps_r"] == pytest.approx(12.0)
    assert record["screening_evidence"]["readiness"] == "promising"
    assert record["JV_metrics"]["forward"] is None
    assert record["error"]["error_type"] in {"FileNotFoundError", "ValueError"}
