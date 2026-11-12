"""Integration and parity tests for SolarScale screening and evaluation reports."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest
import yaml

from perovskite_sim.experiments.jv_sweep import run_jv_sweep as legacy_run_jv_sweep
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.screening import solarscale as legacy_ss
from solarlab.contracts.units import ELEMENTARY_CHARGE_C
from solarlab.experiments.evaluation_report import (
    ReportQualityError,
    run_material_evaluation_report,
    run_solarscale_suns_voc_1d,
)
from solarlab.experiments.solarscale import (
    BOLTZMANN_CONSTANT_J_K,
    expand_sweep_manifest,
    generate_solarlab_inputs,
    manifest_sha256,
    plan_solarlab_import,
    run_smoke_device_results,
    solve_solarscale_jv_1d,
    write_device_results,
)


def _consistent_ni(eg_ev: float, nc: float = 1.0e25, nv: float = 1.0e25) -> float:
    vt = BOLTZMANN_CONSTANT_J_K * 300.0 / ELEMENTARY_CHARGE_C
    return math.sqrt(nc * nv) * math.exp(-eg_ev / (2.0 * vt))


def _band_aligned_template(
    *,
    absorber_eg: float = 1.60,
    interface_srv: float = 0.0,
    optical_material: str | None = None,
) -> dict[str, Any]:
    ni_htl = _consistent_ni(2.0, 2.0e25, 3.0e25)
    ni_abs = _consistent_ni(absorber_eg, 1.0e25, 1.0e25)
    ni_etl = _consistent_ni(2.0, 4.0e25, 2.0e25)
    absorber_layer: dict[str, Any] = {
        "name": "absorber",
        "role": "absorber",
        "thickness": 300.0e-9,
        "eps_r": 25.0,
        "mu_n": 1.934e-3,
        "mu_p": 1.934e-3,
        "chi": 3.9,
        "Eg": absorber_eg,
        "Nc300": 1.0e25,
        "Nv300": 1.0e25,
        "ni": ni_abs,
        "n1": ni_abs,
        "p1": ni_abs,
        "N_A": 0.0,
        "N_D": 0.0,
        "tau_n": 2.0e-7,
        "tau_p": 2.0e-7,
        "B_rad": 1.0e-18,
        "C_n": 1.0e-42,
        "C_p": 1.0e-42,
        "alpha": 2.0e6,
        "P0": 0.0,
        "D_ion": 0.0,
        "P_lim": 1.0e28,
    }
    if optical_material is not None:
        absorber_layer["optical_material"] = optical_material
    return {
        "device": {
            "T": 300.0,
            "Phi": 2.0e21,
            "built_in_potential_mode": "semiconductor_work_function",
            "dos_band_potentials": True,
            "interfaces": [
                [interface_srv, interface_srv],
                [interface_srv, interface_srv],
            ],
        },
        "layers": [
            {
                "name": "HTL",
                "role": "HTL",
                "thickness": 80.0e-9,
                "eps_r": 3.0,
                "mu_n": 3.868e-5,
                "mu_p": 3.868e-5,
                "chi": 3.6,
                "Eg": 2.0,
                "Nc300": 2.0e25,
                "Nv300": 3.0e25,
                "ni": ni_htl,
                "n1": ni_htl,
                "p1": ni_htl,
                "N_A": 1.0e24,
                "N_D": 0.0,
                "tau_n": 1.0e-6,
                "tau_p": 1.0e-6,
                "B_rad": 1.0e-18,
                "C_n": 1.0e-42,
                "C_p": 1.0e-42,
                "alpha": 0.0,
                "P0": 0.0,
                "D_ion": 0.0,
                "P_lim": 1.0e28,
            },
            absorber_layer,
            {
                "name": "ETL",
                "role": "ETL",
                "thickness": 80.0e-9,
                "eps_r": 9.0,
                "mu_n": 3.868e-5,
                "mu_p": 3.868e-5,
                "chi": 4.0,
                "Eg": 2.0,
                "Nc300": 4.0e25,
                "Nv300": 2.0e25,
                "ni": ni_etl,
                "n1": ni_etl,
                "p1": ni_etl,
                "N_A": 0.0,
                "N_D": 1.0e24,
                "tau_n": 1.0e-6,
                "tau_p": 1.0e-6,
                "B_rad": 1.0e-18,
                "C_n": 1.0e-42,
                "C_p": 1.0e-42,
                "alpha": 0.0,
                "P0": 0.0,
                "D_ion": 0.0,
                "P_lim": 1.0e28,
            },
        ],
    }


def _sample_records_payload() -> dict[str, Any]:
    return {
        "records": [
            {
                "material_id": "cand_alpha",
                "formula": "CsPbI2Br",
                "screening": {
                    "production_solarlab_ready": True,
                    "provisional_solarlab_ready": True,
                    "readiness": "promising",
                    "final_fom_score": 0.94,
                    "gates": {"stability": "pass", "bandgap": True, "toxicity": "pass"},
                },
                "properties": {
                    "band_gap_hse_ev": {
                        "value": 1.60,
                        "unit": "eV",
                        "provenance": {"kind": "computed", "source": "vasp_hse06"},
                    },
                    "dielectric_static_avg": {
                        "value": 25.0,
                        "unit": "1",
                        "provenance": {"kind": "computed", "source": "dfpt"},
                    },
                    "electron_affinity_ev": {
                        "value": 3.9,
                        "unit": "eV",
                        "provenance": {"kind": "computed", "source": "slab"},
                    },
                    "electron_mobility_cm2_v_s": {
                        "value": 19.34,
                        "unit": "cm^2/(V s)",
                        "provenance": {"kind": "computed", "source": "amset"},
                    },
                    "hole_mobility_cm2_v_s": {
                        "value": 19.34,
                        "unit": "cm^2/(V s)",
                        "provenance": {"kind": "computed", "source": "amset"},
                    },
                    "band_alignment_provenance": {
                        "value": "dft_slab_v1",
                        "unit": "",
                        "provenance": {"kind": "computed", "source": "slab"},
                    },
                },
            },
            {
                "material_id": "cand_exploratory_only",
                "formula": "RbSnI3",
                "screening": {
                    "production_solarlab_ready": False,
                    "provisional_solarlab_ready": True,
                    "readiness": "phonon",
                    "ml_pv_score": 0.72,
                },
                "properties": {
                    "band_gap_hse_ev": {
                        "value": 1.60,
                        "unit": "eV",
                        "provenance": {"kind": "computed", "source": "vasp"},
                    },
                    "dielectric_static_avg": {
                        "value": 22.0,
                        "unit": "1",
                        "provenance": {"kind": "computed", "source": "dfpt"},
                    },
                },
            },
        ]
    }


def test_import_plan_and_sweep_manifest_parity(tmp_path: Path) -> None:
    records_path = tmp_path / "records.json"
    template_path = tmp_path / "template.yaml"
    records_path.write_text(json.dumps(_sample_records_payload()), encoding="utf-8")
    template_path.write_text(
        yaml.safe_dump(_band_aligned_template(), sort_keys=False), encoding="utf-8"
    )

    for policy in ("production", "exploratory"):
        plan_old = legacy_ss.plan_solarlab_import(
            records_path,
            template_path=template_path,
            import_policy=policy,
            activate_bandgap=True,
        )
        plan_new = plan_solarlab_import(
            records_path,
            template_path=template_path,
            import_policy=policy,
            activate_bandgap=True,
        )
        assert plan_new["selected"] == plan_old["selected"]
        assert plan_new["skipped"] == plan_old["skipped"]
        assert plan_new["summary"] == plan_old["summary"]

    out_old = tmp_path / "out_old"
    out_new = tmp_path / "out_new"
    man_old = legacy_ss.generate_solarlab_inputs(
        records_path,
        template_path=template_path,
        out_dir=out_old,
        import_policy="production",
        activate_bandgap=True,
    )
    man_new = generate_solarlab_inputs(
        records_path,
        template_path=template_path,
        out_dir=out_new,
        import_policy="production",
        activate_bandgap=True,
    )
    assert len(man_new["generated"]) == len(man_old["generated"]) == 1
    assert (out_new / "cand_alpha.yaml").read_text(encoding="utf-8") == (
        out_old / "cand_alpha.yaml"
    ).read_text(encoding="utf-8")

    sw_old = legacy_ss.expand_sweep_manifest(
        man_old,
        out_dir=tmp_path / "sw_old",
        max_points=4,
    )
    sw_new = expand_sweep_manifest(
        man_new,
        out_dir=tmp_path / "sw_new",
        max_points=4,
    )
    assert [g["sweep_values"] for g in sw_new["generated"]] == [
        g["sweep_values"] for g in sw_old["generated"]
    ]
    assert len(manifest_sha256(sw_new)) == 64


@pytest.mark.parametrize("iface_srv", [0.0, 1.0])
def test_solarscale_steady_jv_numerical_parity_against_legacy(
    tmp_path: Path, iface_srv: float
) -> None:
    cfg_path = tmp_path / f"device_srv_{iface_srv}.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            _band_aligned_template(absorber_eg=1.60, interface_srv=iface_srv),
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    res_new = solve_solarscale_jv_1d(cfg_path, N_grid=24, n_points=7, V_max=1.25)
    stack_old = load_device_from_yaml(str(cfg_path))
    res_old = legacy_run_jv_sweep(
        stack_old,
        N_grid=24,
        n_points=7,
        v_rate=0.01,
        V_max=1.25,
    )

    m_new = res_new.metrics_fwd
    m_old = res_old.metrics_fwd
    assert m_new.voc_bracketed is True and m_old.voc_bracketed is True
    assert abs(m_new.V_oc - m_old.V_oc) <= 1.0e-4
    assert abs(m_new.J_sc - m_old.J_sc) / abs(m_old.J_sc) <= 1.0e-4
    assert abs(m_new.FF - m_old.FF) <= 1.0e-4
    assert abs(m_new.PCE - m_old.PCE) <= 1.0e-4
    assert res_new.scientific_qualification is False


def test_unsupported_legacy_manual_vbi_retained_as_failed_and_report_packs(
    tmp_path: Path,
) -> None:
    records_path = tmp_path / "records.json"
    records_path.write_text(json.dumps(_sample_records_payload()), encoding="utf-8")

    # 1. Unsupported legacy manual V_bi template (chi=0, Eg=0) MUST fail closed per candidate.
    bad_template = _band_aligned_template()
    bad_template["device"]["built_in_potential_mode"] = "manual"
    bad_template["device"]["V_bi"] = 1.0
    for layer in bad_template["layers"]:
        layer["chi"] = 0.0
        layer["Eg"] = 0.0
    bad_tmpl_path = tmp_path / "bad_template.yaml"
    bad_tmpl_path.write_text(yaml.safe_dump(bad_template, sort_keys=False), encoding="utf-8")

    bad_inputs = tmp_path / "bad_inputs"
    bad_man = generate_solarlab_inputs(
        records_path,
        template_path=bad_tmpl_path,
        out_dir=bad_inputs,
        import_policy="exploratory",
        activate_bandgap=False,
    )
    bad_results = run_smoke_device_results(
        bad_man, N_grid=12, n_points=3, V_max=0.2, max_configs=None
    )
    assert bad_results["summary"]["status_counts"]["failed"] == 2
    for rec in bad_results["records"]:
        assert rec["simulation_status"] == "failed"
        assert rec["error"]["error_type"] in {"CapabilityError", "ValidationError"}

    # 2. Valid band-aligned template -> smoke device results + Suns-Voc + report packs.
    good_tmpl_path = tmp_path / "good_template.yaml"
    good_tmpl_path.write_text(
        yaml.safe_dump(_band_aligned_template(), sort_keys=False), encoding="utf-8"
    )
    good_inputs = tmp_path / "good_inputs"
    good_man = generate_solarlab_inputs(
        records_path,
        template_path=good_tmpl_path,
        out_dir=good_inputs,
        import_policy="production",
        activate_bandgap=True,
    )
    cfg_file = good_inputs / "cand_alpha.yaml"
    dev_res = run_smoke_device_results(
        good_man, N_grid=12, n_points=4, V_max=1.25, max_configs=None
    )
    assert dev_res["summary"]["status_counts"]["completed"] == 1
    dev_res_json = tmp_path / "device_results.json"
    dev_res_csv = tmp_path / "device_results.csv"
    write_device_results(dev_res, json_path=dev_res_json, csv_path=dev_res_csv)
    assert dev_res_json.is_file() and dev_res_csv.is_file()

    suns = run_solarscale_suns_voc_1d(cfg_file, suns_levels=(0.2, 1.0, 5.0), N_grid=12)
    assert suns.V_oc.shape == (3,)
    assert float(suns.V_oc[2]) > float(suns.V_oc[1]) > float(suns.V_oc[0]) > 0.5
    assert 0.5 < suns.pseudo_FF < 0.95

    compact_smoke_settings = {
        "jv": {"N_grid": 12, "n_points": 3, "v_rate": 5.0, "V_max": 0.2},
        "suns_voc": {"suns_levels": (0.5, 1.0, 2.0), "N_grid": 12, "t_settle": 5.0e-4},
    }
    smoke_report = run_material_evaluation_report(
        config_path=cfg_file,
        material_record_path=records_path,
        device_results_path=dev_res_json,
        out_dir=tmp_path / "report_smoke",
        profile="smoke",
        settings_override=compact_smoke_settings,
    )
    assert smoke_report["report_profile"] == "smoke"
    assert smoke_report["publication_ready"] is False
    assert smoke_report["scientific_qualification"] is False
    assert smoke_report["experiment_status"]["jv"]["status"] == "completed"
    assert smoke_report["experiment_status"]["suns_voc"]["status"] == "completed"
    assert smoke_report["experiment_status"]["eqe"]["status"] == "skipped"
    assert Path(smoke_report["figures"]["eqe_skip_reason"]).is_file()
    assert Path(smoke_report["report_path"]).is_file()

    # Production profile rejects missing absorber optical_material before running plots.
    with pytest.raises(ReportQualityError, match="optical_material"):
        run_material_evaluation_report(
            config_path=cfg_file,
            material_record_path=records_path,
            device_results_path=dev_res_json,
            out_dir=tmp_path / "report_prod_missing_optics",
            profile="production",
        )

    # Production profile with optical_material placeholder runs JV + Suns-Voc and marks
    # publication_blocked when material-specific n/k EQE resources are absent.
    opt_tmpl_path = tmp_path / "opt_template.yaml"
    opt_tmpl_path.write_text(
        yaml.safe_dump(
            _band_aligned_template(optical_material="MAPbI3"), sort_keys=False
        ),
        encoding="utf-8",
    )
    opt_inputs = tmp_path / "opt_inputs"
    generate_solarlab_inputs(
        records_path,
        template_path=opt_tmpl_path,
        out_dir=opt_inputs,
        import_policy="production",
        activate_bandgap=True,
    )
    compact_prod_settings = {
        "jv": {"N_grid": 12, "n_points": 5, "v_rate": 0.5, "V_max": 1.25},
        "suns_voc": {"suns_levels": (0.5, 1.0, 2.0), "N_grid": 12, "t_settle": 1.0e-2},
    }
    prod_report = run_material_evaluation_report(
        config_path=opt_inputs / "cand_alpha.yaml",
        material_record_path=records_path,
        device_results_path=dev_res_json,
        out_dir=tmp_path / "report_prod",
        profile="production",
        settings_override=compact_prod_settings,
    )
    assert prod_report["physics_quality_status"] == "publication_blocked"
    assert prod_report["publication_ready"] is False

    # Production profile fails closed with evaluation_failure.json if Voc is not bracketed.
    unbracketed_cfg = yaml.safe_load((opt_inputs / "cand_alpha.yaml").read_text(encoding="utf-8"))
    unbracketed_cfg["device"]["Phi"] = 0.0
    unbracketed_path = tmp_path / "unbracketed.yaml"
    unbracketed_path.write_text(
        yaml.safe_dump(unbracketed_cfg, sort_keys=False), encoding="utf-8"
    )
    fail_dir = tmp_path / "report_prod_fail"
    with pytest.raises(ReportQualityError, match="Production JV quality gate failed"):
        run_material_evaluation_report(
            config_path=unbracketed_path,
            material_record_path=records_path,
            device_results_path=dev_res_json,
            out_dir=fail_dir,
            profile="production",
            settings_override=compact_prod_settings,
        )
    assert (fail_dir / "evaluation_failure.json").is_file()
