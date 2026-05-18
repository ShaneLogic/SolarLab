from __future__ import annotations

import json
from pathlib import Path

import pytest

from perovskite_sim.screening.evaluation_report import run_material_evaluation_report
from perovskite_sim.screening.solarscale import generate_solarlab_inputs, run_smoke_device_results, write_device_results
from tests.unit.screening.test_solarscale_bridge import _write_records


def test_material_evaluation_report_pack_writes_expected_artifacts(tmp_path: Path) -> None:
    records_path = _write_records(tmp_path)
    screening_dir = tmp_path / "screening"
    manifest = generate_solarlab_inputs(
        records_path,
        template_path="configs/nip_MAPbI3.yaml",
        out_dir=screening_dir,
        limit=1,
        import_policy="production",
    )
    device_results = run_smoke_device_results(
        manifest,
        N_grid=6,
        n_points=2,
        V_max=0.05,
        max_configs=1,
    )
    device_results_path = screening_dir / "device_results.json"
    write_device_results(device_results, json_path=device_results_path)

    report_dir = tmp_path / "report"
    report = run_material_evaluation_report(
        config_path=manifest["generated"][0]["config_path"],
        material_record_path=records_path,
        device_results_path=device_results_path,
        out_dir=report_dir,
        quick=True,
    )

    assert report["schema"] == "solarlab.material_evaluation_report"
    assert report["material_id"] == "mp-best"
    assert Path(report["report_path"]).exists()
    assert Path(report["dft_parameter_summary_json"]).exists()
    assert Path(report["dft_parameter_summary_csv"]).exists()
    assert Path(report["device_metrics_json"]).exists()
    assert Path(report["figures"]["jv_curve"]).exists()
    assert Path(report["figures"]["suns_voc"]).exists()
    assert Path(report["figures"]["pseudo_jv"]).exists()
    assert Path(report["figures"]["degradation"]).exists()
    assert Path(report["figures"]["screening_gates"]).exists()
    assert "eqe_skip_reason" in report["figures"]
    assert Path(report["figures"]["eqe_skip_reason"]).exists()

    dft = json.loads(Path(report["dft_parameter_summary_json"]).read_text(encoding="utf-8"))
    bandgap = next(row for row in dft["parameters"] if row["parameter"] == "band_gap_hse_ev")
    assert bandgap["used_by_solarlab"] is False
    eps = next(row for row in dft["parameters"] if row["parameter"] == "dielectric_static_avg")
    assert eps["used_by_solarlab"] is True

    metrics = json.loads(Path(report["device_metrics_json"]).read_text(encoding="utf-8"))
    assert metrics["material_id"] == "mp-best"
    assert metrics["experiment_status"]["jv"]["status"] == "completed"
    assert metrics["experiment_status"]["suns_voc"]["status"] == "completed"
    assert metrics["experiment_status"]["degradation"]["status"] == "completed"
    assert metrics["experiment_status"]["eqe"]["status"] == "skipped"


def test_material_evaluation_report_rejects_missing_material(tmp_path: Path) -> None:
    records_path = _write_records(tmp_path)
    screening_dir = tmp_path / "screening"
    manifest = generate_solarlab_inputs(
        records_path,
        template_path="configs/nip_MAPbI3.yaml",
        out_dir=screening_dir,
        limit=1,
        import_policy="production",
    )
    device_results_path = screening_dir / "device_results.json"
    device_results_path.write_text(
        json.dumps({"schema": "solarlab.device_results", "records": []}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="not found"):
        run_material_evaluation_report(
            config_path=manifest["generated"][0]["config_path"],
            material_record_path=records_path,
            device_results_path=device_results_path,
            out_dir=tmp_path / "report",
            material_id="mp-missing",
            quick=True,
        )
