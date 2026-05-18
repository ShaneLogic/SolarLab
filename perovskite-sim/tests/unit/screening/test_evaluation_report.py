from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import yaml

from perovskite_sim.screening.evaluation_report import ReportQualityError, run_material_evaluation_report
from perovskite_sim.screening.solarscale import generate_solarlab_inputs, run_smoke_device_results, write_device_results
from tests.unit.screening.test_solarscale_bridge import _write_records


@dataclass(frozen=True)
class _Metrics:
    V_oc: float = 1.0
    J_sc: float = 20.0
    FF: float = 0.75
    PCE: float = 0.15
    voc_bracketed: bool = True


@dataclass(frozen=True)
class _FakeJV:
    V_fwd: np.ndarray
    J_fwd: np.ndarray
    V_rev: np.ndarray
    J_rev: np.ndarray
    metrics_fwd: _Metrics
    metrics_rev: _Metrics
    hysteresis_index: float


def _fake_bracketed_jv() -> _FakeJV:
    return _FakeJV(
        V_fwd=np.array([0.0, 1.4]),
        J_fwd=np.array([20.0, -1.0]),
        V_rev=np.array([1.4, 0.0]),
        J_rev=np.array([-1.0, 20.0]),
        metrics_fwd=_Metrics(),
        metrics_rev=_Metrics(),
        hysteresis_index=0.0,
    )


def _patch_fast_production_experiments(monkeypatch: pytest.MonkeyPatch) -> None:
    import perovskite_sim.screening.evaluation_report as report_module

    monkeypatch.setattr(report_module, "run_jv_sweep", lambda *_args, **_kwargs: _fake_bracketed_jv())
    monkeypatch.setattr(report_module, "run_suns_voc", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("fast skip")))
    monkeypatch.setattr(report_module, "run_degradation", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("fast skip")))
    monkeypatch.setattr(report_module, "compute_eqe", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("fast skip")))


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
    assert report["report_profile"] == "smoke"
    assert report["physics_quality_status"] == "publication_blocked"
    assert report["publication_ready"] is False
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
    assert metrics["report_profile"] == "smoke"
    assert metrics["figure_quality"]["jv_curve"]["quality_status"] == "workflow_smoke"


def test_diagnostic_report_marks_missing_eqe_as_diagnostic_only(tmp_path: Path) -> None:
    records_path = _write_records(tmp_path)
    screening_dir = tmp_path / "screening"
    manifest = generate_solarlab_inputs(
        records_path,
        template_path="configs/nip_MAPbI3.yaml",
        out_dir=screening_dir,
        limit=1,
        import_policy="production",
    )
    device_results = run_smoke_device_results(manifest, N_grid=6, n_points=2, V_max=0.05, max_configs=1)
    device_results_path = screening_dir / "device_results.json"
    write_device_results(device_results, json_path=device_results_path)

    report = run_material_evaluation_report(
        config_path=manifest["generated"][0]["config_path"],
        material_record_path=records_path,
        device_results_path=device_results_path,
        out_dir=tmp_path / "diagnostic-report",
        profile="diagnostic",
    )

    assert report["report_profile"] == "diagnostic"
    assert report["physics_quality_status"] == "publication_blocked"
    assert "report profile 'diagnostic' is not publication-grade" in report["blocking_reasons"]
    assert report["figure_quality"]["jv_curve"]["quality_status"] == "diagnostic_only"
    assert "eqe_skip_reason" in report["figures"]


def test_production_report_rejects_legacy_template(tmp_path: Path) -> None:
    records_path = _write_records(tmp_path)
    screening_dir = tmp_path / "screening"
    manifest = generate_solarlab_inputs(
        records_path,
        template_path="configs/nip_MAPbI3.yaml",
        out_dir=screening_dir,
        limit=1,
        import_policy="production",
    )
    device_results = run_smoke_device_results(manifest, N_grid=6, n_points=2, V_max=0.05, max_configs=1)
    device_results_path = screening_dir / "device_results.json"
    write_device_results(device_results, json_path=device_results_path)

    with pytest.raises(ReportQualityError, match="activate_bandgap=True"):
        run_material_evaluation_report(
            config_path=manifest["generated"][0]["config_path"],
            material_record_path=records_path,
            device_results_path=device_results_path,
            out_dir=tmp_path / "production-report",
            profile="production",
        )


def test_production_report_rejects_unbracketed_jv(tmp_path: Path) -> None:
    records_path = _write_records(tmp_path)
    screening_dir = tmp_path / "screening"
    manifest = generate_solarlab_inputs(
        records_path,
        template_path="configs/solarscale_nip_band_aligned.yaml",
        out_dir=screening_dir,
        limit=1,
        import_policy="production",
        activate_bandgap=True,
    )
    device_results = run_smoke_device_results(manifest, N_grid=6, n_points=2, V_max=0.05, max_configs=1)
    device_results_path = screening_dir / "device_results.json"
    write_device_results(device_results, json_path=device_results_path)

    @dataclass(frozen=True)
    class Metrics:
        V_oc: float = 0.0
        J_sc: float = 10.0
        FF: float = 0.0
        PCE: float = 0.0
        voc_bracketed: bool = False

    @dataclass(frozen=True)
    class FakeJV:
        V_fwd: np.ndarray
        J_fwd: np.ndarray
        V_rev: np.ndarray
        J_rev: np.ndarray
        metrics_fwd: Metrics
        metrics_rev: Metrics
        hysteresis_index: float

    fake = FakeJV(
        V_fwd=np.array([0.0, 0.05]),
        J_fwd=np.array([10.0, 9.0]),
        V_rev=np.array([0.05, 0.0]),
        J_rev=np.array([9.0, 10.0]),
        metrics_fwd=Metrics(),
        metrics_rev=Metrics(),
        hysteresis_index=0.0,
    )

    import perovskite_sim.screening.evaluation_report as report_module

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(report_module, "run_jv_sweep", lambda *_args, **_kwargs: fake)
    with pytest.raises(ReportQualityError, match="Production JV quality gate failed"):
        run_material_evaluation_report(
            config_path=manifest["generated"][0]["config_path"],
            material_record_path=records_path,
            device_results_path=device_results_path,
            out_dir=tmp_path / "production-report",
            profile="production",
        )
    monkeypatch.undo()


def test_production_report_blocks_missing_band_alignment_provenance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    records_path = _write_records(tmp_path)
    screening_dir = tmp_path / "screening"
    manifest = generate_solarlab_inputs(
        records_path,
        template_path="configs/solarscale_nip_band_aligned.yaml",
        out_dir=screening_dir,
        limit=1,
        import_policy="production",
        activate_bandgap=True,
    )
    config_path = Path(manifest["generated"][0]["config_path"])
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["source"]["band_alignment_provenance"] = None
    config["source"]["material_metadata"]["band_alignment_provenance"] = None
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    device_results = run_smoke_device_results(manifest, N_grid=6, n_points=2, V_max=0.05, max_configs=1)
    device_results_path = screening_dir / "device_results.json"
    write_device_results(device_results, json_path=device_results_path)
    _patch_fast_production_experiments(monkeypatch)

    report = run_material_evaluation_report(
        config_path=config_path,
        material_record_path=records_path,
        device_results_path=device_results_path,
        out_dir=tmp_path / "production-report",
        profile="production",
    )

    assert report["publication_ready"] is False
    assert report["physics_quality_status"] == "publication_blocked"
    assert any("band-alignment" in reason for reason in report["blocking_reasons"])


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
