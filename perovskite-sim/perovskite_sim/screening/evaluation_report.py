"""Material evaluation report pack for SolarScale -> SolarLab workflows."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from perovskite_sim.experiments.degradation import run_degradation
from perovskite_sim.experiments.eqe import compute_eqe
from perovskite_sim.experiments.jv_sweep import run_jv_sweep
from perovskite_sim.experiments.suns_voc import run_suns_voc
from perovskite_sim.models.config_loader import load_device_from_yaml


DFT_PARAMETER_TARGETS = {
    "band_gap_hse_ev": "absorber.Eg",
    "dielectric_static_avg": "absorber.eps_r",
    "electron_mobility_cm2_v_s": "absorber.mu_n",
    "hole_mobility_cm2_v_s": "absorber.mu_p",
    "ion_diffusion_coefficient_m2_s": "absorber.D_ion",
    "ion_activation_energy_ev": "absorber.E_a_ion",
    "electron_effective_mass_m0": None,
    "hole_effective_mass_m0": None,
    "slme_0p5um": None,
    "absorption_edge_ev": None,
}

ReportProfile = Literal["smoke", "diagnostic", "production"]
FIGURE_QUALITY_BY_PROFILE: dict[ReportProfile, str] = {
    "smoke": "workflow_smoke",
    "diagnostic": "diagnostic_only",
    "production": "production_candidate",
}


class ReportQualityError(RuntimeError):
    """Raised when a production report would look physical but is invalid."""


def run_material_evaluation_report(
    *,
    config_path: str | Path,
    material_record_path: str | Path,
    device_results_path: str | Path,
    out_dir: str | Path,
    quick: bool = False,
    profile: ReportProfile = "diagnostic",
    material_id: str | None = None,
) -> dict[str, Any]:
    """Create a report pack for one SolarScale-imported SolarLab config."""

    config_file = Path(config_path)
    record_file = Path(material_record_path)
    device_results_file = Path(device_results_path)
    out = Path(out_dir)
    figures = out / "figures"
    out.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    config = _load_yaml(config_file)
    profile = _resolve_profile(profile=profile, quick=quick)
    if profile == "production":
        _validate_production_config(config)
    material_id = material_id or _config_material_id(config) or _first_record_id(record_file)
    record = _select_material_record(_load_json(record_file), material_id)
    device_results = _load_json(device_results_file)
    device_record = _select_device_result(device_results, material_id, config_file)

    dft_summary = _dft_parameter_summary(record, config, device_record)
    dft_json = out / "dft_parameter_summary.json"
    dft_csv = out / "dft_parameter_summary.csv"
    dft_json.write_text(json.dumps(dft_summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_dft_summary_csv(dft_csv, dft_summary)

    device_metrics = _device_metrics_summary(device_results, device_record, material_id)
    stack = load_device_from_yaml(str(config_file))
    settings = _profile_settings(profile)

    figure_paths: dict[str, str] = {}
    figure_quality: dict[str, dict[str, Any]] = {}
    experiment_status: dict[str, Any] = {}
    blocking_reasons: list[str] = []
    optical_blocking_reasons: list[str] = []
    if profile == "production":
        optical_blocking_reasons = _production_optical_blocking_reasons(config)
        blocking_reasons.extend(optical_blocking_reasons)
        blocking_reasons.extend(_production_band_alignment_blocking_reasons(config))
    else:
        blocking_reasons.append(f"report profile {profile!r} is not publication-grade")
    jv_result = _run_experiment(
        "jv",
        experiment_status,
        lambda: run_jv_sweep(stack, **settings["jv"]),
    )
    if jv_result is not None:
        if profile == "production":
            jv_reasons = _production_jv_blocking_reasons(jv_result)
            if jv_reasons:
                failure_path = out / "evaluation_failure.json"
                failure_path.write_text(
                    json.dumps(
                        {
                            "schema": "solarlab.material_evaluation_failure",
                            "schema_version": "0.1",
                            "material_id": material_id,
                            "profile": profile,
                            "blocking_reasons": jv_reasons,
                        },
                        indent=2,
                        sort_keys=True,
                    ),
                    encoding="utf-8",
                )
                raise ReportQualityError("Production JV quality gate failed: " + "; ".join(jv_reasons))
        path = figures / "jv_curve.png"
        _plot_jv(path, jv_result)
        figure_paths["jv_curve"] = str(path)
        _record_figure_quality(figure_quality, "jv_curve", path, profile)
        device_metrics["quick_experiments"]["jv"] = _jv_metrics_dict(jv_result)
    else:
        if profile == "production":
            raise ReportQualityError(
                "Production JV solver failed: " + experiment_status["jv"]["error_message"]
            )
        path = figures / "jv_curve.png"
        _plot_error(path, "JV sweep failed", experiment_status["jv"]["error_message"])
        figure_paths["jv_curve"] = str(path)
        _record_figure_quality(figure_quality, "jv_curve", path, profile, status="publication_blocked")

    suns_result = _run_experiment(
        "suns_voc",
        experiment_status,
        lambda: run_suns_voc(stack, **settings["suns_voc"]),
    )
    if suns_result is not None:
        suns_path = figures / "suns_voc.png"
        pseudo_path = figures / "pseudo_jv.png"
        _plot_suns_voc(suns_path, suns_result)
        _plot_pseudo_jv(pseudo_path, suns_result)
        figure_paths["suns_voc"] = str(suns_path)
        figure_paths["pseudo_jv"] = str(pseudo_path)
        _record_figure_quality(figure_quality, "suns_voc", suns_path, profile)
        _record_figure_quality(figure_quality, "pseudo_jv", pseudo_path, profile)
        device_metrics["quick_experiments"]["suns_voc"] = _jsonable_dataclass(suns_result)
    else:
        for name, title in (("suns_voc", "Suns-Voc failed"), ("pseudo_jv", "Pseudo-JV failed")):
            path = figures / f"{name}.png"
            _plot_error(path, title, experiment_status["suns_voc"]["error_message"])
            figure_paths[name] = str(path)
            _record_figure_quality(figure_quality, name, path, profile, status="publication_blocked")

    degradation_result = _run_experiment(
        "degradation",
        experiment_status,
        lambda: run_degradation(stack, **settings["degradation"]),
    )
    if degradation_result is not None:
        path = figures / "degradation.png"
        _plot_degradation(path, degradation_result)
        figure_paths["degradation"] = str(path)
        _record_figure_quality(figure_quality, "degradation", path, profile)
        device_metrics["quick_experiments"]["degradation"] = _jsonable_dataclass(degradation_result)
    else:
        path = figures / "degradation.png"
        _plot_error(path, "Degradation failed", experiment_status["degradation"]["error_message"])
        figure_paths["degradation"] = str(path)
        _record_figure_quality(figure_quality, "degradation", path, profile, status="publication_blocked")

    eqe_result = _run_experiment(
        "eqe",
        experiment_status,
        lambda: compute_eqe(stack, **settings["eqe"]),
    )
    if eqe_result is not None:
        path = figures / "eqe.png"
        _plot_eqe(path, eqe_result)
        figure_paths["eqe"] = str(path)
        _record_figure_quality(
            figure_quality,
            "eqe",
            path,
            profile,
            status="publication_blocked" if optical_blocking_reasons else None,
        )
        device_metrics["quick_experiments"]["eqe"] = _jsonable_dataclass(eqe_result)
    else:
        if profile == "production":
            blocking_reasons.append("EQE skipped because material-specific optical n/k data are unavailable or incomplete")
        skip_path = figures / "eqe_skip_reason.json"
        skip_payload = {
            "schema": "solarlab.eqe_skip_reason",
            "schema_version": "0.1",
            "material_id": material_id,
            "profile": profile,
            "quality_status": "publication_blocked" if profile == "production" else FIGURE_QUALITY_BY_PROFILE[profile],
            "reason": experiment_status["eqe"]["error_message"],
            "policy": "EQE is skipped when wavelength-resolved optical n/k data are unavailable or incomplete.",
        }
        skip_path.write_text(json.dumps(skip_payload, indent=2, sort_keys=True), encoding="utf-8")
        figure_paths["eqe_skip_reason"] = str(skip_path)
        _record_figure_quality(
            figure_quality,
            "eqe_skip_reason",
            skip_path,
            profile,
            status="publication_blocked" if profile == "production" else None,
        )

    gates_path = figures / "screening_gates.png"
    _plot_screening_gates(gates_path, record)
    figure_paths["screening_gates"] = str(gates_path)
    _record_figure_quality(figure_quality, "screening_gates", gates_path, profile)

    physics_quality_status = _physics_quality_status(profile, blocking_reasons)
    publication_ready = profile == "production" and not blocking_reasons
    quality_summary = {
        "report_profile": profile,
        "physics_quality_status": physics_quality_status,
        "publication_ready": publication_ready,
        "blocking_reasons": blocking_reasons,
        "figure_quality": figure_quality,
    }
    device_metrics["experiment_status"] = experiment_status
    device_metrics.update(quality_summary)
    device_metrics_json = out / "device_metrics.json"
    device_metrics_json.write_text(json.dumps(device_metrics, indent=2, sort_keys=True), encoding="utf-8")

    report_path = out / "report.md"
    report_path.write_text(
        _render_report_md(
            material_id=material_id,
            config_path=config_file,
            material_record_path=record_file,
            device_results_path=device_results_file,
            dft_summary=dft_summary,
            device_metrics=device_metrics,
            figure_paths=figure_paths,
            quality_summary=quality_summary,
        ),
        encoding="utf-8",
    )

    manifest = {
        "schema": "solarlab.material_evaluation_report",
        "schema_version": "0.1",
        "material_id": material_id,
        "config_path": str(config_file),
        "material_record_path": str(record_file),
        "device_results_path": str(device_results_file),
        "out_dir": str(out),
        "report_profile": profile,
        "physics_quality_status": physics_quality_status,
        "publication_ready": publication_ready,
        "blocking_reasons": blocking_reasons,
        "report_path": str(report_path),
        "dft_parameter_summary_json": str(dft_json),
        "dft_parameter_summary_csv": str(dft_csv),
        "device_metrics_json": str(device_metrics_json),
        "figures": figure_paths,
        "figure_quality": figure_quality,
        "experiment_status": experiment_status,
    }
    manifest_path = out / "evaluation_manifest.json"
    manifest["manifest_path"] = str(manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _config_material_id(config: Mapping[str, Any]) -> str | None:
    source = config.get("source")
    if isinstance(source, Mapping) and source.get("material_id"):
        return str(source["material_id"])
    return None


def _first_record_id(path: Path) -> str:
    payload = _load_json(path)
    records = payload.get("records") if isinstance(payload, Mapping) else None
    if isinstance(records, list) and records:
        return str(records[0].get("material_id"))
    if payload.get("material_id"):
        return str(payload["material_id"])
    raise ValueError(f"No material_id found in {path}")


def _select_material_record(payload: Mapping[str, Any], material_id: str) -> dict[str, Any]:
    if payload.get("material_id") == material_id:
        return dict(payload)
    for record in payload.get("records", []) or []:
        if str(record.get("material_id")) == material_id:
            return dict(record)
    raise ValueError(f"Material {material_id!r} not found in material record payload")


def _select_device_result(payload: Mapping[str, Any], material_id: str, config_path: Path) -> dict[str, Any]:
    candidates = payload.get("records", []) or []
    config_resolved = str(config_path)
    for record in candidates:
        if str(record.get("material_id")) == material_id and str(record.get("config_path")) == config_resolved:
            return dict(record)
    for record in candidates:
        if str(record.get("material_id")) == material_id:
            return dict(record)
    return {}


def _dft_parameter_summary(
    record: Mapping[str, Any],
    config: Mapping[str, Any],
    device_record: Mapping[str, Any],
) -> dict[str, Any]:
    source = config.get("source") if isinstance(config.get("source"), Mapping) else {}
    mapped = dict(source.get("mapped_parameters", {}) or device_record.get("mapped_parameters", {}) or {})
    metadata = dict(source.get("material_metadata", {}) or device_record.get("material_metadata", {}) or {})
    properties = record.get("properties", {}) or {}
    rows: list[dict[str, Any]] = []
    for name, target in DFT_PARAMETER_TARGETS.items():
        prop = properties.get(name)
        if isinstance(prop, Mapping):
            value = prop.get("value")
            unit = prop.get("unit", "")
            provenance = prop.get("provenance") if isinstance(prop.get("provenance"), Mapping) else {}
        else:
            value = metadata.get(name)
            unit = ""
            provenance = {}
        rows.append(
            {
                "parameter": name,
                "value": value,
                "unit": unit,
                "provenance_kind": provenance.get("kind", "metadata" if name in metadata else "missing"),
                "provenance_source": provenance.get("source", ""),
                "target": target,
                "mapped_value": mapped.get(target) if target else None,
                "used_by_solarlab": bool(target and target in mapped),
            }
        )
    return {
        "schema": "solarlab.dft_parameter_summary",
        "schema_version": "0.1",
        "material_id": record.get("material_id"),
        "mapped_parameters": mapped,
        "material_metadata": metadata,
        "parameters": rows,
    }


def _write_dft_summary_csv(path: Path, summary: Mapping[str, Any]) -> None:
    fieldnames = [
        "parameter",
        "value",
        "unit",
        "provenance_kind",
        "provenance_source",
        "target",
        "mapped_value",
        "used_by_solarlab",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary.get("parameters", []) or [])


def _device_metrics_summary(
    device_results: Mapping[str, Any],
    device_record: Mapping[str, Any],
    material_id: str,
) -> dict[str, Any]:
    return {
        "schema": "solarlab.material_device_metrics",
        "schema_version": "0.1",
        "material_id": material_id,
        "simulation_status": device_record.get("simulation_status"),
        "JV_metrics": device_record.get("JV_metrics"),
        "sweep_values": device_record.get("sweep_values", {}),
        "warnings": device_record.get("warnings", []),
        "error": device_record.get("error"),
        "device_results_summary": device_results.get("summary", {}),
        "quick_experiments": {},
    }


def _resolve_profile(*, profile: ReportProfile, quick: bool) -> ReportProfile:
    if quick:
        return "smoke"
    if profile not in {"smoke", "diagnostic", "production"}:
        raise ValueError(f"Unknown report profile {profile!r}")
    return profile


def _validate_production_config(config: Mapping[str, Any]) -> None:
    source = config.get("source") if isinstance(config.get("source"), Mapping) else {}
    if not source.get("activate_bandgap"):
        raise ReportQualityError("Production report requires activate_bandgap=True in the generated SolarLab config source")
    layers = config.get("layers")
    if not isinstance(layers, list):
        raise ReportQualityError("Production report requires a SolarLab config with layers")
    invalid: list[str] = []
    absorber_has_optics = False
    for layer in layers:
        if not isinstance(layer, Mapping) or layer.get("role") == "substrate":
            continue
        role = str(layer.get("role") or layer.get("name") or "layer")
        chi = _optional_float(layer.get("chi"))
        eg = _optional_float(layer.get("Eg"))
        if chi is None or eg is None or eg <= 0.0:
            invalid.append(role)
        if layer.get("role") == "absorber" and layer.get("optical_material"):
            absorber_has_optics = True
    if invalid:
        raise ReportQualityError(
            "Production report requires chi and positive Eg on every electrical layer; invalid layers: "
            + ", ".join(invalid)
        )
    if not absorber_has_optics:
        raise ReportQualityError("Production report requires absorber optical_material for EQE/TMM provenance")


def _production_optical_blocking_reasons(config: Mapping[str, Any]) -> list[str]:
    source = config.get("source") if isinstance(config.get("source"), Mapping) else {}
    if source.get("schema") != "solarlab.solarscale_import_config":
        return []
    absorber = _config_absorber_layer(config)
    optical_material = str(absorber.get("optical_material") or "").strip() if absorber else ""
    metadata = source.get("material_metadata") if isinstance(source.get("material_metadata"), Mapping) else {}
    optical_provenance = (
        source.get("optical_material_provenance")
        or source.get("nk_provenance")
        or metadata.get("optical_material_provenance")
        or metadata.get("nk_provenance")
    )
    reasons: list[str] = []
    if optical_material == "MAPbI3":
        reasons.append(
            "Production optical policy blocked: absorber optical_material is the MAPbI3 template placeholder, not material-specific n/k evidence"
        )
    if not optical_provenance:
        reasons.append(
            "Production optical policy blocked: material-specific optical n/k provenance is missing"
        )
    return reasons


def _production_band_alignment_blocking_reasons(config: Mapping[str, Any]) -> list[str]:
    source = config.get("source") if isinstance(config.get("source"), Mapping) else {}
    if source.get("schema") != "solarlab.solarscale_import_config":
        return []
    metadata = source.get("material_metadata") if isinstance(source.get("material_metadata"), Mapping) else {}
    mapped = source.get("mapped_parameters") if isinstance(source.get("mapped_parameters"), Mapping) else {}
    absorber = _config_absorber_layer(config)
    reasons: list[str] = []
    provenance = (
        source.get("band_alignment_provenance")
        or metadata.get("band_alignment_provenance")
    )
    if not provenance:
        reasons.append(
            "Production band-alignment policy blocked: absorber electron affinity provenance is missing"
        )
    if "absorber.chi" not in mapped and metadata.get("electron_affinity_ev") is None:
        reasons.append(
            "Production band-alignment policy blocked: electron_affinity_ev was not mapped to absorber chi"
        )
    if absorber is None or _optional_float(absorber.get("chi")) is None:
        reasons.append(
            "Production band-alignment policy blocked: absorber chi is missing"
        )
    return reasons


def _config_absorber_layer(config: Mapping[str, Any]) -> Mapping[str, Any] | None:
    layers = config.get("layers")
    if not isinstance(layers, list):
        return None
    for layer in layers:
        if isinstance(layer, Mapping) and layer.get("role") == "absorber":
            return layer
    return None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _profile_settings(profile: ReportProfile) -> dict[str, dict[str, Any]]:
    if profile == "smoke":
        return _smoke_settings()
    if profile == "production":
        return _production_settings()
    return _diagnostic_settings()


def _smoke_settings() -> dict[str, dict[str, Any]]:
    return {
        "jv": {"N_grid": 12, "n_points": 4, "v_rate": 5.0, "V_max": 0.2},
        "suns_voc": {"suns_levels": (0.5, 1.0, 2.0), "N_grid": 20, "t_settle": 5.0e-4},
        "degradation": {
            "t_end": 5.0,
            "n_snapshots": 2,
            "V_bias": 0.9,
            "N_grid": 20,
            "dt_max": 1.0,
            "metric_n_points": 4,
            "metric_settle_time": 1.0e-3,
        },
        "eqe": {
            "wavelengths_nm": np.array([450.0, 650.0, 850.0]),
            "N_grid": 20,
            "t_settle": 5.0e-4,
        },
    }


def _diagnostic_settings() -> dict[str, dict[str, Any]]:
    settings = _smoke_settings()
    settings["jv"] = {"N_grid": 30, "n_points": 8, "v_rate": 5.0, "V_max": 1.4}
    settings["suns_voc"] = {"suns_levels": (0.1, 1.0, 3.0, 10.0), "N_grid": 30, "t_settle": 5.0e-4}
    settings["degradation"] = {
        "t_end": 30.0,
        "n_snapshots": 4,
        "V_bias": 0.9,
        "N_grid": 20,
        "dt_max": 1.0,
        "metric_n_points": 6,
        "metric_settle_time": 1.0e-3,
    }
    settings["eqe"] = {
        "wavelengths_nm": np.linspace(350.0, 950.0, 7),
        "N_grid": 30,
        "t_settle": 5.0e-4,
    }
    return settings


def _production_settings() -> dict[str, dict[str, Any]]:
    settings = _diagnostic_settings()
    settings["jv"] = {"N_grid": 60, "n_points": 16, "v_rate": 0.5, "V_max": 1.4}
    settings["suns_voc"] = {"suns_levels": (0.1, 0.5, 1.0, 3.0, 10.0), "N_grid": 50, "t_settle": 1.0e-2}
    settings["degradation"] = {
        "t_end": 100.0,
        "n_snapshots": 5,
        "V_bias": 0.9,
        "N_grid": 40,
        "dt_max": 1.0,
        "metric_n_points": 10,
        "metric_settle_time": 2.0e-3,
    }
    settings["eqe"] = {
        "wavelengths_nm": np.linspace(350.0, 950.0, 13),
        "N_grid": 40,
        "t_settle": 2.0e-3,
    }
    return settings


def _run_experiment(name: str, status: dict[str, Any], fn):
    try:
        result = fn()
    except Exception as exc:  # noqa: BLE001 - report packs must capture failures.
        status[name] = {
            "status": "skipped" if name == "eqe" else "failed",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }
        return None
    status[name] = {"status": "completed"}
    return result


def _plot_jv(path: Path, result: Any) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(np.asarray(result.V_fwd), _j_to_mA_cm2(result.J_fwd), marker="o", label="forward")
    ax.plot(np.asarray(result.V_rev), _j_to_mA_cm2(result.J_rev), marker="s", label="reverse")
    ax.axhline(0.0, color="0.25", linewidth=0.8)
    ax.set_xlabel("Voltage (V)")
    ax.set_ylabel("Current density (mA cm$^{-2}$)")
    ax.set_title("Quick JV sweep")
    ax.legend()
    ax.grid(True, alpha=0.25)
    _savefig(fig, path)


def _plot_suns_voc(path: Path, result: Any) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.semilogx(np.asarray(result.suns), np.asarray(result.V_oc), marker="o")
    ax.set_xlabel("Illumination (suns)")
    ax.set_ylabel("$V_{oc}$ (V)")
    ax.set_title("Suns-Voc")
    ax.grid(True, which="both", alpha=0.25)
    _savefig(fig, path)


def _plot_pseudo_jv(path: Path, result: Any) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(np.asarray(result.J_pseudo_V), _j_to_mA_cm2(result.J_pseudo_J), marker="o")
    ax.axhline(0.0, color="0.25", linewidth=0.8)
    ax.set_xlabel("Voltage (V)")
    ax.set_ylabel("Pseudo current density (mA cm$^{-2}$)")
    ax.set_title(f"Pseudo-JV, pseudo-FF={_fmt_float(result.pseudo_FF)}")
    ax.grid(True, alpha=0.25)
    _savefig(fig, path)


def _plot_degradation(path: Path, result: Any) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    t = np.asarray(result.t)
    pce = np.asarray(result.PCE)
    ax.plot(t, pce, marker="o", label="PCE")
    ax.set_xlabel("Stress time (s)")
    ax.set_ylabel("PCE")
    ax.set_title("Quick degradation proxy")
    ax.grid(True, alpha=0.25)
    ax.legend()
    _savefig(fig, path)


def _plot_eqe(path: Path, result: Any) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(np.asarray(result.wavelengths_nm), np.asarray(result.EQE), marker="o")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("EQE")
    ax.set_ylim(bottom=0.0)
    ax.set_title("External quantum efficiency")
    ax.grid(True, alpha=0.25)
    _savefig(fig, path)


def _plot_screening_gates(path: Path, record: Mapping[str, Any]) -> None:
    gates = (record.get("screening") or {}).get("gates", {}) or {}
    names = list(gates) or ["no_gates"]
    statuses = [_gate_status(gates.get(name)) for name in names] if gates else ["missing"]
    colors = {
        "pass": "#2b8a3e",
        "fail": "#c92a2a",
        "missing": "#868e96",
        "unknown": "#1c7ed6",
    }
    fig, ax = plt.subplots(figsize=(max(6.4, len(names) * 1.15), 4.2))
    ax.bar(range(len(names)), [1] * len(names), color=[colors.get(status, colors["unknown"]) for status in statuses])
    ax.set_xticks(range(len(names)), names, rotation=30, ha="right")
    ax.set_yticks([])
    ax.set_title("SolarScale screening gates")
    for idx, status in enumerate(statuses):
        ax.text(idx, 0.5, status, ha="center", va="center", color="white", fontsize=9, fontweight="bold")
    _savefig(fig, path)


def _record_figure_quality(
    figure_quality: dict[str, dict[str, Any]],
    name: str,
    path: Path,
    profile: ReportProfile,
    *,
    status: str | None = None,
) -> None:
    quality_status = status or FIGURE_QUALITY_BY_PROFILE[profile]
    figure_quality[name] = {
        "path": str(path),
        "profile": profile,
        "quality_status": quality_status,
    }


def _production_jv_blocking_reasons(result: Any) -> list[str]:
    reasons: list[str] = []
    for label, metrics in (("forward", result.metrics_fwd), ("reverse", result.metrics_rev)):
        if not bool(getattr(metrics, "voc_bracketed", False)):
            reasons.append(f"{label} JV did not bracket Voc")
        for field in ("V_oc", "J_sc", "FF", "PCE"):
            value = getattr(metrics, field)
            if value is None or not math.isfinite(float(value)):
                reasons.append(f"{label} JV metric {field} is not finite")
        if float(getattr(metrics, "V_oc")) <= 0.0:
            reasons.append(f"{label} JV metric V_oc is sentinel or non-positive")
        if float(getattr(metrics, "PCE")) <= 0.0:
            reasons.append(f"{label} JV metric PCE is sentinel or non-positive")
    return reasons


def _physics_quality_status(profile: ReportProfile, blocking_reasons: list[str]) -> str:
    if blocking_reasons:
        return "publication_blocked"
    return FIGURE_QUALITY_BY_PROFILE[profile]


def _plot_error(path: Path, title: str, message: str) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.axis("off")
    ax.text(0.5, 0.65, title, ha="center", va="center", fontsize=14, fontweight="bold")
    ax.text(0.5, 0.42, message[:240], ha="center", va="center", wrap=True, fontsize=10)
    _savefig(fig, path)


def _savefig(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _j_to_mA_cm2(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=float) * 0.1


def _gate_status(value: Any) -> str:
    if isinstance(value, bool):
        return "pass" if value else "fail"
    if isinstance(value, str):
        lower = value.strip().lower()
        return lower if lower in {"pass", "fail", "missing"} else "unknown"
    if isinstance(value, Mapping):
        status = value.get("status")
        if isinstance(status, str):
            lower = status.strip().lower()
            return lower if lower in {"pass", "fail", "missing"} else "unknown"
        if "passed" in value:
            return "pass" if bool(value["passed"]) else "fail"
    if value is None:
        return "missing"
    return "unknown"


def _jv_metrics_dict(result: Any) -> dict[str, Any]:
    return {
        "forward": _jsonable_dataclass(result.metrics_fwd),
        "reverse": _jsonable_dataclass(result.metrics_rev),
        "hysteresis_index": _json_number(result.hysteresis_index),
    }


def _jsonable_dataclass(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _jsonable_dataclass(item) for key, item in asdict(value).items()}
    if isinstance(value, np.ndarray):
        return [_json_number(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_jsonable_dataclass(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _jsonable_dataclass(item) for key, item in value.items()}
    return _json_number(value)


def _json_number(value: Any) -> Any:
    if isinstance(value, (np.floating, float)):
        out = float(value)
        return out if math.isfinite(out) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _fmt_float(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return f"{number:.3g}" if math.isfinite(number) else "n/a"


def _render_report_md(
    *,
    material_id: str,
    config_path: Path,
    material_record_path: Path,
    device_results_path: Path,
    dft_summary: Mapping[str, Any],
    device_metrics: Mapping[str, Any],
    figure_paths: Mapping[str, str],
    quality_summary: Mapping[str, Any],
) -> str:
    jv = device_metrics.get("JV_metrics") or {}
    forward = jv.get("forward") if isinstance(jv, Mapping) else {}
    rows = [
        f"# SolarScale x SolarLab Evaluation: {material_id}",
        "",
        "## Inputs",
        f"- Config: `{config_path}`",
        f"- Material record: `{material_record_path}`",
        f"- Device results: `{device_results_path}`",
        "",
        "## DFT/MD Parameters",
        "| Parameter | Value | Unit | Provenance | SolarLab target | Used |",
        "| --- | ---: | --- | --- | --- | --- |",
    ]
    for row in dft_summary.get("parameters", []) or []:
        rows.append(
            "| {parameter} | {value} | {unit} | {provenance_kind} | {target} | {used} |".format(
                parameter=row.get("parameter"),
                value=_fmt_float(row.get("value")) if isinstance(row.get("value"), (int, float)) else row.get("value"),
                unit=row.get("unit") or "",
                provenance_kind=row.get("provenance_kind") or "",
                target=row.get("target") or "metadata only",
                used="yes" if row.get("used_by_solarlab") else "no",
            )
        )
    rows.extend(
        [
            "",
            "## Device Metrics",
            f"- Status: `{device_metrics.get('simulation_status')}`",
            f"- Voc: `{_fmt_float(_dict_get(forward, 'V_oc'))}` V",
            f"- Jsc: `{_fmt_float(_dict_get(forward, 'J_sc'))}` A/m^2",
            f"- FF: `{_fmt_float(_dict_get(forward, 'FF'))}`",
            f"- PCE: `{_fmt_float(_dict_get(forward, 'PCE'))}`",
            "",
            "## Quality",
            f"- Profile: `{quality_summary.get('report_profile')}`",
            f"- Physics quality status: `{quality_summary.get('physics_quality_status')}`",
            f"- Publication ready: `{quality_summary.get('publication_ready')}`",
            "",
            "## Figures",
        ]
    )
    for name, path in figure_paths.items():
        rows.append(f"- {name}: `{path}`")
    rows.extend(["", "## Notes"])
    warnings = list(device_metrics.get("warnings", []) or [])
    warnings.extend(quality_summary.get("blocking_reasons", []) or [])
    if warnings:
        rows.extend(f"- {warning}" for warning in warnings)
    else:
        rows.append("- No device-result warnings were reported.")
    rows.append("- Smoke and diagnostic figures are not publication-grade device evidence.")
    return "\n".join(rows) + "\n"


def _dict_get(value: Any, key: str) -> Any:
    return value.get(key) if isinstance(value, Mapping) else None
