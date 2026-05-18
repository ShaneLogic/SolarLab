"""Material evaluation report pack for SolarScale -> SolarLab workflows."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping

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


def run_material_evaluation_report(
    *,
    config_path: str | Path,
    material_record_path: str | Path,
    device_results_path: str | Path,
    out_dir: str | Path,
    quick: bool = False,
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
    settings = _quick_settings() if quick else _default_settings()

    figure_paths: dict[str, str] = {}
    experiment_status: dict[str, Any] = {}
    jv_result = _run_experiment(
        "jv",
        experiment_status,
        lambda: run_jv_sweep(stack, **settings["jv"]),
    )
    if jv_result is not None:
        path = figures / "jv_curve.png"
        _plot_jv(path, jv_result)
        figure_paths["jv_curve"] = str(path)
        device_metrics["quick_experiments"]["jv"] = _jv_metrics_dict(jv_result)
    else:
        path = figures / "jv_curve.png"
        _plot_error(path, "JV sweep failed", experiment_status["jv"]["error_message"])
        figure_paths["jv_curve"] = str(path)

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
        device_metrics["quick_experiments"]["suns_voc"] = _jsonable_dataclass(suns_result)
    else:
        for name, title in (("suns_voc", "Suns-Voc failed"), ("pseudo_jv", "Pseudo-JV failed")):
            path = figures / f"{name}.png"
            _plot_error(path, title, experiment_status["suns_voc"]["error_message"])
            figure_paths[name] = str(path)

    degradation_result = _run_experiment(
        "degradation",
        experiment_status,
        lambda: run_degradation(stack, **settings["degradation"]),
    )
    if degradation_result is not None:
        path = figures / "degradation.png"
        _plot_degradation(path, degradation_result)
        figure_paths["degradation"] = str(path)
        device_metrics["quick_experiments"]["degradation"] = _jsonable_dataclass(degradation_result)
    else:
        path = figures / "degradation.png"
        _plot_error(path, "Degradation failed", experiment_status["degradation"]["error_message"])
        figure_paths["degradation"] = str(path)

    eqe_result = _run_experiment(
        "eqe",
        experiment_status,
        lambda: compute_eqe(stack, **settings["eqe"]),
    )
    if eqe_result is not None:
        path = figures / "eqe.png"
        _plot_eqe(path, eqe_result)
        figure_paths["eqe"] = str(path)
        device_metrics["quick_experiments"]["eqe"] = _jsonable_dataclass(eqe_result)
    else:
        skip_path = figures / "eqe_skip_reason.json"
        skip_payload = {
            "schema": "solarlab.eqe_skip_reason",
            "schema_version": "0.1",
            "material_id": material_id,
            "reason": experiment_status["eqe"]["error_message"],
            "policy": "EQE is skipped when wavelength-resolved optical n/k data are unavailable or incomplete.",
        }
        skip_path.write_text(json.dumps(skip_payload, indent=2, sort_keys=True), encoding="utf-8")
        figure_paths["eqe_skip_reason"] = str(skip_path)

    gates_path = figures / "screening_gates.png"
    _plot_screening_gates(gates_path, record)
    figure_paths["screening_gates"] = str(gates_path)

    device_metrics["experiment_status"] = experiment_status
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
        "report_path": str(report_path),
        "dft_parameter_summary_json": str(dft_json),
        "dft_parameter_summary_csv": str(dft_csv),
        "device_metrics_json": str(device_metrics_json),
        "figures": figure_paths,
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


def _quick_settings() -> dict[str, dict[str, Any]]:
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


def _default_settings() -> dict[str, dict[str, Any]]:
    settings = _quick_settings()
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
            "## Figures",
        ]
    )
    for name, path in figure_paths.items():
        rows.append(f"- {name}: `{path}`")
    rows.extend(["", "## Notes"])
    warnings = list(device_metrics.get("warnings", []) or [])
    if warnings:
        rows.extend(f"- {warning}" for warning in warnings)
    else:
        rows.append("- No device-result warnings were reported.")
    rows.append("- Quick figures are workflow validation outputs unless rerun with production settings.")
    return "\n".join(rows) + "\n"


def _dict_get(value: Any, key: str) -> Any:
    return value.get(key) if isinstance(value, Mapping) else None
