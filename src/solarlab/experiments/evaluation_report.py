"""Material evaluation report pack for SolarScale -> SolarLab 1D workflows.

Ports the SolarScale material evaluation report pack onto ``solarlab``'s native
1D device compilation and steady-state solver stack without legacy
``perovskite_sim`` dependencies or silent physics fallbacks.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import csv
from dataclasses import asdict, dataclass, is_dataclass
import json
import math
from pathlib import Path
from typing import Any, Literal, TypeVar

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq
import yaml

from solarlab.device.compile_1d import CapabilityError
from solarlab.device.defaults import DefaultCatalog
from solarlab.equations.dc_1d import State1D
from solarlab.experiments.solarscale import (
    advance_electronic_steady_1d,
    compile_solarscale_electronic_1d,
    manifest_sha256,
    prepare_solarscale_device,
    solve_solarscale_jv_1d,
)
from solarlab.materials.resources import ResourceLibrary
from solarlab.solvers.nonlinear.steady_1d import solve_steady_state

__all__ = [
    "DFT_PARAMETER_TARGETS",
    "FIGURE_QUALITY_BY_PROFILE",
    "ReportProfile",
    "ReportQualityError",
    "SolarScaleSunsVocResult",
    "run_material_evaluation_report",
    "run_solarscale_suns_voc_1d",
]

_T = TypeVar("_T")

DFT_PARAMETER_TARGETS: dict[str, str | None] = {
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
    """Raised when a production report would look physical but fails quality gates."""


@dataclass(frozen=True)
class SolarScaleSunsVocResult:
    """Native 1D steady Suns-V_oc and pseudo-J-V screening result."""

    suns: NDArray[np.float64]
    V_oc: NDArray[np.float64]
    J_sc: NDArray[np.float64]
    J_pseudo_V: NDArray[np.float64]
    J_pseudo_J: NDArray[np.float64]
    pseudo_FF: float
    ideality_factor: float
    compiled_identity: str
    scientific_qualification: bool = False


def _compute_pseudo_ff(
    v_arr: NDArray[np.float64],
    j_arr: NDArray[np.float64],
    v_oc_ref: float,
    j_sc_ref: float,
) -> float:
    if v_oc_ref <= 0.0 or j_sc_ref <= 0.0:
        return float("nan")
    mask = (v_arr > 0.0) & (v_arr < v_oc_ref) & (j_arr > 0.0) & (j_arr < j_sc_ref)
    v_q = np.concatenate(([0.0], v_arr[mask], [v_oc_ref]))
    j_q = np.concatenate(([j_sc_ref], j_arr[mask], [0.0]))
    order = np.argsort(v_q)
    v_s = v_q[order]
    j_s = j_q[order]
    v_grid: NDArray[np.float64] = np.linspace(0.0, v_oc_ref, 500, dtype=np.float64)
    j_grid = np.interp(v_grid, v_s, j_s)
    p_max = float(np.max(v_grid * j_grid))
    return p_max / (v_oc_ref * j_sc_ref)


def run_solarscale_suns_voc_1d(
    config_or_path: str | Path | Mapping[str, Any],
    *,
    suns_levels: Sequence[float] = (0.1, 1.0, 10.0),
    N_grid: int = 20,
    t_settle: float = 5.0e-4,
    defaults: DefaultCatalog | None = None,
    resources: ResourceLibrary | None = None,
) -> SolarScaleSunsVocResult:
    """Execute a native 1D steady Suns-V_oc and pseudo-J-V sweep."""
    _ = t_settle
    suns_sorted = np.array(sorted({float(s) for s in suns_levels}), dtype=np.float64)
    if suns_sorted.size == 0:
        raise ValueError("suns_levels must be non-empty")
    if np.any(~np.isfinite(suns_sorted)) or np.any(suns_sorted <= 0.0):
        raise ValueError("suns_levels must be finite and positive")

    _, device = compile_solarscale_electronic_1d(
        config_or_path, N_grid=int(N_grid), defaults=defaults, resources=resources
    )
    polarity = float(device.voltage_polarity)
    v_T = float(device.thermal_voltage_V)
    eg_ceiling = float(np.min(device.parameters.band_structure.physical_band_gap_eV))
    v_limit = max(1.6, eg_ceiling + 0.35)

    from solarlab.solvers.nonlinear.steady_1d import SteadyOptions

    steady_opts = SteadyOptions(max_iterations=250, max_bias_step_V=0.025)
    dark_sol = solve_steady_state(
        device, voltage_V=0.0, generation_scale=0.0, options=steady_opts
    )
    j_dark = -polarity * float(dark_sol.current_density_A_m2[0])

    v_oc_arr = np.zeros_like(suns_sorted)
    j_sc_arr = np.zeros_like(suns_sorted)

    for k, x_sun in enumerate(suns_sorted):
        gen_scale = float(x_sun)
        sc_sol = solve_steady_state(
            device, voltage_V=0.0, generation_scale=gen_scale, options=steady_opts
        )
        j_zero = -polarity * float(sc_sol.current_density_A_m2[0])
        j_sc_arr[k] = j_zero - j_dark
        if j_zero <= 0.0:
            v_oc_arr[k] = 0.0
            continue

        v_lo = 0.0
        j_lo = j_zero
        seed_state: State1D = sc_sol.state
        v_hi = 0.0
        j_hi = j_lo
        step_v = 0.025
        bracketed = False
        while v_hi < v_limit:
            v_try = min(v_hi + step_v, v_limit)
            sol_try = advance_electronic_steady_1d(
                device,
                seed_state,
                v_try,
                generation_scale=gen_scale,
                options=steady_opts,
            )
            j_try = -polarity * float(sol_try.current_density_A_m2[0])
            if j_try <= 0.0:
                v_hi = v_try
                j_hi = j_try
                bracketed = True
                break
            v_lo = v_try
            j_lo = j_try
            seed_state = sol_try.state
            v_hi = v_try

        if not bracketed:
            raise RuntimeError(
                f"Suns-Voc failed to bracket V_oc up to {v_limit:.3f} V at {gen_scale:g} suns"
            )
        if j_hi == 0.0:
            v_oc_arr[k] = v_hi
            continue

        cached_seed: dict[str, State1D] = {"state": seed_state}

        def _j_at_v(v_eval: float) -> float:
            sol_v = solve_steady_state(
                device,
                initial_state=cached_seed["state"],
                voltage_V=float(v_eval),
                generation_scale=gen_scale,
                options=steady_opts,
            )
            cached_seed["state"] = sol_v.state
            return -polarity * float(sol_v.current_density_A_m2[0])

        root_v = float(brentq(_j_at_v, v_lo, v_hi, xtol=1.0e-7, maxiter=40))
        v_oc_arr[k] = root_v

    if suns_sorted.size >= 2 and np.all(v_oc_arr > 0.0) and v_T > 0.0:
        slope = float(np.polyfit(np.log(suns_sorted), v_oc_arr, 1)[0])
        ideality = slope / v_T
    else:
        ideality = float("nan")

    idx_ref = int(np.argmin(np.abs(suns_sorted - 1.0)))
    v_oc_ref = float(v_oc_arr[idx_ref])
    j_sc_ref = float(j_sc_arr[idx_ref])
    j_pseudo_v = v_oc_arr.copy()
    j_pseudo_j = j_sc_ref - j_sc_arr
    pseudo_ff = _compute_pseudo_ff(j_pseudo_v, j_pseudo_j, v_oc_ref, j_sc_ref)

    return SolarScaleSunsVocResult(
        suns=suns_sorted,
        V_oc=v_oc_arr,
        J_sc=j_sc_arr,
        J_pseudo_V=j_pseudo_v,
        J_pseudo_J=j_pseudo_j,
        pseudo_FF=pseudo_ff,
        ideality_factor=ideality,
        compiled_identity=device.identity,
        scientific_qualification=False,
    )


def _run_degradation_1d(
    config_path: Path,
    *,
    defaults: DefaultCatalog | None = None,
    resources: ResourceLibrary | None = None,
    **kwargs: Any,
) -> Any:
    _ = kwargs
    prepared, _ = prepare_solarscale_device(config_path, defaults=defaults, resources=resources)
    elec_layers = [layer for layer in prepared.layers if layer.role != "substrate"]
    has_ions = any(
        float(dict(layer.parameters)["P0"] or 0.0) > 0.0
        or float(dict(layer.parameters)["P0_neg"] or 0.0) > 0.0
        for layer in elec_layers
    )
    if not has_ions:
        raise CapabilityError(
            "ion_migration_degradation_requires_mobile_ions: P0 == 0 on all electrical layers"
        )
    raise CapabilityError(
        "long_horizon_degradation_unbound_in_screening_report: use explicit ionic_transient workflows"
    )


def _run_eqe_1d(
    config_path: Path,
    *,
    defaults: DefaultCatalog | None = None,
    resources: ResourceLibrary | None = None,
    **kwargs: Any,
) -> Any:
    _ = kwargs
    # Validate config resolution first, then refuse unbound wavelength-resolved EQE without n/k resources.
    prepare_solarscale_device(config_path, defaults=defaults, resources=resources)
    raise CapabilityError(
        "wavelength_resolved_eqe_requires_material_nk_resources: optical n/k tables are unavailable"
    )


def run_material_evaluation_report(
    *,
    config_path: str | Path,
    material_record_path: str | Path,
    device_results_path: str | Path,
    out_dir: str | Path,
    quick: bool = False,
    profile: ReportProfile = "diagnostic",
    material_id: str | None = None,
    defaults: DefaultCatalog | None = None,
    resources: ResourceLibrary | None = None,
    settings_override: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Create a deterministic material evaluation report pack for one SolarScale config."""
    config_file = Path(config_path)
    record_file = Path(material_record_path)
    device_results_file = Path(device_results_path)
    out = Path(out_dir)
    figures = out / "figures"
    out.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    config = _load_yaml(config_file)
    resolved_profile = _resolve_profile(profile=profile, quick=quick)
    if resolved_profile == "production":
        _validate_production_config(config)

    resolved_material_id = (
        material_id or _config_material_id(config) or _first_record_id(record_file)
    )
    record = _select_material_record(_load_json(record_file), resolved_material_id)
    device_results = _load_json(device_results_file)
    device_record = _select_device_result(device_results, resolved_material_id, config_file)

    dft_summary = _dft_parameter_summary(record, config, device_record)
    dft_json = out / "dft_parameter_summary.json"
    dft_csv = out / "dft_parameter_summary.csv"
    dft_json.write_text(json.dumps(dft_summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_dft_summary_csv(dft_csv, dft_summary)

    device_metrics = _device_metrics_summary(device_results, device_record, resolved_material_id)
    settings = _profile_settings(resolved_profile)
    if settings_override is not None:
        for exp_key, exp_vals in settings_override.items():
            if exp_key in settings:
                settings[exp_key] = {**settings[exp_key], **dict(exp_vals)}
            else:
                settings[exp_key] = dict(exp_vals)

    figure_paths: dict[str, str] = {}
    figure_quality: dict[str, dict[str, Any]] = {}
    experiment_status: dict[str, Any] = {}
    blocking_reasons: list[str] = []
    optical_blocking_reasons: list[str] = []
    if resolved_profile == "production":
        optical_blocking_reasons = _production_optical_blocking_reasons(config)
        blocking_reasons.extend(optical_blocking_reasons)
        blocking_reasons.extend(_production_band_alignment_blocking_reasons(config))
    else:
        blocking_reasons.append(f"report profile {resolved_profile!r} is not publication-grade")

    jv_result = _run_experiment(
        "jv",
        experiment_status,
        lambda: solve_solarscale_jv_1d(
            config_file,
            defaults=defaults,
            resources=resources,
            **settings["jv"],
        ),
    )
    if jv_result is not None:
        if resolved_profile == "production":
            jv_reasons = _production_jv_blocking_reasons(jv_result)
            if jv_reasons:
                failure_path = out / "evaluation_failure.json"
                failure_path.write_text(
                    json.dumps(
                        {
                            "schema": "solarlab.material_evaluation_failure",
                            "schema_version": "0.1",
                            "material_id": resolved_material_id,
                            "profile": resolved_profile,
                            "blocking_reasons": jv_reasons,
                        },
                        indent=2,
                        sort_keys=True,
                    ),
                    encoding="utf-8",
                )
                raise ReportQualityError(
                    "Production JV quality gate failed: " + "; ".join(jv_reasons)
                )
        jv_fig = figures / "jv_curve.png"
        _plot_jv(jv_fig, jv_result)
        figure_paths["jv_curve"] = str(jv_fig)
        _record_figure_quality(figure_quality, "jv_curve", jv_fig, resolved_profile)
        device_metrics["quick_experiments"]["jv"] = _jv_metrics_dict(jv_result)
    else:
        if resolved_profile == "production":
            failure_path = out / "evaluation_failure.json"
            err_msg = str(experiment_status["jv"]["error_message"])
            failure_path.write_text(
                json.dumps(
                    {
                        "schema": "solarlab.material_evaluation_failure",
                        "schema_version": "0.1",
                        "material_id": resolved_material_id,
                        "profile": resolved_profile,
                        "blocking_reasons": [f"Production JV solver failed: {err_msg}"],
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            raise ReportQualityError("Production JV solver failed: " + err_msg)
        jv_fig = figures / "jv_curve.png"
        _plot_error(jv_fig, "JV sweep failed", str(experiment_status["jv"]["error_message"]))
        figure_paths["jv_curve"] = str(jv_fig)
        _record_figure_quality(
            figure_quality, "jv_curve", jv_fig, resolved_profile, status="publication_blocked"
        )

    suns_result = _run_experiment(
        "suns_voc",
        experiment_status,
        lambda: run_solarscale_suns_voc_1d(
            config_file,
            defaults=defaults,
            resources=resources,
            **settings["suns_voc"],
        ),
    )
    if suns_result is not None:
        suns_path = figures / "suns_voc.png"
        pseudo_path = figures / "pseudo_jv.png"
        _plot_suns_voc(suns_path, suns_result)
        _plot_pseudo_jv(pseudo_path, suns_result)
        figure_paths["suns_voc"] = str(suns_path)
        figure_paths["pseudo_jv"] = str(pseudo_path)
        _record_figure_quality(figure_quality, "suns_voc", suns_path, resolved_profile)
        _record_figure_quality(figure_quality, "pseudo_jv", pseudo_path, resolved_profile)
        device_metrics["quick_experiments"]["suns_voc"] = _jsonable_dataclass(suns_result)
    else:
        for name, title in (("suns_voc", "Suns-Voc failed"), ("pseudo_jv", "Pseudo-JV failed")):
            err_fig = figures / f"{name}.png"
            _plot_error(err_fig, title, str(experiment_status["suns_voc"]["error_message"]))
            figure_paths[name] = str(err_fig)
            _record_figure_quality(
                figure_quality, name, err_fig, resolved_profile, status="publication_blocked"
            )

    degradation_result = _run_experiment(
        "degradation",
        experiment_status,
        lambda: _run_degradation_1d(
            config_file,
            defaults=defaults,
            resources=resources,
            **settings["degradation"],
        ),
    )
    if degradation_result is not None:
        deg_fig = figures / "degradation.png"
        _plot_degradation(deg_fig, degradation_result)
        figure_paths["degradation"] = str(deg_fig)
        _record_figure_quality(figure_quality, "degradation", deg_fig, resolved_profile)
        device_metrics["quick_experiments"]["degradation"] = _jsonable_dataclass(degradation_result)
    else:
        deg_fig = figures / "degradation.png"
        _plot_error(
            deg_fig, "Degradation failed", str(experiment_status["degradation"]["error_message"])
        )
        figure_paths["degradation"] = str(deg_fig)
        _record_figure_quality(
            figure_quality, "degradation", deg_fig, resolved_profile, status="publication_blocked"
        )

    eqe_result = _run_experiment(
        "eqe",
        experiment_status,
        lambda: _run_eqe_1d(
            config_file,
            defaults=defaults,
            resources=resources,
            **settings["eqe"],
        ),
    )
    if eqe_result is not None:
        eqe_fig = figures / "eqe.png"
        _plot_eqe(eqe_fig, eqe_result)
        figure_paths["eqe"] = str(eqe_fig)
        _record_figure_quality(
            figure_quality,
            "eqe",
            eqe_fig,
            resolved_profile,
            status="publication_blocked" if optical_blocking_reasons else None,
        )
        device_metrics["quick_experiments"]["eqe"] = _jsonable_dataclass(eqe_result)
    else:
        if resolved_profile == "production":
            blocking_reasons.append(
                "EQE skipped because material-specific optical n/k data are unavailable or incomplete"
            )
        skip_path = figures / "eqe_skip_reason.json"
        skip_payload = {
            "schema": "solarlab.eqe_skip_reason",
            "schema_version": "0.1",
            "material_id": resolved_material_id,
            "profile": resolved_profile,
            "quality_status": (
                "publication_blocked"
                if resolved_profile == "production"
                else FIGURE_QUALITY_BY_PROFILE[resolved_profile]
            ),
            "reason": experiment_status["eqe"]["error_message"],
            "policy": "EQE is skipped when wavelength-resolved optical n/k data are unavailable or incomplete.",
        }
        skip_path.write_text(json.dumps(skip_payload, indent=2, sort_keys=True), encoding="utf-8")
        figure_paths["eqe_skip_reason"] = str(skip_path)
        _record_figure_quality(
            figure_quality,
            "eqe_skip_reason",
            skip_path,
            resolved_profile,
            status="publication_blocked" if resolved_profile == "production" else None,
        )

    gates_path = figures / "screening_gates.png"
    _plot_screening_gates(gates_path, record)
    figure_paths["screening_gates"] = str(gates_path)
    _record_figure_quality(figure_quality, "screening_gates", gates_path, resolved_profile)

    physics_quality_status = _physics_quality_status(resolved_profile, blocking_reasons)
    publication_ready = resolved_profile == "production" and not blocking_reasons
    quality_summary: dict[str, Any] = {
        "report_profile": resolved_profile,
        "physics_quality_status": physics_quality_status,
        "publication_ready": publication_ready,
        "blocking_reasons": blocking_reasons,
        "figure_quality": figure_quality,
        "scientific_qualification": False,
    }
    device_metrics["experiment_status"] = experiment_status
    device_metrics.update(quality_summary)
    device_metrics_json = out / "device_metrics.json"
    device_metrics_json.write_text(
        json.dumps(device_metrics, indent=2, sort_keys=True), encoding="utf-8"
    )

    report_path = out / "report.md"
    report_path.write_text(
        _render_report_md(
            material_id=resolved_material_id,
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

    manifest: dict[str, Any] = {
        "schema": "solarlab.material_evaluation_report",
        "schema_version": "0.1",
        "material_id": resolved_material_id,
        "config_path": str(config_file),
        "material_record_path": str(record_file),
        "device_results_path": str(device_results_file),
        "out_dir": str(out),
        "report_profile": resolved_profile,
        "physics_quality_status": physics_quality_status,
        "publication_ready": publication_ready,
        "scientific_qualification": False,
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
    manifest["manifest_sha256"] = manifest_sha256(manifest)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def _load_json(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return loaded


def _load_yaml(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return loaded


def _config_material_id(config: Mapping[str, Any]) -> str | None:
    source = config.get("source")
    if isinstance(source, Mapping) and source.get("material_id"):
        return str(source["material_id"])
    return None


def _first_record_id(path: Path) -> str:
    payload = _load_json(path)
    records = payload.get("records")
    if isinstance(records, list) and records and isinstance(records[0], Mapping):
        if records[0].get("material_id"):
            return str(records[0]["material_id"])
    if payload.get("material_id"):
        return str(payload["material_id"])
    raise ValueError(f"No material_id found in {path}")


def _select_material_record(payload: Mapping[str, Any], material_id: str) -> dict[str, Any]:
    if payload.get("material_id") == material_id:
        return dict(payload)
    for record in payload.get("records", []) or []:
        if isinstance(record, Mapping) and str(record.get("material_id")) == material_id:
            return dict(record)
    raise ValueError(f"Material {material_id!r} not found in material record payload")


def _select_device_result(
    payload: Mapping[str, Any], material_id: str, config_path: Path
) -> dict[str, Any]:
    candidates = payload.get("records", []) or []
    config_resolved = str(config_path)
    for record in candidates:
        if (
            isinstance(record, Mapping)
            and str(record.get("material_id")) == material_id
            and str(record.get("config_path")) == config_resolved
        ):
            return dict(record)
    for record in candidates:
        if isinstance(record, Mapping) and str(record.get("material_id")) == material_id:
            return dict(record)
    return {}


def _dft_parameter_summary(
    record: Mapping[str, Any],
    config: Mapping[str, Any],
    device_record: Mapping[str, Any],
) -> dict[str, Any]:
    source = config.get("source") if isinstance(config.get("source"), Mapping) else {}
    assert isinstance(source, Mapping)
    mapped = dict(source.get("mapped_parameters", {}) or device_record.get("mapped_parameters", {}) or {})
    metadata = dict(
        source.get("material_metadata", {}) or device_record.get("material_metadata", {}) or {}
    )
    properties = record.get("properties", {}) or {}
    rows: list[dict[str, Any]] = []
    for name, target in DFT_PARAMETER_TARGETS.items():
        prop = properties.get(name) if isinstance(properties, Mapping) else None
        if isinstance(prop, Mapping):
            value = prop.get("value")
            unit = prop.get("unit", "")
            prov_raw = prop.get("provenance")
            provenance = prov_raw if isinstance(prov_raw, Mapping) else {}
        else:
            value = metadata.get(name)
            unit = ""
            provenance = {}
        rows.append(
            {
                "parameter": name,
                "value": value,
                "unit": unit,
                "provenance_kind": provenance.get(
                    "kind", "metadata" if name in metadata else "missing"
                ),
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
    assert isinstance(source, Mapping)
    if not source.get("activate_bandgap"):
        raise ReportQualityError(
            "Production report requires activate_bandgap=True in the generated SolarLab config source"
        )
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
        raise ReportQualityError(
            "Production report requires absorber optical_material for EQE/TMM provenance"
        )


def _production_optical_blocking_reasons(config: Mapping[str, Any]) -> list[str]:
    source = config.get("source") if isinstance(config.get("source"), Mapping) else {}
    assert isinstance(source, Mapping)
    if source.get("schema") != "solarlab.solarscale_import_config":
        return []
    absorber = _config_absorber_layer(config)
    optical_material = str(absorber.get("optical_material") or "").strip() if absorber else ""
    meta_raw = source.get("material_metadata")
    metadata = meta_raw if isinstance(meta_raw, Mapping) else {}
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
    assert isinstance(source, Mapping)
    if source.get("schema") != "solarlab.solarscale_import_config":
        return []
    meta_raw = source.get("material_metadata")
    metadata = meta_raw if isinstance(meta_raw, Mapping) else {}
    map_raw = source.get("mapped_parameters")
    mapped = map_raw if isinstance(map_raw, Mapping) else {}
    absorber = _config_absorber_layer(config)
    reasons: list[str] = []
    provenance = source.get("band_alignment_provenance") or metadata.get(
        "band_alignment_provenance"
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
        reasons.append("Production band-alignment policy blocked: absorber chi is missing")
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
            "wavelengths_nm": [450.0, 650.0, 850.0],
            "N_grid": 20,
            "t_settle": 5.0e-4,
        },
    }


def _diagnostic_settings() -> dict[str, dict[str, Any]]:
    settings = _smoke_settings()
    settings["jv"] = {"N_grid": 30, "n_points": 8, "v_rate": 5.0, "V_max": 1.4}
    settings["suns_voc"] = {
        "suns_levels": (0.1, 1.0, 3.0, 10.0),
        "N_grid": 30,
        "t_settle": 5.0e-4,
    }
    return settings


def _production_settings() -> dict[str, dict[str, Any]]:
    settings = _diagnostic_settings()
    settings["jv"] = {"N_grid": 60, "n_points": 16, "v_rate": 0.5, "V_max": 1.4}
    settings["suns_voc"] = {
        "suns_levels": (0.1, 0.5, 1.0, 3.0, 10.0),
        "N_grid": 50,
        "t_settle": 1.0e-2,
    }
    return settings


def _run_experiment(
    name: str, status: dict[str, Any], fn: Callable[[], _T]
) -> _T | None:
    try:
        result = fn()
    except Exception as exc:
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
    ax.plot(np.asarray(result.t), np.asarray(result.PCE), marker="o", label="PCE")
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
    scr_raw = record.get("screening")
    scr = scr_raw if isinstance(scr_raw, Mapping) else {}
    gates_raw = scr.get("gates")
    gates = gates_raw if isinstance(gates_raw, Mapping) else {}
    names = list(gates) or ["no_gates"]
    statuses = [_gate_status(gates.get(name)) for name in names] if gates else ["missing"]
    colors = {
        "pass": "#2b8a3e",
        "fail": "#c92a2a",
        "missing": "#868e96",
        "unknown": "#1c7ed6",
    }
    fig, ax = plt.subplots(figsize=(max(6.4, len(names) * 1.15), 4.2))
    ax.bar(
        range(len(names)),
        [1] * len(names),
        color=[colors.get(status, colors["unknown"]) for status in statuses],
    )
    ax.set_xticks(range(len(names)), names, rotation=30, ha="right")
    ax.set_yticks([])
    ax.set_title("SolarScale screening gates")
    for idx, status in enumerate(statuses):
        ax.text(
            idx,
            0.5,
            status,
            ha="center",
            va="center",
            color="white",
            fontsize=9,
            fontweight="bold",
        )
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
        for field_name in ("V_oc", "J_sc", "FF", "PCE"):
            value = getattr(metrics, field_name)
            if value is None or not math.isfinite(float(value)):
                reasons.append(f"{label} JV metric {field_name} is not finite")
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


def _savefig(fig: Any, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _j_to_mA_cm2(values: Any) -> NDArray[np.float64]:
    return np.asarray(values, dtype=np.float64) * 0.1


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
    if is_dataclass(value) and not isinstance(value, type):
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
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
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
        if not isinstance(row, Mapping):
            continue
        raw_val = row.get("value")
        val_str = _fmt_float(raw_val) if isinstance(raw_val, (int, float)) else str(raw_val)
        rows.append(
            "| {parameter} | {value} | {unit} | {provenance_kind} | {target} | {used} |".format(
                parameter=row.get("parameter"),
                value=val_str,
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
