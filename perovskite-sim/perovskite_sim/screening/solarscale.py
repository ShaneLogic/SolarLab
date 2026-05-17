"""SolarScale MaterialRecord import policy for SolarLab screening.

The bridge is deliberately thin: it reads normalized SolarScale records,
selects device-ready candidates, maps only provenance-backed DFT/MD absorber
properties into a SolarLab template, and leaves device-specific unknowns as
sweep dimensions or diagnostics.
"""

from __future__ import annotations

import copy
import csv
import itertools
import json
import re
import subprocess
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Mapping

import yaml

try:  # numpy is already a runtime dependency, but keep import local-friendly.
    import numpy as np
except Exception:  # pragma: no cover - defensive only
    np = None

ImportPolicy = Literal["production", "exploratory"]
SweepPolicyName = Literal["quick", "exploratory", "production"]
SWEEP_DIMENSION_ORDER = (
    "absorber.thickness",
    "absorber.tau_n",
    "absorber.tau_p",
    "absorber.trap_N_t_bulk",
    "device.interfaces",
)


@dataclass(frozen=True)
class PropertyProvenance:
    kind: str = "missing"
    source: str = "not_available"
    notes: str = ""


@dataclass(frozen=True)
class MaterialProperty:
    value: Any
    unit: str = ""
    provenance: PropertyProvenance = field(default_factory=PropertyProvenance)


@dataclass(frozen=True)
class MaterialRecord:
    material_id: str
    formula: str | None = None
    schema_version: str | None = None
    properties: dict[str, MaterialProperty] = field(default_factory=dict)
    screening: dict[str, Any] = field(default_factory=dict)
    stages: dict[str, Any] = field(default_factory=dict)
    source: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class CandidatePlan:
    material_id: str
    rank: int
    selected: bool
    reason: str
    readiness: str
    ranking_score: float | None
    ranking_score_source: str | None
    ml_pv_score: float | None
    final_fom_score: float | None
    screening_evidence: dict[str, Any]
    mapped_parameters: dict[str, Any]
    material_metadata: dict[str, Any]
    sweep_parameters: dict[str, Any]
    missing_required: list[str]
    missing_optional: list[str]
    assumed_parameters: dict[str, Any]
    diagnostics: list[str]
    config: dict[str, Any] | None = None
    config_path: str | None = None


SWEEP_POLICIES: dict[SweepPolicyName, dict[str, list[float]]] = {
    "quick": {
        "absorber.thickness": [300e-9],
        "absorber.tau_n": [1e-7],
        "absorber.tau_p": [1e-7],
        "absorber.trap_N_t_bulk": [1e22],
        "device.interfaces": [1.0],
    },
    "exploratory": {
        "absorber.thickness": [300e-9, 500e-9, 800e-9],
        "absorber.tau_n": [1e-9, 1e-7, 1e-6],
        "absorber.tau_p": [1e-9, 1e-7, 1e-6],
        "absorber.trap_N_t_bulk": [1e20, 1e22, 1e24],
        "device.interfaces": [0.0, 1.0, 10.0],
    },
    "production": {
        "absorber.thickness": [400e-9, 600e-9],
        "absorber.tau_n": [1e-8, 1e-7],
        "absorber.tau_p": [1e-8, 1e-7],
        "absorber.trap_N_t_bulk": [1e21, 1e22],
        "device.interfaces": [0.1, 1.0],
    },
}
DEFAULT_SWEEP_POLICY: SweepPolicyName = "quick"
DEFAULT_SWEEP_GRID = SWEEP_POLICIES[DEFAULT_SWEEP_POLICY]

DEVICE_SWEEP_PROPERTY_MAP: dict[str, str] = {
    "absorber_thickness_m": "absorber.thickness",
    "carrier_lifetime_s": "absorber.tau_n/tau_p",
    "trap_density_m3": "absorber.trap_N_t_bulk",
    "surface_recombination_velocity_m_s": "device.interfaces",
    "contact_work_function_ev": "device.contact_work_function",
}

MAPPED_PROPERTY_SPECS: dict[str, tuple[str, str | None, float]] = {
    "dielectric_static_avg": ("absorber.eps_r", "eps_r", 1.0),
    "electron_mobility_cm2_v_s": ("absorber.mu_n", "mu_n", 1.0e-4),
    "hole_mobility_cm2_v_s": ("absorber.mu_p", "mu_p", 1.0e-4),
    "ion_diffusion_coefficient_m2_s": ("absorber.D_ion", "D_ion", 1.0),
    "ion_activation_energy_ev": ("absorber.E_a_ion", "E_a_ion", 1.0),
}

REQUIRED_PHYSICAL_INPUTS = (
    "band_gap_hse_ev",
    "dielectric_static_avg",
)

OPTIONAL_PHYSICAL_INPUTS = (
    "electron_mobility_cm2_v_s",
    "hole_mobility_cm2_v_s",
    "ion_diffusion_coefficient_m2_s",
    "ion_activation_energy_ev",
)

SCORE_FIELDS = ("final_fom_score", "ml_pv_score")

_ALLOWED_READINESS: dict[ImportPolicy, set[str]] = {
    "production": {"promising"},
    "exploratory": {"phonon", "promising"},
}

_READY_FLAG_BY_POLICY: dict[ImportPolicy, str] = {
    "production": "solarlab_production_ready",
    "exploratory": "solarlab_provisional_ready",
}


def load_material_records(path: str | Path) -> list[dict[str, Any]]:
    """Load raw SolarScale MaterialRecord dictionaries from JSON."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError("MaterialRecord payload must contain a records list")
    return records


def parse_material_records(path: str | Path) -> list[MaterialRecord]:
    """Load and normalize SolarScale MaterialRecord JSON."""

    return [_parse_record(record) for record in load_material_records(path)]


def plan_solarlab_import(
    records_path: str | Path,
    *,
    template_path: str | Path,
    import_policy: ImportPolicy = "exploratory",
    limit: int | None = None,
    sweep_grid: Mapping[str, list[float]] | None = None,
    sweep_policy: SweepPolicyName = DEFAULT_SWEEP_POLICY,
    include_configs: bool = True,
    activate_bandgap: bool = False,
) -> dict[str, Any]:
    """Return a dry-run import manifest without writing files."""

    if import_policy not in _ALLOWED_READINESS:
        raise ValueError(f"Unknown import_policy {import_policy!r}")

    records = parse_material_records(records_path)
    template = _load_yaml(template_path)
    if activate_bandgap:
        _validate_band_aligned_template(template)
    sweeps = _resolve_sweep_grid(sweep_policy=sweep_policy, sweep_grid=sweep_grid)

    candidates = sorted(
        (
            _candidate_plan(
                record,
                template,
                sweeps,
                import_policy,
                activate_bandgap=activate_bandgap,
                sweep_policy=sweep_policy,
            )
            for record in records
        ),
        key=_candidate_sort_key,
    )
    selected_count = 0
    limited_candidates: list[CandidatePlan] = []
    for rank, candidate in enumerate(candidates, start=1):
        candidate = _replace_candidate(candidate, rank=rank)
        if candidate.selected:
            if limit is not None and selected_count >= limit:
                candidate = _replace_candidate(
                    candidate,
                    selected=False,
                    reason="limit reached",
                    config=None,
                )
            else:
                selected_count += 1
        if not include_configs:
            candidate = _replace_candidate(candidate, config=None)
        limited_candidates.append(candidate)

    selected = [candidate for candidate in limited_candidates if candidate.selected]
    skipped = [candidate for candidate in limited_candidates if not candidate.selected]
    return {
        "schema": "solarlab.solarscale_import_plan",
        "schema_version": "0.4",
        "records_path": str(records_path),
        "template_path": str(template_path),
        "import_policy": import_policy,
        "allowed_readiness": sorted(_ALLOWED_READINESS[import_policy]),
        "activate_bandgap": activate_bandgap,
        "sweep_policy": sweep_policy,
        "sweep_grid": sweeps,
        "sweep_dimensions": _sweep_dimensions(sweeps),
        "dry_run": True,
        "selected_count": len(selected),
        "skipped_count": len(skipped),
        "summary": _screening_summary(limited_candidates),
        "selected": [_candidate_to_dict(candidate) for candidate in selected],
        "skipped": [_candidate_to_dict(candidate, include_config=False) for candidate in skipped],
    }


def generate_solarlab_inputs(
    records_path: str | Path,
    *,
    template_path: str | Path,
    out_dir: str | Path,
    limit: int | None = None,
    sweep_grid: Mapping[str, list[float]] | None = None,
    sweep_policy: SweepPolicyName = DEFAULT_SWEEP_POLICY,
    import_policy: ImportPolicy = "production",
    activate_bandgap: bool = False,
) -> dict[str, Any]:
    """Generate SolarLab YAML configs and a manifest from SolarScale records."""

    plan = plan_solarlab_import(
        records_path,
        template_path=template_path,
        import_policy=import_policy,
        limit=limit,
        sweep_grid=sweep_grid,
        sweep_policy=sweep_policy,
        include_configs=True,
        activate_bandgap=activate_bandgap,
    )
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    generated: list[dict[str, Any]] = []
    for item in plan["selected"]:
        config = item.pop("config")
        config_path = out / f"{_safe_name(item['material_id'])}.yaml"
        _write_yaml(config_path, config)
        item["config_path"] = str(config_path)
        generated.append(item)

    manifest = {
        "schema": "solarlab.solarscale_import_manifest",
        "schema_version": "0.4",
        "records_path": plan["records_path"],
        "template_path": plan["template_path"],
        "out_dir": str(out),
        "import_policy": import_policy,
        "allowed_readiness": plan["allowed_readiness"],
        "activate_bandgap": activate_bandgap,
        "sweep_policy": sweep_policy,
        "sweep_grid": plan["sweep_grid"],
        "sweep_dimensions": plan["sweep_dimensions"],
        "summary": plan["summary"],
        "generated": generated,
        "skipped": plan["skipped"],
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def expand_sweep_manifest(
    manifest: Mapping[str, Any],
    *,
    out_dir: str | Path | None = None,
    max_points: int | None = None,
) -> dict[str, Any]:
    """Expand selected candidates into one YAML config per sweep-grid point."""

    if max_points is not None and max_points < 0:
        raise ValueError("max_points must be non-negative")
    sweeps = _manifest_sweep_grid(manifest)
    root = Path(out_dir or manifest.get("out_dir") or ".")
    root.mkdir(parents=True, exist_ok=True)
    config_dir = root / "sweep_configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    generated: list[dict[str, Any]] = []
    stopped_by_limit = False
    for candidate in manifest.get("generated", []) or []:
        parent_config_path = candidate.get("config_path")
        if not parent_config_path:
            raise ValueError(f"Generated candidate {candidate.get('material_id')!r} is missing config_path")
        parent_config = _load_yaml(parent_config_path)
        material_name = _safe_name(str(candidate.get("material_id", "material")))
        material_dir = config_dir / material_name
        material_dir.mkdir(parents=True, exist_ok=True)
        for candidate_index, sweep_values in enumerate(_iter_sweep_values(sweeps), start=1):
            if max_points is not None and len(generated) >= max_points:
                stopped_by_limit = True
                break
            sweep_point_id = f"{material_name}__sweep_{candidate_index:04d}"
            config = copy.deepcopy(parent_config)
            _apply_sweep_values(
                config,
                sweep_values,
                sweep_point_id=sweep_point_id,
                candidate_sweep_index=candidate_index,
                parent_config_path=str(parent_config_path),
            )
            config_path = material_dir / f"{sweep_point_id}.yaml"
            _write_yaml(config_path, config)
            sweep_item = copy.deepcopy(dict(candidate))
            sweep_item.update(
                {
                    "parent_config_path": str(parent_config_path),
                    "config_path": str(config_path),
                    "sweep_point_id": sweep_point_id,
                    "sweep_index": candidate_index,
                    "global_sweep_index": len(generated) + 1,
                    "sweep_values": dict(sweep_values),
                    "sweep_policy": manifest.get("sweep_policy"),
                }
            )
            generated.append(sweep_item)
        if stopped_by_limit:
            break

    per_candidate = _sweep_dimensions(sweeps)
    total_requested = per_candidate["total_points"] * len(manifest.get("generated", []) or [])
    sweep_manifest = {
        "schema": "solarlab.solarscale_sweep_manifest",
        "schema_version": "0.1",
        "parent_schema": manifest.get("schema"),
        "parent_schema_version": manifest.get("schema_version"),
        "records_path": manifest.get("records_path"),
        "template_path": manifest.get("template_path"),
        "out_dir": str(root),
        "sweep_config_dir": str(config_dir),
        "import_policy": manifest.get("import_policy"),
        "allowed_readiness": manifest.get("allowed_readiness", []),
        "activate_bandgap": manifest.get("activate_bandgap"),
        "sweep_policy": manifest.get("sweep_policy"),
        "sweep_grid": sweeps,
        "sweep_dimensions": {
            **per_candidate,
            "generated_configs_per_candidate": per_candidate["total_points"],
            "candidate_count": len(manifest.get("generated", []) or []),
            "total_requested_points": total_requested,
            "total_generated_points": len(generated),
            "max_points": max_points,
            "truncated": stopped_by_limit,
        },
        "summary": manifest.get("summary", {}),
        "generated": generated,
        "skipped": manifest.get("skipped", []),
    }
    (root / "sweep_plan.json").write_text(
        json.dumps(sweep_manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return sweep_manifest


def run_smoke_jv(
    config_path: str | Path,
    *,
    N_grid: int = 12,
    n_points: int = 4,
    v_rate: float = 5.0,
    V_max: float = 0.2,
) -> dict[str, Any]:
    """Run a tiny JV smoke check for one generated config."""

    from perovskite_sim.experiments.jv_sweep import run_jv_sweep
    from perovskite_sim.models.config_loader import load_device_from_yaml

    stack = load_device_from_yaml(str(config_path))
    result = run_jv_sweep(
        stack,
        N_grid=N_grid,
        n_points=n_points,
        v_rate=v_rate,
        V_max=V_max,
    )
    return {
        "config_path": str(config_path),
        "N_grid": N_grid,
        "n_points": n_points,
        "v_rate": v_rate,
        "V_max": V_max,
        "metrics_fwd": {
            "V_oc": result.metrics_fwd.V_oc,
            "J_sc": result.metrics_fwd.J_sc,
            "FF": result.metrics_fwd.FF,
            "PCE": result.metrics_fwd.PCE,
            "voc_bracketed": result.metrics_fwd.voc_bracketed,
        },
        "metrics_rev": {
            "V_oc": result.metrics_rev.V_oc,
            "J_sc": result.metrics_rev.J_sc,
            "FF": result.metrics_rev.FF,
            "PCE": result.metrics_rev.PCE,
            "voc_bracketed": result.metrics_rev.voc_bracketed,
        },
        "hysteresis_index": result.hysteresis_index,
    }


def run_smoke_device_results(
    manifest: Mapping[str, Any],
    *,
    N_grid: int = 12,
    n_points: int = 4,
    v_rate: float = 5.0,
    V_max: float = 0.2,
    max_configs: int | None = 1,
    result_type: str = "smoke_jv",
) -> dict[str, Any]:
    """Run tiny JV smoke checks and return SolarScale-ingestible device results."""

    generated = list(manifest.get("generated", []) or [])
    if max_configs is not None:
        generated = generated[:max_configs]
    timestamp = _utc_timestamp()
    git_commit = _git_commit(Path(__file__).resolve())
    records = [
        _device_result_record(
            item,
            manifest,
            timestamp=timestamp,
            git_commit=git_commit,
            smoke_settings={
                "N_grid": N_grid,
                "n_points": n_points,
                "v_rate": v_rate,
                "V_max": V_max,
            },
        )
        for item in generated
    ]
    return {
        "schema": "solarlab.device_results",
        "schema_version": "0.2",
        "result_type": result_type,
        "timestamp": timestamp,
        "git_commit": git_commit,
        "records_path": manifest.get("records_path"),
        "template_path": manifest.get("template_path"),
        "import_policy": manifest.get("import_policy"),
        "activate_bandgap": manifest.get("activate_bandgap"),
        "sweep_policy": manifest.get("sweep_policy"),
        "sweep_dimensions": manifest.get("sweep_dimensions", {}),
        "jv_settings": {
            "N_grid": N_grid,
            "n_points": n_points,
            "v_rate": v_rate,
            "V_max": V_max,
            "max_configs": max_configs,
        },
        "smoke_settings": {
            "N_grid": N_grid,
            "n_points": n_points,
            "v_rate": v_rate,
            "V_max": V_max,
            "max_configs": max_configs,
        },
        "records": records,
        "summary": _device_results_summary(records),
    }


def write_device_results(
    results: Mapping[str, Any],
    *,
    json_path: str | Path,
    csv_path: str | Path | None = None,
) -> None:
    """Write device results as JSON plus an optional flat CSV table."""

    json_out = Path(json_path)
    json_out.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    if csv_path is None:
        return
    csv_out = Path(csv_path)
    rows = [_device_result_csv_row(record) for record in results.get("records", []) or []]
    fieldnames = [
        "material_id",
        "config_path",
        "template_path",
        "import_policy",
        "activate_bandgap",
        "sweep_policy",
        "sweep_point_id",
        "sweep_index",
        "global_sweep_index",
        "sweep_values_json",
        "parent_config_path",
        "simulation_status",
        "Voc",
        "Jsc",
        "FF",
        "PCE",
        "hysteresis_index",
        "voc_bracketed_fwd",
        "voc_bracketed_rev",
        "error_type",
        "error_message",
        "git_commit",
        "timestamp",
    ]
    with csv_out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _device_result_record(
    item: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    timestamp: str,
    git_commit: str | None,
    smoke_settings: Mapping[str, Any],
) -> dict[str, Any]:
    warnings = list(item.get("diagnostics", []) or [])
    config_path = item.get("config_path")
    if not config_path:
        return _failed_device_result_record(
            item,
            manifest,
            timestamp=timestamp,
            git_commit=git_commit,
            smoke_settings=smoke_settings,
            error=ValueError("generated manifest item is missing config_path"),
            warnings=warnings,
        )
    try:
        smoke = run_smoke_jv(
            config_path,
            N_grid=int(smoke_settings["N_grid"]),
            n_points=int(smoke_settings["n_points"]),
            v_rate=float(smoke_settings["v_rate"]),
            V_max=float(smoke_settings["V_max"]),
        )
    except Exception as exc:  # noqa: BLE001 - result schema must capture failures.
        return _failed_device_result_record(
            item,
            manifest,
            timestamp=timestamp,
            git_commit=git_commit,
            smoke_settings=smoke_settings,
            error=exc,
            warnings=warnings,
        )

    metrics_fwd = dict(smoke.get("metrics_fwd", {}) or {})
    metrics_rev = dict(smoke.get("metrics_rev", {}) or {})
    if metrics_fwd.get("voc_bracketed") is False:
        warnings.append("forward JV did not bracket Voc; increase smoke V_max for physical metrics")
    if metrics_rev.get("voc_bracketed") is False:
        warnings.append("reverse JV did not bracket Voc; increase smoke V_max for physical metrics")
    return {
        **_device_result_base(
            item,
            manifest,
            timestamp=timestamp,
            git_commit=git_commit,
            smoke_settings=smoke_settings,
            warnings=warnings,
        ),
        "simulation_status": "completed",
        "JV_metrics": {
            "forward": metrics_fwd,
            "reverse": metrics_rev,
            "hysteresis_index": smoke.get("hysteresis_index"),
        },
        "smoke_result": smoke,
        "error": None,
    }


def _failed_device_result_record(
    item: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    timestamp: str,
    git_commit: str | None,
    smoke_settings: Mapping[str, Any],
    error: Exception,
    warnings: list[str],
) -> dict[str, Any]:
    return {
        **_device_result_base(
            item,
            manifest,
            timestamp=timestamp,
            git_commit=git_commit,
            smoke_settings=smoke_settings,
            warnings=warnings,
        ),
        "simulation_status": "failed",
        "JV_metrics": {
            "forward": None,
            "reverse": None,
            "hysteresis_index": None,
        },
        "smoke_result": None,
        "error": {
            "error_type": type(error).__name__,
            "error_message": str(error),
        },
    }


def _device_result_base(
    item: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    timestamp: str,
    git_commit: str | None,
    smoke_settings: Mapping[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    return {
        "material_id": item.get("material_id"),
        "config_path": item.get("config_path"),
        "template_path": manifest.get("template_path"),
        "records_path": manifest.get("records_path"),
        "import_policy": manifest.get("import_policy"),
        "activate_bandgap": manifest.get("activate_bandgap"),
        "sweep_policy": manifest.get("sweep_policy") or item.get("sweep_policy"),
        "sweep_point_id": item.get("sweep_point_id"),
        "sweep_index": item.get("sweep_index"),
        "global_sweep_index": item.get("global_sweep_index"),
        "parent_config_path": item.get("parent_config_path"),
        "sweep_values": item.get("sweep_values", {}),
        "mapped_parameters": item.get("mapped_parameters", {}),
        "material_metadata": item.get("material_metadata", {}),
        "screening_evidence": item.get("screening_evidence", {}),
        "ranking": {
            "ranking_score": item.get("ranking_score"),
            "ranking_score_source": item.get("ranking_score_source"),
            "ml_pv_score": item.get("ml_pv_score"),
            "final_fom_score": item.get("final_fom_score"),
        },
        "missing_required": item.get("missing_required", []),
        "missing_optional": item.get("missing_optional", []),
        "sweep_parameters": item.get("sweep_parameters", {}),
        "warnings": warnings,
        "jv_settings": dict(smoke_settings),
        "smoke_settings": dict(smoke_settings),
        "git_commit": git_commit,
        "timestamp": timestamp,
    }


def _device_results_summary(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    status_counts = Counter(str(record.get("simulation_status", "unknown")) for record in records)
    completed = [record for record in records if record.get("simulation_status") == "completed"]
    failed = [record for record in records if record.get("simulation_status") == "failed"]
    best = sorted(
        completed,
        key=lambda record: _metric_value(record, "PCE") or float("-inf"),
        reverse=True,
    )
    return {
        "total_records": len(records),
        "status_counts": dict(sorted(status_counts.items())),
        "failed_materials": [record.get("material_id") for record in failed],
        "best_by_pce": [
            {
                "material_id": record.get("material_id"),
                "sweep_point_id": record.get("sweep_point_id"),
                "sweep_index": record.get("sweep_index"),
                "PCE": _metric_value(record, "PCE"),
                "Voc": _metric_value(record, "V_oc"),
                "Jsc": _metric_value(record, "J_sc"),
                "FF": _metric_value(record, "FF"),
            }
            for record in best[:10]
        ],
    }


def _metric_value(record: Mapping[str, Any], field: str) -> float | None:
    metrics = record.get("JV_metrics", {})
    if not isinstance(metrics, Mapping):
        return None
    forward = metrics.get("forward")
    if not isinstance(forward, Mapping):
        return None
    value = forward.get(field)
    return None if value is None else float(value)


def _device_result_csv_row(record: Mapping[str, Any]) -> dict[str, Any]:
    metrics = record.get("JV_metrics", {})
    forward = metrics.get("forward") if isinstance(metrics, Mapping) else None
    reverse = metrics.get("reverse") if isinstance(metrics, Mapping) else None
    error = record.get("error") or {}
    return {
        "material_id": record.get("material_id"),
        "config_path": record.get("config_path"),
        "template_path": record.get("template_path"),
        "import_policy": record.get("import_policy"),
        "activate_bandgap": record.get("activate_bandgap"),
        "sweep_policy": record.get("sweep_policy"),
        "sweep_point_id": record.get("sweep_point_id"),
        "sweep_index": record.get("sweep_index"),
        "global_sweep_index": record.get("global_sweep_index"),
        "sweep_values_json": json.dumps(record.get("sweep_values", {}) or {}, sort_keys=True),
        "parent_config_path": record.get("parent_config_path"),
        "simulation_status": record.get("simulation_status"),
        "Voc": _dict_get(forward, "V_oc"),
        "Jsc": _dict_get(forward, "J_sc"),
        "FF": _dict_get(forward, "FF"),
        "PCE": _dict_get(forward, "PCE"),
        "hysteresis_index": _dict_get(metrics, "hysteresis_index"),
        "voc_bracketed_fwd": _dict_get(forward, "voc_bracketed"),
        "voc_bracketed_rev": _dict_get(reverse, "voc_bracketed"),
        "error_type": error.get("error_type"),
        "error_message": error.get("error_message"),
        "git_commit": record.get("git_commit"),
        "timestamp": record.get("timestamp"),
    }


def _dict_get(value: Any, key: str) -> Any:
    if not isinstance(value, Mapping):
        return None
    return value.get(key)


def _utc_timestamp() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _git_commit(anchor: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=anchor if anchor.is_dir() else anchor.parent,
            check=True,
            text=True,
            capture_output=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def _parse_record(record: Mapping[str, Any]) -> MaterialRecord:
    material_id = str(record.get("material_id", "")).strip()
    properties_raw = record.get("properties", {})
    properties: dict[str, MaterialProperty] = {}
    if isinstance(properties_raw, Mapping):
        for name, prop in properties_raw.items():
            properties[str(name)] = _parse_property(prop)
    notes = record.get("notes", [])
    return MaterialRecord(
        material_id=material_id,
        formula=record.get("formula"),
        schema_version=record.get("schema_version"),
        properties=properties,
        screening=dict(record.get("screening", {}) or {}),
        stages=dict(record.get("stages", {}) or {}),
        source=dict(record.get("source", {}) or {}),
        notes=[str(note) for note in notes] if isinstance(notes, list) else [str(notes)],
    )


def _parse_property(prop: Any) -> MaterialProperty:
    if not isinstance(prop, Mapping):
        return MaterialProperty(value=prop)
    provenance = prop.get("provenance", {})
    if not isinstance(provenance, Mapping):
        provenance = {}
    return MaterialProperty(
        value=prop.get("value"),
        unit=str(prop.get("unit", "") or ""),
        provenance=PropertyProvenance(
            kind=str(provenance.get("kind", "missing") or "missing"),
            source=str(provenance.get("source", "not_available") or "not_available"),
            notes=str(provenance.get("notes", "") or ""),
        ),
    )


def _resolve_sweep_grid(
    *,
    sweep_policy: SweepPolicyName,
    sweep_grid: Mapping[str, list[float]] | None,
) -> dict[str, list[float]]:
    if sweep_policy not in SWEEP_POLICIES:
        raise ValueError(
            f"Unknown sweep_policy {sweep_policy!r}; expected one of "
            + ", ".join(sorted(SWEEP_POLICIES))
        )
    grid = {name: list(values) for name, values in SWEEP_POLICIES[sweep_policy].items()}
    if sweep_grid:
        for name, values in sweep_grid.items():
            grid[str(name)] = list(values)
    _validate_sweep_grid(grid)
    return grid


def _validate_sweep_grid(grid: Mapping[str, list[float]]) -> None:
    required = {"absorber.thickness", "absorber.tau_n", "absorber.tau_p", "absorber.trap_N_t_bulk", "device.interfaces"}
    missing = sorted(required - set(grid))
    if missing:
        raise ValueError("sweep grid is missing required dimensions: " + ", ".join(missing))
    empty = [name for name, values in grid.items() if not values]
    if empty:
        raise ValueError("sweep grid dimensions must not be empty: " + ", ".join(sorted(empty)))


def _sweep_dimensions(sweeps: Mapping[str, list[float]]) -> dict[str, Any]:
    sizes = {name: len(values) for name, values in sweeps.items()}
    total = 1
    for size in sizes.values():
        total *= size
    return {
        "dimensions": sizes,
        "total_points": total,
        "generated_configs_per_candidate": 1,
        "note": "Sweep-grid size per selected candidate. Baseline imports generate one config; expand_sweep_manifest generates the full matrix or a capped subset.",
    }


def _candidate_plan(
    record: MaterialRecord,
    template: Mapping[str, Any],
    sweeps: Mapping[str, list[float]],
    import_policy: ImportPolicy,
    *,
    activate_bandgap: bool,
    sweep_policy: SweepPolicyName,
) -> CandidatePlan:
    material_id = record.material_id
    if not material_id:
        return CandidatePlan(
            material_id="",
            rank=0,
            selected=False,
            reason="missing material_id",
            readiness="",
            ranking_score=None,
            ranking_score_source=None,
            ml_pv_score=None,
            final_fom_score=None,
            screening_evidence={
                "readiness": None,
                "resolved_readiness": "",
                "gates": {},
                "thresholds": {},
                "raw_screening": {},
            },
            mapped_parameters={},
            material_metadata={},
            sweep_parameters={},
            missing_required=list(REQUIRED_PHYSICAL_INPUTS),
            missing_optional=list(OPTIONAL_PHYSICAL_INPUTS),
            assumed_parameters={},
            diagnostics=[],
        )

    readiness = str(record.screening.get("readiness", "") or _property_value(record, "screening_readiness") or "")
    ready, reason = _is_device_ready(record, import_policy=import_policy)
    mapped, missing_required, missing_optional, diagnostics = _mapped_parameters(
        record,
        activate_bandgap=activate_bandgap,
    )
    material_metadata = _material_metadata(record)
    sweep_parameters = _sweep_parameters(record, sweeps)
    assumed = _template_assumptions(template, mapped, sweeps)
    ranking_score, ranking_source = _ranking_score(record)
    ml_score = _numeric_property(record, "ml_pv_score")
    fom_score = _numeric_property(record, "final_fom_score")
    screening_evidence = _screening_evidence(record, resolved_readiness=readiness)
    selected = ready and not missing_required
    if ready and missing_required:
        reason = "missing required SolarLab physical inputs: " + ", ".join(missing_required)

    config = None
    if selected:
        config = _build_config(
            template,
            record,
            mapped,
            sweeps,
            missing_optional,
            diagnostics,
            import_policy=import_policy,
            activate_bandgap=activate_bandgap,
            sweep_policy=sweep_policy,
        )

    return CandidatePlan(
        material_id=material_id,
        rank=0,
        selected=selected,
        reason=reason,
        readiness=readiness,
        ranking_score=ranking_score,
        ranking_score_source=ranking_source,
        ml_pv_score=ml_score,
        final_fom_score=fom_score,
        screening_evidence=screening_evidence,
        mapped_parameters=mapped,
        material_metadata=material_metadata,
        sweep_parameters=sweep_parameters,
        missing_required=missing_required,
        missing_optional=missing_optional,
        assumed_parameters=assumed,
        diagnostics=diagnostics,
        config=config,
    )


def _is_device_ready(record: MaterialRecord, *, import_policy: ImportPolicy) -> tuple[bool, str]:
    if import_policy not in _ALLOWED_READINESS:
        raise ValueError(f"Unknown import_policy {import_policy!r}")

    flag_name = _READY_FLAG_BY_POLICY[import_policy]
    flag_value = _property_value(record, flag_name)
    if flag_value is None:
        flag_value = record.screening.get(
            "production_solarlab_ready" if import_policy == "production" else "provisional_solarlab_ready"
        )
    if flag_value is True:
        return True, f"{flag_name}=true"

    readiness = str(record.screening.get("readiness", "") or _property_value(record, "screening_readiness") or "")
    if readiness:
        if readiness in _ALLOWED_READINESS[import_policy]:
            return True, f"screening.readiness={readiness!r}"
        return False, f"screening.readiness={readiness!r} not allowed by {import_policy} policy"

    dft_flag = _property_value(record, "dft_result_available")
    if import_policy == "exploratory" and dft_flag is True:
        return True, "dft_result_available=true"
    return False, f"{flag_name} is not true and screening.readiness is unavailable"


def _mapped_parameters(
    record: MaterialRecord,
    *,
    activate_bandgap: bool,
) -> tuple[dict[str, Any], list[str], list[str], list[str]]:
    mapped: dict[str, Any] = {}
    missing_required: list[str] = []
    missing_optional: list[str] = []
    diagnostics: list[str] = []

    band_gap = record.properties.get("band_gap_hse_ev")
    if band_gap is None or band_gap.value is None or band_gap.provenance.kind == "missing":
        missing_required.append("band_gap_hse_ev")
    elif band_gap.provenance.kind not in {"computed", "derived"}:
        missing_required.append("band_gap_hse_ev")
        diagnostics.append(
            f"band_gap_hse_ev provenance kind {band_gap.provenance.kind!r} is not accepted as a fixed input"
        )
    elif activate_bandgap:
        mapped["absorber.Eg"] = _coerce_numeric(band_gap.value)
        diagnostics.append("band_gap_hse_ev activated as absorber Eg")
    else:
        diagnostics.append("band_gap_hse_ev kept as metadata; absorber Eg follows the template")

    for prop_name, (target, _field, scale) in MAPPED_PROPERTY_SPECS.items():
        prop = record.properties.get(prop_name)
        if prop is None or prop.value is None or prop.provenance.kind == "missing":
            if prop_name in REQUIRED_PHYSICAL_INPUTS:
                missing_required.append(prop_name)
            else:
                missing_optional.append(prop_name)
            continue
        if prop.provenance.kind == "swept":
            missing_optional.append(prop_name)
            diagnostics.append(f"{prop_name} is swept and was not mapped as a fixed absorber input")
            continue
        if prop.provenance.kind not in {"computed", "derived"}:
            if prop_name in REQUIRED_PHYSICAL_INPUTS:
                missing_required.append(prop_name)
            else:
                missing_optional.append(prop_name)
            diagnostics.append(
                f"{prop_name} provenance kind {prop.provenance.kind!r} is not accepted as a fixed input"
            )
            continue
        mapped[target] = _coerce_numeric(prop.value) * scale

    for score_field in SCORE_FIELDS:
        if score_field in record.properties:
            diagnostics.append(f"{score_field} imported as ranking metadata only")
    if _property_value(record, "slme_0p5um") is not None or _property_value(record, "slme_2um") is not None:
        diagnostics.append("SLME is preserved as ranking metadata; it is not converted to absorber alpha")
    if _property_value(record, "absorption_edge_ev") is not None:
        diagnostics.append("absorption_edge_ev is not converted to a scalar absorption coefficient")
    return mapped, missing_required, missing_optional, diagnostics


def _sweep_parameters(
    record: MaterialRecord,
    sweeps: Mapping[str, list[float]],
) -> dict[str, Any]:
    sweep_parameters: dict[str, Any] = {name: list(values) for name, values in sweeps.items()}
    for prop_name, target_name in DEVICE_SWEEP_PROPERTY_MAP.items():
        prop = record.properties.get(prop_name)
        if prop is None:
            continue
        value = prop.value
        kind = prop.provenance.kind
        if kind == "swept":
            if target_name in sweeps:
                sweep_parameters.setdefault(target_name, list(sweeps[target_name]))
        elif value is not None:
            sweep_parameters[target_name] = value
    return sweep_parameters


def _template_assumptions(
    template: Mapping[str, Any],
    mapped: Mapping[str, Any],
    sweeps: Mapping[str, list[float]],
) -> dict[str, Any]:
    layers = template.get("layers")
    if not isinstance(layers, list):
        return {}
    absorber = _absorber_layer(copy.deepcopy(layers))
    assumptions: dict[str, Any] = {}
    if "absorber.Eg" not in mapped:
        assumptions["absorber.Eg"] = absorber.get("Eg", 0.0)
    for prop_name, (_target, field_name, _scale) in MAPPED_PROPERTY_SPECS.items():
        if field_name is None or _target in mapped:
            continue
        if field_name in absorber:
            assumptions[f"absorber.{field_name}"] = absorber[field_name]
    assumptions["absorber.thickness"] = sweeps["absorber.thickness"][0]
    assumptions["absorber.tau_n"] = sweeps["absorber.tau_n"][0]
    assumptions["absorber.tau_p"] = sweeps["absorber.tau_p"][0]
    assumptions["absorber.trap_N_t_bulk"] = sweeps["absorber.trap_N_t_bulk"][0]
    assumptions["device.interfaces"] = sweeps["device.interfaces"][0]
    return assumptions


def _build_config(
    template: Mapping[str, Any],
    record: MaterialRecord,
    mapped: Mapping[str, Any],
    sweeps: Mapping[str, list[float]],
    missing_optional: list[str],
    diagnostics: list[str],
    *,
    import_policy: ImportPolicy,
    activate_bandgap: bool,
    sweep_policy: SweepPolicyName,
) -> dict[str, Any]:
    config = copy.deepcopy(dict(template))
    layers = config.get("layers")
    if not isinstance(layers, list):
        raise ValueError("Template layers must be a list")

    absorber = _absorber_layer(layers)
    material_id = record.material_id
    absorber["name"] = f"SolarScale_{material_id}"
    for target, value in mapped.items():
        field_name = target.split(".", 1)[1]
        absorber[field_name] = value

    absorber["thickness"] = sweeps["absorber.thickness"][0]
    absorber["tau_n"] = sweeps["absorber.tau_n"][0]
    absorber["tau_p"] = sweeps["absorber.tau_p"][0]
    absorber["trap_N_t_bulk"] = sweeps["absorber.trap_N_t_bulk"][0]
    absorber["trap_profile_shape"] = "exponential"

    device = config.setdefault("device", {})
    interface_s = sweeps["device.interfaces"][0]
    device["interfaces"] = [[interface_s, interface_s] for _ in range(max(0, len(layers) - 1))]
    device.setdefault("mode", "fast")

    notes = list(config.get("notes", []) or [])
    notes.append(
        "Generated from SolarScale MaterialRecord. Only computed/derived DFT/MD absorber inputs are fixed; device-only unknowns remain sweep dimensions."
    )
    if import_policy == "exploratory":
        notes.append(
            "Exploratory import may include records whose MD/ion gate is missing; do not treat those as publication-grade candidates."
        )
    if "absorber.Eg" in mapped:
        notes.append(
            "HSE band gap is explicitly activated as absorber Eg. Use only with a fully band-aligned template."
        )
    else:
        notes.append(
            "HSE band gap is preserved as metadata; absorber Eg follows the template to avoid partial band-alignment activation."
        )
    material_metadata = _material_metadata(record)
    screening_evidence = _screening_evidence(
        record,
        resolved_readiness=str(
            record.screening.get("readiness", "") or _property_value(record, "screening_readiness") or ""
        ),
    )
    config["notes"] = notes
    config["source"] = {
        "schema": "solarlab.solarscale_import_config",
        "schema_version": "0.4",
        "material_id": material_id,
        "import_policy": import_policy,
        "activate_bandgap": activate_bandgap,
        "sweep_policy": sweep_policy,
        "sweep_grid": {name: list(values) for name, values in sweeps.items()},
        "sweep_dimensions": _sweep_dimensions(sweeps),
        "screening": record.screening,
        "screening_evidence": screening_evidence,
        "ranking": {
            "ranking_score": _ranking_score(record)[0],
            "ranking_score_source": _ranking_score(record)[1],
            "ml_pv_score": _numeric_property(record, "ml_pv_score"),
            "final_fom_score": _numeric_property(record, "final_fom_score"),
        },
        "material_metadata": material_metadata,
        "mapped_parameters": dict(mapped),
        "missing_optional": list(missing_optional),
        "sweep_parameters": _sweep_parameters(record, sweeps),
        "diagnostics": list(diagnostics),
    }
    config["simulation_hints"] = {
        "N_grid": 30,
        "n_points": 8,
        "v_rate": 5.0,
        "V_max": 1.4,
        "notes": (
            "First-pass SolarScale imports should run low-resolution screening "
            "before production sweeps."
        ),
    }
    return config


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if "device" not in data or "layers" not in data:
        raise ValueError(f"Template is not a SolarLab device config: {path}")
    return data


def _write_yaml(path: Path, config: Mapping[str, Any]) -> None:
    path.write_text(
        yaml.safe_dump(dict(config), sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )


def _manifest_sweep_grid(manifest: Mapping[str, Any]) -> dict[str, list[float]]:
    raw = manifest.get("sweep_grid")
    if not isinstance(raw, Mapping):
        raise ValueError("Manifest is missing sweep_grid")
    sweeps = {str(name): [float(value) for value in values] for name, values in raw.items()}
    _validate_sweep_grid(sweeps)
    return {name: sweeps[name] for name in _sweep_dimension_names(sweeps)}


def _sweep_dimension_names(sweeps: Mapping[str, list[float]]) -> list[str]:
    ordered = [name for name in SWEEP_DIMENSION_ORDER if name in sweeps]
    ordered.extend(sorted(name for name in sweeps if name not in set(ordered)))
    return ordered


def _iter_sweep_values(sweeps: Mapping[str, list[float]]):
    names = _sweep_dimension_names(sweeps)
    for values in itertools.product(*(sweeps[name] for name in names)):
        yield dict(zip(names, values, strict=True))


def _apply_sweep_values(
    config: dict[str, Any],
    sweep_values: Mapping[str, float],
    *,
    sweep_point_id: str,
    candidate_sweep_index: int,
    parent_config_path: str,
) -> None:
    layers = config.get("layers")
    if not isinstance(layers, list):
        raise ValueError("Sweep config layers must be a list")
    absorber = _absorber_layer(layers)
    for target, value in sweep_values.items():
        if target.startswith("absorber."):
            absorber[target.split(".", 1)[1]] = float(value)
        elif target == "device.interfaces":
            device = config.setdefault("device", {})
            interface_s = float(value)
            device["interfaces"] = [[interface_s, interface_s] for _ in range(max(0, len(layers) - 1))]
        else:
            raise ValueError(f"Unsupported sweep target {target!r}")
    source = config.setdefault("source", {})
    source["sweep_point_id"] = sweep_point_id
    source["sweep_index"] = candidate_sweep_index
    source["parent_config_path"] = parent_config_path
    source["sweep_values"] = {name: float(value) for name, value in sweep_values.items()}


def _absorber_layer(layers: list[dict[str, Any]]) -> dict[str, Any]:
    absorbers = [layer for layer in layers if layer.get("role") == "absorber"]
    if len(absorbers) != 1:
        raise ValueError("Template must contain exactly one absorber layer")
    return absorbers[0]


def _ranking_score(record: MaterialRecord) -> tuple[float | None, str | None]:
    for score_name in SCORE_FIELDS:
        score = _numeric_property(record, score_name)
        if score is not None:
            return score, score_name
    return None, None


def _candidate_sort_key(candidate: CandidatePlan) -> tuple[int, float, str]:
    selected_bucket = 0 if candidate.selected else 1
    score = candidate.ranking_score if candidate.ranking_score is not None else float("-inf")
    return selected_bucket, -score, candidate.material_id


def _candidate_to_dict(candidate: CandidatePlan, *, include_config: bool = True) -> dict[str, Any]:
    data = {
        "material_id": candidate.material_id,
        "rank": candidate.rank,
        "selected": candidate.selected,
        "reason": candidate.reason,
        "readiness": candidate.readiness,
        "ranking_score": candidate.ranking_score,
        "ranking_score_source": candidate.ranking_score_source,
        "ml_pv_score": candidate.ml_pv_score,
        "final_fom_score": candidate.final_fom_score,
        "screening_evidence": candidate.screening_evidence,
        "mapped_parameters": candidate.mapped_parameters,
        "material_metadata": candidate.material_metadata,
        "dft_properties": _legacy_dft_properties(candidate.mapped_parameters, candidate.material_metadata),
        "md_properties": _legacy_md_properties(candidate.mapped_parameters),
        "sweep_parameters": candidate.sweep_parameters,
        "missing_required": candidate.missing_required,
        "missing_optional": candidate.missing_optional,
        "assumed_parameters": candidate.assumed_parameters,
        "diagnostics": candidate.diagnostics,
    }
    if candidate.config_path is not None:
        data["config_path"] = candidate.config_path
    if include_config and candidate.config is not None:
        data["config"] = candidate.config
    return data


def _replace_candidate(candidate: CandidatePlan, **changes: Any) -> CandidatePlan:
    data = {
        "material_id": candidate.material_id,
        "rank": candidate.rank,
        "selected": candidate.selected,
        "reason": candidate.reason,
        "readiness": candidate.readiness,
        "ranking_score": candidate.ranking_score,
        "ranking_score_source": candidate.ranking_score_source,
        "ml_pv_score": candidate.ml_pv_score,
        "final_fom_score": candidate.final_fom_score,
        "screening_evidence": candidate.screening_evidence,
        "mapped_parameters": candidate.mapped_parameters,
        "material_metadata": candidate.material_metadata,
        "sweep_parameters": candidate.sweep_parameters,
        "missing_required": candidate.missing_required,
        "missing_optional": candidate.missing_optional,
        "assumed_parameters": candidate.assumed_parameters,
        "diagnostics": candidate.diagnostics,
        "config": candidate.config,
        "config_path": candidate.config_path,
    }
    data.update(changes)
    return CandidatePlan(**data)


def _screening_summary(candidates: list[CandidatePlan]) -> dict[str, Any]:
    readiness_counts = Counter(candidate.readiness or "<missing>" for candidate in candidates)
    skipped_reason_counts = Counter(candidate.reason for candidate in candidates if not candidate.selected)
    selected_candidates = [candidate for candidate in candidates if candidate.selected]
    return {
        "readiness_distribution": dict(sorted(readiness_counts.items())),
        "gate_summary": _gate_summary(candidates),
        "skipped_reason_counts": dict(sorted(skipped_reason_counts.items())),
        "top_selected_candidates": [
            {
                "material_id": candidate.material_id,
                "rank": candidate.rank,
                "readiness": candidate.readiness,
                "ranking_score": candidate.ranking_score,
                "ranking_score_source": candidate.ranking_score_source,
            }
            for candidate in selected_candidates[:10]
        ],
    }


def _gate_summary(candidates: list[CandidatePlan]) -> dict[str, Any]:
    statuses = ("pass", "fail", "missing", "unknown")
    totals: Counter[str] = Counter()
    by_gate: dict[str, Counter[str]] = {}
    records_with_gate_data = 0
    for candidate in candidates:
        gates = candidate.screening_evidence.get("gates", {})
        if not isinstance(gates, Mapping) or not gates:
            continue
        records_with_gate_data += 1
        for gate_name, gate_value in gates.items():
            status = _gate_status(gate_value)
            gate_counts = by_gate.setdefault(str(gate_name), Counter())
            gate_counts[status] += 1
            totals[status] += 1
    return {
        "records_with_gate_data": records_with_gate_data,
        "totals": {status: totals.get(status, 0) for status in statuses},
        "by_gate": {
            gate_name: {status: counts.get(status, 0) for status in statuses}
            for gate_name, counts in sorted(by_gate.items())
        },
    }


def _gate_status(value: Any) -> str:
    if isinstance(value, Mapping):
        for key in ("status", "state", "outcome", "result"):
            if key in value:
                return _gate_status(value[key])
        for key in ("passed", "pass", "is_passed"):
            if key in value:
                return "pass" if bool(value[key]) else "fail"
        if value.get("missing") is True or value.get("available") is False:
            return "missing"
        return "unknown"
    if value is True:
        return "pass"
    if value is False:
        return "fail"
    if value is None:
        return "missing"
    text = str(value).strip().lower().replace("-", "_")
    if text in {"pass", "passed", "ok", "true", "yes", "accepted"}:
        return "pass"
    if text in {"fail", "failed", "false", "no", "blocked", "rejected", "reject"}:
        return "fail"
    if text in {"missing", "not_run", "not_available", "none", ""}:
        return "missing"
    if text in {"unknown", "pending", "inconclusive"}:
        return "unknown"
    return "unknown"


def _screening_evidence(record: MaterialRecord, *, resolved_readiness: str) -> dict[str, Any]:
    raw_screening = copy.deepcopy(record.screening)
    return {
        "readiness": raw_screening.get("readiness"),
        "resolved_readiness": resolved_readiness,
        "gates": copy.deepcopy(raw_screening.get("gates", {})),
        "thresholds": copy.deepcopy(raw_screening.get("thresholds", {})),
        "raw_screening": raw_screening,
    }


def _property_value(record: MaterialRecord, name: str) -> Any:
    prop = record.properties.get(name)
    if prop is None:
        return None
    return prop.value


def _material_metadata(record: MaterialRecord) -> dict[str, Any]:
    return {
        "band_gap_hse_ev": _numeric_property(record, "band_gap_hse_ev"),
        "electron_effective_mass_m0": _numeric_property(record, "electron_effective_mass_m0"),
        "hole_effective_mass_m0": _numeric_property(record, "hole_effective_mass_m0"),
        "slme_0p5um": _numeric_property(record, "slme_0p5um"),
        "slme_2um": _numeric_property(record, "slme_2um"),
        "absorption_edge_ev": _numeric_property(record, "absorption_edge_ev"),
    }


def _validate_band_aligned_template(template: Mapping[str, Any]) -> None:
    layers = template.get("layers")
    if not isinstance(layers, list):
        raise ValueError("Template layers must be a list")
    invalid: list[str] = []
    for layer in layers:
        if layer.get("role") == "substrate":
            continue
        name = str(layer.get("name", "<unnamed>"))
        chi = _optional_float(layer.get("chi"))
        eg = _optional_float(layer.get("Eg"))
        if chi is None or eg is None or eg <= 0.0:
            invalid.append(name)
    if invalid:
        raise ValueError(
            "--activate-bandgap requires a fully band-aligned template with "
            "chi and positive Eg on every electrical layer; invalid layers: "
            + ", ".join(invalid)
        )


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _legacy_dft_properties(
    mapped: Mapping[str, Any],
    material_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "band_gap_hse_ev": mapped.get("absorber.Eg", material_metadata.get("band_gap_hse_ev")),
        "dielectric_static_avg": mapped.get("absorber.eps_r"),
        "electron_mobility_cm2_v_s": _inverse_scaled(mapped.get("absorber.mu_n"), 1.0e-4),
        "hole_mobility_cm2_v_s": _inverse_scaled(mapped.get("absorber.mu_p"), 1.0e-4),
    }


def _legacy_md_properties(mapped: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "ion_diffusion_coefficient_m2_s": mapped.get("absorber.D_ion"),
        "ion_activation_energy_ev": mapped.get("absorber.E_a_ion"),
    }


def _inverse_scaled(value: Any, scale: float) -> float | None:
    if value is None:
        return None
    return float(value) / scale


def _numeric_property(record: MaterialRecord, name: str) -> float | None:
    prop = record.properties.get(name)
    if prop is None or prop.value is None or prop.provenance.kind == "missing":
        return None
    try:
        return float(prop.value)
    except (TypeError, ValueError):
        return None


def _coerce_numeric(value: Any) -> float:
    if np is not None:
        try:
            if isinstance(value, np.generic):
                value = value.item()
        except Exception:
            pass
    return float(value)


def _safe_name(material_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", material_id)
