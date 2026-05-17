"""SolarScale MaterialRecord import policy for SolarLab screening.

The bridge is deliberately thin: it reads normalized SolarScale records,
selects device-ready candidates, maps only provenance-backed DFT/MD absorber
properties into a SolarLab template, and leaves device-specific unknowns as
sweep dimensions or diagnostics.
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping

import yaml

try:  # numpy is already a runtime dependency, but keep import local-friendly.
    import numpy as np
except Exception:  # pragma: no cover - defensive only
    np = None

ImportPolicy = Literal["production", "exploratory"]


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
    mapped_parameters: dict[str, Any]
    sweep_parameters: dict[str, Any]
    missing_required: list[str]
    missing_optional: list[str]
    assumed_parameters: dict[str, Any]
    diagnostics: list[str]
    config: dict[str, Any] | None = None
    config_path: str | None = None


DEFAULT_SWEEP_GRID: dict[str, list[float]] = {
    "absorber.thickness": [300e-9, 500e-9, 800e-9],
    "absorber.tau_n": [1e-9, 1e-7, 1e-6],
    "absorber.tau_p": [1e-9, 1e-7, 1e-6],
    "absorber.trap_N_t_bulk": [1e20, 1e22, 1e24],
    "device.interfaces": [0.0, 1.0, 10.0],
}

DEVICE_SWEEP_PROPERTY_MAP: dict[str, str] = {
    "absorber_thickness_m": "absorber.thickness",
    "carrier_lifetime_s": "absorber.tau_n/tau_p",
    "trap_density_m3": "absorber.trap_N_t_bulk",
    "surface_recombination_velocity_m_s": "device.interfaces",
    "contact_work_function_ev": "device.contact_work_function",
}

MAPPED_PROPERTY_SPECS: dict[str, tuple[str, str | None, float]] = {
    "band_gap_hse_ev": ("absorber.Eg", "Eg", 1.0),
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
    include_configs: bool = True,
) -> dict[str, Any]:
    """Return a dry-run import manifest without writing files."""

    if import_policy not in _ALLOWED_READINESS:
        raise ValueError(f"Unknown import_policy {import_policy!r}")

    records = parse_material_records(records_path)
    template = _load_yaml(template_path)
    sweeps = dict(sweep_grid or DEFAULT_SWEEP_GRID)

    candidates = sorted(
        (_candidate_plan(record, template, sweeps, import_policy) for record in records),
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
        "schema_version": "0.3",
        "records_path": str(records_path),
        "template_path": str(template_path),
        "import_policy": import_policy,
        "allowed_readiness": sorted(_ALLOWED_READINESS[import_policy]),
        "dry_run": True,
        "selected_count": len(selected),
        "skipped_count": len(skipped),
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
    import_policy: ImportPolicy = "production",
) -> dict[str, Any]:
    """Generate SolarLab YAML configs and a manifest from SolarScale records."""

    plan = plan_solarlab_import(
        records_path,
        template_path=template_path,
        import_policy=import_policy,
        limit=limit,
        sweep_grid=sweep_grid,
        include_configs=True,
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
        "schema_version": "0.3",
        "records_path": plan["records_path"],
        "template_path": plan["template_path"],
        "out_dir": str(out),
        "import_policy": import_policy,
        "allowed_readiness": plan["allowed_readiness"],
        "generated": generated,
        "skipped": plan["skipped"],
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


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


def _candidate_plan(
    record: MaterialRecord,
    template: Mapping[str, Any],
    sweeps: Mapping[str, list[float]],
    import_policy: ImportPolicy,
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
            mapped_parameters={},
            sweep_parameters={},
            missing_required=list(REQUIRED_PHYSICAL_INPUTS),
            missing_optional=list(OPTIONAL_PHYSICAL_INPUTS),
            assumed_parameters={},
            diagnostics=[],
        )

    readiness = str(record.screening.get("readiness", "") or _property_value(record, "screening_readiness") or "")
    ready, reason = _is_device_ready(record, import_policy=import_policy)
    mapped, missing_required, missing_optional, diagnostics = _mapped_parameters(record)
    sweep_parameters = _sweep_parameters(record, sweeps)
    assumed = _template_assumptions(template, mapped, sweeps)
    ranking_score, ranking_source = _ranking_score(record)
    ml_score = _numeric_property(record, "ml_pv_score")
    fom_score = _numeric_property(record, "final_fom_score")
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
        mapped_parameters=mapped,
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


def _mapped_parameters(record: MaterialRecord) -> tuple[dict[str, Any], list[str], list[str], list[str]]:
    mapped: dict[str, Any] = {}
    missing_required: list[str] = []
    missing_optional: list[str] = []
    diagnostics: list[str] = []

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
            sweep_parameters.setdefault(target_name, list(sweeps.get(target_name, [])))
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
            "HSE band gap is mapped to absorber Eg. Contact-layer band alignment still comes from the template."
        )
    config["notes"] = notes
    config["source"] = {
        "schema": "solarlab.solarscale_import_config",
        "schema_version": "0.3",
        "material_id": material_id,
        "import_policy": import_policy,
        "screening": record.screening,
        "ranking": {
            "ranking_score": _ranking_score(record)[0],
            "ranking_score_source": _ranking_score(record)[1],
            "ml_pv_score": _numeric_property(record, "ml_pv_score"),
            "final_fom_score": _numeric_property(record, "final_fom_score"),
        },
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
        "mapped_parameters": candidate.mapped_parameters,
        "dft_properties": _legacy_dft_properties(candidate.mapped_parameters),
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
        "mapped_parameters": candidate.mapped_parameters,
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


def _property_value(record: MaterialRecord, name: str) -> Any:
    prop = record.properties.get(name)
    if prop is None:
        return None
    return prop.value


def _legacy_dft_properties(mapped: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "band_gap_hse_ev": mapped.get("absorber.Eg"),
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
