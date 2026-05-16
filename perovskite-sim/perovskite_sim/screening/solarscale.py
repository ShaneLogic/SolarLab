"""Generate SolarLab device inputs from SolarScale MaterialRecord exports.

This module is intentionally a thin bridge. It maps DFT-derived absorber
properties into existing SolarLab YAML configs and keeps device-specific
unknowns as sweep dimensions instead of guessing one exact value.
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass(frozen=True)
class GeneratedConfig:
    material_id: str
    path: Path
    dft_properties: dict[str, Any]
    sweep_parameters: dict[str, list[float]]


DEFAULT_SWEEP_GRID: dict[str, list[float]] = {
    "absorber.thickness": [300e-9, 500e-9, 800e-9],
    "absorber.tau_n": [1e-9, 1e-7, 1e-6],
    "absorber.tau_p": [1e-9, 1e-7, 1e-6],
    "absorber.trap_N_t_bulk": [1e20, 1e22, 1e24],
    "device.interfaces": [0.0, 1.0, 10.0],
}

_REQUIRED_DFT_PROPERTIES = (
    "band_gap_hse_ev",
    "electron_effective_mass_m0",
    "hole_effective_mass_m0",
    "dielectric_static_avg",
)


def load_material_records(path: str | Path) -> list[dict[str, Any]]:
    """Load the canonical SolarScale MaterialRecord JSON export."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError("MaterialRecord payload must contain a records list")
    return records


def generate_solarlab_inputs(
    records_path: str | Path,
    *,
    template_path: str | Path,
    out_dir: str | Path,
    limit: int | None = None,
    sweep_grid: Mapping[str, list[float]] | None = None,
) -> dict[str, Any]:
    """Generate SolarLab YAML configs and a manifest from SolarScale records.

    Parameters
    ----------
    records_path:
        Path to ``material_records.json`` exported by SolarScale.
    template_path:
        Existing SolarLab YAML config used for transport layers and default
        device parameters. The absorber layer is replaced with SolarScale
        material properties.
    out_dir:
        Directory where generated YAML files and ``manifest.json`` are written.
    limit:
        Optional maximum number of ready materials to emit, useful for dry runs.
    sweep_grid:
        Optional override for device-only unknown sweep values.
    """

    records = load_material_records(records_path)
    template = _load_yaml(template_path)
    sweeps = dict(sweep_grid or DEFAULT_SWEEP_GRID)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    generated: list[GeneratedConfig] = []
    skipped: list[dict[str, str]] = []
    for record in records:
        material_id = str(record.get("material_id", "")).strip()
        if not material_id:
            skipped.append({"material_id": "", "reason": "missing material_id"})
            continue
        ready, reason = _is_device_ready(record)
        if not ready:
            skipped.append({"material_id": material_id, "reason": reason})
            continue
        if limit is not None and len(generated) >= limit:
            skipped.append({"material_id": material_id, "reason": "limit reached"})
            continue

        config = _build_config(template, record, sweeps)
        config_path = out / f"{_safe_name(material_id)}.yaml"
        _write_yaml(config_path, config)
        generated.append(
            GeneratedConfig(
                material_id=material_id,
                path=config_path,
                dft_properties=_dft_property_values(record),
                sweep_parameters=sweeps,
            )
        )

    manifest = {
        "schema": "solarlab.solarscale_import_manifest",
        "schema_version": "0.1",
        "records_path": str(records_path),
        "template_path": str(template_path),
        "out_dir": str(out),
        "generated": [
            {
                "material_id": item.material_id,
                "config_path": str(item.path),
                "dft_properties": item.dft_properties,
                "sweep_parameters": item.sweep_parameters,
            }
            for item in generated
        ],
        "skipped": skipped,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


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


def _is_device_ready(record: Mapping[str, Any]) -> tuple[bool, str]:
    properties = _properties(record)
    flag = properties.get("dft_result_available", {})
    if flag.get("value") is not True:
        return False, "dft_result_available is not true"
    missing = [name for name in _REQUIRED_DFT_PROPERTIES if _property_value(record, name) is None]
    if missing:
        return False, "missing required DFT properties: " + ", ".join(missing)
    return True, "ready"


def _build_config(
    template: Mapping[str, Any], record: Mapping[str, Any], sweeps: Mapping[str, list[float]]
) -> dict[str, Any]:
    config = copy.deepcopy(dict(template))
    layers = config.get("layers")
    if not isinstance(layers, list):
        raise ValueError("Template layers must be a list")

    absorber = _absorber_layer(layers)
    material_id = str(record["material_id"])
    dft_values = _dft_property_values(record)

    absorber["name"] = f"SolarScale_{material_id}"
    # Keep active solver Eg/chi at the template value unless the template is
    # already fully band-aligned. Setting only absorber Eg would activate
    # DeviceStack.compute_V_bi() with uncalibrated contact-layer band offsets.
    absorber["eps_r"] = dft_values["dielectric_static_avg"]
    absorber["mu_n"] = _mobility_m2_v_s(record, "electron_mobility_cm2_v_s", fallback=2e-4)
    absorber["mu_p"] = _mobility_m2_v_s(record, "hole_mobility_cm2_v_s", fallback=2e-4)
    absorber["thickness"] = sweeps["absorber.thickness"][0]
    absorber["tau_n"] = sweeps["absorber.tau_n"][0]
    absorber["tau_p"] = sweeps["absorber.tau_p"][0]
    absorber["trap_N_t_bulk"] = sweeps["absorber.trap_N_t_bulk"][0]
    absorber["trap_profile_shape"] = "exponential"

    if _property_value(record, "absorption_edge_ev") is not None:
        absorber["alpha"] = absorber.get("alpha", 1.0e7)
    if _property_value(record, "exciton_binding_ev") is not None:
        absorber.setdefault("B_rad", absorber.get("B_rad", 5.0e-22))

    device = config.setdefault("device", {})
    interface_s = sweeps["device.interfaces"][0]
    device["interfaces"] = [[interface_s, interface_s] for _ in range(max(0, len(layers) - 1))]
    device.setdefault("mode", "fast")

    notes = list(config.get("notes", []) or [])
    notes.append(
        "Generated from SolarScale MaterialRecord. Device-only unknowns use the first value of each sweep dimension; see manifest.json for the full sweep grid."
    )
    config["notes"] = notes
    config["source"] = {
        "material_id": material_id,
        "schema": "solarlab.solarscale_import_config",
        "dft_properties": dft_values,
        "sweep_parameters": dict(sweeps),
    }
    config["simulation_hints"] = {
        "N_grid": 30,
        "n_points": 8,
        "v_rate": 5.0,
        "V_max": 1.4,
        "notes": (
            "First-pass SolarScale imports should run low-resolution screening "
            "with explicit V_max while the DFT band gap is kept in source "
            "metadata until contact-layer band offsets are calibrated."
        ),
    }
    return config


def _absorber_layer(layers: list[dict[str, Any]]) -> dict[str, Any]:
    absorbers = [layer for layer in layers if layer.get("role") == "absorber"]
    if len(absorbers) != 1:
        raise ValueError("Template must contain exactly one absorber layer")
    return absorbers[0]


def _properties(record: Mapping[str, Any]) -> Mapping[str, Any]:
    props = record.get("properties", {})
    if not isinstance(props, Mapping):
        return {}
    return props


def _property_value(record: Mapping[str, Any], name: str) -> Any:
    prop = _properties(record).get(name, {})
    if not isinstance(prop, Mapping):
        return None
    return prop.get("value")


def _dft_property_values(record: Mapping[str, Any]) -> dict[str, Any]:
    return {name: _property_value(record, name) for name in _REQUIRED_DFT_PROPERTIES}


def _mobility_m2_v_s(record: Mapping[str, Any], name: str, *, fallback: float) -> float:
    value = _property_value(record, name)
    if value is None:
        return fallback
    return float(value) * 1.0e-4


def _safe_name(material_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", material_id)
