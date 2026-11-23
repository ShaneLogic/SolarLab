"""Read-only SCAPS Beer-Lambert reference catalog and parameter-point contract.

Packages the 11-family SCAPS reference table extracted from
``reproducibility/ScapsParams.xlsx`` (93 workbook rows, retaining unmeasured
CBO rows as ``None``) plus the two SolarLab-only high-doping ``Nd_PVK``
extrapolation points (``1e19`` and ``1e20 cm^-3`` with ``min_N_grid=90`` and
``None`` reference metrics).

Historical parameter-sweep note:
- In ``historical_effective_lifetime`` mode (default), ``Nt_C_PVK`` and
  ``Nt_V_PVK`` scale the combined absorber bulk defect inventory so that
  ``1/tau_total`` reproduces the historical effective-lifetime sweep, while
  ``Et_C_PVK`` and ``Et_V_PVK`` set the referenced effective trap depth; they
  are NOT independent microscopic-species validation unless
  ``bulk_defect_mode="species_only"`` is explicitly requested.
- Interface density sweeps (``Nt_HTL PVK``, ``Nt_PVK ETL``) retain the
  historical areal ``cm^-2`` interpretation despite the workbook's ``cm^-3``
  column header; no synthetic thickness conversion is inferred.
"""

from __future__ import annotations

from copy import deepcopy
from importlib import resources
import json
import math
from typing import Any, Literal

from solarlab.device.inputs import DeviceInput
from solarlab.physics.contacts import ContactMaterial, semiconductor_work_function
from solarlab.physics.statistics.thermal import thermal_voltage

__all__ = [
    "EXTRA_SWEEP_POINTS",
    "HIGH_DOPING_MIN_N_GRID",
    "SCAPS_BEER_LAMBERT_RESOURCE",
    "SCAPS_BEER_LAMBERT_STUDY_ID",
    "STANDARD_JV_SWEEP_HINTS",
    "SWEEPS",
    "apply_reference_point",
    "load_scaps_beer_lambert_catalog",
    "reference_catalog",
    "validate_reference_point",
]

SCAPS_BEER_LAMBERT_STUDY_ID = "scaps_beer_lambert_v1"
SCAPS_BEER_LAMBERT_RESOURCE = "ScapsBeerLambertReferenceV1.json"
EXTRA_SWEEP_POINTS: dict[str, tuple[float, ...]] = {
    "Nd_PVK": (1.0e19, 1.0e20),
}
STANDARD_JV_SWEEP_HINTS: dict[str, int | float] = {
    "N_grid": 30,
    "n_points": 40,
    "v_rate": 5.0,
    "V_max": 1.6,
}
HIGH_DOPING_MIN_N_GRID = 90
BulkDefectSweepMode = Literal["historical_effective_lifetime", "species_only"]

SWEEPS: dict[str, tuple[str, str, str, dict[str, str]]] = {
    "CHI_ETL": (
        "ETL/PVK conduction-band offset",
        "eV",
        "etl_delta_ec_eV",
        {},
    ),
    "Nd_ETL": (
        "ETL donor doping",
        "cm^-3",
        "etl_doping_cm3",
        {},
    ),
    "Nd_PVK": (
        "PVK donor doping",
        "cm^-3",
        "absorber_doping_cm3",
        {"absorber_doping_type": "donor"},
    ),
    "Nt_C_PVK": (
        "PVK-CB effective bulk defect density",
        "cm^-3",
        "absorber_defect_density_cm3",
        {"species_name": "Perovskite-CB"},
    ),
    "Nt_V_PVK": (
        "PVK-VB effective bulk defect density",
        "cm^-3",
        "absorber_defect_density_cm3",
        {"species_name": "Perovskite-VB"},
    ),
    "Nt_HTL PVK": (
        "HTL/PVK interface defect density",
        "cm^-2",
        "interface_defect_N_t_cm2",
        {"interface_defect_target": "htl/pvk"},
    ),
    "Nt_PVK ETL": (
        "PVK/ETL interface defect density",
        "cm^-2",
        "interface_defect_N_t_cm2",
        {"interface_defect_target": "pvk/etl"},
    ),
    "Et_C_PVK": (
        "PVK-CB effective bulk defect depth",
        "eV",
        "absorber_defect_depth_eV",
        {
            "trap_depth_reference": "below_conduction_band",
            "species_name": "Perovskite-CB",
        },
    ),
    "Et_V_PVK": (
        "PVK-VB effective bulk defect level",
        "eV",
        "absorber_defect_depth_eV",
        {
            "trap_depth_reference": "above_valence_band",
            "species_name": "Perovskite-VB",
        },
    ),
    "Et_HTL PVK": (
        "HTL/PVK interface defect depth",
        "eV",
        "interface_defect_E_t_eV",
        {"interface_defect_target": "htl/pvk"},
    ),
    "Et_PVK ETL": (
        "PVK/ETL interface defect depth",
        "eV",
        "interface_defect_E_t_eV",
        {"interface_defect_target": "pvk/etl"},
    ),
}


def load_scaps_beer_lambert_catalog() -> dict[str, Any]:
    """Return a fresh copy of the 11-family SCAPS Beer-Lambert catalog."""

    raw = (
        resources.files("solarlab.reproducibility.data")
        .joinpath(SCAPS_BEER_LAMBERT_RESOURCE)
        .read_text(encoding="utf-8")
    )
    catalog: dict[str, Any] = json.loads(raw)
    for sweep, values in EXTRA_SWEEP_POINTS.items():
        entry = catalog["sweeps"][sweep]
        entry["workbook_max"] = max(float(point["x"]) for point in entry["points"])
        entry["points"].extend(
            {
                "x": float(val),
                "row": None,
                "origin": "solarlab_extension",
                "reference": {
                    key: None for key in ("V_oc", "J_sc", "FF", "PCE")
                },
            }
            for val in values
        )
    return catalog


def reference_catalog() -> dict[str, Any]:
    """Alias matching the historical validation catalog entry point."""

    return load_scaps_beer_lambert_catalog()


def validate_reference_point(sweep: str, value: float) -> dict[str, Any]:
    """Validate ``(sweep, value)`` fail-closed and return point metadata."""

    if not isinstance(sweep, str) or sweep not in SWEEPS:
        raise ValueError(f"Unknown SCAPS reference sweep: {sweep!r}")
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError("Reference point value must be a finite real number")
    numeric = float(value)
    catalog = load_scaps_beer_lambert_catalog()
    entry = catalog["sweeps"][sweep]
    points = entry["points"]
    xs = [float(row["x"]) for row in points]
    if numeric < min(xs) or numeric > max(xs):
        raise ValueError(
            f"Reference point {numeric!r} is outside {sweep!r} scan interval "
            f"[{min(xs)}, {max(xs)}]"
        )
    workbook_max = entry.get("workbook_max")
    is_extension = workbook_max is not None and numeric > float(workbook_max)
    min_n_grid = (
        HIGH_DOPING_MIN_N_GRID
        if is_extension
        else int(STANDARD_JV_SWEEP_HINTS["N_grid"])
    )
    title, unit, key, extra = SWEEPS[sweep]
    jv_hints = dict(STANDARD_JV_SWEEP_HINTS)
    jv_hints["N_grid"] = min_n_grid
    return {
        "reference_study": SCAPS_BEER_LAMBERT_STUDY_ID,
        "sweep": sweep,
        "value": numeric,
        "unit": unit,
        "title": title,
        "handler_key": key,
        "handler_extra": dict(extra),
        "optical_model": "beer_lambert_spectral",
        "scope": (
            "high_doping_extrapolation" if is_extension else "workbook_interval"
        ),
        "min_N_grid": min_n_grid,
        "jv_sweep": jv_hints,
    }


def _find_layer_by_role(
    layers: list[dict[str, Any]],
    role: str,
) -> dict[str, Any]:
    matches = [
        layer
        for layer in layers
        if str(layer.get("role", "")).lower() == role.lower()
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {role!r} layer, found {len(matches)}")
    return matches[0]


def _find_interface_by_roles(
    layers: list[dict[str, Any]],
    interfaces: list[dict[str, Any]],
    target: str,
) -> dict[str, Any]:
    role_alias = {
        "pvk": "absorber",
        "perovskite": "absorber",
        "absorber": "absorber",
        "htl": "htl",
        "etl": "etl",
    }
    parts = [role_alias.get(p.strip().lower(), p.strip().lower()) for p in target.split("/")]
    if len(parts) != 2:
        raise ValueError(f"Expected directed interface target LEFT/RIGHT, got {target!r}")
    by_id = {str(layer["id"]): str(layer["role"]).lower() for layer in layers}
    matches = [
        iface
        for iface in interfaces
        if [by_id.get(str(iface.get("left"))), by_id.get(str(iface.get("right")))]
        == parts
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one directed interface matching {target!r}, found {len(matches)}"
        )
    return matches[0]


def _recompute_doped_vbi(data: dict[str, Any]) -> float | None:
    electrical = [
        layer
        for layer in data["layers"]
        if str(layer.get("role", "")).lower() != "substrate"
    ]
    if len(electrical) < 2:
        return None
    temp = float(data.get("settings", {}).get("T", 300.0))

    def _wf(layer: dict[str, Any]) -> float | None:
        p = layer.get("parameters", {})
        required = ("chi", "Eg", "Nc300", "Nv300")
        if any(p.get(k) is None for k in required):
            return None
        mat = ContactMaterial(
            temperature_K=temp,
            electron_affinity_eV=float(p["chi"]),
            band_gap_eV=float(p["Eg"]),
            conduction_dos_m3=float(p["Nc300"]),
            valence_dos_m3=float(p["Nv300"]),
            donor_density_m3=float(p.get("N_D", 0.0) or 0.0),
            acceptor_density_m3=float(p.get("N_A", 0.0) or 0.0),
        )
        return semiconductor_work_function(mat)

    wf_left = _wf(electrical[0])
    wf_right = _wf(electrical[-1])
    if wf_left is None or wf_right is None:
        return None
    return float(wf_left - wf_right)


def apply_reference_point(
    device: DeviceInput,
    sweep: str,
    value: float,
    *,
    sync_vbi: bool = True,
    bulk_defect_mode: BulkDefectSweepMode = "historical_effective_lifetime",
) -> tuple[DeviceInput, dict[str, Any]]:
    """Apply one SCAPS Beer-Lambert reference point to a ``DeviceInput``."""

    if not isinstance(device, DeviceInput):
        raise TypeError("device must be a DeviceInput")
    if bulk_defect_mode not in ("historical_effective_lifetime", "species_only"):
        raise ValueError(
            "bulk_defect_mode must be 'historical_effective_lifetime' or 'species_only'"
        )
    meta = validate_reference_point(sweep, value)
    numeric = float(meta["value"])
    key = str(meta["handler_key"])
    extra: dict[str, str] = dict(meta["handler_extra"])

    data = deepcopy(device.editing_data())
    # Editing canonical parameters clears original source-byte legacy_fields check
    # only if the caller re-resolves without changing source_id; retain settings/layers.
    data.pop("legacy_fields", None)
    layers: list[dict[str, Any]] = data["layers"]
    interfaces: list[dict[str, Any]] = data.setdefault("interfaces", [])

    if key == "etl_delta_ec_eV":
        pvk = _find_layer_by_role(layers, "absorber")
        etl = _find_layer_by_role(layers, "etl")
        chi_pvk = float(pvk["parameters"]["chi"])
        etl.setdefault("parameters", {})["chi"] = chi_pvk - numeric
    elif key == "etl_doping_cm3":
        etl = _find_layer_by_role(layers, "etl")
        params = etl.setdefault("parameters", {})
        params["N_D"] = numeric * 1.0e6
        params["N_A"] = 0.0
    elif key == "absorber_doping_cm3":
        pvk = _find_layer_by_role(layers, "absorber")
        params = pvk.setdefault("parameters", {})
        params["N_D"] = numeric * 1.0e6
        params["N_A"] = 0.0
    elif key == "absorber_defect_density_cm3":
        pvk = _find_layer_by_role(layers, "absorber")
        target_m3 = numeric * 1.0e6
        if target_m3 <= 0.0:
            raise ValueError("absorber_defect_density_cm3 must be positive")
        bulk_defects: list[dict[str, Any]] = pvk.get("bulk_defects", [])
        if bulk_defects:
            if bulk_defect_mode == "species_only":
                species_name = extra["species_name"]
                matched = [
                    d for d in bulk_defects if d.get("name") == species_name
                ]
                if len(matched) != 1:
                    raise ValueError(
                        f"Expected one bulk defect named {species_name!r}"
                    )
                matched[0]["distribution"]["total_density_m3"] = target_m3
            else:
                base_sum = sum(
                    float(d["distribution"]["total_density_m3"])
                    for d in bulk_defects
                )
                if base_sum <= 0.0:
                    raise ValueError("Baseline bulk defect density sum must be positive")
                ratio = target_m3 / base_sum
                for d in bulk_defects:
                    d["distribution"]["total_density_m3"] = (
                        float(d["distribution"]["total_density_m3"]) * ratio
                    )
        else:
            params = pvk.setdefault("parameters", {})
            ref_m3 = float(params.get("trap_N_t_bulk") or 1.0e22)
            scale = ref_m3 / target_m3
            params["trap_N_t_bulk"] = target_m3
            if params.get("tau_n") is not None:
                params["tau_n"] = float(params["tau_n"]) * scale
            if params.get("tau_p") is not None:
                params["tau_p"] = float(params["tau_p"]) * scale
    elif key == "absorber_defect_depth_eV":
        pvk = _find_layer_by_role(layers, "absorber")
        ref_kind = extra["trap_depth_reference"]
        bulk_defects = pvk.get("bulk_defects", [])
        if bulk_defects:
            targets = (
                [
                    d
                    for d in bulk_defects
                    if d.get("name") == extra["species_name"]
                ]
                if bulk_defect_mode == "species_only"
                else bulk_defects
            )
            if not targets:
                raise ValueError(
                    f"Expected bulk defect {extra['species_name']!r} in absorber"
                )
            for d in targets:
                d["energy_level"] = {
                    "reference": ref_kind,
                    "value_eV": numeric,
                }
        else:
            params = pvk.setdefault("parameters", {})
            gap = float(params["Eg"])
            temp = float(data.get("settings", {}).get("T", 300.0))
            vt = float(thermal_voltage(temp))
            ni = float(params["ni"])
            delta = (
                gap / 2.0 - numeric
                if ref_kind == "below_conduction_band"
                else numeric - gap / 2.0
            )
            ratio = math.exp(delta / vt)
            params["n1"] = ni * ratio
            params["p1"] = ni / ratio
    elif key == "interface_defect_N_t_cm2":
        if numeric <= 0.0:
            raise ValueError("interface_defect_N_t_cm2 must be positive")
        iface = _find_interface_by_roles(
            layers, interfaces, extra["interface_defect_target"]
        )
        defect = iface.get("defect")
        if not isinstance(defect, dict):
            raise ValueError(
                f"Interface target {extra['interface_defect_target']!r} has no "
                "declared defect"
            )
        old_m2 = float(defect["total_density_m2"])
        new_m2 = numeric * 1.0e4
        defect["total_density_m2"] = new_m2
        if old_m2 > 0.0:
            srv_ratio = new_m2 / old_m2
            for v_key in ("v_n", "v_p"):
                if iface.get(v_key) is not None:
                    iface[v_key] = float(iface[v_key]) * srv_ratio
    elif key == "interface_defect_E_t_eV":
        iface = _find_interface_by_roles(
            layers, interfaces, extra["interface_defect_target"]
        )
        defect = iface.get("defect")
        if not isinstance(defect, dict):
            raise ValueError(
                f"Interface target {extra['interface_defect_target']!r} has no "
                "declared defect"
            )
        defect["trap_depth_eV"] = numeric
    else:
        raise ValueError(f"Unsupported sweep handler key: {key!r}")

    computed_vbi = _recompute_doped_vbi(data)
    settings = data.setdefault("settings", {})
    if sync_vbi and computed_vbi is not None and "V_bi" in settings:
        settings["V_bi"] = computed_vbi
    meta["computed_V_bi"] = computed_vbi
    meta["bulk_defect_mode"] = bulk_defect_mode
    return DeviceInput.model_validate(data), meta
