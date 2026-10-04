"""Shared parameter contract for the report and the SCAPS frontend entry.

This keeps the historical effective-lifetime sweep mapping unchanged. In
particular the two bulk-Nt families scale the combined effective lifetime;
they are not independent microscopic-species validation.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from perovskite_sim.sweeps.device_parameter_sweep import SweepPoint, apply_sweep_point


REFERENCE_PATH = Path(__file__).resolve().parents[1] / "data" / "ScapsBeerLambertReferenceV1.json"
EXTRA_SWEEP_POINTS = {"Nd_PVK": (1e19, 1e20)}
SWEEPS = {
    "CHI_ETL": ("ETL/PVK conduction-band offset", "eV", "etl_delta_ec_eV", {}),
    "Nd_ETL": ("ETL donor doping", "cm^-3", "etl_doping_cm3", {}),
    "Nd_PVK": ("PVK donor doping", "cm^-3", "absorber_doping_cm3", {"absorber_doping_type": "donor"}),
    "Nt_C_PVK": ("PVK-CB effective bulk defect density", "cm^-3", "absorber_defect_density_cm3", {}),
    "Nt_V_PVK": ("PVK-VB effective bulk defect density", "cm^-3", "absorber_defect_density_cm3", {}),
    "Nt_HTL PVK": ("HTL/PVK interface defect density", "cm^-2", "interface_defect_N_t_cm2", {"interface_defect_target": "htl/pvk"}),
    "Nt_PVK ETL": ("PVK/ETL interface defect density", "cm^-2", "interface_defect_N_t_cm2", {"interface_defect_target": "pvk/etl"}),
    "Et_C_PVK": ("PVK-CB effective bulk defect depth", "eV", "absorber_defect_depth_eV", {"trap_depth_reference": "below_cb"}),
    "Et_V_PVK": ("PVK-VB effective bulk defect level", "eV", "absorber_defect_depth_eV", {"trap_depth_reference": "above_vb"}),
    "Et_HTL PVK": ("HTL/PVK interface defect depth", "eV", "interface_defect_E_t_eV", {"interface_defect_target": "htl/pvk"}),
    "Et_PVK ETL": ("PVK/ETL interface defect depth", "eV", "interface_defect_E_t_eV", {"interface_defect_target": "pvk/etl"}),
}


def reference_catalog():
    catalog = json.loads(REFERENCE_PATH.read_text())
    for sweep, values in EXTRA_SWEEP_POINTS.items():
        entry = catalog["sweeps"][sweep]
        entry["workbook_max"] = max(point["x"] for point in entry["points"])
        entry["points"].extend({
            "x": value, "row": None, "origin": "solarlab_extension",
            "reference": {key: None for key in ("V_oc", "J_sc", "FF", "PCE")},
        } for value in values)
    return catalog


def apply_reference_point(stack, sweep: str, value: float):
    if sweep not in SWEEPS:
        raise ValueError(f"Unknown SCAPS reference sweep: {sweep}")
    if isinstance(value, bool) or not math.isfinite(value):
        raise ValueError("Reference point must be finite")
    rows = reference_catalog()["sweeps"][sweep]["points"]
    if value < min(row["x"] for row in rows) or value > max(row["x"] for row in rows):
        raise ValueError("Reference point is outside this study's scan interval")
    _, _, key, extra = SWEEPS[sweep]
    return apply_sweep_point(stack, SweepPoint(sweep, sweep, str(value), {key: value, **extra}))
