"""Loader for SCAPS-shape YAML configurations.

A SCAPS YAML keeps the partner-side schema (cgs units, ``N_C``/``N_V``,
microscopic defect triplets) so the parameter file can be checked against
the SCAPS GUI inputs side-by-side. The loader converts every field to the
SolarLab SI convention and returns a frozen ``DeviceStack`` that the
existing solver consumes without any code-path change.

Required device keys:    ``V_bi``, ``Phi``, ``mode``, ``layers``.
Required layer keys:     ``name``, ``role``, ``thickness_nm``, ``E_g_eV``,
                         ``chi_eV``, ``eps_r``, ``mu_n_cm2``, ``mu_p_cm2``,
                         ``N_C_cm3``, ``N_V_cm3``, ``N_D_cm3``, ``N_A_cm3``,
                         ``v_th_cm_s``.
Optional layer keys:     ``bulk_defect`` (``sigma_n_cm2``, ``sigma_p_cm2``,
                         ``N_t_cm3``, ``E_t_eV_below_cb``), ``B_rad_cm3_s``,
                         ``C_n_cm6_s``, ``C_p_cm6_s``, ``optical_material``,
                         ``alpha_cm``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

from perovskite_sim.models.device import DeviceStack, LayerSpec
from perovskite_sim.models.parameters import MaterialParams
from perovskite_sim.scaps_compat.defects import srh_lifetime
from perovskite_sim.scaps_compat.materials import ni_from_dos
from perovskite_sim.sweeps.device_parameter_sweep import (
    cm3_to_m3,
    cms_to_ms,
    srh_n1_p1_from_trap_depth,
)


_REQUIRED_TOP_KEYS = ("device", "layers")
_REQUIRED_DEVICE_KEYS = ("V_bi", "Phi", "mode")
_REQUIRED_LAYER_KEYS = (
    "name", "role", "thickness_nm", "E_g_eV", "chi_eV", "eps_r",
    "mu_n_cm2", "mu_p_cm2", "N_C_cm3", "N_V_cm3", "N_D_cm3", "N_A_cm3",
    "v_th_cm_s",
)
_DEFECT_FREE_TAU = 1.0e-3  # s; fallback lifetime when no bulk_defect block


def load_scaps_yaml(path: str | Path) -> DeviceStack:
    """Read a SCAPS-shape YAML and return the corresponding ``DeviceStack``."""
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}

    _check_keys(cfg, _REQUIRED_TOP_KEYS, where="top-level config")
    dev = cfg["device"] or {}
    _check_keys(dev, _REQUIRED_DEVICE_KEYS, where="device block")
    if not cfg["layers"]:
        raise ValueError("scaps yaml must define at least one layer")

    layers = tuple(_layer_from_scaps_row(row) for row in cfg["layers"])
    return DeviceStack(
        layers=layers,
        V_bi=float(dev["V_bi"]),
        Phi=float(dev["Phi"]),
        mode=str(dev["mode"]),
    )


def _layer_from_scaps_row(row: Mapping[str, Any]) -> LayerSpec:
    _check_keys(row, _REQUIRED_LAYER_KEYS, where=f"layer {row.get('name', '?')!r}")

    thickness_m = float(row["thickness_nm"]) * 1.0e-9
    Eg = float(row["E_g_eV"])
    chi = float(row["chi_eV"])
    N_C = cm3_to_m3(float(row["N_C_cm3"]))
    N_V = cm3_to_m3(float(row["N_V_cm3"]))
    v_th = cms_to_ms(float(row["v_th_cm_s"]))
    ni = ni_from_dos(N_C, N_V, Eg)

    bulk = row.get("bulk_defect") or {}
    if bulk:
        sigma_n = float(bulk["sigma_n_cm2"]) * 1.0e-4
        sigma_p = float(bulk["sigma_p_cm2"]) * 1.0e-4
        N_t = cm3_to_m3(float(bulk["N_t_cm3"]))
        tau_n = srh_lifetime(sigma_n, v_th, N_t)
        tau_p = srh_lifetime(sigma_p, v_th, N_t)
        depth = float(bulk.get("E_t_eV_below_cb", Eg / 2.0))
        n1, p1 = srh_n1_p1_from_trap_depth(ni, Eg, depth, reference="below_cb")
    else:
        tau_n = tau_p = _DEFECT_FREE_TAU
        n1 = p1 = ni

    params = MaterialParams(
        eps_r=float(row["eps_r"]),
        mu_n=float(row["mu_n_cm2"]) * 1.0e-4,
        mu_p=float(row["mu_p_cm2"]) * 1.0e-4,
        D_ion=float(row.get("D_ion_m2_s", 0.0)),
        P_lim=float(row.get("P_lim_m3", 1.0e30)),
        P0=float(row.get("P0_m3", 0.0)),
        ni=ni,
        tau_n=tau_n,
        tau_p=tau_p,
        n1=n1,
        p1=p1,
        B_rad=float(row.get("B_rad_cm3_s", 0.0)) * 1.0e-6,
        C_n=float(row.get("C_n_cm6_s", 0.0)) * 1.0e-12,
        C_p=float(row.get("C_p_cm6_s", 0.0)) * 1.0e-12,
        alpha=float(row.get("alpha_cm", 0.0)) * 1.0e2,
        N_A=cm3_to_m3(float(row["N_A_cm3"])),
        N_D=cm3_to_m3(float(row["N_D_cm3"])),
        chi=chi,
        Eg=Eg,
        optical_material=row.get("optical_material"),
    )
    return LayerSpec(
        name=str(row["name"]),
        thickness=thickness_m,
        params=params,
        role=str(row["role"]),
    )


def _check_keys(d: Mapping[str, Any], required: tuple[str, ...], *, where: str) -> None:
    missing = [k for k in required if k not in d]
    if missing:
        raise ValueError(f"{where} missing required keys: {missing}")
