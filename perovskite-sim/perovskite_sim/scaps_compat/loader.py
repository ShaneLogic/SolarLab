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

from perovskite_sim.models.device import DeviceStack, InterfaceDefect, LayerSpec
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
_REQUIRED_INTERFACE_KEYS = (
    "target", "sigma_n_cm2", "sigma_p_cm2", "N_t_cm2",
    "v_th_cm_s", "E_t_eV_below_cb",
)
# SCAPS role aliases for the ``target: A/B`` alias resolver. SCAPS UI uses
# ``PVK`` interchangeably with ``absorber``; SolarLab carries ``absorber``
# on the LayerSpec.role.
_ROLE_ALIAS = {
    "pvk": "absorber",
    "perovskite": "absorber",
    "absorber": "absorber",
    "htl": "htl",
    "etl": "etl",
}
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
    interfaces, interface_defects = _parse_interfaces_block(
        cfg.get("interfaces") or [], layers,
    )
    return DeviceStack(
        layers=layers,
        V_bi=float(dev["V_bi"]),
        Phi=float(dev["Phi"]),
        mode=str(dev["mode"]),
        interfaces=interfaces,
        interface_defects=interface_defects,
    )


def _parse_interfaces_block(
    entries: list,
    layers: tuple[LayerSpec, ...],
) -> tuple[
    tuple[tuple[float, float], ...],
    tuple[InterfaceDefect | None, ...],
]:
    """Translate the top-level ``interfaces:`` list into aligned tuples.

    SCAPS specifies an interface defect with the kinetic triplet
    ``(sigma, v_th, N_t_areal)`` plus a trap depth ``E_t`` below the
    reference (lower-Eg / absorber) conduction band. The surface
    recombination velocity follows the same SCAPS kinetic identity:

        v = sigma_si * v_th_si * N_t_areal_si      [m/s]

    With sigma in m², v_th in m/s, and N_t in m⁻² (areal density at the
    heterointerface), the dimensions cancel to m/s. The cgs → SI factors:

        sigma_si       = sigma_cm2  * 1e-4
        v_th_si        = v_th_cm_s  * 1e-2
        N_t_areal_si   = N_t_cm2    * 1e4
    """
    n_interfaces = max(0, len(layers) - 1)
    if not entries:
        return (), ()
    interfaces = [(0.0, 0.0)] * n_interfaces
    defects: list[InterfaceDefect | None] = [None] * n_interfaces
    for entry in entries:
        _check_keys(entry, _REQUIRED_INTERFACE_KEYS, where="interface entry")
        k = _resolve_interface_index(str(entry["target"]), layers)
        sigma_n_si = float(entry["sigma_n_cm2"]) * 1.0e-4
        sigma_p_si = float(entry["sigma_p_cm2"]) * 1.0e-4
        v_th_si = cms_to_ms(float(entry["v_th_cm_s"]))
        N_t_areal_si = float(entry["N_t_cm2"]) * 1.0e4
        v_n = sigma_n_si * v_th_si * N_t_areal_si
        v_p = sigma_p_si * v_th_si * N_t_areal_si
        interfaces[k] = (v_n, v_p)
        defects[k] = InterfaceDefect(E_t_eV=float(entry["E_t_eV_below_cb"]))
    return tuple(interfaces), tuple(defects)


def _resolve_interface_index(target: str, layers: tuple[LayerSpec, ...]) -> int:
    """Map a ``LEFT/RIGHT`` role-pair alias to a ``stack.layers`` interface index.

    Accepts SCAPS role names (``PVK`` ≡ ``absorber``) on either side,
    case-insensitive. Raises ``ValueError`` if no adjacent layer pair
    matches — typos must surface immediately rather than silently
    creating a defect at the wrong interface.
    """
    parts = [p.strip().lower() for p in target.split("/")]
    if len(parts) != 2 or any(not p for p in parts):
        raise ValueError(
            f"unknown interface target {target!r} (expected 'LEFT/RIGHT')"
        )
    left_alias = _ROLE_ALIAS.get(parts[0], parts[0])
    right_alias = _ROLE_ALIAS.get(parts[1], parts[1])
    for k in range(len(layers) - 1):
        left_role = layers[k].role.lower()
        right_role = layers[k + 1].role.lower()
        if left_role == left_alias and right_role == right_alias:
            return k
    raise ValueError(
        f"unknown interface target {target!r} — no adjacent layer pair "
        f"matches roles ({left_alias!r}, {right_alias!r}) in the stack "
        + str([l.role for l in layers])
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
        N_t_bulk_m3 = cm3_to_m3(float(bulk["N_t_cm3"]))
        tau_n = srh_lifetime(sigma_n, v_th, N_t_bulk_m3)
        tau_p = srh_lifetime(sigma_p, v_th, N_t_bulk_m3)
        depth = float(bulk.get("E_t_eV_below_cb", Eg / 2.0))
        n1, p1 = srh_n1_p1_from_trap_depth(ni, Eg, depth, reference="below_cb")
    else:
        tau_n = tau_p = _DEFECT_FREE_TAU
        n1 = p1 = ni
        N_t_bulk_m3 = None

    iface = row.get("interface_defect") or {}
    trap_N_t_interface = trap_N_t_bulk = trap_decay_length = None
    trap_profile_shape = "exponential"
    trap_edge = "both"
    if iface:
        trap_N_t_interface = cm3_to_m3(float(iface["N_t_peak_cm3"]))
        trap_decay_length = float(iface["decay_length_nm"]) * 1.0e-9
        trap_profile_shape = str(iface.get("profile", "exponential")).lower()
        trap_N_t_bulk = (
            N_t_bulk_m3 if N_t_bulk_m3 is not None else trap_N_t_interface
        )
        # SCAPS "target" names the interface face the defect sits on; map
        # to absorber-edge ("left"=HTL/absorber face, "right"=absorber/ETL).
        target = str(iface.get("target", "both")).lower()
        trap_edge = {
            "htl/pvk": "left", "left": "left",
            "pvk/etl": "right", "right": "right",
            "both": "both",
        }.get(target, "both")

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
        trap_N_t_interface=trap_N_t_interface,
        trap_N_t_bulk=trap_N_t_bulk,
        trap_decay_length=trap_decay_length,
        trap_profile_shape=trap_profile_shape,
        trap_edge=trap_edge,
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
