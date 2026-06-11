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
Optional layer keys:     ``bulk_defect`` (singular legacy block),
                         ``bulk_defects`` (Phase E6.3 plural list — see
                         below), ``B_rad_cm3_s``, ``C_n_cm6_s``,
                         ``C_p_cm6_s``, ``optical_material``, ``alpha_cm``.

Phase E6.3 — multi-defect bulk SRH (plural ``bulk_defects:``):
    Each entry carries the same kinetic triplet as the singular block plus
    exactly one of ``E_t_eV_below_cb`` (depth measured from the conduction
    band) or ``E_t_eV_above_vb`` (depth measured from the valence band).
    The loader combines N parallel defects via:

        1/tau_total = sum_i 1/tau_i      (parallel SRH at high injection)
        n1_eff      = (sum_i n1_i / tau_n_i) * tau_n_total
        p1_eff      = (sum_i p1_i / tau_p_i) * tau_p_total

    The n1/p1 weighting reduces to the arithmetic mean when all taus are
    equal — the SCAPS PVK case (two identical-kinetics defects, one near
    CB and one near VB) pins this in
    ``tests/unit/scaps_compat/test_loader_multi_defect.py``.

    Singular ``bulk_defect:`` and plural ``bulk_defects:`` are mutually
    exclusive on a layer — declaring both raises ``ValueError``.

Phase E6.3 — energy distribution (bulk and interface):
    Optional ``distribution: single | gaussian`` field. ``single`` (the
    default) preserves pre-E6.3 behaviour. ``gaussian`` documents the
    energy distribution and accepts optional ``E_char_eV`` and
    ``N_peak_cm3`` informational fields; the loader uses ``N_t_cm3``
    (bulk) or ``N_t_cm2`` (interface) directly without recomputing the
    integral — the SCAPS PDF column-header units make the standard
    Gaussian normalisation ambiguous (5.64e8 cm^-3 peak vs 1e12 cm^-3
    total — ratio 5.64e-4 does not match sqrt(2*pi)*E_char). The YAML
    therefore mirrors the SCAPS GUI inputs verbatim and the loader trusts
    the SCAPS-input value.

Phase E6.3 — strict key validation:
    Defect and interface entries reject unknown keys (raise
    ``ValueError``). Pre-E6.3 the loader silently dropped unrecognised
    fields, which masked a 4-order sigma error hidden behind a typo'd
    ``calibration_facor`` would have escaped detection for months
    (see Phase E1.6 ``calibration_factor`` history in
    ``configs/scaps_mirror.yaml`` comments).
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
    "v_th_cm_s",
)
_OPTIONAL_INTERFACE_KEYS = (
    "E_t_eV_below_cb", "E_t_eV_above_vb", "calibration_factor",
    "distribution", "E_char_eV", "N_peak_cm3",
)
_REQUIRED_BULK_DEFECT_KEYS = ("sigma_n_cm2", "sigma_p_cm2", "N_t_cm3")
_OPTIONAL_BULK_DEFECT_KEYS = (
    "name", "E_t_eV_below_cb", "E_t_eV_above_vb",
    "distribution", "E_char_eV", "N_peak_cm3",
)
_VALID_DISTRIBUTIONS = ("single", "gaussian")
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
    # Phase E1.10 — Robin / selective outer-contact S fields. Mirrors the
    # flat top-level schema accepted by ``models/config_loader.py`` so
    # SCAPS-shape YAMLs can opt into Phase 3.3 Robin contacts without
    # switching loader. Absent → None (Dirichlet ohmic default, pre-3.3
    # behaviour). 0.0 → Neumann blocking sentinel (distinct from absent).
    # Positive finite → Robin surface recombination velocity m/s.
    return DeviceStack(
        layers=layers,
        V_bi=float(dev["V_bi"]),
        Phi=float(dev["Phi"]),
        mode=str(dev["mode"]),
        interfaces=interfaces,
        interface_defects=interface_defects,
        interface_plane_projection=(
            str(dev.get("interface_plane_projection", False)).strip().lower()
            in ("true", "1", "yes", "on")
        ),
        dos_band_potentials=(
            str(dev.get("dos_band_potentials", False)).strip().lower()
            in ("true", "1", "yes", "on")
        ),
        flat_band_contacts=(
            str(dev.get("flat_band_contacts", False)).strip().lower()
            in ("true", "1", "yes", "on")
        ),
        interface_two_sided=(
            str(dev.get("interface_two_sided", False)).strip().lower()
            in ("true", "1", "yes", "on")
        ),
        interface_shared_occupancy=(
            str(dev.get("interface_shared_occupancy", False)).strip().lower()
            in ("true", "1", "yes", "on")
        ),
        S_n_left=_opt_S(dev.get("S_n_left")),
        S_p_left=_opt_S(dev.get("S_p_left")),
        S_n_right=_opt_S(dev.get("S_n_right")),
        S_p_right=_opt_S(dev.get("S_p_right")),
    )


def _opt_S(v: Any) -> float | None:
    """Coerce a YAML S value to float or None.

    None / absent → None (Dirichlet ohmic default).
    Numeric (incl. 0.0) → float; ``0.0`` is the Neumann blocking
    sentinel and must NOT be coerced to None.
    """
    if v is None:
        return None
    return float(v)


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
        target = str(entry.get("target", "<missing>"))
        where = f"interface entry target={target!r}"
        _check_keys_strict(
            entry,
            required=_REQUIRED_INTERFACE_KEYS,
            optional=_OPTIONAL_INTERFACE_KEYS,
            where=where,
        )
        _resolve_distribution(entry, where)
        depth, _ref = _resolve_trap_depth(entry, where, Eg_eV=0.0)
        k = _resolve_interface_index(target, layers)
        sigma_n_si = float(entry["sigma_n_cm2"]) * 1.0e-4
        sigma_p_si = float(entry["sigma_p_cm2"]) * 1.0e-4
        v_th_si = cms_to_ms(float(entry["v_th_cm_s"]))
        N_t_areal_si = float(entry["N_t_cm2"]) * 1.0e4
        v_n = sigma_n_si * v_th_si * N_t_areal_si
        v_p = sigma_p_si * v_th_si * N_t_areal_si
        interfaces[k] = (v_n, v_p)
        # Phase E1.6 — optional calibration_factor (default 1.0 = legacy).
        # Mirrors the schema accepted by ``backend/main.py:stack_from_dict``
        # so SCAPS YAML and inline-device dicts surface the same partner-
        # facing field.
        calibration_factor = float(entry.get("calibration_factor", 1.0))
        defects[k] = InterfaceDefect(
            E_t_eV=depth,
            calibration_factor=calibration_factor,
            N_t_cm2=float(entry["N_t_cm2"]),
        )
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

    layer_name = row.get("name", "?")
    has_singular = "bulk_defect" in row and row["bulk_defect"] is not None
    has_plural = "bulk_defects" in row
    if has_singular and has_plural:
        raise ValueError(
            f"layer {layer_name!r} cannot declare both 'bulk_defect' (singular) "
            "and 'bulk_defects' (plural)"
        )

    parsed_defects: list[dict[str, float]] = []
    if has_singular:
        parsed_defects.append(
            _parse_one_bulk_defect(
                row["bulk_defect"],
                ni_m3=ni,
                Eg_eV=Eg,
                v_th_si=v_th,
                where=f"layer {layer_name!r} bulk_defect",
            )
        )
    elif has_plural:
        for i, entry in enumerate(row.get("bulk_defects") or []):
            parsed_defects.append(
                _parse_one_bulk_defect(
                    entry,
                    ni_m3=ni,
                    Eg_eV=Eg,
                    v_th_si=v_th,
                    where=f"layer {layer_name!r} bulk_defects[{i}]",
                )
            )

    tau_n, tau_p, n1, p1, N_t_bulk_m3 = _combine_bulk_defects(parsed_defects)
    if not parsed_defects:
        # Defect-free fallback: n1/p1 set to ni for the legacy midgap-equivalent.
        n1 = p1 = ni

    iface = row.get("interface_defect") or {}
    trap_N_t_interface = trap_decay_length = None
    # Record the combined declared bulk density so sweeps can ratio-scale
    # tau off the config's own base (sigma-consistent, like the E9 interface
    # fix) instead of the absolute 1e22 m^-3 fallback. Setting this alone
    # does NOT activate the Phase-4a trap profile (which also requires
    # trap_N_t_interface + trap_decay_length).
    trap_N_t_bulk = N_t_bulk_m3
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
        # Effective DOS [m⁻³] — consumed by the dos_band_potentials flag
        # (V_T·ln(N_C)/ln(N_V) heterojunction transport terms).
        Nc300=N_C,
        Nv300=N_V,
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


def _check_keys_strict(
    d: Mapping[str, Any],
    *,
    required: tuple[str, ...],
    optional: tuple[str, ...],
    where: str,
) -> None:
    """Reject unknown keys in addition to enforcing required-key presence.

    Pre-E6.3 the loader silently dropped unrecognised fields, which masked
    schema typos (e.g. ``calibration_facor`` vs ``calibration_factor``)
    and let partner-side schema drift disable defects without surfacing
    an error. Strict validation surfaces typos immediately so the
    SCAPS-mirror YAML cannot lie about its physics.
    """
    _check_keys(d, required, where=where)
    allowed = set(required) | set(optional)
    unknown = sorted(k for k in d if k not in allowed)
    if unknown:
        raise ValueError(f"{where} has unknown keys: {unknown}")


def _resolve_trap_depth(
    d: Mapping[str, Any], where: str, Eg_eV: float
) -> tuple[float, str]:
    """Return (depth_eV, reference) for a defect entry.

    Exactly one of ``E_t_eV_below_cb`` / ``E_t_eV_above_vb`` must be set.
    Raising on both-or-neither prevents the YAML from silently defaulting
    to midgap for a defect the partner explicitly placed near a band edge.
    """
    has_below = "E_t_eV_below_cb" in d
    has_above = "E_t_eV_above_vb" in d
    if has_below == has_above:
        raise ValueError(
            f"{where}: exactly one of E_t_eV_below_cb / E_t_eV_above_vb required"
        )
    if has_below:
        return float(d["E_t_eV_below_cb"]), "below_cb"
    return float(d["E_t_eV_above_vb"]), "above_vb"


def _resolve_distribution(d: Mapping[str, Any], where: str) -> str:
    dist = str(d.get("distribution", "single")).lower()
    if dist not in _VALID_DISTRIBUTIONS:
        raise ValueError(
            f"{where}: distribution {d.get('distribution')!r} unknown "
            f"(use one of {_VALID_DISTRIBUTIONS})"
        )
    return dist


def _parse_one_bulk_defect(
    d: Mapping[str, Any],
    *,
    ni_m3: float,
    Eg_eV: float,
    v_th_si: float,
    where: str,
) -> dict[str, float]:
    """Translate one bulk_defect entry to (sigma_n, sigma_p, N_t_m3, n1, p1).

    Distribution handling: ``single`` and ``gaussian`` both use the
    SCAPS-input ``N_t_cm3`` value directly. The Gaussian variant accepts
    ``E_char_eV`` and ``N_peak_cm3`` as informational fields that are
    validated for structure (must be numeric) but not consumed by the
    SRH lifetime calculation — see the module docstring for rationale.
    """
    _check_keys_strict(
        d,
        required=_REQUIRED_BULK_DEFECT_KEYS,
        optional=_OPTIONAL_BULK_DEFECT_KEYS,
        where=where,
    )
    _resolve_distribution(d, where)  # validate even if unused downstream
    sigma_n_si = float(d["sigma_n_cm2"]) * 1.0e-4
    sigma_p_si = float(d["sigma_p_cm2"]) * 1.0e-4
    N_t_m3 = cm3_to_m3(float(d["N_t_cm3"]))
    depth, ref = _resolve_trap_depth(d, where, Eg_eV)
    n1, p1 = srh_n1_p1_from_trap_depth(ni_m3, Eg_eV, depth, reference=ref)
    return {
        "sigma_n_si": sigma_n_si,
        "sigma_p_si": sigma_p_si,
        "N_t_m3": N_t_m3,
        "v_th_si": v_th_si,
        "n1": n1,
        "p1": p1,
    }


def _combine_bulk_defects(
    parsed: list[dict[str, float]],
) -> tuple[float, float, float, float, float | None]:
    """Combine N parallel SRH defects to a single effective (tau, n1, p1).

    Lifetimes add in inverse (parallel resistors / parallel SRH paths at
    high injection). The n1/p1 are weighted by inverse-lifetime so the
    defect contributing more recombination dominates the effective trap
    location — when all taus are equal this reduces to the arithmetic
    mean. Returns ``_DEFECT_FREE_TAU`` sentinels on an empty input.
    """
    if not parsed:
        return _DEFECT_FREE_TAU, _DEFECT_FREE_TAU, 0.0, 0.0, None

    inv_tau_n_total = 0.0
    inv_tau_p_total = 0.0
    sum_n1_over_tau_n = 0.0
    sum_p1_over_tau_p = 0.0
    N_t_sum_m3 = 0.0
    for d in parsed:
        inv_tau_n_i = d["sigma_n_si"] * d["v_th_si"] * d["N_t_m3"]
        inv_tau_p_i = d["sigma_p_si"] * d["v_th_si"] * d["N_t_m3"]
        inv_tau_n_total += inv_tau_n_i
        inv_tau_p_total += inv_tau_p_i
        sum_n1_over_tau_n += d["n1"] * inv_tau_n_i
        sum_p1_over_tau_p += d["p1"] * inv_tau_p_i
        N_t_sum_m3 += d["N_t_m3"]

    tau_n = 1.0 / inv_tau_n_total if inv_tau_n_total > 0.0 else _DEFECT_FREE_TAU
    tau_p = 1.0 / inv_tau_p_total if inv_tau_p_total > 0.0 else _DEFECT_FREE_TAU
    n1 = sum_n1_over_tau_n / inv_tau_n_total if inv_tau_n_total > 0.0 else 0.0
    p1 = sum_p1_over_tau_p / inv_tau_p_total if inv_tau_p_total > 0.0 else 0.0
    return tau_n, tau_p, n1, p1, (N_t_sum_m3 if N_t_sum_m3 > 0.0 else None)
