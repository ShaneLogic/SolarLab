"""Phase E3 — thermionic-emission flux primitive for interface-plane state.

Implements Pauwels-Vanhoutte 1978 (J.Phys.D 11, 649-667) eq 14a/b
translated to SolarLab's n+p heterojunction convention.

For each heterointerface k, four state-vec densities live on the
interface plane: (n_1s, p_1s, n_2s, p_2s). They couple to the bulk
nodes on each side via a thermionic-emission (TE) flux:

  J_TE = v_th_eff * (density_bulk_projected - density_state)

where ``density_bulk_projected`` is the equilibrium bulk density on the
appropriate side, Boltzmann-projected to the interface plane via the
local band-bending V_1 or V_2 partition of (V_bi - V_app).

Sign convention: positive flux means carriers flow INTO the interface-
plane state (state density grows). The Sprint 6 Day 8-10 work then
applies these fluxes as source terms on the dy/dt vector for the
iface_state block.

Units: m^-3 * m/s = m^-2 s^-1 (surface flux).

The default ``v_th_eff = 1.0e5`` m/s matches SCAPS Manual sec 3.8
typical thermal velocity (~ 10^7 cm/s). Per-layer override via the
``v_th_eff`` kwarg; future Sprint 8 will plumb this through the YAML
loader.
"""
from __future__ import annotations

import math

import numpy as np

from perovskite_sim.constants import V_T as _V_T_300


_DEFAULT_V_TH_MS = 1.0e5  # m/s typical electron / hole thermal velocity
_EXP_CAP = 30.0           # cap exponent to avoid overflow at V_bi/V_T ~ 42


def te_flux(
    density_bulk_projected: float,
    density_state: float,
    v_th: float,
) -> float:
    """Pure-function TE flux primitive (paper eq 14 reduced).

    Returns surface flux INTO the interface-plane state [m^-2 s^-1].
    Positive when bulk-projected > state (state fills); negative when
    state > bulk-projected (state drains).
    """
    return float(v_th) * (float(density_bulk_projected) - float(density_state))


def compute_interface_te_fluxes(
    mat,
    iface_state: np.ndarray,
    V_app: float = 0.0,
    *,
    v_th_eff: float = _DEFAULT_V_TH_MS,
    V_T: float | None = None,
) -> np.ndarray:
    """Compute all 4*N_iface TE fluxes for one RHS evaluation.

    Per interface k, the block layout is (n_1s, p_1s, n_2s, p_2s)
    matching ``StateVec.iface_state`` and ``_compute_iface_state_dark_eq``.

    Bulk-projected densities use cached equilibrium values plus the
    V_app-dependent band-bending partition:

      V_total = V_bi_eff - V_app  (clamped to >= 0)
      V_2     = partition_left[k] * V_total       # PVK (light) side
      V_1     = (1 - partition_left[k]) * V_total # ETL (heavy) side

      n_1s_bulk_proj = n_R_eq * exp(-V_1 / V_T)
      p_1s_bulk_proj = p_R_eq * exp(+V_1 / V_T)
      n_2s_bulk_proj = n_L_eq * exp(+V_2 / V_T)
      p_2s_bulk_proj = p_L_eq * exp(-V_2 / V_T)

    At dark equilibrium (V_app = 0, iface_state = dark-eq init) the
    bulk-projected equals the state on every entry -> all fluxes = 0.

    Returns array shape (4 * N_iface,).
    """
    n_iface = len(mat.interface_V_partition_2)
    if n_iface == 0:
        return np.zeros(0, dtype=float)
    V_T_local = V_T if V_T is not None else (
        mat.V_T_device if hasattr(mat, "V_T_device") else _V_T_300
    )
    V_total = max(0.0, float(mat.V_bi_eff) - float(V_app))
    out = np.zeros(4 * n_iface, dtype=float)
    for k in range(n_iface):
        partition_left = float(mat.interface_V_partition_2[k])
        V_2 = partition_left * V_total
        V_1 = (1.0 - partition_left) * V_total
        v1_norm = max(-_EXP_CAP, min(_EXP_CAP, V_1 / V_T_local))
        v2_norm = max(-_EXP_CAP, min(_EXP_CAP, V_2 / V_T_local))
        n_R = float(mat.interface_n_R_eq[k])
        p_R = float(mat.interface_p_R_eq[k])
        n_L = float(mat.interface_n_L_eq[k])
        p_L = float(mat.interface_p_L_eq[k])
        n_1s_proj = n_R * math.exp(-v1_norm)
        p_1s_proj = p_R * math.exp(+v1_norm)
        n_2s_proj = n_L * math.exp(+v2_norm)
        p_2s_proj = p_L * math.exp(-v2_norm)
        base = 4 * k
        n_1s = float(iface_state[base + 0])
        p_1s = float(iface_state[base + 1])
        n_2s = float(iface_state[base + 2])
        p_2s = float(iface_state[base + 3])
        out[base + 0] = te_flux(n_1s_proj, n_1s, v_th_eff)
        out[base + 1] = te_flux(p_1s_proj, p_1s, v_th_eff)
        out[base + 2] = te_flux(n_2s_proj, n_2s, v_th_eff)
        out[base + 3] = te_flux(p_2s_proj, p_2s, v_th_eff)
    return out
