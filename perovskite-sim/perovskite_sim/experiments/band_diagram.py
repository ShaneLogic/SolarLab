"""Band-diagram extraction: physical band edges and quasi-Fermi levels.

The SCAPS Energy-Bands-Panel equivalent. Returns the conduction/valence band
edges and the electron/hole quasi-Fermi levels (all in eV) as a function of depth
at a chosen bias. The state is settled to steady state first, so the quasi-Fermi
levels are physically meaningful: flat and coincident at equilibrium (zero
current), split by ~qV under bias.

Band edges use the *physical* per-layer electron affinity and effective DOS (not
the DOS-folded transport arrays); with the effective-DOS band-potential fold
active, this makes ``E_Fn = -phi - chi + V_T ln(n/N_C)`` flat at the zero-current
equilibrium (verified). Quasi-Fermi levels are NaN-masked in deep-minority regions
where the carrier density is negligible and the level is numerically ill-defined.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..models.device import DeviceStack, electrical_layers
from ..solver.mol import build_material_arrays, run_transient
from ..solver.newton import solve_equilibrium
from ..solver.illuminated_ss import solve_illuminated_ss
from .jv_sweep import extract_spatial_snapshot
from .steady_state import _grid_for

# Carrier-to-DOS ratio below which a quasi-Fermi level is masked (deep minority,
# where n/N_C ~ 1e-30 makes the level meaningless rather than wrong).
_QFL_MASK = 1.0e-16

# Dark-equilibrium settle ladder: solve_equilibrium returns an unsettled seed
# (non-zero current); a short escalating dark transient relaxes it to the true
# zero-current state so E_F comes out flat.
_DARK_SETTLE = (1.0e-3, 1.0e-2, 1.0e-1)


@dataclass(frozen=True)
class BandDiagram:
    """Band edges and quasi-Fermi levels at a single bias. Energies in eV.

    Node arrays (x, E_C, E_V, E_Fn, E_Fp) have shape (N,). E_Fn / E_Fp are NaN
    where the corresponding carrier is negligible (deep minority).
    """

    x: np.ndarray        # depth [m]
    E_C: np.ndarray      # conduction band edge [eV]
    E_V: np.ndarray      # valence band edge [eV]
    E_Fn: np.ndarray     # electron quasi-Fermi level [eV], NaN in deep minority
    E_Fp: np.ndarray     # hole quasi-Fermi level [eV], NaN in deep minority
    V_app: float         # applied voltage [V]


def _node_band_params(
    x: np.ndarray, stack: DeviceStack
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-node physical (chi, Eg, N_C, N_V) by mapping each node to its layer."""
    elec = electrical_layers(stack)
    edges = np.concatenate([[0.0], np.cumsum([L.thickness for L in elec])])
    li = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(elec) - 1)

    def col(attr: str) -> np.ndarray:
        return np.array([getattr(elec[i].params, attr) for i in li], dtype=float)

    chi, Eg, Nc, Nv = col("chi"), col("Eg"), col("Nc300"), col("Nv300")
    if not (np.all(Nc > 0.0) and np.all(Nv > 0.0)):
        raise ValueError(
            "band_diagram requires per-layer Nc300/Nv300 (effective DOS); this "
            "config does not provide it (set N_C_cm3/N_V_cm3 or Nc300/Nv300)."
        )
    return chi, Eg, Nc, Nv


def compute_band_diagram(
    stack: DeviceStack,
    V_app: float = 0.0,
    *,
    illuminated: bool = True,
    N_grid: int = 80,
    settle_t: float = 1.0e-2,
) -> BandDiagram:
    """Return the settled-state band diagram at ``V_app``.

    ``illuminated`` selects the illuminated steady state (``solve_illuminated_ss``)
    or the dark state. At ``V_app == 0`` with ``illuminated=False`` this is the
    zero-current equilibrium (E_F flat). ``settle_t`` is the illuminated settling
    time; the default 1e-2 s settles the carriers to quasi-steady (ions ~frozen,
    as in a snapshot) and is fast — raise it for full ionic equilibrium. Raises
    ``ValueError`` if the config lacks effective-DOS data.
    """
    x = _grid_for(stack, N_grid)
    mat = build_material_arrays(x, stack)
    chi, Eg, Nc, Nv = _node_band_params(x, stack)

    if illuminated:
        y = solve_illuminated_ss(x, stack, V_app, t_settle=settle_t)
    else:
        y = solve_equilibrium(x, stack)
        for t in _DARK_SETTLE:
            sol = run_transient(
                x, y, (0.0, t), np.array([t]), stack,
                illuminated=False, V_app=V_app, mat=mat, max_step=t / 8.0,
            )
            if sol.success:
                y = sol.y[:, -1]

    s = extract_spatial_snapshot(x, y, stack, V_app, mat=mat)
    V_T = mat.V_T_device
    n = np.maximum(s.n, 1.0e-30)
    p = np.maximum(s.p, 1.0e-30)
    E_C = -s.phi - chi
    E_V = E_C - Eg
    E_Fn = np.where(n / Nc > _QFL_MASK, E_C + V_T * np.log(n / Nc), np.nan)
    E_Fp = np.where(p / Nv > _QFL_MASK, E_V - V_T * np.log(p / Nv), np.nan)
    return BandDiagram(
        x=x.copy(), E_C=E_C, E_V=E_V, E_Fn=E_Fn, E_Fp=E_Fp, V_app=float(V_app),
    )
