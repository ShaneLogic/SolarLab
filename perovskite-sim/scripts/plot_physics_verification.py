"""Physics-verification figure: spatial current conservation + recombination.

Two depth-resolved checks the SolarLab figures did not show before, both
extracted from the settled steady state (columns = short-circuit and MPP):

  Top row  — current-density components Jn(x), Jp(x), Jion(x) and their sum.
             At steady state charge conservation forces Jn+Jp+Jion = const(x);
             the flat total line IS that invariant (residual printed).
  Bottom   — bulk recombination R(x) resolved by mechanism (SRH / radiative /
             Auger), reproduced with the SAME rate code + het-despike blend the
             solver integrates (physics/recombination.py, continuity.py:203-218).

    python scripts/plot_physics_verification.py [config.yaml]
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "Arial", "font.size": 12,
    "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
    "axes.linewidth": 1.0, "axes.grid": False,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.minor.visible": True, "ytick.minor.visible": True,
    "xtick.top": True, "ytick.right": True,
    "legend.frameon": True, "legend.framealpha": 0.95, "legend.edgecolor": "0.7",
})

from perovskite_sim.scaps_compat import load_scaps_yaml
from perovskite_sim.models.device import electrical_layers
from perovskite_sim.solver.mol import build_material_arrays
from perovskite_sim.solver.illuminated_ss import solve_illuminated_ss
from perovskite_sim.experiments.jv_sweep import extract_spatial_snapshot, compute_current_components
from perovskite_sim.experiments.steady_state import _grid_for
from perovskite_sim.physics.recombination import (
    srh_recombination, radiative_recombination, auger_recombination,
)

cfg = Path(sys.argv[1]) if len(sys.argv) > 1 else (
    Path(__file__).resolve().parents[1] / "configs" / "scaps_mirror_v2.yaml")

stack = load_scaps_yaml(cfg)
x = _grid_for(stack, 100)
mat = build_material_arrays(x, stack)
elec = electrical_layers(stack)
edges_nm = np.concatenate([[0.0], np.cumsum([L.thickness for L in elec])]) * 1e9
names = ["HTL", "perovskite", "ETL"] if len(elec) == 3 else [f"L{i+1}" for i in range(len(elec))]
shade = {0: "#4e79a7", len(names) - 1: "#e1812c"}
xn = x * 1e9
xf = 0.5 * (xn[1:] + xn[:-1])

COLS = [("Short circuit", 0.0), ("Maximum power", 1.06)]


def despiked(n, p):
    """Reproduce continuity.py's het-despike density blend for the bulk rate."""
    if mat.het_recomb_despike <= 0.0 or not mat.het_recomb_nodes:
        return n, p
    nr, pr = n.copy(), p.copy()
    f = mat.het_recomb_despike
    for i in mat.het_recomb_nodes:
        if 0 < i < len(nr) - 1:
            nb_n = np.sqrt(max(n[i - 1], 1.0) * max(n[i + 1], 1.0))
            nb_p = np.sqrt(max(p[i - 1], 1.0) * max(p[i + 1], 1.0))
            nr[i] = nb_n + (n[i] - nb_n) * (1.0 - f)
            pr[i] = nb_p + (p[i] - nb_p) * (1.0 - f)
    return nr, pr


def shade_layers(ax):
    for i in shade:
        ax.axvspan(edges_nm[i], edges_nm[i + 1], color=shade[i], alpha=0.09, zorder=0)
    for xe in edges_nm[1:-1]:
        ax.axvline(xe, color="0.55", lw=0.7, ls=(0, (4, 3)), zorder=1)


fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2), sharex=True)
tags = [["(a)", "(b)"], ["(c)", "(d)"]]

for j, (label, V) in enumerate(COLS):
    y = solve_illuminated_ss(x, stack, V, t_settle=1.0e-2)

    # --- top: current components (A/m^2 -> mA/cm^2) ---
    cc = compute_current_components(x, y, stack, V, mat=mat)
    sc = 0.1  # A/m^2 -> mA/cm^2
    aJ = axes[0, j]
    shade_layers(aJ)
    aJ.axhline(0, color="0.6", lw=0.8)
    aJ.plot(xf, cc.J_n * sc, "-", color="#1f6fb4", lw=1.8, label="$J_n$ (electron)")
    aJ.plot(xf, cc.J_p * sc, "-", color="#c0392b", lw=1.8, label="$J_p$ (hole)")
    aJ.plot(xf, cc.J_ion * sc, "-", color="#2e8b57", lw=1.6, label="$J_{ion}$")
    aJ.plot(xf, cc.J_total * sc, "--", color="0.1", lw=2.0, label="$J_{total}$")
    # conservation residual over faces away from every interface: the ~5 faces
    # at each heterojunction/contact carry a TE-cap SG-reconstruction artifact
    # (the integrated transient is conserved; the post-hoc flux at the capped
    # face is not). 25 nm buffer -> the honest quasi-neutral-bulk conservation.
    BUF = 25.0
    keep = np.ones_like(xf, dtype=bool)
    for e in edges_nm:
        keep &= np.abs(xf - e) > BUF
    Jt = cc.J_total[keep]
    resid = np.std(Jt) / max(abs(np.mean(Jt)), 1e-30)
    Jb = abs(np.mean(Jt)) * sc   # mA/cm^2 (plotted units)
    aJ.set_ylim(-0.22 * Jb, 1.35 * Jb)
    aJ.set_title(f"{label}  ($V$={V:.2f} V)", fontsize=12, pad=14)
    aJ.text(0.025, 0.95, tags[0][j], transform=aJ.transAxes, fontsize=13, fontweight="bold", va="top")
    aJ.text(0.5, 0.045, r"bulk $\sigma(J_{tot})/\langle J_{tot}\rangle$ = " + f"{resid:.1e}"
            + "  (interfaces excl.)", transform=aJ.transAxes, ha="center", fontsize=8.5, color="0.25")
    if j == 0:
        aJ.set_ylabel(r"current density (mA cm$^{-2}$)")
        aJ.legend(loc="center left", fontsize=8.5, ncol=1)
    for i, nm in enumerate(names):
        xc = 0.5 * (edges_nm[i] + edges_nm[i + 1])
        aJ.text(xc, 1.02, nm, transform=aJ.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=9, color="0.35", clip_on=False)

    # --- bottom: recombination by mechanism (m^-3 s^-1 -> cm^-3 s^-1) ---
    s = extract_spatial_snapshot(x, y, stack, V, mat=mat)
    nr, pr = despiked(s.n, s.p)
    R_srh = srh_recombination(nr, pr, mat.ni_sq, mat.tau_n, mat.tau_p, mat.n1, mat.p1)
    R_rad = radiative_recombination(nr, pr, mat.ni_sq, mat.B_rad)
    R_aug = auger_recombination(nr, pr, mat.ni_sq, mat.C_n, mat.C_p)
    cs = 1.0e-6
    aR = axes[1, j]
    shade_layers(aR)
    aR.semilogy(xn, np.maximum(R_srh, 1e-30) * cs, "-", color="#8856a7", lw=1.8, label="SRH")
    aR.semilogy(xn, np.maximum(R_rad, 1e-30) * cs, "-", color="#e6842a", lw=1.8, label="radiative")
    aR.semilogy(xn, np.maximum(R_aug, 1e-30) * cs, "-", color="#3182bd", lw=1.8, label="Auger")
    aR.semilogy(xn, np.maximum(R_srh + R_rad + R_aug, 1e-30) * cs, "--", color="0.1", lw=1.6, label="total")
    aR.set_xlabel("Depth (nm)")
    aR.text(0.025, 0.95, tags[1][j], transform=aR.transAxes, fontsize=13, fontweight="bold", va="top")
    if j == 0:
        aR.set_ylabel(r"recombination $R$ (cm$^{-3}$ s$^{-1}$)")
        aR.legend(loc="lower center", fontsize=8.5, ncol=2)

ylo = min(axes[1, 0].get_ylim()[0], axes[1, 1].get_ylim()[0])
for j in (0, 1):
    axes[1, j].set_ylim(bottom=max(ylo, 1e12))
for ax in axes.ravel():
    ax.set_xlim(xn.min(), xn.max())

fig.suptitle("SolarLab physics verification — current conservation (top) and "
             "mechanism-resolved recombination (bottom); scaps_mirror_v2", fontsize=12.5, y=0.99)
fig.tight_layout(rect=(0, 0, 1, 0.975))
out = Path("physics_verification.png")
fig.savefig(out, dpi=300, bbox_inches="tight")
print("wrote", out.resolve())
