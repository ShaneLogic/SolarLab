"""Plot SolarLab spatial charge distribution: dark vs illuminated.

SolarLab (like SCAPS) is a 1-D device solver, so the "charge distribution" is a
depth profile rho(x), not a 2-D (x,y) map. This plots the settled steady state at
short circuit (V=0) in the dark (equilibrium) and under 1-sun illumination, so the
only difference between the columns is the light. Top row: net space-charge density
rho(x). Bottom row: carrier/ion densities n, p, P. Publication-quality (Arial, 300 dpi).

    python scripts/plot_charge_distribution.py [config.yaml] [V_bias]
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 12,
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
    "axes.linewidth": 1.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.top": True,
    "ytick.right": True,
    "legend.frameon": True,
    "legend.framealpha": 0.95,
    "legend.edgecolor": "0.7",
})

from perovskite_sim.scaps_compat import load_scaps_yaml
from perovskite_sim.models.device import electrical_layers
from perovskite_sim.solver.mol import build_material_arrays, run_transient
from perovskite_sim.solver.newton import solve_equilibrium
from perovskite_sim.solver.illuminated_ss import solve_illuminated_ss
from perovskite_sim.experiments.jv_sweep import extract_spatial_snapshot
from perovskite_sim.experiments.steady_state import _grid_for
from perovskite_sim.constants import Q

_DARK_SETTLE = (1.0e-3, 1.0e-2, 1.0e-1)


def snapshot(stack, x, mat, V_app, *, illuminated):
    """Settled spatial snapshot at V_app (dark = equilibrium + relax)."""
    if illuminated:
        y = solve_illuminated_ss(x, stack, V_app, t_settle=1.0e-2)
    else:
        y = solve_equilibrium(x, stack)
        for t in _DARK_SETTLE:
            sol = run_transient(
                x, y, (0.0, t), np.array([t]), stack,
                illuminated=False, V_app=V_app, mat=mat, max_step=t / 8.0,
            )
            if sol.success:
                y = sol.y[:, -1]
    return extract_spatial_snapshot(x, y, stack, V_app, mat=mat)


cfg = Path(sys.argv[1]) if len(sys.argv) > 1 else (
    Path(__file__).resolve().parents[1] / "configs" / "scaps_mirror_v2.yaml")
V_bias = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0

stack = load_scaps_yaml(cfg)
x = _grid_for(stack, 80)
mat = build_material_arrays(x, stack)
dark = snapshot(stack, x, mat, V_bias, illuminated=False)
light = snapshot(stack, x, mat, V_bias, illuminated=True)

elec = electrical_layers(stack)
edges_nm = np.concatenate([[0.0], np.cumsum([L.thickness for L in elec])]) * 1e9
names = ["HTL", "perovskite", "ETL"] if len(elec) == 3 else [f"L{i+1}" for i in range(len(elec))]
shade = {0: "#4e79a7", len(names) - 1: "#e1812c"}

xn = x * 1e9
fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.4), sharex=True)


def shade_layers(ax):
    for i in shade:
        ax.axvspan(edges_nm[i], edges_nm[i + 1], color=shade[i], alpha=0.10, zorder=0)
    for xe in edges_nm[1:-1]:
        ax.axvline(xe, color="0.55", lw=0.7, ls=(0, (4, 3)), zorder=1)


def draw_rho(ax, s, title, tag):
    shade_layers(ax)
    rho_q = s.rho / Q * 1e-6          # net charge number density [cm^-3]
    ax.axhline(0.0, color="0.6", lw=0.8, zorder=1)
    ax.fill_between(xn, rho_q, 0.0, where=rho_q >= 0, color="#c0392b", alpha=0.35, zorder=2)
    ax.fill_between(xn, rho_q, 0.0, where=rho_q < 0, color="#1f6fb4", alpha=0.35, zorder=2)
    ax.plot(xn, rho_q, "-", color="0.1", lw=1.6, zorder=3)
    # symlog: the contact depletion spikes (~1e19) and the bulk absorber charge
    # (~1e15) differ by 4 decades — a linear axis buries the bulk structure.
    ax.set_yscale("symlog", linthresh=1.0e14)
    ax.set_ylim(-1.0e19, 1.0e19)
    ax.set_ylabel(r"net charge $\rho/q$ (cm$^{-3}$)")
    ax.set_title(title, fontsize=12.5, pad=14)
    ax.text(0.025, 0.94, tag, transform=ax.transAxes, fontsize=13,
            fontweight="bold", va="top", ha="left")
    for i, nm in enumerate(names):
        xc = 0.5 * (edges_nm[i] + edges_nm[i + 1])
        ax.text(xc, 1.015, nm, transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=9.5, color="0.35", clip_on=False)


def draw_dens(ax, s, tag):
    shade_layers(ax)
    ax.semilogy(xn, s.n * 1e-6, "-", color="#1f6fb4", lw=2.0, label="$n$ (electrons)")
    ax.semilogy(xn, s.p * 1e-6, "-", color="#c0392b", lw=2.0, label="$p$ (holes)")
    ax.semilogy(xn, s.P * 1e-6, ":", color="#2e8b57", lw=2.0, label="$P$ (ion vacancies)")
    ax.set_xlabel("Depth (nm)")
    ax.set_ylabel(r"density (cm$^{-3}$)")
    ax.text(0.025, 0.94, tag, transform=ax.transAxes, fontsize=13,
            fontweight="bold", va="top", ha="left")
    ax.legend(loc="lower center", fontsize=9.5, ncol=3, columnspacing=1.0,
              handlelength=1.6, borderpad=0.4)


draw_rho(axes[0, 0], dark, f"Dark (equilibrium), $V$ = {V_bias:.2f} V", "(a)")
draw_rho(axes[0, 1], light, f"Illuminated (1 sun), $V$ = {V_bias:.2f} V", "(b)")
draw_dens(axes[1, 0], dark, "(c)")
draw_dens(axes[1, 1], light, "(d)")
for ax in axes[:, 1]:
    ax.set_ylabel("")
for ax in axes.ravel():
    ax.set_xlim(xn.min(), xn.max())

fig.tight_layout()
out = Path("charge_distribution.png")
fig.savefig(out, dpi=300, bbox_inches="tight")
print("wrote", out.resolve())
