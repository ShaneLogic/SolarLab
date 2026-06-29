"""Plot SolarLab band diagrams (equilibrium + a bias) from a config.

Uses experiments.band_diagram.compute_band_diagram — the SCAPS Energy-Bands-Panel
equivalent. At equilibrium the quasi-Fermi levels coincide into a single flat E_F;
under bias they split by ~qV. Publication-quality output (Arial, 300 dpi).

    python scripts/plot_band_diagram.py [config.yaml] [V_bias]
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
from perovskite_sim.experiments.band_diagram import compute_band_diagram
from perovskite_sim.models.device import electrical_layers

cfg = Path(sys.argv[1]) if len(sys.argv) > 1 else (
    Path(__file__).resolve().parents[1] / "configs" / "scaps_mirror_v2.yaml")
V_bias = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0

stack = load_scaps_yaml(cfg)
eq = compute_band_diagram(stack, 0.0, illuminated=False)
bias = compute_band_diagram(stack, V_bias, illuminated=True)

elec = electrical_layers(stack)
edges_nm = np.concatenate([[0.0], np.cumsum([L.thickness for L in elec])]) * 1e9
names = ["HTL", "perovskite", "ETL"] if len(elec) == 3 else [f"L{i+1}" for i in range(len(elec))]

C_EDGE, C_FN, C_FP = "0.1", "#1f6fb4", "#c0392b"
fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), sharey=True)


def draw(ax, bd, title, tag, annotate_qv=False):
    xn = bd.x * 1e9
    # tint the thin transport layers; perovskite stays white
    shade = {0: "#4e79a7", len(names) - 1: "#e1812c"}
    for i in shade:
        ax.axvspan(edges_nm[i], edges_nm[i + 1], color=shade[i], alpha=0.10, zorder=0)
    ax.fill_between(xn, bd.E_C, bd.E_V, color="0.94", zorder=0.5)
    ax.plot(xn, bd.E_C, "-", color=C_EDGE, lw=2.2, label="$E_C$")
    ax.plot(xn, bd.E_V, "-", color=C_EDGE, lw=2.2, label="$E_V$")
    ax.plot(xn, bd.E_Fn, "--", color=C_FN, lw=2.0, label="$E_{Fn}$")
    ax.plot(xn, bd.E_Fp, ":", color=C_FP, lw=2.4, label="$E_{Fp}$")
    for xe in edges_nm[1:-1]:
        ax.axvline(xe, color="0.55", lw=0.7, ls=(0, (4, 3)), zorder=1)
    # layer-region labels just above the frame (no collision with the (a)/(b) tag)
    for i, nm in enumerate(names):
        xc = 0.5 * (edges_nm[i] + edges_nm[i + 1])
        ax.text(xc, 1.015, nm, transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=9.5, color="0.35", clip_on=False)
    if annotate_qv:
        xa = 0.55 * edges_nm[-1]
        ja = int(np.argmin(np.abs(xn - xa)))
        yfn, yfp = bd.E_Fn[ja], bd.E_Fp[ja]
        ax.annotate("", xy=(xa, yfn), xytext=(xa, yfp),
                    arrowprops=dict(arrowstyle="<->", color="0.2", lw=1.3))
        ax.text(xa + 18, 0.5 * (yfn + yfp), "$\\Delta E_F = qV$",
                fontsize=11, va="center", color="0.2")
    ax.set_xlabel("Depth (nm)")
    ax.set_xlim(xn.min(), xn.max())
    ax.set_title(title, fontsize=12.5, pad=16)
    ax.text(0.025, 0.95, tag, transform=ax.transAxes, fontsize=13,
            fontweight="bold", va="top", ha="left")
    # Upper-right: clears the bands in both panels — the dark panel's E_C
    # descends to the right (open triangle above it) and the illuminated panel's
    # E_C sits flat near -4.3 with empty space above. center-left overlapped the
    # E_C / E_V curves.
    ax.legend(loc="upper right", fontsize=11, handlelength=1.9,
              borderpad=0.5, labelspacing=0.4)


draw(axes[0], eq, "Dark equilibrium", "(a)")
draw(axes[1], bias, f"Illuminated, $V$ = {V_bias:.2f} V", "(b)", annotate_qv=True)
axes[0].set_ylabel("Energy (eV)")
fig.tight_layout()
out = Path("band_diagram.png")
fig.savefig(out, dpi=300, bbox_inches="tight")
print("wrote", out.resolve())
