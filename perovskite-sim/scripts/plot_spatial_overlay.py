"""Academic 4-panel spatial overlay (SC / MPP / Voc) for a device config.

Settles the illuminated steady state at three operating points and overlays the
depth profiles: (a) band edges E_C/E_V, (b) carrier density n (solid) / p
(dashed), (c) net space charge rho/q, (d) electric field. Curves are smoothed
per layer with monotone (PCHIP) interpolation — the bulk mesh is sparse, so this
fills between true nodes WITHOUT smearing the real band-offset jumps at the
interfaces (each layer is drawn as its own segment). Publication layout (Arial).

    python scripts/plot_spatial_overlay.py [config.yaml]
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import SymmetricalLogLocator, FixedLocator
from scipy.interpolate import PchipInterpolator

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
from perovskite_sim.constants import Q
from perovskite_sim.solver.mol import build_material_arrays
from perovskite_sim.solver.illuminated_ss import solve_illuminated_ss
from perovskite_sim.experiments.jv_sweep import extract_spatial_snapshot
from perovskite_sim.experiments.steady_state import _grid_for
from perovskite_sim.experiments.band_diagram import _node_band_params

cfg = Path(sys.argv[1]) if len(sys.argv) > 1 else (
    Path(__file__).resolve().parents[1] / "configs" / "scaps_mirror_v2.yaml")
stack = load_scaps_yaml(cfg)

x = _grid_for(stack, 100)
mat = build_material_arrays(x, stack)
chi, Eg, _, _ = _node_band_params(x, stack)
elec = electrical_layers(stack)
edges_nm = np.concatenate([[0.0], np.cumsum([L.thickness for L in elec])]) * 1e9
names = ["HTL", "perovskite", "ETL"] if len(elec) == 3 else [f"L{i+1}" for i in range(len(elec))]
shade = {0: "#4e79a7", len(names) - 1: "#e1812c"}
xn = x * 1e9

BIAS = {
    "SC":  (0.0,  "#1f6fb4", "SC (V=0)"),
    "MPP": (1.06, "#2e8b57", "MPP (V=1.06)"),
    "Voc": (1.17, "#c0392b", r"V$_{oc}$ (V=1.17)"),
}

D = {}
for tag, (V, _c, _l) in BIAS.items():
    y = solve_illuminated_ss(x, stack, V, t_settle=1.0e-2)
    s = extract_spatial_snapshot(x, y, stack, V, mat=mat)
    E_C = -s.phi - chi
    D[tag] = {"E_C": E_C, "E_V": E_C - Eg, "n": s.n * 1e-6, "p": s.p * 1e-6,
              "rho_q": s.rho / Q * 1e-6, "phi": s.phi}
    print(f"settled {tag} (V={V})", flush=True)


def smooth(xnodes, y, *, log=False, n=240):
    """Per-layer PCHIP upsample; yields (xd, yd) per layer (jumps preserved)."""
    out = []
    for lo, hi in zip(edges_nm[:-1], edges_nm[1:]):
        m = (xnodes >= lo - 1e-6) & (xnodes <= hi + 1e-6)
        xs, ys = xnodes[m], y[m]
        xu, idx = np.unique(xs, return_index=True)
        yu = ys[idx]
        if log:
            ok = yu > 0
            xu, yu = xu[ok], yu[ok]
        if len(xu) < 2:
            continue
        xd = np.linspace(xu[0], xu[-1], n)
        yd = 10.0 ** PchipInterpolator(xu, np.log10(yu))(xd) if log else PchipInterpolator(xu, yu)(xd)
        out.append((xd, yd))
    return out


def plot_curve(ax, xnodes, y, col, *, ls="-", lw=1.8, log=False, semilog=False, label=None):
    first = True
    for xd, yd in smooth(xnodes, y, log=log):
        (ax.semilogy if semilog else ax.plot)(xd, yd, ls, color=col, lw=lw,
                                              label=(label if first else None))
        first = False


def shade_layers(ax):
    for i in shade:
        ax.axvspan(edges_nm[i], edges_nm[i + 1], color=shade[i], alpha=0.09, zorder=0)
    for xe in edges_nm[1:-1]:
        ax.axvline(xe, color="0.55", lw=0.7, ls=(0, (4, 3)), zorder=1)


def region_labels(ax):
    for i, nm in enumerate(names):
        xc = 0.5 * (edges_nm[i] + edges_nm[i + 1])
        ax.text(xc, 1.02, nm, transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=9, color="0.35", clip_on=False)


fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2), sharex=True)
(aB, aN), (aR, aE) = axes

for tag, (V, col, lab) in BIAS.items():
    plot_curve(aB, xn, D[tag]["E_C"], col, label=lab)
    plot_curve(aB, xn, D[tag]["E_V"], col)
aB.set_ylabel("Energy (eV)")
aB.set_title("band edges  $E_C$, $E_V$", fontsize=12, pad=14)
aB.text(0.025, 0.95, "(a)", transform=aB.transAxes, fontsize=13, fontweight="bold", va="top")
aB.legend(loc="upper center", fontsize=9, ncol=3, columnspacing=1.1, handlelength=1.6,
          borderpad=0.4, handletextpad=0.5)

for tag, (V, col, lab) in BIAS.items():
    plot_curve(aN, xn, D[tag]["n"], col, log=True, semilog=True)
    plot_curve(aN, xn, D[tag]["p"], col, ls="--", log=True, semilog=True)
aN.set_ylabel(r"density (cm$^{-3}$)")
aN.set_title("carrier density  ($n$ solid, $p$ dashed)", fontsize=12, pad=14)
aN.text(0.025, 0.95, "(b)", transform=aN.transAxes, fontsize=13, fontweight="bold", va="top")
aN.legend(handles=[Line2D([], [], color="0.2", ls="-", label="$n$ (electrons)"),
                   Line2D([], [], color="0.2", ls="--", label="$p$ (holes)")],
          loc="lower center", fontsize=9, ncol=2)

for tag, (V, col, lab) in BIAS.items():
    plot_curve(aR, xn, D[tag]["rho_q"], col, lw=1.6)
aR.axhline(0, color="0.6", lw=0.8)
aR.set_yscale("symlog", linthresh=1.0e14)
aR.set_ylim(-1.0e19, 1.0e19)
aR.set_ylabel(r"net charge  $\rho/q$  (cm$^{-3}$)")
aR.set_xlabel("Depth (nm)")
aR.set_title("space charge", fontsize=12, pad=4)
aR.text(0.025, 0.95, "(c)", transform=aR.transAxes, fontsize=13, fontweight="bold", va="top")

for tag, (V, col, lab) in BIAS.items():
    phi = D[tag]["phi"]
    xf = 0.5 * (xn[1:] + xn[:-1])
    E = -(np.diff(phi) / (np.diff(x) * 1e-9))
    plot_curve(aE, xf, E, col, lw=1.6)
aE.axhline(0, color="0.6", lw=0.8)
aE.set_yscale("symlog", linthresh=1.0e5)
aE.set_ylim(-1.0e16, 1.0e16)
# Label only well-separated decades (every 3) ABOVE the 1e5 linear threshold,
# plus zero. The auto SymmetricalLogLocator(base=100) emits powers of 100 —
# including 1e0/1e2/1e4, which sit *below* linthresh in the compressed near-zero
# linear band and pile their labels on top of each other. A FixedLocator skips
# that band; minor decade marks stay (unlabeled) for scale.
_E_MAJOR = [0.0] + [s * 10.0 ** e for e in (6, 9, 12, 15) for s in (1.0, -1.0)]
aE.yaxis.set_major_locator(FixedLocator(_E_MAJOR))
aE.yaxis.set_minor_locator(SymmetricalLogLocator(base=10.0, linthresh=1.0e5))
aE.set_ylabel(r"electric field  (V m$^{-1}$)")
aE.set_xlabel("Depth (nm)")
aE.set_title("electric field", fontsize=12, pad=4)
aE.text(0.025, 0.95, "(d)", transform=aE.transAxes, fontsize=13, fontweight="bold", va="top")

for ax in axes.ravel():
    shade_layers(ax)
    ax.set_xlim(edges_nm.min(), edges_nm.max())
for ax in (aB, aN):
    region_labels(ax)

fig.suptitle("SolarLab spatial profiles — scaps_mirror_v2 (HTL | perovskite | ETL);  "
             "band edges are electrostatic ($E_C$, $E_V$)", fontsize=12.5, y=0.99)
fig.tight_layout(rect=(0, 0, 1, 0.975))
out = Path("spatial_overlay.png")
fig.savefig(out, dpi=300, bbox_inches="tight")
print("wrote", out.resolve())
