"""Generation-recombination balance + quasi-Fermi splitting (partner device).

(a) G(x) vs R(x) at MPP — the optical generation profile (TMM) against the
    mechanism-summed recombination loss; the shaded G>R band is the net carrier
    surplus available for collection, and where R rises to meet G marks a loss
    hotspot. (b) quasi-Fermi-level splitting (E_Fn-E_Fp)(x) at several biases —
    it grows toward qV in the absorber (dashed reference lines), the direct
    spatial picture of the photovoltage building up. Publication layout (Arial).

    python scripts/plot_generation_recombination.py [config.yaml]
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
from perovskite_sim.experiments.jv_sweep import extract_spatial_snapshot
from perovskite_sim.experiments.steady_state import _grid_for
from perovskite_sim.experiments.band_diagram import compute_band_diagram
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


def despiked(n, p):
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


def region_labels(ax):
    for i, nm in enumerate(names):
        xc = 0.5 * (edges_nm[i] + edges_nm[i + 1])
        ax.text(xc, 1.02, nm, transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=9, color="0.35", clip_on=False)


fig, (aGR, aQF) = plt.subplots(1, 2, figsize=(11.5, 4.8))

# (a) G-R balance at MPP --------------------------------------------------------
V_mpp = 1.06
y = solve_illuminated_ss(x, stack, V_mpp, t_settle=1.0e-2)
s = extract_spatial_snapshot(x, y, stack, V_mpp, mat=mat)
nr, pr = despiked(s.n, s.p)
R = (srh_recombination(nr, pr, mat.ni_sq, mat.tau_n, mat.tau_p, mat.n1, mat.p1)
     + radiative_recombination(nr, pr, mat.ni_sq, mat.B_rad)
     + auger_recombination(nr, pr, mat.ni_sq, mat.C_n, mat.C_p))
G = np.asarray(mat.G_optical, dtype=float)
cs = 1.0e-6  # m^-3 s^-1 -> cm^-3 s^-1
shade_layers(aGR)
Gp = np.maximum(G, 1e-30) * cs
Rp = np.maximum(R, 1e-30) * cs
aGR.fill_between(xn, Rp, Gp, where=Gp > Rp, color="#2e8b57", alpha=0.18, zorder=2,
                 label="net  $G>R$ (collected)")
aGR.semilogy(xn, Gp, "-", color="#1f6fb4", lw=2.0, label="generation $G$ (TMM)", zorder=3)
aGR.semilogy(xn, Rp, "--", color="#c0392b", lw=2.0, label="recombination $R$", zorder=3)
aGR.set_xlabel("Depth (nm)")
aGR.set_ylabel(r"rate (cm$^{-3}$ s$^{-1}$)")
aGR.set_xlim(xn.min(), xn.max())
aGR.set_ylim(1e11, np.nanmax(Gp) * 50)   # hide the R->0 floor plunges at the contacts
aGR.set_title(f"generation vs recombination  ($V$={V_mpp:.2f} V)", fontsize=12, pad=14)
aGR.text(0.025, 0.95, "(a)", transform=aGR.transAxes, fontsize=13, fontweight="bold", va="top")
aGR.legend(loc="lower center", fontsize=8.5)
region_labels(aGR)

# (b) quasi-Fermi splitting vs bias --------------------------------------------
shade_layers(aQF)
biases = [0.0, 0.4, 0.8, 1.06]
cols = ["#1f6fb4", "#2e8b57", "#e6842a", "#c0392b"]
for V, col in zip(biases, cols):
    bd = compute_band_diagram(stack, V, illuminated=True, N_grid=60)
    split = bd.E_Fn - bd.E_Fp
    aQF.plot(bd.x * 1e9, split, "-", color=col, lw=1.9, label=f"$V$={V:.2f} V")
    print(f"QFL bias {V}", flush=True)
aQF.set_xlabel("Depth (nm)")
aQF.set_ylabel(r"$E_{Fn}-E_{Fp}$  (eV)")
aQF.set_xlim(xn.min(), xn.max())
aQF.set_ylim(0, 1.25)
aQF.set_title("quasi-Fermi splitting (implied voltage)", fontsize=12, pad=14)
aQF.text(0.5, 0.93, "flat $=qV$ at open circuit;  rises + flattens with bias",
         transform=aQF.transAxes, ha="center", fontsize=8.5, color="0.25")
aQF.text(0.025, 0.95, "(b)", transform=aQF.transAxes, fontsize=13, fontweight="bold", va="top")
aQF.legend(loc="lower left", fontsize=9, ncol=2)
region_labels(aQF)

fig.suptitle("SolarLab collection physics — scaps_mirror_v2 (SCAPS-1D partner stack)",
             fontsize=12.5, y=1.0)
fig.tight_layout(rect=(0, 0, 1, 0.96))
out = Path("generation_recombination.png")
fig.savefig(out, dpi=300, bbox_inches="tight")
print("wrote", out.resolve())
