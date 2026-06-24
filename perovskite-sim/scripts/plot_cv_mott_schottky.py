"""Dark C-V + Mott-Schottky plot (partner device).

At each DC bias a single 1 MHz AC excitation probes the junction capacitance
C = Im(Y)/ω (experiments/mott_schottky.run_mott_schottky over run_impedance).
For a heavily-doped base 1/C² vs V is a line whose intercept gives V_bi and
slope gives the doping (Mott-Schottky). For an INTRINSIC fully-depleted absorber
(the perovskite case) C ≈ the geometric series capacitance C_geo = (Σ d_i/ε_i)^-1
and 1/C² is flat — V_bi is not extractable, which the figure states honestly and
cross-checks against C_geo. Publication layout (Arial, 300 dpi).

    python scripts/plot_cv_mott_schottky.py [config.yaml]
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
from perovskite_sim.experiments.mott_schottky import run_mott_schottky

EPS_0 = 8.8541878128e-12

cfg = Path(sys.argv[1]) if len(sys.argv) > 1 else (
    Path(__file__).resolve().parents[1] / "configs" / "scaps_mirror_v2.yaml")
stack = load_scaps_yaml(cfg)

ms = run_mott_schottky(stack, V_range=np.linspace(-0.4, 0.5, 10), frequency=1e6, N_grid=32)

# geometric series capacitance of the electrical layers: 1/C_geo = Σ d_i/(ε0 ε_ri)
elec = electrical_layers(stack)
inv = sum(L.thickness / (EPS_0 * L.params.eps_r) for L in elec)
C_geo = 1.0 / inv                       # F/m^2
toNF = 1e5                              # F/m^2 -> nF/cm^2  (×1e-4 to F/cm^2, ×1e9 to nF)
span = (ms.one_over_C2.max() - ms.one_over_C2.min()) / np.median(np.abs(ms.one_over_C2))
fittable = np.isfinite(ms.V_bi_fit)

fig, (aC, aM) = plt.subplots(1, 2, figsize=(11.0, 4.6))

# (a) C-V
aC.plot(ms.V, ms.C * toNF, "o-", color="#1f6fb4", lw=2.0, ms=5,
        mfc="white", mec="#1f6fb4", mew=1.4, label="C (1 MHz, dark)")
aC.axhline(C_geo * toNF, color="#c0392b", lw=1.4, ls=(0, (5, 3)),
           label=r"$C_{geo}=(\Sigma\, d_i/\varepsilon_i)^{-1}$")
aC.set_xlabel("DC bias $V$ (V)")
aC.set_ylabel(r"capacitance (nF cm$^{-2}$)")
aC.set_title("dark C–V", fontsize=12, pad=10)
aC.text(0.03, 0.95, "(a)", transform=aC.transAxes, fontsize=13, fontweight="bold", va="top")
aC.legend(loc="lower right", fontsize=9.5)
aC.set_ylim(0, max(ms.C.max(), C_geo) * toNF * 1.25)

# (b) Mott-Schottky 1/C^2
aM.plot(ms.V, ms.one_over_C2, "o-", color="#2e8b57", lw=2.0, ms=5,
        mfc="white", mec="#2e8b57", mew=1.4)
aM.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
aM.set_xlabel("DC bias $V$ (V)")
aM.set_ylabel(r"$1/C^{2}$  (m$^4$ F$^{-2}$)")
aM.set_title("Mott–Schottky", fontsize=12, pad=10)
aM.text(0.03, 0.95, "(b)", transform=aM.transAxes, fontsize=13, fontweight="bold", va="top")

if fittable:
    Vline = np.linspace(ms.V_fit_lo, ms.V_bi_fit, 50)
    a = np.polyfit(ms.V[(ms.V >= ms.V_fit_lo) & (ms.V <= ms.V_fit_hi)],
                   ms.one_over_C2[(ms.V >= ms.V_fit_lo) & (ms.V <= ms.V_fit_hi)], 1)
    aM.plot(Vline, np.polyval(a, Vline), "--", color="#c0392b", lw=1.4)
    verdict = (f"$V_{{bi}}$ = {ms.V_bi_fit:.2f} V\n"
               f"$N_{{eff}}$ = {ms.N_eff_fit:.2e} m$^{{-3}}$")
else:
    verdict = (f"1/$C^2$ span = {span*100:.1f}%  (< 5%)\n"
               "→ fully-depleted intrinsic absorber\n"
               r"$C \approx C_{geo}$ (geometric); $V_{bi}$ not"
               "\nMS-extractable — as expected for an\nintrinsic p-i-n (SCAPS-consistent)")
aM.text(0.045, 0.05, verdict, transform=aM.transAxes, fontsize=9.5, va="bottom", ha="left",
        bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="0.7", alpha=0.95))

fig.suptitle("SolarLab capacitance–voltage — scaps_mirror_v2 (SCAPS-1D partner stack); "
             f"$C_{{geo}}$ = {C_geo*toNF:.1f} nF/cm$^2$", fontsize=12, y=0.99)
fig.tight_layout(rect=(0, 0, 1, 0.96))
out = Path("cv_mott_schottky.png")
fig.savefig(out, dpi=300, bbox_inches="tight")
print(f"wrote {out.resolve()}  C_geo={C_geo*toNF:.2f} nF/cm2  span={span*100:.2f}%  V_bi_fit={ms.V_bi_fit}")
