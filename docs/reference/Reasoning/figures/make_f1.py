#!/usr/bin/env python3
"""f1_voc_lever.png — THE SURVIVING ROOT CAUSE (Nd_ETL donor-doping reversal)."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

ROOT = "/Users/shane/Library/CloudStorage/OneDrive-HKUST(Guangzhou)/SolarLab"
OUT = os.path.join(ROOT, "docs/reference/ppt_root_cause/figures/f1_voc_lever.png")

plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 12.5, "axes.labelsize": 12,
    "legend.fontsize": 10.5, "figure.facecolor": "white", "axes.facecolor": "white",
})

# ---- read real data (default path = interface projection OFF) ----
summ = json.load(open(os.path.join(ROOT, "outputs/scaps_full_off/summary.json")))["summary"]
rows = summ["Nd_ETL"]["rows"]
nd_sl, voc_sl, nd_sc, voc_sc = [], [], [], []
for x, vsl, _dir, vsc in rows:
    nd_sc.append(x); voc_sc.append(vsc)            # SCAPS reference: all points
    if vsl and vsl > 0:                             # filter non-converged SolarLab Voc==0
        nd_sl.append(x); voc_sl.append(vsl)

C_SL, C_SC = "#1f5fbf", "#c0392b"

fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.0, 5.0),
                               gridspec_kw={"width_ratios": [1.18, 1.0]})

# ================= LEFT: the lever plot =================
axL.plot(nd_sc, voc_sc, "--", color=C_SC, lw=2.4, marker="s", ms=5.5,
         label="SCAPS-1D (reference)")
axL.plot(nd_sl, voc_sl, "-", color=C_SL, lw=2.6, marker="o", ms=6.0,
         label="SolarLab")
axL.set_xscale("log")
axL.set_xlabel("ETL donor density  N$_{D}$  (cm$^{-3}$)")
axL.set_ylabel("Open-circuit voltage  V$_{oc}$  (V)")
axL.set_title("ETL doping lever on V$_{oc}$", fontsize=12.5, pad=8)
axL.grid(True, which="both", ls=":", alpha=0.45)
axL.set_ylim(1.04, 1.26)
axL.legend(loc="upper left", framealpha=0.95)

# rise annotations
axL.annotate("", xy=(nd_sc[-1], voc_sc[-1]), xytext=(nd_sc[3], voc_sc[3]),
             arrowprops=dict(arrowstyle="->", color=C_SC, lw=1.6,
                             connectionstyle="arc3,rad=-0.15"))
axL.text(3e18, 1.205, "+100 mV\n(rises with N$_{D}$)", color=C_SC,
         ha="center", va="center", fontsize=10.5, fontweight="bold")
# flat callout for SolarLab
axL.annotate("nearly FLAT\n(−0/−11 mV)", xy=(1e17, 1.066),
             xytext=(7e13, 1.10), color=C_SL, fontsize=10.5, fontweight="bold",
             ha="center", va="center",
             arrowprops=dict(arrowstyle="->", color=C_SL, lw=1.6,
                             connectionstyle="arc3,rad=0.25"))
# root-cause banner
axL.text(0.5, -0.205, "frozen V$_{bi}$ = 1.30 V severs the lever",
         transform=axL.transAxes, ha="center", va="top",
         fontsize=11.5, fontweight="bold", color="#222222",
         bbox=dict(boxstyle="round,pad=0.45", fc="#fdecec", ec=C_SC, lw=1.4))

# ================= RIGHT: schematic band diagram (cartoon) =================
axR.set_title("Built-in potential pinned\nregardless of N$_{D}$",
              fontsize=12.5, pad=8)
axR.set_xlim(0, 10); axR.set_ylim(0, 10); axR.axis("off")

# x-zones: ETL | absorber | HTL
axR.text(1.6, 9.5, "ETL", ha="center", fontsize=11, color=C_SL, fontweight="bold")
axR.text(5.0, 9.5, "perovskite", ha="center", fontsize=11, color="#555")
axR.text(8.4, 9.5, "HTL", ha="center", fontsize=11, color="#7a5cc0", fontweight="bold")
for xb in (3.1, 6.9):
    axR.axvline(xb, ymin=0.05, ymax=0.88, color="#cccccc", ls="-", lw=1)

# conduction-band edge: same total drop V_bi for both doping levels
xL_, xR_ = 0.6, 9.4
def band(xc, yL, yR):
    # piecewise: flat in ETL, slope across absorber, flat in HTL
    xs = np.array([0.6, 3.1, 6.9, 9.4])
    ys = np.array([yL, yL, yR, yR])
    return xs, ys

# low N_D (lighter) and high N_D (darker) — IDENTICAL drop because V_bi frozen
xs, y_lo = band(0, 7.6, 3.6)
axR.plot(xs, y_lo, color="#9bbce8", lw=2.6, ls=(0, (6, 3)),
         label="low N$_{D}$")
axR.plot(xs, y_lo - 2.2, color="#9bbce8", lw=2.0, ls=(0, (6, 3)))  # valence band
xs2, y_hi = band(0, 7.6, 3.6)
axR.plot(xs2, y_hi, color=C_SL, lw=2.8, label="high N$_{D}$ (10×)")
axR.plot(xs2, y_hi - 2.2, color=C_SL, lw=2.2)

# band labels
axR.text(9.55, 7.55, "E$_C$", fontsize=11, va="center")
axR.text(9.55, 5.35, "E$_V$", fontsize=11, va="center")

# V_bi double arrow (same for both) — drawn on the left ETL flat region
axR.annotate("", xy=(1.4, 7.6), xytext=(1.4, 3.6),
             arrowprops=dict(arrowstyle="<->", color="#222", lw=2.0))
axR.text(1.95, 5.6, "q·V$_{bi}$\n(frozen,\n1.30 V)", fontsize=10.5,
         va="center", ha="left", fontweight="bold")

# annotation: curves overlap -> no lever
axR.annotate("low & high N$_{D}$\nbands overlap\n→ no V$_{oc}$ gain",
             xy=(5.6, 5.4), xytext=(6.6, 1.6), fontsize=10, ha="center",
             va="center", color=C_SL, fontweight="bold",
             arrowprops=dict(arrowstyle="->", color=C_SL, lw=1.5))

axR.legend(loc="lower left", fontsize=10, framealpha=0.95,
           bbox_to_anchor=(0.0, -0.02))

fig.suptitle("Root cause: frozen built-in potential severs the ETL-doping → V$_{oc}$ lever",
             fontsize=13.5, fontweight="bold", y=0.995)
fig.tight_layout(rect=(0, 0.04, 1, 0.96))
fig.savefig(OUT, dpi=200)
print("WROTE", OUT, os.path.getsize(OUT))
