#!/usr/bin/env python3
"""f2_cbo_refute.png — prior-claim refutation: removing the band offset does NOT close the Voc gap."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

SUMMARY = "/Users/shane/Library/CloudStorage/OneDrive-HKUST(Guangzhou)/SolarLab/outputs/scaps_full_off/summary.json"
OUT = "/Users/shane/Library/CloudStorage/OneDrive-HKUST(Guangzhou)/SolarLab/docs/reference/Reasoning/figures/f2_cbo_refute.png"

with open(SUMMARY) as f:
    data = json.load(f)
rows = data["summary"]["CHI_ETL"]["rows"]  # [dEc(eV), solarlab_Voc, dir, scaps_Voc]

x, sl, sc = [], [], []
for r in rows:
    if r[1] is None or r[1] == 0:   # drop non-converged SolarLab points
        continue
    x.append(float(r[0])); sl.append(float(r[1])); sc.append(float(r[3]))
x = np.array(x); sl = np.array(sl); sc = np.array(sc)

# Residual gap at flat-band (ΔE_C = 0)
i0 = int(np.argmin(np.abs(x - 0.0)))
sl0, sc0 = sl[i0], sc[i0]
gap0_mV = (sc0 - sl0) * 1000.0

plt.rcParams.update({"font.size": 12, "font.family": "DejaVu Sans",
                     "axes.linewidth": 1.0, "figure.facecolor": "white",
                     "axes.facecolor": "white"})

fig, ax = plt.subplots(figsize=(8, 5), dpi=200)

C_SC = "#b3261e"   # SCAPS red
C_SL = "#1565c0"   # SolarLab blue

ax.plot(x, sc, "--", color=C_SC, lw=2.4, marker="s", ms=6.5,
        markerfacecolor="white", markeredgecolor=C_SC, markeredgewidth=1.6,
        label="SCAPS-1D", zorder=4)
ax.plot(x, sl, "-", color=C_SL, lw=2.6, marker="o", ms=6.5,
        markerfacecolor=C_SL, markeredgecolor="white", markeredgewidth=0.8,
        label="SolarLab", zorder=5)

# Vertical marker at ΔE_C = 0 (flat-band / no offset)
ax.axvline(0.0, color="0.45", lw=1.4, ls=(0, (4, 3)), zorder=2)
ax.text(0.012, 0.300, "ΔE$_C$ = 0\n(no offset)", fontsize=10.5,
        color="0.30", ha="left", va="bottom", transform=ax.get_xaxis_transform())

# Shade the "prior claim predicts closure here" region (ΔE_C >= 0)
ax.axvspan(0.0, x.max() + 0.05, color="#fff3e0", alpha=0.55, zorder=0)

# --- Annotate the residual gap at ΔE_C = 0 with a brace-style bracket ---
xb = 0.0
# vertical double-headed arrow between the two curves at ΔE_C = 0
ax.annotate("", xy=(xb, sc0), xytext=(xb, sl0),
            arrowprops=dict(arrowstyle="<->", color="#37474f", lw=2.0,
                            shrinkA=0, shrinkB=0), zorder=6)
# small horizontal caps to read like a brace
for yv in (sl0, sc0):
    ax.plot([xb - 0.018, xb + 0.018], [yv, yv], color="#37474f", lw=1.6, zorder=6)
ax.annotate(f"residual gap\n≈ {gap0_mV:.0f} mV\n(NOT closed)",
            xy=(xb, 0.5 * (sl0 + sc0)),
            xytext=(0.165, 0.86), fontsize=11, fontweight="bold", color="#263238",
            ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.4", fc="#fff8e1", ec="#37474f", lw=1.3),
            arrowprops=dict(arrowstyle="-", color="#37474f", lw=1.4,
                            connectionstyle="arc3,rad=-0.18"), zorder=7)

# Mark the base operating point (ΔE_C = -0.16 eV)
ib = int(np.argmin(np.abs(x - (-0.16))))
ax.scatter([x[ib]], [sl[ib]], s=120, facecolor="none", edgecolor=C_SL, lw=1.8, zorder=8)
ax.scatter([x[ib]], [sc[ib]], s=120, facecolor="none", edgecolor=C_SC, lw=1.8, zorder=8)
ax.annotate("base model\nΔE$_C$ = −0.16 eV", xy=(x[ib], sl[ib]),
            xytext=(-0.60, 0.62), fontsize=9.5, color="0.25", ha="left", va="center",
            arrowprops=dict(arrowstyle="->", color="0.45", lw=1.2,
                            connectionstyle="arc3,rad=0.2"))

ax.set_xlabel("Conduction-band offset  ΔE$_C$  (eV)   "
              "[cliff ◀  −  |  +  ▶ spike]", fontsize=12)
ax.set_ylabel("Open-circuit voltage  V$_{oc}$  (V)", fontsize=12)
ax.set_title("Prior-claim refutation: removing the band offset does NOT close the V$_{oc}$ gap",
             fontsize=12.6, fontweight="bold", pad=12)

ax.set_xlim(x.min() - 0.06, x.max() + 0.06)
ax.set_ylim(min(sl.min(), sc.min()) - 0.05, max(sl.max(), sc.max()) + 0.10)
ax.grid(True, ls=":", lw=0.7, color="0.78", zorder=1)
ax.legend(loc="lower right", frameon=True, framealpha=0.95, fontsize=11.5,
          edgecolor="0.6")

# Takeaway caption strip under the axes
fig.text(0.5, 0.005,
         "Takeaway: with the conduction-band offset removed (ΔE$_C$→0) the gap is still ≈ "
         f"{gap0_mV:.0f} mV → the 37×-J$_0$ band-offset-dissipation mechanism is unsupported.",
         ha="center", va="bottom", fontsize=9.6, color="#37474f",
         bbox=dict(boxstyle="round,pad=0.45", fc="#eceff1", ec="0.7", lw=0.8))

fig.tight_layout(rect=(0, 0.055, 1, 1))
fig.savefig(OUT, dpi=200, facecolor="white", bbox_inches="tight")
print("WROTE", OUT)
print("SIZE", os.path.getsize(OUT))
print(f"gap0_mV={gap0_mV:.2f}  sl0={sl0:.4f}  sc0={sc0:.4f}  npts={len(x)}")
