#!/usr/bin/env python3
"""f7: V_oc reality number-line. Which absolute is more realistic?"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams["font.family"] = ["Arial", "DejaVu Sans"]
plt.rcParams["mathtext.default"] = "regular"
plt.rcParams["axes.unicode_minus"] = True
plt.rcParams["font.size"] = 12

BLUE = "#1f4fd8"    # SolarLab
RED = "#d81f2a"     # SCAPS
GREEN = "#2e8b57"   # measured-device median band
GREY = "#555555"

OUT = "/Users/shane/Library/CloudStorage/OneDrive-HKUST(Guangzhou)/SolarLab/docs/partner/ppt_root_cause/figures/f7_experiment_voc.png"

# Verified literature/anchor values (V)
MEAS_LO, MEAS_HI = 1.05, 1.13        # measured-device Voc median band
SOLARLAB = 1.072                      # inside the measured band
SCAPS = 1.1676                        # champion ceiling (~1.17 V)
SQ_LO, SQ_HI = 1.28, 1.33            # Shockley-Queisser radiative limit, Eg=1.53 eV

fig, ax = plt.subplots(figsize=(10, 5))

xlo, xhi = 0.90, 1.35
ybase = 0.0

# --- main number-line axis ---
ax.axhline(ybase, color="0.25", lw=1.6, zorder=2)
for xt in [0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.35]:
    ax.plot([xt, xt], [ybase - 0.018, ybase + 0.018], color="0.25", lw=1.0, zorder=2)
    ax.text(xt, ybase - 0.052, f"{xt:.2f}", ha="center", va="top",
            fontsize=10.5, color="0.2")

# --- measured-device median band (green shade) ---
ax.axvspan(MEAS_LO, MEAS_HI, ymin=0.30, ymax=0.70, color=GREEN, alpha=0.16, zorder=1)
ax.plot([MEAS_LO, MEAS_LO], [-0.075, 0.075], color=GREEN, lw=1.3, ls=":", zorder=2)
ax.plot([MEAS_HI, MEAS_HI], [-0.075, 0.075], color=GREEN, lw=1.3, ls=":", zorder=2)
ax.text((MEAS_LO + MEAS_HI) / 2, 0.092,
        "Measured-device median band\n1.05 - 1.13 V (champions to ~1.17 V)",
        ha="center", va="bottom", fontsize=10.5, color=GREEN, fontweight="bold")

# --- Shockley-Queisser radiative limit band (grey shade) ---
ax.axvspan(SQ_LO, SQ_HI, ymin=0.30, ymax=0.70, color=GREY, alpha=0.16, zorder=1)
ax.plot([SQ_LO, SQ_LO], [-0.075, 0.075], color=GREY, lw=1.3, ls=":", zorder=2)
ax.plot([SQ_HI, SQ_HI], [-0.075, 0.075], color=GREY, lw=1.3, ls=":", zorder=2)
ax.text((SQ_LO + SQ_HI) / 2, 0.092,
        "Shockley-Queisser\nradiative limit\n1.28 - 1.33 V (E$_g$=1.53 eV)",
        ha="center", va="bottom", fontsize=10.5, color=GREY, fontweight="bold")

# --- marker helper: pin above the line, label below ---
def pin(x, color, top_label, sub_label, dy_top=0.052):
    ax.plot([x], [ybase], marker="o", ms=11, color=color,
            mec="white", mew=1.4, zorder=5)
    ax.annotate("", xy=(x, ybase + 0.012), xytext=(x, ybase + dy_top),
                arrowprops=dict(arrowstyle="-", color=color, lw=1.4), zorder=4)
    ax.text(x, ybase + dy_top + 0.006, top_label, ha="center", va="bottom",
            fontsize=11.5, color=color, fontweight="bold")
    ax.text(x, ybase + dy_top + 0.006, sub_label, ha="center", va="bottom",
            fontsize=0)  # placeholder; real sub below

# SolarLab marker (inside measured band)
ax.plot([SOLARLAB], [ybase], marker="o", ms=12, color=BLUE,
        mec="white", mew=1.5, zorder=6)
ax.annotate("", xy=(SOLARLAB, ybase + 0.012), xytext=(SOLARLAB, ybase + 0.175),
            arrowprops=dict(arrowstyle="-", color=BLUE, lw=1.5), zorder=4)
ax.text(SOLARLAB, ybase + 0.182,
        "SolarLab\nV$_{oc}$ = 1.072 V\n(inside measured band)",
        ha="center", va="bottom", fontsize=11.5, color=BLUE, fontweight="bold")

# SCAPS marker (champion ceiling)
ax.plot([SCAPS], [ybase], marker="s", ms=11, color=RED,
        mec="white", mew=1.5, zorder=6)
ax.annotate("", xy=(SCAPS, ybase + 0.012), xytext=(SCAPS, ybase + 0.175),
            arrowprops=dict(arrowstyle="-", color=RED, lw=1.5), zorder=4)
ax.text(SCAPS, ybase + 0.182,
        "SCAPS-1D\nV$_{oc}$ = 1.1676 V\n(champion ceiling)",
        ha="center", va="bottom", fontsize=11.5, color=RED, fontweight="bold")

# --- bottom annotation banner ---
banner = FancyBboxPatch((0.915, -0.235), 0.42, 0.075,
                        boxstyle="round,pad=0.006,rounding_size=0.02",
                        fc="#f4f6ff", ec=BLUE, lw=1.2, zorder=3)
ax.add_patch(banner)
ax.text(1.125, -0.197,
        "SolarLab sits at the measured median;  SCAPS at the champion ceiling.\n"
        "SolarLab's absolute V$_{oc}$ is arguably the more realistic one.",
        ha="center", va="center", fontsize=11.5, color="#1a2a66", zorder=4)

# --- frame / labels ---
ax.set_xlim(xlo, xhi)
ax.set_ylim(-0.27, 0.40)
ax.set_yticks([])
for s in ["left", "right", "top"]:
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.set_xlabel("Open-circuit voltage  V$_{oc}$  (V)", fontsize=13, labelpad=10)

# legend
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
handles = [
    Line2D([0], [0], marker="o", ls="", color=BLUE, ms=10, mec="white",
           label="SolarLab (this work)"),
    Line2D([0], [0], marker="s", ls="", color=RED, ms=10, mec="white",
           label="SCAPS-1D"),
    Patch(fc=GREEN, alpha=0.25, label="Measured median band"),
    Patch(fc=GREY, alpha=0.25, label="S-Q radiative limit"),
]
ax.legend(handles=handles, loc="upper left", fontsize=10, frameon=True,
          ncol=1, bbox_to_anchor=(0.005, 0.99))

ax.set_title("Which absolute is more realistic?", fontsize=16,
             fontweight="bold", pad=14)

fig.tight_layout()
fig.savefig(OUT, dpi=200, facecolor="white", bbox_inches="tight")
plt.close(fig)

sz = os.path.getsize(OUT)
print("OK", OUT, sz)
assert os.path.exists(OUT) and sz > 20000, f"file too small: {sz}"
