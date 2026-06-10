#!/usr/bin/env python3
"""f5: grouped bar chart of the four FOM, SolarLab vs SCAPS."""
import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = "/Users/shane/Library/CloudStorage/OneDrive-HKUST(Guangzhou)/SolarLab/docs/partner/ppt_root_cause/figures/f5_fom_bars.png"
REF = "/Users/shane/Library/CloudStorage/OneDrive-HKUST(Guangzhou)/SolarLab/perovskite-sim/tests/integration/scaps_reference.json"

# SCAPS base FOM read from reference (verify against verbatim numbers)
ref = json.load(open(REF))
b = ref["base_model"]
scaps = {
    "Voc": b["Voc_V"],          # 1.1676
    "Jsc": b["Jsc_mA_cm2"],     # 26.282
    "FF":  b["FF_percent"],     # 86.99
    "PCE": b["PCE_percent"],    # 26.69
}
# SolarLab genuine self-consistent FOM (Jsc = genuine 23.96, not the spliced 25.73)
solarlab = {"Voc": 1.072, "Jsc": 23.96, "FF": 86.2, "PCE": 22.12}

assert abs(scaps["Voc"] - 1.1676) < 1e-4
assert abs(scaps["Jsc"] - 26.282) < 1e-3

plt.rcParams.update({"font.size": 12, "axes.titlesize": 14, "figure.facecolor": "white"})

panels = [
    ("Voc", "V$_{oc}$  (V)",        "{:.4f}", "{:.3f}"),
    ("Jsc", "J$_{sc}$  (mA/cm$^{2}$)", "{:.3f}", "{:.2f}"),
    ("FF",  "FF  (%)",              "{:.2f}", "{:.1f}"),
    ("PCE", "PCE  (%)",             "{:.2f}", "{:.2f}"),
]

C_SL = "#C44E52"   # SolarLab (red)
C_SC = "#4C72B0"   # SCAPS (blue)
BAND = "#9ecae1"   # measured-device Voc band

fig, axes = plt.subplots(1, 4, figsize=(10, 5))

for ax, (key, ylab, fmt_sc, fmt_sl) in zip(axes, panels):
    sl_v = solarlab[key]
    sc_v = scaps[key]
    x = [0, 1]
    bars = ax.bar(x, [sl_v, sc_v], width=0.62,
                  color=[C_SL, C_SC], edgecolor="black", linewidth=0.8, zorder=3)

    # measured-device Voc band behind the Voc bars
    if key == "Voc":
        ax.axhspan(1.05, 1.13, color=BAND, alpha=0.45, zorder=0,
                   label="Measured-device band\n(1.05–1.13 V)")

    # value labels on bars
    ax.text(0, sl_v, fmt_sl.format(sl_v), ha="center", va="bottom",
            fontsize=11, fontweight="bold", color=C_SL, zorder=5)
    ax.text(1, sc_v, fmt_sc.format(sc_v), ha="center", va="bottom",
            fontsize=11, fontweight="bold", color=C_SC, zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels(["SolarLab", "SCAPS"], fontsize=11)
    ax.set_ylabel(ylab, fontsize=12)
    ymax = max(sl_v, sc_v)
    ax.set_ylim(0, ymax * 1.18)
    ax.grid(axis="y", ls=":", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    if key == "Voc":
        ax.legend(loc="lower center", fontsize=8.5, framealpha=0.9)

# shared legend for the two engines
from matplotlib.patches import Patch
handles = [Patch(facecolor=C_SL, edgecolor="black", label="SolarLab"),
           Patch(facecolor=C_SC, edgecolor="black", label="SCAPS-1D")]
fig.legend(handles=handles, loc="upper right", fontsize=11,
           ncol=2, frameon=True, bbox_to_anchor=(0.995, 0.995))

fig.suptitle("Base-model figures of merit: SolarLab vs SCAPS", fontsize=15, y=1.02)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
print("size", os.path.getsize(OUT))
