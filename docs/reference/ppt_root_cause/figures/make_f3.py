#!/usr/bin/env python3
"""f3: the PCE reporting artifact (cross-run Jsc cell-splice)."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({
    "font.size": 12,
    "font.family": "DejaVu Sans",
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

# Verified values (verbatim from adversarially-reviewed report)
JSC_HEADLINE = 25.73   # foreign N_grid=30 cell (cross-run splice)
JSC_GENUINE  = 23.96   # genuine self-consistent (1.0709 x 23.957 x 0.862)
VOC          = 1.0709
FF           = 0.862
PCE_GENUINE  = VOC * 23.957 * FF * 1.0  # = 22.12 % (J in mA/cm^2 -> % directly)

C_HEAD = "#c0392b"   # red  - artifact
C_GEN  = "#1f77b4"   # blue - genuine
C_BOX  = "#2c3e50"

fig, (axb, axt) = plt.subplots(
    1, 2, figsize=(10, 5),
    gridspec_kw={"width_ratios": [1.05, 1.0]},
)

# ---------------- LEFT: bar comparison of the two Jsc values ----------------
labels = ["headline\n(foreign N$_{grid}$=30 cell)", "genuine\n(self-consistent)"]
vals = [JSC_HEADLINE, JSC_GENUINE]
colors = [C_HEAD, C_GEN]
xpos = [0, 1]
bars = axb.bar(xpos, vals, width=0.56, color=colors, edgecolor="black",
               linewidth=1.1, zorder=3)

for x, v in zip(xpos, vals):
    axb.text(x, v + 0.35, f"{v:.2f}", ha="center", va="bottom",
             fontsize=14, fontweight="bold",
             color=C_HEAD if x == 0 else C_GEN)

# delta annotation between bars
axb.annotate(
    "", xy=(1, JSC_GENUINE + 0.9), xytext=(0, JSC_GENUINE + 0.9),
    arrowprops=dict(arrowstyle="<->", color="#555555", lw=1.3),
)
axb.text(0.5, JSC_GENUINE + 1.25,
         f"Δ = {JSC_HEADLINE - JSC_GENUINE:.2f} mA/cm$^2$\n(≈7% splice)",
         ha="center", va="bottom", fontsize=10.5, color="#333333")

axb.set_xticks(xpos)
axb.set_xticklabels(labels, fontsize=10.5)
axb.set_ylabel("SolarLab J$_{sc}$  (mA/cm$^2$)")
axb.set_ylim(0, 30)
axb.set_xlim(-0.6, 1.6)
axb.grid(axis="y", ls=":", alpha=0.45, zorder=0)
axb.set_axisbelow(True)
axb.set_title("Two J$_{sc}$ values, one run label", fontsize=12, pad=8)

# tag which one is wrong
axb.text(0, 1.4, "REPORTED\nHEADLINE", ha="center", va="bottom",
         fontsize=8.5, color="white", fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.25", fc=C_HEAD, ec="none"))
axb.text(1, 1.4, "PHYSICALLY\nCORRECT", ha="center", va="bottom",
         fontsize=8.5, color="white", fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.25", fc=C_GEN, ec="none"))

# ---------------- RIGHT: the arithmetic reconciliation box ----------------
axt.axis("off")
axt.set_xlim(0, 1)
axt.set_ylim(0, 1)

box = FancyBboxPatch((0.04, 0.30), 0.92, 0.56,
                     boxstyle="round,pad=0.02,rounding_size=0.03",
                     linewidth=1.6, edgecolor=C_BOX, facecolor="#f4f7fb",
                     zorder=2)
axt.add_patch(box)

axt.text(0.5, 0.795, "PCE reconciliation", ha="center", va="center",
         fontsize=12.5, fontweight="bold", color=C_BOX)

# the equation with genuine Jsc
axt.text(0.5, 0.665,
         "PCE = V$_{oc}$ × J$_{sc}$ × FF",
         ha="center", va="center", fontsize=13.5, color="black")

axt.text(0.5, 0.545,
         f"= {VOC:.4f} V × {23.957:.3f} mA/cm$^2$ × {FF:.3f}",
         ha="center", va="center", fontsize=12, color=C_GEN)

axt.text(0.5, 0.425,
         f"= {PCE_GENUINE:.2f} %  ✓  matches reported 22.1 %",
         ha="center", va="center", fontsize=13, fontweight="bold",
         color=C_GEN)

# contrast: the headline Jsc does NOT reconcile
pce_head = VOC * JSC_HEADLINE * FF
axt.text(0.5, 0.355,
         f"(headline J$_{{sc}}$ {JSC_HEADLINE:.2f} → {pce_head:.2f} %, "
         f"≠ 22.1 %)",
         ha="center", va="center", fontsize=9.5, color=C_HEAD, style="italic")

# verdict strip
axt.text(0.5, 0.165,
         "The genuine self-consistent J$_{sc}$ reconciles 22.1 %.\n"
         "The 25.73 headline is a cross-run cell-splice —\n"
         "a REPORTING artifact, not a solver defect.",
         ha="center", va="center", fontsize=10.2, color="#222222",
         bbox=dict(boxstyle="round,pad=0.4", fc="#fdf2e9", ec=C_HEAD, lw=1.1))

fig.suptitle(
    "PCE anomaly = cross-run J$_{sc}$ cell-splice "
    "(reporting artifact, not a solver defect)",
    fontsize=13.5, fontweight="bold", y=0.985,
)

fig.tight_layout(rect=[0, 0, 1, 0.945])

OUT = ("/Users/shane/Library/CloudStorage/OneDrive-HKUST(Guangzhou)/"
       "SolarLab/docs/reference/ppt_root_cause/figures/f3_pce_splice.png")
fig.savefig(OUT, dpi=200, facecolor="white", bbox_inches="tight")
plt.close(fig)

sz = os.path.getsize(OUT)
print(f"WROTE {OUT}")
print(f"BYTES {sz}")
assert sz > 20_000, f"file too small: {sz}"
print("OK")
