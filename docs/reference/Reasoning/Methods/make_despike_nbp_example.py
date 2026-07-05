#!/usr/bin/env python3
"""Reproduce despike_nbp_example.{png,pdf} with a clean, non-overlapping layout.

Fixes two text collisions present in the prior ad-hoc render:
  1. green ``nb_p = sqrt(100x10) ~= 32`` formula no longer runs through the
     left ("100") data dot -- it sits in the empty lower-left whitespace with a
     short arrow to the green stand-in point.
  2. the "(plain average would give 55 -- wrong)" caption now lives INSIDE the
     blue box instead of being squeezed between the two panel boxes.

Run:  python3 make_despike_nbp_example.py
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# ---- palette (sampled from the original figure) -------------------------
RED    = "#c71f1f"
ORANGE = "#d58e1d"
GREEN  = "#328036"
NAVY   = "#24527c"
GRAY   = "#5f5f5f"
BLUE_FILL   = "#f3f7fc"
ORANGE_FILL = "#fff8ec"

OUT = Path(__file__).resolve().parent
plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans"})

fig = plt.figure(figsize=(13.6, 7.0))

# =======================================================================
# LEFT: the three-node log-density plot
# =======================================================================
ax = fig.add_axes([0.075, 0.115, 0.535, 0.74])

x = [0, 1, 2]                       # node i-1, node i, node i+1
nbp = float(np.sqrt(100 * 10))      # geometric stand-in ~= 31.6
spike = 10000.0
half = nbp + (spike - nbp) * 0.5    # f=0.5 -> 5016

# neighbour trend (geometric interpolation 100 -> 10 passes through nb_p)
ax.plot([0, 2], [100, 10], "--", color=NAVY, lw=2.2, zorder=2)

# blend arrow: spike -> stand-in
ax.annotate("", xy=(1, nbp * 1.07), xytext=(1, spike),
            arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=3.4,
                            shrinkA=0, shrinkB=0), zorder=3)
ax.text(0.74, 320, "blend\nby $(1-f)$", color=ORANGE, fontsize=13,
        ha="center", va="center", fontweight="bold")

# markers
ax.plot([0], [100], "o", color=NAVY, ms=15, zorder=5)
ax.plot([2], [10],  "o", color=NAVY, ms=15, zorder=5)
ax.plot([1], [spike], "o", color=RED, ms=18, zorder=6)
ax.plot([1], [half], "s", color=ORANGE, ms=15, zorder=6)
ax.plot([1], [nbp], "o", color=GREEN, ms=17, zorder=6)

# node-value labels
ax.text(-0.07, 165, "100", color=NAVY, fontsize=14, fontweight="bold",
        ha="right", va="center")
ax.text(2.13, 10, "10", color=NAVY, fontsize=14, fontweight="bold",
        ha="left", va="center")

# f / spike annotations on the right of each junction marker
ax.annotate("$f$=0", xy=(1, spike), xytext=(1.1, spike), color=RED,
            fontsize=13, fontweight="bold", va="center",
            arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.6,
                            shrinkA=2, shrinkB=4))
ax.text(1.27, spike, "real value = 10000\n(the SPIKE — artifact)",
        color=RED, fontsize=12.5, va="center", ha="left")
ax.text(1.12, half, "$f$=0.5", color=ORANGE, fontsize=13,
        fontweight="bold", va="center", ha="left")
ax.text(1.12, nbp, "$f$=1", color=GREEN, fontsize=13,
        fontweight="bold", va="center", ha="left")

# green de-spike formula -> moved into the empty lower-left whitespace
ax.text(-0.42, 60, r"$nb_p=\sqrt{100\times10}\approx 32$",
        color=GREEN, fontsize=13.5, ha="left", va="center")
ax.text(-0.42, 33, "(de-spiked stand-in)", color=GREEN, fontsize=12,
        ha="left", va="center")
ax.annotate("", xy=(0.93, nbp * 0.99), xytext=(0.34, 41),
            arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=2.0,
                            shrinkA=2, shrinkB=6), zorder=4)

# axes cosmetics
ax.set_yscale("log")
ax.set_ylim(3, 3.2e4)
ax.set_xlim(-0.55, 2.45)
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(["node $i-1$\n(left)", "node $i$\n(junction)",
                    "node $i+1$\n(right)"], fontsize=12.5)
ax.set_ylabel("hole density $p$  (log scale)", fontsize=13)
ax.tick_params(axis="y", labelsize=12)
ax.grid(True, which="both", axis="y", color="0.88", lw=0.7)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

# =======================================================================
# RIGHT: explanation panel (two stacked rounded boxes)
# =======================================================================
pan = fig.add_axes([0.635, 0.115, 0.345, 0.74])
pan.set_xlim(0, 1)
pan.set_ylim(0, 1)
pan.axis("off")


def box(y0, y1, fc, ec):
    pan.add_patch(FancyBboxPatch(
        (0.02, y0), 0.96, y1 - y0,
        boxstyle="round,pad=0.0,rounding_size=0.03",
        linewidth=2.0, edgecolor=ec, facecolor=fc,
        mutation_aspect=0.5, zorder=1))


# --- blue "Why geometric mean?" box ---
box(0.545, 0.99, BLUE_FILL, NAVY)
pan.text(0.5, 0.93, "Why GEOMETRIC mean?", color=NAVY, fontsize=15,
         fontweight="bold", ha="center", va="center")
pan.text(0.5, 0.82, "density changes ×10 per step,\nso average the EXPONENTS:",
         color="0.15", fontsize=12.5, ha="center", va="center")
pan.text(0.5, 0.68,
         r"$10^{2}$  and  $10^{1}\ \ \rightarrow\ \ 10^{1.5}\approx 32$",
         color=NAVY, fontsize=15.5, ha="center", va="center")
pan.text(0.5, 0.585, "(plain average would give 55 — wrong)", color=GRAY,
         fontsize=11, style="italic", ha="center", va="center")

# --- orange formula / blend-table box ---
box(0.02, 0.49, ORANGE_FILL, ORANGE)
pan.text(0.5, 0.435,
         r"$p_{rec}=nb_p+(p[i]-nb_p)\,(1-f)$",
         color="0.12", fontsize=14.5, ha="center", va="center")

rows = [
    ("$f$=0",   "10000", "keep spike", RED),
    ("$f$=0.5", "5016",  "halfway",    ORANGE),
    ("$f$=1",   "32",    "$=nb_p$",    GREEN),
]
ys = [0.31, 0.205, 0.10]
for (flab, val, note, col), yy in zip(rows, ys):
    pan.text(0.135, yy, flab, color=col, fontsize=13.5, fontweight="bold",
             ha="left", va="center")
    pan.text(0.56, yy, val, color=col, fontsize=14, fontweight="bold",
             ha="right", va="center")
    pan.text(0.66, yy, note, color=GRAY, fontsize=12.5, ha="left", va="center")

# =======================================================================
# title + footer
# =======================================================================
fig.text(0.5, 0.955,
         "What is $nb_p$?   —   smoothing the junction-node spike using its "
         "two neighbours", fontsize=17, fontweight="bold", ha="center")
fig.text(0.5, 0.03,
         "Only the recombination RATE uses $nb_p$;  the real spike (10000) "
         "still drives the current.   Electrons use the same recipe:  "
         r"$nb_n=\sqrt{n_{i-1}\,n_{i+1}}$.",
         fontsize=11, color=GRAY, ha="center")

fig.savefig(OUT / "despike_nbp_example.png", dpi=150)
fig.savefig(OUT / "despike_nbp_example.pdf")
print("wrote despike_nbp_example.png / .pdf to", OUT)
