#!/usr/bin/env python3
"""Reproduce despike_principle_and_matrix.{pdf,png} with a clean, overlap-free
six-panel layout and enlarged fonts.

PRINCIPLE row (A,B,C): why a junction-node spike forms and what the de-spike
fraction f removes.  MATRIX row (D,E,F): how the base operating point and the
SCAPS sweeps respond to f.

The prior render was ad-hoc (no source).  This regenerates it with explicit
axes placement so nothing collides even at the larger type sizes: the top
banner clears the panel titles, the panel-B nb_p formula and "blend" label sit
in free whitespace, and the equation band / footer have their own reserved
strips.

Data values (panel-E matrix, panel-D/F anchor points, SCAPS V_oc, cal/trend-opt
f) are read from the original figure and preserved verbatim.

Run:  python3 make_despike_principle_and_matrix.py
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.colors import Normalize
import matplotlib.colorbar as mcolorbar

# ---- palette ----------------------------------------------------------
RED, ORANGE, GREEN, NAVY = "#c71f1f", "#d58e1d", "#328036", "#24527c"
PURPLE, GRAY = "#8746b0", "#5f5f5f"
HTL_BG, PVK_BG, ETL_BG = "#eaf1f9", "#fdf1dd", "#eaf5ec"
BAND_BG = "#f3f7fc"

# ---- font sizes (enlarged) -------------------------------------------
SUP = 23          # suptitle
BANNER = 14.5     # principle / matrix banner
TITLE = 16.5      # panel title
AXL = 15.5        # axis label
TICK = 13.5       # tick labels
ANN = 15          # in-panel annotations
LEG = 13.5        # legends
REGION = 16.5     # panel-A region labels
BANDLBL = 19      # panel-A E_C / E_V
EQ_MATH = 20.5    # equation-band math line
EQ_T1 = 15.5      # equation-band note line 1
EQ_T2 = 14.5      # equation-band note line 2
FOOT1 = 15.5      # footer line 1
FOOT2 = 14.5      # footer line 2
CELL = 15.5       # heatmap cell numbers
DIRSZ = 13.5      # heatmap DIR
HMLBL = 13.5      # heatmap row / col labels
CBLBL = 13        # colorbar label

OUT = Path(__file__).resolve().parent
plt.rcParams.update({"font.family": "DejaVu Sans",
                     "mathtext.fontset": "dejavusans"})

fig = plt.figure(figsize=(19.0, 15.2))

# =======================================================================
# (A) Band offset -> hole pile-up   (schematic)
# =======================================================================
axA = fig.add_axes([0.050, 0.625, 0.262, 0.255])
xj1, xj2 = 0.40, 0.83                       # HTL/PVK and PVK/ETL junctions
axA.axvspan(0, xj1, color=HTL_BG)
axA.axvspan(xj1, xj2, color=PVK_BG)
axA.axvspan(xj2, 1, color=ETL_BG)
axA.axvline(xj1, color="0.45", ls=":", lw=1.5)

ec = [(0, xj1, 3.00), (xj1, xj2, 2.32), (xj2, 1, 2.00)]
ev = [(0, xj1, 1.05), (xj1, xj2, 0.55), (xj2, 1, 0.18)]
for (x0, x1, y), col in ([(s, NAVY) for s in ec] + [(s, RED) for s in ev]):
    axA.plot([x0, x1], [y, y], color=col, lw=3.2, solid_capstyle="butt")
for x, lo, hi, col in [(xj1, 2.32, 3.00, NAVY), (xj2, 2.00, 2.32, NAVY),
                       (xj1, 0.55, 1.05, RED), (xj2, 0.18, 0.55, RED)]:
    axA.plot([x, x], [lo, hi], color=col, lw=3.2, solid_capstyle="butt")

axA.text(xj1 / 2, 3.24, "$E_C$", color=NAVY, fontsize=BANDLBL, ha="center")
axA.text(xj1 / 2, 0.78, "$E_V$", color=RED, fontsize=BANDLBL, ha="center")
hx = np.linspace(xj1 - 0.085, xj1 - 0.015, 7)
axA.scatter(hx, np.full_like(hx, 1.05), s=60, color=RED, zorder=6,
            edgecolor="white", linewidth=0.6)
axA.annotate("holes pile up\non one node", xy=(xj1 - 0.04, 1.08),
             xytext=(xj1 + 0.10, 1.62), color=RED, fontsize=ANN, ha="left",
             va="center",
             arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.8))
axA.annotate("", xy=(xj1 - 0.005, 1.02), xytext=(xj1 - 0.005, 0.58),
             arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=2.6))
axA.text(xj1 + 0.02, 0.64, r"$\Delta E_V\approx0.18$ eV", color=ORANGE,
         fontsize=ANN, ha="left", va="center")

for xc, lab, col in [(xj1 / 2, "HTL", NAVY),
                     ((xj1 + xj2) / 2, "Perovskite", "#b07d1a"),
                     ((xj2 + 1) / 2, "ETL", GREEN)]:
    axA.text(xc, 3.63, lab, color=col, fontsize=REGION, ha="center",
             fontweight="bold")

axA.set_xlim(0, 1)
axA.set_ylim(0, 3.9)
axA.set_xticks([])
axA.set_yticks([])
axA.set_xlabel("position $x$", fontsize=AXL)
axA.set_ylabel("energy (eV, schematic)", fontsize=AXL)
axA.set_title("(A)  Band offset $\\rightarrow$ hole pile-up", fontsize=TITLE,
              fontweight="bold", loc="left", pad=9)

# =======================================================================
# (B) Density spike -> neighbour-mean blend
# =======================================================================
axB = fig.add_axes([0.390, 0.625, 0.255, 0.255])
nodes = np.array([-3, -2, -1, 0, 1, 2, 3])
logp = np.array([1.9, 1.7, 1.5, 4.0, 1.0, 0.8, 0.7])
axB.plot(nodes, logp, "-", color="0.6", lw=1.6, zorder=2)
axB.plot(nodes[nodes != 0], logp[nodes != 0], "o", color="0.55", ms=9, zorder=3)

nbp = (1.5 + 1.0) / 2.0
spike = 4.0
half = nbp + (spike - nbp) * (1 - 0.53)
axB.axhline(nbp, color=NAVY, ls="--", lw=2.4, zorder=2)
axB.axvline(0, color="0.5", ls=":", lw=1.4, zorder=1)
axB.annotate("", xy=(0, nbp + 0.06), xytext=(0, spike),
             arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=3.4),
             zorder=4)

axB.plot([0], [spike], "o", color=RED, ms=18, zorder=6)
axB.plot([0.18], [spike], "s", color=RED, ms=13, zorder=6)
axB.plot([0], [half], "s", color=ORANGE, ms=14, zorder=6)
axB.plot([0], [nbp], "s", color=NAVY, ms=14, zorder=6)

axB.text(0, 4.34, "junction node $i$", color=RED, fontsize=ANN, ha="center")
axB.text(0.42, spike, "$f$=0 (spike)", color=RED, fontsize=ANN, va="center")
axB.text(0.32, half, "$f$=0.53", color=ORANGE, fontsize=ANN, va="center")
axB.text(0.16, nbp - 0.22, "$f$=1", color=NAVY, fontsize=ANN, va="center",
         ha="left")
axB.text(3.45, nbp + 0.10, "$nb_p$ line", color=NAVY, fontsize=ANN,
         va="bottom", ha="right")
axB.text(-0.55, 2.75, "blend\n$(1-f)$", color=ORANGE, fontsize=ANN,
         ha="right", va="center", fontweight="bold")
axB.text(-3.0, 0.60, r"$nb_p=\sqrt{p_{i-1}\,p_{i+1}}$", color=NAVY,
         fontsize=16.5, ha="left", va="center")

axB.set_xlim(-3.5, 3.5)
axB.set_ylim(0.3, 4.65)
axB.set_xlabel("grid node (rel. to junction)", fontsize=AXL)
axB.set_ylabel("$\\log_{10} p$  (hole density)", fontsize=AXL)
axB.tick_params(labelsize=TICK)
axB.set_title("(B)  Density spike $\\rightarrow$ neighbour-mean blend",
              fontsize=TITLE, fontweight="bold", loc="left", pad=9)
for s in ("top", "right"):
    axB.spines[s].set_visible(False)

# =======================================================================
# (C) Recombination at the node   (stacked bars)
# =======================================================================
axC = fig.add_axes([0.720, 0.625, 0.245, 0.255])
xb = [0, 1]
auger = [105, 8]
srh = [25, 26]
axC.bar(xb, auger, width=0.5, color=RED, label=r"bulk Auger ($\propto p^{2}n$)")
axC.bar(xb, srh, width=0.5, bottom=auger, color=PURPLE, label="interface-SRH")
axC.annotate("interface loss\nDOUBLE-counted", xy=(0, 130),
             xytext=(-0.42, 150), color=RED, fontsize=ANN, ha="left")
axC.annotate("Auger spike\nremoved", xy=(1, 12), xytext=(1, 72),
             color=GREEN, fontsize=ANN, ha="center",
             arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=2.0))
axC.set_xlim(-0.6, 1.6)
axC.set_ylim(0, 172)
axC.set_xticks(xb)
axC.set_xticklabels(["$f$=0\n(faithful)", "$f$=1\n(de-spiked)"], fontsize=ANN)
axC.set_ylabel("recomb. rate at node (A/m$^2$)", fontsize=AXL)
axC.tick_params(axis="y", labelsize=TICK)
axC.legend(loc="upper right", fontsize=LEG, frameon=False)
axC.set_title("(C)  Recombination at the node", fontsize=TITLE,
              fontweight="bold", loc="left", pad=9)
for s in ("top", "right"):
    axC.spines[s].set_visible(False)

# =======================================================================
# equation band (full-width)
# =======================================================================
band = fig.add_axes([0.048, 0.476, 0.918, 0.108])
band.axis("off")
band.add_patch(FancyBboxPatch((0.004, 0.05), 0.992, 0.90,
               boxstyle="round,pad=0.0,rounding_size=0.02",
               linewidth=1.8, edgecolor=NAVY, facecolor=BAND_BG,
               mutation_aspect=0.12))
band.text(0.5, 0.75,
          r"$p_{rec}[i]=nb_p+(p[i]-nb_p)(1-f)$"
          r"$\qquad\qquad R=\mathrm{SRH}+\mathrm{rad}+\mathrm{Auger}"
          r"(n_{rec},p_{rec})\qquad\qquad C_n,\,C_p\neq0$",
          fontsize=EQ_MATH, ha="center", va="center", color="0.12")
band.text(0.5, 0.44,
          "Only the recombination-rate density at the junction node is "
          "blended;  transport / SG flux uses the true (spiked) density  "
          r"$\Rightarrow$  current stays physical.",
          fontsize=EQ_T1, ha="center", va="center", color=GRAY)
band.text(0.5, 0.17,
          "$f$=0 over-counts    ·    $f$=1 removes the pile-up    ·    "
          "value is TUNED to SCAPS (emulation knob, default OFF).",
          fontsize=EQ_T2, ha="center", va="center", color=GRAY)

# =======================================================================
# (D) Base operating point vs f
# =======================================================================
axD = fig.add_axes([0.052, 0.122, 0.232, 0.280])
fa = [0, .1, .2, .3, .4, .5, .53, .6, .65, .7, .75, .8, .85, .9, 1.0]
voc = [1.117, 1.124, 1.132, 1.143, 1.152, 1.161, 1.165, 1.176, 1.181,
       1.186, 1.190, 1.194, 1.197, 1.199, 1.201]
pce = [25.25, 25.45, 25.72, 25.92, 26.05, 26.18, 26.22, 26.35, 26.42,
       26.47, 26.50, 26.55, 26.56, 26.57, 26.55]
ff = np.linspace(0, 1, 21)
axD.plot(ff, np.interp(ff, fa, voc), "-o", color=NAVY, ms=5, lw=2,
         label="$V_{oc}$")
axD.axhline(1.168, color="0.45", ls="--", lw=1.6)
axD.text(0.03, 1.1705, "SCAPS $V_{oc}$", color="0.35", fontsize=TICK)
axD.axvline(0.53, color=ORANGE, ls=":", lw=2)
axD.axvline(0.65, color=GREEN, ls="-", lw=2)
axD.text(0.512, 1.127, "cal 0.53", color=ORANGE, fontsize=TICK, rotation=90,
         ha="right", va="bottom")
axD.text(0.668, 1.127, "trend-opt 0.65", color=GREEN, fontsize=TICK,
         rotation=90, ha="left", va="bottom")
axD.set_xlim(-0.02, 1.02)
axD.set_ylim(1.112, 1.206)
axD.set_xlabel("de-spike fraction $f$", fontsize=AXL)
axD.set_ylabel("$V_{oc}$ (V)", color=NAVY, fontsize=AXL)
axD.tick_params(axis="y", labelcolor=NAVY, labelsize=TICK)
axD.tick_params(axis="x", labelsize=TICK)
axD.grid(True, color="0.9", lw=0.7)
axD.set_axisbelow(True)
axDr = axD.twinx()
axDr.plot(ff, np.interp(ff, fa, pce), "-^", color=PURPLE, ms=5, lw=2,
          label="PCE")
axDr.set_ylim(25.2, 26.68)
axDr.set_ylabel("PCE (%)", color=PURPLE, fontsize=AXL)
axDr.tick_params(axis="y", labelcolor=PURPLE, labelsize=TICK)
h1, l1 = axD.get_legend_handles_labels()
h2, l2 = axDr.get_legend_handles_labels()
axD.legend(h1 + h2, l1 + l2, loc="lower right", fontsize=LEG, frameon=True)
axD.set_title("(D)  Base operating point vs $f$", fontsize=TITLE,
              fontweight="bold", loc="left", pad=9)

# =======================================================================
# (E) Trend closure% -- sweep x f   (heatmap, manual cells)
# =======================================================================
axE = fig.add_axes([0.408, 0.122, 0.202, 0.280])
sweeps = ["CHI_ETL", "Nt_PVK ETL", "Nt_C_PVK", "Et_PVK ETL", "Nt_V_PVK",
          "Nd_ETL"]
fcols = ["0.00", "0.20", "0.40", "0.53", "0.66", "1.00"]
M = [[80, 82, 84, 85, 88, 91],
     [53, 58, 67, 72, 80, 92],
     [11, 24, 59, 69, 94, 142],
     [6, 12, 13, 23, 49, 57],
     [41, 86, 212, 248, 339, 509],
     [None] * 6]
cmap = plt.cm.RdYlGn
norm = Normalize(0, 100)
nrow, ncol = 6, 6
for r in range(nrow):
    yy = nrow - 1 - r
    for c in range(ncol):
        v = M[r][c]
        if v is None:
            axE.add_patch(Rectangle((c, yy), 1, 1, facecolor="white",
                                    edgecolor="0.7", lw=0.8))
            axE.text(c + 0.5, yy + 0.5, "DIR", ha="center", va="center",
                     fontsize=DIRSZ, color="0.25")
        else:
            axE.add_patch(Rectangle((c, yy), 1, 1, facecolor=cmap(norm(v)),
                                    edgecolor="white", lw=1.2))
            tc = "white" if (v <= 25 or v >= 86) else "black"
            axE.text(c + 0.5, yy + 0.5, str(v), ha="center", va="center",
                     fontsize=CELL, fontweight="bold", color=tc)
axE.add_patch(Rectangle((3, 0), 1, nrow, fill=False, edgecolor=ORANGE,
                        lw=3, zorder=5))
axE.set_xlim(0, ncol)
axE.set_ylim(0, nrow)
axE.set_xticks(np.arange(ncol) + 0.5)
axE.set_xticklabels(fcols, fontsize=HMLBL)
axE.set_yticks(np.arange(nrow) + 0.5)
axE.set_yticklabels(sweeps[::-1], fontsize=HMLBL)
axE.set_xlabel("de-spike fraction $f$", fontsize=AXL)
for s in axE.spines.values():
    s.set_visible(False)
axE.tick_params(length=0)
axE.set_title("(E)  Trend closure% — sweep $\\times$ $f$", fontsize=TITLE,
              fontweight="bold", loc="left", pad=9)
caxE = fig.add_axes([0.618, 0.122, 0.011, 0.280])
cb = mcolorbar.ColorbarBase(caxE, cmap=cmap, norm=norm)
cb.set_label("closure %", fontsize=CBLBL)
cb.ax.tick_params(labelsize=11.5)

# =======================================================================
# (F) Trend-optimal f: fidelity vs base cost
# =======================================================================
axF = fig.add_axes([0.712, 0.122, 0.215, 0.280])
ffa = [0, .1, .2, .3, .4, .5, .55, .6, .65, .7, .8, .9, 1.0]
fid = [.32, .39, .44, .40, .37, .41, .43, .48, .52, .50, .50, .49, .49]
err = [50, 43, 36, 26, 16, 5, 0.5, 9, 16, 23, 29, 31, 34]
fg = np.linspace(0, 1, 21)
axF.plot(fg, np.interp(fg, ffa, fid), "-o", color=NAVY, ms=5, lw=2,
         label="trend fidelity")
axF.axvline(0.53, color=ORANGE, ls=":", lw=2)
axF.axvline(0.65, color=GREEN, ls="-", lw=2)
axF.text(0.512, 0.04, "cal", color=ORANGE, fontsize=TICK, rotation=90,
         ha="right", va="bottom")
axF.text(0.668, 0.05, "$f^{*}\\approx0.65$", color=GREEN, fontsize=TICK,
         ha="left", va="bottom")
axF.set_xlim(-0.02, 1.02)
axF.set_ylim(0, 1.0)
axF.set_xlabel("de-spike fraction $f$", fontsize=AXL)
axF.set_ylabel("trend fidelity (1=closure 100%)", color=NAVY, fontsize=AXL)
axF.tick_params(axis="y", labelcolor=NAVY, labelsize=TICK)
axF.tick_params(axis="x", labelsize=TICK)
axF.grid(True, color="0.9", lw=0.7)
axF.set_axisbelow(True)
axFr = axF.twinx()
axFr.plot(fg, np.interp(fg, ffa, err), "-^", color=RED, ms=5, lw=2,
          label="|base $V_{oc}$ err| (mV)")
axFr.set_ylim(0, 52)
axFr.set_ylabel("|base $V_{oc}$ − SCAPS| (mV)", color=RED, fontsize=AXL)
axFr.tick_params(axis="y", labelcolor=RED, labelsize=TICK)
h1, l1 = axF.get_legend_handles_labels()
h2, l2 = axFr.get_legend_handles_labels()
axF.legend(h1 + h2, l1 + l2, loc="upper center", fontsize=LEG, frameon=True)
axF.set_title("(F)  Trend-optimal $f$: fidelity vs base cost", fontsize=TITLE,
              fontweight="bold", loc="left", pad=9)

# =======================================================================
# title / banner / footer
# =======================================================================
fig.text(0.5, 0.967, "Heterointerface Auger de-spike $f$  —  principle and "
         "matrix response", fontsize=SUP, fontweight="bold", ha="center")
fig.text(0.265, 0.928,
         "PRINCIPLE (top): why a spike forms and what $f$ removes.",
         fontsize=BANNER, color="0.2", ha="center")
fig.text(0.700, 0.928,
         "MATRIX TEST (bottom): how the base operating point and the 10 SCAPS "
         "sweeps respond to $f$.", fontsize=BANNER, color="0.2", ha="center")

fig.text(0.5, 0.060,
         "Solar-cell reading:  the over-counted junction Auger artificially "
         "depresses $V_{oc}$ AND compresses every sweep's $V_{oc}$ range.",
         fontsize=FOOT1, color="0.12", ha="center")
fig.text(0.5, 0.028,
         r"Removing it ($\uparrow f$) raises base $V_{oc}$ past SCAPS near "
         r"$f\approx0.55$ and OPENS the sweep ranges (closure$\uparrow$).   "
         "Nd_ETL & HTL/PVK stay direction-locked.   "
         "Trend-optimum $f^{*}\\approx0.65$.",
         fontsize=FOOT2, color="0.12", ha="center")

fig.savefig(OUT / "despike_principle_and_matrix.pdf")
fig.savefig(OUT / "despike_principle_and_matrix.png", dpi=110)
print("wrote despike_principle_and_matrix.pdf / .png to", OUT)
