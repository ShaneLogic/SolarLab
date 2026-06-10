#!/usr/bin/env python3
"""f4: Solver / physics architecture side-by-side block diagram (schematic only).

LEFT  column -> SCAPS-1D engine.
RIGHT column -> SolarLab engine.
Rows are aligned conceptual layers (variables, time integration, spatial scheme,
statistics, contacts, interface SRH). Rows that DIFFER AND MATTER for the
surviving root cause (contacts / V_bi, interface SRH) are highlighted in red.
No data -- pure matplotlib rectangles + text.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams["font.family"] = ["Arial", "DejaVu Sans"]
plt.rcParams["mathtext.default"] = "regular"
plt.rcParams["axes.unicode_minus"] = True

BLUE = "#1f4fd8"   # SolarLab
RED = "#d81f2a"    # SCAPS / divergent rows
GREY = "#555555"
OUT = Path(__file__).resolve().parent / "f4_solver_arch.png"

# Row labels (the conceptual layer) and the text for each engine.
# diverge=True -> the row differs AND matters (highlight red).
ROWS = [
    ("State variables",
     r"$\psi,\ E_{Fn},\ E_{Fp}$" + "\n(potential + 2 quasi-Fermi levels)",
     "Poisson + 2 continuity\n" + r"($\psi,\ n,\ p$)",
     False),
    ("Nonlinear / time solver",
     "Gummel iteration\n+ Newton substeps\n(steady state)",
     "Method-of-lines pseudo-transient\nRadau (solve_ivp)",
     False),
    ("Spatial discretization",
     "Finite-difference mesh\n(doubled interface nodes)",
     "Scharfetter–Gummel\nfinite-volume flux",
     False),
    ("Carrier statistics",
     "Boltzmann",
     "Boltzmann",
     False),
    ("Contacts / built-in potential",
     "Thermionic-emission\ninterface transport",
     r"Dirichlet / TE-capped contacts" + "\n(mode: fast), frozen $V_{bi}$",
     True),
    ("Interface recombination",
     "Pauwels–Vanhoutte\ninterface SRH",
     "Projected interface SRH",
     True),
]

# Geometry -----------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6.4))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")

# column x-extents
LX0, LX1 = 0.45, 4.55     # SCAPS box
RX0, RX1 = 5.45, 9.55     # SolarLab box
LCX = (LX0 + LX1) / 2
RCX = (RX0 + RX1) / 2
LBLX = 5.0                # row-label divider (center gutter)

# row vertical layout
top_y = 8.55
row_h = 1.18
gap = 0.16
n = len(ROWS)


def cell(cx, x0, x1, yc, h, text, edge, face, txtcolor="black", bold=False):
    w = x1 - x0
    box = FancyBboxPatch((x0, yc - h / 2), w, h,
                         boxstyle="round,pad=0.02,rounding_size=0.10",
                         linewidth=1.8, edgecolor=edge, facecolor=face,
                         zorder=2)
    ax.add_patch(box)
    ax.text(cx, yc, text, ha="center", va="center",
            fontsize=11, color=txtcolor,
            fontweight=("bold" if bold else "normal"), zorder=3)


# Column header banners
ax.text(LCX, 9.55, "SCAPS-1D", ha="center", va="center", fontsize=17,
        fontweight="bold", color=RED)
ax.text(RCX, 9.55, "SolarLab", ha="center", va="center", fontsize=17,
        fontweight="bold", color=BLUE)
ax.add_patch(FancyBboxPatch((LX0, 9.18), LX1 - LX0, 0.62,
             boxstyle="round,pad=0.02,rounding_size=0.10",
             linewidth=2.0, edgecolor=RED, facecolor="#fdecec", zorder=1))
ax.add_patch(FancyBboxPatch((RX0, 9.18), RX1 - RX0, 0.62,
             boxstyle="round,pad=0.02,rounding_size=0.10",
             linewidth=2.0, edgecolor=BLUE, facecolor="#eaf0ff", zorder=1))

# Rows
for i, (label, ltext, rtext, diverge) in enumerate(ROWS):
    yc = top_y - i * (row_h + gap)
    # row label in the gutter
    lblcolor = RED if diverge else GREY
    ax.text(LBLX, yc, label, ha="center", va="center", fontsize=9.5,
            color=lblcolor, fontweight=("bold" if diverge else "normal"),
            rotation=0, zorder=4,
            bbox=dict(boxstyle="round,pad=0.18", fc="white",
                      ec=(RED if diverge else "0.8"),
                      lw=(1.4 if diverge else 0.8), alpha=0.97))
    if diverge:
        ledge, lface = RED, "#fde3e3"
        redge, rface = RED, "#fde3e3"
        ltc, rtc = "black", "black"
    else:
        ledge, lface = "#c9c9c9", "#f6f6f6"
        redge, rface = "#c9c9c9", "#f6f6f6"
        ltc, rtc = "black", "black"
    cell(LCX, LX0, LX1, yc, row_h, ltext, ledge, lface, ltc, bold=diverge)
    cell(RCX, RX0, RX1, yc, row_h, rtext, redge, rface, rtc, bold=diverge)

# Red highlight legend / callout (placed below the rows, no clipping)
bottom_y = top_y - (n - 1) * (row_h + gap) - row_h / 2
cy = bottom_y - 0.62
ax.add_patch(FancyBboxPatch((LX0, cy - 0.40), RX1 - LX0, 0.80,
             boxstyle="round,pad=0.02,rounding_size=0.10",
             linewidth=1.6, edgecolor=RED, facecolor="#fff4f4", zorder=1))
ax.text((LX0 + RX1) / 2, cy,
        "Red rows = the two layers that differ AND matter: the frozen "
        r"built-in potential $V_{bi}$ at the contacts" + "\n"
        "drives the surviving donor-doping reversal; interface-SRH "
        "formulation differs but is not the seat of the deficit.",
        ha="center", va="center", fontsize=9.7, color=RED, zorder=3)

ax.set_title("Two drift-diffusion engines: where they diverge",
             fontsize=15, fontweight="bold", pad=12)

fig.tight_layout()
fig.savefig(OUT, dpi=200, facecolor="white", bbox_inches="tight")
plt.close(fig)

import os
sz = os.path.getsize(OUT)
print("WROTE", OUT, sz)
assert sz > 20000, f"file too small: {sz}"
