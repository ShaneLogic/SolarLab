"""Refactored slide schematic for SolarLab interface SRH.

The figure intentionally separates the mechanism from the takeaway:

1. bulk equilibrium densities are Boltzmann-projected into interface-plane
   states by the available junction bending;
2. SRH then consumes cross-side carrier pairs through the interface trap.

Source of truth:
  perovskite_sim/physics/interface_plane.py
    - compute_interface_te_fluxes: projected interface-plane states
    - compute_interface_srh_on_state: cross-carrier SRH paths
  perovskite_sim/physics/recombination.py
    - interface_recombination: surface SRH form
"""
from __future__ import annotations

import os
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10.5,
    "mathtext.default": "regular",
})

PVK = "#3153a5"
ETL = "#8a4ab1"
ELECTRON = "#1f6fd6"
HOLE = "#d83b2d"
GREEN = "#16845a"
ORANGE = "#c86500"
INK = "#172033"
MUTED = "#5e6675"
GRID = "#d9dee9"
ROOT = "#b3261e"


def rounded_box(ax, xy, w, h, *, fc, ec="#d7dce8", lw=1.0, radius=0.02, z=1):
    box = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        transform=ax.transAxes,
        zorder=z,
    )
    ax.add_patch(box)
    return box


def wrapped(ax, x, y, text, width, *, size=9.5, color=INK, weight="normal",
            va="top", ha="left", linespacing=1.16, transform=None):
    if transform is None:
        transform = ax.transAxes
    ax.text(
        x,
        y,
        "\n".join(textwrap.wrap(text, width=width)),
        fontsize=size,
        color=color,
        fontweight=weight,
        va=va,
        ha=ha,
        linespacing=linespacing,
        transform=transform,
    )


def data_label(ax, x, y, text, *, color=INK, size=9.2, weight="bold",
               ha="center", va="center"):
    ax.text(
        x,
        y,
        text,
        color=color,
        fontsize=size,
        fontweight=weight,
        ha=ha,
        va=va,
        zorder=8,
    )


def pill(ax, x, y, text, *, color, fc="#ffffff", size=8.8):
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=size,
        color=color,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.26", fc=fc, ec=color, lw=1.0),
        zorder=8,
    )


def arrow(ax, start, end, *, color, lw=2.0, ls="solid", rad=0.0,
          scale=13, z=5, alpha=1.0):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=scale,
        linewidth=lw,
        linestyle=ls,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=8,
        shrinkB=10,
        zorder=z,
        alpha=alpha,
    )
    ax.add_patch(patch)
    return patch


def carrier(ax, x, y, label, *, color, symbol, label_dx=0.0, label_dy=0.0,
            label_ha="center"):
    ax.scatter([x], [y], s=245, color=color, edgecolor="white", lw=1.8, zorder=7)
    ax.text(
        x,
        y,
        symbol,
        color="white",
        fontsize=14,
        fontweight="bold",
        ha="center",
        va="center",
        zorder=8,
    )
    data_label(
        ax,
        x + label_dx,
        y + label_dy,
        label,
        color=color,
        size=10.2,
        ha=label_ha,
    )


def add_card(ax, y, h, number, title, body, *, accent, fc="#ffffff",
             body_width=42, body_size=9.2):
    rounded_box(ax, (0.0, y), 1.0, h, fc=fc, ec="#ccd4e3", lw=1.0, radius=0.03)
    ax.text(
        0.045,
        y + h - 0.065,
        number,
        color="white",
        fontsize=9.2,
        fontweight="bold",
        ha="center",
        va="center",
        transform=ax.transAxes,
        bbox=dict(boxstyle="circle,pad=0.26", fc=accent, ec=accent, lw=0.0),
        zorder=4,
    )
    ax.text(
        0.092,
        y + h - 0.045,
        title,
        color=accent,
        fontsize=11.2,
        fontweight="bold",
        va="top",
        transform=ax.transAxes,
    )
    wrapped(ax, 0.058, y + h - 0.112, body, body_width, size=body_size, color=INK)


fig = plt.figure(figsize=(13.333, 7.5), dpi=200)
fig.patch.set_facecolor("white")

# Header
fig.text(
    0.04,
    0.952,
    "SolarLab interface SRH",
    color=INK,
    fontsize=18,
    fontweight="bold",
    ha="left",
    va="top",
)
fig.text(
    0.04,
    0.914,
    "Bulk-projected interface states feed cross-carrier recombination",
    color=MUTED,
    fontsize=11.5,
    ha="left",
    va="top",
)

axB = fig.add_axes([0.04, 0.105, 0.61, 0.78])
axB.set_xlim(0, 10)
axB.set_ylim(0, 5.2)
axB.axis("off")

axR = fig.add_axes([0.685, 0.105, 0.275, 0.78])
axR.axis("off")

# Layer backgrounds
axB.add_patch(Rectangle((0.0, 0.0), 4.75, 5.2, facecolor="#edf3ff", edgecolor="none", zorder=0))
axB.add_patch(Rectangle((5.25, 0.0), 4.75, 5.2, facecolor="#f6efff", edgecolor="none", zorder=0))
axB.add_patch(Rectangle((4.75, 0.0), 0.50, 5.2, facecolor="#e1e5ec", edgecolor="none", zorder=0))
axB.axvline(5.0, color="#9098a8", lw=1.3, ls=(0, (4, 3)), zorder=1)

axB.text(2.35, 4.96, "PEROVSKITE / side 2", color=PVK, fontsize=12,
         fontweight="bold", ha="center", va="center")
axB.text(7.65, 4.96, "ETL / side 1", color=ETL, fontsize=12,
         fontweight="bold", ha="center", va="center")
axB.text(5.05, 0.23, "interface plane", color=MUTED, fontsize=9.4,
         fontstyle="italic", ha="center", va="center")

# Band edges and offsets
Ec_pvk, Ev_pvk = 3.55, 1.55
Ec_etl, Ev_etl = 3.23, 0.63
for x0, x1, y, color in [
    (0.55, 4.62, Ec_pvk, PVK),
    (0.55, 4.62, Ev_pvk, PVK),
    (5.38, 9.45, Ec_etl, ETL),
    (5.38, 9.45, Ev_etl, ETL),
]:
    axB.plot([x0, x1], [y, y], color=color, lw=3.3, solid_capstyle="round", zorder=3)

axB.plot([4.62, 5.38], [Ec_pvk, Ec_etl], color="#697080", lw=1.3, ls=(0, (2, 2)), zorder=3)
axB.plot([4.62, 5.38], [Ev_pvk, Ev_etl], color="#697080", lw=1.3, ls=(0, (2, 2)), zorder=3)
axB.annotate("", (5.14, Ec_pvk - 0.02), (5.14, Ec_etl + 0.02),
             arrowprops=dict(arrowstyle="<->", color=MUTED, lw=1.1))
axB.annotate("", (4.62, Ev_pvk - 0.02), (4.62, Ev_etl + 0.02),
             arrowprops=dict(arrowstyle="<->", color=MUTED, lw=1.1))
data_label(axB, 5.48, (Ec_pvk + Ec_etl) / 2, r"$\Delta E_C$", color=MUTED, size=9.4, ha="left")
data_label(axB, 4.28, (Ev_pvk + Ev_etl) / 2, r"$\Delta E_V$", color=MUTED, size=9.4, ha="right")
axB.text(0.70, Ec_pvk + 0.18, r"$E_C$", color=PVK, fontsize=10.3)
axB.text(0.70, Ev_pvk - 0.32, r"$E_V$", color=PVK, fontsize=10.3)

# Interface trap and carrier state markers
Et = 2.28
axB.plot([4.55, 5.45], [Et, Et], color="#111827", lw=2.2, zorder=5)
data_label(axB, 5.52, Et, r"$E_t$ trap", color="#111827", size=9.2, ha="left")

xL, xR = 4.36, 5.64
carrier(axB, xL, Ec_pvk, r"$n_{2s}$", color=ELECTRON, symbol="-", label_dy=0.38)
carrier(axB, xR, Ec_etl, r"$n_{1s}$", color=ELECTRON, symbol="-", label_dy=0.38)
carrier(axB, xL, Ev_pvk, r"$p_{2s}$", color=HOLE, symbol="+", label_dy=-0.42)
carrier(axB, xR, Ev_etl, r"$p_{1s}$", color=HOLE, symbol="+", label_dy=-0.42)

# Bulk reservoirs and projection arrows
pill(axB, 1.25, 4.08, r"$n_{L,eq}$", color=ELECTRON, fc="#f8fbff")
pill(axB, 1.25, 1.02, r"$p_{L,eq}$", color=HOLE, fc="#fff8f7")
pill(axB, 8.75, 3.78, r"$n_{R,eq}$", color=ELECTRON, fc="#f8fbff")
pill(axB, 8.75, 0.30, r"$p_{R,eq}$", color=HOLE, fc="#fff8f7")

arrow(axB, (1.62, 4.08), (xL, Ec_pvk), color=ELECTRON, lw=1.6, ls=(0, (4, 3)), rad=-0.03, alpha=0.9)
arrow(axB, (1.62, 1.02), (xL, Ev_pvk), color=HOLE, lw=1.6, ls=(0, (4, 3)), rad=0.03, alpha=0.9)
arrow(axB, (8.38, 3.78), (xR, Ec_etl), color=ELECTRON, lw=1.6, ls=(0, (4, 3)), rad=0.04, alpha=0.9)
arrow(axB, (8.38, 0.30), (xR, Ev_etl), color=HOLE, lw=1.6, ls=(0, (4, 3)), rad=-0.04, alpha=0.9)

axB.text(
    5.0,
    4.42,
    r"TE projection:  $J_{TE}=v_{th}\,(density_{bulk,proj}-density_{state})$",
    ha="center",
    va="center",
    fontsize=9.7,
    color=INK,
    bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=GRID, lw=1.0),
    zorder=9,
)
axB.text(
    5.0,
    4.05,
    r"$V_{total}=V_{bi,eff}-V_{app}$  split into $V_2$ (PVK) and $V_1$ (ETL)",
    ha="center",
    va="center",
    fontsize=9.0,
    color=MUTED,
    bbox=dict(boxstyle="round,pad=0.22", fc="#f8fafc", ec=GRID, lw=0.9),
    zorder=9,
)

# Cross-carrier SRH arrows
arrow(axB, (xR, Ec_etl), (5.03, Et + 0.02), color=GREEN, lw=2.5, rad=0.18, scale=15)
arrow(axB, (xL, Ev_pvk), (4.97, Et - 0.02), color=GREEN, lw=2.5, rad=-0.20, scale=15)
arrow(axB, (xL, Ec_pvk), (4.95, Et + 0.02), color=ORANGE, lw=2.5, rad=-0.20, scale=15)
arrow(axB, (xR, Ev_etl), (5.05, Et - 0.02), color=ORANGE, lw=2.5, rad=0.18, scale=15)

rounded_box(axB, (0.19, 0.45), 0.28, 0.145, fc="#ffffff", ec="#b8c1d0", lw=1.0, radius=0.025, z=6)
axB.text(0.215, 0.562, r"$R_{s1}=SRH(n_{1s},p_{2s})$", color=GREEN,
         fontsize=10.1, fontweight="bold", transform=axB.transAxes, va="top", zorder=8)
axB.text(0.215, 0.505, "ETL electron + PVK hole", color=MUTED,
         fontsize=8.6, transform=axB.transAxes, va="top", zorder=8)

rounded_box(axB, (0.61, 0.29), 0.30, 0.145, fc="#ffffff", ec="#b8c1d0", lw=1.0, radius=0.025, z=6)
axB.text(0.635, 0.402, r"$R_{s2}=SRH(n_{2s},p_{1s})$", color=ORANGE,
         fontsize=10.1, fontweight="bold", transform=axB.transAxes, va="top", zorder=8)
axB.text(0.635, 0.345, "PVK electron + ETL hole", color=MUTED,
         fontsize=8.6, transform=axB.transAxes, va="top", zorder=8)

axB.text(
    5.0,
    3.02,
    "cross-carrier pairs",
    ha="center",
    va="center",
    fontsize=10.0,
    color=INK,
    fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.28", fc="#fffaf0", ec="#e3b76c", lw=1.0),
    zorder=10,
)

# Right-side takeaway cards
axR.text(
    0.0,
    1.0,
    "Key points",
    transform=axR.transAxes,
    color=INK,
    fontsize=15,
    fontweight="bold",
    ha="left",
    va="top",
)

add_card(
    axR,
    0.735,
    0.205,
    "1",
    "Bulk projection",
    "Bulk equilibrium densities are projected to the interface plane by the remaining bending:",
    accent=ELECTRON,
    fc="#f8fbff",
    body_width=45,
    body_size=9.0,
)
axR.text(
    0.058,
    0.755,
    r"$V_{total}=max(V_{bi,eff}-V_{app},0)$",
    color="#16407a",
    fontsize=9.0,
    transform=axR.transAxes,
)

add_card(
    axR,
    0.485,
    0.185,
    "2",
    "Cross-carrier SRH",
    "The trap pairs opposite-side carriers.",
    accent=GREEN,
    fc="#f7fcf9",
    body_width=43,
    body_size=9.0,
)
axR.text(
    0.058,
    0.515,
    r"$R_{s1}=SRH(n_{1s},p_{2s})$     $R_{s2}=SRH(n_{2s},p_{1s})$",
    color="#0f5c39",
    fontsize=8.6,
    transform=axR.transAxes,
)

rounded_box(axR, (0.0, 0.255), 1.0, 0.165, fc="#fff7ed", ec="#f0b36a", lw=1.1, radius=0.03)
axR.text(0.058, 0.375, "3", color="white", fontsize=9.2, fontweight="bold",
         ha="center", va="center", transform=axR.transAxes,
         bbox=dict(boxstyle="circle,pad=0.26", fc=ORANGE, ec=ORANGE, lw=0.0))
axR.text(0.092, 0.397, "Band-step coupling", color=ORANGE,
         fontsize=11.0, fontweight="bold", va="top", transform=axR.transAxes)
wrapped(
    axR,
    0.058,
    0.327,
    "The chi/Eg step links side-1 and side-2 states, so the rate response is structural rather than a calibration knob.",
    43,
    size=8.9,
)

rounded_box(axR, (0.0, 0.02), 1.0, 0.18, fc="#fdecea", ec=ROOT, lw=1.2, radius=0.03)
axR.text(0.045, 0.17, "Root-cause consequence", color=ROOT,
         fontsize=10.8, fontweight="bold", va="top", transform=axR.transAxes)
wrapped(
    axR,
    0.045,
    0.124,
    "Frozen V_bi means frozen projection bending. ETL doping then cannot move interface recombination correctly, severing the V_oc lever.",
    50,
    size=8.65,
    color="#7a1a12",
)

fig.text(
    0.04,
    0.038,
    "Sources: interface_plane.py:63-177 (TE projection), 180-268 (cross-carrier SRH); recombination.py:28-44 (surface SRH form).",
    color=MUTED,
    fontsize=8.8,
    ha="left",
    va="bottom",
)

out = os.path.join(os.path.dirname(__file__), "f8_interface_srh.png")
fig.savefig(out, dpi=200, facecolor="white")
print("WROTE", out, os.path.getsize(out), "bytes")
