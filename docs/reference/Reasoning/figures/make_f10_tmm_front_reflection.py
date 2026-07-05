"""Slide schematic for SolarLab's TMM front reflection.

LEFT   energy-flow schematic of the hybrid incoherent-front / coherent-substack
       optical model: air -> thick glass (incoherent Fresnel + Beer-Lambert)
       -> coherent thin-film stack (spiro / MAPbI3 / TiO2). The front Fresnel
       bounce R_front and the coherent sub-stack reflection R_sub are the two
       pieces of the reflected power; only (1-R) of AM1.5G ever reaches G(x).

RIGHT  the REAL spectral reflectance R(lambda) of the shipped nip_MAPbI3_tmm
       stack, computed with the project's own optics module, decomposed into
       the flat ~4% air/glass Fresnel floor and the interference fringes the
       coherent sub-stack adds on top.

Source of truth:
  perovskite_sim/physics/optics.py
    - _transfer_matrix_stack   Fresnel interface + propagation matrices  (49-137)
    - tmm_reflectance          R = |r|^2, incoherent-front bypass         (450-494)
    - _incoherent_front_factors R_front = |r0|^2, T_bulk = exp(-alpha d)  (419-447)
    - tmm_absorption_profile   A = alpha * (n/n_ambient) * |E|^2          (273-)
  perovskite_sim/solver/mol.py:_compute_tmm_generation (builds the stack)
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

from perovskite_sim.data import load_nk
from perovskite_sim.physics.optics import (
    TMMLayer,
    _incoherent_front_factors,
    _transfer_matrix_stack,
    tmm_reflectance,
)

# ── shared palette / type (matches make_f8 / make_f9) ────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10.5,
    "mathtext.default": "regular",
})

AIR = "#eef3fb"
GLASS = "#cfe3f0"
HTLc = "#8a4ab1"
PVKc = "#3153a5"
ETLc = "#16845a"
SUN = "#e0a106"
RFRONT = "#c86500"   # orange  -> front Fresnel
RSUB = "#b3261e"     # red     -> coherent sub-stack
ENTER = "#1f6fd6"    # blue    -> power entering the cell
INK = "#172033"
MUTED = "#5e6675"
GRID = "#d9dee9"

# ── shipped nip_MAPbI3_tmm stack ─────────────────────────────────────────────
SPECS = [
    ("glass", 1.0e-3, True, "glass"),
    ("spiro_OMeTAD", 200e-9, False, "HTL"),
    ("MAPbI3", 400e-9, False, "absorber"),
    ("TiO2", 100e-9, False, "ETL"),
]


def build_reflectance():
    wl_nm = np.linspace(300.0, 850.0, 400)
    wl_m = wl_nm * 1e-9
    layers = []
    for mat, d, inc, _role in SPECS:
        _, n, k = load_nk(mat, wl_nm)
        layers.append(TMMLayer(d=d, n=n, k=k, incoherent=inc))

    R = tmm_reflectance(layers, wl_m, n_ambient=1.0, n_substrate=1.0)
    R_front, T_bulk, n_real = _incoherent_front_factors(layers[0], wl_m, 1.0)
    S_sub, _ = _transfer_matrix_stack(layers[1:], wl_m, n_ambient=n_real,
                                      n_substrate=1.0)
    r_sub = S_sub[:, 1, 0] / S_sub[:, 0, 0]
    R_sub = np.abs(r_sub) ** 2
    return wl_nm, R, R_front, R_sub, n_real


def main() -> None:
    wl_nm, R, R_front, R_sub, n_glass = build_reflectance()

    fig = plt.figure(figsize=(13.4, 5.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.08, 1.0], wspace=0.2,
                          left=0.035, right=0.975, top=0.85, bottom=0.13)
    ax_s = fig.add_subplot(gs[0, 0])
    ax_r = fig.add_subplot(gs[0, 1])

    fig.suptitle("TMM front reflection — only $(1-R)$ of AM1.5G enters the cell",
                 fontsize=15, fontweight="bold", color=INK, x=0.5, y=0.96)

    # ════════════════════════════════════════════════════════════════════════
    # LEFT — energy-flow schematic
    # ════════════════════════════════════════════════════════════════════════
    ax = ax_s
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.text(0.0, 9.7, "(a)  hybrid incoherent-front / coherent-substack model",
            fontsize=12, fontweight="bold", color=INK)

    yb, yt = 1.6, 7.4   # layer band bottom / top
    bands = [
        (0.4, 1.4, AIR, "air\n$n{=}1$", INK),
        (1.4, 4.4, GLASS, "glass  ~1 mm\n(incoherent)", INK),
        (4.4, 5.6, HTLc, "spiro\nHTL", "white"),
        (5.6, 7.6, PVKc, "MAPbI$_3$\nabsorber", "white"),
        (7.6, 8.5, ETLc, "TiO$_2$\nETL", "white"),
        (8.5, 9.4, AIR, "back", INK),
    ]
    for x0, x1, col, lab, tc in bands:
        ax.add_patch(plt.Rectangle((x0, yb), x1 - x0, yt - yb, facecolor=col,
                                   edgecolor="white", lw=1.4, zorder=2))
        ax.text((x0 + x1) / 2, yb + 0.32, lab, ha="center", va="bottom",
                fontsize=8.6, color=tc, zorder=4)
    # hatch the incoherent glass slab
    ax.add_patch(plt.Rectangle((1.4, yb), 3.0, yt - yb, facecolor="none",
                               edgecolor="white", hatch="////", lw=0.0,
                               alpha=0.5, zorder=3))

    air_glass = 1.4    # front Fresnel face
    glass_sub = 4.4    # glass -> coherent stack face
    ymid = (yb + yt) / 2 + 0.6

    # incident AM1.5G
    ax.annotate("", xy=(air_glass, ymid), xytext=(0.2, ymid + 1.7),
                arrowprops=dict(arrowstyle="-|>", color=SUN, lw=3.4))
    ax.text(0.05, ymid + 1.95, "AM1.5G  100%", color="#9a6f00",
            fontsize=10, fontweight="bold")

    # front Fresnel reflection R_front
    arc1 = FancyArrowPatch((air_glass, ymid), (0.5, ymid + 2.7),
                           connectionstyle="arc3,rad=0.28", arrowstyle="-|>",
                           mutation_scale=15, color=RFRONT, lw=2.6, zorder=6)
    ax.add_patch(arc1)
    ax.text(0.45, ymid + 2.95,
            "$R_{front}=\\left(\\dfrac{n_{air}-n_{glass}}{n_{air}+n_{glass}}\\right)^2"
            "\\approx 4\\%$", color=RFRONT, fontsize=9.4)

    # transmitted into glass (1 - R_front), Beer-Lambert T_bulk
    ax.annotate("", xy=(glass_sub, ymid), xytext=(air_glass, ymid),
                arrowprops=dict(arrowstyle="-|>", color=ENTER, lw=2.8))
    ax.text((air_glass + glass_sub) / 2, ymid + 0.28,
            "$(1{-}R_{front})\\times T_{bulk}$", color=ENTER, fontsize=8.8,
            ha="center")
    ax.text((air_glass + glass_sub) / 2, ymid - 0.42,
            "glass: $T_{bulk}=e^{-\\alpha d}\\approx1$\n(power form, no fringes)",
            color=MUTED, fontsize=7.6, ha="center", va="top")

    # coherent sub-stack reflection R_sub (interference)
    arc2 = FancyArrowPatch((glass_sub, ymid), (air_glass + 0.15, ymid + 1.7),
                           connectionstyle="arc3,rad=-0.3", arrowstyle="-|>",
                           mutation_scale=14, color=RSUB, lw=2.4, zorder=6)
    ax.add_patch(arc2)
    ax.text(3.05, ymid + 2.0,
            "$R_{sub}$: coherent TMM\nof thin films\n(interference fringes)",
            color=RSUB, fontsize=8.6, ha="center")

    # power entering the absorber -> G(x): wavy absorbed arrows in MAPbI3
    ax.annotate("", xy=(7.55, ymid), xytext=(glass_sub, ymid),
                arrowprops=dict(arrowstyle="-|>", color=ENTER, lw=2.4))
    for xx in (6.0, 6.55, 7.1):
        ax.annotate("", xy=(xx, yb + 0.9), xytext=(xx, ymid - 0.05),
                    arrowprops=dict(arrowstyle="-|>", color="#f2c84b", lw=1.8))
    ax.text(6.6, yb + 0.55, "absorbed  $\\to G(x)$", color="#7a5c00",
            fontsize=8.6, ha="center", va="top")

    # total reflectance formula
    ax.text(5.0, 0.55,
            "$R = R_{front} + (1-R_{front})^2\\,T_{bulk}^{2}\\,R_{sub}$",
            ha="center", fontsize=11.5, color=INK, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", fc="#eef1f7", ec=GRID, lw=0.9))

    # ════════════════════════════════════════════════════════════════════════
    # RIGHT — real spectral reflectance
    # ════════════════════════════════════════════════════════════════════════
    ax = ax_r
    Rpct = R * 100.0
    Rf_pct = R_front * 100.0

    ax.fill_between(wl_nm, 0, Rf_pct, color=RFRONT, alpha=0.28,
                    label="$R_{front}$  air/glass Fresnel (~4%)")
    ax.fill_between(wl_nm, Rf_pct, Rpct, color=RSUB, alpha=0.22,
                    label="coherent sub-stack (interference)")
    ax.plot(wl_nm, Rpct, color=RSUB, lw=2.4, label="total $R(\\lambda)$")
    ax.plot(wl_nm, Rf_pct, color=RFRONT, lw=1.8, ls=(0, (5, 2)))

    ax.axhline(Rpct.mean(), color=MUTED, lw=1.0, ls=":")
    ax.text(845, Rpct.mean() + 0.4, f"mean R ≈ {Rpct.mean():.1f}%",
            color=MUTED, fontsize=8.6, ha="right", va="bottom")

    ax.set_xlim(300, 850)
    ax.set_ylim(0, max(Rpct.max() * 1.12, 12))
    ax.set_xlabel("wavelength  $\\lambda$  (nm)", fontsize=11)
    ax.set_ylabel("reflectance  $R$  (%)", fontsize=11)
    ax.text(0.0, 1.02, "(b)  real R(λ) of shipped nip_MAPbI3_tmm stack",
            transform=ax.transAxes, fontsize=12, fontweight="bold", color=INK)
    ax.grid(True, color=GRID, lw=0.6, alpha=0.7)
    ax.legend(loc="upper right", fontsize=8.6, framealpha=0.95, edgecolor=GRID)

    ax.text(0.985, 0.52,
            f"$(1-R)$ enters G(x):\n≈{100 - Rpct.mean():.0f}% of AM1.5G\n"
            "fringes ride the\n4% Fresnel floor",
            transform=ax.transAxes, fontsize=8.4, color=ENTER, ha="right",
            va="top",
            bbox=dict(boxstyle="round,pad=0.4", fc="#eef5ff", ec=ENTER,
                      lw=0.9, alpha=0.95))

    fig.text(0.5, 0.02,
             "source: optics.py:49–137 (Fresnel + propagation) · 450–494 "
             "(tmm_reflectance) · 419–447 (incoherent front) · "
             "A = α·(n/n$_{amb}$)·|E|² gives R+T+A = 1",
             ha="center", fontsize=8.0, color=MUTED)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "f10_tmm_front_reflection.png")
    fig.savefig(out, dpi=200, facecolor="white")
    print(f"wrote {out}")
    print(f"  n_glass≈{np.median(n_glass):.3f}  R_front≈{R_front.mean()*100:.2f}%  "
          f"R_total≈{R.mean()*100:.2f}%  (1-R)≈{(1-R.mean())*100:.1f}%")


if __name__ == "__main__":
    main()
