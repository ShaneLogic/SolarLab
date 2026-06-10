"""Slide schematic for SolarLab's thermionic-emission (TE) flux-cap on the
Scharfetter-Gummel (SG) flux at heterointerfaces.

The figure separates mechanism from quantitative behaviour:

  LEFT  band diagram of a PVK/ETL conduction-band step resolved in a SINGLE
        grid cell h. SG interprets the offset as an enormous drift field
        E = dE_c / (q*h) and over-injects; TE is the physical Richardson
        ceiling for emission over the barrier.

  RIGHT |J_SG|, |J_TE| and the capped flux J_used = sign(J_SG)*min(|J_SG|,|J_TE|)
        versus barrier height dE_c. Shaded band = where the cap actually binds.

Source of truth:
  perovskite_sim/discretization/fe_operators.py
    - sg_fluxes_n / sg_fluxes_p   (SG exponentially-fitted flux)   lines 58-81
    - thermionic_emission_flux    (Richardson-Dushman TE flux)     lines 84-98
  perovskite_sim/physics/continuity.py
    - carrier_continuity_rhs      (the |J_SG|>|J_TE| cap)          lines 144-181

Note (root-cause caveat, annotated on the plot): the shipped TE primitive
returns A*T^2 * n  (Richardson constant x density), which is ~N_c larger than
the textbook current q*v_R*n with v_R = A*T^2/(q*N_c). On the shipped stack
that pushes the TE ceiling far above the SG flux, so the cap rarely engages
(consistent with the observed TE-toggle having ~no effect on V_oc).
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

# ── shared palette / type (matches make_f8_interface_srh.py) ─────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10.5,
    "mathtext.default": "regular",
})

PVK = "#3153a5"
ETL = "#8a4ab1"
ELECTRON = "#1f6fd6"
SG = "#c86500"      # orange  -> SG flux
TE = "#16845a"      # green   -> TE ceiling
USED = "#b3261e"    # red     -> capped flux actually used
INK = "#172033"
MUTED = "#5e6675"
GRID = "#d9dee9"
SHADE = "#fde2cf"

# ── physical constants (SI) ─────────────────────────────────────────────────
Q = 1.602176634e-19
K_B = 1.380649e-23
T = 300.0
V_T = K_B * T / Q            # ~0.02585 V

# representative interface-face parameters
D_N = 1.0e-4                 # m^2/s  (electron diffusivity)
H = 1.0e-9                   # m      (clustered interface grid spacing)
N_L = 1.0e22                 # m^-3   carrier density, PVK side of face
N_R = 1.0e22                 # m^-3   carrier density, ETL side of face
A_STAR = 1.2017e6            # A/(m^2 K^2)  free-electron Richardson constant
N_C = 2.2e24                 # m^-3   effective CB DOS (MAPbI3)
V_R = A_STAR * T**2 / (Q * N_C)   # textbook thermionic emission velocity [m/s]


def bernoulli(x: np.ndarray) -> np.ndarray:
    """B(x)=x/(exp(x)-1), numerically stable (mirrors fe_operators.bernoulli)."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    small = np.abs(x) < 1e-8
    huge = x > 700.0
    rest = ~small & ~huge
    out[small] = 1.0 - x[small] / 2.0 + x[small]**2 / 12.0
    out[huge] = 0.0
    out[rest] = x[rest] / np.expm1(x[rest])
    return out


def j_sg(dEc: np.ndarray) -> np.ndarray:
    """SG electron flux at the face for a pure band step (electrostatic Dphi=0).

    phi_n = phi + chi  ->  xi = (Dphi - dEc)/V_T = -dEc/V_T here.
    """
    xi = -dEc / V_T
    return Q * D_N / H * (bernoulli(xi) * N_R - bernoulli(-xi) * N_L)


def j_te(dEc: np.ndarray) -> np.ndarray:
    """Physical Richardson TE flux J = q*v_R*(n_L e^- - n_R e^-) [A/m^2]."""
    fwd = N_L * np.exp(-np.maximum(dEc, 0.0) / V_T)
    bwd = N_R * np.exp(-np.maximum(-dEc, 0.0) / V_T)
    return Q * V_R * (fwd - bwd)


def main() -> None:
    fig = plt.figure(figsize=(13.2, 5.7))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0], wspace=0.22,
                          left=0.045, right=0.975, top=0.86, bottom=0.12)
    ax_band = fig.add_subplot(gs[0, 0])
    ax_flux = fig.add_subplot(gs[0, 1])

    fig.suptitle(
        "TE flux-cap on the Scharfetter–Gummel flux at a heterointerface",
        fontsize=15, fontweight="bold", color=INK, x=0.5, y=0.965,
    )

    # ════════════════════════════════════════════════════════════════════════
    # LEFT — band diagram, single-cell band step
    # ════════════════════════════════════════════════════════════════════════
    ax = ax_band
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.text(0.0, 9.7, "(a)  band step resolved in one grid cell",
            fontsize=12, fontweight="bold", color=INK)

    # region tints
    ax.axvspan(0.6, 4.5, ymin=0.04, ymax=0.96, color=PVK, alpha=0.06)
    ax.axvspan(5.5, 9.6, ymin=0.04, ymax=0.96, color=ETL, alpha=0.07)
    ax.text(2.55, 1.05, "perovskite", color=PVK, fontsize=11,
            fontweight="bold", ha="center")
    ax.text(7.55, 1.05, "ETL", color=ETL, fontsize=11,
            fontweight="bold", ha="center")

    # conduction-band edge: low on PVK, step UP by dE_c onto ETL (barrier)
    cb_pvk, cb_etl = 5.0, 7.3
    ax.plot([0.6, 4.5], [cb_pvk, cb_pvk], color=PVK, lw=3.2, solid_capstyle="round")
    ax.plot([5.5, 9.6], [cb_etl, cb_etl], color=ETL, lw=3.2, solid_capstyle="round")
    # true (vertical) step at the interface
    ax.plot([5.0, 5.0], [cb_pvk, cb_etl], color=MUTED, lw=2.0, ls=(0, (1, 1)))
    ax.text(9.5, cb_pvk - 0.05, "$E_C$", color=INK, fontsize=11, ha="right", va="top")

    # grid nodes i (PVK side) and i+1 (ETL side) bracketing the face
    xi_node, xip_node = 4.5, 5.5
    for xn, yn, lab, col in [(xi_node, cb_pvk, "node $i$", PVK),
                             (xip_node, cb_etl, "node $i{+}1$", ETL)]:
        ax.plot([xn], [yn], "o", color=col, ms=9, zorder=6)
        ax.plot([xn, xn], [0.6, yn], color=GRID, lw=1.0, ls=":", zorder=1)
        ax.text(xn, 0.35, lab, color=col, fontsize=9.5, ha="center", va="top")

    # grid-cell width h marker
    ax.annotate("", xy=(xi_node, 8.5), xytext=(xip_node, 8.5),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=1.4))
    ax.text(5.0, 8.75, "grid cell  $h$  (~1 nm)", color=INK, fontsize=9.5,
            ha="center", va="bottom")

    # dE_c marker
    ax.annotate("", xy=(5.05, cb_pvk), xytext=(5.05, cb_etl),
                arrowprops=dict(arrowstyle="<->", color=USED, lw=1.6))
    ax.text(5.2, (cb_pvk + cb_etl) / 2, "$\\Delta E_C$", color=USED,
            fontsize=12, ha="left", va="center", fontweight="bold")

    # SG interpretation: straight ramp across the single cell (huge field)
    ax.plot([xi_node, xip_node], [cb_pvk, cb_etl], color=SG, lw=2.4,
            ls=(0, (5, 2)), zorder=5)
    ax.text(4.0, 6.55, "SG: linear ramp\n$\\Rightarrow E=\\Delta E_C/(qh)$ huge\n"
                       "$\\Rightarrow$ over-injects",
            color=SG, fontsize=9.2, ha="center", va="center")

    # TE: electron emitted over the barrier (Richardson)
    e_x = 2.6
    for dy in (0.0, 0.42, 0.84):
        ax.plot([e_x + 0.0], [cb_pvk + 0.55 + dy], "o", color=ELECTRON,
                ms=6.5, zorder=6)
    arc = FancyArrowPatch((4.45, cb_pvk + 0.6), (5.65, cb_etl + 0.55),
                          connectionstyle="arc3,rad=-0.45",
                          arrowstyle="-|>", mutation_scale=16,
                          color=TE, lw=2.2, zorder=6)
    ax.add_patch(arc)
    ax.text(7.55, cb_etl + 0.95,
            "TE ceiling: Richardson\nemission over $\\Delta E_C$\n"
            "$J_{TE}\\propto e^{-\\Delta E_C/V_T}$",
            color=TE, fontsize=9.0, ha="center", va="center")

    ax.text(5.0, 3.55,
            "Cap:  if  $|J_{SG}|>|J_{TE}|$   then   $J\\leftarrow J_{TE}$",
            ha="center", fontsize=10.5, color=INK, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.35", fc="#eef1f7", ec=GRID, lw=0.9))

    # ════════════════════════════════════════════════════════════════════════
    # RIGHT — quantitative flux vs barrier height
    # ════════════════════════════════════════════════════════════════════════
    ax = ax_flux
    dEc = np.linspace(0.0, 0.5, 400)
    Jsg = np.abs(j_sg(dEc))
    Jte = np.abs(j_te(dEc))
    Jused = np.minimum(Jsg, Jte)

    ax.set_yscale("log")
    ax.plot(dEc, Jsg, color=SG, lw=2.4, label="$|J_{SG}|$  (uncapped SG flux)")
    ax.plot(dEc, Jte, color=TE, lw=2.4, ls=(0, (5, 2)),
            label="$|J_{TE}|$  (Richardson ceiling)")
    ax.plot(dEc, Jused, color=USED, lw=3.4, alpha=0.85,
            label="$J_{used}=\\min(|J_{SG}|,|J_{TE}|)$")

    # cap-active region: where SG exceeds TE
    active = Jsg > Jte
    if active.any():
        ax.fill_between(dEc, 1e-2, 1e30, where=active, color=SHADE,
                        alpha=0.55, zorder=0, label="cap binds  ($|J_{SG}|>|J_{TE}|$)")
        # crossover marker
        idx = np.argmax(active) if active[0] else np.argmax(active)
        xc = dEc[idx]
        ax.axvline(xc, color=MUTED, lw=1.0, ls=":")
        ax.text(xc + 0.006, Jte[idx] * 4, f"cap activates\n$\\Delta E_C\\approx{xc:.2f}$ eV",
                color=MUTED, fontsize=8.6, ha="left", va="bottom")

    # 0.05 eV activation threshold from the code
    ax.axvline(0.05, color=INK, lw=1.0, ls=(0, (1, 2)))
    ax.text(0.052, Jused.max() * 0.5, "0.05 eV\nthreshold", color=INK,
            fontsize=8.4, ha="left", va="top")

    ax.set_xlim(0, 0.5)
    ax.set_ylim(max(Jused.min(), 1e3), Jsg.max() * 6)
    ax.set_xlabel("conduction-band offset  $\\Delta E_C$  (eV)", fontsize=11)
    ax.set_ylabel("flux magnitude  |J|  (A m$^{-2}$)", fontsize=11)
    ax.text(0.0, 1.02, "(b)  cap takes the smaller of the two fluxes",
            transform=ax.transAxes, fontsize=12, fontweight="bold", color=INK)
    ax.grid(True, which="both", color=GRID, lw=0.6, alpha=0.7)
    ax.legend(loc="lower left", fontsize=8.6, framealpha=0.95,
              edgecolor=GRID, ncol=1)

    # root-cause caveat box
    ax.text(0.985, 0.985,
            "shipped TE returns $A^*T^2 n$ (no $/qN_c$):\n"
            f"ceiling ~$N_c$≈{N_C:.0e} m$^{{-3}}$ higher\n"
            "$\\Rightarrow$ cap rarely engages in practice",
            transform=ax.transAxes, fontsize=8.0, color=USED,
            ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.4", fc="#fff4f2",
                      ec=USED, lw=0.9, alpha=0.95))

    fig.text(0.5, 0.018,
             "source: fe_operators.py:58–98 (SG + TE primitives) · "
             "continuity.py:144–181 (cap) · activates only where "
             "$|\\Delta E_C|$ or $|\\Delta E_V|>0.05$ eV",
             ha="center", fontsize=8.0, color=MUTED)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "f9_te_flux_cap.png")
    fig.savefig(out, dpi=200, facecolor="white")
    print(f"wrote {out}")
    print(f"  V_T={V_T*1e3:.3f} mV   v_R={V_R:.3e} m/s   "
          f"crossover dEc={dEc[np.argmax(Jsg>Jte)]:.3f} eV")


if __name__ == "__main__":
    main()
