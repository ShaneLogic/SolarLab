#!/usr/bin/env python3
"""Explanatory schematics for the SolarLab-vs-SCAPS-1D discrepancy analysis.

These figures are pedagogical schematics (not solver output) that visualise the
four mechanisms identified in the validation report:
  1. open-circuit-voltage deficit -> dark-saturation-current accounting and
     quasi-Fermi-level separation lost across heterojunction band offsets;
  2. donor-doping divergences      -> contact / built-in-potential and
     high-injection behaviour;
  3. bulk-defect insensitivity     -> the interface-recombination-limited regime;
  4. short-circuit-current residual -> front-surface reflection (R + T + A).

House style: Arial, SolarLab solid blue, SCAPS dashed red, matching the existing
validation overlays. All callouts are placed in empty regions and boxed for
legibility.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = ["Arial", "DejaVu Sans"]
plt.rcParams["mathtext.default"] = "regular"
plt.rcParams["axes.unicode_minus"] = True
plt.rcParams["savefig.dpi"] = 150

BLUE = "#1f4fd8"   # SolarLab
RED = "#d81f2a"    # SCAPS
GREY = "#555555"
OUT = Path(__file__).resolve().parent
kT = 0.02585       # kT/q at 300 K (V)

# light box for callouts so they read clearly even near a line
BOX = dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.92)


def save(fig, name):
    fig.savefig(OUT / name)
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------------------
# Fig 1 - the four figures of merit, side by side
# ---------------------------------------------------------------------------
def fig_overview():
    metrics = [
        ("$V_{oc}$ (V)", 1.118, 1.168, "$-$50 mV"),
        ("$J_{sc}$ (mA/cm$^2$)", 25.70, 26.28, "$-$2 %"),
        ("FF (%)", 87.9, 87.0, "$+$0.9 pp"),
        ("PCE (%)", 25.26, 26.69, "$-$1.4 pp"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.3), layout="constrained")
    for ax, (label, sl, sc, gap) in zip(axes, metrics):
        bars = ax.bar([0, 1], [sl, sc], width=0.6,
                      color=[BLUE, RED], edgecolor="black", linewidth=0.6)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["SolarLab", "SCAPS"], fontsize=9)
        ax.set_title(label, fontsize=11)
        top = max(sl, sc)
        ax.set_ylim(0, top * 1.22)
        for b, v in zip(bars, [sl, sc]):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:g}",
                    ha="center", va="bottom", fontsize=8.5)
        ax.text(0.5, top * 1.15, gap, ha="center", fontsize=10,
                color=GREY, weight="bold")
        ax.tick_params(labelsize=8)
    fig.suptitle("Base operating point: SolarLab versus SCAPS-1D",
                 fontsize=12.5)
    save(fig, "01_gap_overview.png")


# ---------------------------------------------------------------------------
# Fig 2 - V_oc deficit, diode-equation accounting (37x J_0)
# ---------------------------------------------------------------------------
def fig_voc_j0():
    fig, ax = plt.subplots(figsize=(7.4, 4.6), layout="constrained")
    ratio = np.logspace(0, 2, 300)
    voc = 1.168 - kT * np.log(ratio)
    ax.plot(ratio, voc, color="black", lw=2,
            label=r"$V_{oc}=V_{oc}^{\,SCAPS}-(kT/q)\,\ln(J_0/J_0^{\,SCAPS})$")
    ax.scatter([1], [1.168], color=RED, zorder=5, s=70, edgecolor="black",
               label=r"SCAPS ($J_0$ reference)")
    ax.scatter([37], [1.072], color=BLUE, zorder=5, s=70, edgecolor="black",
               label=r"SolarLab ($37\times J_0$)")
    ax.plot([37, 37], [1.072, 1.168], color=GREY, ls=":", lw=1.4)
    ax.plot([1, 37], [1.168, 1.168], color=GREY, ls=":", lw=1.0)
    ax.annotate(r"$(kT/q)\,\ln(37)\approx 93$ mV",
                xy=(37, 1.120), xytext=(3.2, 1.092), fontsize=11, color=GREY,
                bbox=BOX, arrowprops=dict(arrowstyle="->", color=GREY))
    ax.set_xscale("log")
    ax.set_xlabel(r"dark-saturation-current ratio  $J_0/J_0^{\,SCAPS}$",
                  fontsize=11)
    ax.set_ylabel(r"open-circuit voltage  $V_{oc}$ (V)", fontsize=11)
    ax.set_title(r"Diode-equation accounting of the $V_{oc}$ deficit",
                 fontsize=12)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_ylim(1.05, 1.18)
    save(fig, "02_voc_j0_lever.png")


# ---------------------------------------------------------------------------
# Fig 3 - V_oc deficit, QFL separation lost across band offsets
# ---------------------------------------------------------------------------
def fig_band_diagram():
    fig, ax = plt.subplots(figsize=(8.6, 5.0), layout="constrained")
    xH, xP1, xP2, xE = 0.0, 1.0, 6.0, 7.0
    Ec = {"HTL": 2.55, "PVK": 1.95, "ETL": 1.75}
    Ev = {"HTL": 0.20, "PVK": 0.42, "ETL": -0.55}

    def seg(x0, x1, y):
        ax.plot([x0, x1], [y, y], color="black", lw=2)

    for band in (Ec, Ev):
        seg(xH, xP1, band["HTL"]); seg(xP1, xP2, band["PVK"])
        seg(xP2, xE, band["ETL"])
        ax.plot([xP1, xP1], [band["HTL"], band["PVK"]], color="black", lw=2)
        ax.plot([xP2, xP2], [band["PVK"], band["ETL"]], color="black", lw=2)

    # Absorber quasi-Fermi levels: flat at the absorber's internal split.
    EFn, EFp = 1.55, 0.78
    ax.plot([xP1, xP2], [EFn, EFn], color=BLUE, ls="--", lw=1.9)
    ax.plot([xP1, xP2], [EFp, EFp], color=RED, ls="--", lw=1.9)
    ax.text(xP1 + 0.12, EFn + 0.07, r"$E_{Fn}$ (electron quasi-Fermi level)",
            color=BLUE, va="bottom", fontsize=10)
    ax.text(xP1 + 0.12, EFp - 0.07, r"$E_{Fp}$ (hole quasi-Fermi level)",
            color=RED, va="top", fontsize=10)

    # Pre-fix (omitted effective-DOS terms): spurious QFL steps of
    # kT*ln(DOS ratio) AT the heterojunctions — EFp jumps up entering the
    # HTL (kT*ln 25 = 83 meV), EFn drops entering the ETL (kT*ln 8 = 54 meV).
    dV_htl, dC_etl = 0.20, 0.14  # exaggerated for visibility
    ax.plot([xH, xP1], [EFp + dV_htl, EFp + dV_htl], color=RED, ls=":", lw=1.9)
    ax.plot([xP1, xP1], [EFp + dV_htl, EFp], color=RED, ls=":", lw=1.6)
    ax.plot([xP2, xE], [EFn - dC_etl, EFn - dC_etl], color=BLUE, ls=":", lw=1.9)
    ax.plot([xP2, xP2], [EFn, EFn - dC_etl], color=BLUE, ls=":", lw=1.6)
    # Corrected (dos_band_potentials): QFLs continue flat to the contacts.
    ax.plot([xH, xP1], [EFp, EFp], color=RED, ls="--", lw=1.9)
    ax.plot([xP2, xE], [EFn, EFn], color=BLUE, ls="--", lw=1.9)

    xm = 3.3
    ax.annotate("", xy=(xm, EFn), xytext=(xm, EFp),
                arrowprops=dict(arrowstyle="<->", color="green", lw=2.0))
    ax.text(xm + 0.18, (EFn + EFp) / 2,
            "absorber $\\Delta E_F$\n(internal voltage)",
            color="green", fontsize=9.5, va="center")

    ax.annotate("spurious $E_{Fp}$ step\n$kT\\,\\ln(N_{V,HTL}/N_{V,PVK})$"
                "\n$= kT\\,\\ln 25 = 83$ meV",
                xy=(xP1, EFp + dV_htl / 2), xytext=(1.95, -0.40),
                fontsize=8.5, color=GREY, ha="center", bbox=BOX,
                arrowprops=dict(arrowstyle="->", color=GREY))
    ax.annotate("spurious $E_{Fn}$ step\n$kT\\,\\ln(N_{C,ETL}/N_{C,PVK})$"
                "\n$= kT\\,\\ln 8 = 54$ meV",
                xy=(xP2, EFn - dC_etl / 2), xytext=(5.0, -0.40),
                fontsize=8.5, color=GREY, ha="center", bbox=BOX,
                arrowprops=dict(arrowstyle="->", color=GREY))
    ax.text(3.4, 2.42,
            "dotted: omitted effective-DOS terms $\\rightarrow$ the terminal reads the split\n"
            "minus 137 mV of junction steps.  dashed: corrected transport\n"
            "($V_T\\ln N_C$ / $V_T\\ln N_V$ included) — the steps vanish.",
            color="black", fontsize=9.0, ha="center", va="center", bbox=BOX)

    for x0, x1, name, c in [(xH, xP1, "HTL\n(spiro)", "#eef2ff"),
                            (xP1, xP2, "perovskite (MAPbI$_3$)", "#fff7ee"),
                            (xP2, xE, "ETL\n(TiO$_2$)", "#eefcf0")]:
        ax.axvspan(x0, x1, color=c, zorder=0)
        ax.text((x0 + x1) / 2, 2.86, name, ha="center", fontsize=9.5)

    ax.set_xlim(xH, xE + 0.9)
    ax.set_ylim(-0.85, 3.05)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_ylabel("electron energy", fontsize=11)
    ax.set_title("Spurious quasi-Fermi-level steps from the omitted effective-DOS\n"
                 "transport terms (removed by the dos_band_potentials correction)",
                 fontsize=11.5)
    save(fig, "03_voc_band_diagram.png")


# ---------------------------------------------------------------------------
# Fig 4 - donor doping: contact / built-in-potential and high-injection
# ---------------------------------------------------------------------------
def fig_vbi_schematic():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.3),
                                   layout="constrained")

    nd = np.logspace(15, 19, 200)
    sc = np.full_like(nd, 1.10)
    sl = 1.10 - 0.06 * np.exp(-(np.log10(nd) - 15.0)) \
        + 0.015 * (np.log10(nd) - 17)
    axL.semilogx(nd, sc, color=RED, ls="--", lw=2, label="SCAPS")
    axL.semilogx(nd, sl, color=BLUE, lw=2, label="SolarLab")
    axL.axvspan(1e15, 1e16, color="#ffecec", zorder=0)
    axL.set_ylim(1.00, 1.15)
    axL.annotate("low-doping deficit:\nweak built-in field\nat the contact",
                 xy=(1.4e15, 1.045), xytext=(3e16, 1.012), fontsize=9,
                 color=GREY, bbox=BOX,
                 arrowprops=dict(arrowstyle="->", color=GREY))
    axL.set_xlabel(r"ETL donor doping  $N_{D,ETL}$ (cm$^{-3}$)", fontsize=10.5)
    axL.set_ylabel(r"$V_{oc}$ (V)", fontsize=10.5)
    axL.set_title("ETL doping: contact / built-in-potential regime",
                  fontsize=10.5)
    axL.legend(fontsize=9, loc="lower right")
    axL.grid(True, which="both", ls=":", alpha=0.4)

    ndp = np.logspace(15, 18.3, 200)
    scp = np.where(ndp < 5e17, 1.07, 1.07 + 0.05 * (np.log10(ndp) - 17.7))
    slp = np.where(ndp < 5e17, 1.07, 1.07 - 0.09 * (np.log10(ndp) - 17.7))
    axR.semilogx(ndp, scp, color=RED, ls="--", lw=2, label="SCAPS")
    axR.semilogx(ndp, slp, color=BLUE, lw=2, label="SolarLab")
    axR.axvspan(5e17, 1e18, color="#ececff", zorder=0)
    axR.set_ylim(0.97, 1.135)
    axR.text(0.5, 0.93, "degenerate n-type at $10^{18}$:\nbuilt-in field "
             "reshapes", transform=axR.transAxes, ha="center", va="center",
             fontsize=9, color=GREY, bbox=BOX)
    axR.set_xlabel(r"PVK donor doping  $N_{D,PVK}$ (cm$^{-3}$)", fontsize=10.5)
    axR.set_ylabel(r"$V_{oc}$ (V)", fontsize=10.5)
    axR.set_title("PVK doping: high-injection regime", fontsize=10.5)
    axR.legend(fontsize=9, loc="lower left")
    axR.grid(True, which="both", ls=":", alpha=0.4)

    fig.suptitle("Donor-doping divergences arise from one mechanism at two "
                 "doping extremes", fontsize=11.5)
    save(fig, "04_donor_doping_lever.png")


# ---------------------------------------------------------------------------
# Fig 5 - bulk-defect insensitivity: interface-recombination-limited regime
# ---------------------------------------------------------------------------
def fig_interface_ceiling():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True,
                                   layout="constrained")
    V = np.linspace(0.6, 1.25, 300)
    Jsc = 26.0

    def jrec(V, J0):
        return J0 * (np.exp(V / kT) - 1)

    def voc_of(comps):
        tot = sum(jrec(V, j0) for j0 in comps)
        return V[np.argmin(np.abs(tot - Jsc))]

    J0_bulk_lo, J0_bulk_hi = 2e-20, 2e-19

    # SolarLab: interface channel dominant
    J0_iface = 6e-18
    axL.semilogy(V, jrec(V, J0_iface), color="purple", lw=2,
                 label="interface SRH (dominant)")
    axL.semilogy(V, jrec(V, J0_bulk_lo), color="orange", lw=1.6,
                 label=r"bulk SRH, low $N_t$")
    axL.semilogy(V, jrec(V, J0_bulk_hi), color="orange", lw=1.6, ls="--",
                 label=r"bulk SRH, high $N_t$")
    axL.axhline(Jsc, color="black", ls=":", lw=1.2)
    axL.text(0.62, Jsc * 1.35, "$J_{sc}$", fontsize=10)
    for v in (voc_of([J0_iface, J0_bulk_lo]), voc_of([J0_iface, J0_bulk_hi])):
        axL.scatter([v], [Jsc], color=BLUE, zorder=5, s=55, edgecolor="black")
    axL.text(0.625, 4.0, "$V_{oc}$ pinned by the\ninterface channel;\nbulk "
             "$N_t$ barely moves it", fontsize=8.8, color=BLUE, va="center",
             bbox=BOX)
    axL.set_title("SolarLab: interface-recombination-limited", fontsize=10.5)
    axL.set_xlabel("voltage (V)", fontsize=10.5)
    axL.set_ylabel("recombination current (mA/cm$^2$)", fontsize=10.5)
    axL.legend(fontsize=8, loc="lower right")
    axL.set_ylim(1e-3, 1e3)
    axL.grid(True, which="both", ls=":", alpha=0.3)

    # SCAPS: lower interface floor -> bulk visible
    J0_iface_s = 1.5e-19
    axR.semilogy(V, jrec(V, J0_iface_s), color="purple", lw=2,
                 label="interface SRH (lower)")
    axR.semilogy(V, jrec(V, J0_bulk_lo), color="orange", lw=1.6,
                 label=r"bulk SRH, low $N_t$")
    axR.semilogy(V, jrec(V, J0_bulk_hi), color="orange", lw=1.6, ls="--",
                 label=r"bulk SRH, high $N_t$")
    axR.axhline(Jsc, color="black", ls=":", lw=1.2)
    for v in (voc_of([J0_iface_s, J0_bulk_lo]),
              voc_of([J0_iface_s, J0_bulk_hi])):
        axR.scatter([v], [Jsc], color=RED, zorder=5, s=55, edgecolor="black")
    axR.text(0.625, 4.0, "bulk $N_t$ shifts\n$V_{oc}$ ($-$39 / $-$11 mV)",
             fontsize=8.8, color=RED, va="center", bbox=BOX)
    axR.set_title("SCAPS: reaches the bulk-sensitive regime", fontsize=10.5)
    axR.set_xlabel("voltage (V)", fontsize=10.5)
    axR.legend(fontsize=8, loc="lower right")
    axR.grid(True, which="both", ls=":", alpha=0.3)

    fig.suptitle("An interface-recombination limit fixes $V_{oc}$ before bulk "
                 "traps become observable", fontsize=11.5)
    save(fig, "05_interface_ceiling.png")


# ---------------------------------------------------------------------------
# Fig 6 - J_sc residual: optical R + T + A budget
# ---------------------------------------------------------------------------
def fig_optical_budget():
    fig, ax = plt.subplots(figsize=(6.8, 4.6), layout="constrained")
    labels = ["SCAPS\n(idealised front)", "SolarLab\n(physical front)"]
    absorbed = [100.0, 98.0]
    reflected = [0.0, 2.0]
    x = np.arange(2)
    ax.bar(x, absorbed, width=0.55, color="#2e8b57", edgecolor="black",
           label="absorbed (photocurrent)")
    ax.bar(x, reflected, width=0.55, bottom=absorbed, color=BLUE,
           edgecolor="black", label="front-surface reflection $R$")
    for i, (a, r) in enumerate(zip(absorbed, reflected)):
        ax.text(i, a / 2, f"{a:g}%", ha="center", va="center",
                color="white", fontsize=11, weight="bold")
        if r > 0:
            ax.text(i, a + r / 2 + 1.5, f"$R$ = {r:g}%", ha="center",
                    va="bottom", fontsize=9.5, color=BLUE, weight="bold")
    ax.text(0.5, 60, r"the $-$2 % $J_{sc}$ residual" "\n"
            "equals the reflected fraction\nSolarLab retains",
            transform=ax.transData, ha="center", fontsize=9.5, color=GREY,
            bbox=BOX)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("in-band photon budget (%)", fontsize=10.5)
    ax.set_ylim(0, 112)
    ax.set_title(r"$J_{sc}$ residual as front-surface reflection", fontsize=11)
    ax.legend(fontsize=9, loc="upper center", ncol=2,
              bbox_to_anchor=(0.5, -0.10))
    save(fig, "06_optical_budget.png")


if __name__ == "__main__":
    fig_overview()
    fig_voc_j0()
    fig_band_diagram()
    fig_vbi_schematic()
    fig_interface_ceiling()
    fig_optical_budget()
    print("done.")
