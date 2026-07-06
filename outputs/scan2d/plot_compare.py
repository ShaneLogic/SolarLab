"""Render the SolarLab-vs-SCAPS 2D-scan comparison figures (compare_ntet.png,
compare_ntcbo.png) from the cached grids.

Each figure stacks the SolarLab 4-panel block over the SCAPS-1D 4-panel block.
For every figure of merit the two blocks share one colour scale (vmin/vmax =
union of both grids) so colours are directly comparable between solvers; the
unresolved SCAPS cells (NaN) stay white.

Inputs : scan2d_results.json (SolarLab, from scan2d.py), scaps_digitized.json
Outputs: compare_ntet.png, compare_ntcbo.png
"""
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
METRICS = ("PCE", "Voc", "FF", "Jsc")
PANEL_LABELS = ("PCE (%)", "$V_{oc}$ (V)", "FF (%)", "$J_{sc}$ (mA cm$^{-2}$)")
LOGN = [9, 10, 11, 12, 13, 14, 15]

plt.rcParams["font.family"] = ["Arial", "DejaVu Sans"]


def load_grids():
    sl = json.loads((OUT / "scan2d_results.json").read_text())
    sc = json.loads((OUT / "scaps_digitized.json").read_text())
    to_arr = lambda block: {k: np.array(block[k], float) for k in METRICS}
    return {
        "ntet": (to_arr(sl["ntet"]), to_arr(sc["A"]), sl["ET"], "Defect energy $E_t$ (eV)"),
        "ntcbo": (to_arr(sl["ntcbo"]), to_arr(sc["B"]), sl["DEC"], "$\\Delta E_C$ (eV)"),
    }


def draw_block(fig, gs, row0, grids, limits, xvals, xlabel):
    for k, (key, lab) in enumerate(zip(METRICS, PANEL_LABELS)):
        ax = fig.add_subplot(gs[row0 + k // 2, k % 2])
        vmin, vmax = limits[key]
        im = ax.pcolormesh(np.arange(len(xvals) + 1), np.arange(len(LOGN) + 1),
                           np.ma.masked_invalid(grids[key]),
                           cmap="viridis", shading="flat", vmin=vmin, vmax=vmax)
        ax.set_xticks(np.arange(len(xvals)) + 0.5)
        ax.set_xticklabels([f"{v:g}" for v in xvals], rotation=45, fontsize=7)
        ax.set_yticks(np.arange(len(LOGN)) + 0.5)
        ax.set_yticklabels(LOGN, fontsize=8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("log$_{10}$ $N_t$ (cm$^{-2}$)")
        ax.set_title(lab, fontsize=11, fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def draw_banner(fig, gs, row, text, fg, bg):
    ax = fig.add_subplot(gs[row, :])
    ax.set_facecolor(bg)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(0.005, 0.5, text, color=fg, fontsize=12, va="center", ha="left",
            transform=ax.transAxes)


def compare_figure(sl, sc, xvals, xlabel, fname):
    limits = {k: (float(min(np.nanmin(sl[k]), np.nanmin(sc[k]))),
                  float(max(np.nanmax(sl[k]), np.nanmax(sc[k])))) for k in METRICS}
    fig = plt.figure(figsize=(11, 15.5))
    gs = fig.add_gridspec(6, 2, height_ratios=[0.14, 1, 1, 0.14, 1, 1],
                          hspace=0.55, wspace=0.28,
                          left=0.07, right=0.97, top=0.985, bottom=0.045)
    draw_banner(fig, gs, 0, "SolarLab  —  scaps_mirror_v2 (transient)",
                fg="#1a237e", bg="#e8eaf6")
    draw_block(fig, gs, 1, sl, limits, xvals, xlabel)
    draw_banner(fig, gs, 3, "SCAPS-1D  —  reference",
                fg="#b71c1c", bg="#fdecea")
    draw_block(fig, gs, 4, sc, limits, xvals, xlabel)
    fig.savefig(OUT / fname, dpi=140)
    plt.close(fig)
    print(f"wrote {OUT / fname}")
    for k in METRICS:
        print(f"  {k}: shared scale [{limits[k][0]:.4f}, {limits[k][1]:.4f}]")


if __name__ == "__main__":
    for scan, (sl, sc, xvals, xlabel) in load_grids().items():
        compare_figure(sl, sc, xvals, xlabel,
                       "compare_ntet.png" if scan == "ntet" else "compare_ntcbo.png")
