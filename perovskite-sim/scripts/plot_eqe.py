"""EQE(λ) spectrum + AM1.5G-integrated J_sc cross-check (partner device).

EQE(λ) = J_sc(λ)/(q·Φ(λ)) per wavelength from the TMM optics + drift-diffusion
collection. The integral q·∫EQE·Φ_AM1.5G dλ must reproduce the full-spectrum
J_sc at V=0 — the standard optical↔electrical consistency check. Publication
layout (Arial, 300 dpi).

    python scripts/plot_eqe.py [config.yaml]
"""
import sys
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter, medfilt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "Arial", "font.size": 12,
    "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
    "axes.linewidth": 1.0, "axes.grid": False,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.minor.visible": True, "ytick.minor.visible": True,
    "legend.frameon": True, "legend.framealpha": 0.95, "legend.edgecolor": "0.7",
})

from perovskite_sim.scaps_compat import load_scaps_yaml
from perovskite_sim.data import load_am15g
from perovskite_sim.experiments.eqe import compute_eqe
from perovskite_sim.experiments.steady_state import _grid_for
from perovskite_sim.solver.mol import build_material_arrays
from perovskite_sim.solver.illuminated_ss import solve_illuminated_ss
from perovskite_sim.experiments.jv_sweep import _compute_current_ss

cfg = Path(sys.argv[1]) if len(sys.argv) > 1 else (
    Path(__file__).resolve().parents[1] / "configs" / "scaps_mirror_v2.yaml")
stack = load_scaps_yaml(cfg)

# EQE spectrum. The partner stack has NO mobile ions (D_ion=0), so only the
# fast electronic transient settles (~7-10 s/point on a mat-reuse path) — run
# this script ALONE (concurrent settle jobs contend for CPU and stall it). The
# undulations on a coarse grid come from (1) sparse points joined by straight
# segments and (2) uncorrelated per-point terminal-current settle noise that
# nudges a few points above EQE=1. Fix: a DENSE 80-point grid (~7 nm) so the
# plateau + ~780 nm edge render as a smooth curve, and a longer t_settle=1e-1
# (100 ms, was 50 ms) so the electronic transient fully damps and adjacent
# points stop jumping. 80 points keeps the full solve under the 10-min
# wall-clock ceiling (~6 s/point). The plotted curve is capped at EQE=1 (no
# real EQE exceeds unity; residual settle noise can print ~1.004). Run alone.
def _progress(stage, current, total, message):
    print(f"  [{current:3d}/{total}] {message}", flush=True)

wl = np.linspace(320.0, 880.0, 80)
eq = compute_eqe(stack, wavelengths_nm=wl, N_grid=60, t_settle=1e-1, Phi_incident=1e22,
                 rtol=1e-5, atol=1e-8, progress=_progress)

# full-spectrum device J_sc at V=0 (the cross-check target)
x = _grid_for(stack, 80)
mat = build_material_arrays(x, stack)
y0 = solve_illuminated_ss(x, stack, 0.0, t_settle=1e-2)
Jsc_full = abs(float(_compute_current_ss(x, y0, stack, 0.0, mat=mat))) * 0.1  # mA/cm^2
Jsc_eqe = eq.J_sc_integrated * 0.1
agree = 100.0 * abs(Jsc_eqe - Jsc_full) / Jsc_full

# AM1.5G photon flux for spectral context
wl_fine = np.linspace(300.0, 1000.0, 300)
_, phi_am15g = load_am15g(wl_fine)

fig, ax = plt.subplots(figsize=(7.6, 5.0))

# AM1.5G flux on a faint twin axis (context)
axr = ax.twinx()
axr.fill_between(wl_fine, phi_am15g * 1e-21, color="#f0c419", alpha=0.18, zorder=0)
axr.plot(wl_fine, phi_am15g * 1e-21, color="#d4a017", lw=1.0, alpha=0.5, zorder=0)
axr.set_ylabel(r"AM1.5G photon flux  ($10^{21}$ m$^{-2}$s$^{-1}$nm$^{-1}$)", color="#a07d12")
axr.tick_params(axis="y", colors="#a07d12")
axr.set_ylim(bottom=0)
axr.spines["right"].set_color("#cbb26b")

# EQE curve.
# The plateau carries ~±10% high-frequency (~14 nm period) NUMERICAL noise: each
# point is an independent SS solve and the terminal current is the small
# difference of near-cancelling drift/diffusion fluxes minus the dark baseline,
# so the noise is uncorrelated point-to-point and does NOT damp with a longer
# settle. Real TMM interference would be ~180 nm period (λ²/2nd), so a two-stage
# smooth — median-5 (removes spike outliers) then Savitzky-Golay 11/quadratic
# (~77 nm window) — removes the noise without touching the physics or the sharp
# band edge. The faint raw markers keep the underlying data visible. Cap at unity
# (EQE > 1 is unphysical; residual noise prints ~1.01). The J_sc integral above
# uses the raw values, not the smoothed/capped curve.
ax.axhline(1.0, color="0.7", lw=0.8, ls=(0, (4, 3)), zorder=1)
eqe_smooth = np.minimum(savgol_filter(medfilt(eq.EQE, 5), 11, 2), 1.0)
ax.plot(eq.wavelengths_nm, np.minimum(eq.EQE, 1.0), "o", color="#1f6fb4", ms=2.4,
        alpha=0.22, zorder=2, label="_raw")
ax.plot(eq.wavelengths_nm, eqe_smooth, "-", color="#1f6fb4", lw=2.0,
        solid_capstyle="round", zorder=3, label="EQE (SolarLab)")
ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("External quantum efficiency", color="#1f6fb4")
ax.tick_params(axis="y", colors="#1f6fb4")
ax.set_xlim(300, 1000)
ax.set_ylim(0, 1.08)
ax.set_zorder(axr.get_zorder() + 1)
ax.patch.set_visible(False)

# band-edge marker (first big drop) — detect on the smoothed curve so noise
# spikes on the plateau cannot win the argmax.
edge = eq.wavelengths_nm[int(np.argmax(np.abs(np.diff(eqe_smooth))))]
ax.axvline(edge, color="#c0392b", lw=1.0, ls=":", zorder=2)
ax.text(edge - 8, 0.45, f"band edge\n~{edge:.0f} nm", color="#c0392b",
        fontsize=9, ha="right", va="center")

# J_sc cross-check annotation
txt = (r"$J_{sc}$ cross-check:" + "\n"
       + r"$\int$EQE$\cdot\Phi_{AM1.5G}$ = " + f"{Jsc_eqe:.1f} mA/cm$^2$\n"
       + r"full-spectrum = " + f"{Jsc_full:.1f} mA/cm$^2$\n"
       + f"agreement {agree:.1f}%")
ax.text(0.035, 0.045, txt, transform=ax.transAxes, fontsize=9.5, va="bottom", ha="left",
        bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="0.7", alpha=0.95))

ax.legend(loc="upper right", fontsize=10)
ax.set_title("SolarLab EQE spectrum — scaps_mirror_v2 (SCAPS-1D partner stack)",
             fontsize=12, pad=10)
fig.tight_layout()
out = Path("eqe_spectrum.png")
fig.savefig(out, dpi=300, bbox_inches="tight")
print(f"wrote {out.resolve()}  Jsc_EQE={Jsc_eqe:.2f} Jsc_full={Jsc_full:.2f} agree={agree:.1f}%")
