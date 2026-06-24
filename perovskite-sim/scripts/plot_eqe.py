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
# fast electronic transient must settle — a short t_settle + modest grid is
# correct and avoids the high-N Radau stall. 16 wavelengths resolve the
# plateau + the ~780 nm band edge without a multi-minute per-point settle.
wl = np.linspace(320.0, 880.0, 12)
eq = compute_eqe(stack, wavelengths_nm=wl, N_grid=30, t_settle=1e-2)

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

# EQE curve
ax.axhline(1.0, color="0.7", lw=0.8, ls=(0, (4, 3)), zorder=1)
ax.plot(eq.wavelengths_nm, eq.EQE, "o-", color="#1f6fb4", lw=2.0, ms=5,
        mfc="white", mec="#1f6fb4", mew=1.4, zorder=3, label="EQE (SolarLab)")
ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("External quantum efficiency", color="#1f6fb4")
ax.tick_params(axis="y", colors="#1f6fb4")
ax.set_xlim(300, 1000)
ax.set_ylim(0, 1.08)
ax.set_zorder(axr.get_zorder() + 1)
ax.patch.set_visible(False)

# band-edge marker (first big drop)
edge = eq.wavelengths_nm[int(np.argmax(np.abs(np.diff(eq.EQE)))) ]
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
