"""Plot SolarLab band diagrams (equilibrium + a bias) from a config.

Uses experiments.band_diagram.compute_band_diagram — the SCAPS Energy-Bands-Panel
equivalent. At equilibrium the quasi-Fermi levels coincide into a single flat E_F;
under bias they split by ~qV.

    python scripts/plot_band_diagram.py [config.yaml] [V_bias]
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["mathtext.default"] = "regular"

from perovskite_sim.scaps_compat import load_scaps_yaml
from perovskite_sim.experiments.band_diagram import compute_band_diagram
from perovskite_sim.models.device import electrical_layers

cfg = Path(sys.argv[1]) if len(sys.argv) > 1 else (
    Path(__file__).resolve().parents[1] / "configs" / "scaps_mirror_v2.yaml")
V_bias = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0

stack = load_scaps_yaml(cfg)
eq = compute_band_diagram(stack, 0.0, illuminated=False)
bias = compute_band_diagram(stack, V_bias, illuminated=True)
ifc = np.cumsum([L.thickness for L in electrical_layers(stack)])[:-1] * 1e9

fig, ax = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)


def draw(a, bd, title):
    xn = bd.x * 1e9
    a.plot(xn, bd.E_C, "k-", lw=2, label="$E_C$")
    a.plot(xn, bd.E_V, "k-", lw=2, label="$E_V$")
    a.fill_between(xn, bd.E_C, bd.E_V, color="0.93")
    a.plot(xn, bd.E_Fn, "C0--", lw=2, label="$E_{Fn}$")
    a.plot(xn, bd.E_Fp, "C3:", lw=2, label="$E_{Fp}$")
    for xi in ifc:
        a.axvline(xi, color="0.6", lw=0.7, ls=":")
    a.set_xlabel("Depth (nm)")
    a.set_title(title, fontweight="bold")
    a.legend(loc="center left")
    a.grid(alpha=0.25)


draw(ax[0], eq, "Dark equilibrium  —  E$_F$ flat")
draw(ax[1], bias, f"Illuminated, V={V_bias:.2f} V  —  E$_{{Fn}}$, E$_{{Fp}}$ split")
ax[0].set_ylabel("Energy (eV)")
out = Path("band_diagram.png")
fig.tight_layout()
fig.savefig(out, dpi=135)
print("wrote", out.resolve())
