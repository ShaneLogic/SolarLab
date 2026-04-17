"""
Phase 2 Characterisation Experiments — Quantitative Benchmark
==============================================================
Exercises the four Phase 2 wrappers end-to-end on shipped configs and
checks each result against published or analytical expectations.

Experiments covered
-------------------
  1. Dark J-V       — extract ideality n and saturation current J_0
                      from a forward dark sweep. Expect 1 <= n <= 2.5 and
                      J_0 positive.
  2. Suns-V_oc      — sweep illumination intensity, fit n_eff from the
                      V_oc vs ln(suns) slope, and build the pseudo J-V
                      curve. Expect 0 < pseudo_FF < 1 and V_oc rising
                      with suns.
  3. EQE / IPCE     — monochromatic short-circuit sweep, AM1.5G-integrated
                      J_sc. Expect 0 <= EQE <= 1 (small TMM slack) and
                      visible-peak EQE > red-tail EQE.
  4. Mott-Schottky  — dark C-V sweep at 100 kHz, linear fit of 1/C^2 vs
                      V. Expect finite V_bi_fit, N_eff_fit, fit window
                      bounded by the swept range.

Usage
-----
  python notebooks/07_phase2_characterisation_benchmark.py

No network access or external data is required. Runs against configs
shipped in perovskite-sim/configs/.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from perovskite_sim.constants import K_B, Q  # noqa: E402
from perovskite_sim.experiments.dark_jv import run_dark_jv  # noqa: E402
from perovskite_sim.experiments.eqe import compute_eqe  # noqa: E402
from perovskite_sim.experiments.mott_schottky import run_mott_schottky  # noqa: E402
from perovskite_sim.experiments.suns_voc import run_suns_voc  # noqa: E402
from perovskite_sim.models.config_loader import load_device_from_yaml  # noqa: E402


CONFIGS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "configs")
)

PASS = 0
FAIL = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    """Tag a quantitative assertion as PASS/FAIL and bump the global tally."""
    global PASS, FAIL
    tag = "[PASS]" if ok else "[FAIL]"
    if ok:
        PASS += 1
    else:
        FAIL += 1
    extra = f"  ({detail})" if detail else ""
    print(f"  {tag} {name}{extra}")


def banner(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


# ───────────────────────────────────────────────────────────────────────────
# 1. Dark J-V on nip_MAPbI3
# ───────────────────────────────────────────────────────────────────────────
banner("[1] DARK J-V — ideality factor + saturation current")

stack_dark = load_device_from_yaml(os.path.join(CONFIGS_DIR, "nip_MAPbI3.yaml"))
t0 = time.time()
dark = run_dark_jv(stack_dark, V_max=1.1, n_points=40, N_grid=40)
dt = time.time() - t0

V_T = K_B * 300.0 / Q
print(f"Simulation time: {dt:.1f} s")
print(f"Fit window      : [{dark.V_fit_lo:.3f}, {dark.V_fit_hi:.3f}] V")
print(f"Ideality n      : {dark.n_ideality:.3f}  (V_T = {V_T:.4f} V)")
print(f"Saturation J_0  : {dark.J_0:.3e} A/m^2")
print(f"|J|@V=0         : {abs(dark.J[np.argmin(np.abs(dark.V))]):.3e} A/m^2")

check("ideality n in [1.0, 2.5]", 1.0 <= dark.n_ideality <= 2.5,
      detail=f"n = {dark.n_ideality:.3f}")
check("J_0 positive and finite",
      dark.J_0 > 0 and np.isfinite(dark.J_0),
      detail=f"J_0 = {dark.J_0:.2e}")
check("|J(V=0)| < 1 A/m^2 (negligible dark photocurrent)",
      abs(dark.J[np.argmin(np.abs(dark.V))]) < 1.0)
check("forward injection monotone (J(V_max) > J(V=0))",
      dark.J[-1] > dark.J[0],
      detail=f"J_max={dark.J[-1]:.2e}, J_0bias={dark.J[0]:.2e}")

# ───────────────────────────────────────────────────────────────────────────
# 2. Suns-V_oc on nip_MAPbI3
# ───────────────────────────────────────────────────────────────────────────
banner("[2] SUNS-V_oc — effective ideality + pseudo FF")

stack_sv = load_device_from_yaml(os.path.join(CONFIGS_DIR, "nip_MAPbI3.yaml"))
suns_levels = (0.01, 0.1, 1.0, 5.0, 10.0)
t0 = time.time()
sv = run_suns_voc(stack_sv, suns_levels=suns_levels, N_grid=40, t_settle=1e-3)
dt = time.time() - t0

# Derived ideality from the V_oc vs ln(suns) slope.
ln_suns = np.log(np.asarray(sv.suns))
slope, _intercept = np.polyfit(ln_suns, sv.V_oc, 1)
n_eff_derived = slope / V_T

print(f"Simulation time   : {dt:.1f} s")
print("Suns levels       : " + ", ".join(f"{s:g}" for s in sv.suns))
print("V_oc (V)          : " + ", ".join(f"{v:.3f}" for v in sv.V_oc))
print(f"n_eff (derived)   : {n_eff_derived:.3f}")
print(f"pseudo FF         : {sv.pseudo_FF * 100:.1f} %")
print(f"pseudo J-V points : {len(sv.J_pseudo_V)}")

check("n_eff in [1.0, 2.5]", 1.0 <= n_eff_derived <= 2.5,
      detail=f"n_eff = {n_eff_derived:.3f}")
check("pseudo FF in (0, 1)", 0.0 < sv.pseudo_FF < 1.0,
      detail=f"pFF = {sv.pseudo_FF:.3f}")
check("V_oc monotone increasing with suns",
      bool(np.all(np.diff(sv.V_oc) > 0)))
j_per_sun = np.asarray(sv.J_sc) / np.asarray(sv.suns)
ref_j = j_per_sun[np.argmin(np.abs(np.asarray(sv.suns) - 1.0))]
check("J_sc/suns within 2x of 1-sun baseline across sweep",
      bool(np.all(j_per_sun > ref_j / 2) and np.all(j_per_sun < ref_j * 2)))

# ───────────────────────────────────────────────────────────────────────────
# 3. EQE / IPCE on nip_MAPbI3_tmm
# ───────────────────────────────────────────────────────────────────────────
banner("[3] EQE / IPCE — monochromatic QE + AM1.5G integration")

stack_eqe = load_device_from_yaml(os.path.join(CONFIGS_DIR, "nip_MAPbI3_tmm.yaml"))
wavelengths_nm = np.linspace(350.0, 850.0, 11)
t0 = time.time()
eqe = compute_eqe(stack_eqe, wavelengths_nm=wavelengths_nm, N_grid=40,
                  t_settle=1e-3)
dt = time.time() - t0

Jsc_mA = eqe.J_sc_integrated / 10.0
peak_idx = int(np.argmax(eqe.EQE))
red_idx = int(np.argmin(np.abs(eqe.wavelengths_nm - 800.0)))

print(f"Simulation time     : {dt:.1f} s")
print(f"Wavelength count    : {len(eqe.wavelengths_nm)}  ({eqe.wavelengths_nm[0]:.0f}–{eqe.wavelengths_nm[-1]:.0f} nm)")
print(f"Peak EQE            : {eqe.EQE[peak_idx] * 100:.1f} % at {eqe.wavelengths_nm[peak_idx]:.0f} nm")
print(f"EQE @ 800 nm        : {eqe.EQE[red_idx] * 100:.1f} %")
print(f"J_sc (AM1.5G)       : {Jsc_mA:.2f} mA/cm^2")

check("EQE in [0, 1.01]",
      float(eqe.EQE.min()) >= 0.0 and float(eqe.EQE.max()) <= 1.01,
      detail=f"range = [{eqe.EQE.min():.3f}, {eqe.EQE.max():.3f}]")
check("peak EQE > EQE(~800 nm) + 0.1 (visible-band absorption peak)",
      float(eqe.EQE[peak_idx] - eqe.EQE[red_idx]) > 0.1)
check("integrated J_sc (AM1.5G) in [5, 30] mA/cm^2 for MAPbI3",
      5.0 < Jsc_mA < 30.0, detail=f"Jsc = {Jsc_mA:.2f} mA/cm^2")
check("Phi_incident positive and finite",
      eqe.Phi_incident > 0 and np.isfinite(eqe.Phi_incident))

# ───────────────────────────────────────────────────────────────────────────
# 4. Mott–Schottky on cSi_homojunction
# ───────────────────────────────────────────────────────────────────────────
banner("[4] MOTT-SCHOTTKY — 1/C^2 fit gives V_bi and N_eff")

stack_ms = load_device_from_yaml(
    os.path.join(CONFIGS_DIR, "cSi_homojunction.yaml")
)
V_range = np.linspace(-0.3, 0.4, 8)
t0 = time.time()
ms = run_mott_schottky(stack_ms, V_range=V_range, frequency=1.0e5,
                       delta_V=0.01, N_grid=40)
dt = time.time() - t0

print(f"Simulation time     : {dt:.1f} s")
print(f"Bias range          : [{ms.V[0]:.2f}, {ms.V[-1]:.2f}] V at f = {ms.frequency:.1e} Hz")
print(f"Fit window          : [{ms.V_fit_lo:.3f}, {ms.V_fit_hi:.3f}] V")
print(f"V_bi (fit)          : {ms.V_bi_fit:.3f} V")
print(f"N_eff (fit)         : {ms.N_eff_fit:.3e} m^-3")
print(f"eps_r used          : {ms.eps_r_used:.2f}")

check("V_bi_fit finite", bool(np.isfinite(ms.V_bi_fit)))
check("N_eff_fit finite", bool(np.isfinite(ms.N_eff_fit)))
check("fit window inside swept range",
      ms.V[0] - 1e-9 <= ms.V_fit_lo <= ms.V_fit_hi <= ms.V[-1] + 1e-9)
check("1/C^2 matches 1/C^2 from reported C array (within 1e-12)",
      float(np.max(np.abs(ms.one_over_C2 - 1.0 / (ms.C ** 2)))) < 1e-12)
check("frequency round-trips from input",
      abs(ms.frequency - 1.0e5) / 1.0e5 < 1e-12)

# ───────────────────────────────────────────────────────────────────────────
# Summary
# ───────────────────────────────────────────────────────────────────────────
banner("VALIDATION SUMMARY")
print(f"  {PASS} passed, {FAIL} failed")
print()
if FAIL == 0:
    print("ALL CHECKS PASSED — Phase 2 wrappers produce physically sensible")
    print("results on shipped configs.")
    sys.exit(0)
else:
    print("SOME CHECKS FAILED — inspect output above.")
    sys.exit(1)
