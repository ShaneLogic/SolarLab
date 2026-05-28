"""E7 Probe B — multi-defect SRH collapse audit.

Compares true two-defect R_SRH(n, p) (summing per-defect rates with their
own (τ_i, n1_i, p1_i)) against loader-collapsed R_SRH (single effective
(τ_eff, n1_eff, p1_eff) from `_combine_bulk_defects`).

Per scaps_mirror_v2.yaml, PVK has two bulk defects:
  - Perovskite-CB: E_t = 0.1 eV below CB
  - Perovskite-VB: E_t = 0.1 eV above VB
Both with identical σ_n = σ_p = 1e-15 cm², N_t = 1e12 cm⁻³, v_th = 1e7 cm/s.

Output: ratio R_true / R_collapsed at 5 (n, p) sample points spanning
the injection range. Decides Y1 branch (B1 SRV tune vs B2 multi-defect
solver hook).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import yaml

from perovskite_sim.scaps_compat.loader import load_scaps_yaml

CFG_PATH = REPO_ROOT / "configs" / "scaps_mirror_v2.yaml"
OUT_DIR = REPO_ROOT.parent / "outputs" / "scaps_e7_probe_b"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# Constants
Q = 1.602176634e-19  # C
KB = 1.380649e-23  # J/K
T = 300.0  # K
V_T = KB * T / Q  # ≈ 0.025852 V


def _n1_p1_below_cb(N_C: float, N_V: float, E_g: float, E_t_below_cb: float) -> tuple[float, float]:
    """SRH n1 / p1 for a defect E_t below the conduction band.

    n1 = N_C · exp(-E_t/V_T)
    p1 = N_V · exp(-(E_g - E_t)/V_T)
    """
    n1 = N_C * math.exp(-E_t_below_cb / V_T)
    p1 = N_V * math.exp(-(E_g - E_t_below_cb) / V_T)
    return n1, p1


def _srh_rate(n: float, p: float, ni_sq: float, tau_n: float, tau_p: float, n1: float, p1: float) -> float:
    """Single-defect SRH recombination rate."""
    num = n * p - ni_sq
    den = tau_p * (n + n1) + tau_n * (p + p1)
    return num / den


def main() -> int:
    # Load v2 stack
    stack = load_scaps_yaml(CFG_PATH)
    pvk = next(layer for layer in stack.layers if layer.role == "absorber")
    pvk_params = pvk.params

    # MaterialParams doesn't store N_C / N_V (loader uses them only to derive
    # ni then discards). Read directly from raw YAML for N_C / N_V.
    raw = yaml.safe_load(CFG_PATH.read_text())
    pvk_raw = next(l for l in raw["layers"] if l.get("role") == "absorber")
    # YAML stores N_C_cm3, N_V_cm3 in cm⁻³ → m⁻³
    N_C = float(pvk_raw["N_C_cm3"]) * 1e6
    N_V = float(pvk_raw["N_V_cm3"]) * 1e6
    E_g = pvk_params.Eg  # eV (loader → SolarLab field)
    ni_sq = pvk_params.ni ** 2  # m⁻⁶ (loader-derived)

    # Loader-collapsed effective (τ, n1, p1)
    tau_n_eff = pvk_params.tau_n
    tau_p_eff = pvk_params.tau_p
    n1_eff = pvk_params.n1
    p1_eff = pvk_params.p1

    # Raw per-defect parameters from YAML
    # PVK-CB: σ=1e-15 cm² = 1e-19 m², N_t=1e12 cm⁻³ = 1e18 m⁻³, v_th=1e7 cm/s = 1e5 m/s
    sigma_si = 1.0e-19
    v_th_si = 1.0e5
    N_t_si = 1.0e18

    tau_cb = 1.0 / (sigma_si * v_th_si * N_t_si)
    tau_vb = tau_cb

    n1_cb, p1_cb = _n1_p1_below_cb(N_C, N_V, E_g, E_t_below_cb=0.1)
    n1_vb, p1_vb = _n1_p1_below_cb(N_C, N_V, E_g, E_t_below_cb=E_g - 0.1)

    print(f"Probe B — SRH collapse audit on {CFG_PATH.name}")
    print(f"  T = {T} K, V_T = {V_T*1000:.3f} mV")
    print(f"  N_C = {N_C:.3e} m⁻³, N_V = {N_V:.3e} m⁻³, E_g = {E_g:.3f} eV")
    print(f"  ni² = {ni_sq:.3e} m⁻⁶")
    print()
    print("Per-defect (SI):")
    print(f"  PVK-CB: τ_n=τ_p={tau_cb:.3e} s, n1={n1_cb:.3e} m⁻³, p1={p1_cb:.3e} m⁻³")
    print(f"  PVK-VB: τ_n=τ_p={tau_vb:.3e} s, n1={n1_vb:.3e} m⁻³, p1={p1_vb:.3e} m⁻³")
    print()
    print("Loader-collapsed effective (from MaterialParams):")
    print(f"  τ_n_eff={tau_n_eff:.3e} s, τ_p_eff={tau_p_eff:.3e} s")
    print(f"  n1_eff={n1_eff:.3e} m⁻³, p1_eff={p1_eff:.3e} m⁻³")
    print()

    # Sample (n, p) injection points
    samples = [
        ("dark-eq (low)", 1e18, 1e18),
        ("low injection", 1e20, 1e20),
        ("mid injection", 1e22, 1e22),
        ("high injection (V_oc)", 1e23, 1e23),
        ("asymmetric n-rich", 1e23, 1e18),
        ("asymmetric p-rich", 1e18, 1e23),
    ]

    print(f"{'sample':>30}  {'n (m⁻³)':>10}  {'p (m⁻³)':>10}  "
          f"{'R_true':>12}  {'R_collapsed':>12}  {'ratio':>8}")
    print("-" * 100)

    max_dev = 0.0
    for label, n, p in samples:
        r_cb = _srh_rate(n, p, ni_sq, tau_cb, tau_cb, n1_cb, p1_cb)
        r_vb = _srh_rate(n, p, ni_sq, tau_vb, tau_vb, n1_vb, p1_vb)
        r_true = r_cb + r_vb
        r_collapsed = _srh_rate(n, p, ni_sq, tau_n_eff, tau_p_eff, n1_eff, p1_eff)
        ratio = r_true / r_collapsed if abs(r_collapsed) > 0 else float("nan")
        dev = abs(ratio - 1.0)
        max_dev = max(max_dev, dev)
        print(f"{label:>30}  {n:>10.2e}  {p:>10.2e}  "
              f"{r_true:>12.3e}  {r_collapsed:>12.3e}  {ratio:>8.4f}")

    print()
    print(f"Max deviation from unity ratio: {max_dev*100:.2f}%")
    print()
    if max_dev <= 0.1:
        verdict = "B1 — collapse fine. Y1 = YAML-only PVK/ETL SRV tune."
    elif max_dev >= 1.0:
        verdict = "B2 — collapse wrong (≥ 2×). Y1 = true multi-defect solver hook."
    else:
        verdict = "B3 — mixed. Y1 = both branches."
    print(f"  → Verdict: {verdict}")

    # Save summary CSV
    csv_path = OUT_DIR / "srh_collapse_ratio.csv"
    with open(csv_path, "w") as f:
        f.write("sample,n_m3,p_m3,R_true_m3_s,R_collapsed_m3_s,ratio\n")
        for label, n, p in samples:
            r_cb = _srh_rate(n, p, ni_sq, tau_cb, tau_cb, n1_cb, p1_cb)
            r_vb = _srh_rate(n, p, ni_sq, tau_vb, tau_vb, n1_vb, p1_vb)
            r_true = r_cb + r_vb
            r_collapsed = _srh_rate(n, p, ni_sq, tau_n_eff, tau_p_eff, n1_eff, p1_eff)
            ratio = r_true / r_collapsed if abs(r_collapsed) > 0 else float("nan")
            f.write(f"{label},{n:.3e},{p:.3e},{r_true:.6e},{r_collapsed:.6e},{ratio:.6f}\n")
    print(f"\nCSV: {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
