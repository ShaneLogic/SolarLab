"""E7 probe — A* (Richardson-Dushman) tune to mimic SCAPS v_th-based TE.

SCAPS manual (section 3.8) documents thermionic emission at interfaces
using v_th directly. SolarLab uses Richardson-Dushman as a cap on the
SG flux with default A* = 1.2017e6 A/(m²·K²) (free-electron value).
SCAPS' effective TE coefficient implied by `v_th = 1e7 cm/s = 1e5 m/s`
gives an effective A* ≈ 1.6e10 / 9e4 ≈ 1.8e5 A/(m²·K²) — about 7× lower
than SolarLab's default.

Step 1: vary A_star_n / A_star_p on PVK + ETL layers, measure base J-V
V_oc shift. Variants:
  - baseline A* = 1.2017e6  (current default)
  - 10x lower (A* = 1.2e5)
  - 100x lower (A* = 1.2e4)
  - 1000x lower (A* = 1.2e3)

Look for V_oc rise toward SCAPS 1.168 V baseline.

Step 2 (if V_oc rises): run CBO sweep on best A* to confirm CBO
direction preserved.
"""

from __future__ import annotations

import csv
import dataclasses
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from perovskite_sim.experiments.jv_sweep import run_jv_sweep
from perovskite_sim.scaps_compat.loader import load_scaps_yaml

BASE_CFG = REPO_ROOT / "configs" / "scaps_mirror_v2.yaml"
OUT_DIR = REPO_ROOT.parent / "outputs" / "scaps_e7_a_star"
OUT_DIR.mkdir(parents=True, exist_ok=True)

JV_KWARGS = dict(N_grid=30, n_points=20, v_rate=5.0, V_max=1.6)

# A_star variants in A/(m²·K²). Default SolarLab = 1.2017e6.
A_STAR_VARIANTS = {
    "baseline_1.2e6": 1.2017e6,
    "1.2e5_10x_lower": 1.2017e5,
    "1.2e4_100x_lower": 1.2017e4,
    "1.2e3_1000x_lower": 1.2017e3,
}


def _override_a_star(stack, a_star: float):
    """Return a new DeviceStack with A_star_n / A_star_p overridden on absorber + ETL layers.

    The SCAPS YAML loader (scaps_compat) does not parse A_star fields, so we
    override at runtime via dataclasses.replace.
    """
    new_layers = []
    for layer in stack.layers:
        if layer.role in ("absorber", "ETL"):
            new_params = dataclasses.replace(
                layer.params, A_star_n=a_star, A_star_p=a_star
            )
            new_layer = dataclasses.replace(layer, params=new_params)
            new_layers.append(new_layer)
        else:
            new_layers.append(layer)
    return dataclasses.replace(stack, layers=tuple(new_layers))


def main() -> int:
    rows = []

    print("Probe — A* hand-tune. Base J-V V_oc per variant.")
    print(f"  SCAPS reference: V_oc = 1.168 V (default Nd_ETL=1e18)")
    print()

    base_stack = load_scaps_yaml(BASE_CFG)

    for label, a_star in A_STAR_VARIANTS.items():
        stack = _override_a_star(base_stack, a_star)
        try:
            res = run_jv_sweep(stack, **JV_KWARGS)
            m = res.metrics_fwd
            row = dict(
                variant=label,
                a_star=a_star,
                V_oc=m.V_oc,
                J_sc=m.J_sc,
                FF=m.FF,
                PCE=m.PCE,
                voc_bracketed=m.voc_bracketed,
            )
        except Exception as exc:  # noqa: BLE001
            row = dict(
                variant=label,
                a_star=a_star,
                V_oc=0.0,
                J_sc=0.0,
                FF=0.0,
                PCE=0.0,
                voc_bracketed=False,
                error=f"{type(exc).__name__}: {exc}",
            )
        rows.append(row)
        print(f"  {label:>25}  A*={a_star:>9.2e}  "
              f"V_oc={row['V_oc']:>7.4f}  "
              f"J_sc={row['J_sc']:>8.1f}  "
              f"bracketed={row['voc_bracketed']}")

    print()
    print(f"Baseline V_oc: {rows[0]['V_oc']:.4f} V")
    for r in rows[1:]:
        shift = r["V_oc"] - rows[0]["V_oc"]
        print(f"  {r['variant']:>25}  ΔV_oc = {shift * 1000:+7.1f} mV")

    print()
    best = max(rows, key=lambda r: r["V_oc"] if r["voc_bracketed"] else 0.0)
    print(f"Highest V_oc: {best['variant']} → {best['V_oc']:.4f} V "
          f"(SCAPS target 1.168 V, gap {(1.168 - best['V_oc']) * 1000:.1f} mV)")
    if best["V_oc"] > rows[0]["V_oc"] + 0.020:
        print("→ A* tuning lifts V_oc. Worth proceeding to CBO sweep test.")
    else:
        print("→ A* tuning has < 20 mV effect. TE formula isn't the main lever.")

    csv_path = OUT_DIR / "a_star_base_voc.csv"
    with open(csv_path, "w", newline="") as f:
        fields = ["variant", "a_star", "V_oc", "J_sc", "FF", "PCE", "voc_bracketed", "error"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fields})
    print(f"\nCSV: {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
