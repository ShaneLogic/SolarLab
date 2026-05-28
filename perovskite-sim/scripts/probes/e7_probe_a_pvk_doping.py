"""E7 Probe A — PVK donor doping direction check under v2.

Sweeps PVK donor doping N_D ∈ {1e16, 5e16, 1e17, 5e17, 1e18, 5e18, 1e19}
under scaps_mirror_v2.yaml. Reports V_oc(log N_D) trend direction.

Purpose: decide whether Phase Y3 (PVK doping direction fix) is needed.
SCAPS direction under PDF page-2 sweeps: V_oc RISES with N_D.
If SolarLab v2 matches → skip Y3. If reversed → full Y3 scoped.

No code changes, no commits. Output: stdout + CSV under
outputs/scaps_e7_probe_a/.
"""

from __future__ import annotations

import csv
import sys
import math
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from perovskite_sim.experiments.jv_sweep import run_jv_sweep
from perovskite_sim.scaps_compat.loader import load_scaps_yaml
from perovskite_sim.sweeps.device_parameter_sweep import (
    SweepPoint,
    apply_sweep_point,
)

CFG_PATH = REPO_ROOT / "configs" / "scaps_mirror_v2.yaml"
OUT_DIR = REPO_ROOT.parent / "outputs" / "scaps_e7_probe_a"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_D_POINTS = [1e16, 5e16, 1e17, 5e17, 1e18, 5e18, 1e19]
JV_KWARGS = dict(N_grid=30, n_points=20, v_rate=5.0, V_max=1.6)


def main() -> int:
    base = load_scaps_yaml(CFG_PATH)
    rows: list[dict] = []

    print(f"Probe A — PVK donor doping under v2 ({CFG_PATH.name})")
    print(f"{'N_D (cm^-3)':>14}  {'V_oc (V)':>10}  {'J_sc (A/m^2)':>14}  {'bracketed':>10}")
    print("-" * 60)

    for x in N_D_POINTS:
        updates = {
            "absorber_doping_cm3": x,
            "absorber_doping_type": "donor",
        }
        sp = SweepPoint("p", "absorber_doping_cm3", f"{x:.3e}", updates)
        swept = apply_sweep_point(base, sp)
        try:
            res = run_jv_sweep(swept, **JV_KWARGS)
            m = res.metrics_fwd
            row = dict(
                N_D_cm3=x,
                V_oc=m.V_oc,
                J_sc=m.J_sc,
                FF=m.FF,
                PCE=m.PCE,
                voc_bracketed=m.voc_bracketed,
            )
        except Exception as exc:  # noqa: BLE001
            row = dict(
                N_D_cm3=x,
                V_oc=0.0,
                J_sc=0.0,
                FF=0.0,
                PCE=0.0,
                voc_bracketed=False,
                error=f"{type(exc).__name__}: {exc}",
            )
        rows.append(row)
        print(
            f"  {row['N_D_cm3']:.2e}    "
            f"{row['V_oc']:>8.4f}    "
            f"{row['J_sc']:>12.2f}    "
            f"{str(row['voc_bracketed']):>10}"
        )

    # Direction analysis: V_oc derivative wrt log10(N_D), per segment.
    print()
    print("Direction analysis (V_oc derivative across consecutive points):")
    bracketed = [r for r in rows if r["voc_bracketed"]]
    if len(bracketed) < 2:
        print("  insufficient bracketed points")
    else:
        signs_up = signs_down = 0
        for i in range(1, len(bracketed)):
            dlog = math.log10(bracketed[i]["N_D_cm3"]) - math.log10(bracketed[i - 1]["N_D_cm3"])
            dV = bracketed[i]["V_oc"] - bracketed[i - 1]["V_oc"]
            slope = dV / dlog
            sign = "+" if slope > 0 else "-"
            print(
                f"  log10 step {math.log10(bracketed[i - 1]['N_D_cm3']):.1f} → "
                f"{math.log10(bracketed[i]['N_D_cm3']):.1f}: "
                f"ΔV_oc = {dV * 1000:+7.1f} mV  ({sign})"
            )
            if dV > 0:
                signs_up += 1
            else:
                signs_down += 1
        total = signs_up + signs_down
        print()
        print(f"  Segments rising (+): {signs_up}/{total}")
        print(f"  Segments falling (-): {signs_down}/{total}")
        net_dV = bracketed[-1]["V_oc"] - bracketed[0]["V_oc"]
        print(f"  Net ΔV_oc across full sweep: {net_dV * 1000:+7.1f} mV")
        print()
        print("SCAPS expected direction: V_oc RISES with N_D")
        if signs_up >= 0.8 * total:
            verdict = "MATCH — Y3 skipped"
        elif signs_down >= 0.8 * total:
            verdict = "REVERSED — full Y3 scoped"
        else:
            verdict = "MIXED — re-examine sweep range"
        print(f"  → Verdict: {verdict}")

    # Write CSV
    csv_path = OUT_DIR / "pvk_doping_v2_direction.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["N_D_cm3", "V_oc", "J_sc", "FF", "PCE", "voc_bracketed", "error"],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in writer.fieldnames})
    print()
    print(f"CSV: {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
