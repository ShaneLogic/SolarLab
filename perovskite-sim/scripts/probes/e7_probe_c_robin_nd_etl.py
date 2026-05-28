"""E7 Probe C — Robin contact BC dry-run on Nd_ETL sweep.

Runs Nd_ETL sheet (11 points, 1e10 → 1e20 cm⁻³) under three configs:
  1. scaps_mirror_v2.yaml          (Dirichlet baseline, current E6.4)
  2. scaps_mirror_v2_robin_moderate.yaml (S_majority=1e3, S_minority=1e1 m/s)

Compares:
  - V_oc upper bound: any point above 1.53 V (E_g/q) is unphysical
  - V_oc bracketing success across the sweep
  - V_oc range across the working (bracketed AND physical) regime
  - SCAPS reference range from scaps_reference.json

Per spec Probe C: partner PDF specifies no contact workfunction, so
this is a sensitivity probe rather than a SCAPS-exact match. Decides
Y2 branch (C1 clean Robin / C2 two-preset split / C3 re-scope).
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from perovskite_sim.experiments.jv_sweep import run_jv_sweep
from perovskite_sim.scaps_compat.loader import load_scaps_yaml
from perovskite_sim.sweeps.device_parameter_sweep import (
    SweepPoint,
    apply_sweep_point,
)

REF_PATH = REPO_ROOT / "tests" / "integration" / "scaps_reference.json"
OUT_DIR = REPO_ROOT.parent / "outputs" / "scaps_e7_probe_c"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONFIGS = {
    "v2_dirichlet": REPO_ROOT / "configs" / "scaps_mirror_v2.yaml",
    "v2_robin_moderate": REPO_ROOT / "configs" / "scaps_mirror_v2_robin_moderate.yaml",
    "v2_robin_strong": REPO_ROOT / "configs" / "scaps_mirror_v2_robin_strong.yaml",
}

JV_KWARGS = dict(N_grid=30, n_points=20, v_rate=5.0, V_max=1.6)
E_G_PVK = 1.53  # eV — unphysical bound on V_oc


def main() -> int:
    ref = json.loads(REF_PATH.read_text())
    nd_sweep = ref["sweeps"]["Nd_ETL"]
    ref_points = nd_sweep["points"]
    print(f"Probe C — Robin Nd_ETL sweep comparison")
    print(f"  SCAPS reference: {len(ref_points)} points")
    print()

    # Collect SCAPS reference V_ocs
    scaps_voc = [pt["Voc_V"] for pt in ref_points]
    scaps_voc_range = max(scaps_voc) - min(scaps_voc)
    print(f"  SCAPS V_oc range (full sweep): {scaps_voc_range * 1000:.1f} mV")
    print(f"  SCAPS V_oc min/max: {min(scaps_voc):.4f} / {max(scaps_voc):.4f} V")
    print()

    results: dict[str, list[dict]] = {}

    for label, cfg_path in CONFIGS.items():
        print(f"Running config: {label}  ({cfg_path.name})")
        base = load_scaps_yaml(cfg_path)
        rows: list[dict] = []
        for pt in ref_points:
            x = float(pt["x"])
            updates = {"etl_doping_cm3": x}
            sp = SweepPoint("p", "etl_doping_cm3", f"{x:.3e}", updates)
            swept = apply_sweep_point(base, sp)
            try:
                res = run_jv_sweep(swept, **JV_KWARGS)
                m = res.metrics_fwd
                physical = m.voc_bracketed and m.V_oc <= E_G_PVK
                row = dict(
                    N_d=x,
                    V_oc=m.V_oc,
                    J_sc=m.J_sc,
                    voc_bracketed=m.voc_bracketed,
                    physical=physical,
                    V_oc_scaps=pt["Voc_V"],
                )
            except Exception as exc:  # noqa: BLE001
                row = dict(
                    N_d=x,
                    V_oc=0.0,
                    J_sc=0.0,
                    voc_bracketed=False,
                    physical=False,
                    V_oc_scaps=pt["Voc_V"],
                    error=f"{type(exc).__name__}: {exc}",
                )
            rows.append(row)
            tag = "ok" if row["physical"] else (
                "UNPHYSICAL" if row["V_oc"] > E_G_PVK else "no-bracket"
            )
            print(
                f"  N_d={row['N_d']:.2e}  "
                f"V_oc_sl={row['V_oc']:>7.4f}  "
                f"V_oc_scaps={row['V_oc_scaps']:>7.4f}  "
                f"({tag})"
            )
        results[label] = rows
        print()

    # Compare summaries
    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    for label in CONFIGS:
        rows = results[label]
        physical = [r for r in rows if r["physical"]]
        n_brk = sum(1 for r in rows if r["voc_bracketed"])
        n_unphys = sum(1 for r in rows if r["voc_bracketed"] and r["V_oc"] > E_G_PVK)
        n_phys = len(physical)
        if physical:
            sl_voc = [r["V_oc"] for r in physical]
            sl_range = max(sl_voc) - min(sl_voc)
            scaps_at_same_pts = [r["V_oc_scaps"] for r in physical]
            scaps_range = max(scaps_at_same_pts) - min(scaps_at_same_pts)
            closure = sl_range / scaps_range if scaps_range > 0 else float("inf")
        else:
            sl_range = scaps_range = closure = 0.0
        print(f"\n{label}:")
        print(f"  bracketed:      {n_brk}/{len(rows)}")
        print(f"  unphysical:     {n_unphys} (V_oc > {E_G_PVK} V)")
        print(f"  working subset: {n_phys}/{len(rows)}")
        print(f"  V_oc range (SolarLab, working): {sl_range * 1000:.1f} mV")
        print(f"  V_oc range (SCAPS, same pts):   {scaps_range * 1000:.1f} mV")
        if scaps_range > 0:
            print(f"  Closure: {closure * 100:.1f} %")

    # Write CSV
    for label, rows in results.items():
        csv_path = OUT_DIR / f"nd_etl_{label}.csv"
        with open(csv_path, "w", newline="") as f:
            fields = ["N_d", "V_oc", "J_sc", "voc_bracketed", "physical", "V_oc_scaps", "error"]
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for r in rows:
                writer.writerow({k: r.get(k, "") for k in fields})
        print(f"\n  CSV: {csv_path}")

    # Verdict
    print()
    print("Verdict logic:")
    robin = results["v2_robin_moderate"]
    base = results["v2_dirichlet"]
    n_unphys_robin = sum(1 for r in robin if r["voc_bracketed"] and r["V_oc"] > E_G_PVK)
    n_unphys_base = sum(1 for r in base if r["voc_bracketed"] and r["V_oc"] > E_G_PVK)
    print(f"  Dirichlet unphysical branches: {n_unphys_base}")
    print(f"  Robin unphysical branches:     {n_unphys_robin}")
    if n_unphys_robin == 0 and n_unphys_base > 0:
        print("  → Robin kills unphysical branch. C1/C2 viable. Run full 4-sweep next.")
    elif n_unphys_robin > 0:
        print("  → Robin alone insufficient. C3 — re-scope or partner SCAPS Φ_b.")
    else:
        print("  → No unphysical branches in either config. Re-examine sweep range.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
