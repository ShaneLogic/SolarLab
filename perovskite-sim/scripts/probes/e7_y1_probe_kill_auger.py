"""E7 Y1 confirmation — kill Auger / radiative, re-run Nt_C_PVK.

Tests the hypothesis "Auger+radiative dominate bulk SRH at V_oc, so
Nt_C_PVK sweep can't move V_oc." Three variants of scaps_mirror_v2.yaml:

  1. baseline      — Auger ON, Radiative ON (control)
  2. auger_off     — C_n=C_p=0, Radiative ON
  3. all_recomb_off — C_n=C_p=0, B_rad=0 (SRH only)

If variant 2 opens V_oc range significantly → Auger is the ceiling.
If only variant 3 opens it → radiative is the ceiling (not Auger).
If neither opens it → some other limit (transport / contact / numerical).

No commits. Output: stdout + CSV.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from perovskite_sim.experiments.jv_sweep import run_jv_sweep
from perovskite_sim.scaps_compat.loader import load_scaps_yaml
from perovskite_sim.sweeps.device_parameter_sweep import (
    SweepPoint,
    apply_sweep_point,
)

BASE_CFG = REPO_ROOT / "configs" / "scaps_mirror_v2.yaml"
REF_PATH = REPO_ROOT / "tests" / "integration" / "scaps_reference.json"
OUT_DIR = REPO_ROOT.parent / "outputs" / "scaps_e7_y1_kill_auger"
OUT_DIR.mkdir(parents=True, exist_ok=True)

JV_KWARGS = dict(N_grid=30, n_points=20, v_rate=5.0, V_max=1.6)

VARIANTS = {
    "baseline":        dict(C=2.3e-29, B=1.0e-12),  # PDF values
    "auger_off":       dict(C=0.0,     B=1.0e-12),
    "all_recomb_off":  dict(C=0.0,     B=0.0),
}


def _write_variant(cfg_dict: dict, c_aug: float, b_rad: float, out_path: Path) -> Path:
    new = json.loads(json.dumps(cfg_dict))
    for layer in new["layers"]:
        if layer.get("role") == "absorber":
            layer["C_n_cm6_s"] = c_aug
            layer["C_p_cm6_s"] = c_aug
            layer["B_rad_cm3_s"] = b_rad
    out_path.write_text(yaml.safe_dump(new, sort_keys=False))
    return out_path


def main() -> int:
    cfg_dict = yaml.safe_load(BASE_CFG.read_text())
    ref = json.loads(REF_PATH.read_text())
    nt_sheet = ref["sweeps"]["Nt_C_PVK"]
    ref_points = nt_sheet["points"]
    scaps_voc = [pt["Voc_V"] for pt in ref_points]
    scaps_range = max(scaps_voc) - min(scaps_voc)

    print(f"Kill-Auger probe — Nt_C_PVK sweep, three recombination variants")
    print(f"  SCAPS reference: {len(ref_points)} points, V_oc range {scaps_range * 1000:.1f} mV")
    print()

    results: dict[str, list[dict]] = {}

    for variant, params in VARIANTS.items():
        print(f"Variant: {variant}  (C_aug={params['C']:.2e}, B_rad={params['B']:.2e})")
        tmp_path = OUT_DIR / f"variant_{variant}.yaml"
        _write_variant(cfg_dict, params["C"], params["B"], tmp_path)
        base = load_scaps_yaml(tmp_path)
        rows: list[dict] = []
        for pt in ref_points:
            x = float(pt["x"])
            updates = {"absorber_defect_density_cm3": x}
            sp = SweepPoint("p", "absorber_defect_density_cm3", f"{x:.3e}", updates)
            swept = apply_sweep_point(base, sp)
            try:
                res = run_jv_sweep(swept, **JV_KWARGS)
                m = res.metrics_fwd
                row = dict(
                    N_t=x,
                    V_oc=m.V_oc,
                    voc_bracketed=m.voc_bracketed,
                    V_oc_scaps=pt["Voc_V"],
                )
            except Exception as exc:  # noqa: BLE001
                row = dict(N_t=x, V_oc=0.0, voc_bracketed=False, V_oc_scaps=pt["Voc_V"])
            rows.append(row)
            print(f"  N_t={row['N_t']:.2e}  V_oc_sl={row['V_oc']:>7.4f}  V_oc_scaps={row['V_oc_scaps']:>7.4f}")
        results[variant] = rows
        print()

    print("=" * 72)
    print("Summary")
    print("=" * 72)
    print(f"{'variant':>20}  {'V_oc range SL':>15}  {'V_oc range SCAPS':>17}  {'closure':>10}")
    print("-" * 72)
    for variant, rows in results.items():
        physical = [r for r in rows if r["voc_bracketed"]]
        if physical:
            sl_voc = [r["V_oc"] for r in physical]
            sl_range = max(sl_voc) - min(sl_voc)
            scaps_at_pts = [r["V_oc_scaps"] for r in physical]
            sc_range = max(scaps_at_pts) - min(scaps_at_pts)
            closure = sl_range / sc_range if sc_range > 0 else float("inf")
        else:
            sl_range = sc_range = closure = 0.0
        print(f"{variant:>20}  {sl_range * 1000:>13.2f} mV  {sc_range * 1000:>15.2f} mV  {closure * 100:>9.2f}%")

    # Verdict
    base_range = max(r["V_oc"] for r in results["baseline"]) - min(r["V_oc"] for r in results["baseline"])
    aug_range = max(r["V_oc"] for r in results["auger_off"]) - min(r["V_oc"] for r in results["auger_off"])
    all_range = max(r["V_oc"] for r in results["all_recomb_off"]) - min(r["V_oc"] for r in results["all_recomb_off"])
    print()
    if aug_range > 5.0e-3 and aug_range > 2 * base_range:
        verdict = "AUGER CONFIRMED as ceiling — Y1 diagnosis correct"
    elif all_range > 5.0e-3 and all_range > 2 * aug_range:
        verdict = "RADIATIVE is the ceiling (not Auger) — diagnosis WRONG"
    elif all_range < 5.0e-3:
        verdict = "Something else limits V_oc — re-investigate transport / contacts / numerics"
    else:
        verdict = "Mixed signals — closer inspection needed"
    print(f"Verdict: {verdict}")

    # CSV
    csv_path = OUT_DIR / "nt_c_pvk_kill_auger.csv"
    with open(csv_path, "w", newline="") as f:
        fields = ["variant", "N_t", "V_oc", "voc_bracketed", "V_oc_scaps"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for variant, rows in results.items():
            for r in rows:
                writer.writerow({"variant": variant, **{k: r.get(k, "") for k in fields[1:]}})
    print(f"\nCSV: {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
