"""E7 Y1 probe — PVK/ETL SRV reduction sensitivity test.

Tests whether lowering PVK/ETL interface SRV unmasks the Nt_C_PVK bulk
sweep (currently 0.2% closure). Sweeps the PVK/ETL N_t_cm2 calibration
at three levels: 1e12 (baseline), 1e10 (100× lower), 1e8 (10000× lower).
For each, runs the partner xlsx Nt_C_PVK sheet and reports SolarLab V_oc
range vs SCAPS reference 38.6 mV.

The sweep itself uses the existing absorber_defect_density_cm3 axis on
apply_sweep_point. The PVK/ETL N_t calibration is injected by writing
a temporary YAML variant under /tmp before each variant run.

No code changes; no commits. Output: stdout + CSV.
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
OUT_DIR = REPO_ROOT.parent / "outputs" / "scaps_e7_y1_probe"
OUT_DIR.mkdir(parents=True, exist_ok=True)

JV_KWARGS = dict(N_grid=30, n_points=20, v_rate=5.0, V_max=1.6)

# Variant N_t_cm2 values for the PVK/ETL interface defect
VARIANTS = {
    "baseline_1e12": 1.0e12,
    "lo_1e10":      1.0e10,
    "lo_1e8":       1.0e8,
}


def _write_variant(cfg_dict: dict, pvk_etl_nt_cm2: float, out_path: Path) -> Path:
    """Write a temporary YAML variant with the given PVK/ETL N_t_cm2."""
    new = json.loads(json.dumps(cfg_dict))  # deep copy via JSON round-trip
    found = False
    for iface in new.get("interfaces", []):
        if iface.get("target") == "PVK/ETL":
            iface["N_t_cm2"] = pvk_etl_nt_cm2
            found = True
    if not found:
        raise RuntimeError("PVK/ETL interface entry not found in YAML")
    out_path.write_text(yaml.safe_dump(new, sort_keys=False))
    return out_path


def main() -> int:
    cfg_dict = yaml.safe_load(BASE_CFG.read_text())
    ref = json.loads(REF_PATH.read_text())
    nt_sheet = ref["sweeps"]["Nt_C_PVK"]
    ref_points = nt_sheet["points"]
    scaps_voc = [pt["Voc_V"] for pt in ref_points]
    scaps_range = max(scaps_voc) - min(scaps_voc)

    print(f"Y1 probe — PVK/ETL SRV tune sensitivity on Nt_C_PVK sweep")
    print(f"  SCAPS reference: {len(ref_points)} points")
    print(f"  SCAPS V_oc range: {scaps_range * 1000:.1f} mV")
    print()

    results: dict[str, list[dict]] = {}

    for variant, nt_cm2 in VARIANTS.items():
        print(f"Running variant: {variant}  (PVK/ETL N_t_cm2 = {nt_cm2:.2e})")
        tmp_path = OUT_DIR / f"variant_{variant}.yaml"
        _write_variant(cfg_dict, nt_cm2, tmp_path)
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
                row = dict(
                    N_t=x,
                    V_oc=0.0,
                    voc_bracketed=False,
                    V_oc_scaps=pt["Voc_V"],
                    error=f"{type(exc).__name__}: {exc}",
                )
            rows.append(row)
            print(f"  N_t={row['N_t']:.2e}  V_oc_sl={row['V_oc']:>7.4f}  V_oc_scaps={row['V_oc_scaps']:>7.4f}")
        results[variant] = rows
        print()

    # Summarise per variant
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"{'variant':>20}  {'V_oc range SL':>15}  {'V_oc range SCAPS':>17}  {'closure':>10}")
    print("-" * 70)
    for variant, rows in results.items():
        physical = [r for r in rows if r["voc_bracketed"]]
        if physical:
            sl_voc = [r["V_oc"] for r in physical]
            sl_range = max(sl_voc) - min(sl_voc)
            scaps_at_pts = [r["V_oc_scaps"] for r in physical]
            scaps_range_v = max(scaps_at_pts) - min(scaps_at_pts)
            closure = sl_range / scaps_range_v if scaps_range_v > 0 else float("inf")
        else:
            sl_range = scaps_range_v = closure = 0.0
        print(f"{variant:>20}  {sl_range * 1000:>13.2f} mV  {scaps_range_v * 1000:>15.2f} mV  {closure * 100:>9.2f}%")

    # Write CSV
    csv_path = OUT_DIR / "nt_c_pvk_srv_tune.csv"
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
