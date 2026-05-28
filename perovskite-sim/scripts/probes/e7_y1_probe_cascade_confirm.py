"""E7 Y1 cascade-confirm probe — kill all ceilings simultaneously.

Tests cascade theory: V_oc is gated by max(interface_SRH, Auger, radiative,
bulk_SRH). Killing each in turn (prior kill-Auger probe) revealed each
removes only one ceiling, exposing the next. This probe removes ALL
three competing ceilings simultaneously:
  - Auger off (C_n = C_p = 0)
  - Radiative off (B_rad = 0)
  - PVK/ETL interface SRH reduced 10000× (N_t_cm2: 1e12 → 1e8)

If the Nt_C_PVK sweep V_oc range opens to ≥20 mV, cascade theory is
confirmed and we can write the E7 report with experiment-locked diagnosis.

If it stays flat, something deeper than recombination is limiting V_oc
(transport / contact-equilibrium / numerical) and further investigation
is needed.

No commits.
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
OUT_DIR = REPO_ROOT.parent / "outputs" / "scaps_e7_y1_cascade"
OUT_DIR.mkdir(parents=True, exist_ok=True)

JV_KWARGS = dict(N_grid=30, n_points=20, v_rate=5.0, V_max=1.6)


def _write_variant(cfg_dict: dict, out_path: Path) -> Path:
    new = json.loads(json.dumps(cfg_dict))
    for layer in new["layers"]:
        if layer.get("role") == "absorber":
            layer["C_n_cm6_s"] = 0.0
            layer["C_p_cm6_s"] = 0.0
            layer["B_rad_cm3_s"] = 0.0
    for iface in new.get("interfaces", []):
        if iface.get("target") == "PVK/ETL":
            iface["N_t_cm2"] = 1.0e8  # 10000× lower than PDF spec
    out_path.write_text(yaml.safe_dump(new, sort_keys=False))
    return out_path


def main() -> int:
    cfg_dict = yaml.safe_load(BASE_CFG.read_text())
    ref = json.loads(REF_PATH.read_text())
    nt_sheet = ref["sweeps"]["Nt_C_PVK"]
    ref_points = nt_sheet["points"]
    scaps_voc = [pt["Voc_V"] for pt in ref_points]
    scaps_range = max(scaps_voc) - min(scaps_voc)

    print(f"Cascade-confirm — Auger=0, B_rad=0, PVK/ETL N_t=1e8 (10000× lower)")
    print(f"  SCAPS reference: V_oc range {scaps_range * 1000:.1f} mV")
    print()

    tmp_path = OUT_DIR / "variant_cascade_off.yaml"
    _write_variant(cfg_dict, tmp_path)
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
            row = dict(N_t=x, V_oc=m.V_oc, voc_bracketed=m.voc_bracketed, V_oc_scaps=pt["Voc_V"])
        except Exception as exc:  # noqa: BLE001
            row = dict(N_t=x, V_oc=0.0, voc_bracketed=False, V_oc_scaps=pt["Voc_V"])
        rows.append(row)
        print(f"  N_t={row['N_t']:.2e}  V_oc_sl={row['V_oc']:>7.4f}  V_oc_scaps={row['V_oc_scaps']:>7.4f}")

    physical = [r for r in rows if r["voc_bracketed"]]
    if physical:
        sl_voc = [r["V_oc"] for r in physical]
        sl_range = max(sl_voc) - min(sl_voc)
        scaps_at_pts = [r["V_oc_scaps"] for r in physical]
        sc_range = max(scaps_at_pts) - min(scaps_at_pts)
        closure = sl_range / sc_range if sc_range > 0 else float("inf")
    else:
        sl_range = sc_range = closure = 0.0

    print()
    print(f"V_oc range SL:    {sl_range * 1000:.2f} mV")
    print(f"V_oc range SCAPS: {sc_range * 1000:.2f} mV")
    print(f"Closure:          {closure * 100:.2f}%")
    print()
    if sl_range >= 0.020:
        verdict = "CASCADE CONFIRMED — bulk SRH dominates when all 3 ceilings removed. Y1 = architectural, requires modifying PDF-spec values."
    elif sl_range >= 0.005:
        verdict = "PARTIAL — bulk SRH visible but other limits still present. Further investigation needed."
    else:
        verdict = "BULK SRH NOT VISIBLE even with all ceilings removed. Likely solver/transport/contact issue, not recombination cascade."
    print(f"Verdict: {verdict}")

    csv_path = OUT_DIR / "nt_c_pvk_cascade_confirm.csv"
    with open(csv_path, "w", newline="") as f:
        fields = ["N_t", "V_oc", "voc_bracketed", "V_oc_scaps"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fields})
    print(f"\nCSV: {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
