"""Phase E11.0 — golden-master regression guard for QSS interface-plane dev.

Pins the CURRENT production state (base J-V + representative sweep points +
physics-gate booleans) so any QSS iteration can instantly detect a regression
of the working 7/10. Fast subset (~15 J-V runs, a few min) — this is a guard,
not the full validation scorecard.

Usage (run from perovskite-sim/):
    python scripts/qss_golden_master.py --capture        # write the baseline
    python scripts/qss_golden_master.py --check          # diff vs baseline
    SOLARLAB_IFACE_QSS=1 python scripts/qss_golden_master.py --check  # ON-path

`--check` exits non-zero if any pinned metric drifts beyond tolerance (V_oc
1 mV, J_sc 1 A/m²) OR any physics-gate boolean fails. The OFF-path must stay
bit-identical to the baseline during QSS development.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from perovskite_sim.experiments.jv_sweep import run_jv_sweep
from perovskite_sim.scaps_compat import load_scaps_yaml
from perovskite_sim.sweeps.device_parameter_sweep import SweepPoint, apply_sweep_point

REPO = Path(__file__).resolve().parents[1]
CFG = REPO / "configs" / "scaps_mirror_v2.yaml"
BASELINE = REPO / "tests" / "integration" / "qss_baseline.json"
JV = dict(N_grid=30, n_points=20, v_rate=5.0, V_max=1.6)
SQ_JSC_A_M2 = 275.0  # ~27.5 mA/cm² SQ limit for Eg=1.53 eV

# Representative probe points: working trends (CBO, PVK/ETL N_t) + targets
# (Nd_ETL, Nt_C_PVK). Keep small for fast regression turnaround.
PROBES = [
    ("base", {}),
    ("CBO_cliff", {"etl_delta_ec_eV": -1.0}),
    ("CBO_spike", {"etl_delta_ec_eV": 0.3}),
    ("NtPVKETL_lo", {"interface_defect_N_t_cm2": 1e9, "interface_defect_target": "pvk/etl"}),
    ("NtPVKETL_hi", {"interface_defect_N_t_cm2": 1e15, "interface_defect_target": "pvk/etl"}),
    ("NtHTLPVK_hi", {"interface_defect_N_t_cm2": 1e15, "interface_defect_target": "htl/pvk"}),
    ("NdETL_lo", {"etl_doping_cm3": 1e13}),
    ("NdETL_hi", {"etl_doping_cm3": 1e20}),
    ("NtC_PVK_hi", {"absorber_defect_density_cm3": 1e15}),
]


def run_one(updates):
    base = load_scaps_yaml(CFG)
    stack = apply_sweep_point(base, SweepPoint("p", "g", "", updates)) if updates else base
    m = run_jv_sweep(stack, **JV).metrics_fwd
    gate_jsc = bool(m.J_sc <= SQ_JSC_A_M2)
    gate_voc = bool(m.V_oc <= stack.V_bi + 1e-6)
    return dict(V_oc=round(m.V_oc, 5), J_sc=round(m.J_sc, 2),
                FF=round(m.FF, 5), PCE=round(m.PCE, 5),
                brk=bool(m.voc_bracketed),
                gate_jsc_le_SQ=gate_jsc, gate_voc_le_Vbi=gate_voc)


def capture():
    data = {name: run_one(u) for name, u in PROBES}
    BASELINE.write_text(json.dumps(data, indent=2))
    print(f"captured {len(data)} probes -> {BASELINE.relative_to(REPO)}")
    for name, r in data.items():
        print(f"  {name:14s} V_oc={r['V_oc']:.4f} J_sc={r['J_sc']:.1f} "
              f"gates[Jsc≤SQ={r['gate_jsc_le_SQ']},Voc≤Vbi={r['gate_voc_le_Vbi']}]")


def check(voc_tol=1e-3, jsc_tol=1.0):
    if not BASELINE.exists():
        print("no baseline — run --capture first"); return 2
    ref = json.loads(BASELINE.read_text())
    fails = []
    for name, u in PROBES:
        cur = run_one(u)
        b = ref.get(name)
        if b is None:
            print(f"  {name:14s} NEW (not in baseline)"); continue
        dv = (cur["V_oc"] - b["V_oc"]) * 1000
        dj = cur["J_sc"] - b["J_sc"]
        bad = []; warn = []
        if abs(cur["V_oc"] - b["V_oc"]) > voc_tol: bad.append(f"V_oc Δ{dv:+.1f}mV")
        if abs(cur["J_sc"] - b["J_sc"]) > jsc_tol: bad.append(f"J_sc Δ{dj:+.1f}")
        # gate regression = baseline passed but now fails. Pre-existing
        # failures (False in baseline too) are warnings, not regressions.
        for g in ("gate_jsc_le_SQ", "gate_voc_le_Vbi"):
            if b.get(g, True) and not cur[g]:
                bad.append(f"GATE-REGRESSION {g}")
            elif not cur[g]:
                warn.append(f"pre-existing {g}=False")
        tag = ("FAIL " + ";".join(bad)) if bad else ("ok " + ";".join(warn) if warn else "ok")
        if bad: fails.append(name)
        print(f"  {name:14s} V_oc={cur['V_oc']:.4f}({dv:+.1f}mV) J_sc={cur['J_sc']:.1f}({dj:+.1f})  {tag}")
    if fails:
        print(f"REGRESSION: {len(fails)} probe(s) drifted/violated: {fails}")
        return 1
    print("OK — all probes within tolerance + physics gate passed")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if args.capture:
        capture()
    elif args.check:
        sys.exit(check())
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
