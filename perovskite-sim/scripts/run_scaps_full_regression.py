"""Phase E8 — full SCAPS-PDF trend regression (all sweeps in 1D-SCAPS 模拟.pdf).

Extends run_scaps_v2_regression.py from the 4 marquee sweeps to every
single-variable sweep present in tests/integration/scaps_reference.json,
and reports per-sweep trend fidelity (direction + range closure) so we can
see at a glance which PDF tests SolarLab already matches.

Run from `perovskite-sim/`:

    python scripts/run_scaps_full_regression.py [--out-dir DIR]
    SOLARLAB_IFACE_PROJ=1 python scripts/run_scaps_full_regression.py --out-dir DIR

The PROJ env var toggles the Phase E8 interface-plane projection.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from perovskite_sim.experiments.jv_sweep import run_jv_sweep
from perovskite_sim.scaps_compat import load_scaps_yaml
from perovskite_sim.sweeps.device_parameter_sweep import SweepPoint, apply_sweep_point

REPO_ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = REPO_ROOT / "configs" / "scaps_mirror_v2.yaml"
REF_PATH = REPO_ROOT / "tests" / "integration" / "scaps_reference.json"

# Map each reference sheet → the updates dict passed to apply_sweep_point.
# `value_to_updates(x)` returns the updates for sweep value x.
# Sheets with no available sweep axis are marked unsupported (skipped with
# an explicit note rather than a silent miss).


def _updates_chi_etl(x):
    return {"etl_delta_ec_eV": float(x)}


def _updates_nd_etl(x):
    return {"etl_doping_cm3": float(x)}


def _updates_bulk_nt(x):
    # Drives the absorber combined SRH lifetime (CB+VB collapsed by loader).
    return {"absorber_defect_density_cm3": float(x)}


def _updates_bulk_et_cb(x):
    return {"absorber_defect_depth_eV": float(x), "trap_depth_reference": "below_cb"}


def _updates_bulk_et_vb(x):
    return {"absorber_defect_depth_eV": float(x), "trap_depth_reference": "above_vb"}


def _updates_iface_nt_pvk_etl(x):
    return {"interface_defect_N_t_cm2": float(x), "interface_defect_target": "pvk/etl"}


def _updates_iface_nt_htl_pvk(x):
    return {"interface_defect_N_t_cm2": float(x), "interface_defect_target": "htl/pvk"}


def _updates_iface_et_pvk_etl(x):
    return {"interface_defect_E_t_eV": float(x), "interface_defect_target": "pvk/etl"}


def _updates_iface_et_htl_pvk(x):
    return {"interface_defect_E_t_eV": float(x), "interface_defect_target": "htl/pvk"}


# sheet -> (updates_fn, note). updates_fn=None => unsupported (no axis yet).
SHEET_MAP = {
    "CHI_ETL": (_updates_chi_etl, "CBO band offset"),
    "Nd_ETL": (_updates_nd_etl, "ETL donor doping"),
    "Nt_C_PVK": (_updates_bulk_nt, "PVK bulk N_t (CB; combined SRH)"),
    "Nt_V_PVK": (_updates_bulk_nt, "PVK bulk N_t (VB; same combined SRH as CB)"),
    "Nt_HTL PVK": (_updates_iface_nt_htl_pvk, "HTL/PVK interface N_t"),
    "Nt_PVK ETL": (_updates_iface_nt_pvk_etl, "PVK/ETL interface N_t"),
    "Et_C_PVK": (_updates_bulk_et_cb, "PVK bulk E_t below CB"),
    "Et_V_PVK": (_updates_bulk_et_vb, "PVK bulk E_t above VB"),
    "Et_HTL PVK": (_updates_iface_et_htl_pvk, "HTL/PVK interface E_t"),
    "Et_PVK ETL": (_updates_iface_et_pvk_etl, "PVK/ETL interface E_t"),
}

# n_points=40: the 20-point grid's linear interpolation across the diode knee
# under-read V_oc by 10-16 mV (see run_scaps_validation.py).
JV_KWARGS = dict(N_grid=30, n_points=40, v_rate=5.0, V_max=1.6)

# Detailed-balance-ceiling guard (same criterion as run_scaps_validation.py):
# degenerate sweep points whose extracted V_oc reaches the absorber's
# radiative-limit ceiling are counted as unbracketed so they cannot pollute
# the direction/range statistics.
from run_scaps_validation import _radiative_voc_ceiling  # noqa: E402


def run_sheet(sheet, updates_fn, ref_points):
    base = load_scaps_yaml(CFG_PATH)
    rows = []
    for pt in ref_points:
        x = float(pt["x"])
        sp = SweepPoint("p", sheet, f"{x:.3e}", updates_fn(x))
        try:
            swept = apply_sweep_point(base, sp)
            m = run_jv_sweep(swept, **JV_KWARGS).metrics_fwd
            bracketed = m.voc_bracketed
            if bracketed:
                ceiling = _radiative_voc_ceiling(swept, max(float(m.J_sc), 1.0))
                if m.V_oc >= ceiling:
                    print(f"    x={x:.2e} EXCLUDED V_oc={m.V_oc:.3f} >= ceiling {ceiling:.3f}")
                    bracketed = False
            rows.append((x, m.V_oc, bracketed, pt["Voc_V"]))
        except Exception as e:
            rows.append((x, float("nan"), False, pt["Voc_V"]))
            print(f"    x={x:.2e} FAILED {type(e).__name__}: {e}")
    return rows


def trend_stats(rows):
    brk = [(x, v, s) for (x, v, b, s) in rows if b and v == v]
    if len(brk) < 2:
        return None
    xs = [r[0] for r in brk]
    sl = [r[1] for r in brk]
    sc = [r[2] for r in brk]
    sl_range = (max(sl) - min(sl)) * 1000
    sc_range = (max(sc) - min(sc)) * 1000
    # net direction: sign of V_oc change from first to last bracketed point
    sl_dir = (sl[-1] - sl[0])
    sc_dir = (sc[-1] - sc[0])
    dir_match = (sl_dir >= 0) == (sc_dir >= 0)
    closure = 100 * sl_range / sc_range if sc_range > 1e-6 else float("nan")
    # "flat both" if SCAPS range < 5 mV
    flat_scaps = sc_range < 5.0
    flat_sl = sl_range < 5.0
    return dict(n=len(brk), sl_range=sl_range, sc_range=sc_range,
                closure=closure, dir_match=dir_match, flat_scaps=flat_scaps,
                flat_sl=flat_sl, sl_dir=sl_dir, sc_dir=sc_dir)


def verdict(st):
    if st is None:
        return "INSUFFICIENT (<2 bracketed)"
    if st["flat_scaps"]:
        return "TREND-MET (flat both)" if st["flat_sl"] else f"SCAPS flat, SL moves {st['sl_range']:.0f}mV"
    if not st["dir_match"]:
        return f"DIR MISMATCH (SL {st['sl_dir']*1000:+.0f} vs SCAPS {st['sc_dir']*1000:+.0f} mV)"
    if st["closure"] >= 70:
        return f"TREND-MET ({st['closure']:.0f}%)"
    if st["closure"] >= 40:
        return f"PARTIAL ({st['closure']:.0f}%)"
    return f"WEAK ({st['closure']:.0f}%)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(REPO_ROOT.parent / "outputs" / "scaps_full"))
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ref = json.loads(REF_PATH.read_text())
    proj = os.environ.get("SOLARLAB_IFACE_PROJ", "") == "1"
    dos = os.environ.get("SOLARLAB_DOS_BAND", "") == "1"
    print(f"PROJ={'ON' if proj else 'off'}  DOS={'ON' if dos else 'off'}  config={CFG_PATH.name}\n")
    summary = {}
    t0 = time.time()
    for sheet, (fn, note) in SHEET_MAP.items():
        if sheet not in ref["sweeps"]:
            print(f"[skip] {sheet}: not in reference")
            continue
        if fn is None:
            print(f"[skip] {sheet}: no sweep axis ({note})")
            continue
        rows = run_sheet(sheet, fn, ref["sweeps"][sheet]["points"])
        st = trend_stats(rows)
        v = verdict(st)
        summary[sheet] = dict(note=note, verdict=v, stats=st,
                              rows=[[r[0], r[1], r[2], r[3]] for r in rows])
        print(f"{sheet:14s} {v:32s} [{note}]")
    (out / "summary.json").write_text(json.dumps(
        {"proj": proj, "dos": dos, "summary": summary}, indent=2))
    print(f"\nWrote {out/'summary.json'}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
