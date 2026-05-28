"""Phase E6.4 — scaps_mirror_v2.yaml regression vs scaps_reference.json.

Runs the marquee SCAPS sweeps (CBO, ETL doping, PVK/ETL interface N_t,
PVK bulk N_t) through the Phase E6.3 loader extension and compares the
resulting SolarLab metrics against the partner xlsx ground truth. Emits
CSV per sweep + a JSON summary so Phase E6.5 (decision gate) can pick a
pass/fail threshold from real data.

Run from `perovskite-sim/`:

    python scripts/run_scaps_v2_regression.py [--out-dir outputs/scaps_validation_e6]
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median

from perovskite_sim.experiments.jv_sweep import run_jv_sweep
from perovskite_sim.scaps_compat import load_scaps_yaml
from perovskite_sim.sweeps.device_parameter_sweep import (
    SweepPoint,
    apply_sweep_point,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = REPO_ROOT / "configs" / "scaps_mirror_v2.yaml"
REF_PATH = REPO_ROOT / "tests" / "integration" / "scaps_reference.json"

# Map ground-truth sheet name → apply_sweep_point axis key.
_SHEET_TO_AXIS = {
    "CHI_ETL": "etl_delta_ec_eV",
    "Nd_ETL": "etl_doping_cm3",
    "Nt_PVK ETL": "interface_defect_N_t_cm2",
    "Nt_C_PVK": "absorber_defect_density_cm3",
}

# SCAPS xlsx N_t for interface defects is cm^-3 (PDF column-header is
# unit-ambiguous). The existing scaps_compat.loader interface handler
# treats `N_t_cm2` as areal cm^-2 at the heterojunction face; the partner
# sweep range is the same numerical sequence regardless of nominal unit.
# `apply_sweep_point` keys carry the SolarLab-side semantics, so the
# sweep value passed is the SCAPS-input number used as-is.

JV_KWARGS = dict(N_grid=30, n_points=20, v_rate=5.0, V_max=1.6)


@dataclass
class PointResult:
    x: float
    V_oc_sl: float
    J_sc_sl: float
    FF_sl: float
    PCE_sl: float
    V_oc_scaps: float
    J_sc_scaps: float
    FF_scaps: float
    PCE_scaps: float
    voc_bracketed: bool


def _scaps_jsc_A_m2(jsc_ma_cm2: float) -> float:
    return jsc_ma_cm2 * 10.0  # mA/cm^2 → A/m^2


def run_sweep(stack_path: Path, sheet: str, ref_points: list[dict]) -> list[PointResult]:
    base_stack = load_scaps_yaml(stack_path)
    axis = _SHEET_TO_AXIS[sheet]
    out: list[PointResult] = []
    for pt in ref_points:
        x = float(pt["x"])
        updates = {axis: x}
        if axis == "absorber_doping_cm3":
            updates["absorber_doping_type"] = "donor"
        sp = SweepPoint("p", axis, f"{x:.3e}", updates)
        swept = apply_sweep_point(base_stack, sp)
        try:
            res = run_jv_sweep(swept, **JV_KWARGS)
            m = res.metrics_fwd
            out.append(PointResult(
                x=x,
                V_oc_sl=m.V_oc, J_sc_sl=m.J_sc, FF_sl=m.FF, PCE_sl=m.PCE,
                V_oc_scaps=pt["Voc_V"],
                J_sc_scaps=_scaps_jsc_A_m2(pt["Jsc_mA_cm2"]),
                FF_scaps=pt["FF_percent"] / 100.0,
                PCE_scaps=pt["PCE_percent"] / 100.0,
                voc_bracketed=m.voc_bracketed,
            ))
        except Exception as e:
            print(f"  [{sheet} x={x:.3e}] FAILED: {type(e).__name__}: {e}")
            out.append(PointResult(
                x=x,
                V_oc_sl=float("nan"), J_sc_sl=float("nan"),
                FF_sl=float("nan"), PCE_sl=float("nan"),
                V_oc_scaps=pt["Voc_V"],
                J_sc_scaps=_scaps_jsc_A_m2(pt["Jsc_mA_cm2"]),
                FF_scaps=pt["FF_percent"] / 100.0,
                PCE_scaps=pt["PCE_percent"] / 100.0,
                voc_bracketed=False,
            ))
    return out


def summarize(sheet: str, points: list[PointResult]) -> dict:
    """Compute parity metrics, restricting range to points where SL bracketed V_oc.

    Unbracketed points (V_oc=0 sentinel) would inflate the SL range
    artificially — at low ETL doping SolarLab fails to bracket V_oc
    within the V_max=1.6 sweep, but SCAPS still reports a finite V_oc.
    Range comparison only makes sense on the working-regime sub-set.
    """
    bracketed = [p for p in points if p.voc_bracketed and p.V_oc_sl == p.V_oc_sl]
    voc_deltas_mv = [(p.V_oc_sl - p.V_oc_scaps) * 1000.0 for p in bracketed]
    # SCAPS range across ALL points (SCAPS always brackets).
    scaps_voc_range_mv_all = (
        max(p.V_oc_scaps for p in points) - min(p.V_oc_scaps for p in points)
    ) * 1000.0
    # SCAPS range restricted to the same sub-set SolarLab brackets — this
    # is the comparable range for closure %.
    if bracketed:
        scaps_voc_range_mv = (
            max(p.V_oc_scaps for p in bracketed)
            - min(p.V_oc_scaps for p in bracketed)
        ) * 1000.0
        sl_voc_range_mv = (
            max(p.V_oc_sl for p in bracketed) - min(p.V_oc_sl for p in bracketed)
        ) * 1000.0
    else:
        scaps_voc_range_mv = float("nan")
        sl_voc_range_mv = float("nan")
    closure_pct = (
        100.0 * sl_voc_range_mv / scaps_voc_range_mv
        if scaps_voc_range_mv and scaps_voc_range_mv > 0.0 else float("nan")
    )
    return {
        "sheet": sheet,
        "n_points": len(points),
        "n_voc_bracketed": len(bracketed),
        "scaps_voc_range_full_mV": scaps_voc_range_mv_all,
        "scaps_voc_range_bracketed_subset_mV": scaps_voc_range_mv,
        "sl_voc_range_mV": sl_voc_range_mv,
        "voc_range_closure_pct": closure_pct,
        "voc_delta_mean_mV": mean(voc_deltas_mv) if voc_deltas_mv else float("nan"),
        "voc_delta_median_mV": median(voc_deltas_mv) if voc_deltas_mv else float("nan"),
        "voc_delta_max_abs_mV": max((abs(v) for v in voc_deltas_mv), default=float("nan")),
    }


def write_csv(path: Path, points: list[PointResult]) -> None:
    lines = [
        "x,V_oc_sl,V_oc_scaps,V_oc_delta_mV,"
        "J_sc_sl,J_sc_scaps,FF_sl,FF_scaps,PCE_sl,PCE_scaps,voc_bracketed"
    ]
    for p in points:
        lines.append(
            f"{p.x:.6e},{p.V_oc_sl:.6f},{p.V_oc_scaps:.6f},"
            f"{(p.V_oc_sl - p.V_oc_scaps) * 1000:.3f},"
            f"{p.J_sc_sl:.4f},{p.J_sc_scaps:.4f},"
            f"{p.FF_sl:.6f},{p.FF_scaps:.6f},"
            f"{p.PCE_sl:.6f},{p.PCE_scaps:.6f},{p.voc_bracketed}"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out-dir",
        default=str(REPO_ROOT.parent / "outputs" / "scaps_validation_e6"),
        help="output directory for CSV + JSON summary",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ref = json.loads(REF_PATH.read_text())
    print(f"Config: {CFG_PATH.relative_to(REPO_ROOT)}")
    print(f"Ref:    {REF_PATH.relative_to(REPO_ROOT)} (extracted {ref['extracted_at']})")
    print(f"Out:    {out_dir}")
    print()

    summaries: list[dict] = []
    t0_all = time.time()
    for sheet in _SHEET_TO_AXIS:
        if sheet not in ref["sweeps"]:
            print(f"[skip] sheet {sheet!r} not in reference")
            continue
        ref_points = ref["sweeps"][sheet]["points"]
        print(f"=== {sheet} ({len(ref_points)} pts) ===")
        t0 = time.time()
        results = run_sweep(CFG_PATH, sheet, ref_points)
        dt = time.time() - t0
        sl_csv = out_dir / f"{sheet.replace(' ', '_').replace('/', '_')}.csv"
        write_csv(sl_csv, results)
        summary = summarize(sheet, results)
        summary["wall_time_s"] = dt
        summaries.append(summary)
        print(
            f"  voc closure: {summary['voc_range_closure_pct']:.1f}%  "
            f"(SL Δ={summary['sl_voc_range_mV']:.1f} mV vs SCAPS "
            f"{summary['scaps_voc_range_bracketed_subset_mV']:.1f} mV "
            f"[full subset: {summary['scaps_voc_range_full_mV']:.1f} mV])"
        )
        print(
            f"  voc Δ: median={summary['voc_delta_median_mV']:.1f} mV, "
            f"max|Δ|={summary['voc_delta_max_abs_mV']:.1f} mV"
        )
        print(f"  bracketed: {summary['n_voc_bracketed']}/{summary['n_points']}  "
              f"({dt:.1f}s)")
        print()

    out_json = out_dir / "summary.json"
    out_json.write_text(json.dumps({
        "config": str(CFG_PATH.relative_to(REPO_ROOT)),
        "reference": str(REF_PATH.relative_to(REPO_ROOT)),
        "extracted_at": ref["extracted_at"],
        "jv_kwargs": JV_KWARGS,
        "total_wall_time_s": time.time() - t0_all,
        "summaries": summaries,
    }, indent=2))
    print(f"Wrote {out_json}")
    print(f"Total wall time: {time.time() - t0_all:.1f}s")


if __name__ == "__main__":
    main()
