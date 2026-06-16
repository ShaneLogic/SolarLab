# perovskite_sim/autoloop/scorecard.py
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Callable, Optional

from perovskite_sim.autoloop.types import Gap, ParityScore, SweepScore

# Only these reference sheets have a SolarLab sweep-axis mapping (Stage 1).
SHEET_TO_AXIS: dict[str, str] = {
    "CHI_ETL":     "etl_delta_ec_eV",
    "Nd_ETL":      "etl_doping_cm3",
    "Nt_PVK ETL":  "interface_defect_N_t_cm2",
    "Nt_C_PVK":    "absorber_defect_density_cm3",
}

# Callable injected by the orchestrator (real solver) or a test (fake).
#   run_point(axis, x)  -> (V_oc, J_sc_A_m2, FF_frac, PCE_frac, bracketed)
#   base_point()        -> (V_oc, J_sc_A_m2, FF_frac, PCE_frac, bracketed)
RunPoint = Callable[[str, float], tuple[float, float, float, float, bool]]
BasePoint = Callable[[], tuple[float, float, float, float, bool]]


def _voc_closure(sl_vocs: list[float], scaps_vocs: list[float]) -> float:
    if len(sl_vocs) < 2:
        return float("nan")
    sl_range = (max(sl_vocs) - min(sl_vocs)) * 1000.0
    scaps_range = (max(scaps_vocs) - min(scaps_vocs)) * 1000.0
    if scaps_range <= 0.0:
        return float("nan")
    return 100.0 * sl_range / scaps_range


def score_parity(*, reference_path: Path, config_path: Path,
                 run_point: RunPoint, base_point: BasePoint,
                 skip_log: Optional[list[str]] = None) -> ParityScore:
    """Score SolarLab parity against the SCAPS reference JSON.

    ``run_point`` / ``base_point`` are injected so the math is testable
    without the solver; the orchestrator wires the real ``run_jv_sweep``.
    Unmapped reference sheets are skipped and logged (no silent cap).
    """
    ref = json.loads(Path(reference_path).read_text(encoding="utf-8"))
    per_sweep: dict[str, SweepScore] = {}

    for sheet, axis in SHEET_TO_AXIS.items():
        sweep = ref["sweeps"].get(sheet)
        if sweep is None:
            if skip_log is not None:
                skip_log.append(f"reference missing sweep '{sheet}' — skipped")
            continue
        sl_vocs, scaps_vocs, n_brk = [], [], 0
        pts = sweep["points"]
        for pt in pts:
            x = float(pt["x"])
            voc_sl, _jsc, _ff, _pce, bracketed = run_point(axis, x)
            if bracketed and voc_sl == voc_sl:   # not NaN
                sl_vocs.append(voc_sl)
                scaps_vocs.append(float(pt["Voc_V"]))
                n_brk += 1
        per_sweep[sheet] = SweepScore(
            sweep=sheet,
            voc_closure_pct=_voc_closure(sl_vocs, scaps_vocs),
            n_points=len(pts),
            n_bracketed=n_brk,
        )

    # base-point absolute deltas (solarlab - reference)
    bm = ref["base_model"]
    voc, jsc_A, ff, pce, _brk = base_point()
    base_deltas = {
        "V_oc": voc - float(bm["Voc_V"]),
        "J_sc": jsc_A - float(bm["Jsc_mA_cm2"]) * 10.0,
        "FF":   ff - float(bm["FF_percent"]) / 100.0,
        "PCE":  pce - float(bm["PCE_percent"]) / 100.0,
    }

    closures = [s.voc_closure_pct for s in per_sweep.values() if s.voc_closure_pct == s.voc_closure_pct]
    overall = max(0.0, min(1.0, mean(closures) / 100.0)) if closures else 0.0
    return ParityScore(overall=overall, base_deltas=base_deltas, per_sweep=per_sweep)


def gaps_from_score(score: ParityScore, *, cycle: int,
                    closure_target: float = 70.0,
                    base_tol: Optional[dict[str, float]] = None) -> list[Gap]:
    """Emit a ranked Gap per under-target sweep + per out-of-tolerance base metric."""
    base_tol = base_tol or {"V_oc": 0.02, "J_sc": 10.0, "FF": 0.03, "PCE": 0.02}
    gaps: list[Gap] = []

    for sheet, s in score.per_sweep.items():
        if s.voc_closure_pct == s.voc_closure_pct and s.voc_closure_pct < closure_target:
            mag = (closure_target - s.voc_closure_pct) / 100.0
            gaps.append(Gap(
                id=f"trend:{sheet}:V_oc", metric="V_oc", sweep=sheet, sweep_point=float("nan"),
                solarlab_val=s.voc_closure_pct, reference_val=closure_target, gap_mag=mag,
                kind="trend", status="open", found_cycle=cycle, last_attempt_cycle=cycle,
            ))

    for metric, delta in score.base_deltas.items():
        tol = base_tol.get(metric, float("inf"))
        if abs(delta) > tol:
            gaps.append(Gap(
                id=f"absolute:base:{metric}", metric=metric, sweep="base", sweep_point=0.0,
                solarlab_val=delta, reference_val=0.0, gap_mag=abs(delta),
                kind="absolute", status="open", found_cycle=cycle, last_attempt_cycle=cycle,
            ))

    gaps.sort(key=lambda g: -g.gap_mag)
    return gaps
