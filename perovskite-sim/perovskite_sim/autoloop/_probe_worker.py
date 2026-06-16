"""Probe worker (Stage 2). Runs ONE ablation variant and prints its badness.

Invoked as: python -m perovskite_sim.autoloop._probe_worker '<json payload>'
Badness (lower = closer to the SCAPS reference):
  measure="gap", trend gap     -> 100 - V_oc closure% for the sweep sheet
  measure="gap", absolute gap  -> |base_metric - reference_base_metric|
  measure="dark"               -> |dark J_sc|
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from perovskite_sim.autoloop.ladder import DEFAULT_JV_KWARGS, build_run_callables
from perovskite_sim.autoloop.reference import build_reference_source
from perovskite_sim.autoloop.scorecard import SHEET_TO_AXIS, _voc_closure

_METRIC_KEY = {"V_oc": "Voc_V", "J_sc": "Jsc_mA_cm2", "FF": "FF_percent", "PCE": "PCE_percent"}


def _badness(payload: dict) -> float:
    config = Path(payload["config"])
    jv = {**DEFAULT_JV_KWARGS, **payload.get("jv_overrides", {})}
    run_point, base_point = build_run_callables(config, jv_kwargs=jv)

    if payload["measure"] == "dark":
        # base point under no illumination -> J_sc should be ~0
        _voc, jsc, _ff, _pce, _brk = base_point()
        return abs(jsc)

    source = build_reference_source(payload["reference"])

    if payload["gap_kind"] == "trend":
        sheet = payload["gap_sweep"]
        axis = SHEET_TO_AXIS[sheet]
        sl, scaps = [], []
        for pt in source.sweep(sheet)["points"]:
            x = float(pt["x"])
            voc, _j, _f, _p, brk = run_point(axis, x)
            if brk and voc == voc:
                sl.append(voc)
                scaps.append(float(pt["Voc_V"]))
        closure = _voc_closure(sl, scaps)
        return 100.0 if closure != closure else max(0.0, 100.0 - closure)

    # absolute base gap
    voc, jsc_A, ff, pce, _brk = base_point()
    sl_map = {"V_oc": voc, "J_sc": jsc_A, "FF": ff, "PCE": pce}
    bm = source.base_metrics()
    ref_v = float(bm[_METRIC_KEY[payload["gap_metric"]]])
    if payload["gap_metric"] == "J_sc":
        ref_v *= 10.0
    elif payload["gap_metric"] in ("FF", "PCE"):
        ref_v /= 100.0
    return abs(sl_map[payload["gap_metric"]] - ref_v)


def main(argv: list[str]) -> int:
    payload = json.loads(argv[0])
    print(json.dumps({"metric": _badness(payload)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
