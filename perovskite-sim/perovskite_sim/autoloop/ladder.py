# perovskite_sim/autoloop/ladder.py
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional

from perovskite_sim.autoloop.scorecard import (
    SHEET_TO_AXIS, score_parity, RunPoint, BasePoint,
)
from perovskite_sim.autoloop.types import LadderResult, ParityScore

# Default JV envelope — the campaign parity setting (verified).
DEFAULT_JV_KWARGS = dict(N_grid=30, n_points=20, v_rate=5.0, V_max=1.6)


def run_l0(paths: list[str]) -> tuple[bool, str]:
    """L0 numerics gate: run a fast pytest subset; pass iff returncode == 0."""
    proc = subprocess.run(
        ["python", "-m", "pytest", "-q", "-m", "not slow", *paths],
        capture_output=True, text=True,
    )
    ok = proc.returncode == 0
    tail = (proc.stdout or proc.stderr).strip().splitlines()
    return ok, (tail[-1] if tail else "")


def run_l1_limiting_cases(run_voc_radiative_only, detailed_balance_ceiling: float,
                          run_dark_jsc) -> tuple[bool, dict]:
    """L1 physics-sanity gate. Two cheap limiting checks:

    - rad-only V_oc must not exceed the detailed-balance ceiling.
    - dark J_sc must be ~0 (no photocurrent without light).
    Both probes are injected so the module stays solver-agnostic/testable.
    """
    voc_rad = run_voc_radiative_only()
    dark_jsc = run_dark_jsc()
    ok = (voc_rad <= detailed_balance_ceiling + 1e-6) and (abs(dark_jsc) < 1.0)  # A/m^2
    return ok, {"voc_radiative_only": voc_rad, "ceiling": detailed_balance_ceiling,
                "dark_jsc": dark_jsc}


def build_run_callables(config_path: Path,
                        jv_kwargs: Optional[dict] = None) -> tuple[RunPoint, BasePoint]:
    """Wire the real solver into the scorecard's injected interface.

    Imports the heavy sim lazily so importing this module is cheap.
    """
    from perovskite_sim.experiments.jv_sweep import run_jv_sweep
    from perovskite_sim.scaps_compat import load_scaps_yaml
    from perovskite_sim.sweeps.device_parameter_sweep import SweepPoint, apply_sweep_point

    jv_kwargs = {**DEFAULT_JV_KWARGS, **(jv_kwargs or {})}
    base_stack = load_scaps_yaml(config_path)

    def _metrics(stack):
        res = run_jv_sweep(stack, **jv_kwargs)
        m = res.metrics_fwd
        return (m.V_oc, m.J_sc, m.FF, m.PCE, m.voc_bracketed)

    def run_point(axis: str, x: float):
        sp = SweepPoint("p", axis, f"{x:.3e}", {axis: x})
        try:
            return _metrics(apply_sweep_point(base_stack, sp))
        except Exception:
            return (float("nan"), float("nan"), float("nan"), float("nan"), False)

    def base_point():
        return _metrics(base_stack)

    return run_point, base_point


def run_ladder(*, reference_path: Path, config_path: Path,
               l0_paths: list[str],
               run_point: Optional[RunPoint] = None,
               base_point: Optional[BasePoint] = None,
               l1: Optional[tuple[bool, dict]] = None) -> LadderResult:
    """Run L0 -> L1 -> L2 with fail-fast short-circuiting."""
    l0_ok, l0_detail = run_l0(l0_paths)
    if not l0_ok:
        return LadderResult(l0_pass=False, l1_pass=False, score=None,
                            details={"l0": l0_detail})

    l1_ok, l1_detail = (True, {}) if l1 is None else l1
    if not l1_ok:
        return LadderResult(l0_pass=True, l1_pass=False, score=None,
                            details={"l0": l0_detail, "l1": l1_detail})

    if run_point is None or base_point is None:
        run_point, base_point = build_run_callables(config_path)

    skip_log: list[str] = []
    score = score_parity(reference_path=reference_path, config_path=config_path,
                         run_point=run_point, base_point=base_point, skip_log=skip_log)
    return LadderResult(l0_pass=True, l1_pass=True, score=score,
                        details={"l0": l0_detail, "l1": l1_detail, "skipped": skip_log})
