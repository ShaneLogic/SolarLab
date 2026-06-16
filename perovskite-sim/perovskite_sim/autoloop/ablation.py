# perovskite_sim/autoloop/ablation.py
from __future__ import annotations

import math
from typing import Protocol

from perovskite_sim.autoloop.types import AblationMatrix, AblationProbe, Gap

# Candidate physics flags per gap bucket. Data, not logic — extend as the
# campaign learns which flags lever which gaps. Logged, never silently capped.
CANDIDATE_FLAGS: dict[str, list[str]] = {
    "interface": ["SOLARLAB_IFACE_PROJ", "SOLARLAB_IFACE_PLANE", "SOLARLAB_INTERFACE_PLANE_STATE"],
    "base":      ["SOLARLAB_DOS_BAND"],
}

_INTERFACE_SWEEPS = {"Nt_PVK ETL", "CHI_ETL", "Nd_ETL"}

GRID_N_POINTS = 80   # the high-resolution grid for the numerics probe


class ProbeRunner(Protocol):
    def run(self, variant: dict) -> float: ...


def bucket_for_gap(gap: Gap) -> str:
    """Map a gap to a candidate-flag bucket."""
    if gap.sweep in _INTERFACE_SWEEPS:
        return "interface"
    return "base"


def _safe_run(runner: ProbeRunner, variant: dict) -> tuple[float, bool, str]:
    try:
        return float(runner.run(variant)), True, ""
    except Exception as exc:           # noqa: BLE001 — recorded, not swallowed
        return math.nan, False, f"{type(exc).__name__}: {exc}"


def run_ablation(gap: Gap, probe_runner: ProbeRunner) -> AblationMatrix:
    """Run flag + grid + dark-limiting probes; record badness deltas.

    Badness is lower = closer to the SCAPS reference, so a flag probe with a
    negative delta improves the gap. A failing probe is recorded ok=False
    (not raised) so attribution can proceed on the surviving signal.
    """
    bucket = bucket_for_gap(gap)
    base_val, base_ok, base_note = _safe_run(
        probe_runner, {"env_flags": {}, "jv_overrides": {}, "measure": "gap"})

    probes: list[AblationProbe] = []
    skipped: list[str] = []

    for flag in CANDIDATE_FLAGS.get(bucket, []):
        val, ok, note = _safe_run(
            probe_runner, {"env_flags": {flag: "1"}, "jv_overrides": {}, "measure": "gap"})
        probes.append(AblationProbe(
            name=flag, kind="flag", variant={"flag": flag},
            baseline_val=base_val, variant_val=val,
            delta=(val - base_val), ok=(ok and base_ok), note=note))
    if bucket not in CANDIDATE_FLAGS:
        skipped.append(f"no candidate flags for bucket '{bucket}'")

    gval, gok, gnote = _safe_run(
        probe_runner, {"env_flags": {}, "jv_overrides": {"n_points": GRID_N_POINTS}, "measure": "gap"})
    probes.append(AblationProbe(
        name=f"grid_n{GRID_N_POINTS}", kind="grid", variant={"n_points": GRID_N_POINTS},
        baseline_val=base_val, variant_val=gval,
        delta=(gval - base_val), ok=(gok and base_ok), note=gnote))

    dval, dok, dnote = _safe_run(
        probe_runner, {"env_flags": {}, "jv_overrides": {"illuminated": False}, "measure": "dark"})
    probes.append(AblationProbe(
        name="dark_jsc", kind="limiting", variant={"illuminated": False},
        baseline_val=0.0, variant_val=dval, delta=dval, ok=dok,
        note=(dnote or "dark J_sc; expect ~0")))

    return AblationMatrix(gap_id=gap.id, baseline_val=base_val,
                          probes=tuple(probes), skipped=tuple(skipped))
