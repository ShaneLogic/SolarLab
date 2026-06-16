from __future__ import annotations

from typing import Optional

from perovskite_sim.autoloop.types import GateVerdict, LadderResult, ParityScore


def run_gate_stack(result: LadderResult, *, baseline: Optional[ParityScore] = None,
                   regression_tol: float = 0.01) -> list[GateVerdict]:
    """Deterministic G1-G3. G0/G4/G5 are deferred (Stage 3, cognition).

    G1 numerics    : L0 pytest subset green.
    G2 limiting    : L1 limiting cases hold.
    G3 scorecard   : overall parity did not regress vs baseline beyond tol.
    """
    g1 = GateVerdict("G1_numerics", result.l0_pass,
                     "pytest subset green" if result.l0_pass else "L0 unit/numerics failed")
    g2 = GateVerdict("G2_limiting", result.l1_pass,
                     "limiting cases hold" if result.l1_pass else "L1 limiting case violated")

    if result.score is None:
        g3 = GateVerdict("G3_scorecard", False, "no parity score (L0/L1 short-circuited)")
    elif baseline is None:
        g3 = GateVerdict("G3_scorecard", True, f"no baseline; overall={result.score.overall:.3f}")
    else:
        regressed = result.score.overall < baseline.overall - regression_tol
        g3 = GateVerdict(
            "G3_scorecard", not regressed,
            f"overall {result.score.overall:.3f} vs baseline {baseline.overall:.3f} "
            f"(tol {regression_tol})",
        )
    return [g1, g2, g3]


def all_passed(verdicts: list[GateVerdict]) -> bool:
    return all(v.passed for v in verdicts)


# ---- Deferred gates (Stage 3 — require a proposed flag-gated change + cognition) ----
def gate_g0_deferred(*args, **kwargs):
    raise NotImplementedError("G0 legacy-bit-identical needs a proposed flagged change (Stage 3)")


def gate_g4_deferred(*, mechanism, residual):
    raise NotImplementedError("G4 honest-residual fudge-guard needs a cognition mechanism (Stage 3)")


def gate_g5_deferred(*args, **kwargs):
    raise NotImplementedError("G5 adversarial-verify needs the cognition skeptics (Stage 3)")
