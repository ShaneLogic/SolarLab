# perovskite_sim/autoloop/attribution.py
from __future__ import annotations

from typing import Protocol

from perovskite_sim.autoloop.ledger import Ledger
from perovskite_sim.autoloop.types import AblationMatrix, Gap, Hypothesis

# Thresholds (badness units; tunable). A probe must move badness by more than
# the tolerance to count as a signal.
GRID_TOL = 5.0     # grid-convergence shift this large -> numerics artifact
FLAG_TOL = 5.0     # a flag improving badness by this much -> physics lever
DARK_TOL = 1.0     # dark J_sc magnitude (A/m^2) above this -> bug
DOMINANCE = 2.0    # winning |signal| must beat the runner-up by this factor


class Attributor(Protocol):
    def attribute(self, gap: Gap, matrix: AblationMatrix, negatives: Ledger) -> Hypothesis: ...


def _ok(probes, kind):
    return [p for p in probes if p.kind == kind and p.ok and p.delta == p.delta]


class HeuristicAttributor:
    """Rule-based attributor over an AblationMatrix. Honest: falls back to
    'uncertain' with no dominant lever, and never confirms a mechanism that
    matches a refuted approach in the negatives ledger."""

    def attribute(self, gap: Gap, matrix: AblationMatrix, negatives: Ledger) -> Hypothesis:
        grids = _ok(matrix.probes, "grid")
        flags = _ok(matrix.probes, "flag")
        darks = _ok(matrix.probes, "limiting")

        grid_sig = max((abs(p.delta) for p in grids), default=0.0)
        best_flag = min(flags, key=lambda p: p.delta, default=None)   # most negative = best improvement
        flag_sig = -best_flag.delta if best_flag is not None and best_flag.delta < 0 else 0.0
        dark_sig = max((p.variant_val for p in darks), default=0.0)

        # 1. numerics
        if grid_sig > GRID_TOL and grid_sig >= DOMINANCE * flag_sig:
            return Hypothesis(
                gap_id=gap.id, cause="numerics",
                mechanism=f"grid-convergence sensitive (n_points->80 shifts badness by {grid_sig:.3g})",
                evidence_for=(f"grid delta {grid_sig:.3g} > tol {GRID_TOL}",),
                verifier_votes=1, verdict="confirmed")

        # 2. physics
        if best_flag is not None and flag_sig > FLAG_TOL:
            mechanism = f"flag {best_flag.name} term"
            if negatives.is_refuted(mechanism):
                return Hypothesis(
                    gap_id=gap.id, cause="physics", mechanism=mechanism,
                    evidence_for=(f"{best_flag.name} improves badness by {flag_sig:.3g}",),
                    evidence_against=("matches a REFUTED approach in the negatives ledger",),
                    verifier_votes=0, verdict="uncertain")
            return Hypothesis(
                gap_id=gap.id, cause="physics", mechanism=mechanism,
                evidence_for=(f"{best_flag.name} improves badness by {flag_sig:.3g} (> tol {FLAG_TOL})",),
                verifier_votes=1, verdict="confirmed",
                predicted_delta=best_flag.delta)   # negative = improvement, fed to G4

        # 3. bug
        if dark_sig > DARK_TOL:
            return Hypothesis(
                gap_id=gap.id, cause="bug",
                mechanism=f"limiting-case violation: dark J_sc = {dark_sig:.3g} (expect ~0)",
                evidence_for=(f"dark J_sc {dark_sig:.3g} > tol {DARK_TOL}",),
                verifier_votes=1, verdict="confirmed")

        # 4. uncertain (honest fallback)
        return Hypothesis(
            gap_id=gap.id, cause="uncertain",
            mechanism="no single ablation lever identified",
            evidence_against=(f"grid {grid_sig:.3g}, best-flag {flag_sig:.3g}, dark {dark_sig:.3g} "
                              f"all below dominance",),
            verifier_votes=0, verdict="uncertain")
