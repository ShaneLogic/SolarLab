# perovskite_sim/autoloop/gates_impl.py
from __future__ import annotations

from typing import Callable

from perovskite_sim.autoloop.types import GateVerdict


def gate_g4_reconciles(predicted_delta: float, realized_delta: float,
                       *, tol: float = 0.5) -> GateVerdict:
    """G4 honest gate for flag-promotion: the promoted flag's measured benefit
    must actually materialize. Deltas are badness changes (negative = improvement).
    Pass iff realized improves (negative) AND reconciles with the predicted
    magnitude within ``tol`` (relative)."""
    if predicted_delta >= 0:
        return GateVerdict("G4_reconcile", False,
                           f"no predicted improvement (Δpred {predicted_delta:.3g} >= 0)")
    improved = realized_delta < 0
    reconciles = abs(realized_delta - predicted_delta) <= tol * abs(predicted_delta)
    passed = improved and reconciles
    return GateVerdict("G4_reconcile", passed,
                       f"Δpred {predicted_delta:.3g}, Δreal {realized_delta:.3g}, "
                       f"tol {tol} (improved={improved}, reconciles={reconciles})")


def gate_g0_bit_identical(golden_runner: Callable[[], tuple[bool, str]]) -> GateVerdict:
    """G0: the legacy/golden regression suite must stay green with the edit
    applied (legacy tier forces the flag off, so it holds by construction; this
    verifies it). ``golden_runner() -> (ok, detail)`` is injected."""
    ok, detail = golden_runner()
    return GateVerdict("G0_legacy_bit_identical", ok, detail)
