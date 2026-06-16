"""Autoloop — continuous autonomous research-loop orchestrator (Stage 1: guardian)."""
from perovskite_sim.autoloop.types import (
    Gap, Hypothesis, NegativeResult, SweepScore, ParityScore,
    LadderResult, GateVerdict, Provenance,
)
from perovskite_sim.autoloop.orchestrator import guardian_once  # noqa: E402

__all__ = [
    "Gap", "Hypothesis", "NegativeResult", "SweepScore", "ParityScore",
    "LadderResult", "GateVerdict", "Provenance",
    "guardian_once",
]
