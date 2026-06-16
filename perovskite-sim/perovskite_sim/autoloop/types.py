from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Optional


def _norm(text: str) -> str:
    """Normalise free text for dedup: lowercase, collapse whitespace."""
    return " ".join(text.lower().split())


@dataclass(frozen=True)
class Gap:
    """One open discrepancy between SolarLab and the reference."""
    id: str
    metric: str            # "V_oc" | "J_sc" | "FF" | "PCE"
    sweep: str             # reference sweep key, or "base"
    sweep_point: float
    solarlab_val: float
    reference_val: float
    gap_mag: float         # ranking magnitude (>= 0)
    kind: str              # "trend" | "absolute"
    status: str            # "open" | "closed" | "refuted" | "blocked"
    found_cycle: int
    last_attempt_cycle: int
    mechanism: Optional[str] = None

    def with_status(self, status: str, *, last_attempt_cycle: Optional[int] = None) -> "Gap":
        return dataclasses.replace(
            self, status=status,
            last_attempt_cycle=self.last_attempt_cycle if last_attempt_cycle is None else last_attempt_cycle,
        )

    def with_mechanism(self, mechanism: str) -> "Gap":
        return dataclasses.replace(self, mechanism=mechanism)


@dataclass(frozen=True)
class Hypothesis:
    """One attribution attempt for a gap (populated by Stage 2; defined here for stability)."""
    gap_id: str
    cause: str             # "bug" | "numerics" | "physics" | "data"
    mechanism: str
    evidence_for: tuple[str, ...] = ()
    evidence_against: tuple[str, ...] = ()
    verifier_votes: int = 0
    verdict: str = "uncertain"   # "confirmed" | "refuted" | "uncertain"
    cycle: int = 0


@dataclass(frozen=True)
class NegativeResult:
    """A refuted/failed approach the loop must never re-try."""
    approach: str
    why_failed: str
    evidence: str
    never_retry: bool = True

    def dedup_key(self) -> str:
        return _norm(self.approach)


@dataclass(frozen=True)
class SweepScore:
    sweep: str
    voc_closure_pct: float   # 100 * sl_voc_range / scaps_voc_range over bracketed points
    n_points: int
    n_bracketed: int


@dataclass(frozen=True)
class ParityScore:
    overall: float                       # 0..1, mean clamped closure across scored sweeps
    base_deltas: dict[str, float]        # metric -> (solarlab - reference) at base point
    per_sweep: dict[str, SweepScore]


@dataclass(frozen=True)
class LadderResult:
    l0_pass: bool                # numerics / unit tests
    l1_pass: bool                # limiting cases
    score: Optional[ParityScore] # L2 parity (None if L0/L1 short-circuited)
    details: dict


@dataclass(frozen=True)
class GateVerdict:
    name: str
    passed: bool
    reason: str


@dataclass(frozen=True)
class Provenance:
    run_id: str
    git_sha: str
    git_dirty: bool
    config_hash: str
    flags: dict[str, str]
    seed: int
    timestamp: str           # ISO-8601, passed in (not generated, for reproducibility)


@dataclass(frozen=True)
class AblationProbe:
    """One ablation variant's effect on the gap's badness scalar."""
    name: str
    kind: str            # "flag" | "grid" | "limiting"
    variant: dict        # the variant applied (env_flags / jv_overrides summary)
    baseline_val: float
    variant_val: float
    delta: float         # variant_val - baseline_val (negative = closer to reference)
    ok: bool
    note: str = ""


@dataclass(frozen=True)
class AblationMatrix:
    gap_id: str
    baseline_val: float
    probes: tuple[AblationProbe, ...]
    skipped: tuple[str, ...] = ()
