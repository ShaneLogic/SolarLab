from __future__ import annotations

from perovskite_sim.autoloop.ledger import Ledger
from perovskite_sim.autoloop.types import NegativeResult

SEED_NEGATIVE_RESULTS: list[NegativeResult] = [
    NegativeResult(
        approach="DOS-cap projection target",
        why_failed="false convergence, high residual at bulk node 19",
        evidence="project_scaps_validation_parked memory; TE-ceiling analysis 2026-06-14",
    ),
    NegativeResult(
        approach="BBD face-density interface term",
        why_failed="V=0.08 blow-up",
        evidence="project_scaps_validation_parked memory (E8 prototype)",
    ),
    NegativeResult(
        approach="1.40 V_bi fudge",
        why_failed="+106 mV unexplained over derived flat-band V_bi; fails G4 honest-residual",
        evidence="project_scaps_root_cause_reanalysis memory 2026-06-07",
    ),
    NegativeResult(
        approach="two-sided additive mirror interface channel",
        why_failed="measured no-op (mirror pair minority-limited, ~uA/m2)",
        evidence="perovskite-sim CLAUDE.md interface_two_sided note",
    ),
    NegativeResult(
        approach="shared-occupancy on bulk-node densities",
        why_failed="CBO trend collapse 80->22%; trap algebra invisible under bulk-node sampling",
        evidence="perovskite-sim CLAUDE.md interface_shared_occupancy note",
    ),
]


def seed_negative_results(ledger: Ledger) -> None:
    """Idempotently load the known-refuted approaches into the ledger."""
    for neg in SEED_NEGATIVE_RESULTS:
        ledger.add_negative(neg)   # add_negative already dedups on normalised key
