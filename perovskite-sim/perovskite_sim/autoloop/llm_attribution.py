# perovskite_sim/autoloop/llm_attribution.py
from __future__ import annotations

import logging

from perovskite_sim.autoloop.ablation import CANDIDATE_FLAGS
from perovskite_sim.autoloop.attribution import HeuristicAttributor
from perovskite_sim.autoloop.cognition import CognitionRuntime
from perovskite_sim.autoloop.types import Hypothesis

logger = logging.getLogger(__name__)

ATTRIBUTION_SCHEMA = {
    "required": ["cause", "mechanism"],
    "cause_enum": ["bug", "numerics", "physics", "data"],
}


def build_attribution_prompt(gap, matrix, negatives) -> str:
    probes = "\n".join(
        f"  - {p.name} [{p.kind}]: delta={p.delta:.4g} ok={p.ok}" for p in matrix.probes)
    flags = ", ".join(sorted({f for fl in CANDIDATE_FLAGS.values() for f in fl}))
    refuted = "\n".join(f"  - {n.approach}" for n in getattr(negatives, "negatives", []))
    return (
        "You are diagnosing why a perovskite drift-diffusion simulator (SolarLab) "
        "disagrees with its reference on one metric. The deterministic heuristic found "
        "NO known-flag lever for this gap.\n\n"
        f"GAP: metric={gap.metric}, sweep={gap.sweep}, kind={gap.kind}, "
        f"solarlab={gap.solarlab_val:.4g} vs reference={gap.reference_val:.4g}.\n\n"
        f"ABLATION MATRIX (baseline badness {matrix.baseline_val:.4g}; "
        "delta<0 means that variant improved the gap):\n" + probes + "\n\n"
        f"FLAGS ALREADY TRIED (none a clear lever): {flags}\n\n"
        "REFUTED approaches — do NOT propose any of these:\n" + (refuted or "  (none)") + "\n\n"
        "Propose the single most likely NOVEL root cause — a physics term, numerical "
        "issue, bug, or data problem NOT behind any existing flag. Be specific and "
        "mechanistic (name the term, equation, or site).\n"
        "Output ONLY a JSON object: "
        '{"cause": "bug|numerics|physics|data", "mechanism": "<specific>", '
        '"evidence_for": ["..."], "evidence_against": ["..."], "confidence": 0.0}')


class LLMAttributor:
    """Composes over HeuristicAttributor: runs the heuristic first; on an
    'uncertain' verdict, asks the LLM for a NOVEL cause. The LLM result is ALWAYS
    a verdict='uncertain' lead (never auto-confirmed). Any runtime failure
    degrades to the heuristic's uncertain — the LLM never blocks the loop."""

    def __init__(self, runtime: CognitionRuntime, *, heuristic=None):
        self.runtime = runtime
        self.heuristic = heuristic or HeuristicAttributor()

    def attribute(self, gap, matrix, negatives) -> Hypothesis:
        hyp = self.heuristic.attribute(gap, matrix, negatives)
        if hyp.verdict != "uncertain":
            return hyp
        try:
            out = self.runtime.complete(
                build_attribution_prompt(gap, matrix, negatives), ATTRIBUTION_SCHEMA)
        except Exception as exc:                       # never block on the LLM
            logger.warning("LLM attribution failed for %s: %r — keeping heuristic uncertain",
                           gap.id, exc)
            return hyp
        mechanism = out["mechanism"]
        ev_for = tuple(out.get("evidence_for", ())) + (
            f"LLM novel-cause lead (confidence {out.get('confidence', '?')})",)
        ev_against = tuple(out.get("evidence_against", ()))
        if negatives.is_refuted(mechanism):
            ev_against = ev_against + ("matches a REFUTED approach in the negatives ledger",)
        return Hypothesis(gap_id=gap.id, cause=out["cause"], mechanism=mechanism,
                          evidence_for=ev_for, evidence_against=ev_against,
                          verifier_votes=0, verdict="uncertain")
