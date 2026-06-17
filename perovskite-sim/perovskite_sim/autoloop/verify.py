# perovskite_sim/autoloop/verify.py
from __future__ import annotations

import dataclasses
import logging

from perovskite_sim.autoloop.cognition import CognitionRuntime
from perovskite_sim.autoloop.types import GateVerdict, Hypothesis

logger = logging.getLogger(__name__)

SKEPTIC_LENSES = ["physical-plausibility", "numerical-artifact", "data-support"]
VOTE_SCHEMA = {"required": ["refuted"]}

_LENS_QUESTION = {
    "physical-plausibility": "Could this mechanism physically produce the observed gap in a "
                             "perovskite drift-diffusion model?",
    "numerical-artifact": "Could the gap be a numerical/discretisation artifact rather than this "
                          "mechanism? Does the ablation evidence rule that out?",
    "data-support": "Do the ablation probes actually SUPPORT this mechanism, or is the claim "
                    "unsupported by the evidence shown?",
}


def refute_prompt(hyp, gap, matrix, lens: str) -> str:
    probes = "\n".join(f"  - {p.name} [{p.kind}]: delta={p.delta:.4g} ok={p.ok}"
                       for p in matrix.probes)
    return (
        f"You are a skeptic reviewing a proposed root-cause via the '{lens}' lens.\n\n"
        f"CLAIM: cause={hyp.cause}; mechanism={hyp.mechanism}\n"
        f"GAP: metric={gap.metric}, sweep={gap.sweep}, "
        f"solarlab={gap.solarlab_val:.4g} vs reference={gap.reference_val:.4g}\n"
        f"ABLATION EVIDENCE (delta<0 = that variant improved the gap):\n{probes}\n\n"
        f"{_LENS_QUESTION.get(lens, 'Is the claim well-supported?')}\n"
        "Try to REFUTE the claim from your lens. Default refuted=true if you cannot find solid "
        "support. Output ONLY a JSON object: "
        '{"refuted": true|false, "reason": "<one sentence>"}')


class MultiSkepticVerifier:
    """N diverse-lens skeptics each try to refute a hypothesis mechanism. Strict
    majority of a quorum decides; errored skeptics are excluded (never counted as
    a refutation); below quorum the hypothesis is returned unchanged (uncertain)."""

    def __init__(self, runtime: CognitionRuntime, *, lenses=None, quorum: int = 2):
        self.runtime = runtime
        self.lenses = lenses or SKEPTIC_LENSES
        self.quorum = quorum

    def verify(self, hyp, gap, matrix) -> Hypothesis:
        ran = []   # (lens, refuted, reason) for skeptics that succeeded
        for lens in self.lenses:
            try:
                v = self.runtime.complete(refute_prompt(hyp, gap, matrix, lens), VOTE_SCHEMA)
                ran.append((lens, bool(v["refuted"]), str(v.get("reason", ""))))
            except Exception as exc:               # excluded, NOT a refutation
                logger.warning("G5 skeptic %s failed: %r", lens, exc)
        if len(ran) < self.quorum:
            logger.warning("G5 quorum not met (%d/%d) — leaving %s uncertain",
                           len(ran), self.quorum, hyp.gap_id)
            return hyp
        refutes = sum(1 for _, r, _ in ran if r)
        verdict = "refuted" if refutes > len(ran) / 2 else "confirmed"
        ev_for = tuple(f"G5 {lens}: {reason}" for lens, r, reason in ran if not r)
        ev_against = tuple(f"G5 {lens}: {reason}" for lens, r, reason in ran if r)
        return dataclasses.replace(
            hyp, verdict=verdict, verifier_votes=len(ran) - refutes,
            evidence_for=hyp.evidence_for + ev_for,
            evidence_against=hyp.evidence_against + ev_against)


def gate_g5_verify(hyp, gap, matrix, verifier: MultiSkepticVerifier) -> GateVerdict:
    """Thin wrapper filling the gate_g5 stub (pre-land reuse)."""
    out = verifier.verify(hyp, gap, matrix)
    return GateVerdict("G5_adversarial_verify", out.verdict == "confirmed",
                       f"verdict={out.verdict}, votes={out.verifier_votes}")
