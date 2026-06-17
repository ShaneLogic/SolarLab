# perovskite_sim/autoloop/codegen.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

from perovskite_sim.autoloop.cognition import CognitionRuntime

CODEGEN_SCHEMA = {"required": ["body", "rationale"]}   # consumed by cognition._validate

# Fields a generated lever may shift (per-node arrays on the frozen MaterialArrays).
_ALLOWED_FIELDS = ("chi", "Eg", "ni_sq", "tau_n", "tau_p", "B_rad", "alpha")


@dataclass(frozen=True)
class GeneratedLever:
    body: str          # source statements for the adjust_material_arrays body (NO def line)
    rationale: str     # one-paragraph why, for the provenance/report


class Codegen(Protocol):
    def generate(self, gap, hyp, matrix=None) -> GeneratedLever: ...


class FakeCodegen:
    """Test seam: returns a canned body, ignoring inputs."""
    def __init__(self, body: str, rationale: str = "fake"):
        self.body = body
        self.rationale = rationale

    def generate(self, gap, hyp, matrix=None) -> GeneratedLever:
        return GeneratedLever(body=self.body, rationale=self.rationale)


def codegen_prompt(gap, hyp, matrix=None) -> str:
    probes = ""
    if matrix is not None and getattr(matrix, "probes", ()):
        probes = "\n".join(f"  - {p.name} [{p.kind}]: delta={p.delta:.4g} ok={p.ok}"
                           for p in matrix.probes)
    ev = "; ".join(hyp.evidence_for) if hyp.evidence_for else "(none)"
    return (
        "You write ONE pure Python function body for a perovskite drift-diffusion "
        "solver extension point. The function is:\n\n"
        "    def adjust_material_arrays(arrays, ctx):\n"
        "        # arrays: a frozen MaterialArrays (per-node numpy arrays). ctx.x is the\n"
        "        # spatial grid, ctx.stack is the DeviceStack. Return a NEW arrays via\n"
        "        # dataclasses.replace(arrays, <field>=...). MUST be pure: no I/O, no\n"
        "        # globals, no mutation. Use `import dataclasses` and `import numpy as np`\n"
        "        # at the top of the body if needed.\n\n"
        f"CONFIRMED CAUSE: cause={hyp.cause}; mechanism={hyp.mechanism}\n"
        f"GAP: metric={gap.metric}, sweep={gap.sweep}, "
        f"solarlab={gap.solarlab_val:.4g} vs reference={gap.reference_val:.4g}\n"
        f"EVIDENCE: {ev}\n"
        f"{('ABLATION PROBES (delta<0 = improved):' + chr(10) + probes) if probes else ''}\n\n"
        f"You may shift ONLY these MaterialArrays fields: {', '.join(_ALLOWED_FIELDS)}.\n"
        "If you cannot justify a change from the evidence, return arrays unchanged "
        "(`return arrays`). Output ONLY a JSON object: "
        '{"body": "<python statements, no def line>", "rationale": "<one paragraph>"}')


class ClaudeCodegen:
    """Live codegen over the 5.1 cognition runtime."""
    def __init__(self, runtime: CognitionRuntime):
        self.runtime = runtime

    def generate(self, gap, hyp, matrix=None) -> GeneratedLever:
        out = self.runtime.complete(codegen_prompt(gap, hyp, matrix), CODEGEN_SCHEMA)
        return GeneratedLever(body=str(out["body"]), rationale=str(out.get("rationale", "")))
