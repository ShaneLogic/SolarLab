# perovskite_sim/autoloop/codegen.py
from __future__ import annotations

import ast
import textwrap
from dataclasses import dataclass
from typing import Optional, Protocol

from perovskite_sim.autoloop.cognition import CognitionRuntime

CODEGEN_SCHEMA = {"required": ["body", "rationale"]}   # consumed by cognition._validate

# Static-analysis denylist for a generated lever body. Defence-in-depth: G6 runs
# the candidate in a child interpreter (real containment), but the body must be a
# pure per-node MaterialArrays transform — no I/O, no imports, no introspection —
# so we reject anything that names these out of hand BEFORE it is ever spliced or
# imported. Names that cannot appear in a pure dataclasses.replace transform.
_DENYLISTED_NAMES = frozenset({
    "os", "sys", "subprocess", "open", "exec", "eval", "compile",
    "__import__", "globals", "locals", "getattr", "setattr",
})

# Fields a generated lever may shift (per-node arrays on the frozen MaterialArrays).
_ALLOWED_FIELDS = ("chi", "Eg", "ni_sq", "tau_n", "tau_p", "B_rad", "alpha")

# Sentinels (4-space indented under the def) bounding the spliced body region.
_BODY_OPEN = "    # >>> AUTOLOOP BODY"
_BODY_CLOSE = "    # <<< AUTOLOOP BODY"

# Canonical text of the identity ``generated/lever.py``. ``splice_lever_body`` swaps
# ONLY the statements between the indented sentinels; splicing ``return arrays`` must
# reproduce this byte-for-byte so the flag-OFF / identity path stays bit-identical.
# Keep in exact sync with ``perovskite_sim/autoloop/generated/lever.py``.
LEVER_TEMPLATE = '''\
"""Sandboxed extension point for autoloop Stage 5.3 LLM codegen.

`solver/mol.build_material_arrays` calls `adjust_material_arrays` ONCE on the
assembled MaterialArrays, but only when the `autoloop_generated_lever` device
flag (or env `SOLARLAB_AUTOLOOP_GEN=1`) is set. The default body is the identity
transform; with the flag off (every legacy/parity config) this module is never
imported, so the solver is bit-identical.

Stage 5.3 codegen replaces ONLY the statements between the
`# >>> AUTOLOOP BODY` / `# <<< AUTOLOOP BODY` sentinels (via
`codegen.splice_lever_body`) — never the signature and never the imports. The
read-only context type the hook passes as `ctx` lives in the non-overwritten
`_ctx` module, so a freshly spliced `lever.py` can never drop it. This module is
re-entrant-safe only because `solver.mol` is fully imported by the time the hook
runs (the import is guarded inside `build_material_arrays`)."""
from __future__ import annotations

import dataclasses


def adjust_material_arrays(arrays, ctx):
    # >>> AUTOLOOP BODY
    return arrays
    # <<< AUTOLOOP BODY
'''


def splice_lever_body(template: str, body: str) -> str:
    """Return ``template`` with the region between the indented
    ``# >>> AUTOLOOP BODY`` / ``# <<< AUTOLOOP BODY`` sentinels replaced by ``body``.

    ``body`` is body-only statements (no ``def`` line, no imports). It is dedented
    then re-indented to the function-body level (the same column as the sentinels,
    i.e. one indent under the module-scope ``def``). Splicing ``return arrays`` into
    ``LEVER_TEMPLATE`` reproduces the identity module byte-for-byte."""
    lines = template.splitlines(keepends=True)
    try:
        open_i = next(i for i, ln in enumerate(lines) if ln.rstrip("\n") == _BODY_OPEN)
        close_i = next(i for i, ln in enumerate(lines)
                       if i > open_i and ln.rstrip("\n") == _BODY_CLOSE)
    except StopIteration as exc:  # pragma: no cover - template is a constant
        raise ValueError("lever template is missing the AUTOLOOP BODY sentinels") from exc
    body_indent = _BODY_OPEN[: len(_BODY_OPEN) - len(_BODY_OPEN.lstrip())]
    indented = textwrap.indent(textwrap.dedent(body).strip("\n") + "\n", body_indent)
    spliced = (
        "".join(lines[:open_i + 1])      # through the open sentinel
        + indented                       # the (re-indented) body
        + "".join(lines[close_i:])       # from the close sentinel onward
    )
    return spliced


def validate_lever_body(body: str) -> None:
    """Statically reject an unsafe generated lever body BEFORE it is spliced or
    imported. Raises ``ValueError`` if the body (parsed as the statements of a
    dummy ``def``) contains any ``import`` / ``from ... import``, references any
    denylisted name (``os``/``sys``/``subprocess``/``open``/``exec``/``eval``/
    ``compile``/``__import__``/``globals``/``locals``/``getattr``/``setattr``),
    or uses any dunder name. This is defence-in-depth in front of G6's real
    subprocess containment, not a replacement for it.

    The body is wrapped in a dummy ``def`` so a bare ``return`` statement parses;
    a syntax error in the body surfaces as ``ValueError`` too."""
    wrapped = "def _lever(arrays, ctx):\n" + textwrap.indent(
        textwrap.dedent(body).strip("\n") + "\n", "    ")
    try:
        tree = ast.parse(wrapped)
    except SyntaxError as exc:
        raise ValueError(f"lever body is not valid Python: {exc}") from exc
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise ValueError("lever body may not import (imports are denied)")
        if isinstance(node, ast.Name) and node.id in _DENYLISTED_NAMES:
            raise ValueError(f"lever body references denylisted name {node.id!r}")
        if isinstance(node, ast.Attribute) and node.attr in _DENYLISTED_NAMES:
            raise ValueError(f"lever body references denylisted attribute {node.attr!r}")
        name = getattr(node, "id", None) or getattr(node, "attr", None)
        if isinstance(name, str) and name.startswith("__") and name.endswith("__"):
            raise ValueError(f"lever body uses dunder name {name!r}")


@dataclass(frozen=True)
class GeneratedLever:
    body: str          # ONLY the adjust_material_arrays body statements (NO def line, NO imports)
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
        "You write ONLY the function body statements for a perovskite drift-diffusion "
        "solver extension point. The function (already declared for you) is:\n\n"
        "    def adjust_material_arrays(arrays, ctx):\n"
        "        # arrays: a frozen MaterialArrays (per-node numpy arrays). ctx.x is the\n"
        "        # spatial grid, ctx.stack is the DeviceStack. Return a NEW arrays via\n"
        "        # dataclasses.replace(arrays, <field>=...). MUST be pure: no I/O, no\n"
        "        # globals, no mutation. `dataclasses` is already imported at module\n"
        "        # scope — do NOT emit a `def` line and do NOT emit any `import`.\n\n"
        f"CONFIRMED CAUSE: cause={hyp.cause}; mechanism={hyp.mechanism}\n"
        f"GAP: metric={gap.metric}, sweep={gap.sweep}, "
        f"solarlab={gap.solarlab_val:.4g} vs reference={gap.reference_val:.4g}\n"
        f"EVIDENCE: {ev}\n"
        f"{('ABLATION PROBES (delta<0 = improved):' + chr(10) + probes) if probes else ''}\n\n"
        f"You may shift ONLY these MaterialArrays fields: {', '.join(_ALLOWED_FIELDS)}.\n"
        "If you cannot justify a change from the evidence, return arrays unchanged "
        "(`return arrays`). Output ONLY a JSON object: "
        '{"body": "<python body statements ONLY, no def line, no imports>", '
        '"rationale": "<one paragraph>"}')


class ClaudeCodegen:
    """Live codegen over the 5.1 cognition runtime."""
    def __init__(self, runtime: CognitionRuntime):
        self.runtime = runtime

    def generate(self, gap, hyp, matrix=None) -> GeneratedLever:
        out = self.runtime.complete(codegen_prompt(gap, hyp, matrix), CODEGEN_SCHEMA)
        body = str(out["body"])
        validate_lever_body(body)   # reject unsafe bodies before they leave codegen
        return GeneratedLever(body=body, rationale=str(out.get("rationale", "")))
