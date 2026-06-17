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
