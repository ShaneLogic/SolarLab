"""Sandboxed extension point for autoloop Stage 5.3 LLM codegen.

`solver/mol.build_material_arrays` calls `adjust_material_arrays` ONCE on the
assembled MaterialArrays, but only when the `autoloop_generated_lever` device
flag (or env `SOLARLAB_AUTOLOOP_GEN=1`) is set. The default body is the identity
transform; with the flag off (every legacy/parity config) this module is never
imported, so the solver is bit-identical. Stage 5.3 codegen overwrites ONLY the
body of `adjust_material_arrays` — never the signature, the context, or the hook."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class _LeverContext:
    """Read-only inputs a generated lever may use: the spatial grid `x` and the
    `DeviceStack` (`stack`). Typed as `object` to keep this module dependency-light
    (no solver/models import → no cycle when imported inside the hook guard)."""
    x: object
    stack: object


def adjust_material_arrays(arrays, ctx):
    """Identity transform (default). A generated lever returns
    `dataclasses.replace(arrays, <field>=...)` with shifted band parameters
    (chi / Eg / ni_sq / tau_n / tau_p / B_rad ...). MUST be pure: no I/O, no
    globals, no mutation of `arrays` (it is frozen) — return a replaced copy."""
    return arrays
