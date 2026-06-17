"""Stable context type for the autoloop Stage 5.3 generated lever.

`_LeverContext` lives in this NON-overwritten module so that codegen can splice
a new `adjust_material_arrays` body into `lever.py` without ever touching the
context definition. `solver/mol`'s hook imports `adjust_material_arrays` from
`...generated.lever` and `_LeverContext` from `...generated._ctx`, so a freshly
spliced `lever.py` can never drop the type the hook depends on."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class _LeverContext:
    """Read-only inputs a generated lever may use: the spatial grid `x` and the
    `DeviceStack` (`stack`). Typed as `object` to keep this module dependency-light
    (no solver/models import -> no cycle when imported inside the hook guard)."""
    x: object
    stack: object
