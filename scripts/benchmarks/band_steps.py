"""Isolated band-drive arithmetic for the SCAPS conditioning diagnostic.

The effective affinity and gap are the original transport inputs, including
any already resolved DOS correction. Their numerical eV values contribute to
the transport potentials in volts. This module changes only the order of
arithmetic; it does not select a model, form a new state, or run a solver.
"""

from __future__ import annotations

import numpy as np


def band_steps(phi, effective_affinity, effective_gap):
    """Return adjacent electron and hole transport-potential differences.

    Subtract each component before adding the differences. Adding a common
    band offset to every potential first can erase a small electric drive.
    The returned arrays are ordinary binary64 values, not exact or interval
    bounds; the original Bernoulli and flux evaluation remain separate.
    """
    fields = []
    for value in (phi, effective_affinity, effective_gap):
        array = np.asarray(value)
        if array.dtype.kind not in "fiu" or array.ndim != 1 or array.size < 2:
            raise ValueError("band components require real one-dimensional arrays")
        array = np.asarray(array, dtype=np.float64)
        if not np.isfinite(array).all():
            raise ValueError("band components must be finite")
        fields.append(array)
    potential, affinity, gap = fields
    if potential.shape != affinity.shape or potential.shape != gap.shape:
        raise ValueError("band components must share the same nodes")
    with np.errstate(over="raise", invalid="raise"):
        electron = np.diff(potential) + np.diff(affinity)
        hole = electron + np.diff(gap)
    return electron, hole
