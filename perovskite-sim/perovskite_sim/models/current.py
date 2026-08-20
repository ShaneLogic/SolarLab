"""Decomposed current density dataclass."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CurrentComponents:
    """Per-face current density decomposition [A/m^2].

    All arrays have shape (N-1,), one value per mesh face.
    Sign convention: positive when the device delivers power (solar convention).
    """

    J_n: np.ndarray       # electron conduction current
    J_p: np.ndarray       # hole conduction current
    J_ion: np.ndarray     # ionic current (positive + negative species)
    J_disp: np.ndarray    # dielectric displacement current
    J_total: np.ndarray   # sum of all components


@dataclass(frozen=True)
class IonicCurrentComponents:
    """Per-species ionic face currents [A/m^2].

    Keeping the charge-current channels separate prevents a dual-ion state
    from passing a DC gate merely because two non-zero species currents
    cancel in ``J_total``.
    """

    J_positive: np.ndarray
    J_negative: np.ndarray | None
    J_total: np.ndarray
