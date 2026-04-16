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
