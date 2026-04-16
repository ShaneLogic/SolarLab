"""Spatial profile snapshot dataclasses."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SpatialSnapshot:
    """Spatial profiles at a single voltage/time point.

    Node arrays (x, phi, n, p, P, rho) have shape (N,).
    Face array (E) has shape (N-1,).
    All quantities in SI units.
    """

    x: np.ndarray        # grid positions [m]
    phi: np.ndarray      # electrostatic potential [V]
    E: np.ndarray        # electric field [V/m], shape (N-1,)
    n: np.ndarray        # electron density [m^-3]
    p: np.ndarray        # hole density [m^-3]
    P: np.ndarray        # ion (vacancy) density [m^-3]
    rho: np.ndarray      # space charge density [C/m^3]
    V_app: float         # applied voltage [V]
