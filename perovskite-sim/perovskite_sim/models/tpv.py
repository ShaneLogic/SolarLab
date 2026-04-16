"""Transient photovoltage result dataclass."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TPVResult:
    """Result of a transient photovoltage (TPV) experiment.

    The device is held at open circuit under steady illumination, then a
    small light pulse is applied. The voltage transient V(t) decays back
    to V_oc as the excess carriers recombine.

    All quantities in SI units.
    """

    t: np.ndarray          # time array [s]
    V: np.ndarray          # voltage transient [V]
    J: np.ndarray          # terminal current density [A/m^2] (≈ 0 at OC)
    V_oc: float            # steady-state open-circuit voltage [V]
    tau: float             # fitted mono-exponential decay time [s]
    delta_V0: float        # initial voltage perturbation amplitude [V]
