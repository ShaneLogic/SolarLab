"""V_oc(T) temperature-sweep result dataclass."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VocTResult:
    """Result of a V_oc-vs-temperature sweep.

    A linear fit ``V_oc(T) = slope · T + intercept_0K`` is applied to the raw
    samples. The intercept at T → 0 K is the standard textbook proxy for the
    activation energy of the dominant recombination pathway, measured relative
    to a nominal bandgap reference: ``E_A ≈ q · intercept_0K`` in eV.

    All quantities in SI units except where noted.
    """

    T_arr: np.ndarray            # temperature sweep points [K]
    V_oc_arr: np.ndarray         # measured V_oc at each T [V]
    J_sc_arr: np.ndarray         # J_sc at each T [A/m^2] (diagnostic)
    slope: float                 # linear-fit dV_oc/dT [V/K]
    intercept_0K: float          # linear-fit V_oc extrapolated to T=0 [V]
    E_A_eV: float                # activation energy proxy = intercept_0K [eV]
    R_squared: float             # coefficient of determination for the fit
