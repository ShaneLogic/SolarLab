"""Electroluminescence (EL) + EQE_EL result dataclass."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ELResult:
    """Reciprocity-based electroluminescence spectrum and non-radiative loss.

    Built from Rau (2007) reciprocity:

        Phi_EL(lambda) = A_abs(lambda) . phi_bb(lambda, T) . exp(q V_inj / kT)

    where ``A_abs`` is the absorber absorptance extracted from the TMM
    stack and ``phi_bb`` is the blackbody spectral photon flux. The
    dark-current J_inj is measured by the drift-diffusion solver at
    V_app = V_inj, and

        EQE_EL = J_em_rad / |J_inj|     (dimensionless, in [0, 1])
        dV_nr  = -(kT/q) . ln(EQE_EL)   (non-radiative V_oc penalty, V)

    All fields in SI except ``delta_V_nr_mV`` (millivolts) and
    ``wavelengths_nm`` (nanometres).
    """

    wavelengths_nm: np.ndarray     # probe wavelengths [nm]
    EL_spectrum: np.ndarray        # external EL flux [photons / m^2 / s / nm]
    absorber_absorptance: np.ndarray  # A_abs(lambda) [-], in [0, 1]
    V_inj: float                   # applied forward bias [V]
    J_inj: float                   # dark injection current [A/m^2] (signed)
    J_em_rad: float                # radiative emission current [A/m^2]
    EQE_EL: float                  # external radiative efficiency [-]
    delta_V_nr_mV: float           # non-radiative V_oc loss [mV]
    T: float                       # device temperature [K]
