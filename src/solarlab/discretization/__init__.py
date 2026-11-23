"""Shared discrete operators: Bernoulli, Scharfetter-Gummel, face means, divergence.

The package namespace exports the M-class (legacy bit-identical) kernels. The
N-class stable evaluations are imported from their own modules:
:mod:`solarlab.discretization.stable_bernoulli` and
:mod:`solarlab.discretization.particle_flux`.
"""

from solarlab.discretization.bernoulli import bernoulli, bernoulli_derivative
from solarlab.discretization.divergence import flux_divergence
from solarlab.discretization.face_average import (
    DIELECTRIC_RULES, LAYER_SERIES_CAPACITANCE_V2, NODAL_HARMONIC_DIELECTRIC_V1,
    VACUUM_PERMITTIVITY_F_M, arithmetic_face_mean, guarded_harmonic_face_mean,
    harmonic_face_mean, harmonic_mean, series_face_capacitance,
    subface_series_capacitance,
)
from solarlab.discretization.scharfetter_gummel import (
    LegacyFaceJacobian, legacy_sg_current_n, legacy_sg_current_n_jacobian,
    legacy_sg_current_p, legacy_sg_current_p_jacobian,
)

__all__ = [
    "DIELECTRIC_RULES", "LAYER_SERIES_CAPACITANCE_V2", "LegacyFaceJacobian",
    "NODAL_HARMONIC_DIELECTRIC_V1", "VACUUM_PERMITTIVITY_F_M", "arithmetic_face_mean",
    "bernoulli", "bernoulli_derivative", "flux_divergence", "guarded_harmonic_face_mean",
    "harmonic_face_mean", "harmonic_mean", "legacy_sg_current_n",
    "legacy_sg_current_n_jacobian", "legacy_sg_current_p", "legacy_sg_current_p_jacobian",
    "series_face_capacitance", "subface_series_capacitance",
]
