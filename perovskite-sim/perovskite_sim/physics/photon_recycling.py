"""Photon recycling / detailed-balance radiative recombination.

Reabsorbs internally emitted band-edge photons so that the *effective*
radiative recombination coefficient becomes ``B_rad_eff = B_rad * P_esc``
inside each absorber layer, where ``P_esc`` is the probability that a
photon emitted isotropically inside the absorber escapes the front
surface without being reabsorbed. Only the escaping fraction is a net
carrier loss — the rest generates another e-h pair and feeds back into
the continuity equations.

Model (Yablonovitch limit, single-pass approximation)
-----------------------------------------------------
For weak escape we take

    P_esc ≈ 1 / (4 · n_abs² · OD_gap)

where

    OD_gap = α(λ_gap) · d_absorber

is the dimensionless *intrinsic* optical depth at the band-edge
wavelength ``λ_gap = h c / Eg``, ``α(λ_gap) = 4 π k(λ_gap) / λ_gap`` is
the absorber's linear absorption coefficient, and ``n_abs`` is its real
refractive index at that wavelength.

Using the intrinsic α·d (rather than the TMM absorbance A(x, λ) which
already folds in Fresnel losses, Beer-Lambert attenuation and thin-film
interference) matches the original Yablonovitch derivation and gives
physically meaningful escape probabilities for device-scale films.

Limits:
  * weak absorption, OD_gap → 0 ⇒ formula diverges, so ``P_esc`` is
    clamped to 1 (thin-film photons escape freely; no recycling).
  * strong absorption, OD_gap → ∞ ⇒ ``P_esc → 0`` — every emitted
    photon is reabsorbed before reaching the surface. Effective B_rad
    vanishes and V_oc approaches the non-radiative bound.
  * Intermediate OD_gap gives the interesting regime: ``B_rad`` is
    reduced by factor ``P_esc`` and V_oc rises by ``V_T · ln(1/P_esc)``
    relative to the no-recycling case.

This is the recombination-side leg of the 3.1 plan: internally generated
photons that get reabsorbed no longer count as lost carriers, so the
effective ``B_rad`` is scaled by ``P_esc``. The emission-side
reciprocity (radiative source term into G(x)) is deferred to Phase 3.1b
because it breaks the "G computed once at build time" invariant.
"""
from __future__ import annotations

import numpy as np

from perovskite_sim.constants import Q

# Planck constant × speed of light [J·m]. Using CODATA values.
_H_PLANCK = 6.62607015e-34   # J·s
_C_LIGHT = 2.99792458e8      # m/s
_HC = _H_PLANCK * _C_LIGHT    # J·m


def wavelength_at_gap(Eg_eV: float) -> float:
    """Return the band-edge wavelength λ_gap = hc/Eg [m]."""
    if Eg_eV <= 0.0:
        raise ValueError(f"Eg must be positive, got {Eg_eV}")
    return _HC / (Eg_eV * Q)


def _interp_absorbance_at_gap(
    A_xl: np.ndarray,
    wavelengths_m: np.ndarray,
    lam_gap: float,
) -> np.ndarray:
    """Interpolate A(x, λ) at λ = lam_gap → shape (N_x,).

    If ``lam_gap`` is outside the tabulated range the absorbance is
    clamped to the nearest endpoint (``np.interp`` default), which for a
    band-edge past the longest tabulated wavelength means the long-wave
    absorbance — usually near-zero — is used; for a gap inside the
    visible band the interpolation is direct.
    """
    # np.interp needs the x-axis increasing; wavelengths_m is in metres
    # and already monotone from build_material_arrays.
    if wavelengths_m[0] > wavelengths_m[-1]:
        wavelengths_m = wavelengths_m[::-1]
        A_xl = A_xl[:, ::-1]
    # (N_x,) — vectorised per-row interp using broadcasting via np.interp
    # in a loop (N_x is small, ~50-300, not worth custom vectorisation).
    out = np.empty(A_xl.shape[0])
    for i in range(A_xl.shape[0]):
        out[i] = np.interp(lam_gap, wavelengths_m, A_xl[i, :])
    return out


def _clamp(p: float) -> float:
    if not np.isfinite(p):
        return 1.0
    if p >= 1.0:
        return 1.0
    if p <= 0.0:
        return 0.0
    return float(p)


def compute_p_esc(
    alpha_gap: float,
    thickness: float,
    n_at_gap: float,
) -> float:
    """Escape probability from intrinsic α·d (Yablonovitch form).

    ``P_esc = min(1, 1 / (4 · n² · α · d))``.

    Parameters
    ----------
    alpha_gap : α(λ_gap) in [m⁻¹]; use ``4πk/λ`` from the layer's
        complex index at the band edge.
    thickness : absorber thickness d [m].
    n_at_gap : real refractive index at λ_gap.

    Returns
    -------
    P_esc in [0, 1].
    """
    if alpha_gap <= 0.0 or thickness <= 0.0 or n_at_gap <= 0.0:
        return 1.0
    OD = alpha_gap * thickness
    if not np.isfinite(OD):
        return 1.0
    return _clamp(1.0 / (4.0 * n_at_gap * n_at_gap * OD))


def compute_p_esc_for_absorber(
    A_xl: np.ndarray,
    wavelengths_m: np.ndarray,
    x_absorber: np.ndarray,
    Eg_eV: float,
    n_at_gap: float,
) -> float:
    """Escape probability from the integrated TMM absorbance (alt form).

    Uses ``OD_gap = ∫_{absorber} A(x, λ_gap) dx`` (dimensionless). This
    already folds in Fresnel + thin-film interference, so it gives a
    *smaller* OD than the intrinsic α·d form and therefore a larger
    P_esc for the same device. It is kept as an alternative entry point
    for tests; ``compute_p_esc`` is the primary path used in the solver.
    """
    if Eg_eV <= 0.0 or n_at_gap <= 0.0:
        return 1.0
    if A_xl.ndim != 2:
        raise ValueError(
            f"A_xl must be 2D (N_x, N_wl); got shape {A_xl.shape}"
        )
    if A_xl.shape[0] != x_absorber.shape[0]:
        raise ValueError(
            f"A_xl first axis ({A_xl.shape[0]}) must match x_absorber "
            f"length ({x_absorber.shape[0]})"
        )
    if A_xl.shape[1] != wavelengths_m.shape[0]:
        raise ValueError(
            f"A_xl second axis ({A_xl.shape[1]}) must match "
            f"wavelengths_m length ({wavelengths_m.shape[0]})"
        )
    if x_absorber.size < 2:
        return 1.0

    lam_gap = wavelength_at_gap(Eg_eV)
    A_gap = _interp_absorbance_at_gap(A_xl, wavelengths_m, lam_gap)
    OD_gap = float(np.trapezoid(A_gap, x_absorber))
    if not np.isfinite(OD_gap) or OD_gap <= 0.0:
        return 1.0
    return _clamp(1.0 / (4.0 * n_at_gap * n_at_gap * OD_gap))


__all__ = [
    "compute_p_esc",
    "compute_p_esc_for_absorber",
    "wavelength_at_gap",
]
