"""Field-dependent mobility models for drift-diffusion transport.

Two empirical mobility models are provided here and composed multiplicatively
when both are requested:

1. **Caughey-Thomas velocity saturation** — high-field reduction of the
   drift mobility when ``μ₀ · |E|`` approaches the saturation velocity
   ``v_sat``:

       μ_CT(E) = μ₀ / (1 + (μ₀ · |E| / v_sat)^β)^(1/β)

   For ``β = 2`` this is the Canali form used for silicon electrons; β = 1
   gives the Thornber / Scharfetter-Gummel soft-saturation form used for
   silicon holes. The asymptote at large |E| is v_sat / |E| independent of
   β, so the carrier drift velocity saturates at v_sat as expected.

2. **Poole-Frenkel field-assisted hopping** — low-field enhancement from
   the field-lowered trap barrier in disordered / organic transport
   layers:

       μ_PF(E) = μ₀ · exp(γ_PF · √|E|)

   with γ_PF in units of [(V/m)^-0.5]. Reduces to μ₀ as E → 0 by
   construction; for a hopping transport layer with γ_PF ~ 3e-4
   (V/m)^-0.5 (typical for spiro-OMeTAD), μ roughly doubles at |E| = 1e6
   V/m — a regime regularly reached inside perovskite devices.

The two models target different materials, so in a general stack the
absorber may need only CT and the HTL only PF. Composition is
multiplicative:

    μ(E) = μ₀ · PF(E) · CT(E; μ₀ · PF(E))

i.e. PF scales the low-field mobility first, then CT caps that enhanced
mobility as the field saturates. For the common case where only one
model is active in a given layer (v_sat = 0 disables CT, γ_PF = 0
disables PF) the composition collapses to the active model.

Numerical notes
---------------
* ``|E|`` is taken as ``np.abs(E)`` so both models are symmetric in the
  sign of the applied field — physically correct because the drift speed
  depends only on |E|, not its direction.
* All three field-mobility parameters degenerate gracefully: v_sat = 0,
  β = 0, and γ_PF = 0 each leave μ untouched. This lets
  ``build_material_arrays`` mask out layers that opted out of field
  enhancement without branching.

The functions here operate on numpy arrays of any shape (scalar, per
node, per face), so they can drop into the RHS hot path on the face
grid once per call without reallocating.
"""
from __future__ import annotations

import numpy as np


def caughey_thomas(
    mu0: np.ndarray,
    E_abs: np.ndarray,
    v_sat: np.ndarray,
    beta: np.ndarray,
) -> np.ndarray:
    """Caughey-Thomas velocity-saturation mobility.

    Parameters
    ----------
    mu0
        Low-field (field-independent) mobility [m²/(V·s)].
    E_abs
        Absolute electric field magnitude [V/m]. Must be ≥ 0.
    v_sat
        Carrier saturation velocity [m/s]. ``v_sat = 0`` returns ``mu0``
        unchanged (CT disabled at this location).
    beta
        CT exponent. ``β ≤ 0`` returns ``mu0`` unchanged.

    Returns
    -------
    μ(E) with the same shape as ``mu0``.
    """
    mu0 = np.asarray(mu0, dtype=float)
    E_abs = np.abs(np.asarray(E_abs, dtype=float))
    v_sat = np.asarray(v_sat, dtype=float)
    beta = np.asarray(beta, dtype=float)

    # Broadcast to the largest shape, then disable CT where inputs are
    # degenerate (v_sat == 0 or beta <= 0) by forcing the denominator to 1.
    active = (v_sat > 0.0) & (beta > 0.0)

    # Safe denominator: compute only where active, else return mu0.
    # (mu0 * E_abs) / v_sat can overflow for tiny v_sat, so clip inputs.
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        ratio = np.where(
            active,
            mu0 * E_abs / np.where(v_sat > 0.0, v_sat, 1.0),
            0.0,
        )
        denom = np.where(
            active,
            (1.0 + ratio ** np.where(active, beta, 1.0)) ** (1.0 / np.where(active, beta, 1.0)),
            1.0,
        )

    return mu0 / denom


def poole_frenkel(
    mu0: np.ndarray,
    E_abs: np.ndarray,
    gamma_pf: np.ndarray,
) -> np.ndarray:
    """Poole-Frenkel field-enhanced mobility.

    Parameters
    ----------
    mu0
        Low-field mobility [m²/(V·s)].
    E_abs
        Absolute electric field magnitude [V/m]. Must be ≥ 0.
    gamma_pf
        PF prefactor [(V/m)^-0.5]. ``γ_PF = 0`` returns ``mu0``
        unchanged.

    Returns
    -------
    μ(E) with the same shape as ``mu0``.
    """
    mu0 = np.asarray(mu0, dtype=float)
    E_abs = np.abs(np.asarray(E_abs, dtype=float))
    gamma_pf = np.asarray(gamma_pf, dtype=float)

    # exp(γ · √|E|) grows fast for large E; cap the argument to prevent
    # overflow. exp(80) ≈ 5.5e34 is comfortably inside float64 range; any
    # larger field-enhancement factor is almost certainly a config error.
    arg = gamma_pf * np.sqrt(E_abs)
    arg = np.clip(arg, -80.0, 80.0)
    return mu0 * np.exp(arg)


def apply_field_mobility(
    mu0: np.ndarray,
    E_abs: np.ndarray,
    v_sat: np.ndarray,
    beta: np.ndarray,
    gamma_pf: np.ndarray,
) -> np.ndarray:
    """Compose Poole-Frenkel and Caughey-Thomas: PF first, then CT.

    For the common case where only one model is active per layer (v_sat =
    0 or γ_PF = 0), the composition reduces to that active model.

    Parameters mirror :func:`caughey_thomas` and :func:`poole_frenkel`.
    """
    mu_pf = poole_frenkel(mu0, E_abs, gamma_pf)
    return caughey_thomas(mu_pf, E_abs, v_sat, beta)


__all__ = ["caughey_thomas", "poole_frenkel", "apply_field_mobility"]
