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

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class FieldMobilityLinearization:
    """Mobility and its signed-electric-field derivative on each face."""

    mobility_m2_V_s: np.ndarray
    field_derivative_m3_V2_s: np.ndarray
    differentiable: np.ndarray


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
    *,
    field_regularization_width_V_m: float = 0.0,
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
    field_regularization_width_V_m
        Opt-in compact transition width around ``E = 0`` [V/m]. Zero keeps
        the historical ``sqrt(abs(E))`` expression exactly.

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
    from perovskite_sim.physics.regularization import compact_sqrt_abs

    arg = gamma_pf * compact_sqrt_abs(
        E_abs,
        field_regularization_width_V_m,
    )
    arg = np.clip(arg, -80.0, 80.0)
    return mu0 * np.exp(arg)


def apply_field_mobility(
    mu0: np.ndarray,
    E_abs: np.ndarray,
    v_sat: np.ndarray,
    beta: np.ndarray,
    gamma_pf: np.ndarray,
    *,
    pf_field_regularization_width_V_m: float = 0.0,
) -> np.ndarray:
    """Compose Poole-Frenkel and Caughey-Thomas: PF first, then CT.

    For the common case where only one model is active per layer (v_sat =
    0 or γ_PF = 0), the composition reduces to that active model.

    Parameters mirror :func:`caughey_thomas` and :func:`poole_frenkel`.
    """
    mu_pf = poole_frenkel(
        mu0,
        E_abs,
        gamma_pf,
        field_regularization_width_V_m=pf_field_regularization_width_V_m,
    )
    return caughey_thomas(mu_pf, E_abs, v_sat, beta)


def linearize_field_mobility(
    mu0: np.ndarray,
    E: np.ndarray,
    v_sat: np.ndarray,
    beta: np.ndarray,
    gamma_pf: np.ndarray,
    *,
    pf_field_regularization_width_V_m: float = 0.0,
) -> FieldMobilityLinearization:
    """Differentiate the production CT/PF composition with respect to ``E``.

    The derivative is with respect to the signed electric field even though
    the constitutive law depends on ``abs(E)``.  The returned mobility is
    evaluated by :func:`apply_field_mobility` itself; the analytic work here is
    limited to its tangent.  ``differentiable`` is false at the historical
    zero-field Poole-Frenkel cusp, the zero-field CT cusp for ``beta <= 1``,
    and the exact ``exp`` clipping surfaces.  Callers performing numerical
    certification must reject those faces rather than interpreting the finite
    placeholder derivative as a physical tangent.
    """

    from perovskite_sim.physics.regularization import compact_sqrt_abs

    try:
        mobility0, field, saturation_velocity, exponent, gamma = (
            np.broadcast_arrays(
                np.asarray(mu0, dtype=float),
                np.asarray(E, dtype=float),
                np.asarray(v_sat, dtype=float),
                np.asarray(beta, dtype=float),
                np.asarray(gamma_pf, dtype=float),
            )
        )
    except ValueError as exc:
        raise ValueError("field-mobility inputs must be broadcast compatible") from exc
    inputs = (mobility0, field, saturation_velocity, exponent, gamma)
    if (
        any(not np.all(np.isfinite(value)) for value in inputs)
        or np.any(mobility0 < 0.0)
    ):
        raise ValueError(
            "field-mobility linearization requires finite inputs and "
            "non-negative low-field mobility"
        )

    root_field = compact_sqrt_abs(
        field,
        pf_field_regularization_width_V_m,
    )
    width = float(pf_field_regularization_width_V_m)
    field_magnitude = np.abs(field)
    field_sign = np.sign(field)
    root_derivative = np.zeros_like(field_magnitude)
    positive_field = field_magnitude > 0.0
    if width == 0.0:
        root_derivative[positive_field] = (
            field_sign[positive_field]
            / (2.0 * np.sqrt(field_magnitude[positive_field]))
        )
    else:
        exact = positive_field & (field_magnitude >= width)
        root_derivative[exact] = (
            field_sign[exact]
            / (2.0 * np.sqrt(field_magnitude[exact]))
        )
        transition = positive_field & (field_magnitude < width)
        z = field_magnitude[transition] / width
        polynomial_derivative = (
            154.0 * z - 264.0 * z**3 + 126.0 * z**5
        ) / 32.0
        root_derivative[transition] = (
            field_sign[transition]
            * polynomial_derivative
            / np.sqrt(width)
        )

    raw_pf_argument = gamma * root_field
    clipped_pf_argument = np.clip(raw_pf_argument, -80.0, 80.0)
    pf_clip_interior = (raw_pf_argument > -80.0) & (raw_pf_argument < 80.0)
    pf_log_derivative = np.where(
        pf_clip_interior,
        gamma * root_derivative,
        0.0,
    )
    mobility_pf = mobility0 * np.exp(clipped_pf_argument)

    ct_active = (saturation_velocity > 0.0) & (exponent > 0.0)
    safe_velocity = np.where(saturation_velocity > 0.0, saturation_velocity, 1.0)
    ratio = np.where(
        ct_active,
        mobility_pf * field_magnitude / safe_velocity,
        0.0,
    )
    with np.errstate(invalid="ignore", over="ignore"):
        ratio_power = np.where(
            ct_active,
            ratio ** np.where(ct_active, exponent, 1.0),
            0.0,
        )
    ct_weight = np.zeros_like(ratio_power)
    finite_power = ct_active & np.isfinite(ratio_power)
    ct_weight[finite_power] = (
        ratio_power[finite_power] / (1.0 + ratio_power[finite_power])
    )
    ct_weight[ct_active & np.isposinf(ratio_power)] = 1.0

    signed_inverse_field = np.divide(
        field_sign,
        field_magnitude,
        out=np.zeros_like(field_magnitude),
        where=positive_field,
    )
    mobility = apply_field_mobility(
        mobility0,
        field,
        saturation_velocity,
        exponent,
        gamma,
        pf_field_regularization_width_V_m=(
            pf_field_regularization_width_V_m
        ),
    )
    field_derivative = mobility * (
        (1.0 - ct_weight) * pf_log_derivative
        - ct_weight * signed_inverse_field
    )

    differentiable = np.ones(field.shape, dtype=bool)
    if width == 0.0:
        differentiable &= ~(
            (field_magnitude == 0.0)
            & (gamma != 0.0)
            & (mobility0 > 0.0)
        )
    differentiable &= ~(
        (field_magnitude == 0.0)
        & ct_active
        & (exponent <= 1.0)
        & (mobility_pf > 0.0)
    )
    differentiable &= ~(
        (np.abs(raw_pf_argument) == 80.0)
        & (mobility0 > 0.0)
    )
    if (
        not np.all(np.isfinite(mobility))
        or np.any(mobility < 0.0)
        or np.any(~np.isfinite(field_derivative) & differentiable)
    ):
        raise ValueError("field-mobility linearization produced a non-finite tangent")
    return FieldMobilityLinearization(
        mobility_m2_V_s=np.asarray(mobility, dtype=float),
        field_derivative_m3_V2_s=np.asarray(field_derivative, dtype=float),
        differentiable=differentiable,
    )


__all__ = [
    "FieldMobilityLinearization",
    "apply_field_mobility",
    "caughey_thomas",
    "linearize_field_mobility",
    "poole_frenkel",
]
