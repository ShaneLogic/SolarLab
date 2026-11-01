"""One-sided directional derivatives of the field-dependent mobility (N class).

The composition ``mu(E) = CT(PF(mu_0, |E|), |E|)`` (``field_mobility``) is
smooth except at three kinds of faces, where the historical linearization
only stores a placeholder and ``differentiable = False``:

* zero field with a PF term and no regularization width: ``sqrt|E|`` gives
  an infinite one-sided slope, ``mu_0 gamma sqrt|d| / sqrt(h)``;
* zero field with an active CT term and ``beta <= 1``: ``|E|^beta`` gives the
  finite slope ``-mu_0^2 |d| / v_sat`` for ``beta = 1`` and an infinite
  negative slope for ``beta < 1``;
* the exact exp-clip surfaces ``|gamma s(E)| = 80``: the PF factor is frozen
  on the clipped side, so the slope depends on whether the direction enters
  the clipped region.

``field_mobility_directional_derivative`` returns
``lim_{h->0+} [mu(E + h d) - mu(E)] / h`` for every face and direction ``d``
-- equal to ``mu'(E) d`` at smooth faces, finite or signed-infinite at the
cusps -- and :func:`require_finite_directional_derivative` rejects the
infinite ones instead of handing a solver a meaningless tangent.

The field measure the law is evaluated with is a declared capability: in 1D
the face field is the whole field; the 2D solver applies the 1D law to the
face-normal component only (x faces use ``|E_x|``, y faces ``|E_y|``), an
approximation carried as its own label. Evaluating with the total field
magnitude is a different model and is refused here.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from numpy.typing import ArrayLike

from solarlab.physics.transport.field_mobility import compact_sqrt_abs, linearize_field_mobility

__all__ = [
    "FACE_FIELD_1D",
    "FACE_NORMAL_COMPONENT_2D",
    "FIELD_MEASURES",
    "FieldMeasureCapability",
    "FieldMobilityCapabilityError",
    "FieldMobilityDirectionalDerivative",
    "field_mobility_directional_derivative",
    "require_field_measure",
    "require_finite_directional_derivative",
]

_CLIP = 80.0


class FieldMobilityCapabilityError(ValueError):
    """A field-mobility request outside the declared capability."""


@dataclass(frozen=True, slots=True)
class FieldMeasureCapability:
    """Which field magnitude a face law is evaluated with."""

    label: str
    dimension: int
    exact: bool
    description: str


FACE_FIELD_1D = "face_field_1d_v1"
FACE_NORMAL_COMPONENT_2D = "face_normal_component_2d_v1"
FIELD_MEASURES: dict[str, FieldMeasureCapability] = {
    FACE_FIELD_1D: FieldMeasureCapability(
        FACE_FIELD_1D, 1, True, "1D face field -(phi[i+1]-phi[i])/dx; the complete field"
    ),
    FACE_NORMAL_COMPONENT_2D: FieldMeasureCapability(
        FACE_NORMAL_COMPONENT_2D,
        2,
        False,
        "2D approximation: x faces use |E_x| only, y faces |E_y| only (no cross-axis interpolation)",
    ),
}


def require_field_measure(label: str) -> FieldMeasureCapability:
    """Return the capability of a declared field measure or refuse it."""

    capability = FIELD_MEASURES.get(label)
    if capability is None:
        raise FieldMobilityCapabilityError(
            f"field measure {label!r} is not a declared capability; evaluating the "
            "field-dependent mobility with the total field magnitude (or any other "
            "measure) is a model change, not a numerical option"
        )
    return capability


@dataclass(frozen=True)
class FieldMobilityDirectionalDerivative:
    """``mu(E)`` and its one-sided derivative along ``direction`` on each face."""

    mobility_m2_V_s: np.ndarray
    derivative_m3_V2_s: np.ndarray
    smooth: np.ndarray

    @property
    def finite(self) -> np.ndarray:
        return np.isfinite(self.derivative_m3_V2_s)


def field_mobility_directional_derivative(
    mu0: ArrayLike,
    E: ArrayLike,
    direction: ArrayLike,
    v_sat: ArrayLike,
    beta: ArrayLike,
    gamma_pf: ArrayLike,
    *,
    pf_field_regularization_width_V_m: float = 0.0,
) -> FieldMobilityDirectionalDerivative:
    """One-sided directional derivative ``D mu(E; d)`` [m^3 V^-2 s^-1 per unit d].

    ``direction`` is the field perturbation ``d`` [V m^-1] (any real value;
    the result is positively homogeneous in ``d``). Infinite values are
    signed. When ``beta = 1/2`` and the leading PF and CT cusps cancel,
    the next term gives the finite derivative ``-mu0**2 * abs(d) / v_sat``.
    """

    linear = linearize_field_mobility(
        mu0, E, v_sat, beta, gamma_pf, pf_field_regularization_width_V_m=pf_field_regularization_width_V_m
    )
    mobility0, field, d, velocity, exponent, gamma = np.broadcast_arrays(
        np.asarray(mu0, dtype=float),
        np.asarray(E, dtype=float),
        np.asarray(direction, dtype=float),
        np.asarray(v_sat, dtype=float),
        np.asarray(beta, dtype=float),
        np.asarray(gamma_pf, dtype=float),
    )
    if not np.all(np.isfinite(d)):
        raise ValueError("direction must be finite")
    width = float(pf_field_regularization_width_V_m)
    smooth = np.broadcast_to(linear.differentiable, field.shape)
    derivative = np.array(np.broadcast_to(linear.field_derivative_m3_V2_s, field.shape) * d, dtype=float)
    mobility = np.broadcast_to(linear.mobility_m2_V_s, field.shape)

    magnitude = np.abs(field)
    ct_active = (velocity > 0.0) & (exponent > 0.0)
    zero_field = (magnitude == 0.0) & ~smooth
    pf_cusp = zero_field & (width == 0.0) & (gamma != 0.0) & (mobility0 > 0.0)
    ct_cusp = zero_field & ~pf_cusp & ct_active & (exponent <= 1.0) & (mobility0 > 0.0)
    clip_surface = ~smooth & ~zero_field

    with np.errstate(invalid="ignore", divide="ignore"):
        # PF cusp: sqrt slope ~ h^-1/2; an active CT term contributes h^(beta-1).
        pf_coeff = mobility0 * gamma * np.sqrt(np.abs(d))
        safe_velocity = np.where(velocity > 0.0, velocity, 1.0)
        pf_value = np.where(ct_active & (exponent < 0.5), -np.inf, np.sign(pf_coeff) * np.inf)
        # Set t=sqrt(|E|), a=sqrt(mu0/v_sat). For beta=1/2,
        # mu/mu0 = exp(gamma*t)/(1+a*t*exp(gamma*t/2))**2.
        # At gamma=2*a its t term vanishes and its t**2 term is -a**2.
        competing = pf_cusp & ct_active & (exponent == 0.5) & (d != 0.0)
        for index in np.flatnonzero(competing):
            # The sign of gamma - 2*sqrt(mu0/vsat) must be decided from
            # the exact supplied floats, not a rounded square root. A tiny
            # nonzero coefficient still gives an infinite limiting slope.
            g = Fraction(float(gamma.flat[index]))
            mu = Fraction(float(mobility0.flat[index]))
            velocity_exact = Fraction(float(velocity.flat[index]))
            difference = g * g * velocity_exact - 4 * mu
            if g <= 0 or difference < 0:
                pf_value.flat[index] = -np.inf
            elif difference > 0:
                pf_value.flat[index] = np.inf
            else:
                pf_value.flat[index] = -float(mu * mu * Fraction(abs(float(d.flat[index]))) / velocity_exact)
        pf_value = np.where(d == 0.0, 0.0, pf_value)
        # CT cusp (PF smooth at zero field): beta = 1 finite, beta < 1 infinite.
        ct_value = np.where(
            exponent == 1.0,
            -(mobility0 * mobility0) * np.abs(d) / safe_velocity,
            -np.inf,
        )
        ct_value = np.where(d == 0.0, 0.0, ct_value)
    derivative[pf_cusp] = pf_value[pf_cusp]
    derivative[ct_cusp] = ct_value[ct_cusp]

    if np.any(clip_surface):
        root = compact_sqrt_abs(field, width)
        sign = np.sign(field)
        root_slope = np.zeros_like(magnitude)
        positive = magnitude > 0.0
        if width == 0.0:
            root_slope[positive] = sign[positive] / (2.0 * np.sqrt(magnitude[positive]))
        else:
            exact = positive & (magnitude >= width)
            root_slope[exact] = sign[exact] / (2.0 * np.sqrt(magnitude[exact]))
            transition = positive & (magnitude < width)
            z = magnitude[transition] / width
            root_slope[transition] = (
                sign[transition] * (154.0 * z - 264.0 * z**3 + 126.0 * z**5) / 32.0 / np.sqrt(width)
            )
        argument = gamma * root
        argument_rate = gamma * root_slope * d
        entering = ((argument >= _CLIP) & (argument_rate > 0.0)) | ((argument <= -_CLIP) & (argument_rate < 0.0))
        pf_log_rate = np.where(entering, 0.0, argument_rate)
        mobility_pf = mobility0 * np.exp(np.clip(argument, -_CLIP, _CLIP))
        ratio = np.where(ct_active, mobility_pf * magnitude / safe_velocity, 0.0)
        with np.errstate(invalid="ignore", over="ignore"):
            power = np.where(ct_active, ratio ** np.where(ct_active, exponent, 1.0), 0.0)
        weight = np.where(np.isposinf(power), 1.0, np.where(ct_active, power / (1.0 + power), 0.0))
        inverse_field = np.divide(sign * d, magnitude, out=np.zeros_like(magnitude), where=positive)
        clipped = mobility * ((1.0 - weight) * pf_log_rate - weight * inverse_field)
        derivative[clip_surface] = clipped[clip_surface]

    derivative.setflags(write=False)
    return FieldMobilityDirectionalDerivative(
        mobility_m2_V_s=np.array(mobility, dtype=float),
        derivative_m3_V2_s=derivative,
        smooth=np.array(smooth, dtype=bool),
    )


def require_finite_directional_derivative(result: FieldMobilityDirectionalDerivative) -> np.ndarray:
    """The directional derivative, refusing faces where it is not finite."""

    if not np.all(result.finite):
        count = int(np.sum(~result.finite))
        raise FieldMobilityCapabilityError(
            f"field-dependent mobility has no finite one-sided derivative on {count} face(s) "
            "(zero-field Poole-Frenkel cusp or Caughey-Thomas beta < 1); use a regularization "
            "width ladder or exclude the faces instead of a placeholder tangent"
        )
    return result.derivative_m3_V2_s
