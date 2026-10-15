"""Field-dependent carrier mobility on transport faces (M class).

Two empirical laws compose multiplicatively, in a FIXED order -- Poole-Frenkel
first, Caughey-Thomas second:

    mu(E) = CT( PF(mu_0, |E|), |E| )
    PF(mu_0, |E|) = mu_0 exp(gamma sqrt|E|)          (argument clipped to [-80, 80])
    CT(mu, |E|)   = mu / (1 + (mu |E| / v_sat)^beta)^(1/beta)

so PF raises the low-field mobility and CT then caps the enhanced mobility as
the drift velocity saturates. ``v_sat = 0`` or ``beta <= 0`` disables CT and
``gamma = 0`` disables PF at that face; zero field is the exact identity
``mu(0) = mu_0``. The laws depend on ``|E|`` only.

The mobility is a function of the state (the face field), so it is evaluated
on every call and is never stored in a parameter binding; only the
state-free face parameters are bound (:class:`FieldMobilityFaceParameters`).
Einstein's relation is applied around it exactly as historically:
``D_eff = mu(E) V_T`` with ``mu_0 = D_face / V_T``.

An optional compact regularization width replaces ``sqrt|E|`` in PF near
zero field by a C2 even polynomial (research width ladders only); zero width
is the historical expression.

All arithmetic is the historical one (``perovskite_sim.physics.field_mobility``,
``physics.regularization.compact_sqrt_abs`` and the RHS face recompute in
``solver/mol.assemble_rhs``). One-sided directional derivatives at the
non-smooth points and the field-measure capability labels are in
``field_mobility_tangents`` (N class).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from solarlab.discretization.face_average import arithmetic_face_mean

__all__ = [
    "COMPOSITION_ORDER",
    "FieldMobilityFaceParameters",
    "FieldMobilityLinearization",
    "apply_field_mobility",
    "caughey_thomas",
    "compact_sqrt_abs",
    "face_electric_field",
    "field_dependent_face_diffusivity",
    "linearize_field_mobility",
    "poole_frenkel",
]

#: The composition order is part of the model; it is not configurable.
COMPOSITION_ORDER = ("poole_frenkel", "caughey_thomas")


def _finite_nonnegative(name: str, value: float) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite and non-negative")
    value = float(value)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return 0.0 if value == 0.0 else value


def compact_sqrt_abs(value: np.ndarray | float, width: float = 0.0) -> np.ndarray:
    """C2 regularization of ``sqrt(abs(value))`` inside ``|value| < width``.

    Outside the transition region the exact NumPy expression is returned; a
    zero width is the exact expression everywhere.
    """

    width = _finite_nonnegative("width", width)
    values = np.asarray(value, dtype=float)
    magnitude = np.abs(values)
    exact = np.sqrt(magnitude)
    if width == 0.0:
        return exact
    inside = magnitude < width
    if not np.any(inside):
        return exact
    z = magnitude[inside] / width
    polynomial = (77.0 * z**2 - 66.0 * z**4 + 21.0 * z**6) / 32.0
    result = np.array(exact, copy=True)
    result[inside] = np.sqrt(width) * polynomial
    return result


@dataclass(frozen=True, slots=True)
class FieldMobilityLinearization:
    """Mobility and its signed-electric-field derivative on each face.

    ``differentiable`` is False at the zero-field PF cusp (zero width), the
    zero-field CT cusp for ``beta <= 1`` and the exact exp-clip surfaces;
    the derivative stored there is a finite placeholder, not a tangent.
    """

    mobility_m2_V_s: np.ndarray
    field_derivative_m3_V2_s: np.ndarray
    differentiable: np.ndarray


def caughey_thomas(mu0: np.ndarray, E_abs: np.ndarray, v_sat: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Caughey-Thomas velocity-saturation mobility (CT disabled where degenerate)."""

    mu0 = np.asarray(mu0, dtype=float)
    E_abs = np.abs(np.asarray(E_abs, dtype=float))
    v_sat = np.asarray(v_sat, dtype=float)
    beta = np.asarray(beta, dtype=float)
    active = (v_sat > 0.0) & (beta > 0.0)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        ratio = np.where(active, mu0 * E_abs / np.where(v_sat > 0.0, v_sat, 1.0), 0.0)
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
    """Poole-Frenkel enhanced mobility ``mu_0 exp(clip(gamma sqrt|E|, -80, 80))``."""

    mu0 = np.asarray(mu0, dtype=float)
    E_abs = np.abs(np.asarray(E_abs, dtype=float))
    gamma_pf = np.asarray(gamma_pf, dtype=float)
    arg = gamma_pf * compact_sqrt_abs(E_abs, field_regularization_width_V_m)
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
    """``CT(PF(mu_0, |E|), |E|)`` -- the only composition order."""

    mu_pf = poole_frenkel(
        mu0, E_abs, gamma_pf, field_regularization_width_V_m=pf_field_regularization_width_V_m
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
    """The composition and its derivative with respect to the SIGNED field."""

    try:
        mobility0, field, saturation_velocity, exponent, gamma = np.broadcast_arrays(
            np.asarray(mu0, dtype=float),
            np.asarray(E, dtype=float),
            np.asarray(v_sat, dtype=float),
            np.asarray(beta, dtype=float),
            np.asarray(gamma_pf, dtype=float),
        )
    except ValueError as exc:
        raise ValueError("field-mobility inputs must be broadcast compatible") from exc
    inputs = (mobility0, field, saturation_velocity, exponent, gamma)
    if any(not np.all(np.isfinite(value)) for value in inputs) or np.any(mobility0 < 0.0):
        raise ValueError(
            "field-mobility linearization requires finite inputs and non-negative low-field mobility"
        )

    root_field = compact_sqrt_abs(field, pf_field_regularization_width_V_m)
    width = float(pf_field_regularization_width_V_m)
    field_magnitude = np.abs(field)
    field_sign = np.sign(field)
    root_derivative = np.zeros_like(field_magnitude)
    positive_field = field_magnitude > 0.0
    if width == 0.0:
        root_derivative[positive_field] = (
            field_sign[positive_field] / (2.0 * np.sqrt(field_magnitude[positive_field]))
        )
    else:
        exact = positive_field & (field_magnitude >= width)
        root_derivative[exact] = field_sign[exact] / (2.0 * np.sqrt(field_magnitude[exact]))
        transition = positive_field & (field_magnitude < width)
        z = field_magnitude[transition] / width
        polynomial_derivative = (154.0 * z - 264.0 * z**3 + 126.0 * z**5) / 32.0
        root_derivative[transition] = field_sign[transition] * polynomial_derivative / np.sqrt(width)

    raw_pf_argument = gamma * root_field
    clipped_pf_argument = np.clip(raw_pf_argument, -80.0, 80.0)
    pf_clip_interior = (raw_pf_argument > -80.0) & (raw_pf_argument < 80.0)
    pf_log_derivative = np.where(pf_clip_interior, gamma * root_derivative, 0.0)
    mobility_pf = mobility0 * np.exp(clipped_pf_argument)

    ct_active = (saturation_velocity > 0.0) & (exponent > 0.0)
    safe_velocity = np.where(saturation_velocity > 0.0, saturation_velocity, 1.0)
    ratio = np.where(ct_active, mobility_pf * field_magnitude / safe_velocity, 0.0)
    with np.errstate(invalid="ignore", over="ignore"):
        ratio_power = np.where(ct_active, ratio ** np.where(ct_active, exponent, 1.0), 0.0)
    ct_weight = np.zeros_like(ratio_power)
    finite_power = ct_active & np.isfinite(ratio_power)
    ct_weight[finite_power] = ratio_power[finite_power] / (1.0 + ratio_power[finite_power])
    ct_weight[ct_active & np.isposinf(ratio_power)] = 1.0

    signed_inverse_field = np.divide(
        field_sign, field_magnitude, out=np.zeros_like(field_magnitude), where=positive_field
    )
    mobility = apply_field_mobility(
        mobility0,
        field,
        saturation_velocity,
        exponent,
        gamma,
        pf_field_regularization_width_V_m=pf_field_regularization_width_V_m,
    )
    field_derivative = mobility * ((1.0 - ct_weight) * pf_log_derivative - ct_weight * signed_inverse_field)

    differentiable = np.ones(field.shape, dtype=bool)
    if width == 0.0:
        differentiable &= ~((field_magnitude == 0.0) & (gamma != 0.0) & (mobility0 > 0.0))
    differentiable &= ~(
        (field_magnitude == 0.0) & ct_active & (exponent <= 1.0) & (mobility_pf > 0.0)
    )
    differentiable &= ~((np.abs(raw_pf_argument) == 80.0) & (mobility0 > 0.0))
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


@dataclass(frozen=True, slots=True)
class FieldMobilityFaceParameters:
    """State-free CT/PF parameters of one carrier on the transport faces.

    Face values are the arithmetic mean of the node values (historical), so
    a face between an opted-in and an opted-out layer gets half strength.
    """

    saturation_velocity_m_s: np.ndarray
    ct_exponent: np.ndarray
    pf_gamma_sqrt_m_V: np.ndarray

    def __post_init__(self) -> None:
        arrays = []
        for name in ("saturation_velocity_m_s", "ct_exponent", "pf_gamma_sqrt_m_V"):
            array = np.array(getattr(self, name), dtype=float, copy=True)
            if array.ndim != 1 or not np.all(np.isfinite(array)):
                raise ValueError(f"{name} must be a finite face array")
            array.setflags(write=False)
            object.__setattr__(self, name, array)
            arrays.append(array)
        if any(array.shape != arrays[0].shape for array in arrays):
            raise ValueError("field-mobility face parameters must share one face set")

    @classmethod
    def from_nodes(
        cls, saturation_velocity_m_s: ArrayLike, ct_exponent: ArrayLike, pf_gamma_sqrt_m_V: ArrayLike
    ) -> FieldMobilityFaceParameters:
        return cls(
            arithmetic_face_mean(saturation_velocity_m_s),
            arithmetic_face_mean(ct_exponent),
            arithmetic_face_mean(pf_gamma_sqrt_m_V),
        )

    @property
    def opted_in(self) -> bool:
        """Whether any face carries an active CT or PF parameter (tier gate aside)."""

        return bool(np.any(self.saturation_velocity_m_s > 0.0) or np.any(self.pf_gamma_sqrt_m_V > 0.0))


def face_electric_field(phi_V: ArrayLike, x_nodes_m: ArrayLike) -> np.ndarray:
    """Face field ``E = -(phi[i+1] - phi[i]) / dx`` [V m^-1] (historical RHS)."""

    phi = np.asarray(phi_V, dtype=float)
    x = np.asarray(x_nodes_m, dtype=float)
    if phi.ndim != 1 or phi.shape != x.shape:
        raise ValueError("phi and x must be 1-D arrays on the same nodes")
    dx = np.diff(x)
    return -(phi[1:] - phi[:-1]) / dx


def field_dependent_face_diffusivity(
    diffusivity_face_m2_s: ArrayLike,
    thermal_voltage_V: float,
    field_V_m: ArrayLike,
    parameters: FieldMobilityFaceParameters,
    *,
    pf_field_regularization_width_V_m: float = 0.0,
) -> np.ndarray:
    """Per-face ``D_eff = mu(E) V_T`` with ``mu_0 = D_face / V_T`` [m^2 s^-1].

    A fresh array on every call: the result depends on the state and must
    not be cached.
    """

    mobility_base = np.asarray(diffusivity_face_m2_s, dtype=float) / thermal_voltage_V
    mobility = apply_field_mobility(
        mobility_base,
        field_V_m,
        parameters.saturation_velocity_m_s,
        parameters.ct_exponent,
        parameters.pf_gamma_sqrt_m_V,
        pf_field_regularization_width_V_m=pf_field_regularization_width_V_m,
    )
    return mobility * thermal_voltage_V
