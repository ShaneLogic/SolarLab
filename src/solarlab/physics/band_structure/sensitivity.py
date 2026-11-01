"""Temperature and composition tangents of the parameter laws (N class).

The historical engine evaluates the laws only; these analytic tangents are
new. They differentiate the SAME expressions the M-class value laws
evaluate (``temperature``, ``grading``, ``narrowing``, ``binding``):

* ``dV_T/dT = k_B/q``; ``N(T)``: ``1.5 N/T``; ``v_th``: ``0.5 v/T``;
  ``mu``: ``gamma mu/T``; ``B``: ``gamma B/T``;
  ``D_ion``: ``D E_a q / (k_B T^2)``;
* Varshni: ``dEg/dT = -alpha T (T + 2 beta) / (T + beta)^2``;
* intrinsic density: ``d ln ni / dT |_Eg = 3/(2T) + Eg q / (2 k_B T^2)`` on
  both branches, ``d ln ni / dEg = -q/(2 k_B T)`` (DOS branch) or
  ``-q/(2 k_B) (1/T - 1/300)`` (scaled branch);
* grading: ``dEg/dy = Eg_B - Eg_A - b (1 - 2y)``, ``dchi/dy = chi_B - chi_A``;
* DOS fold: ``d(chi_t - chi)/dT = (chi_t - chi)/T`` (the fold is V_T times
  a temperature-independent logarithm).

Reference-temperature caveat (kept, not repaired): with both DOS declared
the intrinsic density returns the DECLARED ``ni_300`` at exactly 300 K but
the DOS formula elsewhere, so it jumps at 300 K whenever the declaration is
not mass-action consistent. ``intrinsic_density_reference_jump`` measures
the jump; the tangent at 300 K is the formula's one-sided limit, and
:func:`require_continuous_reference` refuses it when the jump is not zero.

These tangents are DC/parameter sensitivities. Using them as explicit
time-domain input derivatives for an externally driven T(t) is P04-B.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from solarlab.physics.band_structure.binding import ParameterBinding, TemperatureCapabilityError
from solarlab.physics.band_structure.temperature import T_REF, intrinsic_density_at_temperature
from solarlab.physics.statistics.thermal import BOLTZMANN_CONSTANT_J_K as K_B
from solarlab.physics.statistics.thermal import ELEMENTARY_CHARGE_C as Q

__all__ = [
    "BindingTemperatureSensitivity",
    "affinity_composition_derivative",
    "band_edge_dos_temperature_derivative",
    "band_gap_at_reference",
    "band_gap_composition_derivative",
    "band_gap_temperature_derivative",
    "binding_temperature_sensitivity",
    "grading_coordinate_slope",
    "grading_position",
    "intrinsic_density_reference_jump",
    "intrinsic_density_temperature_partials",
    "ion_diffusivity_temperature_derivative",
    "mobility_temperature_derivative",
    "radiative_coefficient_temperature_derivative",
    "require_continuous_reference",
    "thermal_velocity_temperature_derivative",
    "thermal_voltage_temperature_derivative",
]


def thermal_voltage_temperature_derivative() -> float:
    """``dV_T/dT = k_B/q`` [V K^-1]."""

    return K_B / Q


def band_edge_dos_temperature_derivative(dos_300_m3: float, temperature_K: float) -> float:
    """``d/dT [N_300 (T/300)^1.5] = 1.5 N(T)/T`` [m^-3 K^-1]."""

    return 1.5 * float(dos_300_m3) * (temperature_K / T_REF) ** 1.5 / temperature_K


def thermal_velocity_temperature_derivative(velocity_300_m_s: float, temperature_K: float) -> float:
    """``d/dT [v_300 (T/300)^0.5] = 0.5 v(T)/T``."""

    return 0.5 * float(velocity_300_m_s) * (temperature_K / T_REF) ** 0.5 / temperature_K


def mobility_temperature_derivative(mobility_300: float, temperature_K: float, exponent: float = -1.5) -> float:
    """``d/dT [mu_300 (T/300)^gamma] = gamma mu(T)/T``."""

    return exponent * mobility_300 * (temperature_K / T_REF) ** exponent / temperature_K


def radiative_coefficient_temperature_derivative(
    coefficient_300_m3_s: float, temperature_K: float, exponent: float = 0.0
) -> float:
    """``gamma B(T)/T``; exactly zero when the law is disabled."""

    if coefficient_300_m3_s == 0.0 or exponent == 0.0:
        return 0.0
    return exponent * coefficient_300_m3_s * (temperature_K / T_REF) ** exponent / temperature_K


def ion_diffusivity_temperature_derivative(
    diffusivity_300_m2_s: float, temperature_K: float, activation_energy_eV: float
) -> float:
    """``D(T) E_a q / (k_B T^2)``; exactly zero for a zero diffusivity."""

    if diffusivity_300_m2_s == 0.0:
        return 0.0
    T = temperature_K
    value = diffusivity_300_m2_s * np.exp(-activation_energy_eV * Q / K_B * (1.0 / T - 1.0 / T_REF))
    return float(value * activation_energy_eV * Q / (K_B * T * T))


def band_gap_temperature_derivative(
    temperature_K: float, varshni_alpha_eV_K: float = 0.0, varshni_beta_K: float = 0.0
) -> float:
    """``dEg/dT = -alpha T (T + 2 beta) / (T + beta)^2`` [eV K^-1].

    Zero when the shift is disabled (``alpha = 0``) or degenerate at the
    reference (``300 + beta = 0``, where the value law is the constant
    ``Eg_300``); undefined at the pole ``T + beta = 0``.
    """

    if varshni_alpha_eV_K == 0.0 or T_REF + varshni_beta_K == 0.0:
        return 0.0
    denominator = temperature_K + varshni_beta_K
    if denominator == 0.0:
        raise TemperatureCapabilityError("the Varshni band gap has no derivative at T = -beta")
    return -varshni_alpha_eV_K * temperature_K * (temperature_K + 2.0 * varshni_beta_K) / (denominator * denominator)


def intrinsic_density_temperature_partials(
    ni_300_m3: float,
    band_gap_eV: float,
    temperature_K: float,
    conduction_dos_300_m3: float | None = None,
    valence_dos_300_m3: float | None = None,
) -> tuple[float, float, float]:
    """``(ni, dni/dT|_Eg, dni/dEg|_T)`` of the formula branch at ``T``.

    At exactly 300 K the formula branch is evaluated (its one-sided limit),
    which differs from the declared ``ni_300`` by the reference jump.
    """

    T = temperature_K
    if conduction_dos_300_m3 is not None and valence_dos_300_m3 is not None:
        ratio = T / T_REF
        Nc_T = conduction_dos_300_m3 * ratio**1.5
        Nv_T = valence_dos_300_m3 * ratio**1.5
        ni = float(np.sqrt(Nc_T * Nv_T) * np.exp(-band_gap_eV * Q / (2.0 * K_B * T)))
        d_eg = -ni * Q / (2.0 * K_B * T)
    else:
        ni = float(ni_300_m3 * (T / T_REF) ** 1.5 * np.exp(-band_gap_eV * Q / (2.0 * K_B) * (1.0 / T - 1.0 / T_REF)))
        d_eg = -ni * Q / (2.0 * K_B) * (1.0 / T - 1.0 / T_REF)
    d_t = ni * (1.5 / T + band_gap_eV * Q / (2.0 * K_B * T * T))
    return ni, d_t, d_eg


def intrinsic_density_reference_jump(
    ni_300_m3: float,
    band_gap_300_eV: float,
    conduction_dos_300_m3: float | None = None,
    valence_dos_300_m3: float | None = None,
) -> float:
    """Formula value at 300 K minus the declared ``ni_300`` (0 when consistent)."""

    formula, _, _ = intrinsic_density_temperature_partials(
        ni_300_m3, band_gap_300_eV, T_REF, conduction_dos_300_m3, valence_dos_300_m3
    )
    return formula - float(intrinsic_density_at_temperature(ni_300_m3, band_gap_300_eV, T_REF))


def require_continuous_reference(
    ni_300_m3: float,
    band_gap_300_eV: float,
    conduction_dos_300_m3: float | None = None,
    valence_dos_300_m3: float | None = None,
    *,
    relative_tolerance: float = 1.0e-12,
) -> None:
    """Refuse a 300 K temperature tangent across a discontinuous ``ni(T)``.

    The default tolerance only absorbs the round-off of evaluating the DOS
    formula; a declared ``ni_300`` that differs from it by more is a jump.
    """

    jump = intrinsic_density_reference_jump(ni_300_m3, band_gap_300_eV, conduction_dos_300_m3, valence_dos_300_m3)
    if abs(jump) > relative_tolerance * abs(ni_300_m3):
        raise TemperatureCapabilityError(
            f"ni(T) is discontinuous at 300 K (declared ni_300={ni_300_m3!r}, DOS formula "
            f"differs by {jump!r} m^-3); its temperature derivative at the reference is undefined"
        )


def band_gap_at_reference(
    band_gap_eV: float, temperature_K: float, varshni_alpha_eV_K: float = 0.0, varshni_beta_K: float = 0.0
) -> float:
    """Inverse Varshni map: the 300 K gap whose shift gives ``band_gap_eV`` at ``T``."""

    T = temperature_K
    if T == T_REF or varshni_alpha_eV_K == 0.0:
        return band_gap_eV
    denom_T = T + varshni_beta_K
    denom_ref = T_REF + varshni_beta_K
    if denom_T == 0.0 or denom_ref == 0.0:
        return band_gap_eV
    return float(band_gap_eV + varshni_alpha_eV_K * (T * T / denom_T - T_REF * T_REF / denom_ref))


def grading_position(
    y: np.ndarray,
    thickness_m: float,
    profile: str = "linear",
    char_length_m: float | None = None,
    direction: str = "front_to_back",
) -> np.ndarray:
    """Inverse of ``grading_coordinate``: the local position [m] of composition ``y``."""

    composition = np.asarray(y, dtype=float)
    if not np.all(np.isfinite(composition)) or np.any(composition < 0.0) or np.any(composition > 1.0):
        raise ValueError("composition must lie in [0, 1]")
    u = 1.0 - composition if direction == "back_to_front" else composition
    if profile == "parabolic":
        return np.sqrt(u) * thickness_m
    if profile == "exponential" and char_length_m is not None and char_length_m > 0.0:
        ratio = thickness_m / char_length_m
        denom = -np.expm1(-ratio)
        if denom != 0.0:
            # 1-u+u*exp(-ratio) retains the exact back endpoint even when
            # 1-exp(-ratio) rounds to one. log1p is more accurate near u=0.
            with np.errstate(divide="ignore"):
                if direction == "back_to_front":
                    # Preserve a small supplied y; forming 1-y first can round
                    # to one and lose a physically meaningful interior point.
                    log_remaining = np.logaddexp(np.log(composition), np.log1p(-composition) - ratio)
                else:
                    log_remaining = np.logaddexp(np.log1p(-composition), np.log(composition) - ratio)
                log_remaining = np.where(u < 0.5, np.log1p(-u * denom), log_remaining)
            position = -char_length_m * log_remaining
            endpoint = composition == (0.0 if direction == "back_to_front" else 1.0)
            return np.where(endpoint, thickness_m, position)
    return u * thickness_m


def grading_coordinate_slope(
    x_local_m: np.ndarray,
    thickness_m: float,
    profile: str = "linear",
    char_length_m: float | None = None,
    direction: str = "front_to_back",
) -> np.ndarray:
    """``dy/dx`` [m^-1] inside the layer (the clip to [0, 1] is inactive there)."""

    x = np.asarray(x_local_m, dtype=float)
    if profile == "parabolic":
        slope = 2.0 * x / (thickness_m * thickness_m)
    elif profile == "exponential" and char_length_m is not None and char_length_m > 0.0 and (
        1.0 - np.exp(-thickness_m / char_length_m)
    ) != 0.0:
        slope = np.exp(-x / char_length_m) / (char_length_m * (1.0 - np.exp(-thickness_m / char_length_m)))
    else:
        slope = np.full_like(x, 1.0 / thickness_m)
    return -slope if direction == "back_to_front" else slope


def band_gap_composition_derivative(
    y: np.ndarray, band_gap_front_eV: float, band_gap_back_eV: float, bowing_eV: float = 0.0
) -> np.ndarray:
    """``dEg/dy = Eg_B - Eg_A - b (1 - 2y)``; exactly zero on a flat grade."""

    composition = np.asarray(y, dtype=float)
    if band_gap_front_eV == band_gap_back_eV and bowing_eV == 0.0:
        return np.zeros_like(composition)
    return (band_gap_back_eV - band_gap_front_eV) - bowing_eV * (1.0 - 2.0 * composition)


def affinity_composition_derivative(y: np.ndarray, chi_front_eV: float, chi_back_eV: float) -> np.ndarray:
    """``dchi/dy = chi_B - chi_A`` (Vegard)."""

    return np.full_like(np.asarray(y, dtype=float), float(chi_back_eV) - float(chi_front_eV))


@dataclass(frozen=True)
class BindingTemperatureSensitivity:
    """Node-wise ``d(parameter)/dT`` of one binding at its temperature [unit K^-1].

    All zero, with ``temperature_dependent=False``, when the binding was
    compiled with temperature scaling off (every law is then frozen at 300 K).
    """

    temperature_K: float
    temperature_dependent: bool
    thermal_voltage_V_K: float
    physical_band_gap_eV_K: np.ndarray
    physical_electron_affinity_eV_K: np.ndarray
    transport_band_gap_eV_K: np.ndarray
    transport_electron_affinity_eV_K: np.ndarray
    intrinsic_product_m6_K: np.ndarray
    srh_n1_m3_K: np.ndarray
    srh_p1_m3_K: np.ndarray
    radiative_coefficient_m3_s_K: np.ndarray
    electron_mobility_K: np.ndarray
    hole_mobility_K: np.ndarray
    electron_diffusivity_K: np.ndarray
    hole_diffusivity_K: np.ndarray
    ion_diffusivity_K: np.ndarray
    negative_ion_diffusivity_K: np.ndarray
    conduction_dos_m3_K: np.ndarray
    valence_dos_m3_K: np.ndarray
    thermal_velocity_K: np.ndarray
    time_domain_qualified: bool = False


def binding_temperature_sensitivity(binding: ParameterBinding) -> BindingTemperatureSensitivity:
    """Analytic ``d/dT`` of every temperature-dependent node parameter.

    Refused at exactly 300 K for a layer whose ``ni(T)`` jumps there.
    """

    N = binding.node_count
    zeros = np.zeros(N)
    names = (
        "physical_band_gap_eV_K", "physical_electron_affinity_eV_K", "transport_band_gap_eV_K",
        "transport_electron_affinity_eV_K", "intrinsic_product_m6_K", "srh_n1_m3_K", "srh_p1_m3_K",
        "radiative_coefficient_m3_s_K", "electron_mobility_K", "hole_mobility_K", "electron_diffusivity_K",
        "hole_diffusivity_K", "ion_diffusivity_K", "negative_ion_diffusivity_K", "conduction_dos_m3_K",
        "valence_dos_m3_K", "thermal_velocity_K",
    )
    if not binding.temperature_scaling:
        frozen: dict[str, Any] = {name: zeros.copy() for name in names}
        return BindingTemperatureSensitivity(binding.temperature_K, False, 0.0, **frozen)
    T = binding.temperature_K
    V_T = binding.thermal_voltage_V
    dV_T = thermal_voltage_temperature_derivative()
    out: dict[str, Any] = {name: np.zeros(N) for name in names}
    band = binding.band_structure
    for layer, mask, state in zip(binding.layers, binding.layer_masks, binding.layer_states):
        laws = layer.temperature_laws
        if T == T_REF:
            require_continuous_reference(
                layer.intrinsic_density_m3, layer.band_gap_eV, layer.conduction_dos_300_m3, layer.valence_dos_300_m3
            )
        # One Varshni law shifts both grading endpoints alike and narrowing
        # has no temperature law, so dEg/dT is the front-endpoint slope on
        # every node of the layer and Eg(x) - Eg_T is temperature independent.
        dEg_front = band_gap_temperature_derivative(T, laws.varshni_alpha_eV_K, laws.varshni_beta_K)
        out["physical_band_gap_eV_K"][mask] = dEg_front
        offset_eV = binding.band_structure.physical_band_gap_eV[mask] - state.band_gap_front_eV

        ni_T, dni_dT, dni_dEg = intrinsic_density_temperature_partials(
            layer.intrinsic_density_m3, state.band_gap_front_eV, T,
            layer.conduction_dos_300_m3, layer.valence_dos_300_m3,
        )
        dni_total = dni_dT + dni_dEg * dEg_front
        ni_sq = binding.intrinsic_product_m6[mask]
        # ln ni^2 = 2 ln ni_T - (Eg(x) - Eg_T)/V_T [graded] + dEg_BGN/V_T [narrowed, ungraded]
        d_ln_ni_sq = np.full(ni_sq.shape, 2.0 * dni_total / ni_T if ni_T != 0.0 else 0.0)
        if state.graded:
            d_ln_ni_sq = d_ln_ni_sq + offset_eV * dV_T / (V_T * V_T)
        elif state.narrowed:
            d_ln_ni_sq = d_ln_ni_sq - binding.band_gap_narrowing_eV[mask] * dV_T / (V_T * V_T)
        out["intrinsic_product_m6_K"][mask] = ni_sq * d_ln_ni_sq

        d_ln_scale = dni_total / ni_T if (T != T_REF and layer.intrinsic_density_m3 > 0.0 and ni_T != 0.0) else 0.0
        if T == T_REF and layer.intrinsic_density_m3 > 0.0:
            d_ln_scale = dni_total / layer.intrinsic_density_m3
        d_ln_ref = np.full(int(np.sum(mask)), d_ln_scale)
        if state.graded:
            d_ln_ref = d_ln_ref + 0.5 * offset_eV * dV_T / (V_T * V_T)
        elif state.narrowed:
            d_ln_ref = d_ln_ref - 0.5 * binding.band_gap_narrowing_eV[mask] * dV_T / (V_T * V_T)
        out["srh_n1_m3_K"][mask] = binding.srh_n1_m3[mask] * d_ln_ref
        out["srh_p1_m3_K"][mask] = binding.srh_p1_m3[mask] * d_ln_ref

        out["radiative_coefficient_m3_s_K"][mask] = radiative_coefficient_temperature_derivative(
            layer.radiative_coefficient_m3_s, T, laws.radiative_exponent
        )
        dmu_n = mobility_temperature_derivative(layer.electron_mobility_m2_V_s, T, laws.mobility_exponent)
        dmu_p = mobility_temperature_derivative(layer.hole_mobility_m2_V_s, T, laws.mobility_exponent)
        out["electron_mobility_K"][mask] = dmu_n
        out["hole_mobility_K"][mask] = dmu_p
        out["electron_diffusivity_K"][mask] = dmu_n * V_T + state.electron_mobility_m2_V_s * dV_T
        out["hole_diffusivity_K"][mask] = dmu_p * V_T + state.hole_mobility_m2_V_s * dV_T
        out["ion_diffusivity_K"][mask] = ion_diffusivity_temperature_derivative(
            layer.ion_diffusivity_m2_s, T, laws.ion_activation_energy_eV
        )
        if binding.dual_ions:
            out["negative_ion_diffusivity_K"][mask] = ion_diffusivity_temperature_derivative(
                layer.negative_ion_diffusivity_m2_s, T, laws.ion_activation_energy_eV
            )
        out["conduction_dos_m3_K"][mask] = (
            band_edge_dos_temperature_derivative(layer.conduction_dos_300_m3, T) if layer.conduction_dos_300_m3 else np.nan
        )
        out["valence_dos_m3_K"][mask] = (
            band_edge_dos_temperature_derivative(layer.valence_dos_300_m3, T) if layer.valence_dos_300_m3 else np.nan
        )
        out["thermal_velocity_K"][mask] = thermal_velocity_temperature_derivative(layer.thermal_velocity_m_s, T)

    # The narrowing (eV) and the electron affinity have no temperature law.
    fold_chi = band.transport_electron_affinity_eV - band.physical_electron_affinity_eV
    fold_gap = band.transport_band_gap_eV - band.physical_band_gap_eV
    out["transport_electron_affinity_eV_K"] = out["physical_electron_affinity_eV_K"] + fold_chi / T
    out["transport_band_gap_eV_K"] = out["physical_band_gap_eV_K"] + fold_gap / T
    for array in out.values():
        array.setflags(write=False)
    return BindingTemperatureSensitivity(T, True, dV_T, **out)

