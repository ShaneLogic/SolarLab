"""Temperature laws of the resolved material parameters (M class).

Every law takes the parameter's value at the reference temperature
``T_REF = 300 K`` and returns its value at ``T``. The arithmetic, including
the reference-temperature short circuits and the use of NumPy versus Python
scalar operations, is the historical one, so a migrated consumer reproduces
the historical binary64 values exactly:

* effective DOS ``N(T) = N_300 (T/300)^(3/2)``
* intrinsic density: with both DOS, ``sqrt(Nc(T) Nv(T)) exp(-Eg q / 2kT)``;
  otherwise ``ni_300 (T/300)^(3/2) exp(-Eg q / 2k (1/T - 1/300))``
* mobility ``mu_300 (T/300)^gamma`` (default ``gamma = -1.5``)
* ion diffusivity ``D_300 exp(-E_a q/k (1/T - 1/300))`` (Arrhenius)
* radiative coefficient ``B_300 (T/300)^gamma`` (``gamma = 0`` disables it)
* band gap, Varshni referenced to 300 K:
  ``Eg(T) = Eg_300 - alpha [T^2/(T+beta) - 300^2/(300+beta)]`` (``alpha = 0``
  disables it); the shift moves the valence edge only -- the electron
  affinity is not temperature dependent in the historical engine
* thermal velocity ``v_300 (T/300)^(1/2)``
* SRH reference densities ``n1, p1`` scaled by ``ni(T)/ni(300)`` (trap level
  fixed relative to midgap), so ``n1 p1 = ni(T)^2`` follows from a
  mass-action-consistent 300 K input.

Reference-temperature semantics, kept on purpose: at exactly ``T_REF`` every
law returns its input unchanged. For the intrinsic density with both DOS this
is the DECLARED ``ni_300``, which need not equal the DOS formula at 300 K; the
law is then discontinuous at ``T_REF`` (see the temperature tangents).

Terms that remain calibrated at 300 K are listed in
:data:`REFERENCE_CALIBRATED_TERMS`. They keep their 300 K values at every
temperature; giving any of them a temperature law is a new, separately
reviewed law, never an incidental change.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from solarlab.physics.statistics.thermal import (
    BOLTZMANN_CONSTANT_J_K as K_B,
    ELEMENTARY_CHARGE_C as Q,
    REFERENCE_TEMPERATURE_K,
    REFERENCE_THERMAL_VOLTAGE_V,
    thermal_voltage,
)

__all__ = [
    "REFERENCE_CALIBRATED_TERMS",
    "ReferenceCalibratedTerm",
    "T_REF",
    "TEMPERATURE_LAWS",
    "band_edge_dos_at_temperature",
    "band_gap_at_temperature",
    "device_temperature",
    "intrinsic_density_at_temperature",
    "ion_diffusivity_at_temperature",
    "mobility_at_temperature",
    "radiative_coefficient_at_temperature",
    "srh_reference_scale",
    "thermal_velocity_at_temperature",
]

T_REF = REFERENCE_TEMPERATURE_K

#: Stable identities of the temperature laws a parameter binding applies.
#: A binding records this table in its identity; a new or changed law gets a
#: new identity and its own review.
TEMPERATURE_LAWS: tuple[tuple[str, str], ...] = (
    ("band_edge_dos", "dos_power_three_halves_v1"),
    ("intrinsic_density", "intrinsic_density_dos_or_scaled_v1"),
    ("mobility", "mobility_power_law_v1"),
    ("ion_diffusivity", "ion_diffusivity_arrhenius_v1"),
    ("radiative_coefficient", "radiative_power_law_v1"),
    ("band_gap", "varshni_referenced_to_300k_v1"),
    ("thermal_velocity", "thermal_velocity_square_root_v1"),
    ("srh_reference_densities", "srh_reference_midgap_ratio_v1"),
)


@dataclass(frozen=True, slots=True)
class ReferenceCalibratedTerm:
    """A historical term that keeps its 300 K value at every temperature."""

    term: str
    owner: str
    semantics: str


REFERENCE_CALIBRATED_TERMS: tuple[ReferenceCalibratedTerm, ...] = (
    ReferenceCalibratedTerm(
        "interface_plane_thermal_velocity",
        "P04-09",
        "interface_plane._DEFAULT_V_TH_MS and _S_EFF have no T^(1/2) law",
    ),
    ReferenceCalibratedTerm(
        "qss_interface_thermal_velocity", "P04-09", "mol._QSS_V_TH_MS is a 300 K constant"
    ),
    ReferenceCalibratedTerm(
        "steady_state_interface_state_thermal_velocity",
        "P04-09",
        "steady_state._IFACE_STATE_V_TH is a 300 K constant",
    ),
    ReferenceCalibratedTerm(
        "interface_defect_reference_densities",
        "P04-09/P04-10",
        "per-side interface-defect n1/p1 keep a 300 K DOS prefactor",
    ),
)


def device_temperature(temperature_K: float, *, temperature_scaling: bool) -> tuple[float, float]:
    """``(T, V_T)`` the material build evaluates at.

    With temperature scaling disabled the historical engine evaluates every
    law at 300 K with its module constant ``V_T``, whatever the declared
    device temperature; that tier choice is explicit here, not a silent
    freeze.
    """

    if temperature_scaling:
        return temperature_K, thermal_voltage(temperature_K)
    return T_REF, REFERENCE_THERMAL_VOLTAGE_V


def intrinsic_density_at_temperature(
    ni_300_m3: float,
    band_gap_eV: float,
    temperature_K: float,
    conduction_dos_300_m3: float | None = None,
    valence_dos_300_m3: float | None = None,
) -> float:
    """Intrinsic carrier density [m^-3] at ``T`` (historical ``ni_at_T``).

    ``band_gap_eV`` is the gap at ``T`` (already Varshni-shifted by the caller).
    """

    T = temperature_K
    if T == T_REF:
        return ni_300_m3
    if conduction_dos_300_m3 is not None and valence_dos_300_m3 is not None:
        ratio = T / T_REF
        Nc_T = conduction_dos_300_m3 * ratio**1.5
        Nv_T = valence_dos_300_m3 * ratio**1.5
        return float(np.sqrt(Nc_T * Nv_T) * np.exp(-band_gap_eV * Q / (2.0 * K_B * T)))
    ratio = T / T_REF
    return float(
        ni_300_m3 * ratio**1.5 * np.exp(-band_gap_eV * Q / (2.0 * K_B) * (1.0 / T - 1.0 / T_REF))
    )


def mobility_at_temperature(mobility_300: float, temperature_K: float, exponent: float = -1.5) -> float:
    """Power-law mobility ``mu_300 (T/300)^gamma`` [m^2 V^-1 s^-1]."""

    T = temperature_K
    if T == T_REF:
        return mobility_300
    return float(mobility_300 * (T / T_REF) ** exponent)


def ion_diffusivity_at_temperature(
    diffusivity_300_m2_s: float, temperature_K: float, activation_energy_eV: float
) -> float:
    """Arrhenius ion diffusivity [m^2 s^-1]; a zero diffusivity stays zero."""

    T = temperature_K
    if T == T_REF or diffusivity_300_m2_s == 0.0:
        return diffusivity_300_m2_s
    return float(
        diffusivity_300_m2_s * np.exp(-activation_energy_eV * Q / K_B * (1.0 / T - 1.0 / T_REF))
    )


def radiative_coefficient_at_temperature(
    coefficient_300_m3_s: float, temperature_K: float, exponent: float = 0.0
) -> float:
    """Power-law band-to-band radiative coefficient [m^3 s^-1]."""

    T = temperature_K
    if T == T_REF or coefficient_300_m3_s == 0.0 or exponent == 0.0:
        return coefficient_300_m3_s
    return float(coefficient_300_m3_s * (T / T_REF) ** exponent)


def band_gap_at_temperature(
    band_gap_300_eV: float,
    temperature_K: float,
    varshni_alpha_eV_K: float = 0.0,
    varshni_beta_K: float = 0.0,
) -> float:
    """Varshni band gap referenced to 300 K [eV].

    A vanishing denominator (``T + beta = 0`` or ``300 + beta = 0``) returns
    the reference gap, as historically, instead of dividing by zero.
    """

    T = temperature_K
    if T == T_REF or varshni_alpha_eV_K == 0.0:
        return band_gap_300_eV
    denom_T = T + varshni_beta_K
    denom_ref = T_REF + varshni_beta_K
    if denom_T == 0.0 or denom_ref == 0.0:
        return band_gap_300_eV
    shift = varshni_alpha_eV_K * (T * T / denom_T - T_REF * T_REF / denom_ref)
    return float(band_gap_300_eV - shift)


def band_edge_dos_at_temperature(dos_300_m3: float, temperature_K: float) -> float:
    """Effective band-edge DOS ``N_300 (T/300)^(3/2)`` [m^-3]."""

    scale = 1.0
    if temperature_K != T_REF:
        scale = (temperature_K / T_REF) ** 1.5
    return float(dos_300_m3) * scale


def thermal_velocity_at_temperature(velocity_300_m_s: float, temperature_K: float) -> float:
    """Carrier thermal velocity ``v_300 (T/300)^(1/2)`` [m s^-1]."""

    scale = (temperature_K / T_REF) ** 0.5 if temperature_K != T_REF else 1.0
    return float(velocity_300_m_s) * scale


def srh_reference_scale(intrinsic_density_m3: float, ni_300_m3: float, temperature_K: float) -> float:
    """Factor ``ni(T)/ni(300)`` applied to the 300 K SRH reference densities."""

    if temperature_K != T_REF and ni_300_m3 > 0.0:
        return intrinsic_density_m3 / ni_300_m3
    return 1.0
