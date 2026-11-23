"""Thermionic emission constants, Fermi-Dirac supply integrals, and flux kernels (P04-09).

Provides:
* CODATA 2018 free-electron Richardson constant ``A_STAR_FREE_ELECTRON_A_M2_K2``
  and DOS-consistent effective-mass / Richardson helpers ``effective_mass_from_dos``,
  ``richardson_from_dos``, ``emission_velocity``, ``is_free_electron_default``,
  and ``resolve_layer_richardson`` (retaining the validated 300 K / existing DOS
  convention for ``N``-class parity; prospective TE supply corrections belong to
  ``P08-03``).
* Fermi-Dirac integrals ``fermi_dirac_zero`` (``F_0``) and ``fermi_dirac_one``
  (``F_1``) with exact derivative relation ``d F_1 / d eta = F_0``.
* Direction-preserving 1D thermionic emission particle-flux bound and smooth /
  hard Scharfetter-Gummel flux capping ``cap_particle_flux_thermionic`` with
  exact analytical derivatives wrt ``(sg_flux, n_left, n_right)``.
* Two-sided cross-interface transport kernels ``fermi_dirac_richardson_flux``,
  ``fermi_richardson_flux``, and ``scaps_thermionic_flux`` with exact analytical
  derivatives wrt trace log-densities and trace potentials.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from typing import Final, Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.special import expit, spence

from solarlab.contracts.units import ELEMENTARY_CHARGE_C
from solarlab.physics.statistics.fermi_dirac import (
    FERMI_DIRAC_HALF_TABLE_ETA_MAX,
    fermi_dirac_half_log_derivative,
    fermi_dirac_half_table,
    inverse_fermi_dirac_half,
)
from solarlab.physics.statistics.thermal import (
    BOLTZMANN_CONSTANT_J_K,
    REFERENCE_TEMPERATURE_K,
    thermal_voltage,
)

__all__ = [
    "A_STAR_FREE_ELECTRON_A_M2_K2",
    "ELECTRON_MASS_KG",
    "LEGACY_UNNORMALIZED_THERMIONIC_V1",
    "PLANCK_CONSTANT_J_S",
    "RECIPROCAL_PHYSICAL_THERMIONIC_V2",
    "ReciprocalThermionicCompositeEvaluation",
    "ReciprocalThermionicEvaluation",
    "SCAPS_EFFECTIVE_VELOCITY_THERMIONIC_V1",
    "TE_BARRIER_THRESHOLD_EV",
    "THERMIONIC_MODEL_VERSIONS",
    "ThermionicCapEvaluation",
    "ThermionicCarrierKind",
    "ThermionicCombinationMode",
    "ThermionicModelVersion",
    "ThermionicStatisticsKind",
    "TwoSidedCrossFluxEvaluation",
    "cap_particle_flux_thermionic",
    "combine_sg_and_reciprocal_thermionic_flux",
    "effective_mass_from_dos",
    "emission_velocity",
    "fermi_dirac_one",
    "fermi_dirac_richardson_flux",
    "fermi_dirac_zero",
    "fermi_richardson_flux",
    "is_free_electron_default",
    "physical_barrier_steps_eV",
    "physical_carrier_barrier_step_eV",
    "reciprocal_richardson",
    "reciprocal_thermionic_flux_m2_s",
    "resolve_layer_richardson",
    "richardson_from_dos",
    "scaps_thermionic_flux",
    "thermionic_emission_bound_m2_s",
]

Q: float = ELEMENTARY_CHARGE_C
K_B: float = BOLTZMANN_CONSTANT_J_K
PLANCK_CONSTANT_J_S: float = 6.62607015e-34
ELECTRON_MASS_KG: float = 9.1093837015e-31
A_STAR_FREE_ELECTRON_A_M2_K2: float = (
    4.0 * math.pi * Q * ELECTRON_MASS_KG * K_B**2 / PLANCK_CONSTANT_J_S**3
)
TE_BARRIER_THRESHOLD_EV: float = 0.05

LEGACY_UNNORMALIZED_THERMIONIC_V1: Final = "legacy_unnormalized_thermionic_v1"
RECIPROCAL_PHYSICAL_THERMIONIC_V2: Final = "reciprocal_physical_thermionic_v2"
SCAPS_EFFECTIVE_VELOCITY_THERMIONIC_V1: Final = "scaps_effective_velocity_thermionic_v1"
THERMIONIC_MODEL_VERSIONS = (
    LEGACY_UNNORMALIZED_THERMIONIC_V1,
    RECIPROCAL_PHYSICAL_THERMIONIC_V2,
    SCAPS_EFFECTIVE_VELOCITY_THERMIONIC_V1,
)

TERegularizationMode: TypeAlias = Literal["legacy_logistic", "compact_support"]
ThermionicModelVersion: TypeAlias = Literal[
    "legacy_unnormalized_thermionic_v1",
    "reciprocal_physical_thermionic_v2",
    "scaps_effective_velocity_thermionic_v1",
]
ThermionicCarrierKind: TypeAlias = Literal["electron", "hole"]
ThermionicStatisticsKind: TypeAlias = Literal["maxwell_boltzmann", "fermi_dirac"]
ThermionicCombinationMode: TypeAlias = Literal[
    "hard", "legacy_logistic", "compact_support", "harmonic_series"
]
RealArray: TypeAlias = NDArray[np.float64]


def is_free_electron_default(a_star_A_m2_K2: float, *, rel_tol: float = 1.0e-3) -> bool:
    """Return True if ``a_star_A_m2_K2`` matches the free-electron default ``~1.2017e6``."""
    return math.isclose(
        float(a_star_A_m2_K2),
        A_STAR_FREE_ELECTRON_A_M2_K2,
        rel_tol=rel_tol,
        abs_tol=0.0,
    )


def effective_mass_from_dos(dos_m3: float, temperature_K: float = REFERENCE_TEMPERATURE_K) -> float:
    """Invert ``N_C = 2 (2 pi m* k_B T / h^2)^{3/2}`` to return ``m* / m_0``."""
    n_dos = float(dos_m3)
    temp = float(temperature_K)
    if not (math.isfinite(n_dos) and n_dos > 0.0):
        raise ValueError("dos_m3 must be finite and positive")
    if not (math.isfinite(temp) and temp > 0.0):
        raise ValueError("temperature_K must be finite and positive")
    prefactor = 2.0 * math.pi * ELECTRON_MASS_KG * K_B * temp / PLANCK_CONSTANT_J_S**2
    return float((0.5 * n_dos) ** (2.0 / 3.0) / prefactor)


def richardson_from_dos(dos_m3: float, temperature_K: float = REFERENCE_TEMPERATURE_K) -> float:
    """Effective Richardson constant ``A* = A_0 * (m* / m_0)`` [A m^-2 K^-2] from DOS."""
    return float(A_STAR_FREE_ELECTRON_A_M2_K2 * effective_mass_from_dos(dos_m3, temperature_K))


def emission_velocity(temperature_K: float = REFERENCE_TEMPERATURE_K, m_eff_ratio: float = 1.0) -> float:
    """Mean one-way thermal emission velocity ``sqrt(k_B T / (2 pi m*))`` [m/s]."""
    temp = float(temperature_K)
    ratio = float(m_eff_ratio)
    if not (math.isfinite(temp) and temp > 0.0):
        raise ValueError("temperature_K must be finite and positive")
    if not (math.isfinite(ratio) and ratio > 0.0):
        raise ValueError("m_eff_ratio must be finite and positive")
    return float(math.sqrt(K_B * temp / (2.0 * math.pi * ratio * ELECTRON_MASS_KG)))


def resolve_layer_richardson(
    a_star_A_m2_K2: float,
    dos_300_m3: float,
    temperature_K: float = REFERENCE_TEMPERATURE_K,
) -> float:
    """Resolve layer ``A*`` using the existing 300 K DOS convention when defaulted.

    ``a_star_A_m2_K2 == 0.0`` disables thermionic emission on that carrier and is
    preserved exactly.
    """
    a_star = float(a_star_A_m2_K2)
    if not math.isfinite(a_star) or a_star < 0.0:
        raise ValueError("a_star_A_m2_K2 must be finite and non-negative")
    if a_star == 0.0:
        return 0.0
    dos = float(dos_300_m3)
    if is_free_electron_default(a_star) and math.isfinite(dos) and dos > 0.0:
        return richardson_from_dos(dos, temperature_K)
    return a_star


def fermi_dirac_zero(eta: float) -> float:
    """Complete Fermi-Dirac integral ``F_0(eta) = ln(1 + exp(eta))``."""
    x = float(eta)
    if math.isnan(x):
        raise ValueError("eta must not be NaN")
    if x <= -700.0:
        return 0.0
    if x <= -40.0:
        return math.exp(x)
    if x <= 20.0:
        return math.log1p(math.exp(x))
    return x + math.log1p(math.exp(-x))


def fermi_dirac_one(eta: float) -> float:
    """Complete Fermi-Dirac integral ``F_1(eta) = integral_0^inf x/(1+exp(x-eta)) dx``."""
    x = float(eta)
    if math.isnan(x):
        raise ValueError("eta must not be NaN")
    if x <= -700.0:
        return 0.0
    if x < -4.0:
        exp_x = math.exp(x)
        power = exp_x
        total = 0.0
        sign = 1.0
        for k in range(1, 13):
            total += sign * power / (k * k)
            power *= exp_x
            sign = -sign
        return total
    if x > 40.0:
        return 0.5 * x * x + (math.pi * math.pi) / 6.0 - math.exp(-x)
    return float(-spence(1.0 + math.exp(x)))


def thermionic_emission_bound_m2_s(
    n_left_m3: float,
    n_right_m3: float,
    delta_barrier_eV: float,
    temperature_K: float,
    a_star_A_m2_K2: float,
    *,
    dos_norm_m3: float | None = None,
) -> tuple[float, float, float]:
    """Signed 1D thermionic particle-flux bound ``J_te / q`` [m^-2 s^-1] and partials.

    Returns ``(gamma_te, d_gamma_te_d_n_left, d_gamma_te_d_n_right)`` where
    ``Q * gamma_te`` reproduces the historical ``thermionic_emission_flux`` in
    ``A m^-2`` bit-for-bit up to the elementary-charge unit conversion.
    """
    n_l = float(n_left_m3)
    n_r = float(n_right_m3)
    d_eb = float(delta_barrier_eV)
    temp = float(temperature_K)
    a_star = float(a_star_A_m2_K2)
    if not all(math.isfinite(v) for v in (n_l, n_r, d_eb, temp, a_star)):
        raise ValueError("thermionic inputs must be finite")
    if temp <= 0.0 or a_star < 0.0:
        raise ValueError("temperature must be positive and A* non-negative")
    if a_star == 0.0:
        return 0.0, 0.0, 0.0

    vt = thermal_voltage(temp)
    bar_lr = max(d_eb, 0.0)
    bar_rl = max(-d_eb, 0.0)
    exp_lr = math.exp(max(-500.0, -bar_lr / vt))
    exp_rl = math.exp(max(-500.0, -bar_rl / vt))
    scale = (a_star * temp * temp) / Q
    if dos_norm_m3 is not None:
        dos = float(dos_norm_m3)
        if not (math.isfinite(dos) and dos > 0.0):
            raise ValueError("dos_norm_m3 must be finite and positive when supplied")
        scale /= dos
    d_left = scale * exp_lr
    d_right = -scale * exp_rl
    return scale * (n_l * exp_lr - n_r * exp_rl), d_left, d_right


@dataclass(frozen=True)
class ThermionicCapEvaluation:
    """Direction-preserving capped particle flux [m^-2 s^-1] and exact partials."""

    flux_m2_s: float
    d_flux_d_sg_flux: float
    d_flux_d_n_left_m_s: float
    d_flux_d_n_right_m_s: float


def cap_particle_flux_thermionic(
    sg_flux_m2_s: float,
    *,
    n_left_m3: float,
    n_right_m3: float,
    delta_barrier_eV: float,
    temperature_K: float,
    a_star_A_m2_K2: float,
    te_softness: float = 0.0,
    regularization_mode: TERegularizationMode = "legacy_logistic",
    dos_norm_m3: float | None = None,
) -> ThermionicCapEvaluation:
    """Limit ``|sg_flux_m2_s|`` by the 1D thermionic particle-flux bound without sign flip."""
    sg = float(sg_flux_m2_s)
    soft = float(te_softness)
    if not (math.isfinite(sg) and math.isfinite(soft)) or soft < 0.0:
        raise ValueError("sg_flux_m2_s must be finite and te_softness non-negative")

    te_signed, d_te_dl, d_te_dr = thermionic_emission_bound_m2_s(
        n_left_m3,
        n_right_m3,
        delta_barrier_eV,
        temperature_K,
        a_star_A_m2_K2,
        dos_norm_m3=dos_norm_m3,
    )
    a_sg = abs(sg)
    a_te = abs(te_signed)
    sign_sg = math.copysign(1.0, sg)
    sign_te = 0.0 if te_signed == 0.0 else math.copysign(1.0, te_signed)

    if soft == 0.0:
        if a_sg <= a_te:
            return ThermionicCapEvaluation(sg, 1.0, 0.0, 0.0)
        flux = math.copysign(a_te, sg)
        scale_te = sign_sg * sign_te
        return ThermionicCapEvaluation(flux, 0.0, scale_te * d_te_dl, scale_te * d_te_dr)

    if regularization_mode == "legacy_logistic":
        denom = soft * (a_sg + a_te) + 1.0e-300
        t_arg = (a_sg - a_te) / denom
        if t_arg > 40.0:
            weight = 1.0
            dw_dt = 0.0
        elif t_arg < -40.0:
            weight = 0.0
            dw_dt = 0.0
        else:
            weight = 1.0 / (1.0 + math.exp(-t_arg))
            dw_dt = weight * (1.0 - weight)
        magnitude = weight * a_te + (1.0 - weight) * a_sg
        denom_sq = denom * denom
        dt_da_sg = (denom - (a_sg - a_te) * soft) / denom_sq
        dt_da_te = (-denom - (a_sg - a_te) * soft) / denom_sq
        dm_da_sg = (1.0 - weight) + (a_te - a_sg) * dw_dt * dt_da_sg
        dm_da_te = weight + (a_te - a_sg) * dw_dt * dt_da_te
    elif regularization_mode == "compact_support":
        scale = soft * (a_sg + a_te)
        if scale == 0.0:
            return ThermionicCapEvaluation(math.copysign(0.0, sg), 0.5, 0.0, 0.0)
        normalized = (a_sg - a_te) / scale
        if normalized <= -1.0:
            magnitude = a_sg
            dm_da_sg = 1.0
            dm_da_te = 0.0
        elif normalized >= 1.0:
            magnitude = a_te
            dm_da_sg = 0.0
            dm_da_te = 1.0
        else:
            u = 0.5 * (normalized + 1.0)
            weight = 6.0 * u**5 - 15.0 * u**4 + 10.0 * u**3
            dw_du = 30.0 * u**2 * (1.0 - u) ** 2
            scale_sq = scale * scale
            dn_da_sg = 2.0 * soft * a_te / scale_sq
            dn_da_te = -2.0 * soft * a_sg / scale_sq
            dw_da_sg = 0.5 * dw_du * dn_da_sg
            dw_da_te = 0.5 * dw_du * dn_da_te
            magnitude = (1.0 - weight) * a_sg + weight * a_te
            dm_da_sg = (1.0 - weight) + (a_te - a_sg) * dw_da_sg
            dm_da_te = weight + (a_te - a_sg) * dw_da_te
    else:
        raise ValueError(
            "regularization_mode must be 'legacy_logistic' or 'compact_support'"
        )

    flux = math.copysign(magnitude, sg)
    # flux = sign(sg) * M(|sg|, |te|), so d(flux)/d(sg) = dM/d(|sg|) for sg != 0,
    # and d(flux)/d(n_side) = sign(sg) * dM/d(|te|) * sign(te) * d(te)/d(n_side).
    d_flux_d_sg = dm_da_sg
    te_chain = sign_sg * dm_da_te * sign_te
    return ThermionicCapEvaluation(
        flux_m2_s=flux,
        d_flux_d_sg_flux=d_flux_d_sg,
        d_flux_d_n_left_m_s=te_chain * d_te_dl,
        d_flux_d_n_right_m_s=te_chain * d_te_dr,
    )


@dataclass(frozen=True)
class TwoSidedCrossFluxEvaluation:
    """Two-sided cross-plane particle fluxes ``[Gamma_n, Gamma_p]`` [m^-2 s^-1].

    State log-density column order is ``(log n_L, log p_L, log n_R, log p_R)``.
    Trace-potential column order is ``(phi_trace_L, phi_trace_R)``.
    """

    cross_flux_m2_s: RealArray
    d_cross_d_log_state_m2_s: RealArray
    d_cross_d_trace_potentials_m2_s_V: RealArray
    one_way_scale_m2_s: RealArray


def _inverse_fermi_dirac_half_log(log_ratio: float) -> float:
    """Invert ``F_{1/2}(eta) = exp(log_ratio)`` directly in log-ratio space."""
    lr = float(log_ratio)
    eta_grid, _table, log_table = fermi_dirac_half_table()
    if lr <= float(log_table[0]):
        return lr
    if lr <= float(log_table[-1]):
        return float(np.interp(lr, log_table, eta_grid))
    ratio = math.exp(lr)
    coeff = 4.0 / (3.0 * math.sqrt(math.pi))
    eta = max(FERMI_DIRAC_HALF_TABLE_ETA_MAX, math.exp(2.0 / 3.0 * (lr - math.log(coeff))))
    for _ in range(6):
        root = math.sqrt(eta)
        val = (
            4.0 * eta * root / (3.0 * math.sqrt(math.pi))
            + math.pi**2 / (6.0 * math.sqrt(math.pi) * root)
        )
        der = (
            2.0 * root / math.sqrt(math.pi)
            - math.pi**2 / (12.0 * math.sqrt(math.pi) * eta * root)
        )
        eta = max(FERMI_DIRAC_HALF_TABLE_ETA_MAX, eta - (val - ratio) / der)
    return eta


@lru_cache(maxsize=1)
def _excess_table() -> NDArray[np.float64]:
    """Cached ``eta_grid - log F_{1/2}(eta_grid)`` table for sawtooth-free excess interpolation."""
    eta_grid, _table, log_table = fermi_dirac_half_table()
    excess = np.asarray(eta_grid - log_table, dtype=np.float64)
    excess[0] = 0.0
    excess.setflags(write=False)
    return excess


def _eta_excess_from_log_ratio(log_ratio: float) -> tuple[float, int]:
    """Return ``(eta(lr) - lr, table_interval_index)`` without subtracting ``lr`` from ``eta``."""
    lr = float(log_ratio)
    _eta_grid, _table, log_table = fermi_dirac_half_table()
    if lr <= float(log_table[0]):
        return 0.0, -1
    if lr <= float(log_table[-1]):
        excess = _excess_table()
        idx = min(max(int(np.searchsorted(log_table, lr, side="right")) - 1, 0), len(log_table) - 2)
        t = float((lr - log_table[idx]) / (log_table[idx + 1] - log_table[idx]))
        ex = float(excess[idx] + t * (excess[idx + 1] - excess[idx]))
        return ex, idx
    return _inverse_fermi_dirac_half_log(lr) - lr, len(log_table)


def _eta_excess_and_delta(base_log_ratio: float, u_shift: float) -> tuple[float, float, float]:
    """Return ``(eta_full, eta(base) - base, eta(base + u) - eta(base))`` without sawtooth or ``u`` quantization."""
    base_lr = float(base_log_ratio)
    shift = float(u_shift)
    lr_full = base_lr + shift
    ex0, idx0 = _eta_excess_from_log_ratio(base_lr)
    if idx0 == -1 and lr_full <= float(fermi_dirac_half_table()[2][0]):
        return lr_full, 0.0, shift
    ex_full, idx1 = _eta_excess_from_log_ratio(lr_full)
    if 0 <= idx0 == idx1 < len(fermi_dirac_half_table()[2]) - 1:
        eta_grid, _table, log_table = fermi_dirac_half_table()
        slope = float((eta_grid[idx0 + 1] - eta_grid[idx0]) / (log_table[idx0 + 1] - log_table[idx0]))
        d_eta = slope * shift
        return (base_lr + ex0) + d_eta, ex0, d_eta
    d_eta = shift + (ex_full - ex0)
    return lr_full + ex_full, ex0, d_eta


def _one_way_fermi_dirac_supply(
    density_m3: float,
    dos_m3: float,
    barrier_eV: float,
    vt_V: float,
    *,
    eta_override: float | None = None,
) -> tuple[float, float, float, float]:
    if eta_override is None:
        ratio = float(density_m3) / float(dos_m3)
        eta = inverse_fermi_dirac_half(ratio)
    else:
        eta = float(eta_override)
    argument = eta - float(barrier_eV) / float(vt_V)
    supply = fermi_dirac_one(argument)
    log_slope = fermi_dirac_half_log_derivative(eta)
    if not math.isfinite(log_slope) or log_slope <= 0.0:
        raise FloatingPointError("invalid Fermi-Dirac constitutive derivative")
    f0 = fermi_dirac_zero(argument)
    return supply, f0 / log_slope, -f0 / float(vt_V), argument


def _fermi_dirac_one_difference(arg_left: float, arg_right: float, delta: float, sup_left: float, sup_right: float) -> float:
    """Evaluate ``F_1(arg_left) - F_1(arg_right)`` without catastrophic cancellation."""
    if abs(delta) <= 1.0e-3 and math.isfinite(arg_left) and math.isfinite(arg_right):
        mid = 0.5 * (arg_left + arg_right)
        s = float(expit(mid))
        return delta * (fermi_dirac_zero(mid) + (delta * delta / 24.0) * s * (1.0 - s))
    return sup_left - sup_right


def fermi_dirac_richardson_flux(
    state_m3: NDArray[np.float64],
    *,
    phi_trace_left_V: float,
    phi_trace_right_V: float,
    conduction_band_step_eV: float,
    hole_transport_step_eV: float,
    N_C_left_m3: float,
    N_C_right_m3: float,
    N_V_left_m3: float,
    N_V_right_m3: float,
    richardson_n_A_m2_K2: float,
    richardson_p_A_m2_K2: float,
    temperature_K: float,
    thermal_voltage_V: float,
    transmission: float = 1.0,
    base_log_to_dos: NDArray[np.float64] | None = None,
    u_delta: NDArray[np.float64] | None = None,
    electrochemical_steps: tuple[float | None, float | None] = (None, None),
) -> TwoSidedCrossFluxEvaluation:
    """Reciprocal 3D Fermi-Dirac Bethe/Richardson cross-plane particle flux."""
    n_l, p_l, n_r, p_r = (float(v) for v in state_m3)
    potential_jump = float(phi_trace_right_V - phi_trace_left_V)
    electron_step = float(conduction_band_step_eV) - potential_jump
    hole_step = float(hole_transport_step_eV) + potential_jump
    vt = float(thermal_voltage_V)

    eta_nl: float | None
    eta_pl: float | None
    eta_nr: float | None
    eta_pr: float | None
    if base_log_to_dos is not None and u_delta is not None:
        eta_nl, ex_nl, d_nl = _eta_excess_and_delta(float(base_log_to_dos[0]), float(u_delta[0]))
        eta_pl, ex_pl, d_pl = _eta_excess_and_delta(float(base_log_to_dos[1]), float(u_delta[1]))
        eta_nr, ex_nr, d_nr = _eta_excess_and_delta(float(base_log_to_dos[2]), float(u_delta[2]))
        eta_pr, ex_pr, d_pr = _eta_excess_and_delta(float(base_log_to_dos[3]), float(u_delta[3]))
    else:
        eta_nl = eta_pl = eta_nr = eta_pr = None
        ex_nl = ex_pl = ex_nr = ex_pr = 0.0
        d_nl = d_pl = d_nr = d_pr = 0.0

    n_l_sup, dn_l, dn_l_bar, arg_nl = _one_way_fermi_dirac_supply(
        n_l, N_C_left_m3, max(electron_step, 0.0), vt, eta_override=eta_nl
    )
    n_r_sup, dn_r, dn_r_bar, arg_nr = _one_way_fermi_dirac_supply(
        n_r, N_C_right_m3, max(-electron_step, 0.0), vt, eta_override=eta_nr
    )
    p_l_sup, dp_l, dp_l_bar, arg_pl = _one_way_fermi_dirac_supply(
        p_l, N_V_left_m3, max(hole_step, 0.0), vt, eta_override=eta_pl
    )
    p_r_sup, dp_r, dp_r_bar, arg_pr = _one_way_fermi_dirac_supply(
        p_r, N_V_right_m3, max(-hole_step, 0.0), vt, eta_override=eta_pr
    )

    ec_n, ec_p = electrochemical_steps
    if eta_nl is not None and eta_nr is not None and ec_n is not None:
        delta_arg_n = -float(ec_n) + (ex_nl - ex_nr) + (d_nl - d_nr)
    elif eta_nl is not None and eta_nr is not None:
        arg_nl0 = (eta_nl - d_nl) - max(electron_step, 0.0) / vt
        arg_nr0 = (eta_nr - d_nr) - max(-electron_step, 0.0) / vt
        delta_arg_n = (arg_nl0 - arg_nr0) + (d_nl - d_nr)
    else:
        delta_arg_n = arg_nl - arg_nr

    if eta_pl is not None and eta_pr is not None and ec_p is not None:
        delta_arg_p = -float(ec_p) + (ex_pl - ex_pr) + (d_pl - d_pr)
    elif eta_pl is not None and eta_pr is not None:
        arg_pl0 = (eta_pl - d_pl) - max(hole_step, 0.0) / vt
        arg_pr0 = (eta_pr - d_pr) - max(-hole_step, 0.0) / vt
        delta_arg_p = (arg_pl0 - arg_pr0) + (d_pl - d_pr)
    else:
        delta_arg_p = arg_pl - arg_pr

    pref_n = float(transmission) * float(richardson_n_A_m2_K2) * float(temperature_K) ** 2 / Q
    pref_p = float(transmission) * float(richardson_p_A_m2_K2) * float(temperature_K) ** 2 / Q
    cross = np.array(
        [
            pref_n * _fermi_dirac_one_difference(arg_nl, arg_nr, delta_arg_n, n_l_sup, n_r_sup),
            pref_p * _fermi_dirac_one_difference(arg_pl, arg_pr, delta_arg_p, p_l_sup, p_r_sup),
        ],
        dtype=np.float64,
    )
    d_log: NDArray[np.float64] = np.zeros((2, 4), dtype=np.float64)
    d_log[0, 0] = pref_n * dn_l
    d_log[0, 2] = -pref_n * dn_r
    d_log[1, 1] = pref_p * dp_l
    d_log[1, 3] = -pref_p * dp_r

    e_jump_deriv = 0.0
    if electron_step > 0.0:
        e_jump_deriv = pref_n * (-dn_l_bar)
    elif electron_step < 0.0:
        e_jump_deriv = pref_n * (-dn_r_bar)

    h_jump_deriv = 0.0
    if hole_step > 0.0:
        h_jump_deriv = pref_p * dp_l_bar
    elif hole_step < 0.0:
        h_jump_deriv = pref_p * dp_r_bar

    d_trace = np.array(
        [[-e_jump_deriv, e_jump_deriv], [-h_jump_deriv, h_jump_deriv]],
        dtype=np.float64,
    )
    scale = np.array(
        [pref_n * max(n_l_sup, n_r_sup), pref_p * max(p_l_sup, p_r_sup)],
        dtype=np.float64,
    )
    return TwoSidedCrossFluxEvaluation(cross, d_log, d_trace, scale)


def fermi_richardson_flux(
    state_m3: NDArray[np.float64],
    *,
    phi_trace_left_V: float,
    phi_trace_right_V: float,
    conduction_band_step_eV: float,
    hole_transport_step_eV: float,
    N_C_left_m3: float,
    N_C_right_m3: float,
    N_V_left_m3: float,
    N_V_right_m3: float,
    richardson_n_A_m2_K2: float,
    richardson_p_A_m2_K2: float,
    temperature_K: float,
    thermal_voltage_V: float,
    transmission: float = 1.0,
) -> TwoSidedCrossFluxEvaluation:
    """Boltzmann-limit reciprocal Bethe/Richardson cross-plane particle flux."""
    n_l, p_l, n_r, p_r = (float(v) for v in state_m3)
    potential_jump = float(phi_trace_right_V - phi_trace_left_V)
    electron_step = float(conduction_band_step_eV) - potential_jump
    hole_step = float(hole_transport_step_eV) + potential_jump
    vt = float(thermal_voltage_V)

    exp_n_l = math.exp(max(-500.0, -max(electron_step, 0.0) / vt))
    exp_n_r = math.exp(max(-500.0, -max(-electron_step, 0.0) / vt))
    exp_p_l = math.exp(max(-500.0, -max(hole_step, 0.0) / vt))
    exp_p_r = math.exp(max(-500.0, -max(-hole_step, 0.0) / vt))

    n_l_sup = (n_l / float(N_C_left_m3)) * exp_n_l
    n_r_sup = (n_r / float(N_C_right_m3)) * exp_n_r
    p_l_sup = (p_l / float(N_V_left_m3)) * exp_p_l
    p_r_sup = (p_r / float(N_V_right_m3)) * exp_p_r

    pref_n = float(transmission) * float(richardson_n_A_m2_K2) * float(temperature_K) ** 2 / Q
    pref_p = float(transmission) * float(richardson_p_A_m2_K2) * float(temperature_K) ** 2 / Q
    cross = np.array(
        [pref_n * (n_l_sup - n_r_sup), pref_p * (p_l_sup - p_r_sup)],
        dtype=np.float64,
    )
    d_log: NDArray[np.float64] = np.zeros((2, 4), dtype=np.float64)
    d_log[0, 0] = pref_n * n_l_sup
    d_log[0, 2] = -pref_n * n_r_sup
    d_log[1, 1] = pref_p * p_l_sup
    d_log[1, 3] = -pref_p * p_r_sup

    e_jump_deriv = 0.0
    if electron_step > 0.0:
        e_jump_deriv = pref_n * n_l_sup / vt
    elif electron_step < 0.0:
        e_jump_deriv = pref_n * n_r_sup / vt

    h_jump_deriv = 0.0
    if hole_step > 0.0:
        h_jump_deriv = -pref_p * p_l_sup / vt
    elif hole_step < 0.0:
        h_jump_deriv = -pref_p * p_r_sup / vt

    d_trace = np.array(
        [[-e_jump_deriv, e_jump_deriv], [-h_jump_deriv, h_jump_deriv]],
        dtype=np.float64,
    )
    scale = np.array(
        [pref_n * max(n_l_sup, n_r_sup), pref_p * max(p_l_sup, p_r_sup)],
        dtype=np.float64,
    )
    return TwoSidedCrossFluxEvaluation(cross, d_log, d_trace, scale)


def scaps_thermionic_flux(
    state_m3: NDArray[np.float64],
    *,
    phi_trace_left_V: float,
    phi_trace_right_V: float,
    conduction_band_step_eV: float,
    hole_transport_step_eV: float,
    N_C_left_m3: float,
    N_C_right_m3: float,
    N_V_left_m3: float,
    N_V_right_m3: float,
    richardson_n_left_A_m2_K2: float,
    richardson_n_right_A_m2_K2: float,
    richardson_p_left_A_m2_K2: float,
    richardson_p_right_A_m2_K2: float,
    temperature_K: float,
    thermal_voltage_V: float,
    transmission: float = 1.0,
) -> TwoSidedCrossFluxEvaluation:
    """SCAPS effective-velocity cross-plane particle flux ``v_eff (n_L_eff - n_R_eff)``."""
    n_l, p_l, n_r, p_r = (float(v) for v in state_m3)
    nc_l, nc_r = float(N_C_left_m3), float(N_C_right_m3)
    nv_l, nv_r = float(N_V_left_m3), float(N_V_right_m3)
    potential_jump = float(phi_trace_right_V - phi_trace_left_V)
    electron_step = float(conduction_band_step_eV) - potential_jump
    hole_step = float(hole_transport_step_eV) + potential_jump
    vt = float(thermal_voltage_V)
    temp_sq = float(temperature_K) ** 2
    trans = float(transmission)

    vn_l = float(richardson_n_left_A_m2_K2) * temp_sq / (Q * nc_l)
    vn_r = float(richardson_n_right_A_m2_K2) * temp_sq / (Q * nc_r)
    vp_l = float(richardson_p_left_A_m2_K2) * temp_sq / (Q * nv_l)
    vp_r = float(richardson_p_right_A_m2_K2) * temp_sq / (Q * nv_r)
    v_eff_n = trans * min(vn_l, vn_r * (nc_r / nc_l))
    v_eff_p = trans * min(vp_l, vp_r * (nv_r / nv_l))

    exp_n_l = math.exp(max(-500.0, -max(electron_step, 0.0) / vt))
    exp_n_r = math.exp(max(-500.0, -max(-electron_step, 0.0) / vt))
    exp_p_l = math.exp(max(-500.0, -max(hole_step, 0.0) / vt))
    exp_p_r = math.exp(max(-500.0, -max(-hole_step, 0.0) / vt))

    term_n_l = v_eff_n * n_l * exp_n_l
    term_n_r = v_eff_n * n_r * (nc_l / nc_r) * exp_n_r
    term_p_l = v_eff_p * p_l * exp_p_l
    term_p_r = v_eff_p * p_r * (nv_l / nv_r) * exp_p_r

    cross = np.array([term_n_l - term_n_r, term_p_l - term_p_r], dtype=np.float64)
    d_log: NDArray[np.float64] = np.zeros((2, 4), dtype=np.float64)
    d_log[0, 0] = term_n_l
    d_log[0, 2] = -term_n_r
    d_log[1, 1] = term_p_l
    d_log[1, 3] = -term_p_r

    e_jump_deriv = 0.0
    if electron_step > 0.0:
        e_jump_deriv = term_n_l / vt
    elif electron_step < 0.0:
        e_jump_deriv = term_n_r / vt

    h_jump_deriv = 0.0
    if hole_step > 0.0:
        h_jump_deriv = -term_p_l / vt
    elif hole_step < 0.0:
        h_jump_deriv = -term_p_r / vt

    d_trace = np.array(
        [[-e_jump_deriv, e_jump_deriv], [-h_jump_deriv, h_jump_deriv]],
        dtype=np.float64,
    )
    scale = np.array([max(term_n_l, term_n_r), max(term_p_l, term_p_r)], dtype=np.float64)
    return TwoSidedCrossFluxEvaluation(cross, d_log, d_trace, scale)


def reciprocal_richardson(
    a_star_left_A_m2_K2: float,
    a_star_right_A_m2_K2: float,
) -> float:
    """Return reciprocal Bethe/Richardson constant ``min(A*_left, A*_right)`` [A m^-2 K^-2].

    Planar transverse crystal-momentum conservation restricts thermally emitted
    carriers to the smaller transverse effective-mass disk on either side of the
    heterointerface, guaranteeing symmetry under ``left <-> right`` swap and
    eliminating spurious equilibrium emission across unequal effective masses.
    """
    left = float(a_star_left_A_m2_K2)
    right = float(a_star_right_A_m2_K2)
    if not (math.isfinite(left) and math.isfinite(right)) or left < 0.0 or right < 0.0:
        raise ValueError("Richardson constants must be finite and non-negative")
    return min(left, right)


def physical_barrier_steps_eV(
    *,
    phi_left_V: float,
    phi_right_V: float,
    chi_phys_left_eV: float,
    chi_phys_right_eV: float,
    eg_phys_left_eV: float,
    eg_phys_right_eV: float,
) -> tuple[float, float]:
    """Return ``(electron_step_eV, hole_step_eV)`` from unfolded physical band edges.

    Uses true physical electron affinities and band gaps (free of DOS-folded
    Scharfetter-Gummel drift potentials) together with the electrostatic step
    ``phi_right_V - phi_left_V``. A positive return value denotes an upward
    potential-energy barrier for the respective carrier moving ``left -> right``.
    """
    phi_l = float(phi_left_V)
    phi_r = float(phi_right_V)
    chi_l = float(chi_phys_left_eV)
    chi_r = float(chi_phys_right_eV)
    eg_l = float(eg_phys_left_eV)
    eg_r = float(eg_phys_right_eV)
    if not all(math.isfinite(v) for v in (phi_l, phi_r, chi_l, chi_r, eg_l, eg_r)):
        raise ValueError("potentials and physical band parameters must be finite")
    if eg_l <= 0.0 or eg_r <= 0.0:
        raise ValueError("physical band gaps must be positive")
    dphi = phi_r - phi_l
    electron_step = (chi_l - chi_r) - dphi
    hole_step = ((chi_r + eg_r) - (chi_l + eg_l)) + dphi
    return electron_step, hole_step


def physical_carrier_barrier_step_eV(
    carrier: ThermionicCarrierKind,
    *,
    phi_left_V: float,
    phi_right_V: float,
    band_offset_eV: float,
) -> tuple[float, float, float]:
    """Return ``(barrier_step_eV, d_step_d_phi_left, d_step_d_phi_right)``.

    ``band_offset_eV`` is the electrostatic-free physical band-edge offset
    oriented for ``carrier`` moving ``left -> right``:
    * ``carrier == "electron"``: ``chi_phys_left_eV - chi_phys_right_eV``;
    * ``carrier == "hole"``: ``(chi_phys_right_eV + eg_phys_right_eV) - (chi_phys_left_eV + eg_phys_left_eV)``.
    """
    phi_l = float(phi_left_V)
    phi_r = float(phi_right_V)
    offset = float(band_offset_eV)
    if not (math.isfinite(phi_l) and math.isfinite(phi_r) and math.isfinite(offset)):
        raise ValueError("phi_left_V, phi_right_V, and band_offset_eV must be finite")
    dphi = phi_r - phi_l
    if carrier == "electron":
        return offset - dphi, 1.0, -1.0
    if carrier == "hole":
        return offset + dphi, -1.0, 1.0
    raise ValueError(f"unknown carrier {carrier!r}; expected 'electron' or 'hole'")


def _symmetric_log_ratio(left: float, right: float) -> float:
    """Return ``ln(right / left)`` with 0-ULP swap antisymmetry and no near-unity cancellation."""
    diff = right - left
    if diff == 0.0:
        return 0.0
    scale = max(left, right)
    z = (diff / scale) / ((right / scale) + (left / scale))
    if abs(z) <= 0.5:
        return 2.0 * math.atanh(z)
    return math.log(right) - math.log(left)


def _boltzmann_over_barrier_supply(activity: float, reduced_barrier: float) -> float:
    """Evaluate ``activity * exp(-reduced_barrier)`` without premature factor underflow."""
    if reduced_barrier == 0.0:
        return activity
    if reduced_barrier <= 650.0:
        return activity * math.exp(-reduced_barrier)
    log_prod = math.log(activity) - reduced_barrier
    if log_prod < -745.0:
        return 0.0
    return math.exp(log_prod)


@dataclass(frozen=True, slots=True)
class ReciprocalThermionicEvaluation:
    """Constitutive evaluation of versioned 1D heterointerface thermionic emission (P08-03)."""

    model_version: ThermionicModelVersion
    carrier: ThermionicCarrierKind
    statistics: ThermionicStatisticsKind
    flux_m2_s: float
    current_A_m2: float
    one_way_scale_m2_s: float
    barrier_step_eV: float
    electrochemical_step: float
    effective_richardson_A_m2_K2: float
    transmission: float
    d_flux_d_phi_left_m2_s_V: float
    d_flux_d_phi_right_m2_s_V: float
    d_flux_d_c_left_m_s: float
    d_flux_d_c_right_m_s: float
    d_flux_d_log_c_left_m2_s: float
    d_flux_d_log_c_right_m2_s: float
    left_barrier_weight: float = 0.0
    right_barrier_weight: float = 0.0


def reciprocal_thermionic_flux_m2_s(
    c_left_m3: float,
    c_right_m3: float,
    *,
    carrier: ThermionicCarrierKind,
    phi_left_V: float,
    phi_right_V: float,
    band_offset_eV: float,
    temperature_K: float,
    a_star_left_A_m2_K2: float,
    a_star_right_A_m2_K2: float | None = None,
    dos_left_m3: float | None = None,
    dos_right_m3: float | None = None,
    transmission: float = 1.0,
    statistics: ThermionicStatisticsKind = "maxwell_boltzmann",
    model_version: ThermionicModelVersion = RECIPROCAL_PHYSICAL_THERMIONIC_V2,
    thermal_voltage_V: float | None = None,
    electrochemical_step: float | None = None,
    barrier_corner_kbt: float = 0.25,
) -> ReciprocalThermionicEvaluation:
    """Evaluate versioned thermionic particle flux [m^-2 s^-1] and analytical Jacobians.

    Under ``RECIPROCAL_PHYSICAL_THERMIONIC_V2``, separates:
    1. True physical electrostatic + band-edge barrier ``barrier_step_eV``;
    2. Per-side DOS activities ``(c_left / dos_left, c_right / dos_right)``;
    3. Reciprocal Bethe/Richardson constant ``min(a_star_left, a_star_right)``;
    4. Explicit transmission factor ``transmission`` in ``[0, 1]``;
    5. Cancellation-free electrochemical difference ``electrochemical_step``.
    """
    c_l = float(c_left_m3)
    c_r = float(c_right_m3)
    temp = float(temperature_K)
    trans = float(transmission)
    if not (math.isfinite(c_l) and math.isfinite(c_r)) or c_l <= 0.0 or c_r <= 0.0:
        raise ValueError("carrier densities must be finite and positive for reciprocal thermionic flux")
    if not math.isfinite(temp) or temp <= 0.0:
        raise ValueError("temperature_K must be finite and positive")
    if not math.isfinite(trans) or not 0.0 <= trans <= 1.0:
        raise ValueError("transmission must be finite and lie in [0, 1]")
    vt = float(thermal_voltage(temp) if thermal_voltage_V is None else thermal_voltage_V)
    if not math.isfinite(vt) or vt <= 0.0:
        raise ValueError("thermal_voltage_V must be finite and positive")
    if electrochemical_step is not None and not math.isfinite(float(electrochemical_step)):
        raise ValueError("electrochemical_step must be finite when supplied")
    corner = float(barrier_corner_kbt)
    if not math.isfinite(corner) or corner < 0.0:
        raise ValueError("barrier_corner_kbt must be finite and non-negative")

    barrier_step, d_step_d_phi_l, d_step_d_phi_r = physical_carrier_barrier_step_eV(
        carrier,
        phi_left_V=phi_left_V,
        phi_right_V=phi_right_V,
        band_offset_eV=band_offset_eV,
    )
    current_sign = -Q if carrier == "electron" else Q

    if model_version == LEGACY_UNNORMALIZED_THERMIONIC_V1:
        if statistics != "maxwell_boltzmann":
            raise ValueError("legacy_unnormalized_thermionic_v1 supports only maxwell_boltzmann")
        a_left = float(a_star_left_A_m2_K2)
        if not math.isfinite(a_left) or a_left < 0.0:
            raise ValueError("a_star_left_A_m2_K2 must be finite and non-negative")
        flux_leg, d_dl, d_dr = thermionic_emission_bound_m2_s(
            c_l,
            c_r,
            barrier_step,
            temp,
            a_left,
            dos_norm_m3=dos_left_m3,
        )
        flux_leg *= trans
        d_dl *= trans
        d_dr *= trans
        term_l = d_dl * c_l
        term_r = -d_dr * c_r
        if barrier_step > 0.0:
            g_step = -term_l / vt
            w_leg_l, w_leg_r = 1.0, 0.0
        elif barrier_step < 0.0:
            g_step = -term_r / vt
            w_leg_l, w_leg_r = 0.0, 1.0
        else:
            g_step = -0.5 * (term_l + term_r) / vt
            w_leg_l, w_leg_r = 0.5, 0.5
        ec_step = (
            float(electrochemical_step)
            if electrochemical_step is not None
            else (_symmetric_log_ratio(c_l, c_r) + barrier_step / vt)
        )
        return ReciprocalThermionicEvaluation(
            model_version=model_version,
            carrier=carrier,
            statistics=statistics,
            flux_m2_s=flux_leg,
            current_A_m2=current_sign * flux_leg,
            one_way_scale_m2_s=max(term_l, term_r),
            barrier_step_eV=barrier_step,
            electrochemical_step=ec_step,
            effective_richardson_A_m2_K2=a_left,
            transmission=trans,
            d_flux_d_phi_left_m2_s_V=g_step * d_step_d_phi_l,
            d_flux_d_phi_right_m2_s_V=g_step * d_step_d_phi_r,
            d_flux_d_c_left_m_s=d_dl,
            d_flux_d_c_right_m_s=d_dr,
            d_flux_d_log_c_left_m2_s=term_l,
            d_flux_d_log_c_right_m2_s=-term_r,
            left_barrier_weight=w_leg_l,
            right_barrier_weight=w_leg_r,
        )

    if dos_left_m3 is None or dos_right_m3 is None:
        raise ValueError(f"{model_version} requires explicit positive dos_left_m3 and dos_right_m3")
    n_dos_l = float(dos_left_m3)
    n_dos_r = float(dos_right_m3)
    if not (math.isfinite(n_dos_l) and math.isfinite(n_dos_r)) or n_dos_l <= 0.0 or n_dos_r <= 0.0:
        raise ValueError("dos_left_m3 and dos_right_m3 must be finite and positive")

    a_right_raw = a_star_left_A_m2_K2 if a_star_right_A_m2_K2 is None else a_star_right_A_m2_K2
    if model_version == RECIPROCAL_PHYSICAL_THERMIONIC_V2:
        a_eff = reciprocal_richardson(a_star_left_A_m2_K2, a_right_raw)
    elif model_version == SCAPS_EFFECTIVE_VELOCITY_THERMIONIC_V1:
        if statistics != "maxwell_boltzmann":
            raise ValueError("scaps_effective_velocity_thermionic_v1 supports only maxwell_boltzmann")
        a_l = float(a_star_left_A_m2_K2)
        a_r = float(a_right_raw)
        if not (math.isfinite(a_l) and math.isfinite(a_r)) or a_l < 0.0 or a_r < 0.0:
            raise ValueError("Richardson constants must be finite and non-negative")
        v_th_l = a_l * temp * temp / (Q * n_dos_l)
        v_th_r = a_r * temp * temp / (Q * n_dos_r)
        a_eff = (Q / (temp * temp)) * min(v_th_l, v_th_r) * math.sqrt(n_dos_l * n_dos_r)
    else:
        raise ValueError(f"unknown thermionic model_version {model_version!r}")

    pref = trans * a_eff * temp * temp / Q
    s_vt = barrier_step / vt
    if corner == 0.0:
        b_l = max(s_vt, 0.0)
        b_r = max(-s_vt, 0.0)
        w_l = 1.0 if barrier_step > 0.0 else (0.0 if barrier_step < 0.0 else 0.5)
        w_r = 1.0 - w_l
    else:
        # Even C^infinity corner potential: b_l - b_r == s_vt identically (exact detailed balance),
        # while d(b_l)/ds = expit(s_vt/corner) eliminates the non-equilibrium Jacobian jump at s_vt = 0.
        corner_shift = corner * math.log1p(math.exp(-abs(s_vt) / corner))
        b_l = max(s_vt, 0.0) + corner_shift
        b_r = max(-s_vt, 0.0) + corner_shift
        w_l = float(expit(s_vt / corner))
        w_r = float(expit(-s_vt / corner))
    act_l = c_l / n_dos_l
    act_r = c_r / n_dos_r

    if statistics == "maxwell_boltzmann":
        sup_l = _boltzmann_over_barrier_supply(act_l, b_l)
        sup_r = _boltzmann_over_barrier_supply(act_r, b_r)
        if electrochemical_step is not None:
            ec_step = float(electrochemical_step)
            if ec_step > 0.0:
                supply_diff = sup_r * math.expm1(-ec_step)
            elif ec_step < 0.0:
                supply_diff = -sup_l * math.expm1(ec_step)
            else:
                supply_diff = 0.0
        else:
            supply_diff = sup_l - sup_r
            if sup_l > 0.0 and sup_r > 0.0:
                ec_step = _symmetric_log_ratio(sup_l, sup_r)
            else:
                ec_step = _symmetric_log_ratio(act_l, act_r) + s_vt

        flux = pref * supply_diff
        d_log_l = pref * sup_l
        d_log_r = -pref * sup_r
        d_c_l = d_log_l / c_l
        d_c_r = d_log_r / c_r
        g_step = -pref * (w_l * sup_l + w_r * sup_r) / vt
        scale_one_way = pref * max(sup_l, sup_r)
    elif statistics == "fermi_dirac":
        sup_l, dn_log_l, dn_bar_l, z_l = _one_way_fermi_dirac_supply(
            c_l, n_dos_l, b_l * vt, vt
        )
        sup_r, dn_log_r, dn_bar_r, z_r = _one_way_fermi_dirac_supply(
            c_r, n_dos_r, b_r * vt, vt
        )
        if electrochemical_step is not None:
            ec_step = float(electrochemical_step)
            delta_z = -ec_step
        elif z_l <= -40.0 and z_r <= -40.0 and sup_l > 0.0 and sup_r > 0.0:
            ec_step = _symmetric_log_ratio(sup_l, sup_r)
            delta_z = -ec_step
        else:
            lr_l = math.log(act_l)
            lr_r = math.log(act_r)
            ex_l, _ = _eta_excess_from_log_ratio(lr_l)
            ex_r, _ = _eta_excess_from_log_ratio(lr_r)
            ec_step = _symmetric_log_ratio(act_l, act_r) + s_vt + (ex_r - ex_l)
            delta_z = -ec_step

        if z_l <= -38.0 and z_r <= -38.0:
            if delta_z > 0.0:
                supply_diff = -sup_l * math.expm1(-delta_z)
            elif delta_z < 0.0:
                supply_diff = sup_r * math.expm1(delta_z)
            else:
                supply_diff = 0.0
        else:
            supply_diff = _fermi_dirac_one_difference(z_l, z_r, delta_z, sup_l, sup_r)

        flux = pref * supply_diff
        d_log_l = pref * dn_log_l
        d_log_r = -pref * dn_log_r
        d_c_l = d_log_l / c_l
        d_c_r = d_log_r / c_r
        g_step = pref * (w_l * dn_bar_l + w_r * dn_bar_r)
        scale_one_way = pref * max(sup_l, sup_r)
    else:
        raise ValueError(
            f"unknown statistics {statistics!r}; expected 'maxwell_boltzmann' or 'fermi_dirac'"
        )

    return ReciprocalThermionicEvaluation(
        model_version=model_version,
        carrier=carrier,
        statistics=statistics,
        flux_m2_s=flux,
        current_A_m2=current_sign * flux,
        one_way_scale_m2_s=scale_one_way,
        barrier_step_eV=barrier_step,
        electrochemical_step=ec_step,
        effective_richardson_A_m2_K2=a_eff,
        transmission=trans,
        d_flux_d_phi_left_m2_s_V=g_step * d_step_d_phi_l,
        d_flux_d_phi_right_m2_s_V=g_step * d_step_d_phi_r,
        d_flux_d_c_left_m_s=d_c_l,
        d_flux_d_c_right_m_s=d_c_r,
        d_flux_d_log_c_left_m2_s=d_log_l,
        d_flux_d_log_c_right_m2_s=d_log_r,
        left_barrier_weight=w_l,
        right_barrier_weight=w_r,
    )


@dataclass(frozen=True, slots=True)
class ReciprocalThermionicCompositeEvaluation:
    """Combined SG + reciprocal physical thermionic face flux and full Jacobians."""

    combination_mode: ThermionicCombinationMode
    flux_m2_s: float
    current_A_m2: float
    sg_flux_m2_s: float
    thermionic: ReciprocalThermionicEvaluation
    d_flux_d_phi_left_m2_s_V: float
    d_flux_d_phi_right_m2_s_V: float
    d_flux_d_c_left_m_s: float
    d_flux_d_c_right_m_s: float
    d_flux_d_log_c_left_m2_s: float
    d_flux_d_log_c_right_m2_s: float
    sg_weight: float = 0.0
    te_weight: float = 0.0


def combine_sg_and_reciprocal_thermionic_flux(
    sg_flux_m2_s: float,
    *,
    d_sg_d_phi_left_m2_s_V: float,
    d_sg_d_phi_right_m2_s_V: float,
    d_sg_d_c_left_m_s: float,
    d_sg_d_c_right_m_s: float,
    c_left_m3: float,
    c_right_m3: float,
    thermionic: ReciprocalThermionicEvaluation,
    combination_mode: ThermionicCombinationMode = "harmonic_series",
    te_softness: float = 0.05,
) -> ReciprocalThermionicCompositeEvaluation:
    """Combine face Scharfetter-Gummel and reciprocal thermionic fluxes with full tangents.

    Includes exact chain-rule derivatives wrt ``(phi_left, phi_right, c_left, c_right)``
    and ``(log c_left, log c_right)``. Under ``combination_mode="harmonic_series"``,
    combines bulk drift-diffusion and interface emission conductances in series with
    strictly positive smooth weights at every operating bias including ``m == 0``.
    """
    sg = float(sg_flux_m2_s)
    c_l = float(c_left_m3)
    c_r = float(c_right_m3)
    soft = float(te_softness)
    if not (math.isfinite(sg) and math.isfinite(c_l) and math.isfinite(c_r) and math.isfinite(soft)):
        raise ValueError("flux, densities, and te_softness must be finite")
    if c_l <= 0.0 or c_r <= 0.0:
        raise ValueError("c_left_m3 and c_right_m3 must be positive")
    if soft < 0.0:
        raise ValueError("te_softness must be non-negative")

    te = thermionic.flux_m2_s
    a_sg = abs(sg)
    a_te = abs(te)
    sign_sg = math.copysign(1.0, sg)
    sign_te = 0.0 if te == 0.0 else math.copysign(1.0, te)
    sign_pair = 1.0 if (sg == 0.0 and te == 0.0) else sign_sg * sign_te
    current_sign = -Q if thermionic.carrier == "electron" else Q

    g_sg = 0.5 * (abs(float(d_sg_d_c_left_m_s) * c_l) + abs(float(d_sg_d_c_right_m_s) * c_r))
    g_te = 0.5 * (
        abs(thermionic.d_flux_d_log_c_left_m2_s) + abs(thermionic.d_flux_d_log_c_right_m2_s)
    )

    if combination_mode == "harmonic_series":
        g_sum = g_sg + g_te
        if g_sum == 0.0:
            flux = math.copysign(0.0, sg)
            w_sg = 0.0
            w_te = 0.0
        elif a_sg <= 1.0e-6 * g_sg and a_te <= 1.0e-6 * g_te:
            # Exact C^infinity Taylor-exact asymptotic form around m = 0:
            # u*v/(u+v) == theta0^2 * u + (1 - theta0)^2 * v + O(m^3), avoiding 0/0 and 1-ULP noise.
            theta0 = g_te / g_sum
            w_sg = theta0 * theta0
            w_te = (1.0 - theta0) * (1.0 - theta0)
            flux = w_sg * sg + w_te * te
        elif sg * te > 0.0:
            denom_st = sg + te
            theta = te / denom_st
            flux = theta * sg
            w_sg = theta * theta
            w_te = (1.0 - theta) * (1.0 - theta)
        elif a_sg == 0.0 or a_te == 0.0:
            flux = math.copysign(0.0, sg)
            theta0 = g_te / g_sum
            w_sg = theta0 * theta0
            w_te = (1.0 - theta0) * (1.0 - theta0)
        else:
            a_sum = a_sg + a_te
            theta = a_te / a_sum
            flux = math.copysign(theta * a_sg, sg)
            w_sg = theta * theta
            w_te = sign_pair * (1.0 - theta) * (1.0 - theta)
    elif combination_mode == "hard" or soft == 0.0:
        if a_sg == 0.0 and a_te == 0.0:
            flux = math.copysign(0.0, sg)
            if g_sg <= g_te:
                w_sg, w_te = 1.0, 0.0
            else:
                w_sg, w_te = 0.0, 1.0
        elif a_sg <= a_te:
            flux = sg
            w_sg, w_te = 1.0, 0.0
        else:
            flux = math.copysign(a_te, sg)
            w_sg, w_te = 0.0, sign_pair
    elif combination_mode == "legacy_logistic":
        if a_sg <= 1.0e-12 * g_sg and a_te <= 1.0e-12 * g_te:
            g_sum = g_sg + g_te
            if g_sum == 0.0:
                w_sg, w_te = 0.5, 0.0
            else:
                t0 = (g_sg - g_te) / (soft * g_sum)
                if t0 > 40.0:
                    weight0 = 1.0
                elif t0 < -40.0:
                    weight0 = 0.0
                else:
                    weight0 = 1.0 / (1.0 + math.exp(-t0))
                w_sg, w_te = 1.0 - weight0, weight0
            flux = w_sg * sg + w_te * te
        else:
            denom = soft * (a_sg + a_te) + 1.0e-300
            t_arg = (a_sg - a_te) / denom
            if t_arg > 40.0:
                weight = 1.0
                dw_dt = 0.0
            elif t_arg < -40.0:
                weight = 0.0
                dw_dt = 0.0
            else:
                weight = 1.0 / (1.0 + math.exp(-t_arg))
                dw_dt = weight * (1.0 - weight)
            magnitude = weight * a_te + (1.0 - weight) * a_sg
            denom_sq = denom * denom
            dt_da_sg = (denom - (a_sg - a_te) * soft) / denom_sq
            dt_da_te = (-denom - (a_sg - a_te) * soft) / denom_sq
            w_sg = (1.0 - weight) + (a_te - a_sg) * dw_dt * dt_da_sg
            dm_da_te = weight + (a_te - a_sg) * dw_dt * dt_da_te
            w_te = sign_pair * dm_da_te
            flux = math.copysign(magnitude, sg)
    elif combination_mode == "compact_support":
        if a_sg <= 1.0e-12 * g_sg and a_te <= 1.0e-12 * g_te:
            g_sum = g_sg + g_te
            if g_sum == 0.0:
                w_sg, w_te = 0.5, 0.0
            else:
                n0 = (g_sg - g_te) / (soft * g_sum)
                if n0 <= -1.0:
                    w_sg, w_te = 1.0, 0.0
                elif n0 >= 1.0:
                    w_sg, w_te = 0.0, 1.0
                else:
                    u0 = 0.5 * (n0 + 1.0)
                    weight0 = 6.0 * u0**5 - 15.0 * u0**4 + 10.0 * u0**3
                    w_sg, w_te = 1.0 - weight0, weight0
            flux = w_sg * sg + w_te * te
        else:
            scale = soft * (a_sg + a_te)
            normalized = (a_sg - a_te) / scale
            if normalized <= -1.0:
                magnitude = a_sg
                w_sg = 1.0
                dm_da_te = 0.0
            elif normalized >= 1.0:
                magnitude = a_te
                w_sg = 0.0
                dm_da_te = 1.0
            else:
                u = 0.5 * (normalized + 1.0)
                weight = 6.0 * u**5 - 15.0 * u**4 + 10.0 * u**3
                dw_du = 30.0 * u**2 * (1.0 - u) ** 2
                scale_sq = scale * scale
                dn_da_sg = 2.0 * soft * a_te / scale_sq
                dn_da_te = -2.0 * soft * a_sg / scale_sq
                dw_da_sg = 0.5 * dw_du * dn_da_sg
                dw_da_te = 0.5 * dw_du * dn_da_te
                magnitude = (1.0 - weight) * a_sg + weight * a_te
                w_sg = (1.0 - weight) + (a_te - a_sg) * dw_da_sg
                dm_da_te = weight + (a_te - a_sg) * dw_da_te
            w_te = sign_pair * dm_da_te
            flux = math.copysign(magnitude, sg)
    else:
        raise ValueError(
            f"unknown combination_mode {combination_mode!r}; expected 'harmonic_series', "
            "'compact_support', 'legacy_logistic', or 'hard'"
        )

    d_phi_l = w_sg * float(d_sg_d_phi_left_m2_s_V) + w_te * thermionic.d_flux_d_phi_left_m2_s_V
    d_phi_r = w_sg * float(d_sg_d_phi_right_m2_s_V) + w_te * thermionic.d_flux_d_phi_right_m2_s_V
    d_c_l = w_sg * float(d_sg_d_c_left_m_s) + w_te * thermionic.d_flux_d_c_left_m_s
    d_c_r = w_sg * float(d_sg_d_c_right_m_s) + w_te * thermionic.d_flux_d_c_right_m_s
    d_log_l = w_sg * (float(d_sg_d_c_left_m_s) * c_l) + w_te * thermionic.d_flux_d_log_c_left_m2_s
    d_log_r = w_sg * (float(d_sg_d_c_right_m_s) * c_r) + w_te * thermionic.d_flux_d_log_c_right_m2_s

    return ReciprocalThermionicCompositeEvaluation(
        combination_mode=combination_mode,
        flux_m2_s=flux,
        current_A_m2=current_sign * flux,
        sg_flux_m2_s=sg,
        thermionic=thermionic,
        d_flux_d_phi_left_m2_s_V=d_phi_l,
        d_flux_d_phi_right_m2_s_V=d_phi_r,
        d_flux_d_c_left_m_s=d_c_l,
        d_flux_d_c_right_m_s=d_c_r,
        d_flux_d_log_c_left_m2_s=d_log_l,
        d_flux_d_log_c_right_m2_s=d_log_r,
        sg_weight=w_sg,
        te_weight=w_te,
    )
