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
from typing import Literal, TypeAlias

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
    "PLANCK_CONSTANT_J_S",
    "TE_BARRIER_THRESHOLD_EV",
    "ThermionicCapEvaluation",
    "TwoSidedCrossFluxEvaluation",
    "cap_particle_flux_thermionic",
    "effective_mass_from_dos",
    "emission_velocity",
    "fermi_dirac_one",
    "fermi_dirac_richardson_flux",
    "fermi_dirac_zero",
    "fermi_richardson_flux",
    "is_free_electron_default",
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

TERegularizationMode: TypeAlias = Literal["legacy_logistic", "compact_support"]
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
