"""Multivalent / amphoteric defect grand-partition closure and exact tangents.

Implements the Decock-Khelifi-Burgelman coupled charge-state chain on 2..5
descending integer charge states ``q_0 > q_1 > ... > q_{M-1}`` with explicit
state degeneracies ``g_s`` (`unity`, `scaps_binomial`, or `explicit`), log-space
partition normalization (preventing overflow/underflow without clipping),
cancellation-free ``(P_s + P_{s+1}) * c_n c_p (n p - n_i^2) / (a_s + b_s)``
stationary SRH transition rates, and exact analytical tangents with respect to
``(n, p)`` and fixed-QF ``phi``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

from solarlab.contracts.units import ELEMENTARY_CHARGE_C as Q
from solarlab.physics.statistics import (
    CHARGE_NEUTRALITY_RELATIVE_TOLERANCE,
    thermal_voltage,
)

__all__ = [
    "MultivalentConfigurationSpec",
    "MultivalentDefectClosureResult",
    "MultivalentDefectNeutralityState",
    "MultivalentDefectRegion",
    "MultivalentSpeciesSpec",
    "MultivalentTransitionKineticsSpec",
    "evaluate_multivalent_defect_closure",
    "evaluate_multivalent_region_closure",
    "solve_multivalent_defect_charge_neutrality",
]


def _ro_float(values: ArrayLike) -> NDArray[np.float64]:
    arr = np.array(values, dtype=np.float64, copy=True)
    arr.setflags(write=False)
    return arr


def _ro_bool(values: ArrayLike) -> NDArray[np.bool_]:
    arr = np.array(values, dtype=np.bool_, copy=True)
    arr.setflags(write=False)
    return arr


def _finite_positive(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite and positive")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return number


def _finite_nonnegative(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite and non-negative")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return number


@dataclass(frozen=True, slots=True)
class MultivalentTransitionKineticsSpec:
    """Capture kinetics for one adjacent charge-state transition ``q_s <-> q_{s+1}``."""

    sigma_n_m2: float
    sigma_p_m2: float
    thermal_velocity_n_m_s: float
    thermal_velocity_p_m_s: float

    def __post_init__(self) -> None:
        sn = _finite_nonnegative(self.sigma_n_m2, "sigma_n_m2")
        sp = _finite_nonnegative(self.sigma_p_m2, "sigma_p_m2")
        vn = _finite_positive(self.thermal_velocity_n_m_s, "thermal_velocity_n_m_s")
        vp = _finite_positive(self.thermal_velocity_p_m_s, "thermal_velocity_p_m_s")
        if sn == 0.0 and sp == 0.0:
            raise ValueError("multivalent transition requires at least one nonzero capture leg")
        object.__setattr__(self, "sigma_n_m2", sn)
        object.__setattr__(self, "sigma_p_m2", sp)
        object.__setattr__(self, "thermal_velocity_n_m_s", vn)
        object.__setattr__(self, "thermal_velocity_p_m_s", vp)

    @property
    def capture_n_m3_s(self) -> float:
        return self.sigma_n_m2 * self.thermal_velocity_n_m_s

    @property
    def capture_p_m3_s(self) -> float:
        return self.sigma_p_m2 * self.thermal_velocity_p_m_s


@dataclass(frozen=True, slots=True)
class MultivalentConfigurationSpec:
    """Charge-state ladder, degeneracies, transition energies, and kinetics."""

    family: str
    charge_states_e: tuple[int, ...]
    degeneracy_convention: str
    state_degeneracies: tuple[float, ...]
    transition_energies_eV_above_vb: tuple[float, ...]
    transition_kinetics: tuple[MultivalentTransitionKineticsSpec, ...]

    def __post_init__(self) -> None:
        charges = tuple(self.charge_states_e)
        if not 2 <= len(charges) <= 5 or any(
            isinstance(c, bool) or not isinstance(c, int) or not -3 <= c <= 3
            for c in charges
        ):
            raise ValueError("multivalent charge_states_e requires 2..5 integers in [-3, 3]")
        if any(right_q != left_q - 1 for left_q, right_q in zip(charges, charges[1:])):
            raise ValueError("multivalent charge_states_e must descend by 1")
        degs = tuple(_finite_positive(g, "state_degeneracy") for g in self.state_degeneracies)
        if len(degs) != len(charges):
            raise ValueError("state_degeneracies length must match charge_states_e")
        if self.degeneracy_convention == "unity" and any(g != 1.0 for g in degs):
            raise ValueError("unity degeneracy_convention requires all degeneracies == 1")
        if self.degeneracy_convention == "scaps_binomial" and degs != tuple(
            float(math.comb(len(charges) - 1, i)) for i in range(len(charges))
        ):
            raise ValueError("scaps_binomial requires binomial degeneracies")
        energies = tuple(
            float(e) for e in self.transition_energies_eV_above_vb
        )
        if any(not math.isfinite(e) for e in energies) or len(energies) != len(charges) - 1:
            raise ValueError("one finite transition energy is required per adjacent pair")
        kins = tuple(self.transition_kinetics)
        if len(kins) != len(charges) - 1:
            raise ValueError("one kinetics specification is required per adjacent pair")
        object.__setattr__(self, "charge_states_e", charges)
        object.__setattr__(self, "state_degeneracies", degs)
        object.__setattr__(self, "transition_energies_eV_above_vb", energies)
        object.__setattr__(self, "transition_kinetics", kins)

    def validate_band_gap(self, band_gap_eV: float) -> None:
        gap = _finite_positive(band_gap_eV, "band_gap_eV")
        tol = 16.0 * math.ulp(max(gap, *(abs(e) for e in self.transition_energies_eV_above_vb), 1.0))
        if any(e < -tol or e > gap + tol for e in self.transition_energies_eV_above_vb):
            raise ValueError("multivalent transition energy lies outside [0, band_gap_eV]")


@dataclass(frozen=True, slots=True)
class MultivalentSpeciesSpec:
    """Named multivalent bulk defect species with total physical center density."""

    identifier: str
    total_density_m3: float
    configuration: MultivalentConfigurationSpec

    def __post_init__(self) -> None:
        ident = str(self.identifier).strip()
        if not ident:
            raise ValueError("MultivalentSpeciesSpec requires a non-empty identifier")
        dens = _finite_nonnegative(self.total_density_m3, "total_density_m3")
        object.__setattr__(self, "identifier", ident)
        object.__setattr__(self, "total_density_m3", dens)


@dataclass(frozen=True, slots=True)
class MultivalentDefectClosureResult:
    """Stationary state probabilities, charge, recombination, and exact tangents."""

    species_identifier: str
    total_density_m3: float
    charge_states_e: tuple[int, ...]
    state_degeneracies: tuple[float, ...]
    state_probability: NDArray[np.float64]
    state_probability_derivative_n_m3: NDArray[np.float64]
    state_probability_derivative_p_m3: NDArray[np.float64]
    transition_recombination_rate_m3_s: NDArray[np.float64]
    transition_recombination_derivative_n_s1: NDArray[np.float64]
    transition_recombination_derivative_p_s1: NDArray[np.float64]
    mean_charge_e: NDArray[np.float64]
    charge_density_C_m3: NDArray[np.float64]
    charge_derivative_n_C: NDArray[np.float64]
    charge_derivative_p_C: NDArray[np.float64]
    charge_derivative_fixed_qf_C_m3_V: NDArray[np.float64]
    total_recombination_rate_m3_s: NDArray[np.float64]
    total_recombination_derivative_n_s1: NDArray[np.float64]
    total_recombination_derivative_p_s1: NDArray[np.float64]
    recombination_derivative_fixed_qf_m3_s_V: NDArray[np.float64]
    maximum_probability_sum_error: float
    maximum_master_residual_s1: float

    def __post_init__(self) -> None:
        for name in (
            "state_probability",
            "state_probability_derivative_n_m3",
            "state_probability_derivative_p_m3",
            "transition_recombination_rate_m3_s",
            "transition_recombination_derivative_n_s1",
            "transition_recombination_derivative_p_s1",
            "mean_charge_e",
            "charge_density_C_m3",
            "charge_derivative_n_C",
            "charge_derivative_p_C",
            "charge_derivative_fixed_qf_C_m3_V",
            "total_recombination_rate_m3_s",
            "total_recombination_derivative_n_s1",
            "total_recombination_derivative_p_s1",
            "recombination_derivative_fixed_qf_m3_s_V",
        ):
            object.__setattr__(self, name, _ro_float(getattr(self, name)))


def evaluate_multivalent_defect_closure(
    electron_density_m3: ArrayLike,
    hole_density_m3: ArrayLike,
    species: MultivalentSpeciesSpec,
    *,
    band_gap_eV: ArrayLike,
    effective_conduction_dos_m3: ArrayLike,
    effective_valence_dos_m3: ArrayLike,
    temperature_K: float,
) -> MultivalentDefectClosureResult:
    """Solve coupled adjacent-state probabilities and exact derivatives without clipping."""
    n, p = np.broadcast_arrays(
        np.asarray(electron_density_m3, dtype=np.float64),
        np.asarray(hole_density_m3, dtype=np.float64),
    )
    if np.any(~np.isfinite(n)) or np.any(~np.isfinite(p)) or np.any(n <= 0.0) or np.any(p <= 0.0):
        raise ValueError("multivalent closure requires finite positive carrier densities")
    gap = np.broadcast_to(np.asarray(band_gap_eV, dtype=np.float64), n.shape)
    nc = np.broadcast_to(
        np.asarray(effective_conduction_dos_m3, dtype=np.float64), n.shape
    )
    nv = np.broadcast_to(
        np.asarray(effective_valence_dos_m3, dtype=np.float64), n.shape
    )
    if np.any(gap <= 0.0) or np.any(nc <= 0.0) or np.any(nv <= 0.0):
        raise ValueError("band_gap_eV, Nc, and Nv must be positive")
    vt = thermal_voltage(_finite_positive(temperature_K, "temperature_K"))

    cfg = species.configuration
    cfg.validate_band_gap(float(np.min(gap)))
    t_count = len(cfg.transition_energies_eV_above_vb)
    s_count = t_count + 1
    t_exp = (t_count,) + (1,) * n.ndim
    s_shape = (s_count, *n.shape)

    energies = np.asarray(cfg.transition_energies_eV_above_vb, dtype=np.float64).reshape(t_exp)
    cn = np.asarray(
        [k.capture_n_m3_s for k in cfg.transition_kinetics], dtype=np.float64
    ).reshape(t_exp)
    cp = np.asarray(
        [k.capture_p_m3_s for k in cfg.transition_kinetics], dtype=np.float64
    ).reshape(t_exp)
    degs: NDArray[np.float64] = np.asarray(cfg.state_degeneracies, dtype=np.float64)
    deg_ratio = (degs[1:] / degs[:-1]).reshape(t_exp)

    n_e = n[None, ...]
    p_e = p[None, ...]
    gap_e = gap[None, ...]
    nc_e = nc[None, ...]
    nv_e = nv[None, ...]

    n1 = nc_e * np.exp(-(gap_e - energies) / vt)
    p1 = nv_e * np.exp(-energies / vt)
    em_n = cn * n1 / deg_ratio
    em_p = cp * p1 * deg_ratio

    forward = cn * n_e + em_p
    backward = em_n + cp * p_e
    if np.any(~np.isfinite(forward)) or np.any(~np.isfinite(backward)) or np.any(forward <= 0.0) or np.any(backward <= 0.0):
        raise FloatingPointError("multivalent adjacent-state transition rates are invalid")

    log_ratio = np.log(forward) - np.log(backward)
    log_w = np.zeros(s_shape, dtype=np.float64)
    log_w[1:] = np.cumsum(log_ratio, axis=0)
    log_w -= np.max(log_w, axis=0, keepdims=True)
    w = np.exp(log_w)
    prob = w / np.sum(w, axis=0, keepdims=True)

    dlog_dn = np.zeros(s_shape, dtype=np.float64)
    dlog_dp = np.zeros(s_shape, dtype=np.float64)
    dlog_dn[1:] = np.cumsum(cn / forward, axis=0)
    dlog_dp[1:] = np.cumsum(-cp / backward, axis=0)
    mean_dn = np.sum(prob * dlog_dn, axis=0, keepdims=True)
    mean_dp = np.sum(prob * dlog_dp, axis=0, keepdims=True)
    dprob_dn = prob * (dlog_dn - mean_dn)
    dprob_dp = prob * (dlog_dp - mean_dp)

    # Master-equation stationary residual diagnostic:
    master_res = np.zeros(s_shape, dtype=np.float64)
    flux_diff = backward * prob[1:] - forward * prob[:-1]
    master_res[:-1] += flux_diff
    master_res[1:] -= flux_diff
    max_master_res = float(np.max(np.abs(master_res)))
    max_sum_err = float(np.max(np.abs(np.sum(prob, axis=0) - 1.0)))

    ni_sq = (nc * nv * np.exp(-gap / vt))[None, ...]
    pair_prob = prob[:-1] + prob[1:]
    dpair_dn = dprob_dn[:-1] + dprob_dn[1:]
    dpair_dp = dprob_dp[:-1] + dprob_dp[1:]

    cap_prod = cn * cp
    num = cap_prod * (n_e * p_e - ni_sq)
    denom = forward + backward
    frac = num / denom
    dfrac_dn = (cap_prod * p_e * denom - num * cn) / (denom * denom)
    dfrac_dp = (cap_prod * n_e * denom - num * cp) / (denom * denom)

    nt = species.total_density_m3
    tr_rate = nt * pair_prob * frac
    dtr_dn = nt * (dpair_dn * frac + pair_prob * dfrac_dn)
    dtr_dp = nt * (dpair_dp * frac + pair_prob * dfrac_dp)

    tot_rate = np.sum(tr_rate, axis=0)
    dtot_dn = np.sum(dtr_dn, axis=0)
    dtot_dp = np.sum(dtr_dp, axis=0)
    dtot_dphi = (dtot_dn * n - dtot_dp * p) / vt

    charges_e = np.asarray(cfg.charge_states_e, dtype=np.float64).reshape(
        (s_count,) + (1,) * n.ndim
    )
    mean_q = np.sum(charges_e * prob, axis=0)
    rho = Q * nt * mean_q
    drho_dn = Q * nt * np.sum(charges_e * dprob_dn, axis=0)
    drho_dp = Q * nt * np.sum(charges_e * dprob_dp, axis=0)
    drho_dphi = (drho_dn * n - drho_dp * p) / vt

    return MultivalentDefectClosureResult(
        species_identifier=species.identifier,
        total_density_m3=nt,
        charge_states_e=cfg.charge_states_e,
        state_degeneracies=cfg.state_degeneracies,
        state_probability=prob,
        state_probability_derivative_n_m3=dprob_dn,
        state_probability_derivative_p_m3=dprob_dp,
        transition_recombination_rate_m3_s=tr_rate,
        transition_recombination_derivative_n_s1=dtr_dn,
        transition_recombination_derivative_p_s1=dtr_dp,
        mean_charge_e=mean_q,
        charge_density_C_m3=rho,
        charge_derivative_n_C=drho_dn,
        charge_derivative_p_C=drho_dp,
        charge_derivative_fixed_qf_C_m3_V=drho_dphi,
        total_recombination_rate_m3_s=tot_rate,
        total_recombination_derivative_n_s1=dtot_dn,
        total_recombination_derivative_p_s1=dtot_dp,
        recombination_derivative_fixed_qf_m3_s_V=dtot_dphi,
        maximum_probability_sum_error=max_sum_err,
        maximum_master_residual_s1=max_master_res,
    )


@dataclass(frozen=True, slots=True)
class MultivalentDefectRegion:
    """Compiled multivalent defect inventory on a disjoint region."""

    identifier: str
    document_sha256: str
    active_nodes: NDArray[np.bool_]
    band_gap_eV: NDArray[np.float64]
    effective_conduction_dos_m3: NDArray[np.float64]
    effective_valence_dos_m3: NDArray[np.float64]
    temperature_K: float
    species: tuple[MultivalentSpeciesSpec, ...]

    def __post_init__(self) -> None:
        mask = _ro_bool(self.active_nodes)
        if mask.ndim != 1 or not np.any(mask):
            raise ValueError("active_nodes must be a non-empty 1D boolean mask")
        active_count = int(np.sum(mask))
        gap = _ro_float(np.broadcast_to(self.band_gap_eV, (active_count,)))
        nc = _ro_float(
            np.broadcast_to(self.effective_conduction_dos_m3, (active_count,))
        )
        nv = _ro_float(
            np.broadcast_to(self.effective_valence_dos_m3, (active_count,))
        )
        temp = _finite_positive(self.temperature_K, "temperature_K")
        sp = tuple(self.species)
        if not sp:
            raise ValueError("MultivalentDefectRegion requires at least one species")
        min_gap = float(np.min(gap))
        for item in sp:
            item.configuration.validate_band_gap(min_gap)
        object.__setattr__(self, "active_nodes", mask)
        object.__setattr__(self, "band_gap_eV", gap)
        object.__setattr__(self, "effective_conduction_dos_m3", nc)
        object.__setattr__(self, "effective_valence_dos_m3", nv)
        object.__setattr__(self, "temperature_K", temp)
        object.__setattr__(self, "species", sp)


def evaluate_multivalent_region_closure(
    region: MultivalentDefectRegion,
    electron_density_m3: ArrayLike,
    hole_density_m3: ArrayLike,
) -> tuple[MultivalentDefectClosureResult, ...]:
    """Evaluate all multivalent species in ``region`` on full-grid carrier vectors."""
    n = np.asarray(electron_density_m3, dtype=np.float64)
    p = np.asarray(hole_density_m3, dtype=np.float64)
    n_loc = n[region.active_nodes]
    p_loc = p[region.active_nodes]
    return tuple(
        evaluate_multivalent_defect_closure(
            n_loc,
            p_loc,
            sp,
            band_gap_eV=region.band_gap_eV,
            effective_conduction_dos_m3=region.effective_conduction_dos_m3,
            effective_valence_dos_m3=region.effective_valence_dos_m3,
            temperature_K=region.temperature_K,
        )
        for sp in region.species
    )


@dataclass(frozen=True, slots=True)
class MultivalentDefectNeutralityState:
    """Equilibrium charge-neutrality state including multivalent defects."""

    temperature_K: float
    thermal_voltage_V: float
    band_gap_eV: float
    effective_conduction_dos_m3: float
    effective_valence_dos_m3: float
    donor_density_m3: float
    acceptor_density_m3: float
    reduced_electron_fermi_level: float
    reduced_hole_fermi_level: float
    electron_density_m3: float
    hole_density_m3: float
    defect_charge_density_C_m3: float
    normalized_charge_residual: float
    work_function_eV: float | None = None


def solve_multivalent_defect_charge_neutrality(
    *,
    temperature_K: float,
    band_gap_eV: float,
    effective_conduction_dos_m3: float,
    effective_valence_dos_m3: float,
    donor_density_m3: float,
    acceptor_density_m3: float,
    species: Sequence[MultivalentSpeciesSpec],
    electron_affinity_eV: float | None = None,
) -> MultivalentDefectNeutralityState:
    """Solve equilibrium charge neutrality ``p - n + N_D - N_A + rho_mv/q == 0``.

    Even under negative-U (inverted correlation energies), the grand-canonical
    mean charge ``<q>(E_F)`` satisfies ``d<q>/d(eta_n) = -Var(q) <= 0``, so total
    charge is strictly decreasing in ``eta_n`` and has a unique root.
    """
    temp = _finite_positive(temperature_K, "temperature_K")
    gap = _finite_positive(band_gap_eV, "band_gap_eV")
    nc = _finite_positive(effective_conduction_dos_m3, "effective_conduction_dos_m3")
    nv = _finite_positive(effective_valence_dos_m3, "effective_valence_dos_m3")
    nd = _finite_nonnegative(donor_density_m3, "donor_density_m3")
    na = _finite_nonnegative(acceptor_density_m3, "acceptor_density_m3")
    vt = thermal_voltage(temp)

    def eval_eta(eta_n: float) -> tuple[float, float, float, float, float]:
        eta_p = -gap / vt - eta_n
        n_val = nc * math.exp(eta_n)
        p_val = nv * math.exp(eta_p)
        rho_mv = 0.0
        for sp in species:
            res = evaluate_multivalent_defect_closure(
                np.array([n_val]),
                np.array([p_val]),
                sp,
                band_gap_eV=gap,
                effective_conduction_dos_m3=nc,
                effective_valence_dos_m3=nv,
                temperature_K=temp,
            )
            rho_mv += float(res.charge_density_C_m3[0])
        mv_num = rho_mv / Q
        net = (p_val + nd + max(mv_num, 0.0)) - (n_val + na + max(-mv_num, 0.0))
        scale = max(p_val + n_val + nd + na + abs(mv_num), 1.0)
        return net, scale, n_val, p_val, rho_mv

    eta_mid = -0.5 * gap / vt + 0.5 * math.log(nv / nc)
    lo = eta_mid - 1.0
    hi = eta_mid + 1.0
    f_lo, _, _, _, _ = eval_eta(lo)
    f_hi, _, _, _, _ = eval_eta(hi)
    step = 2.0
    for _ in range(60):
        if f_lo >= 0.0 and f_hi <= 0.0:
            break
        if f_lo < 0.0:
            lo -= step
            f_lo, _, _, _, _ = eval_eta(lo)
        if f_hi > 0.0:
            hi += step
            f_hi, _, _, _, _ = eval_eta(hi)
        step = min(step * 1.5, 40.0)
    else:
        raise RuntimeError("failed to bracket multivalent equilibrium Fermi level")

    mid = 0.5 * (lo + hi)
    for _ in range(120):
        mid = 0.5 * (lo + hi)
        val, scale, _, _, _ = eval_eta(mid)
        if abs(val) / scale <= CHARGE_NEUTRALITY_RELATIVE_TOLERANCE or hi - lo <= 1e-15 * max(
            1.0, abs(lo), abs(hi)
        ):
            break
        if val > 0.0:
            lo = mid
        else:
            hi = mid

    val, scale, n_eq, p_eq, rho_eq = eval_eta(mid)
    norm_res = abs(val) / scale
    if norm_res > 10.0 * CHARGE_NEUTRALITY_RELATIVE_TOLERANCE:
        raise RuntimeError(
            f"multivalent neutrality residual {norm_res:.3e} exceeded tolerance"
        )
    wf: float | None = None
    if electron_affinity_eV is not None:
        wf = _finite_positive(electron_affinity_eV, "electron_affinity_eV") - vt * mid
    return MultivalentDefectNeutralityState(
        temperature_K=temp,
        thermal_voltage_V=vt,
        band_gap_eV=gap,
        effective_conduction_dos_m3=nc,
        effective_valence_dos_m3=nv,
        donor_density_m3=nd,
        acceptor_density_m3=na,
        reduced_electron_fermi_level=mid,
        reduced_hole_fermi_level=-gap / vt - mid,
        electron_density_m3=n_eq,
        hole_density_m3=p_eq,
        defect_charge_density_C_m3=rho_eq,
        normalized_charge_residual=norm_res,
        work_function_eV=wf,
    )
