"""Metastable donor/acceptor configuration kinetics, frozen closure, and step advance.

Implements the Lany-Zunger / Igalson two-configuration metastable defect model:
- Stationary infinite-time configuration closure ``y* = k_AD / (k_DA + k_AD)``
  with exact detailed-balance barrier verification and carrier derivatives;
- Frozen-configuration bulk evaluation weighting separate donor ``N_t * y`` and
  acceptor ``N_t * (1 - y)`` multivalent closures;
- Exact piecewise-exponential step advance ``y(t + dt) = y* + (y(t) - y*) exp(-(k_DA + k_AD) dt)``
  and fixed-carrier configuration charge transfer ``Delta_rho_cfg = q * N_t * (<q_D> - <q_A>) * Delta_y``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

from solarlab.contracts.units import ELEMENTARY_CHARGE_C as Q
from solarlab.physics.defects.dynamic_single_level import stable_expit
from solarlab.physics.defects.multivalent import (
    MultivalentConfigurationSpec,
    MultivalentDefectNeutralityState,
    MultivalentSpeciesSpec,
    evaluate_multivalent_defect_closure,
    solve_multivalent_defect_charge_neutrality,
)
from solarlab.physics.statistics import thermal_voltage

DOUBLE_ELECTRON_CAPTURE = "double_electron_capture"
ELECTRON_CAPTURE_PLUS_HOLE_EMISSION = "electron_capture_plus_hole_emission"
DOUBLE_HOLE_CAPTURE = "double_hole_capture"
HOLE_CAPTURE_PLUS_ELECTRON_EMISSION = "hole_capture_plus_electron_emission"

__all__ = [
    "DOUBLE_ELECTRON_CAPTURE",
    "DOUBLE_HOLE_CAPTURE",
    "ELECTRON_CAPTURE_PLUS_HOLE_EMISSION",
    "FrozenMetastableEvaluation",
    "FrozenMetastableRegion",
    "HOLE_CAPTURE_PLUS_ELECTRON_EMISSION",
    "MetastableConfigurationClosureResult",
    "MetastableConversionKineticsSpec",
    "MetastableDefectSpec",
    "MetastableDynamicStep",
    "advance_metastable_configuration",
    "configuration_charge_transfer_C_m3",
    "configuration_from_logit",
    "configuration_logit",
    "evaluate_frozen_metastable_region",
    "evaluate_metastable_configuration_closure",
    "solve_frozen_metastable_charge_neutrality",
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
class MetastableConversionKineticsSpec:
    """Thermally activated two-carrier conversion barriers and attempt frequency."""

    transition_energy_eV_above_vb: float
    electron_capture_activation_eV: float
    electron_emission_activation_eV: float
    hole_capture_activation_eV: float
    hole_emission_activation_eV: float
    electron_capture_path: str
    hole_capture_path: str
    capture_n_m3_s: float
    capture_p_m3_s: float
    phonon_frequency_Hz: float

    def __post_init__(self) -> None:
        if self.electron_capture_path not in (
            DOUBLE_ELECTRON_CAPTURE,
            ELECTRON_CAPTURE_PLUS_HOLE_EMISSION,
        ):
            raise ValueError(
                f"unsupported electron_capture_path: {self.electron_capture_path!r}"
            )
        if self.hole_capture_path not in (
            DOUBLE_HOLE_CAPTURE,
            HOLE_CAPTURE_PLUS_ELECTRON_EMISSION,
        ):
            raise ValueError(
                f"unsupported hole_capture_path: {self.hole_capture_path!r}"
            )
        _finite_nonnegative(
            self.transition_energy_eV_above_vb, "transition_energy_eV_above_vb"
        )
        _finite_nonnegative(
            self.electron_capture_activation_eV, "electron_capture_activation_eV"
        )
        _finite_nonnegative(
            self.electron_emission_activation_eV, "electron_emission_activation_eV"
        )
        _finite_nonnegative(
            self.hole_capture_activation_eV, "hole_capture_activation_eV"
        )
        _finite_nonnegative(
            self.hole_emission_activation_eV, "hole_emission_activation_eV"
        )
        _finite_positive(self.capture_n_m3_s, "capture_n_m3_s")
        _finite_positive(self.capture_p_m3_s, "capture_p_m3_s")
        _finite_positive(self.phonon_frequency_Hz, "phonon_frequency_Hz")

    def validate_band_gap(self, band_gap_eV: float) -> None:
        gap = _finite_positive(band_gap_eV, "band_gap_eV")
        tr = self.transition_energy_eV_above_vb
        if tr > gap:
            raise ValueError("metastable transition energy lies outside [0, band_gap_eV]")
        ec = self.electron_capture_activation_eV
        hc = self.hole_capture_activation_eV
        expected_ee = (
            ec + 2.0 * (gap - tr)
            if self.electron_capture_path == DOUBLE_ELECTRON_CAPTURE
            else ec + gap - 2.0 * tr
        )
        expected_he = (
            hc + 2.0 * tr
            if self.hole_capture_path == DOUBLE_HOLE_CAPTURE
            else hc + 2.0 * tr - gap
        )
        tol = 64.0 * math.ulp(
            max(
                gap,
                tr,
                abs(expected_ee),
                abs(expected_he),
                self.electron_emission_activation_eV,
                self.hole_emission_activation_eV,
                1.0,
            )
        )
        if (
            abs(self.electron_emission_activation_eV - expected_ee) > tol
            or abs(self.hole_emission_activation_eV - expected_he) > tol
        ):
            raise ValueError("metastable activation barriers violate detailed balance")


@dataclass(frozen=True, slots=True)
class MetastableDefectSpec:
    """Two-configuration metastable defect species (donor vs acceptor branches)."""

    identifier: str
    total_density_m3: float
    donor_configuration: MultivalentConfigurationSpec
    acceptor_configuration: MultivalentConfigurationSpec
    donor_conversion_state_index: int
    acceptor_conversion_state_index: int
    conversion_kinetics: MetastableConversionKineticsSpec

    def __post_init__(self) -> None:
        ident = str(self.identifier).strip()
        if not ident:
            raise ValueError("MetastableDefectSpec requires a non-empty identifier")
        nt = _finite_positive(self.total_density_m3, "total_density_m3")
        di = int(self.donor_conversion_state_index)
        ai = int(self.acceptor_conversion_state_index)
        d_charges = self.donor_configuration.charge_states_e
        a_charges = self.acceptor_configuration.charge_states_e
        if not (0 <= di < len(d_charges) and 0 <= ai < len(a_charges)):
            raise ValueError("conversion state indices out of configuration bounds")
        if d_charges[di] - a_charges[ai] != 2:
            raise ValueError("metastable conversion states must differ by +2e")
        object.__setattr__(self, "identifier", ident)
        object.__setattr__(self, "total_density_m3", nt)

    def validate_band_gap(self, band_gap_eV: float) -> None:
        self.donor_configuration.validate_band_gap(band_gap_eV)
        self.acceptor_configuration.validate_band_gap(band_gap_eV)
        self.conversion_kinetics.validate_band_gap(band_gap_eV)


@dataclass(frozen=True, slots=True)
class MetastableConfigurationClosureResult:
    """Stationary donor/acceptor configuration fractions, rates, and tangents."""

    defect_identifier: str
    total_density_m3: float
    donor_to_acceptor_rate_s1: NDArray[np.float64]
    acceptor_to_donor_rate_s1: NDArray[np.float64]
    total_conversion_rate_s1: NDArray[np.float64]
    donor_fraction: NDArray[np.float64]
    acceptor_fraction: NDArray[np.float64]
    donor_fraction_derivative_n_m3: NDArray[np.float64]
    donor_fraction_derivative_p_m3: NDArray[np.float64]
    donor_density_m3: NDArray[np.float64]
    acceptor_density_m3: NDArray[np.float64]
    maximum_stationary_residual_s1: float
    maximum_rate_s1: float

    def __post_init__(self) -> None:
        for name in (
            "donor_to_acceptor_rate_s1",
            "acceptor_to_donor_rate_s1",
            "total_conversion_rate_s1",
            "donor_fraction",
            "acceptor_fraction",
            "donor_fraction_derivative_n_m3",
            "donor_fraction_derivative_p_m3",
            "donor_density_m3",
            "acceptor_density_m3",
        ):
            object.__setattr__(self, name, _ro_float(getattr(self, name)))


def evaluate_metastable_configuration_closure(
    electron_density_m3: ArrayLike,
    hole_density_m3: ArrayLike,
    spec: MetastableDefectSpec,
    *,
    band_gap_eV: ArrayLike,
    effective_conduction_dos_m3: ArrayLike,
    effective_valence_dos_m3: ArrayLike,
    temperature_K: float,
) -> MetastableConfigurationClosureResult:
    """Evaluate stationary ``y* = k_AD / (k_DA + k_AD)`` and exact carrier tangents."""
    n, p = np.broadcast_arrays(
        np.asarray(electron_density_m3, dtype=np.float64),
        np.asarray(hole_density_m3, dtype=np.float64),
    )
    if np.any(~np.isfinite(n)) or np.any(~np.isfinite(p)) or np.any(n <= 0.0) or np.any(p <= 0.0):
        raise ValueError("metastable closure requires finite positive carrier densities")
    gap = np.broadcast_to(np.asarray(band_gap_eV, dtype=np.float64), n.shape)
    nc = np.broadcast_to(
        np.asarray(effective_conduction_dos_m3, dtype=np.float64), n.shape
    )
    nv = np.broadcast_to(
        np.asarray(effective_valence_dos_m3, dtype=np.float64), n.shape
    )
    spec.validate_band_gap(float(np.min(gap)))
    vt = thermal_voltage(_finite_positive(temperature_K, "temperature_K"))
    kin = spec.conversion_kinetics

    red_n = n / nc
    red_p = p / nv
    e_act = (
        red_n * red_n
        if kin.electron_capture_path == DOUBLE_ELECTRON_CAPTURE
        else red_n / red_p
    )
    h_act = (
        red_p * red_p
        if kin.hole_capture_path == DOUBLE_HOLE_CAPTURE
        else red_p / red_n
    )

    e_pref = kin.capture_n_m3_s * nc
    h_pref = kin.capture_p_m3_s * nv
    e_cap = e_pref * e_act * math.exp(-kin.electron_capture_activation_eV / vt)
    e_em = e_pref * math.exp(-kin.electron_emission_activation_eV / vt)
    h_cap = h_pref * h_act * math.exp(-kin.hole_capture_activation_eV / vt)
    h_em = h_pref * math.exp(-kin.hole_emission_activation_eV / vt)

    k_da = e_cap + h_em
    k_ad = h_cap + e_em
    k_tot = k_da + k_ad
    if np.any(~np.isfinite(k_tot)) or np.any(k_da <= 0.0) or np.any(k_ad <= 0.0):
        raise FloatingPointError("metastable conversion rates must be finite and positive")
    peak_rate = float(np.max(k_tot))
    if peak_rate > kin.phonon_frequency_Hz:
        raise ValueError(
            f"metastable conversion rate {peak_rate:.6g} Hz exceeds phonon attempt "
            f"frequency {kin.phonon_frequency_Hz:.6g} Hz"
        )

    y_star = k_ad / k_tot
    one_minus_y = k_da / k_tot
    res = np.abs(k_ad * one_minus_y - k_da * y_star)

    if kin.electron_capture_path == DOUBLE_ELECTRON_CAPTURE:
        de_dn = 2.0 * e_cap / n
        de_dp = np.zeros_like(n)
    else:
        de_dn = e_cap / n
        de_dp = -e_cap / p
    if kin.hole_capture_path == DOUBLE_HOLE_CAPTURE:
        dh_dp = 2.0 * h_cap / p
        dh_dn = np.zeros_like(n)
    else:
        dh_dp = h_cap / p
        dh_dn = -h_cap / n

    dy_dn = (dh_dn * k_tot - k_ad * (de_dn + dh_dn)) / (k_tot * k_tot)
    dy_dp = (dh_dp * k_tot - k_ad * (de_dp + dh_dp)) / (k_tot * k_tot)
    nt = spec.total_density_m3

    return MetastableConfigurationClosureResult(
        defect_identifier=spec.identifier,
        total_density_m3=nt,
        donor_to_acceptor_rate_s1=k_da,
        acceptor_to_donor_rate_s1=k_ad,
        total_conversion_rate_s1=k_tot,
        donor_fraction=y_star,
        acceptor_fraction=one_minus_y,
        donor_fraction_derivative_n_m3=dy_dn,
        donor_fraction_derivative_p_m3=dy_dp,
        donor_density_m3=nt * y_star,
        acceptor_density_m3=nt * one_minus_y,
        maximum_stationary_residual_s1=float(np.max(res)),
        maximum_rate_s1=peak_rate,
    )


def configuration_logit(donor_fraction: ArrayLike) -> NDArray[np.float64]:
    """Unbounded logit coordinate ``log(y / (1 - y))`` without clipping."""
    y = np.asarray(donor_fraction, dtype=np.float64)
    if np.any(~np.isfinite(y)) or np.any((y <= 0.0) | (y >= 1.0)):
        raise ValueError("metastable donor_fraction must lie strictly inside (0, 1)")
    return _ro_float(np.log(y) - np.log1p(-y))


def configuration_from_logit(logit: ArrayLike) -> NDArray[np.float64]:
    """Map logit coordinate back to ``y in (0, 1)`` without clipping."""
    x = np.asarray(logit, dtype=np.float64)
    if np.any(~np.isfinite(x)):
        raise ValueError("metastable logit must be finite")
    y = stable_expit(x)
    if np.any((y <= 0.0) | (y >= 1.0)):
        raise ValueError("metastable logit saturated outside (0, 1)")
    return _ro_float(y)


@dataclass(frozen=True, slots=True)
class FrozenMetastableRegion:
    """Metastable bulk-defect region evaluated at a fixed donor fraction profile ``y``."""

    identifier: str
    document_sha256: str
    active_nodes: NDArray[np.bool_]
    band_gap_eV: NDArray[np.float64]
    effective_conduction_dos_m3: NDArray[np.float64]
    effective_valence_dos_m3: NDArray[np.float64]
    temperature_K: float
    species: tuple[MetastableDefectSpec, ...]
    donor_fractions: NDArray[np.float64]  # shape: (len(species), active_node_count)
    frozen: bool = True

    def __post_init__(self) -> None:
        mask = _ro_bool(self.active_nodes)
        if mask.ndim != 1 or not np.any(mask):
            raise ValueError("active_nodes must be a non-empty 1D boolean mask")
        active_count = int(np.sum(mask))
        sp = tuple(self.species)
        if not sp:
            raise ValueError("FrozenMetastableRegion requires at least one species")
        y = _ro_float(self.donor_fractions)
        if y.shape != (len(sp), active_count) or np.any(~np.isfinite(y)) or np.any((y < 0.0) | (y > 1.0)):
            raise ValueError(
                "donor_fractions must have shape (len(species), active_count) in [0, 1]"
            )
        gap = _ro_float(np.broadcast_to(self.band_gap_eV, (active_count,)))
        nc = _ro_float(
            np.broadcast_to(self.effective_conduction_dos_m3, (active_count,))
        )
        nv = _ro_float(
            np.broadcast_to(self.effective_valence_dos_m3, (active_count,))
        )
        temp = _finite_positive(self.temperature_K, "temperature_K")
        min_gap = float(np.min(gap))
        for item in sp:
            item.validate_band_gap(min_gap)
        object.__setattr__(self, "active_nodes", mask)
        object.__setattr__(self, "band_gap_eV", gap)
        object.__setattr__(self, "effective_conduction_dos_m3", nc)
        object.__setattr__(self, "effective_valence_dos_m3", nv)
        object.__setattr__(self, "temperature_K", temp)
        object.__setattr__(self, "species", sp)
        object.__setattr__(self, "donor_fractions", y)


@dataclass(frozen=True, slots=True)
class FrozenMetastableEvaluation:
    """Grid-aligned charge, SRH recombination, and exact tangents at fixed ``y``."""

    donor_mean_charge_e: NDArray[np.float64]  # (len(species), active_count)
    acceptor_mean_charge_e: NDArray[np.float64]  # (len(species), active_count)
    total_charge_density_C_m3: NDArray[np.float64]  # (node_count,)
    total_charge_derivative_n_C: NDArray[np.float64]
    total_charge_derivative_p_C: NDArray[np.float64]
    total_charge_derivative_fixed_qf_C_m3_V: NDArray[np.float64]
    total_recombination_rate_m3_s: NDArray[np.float64]
    total_recombination_derivative_n_s1: NDArray[np.float64]
    total_recombination_derivative_p_s1: NDArray[np.float64]

    def __post_init__(self) -> None:
        for name in (
            "donor_mean_charge_e",
            "acceptor_mean_charge_e",
            "total_charge_density_C_m3",
            "total_charge_derivative_n_C",
            "total_charge_derivative_p_C",
            "total_charge_derivative_fixed_qf_C_m3_V",
            "total_recombination_rate_m3_s",
            "total_recombination_derivative_n_s1",
            "total_recombination_derivative_p_s1",
        ):
            object.__setattr__(self, name, _ro_float(getattr(self, name)))


def evaluate_frozen_metastable_region(
    region: FrozenMetastableRegion,
    electron_density_m3: ArrayLike,
    hole_density_m3: ArrayLike,
) -> FrozenMetastableEvaluation:
    """Evaluate donor + acceptor multivalent branches weighted by frozen ``y``."""
    n = np.asarray(electron_density_m3, dtype=np.float64)
    p = np.asarray(hole_density_m3, dtype=np.float64)
    node_count = region.active_nodes.size
    if n.shape != (node_count,) or p.shape != (node_count,):
        raise ValueError("carrier vectors must match region.active_nodes")
    mask = region.active_nodes
    n_loc = n[mask]
    p_loc = p[mask]
    active_count = n_loc.size
    s_count = len(region.species)

    q_don = np.zeros((s_count, active_count), dtype=np.float64)
    q_acc = np.zeros((s_count, active_count), dtype=np.float64)
    tot_rho = np.zeros(node_count, dtype=np.float64)
    tot_drho_dn = np.zeros(node_count, dtype=np.float64)
    tot_drho_dp = np.zeros(node_count, dtype=np.float64)
    tot_drho_dphi = np.zeros(node_count, dtype=np.float64)
    tot_r = np.zeros(node_count, dtype=np.float64)
    tot_dr_dn = np.zeros(node_count, dtype=np.float64)
    tot_dr_dp = np.zeros(node_count, dtype=np.float64)

    for idx, sp in enumerate(region.species):
        y = region.donor_fractions[idx]
        one_minus_y = 1.0 - y
        # Evaluate unit-density multivalent branches, then scale by N_t * y and N_t * (1 - y):
        don_sp = MultivalentSpeciesSpec(
            f"{sp.identifier}:donor", sp.total_density_m3, sp.donor_configuration
        )
        acc_sp = MultivalentSpeciesSpec(
            f"{sp.identifier}:acceptor", sp.total_density_m3, sp.acceptor_configuration
        )
        d_res = evaluate_multivalent_defect_closure(
            n_loc,
            p_loc,
            don_sp,
            band_gap_eV=region.band_gap_eV,
            effective_conduction_dos_m3=region.effective_conduction_dos_m3,
            effective_valence_dos_m3=region.effective_valence_dos_m3,
            temperature_K=region.temperature_K,
        )
        a_res = evaluate_multivalent_defect_closure(
            n_loc,
            p_loc,
            acc_sp,
            band_gap_eV=region.band_gap_eV,
            effective_conduction_dos_m3=region.effective_conduction_dos_m3,
            effective_valence_dos_m3=region.effective_valence_dos_m3,
            temperature_K=region.temperature_K,
        )
        q_don[idx] = d_res.mean_charge_e
        q_acc[idx] = a_res.mean_charge_e

        tot_rho[mask] += y * d_res.charge_density_C_m3 + one_minus_y * a_res.charge_density_C_m3
        tot_drho_dn[mask] += y * d_res.charge_derivative_n_C + one_minus_y * a_res.charge_derivative_n_C
        tot_drho_dp[mask] += y * d_res.charge_derivative_p_C + one_minus_y * a_res.charge_derivative_p_C
        tot_drho_dphi[mask] += (
            y * d_res.charge_derivative_fixed_qf_C_m3_V
            + one_minus_y * a_res.charge_derivative_fixed_qf_C_m3_V
        )
        tot_r[mask] += (
            y * d_res.total_recombination_rate_m3_s
            + one_minus_y * a_res.total_recombination_rate_m3_s
        )
        tot_dr_dn[mask] += (
            y * d_res.total_recombination_derivative_n_s1
            + one_minus_y * a_res.total_recombination_derivative_n_s1
        )
        tot_dr_dp[mask] += (
            y * d_res.total_recombination_derivative_p_s1
            + one_minus_y * a_res.total_recombination_derivative_p_s1
        )

    return FrozenMetastableEvaluation(
        donor_mean_charge_e=q_don,
        acceptor_mean_charge_e=q_acc,
        total_charge_density_C_m3=tot_rho,
        total_charge_derivative_n_C=tot_drho_dn,
        total_charge_derivative_p_C=tot_drho_dp,
        total_charge_derivative_fixed_qf_C_m3_V=tot_drho_dphi,
        total_recombination_rate_m3_s=tot_r,
        total_recombination_derivative_n_s1=tot_dr_dn,
        total_recombination_derivative_p_s1=tot_dr_dp,
    )


def solve_frozen_metastable_charge_neutrality(
    *,
    temperature_K: float,
    band_gap_eV: float,
    effective_conduction_dos_m3: float,
    effective_valence_dos_m3: float,
    donor_density_m3: float,
    acceptor_density_m3: float,
    species: Sequence[MetastableDefectSpec],
    donor_fractions: Sequence[float],
    electron_affinity_eV: float | None = None,
) -> MultivalentDefectNeutralityState:
    """Solve equilibrium contact/bulk charge neutrality at frozen donor fractions."""
    if len(species) != len(donor_fractions):
        raise ValueError("one donor_fraction is required per metastable species")
    mv_species: list[MultivalentSpeciesSpec] = []
    for sp, y in zip(species, donor_fractions):
        yf = float(y)
        if not 0.0 <= yf <= 1.0:
            raise ValueError("donor_fraction must lie in [0, 1]")
        if yf > 0.0:
            mv_species.append(
                MultivalentSpeciesSpec(
                    f"{sp.identifier}:donor",
                    sp.total_density_m3 * yf,
                    sp.donor_configuration,
                )
            )
        if 1.0 - yf > 0.0:
            mv_species.append(
                MultivalentSpeciesSpec(
                    f"{sp.identifier}:acceptor",
                    sp.total_density_m3 * (1.0 - yf),
                    sp.acceptor_configuration,
                )
            )
    return solve_multivalent_defect_charge_neutrality(
        temperature_K=temperature_K,
        band_gap_eV=band_gap_eV,
        effective_conduction_dos_m3=effective_conduction_dos_m3,
        effective_valence_dos_m3=effective_valence_dos_m3,
        donor_density_m3=donor_density_m3,
        acceptor_density_m3=acceptor_density_m3,
        species=mv_species,
        electron_affinity_eV=electron_affinity_eV,
    )


def configuration_charge_transfer_C_m3(
    total_density_m3: float,
    donor_mean_charge_e: ArrayLike,
    acceptor_mean_charge_e: ArrayLike,
    delta_donor_fraction: ArrayLike,
) -> NDArray[np.float64]:
    """Exact fixed-carrier space-charge transfer ``q N_t (<q_D> - <q_A>) Delta_y``."""
    nt = _finite_positive(total_density_m3, "total_density_m3")
    qd = np.asarray(donor_mean_charge_e, dtype=np.float64)
    qa = np.asarray(acceptor_mean_charge_e, dtype=np.float64)
    dy = np.asarray(delta_donor_fraction, dtype=np.float64)
    return _ro_float(Q * nt * (qd - qa) * dy)


@dataclass(frozen=True, slots=True)
class MetastableDynamicStep:
    """Exact frozen-rate exponential advance of ``y(t)`` over one interval ``dt_s``."""

    dt_s: float
    previous_donor_fraction: NDArray[np.float64]
    stationary_donor_fraction: NDArray[np.float64]
    next_donor_fraction: NDArray[np.float64]
    delta_donor_fraction: NDArray[np.float64]
    next_logit: NDArray[np.float64]
    total_conversion_rate_s1: NDArray[np.float64]
    clipping_used: bool = False

    def __post_init__(self) -> None:
        for name in (
            "previous_donor_fraction",
            "stationary_donor_fraction",
            "next_donor_fraction",
            "delta_donor_fraction",
            "next_logit",
            "total_conversion_rate_s1",
        ):
            object.__setattr__(self, name, _ro_float(getattr(self, name)))


def advance_metastable_configuration(
    previous_donor_fraction: ArrayLike,
    closure: MetastableConfigurationClosureResult,
    dt_s: float,
) -> MetastableDynamicStep:
    """Advance ``dy/dt = -k_DA y + k_AD (1 - y)`` analytically at frozen rates."""
    dt = _finite_positive(dt_s, "dt_s")
    y_prev = np.asarray(previous_donor_fraction, dtype=np.float64)
    y_star = closure.donor_fraction
    if y_prev.shape != y_star.shape:
        raise ValueError("previous_donor_fraction shape must match closure.donor_fraction")
    if np.any((y_prev <= 0.0) | (y_prev >= 1.0)):
        raise ValueError("previous_donor_fraction must lie strictly inside (0, 1)")

    k_tot = closure.total_conversion_rate_s1
    # Cancellation-safe increment: y_next - y_prev = (y_star - y_prev) * (-expm1(-k_tot * dt))
    decay_complement = -np.expm1(-k_tot * dt)
    delta_y = (y_star - y_prev) * decay_complement
    y_next = y_prev + delta_y
    if np.any((y_next <= 0.0) | (y_next >= 1.0)):
        raise ValueError(
            "metastable configuration advance saturated outside (0, 1) without clipping"
        )
    logit_next = configuration_logit(y_next)
    return MetastableDynamicStep(
        dt_s=dt,
        previous_donor_fraction=y_prev,
        stationary_donor_fraction=y_star,
        next_donor_fraction=y_next,
        delta_donor_fraction=delta_y,
        next_logit=logit_next,
        total_conversion_rate_s1=k_tot,
        clipping_used=False,
    )
