"""Monovalent single-level, energy-distributed, and dynamic trap constitutive laws.

Covers:
1. Quasi-steady (QSS) monovalent defect closure (neutral, acceptor, donor;
   single-level, continuous energy quadrature, and spatial density profiles)
   with exact analytical derivatives with respect to ``(n, p)`` and fixed
   quasi-Fermi potential ``phi``.
2. Equilibrium charge-neutrality solver including monovalent/distributed traps.
3. Dynamic bulk trap occupancy layout, unbounded logit coordinates,
   cancellation-free logistic secants, and non-equilibrium capture/emission +
   trap storage + electrostatic charge with exact analytical derivatives.
4. Dynamic two-sided heterointerface trap kinetics, storage, and equilibrium-
   referenced sheet charge ``Delta_sigma = -q * N_it * (f - f_eq)`` with exact
   analytical derivatives.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
import hashlib
import json
import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

from solarlab.contracts.units import ELEMENTARY_CHARGE_C as Q
from solarlab.physics.defects.distributions import (
    DEFAULT_DEFECT_ENERGY_QUADRATURE_ORDER,
    DIST_SINGLE_LEVEL,
    DefectDistributionSpec,
    DefectEnergyQuadrature,
    SpatialDensityProfileSpec,
    build_defect_energy_quadrature,
    evaluate_spatial_density_profile,
    validate_defect_energy_quadrature_order,
)
from solarlab.physics.statistics import (
    CHARGE_NEUTRALITY_RELATIVE_TOLERANCE,
    thermal_voltage,
)

TRANSITION_NEUTRAL = "neutral"
TRANSITION_ACCEPTOR = "acceptor"
TRANSITION_DONOR = "donor"
SUPPORTED_MONOVALENT_TRANSITIONS = frozenset(
    {TRANSITION_NEUTRAL, TRANSITION_ACCEPTOR, TRANSITION_DONOR}
)

__all__ = [
    "DynamicBulkTrapEvaluation",
    "DynamicBulkTrapLayout",
    "DynamicBulkTrapMode",
    "DynamicInterfaceTrapEvaluation",
    "DynamicInterfaceTrapSpec",
    "MonovalentDefectClosureResult",
    "MonovalentDefectNeutralityState",
    "MonovalentDefectRegion",
    "MonovalentSpeciesSpec",
    "SUPPORTED_MONOVALENT_TRANSITIONS",
    "TRANSITION_ACCEPTOR",
    "TRANSITION_DONOR",
    "TRANSITION_NEUTRAL",
    "bulk_trap_charge_density",
    "compile_dynamic_bulk_trap_layout",
    "evaluate_dynamic_bulk_traps",
    "evaluate_dynamic_bulk_traps_about_qss",
    "evaluate_dynamic_interface_trap",
    "evaluate_monovalent_defect_closure",
    "evaluate_monovalent_region_closure",
    "logit_occupancy_secant",
    "occupancy_from_logit_increment",
    "occupancy_logit",
    "one_minus_occupancy_from_logit_increment",
    "quasi_steady_bulk_trap_logit",
    "quasi_steady_bulk_trap_occupancy",
    "solve_monovalent_defect_charge_neutrality",
    "stable_expit",
]


def _ro_float(values: ArrayLike) -> NDArray[np.float64]:
    arr = np.array(values, dtype=np.float64, copy=True)
    arr.setflags(write=False)
    return arr


def _ro_int(values: ArrayLike) -> NDArray[np.int64]:
    arr = np.array(values, dtype=np.int64, copy=True)
    arr.setflags(write=False)
    return arr


def _ro_bool(values: ArrayLike) -> NDArray[np.bool_]:
    arr = np.array(values, dtype=np.bool_, copy=True)
    arr.setflags(write=False)
    return arr


def _canonical_sha256(payload: object) -> str:
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return hashlib.sha256(raw.encode("ascii")).hexdigest()


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


def stable_expit(logit: ArrayLike) -> NDArray[np.float64]:
    """Overflow-safe logistic function ``1 / (1 + exp(-x))``."""
    x = np.asarray(logit, dtype=np.float64)
    out = np.empty_like(x)
    pos = x >= 0.0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    return out


def logit_occupancy_secant(
    new_logit: ArrayLike, old_logit: ArrayLike
) -> NDArray[np.float64]:
    """Cancellation-free ``expit(new_logit) - expit(old_logit)``."""
    ln = np.asarray(new_logit, dtype=np.float64)
    lo = np.asarray(old_logit, dtype=np.float64)
    if ln.shape != lo.shape:
        raise ValueError("new_logit and old_logit must have identical shape")
    d = ln - lo
    out = np.empty_like(d)
    nonneg = d >= 0.0
    if np.any(nonneg):
        f_new = stable_expit(ln[nonneg])
        one_minus_old = stable_expit(-lo[nonneg])
        out[nonneg] = f_new * one_minus_old * (-np.expm1(-d[nonneg]))
    neg = ~nonneg
    if np.any(neg):
        one_minus_new = stable_expit(-ln[neg])
        f_old = stable_expit(lo[neg])
        out[neg] = one_minus_new * f_old * np.expm1(d[neg])
    return out


@dataclass(frozen=True, slots=True)
class MonovalentSpeciesSpec:
    """Canonical monovalent defect species (single-level or continuous distribution)."""

    identifier: str
    charge_transition: str
    distribution: DefectDistributionSpec
    sigma_n_m2: float
    sigma_p_m2: float
    thermal_velocity_n_m_s: float
    thermal_velocity_p_m_s: float
    degeneracy: float = 1.0
    spatial_profile: SpatialDensityProfileSpec | None = None

    def __post_init__(self) -> None:
        ident = str(self.identifier).strip()
        if not ident:
            raise ValueError("MonovalentSpeciesSpec requires a non-empty identifier")
        if self.charge_transition not in SUPPORTED_MONOVALENT_TRANSITIONS:
            raise ValueError(
                f"unsupported monovalent charge_transition: {self.charge_transition!r}"
            )
        sn = _finite_nonnegative(self.sigma_n_m2, "sigma_n_m2")
        sp = _finite_nonnegative(self.sigma_p_m2, "sigma_p_m2")
        vn = _finite_positive(self.thermal_velocity_n_m_s, "thermal_velocity_n_m_s")
        vp = _finite_positive(self.thermal_velocity_p_m_s, "thermal_velocity_p_m_s")
        deg = _finite_positive(self.degeneracy, "degeneracy")
        if deg != 1.0:
            raise ValueError(
                "monovalent DEF-2 species require unit degeneracy (degeneracy == 1.0); "
                "use MultivalentSpeciesSpec for explicit state degeneracies"
            )
        if sn == 0.0 and sp == 0.0:
            raise ValueError(
                f"monovalent defect {ident!r} has zero electron and hole capture cross sections"
            )
        object.__setattr__(self, "identifier", ident)
        object.__setattr__(self, "sigma_n_m2", sn)
        object.__setattr__(self, "sigma_p_m2", sp)
        object.__setattr__(self, "thermal_velocity_n_m_s", vn)
        object.__setattr__(self, "thermal_velocity_p_m_s", vp)
        object.__setattr__(self, "degeneracy", deg)

    @property
    def capture_n_m3_s(self) -> float:
        return self.sigma_n_m2 * self.thermal_velocity_n_m_s

    @property
    def capture_p_m3_s(self) -> float:
        return self.sigma_p_m2 * self.thermal_velocity_p_m_s


@dataclass(frozen=True, slots=True)
class MonovalentDefectClosureResult:
    """Exact QSS occupancy, SRH rate, space charge, and analytical tangents."""

    species_identifiers: tuple[str, ...]
    charge_transitions: tuple[str, ...]
    occupancy: NDArray[np.float64]
    occupancy_derivative_n_m3: NDArray[np.float64]
    occupancy_derivative_p_m3: NDArray[np.float64]
    occupied_density_m3: NDArray[np.float64]
    charge_density_C_m3: NDArray[np.float64]
    charge_derivative_n_C: NDArray[np.float64]
    charge_derivative_p_C: NDArray[np.float64]
    charge_derivative_fixed_qf_C_m3_V: NDArray[np.float64]
    recombination_rate_m3_s: NDArray[np.float64]
    recombination_derivative_n_s1: NDArray[np.float64]
    recombination_derivative_p_s1: NDArray[np.float64]
    recombination_derivative_fixed_qf_m3_s_V: NDArray[np.float64]
    total_charge_density_C_m3: NDArray[np.float64]
    total_charge_derivative_n_C: NDArray[np.float64]
    total_charge_derivative_p_C: NDArray[np.float64]
    total_charge_derivative_fixed_qf_C_m3_V: NDArray[np.float64]
    total_recombination_rate_m3_s: NDArray[np.float64]
    total_recombination_derivative_n_s1: NDArray[np.float64]
    total_recombination_derivative_p_s1: NDArray[np.float64]
    total_recombination_derivative_fixed_qf_m3_s_V: NDArray[np.float64]
    kinetic_denominator_s1: NDArray[np.float64]

    def __post_init__(self) -> None:
        for name in (
            "occupancy",
            "occupancy_derivative_n_m3",
            "occupancy_derivative_p_m3",
            "occupied_density_m3",
            "charge_density_C_m3",
            "charge_derivative_n_C",
            "charge_derivative_p_C",
            "charge_derivative_fixed_qf_C_m3_V",
            "recombination_rate_m3_s",
            "recombination_derivative_n_s1",
            "recombination_derivative_p_s1",
            "recombination_derivative_fixed_qf_m3_s_V",
            "total_charge_density_C_m3",
            "total_charge_derivative_n_C",
            "total_charge_derivative_p_C",
            "total_charge_derivative_fixed_qf_C_m3_V",
            "total_recombination_rate_m3_s",
            "total_recombination_derivative_n_s1",
            "total_recombination_derivative_p_s1",
            "total_recombination_derivative_fixed_qf_m3_s_V",
            "kinetic_denominator_s1",
        ):
            object.__setattr__(self, name, _ro_float(getattr(self, name)))


def evaluate_monovalent_defect_closure(
    electron_density_m3: ArrayLike,
    hole_density_m3: ArrayLike,
    species: Sequence[MonovalentSpeciesSpec],
    *,
    band_gap_eV: ArrayLike,
    effective_conduction_dos_m3: ArrayLike,
    effective_valence_dos_m3: ArrayLike,
    temperature_K: float,
    energy_quadrature_order: int = DEFAULT_DEFECT_ENERGY_QUADRATURE_ORDER,
    normalized_layer_positions: ArrayLike | None = None,
) -> MonovalentDefectClosureResult:
    """Evaluate QSS monovalent/distributed/spatial defect closure and exact tangents."""
    if not species:
        raise ValueError("species sequence must not be empty")
    n, p = np.broadcast_arrays(
        np.asarray(electron_density_m3, dtype=np.float64),
        np.asarray(hole_density_m3, dtype=np.float64),
    )
    if np.any(~np.isfinite(n)) or np.any(~np.isfinite(p)) or np.any(n < 0.0) or np.any(p < 0.0):
        raise ValueError("carrier densities must be finite and non-negative")
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
    order = validate_defect_energy_quadrature_order(energy_quadrature_order)

    min_gap = float(np.min(gap))
    s_count = len(species)
    out_shape = (s_count, *n.shape)
    occ_arr = np.zeros(out_shape, dtype=np.float64)
    docc_dn_arr = np.zeros(out_shape, dtype=np.float64)
    docc_dp_arr = np.zeros(out_shape, dtype=np.float64)
    occ_dens_arr = np.zeros(out_shape, dtype=np.float64)
    rho_arr = np.zeros(out_shape, dtype=np.float64)
    drho_dn_arr = np.zeros(out_shape, dtype=np.float64)
    drho_dp_arr = np.zeros(out_shape, dtype=np.float64)
    drho_dphi_arr = np.zeros(out_shape, dtype=np.float64)
    rate_arr = np.zeros(out_shape, dtype=np.float64)
    drate_dn_arr = np.zeros(out_shape, dtype=np.float64)
    drate_dp_arr = np.zeros(out_shape, dtype=np.float64)
    drate_dphi_arr = np.zeros(out_shape, dtype=np.float64)
    denom_min_arr = np.zeros(out_shape, dtype=np.float64)

    ni_sq = nc * nv * np.exp(-gap / vt)
    excess = n * p - ni_sq

    for idx, item in enumerate(species):
        quad = build_defect_energy_quadrature(
            item.distribution, order=order, band_gap_eV=min_gap
        )
        if item.spatial_profile is not None:
            if normalized_layer_positions is None:
                raise ValueError(
                    f"species {item.identifier!r} has a spatial_profile, so "
                    "normalized_layer_positions must be provided"
                )
            mult = np.broadcast_to(
                evaluate_spatial_density_profile(
                    item.spatial_profile, normalized_layer_positions
                ),
                n.shape,
            )
        else:
            mult = np.ones(n.shape, dtype=np.float64)

        cn = item.capture_n_m3_s
        cp = item.capture_p_m3_s
        # Broadcast over quadrature axis 0 and spatial axes 1..
        energies = quad.energies_eV_above_vb.reshape((quad.order,) + (1,) * n.ndim)
        weights_norm = quad.normalized_weights.reshape((quad.order,) + (1,) * n.ndim)
        dens_k = quad.density_weights_m3.reshape((quad.order,) + (1,) * n.ndim) * mult[None, ...]

        n_e = n[None, ...]
        p_e = p[None, ...]
        nc_e = nc[None, ...]
        nv_e = nv[None, ...]
        gap_e = gap[None, ...]
        excess_e = excess[None, ...]

        n1 = nc_e * np.exp(-(gap_e - energies) / vt)
        p1 = nv_e * np.exp(-energies / vt)
        filled = cn * n_e + cp * p1
        empty = cn * n1 + cp * p_e
        denom = filled + empty
        if np.any(~np.isfinite(denom)) or np.any(denom <= 0.0):
            raise FloatingPointError(
                f"non-positive or non-finite kinetic denominator for {item.identifier!r}"
            )
        f_k = filled / denom
        one_minus_f_k = empty / denom
        df_dn_k = cn * one_minus_f_k / denom
        df_dp_k = -cp * f_k / denom

        pref = dens_k * cn * cp
        r_k = pref * excess_e / denom
        dr_dn_k = pref * (p_e * denom - excess_e * cn) / (denom * denom)
        dr_dp_k = pref * (n_e * denom - excess_e * cp) / (denom * denom)

        occ_arr[idx] = np.sum(weights_norm * f_k, axis=0)
        docc_dn_arr[idx] = np.sum(weights_norm * df_dn_k, axis=0)
        docc_dp_arr[idx] = np.sum(weights_norm * df_dp_k, axis=0)
        occ_dens_arr[idx] = np.sum(dens_k * f_k, axis=0)
        rate_arr[idx] = np.sum(r_k, axis=0)
        drate_dn_arr[idx] = np.sum(dr_dn_k, axis=0)
        drate_dp_arr[idx] = np.sum(dr_dp_k, axis=0)
        drate_dphi_arr[idx] = (
            drate_dn_arr[idx] * (n / vt) - drate_dp_arr[idx] * (p / vt)
        )
        denom_min_arr[idx] = np.min(denom, axis=0)

        if item.charge_transition == TRANSITION_ACCEPTOR:
            rho_arr[idx] = -Q * occ_dens_arr[idx]
            drho_dn_arr[idx] = -Q * np.sum(dens_k * df_dn_k, axis=0)
            drho_dp_arr[idx] = -Q * np.sum(dens_k * df_dp_k, axis=0)
            drho_dphi_arr[idx] = (
                drho_dn_arr[idx] * (n / vt) - drho_dp_arr[idx] * (p / vt)
            )
        elif item.charge_transition == TRANSITION_DONOR:
            rho_arr[idx] = Q * np.sum(dens_k * one_minus_f_k, axis=0)
            drho_dn_arr[idx] = -Q * np.sum(dens_k * df_dn_k, axis=0)
            drho_dp_arr[idx] = -Q * np.sum(dens_k * df_dp_k, axis=0)
            drho_dphi_arr[idx] = (
                drho_dn_arr[idx] * (n / vt) - drho_dp_arr[idx] * (p / vt)
            )

    return MonovalentDefectClosureResult(
        species_identifiers=tuple(item.identifier for item in species),
        charge_transitions=tuple(item.charge_transition for item in species),
        occupancy=occ_arr,
        occupancy_derivative_n_m3=docc_dn_arr,
        occupancy_derivative_p_m3=docc_dp_arr,
        occupied_density_m3=occ_dens_arr,
        charge_density_C_m3=rho_arr,
        charge_derivative_n_C=drho_dn_arr,
        charge_derivative_p_C=drho_dp_arr,
        charge_derivative_fixed_qf_C_m3_V=drho_dphi_arr,
        recombination_rate_m3_s=rate_arr,
        recombination_derivative_n_s1=drate_dn_arr,
        recombination_derivative_p_s1=drate_dp_arr,
        recombination_derivative_fixed_qf_m3_s_V=drate_dphi_arr,
        total_charge_density_C_m3=np.sum(rho_arr, axis=0),
        total_charge_derivative_n_C=np.sum(drho_dn_arr, axis=0),
        total_charge_derivative_p_C=np.sum(drho_dp_arr, axis=0),
        total_charge_derivative_fixed_qf_C_m3_V=np.sum(drho_dphi_arr, axis=0),
        total_recombination_rate_m3_s=np.sum(rate_arr, axis=0),
        total_recombination_derivative_n_s1=np.sum(drate_dn_arr, axis=0),
        total_recombination_derivative_p_s1=np.sum(drate_dp_arr, axis=0),
        total_recombination_derivative_fixed_qf_m3_s_V=np.sum(drate_dphi_arr, axis=0),
        kinetic_denominator_s1=denom_min_arr,
    )


@dataclass(frozen=True, slots=True)
class MonovalentDefectRegion:
    """Compiled monovalent/distributed/spatial defect inventory on a disjoint region."""

    identifier: str
    document_sha256: str
    active_nodes: NDArray[np.bool_]
    band_gap_eV: NDArray[np.float64]
    effective_conduction_dos_m3: NDArray[np.float64]
    effective_valence_dos_m3: NDArray[np.float64]
    temperature_K: float
    species: tuple[MonovalentSpeciesSpec, ...]
    energy_quadrature_order: int = DEFAULT_DEFECT_ENERGY_QUADRATURE_ORDER
    normalized_layer_positions: NDArray[np.float64] | None = None
    quadratures: tuple[DefectEnergyQuadrature, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        ident = str(self.identifier).strip()
        if not ident:
            raise ValueError("region identifier must be non-empty")
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
        order = validate_defect_energy_quadrature_order(self.energy_quadrature_order)
        sp = tuple(self.species)
        if not sp:
            raise ValueError("MonovalentDefectRegion requires at least one species")
        min_gap = float(np.min(gap))
        quads = tuple(
            build_defect_energy_quadrature(
                item.distribution, order=order, band_gap_eV=min_gap
            )
            for item in sp
        )
        pos: NDArray[np.float64] | None = None
        if self.normalized_layer_positions is not None:
            pos = _ro_float(self.normalized_layer_positions)
            if pos.shape != (active_count,):
                raise ValueError(
                    "normalized_layer_positions must match active node count"
                )
        elif any(item.spatial_profile is not None for item in sp):
            raise ValueError("spatial defect profile requires normalized_layer_positions")

        object.__setattr__(self, "identifier", ident)
        object.__setattr__(self, "active_nodes", mask)
        object.__setattr__(self, "band_gap_eV", gap)
        object.__setattr__(self, "effective_conduction_dos_m3", nc)
        object.__setattr__(self, "effective_valence_dos_m3", nv)
        object.__setattr__(self, "temperature_K", temp)
        object.__setattr__(self, "species", sp)
        object.__setattr__(self, "energy_quadrature_order", order)
        object.__setattr__(self, "normalized_layer_positions", pos)
        object.__setattr__(self, "quadratures", quads)

    @property
    def has_distributed_species(self) -> bool:
        return any(item.distribution.kind != DIST_SINGLE_LEVEL for item in self.species)

    @property
    def has_spatial_profiles(self) -> bool:
        return any(item.spatial_profile is not None for item in self.species)


def evaluate_monovalent_region_closure(
    region: MonovalentDefectRegion,
    electron_density_m3: ArrayLike,
    hole_density_m3: ArrayLike,
) -> MonovalentDefectClosureResult:
    """Evaluate a compiled ``MonovalentDefectRegion`` on full-grid carrier vectors."""
    n = np.asarray(electron_density_m3, dtype=np.float64)
    p = np.asarray(hole_density_m3, dtype=np.float64)
    if n.shape != region.active_nodes.shape or p.shape != region.active_nodes.shape:
        raise ValueError("carrier vectors must match region.active_nodes shape")
    return evaluate_monovalent_defect_closure(
        n[region.active_nodes],
        p[region.active_nodes],
        region.species,
        band_gap_eV=region.band_gap_eV,
        effective_conduction_dos_m3=region.effective_conduction_dos_m3,
        effective_valence_dos_m3=region.effective_valence_dos_m3,
        temperature_K=region.temperature_K,
        energy_quadrature_order=region.energy_quadrature_order,
        normalized_layer_positions=region.normalized_layer_positions,
    )


@dataclass(frozen=True, slots=True)
class MonovalentDefectNeutralityState:
    """Equilibrium contact/bulk charge-neutrality state with monovalent/distributed traps."""

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
    trap_charge_density_C_m3: float
    normalized_charge_residual: float
    work_function_eV: float | None = None


def solve_monovalent_defect_charge_neutrality(
    *,
    temperature_K: float,
    band_gap_eV: float,
    effective_conduction_dos_m3: float,
    effective_valence_dos_m3: float,
    donor_density_m3: float,
    acceptor_density_m3: float,
    species: Sequence[MonovalentSpeciesSpec],
    electron_affinity_eV: float | None = None,
    energy_quadrature_order: int = DEFAULT_DEFECT_ENERGY_QUADRATURE_ORDER,
    normalized_layer_position: float = 0.0,
) -> MonovalentDefectNeutralityState:
    """Solve MB equilibrium neutrality ``p - n + N_D - N_A + rho_trap/q == 0``.

    In thermal equilibrium ``n*p == n_i^2``, so every monovalent trap energy
    node satisfies exact Fermi-Dirac occupancy ``f(E) = 1 / (1 + exp((E - E_F)/kT))``,
    making total charge strictly decreasing in ``eta_n = (E_F - E_c) / V_T``.
    """
    temp = _finite_positive(temperature_K, "temperature_K")
    gap = _finite_positive(band_gap_eV, "band_gap_eV")
    nc = _finite_positive(effective_conduction_dos_m3, "effective_conduction_dos_m3")
    nv = _finite_positive(effective_valence_dos_m3, "effective_valence_dos_m3")
    nd = _finite_nonnegative(donor_density_m3, "donor_density_m3")
    na = _finite_nonnegative(acceptor_density_m3, "acceptor_density_m3")
    vt = thermal_voltage(temp)
    pos_arr = np.array([normalized_layer_position], dtype=np.float64)

    def eval_eta(eta_n: float) -> tuple[float, float, float, float, float]:
        eta_p = -gap / vt - eta_n
        n_val = nc * math.exp(eta_n)
        p_val = nv * math.exp(eta_p)
        if species:
            closure = evaluate_monovalent_defect_closure(
                np.array([n_val]),
                np.array([p_val]),
                species,
                band_gap_eV=gap,
                effective_conduction_dos_m3=nc,
                effective_valence_dos_m3=nv,
                temperature_K=temp,
                energy_quadrature_order=energy_quadrature_order,
                normalized_layer_positions=pos_arr,
            )
            rho_trap = float(closure.total_charge_density_C_m3[0])
        else:
            rho_trap = 0.0
        trap_num = rho_trap / Q
        net = (p_val + nd + max(trap_num, 0.0)) - (n_val + na + max(-trap_num, 0.0))
        scale = max(p_val + n_val + nd + na + abs(trap_num), 1.0)
        return net, scale, n_val, p_val, rho_trap

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
        raise RuntimeError("failed to bracket defect-charged equilibrium Fermi level")

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
            f"monovalent defect neutrality residual {norm_res:.3e} exceeded tolerance"
        )
    wf: float | None = None
    if electron_affinity_eV is not None:
        chi = _finite_positive(electron_affinity_eV, "electron_affinity_eV")
        wf = chi - vt * mid
    return MonovalentDefectNeutralityState(
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
        trap_charge_density_C_m3=rho_eq,
        normalized_charge_residual=norm_res,
        work_function_eV=wf,
    )


@dataclass(frozen=True, slots=True)
class DynamicBulkTrapMode:
    """One spatially contiguous dynamic trap mode (one region/species/energy node)."""

    mode_id: str
    region_identifier: str
    source_identifier: str
    energy_index: int
    charge_transition: str
    device_node_indices: NDArray[np.int64]
    population_density_m3: NDArray[np.float64]
    capture_n_m3_s: float
    capture_p_m3_s: float
    n1_m3: NDArray[np.float64]
    p1_m3: NDArray[np.float64]
    intrinsic_product_m6: NDArray[np.float64]
    flat_slice: slice

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "device_node_indices", _ro_int(self.device_node_indices)
        )
        for name in (
            "population_density_m3",
            "n1_m3",
            "p1_m3",
            "intrinsic_product_m6",
        ):
            object.__setattr__(self, name, _ro_float(getattr(self, name)))

    @property
    def size(self) -> int:
        return int(self.device_node_indices.size)


@dataclass(frozen=True, slots=True)
class DynamicBulkTrapLayout:
    """Canonical ordering of dynamic bulk trap states across regions and energy nodes."""

    model_identity_sha256: str
    node_count: int
    modes: tuple[DynamicBulkTrapMode, ...]
    device_node_indices: NDArray[np.int64]
    region_identifiers: tuple[str, ...]
    source_identifiers: tuple[str, ...]
    energy_node_identifiers: tuple[str, ...]
    charge_transitions: tuple[str, ...]
    population_density_m3: NDArray[np.float64]
    capture_n_m3_s: NDArray[np.float64]
    capture_p_m3_s: NDArray[np.float64]
    n1_m3: NDArray[np.float64]
    p1_m3: NDArray[np.float64]
    intrinsic_product_m6: NDArray[np.float64]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "device_node_indices", _ro_int(self.device_node_indices)
        )
        for name in (
            "population_density_m3",
            "capture_n_m3_s",
            "capture_p_m3_s",
            "n1_m3",
            "p1_m3",
            "intrinsic_product_m6",
        ):
            object.__setattr__(self, name, _ro_float(getattr(self, name)))

    @property
    def size(self) -> int:
        return int(self.device_node_indices.size)

    @property
    def identity_sha256(self) -> str:
        return _canonical_sha256(
            {
                "model_identity_sha256": self.model_identity_sha256,
                "node_count": self.node_count,
                "device_node_indices": self.device_node_indices.tolist(),
                "energy_node_identifiers": list(self.energy_node_identifiers),
                "charge_transitions": list(self.charge_transitions),
                "population_density_m3": self.population_density_m3.tolist(),
                "capture_n_m3_s": self.capture_n_m3_s.tolist(),
                "capture_p_m3_s": self.capture_p_m3_s.tolist(),
                "n1_m3": self.n1_m3.tolist(),
                "p1_m3": self.p1_m3.tolist(),
            }
        )


def compile_dynamic_bulk_trap_layout(
    regions: Sequence[MonovalentDefectRegion],
    *,
    node_count: int,
    dynamic_node_mask: ArrayLike | None = None,
) -> DynamicBulkTrapLayout:
    """Compile disjoint ``MonovalentDefectRegion`` objects into a dynamic trap layout.

    By default, contact boundary nodes ``0`` and ``node_count - 1`` are excluded
    from dynamic trap DOFs because fixed-reservoir/Robin contact equations own
    the boundary carrier balances; interior nodes ``1..node_count-2`` carry
    explicit dynamic trap storage DOFs.
    """
    if node_count < 3:
        raise ValueError("dynamic bulk trap layout requires node_count >= 3")
    if dynamic_node_mask is None:
        dyn_mask: NDArray[np.bool_] = np.ones(node_count, dtype=bool)
        dyn_mask[0] = False
        dyn_mask[-1] = False
    else:
        dyn_mask = np.asarray(dynamic_node_mask, dtype=bool)
        if dyn_mask.shape != (node_count,):
            raise ValueError("dynamic_node_mask must match node_count")

    modes: list[DynamicBulkTrapMode] = []
    dev_nodes_all: list[int] = []
    reg_ids_all: list[str] = []
    src_ids_all: list[str] = []
    e_ids_all: list[str] = []
    trans_all: list[str] = []
    pop_all: list[float] = []
    cn_all: list[float] = []
    cp_all: list[float] = []
    n1_all: list[float] = []
    p1_all: list[float] = []
    nisq_all: list[float] = []

    offset = 0
    region_digests: list[tuple[str, str]] = []
    for region in regions:
        if region.active_nodes.size != node_count:
            raise ValueError("region.active_nodes must match node_count")
        region_digests.append((region.identifier, region.document_sha256))
        vt = thermal_voltage(region.temperature_K)
        active_full_indices = np.flatnonzero(region.active_nodes)
        selected_local = np.flatnonzero(dyn_mask[active_full_indices])
        if selected_local.size == 0:
            continue
        sel_nodes = active_full_indices[selected_local]
        local_gap = region.band_gap_eV[selected_local]
        local_nc = region.effective_conduction_dos_m3[selected_local]
        local_nv = region.effective_valence_dos_m3[selected_local]
        local_nisq = local_nc * local_nv * np.exp(-local_gap / vt)

        for sp_idx, (sp, quad) in enumerate(zip(region.species, region.quadratures)):
            if sp.spatial_profile is not None:
                assert region.normalized_layer_positions is not None
                mult = evaluate_spatial_density_profile(
                    sp.spatial_profile,
                    region.normalized_layer_positions[selected_local],
                )
            else:
                mult = np.ones(selected_local.size, dtype=np.float64)
            cn = sp.capture_n_m3_s
            cp = sp.capture_p_m3_s

            for e_idx in range(quad.order):
                dens_base = float(quad.density_weights_m3[e_idx])
                pop_vec = dens_base * mult
                if np.all(pop_vec <= 0.0):
                    continue
                if np.any(pop_vec <= 0.0):
                    raise ValueError(
                        f"dynamic trap mode {region.identifier}/{sp.identifier}[{e_idx}] "
                        "has zero density on a subset of active nodes"
                    )
                energy = float(quad.energies_eV_above_vb[e_idx])
                n1_vec = local_nc * np.exp(-(local_gap - energy) / vt)
                p1_vec = local_nv * np.exp(-energy / vt)
                mode_id = (
                    f"{region.identifier}:{sp.identifier}"
                    if quad.order == 1
                    else f"{region.identifier}:{sp.identifier}:e{e_idx:03d}"
                )
                sl = slice(offset, offset + sel_nodes.size)
                offset += sel_nodes.size
                modes.append(
                    DynamicBulkTrapMode(
                        mode_id=mode_id,
                        region_identifier=region.identifier,
                        source_identifier=sp.identifier,
                        energy_index=e_idx,
                        charge_transition=sp.charge_transition,
                        device_node_indices=sel_nodes,
                        population_density_m3=pop_vec,
                        capture_n_m3_s=cn,
                        capture_p_m3_s=cp,
                        n1_m3=n1_vec,
                        p1_m3=p1_vec,
                        intrinsic_product_m6=local_nisq,
                        flat_slice=sl,
                    )
                )
                for k in range(sel_nodes.size):
                    dev_nodes_all.append(int(sel_nodes[k]))
                    reg_ids_all.append(region.identifier)
                    src_ids_all.append(sp.identifier)
                    e_ids_all.append(f"{mode_id}@{int(sel_nodes[k])}")
                    trans_all.append(sp.charge_transition)
                    pop_all.append(float(pop_vec[k]))
                    cn_all.append(cn)
                    cp_all.append(cp)
                    n1_all.append(float(n1_vec[k]))
                    p1_all.append(float(p1_vec[k]))
                    nisq_all.append(float(local_nisq[k]))

    if not modes:
        raise ValueError("compile_dynamic_bulk_trap_layout found zero active trap modes")

    model_sha = _canonical_sha256({"regions": region_digests, "node_count": node_count})
    return DynamicBulkTrapLayout(
        model_identity_sha256=model_sha,
        node_count=node_count,
        modes=tuple(modes),
        device_node_indices=np.asarray(dev_nodes_all, dtype=np.int64),
        region_identifiers=tuple(reg_ids_all),
        source_identifiers=tuple(src_ids_all),
        energy_node_identifiers=tuple(e_ids_all),
        charge_transitions=tuple(trans_all),
        population_density_m3=np.asarray(pop_all, dtype=np.float64),
        capture_n_m3_s=np.asarray(cn_all, dtype=np.float64),
        capture_p_m3_s=np.asarray(cp_all, dtype=np.float64),
        n1_m3=np.asarray(n1_all, dtype=np.float64),
        p1_m3=np.asarray(p1_all, dtype=np.float64),
        intrinsic_product_m6=np.asarray(nisq_all, dtype=np.float64),
    )


def quasi_steady_bulk_trap_occupancy(
    electron_density_m3: ArrayLike,
    hole_density_m3: ArrayLike,
    layout: DynamicBulkTrapLayout,
) -> NDArray[np.float64]:
    """Exact local QSS occupancy ``f_0 in (0, 1)`` in flat layout order."""
    n = np.asarray(electron_density_m3, dtype=np.float64)
    p = np.asarray(hole_density_m3, dtype=np.float64)
    if n.shape != (layout.node_count,) or p.shape != (layout.node_count,):
        raise ValueError("carrier densities must match layout.node_count")
    idx = layout.device_node_indices
    filled = layout.capture_n_m3_s * n[idx] + layout.capture_p_m3_s * layout.p1_m3
    empty = layout.capture_n_m3_s * layout.n1_m3 + layout.capture_p_m3_s * p[idx]
    denom = filled + empty
    if np.any(~np.isfinite(denom)) or np.any(denom <= 0.0):
        raise ValueError("dynamic bulk trap kinetic denominator must be positive")
    f = filled / denom
    if np.any((f <= 0.0) | (f >= 1.0)):
        raise ValueError("QSS trap occupancy saturated outside (0, 1)")
    return _ro_float(f)


def quasi_steady_bulk_trap_logit(
    electron_density_m3: ArrayLike,
    hole_density_m3: ArrayLike,
    layout: DynamicBulkTrapLayout,
) -> NDArray[np.float64]:
    """Cancellation-free QSS trap logit ``log(filled) - log(empty)`` in flat order."""
    n = np.asarray(electron_density_m3, dtype=np.float64)
    p = np.asarray(hole_density_m3, dtype=np.float64)
    if n.shape != (layout.node_count,) or p.shape != (layout.node_count,):
        raise ValueError("carrier densities must match layout.node_count")
    idx = layout.device_node_indices
    filled = layout.capture_n_m3_s * n[idx] + layout.capture_p_m3_s * layout.p1_m3
    empty = layout.capture_n_m3_s * layout.n1_m3 + layout.capture_p_m3_s * p[idx]
    if np.any(~np.isfinite(filled)) or np.any(~np.isfinite(empty)) or np.any(filled <= 0.0) or np.any(empty <= 0.0):
        raise ValueError("dynamic bulk trap capture/emission rates must be finite and positive")
    return _ro_float(np.log(filled) - np.log(empty))


def occupancy_logit(
    occupancy: ArrayLike, layout: DynamicBulkTrapLayout | None = None
) -> NDArray[np.float64]:
    """Map physical occupancy ``f in (0, 1)`` to unbounded logit ``log(f/(1-f))``."""
    f = np.asarray(occupancy, dtype=np.float64)
    if layout is not None and f.shape != (layout.size,):
        raise ValueError("occupancy shape does not match layout.size")
    if np.any(~np.isfinite(f)) or np.any((f <= 0.0) | (f >= 1.0)):
        raise ValueError("logit occupancy must be finite and strictly inside (0, 1)")
    return _ro_float(np.log(f) - np.log1p(-f))


def occupancy_from_logit_increment(
    reference_logit: ArrayLike,
    increment: ArrayLike,
    layout: DynamicBulkTrapLayout | None = None,
) -> NDArray[np.float64]:
    """Map ``reference_logit + increment`` back to physical occupancy ``f in (0, 1)``."""
    ref = np.asarray(reference_logit, dtype=np.float64)
    inc = np.asarray(increment, dtype=np.float64)
    if ref.shape != inc.shape:
        raise ValueError("reference_logit and increment must have matching shapes")
    if layout is not None and ref.shape != (layout.size,):
        raise ValueError("logit shape does not match layout.size")
    if np.any(~np.isfinite(ref)) or np.any(~np.isfinite(inc)):
        raise ValueError("logit values must be finite")
    f = stable_expit(ref + inc)
    return _ro_float(f)


def one_minus_occupancy_from_logit_increment(
    reference_logit: ArrayLike,
    increment: ArrayLike,
    layout: DynamicBulkTrapLayout | None = None,
) -> NDArray[np.float64]:
    """Cancellation-free ``1 - f = expit(-(reference_logit + increment))``."""
    ref = np.asarray(reference_logit, dtype=np.float64)
    inc = np.asarray(increment, dtype=np.float64)
    if ref.shape != inc.shape:
        raise ValueError("reference_logit and increment must have matching shapes")
    if layout is not None and ref.shape != (layout.size,):
        raise ValueError("logit shape does not match layout.size")
    if np.any(~np.isfinite(ref)) or np.any(~np.isfinite(inc)):
        raise ValueError("logit values must be finite")
    omf = stable_expit(-(ref + inc))
    return _ro_float(omf)


def bulk_trap_charge_density(
    occupancy: ArrayLike,
    layout: DynamicBulkTrapLayout,
    *,
    one_minus_occupancy: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """Aggregate dynamic bulk-trap space charge [C m^-3] onto the device grid."""
    f = np.asarray(occupancy, dtype=np.float64)
    if f.shape != (layout.size,):
        raise ValueError("occupancy must match layout.size")
    omf = (
        1.0 - f
        if one_minus_occupancy is None
        else np.asarray(one_minus_occupancy, dtype=np.float64)
    )
    trans = np.asarray(layout.charge_transitions)
    local_rho: NDArray[np.float64] = np.zeros(layout.size, dtype=np.float64)
    acc = trans == TRANSITION_ACCEPTOR
    don = trans == TRANSITION_DONOR
    local_rho[acc] = -Q * layout.population_density_m3[acc] * f[acc]
    local_rho[don] = Q * layout.population_density_m3[don] * omf[don]
    total: NDArray[np.float64] = np.zeros(layout.node_count, dtype=np.float64)
    np.add.at(total, layout.device_node_indices, local_rho)
    return _ro_float(total)


@dataclass(frozen=True, slots=True)
class DynamicBulkTrapEvaluation:
    """Non-equilibrium dynamic bulk-trap rates, storage, charge, and exact tangents."""

    occupancy: NDArray[np.float64]
    one_minus_occupancy: NDArray[np.float64]
    relaxation_rate_s1: NDArray[np.float64]
    occupied_storage_m3: NDArray[np.float64]
    trap_storage_rate_m3_s: NDArray[np.float64]
    electron_capture_rate_m3_s: NDArray[np.float64]
    hole_capture_rate_m3_s: NDArray[np.float64]
    charge_density_C_m3: NDArray[np.float64]
    # Exact analytical partial derivatives per flat trap state:
    d_electron_capture_dn_s1: NDArray[np.float64]
    d_electron_capture_df_m3_s: NDArray[np.float64]
    d_hole_capture_dp_s1: NDArray[np.float64]
    d_hole_capture_df_m3_s: NDArray[np.float64]
    d_trap_storage_rate_dn_s1: NDArray[np.float64]
    d_trap_storage_rate_dp_s1: NDArray[np.float64]
    d_trap_storage_rate_df_m3_s: NDArray[np.float64]
    d_charge_df_C_m3: NDArray[np.float64]
    # Grid-summed totals (shape == (node_count,)):
    total_electron_capture_rate_m3_s: NDArray[np.float64]
    total_hole_capture_rate_m3_s: NDArray[np.float64]
    total_trap_storage_rate_m3_s: NDArray[np.float64]
    total_charge_density_C_m3: NDArray[np.float64]


def evaluate_dynamic_bulk_traps(
    electron_density_m3: ArrayLike,
    hole_density_m3: ArrayLike,
    occupancy: ArrayLike,
    layout: DynamicBulkTrapLayout,
    *,
    one_minus_occupancy: ArrayLike | None = None,
) -> DynamicBulkTrapEvaluation:
    """Evaluate dynamic bulk trap kinetics, storage, charge, and exact derivatives."""
    n = np.asarray(electron_density_m3, dtype=np.float64)
    p = np.asarray(hole_density_m3, dtype=np.float64)
    f = np.asarray(occupancy, dtype=np.float64)
    if n.shape != (layout.node_count,) or p.shape != (layout.node_count,):
        raise ValueError("carrier vectors must match layout.node_count")
    if f.shape != (layout.size,) or np.any(~np.isfinite(f)) or np.any((f < 0.0) | (f > 1.0)):
        raise ValueError("occupancy must be finite in [0, 1] and match layout.size")
    omf = (
        1.0 - f
        if one_minus_occupancy is None
        else np.asarray(one_minus_occupancy, dtype=np.float64)
    )

    idx = layout.device_node_indices
    nt = layout.population_density_m3
    cn = layout.capture_n_m3_s
    cp = layout.capture_p_m3_s
    n_loc = n[idx]
    p_loc = p[idx]

    filled = cn * n_loc + cp * layout.p1_m3
    empty = cn * layout.n1_m3 + cp * p_loc
    relax = filled + empty
    offset = (f * empty - omf * filled) / relax

    qss_per_trap = cn * cp * (n_loc * p_loc - layout.intrinsic_product_m6) / relax
    cap_n = nt * (qss_per_trap - cn * (n_loc + layout.n1_m3) * offset)
    cap_p = nt * (qss_per_trap + cp * (p_loc + layout.p1_m3) * offset)
    storage_rate = -nt * relax * offset
    storage = nt * f

    trans = np.asarray(layout.charge_transitions)
    acc = trans == TRANSITION_ACCEPTOR
    don = trans == TRANSITION_DONOR
    rho: NDArray[np.float64] = np.zeros(layout.size, dtype=np.float64)
    d_rho_df: NDArray[np.float64] = np.zeros(layout.size, dtype=np.float64)
    rho[acc] = -Q * storage[acc]
    rho[don] = Q * nt[don] * omf[don]
    d_rho_df[acc | don] = -Q * nt[acc | don]

    d_cap_n_dn = nt * cn * omf
    d_cap_n_df = -nt * cn * (n_loc + layout.n1_m3)
    d_cap_p_dp = nt * cp * f
    d_cap_p_df = nt * cp * (p_loc + layout.p1_m3)
    d_sr_dn = d_cap_n_dn
    d_sr_dp = -d_cap_p_dp
    d_sr_df = -nt * relax

    tot_cn: NDArray[np.float64] = np.zeros(layout.node_count, dtype=np.float64)
    tot_cp: NDArray[np.float64] = np.zeros(layout.node_count, dtype=np.float64)
    tot_sr: NDArray[np.float64] = np.zeros(layout.node_count, dtype=np.float64)
    tot_rho: NDArray[np.float64] = np.zeros(layout.node_count, dtype=np.float64)
    np.add.at(tot_cn, idx, cap_n)
    np.add.at(tot_cp, idx, cap_p)
    np.add.at(tot_sr, idx, storage_rate)
    np.add.at(tot_rho, idx, rho)

    return DynamicBulkTrapEvaluation(
        occupancy=_ro_float(f),
        one_minus_occupancy=_ro_float(omf),
        relaxation_rate_s1=_ro_float(relax),
        occupied_storage_m3=_ro_float(storage),
        trap_storage_rate_m3_s=_ro_float(storage_rate),
        electron_capture_rate_m3_s=_ro_float(cap_n),
        hole_capture_rate_m3_s=_ro_float(cap_p),
        charge_density_C_m3=_ro_float(rho),
        d_electron_capture_dn_s1=_ro_float(d_cap_n_dn),
        d_electron_capture_df_m3_s=_ro_float(d_cap_n_df),
        d_hole_capture_dp_s1=_ro_float(d_cap_p_dp),
        d_hole_capture_df_m3_s=_ro_float(d_cap_p_df),
        d_trap_storage_rate_dn_s1=_ro_float(d_sr_dn),
        d_trap_storage_rate_dp_s1=_ro_float(d_sr_dp),
        d_trap_storage_rate_df_m3_s=_ro_float(d_sr_df),
        d_charge_df_C_m3=_ro_float(d_rho_df),
        total_electron_capture_rate_m3_s=_ro_float(tot_cn),
        total_hole_capture_rate_m3_s=_ro_float(tot_cp),
        total_trap_storage_rate_m3_s=_ro_float(tot_sr),
        total_charge_density_C_m3=_ro_float(tot_rho),
    )


def evaluate_dynamic_bulk_traps_about_qss(
    electron_density_m3: ArrayLike,
    hole_density_m3: ArrayLike,
    occupancy: ArrayLike,
    layout: DynamicBulkTrapLayout,
    *,
    reference_electron_density_m3: ArrayLike,
    reference_hole_density_m3: ArrayLike,
    reference_occupancy: ArrayLike,
    occupancy_increment: ArrayLike | None = None,
    one_minus_occupancy: ArrayLike | None = None,
) -> DynamicBulkTrapEvaluation:
    """Exact bilinear increment evaluation about a certified QSS reference state."""
    n = np.asarray(electron_density_m3, dtype=np.float64)
    p = np.asarray(hole_density_m3, dtype=np.float64)
    n0 = np.asarray(reference_electron_density_m3, dtype=np.float64)
    p0 = np.asarray(reference_hole_density_m3, dtype=np.float64)
    f = np.asarray(occupancy, dtype=np.float64)
    f0 = np.asarray(reference_occupancy, dtype=np.float64)
    df = f - f0 if occupancy_increment is None else np.asarray(occupancy_increment, dtype=np.float64)
    omf = (
        1.0 - f
        if one_minus_occupancy is None
        else np.asarray(one_minus_occupancy, dtype=np.float64)
    )

    idx = layout.device_node_indices
    nt = layout.population_density_m3
    cn = layout.capture_n_m3_s
    cp = layout.capture_p_m3_s
    n_loc, p_loc = n[idx], p[idx]
    n0_loc, p0_loc = n0[idx], p0[idx]
    dn = n_loc - n0_loc
    dp = p_loc - p0_loc

    relax = cn * (n_loc + layout.n1_m3) + cp * (p_loc + layout.p1_m3)
    relax0 = cn * (n0_loc + layout.n1_m3) + cp * (p0_loc + layout.p1_m3)
    qss0 = cn * cp * (n0_loc * p0_loc - layout.intrinsic_product_m6) / relax0

    cap_n = nt * (qss0 + cn * (1.0 - f0) * dn - cn * (n_loc + layout.n1_m3) * df)
    cap_p = nt * (qss0 + cp * f0 * dp + cp * (p_loc + layout.p1_m3) * df)
    storage_rate = nt * (cn * (1.0 - f0) * dn - cp * f0 * dp - relax * df)
    storage = nt * f

    trans = np.asarray(layout.charge_transitions)
    acc = trans == TRANSITION_ACCEPTOR
    don = trans == TRANSITION_DONOR
    rho: NDArray[np.float64] = np.zeros(layout.size, dtype=np.float64)
    d_rho_df: NDArray[np.float64] = np.zeros(layout.size, dtype=np.float64)
    rho[acc] = -Q * storage[acc]
    rho[don] = Q * nt[don] * omf[don]
    d_rho_df[acc | don] = -Q * nt[acc | don]

    d_cap_n_dn = nt * cn * omf
    d_cap_n_df = -nt * cn * (n_loc + layout.n1_m3)
    d_cap_p_dp = nt * cp * f
    d_cap_p_df = nt * cp * (p_loc + layout.p1_m3)
    d_sr_dn = d_cap_n_dn
    d_sr_dp = -d_cap_p_dp
    d_sr_df = -nt * relax

    tot_cn: NDArray[np.float64] = np.zeros(layout.node_count, dtype=np.float64)
    tot_cp: NDArray[np.float64] = np.zeros(layout.node_count, dtype=np.float64)
    tot_sr: NDArray[np.float64] = np.zeros(layout.node_count, dtype=np.float64)
    tot_rho: NDArray[np.float64] = np.zeros(layout.node_count, dtype=np.float64)
    np.add.at(tot_cn, idx, cap_n)
    np.add.at(tot_cp, idx, cap_p)
    np.add.at(tot_sr, idx, storage_rate)
    np.add.at(tot_rho, idx, rho)

    return DynamicBulkTrapEvaluation(
        occupancy=_ro_float(f),
        one_minus_occupancy=_ro_float(omf),
        relaxation_rate_s1=_ro_float(relax),
        occupied_storage_m3=_ro_float(storage),
        trap_storage_rate_m3_s=_ro_float(storage_rate),
        electron_capture_rate_m3_s=_ro_float(cap_n),
        hole_capture_rate_m3_s=_ro_float(cap_p),
        charge_density_C_m3=_ro_float(rho),
        d_electron_capture_dn_s1=_ro_float(d_cap_n_dn),
        d_electron_capture_df_m3_s=_ro_float(d_cap_n_df),
        d_hole_capture_dp_s1=_ro_float(d_cap_p_dp),
        d_hole_capture_df_m3_s=_ro_float(d_cap_p_df),
        d_trap_storage_rate_dn_s1=_ro_float(d_sr_dn),
        d_trap_storage_rate_dp_s1=_ro_float(d_sr_dp),
        d_trap_storage_rate_df_m3_s=_ro_float(d_sr_df),
        d_charge_df_C_m3=_ro_float(d_rho_df),
        total_electron_capture_rate_m3_s=_ro_float(tot_cn),
        total_hole_capture_rate_m3_s=_ro_float(tot_cp),
        total_trap_storage_rate_m3_s=_ro_float(tot_sr),
        total_charge_density_C_m3=_ro_float(tot_rho),
    )


@dataclass(frozen=True, slots=True)
class DynamicInterfaceTrapSpec:
    """Microscopic two-sided heterointerface trap parameters."""

    interface_id: str
    left_node: int
    right_node: int
    charge_type: str
    sheet_density_m2: float
    v_n_m_s: float
    v_p_m_s: float
    n1_left_m3: float
    p1_left_m3: float
    n1_right_m3: float
    p1_right_m3: float
    equilibrium_occupancy: float
    equilibrium_referenced_charge: bool = True

    def __post_init__(self) -> None:
        if self.right_node != self.left_node + 1 or self.left_node < 0:
            raise ValueError("DynamicInterfaceTrapSpec requires adjacent left/right nodes")
        if self.charge_type not in SUPPORTED_MONOVALENT_TRANSITIONS:
            raise ValueError(f"unsupported interface charge_type: {self.charge_type!r}")
        nt = _finite_positive(self.sheet_density_m2, "sheet_density_m2")
        vn = _finite_nonnegative(self.v_n_m_s, "v_n_m_s")
        vp = _finite_nonnegative(self.v_p_m_s, "v_p_m_s")
        if vn == 0.0 and vp == 0.0:
            raise ValueError("dynamic interface trap requires v_n > 0 or v_p > 0")
        feq = float(self.equilibrium_occupancy)
        if not math.isfinite(feq) or not 0.0 < feq < 1.0:
            raise ValueError("equilibrium_occupancy must lie strictly inside (0, 1)")
        object.__setattr__(self, "sheet_density_m2", nt)
        object.__setattr__(self, "v_n_m_s", vn)
        object.__setattr__(self, "v_p_m_s", vp)


@dataclass(frozen=True, slots=True)
class DynamicInterfaceTrapEvaluation:
    """Two-sided interface trap capture, storage, sheet charge, and exact tangents."""

    occupancy: float
    one_minus_occupancy: float
    qss_occupancy: float
    relaxation_rate_s1: float
    occupied_storage_m2: float
    trap_storage_rate_m2_s: float
    cap_n_left_m2_s: float
    cap_n_right_m2_s: float
    cap_p_left_m2_s: float
    cap_p_right_m2_s: float
    sheet_charge_C_m2: float
    # Exact partial derivatives wrt (n_L, p_L, n_R, p_R, f):
    d_cap_n_left_dn_L: float
    d_cap_n_left_df: float
    d_cap_n_right_dn_R: float
    d_cap_n_right_df: float
    d_cap_p_left_dp_L: float
    d_cap_p_left_df: float
    d_cap_p_right_dp_R: float
    d_cap_p_right_df: float
    d_storage_rate_dn_L: float
    d_storage_rate_dp_L: float
    d_storage_rate_dn_R: float
    d_storage_rate_dp_R: float
    d_storage_rate_df: float
    d_sheet_charge_df: float


def evaluate_dynamic_interface_trap(
    spec: DynamicInterfaceTrapSpec,
    *,
    n_left_m3: float,
    p_left_m3: float,
    n_right_m3: float,
    p_right_m3: float,
    occupancy: float,
    one_minus_occupancy: float | None = None,
    occupancy_minus_equilibrium: float | None = None,
) -> DynamicInterfaceTrapEvaluation:
    """Evaluate two-sided interface trap kinetics, sheet charge, and tangents."""
    f = float(occupancy)
    omf = 1.0 - f if one_minus_occupancy is None else float(one_minus_occupancy)
    vn = spec.v_n_m_s
    vp = spec.v_p_m_s
    nt = spec.sheet_density_m2

    filled = vn * (n_left_m3 + n_right_m3) + vp * (spec.p1_left_m3 + spec.p1_right_m3)
    empty = vn * (spec.n1_left_m3 + spec.n1_right_m3) + vp * (p_left_m3 + p_right_m3)
    denom = filled + empty
    if not math.isfinite(denom) or denom <= 0.0:
        raise FloatingPointError("dynamic interface trap kinetic denominator is invalid")
    f_qss = filled / denom
    relax = denom / nt

    cap_n_l = vn * (n_left_m3 * omf - spec.n1_left_m3 * f)
    cap_n_r = vn * (n_right_m3 * omf - spec.n1_right_m3 * f)
    cap_p_l = vp * (p_left_m3 * f - spec.p1_left_m3 * omf)
    cap_p_r = vp * (p_right_m3 * f - spec.p1_right_m3 * omf)
    # Exact sum (cap_n_l + cap_n_r) - (cap_p_l + cap_p_r) == filled - denom * f:
    storage_rate = -denom * (f - f_qss)

    if spec.charge_type == TRANSITION_NEUTRAL:
        sigma = 0.0
        d_sigma_df = 0.0
    elif spec.equilibrium_referenced_charge:
        df_eq = (
            f - spec.equilibrium_occupancy
            if occupancy_minus_equilibrium is None
            else float(occupancy_minus_equilibrium)
        )
        sigma = -Q * nt * df_eq
        d_sigma_df = -Q * nt
    elif spec.charge_type == TRANSITION_ACCEPTOR:
        sigma = -Q * nt * f
        d_sigma_df = -Q * nt
    else:
        sigma = Q * nt * omf
        d_sigma_df = -Q * nt

    return DynamicInterfaceTrapEvaluation(
        occupancy=f,
        one_minus_occupancy=omf,
        qss_occupancy=f_qss,
        relaxation_rate_s1=relax,
        occupied_storage_m2=nt * f,
        trap_storage_rate_m2_s=storage_rate,
        cap_n_left_m2_s=cap_n_l,
        cap_n_right_m2_s=cap_n_r,
        cap_p_left_m2_s=cap_p_l,
        cap_p_right_m2_s=cap_p_r,
        sheet_charge_C_m2=sigma,
        d_cap_n_left_dn_L=vn * omf,
        d_cap_n_left_df=-vn * (n_left_m3 + spec.n1_left_m3),
        d_cap_n_right_dn_R=vn * omf,
        d_cap_n_right_df=-vn * (n_right_m3 + spec.n1_right_m3),
        d_cap_p_left_dp_L=vp * f,
        d_cap_p_left_df=vp * (p_left_m3 + spec.p1_left_m3),
        d_cap_p_right_dp_R=vp * f,
        d_cap_p_right_df=vp * (p_right_m3 + spec.p1_right_m3),
        d_storage_rate_dn_L=vn * omf,
        d_storage_rate_dp_L=-vp * f,
        d_storage_rate_dn_R=vn * omf,
        d_storage_rate_dp_R=-vp * f,
        d_storage_rate_df=-denom,
        d_sheet_charge_df=d_sigma_df,
    )
