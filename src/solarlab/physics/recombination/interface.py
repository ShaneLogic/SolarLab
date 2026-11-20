"""Heterointerface recombination and static sheet-charge constitutive laws (P04-09).

All rates returned here are areal particle rates in ``m^-2 s^-1`` (never
pre-divided by cell width ``dx`` or ``dy``), and all interface sheet charges
are areal charge densities in ``C m^-2``. Deposition onto control volumes or
Gauss jump rows belongs exclusively to geometry and the unified Assembler.

Provided laws:

* :func:`interface_srh_rate` -- single-plane or cross-carrier Shockley-Read-Hall
  rate ``v_n v_p (n p - n_i^2) / (v_n (n + n_1) + v_p (p + p_1))`` with
  cancellation-free analytical density derivatives (real and complex-step).
* :func:`resolve_interface_defect_references` -- physical band-aligned single-level
  interface defect emission densities ``(n_1, p_1)`` on the reference side and
  across both sides of a two-sided heterointerface.
* :func:`evaluate_shared_interface_trap` -- single-evaluation two-sided shared
  trap occupancy ``f``, side-resolved capture rates ``(C_{n,L}, C_{p,L},
  C_{n,R}, C_{p,R})`` obeying ``C_{n,L} + C_{n,R} == C_{p,L} + C_{p,R}``, and
  optional equilibrium-referenced sheet charge ``Delta sigma = -q N_t (f - f_ref)``
  with complete analytical derivatives.
* :func:`solve_plane_closure` -- local 2x2 carrier-plane QSS closure with exact
  implicit-function-theorem derivatives wrt projected adjacent densities.
* :class:`TwoDimensionalInterfaceSRHMode` and
  :func:`evaluate_cross_carrier_cell_pair_srh` -- explicit constitutive contract
  for the 2D cross-row cell-pair SRH channel (retaining ``interface_two_sided``
  for 2D while 1D additive mirror channels remain refused).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import math
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from solarlab.contracts.units import ELEMENTARY_CHARGE_C

__all__ = [
    "CrossCarrierCellPairEvaluation",
    "INTERFACE_BAND_REFERENCE_MODES",
    "InterfaceBandReferenceMode",
    "InterfaceDefectReferences",
    "InterfaceDefectSpec",
    "InterfaceSRHRate",
    "PlaneClosureParams",
    "PlaneClosureResult",
    "REFERENCE_LAYER_BAND_ANCHOR",
    "SCAPS_INTERFACE_EXTREMA_ANCHOR",
    "S_SUPPLY_M_S",
    "SharedInterfaceTrapEvaluation",
    "TwoDimensionalInterfaceSRHMode",
    "build_plane_closure_params",
    "equilibrium_referenced_sheet_charge",
    "evaluate_cross_carrier_cell_pair_srh",
    "evaluate_shared_interface_trap",
    "interface_srh_rate",
    "resolve_interface_defect_references",
    "solve_plane_closure",
]

Q = ELEMENTARY_CHARGE_C
S_SUPPLY_M_S: float = 2.5e4
_DENOM_FLOOR: float = 1.0e-300
_EXP_CLIP: float = 500.0
_QSS_MAX_ITER: int = 40
_QSS_TOL: float = 1.0e-13
_QSS_STEP_TOL: float = 1.0e-14
_QSS_DENSITY_FLOOR: float = 1.0e-30
_QSS_LOG_STEP_CLIP: float = 30.0

NumericArray: TypeAlias = NDArray[np.float64] | NDArray[np.complex128]
RealArray: TypeAlias = NDArray[np.float64]
EnergyReference: TypeAlias = Literal["below_conduction_band", "above_valence_band"]
InterfaceBandReferenceMode: TypeAlias = Literal[
    "reference_layer", "scaps_interface_extrema"
]
REFERENCE_LAYER_BAND_ANCHOR: InterfaceBandReferenceMode = "reference_layer"
SCAPS_INTERFACE_EXTREMA_ANCHOR: InterfaceBandReferenceMode = "scaps_interface_extrema"
INTERFACE_BAND_REFERENCE_MODES: tuple[InterfaceBandReferenceMode, ...] = (
    REFERENCE_LAYER_BAND_ANCHOR,
    SCAPS_INTERFACE_EXTREMA_ANCHOR,
)


def _finite_real(value: float, name: str) -> float:
    scalar = float(value)
    if not math.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


def _finite_nonnegative(value: float, name: str) -> float:
    scalar = float(value)
    if not math.isfinite(scalar) or scalar < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return scalar


def _finite_positive(value: float, name: str) -> float:
    scalar = float(value)
    if not math.isfinite(scalar) or scalar <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return scalar


def _boltzmann_emission_density(dos_m3: float, delta_eV: float, vt_V: float) -> float:
    dos = _finite_positive(dos_m3, "dos_m3")
    vt = _finite_positive(vt_V, "thermal_voltage_V")
    exponent = max(-_EXP_CLIP, min(_EXP_CLIP, -float(delta_eV) / vt))
    return float(dos * math.exp(exponent))


@dataclass(frozen=True)
class InterfaceSRHRate:
    """Areal interface SRH rate [m^-2 s^-1] and its carrier-density partials [m/s]."""

    rate_m2_s: NumericArray
    d_rate_dn_m_s: NumericArray
    d_rate_dp_m_s: NumericArray


def interface_srh_rate(
    n_m3: ArrayLike,
    p_m3: ArrayLike,
    *,
    v_n_m_s: ArrayLike,
    v_p_m_s: ArrayLike,
    ni_sq_m6: ArrayLike,
    n1_m3: ArrayLike,
    p1_m3: ArrayLike,
    clamp_nonnegative: bool = False,
) -> InterfaceSRHRate:
    """Evaluate areal Shockley-Read-Hall interface recombination [m^-2 s^-1].

    Supports scalar or array inputs and both real ``float64`` and complex-step
    ``complex128`` carrier densities. The density derivatives use the exact
    positive-numerator quotient form so no catastrophic cancellation occurs near
    thermal equilibrium ``n p == n_i^2``.
    """
    raw_n = np.asarray(n_m3)
    raw_p = np.asarray(p_m3)
    is_complex = np.iscomplexobj(raw_n) or np.iscomplexobj(raw_p)
    dtype = np.complex128 if is_complex else np.float64

    n_arr, p_arr = np.broadcast_arrays(
        np.asarray(raw_n, dtype=dtype),
        np.asarray(raw_p, dtype=dtype),
    )
    vn = np.broadcast_to(np.asarray(v_n_m_s, dtype=np.float64), n_arr.shape)
    vp = np.broadcast_to(np.asarray(v_p_m_s, dtype=np.float64), n_arr.shape)
    ni_sq = np.broadcast_to(np.asarray(ni_sq_m6, dtype=np.float64), n_arr.shape)
    n1 = np.broadcast_to(np.asarray(n1_m3, dtype=np.float64), n_arr.shape)
    p1 = np.broadcast_to(np.asarray(p1_m3, dtype=np.float64), n_arr.shape)

    for name, arr in (
        ("v_n_m_s", vn),
        ("v_p_m_s", vp),
        ("ni_sq_m6", ni_sq),
        ("n1_m3", n1),
        ("p1_m3", p1),
    ):
        if not np.all(np.isfinite(arr)) or np.any(arr < 0.0):
            raise ValueError(f"{name} must be finite and non-negative")
    if not (np.all(np.isfinite(n_arr)) and np.all(np.isfinite(p_arr))):
        raise ValueError("carrier densities must be finite")

    rate = np.zeros(n_arr.shape, dtype=dtype)
    d_dn = np.zeros(n_arr.shape, dtype=dtype)
    d_dp = np.zeros(n_arr.shape, dtype=dtype)

    active = (vn > 0.0) & (vp > 0.0)
    if not np.any(active):
        return InterfaceSRHRate(rate, d_dn, d_dp)

    denom = vn * (n_arr + n1) + vp * (p_arr + p1)
    valid = active & (np.abs(denom) > _DENOM_FLOOR)
    if not np.any(valid):
        return InterfaceSRHRate(rate, d_dn, d_dp)

    vn_v = vn[valid]
    vp_v = vp[valid]
    n_v = n_arr[valid]
    p_v = p_arr[valid]
    n1_v = n1[valid]
    p1_v = p1[valid]
    ni_v = ni_sq[valid]
    den_v = denom[valid]
    den_sq = den_v * den_v
    pref = vn_v * vp_v

    raw_rate = pref * (n_v * p_v - ni_v) / den_v
    raw_dn = pref * (vn_v * (p_v * n1_v + ni_v) + vp_v * p_v * (p_v + p1_v)) / den_sq
    raw_dp = pref * (vp_v * (n_v * p1_v + ni_v) + vn_v * n_v * (n_v + n1_v)) / den_sq

    if clamp_nonnegative:
        keep = np.real(raw_rate) >= 0.0
        zero = np.zeros_like(raw_rate)
        raw_rate = np.where(keep, raw_rate, zero)
        raw_dn = np.where(keep, raw_dn, zero)
        raw_dp = np.where(keep, raw_dp, zero)

    rate[valid] = raw_rate
    d_dn[valid] = raw_dn
    d_dp[valid] = raw_dp
    return InterfaceSRHRate(rate, d_dn, d_dp)


@dataclass(frozen=True)
class InterfaceDefectSpec:
    """Single-level microscopic heterointerface defect specification."""

    areal_density_m2: float
    trap_depth_eV: float
    energy_reference: EnergyReference
    sigma_n_m2: float
    sigma_p_m2: float
    v_th_n_m_s: float
    v_th_p_m_s: float
    calibration_factor: float = 1.0
    charge_state_empty: int = 0
    charge_state_occupied: int = -1

    def __post_init__(self) -> None:
        _finite_nonnegative(self.areal_density_m2, "areal_density_m2")
        _finite_nonnegative(self.trap_depth_eV, "trap_depth_eV")
        if self.energy_reference not in ("below_conduction_band", "above_valence_band"):
            raise ValueError(f"unsupported energy_reference {self.energy_reference!r}")
        _finite_nonnegative(self.sigma_n_m2, "sigma_n_m2")
        _finite_nonnegative(self.sigma_p_m2, "sigma_p_m2")
        _finite_positive(self.v_th_n_m_s, "v_th_n_m_s")
        _finite_positive(self.v_th_p_m_s, "v_th_p_m_s")
        _finite_nonnegative(self.calibration_factor, "calibration_factor")
        if (self.charge_state_empty, self.charge_state_occupied) not in ((0, -1), (1, 0)):
            raise ValueError("interface defect charge states must be (0, -1) or (1, 0)")

    @property
    def microscopic_capture_velocities_m_s(self) -> tuple[float, float]:
        return (
            float(self.areal_density_m2 * self.sigma_n_m2 * self.v_th_n_m_s),
            float(self.areal_density_m2 * self.sigma_p_m2 * self.v_th_p_m_s),
        )

    def effective_capture_velocities_m_s(
        self,
        *,
        override_v_n_m_s: float | None = None,
        override_v_p_m_s: float | None = None,
    ) -> tuple[float, float]:
        micro_n, micro_p = self.microscopic_capture_velocities_m_s
        base_n = micro_n if override_v_n_m_s is None else _finite_nonnegative(override_v_n_m_s, "v_n")
        base_p = micro_p if override_v_p_m_s is None else _finite_nonnegative(override_v_p_m_s, "v_p")
        cal = float(self.calibration_factor)
        return cal * base_n, cal * base_p


@dataclass(frozen=True)
class InterfaceDefectReferences:
    """Resolved reference side and SRH emission densities for an interface defect."""

    reference_side: Literal["left", "right"]
    reference_gap_eV: float
    n1_reference_m3: float
    p1_reference_m3: float
    n1_midgap_symmetric_m3: float
    p1_midgap_symmetric_m3: float
    n1_left_m3: float
    p1_left_m3: float
    n1_right_m3: float
    p1_right_m3: float


def resolve_interface_defect_references(
    *,
    left_role: str,
    right_role: str,
    left_chi_phys_eV: float,
    left_Eg_phys_eV: float,
    left_Nc_m3: float,
    left_Nv_m3: float,
    right_chi_phys_eV: float,
    right_Eg_phys_eV: float,
    right_Nc_m3: float,
    right_Nv_m3: float,
    thermal_voltage_V: float,
    trap_depth_eV: float,
    energy_reference: EnergyReference,
    require_in_gap: bool = False,
    band_reference_mode: InterfaceBandReferenceMode = REFERENCE_LAYER_BAND_ANCHOR,
) -> InterfaceDefectReferences:
    """Resolve reference side and Boltzmann emission densities on physical bands.

    Reference-layer rule (validated legacy contract):
    1. Unique ``role == "absorber"`` among ``(left, right)``; otherwise
    2. Smaller physical band gap ``Eg_phys`` (left wins exact ties).

    ``band_reference_mode`` selects the physical band-edge anchor:
    - ``"reference_layer"`` (default): anchors ``trap_depth_eV`` to the selected
      reference layer's physical ``E_c`` or ``E_v``.
    - ``"scaps_interface_extrema"``: anchors ``below_conduction_band`` below the
      lowest adjacent conduction band ``min(E_{c,L}, E_{c,R}) = -max(chi_L, chi_R)``
      and ``above_valence_band`` above the highest adjacent valence band
      ``max(E_{v,L}, E_{v,R}) = -min(chi_L + Eg_L, chi_R + Eg_R)``, matching
      SCAPS-1D's heterointerface defect energy convention.

    Physical band edges ``E_c = -chi_phys`` and ``E_v = -chi_phys - Eg_phys`` are
    kept strictly distinct from DOS-folded transport potentials.
    """
    vt = _finite_positive(thermal_voltage_V, "thermal_voltage_V")
    depth = _finite_nonnegative(trap_depth_eV, "trap_depth_eV")
    left_eg = _finite_positive(left_Eg_phys_eV, "left_Eg_phys_eV")
    right_eg = _finite_positive(right_Eg_phys_eV, "right_Eg_phys_eV")
    left_nc = _finite_positive(left_Nc_m3, "left_Nc_m3")
    left_nv = _finite_positive(left_Nv_m3, "left_Nv_m3")
    right_nc = _finite_positive(right_Nc_m3, "right_Nc_m3")
    right_nv = _finite_positive(right_Nv_m3, "right_Nv_m3")
    left_chi = float(left_chi_phys_eV)
    right_chi = float(right_chi_phys_eV)
    if not (math.isfinite(left_chi) and math.isfinite(right_chi)):
        raise ValueError("physical electron affinities must be finite")
    if energy_reference not in ("below_conduction_band", "above_valence_band"):
        raise ValueError(f"unsupported energy_reference {energy_reference!r}")
    if band_reference_mode not in INTERFACE_BAND_REFERENCE_MODES:
        raise ValueError(
            f"unsupported band_reference_mode {band_reference_mode!r}; "
            f"expected one of {INTERFACE_BAND_REFERENCE_MODES}"
        )

    if left_role == "absorber" and right_role != "absorber":
        ref_side: Literal["left", "right"] = "left"
    elif right_role == "absorber" and left_role != "absorber":
        ref_side = "right"
    elif left_eg <= right_eg:
        ref_side = "left"
    else:
        ref_side = "right"

    ref_chi = left_chi if ref_side == "left" else right_chi
    ref_eg = left_eg if ref_side == "left" else right_eg
    if require_in_gap and depth > ref_eg:
        raise ValueError(
            f"trap_depth_eV={depth} lies outside reference physical band gap {ref_eg}"
        )

    if band_reference_mode == SCAPS_INTERFACE_EXTREMA_ANCHOR:
        if energy_reference == "below_conduction_band":
            trap_energy_eV = -max(left_chi, right_chi) - depth
        else:
            trap_energy_eV = -min(left_chi + left_eg, right_chi + right_eg) + depth
    else:
        if energy_reference == "below_conduction_band":
            trap_energy_eV = -ref_chi - depth
        else:
            trap_energy_eV = -ref_chi - ref_eg + depth

    ec_left, ev_left = -left_chi, -left_chi - left_eg
    ec_right, ev_right = -right_chi, -right_chi - right_eg

    n1_left = _boltzmann_emission_density(left_nc, ec_left - trap_energy_eV, vt)
    p1_left = _boltzmann_emission_density(left_nv, trap_energy_eV - ev_left, vt)
    n1_right = _boltzmann_emission_density(right_nc, ec_right - trap_energy_eV, vt)
    p1_right = _boltzmann_emission_density(right_nv, trap_energy_eV - ev_right, vt)

    if ref_side == "left":
        n1_ref, p1_ref = n1_left, p1_left
        ref_dos = math.sqrt(left_nc) * math.sqrt(left_nv)
        ec_ref, ev_ref = ec_left, ev_left
    else:
        n1_ref, p1_ref = n1_right, p1_right
        ref_dos = math.sqrt(right_nc) * math.sqrt(right_nv)
        ec_ref, ev_ref = ec_right, ev_right

    n1_sym = _boltzmann_emission_density(ref_dos, ec_ref - trap_energy_eV, vt)
    p1_sym = _boltzmann_emission_density(ref_dos, trap_energy_eV - ev_ref, vt)

    return InterfaceDefectReferences(
        reference_side=ref_side,
        reference_gap_eV=ref_eg,
        n1_reference_m3=n1_ref,
        p1_reference_m3=p1_ref,
        n1_midgap_symmetric_m3=n1_sym,
        p1_midgap_symmetric_m3=p1_sym,
        n1_left_m3=n1_left,
        p1_left_m3=p1_left,
        n1_right_m3=n1_right,
        p1_right_m3=p1_right,
    )


def equilibrium_referenced_sheet_charge(
    occupancy: float,
    *,
    areal_density_m2: float,
    equilibrium_occupancy: float,
) -> float:
    """Static interface sheet-charge closure ``Delta sigma = -q N_t (f - f_ref)`` [C/m^2].

    Because each change of electron occupancy ``Delta f`` adds electronic charge
    ``-q N_t Delta f`` regardless of whether the trap is acceptor ``(0/-)`` or
    donor ``(+/0)`` relative to its certified dark equilibrium occupancy
    ``f_ref``, the incremental sheet charge has one universal sign and vanishes
    identically at dark equilibrium ``f == f_ref``.
    """
    nt = _finite_nonnegative(areal_density_m2, "areal_density_m2")
    f = float(occupancy)
    f_ref = float(equilibrium_occupancy)
    if not (math.isfinite(f) and 0.0 <= f <= 1.0):
        raise ValueError("occupancy must lie in [0, 1]")
    if not (math.isfinite(f_ref) and 0.0 <= f_ref <= 1.0):
        raise ValueError("equilibrium_occupancy must lie in [0, 1]")
    return float(-Q * nt * (f - f_ref))


@dataclass(frozen=True)
class SharedInterfaceTrapEvaluation:
    """Single constitutive evaluation of a two-sided shared interface trap.

    Density ordering for derivative vectors and matrices is
    ``(n_left, p_left, n_right, p_right)``.
    Capture-rate ordering is ``(C_n_left, C_p_left, C_n_right, C_p_right)`` in
    ``m^-2 s^-1``.
    """

    occupancy: float
    d_occupancy_d_densities_m3: RealArray
    capture_rates_m2_s: RealArray
    d_capture_d_densities_m_s: RealArray
    incremental_sheet_charge_C_m2: float
    d_sheet_charge_d_densities_C_m: RealArray


def evaluate_shared_interface_trap(
    n_left_m3: float,
    p_left_m3: float,
    n_right_m3: float,
    p_right_m3: float,
    *,
    v_n_left_m_s: float,
    v_p_left_m_s: float,
    v_n_right_m_s: float,
    v_p_right_m_s: float,
    n1_left_m3: float,
    p1_left_m3: float,
    n1_right_m3: float,
    p1_right_m3: float,
    areal_density_m2: float = 0.0,
    equilibrium_occupancy: float | None = None,
) -> SharedInterfaceTrapEvaluation:
    """Evaluate shared trap occupancy, side capture rates, and optional sheet charge once."""
    densities = (
        _finite_nonnegative(n_left_m3, "n_left_m3"),
        _finite_nonnegative(p_left_m3, "p_left_m3"),
        _finite_nonnegative(n_right_m3, "n_right_m3"),
        _finite_nonnegative(p_right_m3, "p_right_m3"),
    )
    vn_l = _finite_nonnegative(v_n_left_m_s, "v_n_left_m_s")
    vp_l = _finite_nonnegative(v_p_left_m_s, "v_p_left_m_s")
    vn_r = _finite_nonnegative(v_n_right_m_s, "v_n_right_m_s")
    vp_r = _finite_nonnegative(v_p_right_m_s, "v_p_right_m_s")
    n1_l = _finite_nonnegative(n1_left_m3, "n1_left_m3")
    p1_l = _finite_nonnegative(p1_left_m3, "p1_left_m3")
    n1_r = _finite_nonnegative(n1_right_m3, "n1_right_m3")
    p1_r = _finite_nonnegative(p1_right_m3, "p1_right_m3")
    nt = _finite_nonnegative(areal_density_m2, "areal_density_m2")

    n_l, p_l, n_r, p_r = densities
    alpha = vn_l * n_l + vn_r * n_r + vp_l * p1_l + vp_r * p1_r
    beta = vn_l * n1_l + vn_r * n1_r + vp_l * p_l + vp_r * p_r
    denom = alpha + beta

    if denom <= _DENOM_FLOOR:
        zero4: RealArray = np.zeros(4, dtype=np.float64)
        zero44: RealArray = np.zeros((4, 4), dtype=np.float64)
        sheet = (
            0.0
            if equilibrium_occupancy is None
            else equilibrium_referenced_sheet_charge(
                0.0,
                areal_density_m2=nt,
                equilibrium_occupancy=equilibrium_occupancy,
            )
        )
        return SharedInterfaceTrapEvaluation(
            occupancy=0.0,
            d_occupancy_d_densities_m3=zero4,
            capture_rates_m2_s=zero4.copy(),
            d_capture_d_densities_m_s=zero44,
            incremental_sheet_charge_C_m2=sheet,
            d_sheet_charge_d_densities_C_m=zero4.copy(),
        )

    occupancy = float(alpha / denom)
    one_minus_f = float(beta / denom)
    denom_sq = denom * denom

    df_d = np.array(
        [
            vn_l * beta / denom_sq,
            -vp_l * alpha / denom_sq,
            vn_r * beta / denom_sq,
            -vp_r * alpha / denom_sq,
        ],
        dtype=np.float64,
    )

    # Numerators avoid rounding f and (1 - f) independently near equilibrium.
    cn_l = vn_l * (n_l * beta - n1_l * alpha) / denom
    cp_l = vp_l * (p_l * alpha - p1_l * beta) / denom
    cn_r = vn_r * (n_r * beta - n1_r * alpha) / denom
    cp_r = vp_r * (p_r * alpha - p1_r * beta) / denom
    capture = np.array([cn_l, cp_l, cn_r, cp_r], dtype=np.float64)

    # C_n,s = v_n,s * (n_s * (1 - f) - n1_s * f)
    # dC_n,s/dx = v_n,s * (delta_{x,n_s}*(1 - f) - (n_s + n1_s) * df/dx)
    # C_p,s = v_p,s * (p_s * f - p1_s * (1 - f))
    # dC_p,s/dx = v_p,s * (delta_{x,p_s}*f + (p_s + p1_s) * df/dx)
    d_cap: RealArray = np.empty((4, 4), dtype=np.float64)
    d_cap[0, :] = -vn_l * (n_l + n1_l) * df_d
    d_cap[0, 0] += vn_l * one_minus_f
    d_cap[1, :] = vp_l * (p_l + p1_l) * df_d
    d_cap[1, 1] += vp_l * occupancy
    d_cap[2, :] = -vn_r * (n_r + n1_r) * df_d
    d_cap[2, 2] += vn_r * one_minus_f
    d_cap[3, :] = vp_r * (p_r + p1_r) * df_d
    d_cap[3, 3] += vp_r * occupancy

    if equilibrium_occupancy is None:
        sheet = 0.0
        d_sheet: RealArray = np.zeros(4, dtype=np.float64)
    else:
        sheet = equilibrium_referenced_sheet_charge(
            occupancy,
            areal_density_m2=nt,
            equilibrium_occupancy=equilibrium_occupancy,
        )
        d_sheet = -Q * nt * df_d

    return SharedInterfaceTrapEvaluation(
        occupancy=occupancy,
        d_occupancy_d_densities_m3=df_d,
        capture_rates_m2_s=capture,
        d_capture_d_densities_m_s=d_cap,
        incremental_sheet_charge_C_m2=sheet,
        d_sheet_charge_d_densities_C_m=d_sheet,
    )


@dataclass(frozen=True)
class PlaneClosureParams:
    """Build-time constants of one defect interface's 1D plane closure.

    Constructed from post-DOS-fold ``chi`` and ``chi + Eg`` values at the two
    interface-adjacent nodes plus the reference-layer physical DOS and raw
    electron affinity.
    """

    bn_L: float
    bn_R: float
    bp_L: float
    bp_R: float
    ni_s_sq_m6: float
    n1_s_m3: float
    p1_s_m3: float
    s_supply_m_s: float = S_SUPPLY_M_S


@dataclass(frozen=True)
class PlaneClosureResult:
    """Converged plane densities, net areal rate, and exact IFT derivatives."""

    n_plane_m3: float
    p_plane_m3: float
    rate_m2_s: float
    d_rate_d_n_left_proj_m_s: float
    d_rate_d_n_right_proj_m_s: float
    d_rate_d_p_left_proj_m_s: float
    d_rate_d_p_right_proj_m_s: float
    converged: bool
    iterations: int


def build_plane_closure_params(
    *,
    chi_left_folded_eV: float,
    chi_right_folded_eV: float,
    valence_left_folded_eV: float,
    valence_right_folded_eV: float,
    Nc_ref_m3: float,
    Nv_ref_m3: float,
    chi_ref_phys_eV: float,
    trap_depth_below_ref_cb_eV: float,
    thermal_voltage_V: float,
    s_supply_m_s: float = S_SUPPLY_M_S,
) -> PlaneClosureParams:
    """Construct :class:`PlaneClosureParams` from folded interface band edges.

    ``valence_*_folded_eV`` are the folded ``chi + Eg`` quantities at the two
    interface-adjacent nodes; ``chi_ref_phys_eV`` is the raw physical electron
    affinity on the defect reference layer.
    """
    chi_l = _finite_real(chi_left_folded_eV, "chi_left_folded_eV")
    chi_r = _finite_real(chi_right_folded_eV, "chi_right_folded_eV")
    ceg_l = _finite_real(valence_left_folded_eV, "valence_left_folded_eV")
    ceg_r = _finite_real(valence_right_folded_eV, "valence_right_folded_eV")
    nc_ref = _finite_positive(Nc_ref_m3, "Nc_ref_m3")
    nv_ref = _finite_positive(Nv_ref_m3, "Nv_ref_m3")
    chi_ref = _finite_real(chi_ref_phys_eV, "chi_ref_phys_eV")
    et_depth = _finite_nonnegative(trap_depth_below_ref_cb_eV, "trap_depth_below_ref_cb_eV")
    vt = _finite_positive(thermal_voltage_V, "thermal_voltage_V")
    s_sup = _finite_positive(s_supply_m_s, "s_supply_m_s")

    chi_s = max(chi_l, chi_r)
    ceg_s = min(ceg_l, ceg_r)
    eg_s = ceg_s - chi_s

    def _penalty(delta: float) -> float:
        return math.exp(-min(max(delta, 0.0), 60.0 * vt) / vt)

    depth_n = min(max(et_depth - (chi_s - chi_ref), 0.0), max(eg_s, 0.0))

    def _exp_guarded(arg: float) -> float:
        return math.exp(max(-120.0, min(120.0, arg)))

    return PlaneClosureParams(
        bn_L=_penalty(chi_s - chi_l),
        bn_R=_penalty(chi_s - chi_r),
        bp_L=_penalty(ceg_l - ceg_s),
        bp_R=_penalty(ceg_r - ceg_s),
        ni_s_sq_m6=nc_ref * nv_ref * _exp_guarded(-eg_s / vt),
        n1_s_m3=nc_ref * _exp_guarded(-depth_n / vt),
        p1_s_m3=nv_ref * _exp_guarded(-(eg_s - depth_n) / vt),
        s_supply_m_s=s_sup,
    )


def solve_plane_closure(
    n_left_proj_m3: float,
    n_right_proj_m3: float,
    p_left_proj_m3: float,
    p_right_proj_m3: float,
    params: PlaneClosureParams,
    v_n_m_s: float,
    v_p_m_s: float,
    *,
    allow_generation: bool = False,
) -> PlaneClosureResult:
    """Solve the 2x2 carrier-plane QSS balance with exact IFT sensitivities."""
    s = _finite_positive(params.s_supply_m_s, "s_supply_m_s")
    vn = _finite_nonnegative(v_n_m_s, "v_n_m_s")
    vp = _finite_nonnegative(v_p_m_s, "v_p_m_s")
    nl = float(n_left_proj_m3)
    nr = float(n_right_proj_m3)
    pl = float(p_left_proj_m3)
    pr = float(p_right_proj_m3)
    nb = params.bn_L * max(nl, 0.0) + params.bn_R * max(nr, 0.0)
    pb = params.bp_L * max(pl, 0.0) + params.bp_R * max(pr, 0.0)

    if vn <= 0.0 or vp <= 0.0:
        return PlaneClosureResult(
            n_plane_m3=0.5 * nb,
            p_plane_m3=0.5 * pb,
            rate_m2_s=0.0,
            d_rate_d_n_left_proj_m_s=0.0,
            d_rate_d_n_right_proj_m_s=0.0,
            d_rate_d_p_left_proj_m_s=0.0,
            d_rate_d_p_right_proj_m_s=0.0,
            converged=True,
            iterations=0,
        )

    s2 = 2.0 * s
    ln_n = math.log(max(0.5 * nb, _QSS_DENSITY_FLOOR))
    ln_p = math.log(max(0.5 * pb, _QSS_DENSITY_FLOOR))
    converged = False
    iters = 0

    for it in range(1, _QSS_MAX_ITER + 1):
        iters = it
        n_s = math.exp(ln_n)
        p_s = math.exp(ln_p)
        num = n_s * p_s - params.ni_s_sq_m6
        den = (n_s + params.n1_s_m3) / vp + (p_s + params.p1_s_m3) / vn
        signed = num > 0.0 or allow_generation
        rate = num / den if signed else 0.0
        f1 = s * nb - s2 * n_s - rate
        f2 = s * pb - s2 * p_s - rate
        scale = s * (nb + pb) + abs(rate) + _QSS_DENSITY_FLOOR
        if abs(f1) + abs(f2) < _QSS_TOL * scale:
            converged = True
            break
        if signed:
            dr_dn = (p_s * den - num / vp) / (den * den)
            dr_dp = (n_s * den - num / vn) / (den * den)
        else:
            dr_dn = 0.0
            dr_dp = 0.0
        j11 = (-s2 - dr_dn) * n_s
        j12 = (-dr_dp) * p_s
        j21 = (-dr_dn) * n_s
        j22 = (-s2 - dr_dp) * p_s
        det = j11 * j22 - j12 * j21
        if det == 0.0 or not math.isfinite(det):
            break
        d_ln_n = (-f1 * j22 + f2 * j12) / det
        d_ln_p = (-f2 * j11 + f1 * j21) / det
        d_ln_n = max(-_QSS_LOG_STEP_CLIP, min(_QSS_LOG_STEP_CLIP, d_ln_n))
        d_ln_p = max(-_QSS_LOG_STEP_CLIP, min(_QSS_LOG_STEP_CLIP, d_ln_p))
        ln_n += d_ln_n
        ln_p += d_ln_p
        if abs(d_ln_n) + abs(d_ln_p) <= _QSS_STEP_TOL:
            converged = True
            break

    n_s = math.exp(ln_n)
    p_s = math.exp(ln_p)
    srh_final = interface_srh_rate(
        n_s,
        p_s,
        v_n_m_s=vn,
        v_p_m_s=vp,
        ni_sq_m6=params.ni_s_sq_m6,
        n1_m3=params.n1_s_m3,
        p1_m3=params.p1_s_m3,
        clamp_nonnegative=not allow_generation,
    )
    r_final = float(np.real(srh_final.rate_m2_s))
    r_n_final = float(np.real(srh_final.d_rate_dn_m_s))
    r_p_final = float(np.real(srh_final.d_rate_dp_m_s))

    # Exact IFT wrt (nb, pb) via F1 - F2 = s*(nb - pb) - 2*s*(n_s - p_s) == 0:
    # dR/dnb = s * r_n / (2*s + r_n + r_p), dR/dpb = s * r_p / (2*s + r_n + r_p).
    ift_denom = s2 + r_n_final + r_p_final
    dr_dnb = s * r_n_final / ift_denom
    dr_dpb = s * r_p_final / ift_denom

    return PlaneClosureResult(
        n_plane_m3=n_s,
        p_plane_m3=p_s,
        rate_m2_s=r_final,
        d_rate_d_n_left_proj_m_s=(params.bn_L if nl > 0.0 else 0.0) * dr_dnb,
        d_rate_d_n_right_proj_m_s=(params.bn_R if nr > 0.0 else 0.0) * dr_dnb,
        d_rate_d_p_left_proj_m_s=(params.bp_L if pl > 0.0 else 0.0) * dr_dpb,
        d_rate_d_p_right_proj_m_s=(params.bp_R if pr > 0.0 else 0.0) * dr_dpb,
        converged=converged,
        iterations=iters,
    )


class TwoDimensionalInterfaceSRHMode(StrEnum):
    """Explicit 2D heterointerface SRH modes (P04-09 item 4).

    The legacy ``settings.interface_two_sided`` boolean remains in configuration
    for the 2D cross-row cell-pair channel while the falsified 1D additive
    mirror channel stays refused in ``compile_device_1d``.
    """

    OFF = "off"
    CROSS_CARRIER_CELL_PAIR_V1 = "cross_carrier_cell_pair_v1"


@dataclass(frozen=True)
class CrossCarrierCellPairEvaluation:
    """Areal rate [m^-2 s^-1] and half-plane split for 2D cell-pair SRH."""

    areal_rate_m2_s: RealArray
    d_areal_rate_d_n_other_m_s: RealArray
    d_areal_rate_d_p_self_m_s: RealArray
    half_rate_m2_s: RealArray


def evaluate_cross_carrier_cell_pair_srh(
    mode: TwoDimensionalInterfaceSRHMode,
    *,
    n_other_m3: ArrayLike,
    p_self_m3: ArrayLike,
    v_n_m_s: ArrayLike,
    v_p_m_s: ArrayLike,
    ni_sq_m6: ArrayLike,
    n1_m3: ArrayLike,
    p1_m3: ArrayLike,
    clamp_nonnegative: bool = True,
) -> CrossCarrierCellPairEvaluation:
    """Evaluate 2D cross-row cell-pair SRH areal rate without ``dy`` division."""
    n_arr = np.asarray(n_other_m3, dtype=np.float64)
    p_arr = np.asarray(p_self_m3, dtype=np.float64)
    if mode == TwoDimensionalInterfaceSRHMode.OFF:
        shape = np.broadcast_shapes(n_arr.shape, p_arr.shape)
        zeros = np.zeros(shape, dtype=np.float64)
        return CrossCarrierCellPairEvaluation(zeros, zeros.copy(), zeros.copy(), zeros.copy())
    if mode != TwoDimensionalInterfaceSRHMode.CROSS_CARRIER_CELL_PAIR_V1:
        raise ValueError(f"unsupported 2D interface SRH mode {mode!r}")
    srh = interface_srh_rate(
        n_arr,
        p_arr,
        v_n_m_s=v_n_m_s,
        v_p_m_s=v_p_m_s,
        ni_sq_m6=ni_sq_m6,
        n1_m3=n1_m3,
        p1_m3=p1_m3,
        clamp_nonnegative=clamp_nonnegative,
    )
    rate = np.asarray(srh.rate_m2_s, dtype=np.float64)
    dn = np.asarray(srh.d_rate_dn_m_s, dtype=np.float64)
    dp = np.asarray(srh.d_rate_dp_m_s, dtype=np.float64)
    return CrossCarrierCellPairEvaluation(rate, dn, dp, 0.5 * rate)
