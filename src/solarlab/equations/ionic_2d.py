"""Executable single-positive-ion finite-site 2D equations and bounded 2D extensions.

Unknowns are ``(phi, z_n, z_p, y_ion)`` where ``phi, z_n, z_p`` live on the
complete ``(Ny, Nx)`` rectangular Neumann-x support and ``y_ion`` lives on the
closed four-side-blocking 2D ion support ``(Ny_ion, Nx)`` via the shifted
finite-site logistic coordinate ``P = P_lim * expit(logit_ref + y_ion)``.
Dual-ion, shared-site, periodic-x mobile-ion, Robin-contact mobile-ion, and
dynamic-interface 2D bindings are explicitly refused.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import csc_matrix
from scipy.special import expit

from solarlab.contracts import (
    AREA, COULOMB, LENGTH, PARTICLE, SECOND, VOLUME, CellSource, EquationSpec,
    FaceFlux, Geometry, Layout, Support, TermSink, frozen_array, integer_indices,
)
from solarlab.contracts.units import ELEMENTARY_CHARGE_C
from solarlab.discretization.bernoulli import bernoulli, bernoulli_derivative
from solarlab.equations.dc_1d import CapabilityError, quasi_fermi_potential_tangents
from solarlab.equations.dc_2d import (
    CompiledDevice2D,
    State2D,
    evaluate_2d_carrier_face_fluxes,
    has_prospective_2d_transport,
)
from solarlab.mesh.geometry_2d import RECTANGULAR_HALF_ENDPOINT_2D_V1, RectangularGeometry2D
from solarlab.physics.electrostatics.charge import ChargeEntry, ChargeLedger
from solarlab.physics.ions import (
    DIFFUSION_ONLY_STERIC_V1, WHOLE_FLUX_STERIC_V1, IonFaceFluxJacobian,
    ion_face_flux, single_site_density,
)
from solarlab.physics.recombination.bulk import RecombinationRate
from solarlab.physics.transport.electrochemical_flux import CarrierFaceFlux, CarrierFaceFluxes, stable_sg_particle_flux

__all__ = [
    "CompiledIonicDevice2D",
    "Ionic2DProblem",
    "IonicState2D",
    "IonFluxes2D",
    "MobileIonDiagnostics2D",
    "RadiativeReabsorptionSpec2D",
    "TwoSidedInterfaceSRHCoupling2D",
    "TwoSidedInterfaceSRHReport2D",
    "assess_mobile_ion_terminal_2d",
    "control_volume_areas_2d",
    "evaluate_two_sided_interface_srh_2d",
    "finite_site_secant_2d_m3",
    "interface_srh_sheet_rate_and_partials",
    "ion_inventory_2d",
    "ion_pair_flux_jacobian_2d",
    "positive_ion_fluxes_2d",
    "recompute_g_with_rad_2d",
    "trapezoid_weights_1d",
]

Vector: TypeAlias = NDArray[np.float64]
Q = ELEMENTARY_CHARGE_C


def _read_only(values: ArrayLike) -> Vector:
    arr = np.asarray(values, dtype=float).copy()
    arr.setflags(write=False)
    return arr


def trapezoid_weights_1d(coords: ArrayLike) -> Vector:
    """Exact 1D composite trapezoidal weights matching ``np.trapezoid``."""
    x = np.asarray(coords, dtype=float)
    if x.ndim != 1 or x.size < 2 or not np.all(np.isfinite(x)) or np.any(np.diff(x) <= 0.0):
        raise ValueError("trapezoid coordinates must be finite, 1D, and strictly increasing")
    dx = np.diff(x)
    w = np.empty_like(x)
    w[0] = 0.5 * dx[0]
    w[-1] = 0.5 * dx[-1]
    if x.size > 2:
        w[1:-1] = 0.5 * (dx[:-1] + dx[1:])
    return w


@dataclass(frozen=True, slots=True)
class RadiativeReabsorptionSpec2D:
    """Self-consistent uniform absorber radiative recycling spec on 2D stack."""

    layer_id: str
    y_start: int
    y_stop: int
    escape_probability: float
    area_m2: float

    def __post_init__(self) -> None:
        if not self.layer_id or self.y_stop - self.y_start < 2 or self.y_start < 0:
            raise ValueError("radiative reabsorption requires at least two contiguous absorber rows")
        if not math.isfinite(self.escape_probability) or not 0.0 <= self.escape_probability < 1.0:
            raise ValueError("escape_probability must lie in [0, 1)")
        if not math.isfinite(self.area_m2) or self.area_m2 <= 0.0:
            raise ValueError("absorber 2D area must be finite and positive")


def recompute_g_with_rad_2d(
    *,
    G_optical: ArrayLike,
    n: ArrayLike,
    p: ArrayLike,
    B_rad: ArrayLike,
    ni_sq: ArrayLike,
    x: ArrayLike,
    y: ArrayLike,
    absorber_y_ranges: tuple[tuple[int, int], ...],
    absorber_p_esc: tuple[float, ...],
    absorber_areas: tuple[float, ...],
) -> Vector:
    """Return a new ``(Ny, Nx)`` optical+reabsorption generation array."""
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if x_arr.ndim != 1 or y_arr.ndim != 1:
        raise ValueError("x and y must be 1D coordinate vectors")
    shape = (y_arr.size, x_arr.size)
    g_opt = np.asarray(G_optical, dtype=float)
    n_arr = np.asarray(n, dtype=float)
    p_arr = np.asarray(p, dtype=float)
    b_arr = np.asarray(B_rad, dtype=float)
    ni_arr = np.asarray(ni_sq, dtype=float)
    for name, arr in (("G_optical", g_opt), ("n", n_arr), ("p", p_arr), ("B_rad", b_arr), ("ni_sq", ni_arr)):
        if arr.shape != shape:
            raise ValueError(f"shape mismatch for {name}: got {arr.shape}, expected {shape}")
    if not (len(absorber_y_ranges) == len(absorber_p_esc) == len(absorber_areas)):
        raise ValueError("per-absorber reabsorption tuple lengths must match")
    result = g_opt.copy()
    for (y_lo, y_hi), p_esc, area in zip(absorber_y_ranges, absorber_p_esc, absorber_areas):
        if area <= 0.0 or p_esc >= 1.0 or y_hi - y_lo < 2 or x_arr.size < 2:
            continue
        emission = b_arr[y_lo:y_hi, :] * (n_arr[y_lo:y_hi, :] * p_arr[y_lo:y_hi, :] - ni_arr[y_lo:y_hi, :])
        emission_x = np.trapezoid(emission, y_arr[y_lo:y_hi], axis=0)
        r_tot = float(np.trapezoid(emission_x, x_arr))
        if r_tot <= 0.0:
            continue
        result[y_lo:y_hi, :] += r_tot * (1.0 - p_esc) / area
    return result


@dataclass(frozen=True, slots=True)
class TwoSidedInterfaceSRHCoupling2D:
    """Validated horizontal clamp-passive two-sided cross-node SRH sheet."""

    interface_index: int
    interface_row: int
    left_sample_row: int
    right_sample_row: int
    electron_capture_velocity_m_s: float
    hole_capture_velocity_m_s: float
    n1_m3: float
    p1_m3: float
    pair_a_equilibrium_product_m6: float
    pair_b_equilibrium_product_m6: float

    def __post_init__(self) -> None:
        if (self.interface_row <= 0
                or self.left_sample_row != self.interface_row - 1
                or self.right_sample_row != self.interface_row + 1):
            raise ValueError("2D two-sided interface SRH requires interior idx-1/idx+1 sampling")
        for name in ("electron_capture_velocity_m_s", "hole_capture_velocity_m_s"):
            val = float(getattr(self, name))
            if not math.isfinite(val) or val <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("n1_m3", "p1_m3", "pair_a_equilibrium_product_m6", "pair_b_equilibrium_product_m6"):
            val = float(getattr(self, name))
            if not math.isfinite(val) or val < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class TwoSidedInterfaceSRHReport2D:
    """Surface rates for both capture pairs and area-conservative volumetric sink."""

    interface_rows: tuple[int, ...]
    pair_a_surface_rate_m2_s: Vector
    pair_b_surface_rate_m2_s: Vector
    total_surface_rate_m2_s: Vector
    pair_a_clamped: NDArray[np.bool_]
    pair_b_clamped: NDArray[np.bool_]
    volumetric_sink_m3_s: Vector


def interface_srh_sheet_rate_and_partials(
    n_sample: Vector,
    p_sample: Vector,
    eq_product: float,
    n1: float,
    p1: float,
    v_n: float,
    v_p: float,
) -> tuple[Vector, Vector, Vector]:
    """Raw surface SRH rate ``(n*p - ni_eff^2) / ((n+n1)/v_p + (p+p1)/v_n)`` and partials."""
    denom = (n_sample + n1) / v_p + (p_sample + p1) / v_n
    numer = n_sample * p_sample - eq_product
    raw = numer / denom
    d_raw_dn = (p_sample * denom - numer / v_p) / (denom * denom)
    d_raw_dp = (n_sample * denom - numer / v_n) / (denom * denom)
    return raw, d_raw_dn, d_raw_dp


def evaluate_two_sided_interface_srh_2d(
    n: ArrayLike,
    p: ArrayLike,
    y: ArrayLike,
    couplings: tuple[TwoSidedInterfaceSRHCoupling2D, ...],
) -> TwoSidedInterfaceSRHReport2D:
    """Evaluate both cross-node capture directions and construct volumetric sink."""
    electrons = np.asarray(n, dtype=float)
    holes = np.asarray(p, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if (electrons.ndim != 2 or electrons.shape != holes.shape
            or electrons.shape[0] != y_arr.size or y_arr.size < 3
            or not np.all(np.isfinite(electrons)) or not np.all(np.isfinite(holes))):
        raise ValueError("2D interface-SRH carrier arrays must be finite and shape matched")
    if not couplings:
        raise ValueError("at least one two-sided interface-SRH coupling is required")
    rows = tuple(c.interface_row for c in couplings)
    indices = tuple(c.interface_index for c in couplings)
    if len(set(rows)) != len(rows) or len(set(indices)) != len(indices):
        raise ValueError("2D interface-SRH coupling rows and indices must be unique")
    normal_widths = trapezoid_weights_1d(y_arr)
    pair_a = np.empty((len(couplings), electrons.shape[1]), dtype=float)
    pair_b = np.empty_like(pair_a)
    clamped_a = np.empty(pair_a.shape, dtype=bool)
    clamped_b = np.empty(pair_b.shape, dtype=bool)
    sink = np.zeros_like(electrons)
    for idx, c in enumerate(couplings):
        if c.interface_row <= 0 or c.interface_row >= electrons.shape[0] - 1:
            raise ValueError("interface-SRH sheet must use interior idx-1/idx+1 sampling")
        raw_a, _, _ = interface_srh_sheet_rate_and_partials(
            electrons[c.right_sample_row, :], holes[c.left_sample_row, :],
            c.pair_a_equilibrium_product_m6, c.n1_m3, c.p1_m3,
            c.electron_capture_velocity_m_s, c.hole_capture_velocity_m_s,
        )
        raw_b, _, _ = interface_srh_sheet_rate_and_partials(
            np.maximum(electrons[c.left_sample_row, :], 0.0),
            np.maximum(holes[c.right_sample_row, :], 0.0),
            c.pair_b_equilibrium_product_m6, c.n1_m3, c.p1_m3,
            c.electron_capture_velocity_m_s, c.hole_capture_velocity_m_s,
        )
        if not np.all(np.isfinite(raw_a)) or not np.all(np.isfinite(raw_b)):
            raise ValueError("2D interface SRH produced a non-finite surface rate")
        clamped_a[idx, :] = raw_a <= 0.0
        clamped_b[idx, :] = raw_b <= 0.0
        pair_a[idx, :] = np.maximum(raw_a, 0.0)
        pair_b[idx, :] = np.maximum(raw_b, 0.0)
        sink[c.interface_row, :] += (pair_a[idx, :] + pair_b[idx, :]) / normal_widths[c.interface_row]
    total = pair_a + pair_b
    clamped_a.setflags(write=False)
    clamped_b.setflags(write=False)
    return TwoSidedInterfaceSRHReport2D(
        interface_rows=rows,
        pair_a_surface_rate_m2_s=_read_only(pair_a),
        pair_b_surface_rate_m2_s=_read_only(pair_b),
        total_surface_rate_m2_s=_read_only(total),
        pair_a_clamped=clamped_a,
        pair_b_clamped=clamped_b,
        volumetric_sink_m3_s=_read_only(sink),
    )


@dataclass(frozen=True, slots=True)
class IonFluxes2D:
    """Positive-ion particle fluxes on x and y faces [m^-2 s^-1]."""

    x: Vector
    y: Vector


@dataclass(frozen=True, slots=True)
class MobileIonDiagnostics2D:
    """Terminal conservation and physical-bound report for one 2D ion solve."""

    initial_inventory_m1: float
    terminal_inventory_m1: float
    relative_inventory_drift: float
    terminal_min_electron_density_m3: float
    terminal_min_hole_density_m3: float
    terminal_min_density_m3: float
    terminal_max_site_fraction: float
    inventory_rtol: float
    passed: bool
    violations: tuple[str, ...]


def control_volume_areas_2d(x: ArrayLike, y: ArrayLike) -> Vector:
    """Return Neumann-domain nodal control-volume areas [m^2]."""
    return np.outer(trapezoid_weights_1d(y), trapezoid_weights_1d(x))


def ion_inventory_2d(x: ArrayLike, y: ArrayLike, density: ArrayLike) -> float:
    """Return discrete 2D ion inventory per out-of-plane depth [m^-1]."""
    areas = control_volume_areas_2d(x, y)
    values = np.asarray(density, dtype=float)
    if values.shape != areas.shape or not np.all(np.isfinite(values)):
        raise ValueError(f"ion density must be finite with shape {areas.shape}")
    return math.fsum((values * areas).ravel().tolist())


def _harmonic_faces_2d(values: Vector, axis: int) -> Vector:
    left = values[:, :-1] if axis == 1 else values[:-1, :]
    right = values[:, 1:] if axis == 1 else values[1:, :]
    denom = left + right
    return np.divide(2.0 * left * right, denom, out=np.zeros_like(denom), where=denom > 0.0)


def positive_ion_fluxes_2d(
    x: ArrayLike,
    y: ArrayLike,
    phi: ArrayLike,
    density: ArrayLike,
    diffusion: ArrayLike,
    thermal_voltage: float,
    site_limit: ArrayLike,
    *,
    steric_diffusion_only: bool = False,
) -> IonFluxes2D:
    """Positive-ion SG particle fluxes on a tensor-product 2D grid."""
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    _ = trapezoid_weights_1d(x_arr), trapezoid_weights_1d(y_arr)
    shape = (y_arr.size, x_arr.size)
    pot = np.asarray(phi, dtype=float)
    ions = np.asarray(density, dtype=float)
    diff = np.asarray(diffusion, dtype=float)
    lim = np.asarray(site_limit, dtype=float)
    for name, arr in (("phi", pot), ("density", ions), ("diffusion", diff), ("site_limit", lim)):
        if arr.shape != shape or not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} must be finite with shape {shape}")
    if np.any(diff < 0.0) or np.any(lim <= 0.0) or not math.isfinite(thermal_voltage) or thermal_voltage <= 0.0:
        raise ValueError("ionic parameters must have D >= 0, P_lim > 0, and V_T > 0")
    dx = np.diff(x_arr)
    dy = np.diff(y_arr)
    diff_x = _harmonic_faces_2d(diff, axis=1)
    diff_y = _harmonic_faces_2d(diff, axis=0)
    flux_x = np.empty((y_arr.size, x_arr.size - 1), dtype=float)
    for row in range(y_arr.size):
        flux_x[row, :] = ion_face_flux(
            pot[row, :], ions[row, :], dx, diff_x[row, :], thermal_voltage,
            0.5 * (lim[row, :-1] + lim[row, 1:]),
            steric_diffusion_only=steric_diffusion_only, P_lim_node=lim[row, :],
        )
    flux_y = np.empty((y_arr.size - 1, x_arr.size), dtype=float)
    for col in range(x_arr.size):
        flux_y[:, col] = ion_face_flux(
            pot[:, col], ions[:, col], dy, diff_y[:, col], thermal_voltage,
            0.5 * (lim[:-1, col] + lim[1:, col]),
            steric_diffusion_only=steric_diffusion_only, P_lim_node=lim[:, col],
        )
    return IonFluxes2D(x=_read_only(flux_x), y=_read_only(flux_y))


def assess_mobile_ion_terminal_2d(
    x: ArrayLike,
    y: ArrayLike,
    initial_density: ArrayLike,
    terminal_density: ArrayLike,
    site_limit: ArrayLike,
    *,
    terminal_electron_density: ArrayLike,
    terminal_hole_density: ArrayLike,
    inventory_rtol: float = 1.0e-12,
    active_mask: ArrayLike | None = None,
    control_volume_areas_m2: ArrayLike | None = None,
) -> MobileIonDiagnostics2D:
    """Assess physical conservation and positivity/site gates at terminal 2D state."""
    areas = (
        control_volume_areas_2d(x, y)
        if control_volume_areas_m2 is None
        else np.asarray(control_volume_areas_m2, dtype=float)
    )
    init_p = np.asarray(initial_density, dtype=float)
    term_p = np.asarray(terminal_density, dtype=float)
    lim = np.asarray(site_limit, dtype=float)
    n_arr = np.asarray(terminal_electron_density, dtype=float)
    p_arr = np.asarray(terminal_hole_density, dtype=float)
    if any(arr.shape != areas.shape for arr in (init_p, term_p, lim)):
        raise ValueError("ion arrays must match the 2D ion control-volume grid")
    if n_arr.shape != p_arr.shape or n_arr.ndim != 2:
        raise ValueError("carrier density arrays must be matching 2D arrays")
    initial_inv = math.fsum((init_p * areas).ravel().tolist())
    terminal_inv = math.fsum((term_p * areas).ravel().tolist())
    rel_drift = abs(terminal_inv - initial_inv) / max(abs(initial_inv), 1.0e-30)
    min_n = float(np.min(n_arr))
    min_p = float(np.min(p_arr))
    active = lim > 0.0 if active_mask is None else (np.asarray(active_mask, dtype=bool) & (lim > 0.0))
    min_ion = float(np.min(term_p[active])) if np.any(active) else float(np.min(term_p))
    max_frac = float(np.max(np.divide(term_p, lim, out=np.zeros_like(term_p), where=lim > 0.0)))
    violations: list[str] = []
    if not np.all(np.isfinite(n_arr)) or min_n < 0.0:
        violations.append("negative_terminal_electron_density")
    if not np.all(np.isfinite(p_arr)) or min_p < 0.0:
        violations.append("negative_terminal_hole_density")
    if not np.all(np.isfinite(term_p)) or min_ion < 0.0:
        violations.append("negative_terminal_density")
    if not math.isfinite(max_frac) or max_frac > 1.0:
        violations.append("terminal_site_limit_exceeded")
    if not math.isfinite(rel_drift) or rel_drift > inventory_rtol:
        violations.append("inventory_drift_exceeded")
    return MobileIonDiagnostics2D(
        initial_inventory_m1=initial_inv,
        terminal_inventory_m1=terminal_inv,
        relative_inventory_drift=rel_drift,
        terminal_min_electron_density_m3=min_n,
        terminal_min_hole_density_m3=min_p,
        terminal_min_density_m3=min_ion,
        terminal_max_site_fraction=max_frac,
        inventory_rtol=float(inventory_rtol),
        passed=len(violations) == 0,
        violations=tuple(violations),
    )


def finite_site_secant_2d_m3(
    capacity_m3: ArrayLike,
    logit_reference: ArrayLike,
    old_coordinate: ArrayLike,
    new_coordinate: ArrayLike,
) -> Vector:
    """Cancellation-free ``P(new) - P(old)`` for ``P = P_lim * expit(a0 + y)``."""
    cap = np.asarray(capacity_m3, dtype=float)
    a0 = np.asarray(logit_reference, dtype=float)
    y_old = np.asarray(old_coordinate, dtype=float)
    y_new = np.asarray(new_coordinate, dtype=float)
    return _stable_logistic_diff(cap, a0, y_old, y_new)


def _stable_logistic_diff(cap: Vector, a0: Vector, y_old: Vector, y_new: Vector) -> Vector:
    """Exact cancellation-free identity: ``sig(b) - sig(a) = sig(a)*(1 - sig(b))*expm1(b - a)``."""
    dy = y_new - y_old
    sig_old: Vector = np.asarray(expit(a0 + y_old), dtype=np.float64)
    vac_new: Vector = np.asarray(expit(-(a0 + y_new)), dtype=np.float64)
    vac_old: Vector = np.asarray(expit(-(a0 + y_old)), dtype=np.float64)
    sig_new: Vector = np.asarray(expit(a0 + y_new), dtype=np.float64)
    pos = dy >= 0.0
    # For dy >= 0, expm1(-dy) is bounded in [-1, 0]: sig(b) - sig(a) = -sig(b)*(1 - sig(a))*expm1(-dy)
    # For dy < 0, expm1(dy) is bounded in (-1, 0): sig(b) - sig(a) = sig(a)*(1 - sig(b))*expm1(dy)
    result: Vector = np.empty_like(dy, dtype=np.float64)
    result[pos] = -cap[pos] * sig_new[pos] * vac_old[pos] * np.expm1(-dy[pos])
    result[~pos] = cap[~pos] * sig_old[~pos] * vac_new[~pos] * np.expm1(dy[~pos])
    return result


def ion_pair_flux_jacobian_2d(
    phi: Vector,
    density: Vector,
    pairs: NDArray[np.int64],
    spacing: Vector,
    diffusion_faces: Vector,
    thermal_voltage_V: float,
    site_limit: Vector,
    *,
    steric_diffusion_only: bool = False,
    drift_sign: float = 1.0,
) -> IonFaceFluxJacobian:
    """Vectorized SG ionic flux and exact local derivatives on arbitrary 2D face pairs."""
    left, right = pairs[:, 0], pairs[:, 1]
    pot_l, pot_r = phi[left], phi[right]
    den_l, den_r = density[left], density[right]
    lim_l, lim_r = site_limit[left], site_limit[right]
    prefactor = diffusion_faces / spacing
    zero = np.zeros_like(spacing)

    if steric_diffusion_only:
        occ = density / site_limit
        clipped = np.clip(occ, 0.0, 0.999999)
        mu_ex = -np.log1p(-clipped)
        xi = drift_sign * (pot_r - pot_l) / thermal_voltage_V + (mu_ex[right] - mu_ex[left])
        B_pos = bernoulli(xi)
        B_neg = bernoulli(-xi)
        flux = prefactor * (B_pos * den_l - B_neg * den_r)
        interior = (occ > 0.0) & (occ < 0.999999)
        chem_deriv = np.divide(1.0, site_limit * (1.0 - occ), out=np.zeros_like(density), where=interior)
        node_diff = (occ != 0.0) & (occ != 0.999999)
        differentiable = (diffusion_faces == 0.0) | (node_diff[left] & node_diff[right])
        xi_deriv = bernoulli_derivative(xi) * den_l + bernoulli_derivative(-xi) * den_r
        density_left = prefactor * (B_pos - xi_deriv * chem_deriv[left])
        density_right = prefactor * (-B_neg + xi_deriv * chem_deriv[right])
    else:
        face_limit = 0.5 * (lim_l + lim_r)
        avg_den = 0.5 * (den_l + den_r)
        occ = avg_den / face_limit
        clipped = np.clip(occ, 0.0, 0.999999)
        denom = np.maximum(1.0 - clipped, 1.0e-6)
        steric = 1.0 / denom
        xi = drift_sign * (pot_r - pot_l) / thermal_voltage_V
        B_pos = bernoulli(xi)
        B_neg = bernoulli(-xi)
        flux_core = B_pos * den_l - B_neg * den_r
        flux = prefactor * steric * flux_core
        xi_deriv_raw = bernoulli_derivative(xi) * den_l + bernoulli_derivative(-xi) * den_r
        active_clip = (occ > 0.0) & (occ < 0.999999)
        active_max = (1.0 - clipped) > 1.0e-6
        steric_den_deriv = np.divide(0.5, face_limit * denom * denom, out=np.zeros_like(spacing),
                                     where=active_clip & active_max)
        differentiable = (diffusion_faces == 0.0) | (
            (occ != 0.0) & (occ != 0.999999) & ((1.0 - clipped) != 1.0e-6)
        )
        density_left = prefactor * (steric * B_pos + flux_core * steric_den_deriv)
        density_right = prefactor * (-steric * B_neg + flux_core * steric_den_deriv)
        xi_deriv = steric * xi_deriv_raw

    potential_scale = prefactor * xi_deriv * drift_sign / thermal_voltage_V
    for arr in (flux, density_left, density_right, potential_scale):
        if not np.all(np.isfinite(arr)):
            raise ValueError("2D ion-flux Jacobian produced a non-finite derivative")
    return IonFaceFluxJacobian(
        flux=flux,
        density_left_derivative=density_left,
        density_right_derivative=density_right,
        partner_left_derivative=zero,
        partner_right_derivative=zero.copy(),
        potential_left_derivative=-potential_scale,
        potential_right_derivative=potential_scale,
        differentiable_faces=differentiable,
    )


@dataclass(frozen=True)
class IonicState2D:
    """Physical trial state bound to the complete 2D single-mobile-ion device."""

    u: Vector
    electronic_state: State2D
    positive_ion_density_m3: Vector
    device_sha256: str
    domain_inventory_m1: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "u", frozen_array(self.u))
        object.__setattr__(self, "positive_ion_density_m3", frozen_array(self.positive_ion_density_m3))
        elec_count = self.electronic_state.potential_V.size
        ion_count = self.positive_ion_density_m3.size
        if (self.u.shape != (3 * elec_count + ion_count,)
                or not np.array_equal(self.u[:3 * elec_count], self.electronic_state.u)):
            raise ValueError("2D ionic coordinates must include their complete electronic component")
        if (self.positive_ion_density_m3.ndim != 2
                or self.positive_ion_density_m3.shape[1] != self.electronic_state.potential_V.shape[1]
                or self.positive_ion_density_m3.shape[0] < 3
                or np.any(self.positive_ion_density_m3 <= 0.0)):
            raise ValueError("2D positive-ion density must be positive on its declared (Ny_ion, Nx) domain")
        if not self.device_sha256 or not math.isfinite(self.domain_inventory_m1) or self.domain_inventory_m1 <= 0.0:
            raise ValueError("2D ionic state requires complete device identity and finite positive inventory")

    @property
    def voltage_V(self) -> float:
        return self.electronic_state.voltage_V


@dataclass(frozen=True)
class CompiledIonicDevice2D:
    """One 2D Neumann electronic component and one closed 4-side-blocking 2D ion domain."""

    device_id: str
    sha256: str
    original_prepared_sha256: str
    electronic_component: CompiledDevice2D
    ion_geometry: RectangularGeometry2D
    domain_layer_id: str
    ion_row_indices: NDArray[np.intp]
    positive_ion_reference_m3: Vector
    positive_ion_site_limit_m3: Vector
    positive_ion_logit_reference: Vector
    positive_ion_diffusivity_faces_m2_s: Vector
    steric_model: str
    binding_manifest_json: str
    radiative_reabsorption: tuple[RadiativeReabsorptionSpec2D, ...] = ()
    interface_srh_couplings: tuple[TwoSidedInterfaceSRHCoupling2D, ...] = ()
    capabilities: tuple[str, ...] = (
        "classical_2d_electronic_component",
        "single_positive_ion_2d_residual",
        "finite_sites",
        "blocking_4_side_ion_contacts",
        "physical_ion_storage_2d",
    )
    ion_node_indices: NDArray[np.intp] = field(init=False)
    layout: Layout = field(init=False, repr=False)

    def __post_init__(self) -> None:
        comp = self.electronic_component
        if comp.geometry.name != RECTANGULAR_HALF_ENDPOINT_2D_V1 or self.ion_geometry.name != RECTANGULAR_HALF_ENDPOINT_2D_V1:
            raise CapabilityError("2D mobile-ion requires Neumann half-endpoint rectangular geometry")
        if any(ch.kind != "fixed_reservoir" for ch in comp.column.channels):
            raise CapabilityError("selective_robin_ionic_2d_unsupported: 2D mobile-ion requires ohmic fixed-reservoir contacts")
        rows = integer_indices(self.ion_row_indices)
        if (rows.ndim != 1 or rows.size != self.ion_geometry.ny or rows.size < 3
                or np.any(np.diff(rows) != 1) or rows[0] < 0 or rows[-1] >= comp.geometry.ny):
            raise ValueError("ion_row_indices must be ordered, contiguous and inside the electronic stack")
        if not np.array_equal(self.ion_geometry.x_m, comp.geometry.x_m):
            raise ValueError("ion and electronic lateral grids must coincide")
        if not np.array_equal(self.ion_geometry.y_geometry.nodes_m, comp.geometry.y_geometry.nodes_m[rows]):
            raise ValueError("ion and electronic vertical nodes must coincide on ion_row_indices")
        object.__setattr__(self, "ion_row_indices", np.frombuffer(rows.tobytes(), dtype=np.intp))
        nx = comp.geometry.nx
        nodes = (rows[:, None] * nx + np.arange(nx, dtype=np.intp)[None, :]).ravel()
        object.__setattr__(self, "ion_node_indices", np.frombuffer(nodes.tobytes(), dtype=np.intp))
        ion_count = self.ion_geometry.node_count
        for name in ("positive_ion_reference_m3", "positive_ion_site_limit_m3", "positive_ion_logit_reference"):
            arr = frozen_array(getattr(self, name))
            if arr.shape != (ion_count,):
                raise ValueError(f"{name} must match the 2D ion node count")
            object.__setattr__(self, name, arr)
        if (np.any(self.positive_ion_reference_m3 <= 0.0)
                or np.any(self.positive_ion_reference_m3 >= self.positive_ion_site_limit_m3)):
            raise ValueError("finite-site ion reference must lie strictly between 0 and P_lim")
        faces = frozen_array(self.positive_ion_diffusivity_faces_m2_s)
        expected_faces = self.ion_geometry.face_spacing_m.size
        if faces.shape != (expected_faces,) or np.any(faces < 0.0):
            raise ValueError("positive_ion_diffusivity_faces_m2_s must match 2D ion faces and be non-negative")
        object.__setattr__(self, "positive_ion_diffusivity_faces_m2_s", faces)
        if self.steric_model not in {WHOLE_FLUX_STERIC_V1, DIFFUSION_ONLY_STERIC_V1}:
            raise ValueError("unsupported 2D ionic steric model")
        if not self.sha256 or not self.original_prepared_sha256 or not self.domain_layer_id:
            raise ValueError("2D ionic device requires complete provenance identities")
        elec_faces = comp.geometry.face_spacing_m.size
        object.__setattr__(self, "layout", Layout(
            (
                Support("cells", "cell", (comp.node_count,)),
                Support("faces", "face", (elec_faces,)),
                Support("ion_cells", "cell", (ion_count,)),
                Support("ion_faces", "face", (expected_faces,)),
            ),
            (),
            (
                EquationSpec("gauss", "electrostatics", "cells", (comp.node_count,), COULOMB / LENGTH, role="constraint"),
                EquationSpec("electron", "electron", "cells", (comp.node_count,), PARTICLE / (LENGTH * SECOND)),
                EquationSpec("hole", "hole", "cells", (comp.node_count,), PARTICLE / (LENGTH * SECOND)),
                EquationSpec("positive_ion", "positive_ion", "ion_cells", (ion_count,), PARTICLE / (LENGTH * SECOND)),
            ),
        ))

    @property
    def identity(self) -> str:
        return self.sha256

    @property
    def node_count(self) -> int:
        return self.electronic_component.node_count

    @property
    def ion_node_count(self) -> int:
        return self.ion_geometry.node_count

    @property
    def state_size(self) -> int:
        return 3 * self.node_count + self.ion_node_count

    @property
    def ion_species_names(self) -> tuple[str, ...]:
        return ("positive",)

    @property
    def ion_species_slices(self) -> tuple[slice, ...]:
        start = 3 * self.node_count
        return (slice(start, start + self.ion_node_count),)

    @property
    def binding_manifest(self) -> dict[str, Any]:
        loaded = json.loads(self.binding_manifest_json)
        if not isinstance(loaded, dict):
            raise ValueError("binding_manifest_json must decode to an object")
        return loaded

    @property
    def ion_charge_weight_ratio(self) -> Vector:
        elec_vol = self.electronic_component.geometry.contract_geometry.volumes[self.ion_node_indices]
        return frozen_array(self.ion_geometry.contract_geometry.volumes / elec_vol)

    @property
    def reference_inventory_m1(self) -> float:
        return math.fsum(
            (self.ion_geometry.contract_geometry.volumes * self.positive_ion_reference_m3).tolist()
        )

    def at_temperature(self, temperature_K: float) -> CompiledIonicDevice2D:
        """Verify or evaluate at the compiled device temperature."""
        if not math.isclose(temperature_K, self.electronic_component.column.parameters.temperature_K, rel_tol=0.0, abs_tol=1e-12):
            raise CapabilityError("2D ionic device is bound to its compiled temperature; recompile for a new T")
        return self

    def fields(self, u: ArrayLike, *, voltage_V: float = 0.0) -> tuple[Vector, Vector, Vector, Vector]:
        values = np.asarray(u, dtype=float)
        if values.shape != (self.state_size,) or not np.all(np.isfinite(values)):
            raise ValueError("coordinates must be finite and match the 2D ionic state layout")
        elec_size = 3 * self.node_count
        phi, n, p = self.electronic_component.fields(values[:elec_size], voltage_V=voltage_V)
        y_ion = values[elec_size:]
        try:
            ion_p = single_site_density(
                self.positive_ion_reference_m3,
                self.positive_ion_site_limit_m3,
                self.positive_ion_logit_reference,
                y_ion,
            )
        except ValueError as exc:
            raise FloatingPointError(str(exc)) from exc
        if not np.all(np.isfinite(ion_p)) or np.any(ion_p <= 0.0) or np.any(ion_p >= self.positive_ion_site_limit_m3):
            raise FloatingPointError("2D ion coordinate is outside the open finite-site interval (0, P_lim)")
        return phi, n, p, ion_p

    def state(self, u: ArrayLike, *, voltage_V: float = 0.0) -> IonicState2D:
        values = np.asarray(u, dtype=float)
        elec_size = 3 * self.node_count
        elec_state = self.electronic_component.state(values[:elec_size], voltage_V=voltage_V)
        _, _, _, ion_p = self.fields(values, voltage_V=voltage_V)
        inventory = math.fsum((self.ion_geometry.contract_geometry.volumes * ion_p).tolist())
        return IonicState2D(
            values,
            elec_state,
            ion_p.reshape(self.ion_geometry.shape),
            self.sha256,
            inventory,
        )

    def full_grid_ion_density(self, ion_density_domain: ArrayLike) -> Vector:
        """Embed ``(Ny_ion, Nx)`` or ``(N_ion,)`` ion density onto ``(Ny, Nx)`` stack."""
        full: Vector = np.zeros(self.electronic_component.geometry.shape, dtype=np.float64)
        full[self.ion_row_indices, :] = np.asarray(ion_density_domain, dtype=np.float64).reshape(self.ion_geometry.shape)
        return full

    def full_grid_ion_diffusivity_m2_s(self) -> Vector:
        """Nodal ion diffusivity on ``(Ny, Nx)`` stack for current/diagnostic helpers."""
        full: Vector = np.zeros(self.electronic_component.geometry.shape, dtype=np.float64)
        manifest = self.binding_manifest
        d_nodal = float(manifest["ion_model"]["scaled_nodal_diffusivity_m2_s"])
        full[self.ion_row_indices, :] = d_nodal
        return full

    def full_grid_ion_site_limit_m3(self) -> Vector:
        """Positive site limit on ``(Ny, Nx)`` stack (positive floor outside ion domain)."""
        cap = float(self.positive_ion_site_limit_m3[0])
        full: Vector = np.full(self.electronic_component.geometry.shape, cap, dtype=np.float64)
        full[self.ion_row_indices, :] = self.positive_ion_site_limit_m3.reshape(self.ion_geometry.shape)
        return full

    def charge_ledger(self, n: Vector, p: Vector, ion_density_m3: Vector) -> ChargeLedger:
        comp = self.electronic_component
        ion_pos: Vector = np.zeros(self.node_count, dtype=np.float64)
        ion_ref: Vector = np.zeros(self.node_count, dtype=np.float64)
        ratio = self.ion_charge_weight_ratio
        ion_pos[self.ion_node_indices] = ion_density_m3 * ratio
        ion_ref[self.ion_node_indices] = self.positive_ion_reference_m3 * ratio
        return ChargeLedger(self.node_count, 0, (
            ChargeEntry("hole", "hole", 1, p),
            ChargeEntry("electron", "electron", -1, n),
            ChargeEntry("positive_ion", "ion_charge_projection", 1, ion_pos, ion_ref),
            ChargeEntry("acceptor", "dopant", -1, comp.extrude(comp.column.parameters.acceptor_density_m3)),
            ChargeEntry("donor", "dopant", 1, comp.extrude(comp.column.parameters.donor_density_m3)),
        ))


@dataclass(frozen=True)
class Ionic2DProblem:
    """Reference legacy-free 2D single-mobile-ion DAE residual and tangent matrices."""

    device: CompiledIonicDevice2D
    voltage_V: float = 0.0
    generation_scale: float = 0.0

    def __post_init__(self) -> None:
        self.device.electronic_component.column.contact_potentials(self.voltage_V)
        if not math.isfinite(self.generation_scale) or self.generation_scale < 0.0:
            raise ValueError("generation_scale must be finite and nonnegative")

    def carrier_fluxes(self, u: ArrayLike) -> CarrierFaceFluxes:
        comp = self.device.electronic_component
        values = np.asarray(u, dtype=float)[:3 * comp.node_count]
        phi, n, p = comp.fields(values, voltage_V=self.voltage_V)
        if has_prospective_2d_transport(comp):
            return evaluate_2d_carrier_face_fluxes(
                comp,
                phi,
                n,
                p,
                values[comp.node_count:2 * comp.node_count],
                values[2 * comp.node_count:3 * comp.node_count],
                derivatives=False,
            ).fluxes
        left, right = comp.geometry.contract_geometry.pairs.T
        potentials = comp.band_structure.transport_potentials(phi)
        if comp.field_transport is not None:
            return comp.field_transport.fluxes(
                phi, n, p, potentials[0], potentials[1],
                electron_electrochemical_step=values[comp.node_count + right] - values[comp.node_count + left],
                hole_electrochemical_step=-(values[2 * comp.node_count + right] - values[2 * comp.node_count + left]),
            )
        fluxes = []
        for density, potential, diffusivity, offset, sign in (
            (n, potentials[0], comp.electron_diffusivity_faces_m2_s, comp.node_count, 1.0),
            (p, potentials[1], comp.hole_diffusivity_faces_m2_s, 2 * comp.node_count, -1.0),
        ):
            step = -sign * (potential[right] - potential[left]) / comp.thermal_voltage_V
            coefficient = diffusivity / comp.geometry.face_spacing_m
            drive = sign * (values[offset + right] - values[offset + left])
            evaluated = stable_sg_particle_flux(density[left], density[right], step, coefficient,
                                                electrochemical_step=drive)
            per_volt = -sign / comp.thermal_voltage_V * evaluated.energy_step_derivative
            fluxes.append(CarrierFaceFlux(
                evaluated.flux, evaluated.left_density_derivative, evaluated.right_density_derivative,
                -per_volt, per_volt, step, coefficient,
            ))
        return CarrierFaceFluxes(fluxes[0], fluxes[1])

    def ion_flux_evaluation(self, u: ArrayLike) -> IonFaceFluxJacobian:
        d = self.device
        phi, _, _, ion_p = d.fields(u, voltage_V=self.voltage_V)
        phi_ion = phi[d.ion_node_indices]
        return ion_pair_flux_jacobian_2d(
            phi_ion,
            ion_p,
            d.ion_geometry.contract_geometry.pairs,
            d.ion_geometry.face_spacing_m,
            d.positive_ion_diffusivity_faces_m2_s,
            d.electronic_component.thermal_voltage_V,
            d.positive_ion_site_limit_m3,
            steric_diffusion_only=(d.steric_model == DIFFUSION_ONLY_STERIC_V1),
            drift_sign=1.0,
        )

    def effective_generation_m3_s(self, n: Vector, p: Vector) -> Vector:
        d = self.device
        comp = d.electronic_component
        base = self.generation_scale * comp.generation_source_m3_s
        if not d.radiative_reabsorption:
            return base
        shape = comp.geometry.shape
        b_rad = comp.recombination.coefficients.radiative_m3_s.reshape(shape)
        ni_sq = comp.recombination.coefficients.ni_sq_m6.reshape(shape)
        augmented = recompute_g_with_rad_2d(
            G_optical=base.reshape(shape),
            n=n.reshape(shape),
            p=p.reshape(shape),
            B_rad=b_rad,
            ni_sq=ni_sq,
            x=comp.geometry.x_m,
            y=comp.geometry.y_geometry.nodes_m,
            absorber_y_ranges=tuple((s.y_start, s.y_stop) for s in d.radiative_reabsorption),
            absorber_p_esc=tuple(s.escape_probability for s in d.radiative_reabsorption),
            absorber_areas=tuple(s.area_m2 for s in d.radiative_reabsorption),
        )
        return augmented.ravel()

    def recombination_rate(self, n: Vector, p: Vector, *, derivatives: bool = True) -> RecombinationRate:
        comp = self.device.electronic_component
        evaluation = comp.recombination.evaluate(n, p, derivatives=derivatives)
        evaluation.require_domain()
        if comp.grain_recombination is not None:
            return comp.grain_recombination.evaluate(evaluation, comp.recombination.coefficients)
        return evaluation.total

    def interface_srh_sink_m3_s(self, n: Vector, p: Vector) -> Vector:
        d = self.device
        if not d.interface_srh_couplings:
            return np.zeros_like(n)
        shape = d.electronic_component.geometry.shape
        report = evaluate_two_sided_interface_srh_2d(
            n.reshape(shape), p.reshape(shape),
            d.electronic_component.geometry.y_geometry.nodes_m,
            d.interface_srh_couplings,
        )
        return report.volumetric_sink_m3_s.ravel()

    def displacement(self, u: ArrayLike) -> Vector:
        comp = self.device.electronic_component
        phi, _, _, _ = self.device.fields(u, voltage_V=self.voltage_V)
        left, right = comp.geometry.contract_geometry.pairs.T
        return -comp.face_capacitance_F_m2 * (phi[right] - phi[left])

    def residual(self, u: ArrayLike) -> Vector:
        d = self.device
        comp = d.electronic_component
        count = comp.node_count
        values = np.asarray(u, dtype=float)
        try:
            phi, n, p, ion_p = d.fields(values, voltage_V=self.voltage_V)
        except (ValueError, FloatingPointError):
            return np.full(d.state_size, np.nan)
        if not (np.all(np.isfinite(n)) and np.all(np.isfinite(p)) and np.all(n > 0.0) and np.all(p > 0.0)):
            return np.full(d.state_size, np.nan)
        c_fluxes = self.carrier_fluxes(values)
        ion_eval = self.ion_flux_evaluation(values)
        gen = self.effective_generation_m3_s(n, p)
        rec = self.recombination_rate(n, p, derivatives=False).rate + self.interface_srh_sink_m3_s(n, p)
        source = gen - rec
        outputs: list[Vector] = []
        for name, flux, flux_unit, cell_source, source_unit, geom in (
            ("gauss", self.displacement(values), COULOMB / AREA,
             d.charge_ledger(n, p, ion_p).space_charge_C_m3(), COULOMB / VOLUME, comp.geometry.contract_geometry),
            ("electron", c_fluxes.electron.flux, PARTICLE / (AREA * SECOND),
             source, PARTICLE / (VOLUME * SECOND), comp.geometry.contract_geometry),
            ("hole", c_fluxes.hole.flux, PARTICLE / (AREA * SECOND),
             source, PARTICLE / (VOLUME * SECOND), comp.geometry.contract_geometry),
        ):
            sink = TermSink(d.layout, name, geom)
            sink.add(FaceFlux(name + ".transport", "faces", flux, flux_unit))
            sink.add(CellSource(name + ".source", "cells", cell_source, source_unit))
            outputs.append(np.array(sink.value(), copy=True))
        ig = d.ion_geometry.contract_geometry
        ion_geom = Geometry(
            ig.volumes, ig.pairs, ig.face_measures, AREA, LENGTH,
            cell_support="ion_cells", face_support="ion_faces",
        )
        ion_sink = TermSink(d.layout, "positive_ion", ion_geom)
        ion_sink.add(FaceFlux("positive_ion.transport", "ion_faces", ion_eval.flux, PARTICLE / (AREA * SECOND)))
        outputs.append(np.array(ion_sink.value(), copy=True))

        bottom, top = comp.column.contact_potentials(self.voltage_V)
        outputs[0][:comp.geometry.nx] = phi[:comp.geometry.nx] - bottom
        outputs[0][-comp.geometry.nx:] = phi[-comp.geometry.nx:] - top
        sides = comp.geometry.contact_nodes.reshape(2, comp.geometry.nx)
        for rows, offset, sign, reference in zip(
            outputs[1:3], (count, 2 * count), (1.0, -1.0), comp.reference_potentials(self.voltage_V)
        ):
            for nodes in sides:
                rows[nodes] = sign * ((phi[nodes] - reference[nodes]) / comp.thermal_voltage_V + values[offset + nodes])
        return np.concatenate(outputs)

    def storage(self, u: ArrayLike, *, physical: bool = False) -> Vector:
        d = self.device
        comp = d.electronic_component
        count = comp.node_count
        _, n, p, ion_p = d.fields(u, voltage_V=self.voltage_V)
        elec_vol = comp.geometry.contract_geometry.volumes
        ion_vol = d.ion_geometry.contract_geometry.volumes
        s_n = elec_vol * n
        s_p = elec_vol * p
        if not physical:
            mask = comp.fixed_carrier_mask
            s_n[mask[0]] = 0.0
            s_p[mask[1]] = 0.0
        return np.concatenate((np.zeros(count), s_n, s_p, ion_vol * ion_p))

    def storage_tangent(self, u: ArrayLike) -> csc_matrix:
        """``S_u`` on non-constraint rows; ``mass_matrix = -S_u``."""
        d = self.device
        comp = d.electronic_component
        count = comp.node_count
        _, n, p, ion_p = d.fields(u, voltage_V=self.voltage_V)
        thermal = comp.thermal_voltage_V
        elec_vol = comp.geometry.contract_geometry.volumes
        ion_vol = d.ion_geometry.contract_geometry.volumes
        mask = comp.fixed_carrier_mask
        active_n = ~mask[0]
        active_p = ~mask[1]
        nodes: NDArray[np.intp] = np.arange(count, dtype=np.intp)
        n_nodes = nodes[active_n]
        p_nodes = nodes[active_p]
        ion_nodes: NDArray[np.intp] = np.arange(d.ion_node_count, dtype=np.intp)
        dp_dy = ion_p * (1.0 - ion_p / d.positive_ion_site_limit_m3)
        rows = np.concatenate((
            count + n_nodes, count + n_nodes,
            2 * count + p_nodes, 2 * count + p_nodes,
            3 * count + ion_nodes,
        ))
        cols = np.concatenate((
            n_nodes, count + n_nodes,
            p_nodes, 2 * count + p_nodes,
            3 * count + ion_nodes,
        ))
        vals = np.concatenate((
            elec_vol[active_n] * n[active_n] / thermal,
            elec_vol[active_n] * n[active_n],
            -elec_vol[active_p] * p[active_p] / thermal,
            -elec_vol[active_p] * p[active_p],
            ion_vol * dp_dy,
        ))
        return csc_matrix((vals, (rows, cols)), shape=(d.state_size, d.state_size))

    def storage_jacobian(self, u: ArrayLike) -> csc_matrix:
        """Signed DAE mass matrix ``M = -S_u``."""
        return -self.storage_tangent(u)

    def jacobian(self, u: ArrayLike) -> csc_matrix:
        """Full sparse analytic Jacobian via vectorized COO-to-CSC assembly (no dense matrix)."""
        d = self.device
        comp = d.electronic_component
        count = comp.node_count
        values = np.asarray(u, dtype=float)
        phi, n, p, ion_p = d.fields(values, voltage_V=self.voltage_V)
        thermal = comp.thermal_voltage_V
        geom = comp.geometry.contract_geometry
        left, right = geom.pairs[:, 0], geom.pairs[:, 1]

        row_chunks: list[NDArray[np.intp]] = []
        col_chunks: list[NDArray[np.intp]] = []
        val_chunks: list[Vector] = []

        def emit(r: ArrayLike, c: ArrayLike, v: ArrayLike) -> None:
            row_chunks.append(np.asarray(r, dtype=np.intp).ravel())
            col_chunks.append(np.asarray(c, dtype=np.intp).ravel())
            val_chunks.append(np.asarray(v, dtype=float).ravel())

        # 1. Gauss rows (interior only; contact rows are Dirichlet)
        cap_w = comp.face_capacitance_F_m2 * geom.face_measures
        emit(left, left, -cap_w)
        emit(left, right, cap_w)
        emit(right, left, cap_w)
        emit(right, right, -cap_w)

        nodes: NDArray[np.intp] = np.arange(count, dtype=np.intp)
        vol = geom.volumes
        emit(nodes, nodes, -Q * vol * (n + p) / thermal)
        emit(nodes, count + nodes, -Q * vol * n)
        emit(nodes, 2 * count + nodes, -Q * vol * p)
        dp_dy = ion_p * (1.0 - ion_p / d.positive_ion_site_limit_m3)
        ion_vol = d.ion_geometry.contract_geometry.volumes
        emit(d.ion_node_indices, 3 * count + np.arange(d.ion_node_count, dtype=np.intp), Q * ion_vol * dp_dy)

        # 2. Carrier transport + bulk/grain recombination + optional reabsorption + optional cross-node SRH
        rate = self.recombination_rate(n, p, derivatives=True)
        assert rate.d_rate_dn is not None and rate.d_rate_dp is not None
        source_n = -np.asarray(rate.d_rate_dn) * n
        source_p = np.asarray(rate.d_rate_dp) * p
        source_phi = (source_n + source_p) / thermal

        if has_prospective_2d_transport(comp):
            prospective = evaluate_2d_carrier_face_fluxes(
                comp,
                phi,
                n,
                p,
                values[count:2 * count],
                values[2 * count:3 * count],
                derivatives=True,
            )
            fm = geom.face_measures
            for offset, phi_left, phi_right, z_left, z_right in (
                (
                    count,
                    prospective.e_phi_left,
                    prospective.e_phi_right,
                    prospective.e_z_left,
                    prospective.e_z_right,
                ),
                (
                    2 * count,
                    prospective.h_phi_left,
                    prospective.h_phi_right,
                    prospective.h_z_left,
                    prospective.h_z_right,
                ),
            ):
                emit(offset + left, left, -fm * phi_left)
                emit(offset + left, right, -fm * phi_right)
                emit(offset + left, offset + left, -fm * z_left)
                emit(offset + left, offset + right, -fm * z_right)
                emit(offset + right, left, fm * phi_left)
                emit(offset + right, right, fm * phi_right)
                emit(offset + right, offset + left, fm * z_left)
                emit(offset + right, offset + right, fm * z_right)
                emit(offset + nodes, nodes, vol * source_phi)
                emit(offset + nodes, count + nodes, vol * source_n)
                emit(offset + nodes, 2 * count + nodes, vol * source_p)
        else:
            c_fluxes = self.carrier_fluxes(values)
            evaluated_transport = None if comp.field_transport is None else comp.field_transport.evaluate(phi)
            for offset, sign, density, flux in (
                (count, 1.0, n, c_fluxes.electron),
                (2 * count, -1.0, p, c_fluxes.hole),
            ):
                drive = sign * (values[offset + right] - values[offset + left])
                phi_left, phi_right = quasi_fermi_potential_tangents(flux, drive, sign, thermal)
                if evaluated_transport is not None:
                    diff = (evaluated_transport.electron_diffusivity_m2_s if offset == count
                            else evaluated_transport.hole_diffusivity_m2_s)
                    slope = (evaluated_transport.electron_d_diffusivity_d_field if offset == count
                             else evaluated_transport.hole_d_diffusivity_d_field)
                    field_chain = (flux.flux / diff) * slope / comp.geometry.face_spacing_m
                    phi_left = phi_left + field_chain
                    phi_right = phi_right - field_chain
                z_left = sign * flux.left_density_derivative * density[left]
                z_right = sign * flux.right_density_derivative * density[right]
                fm = geom.face_measures
                emit(offset + left, left, -fm * phi_left)
                emit(offset + left, right, -fm * phi_right)
                emit(offset + left, offset + left, -fm * z_left)
                emit(offset + left, offset + right, -fm * z_right)
                emit(offset + right, left, fm * phi_left)
                emit(offset + right, right, fm * phi_right)
                emit(offset + right, offset + left, fm * z_left)
                emit(offset + right, offset + right, fm * z_right)
                emit(offset + nodes, nodes, vol * source_phi)
                emit(offset + nodes, count + nodes, vol * source_n)
                emit(offset + nodes, 2 * count + nodes, vol * source_p)

        # Optional self-consistent radiative reabsorption nonlocal rank-1 coupling
        if d.radiative_reabsorption:
            shape = comp.geometry.shape
            n_2d, p_2d = n.reshape(shape), p.reshape(shape)
            b_2d = comp.recombination.coefficients.radiative_m3_s.reshape(shape)
            ni_2d = comp.recombination.coefficients.ni_sq_m6.reshape(shape)
            wx_trap = trapezoid_weights_1d(comp.geometry.x_m)
            for spec in d.radiative_reabsorption:
                y_slice = slice(spec.y_start, spec.y_stop)
                emission = b_2d[y_slice, :] * (n_2d[y_slice, :] * p_2d[y_slice, :] - ni_2d[y_slice, :])
                r_tot = float(np.trapezoid(np.trapezoid(emission, comp.geometry.y_geometry.nodes_m[y_slice], axis=0),
                                           comp.geometry.x_m))
                if r_tot <= 0.0:
                    continue
                wy_trap = trapezoid_weights_1d(comp.geometry.y_geometry.nodes_m[y_slice])
                quad_w = (wy_trap[:, None] * wx_trap[None, :]).ravel()
                abs_nodes = (np.arange(spec.y_start, spec.y_stop, dtype=np.intp)[:, None] * comp.geometry.nx
                             + np.arange(comp.geometry.nx, dtype=np.intp)[None, :]).ravel()
                # d(n*p)/dphi == 0 identically; d(n*p)/dz_n = +n*p; d(n*p)/dz_p = -n*p
                grad_zn = (1.0 - spec.escape_probability) / spec.area_m2 * quad_w * b_2d[y_slice, :].ravel() * (
                    n_2d[y_slice, :].ravel() * p_2d[y_slice, :].ravel()
                )
                row_areas = vol[abs_nodes]
                rr: NDArray[np.intp]
                cc: NDArray[np.intp]
                rr, cc = np.meshgrid(abs_nodes, abs_nodes, indexing="ij")
                outer = row_areas[:, None] * grad_zn[None, :]
                for offset in (count, 2 * count):
                    emit(offset + rr, count + cc, outer)
                    emit(offset + rr, 2 * count + cc, -outer)

        # Optional clamp-passive two-sided cross-node interface SRH derivatives
        if d.interface_srh_couplings:
            shape = comp.geometry.shape
            nx = comp.geometry.nx
            n_2d, p_2d = n.reshape(shape), p.reshape(shape)
            normal_widths = trapezoid_weights_1d(comp.geometry.y_geometry.nodes_m)
            cols_x: NDArray[np.intp] = np.arange(nx, dtype=np.intp)
            for c in d.interface_srh_couplings:
                row_nodes: NDArray[np.intp] = c.interface_row * nx + cols_x
                left_nodes: NDArray[np.intp] = c.left_sample_row * nx + cols_x
                right_nodes: NDArray[np.intp] = c.right_sample_row * nx + cols_x
                scale = vol[row_nodes] / normal_widths[c.interface_row]

                raw_a, da_dn, da_dp = interface_srh_sheet_rate_and_partials(
                    n_2d[c.right_sample_row, :], p_2d[c.left_sample_row, :],
                    c.pair_a_equilibrium_product_m6, c.n1_m3, c.p1_m3,
                    c.electron_capture_velocity_m_s, c.hole_capture_velocity_m_s,
                )
                act_a = (raw_a > 0.0).astype(float)
                # Sink enters carrier row with minus sign (-sink * area)
                dn_r = -scale * act_a * da_dn * n_2d[c.right_sample_row, :]
                dp_l = -scale * act_a * da_dp * p_2d[c.left_sample_row, :]

                raw_b, db_dn, db_dp = interface_srh_sheet_rate_and_partials(
                    n_2d[c.left_sample_row, :], p_2d[c.right_sample_row, :],
                    c.pair_b_equilibrium_product_m6, c.n1_m3, c.p1_m3,
                    c.electron_capture_velocity_m_s, c.hole_capture_velocity_m_s,
                )
                act_b = (raw_b > 0.0).astype(float)
                dn_l = -scale * act_b * db_dn * n_2d[c.left_sample_row, :]
                dp_r = -scale * act_b * db_dp * p_2d[c.right_sample_row, :]

                for offset in (count, 2 * count):
                    emit(offset + row_nodes, right_nodes, (dn_r - dp_r) / thermal)
                    emit(offset + row_nodes, count + right_nodes, dn_r)
                    emit(offset + row_nodes, 2 * count + right_nodes, -dp_r)
                    emit(offset + row_nodes, left_nodes, (dn_l - dp_l) / thermal)
                    emit(offset + row_nodes, count + left_nodes, dn_l)
                    emit(offset + row_nodes, 2 * count + left_nodes, -dp_l)

        # 3. Positive-ion 4-side-blocking face divergence derivatives
        ion_eval = self.ion_flux_evaluation(values)
        ion_geom = d.ion_geometry.contract_geometry
        i_left, i_right = ion_geom.pairs[:, 0], ion_geom.pairs[:, 1]
        i_fm = ion_geom.face_measures
        e_left = d.ion_node_indices[i_left]
        e_right = d.ion_node_indices[i_right]
        ion_offset = 3 * count
        y_left = ion_eval.density_left_derivative * dp_dy[i_left]
        y_right = ion_eval.density_right_derivative * dp_dy[i_right]
        emit(ion_offset + i_left, e_left, -i_fm * ion_eval.potential_left_derivative)
        emit(ion_offset + i_left, e_right, -i_fm * ion_eval.potential_right_derivative)
        emit(ion_offset + i_left, ion_offset + i_left, -i_fm * y_left)
        emit(ion_offset + i_left, ion_offset + i_right, -i_fm * y_right)
        emit(ion_offset + i_right, e_left, i_fm * ion_eval.potential_left_derivative)
        emit(ion_offset + i_right, e_right, i_fm * ion_eval.potential_right_derivative)
        emit(ion_offset + i_right, ion_offset + i_left, i_fm * y_left)
        emit(ion_offset + i_right, ion_offset + i_right, i_fm * y_right)

        # Zero replaced Dirichlet / fixed-reservoir rows and insert exact constraints
        constraint_mask: NDArray[np.bool_] = np.zeros(d.state_size, dtype=bool)
        contact_nodes = comp.geometry.contact_nodes.astype(np.intp)
        constraint_mask[contact_nodes] = True
        fixed_mask = comp.fixed_carrier_mask
        constraint_mask[count:2 * count] = fixed_mask[0]
        constraint_mask[2 * count:3 * count] = fixed_mask[1]

        all_r = np.concatenate(row_chunks)
        all_c = np.concatenate(col_chunks)
        all_v = np.concatenate(val_chunks)
        keep = ~constraint_mask[all_r]
        kept_r = [all_r[keep], contact_nodes]
        kept_c = [all_c[keep], contact_nodes]
        kept_v = [all_v[keep], np.ones(contact_nodes.size, dtype=float)]
        for offset, sign, mask_row in ((count, 1.0, fixed_mask[0]), (2 * count, -1.0, fixed_mask[1])):
            c_nodes = nodes[mask_row]
            kept_r.extend((offset + c_nodes, offset + c_nodes))
            kept_c.extend((c_nodes, offset + c_nodes))
            kept_v.extend((np.full(c_nodes.size, sign / thermal), np.full(c_nodes.size, sign)))

        return csc_matrix(
            (np.concatenate(kept_v), (np.concatenate(kept_r), np.concatenate(kept_c))),
            shape=(d.state_size, d.state_size),
        )
