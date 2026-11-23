"""Coordinate maps and exponential-fitted Scharfetter-Gummel face fluxes.

Provides:
1. :class:`PositiveSpeciesMap` -- overflow-free dimensionless Fischer-Burmeister
   coordinate map ``u -> c(u) > 0`` with exact subtraction-free tangent and BDF
   storage secant, resolving the singular-mass breakdown when chemical species
   invade initially empty (``c = 0``) transport/blocking layers.
2. :class:`BoundedSpeciesMap` -- smooth finite-site logistic map ``c in (0, c_lim)``
   for optional steric exclusion on mobile iodide ``I^-``.
3. :func:`species_sg_face_flux` -- unified Scharfetter-Gummel face flux for
   charged (``z = -1``) and neutral (``z = 0``) species with harmonic face
   diffusivity and optional heterointerface partition ratios ``K_part``.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math

import numpy as np
from numpy.typing import ArrayLike

from solarlab.contracts import ContractError, Vector, frozen_array
from solarlab.discretization.bernoulli import bernoulli, bernoulli_derivative
from solarlab.physics.ions import ion_face_flux, ion_face_flux_jacobian

__all__ = [
    "BoundedSpeciesMap",
    "IODINE_TRANSPORT_SCHEMA",
    "PositiveSpeciesMap",
    "SpeciesCoordinateMap",
    "SpeciesFaceFluxEvaluation",
    "harmonic_face_diffusivity_m2_s",
    "species_sg_face_flux",
]

IODINE_TRANSPORT_SCHEMA = "solarlab.iodine_transport.v1"


def _positive_vector(values: ArrayLike, size: int, name: str) -> Vector:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 0:
        arr = np.full(size, float(arr), dtype=np.float64)
    if arr.shape != (size,) or not np.all(np.isfinite(arr)) or np.any(arr <= 0.0):
        raise ContractError(f"{name}_must_be_strictly_positive_vector_of_length_{size}")
    return frozen_array(arr)


def _psi_and_radius(u: np.ndarray, eta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    radius = np.hypot(u, 2.0 * eta)
    two_eta_sq = 2.0 * (eta * eta)
    nonneg = u >= 0.0
    safe_denom = np.where(nonneg, 1.0, radius - u)
    psi = np.where(nonneg, 0.5 * (u + radius), two_eta_sq / safe_denom)
    return psi, radius


@dataclass(frozen=True, slots=True)
class PositiveSpeciesMap:
    """Dimensionless Fischer-Burmeister positive map ``c(u) = scale * psi(u) > 0``."""

    scale_m3: Vector
    eta: Vector
    vacuum_floor_m3: float = 1.0

    def __post_init__(self) -> None:
        raw = np.atleast_1d(np.asarray(self.scale_m3, dtype=np.float64))
        if raw.ndim != 1 or raw.size < 1:
            raise ContractError("scale_m3_must_be_nonempty_1d_vector")
        size = int(raw.size)
        scale = _positive_vector(self.scale_m3, size, "scale_m3")
        eta = _positive_vector(self.eta, size, "eta")
        if np.any(eta > 1.0):
            raise ContractError("eta_must_not_exceed_unity")
        floor = float(self.vacuum_floor_m3)
        if not math.isfinite(floor) or floor <= 0.0:
            raise ContractError("vacuum_floor_m3_must_be_strictly_positive")
        object.__setattr__(self, "scale_m3", scale)
        object.__setattr__(self, "eta", eta)
        object.__setattr__(self, "vacuum_floor_m3", floor)

    @property
    def size(self) -> int:
        return int(self.scale_m3.size)

    @property
    def identity(self) -> str:
        digest = sha256(
            json.dumps(
                [IODINE_TRANSPORT_SCHEMA, "positive_fb_v1", self.vacuum_floor_m3],
                separators=(",", ":"),
            ).encode()
        )
        digest.update(np.ascontiguousarray(self.scale_m3, dtype=np.float64).tobytes())
        digest.update(np.ascontiguousarray(self.eta, dtype=np.float64).tobytes())
        return digest.hexdigest()

    def density_and_tangent_m3(self, u: ArrayLike) -> tuple[Vector, Vector]:
        coord = np.asarray(u, dtype=np.float64)
        if coord.shape != (self.size,) or not np.all(np.isfinite(coord)):
            raise ContractError("coordinate_must_be_finite_vector")
        psi, radius = _psi_and_radius(coord, self.eta)
        density = self.scale_m3 * psi
        tangent = self.scale_m3 * (psi / radius)
        if not (np.all(np.isfinite(density)) and np.all(np.isfinite(tangent))):
            raise ContractError("nonfinite_positive_species_map_evaluation")
        return frozen_array(density), frozen_array(tangent)

    def density_m3(self, u: ArrayLike) -> Vector:
        density, _ = self.density_and_tangent_m3(u)
        return density

    def coordinate_from_density(self, density_m3: ArrayLike) -> Vector:
        """Exact inverse ``u(c)``; maps ``c == 0`` cleanly to ``vacuum_floor_m3``."""
        raw = np.asarray(density_m3, dtype=np.float64)
        if raw.ndim == 0:
            raw = np.full(self.size, float(raw), dtype=np.float64)
        if raw.shape != (self.size,) or not np.all(np.isfinite(raw)) or np.any(raw < 0.0):
            raise ContractError("density_m3_must_be_finite_nonnegative_vector")
        eff = np.maximum(raw, self.vacuum_floor_m3)
        psi = eff / self.scale_m3
        coord = psi - (self.eta * self.eta) / psi
        return frozen_array(coord)

    def secant_increment_m3(self, u_new: ArrayLike, u_old: ArrayLike) -> Vector:
        """Cancellation-free ``c(u_new) - c(u_old)`` valid for all ``u in R``."""
        u2 = np.asarray(u_new, dtype=np.float64)
        u1 = np.asarray(u_old, dtype=np.float64)
        if (
            u2.shape != (self.size,)
            or u1.shape != (self.size,)
            or not (np.all(np.isfinite(u2)) and np.all(np.isfinite(u1)))
        ):
            raise ContractError("secant_coordinates_must_be_finite_vectors")
        psi1, r1 = _psi_and_radius(u1, self.eta)
        psi2, r2 = _psi_and_radius(u2, self.eta)
        delta = self.scale_m3 * (u2 - u1) * ((psi1 + psi2) / (r1 + r2))
        return frozen_array(delta)


@dataclass(frozen=True, slots=True)
class BoundedSpeciesMap:
    """Smooth logistic map ``c(u) = c_lim / (1 + exp(-(u + u_ref)))``."""

    site_limit_m3: Vector
    logit_reference: Vector
    vacuum_floor_m3: float = 1.0

    def __post_init__(self) -> None:
        raw = np.atleast_1d(np.asarray(self.site_limit_m3, dtype=np.float64))
        if raw.ndim != 1 or raw.size < 1:
            raise ContractError("site_limit_m3_must_be_nonempty_1d_vector")
        size = int(raw.size)
        limit = _positive_vector(self.site_limit_m3, size, "site_limit_m3")
        ref = np.asarray(self.logit_reference, dtype=np.float64)
        if ref.ndim == 0:
            ref = np.full(size, float(ref), dtype=np.float64)
        if ref.shape != (size,) or not np.all(np.isfinite(ref)):
            raise ContractError("logit_reference_must_be_finite_vector")
        floor = float(self.vacuum_floor_m3)
        if not math.isfinite(floor) or floor <= 0.0 or np.any(floor >= limit):
            raise ContractError("vacuum_floor_m3_must_be_below_site_limit")
        object.__setattr__(self, "site_limit_m3", limit)
        object.__setattr__(self, "logit_reference", frozen_array(ref))
        object.__setattr__(self, "vacuum_floor_m3", floor)

    @property
    def size(self) -> int:
        return int(self.site_limit_m3.size)

    @property
    def identity(self) -> str:
        digest = sha256(
            json.dumps(
                [IODINE_TRANSPORT_SCHEMA, "bounded_logistic_v1", self.vacuum_floor_m3],
                separators=(",", ":"),
            ).encode()
        )
        digest.update(
            np.ascontiguousarray(self.site_limit_m3, dtype=np.float64).tobytes()
        )
        digest.update(
            np.ascontiguousarray(self.logit_reference, dtype=np.float64).tobytes()
        )
        return digest.hexdigest()

    @staticmethod
    def _sigmoid(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pos = z >= 0.0
        exp_neg = np.exp(np.where(pos, -z, z))
        denom = 1.0 + exp_neg
        theta = np.where(pos, 1.0 / denom, exp_neg / denom)
        one_minus = np.where(pos, exp_neg / denom, 1.0 / denom)
        return theta, one_minus

    def density_and_tangent_m3(self, u: ArrayLike) -> tuple[Vector, Vector]:
        coord = np.asarray(u, dtype=np.float64)
        if coord.shape != (self.size,) or not np.all(np.isfinite(coord)):
            raise ContractError("coordinate_must_be_finite_vector")
        theta, one_minus = self._sigmoid(coord + self.logit_reference)
        density = self.site_limit_m3 * theta
        tangent = density * one_minus
        return frozen_array(density), frozen_array(tangent)

    def density_m3(self, u: ArrayLike) -> Vector:
        density, _ = self.density_and_tangent_m3(u)
        return density

    def coordinate_from_density(self, density_m3: ArrayLike) -> Vector:
        raw = np.asarray(density_m3, dtype=np.float64)
        if raw.ndim == 0:
            raw = np.full(self.size, float(raw), dtype=np.float64)
        if raw.shape != (self.size,) or not np.all(np.isfinite(raw)) or np.any(raw < 0.0):
            raise ContractError("density_m3_must_be_finite_nonnegative_vector")
        if np.any(raw >= self.site_limit_m3):
            raise ContractError("density_exceeds_finite_site_limit")
        eff = np.maximum(raw, self.vacuum_floor_m3)
        theta = eff / self.site_limit_m3
        logit = np.log(theta) - np.log1p(-theta)
        return frozen_array(logit - self.logit_reference)

    def secant_increment_m3(self, u_new: ArrayLike, u_old: ArrayLike) -> Vector:
        """Cancellation-free logistic difference ``c(u_new) - c(u_old)``."""
        u2 = np.asarray(u_new, dtype=np.float64)
        u1 = np.asarray(u_old, dtype=np.float64)
        if (
            u2.shape != (self.size,)
            or u1.shape != (self.size,)
            or not (np.all(np.isfinite(u2)) and np.all(np.isfinite(u1)))
        ):
            raise ContractError("secant_coordinates_must_be_finite_vectors")
        du = u2 - u1
        half = 0.5 * du
        # sigmoid(z2) - sigmoid(z1) = 2 * sinh(du/2) / [2 cosh(du/2) + exp(-(z1+z2)/2)...]
        # Equivalently: theta1 * (1 - theta2) * expm1(du) when du <= 0, and
        #               theta2 * (1 - theta1) * (-expm1(-du)) when du > 0!
        theta1, om1 = self._sigmoid(u1 + self.logit_reference)
        theta2, om2 = self._sigmoid(u2 + self.logit_reference)
        abs_du = np.abs(du)
        # Avoid expm1 overflow for huge |du| by using bounded sinh/cosh identity
        # when |du| > 40, and exact product * expm1(-|du|) for all |du|!
        # Notice expm1(-|du|) is STRICTLY in (-1, 0] for ALL |du| >= 0 -- NEVER OVERFLOWS!
        neg_expm1 = -np.expm1(-abs_du)  # = 1 - exp(-|du|) in [0, 1)
        # Identity check:
        # theta2 - theta1 = sign(du) * theta_hi * (1 - theta_lo) * (1 - exp(-|du|))?
        # Wait: let z2 = z1 + a (a >= 0).
        # theta2 * (1 - theta1) * (1 - e^-a)
        #   = [e^z2 / (1 + e^z2)] * [1 / (1 + e^z1)] * (1 - e^-(z2 - z1))
        #   = (e^z2 - e^z1) / [(1 + e^z2)(1 + e^z1)] == theta2 - theta1 IDENTICALLY!!
        pos_step = half >= 0.0
        factor = np.where(pos_step, theta2 * om1, -theta1 * om2)
        return frozen_array(self.site_limit_m3 * factor * neg_expm1)


SpeciesCoordinateMap = PositiveSpeciesMap | BoundedSpeciesMap


def harmonic_face_diffusivity_m2_s(
    left_diffusivity_m2_s: ArrayLike,
    right_diffusivity_m2_s: ArrayLike,
) -> Vector:
    """Harmonic average ``2 D_L D_R / (D_L + D_R)`` returning exact ``0`` at blocking faces."""
    dl = np.asarray(left_diffusivity_m2_s, dtype=np.float64)
    dr = np.asarray(right_diffusivity_m2_s, dtype=np.float64)
    if dl.shape != dr.shape or dl.ndim != 1:
        raise ContractError("face_diffusivity_inputs_must_match_1d_shape")
    if not (np.all(np.isfinite(dl)) and np.all(np.isfinite(dr))):
        raise ContractError("face_diffusivity_inputs_must_be_finite")
    if np.any(dl < 0.0) or np.any(dr < 0.0):
        raise ContractError("face_diffusivity_inputs_must_be_nonnegative")
    denom = dl + dr
    positive = (dl > 0.0) & (dr > 0.0) & (denom > 0.0)
    safe_denom = np.where(positive, denom, 1.0)
    harmonic = np.where(positive, (2.0 * dl * dr) / safe_denom, 0.0)
    return frozen_array(harmonic)


@dataclass(frozen=True, slots=True)
class SpeciesFaceFluxEvaluation:
    """Particle face flux ``[1 / (m^2 s)]`` and exact node-density/potential tangents."""

    flux_m2_s: Vector
    dflux_dc_left: Vector | None = None
    dflux_dc_right: Vector | None = None
    dflux_dphi_left: Vector | None = None
    dflux_dphi_right: Vector | None = None


def species_sg_face_flux(
    phi_V: ArrayLike,
    density_m3: ArrayLike,
    dx_m: ArrayLike,
    face_diffusivity_m2_s: ArrayLike,
    thermal_voltage_V: float,
    *,
    charge_number: int = 0,
    log_partition_step: ArrayLike | None = None,
    steric_site_limit_m3: ArrayLike | None = None,
    steric_diffusion_only: bool = True,
    derivatives: bool = True,
) -> SpeciesFaceFluxEvaluation:
    """Evaluate 1D Scharfetter-Gummel face flux for charged or neutral species.

    With ``xi_f = z * (phi_R - phi_L) / V_T - ln(K_part_R / K_part_L)``:
    ``Gamma_f = (D_f / dx_f) * [B(xi_f) * c_L - B(-xi_f) * c_R]``.
    """
    phi = np.asarray(phi_V, dtype=np.float64)
    c = np.asarray(density_m3, dtype=np.float64)
    dx = np.asarray(dx_m, dtype=np.float64)
    diff = np.asarray(face_diffusivity_m2_s, dtype=np.float64)
    vt = float(thermal_voltage_V)
    if phi.ndim != 1 or phi.size < 2 or c.shape != phi.shape:
        raise ContractError("species_flux_requires_matching_1d_node_vectors")
    faces = phi.size - 1
    if dx.shape != (faces,) or diff.shape != (faces,):
        raise ContractError("species_flux_requires_matching_face_vectors")
    if not (
        np.all(np.isfinite(phi))
        and np.all(np.isfinite(c))
        and np.all(np.isfinite(dx))
        and np.all(np.isfinite(diff))
        and math.isfinite(vt)
    ):
        raise ContractError("species_flux_inputs_must_be_finite")
    if vt <= 0.0 or np.any(dx <= 0.0) or np.any(diff < 0.0) or np.any(c < 0.0):
        raise ContractError("species_flux_invalid_physical_domain")

    log_part = (
        np.zeros(faces, dtype=np.float64)
        if log_partition_step is None
        else np.asarray(log_partition_step, dtype=np.float64)
    )
    if log_part.shape != (faces,) or not np.all(np.isfinite(log_part)):
        raise ContractError("log_partition_step_must_be_finite_face_vector")

    if steric_site_limit_m3 is not None:
        if charge_number not in (-1, 1):
            raise ContractError("steric_ion_kernel_requires_unit_charge_species")
        if np.any(log_part != 0.0):
            raise ContractError("steric_ion_kernel_does_not_combine_partition_step")
        limit = np.asarray(steric_site_limit_m3, dtype=np.float64)
        if limit.shape != phi.shape:
            raise ContractError("steric_site_limit_must_match_node_count")
        sign = float(charge_number)
        flux = ion_face_flux(
            phi,
            c,
            dx,
            diff,
            vt,
            limit[:-1],
            steric_diffusion_only=steric_diffusion_only,
            P_lim_node=limit,
            drift_sign=sign,
        )
        if not derivatives:
            return SpeciesFaceFluxEvaluation(flux_m2_s=frozen_array(flux))
        jac = ion_face_flux_jacobian(
            phi,
            c,
            dx,
            diff,
            vt,
            limit[:-1],
            steric_diffusion_only=steric_diffusion_only,
            P_lim_node=limit,
            drift_sign=sign,
        )
        if not np.all(jac.differentiable_faces):
            raise ContractError("steric_ion_flux_hit_nondifferentiable_kink")
        return SpeciesFaceFluxEvaluation(
            flux_m2_s=frozen_array(jac.flux),
            dflux_dc_left=frozen_array(jac.density_left_derivative),
            dflux_dc_right=frozen_array(jac.density_right_derivative),
            dflux_dphi_left=frozen_array(jac.potential_left_derivative),
            dflux_dphi_right=frozen_array(jac.potential_right_derivative),
        )

    conductance = diff / dx
    if charge_number == 0 and log_partition_step is None:
        flux = conductance * (c[:-1] - c[1:])
        if not derivatives:
            return SpeciesFaceFluxEvaluation(flux_m2_s=frozen_array(flux))
        zeros = np.zeros(faces, dtype=np.float64)
        return SpeciesFaceFluxEvaluation(
            flux_m2_s=frozen_array(flux),
            dflux_dc_left=frozen_array(conductance),
            dflux_dc_right=frozen_array(-conductance),
            dflux_dphi_left=frozen_array(zeros),
            dflux_dphi_right=frozen_array(zeros),
        )

    z_float = float(charge_number)
    xi = (z_float * np.diff(phi) / vt) - log_part
    bp = bernoulli(xi)
    bm = bernoulli(-xi)
    flux = conductance * (bp * c[:-1] - bm * c[1:])
    if not derivatives:
        return SpeciesFaceFluxEvaluation(flux_m2_s=frozen_array(flux))

    dc_left = conductance * bp
    dc_right = -conductance * bm
    if charge_number == 0:
        zeros = np.zeros(faces, dtype=np.float64)
        return SpeciesFaceFluxEvaluation(
            flux_m2_s=frozen_array(flux),
            dflux_dc_left=frozen_array(dc_left),
            dflux_dc_right=frozen_array(dc_right),
            dflux_dphi_left=frozen_array(zeros),
            dflux_dphi_right=frozen_array(zeros),
        )

    dbp = bernoulli_derivative(xi)
    dbm = bernoulli_derivative(-xi)
    dflux_dxi = conductance * (dbp * c[:-1] + dbm * c[1:])
    scale_phi = z_float / vt
    dphi_left = -scale_phi * dflux_dxi
    dphi_right = scale_phi * dflux_dxi
    return SpeciesFaceFluxEvaluation(
        flux_m2_s=frozen_array(flux),
        dflux_dc_left=frozen_array(dc_left),
        dflux_dc_right=frozen_array(dc_right),
        dflux_dphi_left=frozen_array(dphi_left),
        dflux_dphi_right=frozen_array(dphi_right),
    )
