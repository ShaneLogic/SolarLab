"""M-class route-independent migration of the historical ionic kernels.

The imported Bernoulli/divergence functions are the existing core M kernels.
All flux formulas and inherited branches below are preserved. The historical
names distinguish empirical whole-flux crowding from the implemented nodal
crowding-in-potential form; neither name grants scientific qualification.
"""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from solarlab.discretization.bernoulli import (
    bernoulli,
    bernoulli_derivative,
)


__all__ = [
    "WHOLE_FLUX_STERIC_V1", "DIFFUSION_ONLY_STERIC_V1",
    "HISTORICAL_OCCUPANCY_CLIP", "HISTORICAL_STERIC_FLOOR",
    "IonFaceFluxJacobian", "ion_face_flux", "ion_face_flux_jacobian",
    "ion_flux_steric", "ion_continuity_rhs", "ion_continuity_rhs_neg",
]

WHOLE_FLUX_STERIC_V1 = "historical_whole_flux_steric_v1"
DIFFUSION_ONLY_STERIC_V1 = "historical_diffusion_only_steric_v1"
# Descriptions of the retained literals, not additional solver controls.
HISTORICAL_OCCUPANCY_CLIP = (0.0, 0.999999)
HISTORICAL_STERIC_FLOOR = 1.0e-6



def _steric_diffusion_only_flux(
    P: np.ndarray, phi: np.ndarray, dx: np.ndarray,
    D_I_face: np.ndarray | float, V_T: float, P_lim: np.ndarray | float,
    P_lim_node: np.ndarray | float | None, P_other_node: np.ndarray | None,
    drift_sign: float,
) -> np.ndarray:
    """Face flux for the diffusion-only steric ion transport (review F05).

    Folds the lattice-gas crowding chemical potential
    ``mu_ex/kT = -ln(1 - theta)`` into the Scharfetter-Gummel drift argument
    (bare ``D_I``), so the excluded-volume correction acts on diffusion only
    while the Bernoulli exponential fitting is preserved. ``drift_sign`` is
    ``+1`` for a positive species and ``-1`` for a negative one.

    Site occupancy ``theta``:

    * single species, or distinct-sublattice dual-ion (``P_other_node`` None):
      ``theta = P / P_lim`` — each species crowds only against itself;
    * shared-site dual-ion (``P_other_node`` given): the crowding potential
      uses the TOTAL occupancy ``theta = (P + P_other) / P_lim``, so the two
      species compete for the same reservoir — the physically standard
      multi-species finite-size PNP coupling. Reduces exactly to the
      single-species form when the other density is zero.
    """
    Plim_n = P_lim_node if P_lim_node is not None else P_lim
    Plim_n = np.broadcast_to(np.asarray(Plim_n, dtype=float), P.shape)
    P_tot = P if P_other_node is None else P + np.asarray(P_other_node, dtype=float)
    c = np.clip(P_tot / Plim_n, 0.0, 0.999999)          # site occupancy theta
    mu_ex = -np.log1p(-c)                                # -ln(1 - theta), per node
    xi = drift_sign * (phi[1:] - phi[:-1]) / V_T + (mu_ex[1:] - mu_ex[:-1])
    return D_I_face / dx * (bernoulli(xi) * P[:-1] - bernoulli(-xi) * P[1:])


def ion_face_flux(
    phi: np.ndarray,
    P: np.ndarray,
    dx: np.ndarray,
    D_I_face: np.ndarray,
    V_T: float,
    P_lim: np.ndarray | float,
    *,
    steric_diffusion_only: bool = False,
    P_lim_node: np.ndarray | float | None = None,
    P_other_node: np.ndarray | None = None,
    drift_sign: float = 1.0,
) -> np.ndarray:
    """Historical particle flux [m^-2 s^-1] for either unit-charge sign.

    M migration of both implemented steric forms. The default whole-flux
    multiplier uses face-averaged own density; the diffusion-only flag uses
    the nodal crowding potential and optional shared-site partner density.
    Clips, denominator floors, capacity fallbacks and arithmetic order are
    retained exactly. This kernel supplies no device/time qualification.
    """
    if steric_diffusion_only:
        return _steric_diffusion_only_flux(
            P, phi, dx, D_I_face, V_T, P_lim, P_lim_node, P_other_node,
            drift_sign=drift_sign,
        )
    P_lim_face = np.broadcast_to(np.asarray(P_lim, dtype=float), dx.shape)
    P_avg = 0.5 * (P[:-1] + P[1:])
    steric = 1.0 / np.maximum(
        1.0 - np.clip(P_avg / P_lim_face, 0.0, 0.999999), 1e-6)
    xi = drift_sign * (phi[1:] - phi[:-1]) / V_T
    D_eff = D_I_face * steric
    return D_eff / dx * (bernoulli(xi) * P[:-1] - bernoulli(-xi) * P[1:])


@dataclass(frozen=True, slots=True)
class IonFaceFluxJacobian:
    """Local derivatives of one ionic particle flux on every face."""

    flux: np.ndarray
    density_left_derivative: np.ndarray
    density_right_derivative: np.ndarray
    partner_left_derivative: np.ndarray
    partner_right_derivative: np.ndarray
    potential_left_derivative: np.ndarray
    potential_right_derivative: np.ndarray
    differentiable_faces: np.ndarray


def ion_face_flux_jacobian(
    phi: np.ndarray,
    P: np.ndarray,
    dx: np.ndarray,
    D_I_face: np.ndarray,
    V_T: float,
    P_lim: np.ndarray | float,
    *,
    steric_diffusion_only: bool = False,
    P_lim_node: np.ndarray | float | None = None,
    P_other_node: np.ndarray | None = None,
    drift_sign: float = 1.0,
) -> IonFaceFluxJacobian:
    """M migration of the already-implemented legacy local derivatives.

    Values and differentiable-face masks follow the original clipped law,
    including its existing numerical limits and refusal semantics. No new
    derivative, formula correction or tangent qualification is introduced.
    """
    potential = np.asarray(phi, dtype=float)
    density = np.asarray(P, dtype=float)
    spacing = np.asarray(dx, dtype=float)
    if (
        potential.ndim != 1
        or potential.size < 2
        or density.shape != potential.shape
        or spacing.shape != (potential.size - 1,)
        or not np.all(np.isfinite(potential))
        or not np.all(np.isfinite(density))
        or not np.all(np.isfinite(spacing))
        or np.any(spacing <= 0.0)
        or not np.isfinite(V_T)
        or V_T <= 0.0
        or drift_sign not in (-1.0, 1.0)
    ):
        raise ValueError(
            "ion-flux Jacobian inputs must be finite, shape matched, and "
            "use positive spacing, thermal voltage, and unit charge sign"
        )
    diffusion = np.broadcast_to(
        np.asarray(D_I_face, dtype=float), spacing.shape
    )
    if not np.all(np.isfinite(diffusion)) or np.any(diffusion < 0.0):
        raise ValueError("ionic diffusivity must be finite and non-negative")
    partner = None
    if P_other_node is not None:
        partner = np.asarray(P_other_node, dtype=float)
        if partner.shape != density.shape or not np.all(np.isfinite(partner)):
            raise ValueError("partner ion density must be finite and shape matched")

    flux = ion_face_flux(
        potential,
        density,
        spacing,
        diffusion,
        V_T,
        P_lim,
        steric_diffusion_only=steric_diffusion_only,
        P_lim_node=P_lim_node,
        P_other_node=partner,
        drift_sign=drift_sign,
    )
    prefactor = diffusion / spacing
    zero = np.zeros_like(spacing)

    if steric_diffusion_only:
        node_limit_source = P_lim_node if P_lim_node is not None else P_lim
        node_limit = np.broadcast_to(
            np.asarray(node_limit_source, dtype=float), density.shape
        )
        if (
            not np.all(np.isfinite(node_limit))
            or np.any(node_limit <= 0.0)
        ):
            raise ValueError("node site limits must be finite and positive")
        total_density = density if partner is None else density + partner
        occupancy = total_density / node_limit
        clipped = np.clip(occupancy, 0.0, 0.999999)
        chemical_potential = -np.log1p(-clipped)
        xi = (
            drift_sign * (potential[1:] - potential[:-1]) / V_T
            + chemical_potential[1:]
            - chemical_potential[:-1]
        )
        interior = (occupancy > 0.0) & (occupancy < 0.999999)
        chemical_derivative = np.divide(
            1.0,
            node_limit * (1.0 - occupancy),
            out=np.zeros_like(density),
            where=interior,
        )
        node_differentiable = (
            (occupancy != 0.0) & (occupancy != 0.999999)
        )
        differentiable = (
            (diffusion == 0.0)
            | (node_differentiable[:-1] & node_differentiable[1:])
        )
        B_pos = bernoulli(xi)
        B_neg = bernoulli(-xi)
        xi_derivative = (
            bernoulli_derivative(xi) * density[:-1]
            + bernoulli_derivative(-xi) * density[1:]
        )
        left_mu_derivative = chemical_derivative[:-1]
        right_mu_derivative = chemical_derivative[1:]
        density_left = prefactor * (
            B_pos - xi_derivative * left_mu_derivative
        )
        density_right = prefactor * (
            -B_neg + xi_derivative * right_mu_derivative
        )
        if partner is None:
            partner_left = zero.copy()
            partner_right = zero.copy()
        else:
            partner_left = -prefactor * xi_derivative * left_mu_derivative
            partner_right = prefactor * xi_derivative * right_mu_derivative
    else:
        face_limit = np.broadcast_to(
            np.asarray(P_lim, dtype=float), spacing.shape
        )
        if (
            not np.all(np.isfinite(face_limit))
            or np.any(face_limit <= 0.0)
        ):
            raise ValueError("face site limits must be finite and positive")
        average_density = 0.5 * (density[:-1] + density[1:])
        occupancy = average_density / face_limit
        clipped = np.clip(occupancy, 0.0, 0.999999)
        denominator = np.maximum(1.0 - clipped, 1.0e-6)
        steric = 1.0 / denominator
        xi = drift_sign * (potential[1:] - potential[:-1]) / V_T
        B_pos = bernoulli(xi)
        B_neg = bernoulli(-xi)
        flux_core = B_pos * density[:-1] - B_neg * density[1:]
        xi_derivative = (
            bernoulli_derivative(xi) * density[:-1]
            + bernoulli_derivative(-xi) * density[1:]
        )
        active_clip = (occupancy > 0.0) & (occupancy < 0.999999)
        active_max = (1.0 - clipped) > 1.0e-6
        steric_density_derivative = np.divide(
            0.5,
            face_limit * denominator**2,
            out=np.zeros_like(spacing),
            where=active_clip & active_max,
        )
        differentiable = (
            (diffusion == 0.0)
            | (
                (occupancy != 0.0)
                & (occupancy != 0.999999)
                & ((1.0 - clipped) != 1.0e-6)
            )
        )
        density_left = prefactor * (
            steric * B_pos + flux_core * steric_density_derivative
        )
        density_right = prefactor * (
            -steric * B_neg + flux_core * steric_density_derivative
        )
        partner_left = zero.copy()
        partner_right = zero.copy()
        xi_derivative = steric * xi_derivative

    potential_scale = prefactor * xi_derivative * drift_sign / V_T
    arrays = (
        flux,
        density_left,
        density_right,
        partner_left,
        partner_right,
        potential_scale,
    )
    if any(not np.all(np.isfinite(array)) for array in arrays):
        raise ValueError("ion-flux Jacobian produced a non-finite derivative")
    return IonFaceFluxJacobian(
        flux=np.asarray(flux, dtype=float),
        density_left_derivative=density_left,
        density_right_derivative=density_right,
        partner_left_derivative=partner_left,
        partner_right_derivative=partner_right,
        potential_left_derivative=-potential_scale,
        potential_right_derivative=potential_scale,
        differentiable_faces=np.asarray(differentiable, dtype=bool),
    )


def ion_flux_steric(
    phi: np.ndarray,   # [phi_i, phi_{i+1}]
    P: np.ndarray,     # [P_i,   P_{i+1}]
    h: float,
    D_I: float,
    V_T: float,
    P_lim: float,
) -> float:
    """
    Sterically corrected positive-vacancy flux F_P [m⁻² s⁻¹].

    The vacancy drift term uses a Scharfetter-Gummel face discretisation so the
    flux has the correct sign for a positively charged mobile ion and remains
    conservative on strongly biased, non-uniform grids.
    """
    P_avg = 0.5 * (P[0] + P[1])
    steric = 1.0 / max(1.0 - np.clip(P_avg / P_lim, 0.0, 0.999999), 1e-6)
    xi = (phi[1] - phi[0]) / V_T
    D_eff = D_I * steric
    return float(D_eff / h * (bernoulli(np.array([xi]))[0] * P[0]
                              - bernoulli(np.array([-xi]))[0] * P[1]))


def ion_continuity_rhs(
    x: np.ndarray,
    phi: np.ndarray,
    P: np.ndarray,
    D_I: np.ndarray | float,
    V_T: float,
    P_lim: np.ndarray | float,
    steric_diffusion_only: bool = False,
    P_lim_node: np.ndarray | float | None = None,
    P_other_node: np.ndarray | None = None,
    *,
    cell_widths_m: np.ndarray | None = None,
) -> np.ndarray:
    """Historical positive-ion rate -div(F), with blocking outer contacts.

    The two original steric forms and all parameters are retained. Without
    cell_widths_m, endpoint widths equal the adjacent full grid spacing;
    explicit widths select the existing control-volume divergence instead.
    This callable rate kernel does not implement a time solver or ion device.
    """
    P = np.asarray(P, dtype=float)
    dx = np.diff(x)                              # (N-1,)
    D_I_face = np.broadcast_to(np.asarray(D_I, dtype=float), dx.shape)
    F_int = ion_face_flux(
        phi, P, dx, D_I_face, V_T, P_lim,
        steric_diffusion_only=steric_diffusion_only,
        P_lim_node=P_lim_node,
        P_other_node=P_other_node,
        drift_sign=+1.0,
    )

    if cell_widths_m is not None:
        from solarlab.discretization.divergence import flux_divergence
        return flux_divergence(F_int, cell_widths_m)

    # Zero-flux BCs: pad with 0 at both ends
    F_full = np.concatenate([[0.0], F_int, [0.0]])   # (N+1,)

    # Dual-grid cell widths
    dx_cell = np.empty(len(x))
    dx_cell[0]    = dx[0]
    dx_cell[-1]   = dx[-1]
    dx_cell[1:-1] = 0.5 * (dx[:-1] + dx[1:])

    return -(F_full[1:] - F_full[:-1]) / dx_cell


def ion_continuity_rhs_neg(
    x: np.ndarray,
    phi: np.ndarray,
    P_neg: np.ndarray,
    D_I: np.ndarray | float,
    V_T: float,
    P_lim: np.ndarray | float,
    steric_diffusion_only: bool = False,
    P_lim_node: np.ndarray | float | None = None,
    P_other_node: np.ndarray | None = None,
    *,
    cell_widths_m: np.ndarray | None = None,
) -> np.ndarray:
    """Historical negative-ion rate, with reversed drift and blocking contacts.

    The diffusion-only flag and optional partner use the same clipped total
    occupancy as the positive species. Capacity and width semantics are
    unchanged; this pure kernel supplies no device/time qualification.
    """
    P_neg = np.asarray(P_neg, dtype=float)
    dx = np.diff(x)
    D_I_face = np.broadcast_to(np.asarray(D_I, dtype=float), dx.shape)
    # Reversed drift: negative charge → drift_sign = -1.
    F_int = ion_face_flux(
        phi, P_neg, dx, D_I_face, V_T, P_lim,
        steric_diffusion_only=steric_diffusion_only,
        P_lim_node=P_lim_node,
        P_other_node=P_other_node,
        drift_sign=-1.0,
    )

    if cell_widths_m is not None:
        from solarlab.discretization.divergence import flux_divergence
        return flux_divergence(F_int, cell_widths_m)

    F_full = np.concatenate([[0.0], F_int, [0.0]])

    dx_cell = np.empty(len(x))
    dx_cell[0]    = dx[0]
    dx_cell[-1]   = dx[-1]
    dx_cell[1:-1] = 0.5 * (dx[:-1] + dx[1:])

    return -(F_full[1:] - F_full[:-1]) / dx_cell
