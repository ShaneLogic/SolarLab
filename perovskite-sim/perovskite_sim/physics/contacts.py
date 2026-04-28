"""Selective / Schottky outer-contact boundary conditions.

The drift-diffusion solver historically used Dirichlet contacts — the
boundary nodes were pinned to the doping-derived equilibrium densities
``n_L, p_L, n_R, p_R`` and the RHS forced ``dn[0] = dn[-1] = 0``. That is
the ideal ohmic limit: the contact supplies or sinks an unlimited current
to keep the carrier at equilibrium. Real contacts have a finite surface
recombination velocity ``S`` and may be selective (majority conducting,
minority blocking) or Schottky (a rectifying barrier that limits the
majority current).

This module provides the Robin-type flux that replaces the Dirichlet pin
when a contact opts in. For a selective contact with surface
recombination velocity ``S`` on carrier ``n`` at the left boundary:

    J_contact = q · S · (n - n_eq)                        [A/m², +x sign convention]

which, when used as the pad value in ``carrier_continuity_rhs`` in place
of the existing zero-flux pad, gives the mass balance at the boundary
cell:

    dn[0]/dt ∝ +J_face_interior − J_contact              (node 0, left)

At the right boundary the sign flips because "into the contact" is the
+x direction:

    J_contact = − q · S · (n − n_eq)                       (right)

Limit checks:

* ``S → ∞``  → ``n[0] → n_eq`` exponentially fast, recovering the
  Dirichlet pin. Any S larger than ~1e7 m/s is already in this regime
  on typical 1e-8 m grid spacings — the relaxation time ``dx/S`` is
  sub-picosecond, far faster than any external time scale.
* ``S = 0``  → zero flux, i.e. a perfectly blocking contact (useful for
  modelling electron-blocking HTLs that are thinner than the diffusion
  length). This is the Neumann limit.

The holes obey the same formula with the charge sign:

    J_p_contact(left)  = − q · S_p · (p − p_eq)            (left, hole)
    J_p_contact(right) = + q · S_p · (p − p_eq)            (right, hole)

The opposite sign for holes follows from ``dp`` carrying a leading
``−`` relative to ``∇·J_p`` in the continuity equation — the same
bookkeeping that already exists in the interior SG fluxes. See the
derivation in the docstring of :func:`selective_contact_flux`.

Schottky contacts differ only in that ``n_eq`` is set by the barrier
height ``φ_B`` via the thermionic relation ``n_eq = N_c · exp(-φ_B/V_T)``
rather than by the layer doping. We expose ``schottky_equilibrium_n``
for callers that want this override, but the BC machinery is shared —
only ``n_eq`` changes.
"""
from __future__ import annotations

import numpy as np

from perovskite_sim.constants import Q


def selective_contact_flux(
    density: float | np.ndarray,
    density_eq: float | np.ndarray,
    S: float,
    *,
    carrier: str,
    side: str,
) -> float | np.ndarray:
    """Robin-type boundary current density for a selective outer contact.

    Parameters
    ----------
    density
        Carrier density at the boundary node [m⁻³]. Can be a scalar or array.
    density_eq
        Equilibrium carrier density at the contact [m⁻³]. For an ohmic
        doping-derived contact this is the ``_equilibrium_np`` result;
        for a Schottky contact it is ``N_c · exp(-φ_B/V_T)``. Can be a
        scalar or array matching the shape of ``density``.
    S
        Surface recombination velocity [m/s]. ``S = 0`` gives a
        perfectly blocking contact (zero flux, Neumann BC). Any finite
        ``S`` interpolates between blocking and ohmic.
    carrier
        ``"n"`` for electrons, ``"p"`` for holes. The carrier type
        determines the sign of the flux-to-charge-density relation.
    side
        ``"left"`` for the x=0 contact, ``"right"`` for the x=L contact.

    Returns
    -------
    J_contact : float | ndarray
        Current density at the boundary face [A/m²], signed in the
        global +x convention, ready to drop into the SG flux pad.

    Notes
    -----
    Sign conventions are derived by requiring that the boundary node
    relaxes toward ``density_eq`` when the interior is at equilibrium:

    * ``dn[0]/dt ∝ +(J_face_interior − J_contact)`` in the electron
      continuity equation → ``J_contact_n_L = +q·S·(n − n_eq)`` so a
      positive excess pulls the node down.
    * ``dp[0]/dt ∝ −(J_face_interior − J_contact)`` for holes (leading
      minus in ``dp`` since ``∂p/∂t = −∇·J_p/q``) → the sign is
      reversed: ``J_contact_p_L = −q·S·(p − p_eq)``.
    * At the right contact "into the contact" flips direction, so each
      formula picks up an additional minus sign.

    This asymmetry is intentional and must not be "tidied" — the
    ``dn`` and ``dp`` equations already carry the carrier-sign difference,
    and flipping either of them in one place only would break the
    ohmic limit.
    """
    if side not in ("left", "right"):
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")
    if carrier not in ("n", "p"):
        raise ValueError(f"carrier must be 'n' or 'p', got {carrier!r}")

    excess = np.asarray(density, dtype=float) - np.asarray(density_eq, dtype=float)
    mag = Q * float(S) * excess

    # Base sign (left contact): + for electrons, - for holes.
    if carrier == "n":
        J = mag
    else:
        J = -mag

    # Right contact flips the sign relative to left.
    if side == "right":
        J = -J

    return float(J) if np.isscalar(density) else J


def apply_selective_contacts(
    J_n_full: np.ndarray,
    J_p_full: np.ndarray,
    n: np.ndarray,
    p: np.ndarray,
    *,
    S_n_L: float,
    S_p_L: float,
    S_n_R: float,
    S_p_R: float,
    n_L: float,
    p_L: float,
    n_R: float,
    p_R: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Overwrite the boundary pad in the padded SG flux arrays.

    ``carrier_continuity_rhs`` pads the interior SG fluxes with a zero
    at each boundary (the Dirichlet / no-flux pad). This helper replaces
    those two zeros with the Robin flux on each side, for each carrier,
    returning new arrays. The input arrays are not mutated.

    Parameters
    ----------
    J_n_full, J_p_full
        Padded SG flux arrays of length ``N`` (one per node boundary).
        Index 0 is the left-contact face; index -1 is the right-contact
        face.
    n, p
        Full-grid carrier density arrays; only indices 0 and -1 are
        read.
    S_n_L, S_p_L, S_n_R, S_p_R
        Per-carrier, per-side surface recombination velocities [m/s].
    n_L, p_L, n_R, p_R
        Equilibrium contact densities [m⁻³].
    """
    J_n_out = np.array(J_n_full, copy=True)
    J_p_out = np.array(J_p_full, copy=True)

    J_n_out[0] = selective_contact_flux(
        float(n[0]), n_L, S_n_L, carrier="n", side="left",
    )
    J_p_out[0] = selective_contact_flux(
        float(p[0]), p_L, S_p_L, carrier="p", side="left",
    )
    J_n_out[-1] = selective_contact_flux(
        float(n[-1]), n_R, S_n_R, carrier="n", side="right",
    )
    J_p_out[-1] = selective_contact_flux(
        float(p[-1]), p_R, S_p_R, carrier="p", side="right",
    )
    return J_n_out, J_p_out


def schottky_equilibrium_n(N_c: float, phi_B: float, V_T: float) -> float:
    """Equilibrium electron density behind a Schottky barrier.

    ``n_eq = N_c · exp(-φ_B / V_T)``. Useful when the contact is
    metal/semiconductor rather than a highly-doped selective layer;
    the caller substitutes this value for ``n_L`` or ``n_R`` in
    :func:`apply_selective_contacts`.
    """
    return float(N_c) * float(np.exp(-float(phi_B) / float(V_T)))


def schottky_equilibrium_p(N_v: float, phi_B: float, V_T: float) -> float:
    """Equilibrium hole density behind a Schottky barrier. Mirror of
    :func:`schottky_equilibrium_n` for the valence band."""
    return float(N_v) * float(np.exp(-float(phi_B) / float(V_T)))


__all__ = [
    "selective_contact_flux",
    "apply_selective_contacts",
    "schottky_equilibrium_n",
    "schottky_equilibrium_p",
]
