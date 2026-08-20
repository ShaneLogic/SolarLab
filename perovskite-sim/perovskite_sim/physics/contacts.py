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

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

from perovskite_sim.constants import Q

if TYPE_CHECKING:
    from perovskite_sim.models.device import DeviceStack
    from perovskite_sim.solver.mol import MaterialArrays


CONTACT_THERMODYNAMIC_TOLERANCE_EV = 5.0e-3


@dataclass(frozen=True)
class ContactThermodynamicCertificate:
    """Internal compatibility evidence for Poisson and carrier reservoirs."""

    status: Literal[
        "certified",
        "inconsistent",
        "compatible_unverified",
        "not_assessable",
    ]
    built_in_potential_mode: str
    tolerance_eV: float
    fermi_level_span_eV: float | None
    potential_mismatch_V: float | None
    metal_work_function_mismatch_eV: float | None
    contact_quasi_fermi_levels_eV: tuple[float, ...]
    message: str

    @property
    def certified(self) -> bool:
        return self.status == "certified"


class ContactThermodynamicError(ValueError):
    """The selected Poisson/contact-reservoir pair lacks a certificate."""


def assess_contact_thermodynamics(
    stack: "DeviceStack",
    mat: "MaterialArrays",
    *,
    tolerance_eV: float = CONTACT_THERMODYNAMIC_TOLERANCE_EV,
) -> ContactThermodynamicCertificate:
    """Assess whether all four Maxwell-Boltzmann contact QFLs agree.

    A zero-bias equilibrium requires the electron and hole reservoir on both
    contacts to share one Fermi level. This check uses the exact densities and
    Poisson drop consumed by the solver. When endpoint DOS data are absent, a
    mismatch against the repository's band-derived contact potential can
    still disprove compatibility, but equality is labelled unverified rather
    than promoted to a thermodynamic certificate.
    """
    if not np.isfinite(tolerance_eV) or tolerance_eV <= 0.0:
        raise ValueError("tolerance_eV must be finite and positive")
    if tolerance_eV > CONTACT_THERMODYNAMIC_TOLERANCE_EV:
        raise ValueError(
            "tolerance_eV cannot exceed the fixed internal certificate gate "
            f"of {CONTACT_THERMODYNAMIC_TOLERANCE_EV:g} eV"
        )

    mode = stack.resolved_built_in_potential_mode()
    potential_mismatch: float | None
    try:
        potential_mismatch = float(
            mat.V_bi_bc - stack.compute_semiconductor_V_bi()
        )
        if not np.isfinite(potential_mismatch):
            potential_mismatch = None
    except (AttributeError, TypeError, ValueError, OverflowError):
        potential_mismatch = None

    levels: list[float] = []
    semiconductor_work_functions: list[float] = []
    nc_values = getattr(mat, "N_C_physical", None)
    nv_values = getattr(mat, "N_V_physical", None)
    chi_values = getattr(mat, "chi_phys", None)
    eg_values = getattr(mat, "Eg_phys", None)
    if all(value is not None for value in (
        nc_values, nv_values, chi_values, eg_values,
    )):
        nc = np.asarray(nc_values, dtype=float)
        nv = np.asarray(nv_values, dtype=float)
        chi = np.asarray(chi_values, dtype=float)
        eg = np.asarray(eg_values, dtype=float)
        reservoirs = (
            (0, 0.0, float(mat.n_L), float(mat.p_L)),
            (-1, float(mat.V_bi_bc), float(mat.n_R), float(mat.p_R)),
        )
        for index, phi, density_n, density_p in reservoirs:
            values = (nc[index], nv[index], chi[index], eg[index])
            if (
                all(np.isfinite(value) for value in values)
                and nc[index] > 0.0
                and nv[index] > 0.0
                and density_n > 0.0
                and density_p > 0.0
            ):
                conduction_edge = -phi - chi[index]
                valence_edge = conduction_edge - eg[index]
                levels.extend((
                    float(
                        conduction_edge
                        + mat.V_T_device * np.log(density_n / nc[index])
                    ),
                    float(
                        valence_edge
                        - mat.V_T_device * np.log(density_p / nv[index])
                    ),
                ))
                semiconductor_work_functions.extend((
                    float(
                        chi[index]
                        - mat.V_T_device * np.log(density_n / nc[index])
                    ),
                    float(
                        chi[index] + eg[index]
                        + mat.V_T_device * np.log(density_p / nv[index])
                    ),
                ))

    if len(levels) == 4:
        span = float(max(levels) - min(levels))
        metal_mismatch: float | None = None
        if mode == "metal_work_function":
            metal_work_functions = (
                float(stack.work_function_left_eV),
                float(stack.work_function_left_eV),
                float(stack.work_function_right_eV),
                float(stack.work_function_right_eV),
            )
            metal_mismatch = float(np.max(np.abs(
                np.asarray(metal_work_functions)
                - np.asarray(semiconductor_work_functions)
            )))
        qfl_compatible = span <= tolerance_eV
        metal_compatible = (
            metal_mismatch is None or metal_mismatch <= tolerance_eV
        )
        status = (
            "certified"
            if qfl_compatible and metal_compatible
            else "inconsistent"
        )
        if status == "certified":
            message = (
                "carrier reservoirs, Poisson drop, and any explicit metal "
                "work functions agree within the fixed internal "
                f"{tolerance_eV:g} eV gate"
            )
        elif not qfl_compatible:
            message = (
                "Poisson contact potential and carrier reservoirs impose a "
                f"{span:.6g} eV equilibrium quasi-Fermi-level span"
            )
        else:
            message = (
                "explicit metal work functions differ from the modeled "
                "semiconductor reservoirs by as much as "
                f"{metal_mismatch:.6g} eV"
            )
        return ContactThermodynamicCertificate(
            status=status,
            built_in_potential_mode=mode,
            tolerance_eV=float(tolerance_eV),
            fermi_level_span_eV=span,
            potential_mismatch_V=potential_mismatch,
            metal_work_function_mismatch_eV=metal_mismatch,
            contact_quasi_fermi_levels_eV=tuple(levels),
            message=message,
        )

    if potential_mismatch is not None and abs(potential_mismatch) > tolerance_eV:
        status = "inconsistent"
        message = (
            "endpoint DOS data are incomplete, but the imposed Poisson drop "
            "already differs from the band-derived contact potential by "
            f"{abs(potential_mismatch):.6g} V"
        )
    else:
        from perovskite_sim.models.device import electrical_layers

        layers = electrical_layers(stack)
        has_band_reference = any(
            layer.params is not None
            and (layer.params.chi != 0.0 or layer.params.Eg != 0.0)
            for layer in layers
        )
        if has_band_reference:
            status = "compatible_unverified"
            message = (
                "the imposed and band-derived contact potentials agree, but "
                "endpoint effective-DOS data are missing, so the four contact "
                "quasi-Fermi levels cannot be certified"
            )
        else:
            status = "not_assessable"
            message = (
                "the stack lacks endpoint band/DOS data required for a "
                "thermodynamic contact assessment"
            )
    return ContactThermodynamicCertificate(
        status=status,
        built_in_potential_mode=mode,
        tolerance_eV=float(tolerance_eV),
        fermi_level_span_eV=None,
        potential_mismatch_V=potential_mismatch,
        metal_work_function_mismatch_eV=None,
        contact_quasi_fermi_levels_eV=(),
        message=message,
    )


def require_contact_thermodynamic_certificate(
    stack: "DeviceStack",
    mat: "MaterialArrays",
    *,
    tolerance_eV: float = CONTACT_THERMODYNAMIC_TOLERANCE_EV,
) -> ContactThermodynamicCertificate:
    """Return a contact certificate or fail closed for research workflows."""
    certificate = assess_contact_thermodynamics(
        stack, mat, tolerance_eV=tolerance_eV,
    )
    if not certificate.certified:
        raise ContactThermodynamicError(
            f"contact thermodynamic status={certificate.status}: "
            f"{certificate.message}"
        )
    return certificate


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
    "CONTACT_THERMODYNAMIC_TOLERANCE_EV",
    "ContactThermodynamicCertificate",
    "ContactThermodynamicError",
    "assess_contact_thermodynamics",
    "require_contact_thermodynamic_certificate",
    "selective_contact_flux",
    "apply_selective_contacts",
    "schottky_equilibrium_n",
    "schottky_equilibrium_p",
]
