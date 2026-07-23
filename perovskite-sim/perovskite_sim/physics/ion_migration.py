from __future__ import annotations
import numpy as np

from perovskite_sim.discretization.fe_operators import bernoulli


def _steric_diffusion_only_flux(P, phi, dx, D_I_face, V_T, P_lim,
                                P_lim_node, P_other_node, drift_sign):
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
) -> np.ndarray:
    """
    Vectorized dP/dt = -dF_P/dx for all nodes.
    Zero-flux BCs at both contacts.

    Two steric forms are supported (review F05):

    * Legacy (``steric_diffusion_only=False``, default). The steric factor
      ``s = 1/(1 - P/P_lim)`` multiplies the WHOLE Scharfetter-Gummel flux
      ``D_eff = D_I * s``, i.e. it amplifies drift and diffusion equally.
      This is an empirical crowding regularization, not the strict
      modified-Poisson-Nernst-Planck flux (which corrects only the
      concentration-gradient term). Kept as the default because it is what
      every validated benchmark was pinned against.

    * Physical diffusion-only (``steric_diffusion_only=True``). Folds the
      lattice-gas crowding chemical potential ``mu_ex/kT = -ln(1 - P/P_lim)``
      into the SG drift argument and leaves the diffusion coefficient at the
      bare ``D_I``. This is the dimensionally faithful modified-PNP flux
      ``F = -D_I[ (1/(1-P/P_lim)) dP/dx + (P/V_T) dphi/dx ]`` written in a
      generalized-Slotboom (crowding-in-potential) form, so it applies
      steric to diffusion only AND preserves the Bernoulli exponential
      fitting (stability). Equilibrium ``F=0`` is preserved: the flux
      vanishes at the steric-Boltzmann profile. Measured negligible in the
      dilute regime of the shipped presets (max P/P_lim ~ 0.011 on
      ionmonger_benchmark, so ``s ~ 1.011``); it only diverges from the
      legacy form as P approaches P_lim. Requires ``P_lim_node`` (per-node
      site density) for the node crowding potential; falls back to the
      broadcast ``P_lim`` when None.
    """
    P = np.asarray(P, dtype=float)
    dx = np.diff(x)                              # (N-1,)
    D_I_face = np.broadcast_to(np.asarray(D_I, dtype=float), dx.shape)
    if steric_diffusion_only:
        F_int = _steric_diffusion_only_flux(
            P, phi, dx, D_I_face, V_T, P_lim, P_lim_node, P_other_node,
            drift_sign=+1.0,
        )
    else:
        P_lim_face = np.broadcast_to(np.asarray(P_lim, dtype=float), dx.shape)
        P_avg = 0.5 * (P[:-1] + P[1:])          # (N-1,)
        steric = 1.0 / np.maximum(
            1.0 - np.clip(P_avg / P_lim_face, 0.0, 0.999999), 1e-6)
        xi = (phi[1:] - phi[:-1]) / V_T         # (N-1,)
        D_eff = D_I_face * steric
        F_int = D_eff / dx * (bernoulli(xi) * P[:-1] - bernoulli(-xi) * P[1:])

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
) -> np.ndarray:
    """dP_neg/dt for a negatively charged mobile ion species.

    Same SG discretization as the positive species but with reversed drift
    direction (``q_neg = -q``). Supports the same two steric forms (review
    F05): the legacy whole-flux factor (default), or the physical
    diffusion-only crowding form via ``_steric_diffusion_only_flux`` with
    ``drift_sign=-1``. For shared-site dual-ion the crowding uses the total
    occupancy (pass the positive-species density as ``P_other_node``), so
    the two species compete for one reservoir symmetrically with the
    positive equation. Zero-flux BCs at both contacts.
    """
    P_neg = np.asarray(P_neg, dtype=float)
    dx = np.diff(x)
    D_I_face = np.broadcast_to(np.asarray(D_I, dtype=float), dx.shape)
    if steric_diffusion_only:
        F_int = _steric_diffusion_only_flux(
            P_neg, phi, dx, D_I_face, V_T, P_lim, P_lim_node, P_other_node,
            drift_sign=-1.0,
        )
    else:
        P_lim_face = np.broadcast_to(np.asarray(P_lim, dtype=float), dx.shape)
        P_avg = 0.5 * (P_neg[:-1] + P_neg[1:])
        steric = 1.0 / np.maximum(
            1.0 - np.clip(P_avg / P_lim_face, 0.0, 0.999999), 1e-6)
        # Reversed drift: negative charge → xi flipped
        xi = -(phi[1:] - phi[:-1]) / V_T
        D_eff = D_I_face * steric
        F_int = D_eff / dx * (bernoulli(xi) * P_neg[:-1] - bernoulli(-xi) * P_neg[1:])

    F_full = np.concatenate([[0.0], F_int, [0.0]])

    dx_cell = np.empty(len(x))
    dx_cell[0]    = dx[0]
    dx_cell[-1]   = dx[-1]
    dx_cell[1:-1] = 0.5 * (dx[:-1] + dx[1:])

    return -(F_full[1:] - F_full[:-1]) / dx_cell
