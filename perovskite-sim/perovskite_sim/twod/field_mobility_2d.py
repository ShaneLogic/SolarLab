"""Field-dependent mobility recompute for the 2D solver (Stage B(c.2)).

Stage B(c.2) uses **face-normal** μ(E):
  - x-faces use only |E_x_face|
  - y-faces use only |E_y_face|
  - μ_n / μ_p are recomputed per face from the existing 1D
    ``apply_field_mobility`` primitive
  - D_eff is recovered via the Einstein relation D = μ V_T

Option B (total-|E| with cross-axis interpolation) is explicitly deferred —
see docs/superpowers/specs/2026-04-29-2d-stage-b-c2-field-mobility-design.md.

Mean choices:
  - D_n / D_p face averaging: HARMONIC mean (matches sg_fluxes_2d_* and the
    1D MaterialArrays D_n_face convention).
  - v_sat / ct_beta / pf_gamma face averaging: ARITHMETIC mean (avoids
    harmonic suppression when one side is zero, which would silently disable
    CT/PF at heterointerfaces; the empirical primitives already short-circuit
    on zero individually).
"""
from __future__ import annotations
from typing import NamedTuple
import numpy as np

from perovskite_sim.physics.field_mobility import apply_field_mobility


# Tiny floor used in harmonic mean to avoid 0/0 when both sides are zero.
_EPS_HARMONIC = 1e-300


class FieldMobilityDEff(NamedTuple):
    """Effective per-face diffusion coefficients computed from μ(E)."""
    D_n_x: np.ndarray                  # (Ny, Nx-1)
    D_n_y: np.ndarray                  # (Ny-1, Nx)
    D_p_x: np.ndarray                  # (Ny, Nx-1)
    D_p_y: np.ndarray                  # (Ny-1, Nx)
    D_n_wrap: np.ndarray | None        # (Ny,) when periodic, else None
    D_p_wrap: np.ndarray | None        # (Ny,) when periodic, else None


# ---------------------------------------------------------------------------
# Pure arithmetic-mean face builders (used at build time for v_sat / beta / gamma_pf)
# ---------------------------------------------------------------------------


def arith_mean_face_x(A: np.ndarray) -> np.ndarray:
    """Arithmetic mean of A along the x-axis to produce x-face values.

    A is (Ny, Nx) per-node; output is (Ny, Nx-1) on interior x-faces.
    """
    return 0.5 * (A[:, :-1] + A[:, 1:])


def arith_mean_face_y(A: np.ndarray) -> np.ndarray:
    """Arithmetic mean of A along the y-axis to produce y-face values.

    A is (Ny, Nx) per-node; output is (Ny-1, Nx) on interior y-faces.
    """
    return 0.5 * (A[:-1, :] + A[1:, :])


def arith_mean_face_wrap(A: np.ndarray) -> np.ndarray:
    """Arithmetic mean of A across the periodic-x wrap face (col 0 ↔ col -1).

    A is (Ny, Nx) per-node; output is (Ny,) on the wrap face.
    """
    return 0.5 * (A[:, -1] + A[:, 0])


# ---------------------------------------------------------------------------
# Internal harmonic-mean helpers for D (matches sg_fluxes_2d_* convention)
# ---------------------------------------------------------------------------


def _harmonic_face_x(D: np.ndarray) -> np.ndarray:
    return 2.0 * D[:, :-1] * D[:, 1:] / (D[:, :-1] + D[:, 1:] + _EPS_HARMONIC)


def _harmonic_face_y(D: np.ndarray) -> np.ndarray:
    return 2.0 * D[:-1, :] * D[1:, :] / (D[:-1, :] + D[1:, :] + _EPS_HARMONIC)


def _harmonic_face_wrap(D: np.ndarray) -> np.ndarray:
    return 2.0 * D[:, -1] * D[:, 0] / (D[:, -1] + D[:, 0] + _EPS_HARMONIC)


# ---------------------------------------------------------------------------
# Shape validators
# ---------------------------------------------------------------------------


def _check_x_face(name: str, A: np.ndarray, Ny: int, Nx: int) -> None:
    if A.shape != (Ny, Nx - 1):
        raise ValueError(
            f"Stage B(c.2) face-array shape mismatch for {name}: "
            f"got {A.shape}, expected ({Ny}, {Nx - 1})."
        )


def _check_y_face(name: str, A: np.ndarray, Ny: int, Nx: int) -> None:
    if A.shape != (Ny - 1, Nx):
        raise ValueError(
            f"Stage B(c.2) face-array shape mismatch for {name}: "
            f"got {A.shape}, expected ({Ny - 1}, {Nx})."
        )


def _check_wrap(name: str, A: np.ndarray, Ny: int) -> None:
    if A.shape != (Ny,):
        raise ValueError(
            f"Stage B(c.2) wrap-face shape mismatch for {name}: "
            f"got {A.shape}, expected ({Ny},)."
        )


# ---------------------------------------------------------------------------
# The per-RHS recompute helper
# ---------------------------------------------------------------------------


def recompute_d_eff_2d(
    *,
    phi: np.ndarray,                                # (Ny, Nx)
    x: np.ndarray,                                  # (Nx,)
    y: np.ndarray,                                  # (Ny,)
    D_n: np.ndarray,                                # (Ny, Nx) per-node
    D_p: np.ndarray,                                # (Ny, Nx) per-node
    V_T: float,
    v_sat_n_x_face: np.ndarray,                     # (Ny, Nx-1)
    v_sat_n_y_face: np.ndarray,                     # (Ny-1, Nx)
    ct_beta_n_x_face: np.ndarray,
    ct_beta_n_y_face: np.ndarray,
    pf_gamma_n_x_face: np.ndarray,
    pf_gamma_n_y_face: np.ndarray,
    v_sat_p_x_face: np.ndarray,
    v_sat_p_y_face: np.ndarray,
    ct_beta_p_x_face: np.ndarray,
    ct_beta_p_y_face: np.ndarray,
    pf_gamma_p_x_face: np.ndarray,
    pf_gamma_p_y_face: np.ndarray,
    lateral_bc: str = "neumann",
    v_sat_n_wrap: np.ndarray | None = None,
    v_sat_p_wrap: np.ndarray | None = None,
    ct_beta_n_wrap: np.ndarray | None = None,
    ct_beta_p_wrap: np.ndarray | None = None,
    pf_gamma_n_wrap: np.ndarray | None = None,
    pf_gamma_p_wrap: np.ndarray | None = None,
) -> FieldMobilityDEff:
    """Recompute per-face D_eff from φ using face-normal μ(E).

    Stage B(c.2) face-normal convention:
        x-faces use |E_x_face|; y-faces use |E_y_face|.
    No cross-axis interpolation. apply_field_mobility takes np.abs(E)
    internally, so the input sign is irrelevant.

    Einstein roundtrip: when all v_sat / pf_gamma face arrays are zero,
    D_eff equals harmonic-mean(D_node) on every face — i.e.
    (D / V_T) * V_T = D exactly.
    """
    Ny, Nx = D_n.shape
    if D_p.shape != (Ny, Nx):
        raise ValueError(
            f"Stage B(c.2): D_n shape {D_n.shape} != D_p shape {D_p.shape}"
        )
    if phi.shape != (Ny, Nx):
        raise ValueError(
            f"Stage B(c.2): phi shape {phi.shape} != ({Ny}, {Nx})"
        )

    # Shape checks for interior face params (fail early, no silent broadcast).
    _check_x_face("v_sat_n_x_face",    v_sat_n_x_face,    Ny, Nx)
    _check_y_face("v_sat_n_y_face",    v_sat_n_y_face,    Ny, Nx)
    _check_x_face("ct_beta_n_x_face",  ct_beta_n_x_face,  Ny, Nx)
    _check_y_face("ct_beta_n_y_face",  ct_beta_n_y_face,  Ny, Nx)
    _check_x_face("pf_gamma_n_x_face", pf_gamma_n_x_face, Ny, Nx)
    _check_y_face("pf_gamma_n_y_face", pf_gamma_n_y_face, Ny, Nx)
    _check_x_face("v_sat_p_x_face",    v_sat_p_x_face,    Ny, Nx)
    _check_y_face("v_sat_p_y_face",    v_sat_p_y_face,    Ny, Nx)
    _check_x_face("ct_beta_p_x_face",  ct_beta_p_x_face,  Ny, Nx)
    _check_y_face("ct_beta_p_y_face",  ct_beta_p_y_face,  Ny, Nx)
    _check_x_face("pf_gamma_p_x_face", pf_gamma_p_x_face, Ny, Nx)
    _check_y_face("pf_gamma_p_y_face", pf_gamma_p_y_face, Ny, Nx)

    dx = np.diff(x)
    dy = np.diff(y)

    # Face fields. Sign cancels inside apply_field_mobility (np.abs).
    E_x_face = -(phi[:,  1:] - phi[:, :-1]) / dx[None, :]   # (Ny,   Nx-1)
    E_y_face = -(phi[1:, :] - phi[:-1, :]) / dy[:,  None]   # (Ny-1, Nx)

    # Base mobility at faces via Einstein on harmonic-mean D.
    mu_n_x_base = _harmonic_face_x(D_n) / V_T
    mu_n_y_base = _harmonic_face_y(D_n) / V_T
    mu_p_x_base = _harmonic_face_x(D_p) / V_T
    mu_p_y_base = _harmonic_face_y(D_p) / V_T

    # Apply field mobility (face-normal: x-face uses |E_x|, y-face uses |E_y|).
    mu_n_x_eff = apply_field_mobility(
        mu_n_x_base, np.abs(E_x_face),
        v_sat_n_x_face, ct_beta_n_x_face, pf_gamma_n_x_face,
    )
    mu_n_y_eff = apply_field_mobility(
        mu_n_y_base, np.abs(E_y_face),
        v_sat_n_y_face, ct_beta_n_y_face, pf_gamma_n_y_face,
    )
    mu_p_x_eff = apply_field_mobility(
        mu_p_x_base, np.abs(E_x_face),
        v_sat_p_x_face, ct_beta_p_x_face, pf_gamma_p_x_face,
    )
    mu_p_y_eff = apply_field_mobility(
        mu_p_y_base, np.abs(E_y_face),
        v_sat_p_y_face, ct_beta_p_y_face, pf_gamma_p_y_face,
    )

    # Recover D via Einstein relation. Use the SAME V_T as the divide above
    # (single source of truth) so the v_sat=pf_gamma=0 path is exact.
    D_n_x_eff = mu_n_x_eff * V_T
    D_n_y_eff = mu_n_y_eff * V_T
    D_p_x_eff = mu_p_x_eff * V_T
    D_p_y_eff = mu_p_y_eff * V_T

    # Periodic wrap face (only when lateral_bc == "periodic").
    D_n_wrap_eff: np.ndarray | None = None
    D_p_wrap_eff: np.ndarray | None = None
    if lateral_bc == "periodic":
        if (v_sat_n_wrap is None or v_sat_p_wrap is None
                or ct_beta_n_wrap is None or ct_beta_p_wrap is None
                or pf_gamma_n_wrap is None or pf_gamma_p_wrap is None):
            raise ValueError(
                "Stage B(c.2): lateral_bc='periodic' requires all six "
                "*_wrap face arrays to be provided."
            )
        _check_wrap("v_sat_n_wrap",    v_sat_n_wrap,    Ny)
        _check_wrap("v_sat_p_wrap",    v_sat_p_wrap,    Ny)
        _check_wrap("ct_beta_n_wrap",  ct_beta_n_wrap,  Ny)
        _check_wrap("ct_beta_p_wrap",  ct_beta_p_wrap,  Ny)
        _check_wrap("pf_gamma_n_wrap", pf_gamma_n_wrap, Ny)
        _check_wrap("pf_gamma_p_wrap", pf_gamma_p_wrap, Ny)
        dx_wrap = 0.5 * (dx[0] + dx[-1])
        E_x_wrap = -(phi[:, 0] - phi[:, -1]) / dx_wrap     # (Ny,)
        mu_n_wrap_base = _harmonic_face_wrap(D_n) / V_T
        mu_p_wrap_base = _harmonic_face_wrap(D_p) / V_T
        mu_n_wrap_eff = apply_field_mobility(
            mu_n_wrap_base, np.abs(E_x_wrap),
            v_sat_n_wrap, ct_beta_n_wrap, pf_gamma_n_wrap,
        )
        mu_p_wrap_eff = apply_field_mobility(
            mu_p_wrap_base, np.abs(E_x_wrap),
            v_sat_p_wrap, ct_beta_p_wrap, pf_gamma_p_wrap,
        )
        D_n_wrap_eff = mu_n_wrap_eff * V_T
        D_p_wrap_eff = mu_p_wrap_eff * V_T

    return FieldMobilityDEff(
        D_n_x=D_n_x_eff,
        D_n_y=D_n_y_eff,
        D_p_x=D_p_x_eff,
        D_p_y=D_p_y_eff,
        D_n_wrap=D_n_wrap_eff,
        D_p_wrap=D_p_wrap_eff,
    )


__all__ = [
    "arith_mean_face_x", "arith_mean_face_y", "arith_mean_face_wrap",
    "recompute_d_eff_2d", "FieldMobilityDEff",
]
