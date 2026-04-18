"""Spatially varying trap profiles (Phase 4a — Apr 2026).

The bulk SRH lifetime ``tau_{n,p}`` carried on each ``MaterialParams`` is
the lifetime in the cleanest part of the layer. In real perovskite films
the trap density is not uniform — grain boundaries near each transport-
layer interface concentrate point defects (Ti–O dangling bonds at the
ETL contact, halide vacancies at the HTL contact) and the local SRH
lifetime drops accordingly. This module computes a per-node trap density
``N_t(x)`` from a layer's ``trap_N_t_interface``, ``trap_N_t_bulk`` and
``trap_decay_length`` parameters, and converts it into a per-node SRH
lifetime via the inverse-density rule ``tau(x) = tau_bulk · N_t_bulk /
N_t(x)``.

Two profile shapes are exposed:

- ``exponential_edge_profile``: ``N_t(x) = N_t_bulk + (N_t_interface −
  N_t_bulk) · (e^{−x_local/L_d} + e^{−(d−x_local)/L_d})``. This is the
  baseline introduced in Phase 4. The two exponentials are added rather
  than maxed so that very thin layers with overlapping decay regions
  smoothly saturate at ``N_t_interface`` instead of double-counting only
  one side. ``L_d`` controls how far the interface contamination reaches
  into the bulk; a typical perovskite L_d is 5–30 nm.

- ``gaussian_edge_profile``: ``N_t(x) = N_t_bulk + (N_t_interface −
  N_t_bulk) · (G(x_local) + G(d − x_local))`` where ``G(s) =
  exp(−(s/sigma)²)``. Gaussian decay is faster than exponential at
  ``s ≫ sigma`` and slower at ``s ≪ sigma`` — useful when the interface
  trap layer has a finite well-defined extent rather than a long tail.

The conversion ``tau_from_trap_density`` is shared by both shapes; a
floor of ``N_t_bulk`` on the denominator keeps the result physical when
a user accidentally passes ``N_t_interface < N_t_bulk`` (no negative
correction).

The whole module is gated downstream by
``SimulationMode.use_trap_profile`` (on in FAST and FULL, off in
LEGACY); when the flag is off ``build_material_arrays`` skips the
profile computation entirely and ``MaterialArrays.N_t_node`` is filled
with ``N_t_bulk`` (or ``0.0`` outside the layer) so that diagnostics
clients can still read it without a None check.
"""
from __future__ import annotations

import numpy as np


def exponential_edge_profile(
    x_local: np.ndarray,
    thickness: float,
    N_t_interface: float,
    N_t_bulk: float,
    L_d: float,
) -> np.ndarray:
    """Return ``N_t(x)`` for an exponential interface-trap distribution.

    Parameters
    ----------
    x_local
        Position within the layer (``0 <= x_local <= thickness``), in m.
    thickness
        Layer thickness in m.
    N_t_interface
        Trap density at each interface (in m⁻³).
    N_t_bulk
        Trap density deep in the bulk (in m⁻³).
    L_d
        Decay length in m.
    """
    if L_d <= 0.0:
        # No decay → uniform bulk everywhere; no edge contribution.
        return np.full_like(x_local, N_t_bulk)
    d_left = x_local
    d_right = thickness - x_local
    edge = (
        np.exp(-d_left / L_d) + np.exp(-d_right / L_d)
    )
    return N_t_bulk + (N_t_interface - N_t_bulk) * edge


def gaussian_edge_profile(
    x_local: np.ndarray,
    thickness: float,
    N_t_interface: float,
    N_t_bulk: float,
    sigma: float,
) -> np.ndarray:
    """Return ``N_t(x)`` for a Gaussian interface-trap distribution.

    The Gaussian falls off faster than the exponential at large
    ``s = distance-from-interface``, so the bulk recovers the clean tau
    more quickly. Useful for systems where the interface defect layer
    has a measured finite thickness (e.g. a few nm grain-boundary slab).
    """
    if sigma <= 0.0:
        return np.full_like(x_local, N_t_bulk)
    d_left = x_local
    d_right = thickness - x_local
    edge = np.exp(-((d_left / sigma) ** 2)) + np.exp(-((d_right / sigma) ** 2))
    return N_t_bulk + (N_t_interface - N_t_bulk) * edge


def tau_from_trap_density(
    tau_bulk: np.ndarray | float,
    N_t: np.ndarray | float,
    N_t_bulk: float,
) -> np.ndarray | float:
    """Apply the SRH inverse-density rule ``tau(x) = tau_bulk · N_t_bulk / N_t(x)``.

    The denominator is floored at ``N_t_bulk`` so that
    ``N_t_interface < N_t_bulk`` (a passivation profile rather than a
    contamination one) still yields a physical, monotone correction.
    Callers that explicitly want the passivation regime should compute
    the ratio themselves.
    """
    N_t_arr = np.asarray(N_t, dtype=float)
    floor = max(float(N_t_bulk), 1.0)
    safe = np.maximum(N_t_arr, floor)
    return tau_bulk * float(N_t_bulk) / safe


def has_trap_profile_params(params) -> bool:
    """Return True iff the layer's params provide a complete trap profile."""
    return (
        getattr(params, "trap_N_t_interface", None) is not None
        and getattr(params, "trap_N_t_bulk", None) is not None
        and getattr(params, "trap_decay_length", None) is not None
    )
