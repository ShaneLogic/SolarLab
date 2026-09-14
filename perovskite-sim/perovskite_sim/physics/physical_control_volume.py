"""Physical 1D volumes for the opt-in R1 lane, not the legacy MoL geometry."""

from __future__ import annotations

import numpy as np

PHYSICAL_GEOMETRY_V1 = "physical_boundary_two_sided_v1"


def physical_cell_faces(x, interface_positions=()):
    """Clip outer cells at contacts; split unlike materials at the true plane."""
    x = np.asarray(x, dtype=float)
    positions = np.asarray(interface_positions, dtype=float)
    if (
        x.ndim != 1
        or x.size < 3
        or not np.all(np.isfinite(x))
        or np.any(np.diff(x) <= 0)
    ):
        raise ValueError("physical grid must be finite and strictly increasing")
    if (
        positions.ndim != 1
        or not np.all(np.isfinite(positions))
        or np.any(np.diff(positions) <= 0)
    ):
        raise ValueError("interface positions must be finite and ordered")
    faces = np.r_[x[0], (x[:-1] + x[1:]) / 2, x[-1]]
    used = set()
    for position in positions:
        right = int(np.searchsorted(x, position))
        if right == 0 or right == x.size or x[right] == position or right in used:
            raise ValueError("each interface needs its own two-sided grid gap")
        faces[right] = position
        used.add(right)
    if np.any(np.diff(faces) <= 0):
        raise ValueError("nonpositive physical control volume")
    faces.setflags(write=False)
    return faces


def flux_divergence(flux, widths, *, left_flux=0.0, right_flux=0.0):
    """Particle rate: w*dP/dt = F_left - F_right, including real boundaries."""
    widths = np.asarray(widths, dtype=float)
    flux = np.asarray(flux)
    if (
        widths.shape != (flux.size + 1,)
        or flux.ndim != 1
        or not np.all(np.isfinite(widths))
        or np.any(widths <= 0)
    ):
        raise ValueError("flux and positive finite volumes must align")
    return -np.diff(np.r_[left_flux, flux, right_flux]) / widths


def physical_contact_displacement(face_displacement, rho, widths):
    """Gauss extension from adjacent internal faces to the physical contacts."""
    return np.asarray(
        [
            face_displacement[0] - rho[0] * widths[0],
            face_displacement[-1] + rho[-1] * widths[-1],
        ]
    )
