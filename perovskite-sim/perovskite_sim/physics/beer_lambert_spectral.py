"""Single-pass AM1.5G absorption, without interference or reflection.

Uses the same material k(lambda), wavelength grid and solar spectrum as TMM.
Exact dual-cell absorption keeps the photon budget independent of the mesh.
The scalar Beer-Lambert path remains available for prescribed alpha and Phi.
"""
from __future__ import annotations

import numpy as np

from perovskite_sim._compat.numpy_compat import trapezoid
from perovskite_sim.data import load_am15g
from perovskite_sim.physics.generation import dual_cell_faces, dual_cell_widths
from perovskite_sim.physics.optical_stack import (
    build_device_optical_stack,
    has_wavelength_resolved_optics,
)


def spectral_beer_lambert_generation(x, stack, n_wavelengths=200):
    """Return G [m^-3 s^-1] for one sun, 300-1000 nm, alpha=4*pi*k/lambda.

    Substrate absorption attenuates the incident spectrum but does not create
    carriers on the electrical mesh. No photon-recycling correction is used.
    """
    if not has_wavelength_resolved_optics(stack):
        raise ValueError("Spectral Beer-Lambert requires optical material data")
    wavelengths_nm = np.linspace(300.0, 1000.0, n_wavelengths)
    wavelengths_m = wavelengths_nm * 1e-9
    _, flux = load_am15g(wavelengths_nm)
    optical = build_device_optical_stack(stack, wavelengths_nm)
    offset = sum(layer.thickness for layer in stack.layers if layer.role == "substrate")
    faces = dual_cell_faces(np.asarray(x, dtype=float)) + offset
    tau = np.zeros((len(faces), n_wavelengths))
    for index, layer in enumerate(optical.layers):
        alpha = 4.0 * np.pi * layer.k / wavelengths_m
        if np.any(~np.isfinite(alpha)) or np.any(alpha < 0):
            raise ValueError("Beer-Lambert absorption coefficients must be finite and nonnegative")
        distance = np.clip(faces - optical.boundaries_m[index], 0.0, layer.d)
        tau += distance[:, None] * alpha[None, :]
    absorbed = np.exp(-tau[:-1]) * -np.expm1(-np.diff(tau, axis=0))
    return trapezoid(absorbed * flux[None, :], wavelengths_m, axis=1) / dual_cell_widths(x)
