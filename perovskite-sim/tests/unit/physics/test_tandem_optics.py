import numpy as np
import pytest

from perovskite_sim.physics.tandem_optics import (
    TandemGeneration,
    partition_absorption,
)


def test_partition_assigns_layer_ranges_correctly():
    # 10 grid points, 3 wavelengths. Points 0-3: top, 4-5: junction, 6-9: bottom.
    A = np.ones((10, 3))  # uniform absorption rate [m^-1]
    x = np.linspace(0.0, 1.0, 10)
    spectral_flux = np.array([1.0, 2.0, 3.0])
    wavelengths = np.array([400e-9, 500e-9, 600e-9])
    top_slice = slice(0, 4)
    junction_slice = slice(4, 6)
    bot_slice = slice(6, 10)

    G_top, G_bot, parasitic = partition_absorption(
        A, x, wavelengths, spectral_flux,
        top_slice, junction_slice, bot_slice,
    )

    assert G_top.shape == (4,)
    assert G_bot.shape == (4,)
    assert np.all(G_top > 0)
    assert np.all(G_bot > 0)
    assert 0.0 < parasitic < 1.0


def test_partition_conserves_photon_count():
    rng = np.random.default_rng(0)
    N = 30
    n_wl = 8
    A = rng.uniform(0.0, 1.0, size=(N, n_wl))
    x = np.linspace(0.0, 500e-9, N)
    wavelengths = np.linspace(400e-9, 800e-9, n_wl)
    spectral_flux = np.full(n_wl, 1e21)

    top = slice(0, 10)
    junc = slice(10, 14)
    bot = slice(14, N)

    G_top, G_bot, parasitic = partition_absorption(
        A, x, wavelengths, spectral_flux, top, junc, bot,
    )

    top_photons = float(np.trapezoid(G_top, x[top]))
    bot_photons = float(np.trapezoid(G_bot, x[bot]))
    total_incident = float(np.trapezoid(spectral_flux, wavelengths))
    junc_photons = parasitic * total_incident

    integrand = A * spectral_flux[None, :]
    G_full = np.trapezoid(integrand, wavelengths, axis=1)
    full_photons = float(np.trapezoid(G_full, x))

    assert top_photons + junc_photons + bot_photons == pytest.approx(
        full_photons, rel=1e-9
    )
