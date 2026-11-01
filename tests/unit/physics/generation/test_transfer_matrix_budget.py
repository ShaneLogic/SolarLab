"""Analytic and conservation checks of the transfer-matrix optics and photon budget."""

from __future__ import annotations

import math

import numpy as np
import pytest

from solarlab.mesh.geometry import half_endpoint_geometry, integer_endpoint_geometry
from solarlab.physics.generation import (
    OpticalLayerSpec,
    OpticalStackBinding,
    SpectralPhotonFlux,
    TMMLayer,
    dropped_multipass_fraction,
    tmm_absorption_profile,
    tmm_cumulative_absorptance,
    tmm_reflectance,
    tmm_transmittance,
)

WL_NM = np.linspace(350.0, 780.0, 60)
WL_M = WL_NM * 1e-9


def _spectrum(wavelengths_nm=WL_NM):
    return SpectralPhotonFlux(wavelengths_nm, np.linspace(2e27, 4e27, wavelengths_nm.size), "synthetic")


def _coherent_specs():
    n_wl = WL_NM.size
    return (
        OpticalLayerSpec("etl", 50e-9, np.full(n_wl, 2.4), np.where(WL_NM < 380, 0.05, 0.001)),
        OpticalLayerSpec("absorber", 400e-9, np.full(n_wl, 2.5), 0.4 * np.exp(-(WL_NM - 400) / 150)),
        OpticalLayerSpec("htl", 150e-9, np.full(n_wl, 1.8), np.where(WL_NM < 420, 0.02, 0.001)),
    )


def _glass(k=1e-6):
    n_wl = WL_NM.size
    return OpticalLayerSpec("glass", 1e-3, np.full(n_wl, 1.52), np.full(n_wl, k), incoherent=True, electrical=False)


def test_lossless_slab_matches_the_airy_formula():
    n, d = 2.3, 410e-9
    layers = [TMMLayer(d=d, n=np.full(WL_M.size, n), k=np.zeros(WL_M.size))]
    r = (1.0 - n) / (1.0 + n)
    cos2 = np.cos(4.0 * np.pi * n * d / WL_M)
    airy = 2.0 * r**2 * (1.0 - cos2) / (1.0 + r**4 - 2.0 * r**2 * cos2)
    np.testing.assert_allclose(tmm_reflectance(layers, WL_M), airy, rtol=0.0, atol=1e-13)
    np.testing.assert_allclose(tmm_transmittance(layers, WL_M), 1.0 - airy, rtol=0.0, atol=1e-13)
    assert np.all(tmm_cumulative_absorptance(layers, WL_M, np.linspace(0, d, 21), np.array([0.0, d])) == 0.0)


def test_quarter_wave_coating_cancels_reflection_at_its_design_wavelength():
    n_ambient, n_exit, lam = 1.0, 2.25, 550e-9
    n_coat = math.sqrt(n_ambient * n_exit)
    layers = [TMMLayer(d=lam / (4.0 * n_coat), n=np.array([n_coat]), k=np.array([0.0]))]
    R = tmm_reflectance(layers, np.array([lam]), n_ambient=n_ambient, n_substrate=n_exit)
    T = tmm_transmittance(layers, np.array([lam]), n_ambient=n_ambient, n_substrate=n_exit)
    assert R[0] < 1e-30 and T[0] == pytest.approx(1.0, abs=1e-14)


def test_coherent_photon_budget_closes_to_round_off():
    budget = OpticalStackBinding(_coherent_specs(), _spectrum()).photon_budget()
    assert np.max(np.abs(budget.spectral_closure_residual())) < 1e-12
    assert np.all(budget.dropped_multipass == 0.0)
    assert np.all(budget.layer_absorptance >= 0.0)
    assert budget.parasitic_absorbed_m2_s == 0.0
    total = budget.reflected_m2_s + budget.transmitted_m2_s + float(np.sum(budget.layer_absorbed_m2_s))
    assert total == pytest.approx(budget.incident_m2_s, rel=1e-12)


def test_incoherent_substrate_budget_books_the_dropped_round_trips():
    binding = OpticalStackBinding((_glass(k=2e-7), *_coherent_specs()), _spectrum())
    budget = binding.photon_budget()
    assert np.max(np.abs(budget.spectral_closure_residual())) < 1e-12
    assert np.all(budget.dropped_multipass > 0.0)
    assert budget.parasitic_absorbed_m2_s > 0.0
    # The substrate is optical-only: its absorption is parasitic, never generation.
    glass_alpha = 4.0 * np.pi * 2e-7 / WL_M
    r0 = (1.0 - 1.52) / (1.0 + 1.52)
    analytic_glass = (1.0 - r0**2) * -np.expm1(-glass_alpha * 1e-3)
    np.testing.assert_allclose(budget.layer_absorptance[0], analytic_glass, rtol=1e-12)
    layers = list(binding.tmm_layers)
    np.testing.assert_array_equal(budget.dropped_multipass, dropped_multipass_fraction(layers, WL_M))


def test_cumulative_absorptance_is_the_exact_antiderivative_of_the_absorption_rate():
    layers = [TMMLayer(d=s.thickness_m, n=s.n, k=s.k) for s in _coherent_specs()]
    boundaries = np.array([0.0, 50e-9, 450e-9, 600e-9])
    # Strictly inside the absorber, beyond the 1e-15 m boundary-snapping
    # tolerance of the layer map, so the reference never straddles a jump of A.
    inset = 1e-6 * 400e-9
    xa, xb = 50e-9 + inset, 450e-9 - inset
    C = tmm_cumulative_absorptance(layers, WL_M, np.array([xa, xb]), boundaries)
    exact = C[1] - C[0]
    errors = []
    for n_pts in (2001, 8001):
        xs = np.linspace(xa, xb, n_pts)
        errors.append(float(np.max(np.abs(exact - np.trapezoid(tmm_absorption_profile(layers, WL_M, xs, boundaries),
                                                                xs, axis=0)))))
    # Second-order trapezoid convergence onto the closed form (x16 per 4x refinement).
    assert errors[1] < errors[0] / 8.0
    assert errors[1] < 1e-7


def test_float64_window_guards_still_raise():
    n_wl = 20
    wl = np.linspace(400e-9, 700e-9, n_wl)
    huge = [TMMLayer(d=20e-6, n=np.full(n_wl, 3.0), k=np.full(n_wl, 2.0))]
    with pytest.raises(ValueError, match="not representable in float64"):
        tmm_cumulative_absorptance(huge, wl, np.linspace(0.0, 20e-6, 51), np.array([0.0, 20e-6]))
    lam = np.array([500e-9])
    k = 90.0 * 500e-9 / (4.0 * np.pi * 20e-6)  # alpha * d = 90: past the ~70 accuracy limit
    lossy = [TMMLayer(d=20e-6, n=np.array([3.0]), k=np.array([k]))]
    with pytest.raises(ValueError, match="above the physical"):
        tmm_cumulative_absorptance(lossy, lam, np.linspace(0.0, 20e-6, 41), np.array([0.0, 20e-6]))


def test_only_the_first_layer_may_be_incoherent():
    layers = [TMMLayer(d=1e-7, n=np.ones(WL_M.size) * 2, k=np.zeros(WL_M.size)),
              TMMLayer(d=1e-3, n=np.ones(WL_M.size) * 1.5, k=np.zeros(WL_M.size), incoherent=True)]
    with pytest.raises(ValueError, match="only the first layer"):
        tmm_reflectance(layers, WL_M)


@pytest.mark.parametrize("builder", [integer_endpoint_geometry, half_endpoint_geometry])
def test_cell_absorbed_photons_are_mesh_independent_and_equal_the_electrical_budget(builder):
    binding = OpticalStackBinding((_glass(), *_coherent_specs()), _spectrum())
    budget = binding.photon_budget()
    reference = budget.electrical_absorbed_m2_s
    ceiling = budget.incident_m2_s - budget.reflected_m2_s - budget.transmitted_m2_s
    for n_nodes in (4, 9, 31, 120, 400):
        x = np.linspace(0.0, binding.electrical_thickness_m, n_nodes)
        profile = binding.generation_profile(builder(x))
        assert profile.areal_generation_m2_s == pytest.approx(reference, rel=1e-12)
        assert profile.areal_generation_m2_s <= ceiling * (1.0 + 1e-12)
        np.testing.assert_allclose(profile.source_m3_s * builder(x).cell_measures_m, profile.absorbed_m2_s,
                                   rtol=4e-16, atol=0.0)
        assert np.all(profile.absorbed_m2_s >= 0.0)


def test_absorption_profile_reads_the_electrical_coordinate():
    binding = OpticalStackBinding((_glass(), *_coherent_specs()), _spectrum())
    x = np.linspace(0.0, binding.electrical_thickness_m, 11)
    layers = list(binding.tmm_layers)
    expected = tmm_absorption_profile(layers, WL_M, x + 1e-3, binding.boundaries_m)
    np.testing.assert_array_equal(binding.absorption_profile(x), expected)
