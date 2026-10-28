"""Optical binding: declared inputs, rejection predicates and stale-profile identity."""

from __future__ import annotations

import math

import numpy as np
import pytest

from solarlab.mesh.geometry import half_endpoint_geometry, integer_endpoint_geometry
from solarlab.physics.generation import (
    FixedGenerationProfile,
    OpticalLayerSpec,
    OpticalStackBinding,
    SpectralPhotonFlux,
    StaleGenerationProfileError,
    constant_index_optical_constants,
    graded_slice_centres,
)

WL = np.linspace(400.0, 800.0, 41)


def _spectrum(scale=1.0):
    return SpectralPhotonFlux(WL, scale * np.full(WL.size, 3e27), "flat")


def _layer(name="absorber", thickness=300e-9, n=2.5, k=0.1, **kwargs):
    return OpticalLayerSpec(name, thickness, np.full(WL.size, n), np.full(WL.size, k), **kwargs)


def _binding(**kwargs):
    layers = kwargs.pop("layers", (_layer("glass", 1e-3, 1.5, 0.0, incoherent=True, electrical=False),
                                   _layer("etl", 50e-9, 2.1, 0.0), _layer()))
    return OpticalStackBinding(layers, kwargs.pop("spectrum", _spectrum()), **kwargs)


def test_identity_is_reproducible_and_tracks_every_declared_input():
    base = _binding()
    assert _binding().sha256 == base.sha256
    variants = {
        "thickness": _binding(layers=(_layer("glass", 1e-3, 1.5, 0.0, incoherent=True, electrical=False),
                                      _layer("etl", 50e-9, 2.1, 0.0), _layer(thickness=301e-9))),
        "n": _binding(layers=(_layer("glass", 1e-3, 1.5, 0.0, incoherent=True, electrical=False),
                              _layer("etl", 50e-9, 2.1, 0.0), _layer(n=2.51))),
        "k": _binding(layers=(_layer("glass", 1e-3, 1.5, 0.0, incoherent=True, electrical=False),
                              _layer("etl", 50e-9, 2.1, 0.0), _layer(k=0.11))),
        "spectrum": _binding(spectrum=_spectrum(1.01)),
        "temperature": _binding(dependencies={"temperature_K": 310.0}),
        "composition": _binding(dependencies={"composition:absorber:ggi": 0.3}),
        "ambient": _binding(ambient_index=1.33),
        "coherence": _binding(layers=(_layer("glass", 1e-3, 1.5, 0.0, electrical=False),
                                      _layer("etl", 50e-9, 2.1, 0.0), _layer())),
    }
    digests = {name: item.sha256 for name, item in variants.items()}
    assert len(set(digests.values()) | {base.sha256}) == len(digests) + 1
    assert _binding(dependencies={"temperature_K": 310.0}).sha256 == digests["temperature"]


def test_a_profile_is_never_reused_after_its_inputs_change():
    base = _binding()
    x = np.linspace(0.0, base.electrical_thickness_m, 25)
    geometry = integer_endpoint_geometry(x)
    profile = base.generation_profile(geometry)
    profile.require_inputs(base.sha256, geometry)
    hotter = _binding(dependencies={"temperature_K": 350.0})
    with pytest.raises(StaleGenerationProfileError):
        profile.require_inputs(hotter.sha256, geometry)
    with pytest.raises(StaleGenerationProfileError):
        profile.require_inputs(base.sha256, half_endpoint_geometry(x))
    with pytest.raises(StaleGenerationProfileError):
        profile.require_inputs(base.sha256, integer_endpoint_geometry(np.linspace(0.0, base.electrical_thickness_m, 26)))


def test_geometry_convention_is_recorded_and_changes_only_the_contact_cells():
    base = _binding()
    x = np.linspace(0.0, base.electrical_thickness_m, 25)
    whole = base.generation_profile(integer_endpoint_geometry(x))
    half = base.generation_profile(half_endpoint_geometry(x))
    assert whole.geometry_name == "legacy_mol_integer_endpoint_v1"
    assert half.geometry_name == "legacy_twod_half_endpoint_v1"
    assert whole.absorbed_m2_s.tobytes() == half.absorbed_m2_s.tobytes()
    assert np.array_equal(whole.source_m3_s[1:-1], half.source_m3_s[1:-1])
    # Integer endpoint: the contact cell's photons spread over the full dx.
    assert np.array_equal(half.source_m3_s[[0, -1]], 2.0 * whole.source_m3_s[[0, -1]])


@pytest.mark.parametrize(
    "layers, message",
    [
        ((_layer("etl", electrical=True), _layer("glass", 1e-3, 1.5, 0.0, electrical=False)), "contiguous front prefix"),
        ((_layer("etl"), _layer("glass", 1e-3, 1.5, 0.0, incoherent=True)), "only the first"),
        ((_layer("glass", 1e-3, 1.5, 0.0, electrical=False),), "at least one electrical"),
        ((_layer("a"), _layer("b"), _layer("a")), "contiguous"),
    ],
)
def test_stack_level_rejections(layers, message):
    with pytest.raises(ValueError, match=message):
        OpticalStackBinding(layers, _spectrum())


@pytest.mark.parametrize(
    "kwargs",
    [dict(incidence="back"), dict(ambient_index=0.0), dict(exit_index=math.nan),
     dict(dependencies={"temperature_K": math.inf}), dict(dependencies={"flag": True})],
)
def test_declaration_level_rejections(kwargs):
    with pytest.raises(ValueError):
        _binding(**kwargs)


@pytest.mark.parametrize(
    "spec",
    [
        dict(thickness=0.0),
        dict(n=0.0),
        dict(k=-1e-3),
        dict(n=math.nan),
    ],
)
def test_layer_level_rejections(spec):
    with pytest.raises(ValueError):
        _layer(**spec)


def test_optical_constants_must_share_the_spectrum_grid():
    short = OpticalLayerSpec("a", 1e-7, np.full(5, 2.0), np.zeros(5))
    with pytest.raises(ValueError, match="wavelength grid"):
        OpticalStackBinding((short,), _spectrum())


def test_spectrum_declares_its_unit_and_rejects_irradiance():
    with pytest.raises(ValueError, match="m\\^-2 s\\^-1 m\\^-1"):
        SpectralPhotonFlux(WL, np.ones(WL.size), "am15g", unit="W m^-2 nm^-1")
    with pytest.raises(ValueError):
        SpectralPhotonFlux(WL, -np.ones(WL.size), "bad")
    flat = _spectrum()
    assert flat.integrated_photon_flux_m2_s() == pytest.approx(3e27 * 400e-9, rel=1e-14)


def test_faces_outside_the_electrical_window_are_rejected():
    base = _binding()
    too_long = np.linspace(0.0, 1.01 * base.electrical_thickness_m, 11)
    with pytest.raises(ValueError, match="electrical window"):
        base.generation_profile(integer_endpoint_geometry(too_long))


def test_constant_index_constants_reproduce_the_absorption_coefficient():
    n, k = constant_index_optical_constants(WL, absorption_coefficient_m1=1e7, refractive_index=2.2)
    assert np.all(n == 2.2)
    np.testing.assert_allclose(4.0 * np.pi * k / (WL * 1e-9), 1e7, rtol=1e-15)
    n_eps, _ = constant_index_optical_constants(WL, absorption_coefficient_m1=0.0, relative_permittivity=6.25)
    assert np.all(n_eps == 2.5)
    with pytest.raises(ValueError):
        constant_index_optical_constants(WL, absorption_coefficient_m1=0.0)


def test_graded_slices_partition_the_layer():
    centres, thickness = graded_slice_centres(500e-9, 8)
    assert thickness * 8 == pytest.approx(500e-9, rel=1e-15)
    assert centres[0] == pytest.approx(thickness / 2) and np.all(np.diff(centres) > 0)
    with pytest.raises(ValueError):
        graded_slice_centres(500e-9, 0)


def test_fixed_generation_is_injected_verbatim_with_its_provenance():
    values = np.linspace(1e26, 3e27, 15)
    fixed = FixedGenerationProfile(values, 15)
    geometry = integer_endpoint_geometry(np.linspace(0.0, 4e-7, 15))
    profile = fixed.generation_profile(geometry)
    assert profile.source_m3_s.tobytes() == values.tobytes()
    assert np.array_equal(profile.absorbed_m2_s, values * geometry.cell_measures_m)
    assert profile.binding_sha256 == fixed.source_reference and fixed.source_reference.startswith("sha256:")
    with pytest.raises(ValueError):
        FixedGenerationProfile(values, 14)
    with pytest.raises(ValueError):
        FixedGenerationProfile(np.array([1.0, np.nan]), 2)
    with pytest.raises(ValueError):
        fixed.generation_profile(integer_endpoint_geometry(np.linspace(0.0, 4e-7, 16)))
