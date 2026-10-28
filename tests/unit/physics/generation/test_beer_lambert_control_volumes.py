"""Analytic checks of the per-control-volume Beer-Lambert integration."""

from __future__ import annotations

import math

import numpy as np
import pytest

from solarlab.mesh.geometry import (
    half_endpoint_geometry,
    integer_endpoint_geometry,
    physical_boundary_two_sided_geometry,
    two_sided_trace_geometry,
)
from solarlab.mesh.grid import multilayer_grid, remove_shared_interface_nodes
from solarlab.physics.generation import (
    BeerLambertBinding,
    StaleGenerationProfileError,
    beer_lambert_absorbed_photon_flux,
    beer_lambert_generation,
    beer_lambert_optical_depth,
)

#: Floating-point round-off of a telescoped sum over <= ~1000 cells.
RTOL_FP = 1e-14
PHI = 2.5e21


def _layered(counts=(12, 40, 15)):
    thicknesses = (100e-9, 400e-9, 150e-9)
    alphas = (0.0, 1.3e7, 2.0e5)
    x = multilayer_grid(list(zip(thicknesses, counts)))
    edges = np.concatenate([[0.0], np.cumsum(thicknesses)])
    alpha = np.empty_like(x)
    for k, a in enumerate(alphas):
        # The material build assigns a shared boundary node to the right layer.
        alpha[(x >= edges[k] - 1e-15) & (x <= edges[k + 1] + 1e-15)] = a
    return x, alpha, thicknesses, alphas, edges


@pytest.mark.parametrize("n_nodes", [2, 3, 5, 10, 50, 200, 1000])
@pytest.mark.parametrize("builder", [integer_endpoint_geometry, half_endpoint_geometry])
def test_uniform_slab_absorbs_exactly_the_closed_form_budget(n_nodes, builder):
    L, alpha = 400e-9, 1.0e7
    geometry = builder(np.linspace(0.0, L, n_nodes))
    absorbed = beer_lambert_absorbed_photon_flux(geometry, alpha, PHI)
    expected = PHI * (1.0 - math.exp(-alpha * L))
    assert math.fsum(absorbed) == pytest.approx(expected, rel=RTOL_FP)
    generation = beer_lambert_generation(geometry, alpha, PHI)
    np.testing.assert_allclose(generation * geometry.cell_measures_m, absorbed, rtol=4e-16, atol=0.0)
    assert np.all(absorbed > 0.0)


def test_layered_budget_telescopes_to_the_discrete_optical_depth_on_any_mesh():
    for counts in ((3, 5, 4), (12, 40, 15), (60, 200, 80)):
        x, alpha, *_ = _layered(counts)
        geometry = integer_endpoint_geometry(x)
        absorbed = beer_lambert_absorbed_photon_flux(geometry, alpha, PHI)
        tau_total = beer_lambert_optical_depth(x, alpha)[-1]
        assert math.fsum(absorbed) == pytest.approx(PHI * -math.expm1(-tau_total), rel=RTOL_FP)
        assert math.fsum(absorbed) < PHI


def test_non_absorbing_front_layer_generates_exactly_zero():
    x, alpha, _, _, edges = _layered()
    generation = beer_lambert_generation(integer_endpoint_geometry(x), alpha, PHI)
    interior = x < edges[1] - 1e-12
    # Both faces of every cell inside the transparent layer carry tau = 0.
    assert np.all(generation[interior][:-1] == 0.0)


def test_layer_budgets_converge_to_the_analytic_layer_integrals():
    """The residual mesh error is the O(h) trapezoid across material steps."""

    errors = []
    for scale in (1, 2, 4, 8):
        x, alpha, thicknesses, alphas, edges = _layered(tuple(scale * c for c in (12, 40, 15)))
        absorbed = beer_lambert_absorbed_photon_flux(integer_endpoint_geometry(x), alpha, PHI)
        faces = integer_endpoint_geometry(x).cell_faces_m
        centre = 0.5 * (faces[:-1] + faces[1:])
        tau_front = np.concatenate([[0.0], np.cumsum(np.multiply(alphas, thicknesses))])
        analytic = [PHI * math.exp(-tau_front[k]) * -math.expm1(-alphas[k] * thicknesses[k]) for k in range(3)]
        numeric = [math.fsum(absorbed[(centre >= edges[k]) & (centre < edges[k + 1])]) for k in range(3)]
        errors.append(max(abs(a - b) for a, b in zip(analytic, numeric)))
    assert all(later < earlier for earlier, later in zip(errors, errors[1:]))


@pytest.mark.parametrize("builder", [two_sided_trace_geometry, physical_boundary_two_sided_geometry])
def test_two_sided_geometries_conserve_photons_with_faces_on_the_true_plane(builder):
    x, alpha, _, _, edges = _layered()
    positions = edges[1:-1]
    reduced = remove_shared_interface_nodes(x, positions)
    keep = np.isin(x, reduced)
    geometry = builder(reduced, positions)
    absorbed = beer_lambert_absorbed_photon_flux(geometry, alpha[keep], PHI)
    tau_total = beer_lambert_optical_depth(reduced, alpha[keep])[-1]
    assert math.fsum(absorbed) == pytest.approx(PHI * -math.expm1(-tau_total), rel=RTOL_FP)
    assert np.all(absorbed >= 0.0)
    assert np.all(beer_lambert_generation(geometry, alpha[keep], PHI) * geometry.cell_measures_m
                  == pytest.approx(absorbed, rel=4e-16))


@pytest.mark.parametrize(
    "alpha, phi",
    [(-1.0, PHI), (np.nan, PHI), (1e7, -1.0), (1e7, math.inf), (np.ones(3), PHI)],
)
def test_invalid_optical_inputs_are_rejected(alpha, phi):
    geometry = integer_endpoint_geometry(np.linspace(0.0, 1e-7, 5))
    with pytest.raises(ValueError):
        beer_lambert_absorbed_photon_flux(geometry, alpha, phi)


def test_binding_identity_and_stale_profile_rejection():
    x = np.linspace(0.0, 4e-7, 31)
    alpha = np.full(31, 1e7)
    binding = BeerLambertBinding(x, alpha, PHI)
    geometry = integer_endpoint_geometry(x)
    profile = binding.generation_profile(geometry)
    profile.require_inputs(binding.sha256, geometry)
    assert BeerLambertBinding(x.copy(), alpha.copy(), PHI).sha256 == binding.sha256
    changed = [BeerLambertBinding(x, alpha * 1.01, PHI), BeerLambertBinding(x, alpha, PHI * 0.5),
               BeerLambertBinding(x * 1.001, alpha, PHI)]
    assert len({binding.sha256, *(item.sha256 for item in changed)}) == 4
    with pytest.raises(StaleGenerationProfileError):
        profile.require_inputs(changed[0].sha256, geometry)
    with pytest.raises(StaleGenerationProfileError):
        profile.require_inputs(binding.sha256, half_endpoint_geometry(x))
    with pytest.raises(StaleGenerationProfileError):
        binding.generation_profile(integer_endpoint_geometry(x * 1.001))
    assert profile.geometry_name == "legacy_mol_integer_endpoint_v1"
    assert profile.areal_generation_m2_s == pytest.approx(PHI * -math.expm1(-1e7 * 4e-7), rel=RTOL_FP)
