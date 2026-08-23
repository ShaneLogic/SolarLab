"""Physical/numerical gates for the opt-in graded CIGS optical path."""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from perovskite_sim._compat.numpy_compat import trapezoid
from perovskite_sim.data import load_am15g
from perovskite_sim.discretization.grid import Layer, multilayer_grid
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.models.device import electrical_layers
from perovskite_sim.physics.cigs_optics import (
    HC_EV_NM,
    carron_absorption_coefficient,
    minoura_nk,
)
from perovskite_sim.physics.generation import dual_cell_widths
from perovskite_sim.physics.optical_stack import build_device_optical_stack
from perovskite_sim.physics.optics import tmm_reflectance
from perovskite_sim.solver.mol import _compute_tmm_generation, build_material_arrays


def _with_optical_resolution(stack, *, slices: int, quadrature_order: int):
    layers = list(stack.layers)
    absorber_index = next(
        index for index, layer in enumerate(layers) if layer.role == "absorber"
    )
    absorber = layers[absorber_index]
    model = replace(
        absorber.params.cigs_graded_optics,
        slices=slices,
        kk_quadrature_order=quadrature_order,
    )
    layers[absorber_index] = replace(
        absorber,
        params=replace(absorber.params, cigs_graded_optics=model),
    )
    return replace(stack, layers=tuple(layers))


def _electrical_grid(stack, intervals_per_layer: int = 20) -> np.ndarray:
    return multilayer_grid(
        [
            Layer(layer.thickness, intervals_per_layer)
            for layer in electrical_layers(stack)
        ]
    )


def test_shipped_research_config_builds_finite_photon_bounded_generation() -> None:
    stack = load_device_from_yaml("configs/cigs_graded_optics.yaml")
    assert stack.band_grading
    assert stack.graded_optics
    absorber = next(layer for layer in stack.layers if layer.role == "absorber")
    assert absorber.params.cigs_graded_optics.slices == 25
    assert absorber.params.cigs_graded_optics.kk_quadrature_order == 192

    x = _electrical_grid(stack, intervals_per_layer=12)
    material = build_material_arrays(x, stack)
    assert material.G_optical is not None
    assert np.all(np.isfinite(material.G_optical))
    assert np.all(material.G_optical >= 0.0)

    absorbed_flux = float(
        np.sum(material.G_optical * dual_cell_widths(x))
    )
    wavelengths_nm = np.linspace(300.0, 1000.0, 200)
    _, incident_spectrum = load_am15g(wavelengths_nm)
    incident_flux = float(
        trapezoid(incident_spectrum, wavelengths_nm * 1e-9)
    )
    assert 0.0 < absorbed_flux <= incident_flux * (1.0 + 2e-12)


def test_optical_slice_refinement_converges_generation_and_budget() -> None:
    base = load_device_from_yaml("configs/cigs_graded_optics.yaml")
    x = _electrical_grid(base, intervals_per_layer=20)
    weights = dual_cell_widths(x)
    results = []
    for slices in (8, 16, 32):
        stack = _with_optical_resolution(
            base, slices=slices, quadrature_order=192
        )
        generation = _compute_tmm_generation(x, stack, n_wavelengths=100)
        results.append(
            (generation, float(np.sum(generation * weights)))
        )

    coarse_change = float(
        np.sum(np.abs(results[0][0] - results[1][0]) * weights)
        / np.sum(np.abs(results[1][0]) * weights)
    )
    fine_change = float(
        np.sum(np.abs(results[1][0] - results[2][0]) * weights)
        / np.sum(np.abs(results[2][0]) * weights)
    )
    budget_change = abs(results[1][1] - results[2][1]) / results[2][1]
    assert fine_change < coarse_change
    assert fine_change < 5e-3
    assert budget_change < 5e-3


def test_kk_parameter_quadrature_converges_generation() -> None:
    base = load_device_from_yaml("configs/cigs_graded_optics.yaml")
    x = _electrical_grid(base, intervals_per_layer=16)
    weights = dual_cell_widths(x)
    generation = []
    for order in (96, 192, 384):
        stack = _with_optical_resolution(
            base, slices=16, quadrature_order=order
        )
        generation.append(
            _compute_tmm_generation(x, stack, n_wavelengths=80)
        )
    coarse_change = float(
        np.sum(np.abs(generation[0] - generation[1]) * weights)
        / np.sum(np.abs(generation[1]) * weights)
    )
    fine_change = float(
        np.sum(np.abs(generation[1] - generation[2]) * weights)
        / np.sum(np.abs(generation[2]) * weights)
    )
    assert fine_change < coarse_change
    assert fine_change < 5e-3


def test_uniform_composition_slice_limit_matches_one_homogeneous_layer() -> None:
    base = load_device_from_yaml("configs/cigs_graded_optics.yaml")
    layers = list(base.layers)
    absorber_index = next(
        index for index, layer in enumerate(layers) if layer.role == "absorber"
    )
    absorber = layers[absorber_index]
    flat_model = replace(
        absorber.params.cigs_graded_optics,
        ggi_back=absorber.params.cigs_graded_optics.ggi_front,
    )
    layers[absorber_index] = replace(
        absorber,
        params=replace(absorber.params, cigs_graded_optics=flat_model),
    )
    flat = replace(base, layers=tuple(layers))
    wavelengths_nm = np.linspace(350.0, 1000.0, 61)
    wavelengths_m = wavelengths_nm * 1e-9
    one = build_device_optical_stack(
        _with_optical_resolution(flat, slices=1, quadrature_order=192),
        wavelengths_nm,
    )
    many = build_device_optical_stack(
        _with_optical_resolution(flat, slices=16, quadrature_order=192),
        wavelengths_nm,
    )
    np.testing.assert_allclose(
        tmm_reflectance(one.layers, wavelengths_m),
        tmm_reflectance(many.layers, wavelengths_m),
        rtol=2e-11,
        atol=2e-12,
    )


def test_minoura_high_energy_absorption_agrees_with_independent_carron_model() -> None:
    # Carron explicitly reports larger alpha very close to Eg.  The independent
    # comparison is therefore pre-registered over 1.35-2.5 eV, where their
    # Figure 8 shows the two models should track one another.
    energy = np.linspace(1.35, 2.5, 151)
    wavelengths_nm = HC_EV_NM / energy
    _, k = minoura_nk(
        wavelengths_nm, 0.225, 0.90, quadrature_order=384
    )
    alpha_minoura = 4.0 * np.pi * k / (wavelengths_nm * 1e-9)
    alpha_carron = carron_absorption_coefficient(energy, 0.225, 0.90)
    ratio = alpha_minoura / alpha_carron
    assert np.min(ratio) > 0.75
    assert np.max(ratio) < 1.15
    assert np.median(np.abs(ratio - 1.0)) < 0.05
