"""TMM stack plumbing for opt-in composition-graded CIGS optics."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.models.device import electrical_layers
from perovskite_sim.physics.cigs_optics import CIGSGradedOptics
from perovskite_sim.physics.optical_stack import (
    build_device_optical_stack,
    cigs_nk_at_electrical_gap_edge,
    has_wavelength_resolved_optics,
)


def _stack(*, active: bool, slices: int = 3, direction: str = "front_to_back"):
    base = load_device_from_yaml("configs/cigs_graded_notch.yaml")
    layers = list(base.layers)
    absorber_index = next(
        index for index, layer in enumerate(layers) if layer.role == "absorber"
    )
    absorber = layers[absorber_index]
    model = CIGSGradedOptics(
        ggi_front=0.20,
        ggi_back=0.55,
        cgi=0.90,
        slices=slices,
        kk_quadrature_order=48,
    )
    params = replace(
        absorber.params,
        grading_direction=direction,
        cigs_graded_optics=model,
    )
    layers[absorber_index] = replace(absorber, params=params)
    return replace(base, layers=tuple(layers), graded_optics=active)


def test_declared_model_is_inert_with_master_gate_off() -> None:
    stack = _stack(active=False)
    wavelengths = np.array([500.0, 800.0])
    optical = build_device_optical_stack(stack, wavelengths)
    assert not has_wavelength_resolved_optics(stack)
    assert len(optical.layers) == len(stack.layers)
    assert optical.graded_physical_layer_indices == ()

    absorber_index = next(
        index for index, layer in enumerate(stack.layers) if layer.role == "absorber"
    )
    tmm_index = optical.physical_layer_slices[absorber_index].start
    absorber = stack.layers[absorber_index]
    np.testing.assert_array_equal(
        optical.layers[tmm_index].n,
        np.full(2, np.sqrt(absorber.params.eps_r)),
    )


def test_active_model_expands_only_absorber_and_preserves_thickness() -> None:
    stack = _stack(active=True, slices=4)
    optical = build_device_optical_stack(stack, np.array([500.0, 800.0]))
    absorber_index = next(
        index for index, layer in enumerate(stack.layers) if layer.role == "absorber"
    )
    absorber_slice = optical.physical_layer_slices[absorber_index]
    assert has_wavelength_resolved_optics(stack)
    assert absorber_slice.stop - absorber_slice.start == 4
    assert len(optical.layers) == len(stack.layers) + 3
    assert optical.graded_physical_layer_indices == (absorber_index,)
    assert optical.boundaries_m[-1] == pytest.approx(
        sum(layer.thickness for layer in stack.layers), abs=1e-18
    )
    slice_thicknesses = [
        layer.d for layer in optical.layers[absorber_slice]
    ]
    np.testing.assert_allclose(
        slice_thicknesses,
        np.full(4, stack.layers[absorber_index].thickness / 4.0),
        atol=0.0,
        rtol=0.0,
    )


def test_reversing_shared_grade_reverses_optical_slice_spectra() -> None:
    wavelengths = np.array([500.0, 700.0, 900.0])
    forward_stack = _stack(active=True, slices=5, direction="front_to_back")
    reverse_stack = _stack(active=True, slices=5, direction="back_to_front")
    forward = build_device_optical_stack(forward_stack, wavelengths)
    reverse = build_device_optical_stack(reverse_stack, wavelengths)
    absorber_index = next(
        index
        for index, layer in enumerate(forward_stack.layers)
        if layer.role == "absorber"
    )
    forward_layers = forward.layers[forward.physical_layer_slices[absorber_index]]
    reverse_layers = reverse.layers[reverse.physical_layer_slices[absorber_index]]
    for left, right in zip(forward_layers, reversed(reverse_layers), strict=True):
        np.testing.assert_allclose(left.n, right.n, rtol=2e-14, atol=2e-14)
        np.testing.assert_allclose(left.k, right.k, rtol=2e-14, atol=2e-14)


def test_active_model_requires_electrical_grade_and_nonlegacy_mode() -> None:
    stack = _stack(active=True)
    with pytest.raises(ValueError, match="requires band_grading"):
        build_device_optical_stack(
            replace(stack, band_grading=False), np.array([600.0])
        )
    with pytest.raises(ValueError, match="legacy"):
        build_device_optical_stack(replace(stack, mode="legacy"), np.array([600.0]))


def test_device_stack_rejects_nonboolean_master_gate() -> None:
    with pytest.raises(ValueError, match="graded_optics must be boolean"):
        replace(_stack(active=False), graded_optics=1)


def test_photon_recycling_edge_uses_narrower_gap_composition() -> None:
    stack = _stack(active=True)
    absorber = next(layer for layer in electrical_layers(stack) if layer.role == "absorber")
    wavelength_m = 1239.8419843320026 / absorber.params.Eg * 1e-9
    n_edge, k_edge = cigs_nk_at_electrical_gap_edge(absorber, wavelength_m)
    assert n_edge > 1.0
    assert k_edge >= 0.0
    assert np.isfinite(n_edge)
    assert np.isfinite(k_edge)


def test_material_schema_rejects_ambiguous_tabulated_and_cigs_sources() -> None:
    stack = _stack(active=False)
    absorber = next(layer for layer in stack.layers if layer.role == "absorber")
    with pytest.raises(ValueError, match="cannot be combined"):
        replace(absorber.params, optical_material="MAPbI3")
