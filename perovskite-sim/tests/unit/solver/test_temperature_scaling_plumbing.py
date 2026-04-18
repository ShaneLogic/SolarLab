"""Phase 4b: temperature scaling of B_rad and Eg propagates through
``build_material_arrays``.

These are fast plumbing tests — no J-V sweep — that confirm the new
``B_rad_at_T`` and ``eg_at_T`` calls are actually wired in under the
``use_temperature_scaling`` flag, and that configs which do not opt into
the new fields (``B_rad_T_gamma = 0``, ``varshni_alpha = 0``, the
shipped defaults) stay bit-identical to the Phase 4a path.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.solver.mol import build_material_arrays


@pytest.fixture
def stack():
    return load_device_from_yaml("configs/ionmonger_benchmark.yaml")


@pytest.fixture
def x(stack):
    return np.linspace(0.0, stack.total_thickness, 60)


class TestBRadTemperatureScaling:
    def test_default_gamma_leaves_brad_unchanged_at_elevated_T(self, stack, x):
        """B_rad_T_gamma defaults to 0 — hot stack should match cold stack."""
        stack_hot = dataclasses.replace(stack, T=350.0)
        mat_hot = build_material_arrays(x, stack_hot)
        mat_cold = build_material_arrays(x, stack)
        np.testing.assert_allclose(mat_hot.B_rad, mat_cold.B_rad, rtol=1e-12)

    def test_gamma_minus_1p5_suppresses_brad_at_hot(self, stack, x):
        """Setting gamma=-1.5 on the absorber reduces B_rad at T=400 K.

        The ionmonger benchmark uses ``B_rad = 0`` on the absorber, so
        we inject a nonzero value here to verify the scaling actually
        runs.
        """
        absorber = stack.layers[1]
        B_300 = 4.6e-17  # detailed-balance literature value for MAPbI3
        p_new = dataclasses.replace(
            absorber.params, B_rad=B_300, B_rad_T_gamma=-1.5,
        )
        layers = list(stack.layers)
        layers[1] = dataclasses.replace(absorber, params=p_new)
        stack_new = dataclasses.replace(stack, layers=tuple(layers), T=400.0)

        mat = build_material_arrays(x, stack_new)
        # Absorber occupies ~200–600 nm in the ionmonger benchmark;
        # check one node safely inside.
        idx = np.argmin(np.abs(x - 0.4 * stack_new.total_thickness))
        expected = B_300 * (400.0 / 300.0) ** -1.5
        assert mat.B_rad[idx] == pytest.approx(expected, rel=1e-10)
        assert mat.B_rad[idx] < B_300

    def test_legacy_mode_skips_brad_scaling_even_if_gamma_set(self, stack, x):
        """use_temperature_scaling=off must freeze B_rad at its 300 K value."""
        absorber = stack.layers[1]
        B_300 = 4.6e-17
        p_new = dataclasses.replace(
            absorber.params, B_rad=B_300, B_rad_T_gamma=-1.5,
        )
        layers = list(stack.layers)
        layers[1] = dataclasses.replace(absorber, params=p_new)
        stack_new = dataclasses.replace(
            stack, layers=tuple(layers), T=400.0, mode="legacy",
        )

        mat = build_material_arrays(x, stack_new)
        idx = np.argmin(np.abs(x - 0.4 * stack_new.total_thickness))
        assert mat.B_rad[idx] == pytest.approx(B_300, rel=1e-12)


class TestVarshniBandgapScaling:
    def test_default_alpha_leaves_eg_unchanged_at_elevated_T(self, stack, x):
        """varshni_alpha defaults to 0 — Eg array is T-independent by default."""
        stack_hot = dataclasses.replace(stack, T=350.0)
        mat_hot = build_material_arrays(x, stack_hot)
        mat_cold = build_material_arrays(x, stack)
        np.testing.assert_allclose(mat_hot.Eg, mat_cold.Eg, rtol=1e-12)

    def test_positive_alpha_narrows_eg_at_hot(self, stack, x):
        """Silicon-style Varshni: α>0 narrows Eg as T rises."""
        absorber = stack.layers[1]
        # Pretend the absorber has an Eg and a Varshni shift. We do NOT
        # need Eg > 0 in ionmonger_benchmark — we inject one here to
        # exercise the plumbing.
        p_new = dataclasses.replace(
            absorber.params,
            Eg=1.6,
            varshni_alpha=4.7e-4,
            varshni_beta=636.0,
        )
        layers = list(stack.layers)
        layers[1] = dataclasses.replace(absorber, params=p_new)
        stack_new = dataclasses.replace(stack, layers=tuple(layers), T=400.0)

        mat = build_material_arrays(x, stack_new)
        idx = np.argmin(np.abs(x - 0.4 * stack_new.total_thickness))
        # Absorber node should see a narrowed Eg (< 1.6 eV).
        assert mat.Eg[idx] < 1.6
        # And ni_sq should rise because ni² ∝ exp(−Eg/kT).
        mat_cold = build_material_arrays(x, stack)
        assert mat.ni_sq[idx] > mat_cold.ni_sq[idx]

    def test_legacy_mode_freezes_eg_at_reference(self, stack, x):
        """Legacy mode must not apply Varshni even if alpha is set."""
        absorber = stack.layers[1]
        p_new = dataclasses.replace(
            absorber.params,
            Eg=1.6,
            varshni_alpha=4.7e-4,
            varshni_beta=636.0,
        )
        layers = list(stack.layers)
        layers[1] = dataclasses.replace(absorber, params=p_new)
        stack_new = dataclasses.replace(
            stack, layers=tuple(layers), T=400.0, mode="legacy",
        )

        mat = build_material_arrays(x, stack_new)
        idx = np.argmin(np.abs(x - 0.4 * stack_new.total_thickness))
        assert mat.Eg[idx] == pytest.approx(1.6, rel=1e-12)


class TestDefaultsArePhase4aIdentical:
    """Regression: shipped configs give bit-identical output pre/post 4b."""

    def test_brad_and_eg_arrays_are_unchanged_on_ionmonger(self, stack, x):
        # Pure Phase 4a: every MaterialParams uses B_rad_T_gamma=0 and
        # varshni_alpha=0 (the defaults). The computed arrays should be
        # bit-identical to what an unscaled build would produce.
        mat = build_material_arrays(x, stack)
        # Reconstruct the naive per-node B_rad without scaling.
        total = 0.0
        for layer in stack.layers:
            mask = (
                (x >= total - 1e-12)
                & (x <= total + layer.thickness + 1e-12)
            )
            np.testing.assert_allclose(
                mat.B_rad[mask], layer.params.B_rad, rtol=1e-14,
            )
            np.testing.assert_allclose(
                mat.Eg[mask], layer.params.Eg, rtol=1e-14,
            )
            total += layer.thickness
