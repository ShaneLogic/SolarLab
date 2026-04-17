"""Unit tests for the photon-recycling escape-probability model."""
from __future__ import annotations

import math

import numpy as np
import pytest

from perovskite_sim.physics.photon_recycling import (
    compute_p_esc,
    compute_p_esc_for_absorber,
    wavelength_at_gap,
)


# ─── wavelength_at_gap ──────────────────────────────────────────────────


def test_wavelength_at_gap_MAPbI3():
    # MAPbI3, Eg = 1.6 eV -> λ ≈ 774.9 nm
    lam = wavelength_at_gap(1.6)
    assert lam == pytest.approx(7.749e-7, rel=5e-3)


def test_wavelength_at_gap_rejects_nonpositive_Eg():
    with pytest.raises(ValueError):
        wavelength_at_gap(0.0)
    with pytest.raises(ValueError):
        wavelength_at_gap(-1.1)


# ─── compute_p_esc (α·d form) ───────────────────────────────────────────


def test_p_esc_weak_absorption_clamps_to_one():
    # α·d = 1e-6, n=2 -> formula gives 1/(16·1e-6) >> 1, clamp to 1
    assert compute_p_esc(alpha_gap=1.0, thickness=1e-6, n_at_gap=2.0) == 1.0


def test_p_esc_strong_absorption_goes_to_zero():
    # Very thick absorber -> P_esc essentially zero (well below the
    # 1/(4n²) solid-angle ceiling of ~0.04 for n=2.4).
    p = compute_p_esc(alpha_gap=1e7, thickness=1e-3, n_at_gap=2.4)
    assert 0.0 <= p < 1e-3
    # And the closed-form value matches
    expected = 1.0 / (4.0 * 2.4 ** 2 * 1e7 * 1e-3)
    assert p == pytest.approx(expected, rel=1e-12)


def test_p_esc_formula_matches_closed_form():
    # Pick α, d, n such that the raw value is in (0, 1): 1/(4·n²·α·d)
    alpha = 1e6
    d = 300e-9
    n = 2.4
    expected = 1.0 / (4.0 * n * n * alpha * d)
    assert 0.0 < expected < 1.0, "choose params so the formula is unclamped"
    p = compute_p_esc(alpha_gap=alpha, thickness=d, n_at_gap=n)
    assert p == pytest.approx(expected, rel=1e-12)


def test_p_esc_zero_or_negative_inputs_return_one():
    # Degenerate inputs default to "no recycling"
    assert compute_p_esc(0.0, 1e-7, 2.4) == 1.0
    assert compute_p_esc(1e6, 0.0, 2.4) == 1.0
    assert compute_p_esc(1e6, 1e-7, 0.0) == 1.0
    assert compute_p_esc(-1.0, 1e-7, 2.4) == 1.0


def test_p_esc_is_monotone_in_thickness():
    # Thicker absorbers reabsorb more -> smaller P_esc
    alpha = 5e5
    n = 2.4
    d1 = 200e-9
    d2 = 800e-9
    p1 = compute_p_esc(alpha, d1, n)
    p2 = compute_p_esc(alpha, d2, n)
    assert p1 > p2


# ─── compute_p_esc_for_absorber (integrated-A form) ─────────────────────


def test_p_esc_from_A_weak_absorbance_clamps_to_one():
    # Uniform A = 1e-3 across 400 nm → OD ≈ 4e-10, way below 1/(4n²) ≈ 0.04
    A = np.full((20, 50), 1e-3)
    wl = np.linspace(3e-7, 1e-6, 50)
    x_abs = np.linspace(0.0, 4e-7, 20)
    assert compute_p_esc_for_absorber(A, wl, x_abs, 1.6, 2.4) == 1.0


def test_p_esc_from_A_strong_absorbance_goes_small():
    # Very strong integrated absorbance drives P_esc → 0
    A = np.full((20, 50), 1e9)
    wl = np.linspace(3e-7, 1e-6, 50)
    x_abs = np.linspace(0.0, 4e-7, 20)
    p = compute_p_esc_for_absorber(A, wl, x_abs, 1.6, 2.4)
    assert 0.0 <= p < 1e-3


def test_p_esc_from_A_closed_form():
    # Constant A(x, λ_gap) = A0 → OD = A0·d_abs, P_esc = 1/(4n²·A0·d_abs).
    # Pick params so the result lands in the unclamped (0, 1) range.
    A0 = 1e6
    d_abs = 400e-9
    n = 2.4
    Eg = 1.6

    wl = np.linspace(3e-7, 1e-6, 50)
    A = np.full((30, 50), A0)
    x_abs = np.linspace(0.0, d_abs, 30)

    expected_OD = A0 * d_abs
    expected_p = 1.0 / (4.0 * n * n * expected_OD)
    assert 0.0 < expected_p < 1.0

    p = compute_p_esc_for_absorber(A, wl, x_abs, Eg, n)
    assert p == pytest.approx(expected_p, rel=1e-6)


def test_p_esc_from_A_validates_shapes():
    wl = np.linspace(3e-7, 1e-6, 50)
    x_abs = np.linspace(0.0, 4e-7, 20)
    with pytest.raises(ValueError):
        # Wrong row count
        compute_p_esc_for_absorber(np.ones((10, 50)), wl, x_abs, 1.6, 2.4)
    with pytest.raises(ValueError):
        # Wrong wavelength dim
        compute_p_esc_for_absorber(np.ones((20, 49)), wl, x_abs, 1.6, 2.4)
    with pytest.raises(ValueError):
        # 1D array
        compute_p_esc_for_absorber(np.ones(20), wl, x_abs, 1.6, 2.4)
