"""Unit tests for the continuous bandgap grading pure-function module."""
import numpy as np
import pytest

from perovskite_sim.physics.grading import (
    has_grading_params,
    grading_coordinate,
    band_gap_profile,
    affinity_profile,
    grade_ni_sq,
    grade_n1_p1,
)
from perovskite_sim.models.parameters import MaterialParams


def _bare_params(**kw) -> MaterialParams:
    base = dict(
        eps_r=10.0, mu_n=1e-3, mu_p=1e-3, D_ion=0.0, P_lim=1e24, P0=1e24,
        ni=1e10, tau_n=1e-6, tau_p=1e-6, n1=1e10, p1=1e10, B_rad=0.0,
        C_n=0.0, C_p=0.0, alpha=0.0, N_A=0.0, N_D=0.0, chi=4.0, Eg=1.5,
    )
    base.update(kw)
    return MaterialParams(**base)


# --- has_grading_params -----------------------------------------------------

def test_has_grading_params_false_for_uniform():
    assert has_grading_params(_bare_params()) is False


def test_has_grading_params_true_with_eg_back():
    assert has_grading_params(_bare_params(Eg_back=1.7)) is True


def test_has_grading_params_true_with_chi_back():
    assert has_grading_params(_bare_params(chi_back=3.8)) is True


def test_has_grading_params_none_layer():
    assert has_grading_params(None) is False


# --- grading_coordinate y(x) ------------------------------------------------

def test_linear_coordinate_endpoints():
    x = np.array([0.0, 0.5, 1.0])
    y = grading_coordinate(x, 1.0, "linear")
    assert y[0] == 0.0 and y[-1] == 1.0
    assert np.isclose(y[1], 0.5)


def test_parabolic_midpoint():
    y = grading_coordinate(np.array([0.5]), 1.0, "parabolic")
    assert np.isclose(y[0], 0.25)


def test_exponential_monotone_and_endpoints():
    x = np.linspace(0.0, 1.0, 11)
    y = grading_coordinate(x, 1.0, "exponential", char_length=0.2)
    assert y[0] == pytest.approx(0.0, abs=1e-12)
    assert y[-1] == pytest.approx(1.0, abs=1e-12)
    assert np.all(np.diff(y) >= -1e-12)  # monotone non-decreasing


def test_exponential_degenerate_L_falls_back_to_linear():
    x = np.array([0.0, 0.5, 1.0])
    y = grading_coordinate(x, 1.0, "exponential", char_length=None)
    assert np.allclose(y, [0.0, 0.5, 1.0])


def test_direction_flip():
    x = np.array([0.0, 0.25, 1.0])
    fwd = grading_coordinate(x, 1.0, "linear", direction="front_to_back")
    rev = grading_coordinate(x, 1.0, "linear", direction="back_to_front")
    assert np.allclose(rev, 1.0 - fwd)


# --- band_gap_profile / affinity_profile ------------------------------------

def test_band_gap_endpoints_and_bowing_midpoint():
    y = np.array([0.0, 0.5, 1.0])
    Eg = band_gap_profile(y, 1.0, 1.5, bowing=0.4)
    assert np.isclose(Eg[0], 1.0)
    assert np.isclose(Eg[-1], 1.5)
    # midpoint = 0.5*1.0 + 0.5*1.5 - 0.4*0.25 = 1.25 - 0.1
    assert np.isclose(Eg[1], 1.25 - 0.4 * 0.25)


def test_band_gap_flat_is_constant_front_byte_identical():
    y = np.array([0.0, 0.123, 0.5, 0.777, 1.0])
    Eg = band_gap_profile(y, 1.5, 1.5, bowing=0.0)
    assert np.array_equal(Eg, np.full_like(y, 1.5))


def test_affinity_endpoints():
    y = np.array([0.0, 1.0])
    chi = affinity_profile(y, 4.0, 3.8)
    assert np.isclose(chi[0], 4.0) and np.isclose(chi[-1], 3.8)


def test_affinity_flat_is_constant_front_byte_identical():
    y = np.array([0.0, 0.3, 0.6, 1.0])
    chi = affinity_profile(y, 4.0, 4.0)
    assert np.array_equal(chi, np.full_like(y, 4.0))


# --- grade_ni_sq / grade_n1_p1 ----------------------------------------------

V_T = 0.025852  # ~kT/q at 300 K


def test_grade_ni_sq_identity_at_front():
    Eg_node = np.array([1.5, 1.5, 1.5])
    out = grade_ni_sq(1e20, Eg_node, 1.5, V_T)
    assert np.array_equal(out, np.full_like(Eg_node, 1e20))


def test_grade_ni_sq_monotone_decreasing_with_gap():
    Eg_node = np.array([1.5, 1.6, 1.7])
    out = grade_ni_sq(1e20, Eg_node, 1.5, V_T)
    assert np.all(np.diff(out) < 0.0)  # wider gap -> smaller ni²


def test_grade_n1_p1_preserves_detailed_balance():
    """n1(x)·p1(x) must equal ni²(x) per node (front-anchored)."""
    Eg_node = np.array([1.5, 1.62, 1.71])
    Eg_front = 1.5
    ni_front_sq = 1e20
    # front n1·p1 == ni_front² (self-consistent input)
    n1_front = 3e10
    p1_front = ni_front_sq / n1_front
    n1, p1 = grade_n1_p1(n1_front, p1_front, Eg_node, Eg_front, V_T)
    ni_sq = grade_ni_sq(ni_front_sq, Eg_node, Eg_front, V_T)
    assert np.allclose(n1 * p1, ni_sq, rtol=1e-12)


def test_grade_n1_p1_front_values_at_flat():
    Eg_node = np.array([1.5, 1.5])
    n1, p1 = grade_n1_p1(3e10, 4e9, Eg_node, 1.5, V_T)
    assert np.array_equal(n1, np.full_like(Eg_node, 3e10))
    assert np.array_equal(p1, np.full_like(Eg_node, 4e9))
