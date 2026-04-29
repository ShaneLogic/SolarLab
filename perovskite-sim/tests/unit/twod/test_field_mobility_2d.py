from __future__ import annotations
import numpy as np
import pytest

from perovskite_sim.twod.field_mobility_2d import (
    arith_mean_face_x, arith_mean_face_y, arith_mean_face_wrap,
    recompute_d_eff_2d, FieldMobilityDEff,
)


def test_arith_mean_face_x_shape_and_value():
    A = np.array([[1.0, 3.0, 5.0],
                  [2.0, 4.0, 6.0]])  # (2, 3)
    out = arith_mean_face_x(A)
    assert out.shape == (2, 2)
    np.testing.assert_array_equal(out, np.array([[2.0, 4.0], [3.0, 5.0]]))


def test_arith_mean_face_y_shape_and_value():
    A = np.array([[1.0, 2.0],
                  [3.0, 4.0],
                  [5.0, 6.0]])  # (3, 2)
    out = arith_mean_face_y(A)
    assert out.shape == (2, 2)
    np.testing.assert_array_equal(out, np.array([[2.0, 3.0], [4.0, 5.0]]))


def test_arith_mean_face_wrap_shape_and_value():
    A = np.array([[1.0, 2.0, 3.0],
                  [10.0, 20.0, 30.0]])  # (2, 3); wrap face avg col 0 and col -1
    out = arith_mean_face_wrap(A)
    assert out.shape == (2,)
    np.testing.assert_array_equal(out, np.array([2.0, 20.0]))


def test_arith_mean_face_x_uniform_identity():
    A = np.full((5, 4), 7.0)
    out = arith_mean_face_x(A)
    assert out.shape == (5, 3)
    np.testing.assert_array_equal(out, np.full((5, 3), 7.0))


def test_recompute_d_eff_einstein_roundtrip_zero_field_mobility_neumann():
    """When all field-mobility face params are zero (CT off, PF off), the recompute
    must return D_eff equal to harmonic-mean(D_node) on every face. This catches
    the bug where * V_T is omitted (D_eff would be off by 1/V_T) or applied twice
    (D_eff would be off by V_T)."""
    Ny, Nx = 5, 4
    x = np.linspace(0.0, 1e-6, Nx)
    y = np.linspace(0.0, 1e-6, Ny)
    D_n_const = 1.5e-3
    D_p_const = 7.0e-4
    D_n_node = np.full((Ny, Nx), D_n_const)
    D_p_node = np.full((Ny, Nx), D_p_const)
    V_T = 0.025852
    # Non-trivial phi to give non-zero E
    phi = np.linspace(0.0, 1.0, Ny)[:, None] * np.ones((Ny, Nx))
    # Zero field-mobility face arrays
    zero_x = np.zeros((Ny, Nx - 1))
    zero_y = np.zeros((Ny - 1, Nx))
    res = recompute_d_eff_2d(
        phi=phi, x=x, y=y,
        D_n=D_n_node, D_p=D_p_node, V_T=V_T,
        v_sat_n_x_face=zero_x, v_sat_n_y_face=zero_y,
        ct_beta_n_x_face=zero_x, ct_beta_n_y_face=zero_y,
        pf_gamma_n_x_face=zero_x, pf_gamma_n_y_face=zero_y,
        v_sat_p_x_face=zero_x, v_sat_p_y_face=zero_y,
        ct_beta_p_x_face=zero_x, ct_beta_p_y_face=zero_y,
        pf_gamma_p_x_face=zero_x, pf_gamma_p_y_face=zero_y,
        lateral_bc="neumann",
    )
    assert res.D_n_x.shape == (Ny, Nx - 1)
    assert res.D_n_y.shape == (Ny - 1, Nx)
    assert res.D_p_x.shape == (Ny, Nx - 1)
    assert res.D_p_y.shape == (Ny - 1, Nx)
    assert res.D_n_wrap is None and res.D_p_wrap is None
    # Harmonic mean of two equal values equals the value itself.
    np.testing.assert_allclose(res.D_n_x, D_n_const, rtol=1e-15)
    np.testing.assert_allclose(res.D_n_y, D_n_const, rtol=1e-15)
    np.testing.assert_allclose(res.D_p_x, D_p_const, rtol=1e-15)
    np.testing.assert_allclose(res.D_p_y, D_p_const, rtol=1e-15)


def test_recompute_d_eff_einstein_roundtrip_zero_field_mobility_periodic():
    """Same Einstein roundtrip but with lateral_bc='periodic' — the wrap face must
    also be returned and equal to harmonic-mean of D[:, -1] and D[:, 0]."""
    Ny, Nx = 4, 5
    x = np.linspace(0.0, 1e-6, Nx)
    y = np.linspace(0.0, 1e-6, Ny)
    D_n_node = np.full((Ny, Nx), 1.5e-3)
    D_p_node = np.full((Ny, Nx), 7.0e-4)
    V_T = 0.025852
    phi = np.linspace(0.0, 1.0, Ny)[:, None] * np.ones((Ny, Nx))
    zero_x = np.zeros((Ny, Nx - 1))
    zero_y = np.zeros((Ny - 1, Nx))
    zero_wrap = np.zeros((Ny,))
    res = recompute_d_eff_2d(
        phi=phi, x=x, y=y,
        D_n=D_n_node, D_p=D_p_node, V_T=V_T,
        v_sat_n_x_face=zero_x, v_sat_n_y_face=zero_y,
        ct_beta_n_x_face=zero_x, ct_beta_n_y_face=zero_y,
        pf_gamma_n_x_face=zero_x, pf_gamma_n_y_face=zero_y,
        v_sat_p_x_face=zero_x, v_sat_p_y_face=zero_y,
        ct_beta_p_x_face=zero_x, ct_beta_p_y_face=zero_y,
        pf_gamma_p_x_face=zero_x, pf_gamma_p_y_face=zero_y,
        lateral_bc="periodic",
        v_sat_n_wrap=zero_wrap, v_sat_p_wrap=zero_wrap,
        ct_beta_n_wrap=zero_wrap, ct_beta_p_wrap=zero_wrap,
        pf_gamma_n_wrap=zero_wrap, pf_gamma_p_wrap=zero_wrap,
    )
    assert res.D_n_wrap is not None
    assert res.D_p_wrap is not None
    assert res.D_n_wrap.shape == (Ny,)
    assert res.D_p_wrap.shape == (Ny,)
    np.testing.assert_allclose(res.D_n_wrap, 1.5e-3, rtol=1e-15)
    np.testing.assert_allclose(res.D_p_wrap, 7.0e-4, rtol=1e-15)


def test_recompute_d_eff_aggressive_ct_reduces_mobility():
    """At aggressive v_sat=1e2 m/s and a non-trivial E, D_eff must be strictly
    less than harmonic-mean(D_node) on every face — i.e., CT actually fires."""
    Ny, Nx = 5, 4
    x = np.linspace(0.0, 1e-6, Nx)
    y = np.linspace(0.0, 1e-6, Ny)
    D_n_node = np.full((Ny, Nx), 1.5e-3)
    D_p_node = np.full((Ny, Nx), 7.0e-4)
    V_T = 0.025852
    phi = np.linspace(0.0, 1.0, Ny)[:, None] * np.ones((Ny, Nx))  # non-zero E_y
    # Aggressive blocking
    v_sat_x = np.full((Ny, Nx - 1), 1e2)
    v_sat_y = np.full((Ny - 1, Nx), 1e2)
    beta_x  = np.full((Ny, Nx - 1), 2.0)
    beta_y  = np.full((Ny - 1, Nx), 2.0)
    zero_x  = np.zeros((Ny, Nx - 1))
    zero_y  = np.zeros((Ny - 1, Nx))
    res = recompute_d_eff_2d(
        phi=phi, x=x, y=y,
        D_n=D_n_node, D_p=D_p_node, V_T=V_T,
        v_sat_n_x_face=v_sat_x, v_sat_n_y_face=v_sat_y,
        ct_beta_n_x_face=beta_x, ct_beta_n_y_face=beta_y,
        pf_gamma_n_x_face=zero_x, pf_gamma_n_y_face=zero_y,
        v_sat_p_x_face=v_sat_x, v_sat_p_y_face=v_sat_y,
        ct_beta_p_x_face=beta_x, ct_beta_p_y_face=beta_y,
        pf_gamma_p_x_face=zero_x, pf_gamma_p_y_face=zero_y,
        lateral_bc="neumann",
    )
    # On y-faces (where E_y > 0), D_eff_n must be strictly less than D_n_node.
    assert np.all(res.D_n_y < 1.5e-3)
    assert np.all(res.D_p_y < 7.0e-4)
    # On x-faces, E_x = 0 (phi varies only in y), so CT should be inactive and D_eff = D_node.
    np.testing.assert_allclose(res.D_n_x, 1.5e-3, rtol=1e-12)
    np.testing.assert_allclose(res.D_p_x, 7.0e-4, rtol=1e-12)


def test_recompute_d_eff_sign_invariance():
    """Flipping the sign of phi must not change D_eff (apply_field_mobility uses |E|)."""
    Ny, Nx = 5, 4
    x = np.linspace(0.0, 1e-6, Nx)
    y = np.linspace(0.0, 1e-6, Ny)
    D_n_node = np.full((Ny, Nx), 1.5e-3)
    D_p_node = np.full((Ny, Nx), 7.0e-4)
    V_T = 0.025852
    phi_pos = np.linspace(0.0, 1.0, Ny)[:, None] * np.ones((Ny, Nx))
    phi_neg = -phi_pos
    v_sat_x = np.full((Ny, Nx - 1), 1e2)
    v_sat_y = np.full((Ny - 1, Nx), 1e2)
    beta_x  = np.full((Ny, Nx - 1), 2.0)
    beta_y  = np.full((Ny - 1, Nx), 2.0)
    zero_x  = np.zeros((Ny, Nx - 1))
    zero_y  = np.zeros((Ny - 1, Nx))
    common = dict(
        x=x, y=y, D_n=D_n_node, D_p=D_p_node, V_T=V_T,
        v_sat_n_x_face=v_sat_x, v_sat_n_y_face=v_sat_y,
        ct_beta_n_x_face=beta_x, ct_beta_n_y_face=beta_y,
        pf_gamma_n_x_face=zero_x, pf_gamma_n_y_face=zero_y,
        v_sat_p_x_face=v_sat_x, v_sat_p_y_face=v_sat_y,
        ct_beta_p_x_face=beta_x, ct_beta_p_y_face=beta_y,
        pf_gamma_p_x_face=zero_x, pf_gamma_p_y_face=zero_y,
        lateral_bc="neumann",
    )
    res_pos = recompute_d_eff_2d(phi=phi_pos, **common)
    res_neg = recompute_d_eff_2d(phi=phi_neg, **common)
    np.testing.assert_array_equal(res_pos.D_n_y, res_neg.D_n_y)
    np.testing.assert_array_equal(res_pos.D_p_y, res_neg.D_p_y)


def test_recompute_d_eff_shape_mismatch_raises():
    """Wrong shape on any face param must raise a clear ValueError, not silently broadcast."""
    Ny, Nx = 4, 3
    x = np.linspace(0.0, 1e-6, Nx)
    y = np.linspace(0.0, 1e-6, Ny)
    D_n_node = np.full((Ny, Nx), 1.5e-3)
    D_p_node = np.full((Ny, Nx), 7.0e-4)
    V_T = 0.025852
    phi = np.zeros((Ny, Nx))
    bad_x = np.zeros((Ny, Nx))      # wrong: should be (Ny, Nx-1)
    zero_y = np.zeros((Ny - 1, Nx))
    with pytest.raises(ValueError, match="v_sat_n_x_face"):
        recompute_d_eff_2d(
            phi=phi, x=x, y=y,
            D_n=D_n_node, D_p=D_p_node, V_T=V_T,
            v_sat_n_x_face=bad_x, v_sat_n_y_face=zero_y,
            ct_beta_n_x_face=np.zeros((Ny, Nx - 1)), ct_beta_n_y_face=zero_y,
            pf_gamma_n_x_face=np.zeros((Ny, Nx - 1)), pf_gamma_n_y_face=zero_y,
            v_sat_p_x_face=np.zeros((Ny, Nx - 1)), v_sat_p_y_face=zero_y,
            ct_beta_p_x_face=np.zeros((Ny, Nx - 1)), ct_beta_p_y_face=zero_y,
            pf_gamma_p_x_face=np.zeros((Ny, Nx - 1)), pf_gamma_p_y_face=zero_y,
            lateral_bc="neumann",
        )
