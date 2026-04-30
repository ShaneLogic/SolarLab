from __future__ import annotations
import numpy as np
import pytest

from perovskite_sim.twod.radiative_reabsorption_2d import recompute_g_with_rad_2d


def _setup(Ny=8, Nx=5, lateral=1e-6, n_const=1e16, p_const=1e16, B_rad=4e-17):
    x = np.linspace(0.0, lateral, Nx)
    y = np.linspace(0.0, 1e-6, Ny)
    n = np.full((Ny, Nx), n_const)
    p = np.full((Ny, Nx), p_const)
    B = np.full((Ny, Nx), B_rad)
    G_optical = np.zeros((Ny, Nx))
    return x, y, n, p, B, G_optical


def test_recompute_returns_new_array_with_same_shape():
    """Result has shape (Ny, Nx) and is a NEW array (caller's G_optical not mutated)."""
    x, y, n, p, B, G = _setup()
    G_in = G.copy()
    G_out = recompute_g_with_rad_2d(
        G_optical=G_in, n=n, p=p, B_rad=B, x=x, y=y,
        absorber_y_ranges=((2, 6),),
        absorber_p_esc=(0.5,),
        absorber_areas=(2.0e-7 * 1e-6,),   # thickness 200 nm × lateral 1 µm
    )
    assert G_out.shape == G_in.shape
    assert G_out is not G_in
    np.testing.assert_array_equal(G_in, np.zeros_like(G_in))   # original untouched


def test_recompute_no_absorbers_returns_g_optical_copy():
    """Empty tuples → returned G is a copy of G_optical (no augmentation)."""
    x, y, n, p, B, G = _setup()
    G_in = np.full_like(G, 1.5e25)
    G_out = recompute_g_with_rad_2d(
        G_optical=G_in, n=n, p=p, B_rad=B, x=x, y=y,
        absorber_y_ranges=(),
        absorber_p_esc=(),
        absorber_areas=(),
    )
    np.testing.assert_array_equal(G_out, G_in)
    assert G_out is not G_in


def test_recompute_p_esc_one_no_augmentation():
    """P_esc = 1.0 → no reabsorption (everything escapes) → G_out == G_optical."""
    x, y, n, p, B, G = _setup()
    G_in = np.full_like(G, 1.5e25)
    G_out = recompute_g_with_rad_2d(
        G_optical=G_in, n=n, p=p, B_rad=B, x=x, y=y,
        absorber_y_ranges=((2, 6),),
        absorber_p_esc=(1.0,),
        absorber_areas=(2.0e-7 * 1e-6,),
    )
    np.testing.assert_array_equal(G_out, G_in)


def test_recompute_uniform_state_lateral_extension_matches_1d():
    """Lateral-uniform n,p,B → G_rad must reduce to the 1D formula
    R_tot_1D · (1 − P_esc) / thickness, exactly. This catches a missing
    lateral_length factor in the area calculation."""
    Ny, Nx = 8, 5
    lateral = 1e-6                           # 1 µm
    thickness = 2.0e-7                       # 200 nm absorber span
    x = np.linspace(0.0, lateral, Nx)
    y = np.linspace(0.0, 1e-6, Ny)
    n_const = 1e22
    p_const = 1e22
    B_rad = 4e-17
    n = np.full((Ny, Nx), n_const)
    p = np.full((Ny, Nx), p_const)
    B = np.full((Ny, Nx), B_rad)
    G_in = np.zeros((Ny, Nx))
    p_esc = 0.05
    y_lo, y_hi = 2, 6                        # absorber rows 2..5
    # Force y[2..5] to span exactly thickness (200 nm) so the 1D analog is well-defined.
    y_abs = np.linspace(0.0, thickness, y_hi - y_lo)
    y[y_lo:y_hi] = y_abs
    area = thickness * lateral
    G_out = recompute_g_with_rad_2d(
        G_optical=G_in, n=n, p=p, B_rad=B, x=x, y=y,
        absorber_y_ranges=((y_lo, y_hi),),
        absorber_p_esc=(p_esc,),
        absorber_areas=(area,),
    )
    # Expected 1D analog: R_tot_1D = B·n·p · thickness; G_rad = R_tot_1D · (1−P_esc) / thickness
    #                            = B·n·p · (1 − P_esc)
    expected_g_rad = B_rad * n_const * p_const * (1.0 - p_esc)
    # All absorber cells get this value uniformly (additively, base was 0)
    np.testing.assert_allclose(G_out[y_lo:y_hi, :], expected_g_rad, rtol=1e-12)
    # Non-absorber rows untouched
    np.testing.assert_array_equal(G_out[:y_lo, :], 0.0)
    np.testing.assert_array_equal(G_out[y_hi:, :], 0.0)


def test_recompute_only_absorber_rows_augmented():
    """Non-absorber rows must remain bit-identical to G_optical."""
    Ny, Nx = 10, 4
    x = np.linspace(0.0, 1e-6, Nx)
    y = np.linspace(0.0, 1e-6, Ny)
    n = np.full((Ny, Nx), 1e22)
    p = np.full((Ny, Nx), 1e22)
    B = np.full((Ny, Nx), 4e-17)
    G_in = np.full((Ny, Nx), 7.0e25)         # non-zero baseline for visibility
    y_lo, y_hi = 4, 8
    G_out = recompute_g_with_rad_2d(
        G_optical=G_in, n=n, p=p, B_rad=B, x=x, y=y,
        absorber_y_ranges=((y_lo, y_hi),),
        absorber_p_esc=(0.5,),
        absorber_areas=(2e-7 * 1e-6,),
    )
    np.testing.assert_array_equal(G_out[:y_lo, :], G_in[:y_lo, :])
    np.testing.assert_array_equal(G_out[y_hi:, :], G_in[y_hi:, :])
    assert np.all(G_out[y_lo:y_hi, :] >= G_in[y_lo:y_hi, :])


def test_recompute_zero_n_p_returns_g_optical_copy():
    """When n·p = 0 inside the absorber, R_tot = 0 and no augmentation occurs."""
    x, y, _, _, B, G = _setup()
    Ny, Nx = G.shape
    n0 = np.zeros((Ny, Nx))                  # n = 0 everywhere
    p0 = np.full((Ny, Nx), 1e22)
    G_in = np.full_like(G, 1.5e25)
    G_out = recompute_g_with_rad_2d(
        G_optical=G_in, n=n0, p=p0, B_rad=B, x=x, y=y,
        absorber_y_ranges=((2, 6),),
        absorber_p_esc=(0.5,),
        absorber_areas=(2e-7 * 1e-6,),
    )
    np.testing.assert_array_equal(G_out, G_in)


def test_recompute_shape_mismatch_raises():
    """Wrong shape on G_optical raises ValueError with the field name."""
    x, y, n, p, B, _ = _setup()
    bad_G = np.zeros((n.shape[0] + 1, n.shape[1]))
    with pytest.raises(ValueError, match="G_optical"):
        recompute_g_with_rad_2d(
            G_optical=bad_G, n=n, p=p, B_rad=B, x=x, y=y,
            absorber_y_ranges=((2, 6),),
            absorber_p_esc=(0.5,),
            absorber_areas=(2e-7 * 1e-6,),
        )
