from __future__ import annotations
import numpy as np
import pytest

from perovskite_sim.twod.solver_2d import build_material_arrays_2d, MaterialArrays2D
from perovskite_sim.twod.microstructure import Microstructure
from perovskite_sim.twod.grid_2d import build_grid_2d
from perovskite_sim.discretization.grid import Layer
from perovskite_sim.models.config_loader import load_device_from_yaml


def _stack():
    return load_device_from_yaml("configs/nip_MAPbI3.yaml")


def _layers_for_stack(stack):
    from perovskite_sim.models.device import electrical_layers
    return [Layer(L.thickness, 10) for L in electrical_layers(stack)]


def test_material_arrays_2d_shapes():
    stack = _stack()
    layers = _layers_for_stack(stack)
    g = build_grid_2d(layers, lateral_length=500e-9, Nx=20, lateral_uniform=True)
    mat = build_material_arrays_2d(g, stack, Microstructure())
    assert mat.eps_r.shape == (g.Ny, g.Nx)
    assert mat.D_n.shape == (g.Ny, g.Nx)
    assert mat.D_p.shape == (g.Ny, g.Nx)
    assert mat.tau_n.shape == (g.Ny, g.Nx)
    assert mat.tau_p.shape == (g.Ny, g.Nx)
    assert mat.G_optical.shape == (g.Ny, g.Nx)
    assert mat.poisson_factor is not None


def test_material_arrays_2d_uniform_in_x():
    """With Microstructure() (no GBs), every per-node field is x-invariant."""
    stack = _stack()
    layers = _layers_for_stack(stack)
    g = build_grid_2d(layers, lateral_length=500e-9, Nx=20, lateral_uniform=True)
    mat = build_material_arrays_2d(g, stack, Microstructure())
    for arr_name in ("eps_r", "D_n", "D_p", "tau_n", "tau_p", "G_optical"):
        arr = getattr(mat, arr_name)
        assert np.allclose(arr, arr[:, [0]]), f"{arr_name} varies in x"


def test_assemble_rhs_2d_returns_finite_dydt():
    from perovskite_sim.twod.solver_2d import assemble_rhs_2d
    stack = _stack()
    layers = _layers_for_stack(stack)
    g = build_grid_2d(layers, lateral_length=500e-9, Nx=20, lateral_uniform=True)
    mat = build_material_arrays_2d(g, stack, Microstructure(), lateral_bc="periodic")
    n0 = float(mat.n_eq_left[0]) * np.ones((g.Ny, g.Nx))
    p0 = float(mat.p_eq_left[0]) * np.ones((g.Ny, g.Nx))
    y_state = np.concatenate([n0.flatten(), p0.flatten()])
    dydt = assemble_rhs_2d(0.0, y_state, mat, V_app=0.0)
    assert np.all(np.isfinite(dydt))
    assert dydt.shape == y_state.shape


def test_assemble_rhs_2d_lateral_invariance_at_uniform_state():
    """At a y-only-varying state with empty Microstructure, dy/dt is x-invariant.
    This is the crucial Stage-A invariant — without it the validation gate fails."""
    from perovskite_sim.twod.solver_2d import assemble_rhs_2d
    stack = _stack()
    layers = _layers_for_stack(stack)
    g = build_grid_2d(layers, lateral_length=500e-9, Nx=20, lateral_uniform=True)
    mat = build_material_arrays_2d(g, stack, Microstructure(), lateral_bc="periodic")
    # x-uniform initial state: n(y) and p(y) replicated along x
    n0_y = np.linspace(float(mat.n_eq_left[0]), float(mat.n_eq_right[0]), g.Ny)
    p0_y = np.linspace(float(mat.p_eq_left[0]), float(mat.p_eq_right[0]), g.Ny)
    n0 = np.broadcast_to(n0_y[:, None], (g.Ny, g.Nx)).copy()
    p0 = np.broadcast_to(p0_y[:, None], (g.Ny, g.Nx)).copy()
    y_state = np.concatenate([n0.flatten(), p0.flatten()])
    dydt = assemble_rhs_2d(0.0, y_state, mat, V_app=0.0)
    Nn = g.n_nodes
    dn = dydt[:Nn].reshape((g.Ny, g.Nx))
    dp = dydt[Nn:].reshape((g.Ny, g.Nx))
    rel_n = np.max(np.abs(dn - dn[:, [0]])) / max(1.0, np.max(np.abs(dn)))
    rel_p = np.max(np.abs(dp - dp[:, [0]])) / max(1.0, np.max(np.abs(dp)))
    assert rel_n < 1e-9, f"dn lateral variation = {rel_n:.2e}"
    assert rel_p < 1e-9, f"dp lateral variation = {rel_p:.2e}"


def test_run_transient_2d_short_settle_returns_finite_state():
    from perovskite_sim.twod.solver_2d import run_transient_2d
    stack = _stack()
    layers = _layers_for_stack(stack)
    g = build_grid_2d(layers, lateral_length=500e-9, Nx=10, lateral_uniform=True)
    mat = build_material_arrays_2d(g, stack, Microstructure(), lateral_bc="periodic")
    # Use ni (intrinsic carrier density) as initial state for stability.
    # This avoids the extreme numerical stiffness at equilibrium (n_eq ~ 1e-24).
    n0 = np.broadcast_to(mat.ni[0, :], (g.Ny, g.Nx)).copy()
    p0 = np.broadcast_to(mat.ni[0, :], (g.Ny, g.Nx)).copy()
    y0 = np.concatenate([n0.flatten(), p0.flatten()])
    y_end = run_transient_2d(y0, mat, V_app=0.0, t_end=1e-12, max_step=1e-13)
    assert np.all(np.isfinite(y_end))
    assert y_end.shape == y0.shape


def test_material_arrays_2d_default_no_selective_contacts():
    """Without S values on the stack, has_selective_contacts is False and S fields are 0."""
    stack = _stack()  # configs/nip_MAPbI3.yaml — no S values
    layers = _layers_for_stack(stack)
    g = build_grid_2d(layers, lateral_length=300e-9, Nx=4, lateral_uniform=True)
    mat = build_material_arrays_2d(g, stack, Microstructure())
    assert mat.has_selective_contacts is False
    assert mat.S_n_top == 0.0
    assert mat.S_p_top == 0.0
    assert mat.S_n_bot == 0.0
    assert mat.S_p_bot == 0.0


def test_material_arrays_2d_right_maps_to_bot():
    """DeviceStack.S_n_right must appear in mat.S_n_bot (bottom contact, ETL)."""
    from dataclasses import replace as dc_replace
    import pytest
    stack_with_s = dc_replace(_stack(), S_n_right=1e-2)
    layers = _layers_for_stack(stack_with_s)
    g = build_grid_2d(layers, lateral_length=300e-9, Nx=4, lateral_uniform=True)
    mat = build_material_arrays_2d(g, stack_with_s, Microstructure())
    assert mat.has_selective_contacts is True
    assert mat.S_n_bot == pytest.approx(1e-2)
    assert mat.S_n_top == 0.0
    assert mat.S_p_top == 0.0
    assert mat.S_p_bot == 0.0


def test_material_arrays_2d_left_maps_to_top():
    """DeviceStack.S_p_left must appear in mat.S_p_top (top contact, HTL)."""
    from dataclasses import replace as dc_replace
    import pytest
    stack_with_s = dc_replace(_stack(), S_p_left=5e3)
    layers = _layers_for_stack(stack_with_s)
    g = build_grid_2d(layers, lateral_length=300e-9, Nx=4, lateral_uniform=True)
    mat = build_material_arrays_2d(g, stack_with_s, Microstructure())
    assert mat.has_selective_contacts is True
    assert mat.S_p_top == pytest.approx(5e3)
    assert mat.S_n_top == 0.0
    assert mat.S_n_bot == 0.0
    assert mat.S_p_bot == 0.0


def test_material_arrays_2d_tau_field_with_singleGB():
    """End-to-end Stage-B: build_material_arrays_2d driven by a stack carrying
    a non-empty microstructure must produce a 2D τ field with reduced lifetime
    at the GB column on rows tagged ``layer_role="absorber"``, and the bulk
    lifetime everywhere else."""
    from perovskite_sim.twod.solver_2d import _layer_role_at_each_y
    stack = load_device_from_yaml("configs/twod/nip_MAPbI3_singleGB.yaml")
    layers = _layers_for_stack(stack)
    g = build_grid_2d(layers, lateral_length=500e-9, Nx=10, lateral_uniform=True)
    mat = build_material_arrays_2d(g, stack, stack.microstructure)

    assert mat.tau_n.shape == (g.Ny, g.Nx)
    assert mat.tau_p.shape == (g.Ny, g.Nx)

    i_gb = int(np.argmin(np.abs(g.x - 250e-9)))
    i_bulk = 0

    # Use the role-tag-per-y the build path itself uses. This is the ground
    # truth for which rows the GB band should touch.
    roles = _layer_role_at_each_y(g.y, stack)
    is_absorber = np.array([r == "absorber" for r in roles])
    is_other = ~is_absorber

    # Absorber rows in the GB column → τ_GB (5e-8 s).
    assert np.allclose(mat.tau_n[is_absorber, i_gb], 5e-8)
    assert np.allclose(mat.tau_p[is_absorber, i_gb], 5e-8)
    # Non-absorber rows in the GB column → unchanged from the bulk column
    # (the GB band only paints absorber rows).
    assert np.allclose(
        mat.tau_n[is_other, i_gb], mat.tau_n[is_other, i_bulk]
    )


def test_assemble_rhs_2d_dirichlet_boundary_rows_exactly_zero():
    """Backward-compat: without selective contacts, all four boundary rows of dydt are 0."""
    from perovskite_sim.twod.solver_2d import assemble_rhs_2d
    stack = _stack()  # no S values → Dirichlet
    layers = _layers_for_stack(stack)
    g = build_grid_2d(layers, lateral_length=300e-9, Nx=4, lateral_uniform=True)
    mat = build_material_arrays_2d(g, stack, Microstructure(), lateral_bc="periodic")
    n0 = float(mat.n_eq_left[0]) * np.ones((g.Ny, g.Nx))
    p0 = float(mat.p_eq_left[0]) * np.ones((g.Ny, g.Nx))
    y0 = np.concatenate([n0.flatten(), p0.flatten()])
    dydt = assemble_rhs_2d(0.0, y0, mat, V_app=0.0)
    Nn = g.n_nodes
    dn = dydt[:Nn].reshape((g.Ny, g.Nx))
    dp = dydt[Nn:].reshape((g.Ny, g.Nx))
    np.testing.assert_array_equal(dn[0, :],  0.0, err_msg="dn top row should be 0 (Dirichlet)")
    np.testing.assert_array_equal(dn[-1, :], 0.0, err_msg="dn bot row should be 0 (Dirichlet)")
    np.testing.assert_array_equal(dp[0, :],  0.0, err_msg="dp top row should be 0 (Dirichlet)")
    np.testing.assert_array_equal(dp[-1, :], 0.0, err_msg="dp bot row should be 0 (Dirichlet)")
