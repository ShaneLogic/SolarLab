from __future__ import annotations
from dataclasses import replace as dc_replace

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


def _make_grid_and_mat(stack, Nx=4):
    layers = _layers_for_stack(stack)
    g = build_grid_2d(layers, lateral_length=300e-9, Nx=Nx, lateral_uniform=True)
    mat = build_material_arrays_2d(g, stack, Microstructure(), lateral_bc="periodic")
    return g, mat


def test_robin_dp_top_decreases_with_excess_holes():
    """dp[0,:] must be smaller under Robin (S_p_top>0, p>p_eq) than under pure Neumann."""
    from perovskite_sim.twod.solver_2d import assemble_rhs_2d
    stack_base = _stack()
    # Neumann baseline: S_p_left=0.0 triggers Robin mode but contributes zero correction.
    stack_neumann = dc_replace(stack_base, S_p_left=0.0)
    stack_robin   = dc_replace(stack_base, S_p_left=1e3)
    g, mat_neumann = _make_grid_and_mat(stack_neumann)
    _, mat_robin   = _make_grid_and_mat(stack_robin)
    # State: p[0,:] = 2 × p_eq (excess holes at top boundary)
    n0 = float(mat_neumann.n_eq_left[0]) * np.ones((g.Ny, g.Nx))
    p0 = float(mat_neumann.p_eq_left[0]) * np.ones((g.Ny, g.Nx))
    p0[0, :] = 2.0 * float(mat_neumann.p_eq_left[0])
    y0 = np.concatenate([n0.flatten(), p0.flatten()])
    Nn = g.n_nodes
    dydt_n = assemble_rhs_2d(0.0, y0, mat_neumann, V_app=0.0)
    dydt_r = assemble_rhs_2d(0.0, y0, mat_robin,   V_app=0.0)
    dp_neumann = dydt_n[Nn:].reshape(g.Ny, g.Nx)
    dp_robin   = dydt_r[Nn:].reshape(g.Ny, g.Nx)
    # Robin removes excess holes → dp[0,:] must decrease
    assert np.all(dp_robin[0, :] < dp_neumann[0, :]), (
        "dp[0,:] should decrease under Robin when p > p_eq (wrong sign or no correction)"
    )


def test_robin_dp_bot_decreases_with_excess_holes():
    """dp[-1,:] must be smaller under Robin (S_p_bot>0, p>p_eq) than pure Neumann."""
    from perovskite_sim.twod.solver_2d import assemble_rhs_2d
    stack_base = _stack()
    stack_neumann = dc_replace(stack_base, S_p_right=0.0)
    stack_robin   = dc_replace(stack_base, S_p_right=1e3)
    g, mat_neumann = _make_grid_and_mat(stack_neumann)
    _, mat_robin   = _make_grid_and_mat(stack_robin)
    n0 = float(mat_neumann.n_eq_right[0]) * np.ones((g.Ny, g.Nx))
    p0 = float(mat_neumann.p_eq_right[0]) * np.ones((g.Ny, g.Nx))
    p0[-1, :] = 2.0 * float(mat_neumann.p_eq_right[0])
    y0 = np.concatenate([n0.flatten(), p0.flatten()])
    Nn = g.n_nodes
    dydt_n = assemble_rhs_2d(0.0, y0, mat_neumann, V_app=0.0)
    dydt_r = assemble_rhs_2d(0.0, y0, mat_robin,   V_app=0.0)
    dp_neumann = dydt_n[Nn:].reshape(g.Ny, g.Nx)
    dp_robin   = dydt_r[Nn:].reshape(g.Ny, g.Nx)
    assert np.all(dp_robin[-1, :] < dp_neumann[-1, :]), (
        "dp[-1,:] should decrease under Robin when p > p_eq at bottom"
    )


def test_robin_correction_routes_to_correct_boundary():
    """S_n_right correction appears on dn[-1,:] not dn[0,:]; top row is unaffected."""
    from perovskite_sim.twod.solver_2d import assemble_rhs_2d
    stack_base = _stack()
    # Only S_n_right set (bottom, ETL). Top correction should be zero.
    stack_neumann = dc_replace(stack_base, S_n_right=0.0)
    stack_robin   = dc_replace(stack_base, S_n_right=1e3)
    g, mat_neumann = _make_grid_and_mat(stack_neumann)
    _, mat_robin   = _make_grid_and_mat(stack_robin)
    # State: n[-1,:] = 2 × n_eq_right (excess electrons at bottom boundary)
    n0 = float(mat_neumann.n_eq_right[0]) * np.ones((g.Ny, g.Nx))
    p0 = float(mat_neumann.p_eq_right[0]) * np.ones((g.Ny, g.Nx))
    n0[-1, :] = 2.0 * float(mat_neumann.n_eq_right[0])
    y0 = np.concatenate([n0.flatten(), p0.flatten()])
    Nn = g.n_nodes
    dydt_n = assemble_rhs_2d(0.0, y0, mat_neumann, V_app=0.0)
    dydt_r = assemble_rhs_2d(0.0, y0, mat_robin,   V_app=0.0)
    dn_neumann = dydt_n[:Nn].reshape(g.Ny, g.Nx)
    dn_robin   = dydt_r[:Nn].reshape(g.Ny, g.Nx)
    # Bottom row: Robin removes excess electrons → dn[-1,:] decreases
    assert np.all(dn_robin[-1, :] < dn_neumann[-1, :]), (
        "dn[-1,:] should decrease under Robin when n > n_eq at bottom (mapping swap?)"
    )
    # Top row: no S_n_top → correction = 0 → top rows identical
    np.testing.assert_array_equal(
        dn_robin[0, :], dn_neumann[0, :],
        err_msg="dn[0,:] should be unchanged when only S_n_right is set (mapping swap?)"
    )


def test_legacy_mode_disables_selective_contacts_in_2d():
    """Tier-as-ceiling invariant: device.mode='legacy' must keep
    has_selective_contacts=False even when S values are configured.

    Mirrors the 1D pattern in tests/unit/solver/test_temperature_scaling_plumbing.py
    so the 2D solver respects the same Phase 5 tier gate as the 1D solver.
    """
    base = _stack()  # default mode='full' if unset → resolves to FULL
    # Stack that DOES configure S values, but pins the tier to legacy.
    stack_legacy = dc_replace(
        base, mode="legacy",
        S_n_left=1e-4, S_p_left=1e-3,
        S_n_right=1e-3, S_p_right=1e-4,
    )
    layers = _layers_for_stack(stack_legacy)
    g = build_grid_2d(layers, lateral_length=300e-9, Nx=4, lateral_uniform=True)
    mat = build_material_arrays_2d(g, stack_legacy, Microstructure())
    # Legacy tier ⇒ Robin off, even though S values are present.
    assert mat.has_selective_contacts is False, (
        "device.mode='legacy' must disable Robin even when S_* are configured"
    )

    # Sanity: same stack with mode='full' DOES enable Robin.
    stack_full = dc_replace(stack_legacy, mode="full")
    mat_full = build_material_arrays_2d(g, stack_full, Microstructure())
    assert mat_full.has_selective_contacts is True, (
        "device.mode='full' with S_* configured must enable Robin"
    )


# ---------------------------------------------------------------------------
# Stage B(c.2): Field-dependent mobility μ(E) — MaterialArrays2D wiring tests
# ---------------------------------------------------------------------------


def _stack_with_layer_params(stack, **layer_param_overrides):
    """Return a DeviceStack with every electrical layer's params updated.

    `v_sat_n`, `v_sat_p`, `ct_beta_n`, `ct_beta_p`, `pf_gamma_n`, `pf_gamma_p`
    live on MaterialParams (per-layer), not on DeviceStack — so the wiring
    tests have to push the override down a level via dc_replace().
    """
    new_layers = []
    for L in stack.layers:
        if L.params is None:
            new_layers.append(L)
            continue
        new_params = dc_replace(L.params, **layer_param_overrides)
        new_layers.append(dc_replace(L, params=new_params))
    return dc_replace(stack, layers=tuple(new_layers))


def test_material_arrays_2d_default_no_field_mobility():
    """Default preset → has_field_mobility=False and all 18 face fields None."""
    stack = _stack()
    layers = _layers_for_stack(stack)
    g = build_grid_2d(layers, lateral_length=300e-9, Nx=4, lateral_uniform=True)
    mat = build_material_arrays_2d(g, stack, Microstructure())
    assert mat.has_field_mobility is False
    for name in (
        "v_sat_n_x_face", "v_sat_n_y_face", "v_sat_p_x_face", "v_sat_p_y_face",
        "ct_beta_n_x_face", "ct_beta_n_y_face", "ct_beta_p_x_face", "ct_beta_p_y_face",
        "pf_gamma_n_x_face", "pf_gamma_n_y_face", "pf_gamma_p_x_face", "pf_gamma_p_y_face",
        "v_sat_n_wrap", "v_sat_p_wrap",
        "ct_beta_n_wrap", "ct_beta_p_wrap",
        "pf_gamma_n_wrap", "pf_gamma_p_wrap",
    ):
        assert getattr(mat, name) is None, f"{name} should be None when field-mobility is off"


def test_material_arrays_2d_v_sat_activates_flag_and_shapes_neumann():
    """v_sat>0 with mode='full' → has_field_mobility=True and all interior face
    arrays have correct shapes; wrap arrays remain None for non-periodic BC."""
    stack = _stack_with_layer_params(_stack(), v_sat_n=1e2, v_sat_p=1e2)
    layers = _layers_for_stack(stack)
    g = build_grid_2d(layers, lateral_length=300e-9, Nx=4, lateral_uniform=True)
    mat = build_material_arrays_2d(g, stack, Microstructure(), lateral_bc="neumann")
    assert mat.has_field_mobility is True
    assert mat.v_sat_n_x_face.shape == (g.Ny, g.Nx - 1)
    assert mat.v_sat_n_y_face.shape == (g.Ny - 1, g.Nx)
    assert mat.v_sat_p_x_face.shape == (g.Ny, g.Nx - 1)
    assert mat.v_sat_p_y_face.shape == (g.Ny - 1, g.Nx)
    assert mat.ct_beta_n_x_face.shape == (g.Ny, g.Nx - 1)
    assert mat.ct_beta_p_y_face.shape == (g.Ny - 1, g.Nx)
    assert mat.pf_gamma_n_x_face.shape == (g.Ny, g.Nx - 1)
    assert mat.pf_gamma_p_y_face.shape == (g.Ny - 1, g.Nx)
    # Wrap arrays not populated for Neumann BC
    assert mat.v_sat_n_wrap is None
    assert mat.v_sat_p_wrap is None


def test_material_arrays_2d_periodic_populates_wrap_arrays():
    """Periodic BC with v_sat>0 → all six wrap arrays populated with shape (Ny,)."""
    stack = _stack_with_layer_params(_stack(), v_sat_n=1e2, v_sat_p=1e2)
    layers = _layers_for_stack(stack)
    g = build_grid_2d(layers, lateral_length=300e-9, Nx=4, lateral_uniform=True)
    mat = build_material_arrays_2d(g, stack, Microstructure(), lateral_bc="periodic")
    assert mat.has_field_mobility is True
    for name in (
        "v_sat_n_wrap", "v_sat_p_wrap",
        "ct_beta_n_wrap", "ct_beta_p_wrap",
        "pf_gamma_n_wrap", "pf_gamma_p_wrap",
    ):
        arr = getattr(mat, name)
        assert arr is not None, f"{name} must be populated under periodic BC"
        assert arr.shape == (g.Ny,)


def test_material_arrays_2d_field_mobility_values_match_layer_params():
    """Layer v_sat_n=1e2 → mat.v_sat_n_y_face equals 1e2 inside that layer (arithmetic mean
    of two equal nodes is the node value)."""
    stack = _stack_with_layer_params(_stack(), v_sat_n=1e2, v_sat_p=2e2)
    layers = _layers_for_stack(stack)
    g = build_grid_2d(layers, lateral_length=300e-9, Nx=4, lateral_uniform=True)
    mat = build_material_arrays_2d(g, stack, Microstructure(), lateral_bc="neumann")
    # Every layer has the same v_sat → arithmetic mean across any face equals v_sat.
    np.testing.assert_allclose(mat.v_sat_n_y_face, 1e2)
    np.testing.assert_allclose(mat.v_sat_p_y_face, 2e2)
    np.testing.assert_allclose(mat.v_sat_n_x_face, 1e2)
    np.testing.assert_allclose(mat.v_sat_p_x_face, 2e2)


def test_legacy_mode_disables_field_mobility_in_2d():
    """Tier-as-ceiling: device.mode='legacy' must keep has_field_mobility=False
    even when v_sat is set on the stack. Mirrors the B(c.1) Robin tier-gate test
    that was added during Issue I1 fix."""
    base = _stack()
    stack_legacy = dc_replace(
        _stack_with_layer_params(base, v_sat_n=1e2, v_sat_p=1e2),
        mode="legacy",
    )
    layers = _layers_for_stack(stack_legacy)
    g = build_grid_2d(layers, lateral_length=300e-9, Nx=4, lateral_uniform=True)
    mat = build_material_arrays_2d(g, stack_legacy, Microstructure())
    assert mat.has_field_mobility is False
    # Sanity: same params with mode='full' enables.
    stack_full = dc_replace(stack_legacy, mode="full")
    mat_full = build_material_arrays_2d(g, stack_full, Microstructure())
    assert mat_full.has_field_mobility is True


def test_pf_gamma_alone_activates_flag():
    """Setting only pf_gamma (with v_sat=0) is enough to trip the activation gate."""
    stack = _stack_with_layer_params(_stack(), pf_gamma_n=3e-4, pf_gamma_p=3e-4)
    layers = _layers_for_stack(stack)
    g = build_grid_2d(layers, lateral_length=300e-9, Nx=4, lateral_uniform=True)
    mat = build_material_arrays_2d(g, stack, Microstructure())
    assert mat.has_field_mobility is True
    assert mat.pf_gamma_n_x_face is not None
    np.testing.assert_allclose(mat.pf_gamma_n_x_face, 3e-4)


# ---------------------------------------------------------------------------
# Stage B(c.2) Task 4: assemble_rhs_2d field-mobility per-RHS recompute
# ---------------------------------------------------------------------------


def test_assemble_rhs_2d_field_mobility_disabled_path_unchanged():
    """When v_sat=pf_gamma=0 (default preset), mat.has_field_mobility is False
    and assemble_rhs_2d output is bit-identical to legacy-mode-with-vsat (which
    is also disabled via the tier gate)."""
    from perovskite_sim.twod.solver_2d import assemble_rhs_2d
    base = _stack()
    stack_off    = base                                                       # mode=full, no v_sat
    stack_legacy = dc_replace(_stack_with_layer_params(base, v_sat_n=1e2, v_sat_p=1e2), mode="legacy")
    layers = _layers_for_stack(base)
    g = build_grid_2d(layers, lateral_length=300e-9, Nx=4, lateral_uniform=True)
    mat_off    = build_material_arrays_2d(g, stack_off,    Microstructure(), lateral_bc="periodic")
    mat_legacy = build_material_arrays_2d(g, stack_legacy, Microstructure(), lateral_bc="periodic")
    assert mat_off.has_field_mobility is False
    assert mat_legacy.has_field_mobility is False
    n0 = float(mat_off.n_eq_left[0]) * np.ones((g.Ny, g.Nx))
    p0 = float(mat_off.p_eq_left[0]) * np.ones((g.Ny, g.Nx))
    y0 = np.concatenate([n0.flatten(), p0.flatten()])
    dydt_off    = assemble_rhs_2d(0.0, y0, mat_off,    V_app=0.0)
    dydt_legacy = assemble_rhs_2d(0.0, y0, mat_legacy, V_app=0.0)
    np.testing.assert_array_equal(dydt_off, dydt_legacy)


def test_assemble_rhs_2d_field_mobility_enabled_changes_dydt():
    """When v_sat=1e2 with mode='full', assemble_rhs_2d output differs from
    the constant-mobility baseline at a state with non-zero E."""
    from perovskite_sim.twod.solver_2d import assemble_rhs_2d
    base = _stack()
    stack_off = base
    stack_on  = _stack_with_layer_params(base, v_sat_n=1e2, v_sat_p=1e2)
    layers = _layers_for_stack(base)
    g = build_grid_2d(layers, lateral_length=300e-9, Nx=4, lateral_uniform=True)
    mat_off = build_material_arrays_2d(g, stack_off, Microstructure(), lateral_bc="periodic")
    mat_on  = build_material_arrays_2d(g, stack_on,  Microstructure(), lateral_bc="periodic")
    assert mat_off.has_field_mobility is False
    assert mat_on.has_field_mobility is True
    # Build a state with non-trivial gradients in y so E_y != 0.
    n_grad = np.linspace(float(mat_off.n_eq_left[0]),
                        float(mat_off.n_eq_right[0]), g.Ny)
    p_grad = np.linspace(float(mat_off.p_eq_left[0]),
                        float(mat_off.p_eq_right[0]), g.Ny)
    n0 = np.broadcast_to(n_grad[:, None], (g.Ny, g.Nx)).copy()
    p0 = np.broadcast_to(p_grad[:, None], (g.Ny, g.Nx)).copy()
    y0 = np.concatenate([n0.flatten(), p0.flatten()])
    dydt_off = assemble_rhs_2d(0.0, y0, mat_off, V_app=0.5)
    dydt_on  = assemble_rhs_2d(0.0, y0, mat_on,  V_app=0.5)
    # mu(E) actively perturbs the RHS at non-trivial fields.
    assert not np.array_equal(dydt_off, dydt_on)
    rel = np.max(np.abs(dydt_on - dydt_off)) / max(1.0, np.max(np.abs(dydt_off)))
    assert rel > 1e-6, f"mu(E) effect on RHS too small (rel diff {rel:.2e})"


def test_assemble_rhs_2d_field_mobility_finite_periodic():
    """mu(E) on with lateral_bc='periodic' produces a finite RHS at a non-trivial
    state. Catches a missing wrap-face override or a periodic-wrap shape bug."""
    from perovskite_sim.twod.solver_2d import assemble_rhs_2d
    stack = _stack_with_layer_params(_stack(), v_sat_n=1e2, v_sat_p=1e2)
    layers = _layers_for_stack(stack)
    g = build_grid_2d(layers, lateral_length=300e-9, Nx=5, lateral_uniform=True)
    mat = build_material_arrays_2d(g, stack, Microstructure(), lateral_bc="periodic")
    assert mat.has_field_mobility is True
    assert mat.v_sat_n_wrap is not None
    # Build a state with broken lateral symmetry -> non-zero E_x at wrap face
    n0 = float(mat.n_eq_left[0]) * np.ones((g.Ny, g.Nx))
    p0 = float(mat.p_eq_left[0]) * np.ones((g.Ny, g.Nx))
    n0[:, 0] *= 1.5   # asymmetric -- drives non-trivial wrap-face E_x
    y0 = np.concatenate([n0.flatten(), p0.flatten()])
    dydt = assemble_rhs_2d(0.0, y0, mat, V_app=0.0)
    assert np.all(np.isfinite(dydt)), "mu(E) periodic wrap produced non-finite RHS"
