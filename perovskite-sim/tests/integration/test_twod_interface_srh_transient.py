from __future__ import annotations

import numpy as np

from perovskite_sim.discretization.grid import Layer
from perovskite_sim.models.device import DeviceStack, InterfaceDefect, LayerSpec
from perovskite_sim.models.parameters import MaterialParams
from perovskite_sim.twod.grid_2d import build_grid_2d
from perovskite_sim.twod.microstructure import Microstructure
from perovskite_sim.twod.solver_2d import (
    build_material_arrays_2d,
    run_transient_2d,
)


def _problem():
    material = MaterialParams(
        eps_r=10.0,
        mu_n=1.0e-3,
        mu_p=1.0e-3,
        D_ion=0.0,
        P_lim=1.0e24,
        P0=0.0,
        ni=1.0e12,
        tau_n=1.0e30,
        tau_p=1.0e30,
        n1=1.0e12,
        p1=1.0e12,
        B_rad=0.0,
        C_n=0.0,
        C_p=0.0,
        alpha=0.0,
        N_A=0.0,
        N_D=0.0,
        chi=4.0,
        Eg=1.5,
        Nc300=1.0e25,
        Nv300=1.0e25,
    )
    stack = DeviceStack(
        layers=(
            LayerSpec("left", 1.0e-7, material, role="absorber"),
            LayerSpec("right", 1.0e-7, material, role="ETL"),
        ),
        interfaces=((0.03, 0.05),),
        interface_defects=(InterfaceDefect(E_t_eV=0.5),),
        interface_two_sided=True,
        V_bi=0.0,
        Phi=0.0,
        mode="full",
    )
    grid = build_grid_2d(
        [Layer(layer.thickness, 2) for layer in stack.layers],
        lateral_length=1.0e-7,
        Nx=3,
        alpha_y=1.0,
        lateral_uniform=True,
    )
    return stack, grid


def test_two_sided_interface_srh_transient_remains_uniform_and_removes_pairs():
    stack, grid = _problem()
    enabled = build_material_arrays_2d(
        grid,
        stack,
        Microstructure(),
        lateral_bc="neumann",
        interface_srh="two_sided_cross_node",
    )
    disabled = build_material_arrays_2d(
        grid,
        stack,
        Microstructure(),
        lateral_bc="neumann",
    )
    n0 = np.full((grid.Ny, grid.Nx), 1.0e20)
    p0 = np.full_like(n0, 1.0e20)
    state0 = np.concatenate([n0.ravel(), p0.ravel()])
    options = {
        "V_app": 0.0,
        "t_end": 1.0e-10,
        "max_step": 2.0e-11,
        "rtol": 1.0e-7,
        "atol": 1.0e6,
        "max_nfev": 20_000,
    }

    terminal_off = run_transient_2d(state0, disabled, **options)
    terminal_on = run_transient_2d(state0, enabled, **options)
    n_off, p_off = np.split(terminal_off, 2)
    n_on, p_on = np.split(terminal_on, 2)
    n_on_2d = n_on.reshape(grid.Ny, grid.Nx)
    p_on_2d = p_on.reshape(grid.Ny, grid.Nx)

    assert np.sum(n_on) < np.sum(n_off)
    assert np.sum(p_on) < np.sum(p_off)
    assert np.min(n_on) > 0.0
    assert np.min(p_on) > 0.0
    np.testing.assert_allclose(
        n_on_2d,
        np.broadcast_to(n_on_2d[:, [0]], n_on_2d.shape),
        rtol=2.0e-13,
    )
    np.testing.assert_allclose(
        p_on_2d,
        np.broadcast_to(p_on_2d[:, [0]], p_on_2d.shape),
        rtol=2.0e-13,
    )
