from __future__ import annotations

from dataclasses import replace

import numpy as np

from perovskite_sim.discretization.grid import Layer
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.models.device import electrical_layers
from perovskite_sim.solver.tolerances import ComponentwiseAtol
from perovskite_sim.twod.grid_2d import build_grid_2d
from perovskite_sim.twod.microstructure import Microstructure
from perovskite_sim.twod.solver_2d import (
    build_material_arrays_2d,
    run_transient_2d,
)


def test_single_mobile_ion_transient_redistributes_and_conserves_inventory():
    base = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    layers = list(base.layers)
    absorber = layers[1]
    layers[1] = replace(
        absorber,
        params=replace(absorber.params, D_ion=1.0e-10),
    )
    stack = replace(base, layers=tuple(layers))
    grid = build_grid_2d(
        [Layer(layer.thickness, 2) for layer in electrical_layers(stack)],
        lateral_length=100.0e-9,
        Nx=2,
        lateral_uniform=True,
    )
    material = build_material_arrays_2d(
        grid,
        stack,
        Microstructure(),
        lateral_bc="neumann",
        ion_dynamics="single_mobile",
    )
    n0 = np.maximum(material.ni, 1.0)
    p0 = np.maximum(material.ni, 1.0)
    P0 = material.P_ion0_2d.copy()
    state0 = np.concatenate([n0.ravel(), p0.ravel(), P0.ravel()])

    terminal, report = run_transient_2d(
        state0,
        material,
        V_app=0.02,
        t_end=1.0e-12,
        max_step=2.0e-13,
        rtol=1.0e-6,
        atol=ComponentwiseAtol(),
        max_nfev=20_000,
        return_ion_diagnostics=True,
    )

    assert material.D_ion_2d is not None
    P1 = terminal[2 * grid.n_nodes:].reshape(P0.shape)
    active = material.D_ion_2d > 0.0
    relative_change = np.max(np.abs(P1[active] - P0[active]) / P0[active])
    lateral_nonuniformity = np.max(np.abs(P1 - P1[:, [0]])) / np.max(P1)

    assert report.passed is True
    assert report.relative_inventory_drift < 1.0e-12
    assert relative_change > 1.0e-8
    assert lateral_nonuniformity < 1.0e-12
    assert report.terminal_min_electron_density_m3 >= 0.0
    assert report.terminal_min_hole_density_m3 >= 0.0
