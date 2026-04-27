from __future__ import annotations
import numpy as np
import pytest

from perovskite_sim.twod.microstructure import (
    GrainBoundary, Microstructure, build_tau_field,
)
from perovskite_sim.twod.grid_2d import build_grid_2d
from perovskite_sim.discretization.grid import Layer


def _grid():
    layers = [Layer(thickness=400e-9, N=20)]
    return build_grid_2d(layers, lateral_length=500e-9, Nx=10, lateral_uniform=True)


def test_empty_microstructure_returns_uniform_tau():
    g = _grid()
    tau_bulk_per_layer = np.full((g.Ny,), 1e-6)
    ustruct = Microstructure()
    tau_n, tau_p = build_tau_field(g, ustruct, tau_bulk_per_layer, tau_bulk_per_layer,
                                   layer_role_per_y=["absorber"] * g.Ny)
    assert tau_n.shape == (g.Ny, g.Nx)
    assert tau_p.shape == (g.Ny, g.Nx)
    assert np.allclose(tau_n, 1e-6)
    assert np.allclose(tau_p, 1e-6)


def test_grain_boundary_dataclass_is_frozen():
    gb = GrainBoundary(x_position=250e-9, width=5e-9,
                       tau_n=1e-9, tau_p=1e-9, layer_role="absorber")
    with pytest.raises(Exception):
        gb.x_position = 100e-9  # frozen — should raise


def test_microstructure_dataclass_default_is_empty():
    ustruct = Microstructure()
    assert ustruct.grain_boundaries == ()
