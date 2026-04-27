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
