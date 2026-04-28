from __future__ import annotations
import numpy as np
import pytest
import yaml

from perovskite_sim.twod.microstructure import (
    GrainBoundary, Microstructure, build_tau_field,
    load_microstructure_from_yaml_block,
)
from perovskite_sim.twod.grid_2d import build_grid_2d
from perovskite_sim.discretization.grid import Layer
from perovskite_sim.models.config_loader import load_device_from_yaml


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


def test_microstructure_yaml_loader_single_gb():
    yaml_text = """
microstructure:
  grain_boundaries:
    - x_position: 250e-9
      width: 5e-9
      tau_n: 1e-9
      tau_p: 1e-9
      layer_role: absorber
"""
    block = yaml.safe_load(yaml_text)["microstructure"]
    ms = load_microstructure_from_yaml_block(block)
    assert len(ms.grain_boundaries) == 1
    gb = ms.grain_boundaries[0]
    assert gb.x_position == pytest.approx(250e-9)
    assert gb.width == pytest.approx(5e-9)
    assert gb.tau_n == pytest.approx(1e-9)
    assert gb.tau_p == pytest.approx(1e-9)
    assert gb.layer_role == "absorber"


def test_microstructure_yaml_loader_empty_block_returns_empty():
    assert load_microstructure_from_yaml_block(None).grain_boundaries == ()
    assert load_microstructure_from_yaml_block({}).grain_boundaries == ()
    assert load_microstructure_from_yaml_block(
        {"grain_boundaries": []}
    ).grain_boundaries == ()


def test_microstructure_yaml_loader_rejects_unknown_keys():
    bad_block = {
        "grain_boundaries": [
            {
                "x_position": 250e-9,
                "width": 5e-9,
                "tau_n": 1e-9,
                "tau_p": 1e-9,
                "tau_typo": 1e-9,  # unknown — must raise
            }
        ]
    }
    with pytest.raises(ValueError, match="unknown key"):
        load_microstructure_from_yaml_block(bad_block)


def test_load_device_from_yaml_attaches_microstructure():
    stack = load_device_from_yaml("configs/twod/nip_MAPbI3_singleGB.yaml")
    assert hasattr(stack, "microstructure")
    assert len(stack.microstructure.grain_boundaries) == 1
    gb = stack.microstructure.grain_boundaries[0]
    assert gb.x_position == pytest.approx(250e-9)
    assert gb.width == pytest.approx(5e-9)
    assert gb.tau_n == pytest.approx(5e-8)
    assert gb.tau_p == pytest.approx(5e-8)
    assert gb.layer_role == "absorber"


def test_load_device_from_yaml_empty_microstructure_default():
    stack = load_device_from_yaml("configs/twod/nip_MAPbI3_uniform.yaml")
    assert hasattr(stack, "microstructure")
    assert stack.microstructure.grain_boundaries == ()


def test_grain_boundary_overrides_tau_in_band():
    """build_tau_field replaces the bulk τ inside a GB band and leaves it
    untouched everywhere else. Per-carrier independence: τ_n and τ_p override
    independently, so the test uses different values for each."""
    g = build_grid_2d([Layer(thickness=400e-9, N=20)],
                      lateral_length=500e-9, Nx=20, lateral_uniform=True)
    tau_bulk = np.full((g.Ny,), 1e-6)
    gb = GrainBoundary(x_position=250e-9, width=20e-9,
                       tau_n=1e-9, tau_p=2e-9, layer_role="absorber")
    ustruct = Microstructure(grain_boundaries=(gb,))
    tau_n, tau_p = build_tau_field(g, ustruct, tau_bulk, tau_bulk,
                                   layer_role_per_y=["absorber"] * g.Ny)
    in_band = np.abs(g.x - 250e-9) < 10e-9
    assert np.allclose(tau_n[:, in_band], 1e-9)
    assert np.allclose(tau_p[:, in_band], 2e-9)
    assert np.allclose(tau_n[:, ~in_band], 1e-6)
    assert np.allclose(tau_p[:, ~in_band], 1e-6)


def test_grain_boundary_skipped_outside_target_layer():
    """A GB tagged with layer_role="absorber" must not affect rows that
    belong to a different layer role, even when its lateral band would
    otherwise cover those nodes."""
    g = build_grid_2d([Layer(thickness=400e-9, N=20)],
                      lateral_length=500e-9, Nx=20, lateral_uniform=True)
    tau_bulk = np.full((g.Ny,), 1e-6)
    gb = GrainBoundary(x_position=250e-9, width=20e-9,
                       tau_n=1e-9, tau_p=1e-9, layer_role="absorber")
    ustruct = Microstructure(grain_boundaries=(gb,))
    tau_n, _ = build_tau_field(g, ustruct, tau_bulk, tau_bulk,
                               layer_role_per_y=["transport"] * g.Ny)
    assert np.allclose(tau_n, 1e-6)
