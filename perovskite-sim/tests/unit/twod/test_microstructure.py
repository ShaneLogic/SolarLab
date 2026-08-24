from __future__ import annotations
import numpy as np
import pytest
import yaml

from perovskite_sim.twod.microstructure import (
    GrainBoundary, Microstructure, build_grain_boundary_regions,
    build_tau_field, lateral_dual_cell_widths,
    load_microstructure_from_yaml_block,
)
from perovskite_sim.twod.grid_2d import Grid2D, build_grid_2d
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


def test_microstructure_yaml_loader_rejects_unknown_top_level_key():
    with pytest.raises(ValueError, match="microstructure unknown key"):
        load_microstructure_from_yaml_block({"grain_boundary": []})


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("x_position", np.nan, "x_position must be finite"),
        ("width", 0.0, "width must be positive"),
        ("width", np.inf, "width must be finite"),
        ("tau_n", -1.0, "lifetimes must be positive"),
        ("tau_p", 0.0, "lifetimes must be positive"),
    ],
)
def test_grain_boundary_rejects_invalid_physical_parameters(field, value, match):
    kwargs = dict(x_position=250e-9, width=5e-9, tau_n=1e-9, tau_p=1e-9)
    kwargs[field] = value
    with pytest.raises(ValueError, match=match):
        GrainBoundary(**kwargs)


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


def test_nonempty_microstructure_rejects_legacy_node_painted_tau():
    g = build_grid_2d([Layer(thickness=400e-9, N=20)],
                      lateral_length=500e-9, Nx=20, lateral_uniform=True)
    tau_bulk = np.full((g.Ny,), 1e-6)
    gb = GrainBoundary(x_position=250e-9, width=20e-9,
                       tau_n=1e-9, tau_p=2e-9, layer_role="absorber")
    ustruct = Microstructure(grain_boundaries=(gb,))
    with pytest.raises(ValueError, match="not area-conservative"):
        build_tau_field(
            g,
            ustruct,
            tau_bulk,
            tau_bulk,
            layer_role_per_y=["absorber"] * g.Ny,
        )


@pytest.mark.parametrize("intervals", [4, 10, 63])
def test_grain_boundary_overlap_preserves_subcell_physical_width(intervals):
    g = build_grid_2d(
        [Layer(thickness=400e-9, N=4)],
        lateral_length=500e-9,
        Nx=intervals,
        lateral_uniform=True,
    )
    gb = GrainBoundary(
        x_position=247e-9,
        width=5e-9,
        tau_n=1e-9,
        tau_p=2e-9,
    )
    (region,) = build_grain_boundary_regions(
        g,
        Microstructure((gb,)),
        ["absorber"] * g.Ny,
        lateral_bc="neumann",
    )
    weights = lateral_dual_cell_widths(g.x)
    assert np.all(region.x_overlap_fraction >= 0.0)
    assert np.all(region.x_overlap_fraction <= 1.0)
    assert np.dot(region.x_overlap_fraction, weights) == pytest.approx(
        gb.width,
        rel=2e-14,
        abs=1e-21,
    )
    assert not region.x_overlap_fraction.flags.writeable
    assert not region.y_mask.flags.writeable


def test_grain_boundary_overlap_supports_nonuniform_lateral_grid():
    g = Grid2D(
        x=np.array([0.0, 20e-9, 90e-9, 240e-9, 500e-9]),
        y=np.array([0.0, 100e-9]),
    )
    gb = GrainBoundary(215e-9, 17e-9, 1e-9, 2e-9)
    (region,) = build_grain_boundary_regions(
        g,
        Microstructure((gb,)),
        ["absorber", "absorber"],
        lateral_bc="neumann",
    )
    integrated = np.dot(
        region.x_overlap_fraction,
        lateral_dual_cell_widths(g.x),
    )
    assert integrated == pytest.approx(gb.width, rel=2e-14, abs=1e-21)


def test_grain_boundary_region_targets_only_declared_layer_role():
    g = _grid()
    roles = ["HTL"] * 3 + ["absorber"] * (g.Ny - 5) + ["ETL"] * 2
    gb = GrainBoundary(250e-9, 5e-9, 1e-9, 1e-9)
    (region,) = build_grain_boundary_regions(
        g,
        Microstructure((gb,)),
        roles,
        lateral_bc="neumann",
    )
    np.testing.assert_array_equal(region.y_mask, np.asarray(roles) == "absorber")


def test_grain_boundary_region_rejects_unknown_layer_role():
    g = _grid()
    gb = GrainBoundary(250e-9, 5e-9, 1e-9, 1e-9, layer_role="typo")
    with pytest.raises(ValueError, match="matches no grid rows"):
        build_grain_boundary_regions(
            g,
            Microstructure((gb,)),
            ["absorber"] * g.Ny,
            lateral_bc="neumann",
        )


def test_grain_boundary_region_rejects_band_outside_domain():
    g = _grid()
    gb = GrainBoundary(1e-9, 5e-9, 1e-9, 1e-9)
    with pytest.raises(ValueError, match="fully inside"):
        build_grain_boundary_regions(
            g,
            Microstructure((gb,)),
            ["absorber"] * g.Ny,
            lateral_bc="neumann",
        )


def test_grain_boundary_region_rejects_overlapping_bands():
    g = _grid()
    microstructure = Microstructure(
        (
            GrainBoundary(250e-9, 20e-9, 1e-9, 1e-9),
            GrainBoundary(255e-9, 20e-9, 2e-9, 2e-9),
        )
    )
    with pytest.raises(ValueError, match="overlapping"):
        build_grain_boundary_regions(
            g,
            microstructure,
            ["absorber"] * g.Ny,
            lateral_bc="neumann",
        )


def test_nonempty_microstructure_fails_closed_on_periodic_topology():
    g = _grid()
    gb = GrainBoundary(250e-9, 5e-9, 1e-9, 1e-9)
    with pytest.raises(ValueError, match="not area-certified"):
        build_grain_boundary_regions(
            g,
            Microstructure((gb,)),
            ["absorber"] * g.Ny,
            lateral_bc="periodic",
        )


def test_empty_microstructure_keeps_periodic_compatibility_path():
    g = _grid()
    assert build_grain_boundary_regions(
        g,
        Microstructure(),
        ["absorber"] * g.Ny,
        lateral_bc="periodic",
    ) == ()
