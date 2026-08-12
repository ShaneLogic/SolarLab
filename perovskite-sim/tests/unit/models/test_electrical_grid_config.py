from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest
import yaml

from backend.main import _stack_to_config_dict, stack_from_dict
from perovskite_sim.discretization.grid import Layer, multilayer_grid
from perovskite_sim.experiments.jv_sweep import (
    _layer_node_counts,
    build_electrical_grid,
)
from perovskite_sim.models.config_loader import (
    electrical_grid_from_config_dict,
    load_device_from_yaml,
)
from perovskite_sim.models.device import LayerSpec, electrical_layers


CSI_CONFIG = Path("configs/cSi_homojunction.yaml")


def _named_layers() -> tuple[LayerSpec, ...]:
    return (
        LayerSpec("glass", 1.0e-3, None, "substrate"),
        LayerSpec("left", 100.0e-9, None, "ETL"),
        LayerSpec("right", 500.0e-9, None, "absorber"),
    )


def _grid_block() -> dict:
    return {
        "electrical_grid": {
            "interval_weights": {"right": 4, "left": 1},
            "alphas": {"right": 3, "left": 2},
        }
    }


def test_grid_parser_orders_maps_by_electrical_layers_and_ignores_substrate():
    weights, alphas = electrical_grid_from_config_dict(
        _grid_block(), _named_layers()
    )

    assert weights == (1.0, 4.0)
    assert alphas == (2.0, 3.0)
    assert electrical_grid_from_config_dict({}, _named_layers()) == ((), ())


@pytest.mark.parametrize(
    ("cfg", "message"),
    [
        ({"electrical_grid": []}, "must be a mapping"),
        (
            {"electrical_grid": {"interval_weights": {}}},
            "contain exactly interval_weights and alphas",
        ),
        (
            {
                "electrical_grid": {
                    "interval_weights": {"left": 1, "right": 4, "glass": 1},
                    "alphas": {"left": 2, "right": 3},
                }
            },
            "cover exactly the electrical layers",
        ),
        (
            {
                "electrical_grid": {
                    "interval_weights": {"left": 1, "right": 4},
                    "alphas": {"left": 2, "right": 0},
                }
            },
            "finite and positive",
        ),
        (
            {
                "electrical_grid": {
                    "interval_weights": {"left": True, "right": 4},
                    "alphas": {"left": 2, "right": 3},
                }
            },
            "finite and positive",
        ),
    ],
)
def test_grid_parser_rejects_incomplete_extra_or_nonpositive_maps(cfg, message):
    with pytest.raises(ValueError, match=message):
        electrical_grid_from_config_dict(cfg, _named_layers())


@pytest.mark.parametrize("value", [-1.0, np.inf, np.nan, "nan"])
def test_grid_parser_rejects_nonpositive_or_nonfinite_numbers(value):
    cfg = _grid_block()
    cfg["electrical_grid"]["interval_weights"]["left"] = value

    with pytest.raises(ValueError, match="finite and positive"):
        electrical_grid_from_config_dict(cfg, _named_layers())


def test_yaml_loader_rejects_incomplete_electrical_grid(tmp_path):
    raw = yaml.safe_load(CSI_CONFIG.read_text(encoding="utf-8"))
    del raw["electrical_grid"]["alphas"]["p_base"]
    path = tmp_path / "incomplete_grid.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="cover exactly the electrical layers"):
        load_device_from_yaml(path)


def test_csi_loader_and_backend_roundtrip_preserve_executable_grid():
    stack = load_device_from_yaml(CSI_CONFIG)

    assert stack.grid_interval_weights == (1.0, 4.0)
    assert stack.grid_alphas == (2.0, 3.0)
    assert stack.V_bi == pytest.approx(0.8928964399850017, rel=0.0, abs=1.0e-15)
    assert abs(stack.compute_V_bi()) == pytest.approx(
        stack.V_bi, rel=0.0, abs=1.0e-15
    )
    serialized = _stack_to_config_dict(stack)
    assert serialized["electrical_grid"] == {
        "interval_weights": {"n_emitter": 1.0, "p_base": 4.0},
        "alphas": {"n_emitter": 2.0, "p_base": 3.0},
    }

    inline = stack_from_dict(serialized)
    assert inline.grid_interval_weights == stack.grid_interval_weights
    assert inline.grid_alphas == stack.grid_alphas
    np.testing.assert_array_equal(
        build_electrical_grid(inline, 200),
        build_electrical_grid(stack, 200),
    )


def test_backend_inline_path_reuses_strict_grid_parser():
    stack = load_device_from_yaml(CSI_CONFIG)
    serialized = _stack_to_config_dict(stack)
    del serialized["electrical_grid"]["alphas"]["p_base"]

    with pytest.raises(ValueError, match="cover exactly the electrical layers"):
        stack_from_dict(serialized)


@pytest.mark.parametrize(
    ("n_grid", "expected"),
    [(200, [40, 160]), (300, [60, 240]), (400, [80, 320])],
)
def test_csi_weighted_allocation_uses_exact_total(n_grid, expected):
    stack = load_device_from_yaml(CSI_CONFIG)

    counts = _layer_node_counts(stack, n_grid)

    assert counts == expected
    assert sum(counts) == n_grid
    assert len(build_electrical_grid(stack, n_grid)) == n_grid + 1


def test_largest_remainder_ties_are_layer_ordered_and_every_layer_gets_one():
    base = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    tie = dataclasses.replace(
        base,
        grid_interval_weights=(1.0, 1.0, 1.0),
        grid_alphas=(3.0, 3.0, 3.0),
    )
    skewed = dataclasses.replace(
        base,
        grid_interval_weights=(1000.0, 1.0, 1.0),
        grid_alphas=(3.0, 3.0, 3.0),
    )

    assert _layer_node_counts(tie, 4) == [2, 1, 1]
    assert _layer_node_counts(skewed, 3) == [1, 1, 1]
    with pytest.raises(ValueError, match="at least 3 intervals"):
        _layer_node_counts(skewed, 2)


def test_custom_weights_fail_loudly_with_graded_multiplier():
    graded = load_device_from_yaml("configs/cigs_graded_notch.yaml")
    n_layers = len(electrical_layers(graded))
    custom = dataclasses.replace(
        graded,
        grid_interval_weights=(1.0,) * n_layers,
        grid_alphas=(3.0,) * n_layers,
    )

    assert custom.band_grading
    assert any(
        layer.params.grading_N_mult > 1 for layer in electrical_layers(custom)
    )
    with pytest.raises(ValueError, match="cannot be combined"):
        _layer_node_counts(custom, 120)


def test_absent_grid_config_preserves_historical_grid_exactly():
    stack = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    elec = electrical_layers(stack)
    n_grid = 100
    n_per = n_grid // len(elec)
    historical = multilayer_grid([
        Layer(layer.thickness, n_per) for layer in elec
    ], alpha=3.0)

    assert stack.grid_interval_weights == ()
    assert stack.grid_alphas == ()
    assert "electrical_grid" not in _stack_to_config_dict(stack)
    assert _layer_node_counts(stack, n_grid) == [n_per] * len(elec)
    np.testing.assert_array_equal(build_electrical_grid(stack, n_grid), historical)


def test_csi_grid_uses_per_layer_alpha_and_balanced_debye_resolution():
    stack = load_device_from_yaml(CSI_CONFIG)
    x = build_electrical_grid(stack, 200)
    interface = stack.layers[0].thickness
    node = int(np.flatnonzero(x == interface)[0])

    emitter_dx = x[node] - x[node - 1]
    base_dx = x[node + 1] - x[node]
    assert emitter_dx == pytest.approx(1.212121225950591e-9, rel=1.0e-12)
    assert base_dx == pytest.approx(34.74351420672539e-9, rel=1.0e-12)
    assert np.min(np.diff(x[: node + 1])) > 1.0e-10
