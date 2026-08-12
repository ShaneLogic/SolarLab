"""V_bi_eff propagation from DeviceStack → MaterialArrays.

Phase 1 wires ``DeviceStack.compute_V_bi()`` through ``MaterialArrays.V_bi_eff``
so that voltage-sweep defaults (``run_jv_sweep`` ``V_max``) can open the window
far enough to capture V_oc on heterostacks. This file locks down the contract:

    mat.V_bi_eff == stack.compute_V_bi()

holds after ``build_material_arrays``, for both heterointerface configs (where
V_bi_eff differs from the manual ``stack.V_bi`` field) and legacy configs with
chi = Eg = 0 (where V_bi_eff falls back to ``stack.V_bi`` by construction of
``compute_V_bi``).

Scope note — Poisson BC
-----------------------
The default Poisson BC still uses the configured ``stack.V_bi`` magnitude to
match IonMonger's convention of V_bi as a free parameter representing the
degenerate-doping limit. Its coordinate sign, however, is inferred from the
signed contact work-function difference ``V_bi_eff``. Positive ``V_app`` is a
forward-bias magnitude for either contact order and must reduce the absolute
built-in drop. Flat-band contacts continue to use ``V_bi_eff`` directly.
"""
from __future__ import annotations

import math

import pytest

from perovskite_sim.discretization.grid import Layer, multilayer_grid
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.solver.mol import (
    build_material_arrays,
    poisson_right_boundary,
)


def _build_grid_and_mat(config_path: str):
    stack = load_device_from_yaml(config_path)
    # Only electrical layers carry drift-diffusion nodes; substrate (optical)
    # layers are filtered out inside build_material_arrays. Use the same
    # per-layer node count the hot path uses by default.
    from perovskite_sim.models.device import electrical_layers
    layers_grid = [Layer(l.thickness, 10) for l in electrical_layers(stack)]
    x = multilayer_grid(layers_grid)
    mat = build_material_arrays(x, stack)
    return stack, mat


def test_material_arrays_caches_computed_vbi_on_heterostack():
    """On a config with chi/Eg set, V_bi_eff matches compute_V_bi()."""
    stack, mat = _build_grid_and_mat("configs/ionmonger_benchmark.yaml")
    expected = stack.compute_V_bi()
    assert math.isclose(mat.V_bi_eff, expected, rel_tol=1e-12, abs_tol=1e-12), (
        f"V_bi_eff propagation broken: mat.V_bi_eff={mat.V_bi_eff:.6f}, "
        f"stack.compute_V_bi()={expected:.6f}"
    )


def test_material_arrays_vbi_eff_differs_from_manual_vbi_for_heterostack():
    """Sanity: on a heterointerface config, V_bi_eff is not just a passthrough
    of the manual ``stack.V_bi`` field — it's a derived quantity that generally
    differs from any configured placeholder.
    """
    stack, mat = _build_grid_and_mat("configs/ionmonger_benchmark.yaml")
    # If the two ever happen to coincide for this particular config, the plan
    # intent (V_bi_eff is band-offset-aware) is still satisfied — but it would
    # mean this regression loses its teeth. Guard with an explicit inequality.
    assert abs(mat.V_bi_eff - stack.V_bi) > 1e-6, (
        f"V_bi_eff ({mat.V_bi_eff:.4f}) coincidentally equals stack.V_bi "
        f"({stack.V_bi:.4f}) — pick a config whose heterostack Fermi difference "
        "diverges from the manual V_bi field to keep this test meaningful."
    )


def test_material_arrays_vbi_eff_falls_back_on_legacy_config():
    """On a legacy config (chi=Eg=0), V_bi_eff == stack.V_bi."""
    stack, mat = _build_grid_and_mat("configs/nip_MAPbI3.yaml")
    assert math.isclose(mat.V_bi_eff, stack.V_bi, rel_tol=1e-12, abs_tol=1e-12), (
        f"V_bi_eff legacy fallback broken: mat.V_bi_eff={mat.V_bi_eff:.6f}, "
        f"stack.V_bi={stack.V_bi:.6f}"
    )


def test_p_left_forward_bias_reduces_positive_poisson_drop():
    """Existing p-left/n-right stacks retain the original boundary mapping."""
    stack, mat = _build_grid_and_mat("configs/ionmonger_benchmark.yaml")
    assert mat.V_bi_eff > 0.0
    assert mat.junction_polarity == 1.0
    assert mat.V_bi_bc == pytest.approx(abs(stack.V_bi))
    assert poisson_right_boundary(mat, 0.0) == pytest.approx(abs(stack.V_bi))
    assert poisson_right_boundary(mat, 0.4) == pytest.approx(abs(stack.V_bi) - 0.4)


def test_n_left_forward_bias_reduces_negative_poisson_drop():
    """CIGS n-left/p-right order uses the signed coordinate convention."""
    stack, mat = _build_grid_and_mat("configs/cigs_baseline.yaml")
    assert mat.V_bi_eff < 0.0
    assert mat.junction_polarity == -1.0
    assert mat.V_bi_bc == pytest.approx(-abs(stack.V_bi))
    assert poisson_right_boundary(mat, 0.0) == pytest.approx(-abs(stack.V_bi))
    assert poisson_right_boundary(mat, 0.4) == pytest.approx(-abs(stack.V_bi) + 0.4)
