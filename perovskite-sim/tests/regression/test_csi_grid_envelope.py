"""P1 guard and bounded characterization of the c-Si J-V grid envelope.

This does not certify a c-Si J-V curve. It verifies why the generic N_grid=100
J-V default is outside the preset's declared envelope, fails before time
integration, and can be bypassed only by an explicit diagnostic override. The
200/300/400 ladder below certifies grid geometry and finite quasi-neutral dark
initialisation only. Residual/current and voltage-domain certificates live in
``test_csi_quasi_fermi_convergence.py`` and must not be inferred from this
necessary grid guard alone.
"""
from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest
import yaml

import perovskite_sim.experiments.jv_sweep as jv_module
import perovskite_sim.experiments.steady_state as ss_module
from perovskite_sim.discretization.grid import (
    GridResolutionError,
    MAX_INTERFACE_CELL_DEBYE_RATIO,
    interface_grid_diagnostics,
    require_thick_layer_interface_resolution,
)
from perovskite_sim.experiments.jv_sweep import (
    build_electrical_grid,
    run_jv_sweep,
)
from perovskite_sim.experiments.steady_state import run_jv_sweep_ss, solve_voc_ss
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.solver.newton import solve_equilibrium


CONFIG = Path("configs/cSi_homojunction.yaml")
LADDER = (200, 300, 400)


def _declared_minimum() -> int:
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    return int(raw["simulation_hints"]["min_N_grid"])


def _first_base_spacing(stack, n_grid: int) -> tuple[np.ndarray, float]:
    x = build_electrical_grid(stack, n_grid)
    interface = stack.layers[0].thickness
    interface_index = int(np.argmin(np.abs(x - interface)))
    assert x[interface_index] == interface
    return x, float(x[interface_index + 1] - x[interface_index])


def _base_interface_diagnostic(stack, n_grid: int):
    x = build_electrical_grid(stack, n_grid)
    return next(
        item for item in interface_grid_diagnostics(x, stack)
        if item.side == "right" and item.layer_name == "p_base"
    )


def test_default_jv_grid_is_outside_csi_declared_envelope():
    stack = load_device_from_yaml(CONFIG)
    default_n = inspect.signature(run_jv_sweep).parameters["N_grid"].default
    declared_min = _declared_minimum()
    diagnostic = _base_interface_diagnostic(stack, int(default_n))

    assert default_n == 100
    assert default_n < declared_min
    assert diagnostic.cell_debye_ratio > MAX_INTERFACE_CELL_DEBYE_RATIO


def test_underresolved_transient_and_ss_jv_fail_before_integration(monkeypatch):
    stack = load_device_from_yaml(CONFIG)

    def integration_must_not_start(*args, **kwargs):
        raise AssertionError("time integration started before grid validation")

    monkeypatch.setattr(jv_module, "solve_illuminated_ss", integration_must_not_start)
    with pytest.raises(GridResolutionError, match="before integration") as transient:
        run_jv_sweep(stack, N_grid=100, n_points=2, V_max=0.1)
    assert transient.value.N_grid == 100
    assert transient.value.diagnostic.layer_name == "p_base"
    assert transient.value.diagnostic.cell_debye_ratio == pytest.approx(
        1.7654108069, rel=1.0e-9
    )

    with pytest.raises(GridResolutionError, match="before integration"):
        run_jv_sweep_ss(stack, N_grid=100, n_points=2, V_max=0.1)
    with pytest.raises(GridResolutionError, match="before integration"):
        solve_voc_ss(stack, N_grid=100, V_hi=0.1)


def test_underresolved_overrides_are_explicit_and_reach_both_drivers(monkeypatch):
    stack = load_device_from_yaml(CONFIG)

    class IntegrationReached(RuntimeError):
        pass

    def stop_at_integration(*args, **kwargs):
        raise IntegrationReached

    monkeypatch.setattr(jv_module, "solve_illuminated_ss", stop_at_integration)
    with pytest.warns(jv_module.JVCertificationWarning, match="diagnostic"):
        with pytest.raises(IntegrationReached):
            run_jv_sweep(
                stack,
                N_grid=100,
                n_points=2,
                V_max=0.1,
                allow_underresolved_grid=True,
                allow_unvalidated_driver=True,
                v_max_max_attempts=2,
            )

    monkeypatch.setattr(ss_module, "solve_steady_state", stop_at_integration)
    with pytest.warns(jv_module.JVCertificationWarning, match="diagnostic"):
        with pytest.raises(IntegrationReached):
            run_jv_sweep_ss(
                stack,
                N_grid=100,
                n_points=2,
                V_max=0.1,
                allow_underresolved_grid=True,
                allow_unvalidated_driver=True,
            )


def test_declared_csi_minimum_meets_debye_guard_and_has_finite_seed():
    """The ladder passes a necessary guard; it does not certify the junction."""
    stack = load_device_from_yaml(CONFIG)
    declared_min = _declared_minimum()
    assert LADDER[0] == declared_min

    diagnostics = []
    for n_grid in LADDER:
        x, first_base_dx = _first_base_spacing(stack, n_grid)
        diagnostic = _base_interface_diagnostic(stack, n_grid)
        assert first_base_dx == pytest.approx(diagnostic.cell_width, rel=1.0e-12)
        assert diagnostic.cell_debye_ratio <= MAX_INTERFACE_CELL_DEBYE_RATIO
        require_thick_layer_interface_resolution(x, stack, N_grid=n_grid)
        seed = solve_equilibrium(x, stack)
        assert np.all(np.isfinite(seed))
        diagnostics.append(diagnostic)

    ratios = tuple(item.cell_debye_ratio for item in diagnostics)
    widths = tuple(item.cell_width for item in diagnostics)
    assert ratios == pytest.approx(
        (0.8497957859, 0.5594569352, 0.4169728857), rel=1.0e-9
    )
    assert ratios[0] > ratios[1] > ratios[2]
    assert widths[0] > widths[1] > widths[2]


def test_thin_film_smoke_mesh_is_not_misrepresented_as_globally_certified():
    """The thick-layer guard must remain scoped, not become a mesh certificate."""
    stack = load_device_from_yaml("configs/ionmonger_benchmark.yaml")
    x = build_electrical_grid(stack, 12)
    diagnostics = require_thick_layer_interface_resolution(x, stack, N_grid=12)

    assert max(item.cell_debye_ratio for item in diagnostics) > 1.5
    assert max(item.layer_debye_span for item in diagnostics) < 1.0e3
