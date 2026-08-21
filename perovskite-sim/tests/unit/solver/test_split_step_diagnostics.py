from types import SimpleNamespace

import numpy as np
import pytest

from perovskite_sim.physics import ion_migration
from perovskite_sim.physics.generation import dual_cell_integral
from perovskite_sim.solver import mol
from perovskite_sim.solver.numerical_diagnostics import (
    SplitStepDiagnosticsError,
    SplitStepDiagnosticsMonitor,
    SplitStepDiagnosticsPolicy,
)


def _material(*, dual: bool = False) -> SimpleNamespace:
    n = 3
    return SimpleNamespace(
        N_iface_state=0,
        has_selective_contacts=False,
        n_L=10.0,
        n_R=10.0,
        p_L=10.0,
        p_R=10.0,
        has_dual_ions=dual,
        P_ion0=np.ones(n),
        P_lim_node=np.full(n, 10.0),
        P_ion0_neg=np.full(n, 2.0) if dual else None,
        P_lim_neg_node=np.full(n, 10.0) if dual else None,
        N_A=np.zeros(n),
        N_D=np.zeros(n),
        poisson_factor=object(),
        D_ion_face=np.zeros(n - 1),
        D_ion_neg_face=np.zeros(n - 1) if dual else None,
        V_T_device=0.025,
        P_lim_face=np.full(n - 1, 10.0),
        P_lim_neg_face=np.full(n - 1, 10.0) if dual else None,
        ion_steric_diffusion_only=False,
        ion_steric_shared_site=True,
    )


def _state(*, dual: bool = False, positive=None, negative=None) -> np.ndarray:
    positive = np.ones(3) if positive is None else np.asarray(positive, dtype=float)
    if dual:
        negative = (
            np.full(3, 2.0)
            if negative is None
            else np.asarray(negative, dtype=float)
        )
    return mol.StateVec.pack(
        np.full(3, 10.0),
        np.full(3, 10.0),
        positive,
        negative,
    )


def _patch_split_physics(monkeypatch, *, terminal, trial=None, carrier_ok=True):
    terminal = np.asarray(terminal, dtype=float)

    def fake_solve_ivp(fun, _span, state, **_kwargs):
        if trial is not None:
            fun(0.5, np.asarray(trial, dtype=float))
        return SimpleNamespace(success=True, y=terminal[:, None])

    def fake_run_transient(_x, state, *_args, **_kwargs):
        return SimpleNamespace(
            success=carrier_ok,
            y=np.asarray(state, dtype=float)[:, None],
        )

    monkeypatch.setattr(mol, "solve_ivp", fake_solve_ivp)
    monkeypatch.setattr(mol, "run_transient", fake_run_transient)
    monkeypatch.setattr(
        mol,
        "solve_poisson_prefactored",
        lambda _factor, rho, **_kwargs: np.zeros_like(rho),
    )
    monkeypatch.setattr(mol, "poisson_right_boundary", lambda _mat, _v: 0.0)
    monkeypatch.setattr(
        mol,
        "ion_continuity_rhs",
        lambda _x, _phi, density, *_args, **_kwargs: np.zeros_like(density),
    )
    monkeypatch.setattr(
        ion_migration,
        "ion_continuity_rhs_neg",
        lambda _x, _phi, density, *_args, **_kwargs: np.zeros_like(density),
    )


def test_split_step_default_keeps_two_tuple_and_historical_terminal_clip(
    monkeypatch,
):
    mat = _material()
    initial = _state()
    raw_terminal = np.array([-1.0, 4.0, 12.0])
    _patch_split_physics(monkeypatch, terminal=raw_terminal)

    result = mol.split_step(
        np.array([0.0, 0.5, 1.0]),
        initial,
        0.1,
        object(),
        mat=mat,
    )
    assert isinstance(result, tuple)
    assert len(result) == 2
    state, success = result
    assert success is True
    np.testing.assert_array_equal(
        mol.StateVec.unpack(state, 3).P,
        np.array([0.0, 4.0, 10.0]),
    )


def test_inventory_uses_solver_dual_cell_weights_on_nonuniform_grid():
    x = np.array([0.0, 0.25, 1.0])
    initial = np.ones(3)
    terminal = np.array([2.0, 0.5, 1.0])
    assert dual_cell_integral(x, initial) == pytest.approx(1.5)
    assert dual_cell_integral(x, terminal) == pytest.approx(1.5)
    assert np.trapezoid(initial, x) != pytest.approx(np.trapezoid(terminal, x))

    monitor = SplitStepDiagnosticsMonitor(
        x,
        initial,
        np.full(3, 10.0),
        full_initial_state=_state(positive=initial),
        policy=SplitStepDiagnosticsPolicy.research_strict(
            maximum_relative_inventory_drift=0.0
        ),
    )
    monitor.observe_raw_terminal(terminal, solver_success=True)
    monitor.observe_projected_terminal(terminal)
    final = _state(positive=terminal)
    report = monitor.finalize(
        terminal,
        None,
        full_final_state=final,
        carrier_reequilibration_success=True,
    )
    assert report.positive_ion.inventory.raw_terminal_relative_drift == 0.0
    assert report.positive_ion.inventory.final_relative_drift == 0.0
    assert report.would_pass_strict is True


@pytest.mark.parametrize(
    ("state_index", "violation", "count_field"),
    [
        (0, "final_electron_density_nonpositive", "final_electron_nonpositive_entries"),
        (3, "final_hole_density_nonpositive", "final_hole_nonpositive_entries"),
    ],
)
def test_split_monitor_reports_and_strictly_rejects_nonpositive_carriers(
    state_index, violation, count_field
):
    x = np.array([0.0, 0.5, 1.0])
    final = _state()
    final[state_index] = -1.0

    observing = SplitStepDiagnosticsMonitor(
        x,
        np.ones(3),
        np.full(3, 10.0),
        full_initial_state=_state(),
    )
    observing.observe_raw_terminal(np.ones(3), solver_success=True)
    observing.observe_projected_terminal(np.ones(3))
    report = observing.finalize(
        np.ones(3),
        None,
        full_final_state=final,
        carrier_reequilibration_success=True,
    )
    assert violation in report.violations
    assert getattr(report, count_field) == 1
    assert report.would_pass_strict is False

    strict = SplitStepDiagnosticsMonitor(
        x,
        np.ones(3),
        np.full(3, 10.0),
        full_initial_state=_state(),
        policy=SplitStepDiagnosticsPolicy.research_strict(),
    )
    strict.observe_raw_terminal(np.ones(3), solver_success=True)
    strict.observe_projected_terminal(np.ones(3))
    with pytest.raises(SplitStepDiagnosticsError, match=violation):
        strict.finalize(
            np.ones(3),
            None,
            full_final_state=final,
            carrier_reequilibration_success=True,
        )


def test_split_monitor_reports_and_strictly_rejects_nonpositive_interface_state():
    x = np.array([0.0, 0.5, 1.0])
    interface = np.ones(4)
    initial = mol.StateVec.pack(
        np.full(3, 10.0), np.full(3, 10.0), np.ones(3), iface_state=interface
    )
    final = initial.copy()
    final[-1] = 0.0
    monitor = SplitStepDiagnosticsMonitor(
        x,
        np.ones(3),
        np.full(3, 10.0),
        full_initial_state=initial,
        policy=SplitStepDiagnosticsPolicy.research_strict(),
    )
    monitor.observe_raw_terminal(np.ones(3), solver_success=True)
    monitor.observe_projected_terminal(np.ones(3))
    with pytest.raises(
        SplitStepDiagnosticsError,
        match="final_interface_state_density_nonpositive",
    ) as captured:
        monitor.finalize(
            np.ones(3),
            None,
            full_final_state=final,
            carrier_reequilibration_success=True,
        )
    report = captured.value.report
    assert report.final_interface_state_nonpositive_entries == 1
    assert report.final_interface_state_minimum_density_m3 == 0.0

@pytest.mark.parametrize("dual", [False, True], ids=["single-ion", "dual-ion"])
def test_strict_split_step_certifies_species_inventory_conservation(
    monkeypatch, dual
):
    mat = _material(dual=dual)
    initial = _state(dual=dual)
    positive_terminal = np.array([0.5, 1.5, 1.0])
    negative_terminal = np.array([1.5, 2.5, 2.0])
    terminal = (
        np.concatenate([positive_terminal, negative_terminal])
        if dual
        else positive_terminal
    )
    _patch_split_physics(monkeypatch, terminal=terminal, trial=terminal)
    policy = SplitStepDiagnosticsPolicy.research_strict(
        maximum_relative_inventory_drift=1.0e-12
    )

    state, success, report = mol.split_step(
        np.array([0.0, 0.5, 1.0]),
        initial,
        0.1,
        object(),
        mat=mat,
        split_diagnostics=policy,
        return_diagnostics=True,
    )

    assert success is True
    assert report.would_pass_strict is True
    assert report.projection_events == 0
    assert report.positive_ion.inventory.raw_terminal_relative_drift == 0.0
    assert report.positive_ion.inventory.final_relative_drift == 0.0
    if dual:
        assert report.negative_ion is not None
        assert report.negative_ion.inventory.raw_terminal_relative_drift == 0.0
        assert report.negative_ion.inventory.final_relative_drift == 0.0
    np.testing.assert_array_equal(
        mol.StateVec.unpack(state, 3).P,
        positive_terminal,
    )


def test_negative_implicit_trial_is_recorded_and_rejection_is_explicit(
    monkeypatch,
):
    mat = _material()
    initial = _state()
    trial = np.array([-0.25, 1.0, 1.0])
    terminal = np.ones(3)
    _patch_split_physics(monkeypatch, terminal=terminal, trial=trial)

    with pytest.warns(RuntimeWarning, match="clipped to zero"):
        _, _, report = mol.split_step(
            np.array([0.0, 0.5, 1.0]),
            initial,
            0.1,
            object(),
            mat=mat,
            split_diagnostics=SplitStepDiagnosticsPolicy.research_strict(),
            return_diagnostics=True,
        )
    assert report.negative_trial_evaluations == 1
    assert report.positive_ion.negative_trial_entries == 1
    assert report.reject_negative_trial_states is False
    assert report.would_pass_strict is True

    rejecting = SplitStepDiagnosticsPolicy.research_strict(
        reject_negative_trial_states=True
    )
    with pytest.raises(
        SplitStepDiagnosticsError, match="negative_ion_trial_state"
    ) as captured:
        mol.split_step(
            np.array([0.0, 0.5, 1.0]),
            initial,
            0.1,
            object(),
            mat=mat,
            split_diagnostics=rejecting,
        )
    assert captured.value.report.positive_ion.negative_trial_entries == 1


def test_strict_split_step_always_rejects_nonfinite_implicit_trial(
    monkeypatch,
):
    trial = np.array([np.nan, 1.0, 1.0])
    _patch_split_physics(monkeypatch, terminal=np.ones(3), trial=trial)

    with pytest.raises(
        SplitStepDiagnosticsError, match="nonfinite_ion_trial_state"
    ) as captured:
        mol.split_step(
            np.array([0.0, 0.5, 1.0]),
            _state(),
            0.1,
            object(),
            mat=_material(),
            split_diagnostics=SplitStepDiagnosticsPolicy.research_strict(),
        )

    assert captured.value.report.nonfinite_trial_evaluations == 1
    assert captured.value.report.positive_ion.nonfinite_trial_entries == 1


def test_overlimit_implicit_trial_is_recorded_and_optionally_rejected(
    monkeypatch,
):
    trial = np.array([11.0, 1.0, 1.0])
    _patch_split_physics(monkeypatch, terminal=np.ones(3), trial=trial)

    _, _, report = mol.split_step(
        np.array([0.0, 0.5, 1.0]),
        _state(),
        0.1,
        object(),
        mat=_material(),
        split_diagnostics=SplitStepDiagnosticsPolicy.research_strict(),
        return_diagnostics=True,
    )
    assert report.overlimit_trial_evaluations == 1
    assert report.positive_ion.overlimit_trial_entries == 1
    assert report.would_pass_strict is True

    with pytest.raises(
        SplitStepDiagnosticsError, match="overlimit_ion_trial_state"
    ):
        mol.split_step(
            np.array([0.0, 0.5, 1.0]),
            _state(),
            0.1,
            object(),
            mat=_material(),
            split_diagnostics=SplitStepDiagnosticsPolicy.research_strict(
                reject_overlimit_trial_states=True
            ),
        )


def test_strict_split_step_rejects_negative_initial_state_before_projection(
    monkeypatch,
):
    solve_called = False

    def forbidden_solver(*_args, **_kwargs):
        nonlocal solve_called
        solve_called = True
        raise AssertionError("ion solver must not see a clipped strict initial state")

    monkeypatch.setattr(mol, "solve_ivp", forbidden_solver)
    initial = _state(positive=np.array([-1.0, 1.0, 1.0]))

    with pytest.raises(
        SplitStepDiagnosticsError, match="positive_ion_initial_negative"
    ) as captured:
        mol.split_step(
            np.array([0.0, 0.5, 1.0]),
            initial,
            0.1,
            object(),
            mat=_material(),
            split_diagnostics=SplitStepDiagnosticsPolicy.research_strict(),
        )

    assert solve_called is False
    assert captured.value.report.projection_events == 1
    assert captured.value.report.positive_ion.initial_projection_entries == 1


@pytest.mark.parametrize(
    ("terminal", "violation"),
    [
        (np.array([-1.0, 1.0, 1.0]), "positive_ion_raw_terminal_negative"),
        (np.array([11.0, 1.0, 1.0]), "positive_ion_raw_terminal_overlimit"),
        (np.array([np.nan, 1.0, 1.0]), "positive_ion_raw_terminal_nonfinite"),
    ],
)
def test_strict_split_step_rejects_raw_terminal_before_clip(
    monkeypatch, terminal, violation
):
    _patch_split_physics(monkeypatch, terminal=terminal)

    with pytest.raises(SplitStepDiagnosticsError, match=violation) as captured:
        mol.split_step(
            np.array([0.0, 0.5, 1.0]),
            _state(),
            0.1,
            object(),
            mat=_material(),
            split_diagnostics=SplitStepDiagnosticsPolicy.research_strict(),
        )

    report = captured.value.report
    assert report.ion_solver_success is True
    if np.all(np.isfinite(terminal)):
        assert report.positive_ion.terminal_projection_entries == 1


def test_strict_dual_ion_terminal_gate_reports_each_raw_species(
    monkeypatch,
):
    terminal = np.array([1.0, 1.0, 11.0, 2.0, -1.0, 2.0])
    _patch_split_physics(monkeypatch, terminal=terminal)

    with pytest.raises(SplitStepDiagnosticsError) as captured:
        mol.split_step(
            np.array([0.0, 0.5, 1.0]),
            _state(dual=True),
            0.1,
            object(),
            mat=_material(dual=True),
            split_diagnostics=SplitStepDiagnosticsPolicy.research_strict(),
        )

    report = captured.value.report
    assert "positive_ion_raw_terminal_overlimit" in report.violations
    assert "negative_ion_raw_terminal_negative" in report.violations
    assert report.positive_ion.terminal_projection_entries == 1
    assert report.negative_ion is not None
    assert report.negative_ion.terminal_projection_entries == 1


def test_strict_split_step_rejects_raw_inventory_drift(monkeypatch):
    terminal = np.array([1.0, 2.0, 1.0])
    _patch_split_physics(monkeypatch, terminal=terminal)

    with pytest.raises(
        SplitStepDiagnosticsError,
        match="positive_ion_raw_terminal_inventory_drift",
    ) as captured:
        mol.split_step(
            np.array([0.0, 0.5, 1.0]),
            _state(),
            0.1,
            object(),
            mat=_material(),
            split_diagnostics=SplitStepDiagnosticsPolicy.research_strict(
                maximum_relative_inventory_drift=1.0e-3
            ),
        )

    assert (
        captured.value.report.positive_ion.inventory.raw_terminal_relative_drift
        == pytest.approx(1.0 / 3.0)
    )


def test_observe_mode_discloses_projection_and_inventory_change(monkeypatch):
    terminal = np.array([-1.0, 1.0, 12.0])
    _patch_split_physics(monkeypatch, terminal=terminal)

    _, success, report = mol.split_step(
        np.array([0.0, 0.5, 1.0]),
        _state(),
        0.1,
        object(),
        mat=_material(),
        return_diagnostics=True,
    )

    assert success is True
    assert report.mode == "observe"
    assert report.projection_events == 1
    assert report.positive_ion.raw_terminal_negative_entries == 1
    assert report.positive_ion.raw_terminal_overlimit_entries == 1
    assert report.positive_ion.terminal_projection_entries == 2
    assert report.positive_ion.inventory.raw_terminal_relative_drift > 0.0
    assert report.positive_ion.inventory.projected_terminal_relative_drift > 0.0
    assert report.would_pass_strict is False


def test_strict_split_step_rejects_carrier_fallback(monkeypatch):
    _patch_split_physics(
        monkeypatch,
        terminal=np.ones(3),
        carrier_ok=False,
    )

    with pytest.raises(
        SplitStepDiagnosticsError,
        match="carrier_reequilibration_not_successful",
    ):
        mol.split_step(
            np.array([0.0, 0.5, 1.0]),
            _state(),
            0.1,
            object(),
            mat=_material(),
            split_diagnostics=SplitStepDiagnosticsPolicy.research_strict(),
        )


def test_real_dual_ion_split_step_passes_strict_inventory_gate():
    from dataclasses import replace

    from perovskite_sim.discretization.grid import Layer, multilayer_grid
    from perovskite_sim.models.config_loader import load_device_from_yaml
    from perovskite_sim.solver.newton import solve_equilibrium

    stack = load_device_from_yaml("configs/ionmonger_benchmark.yaml")
    layers = []
    for layer in stack.layers:
        params = layer.params
        if layer.role == "absorber":
            params = replace(
                params,
                D_ion_neg=3.2e-18,
                P0_neg=1.6e25,
                P_lim_neg=1.6e27,
            )
        layers.append(replace(layer, params=params))
    stack = replace(stack, layers=tuple(layers))
    x = multilayer_grid([Layer(layer.thickness, 3) for layer in stack.layers])
    mat = mol.build_material_arrays(x, stack)
    initial = solve_equilibrium(x, stack)

    _, success, report = mol.split_step(
        x,
        initial,
        1.0e-8,
        stack,
        mat=mat,
        split_diagnostics=SplitStepDiagnosticsPolicy.research_strict(
            maximum_relative_inventory_drift=1.0e-7
        ),
        return_diagnostics=True,
    )

    assert success is True
    assert report.would_pass_strict is True
    assert report.trial_evaluations > 0
    assert report.negative_ion is not None
    assert report.positive_ion.inventory.raw_terminal_relative_drift < 1.0e-12
    assert report.negative_ion.inventory.raw_terminal_relative_drift < 1.0e-12
    assert report.positive_ion.inventory.final_relative_drift < 1.0e-12
    assert report.negative_ion.inventory.final_relative_drift < 1.0e-12


@pytest.mark.parametrize(
    "kwargs",
    [
        {"maximum_relative_inventory_drift": -1.0},
        {"maximum_relative_inventory_drift": np.inf},
        {"maximum_relative_inventory_drift": True},
        {"reject_negative_trial_states": 1},
        {"reject_overlimit_trial_states": "yes"},
    ],
)
def test_split_step_policy_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        SplitStepDiagnosticsPolicy(**kwargs)
