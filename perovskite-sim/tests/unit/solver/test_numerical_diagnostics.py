from types import SimpleNamespace

import numpy as np
import pytest

from perovskite_sim.physics.recombination import (
    interface_recombination,
    srh_recombination,
)
from perovskite_sim.physics.regularization import RHSRegularization
from perovskite_sim.solver import mol
from perovskite_sim.solver.numerical_diagnostics import (
    NumericalDiagnosticsError,
    NumericalDiagnosticsMonitor,
    NumericalDiagnosticsPolicy,
    StateLayout,
)


def _material(*, dual: bool = False, n_interfaces: int = 0):
    return SimpleNamespace(
        ni_sq=np.full(2, 1.0),
        N_A=np.zeros(2),
        N_D=np.zeros(2),
        P_ion0=np.array([50.0, 0.0]),
        P_ion0_neg=np.array([0.0, 60.0]) if dual else None,
        has_dual_ions=dual,
        N_iface_state=n_interfaces,
    )


def _single_ion_state() -> np.ndarray:
    return np.array([10.0, 20.0, 30.0, 40.0, 50.0, 0.0])


def _srh_rhs(_t, state, *_args, **_kwargs):
    n = state[:2]
    p = state[2:4]
    srh_recombination(
        n,
        p,
        1.0,
        tau_n=1.0,
        tau_p=1.0,
        n1=1.0,
        p1=1.0,
    )
    return np.zeros_like(state)


def test_layout_observes_dual_ion_and_interface_blocks_without_clipping():
    layout = StateLayout(
        n_nodes=2,
        has_dual_ions=True,
        n_interface_states=1,
        positive_ion_active=(True, False),
        negative_ion_active=(False, True),
    )
    monitor = NumericalDiagnosticsMonitor(layout)
    state = np.array([
        10.0,
        20.0,
        30.0,
        40.0,
        50.0,
        -2.0,
        -3.0,
        60.0,
        70.0,
        -4.0,
        80.0,
        90.0,
    ])
    original = state.copy()

    monitor.observe_trial_state(state)
    report = monitor.report()

    np.testing.assert_array_equal(state, original)
    assert report.negative_trial_evaluations == 1
    assert report.negative_trial_entries.positive_ion == 1
    assert report.negative_trial_entries.negative_ion == 1
    assert report.negative_trial_entries.interface_state == 1
    assert report.minimum_trial_density_m3.positive_ion_active == 50.0
    assert report.minimum_trial_density_m3.negative_ion_active == 60.0
    assert report.minimum_trial_density_m3.interface_state == -4.0


@pytest.mark.parametrize(
    ("layout", "size"),
    [
        (StateLayout(2), 5),
        (StateLayout(2, has_dual_ions=True), 7),
        (StateLayout(2, n_interface_states=1), 9),
        (StateLayout(2, has_dual_ions=True, n_interface_states=1), 11),
    ],
)
def test_layout_fails_closed_on_malformed_packed_state(layout, size):
    with pytest.raises(ValueError, match="expected one-dimensional layout"):
        layout.split(np.zeros(size))


def test_default_run_transient_records_trials_and_preserves_solver_result(
    monkeypatch,
):
    initial = _single_ion_state()
    negative_trial = initial.copy()
    negative_trial[1] = -5.0
    negative_trial[-1] = -7.0

    def fake_solve_ivp(rhs, *_args, **_kwargs):
        rhs(0.0, initial)
        rhs(0.5, negative_trial)
        return SimpleNamespace(
            success=True,
            y=initial[:, None].copy(),
            message="unchanged",
        )

    monkeypatch.setattr(mol, "assemble_rhs", _srh_rhs)
    monkeypatch.setattr(mol, "solve_ivp", fake_solve_ivp)

    solution = mol.run_transient(
        np.array([0.0, 1.0]),
        initial,
        (0.0, 1.0),
        np.array([1.0]),
        object(),
        mat=_material(),
    )

    assert solution.success is True
    assert solution.message == "unchanged"
    np.testing.assert_array_equal(solution.y[:, -1], initial)
    np.testing.assert_array_equal(negative_trial[[-1, 1]], [-7.0, -5.0])
    report = solution.numerical_diagnostics
    assert report.mode == "observe"
    assert report.trial_evaluations == 2
    assert report.negative_trial_evaluations == 1
    assert report.negative_trial_entries.n == 1
    assert report.negative_trial_entries.positive_ion == 1
    assert report.minimum_trial_density_m3.n == -5.0
    assert report.minimum_trial_density_m3.positive_ion_active == 50.0
    assert report.final_minimum_density_m3.n == 10.0
    assert report.final_minimum_density_m3.positive_ion_active == 50.0
    assert report.minimum_bulk_srh_denominator_s_m3 == 37.0
    assert report.would_pass_strict is True


def test_run_transient_forwards_and_records_explicit_rhs_regularization(
    monkeypatch,
):
    initial = _single_ion_state()
    policy = RHSRegularization(
        poole_frenkel_field_width_V_m=100.0,
        interface_density_width_m3=1.0e12,
        te_cap_relative_width=0.02,
    )
    observed = []

    def recording_rhs(*args, regularization=None, **kwargs):
        observed.append(regularization)
        return _srh_rhs(*args, **kwargs)

    def fake_solve_ivp(rhs, *_args, **_kwargs):
        rhs(0.0, initial)
        return SimpleNamespace(success=True, y=initial[:, None])

    monkeypatch.setattr(mol, "assemble_rhs", recording_rhs)
    monkeypatch.setattr(mol, "solve_ivp", fake_solve_ivp)

    solution = mol.run_transient(
        np.array([0.0, 1.0]),
        initial,
        (0.0, 1.0),
        np.array([1.0]),
        object(),
        mat=_material(),
        regularization=policy,
    )

    assert observed == [policy]
    assert solution.rhs_regularization is policy


def test_run_transient_default_does_not_change_assemble_rhs_call_shape(
    monkeypatch,
):
    initial = _single_ion_state()

    def legacy_positional_rhs(t, state, x, stack, mat, illuminated, voltage):
        return _srh_rhs(t, state, x, stack, mat, illuminated, voltage)

    def fake_solve_ivp(rhs, *_args, **_kwargs):
        rhs(0.0, initial)
        return SimpleNamespace(success=True, y=initial[:, None])

    monkeypatch.setattr(mol, "assemble_rhs", legacy_positional_rhs)
    monkeypatch.setattr(mol, "solve_ivp", fake_solve_ivp)

    solution = mol.run_transient(
        np.array([0.0, 1.0]),
        initial,
        (0.0, 1.0),
        np.array([1.0]),
        object(),
        mat=_material(),
    )

    assert solution.rhs_regularization == RHSRegularization()


def test_observational_mode_records_nonfinite_rhs_without_changing_success(
    monkeypatch,
):
    initial = _single_ion_state()

    def nonfinite_rhs(*args, **kwargs):
        value = _srh_rhs(*args, **kwargs)
        value[0] = np.nan
        return value

    def fake_solve_ivp(rhs, *_args, **_kwargs):
        rhs(0.0, initial)
        return SimpleNamespace(success=True, y=initial[:, None])

    monkeypatch.setattr(mol, "assemble_rhs", nonfinite_rhs)
    monkeypatch.setattr(mol, "solve_ivp", fake_solve_ivp)

    solution = mol.run_transient(
        np.array([0.0, 1.0]),
        initial,
        (0.0, 1.0),
        np.array([1.0]),
        object(),
        mat=_material(),
    )

    assert solution.success is True
    assert solution.numerical_diagnostics.nonfinite_rhs_evaluations == 1
    assert "nonfinite_rhs" in solution.numerical_diagnostics.violations


def test_research_strict_rejects_nonpositive_terminal_density(monkeypatch):
    initial = _single_ion_state()
    terminal = initial.copy()
    terminal[0] = 0.0

    def fake_solve_ivp(rhs, *_args, **_kwargs):
        rhs(0.0, initial)
        return SimpleNamespace(success=True, y=terminal[:, None])

    monkeypatch.setattr(mol, "assemble_rhs", _srh_rhs)
    monkeypatch.setattr(mol, "solve_ivp", fake_solve_ivp)
    policy = NumericalDiagnosticsPolicy.research_strict(
        bulk_srh_denominator_floor_s_m3=1.0
    )

    with pytest.raises(
        NumericalDiagnosticsError, match="terminal_n_not_above_floor"
    ) as captured:
        mol.run_transient(
            np.array([0.0, 1.0]),
            initial,
            (0.0, 1.0),
            np.array([1.0]),
            object(),
            mat=_material(),
            numerical_diagnostics=policy,
        )

    assert captured.value.report.final_minimum_density_m3.n == 0.0
    assert captured.value.report.would_pass_strict is False


def test_research_strict_rejects_nonfinite_rhs_immediately(monkeypatch):
    initial = _single_ion_state()

    def nonfinite_rhs(*args, **kwargs):
        value = _srh_rhs(*args, **kwargs)
        value[0] = np.inf
        return value

    def fake_solve_ivp(rhs, *_args, **_kwargs):
        rhs(0.0, initial)
        pytest.fail("strict RHS evaluation should have raised")

    monkeypatch.setattr(mol, "assemble_rhs", nonfinite_rhs)
    monkeypatch.setattr(mol, "solve_ivp", fake_solve_ivp)
    policy = NumericalDiagnosticsPolicy.research_strict(
        bulk_srh_denominator_floor_s_m3=1.0
    )

    with pytest.raises(NumericalDiagnosticsError, match="non-finite RHS") as captured:
        mol.run_transient(
            np.array([0.0, 1.0]),
            initial,
            (0.0, 1.0),
            np.array([1.0]),
            object(),
            mat=_material(),
            numerical_diagnostics=policy,
        )

    assert captured.value.report.nonfinite_rhs_evaluations == 1


def test_research_strict_rejects_declared_near_zero_srh_denominator(
    monkeypatch,
):
    initial = _single_ion_state()

    def fake_solve_ivp(rhs, *_args, **_kwargs):
        rhs(0.0, initial)
        pytest.fail("strict SRH observation should have raised")

    monkeypatch.setattr(mol, "assemble_rhs", _srh_rhs)
    monkeypatch.setattr(mol, "solve_ivp", fake_solve_ivp)
    policy = NumericalDiagnosticsPolicy.research_strict(
        bulk_srh_denominator_floor_s_m3=50.0
    )

    with pytest.raises(
        NumericalDiagnosticsError, match="bulk SRH denominator"
    ) as captured:
        mol.run_transient(
            np.array([0.0, 1.0]),
            initial,
            (0.0, 1.0),
            np.array([1.0]),
            object(),
            mat=_material(),
            numerical_diagnostics=policy,
        )

    assert (
        captured.value.report.minimum_bulk_srh_denominator_s_m3
        == pytest.approx(42.0)
    )


def test_research_strict_uses_separate_interface_denominator_units(monkeypatch):
    initial = _single_ion_state()

    def interface_rhs(*args, **kwargs):
        value = _srh_rhs(*args, **kwargs)
        interface_recombination(
            n=2.0,
            p=3.0,
            ni_sq=1.0,
            n1=5.0,
            p1=7.0,
            v_n=11.0,
            v_p=13.0,
        )
        return value

    def fake_solve_ivp(rhs, *_args, **_kwargs):
        rhs(0.0, initial)
        pytest.fail("strict interface SRH observation should have raised")

    monkeypatch.setattr(mol, "assemble_rhs", interface_rhs)
    monkeypatch.setattr(mol, "solve_ivp", fake_solve_ivp)
    policy = NumericalDiagnosticsPolicy.research_strict(
        bulk_srh_denominator_floor_s_m3=1.0,
        interface_srh_denominator_floor_s_m4=2.0,
    )

    with pytest.raises(
        NumericalDiagnosticsError, match="interface SRH denominator"
    ) as captured:
        mol.run_transient(
            np.array([0.0, 1.0]),
            initial,
            (0.0, 1.0),
            np.array([1.0]),
            object(),
            mat=_material(),
            numerical_diagnostics=policy,
        )

    assert (
        captured.value.report.minimum_interface_srh_denominator_s_m4
        == pytest.approx((2.0 + 5.0) / 13.0 + (3.0 + 7.0) / 11.0)
    )


@pytest.mark.parametrize(
    ("layout", "terminal", "violation"),
    [
        (
            StateLayout(
                1,
                has_dual_ions=True,
                positive_ion_active=(True,),
                negative_ion_active=(True,),
            ),
            np.array([1.0, 1.0, 1.0, 0.0]),
            "terminal_P_neg_active_not_above_floor",
        ),
        (
            StateLayout(
                1,
                n_interface_states=1,
                positive_ion_active=(True,),
            ),
            np.array([1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0]),
            "terminal_interface_state_not_above_floor",
        ),
    ],
)
def test_research_strict_terminal_gate_covers_dual_ion_and_interface_layouts(
    layout, terminal, violation
):
    policy = NumericalDiagnosticsPolicy.research_strict(
        bulk_srh_denominator_floor_s_m3=1.0
    )
    monitor = NumericalDiagnosticsMonitor(layout, policy)
    monitor.observe_trial_state(terminal)
    monitor.observe_rhs(np.zeros(layout.expected_size))
    monitor.observe_srh_denominator("bulk", np.array([2.0]))

    with pytest.raises(NumericalDiagnosticsError, match=violation):
        monitor.finalize(terminal, solver_success=True)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"terminal_density_floor_m3": -1.0},
        {"bulk_srh_denominator_floor_s_m3": np.nan},
        {"interface_srh_denominator_floor_s_m4": np.inf},
        {"terminal_density_floor_m3": True},
    ],
)
def test_policy_rejects_invalid_thresholds(kwargs):
    with pytest.raises(ValueError, match="finite and non-negative"):
        NumericalDiagnosticsPolicy(**kwargs)


def test_real_dark_transient_reports_exact_terminal_block_minima():
    from perovskite_sim.discretization.grid import Layer, multilayer_grid
    from perovskite_sim.models.config_loader import load_device_from_yaml
    from perovskite_sim.solver.newton import solve_equilibrium

    stack = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    x = multilayer_grid([Layer(layer.thickness, 5) for layer in stack.layers])
    mat = mol.build_material_arrays(x, stack)
    initial = solve_equilibrium(x, stack)

    solution = mol.run_transient(
        x,
        initial,
        (0.0, 1.0e-10),
        np.array([1.0e-10]),
        stack,
        illuminated=False,
        mat=mat,
    )

    assert solution.success
    state = mol.StateVec.unpack(solution.y[:, -1], len(x))
    active_ions = mat.P_ion0 > 0.0
    report = solution.numerical_diagnostics
    assert report.trial_evaluations > 0
    assert report.nonfinite_rhs_evaluations == 0
    assert report.minimum_bulk_srh_denominator_s_m3 is not None
    assert report.final_minimum_density_m3.n == np.min(state.n)
    assert report.final_minimum_density_m3.p == np.min(state.p)
    assert report.final_minimum_density_m3.positive_ion_active == np.min(
        state.P[active_ions]
    )
