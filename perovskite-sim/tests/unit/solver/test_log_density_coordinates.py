from types import SimpleNamespace

import numpy as np
import pytest

from perovskite_sim.solver import mol
from perovskite_sim.solver.numerical_diagnostics import (
    LogDensityCoordinateError,
    LogDensityCoordinateTransform,
    StateLayout,
)
from perovskite_sim.solver.tolerances import (
    ComponentwiseAtol,
    build_componentwise_atol_1d,
)


def _material(*, dual: bool = False, n_interfaces: int = 0):
    return SimpleNamespace(
        ni_sq=np.array([100.0, 400.0]),
        N_A=np.zeros(2),
        N_D=np.zeros(2),
        P_ion0=np.array([50.0, 0.0]),
        P_ion0_neg=np.array([0.0, 60.0]) if dual else None,
        has_dual_ions=dual,
        N_iface_state=n_interfaces,
    )


def _single_ion_state() -> np.ndarray:
    return np.array([10.0, 20.0, 30.0, 40.0, 50.0, 0.0])


def _dual_interface_layout() -> StateLayout:
    return StateLayout(
        n_nodes=2,
        has_dual_ions=True,
        n_interface_states=1,
        positive_ion_active=(True, False),
        negative_ion_active=(False, True),
    )


def _dual_interface_state() -> np.ndarray:
    return np.array(
        [10.0, 20.0, 30.0, 40.0, 50.0, 0.0, 0.0, 60.0,
         70.0, 80.0, 90.0, 100.0]
    )


def test_hybrid_coordinates_round_trip_active_densities_and_structural_zeros():
    layout = _dual_interface_layout()
    initial = _dual_interface_state()
    transform = LogDensityCoordinateTransform(
        layout,
        initial,
        physical_atol=np.arange(1.0, initial.size + 1.0),
        physical_rtol=1.0e-4,
    )

    coordinates = transform.initial_coordinates()
    physical = transform.to_physical(coordinates)

    np.testing.assert_array_equal(physical, initial)
    assert np.count_nonzero(transform.active_mask) == 10
    np.testing.assert_array_equal(physical[~transform.active_mask], 0.0)
    assert np.all(physical[transform.active_mask] > 0.0)
    report = transform.report()
    assert report.active_density_components == 10
    assert report.inactive_structural_ion_components == 2
    assert report.inactive_coordinate_atol_min_m3 is not None


@pytest.mark.parametrize(("index", "value"), [(0, 0.0), (3, -1.0), (4, 0.0)])
def test_nonpositive_active_initial_density_fails_before_solver(
    monkeypatch, index, value
):
    initial = _single_ion_state()
    initial[index] = value

    def solver_must_not_run(*_args, **_kwargs):
        pytest.fail("solve_ivp must not see a non-positive active initial state")

    monkeypatch.setattr(mol, "solve_ivp", solver_must_not_run)

    with pytest.raises(
        LogDensityCoordinateError, match="strictly positive"
    ):
        mol.run_transient(
            np.array([0.0, 1.0]),
            initial,
            (0.0, 1.0),
            np.array([1.0]),
            object(),
            mat=_material(),
            state_coordinates="research_log_density",
        )


@pytest.mark.parametrize(
    ("coordinate", "message"),
    [(np.inf, "non-finite"), (1.0e3, "overflows"), (-1.0e3, "underflows")],
)
def test_nonrepresentable_log_coordinate_trials_fail_closed(coordinate, message):
    initial = _single_ion_state()
    transform = LogDensityCoordinateTransform(
        StateLayout(
            n_nodes=2,
            positive_ion_active=(True, False),
        ),
        initial,
        physical_atol=1.0e-6,
        physical_rtol=1.0e-4,
    )
    trial = transform.initial_coordinates()
    trial[0] = coordinate

    with pytest.raises(LogDensityCoordinateError, match=message):
        transform.to_physical(trial)


def test_run_transient_keeps_rhs_output_and_diagnostics_in_physical_density(
    monkeypatch,
):
    initial = _single_ion_state()
    physical_rhs = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 0.0])
    observed_physical_states = []
    observed_coordinate_rhs = []
    solver_calls = []

    def recording_rhs(_t, state, *_args, **_kwargs):
        observed_physical_states.append(state.copy())
        return physical_rhs.copy()

    def fake_solve_ivp(rhs, _t_span, solver_y0, **kwargs):
        solver_calls.append((solver_y0.copy(), kwargs))
        trial = solver_y0.copy()
        trial[:5] = np.log(np.array([1.1, 0.9, 1.2, 0.8, 1.05]))
        observed_coordinate_rhs.append(rhs(0.5, trial))
        return SimpleNamespace(
            success=True,
            y=np.column_stack([solver_y0, trial]),
            nfev=1,
            message="coordinate solve",
        )

    monkeypatch.setattr(mol, "assemble_rhs", recording_rhs)
    monkeypatch.setattr(mol, "solve_ivp", fake_solve_ivp)

    solution = mol.run_transient(
        np.array([0.0, 1.0]),
        initial,
        (0.0, 1.0),
        np.array([1.0]),
        object(),
        mat=_material(),
        rtol=1.0e-4,
        atol=1.0e-6,
        state_coordinates="research_log_density",
    )

    solver_y0, solver_kwargs = solver_calls[0]
    np.testing.assert_array_equal(solver_y0, 0.0)
    assert solver_kwargs["rtol"] == 100.0 * np.finfo(float).eps
    assert np.ndim(solver_kwargs["atol"]) == 1

    expected_terminal = initial.copy()
    expected_terminal[:5] *= np.array([1.1, 0.9, 1.2, 0.8, 1.05])
    np.testing.assert_allclose(observed_physical_states[0], expected_terminal)
    np.testing.assert_allclose(solution.y[:, 0], initial)
    np.testing.assert_allclose(solution.y[:, -1], expected_terminal)
    np.testing.assert_allclose(
        observed_coordinate_rhs[0][:5],
        physical_rhs[:5] / expected_terminal[:5],
    )
    assert observed_coordinate_rhs[0][-1] == physical_rhs[-1]
    assert solution.numerical_diagnostics.minimum_trial_density_m3.n == pytest.approx(
        11.0
    )
    assert solution.state_coordinate_report.mode == "research_log_density"


@pytest.mark.parametrize("explicit_mode", [False, True])
def test_density_coordinate_path_preserves_solver_arguments_and_result(
    monkeypatch, explicit_mode
):
    initial = _single_ion_state()
    captured = []

    def fake_solve_ivp(_rhs, _t_span, solver_y0, **kwargs):
        captured.append((solver_y0, kwargs))
        return SimpleNamespace(
            success=True,
            y=solver_y0[:, None],
            message="legacy density result",
        )

    monkeypatch.setattr(mol, "solve_ivp", fake_solve_ivp)
    coordinate_option = {"state_coordinates": "density"} if explicit_mode else {}
    solution = mol.run_transient(
        np.array([0.0, 1.0]),
        initial,
        (0.0, 1.0),
        np.array([1.0]),
        object(),
        mat=_material(),
        rtol=2.0e-4,
        atol=7.0,
        **coordinate_option,
    )

    solver_y0, solver_kwargs = captured[0]
    assert solver_y0 is initial
    assert solver_kwargs["rtol"] == 2.0e-4
    assert solver_kwargs["atol"] == 7.0
    np.testing.assert_array_equal(solution.y[:, -1], initial)
    assert solution.message == "legacy density result"
    assert not hasattr(solution, "state_coordinate_report")


def test_componentwise_physical_error_budget_maps_exactly_at_reference():
    initial = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 7.0, 8.0, 60.0])
    material = _material(dual=True)
    policy = ComponentwiseAtol(
        carrier_fraction=1.0e-3,
        ion_fraction=2.0e-3,
        minimum_atol=1.0e-2,
    )
    physical_rtol = 2.0e-4
    physical_atol = build_componentwise_atol_1d(
        policy,
        y0=initial,
        ni_sq=material.ni_sq,
        N_A=material.N_A,
        N_D=material.N_D,
        P_ion0=material.P_ion0,
        has_dual_ions=True,
        P_ion0_neg=material.P_ion0_neg,
    )
    transform = LogDensityCoordinateTransform(
        StateLayout(
            n_nodes=2,
            has_dual_ions=True,
            positive_ion_active=(True, False),
            negative_ion_active=(False, True),
        ),
        initial,
        physical_atol,
        physical_rtol,
    )

    expected_active_atol = np.log1p(
        physical_rtol
        + physical_atol[transform.active_mask]
        / initial[transform.active_mask]
    )
    expected_inactive_atol = (
        physical_atol[~transform.active_mask]
        + physical_rtol * np.abs(initial[~transform.active_mask])
    )
    np.testing.assert_allclose(
        transform.coordinate_atol[transform.active_mask],
        expected_active_atol,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        transform.coordinate_atol[~transform.active_mask],
        expected_inactive_atol,
        rtol=0.0,
        atol=0.0,
    )

    positive_budget_trial = transform.initial_coordinates()
    positive_budget_trial[transform.active_mask] = expected_active_atol
    positive_budget_trial[~transform.active_mask] += expected_inactive_atol
    physical_trial = transform.to_physical(positive_budget_trial)
    expected_physical_increment = (
        physical_atol + physical_rtol * np.abs(initial)
    )
    np.testing.assert_allclose(
        physical_trial - initial,
        expected_physical_increment,
        rtol=5.0e-13,
        atol=1.0e-14,
    )


def test_invalid_state_coordinate_mode_is_rejected():
    with pytest.raises(ValueError, match="state_coordinates must be"):
        mol.run_transient(
            np.array([0.0, 1.0]),
            _single_ion_state(),
            (0.0, 1.0),
            np.array([1.0]),
            object(),
            mat=_material(),
            state_coordinates="log",  # type: ignore[arg-type]
        )
