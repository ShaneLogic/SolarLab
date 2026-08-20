from types import SimpleNamespace

import numpy as np
import pytest

from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.discretization.grid import multilayer_grid, Layer
from perovskite_sim.solver import illuminated_ss
from perovskite_sim.solver.illuminated_ss import (
    IlluminatedSteadyStateError,
    solve_illuminated_ss,
)
from perovskite_sim.solver.newton import solve_equilibrium
from perovskite_sim.solver.numerical_diagnostics import (
    NumericalDiagnosticsMonitor,
    StateLayout,
)


def _fake_mat(N=2, *, dual=False, limit=1.0e30):
    return SimpleNamespace(
        has_dual_ions=dual,
        N_iface_state=0,
        P_lim_node=np.full(N, limit),
        P_lim_neg_node=np.full(N, limit) if dual else None,
        ion_steric_shared_site=True,
    )


@pytest.fixture(scope="module")
def grid_and_stack():
    stack = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    layers_grid = [Layer(l.thickness, 10) for l in stack.layers]
    x = multilayer_grid(layers_grid)
    return x, stack


def test_transient_failure_raises_with_solver_message(monkeypatch):
    """A failed light settle must never be reported as the dark state."""
    x = np.array([0.0, 1.0])
    y_dark = np.arange(6, dtype=float)
    monkeypatch.setattr(
        illuminated_ss, "solve_equilibrium", lambda *_args, **_kwargs: y_dark,
    )
    monkeypatch.setattr(
        illuminated_ss,
        "run_transient",
        lambda *_args, **_kwargs: SimpleNamespace(
            success=False,
            message="synthetic Radau failure",
            y=y_dark[:, None],
        ),
    )

    with pytest.raises(
        IlluminatedSteadyStateError, match="synthetic Radau failure",
    ) as exc_info:
        solve_illuminated_ss(
            x, object(), V_app=0.73, t_settle=2e-3, mat=_fake_mat(),
        )

    assert exc_info.value.V_app == pytest.approx(0.73)
    assert exc_info.value.t_settle == pytest.approx(2e-3)
    assert exc_info.value.solver_message == "synthetic Radau failure"


def test_transient_failure_without_message_still_raises(monkeypatch):
    """Failure objects without an optional message remain fail-closed."""
    x = np.array([0.0, 1.0])
    y_dark = np.arange(6, dtype=float)
    monkeypatch.setattr(
        illuminated_ss, "solve_equilibrium", lambda *_args, **_kwargs: y_dark,
    )
    monkeypatch.setattr(
        illuminated_ss,
        "run_transient",
        lambda *_args, **_kwargs: SimpleNamespace(
            success=False,
            y=y_dark[:, None],
        ),
    )

    with pytest.raises(IlluminatedSteadyStateError, match="settling failed"):
        solve_illuminated_ss(x, object(), mat=_fake_mat())


def test_transient_numeric_exception_is_normalised(monkeypatch):
    x = np.array([0.0, 1.0])
    y_dark = np.arange(6, dtype=float)
    monkeypatch.setattr(
        illuminated_ss, "solve_equilibrium", lambda *_args, **_kwargs: y_dark,
    )
    monkeypatch.setattr(
        illuminated_ss,
        "run_transient",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ValueError("synthetic scipy Jacobian failure")
        ),
    )

    with pytest.raises(
        IlluminatedSteadyStateError, match="synthetic scipy Jacobian failure",
    ) as exc_info:
        solve_illuminated_ss(x, object(), mat=_fake_mat())

    assert exc_info.value.reason_code == "solver_exception"
    assert isinstance(exc_info.value.__cause__, ValueError)


@pytest.mark.parametrize(
    "bad_states",
    (
        np.empty((6, 0)),
        np.full((6, 1), np.nan),
        np.zeros((5, 1)),
    ),
)
def test_success_without_finite_terminal_state_raises(monkeypatch, bad_states):
    """A malformed successful result is not a certified illuminated state."""
    x = np.array([0.0, 1.0])
    y_dark = np.arange(6, dtype=float)
    monkeypatch.setattr(
        illuminated_ss, "solve_equilibrium", lambda *_args, **_kwargs: y_dark,
    )
    monkeypatch.setattr(
        illuminated_ss,
        "run_transient",
        lambda *_args, **_kwargs: SimpleNamespace(
            success=True,
            message="synthetic success",
            y=bad_states,
        ),
    )

    with pytest.raises(
        IlluminatedSteadyStateError, match="no finite terminal state",
    ):
        solve_illuminated_ss(x, object(), mat=_fake_mat())


def test_success_with_negative_density_is_rejected(monkeypatch):
    x = np.array([0.0, 1.0])
    y_dark = np.arange(6, dtype=float)
    bad = y_dark.copy()
    bad[0] = -1.0
    monkeypatch.setattr(
        illuminated_ss, "solve_equilibrium", lambda *_args, **_kwargs: y_dark,
    )
    monkeypatch.setattr(
        illuminated_ss,
        "run_transient",
        lambda *_args, **_kwargs: SimpleNamespace(success=True, y=bad[:, None]),
    )

    with pytest.raises(
        IlluminatedSteadyStateError, match="negative density",
    ) as exc_info:
        solve_illuminated_ss(x, object(), mat=_fake_mat())

    assert exc_info.value.reason_code == "unphysical_terminal_state"


def test_success_with_ion_site_overfill_is_rejected(monkeypatch):
    x = np.array([0.0, 1.0])
    y_dark = np.ones(6, dtype=float)
    bad = y_dark.copy()
    bad[4:] = 6.0
    monkeypatch.setattr(
        illuminated_ss, "solve_equilibrium", lambda *_args, **_kwargs: y_dark,
    )
    monkeypatch.setattr(
        illuminated_ss,
        "run_transient",
        lambda *_args, **_kwargs: SimpleNamespace(success=True, y=bad[:, None]),
    )

    with pytest.raises(
        IlluminatedSteadyStateError, match="site-occupancy limit",
    ) as exc_info:
        solve_illuminated_ss(x, object(), mat=_fake_mat(limit=5.0))

    assert exc_info.value.reason_code == "unphysical_terminal_state"


def test_dual_ion_state_shape_is_supported(monkeypatch):
    x = np.array([0.0, 1.0])
    y_dark = np.ones(8, dtype=float)
    terminal = y_dark * 2.0
    monkeypatch.setattr(
        illuminated_ss, "solve_equilibrium", lambda *_args, **_kwargs: y_dark,
    )
    monkeypatch.setattr(
        illuminated_ss,
        "run_transient",
        lambda *_args, **_kwargs: SimpleNamespace(
            success=True, y=terminal[:, None],
        ),
    )

    result = solve_illuminated_ss(
        x, object(), mat=_fake_mat(dual=True, limit=10.0),
    )

    np.testing.assert_array_equal(result, terminal)


def test_opt_in_return_includes_immutable_numerical_diagnostics(monkeypatch):
    x = np.array([0.0, 1.0])
    terminal = np.ones(6, dtype=float)
    monitor = NumericalDiagnosticsMonitor(
        StateLayout(2, positive_ion_active=(True, True))
    )
    monitor.observe_trial_state(terminal)
    monitor.observe_srh_denominator("bulk", np.array([2.0]))
    report = monitor.finalize(terminal, solver_success=True)
    monkeypatch.setattr(
        illuminated_ss,
        "solve_equilibrium",
        lambda *_args, **_kwargs: terminal,
    )
    monkeypatch.setattr(
        illuminated_ss,
        "run_transient",
        lambda *_args, **_kwargs: SimpleNamespace(
            success=True,
            y=terminal[:, None],
            numerical_diagnostics=report,
            nfev=11,
            njev=3,
            nlu=4,
        ),
    )

    state, diagnostics = solve_illuminated_ss(
        x,
        object(),
        mat=_fake_mat(),
        return_diagnostics=True,
    )

    np.testing.assert_array_equal(state, terminal)
    assert diagnostics.numerical_diagnostics is report
    assert (diagnostics.nfev, diagnostics.njev, diagnostics.nlu) == (11, 3, 4)


def test_shape(grid_and_stack):
    x, stack = grid_and_stack
    y = solve_illuminated_ss(x, stack, V_app=0.0)
    assert y.shape == (3 * len(x),)


def test_carriers_finite(grid_and_stack):
    """All carrier values must be finite after illuminated settling."""
    x, stack = grid_and_stack
    N = len(x)
    y = solve_illuminated_ss(x, stack, V_app=0.0)
    n, p = y[:N], y[N:2*N]
    assert np.all(np.isfinite(n))
    assert np.all(np.isfinite(p))


def test_absorber_np_product_larger_under_illumination(grid_and_stack):
    """Under illumination n·p > ni² in absorber (quasi-Fermi level splitting)."""
    x, stack = grid_and_stack
    N = len(x)
    offset = stack.layers[0].thickness
    abs_mask = (x > offset) & (x < offset + stack.layers[1].thickness)
    y_dark = solve_equilibrium(x, stack)
    y_light = solve_illuminated_ss(x, stack, V_app=0.0)
    n_dark, p_dark = y_dark[:N][abs_mask], y_dark[N:2*N][abs_mask]
    n_light, p_light = y_light[:N][abs_mask], y_light[N:2*N][abs_mask]
    # Dark: n·p ≈ ni²  (thermal equilibrium)
    # Illuminated: n·p >> ni²  (photogeneration splits quasi-Fermi levels)
    assert np.mean(n_light * p_light) > np.mean(n_dark * p_dark) * 10


def test_ions_in_absorber_unchanged(grid_and_stack):
    """Ion density in absorber must be essentially unchanged after 1 ms."""
    x, stack = grid_and_stack
    N = len(x)
    offset = stack.layers[0].thickness
    abs_mask = (x > offset) & (x < offset + stack.layers[1].thickness)
    y_dark = solve_equilibrium(x, stack)
    y_light = solve_illuminated_ss(x, stack, V_app=0.0, t_settle=1e-3)
    P_dark = y_dark[2*N:][abs_mask]
    P_light = y_light[2*N:][abs_mask]
    # Ion displacement in 1 ms ~ 0.3 nm, negligible vs absorber thickness
    np.testing.assert_allclose(P_light, P_dark, rtol=0.05)


def test_v_app_changes_carriers(grid_and_stack):
    """Different V_app values must produce different carrier distributions."""
    x, stack = grid_and_stack
    N = len(x)
    y_sc = solve_illuminated_ss(x, stack, V_app=0.0)
    y_oc = solve_illuminated_ss(x, stack, V_app=0.9)
    n_sc, p_sc = y_sc[:N], y_sc[N:2*N]
    n_oc, p_oc = y_oc[:N], y_oc[N:2*N]
    # Near-OC bias injects more carriers into absorber
    assert np.mean(n_oc) > np.mean(n_sc)
    assert np.mean(p_oc) > np.mean(p_sc)
