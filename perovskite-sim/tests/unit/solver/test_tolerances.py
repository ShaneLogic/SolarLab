from types import SimpleNamespace

import numpy as np
import pytest

from perovskite_sim._compat.scipy_shim import solve_ivp as shim_solve_ivp
from perovskite_sim.experiments import degradation
from perovskite_sim.solver import illuminated_ss, mol
from perovskite_sim.solver.tolerances import (
    ComponentwiseAtol,
    build_componentwise_atol_1d,
    build_componentwise_atol_2d,
    build_componentwise_atol_ions,
)
from perovskite_sim.twod import solver_2d


def _one_dimensional_material(N: int = 2, *, dual: bool = False):
    return SimpleNamespace(
        ni_sq=np.full(N, 100.0),
        N_A=np.zeros(N),
        N_D=np.zeros(N),
        P_ion0=np.array([1000.0, 0.0]),
        P_ion0_neg=np.array([100.0, 200.0]) if dual else None,
        P_lim_node=np.full(N, 1.0e6),
        P_lim_neg_node=np.full(N, 1.0e6) if dual else None,
        has_dual_ions=dual,
        N_iface_state=0,
        has_selective_contacts=False,
        n_L=10.0,
        p_L=10.0,
        n_R=10.0,
        p_R=10.0,
        ion_steric_shared_site=True,
    )


def test_1d_policy_orders_dual_ion_and_interface_blocks():
    policy = ComponentwiseAtol(
        carrier_fraction=0.1,
        ion_fraction=0.01,
        interface_fraction=0.5,
        minimum_atol=1.0,
        refinement_factor=0.1,
    )
    y0 = np.concatenate([
        np.zeros(8),
        np.array([2.0, 4.0, 6.0, 8.0]),
    ])

    atol = build_componentwise_atol_1d(
        policy,
        y0=y0,
        ni_sq=np.full(2, 100.0),
        N_A=np.zeros(2),
        N_D=np.zeros(2),
        P_ion0=np.array([1000.0, 0.0]),
        has_dual_ions=True,
        P_ion0_neg=np.array([100.0, 200.0]),
        n_interface_states=1,
    )

    expected = np.array([
        0.1, 0.1,  # n
        0.1, 0.1,  # p
        1.0, 0.1,  # P
        0.1, 0.2,  # P_neg
        0.1, 0.2, 0.3, 0.4,  # interface state
    ])
    np.testing.assert_allclose(atol, expected, rtol=0.0, atol=1.0e-15)
    assert atol.shape == y0.shape


def test_refined_policy_scales_every_component_uniformly():
    base = ComponentwiseAtol()
    kwargs = dict(
        P_ion0=np.array([0.0, 1.0e24]),
        P_ion0_neg=np.array([1.0e20, 2.0e20]),
    )
    loose = build_componentwise_atol_ions(base, **kwargs)
    tight = build_componentwise_atol_ions(base.refined(0.1), **kwargs)
    np.testing.assert_allclose(tight, 0.1 * loose, rtol=0.0, atol=0.0)


def test_policy_normalizes_numeric_strings_from_config_boundaries():
    policy = ComponentwiseAtol(
        carrier_fraction="1e-10",
        ion_fraction="2e-10",
        interface_fraction="3e-10",
        minimum_atol="1e-7",
        refinement_factor="0.5",
    )
    assert policy.carrier_fraction == pytest.approx(1.0e-10)
    assert policy.ion_fraction == pytest.approx(2.0e-10)
    assert policy.interface_fraction == pytest.approx(3.0e-10)
    assert policy.minimum_atol == pytest.approx(1.0e-7)
    assert policy.refinement_factor == pytest.approx(0.5)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"carrier_fraction": 0.0},
        {"ion_fraction": -1.0},
        {"interface_fraction": np.inf},
        {"minimum_atol": np.nan},
        {"refinement_factor": True},
    ],
)
def test_policy_rejects_invalid_parameters(kwargs):
    with pytest.raises(ValueError, match="finite positive"):
        ComponentwiseAtol(**kwargs)


@pytest.mark.parametrize("factor", [0.0, -0.1, np.inf, False])
def test_policy_rejects_invalid_refinement_factor(factor):
    with pytest.raises(ValueError, match="finite and positive"):
        ComponentwiseAtol().refined(factor)


def test_1d_policy_fails_closed_on_invalid_state_layout():
    with pytest.raises(ValueError, match="P_ion0_neg is required"):
        build_componentwise_atol_1d(
            ComponentwiseAtol(),
            y0=np.zeros(8),
            ni_sq=np.ones(2),
            N_A=np.zeros(2),
            N_D=np.zeros(2),
            P_ion0=np.zeros(2),
            has_dual_ions=True,
        )

    with pytest.raises(ValueError, match="does not match the 1D state layout"):
        build_componentwise_atol_1d(
            ComponentwiseAtol(),
            y0=np.zeros(5),
            ni_sq=np.ones(2),
            N_A=np.zeros(2),
            N_D=np.zeros(2),
            P_ion0=np.zeros(2),
        )


def test_run_transient_keeps_scalar_atol_object_unchanged(monkeypatch):
    captured = []

    def fake_solve_ivp(*_args, **kwargs):
        captured.append(kwargs["atol"])
        return SimpleNamespace(success=True, y=np.zeros((6, 1)))

    monkeypatch.setattr(mol, "solve_ivp", fake_solve_ivp)
    scalar_atol = 7.5e-9
    mol.run_transient(
        np.array([0.0, 1.0]),
        np.zeros(6),
        (0.0, 1.0),
        np.array([1.0]),
        object(),
        atol=scalar_atol,
        mat=_one_dimensional_material(),
    )

    assert captured == [scalar_atol]
    assert captured[0] is scalar_atol


def test_run_transient_expands_componentwise_policy(monkeypatch):
    captured = []

    def fake_solve_ivp(*_args, **kwargs):
        captured.append(kwargs["atol"])
        return SimpleNamespace(success=True, y=np.zeros((6, 1)))

    monkeypatch.setattr(mol, "solve_ivp", fake_solve_ivp)
    mat = _one_dimensional_material()
    policy = ComponentwiseAtol()
    y0 = np.zeros(6)
    mol.run_transient(
        np.array([0.0, 1.0]),
        y0,
        (0.0, 1.0),
        np.array([1.0]),
        object(),
        atol=policy,
        mat=mat,
    )

    expected = build_componentwise_atol_1d(
        policy,
        y0=y0,
        ni_sq=mat.ni_sq,
        N_A=mat.N_A,
        N_D=mat.N_D,
        P_ion0=mat.P_ion0,
    )
    np.testing.assert_array_equal(captured[0], expected)


def test_run_transient_evaluates_continuous_voltage_at_rhs_time(monkeypatch):
    observed = []

    def fake_assemble_rhs(t, y, _x, _stack, _mat, _lit, V_app):
        observed.append((float(t), float(V_app)))
        return np.zeros_like(y)

    def fake_solve_ivp(rhs, *_args, **_kwargs):
        state = np.zeros(6)
        rhs(0.25, state)
        rhs(0.75, state)
        return SimpleNamespace(success=True, y=state[:, None])

    monkeypatch.setattr(mol, "assemble_rhs", fake_assemble_rhs)
    monkeypatch.setattr(mol, "solve_ivp", fake_solve_ivp)
    mol.run_transient(
        np.array([0.0, 1.0]),
        np.zeros(6),
        (0.0, 1.0),
        np.array([1.0]),
        object(),
        V_app=lambda t: 0.1 + 0.2 * t,
        mat=_one_dimensional_material(),
    )

    np.testing.assert_allclose(observed, [(0.25, 0.15), (0.75, 0.25)])


def test_run_transient_rejects_nonfinite_callable_voltage(monkeypatch):
    def fake_solve_ivp(rhs, *_args, **_kwargs):
        rhs(0.5, np.zeros(6))

    monkeypatch.setattr(mol, "solve_ivp", fake_solve_ivp)
    with pytest.raises(ValueError, match="non-finite"):
        mol.run_transient(
            np.array([0.0, 1.0]),
            np.zeros(6),
            (0.0, 1.0),
            np.array([1.0]),
            object(),
            V_app=lambda _t: np.nan,
            mat=_one_dimensional_material(),
        )


def test_split_step_uses_dual_ion_vector_and_forwards_policy(monkeypatch):
    mat = _one_dimensional_material(dual=True)
    policy = ComponentwiseAtol()
    y0 = mol.StateVec.pack(
        np.full(2, 10.0),
        np.full(2, 10.0),
        mat.P_ion0,
        mat.P_ion0_neg,
    )
    ion_atols = []
    carrier_atols = []

    def fake_solve_ivp(_fun, _span, state, **kwargs):
        ion_atols.append(kwargs["atol"])
        return SimpleNamespace(success=True, y=np.asarray(state)[:, None])

    def fake_run_transient(_x, state, *_args, **kwargs):
        carrier_atols.append(kwargs["atol"])
        return SimpleNamespace(success=True, y=np.asarray(state)[:, None])

    monkeypatch.setattr(mol, "solve_ivp", fake_solve_ivp)
    monkeypatch.setattr(mol, "run_transient", fake_run_transient)

    y1, success = mol.split_step(
        np.array([0.0, 1.0]),
        y0,
        0.1,
        object(),
        atol=policy,
        mat=mat,
    )

    assert success is True
    np.testing.assert_array_equal(y1, y0)
    expected = build_componentwise_atol_ions(
        policy,
        P_ion0=mat.P_ion0,
        P_ion0_neg=mat.P_ion0_neg,
    )
    np.testing.assert_array_equal(ion_atols[0], expected)
    assert ion_atols[0].shape == (4,)
    assert carrier_atols == [policy]


def test_2d_policy_flattens_n_then_p_and_reaches_solver(monkeypatch):
    ni = np.array([[10.0, 20.0], [30.0, 40.0]])
    mat = SimpleNamespace(
        ni=ni,
        N_A=np.zeros_like(ni),
        N_D=np.zeros_like(ni),
    )
    policy = ComponentwiseAtol(
        carrier_fraction=0.1,
        minimum_atol=1.0,
    )
    y0 = np.zeros(2 * ni.size)
    captured = []

    def fake_solve_ivp(*_args, **kwargs):
        captured.append(kwargs["atol"])
        return SimpleNamespace(success=True, y=y0[:, None])

    monkeypatch.setattr(solver_2d, "solve_ivp", fake_solve_ivp)
    result = solver_2d.run_transient_2d(
        y0,
        mat,
        V_app=0.0,
        t_end=1.0e-6,
        atol=policy,
    )

    expected = build_componentwise_atol_2d(
        policy,
        ni=mat.ni,
        N_A=mat.N_A,
        N_D=mat.N_D,
    )
    np.testing.assert_array_equal(captured[0], expected)
    np.testing.assert_array_equal(result, y0)
    assert captured[0].shape == y0.shape


def test_2d_scalar_default_is_unchanged(monkeypatch):
    ni = np.ones((1, 2))
    mat = SimpleNamespace(ni=ni, N_A=np.zeros_like(ni), N_D=np.zeros_like(ni))
    y0 = np.zeros(4)
    captured = []

    def fake_solve_ivp(*_args, **kwargs):
        captured.append(kwargs["atol"])
        return SimpleNamespace(success=True, y=y0[:, None])

    monkeypatch.setattr(solver_2d, "solve_ivp", fake_solve_ivp)
    solver_2d.run_transient_2d(y0, mat, V_app=0.0, t_end=1.0e-6)
    assert captured == [1.0e-8]


def test_2d_policy_rejects_state_shape_mismatch():
    ni = np.ones((1, 2))
    mat = SimpleNamespace(ni=ni, N_A=np.zeros_like(ni), N_D=np.zeros_like(ni))
    with pytest.raises(ValueError, match="does not match the 2D state layout"):
        solver_2d.run_transient_2d(
            np.zeros(3),
            mat,
            V_app=0.0,
            t_end=1.0e-6,
            atol=ComponentwiseAtol(),
        )


def test_illuminated_preconditioner_uses_componentwise_negative_bound(monkeypatch):
    x = np.array([0.0, 1.0])
    mat = _one_dimensional_material()
    y_dark = np.array([10.0, 10.0, 10.0, 10.0, 1000.0, 0.0])
    terminal = y_dark.copy()
    terminal[0] = -0.5
    policy = ComponentwiseAtol(
        carrier_fraction=0.1,
        minimum_atol=0.1,
    )
    monkeypatch.setattr(
        illuminated_ss,
        "solve_equilibrium",
        lambda *_args, **_kwargs: y_dark,
    )
    monkeypatch.setattr(
        illuminated_ss,
        "run_transient",
        lambda *_args, **_kwargs: SimpleNamespace(
            success=True,
            y=terminal[:, None],
        ),
    )

    result = illuminated_ss.solve_illuminated_ss(
        x,
        object(),
        atol=policy,
        mat=mat,
    )
    np.testing.assert_array_equal(result, terminal)


def test_degradation_snapshot_tightens_componentwise_policy(monkeypatch):
    policy = ComponentwiseAtol()
    captured = []
    y_ref = np.ones(6)
    monkeypatch.setattr(degradation, "_freeze_ions", lambda stack: stack)
    monkeypatch.setattr(
        degradation,
        "build_material_arrays",
        lambda *_args, **_kwargs: object(),
    )

    def fake_run_transient(*_args, **kwargs):
        captured.append(kwargs["atol"])
        return SimpleNamespace(success=True, y=y_ref[:, None])

    monkeypatch.setattr(degradation, "run_transient", fake_run_transient)
    monkeypatch.setattr(degradation, "_compute_current", lambda *_a, **_k: 0.0)
    monkeypatch.setattr(
        degradation,
        "compute_metrics",
        lambda voltages, current: (voltages.copy(), current.copy()),
    )

    degradation._measure_snapshot_metrics(
        np.array([0.0, 1.0]),
        y_ref,
        object(),
        np.array([0.0]),
        1.0e-3,
        rtol=1.0e-4,
        atol=policy,
    )

    assert len(captured) == 1
    assert isinstance(captured[0], ComponentwiseAtol)
    assert captured[0].refinement_factor == pytest.approx(0.01)


def test_scipy_shim_accepts_componentwise_array():
    y0 = np.array([1.0, 2.0])
    sol = shim_solve_ivp(
        lambda _t, y: np.zeros_like(y),
        (0.0, 1.0e-6),
        y0,
        t_eval=np.array([1.0e-6]),
        atol=np.array([1.0e-9, 1.0e-3]),
    )
    assert sol.success
    np.testing.assert_array_equal(sol.y[:, -1], y0)
