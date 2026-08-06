"""Contracts for explicit J-V driver capability and QF production mode."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from perovskite_sim.discretization.grid import (
    GridResolutionError,
    Layer,
    multilayer_grid,
)
from perovskite_sim.experiments import jv_sweep
from perovskite_sim.experiments import quasi_fermi_steady_state as qf
from perovskite_sim.experiments import steady_state
from perovskite_sim.models.device import DeviceStack, LayerSpec
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.models.parameters import MaterialParams


CSI_CONFIG = Path("configs/cSi_homojunction.yaml")


def _stack(*, policy: str = "general") -> DeviceStack:
    material = MaterialParams(
        eps_r=11.7,
        mu_n=0.1,
        mu_p=0.05,
        D_ion=0.0,
        P_lim=1.0e30,
        P0=0.0,
        ni=1.0e16,
        tau_n=1.0e-6,
        tau_p=1.0e-6,
        n1=1.0e16,
        p1=1.0e16,
        B_rad=0.0,
        C_n=0.0,
        C_p=0.0,
        alpha=0.0,
        N_A=0.0,
        N_D=0.0,
        chi=4.05,
        Eg=1.12,
    )
    return DeviceStack(
        layers=(LayerSpec("si", 1.0e-6, material, role="absorber"),),
        V_bi=0.0,
        Phi=0.0,
        mode="legacy",
        jv_solver_policy=policy,
    )


def test_device_stack_rejects_unknown_jv_solver_policy():
    with pytest.raises(ValueError, match="jv_solver_policy"):
        _stack(policy="automatic")


def test_shipped_csi_config_enforces_grid_then_qf_driver_capability():
    stack = load_device_from_yaml(CSI_CONFIG)
    assert stack.jv_solver_policy == "cancellation_safe_qf_required"

    with pytest.raises(GridResolutionError):
        jv_sweep.run_jv_sweep(
            stack,
            N_grid=100,
            n_points=2,
            illuminated=False,
        )
    with pytest.raises(jv_sweep.JVDriverCapabilityError) as exc_info:
        jv_sweep.run_jv_sweep(
            stack,
            N_grid=200,
            n_points=2,
            illuminated=False,
        )
    assert exc_info.value.requested_driver == "transient"


@pytest.mark.parametrize(
    ("requested_driver", "entrypoint"),
    [
        (
            "transient",
            lambda stack: jv_sweep.run_jv_sweep(
                stack,
                N_grid=3,
                n_points=2,
                illuminated=False,
            ),
        ),
        (
            "steady_state",
            lambda stack: steady_state.run_jv_sweep_ss(
                stack,
                N_grid=3,
                n_points=2,
            ),
        ),
        (
            "steady_state_voc",
            lambda stack: steady_state.solve_voc_ss(stack, N_grid=3),
        ),
    ],
)
def test_qf_required_policy_rejects_general_drivers_before_solve(
    requested_driver,
    entrypoint,
):
    stack = _stack(policy="cancellation_safe_qf_required")
    with pytest.raises(jv_sweep.JVDriverCapabilityError) as exc_info:
        entrypoint(stack)
    assert exc_info.value.requested_driver == requested_driver
    assert "solver='quasi_fermi'" in str(exc_info.value)


def test_unvalidated_driver_override_is_explicit_and_diagnostic_only(monkeypatch):
    stack = _stack(policy="cancellation_safe_qf_required")

    class ReachedUnvalidatedSolve(RuntimeError):
        pass

    def _marker(*_args, **_kwargs):
        raise ReachedUnvalidatedSolve

    monkeypatch.setattr(jv_sweep, "build_material_arrays", _marker)
    with pytest.warns(jv_sweep.JVCertificationWarning, match="diagnostic"):
        with pytest.raises(ReachedUnvalidatedSolve):
            jv_sweep.run_jv_sweep(
                stack,
                N_grid=3,
                n_points=2,
                illuminated=False,
                allow_unvalidated_driver=True,
            )


def test_qf_stop_after_voc_retains_the_certified_bracket(monkeypatch):
    stack = _stack()
    x = multilayer_grid([Layer(stack.layers[0].thickness, 4)])
    currents = iter((2.0, 1.0, -1.0, -2.0))

    def _fake_point(_x, _stack, *, V_app, **_kwargs):
        return SimpleNamespace(
            V_app=V_app,
            current_A_m2=next(currents),
            certified=True,
        )

    monkeypatch.setattr(qf, "solve_quasi_fermi_steady_state", _fake_point)
    sweep = qf.solve_quasi_fermi_jv_sweep(
        x,
        stack,
        np.array([0.0, 0.1, 0.2, 0.3]),
        stop_after_voc=True,
    )

    assert sweep.voltages_V == pytest.approx([0.0, 0.1, 0.2])
    assert sweep.currents_A_m2 == pytest.approx([2.0, 1.0, -1.0])
    assert len(sweep.points) == 3
    assert sweep.metrics.voc_bracketed
