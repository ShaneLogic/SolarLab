"""Time-origin reset retry for stiff J-V transient steps.

Radau's minimum internal step scales with eps * |t|, and the sweep drivers
integrate step k on the absolute interval [k*dt, (k+1)*dt]. On stiff-sink
stacks (Calado 2016 Fig 1f toy model, contact SRH tau = 2e-15 s) every
"Required step size is less than spacing between numbers" failure was measured
at t_lo = 21-66 s while the identical interval integrated cleanly from t = 0
(the RHS is autonomous — ``assemble_rhs`` never reads its ``t`` argument,
AST-verified 2026-09-04).

The fix is deliberately DEFECT-SCOPED: ``_integrate_step`` retries in the
t=0-shifted frame only after the in-place attempt has failed, so every
healthy-path trajectory is bit-identical to the pre-fix solver. A wholesale
switch to relative spans was measured to relocate the near-flat-band
wrong-branch landing to a grid/points combination the branch rejector does
not cover (ionmonger_benchmark steric-off N_grid=40/n_points=22:
FF_rev 0.779 -> 1.126 with the guard blind to it) — these tests pin the
scoped design, not the blanket one.
"""
from __future__ import annotations

import numpy as np
import pytest

from perovskite_sim.experiments import jv_sweep as jv
from perovskite_sim.models.config_loader import load_device_from_yaml


@pytest.fixture(scope="module")
def stack():
    return load_device_from_yaml("configs/nip_MAPbI3.yaml")


def test_time_origin_reset_retry_rescues_absolute_span_failure(
    monkeypatch, stack
):
    """A step that fails at an absolute origin must be retried from t=0."""
    x = jv.build_electrical_grid(stack, 25)
    mat = jv.build_material_arrays(x, stack)
    y0 = jv.solve_equilibrium(x, stack)

    original = jv.run_transient
    spans: list[tuple[float, float]] = []

    def failing_at_absolute_origin(x_arg, y_arg, t_span, t_eval, stack_arg,
                                   **kwargs):
        spans.append((float(t_span[0]), float(t_span[1])))
        if t_span[0] > 0.0:
            raise RuntimeError("synthetic eps*t step-size underflow")
        return original(x_arg, y_arg, t_span, t_eval, stack_arg, **kwargs)

    monkeypatch.setattr(jv, "run_transient", failing_at_absolute_origin)

    y_out = jv._integrate_step(
        x, y0, stack, mat, 0.05, 120.0, 120.5, 1e-4, 1e-6,
    )
    assert np.all(np.isfinite(y_out))
    assert spans[0] == (120.0, 120.5), "first attempt must stay in place"
    assert (0.0, 0.5) in spans, "shifted-frame retry never happened"


def test_healthy_steps_never_enter_the_shifted_frame(monkeypatch, stack):
    """The retry is failure-gated: a converging step keeps absolute spans,
    so pre-fix trajectories are bit-identical.

    t_lo is kept small here because the defect is real: the same jump-start
    state at t_lo = 120 s genuinely underflows in place and IS rescued by the
    shifted frame (measured while writing this test), which is the other
    test's subject."""
    x = jv.build_electrical_grid(stack, 25)
    mat = jv.build_material_arrays(x, stack)
    y0 = jv.solve_equilibrium(x, stack)

    original = jv.run_transient
    spans: list[tuple[float, float]] = []

    def recording(x_arg, y_arg, t_span, t_eval, stack_arg, **kwargs):
        spans.append((float(t_span[0]), float(t_span[1])))
        return original(x_arg, y_arg, t_span, t_eval, stack_arg, **kwargs)

    monkeypatch.setattr(jv, "run_transient", recording)

    jv._integrate_step(x, y0, stack, mat, 0.05, 2.0, 2.5, 1e-4, 1e-6)
    assert spans == [(2.0, 2.5)], (
        f"healthy step took extra attempts or shifted frames: {spans}"
    )


def test_calado2016_fig1f_preset_loads():
    stack = load_device_from_yaml("configs/calado2016_fig1f.yaml")
    layers = stack.layers
    assert len(layers) == 3
    absorber = layers[1].params
    # paper SI Table 1 values, SI units
    assert absorber.D_ion == pytest.approx(2.585e-18)
    assert absorber.P0 == pytest.approx(1e25)
    assert layers[0].params.tau_n == pytest.approx(2e-15)
    assert layers[2].params.tau_p == pytest.approx(2e-15)
    assert all(lay.params.mu_n == pytest.approx(2e-3) for lay in layers)
    assert stack.V_bi == pytest.approx(1.3)


@pytest.mark.slow
def test_calado2016_fig1f_full_sweep_survives_paper_tau():
    """The paper-value contact sink (tau=2e-15 s) must complete a certified
    built-in sweep. Pre-fix this died with step-size underflow on the
    reverse leg (the eps*t floor at t_elapsed ~ 55 s)."""
    stack = load_device_from_yaml("configs/calado2016_fig1f.yaml")
    res = jv.run_jv_sweep(stack, v_rate=0.04, V_max=1.2, n_points=30)
    assert np.all(np.isfinite(res.J_fwd))
    assert np.all(np.isfinite(res.J_rev))
    # reverse scan collects better than forward (ionic hysteresis direction)
    assert res.metrics_rev.PCE >= res.metrics_fwd.PCE
    # J_sc within the 16 mA/cm2 absorbed-photon budget, not above it
    assert 100.0 < res.metrics_fwd.J_sc < 165.0
