"""Direct steady-state driver (2026-06) — experiments/steady_state.py.

Solves F(y) = 0 on the SAME assemble_rhs the transient uses (d/dt = 0,
ions frozen) — the second structural piece of the parity architecture:
SCAPS is an ion-free steady-state solver, so parity quantities are defined
at the ion-free steady state, while the transient MOL core remains the
engine for ion-migration physics.

Gates:
  * dark equilibrium: converges, residual below tolerance, J ~ 0
  * parity: SS J-V matches a frozen-ion slow-scan transient J-V on the
    SCAPS-mirror config (V_oc within a few mV, J_sc within 1 %)
  * direct V_oc solve consistent with the J-V interpolation
  * the payoff regime: converges at near-insulating ETL doping where the
    transient solver cannot settle
  * no silent fallback: non-convergence raises
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

import perovskite_sim.experiments.steady_state as steady_state_mod
from perovskite_sim.discretization.grid import multilayer_grid, Layer
from perovskite_sim.experiments.jv_sweep import (
    _compute_current_ss,
    run_jv_sweep,
    thermodynamic_voc_ceiling,
)
from perovskite_sim.experiments.steady_state import (
    SteadyStateError,
    run_jv_sweep_ss,
    solve_steady_state,
    solve_voc_ss,
)
from perovskite_sim.models.device import electrical_layers
from perovskite_sim.scaps_compat import load_scaps_yaml
from perovskite_sim.solver.mol import build_material_arrays

_V2 = "configs/scaps_mirror_v2.yaml"


def _stack():
    return dataclasses.replace(load_scaps_yaml(_V2), dos_band_potentials=True)


def _frozen_ion(stack):
    layers = tuple(
        dataclasses.replace(L, params=dataclasses.replace(L.params, D_ion=0.0))
        for L in stack.layers
    )
    return dataclasses.replace(stack, layers=layers)


def _grid(stack, n_per=10):
    elec = electrical_layers(stack)
    return multilayer_grid([Layer(thickness=L.thickness, N=n_per) for L in elec])


def test_dark_equilibrium_converges_with_small_residual():
    stack = _stack()
    x = _grid(stack)
    res = solve_steady_state(x, stack, V_app=0.0, illuminated=False)
    assert res.converged
    mat = build_material_arrays(x, stack)
    J = _compute_current_ss(x, res.y, stack, 0.0, mat=mat)
    assert abs(J) < 1.0  # A/m^2 — dark short-circuit current ~ 0


def test_illuminated_jsc_physical():
    stack = _stack()
    x = _grid(stack)
    res = solve_steady_state(x, stack, V_app=0.0, illuminated=True)
    assert res.converged
    mat = build_material_arrays(x, stack)
    J = _compute_current_ss(x, res.y, stack, 0.0, mat=mat)
    assert 230.0 < J < 280.0  # ~25.7 mA/cm^2 = 257 A/m^2


@pytest.mark.slow
def test_ss_jv_matches_frozen_ion_transient():
    """Parity gate: same physics, two drivers, frozen ions both."""
    stack = _frozen_ion(_stack())
    ss = run_jv_sweep_ss(stack, N_grid=30, V_max=1.25, n_points=26)
    tr = run_jv_sweep(stack, N_grid=30, n_points=40, v_rate=5.0, V_max=1.25,
                      v_max_max_attempts=2)
    assert ss.metrics.voc_bracketed and tr.metrics_fwd.voc_bracketed
    assert ss.metrics.V_oc == pytest.approx(tr.metrics_fwd.V_oc, abs=5e-3)
    assert ss.metrics.J_sc == pytest.approx(tr.metrics_fwd.J_sc, rel=0.01)


@pytest.mark.slow
def test_direct_voc_consistent_with_jv():
    stack = _frozen_ion(_stack())
    ss = run_jv_sweep_ss(stack, N_grid=30, V_max=1.25, n_points=26)
    voc = solve_voc_ss(stack, N_grid=30)
    assert voc == pytest.approx(ss.metrics.V_oc, abs=2e-3)


@pytest.mark.slow
def test_stop_after_voc_truncates_and_avoids_deep_injection():
    """``stop_after_voc`` halts the continuation once J crosses zero, so a
    high ``V_max`` no longer drags the sweep into the deep-forward-injection
    points (V >> V_oc) whose certified fallback grinds for minutes. The 0->V_oc
    arc fully determines the metrics, so V_oc is unchanged."""
    stack = _frozen_ion(_stack())
    # V_max=1.6 is well above V_oc (~1.16): without stop_after_voc the tail
    # would spend minutes on non-convergent deep-injection points.
    ss = run_jv_sweep_ss(stack, N_grid=30, V_max=1.6, n_points=24,
                         stop_after_voc=True)
    assert ss.metrics.voc_bracketed
    voc = solve_voc_ss(stack, N_grid=30)
    assert ss.metrics.V_oc == pytest.approx(voc, abs=1e-2)
    # stopped just past V_oc, nowhere near V_max
    assert ss.V[-1] < voc + 0.1
    assert ss.V[-1] < 1.5


@pytest.mark.slow
def test_stop_after_voc_default_off_reaches_vmax():
    """Default (False) is the legacy full-range sweep (bit-identical path)."""
    stack = _frozen_ion(_stack())
    ss = run_jv_sweep_ss(stack, N_grid=30, V_max=1.25, n_points=20)
    assert ss.V[-1] == pytest.approx(1.25, abs=1e-9)


def _low_doping_etl_stack(*, metal_reservoir: bool):
    """Build the same low-doping device with an explicit contact model."""
    from perovskite_sim.sweeps.device_parameter_sweep import (
        SweepPoint,
        apply_sweep_point,
    )
    base = apply_sweep_point(
        load_scaps_yaml(_V2),
        SweepPoint("p", "nd", "1e10", {"etl_doping_cm3": 1e10}),
    )
    return _frozen_ion(dataclasses.replace(
        base, dos_band_potentials=True, flat_band_contacts=True,
        flat_band_metal_contacts=metal_reservoir))


@pytest.mark.slow
def test_low_doping_etl_metal_reservoir_brackets_physical_voc():
    """The shipped metal-reservoir BC closes the old no-crossing branch.

    The SCAPS value is a calibrated comparison, not a holdout prediction: the
    contact barrier in this preset is itself a model parameter.  This gate
    therefore checks finite/bracketed/sub-bandgap behaviour and retains the
    measured 60 mV comparison window without promoting it to external proof.
    """
    stack = _low_doping_etl_stack(metal_reservoir=True)
    voc = solve_voc_ss(stack, N_grid=30)
    assert np.isfinite(voc)
    ceiling = thermodynamic_voc_ceiling(stack)
    assert ceiling is not None
    assert 0.5 < voc < ceiling
    assert voc == pytest.approx(1.100196, abs=0.060)


@pytest.mark.slow
def test_low_doping_etl_without_metal_reservoir_is_explicit_no_crossing():
    """The legacy contact choice remains a diagnosed model branch, not xfail."""
    stack = _low_doping_etl_stack(metal_reservoir=False)
    with pytest.raises(SteadyStateError, match="does not cross zero"):
        solve_voc_ss(stack, N_grid=30)


def test_nonconvergence_raises():
    stack = _stack()
    x = _grid(stack)
    with pytest.raises(SteadyStateError):
        solve_steady_state(x, stack, V_app=0.0, illuminated=True,
                           max_newton=1, tol=1e-30, tol_step=0.0,
                           tol_accept=0.0, assist_times=())


def _small_step_fault_case(
    monkeypatch, residual, *, density=1.0e20, device_width=None,
):
    """Inject a well-conditioned residual whose Newton step is tiny."""
    stack = _stack()
    x = _grid(stack)
    mat = build_material_arrays(x, stack)
    N = len(x)
    n_blocks = 4 if mat.has_dual_ions else 3
    y0 = np.full(n_blocks * N, density)
    if device_width is not None:
        x = x * (device_width / x[-1])

    def fake_residual_fn(
        _x, _stack, _mat, y_template, _V_app, _illuminated,
        pin, z_pin, _n_ref, unk_idx, _phi_frozen=None,
    ):
        anchor = np.log(np.maximum(
            y_template[unk_idx], steady_state_mod._DENSITY_FLOOR))
        anchor[pin] = z_pin[pin]
        slope = 1.0e12

        def fake_residual(z):
            return np.full(z.shape, residual) + slope * (z - anchor)

        return fake_residual

    monkeypatch.setattr(steady_state_mod, "_residual_fn", fake_residual_fn)
    monkeypatch.setattr(
        steady_state_mod, "_qfl_poisson_relax",
        lambda _x, _mat, y, _V_app: y,
    )
    return stack, x, mat, y0


def test_tiny_newton_step_with_high_residual_raises(monkeypatch):
    """A tiny update is stagnation, not proof that F(y) is near zero."""
    stack, x, mat, y0 = _small_step_fault_case(monkeypatch, residual=1.0)

    with pytest.raises(SteadyStateError, match="Newton step stalled"):
        solve_steady_state(
            x, stack, V_app=0.0, mat=mat, y0=y0,
            max_newton=1, tol=1.0e-6, tol_step=1.0e-8,
            tol_accept=0.5, assist_times=(),
        )


def test_tiny_newton_step_keeps_stall_acceptance_semantics(monkeypatch):
    """The established kink/stall residual allowance remains available."""
    stack, x, mat, y0 = _small_step_fault_case(monkeypatch, residual=0.1)

    result = solve_steady_state(
        x, stack, V_app=0.0, mat=mat, y0=y0,
        max_newton=1, tol=1.0e-6, tol_step=1.0e-8,
        tol_accept=0.5, assist_times=(),
    )

    assert result.converged
    assert result.residual == pytest.approx(0.1)
    assert result.step_inf < 1.0e-8
    assert result.continuity_current_bound < 0.1


def test_stall_acceptance_is_current_scaled_for_thick_high_density_device(
    monkeypatch,
):
    """The same rate residual cannot certify thin PVK and thick c-Si alike."""
    stack, x, mat, y0 = _small_step_fault_case(
        monkeypatch,
        residual=0.1,
        density=1.0e25,
        device_width=180.0e-6,
    )

    with pytest.raises(SteadyStateError, match="Newton step stalled"):
        solve_steady_state(
            x, stack, V_app=0.0, mat=mat, y0=y0,
            max_newton=1, tol=1.0e-6, tol_step=1.0e-8,
            tol_accept=0.5, max_continuity_current_error=0.1,
            assist_times=(),
        )


@pytest.mark.parametrize("bound", [0.0, np.nan, np.inf, -np.inf])
def test_continuity_current_error_bound_must_be_finite_and_positive(bound):
    stack = _stack()
    x = _grid(stack)
    with pytest.raises(ValueError, match="must be finite and positive"):
        solve_steady_state(
            x, stack, V_app=0.0, max_continuity_current_error=bound,
        )


def test_relative_log_residual_resolves_contact_scale_correction(monkeypatch):
    """Centred logs retain a correction lost by absolute ``ln(n)``."""
    density_scale = 1.0e24
    density_correction = 1.0e9
    target_density = density_scale + density_correction
    assert np.log(target_density) == np.log(density_scale)

    y_template = np.array([density_scale, 1.0, 1.0])

    def fake_assemble_rhs(_t, y, *_args, **_kwargs):
        residual = np.zeros_like(y)
        residual[0] = y[0] - target_density
        return residual

    monkeypatch.setattr(
        steady_state_mod, "assemble_rhs", fake_assemble_rhs,
    )
    residual = steady_state_mod._relative_log_residual_fn(
        np.array([0.0, 1.0]),
        object(),
        object(),
        y_template,
        0.0,
        True,
        np.array([False]),
        np.zeros(1),
        1.0,
        np.array([0]),
        np.array([density_scale]),
    )
    u_correction = np.log1p(density_correction / density_scale)

    assert abs(residual(np.array([u_correction]))[0]) < 0.2 * abs(
        residual(np.zeros(1))[0]
    )


# ----------------------- Gummel phi-step primitive --------------------------

def test_qfl_relax_identity_at_converged_state():
    """The phi-step is an identity at a Poisson-consistent converged state
    (quasi-Fermi levels preserved, delta-phi ~ 0)."""
    from perovskite_sim.experiments.steady_state import (
        _TE_SOFTNESS, _qfl_poisson_relax)
    from perovskite_sim.solver.mol import build_material_arrays
    stack = _frozen_ion(_stack())
    x = _grid(stack)
    mat = dataclasses.replace(build_material_arrays(x, stack),
                              te_softness=_TE_SOFTNESS)
    r = solve_steady_state(x, stack, 0.5, illuminated=True, mat=mat)
    y2 = _qfl_poisson_relax(x, mat, r.y, 0.5)
    N = len(x)
    np.testing.assert_allclose(y2[: 2 * N], r.y[: 2 * N], rtol=1e-9)


def test_qfl_relax_preserves_negative_overshoots():
    """A transient-overshoot negative density passes through untouched —
    flooring it to hard zero made the RHS evaluate catastrophically
    (measured res 2e-3 -> 5.5e+02 at the V=0.8625 fallback)."""
    from perovskite_sim.experiments.steady_state import (
        _TE_SOFTNESS, _qfl_poisson_relax)
    from perovskite_sim.solver.mol import build_material_arrays
    stack = _frozen_ion(_stack())
    x = _grid(stack)
    mat = dataclasses.replace(build_material_arrays(x, stack),
                              te_softness=_TE_SOFTNESS)
    r = solve_steady_state(x, stack, 0.3, illuminated=True, mat=mat)
    y = r.y.copy()
    y[5] = -1.0e10          # inject an overshoot-style negative
    y2 = _qfl_poisson_relax(x, mat, y, 0.3)
    assert y2[5] == y[5]
