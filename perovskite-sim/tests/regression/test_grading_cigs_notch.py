"""Regression: Ga-rich back-graded CIGS absorber (Burgelman back-surface field).

A wider-gap, raised-conduction-band layer at the back contact repels electrons
from the (recombination-active) back contact, so V_oc should rise vs the
ungraded baseline WITHOUT J_sc collapsing — the canonical SCAPS graded-bandgap
result (Burgelman & Marlein 2008). The effect only appears when back-contact
recombination is significant, so the test sets a finite back-contact S; that is
the documented precondition, not a free win.

P1 thickness continuation identified a contact-orientation defect before the
grading comparison: the n-left stack used a positive Poisson boundary and the
p-left voltage mapping. The signed junction-polarity contract now certifies
the 0.5/1.0/2.0 um ungraded ladder. The grading comparison additionally keeps
each stack's configured contact-potential magnitude synchronized with its
material endpoints; holding one manual V_bi fixed while changing the outer
band edges is not a like-for-like contact model.
"""
import dataclasses
from pathlib import Path

import numpy as np
import pytest
import yaml

from perovskite_sim.constants import Q
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.experiments.jv_sweep import (
    JVCertificationError,
    build_electrical_grid,
    compute_metrics,
    run_jv_sweep,
)
from perovskite_sim.solver.mol import (
    build_material_arrays,
    poisson_right_boundary,
)


ROOT = Path(__file__).resolve().parents[2]


class CIGSSolverEnvelopeError(RuntimeError):
    """Expected, diagnosed failure of the bounded CIGS solver protocol."""

    def __init__(self, message, *, voltage=None, reason_code=None):
        super().__init__(message)
        self.voltage = voltage
        self.reason_code = reason_code


def _back_graded(stack, *, eg_step: float, chi_step: float, N_mult: int):
    """Return a copy with the absorber Ga-rich graded toward the back face:
    Eg_back = Eg + eg_step (wider gap), chi_back = chi - chi_step (raised E_C =
    electron back-surface field). Linear profile, mesh refined by N_mult."""
    new_layers = []
    for layer in stack.layers:
        if layer.role == "absorber":
            p = dataclasses.replace(
                layer.params,
                Eg_back=layer.params.Eg + eg_step,
                chi_back=layer.params.chi - chi_step,
                grading_profile="linear",
                grading_N_mult=N_mult,
            )
            new_layers.append(dataclasses.replace(layer, params=p))
        else:
            new_layers.append(layer)
    graded = dataclasses.replace(
        stack,
        layers=tuple(new_layers),
        band_grading=True,
    )
    return dataclasses.replace(graded, V_bi=abs(graded.compute_V_bi()))


def _with_absorber_thickness(stack, thickness_um: float):
    """Change only the absorber thickness for the P1 continuation ladder."""
    layers = tuple(
        dataclasses.replace(layer, thickness=thickness_um * 1e-6)
        if layer.role == "absorber"
        else layer
        for layer in stack.layers
    )
    return dataclasses.replace(stack, layers=layers)


def _ungraded_continuation_stack(thickness_um: float):
    """Rung zero: no Robin contact, grading, or graded-mesh multiplier."""
    base = load_device_from_yaml("configs/cigs_baseline.yaml")
    assert base.S_n_right is None
    assert not base.band_grading
    assert all(
        layer.params.grading_N_mult == 1
        for layer in base.layers
        if layer.role == "absorber"
    )
    return _with_absorber_thickness(base, thickness_um)


def _configured_stacks():
    base = load_device_from_yaml("configs/cigs_baseline.yaml")
    base = dataclasses.replace(base, S_n_right=1e3, mode="full")
    graded = _back_graded(base, eg_step=0.25, chi_step=0.15, N_mult=2)
    return base, graded


def _run_bounded(stack, *, label: str, **kwargs):
    """Translate only the documented numeric failure into the strict xfail."""
    try:
        return run_jv_sweep(stack, **kwargs)
    except JVCertificationError as exc:
        assert np.isfinite(exc.status.voltage)
        assert exc.status.reason_code in {
            "integration_failed",
            "current_extraction_failed",
            "nonfinite_candidate",
            "zero_bias_probe_failed",
            "zero_bias_confirmation_failed",
            "zero_bias_refinement_disagreed",
            "positive_bias_refinement_exhausted",
            "zero_crossing_refinement_exhausted",
        }
        if exc.status.reason_code == "integration_failed":
            assert "coupled solver failed to converge on [" in exc.status.message
            assert "after bisection" in exc.status.message
        raise CIGSSolverEnvelopeError(
            f"{label} branch was rejected by its J-V certificate at "
            f"V={exc.status.voltage:.6g} V ({exc.status.reason_code})",
            voltage=float(exc.status.voltage),
            reason_code=exc.status.reason_code,
        ) from exc
    except RuntimeError as exc:
        message = str(exc)
        assert message.startswith("JV sweep: coupled solver failed to converge on [")
        assert "after bisection" in message
        raise CIGSSolverEnvelopeError(
            f"{label} transient integration exhausted bounded bisection"
        ) from exc


@pytest.fixture(scope="module")
def cigs_ungraded_grid_ladder():
    """Certified 25 mV protocol on the registered nominal grid ladder."""
    stack = load_device_from_yaml("configs/cigs_baseline.yaml")
    results = {
        n_grid: _run_bounded(
            stack,
            label=f"ungraded-grid-{n_grid}",
            N_grid=n_grid,
            n_points=43,
            V_max=1.05,
            illuminated=True,
        )
        for n_grid in (40, 80, 120)
    }
    return stack, results


def _grading_pair(n_grid: int):
    base, graded = _configured_stacks()
    common = dict(
        N_grid=n_grid,
        n_points=43,
        V_max=1.05,
        illuminated=True,
    )
    return (
        base,
        graded,
        _run_bounded(base, label=f"ungraded-{n_grid}", **common),
        _run_bounded(graded, label=f"graded-{n_grid}", **common),
    )


@pytest.fixture(scope="module")
def cigs_grading_diagnostic_pair():
    return _grading_pair(40)


@pytest.fixture(scope="module")
def cigs_grading_production_pair():
    return _grading_pair(120)


def test_dynamic_cigs_grading_matches_the_shipped_config():
    """The slow diagnostic must exercise the config registered in the matrix."""
    _, dynamic_graded = _configured_stacks()
    shipped = load_device_from_yaml("configs/cigs_graded_notch.yaml")
    assert dynamic_graded == shipped


def test_cigs_n_left_voltage_conventions_are_explicit():
    """Positive V_app reduces the built-in drop for the n-left stack."""
    base = load_device_from_yaml("configs/cigs_baseline.yaml")
    x = build_electrical_grid(base, 40)
    mat = build_material_arrays(x, base)
    assert base.compute_V_bi() < 0.0
    assert base.V_bi > 0.0
    assert base.V_bi == pytest.approx(abs(base.compute_V_bi()))
    assert mat.junction_polarity == -1.0
    assert mat.V_bi_bc == pytest.approx(-base.V_bi)
    assert poisson_right_boundary(mat, 0.0) == pytest.approx(-base.V_bi)
    assert poisson_right_boundary(mat, 0.4) == pytest.approx(-base.V_bi + 0.4)


@pytest.mark.slow
@pytest.mark.parametrize("thickness_um", [0.5, 1.0, 2.0])
def test_cigs_ungraded_thickness_continuation_certifies(thickness_um):
    """Lock the repaired n-left contract on a bounded diagnostic grid.

    N_grid=40 is intentionally below the shipped N_grid>=120 validation hint,
    so this is an orientation/solver regression and not a mesh-convergence
    claim. The production-grid rung is tracked separately in the P1 matrix.
    """
    stack = _ungraded_continuation_stack(thickness_um)
    result = _run_bounded(
        stack,
        label=f"ungraded-{thickness_um:g}um",
        N_grid=40,
        n_points=40,
        V_max=0.975,
        illuminated=True,
    )
    assert result.certified
    assert result.metrics_fwd.voc_bracketed
    assert result.metrics_rev.voc_bracketed
    assert np.all(np.isfinite(result.J_fwd))
    assert np.all(np.isfinite(result.J_rev))
    photon_ceiling = Q * stack.Phi
    for metrics in (result.metrics_fwd, result.metrics_rev):
        assert 0.0 < metrics.J_sc <= photon_ceiling * (1.0 + 1.0e-6)
        assert 0.0 < metrics.V_oc < 0.975
        assert 0.0 < metrics.FF <= 1.0
        assert 0.0 < metrics.PCE <= 1.0


@pytest.mark.slow
def test_cigs_back_grading_raises_voc_without_jsc_collapse(
    cigs_grading_diagnostic_pair,
):
    # This N=40 diagnostic isolates model direction. Production-grid
    # certification is a separate gate below.
    _, _, res_off, res_on = cigs_grading_diagnostic_pair

    J_off = np.asarray(res_off.J_fwd)
    J_on = np.asarray(res_on.J_fwd)
    # Finiteness under the graded notch + refined mesh (the stability gate).
    assert np.all(np.isfinite(J_on)), "graded sweep produced non-finite J"
    assert np.all(np.isfinite(J_off))

    m_off = compute_metrics(np.asarray(res_off.V_fwd), J_off)
    m_on = compute_metrics(np.asarray(res_on.V_fwd), J_on)

    # J_sc must not collapse — back-grading is an electrical effect; with
    # uniform optics J_sc should be essentially unchanged (within 10%).
    assert m_on.J_sc == pytest.approx(m_off.J_sc, rel=0.10), (
        f"J_sc collapsed: {m_off.J_sc:.1f} -> {m_on.J_sc:.1f}"
    )
    # Electrical (BSF) win: V_oc rises (or at least does not fall) with the
    # Ga-rich back-surface field. Missing brackets are failures, never a pass.
    assert m_off.voc_bracketed, "ungraded CIGS curve did not bracket V_oc"
    assert m_on.voc_bracketed, "graded CIGS curve did not bracket V_oc"
    assert m_on.V_oc >= m_off.V_oc - 1e-3, (
        f"back-surface field did not help V_oc: {m_off.V_oc:.4f} -> {m_on.V_oc:.4f}"
    )
    for result in (res_off, res_on):
        assert result.certified
        assert abs(result.hysteresis_index) < 1.0e-3
        for metrics in (result.metrics_fwd, result.metrics_rev):
            assert 0.0 < metrics.FF <= 1.0
            assert 0.0 < metrics.PCE <= 1.0


@pytest.mark.slow
def test_cigs_back_grading_certifies_at_production_grid(
    cigs_grading_production_pair,
):
    base, graded, res_off, res_on = cigs_grading_production_pair
    assert len(build_electrical_grid(base, 120)) - 1 == 120
    assert len(build_electrical_grid(graded, 120)) - 1 == 160
    assert res_off.certified
    assert res_on.certified
    assert abs(res_off.hysteresis_index) < 1.0e-3
    assert abs(res_on.hysteresis_index) < 1.0e-3
    assert res_on.metrics_fwd.J_sc == pytest.approx(
        res_off.metrics_fwd.J_sc, rel=0.10,
    )
    assert res_on.metrics_fwd.V_oc >= res_off.metrics_fwd.V_oc + 1.0e-3
    for stack, result in ((base, res_off), (graded, res_on)):
        for metrics in (result.metrics_fwd, result.metrics_rev):
            assert metrics.voc_bracketed
            assert 0.0 < metrics.J_sc <= Q * stack.Phi * (1.0 + 1.0e-6)
            assert 0.0 < metrics.V_oc < 1.05
            assert 0.0 < metrics.FF <= 1.0
            assert 0.0 < metrics.PCE <= 1.0
        assert abs(result.metrics_fwd.V_oc - result.metrics_rev.V_oc) < 1.0e-4


@pytest.mark.slow
def test_cigs_ungraded_production_grid_ladder_is_physical(
    cigs_ungraded_grid_ladder,
):
    stack, results = cigs_ungraded_grid_ladder
    actual_intervals = tuple(
        len(build_electrical_grid(stack, n_grid)) - 1
        for n_grid in results
    )
    assert actual_intervals == (39, 78, 120)
    photon_ceiling = Q * stack.Phi
    for result in results.values():
        assert result.certified
        assert abs(result.hysteresis_index) < 1.0e-3
        for metrics in (result.metrics_fwd, result.metrics_rev):
            assert metrics.voc_bracketed
            assert 0.0 < metrics.J_sc <= photon_ceiling * (1.0 + 1.0e-6)
            assert 0.0 < metrics.V_oc < 1.05
            assert 0.0 < metrics.FF <= 1.0
            assert 0.0 < metrics.PCE <= 1.0
        assert abs(result.metrics_fwd.V_oc - result.metrics_rev.V_oc) < 1.0e-4
        for field in ("J_sc", "FF", "PCE"):
            fwd = getattr(result.metrics_fwd, field)
            rev = getattr(result.metrics_rev, field)
            assert abs(fwd - rev) / abs(fwd) < 1.0e-4


@pytest.mark.slow
def test_cigs_ungraded_production_grid_metrics_converge(
    cigs_ungraded_grid_ladder,
):
    _, results = cigs_ungraded_grid_ladder
    metrics = [results[n].metrics_fwd for n in (40, 80, 120)]
    for field in ("V_oc", "J_sc", "FF", "PCE"):
        values = [getattr(item, field) for item in metrics]
        coarse_gap = abs(values[1] - values[0])
        fine_gap = abs(values[2] - values[1])
        assert fine_gap < coarse_gap, (
            f"{field} grid error did not contract: {values}"
        )
        assert fine_gap / max(abs(values[2]), 1.0e-30) < 0.01, (
            f"{field} finest-pair change exceeds 1%: {values}"
        )


@pytest.mark.slow
def test_cigs_observations_match_reproducibility_registry(
    cigs_ungraded_grid_ladder,
    cigs_grading_production_pair,
):
    matrix = yaml.safe_load(
        (ROOT / "reproducibility/config_benchmark_matrix.yaml").read_text()
    )
    benchmark = matrix["benchmarks"]["cigs-internal-validation"]
    tolerance = benchmark["regression_tolerance"]

    def _assert_metrics(metrics, observed):
        assert metrics.V_oc == pytest.approx(
            observed["Voc_V"], abs=tolerance["Voc_V"],
        )
        assert metrics.J_sc == pytest.approx(
            observed["Jsc_A_m2"], abs=tolerance["Jsc_A_m2"],
        )
        assert metrics.FF == pytest.approx(
            observed["FF"], abs=tolerance["FF"],
        )
        assert 100.0 * metrics.PCE == pytest.approx(
            observed["PCE_percent"], abs=tolerance["PCE_percent"],
        )

    _, grid_results = cigs_ungraded_grid_ladder
    observed_grid = {
        item["N_grid"]: item
        for item in benchmark["observed_ungraded_forward"]
    }
    assert set(observed_grid) == set(grid_results)
    for n_grid, result in grid_results.items():
        _assert_metrics(result.metrics_fwd, observed_grid[n_grid])

    _, _, ungraded, graded = cigs_grading_production_pair
    observed_grading = benchmark["observed_production_grading"]
    _assert_metrics(
        ungraded.metrics_fwd, observed_grading["ungraded_robin"],
    )
    _assert_metrics(
        graded.metrics_fwd, observed_grading["graded_robin"],
    )
