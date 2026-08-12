"""Residual-certified c-Si QF J-V and grid convergence.

The certificate is limited to the opt-in local homojunction QF solver. It
does not establish convergence of the default transient/algebraic drivers or
provide an external c-Si device validation.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import yaml

from perovskite_sim.constants import Q
from perovskite_sim.experiments.jv_sweep import (
    build_electrical_grid,
    compute_metrics,
)
from perovskite_sim.experiments.quasi_fermi_steady_state import (
    solve_quasi_fermi_jv_sweep,
)
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.physics.generation import beer_lambert_generation
from perovskite_sim.solver.mol import build_material_arrays


pytestmark = [pytest.mark.slow, pytest.mark.regression]
ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/cSi_homojunction.yaml"
MATRIX = yaml.safe_load(
    (ROOT / "reproducibility/config_benchmark_matrix.yaml").read_text()
)
BENCHMARK = MATRIX["benchmarks"]["csi-qf-internal-validation"]
PROTOCOL = BENCHMARK["protocol"]
OBSERVED = BENCHMARK["observed"]
TOLERANCE = BENCHMARK["regression_tolerance"]
LADDER = tuple(PROTOCOL["N_grid"])
VOLTAGES_V = np.asarray(PROTOCOL["voltage_grid_V"], dtype=float)


@pytest.fixture(scope="module")
def certified_jv_ladder():
    stack = load_device_from_yaml(CONFIG)
    rows = []
    for requested in LADDER:
        x = build_electrical_grid(stack, requested)
        sweep = solve_quasi_fermi_jv_sweep(x, stack, VOLTAGES_V)
        mat = build_material_arrays(x, stack)
        generation = mat.G_optical
        if generation is None:
            generation = beer_lambert_generation(x, mat.alpha, stack.Phi)
        photon_budget = float(Q * np.sum(generation * mat.dx_cell))
        rows.append((x, mat, sweep, photon_budget))
    return stack, tuple(rows)


def _grid_observation(requested: int) -> dict:
    return next(
        row for row in OBSERVED["grids"] if int(row["N_grid"]) == requested
    )


def _sign_change(sweep) -> tuple[int, float]:
    crossings = np.flatnonzero(
        (sweep.currents_A_m2[:-1] > 0.0)
        & (sweep.currents_A_m2[1:] <= 0.0)
    )
    assert crossings.size == 1
    index = int(crossings[0])
    v_lo, v_hi = sweep.voltages_V[index : index + 2]
    j_lo, j_hi = sweep.currents_A_m2[index : index + 2]
    root = float(v_lo - j_lo * (v_hi - v_lo) / (j_hi - j_lo))
    return index, root


def _nested_10mv_metrics(sweep):
    refinement = PROTOCOL["mpp_refinement"]
    fine = {
        round(float(value), 6)
        for value in np.arange(
            refinement["start_V"],
            refinement["stop_V"] + 0.5 * refinement["step_V"],
            refinement["step_V"],
        )
    }
    coarse = {
        round(float(value), 6)
        for value in np.arange(
            refinement["start_V"],
            refinement["stop_V"]
            + 0.5 * PROTOCOL["nested_mpp_check_step_V"],
            PROTOCOL["nested_mpp_check_step_V"],
        )
    }
    keep = np.array(
        [round(float(value), 6) not in fine or round(float(value), 6) in coarse
         for value in sweep.voltages_V],
        dtype=bool,
    )
    return compute_metrics(
        sweep.voltages_V[keep],
        sweep.currents_A_m2[keep],
    )


def test_csi_config_and_registered_voltage_protocol(certified_jv_ladder):
    stack, rows = certified_jv_ladder
    assert stack.V_bi == pytest.approx(abs(stack.compute_V_bi()), abs=1.0e-14)
    assert stack.grid_interval_weights == (1.0, 4.0)
    assert stack.grid_alphas == (2.0, 3.0)
    assert VOLTAGES_V[0] == 0.0
    assert VOLTAGES_V[-1] == 0.6
    assert np.all(np.diff(VOLTAGES_V) > 0.0)
    canonical = ",".join(f"{value:.6f}" for value in VOLTAGES_V)
    assert hashlib.sha256(canonical.encode("ascii")).hexdigest() == (
        PROTOCOL["voltage_grid_canonical_sha256"]
    )

    interface = stack.layers[0].thickness
    for requested, (x, _mat, sweep, _budget) in zip(LADDER, rows):
        observation = _grid_observation(requested)
        assert len(x) - 1 == requested
        interface_index = int(np.argmin(np.abs(x - interface)))
        assert x[interface_index] == pytest.approx(interface, abs=1.0e-18)
        assert [interface_index, requested - interface_index] == (
            observation["layer_intervals"]
        )
        assert sweep.voltages_V == pytest.approx(VOLTAGES_V)


def test_every_csi_jv_point_has_independent_physics_certificate(
    certified_jv_ladder,
):
    _stack, rows = certified_jv_ladder
    gates = PROTOCOL["certificates"]
    for requested, (x, mat, sweep, _budget) in zip(LADDER, rows):
        assert sweep.certified, requested
        assert sweep.metrics_certified, requested
        for point in sweep.points:
            assert point.max_normalized_cell_residual < (
                gates["normalized_cell_residual"]
            )
            assert point.electron_continuity_bound_A_m2 < (
                gates["continuity_bound_A_m2"]
            )
            assert point.hole_continuity_bound_A_m2 < (
                gates["continuity_bound_A_m2"]
            )
            assert point.face_current_spread_A_m2 < (
                gates["all_face_current_spread_A_m2"]
            )
            assert point.poisson_residual < gates["normalized_poisson_residual"]
            assert point.face_current_spread_A_m2 == pytest.approx(
                np.ptp(point.total_face_current_A_m2),
                abs=1.0e-15,
            )
            assert point.current_A_m2 == point.total_face_current_A_m2[0]

            actual = np.diff(point.total_face_current_A_m2)
            expected = -mat.junction_polarity * Q * mat.dx_cell[1:-1] * (
                point.electron_rate_per_s[1:-1]
                - point.hole_rate_per_s[1:-1]
            )
            assert actual == pytest.approx(expected, abs=1.0e-9)

            node_count = len(x)
            log_n = (
                point.electron_quasi_fermi_potential_V + point.phi + mat.chi
            ) / mat.V_T_device
            log_p = (
                point.hole_quasi_fermi_potential_V
                - point.phi
                - mat.chi
                - mat.Eg
            ) / mat.V_T_device
            assert np.log(point.y[:node_count]) == pytest.approx(
                log_n,
                abs=1.0e-11,
            )
            assert np.log(point.y[node_count : 2 * node_count]) == pytest.approx(
                log_p,
                abs=1.0e-11,
            )


def test_csi_jv_curves_are_physical_and_bracket_open_circuit(
    certified_jv_ladder,
):
    _stack, rows = certified_jv_ladder
    for requested, (_x, _mat, sweep, photon_budget) in zip(LADDER, rows):
        assert np.all(np.diff(sweep.currents_A_m2) < 0.0), requested
        assert 0.0 < sweep.metrics.J_sc <= photon_budget, requested
        assert sweep.currents_A_m2[-1] < 0.0, requested
        assert 0.0 < sweep.metrics.FF < 1.0, requested
        assert 0.0 < sweep.metrics.PCE < 1.0, requested
        crossing, root = _sign_change(sweep)
        assert sweep.voltages_V[crossing] < root < sweep.voltages_V[crossing + 1]


def test_csi_voltage_sampling_resolves_maximum_power(certified_jv_ladder):
    _stack, rows = certified_jv_ladder
    for requested, (_x, _mat, sweep, _budget) in zip(LADDER, rows):
        coarse = _nested_10mv_metrics(sweep)
        relative_change = abs(sweep.metrics.PCE - coarse.PCE) / sweep.metrics.PCE
        assert relative_change < 1.0e-3, requested
        assert sweep.metrics.PCE >= coarse.PCE, requested


def test_csi_full_jv_curve_and_metrics_converge(certified_jv_ladder):
    _stack, rows = certified_jv_ladder
    sweeps = tuple(row[2] for row in rows)
    curve_changes = tuple(
        float(np.max(np.abs(fine.currents_A_m2 - coarse.currents_A_m2)))
        for coarse, fine in zip(sweeps[:-1], sweeps[1:])
    )
    assert curve_changes[1] < curve_changes[0]
    assert curve_changes[1] / abs(sweeps[-1].metrics.J_sc) < 1.0e-2

    fields = ("J_sc", "V_oc", "FF", "PCE")
    for field in fields:
        values = tuple(float(getattr(sweep.metrics, field)) for sweep in sweeps)
        assert abs(values[2] - values[1]) < abs(values[1] - values[0]), field

    coarse, fine = sweeps[-2:]
    assert abs(fine.metrics.J_sc - coarse.metrics.J_sc) / fine.metrics.J_sc < 0.01
    assert abs(fine.metrics.V_oc - coarse.metrics.V_oc) < 1.0e-3
    assert abs(fine.metrics.FF - coarse.metrics.FF) / fine.metrics.FF < 5.0e-3
    assert abs(fine.metrics.PCE - coarse.metrics.PCE) / fine.metrics.PCE < 0.01

    registered = OBSERVED["finest_changes"]
    actual = {
        "Jsc_relative": abs(fine.metrics.J_sc - coarse.metrics.J_sc)
        / fine.metrics.J_sc,
        "Voc_absolute_V": abs(fine.metrics.V_oc - coarse.metrics.V_oc),
        "FF_relative": abs(fine.metrics.FF - coarse.metrics.FF) / fine.metrics.FF,
        "PCE_relative": abs(fine.metrics.PCE - coarse.metrics.PCE)
        / fine.metrics.PCE,
        "max_curve_current_relative_to_Jsc": curve_changes[1]
        / abs(fine.metrics.J_sc),
        "curve_change_contraction": curve_changes[1] / curve_changes[0],
        "max_nested_mpp_PCE_relative": max(
            abs(sweep.metrics.PCE - _nested_10mv_metrics(sweep).PCE)
            / sweep.metrics.PCE
            for sweep in sweeps
        ),
    }
    assert actual == pytest.approx(registered, rel=1.0e-8, abs=1.0e-12)


def test_csi_jv_observations_match_reproducibility_registry(
    certified_jv_ladder,
):
    _stack, rows = certified_jv_ladder
    curve_changes = []
    for requested, (_x, _mat, sweep, _budget) in zip(LADDER, rows):
        observation = _grid_observation(requested)
        coarse = _nested_10mv_metrics(sweep)
        crossing, sign_root = _sign_change(sweep)
        assert sweep.metrics.J_sc == pytest.approx(
            observation["Jsc_A_m2"], abs=TOLERANCE["Jsc_A_m2"]
        )
        assert sweep.metrics.V_oc == pytest.approx(
            observation["Voc_V"], abs=TOLERANCE["Voc_V"]
        )
        assert sign_root == pytest.approx(
            observation["sign_change_Voc_V"],
            abs=TOLERANCE["sign_change_Voc_V"],
        )
        assert sweep.voltages_V[crossing : crossing + 2] == pytest.approx(
            observation["sign_bracket_V"]
        )
        assert sweep.metrics.FF == pytest.approx(
            observation["FF"], abs=TOLERANCE["FF"]
        )
        assert 100.0 * sweep.metrics.PCE == pytest.approx(
            observation["PCE_percent"], abs=TOLERANCE["PCE_percent"]
        )
        assert 100.0 * coarse.PCE == pytest.approx(
            observation["nested_10mV_PCE_percent"],
            abs=TOLERANCE["PCE_percent"],
        )
        assert max(
            max(
                point.electron_continuity_bound_A_m2,
                point.hole_continuity_bound_A_m2,
            )
            for point in sweep.points
        ) == pytest.approx(
            observation["max_continuity_bound_A_m2"],
            abs=TOLERANCE["diagnostic_A_m2"],
        )
        assert max(
            point.face_current_spread_A_m2 for point in sweep.points
        ) == pytest.approx(
            observation["max_all_face_current_spread_A_m2"],
            abs=TOLERANCE["diagnostic_A_m2"],
        )
        assert max(point.poisson_residual for point in sweep.points) == pytest.approx(
            observation["max_normalized_poisson_residual"],
            abs=1.0e-9,
        )

    for coarse, fine in zip((row[2] for row in rows[:-1]), (row[2] for row in rows[1:])):
        delta = fine.currents_A_m2 - coarse.currents_A_m2
        curve_changes.append(
            (
                float(np.max(np.abs(delta))),
                float(np.sqrt(np.mean(delta * delta))),
            )
        )
    for actual, observation in zip(
        curve_changes,
        OBSERVED["adjacent_curve_changes"],
    ):
        assert actual[0] == pytest.approx(
            observation["max_abs_current_A_m2"],
            abs=TOLERANCE["diagnostic_A_m2"],
        )
        assert actual[1] == pytest.approx(
            observation["rms_current_A_m2"],
            abs=TOLERANCE["diagnostic_A_m2"],
        )
