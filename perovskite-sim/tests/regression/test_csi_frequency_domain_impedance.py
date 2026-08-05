"""Root-cause regression for c-Si depletion capacitance.

The former endpoint-sampled staircase transient returned the 180 um wafer's
geometric capacitance. This test requires the frequency-domain QF operator to
recover the independently observed carrier-inventory/depletion scale instead.
It is an internal numerical certificate, not external C-V validation.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from perovskite_sim.constants import EPS_0, Q
from perovskite_sim.experiments.jv_sweep import build_electrical_grid
from perovskite_sim.experiments.mott_schottky import _fit_mott_schottky
from perovskite_sim.experiments.quasi_fermi_impedance import (
    run_quasi_fermi_impedance,
)
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.solver.mol import build_material_arrays


pytestmark = [pytest.mark.slow, pytest.mark.regression]
ROOT = Path(__file__).resolve().parents[2]
MATRIX = yaml.safe_load(
    (ROOT / "reproducibility/config_benchmark_matrix.yaml").read_text()
)
BENCHMARK = MATRIX["benchmarks"]["csi-qf-frequency-domain-cv"]
PROTOCOL = BENCHMARK["protocol"]
OBSERVED = BENCHMARK["observed"]
TOLERANCE = BENCHMARK["regression_tolerance"]
FREQUENCIES_HZ = np.asarray(PROTOCOL["frequencies_Hz"], dtype=float)
DEPLETION_BIASES_V = np.asarray(PROTOCOL["biases_V"], dtype=float)
GRID_LADDER = tuple(PROTOCOL["N_grid"])


@pytest.fixture(scope="module")
def csi_frequency_response():
    stack = load_device_from_yaml("configs/cSi_homojunction.yaml")
    x = build_electrical_grid(stack, 200)
    mat = build_material_arrays(x, stack)
    response = run_quasi_fermi_impedance(
        x,
        stack,
        FREQUENCIES_HZ,
        V_dc=-0.2,
        delta_V=0.01,
        illuminated=False,
        mat=mat,
    )
    return stack, x, mat, response


def _capacitance(response) -> np.ndarray:
    admittance = 1.0 / response.Z
    return admittance.imag / (2.0 * np.pi * FREQUENCIES_HZ)


def test_csi_frequency_domain_recovers_depletion_not_geometric_capacitance(
    csi_frequency_response,
):
    stack, x, _mat, response = csi_frequency_response
    capacitance = _capacitance(response)
    central = float(capacitance[1])
    analytic = np.sqrt(
        Q
        * EPS_0
        * 11.7
        * 1.0e22
        / (2.0 * (abs(stack.compute_V_bi()) - (-0.2)))
    )
    geometric = EPS_0 * 11.7 / float(x[-1] - x[0])

    assert response.dc_state.certified
    assert response.dc_state.V_app == pytest.approx(-0.2, abs=1.0e-14)
    assert not response.dc_state.illuminated
    assert central == pytest.approx(2.886e-4, rel=5.0e-3)
    assert central == pytest.approx(analytic, rel=0.1)
    assert central > 400.0 * geometric


def test_csi_frequency_domain_has_capacitance_plateau_and_current_certificate(
    csi_frequency_response,
):
    _stack, _x, _mat, response = csi_frequency_response
    capacitance = _capacitance(response)
    admittance = 1.0 / response.Z

    assert np.all(np.isfinite(response.Z))
    assert np.all(response.Z.imag < 0.0)
    assert np.all(admittance.imag > 0.0)
    assert np.ptp(capacitance) / capacitance[1] < 1.0e-4
    assert np.max(response.max_relative_face_spread) < 5.0e-4
    assert response.max_relative_face_spread[1] < 1.0e-4
    assert np.all(response.reciprocal_condition > 0.0)
    assert np.max(response.backward_error) < 1.0e-10
    assert 0.0 <= admittance[1].real / admittance[1].imag < 1.0e-3


def test_csi_frequency_domain_is_stable_to_steps_and_nominal_amplitude(
    csi_frequency_response,
):
    stack, x, mat, baseline = csi_frequency_response
    amplitude = run_quasi_fermi_impedance(
        x,
        stack,
        np.array([1.0e5]),
        V_dc=-0.2,
        delta_V=0.005,
        illuminated=False,
        mat=mat,
        dc_state=baseline.dc_state,
    )
    refined = run_quasi_fermi_impedance(
        x,
        stack,
        np.array([1.0e5]),
        V_dc=-0.2,
        delta_V=0.01,
        illuminated=False,
        mat=mat,
        dc_state=baseline.dc_state,
        state_step=5.0e-6,
        voltage_step=5.0e-6,
    )
    baseline_c = _capacitance(baseline)[1]
    assert amplitude.Z[0] == pytest.approx(baseline.Z[1], rel=1.0e-12)
    refined_y = 1.0 / refined.Z[0]
    refined_c = refined_y.imag / (2.0 * np.pi * 1.0e5)
    assert refined_c == pytest.approx(baseline_c, rel=1.0e-4)


@pytest.fixture(scope="module")
def csi_cv_ladder():
    stack = load_device_from_yaml("configs/cSi_homojunction.yaml")
    rows = []
    for requested in GRID_LADDER:
        x = build_electrical_grid(stack, requested)
        mat = build_material_arrays(x, stack)
        capacitance = np.empty(
            (FREQUENCIES_HZ.size, DEPLETION_BIASES_V.size),
            dtype=float,
        )
        spreads = np.empty_like(capacitance)
        reciprocal_condition = np.empty_like(capacitance)
        backward_error = np.empty_like(capacitance)
        for column, voltage in enumerate(DEPLETION_BIASES_V):
            response = run_quasi_fermi_impedance(
                x,
                stack,
                FREQUENCIES_HZ,
                V_dc=float(voltage),
                delta_V=0.01,
                illuminated=False,
                mat=mat,
            )
            admittance = 1.0 / response.Z
            capacitance[:, column] = admittance.imag / (
                2.0 * np.pi * FREQUENCIES_HZ
            )
            spreads[:, column] = response.max_relative_face_spread
            reciprocal_condition[:, column] = response.reciprocal_condition
            backward_error[:, column] = response.backward_error
        fits = tuple(
            _fit_mott_schottky(
                DEPLETION_BIASES_V,
                capacitance[index],
                eps_r=11.7,
                T=300.0,
            )
            for index in range(FREQUENCIES_HZ.size)
        )
        rows.append(
            (
                capacitance,
                spreads,
                reciprocal_condition,
                backward_error,
                fits,
            )
        )
    return stack, tuple(rows)


def test_csi_depletion_cv_curve_converges_on_registered_grid_ladder(
    csi_cv_ladder,
):
    _stack, rows = csi_cv_ladder
    central_curves = tuple(row[0][1] for row in rows)
    changes = tuple(
        float(np.max(np.abs(fine - coarse) / fine))
        for coarse, fine in zip(central_curves[:-1], central_curves[1:])
    )

    for curve in central_curves:
        assert np.all(np.isfinite(curve))
        assert np.all(curve > 0.0)
        assert np.all(np.diff(curve) > 0.0)
        assert np.all(np.diff(1.0 / (curve * curve)) < 0.0)
    assert changes[1] < changes[0]
    assert changes[1] < 5.0e-3

    middle_fit = rows[1][4][1]
    fine_fit = rows[2][4][1]
    assert abs(fine_fit[0] - middle_fit[0]) < 2.0e-3
    assert abs(fine_fit[1] - middle_fit[1]) / fine_fit[1] < 1.0e-2
    assert fine_fit[1] == pytest.approx(1.0e22, rel=0.1)
    # The internally converged 0.756 V intercept remains below the configured
    # 0.893 V contact potential. Keep that discrepancy visible until the
    # finite-junction/contact interpretation is independently resolved.
    assert 0.7 < fine_fit[0] < 0.8


def test_csi_depletion_cv_has_frequency_and_all_face_plateaus(csi_cv_ladder):
    _stack, rows = csi_cv_ladder
    capacitance, _spreads, _condition, _backward_error, fits = rows[-1]
    frequency_change = np.max(
        np.ptp(capacitance, axis=0) / capacitance[1]
    )
    vbi = np.asarray([fit[0] for fit in fits])
    doping = np.asarray([fit[1] for fit in fits])

    assert frequency_change < 1.0e-3
    assert np.ptp(vbi) / vbi[1] < 1.0e-3
    assert np.ptp(doping) / doping[1] < 1.0e-3
    # This includes the highly conducting emitter at +0.2 V and 10 kHz,
    # where an unbalanced complex solve loses the small AC current difference.
    certificates = PROTOCOL["certificates"]
    assert max(float(np.max(row[1])) for row in rows) < (
        certificates["max_relative_all_face_admittance_spread"]
    )
    assert min(float(np.min(row[2])) for row in rows) > 0.0
    assert max(float(np.max(row[3])) for row in rows) < (
        certificates["max_linear_solve_backward_error"]
    )


def test_csi_frequency_domain_observations_match_registry(csi_cv_ladder):
    _stack, rows = csi_cv_ladder
    for requested, row, registered in zip(
        GRID_LADDER,
        rows,
        OBSERVED["grids"],
    ):
        capacitance, _spreads, _condition, _backward_error, fits = row
        assert int(registered["N_grid"]) == requested
        assert capacitance[1] == pytest.approx(
            registered["capacitance_100k_F_m2"],
            rel=TOLERANCE["capacitance_relative"],
        )
        assert fits[1][0] == pytest.approx(
            registered["mott_intercept_100k_V"],
            abs=TOLERANCE["mott_intercept_V"],
        )
        assert fits[1][1] == pytest.approx(
            registered["effective_doping_100k_m3"],
            rel=TOLERANCE["effective_doping_relative"],
        )

    central_curves = tuple(row[0][1] for row in rows)
    actual_changes = [
        float(np.max(np.abs(fine - coarse) / fine))
        for coarse, fine in zip(central_curves[:-1], central_curves[1:])
    ]
    assert actual_changes == pytest.approx(
        OBSERVED["adjacent_max_capacitance_changes"],
        rel=TOLERANCE["convergence_metric_relative"],
    )
    fine_capacitance, _spreads, _condition, _backward_error, fine_fits = rows[-1]
    assert np.max(
        np.ptp(fine_capacitance, axis=0) / fine_capacitance[1]
    ) == pytest.approx(
        OBSERVED["finest_max_frequency_change"],
        rel=TOLERANCE["convergence_metric_relative"],
    )
    assert [fit[0] for fit in fine_fits] == pytest.approx(
        OBSERVED["finest_mott_intercepts_V"],
        abs=TOLERANCE["mott_intercept_V"],
    )
    assert [fit[1] for fit in fine_fits] == pytest.approx(
        OBSERVED["finest_effective_doping_m3"],
        rel=TOLERANCE["effective_doping_relative"],
    )
