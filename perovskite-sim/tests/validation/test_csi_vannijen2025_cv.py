"""Partial external c-Si C-V comparison against van Nijen et al. (2025).

The source is an independent 2-D Sentaurus calculation with partial-width
contacts. SolarLab is a 1-D full-area model, so this suite certifies source
extraction, input mapping, local convergence, and the observed discrepancy. It
deliberately does not claim pointwise parity or close the strict external gap.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from perovskite_sim.experiments.jv_sweep import build_electrical_grid
from perovskite_sim.experiments.quasi_fermi_impedance import (
    run_quasi_fermi_impedance,
)
from perovskite_sim.experiments.quasi_fermi_steady_state import (
    solve_quasi_fermi_steady_state,
)
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.physics.doping import doping_at_position
from perovskite_sim.solver.mol import build_material_arrays


pytestmark = [pytest.mark.slow, pytest.mark.validation]
ROOT = Path(__file__).resolve().parents[2]
MATRIX = yaml.safe_load(
    (ROOT / "reproducibility/config_benchmark_matrix.yaml").read_text()
)
BENCHMARK = MATRIX["benchmarks"]["vannijen2025-csi-pn-cv"]
PROTOCOL = BENCHMARK["protocol"]
OBSERVED = BENCHMARK["observed"]
TOLERANCE = BENCHMARK["regression_tolerance"]
REFERENCE = yaml.safe_load(
    (
        ROOT
        / "perovskite_sim/data/references/csi_vannijen2025_pn_cv.yaml"
    ).read_text()
)
ALL_SOURCE_ROWS = tuple(REFERENCE["selected_rows"])
SOURCE_ROWS = tuple(row for row in ALL_SOURCE_ROWS if row["used_for_benchmark"])
SOURCE_CAPACITANCE = np.asarray(
    [row["capacitance_F_m2"] for row in SOURCE_ROWS], dtype=float
)
FREQUENCY_HZ = float(PROTOCOL["source"]["selected_frequency_Hz"])
BIASES_V = np.asarray(PROTOCOL["local"]["biases_V"], dtype=float)
GRID_LADDER = tuple(PROTOCOL["local"]["N_grid"])


def _extract_source_capacitance(row: dict) -> float:
    admittance = complex(
        float(row["conductance_S_cm2"]),
        float(row["susceptance_S_cm2"]),
    )
    impedance = 1.0 / admittance
    corrected = 1.0 / (
        impedance - float(row["series_resistance_ohm_cm2"])
    )
    capacitance_F_cm2 = corrected.imag / (
        2.0 * np.pi * float(row["frequency_Hz"])
    )
    return float(capacitance_F_cm2 * 1.0e4)


@pytest.fixture(scope="module")
def local_cv_ladder():
    stack = load_device_from_yaml("configs/csi_vannijen2025_pn_cv.yaml")
    rows = []
    for requested in GRID_LADDER:
        x = build_electrical_grid(stack, requested)
        mat = build_material_arrays(x, stack)
        capacitance = []
        dc_states = []
        responses = []
        for bias in BIASES_V:
            dc_state = solve_quasi_fermi_steady_state(
                x,
                stack,
                V_app=float(bias),
                illuminated=False,
                mat=mat,
                newton_residual_tolerance=float(
                    PROTOCOL["local"]["newton_residual_tolerance"]
                ),
            )
            response = run_quasi_fermi_impedance(
                x,
                stack,
                np.asarray([FREQUENCY_HZ]),
                V_dc=float(bias),
                delta_V=float(PROTOCOL["local"]["nominal_delta_V"]),
                illuminated=False,
                mat=mat,
                dc_state=dc_state,
            )
            admittance = 1.0 / response.Z[0]
            capacitance.append(
                float(admittance.imag / (2.0 * np.pi * FREQUENCY_HZ))
            )
            dc_states.append(dc_state)
            responses.append(response)
        rows.append(
            {
                "x": x,
                "capacitance": np.asarray(capacitance),
                "dc_states": tuple(dc_states),
                "responses": tuple(responses),
            }
        )
    return stack, tuple(rows)


def test_vannijen_source_rows_reproduce_frozen_author_extraction():
    assert [row["bias_V"] for row in SOURCE_ROWS] == pytest.approx(BIASES_V)
    assert all(row["frequency_Hz"] == FREQUENCY_HZ for row in ALL_SOURCE_ROWS)
    assert all(
        len(row["source_file_sha256"]) == 64
        and set(row["source_file_sha256"]) <= set("0123456789abcdef")
        for row in ALL_SOURCE_ROWS
    )
    extracted = np.asarray(
        [_extract_source_capacitance(row) for row in ALL_SOURCE_ROWS]
    )
    frozen = np.asarray(
        [row["capacitance_F_m2"] for row in ALL_SOURCE_ROWS], dtype=float
    )
    np.testing.assert_allclose(extracted, frozen, rtol=2.0e-15)
    assert REFERENCE["extraction"]["calibration_parameters"] == []


def test_vannijen_config_maps_published_profile_without_curve_calibration():
    stack = load_device_from_yaml("configs/csi_vannijen2025_pn_cv.yaml")
    source = REFERENCE["source_model"]
    emitter, base = stack.layers

    assert BENCHMARK["claim_level"] == "partial_external_comparison"
    assert BENCHMARK["calibration"]["parameters"] == []
    assert stack.total_thickness == pytest.approx(source["device_thickness_m"])
    assert stack.T == pytest.approx(source["temperature_K"])
    assert base.params.N_D == pytest.approx(source["bulk_donor_density_m3"])
    assert emitter.params.N_A == pytest.approx(
        source["emitter_peak_acceptor_density_m3"]
    )
    assert stack.compute_V_bi() == pytest.approx(stack.V_bi, abs=1.0e-14)

    junction = float(source["junction_depth_m"])
    N_A_front, N_D_front = doping_at_position(
        emitter.params, 0.0, emitter.thickness
    )
    N_A_junction, N_D_junction = doping_at_position(
        emitter.params, junction, emitter.thickness
    )
    assert N_A_front == pytest.approx(1.0e25)
    assert N_D_front == pytest.approx(1.0e21)
    assert N_A_junction == pytest.approx(N_D_junction, rel=2.0e-14)


def test_vannijen_local_cv_is_residual_certified_and_grid_converged(
    local_cv_ladder,
):
    _stack, rows = local_cv_ladder
    certificates = PROTOCOL["local"]["certificates"]
    curves = tuple(row["capacitance"] for row in rows)
    changes = tuple(
        float(np.max(np.abs(fine - coarse) / fine))
        for coarse, fine in zip(curves[:-1], curves[1:])
    )

    for curve, row in zip(curves, rows):
        assert np.all(np.isfinite(curve))
        assert np.all(curve > 0.0)
        assert np.all(np.diff(curve) > 0.0)
        for dc_state, response in zip(row["dc_states"], row["responses"]):
            assert dc_state.certified
            assert dc_state.max_normalized_cell_residual < (
                certificates["max_dc_normalized_cell_residual"]
            )
            assert max(
                dc_state.electron_continuity_bound_A_m2,
                dc_state.hole_continuity_bound_A_m2,
            ) < certificates["max_continuity_bound_A_m2"]
            assert dc_state.face_current_spread_A_m2 < (
                certificates["max_dc_face_current_spread_A_m2"]
            )
            assert response.Z[0].imag < 0.0
            assert (1.0 / response.Z[0]).imag > 0.0
            assert response.max_relative_face_spread[0] < (
                certificates["max_ac_relative_face_spread"]
            )
            assert response.reciprocal_condition[0] > 0.0
            assert response.backward_error[0] < (
                certificates["max_linear_solve_backward_error"]
            )

    assert changes[1] < changes[0]
    assert changes[1] < certificates["max_finest_grid_curve_change"]


def test_vannijen_cv_records_partial_agreement_not_pointwise_parity(
    local_cv_ladder,
):
    _stack, rows = local_cv_ladder
    finest = rows[-1]["capacitance"]
    relative_error = np.abs(finest - SOURCE_CAPACITANCE) / SOURCE_CAPACITANCE

    assert BENCHMARK["status"] == "partial"
    assert np.all(finest > SOURCE_CAPACITANCE)
    # This lower bound makes the evidence label executable: the current result
    # is not allowed to drift into a false pointwise-parity claim unnoticed.
    assert np.max(relative_error) > PROTOCOL["local"][
        "pointwise_parity_relative_error"
    ]


def test_vannijen_observations_match_reproducibility_registry(
    local_cv_ladder,
):
    _stack, rows = local_cv_ladder
    curves = tuple(row["capacitance"] for row in rows)
    relative_error = np.abs(curves[-1] - SOURCE_CAPACITANCE) / SOURCE_CAPACITANCE
    changes = [
        float(np.max(np.abs(fine - coarse) / fine))
        for coarse, fine in zip(curves[:-1], curves[1:])
    ]

    np.testing.assert_allclose(
        SOURCE_CAPACITANCE,
        OBSERVED["source_capacitance_F_m2"],
        rtol=TOLERANCE["source_capacitance_F_m2"],
    )
    for requested, curve in zip(GRID_LADDER, curves):
        np.testing.assert_allclose(
            curve,
            OBSERVED["local_capacitance_F_m2"][str(requested)],
            rtol=TOLERANCE["local_capacitance_F_m2"],
        )
    np.testing.assert_allclose(
        relative_error,
        OBSERVED["pointwise_relative_error_N800"],
        rtol=TOLERANCE["pointwise_relative_error_N800"],
    )
    assert float(np.max(relative_error)) == pytest.approx(
        OBSERVED["max_pointwise_relative_error"],
        rel=TOLERANCE["max_pointwise_relative_error"],
    )
    assert float(np.sqrt(np.mean(relative_error**2))) == pytest.approx(
        OBSERVED["rms_pointwise_relative_error"],
        rel=TOLERANCE["rms_pointwise_relative_error"],
    )
    assert changes == pytest.approx(
        OBSERVED["adjacent_max_local_curve_changes"],
        rel=TOLERANCE["adjacent_max_local_curve_changes"],
    )
