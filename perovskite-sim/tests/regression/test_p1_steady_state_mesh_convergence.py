"""P1 residual-certified steady-state grid convergence at short circuit."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from perovskite_sim.experiments.jv_sweep import (
    _compute_current_ss_with_spread,
    build_electrical_grid,
)
from perovskite_sim.experiments.steady_state import solve_steady_state
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.validation.grid_convergence import (
    ConvergenceSeries,
    prolong_packed_state,
)


pytestmark = [pytest.mark.slow, pytest.mark.regression]
CONFIG = "configs/ionmonger_benchmark.yaml"
COARSE_LADDER = (12, 24, 48)
PROLONGED_LADDER = (48, 72, 96)
MAX_RESIDUAL_PER_S = 0.5
MAX_CONTINUITY_CURRENT_A_M2 = 0.1
MAX_CURRENT_SPREAD_A_M2 = 0.01


@pytest.fixture(scope="module")
def certified_ladder():
    stack = load_device_from_yaml(CONFIG)
    rows = []
    source_x = None
    source_result = None
    for requested in COARSE_LADDER:
        x = build_electrical_grid(stack, requested)
        result = solve_steady_state(
            x,
            stack,
            V_app=0.0,
            illuminated=True,
            tol_accept=MAX_RESIDUAL_PER_S,
            max_continuity_current_error=MAX_CONTINUITY_CURRENT_A_M2,
        )
        current, spread = _compute_current_ss_with_spread(
            x, result.y, stack, V_app=0.0,
        )
        rows.append((len(x) - 1, result, current, spread))
        if requested == 48:
            source_x, source_result = x, result

    assert source_x is not None and source_result is not None
    for requested in PROLONGED_LADDER[1:]:
        x = build_electrical_grid(stack, requested)
        y0 = prolong_packed_state(source_x, source_result.y, x, stack)
        result = solve_steady_state(
            x,
            stack,
            V_app=0.0,
            illuminated=True,
            y0=y0,
            tol_accept=MAX_RESIDUAL_PER_S,
            max_continuity_current_error=MAX_CONTINUITY_CURRENT_A_M2,
            relative_log_variables=True,
        )
        current, spread = _compute_current_ss_with_spread(
            x, result.y, stack, V_app=0.0,
        )
        rows.append((len(x) - 1, result, current, spread))
    return tuple(rows)


def test_actual_interval_ladder_and_residual_certificates(certified_ladder):
    assert tuple(row[0] for row in certified_ladder) == (12, 24, 48, 72, 96)
    for intervals, result, _current, spread in certified_ladder:
        assert result.converged, intervals
        assert result.residual <= MAX_RESIDUAL_PER_S, intervals
        assert (
            result.continuity_current_bound <= MAX_CONTINUITY_CURRENT_A_M2
        ), intervals
        assert spread <= MAX_CURRENT_SPREAD_A_M2, intervals


def test_certified_short_circuit_current_contracts(certified_ladder):
    coarse_rows = certified_ladder[:3]
    series = ConvergenceSeries(
        intervals=tuple(row[0] for row in coarse_rows),
        values=tuple(row[2] for row in coarse_rows),
    )
    assert series.contracts
    assert series.richardson_residual < 0.01 * abs(series.values[-1])


def test_prolonged_fine_grid_current_changes_contract(certified_ladder):
    """48/72/96 are non-geometric, so use their raw successive changes."""
    values = tuple(row[2] for row in certified_ladder[2:])
    coarse_change = abs(values[1] - values[0])
    fine_change = abs(values[2] - values[1])
    assert fine_change < coarse_change
    assert fine_change < 0.01 * abs(values[-1])


def test_observations_match_reproducibility_registry(certified_ladder):
    """Keep reported local values executable without widening physics gates."""
    matrix = yaml.safe_load(
        Path("reproducibility/config_benchmark_matrix.yaml").read_text(
            encoding="utf-8"
        )
    )
    contract = matrix["benchmarks"]["ionmonger-residual-ss-mesh"]
    tolerance = contract["regression_tolerance"]
    expected_rows = contract["observed_short_circuit"]

    assert [row[0] for row in certified_ladder] == [
        row["intervals"] for row in expected_rows
    ]
    for actual_row, expected in zip(certified_ladder, expected_rows):
        _intervals, result, current, spread = actual_row
        actual = {
            "residual_per_s": result.residual,
            "continuity_bound_A_m2": result.continuity_current_bound,
            "current_A_m2": current,
            "current_spread_A_m2": spread,
        }
        for metric, value in actual.items():
            assert value == pytest.approx(
                expected[metric], abs=float(tolerance[metric])
            ), (expected["intervals"], metric)

    values = tuple(row[2] for row in certified_ladder[2:])
    coarse_change = abs(values[1] - values[0])
    fine_change = abs(values[2] - values[1])
    fine_expected = contract["observed_fine_grid_change"]
    fine_actual = {
        "change_48_to_72_A_m2": coarse_change,
        "change_72_to_96_A_m2": fine_change,
        "contraction": fine_change / coarse_change,
        "finest_relative_change_percent": 100.0 * fine_change / abs(values[-1]),
    }
    fine_tolerance = {
        "change_48_to_72_A_m2": tolerance["fine_change_A_m2"],
        "change_72_to_96_A_m2": tolerance["fine_change_A_m2"],
        "contraction": tolerance["contraction"],
        "finest_relative_change_percent": tolerance[
            "finest_relative_change_percent"
        ],
    }
    for metric, value in fine_actual.items():
        assert value == pytest.approx(
            fine_expected[metric], abs=float(fine_tolerance[metric])
        ), metric
