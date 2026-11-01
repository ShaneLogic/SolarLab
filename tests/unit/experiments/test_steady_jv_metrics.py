"""Manufactured curve tests; these do not establish solver/physical accuracy."""
import numpy as np
import pytest

from solarlab.experiments.jv.steady import steady_jv_metrics


@pytest.mark.parametrize('descending', [False, True])
def test_linear_curve_has_exact_piecewise_linear_maximum(descending):
    v = np.array([0.0, 0.2, 0.8, 1.2])
    j = 100.0 * (1.0 - v)
    if descending:
        v, j = v[::-1], j[::-1]
    metrics = steady_jv_metrics(v, j, incident_power_W_m2=1000.0)
    assert metrics.short_circuit_current_A_m2 == 100.0
    assert metrics.open_circuit_voltage_V == 1.0
    assert metrics.maximum_power_W_m2 == pytest.approx(25.0)
    assert metrics.fill_factor == pytest.approx(0.25)
    assert metrics.efficiency == pytest.approx(0.025)


def test_unbracketed_metrics_are_missing_not_zero_sentinels():
    metrics = steady_jv_metrics([0.0, 0.5], [100.0, 50.0])
    assert metrics.short_circuit_current_A_m2 == 100.0
    assert metrics.open_circuit_voltage_V is None
    assert metrics.fill_factor is None and metrics.efficiency is None
    assert metrics.limitation == 'open_circuit_not_bracketed'


def test_missing_zero_and_nonpositive_output_are_explicit():
    metrics = steady_jv_metrics([0.2, 1.0], [80.0, 0.0])
    assert metrics.short_circuit_current_A_m2 is None
    assert metrics.limitation == 'zero_voltage_not_bracketed'
    metrics = steady_jv_metrics([0.0, 1.0], [0.0, -100.0])
    assert metrics.open_circuit_voltage_V is None
    assert metrics.limitation == 'no_positive_short_circuit_output'


def test_incident_power_is_required_for_efficiency_only():
    metrics = steady_jv_metrics([0.0, 1.0], [100.0, 0.0])
    assert metrics.maximum_power_W_m2 == pytest.approx(25.0)
    assert metrics.efficiency is None


@pytest.mark.parametrize('v,j', [([0, 0], [1, 2]), ([0, 1], [1]), ([0, np.nan], [1, 2])])
def test_invalid_curves_rejected(v, j):
    with pytest.raises(ValueError):
        steady_jv_metrics(v, j)
