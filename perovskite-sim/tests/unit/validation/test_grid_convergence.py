import math

import numpy as np
import pytest

from perovskite_sim.experiments.jv_sweep import build_electrical_grid
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.solver.mol import StateVec, build_material_arrays
from perovskite_sim.validation.grid_convergence import (
    ConvergenceSeries,
    prolong_packed_state,
)


def test_second_order_series_reports_actual_order_and_residual():
    series = ConvergenceSeries(
        intervals=(12, 24, 48),
        values=(1.0 + 1 / 12**2, 1.0 + 1 / 24**2, 1.0 + 1 / 48**2),
    )
    assert series.contracts
    assert series.contraction == pytest.approx(0.25)
    assert series.observed_order == pytest.approx(2.0)
    assert series.richardson_residual == pytest.approx(1 / 48**2)


def test_noncontracting_series_has_no_finite_richardson_certificate():
    series = ConvergenceSeries((10, 20, 40), (1.0, 1.1, 1.3))
    assert not series.contracts
    assert math.isinf(series.richardson_residual)


@pytest.mark.parametrize(
    ("intervals", "values"),
    [
        ((12, 24, 47), (1.0, 2.0, 3.0)),
        ((12, 12, 24), (1.0, 2.0, 3.0)),
        ((12, 24, 48), (1.0, math.nan, 3.0)),
    ],
)
def test_invalid_series_is_rejected(intervals, values):
    with pytest.raises(ValueError):
        ConvergenceSeries(intervals, values)


def test_prolongation_preserves_nested_carriers_and_neutral_ion_background():
    stack = load_device_from_yaml("configs/ionmonger_benchmark.yaml")
    source_x = build_electrical_grid(stack, 48)
    target_x = build_electrical_grid(stack, 96)
    source_mat = build_material_arrays(source_x, stack)
    target_mat = build_material_arrays(target_x, stack)
    n = 1.0e20 * np.exp(source_x / source_x[-1])
    p = 2.0e19 * np.exp(-source_x / source_x[-1])
    source_state = StateVec.pack(n, p, source_mat.P_ion0.copy())

    target_state = prolong_packed_state(
        source_x, source_state, target_x, stack,
    )
    target = StateVec.unpack(target_state, len(target_x))

    np.testing.assert_allclose(target.n[::2], n, rtol=1.0e-14)
    np.testing.assert_allclose(target.p[::2], p, rtol=1.0e-14)
    np.testing.assert_array_equal(target.P, target_mat.P_ion0)


def test_prolongation_rejects_malformed_source_state():
    stack = load_device_from_yaml("configs/ionmonger_benchmark.yaml")
    source_x = build_electrical_grid(stack, 48)
    target_x = build_electrical_grid(stack, 96)

    with pytest.raises(ValueError, match="expected"):
        prolong_packed_state(
            source_x, np.ones(3 * len(source_x) - 1), target_x, stack,
        )
