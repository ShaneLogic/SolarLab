"""Original residuals and conservation still decide R1 direction recovery."""
from decimal import Decimal, localcontext
import math
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

from perovskite_sim.experiments.interface_defect_transient import (
    InterfaceDefectTransientError, InterfaceDefectTransientPolicy, _solve_step,
)
from perovskite_sim.experiments.one_dimensional_mechanism_r1_dynamics import ControlledPhysicalInterfaceIonSystem
from perovskite_sim.physics.two_sided_interface import (
    InterfaceTracePotentials, _bulk_flux_and_log_jacobian, _product_difference,
)
from tests.unit.physics.test_two_sided_interface import _bulk, _geometry, _physics


@pytest.mark.parametrize("fallback", [False, True])
def test_compensated_products_match_independent_decimal(fallback, monkeypatch):
    if fallback:
        monkeypatch.delattr(math, "fma", raising=False)
    for a, b, c, d in [(1.00001, 2e13, 1., 2.00002e13),
                       (1. + 2**-40, 1. - 2**-40, 1., 1.),
                       (2e150, 3e-150, 1.5e150, 4e-150)]:
        with localcontext() as context:
            context.prec = 100
            exact = float(Decimal.from_float(a) * Decimal.from_float(b)
                          - Decimal.from_float(c) * Decimal.from_float(d))
        assert _product_difference(a, b, c, d) == exact


@pytest.mark.parametrize("drop", [-.003407, .003407])
def test_paired_bernoulli_resolves_near_balanced_sg_flux(drop):
    geometry = _geometry(left_distance_m=1., right_distance_m=1.)
    physics = _physics(D_n_left_m2_s=1., transmission=0.)
    bulk = _bulk(phi_left_V=0., phi_right_V=0., n_left_m3=1e13)
    traces = InterfaceTracePotentials(drop * physics.thermal_voltage_V, 0.)
    xi = traces.phi_left_V / physics.thermal_voltage_V
    state = np.asarray([bulk.n_left_m3 * math.exp(xi) * (1. + 3e-9), 1e13, 1e13, 1e13])
    with localcontext() as context:
        context.prec = 80
        x = Decimal.from_float(xi)
        b = x / (x.exp() - 1)
        oracle = float((b + x) * Decimal.from_float(bulk.n_left_m3)
                       - b * Decimal.from_float(float(state[0])))
    legacy = _bulk_flux_and_log_jacobian(state, traces, geometry, physics, bulk)
    paired = _bulk_flux_and_log_jacobian(state, traces, geometry, physics, bulk,
                                        paired_bernoulli=True)
    assert abs(paired[0][0] - oracle) < 1e-5
    assert abs(paired[0][0] - oracle) < .05 * abs(legacy[0][0] - oracle)
    for old, new in zip(legacy[1:], paired[1:]):
        np.testing.assert_array_equal(old, new)


class OneCellGauss:
    """One-cell absolute Gauss error and its finite-step displacement current."""
    interface_count = 0
    newton_residual_target = ControlledPhysicalInterfaceIonSystem.newton_residual_target

    def storage_scale(self, *args):
        return np.empty(0)

    def poisson_scale(self, policy):
        return np.ones(1)

    def local_algebraic_scale(self, policy):
        return np.empty(0)

    def residual_and_jacobian(self, coordinate, voltage, previous, dt, *scales):
        state = SimpleNamespace(coordinate=np.asarray(coordinate), error=float(coordinate[0]))
        return np.asarray([state.error]), sparse.eye(1, format="csr"), state

    def charge_balance_metrics(self, state, previous, dt):
        return 0., 0.

    def solver_current_metrics(self, state, previous, dt):
        displacement_imbalance = abs(state.error - previous.poisson_residual[0]) / dt
        return None, None, None, None, displacement_imbalance, 0.


def previous(error):
    return SimpleNamespace(storage=np.empty(0), poisson_residual=np.asarray([error]),
                           local_residual=np.empty(0))


def test_direction_retains_original_absolute_residual_and_closes_current():
    policy = InterfaceDefectTransientPolicy()
    system, old, dt = OneCellGauss(), previous(1e-8), 1e-4
    result = _solve_step(system, np.zeros(1), old, 0., dt, policy, check_jacobian=False)
    state = result[0]
    assert state.error == 1e-8
    assert result[2] == 1e-8  # the actual absolute residual is not reset to zero
    assert system.solver_current_metrics(state, old, dt)[4] == 0.
    assert result[2] <= policy.maximum_scaled_nonlinear_residual


def test_incompatible_gauss_and_current_constraints_still_fail():
    policy = InterfaceDefectTransientPolicy()
    # Retaining this error would violate the original absolute-residual limit;
    # removing it would violate continuity. No permitted state satisfies both.
    with pytest.raises(InterfaceDefectTransientError, match="line search stalled"):
        _solve_step(OneCellGauss(), np.zeros(1), previous(1.), 0., 1e-4,
                    policy, check_jacobian=False)
