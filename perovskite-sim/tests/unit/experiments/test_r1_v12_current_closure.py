"""Independent algebraic and acceptance contracts for the pair direction target.

Small algebraic states below intentionally contain only the fields consumed by
the direction helper. They are not production snapshots or trajectory proofs.
The one-cell Gauss model checks the physical residual/current compatibility;
the saved device step is covered by the separate source-bound causal study.
"""
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

from perovskite_sim.constants import EPS_0
from perovskite_sim.experiments.interface_defect_transient import (
    InterfaceDefectTransientError, InterfaceDefectTransientPolicy, _solve_step,
)
from perovskite_sim.experiments.one_dimensional_mechanism_r1_dynamics import (
    ControlledPhysicalInterfaceIonSystem,
)
from perovskite_sim.experiments.one_dimensional_mechanism_r1_precision import (
    CompensatedR1System, PrecisionState,
)
from perovskite_sim.physics.compensated import DD


def algebraic_state(poisson, *, trace=None, phi=None, sigma=None):
    state = object.__new__(PrecisionState)
    state.fine = {
        "poisson_residual_C_m2": poisson,
        "trace_potential_V": DD([[0., 0.]]) if trace is None else trace,
        "phi_V": DD([0., 0.]) if phi is None else phi,
        "sheet_charge_C_m2": DD([0.]) if sigma is None else sigma,
    }
    state.poisson_residual = poisson.hi.copy()
    state.local_residual = np.full(6, 999.)  # must never replace the fine local fields
    state.storage = np.empty(0)
    state.coordinate = np.zeros(poisson.size)
    return state


def interface_system():
    system = object.__new__(CompensatedR1System)
    system.interface_count = 1
    system.left_nodes, system.right_nodes = (0,), (1,)
    # Both face capacitances are exactly one for this independent algebraic case.
    system.material = SimpleNamespace(eps_r=np.ones(2),
        iface_qss_left_distances_m=np.array([EPS_0]),
        iface_qss_right_distances_m=np.array([EPS_0]))
    system._prescribed_trace_jump = DD([0.])
    return system


def direction_case():
    tiny = 2.**-70
    previous = algebraic_state(DD([2.**-10], [tiny]),
        trace=DD([[2.**-8, 2.**-7]], [[tiny, 0.]]), sigma=DD([3*2.**-8]))
    state = algebraic_state(DD([2.**-10], [3*tiny]),
        trace=DD([[2.**-8, 2.**-7]], [[3*tiny, 4*tiny]]), sigma=DD([3*2.**-8]))
    scales = (np.ones(2), np.array([2.**-4]), np.array([1., 2., 4., 8., 16., 32.]))
    # The third and fourth entries are deliberately the raw rounded physical
    # residuals. Their high-word subtraction is exactly zero in this case.
    residual = np.array([.017, -.013, 2.**-6, 2.**-8, 7*tiny/2, .02, -.03, .04, -.01])
    return interface_system(), previous, state, residual, scales


def fine_bytes(state):
    return {name: (value.hi.tobytes(), value.lo.tobytes(), value.shape)
            for name, value in state.fine.items()}


def test_previous_target_retains_only_electrostatic_equations_and_local_pair_words():
    system, previous, _, _, scales = direction_case()
    target = system.newton_residual_target(previous, *scales)
    assert target.shape == (9,)
    np.testing.assert_array_equal(target[[0, 1, 5, 6, 7, 8]], 0.)
    assert target[2] == 2.**-6
    assert target[3] == 2.**-8
    assert target[4] == 2.**-71
    assert np.max(np.abs(target)) < .05
    # A future implementation that reads previous.local_residual instead of
    # reconstructing the physical local pair would inherit the poison 999.
    assert not np.any(target == 999.)


def test_direction_retains_poisson_and_trace_low_word_differences_against_exact_rational():
    system, previous, state, residual, scales = direction_case()
    target = system.newton_residual_target(previous, *scales)
    rhs = system.newton_direction_rhs(state, previous, residual, target, *scales)
    tiny = Fraction(1, 2**70)
    expected = [float(2*tiny/Fraction(1, 16)), float(2*tiny), float(6*tiny/2)]
    np.testing.assert_array_equal(rhs[2:5], expected)
    assert (residual-target)[2] == (residual-target)[3] == 0.
    assert rhs[2] != 0. and rhs[3] != 0.
    np.testing.assert_array_equal(rhs[[0, 1, 5, 6, 7, 8]], residual[[0, 1, 5, 6, 7, 8]])


def test_direction_never_mutates_raw_acceptance_residual_or_physical_state():
    system, previous, state, residual, scales = direction_case()
    before, current, raw = fine_bytes(previous), fine_bytes(state), residual.tobytes()
    target = system.newton_residual_target(previous, *scales)
    saved_target = target.tobytes()
    rhs = system.newton_direction_rhs(state, previous, residual, target, *scales)
    assert fine_bytes(previous) == before and fine_bytes(state) == current
    assert residual.tobytes() == raw and target.tobytes() == saved_target
    assert not np.shares_memory(rhs, residual)
    assert set(previous.fine) == set(before) and set(state.fine) == set(current)


def test_target_uses_previous_state_not_current_work_or_independent_poisson():
    system, previous, state, residual, scales = direction_case()
    before = system.newton_residual_target(previous, *scales)
    def forbidden(*args, **kwargs):
        raise AssertionError("direction target may not request or replace an eliminated potential")
    system.independent_poisson_inputs = forbidden
    system.eliminated_operator_diagnostics = forbidden
    system._fine_work = state.fine
    state.fine["phi_V"] = DD([5., -3.])
    state.fine["trace_potential_V"] = DD([[8., -2.]])
    after = system.newton_residual_target(previous, *scales)
    np.testing.assert_array_equal(after, before)
    system.newton_direction_rhs(state, previous, residual, after, *scales)


@pytest.mark.parametrize("change", ["zero", "current_residual", "wrong_component", "shape", "nan"])
def test_direction_rejects_a_target_that_does_not_belong_to_previous_state(change):
    system, previous, state, residual, scales = direction_case()
    target = system.newton_residual_target(previous, *scales)
    if change == "zero":
        target[:] = 0.
    elif change == "current_residual":
        target = residual.copy()
    elif change == "wrong_component":
        target[4] = np.nextafter(target[4], np.inf)
    elif change == "shape":
        target = target[:-1]
    else:
        target[2] = np.nan
    with pytest.raises(ValueError, match="previous accepted state"):
        system.newton_direction_rhs(state, previous, residual, target, *scales)


def test_previous_low_word_change_invalidates_its_old_target_without_changing_high_word():
    system, previous, state, residual, scales = direction_case()
    target = system.newton_residual_target(previous, *scales)
    previous.fine["trace_potential_V"] = DD([[2.**-8, 2.**-7]], [[2.**-69, 0.]])
    with pytest.raises(ValueError, match="previous accepted state"):
        system.newton_direction_rhs(state, previous, residual, target, *scales)


def test_exactly_zero_previous_residual_retains_original_direction_bitwise():
    system, _, state, residual, scales = direction_case()
    previous = algebraic_state(DD([0.]))
    target = system.newton_residual_target(previous, *scales)
    expected = residual-target
    rhs = system.newton_direction_rhs(state, previous, residual, target, *scales)
    assert rhs.dtype == expected.dtype and rhs.tobytes() == expected.tobytes()


def test_nonprecision_state_preserves_legacy_target_and_direction_semantics():
    system = interface_system()
    previous = SimpleNamespace(poisson_residual=np.array([1e-8]),
        local_residual=np.array([1e-9, 2e-9, 3., 4., 5., 6.]))
    scales = (np.ones(2), np.ones(1), np.ones(6))
    target = system.newton_residual_target(previous, *scales)
    expected = ControlledPhysicalInterfaceIonSystem.newton_residual_target(system, previous, *scales)
    np.testing.assert_array_equal(target, expected)
    residual = np.arange(9, dtype=float)
    np.testing.assert_array_equal(system.newton_direction_rhs(SimpleNamespace(), previous,
        residual, target, *scales), residual-target)


class PairOneCellGauss(CompensatedR1System):
    """q-div(D) and the associated imbalance from discharging an allowed error."""
    def __init__(self):
        self.interface_count = 0
        self.left_nodes = self.right_nodes = ()

    def storage_scale(self, *args):
        return np.empty(0)

    def poisson_scale(self, policy):
        return np.ones(1)

    def local_algebraic_scale(self, policy):
        return np.empty(0)

    def residual_and_jacobian(self, coordinate, voltage, previous, dt, *scales):
        state = algebraic_state(DD(np.asarray(coordinate)))
        state.coordinate = np.asarray(coordinate).copy()
        return state.poisson_residual.copy(), sparse.eye(1, format="csr"), state

    def charge_balance_metrics(self, state, previous, dt):
        return 0., 0.

    def solver_current_metrics(self, state, previous, dt):
        difference = (state.fine["poisson_residual_C_m2"]-previous.fine["poisson_residual_C_m2"])/dt
        return None, None, None, None, float(np.max(np.abs(difference.to_float()))), 0.


def test_pair_direction_closes_current_without_zeroing_actual_accepted_residual():
    system, previous = PairOneCellGauss(), algebraic_state(DD([1e-8]))
    result = _solve_step(system, np.zeros(1), previous, 0., 1e-4,
        InterfaceDefectTransientPolicy(), check_jacobian=False)
    assert result[2] == 1e-8
    assert result[0].poisson_residual[0] == 1e-8
    assert system.solver_current_metrics(result[0], previous, 1e-4)[4] == 0.


def test_incompatible_previous_gauss_error_is_not_clipped_or_accepted():
    system, previous = PairOneCellGauss(), algebraic_state(DD([1.]))
    assert system.newton_residual_target(previous, np.empty(0), np.ones(1), np.empty(0))[0] == 1.
    with pytest.raises(InterfaceDefectTransientError, match="line search stalled"):
        _solve_step(system, np.zeros(1), previous, 0., 1e-4,
            InterfaceDefectTransientPolicy(), check_jacobian=False)
