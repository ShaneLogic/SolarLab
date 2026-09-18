"""Independent derivative-Gauss and regular-current analytic examples."""
from copy import deepcopy
from types import SimpleNamespace as Namespace

import numpy as np
import pytest

from perovskite_sim.constants import EPS_0, Q
from perovskite_sim.experiments import one_dimensional_mechanism_r1_independent_regular as regular
from perovskite_sim.experiments.one_dimensional_mechanism_r1_protocol import r1_policy
from tests.unit.experiments.test_r1_independent_physics import fixture


@pytest.fixture
def system_state(fixture):
    system, state, _ = fixture
    mat = system.material
    mat.eps_r = np.ones(4) / EPS_0
    for field in ("ni_sq", "tau_n", "tau_p", "n1", "p1"):
        setattr(mat, field, np.ones(4))
    for field in ("B_rad", "C_n", "C_p"):
        setattr(mat, field, np.zeros(4))
    system.system = Namespace(current_scale=1.)
    return system, state


def test_gauss_solution_has_analytic_parabolic_potential_and_half_contact_volume(system_state):
    system, _ = system_state
    widths, capacitance, sides = regular._geometry(system)
    phi, _, displacement, _, contacts = regular._gauss_derivative(
        widths, capacitance, np.full(4, 2.), np.empty(0), sides, system)
    np.testing.assert_allclose(phi, [0., 2., 2., 0.], atol=1e-15, rtol=0.)
    np.testing.assert_allclose(displacement, [-2., 0., 2.], atol=1e-15, rtol=0.)
    np.testing.assert_allclose(contacts, [-3., 3.], atol=1e-15, rtol=0.)


def test_sheet_derivative_has_distinct_two_sided_displacement(system_state):
    system, _ = system_state
    system.interface_count, system.interface_faces = 1, (1,)
    system.left_nodes, system.right_nodes = (1,), (2,)
    system.material.iface_qss_left_distances_m = (.5,)
    system.material.iface_qss_right_distances_m = (.5,)
    widths, capacitance, sides = regular._geometry(system)
    phi, traces, displacement, interfaces, contacts = regular._gauss_derivative(
        widths, capacitance, np.zeros(4), np.array([2.]), sides, system)
    np.testing.assert_allclose(phi, [0., 1., 1., 0.], atol=1e-15, rtol=0.)
    np.testing.assert_allclose(traces, [[1.5, 1.5]], atol=1e-15, rtol=0.)
    np.testing.assert_allclose(interfaces, [[-1., 1.]], atol=1e-15, rtol=0.)
    np.testing.assert_allclose(contacts, [-1., 1.], atol=1e-15, rtol=0.)
    assert displacement[1] == pytest.approx(-1.)


@pytest.mark.parametrize("relative", [True, False])
def test_regular_current_uses_neither_stored_currents_nor_rate_or_solver_helpers(system_state, relative):
    system, state = system_state
    state.phi = 1e-3 * system.grid
    state.dqfn, state.dqfp = -state.phi, state.phi
    state.current_n, state.current_p, state.rate = np.ones(3), np.ones(3), np.full(8, 1e99)
    def forbidden(*args, **kwargs):
        pytest.fail("independent current used a solver helper")
    for name in ("_increment_charge_density", "_sheet_weights", "interface_current_sides",
                 "solver_current_metrics", "transient_current_metrics"):
        setattr(system, name, forbidden)
    report = regular.independent_regular_current(system, state, policy=r1_policy(), require_relative_closure=relative)
    assert set(report) == regular.ROW_FIELDS
    assert set(report["arrays"]) == regular.ARRAY_FIELDS
    assert report["passed"]
    np.testing.assert_allclose(report["arrays"]["contact_maxwell_A_m2"], -2 * Q * 1e-3, atol=0., rtol=5e-15)
    assert report["checks"]["internal_face_current_spread_relative"] == {
        "applicable": relative, "passed": True if relative else None}


def test_nonzero_regular_content_mutation_is_rejected(system_state):
    system, state = system_state
    state.phi = 1e-3 * system.grid
    state.dqfn, state.dqfp = -state.phi, state.phi
    honest = regular.independent_regular_current(system, state, policy=r1_policy())
    published = deepcopy(honest["arrays"])
    published["contact_maxwell_A_m2"][0] *= 2.
    checked = regular.independent_regular_current(system, state, policy=r1_policy(), reported=published)
    assert not checked["passed"]
    assert checked["assessment"]["content_consistent"] is False
    assert checked["assessment"]["conservation_compliant"] is True
    assert checked["reasons"] == ["contact_maxwell_A_m2_reported_content_matches"]


def test_zero_excitation_keeps_absolute_failure_visible(system_state, monkeypatch):
    system, state = system_state
    monkeypatch.setattr(regular, "_rates", lambda *args: (
        np.array([0., 1e17, -1e17, 0.]), np.zeros(4), np.zeros(4), np.empty(0)))
    report = regular.independent_regular_current(system, state, policy=r1_policy(), require_relative_closure=False)
    assert report["checks"]["internal_face_current_spread_relative"] == {"applicable": False, "passed": None}
    assert not report["checks"]["electron_continuity_A_m2"]["passed"]
    assert not report["passed"]
