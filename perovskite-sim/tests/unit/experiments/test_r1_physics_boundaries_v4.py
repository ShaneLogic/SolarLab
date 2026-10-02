"""Externally fixed boundary and formula examples for the R1 physical gates."""
from copy import deepcopy

import numpy as np
import pytest

from perovskite_sim.constants import Q
from perovskite_sim.experiments import one_dimensional_mechanism_r1_independent_physics as physics
from tests.unit.experiments.test_r1_independent_physics import fixture

LIMITS = {"internal_face_current_spread_relative": 2e-6,
          "contact_internal_current_spread_relative": 2e-6,
          "interface_current_spread_relative": 2e-6,
          "charge_balance_normalized": 1e-10, "inventory_relative_drift": 1e-10}


def test_units_and_phase_applicability_are_explicit_and_fixed():
    assert physics.METRIC_UNITS == {name: "1" for name in LIMITS}
    assert physics.METRIC_APPLICABILITY == {
        "internal_face_current_spread_relative": "finite_step",
        "contact_internal_current_spread_relative": "finite_step",
        "interface_current_spread_relative": "finite_step",
        "charge_balance_normalized": "finite_step",
        "inventory_relative_drift": "all_saved_rows"}
    checks = physics._metric_checks(dict.fromkeys(LIMITS, 3e-6), False)
    assert checks["inventory_relative_drift"] == {"applicable": True, "passed": False}
    for name in ("internal_face_current_spread_relative", "contact_internal_current_spread_relative",
                 "interface_current_spread_relative", "charge_balance_normalized"):
        assert checks[name] == {"applicable": False, "passed": None}


@pytest.mark.parametrize("name,limit", LIMITS.items())
@pytest.mark.parametrize("relation", ["below", "boundary", "above"])
def test_each_original_limit_inclusive_boundary(name, limit, relation):
    value = {"below": np.nextafter(limit, 0.), "boundary": limit,
             "above": np.nextafter(limit, np.inf)}[relation]
    assert physics.METRIC_LIMITS == LIMITS
    assert physics._metric_checks({name: value}, True)[name] == {
        "applicable": True, "passed": relation != "above"}


@pytest.mark.parametrize("name", LIMITS)
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_each_original_limit_rejects_nonfinite(name, value):
    assert physics._metric_checks({name: value}, True)[name]["passed"] is False


@pytest.mark.parametrize("scale", [1., 1e-20, 1e-26])
@pytest.mark.parametrize("ratio", [1.5e-6, 3e-6])
def test_reconstructed_face_range_and_floor(fixture, monkeypatch, scale, ratio):
    system, state, previous = fixture
    # Both significant-signal and below-floor cases. A std-based or inflated
    # floor statistic would pass the above-limit three-face example.
    magnitude = max(scale, 1e-20)
    current = np.array([scale, scale - ratio * magnitude, scale]) if scale > 1e-20 else np.array([0., ratio * magnitude, 0.])
    state.current_n = current
    monkeypatch.setattr(physics, "_currents", lambda *args: (current, np.zeros(3), np.zeros(3), np.empty((0, 2))))
    report = physics.independent_physics_row(system, state, previous, 1.)
    assert report["metrics"]["internal_face_current_spread_relative"] == pytest.approx(ratio, rel=5e-11)
    assert report["checks"]["internal_face_current_spread_relative"]["passed"] is (ratio < 2e-6)


@pytest.mark.parametrize("ratio", [0.5e-10, 1.5e-10])
def test_inventory_and_charge_reconstructed_limits(fixture, ratio):
    system, state, previous = fixture
    state.positive *= 1 + ratio
    system.positive_targets = np.array([3. * (1 + ratio)])  # cannot reset original inventory
    report = physics.independent_physics_row(system, state, None, 0.)
    assert report["metrics"]["inventory_relative_drift"] == pytest.approx(ratio, abs=3e-16)
    assert report["checks"]["inventory_relative_drift"]["passed"] is (ratio < 1e-10)
    # No conduction, original 1 A/m² normalization floor. A positive hole
    # increment creates exactly the specified charge-rate residual.
    previous.p[:] = 1. / Q
    state.p = previous.p.copy()
    state.coordinate[2:4] = np.log1p(ratio / 2.)
    report = physics.independent_physics_row(system, state, previous, 1.)
    assert report["arrays"]["charge_balance_scale_A_m2"] == 1.
    assert report["metrics"]["charge_balance_normalized"] == pytest.approx(ratio, rel=1e-14)
    assert report["checks"]["charge_balance_normalized"]["passed"] is (ratio < 1e-10)


@pytest.mark.parametrize("occupancy,passed", [(np.nextafter(.999, 0.), True), (.999, False), (1., False)])
def test_site_ceiling_is_strict(fixture, occupancy, passed):
    system, state, _ = fixture
    state.positive[:] = 2. * occupancy
    report = physics.independent_physics_row(system, state, None, 0.)
    assert report["checks"]["ion_site_occupancy_physical"]["passed"] is passed


def test_content_failure_does_not_claim_failed_conservation_or_reference_accuracy(fixture):
    system, state, previous = fixture
    state.current_n[:] = 1.
    report = physics.independent_physics_row(system, state, previous, 1.)
    assert report["assessment"] == {"content_consistent": False, "conservation_compliant": True,
        "reference_accuracy_qualified": None, "reference_accuracy_status": "not_assessed_shared_constitutive_laws"}


@pytest.mark.parametrize("factor", [1., 1.001])
def test_shared_bernoulli_scale_against_analytic_law(fixture, monkeypatch, factor):
    system, state, _ = fixture
    original = physics.bernoulli
    monkeypatch.setattr(physics, "bernoulli", lambda x: factor * original(x))
    gradient = 1e-3
    state.phi = gradient * system.grid
    state.dqfn, state.dqfp = -state.phi, state.phi
    current_n, current_p, _, _ = physics._currents(system, state)
    error = max(float(np.max(np.abs(current / (-Q * gradient) - 1.))) for current in (current_n, current_p))
    assert (error <= 5e-15) is (factor == 1.)


@pytest.mark.parametrize("ratio", [1.5e-6, 3e-6])
def test_physical_contact_boundary_is_not_inferred_from_internal_faces(fixture, monkeypatch, ratio):
    system, state, previous = fixture
    previous.positive[:] = 1e22
    state.positive = previous.positive.copy()
    system.common_dc_state.positive_ion_density_m3 = previous.positive.copy()
    system.material.P_lim_node[:] = 1e25
    state.coordinate[4] = np.log1p(ratio / (.5 * Q * 1e22))
    state.positive[0] *= np.exp(state.coordinate[4])
    state.current_n[:] = 1.
    monkeypatch.setattr(physics, "_currents", lambda *args: (np.ones(3), np.zeros(3), np.zeros(3), np.empty((0, 2))))
    report = physics.independent_physics_row(system, state, previous, 1.)
    assert report["checks"]["internal_face_current_spread_relative"]["passed"]
    assert report["metrics"]["contact_internal_current_spread_relative"] == pytest.approx(ratio, rel=5e-11)
    assert report["checks"]["contact_internal_current_spread_relative"]["passed"] is (ratio < 2e-6)


@pytest.mark.parametrize("ratio", [1.5e-6, 3e-6])
def test_two_sided_interface_boundary_is_separately_evaluated(fixture, monkeypatch, ratio):
    system, state, previous = fixture
    system.interface_count, system.interface_faces = 1, (1,)
    system.left_nodes, system.right_nodes = (1,), (2,)
    system.local_slice = slice(10, 16)
    system.material.eps_r = np.ones(4)
    system.material.iface_qss_left_distances_m = (.5,)
    system.material.iface_qss_right_distances_m = (.5,)
    state.coordinate, previous.coordinate = np.zeros(16), np.zeros(16)
    state.current_n[:] = 1.
    monkeypatch.setattr(physics, "_currents", lambda *args: (np.ones(3), np.zeros(3), np.zeros(3), np.array([[1., 1. - ratio]])))
    report = physics.independent_physics_row(system, state, previous, 1.)
    assert report["checks"]["contact_internal_current_spread_relative"]["passed"]
    assert report["metrics"]["interface_current_spread_relative"] == pytest.approx(ratio, rel=5e-11)
    assert report["checks"]["interface_current_spread_relative"]["passed"] is (ratio < 2e-6)
