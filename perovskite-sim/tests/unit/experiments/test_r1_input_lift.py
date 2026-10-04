"""Synthetic four-node input/dataflow tests; no DC or nonlinear step is run."""
from dataclasses import fields, replace
from decimal import Decimal, localcontext
import hashlib
import json
from types import SimpleNamespace
from types import MappingProxyType

import numpy as np
import pytest
from scipy import sparse

from perovskite_sim.constants import EPS_0, Q
from perovskite_sim.experiments import interface_defect_transient as base
from perovskite_sim.experiments.interface_defect_ion_transient import _InterfaceIonDeviceState
from perovskite_sim.experiments.interface_defect_ion_transient import _InterfaceIonTransientSystem
from perovskite_sim.experiments import one_dimensional_mechanism_r1_input_lift as lift
from perovskite_sim.experiments import one_dimensional_mechanism_r1_local_carrier as local
from perovskite_sim.experiments.one_dimensional_mechanism_r1_dynamics import (
    ControlledPhysicalInterfaceIonSystem, R1DynamicsControls,
)
from perovskite_sim.physics.compensated import DD
from perovskite_sim.physics.two_sided_interface import (
    TwoSidedBulkState, TwoSidedInterfaceGeometry, TwoSidedInterfacePhysics,
)


def decimal(value):
    return Decimal.from_float(float(value))


def represented(value):
    return decimal(value.hi)+decimal(value.lo)


@pytest.fixture
def synthetic(monkeypatch):
    system = object.__new__(ControlledPhysicalInterfaceIonSystem)
    system.node_count, system.interior_count, system.interface_count, system.dimension = 4, 2, 1, 15
    system.electron_slice, system.hole_slice, system.trap_slice = slice(0, 2), slice(2, 4), slice(4, 5)
    system.positive_slice, system.negative_slice = slice(5, 7), slice(7, 7)
    system.potential_slice, system.local_slice = slice(7, 9), slice(9, 15)
    system.left_nodes, system.right_nodes, system.interface_faces = (1,), (2,), (1,)
    system.positive_nodes, system.negative_nodes = np.array([1, 2]), np.empty(0, dtype=int)
    system.grid, system.widths = np.arange(4.), np.array([.5, 1., 1., .5])
    system.thermal_voltage, system.polarity, system.capture_multiplier = .25, 1., 1.
    system._controls = R1DynamicsControls(1, 1)
    system.trap_density, system.equilibrium_occupancy = np.array([3.]), np.array([.3])
    system.reference_local_scale = np.ones((1, 6))
    system.reference_n = np.array([4., 4., 8., 8.])
    system.reference_p = np.array([6., 6., 10., 10.])
    system.reference_positive = np.ones(4)
    system.site_occupancy_ceiling = .999
    system.stack = object()
    system.dark_reference = SimpleNamespace(interface_transmission=0.)
    source = SimpleNamespace(carrier_params={}, ni_sq=np.zeros(4), tau_n=np.full(4, np.inf),
        tau_p=np.full(4, np.inf), n1=np.ones(4), p1=np.ones(4), B_rad=np.zeros(4),
        C_n=np.zeros(4), C_p=np.zeros(4), neutral_bulk_defects=None)
    system.material = SimpleNamespace(has_dual_ions=False, ion_steric_diffusion_only=True,
        ion_steric_shared_site=False, P_lim_node=np.full(4, 100.), D_ion_face=np.ones(3),
        V_T_device=.25, chi=np.zeros(4), Eg=np.zeros(4), D_n_face=np.ones(3), D_p_face=np.ones(3),
        eps_r=np.ones(4), iface_qss_left_distances_m=np.array([EPS_0]),
        iface_qss_right_distances_m=np.array([EPS_0]),
        iface_qss_interface_positions_m=np.array([1.5]), P_ion0=np.ones(4),
        N_D=np.zeros(4), N_A=np.zeros(4),
        poisson_factor=SimpleNamespace(C=np.ones(3), h_cell=np.ones(2)))
    system.system = SimpleNamespace(source_mat=source, reference_edge_drop_n=np.zeros(3),
                                    reference_edge_drop_p=np.zeros(3))
    system.common_dc_state = SimpleNamespace(positive_ion_density_m3=np.ones(4))
    system.ion_layout = SimpleNamespace(positive_components=(np.array([1, 2]),))
    system.eps_face = np.full(3, EPS_0)
    zero = sparse.csr_matrix((15, 15))
    values = {f.name: None for f in fields(_InterfaceIonDeviceState)}
    values.update(coordinate=np.zeros(15), dqfn=np.full(4, .125), dqfp=np.full(4, -.125),
        n=system.reference_n.copy(), p=system.reference_p.copy(), occupancy=np.array([.25]),
        positive=np.ones(4), negative=None, phi=np.full(4, .5),
        storage=np.r_[system.reference_n[1:-1], system.reference_p[1:-1], .75, 1., 1.],
        rate=np.zeros(7), positive_rate=np.zeros(4), current_n=np.zeros(3), current_p=np.zeros(3),
        positive_flux=np.zeros(3), carrier_conduction=np.zeros(3), positive_current=np.zeros(3),
        conduction=np.zeros(3), sheet_charge=np.array([Q*.15]), poisson_residual=np.zeros(2),
        local_residual=np.zeros(6), storage_jacobian=zero[:7], rate_jacobian=zero[:7],
        poisson_jacobian=zero[:2], local_jacobian=zero[:6], direct_poisson_residual=np.zeros(2))
    values["local"] = (base._LocalState(trace_potential=np.array([.5, .5]),
        log_state=np.log([4., 6., 8., 10.]), state_m3=np.array([4., 6., 8., 10.]),
        quasi_steady_occupancy=None, sheet_charge_C_m2=Q*.15, electrostatic_residual=np.zeros(2),
        tangent=SimpleNamespace(balance=SimpleNamespace(capture_flux_m2_s=np.zeros(4)))),)
    previous = _InterfaceIonDeviceState(**values)
    system._step_reference = previous
    system._lift, system._trace_lift = np.zeros(4), np.zeros((1, 2))

    # This fixture supplies only contact/high-word views. The real new map and
    # every constitutive DD call below are production functions under test.
    def high_views(self, coordinate, voltage):
        if np.shape(coordinate) != (15,) or not np.isfinite(coordinate).all():
            raise base.InterfaceDefectTransientError("invalid coordinate")
        return (previous.dqfn.copy(), previous.dqfp.copy(), previous.phi.copy(),
                previous.n.copy(), previous.p.copy(), previous.occupancy.copy(),
                np.array([[.5, .5]]), np.log([[4., 6., 8., 10.]]))
    monkeypatch.setattr(ControlledPhysicalInterfaceIonSystem, "_coordinates", high_views)
    monkeypatch.setattr(base._InterfaceTransientSystem, "_source", lambda *a, **k: np.zeros(8))
    table = {"eta": [-40., 20.], "log_half": [-40., 20.]}
    digest = hashlib.sha256(json.dumps(table, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    monkeypatch.setattr(local, "default_fd_table", lambda: (table, digest))
    geometry = TwoSidedInterfaceGeometry(left_distance_m=EPS_0, right_distance_m=EPS_0,
                                        eps_r_left=1., eps_r_right=1.)
    physics = TwoSidedInterfacePhysics(thermal_voltage_V=.25, temperature_K=1.,
        D_n_left_m2_s=1., D_p_left_m2_s=1., D_n_right_m2_s=1., D_p_right_m2_s=1.,
        N_C_left_m3=1024., N_V_left_m3=1024., N_C_right_m3=1024., N_V_right_m3=1024.,
        richardson_n_A_m2_K2=Q, richardson_p_A_m2_K2=Q, transmission=0.,
        surface_recombination_velocity_n_m_s=2., surface_recombination_velocity_p_m_s=3.,
        n1_left_m3=1., p1_left_m3=2., n1_right_m3=3., p1_right_m3=4.)
    def problem(material, stack, n, p, phi, index, **kwargs):
        return geometry, physics, TwoSidedBulkState(phi[1], phi[2], n[1], p[1], n[2], p[2])
    monkeypatch.setattr(local, "_material_two_sided_interface_problem", problem)
    working, saved = lift.from_saved_step(system, previous)

    def evaluate(z, *, active=None, prior=None):
        """Exercise the actual operator hooks on a bounded synthetic layout."""
        operator = working if active is None else active
        reference = previous if prior is None else prior
        qn, qp, phi, n, p, occupancy, trace, logs = operator._coordinates(z, 0.)
        positive, _ = operator._ion_coordinates(z)
        states, aggregate = operator._local_states(n, p, phi, occupancy, trace, logs,
            trace_density_m3=operator._trace_density_coordinates(z))
        source = operator._source(n, p, phi, 0., aggregate)
        tn, tp, jn, jp = operator._currents(qn, qp, phi, n, p, states)
        operator._carrier_rate_fields(source, tn, tp, states)
        irate, _, iflux, _ = operator._ion_fields(phi, positive, None)
        skeleton = replace(reference, coordinate=z.copy(), dqfn=qn, dqfp=qp, phi=phi, n=n, p=p,
            occupancy=occupancy, local=states, positive=positive, positive_rate=irate,
            positive_flux=iflux, current_n=jn, current_p=jp, positive_current=Q*iflux,
            carrier_conduction=jn+jp, conduction=jn+jp+Q*iflux)
        result = operator._with_step_electrostatics(skeleton)
        operator._input_lift_work = None
        return result, source, aggregate
    return system, previous, working, saved, evaluate


def test_adapter_retains_saved_inputs_without_evaluation_or_aliasing(synthetic, monkeypatch):
    system, original, working, saved, _ = synthetic
    monkeypatch.setattr(ControlledPhysicalInterfaceIonSystem, "evaluate", lambda *a: pytest.fail("constructor evaluated physics"))
    again, state = lift.from_saved_step(system, original)
    assert again is not system and type(again) is lift.RebasedInputLiftR1System
    assert again._step_reference is state and system._step_reference is original
    assert state.operator_representation == lift.REPRESENTATION != "float64-pair-v1"
    assert not hasattr(state, "fine")
    for name in lift.PRIMARY_FIELDS:
        np.testing.assert_array_equal(state.input_lift[name].lo, 0.)
    for name in ("n", "p", "phi", "dqfn", "dqfp", "positive", "occupancy", "storage", "sheet_charge", "local_residual"):
        np.testing.assert_array_equal(getattr(state, name), getattr(original, name))
        assert not np.shares_memory(getattr(state, name), getattr(original, name))
    with pytest.raises(TypeError):
        state.input_lift["n_m3"] = DD(0.)
    with pytest.raises(ValueError):
        state.input_lift["n_m3"].lo[0] = 1.
    with pytest.raises(TypeError):
        lift.from_saved_step(working, saved)


def test_zero_coordinate_has_exact_zero_increments(synthetic):
    _, _, working, saved, evaluate = synthetic
    state, _, _ = evaluate(np.zeros(15))
    for name in lift.PRIMARY_FIELDS:
        np.testing.assert_array_equal(state.input_lift[name].hi, saved.input_lift[name].hi)
        np.testing.assert_array_equal(state.input_lift[name].lo, 0.)
    np.testing.assert_array_equal(working.storage_increment(state, saved), 0.)
    np.testing.assert_array_equal(working.potential_increment(state, saved), 0.)
    np.testing.assert_array_equal((state.input_lift["sheet_charge_C_m2"]-saved.input_lift["sheet_charge_C_m2"]).hi, 0.)
    assert working.integrated_charge_increment(state, saved) == 0.


def test_sub_ulp_coordinate_lift_matches_decimal_and_reaches_all_four_balances(synthetic):
    _, _, working, saved, evaluate = synthetic
    origin, _, _ = evaluate(np.zeros(15))
    z = np.zeros(15)
    z[[0, 2, 7, 9, 10, 11, 12, 13, 14]] = np.array([1, -1, 2, 1, -1, 2, 3, 4, 5])*2.**-62
    state, source, aggregate = evaluate(z)
    with localcontext() as context:
        context.prec = 90
        expected_n = decimal(saved.n[1])*(decimal(z[0])+decimal(z[7])).exp()
        expected_phi = decimal(saved.phi[1])+decimal(working.thermal_voltage)*decimal(z[7])
        assert abs(represented(state.input_lift["n_m3"][1])-expected_n) < Decimal("1e-30")
        assert abs(represented(state.input_lift["phi_V"][1])-expected_phi) < Decimal("1e-32")
    np.testing.assert_array_equal(state.n, saved.n)
    np.testing.assert_array_equal(state.phi, saved.phi)
    assert state.input_lift["n_m3"].lo[1] != 0.
    assert state.input_lift["phi_V"].lo[1] != 0.
    payload = state.local[0].carrier_data
    assert aggregate.carrier_data[0] is payload
    changes = payload.balance["residual_m2_s"]-origin.local[0].carrier_data.balance["residual_m2_s"]
    assert np.all(np.abs(changes.hi) > 0.)
    np.testing.assert_array_equal(payload.inputs.state_density.lo, state.input_lift["trace_state_m3"].lo[0])
    np.testing.assert_array_equal(payload.inputs.bulk_potential.lo, state.input_lift["phi_V"].lo[[1, 2]])
    # Each receiving-volume source is the loss corresponding to its same-call
    # flux; no zero-low reconstruction from rounded n/p is allowed here.
    flux = payload.balance["bulk_flux_m2_s"]
    for component, row in enumerate((1, 5, 2, 6)):
        assert represented(source[row]+flux[component]/DD(working.widths[row % 4])) == 0


def test_local_analytic_chain_matches_tiny_dd_secants(synthetic):
    _, _, working, _, evaluate = synthetic
    center, _, _ = evaluate(np.zeros(15))
    _, _, jacobian = working._local_carrier_jacobians(0, 1, 2, center.local[0], 0.)
    h = 2.**-60
    for column in (0, 1, 2, 3, 7, 8, 9, 10, 11, 12, 13, 14):
        z = np.zeros(15)
        z[column] = h
        plus, _, _ = evaluate(z)
        minus, _, _ = evaluate(-z)
        difference = (plus.local[0].carrier_data.balance["residual_m2_s"]
                      -minus.local[0].carrier_data.balance["residual_m2_s"])/DD(2*h)
        np.testing.assert_allclose(difference.hi, jacobian[:, column].toarray().ravel(), rtol=2e-13, atol=1e-28)


def test_state_lifetime_independent_currents_and_charge_use_primary_words(synthetic):
    _, _, working, saved, evaluate = synthetic
    z = np.zeros(15)
    z[7], z[11:15], z[5:7] = 2.**-62, 3*2.**-62, [2.**-60, -2.**-60]
    state, _, _ = evaluate(z)
    before = lift._words(state.input_lift)
    expected = lift.independent_currents(working, state)
    np.testing.assert_array_equal(expected[0], state.current_n)
    np.testing.assert_array_equal(expected[1], state.current_p)
    np.testing.assert_array_equal(expected[2], state.positive_flux)
    evaluate(np.ones(15)*1e-4)
    assert lift._words(state.input_lift) == before
    poisoned = replace(state, local=(), current_n=np.full(3, 999.), current_p=np.full(3, 999.))
    for actual, correct in zip(lift.independent_currents(working, poisoned), expected):
        np.testing.assert_array_equal(actual, correct)
    row = lift.independent_physics_row(working, state, saved, 1.)
    assert row["checks"]["electron_current_A_m2_state_content_matches"]["passed"]
    assert row["checks"]["hole_current_A_m2_state_content_matches"]["passed"]
    assert row["checks"]["positive_ion_flux_m2_s_state_content_matches"]["passed"]
    assert row["operator_representation"] == lift.REPRESENTATION
    np.testing.assert_array_equal(row["arrays"]["internal_displacement_A_m2"], working.transient_current_metrics(state, saved, 1.)[0])
    assert row["arrays"]["charge_rate_A_m2"] == working.integrated_charge_increment(state, saved)


def test_low_input_recombination_correction_matches_decimal(synthetic):
    _, _, working, _, _ = synthetic
    mat = working.system.source_mat
    mat.tau_n, mat.tau_p = np.full(4, 2.), np.full(4, 3.)
    mat.B_rad, mat.C_n, mat.C_p = .125, .25, .5
    n, p = DD(np.full(4, 4.), np.full(4, 2.**-60)), DD(np.full(4, 6.), np.full(4, -2.**-61))
    correction = lift._recombination(working, DD(n.hi), DD(p.hi))-lift._recombination(working, n, p)
    with localcontext() as context:
        context.prec = 90
        def oracle(a, b):
            return a*b/(3*(a+1)+2*(b+1))+(Decimal(".125")+Decimal(".25")*a+Decimal(".5")*b)*a*b
        expected = oracle(Decimal(4), Decimal(6))-oracle(represented(n[0]), represented(p[0]))
        assert abs(represented(correction[0])-expected) < Decimal("1e-29")
    assert np.all(correction.hi != 0.)


def test_original_solver_and_acceptance_limits_are_not_overridden():
    assert base.InterfaceDefectTransientPolicy().maximum_scaled_nonlinear_residual == .05
    assert "solve_step" not in lift.RebasedInputLiftR1System.__dict__
    assert "newton_direction_rhs" not in lift.RebasedInputLiftR1System.__dict__
    assert lift.RebasedInputLiftR1System.newton_residual_target is ControlledPhysicalInterfaceIonSystem.newton_residual_target


def test_eliminated_ion_comparison_uses_its_own_supplied_inputs(synthetic, monkeypatch):
    _, _, working, _, _ = synthetic
    phi, positive = np.arange(4.)/10, np.array([1., 2., 3., 1.])
    expected = (np.arange(4.), None, np.arange(3.), None)
    observed = []
    def independent(self, actual_phi, actual_positive, actual_negative):
        observed.append((actual_phi, actual_positive, actual_negative))
        return expected
    monkeypatch.setattr(ControlledPhysicalInterfaceIonSystem, "_ion_fields", independent)
    assert working._input_lift_work is None
    assert working._ion_fields(phi, positive, None) is expected
    assert observed[0][0] is phi and observed[0][1] is positive and observed[0][2] is None
    assert working._input_lift_work is None


def test_actual_inherited_evaluate_wiring_keeps_residual_inputs_and_clears_work(synthetic, monkeypatch):
    _, _, working, previous, _ = synthetic
    working.system._bulk_space_charge_and_tangent = lambda *a, **k: (np.zeros(4), np.zeros(4))
    working._poisson_laplacian = sparse.csr_matrix((2, 4))
    monkeypatch.setattr(base._InterfaceTransientSystem, "_jacobians", lambda *a:
        tuple(sparse.csr_matrix((rows, 15)) for rows in (5, 5, 2, 6)))
    monkeypatch.setattr(_InterfaceIonTransientSystem, "_ion_jacobians", lambda *a:
        tuple(sparse.csr_matrix((4, 15)) for _ in range(4)))
    z = np.zeros(15)
    z[7], z[11] = 2.**-62, 2.**-61
    residual, jacobian, state = working.residual_and_jacobian(z, 0., previous, 1.,
        np.ones(7), np.ones(2), np.ones(6))
    assert residual.shape == (15,) and jacobian.shape == (15, 15)
    np.testing.assert_array_equal(residual[:7],
        (state.input_lift["storage"]-previous.input_lift["storage"]-state.input_lift["rate"]).hi)
    np.testing.assert_array_equal(residual[-4:], state.local[0].carrier_data.balance["residual_m2_s"].hi)
    assert working._input_lift_work is None
    with pytest.raises(base.InterfaceDefectTransientError):
        working.evaluate(np.full(15, np.nan), 0.)
    assert working._input_lift_work is None


def test_local_input_guard_rejects_mixed_high_views(synthetic):
    _, _, working, _, _ = synthetic
    qn, qp, phi, n, p, occupancy, trace, logs = working._coordinates(np.zeros(15), 0.)
    n[1] += 1.
    with pytest.raises(ValueError, match="disagree with current high words"):
        working._local_carrier_inputs(0, n, p, phi, occupancy, trace, logs,
            trace_density_m3=working._trace_density_coordinates(np.zeros(15)))
    working._input_lift_work = None


@pytest.mark.parametrize("setting", [{"carrier_statistics": "fermi_dirac"},
                                     {"degenerate_recombination_model": "unimplemented"}])
def test_adapter_rejects_unsupported_bulk_statistics(synthetic, setting):
    original, previous, _, _, _ = synthetic
    original.system.source_mat.carrier_params = setting
    with pytest.raises(ValueError, match="Maxwell-Boltzmann"):
        lift.from_saved_step(original, previous)


def test_adapter_rejects_unhandled_density_dependent_generation(synthetic):
    original, previous, _, _, _ = synthetic
    original.system.source_mat.has_radiative_reabsorption = True
    with pytest.raises(ValueError, match="radiative reabsorption"):
        lift.from_saved_step(original, previous)


def test_accepted_rebase_keeps_every_word_identity_and_fixed_scales(synthetic, monkeypatch):
    _, _, working, _, evaluate = synthetic
    z = np.arange(15)*2.**-63
    accepted, _, _ = evaluate(z)
    words, old_identity = lift._words(accepted.input_lift), accepted.coordinate_reference_identity
    monkeypatch.setattr(ControlledPhysicalInterfaceIonSystem, "evaluate",
                        lambda *a: pytest.fail("accepted rebase evaluated physics"))
    rebased, previous = working.rebase(accepted)
    assert rebased is not working and previous is not accepted
    assert rebased._step_reference is previous
    assert lift._words(previous.input_lift) == words
    assert lift._words(accepted.input_lift) == words
    assert accepted.coordinate_reference_identity == old_identity
    np.testing.assert_array_equal(accepted.coordinate, z)
    np.testing.assert_array_equal(previous.coordinate, 0.)
    expected_identity = lift._identity({name: accepted.input_lift[name] for name in lift.PRIMARY_FIELDS})
    assert previous.coordinate_reference_identity == expected_identity != old_identity
    assert rebased._input_lift_reference_identity == expected_identity
    assert rebased.reference_n is working.reference_n
    assert rebased.reference_p is working.reference_p
    assert rebased.reference_local_scale is working.reference_local_scale
    assert rebased.thermal_voltage == working.thermal_voltage
    previous.phi[1] += 1.
    assert accepted.phi[1] != previous.phi[1]
    assert rebased.input_lift_contract["initialization_evaluations"] == 0


def test_consecutive_rebases_accumulate_low_inputs_without_rounding(synthetic):
    _, _, working, saved, evaluate = synthetic
    first_z, second_z = np.zeros(15), np.zeros(15)
    first_z[[0, 7, 11]], second_z[[0, 7, 11]] = 2.**-60, 3*2.**-61
    first, _, _ = evaluate(first_z)
    next_system, local_first = working.rebase(first)
    second, _, _ = evaluate(second_z, active=next_system, prior=local_first)
    final_system, local_second = next_system.rebase(second)
    zero, _, _ = evaluate(np.zeros(15), active=final_system, prior=local_second)
    for name in lift.PRIMARY_FIELDS:
        np.testing.assert_array_equal(zero.input_lift[name].hi, second.input_lift[name].hi)
        np.testing.assert_array_equal(zero.input_lift[name].lo, second.input_lift[name].lo)
    with localcontext() as context:
        context.prec = 90
        exponent = sum(decimal(z[0])+decimal(z[7]) for z in (first_z, second_z))
        expected_n = decimal(saved.n[1])*exponent.exp()
        expected_phi = decimal(saved.phi[1])+decimal(working.thermal_voltage)*sum(decimal(z[7]) for z in (first_z, second_z))
        assert abs(represented(second.input_lift["n_m3"][1])-expected_n) < Decimal("1e-30")
        assert abs(represented(second.input_lift["phi_V"][1])-expected_phi) < Decimal("1e-32")
    assert second.input_lift["n_m3"].lo[1] != first.input_lift["n_m3"].lo[1]
    np.testing.assert_array_equal(second.n, saved.n)
    np.testing.assert_array_equal(final_system.storage_increment(zero, local_second), 0.)


def test_rebase_keeps_historical_anchors_but_evaluates_physical_electrostatics(synthetic):
    _, _, working, _, evaluate = synthetic
    accepted, _, _ = evaluate(np.zeros(15))
    values = dict(accepted.input_lift)
    values["poisson_residual_C_m2"] = DD([1., 2.], [2.**-60, -2.**-61])
    values["local_residual"] = lift.put(values["local_residual"], slice(0, 2), DD([3., 4.], [2.**-61, -2.**-62]))
    accepted = replace(accepted, input_lift=MappingProxyType(values),
        poisson_residual=values["poisson_residual_C_m2"].hi.copy(),
        local_residual=values["local_residual"].hi.copy())
    rebased, previous = working.rebase(accepted)
    state, _, _ = evaluate(np.zeros(15), active=rebased, prior=previous)
    for name, selection in (("poisson_residual_C_m2", slice(None)), ("local_residual", slice(0, 2))):
        np.testing.assert_array_equal(previous.input_lift[name][selection].hi, values[name][selection].hi)
        np.testing.assert_array_equal(previous.input_lift[name][selection].lo, values[name][selection].lo)
        assert not np.array_equal(state.input_lift[name][selection].hi, values[name][selection].hi)
    z = np.zeros(15)
    z[7] = 2.**-61
    changed, _, _ = evaluate(z, active=rebased, prior=previous)
    dphi = changed.input_lift["phi_V"]-previous.input_lift["phi_V"]
    delta = changed.input_lift["storage"]-previous.input_lift["storage"]
    rho = DD(Q)*(delta[2:4]-delta[:2])
    expected = state.input_lift["poisson_residual_C_m2"]+lift.diff(lift.diff(dphi))+rho
    np.testing.assert_allclose((changed.input_lift["poisson_residual_C_m2"]-expected).to_float(),
                               0., rtol=0., atol=1e-30)


def test_rebase_drops_old_voltage_lift_and_builds_from_new_reference(synthetic, monkeypatch):
    from perovskite_sim.solver import mol
    _, _, working, _, evaluate = synthetic
    accepted, _, _ = evaluate(np.zeros(15))
    working._lift, working._trace_lift = np.ones(4), np.ones((1, 2))
    working._lift_displacement, working._lift_free_residual = 4., True
    rebased, previous = working.rebase(accepted)
    for name in ("_lift", "_trace_lift", "_lift_displacement", "_lift_free_residual"):
        assert not hasattr(rebased, name)
    monkeypatch.setattr(mol, "poisson_right_boundary", lambda material, voltage: .5+voltage)
    rebased.set_voltage_lift(.125, previous)
    np.testing.assert_array_equal(rebased._lift, .125*np.arange(4)/3)
    assert rebased._lift[-1] == .125
    next_system, next_previous = rebased.rebase(previous)
    next_system.set_voltage_lift(0., next_previous)
    np.testing.assert_array_equal(next_system._lift, 0.)
    np.testing.assert_array_equal(next_system._trace_lift, 0.)
    np.testing.assert_array_equal(working._lift, 1.)


def test_explicit_constructor_lifts_baseline_seed_without_solving(synthetic, monkeypatch):
    original, previous, _, _, _ = synthetic
    monkeypatch.setattr(ControlledPhysicalInterfaceIonSystem, "evaluate", lambda *a: pytest.fail("constructor solved"))
    system = lift.RebasedInputLiftR1System(original, previous)
    saved = system._step_reference
    assert isinstance(saved, lift.InputLiftState)
    assert system.input_lift_contract["scope"] == "accepted_state_continuation"
    assert system.input_lift_contract["seed_representation"] == "float64-baseline"
    assert not hasattr(system, "_lift")
    for name in lift.PRIMARY_FIELDS:
        np.testing.assert_array_equal(saved.input_lift[name].lo, 0.)
    assert original._step_reference is previous
    with pytest.raises(TypeError, match="cannot replace"):
        system.rebase(previous)


def test_continuation_rejects_incoherent_or_incomplete_accepted_state(synthetic):
    _, _, working, _, evaluate = synthetic
    accepted, _, _ = evaluate(np.zeros(15))
    with pytest.raises(ValueError, match="disagrees"):
        working.rebase(replace(accepted, n=accepted.n+1.))
    incomplete = dict(accepted.input_lift)
    del incomplete["poisson_residual_C_m2"]
    with pytest.raises(ValueError, match="poisson_residual_C_m2"):
        working.rebase(replace(accepted, input_lift=MappingProxyType(incomplete)))
    working._input_lift_work = dict(accepted.input_lift)
    with pytest.raises(RuntimeError, match="in progress"):
        working.rebase(accepted)
    working._input_lift_work = None
    working.system.source_mat.carrier_params = {"carrier_statistics": "fermi_dirac"}
    with pytest.raises(ValueError, match="Maxwell-Boltzmann"):
        working.rebase(accepted)
