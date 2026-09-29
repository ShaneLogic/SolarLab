"""R1 consumer identities and data lifetime, plus one actual N16 smoke."""
from copy import deepcopy
from dataclasses import dataclass, replace
import hashlib
import json
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

from perovskite_sim.constants import Q
from perovskite_sim.experiments import interface_defect_transient as transient
from perovskite_sim.experiments import interface_defect_ion_transient as ionic
from perovskite_sim.experiments import one_dimensional_mechanism_r1_local_carrier as local
from perovskite_sim.experiments.one_dimensional_mechanism_r1_dynamics import ControlledPhysicalInterfaceIonSystem
from perovskite_sim.physics.compensated import DD
from perovskite_sim.physics.two_sided_interface import (
    TwoSidedBulkState, TwoSidedInterfaceGeometry, TwoSidedInterfacePhysics,
)
from tests.integration.test_r1_v9_backend import pair_preparation


class ConsumerSystem(ControlledPhysicalInterfaceIonSystem):
    """Four physical nodes and one local interface; no preparation or solve."""

    def __init__(self):
        self.node_count, self.interior_count, self.interface_count = 4, 2, 1
        self.dimension = 13
        self.electron_slice, self.hole_slice = slice(0, 2), slice(2, 4)
        self.trap_slice, self.potential_slice, self.local_slice = slice(4, 5), slice(5, 7), slice(7, 13)
        self.positive_slice = self.negative_slice = slice(5, 5)
        self.left_nodes, self.right_nodes, self.interface_faces = (1,), (2,), (1,)
        self.widths = np.array([0.5, 2.0, 4.0, 0.5])
        self.thermal_voltage, self.polarity, self.capture_multiplier = 1.0, 1.0, 1.0
        self.trap_density, self.equilibrium_occupancy = np.array([2.0]), np.array([0.25])
        self.reference_local_scale = np.ones((1, 6))
        self.stack, self.material = object(), object()
        self.dark_reference = SimpleNamespace(interface_transmission=0.0)
        self._step_reference = None
        self.geometry = TwoSidedInterfaceGeometry(2.0, 4.0, 1.0, 1.0)
        self.physics = TwoSidedInterfacePhysics(
            thermal_voltage_V=1.0, temperature_K=1.0,
            D_n_left_m2_s=1.0, D_p_left_m2_s=1.0,
            D_n_right_m2_s=1.0, D_p_right_m2_s=1.0,
            N_C_left_m3=1024.0, N_V_left_m3=1024.0,
            N_C_right_m3=1024.0, N_V_right_m3=1024.0,
            richardson_n_A_m2_K2=Q, richardson_p_A_m2_K2=Q,
            transmission=0.0, surface_recombination_velocity_n_m_s=2.0,
            surface_recombination_velocity_p_m_s=3.0,
            n1_left_m3=1.0, p1_left_m3=2.0, n1_right_m3=3.0, p1_right_m3=4.0,
        )


@dataclass
class RetainedState:
    coordinate: np.ndarray
    dqfn: np.ndarray
    dqfp: np.ndarray
    n: np.ndarray
    p: np.ndarray
    phi: np.ndarray
    occupancy: np.ndarray
    positive: np.ndarray
    negative: object
    local: tuple


@pytest.fixture
def consumer(monkeypatch):
    system = ConsumerSystem()
    table = {"eta": [-40.0, 20.0], "log_half": [-40.0, 20.0]}
    identity = hashlib.sha256(json.dumps(table, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    monkeypatch.setattr(local, "default_fd_table", lambda: (table, identity))

    def material_problem(material, stack, n, p, phi, index, *, cross_transmission):
        assert material is system.material and stack is system.stack and index == 0
        assert cross_transmission == 0.0
        return system.geometry, system.physics, TwoSidedBulkState(
            phi[1], phi[2], n[1], p[1], n[2], p[2])

    monkeypatch.setattr(local, "_material_two_sided_interface_problem", material_problem)

    def build(*, trace=(4.0, 6.0, 8.0, 10.0), bulk=(6.0, 4.0, 20.0, 2.0), occupancy=0.25):
        n, p = np.ones(4), np.ones(4)
        n[[1, 2]], p[[1, 2]] = [bulk[0], bulk[2]], [bulk[1], bulk[3]]
        phi, trace_phi = np.zeros(4), np.zeros((1, 2))
        density = np.asarray([trace], dtype=float)
        locals_, aggregate = system._local_states(n, p, phi, np.array([occupancy]),
            trace_phi, np.log(density), trace_density_m3=density)
        state = RetainedState(np.zeros(13), np.zeros(4), np.zeros(4), n, p, phi,
                              np.array([occupancy]), np.ones(4), None, locals_)
        return state, aggregate

    return system, build


def consumers(system, state, aggregate):
    source = local.assemble_r1_carrier_source(system, np.zeros(8), aggregate)
    rates = system._carrier_rate_fields(source, np.zeros(3), np.zeros(3), state.local)
    jacobians = system._local_carrier_jacobians(0, 1, 2, state.local[0], 123.0)
    conduction = system._interface_carrier_conduction(state.local[0])
    return (source.hi.copy(), source.lo.copy(), *[value.copy() for value in rates],
            *[value.toarray() for value in jacobians], conduction.copy())


def test_state_owned_payload_survives_unrelated_evaluation_replace_rebase_and_copy(consumer):
    system, build = consumer
    a, aggregate_a = build()
    payload = a.local[0].carrier_data
    assert aggregate_a.carrier_data[0] is payload
    before = consumers(system, a, aggregate_a)
    b, aggregate_b = build(trace=(1.0, 2.0, 3.0, 4.0), occupancy=0.6)
    assert b.local[0].carrier_data is aggregate_b.carrier_data[0]
    assert b.local[0].carrier_data is not payload
    assert b.local[0].carrier_data.coefficient_identity == payload.coefficient_identity
    for actual, expected in zip(consumers(system, a, aggregate_a), before, strict=True):
        np.testing.assert_array_equal(actual, expected)
    replaced = replace(a.local[0], electrostatic_residual=np.array([0.0, 1e-30]))
    assert replaced.carrier_data is payload
    _, rebased = system.rebase(a)
    assert rebased.local[0].carrier_data is payload
    assert deepcopy(a).local[0].carrier_data is payload
    with pytest.raises(TypeError):
        payload.balance["bulk_flux_m2s"] = DD(np.zeros(4))
    with pytest.raises(ValueError):
        payload.balance["bulk_flux_m2_s"].hi[0] = 99.0


def test_four_carrier_signs_right_first_layout_receiving_widths_and_trap_reduction(consumer, monkeypatch):
    system, build = consumer
    state, aggregate = build()
    np.testing.assert_array_equal(aggregate.state_m3, [8.0, 10.0, 4.0, 6.0])
    np.testing.assert_array_equal(aggregate.bulk_flux_m2_s, [3.0, -2.0, 1.0, -1.0])
    np.testing.assert_array_equal(state.local[0].tangent.balance.capture_flux_m2_s, [5.5, 0.0, 10.5, -1.5])
    np.testing.assert_array_equal(state.local[0].tangent.balance.residual_m2_s, [-4.5, -1.0, -7.5, -0.5])
    calls = []

    def bulk_source(self, n, p, phi, voltage, explicit_qss):
        calls.append(explicit_qss)
        assert explicit_qss is not aggregate
        assert explicit_qss.carrier_data is aggregate.carrier_data
        np.testing.assert_array_equal(explicit_qss.bulk_flux_m2_s, np.zeros(4))
        np.testing.assert_array_equal(explicit_qss.capture_flux_m2_s, aggregate.capture_flux_m2_s)
        return np.zeros(8)

    monkeypatch.setattr(transient._InterfaceTransientSystem, "_source", bulk_source)
    source = system._source(state.n, state.p, state.phi, 0.0, aggregate)
    assert len(calls) == 1 and isinstance(source, DD)
    np.testing.assert_array_equal(source.to_float(), [0.0, -0.5, -0.75, 0.0, 0.0, 0.5, 0.5, 0.0])
    rn, rp, rt = system._carrier_rate_fields(source, DD(Q)*DD([2, 6, 10]), DD(Q)*DD([3, 2, 7]), state.local)
    np.testing.assert_allclose(rn, [4.0, 1.5, 0.25, -20.0], rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(rp, [-6.0, 1.0, -0.75, 14.0], rtol=1e-14, atol=1e-14)
    np.testing.assert_array_equal(rt, [17.5])
    jn, jp = local.r1_reported_interface_currents(system, state.local, np.zeros(3), np.zeros(3))
    np.testing.assert_allclose([jn[1], jp[1]], [-Q, -Q], rtol=1e-15)
    np.testing.assert_allclose(system._interface_carrier_conduction(state.local[0]), Q*np.array([-2.0, 5.0]), rtol=1e-15)
    loss, trap, carrier = system._local_carrier_jacobians(0, 1, 2, state.local[0], -999.0)
    assert loss.shape == carrier.shape == (4, 13) and trap.shape == (1, 13)
    assert loss[0, 0] == 1.5 and loss[0, 9] == -1.0
    assert loss[0, 5] == 0.25  # VT*dF/dphi + dF/dlog(n), before receiving width.
    assert carrier[0, 9] == -8.0 and trap[0, 9] == 6.0
    np.testing.assert_array_equal(carrier[:, 4].toarray().ravel(), [1.875, -4.5, 4.125, -7.875])
    assert trap[0, 4] == -18.375


def test_full_low_word_reaches_residual_source_reported_current_and_cancelled_jacobian(consumer):
    system, _ = consumer
    system.physics = replace(system.physics, n1_left_m3=1.0, p1_left_m3=1.0,
                             n1_right_m3=1.0, p1_right_m3=1.0)
    n = p = np.ones(4)
    phi, trace_phi, trace_density = np.zeros(4), np.zeros((1, 2)), np.ones((1, 4))
    occupancy = np.array([0.5])
    geometry, physics = system.geometry, system.physics
    fine = {"n_m3": DD(n), "p_m3": DD(p), "phi_V": DD(phi), "occupancy": DD(occupancy),
            "trace_potential_V": DD(trace_phi), "trace_state_m3": DD(trace_density)}

    def evaluate(values):
        inputs = local.fine_local_carrier_inputs(system, 0, n, p, phi, occupancy,
            trace_phi, None, trace_density_m3=trace_density, fine=values)
        data = local.evaluate_local_carrier_inputs(system, 0, geometry, physics, inputs)
        item = local._StableLocalState(trace_potential=trace_phi[0], log_state=np.zeros(4),
            state_m3=np.ones(4), quasi_steady_occupancy=0.5, sheet_charge_C_m2=0.0,
            electrostatic_residual=np.zeros(2), tangent=data.production_tangent(), carrier_data=data)
        aggregate = local._StableInterfaceQSS(state_m3=np.ones(4), bulk_flux_m2_s=np.zeros(4),
            cross_flux_m2_s=np.zeros(4), state_flux_m2_s=np.zeros(4), normalized_residual=0.0,
            evaluations=0, transport_model="unused_by_this_consumer", capture_flux_m2_s=np.zeros(4),
            occupancy=occupancy, carrier_data=(data,))
        return item, aggregate

    a, qa = evaluate(fine)
    delta = np.ldexp(1.0, -70)
    changed = dict(fine, n_m3=DD(n, [0.0, delta, 0.0, 0.0]))
    b, qb = evaluate(changed)
    np.testing.assert_array_equal(changed["n_m3"].hi, fine["n_m3"].hi)
    assert bool(a.carrier_data.balance["residual_m2_s"][0] == 0)
    assert bool(b.carrier_data.balance["residual_m2_s"][0] > 0)
    source_a = local.assemble_r1_carrier_source(system, np.zeros(8), qa)
    source_b = local.assemble_r1_carrier_source(system, np.zeros(8), qb)
    assert bool(source_a[1] == 0) and bool(source_b[1] < 0)
    ca = system._interface_carrier_conduction(a)
    cb = system._interface_carrier_conduction(b)
    assert ca[0] == 0.0 and cb[0] < 0.0
    ja = system._local_carrier_jacobians(0, 1, 2, a, 0.25)[0]
    jb = system._local_carrier_jacobians(0, 1, 2, b, 0.25)[0]
    assert ja[0, 5] == 0.0 and jb[0, 5] > 0.0
    np.testing.assert_allclose(jb[0, 5], delta/8.0, rtol=1e-12, atol=0.0)


def test_initial_float_map_and_explicit_fine_words_are_kept_distinct(consumer):
    system, _ = consumer
    n, p, phi = np.ones(4), np.ones(4), np.zeros(4)
    log_state = np.array([[0.1, 0.2, 0.3, 0.4]])
    trace_phi, occupancy = np.zeros((1, 2)), np.array([0.25])
    initial = local.float_local_carrier_inputs(system, 0, n, p, phi, occupancy, trace_phi, log_state)
    np.testing.assert_array_equal(initial.state_density.hi, np.exp(log_state[0]))
    np.testing.assert_array_equal(initial.state_density.lo, np.zeros(4))
    density = np.array([[1.0, 2.0, 3.0, 4.0]])
    resolved = local.float_local_carrier_inputs(system, 0, n, p, phi, occupancy, trace_phi,
        np.full((1, 4), 999.0), trace_density_m3=density)
    np.testing.assert_array_equal(resolved.state_density.hi, density[0])
    tiny = np.ldexp(1.0, -70)
    fine = {"n_m3": DD(n), "p_m3": DD(p), "phi_V": DD(phi), "occupancy": DD(occupancy),
            "trace_potential_V": DD(trace_phi), "trace_state_m3": DD(density, [[tiny, 0, 0, 0]])}
    complete = local.fine_local_carrier_inputs(system, 0, n, p, phi, occupancy, trace_phi,
        log_state, trace_density_m3=density, fine=fine)
    assert float(complete.state_density.lo[0]) == tiny
    with pytest.raises(ValueError, match="high words"):
        local.fine_local_carrier_inputs(system, 0, n*2, p, phi, occupancy, trace_phi,
            log_state, trace_density_m3=density, fine=fine)
    with pytest.raises(ValueError, match="resolved trace"):
        local.fine_local_carrier_inputs(system, 0, n, p, phi, occupancy, trace_phi, log_state, fine=fine)


def test_independent_reconstruction_ignores_corrupted_payload_and_later_direct_work(consumer, monkeypatch):
    system, build = consumer
    a, _ = build()
    honest = local.independent_interface_evaluation(system, a, 0)
    data = a.local[0].carrier_data
    bad = replace(data, balance=MappingProxyType(dict(data.balance, bulk_flux_m2_s=DD([99, 98, 97, 96]))))
    wrong_local = replace(a.local[0], carrier_data=bad,
        tangent=replace(a.local[0].tangent, balance=replace(a.local[0].tangent.balance,
            bulk_flux_m2_s=np.array([99.0, 98.0, 97.0, 96.0]))))
    wrong = replace(a, local=(wrong_local,))
    build(trace=(9.0, 7.0, 5.0, 3.0))
    system._fine_work = {"n_m3": DD(np.full(4, 999.0))}
    monkeypatch.setattr(system, "_local_states", lambda *args, **kwargs: pytest.fail("independent path called direct builder"))
    recovered = local.independent_interface_evaluation(system, wrong, 0)
    assert recovered is not honest and recovered is not data and recovered is not bad
    for name in ("bulk_flux_m2_s", "cross_flux_m2_s", "capture_flux_m2_s", "residual_m2_s"):
        np.testing.assert_array_equal(recovered.balance[name].hi, honest.balance[name].hi)
        np.testing.assert_array_equal(recovered.balance[name].lo, honest.balance[name].lo)
    assert not np.array_equal(recovered.balance["bulk_flux_m2_s"].to_float(), wrong.local[0].tangent.balance.bulk_flux_m2_s)


@pytest.mark.parametrize("evaluate", [transient._InterfaceTransientSystem.evaluate, ionic._InterfaceIonTransientSystem.evaluate])
def test_both_actual_evaluate_paths_dispatch_the_carrier_rate_hook(evaluate):
    class ReachedHook(Exception):
        pass
    source, tn, tp = object(), object(), object()
    locals_ = (object(),)
    fake = SimpleNamespace(
        _coordinates=lambda *args: (None, None, None, None, None, None, None, None),
        _ion_coordinates=lambda *args: (None, None),
        _trace_density_coordinates=lambda *args: None,
        _local_states=lambda *args, **kwargs: (locals_, object()),
        _source=lambda *args: source,
        _currents=lambda *args: (tn, tp, None, None),
    )
    def rate(actual_source, actual_tn, actual_tp, actual_local):
        assert actual_source is source and actual_tn is tn and actual_tp is tp and actual_local is locals_
        raise ReachedHook
    fake._carrier_rate_fields = rate
    with pytest.raises(ReachedHook):
        evaluate(fake, np.zeros(1), 0.0)


def test_non_r1_default_rate_hook_keeps_original_float_arithmetic():
    divergence = sparse.csr_matrix([[1., 0., 0.], [-1., 1., 0.], [0., -1., 1.], [0., 0., -1.]])
    fake = SimpleNamespace(_divergence=divergence, node_count=4, widths=np.array([1., 2., 4., 8.]))
    source = np.array([1e16, -3., 5., 7., 11., 13., 17., 19.])
    tn, tp = np.array([1e-3, -2e-3, 3e-3]), np.array([-5e-3, 7e-3, -11e-3])
    capture = np.array([[1e16, 1., -1e16, 2.]])
    locals_ = (SimpleNamespace(tangent=SimpleNamespace(balance=SimpleNamespace(capture_flux_m2_s=capture[0]))),)
    rn, rp, rt = transient._InterfaceTransientSystem._carrier_rate_fields(fake, source, tn, tp, locals_)
    np.testing.assert_array_equal(rn, source[:4] + (divergence@tn)/(Q*fake.widths))
    np.testing.assert_array_equal(rp, source[4:] - (divergence@tp)/(Q*fake.widths))
    np.testing.assert_array_equal(rt, capture[:, [0, 2]].sum(axis=1) - capture[:, [1, 3]].sum(axis=1))


@pytest.mark.slow
def test_actual_n16_state_lifetime_and_existing_snapshot_schema(pair_preparation):
    from threadpoolctl import threadpool_limits
    from perovskite_sim.experiments import one_dimensional_mechanism_r1_state as states
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_pair_codec import FINE_FIELDS, SNAPSHOT_FIELDS

    stack, binding, prepared = pair_preparation
    with threadpool_limits(1):
        system, a = states.verify_prepared_physics(prepared, stack, binding, backend="pair")
        before = states.snapshot(system, a)
        assert set(before) == SNAPSHOT_FIELDS and set(a.fine) == FINE_FIELDS
        payload = a.local[0].carrier_data
        conduction = system._interface_carrier_conduction(a.local[0]).copy()
        jacobian = system._local_carrier_jacobians(0, system.left_nodes[0], system.right_nodes[0], a.local[0], 0.0)
        coordinate = a.coordinate.copy()
        coordinate[system._local_block_slice(0).start+2] += 1e-8
        b = system.evaluate(coordinate, 0.0)
        assert b.local[0].carrier_data is not payload
        assert states.snapshot(system, a) == before
        np.testing.assert_array_equal(system._interface_carrier_conduction(a.local[0]), conduction)
        again = system._local_carrier_jacobians(0, system.left_nodes[0], system.right_nodes[0], a.local[0], 0.0)
        for expected, actual in zip(jacobian, again, strict=True):
            np.testing.assert_array_equal(expected.toarray(), actual.toarray())
        _, rebased = system.rebase(a)
        assert rebased.local[0].carrier_data is payload
        assert replace(a.local[0], sheet_charge_C_m2=a.local[0].sheet_charge_C_m2).carrier_data is payload
        assert set(states.snapshot(system, b)) == SNAPSHOT_FIELDS
