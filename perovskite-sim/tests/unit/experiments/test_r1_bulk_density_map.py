"""V35 bulk-map regressions, with explicitly separated execution scopes.

The ``test_array_*`` cases allocate synthetic arrays and call coordinate/input
maps only; their systems bypass constructors and no physical kernel is called.
``test_actual_n16_*`` uses the existing pair preparation and real evaluation
kernel. Its body takes no transient Newton direction or trajectory step;
fixture preparation includes DC, Poisson and local solves in the same budget.
It is marked slow.

The two A35 cases contain only the eight independently audited local outputs.
They do not claim correct rounding for the other 1008 outputs in that probe.
"""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from perovskite_sim.experiments import interface_defect_transient as transient
from perovskite_sim.experiments import one_dimensional_mechanism_r1_dynamics as dynamics
from perovskite_sim.experiments.one_dimensional_mechanism_r1_precision import (
    CompensatedR1System,
)
from perovskite_sim.physics.compensated import DD
from tests.integration.test_r1_v9_backend import pair_preparation


@pytest.fixture
def coordinate_factory(monkeypatch):
    """Use real coordinate methods, replacing only contact boundary providers."""
    monkeypatch.setattr(transient, "_contact_quasi_fermi_increments", lambda *args: None)
    monkeypatch.setattr(transient, "poisson_right_boundary", lambda *args: 0.0)

    def build(cls=dynamics.ControlledPhysicalInterfaceIonSystem, *, nodes=256,
              interfaces=0, rebased=True):
        system = cls.__new__(cls)
        interior = nodes - 2
        system.node_count, system.interior_count = nodes, interior
        system.interface_count = interfaces
        system.electron_slice = slice(0, interior)
        system.hole_slice = slice(interior, 2 * interior)
        system.trap_slice = slice(2 * interior, 2 * interior + interfaces)
        start = system.trap_slice.stop
        system.positive_slice = system.negative_slice = slice(start, start)
        system.potential_slice = slice(start, start + interior)
        system.local_slice = slice(start + interior, start + interior + 6 * interfaces)
        system.dimension = system.local_slice.stop
        system.left_nodes = (1,) if interfaces else ()
        system.right_nodes = (nodes - 2,) if interfaces else ()
        system.thermal_voltage = 1.0
        system.material = object()
        system.system = SimpleNamespace(
            phi0=np.zeros(nodes), log_n0=np.zeros(nodes), log_p0=np.zeros(nodes),
        )
        system.dqfn_dc, system.dqfp_dc = np.zeros(nodes), np.zeros(nodes)
        system.qfn_reference, system.qfp_reference = np.zeros(nodes), np.zeros(nodes)
        system.reference_phi = np.zeros(nodes)
        system.reference_n, system.reference_p = np.full(nodes, 4.0), np.full(nodes, 8.0)
        system.reference_n[[0, -1]], system.reference_p[[0, -1]] = [7.0, 11.0], [13.0, 17.0]
        system.reference_occupancy = np.full(interfaces, 0.25)
        system.reference_logit = np.full(interfaces, np.log(0.25) - np.log(0.75))
        system.reference_trace_potential = np.zeros((interfaces, 2))
        system.reference_trace_log_state = np.zeros((interfaces, 4))
        system._step_reference = (SimpleNamespace(
            n=np.arange(nodes, dtype=float) + 64.0,
            p=np.arange(nodes, dtype=float) + 512.0,
            occupancy=system.reference_occupancy.copy(),
        ) if rebased else None)
        return system

    return build


def test_array_baseline_dispatches_all_254_interior_nodes_without_old_exp_work(
    coordinate_factory, monkeypatch,
):
    system = coordinate_factory()
    coordinate = np.zeros(system.dimension, dtype=np.float32)
    coordinate[system.electron_slice] = np.resize([0.125, -0.25, 0.0], 254)
    coordinate[system.hole_slice] = np.resize([0.0625, 0.125, 0.0], 254)
    coordinate[system.potential_slice] = np.resize([0.25, -0.125, 0.0], 254)
    previous = system._step_reference
    before_n, before_p, before_coordinate = previous.n.copy(), previous.p.copy(), coordinate.copy()
    previous.n.flags.writeable = previous.p.flags.writeable = False
    coordinate.flags.writeable = False
    outputs = (np.arange(254, dtype=float) + 4096.0,
               np.arange(254, dtype=float) + 8192.0)
    calls, exp_sizes = [], []

    def resolved_update(anchor, increment):
        calls.append((anchor.copy(), increment.copy()))
        assert len(calls) <= 2, "bulk map split or repeated an interior helper call"
        return outputs[len(calls) - 1]

    original_exp = np.exp

    def endpoint_exp_only(value, *args, **kwargs):
        size = np.asarray(value).size
        assert size <= 4, "baseline computed a discarded full/interior exp array"
        exp_sizes.append(size)
        return original_exp(value, *args, **kwargs)

    monkeypatch.setattr(dynamics, "log_density_update", resolved_update)
    monkeypatch.setattr(np, "exp", endpoint_exp_only)
    result = system._coordinates(coordinate, 0.0)
    n, p = result[3:5]

    assert len(calls) == 2 and sum(exp_sizes) == 4
    np.testing.assert_array_equal(calls[0][0], before_n[1:-1])
    np.testing.assert_array_equal(calls[1][0], before_p[1:-1])
    np.testing.assert_array_equal(calls[0][1], np.resize([0.375, -0.375, 0.0], 254))
    np.testing.assert_array_equal(calls[1][1], np.resize([-0.1875, 0.25, 0.0], 254))
    for actual, expected, anchor in ((n, outputs[0], previous.n), (p, outputs[1], previous.p)):
        assert actual.shape == (256,) and actual.dtype == np.dtype(np.float64)
        np.testing.assert_array_equal(actual[1:-1], expected)
        assert not np.shares_memory(actual, anchor)
        assert not np.shares_memory(actual, expected)
    assert not np.shares_memory(n, p)
    np.testing.assert_array_equal(n[[0, -1]], system.reference_n[[0, -1]])
    np.testing.assert_array_equal(p[[0, -1]], system.reference_p[[0, -1]])
    np.testing.assert_array_equal(previous.n, before_n)
    np.testing.assert_array_equal(previous.p, before_p)
    np.testing.assert_array_equal(coordinate, before_coordinate)


def test_array_baseline_zero_increment_preserves_previous_words_and_ownership(coordinate_factory):
    system = coordinate_factory()
    previous = system._step_reference
    previous.n[1:-1] = np.nextafter(previous.n[1:-1], np.inf)
    previous.p[1:-1] = np.nextafter(previous.p[1:-1], 0.0)
    before_n, before_p = previous.n.copy(), previous.p.copy()
    previous.n.flags.writeable = previous.p.flags.writeable = False
    coordinate = np.zeros(system.dimension)
    coordinate.flags.writeable = False

    n, p = system._coordinates(coordinate, 0.0)[3:5]

    np.testing.assert_array_equal(n[1:-1], before_n[1:-1])
    np.testing.assert_array_equal(p[1:-1], before_p[1:-1])
    np.testing.assert_array_equal(n[[0, -1]], system.reference_n[[0, -1]])
    np.testing.assert_array_equal(p[[0, -1]], system.reference_p[[0, -1]])
    for actual, anchor in ((n, previous.n), (p, previous.p)):
        assert actual.shape == (256,) and actual.dtype == np.dtype(np.float64)
        assert not np.shares_memory(actual, anchor)
        actual[127] += 1.0
    assert not np.shares_memory(n, p)
    np.testing.assert_array_equal(previous.n, before_n)
    np.testing.assert_array_equal(previous.p, before_p)
    np.testing.assert_array_equal(coordinate, np.zeros(system.dimension))


@pytest.mark.parametrize("kind", ["generic_initial", "generic_rebased", "r1_initial"])
def test_array_generic_and_unrebased_r1_keep_original_exp_path(coordinate_factory, monkeypatch, kind):
    cls = (dynamics.ControlledPhysicalInterfaceIonSystem if kind == "r1_initial"
           else transient._InterfaceTransientSystem)
    system = coordinate_factory(cls, rebased=kind == "generic_rebased")
    calls = []
    original_exp = np.exp

    def observed_exp(value, *args, **kwargs):
        if np.asarray(value).size:
            calls.append(np.asarray(value).copy())
        return original_exp(value, *args, **kwargs)

    monkeypatch.setattr(np, "exp", observed_exp)
    monkeypatch.setattr(dynamics, "log_density_update",
                        lambda *args: pytest.fail("default path entered the R1 stable map"))
    n, p = system._coordinates(np.zeros(system.dimension), 0.0)[3:5]

    expected_shapes = [(256,), (256,)]
    if kind == "generic_rebased":
        expected_shapes += [(254,), (254,)]
        np.testing.assert_array_equal(n[1:-1], system._step_reference.n[1:-1])
        np.testing.assert_array_equal(p[1:-1], system._step_reference.p[1:-1])
    else:
        np.testing.assert_array_equal(n[1:-1], np.ones(254))
        np.testing.assert_array_equal(p[1:-1], np.ones(254))
    assert [value.shape for value in calls] == expected_shapes
    for value in calls:
        np.testing.assert_array_equal(value, np.zeros(value.shape))
    expected_n = system.reference_n[[0, -1]] if kind == "r1_initial" else np.ones(2)
    expected_p = system.reference_p[[0, -1]] if kind == "r1_initial" else np.ones(2)
    np.testing.assert_array_equal(n[[0, -1]], expected_n)
    np.testing.assert_array_equal(p[[0, -1]], expected_p)


@pytest.mark.parametrize("cls", [transient._InterfaceTransientSystem,
                                  dynamics.ControlledPhysicalInterfaceIonSystem])
def test_array_actual_coordinates_dispatch_density_hook_before_reservoir_pin(
    coordinate_factory, monkeypatch, cls,
):
    system = coordinate_factory(cls, rebased=False)
    coordinate = np.zeros(system.dimension)
    coordinate[system.electron_slice] = 0.25
    coordinate[system.hole_slice] = 0.5
    coordinate[system.potential_slice] = -0.125
    supplied_n, supplied_p = np.arange(256, dtype=float) + 2.0, np.arange(256, dtype=float) + 3.0
    calls = []

    def density_hook(values, log_n, log_p):
        calls.append(values.copy())
        np.testing.assert_array_equal(log_n[1:-1], np.full(254, 0.125))
        np.testing.assert_array_equal(log_p[1:-1], np.full(254, 0.625))
        return supplied_n.copy(), supplied_p.copy()

    monkeypatch.setattr(system, "_bulk_density_coordinates", density_hook)
    n, p = system._coordinates(coordinate, 0.0)[3:5]
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0], coordinate)
    np.testing.assert_array_equal(n[1:-1], supplied_n[1:-1])
    np.testing.assert_array_equal(p[1:-1], supplied_p[1:-1])
    pinned = cls is dynamics.ControlledPhysicalInterfaceIonSystem
    np.testing.assert_array_equal(n[[0, -1]],
                                  (system.reference_n if pinned else supplied_n)[[0, -1]])
    np.testing.assert_array_equal(p[[0, -1]],
                                  (system.reference_p if pinned else supplied_p)[[0, -1]])


@pytest.mark.parametrize("invalid", ["shape", "nonfinite"])
def test_array_coordinate_validation_still_precedes_bulk_hook(coordinate_factory, monkeypatch, invalid):
    system = coordinate_factory()
    coordinate = np.zeros(system.dimension + (invalid == "shape"))
    if invalid == "nonfinite":
        coordinate[system.electron_slice.start] = np.nan
    monkeypatch.setattr(system, "_bulk_density_coordinates",
                        lambda *args: pytest.fail("invalid coordinate reached density hook"))
    with pytest.raises(transient.InterfaceDefectTransientError, match="finite vector"):
        system._coordinates(coordinate, 0.0)


@pytest.mark.parametrize("field,node", [("log_n0", 127), ("log_p0", 0)])
def test_array_absolute_log_overflow_still_precedes_bulk_hook(
    coordinate_factory, monkeypatch, field, node,
):
    system = coordinate_factory()
    getattr(system.system, field)[node] = 800.0
    monkeypatch.setattr(system, "_bulk_density_coordinates",
                        lambda *args: pytest.fail("absolute log overflow reached density hook"))
    with pytest.raises(transient.InterfaceDefectTransientError, match="carrier coordinate overflow"):
        system._coordinates(np.zeros(system.dimension), 0.0)


@pytest.mark.parametrize("cls", [transient._InterfaceTransientSystem,
                                  dynamics.ControlledPhysicalInterfaceIonSystem])
@pytest.mark.parametrize("species,node,value", [("n", 0, np.nan), ("p", -1, 0.0),
                                                ("n", 127, np.inf), ("p", 128, -1.0)])
def test_array_parent_rejects_invalid_hook_density_including_pinned_endpoints(
    coordinate_factory, monkeypatch, cls, species, node, value,
):
    system = coordinate_factory(cls)
    n, p = np.ones(256), np.ones(256)
    (n if species == "n" else p)[node] = value
    monkeypatch.setattr(system, "_bulk_density_coordinates", lambda *args: (n, p))
    with pytest.raises(transient.InterfaceDefectTransientError, match="non-positive or non-finite"):
        system._coordinates(np.zeros(system.dimension), 0.0)


@pytest.mark.parametrize("field,node", [("log_n0", 0), ("log_p0", -1)])
def test_array_real_endpoint_underflow_is_not_hidden_by_reservoir_pin(coordinate_factory, field, node):
    system = coordinate_factory()
    getattr(system.system, field)[node] = -1000.0
    with pytest.raises(transient.InterfaceDefectTransientError, match="non-positive or non-finite"):
        system._coordinates(np.zeros(system.dimension), 0.0)


# From A35 BulkMapProbeResultV1.json, contract
# 2258a905e532e0b49a1fd29b8f9669a1cf6f2eaa7eb073b7033c3ad6e96484ac.
# Anchors are n(left), p(left), n(right), p(right) at nodes 127 and 128.
# Coordinates are qfn(left/right), qfp(left/right), phi(left/right), in that order.
# Expected words agree with both 60/90-digit exact-coordinate and rounded-
# exponent references. No external archive is read when these tests run.
_A35_ANCHORS = (
    "0x1.8e64f96073565p+38", "0x1.7563c9e1c95fep+43",
    "0x1.278c851ea23ccp+44", "0x1.f79a9ab705696p+37",
)
_A35_POINTS = (
    pytest.param(
        ("-0x1.e7b108785de31p-28", "-0x1.5fc9d448b08b8p-27",
         "0x1.1d261c5f8c92bp-26", "0x1.d9dac78ba0e27p-27",
         "-0x1.11f253b48b130p-19", "-0x1.114ce501e1b9ep-19"),
        ("0x1.8e64c3e6ae1ebp+38", "0x1.7563fc3be2472p+43",
         "0x1.278c5d7b2c8dap+44", "0x1.f79ade5ff3c2bp+37"),
        id="terminal",
    ),
    pytest.param(
        ("-0x1.e7b106dad6e7cp-28", "-0x1.5fc9d464d3ce7p-27",
         "0x1.1d261c541b505p-26", "0x1.d9dac84c3906dp-27",
         "-0x1.11f253b48b1abp-19", "-0x1.114ce501e1c19p-19"),
        ("0x1.8e64c3e6ae1eep+38", "0x1.7563fc3be2472p+43",
         "0x1.278c5d7b2c8d9p+44", "0x1.f79ade5ff3c2ep+37"),
        id="alpha1",
    ),
)


@pytest.mark.parametrize("coordinate_hex,expected_hex", _A35_POINTS)
def test_array_a35_saved_local_bulk_words_match_independent_hex_reference(coordinate_hex, expected_hex):
    system = dynamics.ControlledPhysicalInterfaceIonSystem.__new__(dynamics.ControlledPhysicalInterfaceIonSystem)
    system.node_count, system.interior_count = 256, 254
    system.electron_slice, system.hole_slice = slice(0, 254), slice(254, 508)
    system.potential_slice = slice(637, 891)
    n_before, p_before = np.ones(256), np.ones(256)
    n_left, p_left, n_right, p_right = map(float.fromhex, _A35_ANCHORS)
    n_before[[127, 128]], p_before[[127, 128]] = [n_left, n_right], [p_left, p_right]
    n_before.flags.writeable = p_before.flags.writeable = False
    system._step_reference = SimpleNamespace(n=n_before, p=p_before)
    coordinate = np.zeros(897)
    coordinate[[126, 127, 380, 381, 763, 764]] = [float.fromhex(word) for word in coordinate_hex]
    coordinate.flags.writeable = False

    n, p = system._bulk_density_coordinates(coordinate, np.zeros(256), np.zeros(256))

    actual = (n[127], p[127], n[128], p[128])
    assert tuple(float(value).hex() for value in actual) == expected_hex
    assert n.dtype == p.dtype == np.dtype(np.float64)
    assert n.shape == p.shape == (256,)
    assert not np.shares_memory(n, n_before) and not np.shares_memory(p, p_before)


def test_array_compensated_normal_fine_map_bypasses_baseline_and_preserves_low_words(
    coordinate_factory, monkeypatch,
):
    system = coordinate_factory(CompensatedR1System, nodes=4, interfaces=1)
    system._fine_work = {}
    monkeypatch.setattr(system, "_refresh_fine_constants", lambda: {"vt": DD(1.0)})
    monkeypatch.setattr(dynamics.ControlledPhysicalInterfaceIonSystem, "_bulk_density_coordinates",
                        lambda *args: pytest.fail("compensated path entered baseline stable bulk map"))
    original_generic = transient._InterfaceTransientSystem._bulk_density_coordinates
    generic_calls = []

    def observed_generic(current, values, log_n, log_p):
        generic_calls.append(current)
        return original_generic(current, values, log_n, log_p)

    monkeypatch.setattr(transient._InterfaceTransientSystem, "_bulk_density_coordinates", observed_generic)
    coordinate = np.zeros(system.dimension)
    tiny = np.ldexp(1.0, -70)
    retained, local_inputs, references = [], [], []
    before_n, before_p = system._step_reference.n.copy(), system._step_reference.p.copy()
    for multiplier in (1.0, 2.0):
        low_n = multiplier * tiny * np.array([0.0, 1.0, 2.0, 0.0])
        low_p = multiplier * tiny * np.array([0.0, -3.0, -4.0, 0.0])
        reference = {
            "phi_V": DD(system.reference_phi),
            "dqfn_V": DD(system.dqfn_dc), "dqfp_V": DD(system.dqfp_dc),
            "n_m3": DD(system.reference_n, low_n), "p_m3": DD(system.reference_p, low_p),
            "occupancy": DD(system.reference_occupancy),
            "trace_potential_V": DD(system.reference_trace_potential),
            "trace_state_m3": DD(np.ones((1, 4))),
        }
        system._fine_reference = reference
        result = system._coordinates(coordinate, 0.0)
        for name, index in (("n_m3", 3), ("p_m3", 4)):
            np.testing.assert_array_equal(result[index], reference[name].hi)
            np.testing.assert_array_equal(system._fine_work[name].hi, reference[name].hi)
            np.testing.assert_array_equal(system._fine_work[name].lo, reference[name].lo)
        inputs = system._local_carrier_inputs(
            0, result[3], result[4], result[2], result[5], result[6], result[7],
            trace_density_m3=system._trace_density_coordinates(coordinate),
        )
        np.testing.assert_array_equal(inputs.bulk_density.hi, [4.0, 8.0, 4.0, 8.0])
        np.testing.assert_array_equal(inputs.bulk_density.lo,
                                      multiplier * tiny * np.array([1.0, -3.0, 2.0, -4.0]))
        retained.append(system._fine_work)
        local_inputs.append(inputs)
        references.append(reference)

    assert generic_calls == [system, system]
    assert retained[0] is not retained[1]
    for name in ("n_m3", "p_m3"):
        np.testing.assert_array_equal(retained[0][name].hi, retained[1][name].hi)
        assert np.all(retained[0][name].lo[1:-1] != 0.0)
        assert np.all(retained[0][name].lo[1:-1] != retained[1][name].lo[1:-1])
        np.testing.assert_array_equal(retained[0][name].lo, references[0][name].lo)
    np.testing.assert_array_equal(local_inputs[0].bulk_density.lo,
                                  tiny * np.array([1.0, -3.0, 2.0, -4.0]))
    np.testing.assert_array_equal(system._step_reference.n, before_n)
    np.testing.assert_array_equal(system._step_reference.p, before_p)


@pytest.mark.slow
def test_actual_n16_compensated_bulk_map_keeps_state_and_local_low_words(pair_preparation, monkeypatch):
    """Real N16 preparation/evaluation; synthetic low words, no trajectory step."""
    from threadpoolctl import threadpool_limits
    from perovskite_sim.experiments import one_dimensional_mechanism_r1_state as states

    stack, binding, prepared = pair_preparation
    with threadpool_limits(1):
        system, initial = states.verify_prepared_physics(prepared, stack, binding, backend="pair")
        monkeypatch.setattr(dynamics.ControlledPhysicalInterfaceIonSystem, "_bulk_density_coordinates",
                            lambda *args: pytest.fail("normal pair evaluation entered baseline bulk map"))
        results, references = [], []
        for fraction in (0.125, 0.25):
            fine = dict(initial.fine)
            for name, sign in (("n_m3", 1.0), ("p_m3", -1.0)):
                original = initial.fine[name]
                low = original.lo.copy()
                low[1:-1] = sign * fraction * np.spacing(original.hi[1:-1])
                fine[name] = DD(original.hi, low)
                np.testing.assert_array_equal(fine[name].hi, original.hi)
                assert np.all(fine[name].lo[1:-1] != 0.0)
            previous = replace(initial, fine=fine)
            working, _ = system.rebase(previous)
            current = working.evaluate(np.zeros(working.dimension), 0.0)
            assert working.interface_count > 0
            assert len(current.local) == working.interface_count
            for name, physical in (("n_m3", current.n), ("p_m3", current.p)):
                np.testing.assert_array_equal(current.fine[name].hi, fine[name].hi)
                np.testing.assert_array_equal(current.fine[name].lo, fine[name].lo)
                np.testing.assert_array_equal(physical, current.fine[name].hi)
                # DD.copy may safely share immutable backing bytes. State
                # ownership concerns its value object and retained words.
                assert current.fine[name] is not working._fine_work[name]
                assert not current.fine[name].hi.flags.writeable
                assert not current.fine[name].lo.flags.writeable
            for index, item in enumerate(current.local):
                left, right = working.left_nodes[index], working.right_nodes[index]
                expected = (fine["n_m3"][left], fine["p_m3"][left],
                            fine["n_m3"][right], fine["p_m3"][right])
                bulk = item.carrier_data.inputs.bulk_density
                np.testing.assert_array_equal(bulk.hi, [float(value.hi) for value in expected])
                np.testing.assert_array_equal(bulk.lo, [float(value.lo) for value in expected])
            results.append(current)
            references.append(fine)

        for name in ("n_m3", "p_m3"):
            np.testing.assert_array_equal(results[0].fine[name].hi, results[1].fine[name].hi)
            assert np.all(results[0].fine[name].lo[1:-1] != results[1].fine[name].lo[1:-1])
            np.testing.assert_array_equal(results[0].fine[name].lo, references[0][name].lo)
        for index, item in enumerate(results[0].local):
            left, right = system.left_nodes[index], system.right_nodes[index]
            old = references[0]
            np.testing.assert_array_equal(item.carrier_data.inputs.bulk_density.lo,
                                          [old["n_m3"].lo[left], old["p_m3"].lo[left],
                                           old["n_m3"].lo[right], old["p_m3"].lo[right]])
