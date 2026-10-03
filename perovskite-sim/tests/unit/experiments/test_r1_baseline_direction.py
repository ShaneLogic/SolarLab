"""Baseline direction identities on algebraic states; no DC or Newton solve."""
from decimal import Decimal, localcontext
from types import SimpleNamespace

import numpy as np
import pytest

from perovskite_sim.constants import EPS_0, Q
from perovskite_sim.experiments import one_dimensional_mechanism_r1_baseline_electrostatics as electro
from perovskite_sim.experiments import one_dimensional_mechanism_r1_local_carrier as local_carrier
from perovskite_sim.experiments.one_dimensional_mechanism_r1_dynamics import ControlledPhysicalInterfaceIonSystem
from perovskite_sim.physics.two_sided_interface import TwoSidedInterfaceGeometry


@pytest.fixture
def direction_case(monkeypatch):
    system = object.__new__(ControlledPhysicalInterfaceIonSystem)
    system.node_count, system.interior_count, system.interface_count = 6, 4, 2
    system.electron_slice, system.hole_slice = slice(0, 4), slice(4, 8)
    system.trap_slice, system.positive_slice = slice(8, 10), slice(10, 16)
    system.negative_slice, system.potential_slice = slice(16, 16), slice(16, 20)
    system.local_slice, system.dimension = slice(20, 32), 32
    system.positive_nodes, system.negative_nodes = np.arange(6), np.empty(0, dtype=int)
    system.left_nodes, system.right_nodes, system.interface_faces = (1, 3), (2, 4), (1, 3)
    system.thermal_voltage, system.polarity = .25, 1.
    system.trap_density = np.array([3e17, 5e17])
    system.equilibrium_occupancy = np.array([.3, .4])
    system.widths = np.array([.5, 1.5, 2.25, 1.25, 2., .75])
    system.stack, system.dark_reference = object(), SimpleNamespace(interface_transmission=0.)
    system.system = SimpleNamespace(source_mat=SimpleNamespace())
    system.material = SimpleNamespace(has_dual_ions=False,
        N_D=np.array([.5, .125, 2., .5, 1., .75]) * 1e18,
        N_A=np.array([1., 3., .25, 2., .5, 1.5]) * 1e18,
        P_ion0=np.array([1., 1.25, .875, 1., 2., 1.5]) * 1e18,
        eps_r=np.array([1., 1.5, 2.5, 3., 1.25, 1.]),
        iface_qss_left_distances_m=np.array([2., 3.]) * EPS_0,
        iface_qss_right_distances_m=np.array([3., 2.]) * EPS_0,
        poisson_factor=SimpleNamespace(C=np.array([.75, 1.25, 2., 1.5, .875]),
                                       h_cell=system.widths[1:-1].copy()))
    geometries = tuple(TwoSidedInterfaceGeometry(
        left_distance_m=system.material.iface_qss_left_distances_m[k],
        right_distance_m=system.material.iface_qss_right_distances_m[k],
        eps_r_left=system.material.eps_r[left], eps_r_right=system.material.eps_r[right],
        fixed_sheet_charge_C_m2=.03125 * (k + 1),
        potential_jump_right_minus_left_V=.015625 * (k + 1))
        for k, (left, right) in enumerate(zip(system.left_nodes, system.right_nodes)))
    monkeypatch.setattr(local_carrier, "_material_two_sided_interface_problem",
                        lambda material, stack, n, p, phi, index, **kwargs:
                        (geometries[index], None, None))
    previous = SimpleNamespace(coordinate=np.linspace(-.03125, .0625, 32),
        n=np.arange(4., 10.) * 1e18, p=np.arange(6., 12.) * 1e18,
        positive=np.arange(1., 7.) * 1e18, negative=None,
        phi=np.array([.5, .75, -.5, 1.25, .875, -.25]),
        occupancy=np.array([.25, .375]),
        poisson_residual=np.array([999., -999., 888., -888.]),
        local_residual=np.arange(12.) + 777.,
        local=tuple(SimpleNamespace(trace_potential=np.array([.375, .875]) + k)
                    for k in range(2)),
        poisson_jacobian=object(), local_jacobian=object(), storage_jacobian=object(),
        rate_jacobian=object(), current_n=np.zeros(5), current_p=np.zeros(5),
        positive_current=np.zeros(5), negative_current=None)
    system._step_reference = previous
    system._lift = np.array([0., .007, .019, .031, .043, .0625])
    system._trace_lift = np.array([[.011, .012], [.037, .039]])
    system._lift_displacement, system._lift_free_residual = -.015, False
    delta = np.linspace(-2., 3., 32) * 2.**-18
    state = SimpleNamespace(**vars(previous))
    state.coordinate = previous.coordinate + delta
    state.phi = previous.phi.copy()
    state.phi[0] += .015625
    state.phi[-1] += system._lift[-1]
    state.phi[1:-1] += system.thermal_voltage * delta[system.potential_slice] + system._lift[1:-1]
    state.local = tuple(SimpleNamespace(trace_potential=item.trace_potential
        + system.thermal_voltage * delta[system._local_block_slice(k)][:2]
        + system._trace_lift[k]) for k, item in enumerate(previous.local))
    scales = (np.linspace(1., 2., 16), np.array([2., 4., 8., 16.]), np.arange(1., 13.))
    residual = np.linspace(-.04, .03, 32)
    return system, previous, state, residual, scales


def _decimal(value):
    return Decimal.from_float(float(value))


def _increment_oracle(system, state, previous):
    """Independent scalar divergence/trace assembly of the closure increments."""
    storage = system.storage_increment(state, previous)
    rho = system._increment_charge_density(storage)
    dz = state.coordinate - previous.coordinate
    potential = np.r_[state.phi[0] - previous.phi[0],
                       system.thermal_voltage * dz[system.potential_slice],
                       state.phi[-1] - previous.phi[-1]]
    if hasattr(system, "_lift"):
        potential[1:-1] += system._lift[1:-1]
    with localcontext() as context:
        context.prec = 90
        phi = list(map(_decimal, potential))
        capacitance = list(map(_decimal, system.material.poisson_factor.C))
        displacement = [-c * (right - left)
                        for c, left, right in zip(capacitance, phi[:-1], phi[1:])]
        charge = [_decimal(rho[j]) * _decimal(system.widths[j])
                  for j in range(system.node_count)]
        poisson = [displacement[j - 1] - displacement[j] + charge[j]
                   for j in range(1, system.node_count - 1)]
        local, sheet, trace_displacement = [], [], []
        for k, (left, right) in enumerate(zip(system.left_nodes, system.right_nodes)):
            cl = EPS_0 * system.material.eps_r[left] / system.material.iface_qss_left_distances_m[k]
            cr = EPS_0 * system.material.eps_r[right] / system.material.iface_qss_right_distances_m[k]
            weight_left, weight_right = cl / (cl + cr), cr / (cl + cr)
            sigma = -_decimal(Q) * _decimal(storage[2 * system.interior_count + k])
            sheet.append(sigma)
            poisson[left - 1] += _decimal(weight_left) * sigma
            poisson[right - 1] += _decimal(weight_right) * sigma
            trace = system.thermal_voltage * dz[system._local_block_slice(k)][:2]
            if hasattr(system, "_lift"):
                trace = trace + system._trace_lift[k]
            tl, tr = map(_decimal, trace)
            dl, dr = -_decimal(cl) * (tl - phi[left]), _decimal(cr) * (tr - phi[right])
            trace_displacement.append((dl, dr))
            local.extend((tr - tl, dr - dl - sigma))
        return SimpleNamespace(poisson=np.array(list(map(float, poisson))),
            local=np.array(list(map(float, local))), sheet=np.array(list(map(float, sheet))),
            displacement=displacement, charge=charge, trace_displacement=trace_displacement)


def _signature(value):
    if isinstance(value, np.ndarray):
        return (id(value), value.shape, value.dtype.str, value.tobytes())
    if isinstance(value, SimpleNamespace) or type(value) is ControlledPhysicalInterfaceIonSystem:
        return (id(value), {key: _signature(item) for key, item in vars(value).items()})
    if isinstance(value, (tuple, list)):
        return (id(value), tuple(_signature(item) for item in value))
    return value


def test_complete_increment_defect_matches_independent_charge_divergence(direction_case):
    system, previous, state, _, _ = direction_case
    assert np.all(previous.coordinate != 0.)
    expected = _increment_oracle(system, state, previous)
    rows = electro.baseline_incremental_electrostatic_defect(system, state, previous)
    np.testing.assert_array_equal(rows.poisson_residual_C_m2, expected.poisson)
    np.testing.assert_array_equal(rows.local_electrostatic_residual, expected.local)
    np.testing.assert_array_equal(rows.sheet_charge_C_m2, expected.sheet)
    # Telescoping the interior divergence and extending to both real contacts
    # accounts for endpoint ion storage as well as every weighted sheet.
    with localcontext() as context:
        context.prec = 90
        left_contact = expected.displacement[0] - expected.charge[0]
        right_contact = expected.displacement[-1] + expected.charge[-1]
        sheet = Decimal(0)
        for k in range(system.interface_count):
            wl, wr = system._sheet_weights(k)
            occupied = system.storage_increment(state, previous)[2 * system.interior_count + k]
            sheet += (_decimal(wl) + _decimal(wr)) * (-_decimal(Q) * _decimal(occupied))
        total = left_contact - right_contact + sum(expected.charge) + sheet
        assert abs(sum(rows.poisson_residual_C_m2) - float(total)) <= 8 * np.finfo(float).eps * max(abs(float(total)), .1)


def test_trace_rows_are_the_two_sided_displacement_and_sheet_balance(direction_case):
    system, previous, state, _, _ = direction_case
    expected = _increment_oracle(system, state, previous)
    rows = electro.baseline_incremental_electrostatic_defect(system, state, previous)
    storage = system.storage_increment(state, previous)
    dt = _decimal(2.5e-10)
    with localcontext() as context:
        context.prec = 90
        for k, (left, right) in enumerate(expected.trace_displacement):
            occupied = _decimal(storage[2 * system.interior_count + k])
            # Multiplication by 1/dt gives the exact displacement-current
            # difference plus trap charge rate, with both trace drops present.
            balance = right / dt - left / dt + _decimal(Q) * occupied / dt
            np.testing.assert_allclose(rows.local_electrostatic_residual[2 * k + 1] / float(dt),
                                       float(balance), rtol=3e-16, atol=0.)


def test_lift_is_complete_even_during_inherited_lift_free_mode(direction_case):
    system, previous, state, _, _ = direction_case
    expected = _increment_oracle(system, state, previous)
    system._lift_free_residual = True
    rows = electro.baseline_incremental_electrostatic_defect(system, state, previous)
    np.testing.assert_array_equal(rows.poisson_residual_C_m2, expected.poisson)
    np.testing.assert_array_equal(rows.local_electrostatic_residual, expected.local)
    assert system._lift_free_residual is True
    # Real contact increments are taken from the supplied states; neither
    # replacing the right difference with a lift nor forcing the left to zero
    # is compatible with this state pair.
    state.phi = state.phi.copy()
    state.phi[[0, -1]] += [.0078125, .03125]
    changed = electro.baseline_incremental_electrostatic_defect(system, state, previous)
    expected = _increment_oracle(system, state, previous)
    np.testing.assert_array_equal(changed.poisson_residual_C_m2, expected.poisson)
    assert changed.poisson_residual_C_m2[0] != rows.poisson_residual_C_m2[0]
    assert changed.poisson_residual_C_m2[-1] != rows.poisson_residual_C_m2[-1]


def test_sampled_harmonic_lift_residue_is_retained_at_every_interface(direction_case, monkeypatch):
    from perovskite_sim.solver import mol
    system, previous, state, _, _ = direction_case
    previous.phi = np.zeros(system.node_count)
    state.coordinate = previous.coordinate.copy()
    for k, (left, right) in enumerate(zip(system.left_nodes, system.right_nodes)):
        cl = EPS_0 * system.material.eps_r[left] / system.material.iface_qss_left_distances_m[k]
        cr = EPS_0 * system.material.eps_r[right] / system.material.iface_qss_right_distances_m[k]
        system.material.poisson_factor.C[system.interface_faces[k]] = cl * cr / (cl + cr)
    monkeypatch.setattr(mol, "poisson_right_boundary", lambda *args: .13)
    system.set_voltage_lift(.13, previous)
    state.phi = system._lift.copy()
    expected = _increment_oracle(system, state, previous)
    rows = electro.baseline_incremental_electrostatic_defect(system, state, previous)
    np.testing.assert_array_equal(rows.poisson_residual_C_m2, expected.poisson)
    np.testing.assert_array_equal(rows.local_electrostatic_residual, expected.local)
    assert np.any(expected.poisson != 0.)
    # Stored capacitances and sampled lift values are not assumed to cancel
    # exactly, even with zero coordinate and storage changes.
    assert np.any(expected.local[1::2] != 0.)


def test_subulp_increments_survive_identical_rounded_primary_fields(direction_case):
    system, previous, state, residual, scales = direction_case
    for name in ("_lift", "_trace_lift", "_lift_displacement"):
        delattr(system, name)
    previous.coordinate = np.full(system.dimension, 2.**-40)
    state.coordinate = previous.coordinate + np.arange(1., 33.) * 2.**-70
    state.phi, state.local = previous.phi, previous.local
    before = electro.baseline_physical_electrostatic_rows(system, previous)
    after = electro.baseline_physical_electrostatic_rows(system, state)
    np.testing.assert_array_equal(before.poisson_residual_C_m2, after.poisson_residual_C_m2)
    np.testing.assert_array_equal(before.local_electrostatic_residual, after.local_electrostatic_residual)
    target = system.newton_residual_target(previous, *scales)
    residual[16:20] = target[16:20]
    for k in range(system.interface_count):
        residual[20 + 6 * k:22 + 6 * k] = target[20 + 6 * k:22 + 6 * k]
    rhs = system.newton_direction_rhs(state, previous, residual, target, *scales)
    expected = _increment_oracle(system, state, previous)
    np.testing.assert_array_equal(rhs[16:20], expected.poisson / scales[1])
    assert np.any(rhs[16:20] != 0.)
    assert np.all((residual - target)[16:20] == 0.)


def test_direction_preserves_dynamic_carrier_rows_and_all_source_objects(direction_case):
    system, previous, state, residual, scales = direction_case
    target = system.newton_residual_target(previous, *scales)
    before = _signature((system, previous, state, residual, target, scales))
    rhs = system.newton_direction_rhs(state, previous, residual, target, *scales)
    assert _signature((system, previous, state, residual, target, scales))[1] == before[1]
    assert not np.shares_memory(rhs, residual) and not np.shares_memory(rhs, target)
    rows = electro.baseline_incremental_electrostatic_defect(system, state, previous)
    np.testing.assert_array_equal(rhs[16:20], rows.poisson_residual_C_m2 / scales[1])
    kept = list(range(16))
    for k in range(system.interface_count):
        start = 20 + 6 * k
        np.testing.assert_array_equal(rhs[start:start + 2],
            rows.local_electrostatic_residual[2 * k:2 * k + 2] / scales[2][6 * k:6 * k + 2])
        kept.extend(range(start + 2, start + 6))
    np.testing.assert_array_equal(rhs[kept], residual[kept])


@pytest.mark.parametrize("change", ["zero", "current", "component", "shape", "nan", "stale_previous"])
def test_direction_rejects_targets_not_bound_to_original_previous_physics(direction_case, change):
    system, previous, state, residual, scales = direction_case
    target = system.newton_residual_target(previous, *scales)
    if change == "zero":
        target[:] = 0.
    elif change == "current":
        target = residual.copy()
    elif change == "component":
        target[21] = np.nextafter(target[21], np.inf)
    elif change == "shape":
        target = target[:-1]
    elif change == "nan":
        target[16] = np.nan
    else:
        previous.phi = previous.phi.copy()
        previous.phi[1] += .125
    with pytest.raises(ValueError, match="previous accepted state"):
        system.newton_direction_rhs(state, previous, residual, target, *scales)


def test_poisoned_saved_residuals_do_not_change_the_previous_bound_direction(direction_case):
    system, previous, state, residual, scales = direction_case
    target = system.newton_residual_target(previous, *scales)
    rhs = system.newton_direction_rhs(state, previous, residual, target, *scales)
    previous.poisson_residual[:] = -12345.
    previous.local_residual[:] = 12345.
    np.testing.assert_array_equal(system.newton_residual_target(previous, *scales), target)
    np.testing.assert_array_equal(system.newton_direction_rhs(state, previous, residual, target, *scales), rhs)


def test_other_modes_keep_existing_override_and_residual_minus_target_fallback(monkeypatch):
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_input_lift import RebasedInputLiftR1System
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_precision import CompensatedR1System

    class OtherSystem(ControlledPhysicalInterfaceIonSystem):
        pass

    def forbidden(*args, **kwargs):
        pytest.fail("baseline direction helper entered another representation")

    monkeypatch.setattr(electro, "baseline_physical_electrostatic_rows", forbidden)
    monkeypatch.setattr(electro, "baseline_incremental_electrostatic_defect", forbidden)
    residual, target = np.arange(9.), np.arange(9.) / 7.
    assert CompensatedR1System.newton_direction_rhs is not ControlledPhysicalInterfaceIonSystem.newton_direction_rhs
    assert RebasedInputLiftR1System.newton_direction_rhs is ControlledPhysicalInterfaceIonSystem.newton_direction_rhs
    for kind in (OtherSystem, RebasedInputLiftR1System, CompensatedR1System):
        system = object.__new__(kind)
        actual = system.newton_direction_rhs(SimpleNamespace(), SimpleNamespace(), residual, target,
                                            np.ones(2), np.ones(1), np.ones(6))
        np.testing.assert_array_equal(actual, residual - target)
