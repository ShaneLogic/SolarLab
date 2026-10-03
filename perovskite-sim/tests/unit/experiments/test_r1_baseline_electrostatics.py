"""Actual binary64 baseline electrostatics; no DC or nonlinear solve."""
from dataclasses import fields, replace
from decimal import Decimal, localcontext
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

from perovskite_sim.constants import EPS_0, Q
from perovskite_sim.experiments import one_dimensional_mechanism_r1_baseline_electrostatics as electro
from perovskite_sim.experiments import one_dimensional_mechanism_r1_local_carrier as local_carrier
from perovskite_sim.experiments import interface_defect_transient as base
from perovskite_sim.experiments.interface_defect_transient import _LocalState, _InterfaceTransientSystem
from perovskite_sim.experiments.interface_defect_ion_transient import _InterfaceIonDeviceState
from perovskite_sim.experiments.one_dimensional_mechanism_r1 import PhysicalInterfaceIonSystem
from perovskite_sim.experiments.one_dimensional_mechanism_r1_dynamics import ControlledPhysicalInterfaceIonSystem
from perovskite_sim.physics.two_sided_interface import TwoSidedBulkState, TwoSidedInterfaceGeometry


@pytest.fixture
def baseline(monkeypatch):
    system = object.__new__(ControlledPhysicalInterfaceIonSystem)
    system.node_count, system.interior_count, system.interface_count, system.dimension = 4, 2, 1, 15
    system.electron_slice, system.hole_slice, system.trap_slice = slice(0, 2), slice(2, 4), slice(4, 5)
    system.positive_slice, system.negative_slice = slice(5, 7), slice(7, 7)
    system.potential_slice, system.local_slice = slice(7, 9), slice(9, 15)
    system.positive_nodes, system.negative_nodes = np.array([1, 2]), np.empty(0, dtype=int)
    system.left_nodes, system.right_nodes = (1,), (2,)
    system.trap_density, system.equilibrium_occupancy = np.array([3.]), np.array([.3])
    system.thermal_voltage = .25
    system.stack, system.dark_reference = object(), SimpleNamespace(interface_transmission=0.)
    system.system = SimpleNamespace(source_mat=SimpleNamespace())
    system.material = SimpleNamespace(has_dual_ions=False,
        N_D=np.array([.5, .125, 2., .5]), N_A=np.array([1., 3., .25, 2.]),
        P_ion0=np.array([1., 1.25, .875, 1.]),
        eps_r=np.array([1., 1.5, 2.5, 1.]),
        iface_qss_left_distances_m=np.array([2 * EPS_0]),
        iface_qss_right_distances_m=np.array([3 * EPS_0]),
        poisson_factor=SimpleNamespace(C=np.array([.75, 1.25, 2.]), h_cell=np.array([1.5, 2.25])))
    geometry = TwoSidedInterfaceGeometry(left_distance_m=2 * EPS_0,
        right_distance_m=3 * EPS_0, eps_r_left=1.5, eps_r_right=2.5)
    monkeypatch.setattr(local_carrier, "_material_two_sided_interface_problem",
                        lambda *args, **kwargs: (geometry, None, None))
    values = {item.name: None for item in fields(_InterfaceIonDeviceState)}
    zero = sparse.csr_matrix((15, 15))
    values.update(coordinate=np.zeros(15), dqfn=np.full(4, .125), dqfp=np.full(4, -.125),
        n=np.array([4., 4., 8., 8.]), p=np.array([6., 6., 10., 10.]),
        phi=np.array([.5, .75, -.5, 1.25]), occupancy=np.array([.25]),
        positive=np.ones(4), storage=np.array([4., 8., 6., 10., .75, 1., 1.]),
        rate=np.zeros(7), positive_rate=np.zeros(4), current_n=np.zeros(3), current_p=np.zeros(3),
        positive_flux=np.zeros(3), carrier_conduction=np.zeros(3), positive_current=np.zeros(3),
        conduction=np.zeros(3), sheet_charge=np.array([Q * .15]),
        poisson_residual=np.array([.125, -.25]), local_residual=np.array([.2, -.3, 2., 3., 4., 5.]),
        storage_jacobian=zero[:7], rate_jacobian=zero[:7],
        poisson_jacobian=zero[:2], local_jacobian=zero[:6])
    values["local"] = (_LocalState(trace_potential=np.array([.375, .875]),
        log_state=np.log([4., 6., 8., 10.]), state_m3=np.array([4., 6., 8., 10.]),
        quasi_steady_occupancy=.25, sheet_charge_C_m2=Q * .15,
        electrostatic_residual=np.array([.2, -.3]), tangent=object()),)
    previous = _InterfaceIonDeviceState(**values)
    system._step_reference = previous
    return system, previous, geometry


def _decimal(value):
    return Decimal.from_float(float(value))


def _oracle(system, state, geometry):
    """Scalar physical equations independent of DD and production helpers."""
    with localcontext() as context:
        context.prec = 90
        phi = list(map(_decimal, state.phi))
        mat = system.material
        sheet = -_decimal(Q) * _decimal(system.trap_density[0]) * (
            _decimal(state.occupancy[0]) - _decimal(system.equilibrium_occupancy[0]))
        result = []
        for node, weight in zip((1, 2), system._sheet_weights(0)):
            rho = _decimal(Q) * (_decimal(state.p[node]) - _decimal(state.n[node])
                + _decimal(mat.N_D[node]) - _decimal(mat.N_A[node])
                + _decimal(state.positive[node]) - _decimal(mat.P_ion0[node]))
            result.append(_decimal(mat.poisson_factor.C[node]) * (phi[node + 1] - phi[node])
                - _decimal(mat.poisson_factor.C[node - 1]) * (phi[node] - phi[node - 1])
                + rho * _decimal(mat.poisson_factor.h_cell[node - 1]) + _decimal(weight) * sheet)
        trace = list(map(_decimal, state.local[0].trace_potential))
        cl = _decimal(EPS_0 * geometry.eps_r_left / geometry.left_distance_m)
        cr = _decimal(EPS_0 * geometry.eps_r_right / geometry.right_distance_m)
        local = [trace[1] - trace[0] - _decimal(geometry.potential_jump_right_minus_left_V),
                 cl * (trace[0] - phi[1]) + cr * (trace[1] - phi[2])
                 - _decimal(geometry.fixed_sheet_charge_C_m2) - sheet]
        return np.array(list(map(float, result))), np.array(list(map(float, local))), float(sheet)


def _copy_state(state, **changes):
    return replace(state, poisson_residual=state.poisson_residual.copy(),
                   local_residual=state.local_residual.copy(), **changes)


def test_absolute_rows_match_decimal_and_ignore_poisoned_saved_residuals(baseline):
    system, previous, geometry = baseline
    expected_poisson, expected_local, expected_sheet = _oracle(system, previous, geometry)
    rows = electro.baseline_physical_electrostatic_rows(system, previous)
    np.testing.assert_array_equal(rows.poisson_residual_C_m2, expected_poisson)
    np.testing.assert_array_equal(rows.local_electrostatic_residual, expected_local)
    assert rows.sheet_charge_C_m2[0] == expected_sheet
    before = previous.poisson_residual.copy(), previous.local_residual.copy()
    diagnostic = electro.baseline_electrostatic_diagnostics(system, previous)
    np.testing.assert_array_equal(diagnostic["historical_incremental_poisson_residual_C_m2"], before[0])
    np.testing.assert_array_equal(diagnostic["historical_incremental_local_electrostatic_residual"], before[1][:2])
    assert np.max(np.abs(diagnostic["absolute_minus_historical_poisson_C_m2"])) > 1.
    assert not diagnostic["historical_incremental_is_live_equation"]
    np.testing.assert_array_equal(previous.poisson_residual, before[0])
    np.testing.assert_array_equal(previous.local_residual, before[1])
    previous.poisson_residual[:] = 999.
    previous.local_residual[:2] = -999.
    changed = electro.baseline_physical_electrostatic_rows(system, previous)
    np.testing.assert_array_equal(changed.poisson_residual_C_m2, expected_poisson)
    np.testing.assert_array_equal(changed.local_electrostatic_residual, expected_local)


def test_live_hook_retains_binary64_state_and_primary_array_identity(baseline):
    system, previous, _ = baseline
    state = _copy_state(previous)
    names = ("coordinate", "dqfn", "dqfp", "n", "p", "phi", "positive", "occupancy", "storage", "rate")
    primary = {name: getattr(state, name) for name in names}
    local_primary = state.local[0]
    jacobians = {name: getattr(state, name) for name in (
        "storage_jacobian", "rate_jacobian", "poisson_jacobian", "local_jacobian")}
    result = system._with_step_electrostatics(state)
    assert result is state and type(result) is _InterfaceIonDeviceState
    assert not hasattr(result, "fine") and not hasattr(result, "input_lift")
    for name, value in primary.items():
        assert getattr(result, name) is value and value.dtype == np.float64
    for name, value in jacobians.items():
        assert getattr(result, name) is value
    for name in ("trace_potential", "log_state", "state_m3", "tangent"):
        assert getattr(result.local[0], name) is getattr(local_primary, name)
    np.testing.assert_array_equal(result.local_residual[2:], previous.local_residual[2:])
    np.testing.assert_array_equal(result.direct_poisson_residual, previous.poisson_residual)
    np.testing.assert_array_equal(previous.local[0].electrostatic_residual, [.2, -.3])


def test_actual_sampled_harmonic_lift_is_not_assumed_exactly_divergence_free(baseline, monkeypatch):
    from perovskite_sim.solver import mol
    system, previous, geometry = baseline
    previous.phi[:] = 0.
    previous.n[:] = previous.p[:]
    previous.positive[:] = system.material.P_ion0[:]
    system.material.N_D[:] = system.material.N_A[:]
    previous.occupancy[:] = system.equilibrium_occupancy[:]
    previous.poisson_residual[:] = 0.
    previous.local_residual[:2] = 0.
    previous.local[0].trace_potential[:] = 0.
    monkeypatch.setattr(mol, "poisson_right_boundary", lambda *args: .13)
    system.set_voltage_lift(.13, previous)
    state = _copy_state(previous, phi=system._lift.copy(),
        local=(replace(previous.local[0], trace_potential=system._trace_lift[0].copy()),))
    before_system = vars(system).copy()
    diagnostic = electro.baseline_electrostatic_diagnostics(system, state)
    np.testing.assert_array_equal(diagnostic["historical_incremental_poisson_residual_C_m2"], 0.)
    expected, expected_local, _ = _oracle(system, state, geometry)
    assert np.any(expected != 0.)
    result = system._with_step_electrostatics(state)
    np.testing.assert_array_equal(result.poisson_residual, expected)
    np.testing.assert_array_equal(result.local_residual[:2], expected_local)
    assert vars(system).keys() == before_system.keys()
    for name, value in before_system.items():
        assert vars(system)[name] is value


def test_reassembly_at_identical_primary_state_is_reproducible_after_rebase(baseline):
    system, previous, _ = baseline
    state = system._with_step_electrostatics(_copy_state(previous))
    # Rebase's zero coordinate and updated historical residuals do not change
    # either primary array or the absolute equation evaluated from it.
    zero = _copy_state(state, coordinate=np.zeros(system.dimension))
    system._step_reference = state
    again = system._with_step_electrostatics(zero)
    assert again.poisson_residual.tobytes() == state.poisson_residual.tobytes()
    assert again.local_residual.tobytes() == state.local_residual.tobytes()


def test_direction_target_uses_previous_physics_without_changing_other_rows(baseline):
    system, previous, geometry = baseline
    expected_poisson, expected_local, _ = _oracle(system, previous, geometry)
    storage_scale, poisson_scale, local_scale = np.ones(7), np.array([2., 4.]), np.arange(1., 7.)
    raw_poisson, raw_local = previous.poisson_residual.copy(), previous.local_residual.copy()
    target = system.newton_residual_target(previous, storage_scale, poisson_scale, local_scale)
    np.testing.assert_array_equal(target[:7], 0.)
    np.testing.assert_array_equal(target[7:9], expected_poisson / poisson_scale)
    np.testing.assert_array_equal(target[9:11], expected_local / local_scale[:2])
    np.testing.assert_array_equal(target[11:], 0.)
    np.testing.assert_array_equal(previous.poisson_residual, raw_poisson)
    np.testing.assert_array_equal(previous.local_residual, raw_local)
    assert not np.allclose(target[7:9], raw_poisson / poisson_scale)


def test_absolute_rows_match_inherited_analytic_electrostatic_jacobian(baseline, monkeypatch):
    system, state, geometry = baseline
    # Raise all charge densities together so finite differences can resolve
    # every electrostatic block, while keeping the original physical charge Q.
    for name in ("n", "p", "positive"):
        getattr(state, name)[:] *= 1e18
    for name in ("N_D", "N_A", "P_ion0"):
        getattr(system.material, name)[:] *= 1e18
    system.trap_density[:] *= 1e18
    system.grid, system.widths = np.arange(4.), np.array([.5, 1.5, 2.25, .5])
    system.interface_faces = (1,)
    system._poisson_laplacian = system._build_poisson_laplacian()
    system._divergence = system._build_divergence_matrix()
    for name in ("ni_sq", "tau_n", "tau_p", "n1", "p1", "B_rad", "C_n", "C_p", "chi", "Eg"):
        setattr(system.material, name, np.zeros(4))
    system.material.D_n_face = system.material.D_p_face = np.zeros(3)
    system.system.source_mat.neutral_bulk_defects = None
    system._local_carrier_jacobians = lambda *args: (
        sparse.csr_matrix((4, 15)), sparse.csr_matrix((1, 15)), sparse.csr_matrix((4, 15)))
    monkeypatch.setattr(base, "total_recombination_derivatives", lambda *args, **kwargs:
        SimpleNamespace(electron_density_derivative=np.zeros(4), hole_density_derivative=np.zeros(4)))
    empty_flux = SimpleNamespace(density_left_derivative=np.zeros(3), density_right_derivative=np.zeros(3),
        potential_left_derivative=np.zeros(3), potential_right_derivative=np.zeros(3))
    monkeypatch.setattr(base, "sg_fluxes_n_jacobian", lambda *args: empty_flux)
    monkeypatch.setattr(base, "sg_fluxes_p_jacobian", lambda *args: empty_flux)
    monkeypatch.setattr(base, "_material_two_sided_interface_problem", lambda mat, stack, n, p, phi, *a, **k:
        (geometry, None, TwoSidedBulkState(phi[1], phi[2], n[1], p[1], n[2], p[2])))
    _, _, poisson, local = system._jacobians(state.phi, state.n, state.p, state.occupancy, state.local)
    ion_density = system._density_jacobian(state.positive, system.positive_nodes, system.positive_slice, 15)
    poisson += sparse.diags(system.material.poisson_factor.h_cell) @ (Q * ion_density[1:-1])
    expected = np.vstack((poisson.toarray(), local[:2].toarray()))

    def actual(coordinate):
        n, p, phi, ion = (getattr(state, name).copy() for name in ("n", "p", "phi", "positive"))
        dphi = coordinate[system.potential_slice]
        n[1:-1] *= np.exp(coordinate[system.electron_slice] + dphi)
        p[1:-1] *= np.exp(coordinate[system.hole_slice] - dphi)
        phi[1:-1] += system.thermal_voltage * dphi
        ion[system.positive_nodes] *= np.exp(coordinate[system.positive_slice])
        f = state.occupancy
        occupancy = 1. / (1. + (1. - f) / f * np.exp(-coordinate[system.trap_slice]))
        trace = state.local[0].trace_potential + system.thermal_voltage * coordinate[9:11]
        shifted = replace(state, n=n, p=p, phi=phi, positive=ion, occupancy=occupancy,
                          local=(replace(state.local[0], trace_potential=trace),))
        rows = electro.baseline_physical_electrostatic_rows(system, shifted)
        return np.r_[rows.poisson_residual_C_m2, rows.local_electrostatic_residual]

    step = 2e-6
    for column in range(15):
        coordinate = np.zeros(15)
        coordinate[column] = step
        secant = (actual(coordinate) - actual(-coordinate)) / (2 * step)
        np.testing.assert_allclose(secant, expected[:, column], rtol=2e-8, atol=3e-10)


def test_static_trace_coefficients_are_preserved(baseline, monkeypatch):
    system, previous, geometry = baseline
    geometry = replace(geometry, fixed_sheet_charge_C_m2=.125,
                       potential_jump_right_minus_left_V=.03125)
    monkeypatch.setattr(local_carrier, "_material_two_sided_interface_problem",
                        lambda *args, **kwargs: (geometry, None, None))
    rows = electro.baseline_physical_electrostatic_rows(system, previous)
    _, expected, _ = _oracle(system, previous, geometry)
    np.testing.assert_array_equal(rows.local_electrostatic_residual, expected)


@pytest.mark.parametrize("owner", ["material", "source"])
def test_additional_charged_species_cannot_be_silently_omitted(baseline, owner):
    system, previous, _ = baseline
    target = system.material if owner == "material" else system.system.source_mat
    target.monovalent_bulk_defects = object()
    with pytest.raises(ValueError, match="additional bulk charge"):
        electro.baseline_physical_electrostatic_rows(system, previous)


def test_pair_fallback_and_other_subclasses_keep_their_original_hook(monkeypatch):
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_precision import CompensatedR1System
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_input_lift import RebasedInputLiftR1System
    sentinel = object()
    calls = []
    monkeypatch.setattr(PhysicalInterfaceIonSystem, "_with_step_electrostatics",
                        lambda self, state: calls.append((self, state)) or sentinel)
    monkeypatch.setattr(electro, "baseline_physical_electrostatic_rows",
                        lambda *args: pytest.fail("baseline helper entered another representation"))
    pair = object.__new__(CompensatedR1System)
    pair._fine_work = {}
    state = object()
    assert pair._with_step_electrostatics(state) is sentinel
    lift = object.__new__(RebasedInputLiftR1System)
    assert ControlledPhysicalInterfaceIonSystem._with_step_electrostatics(lift, state) is sentinel
    assert calls == [(pair, state), (lift, state)]
    assert "_jacobians" not in ControlledPhysicalInterfaceIonSystem.__dict__
    assert ControlledPhysicalInterfaceIonSystem._jacobians is _InterfaceTransientSystem._jacobians
