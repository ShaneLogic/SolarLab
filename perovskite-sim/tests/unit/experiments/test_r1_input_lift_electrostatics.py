"""Manufactured DD electrostatic references; no DC or nonlinear solve."""
from dataclasses import replace
from decimal import Decimal, localcontext
from types import MappingProxyType

import numpy as np
import pytest

from perovskite_sim.constants import EPS_0, Q
from perovskite_sim.experiments import one_dimensional_mechanism_r1_input_lift as lift
from perovskite_sim.experiments import one_dimensional_mechanism_r1_local_carrier as local
from perovskite_sim.experiments.one_dimensional_mechanism_r1_dynamics import ControlledPhysicalInterfaceIonSystem
from perovskite_sim.physics.compensated import DD
from tests.unit.experiments.test_r1_input_lift import synthetic


def _decimal(value):
    return Decimal.from_float(float(value))


def _represented(value):
    return _decimal(value.hi)+_decimal(value.lo)


@pytest.fixture
def asymmetric(synthetic, monkeypatch):
    baseline, raw, _, _, evaluate = synthetic
    mat = baseline.material
    mat.N_D, mat.N_A = np.array([.5, .125, 2., .5]), np.array([1., 3., .25, 2.])
    mat.P_ion0 = np.array([1., 1.25, .875, 1.])
    mat.poisson_factor.C = np.array([.75, 1.25, 2.])
    mat.poisson_factor.h_cell = np.array([1.5, 2.25])
    mat.eps_r = np.array([1., 1.5, 2.5, 1.])
    mat.iface_qss_left_distances_m = np.array([2*EPS_0])
    mat.iface_qss_right_distances_m = np.array([3*EPS_0])
    original_problem = local._material_two_sided_interface_problem

    def problem(material, *args, **kwargs):
        geometry, physics, bulk = original_problem(material, *args, **kwargs)
        geometry = replace(geometry,
            left_distance_m=material.iface_qss_left_distances_m[0],
            right_distance_m=material.iface_qss_right_distances_m[0],
            eps_r_left=material.eps_r[1], eps_r_right=material.eps_r[2])
        return geometry, physics, bulk

    monkeypatch.setattr(local, "_material_two_sided_interface_problem", problem)
    raw.phi[:] = [.5, .75, -.5, 1.25]
    raw.local[0].trace_potential[:] = [.375, .875]
    raw.poisson_residual[:] = [.125, -.25]
    raw.local_residual[:2] = [.2, -.3]
    operator, seed = lift.from_saved_step(baseline, raw)
    value = dict(seed.input_lift)
    value["phi_V"] = DD(seed.phi, [0., 2.**-60, -2.**-61, 0.])
    value["n_m3"] = DD(seed.n, [0., 2.**-58, -2.**-59, 0.])
    value["occupancy"] = DD(seed.occupancy, 2.**-61)
    value["trace_potential_V"] = DD([[.375, .875]], [[-2.**-62, 2.**-61]])
    value["storage"] = lift._storage(operator, value)
    value["sheet_charge_C_m2"] = -DD(Q)*DD(operator.trap_density)*(value["occupancy"]-DD(operator.equilibrium_occupancy))
    value["poisson_residual_C_m2"] = DD(seed.poisson_residual, [2.**-64, -2.**-65])
    value["local_residual"] = DD(seed.local_residual, [2.**-65, -2.**-66, 0., 0., 0., 0.])
    seed = replace(seed, input_lift=MappingProxyType(value),
        storage=value["storage"].hi.copy(), sheet_charge=value["sheet_charge_C_m2"].hi.copy())
    operator, previous = operator.rebase(seed)
    geometry = problem(mat, baseline.stack, previous.n, previous.p, previous.phi, 0)[0]

    def run(coordinate, *, active=operator, prior=previous):
        return evaluate(coordinate, active=active, prior=prior)[0]

    return operator, previous, run, geometry


def _decimal_physical_rows(system, state, geometry):
    """Independent scalar equations, without either implementation helper."""
    value, mat = state.input_lift, system.material
    phi = [_represented(value["phi_V"][k]) for k in range(4)]
    charge = _decimal(Q)
    sheet = -charge*_decimal(system.trap_density[0])*(
        _represented(value["occupancy"][0])-_decimal(system.equilibrium_occupancy[0]))
    wl, wr = map(_decimal, system._sheet_weights(0))
    poisson = []
    for k, weight in ((1, wl), (2, wr)):
        density = (_represented(value["p_m3"][k])-_represented(value["n_m3"][k])
                   +_decimal(mat.N_D[k])-_decimal(mat.N_A[k])
                   +_represented(value["positive_m3"][k])-_decimal(mat.P_ion0[k]))
        poisson.append(_decimal(mat.poisson_factor.C[k])*(phi[k+1]-phi[k])
            -_decimal(mat.poisson_factor.C[k-1])*(phi[k]-phi[k-1])
            +charge*density*_decimal(mat.poisson_factor.h_cell[k-1])+weight*sheet)
    trace = [_represented(value["trace_potential_V"][0, k]) for k in (0, 1)]
    cl = _decimal(EPS_0*geometry.eps_r_left/geometry.left_distance_m)
    cr = _decimal(EPS_0*geometry.eps_r_right/geometry.right_distance_m)
    local_rows = [trace[1]-trace[0]-_decimal(geometry.potential_jump_right_minus_left_V),
        cl*(trace[0]-phi[1])+cr*(trace[1]-phi[2])
        -_decimal(geometry.fixed_sheet_charge_C_m2)-sheet]
    return poisson, local_rows


def _assert_physical_rows(system, state, geometry):
    with localcontext() as context:
        context.prec = 90
        poisson, local_rows = _decimal_physical_rows(system, state, geometry)
        for actual, expected in zip(state.input_lift["poisson_residual_C_m2"], poisson):
            assert abs(_represented(actual)-expected) < Decimal("1e-29")
        for actual, expected in zip(state.input_lift["local_residual"][:2], local_rows):
            assert abs(_represented(actual)-expected) < Decimal("1e-29")


def test_reference_reconstructs_nonzero_physics_and_retains_historical_words(asymmetric, monkeypatch):
    system, previous, run, geometry = asymmetric
    words = lift._words(previous.input_lift)
    monkeypatch.setattr(ControlledPhysicalInterfaceIonSystem, "evaluate",
                        lambda *args: pytest.fail("physical reference entered a nonlinear evaluation"))
    reference = lift.physical_electrostatic_reference(system, previous)
    assert system._input_lift_electrostatic_cache is None
    assert np.max(np.abs(reference["poisson_anchor_correction_C_m2"].hi)) > .1
    assert np.max(np.abs(reference["local_anchor_correction"].hi)) > .1
    for actual, saved in ((reference["historical_poisson_residual_C_m2"], previous.input_lift["poisson_residual_C_m2"]),
                          (reference["historical_local_electrostatic_residual"], previous.input_lift["local_residual"][:2])):
        assert actual.hi.tobytes() == saved.hi.tobytes()
        assert actual.lo.tobytes() == saved.lo.tobytes()
    state = run(np.zeros(15))
    _assert_physical_rows(system, state, geometry)
    assert lift._words(previous.input_lift) == words
    np.testing.assert_array_equal(previous.poisson_residual, [.125, -.25])
    np.testing.assert_array_equal(previous.local_residual[:2], [.2, -.3])


@pytest.mark.parametrize("size", [2.**-61, 1e-3])
def test_corrected_increments_match_absolute_dd_physics_at_new_state(asymmetric, size):
    system, previous, run, geometry = asymmetric
    words = lift._words(previous.input_lift)
    state = run((np.arange(15)-7)*size)
    _assert_physical_rows(system, state, geometry)
    absolute = lift.physical_electrostatic_reference(system, state)
    assert np.max(np.abs((state.input_lift["poisson_residual_C_m2"]-absolute["poisson_residual_C_m2"]).hi)) < 1e-29
    assert np.max(np.abs((state.input_lift["local_residual"][:2]-absolute["local_electrostatic_residual"]).hi)) < 1e-29
    assert lift._words(previous.input_lift) == words


def test_actual_nonzero_voltage_lift_is_retained_in_all_electrostatic_rows(asymmetric, monkeypatch):
    from perovskite_sim.solver import mol
    system, previous, run, geometry = asymmetric
    target = previous.phi[-1]+.13
    monkeypatch.setattr(mol, "poisson_right_boundary", lambda *args: target)
    high_views = ControlledPhysicalInterfaceIonSystem._coordinates

    def with_contact(self, coordinate, voltage):
        values = list(high_views(self, coordinate, voltage))
        values[2][-1] = target
        return tuple(values)

    monkeypatch.setattr(ControlledPhysicalInterfaceIonSystem, "_coordinates", with_contact)
    system.set_voltage_lift(.13, previous)
    represented_laplacian = lift.diff(DD(system.material.poisson_factor.C)*lift.diff(DD(system._lift)))
    assert np.any(represented_laplacian.hi != 0.)
    state = run(np.zeros(15))
    _assert_physical_rows(system, state, geometry)
    delta_phi = state.input_lift["phi_V"]-previous.input_lift["phi_V"]
    np.testing.assert_allclose(delta_phi.hi, system._lift, rtol=0., atol=1e-30)


def test_reference_cache_tracks_previous_and_coefficients_without_aliasing(asymmetric, monkeypatch):
    system, previous, run, _ = asymmetric
    calls = []
    assemble = lift._assemble_physical_electrostatic_reference

    def counted(*args):
        calls.append(args[1])
        return assemble(*args)

    monkeypatch.setattr(lift, "_assemble_physical_electrostatic_reference", counted)
    original_words = lift._words(previous.input_lift)
    run(np.zeros(15))
    cached = system._input_lift_electrostatic_cache[1]
    run(np.arange(15)*2.**-60)
    assert len(calls) == 1
    system.material.N_D[1] += .25
    accepted = run(np.zeros(15))
    changed = system._input_lift_electrostatic_cache[1]
    assert len(calls) == 2
    assert changed["source_coefficients_identity"] != cached["source_coefficients_identity"]
    assert cached["source_coefficients"]["N_D_m3"].hi[1] == .125
    with pytest.raises(ValueError):
        cached["poisson_residual_C_m2"].hi[0] = 1.
    next_system, next_previous = system.rebase(accepted)
    assert next_system._input_lift_electrostatic_cache is None
    run(np.zeros(15), active=next_system, prior=next_previous)
    assert len(calls) == 3 and calls[-1] is next_previous
    assert lift._words(previous.input_lift) == original_words


def test_canonical_electrostatic_words_repeat_exactly_after_rebase(asymmetric):
    system, _, run, _ = asymmetric
    accepted = run((np.arange(15)-7)*1e-3)
    next_system, next_previous = system.rebase(accepted)
    zero = run(np.zeros(15), active=next_system, prior=next_previous)
    for name in ("poisson_residual_C_m2", "local_residual"):
        assert zero.input_lift[name].hi.tobytes() == accepted.input_lift[name].hi.tobytes()
        assert zero.input_lift[name].lo.tobytes() == accepted.input_lift[name].lo.tobytes()


def test_reference_fails_closed_when_required_static_data_are_missing(asymmetric):
    system, previous, _, _ = asymmetric
    del system.material.N_D
    with pytest.raises(ValueError, match="complete material and interface geometry"):
        lift.physical_electrostatic_reference(system, previous)


@pytest.mark.parametrize("field", ["fixed_sheet_charge_C_m2", "potential_jump_right_minus_left_V"])
def test_reference_rejects_unsupported_static_sheet_or_jump(asymmetric, monkeypatch, field):
    system, previous, _, _ = asymmetric
    original_problem = local._material_two_sided_interface_problem

    def changed(*args, **kwargs):
        geometry, physics, bulk = original_problem(*args, **kwargs)
        return replace(geometry, **{field: .125}), physics, bulk

    monkeypatch.setattr(local, "_material_two_sided_interface_problem", changed)
    with pytest.raises(ValueError, match="zero-static-sheet, zero-jump"):
        lift.physical_electrostatic_reference(system, previous)
