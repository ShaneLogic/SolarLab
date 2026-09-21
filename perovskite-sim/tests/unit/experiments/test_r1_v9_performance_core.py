"""Exact-word V9 performance equivalence and constant-cache ownership.

The reference implementation is loaded from the approved ceec959 Git blob
under a test-only module name. No production module is replaced or patched.
There is one common N16 initialization and bounded operator evaluations, no
new trajectory integration or profile run.
"""
import copy
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from perovskite_sim.experiments import one_dimensional_mechanism_r1_precision as current
from perovskite_sim.experiments import one_dimensional_mechanism_r1_state as states
from perovskite_sim.experiments.one_dimensional_mechanism_r1_backend import PAIR
from perovskite_sim.experiments.one_dimensional_mechanism_r1_dynamics import R1DynamicsControls
from perovskite_sim.experiments.one_dimensional_mechanism_r1_protocol import r1_policy
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.physics.compensated import DD

BASELINE = "ceec959a90d219c4ced06485c6d13ee734b685ad"
PROJECT = Path(__file__).resolve().parents[3]


def same_array(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    assert actual.tobytes() == expected.tobytes()


def same_pair(actual, expected):
    same_array(actual.hi, expected.hi)
    same_array(actual.lo, expected.lo)


def same_state(actual, expected):
    assert set(actual.fine) == set(expected.fine)
    for name in actual.fine:
        same_pair(actual.fine[name], expected.fine[name])
    for name in ("coordinate", "storage", "rate", "local_residual", "poisson_residual",
                 "current_n", "current_p", "positive_flux", "positive_rate", "conduction"):
        same_array(getattr(actual, name), getattr(expected, name))


def constant_owner():
    owner = object.__new__(current.CompensatedR1System)
    owner.grid = np.array([0., 1.23456789e-8, 2.34567891e-8, 3.45678912e-8])
    nodes, faces = len(owner.grid), len(owner.grid) - 1
    owner.material = SimpleNamespace(V_T_device=.025851999786,
        P_lim_node=np.full(nodes, 1e27), D_ion_face=np.full(faces, 1.37e-14),
        chi=np.full(nodes, 4.013), Eg=np.full(nodes, 1.61),
        D_n_face=np.array([.0123456789, .0234567891, .0345678912]),
        D_p_face=np.array([.0098765432, .0087654321, .0076543219]),
        N_D=np.full(nodes, 1e14), N_A=np.full(nodes, 1e13), P_ion0=np.full(nodes, 1e22),
        poisson_factor=SimpleNamespace(C=np.full(faces, .0123), h_cell=np.array([1e-8, 1e-8])))
    owner.system = SimpleNamespace(reference_edge_drop_n=np.full(faces, .003),
        reference_edge_drop_p=np.full(faces, .004), phi0=np.linspace(0., .4, nodes),
        log_n0=np.full(nodes, 42.), log_p0=np.full(nodes, 43.))
    owner.thermal_voltage = owner.material.V_T_device
    owner.widths = np.full(nodes, 1e-8)
    owner.trap_density, owner.equilibrium_occupancy = np.array([1e15]), np.array([.3])
    owner.precision_fault = "none"
    owner._fine_constants, owner._fine_evaluation_cache = {}, {}
    return owner


def assert_prefactors(owner):
    constants = owner._refresh_fine_constants()
    expected = tuple(current._CHARGE * DD(getattr(owner.material, name)) / DD(np.diff(owner.grid))
                     for name in ("D_n_face", "D_p_face"))
    for actual, value in zip(owner._fine_carrier_prefactors, expected):
        same_pair(actual, value)
    assert set(constants) == {"vt", "ion_vt", "ion_limit", "ion_diffusion", "spacing", "widths",
        "chi", "Eg", "Dn", "Dp", "edge_n", "edge_p", "ND", "NA", "ion_background", "poisson_C",
        "poisson_h", "trap_density", "equilibrium_occupancy", "phi0", "log_n0", "log_p0"}


def test_prefactors_reuse_only_state_independent_words():
    owner = constant_owner()
    assert_prefactors(owner)
    prefactors = owner._fine_carrier_prefactors
    constants = owner._fine_constants
    assert any(np.any(value.lo != 0.) for value in prefactors)
    owner._fine_work = {"phi_V": DD(np.arange(4.) * 1e-10)}
    assert_prefactors(owner)
    assert owner._fine_carrier_prefactors is prefactors
    assert owner._fine_constants is constants


@pytest.mark.parametrize("change", ("Dn", "Dp", "grid", "material_identity", "system_identity", "fault"))
def test_prefactor_dependencies_invalidate_without_changing_old_owner(change):
    original = constant_owner()
    assert_prefactors(original)
    prefactors = original._fine_carrier_prefactors
    captured = [(value.hi.copy(), value.lo.copy()) for value in prefactors]
    other = copy.copy(original)
    if change in ("Dn", "Dp"):
        other.material = copy.copy(original.material)
        field = "D_n_face" if change == "Dn" else "D_p_face"
        setattr(other.material, field, getattr(original.material, field).copy())
        getattr(other.material, field)[1] *= 1.000000001
    elif change == "grid":
        other.grid = original.grid.copy()
        other.grid[1] *= 1.000000001
    elif change == "material_identity":
        other.material = copy.copy(original.material)
    elif change == "system_identity":
        other.system = copy.copy(original.system)
    else:
        other.precision_fault = "diffusion"
    assert_prefactors(other)
    assert other._fine_carrier_prefactors is not prefactors
    assert original._fine_carrier_prefactors is prefactors
    for value, (hi, lo) in zip(prefactors, captured):
        same_array(value.hi, hi)
        same_array(value.lo, lo)


def test_inplace_diffusion_and_exception_recovery_cannot_leave_stale_prefactors():
    owner = constant_owner()
    assert_prefactors(owner)
    owner.material.D_n_face[0] *= 1.000000001
    assert_prefactors(owner)
    valid = owner.material.D_p_face[0]
    constants, prefactors = owner._fine_constants, owner._fine_carrier_prefactors
    owner.material.D_p_face[0] = np.nan
    with pytest.raises(ArithmeticError):
        owner._refresh_fine_constants()
    assert owner._fine_constants is constants and owner._fine_carrier_prefactors is prefactors
    owner.material.D_p_face[0] = valid
    assert_prefactors(owner)


@pytest.fixture(scope="module")
def actual_systems():
    name = "r1_v9_performance_frozen_precision_reference"
    source = subprocess.run(["git", "show", BASELINE + ":perovskite-sim/perovskite_sim/experiments/one_dimensional_mechanism_r1_precision.py"],
        cwd=PROJECT, check=True, capture_output=True).stdout
    reference = ModuleType(name)
    reference.__package__ = "perovskite_sim.experiments"
    reference.__file__ = "git:" + BASELINE + ":one_dimensional_mechanism_r1_precision.py"
    sys.modules[name] = reference
    try:
        exec(compile(source, reference.__file__, "exec"), reference.__dict__)
        stack = load_device_from_yaml(PROJECT / "tests/fixtures/configs/dynamic_interface_defect_ion_transient_absorber_only.yaml")
        binding = json.loads((PROJECT / "tests/fixtures/OneDimensionalMechanismR1ReferenceBindingV1.json").read_text())
        with threadpool_limits(1):
            prepared = states.prepare_common_state(stack, 16, binding)
            baseline, initial = states.verify_prepared_physics(prepared, stack, binding)
            old, old_state = reference.CompensatedR1System.from_baseline(baseline, initial)
            new, new_state = current.CompensatedR1System.from_baseline(baseline, initial)
            PAIR.bind(old)
            PAIR.bind(new)
        yield stack, binding, reference, old, old_state, new, new_state
    finally:
        sys.modules.pop(name, None)


@pytest.mark.parametrize("voltage", (0., .005, -.005))
def test_actual_evaluation_residual_jacobian_and_independent_solve_match_frozen_words(actual_systems, voltage):
    _, _, reference, old, old_state, new, new_state = actual_systems
    with threadpool_limits(1):
        same_state(new_state, old_state)
        old_work, old_previous = old.rebase(old_state)
        new_work, new_previous = new.rebase(new_state)
        old_work.set_voltage_lift(voltage, old_previous)
        new_work.set_voltage_lift(voltage, new_previous)
        coordinate = np.zeros(new.dimension)
        coordinate[new.electron_slice.start + 2] = 1e-7
        coordinate[new.hole_slice.start + 3] = -1e-7
        coordinate[new.positive_slice.start + 4] = 1e-8
        coordinate[new.potential_slice.start + 5] = -1e-8
        old_value = old_work.evaluate(coordinate, voltage)
        new_value = new_work.evaluate(coordinate, voltage)
        same_state(new_value, old_value)
        scales = (new.storage_scale(new_previous.storage, new_previous, 1e-9, r1_policy()),
                  new.poisson_scale(r1_policy()), new.local_algebraic_scale(r1_policy()))
        old_residual, old_jacobian, old_value = old_work.residual_and_jacobian(
            coordinate, voltage, old_previous, 1e-9, *scales)
        new_residual, new_jacobian, new_value = new_work.residual_and_jacobian(
            coordinate, voltage, new_previous, 1e-9, *scales)
        same_array(new_residual, old_residual)
        for field in ("data", "indices", "indptr"):
            same_array(getattr(new_jacobian, field), getattr(old_jacobian, field))
        same_state(new_value, old_value)
        old_diagnostics = old_work.eliminated_operator_diagnostics(old_value, voltage)
        new_diagnostics = new_work.eliminated_operator_diagnostics(new_value, voltage)
        assert states.canonical(new_diagnostics) == states.canonical(old_diagnostics)
        assert states.canonical(new_diagnostics.precision_evidence) == states.canonical(old_diagnostics.precision_evidence)
        shared = new_diagnostics.precision_evidence["shared_inputs"]["fields"]
        payload = new_diagnostics.precision_evidence["solve_inputs"]["fixed_inputs"]
        assert shared == payload and shared is not payload


def test_actual_control_construction_and_rebase_preserve_constant_cache_ownership(actual_systems):
    stack, binding, _, _, _, system, initial = actual_systems
    anchor = system._fine_carrier_prefactors
    before = [current.pair_words(value) for value in anchor]
    with threadpool_limits(1):
        for label in "ABCD":
            controlled, state = PAIR.controlled(system, initial, stack, binding,
                                                R1DynamicsControls.from_label(label), r1_policy())
            assert controlled._fine_carrier_prefactors is not anchor
            for a, b in zip(controlled._fine_carrier_prefactors, anchor):
                same_pair(a, b)
            working, local = controlled.rebase(state)
            after = working.evaluate(np.zeros(working.dimension), 0.)
            same_state(after, state)
            assert working._fine_carrier_prefactors is controlled._fine_carrier_prefactors
    assert system._fine_carrier_prefactors is anchor
    assert [current.pair_words(value) for value in anchor] == before
