"""A-D rate controls preserve population, charge, and constitutive closure."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from perovskite_sim.experiments.defect_ion_combined_impedance import _build_ion_layout
from perovskite_sim.experiments.interface_defect_ion_transient import (
    InterfaceDefectIonTransientPolicy,
    InterfaceIonDarkReference,
)
from perovskite_sim.experiments.interface_defect_transient import (
    _integrate_trace,
    _jacobian_error,
)
from perovskite_sim.experiments.one_dimensional_mechanism_r1 import solve_r1_dc
from perovskite_sim.experiments.one_dimensional_mechanism_r1_dynamics import (
    ControlledPhysicalInterfaceIonSystem,
    R1DynamicsControls,
)
from perovskite_sim.experiments.quasi_fermi_steady_state import (
    _QuasiFermiSystem,
    _research_charge_off_stack,
)
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.physics.interface_plane import FERMI_DIRAC_RICHARDSON
from perovskite_sim.physics.two_sided_interface import (
    TWO_SIDED_TRACE,
    fixed_occupancy_trap_capture_flux_and_log_jacobian,
)
from tests.integration.test_one_dimensional_mechanism_r1 import FIXTURE, binding
from tests.unit.physics.test_two_sided_interface import _physics


@pytest.mark.parametrize("label,pair", [("A", (0, 0)), ("B", (1, 0)), ("C", (0, 1)), ("D", (1, 1))])
def test_control_identity_is_explicit_and_immutable(label, pair):
    controls = R1DynamicsControls.from_label(label)
    assert (controls.nu_I, controls.nu_t) == pair
    assert controls.label == label
    with pytest.raises(FrozenInstanceError):
        controls.nu_I = 1


@pytest.mark.parametrize("value", [-1, 2, 0.5, 1.0, True, "1", None])
def test_controls_reject_nonbinary_or_noninteger_values(value):
    with pytest.raises((TypeError, ValueError)):
        R1DynamicsControls(nu_I=value)
    with pytest.raises((TypeError, ValueError)):
        R1DynamicsControls(nu_t=value)


@pytest.fixture(scope="module")
def prepared_operator_arguments():
    stack = load_device_from_yaml(FIXTURE)
    fixed_reference = binding(stack)
    grid, material, dc, _ = solve_r1_dc(stack, 4, np.asarray(fixed_reference["f_ref"]))
    _, _, dark_dc, _ = solve_r1_dc(stack, 4)
    charge_off, microscopic = _research_charge_off_stack(stack)
    reference = InterfaceIonDarkReference(
        equilibrium_occupancy=np.asarray(fixed_reference["f_ref"]),
        trap_density_m2=np.asarray(microscopic.trap_density_m2),
        capture_velocities_m_s=np.asarray(microscopic.capture_velocities_m_s),
        interface_defect_document_sha256=microscopic.document_sha256,
        interface_transmission=1.0,
        dc_state=dark_dc,
    )
    qf = _QuasiFermiSystem(
        grid, charge_off, material, 0.0,
        interface_boundary=True,
        interface_topology=TWO_SIDED_TRACE,
        interface_transmission=1.0,
        interface_transport_model=FERMI_DIRAC_RICHARDSON,
        interface_charge_reference_occupancy=reference.equilibrium_occupancy,
        interface_charge_trap_density_m2=reference.trap_density_m2,
        poisson_tolerance_V=1e-13,
        poisson_max_iterations=100,
    )
    occupancy = np.asarray(dc.interface_occupancy)
    dynamic = qf.evaluate_quasi_fermi_increments_defect_ion_combined(
        dc.electron_qf_increment_V, dc.hole_qf_increment_V, 0.0,
        positive_ion_density_m3=dc.positive_ion_density_m3,
        dynamic_interface_occupancy=occupancy,
        V_app=0.0,
    )
    return (grid, charge_off, material, dc, qf, reference, occupancy, dynamic, _build_ion_layout(material))


def _system(arguments, label):
    return ControlledPhysicalInterfaceIonSystem(
        *arguments, controls=R1DynamicsControls.from_label(label), voltage=0.0,
        illuminated=False, site_occupancy_ceiling=0.999,
    )


@pytest.mark.parametrize("label", ["A", "B", "C", "D"])
def test_controls_preserve_storage_charge_and_zero_disabled_rates(prepared_operator_arguments, label):
    system = _system(prepared_operator_arguments, label)
    full = _system(prepared_operator_arguments, "D")
    coordinate = np.zeros(system.dimension)
    coordinate[system.trap_slice] = 0.1
    coordinate[system.positive_slice] = np.linspace(-0.01, 0.01, system.positive_nodes.size)
    coordinate[system.local_slice] = np.linspace(-0.02, 0.02, 6 * system.interface_count)
    state = system.evaluate(coordinate, 0.0)
    full_state = full.evaluate(coordinate, 0.0)
    np.testing.assert_array_equal(state.storage, full_state.storage)
    np.testing.assert_array_equal(state.sheet_charge, full_state.sheet_charge)
    assert np.any(state.sheet_charge != 0.0)
    np.testing.assert_array_equal(state.storage_jacobian.toarray(), full_state.storage_jacobian.toarray())
    np.testing.assert_array_equal(state.poisson_jacobian.toarray(), full_state.poisson_jacobian.toarray())
    np.testing.assert_array_equal(system.material.P_ion0, full.material.P_ion0)
    assert system.material is full.material
    if not system.controls.nu_I:
        for value in (state.positive_rate, state.positive_flux, state.positive_current):
            np.testing.assert_array_equal(value, 0.0)
        np.testing.assert_array_equal(state.rate_jacobian[system.positive_slice].toarray(), 0.0)
    if not system.controls.nu_t:
        np.testing.assert_array_equal(state.rate[system.trap_slice], 0.0)
        np.testing.assert_array_equal(state.rate_jacobian[system.trap_slice].toarray(), 0.0)
        for local in state.local:
            np.testing.assert_array_equal(local.tangent.balance.capture_flux_m2_s, 0.0)
            np.testing.assert_array_equal(local.tangent.capture_flux_jacobian_log_state_m2_s, 0.0)
            np.testing.assert_array_equal(local.tangent.capture_flux_occupancy_derivative_m2_s, 0.0)
            assert np.any(local.tangent.balance.cross_flux_m2_s != 0.0)
    rebased, previous = system.rebase(state)
    repeated = rebased.evaluate(np.zeros(system.dimension), 0.0)
    assert rebased.controls == system.controls
    np.testing.assert_array_equal(repeated.occupancy, state.occupancy)
    np.testing.assert_array_equal(repeated.positive, state.positive)


@pytest.mark.parametrize("label", ["A", "B", "C", "D"])
def test_controlled_joint_jacobian_and_eliminated_operator(prepared_operator_arguments, label):
    system = _system(prepared_operator_arguments, label)
    initial = system.evaluate(np.zeros(system.dimension), 0.0)
    assert system.eliminated_operator_error(initial, 0.0) <= 1e-5
    working, previous = system.rebase(initial)
    coordinate = np.zeros(system.dimension)
    coordinate[system.electron_slice] = 1e-3
    coordinate[system.hole_slice] = -2e-3
    coordinate[system.trap_slice] = 3e-3
    coordinate[system.positive_slice] = np.linspace(-1e-3, 1e-3, system.positive_nodes.size)
    policy = InterfaceDefectIonTransientPolicy()
    dt = 1e-7
    storage_scale = system.storage_scale(previous.storage, previous, dt, policy)
    poisson_scale = system.poisson_scale(policy)
    local_scale = system.local_algebraic_scale(policy)
    _, jacobian, _ = working.residual_and_jacobian(
        coordinate, 0.0, previous, dt, storage_scale, poisson_scale, local_scale,
    )
    error = _jacobian_error(
        working, coordinate, 0.0, previous, dt,
        storage_scale, poisson_scale, local_scale, jacobian, 1e-5,
    )
    assert error <= policy.maximum_jacobian_column_relative_error


def test_imported_initial_state_and_regular_currents_are_not_recomputed(prepared_operator_arguments, monkeypatch):
    system = _system(prepared_operator_arguments, "D")
    initial = system.evaluate(np.zeros(system.dimension), 0.0)
    def forbidden(*args, **kwargs):
        raise AssertionError("imported initial state must not be recomputed")
    monkeypatch.setattr(system, "evaluate", forbidden)
    faces = system.node_count - 1
    initial_metrics = (
        np.full(faces, 0.5), np.full(faces, 0.75),
        np.full((1, 2), 0.25), np.full((1, 2), 0.5), 1e-7, 2e-7,
    )
    trace = _integrate_trace(
        system, np.array([0.0]), np.array([0.0]), 1, InterfaceDefectIonTransientPolicy(),
        initial_state=initial, initial_current_metrics=initial_metrics,
    )
    assert trace.states[0].n is initial.n
    np.testing.assert_array_equal(trace.coordinates[0], 0.0)
    np.testing.assert_array_equal(trace.displacement[0], initial_metrics[0])
    np.testing.assert_array_equal(trace.total_current[0], initial_metrics[1])
    np.testing.assert_array_equal(trace.interface_total_current[0], 0.75)
    assert trace.maximum_face_spread == 1e-7
    assert trace.maximum_interface_current_error == 2e-7


@pytest.mark.parametrize("voltage", [0.005, -0.005, 0.00015625])
def test_voltage_lift_keeps_pinned_endpoint_populations_exact(prepared_operator_arguments, voltage):
    system = _system(prepared_operator_arguments, "D")
    initial = system.evaluate(np.zeros(system.dimension), 0.0)
    working, previous = system.rebase(initial)
    working.set_voltage_lift(voltage, previous)
    lifted = working.evaluate(np.zeros(system.dimension), voltage)
    np.testing.assert_array_equal(lifted.n, previous.n)
    np.testing.assert_array_equal(lifted.p, previous.p)
    np.testing.assert_array_equal(lifted.n[[0, -1]], system.reference_n[[0, -1]])
    np.testing.assert_array_equal(lifted.p[[0, -1]], system.reference_p[[0, -1]])


@pytest.mark.parametrize("multiplier", [0.0, 1.0])
def test_fixed_reservoir_trap_matches_exponential_relaxation(multiplier):
    physics = _physics(
        surface_recombination_velocity_n_m_s=0.3,
        surface_recombination_velocity_p_m_s=0.7,
        n1_left_m3=2e10, n1_right_m3=3e10,
        p1_left_m3=4e10, p1_right_m3=5e10,
    )
    densities = np.array([6e10, 7e10, 8e10, 9e10])
    population = 1e12
    k = (0.3 * (6e10 + 8e10 + 2e10 + 3e10) + 0.7 * (7e10 + 9e10 + 4e10 + 5e10)) / population
    equilibrium = (0.3 * (6e10 + 8e10) + 0.7 * (4e10 + 5e10)) / (population * k)
    f0 = 0.23
    times = np.linspace(0.0, 5.0 / k, 31)
    def rate(time, occupancy):
        capture, _ = fixed_occupancy_trap_capture_flux_and_log_jacobian(
            densities, physics, occupancy[0], capture_multiplier=multiplier,
        )
        return [(capture[0] + capture[2] - capture[1] - capture[3]) / population]
    result = solve_ivp(rate, (times[0], times[-1]), [f0], t_eval=times, rtol=1e-10, atol=1e-12)
    expected = equilibrium + (f0 - equilibrium) * np.exp(-multiplier * k * times)
    assert result.success
    np.testing.assert_allclose(result.y[0], expected, rtol=1e-9, atol=1e-11)
