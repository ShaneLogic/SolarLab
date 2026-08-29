"""D6-E3b shared-interface-defect/mobile-ion transient evidence."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from perovskite_sim.constants import Q
from perovskite_sim.experiments.interface_defect_ion_transient import (
    INTERFACE_DEFECT_ION_TRANSIENT_SCOPE,
    InterfaceDefectIonTransientCertificationError,
    InterfaceDefectIonTransientError,
    InterfaceDefectIonTransientPolicy,
    run_interface_defect_ion_device_transient,
)
from perovskite_sim.physics.generation import dual_cell_widths
from tests.integration.test_defect_ion_combined_impedance import (
    _contact_consistent_interface_stack,
    _interface_grid,
    _with_mobile_ions,
)


TIMES_S = np.array([0.0, 1.0e-8, 1.0e-6, 1.0e-4])
VOLTAGE_V = np.array([0.0, 0.01, 0.01, 0.01])


def _stack(*, positive=True, negative=False, diffusivity=1.0e-14):
    stack = _with_mobile_ions(
        _contact_consistent_interface_stack(),
        positive=positive,
        negative=negative,
    )
    return replace(
        stack,
        layers=tuple(
            replace(
                layer,
                params=replace(
                    layer.params,
                    D_ion=diffusivity if positive else 0.0,
                    D_ion_neg=diffusivity if negative else 0.0,
                ),
            )
            for layer in stack.layers
        ),
    )


def _run(*, positive=True, negative=False, diffusivity=1.0e-14, **kwargs):
    stack = _stack(
        positive=positive,
        negative=negative,
        diffusivity=diffusivity,
    )
    return run_interface_defect_ion_device_transient(
        _interface_grid(stack),
        stack,
        TIMES_S,
        VOLTAGE_V,
        **kwargs,
    )


@pytest.fixture(scope="module")
def certified_result():
    return _run()


def test_positive_ion_and_shared_interface_trap_certify_one_sparse_dae(
    certified_result,
):
    result = certified_result
    certificate = result.certificate

    assert result.scope == INTERFACE_DEFECT_ION_TRANSIENT_SCOPE
    assert certificate.certified
    assert certificate.reasons == ()
    assert certificate.dc_operating_point_certified
    assert certificate.dark_reference_certified
    assert certificate.microscopic_binding_certified
    assert certificate.sparse_linear_solver_used
    assert not certificate.clipping_used
    assert certificate.near_acceptance_nonmonotone_step_count == 0
    assert certificate.analytic_jacobian_nnz < certificate.dense_jacobian_entries
    assert certificate.maximum_analytic_jacobian_column_relative_error < 1.0e-5
    assert certificate.maximum_charge_balance_relative_error < 1.0e-10
    assert certificate.maximum_all_face_current_spread_relative < 2.0e-6
    assert certificate.maximum_two_sided_interface_total_current_relative_error < 2.0e-6
    assert certificate.maximum_eliminated_operator_relative_error < 1.0e-6
    assert certificate.maximum_ion_inventory_relative_drift < 1.0e-12
    assert certificate.maximum_current_decomposition_relative_error == 0.0
    assert (
        np.max(
            np.abs(
                result.positive_ion_density_m3[-1] / result.positive_ion_density_m3[0]
                - 1.0
            )
        )
        > 1.0e-4
    )
    assert (
        np.max(np.abs(result.interface_occupancy[-1] - result.interface_occupancy[0]))
        > 1.0e-9
    )
    assert np.any(result.newton_iterations[1:] > 0)


def test_interface_charge_and_current_decompositions_are_exact(certified_result):
    result = certified_result
    reference = np.asarray(result.dark_reference.equilibrium_occupancy)
    density = np.asarray(result.dark_reference.trap_density_m2)

    np.testing.assert_array_equal(
        result.interface_sheet_charge_C_m2,
        -Q * density[None, :] * (result.interface_occupancy - reference[None, :]),
    )
    np.testing.assert_array_equal(
        result.conduction_current_faces_A_m2,
        result.carrier_conduction_current_faces_A_m2
        + result.positive_ion_current_faces_A_m2,
    )
    np.testing.assert_array_equal(
        result.total_current_faces_A_m2,
        result.conduction_current_faces_A_m2 + result.displacement_current_faces_A_m2,
    )
    np.testing.assert_array_equal(
        result.interface_total_current_A_m2,
        result.interface_conduction_current_A_m2
        + result.interface_displacement_current_A_m2,
    )
    np.testing.assert_allclose(
        result.interface_total_current_A_m2[:, :, 0],
        result.interface_total_current_A_m2[:, :, 1],
        rtol=2.0e-6,
        atol=3.0e-15,
    )
    assert np.all(result.displacement_current_faces_A_m2[0] == 0.0)


def test_dynamic_charge_uses_same_interior_control_volume_as_terminal_faces(
    certified_result,
):
    result = certified_result
    stack = _stack()
    grid = _interface_grid(stack)
    widths = dual_cell_widths(grid)
    carrier_increment = Q * np.sum(
        (
            result.hole_density_m3[:, 1:-1]
            - result.hole_density_m3[0, 1:-1]
            - result.electron_density_m3[:, 1:-1]
            + result.electron_density_m3[0, 1:-1]
        )
        * widths[None, 1:-1],
        axis=1,
    )
    sheet_increment = np.sum(
        result.interface_sheet_charge_C_m2 - result.interface_sheet_charge_C_m2[0],
        axis=1,
    )
    ion_increment = Q * np.sum(
        (
            result.positive_ion_density_m3[:, 1:-1]
            - result.positive_ion_density_m3[0, 1:-1]
        )
        * widths[None, 1:-1],
        axis=1,
    )
    expected_increment = carrier_increment + sheet_increment + ion_increment
    measured_increment = (
        result.integrated_free_interface_ion_charge_C_m2
        - result.integrated_free_interface_ion_charge_C_m2[0]
    )
    physical_array_roundoff = (
        8.0
        * np.finfo(float).eps
        * float(
            np.max(
                np.sum(
                    Q
                    * (
                        np.abs(result.hole_density_m3[:, 1:-1])
                        + np.abs(result.electron_density_m3[:, 1:-1])
                        + np.abs(result.positive_ion_density_m3[:, 1:-1])
                    )
                    * widths[None, 1:-1],
                    axis=1,
                )
                + np.sum(np.abs(result.interface_sheet_charge_C_m2), axis=1)
            )
        )
    )

    np.testing.assert_allclose(
        measured_increment,
        expected_increment,
        rtol=2.0e-8,
        # Public density arrays have already rounded away sub-ULP coordinate
        # motion; the certified charge trace retains it through stable increments.
        atol=physical_array_roundoff,
    )


@pytest.mark.parametrize(
    ("positive", "negative"),
    ((True, True), (False, True)),
)
def test_dual_and_negative_only_layouts_keep_separate_ion_evidence(
    positive,
    negative,
):
    result = _run(positive=positive, negative=negative)
    certificate = result.certificate

    assert certificate.certified
    assert bool(result.ion_layout.positive_nodes) is positive
    assert bool(result.ion_layout.negative_nodes) is negative
    assert result.negative_ion_density_m3 is not None
    assert result.negative_ion_current_faces_A_m2 is not None
    assert result.negative_ion_component_inventory_m2 is not None
    assert certificate.maximum_positive_ion_inventory_relative_drift < 1.0e-12
    assert certificate.maximum_negative_ion_inventory_relative_drift < 1.0e-12
    expected = result.carrier_conduction_current_faces_A_m2.copy()
    expected += result.positive_ion_current_faces_A_m2
    expected += result.negative_ion_current_faces_A_m2
    np.testing.assert_array_equal(result.conduction_current_faces_A_m2, expected)
    if not positive:
        np.testing.assert_array_equal(
            result.positive_ion_current_faces_A_m2,
            np.zeros_like(result.positive_ion_current_faces_A_m2),
        )
        assert result.positive_ion_component_inventory_m2.shape[1] == 0


def test_slow_ion_limit_freezes_ions_without_freezing_interface_traps():
    voltage = np.array([0.0, 0.05, 0.05, 0.05])
    slow_stack = _stack(diffusivity=1.0e-20)
    reference_stack = _stack(diffusivity=1.0e-14)
    slow = run_interface_defect_ion_device_transient(
        _interface_grid(slow_stack), slow_stack, TIMES_S, voltage
    )
    reference = run_interface_defect_ion_device_transient(
        _interface_grid(reference_stack), reference_stack, TIMES_S, voltage
    )
    slow_motion = np.max(
        np.abs(slow.positive_ion_density_m3[-1] / slow.positive_ion_density_m3[0] - 1.0)
    )
    reference_motion = np.max(
        np.abs(
            reference.positive_ion_density_m3[-1] / reference.positive_ion_density_m3[0]
            - 1.0
        )
    )

    assert slow.certificate.certified
    assert reference.certificate.certified
    assert slow_motion < 1.0e-8
    assert reference_motion > 1.0e-4
    assert (
        np.max(np.abs(slow.interface_occupancy[-1] - slow.interface_occupancy[0]))
        > 1.0e-9
    )


def test_deep_time_refinement_retains_charge_and_all_face_current_closure():
    stack = _stack()
    policy = replace(
        InterfaceDefectIonTransientPolicy(),
        refinement_substeps=(4, 8, 16),
    )

    result = run_interface_defect_ion_device_transient(
        _interface_grid(stack),
        stack,
        TIMES_S,
        np.array([0.0, 0.05, 0.05, 0.05]),
        policy=policy,
    )

    assert result.certificate.certified
    assert result.certificate.maximum_charge_balance_relative_error < 1.0e-10
    assert result.certificate.maximum_all_face_current_spread_relative < 2.0e-6
    assert result.certificate.maximum_refinement_state_change < 2.0e-5
    assert result.certificate.maximum_refinement_current_relative_change < 1.0e-7


def test_overstrict_interface_current_gate_fails_closed():
    policy = InterfaceDefectIonTransientPolicy(
        maximum_two_sided_interface_total_current_relative_error=1.0e-8,
    )

    with pytest.raises(InterfaceDefectIonTransientCertificationError) as exc_info:
        _run(policy=policy)

    assert exc_info.value.result.certificate.reasons == (
        "two_sided_interface_total_current_closure_failed",
    )


def test_pre_clipping_site_ceiling_fails_closed():
    policy = InterfaceDefectIonTransientPolicy(site_occupancy_ceiling=0.8)

    with pytest.raises(InterfaceDefectIonTransientError, match="site ceiling"):
        _run(policy=policy)


def test_result_arrays_are_immutable(certified_result):
    result = certified_result

    for values in (
        result.electron_density_m3,
        result.interface_occupancy,
        result.positive_ion_density_m3,
        result.positive_ion_component_inventory_m2,
        result.interface_total_current_A_m2,
        result.total_current_faces_A_m2,
    ):
        assert not values.flags.writeable


def test_policy_rejects_invalid_ion_controls():
    with pytest.raises(ValueError, match="ion_storage_atol_m3"):
        InterfaceDefectIonTransientPolicy(ion_storage_atol_m3=0.0)
    with pytest.raises(ValueError, match="site_occupancy_ceiling"):
        InterfaceDefectIonTransientPolicy(site_occupancy_ceiling=1.0)
    with pytest.raises(ValueError, match="dc_max_nfev"):
        InterfaceDefectIonTransientPolicy(dc_max_nfev=0)


@pytest.mark.parametrize(
    ("case", "message"),
    (
        ("missing_ions", "mobile-ion species"),
        ("missing_document", "microscopic interface"),
    ),
)
def test_missing_joint_physics_fails_before_integration(case, message):
    stack = _contact_consistent_interface_stack()
    if case == "missing_document":
        stack = replace(_stack(), interface_defects=())

    with pytest.raises(InterfaceDefectIonTransientError, match=message):
        run_interface_defect_ion_device_transient(
            _interface_grid(stack),
            stack,
            TIMES_S,
            VOLTAGE_V,
        )
