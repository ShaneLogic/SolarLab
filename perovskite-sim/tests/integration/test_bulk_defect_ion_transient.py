"""D6-E3a bulk-defect/mobile-ion transient integration evidence."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from perovskite_sim.constants import Q
from perovskite_sim.experiments.bulk_defect_ion_transient import (
    BULK_DEFECT_ION_TRANSIENT_SCOPE,
    BulkDefectIonTransientCertificationError,
    BulkDefectIonTransientError,
    BulkDefectIonTransientPolicy,
    run_bulk_defect_ion_device_transient,
)
from perovskite_sim.models.defects import EFFECTIVE_LIFETIME
from perovskite_sim.physics.generation import dual_cell_widths
from tests.integration.test_defect_ion_combined_impedance import (
    _bulk_defect_stack,
    _bulk_grid,
    _contact_consistent_interface_stack,
    _interface_grid,
    _with_mobile_ions,
)


TIMES_S = np.array([0.0, 1.0e-8, 1.0e-6, 1.0e-4])
VOLTAGE_V = np.array([0.0, 1.0e-3, 1.0e-3, 1.0e-3])


def _stack(*, positive=True, negative=False, diffusivity=1.0e-14):
    stack = _with_mobile_ions(
        _bulk_defect_stack(),
        positive=positive,
        negative=negative,
    )
    layers = tuple(
        replace(
            layer,
            params=replace(
                layer.params,
                D_ion=diffusivity if positive else 0.0,
                D_ion_neg=diffusivity if negative else 0.0,
            ),
        )
        for layer in stack.layers
    )
    return replace(stack, layers=layers)


def _run(*, positive=True, negative=False, diffusivity=1.0e-14, **kwargs):
    stack = _stack(
        positive=positive,
        negative=negative,
        diffusivity=diffusivity,
    )
    return run_bulk_defect_ion_device_transient(
        _bulk_grid(stack, 4),
        stack,
        TIMES_S,
        VOLTAGE_V,
        **kwargs,
    )


def test_positive_ion_and_bulk_trap_share_one_certified_sparse_dae():
    result = _run()
    certificate = result.certificate

    assert result.scope == BULK_DEFECT_ION_TRANSIENT_SCOPE
    assert certificate.certified
    assert certificate.reasons == ()
    assert certificate.dc_operating_point_certified
    assert certificate.sparse_linear_solver_used
    assert not certificate.clipping_used
    assert certificate.analytic_jacobian_nnz < certificate.dense_jacobian_entries
    assert certificate.maximum_analytic_jacobian_column_relative_error < 1.0e-6
    assert certificate.maximum_ion_inventory_relative_drift < 1.0e-12
    assert certificate.maximum_charge_balance_relative_error < 1.0e-10
    assert certificate.maximum_all_face_current_spread_relative < 1.0e-9
    assert certificate.maximum_current_decomposition_relative_error == 0.0
    assert dict(certificate.eliminated_operator_components)["positive_ion_rate"] < (
        1.0e-10
    )
    assert (
        np.max(
            np.abs(
                result.positive_ion_density_m3[-1] / result.positive_ion_density_m3[0]
                - 1.0
            )
        )
        > 1.0e-6
    )
    assert np.max(np.abs(result.trap_occupancy[-1] - result.trap_occupancy[0])) > 0.0
    np.testing.assert_array_equal(
        result.conduction_current_faces_A_m2,
        result.carrier_conduction_current_faces_A_m2
        + result.positive_ion_current_faces_A_m2,
    )
    np.testing.assert_array_equal(
        result.total_current_faces_A_m2,
        result.conduction_current_faces_A_m2 + result.displacement_current_faces_A_m2,
    )
    assert np.all(result.displacement_current_faces_A_m2[0] == 0.0)
    assert np.any(result.newton_iterations[1:] > 0)


def test_bulk_terminal_charge_uses_the_interior_face_control_volume():
    stack = _stack()
    grid = _bulk_grid(stack, 4)
    result = run_bulk_defect_ion_device_transient(
        grid,
        stack,
        TIMES_S,
        VOLTAGE_V,
    )
    widths = dual_cell_widths(grid)
    carrier_trap_charge = (
        Q * (result.hole_density_m3 - result.electron_density_m3)
        + result.trap_charge_density_C_m3
    )
    bulk_increment = np.sum(
        (carrier_trap_charge[:, 1:-1] - carrier_trap_charge[0, 1:-1])
        * widths[None, 1:-1],
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
    measured_increment = (
        result.integrated_free_trap_ion_charge_C_m2
        - result.integrated_free_trap_ion_charge_C_m2[0]
    )

    np.testing.assert_allclose(
        measured_increment,
        bulk_increment + ion_increment,
        rtol=2.0e-8,
        atol=1.0e-20,
    )


@pytest.mark.parametrize(
    ("positive", "negative"),
    ((True, True), (False, True)),
)
def test_dual_and_negative_only_ion_layouts_keep_separate_inventories(
    positive,
    negative,
):
    result = _run(positive=positive, negative=negative)

    assert result.certificate.certified
    assert bool(result.ion_layout.positive_nodes) is positive
    assert bool(result.ion_layout.negative_nodes) is negative
    assert result.negative_ion_density_m3 is not None
    assert result.negative_ion_current_faces_A_m2 is not None
    assert result.negative_ion_component_inventory_m2 is not None
    assert result.certificate.maximum_positive_ion_inventory_relative_drift < 1.0e-12
    assert result.certificate.maximum_negative_ion_inventory_relative_drift < 1.0e-12
    expected = (
        result.carrier_conduction_current_faces_A_m2
        + result.positive_ion_current_faces_A_m2
        + result.negative_ion_current_faces_A_m2
    )
    np.testing.assert_array_equal(result.conduction_current_faces_A_m2, expected)
    if not positive:
        np.testing.assert_array_equal(
            result.positive_ion_current_faces_A_m2,
            np.zeros_like(result.positive_ion_current_faces_A_m2),
        )
        assert result.positive_ion_component_inventory_m2.shape[1] == 0


def test_slow_ion_limit_is_frozen_while_trap_and_carriers_remain_coupled():
    slow = _run(diffusivity=1.0e-20)
    reference = _run(diffusivity=1.0e-14)

    slow_motion = np.max(
        np.abs(slow.positive_ion_density_m3[-1] / slow.positive_ion_density_m3[0] - 1.0)
    )
    reference_motion = np.max(
        np.abs(
            reference.positive_ion_density_m3[-1] / reference.positive_ion_density_m3[0]
            - 1.0
        )
    )
    assert slow_motion < 1.0e-9
    assert reference_motion > 1.0e-6
    assert slow.certificate.certified
    assert reference.certificate.certified


def test_overstrict_inventory_gate_returns_partial_or_fails_closed():
    policy = BulkDefectIonTransientPolicy(
        maximum_ion_inventory_relative_drift=1.0e-17,
    )
    partial = _run(policy=policy, require_certificate=False)

    assert not partial.certificate.certified
    assert partial.certificate.reasons == ("ion_inventory_drift_exceeds_limit",)
    with pytest.raises(BulkDefectIonTransientCertificationError) as exc_info:
        _run(policy=policy)
    assert exc_info.value.result.certificate.reasons == partial.certificate.reasons


def test_pre_clipping_site_ceiling_fails_closed():
    policy = BulkDefectIonTransientPolicy(site_occupancy_ceiling=0.4)

    with pytest.raises(BulkDefectIonTransientError, match="site ceiling"):
        _run(policy=policy)


def test_result_arrays_are_immutable():
    result = _run()

    for values in (
        result.electron_density_m3,
        result.trap_occupancy,
        result.positive_ion_density_m3,
        result.positive_ion_component_inventory_m2,
        result.total_current_faces_A_m2,
    ):
        assert not values.flags.writeable


def test_missing_mobile_ion_fails_before_integration():
    stack = _bulk_defect_stack()

    with pytest.raises(BulkDefectIonTransientError, match="mobile-ion species"):
        run_bulk_defect_ion_device_transient(
            _bulk_grid(stack, 4),
            stack,
            TIMES_S,
            VOLTAGE_V,
        )


def test_missing_explicit_bulk_defect_fails_before_integration():
    stack = _stack()
    layer = stack.layers[0]
    no_defect = replace(
        stack,
        layers=(
            replace(
                layer,
                params=replace(
                    layer.params,
                    defect_schema_version=None,
                    defect_model=EFFECTIVE_LIFETIME,
                    bulk_defects=(),
                ),
            ),
        ),
    )

    with pytest.raises(BulkDefectIonTransientError, match="explicit bulk defect"):
        run_bulk_defect_ion_device_transient(
            _bulk_grid(no_defect, 4),
            no_defect,
            TIMES_S,
            VOLTAGE_V,
        )


def test_interface_defect_is_reserved_for_e3b():
    stack = _with_mobile_ions(_contact_consistent_interface_stack())

    with pytest.raises(BulkDefectIonTransientError, match="E3b"):
        run_bulk_defect_ion_device_transient(
            _interface_grid(stack),
            stack,
            TIMES_S,
            VOLTAGE_V,
        )


@pytest.mark.parametrize(
    ("updates", "message"),
    (
        ({"ion_storage_atol_m3": 0.0}, "finite and positive"),
        ({"site_occupancy_ceiling": 0.999999}, "below the ion-flux clip"),
        ({"dc_max_nfev": 0}, "must be positive"),
    ),
)
def test_policy_rejects_incomplete_joint_contract(updates, message):
    with pytest.raises(ValueError, match=message):
        BulkDefectIonTransientPolicy(**updates)
