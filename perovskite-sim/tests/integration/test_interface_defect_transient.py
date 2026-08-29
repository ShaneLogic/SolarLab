"""D6-E2 dynamic shared interface occupancy coupled to device transient."""

from __future__ import annotations

from dataclasses import replace
import math

import numpy as np
import pytest

from perovskite_sim.constants import Q
from perovskite_sim.discretization.grid import Layer, multilayer_grid
from perovskite_sim.experiments.interface_defect_transient import (
    INTERFACE_DEFECT_TRANSIENT_SCOPE,
    InterfaceDefectTransientCertificationError,
    InterfaceDefectTransientError,
    InterfaceDefectTransientPolicy,
    run_interface_defect_device_transient,
)
from perovskite_sim.experiments.quasi_fermi_steady_state import (
    build_two_sided_trace_grid,
)
from perovskite_sim.models.device import DeviceStack, InterfaceDefect, LayerSpec
from perovskite_sim.models.interface_defects import InterfaceDefectDocument
from perovskite_sim.models.parameters import MaterialParams
from perovskite_sim.physics.temperature import thermal_voltage


TIMES_S = np.array([0.0, 1.0e-8, 1.0e-6, 1.0e-4])
VOLTAGE_V = np.array([0.0, 0.01, 0.01, 0.01])


def _stack(*, capture_scale: float = 1.0) -> DeviceStack:
    intrinsic_density = math.sqrt(
        1.0e25 * 1.0e25 * math.exp(-1.5 / thermal_voltage(300.0))
    )
    left = MaterialParams(
        eps_r=10.0,
        mu_n=1.0e-3,
        mu_p=1.0e-3,
        D_ion=0.0,
        P_lim=1.0e24,
        P0=0.0,
        ni=intrinsic_density,
        tau_n=1.0e-6,
        tau_p=1.0e-6,
        n1=intrinsic_density,
        p1=intrinsic_density,
        B_rad=0.0,
        C_n=0.0,
        C_p=0.0,
        alpha=0.0,
        N_A=0.0,
        N_D=0.0,
        chi=4.0,
        Eg=1.5,
        Nc300=1.0e25,
        Nv300=1.0e25,
    )
    right_intrinsic_density = math.sqrt(
        1.0e25 * 1.0e25 * math.exp(-1.3 / thermal_voltage(300.0))
    )
    right = replace(
        left,
        chi=4.1,
        Eg=1.3,
        ni=right_intrinsic_density,
        n1=right_intrinsic_density,
        p1=right_intrinsic_density,
    )
    document = InterfaceDefectDocument.from_scaps_cgs(
        sigma_n_cm2=6.0e-18 * capture_scale,
        sigma_p_cm2=1.0e-17 * capture_scale,
        thermal_velocity_cm_s=1.0e7,
        total_density_cm2=5.0e10,
        trap_depth_eV_below_cb=0.55,
    )
    return DeviceStack(
        layers=(
            LayerSpec("left", 1.0e-7, left, role="absorber"),
            LayerSpec("right", 1.0e-7, right, role="ETL"),
        ),
        interfaces=(document.capture_velocities_m_s,),
        interface_defects=(
            InterfaceDefect(
                E_t_eV=0.55,
                N_t_cm2=5.0e10,
                microscopic_document=document,
            ),
        ),
        interface_charge_closure="equilibrium_referenced",
        interface_charge_rebaseline_acknowledged=True,
        V_bi=0.0,
        Phi=0.0,
        mode="full",
        built_in_potential_mode="semiconductor_work_function",
    )


def _grid(stack: DeviceStack) -> np.ndarray:
    shared = multilayer_grid(
        [Layer(layer.thickness, 4) for layer in stack.layers],
        alpha=tuple(2.0 for _layer in stack.layers),
    )
    return build_two_sided_trace_grid(shared, stack)


@pytest.fixture(scope="module")
def certified_result():
    stack = _stack()
    return run_interface_defect_device_transient(
        _grid(stack),
        stack,
        TIMES_S,
        VOLTAGE_V,
        illuminated=False,
    )


def test_real_two_sided_interface_transient_certifies_sparse_closure(
    certified_result,
):
    result = certified_result

    certificate = result.certificate
    assert result.scope == INTERFACE_DEFECT_TRANSIENT_SCOPE
    assert certificate.certified
    assert certificate.reasons == ()
    assert certificate.sparse_linear_solver_used
    assert not certificate.clipping_used
    assert certificate.analytic_jacobian_nnz < certificate.dense_jacobian_entries
    assert certificate.maximum_analytic_jacobian_column_relative_error < 1.0e-6
    assert certificate.maximum_charge_balance_relative_error < 1.0e-10
    assert certificate.maximum_all_face_current_spread_relative < 1.0e-6
    assert certificate.maximum_two_sided_interface_total_current_relative_error < 1.0e-6
    assert certificate.maximum_eliminated_operator_relative_error < 3.0e-7
    assert certificate.maximum_refinement_state_change < 2.0e-2
    assert certificate.maximum_refinement_current_relative_change < 5.0e-2
    assert np.all(
        (result.interface_occupancy > 0.0) & (result.interface_occupancy < 1.0)
    )
    assert (
        abs(result.interface_occupancy[-1, 0] - result.interface_occupancy[0, 0])
        > 1.0e-8
    )
    np.testing.assert_allclose(
        result.interface_sheet_charge_C_m2,
        -Q
        * np.asarray(result.dark_reference.trap_density_m2)[None, :]
        * (
            result.interface_occupancy
            - np.asarray(result.dark_reference.equilibrium_occupancy)[None, :]
        ),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        result.total_current_faces_A_m2,
        result.conduction_current_faces_A_m2 + result.displacement_current_faces_A_m2,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        result.interface_total_current_A_m2,
        result.interface_conduction_current_A_m2
        + result.interface_displacement_current_A_m2,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        result.interface_total_current_A_m2[:, :, 0],
        result.interface_total_current_A_m2[:, :, 1],
        rtol=1.0e-6,
        atol=3.0e-15,
    )
    stack = _stack()
    interface_face = int(np.searchsorted(_grid(stack), stack.layers[0].thickness) - 1)
    np.testing.assert_allclose(
        result.conduction_current_faces_A_m2[:, interface_face],
        result.interface_conduction_current_A_m2[:, 0, 0],
        rtol=1.0e-14,
        atol=0.0,
    )
    np.testing.assert_array_equal(
        result.displacement_current_faces_A_m2[:, interface_face],
        result.interface_displacement_current_A_m2[:, 0, 0],
    )
    assert np.all(result.displacement_current_faces_A_m2[0] == 0.0)
    assert np.max(np.abs(result.displacement_current_faces_A_m2[1:])) > 0.0
    assert np.any(result.newton_iterations[1:] > 0)
    for values in (
        result.electron_density_m3,
        result.interface_occupancy,
        result.interface_quasi_steady_occupancy,
        result.interface_total_current_A_m2,
        result.total_current_faces_A_m2,
    ):
        assert not values.flags.writeable


def test_slow_capture_keeps_shared_occupancy_frozen_and_certified():
    stack = _stack(capture_scale=1.0e-6)
    result = run_interface_defect_device_transient(
        _grid(stack),
        stack,
        TIMES_S,
        VOLTAGE_V,
        illuminated=False,
    )

    assert result.certificate.certified
    assert (
        np.max(np.abs(result.interface_occupancy[-1] - result.interface_occupancy[0]))
        < 1.0e-12
    )


def test_fast_capture_recovers_two_sided_quasi_steady_occupancy():
    stack = _stack(capture_scale=1.0e7)
    result = run_interface_defect_device_transient(
        _grid(stack),
        stack,
        TIMES_S,
        VOLTAGE_V,
        illuminated=False,
    )

    assert result.certificate.certified
    assert (
        abs(result.interface_occupancy[-1, 0] - result.interface_occupancy[0, 0])
        > 5.0e-5
    )
    np.testing.assert_allclose(
        result.interface_occupancy[-1],
        result.interface_quasi_steady_occupancy[-1],
        rtol=0.0,
        atol=1.0e-9,
    )


def test_overstrict_two_sided_current_gate_returns_partial_or_fails_closed():
    stack = _stack()
    policy = InterfaceDefectTransientPolicy(
        maximum_two_sided_interface_total_current_relative_error=1.0e-12,
    )
    partial = run_interface_defect_device_transient(
        _grid(stack),
        stack,
        TIMES_S,
        VOLTAGE_V,
        policy=policy,
        require_certificate=False,
    )
    assert not partial.certificate.certified
    assert partial.certificate.reasons == (
        "two_sided_interface_total_current_closure_failed",
    )
    with pytest.raises(InterfaceDefectTransientCertificationError) as exc_info:
        run_interface_defect_device_transient(
            _grid(stack),
            stack,
            TIMES_S,
            VOLTAGE_V,
            policy=policy,
        )
    assert exc_info.value.result.certificate.reasons == partial.certificate.reasons


def test_unverified_dark_reference_is_rejected_before_integration(
    certified_result,
):
    stack = _stack()
    reference = certified_result.dark_reference
    unverified = replace(
        reference,
        dark_state=replace(
            reference.dark_state,
            contact_thermodynamic_status=None,
            contact_fermi_level_span_eV=None,
        ),
    )
    with pytest.raises(
        InterfaceDefectTransientError, match="dark-reference provenance"
    ):
        run_interface_defect_device_transient(
            _grid(stack),
            stack,
            TIMES_S,
            VOLTAGE_V,
            dark_reference=unverified,
        )


@pytest.mark.parametrize(
    ("case", "message"),
    (
        ("missing_document", "canonical microscopic defect document"),
        ("mobile_ions", "does not support mobile ions"),
        ("clamp_boundary", "barrier clamp switching boundary"),
    ),
)
def test_unsupported_physics_fails_closed_before_transient(case, message):
    stack = _stack()
    if case == "missing_document":
        stack = replace(stack, interface_defects=())
    elif case == "mobile_ions":
        stack = replace(
            stack,
            layers=tuple(
                replace(
                    layer,
                    params=replace(
                        layer.params,
                        D_ion=1.0e-14,
                        P0=1.0e20,
                        P_lim=1.0e21,
                    ),
                )
                for layer in stack.layers
            ),
        )
    else:
        stack = replace(
            stack,
            layers=(
                stack.layers[0],
                replace(stack.layers[1], params=stack.layers[0].params),
            ),
        )
    with pytest.raises(InterfaceDefectTransientError, match=message):
        run_interface_defect_device_transient(
            _grid(stack),
            stack,
            TIMES_S,
            VOLTAGE_V,
        )


@pytest.mark.parametrize(
    ("updates", "message"),
    (
        ({"storage_relative_tolerance": 0.0}, "finite and positive"),
        ({"maximum_newton_iterations": 0}, "must be positive"),
        (
            {"maximum_near_acceptance_nonmonotone_steps": -1},
            "must be non-negative",
        ),
        ({"refinement_substeps": (1,)}, "at least two"),
        ({"refinement_substeps": (1, 3, 4)}, "nested"),
    ),
)
def test_policy_rejects_incomplete_numerical_contract(updates, message):
    with pytest.raises(ValueError, match=message):
        InterfaceDefectTransientPolicy(**updates)


@pytest.mark.parametrize("value", (True, 1.5, "2"))
def test_nonmonotone_budget_requires_an_explicit_integer(value):
    with pytest.raises(TypeError, match="must be an integer"):
        InterfaceDefectTransientPolicy(maximum_near_acceptance_nonmonotone_steps=value)
