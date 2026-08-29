"""D6-E1 charged bulk occupancy coupled to device transient equations."""

from __future__ import annotations

import math

import numpy as np
import pytest

from perovskite_sim.discretization.grid import Layer, multilayer_grid
from perovskite_sim.experiments.bulk_defect_transient import (
    BULK_DEFECT_TRANSIENT_SCOPE,
    BulkDefectTransientCertificationError,
    BulkDefectTransientError,
    BulkDefectTransientPolicy,
    run_bulk_defect_device_transient,
)
from perovskite_sim.models.defects import (
    ACCEPTOR,
    DONOR,
    EXPLICIT_DEFECT_SCHEMA_VERSION,
    EXPLICIT_QUASI_STEADY,
    INTEGRATED_TOTAL,
    NEUTRAL_WHEN_EMPTY,
    NEUTRAL_WHEN_FILLED,
    SINGLE_LEVEL,
    BulkDefectDistribution,
    BulkDefectKinetics,
    BulkDefectSpecies,
)
from perovskite_sim.models.device import DeviceStack, LayerSpec
from perovskite_sim.models.parameters import MaterialParams
from perovskite_sim.physics.dynamic_defect_state import (
    quasi_steady_bulk_trap_occupancy,
)
from perovskite_sim.physics.temperature import thermal_voltage


TEMPERATURE_K = 300.0
GAP_EV = 0.80
NC_M3 = 1.0e24
NV_M3 = 8.0e23
TIMES_S = np.array([0.0, 1.0e-8, 1.0e-6, 1.0e-4, 1.0e-2])
VOLTAGE_V = np.array([0.0, 0.02, 0.02, 0.02, 0.02])


def _species(transition: str, capture_scale: float = 1.0) -> BulkDefectSpecies:
    return BulkDefectSpecies(
        name=f"bulk_{transition}",
        distribution=BulkDefectDistribution(
            kind=SINGLE_LEVEL,
            normalization=INTEGRATED_TOTAL,
            total_density_m3=2.0e21,
            center_eV_above_vb=0.39,
        ),
        charge_transition=transition,
        neutral_reference=(
            NEUTRAL_WHEN_EMPTY if transition == ACCEPTOR else NEUTRAL_WHEN_FILLED
        ),
        kinetics=BulkDefectKinetics(
            sigma_n_m2=2.0e-19 * capture_scale,
            sigma_p_m2=7.0e-20 * capture_scale,
            thermal_velocity_n_m_s=1.0e5,
            thermal_velocity_p_m_s=8.0e4,
        ),
        degeneracy=1.0,
    )


def _stack(
    transition: str = ACCEPTOR,
    *,
    capture_scale: float = 1.0,
    explicit: bool = True,
) -> DeviceStack:
    intrinsic = math.sqrt(
        NC_M3 * NV_M3 * math.exp(-GAP_EV / thermal_voltage(TEMPERATURE_K))
    )
    params = MaterialParams(
        eps_r=20.0,
        mu_n=2.0e-3,
        mu_p=2.0e-3,
        D_ion=0.0,
        P_lim=1.0e30,
        P0=0.0,
        ni=intrinsic,
        tau_n=1.0e-6,
        tau_p=1.0e-6,
        n1=intrinsic,
        p1=intrinsic,
        B_rad=0.0,
        C_n=0.0,
        C_p=0.0,
        alpha=4.0e5,
        N_A=0.0,
        N_D=0.0,
        chi=4.0,
        Eg=GAP_EV,
        Nc300=NC_M3,
        Nv300=NV_M3,
        defect_schema_version=(EXPLICIT_DEFECT_SCHEMA_VERSION if explicit else None),
        defect_model=(EXPLICIT_QUASI_STEADY if explicit else "effective_lifetime"),
        bulk_defects=((_species(transition, capture_scale),) if explicit else ()),
    )
    return DeviceStack(
        layers=(LayerSpec("defective", 300.0e-9, params, "absorber"),),
        V_bi=0.0,
        Phi=1.0e20,
        interfaces=(),
        mode="legacy",
        built_in_potential_mode="semiconductor_work_function",
    )


def _grid(stack: DeviceStack) -> np.ndarray:
    return multilayer_grid([Layer(layer.thickness, 4) for layer in stack.layers])


def _run(
    *,
    transition: str = ACCEPTOR,
    capture_scale: float = 1.0,
    policy: BulkDefectTransientPolicy | None = None,
    require_certificate: bool = True,
):
    stack = _stack(transition, capture_scale=capture_scale)
    return run_bulk_defect_device_transient(
        _grid(stack),
        stack,
        TIMES_S,
        VOLTAGE_V,
        illuminated=True,
        policy=policy,
        require_certificate=require_certificate,
    )


@pytest.mark.parametrize(
    ("transition", "expected_sign"),
    ((ACCEPTOR, -1.0), (DONOR, 1.0)),
)
def test_device_transient_certifies_sparse_charge_current_closure(
    transition,
    expected_sign,
):
    result = _run(transition=transition)
    certificate = result.certificate

    assert certificate.certified
    assert certificate.reasons == ()
    assert certificate.scope == BULK_DEFECT_TRANSIENT_SCOPE
    assert certificate.sparse_linear_solver_used
    assert not certificate.clipping_used
    assert certificate.analytic_jacobian_nnz < certificate.dense_jacobian_entries
    assert certificate.maximum_analytic_jacobian_column_relative_error < 2.0e-7
    assert certificate.maximum_all_face_current_spread_relative < 1.0e-10
    assert certificate.maximum_charge_balance_relative_error < 1.0e-10
    assert certificate.maximum_eliminated_operator_relative_error < 1.0e-10
    assert certificate.maximum_refinement_state_change < 2.0e-2
    assert certificate.maximum_refinement_current_relative_change < 5.0e-2
    assert np.all((result.trap_occupancy > 0.0) & (result.trap_occupancy < 1.0))
    assert np.max(np.abs(result.trap_occupancy[-1] - result.trap_occupancy[0])) > 1.0e-6
    charged_nodes = result.layout.device_node_indices
    assert np.all(
        expected_sign * result.trap_charge_density_C_m3[:, charged_nodes] >= 0.0
    )
    np.testing.assert_allclose(
        result.total_current_faces_A_m2,
        result.conduction_current_faces_A_m2 + result.displacement_current_faces_A_m2,
        rtol=0.0,
        atol=0.0,
    )
    assert np.all(result.displacement_current_faces_A_m2[0] == 0.0)
    assert np.max(np.abs(result.displacement_current_faces_A_m2[1:])) > 0.0
    assert np.any(result.newton_iterations[1:] > 0)
    assert not result.trap_occupancy.flags.writeable
    assert not result.total_current_faces_A_m2.flags.writeable


def test_fast_and_slow_capture_recover_qss_and_frozen_occupancy_limits():
    slow = _run(capture_scale=1.0e-6)
    fast = _run(capture_scale=1.0e2)

    slow_motion = np.max(np.abs(slow.trap_occupancy[-1] - slow.trap_occupancy[0]))
    assert slow_motion < 1.0e-6
    fast_qss = quasi_steady_bulk_trap_occupancy(
        fast.electron_density_m3[-1],
        fast.hole_density_m3[-1],
        fast.layout,
    )
    np.testing.assert_allclose(
        fast.trap_occupancy[-1],
        fast_qss,
        rtol=0.0,
        atol=1.0e-12,
    )
    assert slow.certificate.certified
    assert fast.certificate.certified


def test_overstrict_evidence_gate_returns_partial_or_fails_closed():
    policy = BulkDefectTransientPolicy(
        maximum_charge_balance_relative_error=1.0e-16,
    )
    partial = _run(policy=policy, require_certificate=False)
    assert not partial.certificate.certified
    assert partial.certificate.reasons == ("carrier_trap_charge_balance_failed",)
    with pytest.raises(BulkDefectTransientCertificationError) as exc_info:
        _run(policy=policy)
    assert exc_info.value.result.certificate.reasons == (
        "carrier_trap_charge_balance_failed",
    )


def test_device_transient_rejects_lifetime_only_material_before_integration():
    stack = _stack(explicit=False)
    with pytest.raises(BulkDefectTransientError, match="explicit-defect model"):
        run_bulk_defect_device_transient(
            _grid(stack),
            stack,
            TIMES_S,
            VOLTAGE_V,
            illuminated=True,
        )


@pytest.mark.parametrize(
    ("updates", "message"),
    (
        ({"storage_relative_tolerance": 0.0}, "finite and positive"),
        ({"maximum_newton_iterations": 0}, "must be positive"),
        ({"refinement_substeps": (1,)}, "at least two"),
        ({"refinement_substeps": (1, 3, 4)}, "nested"),
    ),
)
def test_policy_rejects_incomplete_numerical_contract(updates, message):
    with pytest.raises(ValueError, match=message):
        BulkDefectTransientPolicy(**updates)
