"""D6-E3c source-bound transient refinement contract tests."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from perovskite_sim.experiments.interface_defect_ion_transient import (
    InterfaceDefectIonTransientError,
    InterfaceDefectIonTransientPolicy,
    run_interface_defect_ion_device_transient,
)
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.validation.dynamic_defect_transient_refinement import (
    _case_identity,
    _case_stacks,
    _build_grid,
    _execution_protocol,
    _integer_vector_option,
    _line_search_iteration,
    _nested_substeps,
    _nonlinear_failure_iteration,
    _nonlinear_failure_outcome,
    _nonlinear_failure_residual,
    _source_case_identity_verified,
    _validate_lane_contract,
)
from perovskite_sim.validation.numerical_certificate import (
    content_sha256,
    load_refinement_registry,
)


ROOT = Path(__file__).resolve().parents[3]
LANE_ID = "dynamic-defect-ion-transient-timescale-reference-resolved-v5"
ABSORBER_V4_LANE_ID = "dynamic-defect-ion-transient-timescale-absorber-resolved-v4"
NONLINEAR_V3_LANE_ID = "dynamic-defect-ion-transient-timescale-nonlinear-resolved-v3"
RESOLVED_V2_LANE_ID = "dynamic-defect-ion-transient-timescale-resolved-v2"
LEGACY_LANE_ID = "dynamic-defect-ion-transient-timescale-v1"


def _lane():
    return load_refinement_registry(
        ROOT / "reproducibility/numerical_refinement_registry.yaml",
        project_root=ROOT,
    ).lane(LANE_ID)


def _source_and_cases():
    lane = _lane()
    source = load_device_from_yaml(ROOT / lane.config_path)
    cases = _case_stacks(
        source,
        slow_ion_diffusivity_m2_s=1.0e-20,
        slow_capture_scale=1.0e-12,
    )
    identities = {name: _case_identity(stack) for name, stack in cases.items()}
    return lane, source, cases, identities


def test_registered_lane_freezes_three_by_three_timescale_contract():
    lane = _lane()

    assert lane.grid_parameter == "intervals_per_layer"
    assert lane.grid_values == (4, 6, 8)
    assert lane.tolerance_parameter == "backward_euler_time_step_factor"
    assert lane.tolerance_factors == (1.0, 0.5, 0.25)
    assert lane.executor_version == "v5"
    assert len(lane.matrix_points) == 9
    assert len(lane.observables) == 12
    assert lane.options == {
        "base_nested_substeps": [1, 2, 4],
        "config_loader": "standard",
        "fast_ion_diffusivity_m2_s": 1.0e-12,
        "grid_alpha": 1.5,
        "maximum_line_search_steps": 40,
        "maximum_near_acceptance_nonmonotone_steps": 2,
        "maximum_newton_iterations": 100,
        "require_protocol": True,
        "slow_capture_scale": 1.0e-12,
        "slow_ion_diffusivity_m2_s": 1.0e-20,
        "times_s": [0.0, 1.0e-8, 1.0e-6, 1.0e-4],
        "voltage_V": [0.0, 0.05, 0.05, 0.05],
    }
    _validate_lane_contract(lane)


def test_resolved_v2_changes_only_registered_mesh_and_iteration_controls():
    registry = load_refinement_registry(
        ROOT / "reproducibility/numerical_refinement_registry.yaml",
        project_root=ROOT,
    )
    resolved = registry.lane(RESOLVED_V2_LANE_ID)
    legacy = registry.lane(LEGACY_LANE_ID)

    assert resolved.config_path == legacy.config_path
    assert resolved.config_sha256 == legacy.config_sha256
    assert resolved.executor == legacy.executor
    assert resolved.grid_parameter == legacy.grid_parameter
    assert resolved.grid_values == legacy.grid_values
    assert resolved.tolerance_parameter == legacy.tolerance_parameter
    assert resolved.tolerance_factors == legacy.tolerance_factors
    assert resolved.observables == legacy.observables
    assert resolved.quality_gates == legacy.quality_gates
    resolved_options = dict(resolved.options)
    legacy_options = dict(legacy.options)
    assert resolved_options.pop("grid_alpha") == 1.5
    assert legacy_options.pop("grid_alpha") == 2.0
    assert resolved_options.pop("maximum_newton_iterations") == 100
    assert resolved_options.pop("maximum_line_search_steps") == 40
    assert resolved_options == legacy_options
    _validate_lane_contract(resolved)
    _validate_lane_contract(legacy)


def test_nonlinear_resolved_v3_changes_only_registered_globalization_control():
    registry = load_refinement_registry(
        ROOT / "reproducibility/numerical_refinement_registry.yaml",
        project_root=ROOT,
    )
    active = registry.lane(NONLINEAR_V3_LANE_ID)
    resolved = registry.lane(RESOLVED_V2_LANE_ID)

    assert active.config_path == resolved.config_path
    assert active.config_sha256 == resolved.config_sha256
    assert active.executor == resolved.executor
    assert active.grid_parameter == resolved.grid_parameter
    assert active.grid_values == resolved.grid_values
    assert active.tolerance_parameter == resolved.tolerance_parameter
    assert active.tolerance_factors == resolved.tolerance_factors
    assert active.observables == resolved.observables
    active_quality = {gate.metric: gate for gate in active.quality_gates}
    resolved_quality = {gate.metric: gate for gate in resolved.quality_gates}
    nonmonotone_gate = active_quality.pop("max_near_acceptance_nonmonotone_step_count")
    assert nonmonotone_gate.operator == "le"
    assert nonmonotone_gate.limit == 12.0
    assert nonmonotone_gate.units == "1"
    assert active_quality == resolved_quality
    active_options = dict(active.options)
    assert active_options.pop("maximum_near_acceptance_nonmonotone_steps") == 2
    assert active_options == resolved.options


def test_absorber_resolved_v4_changes_fixture_and_transient_observable_reference():
    registry = load_refinement_registry(
        ROOT / "reproducibility/numerical_refinement_registry.yaml",
        project_root=ROOT,
    )
    active = registry.lane(ABSORBER_V4_LANE_ID)
    previous = registry.lane(NONLINEAR_V3_LANE_ID)

    assert active.config_path == (
        "configs/dynamic_interface_defect_ion_transient_absorber_only.yaml"
    )
    assert active.config_sha256 == (
        "f617f230b2d9c144573394e38fcc313225dc84c7b38f7670b97e9a0a7cc12a24"
    )
    assert active.config_sha256 != previous.config_sha256
    assert active.executor == previous.executor
    assert active.grid_parameter == previous.grid_parameter
    assert active.grid_values == previous.grid_values
    assert active.tolerance_parameter == previous.tolerance_parameter
    assert active.tolerance_factors == previous.tolerance_factors
    assert active.quality_gates == previous.quality_gates
    assert active.options == previous.options
    active_observables = {gate.metric for gate in active.observables}
    previous_observables = {gate.metric for gate in previous.observables}
    for case in ("combined", "defect_dominated", "ion_dominated"):
        assert f"{case}_positive_ion_centroid_shift_m" in active_observables
        assert f"{case}_integrated_charge_change_C_m2" in active_observables
        assert f"{case}_positive_ion_centroid_m" in previous_observables
        assert f"{case}_integrated_charge_C_m2" in previous_observables
    assert active_observables - previous_observables == {
        f"{case}_{suffix}"
        for case in ("combined", "defect_dominated", "ion_dominated")
        for suffix in (
            "positive_ion_centroid_shift_m",
            "integrated_charge_change_C_m2",
        )
    }
    assert previous_observables - active_observables == {
        f"{case}_{suffix}"
        for case in ("combined", "defect_dominated", "ion_dominated")
        for suffix in ("positive_ion_centroid_m", "integrated_charge_C_m2")
    }


def test_reference_resolved_v5_changes_only_reference_and_stiffness_contract():
    registry = load_refinement_registry(
        ROOT / "reproducibility/numerical_refinement_registry.yaml",
        project_root=ROOT,
    )
    active = registry.lane(LANE_ID)
    previous = registry.lane(ABSORBER_V4_LANE_ID)

    assert active.config_path == previous.config_path
    assert active.config_sha256 == previous.config_sha256
    assert active.executor == previous.executor
    assert active.grid_parameter == previous.grid_parameter
    assert active.grid_values == previous.grid_values
    assert active.tolerance_parameter == previous.tolerance_parameter
    assert active.tolerance_factors == previous.tolerance_factors
    assert active.options == previous.options

    active_observables = {gate.metric: gate for gate in active.observables}
    previous_observables = {gate.metric: gate for gate in previous.observables}
    for case in ("combined", "defect_dominated", "ion_dominated"):
        active_change = active_observables.pop(f"{case}_interface_occupancy_change")
        previous_absolute = previous_observables.pop(f"{case}_interface_occupancy")
        assert active_change.comparison == previous_absolute.comparison
        assert active_change.limit == previous_absolute.limit
        assert active_change.units == previous_absolute.units
    assert active_observables == previous_observables

    active_quality = {gate.metric: gate for gate in active.quality_gates}
    previous_quality = {gate.metric: gate for gate in previous.quality_gates}
    previous_quality.pop("fast_ion_failure_is_line_search_stall")
    declared = active_quality.pop("fast_ion_failure_is_declared_nonlinear_stall")
    residual = active_quality.pop("fast_ion_failure_residual_above_acceptance")
    assert declared.operator == "eq" and declared.limit == 1.0
    assert residual.operator == "eq" and residual.limit == 1.0
    assert active_quality == previous_quality


def test_nested_substeps_bind_each_outer_time_step_factor():
    assert _nested_substeps(1.0) == (1, 2, 4)
    assert _nested_substeps(0.5) == (2, 4, 8)
    assert _nested_substeps(0.25) == (4, 8, 16)

    for invalid in (0.0, -1.0, 0.3, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            _nested_substeps(invalid)
    with pytest.raises(ValueError, match="nested ladder"):
        _nested_substeps(1.0, (1, 3, 4))


def test_integer_substeps_reject_silent_fractional_or_boolean_coercion():
    assert _integer_vector_option(
        {"base_nested_substeps": [1, 2, 4]},
        "base_nested_substeps",
        (8, 16),
    ) == (1, 2, 4)
    for invalid in ([1.0, 2.0], [True, 2], "1,2,4", 4):
        with pytest.raises(ValueError, match="integer vector"):
            _integer_vector_option(
                {"base_nested_substeps": invalid},
                "base_nested_substeps",
                (1, 2, 4),
            )


def test_case_overrides_change_only_frozen_kinetic_controls():
    _lane_value, source, cases, identities = _source_and_cases()
    combined = cases["combined"]
    defect_dominated = cases["defect_dominated"]
    ion_dominated = cases["ion_dominated"]
    source_defect = source.interface_defects[0]
    ion_defect = ion_dominated.interface_defects[0]

    assert combined is source
    assert [layer.params.D_ion for layer in source.layers] == [1.0e-14, 0.0]
    assert [layer.params.D_ion for layer in defect_dominated.layers] == [
        1.0e-20,
        0.0,
    ]
    assert defect_dominated.interfaces == source.interfaces
    assert defect_dominated.interface_defects == source.interface_defects
    assert ion_defect is not None and source_defect is not None
    assert ion_defect.microscopic_document is not None
    assert source_defect.microscopic_document is not None
    assert ion_defect.microscopic_document.kinetics.sigma_n_m2 == pytest.approx(
        source_defect.microscopic_document.kinetics.sigma_n_m2 * 1.0e-12,
        rel=1.0e-15,
    )
    assert ion_defect.microscopic_document.kinetics.sigma_p_m2 == pytest.approx(
        source_defect.microscopic_document.kinetics.sigma_p_m2 * 1.0e-12,
        rel=1.0e-15,
    )
    assert ion_dominated.interfaces == (
        ion_defect.microscopic_document.capture_velocities_m_s,
    )
    assert [layer.params.D_ion for layer in ion_dominated.layers] == [
        layer.params.D_ion for layer in source.layers
    ]
    assert identities["combined"]["interface_document_sha256"] == [
        source_defect.microscopic_document.sha256
    ]
    assert _source_case_identity_verified(
        source,
        cases,
        identities,
        slow_ion_diffusivity_m2_s=1.0e-20,
        slow_capture_scale=1.0e-12,
    )


def test_case_identity_check_rejects_an_unrecorded_interface_override():
    _lane_value, source, cases, identities = _source_and_cases()
    tampered = dict(cases)
    tampered["ion_dominated"] = replace(
        tampered["ion_dominated"],
        interfaces=source.interfaces,
    )

    assert not _source_case_identity_verified(
        source,
        tampered,
        identities,
        slow_ion_diffusivity_m2_s=1.0e-20,
        slow_capture_scale=1.0e-12,
    )


def test_protocol_binds_lane_case_history_and_matrix_identity():
    lane, _source, cases, identities = _source_and_cases()
    policy = replace(
        InterfaceDefectIonTransientPolicy(),
        refinement_substeps=(1, 2, 4),
        maximum_newton_iterations=100,
        maximum_line_search_steps=40,
        maximum_near_acceptance_nonmonotone_steps=2,
    )
    protocol = _execution_protocol(
        lane,
        times_s=np.asarray(lane.options["times_s"]),
        voltage_V=np.asarray(lane.options["voltage_V"]),
        policy=policy,
        base_nested_substeps=(1, 2, 4),
        case_identities=identities,
        slow_ion_diffusivity_m2_s=1.0e-20,
        slow_capture_scale=1.0e-12,
        fast_ion_diffusivity_m2_s=1.0e-12,
    )

    assert protocol["lane"] == {
        "config_path": lane.config_path,
        "config_sha256": lane.config_sha256,
        "definition_sha256": lane.definition_sha256,
        "executor_version": "v5",
        "lane_id": LANE_ID,
    }
    assert protocol["matrix_controls"] == {
        "grid_parameter": "intervals_per_layer",
        "grid_values": [4, 6, 8],
        "nested_substeps_by_tolerance_factor": {
            "0.25": [4, 8, 16],
            "0.5": [2, 4, 8],
            "1": [1, 2, 4],
        },
        "tolerance_factors": [1.0, 0.5, 0.25],
        "tolerance_parameter": "backward_euler_time_step_factor",
    }
    assert protocol["cases"]["combined"]["source_identity"] == _case_identity(
        cases["combined"]
    )
    assert (
        protocol["solver_policy_common"]["maximum_near_acceptance_nonmonotone_steps"]
        == 2
    )
    assert "refinement_substeps" not in protocol["solver_policy_common"]
    assert protocol["schema_version"] == (
        "dynamic-defect-ion-transient-refinement-protocol-v2"
    )
    assert protocol["stiffness_boundary"] == {
        "accepted_outcomes": [
            "typed_line_search_stall",
            "typed_newton_iteration_limit",
        ],
        "ion_diffusivity_m2_s": 1.0e-12,
        "iteration_and_residual_required": True,
        "not_a_certified_physical_solution": True,
    }
    baseline_hash = content_sha256(protocol)

    another_cell = _execution_protocol(
        lane,
        times_s=np.asarray(lane.options["times_s"]),
        voltage_V=np.asarray(lane.options["voltage_V"]),
        policy=replace(policy, refinement_substeps=(2, 4, 8)),
        base_nested_substeps=(1, 2, 4),
        case_identities=identities,
        slow_ion_diffusivity_m2_s=1.0e-20,
        slow_capture_scale=1.0e-12,
        fast_ion_diffusivity_m2_s=1.0e-12,
    )
    assert content_sha256(another_cell) == baseline_hash


def test_lane_contract_rejects_unknown_options_and_axis_drift():
    lane = _lane()
    options = dict(lane.options)
    options["unregistered_override"] = 1.0
    with pytest.raises(ValueError, match="exact schema"):
        _validate_lane_contract(replace(lane, options_json=json.dumps(options)))
    with pytest.raises(ValueError, match="grid parameter"):
        _validate_lane_contract(replace(lane, grid_parameter="N_grid"))
    with pytest.raises(ValueError, match="tolerance parameter"):
        _validate_lane_contract(replace(lane, tolerance_parameter="atol_factor"))
    with pytest.raises(ValueError, match="reference-resolved-v5"):
        _validate_lane_contract(replace(lane, lane_id="unregistered-d6-e3c-lane"))
    with pytest.raises(ValueError, match="grid values"):
        _validate_lane_contract(replace(lane, grid_values=(4, 6, 10)))
    with pytest.raises(ValueError, match="tolerance factors"):
        _validate_lane_contract(replace(lane, tolerance_factors=(1.0, 0.5, 0.125)))


def test_fast_ion_line_search_iteration_is_machine_readable():
    message = (
        "joint interface/ion transient solve failed: analytic sparse Newton "
        "line search stalled at iteration 7 with residual 1.05126e+08"
    )
    assert _line_search_iteration(message) == 7
    assert _line_search_iteration("line search stalled at residual 1.0") is None


@pytest.mark.parametrize(
    ("message", "iteration", "residual", "outcome"),
    [
        (
            "analytic sparse Newton line search stalled at iteration 7 with "
            "residual 1.05126e+08, charge closure 2",
            7,
            1.05126e8,
            "typed_line_search_stall",
        ),
        (
            "analytic sparse Newton exceeded 100 iterations with residual "
            "6.90255e-4, charge closure 2",
            100,
            6.90255e-4,
            "typed_newton_iteration_limit",
        ),
    ],
)
def test_declared_nonlinear_failure_is_machine_readable(
    message: str,
    iteration: int,
    residual: float,
    outcome: str,
):
    assert _nonlinear_failure_iteration(message) == iteration
    assert _nonlinear_failure_residual(message) == pytest.approx(residual)
    assert _nonlinear_failure_outcome(message) == outcome


def test_undeclared_nonlinear_failure_fails_closed():
    message = "analytic sparse Newton failed without structured diagnostics"
    assert _nonlinear_failure_iteration(message) is None
    assert _nonlinear_failure_residual(message) is None
    assert _nonlinear_failure_outcome(message) is None


def test_grid8_fine_defect_case_requires_bounded_opt_in_nonmonotone_steps():
    lane, source, cases, _identities = _source_and_cases()
    grid = _build_grid(source, intervals_per_layer=8, grid_alpha=1.5)
    times = np.asarray(lane.options["times_s"], dtype=float)
    voltage = np.asarray(lane.options["voltage_V"], dtype=float)
    monotone_policy = replace(
        InterfaceDefectIonTransientPolicy(),
        refinement_substeps=(4, 8, 16),
        maximum_newton_iterations=100,
        maximum_line_search_steps=40,
    )

    assert monotone_policy.maximum_near_acceptance_nonmonotone_steps == 0
    with pytest.raises(InterfaceDefectIonTransientError, match="line search stalled"):
        run_interface_defect_ion_device_transient(
            grid,
            cases["defect_dominated"],
            times,
            voltage,
            policy=monotone_policy,
        )

    resolved = run_interface_defect_ion_device_transient(
        grid,
        cases["defect_dominated"],
        times,
        voltage,
        policy=replace(
            monotone_policy,
            maximum_near_acceptance_nonmonotone_steps=2,
        ),
    )

    assert resolved.certificate.certified
    assert 0 < resolved.certificate.near_acceptance_nonmonotone_step_count <= 12
    assert resolved.certificate.maximum_all_face_current_spread_relative < 2.0e-6
    assert (
        resolved.certificate.maximum_two_sided_interface_total_current_relative_error
        < 2.0e-6
    )
