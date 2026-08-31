"""Fast contract tests for the Phase-1 tolerance-by-grid infrastructure."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest
import yaml

from perovskite_sim.validation.numerical_certificate import (
    CERTIFICATE_SCHEMA,
    REGISTRY_SCHEMA,
    CellResult,
    LaneDefinition,
    MatrixPoint,
    MetricValue,
    NumericalCertificate,
    NumericalCertificateError,
    ObservableGate,
    QualityGate,
    _observable_delta,
    canonical_json_bytes,
    content_sha256,
    evaluate_numerical_certificate,
    load_refinement_registry,
)
from perovskite_sim.validation.refinement_runner import (
    RefinementRunnerError,
    load_executor,
    plan_refinement,
    runtime_environment,
    run_refinement,
    source_provenance,
)


pytestmark = pytest.mark.regression
ROOT = Path(__file__).resolve().parents[2]


def _lane(
    config_path: Path,
    *,
    limit: float = 0.2,
    options: dict | None = None,
) -> LaneDefinition:
    return LaneDefinition(
        lane_id="test-lane",
        claim_level="internal-numerical-candidate",
        config_path=config_path.name,
        config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
        grid_parameter="N_grid",
        grid_values=(10, 20),
        tolerance_parameter="atol_factor",
        tolerance_factors=(1.0, 0.1),
        observables=(
            ObservableGate(
                metric="response",
                comparison="absolute_linf",
                limit=limit,
            ),
        ),
        quality_gates=(QualityGate("finite", "eq", 1.0),),
        options_json=json.dumps(options or {}),
        limitations=("synthetic contract test only",),
    )


def _completed(point: MatrixPoint, response: float, *, finite: float = 1.0):
    return CellResult(
        point=point,
        status="completed",
        observables=(MetricValue.from_value("response", response),),
        quality=(MetricValue.from_value("finite", finite),),
    )


def _certificate(lane: LaneDefinition, cells: list[CellResult]):
    return evaluate_numerical_certificate(
        lane,
        cells,
        run_id="0" * 64,
        source_commit="unknown",
        source_fingerprint_sha256="1" * 64,
        environment={"python": "test"},
        manifest_sha256="2" * 64,
        cell_artifact_sha256=[str(index) * 64 for index in range(3, 3 + len(cells))],
    )


def test_preregistered_numerical_lanes_and_thresholds_are_immutable():
    registry = load_refinement_registry(
        ROOT / "reproducibility/numerical_refinement_registry.yaml",
        project_root=ROOT,
    )

    assert registry.schema_version == REGISTRY_SCHEMA
    assert registry.certificate_schema == CERTIFICATE_SCHEMA
    assert {lane.lane_id for lane in registry.lanes} == {
        "scaps-mirror-frozen-ion-ss",
        "ionmonger-mobile-ion-transient",
        "ionmonger-ion-aware-dc-v1",
        "ionmonger-ion-aware-dc-resolved-v2",
        "ionmonger-ion-aware-impedance-resolved-v1",
        "dynamic-defect-ion-impedance-production-v1",
        "dynamic-defect-ion-transient-timescale-v1",
        "dynamic-defect-ion-transient-timescale-resolved-v2",
        "dynamic-defect-ion-transient-timescale-nonlinear-resolved-v3",
        "dynamic-defect-ion-transient-timescale-absorber-resolved-v4",
        "dynamic-defect-ion-transient-timescale-reference-resolved-v5",
        "dynamic-defect-ion-transient-production-v1",
        "csi-qf-frequency-domain",
        "csi-qf-frequency-domain-resolved-v2",
        "twod-uniform-limit",
        "twod-mobile-ion-interface-srh-v1",
        "external-series-shunt-dc-v1",
        "external-series-shunt-dc-operating-quadrant-v2",
        "electrothermal-terminal-mpp-v1",
        "electrothermal-terminal-mpp-resolved-v2",
        "electrothermal-terminal-mpp-grid-resolved-v3",
        "interface-recombination-charge-off",
        "interface-charge-equilibrium-referenced-v1",
        "interface-charge-device-stress-v1",
        "interface-charge-device-stress-resolved-v2",
        "interface-charge-qf-dc-jv-v1",
        "no-ion-dae-transient-v1",
        "single-positive-ion-dae-transient-v1",
        "dual-mobile-ion-dae-transient-v1",
        "algebraic-interface-state-dae-transient-v1",
        "single-ion-algebraic-interface-dae-transient-v1",
        "single-ion-algebraic-interface-dae-transient-resolved-v2",
        "degenerate-pn-equilibrium-v1",
        "incomplete-ionization-temperature-equilibrium-v1",
        "incomplete-ionization-bgn-temperature-equilibrium-v1",
        "bulk-energy-distributed-trap-equilibrium-v1",
        "charged-explicit-defect-qf-dc-v1",
        "distributed-explicit-defect-qf-dc-v1",
        "spatially-graded-explicit-defect-qf-dc-v1",
        "spatially-graded-explicit-defect-qf-dc-v2",
        "multivalent-explicit-defect-qf-dc-v1",
        "cigs-graded-optics-v1",
        "interface-srh-identifiability-synthetic-v1",
        "wkb-tunnelling-channel-qf-dc-v1",
    }
    assert all(
        len(lane.matrix_points) == 9
        for lane in registry.lanes
        if lane.lane_id
        not in {
            "interface-charge-device-stress-v1",
            "interface-charge-device-stress-resolved-v2",
        }
    )
    assert all(lane.definition_sha256 for lane in registry.lanes)
    assert all(lane.observables for lane in registry.lanes)
    assert all(lane.quality_gates for lane in registry.lanes)
    assert all(lane.executor is not None for lane in registry.lanes)
    assert all(lane.options["require_protocol"] for lane in registry.lanes)
    assert all(load_executor(lane.executor) for lane in registry.lanes)
    tunnelling = registry.lane("wkb-tunnelling-channel-qf-dc-v1")
    assert tunnelling.grid_values == (24, 48, 96)
    assert tunnelling.tolerance_factors == (1.0, 0.1, 0.01)
    assert tunnelling.grid_parameter == "intervals_per_electrical_layer"
    assert tunnelling.tolerance_parameter == "qf_dc_residual_tolerance_factor"
    # The energy quadrature order is this family's third refinement axis. The
    # shared MatrixPoint carries only (grid, tolerance), so the ladder is swept
    # inside each cell and reported as quality metrics — the same shape the
    # energy-distributed trap lanes use.
    assert tunnelling.options["energy_quadrature_orders"] == [96, 192, 384]
    observable_names = {gate.metric for gate in tunnelling.observables}
    # The channel's own flux must be an observable. A terminal-current-only
    # lane would certify nothing here: enabling the channel moves the terminal
    # current by ~1e-5 relative while the channel carries ~20% of it.
    assert "intraband_electron_net_flux_m2_s" in observable_names

    dynamic_transient = registry.lane(
        "dynamic-defect-ion-transient-timescale-reference-resolved-v5"
    )
    assert dynamic_transient.grid_values == (4, 6, 8)
    assert dynamic_transient.tolerance_factors == (1.0, 0.5, 0.25)
    assert dynamic_transient.grid_parameter == "intervals_per_layer"
    assert dynamic_transient.tolerance_parameter == ("backward_euler_time_step_factor")
    assert len(dynamic_transient.observables) == 12
    dynamic_transient_quality = {
        gate.metric: gate for gate in dynamic_transient.quality_gates
    }
    assert dynamic_transient_quality[
        "combined_interface_occupancy_motion"
    ].limit == pytest.approx(1.0e-9)
    assert dynamic_transient_quality[
        "defect_dominated_positive_ion_motion"
    ].limit == pytest.approx(1.0e-7)
    assert dynamic_transient_quality[
        "ion_dominated_interface_occupancy_motion"
    ].limit == pytest.approx(1.0e-12)
    assert dynamic_transient_quality["fast_ion_failure_iteration_reported"].limit == 1.0
    assert (
        dynamic_transient_quality["fast_ion_failure_residual_above_acceptance"].limit
        == 1.0
    )
    legacy_dynamic_transient = registry.lane(
        "dynamic-defect-ion-transient-timescale-v1"
    )
    resolved_dynamic_transient = registry.lane(
        "dynamic-defect-ion-transient-timescale-resolved-v2"
    )
    nonlinear_dynamic_transient = registry.lane(
        "dynamic-defect-ion-transient-timescale-nonlinear-resolved-v3"
    )
    absorber_dynamic_transient = registry.lane(
        "dynamic-defect-ion-transient-timescale-absorber-resolved-v4"
    )
    assert (
        absorber_dynamic_transient.quality_gates
        == nonlinear_dynamic_transient.quality_gates
    )
    assert absorber_dynamic_transient.options == nonlinear_dynamic_transient.options
    assert (
        absorber_dynamic_transient.config_sha256
        != nonlinear_dynamic_transient.config_sha256
    )
    assert dynamic_transient.options == absorber_dynamic_transient.options
    assert dynamic_transient.config_sha256 == absorber_dynamic_transient.config_sha256
    production_dynamic_transient = registry.lane(
        "dynamic-defect-ion-transient-production-v1"
    )
    assert production_dynamic_transient.executor_version == "v6"
    assert production_dynamic_transient.grid_values == dynamic_transient.grid_values
    assert (
        production_dynamic_transient.tolerance_factors
        == dynamic_transient.tolerance_factors
    )
    assert production_dynamic_transient.observables == dynamic_transient.observables
    production_options = dict(production_dynamic_transient.options)
    assert production_options.pop("production_method") == (
        "dynamic_defect_transient_certified"
    )
    assert production_options == dynamic_transient.options
    production_quality = {
        gate.metric: gate for gate in production_dynamic_transient.quality_gates
    }
    for metric in (
        "public_projection_certified",
        "public_protocol_identity_verified",
        "reference_certificate_bound",
    ):
        assert production_quality.pop(metric).limit == 1.0
    assert production_quality == dynamic_transient_quality
    assert (
        nonlinear_dynamic_transient.observables
        == resolved_dynamic_transient.observables
    )
    assert (
        resolved_dynamic_transient.observables == legacy_dynamic_transient.observables
    )
    resolved_quality = {
        gate.metric: gate for gate in resolved_dynamic_transient.quality_gates
    }
    active_quality = {
        gate.metric: gate for gate in nonlinear_dynamic_transient.quality_gates
    }
    nonmonotone_gate = active_quality.pop("max_near_acceptance_nonmonotone_step_count")
    assert active_quality == resolved_quality
    assert resolved_dynamic_transient.quality_gates == (
        legacy_dynamic_transient.quality_gates
    )
    assert nonmonotone_gate.limit == 12.0
    assert dynamic_transient.options["grid_alpha"] == 1.5
    assert resolved_dynamic_transient.options["grid_alpha"] == 1.5
    assert legacy_dynamic_transient.options["grid_alpha"] == 2.0
    assert dynamic_transient.options["maximum_newton_iterations"] == 100
    assert dynamic_transient.options["maximum_line_search_steps"] == 40
    assert dynamic_transient.options["maximum_near_acceptance_nonmonotone_steps"] == 2
    assert (
        nonlinear_dynamic_transient.options["maximum_near_acceptance_nonmonotone_steps"]
        == 2
    )
    assert {
        key: value
        for key, value in nonlinear_dynamic_transient.options.items()
        if key != "maximum_near_acceptance_nonmonotone_steps"
    } == resolved_dynamic_transient.options
    active_observable_names = {gate.metric for gate in dynamic_transient.observables}
    assert "combined_interface_occupancy_change" in active_observable_names
    assert "combined_interface_occupancy" not in active_observable_names
    assert "combined_positive_ion_centroid_shift_m" in active_observable_names
    assert "combined_integrated_charge_change_C_m2" in active_observable_names
    assert "combined_positive_ion_centroid_m" not in active_observable_names
    assert "combined_integrated_charge_C_m2" not in active_observable_names
    absorber_observable_names = {
        gate.metric for gate in absorber_dynamic_transient.observables
    }
    assert "combined_interface_occupancy" in absorber_observable_names
    assert "combined_interface_occupancy_change" not in absorber_observable_names
    assert registry.lane("twod-uniform-limit").grid_values == (1, 2, 4)
    combined_2d = registry.lane("twod-mobile-ion-interface-srh-v1")
    assert combined_2d.grid_values == (4, 6, 8)
    assert combined_2d.tolerance_factors == (1.0, 0.1, 0.01)
    external_circuit = registry.lane("external-series-shunt-dc-v1")
    assert external_circuit.grid_values == (20, 30, 40)
    assert external_circuit.tolerance_factors == (1.0, 0.1, 0.01)
    external_observables = {gate.metric: gate for gate in external_circuit.observables}
    assert external_observables[
        "terminal_current_normalized_trace"
    ].limit == pytest.approx(0.01)
    assert external_observables["terminal_voc_V"].limit == pytest.approx(0.005)
    external_quality = {gate.metric: gate for gate in external_circuit.quality_gates}
    assert external_quality["max_current_balance_error_A_m2"].limit == 0.0
    assert external_quality["min_pce_loss_fraction"].limit == pytest.approx(0.01)
    external_resolved = registry.lane("external-series-shunt-dc-operating-quadrant-v2")
    assert external_resolved.grid_values == external_circuit.grid_values
    assert external_resolved.tolerance_factors == external_circuit.tolerance_factors
    resolved_external_observables = {
        gate.metric: gate for gate in external_resolved.observables
    }
    assert resolved_external_observables[
        "terminal_power_quadrant_normalized_trace"
    ].limit == pytest.approx(0.005)
    assert "terminal_current_normalized_trace" not in (resolved_external_observables)
    resolved_external_quality = {
        gate.metric: gate for gate in external_resolved.quality_gates
    }
    assert resolved_external_quality["terminal_quadrant_points_completed"].limit == 42.0
    electrothermal = registry.lane("electrothermal-terminal-mpp-v1")
    assert electrothermal.grid_values == (10, 15, 20)
    assert electrothermal.tolerance_factors == (1.0, 0.1, 0.01)
    electrothermal_observables = {
        gate.metric: gate for gate in electrothermal.observables
    }
    assert electrothermal_observables["operating_temperature_K"].limit == (
        pytest.approx(0.5)
    )
    assert electrothermal_observables["terminal_mpp_power_W_m2"].limit == (
        pytest.approx(0.02)
    )
    electrothermal_quality = {
        gate.metric: gate for gate in electrothermal.quality_gates
    }
    assert electrothermal_quality["power_balance_residual_W_m2"].limit == (
        pytest.approx(0.2)
    )
    assert electrothermal_quality["temperature_response_active_W_m2"].limit == (
        pytest.approx(1.0)
    )
    electrothermal_resolved = registry.lane("electrothermal-terminal-mpp-resolved-v2")
    assert electrothermal_resolved.grid_values == (20, 30, 40)
    assert electrothermal_resolved.tolerance_factors == electrothermal.tolerance_factors
    assert electrothermal_resolved.observables == electrothermal.observables
    assert electrothermal_resolved.quality_gates == electrothermal.quality_gates
    assert electrothermal_resolved.options == electrothermal.options
    electrothermal_grid_resolved = registry.lane(
        "electrothermal-terminal-mpp-grid-resolved-v3"
    )
    assert electrothermal_grid_resolved.grid_values == (40, 60, 80)
    assert (
        electrothermal_grid_resolved.tolerance_factors
        == electrothermal.tolerance_factors
    )
    assert electrothermal_grid_resolved.observables == electrothermal.observables
    assert electrothermal_grid_resolved.quality_gates == electrothermal.quality_gates
    assert electrothermal_grid_resolved.options == electrothermal.options
    ion_dc = registry.lane("ionmonger-ion-aware-dc-v1")
    ion_dc_resolved = registry.lane("ionmonger-ion-aware-dc-resolved-v2")
    assert ion_dc.grid_values == (30, 60, 90)
    assert ion_dc.tolerance_factors == (1.0, 0.1, 0.01)
    assert ion_dc_resolved.grid_values == (60, 90, 120)
    assert ion_dc_resolved.tolerance_factors == ion_dc.tolerance_factors
    assert ion_dc_resolved.observables == ion_dc.observables
    assert ion_dc_resolved.quality_gates == ion_dc.quality_gates
    assert ion_dc_resolved.options == ion_dc.options
    ion_impedance = registry.lane("ionmonger-ion-aware-impedance-resolved-v1")
    assert ion_impedance.grid_values == (60, 90, 120)
    assert ion_impedance.tolerance_factors == (1.0, 0.5, 0.25)
    assert ion_impedance.observables[0].comparison == ("pointwise_relative_linf")
    csi_minimum = registry.lane("csi-qf-frequency-domain")
    csi_resolved = registry.lane("csi-qf-frequency-domain-resolved-v2")
    assert csi_minimum.grid_values == (100, 200, 300)
    assert csi_resolved.grid_values == (
        200,
        300,
        400,
    )
    assert csi_resolved.tolerance_factors == csi_minimum.tolerance_factors
    assert csi_resolved.observables == csi_minimum.observables
    assert csi_resolved.quality_gates == csi_minimum.quality_gates
    assert csi_resolved.options == csi_minimum.options
    interface_charge_off = registry.lane("interface-recombination-charge-off")
    interface_quality = {
        gate.metric: gate for gate in interface_charge_off.quality_gates
    }
    assert (
        interface_quality["max_interface_state_residual_A_m2"].limit
        == (interface_quality["max_continuity_bound_A_m2"].limit)
        == pytest.approx(1.0e-4)
    )
    interface_charge = registry.lane("interface-charge-equilibrium-referenced-v1")
    assert interface_charge.grid_values == (30, 60, 120)
    assert interface_charge.tolerance_factors == (1.0, 0.5, 0.25)
    charged_observables = {gate.metric: gate for gate in interface_charge.observables}
    assert charged_observables[
        "interface_trace_potential_shift_V"
    ].limit == pytest.approx(1.0e-3)
    assert (
        charged_observables["interface_sheet_charge_C_m2"].comparison
        == "pointwise_relative_linf"
    )
    charged_quality = {gate.metric: gate for gate in interface_charge.quality_gates}
    assert charged_quality["max_normalized_gauss_residual"].limit == (
        pytest.approx(1.0e-10)
    )
    assert charged_quality["dark_incremental_charge_zero_C_m2"].limit == 0.0
    charged_jv = registry.lane("interface-charge-qf-dc-jv-v1")
    assert charged_jv.grid_values == (30, 60, 90)
    assert charged_jv.tolerance_factors == (1.0, 0.5, 0.25)
    assert charged_jv.options["V_max_V"] == pytest.approx(0.1)
    assert charged_jv.options["voltage_points"] == 5
    charged_jv_observables = {gate.metric: gate for gate in charged_jv.observables}
    assert charged_jv_observables["jv_normalized"].limit == pytest.approx(0.005)
    assert (
        charged_jv_observables["interface_sheet_charge_C_m2"].comparison
        == "pointwise_relative_linf"
    )
    charged_jv_quality = {gate.metric: gate for gate in charged_jv.quality_gates}
    assert charged_jv_quality["reported_point_count"].limit == 5.0
    assert charged_jv_quality["max_contact_fermi_level_span_eV"].limit == pytest.approx(
        0.005
    )
    assert charged_jv_quality["continuation_bridge_count"].limit == 8.0
    stress = registry.lane("interface-charge-device-stress-v1")
    assert stress.grid_values == (30, 60)
    assert stress.tolerance_factors == (1.0, 0.5)
    assert len(stress.matrix_points) == 4
    assert len(stress.options["stress_points"]) == 9
    assert {point["axis"] for point in stress.options["stress_points"]} == {
        "baseline",
        "trap_energy",
        "conduction_band_offset",
        "etl_doping",
        "trap_density",
    }
    stress_quality = {gate.metric: gate for gate in stress.quality_gates}
    assert stress_quality["stress_point_count"].limit == 9.0
    assert stress_quality["barrier_shift_charge_sign_consistent"].limit == 1.0
    assert stress_quality["max_normalized_gauss_residual"].limit == (
        pytest.approx(1.0e-10)
    )
    resolved_stress = registry.lane("interface-charge-device-stress-resolved-v2")
    assert resolved_stress.grid_values == (30, 60, 90)
    assert resolved_stress.tolerance_factors == stress.tolerance_factors
    assert len(resolved_stress.matrix_points) == 6
    assert resolved_stress.observables == stress.observables
    assert resolved_stress.quality_gates == stress.quality_gates
    assert stress.options["base_finite_difference_step"] == pytest.approx(1.0e-5)
    assert resolved_stress.options["base_finite_difference_step"] == pytest.approx(
        7.0e-6
    )
    assert stress.options.get("refine_finite_difference_step", True) is True
    assert resolved_stress.options["refine_finite_difference_step"] is False
    assert {
        key: value
        for key, value in resolved_stress.options.items()
        if key
        not in {
            "base_finite_difference_step",
            "refine_finite_difference_step",
        }
    } == {
        key: value
        for key, value in stress.options.items()
        if key
        not in {
            "base_finite_difference_step",
            "refine_finite_difference_step",
        }
    }
    combined_dae = registry.lane("single-ion-algebraic-interface-dae-transient-v1")
    assert combined_dae.grid_values == (4, 8, 16)
    assert combined_dae.tolerance_factors == (1.0, 0.5, 0.25)
    combined_observables = {gate.metric: gate for gate in combined_dae.observables}
    assert combined_observables[
        "structured_dense_terminal_log_density_difference"
    ].limit == pytest.approx(1.0e-9)
    assert combined_observables[
        "terminal_positive_ion_relative_error"
    ].limit == pytest.approx(1.0e-6)
    combined_quality = {gate.metric: gate for gate in combined_dae.quality_gates}
    assert combined_quality[
        "max_positive_ion_inventory_relative_drift"
    ].limit == pytest.approx(1.0e-12)
    assert combined_quality[
        "max_terminal_interface_state_relative_error"
    ].limit == pytest.approx(5.0e-7)
    assert combined_quality["structured_csr_nonzeros_per_node"].limit == pytest.approx(
        27.0
    )
    assert combined_quality["structured_rhs_work_fraction"].limit == (
        pytest.approx(0.1)
    )
    assert combined_dae.options["positive_ion_diffusion_m2_s"] == (
        pytest.approx(1.0e-16)
    )
    assert combined_dae.options["positive_ion_reference_m3"] == (pytest.approx(1.0e22))
    assert combined_dae.options["positive_ion_site_limit_m3"] == (pytest.approx(1.0e24))
    assert combined_dae.options["newton_residual_tolerance"] == (pytest.approx(1.0e-8))
    resolved_combined_dae = registry.lane(
        "single-ion-algebraic-interface-dae-transient-resolved-v2"
    )
    assert resolved_combined_dae.grid_values == combined_dae.grid_values
    assert resolved_combined_dae.tolerance_factors == combined_dae.tolerance_factors
    resolved_observables = {
        gate.metric: gate for gate in resolved_combined_dae.observables
    }
    assert resolved_observables[
        "terminal_positive_ion_relative_error"
    ].limit == pytest.approx(3.0e-6)
    resolved_quality = {
        gate.metric: gate for gate in resolved_combined_dae.quality_gates
    }
    assert resolved_quality["max_normalized_carrier_residual"].limit == pytest.approx(
        5.0e-8
    )
    assert resolved_quality[
        "max_terminal_positive_ion_relative_error"
    ].limit == pytest.approx(1.0e-5)
    assert resolved_combined_dae.options["newton_residual_tolerance"] == (
        pytest.approx(5.0e-8)
    )
    distributed_defect = registry.lane("distributed-explicit-defect-qf-dc-v1")
    assert distributed_defect.grid_values == (16, 32, 64)
    assert distributed_defect.tolerance_factors == (1.0, 0.1, 0.01)
    assert distributed_defect.options["energy_quadrature_orders"] == [16, 32, 64]
    distributed_observables = {
        gate.metric: gate for gate in distributed_defect.observables
    }
    assert distributed_observables["jv_current_A_m2"].limit == pytest.approx(0.002)
    assert distributed_observables[
        "dark_source_occupancy_profile"
    ].limit == pytest.approx(0.005)
    distributed_quality = {
        gate.metric: gate for gate in distributed_defect.quality_gates
    }
    assert distributed_quality["energy_orders_completed"].limit == 3.0
    assert distributed_quality[
        "max_energy_tangent_relative_change"
    ].limit == pytest.approx(0.005)
    assert distributed_quality["default_distributed_path_rejected"].limit == 1.0
    spatial_defect = registry.lane("spatially-graded-explicit-defect-qf-dc-v1")
    assert spatial_defect.grid_values == (16, 32, 64)
    assert spatial_defect.tolerance_factors == (1.0, 0.1, 0.01)
    assert spatial_defect.options["energy_quadrature_orders"] == [16, 32, 64]
    spatial_observables = {gate.metric: gate for gate in spatial_defect.observables}
    assert spatial_observables[
        "dark_source_density_multiplier_profile"
    ].limit == pytest.approx(1.0e-12)
    assert spatial_observables["jv_current_A_m2"].limit == pytest.approx(0.002)
    spatial_quality = {gate.metric: gate for gate in spatial_defect.quality_gates}
    assert spatial_quality["profiled_species_count"].limit == 3.0
    assert spatial_quality["minimum_support_margin_eV"].limit == pytest.approx(0.04)
    assert spatial_quality["graded_profiles_compiled_verified"].limit == 1.0
    cigs_optics = registry.lane("cigs-graded-optics-v1")
    assert cigs_optics.grid_values == (8, 16, 32)
    assert cigs_optics.tolerance_factors == (1.0, 0.5, 0.25)
    cigs_observables = {gate.metric: gate for gate in cigs_optics.observables}
    assert cigs_observables["normalized_generation_profile"].limit == (
        pytest.approx(0.005)
    )
    cigs_quality = {gate.metric: gate for gate in cigs_optics.quality_gates}
    assert cigs_quality[
        "max_electrical_optical_gap_mismatch_eV"
    ].limit == pytest.approx(0.01)
    assert cigs_quality["independent_carron_energy_points_completed"].limit == 453.0
    with pytest.raises(FrozenInstanceError):
        registry.lanes[0].grid_values = (1, 2)  # type: ignore[misc]


def test_pointwise_relative_linf_does_not_hide_small_frequency_bins():
    coarse = MetricValue.from_value("spectrum", [1000.0, 1.0])
    fine = MetricValue.from_value("spectrum", [1000.0, 1.1])
    global_gate = ObservableGate("spectrum", "relative_linf", 0.01)
    pointwise_gate = ObservableGate("spectrum", "pointwise_relative_linf", 0.01)

    assert _observable_delta(coarse, fine, global_gate) < global_gate.limit
    assert _observable_delta(coarse, fine, pointwise_gate) > pointwise_gate.limit


def test_certificate_distinguishes_certified_partial_and_failed(tmp_path):
    config = tmp_path / "device.yaml"
    config.write_text("device: test\n", encoding="utf-8")
    lane = _lane(config)
    points = lane.matrix_points
    passing = [
        _completed(points[0], 1.00),
        _completed(points[1], 1.05),
        _completed(points[2], 1.10),
        _completed(points[3], 1.12),
    ]
    certified = _certificate(lane, passing)
    assert certified.status == "certified"
    assert certified.unconverged_dimensions == ()
    assert len(certified.checks) == 6
    assert NumericalCertificate.from_dict(certified.to_dict()) == certified

    nonconverged = [*passing[:-1], _completed(points[-1], 1.50)]
    partial = _certificate(lane, nonconverged)
    assert partial.status == "partial"
    assert set(partial.unconverged_dimensions) == {
        "grid:response",
        "tolerance:response",
    }

    failed = _certificate(lane, passing[:-1])
    assert failed.status == "failed"
    assert failed.missing_cells == (points[-1].key,)
    assert failed.unconverged_dimensions == (f"matrix:{points[-1].key}",)

    wrong_shape = [*passing[:-1], _completed(points[-1], [1.12, 1.13])]
    malformed = _certificate(lane, wrong_shape)
    assert malformed.status == "failed"
    assert set(malformed.unconverged_dimensions) == {
        "grid:response:shape",
        "observable:response:shape_across_matrix",
        "tolerance:response:shape",
    }


def test_refinement_runner_resumes_content_addressed_cells_without_overwrite(
    tmp_path,
):
    project = tmp_path / "project"
    project.mkdir()
    config = project / "device.yaml"
    config.write_text("device: test\n", encoding="utf-8")
    lane = _lane(config, limit=2.0)
    calls: list[str] = []

    def executor(_lane, point, _root):
        calls.append(point.key)
        return {
            "observables": {"response": point.grid / 10.0},
            "quality": {"finite": 1.0},
            "metadata": {"point": point.key},
        }

    first = run_refinement(
        lane,
        executor,
        project_root=project,
        output_root=tmp_path / "results",
        executor_id="tests.synthetic:resume-v1",
        max_cells=2,
    )
    assert first.executed_cells == 2
    assert first.certificate.status == "failed"
    assert len(first.certificate.missing_cells) == 2

    resumed = run_refinement(
        lane,
        executor,
        project_root=project,
        output_root=tmp_path / "results",
        executor_id="tests.synthetic:resume-v1",
    )
    assert resumed.run_directory == first.run_directory
    assert resumed.executed_cells == 2
    assert resumed.reused_cells == 2
    assert resumed.certificate.status == "certified"
    assert len(calls) == 4
    assert len(tuple((resumed.run_directory / "cells").glob("*.json"))) == 4

    repeated = run_refinement(
        lane,
        executor,
        project_root=project,
        output_root=tmp_path / "results",
        executor_id="tests.synthetic:resume-v1",
    )
    assert repeated.executed_cells == 0
    assert repeated.reused_cells == 4
    assert repeated.certificate.certificate_sha256 == (
        resumed.certificate.certificate_sha256
    )
    assert repeated.certificate_path.read_text(encoding="ascii") == (
        resumed.certificate_path.read_text(encoding="ascii")
    )
    assert len(calls) == 4


def test_failed_cell_can_be_retried_without_deleting_prior_artifact(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    config = project / "device.yaml"
    config.write_text("device: test\n", encoding="utf-8")
    lane = _lane(config, limit=2.0)
    should_fail = {lane.matrix_points[1].key}

    def executor(_lane, point, _root):
        if point.key in should_fail:
            raise RuntimeError("injected failure")
        return {
            "observables": {"response": point.grid / 10.0},
            "quality": {"finite": 1.0},
        }

    first = run_refinement(
        lane,
        executor,
        project_root=project,
        output_root=tmp_path / "results",
        executor_id="tests.synthetic:retry-v1",
    )
    assert first.certificate.status == "failed"
    assert first.certificate.failed_cells == (lane.matrix_points[1].key,)
    assert len(tuple((first.run_directory / "cells").glob("*.json"))) == 4

    should_fail.clear()
    retried = run_refinement(
        lane,
        executor,
        project_root=project,
        output_root=tmp_path / "results",
        executor_id="tests.synthetic:retry-v1",
        retry_failed=True,
    )
    assert retried.executed_cells == 1
    assert retried.reused_cells == 3
    assert retried.certificate.status == "certified"
    assert len(tuple((retried.run_directory / "cells").glob("*.json"))) == 5


def test_refinement_runner_rejects_historical_reference_output_root(tmp_path):
    project = tmp_path / "project"
    protected = project / "perovskite_sim/data/references"
    protected.mkdir(parents=True)
    config = project / "device.yaml"
    config.write_text("device: test\n", encoding="utf-8")
    lane = _lane(config)
    calls = 0

    def executor(_lane, point, _root):
        nonlocal calls
        calls += 1
        return {
            "observables": {"response": point.grid},
            "quality": {"finite": 1.0},
        }

    with pytest.raises(RefinementRunnerError, match="historical reference"):
        run_refinement(
            lane,
            executor,
            project_root=project,
            output_root=protected / "attempted-overwrite",
            executor_id="tests.synthetic:protected-v1",
        )
    assert calls == 0


def test_protocol_hash_is_manifested_and_inconsistency_fails_closed(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    config = project / "device.yaml"
    config.write_text("device: test\n", encoding="utf-8")
    lane = _lane(config, limit=2.0, options={"require_protocol": True})

    def executor(_lane, point, _root):
        protocol = {
            "schema_version": "synthetic-protocol-v1",
            "voltage_V": [0.0, float(point.grid)],
        }
        return {
            "observables": {"response": point.grid / 10.0},
            "quality": {"finite": 1.0},
            "metadata": {
                "protocol": protocol,
                "protocol_hash": content_sha256(protocol),
                "protocol_schema": protocol["schema_version"],
            },
        }

    outcome = run_refinement(
        lane,
        executor,
        project_root=project,
        output_root=tmp_path / "results",
        executor_id="tests.synthetic:protocol-v1",
    )
    assert outcome.certificate.status == "failed"
    assert outcome.certificate.protocol_sha256 is None
    assert "protocol:inconsistent_across_matrix" in (
        outcome.certificate.unconverged_dimensions
    )
    manifest = json.loads(outcome.manifest_path.read_text(encoding="ascii"))
    assert len(manifest["protocols"]) == 2


def test_plan_refinement_is_read_only(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    config = project / "device.yaml"
    config.write_text("device: test\n", encoding="utf-8")
    lane = _lane(config)
    destination = tmp_path / "not-created"

    def executor(_lane, point, _root):
        return {
            "observables": {"response": point.grid},
            "quality": {"finite": 1.0},
        }

    plan = plan_refinement(
        lane,
        executor,
        project_root=project,
        output_root=destination,
        executor_id="tests.synthetic:dry-run-v1",
    )
    assert plan.to_dict()["matrix_cells"] == 4
    assert not destination.exists()


def _git(project: Path, *arguments: str) -> None:
    subprocess.run(
        ["git", *arguments],
        cwd=project,
        check=True,
        capture_output=True,
        text=True,
    )


def test_source_fingerprint_includes_staged_only_changes(tmp_path):
    repository = tmp_path / "repository"
    project = repository / "perovskite-sim"
    source = project / "perovskite_sim/solver.py"
    source.parent.mkdir(parents=True)
    source.write_text("VALUE = 1\n", encoding="utf-8")
    _git(repository, "init")
    _git(repository, "config", "user.email", "tests@example.invalid")
    _git(repository, "config", "user.name", "SolarLab tests")
    _git(repository, "add", "perovskite-sim/perovskite_sim/solver.py")
    _git(repository, "commit", "-m", "baseline")

    clean = source_provenance(project)
    source.write_text("VALUE = 2\n", encoding="utf-8")
    _git(repository, "add", "perovskite-sim/perovskite_sim/solver.py")
    staged = source_provenance(project)

    assert staged["fingerprint_sha256"] != clean["fingerprint_sha256"]
    assert staged["source_changes"] == [
        {
            "path": "perovskite_sim/solver.py",
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        }
    ]

    source.unlink()
    _git(repository, "add", "--update")
    deleted = source_provenance(project)
    assert deleted["fingerprint_sha256"] != staged["fingerprint_sha256"]
    assert deleted["source_changes"] == [
        {"path": "perovskite_sim/solver.py", "sha256": "deleted"}
    ]


def test_behavior_environment_changes_run_identity(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    config = project / "device.yaml"
    config.write_text("device: test\n", encoding="utf-8")
    lane = _lane(config)

    def executor(_lane, point, _root):
        return {
            "observables": {"response": point.grid},
            "quality": {"finite": 1.0},
        }

    monkeypatch.delenv("SOLARLAB_BAND_GRADING", raising=False)
    baseline = plan_refinement(
        lane,
        executor,
        project_root=project,
        output_root=tmp_path / "results",
        executor_id="tests.synthetic:environment-v1",
    )
    monkeypatch.setenv("SOLARLAB_BAND_GRADING", "1")
    enabled = plan_refinement(
        lane,
        executor,
        project_root=project,
        output_root=tmp_path / "results",
        executor_id="tests.synthetic:environment-v1",
    )

    expected_behavior_variables = {
        "PEROVSKITE_RHS_FINITE_CHECK",
        "SOLARLAB_AUTOLOOP_GEN",
        "SOLARLAB_BAND_GRADING",
        "SOLARLAB_DOS_BAND",
        "SOLARLAB_IFACE_ALLOW_GEN",
        "SOLARLAB_IFACE_PLANE",
        "SOLARLAB_IFACE_PLANE_GEN",
        "SOLARLAB_IFACE_PROJ",
        "SOLARLAB_IFACE_QSS",
        "SOLARLAB_IFACE_SHARED_OCC",
        "SOLARLAB_IFACE_TUNNEL",
        "SOLARLAB_IFACE_TWOSIDED",
        "SOLARLAB_INTERFACE_PLANE_STATE",
        "SOLARLAB_ION_STERIC_DIFF",
        "SOLARLAB_QSS_VTH",
        "SOLARLAB_SS_GUMMEL",
        "SOLARLAB_SS_JAC_REUSE",
        "SOLARLAB_TE_PHYSICAL",
    }
    assert expected_behavior_variables == set(
        runtime_environment()["behavior_variables"]
    )
    assert baseline.environment["behavior_variables"]["SOLARLAB_BAND_GRADING"] == (
        "unset"
    )
    assert enabled.environment["behavior_variables"]["SOLARLAB_BAND_GRADING"] == "1"
    assert enabled.run_id != baseline.run_id


def test_every_completed_cell_must_match_registered_observable_contract(tmp_path):
    config = tmp_path / "device.yaml"
    config.write_text("device: test\n", encoding="utf-8")
    lane = _lane(config)
    points = lane.matrix_points
    passing = [_completed(point, 1.0) for point in points]

    unregistered = CellResult(
        point=points[0],
        status="completed",
        observables=(MetricValue.from_value("unregistered", 1.0),),
        quality=(MetricValue.from_value("finite", 1.0),),
    )
    missing = _certificate(lane, [unregistered, *passing[1:]])
    assert missing.status == "failed"
    assert f"observable:response:missing@{points[0].key}" in (
        missing.unconverged_dimensions
    )
    assert f"observable:unregistered:unexpected@{points[0].key}" in (
        missing.unconverged_dimensions
    )

    wrong_units = CellResult(
        point=points[0],
        status="completed",
        observables=(MetricValue.from_value("response", 1.0, units="V"),),
        quality=(MetricValue.from_value("finite", 1.0),),
    )
    unit_failure = _certificate(lane, [wrong_units, *passing[1:]])
    assert unit_failure.status == "failed"
    assert f"observable:response:units@{points[0].key}" in (
        unit_failure.unconverged_dimensions
    )

    wrong_shape = CellResult(
        point=points[0],
        status="completed",
        observables=(MetricValue.from_value("response", [1.0, 1.0]),),
        quality=(MetricValue.from_value("finite", 1.0),),
    )
    shape_failure = _certificate(lane, [wrong_shape, *passing[1:]])
    assert shape_failure.status == "failed"
    assert "observable:response:shape_across_matrix" in (
        shape_failure.unconverged_dimensions
    )


@pytest.mark.parametrize(
    "artifact_hashes",
    [[], ["3" * 64] * 4],
    ids=["missing", "duplicate"],
)
def test_certificate_rejects_incomplete_artifact_hash_evidence(
    tmp_path,
    artifact_hashes,
):
    config = tmp_path / "device.yaml"
    config.write_text("device: test\n", encoding="utf-8")
    lane = _lane(config)
    cells = [_completed(point, 1.0) for point in lane.matrix_points]

    with pytest.raises(NumericalCertificateError, match="artifact"):
        evaluate_numerical_certificate(
            lane,
            cells,
            run_id="0" * 64,
            source_commit="unknown",
            source_fingerprint_sha256="1" * 64,
            environment={"python": "test"},
            manifest_sha256="2" * 64,
            cell_artifact_sha256=artifact_hashes,
        )


def test_certificate_and_registry_reject_unknown_or_malformed_fields(tmp_path):
    config = tmp_path / "device.yaml"
    config.write_text("device: test\n", encoding="utf-8")
    lane = _lane(config)
    certified = _certificate(
        lane,
        [_completed(point, 1.0) for point in lane.matrix_points],
    )

    extra_certificate_field = certified.to_dict()
    extra_certificate_field["unhashed_annotation"] = {"claim": "external"}
    with pytest.raises(NumericalCertificateError, match="unknown keys"):
        NumericalCertificate.from_dict(extra_certificate_field)

    extra_check_field = certified.to_dict()
    extra_check_field["checks"][0]["unhashed_annotation"] = True
    with pytest.raises(NumericalCertificateError, match="unknown keys"):
        NumericalCertificate.from_dict(extra_check_field)

    registry_raw = yaml.safe_load(
        (ROOT / "reproducibility/numerical_refinement_registry.yaml").read_text(
            encoding="utf-8"
        )
    )
    registry_raw["unexpected"] = True
    registry_path = tmp_path / "unknown-registry.yaml"
    registry_path.write_text(yaml.safe_dump(registry_raw), encoding="utf-8")
    with pytest.raises(NumericalCertificateError, match="unknown keys"):
        load_refinement_registry(registry_path, verify_config_hashes=False)

    malformed_raw = yaml.safe_load(
        (ROOT / "reproducibility/numerical_refinement_registry.yaml").read_text(
            encoding="utf-8"
        )
    )
    malformed_raw["lanes"]["scaps-mirror-frozen-ion-ss"]["observables"]["voc_V"] = (
        "not-a-mapping"
    )
    malformed_path = tmp_path / "malformed-registry.yaml"
    malformed_path.write_text(yaml.safe_dump(malformed_raw), encoding="utf-8")
    with pytest.raises(NumericalCertificateError, match="must be a mapping"):
        load_refinement_registry(malformed_path, verify_config_hashes=False)


def test_complex_metric_values_fail_closed():
    with pytest.raises(NumericalCertificateError, match="real-valued"):
        MetricValue.from_value("response", np.array([1.0 + 2.0j]))


def test_resume_rejects_content_addressed_manifest_with_changed_identity(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    config = project / "device.yaml"
    config.write_text("device: test\n", encoding="utf-8")
    lane = _lane(config, limit=2.0)

    def executor(_lane, point, _root):
        return {
            "observables": {"response": point.grid / 10.0},
            "quality": {"finite": 1.0},
        }

    first = run_refinement(
        lane,
        executor,
        project_root=project,
        output_root=tmp_path / "results",
        executor_id="tests.synthetic:manifest-identity-v1",
        max_cells=1,
    )
    manifest = json.loads(first.manifest_path.read_text(encoding="ascii"))
    manifest["environment"]["python"] = "forged-runtime"
    manifest_bytes = canonical_json_bytes(manifest)
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    forged_path = first.run_directory / "manifests" / f"{manifest_sha}.json"
    forged_path.write_bytes(manifest_bytes)
    state_path = first.run_directory / "state.json"
    state = json.loads(state_path.read_text(encoding="ascii"))
    state["latest_manifest_sha256"] = manifest_sha
    state_path.write_bytes(canonical_json_bytes(state))

    with pytest.raises(RefinementRunnerError, match="identity"):
        run_refinement(
            lane,
            executor,
            project_root=project,
            output_root=tmp_path / "results",
            executor_id="tests.synthetic:manifest-identity-v1",
        )
