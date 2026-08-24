"""Numerical refinement adapter for the terminal-MPP electrothermal root."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from perovskite_sim.experiments.electrothermal import (
    ElectrothermalJVProtocol,
    ElectrothermalOperatingPointProtocol,
    solve_electrothermal_operating_point,
)
from perovskite_sim.experiments.external_circuit import ExternalCircuitProtocol
from perovskite_sim.experiments.thermal_balance import LumpedThermalProtocol
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.models.mode import resolve_mode

from .dae_refinement import (
    _finite_option,
    _integer_option,
    _protocol_metadata,
    _string_option,
)
from .numerical_certificate import LaneDefinition, MatrixPoint
from .refinement_runner import CellMeasurement


def _study_protocol(
    lane: LaneDefinition,
    *,
    options: dict[str, Any],
    thermal: LumpedThermalProtocol,
    circuit: ExternalCircuitProtocol,
    operating: ElectrothermalOperatingPointProtocol,
) -> dict[str, Any]:
    return {
        "control_volume": {
            "absorbed_optical_power": "explicit_not_inferred_from_irradiance",
            "boundary": thermal.system_boundary,
            "first_law": (
                "P_abs+P_internal-P_terminal_mpp-h*(T-T_ambient)-"
                "emissivity*sigma*(T^4-T_ambient^4)=0"
            ),
            "protocol": thermal.to_dict(),
        },
        "electrical_execution": {
            "atol_policy": {
                "carrier_fraction": _finite_option(
                    options,
                    "carrier_atol_fraction",
                    1.0e-12,
                ),
                "interface_fraction": _finite_option(
                    options,
                    "interface_atol_fraction",
                    1.0e-12,
                ),
                "ion_fraction": _finite_option(
                    options,
                    "ion_atol_fraction",
                    1.0e-12,
                ),
                "minimum_atol": _finite_option(
                    options,
                    "minimum_atol",
                    1.0e-6,
                ),
                "refinement_factor_source": "matrix.tolerance_factor",
            },
            "certification_mode": "strict",
            "grid_parameter": lane.grid_parameter,
            "illumination": "stack_baseline_generation",
            "incident_power_W_m2": _finite_option(
                options,
                "incident_power_W_m2",
                1000.0,
            ),
            "relative_tolerance": _finite_option(options, "rtol", 1.0e-4),
            "scan_rate_V_s": _finite_option(
                options,
                "scan_rate_V_s",
                20.0,
            ),
            "temperature_initialization": "fresh_state_per_temperature",
            "tolerance_parameter": lane.tolerance_parameter,
            "voltage_max_V": _finite_option(options, "V_max_V", 1.2),
            "voltage_points_per_branch": _integer_option(
                options,
                "voltage_points",
                6,
                minimum=3,
            ),
        },
        "external_circuit": circuit.to_dict(),
        "matrix": {
            "grid_values": list(lane.grid_values),
            "tolerance_factors": list(lane.tolerance_factors),
        },
        "operating_point": operating.to_dict(),
        "schema_version": "electrothermal-terminal-mpp-refinement-protocol-v1",
    }


def _final_mpp_index(result) -> int:
    branch = getattr(
        result.final_external_result,
        result.operating_protocol.operating_branch,
    )
    matches = np.flatnonzero(
        (branch.terminal_voltage_V == result.terminal_voltage_V)
        & (branch.terminal_current_A_m2 == result.terminal_current_A_m2)
        & (branch.terminal_power_W_m2 == result.terminal_power_W_m2)
    )
    if matches.size != 1:
        raise RuntimeError("final terminal MPP is not unique in its source branch")
    return int(matches[0])


def run_electrothermal_terminal_mpp_refinement(
    lane: LaneDefinition,
    point: MatrixPoint,
    project_root: Path,
) -> CellMeasurement:
    """Run one fresh-state electrothermal grid/tolerance matrix cell."""

    options = lane.options
    if _string_option(options, "config_loader", "standard") != "standard":
        raise ValueError("the electrothermal lane requires config_loader='standard'")
    if options.get("require_protocol") is not True:
        raise ValueError("the electrothermal lane requires an explicit protocol")

    thermal = LumpedThermalProtocol(
        absorbed_optical_power_W_m2=_finite_option(
            options,
            "absorbed_optical_power_W_m2",
            800.0,
            positive=False,
        ),
        ambient_temperature_K=_finite_option(
            options,
            "ambient_temperature_K",
            300.0,
        ),
        areal_heat_capacity_J_m2_K=_finite_option(
            options,
            "areal_heat_capacity_J_m2_K",
            2000.0,
        ),
        heat_transfer_coefficient_W_m2_K=_finite_option(
            options,
            "heat_transfer_coefficient_W_m2_K",
            20.0,
            positive=False,
        ),
        emissivity=_finite_option(
            options,
            "emissivity",
            0.85,
            positive=False,
        ),
        maximum_temperature_K=_finite_option(
            options,
            "maximum_temperature_K",
            380.0,
        ),
        constant_internal_heat_W_m2=_finite_option(
            options,
            "constant_internal_heat_W_m2",
            0.0,
            positive=False,
        ),
        steady_power_residual_tolerance_W_m2=_finite_option(
            options,
            "steady_power_residual_tolerance_W_m2",
            0.2,
        ),
    )
    circuit = ExternalCircuitProtocol(
        series_resistance_ohm_m2=_finite_option(
            options,
            "series_resistance_ohm_m2",
            2.0e-4,
            positive=False,
        ),
        shunt_resistance_ohm_m2=_finite_option(
            options,
            "shunt_resistance_ohm_m2",
            0.2,
        ),
    )
    electrical = ElectrothermalJVProtocol(
        grid_points_per_electrical_layer=point.grid,
        voltage_points_per_branch=_integer_option(
            options,
            "voltage_points",
            6,
            minimum=3,
        ),
        scan_rate_V_s=_finite_option(options, "scan_rate_V_s", 20.0),
        voltage_max_V=_finite_option(options, "V_max_V", 1.2),
        relative_tolerance=_finite_option(options, "rtol", 1.0e-4),
        carrier_atol_fraction=_finite_option(
            options,
            "carrier_atol_fraction",
            1.0e-12,
        ),
        ion_atol_fraction=_finite_option(
            options,
            "ion_atol_fraction",
            1.0e-12,
        ),
        interface_atol_fraction=_finite_option(
            options,
            "interface_atol_fraction",
            1.0e-12,
        ),
        minimum_atol=_finite_option(options, "minimum_atol", 1.0e-6),
        atol_refinement_factor=point.tolerance_factor,
        incident_power_W_m2=_finite_option(
            options,
            "incident_power_W_m2",
            1000.0,
        ),
    )
    operating = ElectrothermalOperatingPointProtocol(
        temperature_absolute_tolerance_K=_finite_option(
            options,
            "temperature_absolute_tolerance_K",
            0.005,
        ),
        maximum_root_iterations=_integer_option(
            options,
            "maximum_root_iterations",
            30,
            minimum=1,
        ),
        operating_branch=_string_option(
            options,
            "operating_branch",
            "forward",
        ),
    )
    study_protocol = _study_protocol(
        lane,
        options=options,
        thermal=thermal,
        circuit=circuit,
        operating=operating,
    )
    stack = load_device_from_yaml(project_root / lane.config_path)
    initial_stack_temperature = float(stack.T)
    if not resolve_mode(stack.mode).use_temperature_scaling:
        raise ValueError("the electrothermal lane requires temperature scaling")

    result = solve_electrothermal_operating_point(
        stack,
        thermal,
        circuit,
        electrical,
        operating,
    )
    if not result.certified:
        raise RuntimeError("electrothermal operating point is not certified")
    if float(stack.T) != initial_stack_temperature:
        raise RuntimeError("electrothermal solve mutated the source stack")

    evaluations = result.temperature_evaluations
    ambient_evaluation = next(
        item
        for item in evaluations
        if item.temperature_K == thermal.ambient_temperature_K
    )
    maximum_evaluation = next(
        item
        for item in evaluations
        if item.temperature_K == thermal.maximum_temperature_K
    )
    final_index = _final_mpp_index(result)
    final_branch = getattr(
        result.final_external_result,
        operating.operating_branch,
    )
    mpp_series_drop = float(final_branch.series_voltage_drop_V[final_index])
    mpp_shunt_current = float(final_branch.shunt_current_A_m2[final_index])
    first_law_reconstruction = (
        result.thermal_ledger.absorbed_optical_power_W_m2
        + result.thermal_ledger.constant_internal_heat_W_m2
        - result.thermal_ledger.terminal_electrical_export_W_m2
        - result.thermal_ledger.linear_heat_rejection_W_m2
        - result.thermal_ledger.radiative_heat_rejection_W_m2
    )
    protocol_alignment = all(
        item.source_experiment_protocol.temperature_K == item.temperature_K
        and not item.source_experiment_protocol.implicit_legacy_protocol
        and item.source_experiment_protocol.sha256
        == item.source_experiment_protocol_sha256
        for item in evaluations
    )
    all_evaluations_certified = all(
        item.source_certified and item.external_circuit_certified
        for item in evaluations
    )
    temperature_power_change = abs(
        maximum_evaluation.terminal_power_W_m2
        - ambient_evaluation.terminal_power_W_m2
    )

    return CellMeasurement.from_mapping(
        {
            "observables": {
                "ambient_to_maximum_mpp_power_change_W_m2": (
                    temperature_power_change
                ),
                "operating_temperature_K": result.operating_temperature_K,
                "temperature_rise_K": result.temperature_rise_K,
                "terminal_mpp_current_A_m2": result.terminal_current_A_m2,
                "terminal_mpp_power_W_m2": result.terminal_power_W_m2,
                "terminal_mpp_voltage_V": result.terminal_voltage_V,
            },
            "quality": {
                "all_temperature_evaluations_certified": float(
                    all_evaluations_certified
                ),
                "ambient_residual_positive": float(
                    ambient_evaluation.power_balance_residual_W_m2 > 0.0
                ),
                "base_stack_immutable": float(
                    float(stack.T) == initial_stack_temperature
                ),
                "circuit_protocol_hash_verified": float(
                    result.circuit_protocol_sha256 == circuit.sha256
                ),
                "electrical_protocol_hash_verified": float(
                    result.electrical_protocol_sha256 == electrical.sha256
                ),
                "electrothermal_certified": float(result.certified),
                "final_external_mapping_verified": float(
                    result.final_external_result.mapping_sha256
                    == next(
                        item.external_mapping_sha256
                        for item in evaluations
                        if item.temperature_K == result.operating_temperature_K
                    )
                ),
                "first_law_reconstruction_error_W_m2": abs(
                    first_law_reconstruction
                    - result.power_balance_residual_W_m2
                ),
                "maximum_residual_negative": float(
                    maximum_evaluation.power_balance_residual_W_m2 < 0.0
                ),
                "mpp_series_voltage_drop_V": abs(mpp_series_drop),
                "mpp_shunt_current_A_m2": abs(mpp_shunt_current),
                "operating_protocol_hash_verified": float(
                    result.operating_protocol_sha256 == operating.sha256
                ),
                "power_balance_residual_W_m2": abs(
                    result.power_balance_residual_W_m2
                ),
                "root_iterations": float(result.root_iterations),
                "temperature_evaluations": float(result.electrical_evaluations),
                "temperature_evaluations_within_envelope": float(
                    all(
                        thermal.ambient_temperature_K
                        <= item.temperature_K
                        <= thermal.maximum_temperature_K
                        for item in evaluations
                    )
                ),
                "temperature_protocols_aligned": float(protocol_alignment),
                "temperature_response_active_W_m2": temperature_power_change,
                "terminal_power_below_absorption": float(
                    result.terminal_power_W_m2
                    <= thermal.absorbed_optical_power_W_m2
                ),
                "thermal_protocol_hash_verified": float(
                    result.thermal_protocol_sha256 == thermal.sha256
                ),
            },
            "units": {
                "ambient_to_maximum_mpp_power_change_W_m2": "W m-2",
                "first_law_reconstruction_error_W_m2": "W m-2",
                "mpp_series_voltage_drop_V": "V",
                "mpp_shunt_current_A_m2": "A m-2",
                "operating_temperature_K": "K",
                "power_balance_residual_W_m2": "W m-2",
                "temperature_response_active_W_m2": "W m-2",
                "temperature_rise_K": "K",
                "terminal_mpp_current_A_m2": "A m-2",
                "terminal_mpp_power_W_m2": "W m-2",
                "terminal_mpp_voltage_V": "V",
            },
            "metadata": {
                **_protocol_metadata(study_protocol),
                "actual": {
                    "base_stack_sha256": result.base_stack_sha256,
                    "circuit_protocol_sha256": result.circuit_protocol_sha256,
                    "electrical_protocol": electrical.to_dict(),
                    "electrical_protocol_sha256": (
                        result.electrical_protocol_sha256
                    ),
                    "evaluation_source_result_sha256": [
                        item.source_result_sha256 for item in evaluations
                    ],
                    "evaluation_temperatures_K": [
                        item.temperature_K for item in evaluations
                    ],
                    "mapping_sha256": result.mapping_sha256,
                    "operating_protocol_sha256": result.operating_protocol_sha256,
                    "thermal_protocol_sha256": result.thermal_protocol_sha256,
                },
            },
        }
    )


__all__ = ["run_electrothermal_terminal_mpp_refinement"]
