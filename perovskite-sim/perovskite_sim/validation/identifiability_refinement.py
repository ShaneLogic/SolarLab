"""Refinement adapter for the synthetic interface-SRH identifiability slice."""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from perovskite_sim.experiments.identifiability import (
    PARAMETER_NAMES,
    InterfaceSRHIdentifiabilityProtocol,
    build_interface_srh_identifiability_protocol,
    run_interface_srh_identifiability,
)

from .dae_refinement import (
    _finite_option,
    _protocol_metadata,
    _string_option,
)
from .numerical_certificate import LaneDefinition, MatrixPoint
from .refinement_runner import CellMeasurement


_EXPECTED_OPTION_KEYS = {
    "base_finite_difference_step_log10",
    "calibration_field",
    "config_loader",
    "full_rank_estimated_parameters",
    "observable_family",
    "rank_deficient_estimated_parameters",
    "require_protocol",
    "synthetic_noise_sigma_multiplier",
    "target_interface",
}


def _finite_positive(value: object, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field} must be a finite positive number") from exc
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{field} must be a finite positive number")
    return number


def _load_interface_anchor(
    config_path: Path,
    *,
    target: str,
    calibration_field: str,
) -> dict[str, float | str]:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("identifiability anchor config must be a mapping")
    interfaces = payload.get("interfaces")
    if not isinstance(interfaces, list):
        raise ValueError("identifiability anchor config lacks interfaces")
    matches = [
        item
        for item in interfaces
        if isinstance(item, dict) and str(item.get("target")) == target
    ]
    if len(matches) != 1:
        raise ValueError("target interface must occur exactly once in anchor config")
    block = matches[0]
    required = (
        "sigma_n_cm2",
        "sigma_p_cm2",
        "N_t_cm2",
        "v_th_cm_s",
        calibration_field,
    )
    missing = [name for name in required if name not in block]
    if missing:
        raise ValueError(f"interface anchor is missing fields {missing}")
    return {
        "calibration_factor": _finite_positive(
            block[calibration_field], calibration_field
        ),
        "calibration_field": calibration_field,
        "sigma_n_cm2": _finite_positive(block["sigma_n_cm2"], "sigma_n_cm2"),
        "sigma_p_cm2": _finite_positive(block["sigma_p_cm2"], "sigma_p_cm2"),
        "target": target,
        "thermal_velocity_cm_s": _finite_positive(
            block["v_th_cm_s"], "v_th_cm_s"
        ),
        "trap_density_cm2": _finite_positive(block["N_t_cm2"], "N_t_cm2"),
    }


def _parameter_names(options: dict[str, Any], field: str) -> tuple[str, ...]:
    values = options.get(field)
    if not isinstance(values, (tuple, list)) or not values:
        raise ValueError(f"{field} must be a nonempty sequence")
    names = tuple(str(value) for value in values)
    if len(set(names)) != len(names) or any(name not in PARAMETER_NAMES for name in names):
        raise ValueError(f"{field} contains invalid or duplicate parameter names")
    return names


def _anchored_protocol(
    *,
    anchor: dict[str, float | str],
    estimated_parameters: tuple[str, ...],
    carrier_condition_count: int,
    finite_difference_step_log10: float,
    observable_family: str,
    synthetic_noise_sigma_multiplier: float,
) -> InterfaceSRHIdentifiabilityProtocol:
    protocol = build_interface_srh_identifiability_protocol(
        observable_family=observable_family,
        estimated_parameters=estimated_parameters,
        carrier_condition_count=carrier_condition_count,
        finite_difference_step_log10=finite_difference_step_log10,
        synthetic_noise_sigma_multiplier=synthetic_noise_sigma_multiplier,
    )
    truth_by_name = {
        "trap_density_cm2": math.log10(float(anchor["trap_density_cm2"])),
        "capture_cross_section_scale": 0.0,
        "calibration_factor": math.log10(float(anchor["calibration_factor"])),
    }
    parameters = tuple(
        dataclasses.replace(parameter, truth_log10=truth_by_name[parameter.name])
        for parameter in protocol.parameters
    )
    return dataclasses.replace(
        protocol,
        parameters=parameters,
        base_sigma_n_cm2=float(anchor["sigma_n_cm2"]),
        base_sigma_p_cm2=float(anchor["sigma_p_cm2"]),
        thermal_velocity_cm_s=float(anchor["thermal_velocity_cm_s"]),
    )


def _study_protocol(
    lane: LaneDefinition,
    *,
    anchor: dict[str, float | str],
    rank_deficient_parameters: tuple[str, ...],
    full_rank_parameters: tuple[str, ...],
    base_finite_difference_step_log10: float,
    observable_family: str,
) -> dict[str, Any]:
    return {
        "anchor": {
            "config_path": lane.config_path,
            **anchor,
            "role": "formula_input_anchor_not_material_truth",
        },
        "failure_policy": "penalize_and_invalidate",
        "forward_model": {
            "charge": "-q*N_t*(f-f_eq)",
            "kinetics": "v=sigma*v_th*N_t*calibration_factor",
            "rate": "production_interface_recombination",
        },
        "matrix": {
            "base_finite_difference_step_log10": (
                base_finite_difference_step_log10
            ),
            "carrier_condition_counts": list(lane.grid_values),
            "finite_difference_step_factors": list(lane.tolerance_factors),
            "grid_parameter": lane.grid_parameter,
            "tolerance_parameter": lane.tolerance_parameter,
        },
        "observable_family": observable_family,
        "scenarios": {
            "full_rank_known_capture_scale": {
                "estimated_parameters": list(full_rank_parameters),
                "required_rank": len(full_rank_parameters),
            },
            "rank_deficient_all_free": {
                "estimated_parameters": list(rank_deficient_parameters),
                "required_rank": 2,
            },
        },
        "schema_version": "interface-srh-identifiability-refinement-protocol-v1",
    }


def _normalized_nonzero_singular_values(result) -> np.ndarray:
    values = np.asarray(result.singular_values[: result.numerical_rank], dtype=float)
    return values / math.sqrt(len(result.observable_labels))


def run_interface_srh_identifiability_refinement(
    lane: LaneDefinition,
    point: MatrixPoint,
    project_root: Path,
) -> CellMeasurement:
    """Run one carrier-condition/finite-difference refinement cell."""

    options = lane.options
    if set(options) != _EXPECTED_OPTION_KEYS:
        raise ValueError(
            "identifiability options do not match the registered schema; "
            f"missing={sorted(_EXPECTED_OPTION_KEYS - set(options))}, "
            f"extra={sorted(set(options) - _EXPECTED_OPTION_KEYS)}"
        )
    if _string_option(options, "config_loader", "") != "raw_interface_anchor":
        raise ValueError("identifiability lane requires raw_interface_anchor")
    if options.get("require_protocol") is not True:
        raise ValueError("identifiability lane requires an explicit protocol")
    target = _string_option(options, "target_interface", "")
    calibration_field = _string_option(options, "calibration_field", "")
    observable_family = _string_option(options, "observable_family", "")
    rank_deficient_parameters = _parameter_names(
        options, "rank_deficient_estimated_parameters"
    )
    full_rank_parameters = _parameter_names(
        options, "full_rank_estimated_parameters"
    )
    if rank_deficient_parameters != PARAMETER_NAMES:
        raise ValueError("rank-deficient scenario must estimate all three parameters")
    if full_rank_parameters != ("trap_density_cm2", "calibration_factor"):
        raise ValueError("full-rank scenario must hold capture scale fixed")

    base_step = _finite_option(
        options,
        "base_finite_difference_step_log10",
        1.0e-3,
    )
    noise = _finite_option(
        options,
        "synthetic_noise_sigma_multiplier",
        0.0,
        positive=False,
    )
    if noise < 0.0:
        raise ValueError("synthetic noise multiplier must be non-negative")
    finite_difference_step = base_step * point.tolerance_factor
    anchor = _load_interface_anchor(
        project_root / lane.config_path,
        target=target,
        calibration_field=calibration_field,
    )
    rank_deficient_protocol = _anchored_protocol(
        anchor=anchor,
        estimated_parameters=rank_deficient_parameters,
        carrier_condition_count=point.grid,
        finite_difference_step_log10=finite_difference_step,
        observable_family=observable_family,
        synthetic_noise_sigma_multiplier=noise,
    )
    full_rank_protocol = _anchored_protocol(
        anchor=anchor,
        estimated_parameters=full_rank_parameters,
        carrier_condition_count=point.grid,
        finite_difference_step_log10=finite_difference_step,
        observable_family=observable_family,
        synthetic_noise_sigma_multiplier=noise,
    )
    rank_deficient = run_interface_srh_identifiability(rank_deficient_protocol)
    full_rank = run_interface_srh_identifiability(full_rank_protocol)
    if rank_deficient.numerical_rank != 2 or full_rank.numerical_rank != 2:
        raise RuntimeError("identifiability scenarios returned unexpected ranks")

    rank_nullspace = np.asarray(rank_deficient.nullspace_vectors[0], dtype=float)
    rank_singular = np.asarray(rank_deficient.singular_values, dtype=float)
    rank_nonzero = _normalized_nonzero_singular_values(rank_deficient)
    full_nonzero = _normalized_nonzero_singular_values(full_rank)
    full_truth_error = max(
        abs(value - parameter.truth_log10)
        for value, parameter in zip(
            full_rank.best_fit_log10,
            full_rank_protocol.estimated_parameters,
            strict=True,
        )
    )
    study_protocol = _study_protocol(
        lane,
        anchor=anchor,
        rank_deficient_parameters=rank_deficient_parameters,
        full_rank_parameters=full_rank_parameters,
        base_finite_difference_step_log10=base_step,
        observable_family=observable_family,
    )

    return CellMeasurement.from_mapping(
        {
            "observables": {
                "full_rank_best_fit_log10": full_rank.best_fit_log10,
                "full_rank_condition_number": full_rank.condition_number,
                "full_rank_normalized_singular_values": full_nonzero,
                "rank_deficient_absolute_nullspace_vector": np.abs(
                    rank_nullspace
                ),
                "rank_deficient_normalized_nonzero_singular_values": rank_nonzero,
            },
            "quality": {
                "all_multistarts_converged": float(
                    rank_deficient.all_multistarts_converged
                    and full_rank.all_multistarts_converged
                ),
                "all_profiles_completed": float(
                    rank_deficient.profiles_completed
                    and full_rank.profiles_completed
                ),
                "config_anchor_verified": 1.0,
                "forward_failure_count": float(
                    rank_deficient.forward_failure_count
                    + full_rank.forward_failure_count
                ),
                "full_rank_analysis_certified": float(
                    full_rank.analysis_certified
                ),
                "full_rank_condition_number": float(
                    full_rank.condition_number
                ),
                "full_rank_expected_rank": float(
                    full_rank.numerical_rank == full_rank_protocol.expected_rank
                ),
                "full_rank_parameters_identifiable": float(
                    full_rank.parameters_identifiable
                ),
                "full_rank_truth_error_log10": full_truth_error,
                "full_rank_truth_recovered": float(full_rank.truth_recovered),
                "rank_deficient_analysis_certified": float(
                    rank_deficient.analysis_certified
                ),
                "rank_deficient_capture_calibration_null_sum": abs(
                    rank_nullspace[1] + rank_nullspace[2]
                ),
                "rank_deficient_expected_rank": float(
                    rank_deficient.numerical_rank
                    == rank_deficient_protocol.expected_rank
                ),
                "rank_deficient_parameter_claim_absent": float(
                    not rank_deficient.parameters_identifiable
                ),
                "rank_deficient_smallest_to_largest_singular_ratio": float(
                    rank_singular[-1] / rank_singular[0]
                ),
                "rank_deficient_trap_null_component": abs(rank_nullspace[0]),
            },
            "units": {
                "all_multistarts_converged": "1",
                "all_profiles_completed": "1",
                "config_anchor_verified": "1",
                "forward_failure_count": "1",
                "full_rank_analysis_certified": "1",
                "full_rank_best_fit_log10": "log10",
                "full_rank_condition_number": "1",
                "full_rank_expected_rank": "1",
                "full_rank_normalized_singular_values": "1",
                "full_rank_parameters_identifiable": "1",
                "full_rank_truth_error_log10": "log10",
                "full_rank_truth_recovered": "1",
                "rank_deficient_absolute_nullspace_vector": "1",
                "rank_deficient_analysis_certified": "1",
                "rank_deficient_capture_calibration_null_sum": "1",
                "rank_deficient_expected_rank": "1",
                "rank_deficient_normalized_nonzero_singular_values": "1",
                "rank_deficient_parameter_claim_absent": "1",
                "rank_deficient_smallest_to_largest_singular_ratio": "1",
                "rank_deficient_trap_null_component": "1",
            },
            "metadata": {
                **_protocol_metadata(study_protocol),
                "actual": {
                    "carrier_condition_count": point.grid,
                    "finite_difference_step_log10": finite_difference_step,
                    "full_rank_mapping_sha256": full_rank.mapping_sha256,
                    "full_rank_protocol_sha256": full_rank.protocol_sha256,
                    "rank_deficient_mapping_sha256": (
                        rank_deficient.mapping_sha256
                    ),
                    "rank_deficient_protocol_sha256": (
                        rank_deficient.protocol_sha256
                    ),
                },
            },
        }
    )


__all__ = ["run_interface_srh_identifiability_refinement"]
