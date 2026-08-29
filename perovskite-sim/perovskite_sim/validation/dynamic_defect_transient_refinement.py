"""Source-bound D6-E3c refinement for interface traps plus mobile ions."""

from __future__ import annotations

import dataclasses
from dataclasses import replace
import math
from numbers import Integral
from pathlib import Path
import re
from typing import Any

import numpy as np

from perovskite_sim.discretization.grid import Layer, multilayer_grid
from perovskite_sim.experiments.dynamic_defect_transient import (
    ALLOWED_TIME_STEP_REFINEMENT_FACTORS,
    DYNAMIC_DEFECT_TRANSIENT_EVIDENCE,
    DYNAMIC_DEFECT_TRANSIENT_METHOD,
    DYNAMIC_DEFECT_TRANSIENT_REFERENCE_CERTIFICATE_SHA256,
    DYNAMIC_DEFECT_TRANSIENT_REFERENCE_LANE,
    DYNAMIC_DEFECT_TRANSIENT_SCHEMA,
    DynamicDefectTransientCertificationError,
    DynamicDefectTransientProtocol,
    build_dynamic_defect_transient_protocol,
    default_dynamic_defect_transient_policy,
    run_dynamic_defect_transient,
)
from perovskite_sim.experiments.interface_defect_ion_transient import (
    InterfaceDefectIonTransientError,
    InterfaceDefectIonTransientPolicy,
    run_interface_defect_ion_device_transient,
)
from perovskite_sim.experiments.quasi_fermi_steady_state import (
    build_two_sided_trace_grid,
)
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.models.device import DeviceStack
from perovskite_sim.physics.generation import dual_cell_widths

from .numerical_certificate import LaneDefinition, MatrixPoint, content_sha256
from .refinement_runner import CellMeasurement


_CASE_COMBINED = "combined"
_CASE_DEFECT_DOMINATED = "defect_dominated"
_CASE_ION_DOMINATED = "ion_dominated"
_CASE_NAMES = (
    _CASE_COMBINED,
    _CASE_DEFECT_DOMINATED,
    _CASE_ION_DOMINATED,
)
_LEGACY_LANE_ID = "dynamic-defect-ion-transient-timescale-v1"
_RESOLVED_V2_LANE_ID = "dynamic-defect-ion-transient-timescale-resolved-v2"
_NONLINEAR_V3_LANE_ID = "dynamic-defect-ion-transient-timescale-nonlinear-resolved-v3"
_ABSORBER_V4_LANE_ID = "dynamic-defect-ion-transient-timescale-absorber-resolved-v4"
_LANE_ID = "dynamic-defect-ion-transient-timescale-reference-resolved-v5"
_PRODUCTION_LANE_ID = "dynamic-defect-ion-transient-production-v1"
_GRID_PARAMETER = "intervals_per_layer"
_TOLERANCE_PARAMETER = "backward_euler_time_step_factor"
_GRID_VALUES = (4, 6, 8)
_TOLERANCE_FACTORS = (1.0, 0.5, 0.25)
_COMMON_OPTION_KEYS = frozenset(
    {
        "base_nested_substeps",
        "config_loader",
        "fast_ion_diffusivity_m2_s",
        "grid_alpha",
        "require_protocol",
        "slow_capture_scale",
        "slow_ion_diffusivity_m2_s",
        "times_s",
        "voltage_V",
    }
)
_V2_OPTION_KEYS = _COMMON_OPTION_KEYS | {
    "maximum_line_search_steps",
    "maximum_newton_iterations",
}
_V3_OPTION_KEYS = _V2_OPTION_KEYS | {
    "maximum_near_acceptance_nonmonotone_steps",
}
_PRODUCTION_OPTION_KEYS = _V3_OPTION_KEYS | {"production_method"}
_LANE_CONTRACTS = {
    _LEGACY_LANE_ID: ("v1", _COMMON_OPTION_KEYS),
    _RESOLVED_V2_LANE_ID: ("v2", _V2_OPTION_KEYS),
    _NONLINEAR_V3_LANE_ID: ("v3", _V3_OPTION_KEYS),
    _ABSORBER_V4_LANE_ID: ("v4", _V3_OPTION_KEYS),
    _LANE_ID: ("v5", _V3_OPTION_KEYS),
    _PRODUCTION_LANE_ID: ("v6", _PRODUCTION_OPTION_KEYS),
}
_LINE_SEARCH_ITERATION = re.compile(
    r"line search stalled at iteration ([1-9][0-9]*) with residual"
)
_NEWTON_LIMIT_ITERATION = re.compile(
    r"Newton exceeded ([1-9][0-9]*) iterations with residual"
)
_NONLINEAR_FAILURE_RESIDUAL = re.compile(
    r"(?:iteration [1-9][0-9]*|iterations) with residual "
    r"([+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?)"
)


def _validate_lane_contract(lane: LaneDefinition) -> None:
    contract = _LANE_CONTRACTS.get(lane.lane_id)
    if contract is None:
        raise ValueError(
            "D6-E3c executor requires a registered v1, resolved-v2, or "
            "nonlinear-resolved-v3, absorber-resolved-v4, "
            "reference-resolved-v5, or production-v1 lane"
        )
    expected_version, expected_options = contract
    if lane.executor_version != expected_version:
        raise ValueError(
            f"D6-E3c lane {lane.lane_id!r} requires executor_version="
            f"{expected_version!r}"
        )
    if lane.grid_parameter != _GRID_PARAMETER:
        raise ValueError(f"D6-E3c executor requires grid parameter {_GRID_PARAMETER!r}")
    if lane.tolerance_parameter != _TOLERANCE_PARAMETER:
        raise ValueError(
            f"D6-E3c executor requires tolerance parameter {_TOLERANCE_PARAMETER!r}"
        )
    if lane.grid_values != _GRID_VALUES:
        raise ValueError(f"D6-E3c executor requires grid values {_GRID_VALUES!r}")
    if lane.tolerance_factors != _TOLERANCE_FACTORS:
        raise ValueError(
            f"D6-E3c executor requires tolerance factors {_TOLERANCE_FACTORS!r}"
        )
    options = lane.options
    if set(options) != expected_options:
        raise ValueError(
            "D6-E3c lane options require an exact schema: "
            f"missing={sorted(expected_options - set(options))}, "
            f"extra={sorted(set(options) - expected_options)}"
        )
    if options.get("config_loader") != "standard":
        raise ValueError("D6-E3c requires config_loader='standard'")
    if options.get("require_protocol") is not True:
        raise ValueError("D6-E3c requires protocol provenance")
    if (
        lane.lane_id == _PRODUCTION_LANE_ID
        and options.get("production_method") != DYNAMIC_DEFECT_TRANSIENT_METHOD
    ):
        raise ValueError(
            "D6-E4 production lane requires the certified public transient method"
        )


def _option(
    options: dict[str, Any],
    name: str,
    expected_type: type,
    default: Any,
) -> Any:
    value = options.get(name, default)
    if (
        expected_type is float
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
    ):
        return float(value)
    if expected_type is int and isinstance(value, int) and not isinstance(value, bool):
        return value
    if expected_type is str and isinstance(value, str):
        return value
    raise ValueError(f"lane option {name!r} must be {expected_type.__name__}")


def _numeric_vector_option(
    options: dict[str, Any],
    name: str,
    default: tuple[float, ...],
) -> np.ndarray:
    raw = options.get(name, default)
    if isinstance(raw, (str, bytes)):
        raise ValueError(f"lane option {name!r} must be a numeric vector")
    try:
        result = np.asarray(tuple(float(item) for item in raw), dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"lane option {name!r} must be a numeric vector") from exc
    if result.ndim != 1 or result.size < 2 or not np.all(np.isfinite(result)):
        raise ValueError(f"lane option {name!r} must be a finite numeric vector")
    return result


def _integer_vector_option(
    options: dict[str, Any],
    name: str,
    default: tuple[int, ...],
) -> tuple[int, ...]:
    raw = options.get(name, default)
    if isinstance(raw, (str, bytes)):
        raise ValueError(f"lane option {name!r} must be an integer vector")
    try:
        values = tuple(raw)
    except TypeError as exc:
        raise ValueError(f"lane option {name!r} must be an integer vector") from exc
    if not values or any(
        isinstance(item, (bool, np.bool_)) or not isinstance(item, Integral)
        for item in values
    ):
        raise ValueError(f"lane option {name!r} must be an integer vector")
    return tuple(int(item) for item in values)


def _nested_substeps(
    tolerance_factor: float,
    base: tuple[int, ...] = (1, 2, 4),
) -> tuple[int, ...]:
    factor = float(tolerance_factor)
    if not math.isfinite(factor) or factor <= 0.0:
        raise ValueError("backward-Euler time-step factor must be positive")
    multiplier_float = 1.0 / factor
    multiplier = int(round(multiplier_float))
    if multiplier <= 0 or not math.isclose(
        multiplier_float,
        multiplier,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError(
            "backward-Euler time-step factor must have an integer reciprocal"
        )
    values = tuple(int(item) * multiplier for item in base)
    if (
        len(values) < 2
        or any(item <= 0 for item in values)
        or any(right % left != 0 for left, right in zip(values, values[1:]))
    ):
        raise ValueError("base refinement substeps must form a nested ladder")
    return values


def _with_ion_diffusivity(stack: DeviceStack, diffusivity_m2_s: float) -> DeviceStack:
    value = float(diffusivity_m2_s)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("ion diffusivity override must be finite and positive")
    active = tuple(
        layer.params.P0 > 0.0 and layer.params.D_ion > 0.0 for layer in stack.layers
    )
    if not any(active):
        raise ValueError("ion diffusivity override requires an active source layer")
    return replace(
        stack,
        layers=tuple(
            replace(layer, params=replace(layer.params, D_ion=value))
            if enabled
            else layer
            for layer, enabled in zip(stack.layers, active)
        ),
    )


def _with_interface_capture_scale(stack: DeviceStack, scale: float) -> DeviceStack:
    factor = float(scale)
    if not math.isfinite(factor) or factor <= 0.0:
        raise ValueError("interface capture scale must be finite and positive")
    if len(stack.interface_defects) != 1 or len(stack.interfaces) != 1:
        raise ValueError("D6-E3c requires exactly one microscopic interface defect")
    defect = stack.interface_defects[0]
    if defect is None or defect.microscopic_document is None:
        raise ValueError("D6-E3c requires a canonical microscopic interface defect")
    document = defect.microscopic_document
    kinetics = replace(
        document.kinetics,
        sigma_n_m2=document.kinetics.sigma_n_m2 * factor,
        sigma_p_m2=document.kinetics.sigma_p_m2 * factor,
    )
    scaled = replace(document, kinetics=kinetics)
    return replace(
        stack,
        interfaces=(scaled.capture_velocities_m_s,),
        interface_defects=(replace(defect, microscopic_document=scaled),),
    )


def _case_stacks(
    source: DeviceStack,
    *,
    slow_ion_diffusivity_m2_s: float,
    slow_capture_scale: float,
) -> dict[str, DeviceStack]:
    return {
        _CASE_COMBINED: source,
        _CASE_DEFECT_DOMINATED: _with_ion_diffusivity(
            source,
            slow_ion_diffusivity_m2_s,
        ),
        _CASE_ION_DOMINATED: _with_interface_capture_scale(
            source,
            slow_capture_scale,
        ),
    }


def _case_identity(stack: DeviceStack) -> dict[str, Any]:
    documents = []
    for defect in stack.interface_defects:
        if defect is None or defect.microscopic_document is None:
            raise ValueError("case identity requires microscopic interface documents")
        documents.append(defect.microscopic_document.to_dict())
    return {
        "interface_documents": documents,
        "interface_document_sha256": [
            defect.microscopic_document.sha256
            for defect in stack.interface_defects
            if defect is not None and defect.microscopic_document is not None
        ],
        "interface_capture_velocities_m_s": [
            [float(pair[0]), float(pair[1])] for pair in stack.interfaces
        ],
        "ion_diffusivity_m2_s": [float(layer.params.D_ion) for layer in stack.layers],
        "negative_ion_diffusivity_m2_s": [
            float(layer.params.D_ion_neg) for layer in stack.layers
        ],
        "positive_ion_density_m3": [float(layer.params.P0) for layer in stack.layers],
        "positive_ion_site_limit_m3": [
            float(layer.params.P_lim) for layer in stack.layers
        ],
    }


def _source_case_identity_verified(
    source: DeviceStack,
    cases: dict[str, DeviceStack],
    identities: dict[str, dict[str, Any]],
    *,
    slow_ion_diffusivity_m2_s: float,
    slow_capture_scale: float,
) -> bool:
    if set(cases) != set(_CASE_NAMES):
        return False
    source_defect = source.interface_defects[0]
    if source_defect is None or source_defect.microscopic_document is None:
        return False
    source_document = source_defect.microscopic_document
    expected_defect_dominated = replace(
        source,
        layers=tuple(
            replace(
                layer,
                params=replace(
                    layer.params,
                    D_ion=(
                        slow_ion_diffusivity_m2_s
                        if layer.params.P0 > 0.0 and layer.params.D_ion > 0.0
                        else layer.params.D_ion
                    ),
                ),
            )
            for layer in source.layers
        ),
    )
    expected_kinetics = replace(
        source_document.kinetics,
        sigma_n_m2=source_document.kinetics.sigma_n_m2 * slow_capture_scale,
        sigma_p_m2=source_document.kinetics.sigma_p_m2 * slow_capture_scale,
    )
    expected_document = replace(source_document, kinetics=expected_kinetics)
    expected_ion_dominated = replace(
        source,
        interfaces=(expected_document.capture_velocities_m_s,),
        interface_defects=(
            replace(source_defect, microscopic_document=expected_document),
        ),
    )
    expected_cases = {
        _CASE_COMBINED: source,
        _CASE_DEFECT_DOMINATED: expected_defect_dominated,
        _CASE_ION_DOMINATED: expected_ion_dominated,
    }
    return cases == expected_cases and identities == {
        name: _case_identity(stack) for name, stack in expected_cases.items()
    }


def _line_search_iteration(message: str) -> int | None:
    match = _LINE_SEARCH_ITERATION.search(message)
    return None if match is None else int(match.group(1))


def _nonlinear_failure_iteration(message: str) -> int | None:
    line_search = _line_search_iteration(message)
    if line_search is not None:
        return line_search
    match = _NEWTON_LIMIT_ITERATION.search(message)
    return None if match is None else int(match.group(1))


def _nonlinear_failure_residual(message: str) -> float | None:
    match = _NONLINEAR_FAILURE_RESIDUAL.search(message)
    if match is None:
        return None
    value = float(match.group(1))
    return value if math.isfinite(value) else None


def _nonlinear_failure_outcome(message: str) -> str | None:
    if _LINE_SEARCH_ITERATION.search(message):
        return "typed_line_search_stall"
    if _NEWTON_LIMIT_ITERATION.search(message):
        return "typed_newton_iteration_limit"
    return None


def _build_grid(
    stack: DeviceStack,
    intervals_per_layer: int,
    grid_alpha: float,
) -> np.ndarray:
    intervals = int(intervals_per_layer)
    if intervals < 2:
        raise ValueError("D6-E3c requires at least two intervals per layer")
    alpha = float(grid_alpha)
    if not math.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("grid_alpha must be finite and positive")
    shared = multilayer_grid(
        [Layer(layer.thickness, intervals) for layer in stack.layers],
        alpha=tuple(alpha for _layer in stack.layers),
    )
    return build_two_sided_trace_grid(shared, stack)


def _ion_centroid_trace(result: Any, grid: np.ndarray) -> np.ndarray:
    nodes = np.asarray(result.ion_layout.positive_nodes, dtype=int)
    if nodes.size == 0:
        raise RuntimeError("positive-ion centroid requires an active component")
    density = np.asarray(result.positive_ion_density_m3, dtype=float)[:, nodes]
    widths = dual_cell_widths(grid)[nodes]
    coordinates = np.asarray(grid, dtype=float)[nodes]
    inventory = np.sum(density * widths[None, :], axis=1)
    if np.any(~np.isfinite(inventory)) or np.any(inventory <= 0.0):
        raise RuntimeError("positive-ion centroid requires finite positive inventory")
    return np.sum(density * coordinates[None, :] * widths[None, :], axis=1) / inventory


def _motion_metrics(result: Any) -> tuple[float, float]:
    occupancy = np.asarray(result.interface_occupancy, dtype=float)
    positive = np.asarray(result.positive_ion_density_m3, dtype=float)
    if np.any(positive[0] <= 0.0):
        raise RuntimeError("positive-ion motion requires a positive initial state")
    occupancy_motion = float(np.max(np.abs(occupancy - occupancy[0])))
    ion_motion = float(np.max(np.abs(positive / positive[0] - 1.0)))
    return occupancy_motion, ion_motion


def _execution_protocol(
    lane: LaneDefinition,
    *,
    times_s: np.ndarray,
    voltage_V: np.ndarray,
    policy: InterfaceDefectIonTransientPolicy,
    base_nested_substeps: tuple[int, ...],
    case_identities: dict[str, dict[str, Any]],
    slow_ion_diffusivity_m2_s: float,
    slow_capture_scale: float,
    fast_ion_diffusivity_m2_s: float,
) -> dict[str, Any]:
    common_policy = dataclasses.asdict(policy)
    common_policy.pop("refinement_substeps")
    if lane.lane_id in {_LANE_ID, _PRODUCTION_LANE_ID}:
        schema_version = (
            "dynamic-defect-ion-transient-production-refinement-protocol-v1"
            if lane.lane_id == _PRODUCTION_LANE_ID
            else "dynamic-defect-ion-transient-refinement-protocol-v2"
        )
        stiffness_boundary = {
            "accepted_outcomes": [
                "typed_line_search_stall",
                "typed_newton_iteration_limit",
            ],
            "ion_diffusivity_m2_s": fast_ion_diffusivity_m2_s,
            "iteration_and_residual_required": True,
            "not_a_certified_physical_solution": True,
        }
    else:
        schema_version = "dynamic-defect-ion-transient-refinement-protocol-v1"
        stiffness_boundary = {
            "expected_outcome": "typed_line_search_failure",
            "ion_diffusivity_m2_s": fast_ion_diffusivity_m2_s,
            "not_a_certified_physical_solution": True,
        }
    protocol = {
        "acceptance": {
            "matrix_observables": {
                gate.metric: gate.to_dict() for gate in lane.observables
            },
            "per_cell_quality": {
                gate.metric: gate.to_dict() for gate in lane.quality_gates
            },
        },
        "adapter": (
            "production-dynamic-defect-transient-public-wrapper-refinement"
            if lane.lane_id == _PRODUCTION_LANE_ID
            else "two-sided-interface-defect-ion-transient-timescale-refinement"
        ),
        "cases": {
            _CASE_COMBINED: {
                "capture_scale": 1.0,
                "ion_diffusivity_override_m2_s": None,
                "source_identity": case_identities[_CASE_COMBINED],
            },
            _CASE_DEFECT_DOMINATED: {
                "capture_scale": 1.0,
                "ion_diffusivity_override_m2_s": slow_ion_diffusivity_m2_s,
                "source_identity": case_identities[_CASE_DEFECT_DOMINATED],
            },
            _CASE_ION_DOMINATED: {
                "capture_scale": slow_capture_scale,
                "ion_diffusivity_override_m2_s": None,
                "source_identity": case_identities[_CASE_ION_DOMINATED],
            },
        },
        "history": {
            "times_s": times_s.tolist(),
            "voltage_V": voltage_V.tolist(),
            "voltage_interpolation": "right_continuous_step_and_hold",
        },
        "lane": {
            "config_path": lane.config_path,
            "config_sha256": lane.config_sha256,
            "definition_sha256": lane.definition_sha256,
            "executor_version": lane.executor_version,
            "lane_id": lane.lane_id,
        },
        "matrix_controls": {
            "grid_parameter": lane.grid_parameter,
            "grid_values": list(lane.grid_values),
            "nested_substeps_by_tolerance_factor": {
                format(factor, ".15g"): list(
                    _nested_substeps(factor, base_nested_substeps)
                )
                for factor in lane.tolerance_factors
            },
            "tolerance_factors": list(lane.tolerance_factors),
            "tolerance_parameter": lane.tolerance_parameter,
        },
        "schema_version": schema_version,
        "solver_policy_common": common_policy,
        "stiffness_boundary": stiffness_boundary,
    }
    if lane.lane_id == _PRODUCTION_LANE_ID:
        protocol["production_public_contract"] = {
            "allowed_time_step_refinement_factors": list(
                ALLOWED_TIME_STEP_REFINEMENT_FACTORS
            ),
            "evidence_schema": DYNAMIC_DEFECT_TRANSIENT_EVIDENCE,
            "method": DYNAMIC_DEFECT_TRANSIENT_METHOD,
            "protocol_schema": DYNAMIC_DEFECT_TRANSIENT_SCHEMA,
            "reference_certificate_sha256": (
                DYNAMIC_DEFECT_TRANSIENT_REFERENCE_CERTIFICATE_SHA256
            ),
            "reference_lane_id": DYNAMIC_DEFECT_TRANSIENT_REFERENCE_LANE,
            "required_projection": (
                "terminal/interface current, occupancy, positive-ion centroid, "
                "integrated charge, and physical state"
            ),
        }
    return protocol


def run_dynamic_defect_ion_transient_refinement(
    lane: LaneDefinition,
    point: MatrixPoint,
    project_root: Path,
) -> CellMeasurement:
    """Run one source-bound grid by backward-Euler refinement cell."""
    _validate_lane_contract(lane)
    options = lane.options
    source = load_device_from_yaml(project_root / lane.config_path)
    times = _numeric_vector_option(
        options,
        "times_s",
        (0.0, 1.0e-8, 1.0e-6, 1.0e-4),
    )
    voltage = _numeric_vector_option(
        options,
        "voltage_V",
        (0.0, 0.05, 0.05, 0.05),
    )
    if times.shape != voltage.shape or np.any(np.diff(times) <= 0.0):
        raise ValueError("transient time and voltage histories must align and increase")
    slow_ion = _option(options, "slow_ion_diffusivity_m2_s", float, 1.0e-20)
    slow_capture = _option(options, "slow_capture_scale", float, 1.0e-12)
    fast_ion = _option(options, "fast_ion_diffusivity_m2_s", float, 1.0e-12)
    grid_alpha = _option(options, "grid_alpha", float, 2.0)
    base_substeps = _integer_vector_option(
        options,
        "base_nested_substeps",
        (1, 2, 4),
    )
    policy_overrides: dict[str, Any] = {
        "refinement_substeps": _nested_substeps(
            point.tolerance_factor,
            base_substeps,
        )
    }
    if lane.lane_id in {
        _RESOLVED_V2_LANE_ID,
        _NONLINEAR_V3_LANE_ID,
        _ABSORBER_V4_LANE_ID,
        _LANE_ID,
        _PRODUCTION_LANE_ID,
    }:
        policy_overrides.update(
            maximum_line_search_steps=_option(
                options,
                "maximum_line_search_steps",
                int,
                40,
            ),
            maximum_newton_iterations=_option(
                options,
                "maximum_newton_iterations",
                int,
                100,
            ),
        )
    if lane.lane_id in {
        _NONLINEAR_V3_LANE_ID,
        _ABSORBER_V4_LANE_ID,
        _LANE_ID,
        _PRODUCTION_LANE_ID,
    }:
        policy_overrides["maximum_near_acceptance_nonmonotone_steps"] = _option(
            options,
            "maximum_near_acceptance_nonmonotone_steps",
            int,
            2,
        )
    policy = replace(InterfaceDefectIonTransientPolicy(), **policy_overrides)
    if lane.lane_id == _PRODUCTION_LANE_ID:
        public_policy = default_dynamic_defect_transient_policy(point.tolerance_factor)
        if policy != public_policy:
            raise ValueError(
                "D6-E4 registered solver controls do not match the public policy"
            )
        policy = public_policy
    grid = _build_grid(source, point.grid, grid_alpha)
    cases = _case_stacks(
        source,
        slow_ion_diffusivity_m2_s=slow_ion,
        slow_capture_scale=slow_capture,
    )
    identities = {name: _case_identity(stack) for name, stack in cases.items()}
    public_protocols: dict[str, DynamicDefectTransientProtocol] = {}
    if lane.lane_id == _PRODUCTION_LANE_ID:
        public_protocols = {
            name: build_dynamic_defect_transient_protocol(
                stack,
                grid,
                times,
                voltage,
                requested_grid_intervals=point.grid,
                time_step_refinement_factor=point.tolerance_factor,
            )
            for name, stack in cases.items()
        }
        results = {
            name: run_dynamic_defect_transient(
                grid,
                stack,
                public_protocols[name],
            )
            for name, stack in cases.items()
        }
    else:
        results = {
            name: run_interface_defect_ion_device_transient(
                grid,
                stack,
                times,
                voltage,
                policy=policy,
                require_certificate=True,
            )
            for name, stack in cases.items()
        }

    fast_stack = _with_ion_diffusivity(source, fast_ion)
    fast_failure_type = ""
    fast_failure_message = ""
    fast_failure_typed = False
    fast_line_search_stall = False
    fast_failure_iteration: int | None = None
    fast_nonlinear_outcome: str | None = None
    fast_nonlinear_iteration: int | None = None
    fast_nonlinear_residual: float | None = None
    fast_public_protocol: DynamicDefectTransientProtocol | None = None
    try:
        if lane.lane_id == _PRODUCTION_LANE_ID:
            fast_public_protocol = build_dynamic_defect_transient_protocol(
                fast_stack,
                grid,
                times,
                voltage,
                requested_grid_intervals=point.grid,
                time_step_refinement_factor=point.tolerance_factor,
            )
            run_dynamic_defect_transient(
                grid,
                fast_stack,
                fast_public_protocol,
            )
        else:
            run_interface_defect_ion_device_transient(
                grid,
                fast_stack,
                times,
                voltage,
                policy=policy,
                require_certificate=True,
            )
    except (
        DynamicDefectTransientCertificationError,
        InterfaceDefectIonTransientError,
    ) as exc:
        if lane.lane_id == _PRODUCTION_LANE_ID and not isinstance(
            exc, DynamicDefectTransientCertificationError
        ):
            raise
        if lane.lane_id != _PRODUCTION_LANE_ID and not isinstance(
            exc, InterfaceDefectIonTransientError
        ):
            raise
        fast_failure_type = type(exc).__name__
        fast_failure_message = str(exc)
        fast_failure_typed = True
        fast_line_search_stall = "line search stalled" in fast_failure_message
        fast_failure_iteration = _line_search_iteration(fast_failure_message)
        fast_nonlinear_outcome = _nonlinear_failure_outcome(fast_failure_message)
        fast_nonlinear_iteration = _nonlinear_failure_iteration(fast_failure_message)
        fast_nonlinear_residual = _nonlinear_failure_residual(fast_failure_message)

    traces: dict[str, dict[str, np.ndarray]] = {}
    motion: dict[str, tuple[float, float]] = {}
    certificates = []
    for name in _CASE_NAMES:
        result = results[name]
        if lane.lane_id == _PRODUCTION_LANE_ID:
            traces[name] = {
                "integrated_charge_C_m2": np.asarray(
                    result.integrated_charge_change_C_m2,
                ),
                "interface_occupancy": np.asarray(result.interface_occupancy).reshape(
                    -1
                ),
                "left_terminal_current_A_m2": np.asarray(
                    result.terminal_total_current_A_m2,
                ),
                "positive_ion_centroid_m": np.asarray(
                    result.positive_ion_centroid_m,
                ),
            }
            motion[name] = (
                result.evidence.maximum_interface_occupancy_motion,
                result.evidence.maximum_positive_ion_relative_motion,
            )
            certificates.append(result.evidence.engine_certificate)
        else:
            traces[name] = {
                "integrated_charge_C_m2": np.asarray(
                    result.integrated_free_interface_ion_charge_C_m2,
                ),
                "interface_occupancy": np.asarray(result.interface_occupancy).reshape(
                    -1
                ),
                "left_terminal_current_A_m2": np.asarray(
                    result.total_current_faces_A_m2[:, 0],
                ),
                "positive_ion_centroid_m": _ion_centroid_trace(result, grid),
            }
            motion[name] = _motion_metrics(result)
            certificates.append(result.certificate)

    protocol = _execution_protocol(
        lane,
        times_s=times,
        voltage_V=voltage,
        policy=policy,
        base_nested_substeps=base_substeps,
        case_identities=identities,
        slow_ion_diffusivity_m2_s=slow_ion,
        slow_capture_scale=slow_capture,
        fast_ion_diffusivity_m2_s=fast_ion,
    )
    protocol_hash = content_sha256(protocol)
    if lane.lane_id == _PRODUCTION_LANE_ID:
        protocol_identity_verified = all(
            result.protocol == public_protocols[name]
            and result.evidence.protocol == result.protocol
            and result.evidence.protocol_sha256 == result.protocol.protocol_hash
            and result.protocol.solver_policy == policy
            and np.array_equal(result.times_s, times)
            and np.array_equal(result.voltage_V, voltage)
            for name, result in results.items()
        )
    else:
        protocol_identity_verified = all(
            result.policy == policy
            and np.array_equal(result.times_s, times)
            and np.array_equal(result.voltage_V, voltage)
            for result in results.values()
        )
    source_identity_verified = _source_case_identity_verified(
        source,
        cases,
        identities,
        slow_ion_diffusivity_m2_s=slow_ion,
        slow_capture_scale=slow_capture,
    )
    public_protocol_identity_verified = False
    public_projection_certified = False
    reference_certificate_bound = False
    if lane.lane_id == _PRODUCTION_LANE_ID:
        public_protocol_identity_verified = all(
            DynamicDefectTransientProtocol.from_json(protocol.canonical_json())
            == protocol
            and results[name].protocol == protocol
            and results[name].evidence.protocol_sha256 == protocol.protocol_hash
            for name, protocol in public_protocols.items()
        )
        public_projection_certified = all(
            result.evidence.public_projection_certified and result.evidence.certified
            for result in results.values()
        )
        reference_certificate_bound = all(
            result.evidence.reference_lane_id == DYNAMIC_DEFECT_TRANSIENT_REFERENCE_LANE
            and result.evidence.reference_certificate_sha256
            == DYNAMIC_DEFECT_TRANSIENT_REFERENCE_CERTIFICATE_SHA256
            for result in results.values()
        )

    observables: dict[str, np.ndarray] = {}
    units: dict[str, str] = {}
    reference_relative_occupancy = lane.lane_id in {
        _LANE_ID,
        _PRODUCTION_LANE_ID,
    }
    reference_relative_ion_charge = lane.lane_id in {
        _ABSORBER_V4_LANE_ID,
        _LANE_ID,
        _PRODUCTION_LANE_ID,
    }
    for name in _CASE_NAMES:
        if reference_relative_occupancy:
            observables[f"{name}_interface_occupancy_change"] = (
                traces[name]["interface_occupancy"]
                - traces[name]["interface_occupancy"][0]
            )
        else:
            observables[f"{name}_interface_occupancy"] = traces[name][
                "interface_occupancy"
            ]
        observables[f"{name}_left_terminal_current_A_m2"] = traces[name][
            "left_terminal_current_A_m2"
        ]
        if reference_relative_ion_charge:
            observables[f"{name}_positive_ion_centroid_shift_m"] = (
                traces[name]["positive_ion_centroid_m"]
                - traces[name]["positive_ion_centroid_m"][0]
            )
            observables[f"{name}_integrated_charge_change_C_m2"] = (
                traces[name]["integrated_charge_C_m2"]
                - traces[name]["integrated_charge_C_m2"][0]
            )
        else:
            observables[f"{name}_positive_ion_centroid_m"] = traces[name][
                "positive_ion_centroid_m"
            ]
            observables[f"{name}_integrated_charge_C_m2"] = traces[name][
                "integrated_charge_C_m2"
            ]
        occupancy_metric = (
            f"{name}_interface_occupancy_change"
            if reference_relative_occupancy
            else f"{name}_interface_occupancy"
        )
        units.update(
            {
                occupancy_metric: "1",
                f"{name}_left_terminal_current_A_m2": "A m-2",
            }
        )
        if reference_relative_ion_charge:
            units[f"{name}_positive_ion_centroid_shift_m"] = "m"
            units[f"{name}_integrated_charge_change_C_m2"] = "C m-2"
        else:
            units[f"{name}_positive_ion_centroid_m"] = "m"
            units[f"{name}_integrated_charge_C_m2"] = "C m-2"

    quality = {
        "all_cases_certified": float(all(item.certified for item in certificates)),
        "analytic_sparse_jacobian_verified": float(
            all(
                item.sparse_linear_solver_used
                and item.analytic_jacobian_nnz < item.dense_jacobian_entries
                for item in certificates
            )
        ),
        "case_count": float(len(results)),
        "clipping_used": float(any(item.clipping_used for item in certificates)),
        "combined_interface_occupancy_motion": motion[_CASE_COMBINED][0],
        "combined_positive_ion_motion": motion[_CASE_COMBINED][1],
        "defect_dominated_interface_occupancy_motion": motion[_CASE_DEFECT_DOMINATED][
            0
        ],
        "defect_dominated_positive_ion_motion": motion[_CASE_DEFECT_DOMINATED][1],
        "fast_ion_stiffness_failed_closed": float(bool(fast_failure_type)),
        "fast_ion_stiffness_failure_typed": float(fast_failure_typed),
        "ion_dominated_interface_occupancy_motion": motion[_CASE_ION_DOMINATED][0],
        "ion_dominated_positive_ion_motion": motion[_CASE_ION_DOMINATED][1],
        "max_all_face_current_spread_relative": max(
            item.maximum_all_face_current_spread_relative for item in certificates
        ),
        "max_analytic_jacobian_column_relative_error": max(
            item.maximum_analytic_jacobian_column_relative_error
            for item in certificates
        ),
        "max_charge_balance_relative_error": max(
            item.maximum_charge_balance_relative_error for item in certificates
        ),
        "max_current_decomposition_relative_error": max(
            item.maximum_current_decomposition_relative_error for item in certificates
        ),
        "max_eliminated_operator_relative_error": max(
            item.maximum_eliminated_operator_relative_error for item in certificates
        ),
        "max_interface_total_current_relative_error": max(
            item.maximum_two_sided_interface_total_current_relative_error
            for item in certificates
        ),
        "max_ion_inventory_relative_drift": max(
            item.maximum_ion_inventory_relative_drift for item in certificates
        ),
        "max_local_carrier_normalized_residual": max(
            item.maximum_local_carrier_normalized_residual for item in certificates
        ),
        "max_local_gauss_normalized_residual": max(
            item.maximum_local_gauss_normalized_residual for item in certificates
        ),
        "max_poisson_residual_C_m2": max(
            item.maximum_poisson_residual_C_m2 for item in certificates
        ),
        "max_refinement_current_relative_change": max(
            item.maximum_refinement_current_relative_change for item in certificates
        ),
        "max_refinement_state_change": max(
            item.maximum_refinement_state_change for item in certificates
        ),
        "max_scaled_nonlinear_residual": max(
            item.maximum_scaled_nonlinear_residual for item in certificates
        ),
        "max_site_occupancy_fraction": max(
            item.maximum_site_occupancy_fraction for item in certificates
        ),
        "microscopic_binding_certified": float(
            all(item.microscopic_binding_certified for item in certificates)
        ),
        "operating_points_certified": float(
            all(
                item.dc_operating_point_certified and item.dark_reference_certified
                for item in certificates
            )
        ),
        "protocol_identity_verified": float(protocol_identity_verified),
        "source_case_identity_verified": float(source_identity_verified),
    }
    if lane.lane_id in {_LANE_ID, _PRODUCTION_LANE_ID}:
        quality.update(
            {
                "fast_ion_failure_is_declared_nonlinear_stall": float(
                    fast_nonlinear_outcome is not None
                ),
                "fast_ion_failure_iteration_reported": float(
                    fast_nonlinear_iteration is not None
                ),
                "fast_ion_failure_iteration_within_policy": float(
                    fast_nonlinear_iteration is not None
                    and 1
                    <= fast_nonlinear_iteration
                    <= policy.maximum_newton_iterations
                ),
                "fast_ion_failure_residual_above_acceptance": float(
                    fast_nonlinear_residual is not None
                    and fast_nonlinear_residual
                    > policy.maximum_scaled_nonlinear_residual
                ),
            }
        )
    else:
        quality.update(
            {
                "fast_ion_failure_is_line_search_stall": float(fast_line_search_stall),
                "fast_ion_failure_iteration_reported": float(
                    fast_failure_iteration is not None
                ),
                "fast_ion_failure_iteration_within_policy": float(
                    fast_failure_iteration is not None
                    and 1 <= fast_failure_iteration <= policy.maximum_newton_iterations
                ),
            }
        )
    if lane.lane_id in {
        _NONLINEAR_V3_LANE_ID,
        _ABSORBER_V4_LANE_ID,
        _LANE_ID,
        _PRODUCTION_LANE_ID,
    }:
        quality["max_near_acceptance_nonmonotone_step_count"] = float(
            max(item.near_acceptance_nonmonotone_step_count for item in certificates)
        )
    if lane.lane_id == _PRODUCTION_LANE_ID:
        quality.update(
            {
                "public_projection_certified": float(public_projection_certified),
                "public_protocol_identity_verified": float(
                    public_protocol_identity_verified
                ),
                "reference_certificate_bound": float(reference_certificate_bound),
            }
        )
    units.update(
        {
            "max_poisson_residual_C_m2": "C m-2",
        }
    )
    fast_boundary_metadata = {
        "error_message": fast_failure_message,
        "error_type": fast_failure_type,
        "ion_diffusivity_m2_s": fast_ion,
        "newton_iteration": fast_failure_iteration,
        "nonlinear_iteration": fast_nonlinear_iteration,
        "nonlinear_outcome": fast_nonlinear_outcome,
        "nonlinear_residual": fast_nonlinear_residual,
    }
    metadata = {
        "actual_intervals_per_layer": point.grid,
        "actual_nodes": int(grid.size),
        "case_certificates": {
            name: dataclasses.asdict(
                results[name].evidence.engine_certificate
                if lane.lane_id == _PRODUCTION_LANE_ID
                else results[name].certificate
            )
            for name in _CASE_NAMES
        },
        "case_identities": identities,
        "fast_ion_stiffness_boundary": fast_boundary_metadata,
        "grid_m": grid.tolist(),
        "matrix_tolerance_factor": point.tolerance_factor,
        "motion": {
            name: {
                "interface_occupancy": values[0],
                "positive_ion": values[1],
            }
            for name, values in motion.items()
        },
        "policy": dataclasses.asdict(policy),
        "protocol": protocol,
        "protocol_hash": protocol_hash,
        "protocol_schema": protocol["schema_version"],
        "source_config_sha256": lane.config_sha256,
        "traces": {
            name: {key: value.tolist() for key, value in values.items()}
            for name, values in traces.items()
        },
    }
    if lane.lane_id == _PRODUCTION_LANE_ID:
        if fast_public_protocol is None:  # pragma: no cover - guarded by build above
            raise RuntimeError(
                "D6-E4 production stiffness probe did not bind a public protocol"
            )
        fast_boundary_metadata.update(
            public_protocol=fast_public_protocol.to_dict(),
            public_protocol_sha256=fast_public_protocol.protocol_hash,
        )
        metadata.update(
            public_evidence={
                name: dataclasses.asdict(results[name].evidence) for name in _CASE_NAMES
            },
            public_protocol_hashes={
                name: public_protocols[name].protocol_hash for name in _CASE_NAMES
            },
            public_protocols={
                name: public_protocols[name].to_dict() for name in _CASE_NAMES
            },
        )
    return CellMeasurement.from_mapping(
        {
            "observables": observables,
            "quality": quality,
            "units": units,
            "metadata": metadata,
        }
    )


__all__ = ["run_dynamic_defect_ion_transient_refinement"]
