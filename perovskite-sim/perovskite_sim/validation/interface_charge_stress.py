"""Registered device-stress adapter for equilibrium-referenced interface charge."""

from __future__ import annotations

import dataclasses
from dataclasses import replace
import math
from pathlib import Path
from typing import Any

import numpy as np

from perovskite_sim.constants import Q
from perovskite_sim.experiments.jv_sweep import build_electrical_grid
from perovskite_sim.experiments.quasi_fermi_steady_state import (
    DEFAULT_ILLUMINATION_STEPS,
    build_equilibrium_referenced_interface_charge_dark_reference,
    build_two_sided_trace_grid,
    solve_equilibrium_referenced_interface_charge_steady_state,
)
from perovskite_sim.models.device import (
    MicroscopicInterfaceDefectContractError,
    bind_uncalibrated_microscopic_interface_defects,
    electrical_interface_defects,
)
from perovskite_sim.models.interface_defects import (
    EXPLICIT_INTERFACE_DEFECT_SCHEMA_VERSION,
)
from perovskite_sim.physics.contacts import (
    require_contact_thermodynamic_certificate,
)
from perovskite_sim.physics.interface_plane import FERMI_DIRAC_RICHARDSON
from perovskite_sim.physics.two_sided_interface import TWO_SIDED_TRACE
from perovskite_sim.solver.mol import build_material_arrays
from perovskite_sim.sweeps.device_parameter_sweep import (
    SweepPoint,
    apply_sweep_point,
    describe_stack,
)

from .numerical_certificate import LaneDefinition, MatrixPoint, content_sha256
from .refinement_executors import _load_stack
from .refinement_runner import CellMeasurement


_AXIS_UPDATE_KEY = {
    "baseline": None,
    "trap_energy": "interface_defect_E_t_eV",
    "conduction_band_offset": "etl_delta_ec_eV",
    "etl_doping": "etl_doping_cm3",
    "trap_density": "interface_defect_N_t_cm2",
}
_TARGETS = (
    ("dark_bias", 0.05, False),
    ("illuminated_operating_point", 0.0, True),
)
_DARK_IDENTITY_FIELDS = (
    "y",
    "phi",
    "electron_quasi_fermi_potential_V",
    "hole_quasi_fermi_potential_V",
    "electron_face_current_A_m2",
    "hole_face_current_A_m2",
    "total_face_current_A_m2",
    "electron_rate_per_s",
    "hole_rate_per_s",
)


def _number_option(
    options: dict[str, Any],
    name: str,
    default: float,
) -> float:
    value = options.get(name, default)
    if isinstance(value, bool):
        raise ValueError(f"lane option {name!r} must be finite")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"lane option {name!r} must be finite")
    return number


def _integer_option(
    options: dict[str, Any],
    name: str,
    default: int,
) -> int:
    value = options.get(name, default)
    if isinstance(value, bool) or int(value) != value or int(value) <= 0:
        raise ValueError(f"lane option {name!r} must be a positive integer")
    return int(value)


def _refine_finite_difference_step(options: dict[str, Any]) -> bool:
    value = options.get("refine_finite_difference_step", True)
    if not isinstance(value, bool):
        raise ValueError("refine_finite_difference_step must be boolean")
    return value


def _stress_points(options: dict[str, Any]) -> tuple[SweepPoint, ...]:
    raw_points = options.get("stress_points")
    if not isinstance(raw_points, list) or len(raw_points) != 9:
        raise ValueError("stress_points must contain exactly nine mappings")
    points: list[SweepPoint] = []
    for index, raw in enumerate(raw_points):
        if not isinstance(raw, dict) or set(raw) != {
            "point_id",
            "axis",
            "label",
            "updates",
        }:
            raise ValueError(f"stress_points[{index}] has an invalid schema")
        updates = raw["updates"]
        if not isinstance(updates, dict):
            raise ValueError(f"stress_points[{index}].updates must be a mapping")
        axis = str(raw["axis"])
        if axis not in _AXIS_UPDATE_KEY:
            raise ValueError(f"stress_points[{index}] has an unknown axis")
        expected_key = _AXIS_UPDATE_KEY[axis]
        update_keys = set(updates) - {"interface_defect_target"}
        if expected_key is None:
            if updates:
                raise ValueError("the baseline stress point must have no updates")
        elif update_keys != {expected_key}:
            raise ValueError(
                f"stress axis {axis!r} must update exactly {expected_key!r}"
            )
        for key, value in updates.items():
            if key == "interface_defect_target":
                if not isinstance(value, str) or not value:
                    raise ValueError("interface_defect_target must be a string")
                continue
            if isinstance(value, bool) or not math.isfinite(float(value)):
                raise ValueError(f"stress update {key!r} must be finite")
        points.append(
            SweepPoint(
                point_id=str(raw["point_id"]),
                axis=axis,
                label=str(raw["label"]),
                updates=dict(updates),
            )
        )
    point_ids = [point.point_id for point in points]
    if len(point_ids) != len(set(point_ids)):
        raise ValueError("stress point IDs must be unique")
    counts = {axis: 0 for axis in _AXIS_UPDATE_KEY}
    for point in points:
        counts[point.axis] += 1
    if counts != {
        "baseline": 1,
        "trap_energy": 2,
        "conduction_band_offset": 2,
        "etl_doping": 2,
        "trap_density": 2,
    }:
        raise ValueError("stress_points must contain baseline plus two points per axis")
    return tuple(points)


def _require_research_stack(stack: object):
    violations: list[str] = []
    if stack.interface_charge_closure != "equilibrium_referenced":
        violations.append("equilibrium_referenced charge closure is required")
    if not stack.interface_charge_rebaseline_acknowledged:
        violations.append("rebaseline acknowledgement is required")
    if stack.het_recomb_despike != 0.0:
        violations.append("recombination de-spiking must be disabled")
    if stack.flat_band_contacts or stack.flat_band_metal_contacts:
        violations.append("calibrated flat-band contacts must be disabled")
    if stack.contact_phi_B_eV != 0.0:
        violations.append("the calibrated contact barrier must be zero")
    if getattr(stack, "autoloop_generated_lever", False):
        violations.append("autoloop-generated calibration is forbidden")
    bound_stack = None
    microscopic_contract = None
    try:
        bound_stack, microscopic_contract = (
            bind_uncalibrated_microscopic_interface_defects(
                stack,
                consumer="equilibrium-referenced interface-charge stress",
            )
        )
    except MicroscopicInterfaceDefectContractError as exc:
        violations.append(str(exc))
    if violations:
        raise RuntimeError("; ".join(violations))
    assert bound_stack is not None and microscopic_contract is not None
    return electrical_interface_defects(stack), bound_stack, microscopic_contract


def _solver_controls(
    options: dict[str, Any],
    tolerance_factor: float,
) -> tuple[dict[str, float | int], dict[str, float | int]]:
    base = {
        "finite_difference_step": _number_option(
            options, "base_finite_difference_step", 1.0e-5
        ),
        "newton_residual_tolerance": _number_option(
            options, "base_newton_residual_tolerance", 4.0e-7
        ),
        "max_newton_iterations": _integer_option(
            options, "max_newton_iterations", 60
        ),
        "poisson_tolerance_V": _number_option(
            options, "base_poisson_tolerance_V", 1.0e-12
        ),
        "poisson_max_iterations": _integer_option(
            options, "poisson_max_iterations", 100
        ),
        "continuity_tolerance_A_m2": _number_option(
            options, "continuity_tolerance_A_m2", 1.0e-4
        ),
        "current_spread_tolerance_A_m2": _number_option(
            options, "current_spread_tolerance_A_m2", 1.0e-4
        ),
        "poisson_residual_tolerance": _number_option(
            options, "poisson_residual_tolerance", 1.0e-8
        ),
    }
    positive = tuple(float(value) for value in base.values())
    if any(not math.isfinite(value) or value <= 0.0 for value in positive):
        raise ValueError("all interface-charge solver controls must be positive")
    controls = dict(base)
    if _refine_finite_difference_step(options):
        controls["finite_difference_step"] = float(
            base["finite_difference_step"]
        ) * math.sqrt(tolerance_factor)
    controls["newton_residual_tolerance"] = float(
        base["newton_residual_tolerance"]
    ) * tolerance_factor
    controls["poisson_tolerance_V"] = float(
        base["poisson_tolerance_V"]
    ) * tolerance_factor
    return base, controls


def _protocol(
    points: tuple[SweepPoint, ...],
    base_controls: dict[str, float | int],
    *,
    refine_finite_difference_step: bool,
) -> dict[str, Any]:
    return {
        "acceptance": {
            "charge_law": "-q*N_t*(f-f_eq)",
            "continuity_tolerance_A_m2": base_controls[
                "continuity_tolerance_A_m2"
            ],
            "current_spread_tolerance_A_m2": base_controls[
                "current_spread_tolerance_A_m2"
            ],
            "local_interface_residual_limit": 1.0e-7,
            "normalized_gauss_residual_limit": 1.0e-10,
            "poisson_residual_tolerance": base_controls[
                "poisson_residual_tolerance"
            ],
            "require_charge_barrier_sign_consistency": True,
            "require_contact_thermodynamic_certificate": True,
            "require_dark_charge_off_bit_identity": True,
        },
        "adapter": "equilibrium-referenced-interface-charge-device-stress",
        "dark_reference": {
            "charge_closure": "off",
            "rebuild_for_each_device_point": True,
            "voltage_V": 0.0,
        },
        "interface": {
            "cross_transmission": 1.0,
            "energy_reference": "below_local_conduction_band",
            "kinetics_source": "canonical_microscopic_interface_defect_document",
            "microscopic_schema_version": (
                EXPLICIT_INTERFACE_DEFECT_SCHEMA_VERSION
            ),
            "require_exact_compatibility_srv_identity": True,
            "require_unity_calibration": True,
            "topology": TWO_SIDED_TRACE,
            "transport_model": FERMI_DIRAC_RICHARDSON,
        },
        "measurement": "one_factor_at_a_time_device_stress",
        "schema_version": (
            "interface-charge-device-stress-protocol-v2"
            if refine_finite_difference_step
            else "interface-charge-device-stress-protocol-v3"
        ),
        "solver": {
            "base_controls": dict(base_controls),
            "illumination_steps": list(DEFAULT_ILLUMINATION_STEPS),
            "refinement_factor_source": "matrix.tolerance_factor",
            "refinement_mapping": {
                "finite_difference_step": (
                    "base*sqrt(factor)"
                    if refine_finite_difference_step
                    else "fixed_base"
                ),
                "newton_residual_tolerance": "base*factor",
                "poisson_tolerance_V": "base*factor",
            },
        },
        "stress_points": [
            {
                "axis": point.axis,
                "label": point.label,
                "point_id": point.point_id,
                "updates": dict(point.updates),
            }
            for point in points
        ],
        "targets": [
            {"illuminated": illuminated, "label": label, "voltage_V": voltage}
            for label, voltage, illuminated in _TARGETS
        ],
    }


def _point_applied(
    point: SweepPoint,
    stack: object,
    defects: tuple[object, ...],
) -> bool:
    expected_key = _AXIS_UPDATE_KEY[point.axis]
    if expected_key is None:
        return True
    expected = float(point.updates[expected_key])
    derived = describe_stack(stack)
    if point.axis == "trap_energy":
        actual = float(defects[-1].E_t_eV)
    elif point.axis == "conduction_band_offset":
        actual = float(derived["etl_delta_ec_eV"])
    elif point.axis == "etl_doping":
        actual = float(derived["etl_N_D_cm3"])
    else:
        actual = float(defects[-1].N_t_cm2)
    return bool(np.isclose(actual, expected, rtol=1.0e-12, atol=1.0e-15))


def _solve_variant(
    stack: object,
    point: SweepPoint,
    grid_intervals: int,
    controls: dict[str, float | int],
) -> dict[str, Any]:
    defects, bound_stack, microscopic_contract = _require_research_stack(stack)
    shared_grid = build_electrical_grid(stack, grid_intervals)
    grid = build_two_sided_trace_grid(shared_grid, stack)
    charge_off_stack = replace(bound_stack, interface_charge_closure="off")
    material = build_material_arrays(grid, charge_off_stack)
    if material.iface_state_charge != 0.0:
        raise RuntimeError("legacy shared-node interface charge must remain zero")
    contact = require_contact_thermodynamic_certificate(charge_off_stack, material)
    reference = build_equilibrium_referenced_interface_charge_dark_reference(
        grid,
        stack,
        interface_transmission=1.0,
        **controls,
    )
    charged_dark = solve_equilibrium_referenced_interface_charge_steady_state(
        grid,
        stack,
        0.0,
        dark_reference=reference,
        illuminated=False,
        **controls,
    )
    results = tuple(
        solve_equilibrium_referenced_interface_charge_steady_state(
            grid,
            stack,
            voltage,
            dark_reference=reference,
            illuminated=illuminated,
            **controls,
        )
        for _label, voltage, illuminated in _TARGETS
    )
    interface_count = len(reference.equilibrium_occupancy)
    if interface_count == 0 or interface_count != len(defects):
        raise RuntimeError("stress interface evidence is not defect-aligned")
    equilibrium = np.asarray(reference.equilibrium_occupancy, dtype=float)
    density = np.asarray(reference.trap_density_m2, dtype=float)
    occupancy = np.asarray(
        [result.interface_occupancy for result in results], dtype=float
    )
    charge = np.asarray(
        [result.interface_incremental_sheet_charge_C_m2 for result in results],
        dtype=float,
    )
    trace_shift = np.asarray(
        [result.interface_trace_potential_shift_V for result in results],
        dtype=float,
    )
    gauss = np.asarray(
        [result.interface_normalized_gauss_residual for result in results],
        dtype=float,
    )
    condition = np.asarray(
        [result.interface_scaled_local_jacobian_condition for result in results],
        dtype=float,
    )
    if (
        equilibrium.shape != (interface_count,)
        or density.shape != (interface_count,)
        or occupancy.shape != (len(_TARGETS), interface_count)
        or charge.shape != occupancy.shape
        or gauss.shape != occupancy.shape
        or condition.shape != occupancy.shape
        or trace_shift.shape != (len(_TARGETS), interface_count, 2)
    ):
        raise RuntimeError("stress interface evidence arrays are misaligned")
    expected_charge = -Q * density[np.newaxis, :] * (
        occupancy - equilibrium[np.newaxis, :]
    )
    charge_law_consistent = bool(
        np.allclose(charge, expected_charge, rtol=1.0e-12, atol=0.0)
    )
    dark_arrays_identical = all(
        np.array_equal(
            getattr(reference.dark_state, name),
            getattr(charged_dark, name),
        )
        for name in _DARK_IDENTITY_FIELDS
    )
    dark_charge = np.asarray(
        charged_dark.interface_incremental_sheet_charge_C_m2, dtype=float
    )
    dark_trace = np.asarray(
        charged_dark.interface_trace_potential_shift_V, dtype=float
    )
    sign_consistent = bool(np.all(charge[..., np.newaxis] * trace_shift >= 0.0))
    states = (reference.dark_state, charged_dark, *results)
    target_evidence = []
    for (label, voltage, illuminated), result in zip(_TARGETS, results):
        target_evidence.append(
            {
                "current_A_m2": result.current_A_m2,
                "illuminated": illuminated,
                "incremental_sheet_charge_C_m2": list(
                    result.interface_incremental_sheet_charge_C_m2
                ),
                "normalized_gauss_residual": list(
                    result.interface_normalized_gauss_residual
                ),
                "occupancy": list(result.interface_occupancy),
                "scaled_local_jacobian_condition": list(
                    result.interface_scaled_local_jacobian_condition
                ),
                "state_sha256": content_sha256(
                    {"phi_V": result.phi.tolist(), "state": result.y.tolist()}
                ),
                "target": label,
                "trace_potential_shift_V": [
                    list(values)
                    for values in result.interface_trace_potential_shift_V
                ],
                "voltage_V": voltage,
            }
        )
    return {
        "charge": charge,
        "condition": condition,
        "current": np.asarray([result.current_A_m2 for result in results]),
        "equilibrium": equilibrium,
        "grid": grid,
        "metadata": {
            "contact_thermodynamics": dataclasses.asdict(contact),
            "dark_reference": {
                "capture_velocities_m_s": [
                    list(values) for values in reference.capture_velocities_m_s
                ],
                "dark_state_sha256": reference.dark_state_sha256,
                "equilibrium_occupancy": list(reference.equilibrium_occupancy),
                "grid_sha256": reference.grid_sha256,
                "interface_defect_document_sha256": list(
                    reference.interface_defect_document_sha256
                ),
                "stack_sha256": reference.stack_sha256,
                "trap_density_m2": list(reference.trap_density_m2),
            },
            "microscopic_interface_defects": [
                {
                    "canonical_document": document.to_dict(),
                    "canonical_document_sha256": document.sha256,
                }
                for document in microscopic_contract.documents
            ],
            "derived": describe_stack(stack),
            "point": {
                "axis": point.axis,
                "label": point.label,
                "point_id": point.point_id,
                "updates": dict(point.updates),
            },
            "target_evidence": target_evidence,
        },
        "occupancy": occupancy,
        "quality": {
            "all_points_certified": all(result.certified for result in states),
            "barrier_shift_charge_sign_consistent": sign_consistent,
            "calibration_factors_unity": all(
                defect is not None
                and defect.calibration_factor == 1.0
                and defect.iface_state_calibration_factor == 1.0
                for defect in defects
            ),
            "charge_law_consistent": charge_law_consistent,
            "contact_thermodynamics_certified": contact.certified,
            "dark_charge_off_bit_identical": dark_arrays_identical,
            "dark_incremental_charge_zero_C_m2": float(
                np.max(np.abs(dark_charge))
            ),
            "dark_trace_shift_zero_V": float(np.max(np.abs(dark_trace))),
            "dark_reference_certified": reference.dark_state.certified,
            "dark_reference_hash_verified": all(
                isinstance(value, str) and len(value) == 64
                for value in (
                    reference.grid_sha256,
                    reference.stack_sha256,
                    reference.dark_state_sha256,
                    *reference.interface_defect_document_sha256,
                )
            ),
            "interface_evidence_aligned": True,
            "microscopic_defect_contract_verified": True,
            "max_charge_fraction_of_one_electron": float(
                np.max(np.abs(charge) / (Q * density[np.newaxis, :]))
            ),
            "max_continuity_bound_A_m2": max(
                max(
                    result.electron_continuity_bound_A_m2,
                    result.hole_continuity_bound_A_m2,
                )
                for result in states
            ),
            "max_current_spread_A_m2": max(
                result.face_current_spread_A_m2 for result in states
            ),
            "max_interface_local_residual": max(
                result.interface_local_residual for result in states
            ),
            "max_normalized_cell_residual": max(
                result.max_normalized_cell_residual for result in states
            ),
            "max_normalized_gauss_residual": float(np.max(np.abs(gauss))),
            "max_poisson_residual": max(result.poisson_residual for result in states),
            "max_scaled_local_jacobian_condition": float(np.max(condition)),
            "occupancy_bounded": bool(
                np.all((occupancy >= 0.0) & (occupancy <= 1.0))
                and np.all((equilibrium >= 0.0) & (equilibrium <= 1.0))
            ),
            "parameter_values_applied": _point_applied(point, stack, defects),
            "rebaseline_acknowledged": bool(
                stack.interface_charge_rebaseline_acknowledged
            ),
            "research_charge_closure_active": all(
                result.interface_charge_closure == "equilibrium_referenced"
                for result in (charged_dark, *results)
            ),
            "two_sided_topology_active": all(
                result.interface_topology == TWO_SIDED_TRACE for result in states
            ),
        },
        "trace_shift": trace_shift,
    }


def run_equilibrium_referenced_interface_charge_stress(
    lane: LaneDefinition,
    point: MatrixPoint,
    project_root: Path,
) -> CellMeasurement:
    """Evaluate the frozen Et/CBO/Nd/Nt stress set at one refinement cell."""
    options = lane.options
    points = _stress_points(options)
    baseline = _load_stack(lane, project_root)
    _require_research_stack(baseline)
    base_controls, controls = _solver_controls(options, point.tolerance_factor)
    protocol = _protocol(
        points,
        base_controls,
        refine_finite_difference_step=_refine_finite_difference_step(options),
    )
    records_list: list[dict[str, Any]] = []
    for stress_point in points:
        try:
            record = _solve_variant(
                apply_sweep_point(baseline, stress_point, sync_vbi=True),
                stress_point,
                point.grid,
                controls,
            )
        except Exception as exc:
            raise RuntimeError(
                f"interface-charge stress point {stress_point.point_id!r} failed"
            ) from exc
        records_list.append(record)
    records = tuple(records_list)
    first_grid = records[0]["grid"]
    grids_aligned = all(
        np.array_equal(first_grid, record["grid"]) for record in records[1:]
    )
    stack_hashes = [
        record["metadata"]["dark_reference"]["stack_sha256"]
        for record in records
    ]
    qualities = [record["quality"] for record in records]
    all_true_keys = (
        "all_points_certified",
        "barrier_shift_charge_sign_consistent",
        "calibration_factors_unity",
        "charge_law_consistent",
        "contact_thermodynamics_certified",
        "dark_charge_off_bit_identical",
        "dark_reference_certified",
        "dark_reference_hash_verified",
        "interface_evidence_aligned",
        "microscopic_defect_contract_verified",
        "occupancy_bounded",
        "parameter_values_applied",
        "rebaseline_acknowledged",
        "research_charge_closure_active",
        "two_sided_topology_active",
    )
    max_keys = (
        "dark_incremental_charge_zero_C_m2",
        "dark_trace_shift_zero_V",
        "max_charge_fraction_of_one_electron",
        "max_continuity_bound_A_m2",
        "max_current_spread_A_m2",
        "max_interface_local_residual",
        "max_normalized_cell_residual",
        "max_normalized_gauss_residual",
        "max_poisson_residual",
        "max_scaled_local_jacobian_condition",
    )
    quality = {
        key: float(all(bool(item[key]) for item in qualities))
        for key in all_true_keys
    }
    quality.update(
        {key: float(max(float(item[key]) for item in qualities)) for key in max_keys}
    )
    quality.update(
        {
            "grid_geometry_aligned": float(grids_aligned),
            "stress_point_count": float(len(records)),
            "variant_stack_identities_unique": float(
                len(stack_hashes) == len(set(stack_hashes))
            ),
        }
    )
    return CellMeasurement.from_mapping(
        {
            "observables": {
                "stress_current_density_A_m2": np.stack(
                    [record["current"] for record in records]
                ),
                "stress_equilibrium_occupancy": np.stack(
                    [record["equilibrium"] for record in records]
                ),
                "stress_interface_occupancy": np.stack(
                    [record["occupancy"] for record in records]
                ),
                "stress_sheet_charge_C_m2": np.stack(
                    [record["charge"] for record in records]
                ),
                "stress_trace_potential_shift_V": np.stack(
                    [record["trace_shift"] for record in records]
                ),
            },
            "quality": quality,
            "units": {
                "dark_incremental_charge_zero_C_m2": "C m-2",
                "dark_trace_shift_zero_V": "V",
                "max_continuity_bound_A_m2": "A m-2",
                "max_current_spread_A_m2": "A m-2",
                "stress_current_density_A_m2": "A m-2",
                "stress_sheet_charge_C_m2": "C m-2",
                "stress_trace_potential_shift_V": "V",
            },
            "metadata": {
                **{
                    "protocol": protocol,
                    "protocol_hash": content_sha256(protocol),
                    "protocol_schema": protocol["schema_version"],
                },
                "actual_grid_nodes": len(first_grid),
                "source_grid_intervals": point.grid,
                "stress_records": [record["metadata"] for record in records],
                "tolerance_controls": controls,
            },
        }
    )


__all__ = ["run_equilibrium_referenced_interface_charge_stress"]
