"""Content-addressed refinement adapter for the first research DAE slice."""

from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path
from typing import Any

import numpy as np

from perovskite_sim.constants import Q
from perovskite_sim.discretization.grid import Layer, multilayer_grid
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.physics.generation import dual_cell_integral
from perovskite_sim.physics.poisson import solve_poisson_prefactored
from perovskite_sim.solver.dae import (
    build_consistent_initial_condition,
    build_no_ion_no_interface_dae,
)
from perovskite_sim.solver.dae_integrator import run_backward_euler_reference
from perovskite_sim.solver.dae_jacobian import build_structured_state_jacobian
from perovskite_sim.solver.mol import StateVec, poisson_right_boundary, run_transient
from perovskite_sim.solver.newton import solve_equilibrium
from perovskite_sim.solver.numerical_diagnostics import NumericalDiagnosticsPolicy

from .numerical_certificate import LaneDefinition, MatrixPoint, content_sha256
from .refinement_runner import CellMeasurement


def _finite_option(
    options: dict[str, Any],
    name: str,
    default: float,
    *,
    positive: bool = True,
) -> float:
    value = options.get(name, default)
    if isinstance(value, bool):
        raise ValueError(f"lane option {name!r} must be finite")
    number = float(value)
    if not math.isfinite(number) or (positive and number <= 0.0):
        qualifier = "finite and positive" if positive else "finite"
        raise ValueError(f"lane option {name!r} must be {qualifier}")
    return number


def _integer_option(
    options: dict[str, Any],
    name: str,
    default: int,
    *,
    minimum: int = 1,
) -> int:
    value = options.get(name, default)
    if isinstance(value, bool):
        raise ValueError(f"lane option {name!r} must be an integer >= {minimum}")
    try:
        integer = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"lane option {name!r} must be an integer >= {minimum}"
        ) from exc
    if integer != value or integer < minimum:
        raise ValueError(f"lane option {name!r} must be an integer >= {minimum}")
    return integer


def _string_option(options: dict[str, Any], name: str, default: str) -> str:
    value = options.get(name, default)
    if not isinstance(value, str) or not value:
        raise ValueError(f"lane option {name!r} must be a non-empty string")
    return value


def _time_step_count(
    base_steps: int,
    tolerance_factor: float,
    *,
    grid_intervals: int,
    reference_grid_intervals: int,
) -> int:
    exact = (
        base_steps
        * (grid_intervals / reference_grid_intervals) ** 2
        / tolerance_factor
    )
    rounded = round(exact)
    if not math.isfinite(exact) or rounded < 1 or not math.isclose(
        exact,
        rounded,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError(
            "grid-scaled base_time_steps / tolerance_factor must be a positive integer"
        )
    return int(rounded)


def _protocol_metadata(protocol: dict[str, Any]) -> dict[str, Any]:
    schema = protocol.get("schema_version")
    if not isinstance(schema, str) or not schema:
        raise ValueError("DAE refinement protocol requires a schema_version")
    return {
        "protocol": protocol,
        "protocol_hash": content_sha256(protocol),
        "protocol_schema": schema,
    }


def _execution_protocol(
    lane: LaneDefinition,
    *,
    source_layer_index: int,
    source_layer_name: str,
    alpha_m1: float,
    photon_flux_m2_s: float,
    final_time_s: float,
    reference_time_s: float,
    base_time_steps: int,
    reference_grid_intervals: int,
    residual_tolerance: float,
    max_newton_iterations: int,
    max_line_search_backtracks: int,
    max_log_density_update: float,
    finite_difference_relative_step: float,
    mol_rtol: float,
    mol_atol_m3: float,
    mol_max_step_divisor: int,
) -> dict[str, Any]:
    return {
        "backward_euler": {
            "base_time_steps": base_time_steps,
            "jacobian_modes": ["dense_central", "structured_analytic"],
            "finite_difference_relative_step": finite_difference_relative_step,
            "max_line_search_backtracks": max_line_search_backtracks,
            "max_log_density_update": max_log_density_update,
            "max_newton_iterations": max_newton_iterations,
            "residual_tolerance": residual_tolerance,
            "reference_grid_intervals": reference_grid_intervals,
            "step_count_formula": (
                "base_time_steps * (grid_intervals / "
                "reference_grid_intervals)^2 / tolerance_factor"
            ),
            "time_coordinate": "physical_carrier_density",
        },
        "matrix": {
            "grid_parameter": lane.grid_parameter,
            "grid_values": list(lane.grid_values),
            "tolerance_factors": list(lane.tolerance_factors),
            "tolerance_parameter": lane.tolerance_parameter,
        },
        "mol_reference": {
            "atol_m3": mol_atol_m3,
            "max_step_divisor": mol_max_step_divisor,
            "method": "Radau",
            "max_nfev": None,
            "numerical_diagnostics": "research_strict_zero_floors",
            "rtol": mol_rtol,
            "state_coordinates": "density",
        },
        "operating_point": {
            "applied_voltage_V": 0.0,
            "final_time_s": final_time_s,
            "illuminated": True,
            "photon_flux_m2_s": photon_flux_m2_s,
        },
        "schema_version": "no-ion-dae-refinement-protocol-v1",
        "source_slice": {
            "absorption_coefficient_m1": alpha_m1,
            "config_path": lane.config_path,
            "config_sha256": lane.config_sha256,
            "layer_index": source_layer_index,
            "layer_name": source_layer_name,
        },
        "topology": {
            "carrier_coordinates": ["log_n", "log_p"],
            "contacts": "ohmic_dirichlet",
            "interface_states": "excluded",
            "interfaces": "excluded",
            "mobile_ions": "excluded",
            "poisson_potential": "algebraic",
            "reference_time_s": reference_time_s,
            "structural_ion_block": "exact_zero",
        },
    }


def _terminal_carriers(state: np.ndarray, node_count: int) -> tuple[np.ndarray, np.ndarray]:
    unpacked = StateVec.unpack(np.asarray(state, dtype=float), node_count)
    if (
        not np.all(np.isfinite(unpacked.n))
        or not np.all(np.isfinite(unpacked.p))
        or np.any(unpacked.n <= 0.0)
        or np.any(unpacked.p <= 0.0)
    ):
        raise ValueError("terminal carrier densities must be finite and positive")
    return unpacked.n, unpacked.p


def _max_log_carrier_difference(
    left: np.ndarray,
    right: np.ndarray,
    node_count: int,
) -> float:
    left_array = np.asarray(left, dtype=float)
    right_array = np.asarray(right, dtype=float)
    if left_array.shape != right_array.shape or left_array.shape[-1] < 2 * node_count:
        raise ValueError("carrier trajectories must be shape matched")
    left_carriers = left_array[..., : 2 * node_count]
    right_carriers = right_array[..., : 2 * node_count]
    if (
        not np.all(np.isfinite(left_carriers))
        or not np.all(np.isfinite(right_carriers))
        or np.any(left_carriers <= 0.0)
        or np.any(right_carriers <= 0.0)
    ):
        raise ValueError("carrier trajectories must be finite and positive")
    return float(np.max(np.abs(np.log(left_carriers / right_carriers))))


def run_no_ion_dae_transient(
    lane: LaneDefinition,
    point: MatrixPoint,
    project_root: Path,
) -> CellMeasurement:
    """Execute one no-ion DAE grid-by-time-step refinement cell."""
    options = lane.options
    loader = _string_option(options, "config_loader", "standard")
    if loader != "standard":
        raise ValueError("the no-ion DAE lane requires config_loader='standard'")

    source_layer_index = _integer_option(
        options,
        "source_layer_index",
        1,
        minimum=0,
    )
    source_layer_name = _string_option(options, "source_layer_name", "n_base")
    alpha_m1 = _finite_option(options, "alpha_m1", 2.0e4)
    photon_flux_m2_s = _finite_option(options, "photon_flux_m2_s", 1.0e17)
    final_time_s = _finite_option(options, "final_time_s", 1.0e-9)
    reference_time_s = _finite_option(options, "reference_time_s", 1.0e-9)
    base_time_steps = _integer_option(options, "base_time_steps", 8)
    reference_grid_intervals = _integer_option(
        options,
        "reference_grid_intervals",
        8,
    )
    residual_tolerance = _finite_option(
        options,
        "newton_residual_tolerance",
        1.0e-9,
    )
    max_newton_iterations = _integer_option(
        options,
        "max_newton_iterations",
        16,
    )
    max_line_search_backtracks = _integer_option(
        options,
        "max_line_search_backtracks",
        12,
        minimum=0,
    )
    max_log_density_update = _finite_option(
        options,
        "max_log_density_update",
        2.0,
    )
    finite_difference_relative_step = _finite_option(
        options,
        "finite_difference_relative_step",
        1.0e-6,
    )
    mol_rtol = _finite_option(options, "mol_rtol", 1.0e-10)
    mol_atol_m3 = _finite_option(options, "mol_atol_m3", 1.0e-3)
    mol_max_step_divisor = _integer_option(
        options,
        "mol_max_step_divisor",
        100,
    )
    time_steps = _time_step_count(
        base_time_steps,
        point.tolerance_factor,
        grid_intervals=point.grid,
        reference_grid_intervals=reference_grid_intervals,
    )

    source = load_device_from_yaml(project_root / lane.config_path)
    if source_layer_index >= len(source.layers):
        raise ValueError("source_layer_index is outside the configured stack")
    source_layer = source.layers[source_layer_index]
    if source_layer.name != source_layer_name:
        raise ValueError(
            "source layer identity mismatch: "
            f"expected {source_layer_name!r}, got {source_layer.name!r}"
        )
    if source_layer.params is None:
        raise ValueError("the DAE source slice requires electrical material parameters")
    layer = replace(
        source_layer,
        params=replace(source_layer.params, alpha=alpha_m1),
    )
    stack = replace(
        source,
        layers=(layer,),
        V_bi=0.0,
        built_in_potential_mode="legacy_manual",
        Phi=photon_flux_m2_s,
        interfaces=(),
        interface_defects=(),
        interface_charge_closure="off",
        interface_charge_rebaseline_acknowledged=False,
        S_n_left=None,
        S_p_left=None,
        S_n_right=None,
        S_p_right=None,
        grid_interval_weights=(),
        grid_alphas=(),
    )
    grid = multilayer_grid([Layer(layer.thickness, point.grid)], alpha=1.0)
    equilibrium = solve_equilibrium(grid, stack)
    model = build_no_ion_no_interface_dae(
        grid,
        stack,
        equilibrium,
        illuminated=True,
        reference_time_s=reference_time_s,
    )
    initial = build_consistent_initial_condition(
        model,
        residual_tolerance=residual_tolerance,
    )
    if not initial.certified:
        raise RuntimeError("DAE consistent initial condition is not certified")

    mol = run_transient(
        grid,
        initial.physical_state,
        (0.0, final_time_s),
        np.array([0.0, final_time_s]),
        stack,
        illuminated=True,
        V_app=0.0,
        rtol=mol_rtol,
        atol=mol_atol_m3,
        max_step=final_time_s / mol_max_step_divisor,
        mat=model.material,
        numerical_diagnostics=NumericalDiagnosticsPolicy.research_strict(
            terminal_density_floor_m3=0.0,
            bulk_srh_denominator_floor_s_m3=0.0,
        ),
    )
    if not bool(mol.success) or np.asarray(mol.y).shape != (
        3 * grid.size,
        2,
    ):
        raise RuntimeError(f"high-accuracy MoL reference failed: {mol.message}")
    mol_n, mol_p = _terminal_carriers(mol.y[:, -1], grid.size)
    mol_rho = Q * (mol_p - mol_n + model.material.N_D - model.material.N_A)
    mol_potential = solve_poisson_prefactored(
        model.material.poisson_factor,
        mol_rho,
        phi_left=0.0,
        phi_right=poisson_right_boundary(model.material, 0.0),
    )

    time = np.linspace(0.0, final_time_s, time_steps + 1)
    dense = run_backward_euler_reference(
        model,
        time,
        initial=initial,
        residual_tolerance=residual_tolerance,
        max_newton_iterations=max_newton_iterations,
        max_line_search_backtracks=max_line_search_backtracks,
        max_log_density_update=max_log_density_update,
        finite_difference_relative_step=finite_difference_relative_step,
        jacobian_mode="dense_central",
    )
    structured = run_backward_euler_reference(
        model,
        time,
        initial=initial,
        residual_tolerance=residual_tolerance,
        max_newton_iterations=max_newton_iterations,
        max_line_search_backtracks=max_line_search_backtracks,
        max_log_density_update=max_log_density_update,
        finite_difference_relative_step=finite_difference_relative_step,
        jacobian_mode="structured_analytic",
    )
    structured_n, structured_p = _terminal_carriers(
        structured.physical_states[-1],
        grid.size,
    )
    dense_n, dense_p = _terminal_carriers(dense.physical_states[-1], grid.size)

    terminal_log_error = max(
        float(np.max(np.abs(np.log(structured_n / mol_n)))),
        float(np.max(np.abs(np.log(structured_p / mol_p)))),
    )
    terminal_potential_error = float(
        np.max(np.abs(structured.potentials_V[-1] - mol_potential))
    )
    structured_dense_log_difference = _max_log_carrier_difference(
        structured.physical_states,
        dense.physical_states,
        grid.size,
    )
    structured_dense_potential_difference = float(
        np.max(np.abs(structured.potentials_V - dense.potentials_V))
    )
    photon_time_inventory = photon_flux_m2_s * final_time_s
    initial_state = StateVec.unpack(initial.physical_state, grid.size)
    inventory_response = np.array(
        [
            (
                dual_cell_integral(grid, structured_n)
                - dual_cell_integral(grid, initial_state.n)
            )
            / photon_time_inventory,
            (
                dual_cell_integral(grid, structured_p)
                - dual_cell_integral(grid, initial_state.p)
            )
            / photon_time_inventory,
        ],
        dtype=float,
    )

    initial_tangent = build_structured_state_jacobian(
        model,
        initial.coordinate,
        initial.derivative,
    )
    max_differential_residual = max(
        dense.max_normalized_differential_residual,
        structured.max_normalized_differential_residual,
    )
    max_algebraic_residual = max(
        dense.max_normalized_algebraic_residual,
        structured.max_normalized_algebraic_residual,
    )
    max_electron_balance = max(
        dense.max_electron_balance_defect_A_m2,
        structured.max_electron_balance_defect_A_m2,
    )
    max_hole_balance = max(
        dense.max_hole_balance_defect_A_m2,
        structured.max_hole_balance_defect_A_m2,
    )
    structured_conditions = [
        report.max_scaled_jacobian_condition for report in structured.step_reports
    ]
    maximum_structured_condition = max(structured_conditions, default=0.0)
    dense_work = dense.total_residual_evaluations
    if dense_work <= 0:
        raise RuntimeError("dense DAE reference reported no residual work")
    structured_work_fraction = structured.total_residual_evaluations / dense_work
    topology_verified = bool(
        len(stack.layers) == 1
        and not model.material.interface_nodes
        and model.material.N_iface_state == 0
        and not model.material.has_selective_contacts
        and not model.material.has_dual_ions
        and np.all(model.material.D_ion_node == 0.0)
        and np.all(model.material.P_ion0 == 0.0)
    )
    numerical_health = mol.numerical_diagnostics

    protocol = _execution_protocol(
        lane,
        source_layer_index=source_layer_index,
        source_layer_name=source_layer_name,
        alpha_m1=alpha_m1,
        photon_flux_m2_s=photon_flux_m2_s,
        final_time_s=final_time_s,
        reference_time_s=reference_time_s,
        base_time_steps=base_time_steps,
        reference_grid_intervals=reference_grid_intervals,
        residual_tolerance=residual_tolerance,
        max_newton_iterations=max_newton_iterations,
        max_line_search_backtracks=max_line_search_backtracks,
        max_log_density_update=max_log_density_update,
        finite_difference_relative_step=finite_difference_relative_step,
        mol_rtol=mol_rtol,
        mol_atol_m3=mol_atol_m3,
        mol_max_step_divisor=mol_max_step_divisor,
    )
    return CellMeasurement.from_mapping(
        {
            "observables": {
                "terminal_carrier_inventory_response": inventory_response,
                "terminal_log_density_error": terminal_log_error,
                "terminal_potential_error_V": terminal_potential_error,
                "structured_dense_terminal_log_density_difference": max(
                    float(np.max(np.abs(np.log(structured_n / dense_n)))),
                    float(np.max(np.abs(np.log(structured_p / dense_p)))),
                ),
                "structured_dense_terminal_potential_difference_V": float(
                    np.max(
                        np.abs(
                            structured.potentials_V[-1] - dense.potentials_V[-1]
                        )
                    )
                ),
            },
            "quality": {
                "bulk_srh_denominator_positive": float(
                    initial_tangent.minimum_bulk_srh_denominator_s_m3 > 0.0
                ),
                "consistent_initial_condition_certified": float(initial.certified),
                "dense_reference_success": float(dense.success),
                "first_slice_topology_verified": float(topology_verified),
                "max_electron_balance_defect_A_m2": max_electron_balance,
                "max_hole_balance_defect_A_m2": max_hole_balance,
                "max_normalized_algebraic_residual": max_algebraic_residual,
                "max_normalized_differential_residual": max_differential_residual,
                "max_structured_dense_log_density_difference": (
                    structured_dense_log_difference
                ),
                "max_structured_dense_potential_difference_V": (
                    structured_dense_potential_difference
                ),
                "max_terminal_log_density_error": terminal_log_error,
                "max_terminal_potential_error_V": terminal_potential_error,
                "mol_numerical_health_passed": float(
                    numerical_health.would_pass_strict
                    and numerical_health.nonfinite_rhs_evaluations == 0
                    and numerical_health.nonfinite_trial_evaluations == 0
                ),
                "mol_reference_success": float(mol.success),
                "structured_analytic_success": float(structured.success),
                "structured_csr_nonzeros_per_node": (
                    initial_tangent.nonzero_count / grid.size
                ),
                "structured_jacobian_condition_finite": float(
                    math.isfinite(maximum_structured_condition)
                ),
                "structured_rhs_work_fraction": structured_work_fraction,
                "terminal_densities_positive": float(
                    np.all(structured_n > 0.0)
                    and np.all(structured_p > 0.0)
                    and np.all(dense_n > 0.0)
                    and np.all(dense_p > 0.0)
                    and np.all(mol_n > 0.0)
                    and np.all(mol_p > 0.0)
                ),
            },
            "units": {
                "bulk_srh_denominator_positive": "1",
                "consistent_initial_condition_certified": "1",
                "dense_reference_success": "1",
                "first_slice_topology_verified": "1",
                "max_electron_balance_defect_A_m2": "A m-2",
                "max_hole_balance_defect_A_m2": "A m-2",
                "max_normalized_algebraic_residual": "1",
                "max_normalized_differential_residual": "1",
                "max_structured_dense_log_density_difference": "1",
                "max_structured_dense_potential_difference_V": "V",
                "max_terminal_log_density_error": "1",
                "max_terminal_potential_error_V": "V",
                "mol_numerical_health_passed": "1",
                "mol_reference_success": "1",
                "structured_analytic_success": "1",
                "structured_csr_nonzeros_per_node": "1",
                "structured_dense_terminal_log_density_difference": "1",
                "structured_dense_terminal_potential_difference_V": "V",
                "structured_jacobian_condition_finite": "1",
                "structured_rhs_work_fraction": "1",
                "terminal_carrier_inventory_response": "1",
                "terminal_densities_positive": "1",
                "terminal_log_density_error": "1",
                "terminal_potential_error_V": "V",
            },
            "metadata": {
                **_protocol_metadata(protocol),
                "actual": {
                    "dense_residual_evaluations": dense.total_residual_evaluations,
                    "grid_intervals": point.grid,
                    "grid_nodes": grid.size,
                    "initial_state_sha256": initial.state_sha256,
                    "maximum_structured_jacobian_condition": (
                        maximum_structured_condition
                    ),
                    "mol_nfev": int(mol.nfev),
                    "structured_residual_evaluations": (
                        structured.total_residual_evaluations
                    ),
                    "time_steps": time_steps,
                    "trajectory_sha256": {
                        "dense": dense.trajectory_sha256,
                        "structured": structured.trajectory_sha256,
                    },
                },
            },
        }
    )


__all__ = ["run_no_ion_dae_transient"]
