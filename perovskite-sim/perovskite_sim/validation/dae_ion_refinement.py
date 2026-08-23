"""Content-addressed refinement adapter for the single-positive-ion DAE."""

from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path
from typing import Any

import numpy as np

from perovskite_sim.discretization.grid import Layer, multilayer_grid
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.physics.generation import dual_cell_integral
from perovskite_sim.physics.poisson import solve_poisson_prefactored
from perovskite_sim.solver.dae_ion_integrator import (
    run_single_ion_backward_euler_reference,
)
from perovskite_sim.solver.dae_ion_jacobian import (
    build_single_ion_structured_state_jacobian,
)
from perovskite_sim.solver.dae_ions import (
    build_single_ion_consistent_initial_condition,
    build_single_positive_ion_dae,
)
from perovskite_sim.solver.mol import (
    StateVec,
    _charge_density,
    poisson_right_boundary,
    run_transient,
)
from perovskite_sim.solver.newton import solve_equilibrium
from perovskite_sim.solver.numerical_diagnostics import NumericalDiagnosticsPolicy

from .numerical_certificate import LaneDefinition, MatrixPoint, content_sha256
from .refinement_runner import CellMeasurement


def _finite_option(options: dict[str, Any], name: str, default: float) -> float:
    value = options.get(name, default)
    if isinstance(value, bool):
        raise ValueError(f"lane option {name!r} must be finite and positive")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"lane option {name!r} must be finite and positive")
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
        raise ValueError("single-ion DAE protocol requires a schema_version")
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
    applied_voltage_V: float,
    final_time_s: float,
    carrier_reference_time_s: float,
    ion_reference_time_s: float,
    base_time_steps: int,
    reference_grid_intervals: int,
    residual_tolerance: float,
    max_newton_iterations: int,
    max_line_search_backtracks: int,
    max_log_density_update: float,
    max_ion_logit_update: float,
    finite_difference_relative_step: float,
    mol_rtol: float,
    mol_atol_m3: float,
    mol_max_step_divisor: int,
) -> dict[str, Any]:
    return {
        "backward_euler": {
            "base_time_steps": base_time_steps,
            "finite_difference_relative_step": finite_difference_relative_step,
            "jacobian_modes": ["dense_central", "structured_analytic"],
            "max_ion_logit_update": max_ion_logit_update,
            "max_line_search_backtracks": max_line_search_backtracks,
            "max_log_density_update": max_log_density_update,
            "max_newton_iterations": max_newton_iterations,
            "residual_tolerance": residual_tolerance,
            "reference_grid_intervals": reference_grid_intervals,
            "step_count_formula": (
                "base_time_steps * (grid_intervals / "
                "reference_grid_intervals)^2 / tolerance_factor"
            ),
            "time_coordinate": "physical_carrier_and_positive_ion_density",
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
            "numerical_diagnostics": "research_strict_zero_floors",
            "rtol": mol_rtol,
            "state_coordinates": "density",
        },
        "operating_point": {
            "applied_voltage_V": applied_voltage_V,
            "final_time_s": final_time_s,
            "illuminated": False,
            "photon_flux_m2_s": 0.0,
        },
        "schema_version": "single-positive-ion-dae-refinement-protocol-v1",
        "source_slice": {
            "config_path": lane.config_path,
            "config_sha256": lane.config_sha256,
            "layer_index": source_layer_index,
            "layer_name": source_layer_name,
        },
        "topology": {
            "carrier_coordinates": ["log_n", "log_p"],
            "carrier_reference_time_s": carrier_reference_time_s,
            "contacts": "ohmic_dirichlet",
            "interface_states": "excluded",
            "interfaces": "excluded",
            "ion_boundary": "blocking_zero_flux",
            "ion_coordinate": "shifted_logit_positive_site_occupancy",
            "ion_reference_time_s": ion_reference_time_s,
            "mobile_ions": "single_positive",
            "poisson_potential": "algebraic",
            "steric_law": "device_configured_diffusion_only",
        },
    }


def _max_log_carrier_difference(left: np.ndarray, right: np.ndarray, count: int) -> float:
    left_carriers = np.asarray(left, dtype=float)[..., : 2 * count]
    right_carriers = np.asarray(right, dtype=float)[..., : 2 * count]
    if (
        left_carriers.shape != right_carriers.shape
        or not np.all(np.isfinite(left_carriers))
        or not np.all(np.isfinite(right_carriers))
        or np.any(left_carriers <= 0.0)
        or np.any(right_carriers <= 0.0)
    ):
        raise ValueError("carrier trajectories must be finite, positive, and matched")
    return float(np.max(np.abs(np.log(left_carriers / right_carriers))))


def _max_relative_ion_difference(left: np.ndarray, right: np.ndarray, count: int) -> float:
    left_ion = np.asarray(left, dtype=float)[..., 2 * count : 3 * count]
    right_ion = np.asarray(right, dtype=float)[..., 2 * count : 3 * count]
    if (
        left_ion.shape != right_ion.shape
        or not np.all(np.isfinite(left_ion))
        or not np.all(np.isfinite(right_ion))
        or np.any(left_ion <= 0.0)
        or np.any(right_ion <= 0.0)
    ):
        raise ValueError("ion trajectories must be finite, positive, and matched")
    return float(np.max(np.abs(left_ion - right_ion) / right_ion))


def run_single_ion_dae_transient(
    lane: LaneDefinition,
    point: MatrixPoint,
    project_root: Path,
) -> CellMeasurement:
    """Execute one single-positive-ion DAE grid-by-time-step cell."""
    options = lane.options
    if _string_option(options, "config_loader", "standard") != "standard":
        raise ValueError("the single-ion DAE lane requires config_loader='standard'")
    source_layer_index = _integer_option(options, "source_layer_index", 1, minimum=0)
    source_layer_name = _string_option(options, "source_layer_name", "MAPbI3")
    applied_voltage_V = _finite_option(options, "applied_voltage_V", 0.01)
    final_time_s = _finite_option(options, "final_time_s", 1.0e-2)
    carrier_reference_time_s = _finite_option(
        options, "carrier_reference_time_s", 1.0e-9
    )
    ion_reference_time_s = _finite_option(options, "ion_reference_time_s", 1.0)
    base_time_steps = _integer_option(options, "base_time_steps", 2)
    reference_grid_intervals = _integer_option(
        options, "reference_grid_intervals", 8
    )
    residual_tolerance = _finite_option(
        options, "newton_residual_tolerance", 1.0e-9
    )
    max_newton_iterations = _integer_option(options, "max_newton_iterations", 24)
    max_line_search_backtracks = _integer_option(
        options, "max_line_search_backtracks", 12, minimum=0
    )
    max_log_density_update = _finite_option(
        options, "max_log_density_update", 2.0
    )
    max_ion_logit_update = _finite_option(options, "max_ion_logit_update", 2.0)
    finite_difference_relative_step = _finite_option(
        options, "finite_difference_relative_step", 1.0e-6
    )
    mol_rtol = _finite_option(options, "mol_rtol", 1.0e-10)
    mol_atol_m3 = _finite_option(options, "mol_atol_m3", 1.0e2)
    mol_max_step_divisor = _integer_option(options, "mol_max_step_divisor", 400)
    time_steps = _time_step_count(
        base_time_steps,
        point.tolerance_factor,
        grid_intervals=point.grid,
        reference_grid_intervals=reference_grid_intervals,
    )

    source = load_device_from_yaml(project_root / lane.config_path)
    if source_layer_index >= len(source.layers):
        raise ValueError("source_layer_index is outside the configured stack")
    layer = source.layers[source_layer_index]
    if layer.name != source_layer_name:
        raise ValueError(
            "source layer identity mismatch: "
            f"expected {source_layer_name!r}, got {layer.name!r}"
        )
    if layer.params is None:
        raise ValueError("single-ion source layer requires electrical parameters")
    stack = replace(
        source,
        layers=(layer,),
        V_bi=0.0,
        built_in_potential_mode="legacy_manual",
        Phi=0.0,
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
    model = build_single_positive_ion_dae(
        grid,
        stack,
        solve_equilibrium(grid, stack),
        V_app_V=applied_voltage_V,
        carrier_reference_time_s=carrier_reference_time_s,
        ion_reference_time_s=ion_reference_time_s,
    )
    initial = build_single_ion_consistent_initial_condition(
        model,
        residual_tolerance=residual_tolerance,
    )
    if not initial.certified:
        raise RuntimeError("single-ion DAE initial condition is not certified")

    mol = run_transient(
        grid,
        initial.physical_state,
        (0.0, final_time_s),
        np.array([0.0, final_time_s]),
        stack,
        illuminated=False,
        V_app=applied_voltage_V,
        rtol=mol_rtol,
        atol=mol_atol_m3,
        max_step=final_time_s / mol_max_step_divisor,
        mat=model.material,
        numerical_diagnostics=NumericalDiagnosticsPolicy.research_strict(
            terminal_density_floor_m3=0.0,
            bulk_srh_denominator_floor_s_m3=0.0,
        ),
    )
    if not bool(mol.success) or np.asarray(mol.y).shape != (3 * grid.size, 2):
        raise RuntimeError(f"high-accuracy single-ion MoL reference failed: {mol.message}")
    mol_terminal = StateVec.unpack(mol.y[:, -1], grid.size)
    mol_rho = _charge_density(
        mol_terminal.p,
        mol_terminal.n,
        mol_terminal.P,
        model.material.P_ion0,
        model.material.N_A,
        model.material.N_D,
    )
    mol_potential = solve_poisson_prefactored(
        model.material.poisson_factor,
        mol_rho,
        phi_left=0.0,
        phi_right=poisson_right_boundary(model.material, applied_voltage_V),
    )

    time = np.linspace(0.0, final_time_s, time_steps + 1)
    integrator_options = {
        "initial": initial,
        "residual_tolerance": residual_tolerance,
        "max_newton_iterations": max_newton_iterations,
        "max_line_search_backtracks": max_line_search_backtracks,
        "max_log_density_update": max_log_density_update,
        "max_ion_logit_update": max_ion_logit_update,
        "finite_difference_relative_step": finite_difference_relative_step,
    }
    dense = run_single_ion_backward_euler_reference(
        model,
        time,
        jacobian_mode="dense_central",
        **integrator_options,
    )
    structured = run_single_ion_backward_euler_reference(
        model,
        time,
        jacobian_mode="structured_analytic",
        **integrator_options,
    )
    dense_terminal = StateVec.unpack(dense.physical_states[-1], grid.size)
    structured_terminal = StateVec.unpack(structured.physical_states[-1], grid.size)
    terminal_log_error = max(
        float(np.max(np.abs(np.log(structured_terminal.n / mol_terminal.n)))),
        float(np.max(np.abs(np.log(structured_terminal.p / mol_terminal.p)))),
    )
    terminal_ion_error = float(
        np.max(
            np.abs(structured_terminal.P - mol_terminal.P) / mol_terminal.P
        )
    )
    terminal_potential_error = float(
        np.max(np.abs(structured.potentials_V[-1] - mol_potential))
    )
    structured_dense_log_difference = _max_log_carrier_difference(
        structured.physical_states,
        dense.physical_states,
        grid.size,
    )
    structured_dense_ion_difference = _max_relative_ion_difference(
        structured.physical_states,
        dense.physical_states,
        grid.size,
    )
    structured_dense_potential_difference = float(
        np.max(np.abs(structured.potentials_V - dense.potentials_V))
    )

    initial_state = StateVec.unpack(initial.physical_state, grid.size)
    initial_inventory = dual_cell_integral(grid, initial_state.P)
    mol_inventory = dual_cell_integral(grid, mol_terminal.P)
    inventory_rate_scale = initial_inventory / final_time_s
    maximum_ion_balance_relative = max(
        dense.max_positive_ion_balance_defect_m2_s,
        structured.max_positive_ion_balance_defect_m2_s,
    ) / inventory_rate_scale
    maximum_ion_rhs_inventory_relative = max(
        dense.max_positive_ion_rhs_inventory_rate_m2_s,
        structured.max_positive_ion_rhs_inventory_rate_m2_s,
    ) / inventory_rate_scale
    maximum_inventory_drift = max(
        dense.max_relative_positive_ion_inventory_drift,
        structured.max_relative_positive_ion_inventory_drift,
        abs(mol_inventory - initial_inventory) / initial_inventory,
    )
    ion_motion = float(
        np.max(
            np.abs(structured_terminal.P - initial_state.P) / initial_state.P
        )
    )

    tangent = build_single_ion_structured_state_jacobian(
        model,
        initial.coordinate,
        initial.derivative,
    )
    structured_conditions = [
        report.max_scaled_jacobian_condition for report in structured.step_reports
    ]
    maximum_condition = max(structured_conditions, default=0.0)
    dense_work = dense.total_residual_evaluations
    if dense_work <= 0:
        raise RuntimeError("dense single-ion DAE reference reported no residual work")
    structured_work_fraction = structured.total_residual_evaluations / dense_work
    topology_verified = bool(
        len(stack.layers) == 1
        and not model.material.interface_nodes
        and model.material.N_iface_state == 0
        and not model.material.has_selective_contacts
        and not model.material.has_dual_ions
        and np.all(model.material.D_ion_node > 0.0)
        and np.all(model.material.P_ion0 > 0.0)
        and np.all(model.material.P_ion0 < model.material.P_lim_node)
    )
    site_occupancy_admissible = bool(
        np.all(dense.physical_states[:, 2 * grid.size : 3 * grid.size] > 0.0)
        and np.all(
            dense.physical_states[:, 2 * grid.size : 3 * grid.size]
            < model.material.P_lim_node
        )
        and np.all(structured.physical_states[:, 2 * grid.size : 3 * grid.size] > 0.0)
        and np.all(
            structured.physical_states[:, 2 * grid.size : 3 * grid.size]
            < model.material.P_lim_node
        )
        and np.all(mol_terminal.P > 0.0)
        and np.all(mol_terminal.P < model.material.P_lim_node)
    )
    terminal_densities_positive = bool(
        np.all(dense_terminal.n > 0.0)
        and np.all(dense_terminal.p > 0.0)
        and np.all(structured_terminal.n > 0.0)
        and np.all(structured_terminal.p > 0.0)
        and np.all(mol_terminal.n > 0.0)
        and np.all(mol_terminal.p > 0.0)
    )
    numerical_health = mol.numerical_diagnostics

    protocol = _execution_protocol(
        lane,
        source_layer_index=source_layer_index,
        source_layer_name=source_layer_name,
        applied_voltage_V=applied_voltage_V,
        final_time_s=final_time_s,
        carrier_reference_time_s=carrier_reference_time_s,
        ion_reference_time_s=ion_reference_time_s,
        base_time_steps=base_time_steps,
        reference_grid_intervals=reference_grid_intervals,
        residual_tolerance=residual_tolerance,
        max_newton_iterations=max_newton_iterations,
        max_line_search_backtracks=max_line_search_backtracks,
        max_log_density_update=max_log_density_update,
        max_ion_logit_update=max_ion_logit_update,
        finite_difference_relative_step=finite_difference_relative_step,
        mol_rtol=mol_rtol,
        mol_atol_m3=mol_atol_m3,
        mol_max_step_divisor=mol_max_step_divisor,
    )
    return CellMeasurement.from_mapping(
        {
            "observables": {
                "structured_dense_terminal_log_density_difference": max(
                    float(
                        np.max(
                            np.abs(
                                np.log(structured_terminal.n / dense_terminal.n)
                            )
                        )
                    ),
                    float(
                        np.max(
                            np.abs(
                                np.log(structured_terminal.p / dense_terminal.p)
                            )
                        )
                    ),
                ),
                "structured_dense_terminal_positive_ion_relative_difference": float(
                    np.max(
                        np.abs(structured_terminal.P - dense_terminal.P)
                        / dense_terminal.P
                    )
                ),
                "structured_dense_terminal_potential_difference_V": float(
                    np.max(
                        np.abs(
                            structured.potentials_V[-1] - dense.potentials_V[-1]
                        )
                    )
                ),
                "terminal_log_density_error": terminal_log_error,
                "terminal_positive_ion_relative_error": terminal_ion_error,
                "terminal_potential_error_V": terminal_potential_error,
            },
            "quality": {
                "bulk_srh_denominator_positive": float(
                    tangent.minimum_bulk_srh_denominator_s_m3 > 0.0
                ),
                "consistent_initial_condition_certified": float(initial.certified),
                "dense_reference_success": float(dense.success),
                "max_electron_balance_defect_A_m2": max(
                    dense.max_electron_balance_defect_A_m2,
                    structured.max_electron_balance_defect_A_m2,
                ),
                "max_hole_balance_defect_A_m2": max(
                    dense.max_hole_balance_defect_A_m2,
                    structured.max_hole_balance_defect_A_m2,
                ),
                "max_normalized_algebraic_residual": max(
                    dense.max_normalized_algebraic_residual,
                    structured.max_normalized_algebraic_residual,
                ),
                "max_normalized_carrier_residual": max(
                    dense.max_normalized_carrier_residual,
                    structured.max_normalized_carrier_residual,
                ),
                "max_normalized_positive_ion_residual": max(
                    dense.max_normalized_positive_ion_residual,
                    structured.max_normalized_positive_ion_residual,
                ),
                "max_positive_ion_balance_relative": maximum_ion_balance_relative,
                "max_positive_ion_inventory_relative_drift": maximum_inventory_drift,
                "max_positive_ion_rhs_inventory_relative": (
                    maximum_ion_rhs_inventory_relative
                ),
                "max_structured_dense_log_density_difference": (
                    structured_dense_log_difference
                ),
                "max_structured_dense_positive_ion_relative_difference": (
                    structured_dense_ion_difference
                ),
                "max_structured_dense_potential_difference_V": (
                    structured_dense_potential_difference
                ),
                "max_terminal_log_density_error": terminal_log_error,
                "max_terminal_positive_ion_relative_error": terminal_ion_error,
                "max_terminal_potential_error_V": terminal_potential_error,
                "minimum_positive_ion_relative_motion": ion_motion,
                "mol_numerical_health_passed": float(
                    numerical_health.would_pass_strict
                    and numerical_health.nonfinite_rhs_evaluations == 0
                    and numerical_health.nonfinite_trial_evaluations == 0
                ),
                "mol_reference_success": float(mol.success),
                "single_ion_topology_verified": float(topology_verified),
                "site_occupancy_admissible": float(site_occupancy_admissible),
                "structured_analytic_success": float(structured.success),
                "structured_csr_nonzeros_per_node": tangent.nonzero_count / grid.size,
                "structured_jacobian_condition_finite": float(
                    math.isfinite(maximum_condition)
                ),
                "structured_rhs_work_fraction": structured_work_fraction,
                "terminal_densities_positive": float(terminal_densities_positive),
            },
            "units": {
                "bulk_srh_denominator_positive": "1",
                "consistent_initial_condition_certified": "1",
                "dense_reference_success": "1",
                "max_electron_balance_defect_A_m2": "A m-2",
                "max_hole_balance_defect_A_m2": "A m-2",
                "max_normalized_algebraic_residual": "1",
                "max_normalized_carrier_residual": "1",
                "max_normalized_positive_ion_residual": "1",
                "max_positive_ion_balance_relative": "1",
                "max_positive_ion_inventory_relative_drift": "1",
                "max_positive_ion_rhs_inventory_relative": "1",
                "max_structured_dense_log_density_difference": "1",
                "max_structured_dense_positive_ion_relative_difference": "1",
                "max_structured_dense_potential_difference_V": "V",
                "max_terminal_log_density_error": "1",
                "max_terminal_positive_ion_relative_error": "1",
                "max_terminal_potential_error_V": "V",
                "minimum_positive_ion_relative_motion": "1",
                "mol_numerical_health_passed": "1",
                "mol_reference_success": "1",
                "single_ion_topology_verified": "1",
                "site_occupancy_admissible": "1",
                "structured_analytic_success": "1",
                "structured_csr_nonzeros_per_node": "1",
                "structured_dense_terminal_log_density_difference": "1",
                "structured_dense_terminal_positive_ion_relative_difference": "1",
                "structured_dense_terminal_potential_difference_V": "V",
                "structured_jacobian_condition_finite": "1",
                "structured_rhs_work_fraction": "1",
                "terminal_densities_positive": "1",
                "terminal_log_density_error": "1",
                "terminal_positive_ion_relative_error": "1",
                "terminal_potential_error_V": "V",
            },
            "metadata": {
                **_protocol_metadata(protocol),
                "actual": {
                    "dense_residual_evaluations": dense.total_residual_evaluations,
                    "grid_intervals": point.grid,
                    "grid_nodes": grid.size,
                    "initial_positive_ion_inventory_m2": initial_inventory,
                    "initial_state_sha256": initial.state_sha256,
                    "maximum_structured_jacobian_condition": maximum_condition,
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


__all__ = ["run_single_ion_dae_transient"]
