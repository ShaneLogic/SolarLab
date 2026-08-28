"""Three-axis device refinement for distributed explicit defects."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np

from perovskite_sim.constants import Q
from perovskite_sim.discretization.grid import Layer, multilayer_grid
from perovskite_sim.experiments.quasi_fermi_steady_state import (
    QuasiFermiJVSweepResult,
    QuasiFermiSteadyStateResult,
    solve_quasi_fermi_jv_sweep,
    solve_quasi_fermi_steady_state,
)
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.models.defects import (
    ACCEPTOR,
    CONDUCTION_BAND_TAIL,
    DONOR,
    EXPLICIT_DEFECT_DISTRIBUTION_SCHEMA_VERSION,
    GAUSSIAN,
    NEUTRAL,
    UNIFORM,
    VALENCE_BAND_TAIL,
    BulkDefectSpecies,
)
from perovskite_sim.models.device import DeviceStack
from perovskite_sim.physics.contacts import (
    require_contact_thermodynamic_certificate,
)
from perovskite_sim.physics.defect_closure import (
    MonovalentBulkDefectEvaluation,
)
from perovskite_sim.solver.mol import (
    EXPLICIT_DEFECT_CHARGE_QF_DC,
    ExplicitDefectCapabilityError,
    MaterialArrays,
    build_material_arrays,
)

from .dae_refinement import _finite_option, _integer_option, _protocol_metadata
from .numerical_certificate import LaneDefinition, MatrixPoint
from .refinement_runner import CellMeasurement


DISTRIBUTED_DEFECT_DEVICE_REFINEMENT_VERSION = (
    "distributed-defect-qf-dc-three-axis-v1"
)
_EXPECTED_KINDS = (
    VALENCE_BAND_TAIL,
    UNIFORM,
    GAUSSIAN,
    CONDUCTION_BAND_TAIL,
)
_EXPECTED_TRANSITIONS = (DONOR, NEUTRAL, ACCEPTOR, NEUTRAL)
_TANGENT_FIELDS = (
    "recombination_derivative_n_s1",
    "recombination_derivative_p_s1",
    "charge_derivative_fixed_qf_C_m3_V",
)


@dataclass(frozen=True, slots=True)
class _EnergyRun:
    order: int
    material: MaterialArrays
    dark: QuasiFermiSteadyStateResult
    sweep: QuasiFermiJVSweepResult
    diagnostics: tuple[MonovalentBulkDefectEvaluation, ...]


def _safe_config_path(lane: LaneDefinition, project_root: Path) -> Path:
    relative = Path(lane.config_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("distributed-defect config path must be project-relative")
    root = project_root.resolve()
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("distributed-defect config escapes the project root") from exc
    if not path.is_file():
        raise ValueError("distributed-defect config does not exist")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != lane.config_sha256:
        raise ValueError(
            "distributed-defect config hash drift: "
            f"{digest} != {lane.config_sha256}"
        )
    return path


def _energy_orders(options: dict[str, Any]) -> tuple[int, int, int]:
    raw = options.get("energy_quadrature_orders", [16, 32, 64])
    if not isinstance(raw, list) or len(raw) != 3:
        raise ValueError("energy_quadrature_orders must contain exactly three values")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in raw):
        raise ValueError("energy_quadrature_orders must contain integers")
    orders = tuple(int(value) for value in raw)
    if any(value < 2 or value > 512 for value in orders) or any(
        fine != 2 * coarse for coarse, fine in zip(orders, orders[1:])
    ):
        raise ValueError(
            "energy_quadrature_orders must be a three-level 2x ladder in [2, 512]"
        )
    return orders


def _voltage_grid(options: dict[str, Any]) -> tuple[float, ...]:
    raw = options.get("voltage_grid_V", [0.0, 0.01, 0.02])
    if not isinstance(raw, list) or len(raw) < 3:
        raise ValueError("voltage_grid_V must contain at least three values")
    values = tuple(float(value) for value in raw)
    if (
        not all(math.isfinite(value) for value in values)
        or values[0] != 0.0
        or any(right <= left for left, right in zip(values, values[1:]))
    ):
        raise ValueError("voltage_grid_V must be finite, start at zero, and increase")
    return values


def _illumination_steps(options: dict[str, Any]) -> tuple[float, ...]:
    raw = options.get("illumination_steps", [0.0, 1.0e-4, 1.0e-2, 1.0])
    if not isinstance(raw, list) or not raw:
        raise ValueError("illumination_steps must be a non-empty list")
    values = tuple(float(value) for value in raw)
    if (
        not all(math.isfinite(value) for value in values)
        or values[0] != 0.0
        or values[-1] != 1.0
        or any(right <= left for left, right in zip(values, values[1:]))
    ):
        raise ValueError("illumination_steps must increase exactly from zero to one")
    return values


def _source_species(stack: DeviceStack) -> tuple[BulkDefectSpecies, ...]:
    if len(stack.layers) != 2 or any(layer.params is None for layer in stack.layers):
        raise ValueError("distributed-defect certificate requires two electrical layers")
    if stack.built_in_potential_mode != "semiconductor_work_function":
        raise ValueError("distributed-defect certificate requires physical contacts")
    if stack.band_grading or stack.interfaces:
        raise ValueError("distributed-defect certificate excludes grading/interfaces")
    species: list[BulkDefectSpecies] = []
    for layer in stack.layers:
        params = layer.params
        assert params is not None
        if (
            params.defect_schema_version
            != EXPLICIT_DEFECT_DISTRIBUTION_SCHEMA_VERSION
            or params.defect_model != "explicit_quasi_steady"
        ):
            raise ValueError("certificate layers require canonical v2 defect documents")
        species.extend(params.bulk_defects)
    resolved = tuple(species)
    if tuple(item.distribution.kind for item in resolved) != _EXPECTED_KINDS:
        raise ValueError("certificate distribution families/order changed")
    if tuple(item.charge_transition for item in resolved) != _EXPECTED_TRANSITIONS:
        raise ValueError("certificate charge transitions/order changed")
    return resolved


def _execution_protocol(
    lane: LaneDefinition,
    *,
    species: tuple[BulkDefectSpecies, ...],
    energy_orders: tuple[int, int, int],
    grid_alpha: float,
    profile_points: int,
    voltage_grid_V: tuple[float, ...],
    illumination_steps: tuple[float, ...],
    base_newton_residual_tolerance: float,
    base_poisson_tolerance_V: float,
    base_finite_difference_step: float,
    continuity_tolerance_A_m2: float,
    current_spread_tolerance_A_m2: float,
) -> dict[str, object]:
    return {
        "constitutive_closure": {
            "carrier_statistics": "maxwell_boltzmann",
            "charge_states": [DONOR, NEUTRAL, ACCEPTOR, NEUTRAL],
            "energy_distributions": list(_EXPECTED_KINDS),
            "occupancy": "local_quasi_steady_energy_resolved",
            "occupancy_clipping": "none",
            "poisson_tangent": "analytic_fixed_qf",
            "recombination": "exact_energy_node_sum",
        },
        "matrix": {
            "energy_parameter": "defect_energy_quadrature_order",
            "energy_values": list(energy_orders),
            "grid_parameter": lane.grid_parameter,
            "grid_values": list(lane.grid_values),
            "tolerance_factors": list(lane.tolerance_factors),
            "tolerance_parameter": lane.tolerance_parameter,
        },
        "operating_points": {
            "dark": {"bias_V": 0.0},
            "illuminated_jv": {
                "illumination_steps": list(illumination_steps),
                "voltage_grid_V": list(voltage_grid_V),
            },
        },
        "profile_sampling": {
            "coordinate": "normalized_device_position",
            "points": profile_points,
            "rule": "linear_interpolation_of_certified_nodal_state",
        },
        "schema_version": DISTRIBUTED_DEFECT_DEVICE_REFINEMENT_VERSION,
        "solver": {
            "base_finite_difference_step": base_finite_difference_step,
            "base_newton_residual_tolerance": base_newton_residual_tolerance,
            "base_poisson_tolerance_V": base_poisson_tolerance_V,
            "continuity_tolerance_A_m2": continuity_tolerance_A_m2,
            "current_spread_tolerance_A_m2": current_spread_tolerance_A_m2,
            "effective_finite_difference_step": "base*sqrt(matrix_factor)",
            "effective_newton_tolerance": "base*matrix_factor",
            "effective_poisson_tolerance": "base*matrix_factor",
            "grid_alpha": grid_alpha,
        },
        "source": {
            "config_path": lane.config_path,
            "config_sha256": lane.config_sha256,
            "source_species": [item.to_dict() for item in species],
        },
        "topology": {
            "contacts": "defect_aware_semiconductor_work_function",
            "device": "uniform_two_layer_pn_homojunction",
            "dynamic_occupancy": "excluded",
            "interfaces": "excluded",
            "mobile_ions": "excluded",
            "spatial_grading": "excluded",
            "transport": "qf_dc_only",
        },
    }


def _grid(stack: DeviceStack, intervals: int, alpha: float) -> np.ndarray:
    return multilayer_grid(
        [Layer(layer.thickness, intervals) for layer in stack.layers],
        alpha=tuple(alpha for _ in stack.layers),
    )


def _fixed_profile(x: np.ndarray, values: np.ndarray, points: int) -> np.ndarray:
    coordinate = (x - float(x[0])) / float(x[-1] - x[0])
    return np.interp(np.linspace(0.0, 1.0, points), coordinate, values)


def _relative_change(left: object, right: object) -> float:
    left_array = np.asarray(left, dtype=float)
    right_array = np.asarray(right, dtype=float)
    scale = max(
        float(np.max(np.abs(left_array))),
        float(np.max(np.abs(right_array))),
        1.0,
    )
    return float(np.max(np.abs(right_array - left_array))) / scale


def _mass_action_error(result: QuasiFermiSteadyStateResult, mat: MaterialArrays) -> float:
    count = len(mat.ni_sq)
    n = np.asarray(result.y[:count], dtype=float)
    p = np.asarray(result.y[count : 2 * count], dtype=float)
    product = n * p
    scale = np.maximum.reduce((np.abs(product), np.abs(mat.ni_sq), np.ones(count)))
    return float(np.max(np.abs(product - mat.ni_sq) / scale))


def _qf_span(result: QuasiFermiSteadyStateResult) -> float:
    return max(
        float(np.ptp(result.electron_quasi_fermi_potential_V)),
        float(np.ptp(result.hole_quasi_fermi_potential_V)),
    )


def _state_sha256(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in arrays:
        contiguous = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _diagnostics(result: QuasiFermiSteadyStateResult) -> MonovalentBulkDefectEvaluation:
    value = result.bulk_defect_diagnostics
    if value is None:
        raise RuntimeError("distributed QF/DC state omitted defect diagnostics")
    return value


def _energy_changes(
    coarse: _EnergyRun,
    fine: _EnergyRun,
    species: tuple[BulkDefectSpecies, ...],
) -> dict[str, float]:
    occupancy = 0.0
    charge = 0.0
    recombination = 0.0
    tangent = 0.0
    potential = 0.0
    for left_state, right_state, left_diag, right_diag in zip(
        (coarse.dark, *coarse.sweep.points),
        (fine.dark, *fine.sweep.points),
        coarse.diagnostics,
        fine.diagnostics,
        strict=True,
    ):
        potential = max(
            potential,
            float(np.max(np.abs(right_state.phi - left_state.phi))),
        )
        occupancy = max(
            occupancy,
            float(np.max(np.abs(right_diag.occupancy - left_diag.occupancy))),
        )
        for index, source in enumerate(species):
            charge_scale = Q * source.distribution.total_density_m3
            charge = max(
                charge,
                float(
                    np.max(
                        np.abs(
                            right_diag.charge_density_C_m3[index]
                            - left_diag.charge_density_C_m3[index]
                        )
                    )
                )
                / charge_scale,
            )
        tangent = max(
            tangent,
            *(
                _relative_change(
                    getattr(left_diag, field),
                    getattr(right_diag, field),
                )
                for field in _TANGENT_FIELDS
            ),
        )
    for left_diag, right_diag in zip(
        coarse.diagnostics[1:],
        fine.diagnostics[1:],
        strict=True,
    ):
        recombination = max(
            recombination,
            _relative_change(
                left_diag.recombination_rate_m3_s,
                right_diag.recombination_rate_m3_s,
            ),
        )
    contact_left = np.asarray(
        [
            coarse.material.n_L,
            coarse.material.p_L,
            coarse.material.n_R,
            coarse.material.p_R,
        ]
    )
    contact_right = np.asarray(
        [
            fine.material.n_L,
            fine.material.p_L,
            fine.material.n_R,
            fine.material.p_R,
        ]
    )
    return {
        "charge": charge,
        "contact": float(
            np.max(np.abs(np.log10(contact_right) - np.log10(contact_left)))
        ),
        "current": _relative_change(
            coarse.sweep.currents_A_m2,
            fine.sweep.currents_A_m2,
        ),
        "occupancy": occupancy,
        "potential": potential,
        "recombination": recombination,
        "tangent": tangent,
    }


def _normalized_source_charge(
    diagnostics: MonovalentBulkDefectEvaluation,
    species: tuple[BulkDefectSpecies, ...],
) -> np.ndarray:
    return np.asarray(
        [
            diagnostics.charge_density_C_m3[index]
            / (Q * source.distribution.total_density_m3)
            for index, source in enumerate(species)
        ]
    )


def run_distributed_defect_qf_dc_refinement(
    lane: LaneDefinition,
    point: MatrixPoint,
    project_root: Path,
) -> CellMeasurement:
    """Run one outer grid/tolerance cell with an inner energy-order ladder."""

    config_path = _safe_config_path(lane, project_root)
    stack = load_device_from_yaml(config_path)
    species = _source_species(stack)
    options = lane.options
    orders = _energy_orders(options)
    grid_alpha = _finite_option(options, "grid_alpha", 2.0)
    profile_points = _integer_option(options, "profile_points", 17, minimum=3)
    voltage_grid = _voltage_grid(options)
    illumination_steps = _illumination_steps(options)
    base_newton_tolerance = _finite_option(
        options,
        "base_newton_residual_tolerance",
        1.0e-8,
    )
    base_poisson_tolerance = _finite_option(
        options,
        "base_poisson_tolerance_V",
        1.0e-10,
    )
    base_fd_step = _finite_option(
        options,
        "base_finite_difference_step",
        1.0e-5,
    )
    continuity_tolerance = _finite_option(
        options,
        "continuity_tolerance_A_m2",
        2.0e-4,
    )
    spread_tolerance = _finite_option(
        options,
        "current_spread_tolerance_A_m2",
        2.0e-4,
    )
    newton_tolerance = base_newton_tolerance * point.tolerance_factor
    poisson_tolerance = base_poisson_tolerance * point.tolerance_factor
    finite_difference_step = base_fd_step * math.sqrt(point.tolerance_factor)
    solve_controls = {
        "continuity_tolerance_A_m2": continuity_tolerance,
        "current_spread_tolerance_A_m2": spread_tolerance,
        "finite_difference_step": finite_difference_step,
        "newton_residual_tolerance": newton_tolerance,
        "poisson_tolerance_V": poisson_tolerance,
    }
    grid = _grid(stack, point.grid, grid_alpha)

    try:
        build_material_arrays(grid, stack)
    except ExplicitDefectCapabilityError:
        default_path_rejected = True
    else:
        default_path_rejected = False

    runs: list[_EnergyRun] = []
    contacts = []
    for order in orders:
        material = build_material_arrays(
            grid,
            stack,
            explicit_defect_charge_closure=EXPLICIT_DEFECT_CHARGE_QF_DC,
            defect_energy_quadrature_order=order,
        )
        contacts.append(require_contact_thermodynamic_certificate(stack, material))
        dark = solve_quasi_fermi_steady_state(
            grid,
            stack,
            V_app=0.0,
            illuminated=False,
            mat=material,
            defect_energy_quadrature_order=order,
            **solve_controls,
        )
        sweep = solve_quasi_fermi_jv_sweep(
            grid,
            stack,
            np.asarray(voltage_grid, dtype=float),
            mat=material,
            illumination_steps=illumination_steps,
            minimum_voltage_step_V=1.0e-4,
            defect_energy_quadrature_order=order,
            **solve_controls,
        )
        diagnostics = tuple(
            _diagnostics(state) for state in (dark, *sweep.points)
        )
        runs.append(
            _EnergyRun(
                order=order,
                material=material,
                dark=dark,
                sweep=sweep,
                diagnostics=diagnostics,
            )
        )
    resolved_runs = tuple(runs)
    comparisons = tuple(
        _energy_changes(coarse, fine, species)
        for coarse, fine in zip(resolved_runs, resolved_runs[1:])
    )
    terminal = resolved_runs[-1]
    terminal_dark_diagnostics = terminal.diagnostics[0]
    all_states = tuple(
        state
        for run in resolved_runs
        for state in (run.dark, *run.sweep.points)
    )
    all_diagnostics = tuple(
        diagnostics
        for run in resolved_runs
        for diagnostics in run.diagnostics
    )
    count = len(grid)
    terminal_n = np.asarray(terminal.dark.y[:count], dtype=float)
    terminal_p = np.asarray(terminal.dark.y[count : 2 * count], dtype=float)
    normalized_charge = _normalized_source_charge(
        terminal_dark_diagnostics,
        species,
    )
    source_occupancy_profile = np.concatenate(
        [
            _fixed_profile(grid, row, profile_points)
            for row in terminal_dark_diagnostics.occupancy
        ]
    )
    source_charge_profile = np.concatenate(
        [
            _fixed_profile(grid, row, profile_points)
            for row in normalized_charge
        ]
    )
    integrated_source_charge = np.asarray(
        [
            np.trapezoid(row, grid)
            for row in terminal_dark_diagnostics.charge_density_C_m3
        ]
    )
    model_hashes_verified = bool(
        len(
            {
                run.material.monovalent_bulk_defects.identity_sha256
                for run in resolved_runs
                if run.material.monovalent_bulk_defects is not None
            }
        )
        == len(orders)
        and all(
            run.material.monovalent_bulk_defects is not None
            and all(
                diagnostics.model_identity_sha256
                == run.material.monovalent_bulk_defects.identity_sha256
                for diagnostics in run.diagnostics
            )
            for run in resolved_runs
        )
    )
    energy_metadata_verified = bool(
        all(
            run.material.explicit_defect_energy_quadrature_order == run.order
            and all(
                diagnostics.distribution_kinds == _EXPECTED_KINDS
                and diagnostics.source_energy_orders == (run.order,) * len(species)
                and tuple(map(len, diagnostics.source_node_identifiers))
                == (run.order,) * len(species)
                for diagnostics in run.diagnostics
            )
            for run in resolved_runs
        )
    )
    topology_verified = bool(
        all(
            run.material.carrier_statistics == "maxwell_boltzmann"
            and run.material.dopant_ionization_model == "fully_ionized"
            and run.material.band_gap_narrowing_model == "off"
            and run.material.N_iface_state == 0
            and not run.material.has_dual_ions
            and not run.material.has_selective_contacts
            and not run.material.has_field_mobility
            and np.all(run.material.D_ion_node == 0.0)
            and np.all(run.material.P_ion0 == 0.0)
            for run in resolved_runs
        )
    )
    occupancy_bounded = bool(
        all(
            diagnostics.minimum_occupancy >= 0.0
            and diagnostics.maximum_occupancy <= 1.0
            for diagnostics in all_diagnostics
        )
    )
    terminal_positive = bool(
        all(
            np.all(state.y[: len(state.phi)] > 0.0)
            and np.all(state.y[len(state.phi) : 2 * len(state.phi)] > 0.0)
            for state in all_states
        )
    )
    protocol = _execution_protocol(
        lane,
        species=species,
        energy_orders=orders,
        grid_alpha=grid_alpha,
        profile_points=profile_points,
        voltage_grid_V=voltage_grid,
        illumination_steps=illumination_steps,
        base_newton_residual_tolerance=base_newton_tolerance,
        base_poisson_tolerance_V=base_poisson_tolerance,
        base_finite_difference_step=base_fd_step,
        continuity_tolerance_A_m2=continuity_tolerance,
        current_spread_tolerance_A_m2=spread_tolerance,
    )
    return CellMeasurement.from_mapping(
        {
            "observables": {
                "built_in_potential_V": terminal.material.V_bi_bc,
                "contact_reservoir_log10_m3": np.log10(
                    np.asarray(
                        [
                            terminal.material.n_L,
                            terminal.material.p_L,
                            terminal.material.n_R,
                            terminal.material.p_R,
                        ]
                    )
                ),
                "dark_electron_log10_profile": _fixed_profile(
                    grid,
                    np.log10(terminal_n),
                    profile_points,
                ),
                "dark_hole_log10_profile": _fixed_profile(
                    grid,
                    np.log10(terminal_p),
                    profile_points,
                ),
                "dark_potential_V": _fixed_profile(
                    grid,
                    terminal.dark.phi,
                    profile_points,
                ),
                "dark_source_normalized_charge_profile": source_charge_profile,
                "dark_source_occupancy_profile": source_occupancy_profile,
                "integrated_source_charge_C_m2": integrated_source_charge,
                "jv_current_A_m2": terminal.sweep.currents_A_m2,
                "short_circuit_current_A_m2": terminal.sweep.currents_A_m2[0],
            },
            "quality": {
                "all_states_certified": float(
                    all(state.certified for state in all_states)
                ),
                "contact_thermodynamics_certified": float(
                    all(contact.certified for contact in contacts)
                ),
                "default_distributed_path_rejected": float(default_path_rejected),
                "distributed_energy_metadata_verified": float(
                    energy_metadata_verified
                ),
                "distributed_model_hashes_verified": float(model_hashes_verified),
                "distributed_topology_verified": float(topology_verified),
                "energy_orders_completed": len(resolved_runs),
                "max_contact_fermi_level_span_eV": max(
                    (
                        math.inf
                        if state.contact_fermi_level_span_eV is None
                        else state.contact_fermi_level_span_eV
                    )
                    for state in all_states
                ),
                "max_continuity_bound_A_m2": max(
                    max(
                        state.electron_continuity_bound_A_m2,
                        state.hole_continuity_bound_A_m2,
                    )
                    for state in all_states
                ),
                "max_current_spread_A_m2": max(
                    state.face_current_spread_A_m2 for state in all_states
                ),
                "max_dark_current_A_m2": max(
                    abs(run.dark.current_A_m2) for run in resolved_runs
                ),
                "max_dark_qf_span_V": max(
                    _qf_span(run.dark) for run in resolved_runs
                ),
                "max_dark_recombination_rate_m3_s": max(
                    float(
                        np.max(
                            np.abs(
                                run.diagnostics[0].total_recombination_rate_m3_s
                            )
                        )
                    )
                    for run in resolved_runs
                ),
                "max_energy_contact_log10_change": max(
                    item["contact"] for item in comparisons
                ),
                "max_energy_current_relative_change": max(
                    item["current"] for item in comparisons
                ),
                "max_energy_normalized_charge_change": max(
                    item["charge"] for item in comparisons
                ),
                "max_energy_occupancy_absolute_change": max(
                    item["occupancy"] for item in comparisons
                ),
                "max_energy_potential_change_V": max(
                    item["potential"] for item in comparisons
                ),
                "max_energy_recombination_relative_change": max(
                    item["recombination"] for item in comparisons
                ),
                "max_energy_tangent_relative_change": max(
                    item["tangent"] for item in comparisons
                ),
                "max_mass_action_relative_error": max(
                    _mass_action_error(run.dark, run.material)
                    for run in resolved_runs
                ),
                "max_normalized_cell_residual": max(
                    state.max_normalized_cell_residual for state in all_states
                ),
                "max_poisson_residual": max(
                    state.poisson_residual for state in all_states
                ),
                "minimum_kinetic_denominator_s1": min(
                    diagnostics.minimum_kinetic_denominator_s1
                    for diagnostics in all_diagnostics
                ),
                "occupancy_bounded_without_clipping": float(occupancy_bounded),
                "source_species_count": len(species),
                "terminal_densities_positive": float(terminal_positive),
                "voltage_points_completed": len(terminal.sweep.points),
            },
            "units": {
                "all_states_certified": "1",
                "built_in_potential_V": "V",
                "contact_reservoir_log10_m3": "log10(m-3)",
                "contact_thermodynamics_certified": "1",
                "dark_electron_log10_profile": "log10(m-3)",
                "dark_hole_log10_profile": "log10(m-3)",
                "dark_potential_V": "V",
                "dark_source_normalized_charge_profile": "1",
                "dark_source_occupancy_profile": "1",
                "default_distributed_path_rejected": "1",
                "distributed_energy_metadata_verified": "1",
                "distributed_model_hashes_verified": "1",
                "distributed_topology_verified": "1",
                "energy_orders_completed": "1",
                "integrated_source_charge_C_m2": "C m-2",
                "jv_current_A_m2": "A m-2",
                "max_contact_fermi_level_span_eV": "eV",
                "max_continuity_bound_A_m2": "A m-2",
                "max_current_spread_A_m2": "A m-2",
                "max_dark_current_A_m2": "A m-2",
                "max_dark_qf_span_V": "V",
                "max_dark_recombination_rate_m3_s": "m-3 s-1",
                "max_energy_contact_log10_change": "log10(m-3)",
                "max_energy_current_relative_change": "1",
                "max_energy_normalized_charge_change": "1",
                "max_energy_occupancy_absolute_change": "1",
                "max_energy_potential_change_V": "V",
                "max_energy_recombination_relative_change": "1",
                "max_energy_tangent_relative_change": "1",
                "max_mass_action_relative_error": "1",
                "max_normalized_cell_residual": "1",
                "max_poisson_residual": "1",
                "minimum_kinetic_denominator_s1": "s-1",
                "occupancy_bounded_without_clipping": "1",
                "short_circuit_current_A_m2": "A m-2",
                "source_species_count": "1",
                "terminal_densities_positive": "1",
                "voltage_points_completed": "1",
            },
            "metadata": {
                **_protocol_metadata(protocol),
                "actual": {
                    "effective_finite_difference_step": finite_difference_step,
                    "effective_newton_residual_tolerance": newton_tolerance,
                    "effective_poisson_tolerance_V": poisson_tolerance,
                    "energy_orders": list(orders),
                    "grid_intervals_per_layer": point.grid,
                    "grid_nodes": len(grid),
                    "model_identity_sha256": {
                        str(run.order): (
                            run.material.monovalent_bulk_defects.identity_sha256
                            if run.material.monovalent_bulk_defects is not None
                            else None
                        )
                        for run in resolved_runs
                    },
                    "source_energy_orders": {
                        str(run.order): list(run.diagnostics[-1].source_energy_orders)
                        for run in resolved_runs
                    },
                    "state_sha256": {
                        str(run.order): {
                            "dark": _state_sha256(run.dark.y, run.dark.phi),
                            "jv": _state_sha256(
                                run.sweep.voltages_V,
                                run.sweep.currents_A_m2,
                            ),
                        }
                        for run in resolved_runs
                    },
                },
            },
        }
    )


__all__ = [
    "DISTRIBUTED_DEFECT_DEVICE_REFINEMENT_VERSION",
    "run_distributed_defect_qf_dc_refinement",
]
