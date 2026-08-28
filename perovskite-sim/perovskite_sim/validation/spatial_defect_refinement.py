"""Three-axis device refinement for spatially graded explicit defects."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path

import numpy as np

from perovskite_sim.experiments.quasi_fermi_steady_state import (
    solve_quasi_fermi_jv_sweep,
    solve_quasi_fermi_steady_state,
)
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.models.defects import (
    ACCEPTOR,
    CONDUCTION_BAND_TAIL,
    DONOR,
    EXPLICIT_DEFECT_SPATIAL_SCHEMA_VERSION,
    GAUSSIAN,
    NEUTRAL,
    UNIFORM,
    VALENCE_BAND_TAIL,
    BulkDefectSpecies,
)
from perovskite_sim.models.device import DeviceStack, _edge_params
from perovskite_sim.physics.contacts import (
    build_semiconductor_contact_state,
    require_contact_thermodynamic_certificate,
)
from perovskite_sim.physics.defect_closure import (
    MonovalentBulkDefectEvaluation,
)
from perovskite_sim.physics.grading import has_grading_params
from perovskite_sim.solver.mol import (
    EXPLICIT_DEFECT_CHARGE_QF_DC,
    ExplicitDefectCapabilityError,
    MaterialArrays,
    build_material_arrays,
)

from .dae_refinement import _finite_option, _integer_option, _protocol_metadata
from .distributed_defect_refinement import (
    _EnergyRun,
    _diagnostics,
    _energy_changes,
    _energy_orders,
    _fixed_profile,
    _grid,
    _illumination_steps,
    _mass_action_error,
    _normalized_source_charge,
    _qf_span,
    _state_sha256,
    _voltage_grid,
)
from .numerical_certificate import LaneDefinition, MatrixPoint
from .refinement_runner import CellMeasurement


SPATIAL_DEFECT_DEVICE_REFINEMENT_VERSION = (
    "spatial-defect-qf-dc-three-axis-v1"
)
_EXPECTED_NAMES = (
    "p_vb_tail_donor",
    "p_uniform_neutral",
    "n_gaussian_acceptor",
    "n_cb_tail_neutral",
)
_EXPECTED_KINDS = (
    VALENCE_BAND_TAIL,
    UNIFORM,
    GAUSSIAN,
    CONDUCTION_BAND_TAIL,
)
_EXPECTED_TRANSITIONS = (DONOR, NEUTRAL, ACCEPTOR, NEUTRAL)
_EXPECTED_PROFILE_PRESENCE = (True, False, True, True)


@dataclass(frozen=True, slots=True)
class _SpatialContractEvidence:
    compiled_profiles_verified: bool
    contact_endpoints_verified: bool
    continuous_density_normalization_error: float
    interface_band_discontinuity_eV: float
    minimum_band_gap_span_eV: float
    minimum_density_multiplier: float
    minimum_support_margin_eV: float


def _safe_config_path(lane: LaneDefinition, project_root: Path) -> Path:
    relative = Path(lane.config_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("spatial-defect config path must be project-relative")
    root = project_root.resolve()
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("spatial-defect config escapes the project root") from exc
    if not path.is_file():
        raise ValueError("spatial-defect config does not exist")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != lane.config_sha256:
        raise ValueError(
            "spatial-defect config hash drift: "
            f"{digest} != {lane.config_sha256}"
        )
    return path


def _source_species(stack: DeviceStack) -> tuple[BulkDefectSpecies, ...]:
    if len(stack.layers) != 2 or any(
        layer.params is None for layer in stack.layers
    ):
        raise ValueError("spatial-defect certificate requires two electrical layers")
    if stack.built_in_potential_mode != "semiconductor_work_function":
        raise ValueError("spatial-defect certificate requires physical contacts")
    if not stack.band_grading or stack.interfaces:
        raise ValueError(
            "spatial-defect certificate requires grading and excludes interfaces"
        )
    if not all(has_grading_params(layer.params) for layer in stack.layers):
        raise ValueError("both certificate layers require explicit band endpoints")
    left = stack.layers[0].params
    right = stack.layers[1].params
    assert left is not None and right is not None
    if (
        left.Eg_back != right.Eg
        or left.chi_back != right.chi
    ):
        raise ValueError(
            "certificate band grading must be continuous at the p/n interface"
        )

    species: list[BulkDefectSpecies] = []
    for layer in stack.layers:
        params = layer.params
        assert params is not None
        if (
            params.defect_schema_version
            != EXPLICIT_DEFECT_SPATIAL_SCHEMA_VERSION
            or params.defect_model != "explicit_quasi_steady"
        ):
            raise ValueError("certificate layers require canonical v3 documents")
        species.extend(params.bulk_defects)
    resolved = tuple(species)
    if tuple(item.name for item in resolved) != _EXPECTED_NAMES:
        raise ValueError("certificate source names/order changed")
    if tuple(item.distribution.kind for item in resolved) != _EXPECTED_KINDS:
        raise ValueError("certificate distribution families/order changed")
    if tuple(item.charge_transition for item in resolved) != _EXPECTED_TRANSITIONS:
        raise ValueError("certificate charge transitions/order changed")
    if tuple(item.spatial_profile is not None for item in resolved) != (
        _EXPECTED_PROFILE_PRESENCE
    ):
        raise ValueError("certificate spatial-profile assignment changed")
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
            "charge_states": list(_EXPECTED_TRANSITIONS),
            "energy_distributions": list(_EXPECTED_KINDS),
            "energy_reference": "above_local_valence_band",
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
        "schema_version": SPATIAL_DEFECT_DEVICE_REFINEMENT_VERSION,
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
        "spatial_closure": {
            "band_grading": "continuous_local_Eg_chi",
            "density": "layer_average_Nbar_times_piecewise_linear_multiplier",
            "profile_presence": list(_EXPECTED_PROFILE_PRESENCE),
            "profile_sha256s": [
                None if item.spatial_profile is None else item.spatial_profile.sha256
                for item in species
            ],
            "source_coordinates": "normalized_layer_coordinate",
        },
        "topology": {
            "contacts": "endpoint_localized_defect_aware_work_function",
            "device": "continuous_band_graded_two_layer_pn_notch",
            "dynamic_occupancy": "excluded",
            "interfaces": "excluded",
            "mobile_ions": "excluded",
            "transport": "qf_dc_only",
        },
    }


def _profile_integral_error(species: tuple[BulkDefectSpecies, ...]) -> float:
    errors = []
    for source in species:
        profile = source.spatial_profile
        if profile is None:
            errors.append(0.0)
            continue
        positions = np.asarray(
            [item.position_fraction for item in profile.knots],
            dtype=float,
        )
        multipliers = np.asarray(
            [item.density_multiplier for item in profile.knots],
            dtype=float,
        )
        errors.append(abs(float(np.trapezoid(multipliers, positions)) - 1.0))
    return max(errors, default=0.0)


def _minimum_support_margin(
    material: MaterialArrays,
) -> float:
    model = material.monovalent_bulk_defects
    if model is None:
        return -math.inf
    margins: list[float] = []
    for region in model.regions:
        local_gap = np.asarray(region.local_band_gap_eV, dtype=float)
        minimum_gap = float(np.min(local_gap))
        for source in region.species:
            support = source.distribution.support_bounds_eV()
            if support is None:
                return -math.inf
            margins.extend((float(support[0]), minimum_gap - float(support[1])))
    return min(margins, default=-math.inf)


def _compiled_profiles_verified(
    grid: np.ndarray,
    stack: DeviceStack,
    run: _EnergyRun,
) -> bool:
    model = run.material.monovalent_bulk_defects
    if model is None or not model.has_spatial_profiles:
        return False
    expected_hashes = tuple(
        None if item.spatial_profile is None else item.spatial_profile.sha256
        for layer in stack.layers
        for item in layer.params.bulk_defects
    )
    if model.spatial_profile_sha256s != expected_hashes:
        return False
    if run.sweep.defect_spatial_profile_sha256s != expected_hashes:
        return False
    if run.sweep.defect_density_multiplier_bounds != (
        model.source_density_multiplier_bounds
    ):
        return False

    offset_m = 0.0
    for layer, region in zip(stack.layers, model.regions, strict=True):
        mask = region.active_nodes
        coordinates = (grid[mask] - offset_m) / float(layer.thickness)
        expected_multipliers = np.asarray(
            [
                [
                    1.0
                    if source.spatial_profile is None
                    else source.spatial_profile.density_multiplier_at(position)
                    for position in coordinates
                ]
                for source in layer.params.bulk_defects
            ],
            dtype=float,
        )
        if (
            not np.array_equal(region.normalized_layer_coordinates, coordinates)
            or not np.array_equal(
                region.local_band_gap_eV,
                run.material.Eg_phys[mask],
            )
            or not np.array_equal(
                region.local_effective_conduction_dos_m3,
                run.material.N_C_physical[mask],
            )
            or not np.array_equal(
                region.local_effective_valence_dos_m3,
                run.material.N_V_physical[mask],
            )
            or not np.array_equal(
                region.source_density_multipliers,
                expected_multipliers,
            )
        ):
            return False
        offset_m += float(layer.thickness)
    return all(
        diagnostics.spatial_profile_sha256s == expected_hashes
        and diagnostics.minimum_density_multipliers
        == tuple(value[0] for value in model.source_density_multiplier_bounds)
        and diagnostics.maximum_density_multipliers
        == tuple(value[1] for value in model.source_density_multiplier_bounds)
        for diagnostics in run.diagnostics
    )


def _contact_endpoints_verified(
    stack: DeviceStack,
    material: MaterialArrays,
    energy_order: int,
) -> bool:
    edge_specs = (
        (stack.layers[0], "front", material.n_L, material.p_L),
        (stack.layers[-1], "back", material.n_R, material.p_R),
    )
    for layer, side, expected_n, expected_p in edge_specs:
        endpoint = 0.0 if side == "front" else 1.0
        localized = _edge_params(layer, side, True)
        if any(item.spatial_profile is not None for item in localized.bulk_defects):
            return False
        for original, local in zip(
            layer.params.bulk_defects,
            localized.bulk_defects,
            strict=True,
        ):
            multiplier = (
                1.0
                if original.spatial_profile is None
                else original.spatial_profile.density_multiplier_at(endpoint)
            )
            if local.distribution.total_density_m3 != (
                original.distribution.total_density_m3 * multiplier
            ):
                return False
        contact = build_semiconductor_contact_state(
            localized,
            temperature_K=float(stack.T),
            use_temperature_scaling=True,
            defect_energy_quadrature_order=energy_order,
        )
        if (
            expected_n != contact.electron_density_m3
            or expected_p != contact.hole_density_m3
        ):
            return False
    return True


def _spatial_contract_evidence(
    grid: np.ndarray,
    stack: DeviceStack,
    run: _EnergyRun,
    species: tuple[BulkDefectSpecies, ...],
) -> _SpatialContractEvidence:
    model = run.material.monovalent_bulk_defects
    if model is None:
        raise RuntimeError("spatial QF/DC material omitted the defect model")
    band_spans = [
        float(np.ptp(np.asarray(region.local_band_gap_eV, dtype=float)))
        for region in model.regions
    ]
    left = stack.layers[0].params
    right = stack.layers[1].params
    interface_discontinuity = max(
        abs(float(left.Eg_back) - float(right.Eg)),
        abs(float(left.chi_back) - float(right.chi)),
    )
    return _SpatialContractEvidence(
        compiled_profiles_verified=_compiled_profiles_verified(grid, stack, run),
        contact_endpoints_verified=_contact_endpoints_verified(
            stack,
            run.material,
            run.order,
        ),
        continuous_density_normalization_error=_profile_integral_error(species),
        interface_band_discontinuity_eV=interface_discontinuity,
        minimum_band_gap_span_eV=min(band_spans),
        minimum_density_multiplier=min(
            value[0] for value in model.source_density_multiplier_bounds
        ),
        minimum_support_margin_eV=_minimum_support_margin(run.material),
    )


def _source_multiplier_profile(
    diagnostics: MonovalentBulkDefectEvaluation,
    material: MaterialArrays,
    points: int,
) -> np.ndarray:
    model = material.monovalent_bulk_defects
    multiplier = diagnostics.source_density_multiplier
    if model is None or multiplier is None:
        raise RuntimeError("spatial diagnostics omitted density multipliers")
    fixed = np.linspace(0.0, 1.0, points)
    rows: list[np.ndarray] = []
    offset = 0
    for region in model.regions:
        base_coordinates = np.asarray(
            region.normalized_layer_coordinates,
            dtype=float,
        )
        for local_index, source in enumerate(region.species):
            coordinates = base_coordinates
            values = np.asarray(
                multiplier[offset + local_index, region.active_nodes],
                dtype=float,
            )
            if coordinates[0] != 0.0:
                coordinates = np.r_[0.0, coordinates]
                front = (
                    1.0
                    if source.spatial_profile is None
                    else source.spatial_profile.density_multiplier_at(0.0)
                )
                values = np.r_[front, values]
            if coordinates[-1] != 1.0:
                back = (
                    1.0
                    if source.spatial_profile is None
                    else source.spatial_profile.density_multiplier_at(1.0)
                )
                coordinates = np.r_[coordinates, 1.0]
                values = np.r_[values, back]
            rows.append(np.interp(fixed, coordinates, values))
        offset += len(region.species)
    return np.concatenate(rows)


def run_spatial_defect_qf_dc_refinement(
    lane: LaneDefinition,
    point: MatrixPoint,
    project_root: Path,
) -> CellMeasurement:
    """Run one graded grid/tolerance cell with an inner energy ladder."""

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
    spatial_evidence: list[_SpatialContractEvidence] = []
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
        diagnostics = tuple(_diagnostics(state) for state in (dark, *sweep.points))
        run = _EnergyRun(
            order=order,
            material=material,
            dark=dark,
            sweep=sweep,
            diagnostics=diagnostics,
        )
        runs.append(run)
        spatial_evidence.append(
            _spatial_contract_evidence(grid, stack, run, species)
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
                and diagnostics.source_energy_orders
                == (run.order,) * len(species)
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
                "dark_band_gap_eV_profile": _fixed_profile(
                    grid,
                    terminal.material.Eg_phys,
                    profile_points,
                ),
                "dark_electron_affinity_eV_profile": _fixed_profile(
                    grid,
                    terminal.material.chi_phys,
                    profile_points,
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
                "dark_source_density_multiplier_profile": (
                    _source_multiplier_profile(
                        terminal_dark_diagnostics,
                        terminal.material,
                        profile_points,
                    )
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
                "contact_endpoints_verified": float(
                    all(item.contact_endpoints_verified for item in spatial_evidence)
                ),
                "contact_thermodynamics_certified": float(
                    all(contact.certified for contact in contacts)
                ),
                "default_spatial_path_rejected": float(default_path_rejected),
                "energy_orders_completed": len(resolved_runs),
                "graded_energy_metadata_verified": float(
                    energy_metadata_verified
                ),
                "graded_model_hashes_verified": float(model_hashes_verified),
                "graded_profiles_compiled_verified": float(
                    all(item.compiled_profiles_verified for item in spatial_evidence)
                ),
                "graded_topology_verified": float(topology_verified),
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
                "max_continuous_density_normalization_error": max(
                    item.continuous_density_normalization_error
                    for item in spatial_evidence
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
                "max_interface_band_discontinuity_eV": max(
                    item.interface_band_discontinuity_eV
                    for item in spatial_evidence
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
                "minimum_band_gap_span_eV": min(
                    item.minimum_band_gap_span_eV for item in spatial_evidence
                ),
                "minimum_density_multiplier": min(
                    item.minimum_density_multiplier for item in spatial_evidence
                ),
                "minimum_kinetic_denominator_s1": min(
                    diagnostics.minimum_kinetic_denominator_s1
                    for diagnostics in all_diagnostics
                ),
                "minimum_support_margin_eV": min(
                    item.minimum_support_margin_eV for item in spatial_evidence
                ),
                "occupancy_bounded_without_clipping": float(occupancy_bounded),
                "profiled_species_count": sum(_EXPECTED_PROFILE_PRESENCE),
                "source_species_count": len(species),
                "terminal_densities_positive": float(terminal_positive),
                "voltage_points_completed": len(terminal.sweep.points),
            },
            "units": {
                "all_states_certified": "1",
                "built_in_potential_V": "V",
                "contact_endpoints_verified": "1",
                "contact_reservoir_log10_m3": "log10(m-3)",
                "contact_thermodynamics_certified": "1",
                "dark_band_gap_eV_profile": "eV",
                "dark_electron_affinity_eV_profile": "eV",
                "dark_electron_log10_profile": "log10(m-3)",
                "dark_hole_log10_profile": "log10(m-3)",
                "dark_potential_V": "V",
                "dark_source_density_multiplier_profile": "1",
                "dark_source_normalized_charge_profile": "1",
                "dark_source_occupancy_profile": "1",
                "default_spatial_path_rejected": "1",
                "energy_orders_completed": "1",
                "graded_energy_metadata_verified": "1",
                "graded_model_hashes_verified": "1",
                "graded_profiles_compiled_verified": "1",
                "graded_topology_verified": "1",
                "integrated_source_charge_C_m2": "C m-2",
                "jv_current_A_m2": "A m-2",
                "max_contact_fermi_level_span_eV": "eV",
                "max_continuity_bound_A_m2": "A m-2",
                "max_continuous_density_normalization_error": "1",
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
                "max_interface_band_discontinuity_eV": "eV",
                "max_mass_action_relative_error": "1",
                "max_normalized_cell_residual": "1",
                "max_poisson_residual": "1",
                "minimum_band_gap_span_eV": "eV",
                "minimum_density_multiplier": "1",
                "minimum_kinetic_denominator_s1": "s-1",
                "minimum_support_margin_eV": "eV",
                "occupancy_bounded_without_clipping": "1",
                "profiled_species_count": "1",
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
                    "profile_sha256s": [
                        None
                        if item.spatial_profile is None
                        else item.spatial_profile.sha256
                        for item in species
                    ],
                    "source_density_multiplier_bounds": {
                        str(run.order): [
                            list(value)
                            for value in (
                                run.material.monovalent_bulk_defects
                                .source_density_multiplier_bounds
                            )
                        ]
                        for run in resolved_runs
                    },
                    "source_energy_orders": {
                        str(run.order): list(
                            run.diagnostics[-1].source_energy_orders
                        )
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
    "SPATIAL_DEFECT_DEVICE_REFINEMENT_VERSION",
    "run_spatial_defect_qf_dc_refinement",
]
