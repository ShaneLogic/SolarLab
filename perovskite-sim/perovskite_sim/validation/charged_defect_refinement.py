"""Registered DEF-4 monovalent charged bulk-defect QF/DC refinement lane."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from perovskite_sim.constants import Q
from perovskite_sim.discretization.grid import Layer, multilayer_grid
from perovskite_sim.experiments.quasi_fermi_steady_state import (
    QuasiFermiSteadyStateResult,
    solve_quasi_fermi_jv_sweep,
    solve_quasi_fermi_steady_state,
)
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.models.defects import ACCEPTOR, DONOR, NEUTRAL
from perovskite_sim.models.device import DeviceStack
from perovskite_sim.physics.contacts import (
    require_contact_thermodynamic_certificate,
)
from perovskite_sim.physics.recombination import total_recombination
from perovskite_sim.solver.mol import (
    EXPLICIT_DEFECT_CHARGE_QF_DC,
    ExplicitDefectCapabilityError,
    MaterialArrays,
    build_material_arrays,
)

from .dae_refinement import _finite_option, _integer_option, _protocol_metadata
from .numerical_certificate import LaneDefinition, MatrixPoint
from .refinement_runner import CellMeasurement


_SUITE_SCHEMA = "solarlab.scaps_explicit_defect_reference_suite"
_SUITE_KEYS = {
    "derived_pn_device",
    "external_reference_contract",
    "scenarios",
    "schema",
    "schema_version",
}
_SCENARIO_KEYS = {
    "charge_transition",
    "config_path",
    "config_sha256",
    "doping_polarity",
    "id",
    "purpose",
}
_EXPECTED_SCENARIOS = {
    "S0": (NEUTRAL, "intrinsic"),
    "S1": (ACCEPTOR, "n_type"),
    "S2": (DONOR, "p_type"),
}


@dataclass(frozen=True, slots=True)
class _Scenario:
    identifier: str
    transition: str
    doping_polarity: str
    config_path: str
    config_sha256: str
    purpose: str
    stack: DeviceStack
    document_sha256: str


def _require_exact_keys(
    raw: Mapping[str, Any],
    expected: set[str],
    *,
    where: str,
) -> None:
    keys = set(raw)
    if keys != expected:
        unknown = sorted(keys - expected)
        missing = sorted(expected - keys)
        raise ValueError(
            f"{where} key mismatch; unknown={unknown}, missing={missing}"
        )


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe_project_file(project_root: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("suite config paths must be project-relative")
    resolved = (project_root / path).resolve()
    try:
        resolved.relative_to(project_root.resolve())
    except ValueError as exc:
        raise ValueError("suite config path escapes the project root") from exc
    if not resolved.is_file():
        raise ValueError(f"suite config does not exist: {relative}")
    return resolved


def _load_suite(
    lane: LaneDefinition,
    project_root: Path,
) -> tuple[dict[str, Any], tuple[_Scenario, ...]]:
    suite_path = _safe_project_file(project_root, lane.config_path)
    try:
        raw = json.loads(suite_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("charged-defect suite must be valid JSON") from exc
    if not isinstance(raw, dict):
        raise ValueError("charged-defect suite must contain a JSON object")
    _require_exact_keys(raw, _SUITE_KEYS, where="charged-defect suite")
    if raw["schema"] != _SUITE_SCHEMA or raw["schema_version"] != "1.0":
        raise ValueError("charged-defect suite schema/version mismatch")
    rows = raw["scenarios"]
    if not isinstance(rows, list) or len(rows) != len(_EXPECTED_SCENARIOS):
        raise ValueError("charged-defect suite must contain exactly S0, S1, S2")

    scenarios: list[_Scenario] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"scenario[{index}] must be a mapping")
        _require_exact_keys(row, _SCENARIO_KEYS, where=f"scenario[{index}]")
        identifier = str(row["id"])
        if identifier in seen or identifier not in _EXPECTED_SCENARIOS:
            raise ValueError("scenario identifiers must be unique S0, S1, S2")
        seen.add(identifier)
        expected_transition, expected_doping = _EXPECTED_SCENARIOS[identifier]
        transition = str(row["charge_transition"])
        doping = str(row["doping_polarity"])
        if transition != expected_transition or doping != expected_doping:
            raise ValueError(f"scenario {identifier} physical label mismatch")
        config_path = str(row["config_path"])
        expected_sha = str(row["config_sha256"])
        config_file = _safe_project_file(project_root, config_path)
        actual_sha = _file_sha256(config_file)
        if actual_sha != expected_sha:
            raise ValueError(
                f"scenario {identifier} config hash drift: "
                f"{actual_sha} != {expected_sha}"
            )
        stack = load_device_from_yaml(config_file)
        if len(stack.layers) != 1 or stack.layers[0].params is None:
            raise ValueError(f"scenario {identifier} must be one electrical layer")
        params = stack.layers[0].params
        document = params.defect_document
        if document is None or len(document.bulk_defects) != 1:
            raise ValueError(f"scenario {identifier} must declare one defect")
        species = document.bulk_defects[0]
        if species.charge_transition != transition:
            raise ValueError(f"scenario {identifier} defect transition mismatch")
        if identifier == "S0" and (params.N_A != 0.0 or params.N_D != 0.0):
            raise ValueError("S0 must be intrinsic")
        if identifier == "S1" and not (params.N_D > 0.0 and params.N_A == 0.0):
            raise ValueError("S1 must be n-type")
        if identifier == "S2" and not (params.N_A > 0.0 and params.N_D == 0.0):
            raise ValueError("S2 must be p-type")
        scenarios.append(
            _Scenario(
                identifier=identifier,
                transition=transition,
                doping_polarity=doping,
                config_path=config_path,
                config_sha256=expected_sha,
                purpose=str(row["purpose"]),
                stack=stack,
                document_sha256=document.sha256,
            )
        )
    if seen != set(_EXPECTED_SCENARIOS):
        raise ValueError("charged-defect suite is missing S0, S1, or S2")
    ordered = tuple(sorted(scenarios, key=lambda item: item.identifier))
    return raw, ordered


def _voltage_grid(options: dict[str, Any]) -> tuple[float, ...]:
    raw = options.get(
        "voltage_grid_V",
        [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
    )
    if not isinstance(raw, list) or len(raw) < 3:
        raise ValueError("voltage_grid_V must be a list of at least three values")
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
        raise ValueError("illumination_steps must increase exactly from 0 to 1")
    return values


def _execution_protocol(
    lane: LaneDefinition,
    *,
    suite: Mapping[str, Any],
    scenarios: tuple[_Scenario, ...],
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
            "charge_states": ["neutral", "acceptor", "donor"],
            "dopant_ionization": "fully_ionized",
            "occupancy": "local_quasi_steady_single_level",
            "occupancy_clipping": "none",
            "poisson_tangent": "analytic_fixed_qf",
            "recombination": "exact_per_species_srh",
        },
        "matrix": {
            "grid_parameter": lane.grid_parameter,
            "grid_values": list(lane.grid_values),
            "tolerance_factors": list(lane.tolerance_factors),
            "tolerance_parameter": lane.tolerance_parameter,
        },
        "operating_points": {
            "derived_pn": {
                "illumination_steps": list(illumination_steps),
                "photon_flux_m2_s": suite["derived_pn_device"][
                    "illumination_photon_flux_m2_s"
                ],
                "voltage_grid_V": list(voltage_grid_V),
            },
            "scenarios": "dark_zero_bias",
        },
        "profile_sampling": {
            "coordinate": "normalized_device_position",
            "points": profile_points,
            "rule": "linear_interpolation_of_certified_nodal_state",
        },
        "scenarios": [
            {
                "charge_transition": item.transition,
                "config_path": item.config_path,
                "config_sha256": item.config_sha256,
                "document_sha256": item.document_sha256,
                "id": item.identifier,
            }
            for item in scenarios
        ],
        "schema_version": "charged-defect-qf-dc-refinement-protocol-v1",
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
            "external_reference_contract": suite["external_reference_contract"],
            "suite_path": lane.config_path,
            "suite_sha256": lane.config_sha256,
        },
        "topology": {
            "charged_execution": "qf_dc_only",
            "contacts": "defect_aware_semiconductor_work_function",
            "derived_device": "S2_p_left_plus_S1_n_right_homojunction",
            "energy_distributions": "excluded",
            "interfaces": "zero_recombination_homojunction",
            "mobile_ions": "excluded",
            "spatial_grading": "excluded",
        },
    }


def _grid(stack: DeviceStack, intervals_per_layer: int, alpha: float) -> np.ndarray:
    return multilayer_grid(
        [Layer(layer.thickness, intervals_per_layer) for layer in stack.layers],
        alpha=tuple(alpha for _ in stack.layers),
    )


def _fixed_profile(
    x: np.ndarray,
    values: np.ndarray,
    point_count: int,
) -> np.ndarray:
    coordinate = (np.asarray(x, dtype=float) - float(x[0])) / float(x[-1] - x[0])
    return np.interp(np.linspace(0.0, 1.0, point_count), coordinate, values)


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


def _build_derived_pn(
    scenarios: Mapping[str, _Scenario],
    suite: Mapping[str, Any],
) -> DeviceStack:
    source = suite["derived_pn_device"]
    if not isinstance(source, Mapping):
        raise ValueError("derived_pn_device must be a mapping")
    if source.get("left_scenario") != "S2" or source.get("right_scenario") != "S1":
        raise ValueError("derived p/n device must use S2 left and S1 right")
    if source.get("interfaces") != "homojunction_zero_recombination":
        raise ValueError("derived p/n interface contract mismatch")
    photon_flux = float(source.get("illumination_photon_flux_m2_s"))
    if not math.isfinite(photon_flux) or photon_flux <= 0.0:
        raise ValueError("derived p/n photon flux must be finite and positive")
    left = replace(
        scenarios["S2"].stack.layers[0],
        name="s2_p_donor",
        role="absorber",
    )
    right = replace(
        scenarios["S1"].stack.layers[0],
        name="s1_n_acceptor",
        role="ETL",
    )
    return replace(
        scenarios["S2"].stack,
        layers=(left, right),
        Phi=photon_flux,
        interfaces=(),
    )


def run_charged_defect_qf_dc_refinement(
    lane: LaneDefinition,
    point: MatrixPoint,
    project_root: Path,
) -> CellMeasurement:
    """Execute one S0-S2 plus illuminated p/n refinement matrix cell."""

    suite, ordered_scenarios = _load_suite(lane, project_root)
    scenarios = {item.identifier: item for item in ordered_scenarios}
    options = lane.options
    grid_alpha = _finite_option(options, "grid_alpha", 2.0)
    profile_points = _integer_option(options, "profile_points", 17, minimum=3)
    voltage_grid = _voltage_grid(options)
    illumination_steps = _illumination_steps(options)
    base_newton_tolerance = _finite_option(
        options, "base_newton_residual_tolerance", 1.0e-8
    )
    base_poisson_tolerance = _finite_option(
        options, "base_poisson_tolerance_V", 1.0e-10
    )
    base_fd_step = _finite_option(options, "base_finite_difference_step", 1.0e-5)
    continuity_tolerance = _finite_option(
        options, "continuity_tolerance_A_m2", 2.0e-4
    )
    spread_tolerance = _finite_option(
        options, "current_spread_tolerance_A_m2", 2.0e-4
    )
    newton_tolerance = base_newton_tolerance * point.tolerance_factor
    poisson_tolerance = base_poisson_tolerance * point.tolerance_factor
    finite_difference_step = base_fd_step * math.sqrt(point.tolerance_factor)
    solve_controls = {
        "finite_difference_step": finite_difference_step,
        "newton_residual_tolerance": newton_tolerance,
        "poisson_tolerance_V": poisson_tolerance,
        "continuity_tolerance_A_m2": continuity_tolerance,
        "current_spread_tolerance_A_m2": spread_tolerance,
    }

    grids: dict[str, np.ndarray] = {}
    materials: dict[str, MaterialArrays] = {}
    dark: dict[str, QuasiFermiSteadyStateResult] = {}
    contacts = []
    default_rejections = []
    for identifier, scenario in scenarios.items():
        grid = _grid(scenario.stack, point.grid, grid_alpha)
        grids[identifier] = grid
        if scenario.transition == NEUTRAL:
            material = build_material_arrays(grid, scenario.stack)
        else:
            try:
                build_material_arrays(grid, scenario.stack)
            except ExplicitDefectCapabilityError:
                default_rejections.append(True)
            else:
                default_rejections.append(False)
            material = build_material_arrays(
                grid,
                scenario.stack,
                explicit_defect_charge_closure=EXPLICIT_DEFECT_CHARGE_QF_DC,
            )
        materials[identifier] = material
        contacts.append(
            require_contact_thermodynamic_certificate(scenario.stack, material)
        )
        dark[identifier] = solve_quasi_fermi_steady_state(
            grid,
            scenario.stack,
            V_app=0.0,
            illuminated=False,
            mat=material,
            **solve_controls,
        )

    s0_material = materials["S0"]
    if s0_material.neutral_bulk_defects is None:
        raise RuntimeError("S0 did not compile the exact neutral defect model")
    s0_species = s0_material.neutral_bulk_defects.species
    if len(s0_species) != 1 or not np.all(s0_species[0].active_nodes):
        raise RuntimeError("S0 must compile one device-wide neutral defect species")
    neutral_species = s0_species[0]
    intrinsic_density = math.sqrt(float(s0_material.ni_sq[0]))
    probe_n = np.geomspace(0.5, 10.0, len(grids["S0"])) * intrinsic_density
    probe_p = np.geomspace(8.0, 0.4, len(grids["S0"])) * intrinsic_density
    explicit_rate = total_recombination(
        probe_n,
        probe_p,
        s0_material.ni_sq,
        neutral_species.tau_n_s,
        neutral_species.tau_p_s,
        neutral_species.n1_m3,
        neutral_species.p1_m3,
        s0_material.B_rad,
        s0_material.C_n,
        s0_material.C_p,
        neutral_bulk_defects=s0_material.neutral_bulk_defects,
    )
    lifetime_rate = total_recombination(
        probe_n,
        probe_p,
        s0_material.ni_sq,
        neutral_species.tau_n_s,
        neutral_species.tau_p_s,
        neutral_species.n1_m3,
        neutral_species.p1_m3,
        s0_material.B_rad,
        s0_material.C_n,
        s0_material.C_p,
    )
    neutral_lifetime_bit_identical = np.array_equal(explicit_rate, lifetime_rate)

    charged_evaluations = {
        identifier: dark[identifier].bulk_defect_diagnostics
        for identifier in ("S1", "S2")
    }
    if any(value is None for value in charged_evaluations.values()):
        raise RuntimeError("S1/S2 did not return charged-defect diagnostics")
    acceptor = charged_evaluations["S1"]
    donor = charged_evaluations["S2"]
    assert acceptor is not None and donor is not None

    pn_stack = _build_derived_pn(scenarios, suite)
    pn_grid = _grid(pn_stack, point.grid, grid_alpha)
    pn_material = build_material_arrays(
        pn_grid,
        pn_stack,
        explicit_defect_charge_closure=EXPLICIT_DEFECT_CHARGE_QF_DC,
    )
    contacts.append(require_contact_thermodynamic_certificate(pn_stack, pn_material))
    pn_dark = solve_quasi_fermi_steady_state(
        pn_grid,
        pn_stack,
        V_app=0.0,
        illuminated=False,
        mat=pn_material,
        **solve_controls,
    )
    pn_sweep = solve_quasi_fermi_jv_sweep(
        pn_grid,
        pn_stack,
        np.asarray(voltage_grid, dtype=float),
        mat=pn_material,
        illumination_steps=illumination_steps,
        minimum_voltage_step_V=1.0e-4,
        **solve_controls,
    )
    pn_defect_diagnostics = (
        pn_dark.bulk_defect_diagnostics,
        *(result.bulk_defect_diagnostics for result in pn_sweep.points),
    )
    if any(value is None for value in pn_defect_diagnostics):
        raise RuntimeError("derived p/n states did not return defect diagnostics")
    checked_pn_diagnostics = tuple(
        value for value in pn_defect_diagnostics if value is not None
    )

    all_states = tuple(dark.values()) + (pn_dark,) + pn_sweep.points
    all_material_dark = (
        (dark["S0"], materials["S0"]),
        (dark["S1"], materials["S1"]),
        (dark["S2"], materials["S2"]),
        (pn_dark, pn_material),
    )
    active_acceptor = acceptor.active_nodes
    active_donor = donor.active_nodes
    acceptor_density = scenarios["S1"].stack.layers[0].params.bulk_defects[
        0
    ].distribution.total_density_m3
    donor_density = scenarios["S2"].stack.layers[0].params.bulk_defects[
        0
    ].distribution.total_density_m3
    normalized_acceptor_charge = (
        acceptor.total_charge_density_C_m3 / (Q * acceptor_density)
    )
    normalized_donor_charge = donor.total_charge_density_C_m3 / (Q * donor_density)
    pn_count = len(pn_grid)
    pn_n = np.asarray(pn_dark.y[:pn_count], dtype=float)
    pn_p = np.asarray(pn_dark.y[pn_count : 2 * pn_count], dtype=float)

    model_ids = (
        acceptor.model_identity_sha256,
        donor.model_identity_sha256,
        checked_pn_diagnostics[-1].model_identity_sha256,
    )
    model_hashes_verified = bool(
        materials["S1"].monovalent_bulk_defects is not None
        and materials["S2"].monovalent_bulk_defects is not None
        and pn_material.monovalent_bulk_defects is not None
        and model_ids[0] == materials["S1"].monovalent_bulk_defects.identity_sha256
        and model_ids[1] == materials["S2"].monovalent_bulk_defects.identity_sha256
        and model_ids[2] == pn_material.monovalent_bulk_defects.identity_sha256
    )
    topology_verified = bool(
        all(
            material.carrier_statistics == "maxwell_boltzmann"
            and material.dopant_ionization_model == "fully_ionized"
            and material.band_gap_narrowing_model == "off"
            and material.N_iface_state == 0
            and not material.has_dual_ions
            and not material.has_selective_contacts
            and not material.has_field_mobility
            and np.all(material.D_ion_node == 0.0)
            and np.all(material.P_ion0 == 0.0)
            for material in (*materials.values(), pn_material)
        )
    )
    occupancy_bounded = bool(
        min(acceptor.minimum_occupancy, donor.minimum_occupancy) >= 0.0
        and max(acceptor.maximum_occupancy, donor.maximum_occupancy) <= 1.0
        and all(
            diagnostic.minimum_occupancy >= 0.0
            and diagnostic.maximum_occupancy <= 1.0
            for diagnostic in checked_pn_diagnostics
        )
    )
    charge_signs_verified = bool(
        np.all(acceptor.charge_density_C_m3[active_acceptor] < 0.0)
        and np.all(donor.charge_density_C_m3[active_donor] > 0.0)
    )
    minimum_abs_charge = min(
        float(np.min(np.abs(acceptor.charge_density_C_m3[active_acceptor]))),
        float(np.min(np.abs(donor.charge_density_C_m3[active_donor]))),
    )
    minimum_denominator = min(
        acceptor.minimum_kinetic_denominator_s1,
        donor.minimum_kinetic_denominator_s1,
        *(
            diagnostic.minimum_kinetic_denominator_s1
            for diagnostic in checked_pn_diagnostics
        ),
    )
    terminal_positive = bool(
        all(
            np.all(result.y[: len(result.phi)] > 0.0)
            and np.all(result.y[len(result.phi) : 2 * len(result.phi)] > 0.0)
            for result in all_states
        )
    )
    protocol = _execution_protocol(
        lane,
        suite=suite,
        scenarios=ordered_scenarios,
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
                "acceptor_occupancy_profile": _fixed_profile(
                    grids["S1"], acceptor.occupancy[0], profile_points
                ),
                "acceptor_normalized_charge_profile": _fixed_profile(
                    grids["S1"], normalized_acceptor_charge, profile_points
                ),
                "donor_occupancy_profile": _fixed_profile(
                    grids["S2"], donor.occupancy[0], profile_points
                ),
                "donor_normalized_charge_profile": _fixed_profile(
                    grids["S2"], normalized_donor_charge, profile_points
                ),
                "pn_dark_electron_log10_profile": _fixed_profile(
                    pn_grid, np.log10(pn_n), profile_points
                ),
                "pn_dark_hole_log10_profile": _fixed_profile(
                    pn_grid, np.log10(pn_p), profile_points
                ),
                "pn_dark_potential_V": _fixed_profile(
                    pn_grid, pn_dark.phi, profile_points
                ),
                "pn_ff": pn_sweep.metrics.FF,
                "pn_jsc_A_m2": pn_sweep.metrics.J_sc,
                "pn_jv_current_A_m2": pn_sweep.currents_A_m2,
                "pn_voc_V": pn_sweep.metrics.V_oc,
                "s1_integrated_defect_charge_C_m2": np.trapezoid(
                    acceptor.total_charge_density_C_m3, grids["S1"]
                ),
                "s2_integrated_defect_charge_C_m2": np.trapezoid(
                    donor.total_charge_density_C_m3, grids["S2"]
                ),
            },
            "quality": {
                "all_states_certified": float(
                    all(result.certified for result in all_states)
                ),
                "charged_model_hashes_verified": float(model_hashes_verified),
                "charge_signs_verified": float(charge_signs_verified),
                "contact_thermodynamics_certified": float(
                    all(contact.certified for contact in contacts)
                ),
                "default_charged_path_rejected": float(all(default_rejections)),
                "max_continuity_bound_A_m2": max(
                    max(
                        result.electron_continuity_bound_A_m2,
                        result.hole_continuity_bound_A_m2,
                    )
                    for result in all_states
                ),
                "max_current_spread_A_m2": max(
                    result.face_current_spread_A_m2 for result in all_states
                ),
                "max_dark_current_A_m2": max(
                    abs(result.current_A_m2) for result, _material in all_material_dark
                ),
                "max_dark_qf_span_V": max(
                    _qf_span(result) for result, _material in all_material_dark
                ),
                "max_mass_action_relative_error": max(
                    _mass_action_error(result, material)
                    for result, material in all_material_dark
                ),
                "max_normalized_cell_residual": max(
                    result.max_normalized_cell_residual for result in all_states
                ),
                "max_poisson_residual": max(
                    result.poisson_residual for result in all_states
                ),
                "minimum_absolute_defect_charge_C_m3": minimum_abs_charge,
                "minimum_kinetic_denominator_s1": minimum_denominator,
                "minimum_short_circuit_current_A_m2": abs(
                    pn_sweep.metrics.J_sc
                ),
                "monovalent_topology_verified": float(topology_verified),
                "neutral_lifetime_bit_identical": float(
                    neutral_lifetime_bit_identical
                ),
                "occupancy_bounded_without_clipping": float(occupancy_bounded),
                "scenario_count": len(scenarios),
                "terminal_densities_positive": float(terminal_positive),
                "voc_bracketed": float(pn_sweep.metrics.voc_bracketed),
                "voltage_points_completed": len(pn_sweep.points),
            },
            "units": {
                "acceptor_normalized_charge_profile": "1",
                "acceptor_occupancy_profile": "1",
                "all_states_certified": "1",
                "charged_model_hashes_verified": "1",
                "charge_signs_verified": "1",
                "contact_thermodynamics_certified": "1",
                "default_charged_path_rejected": "1",
                "donor_normalized_charge_profile": "1",
                "donor_occupancy_profile": "1",
                "max_continuity_bound_A_m2": "A m-2",
                "max_current_spread_A_m2": "A m-2",
                "max_dark_current_A_m2": "A m-2",
                "max_dark_qf_span_V": "V",
                "max_mass_action_relative_error": "1",
                "max_normalized_cell_residual": "1",
                "max_poisson_residual": "1",
                "minimum_absolute_defect_charge_C_m3": "C m-3",
                "minimum_kinetic_denominator_s1": "s-1",
                "minimum_short_circuit_current_A_m2": "A m-2",
                "monovalent_topology_verified": "1",
                "neutral_lifetime_bit_identical": "1",
                "occupancy_bounded_without_clipping": "1",
                "pn_dark_electron_log10_profile": "log10(m-3)",
                "pn_dark_hole_log10_profile": "log10(m-3)",
                "pn_dark_potential_V": "V",
                "pn_ff": "1",
                "pn_jsc_A_m2": "A m-2",
                "pn_jv_current_A_m2": "A m-2",
                "pn_voc_V": "V",
                "s1_integrated_defect_charge_C_m2": "C m-2",
                "s2_integrated_defect_charge_C_m2": "C m-2",
                "scenario_count": "1",
                "terminal_densities_positive": "1",
                "voc_bracketed": "1",
                "voltage_points_completed": "1",
            },
            "metadata": {
                **_protocol_metadata(protocol),
                "actual": {
                    "charged_model_identity_sha256": list(model_ids),
                    "effective_finite_difference_step": finite_difference_step,
                    "effective_newton_residual_tolerance": newton_tolerance,
                    "effective_poisson_tolerance_V": poisson_tolerance,
                    "grid_intervals_per_layer": point.grid,
                    "grid_nodes": {
                        **{
                            identifier: int(len(grid))
                            for identifier, grid in grids.items()
                        },
                        "derived_pn": int(len(pn_grid)),
                    },
                    "scenario_document_sha256": {
                        item.identifier: item.document_sha256
                        for item in ordered_scenarios
                    },
                    "state_sha256": {
                        **{
                            identifier: _state_sha256(result.y, result.phi)
                            for identifier, result in dark.items()
                        },
                        "derived_pn_dark": _state_sha256(pn_dark.y, pn_dark.phi),
                        "derived_pn_jv": _state_sha256(
                            pn_sweep.voltages_V, pn_sweep.currents_A_m2
                        ),
                    },
                },
            },
        }
    )


__all__ = ["run_charged_defect_qf_dc_refinement"]
