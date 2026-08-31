"""Registered D7-E2 multivalent bulk-defect QF/DC refinement lane.

The lane sweeps grid and solver tolerance over three SCAPS multivalent
families (double donor, double acceptor, amphoteric) plus a derived
multivalent p-n junction, and reports the charge-state observables that a
SCAPS charge-state export would be compared against. The external comparison
itself is NOT part of this lane: the suite manifest records the raw-export
contract with ``status='not_supplied'``, so this certificate is internal
numerical evidence only.

The lane additionally carries the D2 reduction as a quality metric: a v4
single-transition species solved through the whole QF/DC route must return
the certified v1 monovalent result. That is the one measurement that fails if
any consumer (contact neutrality, Poisson charge, fixed-QF tangent, or the
continuity source) stops using the shared master-equation closure.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from perovskite_sim.constants import Q
from perovskite_sim.experiments.quasi_fermi_steady_state import (
    QuasiFermiSteadyStateResult,
    solve_quasi_fermi_jv_sweep,
    solve_quasi_fermi_steady_state,
)
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.models.defects import (
    DONOR,
    EXPLICIT_DEFECT_SCHEMA_VERSION,
    EXPLICIT_QUASI_STEADY,
    INTEGRATED_TOTAL,
    NEUTRAL_WHEN_FILLED,
    SINGLE_LEVEL,
    BulkDefectDistribution,
    BulkDefectKinetics,
    BulkDefectSpecies,
)
from perovskite_sim.models.device import DeviceStack
from perovskite_sim.models.multivalent_defects import (
    MULTIVALENT_DEFECT_SCHEMA_VERSION,
    MultivalentBulkDefectSpecies,
    MultivalentDefectConfiguration,
    MultivalentEnergyLevels,
)
from perovskite_sim.physics.contacts import (
    require_contact_thermodynamic_certificate,
)
from perovskite_sim.solver.mol import (
    EXPLICIT_DEFECT_CHARGE_QF_DC,
    ExplicitDefectCapabilityError,
    MaterialArrays,
    build_material_arrays,
)

from .charged_defect_refinement import (
    _fixed_profile,
    _grid,
    _mass_action_error,
    _qf_span,
    _safe_project_file,
    _state_sha256,
)
from .dae_refinement import _finite_option, _integer_option, _protocol_metadata
from .numerical_certificate import LaneDefinition, MatrixPoint
from .refinement_runner import CellMeasurement


_SUITE_SCHEMA = "solarlab.scaps_multivalent_defect_reference_suite"
_SUITE_KEYS = {
    "derived_pn_device",
    "external_reference_contract",
    "scenarios",
    "schema",
    "schema_version",
}
_SCENARIO_KEYS = {
    "config_path",
    "config_sha256",
    "doping_polarity",
    "family",
    "id",
    "purpose",
}
_EXPECTED_SCENARIOS = {
    "M1": ("double_donor", "p_type"),
    "M2": ("double_acceptor", "n_type"),
    "M3": ("amphoteric", "intrinsic"),
}


@dataclass(frozen=True, slots=True)
class _Scenario:
    identifier: str
    family: str
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
        raise ValueError(f"{where} key mismatch; unknown={unknown}, missing={missing}")


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_suite(
    lane: LaneDefinition,
    project_root: Path,
) -> tuple[dict[str, Any], tuple[_Scenario, ...]]:
    suite_relative = lane.options.get("suite_manifest")
    suite_sha256 = lane.options.get("suite_manifest_sha256")
    if not isinstance(suite_relative, str) or not suite_relative:
        raise ValueError("multivalent lane must declare suite_manifest")
    if not isinstance(suite_sha256, str) or len(suite_sha256) != 64:
        raise ValueError("multivalent lane must declare suite_manifest_sha256")
    suite_path = _safe_project_file(project_root, suite_relative)
    actual = _file_sha256(suite_path)
    if actual != suite_sha256:
        raise ValueError(
            f"multivalent suite manifest hash drift: {actual} != {suite_sha256}"
        )
    try:
        raw = json.loads(suite_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("multivalent suite must be valid JSON") from exc
    if not isinstance(raw, dict):
        raise ValueError("multivalent suite must contain a JSON object")
    _require_exact_keys(raw, _SUITE_KEYS, where="multivalent suite")
    if raw["schema"] != _SUITE_SCHEMA:
        raise ValueError("multivalent suite schema mismatch")
    entries = raw["scenarios"]
    if not isinstance(entries, list) or len(entries) != len(_EXPECTED_SCENARIOS):
        raise ValueError("multivalent suite must declare exactly three scenarios")

    scenarios: list[_Scenario] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("multivalent suite scenario must be a mapping")
        _require_exact_keys(entry, _SCENARIO_KEYS, where="multivalent scenario")
        identifier = str(entry["id"])
        expected = _EXPECTED_SCENARIOS.get(identifier)
        if expected is None:
            raise ValueError(f"unexpected multivalent scenario {identifier!r}")
        family, polarity = expected
        if entry["family"] != family or entry["doping_polarity"] != polarity:
            raise ValueError(
                f"multivalent scenario {identifier} declares the wrong family/polarity"
            )
        config_path = _safe_project_file(project_root, str(entry["config_path"]))
        actual_config_sha = _file_sha256(config_path)
        if actual_config_sha != entry["config_sha256"]:
            raise ValueError(
                f"multivalent scenario {identifier} config hash drift: "
                f"{actual_config_sha} != {entry['config_sha256']}"
            )
        stack = load_device_from_yaml(str(config_path))
        params = stack.layers[0].params
        if params is None or (
            params.defect_schema_version != MULTIVALENT_DEFECT_SCHEMA_VERSION
        ):
            raise ValueError(
                f"multivalent scenario {identifier} config is not a canonical v4 layer"
            )
        document = params.defect_document
        if document is None:
            raise ValueError(f"multivalent scenario {identifier} has no v4 document")
        species = params.bulk_defects
        if len(species) != 1 or species[0].configuration.family != family:
            raise ValueError(
                f"multivalent scenario {identifier} must declare one {family} species"
            )
        scenarios.append(
            _Scenario(
                identifier=identifier,
                family=family,
                doping_polarity=polarity,
                config_path=str(entry["config_path"]),
                config_sha256=str(entry["config_sha256"]),
                purpose=str(entry["purpose"]),
                stack=stack,
                document_sha256=document.sha256,
            )
        )
    ordered = tuple(sorted(scenarios, key=lambda item: item.identifier))
    if tuple(item.identifier for item in ordered) != tuple(sorted(_EXPECTED_SCENARIOS)):
        raise ValueError("multivalent suite scenario identifiers are incomplete")
    return raw, ordered


def _voltage_grid(options: Mapping[str, Any]) -> tuple[float, ...]:
    values = options.get("voltage_grid_V")
    if not isinstance(values, (list, tuple)) or len(values) < 2:
        raise ValueError("multivalent lane must declare voltage_grid_V")
    grid = tuple(float(value) for value in values)
    if grid[0] != 0.0 or any(right <= left for left, right in zip(grid[:-1], grid[1:])):
        raise ValueError("voltage_grid_V must start at 0 V and strictly increase")
    return grid


def _illumination_steps(options: Mapping[str, Any]) -> tuple[float, ...]:
    values = options.get("illumination_steps")
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError("multivalent lane must declare illumination_steps")
    steps = tuple(float(value) for value in values)
    if steps[-1] != 1.0 or any(
        right <= left for left, right in zip(steps[:-1], steps[1:])
    ):
        raise ValueError("illumination_steps must strictly increase and end at 1.0")
    return steps


def _build_derived_pn(
    scenarios: Mapping[str, _Scenario],
    suite: Mapping[str, Any],
) -> DeviceStack:
    source = suite["derived_pn_device"]
    if not isinstance(source, Mapping):
        raise ValueError("derived_pn_device must be a mapping")
    if source.get("left_scenario") != "M1" or source.get("right_scenario") != "M2":
        raise ValueError("derived p/n device must use M1 left and M2 right")
    if source.get("interfaces") != "homojunction_zero_recombination":
        raise ValueError("derived p/n interface contract mismatch")
    photon_flux = float(source.get("illumination_photon_flux_m2_s"))
    if not math.isfinite(photon_flux) or photon_flux <= 0.0:
        raise ValueError("derived p/n photon flux must be finite and positive")
    left = replace(
        scenarios["M1"].stack.layers[0],
        name="m1_p_double_donor",
        role="absorber",
    )
    right = replace(
        scenarios["M2"].stack.layers[0],
        name="m2_n_double_acceptor",
        role="ETL",
    )
    return replace(
        scenarios["M1"].stack,
        layers=(left, right),
        Phi=photon_flux,
        interfaces=(),
    )


def _single_transition_pair(reference: _Scenario) -> tuple[DeviceStack, DeviceStack]:
    """Build the v4 vs v1 single-transition equivalence probe stacks.

    Both stacks describe the same physical single donor. The v4 side carries
    it as a two-state multivalent species with unity degeneracies; the v1 side
    carries the certified monovalent document. A discrepancy means one of the
    QF/DC consumers is no longer sourcing the shared closure.
    """

    params = reference.stack.layers[0].params
    if params is None:
        raise ValueError("equivalence probe requires layer parameters")
    kinetics = BulkDefectKinetics(
        sigma_n_m2=2.0e-19,
        sigma_p_m2=7.0e-20,
        thermal_velocity_n_m_s=1.0e5,
        thermal_velocity_p_m_s=8.0e4,
    )
    level_eV = 0.39
    density_m3 = 2.0e21
    multivalent = MultivalentBulkDefectSpecies(
        name="equivalence_single_donor",
        total_density_m3=density_m3,
        configuration=MultivalentDefectConfiguration(
            family="single_donor",
            charge_states_e=(1, 0),
            degeneracy_convention="unity",
            state_degeneracies=(1.0, 1.0),
            energy_levels=MultivalentEnergyLevels(
                first_transition_eV_above_vb=level_eV,
                correlation_energies_eV=(),
            ),
            transition_kinetics=(kinetics,),
        ),
    )
    monovalent = BulkDefectSpecies(
        name="equivalence_single_donor",
        distribution=BulkDefectDistribution(
            kind=SINGLE_LEVEL,
            normalization=INTEGRATED_TOTAL,
            total_density_m3=density_m3,
            center_eV_above_vb=level_eV,
        ),
        charge_transition=DONOR,
        neutral_reference=NEUTRAL_WHEN_FILLED,
        kinetics=kinetics,
        degeneracy=1.0,
    )
    multivalent_stack = replace(
        reference.stack,
        layers=(
            replace(
                reference.stack.layers[0],
                params=replace(
                    params,
                    defect_schema_version=MULTIVALENT_DEFECT_SCHEMA_VERSION,
                    defect_model=EXPLICIT_QUASI_STEADY,
                    bulk_defects=(multivalent,),
                ),
            ),
        ),
    )
    monovalent_stack = replace(
        multivalent_stack,
        layers=(
            replace(
                multivalent_stack.layers[0],
                params=replace(
                    params,
                    defect_schema_version=EXPLICIT_DEFECT_SCHEMA_VERSION,
                    defect_model=EXPLICIT_QUASI_STEADY,
                    bulk_defects=(monovalent,),
                ),
            ),
        ),
    )
    return multivalent_stack, monovalent_stack


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
            "charge_states": ["double_donor", "double_acceptor", "amphoteric"],
            "degeneracy_conventions": ["scaps_binomial", "unity"],
            "dopant_ionization": "fully_ionized",
            "occupancy": "local_stationary_master_equation",
            "occupancy_clipping": "none",
            "poisson_tangent": "analytic_fixed_qf",
            "recombination": "per_transition_shared_population",
            "state_normalization": "one_shared_density_per_physical_defect",
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
                "config_path": item.config_path,
                "config_sha256": item.config_sha256,
                "document_sha256": item.document_sha256,
                "family": item.family,
                "id": item.identifier,
            }
            for item in scenarios
        ],
        "schema_version": "multivalent-defect-qf-dc-refinement-protocol-v1",
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
            "primary_config_path": lane.config_path,
            "primary_config_sha256": lane.config_sha256,
            "suite_manifest_path": lane.options["suite_manifest"],
            "suite_manifest_sha256": lane.options["suite_manifest_sha256"],
        },
        "topology": {
            "contacts": "defect_aware_semiconductor_work_function",
            "derived_device": "M1_p_left_plus_M2_n_right_homojunction",
            "dynamic_occupancy": "excluded",
            "energy_distributions": "excluded",
            "interfaces": "zero_recombination_homojunction",
            "metastable_configurations": "excluded",
            "mobile_ions": "excluded",
            "multivalent_execution": "qf_dc_only",
            "spatial_grading": "excluded",
        },
    }


def run_multivalent_defect_qf_dc_refinement(
    lane: LaneDefinition,
    point: MatrixPoint,
    project_root: Path,
) -> CellMeasurement:
    """Execute one M1-M3 plus derived p/n multivalent refinement matrix cell."""

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
    continuity_tolerance = _finite_option(options, "continuity_tolerance_A_m2", 2.0e-4)
    spread_tolerance = _finite_option(options, "current_spread_tolerance_A_m2", 2.0e-4)
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
    default_rejections: list[bool] = []
    for identifier, scenario in scenarios.items():
        grid = _grid(scenario.stack, point.grid, grid_alpha)
        grids[identifier] = grid
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

    diagnostics = {
        identifier: result.multivalent_bulk_defect_diagnostics
        for identifier, result in dark.items()
    }
    if any(value is None for value in diagnostics.values()):
        raise RuntimeError("a multivalent scenario returned no defect diagnostics")
    checked = {
        identifier: value
        for identifier, value in diagnostics.items()
        if value is not None
    }

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
    pn_diagnostics = (
        pn_dark.multivalent_bulk_defect_diagnostics,
        *(result.multivalent_bulk_defect_diagnostics for result in pn_sweep.points),
    )
    if any(value is None for value in pn_diagnostics):
        raise RuntimeError("derived p/n states returned no multivalent diagnostics")
    checked_pn = tuple(value for value in pn_diagnostics if value is not None)

    # D2 reduction measured through the whole device lane, not just the local
    # closure: this is what breaks first if any consumer stops sharing it.
    equivalence_v4_stack, equivalence_v1_stack = _single_transition_pair(
        scenarios["M1"]
    )
    equivalence_grid = _grid(equivalence_v4_stack, point.grid, grid_alpha)
    equivalence_v4 = solve_quasi_fermi_steady_state(
        equivalence_grid,
        equivalence_v4_stack,
        V_app=0.0,
        illuminated=False,
        **solve_controls,
    )
    equivalence_v1 = solve_quasi_fermi_steady_state(
        equivalence_grid,
        equivalence_v1_stack,
        V_app=0.0,
        illuminated=False,
        **solve_controls,
    )
    equivalence_scale = max(
        float(np.max(np.abs(equivalence_v1.y))),
        np.finfo(float).tiny,
    )
    single_transition_state_error = float(
        np.max(np.abs(equivalence_v4.y - equivalence_v1.y)) / equivalence_scale
    )
    single_transition_potential_error = float(
        np.max(np.abs(equivalence_v4.phi - equivalence_v1.phi))
    )

    all_states = (
        *dark.values(),
        pn_dark,
        *pn_sweep.points,
        equivalence_v4,
        equivalence_v1,
    )
    all_material_dark = (
        *((dark[key], materials[key]) for key in sorted(dark)),
        (pn_dark, pn_material),
    )
    all_multivalent = (*checked.values(), *checked_pn)

    density = {
        identifier: scenario.stack.layers[0].params.bulk_defects[0].total_density_m3
        for identifier, scenario in scenarios.items()
    }
    normalized_charge = {
        identifier: (
            checked[identifier].total_charge_density_C_m3 / (Q * density[identifier])
        )
        for identifier in checked
    }
    pn_count = len(pn_grid)
    pn_n = np.asarray(pn_dark.y[:pn_count], dtype=float)
    pn_p = np.asarray(pn_dark.y[pn_count : 2 * pn_count], dtype=float)

    model_ids = {
        identifier: checked[identifier].model_identity_sha256 for identifier in checked
    }
    model_hashes_verified = bool(
        all(
            materials[identifier].multivalent_bulk_defects is not None
            and model_ids[identifier]
            == materials[identifier].multivalent_bulk_defects.identity_sha256
            for identifier in checked
        )
        and pn_material.multivalent_bulk_defects is not None
        and checked_pn[-1].model_identity_sha256
        == pn_material.multivalent_bulk_defects.identity_sha256
    )
    topology_verified = bool(
        all(
            material.carrier_statistics == "maxwell_boltzmann"
            and material.dopant_ionization_model == "fully_ionized"
            and material.band_gap_narrowing_model == "off"
            and material.N_iface_state == 0
            and material.monovalent_bulk_defects is None
            and material.neutral_bulk_defects is None
            and not material.has_dual_ions
            and not material.has_selective_contacts
            and not material.has_field_mobility
            and np.all(material.D_ion_node == 0.0)
            and np.all(material.P_ion0 == 0.0)
            for material in (*materials.values(), pn_material)
        )
    )
    # Probabilities are NaN off a species' own region by contract, so the
    # bound check is taken on owned nodes only.
    probability_bounded = True
    max_owned_sum_error = 0.0
    for evaluation in all_multivalent:
        for index, probabilities in enumerate(evaluation.state_probability):
            owned = evaluation.active_nodes[index]
            block = np.asarray(probabilities[:, owned], dtype=float)
            if block.size == 0:
                continue
            probability_bounded = bool(
                probability_bounded
                and np.all(np.isfinite(block))
                and np.all(block >= 0.0)
                and np.all(block <= 1.0)
            )
            max_owned_sum_error = max(
                max_owned_sum_error,
                float(np.max(np.abs(np.sum(block, axis=0) - 1.0))),
            )
    charge_signs_verified = bool(
        np.all(
            checked["M1"].total_charge_density_C_m3[
                checked["M1"].active_nodes.any(axis=0)
            ]
            > 0.0
        )
        and np.all(
            checked["M2"].total_charge_density_C_m3[
                checked["M2"].active_nodes.any(axis=0)
            ]
            < 0.0
        )
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
                "m1_normalized_charge_profile": _fixed_profile(
                    grids["M1"], normalized_charge["M1"], profile_points
                ),
                "m1_neutral_state_probability_profile": _fixed_profile(
                    grids["M1"],
                    checked["M1"].state_probability[0][-1],
                    profile_points,
                ),
                "m2_normalized_charge_profile": _fixed_profile(
                    grids["M2"], normalized_charge["M2"], profile_points
                ),
                "m2_neutral_state_probability_profile": _fixed_profile(
                    grids["M2"],
                    checked["M2"].state_probability[0][0],
                    profile_points,
                ),
                "m3_normalized_charge_profile": _fixed_profile(
                    grids["M3"], normalized_charge["M3"], profile_points
                ),
                "m3_neutral_state_probability_profile": _fixed_profile(
                    grids["M3"],
                    checked["M3"].state_probability[0][1],
                    profile_points,
                ),
                "m1_integrated_defect_charge_C_m2": np.trapezoid(
                    checked["M1"].total_charge_density_C_m3, grids["M1"]
                ),
                "m2_integrated_defect_charge_C_m2": np.trapezoid(
                    checked["M2"].total_charge_density_C_m3, grids["M2"]
                ),
                "m3_integrated_defect_charge_C_m2": np.trapezoid(
                    checked["M3"].total_charge_density_C_m3, grids["M3"]
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
            },
            "quality": {
                "all_states_certified": float(
                    all(result.certified for result in all_states)
                ),
                "charge_signs_verified": float(charge_signs_verified),
                "contact_thermodynamics_certified": float(
                    all(contact.certified for contact in contacts)
                ),
                "default_multivalent_path_rejected": float(all(default_rejections)),
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
                "max_master_equation_residual_s1": max(
                    evaluation.maximum_master_residual_s1
                    for evaluation in all_multivalent
                ),
                "max_normalized_cell_residual": max(
                    result.max_normalized_cell_residual for result in all_states
                ),
                "max_owned_probability_sum_error": max_owned_sum_error,
                "max_poisson_residual": max(
                    result.poisson_residual for result in all_states
                ),
                "min_transition_rate_s1": min(
                    evaluation.minimum_transition_rate_s1
                    for evaluation in all_multivalent
                ),
                "multivalent_model_hashes_verified": float(model_hashes_verified),
                "multivalent_topology_verified": float(topology_verified),
                "probability_bounded_without_clipping": float(probability_bounded),
                "scenario_count": float(len(scenarios)),
                "single_transition_d2_potential_error_V": (
                    single_transition_potential_error
                ),
                "single_transition_d2_state_relative_error": (
                    single_transition_state_error
                ),
                "terminal_densities_positive": float(terminal_positive),
                "voc_bracketed": float(pn_sweep.metrics.voc_bracketed),
                "voltage_points_completed": float(len(pn_sweep.points)),
            },
            "units": {
                "all_states_certified": "1",
                "charge_signs_verified": "1",
                "contact_thermodynamics_certified": "1",
                "default_multivalent_path_rejected": "1",
                "m1_integrated_defect_charge_C_m2": "C m-2",
                "m1_neutral_state_probability_profile": "1",
                "m1_normalized_charge_profile": "1",
                "m2_integrated_defect_charge_C_m2": "C m-2",
                "m2_neutral_state_probability_profile": "1",
                "m2_normalized_charge_profile": "1",
                "m3_integrated_defect_charge_C_m2": "C m-2",
                "m3_neutral_state_probability_profile": "1",
                "m3_normalized_charge_profile": "1",
                "max_continuity_bound_A_m2": "A m-2",
                "max_current_spread_A_m2": "A m-2",
                "max_dark_current_A_m2": "A m-2",
                "max_dark_qf_span_V": "V",
                "max_mass_action_relative_error": "1",
                "max_master_equation_residual_s1": "s-1",
                "max_normalized_cell_residual": "1",
                "max_owned_probability_sum_error": "1",
                "max_poisson_residual": "1",
                "min_transition_rate_s1": "s-1",
                "multivalent_model_hashes_verified": "1",
                "multivalent_topology_verified": "1",
                "pn_dark_electron_log10_profile": "log10(m-3)",
                "pn_dark_hole_log10_profile": "log10(m-3)",
                "pn_dark_potential_V": "V",
                "pn_ff": "1",
                "pn_jsc_A_m2": "A m-2",
                "pn_jv_current_A_m2": "A m-2",
                "pn_voc_V": "V",
                "probability_bounded_without_clipping": "1",
                "scenario_count": "1",
                "single_transition_d2_potential_error_V": "V",
                "single_transition_d2_state_relative_error": "1",
                "terminal_densities_positive": "1",
                "voc_bracketed": "1",
                "voltage_points_completed": "1",
            },
            "metadata": {
                **_protocol_metadata(protocol),
                "actual": {
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
                        "equivalence_probe": int(len(equivalence_grid)),
                    },
                    "multivalent_model_identity_sha256": {
                        identifier: model_ids[identifier] for identifier in model_ids
                    },
                    "scenario_document_sha256": {
                        item.identifier: item.document_sha256
                        for item in ordered_scenarios
                    },
                    "state_counts": {
                        identifier: list(checked[identifier].state_counts)
                        for identifier in checked
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
                        "equivalence_multivalent": _state_sha256(
                            equivalence_v4.y, equivalence_v4.phi
                        ),
                        "equivalence_monovalent": _state_sha256(
                            equivalence_v1.y, equivalence_v1.phi
                        ),
                    },
                },
            },
        }
    )


__all__ = ["run_multivalent_defect_qf_dc_refinement"]
