"""Pure 2D experiment declarations composed with the actual device resolver.

No grid, material-array compiler, initializer or executor is imported here.
Input words, declared controls and unavailable execution binding are separate.
The legacy adapter supplies defaults and hashes from explicit source documents.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Any, NoReturn

from solarlab.config.resolve_device import resolve_device
from solarlab.device.defaults import DefaultCatalog
from solarlab.experiments.two_dimensional.inputs import (
    GrainSweepInput, JV2DInput, SpatialExperimentInput, invalid,
)
from solarlab.materials.resources import ResourceLibrary
from solarlab.materials.source import SourceDocument
from solarlab.units import UNIT_SCHEMA_VERSION

__all__ = ["PreparedSpatialExperiment", "prepare_spatial_experiment"]


def _bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _document(raw: bytes) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        if len({key for key, _ in items}) != len(items):
            raise ValueError("duplicate experiment JSON declaration field")
        return dict(items)
    return json.loads(raw, object_pairs_hook=pairs, parse_int=lambda word: -0.0 if word == "-0" else int(word))


def _fail(path: tuple[str | int, ...], message: str, value: Any) -> NoReturn:
    invalid("SpatialExperimentPreparation", path, message, value)


def _controls(experiment: JV2DInput | GrainSweepInput, defaults: DefaultCatalog) -> tuple[dict[str, Any], dict[str, str]]:
    values = defaults.experiment_defaults_for(experiment.kind)
    origins = {key: f"default_catalog.experiment_defaults.{experiment.kind}.{key}" for key in values}
    declared = experiment.normalized_data()
    for key in values:
        if key in declared:
            values[key] = declared[key]
            origins[key] = f"input.experiment.{key}"
    return values, origins


def _grains(grains: list[dict[str, Any]], length: float, electrical: list[dict[str, Any]],
            path: tuple[str | int, ...]) -> None:
    """Validate bands against a supplied domain, never allocate cell fractions."""
    seen: set[str] = set()
    ids = {layer["id"] for layer in electrical}
    for index, grain in enumerate(grains):
        here = (*path, index)
        if grain["id"] in seen:
            _fail((*here, "id"), "duplicate grain-boundary ID", grain["id"])
        seen.add(grain["id"])
        if not grain["layer_ids"]:
            _fail((*here, "layer_ids"), "grain bands require electrical layer references", grain["layer_ids"])
        for entry, layer_id in enumerate(grain["layer_ids"]):
            if layer_id not in ids or layer_id in grain["layer_ids"][:entry]:
                _fail((*here, "layer_ids", entry), "unknown or duplicate electrical layer reference", layer_id)
        lo, hi = grain["x_position"] - grain["width"] / 2, grain["x_position"] + grain["width"] / 2
        if not math.isfinite(hi) or lo < 0 or hi > length:
            _fail((*here, "width"), "strict declaration requires the grain band wholly inside the lateral domain; no clipping is performed", grain["width"])
        for previous in grains[:index]:
            if not set(previous["layer_ids"]) & set(grain["layer_ids"]):
                continue
            if min(hi, previous["x_position"] + previous["width"] / 2) > max(lo, previous["x_position"] - previous["width"] / 2):
                _fail(here, "overlapping grain bands in the same electrical layer are unsupported", grain)


def _geometry(controls: dict[str, Any], electrical: list[dict[str, Any]], device: dict[str, Any]) -> dict[str, Any]:
    return {
        "scope": "declared_mesh_controls_only",
        "lateral_interval_count": controls["Nx"],
        "expected_lateral_node_count": controls["Nx"] + 1,
        "vertical_intervals_per_electrical_layer": controls["Ny_per_layer"],
        "electrical_layer_ids": [layer["id"] for layer in electrical],
        "expected_vertical_node_count": len(electrical) * controls["Ny_per_layer"] + 1,
        "lateral_grid_policy": "existing_jv_2d_caller_uniform_x",
        "vertical_grid_policy": "existing_multilayer_grid_Ny_per_layer",
        "device_electrical_grid_applied": False,
        "retained_electrical_grid": device["electrical_grid"],
        "x_coordinates_m": None, "y_coordinates_m": None,
        "mesh_generated": False,
    }


def _protocol(experiment: JV2DInput, values: dict[str, Any], device: dict[str, Any],
              grains: list[dict[str, Any]], atol: dict[str, Any], carrier_bc: str) -> dict[str, Any]:
    supplied = experiment.jv_2d_protocol
    if values["protocol_mode"] == "research_strict" and (supplied is None or supplied.implicit_legacy_protocol):
        _fail(("experiment", "jv_2d_protocol"), "research_strict requires a supplied explicit execution-protocol declaration", None if supplied is None else supplied.editing_data())
    result: dict[str, Any] = {"status": "not_supplied" if supplied is None else "supplied_declaration_only",
                              "execution_binding": "pending_actual_mesh_schedule_state_and_capabilities",
                              "declaration": None, "declaration_sha256": None}
    if supplied is None:
        return result
    declared = supplied.normalized_data()
    known = {
        "temperature_K": device["settings"]["T"], "illuminated": values["illuminated"],
        "save_snapshots": values["save_snapshots"], "dwell_time_per_voltage_s": values["settle_t"],
        "lateral_bc": values["lateral_bc"], "carrier_boundary_condition": carrier_bc,
        "interface_srh": values["interface_srh"], "solver_rtol": values["rtol"],
        "max_nfev_per_solve": values["max_nfev_per_solve"], "max_bisect": values["max_bisect"],
        "ion_inventory_rtol": values["ion_inventory_rtol"], "solver_atol": atol,
        "state_topology": "single_positive_mobile_ion" if values["ion_dynamics"] == "single_mobile" else "frozen_ion_background",
    }
    if values["illuminated"]:
        known["initial_state_settle_s"] = values["initial_state_settle_s"]
    for key, expected in known.items():
        if declared[key] != expected:
            _fail(("experiment", "jv_2d_protocol", key), "supplied protocol disagrees with effective declared request control", declared[key])
    if len(declared["x_coordinates_m"]) != values["Nx"] + 1:
        _fail(("experiment", "jv_2d_protocol", "x_coordinates_m"), "Nx declares intervals; supplied x coordinates require Nx+1 nodes", declared["x_coordinates_m"])
    if not grains and declared["grain_boundaries"]:
        _fail(("experiment", "jv_2d_protocol", "grain_boundaries"), "protocol declares bands absent from the effective microstructure", declared["grain_boundaries"])
    result.update(declaration=declared, declaration_sha256=hashlib.sha256(_bytes(declared)).hexdigest(),
                  checked_declared_fields=sorted(known),
                  unchecked_bindings=["actual_x_y_coordinates", "actual_voltage_schedule", "grain_role_to_layer_compilation",
                                      "initial_state", "material_arrays", "physical_capabilities"])
    return result


def _jv(experiment: JV2DInput, defaults: DefaultCatalog, device: dict[str, Any],
        electrical: list[dict[str, Any]]) -> dict[str, Any]:
    values, origins = _controls(experiment, defaults)
    supplied = experiment.editing_data()
    override = experiment.microstructure
    # Preserve the actual backend truthiness contract. {} is declared, but
    # inherits; {grain_boundaries: []} is truthy and clears the inventory.
    active_override = override is not None and bool(override.editing_data())
    grains = (override.normalized_data().get("grain_boundaries", []) if active_override and override is not None else device["grain_boundaries"])
    grain_path = ("experiment", "microstructure", "grain_boundaries") if active_override else ("device", "grain_boundaries")
    _grains(grains, values["lateral_length"], electrical, grain_path)
    boundary = experiment.lateral_bc if experiment.lateral_bc is not None else ("neumann" if grains else "periodic")
    values["lateral_bc"] = boundary
    origins["lateral_bc"] = "input.experiment.lateral_bc" if experiment.lateral_bc is not None else "legacy_microstructure_boundary_rule"
    extended = values["ion_dynamics"] != "frozen" or values["interface_srh"] != "off"
    if (grains or extended) and boundary != "neumann":
        _fail(("experiment", "lateral_bc"), "grain, mobile-ion and interface-SRH 2D declarations require Neumann-x; periodic support is unavailable", boundary)
    carrier_bc = "selective_robin" if any(contact[key] is not None for contact in device["contacts"] for key in ("S_n", "S_p")) else "ohmic"
    if extended and carrier_bc != "ohmic":
        _fail(("device", "contacts"), "mobile-ion/interface-SRH 2D declarations require ohmic carrier contacts", device["contacts"])
    if extended and values["protocol_mode"] != "research_strict":
        _fail(("experiment", "protocol_mode"), "mobile-ion/interface-SRH 2D declarations require research_strict and an explicit protocol", values["protocol_mode"])
    componentwise = experiment.componentwise_atol
    use_componentwise = componentwise is not None or (extended and "atol" not in supplied)
    names = ("carrier_fraction", "ion_fraction", "interface_fraction", "minimum_atol", "refinement_factor")
    if use_componentwise:
        fields = componentwise.normalized_data() if componentwise is not None else defaults.experiment_defaults_for("jv_2d_componentwise_atol")
        atol = {"mode": "componentwise", "scalar_atol": None, **fields}
        origins["solver_atol"] = "input.experiment.componentwise_atol" if componentwise is not None else "default_catalog.experiment_defaults.jv_2d_componentwise_atol"
    else:
        atol = {"mode": "scalar", "scalar_atol": values["atol"], **dict.fromkeys(names)}
        origins["solver_atol"] = origins["atol"]
    values.pop("atol")
    origins.pop("atol")
    values["solver_atol"] = atol
    protocol = _protocol(experiment, values, device, grains, atol, carrier_bc)
    return {"kind": experiment.kind, "controls": values, "field_origins": origins,
            "geometry": {**_geometry(values, electrical, device), "lateral_length_m": values["lateral_length"], "lateral_bc": boundary},
            "microstructure": {"declared_presence": "omitted" if "microstructure" not in supplied else "null" if override is None else "object",
                               "effective_origin": "experiment.microstructure" if active_override else "device.grain_boundaries",
                               "legacy_empty_object_inherits": True, "grain_boundaries": grains,
                               "geometry_bounds_checked": True, "cell_overlap_fractions_computed": False},
            "carrier_boundary_condition": carrier_bc, "protocol": protocol,
            "compatibility": ["omitted_or_null_boundary_uses_effective_microstructure",
                              "empty_microstructure_object_inherits_but_explicit_empty_grain_array_clears",
                              "device_electrical_grid_retained_but_not_consumed_by_current_jv_2d",
                              "strict_scalar_validation_does_not_coerce_boolean_or_fractional_counts"]}


def _sweep(experiment: GrainSweepInput, defaults: DefaultCatalog, device: dict[str, Any],
           electrical: list[dict[str, Any]]) -> dict[str, Any]:
    values, origins = _controls(experiment, defaults)
    data = experiment.normalized_data()
    name = "grain_sizes_nm" if experiment.grain_sizes_nm else "grain_sizes"
    sizes = data.get(name)
    if not sizes:
        _fail(("experiment", "grain_sizes_nm"), "supply a nonempty grain_sizes_nm or grain_sizes array; empty/null is not a zero size", sizes)
    # This is the sweep's fixed protocol target, explicitly supplied to
    # GrainBoundary in voc_grain_sweep.py, not that class's fallback role.
    role = "absorber"
    target_ids = [layer["id"] for layer in electrical if layer["role"] == role]
    if not target_ids:
        _fail(("device", "layers"), "grain sweep requires the existing grain-boundary target layer role", role)
    for index, size in enumerate(sizes):
        if values["gb_width"] > size:
            _fail(("experiment", name, index), "grain sweep band width exceeds this declared lateral domain", size)
    values.update(grain_sizes_m=sizes, lateral_bc="neumann")
    origins.update(grain_sizes_m=f"input.experiment.{name}", lateral_bc="existing_grain_sweep_neumann_rule")
    return {"kind": experiment.kind, "controls": values, "field_origins": origins,
            "geometry": _geometry(values, electrical, device),
            "sweep_domains": [{"lateral_length_m": size, "lateral_bc": "neumann",
                               "band_declaration": {"x_position_m": size / 2, "width_m": values["gb_width"],
                                   "tau_n_s": values["tau_gb_n"], "tau_p_s": values["tau_gb_p"], "layer_role": role, "layer_ids": target_ids}}
                              for size in sizes],
            "protocol": {"status": "declared_sweep_controls_only", "execution_binding": "pending_per_domain_mesh_schedule_state_and_capabilities"},
            "compatibility": ["both_legacy_size_aliases_interpret_bare_numbers_as_nm",
                              "nonempty_grain_sizes_nm_precedes_grain_sizes",
                              "nonpositive_sizes_rejected_not_silently_filtered_as_in_legacy_sweep",
                              "centered_band_replaces_inherited_device_grain_inventory_for_each_domain",
                              "device_electrical_grid_retained_but_not_consumed_by_current_jv_2d"]}


@dataclass(frozen=True, slots=True)
class PreparedSpatialExperiment:
    input_json: bytes
    defaults: DefaultCatalog
    resources: ResourceLibrary
    sources: tuple[SourceDocument, ...] = ()
    can_execute: bool = field(default=False, init=False)
    _resolved_json: bytes = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if type(self.input_json) is not bytes or not self.input_json or len(self.input_json) > 2**23:
            raise ValueError("experiment declaration requires bounded immutable JSON bytes")
        if not isinstance(self.defaults, DefaultCatalog) or not isinstance(self.resources, ResourceLibrary):
            raise TypeError("experiment preparation requires an explicit trusted catalog and resource library")
        model = SpatialExperimentInput.model_validate(_document(self.input_json))
        sources = tuple(self.sources)
        device = resolve_device(model.device, self.defaults, self.resources, sources=sources)
        resolved_device = device.to_mapping()
        electrical = [layer for layer in resolved_device["layers"] if layer["role"] != "substrate"]
        experiment = (_jv(model.experiment, self.defaults, resolved_device, electrical) if isinstance(model.experiment, JV2DInput)
                      else _sweep(model.experiment, self.defaults, resolved_device, electrical))
        result = {"schema": "solarlab.resolved-spatial-experiment-preparation.v1", "unit_schema": UNIT_SCHEMA_VERSION,
                  "id": model.id, "status": "prepared_pending_dependencies", "can_execute": False,
                  "device": resolved_device, "experiment": experiment,
                  "identity": {"scope": "experiment_declaration_content", "device_content_sha256": device.content_sha256,
                      "default_catalog_sha256": self.defaults.content_sha256,
                      "resource_library_sha256": self.resources.content_sha256, "execution_identity": None,
                      "default_source_bindings": [list(pair) for pair in self.defaults.evidence],
                      "input_source_bindings": [{"id": source.id, "sha256": source.sha256} for source in sources]},
                  "capability_gaps": [*resolved_device["capability_gaps"], "two_dimensional_mesh_not_generated",
                      "two_dimensional_material_state_capabilities_not_qualified", "two_dimensional_protocol_not_execution_bound",
                      "qualified_two_dimensional_executor_not_registered"]}
        # These checks consume actual material-array/state outputs in the old
        # solver. Record their absence instead of inferring node capabilities
        # from a declaration or invoking that compiler during preview.
        result["unperformed_capability_checks"] = [
            "neutral_bulk_defect_2d_binding_is_unsupported",
            "mobile_ion_material_arrays_must_not_contain_dual_ion_species",
            "mobile_ion_nodes_need_positive_diffusion_density_and_adequate_site_limits",
            "mobile_ions_cannot_use_a_static_P_ion_background",
            "two_sided_interface_SRH_requires_qualified_sheet_area_and_couplings",
        ]
        result["validation_contract"] = {
            "scope": "strict_new_preparation_input_not_legacy_request_equivalence",
            "legacy_numerical_sources_changed": False,
            "differences": [
                "nonpositive_sweep_sizes_reject_instead_of_legacy_filtering",
                "band_bounds_are_strict_without_legacy_grid_roundoff_tolerance_or_clipping",
                "counts_and_booleans_are_strict_without_legacy_int_or_bool_coercion",
                "overlap_rejection_uses_shared_stable_layer_ids_while_legacy_uses_matching_roles",
            ],
            "preserved": ["input_presence_units_and_scalar_words", "legacy_truthy_microstructure_override_rule",
                          "legacy_nullable_boundary_rule", "fixed_absorber_role_of_grain_sweep"],
        }
        object.__setattr__(self, "sources", sources)
        object.__setattr__(self, "input_json", _bytes(model.model_dump(mode="json", exclude_unset=True)))
        object.__setattr__(self, "_resolved_json", _bytes(result))

    def to_input(self) -> SpatialExperimentInput:
        return SpatialExperimentInput.model_validate(json.loads(self.input_json))

    def to_mapping(self) -> dict[str, Any]:
        return json.loads(self._resolved_json)

    @property
    def content_sha256(self) -> str:
        return hashlib.sha256(_bytes([json.loads(self.input_json), json.loads(self._resolved_json)])).hexdigest()


def prepare_spatial_experiment(input: SpatialExperimentInput, defaults: DefaultCatalog,
                               resources: ResourceLibrary, *, sources: tuple[SourceDocument, ...] = ()) -> PreparedSpatialExperiment:
    checked = SpatialExperimentInput.model_validate(input)
    return PreparedSpatialExperiment(_bytes(checked.model_dump(mode="json", exclude_unset=True)), defaults, resources, sources)
