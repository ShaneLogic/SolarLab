"""Bounded serial declaration expansion through the actual prepared resolvers."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
from itertools import product
import json
from typing import Any

from pydantic import ValidationError
from solarlab.config.resolve_device import resolve_device
from solarlab.device.defaults import DefaultCatalog
from solarlab.device.inputs import DeviceInput
from solarlab.device.resolved import PreparedDevice
from solarlab.experiments.inputs import invalid, input_document
from solarlab.experiments.jv.inputs import JVExperimentInput
from solarlab.experiments.jv.preparation import PreparedJVExperiment, prepare_jv_experiment
from solarlab.materials.resources import ResourceLibrary
from solarlab.materials.source import SourceDocument
from solarlab.sweeps.inputs import SweepInput
from solarlab.sweeps.reference import SweepReference, baseline_signature
from solarlab.sweeps.targets import bind_target, effective_target, write_coordinate

MAX_PREVIEW_POINTS = 128  # Preview resource bound, not an executable scan limit.


def encode(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(encode(value)).hexdigest()


def prepare_point(raw: dict[str, Any], defaults: DefaultCatalog, resources: ResourceLibrary, sources: tuple[SourceDocument, ...]) -> PreparedDevice | PreparedJVExperiment:
    if raw["schema_version"] == "solarlab.device-preparation.v1":
        return resolve_device(DeviceInput.model_validate(raw), defaults, resources, sources=sources)
    return prepare_jv_experiment(JVExperimentInput.model_validate(raw), defaults, resources, sources=sources)


def _coordinate_key(value: dict[str, Any]) -> str:
    # Unit-equivalent numeric coordinates occupy one axis location. The input
    # token, its type and numeric zero sign remain in the original declaration.
    canonical = deepcopy(value)
    raw = canonical.get("value")
    if type(raw) in (int, float):
        canonical["value"] = float(raw) if raw else 0.0
    return encode(canonical).decode()


@dataclass(frozen=True, slots=True)
class PreparedSweep:
    input_json: bytes
    defaults: DefaultCatalog
    resources: ResourceLibrary
    sources: tuple[SourceDocument, ...] = ()
    references: tuple[SweepReference, ...] = ()
    can_execute: bool = field(default=False, init=False)
    _resolved_json: bytes = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if type(self.input_json) is not bytes or not self.input_json or len(self.input_json) > 2**23:
            raise ValueError("sweep input requires bounded immutable JSON bytes")
        object.__setattr__(self, "sources", tuple(self.sources))
        object.__setattr__(self, "references", tuple(self.references))
        if any(not isinstance(item, SweepReference) for item in self.references) or len({item.id for item in self.references}) != len(self.references):
            raise ValueError("sweep reference identities must be unique")
        model = SweepInput.model_validate(input_document(self.input_json))
        original = model.model_dump(mode="json", exclude_unset=True)
        raw_base = original["base"]
        total = 1
        for axis in model.axes:
            total *= len(axis.coordinates)
        if total > MAX_PREVIEW_POINTS:
            invalid("SweepPreparation", ("axes",), f"preview permits at most {MAX_PREVIEW_POINTS} points; the declared {total} points were not truncated or executed", total)
        bindings, normalized, invalid_coordinates = [], [], {}
        for index, axis in enumerate(model.axes):
            try:
                binding = bind_target(model.base, axis.target)
            except ValueError as error:
                invalid("SweepPreparation", ("axes", index, "target"), str(error), axis.target.editing_data())
            bindings.append(binding)
            values, seen = [], set()
            for position, coordinate in enumerate(axis.coordinates):
                try:
                    value = binding.coordinate(coordinate)
                except ValueError as error:
                    invalid_coordinates[index, position] = str(error)
                    value = {"kind": "invalid", "declared": coordinate.editing_data()}
                key = _coordinate_key(value)
                if key in seen:
                    invalid("SweepPreparation", ("axes", index, "coordinates", position), "duplicate physical coordinate, including equivalent units or signed-zero aliases", coordinate.editing_data())
                seen.add(key)
                values.append(value)
            normalized.append(values)
        writes = [binding.input_path for binding in bindings]
        if len(set(writes)) != len(writes):
            invalid("SweepPreparation", ("axes",), "axes address the same parameter; no last-write-wins ordering", writes)
        reference_affinity = {}
        for index, binding in enumerate(bindings):
            if binding.reference_layer is None:
                continue
            base_device = model.base.device if isinstance(model.base, JVExperimentInput) else model.base
            anchor_input = next(layer for layer in base_device.layers if layer.id == binding.reference_layer)
            if any(other.field == "chi" and (other.target.owner_id == binding.reference_layer
                or other.target.family == "material_parameter" and other.target.owner_id == anchor_input.material) for other in bindings):
                invalid("SweepPreparation", ("axes", index, "target"), "CBO reference affinity cannot be another sweep axis", binding.reference_layer)
            prepared_base = resolve_device(base_device, self.defaults, self.resources, sources=self.sources)
            affinity = dict(next(layer for layer in prepared_base.layers if layer.id == binding.reference_layer).parameters)["chi"]
            assert isinstance(affinity, (int, float))
            reference_affinity[index] = affinity
        supplied_targets = {axis.id: axis.target.model_dump(mode="json", exclude_unset=True) for axis in model.axes}
        reference = next((item for item in self.references if item.id == model.reference_id), None)
        baseline_key = baseline_signature(raw_base) if reference is not None else ""
        points = []
        for positions in product(*(range(len(axis.coordinates)) for axis in model.axes)):
            raw = deepcopy(raw_base)
            declared = {axis.id: original["axes"][index]["coordinates"][position] for index, (axis, position) in enumerate(zip(model.axes, positions))}
            coordinates = {axis.id: normalized[index][position] for index, (axis, position) in enumerate(zip(model.axes, positions))}
            point_id = "point:" + digest([raw_base, supplied_targets, declared])
            point: dict[str, Any] = dict(id=point_id, coordinates=declared, normalized_coordinates=coordinates,
                reference=reference.match(supplied_targets, coordinates, baseline_key) if reference is not None else dict(status="not_requested" if model.reference_id is None else "reference_unavailable", reason="no trusted reference supplied", comparison_qualified=False),
                can_execute=False, execution_identity=None, simulation_status="not_started")
            applied = True
            try:
                for index, (axis, position, binding) in enumerate(zip(model.axes, positions, bindings)):
                    if (index, position) in invalid_coordinates and binding.reference_layer is not None:
                        applied = False
                        continue
                    write_coordinate(raw, binding, axis.coordinates[position], reference_affinity.get(index))
                for index, (axis, position, binding) in enumerate(zip(model.axes, positions, bindings)):
                    if (index, position) in invalid_coordinates and binding.reference_layer is not None:
                        invalid("SweepPoint", ("axes", index, "coordinates", position), invalid_coordinates[index, position], axis.coordinates[position].editing_data())
                # Every materialized point, including invalid direct field
                # edits, passes the actual full device/J-V input resolver.
                prepared = prepare_point(raw, self.defaults, self.resources, self.sources)
                resolved = prepared.to_mapping()
                point.update(status="prepared_pending_dependencies", content_sha256=prepared.content_sha256,
                    resolved_sha256=digest(resolved), capability_gaps=resolved["capability_gaps"],
                    effective_targets={axis.id: effective_target(binding, resolved) for axis, binding in zip(model.axes, bindings)})
            except (ValueError, OverflowError) as error:
                fields = error.errors(include_url=False, include_context=False) if isinstance(error, ValidationError) else [{"loc": ["base"], "type": "configuration_resolution", "msg": str(error)}]
                point.update(status="unresolved", field_errors=fields, effective_targets={})
            point["applied_input"] = raw if applied else None
            points.append(point)
        reference_scope = reference.to_mapping() if reference is not None else None
        result = dict(schema="solarlab.resolved-sweep-preparation.v1", id=model.id,
            status="prepared_pending_dependencies", can_execute=False, execution_identity=None,
            expansion=dict(mode="serial_declaration", declared_points=total, expanded_points=len(points), preview_limit=MAX_PREVIEW_POINTS, truncated=False, numerical_calls=0),
            axes=[dict(id=axis.id, target=supplied_targets[axis.id], stable_path=binding.stable_path,
                input_field_schema=binding.metadata, required=binding.required, normalized_coordinates=values,
                transform="reference_affinity_eV - CBO_eV; explicit decimal subtraction" if binding.reference_layer else "selected_field_only") for axis, binding, values in zip(model.axes, bindings, normalized)],
            points=points, reference_scope={key: value for key, value in reference_scope.items() if key not in {"rows", "targets"}} if reference_scope else None,
            summary=dict(prepared=sum(p["status"]=="prepared_pending_dependencies" for p in points), unresolved=sum(p["status"]=="unresolved" for p in points)),
            identity=dict(scope="sweep_declaration_content", execution_identity=None, default_catalog_sha256=self.defaults.content_sha256,
                resource_library_sha256=self.resources.content_sha256, source_bindings=[dict(id=s.id, sha256=s.sha256) for s in self.sources]),
            capability_gaps=["qualified_executor_not_registered", "serial_numerical_rebind_not_qualified", "no_point_reuse_or_resume_qualification", "no_scientific_scan_comparison"],
            compatibility=["all_unrelated_input_and_partner_metadata_retained", "no_automatic_V_bi_or_tau_update", "no_doping_clear_or_ion_inventory_removal", "D_zero_and_P0_zero_are_distinct", "pair_scan_is_parameter_space_not_two_dimensional_device_geometry"])
        object.__setattr__(self, "input_json", encode(original))
        object.__setattr__(self, "_resolved_json", encode(result))

    def to_input(self) -> SweepInput:
        return SweepInput.model_validate(input_document(self.input_json))

    def to_mapping(self) -> dict[str, Any]:
        return json.loads(self._resolved_json)

    @property
    def content_sha256(self) -> str:
        return digest([input_document(self.input_json), self.to_mapping()])


def prepare_sweep(input: SweepInput, defaults: DefaultCatalog, resources: ResourceLibrary, *, sources: tuple[SourceDocument, ...] = (), references: tuple[SweepReference, ...] = ()) -> PreparedSweep:
    model = SweepInput.model_validate(input)
    return PreparedSweep(encode(model.model_dump(mode="json", exclude_unset=True)), defaults, resources, sources, references)
