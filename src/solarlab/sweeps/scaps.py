"""Bind the thirteen frozen SCAPS coordinate groups to existing stable DTO IDs.

This is an input adapter, not the older combined-SRH sweep or a comparator.
Reported interface density units and the historical areal-input hypothesis are
kept separate. There is no invented thickness, Gaussian normalization or fit.
"""
from __future__ import annotations

from copy import deepcopy
from decimal import Decimal
import json
from typing import Any, Literal

from solarlab.device.inputs import DeviceInput
from solarlab.experiments.jv.inputs import JVExperimentInput
from solarlab.materials.source import SourceDocument
from solarlab.sweeps.inputs import SweepInput, SweepCoordinateInput, SweepTargetInput
from solarlab.sweeps.preparation import digest, encode
from solarlab.sweeps.reference import SweepReference, baseline_signature
from solarlab.sweeps.targets import bind_target

# Coordinate names and field bindings follow scaps_reference.case_document;
# values and shapes are read from the supplied frozen manifest/contract.
_GROUPS = {
    "scaps_cbo": (("CBO", "etl", "chi", "eV"),),
    "scaps_etl_donor": (("N_D", "etl", "N_D", "cm^-3"),),
    "scaps_pvk_donor": (("N_D", "pvk", "N_D", "cm^-3"),),
    "scaps_nt_htl_pvk": (("Nt", "left", "total_density_m2", "cm^-2"),),
    "scaps_nt_pvk_cb": (("Nt", "cb", "distribution.total_density_m3", "cm^-3"),),
    "scaps_nt_pvk_vb": (("Nt", "vb", "distribution.total_density_m3", "cm^-3"),),
    "scaps_nt_pvk_etl": (("Nt", "right", "total_density_m2", "cm^-2"),),
    "scaps_et_htl_pvk": (("Et", "left", "trap_depth_eV", "eV"),),
    "scaps_et_pvk_cb": (("Et", "cb", "energy_level.value_eV", "eV"),),
    "scaps_et_pvk_vb": (("Et", "vb", "energy_level.value_eV", "eV"),),
    "scaps_et_pvk_etl": (("Et", "right", "trap_depth_eV", "eV"),),
    "scaps_nt_et": (("Nt", "right", "total_density_m2", "cm^-2"), ("Et", "right", "trap_depth_eV", "eV")),
    "scaps_nt_cbo": (("Nt", "right", "total_density_m2", "cm^-2"), ("CBO", "etl", "chi", "eV")),
}


def _one(items: tuple[Any, ...], name: str, field: str = "name") -> Any:
    found = [item for item in items if getattr(item, field) == name]
    if len(found) != 1:
        raise ValueError(f"SCAPS binding needs one supplied {field}={name}; no fallback by array index")
    return found[0]


def scaps_sweeps(base: DeviceInput | JVExperimentInput, manifest: SourceDocument,
                 contract: SourceDocument, *, interface_density_interpretation: Literal["historical_areal_input"]
                 ) -> tuple[tuple[SweepInput, ...], tuple[SweepReference, ...]]:
    """Return declarations using an explicitly chosen interface-density channel.

    This choice is an input interpretation, not a dimensional conversion or
    evidence of the missing native SCAPS input. There is no default channel.
    """
    if interface_density_interpretation != "historical_areal_input":
        raise ValueError("SCAPS interface density requires the explicit historical_areal_input interpretation; native volume/areal input is not established")
    device = base.device if isinstance(base, JVExperimentInput) else base
    data, standard = json.loads(manifest.content), json.loads(contract.content)
    groups = data["scaps"]["groups"]
    if digest(groups) != standard["reference_set"]["manifest_groups_payload_sha256"]:
        raise ValueError("SCAPS coordinates differ from the supplied frozen comparison contract")
    if standard["independent_review"]["status"] != "approved" or not standard["frozen_at"]:
        raise ValueError("SCAPS comparison contract has no completed source review")
    if len(groups) != len(_GROUPS) or {g["id"] for g in groups} != set(_GROUPS):
        raise ValueError("the thirteen distinct SCAPS groups must be present")
    pvk, etl, htl = (_one(device.layers, name) for name in ("SCAPS_PVK", "SCAPS_ETL", "SCAPS_HTL"))
    cb, vb = (_one(pvk.bulk_defects, name) for name in ("Perovskite-CB", "Perovskite-VB"))
    joins = {}
    for name, left, right in (("left", htl.id, pvk.id), ("right", pvk.id, etl.id)):
        found = [item for item in device.interfaces if item.left == left and item.right == right]
        if len(found) != 1 or found[0].defect is None:
            raise ValueError("SCAPS interface binding requires a supplied directed interface and defect")
        joins[name] = found[0]
    declarations, references = [], []
    for group in groups:
        rules, axes, bindings = _GROUPS[group["id"]], [], []
        if [axis["name"] for axis in group["axes"]] != [rule[0] for rule in rules]:
            raise ValueError("SCAPS named-axis order changed")
        for axis_index, (axis, location, field, unit) in enumerate(rules):
            if location in {"etl", "pvk"}:
                target: dict[str, Any] = dict(family="cbo" if axis == "CBO" else "layer_parameter", owner_id=etl.id if location == "etl" else pvk.id, parameter=field.split('.'))
                if axis == "CBO":
                    target["reference_id"] = pvk.id
            elif location in {"cb", "vb"}:
                target = dict(family="bulk_defect", owner_id=pvk.id, local_id=cb.id if location == "cb" else vb.id, parameter=field.split('.'))
            else:
                join = joins[location]
                assert join.defect is not None
                target = dict(family="interface_defect", owner_id=join.id, local_id=join.defect.id, parameter=[field])
            binding = bind_target(base, SweepTargetInput.model_validate(target))
            bindings.append(binding)
            axis_coordinates, seen = [], set()
            for point in group["points"]:
                original = point["coordinates"][axis_index]
                if type(original) not in (int, float):
                    raise ValueError("frozen SCAPS coordinates must be numeric")
                axis_key = Decimal(str(original))
                if axis_key not in seen:
                    axis_coordinates.append(dict(kind="value", value=f"{original} {unit}"))
                    seen.add(axis_key)
            axes.append(dict(id=axis, target=target, coordinates=axis_coordinates))
        if len(group["points"]) != group["expected_points"]:
            raise ValueError("SCAPS reference points are missing")
        if len(axes[0]["coordinates"]) * (len(axes[1]["coordinates"]) if len(axes) == 2 else 1) != group["expected_points"]:
            raise ValueError("SCAPS explicit coordinate product does not match the supplied reference shape")
        rows, row_keys = [], set()
        for point in group["points"]:
            if point["status"] not in {"ok", "reference_missing"} or point["status"] == "reference_missing" and not point.get("reason"):
                raise ValueError("each reference status requires its original missing reason")
            row_coordinates = {rule[0]: binding.coordinate(SweepCoordinateInput(kind="value", value=f"{point['coordinates'][i]} {rule[3]}")) for i, (rule, binding) in enumerate(zip(rules, bindings))}
            row_key = digest(row_coordinates)
            if row_key in row_keys:
                raise ValueError("duplicate SCAPS reference coordinate")
            row_keys.add(row_key)
            rows.append(dict(coordinates=row_coordinates, reported_coordinates=deepcopy(point["coordinates"]), worksheet_row=point["worksheet_row"],
                status="available" if point["status"] == "ok" else "reference_missing", reason=point.get("reason"),
                solver_execution_status=point.get("solver_execution_status", "not_established_by_reference_table"),
                **({"original_delta_E_C_eV_cell": point["original_delta_E_C_eV_cell"]} if "original_delta_E_C_eV_cell" in point else {})))
        if sum(row["status"] == "available" for row in rows) != group["numeric_points"]:
            raise ValueError("SCAPS reference count differs from frozen metadata")
        ref_id = group["id"]
        reference_document = dict(targets={axis["id"]: axis["target"] for axis in axes}, rows=rows,
            baseline_input_sha256=baseline_signature(base.model_dump(mode="json", exclude_unset=True)),
            baseline_comparison="exact_json_values_with_number_spelling_independent_and_signed_zero_sensitive",
            missing_original_inputs=deepcopy(data["scaps"]["missing_original_inputs"]),
            interpretation=dict(reported_axes=group["axes"], effective_coordinate_units=[rule[3] for rule in rules],
                interface_density_choice=interface_density_interpretation,
                interface_density_axes=[dict(axis_id=rule[0], reported_unit=group["axes"][index]["unit_as_reported"],
                    input_unit=rule[3], resolved_unit=binding.metadata["unit"], thickness_conversion=False, native_input_verified=False)
                    for index, (rule, binding) in enumerate(zip(rules, bindings))
                    if binding.target.family == "interface_defect" and binding.field == "total_density_m2"],
                interface_density="historical areal-input hypothesis; reported cm^-3 numerals used in cm^-2 fields with no thickness conversion",
                energy="supplied CB/VB authorities remain distinct; partner metadata remains historical and is not overwritten",
                gaussian="total/peak/width uncertainty retained; no normalization or calibration repair",
                protocol="historical steady/transient/ion-state differences are not qualified by preparation",
                source_contract_sha256=contract.sha256, worksheet=group["worksheet"], report_page=group["pdf_page"]))
        references.append(SweepReference(ref_id, manifest.id, manifest.sha256, encode(reference_document)))
        declarations.append(SweepInput.model_validate(dict(schema_version="solarlab.sweep-preparation.v1", id=ref_id,
            base=base.model_dump(mode="json", exclude_unset=True), axes=axes, reference_id=ref_id)))
    return tuple(declarations), tuple(references)
