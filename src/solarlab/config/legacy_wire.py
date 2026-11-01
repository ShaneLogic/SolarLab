"""Source-bound reverse wire preparation; no legacy execution or rollout.

This single transitional adapter belongs to the P06/P09 exit boundary. It
applies edits to detached standard/SCAPS/tandem documents, then checks them
with the existing strict importers. Original evidence and defaults are kept
separate from emitted data; a data roundtrip is not old-driver equivalence.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from decimal import Decimal, localcontext
import json
import math
from typing import Any

import yaml

from solarlab.config.behavior import BehaviorContext
from solarlab.config.device_import import import_standard_device
from solarlab.config.legacy_fields import STANDARD_LOADER_BINDING
from solarlab.config.scaps_input import _FIELDS as SCAPS_FIELDS, import_scaps_device
from solarlab.config.tandem_input import import_tandem
from solarlab.config.yaml import load_yaml_mapping
from solarlab.device.defaults import DefaultCatalog
from solarlab.device.inputs import DeviceInput, TandemInput
from solarlab.materials.parameters import EditableInput
from solarlab.materials.source import SourceDocument
from solarlab.units import normalize_quantity

__all__ = ["LEGACY_WIRE_VERSION", "LegacyWire", "prepare_legacy_wire"]

LEGACY_WIRE_VERSION = "solarlab.legacy-wire-preparation.v1"
_ABSENT = object()


class _Unsupported(ValueError):
    def __init__(self, path: str, code: str, message: str):
        super().__init__(message)
        self.issue = {"path": path, "code": code, "message": message}


def _quantity(value: object, unit: str) -> float:
    number = normalize_quantity(value, unit)
    # The shared resolver deliberately normalizes signed zero. Wire editing
    # must retain the declaration's sign without changing that resolver.
    negative = (str(value).lstrip().startswith("-") if isinstance(value, (str, Decimal))
                else type(value) is float and math.copysign(1, value) < 0)
    return -0.0 if number == 0 and negative else number


def _data(value: Any) -> Any:
    """Present fields and units, retaining bool/null/omission and zero signs."""
    if isinstance(value, EditableInput):
        result: dict[str, Any] = {}
        for name in value.editing_data():
            child = getattr(value, name)
            extra = type(value).model_fields[name].json_schema_extra
            if child is not None and isinstance(extra, dict) and "unit" in extra:
                result[name] = _quantity(child, str(extra["unit"]))
            elif child is not None and isinstance(extra, dict) and "item_unit" in extra:
                result[name] = [_quantity(item, str(extra["item_unit"])) for item in child]
            else:
                result[name] = _data(child)
        return result
    if isinstance(value, Mapping):
        return {key: _data(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [_data(child) for child in value]
    return value


def _same(left: Any, right: Any) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_same(left[k], right[k]) for k in left)
    if isinstance(left, (tuple, list)) and isinstance(right, (tuple, list)):
        return len(left) == len(right) and all(_same(a, b) for a, b in zip(left, right))
    if type(left) is bool or type(right) is bool or left is None or right is None:
        return type(left) is type(right) and left == right
    if isinstance(left, (int, float, Decimal)) and isinstance(right, (int, float, Decimal)):
        a, b = Decimal(str(left)), Decimal(str(right))
        return a == b and (bool(a) or a.is_signed() == b.is_signed())
    return type(left) is type(right) and left == right


def _patch(wire: Any, before: Any, after: Any) -> Any:
    """Change supplied fields only; do not serialize inherited defaults."""
    if _same(before, after):
        return deepcopy(wire)
    if isinstance(before, dict) and isinstance(after, dict):
        result = deepcopy(wire) if isinstance(wire, dict) else {}
        for key in sorted(before.keys() | after.keys()):
            if key not in after:
                result.pop(key, None)
            elif key not in before:
                result[key] = deepcopy(after[key])
            elif not _same(before[key], after[key]):
                result[key] = _patch(result.get(key), before[key], after[key])
        return result
    if isinstance(before, list) and isinstance(after, list) and len(before) == len(after):
        original = wire if isinstance(wire, list) else []
        return [_patch(original[i] if i < len(original) else None, a, b)
                for i, (a, b) in enumerate(zip(before, after))]
    return deepcopy(after)


def _differences(before: Any, after: Any, path: str = "$") -> list[dict[str, Any]]:
    if _same(before, after):
        return []
    if isinstance(before, dict) and isinstance(after, dict):
        return [row for key in sorted(before.keys() | after.keys())
                for row in _differences(before.get(key, _ABSENT), after.get(key, _ABSENT), path + "." + key)]
    if isinstance(before, list) and isinstance(after, list) and len(before) == len(after):
        return [row for i, (a, b) in enumerate(zip(before, after))
                for row in _differences(a, b, f"{path}[{i}]")]
    return [{"path": path, "before_present": before is not _ABSENT, "after_present": after is not _ABSENT,
             **({"before": before} if before is not _ABSENT else {}),
             **({"after": after} if after is not _ABSENT else {})}]


def _wire_quantity(value: Any, canonical: str, wire_unit: str) -> Any:
    if value is None or wire_unit == canonical:
        return value
    # Invert only an existing whitelisted conversion, without another unit
    # registry or a rounded intermediate CGS binary64 value. The shortest
    # decimal representation roundtrips to the declared SI binary64 value.
    # The strict importer combines the token and rational factor before float.
    factor = Decimal(str(normalize_quantity(1, canonical, input_unit=wire_unit)))
    with localcontext() as context:
        context.prec = 50
        return Decimal(str(value)) / factor


class _WireDumper(yaml.SafeDumper):
    pass


_WireDumper.add_representer(Decimal, lambda dumper, value:
                          dumper.represent_scalar("tag:yaml.org,2002:float", str(value)))


def _document(original: SourceDocument, wire: dict[str, Any]) -> SourceDocument:
    if _same(load_yaml_mapping(original.content), wire):
        return original
    content = yaml.dump(wire, Dumper=_WireDumper, sort_keys=False, allow_unicode=True).encode()
    return SourceDocument(original.id, content)


def _binding(source: SourceDocument) -> dict[str, Any]:
    return {"id": source.id, "sha256": source.sha256, "bytes": len(source.content)}


def _flat_defect(value: dict[str, Any], *, standard: bool, path: str) -> dict[str, Any]:
    kinetics = value["kinetics"]
    if not _same(kinetics["thermal_velocity_n_m_s"], kinetics["thermal_velocity_p_m_s"]):
        raise _Unsupported(path + ".kinetics", "single_thermal_velocity", "The old flat interface has one v_th_cm_s for both carriers.")
    below = value["energy_reference"] == "below_conduction_band"
    if standard and (not below or value.get("partner_metadata") is not None):
        raise _Unsupported(path, "standard_flat_interface_schema", "The standard wire accepts below-conduction-band energy and its exact flat interface fields only.")
    result = {"sigma_n_cm2": _wire_quantity(kinetics["sigma_n_m2"], "m^2", "cm^2"),
              "sigma_p_cm2": _wire_quantity(kinetics["sigma_p_m2"], "m^2", "cm^2"),
              "v_th_cm_s": _wire_quantity(kinetics["thermal_velocity_n_m_s"], "m/s", "cm/s"),
              "N_t_cm2": _wire_quantity(value["total_density_m2"], "m^-2", "cm^-2"),
              "E_t_eV_below_cb" if below else "E_t_eV_above_vb": value["trap_depth_eV"],
              "calibration_factor": value["calibration_factor"],
              "iface_state_calibration_factor": value["iface_state_calibration_factor"]}
    metadata = value.get("partner_metadata")
    if metadata is not None:
        result["distribution"] = metadata["distribution"]
        for key, old, unit, wire_unit in (("E_char_eV", "E_char_eV", "eV", "eV"),
                                         ("N_peak_m3", "N_peak_cm3", "m^-3", "cm^-3")):
            if key in metadata:
                result[old] = _wire_quantity(metadata[key], unit, wire_unit)
    return result


def _scaps_bulk(value: dict[str, Any], v_th: Any, path: str) -> dict[str, Any]:
    distribution, kinetics = value["distribution"], value["kinetics"]
    if (distribution["kind"] not in {"single_level", "gaussian"}
            or value.get("energy_level") is None or value.get("spatial_profile") is not None
            or any(not _same(kinetics[key], v_th) for key in ("thermal_velocity_n_m_s", "thermal_velocity_p_m_s"))):
        raise _Unsupported(path, "scaps_bulk_representation", "SCAPS wire needs referenced single/Gaussian energy, no spatial profile and the layer's shared thermal velocity.")
    if distribution["kind"] == "gaussian" and distribution.get("width_convention") not in {"unresolved", "scaps_characteristic_energy"}:
        raise _Unsupported(path + ".distribution", "scaps_width_convention", "SCAPS wire does not encode this Gaussian width convention.")
    energy = value["energy_level"]
    result = {"sigma_n_cm2": _wire_quantity(kinetics["sigma_n_m2"], "m^2", "cm^2"),
              "sigma_p_cm2": _wire_quantity(kinetics["sigma_p_m2"], "m^2", "cm^2"),
              "N_t_cm3": _wire_quantity(distribution["total_density_m3"], "m^-3", "cm^-3"),
              "E_t_eV_below_cb" if energy["reference"] == "below_conduction_band" else "E_t_eV_above_vb": energy["value_eV"],
              "distribution": "single" if distribution["kind"] == "single_level" else "gaussian",
              "charge_transition": value["charge_transition"], "neutral_reference": value["neutral_reference"],
              "degeneracy": value["degeneracy"]}
    if value.get("name") is not None:
        result["name"] = value["name"]
    if "width_eV" in distribution:
        result["E_char_eV"] = distribution["width_eV"]
    return result


def _scaps_layer(raw: dict[str, Any], before: dict[str, Any], after: dict[str, Any], path: str) -> dict[str, Any]:
    inverse = {name: (old, unit, wire_unit) for old, (name, unit, wire_unit) in SCAPS_FIELDS.items()}
    profile = {"trap_N_t_interface": ("N_t_peak_cm3", "m^-3", "cm^-3"),
               "trap_decay_length": ("decay_length_nm", "m", "nm"),
               "trap_profile_shape": ("profile", None, None), "trap_edge": ("target", None, None)}
    a, b = before["parameters"], after["parameters"]
    for name in sorted(a.keys() | b.keys()):
        if _same(a.get(name, _ABSENT), b.get(name, _ABSENT)):
            continue
        if name in inverse:
            old, unit, wire_unit = inverse[name]
            target = raw
        elif name in {"optical_material", "incoherent", "n_optical"}:
            old, unit, wire_unit, target = name, None, None, raw
        elif name in profile:
            old, unit, wire_unit = profile[name]
            if raw.get("interface_defect") is None:
                raw["interface_defect"] = {}
            target = raw["interface_defect"]
        else:
            raise _Unsupported(path + ".parameters." + name, "scaps_parameter_not_encodable", "This typed parameter has no field in the strict SCAPS wire schema.")
        if name not in b:
            target.pop(old, None)
        else:
            target[old] = (_wire_quantity(b[name], unit, wire_unit)
                           if unit is not None and wire_unit is not None else b[name])
    if not _same(before.get("bulk_defects", []), after.get("bulk_defects", [])):
        singular = "bulk_defect" in raw
        old_blocks = [raw["bulk_defect"]] if singular and raw["bulk_defect"] is not None else raw.get("bulk_defects", [])
        blocks = []
        for i, value in enumerate(after.get("bulk_defects", [])):
            target = _scaps_bulk(value, b["v_th"], f"{path}.bulk_defects[{i}]")
            previous = _scaps_bulk(before["bulk_defects"][i], a["v_th"], path) if i < len(before.get("bulk_defects", [])) else {}
            blocks.append(_patch(old_blocks[i] if i < len(old_blocks) else {}, previous, target))
        if singular and len(blocks) == 1:
            raw["bulk_defect"] = blocks[0]
        else:
            raw.pop("bulk_defect", None)
            raw["bulk_defects"] = blocks
    # Partner metadata is retained independently. A divergent duplicate cannot
    # disappear: the complete strict roundtrip below detects it.
    return raw


def _device_wire(after: DeviceInput, before: DeviceInput, source: SourceDocument) -> dict[str, Any]:
    a, b = _data(before), _data(after)
    wire = deepcopy(load_yaml_mapping(source.content))
    if after.source_format not in {before.source_format, "canonical"}:
        raise _Unsupported("$.source_format", "source_format_changed", "Rebase through an explicitly selected strict importer before changing the source format.")
    if after.materials or any(layer.material is not None for layer in after.layers):
        raise _Unsupported("$.materials", "material_inheritance", "Legacy layers have no named-material reference/override graph; it cannot be silently flattened.")
    if [layer.id for layer in after.layers] != [layer.id for layer in before.layers]:
        raise _Unsupported("$.layers", "source_topology_changed", "A changed layer inventory/order needs a newly bound source before reusing its historical interface slots.")
    old_legacy = before.legacy_fields.model_dump(mode="json", exclude_unset=True) if before.legacy_fields else None
    new_legacy = after.legacy_fields.model_dump(mode="json", exclude_unset=True) if after.legacy_fields else None
    if not _same(old_legacy, new_legacy):
        raise _Unsupported("$.legacy_fields", "retained_evidence_changed", "Retained fields must match original source bytes; edit canonical settings/interfaces instead.")
    for name in ("name", "description", "simulation_hints"):
        wire = _patch(wire, {name: a[name]} if name in a else {}, {name: b[name]} if name in b else {})
    wire = _patch(wire, {"schema_version": a["source_schema_version"]} if "source_schema_version" in a else {},
                  {"schema_version": b["source_schema_version"]} if "source_schema_version" in b else {})
    dev = wire["device"]
    settings_a, settings_b = a.get("settings", {}), b.get("settings", {})
    dev = _patch(dev, settings_a, settings_b)
    if "V_bi_override" in dev and not _same(settings_a.get("V_bi", _ABSENT), settings_b.get("V_bi", _ABSENT)):
        if "V_bi" in settings_b:
            dev["V_bi_override"] = settings_b["V_bi"]
            if "V_bi" not in load_yaml_mapping(source.content)["device"]:
                dev.pop("V_bi", None)
        else:
            dev.pop("V_bi_override", None)
    wire["device"] = dev
    standard = before.source_format == "standard"
    for i, (left, right) in enumerate(zip(a["layers"], b["layers"])):
        row = wire["layers"][i]
        for name in ("id", "name", "role"):
            row = _patch(row, {name: left[name]}, {name: right[name]})
        thickness = "thickness" if standard else "thickness_nm"
        row = _patch(row, {thickness: _wire_quantity(left["thickness"], "m", "m" if standard else "nm")},
                     {thickness: _wire_quantity(right["thickness"], "m", "m" if standard else "nm")})
        if standard:
            row = _patch(row, left["parameters"], right["parameters"])
        else:
            row = _scaps_layer(row, left, right, f"$.layers[{i}]")
        names: tuple[str, ...] = ("defect_schema_version", "defect_model", "bulk_trap_distribution", "cigs_graded_optics")
        if standard:
            names += ("bulk_defects",)
        row = _patch(row, {k: left[k] for k in names if k in left}, {k: right[k] for k in names if k in right})
        wire["layers"][i] = row
    if len(a["contacts"]) != len(b.get("contacts", [])):
        raise _Unsupported("$.contacts", "contact_inventory", "The old wire has exactly one contact at each electrical endpoint.")
    for old, new in zip(a["contacts"], b["contacts"]):
        if any(old[k] != new[k] for k in ("id", "side", "layer")):
            raise _Unsupported("$.contacts", "contact_identity", "The old wire fixes contact identity and attachment to the electrical endpoints.")
        nested = dev.get("contacts", {}).get(new["side"], {})
        for carrier in ("S_n", "S_p"):
            if _same(old.get(carrier, _ABSENT), new.get(carrier, _ABSENT)):
                continue
            flat = carrier + "_" + new["side"]
            targets = [(dev, flat)] if flat in dev or carrier not in nested else []
            if carrier in nested:
                targets.append((nested, carrier))
            for target, key in targets:
                if carrier in new:
                    target[key] = new[carrier]
                else:
                    target.pop(key, None)
    if len(a["interfaces"]) != len(b.get("interfaces", [])):
        raise _Unsupported("$.interfaces", "interface_inventory", "The old wire requires every full-layer adjacency, including substrates.")
    for i, (old, new) in enumerate(zip(a["interfaces"], b["interfaces"])):
        path = f"$.interfaces[{i}]"
        if any(old[k] != new[k] for k in ("id", "left", "right")):
            raise _Unsupported(path, "interface_identity", "Nonadjacent/reindexed interfaces cannot reuse an original full-layer slot.")
        pair_changed = any(not _same(old.get(k, _ABSENT), new.get(k, _ABSENT)) for k in ("v_n", "v_p"))
        if pair_changed:
            if not standard or any(k not in new or new[k] is None for k in ("v_n", "v_p")):
                raise _Unsupported(path, "velocity_pair_representation", "The standard wire needs two numeric velocities; SCAPS directed defects have no bare velocity-pair field.")
            pairs = dev.get("interfaces") or []
            while len(pairs) <= i:
                pairs.append([0.0, 0.0])
            pairs[i] = [new["v_n"], new["v_p"]]
            dev["interfaces"] = pairs
        if not _same(old.get("defect", _ABSENT), new.get("defect", _ABSENT)):
            if standard:
                blocks = dev.get("interface_defects") or []
                while len(blocks) <= i:
                    blocks.append(None)
                if "defect" not in new:
                    raise _Unsupported(path + ".defect", "aligned_defect_omission", "An absent interior slot cannot be silently changed into explicit null.")
                blocks[i] = (None if new["defect"] is None else _patch(blocks[i],
                    _flat_defect(old["defect"], standard=True, path=path) if old.get("defect") else {},
                    _flat_defect(new["defect"], standard=True, path=path)))
                dev["interface_defects"] = blocks
            else:
                targets = [b["layers"][i]["role"].lower(), b["layers"][i + 1]["role"].lower()]
                from solarlab.config.scaps_input import _ROLE_ALIAS
                blocks = wire.get("interfaces", [])
                matches = [block for block in blocks if [_ROLE_ALIAS.get(x.strip().lower(), x.strip().lower())
                           for x in block["target"].split("/")] == targets]
                if len(matches) != 1 or not new.get("defect"):
                    raise _Unsupported(path + ".defect", "scaps_directed_slot", "A SCAPS defect edit requires one existing unambiguous directed slot and a non-null defect.")
                replacement = _patch(matches[0], _flat_defect(old["defect"], standard=False, path=path),
                                     _flat_defect(new["defect"], standard=False, path=path))
                blocks[blocks.index(matches[0])] = replacement
    if not _same(a.get("grain_boundaries", []), b.get("grain_boundaries", [])):
        rows = []
        for i, grain in enumerate(b.get("grain_boundaries", [])):
            roles = {layer["role"] for layer in b["layers"] if layer["id"] in grain["layer_ids"]}
            role = grain.get("source_layer_role")
            if len(roles) != 1 or role not in roles or set(grain["layer_ids"]) != {layer["id"] for layer in b["layers"] if layer["role"] == role and role != "substrate"}:
                raise _Unsupported(f"$.grain_boundaries[{i}]", "grain_role_scope", "The old wire selects all electrical layers of one role, not an arbitrary subset.")
            rows.append({"id": grain["id"], "layer_role": role, **{k: grain[k] for k in ("x_position", "width", "tau_n", "tau_p")}})
        wire["microstructure"] = {"grain_boundaries": rows}
    if not _same(a.get("electrical_grid", []), b.get("electrical_grid", [])):
        grid_names = {layer["id"]: layer["name"] for layer in b["layers"] if layer["role"] != "substrate"}
        grid_ids = [g["layer"] for g in b["electrical_grid"]]
        if len(grid_ids) != len(set(grid_ids)) or set(grid_ids) != set(grid_names):
            raise _Unsupported("$.electrical_grid", "grid_layer_identity", "The old wire requires exactly one grid row for each electrical layer.")
        if len(set(grid_names.values())) != len(grid_names):
            raise _Unsupported("$.electrical_grid", "ambiguous_grid_names", "Legacy grid mappings require unique electrical display names.")
        wire["electrical_grid"] = {"interval_weights": {grid_names[g["layer"]]: g["interval_weight"] for g in b["electrical_grid"]},
                                   "alphas": {grid_names[g["layer"]]: g["alpha"] for g in b["electrical_grid"]}}
    wire["device"] = _patch(dev, {"tunnelling_channels": a["tunnelling_channels"]} if "tunnelling_channels" in a else {},
                            {"tunnelling_channels": b["tunnelling_channels"]} if "tunnelling_channels" in b else {})
    return wire


def _tandem_wire(after: TandemInput, before: TandemInput, source: SourceDocument,
                 references: Mapping[str, SourceDocument], defaults: DefaultCatalog) -> tuple[dict[str, Any], dict[str, SourceDocument]]:
    a, b = _data(before), _data(after)
    wire = deepcopy(load_yaml_mapping(source.content))
    documents: dict[str, SourceDocument] = {}
    for side in ("top_cell", "bottom_cell"):
        ref = getattr(after, side + "_reference")
        if ref != getattr(before, side + "_reference"):
            raise _Unsupported("$." + side + "_reference", "reference_rebinding", "Changing a subcell reference requires an explicitly rebound tandem source, not a filename-only edit.")
        original = references[ref]
        new = _document(original, _device_wire(getattr(after, side), getattr(before, side), original))
        if ref in documents and documents[ref].content != new.content:
            raise _Unsupported("$." + side, "shared_reference_conflict", "One old filename cannot encode two different edited subcell instances.")
        documents[ref] = new
    wire["tandem"] = _patch(wire["tandem"], {"junction": {"model": a["junction_model"]}, "light_direction": a["light_direction"]},
                             {"junction": {"model": b["junction_model"]}, "light_direction": b["light_direction"]})
    def optical(row: dict[str, Any]) -> dict[str, Any]:
        return {**{k: row[k] for k in ("id", "name", "optical_material", "incoherent")},
                "thickness_nm": _wire_quantity(row["thickness"], "m", "nm")}
    for name in ("junction_stack", "back_reflector"):
        if not _same(a.get(name, _ABSENT), b.get(name, _ABSENT)):
            previous = [optical(x) for x in a[name]] if name == "junction_stack" else optical(a[name]) if a.get(name) else a.get(name)
            current = [optical(x) for x in b[name]] if name == "junction_stack" else optical(b[name]) if b.get(name) else b.get(name)
            wire = _patch(wire, {name: previous} if name in a else {}, {name: current} if name in b else {})
    if not _same(a.get("benchmark", _ABSENT), b.get("benchmark", _ABSENT)):
        fields = {"target_pce_fraction": ("target_pce", "1", "%"), "target_jsc_A_m2": ("target_jsc_ma_cm2", "A/m^2", "mA/cm^2"),
                  "target_voc_V": ("target_voc_v", "V", "V"), "target_ff_fraction": ("target_ff", "1", "1"), "tolerance_fraction": ("tolerance_pct", "1", "%")}
        def benchmark(value: dict[str, Any] | None) -> Any:
            return None if value is None else {"reference": value["reference"], **{old: _wire_quantity(value[k], unit, old_unit) for k, (old, unit, old_unit) in fields.items()}}
        wire = _patch(wire, {"benchmark": benchmark(a["benchmark"])} if "benchmark" in a else {},
                      {"benchmark": benchmark(b["benchmark"])} if "benchmark" in b else {})
    return wire, documents


def _without_source_metadata(data: dict[str, Any]) -> dict[str, Any]:
    if data["schema_version"] == "solarlab.tandem-preparation.v1":
        data["top_cell"].pop("legacy_fields", None)
        data["bottom_cell"].pop("legacy_fields", None)
        data["top_cell"].pop("source_format", None)
        data["bottom_cell"].pop("source_format", None)
    else:
        data.pop("legacy_fields", None)
        data.pop("source_format", None)
    return data


def _semantic(value: DeviceInput | TandemInput) -> dict[str, Any]:
    return _without_source_metadata(value.normalized_data())


def _json_default(value: Any) -> Any:
    if isinstance(value, Decimal):
        return {"decimal": str(value)}
    raise TypeError(type(value).__name__)


@dataclass(frozen=True, slots=True)
class LegacyWire:
    document: SourceDocument | None
    references: tuple[tuple[str, SourceDocument], ...]
    original_sources: tuple[SourceDocument, ...]
    _report_json: bytes

    @property
    def supported(self) -> bool:
        return self.document is not None

    @property
    def can_execute(self) -> bool:
        return False

    def export(self) -> dict[str, Any]:
        """Detached dry-run evidence, never an executor request."""
        return json.loads(self._report_json)


def prepare_legacy_wire(
    input: DeviceInput | TandemInput, *, source: SourceDocument, defaults: DefaultCatalog,
    references: Mapping[str, SourceDocument] | None = None, behavior: BehaviorContext | None = None,
) -> LegacyWire:
    """Prepare edited wire and a dry-run report against supplied source bytes.

    New topologies/material inheritance graphs and source-reference rebinding
    are explicit boundaries of this source-bound transition. No sources are
    discovered, default-filled into the wire, or written by this function.
    """
    if type(source) is not SourceDocument or type(defaults) is not DefaultCatalog:
        raise TypeError("actual immutable source and default catalog are required")
    supplied = dict(references or {})
    if any(not isinstance(name, str) or type(doc) is not SourceDocument for name, doc in supplied.items()):
        raise TypeError("references must map explicit names to immutable SourceDocuments")
    report: dict[str, Any] = {"schema": LEGACY_WIRE_VERSION, "can_execute": False, "source": _binding(source),
        "original_source_retained": True, "reference_sources": {name: _binding(doc) for name, doc in supplied.items()},
        "default_catalog_sha256": defaults.content_sha256, "default_catalog": defaults.to_mapping(),
        "differences": [], "unsupported": [], "behavior": None,
        "roundtrip_scope": "strict imported declaration data, presence and numeric words; no old-driver or physical equivalence",
        "remaining_dependencies": ["P03/G3 old-driver effective payload and quantitative equivalence", "P06/P09 transition exit ownership"]}
    document = None
    emitted_references: dict[str, SourceDocument] = {}
    try:
        if type(input) not in {DeviceInput, TandemInput}:
            raise _Unsupported("$", "input_schema", "Only the existing device and tandem preparation declarations have these legacy wire schemas.")
        checked = type(input).model_validate(input)
        report["input_declaration"] = checked.editing_data()
        if isinstance(checked, TandemInput):
            before = import_tandem(source, id=checked.id, references=supplied, defaults=defaults)
        else:
            if checked.source_format == "canonical":
                matches = []
                for reader in (import_standard_device, import_scaps_device):
                    try:
                        matches.append((reader, reader(source, id=checked.id, defaults=defaults)))
                    except ValueError:
                        pass
                if len(matches) != 1:
                    raise _Unsupported("$.source_format", "source_schema_binding", "A canonical declaration requires exactly one strict standard/SCAPS interpretation of the supplied source.")
                importer, before = matches[0]
            else:
                importer = import_standard_device if checked.source_format == "standard" else import_scaps_device
                before = importer(source, id=checked.id, defaults=defaults)
            report["source_format_mapping"] = {"declared": checked.source_format, "wire": before.source_format,
                "basis": "the existing strict importer accepted the explicitly supplied source"}
        report["differences"] = _differences(_data(before), _data(checked))
        report["original_declaration"] = before.editing_data()
        report["retained_legacy_evidence"] = (_data(before).get("legacy_fields") if isinstance(before, DeviceInput)
            else {side: _data(before).get(side, {}).get("legacy_fields") for side in ("top_cell", "bottom_cell")})
        devices = (checked,) if isinstance(checked, DeviceInput) else (checked.top_cell, checked.bottom_cell)
        report["inheritance"] = {device.id: {
            "settings_from_catalog": {name: value for name, value in defaults.device if name not in device.settings.editing_data()},
            "undeclared_layer_parameters": {layer.id: sorted(set(type(layer.parameters).model_fields) - set(layer.parameters.editing_data())) for layer in device.layers},
            "policy": "absence is retained; SCAPS derived quantities and physical effects are not evaluated here"} for device in devices}
        if isinstance(checked, TandemInput):
            wire, emitted_references = _tandem_wire(checked, before, source, supplied, defaults)
            candidate = _document(source, wire)
            reopened = import_tandem(candidate, id=checked.id, references=emitted_references, defaults=defaults)
        else:
            wire = _device_wire(checked, before, source)
            candidate = _document(source, wire)
            reopened = importer(candidate, id=checked.id, defaults=defaults)
        report["wire_differences"] = _differences(load_yaml_mapping(source.content), wire)
        report["reference_differences"] = {name: _differences(load_yaml_mapping(supplied[name].content), load_yaml_mapping(doc.content))
                                           for name, doc in emitted_references.items()}
        losses = _differences(_semantic(checked), _semantic(reopened))
        if losses:
            report["roundtrip_differences"] = losses[:64]
            report["roundtrip_difference_count"] = len(losses)
            raise _Unsupported(losses[0]["path"], "roundtrip_loss", "The strict importer cannot retain this declaration/presence; no partial wire is published.")
        report["existing_import_normalization"] = _differences(
            _without_source_metadata(_data(checked)), _without_source_metadata(_data(reopened)))
        report["normalization_policy"] = "Wire/editing evidence retains signed zero; the existing SI importer normalizes converted zero to positive zero, reported separately without changing that contract."
        if behavior is not None:
            if type(behavior) is not BehaviorContext or not isinstance(checked, DeviceInput):
                raise _Unsupported("$.behavior", "behavior_type", "An actual matching device BehaviorContext is required.")
            if (behavior.device.defaults.content_sha256 != defaults.content_sha256
                    or source not in behavior.device.sources):
                raise _Unsupported("$.behavior", "behavior_source_binding", "BehaviorContext source/default identities do not match this dry run.")
            historical = _semantic(behavior.device.to_input())
            if not _same(historical, _semantic(before)) and not _same(historical, _semantic(checked)):
                raise _Unsupported("$.behavior", "behavior_input_binding", "BehaviorContext belongs to neither the source nor the edited declaration.")
            report["behavior"] = {"sha256": behavior.content_sha256, "context": behavior.export(),
                "binding": "edited_declaration" if _same(historical, _semantic(checked)) else "original_only_requires_reinspection",
                "selected_models_encoded_as_legacy_controls": False, "global_environment_mutated": False,
                "model_equivalence": "unestablished; versioned multi-instance selections are retained as evidence, never flattened into legacy flags"}
        else:
            report["behavior"] = {"binding": "not_supplied", "historical_driver_equivalence": "unestablished"}
        report["unsupported_execution"] = [{"path": "$.behavior.selection", "code": "selected_model_equivalence_unestablished",
            "message": "Versioned model instances are not represented by a legacy wire document; driver/selection/effective-payload qualification remains P03/G3."}]
        report["emitted"] = _binding(candidate)
        report["emitted_references"] = {name: _binding(doc) for name, doc in emitted_references.items()}
        report["strict_roundtrip_passed"] = True
        report["standard_loader_source_binding"] = STANDARD_LOADER_BINDING
        document = candidate
    except _Unsupported as error:
        report["unsupported"].append(error.issue)
    except ValueError as error:
        report["unsupported"].append({"path": "$", "code": "invalid_or_unrepresentable_declaration", "message": str(error)[:2000]})
    report["status"] = "prepared_data_only" if document else "unsupported"
    report["supported"] = document is not None
    return LegacyWire(document, tuple(sorted(emitted_references.items())) if document else (), (source, *supplied.values()),
                      json.dumps(report, sort_keys=True, allow_nan=False, default=_json_default).encode())
