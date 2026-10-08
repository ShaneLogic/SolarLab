"""Explicit standard-input preparation; no file discovery or active migration."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from solarlab.config.legacy_fields import retained_legacy_fields
from solarlab.config.yaml import load_yaml_mapping
from solarlab.device.defaults import DefaultCatalog
from solarlab.device.defects import MULTIVALENT_VERSION
from solarlab.device.inputs import DeviceInput
from solarlab.device.settings import DeviceSettingsInput
from solarlab.materials.full_parameters import FullParameterInput
from solarlab.materials.source import SourceDocument
from solarlab.units import normalize_quantity

__all__ = ["import_standard_device"]


def _mapping(value: object, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{path}: expected a string-keyed mapping")
    return dict(value)


def _sequence(value: object, path: str) -> tuple[Any, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{path}: expected an explicit sequence")
    return tuple(value)


def _keys(value: dict[str, Any], allowed: set[str], required: set[str], path: str) -> None:
    unknown, missing = set(value) - allowed, required - set(value)
    if unknown or missing:
        raise ValueError(f"{path}: unknown={sorted(unknown)}, missing={sorted(missing)}")


def _quantity(value: object, unit: str, path: str, input_unit: str | None = None) -> float:
    return normalize_quantity(value, unit, path=path, input_unit=input_unit)


def _flat_defect(raw: object, id: str, defaults: DefaultCatalog, path: str) -> dict[str, Any]:
    value = _mapping(raw, path)
    required = {"sigma_n_cm2", "sigma_p_cm2", "v_th_cm_s", "N_t_cm2"}
    allowed = required | {"E_t_eV_below_cb", "E_t_eV_above_vb", "calibration_factor", "iface_state_calibration_factor",
                          "distribution", "E_char_eV", "N_peak_cm3", "target"}
    _keys(value, allowed, required, path)
    energy = [name for name in ("E_t_eV_below_cb", "E_t_eV_above_vb") if name in value]
    if len(energy) != 1:
        raise ValueError(f"{path}: exactly one trap energy reference is required")
    reference = "below_conduction_band" if energy[0].endswith("below_cb") else "above_valence_band"
    depth = _quantity(value[energy[0]], "eV", path + "." + energy[0])
    velocity = _quantity(value["v_th_cm_s"], "m/s", path + ".v_th_cm_s", "cm/s")
    result: dict[str, Any] = {
        "id": id, "trap_depth_eV": depth, "energy_reference": reference,
        "total_density_m2": _quantity(value["N_t_cm2"], "m^-2", path + ".N_t_cm2", "cm^-2"),
        "kinetics": {"sigma_n_m2": _quantity(value["sigma_n_cm2"], "m^2", path + ".sigma_n_cm2", "cm^2"),
                     "sigma_p_m2": _quantity(value["sigma_p_cm2"], "m^2", path + ".sigma_p_cm2", "cm^2"),
                     "thermal_velocity_n_m_s": velocity, "thermal_velocity_p_m_s": velocity},
        **{name: value[name] if name in value else default for name, default in defaults.interface},
    }
    if any(name in value for name in ("distribution", "E_char_eV", "N_peak_cm3")):
        result["partner_metadata"] = {"defect_id": id, "distribution": value.get("distribution", "single"),
                                      "energy_reference": reference, "trap_depth_eV": depth}
        for old, new, unit, source_unit in (("E_char_eV", "E_char_eV", "eV", "eV"), ("N_peak_cm3", "N_peak_m3", "m^-3", "cm^-3")):
            if old in value:
                result["partner_metadata"][new] = _quantity(value[old], unit, path + "." + old, source_unit)
    return result


def _common_device(
    document: dict[str, Any], layers: list[dict[str, Any]], id: str,
    defaults: DefaultCatalog, source_format: str, interfaces: list[dict[str, Any]] | None = None,
) -> DeviceInput:
    raw = _mapping(document["device"], "device")
    contacts_extra = {"contacts", "S_n_left", "S_p_left", "S_n_right", "S_p_right"}
    retained = {"temperature"} if source_format == "standard" else set()
    _keys(raw, set(DeviceSettingsInput.model_fields) | contacts_extra | retained | {"interfaces", "interface_defects", "V_bi_override", "tunnelling_channels"}, set(), "device")
    settings = {name: raw[name] for name in DeviceSettingsInput.model_fields if name in raw}
    if "V_bi_override" in raw:
        override = _quantity(raw["V_bi_override"], "V", "device.V_bi_override")
        if "V_bi" in raw and override != _quantity(raw["V_bi"], "V", "device.V_bi"):
            raise ValueError("device.V_bi and V_bi_override disagree")
        if settings.get("built_in_potential_mode", "legacy_manual") != "legacy_manual":
            raise ValueError("V_bi_override requires legacy_manual mode")
        settings["V_bi"] = override
        settings["built_in_potential_mode"] = "legacy_manual"
    if "V_bi" in raw and settings.get("built_in_potential_mode") in {"metal_work_function", "semiconductor_work_function"}:
        raise ValueError("legacy V_bi is incompatible with an explicit physical potential source")
    # Absence with no manual alias retains the existing loader's declared
    # physical-source intent, without calculating a work-function potential.
    if not any(name in raw for name in ("V_bi", "V_bi_override", "built_in_potential_mode")):
        settings["built_in_potential_mode"] = "semiconductor_work_function"
    ids = [layer["id"] for layer in layers]
    if not layers or len(set(ids)) != len(ids):
        raise ValueError("layers: require nonempty unique stable IDs")
    electrical = [layer for layer in layers if layer["role"] != "substrate"]
    if not electrical:
        raise ValueError("layers: at least one electrical layer is required")
    nested = _mapping(raw.get("contacts", {}), "device.contacts")
    _keys(nested, {"left", "right"}, set(), "device.contacts")
    contacts = []
    for side, layer in (("left", electrical[0]), ("right", electrical[-1])):
        values = _mapping(nested.get(side, {}), f"device.contacts.{side}")
        _keys(values, {"S_n", "S_p"}, set(), f"device.contacts.{side}")
        contact = {"id": "contact_" + side, "side": side, "layer": layer["id"]}
        for carrier in ("n", "p"):
            flat, short = f"S_{carrier}_{side}", f"S_{carrier}"
            if flat in raw and short in values:
                a = None if raw[flat] is None else _quantity(raw[flat], "m/s", "device." + flat)
                b = None if values[short] is None else _quantity(values[short], "m/s", f"device.contacts.{side}.{short}")
                if a != b:
                    raise ValueError(f"device.{flat}: conflicting flat/nested contact aliases")
            if flat in raw or short in values:
                contact[short] = raw[flat] if flat in raw else values[short]
        contacts.append(contact)
    if interfaces is not None and ("interfaces" in raw or "interface_defects" in raw):
        raise ValueError("cannot mix top-level SCAPS interfaces with device interface declarations")
    if interfaces is None:
        pairs = _sequence(raw["interfaces"], "device.interfaces") if raw.get("interfaces") is not None else ()
        defects = _sequence(raw["interface_defects"], "device.interface_defects") if raw.get("interface_defects") is not None else ()
        for name, aligned in (("interfaces", pairs), ("interface_defects", defects)):
            if len(aligned) > len(layers) - 1:
                raise ValueError(f"device.{name}: expected at most one full-layer-aligned entry per adjacent interface; unused trailing data is unsupported")
        interfaces = []
        for i, (left, right) in enumerate(zip(layers, layers[1:])):
            interface = {"id": f"interface_{i}", "left": left["id"], "right": right["id"]}
            if i < len(pairs):
                pair = _sequence(pairs[i], f"device.interfaces[{i}]")
                if len(pair) != 2:
                    raise ValueError(f"device.interfaces[{i}]: expected [v_n,v_p]")
                interface.update(v_n=pair[0], v_p=pair[1])
            elif pairs or defects:
                # interfaces_from_device_dict pads by the FULL raw layer
                # index, including substrate boundaries. Never shift slots.
                interface.update(v_n=0.0, v_p=0.0)
            if i < len(defects):
                interface["defect"] = None if defects[i] is None else _flat_defect(defects[i], f"interface_defect_{i}", defaults, f"device.interface_defects[{i}]")
            interfaces.append(interface)
    grains = []
    if "microstructure" in document:
        block = _mapping(document["microstructure"], "microstructure")
        _keys(block, {"grain_boundaries"}, set(), "microstructure")
        for i, value in enumerate(_sequence(block.get("grain_boundaries", ()), "microstructure.grain_boundaries")):
            row = _mapping(value, f"microstructure.grain_boundaries[{i}]")
            _keys(row, {"id", "x_position", "width", "tau_n", "tau_p", "layer_role"}, {"x_position", "width", "tau_n", "tau_p"}, f"microstructure.grain_boundaries[{i}]")
            role = row["layer_role"] if "layer_role" in row else dict(defaults.structural)["grain_boundary_layer_role"]
            matching = [layer["id"] for layer in electrical if layer["role"] == role]
            if not matching:
                raise ValueError(f"microstructure.grain_boundaries[{i}].layer_role: no matching electrical layer")
            grains.append({"id": row.get("id", f"gb_{i}"), "layer_ids": matching,
                           **{name: row[name] for name in ("x_position", "width", "tau_n", "tau_p")},
                           "source_layer_role": role})
    grid = []
    if "electrical_grid" in document:
        block = _mapping(document["electrical_grid"], "electrical_grid")
        _keys(block, {"interval_weights", "alphas"}, {"interval_weights", "alphas"}, "electrical_grid")
        names = [layer["name"] for layer in electrical]
        if len(set(names)) != len(names):
            raise ValueError("electrical_grid: legacy display-name references are ambiguous; use stable canonical IDs")
        for field in ("interval_weights", "alphas"):
            _keys(_mapping(block[field], "electrical_grid." + field), set(names), set(names), "electrical_grid." + field)
        grid = [{"layer": layer["id"], "interval_weight": block["interval_weights"][layer["name"]], "alpha": block["alphas"][layer["name"]]} for layer in electrical]
    data = {"schema_version": "solarlab.device-preparation.v1", "id": id, "source_format": source_format,
            "settings": settings, "layers": layers, "interfaces": interfaces, "contacts": contacts,
            "grain_boundaries": grains, "electrical_grid": grid}
    if "tunnelling_channels" in raw:
        data["tunnelling_channels"] = raw["tunnelling_channels"]
    for name in ("name", "description", "simulation_hints"):
        if name in document:
            data[name] = document[name]
    if "schema_version" in document:
        if type(document["schema_version"]) is bool:
            raise ValueError("schema_version cannot be boolean")
        data["source_schema_version"] = document["schema_version"]
    return DeviceInput.model_validate(data)


def import_standard_device(source: SourceDocument, *, id: str, defaults: DefaultCatalog) -> DeviceInput:
    document = load_yaml_mapping(source.content)
    _keys(document, {"schema_version", "name", "description", "device", "layers", "microstructure", "electrical_grid", "simulation_hints"}, {"device", "layers"}, "document")
    layers = []
    for i, value in enumerate(_sequence(document["layers"], "layers")):
        row = _mapping(value, f"layers[{i}]")
        _keys(row, set(FullParameterInput.model_fields) | {"id", "name", "role", "thickness", "defect_schema_version", "defect_model", "bulk_defects", "bulk_trap_distribution", "cigs_graded_optics"},
              {"name", "role", "thickness"}, f"layers[{i}]")
        layer = {"id": row.get("id", f"layer_{i}"), "name": row["name"], "role": row["role"], "thickness": row["thickness"],
                 "parameters": {name: row[name] for name in FullParameterInput.model_fields if name in row}}
        for field in ("bulk_trap_distribution", "cigs_graded_optics"):
            if field in row:
                # Legacy data loaders require mappings here. The canonical
                # editable layer separately permits explicit nullable clearing.
                layer[field] = _mapping(row[field], f"layers[{i}].{field}")
        defect_fields = {"defect_schema_version", "defect_model", "bulk_defects"} & set(row)
        if defect_fields and defect_fields != {"defect_schema_version", "defect_model", "bulk_defects"}:
            raise ValueError(f"layers[{i}]: defect schema/model/inventory must be declared atomically")
        if defect_fields:
            layer.update(defect_schema_version=row["defect_schema_version"], defect_model=row["defect_model"])
            defects = []
            for j, value in enumerate(_sequence(row["bulk_defects"], f"layers[{i}].bulk_defects")):
                defect = _mapping(value, f"layers[{i}].bulk_defects[{j}]")
                implicit = {} if row["defect_schema_version"] == MULTIVALENT_VERSION else {"degeneracy": defaults.defect_degeneracy}
                defects.append({"id": f"bulk_{j}", **implicit, **defect})
            if row["defect_schema_version"] == MULTIVALENT_VERSION:
                names = [value.get("name") for value in defects]
                if len(names) != len(set(names)):
                    raise ValueError(f"layers[{i}].bulk_defects: duplicate legacy species name")
            layer["bulk_defects"] = defects
        layers.append(layer)
    result = _common_device(document, layers, id, defaults, "standard")
    retained = retained_legacy_fields(source)
    if retained is not None:
        result = result.validated_update({"legacy_fields": retained})
    return result
