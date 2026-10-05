"""SCAPS-unit preparation import; derived quantities are recomputed on resolve."""

from __future__ import annotations

from typing import Any

from solarlab.config.device_import import _common_device, _flat_defect, _keys, _mapping, _quantity, _sequence
from solarlab.config.yaml import load_yaml_mapping
from solarlab.device.defaults import DefaultCatalog
from solarlab.device.inputs import DeviceInput
from solarlab.materials.source import SourceDocument

__all__ = ["import_scaps_device"]

# Schema unit mappings, not default physical parameter values.
_FIELDS = {
    "E_g_eV": ("Eg", "eV", "eV"), "chi_eV": ("chi", "eV", "eV"),
    "eps_r": ("eps_r", "1", "1"),
    "mu_n_cm2": ("mu_n", "m^2/(V s)", "cm^2/(V s)"), "mu_p_cm2": ("mu_p", "m^2/(V s)", "cm^2/(V s)"),
    "N_C_cm3": ("Nc300", "m^-3", "cm^-3"), "N_V_cm3": ("Nv300", "m^-3", "cm^-3"),
    "N_D_cm3": ("N_D", "m^-3", "cm^-3"), "N_A_cm3": ("N_A", "m^-3", "cm^-3"),
    "v_th_cm_s": ("v_th", "m/s", "cm/s"),
    "D_ion_m2_s": ("D_ion", "m^2/s", "m^2/s"), "P_lim_m3": ("P_lim", "m^-3", "m^-3"), "P0_m3": ("P0", "m^-3", "m^-3"),
    "B_rad_cm3_s": ("B_rad", "m^3/s", "cm^3/s"), "C_n_cm6_s": ("C_n", "m^6/s", "cm^6/s"),
    "C_p_cm6_s": ("C_p", "m^6/s", "cm^6/s"), "alpha_cm": ("alpha", "m^-1", "cm^-1"),
}
_REQUIRED = {"name", "role", "thickness_nm", "E_g_eV", "chi_eV", "eps_r", "mu_n_cm2", "mu_p_cm2",
             "N_C_cm3", "N_V_cm3", "N_D_cm3", "N_A_cm3", "v_th_cm_s"}
_ROLE_ALIAS = {"pvk": "absorber", "perovskite": "absorber", "absorber": "absorber", "htl": "htl", "etl": "etl"}


def _bulk_defect(raw: object, id: str, v_th: float, defaults: DefaultCatalog, path: str) -> tuple[dict[str, Any], dict[str, Any]]:
    value = _mapping(raw, path)
    required = {"sigma_n_cm2", "sigma_p_cm2", "N_t_cm3"}
    _keys(value, required | {"name", "E_t_eV_below_cb", "E_t_eV_above_vb", "distribution", "E_char_eV", "N_peak_cm3",
                              "charge_transition", "neutral_reference", "degeneracy"}, required, path)
    energy = [name for name in ("E_t_eV_below_cb", "E_t_eV_above_vb") if name in value]
    if len(energy) != 1:
        raise ValueError(f"{path}: exactly one referenced defect energy is required")
    reference = "below_conduction_band" if energy[0].endswith("below_cb") else "above_valence_band"
    depth = _quantity(value[energy[0]], "eV", path + "." + energy[0])
    distribution = value.get("distribution", "single")
    if distribution not in {"single", "gaussian"}:
        raise ValueError(f"{path}.distribution: unsupported SCAPS declaration")
    if ("charge_transition" in value) != ("neutral_reference" in value):
        raise ValueError(f"{path}: charge transition and neutral reference must be declared together")
    distribution_data = {"kind": "single_level" if distribution == "single" else "gaussian", "normalization": "integrated_total",
                         "total_density_m3": _quantity(value["N_t_cm3"], "m^-3", path + ".N_t_cm3", "cm^-3")}
    metadata = {"defect_id": id, "distribution": distribution, "energy_reference": reference, "trap_depth_eV": depth}
    if distribution == "gaussian":
        distribution_data["width_convention"] = "scaps_characteristic_energy" if "E_char_eV" in value else "unresolved"
        if "E_char_eV" in value:
            distribution_data["width_eV"] = _quantity(value["E_char_eV"], "eV", path + ".E_char_eV")
    if "E_char_eV" in value:
        metadata["E_char_eV"] = _quantity(value["E_char_eV"], "eV", path + ".E_char_eV")
    if "N_peak_cm3" in value:
        metadata["N_peak_m3"] = _quantity(value["N_peak_cm3"], "m^-3", path + ".N_peak_cm3", "cm^-3")
    species = {"id": id, "name": value.get("name"), "distribution": distribution_data,
               "energy_level": {"reference": reference, "value_eV": depth},
               "charge_transition": value.get("charge_transition", "unresolved"),
               "neutral_reference": value.get("neutral_reference", "unresolved"),
               "kinetics": {"sigma_n_m2": _quantity(value["sigma_n_cm2"], "m^2", path + ".sigma_n_cm2", "cm^2"),
                            "sigma_p_m2": _quantity(value["sigma_p_cm2"], "m^2", path + ".sigma_p_cm2", "cm^2"),
                            "thermal_velocity_n_m_s": v_th, "thermal_velocity_p_m_s": v_th},
               "degeneracy": value.get("degeneracy", defaults.defect_degeneracy)}
    return species, metadata


def import_scaps_device(source: SourceDocument, *, id: str, defaults: DefaultCatalog) -> DeviceInput:
    document = load_yaml_mapping(source.content)
    _keys(document, {"schema_version", "name", "description", "device", "layers", "interfaces", "electrical_grid", "simulation_hints", "microstructure"}, {"device", "layers"}, "document")
    layers = []
    for i, raw in enumerate(_sequence(document["layers"], "layers")):
        row = _mapping(raw, f"layers[{i}]")
        _keys(row, set(_FIELDS) | {"id", "name", "role", "thickness_nm", "bulk_defect", "bulk_defects", "defect_schema_version", "defect_model",
                                  "optical_material", "incoherent", "n_optical", "interface_defect"}, _REQUIRED, f"layers[{i}]")
        params: dict[str, Any] = {name: _quantity(row[key], unit, f"layers[{i}].{key}", source_unit)
                  for key, (name, unit, source_unit) in _FIELDS.items() if key in row}
        for key in ("optical_material", "incoherent", "n_optical"):
            if key in row:
                params[key] = row[key]
        layer = {"id": row.get("id", f"layer_{i}"), "name": row["name"], "role": row["role"], "parameterization": "scaps",
                 "thickness": _quantity(row["thickness_nm"], "m", f"layers[{i}].thickness_nm", "nm"), "parameters": params}
        if "bulk_defect" in row and "bulk_defects" in row:
            raise ValueError(f"layers[{i}]: singular and plural bulk defect declarations conflict")
        defects = (row["bulk_defect"],) if "bulk_defect" in row and row["bulk_defect"] is not None else _sequence(row.get("bulk_defects", ()), f"layers[{i}].bulk_defects")
        canonical, metadata = [], []
        for j, value in enumerate(defects):
            defect, meta = _bulk_defect(value, f"bulk_{j}", params["v_th"], defaults, f"layers[{i}].bulk_defects[{j}]")
            canonical.append(defect)
            metadata.append(meta)
        if canonical or "defect_model" in row or "defect_schema_version" in row:
            layer.update(defect_schema_version=row.get("defect_schema_version", "solarlab-explicit-bulk-defects-v1"),
                         defect_model=row.get("defect_model", defaults.defect_model),
                         bulk_defects=canonical, scaps_defect_metadata=metadata)
        if "interface_defect" in row and row["interface_defect"] is not None:
            profile = _mapping(row["interface_defect"], f"layers[{i}].interface_defect")
            _keys(profile, {"N_t_peak_cm3", "decay_length_nm", "profile", "target"}, {"N_t_peak_cm3", "decay_length_nm"}, f"layers[{i}].interface_defect")
            params["trap_N_t_interface"] = _quantity(profile["N_t_peak_cm3"], "m^-3", f"layers[{i}].interface_defect.N_t_peak_cm3", "cm^-3")
            params["trap_decay_length"] = _quantity(profile["decay_length_nm"], "m", f"layers[{i}].interface_defect.decay_length_nm", "nm")
            if "profile" in profile:
                params["trap_profile_shape"] = profile["profile"]
            if "target" in profile:
                targets = {"htl/pvk": "left", "pvk/etl": "right", "left": "left", "right": "right", "both": "both"}
                if not isinstance(profile["target"], str) or profile["target"].lower() not in targets:
                    raise ValueError(f"layers[{i}].interface_defect.target: unknown edge")
                params["trap_edge"] = targets[profile["target"].lower()]
        layers.append(layer)
    interfaces = [{"id": f"interface_{i}", "left": left["id"], "right": right["id"]}
                  for i, (left, right) in enumerate(zip(layers, layers[1:]))]
    targets_used = set()
    for i, raw in enumerate(_sequence(document.get("interfaces", ()), "interfaces")):
        entry = _mapping(raw, f"interfaces[{i}]")
        if not isinstance(entry.get("target"), str):
            raise ValueError(f"interfaces[{i}].target: an explicit directed target is required")
        aliases = [part.strip().lower() for part in entry["target"].split("/")]
        if len(aliases) != 2 or not all(aliases):
            raise ValueError(f"interfaces[{i}].target: expected LEFT/RIGHT")
        aliases = [_ROLE_ALIAS.get(part, part) for part in aliases]
        matching = [j for j, (left, right) in enumerate(zip(layers, layers[1:]))
                    if [left["role"].lower(), right["role"].lower()] == aliases]
        if len(matching) != 1 or matching[0] in targets_used:
            raise ValueError(f"interfaces[{i}].target: unknown, ambiguous or duplicate directed interface")
        target = matching[0]
        targets_used.add(target)
        interfaces[target]["defect"] = _flat_defect(entry, f"interface_defect_{target}", defaults, f"interfaces[{i}]")
    return _common_device(document, layers, id, defaults, "scaps", interfaces)
