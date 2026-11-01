"""Tandem preparation from explicitly supplied subcell documents."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from solarlab.config.device_import import _keys, _mapping, _quantity, _sequence, import_standard_device
from solarlab.config.yaml import load_yaml_mapping
from solarlab.device.defaults import DefaultCatalog
from solarlab.device.inputs import TandemInput
from solarlab.materials.source import SourceDocument

__all__ = ["import_tandem"]


def _optical_layer(raw: object, id: str, defaults: DefaultCatalog, path: str) -> dict[str, Any]:
    row = _mapping(raw, path)
    _keys(row, {"id", "name", "thickness_nm", "optical_material", "incoherent"}, {"name", "thickness_nm", "optical_material"}, path)
    return {"id": row.get("id", id), "name": row["name"],
            "thickness": _quantity(row["thickness_nm"], "m", path + ".thickness_nm", "nm"),
            "optical_material": row["optical_material"],
            "incoherent": row["incoherent"] if "incoherent" in row else dict(defaults.structural)["junction_incoherent"]}


def import_tandem(source: SourceDocument, *, id: str, references: Mapping[str, SourceDocument], defaults: DefaultCatalog) -> TandemInput:
    raw = load_yaml_mapping(source.content)
    _keys(raw, {"schema_version", "device_type", "tandem", "junction_stack", "back_reflector", "benchmark"},
          {"schema_version", "device_type", "tandem"}, "document")
    if type(raw["schema_version"]) is not int or raw["schema_version"] != 1:
        raise ValueError("schema_version: only tandem input version 1 is available")
    tandem = _mapping(raw["tandem"], "tandem")
    _keys(tandem, {"top_cell", "bottom_cell", "junction", "light_direction"}, {"top_cell", "bottom_cell", "junction"}, "tandem")
    junction = _mapping(tandem["junction"], "tandem.junction")
    _keys(junction, {"model"}, {"model"}, "tandem.junction")
    cells = {}
    for name in ("top_cell", "bottom_cell"):
        ref = tandem[name]
        if not isinstance(ref, str) or ref not in references or not isinstance(references[ref], SourceDocument):
            raise ValueError(f"tandem.{name}: missing explicitly supplied subcell document {ref!r}")
        cells[name] = import_standard_device(references[ref], id=id + "." + name, defaults=defaults).editing_data()
    result = {"schema_version": "solarlab.tandem-preparation.v1", "id": id,
              "source_schema_version": raw["schema_version"], "device_type": raw["device_type"],
              **cells, "top_cell_reference": tandem["top_cell"], "bottom_cell_reference": tandem["bottom_cell"],
              "junction_model": junction["model"], "light_direction": tandem.get("light_direction", dict(defaults.structural)["tandem_light_direction"]),
              "junction_stack": [_optical_layer(item, f"junction_{i}", defaults, f"junction_stack[{i}]")
                                 for i, item in enumerate(_sequence(raw.get("junction_stack", ()), "junction_stack"))]}
    if "back_reflector" in raw:
        result["back_reflector"] = None if raw["back_reflector"] is None else _optical_layer(raw["back_reflector"], "back_reflector", defaults, "back_reflector")
    if "benchmark" in raw and raw["benchmark"] is None:
        result["benchmark"] = None
    elif "benchmark" in raw:
        benchmark = _mapping(raw["benchmark"], "benchmark")
        fields = {"reference", "target_pce", "target_jsc_ma_cm2", "target_voc_v", "target_ff", "tolerance_pct"}
        _keys(benchmark, fields, fields, "benchmark")
        result["benchmark"] = {"reference": benchmark["reference"],
            "target_pce_fraction": _quantity(benchmark["target_pce"], "1", "benchmark.target_pce", "%"),
            "target_jsc_A_m2": _quantity(benchmark["target_jsc_ma_cm2"], "A/m^2", "benchmark.target_jsc_ma_cm2", "mA/cm^2"),
            "target_voc_V": _quantity(benchmark["target_voc_v"], "V", "benchmark.target_voc_v"),
            "target_ff_fraction": _quantity(benchmark["target_ff"], "1", "benchmark.target_ff"),
            "tolerance_fraction": _quantity(benchmark["tolerance_pct"], "1", "benchmark.tolerance_pct", "%")}
    return TandemInput.model_validate(result)
