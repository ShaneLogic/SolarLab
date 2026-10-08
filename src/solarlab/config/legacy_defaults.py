"""Explicit preparation adapter; normal resolution never imports this module."""

from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import json
from typing import Any

from solarlab.materials.source import SourceDocument
from solarlab.device.defaults import DefaultCatalog
from solarlab.device.settings import DeviceSettingsInput
from solarlab.device.tunnelling import CHANNEL_TYPES
from solarlab.experiments.two_dimensional.inputs import EXPERIMENT_DEFAULT_FIELDS
from solarlab.materials.full_parameters import FullParameterInput
from solarlab.units import normalize_quantity

__all__ = ["read_legacy_default_catalog"]

_SYMBOLS = {"MAXWELL_BOLTZMANN", "FULLY_IONIZED", "BAND_GAP_NARROWING_OFF",
            "EFFECTIVE_LIFETIME", "MINOURA_2015", "WIDTH_NOT_APPLICABLE", "_DEFECT_FREE_TAU", "Q", "K_B", "T",
            "BOTH_CARRIERS", "BOTH_SIDES"}

@dataclass(frozen=True, slots=True)
class _LiteralReader:
    """Default declarations read from the actual supplied legacy sources."""

    sources: tuple[SourceDocument, ...]

    def __post_init__(self) -> None:
        sources = tuple(self.sources)
        if not sources or any(not isinstance(item, SourceDocument) for item in sources):
            raise ValueError("defaults require supplied source documents")
        if len({item.id for item in sources}) != len(sources):
            raise ValueError("duplicate default source ID")
        object.__setattr__(self, "sources", tuple(sorted(sources, key=lambda item: item.id)))

    def _trees(self) -> tuple[ast.Module, ...]:
        return tuple(ast.parse(source.content.decode("utf-8"), filename=source.id) for source in self.sources)

    def _literal(self, node: ast.AST, active: tuple[str, ...] = ()) -> Any:
        if isinstance(node, ast.Constant) and type(node.value) in {str, int, float, bool, type(None)}:
            return node.value
        if isinstance(node, (ast.Tuple, ast.List)):
            return tuple(self._literal(value, active) for value in node.elts)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            value = self._literal(node.operand, active)
            if type(value) not in {int, float}:
                raise ValueError("default sign requires a numeric literal")
            return -value if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.Name):
            if node.id not in _SYMBOLS:
                raise ValueError(f"unsupported symbolic default: {node.id}")
            if node.id in active:
                raise ValueError("cyclic source default reference")
            values = []
            for tree in self._trees():
                for item in tree.body:
                    if isinstance(item, ast.Assign) and any(isinstance(target, ast.Name) and target.id == node.id for target in item.targets):
                        values.append(self._literal(item.value, (*active, node.id)))
                    elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name) and item.target.id == node.id and item.value is not None:
                        values.append(self._literal(item.value, (*active, node.id)))
            if values and all(value == values[0] for value in values):
                return values[0]
            raise ValueError(f"missing or ambiguous literal source default: {node.id}")
        raise ValueError(f"unsupported source default expression: {ast.dump(node, include_attributes=False)}")

    def constant(self, name: str) -> Any:
        return self._literal(ast.Name(id=name))

    def fields(self, class_name: str, names: tuple[str, ...]) -> dict[str, Any]:
        if class_name not in {"MaterialParams", "DeviceStack", "InterfaceDefect", "BulkDefectSpecies", "BulkDefectDistribution", "GrainBoundary", "JunctionLayer",
                              "CIGSGradedOptics", "BandToBandTunnellingChannel", "IntrabandTunnellingChannel",
                              "InterfaceDefectAssistedTunnellingChannel", "ContactTunnellingChannel", "ComponentwiseAtol"}:
            raise ValueError("class is outside the bounded legacy default adapter")
        matches = [node for tree in self._trees() for node in tree.body
                   if isinstance(node, ast.ClassDef) and node.name == class_name]
        if len(matches) != 1:
            raise ValueError(f"expected one supplied {class_name} declaration")
        result: dict[str, Any] = {}
        declared = set()
        for node in matches[0].body:
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                declared.add(node.target.id)
                if node.target.id in names and node.value is not None:
                    result[node.target.id] = self._literal(node.value)
        if set(names) - declared:
            raise ValueError(f"unknown fields in {class_name}: {sorted(set(names) - declared)}")
        return result

    def mapping_defaults(self, function_name: str, mapping_name: str, names: tuple[str, ...]) -> dict[str, Any]:
        """Read only named mapping.get(key, literal) fallback declarations."""
        if (function_name, mapping_name) not in {("_layer_from_scaps_row", "row"), ("load_tandem_from_yaml", "tandem"), ("cigs_graded_optics_from_mapping", "raw")}:
            raise ValueError("function is outside the bounded SCAPS default adapter")
        functions = [node for tree in self._trees() for node in tree.body
                     if isinstance(node, ast.FunctionDef) and node.name == function_name]
        if len(functions) != 1:
            raise ValueError(f"expected one supplied {function_name} declaration")
        result: dict[str, Any] = {}
        for node in ast.walk(functions[0]):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name) and node.func.value.id == mapping_name
                    and node.func.attr == "get" and len(node.args) == 2
                    and isinstance(node.args[0], ast.Constant) and node.args[0].value in names):
                name = node.args[0].value
                value = self._literal(node.args[1])
                if name in result and result[name] != value:
                    raise ValueError(f"conflicting source defaults for {name}")
                result[name] = value
        if set(result) != set(names):
            raise ValueError("required adapter defaults were not found in supplied source")
        return result

    @property
    def content_sha256(self) -> str:
        return hashlib.sha256(json.dumps([(item.id, item.sha256) for item in self.sources], separators=(",", ":")).encode()).hexdigest()

    def experiment_fallbacks(self, kind: str, names: tuple[str, ...]) -> dict[str, Any]:
        """Only the actual start_job branch; never import or execute backend."""
        if kind not in {"jv_2d", "voc_grain_sweep"}:
            raise ValueError("unknown spatial experiment default branch")
        functions = [node for tree in self._trees() for node in tree.body
                     if isinstance(node, ast.FunctionDef) and node.name == "start_job"]
        if len(functions) != 1:
            raise ValueError("expected one supplied start_job declaration")
        branches = [node for node in ast.walk(functions[0]) if isinstance(node, ast.If)
                    and isinstance(node.test, ast.Compare) and isinstance(node.test.left, ast.Name)
                    and node.test.left.id == "kind" and len(node.test.ops) == 1
                    and isinstance(node.test.ops[0], ast.Eq) and len(node.test.comparators) == 1
                    and isinstance(node.test.comparators[0], ast.Constant) and node.test.comparators[0].value == kind]
        if len(branches) != 1:
            raise ValueError(f"expected one supplied {kind} backend branch")
        branch = ast.Module(body=branches[0].body, type_ignores=[])
        values: dict[str, Any] = {}
        for node in ast.walk(branch):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name) and node.func.value.id == "p"
                    and node.func.attr == "get" and len(node.args) == 2
                    and isinstance(node.args[0], ast.Constant) and node.args[0].value in names):
                name, value = node.args[0].value, self._literal(node.args[1])
                if name in values and values[name] != value:
                    raise ValueError(f"ambiguous backend experiment default: {name}")
                values[name] = value
        if kind == "jv_2d" and "atol" in names:
            assignments = [node for node in ast.walk(branch) if isinstance(node, ast.Assign)
                           and any(isinstance(target, ast.Name) and target.id == "solver_atol" for target in node.targets)
                           and isinstance(node.value, ast.IfExp)]
            if len(assignments) != 1:
                raise ValueError("scalar/componentwise tolerance fallback changed")
            conditional = assignments[0].value.orelse
            if not (isinstance(conditional, ast.IfExp) and isinstance(conditional.test, ast.Name)
                    and conditional.test.id == "extended_topology" and isinstance(conditional.body, ast.Call)
                    and isinstance(conditional.body.func, ast.Name) and conditional.body.func.id == "ComponentwiseAtol"
                    and not conditional.body.args and not conditional.body.keywords):
                raise ValueError("unknown 2D tolerance default policy")
            values["atol"] = self._literal(conditional.orelse)
        if set(values) != set(names):
            raise ValueError(f"missing source experiment defaults: {sorted(set(names) - set(values))}")
        return values


def read_legacy_default_catalog(sources: tuple[SourceDocument, ...], *, model_sources: tuple[SourceDocument, ...] = (),
                                experiment_sources: tuple[SourceDocument, ...] = ()) -> DefaultCatalog:
    """Compile data once; the resulting catalog needs no legacy source files."""
    reader = _LiteralReader((*sources, *model_sources))
    material = reader.fields("MaterialParams", tuple(FullParameterInput.model_fields))
    device = reader.fields("DeviceStack", tuple(DeviceSettingsInput.model_fields))
    interface = reader.fields("InterfaceDefect", ("calibration_factor", "iface_state_calibration_factor"))
    contacts = reader.fields("DeviceStack", ("S_n_left", "S_p_left", "S_n_right", "S_p_right"))
    scaps_fields = {
        "D_ion_m2_s": ("D_ion", "m^2/s", "m^2/s"),
        "P_lim_m3": ("P_lim", "m^-3", "m^-3"), "P0_m3": ("P0", "m^-3", "m^-3"),
        "B_rad_cm3_s": ("B_rad", "m^3/s", "cm^3/s"),
        "C_n_cm6_s": ("C_n", "m^6/s", "cm^6/s"), "C_p_cm6_s": ("C_p", "m^6/s", "cm^6/s"),
        "alpha_cm": ("alpha", "m^-1", "cm^-1"),
    }
    raw = reader.mapping_defaults("_layer_from_scaps_row", "row", tuple(scaps_fields))
    scaps = {name: normalize_quantity(raw[key], unit, input_unit=source_unit)
             for key, (name, unit, source_unit) in scaps_fields.items()}
    scaps["tau_n"] = scaps["tau_p"] = reader.constant("_DEFECT_FREE_TAU")
    defect = reader.fields("MaterialParams", ("defect_schema_version", "defect_model"))
    degeneracy = reader.fields("BulkDefectSpecies", ("degeneracy",))["degeneracy"]
    width = reader.fields("BulkDefectDistribution", ("width_convention",))["width_convention"]
    structural = {
        "grain_boundary_layer_role": reader.fields("GrainBoundary", ("layer_role",))["layer_role"],
        "tandem_light_direction": reader.mapping_defaults("load_tandem_from_yaml", "tandem", ("light_direction",))["light_direction"],
        "junction_incoherent": reader.fields("JunctionLayer", ("incoherent",))["incoherent"],
    }
    complex_defaults = {}
    if model_sources:
        model_reader = _LiteralReader(model_sources)
        cigs = model_reader.fields("CIGSGradedOptics", ("model", "slices", "kk_quadrature_order"))
        if cigs != model_reader.mapping_defaults("cigs_graded_optics_from_mapping", "raw", tuple(cigs)):
            raise ValueError("CIGS data and loader defaults disagree")
        complex_defaults["cigs_graded_optics"] = cigs
        classes = {"band_to_band": "BandToBandTunnellingChannel", "intraband": "IntrabandTunnellingChannel",
                   "interface_defect_assisted": "InterfaceDefectAssistedTunnellingChannel", "contact": "ContactTunnellingChannel"}
        for name, model in CHANNEL_TYPES.items():
            complex_defaults[name] = model_reader.fields(classes[name], tuple(model.model_fields))
    experiment_defaults = {}
    if experiment_sources:
        experiment_reader = _LiteralReader(experiment_sources)
        for name in ("jv_2d", "voc_grain_sweep"):
            experiment_defaults[name] = experiment_reader.experiment_fallbacks(name, EXPERIMENT_DEFAULT_FIELDS[name])
        experiment_defaults["jv_2d_componentwise_atol"] = experiment_reader.fields(
            "ComponentwiseAtol", EXPERIMENT_DEFAULT_FIELDS["jv_2d_componentwise_atol"])
    return DefaultCatalog(tuple(material.items()), tuple(device.items()), tuple(scaps.items()), tuple(interface.items()),
                          tuple((name, reader.constant(name)) for name in ("Q", "K_B", "T")),
                          tuple(contacts.items()), tuple(structural.items()),
                          defect["defect_model"], defect["defect_schema_version"], degeneracy, width,
                          tuple((source.id, source.sha256) for source in (*sources, *model_sources, *experiment_sources)),
                          tuple((name, tuple(values.items())) for name, values in complex_defaults.items()),
                          tuple((name, tuple(values.items())) for name, values in experiment_defaults.items()))
