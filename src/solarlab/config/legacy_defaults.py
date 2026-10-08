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
from solarlab.experiments.jv.inputs import JV_DEFAULT_FIELDS
from solarlab.materials.full_parameters import FullParameterInput
from solarlab.units import normalize_quantity

__all__ = ["read_legacy_default_catalog"]

_SYMBOLS = {"MAXWELL_BOLTZMANN", "FULLY_IONIZED", "BAND_GAP_NARROWING_OFF",
            "EFFECTIVE_LIFETIME", "MINOURA_2015", "WIDTH_NOT_APPLICABLE", "_DEFECT_FREE_TAU", "Q", "K_B", "T",
            "BOTH_CARRIERS", "BOTH_SIDES", "DEFAULT_DENSITY_ATOL_M3"}

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
                              "InterfaceDefectAssistedTunnellingChannel", "ContactTunnellingChannel", "ComponentwiseAtol", "JVRequest"}:
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
        if (function_name, mapping_name) not in {("_layer_from_scaps_row", "row"), ("load_tandem_from_yaml", "tandem"), ("cigs_graded_optics_from_mapping", "raw"), ("start_job", "p")}:
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
        if kind not in {"jv_2d", "voc_grain_sweep", "jv", "dark_jv"}:
            raise ValueError("unknown experiment default branch")
        functions = [node for tree in self._trees() for node in tree.body
                     if isinstance(node, ast.FunctionDef) and node.name == "start_job"]
        if len(functions) != 1:
            raise ValueError("expected one supplied start_job declaration")
        branches = [node for node in ast.walk(functions[0]) if isinstance(node, ast.If)
                    and isinstance(node.test, ast.Compare) and isinstance(node.test.left, ast.Name)
                    and node.test.left.id == "kind" and len(node.test.ops) == 1
                    and isinstance(node.test.ops[0], ast.Eq) and len(node.test.comparators) == 1
                    and isinstance(node.test.comparators[0], ast.Constant) and node.test.comparators[0].value == kind]
        if kind in {"jv", "dark_jv"}:
            branches = [branch for branch in branches if any(isinstance(item, ast.FunctionDef) and item.name == "_run" for item in branch.body)]
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
        if kind == "jv" and "V_max" in names:
            maxima = [keyword.value for call in ast.walk(branch) if isinstance(call, ast.Call)
                      and isinstance(call.func, ast.Name) and call.func.id == "_run_jv_dispatch"
                      for keyword in call.keywords if keyword.arg == "V_max"]
            if len(maxima) != 1 or not isinstance(maxima[0], ast.IfExp):
                raise ValueError("J-V maximum-voltage fallback changed")
            maximum = maxima[0]
            assert isinstance(maximum, ast.IfExp)
            expected = ast.parse('p.get("V_max") is not None', mode="eval").body
            if ast.dump(maximum.test) != ast.dump(expected):
                raise ValueError("J-V maximum-voltage presence rule changed")
            values["V_max"] = self._literal(maximum.orelse)
        if kind == "jv_2d" and "atol" in names:
            assignments = [node for node in ast.walk(branch) if isinstance(node, ast.Assign)
                           and any(isinstance(target, ast.Name) and target.id == "solver_atol" for target in node.targets)
                           and isinstance(node.value, ast.IfExp)]
            if len(assignments) != 1:
                raise ValueError("scalar/componentwise tolerance fallback changed")
            default_policy = assignments[0].value
            assert isinstance(default_policy, ast.IfExp)
            conditional = default_policy.orelse
            if not (isinstance(conditional, ast.IfExp) and isinstance(conditional.test, ast.Name)
                    and conditional.test.id == "extended_topology" and isinstance(conditional.body, ast.Call)
                    and isinstance(conditional.body.func, ast.Name) and conditional.body.func.id == "ComponentwiseAtol"
                    and not conditional.body.args and not conditional.body.keywords):
                raise ValueError("unknown 2D tolerance default policy")
            values["atol"] = self._literal(conditional.orelse)
        if set(values) != set(names):
            raise ValueError(f"missing source experiment defaults: {sorted(set(names) - set(values))}")
        return values

    def function_defaults(self, name: str, fields: tuple[str, ...]) -> dict[str, Any]:
        if name not in {"run_jv_sweep", "run_dark_jv"}:
            raise ValueError("function is outside the J-V source default adapter")
        matches = [node for tree in self._trees() for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name]
        if len(matches) != 1:
            raise ValueError(f"expected one supplied {name} declaration")
        arguments = matches[0].args
        pairs = list(zip(arguments.args[-len(arguments.defaults):], arguments.defaults))
        pairs.extend((arg, value) for arg, value in zip(arguments.kwonlyargs, arguments.kw_defaults) if value is not None)
        result = {arg.arg: self._literal(value) for arg, value in pairs if arg.arg in fields}
        if set(result) != set(fields):
            raise ValueError(f"missing {name} source defaults")
        return result

    def jv_defaults(self) -> dict[str, dict[str, Any]]:
        endpoint = self.fields("JVRequest", JV_DEFAULT_FIELDS["jv_endpoint"])
        job_fields = tuple(name for name in JV_DEFAULT_FIELDS["jv_jobs"] if name != "protocol_mode")
        jobs = {**self.experiment_fallbacks("jv", job_fields),
                **self.mapping_defaults("start_job", "p", ("protocol_mode",))}
        # Only the literal default controls block in the existing waveform
        # parser is read. No legacy modules or function bodies are executed.
        functions = [node for tree in self._trees() for node in tree.body if isinstance(node, ast.FunctionDef)]
        parsers = [node for node in functions if node.name == "_parse_jv_waveform"]
        if len(parsers) != 1:
            raise ValueError("expected one supplied waveform request parser")
        controls = [node.value for node in ast.walk(parsers[0]) if isinstance(node, ast.Assign)
                    and isinstance(node.value, ast.Dict) and any(isinstance(target, ast.Name) and target.id == "controls" for target in node.targets)]
        if len(controls) != 1:
            raise ValueError("ambiguous waveform default controls")
        waveform = {}
        for key, value in zip(controls[0].keys, controls[0].values):
            if not isinstance(key, ast.Constant) or key.value not in {"rtol", "atol_m3"}:
                raise ValueError("unsupported waveform control default")
            if isinstance(value, ast.Attribute):
                if not isinstance(value.value, ast.Name) or value.value.id != "waveform_jv_exp" or value.attr != "DEFAULT_DENSITY_ATOL_M3":
                    raise ValueError("unexpected waveform default source reference")
                waveform[key.value] = self.constant(value.attr)
            else:
                waveform[key.value] = self._literal(value)
        builders = [node for node in functions if node.name == "build_jv_experiment_protocol"]
        if len(builders) != 1:
            raise ValueError("expected one supplied J-V protocol builder")
        lit = [item.value for branch in builders[0].body if isinstance(branch, ast.If)
               and isinstance(branch.test, ast.Name) and branch.test.id == "illuminated"
               for item in branch.body if isinstance(item, ast.Assign)
               and any(isinstance(target, ast.Name) and target.id == "soak" for target in item.targets)]
        if len(lit) != 1:
            raise ValueError("missing illuminated J-V soak declaration")
        return {"jv_endpoint": endpoint, "jv_jobs": jobs,
                "dark_jv": self.experiment_fallbacks("dark_jv", JV_DEFAULT_FIELDS["dark_jv"]),
                "jv_numerical": self.function_defaults("run_jv_sweep", JV_DEFAULT_FIELDS["jv_numerical"]),
                "dark_jv_numerical": self.function_defaults("run_dark_jv", JV_DEFAULT_FIELDS["dark_jv_numerical"]),
                "jv_waveform_controls": waveform, "jv_history": {"illuminated_soak_s": self._literal(lit[0])}}


def read_legacy_default_catalog(sources: tuple[SourceDocument, ...], *, model_sources: tuple[SourceDocument, ...] = (),
                                experiment_sources: tuple[SourceDocument, ...] = (),
                                jv_sources: tuple[SourceDocument, ...] = ()) -> DefaultCatalog:
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
    if jv_sources:
        experiment_defaults.update(_LiteralReader(jv_sources).jv_defaults())
    evidence: dict[str, str] = {}
    for source in (*sources, *model_sources, *experiment_sources, *jv_sources):
        if source.id in evidence and evidence[source.id] != source.sha256:
            raise ValueError("conflicting source bytes for the same default identity")
        evidence[source.id] = source.sha256
    return DefaultCatalog(tuple(material.items()), tuple(device.items()), tuple(scaps.items()), tuple(interface.items()),
                          tuple((name, reader.constant(name)) for name in ("Q", "K_B", "T")),
                          tuple(contacts.items()), tuple(structural.items()),
                          defect["defect_model"], defect["defect_schema_version"], degeneracy, width,
                          tuple(evidence.items()),
                          tuple((name, tuple(values.items())) for name, values in complex_defaults.items()),
                          tuple((name, tuple(values.items())) for name, values in experiment_defaults.items()))
