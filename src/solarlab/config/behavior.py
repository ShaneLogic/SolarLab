"""Source-bound inspection of three historical drivers, without importing them.

This is preparation metadata, not a replacement physics schema or an executor.
Source-derived branch decisions and actually captured numerical preparations
occupy separate fields. Neither a source digest nor a capture certifies a solve.
Only the audited source revision below and the two reviewed SCAPS captures are
understood; other revisions need a new inspection rule, not a permissive flag.
"""

from __future__ import annotations

import ast
import base64
from dataclasses import dataclass, field
import hashlib
import json
import math
import re
from typing import Any, Literal

from solarlab.config.legacy_defaults import read_legacy_default_catalog
from solarlab.config.model_parameters import device_topology
from solarlab.config.resolve_device import resolve_device
from solarlab.config.scaps_input import import_scaps_device
from solarlab.device.resolved import PreparedDevice
from solarlab.materials.source import SourceDocument
from solarlab.physics.selection import Selection

__all__ = ["BehaviorContext", "CapturedPreparation", "behavior_source_requirements",
           "inspect_historical_behavior"]

Driver = Literal["scaps_transient", "scaps_steady_state", "ion_hysteresis"]
_SCHEMA = "solarlab.historical-behavior-preparation.v1"
# Source identities of the inspected branches, not a table of physical defaults.
_PINS = {
    "models/mode.py": "7464cf8619401997206fed5b23cded0a361ab6064f6b06a2449fb9f13a700688",
    "models/device.py": "abedb9e39655a5b54997968a7bfa4d1664b7493475dc8e7c55fffa68195330b9",
    "models/parameters.py": "0555a6c9280431889b34328bf1e513c7e6d6789368ea858ff3e1ea5a60db7d9d",
    "models/defects.py": "0bf3cedd435ff8259fb436bbcfcfa3aa586bdee50f89caba161e162c6913159f",
    "models/tandem_config.py": "9aaf4e1fb87f43950c96cabe0ac309cd3416673848b80889fc366be32ad2b595",
    "physics/statistics.py": "f790b5df608ec54ff64ebd1a3203bcec0a00d1bfbc2677a29ccd34208fc29a01",
    "physics/band_gap_narrowing.py": "ee7fe97002c19364750f4aa07be1dc150c2acb450ad74ee637d5b8cc783e0a78",
    "constants.py": "9ae428b44bbe5f475b638d9c3339e69930249e2b1c39ba0acbfd0764fcaade76",
    "twod/microstructure.py": "60d06d32976305a4c29b04e7f46188e8a916e007048a9c6507c680e6d4b6ab35",
    "solver/mol.py": "1c02ff41356cfaf75245e3e9e3a0101064d43a766e9e416053ea1e75ef54b61a",
    "solver/newton.py": "db4c2096108f7c6820dc900ea14bb03cd780d29212ad0fdedd290db84c95d3c2",
    "solver/illuminated_ss.py": "489d3d68dc7e2a6877bc10d91152cf8504f6b878c2a32b19a622f9cf4c1aede7",
    "experiments/jv_sweep.py": "428fbe1c421192e2495ac94fa171df42f2be94dca7733b7a5ab2e07a9628ef96",
    "experiments/steady_state.py": "55580e14e075de0596e2a8e6e55e0a16f2c9581be380f2bb3bacbae2b5d1edbd",
    "experiments/waveform_jv.py": "cac7648c136190db6bca65697cdb3200ab7b57e81787736a67d706cda333d1e7",
    "physics/contacts.py": "a8bc9a2938bd979ea9a5265af8785480b554ef723d4968b0ffdced8b8231709b",
    "physics/thermionic_transport.py": "996e4f1d46a10740a28f133a549ef7a2e190e72c0986b7a420a7c8f3fcfdca3d",
    "physics/ion_migration.py": "19c0b83485c5e1b61dce5ad8fc05b75202a40c383d964cd4bcf38889a5cbda35",
    "physics/field_mobility.py": "62ea0880e8aa05cf90dc0928f6bae7aec2caa132d69338661bff856a4d441737",
    "physics/photon_recycling.py": "cda651b7fdd7c202ed308d3a0e063c1cedc4ea76d7b0ef61db65e9ffcdc3d9f9",
    "physics/optical_stack.py": "ef897393af20ce1a3a096e6db2d3a9b3482ca590ad71a294057124ceb6746543",
    "scaps_compat/loader.py": "a03a31b81497df6118592aebce7000befaa1d5de787ad989d880c263000e0cd6",
}
_DRIVERS = {
    "scaps_transient": ("experiments/jv_sweep.py", "run_jv_sweep"),
    "scaps_steady_state": ("experiments/steady_state.py", "solve_voc_ss"),
    "ion_hysteresis": ("experiments/waveform_jv.py", "run_waveform_jv"),
}
# Setting / environment aliases from build_material_arrays. Only the first
# four are suppressed by LEGACY; the interface switches are not mode ceilings.
_SWITCHES = {
    "band_grading": "SOLARLAB_BAND_GRADING",
    "dos_band_potentials": "SOLARLAB_DOS_BAND",
    "te_physical_norm": "SOLARLAB_TE_PHYSICAL",
    "ion_steric_diffusion_only": "SOLARLAB_ION_STERIC_DIFF",
    "interface_plane_projection": "SOLARLAB_IFACE_PROJ",
    "interface_two_sided": "SOLARLAB_IFACE_TWOSIDED",
    "interface_shared_occupancy": "SOLARLAB_IFACE_SHARED_OCC",
    "interface_plane_closure": "SOLARLAB_IFACE_PLANE",
    "interface_plane_generation": "SOLARLAB_IFACE_PLANE_GEN",
    "interface_tunneling": "SOLARLAB_IFACE_TUNNEL",
}
_ENV_NAMES = frozenset(_SWITCHES.values()) | {
    "SOLARLAB_INTERFACE_PLANE_STATE", "SOLARLAB_SS_JAC_REUSE",
    "SOLARLAB_QSS_VTH", "SOLARLAB_IFACE_QSS", "SOLARLAB_IFACE_ALLOW_GEN",
    "PEROVSKITE_RHS_FINITE_CHECK",
}
_CAPTURES = {
    "solarlab.scaps-driver-preparation.v1": (
        "1568beb08b8ee4cd0d8cb4ddf809efef41203bbc02c9c189228cd3142e73612e",
        "2e15784a745c9ff85e31519b88ac14e1253616bda07e79a41fadbcd4ceb21772"),
    "solarlab.scaps-conditioning-capture.v1": (
        "75c87559093307475857d71f4800bb89cf021fc180cf704df5da3765dd4ed51d",
        "55aa71d7915e522160ec0eef076bbe98986094ce6756502870575118a373b635"),
}
_CASE_SHA = "7204498a03f0257722256216ab1cfd55457beb85b324dbbe1a81337b1a82e7d5"
_CATALOG_SOURCES = ("models/parameters.py", "models/device.py", "models/defects.py", "physics/statistics.py",
                    "physics/band_gap_narrowing.py", "scaps_compat/loader.py", "constants.py",
                    "twod/microstructure.py", "models/tandem_config.py")


def behavior_source_requirements() -> tuple[tuple[str, str], ...]:
    """Explicit relative legacy-source IDs and audited byte digests to supply."""
    return tuple(sorted(_PINS.items()))


def _bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _mapping(source: SourceDocument) -> dict[str, Any]:
    if type(source) is not SourceDocument:
        raise ValueError("an immutable SourceDocument is required")
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result = dict(pairs)
        if len(result) != len(pairs):
            raise ValueError("duplicate JSON member")
        return result
    value = json.loads(source.content, object_pairs_hook=unique)
    if not isinstance(value, dict):
        raise ValueError("a JSON object is required")
    _bytes(value)  # Reject nonfinite JSON numbers, without converting scalar types.
    return value


def _binding(source: SourceDocument) -> dict[str, str]:
    return {"id": source.id, "sha256": source.sha256}


def _tree(sources: dict[str, SourceDocument], path: str) -> ast.Module:
    return ast.parse(sources[path].content, filename=path)


def _literal(tree: ast.Module, name: str, owner: str | None = None) -> Any:
    body = tree.body if owner is None else next(
        item.body for item in tree.body if isinstance(item, ast.ClassDef) and item.name == owner)
    for node in body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == name and node.value is not None:
            return ast.literal_eval(node.value)
    raise ValueError(f"missing audited source literal: {owner}.{name}")


def _signature(tree: ast.Module, name: str) -> dict[str, Any]:
    fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    pairs = list(zip(fn.args.args[-len(fn.args.defaults):], fn.args.defaults)) if fn.args.defaults else []
    pairs.extend((arg, default) for arg, default in zip(fn.args.kwonlyargs, fn.args.kw_defaults) if default is not None)
    result = {}
    for arg, default in pairs:
        try:
            result[arg.arg] = ast.literal_eval(default)
        except ValueError:
            # Waveform atol is symbolic. Its actual supplied control is required.
            if not (name == "run_waveform_jv" and arg.arg == "atol"):
                raise ValueError(f"unsupported source default: {name}.{arg.arg}") from None
    return result


def _mode(sources: dict[str, SourceDocument], mode: str) -> dict[str, Any]:
    for node in _tree(sources, "models/mode.py").body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == mode.upper() for t in node.targets):
            if isinstance(node.value, ast.Call):
                return {str(kw.arg): ast.literal_eval(kw.value) for kw in node.value.keywords}
    raise ValueError("unsupported historical mode")


def _environment(entries: tuple[tuple[str, str | None], ...]) -> tuple[tuple[str, str | None], ...]:
    if not isinstance(entries, tuple) or any(type(p) is not tuple or len(p) != 2 for p in entries):
        raise ValueError("environment requires an explicit immutable snapshot of pairs")
    if len({p[0] for p in entries}) != len(entries):
        raise ValueError("duplicate environment key")
    if any(name not in _ENV_NAMES or value is not None and type(value) is not str for name, value in entries):
        raise ValueError("unsupported environment key or non-string value")
    return tuple(sorted(entries))


def _verify_words(value: Any, *, schema: str, path: tuple[str, ...] = ()) -> None:
    """Validate capture words in place; never turn an array into decimal text."""
    if isinstance(value, dict):
        if set(value) == {"binary64_hex"}:
            # The sole reviewed nonfinite scalar is the step-limit sentinel;
            # permit it only at that exact schema/path, never in physical data.
            word = value["binary64_hex"]
            if type(word) is not str or float.fromhex(word).hex() != word:
                raise ValueError("invalid captured scalar word")
            sentinel = (schema == "solarlab.scaps-conditioning-capture.v1"
                        and path == ("boundary_call", "effective_arguments", "max_step") and word == "inf")
            if not math.isfinite(float.fromhex(word)) and not sentinel:
                raise ValueError("nonfinite scalar outside the reviewed max_step sentinel")
        if "bytes_b64" in value:
            if set(value) != {"dtype", "shape", "bytes_b64", "sha256"}:
                raise ValueError("unsupported captured array encoding")
            dtype = re.fullmatch(r"[<>=|][fiub](1|2|4|8)", value["dtype"])
            if dtype is None or any(type(n) is not int or n < 0 for n in value["shape"]):
                raise ValueError("unsupported captured dtype or shape")
            raw = base64.b64decode(value["bytes_b64"], validate=True)
            if len(raw) != math.prod(value["shape"]) * int(dtype.group(1)) or hashlib.sha256(raw).hexdigest() != value["sha256"]:
                raise ValueError("captured array words do not match shape/digest")
        for key, item in value.items():
            _verify_words(item, schema=schema, path=(*path, key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _verify_words(item, schema=schema, path=(*path, str(index)))


def _scalars(value: Any) -> Any:
    if isinstance(value, dict):
        if set(value) == {"binary64_hex"}:
            return float.fromhex(value["binary64_hex"])
        return {key: _scalars(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_scalars(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class CapturedPreparation:
    """The independently reviewed scaps_base outer or conditioning snapshot.

    Inputs are explicit bytes; no archive path is discovered or loaded. Pins
    are from the independent capture reviews. This intentionally rejects new
    captures until their schema, source and input relationships are reviewed.
    """

    result: SourceDocument
    freeze: SourceDocument
    case: SourceDocument

    def __post_init__(self) -> None:
        result = _mapping(self.result)
        if result.get("schema") not in _CAPTURES:
            raise ValueError("unsupported captured preparation")
        if (self.result.sha256, self.freeze.sha256) != _CAPTURES[result["schema"]] or self.case.sha256 != _CASE_SHA:
            raise ValueError("capture or case differs from independently reviewed bytes")
        frozen = _mapping(self.freeze)["source_sha256"]
        if any(frozen.get("perovskite-sim/perovskite_sim/" + path) != digest for path, digest in _PINS.items()):
            raise ValueError("captured source revision is outside the inspected domain")
        _verify_words(result, schema=result["schema"])

    def inspect(self, driver: Driver, device: PreparedDevice, arguments: dict[str, Any]) -> dict[str, Any]:
        if driver == "ion_hysteresis":
            raise ValueError("SCAPS evidence cannot stand in for an ion trajectory")
        expected = resolve_device(import_scaps_device(self.case, id=device.to_input().id, defaults=device.defaults),
                                  device.defaults, device.resources, sources=(self.case,))
        if device.input_json != expected.input_json:
            raise ValueError("captured input does not match the prepared device")
        result, frozen = _mapping(self.result), _mapping(self.freeze)
        allowed_resources = set(frozen["source_sha256"].values())
        if any(row["source_sha256"] not in allowed_resources for row in device.values["resource_bindings"]):
            raise ValueError("captured resource contents do not match prepared inputs")
        if result["schema"] == "solarlab.scaps-driver-preparation.v1":
            name = {"scaps_transient": "historical_transient_full_scan", "scaps_steady_state": "historical_interface_state_Voc"}[driver]
            record, = [item for item in result["records"] if item["driver"] == name]
            bound = {k: v for k, v in record["effective_bound_signature"].items() if k != "stack"}
            payload = {"record": record, "scope": result["scope"], "head": result["head"],
                       "electronic_solver_calls_executed": result["electronic_solver_calls_executed"]}
        else:
            if driver != "scaps_transient":
                raise ValueError("conditioning capture belongs only to the transient driver")
            bound = {k: v for k, v in result["driver_call"]["effective_arguments"].items() if k != "stack"}
            payload = {key: result[key] for key in ("objects", "driver_call", "precondition_call", "seed_call",
                       "boundary_call", "boundary_identity_checks", "seed_layout", "seed_has_independent_phi_block",
                       "scope", "stop_reason", "head", "comparison")}
        if _bytes(arguments) != _bytes(_scalars(bound)):
            raise ValueError("captured driver arguments do not match source inspection")
        return {"result": _binding(self.result), "freeze": _binding(self.freeze), "input": _binding(self.case),
                "historical_source_identity": frozen["source_identity"],
                "recorded_environment": frozen.get("environment_observations"),
                "unrecorded_environment": "unknown; supplied inspection snapshot is not historical environment proof",
                "payload": payload,
                "normalization_gap": "New exact-unit/parser values are not substituted for historical material words; glass incoherence and sub-ulp unit differences remain explicit."}


def _arguments(driver: Driver, source: SourceDocument, sources: dict[str, SourceDocument]) -> tuple[dict[str, Any], dict[str, Any]]:
    supplied = _mapping(source)
    path, function = _DRIVERS[driver]
    defaults = _signature(_tree(sources, path), function)
    if driver == "ion_hysteresis":
        # These are the original saved frontend request.params, not a new RunSpec.
        required = {"waveform", "waveform_controls", "N_grid", "n_points", "v_rate", "V_max", "illuminated"}
        allowed = set(defaults) | required | {"solver", "iface_states", "interface_boundary", "interface_transport_model"}
        if required - supplied.keys() or supplied.keys() - allowed:
            raise ValueError("ion inspection requires the explicit saved waveform and controls")
        for key, expected in (("solver", "transient"), ("iface_states", False), ("interface_boundary", False)):
            if key in supplied and (type(supplied[key]) is not type(expected) or supplied[key] != expected):
                raise ValueError("saved request selects a different historical driver")
        wf, controls = supplied["waveform"], supplied["waveform_controls"]
        wf_keys = {"schema_version", "start_voltage_V", "dark_seed_s", "dark_prep_s", "branch_dwell_s",
                   "turnaround_s", "turnaround_dark", "uniform_generation_rate_m3_s"}
        if not isinstance(wf, dict) or set(wf) != wf_keys or type(wf["schema_version"]) is not int or wf["schema_version"] != 1:
            raise ValueError("unsupported waveform declaration")
        if not isinstance(controls, dict) or set(controls) != {"rtol", "atol_m3"}:
            raise ValueError("explicit density controls are required")
        if type(wf["turnaround_dark"]) is not bool:
            raise ValueError("turnaround_dark must be boolean")
        for key in wf_keys - {"schema_version", "turnaround_dark"}:
            val = wf[key]
            if val is None and key == "uniform_generation_rate_m3_s":
                continue
            if type(val) not in {int, float} or not math.isfinite(val) or key != "start_voltage_V" and val < 0:
                raise ValueError("invalid waveform quantity")
        defaults.update(rtol=controls["rtol"], atol=controls["atol_m3"])
    elif supplied.keys() - defaults.keys():
        raise ValueError("unknown driver argument")
    effective = {**defaults, **supplied}
    for key in ("N_grid", "n_points", "v_max_max_attempts"):
        if key in effective and (type(effective[key]) is not int or effective[key] < 1):
            raise ValueError(f"{key} requires a positive integer")
    for key in ("rtol", "atol", "v_rate", "tol_v"):
        if key in effective and (type(effective[key]) not in {int, float} or not math.isfinite(effective[key]) or effective[key] <= 0):
            raise ValueError(f"{key} requires a positive finite number")
    for key in ("V_max", "V_lo", "V_hi"):
        if key in effective and effective[key] is not None and (type(effective[key]) not in {int, float} or not math.isfinite(effective[key])):
            raise ValueError(f"{key} requires a finite voltage or its declared sentinel")
    for key, value in defaults.items():
        if type(value) is bool and type(effective[key]) is not bool:
            raise ValueError(f"{key} requires boolean, not zero/null")
    for key in ("rhs_regularization", "progress", "fixed_generation", "experiment_protocol"):
        if effective.get(key) is not None:
            raise ValueError(f"supplied {key} is outside this inspection domain")
    return supplied, effective


@dataclass(frozen=True, slots=True)
class BehaviorContext:
    """Independent immutable context built from actual prepared public inputs.

    The environment argument is a complete snapshot for this scoped inspection:
    unlisted reviewed keys are unset. It never reads or changes process globals.
    Raw sources/capture words are exact; normalized configuration follows the
    existing configuration contract, including its normalized positive zero.
    Selection is bound, not claimed equivalent to the historical mechanisms.
    """

    driver: Driver
    device: PreparedDevice
    selection: Selection
    sources: tuple[SourceDocument, ...]
    environment: tuple[tuple[str, str | None], ...]
    arguments: SourceDocument
    capture: CapturedPreparation | None = None
    _document: bytes = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.driver not in _DRIVERS or type(self.device) is not PreparedDevice or type(self.selection) is not Selection:
            raise ValueError("an inspected driver, actual PreparedDevice and Selection are required")
        if self.selection.topology != device_topology(self.device):
            raise ValueError("selection topology does not match prepared device")
        supplied_sources = tuple(self.sources)
        if any(type(source) is not SourceDocument for source in supplied_sources):
            raise ValueError("behavior sources must be immutable documents")
        sources = {source.id: source for source in supplied_sources}
        if len(sources) != len(supplied_sources) or set(sources) != set(_PINS) or any(sources[p].sha256 != h for p, h in _PINS.items()):
            raise ValueError("exact audited behavior sources are required")
        environment = _environment(self.environment)
        env = dict(environment)
        supplied, arguments = _arguments(self.driver, self.arguments, sources)
        device = self.device.to_mapping()
        settings = device["settings"]
        if device["grain_boundaries"] or device.get("tunnelling_channels") or settings["interface_charge_closure"] != "off":
            raise ValueError("complex/charged devices require a separate historical inspection")
        if any(layer.cigs_graded_optics_json is not None or layer.metastable_document_json is not None
               or layer.defect_model != "effective_lifetime" for layer in self.device.layers):
            raise ValueError("extended material models require a separate historical inspection")
        if any(dict(layer.parameters)["carrier_statistics"] != "maxwell_boltzmann" for layer in self.device.layers):
            raise ValueError("carrier statistics are outside the audited driver subset")
        ceilings = _mode(sources, settings["mode"])
        flags = {name: bool(settings[name] or env.get(key) == "1") for name, key in _SWITCHES.items()}
        for name in ("band_grading", "dos_band_potentials", "te_physical_norm", "ion_steric_diffusion_only"):
            flags[name] = flags[name] and settings["mode"] != "legacy"
        flags["interface_plane_generation"] &= flags["interface_plane_closure"]
        flags["interface_tunneling"] &= ceilings["use_thermionic_emission"] and settings["mode"] != "legacy"
        mol = _tree(sources, "solver/mol.py")
        steady = _tree(sources, "experiments/steady_state.py")
        softness = (_literal(steady, "_TE_SOFTNESS") if self.driver == "scaps_steady_state"
                    else _literal(mol, "te_softness", "MaterialArrays"))
        contact_tree = _tree(sources, "physics/contacts.py")
        contact_fn = next(n for n in contact_tree.body if isinstance(n, ast.FunctionDef) and n.name == "resolved_contact_velocities")
        flat_velocity = next(ast.literal_eval(n.body) for n in ast.walk(contact_fn) if isinstance(n, ast.IfExp))
        contacts = []
        for contact in device["contacts"]:
            for carrier in ("n", "p"):
                original = contact["S_" + carrier]
                effective = flat_velocity if settings["flat_band_contacts"] and original is None else original
                if not settings["flat_band_contacts"] and not ceilings["use_selective_contacts"]:
                    effective = None
                contacts.append({"side": contact["side"], "carrier": carrier, "configured_S_m_s": original,
                                 "effective_S_m_s": effective, "condition": "density_dirichlet" if effective is None else
                                 "blocking_robin" if effective == 0 else "finite_exchange_robin"})
        state_requested = env.get("SOLARLAB_INTERFACE_PLANE_STATE") == "1" or arguments.get("iface_states", False)
        interfaces = []
        for interface in device["interfaces"]:
            if interface["electrical"]:
                defect = interface.get("defect")
                if defect is not None:
                    base, state = defect["calibration_factor"], defect["iface_state_calibration_factor"]
                    interfaces.append({"id": interface["id"], "base_calibration": base, "state_calibration": state,
                                       "calibration_if_plane_exists": base * state if self.driver == "scaps_steady_state" and arguments["iface_states"] else base,
                                       "normalized_capture_velocities_m_s": interface["microscopic_capture_velocities_m_s"]})
        ions, incoherence = [], []
        raw_layers = json.loads(self.device.input_json)["layers"]
        legacy_incoherent = _literal(_tree(sources, "models/parameters.py"), "incoherent", "MaterialParams")
        for raw, layer in zip(raw_layers, self.device.layers):
            params = dict(layer.parameters)
            incoherence.append({"layer": layer.id, "normalized_configured": params["incoherent"],
                                "historical_loader_value": legacy_incoherent if raw.get("parameterization") == "scaps" else params["incoherent"]})
            if layer.role == "substrate":
                continue
            for species, suffix in (("positive", ""), ("negative", "_neg")):
                enabled = species == "positive" or ceilings["use_dual_ions"]
                diffusion, population = params["D_ion" + suffix], params["P0" + suffix]
                ions.append({"layer": layer.id, "species": species, "D_reference_m2_s": diffusion,
                             "configured_initial_population_m3": population, "enabled_by_mode": enabled,
                             "seed_population_and_fixed_background_m3": population if enabled else 0.0,
                             "diffusion_status": "disabled_by_mode" if not enabled else "frozen_zero_diffusion" if diffusion == 0 else "temperature_scaled_not_evaluated",
                             "inventory_status": "zero_population_and_background" if not enabled or population == 0 else "population_and_equal_opposite_background_retained"})
        optical_data = bool(device["resource_bindings"])
        tmm_eligible = ceilings["use_tmm_optics"] and optical_data
        waveform = arguments.get("waveform")
        generation_override = waveform["uniform_generation_rate_m3_s"] if waveform is not None else None
        preconditioning = _signature(_tree(sources, "solver/illuminated_ss.py"), "solve_illuminated_ss")
        preparation = {"method": "steady_carrier_newton_with_bounded_transient_fallback" if self.driver == "scaps_steady_state" else
                       "finite_time_waveform_from_analytic_seed" if self.driver == "ion_hysteresis" else
                       "finite_illuminated_preconditioning" if arguments["illuminated"] else "analytic_seed_only",
                       "solver_executed": False, "converged_state_observed": False,
                       "seed": "quasi_neutral_n_p_clipped_then_ohmic_overrides_and_StateVec_pack; no Poisson/Newton state solve",
                       "actual_driver_arguments": arguments}
        if self.driver == "scaps_transient" and arguments["illuminated"]:
            preparation["preconditioner"] = {**preconditioning, "V_app": 0.0, "rtol": arguments["rtol"], "atol": arguments["atol"]}
            preparation["material_ownership"] = "outer, independent inner preconditioner, independent seed material; outer not forwarded"
        inspection = {
            "mode_ceiling": ceilings, "switches": flags,
            "switch_scope": "resolved requests; face/DOS/parameter activation remains conditional, including interface_tunneling's nonempty capped-face gate",
            "interface_state_requested": state_requested,
            "interface_state_count": {"status": "unobserved", "reason": "requires actual interface partitions; a gate is not a state count"},
            "interface_state_shared_occupancy_if_enabled": self.driver == "scaps_steady_state" and arguments.get("iface_states", False),
            "interfaces": interfaces, "contacts": contacts, "ions": ions,
            "ion_flux_label": "diffusion_only" if flags["ion_steric_diffusion_only"] else "whole_flux",
            "te": {"softness": softness, "cap": "hard_direction_preserving_magnitude_min" if softness == 0 else "legacy_logistic_magnitude_blend",
                   "requested_physical_normalization": flags["te_physical_norm"],
                   "normalization": "unobserved_face_activation_and_DOS; disabled / legacy_density_weighted / physical_dos_normalized / legacy_fallback_missing_dos remain distinct"},
            "optics": {"generation": "uniform_absorber_override" if generation_override is not None else "TMM_eligible_unobserved" if tmm_eligible else "Beer_Lambert",
                       "uniform_generation_m3_s": generation_override, "incoherence": incoherence,
                       "resources": device["resource_bindings"],
                       "recycling": "disabled_without_TMM_data" if not tmm_eligible else
                       "uniform_absorber_nonlocal_RR_eligible_unobserved" if ceilings["use_radiative_reabsorption"] else "escape_scaled_B_eligible_unobserved"},
            "electrostatics": {"configured_V_bi": settings["V_bi"], "configured_potential_mode": settings["built_in_potential_mode"],
                               "signed_V_bi_eff": {"status": "unobserved"}, "signed_V_bi_bc": {"status": "unobserved"},
                               "rule": "compatibility None retains distinct band-derived operating value and signed manual-magnitude BC; explicit modes share the selected value; BC drop is V_bi_bc - junction_polarity*V_app"},
            "import_time_policy": {"ss_jacobian_reuse": env.get("SOLARLAB_SS_JAC_REUSE", "1") != "0",
                                   "rhs_finite_check": env.get("PEROVSKITE_RHS_FINITE_CHECK") == "1",
                                   "QSS_velocity_override_text": env.get("SOLARLAB_QSS_VTH"),
                                   "binding": "supplied snapshot must precede any future isolated old import"},
            "runtime_interface_requests": {"qss": env.get("SOLARLAB_IFACE_QSS") == "1",
                                           "allow_generation": env.get("SOLARLAB_IFACE_ALLOW_GEN") == "1",
                                           "scope": "conditional on actual interface/phi; no RHS is evaluated"},
            "preparation": preparation,
        }
        captured = None
        if self.capture is not None:
            if type(self.capture) is not CapturedPreparation:
                raise ValueError("reviewed captured preparation required")
            catalog = read_legacy_default_catalog(tuple(sources[path] for path in _CATALOG_SOURCES))
            if self.device.defaults.content_sha256 != catalog.content_sha256:
                raise ValueError("captured default catalog does not match prepared inputs")
            captured = self.capture.inspect(self.driver, self.device, arguments)
            payload = captured["payload"]
            if "objects" in payload:
                material = payload["objects"]["inner_material"]["value"]["fields"]
            else:
                record = payload["record"]
                material = record["boundary_calls"][0]["kwargs"].get("mat", record["preparation"][0]["material"])["fields"]
            gate_fields = {"te_physical_norm": "te_physical_norm", "ion_steric_diffusion_only": "ion_steric_diffusion_only",
                           "interface_plane_projection": "iface_plane_projection", "interface_two_sided": "iface_two_sided",
                           "interface_shared_occupancy": "iface_shared_occ", "interface_plane_closure": "iface_plane_closure",
                           "interface_plane_generation": "iface_plane_generation"}
            if any(flags[name] != material[key] for name, key in gate_fields.items()) or state_requested != (material["N_iface_state"] > 0):
                raise ValueError("captured effective gates conflict with the supplied inspection snapshot")
        document = {"schema": _SCHEMA, "version": 1, "status": "prepared_pending_dependencies", "can_execute": False,
                    "driver": self.driver, "driver_source": {"path": _DRIVERS[self.driver][0], "function": _DRIVERS[self.driver][1]},
                    "prepared_device_sha256": self.device.content_sha256, "normalized_supplied": json.loads(self.device.input_json),
                    "input_sources": device["input_sources"], "default_catalog": device["default_catalog"],
                    "layer_parameter_origins": [{"id": row["id"], "provenance": row["provenance"]} for row in device["layers"]],
                    "selection": self.selection.export(), "selection_sha256": self.selection.content_sha256,
                    "sources": [_binding(sources[path]) for path in sorted(sources)],
                    "environment_supplied": dict(environment), "environment_effective_snapshot": {key: env.get(key) for key in sorted(_ENV_NAMES)},
                    "arguments_source": _binding(self.arguments), "arguments_supplied": supplied,
                    "source_inspection": inspection, "captured_preparation": captured,
                    "normalization_policy": "existing normalized configuration; original SourceDocument bytes and captured binary64/array words retained exactly",
                    "unresolved": ["historical_behavior_to_selected_model_equivalence", "complete_P03_04_migration_and_G3",
                                   "uncaptured_material_arrays_and_optical_algebra", "signed_contact_potential_without_capture",
                                   "general_driver_and_import_environment_replay", "quantitative_old_new_comparison",
                                   *device["capability_gaps"]]}
        if self.driver == "ion_hysteresis":
            document["unresolved"].append("original_ion_run_effective_flags_not_recoverable_from_saved_request_alone")
        object.__setattr__(self, "sources", tuple(sources[path] for path in sorted(sources)))
        object.__setattr__(self, "environment", environment)
        object.__setattr__(self, "_document", _bytes(document))

    @property
    def content_sha256(self) -> str:
        return hashlib.sha256(self._document).hexdigest()

    def export(self) -> dict[str, Any]:
        """A detached JSON inspection; mutations cannot affect another context."""
        return json.loads(self._document)


def inspect_historical_behavior(
    device: PreparedDevice, selection: Selection, *, driver: Driver,
    sources: tuple[SourceDocument, ...], environment: tuple[tuple[str, str | None], ...],
    arguments: SourceDocument, capture: CapturedPreparation | None = None,
) -> BehaviorContext:
    return BehaviorContext(driver, device, selection, sources, environment, arguments, capture)
