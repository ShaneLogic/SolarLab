"""Immutable invocations of the existing historical drivers.

These records retain original input bytes and complete numerical arguments.
The existing old reader/driver remains responsible for physical validation;
this module neither imports that engine nor grants scientific qualification.
"""

from __future__ import annotations

import ast
import base64
from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Any

from solarlab.config.behavior import (
    BehaviorContext, _binding, _bytes, _environment, _literal, _mapping, _signature,
    behavior_source_requirements, inspect_r1_material_behavior,
)
from solarlab.config.device_import import import_standard_device
from solarlab.config.resolve_device import resolve_device
from solarlab.config.scaps_input import import_scaps_device
from solarlab.device.resolved import PreparedDevice
from solarlab.materials.source import SourceDocument

__all__ = ["LegacyExecutionContext", "R1ExecutionContext", "r1_execution_source_requirements"]

_R1_EXECUTION_PINS = {
    "experiments/one_dimensional_mechanism_r1_state.py": "49bca65cf639024599781bead35db5524e5d61429efdc92bfb84e742af99422f",
    "experiments/one_dimensional_mechanism_r1_protocol.py": "7b053566d7931c7e16fe60ed46fabd9ca9a6c82461bc6116348ceb69dc8d4974",
    "experiments/one_dimensional_mechanism_r1_binding.py": "376ce06d1fa83e8c99bd5f69a22eb9778d903c6f20a755078a6a68c0aa562417",
    "experiments/one_dimensional_mechanism_r1_dynamics.py": "64d2fbb4bbae8e8e1b887f5ca0e8fd312d4b5cb5bd1faa5ecf31909048f995af",
    "experiments/one_dimensional_mechanism_r1_convergence.py": "a8e2b9738d551a5b5132cc3d99b57e534a0c5b5d4f7ab94e24f4c464406f1ca8",
    "experiments/interface_defect_ion_transient.py": "8c3298a7b5a1b500a175a7687c33248efeb1a8a36794ffac5a4986a02b22a573",
    "experiments/one_dimensional_mechanism_r1_pair_codec.py": "a2c23d94bdc0023ef24777613e6e5af70867bbd1759071381d64bdd47446b912",
}


def r1_execution_source_requirements() -> tuple[tuple[str, str], ...]:
    return tuple(sorted({**dict(behavior_source_requirements(driver="r1_material")), **_R1_EXECUTION_PINS}.items()))


def _snapshot(source: SourceDocument) -> dict[str, Any]:
    return {**_binding(source), "content_b64": base64.b64encode(source.content).decode("ascii")}


def _original_input(device: PreparedDevice, source: SourceDocument, *, scaps: bool) -> None:
    if type(device) is not PreparedDevice or type(source) is not SourceDocument:
        raise ValueError("historical invocation requires a prepared device and original source bytes")
    importer = import_scaps_device if scaps else import_standard_device
    expected = resolve_device(importer(source, id=device.to_input().id, defaults=device.defaults),
                              device.defaults, device.resources, sources=device.sources)
    if (expected.input_json != device.input_json
            or source.sha256 not in {item.sha256 for item in device.sources}):
        raise ValueError("original historical input differs from the prepared device")


@dataclass(frozen=True, slots=True)
class LegacyExecutionContext:
    """Exact original input and source-checked arguments for three old drivers."""

    behavior: BehaviorContext
    input_source: SourceDocument
    _document: bytes = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if type(self.behavior) is not BehaviorContext:
            raise ValueError("an actual historical BehaviorContext is required")
        _original_input(self.behavior.device, self.input_source,
                        scaps=self.behavior.driver.startswith("scaps_"))
        inspected = self.behavior.export()
        arguments = _mapping(self.behavior.arguments)
        # These are the existing saved request -> waveform driver bindings.
        # Retain the supplied request separately, including inactive choices.
        if self.behavior.driver == "ion_hysteresis":
            controls = arguments.pop("waveform_controls")
            for name in ("solver", "iface_states", "interface_boundary", "interface_transport_model"):
                arguments.pop(name, None)
            for key, name in (("rtol", "rtol"), ("atol", "atol_m3")):
                if key in arguments and arguments[key] != controls[name]:
                    raise ValueError("ambiguous waveform numerical controls")
                arguments[key] = controls[name]
        object.__setattr__(self, "_document", _bytes({
            "schema": "solarlab.legacy-execution-context.v1", "driver": self.behavior.driver,
            "input": _snapshot(self.input_source), "arguments_source": _snapshot(self.behavior.arguments),
            "arguments": arguments, "environment": inspected["environment_effective_snapshot"],
            "behavior": inspected, "behavior_sha256": self.behavior.content_sha256,
            "existing_driver_validation_required": True, "scientific_qualification": False,
        }))

    @property
    def content_sha256(self) -> str:
        return hashlib.sha256(self._document).hexdigest()

    def export(self) -> dict[str, Any]:
        return json.loads(self._document)


def _sealed(record: dict[str, Any], *, spaced: bool = False) -> None:
    payload = {key: value for key, value in record.items() if key != "sha256"}
    separators = (", ", ": ") if spaced else (",", ":")
    expected = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=separators, allow_nan=False).encode()).hexdigest()
    if record.get("sha256") != expected:
        raise ValueError("historical record content hash mismatch")


def _numeric_shape(value: Any) -> tuple[int, ...]:
    if type(value) is list:
        shapes = {_numeric_shape(item) for item in value}
        if len(shapes) > 1:
            raise ValueError("ragged R1 state array")
        return (len(value), *(next(iter(shapes)) if shapes else ()))
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("R1 state arrays require finite numbers, not booleans")
    return ()


def _state_shapes(prepared: dict[str, Any], codec: ast.Module, *, pair: bool) -> None:
    def names(key: str) -> tuple[str, ...]:
        value = next(node.value for node in codec.body if isinstance(node, ast.Assign)
                     and any(isinstance(target, ast.Name) and target.id == key for target in node.targets))
        if not isinstance(value, ast.Call) or len(value.args) != 1:
            raise ValueError("R1 snapshot source declaration changed")
        return tuple(ast.literal_eval(value.args[0]))
    state, grid = prepared["state"], prepared["grid"]
    n, interfaces = len(grid["coordinates_m"]), len(grid["interface_positions_m"])
    if n < 3 or interfaces != 1:
        raise ValueError("R1 requires the current one-interface physical geometry")
    required = set(names("LEGACY_SNAPSHOT_FIELDS"))
    fine = names("FINE_FIELDS") if pair else ()
    required.update("precision_" + name + suffix for name in fine for suffix in ("_hi", "_lo"))
    if set(state) != required:
        raise ValueError("R1 state field coverage differs from the bound codec")
    shapes = {name: _numeric_shape(value) for name, value in state.items()}
    expected = {**dict.fromkeys(("n_m3", "p_m3", "positive_m3", "phi_V", "dqfn_V", "dqfp_V", "positive_rate_m3_s"), (n,)),
                **dict.fromkeys(("occupancy", "sheet_charge_C_m2"), (interfaces,)),
                **dict.fromkeys(("electron_current_A_m2", "hole_current_A_m2", "positive_flux_m2_s"), (n - 1,)),
                "trace_potential_V": (interfaces, 2), "trace_state_m3": (interfaces, 4),
                "capture_m2_s": (interfaces, 4), "local_residual": (6 * interfaces,),
                "poisson_residual_C_m2": (n - 2,), "boundary_flux_m2_s": (2,),
                "positive_inventory_m2": (len(grid["positive_components"]),)}
    for name, shape in expected.items():
        if name in shapes and shapes[name] != shape:
            raise ValueError("R1 state shape differs from its actual grid: " + name)
    for name in fine:
        high, low = "precision_" + name + "_hi", "precision_" + name + "_lo"
        if shapes[high] != shapes[low] or name in expected and shapes[high] != expected[name]:
            raise ValueError("R1 high/low state shapes differ: " + name)
        if name in state and state[high] != state[name]:
            raise ValueError("R1 high word differs from its binary64 snapshot: " + name)
        if name == "storage" and (len(shapes[high]) != 1 or not 2 * (n - 2) + interfaces <= shapes[high][0] <= 3 * n - 4 + interfaces):
            raise ValueError("R1 storage has an invalid dimension")
    # Pair-word normalization and all live equations remain checked by the
    # original codec/reader. Their original words are never re-encoded here.


def _r1_arguments(source: SourceDocument, trees: dict[str, ast.Module]) -> dict[str, Any]:
    supplied = _mapping(source)
    allowed = {"intervals", "control", "amplitude_V", "times_s", "nonlinear_factor", "time_substeps", "backend", "physics_evidence"}
    if "intervals" not in supplied or supplied.keys() - allowed:
        raise ValueError("R1 invocation requires intervals and only the current public arguments")
    protocol = trees["experiments/one_dimensional_mechanism_r1_protocol.py"]
    defaults = _signature(protocol, "run_r1_step")
    policy_defaults = _signature(protocol, "r1_policy")
    args = {key: defaults[key] for key in ("control", "amplitude_V", "times_s", "backend", "physics_evidence")}
    args.update(policy_defaults)
    args.update(supplied)
    if type(args["intervals"]) is not int or args["intervals"] < 3:
        raise ValueError("R1 intervals require an integer of at least three")
    dynamics = trees["experiments/one_dimensional_mechanism_r1_dynamics.py"]
    controls = next(node for node in dynamics.body if isinstance(node, ast.ClassDef) and node.name == "R1DynamicsControls")
    factory = next(node for node in controls.body if isinstance(node, ast.FunctionDef) and node.name == "from_label")
    pairs = next(ast.literal_eval(node.value) for node in factory.body if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == "pairs" for target in node.targets))
    if type(args["control"]) is not str or args["control"].strip().upper() not in pairs:
        raise ValueError("R1 control must name a source-defined A-D choice")
    args["control"] = args["control"].strip().upper()
    if (type(args["amplitude_V"]) not in (float, int) or not math.isfinite(args["amplitude_V"])
            or not 0 < abs(args["amplitude_V"]) < .02):
        raise ValueError("R1 step amplitude is outside the current public domain")
    if args["times_s"] is None:
        args["times_s"] = list(_literal(protocol, "DEFAULT_TIMES_S"))
    times = args["times_s"]
    if (type(times) is not list or len(times) < 2
            or any(type(value) not in (int, float) or not math.isfinite(value) for value in times)
            or times[0] != 0 or times[1] < 1e-12 or any(b <= a for a, b in zip(times, times[1:]))):
        raise ValueError("R1 times must be a complete increasing finite protocol from zero")
    if type(args["nonlinear_factor"]) not in (int, float) or args["nonlinear_factor"] not in (1., .1, .01, .001):
        raise ValueError("R1 nonlinear factor is outside the existing choices")
    substeps = args["time_substeps"]
    if (not isinstance(substeps, (list, tuple)) or len(substeps) != 3
            or any(type(value) is not int or value < 1 for value in substeps)
            or substeps[0] & (substeps[0] - 1)
            or substeps[1] != 2 * substeps[0] or substeps[2] != 2 * substeps[1]):
        raise ValueError("R1 time substeps must retain the existing three nested levels")
    args["time_substeps"] = list(substeps)
    if type(args["physics_evidence"]) is not bool:
        raise ValueError("R1 physics_evidence requires an explicit boolean")
    args["controls"] = dict(zip(("nu_I", "nu_t"), pairs[args["control"]]))
    return args


@dataclass(frozen=True, slots=True)
class R1ExecutionContext:
    """Complete immutable R1 common-state invocation, including its history.

    Exact original and prepared bytes stay separate from normalized device
    metadata. The old restore_common_state reader must still recheck every
    live equation and current source before any step; a hash is not that check.
    """

    device: PreparedDevice
    input_source: SourceDocument
    sources: tuple[SourceDocument, ...]
    environment: tuple[tuple[str, str | None], ...]
    arguments: SourceDocument
    reference: SourceDocument
    study_input: SourceDocument
    prepared: SourceDocument
    policy: SourceDocument
    _document: bytes = field(init=False, repr=False)

    def __post_init__(self) -> None:
        _original_input(self.device, self.input_source, scaps=False)
        supplied_sources = tuple(self.sources)
        if any(type(source) is not SourceDocument for source in supplied_sources):
            raise ValueError("R1 execution requires immutable source documents")
        by_id = {source.id: source for source in supplied_sources}
        required = dict(r1_execution_source_requirements())
        if len(by_id) != len(supplied_sources) or set(by_id) != set(required) or any(by_id[key].sha256 != value for key, value in required.items()):
            raise ValueError("exact R1 execution sources are required")
        trees = {name: ast.parse(document.content) for name, document in by_id.items()}
        args = _r1_arguments(self.arguments, trees)
        snapshot = _environment(self.environment)
        material_ids = dict(behavior_source_requirements(driver="r1_material"))
        material = inspect_r1_material_behavior(self.device, sources=tuple(by_id[name] for name in sorted(material_ids)),
            environment=snapshot, backend=args["backend"])
        reference, study, prepared, policy = (_mapping(source) for source in (self.reference, self.study_input, self.prepared, self.policy))
        _sealed(reference, spaced=True)
        _sealed(prepared)
        binding_tree = trees["experiments/one_dimensional_mechanism_r1_binding.py"]
        if (self.study_input.sha256 != _literal(binding_tree, "PINNED_STUDY_INPUT_SHA256")
                or self.input_source.sha256 != study["fixture_sha256"]
                or prepared["source"]["study_input"]["sha256"] != self.study_input.sha256
                or args["controls"] != study["controls"][args["control"]]):
            raise ValueError("R1 input, controls and source must bind the actual fixed study")
        if reference["sha256"] != _literal(binding_tree, "PINNED_REFERENCE_BINDING_SHA256"):
            raise ValueError("R1 fixed reference differs from the existing study")
        state_tree = trees["experiments/one_dimensional_mechanism_r1_state.py"]
        schema = _literal(state_tree, "STATE_SCHEMA")
        pair = material["backend_representation"] == "float64-pair-v1"
        codec = trees["experiments/one_dimensional_mechanism_r1_pair_codec.py"]
        if pair:
            schema = _literal(codec, "SCHEMA")
            if prepared.get("schema") != schema:
                raise ValueError("R1 preparation representation binding mismatch")
            seed = prepared.get("seed_preparation", {})
            _sealed(seed)
            if (prepared.get("seed_preparation_sha256") != seed.get("sha256")
                    or prepared.get("representation") != _literal(codec, "REPRESENTATION")
                    or prepared.get("dc_state_role") != _literal(codec, "DC_STATE_ROLE")):
                raise ValueError("R1 pair state must preserve its separate original seed and representation")
            if set(prepared) != set(seed) | _literal(codec, "EXTRA_FIELDS"):
                raise ValueError("R1 pair preparation field coverage differs from its seed")
            for name in set(seed) - _literal(codec, "REPLACED_FIELDS"):
                if _bytes(prepared[name]) != _bytes(seed[name]):
                    raise ValueError("R1 pair changed retained seed metadata: " + name)
            _state_shapes(seed, codec, pair=False)
        if (prepared.get("schema") != schema or prepared.get("kind") != "equilibrium_D"
                or prepared.get("state_time") != "0-" or prepared.get("intervals") != args["intervals"]
                or prepared.get("preparation_controls") != {"nu_I": 1, "nu_t": 1}
                or prepared.get("fixed_reference") != reference
                or prepared.get("reference_sha256") != reference["sha256"]
                or prepared.get("study_spec_sha256") != _literal(state_tree, "STUDY_SPEC_SHA256")):
            raise ValueError("R1 preparation, reference, representation or protocol binding mismatch")
        _state_shapes(prepared, codec, pair=pair)
        history = next(node for node in state_tree.body if isinstance(node, ast.FunctionDef) and node.name == "_preparation_history")
        expected_history = ast.literal_eval(next(node.value for node in history.body
                                                if isinstance(node, ast.Return) and node.value is not None))
        if prepared.get("history") != expected_history or prepared.get("preparation_checks", {}).get("certified") is not True:
            raise ValueError("R1 requires the complete successful common-state preparation history")
        if hashlib.sha256(_bytes(prepared["physical_stack"])).hexdigest() != prepared.get("stack_sha256"):
            raise ValueError("R1 original physical-stack identity mismatch")
        producer_source = prepared["source"]
        _sealed(producer_source)
        if any(producer_source.get("files", {}).get(name) != digest for name, digest in required.items()):
            raise ValueError("R1 preparation used different current execution sources")
        if policy.get("refinement_substeps") != args["time_substeps"]:
            raise ValueError("R1 complete policy differs from the requested time levels")
        if set(policy) != set(prepared["preparation_policy"]):
            raise ValueError("R1 complete policy fields differ from the real preparation policy")
        object.__setattr__(self, "sources", tuple(by_id[name] for name in sorted(by_id)))
        object.__setattr__(self, "environment", snapshot)
        object.__setattr__(self, "_document", _bytes({
            "schema": "solarlab.r1-execution-context.v1", "driver": "r1_step",
            "prepared_device_sha256": self.device.content_sha256,
            "input": _snapshot(self.input_source), "arguments_source": _snapshot(self.arguments),
            "arguments": args, "environment": material["environment_effective_snapshot"],
            "material_behavior": material, "reference": _snapshot(self.reference),
            "study_input": _snapshot(self.study_input),
            "prepared": _snapshot(self.prepared), "prepared_sha256": prepared["sha256"],
            "policy": _snapshot(self.policy), "history": prepared["history"],
            "source_bindings": [_binding(by_id[name]) for name in sorted(by_id)],
            "preparation_environment": prepared["environment"], "preparation_created_utc": prepared["created_utc"],
            "existing_reader_and_policy_verification_required": True, "scientific_qualification": False,
        }))

    @property
    def content_sha256(self) -> str:
        return hashlib.sha256(self._document).hexdigest()

    def export(self) -> dict[str, Any]:
        return json.loads(self._document)
