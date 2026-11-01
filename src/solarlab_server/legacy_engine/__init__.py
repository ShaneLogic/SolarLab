"""Explicit high-level transition to the three source-audited old drivers.

The public entry prepares data for the shared supervisor. Only its registered
child worker imports the old engine, after manifest/environment/source checks.
Existing backend and research consumers remain registered until their separate
migration tasks; this facade does not activate routes or qualify new physics.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, fields, is_dataclass
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import struct
import sys
from typing import Any

from solarlab.config.behavior import BehaviorContext
from solarlab.config.identity import (
    BuildSource, ExecutionDescription, ExecutionEngine, ExecutionManifest, IdentityInputs,
    RuntimeEnvironment,
)
from solarlab.config.legacy_execution import LegacyExecutionContext
from solarlab.experiments.execution import JobHandle, ProcessSupervisor
from solarlab.experiments.worker import (
    ManifestBinding, ResourceLimits, WorkerBinding, WorkerContext, WorkerRequest, WorkerResult,
)
from solarlab.io.artifacts import bytes_artifact
from solarlab.materials.source import SourceDocument

__all__ = ["LegacyJob", "legacy_environment", "legacy_description", "prepare_legacy_job", "legacy_worker_binding"]

_THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")
_PACKAGES = ("numpy", "scipy", "PyYAML", "pydantic", "threadpoolctl")
_REGISTRY = SourceDocument("solarlab.legacy-facade-registry.v1", b'{"drivers":["ion_hysteresis","scaps_steady_state","scaps_transient"],"engine":"legacy","fallback":"forbidden"}')
_SCHEMA = SourceDocument("solarlab.legacy-facade-request.v1", b'{"fields":["engine","driver","input_id","input_base64","context_sha256"],"version":1}')


def _document(name: str, value: Any) -> SourceDocument:
    return SourceDocument(name, json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def _environment(invocation: LegacyExecutionContext) -> tuple[tuple[str, str | None], ...]:
    return tuple(sorted({**invocation.export()["environment"], **dict.fromkeys(_THREADS, "1")}.items()))


def legacy_environment(invocation: LegacyExecutionContext) -> RuntimeEnvironment:
    """Describe the requested child environment without changing the parent.

    Versions come from this installation. The worker must independently
    observe them after installing the explicit behavior/thread settings.
    """
    if type(invocation) is not LegacyExecutionContext:
        raise ValueError("an explicit supported legacy invocation is required")
    return RuntimeEnvironment(platform.python_implementation(), platform.python_version(), platform.system(), platform.machine(),
        tuple((name, importlib.metadata.version(name)) for name in _PACKAGES),
        tuple((key, json.dumps(value)) for key, value in _environment(invocation)))


def _roles(invocation: LegacyExecutionContext) -> dict[str, SourceDocument]:
    record = invocation.export()
    preparation = record["behavior"]["source_inspection"]["preparation"]
    # These original high-level drivers create their seed/grid internally.
    # Bind that exact recipe, never label it a supplied/qualified working point.
    return {
        "protocol": _document("legacy.protocol.v1", {"driver": record["driver"], "arguments": record["arguments"]}),
        "preparation": _document("legacy.preparation.v1", record),
        "initial_state": _document("legacy.initial-recipe.v1", {
            "input_sha256": invocation.input_source.sha256, "recipe": preparation,
            "source": "created_by_original_driver", "supplied_working_point": False}),
        "grid": _document("legacy.grid-recipe.v1", {"input_sha256": invocation.input_source.sha256,
            "N_grid": record["arguments"]["N_grid"], "source": "created_by_original_driver"}),
        "numeric_controls": _document("legacy.driver-controls.v1", preparation["actual_driver_arguments"]),
    }


def legacy_description(invocation: LegacyExecutionContext, *, implementation: BuildSource,
                       environment: RuntimeEnvironment, current_repository_head: str) -> ExecutionDescription:
    """Bind the full original invocation and its source-defined seed recipe."""
    roles = _roles(invocation)
    return ExecutionDescription(roles["protocol"], roles["preparation"], roles["initial_state"], roles["grid"],
                                roles["numeric_controls"], implementation, environment, current_repository_head)


@dataclass(frozen=True, slots=True)
class LegacyJob:
    request: WorkerRequest
    execution: ManifestBinding

    def submit(self, supervisor: ProcessSupervisor, run_id: str, limits: ResourceLimits) -> JobHandle:
        return supervisor.submit(run_id, "legacy", self.request, limits, execution=self.execution)


def prepare_legacy_job(
    invocation: LegacyExecutionContext, inputs: IdentityInputs, execution: ExecutionDescription, *,
    engine: ExecutionEngine, manifest: ExecutionManifest, expected_sha256: str,
) -> LegacyJob:
    """Prepare one explicit old-engine job; no execution or route activation."""
    if type(engine) is not ExecutionEngine or engine.name != "legacy":
        raise ValueError("this facade requires the explicit legacy engine; fallback is forbidden")
    if type(invocation) is not LegacyExecutionContext or type(engine.behavior) is not BehaviorContext:
        raise ValueError("this facade supports only the three audited BehaviorContext drivers")
    if engine.behavior.content_sha256 != invocation.behavior.content_sha256:
        raise ValueError("the engine and original invocation have different behavior contexts")
    if any(getattr(execution, name) != document for name, document in _roles(invocation).items()):
        raise ValueError("execution roles differ from the original legacy invocation")
    binding = ManifestBinding(manifest, inputs, execution, engine, expected_sha256, _environment(invocation))
    identity = manifest.export()["document"]["execution_identity"]
    request = WorkerRequest(_document("legacy.invocation.v1", {
        "engine": engine.name, "driver": invocation.behavior.driver,
        "input_id": invocation.input_source.id, "input_base64": base64.b64encode(invocation.input_source.content).decode(),
        "context_sha256": invocation.content_sha256,
    }), _REGISTRY, _SCHEMA, _document("legacy.execution-identity.v1", identity),
        identity["payload"]["H_input"], identity["payload"]["H_physics"])
    binding.validate_request(request)
    return LegacyJob(request, binding)


def legacy_worker_binding() -> WorkerBinding:
    return WorkerBinding.from_function("legacy", _run_legacy, registry_sha256=_REGISTRY.sha256, schema_sha256=_SCHEMA.sha256)


def _installed_sources(invocation: LegacyExecutionContext, execution: ExecutionDescription) -> Path:
    if "perovskite_sim" in sys.modules:
        raise ValueError("legacy engine was imported before source and environment verification")
    spec = importlib.util.find_spec("perovskite_sim")
    if spec is None or spec.origin is None:
        raise ValueError("the explicitly requested legacy engine is not installed")
    root = Path(spec.origin).resolve().parent
    required = {"perovskite_sim/" + source.id: source.content for source in invocation.behavior.sources}
    files = {source.id: source.content for source in execution.implementation.files}
    # SourceDocument deliberately excludes empty bytes. The original package
    # initializer is empty and is checked as such, never fabricated content.
    if Path(spec.origin).read_bytes() != b"":
        raise ValueError("legacy package initializer is no longer the original empty file")
    if not {"perovskite_sim/models/config_loader.py", *required} <= files.keys():
        raise ValueError("legacy implementation is missing required original source content")
    if any(files[key] != value for key, value in required.items()):
        raise ValueError("implementation differs from the audited behavior sources")
    for name, expected in files.items():
        if not name.startswith("perovskite_sim/"):
            raise ValueError("legacy implementation source IDs must be relative to perovskite_sim")
        path = root / name.removeprefix("perovskite_sim/")
        if not path.resolve().is_relative_to(root) or path.read_bytes() != expected:
            raise ValueError("installed legacy implementation content differs: " + name)
    actual = {"perovskite_sim/" + str(path.relative_to(root)) for path in root.rglob("*.py") if path.stat().st_size}
    if actual != files.keys():
        raise ValueError("installed nonempty legacy Python source coverage differs")
    for table in invocation.behavior.device.resources.tables:
        path = root / "data/nk" / (table.name + ".csv")
        if table.kind != "nk" or not path.resolve().is_relative_to(root / "data/nk"):
            raise ValueError("legacy facade has no source-bound resolver for this resource")
        if path.read_bytes() != table.source.content:
            raise ValueError("installed legacy resource content differs: " + table.name)
    return root


def _words(value: Any) -> Any:
    """Lossless transitional payload; no decimal projection of numeric words."""
    import numpy as np
    if is_dataclass(value) and not isinstance(value, type):
        return {"type": type(value).__qualname__, "fields": {f.name: _words(getattr(value, f.name)) for f in fields(value)}}
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise ValueError("object arrays are not a supported numeric result")
        return {"dtype": value.dtype.str, "shape": list(value.shape), "raw_hex": value.tobytes(order="C").hex()}
    if isinstance(value, np.generic):
        return {"dtype": value.dtype.str, "raw_hex": value.tobytes().hex()}
    if type(value) is float:
        return {"binary64_be": struct.pack(">d", value).hex()}
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is bytes:
        return {"bytes_b64": base64.b64encode(value).decode()}
    if isinstance(value, (list, tuple)):
        return [_words(item) for item in value]
    if isinstance(value, dict) and all(type(key) is str for key in value):
        return {key: _words(item) for key, item in value.items()}
    raise ValueError("unsupported original result type: " + type(value).__qualname__)


def _run_legacy(request: WorkerRequest, worker: WorkerContext) -> WorkerResult:
    binding = worker.consume_manifest(request, engine="legacy")
    body = json.loads(request.request.content)
    if (set(body) != {"engine", "driver", "input_id", "input_base64", "context_sha256"}
            or body["engine"] != "legacy" or type(binding.engine.behavior) is not BehaviorContext):
        raise ValueError("invalid explicit legacy invocation")
    invocation = LegacyExecutionContext(binding.engine.behavior,
        SourceDocument(body["input_id"], base64.b64decode(body["input_base64"], validate=True)))
    if body["driver"] != invocation.behavior.driver or body["context_sha256"] != invocation.content_sha256:
        raise ValueError("request differs from the source-bound original invocation")
    if _environment(invocation) != binding.environment:
        raise ValueError("worker environment differs from the actual behavior context")
    if any(getattr(binding.execution, name) != document for name, document in _roles(invocation).items()):
        raise ValueError("worker execution roles differ from the original legacy invocation")
    root = _installed_sources(invocation, binding.execution)
    worker.check_cancel()
    # Imports occur only inside the isolated child, after all checks above.
    from perovskite_sim.models.config_loader import load_device_from_yaml
    from perovskite_sim.scaps_compat.loader import load_scaps_yaml
    from perovskite_sim.experiments import jv_sweep, steady_state, waveform_jv
    from threadpoolctl import threadpool_info, threadpool_limits

    path = Path.cwd() / "original-input.yaml"
    with path.open("xb") as stream:
        stream.write(invocation.input_source.content)
    # Public preparation records already import NumPy for immutable resource
    # arrays. Bound the live BLAS pools too, including SciPy's newly loaded one.
    with threadpool_limits(limits=1):
        threadpools = threadpool_info()
        if any(item["num_threads"] != 1 for item in threadpools):
            raise ValueError("legacy child numerical thread limit was not installed")
        stack = (load_scaps_yaml(path) if invocation.behavior.driver.startswith("scaps_") else load_device_from_yaml(path))
        arguments = dict(invocation.export()["arguments"])
        worker.progress({"stage": "legacy_driver", "driver": invocation.behavior.driver,
                         "manifest_sha256": binding.expected_sha256})
        if invocation.behavior.driver == "scaps_transient":
            result = jv_sweep.run_jv_sweep(stack, **arguments)
        elif invocation.behavior.driver == "scaps_steady_state":
            result = steady_state.solve_voc_ss(stack, **arguments)
        elif invocation.behavior.driver == "ion_hysteresis":
            waveform = waveform_jv.JVWaveform.from_dict(arguments.pop("waveform"))
            result = waveform_jv.run_waveform_jv(stack, waveform, **arguments)
        else:
            raise ValueError("unsupported old driver; fallback is forbidden")
    worker.check_cancel()
    data = json.dumps(_words(result), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return WorkerResult.from_dict({"engine": "legacy", "driver": invocation.behavior.driver,
        "manifest_sha256": binding.expected_sha256, "H_execution": binding.manifest.execution_sha256,
        "input_sha256": invocation.input_source.sha256, "context_sha256": invocation.content_sha256,
        "legacy_source_root": str(root), "pid": os.getpid(), "threadpools": threadpools, "scientific_qualification": False,
        "payload_sha256": hashlib.sha256(data).hexdigest()}, artifacts=(("legacy_result", bytes_artifact(data, metadata=
            {"schema": "solarlab.legacy-raw-result.v1", "encoding": "exact_numeric_words"})),))
