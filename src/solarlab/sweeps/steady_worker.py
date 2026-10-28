"""Cold steady parameter points on the existing process scan and result store.

The prepared sweep remains a declaration, not scientific qualification. This
adapter creates ordinary steady jobs and a ScanSpec; the supplied supervisor
owns capacity, cancellation, retries and the sole terminal of every attempt.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from typing import TYPE_CHECKING, Any

from solarlab.config.identity import BuildSource, ExecutionEngine
from solarlab.device.resolved import PreparedDevice
from solarlab.experiments.execution import ProcessSupervisor, ScanPoint, ScanRunner, ScanSpec
from solarlab.experiments.jv.worker import (
    SteadyJVProtocol, prepare_steady_jv_job, read_steady_jv_job,
)
from solarlab.experiments.worker import ManifestBinding, ResourceLimits, RunSpec
from solarlab.sweeps.preparation import PreparedSweep, digest, encode, prepare_point

if TYPE_CHECKING:
    from solarlab.io.run_store import RunStore

__all__ = ["SteadySweepJobs", "prepare_steady_sweep_jobs", "read_steady_sweep_jobs"]


def _limits(limits: ResourceLimits) -> None:
    if type(limits) is not ResourceLimits:
        raise ValueError("explicit per-point ResourceLimits are required")
    if limits.accepted_steps is not None or limits.calls is not None:
        raise ValueError("steady workers do not report accepted_steps or calls; counter caps are unsupported")


@dataclass(frozen=True, slots=True)
class SteadySweepJobs:
    """Independently bound points; submission adds no pool or state cache."""

    spec: ScanSpec
    executions: tuple[ManifestBinding, ...]

    def __post_init__(self) -> None:
        executions = tuple(self.executions)
        if type(self.spec) is not ScanSpec or len(executions) != len(self.spec.points):
            raise ValueError("each declared point requires its own execution binding")
        for point, binding in zip(self.spec.points, executions):
            if point.spec.worker_id != "steady_jv":
                raise ValueError("only the installed steady_jv worker is supported")
            _limits(point.spec.limits)
            point.spec.validate(binding)
        object.__setattr__(self, "executions", executions)

    def submit(self, supervisor: ProcessSupervisor, *, max_in_flight: int = 1) -> ScanRunner:
        """Use this exact shared supervisor; serial admission is the default."""
        return supervisor.scan(self.spec, executions=self.executions, max_in_flight=max_in_flight)


def prepare_steady_sweep_jobs(
    sweep: PreparedSweep, protocol: SteadyJVProtocol, *, limits: ResourceLimits,
    implementation: BuildSource, current_repository_head: str, engine: ExecutionEngine,
    scan_id: str | None = None,
) -> SteadySweepJobs:
    """Resolve every point afresh and bind its actual cold steady invocation.

    An unresolved point or unsupported physics refuses the whole declaration
    before submission. Failures during execution remain individual RunStore
    attempts and do not prevent later points from running. Continuation is
    confined to the voltages of one point, never across parameter changes.
    """
    if type(sweep) is not PreparedSweep:
        raise TypeError("an actual PreparedSweep is required")
    if sweep.to_input().base.schema_version != "solarlab.device-preparation.v1":
        raise ValueError("steady process sweeps require a device base and an explicit steady protocol")
    _limits(limits)
    declaration = sweep.to_mapping()
    scan_id = declaration["id"] if scan_id is None else scan_id
    for point in declaration["points"]:
        if point["status"] != "prepared_pending_dependencies" or point["applied_input"] is None:
            raise ValueError(f"unresolved sweep point {point['id']}: {point.get('field_errors', [])}")
    points, executions = [], []
    for index, point in enumerate(declaration["points"]):
        prepared = prepare_point(point["applied_input"], sweep.defaults, sweep.resources, sweep.sources)
        if type(prepared) is not PreparedDevice:
            raise ValueError("every steady sweep point must resolve to PreparedDevice")
        job = prepare_steady_jv_job(prepared, protocol, implementation=implementation,
            current_repository_head=current_repository_head, engine=engine)
        binding = job.execution
        # Consume the existing declaration contract, including a fresh resolve
        # and comparison against the independently prepared point manifest.
        sweep.consume_point_manifest(point["id"], binding.manifest, binding.inputs, binding.execution,
            engine=engine, environment=binding.execution.environment,
            expected_sha256=binding.expected_sha256)
        coordinates = {
            "index": index, "sweep_sha256": sweep.content_sha256,
            "prepared_sha256": prepared.content_sha256,
            "declared_coordinates": point["coordinates"],
            "normalized_coordinates": point["normalized_coordinates"],
            "effective_targets": point["effective_targets"],
            "targets": {axis["id"]: axis["target"] for axis in declaration["axes"]},
            "units": {axis["id"]: axis["input_field_schema"].get("unit") for axis in declaration["axes"]},
        }
        run_id = "scan:" + digest([scan_id, point["id"]])
        spec = RunSpec.from_binding("steady_jv", job.request, limits, binding)
        points.append(ScanPoint(point["id"], run_id, encode(coordinates).decode(), spec))
        executions.append(binding)
    return SteadySweepJobs(ScanSpec(scan_id, tuple(points)), tuple(executions))


def read_steady_sweep_jobs(store: RunStore, spec: ScanSpec) -> tuple[dict[str, Any], ...]:
    """Read in declared order using the persisted ScanSpec; never run a solver.

    Undispatched points stay ``declared``. Each stored attempt retains its state,
    reason and raw result; only succeeded attempts acquire verified array data.
    Passing the original (or restored) ScanSpec checks the immutable point/run,
    coordinate, complete request, resource and manifest association.
    """
    if type(spec) is not ScanSpec:
        raise ValueError("the original or restored ScanSpec is required")
    associations = {row["point_id"]: row for row in store.sweep_points(spec.scan_id)}
    if set(associations) - {point.point_id for point in spec.points}:
        raise ValueError("stored sweep contains points outside this declaration")
    output = []
    for point in spec.points:
        coordinates = json.loads(point.coordinates_json)
        request = point.spec.request
        body = json.loads(request.request.content)
        if (point.spec.worker_id != "steady_jv" or body.get("engine") != "solarlab"
                or body.get("schema") != "solarlab.steady-jv.worker-request.v1"
                or coordinates.get("prepared_sha256") != body.get("prepared_sha256")):
            raise ValueError("point declaration is not its bound steady device invocation")
        value: dict[str, Any] = {
            "point_id": point.point_id, "run_id": point.run_id, "coordinates": coordinates,
            "state": "declared", "reason": "waiting_for_capacity", "attempt_id": None,
            "attempts": [], "result": None,
        }
        association = associations.get(point.point_id)
        if association is None:
            output.append(value)
            continue
        if (association["run_id"] != point.run_id or association["coordinates"] != coordinates
                or association["opaque_execution_identity"] != request.execution_identity.sha256):
            raise ValueError("stored sweep association differs from the declared point")
        run = store.get_run(point.run_id)
        metadata = run["request"]["input_metadata"]
        identity = {"H_input": request.H_input, "H_physics": request.H_physics,
                    "H_execution": request.execution_identity.sha256}
        if (run["request"]["identity"] != identity or metadata["transport"] != request.to_document()
                or metadata["limits"] != asdict(point.spec.limits)
                or metadata["worker_binding"]["id"] != point.spec.worker_id
                or metadata["execution_manifest"]["manifest_sha256"] != point.spec.manifest_sha256):
            raise ValueError("stored steady request differs from its declared execution")
        attempts = []
        for attempt in store.attempts(point.run_id):
            loaded = (read_steady_jv_job(store, point.run_id, attempt_id=attempt["attempt_id"])
                      if attempt["state"] == "succeeded" else None)
            if loaded is not None and loaded["worker_result"]["prepared_sha256"] != coordinates["prepared_sha256"]:
                raise ValueError("committed result belongs to a different prepared sweep point")
            raw = attempt["result"]
            reason = (raw.get("worker_result", raw).get("reason", attempt["state"])
                      if raw is not None else attempt["state"])
            attempts.append({**attempt, "reason": reason, "steady_result": loaded})
        latest = next(a for a in attempts if a["attempt_id"] == run["current_attempt_id"])
        value.update(state=latest["state"], reason=latest["reason"], attempt_id=latest["attempt_id"],
                     attempts=attempts, result=latest["steady_result"])
        output.append(value)
    return tuple(output)
