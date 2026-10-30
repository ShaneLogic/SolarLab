"""Explicit physical-density preparation followed by the full coupled ionic solve.

Select direct_density_seed_v1 for one fresh preparation, or explicitly supply
a pinned preparation to reuse. Both routes retain the original ionic profile
while forming a guess and then solve every coupled ionic/electronic equation.
The earlier electronic_density_seed_v1 auxiliary policy remains explicit for
its recorded failure evidence; it is never selected as a fallback.
"""

from __future__ import annotations

from solarlab.reproducibility.job_provenance import (
    JobProvenance, result_provenance,
    admitted_source_pins, current_provenance, job_provenance,
)

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from solarlab.equations.assembly.classical_1d import AssembledDC1DProblem
from solarlab.equations.dc_1d import DC1DProblem, State1D
from solarlab.equations.implementation import (
    DCProblem, provenance_binding, substage_binding, use_assembled,
)
from solarlab.equations.ionic_1d import CompiledIonicDevice1D, IonicState1D
from solarlab.experiments.ionic_transient import IonicTransientInitial, _admit_evidence, _same_state
from solarlab.experiments.transient_results import _decode, _encode
from solarlab.solvers.nonlinear.ionic_1d import IonicSteadyOptions, IonicSteadyResult, solve_ionic_steady_state
from solarlab.solvers.nonlinear.newton import NewtonResult
from solarlab.solvers.nonlinear.problem import state_digest
from solarlab.solvers.nonlinear.steady_1d import (
    SteadyState1D, SteadyStateError, density_basin_seed, solve_steady_state,
)

__all__ = ["IonicInitializationOptions", "IonicInitializationResult", "IonicDensitySeedArtifact",
           "load_ionic_density_seed", "initialize_ionic_steady_state",
           "write_ionic_initialization", "read_ionic_initialization"]

POLICY = "electronic_density_seed_v1"
DIRECT_POLICY = "direct_density_seed_v1"
SCHEMA = "solarlab.ionic_initialization.v1"


@dataclass(frozen=True)
class IonicInitializationOptions:
    """Unchanged coupled/electronic controls and a stage-boundary wall limit.

    The component's existing public solver has finite evaluation budgets but
    no interrupt callback. A caller needing a hard process deadline must also
    supervise the whole command; wall_s here is checked between stages.
    """

    steady: IonicSteadyOptions
    policy: str
    wall_s: float

    def __post_init__(self) -> None:
        if not isinstance(self.steady, IonicSteadyOptions):
            raise ValueError("explicit IonicSteadyOptions are required")
        if self.policy not in {POLICY, DIRECT_POLICY}:
            raise ValueError("explicit electronic_density_seed_v1 or direct_density_seed_v1 is required")
        if not math.isfinite(self.wall_s) or self.wall_s <= 0:
            raise ValueError("wall_s must be finite and positive")


@dataclass(frozen=True)
class IonicInitializationResult:
    accepted: bool
    termination: str
    voltage_V: float
    generation_scale: float
    initial: IonicTransientInitial
    solver_controls: dict[str, Any]
    compiled_identity: str
    component_identity: str
    binding_manifest: dict[str, Any]
    source_sha256: dict[str, str]
    initial_electronic_state: State1D
    density_seed: State1D | None
    component_result: SteadyState1D | None
    component_failure: NewtonResult | None
    coupled_initial_state: IonicState1D | None
    coupled_result: IonicSteadyResult | None
    stages: tuple[dict[str, Any], ...]
    evaluation_counts: dict[str, int]
    wall_s: float
    error: str | None
    physical_certificate_status: str = "not_evaluated"
    qualification: bool = False
    density_seed_provenance: dict[str, Any] | None = None

    job_provenance: dict[str, Any] | None = field(default_factory=current_provenance)

    @property
    def state(self) -> IonicState1D | None:
        return self.coupled_result.state if self.accepted and self.coupled_result is not None else None


@dataclass(frozen=True)
class IonicDensitySeedArtifact:
    """Pinned existing preparation evidence, never a certified ionic solution."""

    source_path: str
    source_sha256: str
    evidence_json: str


def load_ionic_density_seed(path: str | Path, sha256: str) -> IonicDensitySeedArtifact:
    """Read a recorded positive physical-density seed; perform no preparation."""
    read_ionic_initialization(path, sha256)
    text = Path(path).read_text()
    if hashlib.sha256(text.encode()).hexdigest() != sha256:
        raise ValueError("density seed artifact changed during read")
    return IonicDensitySeedArtifact(str(Path(path).resolve()), sha256, text)


def _admit_density_seed(
    compiled: CompiledIonicDevice1D, initial: IonicTransientInitial, voltage: float, generation: float,
    artifact: IonicDensitySeedArtifact, electronic_initial: State1D,
) -> tuple[State1D, dict[str, Any]]:
    if hashlib.sha256(artifact.evidence_json.encode()).hexdigest() != artifact.source_sha256:
        raise ValueError("density seed artifact SHA-256 mismatch")
    document = json.loads(artifact.evidence_json)
    saved = _decode(document["result"])
    if document.get("schema") != SCHEMA or document.get("bindings") != _request_bindings(saved):
        raise ValueError("density seed request/control/source identity mismatch")
    _validate_record(saved)
    if (saved["solver_controls"]["policy"] != POLICY or saved["density_seed"] is None
            or saved["compiled_identity"] != compiled.identity or saved["voltage_V"] != voltage
            or saved["generation_scale"] != generation
            or saved["initial"]["source_sha256"] != initial.source_sha256
            or saved["initial"]["evidence_json"] != initial.evidence_json
            or saved["solver_controls"]["steady"] != asdict(initial.steady_controls)
            or not any(stage["name"] == "physical_density_preparation" and stage["accepted"] is True
                       for stage in saved["stages"])):
        raise ValueError("recorded density seed does not match the declared device/initial/request")
    fields = saved["density_seed"]
    state = compiled.electronic_component.state(fields["u"], voltage_V=voltage)
    if (not np.array_equal(state.potential_V, electronic_initial.potential_V)
            or any(not np.array_equal(fields[name], getattr(state, name)) for name in
                   ("u", "potential_V", "electron_density_m3", "hole_density_m3"))
            or any(np.any(value <= 0) or not np.all(np.isfinite(value)) for value in
                   (state.electron_density_m3, state.hole_density_m3))):
        raise ValueError("recorded density fields disagree with their positive frozen-phi coordinates")
    return state, {**asdict(artifact), "mode": "saved", "original_preparation_counts": {
        name: saved["evaluation_counts"][name] for name in
        ("density_preparation_calls", "density_residual", "density_jacobian")},
        "producer_source_sha256": saved["source_sha256"],
        "selected_field": "density_seed", "physical_acceptance": False}




def initialize_ionic_steady_state(
    compiled: CompiledIonicDevice1D, *, initial: IonicTransientInitial,
    voltage_V: float, generation_scale: float, options: IonicInitializationOptions,
    density_seed_artifact: IonicDensitySeedArtifact | None = None,
) -> IonicInitializationResult:
    """A declared numerical seed path; final acceptance uses full ionic equations.

    Admission rechecks the pinned original input. The direct policy prepares
    positive finite carriers once at fixed phi, or reuses an explicitly supplied
    preparation. It immediately solves the full system with untouched ionic
    coordinates/inventory in the guess. Only the separately selected auxiliary
    policy runs an electronic-only Newton. Every stage/failure remains recorded.
    """
    started = time.monotonic()
    if not isinstance(compiled, CompiledIonicDevice1D) or not isinstance(options, IonicInitializationOptions):
        raise ValueError("an actual ionic device and explicit initialization options are required")
    if density_seed_artifact is not None and not isinstance(density_seed_artifact, IonicDensitySeedArtifact):
        raise ValueError("recorded density seed must carry an explicit artifact binding")
    if options.policy == POLICY and density_seed_artifact is not None:
        raise ValueError("electronic_density_seed_v1 performs its own declared preparation")
    if not isinstance(initial, IonicTransientInitial) or initial.steady_controls != options.steady:
        raise ValueError("a pinned accepted same-device initial and identical steady controls are required")
    voltage, generation = float(voltage_V), float(generation_scale)
    if not math.isfinite(voltage) or not math.isfinite(generation) or generation <= 0:
        raise ValueError("voltage must be finite and illuminated generation_scale must be positive")
    if any(channel.kind != "fixed_reservoir" for channel in compiled.electronic_component.channels):
        raise ValueError("ionic initialization currently requires fixed-reservoir contacts")
    domain = compiled.binding_manifest.get("ion_domain")
    if not isinstance(domain, dict) or domain.get("contacts") != ["blocking", "blocking"]:
        raise ValueError("ionic initialization requires one declared blocking domain")
    for species in compiled.ion_species_names:
        diffusion = getattr(compiled, species + "_ion_diffusivity_faces_m2_s")
        if not np.all(np.isfinite(diffusion)) or np.any(diffusion <= 0):
            raise ValueError("ionic initialization requires connected positive-D species")
    # In a process job this is a preparation substage of the already admitted
    # ionic curve, bound to its implementation; standalone initialization
    # declares its own operator. Revalidating the evidence evaluates the
    # equations, so it runs inside the job that names them.
    closure = options.steady.inventory_row_node
    with job_provenance({"compiled_identity": compiled.identity, "initial_sha256": initial.source_sha256, "voltage_V": voltage, "generation_scale": generation, "controls": asdict(options), "density_seed_sha256": None if density_seed_artifact is None else density_seed_artifact.source_sha256},
            **(provenance_binding("Ionic1DProblem.initialization", compiled, inventory_row_node=closure)
               if current_provenance() is None else substage_binding(compiled, inventory_row_node=closure))):
        admitted, old_generation, checks = _admit_evidence(
            compiled, initial.evidence_json, initial.source_sha256, initial.steady_controls)
        if not _same_state(admitted, initial.state) or old_generation != initial.generation_scale:
            raise ValueError("initial state/generation disagrees with its pinned original evidence")
        component = compiled.electronic_component
        electronic_u = component.rebase_coordinates(
            admitted.electronic_state.u, admitted.voltage_V, voltage)
        electronic_initial = component.state(electronic_u, voltage_V=voltage)
        pins = admitted_source_pins()
        density_seed = component_result = component_failure = coupled_initial = coupled_result = None
        density_seed_provenance = ({"mode": "fresh",
            "routine": "solarlab.solvers.nonlinear.steady_1d._density_basin_seed",
            "source_sha256": pins.get("solarlab.solvers.nonlinear.steady_1d"),
            "initial_electronic_sha256": state_digest(electronic_u), "voltage_V": voltage,
            "generation_scale": generation, "physical_acceptance": False}
            if options.policy == DIRECT_POLICY and density_seed_artifact is None else None)
        stages: list[dict[str, Any]] = [{"name": "saved_initial_admission", "accepted": True,
            "checks": checks, "source_sha256": initial.source_sha256,
            "state_sha256": state_digest(admitted.u), "generation_scale": old_generation}]
        counts = {"density_preparation_calls": 0, "density_residual": 0, "density_jacobian": 0,
                  "component_solver_calls": 0, "component_residual": 0, "component_jacobian": 0,
                  "coupled_solver_calls": 0, "coupled_residual": 0, "coupled_jacobian": 0}
        deadline = started + options.wall_s

        def finish(accepted: bool, termination: str, error: str | None = None) -> IonicInitializationResult:
            return IonicInitializationResult(accepted, termination, voltage, generation, initial, asdict(options),
                compiled.identity, component.identity, compiled.binding_manifest, pins, electronic_initial,
                density_seed, component_result, component_failure, coupled_initial, coupled_result,
                tuple(stages), dict(counts), time.monotonic() - started, error,
                density_seed_provenance=density_seed_provenance)

        def expired() -> bool:
            return time.monotonic() >= deadline

        if options.policy == DIRECT_POLICY and density_seed_artifact is not None:
            stage_started = time.monotonic()
            density_seed, density_seed_provenance = _admit_density_seed(
                compiled, initial, voltage, generation, density_seed_artifact, electronic_initial)
            stages.append({"name": "recorded_density_seed_admission", "accepted": True,
                "source_sha256": density_seed_artifact.source_sha256, "physical_acceptance": False,
                "wall_s": time.monotonic() - stage_started})
            electronic_for_coupled = density_seed
        else:
            # Count actual F/J calls to the unchanged physical-density preparation.
            class ObservedDensityProblem(DC1DProblem):
                def residual(self, u: ArrayLike, udot: ArrayLike | None = None) -> np.ndarray:
                    counts["density_residual"] += 1
                    return super().residual(u, udot)

                def jacobian(self, u: ArrayLike, udot: ArrayLike | None = None, *, alpha: float = 0.0) -> Any:
                    counts["density_jacobian"] += 1
                    return super().jacobian(u, udot, alpha=alpha)

            class ObservedAssembledDensityProblem(AssembledDC1DProblem):
                def residual(self, u: ArrayLike, udot: ArrayLike | None = None) -> np.ndarray:
                    counts["density_residual"] += 1
                    return super().residual(u, udot)

                def jacobian(self, u: ArrayLike, udot: ArrayLike | None = None, *, alpha: float = 0.0) -> Any:
                    counts["density_jacobian"] += 1
                    return super().jacobian(u, udot, alpha=alpha)

            observed: DCProblem = (ObservedAssembledDensityProblem(component, voltage, generation) if use_assembled()
                                   else ObservedDensityProblem(component, voltage, generation))
            if expired():
                return finish(False, "wall_budget_exhausted")
            stage_started = time.monotonic()
            counts["density_preparation_calls"] += 1
            try:
                seeded_u = density_basin_seed(observed, electronic_u)
                density_seed = component.state(seeded_u, voltage_V=voltage)
                if (not np.array_equal(density_seed.potential_V, electronic_initial.potential_V)
                        or any(not np.all(np.isfinite(a)) or np.any(a <= 0)
                               for a in (density_seed.electron_density_m3, density_seed.hole_density_m3))):
                    raise ValueError("density preparation violated frozen potential or positive carrier contract")
            except Exception as exc:
                error = f"{type(exc).__name__}: {str(exc)[:400]}"
                stages.append({"name": "physical_density_preparation", "accepted": False, "error": error,
                               "wall_s": time.monotonic() - stage_started})
                return finish(False, "density_preparation_failed", error)
            stages.append({"name": "physical_density_preparation", "accepted": True,
                "initial_coordinates": electronic_u, "candidate_coordinates": density_seed.u,
                "routine": "solarlab.solvers.nonlinear.steady_1d._density_basin_seed",
                "residual_evaluations": counts["density_residual"], "jacobian_evaluations": counts["density_jacobian"],
                "wall_s": time.monotonic() - stage_started, "physical_acceptance": False})
            if expired():
                return finish(False, "wall_budget_exhausted")
            if options.policy == POLICY:
                stage_started = time.monotonic()
                counts["component_solver_calls"] += 1
                try:
                    component_result = solve_steady_state(component, initial_state=density_seed,
                        voltage_V=voltage, generation_scale=generation, options=options.steady.electronic)
                    result = component_result.newton
                    counts["component_residual"] = result.residual_evaluations
                    counts["component_jacobian"] = result.jacobian_evaluations
                    if component_result.preparation or component_result.density_basin_initializations:
                        raise ValueError("supplied-initial component path unexpectedly performed extra preparation")
                except SteadyStateError as exc:
                    component_failure = exc.result
                    counts["component_residual"] = exc.result.residual_evaluations
                    counts["component_jacobian"] = exc.result.jacobian_evaluations
                    error = str(exc)[:500]
                    stages.append({"name": "electronic_component_solve", "accepted": False,
                        "termination": exc.result.termination.value, "error": error,
                        "wall_s": time.monotonic() - stage_started})
                    return finish(False, "component_" + exc.result.termination.value, error)
                except Exception as exc:
                    error = f"{type(exc).__name__}: {str(exc)[:400]}"
                    stages.append({"name": "electronic_component_solve", "accepted": False, "error": error,
                                   "wall_s": time.monotonic() - stage_started})
                    return finish(False, "component_evaluation_error", error)
                stages.append({"name": "electronic_component_solve", "accepted": True,
                    "termination": component_result.newton.termination.value,
                    "component_identity": component.identity, "full_ionic_acceptance": False,
                    "wall_s": time.monotonic() - stage_started})
                if expired():
                    return finish(False, "wall_budget_exhausted")
                electronic_for_coupled = component_result.state
            else:
                electronic_for_coupled = density_seed
        if options.policy == DIRECT_POLICY:
            stages.append({"name": "electronic_component_solve", "accepted": None, "status": "not_run",
                "reason": "explicit direct_density_seed_v1; no auxiliary solve"})
        if expired():
            return finish(False, "wall_budget_exhausted")
        stage_started = time.monotonic()
        try:
            joined = np.r_[electronic_for_coupled.u, admitted.u[3 * compiled.node_count:]]
            coupled_initial = compiled.state(joined, voltage_V=voltage)
            if (not np.array_equal(coupled_initial.u[3 * compiled.node_count:],
                                   admitted.u[3 * compiled.node_count:])
                    or coupled_initial.domain_inventories_m2 != admitted.domain_inventories_m2
                    or any(not np.array_equal(getattr(coupled_initial, species + "_ion_density_m3"),
                                              getattr(admitted, species + "_ion_density_m3"))
                           for species in compiled.ion_species_names)):
                raise ValueError("component seed changed original ionic coordinates/profile/inventory")
            counts["coupled_solver_calls"] += 1
            coupled_result = solve_ionic_steady_state(compiled, voltage_V=voltage,
                generation_scale=generation, options=options.steady, initial_state=coupled_initial)
            counts["coupled_residual"] = coupled_result.evaluation_counts["residual"]
            counts["coupled_jacobian"] = coupled_result.evaluation_counts["jacobian"]
        except Exception as exc:
            error = f"{type(exc).__name__}: {str(exc)[:400]}"
            stages.append({"name": "full_coupled_ionic_solve", "accepted": False, "error": error,
                           "wall_s": time.monotonic() - stage_started})
            return finish(False, "coupled_evaluation_error", error)
        stages.append({"name": "full_coupled_ionic_solve", "accepted": coupled_result.accepted,
            "termination": coupled_result.termination, "wall_s": time.monotonic() - stage_started})
        if expired():
            return finish(False, "wall_budget_exhausted")
        return finish(coupled_result.accepted, coupled_result.termination, coupled_result.error)


def _identity(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _request_bindings(record: dict[str, Any]) -> dict[str, Any]:
    controls = _identity(record["solver_controls"])
    sources = _identity(record["source_sha256"])
    request = {"policy": record["solver_controls"]["policy"], "voltage_V": record["voltage_V"],
        "generation_scale": record["generation_scale"], "compiled_identity": record["compiled_identity"],
        "component_identity": record["component_identity"], "controls_sha256": controls,
        "source_identity": sources, "initial_source_sha256": record["initial"]["source_sha256"],
        "initial_state_sha256": state_digest(record["initial"]["state"]["u"]),
        "initial_voltage_V": record["initial"]["state"]["electronic_state"]["voltage_V"],
        "initial_generation_scale": record["initial"]["generation_scale"]}
    if record.get("density_seed_provenance") is not None:
        request["density_seed_sha256"] = record["density_seed_provenance"]["source_sha256"]
        if record["density_seed_provenance"].get("mode") == "fresh":
            request["density_seed_mode"] = "fresh"
    return {"request": request, "request_sha256": _identity(request),
            "controls_sha256": controls, "source_identity": sources}


def _validate_record(record: dict[str, Any]) -> None:
    """Consistency of captured data only; never compile or recompute a solution."""
    seed = record.get("density_seed_provenance")
    result_provenance(record, expected_inputs=None if record.get("job_provenance") is None else {"compiled_identity": record["compiled_identity"],
        "initial_sha256": record["initial"]["source_sha256"], "voltage_V": record["voltage_V"],
        "generation_scale": record["generation_scale"], "controls": record["solver_controls"],
        "density_seed_sha256": seed["source_sha256"] if seed is not None and seed.get("mode", "saved") == "saved" else None})
    if record["qualification"] is not False or record["physical_certificate_status"] != "not_evaluated":
        raise ValueError("initialization evidence cannot claim physical qualification")
    options, initial = record["solver_controls"], record["initial"]
    if options["policy"] not in {POLICY, DIRECT_POLICY} or options["steady"] != initial["steady_controls"]:
        raise ValueError("initialization policy/controls mismatch")
    if (not math.isfinite(record["voltage_V"]) or not math.isfinite(record["generation_scale"])
            or record["generation_scale"] <= 0.0):
        raise ValueError("invalid illuminated initialization request")
    if hashlib.sha256(initial["evidence_json"].encode()).hexdigest() != initial["source_sha256"]:
        raise ValueError("original initial evidence SHA-256 mismatch")
    saved = json.loads(initial["evidence_json"])
    if (saved["summary"]["accepted"] is not True or saved["result"]["accepted"] is not True
            or saved["result"]["newton"]["accepted"] is not True
            or saved["summary"]["generation_scale"] != initial["generation_scale"]
            or saved["summary"]["voltage_V"] != initial["state"]["electronic_state"]["voltage_V"]
            or saved["result"]["compiled_identity"] != record["compiled_identity"]
            or initial["state"]["device_sha256"] != record["compiled_identity"]
            or saved["result"]["binding_manifest"] != record["binding_manifest"]
            or saved["result"]["solver_controls"] != options["steady"]
            or record["binding_manifest"]["electronic_component"]["compiled_sha256"] != record["component_identity"]
            or not np.array_equal(saved["result"]["state"]["u"], initial["state"]["u"])):
        raise ValueError("original initial/source/component binding mismatch")
    for electronic in (record["initial_electronic_state"], record["density_seed"]):
        if electronic is not None and (electronic["device_sha256"] != record["component_identity"]
                or electronic["voltage_V"] != record["voltage_V"]):
            raise ValueError("electronic seed voltage/component mismatch")
    component = record["component_result"]
    if options["policy"] == DIRECT_POLICY:
        provenance = record.get("density_seed_provenance")
        if provenance is None or component is not None or record["component_failure"] is not None:
            raise ValueError("direct density policy requires preparation provenance and no component solve")
        mode = provenance.get("mode", "saved")
        if mode not in {"fresh", "saved"} or record["evaluation_counts"]["component_solver_calls"] != 0:
            raise ValueError("invalid direct density source mode/component count")
        if mode == "fresh":
            if (provenance["routine"] != "solarlab.solvers.nonlinear.steady_1d._density_basin_seed"
                    or provenance["source_sha256"] != record["source_sha256"].get("solarlab.solvers.nonlinear.steady_1d")
                    or provenance["initial_electronic_sha256"] != state_digest(record["initial_electronic_state"]["u"])
                    or provenance["voltage_V"] != record["voltage_V"]
                    or provenance["generation_scale"] != record["generation_scale"]
                    or record["evaluation_counts"]["density_preparation_calls"] not in (0, 1)):
                raise ValueError("fresh density preparation source/input mismatch")
            seed = record["density_seed"]
            if seed is not None and (
                    record["evaluation_counts"]["density_preparation_calls"] != 1
                    or not np.array_equal(seed["potential_V"], record["initial_electronic_state"]["potential_V"])
                    or any(not np.all(np.isfinite(seed[name])) or np.any(seed[name] <= 0) for name in
                           ("electron_density_m3", "hole_density_m3"))):
                raise ValueError("fresh density seed violated its positive frozen-phi contract")
            if record["accepted"] and any(record["evaluation_counts"][name] != 1 for name in
                                         ("density_preparation_calls", "density_residual", "density_jacobian")):
                raise ValueError("accepted fresh preparation must retain its single F/J evaluation")
        else:
            if hashlib.sha256(provenance["evidence_json"].encode()).hexdigest() != provenance["source_sha256"]:
                raise ValueError("recorded density seed source mismatch")
            seed_document = json.loads(provenance["evidence_json"])
            seed_record = _decode(seed_document["result"])
            if (seed_document.get("bindings") != _request_bindings(seed_record)
                    or seed_record["solver_controls"]["policy"] != POLICY
                    or seed_record["voltage_V"] != record["voltage_V"]
                    or seed_record["generation_scale"] != record["generation_scale"]
                    or seed_record["compiled_identity"] != record["compiled_identity"]
                    or seed_record["initial"]["source_sha256"] != initial["source_sha256"]
                    or provenance["selected_field"] != "density_seed"
                    or not np.array_equal(seed_record["density_seed"]["u"], record["density_seed"]["u"])
                    or record["evaluation_counts"]["density_preparation_calls"] != 0):
                raise ValueError("direct density request/seed/call binding mismatch")
            _validate_record(seed_record)
    if component is not None and (component["voltage_V"] != record["voltage_V"]
            or component["generation_scale"] != record["generation_scale"]
            or component["state"]["device_sha256"] != record["component_identity"]
            or component["newton"]["accepted"] is not True or component["preparation"]
            or component["density_basin_initializations"] != 0
            or component["newton"]["start_digest"] != state_digest(record["density_seed"]["u"])):
        raise ValueError("auxiliary component condition/seed mismatch")
    joined = record["coupled_initial_state"]
    if joined is not None:
        n = record["initial_electronic_state"]["potential_V"].size
        expected = record["density_seed"] if options["policy"] == DIRECT_POLICY else component["state"]
        if expected is None or not np.array_equal(joined["u"][:3*n], expected["u"]):
            raise ValueError("coupled initial is not the declared density/component seed")
        if (joined["device_sha256"] != record["compiled_identity"]
                or joined["electronic_state"]["voltage_V"] != record["voltage_V"]
                or not np.array_equal(joined["u"][3*n:], initial["state"]["u"][3*n:])
                or any(not np.array_equal(joined[key], initial["state"][key]) for key in
                       ("positive_ion_density_m3", "negative_ion_density_m3", "domain_inventory_m2",
                        "negative_domain_inventory_m2"))):
            raise ValueError("coupled seed changed the admitted ionic profile/inventory")
    coupled = record["coupled_result"]
    if coupled is not None:
        if (coupled["compiled_identity"] != record["compiled_identity"]
                or coupled["binding_manifest"] != record["binding_manifest"]
                or coupled["solver_controls"] != options["steady"]
                or joined is None or not np.array_equal(coupled["initial_state"]["u"], joined["u"])):
            raise ValueError("coupled controls/source/initial mismatch")
        state = coupled["state"]
        if state is not None and (state["device_sha256"] != record["compiled_identity"]
                or state["electronic_state"]["voltage_V"] != record["voltage_V"]
                or state["electronic_state"]["device_sha256"] != record["component_identity"]):
            raise ValueError("coupled state voltage/component mismatch")
    if record["accepted"] and (coupled is None or coupled["accepted"] is not True
            or (options["policy"] == POLICY and component is None) or joined is None or coupled["state"] is None
            or coupled["newton"]["accepted"] is not True or not coupled["checks"]
            or any(item["passed"] is not True for item in coupled["checks"])
            or record["evaluation_counts"]["coupled_solver_calls"] != 1):
        raise ValueError("accepted initialization lacks original coupled acceptance")


def write_ionic_initialization(
    result: IonicInitializationResult | dict[str, Any], path: str | Path,
) -> str:
    """Save a result or its decoded readback without changing its producer metadata."""
    encoded = _encode(result)
    record = _decode(encoded)
    _validate_record(record)
    document = {"schema": SCHEMA, "bindings": _request_bindings(record), "result": encoded}
    raw = (json.dumps(document, indent=2, allow_nan=False) + "\n").encode()
    with Path(path).open("xb") as stream:
        stream.write(raw)
    return hashlib.sha256(raw).hexdigest()


def read_ionic_initialization(path: str | Path, sha256: str, *,
    current_identity: JobProvenance | dict[str, Any] | None = None,
    strict_same_source: bool = False,
) -> dict[str, Any]:
    """Verify the supplied digest and read evidence without any solver/device work."""
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha256:
        raise ValueError("ionic initialization SHA-256 mismatch")
    document = json.loads(raw)
    if document.get("schema") != SCHEMA:
        raise ValueError("unsupported ionic initialization schema")
    try:
        result = _decode(document["result"])
        if document.get("bindings") != _request_bindings(result):
            raise ValueError("initialization request/control/source identity mismatch")
        _validate_record(result)
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError("malformed ionic initialization evidence") from exc
    return {"schema": SCHEMA, "bindings": document["bindings"], "result": result,
            "provenance_comparison": result_provenance(result, current_identity, strict_same_source=strict_same_source)}
