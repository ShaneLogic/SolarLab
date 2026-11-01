"""The 2D steady production entry under the assembled default and the legacy opt-out.

At Core's 20 DC2D probe states (5 harness factories x 4 states) the selected production problem
gives the same diagnostics, checks and every readout (the 500 vectors of the binding test, the
fixed-port values through ``Assembler.reaction_flux_ordered``) in every binary64 word. Bounded
``solve_steady_state_2d`` histories from every probe, the research-example dark/light path and a
three-point 2D J-V wave with its saved artifact and reader are identical outside wall-clock
fields. Every assembled run forbids all original DC2D evaluation methods and counts Assembler
calls. The J-V job's Q6 equation binding is asserted for each selection (support.equation_provenance)
before it is set aside; admitted source pins do not depend on the selection.

An explicit wall-prefix continuation admits its job before it readmits the prefix (Science PW2-C1):
under the default and under ``NEWCORE_EQUATIONS=legacy``, a wave, its continuation and its final
endpoint each admit their binding with no equation evaluated, and every diagnostics call runs the
selected class under that exact binding. Continuing one saved prefix gives the same tail and
endpoint under either selection.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
import dataclasses
from dataclasses import asdict, replace
import enum
import json
from pathlib import Path
import time
from typing import Any
from unittest import mock

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from examples.research_2d import dc_assembly, steady as steady_example
from solarlab.device.compile_2d import compile_device_2d
from solarlab.equations.assembly.assembler import Assembler
from solarlab.equations.assembly.two_dimensional import AssembledDC2DProblem
from solarlab.equations.dc_2d import CompiledDevice2D, DC2DProblem
from solarlab.equations.implementation import ASSEMBLED, ENVIRONMENT, LEGACY, equation_implementation
from solarlab.experiments.jv import steady_2d as jv_2d
from solarlab.experiments.transient_results import _encode
from solarlab.reproducibility.job_provenance import current_provenance
from solarlab.solvers.nonlinear import steady_2d
from solarlab.solvers.nonlinear.steady_1d import SteadyOptions
from support import equation_provenance
from support.assembly_references import REFERENCE_FACTORIES
from support.provenance import checked_provenance_owner

FACTORIES = ("dc2d:fixed", "dc2d:contacts", "dc2d:ct", "dc2d:grain", "dc2d:composed")
WALL_CLOCK = frozenset({"wall_s"})
ORIGINAL_EVALUATIONS = ("residual", "jacobian", "fluxes", "recombination_rate", "displacement", "current_density",
                        "cross_section_current_A_m", "contact_particle_fluxes", "terminal_current_A_m",
                        "terminal_current_density_A_m2")
READOUTS = ("displacement", "current_density", "cross_section_current_A_m", "terminal_current_A_m",
            "terminal_current_density_A_m2")
JV = "DC2DProblem.steady_jv"
WAVE = np.array([0.0, 0.025, 0.05, 0.075, 0.1])


@pytest.fixture(scope="module", autouse=True)
def blas1() -> Iterator[None]:
    with threadpool_limits(limits=1):
        yield


@contextmanager
def originals_forbidden() -> Iterator[None]:
    """Every original DC2D evaluation method raises."""
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("an original DC2D evaluation ran under the assembled selection")

    with ExitStack() as stack:
        for method in ORIGINAL_EVALUATIONS:
            stack.enter_context(mock.patch.object(DC2DProblem, method, forbidden))
        yield


@contextmanager
def assembled_only() -> Iterator[list[int]]:
    """The assembled selection with every original DC2D evaluation forbidden; counts Assembler calls."""
    calls: list[int] = []
    original = Assembler.evaluate

    def counted(assembler: Assembler, *args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        return original(assembler, *args, **kwargs)

    with ExitStack() as stack:
        stack.enter_context(originals_forbidden())
        stack.enter_context(mock.patch.object(Assembler, "evaluate", counted))
        stack.enter_context(equation_implementation(ASSEMBLED))
        yield calls


def words(value: Any) -> Any:
    """Every result leaf; floats and arrays as their binary64 words, wall clock dropped."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return (type(value).__name__, tuple((item.name, words(getattr(value, item.name)))
                                            for item in dataclasses.fields(value) if item.name not in WALL_CLOCK))
    if isinstance(value, dict):
        return ("dict", tuple((key, words(item)) for key, item in value.items() if key not in WALL_CLOCK))
    if isinstance(value, (list, tuple)):
        return (type(value).__name__, tuple(words(item) for item in value))
    if isinstance(value, np.ndarray):
        return ("array", value.dtype.str, value.shape, np.ascontiguousarray(value).tobytes())
    if isinstance(value, (float, np.floating)):
        return ("float", np.float64(value).tobytes())
    if isinstance(value, enum.Enum):
        return ("enum", type(value).__name__, value.value)
    return ("value", value)


def masked(document: Any) -> Any:
    if isinstance(document, dict):
        return {key: masked(item) for key, item in document.items() if key not in WALL_CLOCK}
    if isinstance(document, list):
        return [masked(item) for item in document]
    return document


def bits(value: Any) -> bytes:
    return np.ascontiguousarray(value, dtype=np.float64).tobytes()


def readouts(problem: Any, u: np.ndarray, device: CompiledDevice2D, voltage: float) -> list[bytes]:
    """The binding test's readout vectors, in its order, as binary64 words."""
    vectors = [bits(getattr(problem, name)(u)) for name in READOUTS]
    for carrier in asdict(problem.fluxes(u)).values():
        vectors.extend(bits(item) for item in carrier.values())
    _, n, p = device.fields(u, voltage_V=voltage)
    for derivatives in (False, True):
        vectors.extend(bits(item) for item in asdict(problem.recombination_rate(n, p, derivatives=derivatives)).values()
                       if item is not None)
    vectors.extend(bits(item) for item in problem.contact_particle_fluxes(u))
    return vectors


def test_the_twenty_probes_give_the_same_diagnostics_readouts_and_ordered_ports() -> None:
    options = steady_2d.Steady2DOptions()
    probes = vectors = ordered_ports = 0
    for name in FACTORIES:
        case = REFERENCE_FACTORIES[name]()
        device = case.device
        for state in case.states:
            probes += 1
            with equation_implementation(LEGACY):
                legacy = steady_2d._problem(device, state.voltage_V, state.generation_scale)
                legacy_values = steady_2d.diagnostics(legacy, state.u, options)
                legacy_readouts = readouts(legacy, state.u, device, state.voltage_V)
                legacy_ports = legacy.contact_particle_fluxes(state.u)
            assert type(legacy) is DC2DProblem
            with assembled_only() as calls:
                candidate = steady_2d._problem(device, state.voltage_V, state.generation_scale)
                candidate_values = steady_2d.diagnostics(candidate, state.u, options)
                candidate_readouts = readouts(candidate, state.u, device, state.voltage_V)
                assert type(candidate) is AssembledDC2DProblem
                # Fixed-reservoir ports read out through the ordered-port API, word for word.
                system = candidate.binding
                for index, (block, carrier) in enumerate((("electron", "n"), ("hole", "p"))):
                    for side, (port, channel_side) in enumerate((("y_min", "left"), ("y_max", "right"))):
                        if device.column.channel(carrier, channel_side).kind != "fixed_reservoir":
                            continue
                        ordered = system.assembler.reaction_flux_ordered(
                            state.u, candidate.inputs, block, port, normal_faces=system.geometry.normal_faces(port),
                            transverse_faces=system.geometry.transverse_faces)
                        assert bits(ordered) == bits(legacy_ports[index][side])
                        ordered_ports += 1
            assert calls
            assert words(candidate_values) == words(legacy_values)
            assert words(steady_2d.checks(candidate_values, options)) == words(steady_2d.checks(legacy_values, options))
            assert candidate_readouts == legacy_readouts
            vectors += len(candidate_readouts)
    assert (probes, vectors) == (20, 500) and ordered_ports > 0


@pytest.mark.parametrize("name", FACTORIES)
def test_bounded_production_solves_from_every_probe_are_identical(name: str) -> None:
    case = REFERENCE_FACTORIES[name]()
    device = case.device
    options = steady_2d.Steady2DOptions(replace(SteadyOptions(), max_iterations=4), 30.0)
    for state in case.states:
        initial = device.state(state.u, voltage_V=state.voltage_V)
        with equation_implementation(LEGACY):
            reference = steady_2d.solve_steady_state_2d(device, voltage_V=state.voltage_V,
                generation_scale=state.generation_scale, options=options, initial_state=initial)
        with assembled_only() as calls:
            candidate = steady_2d.solve_steady_state_2d(device, voltage_V=state.voltage_V,
                generation_scale=state.generation_scale, options=options, initial_state=initial)
        assert calls and candidate.newton is not None and candidate.newton.iterations
        assert words(candidate) == words(reference)


def test_research_steady_path_saved_artifact_and_reader(tmp_path: Path) -> None:
    saved = {}
    for implementation in (LEGACY, ASSEMBLED):
        output = tmp_path / f"{implementation}.json"
        with equation_implementation(LEGACY) if implementation == LEGACY else assembled_only():
            record = steady_example.run_case(output)
        assert record["completed"]
        saved[implementation] = output
    left, right = (json.loads(saved[name].read_bytes()) for name in (LEGACY, ASSEMBLED))
    assert json.dumps(masked(right)) == json.dumps(masked(left))
    for implementation in (LEGACY, ASSEMBLED):
        with equation_implementation(implementation):
            reads = [steady_example.read_result(path, _sha256(path)) for path in saved.values()]
        assert json.dumps(masked(_encode(reads[1]))) == json.dumps(masked(_encode(reads[0])))


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_jv_wave_admission_points_saved_artifact_and_reader(tmp_path: Path) -> None:
    device = compile_device_2d(steady_example.load_device(), dc_assembly.spatial_input())
    with equation_implementation(LEGACY):
        dark = steady_example.run_case(tmp_path / "Steady.json")
    assert dark["completed"]
    initial_path = tmp_path / "Steady.json"
    controls = jv_2d.SteadyJV2DOptions()
    initial = jv_2d.load_steady_jv_2d_initial(device, initial_path, _sha256(initial_path), options=controls.steady)
    voltages = np.array([0.0, 0.025, 0.05])
    saved = {}
    for implementation in (LEGACY, ASSEMBLED):
        with equation_implementation(LEGACY) if implementation == LEGACY else assembled_only():
            result = jv_2d.run_steady_jv_2d(device, initial, voltages, options=controls,
                                            checkpoint_directory=tmp_path / f"{implementation}.checkpoints")
        assert len(result.attempts) == voltages.size and all(attempt.accepted for attempt in result.attempts)
        path = tmp_path / f"{implementation}Jv.json"
        saved[implementation] = (jv_2d.write_steady_jv_2d(result, path), path, result)
    operator = "DC2DProblem.steady_jv"
    assert words(equation_provenance.checked(saved[ASSEMBLED][2], operator, device, ASSEMBLED)) == words(
        equation_provenance.checked(saved[LEGACY][2], operator, device, LEGACY))
    documents = {}
    for implementation in (LEGACY, ASSEMBLED):
        document, found = equation_provenance.checked_document(
            json.loads(saved[implementation][1].read_bytes()), operator, device, implementation)
        assert found
        documents[implementation] = document
    assert json.dumps(masked(documents[ASSEMBLED])) == json.dumps(masked(documents[LEGACY]))
    reads = {}
    for implementation, (digest, path, _) in saved.items():
        reads[implementation], found = equation_provenance.checked_document(
            _encode(jv_2d.read_steady_jv_2d(path, digest)), operator, device, implementation)
        assert found
    assert json.dumps(masked(reads[ASSEMBLED])) == json.dumps(masked(reads[LEGACY]))


def test_2d_admitted_source_pins_cover_both_implementations_and_do_not_depend_on_the_selection() -> None:
    from solarlab.reproducibility.job_provenance import job_provenance

    device = compile_device_2d(steady_example.load_device(), dc_assembly.spatial_input())
    maps = {}
    for implementation in (LEGACY, ASSEMBLED):
        with equation_implementation(implementation), checked_provenance_owner(), \
                job_provenance({"device": device.identity}, operator="test.source_map"):
            maps[implementation] = list(steady_2d.sources(device).items())
    assert maps[LEGACY] == maps[ASSEMBLED]
    assert {"solarlab.equations.implementation", "solarlab.equations.assembly.two_dimensional",
            "solarlab.equations.assembly.rectangular_2d", "solarlab.equations.dc_2d",
            "solarlab.experiments.jv.steady_2d"} <= set(dict(maps[ASSEMBLED]))


@pytest.fixture(scope="module")
def jv_case(tmp_path_factory: pytest.TempPathFactory) -> tuple[CompiledDevice2D, jv_2d.SteadyJV2DInitial]:
    """The research device and its accepted saved initial (the research dark/light example)."""
    path = tmp_path_factory.mktemp("jv2d") / "Steady.json"
    with equation_implementation(LEGACY):
        assert steady_example.run_case(path)["completed"]
    device = compile_device_2d(steady_example.load_device(), dc_assembly.spatial_input())
    return device, jv_2d.load_steady_jv_2d_initial(device, path, _sha256(path),
                                                   options=jv_2d.SteadyJV2DOptions().steady)


@contextmanager
def wall_stop(count: int, controls: jv_2d.SteadyJV2DOptions) -> Iterator[Callable[[jv_2d.SteadyJV2DAttempt], None]]:
    """An ``on_point`` that moves the driver's clock past the wave's wall budget after ``count`` points.

    The next point then stops before its solver call, as a wave whose wall budget ran out does.
    """
    offset = [0.0]

    class Clock:
        @staticmethod
        def monotonic() -> float:
            return time.monotonic() + offset[0]

    def on_point(attempt: jv_2d.SteadyJV2DAttempt) -> None:
        if attempt.index + 1 == count:
            offset[0] = controls.wall_s

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(jv_2d, "time", Clock)
        yield on_point


@contextmanager
def admitted_first(device: CompiledDevice2D, implementation: str, diagnostics: int) -> Iterator[None]:
    """The J-V producer called inside admits its job before evaluating any equation or diagnostics.

    Then each of its ``diagnostics`` calls evaluates the selected class, and each equation evaluation
    runs, under the exact selected binding; the other implementation evaluates nothing.
    """
    from solarlab.reproducibility import job_provenance as provenance

    binding = equation_provenance.expected(JV, device, implementation)
    events: list[tuple[Any, ...]] = []
    equations: list[Any] = []
    capture, diagnosed = provenance.capture_job_provenance, jv_2d.steady_diagnostics

    def admitted_binding() -> Any:
        active = current_provenance()
        return None if active is None else active["record"]["assembler"]

    with ExitStack() as stack:
        if implementation == ASSEMBLED:
            stack.enter_context(originals_forbidden())
        observed = stack.enter_context(equation_provenance.counted())
        for owner, name in ((Assembler, "evaluate"), (DC2DProblem, "residual"), (DC2DProblem, "jacobian")):
            def evaluated(self: Any, *args: Any, __method: Any = getattr(owner, name), **kwargs: Any) -> Any:
                equations.append(admitted_binding())
                return __method(self, *args, **kwargs)
            stack.enter_context(mock.patch.object(owner, name, evaluated))
        patch = stack.enter_context(pytest.MonkeyPatch.context())

        def admitted(*args: Any, **kwargs: Any) -> Any:
            events.append(("admitted", observed.original, observed.assembled))
            return capture(*args, **kwargs)

        def spied(problem: Any, *args: Any, **kwargs: Any) -> Any:
            events.append(("diagnostics", type(problem), admitted_binding()))
            return diagnosed(problem, *args, **kwargs)

        patch.setattr(provenance, "capture_job_provenance", admitted)
        patch.setattr(jv_2d, "steady_diagnostics", spied)
        yield
    selected = AssembledDC2DProblem if implementation == ASSEMBLED else DC2DProblem
    assert events == [("admitted", 0, 0)] + [("diagnostics", selected, binding)] * diagnostics
    assert equations and all(item == binding for item in equations)
    if implementation == ASSEMBLED:
        assert observed.original == 0 and observed.assembled > 0
        assert observed.identities <= equation_provenance.allowed(JV, device)
    else:
        assert observed.assembled == 0 and observed.original > 0


def saved(result: jv_2d.SteadyJV2DResult, path: Path) -> tuple[Path, str]:
    return path, jv_2d.write_steady_jv_2d(result, path)


@pytest.mark.parametrize("legacy", (False, True), ids=("default", "NEWCORE_EQUATIONS=legacy"))
def test_explicit_continuation_admits_its_binding_before_readmitting_its_prefix(
    jv_case: tuple[CompiledDevice2D, jv_2d.SteadyJV2DInitial], legacy: bool, monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A wave stopped by its wall budget, its continuation (stopped again) and its final original endpoint.

    The process setting selects. Each continuation readmits the original initial and every accepted
    prefix row inside its job, after admitting the selected binding (Science PW2-C1).
    """
    device, initial = jv_case
    if legacy:
        monkeypatch.setenv(ENVIRONMENT, LEGACY)
    else:
        monkeypatch.delenv(ENVIRONMENT, raising=False)
    implementation = LEGACY if legacy else ASSEMBLED
    controls = jv_2d.SteadyJV2DOptions()
    # The saved initial is readmitted, then two new points are checked.
    with admitted_first(device, implementation, 1 + 2), wall_stop(2, controls) as on_point:
        prefix = jv_2d.run_steady_jv_2d(device, initial, WAVE, options=controls, on_point=on_point)
    prefix_path, prefix_sha = saved(prefix, tmp_path / "Prefix.json")
    # The original initial and both prefix rows are readmitted, then two new points are checked.
    with admitted_first(device, implementation, 1 + 2 + 2), wall_stop(2, controls) as on_point:
        tail = jv_2d.continue_steady_jv_2d(device, prefix_path, prefix_sha, options=controls, on_point=on_point)
    tail_path, tail_sha = saved(tail, tmp_path / "Tail.json")
    # The original initial and the four accepted rows of both artifacts, then the endpoint.
    with admitted_first(device, implementation, 1 + 4 + 1):
        endpoint = jv_2d.continue_steady_jv_2d(device, tail_path, tail_sha, options=controls)
    endpoint_path, endpoint_sha = saved(endpoint, tmp_path / "Endpoint.json")
    for stopped in (prefix, tail):
        assert len(stopped.voltage_V) == 2 and not stopped.attempts[-1].solver_called
        assert stopped.attempts[-1].termination == "wave_wall_budget_exhausted"
    assert endpoint.completed and tail.attempts[0].seed_provenance["kind"] == "accepted_wall_prefix"
    np.testing.assert_array_equal(jv_2d.read_steady_jv_2d(endpoint_path, endpoint_sha)["continuation"]["combined_voltage_V"],
                                  WAVE)
    for path, envelopes in ((prefix_path, 1), (tail_path, 3), (endpoint_path, 3)):
        assert equation_provenance.checked_document(json.loads(path.read_bytes()), JV, device, implementation)[1] == envelopes


def continuation_document(document: Any, device: CompiledDevice2D, implementation: str,
                          envelopes: int) -> tuple[Any, str]:
    """A continuation of a legacy artifact with every Q6 binding asserted and set aside.

    The continued artifact's binding is legacy and the others are ``implementation``'s. Also
    returns, set aside, the source comparison between the two, which follows from their bindings.
    """
    producers = document["continuation"]["producer_provenance"]
    record, binding = equation_provenance.split(producers.pop("prefix"))
    assert binding == equation_provenance.expected(JV, device, LEGACY)
    checked, found = equation_provenance.checked_document(document, JV, device, implementation)
    assert found == envelopes
    checked["continuation"]["producer_provenance"]["prefix"] = record
    return checked, checked["continuation"].pop("source_comparison")


def test_one_saved_prefix_continues_to_the_same_tail_and_endpoint_under_either_selection(
    jv_case: tuple[CompiledDevice2D, jv_2d.SteadyJV2DInitial], tmp_path: Path,
) -> None:
    """One saved legacy prefix, and then one stopped legacy tail, continued under each selection.

    The results are equal in every binary64 word outside the admitted binding. The saved documents
    and the reader outputs also differ only in it, and in the source comparison against the legacy
    artifact that follows from it.
    """
    device, initial = jv_case
    controls = jv_2d.SteadyJV2DOptions()
    with equation_implementation(LEGACY), wall_stop(2, controls) as on_point:
        prefix = jv_2d.run_steady_jv_2d(device, initial, WAVE, options=controls, on_point=on_point)
    prefix_path, prefix_sha = saved(prefix, tmp_path / "Prefix.json")
    tails, endpoints = {}, {}
    for implementation in (LEGACY, ASSEMBLED):
        with equation_implementation(LEGACY) if implementation == LEGACY else assembled_only(), \
                wall_stop(2, controls) as on_point:
            tails[implementation] = jv_2d.continue_steady_jv_2d(device, prefix_path, prefix_sha, options=controls,
                                                                on_point=on_point)
    tail_path, tail_sha = saved(tails[LEGACY], tmp_path / "Tail.json")
    for implementation in (LEGACY, ASSEMBLED):
        with equation_implementation(LEGACY) if implementation == LEGACY else assembled_only():
            endpoints[implementation] = jv_2d.continue_steady_jv_2d(device, tail_path, tail_sha, options=controls)
    assert len(tails[LEGACY].voltage_V) == 2 and endpoints[LEGACY].completed
    for name, results in (("Tail", tails), ("Endpoint", endpoints)):
        assert words(equation_provenance.checked(results[ASSEMBLED], JV, device, ASSEMBLED)) == words(
            equation_provenance.checked(results[LEGACY], JV, device, LEGACY))
        documents, reads, comparisons = {}, {}, {}
        for implementation, result in results.items():
            path, digest = saved(result, tmp_path / f"{implementation}{name}.json")
            documents[implementation], comparisons[implementation] = continuation_document(
                json.loads(path.read_bytes()), device, implementation, 2)
            reads[implementation], _ = continuation_document(
                _encode(jv_2d.read_steady_jv_2d(path, digest)), device, implementation, 3)
        assert json.dumps(masked(documents[ASSEMBLED])) == json.dumps(masked(documents[LEGACY]))
        assert json.dumps(masked(reads[ASSEMBLED])) == json.dumps(masked(reads[LEGACY]))
        assert comparisons[LEGACY] in {"same", "unknown"}
        assert comparisons[ASSEMBLED] == ("unknown" if comparisons[LEGACY] == "unknown" else "different")
