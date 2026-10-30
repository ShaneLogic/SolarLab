"""The production entry points under the assembled default and the legacy opt-out.

The selection is checked first: assembled by default, the legacy opt-out by
scope or process setting, unknown values refused. Then each 1D production
entry point runs once per implementation on a research example: steady J-V,
DC transient, DC small signal, single- and two-domain ionic AC and the ionic
initialization (the ionic steady and transient drivers are compared field by
field in test_assembly_ionic_drivers). Every result array has the same
binary64 words, every saved artifact has the same bytes outside wall-clock
fields and the AC observations' code identities, and each reader returns the
same record for either producer.

Each job's Q6 binding is asserted first, against the reviewed specification
in support.equation_provenance: the legacy run records the original
operator and no Assembler, the assembled run its operator, the identity of
its system Assembler and an analysis identity over every Assembler it may
evaluate, and the Assemblers that did evaluate are among those. Only that
binding (and the envelope digest over it) is then set aside; the rest of
each Q6 record is compared like any other field.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
import dataclasses
import enum
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any
from unittest import mock

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from examples.research_1d.ionic_impedance import compile_case
from examples.research_1d.ionic_initialization import compile_shared_case
from examples.research_1d.ionic_steady import controls as steady_controls
from examples.research_1d.multiple_ionic_domains import DOMAINS, INTERVALS, prepare_multiple_ionic_device
from examples.research_1d.steady import load_device
from examples.research_1d.transient import controls as transient_controls, json_value
from solarlab.device.compile_1d import compile_device_1d
from solarlab.device.compile_multiple_ionic_1d import compile_multiple_ionic_device_1d
from solarlab.equations.assembly.assembler import Assembler
from solarlab.equations.dc_1d import DC1DProblem
from solarlab.equations.implementation import (
    ASSEMBLED, ENVIRONMENT, LEGACY, equation_implementation, implementation_source_modules,
    selected_implementation, use_assembled,
)
from solarlab.equations.ionic_1d import Ionic1DProblem
from solarlab.equations.multiple_ionic_1d import MultipleIonic1DProblem
from solarlab.experiments import ionic_initialization as initialization
from solarlab.experiments.impedance import solve_impedance
from solarlab.experiments.ionic_impedance import (
    read_ionic_impedance_result, solve_ionic_impedance, write_ionic_impedance_result,
)
from solarlab.experiments.ionic_transient import load_ionic_steady_initial
from solarlab.experiments.jv.steady import run_steady_jv, steady_jv_payload
from solarlab.experiments.multiple_ionic_impedance import (
    read_multiple_ionic_impedance_result, solve_multiple_ionic_impedance, write_multiple_ionic_impedance_result,
)
from solarlab.experiments.multiple_ionic_transient import load_multiple_ionic_steady_initial
from solarlab.experiments.transient import TransientProtocol, solve_transient
from solarlab.experiments.transient_results import _encode, _payload
from solarlab.reproducibility.job_provenance import admitted_source_pins, job_provenance
from solarlab.solvers.nonlinear.ionic_1d import solve_ionic_steady_state
from solarlab.solvers.nonlinear.multiple_ionic_1d import solve_multiple_ionic_steady_state
from solarlab.solvers.nonlinear.steady_1d import SteadyOptions, solve_steady_state
from support import equation_provenance
from support.provenance import checked_provenance_owner

WALL_CLOCK = frozenset({"wall_s"})
# The original AC operators pin their source files in each observation; the
# assembled operators record content identities instead (job-layer Q6).
LEGACY_IDENTITIES = frozenset({"source_sha256"})
ASSEMBLED_IDENTITIES = frozenset({"analysis_identity", "ac_system_identity", "ac_row_operations"})
ORIGINAL_EVALUATIONS = (
    (DC1DProblem, ("residual", "jacobian", "storage_jacobian", "fluxes", "recombination_rate", "_contact_exchange")),
    (Ionic1DProblem, ("residual", "jacobian", "storage_jacobian", "ion_fluxes")),
    (MultipleIonic1DProblem, ("residual", "jacobian", "storage_jacobian", "ion_fluxes", "physical_storage")),
)


@pytest.fixture(scope="module", autouse=True)
def blas1() -> Iterator[None]:
    with threadpool_limits(limits=1):
        yield


@contextmanager
def assembled_only() -> Iterator[list[str]]:
    """The assembled selection with every original evaluation forbidden; the evaluated Assembler identities."""
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("an original evaluation ran under the assembled selection")

    calls: list[str] = []
    original = Assembler.evaluate

    def counted(assembler: Assembler, *args: Any, **kwargs: Any) -> Any:
        calls.append(assembler.identity)
        return original(assembler, *args, **kwargs)

    with ExitStack() as stack:
        for cls, methods in ORIGINAL_EVALUATIONS:
            for method in methods:
                stack.enter_context(mock.patch.object(cls, method, forbidden))
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
    if isinstance(value, (complex, np.complexfloating)):
        return ("complex", np.complex128(value).tobytes())
    if isinstance(value, enum.Enum):
        return ("enum", type(value).__name__, value.value)
    return ("value", value)


def masked(document: Any, identities: frozenset[str] = frozenset()) -> tuple[Any, set[str]]:
    """A saved document without wall-clock fields and the AC observations' named identities (returned)."""
    found: set[str] = set()

    def strip(value: Any, observation: bool = False) -> Any:
        if isinstance(value, dict):
            dropped = WALL_CLOCK | (identities if observation else frozenset())
            found.update(key for key in value if observation and key in identities)
            return {key: strip(item, key == "observation") for key, item in value.items() if key not in dropped}
        if isinstance(value, (list, tuple)):
            return type(value)(strip(item) for item in value)
        return value

    return strip(document), found


Binding = tuple[str, Any, "int | None"]


def bound(document: Any, binding: Binding, implementation: str) -> Any:
    """A saved or read document with every Q6 binding asserted for ``implementation``, then set aside."""
    operator, device, closure = binding
    result, found = equation_provenance.checked_document(document, operator, device, implementation, closure)
    assert found
    return result


def evaluated(calls: list[str], binding: Binding) -> None:
    """The assembled run evaluated its system Assembler and only Assemblers its binding names."""
    operator, device, closure = binding
    assert equation_provenance.expected(operator, device, ASSEMBLED, closure)["identity"] in calls
    assert set(calls) <= equation_provenance.allowed(operator, device, closure)


def same_read(legacy: dict[str, Any], assembled: dict[str, Any], binding: Binding) -> None:
    """Two decoded reader records agree word for word outside wall clock and the observation identities."""
    assert words(masked(bound(assembled, binding, ASSEMBLED), ASSEMBLED_IDENTITIES)[0]) == words(
        masked(bound(legacy, binding, LEGACY), LEGACY_IDENTITIES)[0])


def same_saved(legacy: bytes, assembled: bytes, binding: Binding | None = None, *, ac: bool = False) -> None:
    """The saved bytes agree outside wall clock (and, for AC, the observations' code identities).

    A saved Q6 record's binding is asserted and set aside first (``binding``).
    """
    left_document, right_document = json.loads(legacy), json.loads(assembled)
    if binding is not None:
        left_document, right_document = bound(left_document, binding, LEGACY), bound(right_document, binding, ASSEMBLED)
    left, legacy_found = masked(left_document, LEGACY_IDENTITIES if ac else frozenset())
    right, assembled_found = masked(right_document, ASSEMBLED_IDENTITIES if ac else frozenset())
    # JSON numbers are the shortest round-trip repr, so equal text is equal binary64 words.
    assert json.dumps(right, sort_keys=False) == json.dumps(left, sort_keys=False)
    if ac:
        assert legacy_found == LEGACY_IDENTITIES and assembled_found == ASSEMBLED_IDENTITIES


def test_the_assembled_system_is_the_default_and_legacy_an_explicit_opt_out(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ENVIRONMENT, raising=False)
    assert selected_implementation() == ASSEMBLED and use_assembled()
    with equation_implementation(LEGACY):
        assert selected_implementation() == LEGACY and not use_assembled()
        with equation_implementation(ASSEMBLED):
            assert use_assembled()
        assert not use_assembled()
    assert use_assembled()
    monkeypatch.setenv(ENVIRONMENT, LEGACY)
    assert selected_implementation() == LEGACY
    with equation_implementation(ASSEMBLED):
        assert selected_implementation() == ASSEMBLED
    with pytest.raises(RuntimeError), equation_implementation(ASSEMBLED):
        raise RuntimeError("the scope is restored on errors")
    assert selected_implementation() == LEGACY
    for value in ("", "Assembled", "original", "unified"):
        monkeypatch.setenv(ENVIRONMENT, value)
        with pytest.raises(ValueError, match="unknown equation implementation"):
            selected_implementation()
    with pytest.raises(ValueError, match="unknown equation implementation"), equation_implementation("new"):
        pass


def test_source_maps_cover_both_implementations_and_do_not_depend_on_the_selection() -> None:
    names = implementation_source_modules("small_signal")
    assert names[0] == "solarlab.equations.implementation"
    assert {"solarlab.equations.assembly.classical_1d", "solarlab.equations.assembly.ionic_1d",
            "solarlab.equations.assembly.small_signal"} <= set(names)
    assert set(implementation_source_modules("dc")) < set(implementation_source_modules("ionic")) < set(names)
    # Job admission (Q6) pins the whole source tree, whichever implementation is selected.
    pins = {}
    for implementation in (LEGACY, ASSEMBLED):
        with equation_implementation(implementation), checked_provenance_owner(), \
                job_provenance({"source_map": True}, operator="test.source_map"):
            pins[implementation] = admitted_source_pins()
    assert list(pins[LEGACY].items()) == list(pins[ASSEMBLED].items())
    assert {*names, "solarlab.equations.dc_1d", "solarlab.solvers.time.assembled_storage"} <= set(pins[ASSEMBLED])


@pytest.fixture(scope="module")
def dc_device() -> Any:
    return compile_device_1d(load_device(), intervals_per_layer=12)


def test_steady_jv_arrays_and_saved_payload(dc_device: Any) -> None:
    voltages = np.round(np.arange(0.0, 0.61, 0.15), 2)
    with assembled_only() as calls:
        candidate = run_steady_jv(dc_device, voltages, generation_scale=1.0)
    assert calls
    with equation_implementation(LEGACY):
        reference = run_steady_jv(dc_device, voltages, generation_scale=1.0)
    assert words(candidate) == words(reference)
    (left, left_payloads), (right, right_payloads) = steady_jv_payload(reference), steady_jv_payload(candidate)
    assert json.dumps(_encode(right)) == json.dumps(_encode(left))
    assert [(name, item.data, item.format) for name, item in right_payloads] == [
        (name, item.data, item.format) for name, item in left_payloads]


def test_dc_transient_arrays_and_saved_payload(dc_device: Any) -> None:
    with equation_implementation(LEGACY):
        initial = solve_steady_state(dc_device, voltage_V=0.0, generation_scale=1.0, options=SteadyOptions())
    protocol = TransientProtocol([0, 1e-7, 2e-7, 4e-7], [0, 0.001, 0.002, 0.002], [1, 1.0025, 1.01, 1.01])
    with assembled_only() as calls:
        candidate = solve_transient(dc_device, initial, protocol, options=transient_controls(),
                                    initial_options=SteadyOptions())
    assert calls and candidate.completed
    with equation_implementation(LEGACY):
        reference = solve_transient(dc_device, initial, protocol, options=transient_controls(),
                                    initial_options=SteadyOptions())
    assert words(candidate) == words(reference)
    (left, left_payloads), (right, right_payloads) = (_payload(result, dc_device.nodes_m)
                                                      for result in (reference, candidate))
    assert masked(right)[0] == masked(left)[0]
    assert set(right_payloads) == set(left_payloads)
    for name, item in left_payloads.items():
        if name != "transient_diagnostics":
            assert right_payloads[name].data == item.data, name
    same_saved(left_payloads["transient_diagnostics"].data, right_payloads["transient_diagnostics"].data)


def test_dc_small_signal_arrays(dc_device: Any) -> None:
    with equation_implementation(LEGACY):
        point = solve_steady_state(dc_device, voltage_V=0.6, generation_scale=1.0)
    frequencies = [0.0, 1.0, 1e3, 1e6]
    with assembled_only() as calls:
        candidate = solve_impedance(dc_device, point, frequencies, steady_options=SteadyOptions())
    evaluated(calls, ("DC1DProblem.small_signal", dc_device, None))
    with equation_implementation(LEGACY):
        reference = solve_impedance(dc_device, point, frequencies, steady_options=SteadyOptions())
    assert words(equation_provenance.checked(candidate, "DC1DProblem.small_signal", dc_device, ASSEMBLED)) == words(
        equation_provenance.checked(reference, "DC1DProblem.small_signal", dc_device, LEGACY))
    assert "solarlab.equations.assembly.classical_1d" in candidate.source_sha256


def written(writer: Any, result: Any, path: Path) -> tuple[bytes, str]:
    digest = writer(result, path)
    return path.read_bytes(), digest


def test_single_domain_ionic_ac_saved_results_and_readers(tmp_path: Path) -> None:
    compiled = compile_case("shared")
    controls = steady_controls(compiled.ion_node_count)
    with equation_implementation(LEGACY):
        point = solve_ionic_steady_state(compiled, voltage_V=0.002, generation_scale=0.0, options=controls)
    assert point.accepted
    summary = {"accepted": True, "compiled_identity": compiled.identity, "voltage_V": 0.002, "generation_scale": 0.0}
    raw = (json.dumps(json_value({"summary": summary, "result": point}), allow_nan=False) + "\n").encode()
    (tmp_path / "Steady.json").write_bytes(raw)
    initial = load_ionic_steady_initial(compiled, tmp_path / "Steady.json", hashlib.sha256(raw).hexdigest(),
                                        controls=controls)
    binding: Binding = ("IonicSmallSignal1D", compiled, controls.inventory_row_node)
    for mode in ("mobile", "frozen"):
        with equation_implementation(LEGACY):
            reference = solve_ionic_impedance(compiled, initial, [0.0, 0.01, 1.0, 1e3], ion_mode=mode)
            left, left_sha = written(write_ionic_impedance_result, reference, tmp_path / f"{mode}Legacy.json")
        with assembled_only() as calls:
            candidate = solve_ionic_impedance(compiled, initial, [0.0, 0.01, 1.0, 1e3], ion_mode=mode)
        evaluated(calls, binding)
        right, right_sha = written(write_ionic_impedance_result, candidate, tmp_path / f"{mode}Assembled.json")
        assert reference.completed and candidate.completed
        assert words(candidate.source_sha256) == words(reference.source_sha256)
        for a, b in zip(reference.attempts, candidate.attempts):
            assert words(b.linear_solution) == words(a.linear_solution)
            assert words(b.checks) == words(a.checks)
        same_saved(left, right, binding, ac=True)
        for implementation in (LEGACY, ASSEMBLED):
            with equation_implementation(implementation):
                same_read(read_ionic_impedance_result(tmp_path / f"{mode}Legacy.json", left_sha),
                          read_ionic_impedance_result(tmp_path / f"{mode}Assembled.json", right_sha), binding)


def test_two_domain_ionic_ac_saved_results_and_readers(tmp_path: Path) -> None:
    device = compile_multiple_ionic_device_1d(prepare_multiple_ionic_device(),
                                              intervals_per_layer=INTERVALS, ion_domains=DOMAINS)
    controls = steady_controls(device.domains[0].ion_node_count)
    with equation_implementation(LEGACY):
        point = solve_multiple_ionic_steady_state(device, voltage_V=0.002, generation_scale=0.0, options=controls)
    assert point.accepted
    raw = (json.dumps(_encode({"schema": "solarlab.multiple_ionic_steady.v1", "result": point,
                               "source_sha256": {}, "qualification": False}), indent=2, allow_nan=False) + "\n").encode()
    (tmp_path / "Steady.json").write_bytes(raw)
    initial = load_multiple_ionic_steady_initial(device, tmp_path / "Steady.json", hashlib.sha256(raw).hexdigest(),
                                                 controls=controls)
    frequencies = [0.0, 0.01, 1.0, 1e3]
    with equation_implementation(LEGACY):
        reference = solve_multiple_ionic_impedance(device, initial, frequencies, voltage_phasor_V=2e-6 + 1e-6j)
    with assembled_only() as calls:
        candidate = solve_multiple_ionic_impedance(device, initial, frequencies, voltage_phasor_V=2e-6 + 1e-6j)
    binding: Binding = ("MultipleIonicSmallSignal1D", device, controls.inventory_row_node)
    evaluated(calls, binding)
    assert reference.completed and candidate.completed
    assert list(candidate.source_sha256.items()) == list(reference.source_sha256.items())
    left, left_sha = written(write_multiple_ionic_impedance_result, reference, tmp_path / "Legacy.json")
    right, right_sha = written(write_multiple_ionic_impedance_result, candidate, tmp_path / "Assembled.json")
    same_saved(left, right, binding, ac=True)
    same_read(read_multiple_ionic_impedance_result(tmp_path / "Legacy.json", left_sha),
              read_multiple_ionic_impedance_result(tmp_path / "Assembled.json", right_sha), binding)


def test_ionic_initialization_saved_results_and_reader(tmp_path: Path) -> None:
    compiled = compile_shared_case()
    controls = steady_controls(compiled.ion_node_count)
    with equation_implementation(LEGACY):
        dark = solve_ionic_steady_state(compiled, voltage_V=0.002, generation_scale=0.0, options=controls)
    assert dark.accepted
    summary = {"accepted": True, "compiled_identity": compiled.identity, "voltage_V": 0.002, "generation_scale": 0.0}
    raw = (json.dumps(json_value({"summary": summary, "result": dark}), allow_nan=False) + "\n").encode()
    (tmp_path / "Dark.json").write_bytes(raw)
    initial = load_ionic_steady_initial(compiled, tmp_path / "Dark.json", hashlib.sha256(raw).hexdigest(),
                                        controls=controls)
    options = initialization.IonicInitializationOptions(controls, initialization.DIRECT_POLICY, 25.0)
    binding: Binding = ("Ionic1DProblem.initialization", compiled, controls.inventory_row_node)
    saved = {}
    for implementation in (LEGACY, ASSEMBLED):
        calls: list[str] = []
        with equation_implementation(LEGACY) if implementation == LEGACY else assembled_only() as observed:
            result = initialization.initialize_ionic_steady_state(
                compiled, initial=initial, voltage_V=0.002, generation_scale=1.0, options=options)
            calls = observed or calls
        if implementation == ASSEMBLED:
            evaluated(calls, binding)
        assert result.accepted, result.termination
        path = tmp_path / f"{implementation}.json"
        saved[implementation] = (initialization.write_ionic_initialization(result, path), path)
    same_saved(saved[LEGACY][1].read_bytes(), saved[ASSEMBLED][1].read_bytes(), binding)
    read = {name: initialization.read_ionic_initialization(path, digest) for name, (digest, path) in saved.items()}
    assert masked(bound(_encode(read[ASSEMBLED]), binding, ASSEMBLED))[0] == masked(
        bound(_encode(read[LEGACY]), binding, LEGACY))[0]
    counts = read[ASSEMBLED]["result"]["evaluation_counts"]
    assert counts["density_residual"] == counts["density_jacobian"] == 1


def test_the_process_setting_is_not_a_legacy_engine_override() -> None:
    """The legacy engine refuses research topologies under any active SOLARLAB_* variable."""
    assert not ENVIRONMENT.startswith("SOLARLAB_")
    assert ENVIRONMENT not in {"PYTHONPATH", "PYTHONHOME", "LD_PRELOAD", "DYLD_INSERT_LIBRARIES"}


def test_a_child_process_reads_the_process_selection() -> None:
    code = "from solarlab.equations.implementation import selected_implementation as s; print(s())"
    for value in (None, LEGACY, ASSEMBLED):
        env = {key: item for key, item in os.environ.items() if key != ENVIRONMENT}
        if value is not None:
            env[ENVIRONMENT] = value
        result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == (value or ASSEMBLED)
