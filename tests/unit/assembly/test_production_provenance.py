"""Q6 job provenance names the equation implementation that ran (Science PW-C1, PW-C2).

For every in-process 1D producer family the same job runs under the legacy
opt-out and under the assembled selection. Before any numerical comparison,
each run's admitted binding must equal the reviewed specification
(support.equation_provenance): the legacy run records the original operator
and no Assembler, the assembled run its operator, its system Assembler and an
analysis identity over every Assembler it may evaluate. The original classes
must evaluate only in the legacy run, and only those Assemblers in the
assembled one. The two records differ in that binding alone, in the result,
the saved artifact and the reader output. A stage inside an admitted job is
admitted only under the job's own implementation; every mismatch is refused.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from threadpoolctl import threadpool_limits

from examples.research_1d.ionic_initialization import compile_shared_case
from examples.research_1d.ionic_steady import controls as steady_controls
from examples.research_1d.steady import load_device
from examples.research_1d.transient import controls as transient_controls, json_value
from solarlab.device.compile_1d import compile_device_1d
from solarlab.equations.implementation import (
    ASSEMBLED, ENVIRONMENT, LEGACY, equation_implementation, provenance_binding, selection_setting, substage_binding,
)
from solarlab.experiments import ionic_initialization as initialization
from solarlab.experiments.impedance import solve_impedance
from solarlab.experiments.ionic_transient import load_ionic_steady_initial
from solarlab.experiments.jv.ionic_steady import (
    IonicSteadyJVOptions, read_ionic_steady_jv, run_ionic_steady_jv, write_ionic_steady_jv,
)
from solarlab.experiments.transient import TransientProtocol
from solarlab.experiments.transient_results import run_transient_for_publication
from solarlab.reproducibility.job_provenance import job_provenance
from solarlab.solvers.nonlinear.ionic_1d import solve_ionic_steady_state
from solarlab.solvers.nonlinear.steady_1d import SteadyOptions, solve_steady_state
from support.equation_provenance import allowed, checked_document, counted, expected, split


@pytest.fixture(scope="module", autouse=True)
def blas1() -> Iterator[None]:
    with threadpool_limits(limits=1):
        yield


@pytest.fixture(scope="module")
def dc_device() -> Any:
    return compile_device_1d(load_device(), intervals_per_layer=12)


@pytest.fixture(scope="module")
def ac_cases(tmp_path_factory: pytest.TempPathFactory) -> dict[str, tuple[Any, Any, Any]]:
    """Admitted dark states of the single- and two-domain AC examples."""
    from examples.research_1d.ionic_impedance import compile_case
    from examples.research_1d.multiple_ionic_domains import DOMAINS, INTERVALS, prepare_multiple_ionic_device
    from solarlab.device.compile_multiple_ionic_1d import compile_multiple_ionic_device_1d
    from solarlab.experiments.multiple_ionic_transient import load_multiple_ionic_steady_initial
    from solarlab.experiments.transient_results import _encode
    from solarlab.solvers.nonlinear.multiple_ionic_1d import solve_multiple_ionic_steady_state

    folder = tmp_path_factory.mktemp("ac")
    single = compile_case("shared")
    controls = steady_controls(single.ion_node_count)
    with equation_implementation(LEGACY):
        point = solve_ionic_steady_state(single, voltage_V=0.002, generation_scale=0.0, options=controls)
    summary = {"accepted": True, "compiled_identity": single.identity, "voltage_V": 0.002, "generation_scale": 0.0}
    raw = (json.dumps(json_value({"summary": summary, "result": point}), allow_nan=False) + "\n").encode()
    (folder / "Single.json").write_bytes(raw)
    cases = {"single": (single, controls, load_ionic_steady_initial(
        single, folder / "Single.json", hashlib.sha256(raw).hexdigest(), controls=controls))}
    device = compile_multiple_ionic_device_1d(prepare_multiple_ionic_device(), intervals_per_layer=INTERVALS,
                                              ion_domains=DOMAINS)
    controls = steady_controls(device.domains[0].ion_node_count)
    with equation_implementation(LEGACY):
        point = solve_multiple_ionic_steady_state(device, voltage_V=0.002, generation_scale=0.0, options=controls)
    raw = (json.dumps(_encode({"schema": "solarlab.multiple_ionic_steady.v1", "result": point, "source_sha256": {},
                               "qualification": False}), indent=2, allow_nan=False) + "\n").encode()
    (folder / "Multiple.json").write_bytes(raw)
    cases["multiple"] = (device, controls, load_multiple_ionic_steady_initial(
        device, folder / "Multiple.json", hashlib.sha256(raw).hexdigest(), controls=controls))
    return cases


@pytest.fixture(scope="module")
def ionic_case(tmp_path_factory: pytest.TempPathFactory) -> tuple[Any, Any, Any]:
    compiled = compile_shared_case()
    controls = steady_controls(compiled.ion_node_count)
    with equation_implementation(LEGACY):
        dark = solve_ionic_steady_state(compiled, voltage_V=0.002, generation_scale=0.0, options=controls)
    assert dark.accepted
    summary = {"accepted": True, "compiled_identity": compiled.identity, "voltage_V": 0.002, "generation_scale": 0.0}
    raw = (json.dumps(json_value({"summary": summary, "result": dark}), allow_nan=False) + "\n").encode()
    path = tmp_path_factory.mktemp("ionic") / "Dark.json"
    path.write_bytes(raw)
    initial = load_ionic_steady_initial(compiled, path, hashlib.sha256(raw).hexdigest(), controls=controls)
    return compiled, controls, initial


def pair(run: Callable[[], Any], envelope: Callable[[Any], dict[str, Any]], operator: str, device: Any,
         closure: int | None = None) -> dict[str, Any]:
    """Run once per implementation; assert each binding and evaluation, then that only the binding differs."""
    results, records = {}, {}
    for implementation in (LEGACY, ASSEMBLED):
        with equation_implementation(implementation), counted() as observed:
            results[implementation] = run()
        record, binding = split(envelope(results[implementation]))
        assert binding == expected(operator, device, implementation, closure)
        if implementation == LEGACY:
            assert observed.original > 0 and observed.assembled == 0
        else:
            assert observed.assembled > 0 and observed.original == 0
            assert binding["identity"] in observed.identities <= allowed(operator, device, closure)
        records[implementation] = (record, binding)
    (left, legacy), (right, assembled) = records[LEGACY], records[ASSEMBLED]
    assert legacy["unified_assembler"] is False and assembled["unified_assembler"] is True
    assert legacy["analysis_identity"] != assembled["analysis_identity"] and legacy["operator"] != assembled["operator"]
    assert right == left
    return results


def test_dc_small_signal_admits_the_implementation_that_ran(dc_device: Any) -> None:
    with equation_implementation(LEGACY):
        point = solve_steady_state(dc_device, voltage_V=0.6, generation_scale=1.0)
    pair(lambda: solve_impedance(dc_device, point, [0.0, 1e3], steady_options=SteadyOptions()),
         lambda result: result.job_provenance, "DC1DProblem.small_signal", dc_device)


def test_dc_transient_publication_admits_the_implementation_that_ran(dc_device: Any) -> None:
    with equation_implementation(LEGACY):
        initial = solve_steady_state(dc_device, voltage_V=0.0, generation_scale=1.0, options=SteadyOptions())
    protocol = TransientProtocol([0, 1e-7, 2e-7], [0, 0, 0], [1, 1.0025, 1.01])
    pair(lambda: run_transient_for_publication(dc_device, initial, protocol, options=transient_controls(),
                                               initial_options=SteadyOptions()),
         lambda capture: json.loads(capture.capture_json)["description"]["job_provenance"],
         "BackwardEuler1DStep", dc_device)


def test_ionic_steady_jv_result_saved_artifact_and_reader_admit_the_implementation(
    ionic_case: tuple[Any, Any, Any], tmp_path: Path,
) -> None:
    compiled, controls, _ = ionic_case
    options = IonicSteadyJVOptions(controls)
    results = pair(lambda: run_ionic_steady_jv(compiled, (0.001, 0.002), generation_scale=0.0, options=options,
                                               initial=None, initial_policy="compiled_cold"),
                   lambda result: result.job_provenance, "Ionic1DProblem.steady_jv", compiled,
                   controls.inventory_row_node)
    documents = {}
    for implementation, result in results.items():
        path = tmp_path / f"{implementation}.json"
        digest = write_ionic_steady_jv(result, path)
        saved, found = checked_document(json.loads(path.read_bytes()), "Ionic1DProblem.steady_jv", compiled,
                                        implementation, controls.inventory_row_node)
        assert found == 1
        read = read_ionic_steady_jv(path, digest)["provenance_comparison"]["producer_identity"]
        assert split(read)[1] == expected("Ionic1DProblem.steady_jv", compiled, implementation,
                                          controls.inventory_row_node)
        documents[implementation] = saved
    assert documents[LEGACY]["result"]["job_provenance"] == documents[ASSEMBLED]["result"]["job_provenance"]


def test_standalone_initialization_admits_the_implementation_that_ran(ionic_case: tuple[Any, Any, Any]) -> None:
    compiled, controls, initial = ionic_case
    options = initialization.IonicInitializationOptions(controls, initialization.DIRECT_POLICY, 25.0)
    pair(lambda: initialization.initialize_ionic_steady_state(compiled, initial=initial, voltage_V=0.002,
                                                              generation_scale=1.0, options=options),
         lambda result: result.job_provenance, "Ionic1DProblem.initialization", compiled, controls.inventory_row_node)


@pytest.mark.parametrize("implementation", (LEGACY, ASSEMBLED))
def test_initialization_inside_a_job_is_bound_to_the_job(ionic_case: tuple[Any, Any, Any], implementation: str) -> None:
    compiled, controls, initial = ionic_case
    options = initialization.IonicInitializationOptions(controls, initialization.DIRECT_POLICY, 25.0)
    closure = controls.inventory_row_node
    with equation_implementation(implementation):
        binding = provenance_binding("Ionic1DProblem.steady_jv", compiled, inventory_row_node=closure)
        with job_provenance({"parent": "ionic steady J-V"}, **binding) as parent:
            assert substage_binding(compiled, inventory_row_node=closure) == binding
            result = initialization.initialize_ionic_steady_state(
                compiled, initial=initial, voltage_V=0.002, generation_scale=1.0, options=options)
    assert result.accepted
    assert result.job_provenance == parent.envelope()
    assert split(parent.envelope())[1] == expected("Ionic1DProblem.steady_jv", compiled, implementation, closure)


@pytest.mark.parametrize("parent,stage", ((ASSEMBLED, LEGACY), (LEGACY, ASSEMBLED)))
def test_a_stage_of_the_other_implementation_is_refused(ionic_case: tuple[Any, Any, Any], parent: str,
                                                         stage: str) -> None:
    compiled, controls, initial = ionic_case
    options = initialization.IonicInitializationOptions(controls, initialization.DIRECT_POLICY, 25.0)
    with equation_implementation(parent), job_provenance({"parent": "ionic steady J-V"}, **provenance_binding(
            "Ionic1DProblem.steady_jv", compiled, inventory_row_node=controls.inventory_row_node)):
        with equation_implementation(stage), counted() as observed, \
                pytest.raises(ValueError, match="declared provenance binding differs from the admitted job"):
            initialization.initialize_ionic_steady_state(compiled, initial=initial, voltage_V=0.002,
                                                         generation_scale=1.0, options=options)
    assert observed.original == observed.assembled == 0


def test_a_stage_of_another_closure_or_device_or_an_unknown_job_is_refused(ionic_case: tuple[Any, Any, Any]) -> None:
    compiled, controls, initial = ionic_case
    options = initialization.IonicInitializationOptions(controls, initialization.DIRECT_POLICY, 25.0)
    other = replace(controls, inventory_row_node=controls.inventory_row_node - 1)
    with equation_implementation(ASSEMBLED):
        with job_provenance({"parent": 1}, **provenance_binding(
                "Ionic1DProblem.steady_jv", compiled, inventory_row_node=other.inventory_row_node)), \
                pytest.raises(ValueError, match="declared provenance binding differs"):
            initialization.initialize_ionic_steady_state(compiled, initial=initial, voltage_V=0.002,
                                                         generation_scale=1.0, options=options)
        with job_provenance({"parent": 2}, operator="test.unknown_operator"), \
                pytest.raises(ValueError, match="no equation binding is defined for the admitted operator"):
            initialization.initialize_ionic_steady_state(compiled, initial=initial, voltage_V=0.002,
                                                         generation_scale=1.0, options=options)
    with pytest.raises(ValueError, match="requires an admitted job"):
        substage_binding(compiled, inventory_row_node=controls.inventory_row_node)


def test_a_producer_inside_a_job_of_the_other_implementation_is_refused(dc_device: Any) -> None:
    with equation_implementation(LEGACY):
        point = solve_steady_state(dc_device, voltage_V=0.6, generation_scale=1.0)
        legacy_binding = provenance_binding("DC1DProblem.small_signal", dc_device)
    with job_provenance({"parent": 1}, **legacy_binding), equation_implementation(ASSEMBLED), counted() as observed, \
            pytest.raises(ValueError, match="declared provenance binding differs from the admitted job"):
        solve_impedance(dc_device, point, [1.0], steady_options=SteadyOptions())
    assert observed.assembled == 0


def test_bindings_are_explicit_distinct_and_fail_closed(dc_device: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    bindings = {}
    for implementation in (LEGACY, ASSEMBLED):
        with equation_implementation(implementation):
            bindings[implementation] = provenance_binding("BackwardEuler1DStep", dc_device)
            assert selection_setting() == (ENVIRONMENT, implementation)
    assert bindings[LEGACY]["assembler_identity"] is None and bindings[LEGACY]["analysis_identity"] is not None
    assert bindings[ASSEMBLED]["assembler_identity"] is not None
    assert len({value["analysis_identity"] for value in bindings.values()}) == 2
    with pytest.raises(ValueError, match="no equation binding is defined for operator"):
        provenance_binding("DC2DProblem.unbound", dc_device)
    monkeypatch.setenv(ENVIRONMENT, "Assembled")
    for call in (lambda: provenance_binding("BackwardEuler1DStep", dc_device), selection_setting):
        with pytest.raises(ValueError, match="unknown equation implementation"):
            call()


@pytest.mark.parametrize("implementation", (LEGACY, ASSEMBLED))
def test_every_producer_admits_its_binding_before_evaluating_an_equation(
    dc_device: Any, ionic_case: tuple[Any, Any, Any], ac_cases: dict[str, tuple[Any, Any, Any]], implementation: str,
) -> None:
    from examples.research_1d.ionic_transient import controls as ionic_transient_controls
    from solarlab.experiments.ionic_impedance import solve_ionic_impedance
    from solarlab.experiments.multiple_ionic_impedance import solve_multiple_ionic_impedance
    from solarlab.experiments.multiple_ionic_transient import solve_multiple_ionic_transient
    from solarlab.reproducibility import job_provenance as provenance

    compiled, controls, initial = ionic_case
    single, _, single_initial = ac_cases["single"]
    multiple, _, multiple_initial = ac_cases["multiple"]
    with equation_implementation(LEGACY):
        ac_point = solve_steady_state(dc_device, voltage_V=0.6, generation_scale=1.0)
        dc_initial = solve_steady_state(dc_device, voltage_V=0.0, generation_scale=1.0, options=SteadyOptions())
    producers = {
        "DC1DProblem.small_signal": lambda: solve_impedance(dc_device, ac_point, [1.0], steady_options=SteadyOptions()),
        "BackwardEuler1DStep": lambda: run_transient_for_publication(
            dc_device, dc_initial, TransientProtocol([0, 1e-7], [0, 0], [1, 1.0025]),
            options=transient_controls(), initial_options=SteadyOptions()),
        "Ionic1DProblem.steady_jv": lambda: run_ionic_steady_jv(
            compiled, (0.001,), generation_scale=0.0, options=IonicSteadyJVOptions(controls),
            initial=None, initial_policy="compiled_cold"),
        "Ionic1DProblem.initialization": lambda: initialization.initialize_ionic_steady_state(
            compiled, initial=initial, voltage_V=0.002, generation_scale=1.0,
            options=initialization.IonicInitializationOptions(controls, initialization.DIRECT_POLICY, 25.0)),
        "IonicSmallSignal1D": lambda: solve_ionic_impedance(single, single_initial, [1.0]),
        "MultipleIonicSmallSignal1D": lambda: solve_multiple_ionic_impedance(
            multiple, multiple_initial, [1.0], voltage_phasor_V=2e-6 + 1e-6j),
        "MultipleIonicBackwardEuler1DStep": lambda: solve_multiple_ionic_transient(
            multiple, multiple_initial, TransientProtocol([0, 1e-3], [0.002, 0.002], [0, 0]),
            options=ionic_transient_controls()),
    }
    capture = provenance.capture_job_provenance
    for operator, run in producers.items():
        at_capture: list[tuple[int, int]] = []
        with equation_implementation(implementation), counted() as observed:
            def admitted(*args: Any, **kwargs: Any) -> Any:
                at_capture.append((observed.original, observed.assembled))
                return capture(*args, **kwargs)
            with pytest.MonkeyPatch.context() as patch:
                patch.setattr(provenance, "capture_job_provenance", admitted)
                run()
        assert at_capture == [(0, 0)], operator
        assert observed.original + observed.assembled > 0, operator


def test_device_compilation_evaluates_no_equation() -> None:
    """Workers compile before admission, so that they can name the bound system."""
    from examples.research_1d.shared_site_ionic_fixture import prepare_confined_shared_site_ion_device
    from solarlab.device.compile_ionic_1d import compile_ionic_device_1d

    with counted() as observed:
        compile_device_1d(load_device(), intervals_per_layer=12)
        compile_ionic_device_1d(prepare_confined_shared_site_ion_device(), intervals_per_layer=(12, 12, 12),
                                ion_domain_layer="absorber", ion_domain_endpoints=("blocking", "blocking"))
    assert observed.original == observed.assembled == 0


def test_2d_jv_admits_the_implementation_that_ran(tmp_path: Path) -> None:
    import numpy as np

    from examples.research_2d import dc_assembly, steady as steady_example
    from solarlab.device.compile_2d import compile_device_2d
    from solarlab.experiments.jv import steady_2d as jv_2d

    device = compile_device_2d(steady_example.load_device(), dc_assembly.spatial_input())
    with equation_implementation(LEGACY):
        assert steady_example.run_case(tmp_path / "Steady.json")["completed"]
    path = tmp_path / "Steady.json"
    controls = jv_2d.SteadyJV2DOptions()
    initial = jv_2d.load_steady_jv_2d_initial(device, path, hashlib.sha256(path.read_bytes()).hexdigest(),
                                              options=controls.steady)
    runs = iter(("legacy", "assembled"))
    pair(lambda: jv_2d.run_steady_jv_2d(device, initial, np.array([0.0]), options=controls,
                                        checkpoint_directory=tmp_path / next(runs)),
         lambda result: result.job_provenance, "DC2DProblem.steady_jv", device)
