"""Manufactured BE/reference histories only; no real B, native IDA or old gzip.

The two-state storage problem and its flux integral are analytic and independent
of the public stepper. Frozen protocol time words are data, not device inputs.
"""

from copy import deepcopy
from dataclasses import asdict, replace
from fractions import Fraction
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

import scripts.benchmarks.g2_reference_be as be
from scripts.benchmarks.conservative_device import EquationBudget
from scripts.benchmarks.contract_prototype import (
    PARTICLE,
    SECOND,
    VOLT,
    ContractError,
    EquationSpec,
    Layout,
    PhysicalScaling,
    PhysicalStorage,
    SparseStructure,
    StateView,
    Support,
    ValidatedProblem,
    VariableSpec,
)
from scripts.benchmarks.coupled_device_prototype import ProtocolSegment
from scripts.benchmarks.precision_prototype import (
    DoubleArithmetic,
    DoubleArray,
    FrameInputExpansion,
    RelativeCoordinates,
    encode_point,
    physical_state_words,
)


def exact(point, name):
    arrays = physical_state_words(point.state)[name]
    return tuple(
        sum((Fraction(float(w[i])) for w in arrays), Fraction())
        for i in range(arrays[0].size)
    )


def enclosure(value, radius=0):
    value, radius = Fraction(value), Fraction(radius)
    return {
        "center": [value.numerator, value.denominator],
        "radius": [radius.numerator, radius.denominator],
    }


def tags_for(segments, times):
    return [
        {
            "segment": s.id,
            "index": i,
            "time_hex": float(t).hex(),
            "event_side": "right"
            if i == 0
            else "left"
            if i == len(group) - 1
            else "interior",
            "protocol_initial": k == i == 0,
            "protocol_terminal": k == len(segments) - 1 and i == len(group) - 1,
        }
        for k, (s, group) in enumerate(zip(segments, times))
        for i, t in enumerate(group)
    ]


def manufactured(kind="tiny", *, segments=None, scientific_at=None, safety_at=None):
    """Q=(n0,n1), phi=n0+2*n1+V, with exact force/decay/driven RHS."""
    if segments is None:
        segments = (
            ProtocolSegment("a", 0.0, 0.25, (0.0, 0.0), (0.0, 0.0)),
            ProtocolSegment("b", 0.25, 0.5, (0.0, 0.25), (0.0, 0.0)),
            ProtocolSegment("c", 0.5, 1.0, (0.25, 0.25), (0.0, 0.0)),
        )
    supports = (Support("cells", "cell", (2,)), Support("port", "global", (1,)))
    variables = (
        VariableSpec("n", "manufactured", "cells", (2,), PARTICLE, lower=0),
        VariableSpec("phi", "manufactured", "port", (1,), VOLT, role="constraint"),
    )
    equations = (
        EquationSpec("balance", "manufactured", "cells", (2,), PARTICLE / SECOND),
        EquationSpec(
            "constraint", "manufactured", "port", (1,), VOLT, role="constraint"
        ),
    )
    layout = Layout(supports, variables, equations)
    coordinates = RelativeCoordinates(layout, {"n": "linear", "phi": "linear"})
    inputs, _ = segments[0].inputs(segments[0].start)
    initial = coordinates.initial(
        StateView(
            layout,
            (("n", DoubleArray([3.0, 3.0])), ("phi", DoubleArray([9.0 + inputs[0]]))),
        ),
        segments[0].start,
        inputs,
    )
    source = sha256(repr((kind, segments)).encode()).hexdigest()
    arithmetic = DoubleArithmetic()
    structure = SparseStructure(
        (3, 3), np.tile(np.arange(3), 3), np.arange(4) * 3, source, layout.identity
    )
    storage = PhysicalStorage(
        lambda p: DoubleArray.from_dd(p.state.field("n").as_dd()),
        lambda a, b, dx: DoubleArray.from_dd(dx.field("n").as_dd()),
        (PARTICLE,) * 2,
        source,
        arithmetic,
    )

    def residual(left, right, increment):
        n = right.state.field("n").as_dd()
        if kind == "tiny":
            rate = arithmetic.array([2.0**-80, -(2.0**-80)])
        elif kind == "decay":
            rate = -n
        else:
            rate = arithmetic.array([2 * right.time, -right.time])
        rows = (
            storage.delta(left, right, increment).as_dd()
            - (right.time - left.time) * rate
        )
        constraint = right.state.field("phi").as_dd() - (
            n[0] + 2 * n[1] + right.inputs[0]
        )
        return arithmetic.freeze(arithmetic.concatenate((rows, constraint)))

    def jacobian(left, right):
        matrix = np.eye(3)
        if kind == "decay":
            matrix[:2, :2] *= 1 + right.time - left.time
        matrix[2] = [-1.0, -2.0, 1.0]
        return structure.filled(matrix.T.ravel())

    def forbidden(*args):
        raise AssertionError("Manufactured BE must not invoke native/tangent F/J")

    problem = ValidatedProblem(
        layout,
        2,
        2,
        storage,
        structure,
        source,
        forbidden,
        forbidden,
        residual,
        jacobian,
        arithmetic=arithmetic,
    )
    budget = 2.0**-140 if kind == "tiny" else 1e-24
    budgets = {
        "balance": EquationBudget(PARTICLE, [budget, budget]),
        "constraint": EquationBudget(VOLT, [budget]),
    }

    def checks(point):
        return {
            "safety_pass": safety_at is None or point.time < safety_at,
            "scientific_failed": scientific_at is not None
            and point.time >= scientific_at,
        }

    def rates(point):
        if kind == "tiny":
            return Fraction(1, 2**80), -Fraction(1, 2**80)
        if kind == "decay":
            return tuple(-x for x in exact(point, "n"))
        return 2 * Fraction(point.time), -Fraction(point.time)

    def observe(point, segment):
        n0, n1 = exact(point, "n")
        a, b = rates(point)
        _, rate = segment.inputs(point.time)
        left = a + 2 * b + Fraction(float(rate[0]))
        charge = exact(point, "phi")[0]
        return {
            "ports": {
                "left": enclosure(left),
                "right": enclosure(-a - b - left),
                "unit": "A",
            },
            "charges": {
                "device": enclosure(n0 + n1),
                "left_metal": enclosure(charge),
                "right_metal": enclosure(-n0 - n1 - charge),
                "unit": "C",
            },
            "observation_uncertainty": "exact rational manufactured oracle",
            "derivative_origin": "manufactured_analytic_tangent",
            "evidence_status": "manufactured",
        }

    def interval(left, right, step, segment, target):
        # Independent integration of the analytic physical RHS on the represented
        # linear path. Using step.storage_delta here would make this test circular.
        h = Fraction(right.time) - Fraction(left.time)
        if kind == "tiny":
            a, b = h / 2**80, -h / 2**80
        elif kind == "decay":
            a, b = tuple(
                -h * (x + y) / 2 for x, y in zip(exact(left, "n"), exact(right, "n"))
            )
        else:
            a = Fraction(right.time) ** 2 - Fraction(left.time) ** 2
            b = -a / 2
        delta_v = Fraction(float(right.inputs[0])) - Fraction(float(left.inputs[0]))
        flux_left = a + 2 * b + delta_v
        return {
            "physical_flux": [
                enclosure(a + b),
                enclosure(flux_left),
                enclosure(-a - b - flux_left),
            ],
            "unit": "C",
            "storage_secant_used_as_flux": False,
            "origin": "manufactured_exact_integral",
            "reference_error_status": "unknown_reference_error",
        }

    return be.Backend(
        problem,
        coordinates,
        initial,
        PhysicalScaling(np.ones(3), np.ones(3)),
        budgets,
        segments,
        lambda p: min(exact(p, "n")) > 0,
        checks,
        observe,
        interval,
        {
            "kind": "manufactured",
            "force": kind,
            "observation_controls": {"kind": "exact_rational_manufactured"},
        },
    )


def execute(
    tmp_path,
    *,
    backend=None,
    schedule=None,
    encoding="raw",
    checkpoint=None,
    limits=None,
    plan_sha256="2" * 64,
    timing=None,
):
    backend = backend or manufactured()
    schedule = schedule or be.compile_schedule(
        backend.segments,
        tags_for(backend.segments, [[s.start, s.end] for s in backend.segments]),
        level=0,
        seed=Fraction(1, 8),
    )
    request = {
        "kind": "manufactured",
        "case": backend.metadata,
        "segments": [asdict(s) for s in backend.segments],
    }
    identity = be.RunIdentity(
        "manufactured",
        be.digest(request),
        "1" * 64,
        plan_sha256,
        "3" * 40,
        "manufactured-test-environment",
        schedule.identity,
        backend.problem.source_identity,
        backend.initial.identity,
        be.digest(be.effective_controls(backend)),
    )
    limits = limits or be.Limits(
        60.0, output_bytes=64 * 1048576, logical_bytes=64 * 1048576
    )
    result = be.run_reference(
        backend,
        schedule,
        identity,
        request,
        tmp_path,
        limits,
        encoding=encoding,
        checkpoint=checkpoint,
        timing=timing,
    )
    readback = be.read_reference_history(tmp_path, backend.problem.layout, identity)
    return backend, schedule, identity, result, readback


def raw_rows(directory):
    return [
        json.loads(line)
        for line in (directory / "History/BE.jsonl").read_text().splitlines()
    ]


def reseal(directory, rows):
    """Adversary recomputes transport hashes; semantic checks must still reject."""
    previous = None
    for row in rows:
        row["previous_record_sha256"] = previous
        row.pop("record_sha256", None)
        row["record_sha256"] = previous = be.digest(row)
    raw = b"".join(be.canonical(row) + b"\n" for row in rows)
    (directory / "History/BE.jsonl").write_bytes(raw)
    path = directory / "RunResult.json"
    result = json.loads(path.read_text())
    result["observed_history"].update(
        records=len(rows),
        logical_sha256=sha256(raw).hexdigest(),
        logical_bytes=len(raw),
        encoded_bytes=len(raw),
    )
    path.write_bytes(be.canonical(result) + b"\n")


@pytest.mark.parametrize("encoding", ["raw", "gzip"])
def test_complete_low_word_history_and_event_tangents(tmp_path, encoding):
    backend, schedule, identity, result, readback = execute(
        tmp_path / "run", encoding=encoding
    )
    assert result["numerical_complete"] and readback["structural_pass"]
    assert result["scientific_status"] == "unknown_reference_error"
    assert readback["scientific_pass"] is False
    samples = readback["observations"]
    assert [s["tag"] for s in samples] == list(schedule.tags)
    assert samples[1]["point_identity"] == samples[2]["point_identity"]
    assert samples[1]["ports"]["left"] != samples[2]["ports"]["left"]
    words = samples[-1]["physical_state_words"]["n"]["words_hex"]
    final = tuple(
        sum((Fraction(float.fromhex(w[i])) for w in words), Fraction())
        for i in range(2)
    )
    assert final == (3 + Fraction(1, 2**80), 3 - Fraction(1, 2**80))
    assert [float.fromhex(x) for x in words[0]] == [3.0, 3.0]
    assert not (tmp_path / "run/History/LastAccepted.json").exists()
    pointer = json.loads(
        (tmp_path / "run/History/LastNumericalAccepted.json").read_text()
    )
    assert (
        pointer["step_index"] == len(schedule.steps)
        and not pointer["physical_acceptance_awarded"]
    )
    assert identity.initial_point_identity == backend.initial.identity


def test_full_affine_words_survive_codec_and_step(tmp_path):
    backend = manufactured(
        segments=(ProtocolSegment("a", 0.0, 0.5, (0.0, 0.0), (0.0, 0.0)),)
    )
    words = [np.zeros(3) for _ in range(12)]
    for i, word in enumerate(words):
        word[:] = (
            2.0 ** (-20 - 60 * i),
            2.0 ** (-20 - 60 * i),
            3 * 2.0 ** (-20 - 60 * i),
        )
    point, _ = backend.coordinates.trial(
        backend.initial, FrameInputExpansion(words), 0.0, [0.0, 0.0]
    )
    reference = encode_point(backend.initial)
    packed = be.pack_point(point, reference)
    restored = be.unpack_point(packed, reference, backend.problem.layout)
    assert restored.identity == point.identity
    assert exact(restored, "n") == exact(point, "n")
    _, _, _, _, readback = execute(
        tmp_path / "words", backend=replace(backend, initial=point)
    )
    saved = readback["observations"][-1]["physical_state_words"]["n"]["words_hex"]
    wanted = (
        exact(point, "n")[0] + Fraction(1, 2**81),
        exact(point, "n")[1] - Fraction(1, 2**81),
    )
    assert (
        tuple(
            sum((Fraction(float.fromhex(w[i])) for w in saved), Fraction())
            for i in range(2)
        )
        == wanted
    )
    assert any(float.fromhex(v) == 2.0**-680 for w in saved for v in w)


def test_driven_analytic_global_error_is_sum_h_squared(tmp_path):
    backend = manufactured("driven")
    _, schedule, _, _, readback = execute(tmp_path / "driven", backend=backend)
    sample = readback["observations"][-1]
    words = sample["physical_state_words"]["n"]["words_hex"]
    values = tuple(
        sum((Fraction(float.fromhex(w[i])) for w in words), Fraction())
        for i in range(2)
    )
    accumulated = sum(Fraction(s.h) ** 2 for s in schedule.steps)
    assert values == (4 + accumulated, Fraction(5, 2) - accumulated / 2)
    # The independent physical-flux integral is analytic; it is not BE's storage delta.
    assert sample["cumulative_physical_flux"][0] == enclosure(Fraction(1, 2))
    assert values[0] + values[1] - 6 != Fraction(1, 2)


def test_decay_three_levels_converge_with_first_order(tmp_path):
    backend = manufactured(
        "decay", segments=(ProtocolSegment("a", 0.0, 1.0, (0.0, 0.0), (0.0, 0.0)),)
    )
    values = []
    for level in range(3):
        schedule = be.compile_schedule(
            backend.segments,
            tags_for(backend.segments, [[0.0, 1.0]]),
            level=level,
            seed=Fraction(1, 16),
        )
        _, _, _, _, readback = execute(
            tmp_path / str(level), backend=backend, schedule=schedule
        )
        words = readback["observations"][-1]["physical_state_words"]["n"]["words_hex"]
        value = sum((Fraction(float.fromhex(w[0])) for w in words), Fraction())
        analytic_be = Fraction(3) * (
            Fraction(1) / (1 + Fraction(1, 16 * 2**level))
        ) ** (16 * 2**level)
        assert abs(value - analytic_be) < Fraction(1, 10**22)
        values.append(value)
    assert (
        be.order_check((values[0] - values[1], 0), (values[1] - values[2], 0))["status"]
        == "passed"
    )
    assert abs(float(values[2]) - 3 * math.exp(-1)) < abs(
        float(values[1]) - 3 * math.exp(-1)
    )


def test_latched_scientific_failure_keeps_full_safe_suffix(tmp_path):
    backend = manufactured(scientific_at=0.25)
    _, schedule, _, result, readback = execute(tmp_path / "latched", backend=backend)
    assert result["numerical_complete"] and result["scientific_status"] == "failed"
    assert readback["accepted_steps"] == len(schedule.steps)
    assert readback["observations"][-1]["tag"]["protocol_terminal"]
    assert result["first_scientific_failure"]["time_hex"] == (0.25).hex()


def test_safety_failure_stops_and_keeps_first_failure(tmp_path):
    backend = manufactured(safety_at=0.25)
    directory = tmp_path / "safety"
    with pytest.raises(ContractError, match="accepted_state_safety"):
        execute(directory, backend=backend)
    result = json.loads((directory / "RunResult.json").read_text())
    identity = be.RunIdentity(
        **{k: result[k] for k in be.RunIdentity.__dataclass_fields__}
    )
    readback = be.read_reference_history(directory, backend.problem.layout, identity)
    assert not readback["numerical_complete"] and readback["accepted_steps"] == 2
    failure = json.loads((directory / "FirstFailure.json").read_text())
    assert failure["reason"] == "reference_be_accepted_state_safety"
    assert failure["last_returned_step_persisted"]


def test_failed_newton_keeps_working_iterate_without_acceptance(tmp_path):
    backend = manufactured()
    # The first initial point is allowed; every trial is rejected by feasibility.
    backend = replace(backend, feasible=lambda p: p.time == 0.0)
    directory = tmp_path / "newton"
    with pytest.raises(ContractError):
        execute(directory, backend=backend)
    failure = json.loads((directory / "FirstFailure.json").read_text())
    assert failure["failed_newton_diagnostics_available"]
    assert "failed_newton_working_iterates" in failure
    assert failure["counts"]["accepted_steps"] == 0
    assert not (directory / "History/LastNumericalAccepted.json").exists()


@pytest.mark.parametrize(
    "mutation,reason",
    [
        ("order", "step_sequence"),
        ("counter", "step_sequence"),
        ("unit", "unaccepted_equation_residual"),
        ("ratio", "unaccepted_equation_residual"),
        ("word_shape", "word_size"),
        ("tag", "observation_tag"),
        ("prefix", "sample_flux_prefix"),
        ("latch", "scientific_latch"),
        ("event_rate", "event_input_rate"),
        ("event_generation", "event_history"),
        ("controls", "controls_identity"),
    ],
)
def test_resealed_semantic_corruption_is_rejected(tmp_path, mutation, reason):
    directory = tmp_path / mutation
    backend, _, identity, _, _ = execute(directory)
    rows = raw_rows(directory)
    step = next(r for r in rows if r["kind"] == "be_accepted_step")
    observation = next(
        r for r in rows if r["kind"] == "be_observation" and r["tag"] is not None
    )
    event = next(r for r in rows if r["kind"] == "be_segment_start")
    if mutation == "order":
        step["order"] = 2
    elif mutation == "counter":
        step["step_index"] = True
    elif mutation == "unit":
        step["residual_units"][0] = asdict(VOLT)
    elif mutation == "ratio":
        step["residual_ratios"][0] = [1, 2]
    elif mutation == "word_shape":
        step["local_increment_words"]["shape"] = [2]
    elif mutation == "tag":
        observation["tag"]["event_side"] = "left"
    elif mutation == "prefix":
        observation["cumulative_physical_flux"][0]["center"] = [1, 1]
    elif mutation == "latch":
        step["scientific_failure_latched"] = True
    elif mutation == "event_rate":
        event["input_rate_hex"][0] = (1.0).hex()
    elif mutation == "event_generation":
        event["generation"] = True
    elif mutation == "controls":
        rows[0]["solver_scaling"]["rows"][0] = (2.0).hex()
    reseal(directory, rows)
    with pytest.raises(ContractError, match=reason):
        be.read_reference_history(directory, backend.problem.layout, identity)


def test_empty_resealed_failed_history_cannot_pass(tmp_path):
    directory = tmp_path / "empty"
    backend, _, identity, _, _ = execute(directory)
    reseal(directory, [])
    path = directory / "RunResult.json"
    result = json.loads(path.read_text())
    result.update(numerical_complete=False, status="failed", scientific_status="failed")
    result["counts"]["accepted_steps"] = 0
    path.write_bytes(be.canonical(result))
    with pytest.raises(ContractError, match="history_seal"):
        be.read_reference_history(directory, backend.problem.layout, identity)


def test_resource_caps_and_cancellation_are_explicit(tmp_path, monkeypatch):
    with pytest.raises(ContractError, match="resource_limit"):
        be.Limits(60.0, rss_bytes=4 * be.GIB + 1).validate()
    monkeypatch.setattr(
        be.resource, "getrusage", lambda who: SimpleNamespace(ru_maxrss=5 * be.GIB)
    )
    with pytest.raises(ContractError, match="RSS_cap"):
        be.Guard(be.Limits(60.0))()
    monkeypatch.undo()
    counter = 0

    def cancel():
        nonlocal counter
        counter += 1
        if counter == 12:
            raise RuntimeError("manufactured_cancelled")

    directory = tmp_path / "cancel"
    with pytest.raises(RuntimeError, match="manufactured_cancelled"):
        execute(directory, checkpoint=cancel)
    failure = json.loads((directory / "FirstFailure.json").read_text())
    assert failure["reason"] == "manufactured_cancelled"
    assert not json.loads((directory / "RunResult.json").read_text())[
        "numerical_complete"
    ]


def test_preexisting_directory_and_changed_schedule_refused(tmp_path):
    directory = tmp_path / "existing"
    directory.mkdir()
    (directory / "user.txt").write_text("keep")
    with pytest.raises(FileExistsError):
        execute(directory)
    assert (directory / "user.txt").read_text() == "keep"
    backend = manufactured()
    schedule = be.compile_schedule(
        backend.segments,
        tags_for(backend.segments, [[s.start, s.end] for s in backend.segments]),
        level=0,
        seed=Fraction(1, 8),
    )
    schedule = replace(
        schedule, steps=(replace(schedule.steps[0], h=0.5),) + schedule.steps[1:]
    )
    with pytest.raises(ContractError, match="runtime_schedule_binding"):
        execute(tmp_path / "changed", backend=backend, schedule=schedule)


def test_order_uncertainty_and_non_time_propagation_are_not_ignored():
    assert (
        be.order_check((2, Fraction(4, 5)), (1, Fraction(2, 5)))["status"] == "unknown"
    )
    assert be.order_check((2, 0), (-1, 0))["status"] == "failed"
    assert be.order_check((0, 0), (0, 0))["reason"] == "unresolved_difference"
    assert be.order_check((4, 0), (1, 0))["status"] == "failed"
    terms = [
        {"G": 2, "R": Fraction(1, 10**12), "U": 0, "evidence_identity": "analytic"}
    ] * 20
    result = be.propagate_error(0, terms)
    assert result[-1] == Fraction(2**20 - 1, 10**12)
    assert result[-1] > 20 * Fraction(1, 10**12)
    with pytest.raises(ContractError, match="missing_propagation_evidence"):
        be.propagate_error(0, [{"G": 1, "R": 0, "U": 0}])


def test_unadmitted_and_unfrozen_B_fail_before_model_construction(
    tmp_path, monkeypatch
):
    import scripts.benchmarks.coupled_device_prototype as coupled

    called = []

    def forbidden(*args, **kwargs):
        called.append(True)
        raise AssertionError("real B constructor forbidden in author tests")

    monkeypatch.setattr(coupled, "AffineCoupledSlab", forbidden)
    with pytest.raises(ContractError, match="B_not_admitted"):
        be.prepare_B_backend({}, {})
    contract = {
        "schema": be.METHOD_SCHEMA,
        "status": "draft",
        "authority": {"freeze_record": None},
    }
    raw = be.canonical(contract)
    path = tmp_path / "method.json"
    path.write_bytes(raw)
    manifest = {
        "schema": "solarlab.g2-reference-be-run.v1",
        "method_contract": {"path": str(path), "sha256": sha256(raw).hexdigest()},
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_bytes(be.canonical(manifest))
    with pytest.raises(ContractError, match="method_unfrozen"):
        be.main(["run", "--manifest", str(manifest_path)])
    assert called == []


def frozen_protocol():
    segments = (
        ProtocolSegment("dark_equilibrium_hold", 0.0, 0.1, (0.0, 0.0), (0.0, 0.0)),
        ProtocolSegment("voltage_ramp", 0.1, 0.2, (0.0, 0.1), (0.0, 0.0)),
        ProtocolSegment("slow_state_hold", 0.2, 9.2, (0.1, 0.1), (0.0, 0.0)),
    )
    times = [
        [float.fromhex(t) for t in words.split()] for words in FROZEN_PROTOCOL_TIMES
    ]
    return segments, tags_for(segments, times)


def test_frozen_time_words_counts_nesting_and_exact_binary_endpoints():
    segments, tags = frozen_protocol()
    assert len(tags) == 131 and len({x["time_hex"] for x in tags}) == 129
    previous = None
    for level in range(4):
        schedule = be.compile_schedule(segments, tags, level=level)
        counts = [sum(s.segment_index == i for s in schedule.steps) for i in range(3)]
        assert counts == [64 * 2**level, 64 * 2**level, 7845 * 2**level]
        assert len(schedule.steps) == 7973 * 2**level
        nodes = {s.right.hex() for s in schedule.steps}
        if previous is not None:
            assert previous <= nodes
        previous = nodes
        assert all(
            Fraction(s.right) - Fraction(s.left) == Fraction(s.h)
            and s.left + s.h == s.right
            for s in schedule.steps
        )
        assert all(s.right <= segments[s.segment_index].end for s in schedule.steps)
    probe = be.compile_schedule(segments[:2], tags[:66], level=0)
    assert len(probe.tags) == 66 and len(probe.steps) == 128
    assert probe.tags[-1]["time_hex"] == (0.2).hex()
    assert probe.tags[-1]["protocol_terminal"] is False


def test_original_request_tags_are_preserved_for_full_and_probe():
    segments, tags = frozen_protocol()
    request = {
        "segments": [asdict(s) for s in segments],
        "observation_times": {
            s.id: [float.fromhex(t["time_hex"]) for t in tags if t["segment"] == s.id]
            for s in segments
        },
    }
    assert be.tags_from_request(request, 3) == tags
    assert be.tags_from_request(request, 2) == tags[:66]
    request["observation_times"][segments[0].id][1] = True
    with pytest.raises(ContractError, match="request_sample_times"):
        be.tags_from_request(request, 3)


def test_E3_diagnostics_and_reader_budget_export(tmp_path):
    directory = tmp_path / "e3"
    backend, _, identity, _, _ = execute(directory)
    rows = raw_rows(directory)
    assert rows[0]["step_options"] == {
        "max_iterations": 12,
        "max_line_search": 8,
        "globalization": "residual",
    }
    step = next(r for r in rows if r["kind"] == "be_accepted_step")
    assert step["newton_iterations"]
    assert len(step["newton_iterations"][0]["rhs_projection_error"]) == 3
    assert len(step["residual_allowances"]) == 3
    with pytest.raises(ContractError, match="reader_limits"):
        be.read_reference_history(
            directory, backend.problem.layout, identity, max_record_bytes=1048577
        )

    def cancelled():
        raise RuntimeError("reader_cancelled")

    with pytest.raises(RuntimeError, match="reader_cancelled"):
        be.read_reference_history(
            directory, backend.problem.layout, identity, checkpoint=cancelled
        )


def test_131_tag_history_using_only_manufactured_physics(tmp_path):
    segments, tags = frozen_protocol()
    backend = manufactured(segments=segments)
    # The schedule/record test uses a coarse manufactured ladder, not a B run.
    schedule = be.compile_schedule(segments, tags, level=0, seed=Fraction(1))
    _, _, _, _, readback = execute(tmp_path / "131", backend=backend, schedule=schedule)
    assert readback["tagged_observations"] == 131
    assert [r["tag"] for r in readback["observations"]] == tags


def test_returned_step_before_observer_failure_is_not_claimed_persisted(tmp_path):
    backend = manufactured()
    original = backend.check_state

    def failing(point):
        if point.time > 0:
            raise RuntimeError("manufactured_state_check_failure")
        return original(point)

    directory = tmp_path / "returned"
    with pytest.raises(RuntimeError, match="manufactured_state_check_failure"):
        execute(directory, backend=replace(backend, check_state=failing))
    failure = json.loads((directory / "FirstFailure.json").read_text())
    assert failure["last_returned_step"] == 1
    assert failure["last_returned_step_persisted"] is False
    assert failure["counts"]["accepted_steps"] == 0


@pytest.mark.parametrize(
    "damage,reason",
    [
        ("wrong_step_input", "step_protocol_inputs"),
        ("local_increment", "step_local_increment"),
        ("storage_secant", "storage_secant_division"),
        ("missing_newton_diagnostics", "newton_diagnostics"),
    ],
)
def test_science_four_resealed_counterexamples(tmp_path, damage, reason):
    directory = tmp_path / damage
    backend = manufactured(
        segments=(ProtocolSegment("one", 0.0, 0.125, (0.0, 0.0), (0.0, 0.0)),)
    )
    _, _, identity, _, _ = execute(directory, backend=backend)
    rows = raw_rows(directory)
    step = next(r for r in rows if r["kind"] == "be_accepted_step")
    if damage == "wrong_step_input":
        root = rows[0]["original_reference"]
        right = be.unpack_point(step["right"], root, backend.problem.layout)
        altered, _ = backend.coordinates.trial(
            right, np.zeros(3), right.time, [0.25, 0.0], predecessor=right
        )
        assert exact(altered, "n") == exact(right, "n")
        step["right"] = be.pack_point(altered, root)
        for row in rows:
            if row.get("point_identity") == right.identity:
                row["point_identity"] = altered.identity
                if "physical_state_words" in row:
                    row["physical_state_words"] = be._physical(altered)
            if row.get("last_point_identity") == right.identity:
                row["last_point_identity"] = altered.identity
        path = directory / "RunResult.json"
        result = json.loads(path.read_text())
        result["last_numerical_accepted"]["identity"] = altered.identity
        path.write_bytes(be.canonical(result))
    elif damage == "missing_newton_diagnostics":
        assert step.pop("newton_iterations")
    else:
        field = (
            "local_increment_words"
            if damage == "local_increment"
            else "storage_secant_words"
        )
        step[field]["words_hex"][0][0] = (1.0).hex()
    reseal(directory, rows)
    with pytest.raises(ContractError, match=reason):
        be.read_reference_history(directory, backend.problem.layout, identity)


@pytest.mark.parametrize("surface", ["result", "side_file", "terminal"])
def test_failure_reason_location_and_record_are_bound(tmp_path, surface):
    directory = tmp_path / surface
    backend = manufactured(safety_at=0.25)
    with pytest.raises(ContractError, match="accepted_state_safety"):
        execute(directory, backend=backend)
    result = json.loads((directory / "RunResult.json").read_text())
    identity = be.RunIdentity(
        **{k: result[k] for k in be.RunIdentity.__dataclass_fields__}
    )
    checked = be.read_reference_history(directory, backend.problem.layout, identity)
    assert checked["structural_pass"] and not checked["numerical_complete"]
    if surface == "result":
        result["first_failure"] = {
            "type": "InventedLaterFailure",
            "reason": "replacement first error",
        }
        (directory / "RunResult.json").write_bytes(be.canonical(result))
    elif surface == "side_file":
        path = directory / "FirstFailure.json"
        failure = json.loads(path.read_text())
        failure["reason"] = "replacement first error"
        path.write_bytes(be.canonical(failure))
    else:
        rows = raw_rows(directory)
        rows[-1]["failure"]["reason"] = "replacement first error"
        reseal(directory, rows)
    with pytest.raises(ContractError, match="failure_.*binding"):
        be.read_reference_history(directory, backend.problem.layout, identity)


def frozen_fixture(tmp_path, schema=be.METHOD_SCHEMA):
    reviewed = {"schema": schema, "status": "reviewed", "body": {"units": "SI"}}
    reviewed_path = tmp_path / "reviewed.json"
    reviewed_raw = be.canonical(reviewed)
    reviewed_path.write_bytes(reviewed_raw)
    value = {
        **reviewed,
        "status": "frozen",
        "freeze": {
            "frozen_at": "2026-10-15T04:16:41+00:00",
            "frozen_by": "manufactured commander",
            "reviewed_path": str(reviewed_path),
            "reviewed_sha256": sha256(reviewed_raw).hexdigest(),
            "reviewed_bytes": len(reviewed_raw),
            "review_records": [
                {"path": "manufactured-review.json", "sha256": "1" * 64}
            ],
            "user_records": ["manufactured decision"],
            "notes": ["fixture, no execution authority"],
            "plan_rule": "prospective only",
        },
    }
    path = tmp_path / "frozen.json"
    raw = be.canonical(value)
    path.write_bytes(raw)
    return value, {
        "path": str(path),
        "sha256": sha256(raw).hexdigest(),
        "bytes": len(raw),
    }


@pytest.mark.parametrize(
    "schema", [be.METHOD_SCHEMA, "solarlab.g2-route.t1-gatespec.v3"]
)
def test_top_level_freeze_body_and_review_pin(tmp_path, schema):
    value, descriptor = frozen_fixture(tmp_path, schema)
    assert be.read_frozen_definition(descriptor, schema=schema) == value


@pytest.mark.parametrize(
    "damage", ["extra", "missing", "bytes", "hash", "body", "status", "pin_bytes"]
)
def test_frozen_definition_tampering_is_rejected(tmp_path, damage):
    value, descriptor = frozen_fixture(tmp_path)
    if damage == "extra":
        value["freeze"]["override"] = True
    elif damage == "missing":
        del value["freeze"]["user_records"]
    elif damage == "bytes":
        value["freeze"]["reviewed_bytes"] += 1
    elif damage == "hash":
        value["freeze"]["reviewed_sha256"] = "f" * 64
    elif damage == "body":
        value["body"]["units"] = "changed"
    elif damage == "status":
        value["status"] = "reviewed"
    raw = be.canonical(value)
    Path(descriptor["path"]).write_bytes(raw)
    descriptor.update(sha256=sha256(raw).hexdigest(), bytes=len(raw))
    if damage == "pin_bytes":
        descriptor["bytes"] += 1
    with pytest.raises(ContractError):
        be.read_frozen_definition(descriptor, schema=be.METHOD_SCHEMA)


def test_valid_freeze_still_requires_independent_execution_admission(
    tmp_path, monkeypatch
):
    _, descriptor = frozen_fixture(tmp_path)

    def forbidden(*args, **kwargs):
        pytest.fail("an unadmitted CLI must not construct B")

    monkeypatch.setattr(be, "prepare_B_backend", forbidden)
    path = tmp_path / "manifest.json"
    path.write_bytes(
        be.canonical(
            {
                "schema": "solarlab.g2-reference-be-run.v1",
                "method_contract": descriptor,
                "configuration": {"scientific_execution_authorized": False},
            }
        )
    )
    with pytest.raises(ContractError, match="B_not_admitted"):
        be.main(["run", "--manifest", str(path)])


@pytest.mark.parametrize("damage", ["rate", "missing_rate", "event_side"])
def test_step_slope_and_event_side_are_bound(tmp_path, damage):
    directory = tmp_path / damage
    backend, _, identity, _, _ = execute(directory)
    rows = raw_rows(directory)
    # Last step on the first segment uses its left-event slope (zero), not the
    # slope of the following ramp (one), even though the state is shared.
    step = [r for r in rows if r["kind"] == "be_accepted_step" and r["segment"] == "a"][
        -1
    ]
    if damage == "rate":
        step["input_rate_hex"][0] = (1.0).hex()
    elif damage == "missing_rate":
        del step["input_rate_hex"]
    else:
        step["input_event_side"] = "right"
    reseal(directory, rows)
    with pytest.raises(ContractError, match="step_protocol_inputs"):
        be.read_reference_history(directory, backend.problem.layout, identity)


def test_non_power_two_secant_uses_declared_rounded_division(tmp_path):
    backend = manufactured(
        "decay", segments=(ProtocolSegment("one", 0.0, 0.1, (0.0, 0.0), (0.0, 0.0)),)
    )
    _, schedule, _, _, receipt = execute(tmp_path / "rounded", backend=backend)
    assert (
        len(schedule.steps) == 1
        and receipt["step_consistency_checked"]["declared_secant_division"]
    )
    step = next(
        r for r in raw_rows(tmp_path / "rounded") if r["kind"] == "be_accepted_step"
    )
    delta = be._word_values(step["storage_delta_words"], 2)
    secant = be._word_values(step["storage_secant_words"], 2)
    h = Fraction(float.fromhex(step["h_hex"]))
    assert any(a / h != b for a, b in zip(delta, secant))


def test_zero_newton_is_explicit_and_missing_diagnostics_are_not_zero(tmp_path):
    backend = manufactured(
        segments=(ProtocolSegment("one", 0.0, 0.125, (0.0, 0.0), (0.0, 0.0)),)
    )
    backend = replace(
        backend,
        budgets={
            "balance": EquationBudget(PARTICLE, [1.0, 1.0]),
            "constraint": EquationBudget(VOLT, [1.0]),
        },
    )
    directory = tmp_path / "zero"
    _, _, identity, _, receipt = execute(directory, backend=backend)
    assert receipt["structural_pass"]
    rows = raw_rows(directory)
    step = next(r for r in rows if r["kind"] == "be_accepted_step")
    assert step["newton_iterations"] == []
    assert step["newton_summary"]["initial_trial_accepted"] is True
    assert step["solver_counts"] == {
        "factorizations": 0,
        "linear_solves": 0,
        "residual_evaluations": 1,
    }
    del step["newton_iterations"]
    reseal(directory, rows)
    with pytest.raises(ContractError, match="newton_diagnostics"):
        be.read_reference_history(directory, backend.problem.layout, identity)


@pytest.mark.parametrize(
    "damage",
    [
        "empty",
        "summary",
        "projection",
        "identity",
        "merit",
        "linear",
        "damping",
        "counts",
    ],
)
def test_mandatory_newton_diagnostics_are_validated(tmp_path, damage):
    directory = tmp_path / damage
    backend, _, identity, _, _ = execute(directory)
    rows = raw_rows(directory)
    step = next(r for r in rows if r["kind"] == "be_accepted_step")
    item = step["newton_iterations"][0]
    if damage == "empty":
        step["newton_iterations"] = []
    elif damage == "summary":
        del step["newton_summary"]
    elif damage == "projection":
        item["rhs_projection_error"].pop()
    elif damage == "identity":
        item["point_identity_before"] = "a" * 64
    elif damage == "merit":
        item["merit_after"] = deepcopy(item["merit_before"])
    elif damage == "linear":
        item["linear_residual_inf"] = -1.0
    elif damage == "damping":
        item["damping"] = 0.75
    else:
        step["solver_counts"]["factorizations"] += 1
    reseal(directory, rows)
    with pytest.raises(ContractError, match="newton_diagnostics"):
        be.read_reference_history(directory, backend.problem.layout, identity)


@pytest.mark.parametrize(
    "damage", ["location", "record_hash", "body_hash", "record_index"]
)
def test_failure_pointer_tampering_is_rejected(tmp_path, damage):
    directory = tmp_path / damage
    backend = manufactured(safety_at=0.25)
    with pytest.raises(ContractError):
        execute(directory, backend=backend)
    path = directory / "RunResult.json"
    result = json.loads(path.read_text())
    identity = be.RunIdentity(
        **{k: result[k] for k in be.RunIdentity.__dataclass_fields__}
    )
    pointer = result["first_failure"]
    if damage == "location":
        pointer["location"]["target_step"]["index"] += 1
    elif damage == "record_hash":
        pointer["record_sha256"] = "f" * 64
    elif damage == "body_hash":
        pointer["failure_sha256"] = "e" * 64
    else:
        pointer["record_index"] += 1
    path.write_bytes(be.canonical(result))
    with pytest.raises(ContractError, match="failure_result_binding"):
        be.read_reference_history(directory, backend.problem.layout, identity)


def test_reference_layout_uses_public_constructor_without_time_evolution(monkeypatch):
    from scripts.benchmarks import coupled_device_prototype as coupled

    # Synthetic material numbers exercise public construction without loading
    # an archived B request or carrying out any physical time integration.
    definition = coupled.SlabDefinition(
        id="ManufacturedLayout",
        source_path="manufactured-only",
        source_sha256="1" * 64,
        dynamic=True,
        length=3e-7,
        area=1.0,
        temperature=300.0,
        epsilon=1e-10,
        mu_n=0.001,
        mu_p=0.001,
        diffusion_ion=1e-14,
        ion_initial=1e20,
        ion_capacity=2e20,
        trap_density=1e18,
        capture_n=1e-14,
        capture_p=1e-14,
        n1=1e15,
        p1=1e15,
        intrinsic_product=1e30,
        ni=1e15,
        n_eq=1e15,
        p_eq=1e15,
        f_eq=0.5,
        alpha=1e5,
        photon_reference=0.0,
    )
    public_model = coupled.AffineCoupledSlab(definition, 8)
    request = {
        "numeric_packet": {
            "definition": asdict(definition),
            "definition_identity": definition.identity,
            "intervals": 8,
            "layout_identity": public_model.layout.identity,
        }
    }

    def forbidden(*args, **kwargs):
        pytest.fail("metadata helper attempted time evolution or observation")

    monkeypatch.setattr(be, "conservative_be_step", forbidden)
    monkeypatch.setattr(be, "prepare_B_backend", forbidden)
    monkeypatch.setattr(be, "run_reference", forbidden)
    monkeypatch.setattr(coupled, "run_voltage_lift_native_pilot", forbidden)
    layout = be.reference_layout(request)
    assert (
        isinstance(layout, Layout) and layout.identity == public_model.layout.identity
    )
    assert layout.size == 45 and tuple(layout.offsets) == tuple(
        public_model.layout.offsets
    )
    altered = deepcopy(request)
    altered["numeric_packet"]["layout_identity"] = "a" * 64
    with pytest.raises(ContractError, match="layout_identity"):
        be.reference_layout(altered)
    altered = deepcopy(request)
    altered["numeric_packet"]["definition"]["area"] = 2.0
    with pytest.raises(ContractError, match="layout_definition_identity"):
        be.reference_layout(altered)


def runtime_fixture(tmp_path, monkeypatch, *, action="run", name="manufactured"):
    monkeypatch.setattr(be, "_boot_identity", lambda: "manufactured-boot")

    def threads():
        return {
            "library": "/manufactured/libBLAS.dylib",
            "getter": "BLASGetThreading",
            "before_mode": 1,
            "measured_mode": 1,
            "required_single_thread_mode": 1,
            "thread_environment": {k: "1" for k in be.THREAD_ENVIRONMENT},
            "python_executable": str(Path(sys.executable).resolve()),
            "python_version": sys.version,
            "checked_ns": be.time.monotonic_ns(),
        }

    monkeypatch.setattr(be, "runtime_threads", threads)
    monkeypatch.setattr(
        be,
        "_runtime_thread_end",
        lambda r: r.update(
            final_measured_mode=1, final_checked_ns=be.time.monotonic_ns()
        ),
    )
    for key in be.THREAD_ENVIRONMENT:
        monkeypatch.setenv(key, "1")
    monkeypatch.delenv("SOLARLAB_G2_ROUTE_CHANNEL_FD", raising=False)
    slots = tmp_path / "slots"
    slots.mkdir(exist_ok=True)
    clock_path = tmp_path / "Clock.json"
    clock_path.write_bytes(
        be.canonical(be.pipeline_clock("manufactured-pipeline", 60.0))
    )

    def pin(path):
        raw = path.read_bytes()
        return {"path": str(path), "bytes": len(raw), "sha256": sha256(raw).hexdigest()}

    execution = {
        "stage": action,
        "directory": str(tmp_path / name),
        "channel": 0,
        "wall_s": 30.0,
        "pipeline": {"id": "manufactured-pipeline", "clock": pin(clock_path)},
        "supervision": {
            "rss_bytes": 4 * be.GIB,
            "artifact_bytes": 64 * 1048576,
            "poll_s": 0.01,
            "term_grace_s": 0.1,
            "kill_wait_s": 0.1,
            "receipt_reserve_s": 0.1,
            "closure_path": str(tmp_path / (name + ".closure.json")),
            "cancel_path": str(tmp_path / (name + ".cancel.json")),
        },
    }
    plan = {
        "status": "admitted",
        "scope": "manufactured runtime test only",
        "authority": {"scientific_execution_authorized": True},
        "resources": {
            "WS1_channels": 4,
            "threads_per_case": 1,
            "rss_bytes_per_case": 4 * be.GIB,
            "combined_persisted_case_bytes": 64 * 1048576,
        },
        "artifacts": {"channel_directory": str(slots)},
        "cases": [{"id": name, "execution": execution}],
    }
    plan_path = tmp_path / (name + ".plan.json")
    plan_path.write_bytes(be.canonical(plan))
    manifest = {
        "case_id": name,
        "directory": execution["directory"],
        "plan": pin(plan_path),
        "limits": asdict(
            be.Limits(20.0, output_bytes=64 * 1048576, logical_bytes=64 * 1048576)
        ),
    }
    return manifest, plan, slots, pin


def test_BE_runtime_imports_reviewed_channel_and_rejects_missing_FD(
    tmp_path, monkeypatch
):
    from scripts.benchmarks import g2_route

    assert be.route_channel is g2_route._channel
    manifest, plan, _, _ = runtime_fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="supervised_channel_fd_required"):
        with be.supervised_stage(manifest, plan, "run"):
            pytest.fail("missing FD entered producer")
    assert not Path(manifest["directory"]).exists()


def test_BE_runtime_rejects_wrong_slot_and_does_not_unlock_parent(
    tmp_path, monkeypatch
):
    import fcntl
    from scripts.benchmarks.g2_route_launcher import _owned_channel

    manifest, plan, slots, _ = runtime_fixture(tmp_path, monkeypatch)
    (slots / "1.lock").touch()
    with _owned_channel(slots, 0) as (fd, _):
        monkeypatch.setenv("SOLARLAB_G2_ROUTE_CHANNEL_FD", str(fd))
        plan["cases"][0]["execution"]["channel"] = 1
        with pytest.raises(ValueError, match="inherited_channel_identity"):
            with be.supervised_stage(manifest, plan, "run"):
                pytest.fail("wrong slot entered producer")
        plan["cases"][0]["execution"]["channel"] = 0
        with be.supervised_stage(manifest, plan, "run") as runtime:
            Path(manifest["directory"]).mkdir()
            runtime.return_code = 0
        os.fstat(fd)  # Child context neither closes nor unlocks this descriptor.
        with (slots / "0.lock").open("a+b") as competitor:
            with pytest.raises(BlockingIOError):
                fcntl.flock(competitor.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    receipt = json.loads(
        (Path(manifest["directory"]) / "RuntimeReceipt.json").read_text()
    )
    assert receipt["thread_receipt"]["final_measured_mode"] == 1
    assert receipt["followup_stages_executed_implicitly"] is False
    assert receipt["first_failure"] is None and receipt["return_code"] == 0


@pytest.mark.parametrize(
    "damage", ["expired", "wrong_boot", "oversized_stage", "sup_cap", "wrong_stage"]
)
def test_pipeline_and_supervision_cannot_be_reset_or_exceeded(
    tmp_path, monkeypatch, damage
):
    manifest, plan, _, pin = runtime_fixture(tmp_path, monkeypatch)
    execution = plan["cases"][0]["execution"]
    path = Path(execution["pipeline"]["clock"]["path"])
    clock = json.loads(path.read_text())
    if damage == "expired":
        clock["deadline_ns"] = time.monotonic_ns() - 1
        clock["started_ns"] = clock["deadline_ns"] - 60 * 10**9
    elif damage == "wrong_boot":
        clock["boot_identity"] = "other-boot"
    elif damage == "oversized_stage":
        execution["wall_s"] = 60.0
    elif damage == "sup_cap":
        execution["supervision"]["rss_bytes"] = 5 * be.GIB
    else:
        execution["stage"] = "read"
    path.write_bytes(be.canonical(clock))
    execution["pipeline"]["clock"] = pin(path)
    with pytest.raises((ContractError, ValueError)):
        be.StageRuntime(manifest, plan, "run")


def test_runtime_cancellation_and_original_failure_receipt(tmp_path, monkeypatch):
    from scripts.benchmarks.g2_route_launcher import _owned_channel

    manifest, plan, slots, _ = runtime_fixture(tmp_path, monkeypatch)
    with _owned_channel(slots, 0) as (fd, _):
        monkeypatch.setenv("SOLARLAB_G2_ROUTE_CHANNEL_FD", str(fd))
        with pytest.raises(ContractError, match="runtime_cancelled"):
            with be.supervised_stage(manifest, plan, "run") as runtime:
                runtime.directory.mkdir()
                Path(runtime.supervision["cancel_path"]).touch()
                runtime.checkpoint()
    receipt = json.loads(
        (Path(manifest["directory"]) / "RuntimeReceipt.json").read_text()
    )
    assert receipt["first_failure"]["reason"] == "reference_be_runtime_cancelled"
    assert receipt["outcome"] == "raised" and receipt["scientific_pass"] is False
    assert receipt["thread_receipt"]["final_measured_mode"] == 1
    assert receipt["thread_receipt"]["final_measurement_status"] == "measured"
    assert (
        receipt["started_ns"]
        <= receipt["thread_receipt"]["final_checked_ns"]
        <= receipt["ended_ns"]
    )


def test_protocol_timing_tracks_only_manufactured_producer(tmp_path):
    backend = manufactured()
    schedule = be.compile_schedule(
        backend.segments,
        tags_for(backend.segments, [[s.start, s.end] for s in backend.segments]),
        level=0,
        seed=Fraction(1, 8),
    )
    request = {
        "kind": "manufactured",
        "segments": [asdict(s) for s in backend.segments],
    }
    identity = be.RunIdentity(
        "timed",
        be.digest(request),
        "1" * 64,
        "2" * 64,
        "3" * 40,
        "test",
        schedule.identity,
        backend.problem.source_identity,
        backend.initial.identity,
        be.digest(be.effective_controls(backend)),
    )
    ledger = be.RuntimeLedger()
    be.run_reference(
        backend,
        schedule,
        identity,
        request,
        tmp_path / "timed",
        be.Limits(60.0),
        timing=ledger,
    )
    receipt = ledger.snapshot()
    assert [s["segment"] for s in receipt["segments"]] == [
        s.id for s in backend.segments
    ]
    assert sum(s["accepted_steps"] for s in receipt["segments"]) == len(schedule.steps)
    assert all(s["wall_ns"] > 0 for s in receipt["segments"])
    assert receipt["components"]["be_step"]["calls"] == len(schedule.steps)
    assert receipt["components"]["interval_observation"]["calls"] == len(schedule.steps)
    assert (
        receipt["components"]["reader"]["calls"]
        == receipt["components"]["reference_error"]["calls"]
        == 0
    )
    assert receipt["peak_process_RSS_bytes"] > 0


def test_actual_thread_getter_receipt_and_failure(monkeypatch):
    class Function:
        def __init__(self, operation):
            self.operation = operation

        def __call__(self, *args):
            return self.operation(*args)

    state = {"mode": 0, "get_calls": 0}

    def get():
        state["get_calls"] += 1
        return state["mode"]

    library = SimpleNamespace(
        BLASGetThreading=Function(get),
        BLASSetThreading=Function(lambda value: state.update(mode=value) or 0),
    )
    monkeypatch.setattr(be.ctypes, "CDLL", lambda path: library)
    monkeypatch.setattr(
        be,
        "sys",
        SimpleNamespace(
            platform="darwin", executable=sys.executable, version=sys.version
        ),
    )
    for key in be.THREAD_ENVIRONMENT:
        monkeypatch.setenv(key, "1")
    receipt = be.runtime_threads()
    assert (
        receipt["before_mode"] == 0
        and receipt["measured_mode"] == 1
        and state["get_calls"] == 2
    )
    state["mode"] = 2
    with pytest.raises(ContractError, match="runtime_final_BLAS_thread_mode"):
        be._runtime_thread_end(receipt)
    assert receipt["final_measured_mode"] == 2


@pytest.mark.parametrize("reset_clock", [False, True])
def test_readback_is_a_separate_stage_with_original_clock(
    tmp_path, monkeypatch, reset_clock
):
    from scripts.benchmarks.g2_route_launcher import _owned_channel

    manifest, plan, slots, pin = runtime_fixture(tmp_path, monkeypatch)
    with _owned_channel(slots, 0) as (fd, _):
        monkeypatch.setenv("SOLARLAB_G2_ROUTE_CHANNEL_FD", str(fd))
        with be.supervised_stage(manifest, plan, "run") as runtime:
            backend, _, identity, _, _ = execute(
                runtime.directory,
                checkpoint=runtime.checkpoint,
                plan_sha256=manifest["plan"]["sha256"],
                timing=runtime.ledger,
            )
            runtime.return_code = 0
        producer = Path(manifest["directory"])
        followup = deepcopy(manifest)
        followup.update(
            case_id="manufactured_read",
            directory=str(tmp_path / "read"),
            history={
                "directory": str(producer),
                "identity": identity.payload(),
                "run_result": pin(producer / "RunResult.json"),
                "runtime_receipt": pin(producer / "RuntimeReceipt.json"),
            },
        )
        later_plan = deepcopy(plan)
        later_plan["cases"][0]["id"] = followup["case_id"]
        execution = later_plan["cases"][0]["execution"]
        execution.update(stage="read", directory=followup["directory"])
        if reset_clock:
            clock_path = tmp_path / "ReplacementClock.json"
            clock_path.write_bytes(
                be.canonical(be.pipeline_clock("manufactured-pipeline", 60.0))
            )
            execution["pipeline"]["clock"] = pin(clock_path)
        monkeypatch.setattr(
            be, "reference_layout", lambda request: backend.problem.layout
        )
        if reset_clock:
            with pytest.raises(ContractError, match="runtime_upstream_clock"):
                with be.supervised_stage(followup, later_plan, "read") as runtime:
                    be._read_manifest(followup, runtime)
            assert not Path(followup["directory"]).exists()
        else:
            with be.supervised_stage(followup, later_plan, "read") as runtime:
                runtime.return_code = be._read_manifest(followup, runtime)
            receipt = json.loads(
                (Path(followup["directory"]) / "RuntimeReceipt.json").read_text()
            )
            assert (
                receipt["pipeline_clock"]
                == json.loads((producer / "RuntimeReceipt.json").read_text())[
                    "pipeline_clock"
                ]
            )
            assert receipt["timing"]["components"]["reader"]["calls"] == 1
            assert receipt["timing"]["components"]["be_step"]["calls"] == 0
            assert json.loads(
                (Path(followup["directory"]) / "Readback.json").read_text()
            )["numerical_complete"]


def test_error_analysis_is_separately_dispatched_and_keeps_nonzero_status(
    tmp_path, monkeypatch
):
    from scripts.benchmarks import g2_reference_error
    from scripts.benchmarks.g2_route_launcher import _owned_channel

    manifest, plan, slots, pin = runtime_fixture(
        tmp_path, monkeypatch, action="evaluate", name="analysis"
    )
    manifest["method_contract"] = {"sha256": "1" * 64}
    path = tmp_path / "ErrorManifest.json"
    path.write_bytes(
        be.canonical(
            {
                "directory": manifest["directory"],
                "limits": manifest["limits"],
                "method_contract": manifest["method_contract"],
                "levels": [
                    {"directory": str(tmp_path / f"level{i}")} for i in range(3)
                ],
            }
        )
    )
    manifest["error_manifest"] = pin(path)
    manifest["upstream_runtime"] = [{"fixture": i} for i in range(3)]
    seen = []
    monkeypatch.setattr(be, "_upstream_runtime", lambda *args: seen.append("upstream"))

    def analyze(argv):
        assert argv == ["evaluate", "--manifest", str(path)]
        seen.append("E1/E3")
        return 2  # Downstream unknown must not become a successful qualification.

    monkeypatch.setattr(g2_reference_error, "main", analyze)
    with _owned_channel(slots, 0) as (fd, _):
        monkeypatch.setenv("SOLARLAB_G2_ROUTE_CHANNEL_FD", str(fd))
        with be.supervised_stage(manifest, plan, "evaluate") as runtime:
            runtime.return_code = be._error_manifest(manifest, runtime)
            assert runtime.return_code == 2
    assert seen == ["upstream"] * 3 + ["E1/E3"]
    assert runtime.ledger.snapshot()["components"]["reference_error"]["calls"] == 1
    assert runtime.ledger.snapshot()["components"]["be_step"]["calls"] == 0


def test_CLI_runs_only_selected_stage_under_supervision(tmp_path, monkeypatch):
    from scripts.benchmarks.g2_route_launcher import _owned_channel

    manifest, plan, slots, pin = runtime_fixture(tmp_path, monkeypatch)
    frozen, descriptor = frozen_fixture(tmp_path)
    manifest.update(
        schema="solarlab.g2-reference-be-run.v1",
        method_contract=descriptor,
        configuration={
            "scientific_execution_authorized": True,
            "source_review_accepted": True,
            "commander_admission_identity": "manufactured-test-only",
        },
    )
    root = Path(be.__file__).parent
    manifest["source_files"] = [
        pin(root / name)
        for name in (
            "g2_reference_be.py",
            "conservative_device.py",
            "contract_prototype.py",
            "precision_prototype.py",
            "coupled_device_prototype.py",
            "interval_observation.py",
            "native_history.py",
            "g2_route.py",
            "g2_route_launcher.py",
        )
    ]
    path = tmp_path / "RuntimeManifest.json"
    path.write_bytes(be.canonical(manifest))
    called = []

    def producer(value, runtime, method):
        assert method == frozen
        called.append("producer")
        runtime.directory.mkdir()
        return 0

    def forbidden(*args):
        pytest.fail("producer implicitly ran a postprocessing stage")

    monkeypatch.setattr(be, "_run_manifest", producer)
    monkeypatch.setattr(be, "_read_manifest", forbidden)
    monkeypatch.setattr(be, "_error_manifest", forbidden)
    with _owned_channel(slots, 0) as (fd, _):
        monkeypatch.setenv("SOLARLAB_G2_ROUTE_CHANNEL_FD", str(fd))
        assert be.main(["run", "--manifest", str(path)]) == 0
    assert called == ["producer"]
    receipt = json.loads(
        (Path(manifest["directory"]) / "RuntimeReceipt.json").read_text()
    )
    assert receipt["return_code"] == 0 and receipt["stage"] == "run"


def test_run_manifest_preserves_probe_binding_and_instrumentation(
    tmp_path, monkeypatch
):
    from scripts.benchmarks.g2_route_launcher import _owned_channel

    manifest, plan, slots, pin = runtime_fixture(tmp_path, monkeypatch)
    segments, tags = frozen_protocol()
    backend = manufactured(segments=segments[:2])
    request = {
        "kind": "manufactured_B_interface",
        "segments": [asdict(s) for s in segments],
        "observation_times": {
            s.id: [float.fromhex(t["time_hex"]) for t in tags if t["segment"] == s.id]
            for s in segments
        },
    }
    request_path = tmp_path / "Request.json"
    request_path.write_bytes(be.canonical(request))
    schedule = be.compile_schedule(backend.segments, tags[:66], level=0)
    identity = be.RunIdentity(
        manifest["case_id"],
        be.digest(request),
        "1" * 64,
        manifest["plan"]["sha256"],
        "3" * 40,
        "manufactured-runtime",
        schedule.identity,
        backend.problem.source_identity,
        backend.initial.identity,
        be.digest(be.effective_controls(backend)),
    )
    manifest.update(
        request=pin(request_path),
        identity=identity.payload(),
        method_contract={"sha256": "1" * 64},
        tags=tags[:66],
        schedule_identity=schedule.identity,
        configuration={"segment_count": 2, "ladder_level": 0},
    )
    monkeypatch.setattr(be, "prepare_B_backend", lambda *a, **k: backend)
    with _owned_channel(slots, 0) as (fd, _):
        monkeypatch.setenv("SOLARLAB_G2_ROUTE_CHANNEL_FD", str(fd))
        with be.supervised_stage(manifest, plan, "run") as runtime:
            runtime.return_code = be._run_manifest(
                manifest, runtime, {"time_ladder": {"seed_cap_s": [1, 512]}}
            )
    result = json.loads((Path(manifest["directory"]) / "RunResult.json").read_text())
    assert (
        result["counts"]["accepted_steps"] == 128
        and result["counts"]["observations"] == 66
    )
    receipt = json.loads(
        (Path(manifest["directory"]) / "RuntimeReceipt.json").read_text()
    )
    assert receipt["timing"]["components"]["setup"]["calls"] == 1
    assert receipt["timing"]["components"]["reader"]["calls"] == 0
    assert receipt["result_file"] == pin(Path(manifest["directory"]) / "RunResult.json")


def runtime_producer_and_reader(tmp_path, monkeypatch, *, body_failure=False):
    """A real serialized pipeline around the existing manufactured model only."""
    from scripts.benchmarks.g2_route_launcher import _owned_channel

    root = tmp_path / "producer"
    root.mkdir()
    manifest, plan, slots, pin = runtime_fixture(root, monkeypatch)
    with _owned_channel(slots, 0) as (fd, _):
        monkeypatch.setenv("SOLARLAB_G2_ROUTE_CHANNEL_FD", str(fd))
        try:
            with be.supervised_stage(manifest, plan, "run") as runtime:
                backend, _, identity, _, _ = execute(
                    runtime.directory,
                    checkpoint=runtime.checkpoint,
                    plan_sha256=manifest["plan"]["sha256"],
                    timing=runtime.ledger,
                )
                manifest["identity"] = identity.payload()
                if body_failure:
                    raise RuntimeError("manufactured post-run failure")
                runtime.return_code = 0
        except RuntimeError:
            assert body_failure
    directory = Path(manifest["directory"])
    root = tmp_path / "reader"
    root.mkdir()
    followup, later_plan, slots, _ = runtime_fixture(
        root, monkeypatch, action="read", name="reader"
    )
    later_plan["cases"][0]["execution"]["pipeline"] = deepcopy(
        plan["cases"][0]["execution"]["pipeline"]
    )
    plan_path = Path(followup["plan"]["path"])
    plan_path.write_bytes(be.canonical(later_plan))
    followup["plan"] = pin(plan_path)
    followup["history"] = {
        "directory": str(directory),
        "identity": identity.payload(),
        "run_result": pin(directory / "RunResult.json"),
        "runtime_receipt": pin(directory / "RuntimeReceipt.json"),
    }
    monkeypatch.setattr(be, "reference_layout", lambda request: backend.problem.layout)
    return followup, later_plan, slots, pin, directory


@pytest.mark.parametrize(
    "damage",
    [
        "missing_fields",
        "impossible_timing",
        "invalid_BLAS",
        "bool_nanoseconds",
        "float_nanoseconds",
        "unknown_field",
        "bad_manifest_digest",
        "manifest_bytes",
        "manifest_pin_bytes",
        "plan_sha",
        "execution_copy",
        "clock_pin_bytes",
        "final_time_before_start",
        "missing_final_reading",
        "component_time",
        "segment_time",
        "phase_list",
        "success_with_first_failure",
        "bool_return_code",
        "negative_RSS",
        "fake_stage",
        "result_plan_chain",
        "segment_name",
        "segment_counts",
    ],
)
def test_BE4_rejects_re_pinned_malformed_runtime_before_reader(
    tmp_path, monkeypatch, damage
):
    from scripts.benchmarks.g2_route_launcher import _owned_channel

    manifest, plan, slots, pin, directory = runtime_producer_and_reader(
        tmp_path, monkeypatch
    )
    path = directory / "RuntimeReceipt.json"
    record = json.loads(path.read_bytes())
    if damage == "missing_fields":
        keep = {
            "schema",
            "stage",
            "pipeline_id",
            "pipeline_clock",
            "original_deadline_ns",
            "result_file",
            "case_id",
        }
        record = {k: v for k, v in record.items() if k in keep}
    elif damage == "impossible_timing":
        record.update(started_ns=2**60 + 1, ended_ns=-1)
    elif damage == "invalid_BLAS":
        record["thread_receipt"].update(measured_mode=2, final_measured_mode=2)
    elif damage == "bool_nanoseconds":
        record["started_ns"] = True
    elif damage == "float_nanoseconds":
        record["ended_ns"] = float(record["ended_ns"])
    elif damage == "unknown_field":
        record["skip_validation"] = True
    elif damage == "bad_manifest_digest":
        record["manifest_digest"] = "a" * 64
    elif damage == "manifest_bytes":
        p = Path(record["manifest"]["path"])
        value = json.loads(p.read_bytes())
        value["directory"] += "_wrong"
        p.write_bytes(be.canonical(value))
        record["manifest"] = pin(p)
    elif damage == "manifest_pin_bytes":
        record["manifest"]["bytes"] += 1
    elif damage == "plan_sha":
        record["plan"]["sha256"] = "a" * 64
    elif damage == "execution_copy":
        record["execution"]["channel"] = True
    elif damage == "clock_pin_bytes":
        record["pipeline_clock"]["bytes"] += 1
    elif damage == "final_time_before_start":
        record["thread_receipt"]["final_checked_ns"] = record["started_ns"] - 1
    elif damage == "missing_final_reading":
        del record["thread_receipt"]["final_checked_ns"]
    elif damage == "component_time":
        record["timing"]["components"]["be_step"]["wall_ns"] = 2**60
    elif damage == "segment_time":
        record["timing"]["segments"][0]["ended_ns"] = record["ended_ns"] + 1
    elif damage == "phase_list":
        record["timing"]["not_entered"] = []
    elif damage == "success_with_first_failure":
        record["first_failure"] = {"type": "X", "reason": "hidden failure"}
    elif damage == "bool_return_code":
        record["return_code"] = False
    elif damage == "negative_RSS":
        record["peak_process_RSS_bytes"] = -1
    elif damage == "fake_stage":
        record["stage"] = "evaluate"
    elif damage == "result_plan_chain":
        p = directory / "RunResult.json"
        value = json.loads(p.read_bytes())
        value["plan_sha256"] = "a" * 64
        p.write_bytes(be.canonical(value))
        record["result_file"] = pin(p)
        manifest["history"]["run_result"] = pin(p)
    elif damage == "segment_name":
        record["timing"]["segments"][0]["segment"] = "different_protocol"
    elif damage == "segment_counts":
        record["timing"]["segments"][0]["accepted_steps"] += 1
    path.write_bytes(be.canonical(record))
    manifest["history"]["runtime_receipt"] = pin(path)
    monkeypatch.setattr(
        be,
        "read_reference_history",
        lambda *a, **k: pytest.fail("malformed runtime reached reader"),
    )
    with _owned_channel(slots, 0) as (fd, _):
        monkeypatch.setenv("SOLARLAB_G2_ROUTE_CHANNEL_FD", str(fd))
        with be.supervised_stage(manifest, plan, "read") as runtime:
            with pytest.raises(ContractError):
                be._read_manifest(manifest, runtime)
    assert not Path(manifest["directory"]).exists()


@pytest.mark.parametrize("failed", [False, True])
def test_BE4_complete_runtime_and_explicit_failed_producer_readback(
    tmp_path, monkeypatch, failed
):
    from scripts.benchmarks.g2_route_launcher import _owned_channel

    manifest, plan, slots, _, directory = runtime_producer_and_reader(
        tmp_path, monkeypatch, body_failure=failed
    )
    with _owned_channel(slots, 0) as (fd, _):
        monkeypatch.setenv("SOLARLAB_G2_ROUTE_CHANNEL_FD", str(fd))
        with be.supervised_stage(manifest, plan, "read") as runtime:
            if failed:
                with pytest.raises(ContractError, match="runtime_upstream_incomplete"):
                    be._upstream_runtime(
                        manifest["history"]["runtime_receipt"], runtime, directory
                    )
            runtime.return_code = be._read_manifest(manifest, runtime)
            assert runtime.return_code == (2 if failed else 0)
    readback = json.loads((Path(manifest["directory"]) / "Readback.json").read_bytes())
    assert readback["numerical_complete"] is True
    assert readback["scientific_pass"] is False
    assert readback["upstream_runtime"]["runtime_qualified"] is (not failed)
    assert (readback["upstream_runtime"]["first_failure"] is not None) is failed


@pytest.mark.parametrize("body_failure", [False, True])
@pytest.mark.parametrize("getter_failure", [False, True])
def test_BE4_final_BLAS_on_exception_preserves_first_failure(
    tmp_path, monkeypatch, body_failure, getter_failure
):
    from scripts.benchmarks.g2_route_launcher import _owned_channel

    manifest, plan, slots, _ = runtime_fixture(tmp_path, monkeypatch)
    calls = []
    original = be._runtime_thread_end

    def final(receipt):
        calls.append("getter")
        if getter_failure:
            raise OSError("manufactured getter unavailable")
        original(receipt)

    monkeypatch.setattr(be, "_runtime_thread_end", final)
    with _owned_channel(slots, 0) as (fd, _):
        monkeypatch.setenv("SOLARLAB_G2_ROUTE_CHANNEL_FD", str(fd))
        try:
            with be.supervised_stage(manifest, plan, "run") as runtime:
                runtime.directory.mkdir()
                if body_failure:
                    raise RuntimeError("manufactured original failure")
                runtime.return_code = 0
        except (OSError, RuntimeError) as error:
            assert type(error) is (RuntimeError if body_failure else OSError)
        else:
            assert not body_failure and not getter_failure
    record = json.loads(
        (Path(manifest["directory"]) / "RuntimeReceipt.json").read_bytes()
    )
    assert calls == ["getter"]
    assert record["thread_receipt"]["final_measurement_status"] == (
        "unavailable" if getter_failure else "measured"
    )
    if body_failure:
        assert record["first_failure"]["reason"] == "manufactured original failure"
    elif getter_failure:
        assert record["first_failure"]["reason"] == "manufactured getter unavailable"
    else:
        assert record["first_failure"] is None


def test_BE4_integer_pipeline_clock_above_binary64_exact_range(tmp_path, monkeypatch):
    from scripts.benchmarks.g2_route_launcher import _owned_channel

    now = 2**60 + 129
    monkeypatch.setattr(be.time, "monotonic_ns", lambda: now)
    manifest, plan, slots, _, directory = runtime_producer_and_reader(
        tmp_path, monkeypatch
    )
    with _owned_channel(slots, 0) as (fd, _):
        monkeypatch.setenv("SOLARLAB_G2_ROUTE_CHANNEL_FD", str(fd))
        with be.supervised_stage(manifest, plan, "read") as runtime:
            record = be._upstream_runtime(
                manifest["history"]["runtime_receipt"], runtime, directory
            )
            assert record["started_ns"] == now and type(record["started_ns"]) is int
            assert record["original_deadline_ns"] == now + 60 * 10**9
            assert type(record["original_deadline_ns"]) is int


# Exact original time words are appended below from the already reviewed plan;
# tests never access the archive, build the device, or open a historical gzip.

FROZEN_PROTOCOL_TIMES = (
    """
    0x0.0p+0 0x1.999999999999ap-9 0x1.999999999999ap-8 0x1.3333333333334p-7
    0x1.999999999999ap-7 0x1.0000000000000p-6 0x1.3333333333334p-6 0x1.6666666666667p-6
    0x1.999999999999ap-6 0x1.ccccccccccccdp-6 0x1.0000000000000p-5 0x1.199999999999ap-5
    0x1.3333333333334p-5 0x1.4cccccccccccdp-5 0x1.6666666666667p-5 0x1.8000000000000p-5
    0x1.999999999999ap-5 0x1.b333333333334p-5 0x1.ccccccccccccdp-5 0x1.e666666666667p-5
    0x1.0000000000000p-4 0x1.0cccccccccccdp-4 0x1.199999999999ap-4 0x1.2666666666667p-4
    0x1.3333333333334p-4 0x1.4000000000000p-4 0x1.4cccccccccccdp-4 0x1.599999999999ap-4
    0x1.6666666666667p-4 0x1.7333333333334p-4 0x1.8000000000000p-4 0x1.8cccccccccccdp-4
    0x1.999999999999ap-4
    """,
    """
    0x1.999999999999ap-4 0x1.a666666666667p-4 0x1.b333333333334p-4 0x1.c000000000000p-4
    0x1.ccccccccccccdp-4 0x1.d99999999999ap-4 0x1.e666666666667p-4 0x1.f333333333334p-4
    0x1.0000000000000p-3 0x1.0666666666667p-3 0x1.0cccccccccccdp-3 0x1.1333333333334p-3
    0x1.199999999999ap-3 0x1.2000000000000p-3 0x1.2666666666667p-3 0x1.2cccccccccccdp-3
    0x1.3333333333334p-3 0x1.399999999999ap-3 0x1.4000000000000p-3 0x1.4666666666667p-3
    0x1.4cccccccccccdp-3 0x1.5333333333334p-3 0x1.599999999999ap-3 0x1.6000000000000p-3
    0x1.6666666666667p-3 0x1.6cccccccccccdp-3 0x1.7333333333334p-3 0x1.799999999999ap-3
    0x1.8000000000000p-3 0x1.8666666666667p-3 0x1.8cccccccccccdp-3 0x1.9333333333334p-3
    0x1.999999999999ap-3
    """,
    """
    0x1.999999999999ap-3 0x1.9999a70592e70p-3 0x1.9999b163a30e5p-3 0x1.9999c3c3c9e6dp-3
    0x1.9999e45596582p-3 0x1.999a1e0fc9042p-3 0x1.999a84612a836p-3 0x1.999b39bb349bep-3
    0x1.999c7b2a4c24ap-3 0x1.999eb4e2bd2cfp-3 0x1.99a2a6ad7aac5p-3 0x1.99a9a4779f060p-3
    0x1.99b608c054337p-3 0x1.99cbff6a58322p-3 0x1.99f2ed3773c4dp-3 0x1.9a37ecf116263p-3
    0x1.9ab238baab35ep-3 0x1.9b8afbb783de1p-3 0x1.9d0b2dd4dc51fp-3 0x1.9fb42402bf2fep-3
    0x1.a46b19a98733dp-3 0x1.acc65b411bf97p-3 0x1.bb960c0667a72p-3 0x1.d5d691787c997p-3
    0x1.022f1fd90a8ddp-2 0x1.2b6b7557af67fp-2 0x1.7481dd261becep-2 0x1.ecccccccccccdp-2
    0x1.f60cd2bba5a84p-2 0x1.6dd3fd850a068p-1 0x1.8666666666666p-1 0x1.0b33333333333p+0
    0x1.1ca78a30c01f8p+0 0x1.5333333333333p+0 0x1.9b33333333333p+0 0x1.d0fb8ca164f25p+0
    0x1.e333333333333p+0 0x1.159999999999ap+1 0x1.399999999999ap+1 0x1.5d9999999999ap+1
    0x1.819999999999ap+1 0x1.884d1af340315p+1 0x1.a59999999999ap+1 0x1.c99999999999ap+1
    0x1.ed9999999999ap+1 0x1.08ccccccccccdp+2 0x1.1accccccccccdp+2 0x1.2cccccccccccdp+2
    0x1.3eccccccccccdp+2 0x1.50ccccccccccdp+2 0x1.51c6deac61ee1p+2 0x1.62ccccccccccdp+2
    0x1.74ccccccccccdp+2 0x1.86ccccccccccdp+2 0x1.98ccccccccccdp+2 0x1.aaccccccccccdp+2
    0x1.bcccccccccccdp+2 0x1.ceccccccccccdp+2 0x1.e0ccccccccccdp+2 0x1.f2ccccccccccdp+2
    0x1.0266666666666p+3 0x1.0b66666666666p+3 0x1.1466666666666p+3 0x1.1d66666666666p+3
    0x1.2666666666666p+3
    """,
)
