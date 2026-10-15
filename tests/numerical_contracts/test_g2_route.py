"""Manufactured D14/D25/route regressions; never launch a physical native run."""
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import time

import pytest

from scripts.benchmarks.native_history import HistoryWriter
from scripts.benchmarks.native_readback import HistoryVerificationError, verify_history, _decode
from scripts.benchmarks.physical_tolerance import certificate_failure_policy
from test_native_readback import fixture, tag, digest


def continued_fixture(*, failure=True, observation="online"):
    def policy(request, observer_policy):
        request["certificate_failure_policy"] = certificate_failure_policy(observation=observation)
        return observer_policy
    request, rows, result = fixture(policy)
    first = None
    progress = {"schema": "solarlab.scientific-progress.v1", "failed_gates": [],
                "first_failure": None, "pending_proofs": ["T1"], "status": "pending", "passed": False}
    for row in rows:
        if failure and first is None and row["kind"] == "voltage_lift_current_certificate":
            evidence = _decode(row["evidence"])
            # Change the actual reference centers; both readback errors must
            # independently reconstruct the now-failing current comparison.
            for key in ("raw_reference_A", "same_state_exact_tangent_reference_A"):
                for value in evidence[key]:
                    value["center"][0] -= value["center"][1]
            evidence["source_and_readback_error_A"] = ((Fraction(1),)*2,)*2
            evidence["combined_current_bound_A"] = (Fraction(2),)*2
            evidence["passed"] = False
            row["evidence"] = tag(evidence)
            first = {"gate": "current", "record_sha256": digest(row)}
            progress.update(first_failure=first, failed_gates=["current"], status="failed")
        if row["kind"] == "voltage_lift_interval_charge":
            row["kind"] = "voltage_lift_interval_certificate"
            row["scientific_progress"] = deepcopy(progress)
    result.update(first_failure=first, last_physically_accepted=None, complete_readback=False,
                  scientific_progress=progress, scientific_status=progress["status"], scientific_pass=False,
                  signed_integral_prefix_C=tag({key: (Fraction(0),)*3 for key in
                                               ("raw_polynomial", "same_state_affine_tangent")}))
    if observation == "sealed-history":
        count = 0
        for row in rows:
            if row["kind"] == "voltage_lift_interval_certificate":
                count += 1
                original = _decode(row["observation"])
                row["kind"] = "voltage_lift_interval_deferred"
                row["observation"] = tag({k: original[k] for k in ("frame_identity", "path_identity", "clock")})
                row["observation"]["charge_observation"] = "pending"
                row["charge_observation"] = {"status": "pending", "deferred_intervals": count}
        result.update(cumulative_absolute_charge_bounds_C=None, cumulative_observation_reference_bounds_C=None,
                      signed_integral_prefix_C=None, charge_observation={"status": "pending", "deferred_intervals": count})
    return request, rows, result


def publish_fixture(folder, data):
    request, rows, result = deepcopy(data)
    writer = HistoryWriter(folder / "NativeHistory.jsonl.gz", encoding="gzip", total_output_bytes=8*1048576)
    for row in rows:
        writer.append(row)
    if result.get("first_failure") is not None:
        writer.publish_first_failure(result["first_failure"])
    writer.finish()
    result["history"] = {"encoding": "gzip", "path": writer.path.name,
                         "logical_bytes": writer.logical_bytes, "encoded_bytes": writer.encoded_bytes,
                         "records": writer.records, "logical_sha256": writer.logical_sha256,
                         "closed": True, "container_complete": True, "error": None}
    result["counts"].update(history_bytes=writer.logical_bytes, history_encoded_bytes=writer.encoded_bytes)
    for name, value in (("NativeRequest.json", request), ("NativeResult.json", result)):
        (folder / name).write_text(json.dumps(value, allow_nan=False))
    return result


def read_fixture(folder, **kwargs):
    return verify_history(folder, max_record_bytes=1048576, max_logical_bytes=4*1048576,
                          max_records=5000, deadline_monotonic=time.perf_counter()+10, **kwargs)


@pytest.mark.parametrize("failure", [False, True])
def test_d14_reader_completes_after_certificate_failure_or_pending(tmp_path, failure):
    data = continued_fixture(failure=failure)
    result = publish_fixture(tmp_path, data)
    actual = read_fixture(tmp_path)
    assert actual["complete_protocol"] and actual["complete_readback"]
    assert actual["history"]["records"] == len(data[1])
    assert actual["history"]["logical_sha256"] == result["history"]["logical_sha256"]
    assert actual["history"]["eof_seen"] and actual["history"]["crc_verified"]
    assert actual["coverage"]["intervals"] == 3
    assert actual["scientific_status"] == ("failed" if failure else "unknown")
    assert actual["scientific_pass"] is False
    assert not (tmp_path / "LastAccepted.json").exists()
    assert result["last_numerical"]["time_s"] == 2 and result["last_physically_accepted"] is None
    assert all(row["scientific_progress"]["first_failure"] == result["first_failure"] for row in data[1]
               if row["kind"] == "voltage_lift_interval_certificate")


@pytest.mark.parametrize("observation", ["online", "sealed-history"])
@pytest.mark.parametrize("target", ["interval", "result"])
@pytest.mark.parametrize("alias", [0, 0.0], ids=["int-zero", "float-zero"])
def test_reader_rejects_rehashed_scientific_progress_alias(tmp_path, observation, target, alias):
    data = continued_fixture(observation=observation)
    row = data[2] if target == "result" else next(row for row in data[1] if row["kind"] in
        ("voltage_lift_interval_certificate", "voltage_lift_interval_deferred"))
    row["scientific_progress"]["passed"] = alias
    publish_fixture(tmp_path, data)  # genuine new logical/container hashes
    with pytest.raises(HistoryVerificationError, match="scientific_progress|deferred_interval_progress"):
        read_fixture(tmp_path)


@pytest.mark.parametrize("mode", ["absent", "null", "false", "valid"])
def test_certificate_policy_producer_reader_modes_agree(monkeypatch, mode):
    from types import SimpleNamespace
    from scripts.benchmarks import coupled_device_prototype as d
    from scripts.benchmarks.native_readback import _Protocol
    request, _, _ = fixture()
    if mode != "absent":
        request["certificate_failure_policy"] = {"null": None, "false": False,
                                                  "valid": certificate_failure_policy()}[mode]
    invalid = mode in ("null", "false")
    seen = []
    original = d._ScientificProgress
    def progress(policy):
        seen.append(policy)
        return original(policy)
    class BeforeBackend(Exception):
        pass
    def stop(*args, **kwargs):
        raise BeforeBackend
    # The actual producer mode-selection is exercised up to its next policy
    # boundary; this synthetic request never reaches imports or F/J/IDA.
    monkeypatch.setattr(d, "validate_voltage_lift_native_request", lambda *args: None)
    monkeypatch.setattr(d, "_ScientificProgress", progress)
    monkeypatch.setattr(d, "voltage_lift_execution_wall_policy", stop)
    mapping = SimpleNamespace(model=None, identity=request["map_identity"])
    admission = {"request_sha256": digest(request), "map_identity": mapping.identity,
                 "voltage_lift_native_authorized": True, "coordinator_message": "msg_manufactured_test_only"}
    with pytest.raises(ValueError if invalid else BeforeBackend):
        d.run_voltage_lift_native_pilot(mapping, (), request, admission, lambda row: None)
    if invalid:
        with pytest.raises(ValueError, match="certificate_policy_binding"):
            _Protocol(request, None)
    else:
        reader = _Protocol(request, None)
        assert (reader.science is not None) == (mode == "valid") == bool(seen)
        if seen:
            assert reader.science.policy == seen[0]


@pytest.mark.parametrize("damage", ["later_interval", "boolean", "negative_error", "density", "owner", "trailer"])
def test_d14_reader_rejects_corruption_after_first_scientific_failure(tmp_path, damage):
    data = continued_fixture()
    rows = data[1]
    intervals = [row for row in rows if row["kind"] == "voltage_lift_interval_certificate"]
    if damage == "later_interval":
        rows.remove(intervals[-1])
    elif damage == "boolean":
        next(row for row in reversed(rows) if row["kind"] == "voltage_lift_current_certificate")["evidence"]["passed"] = False
    elif damage == "negative_error":
        decoded = _decode(intervals[-1]["observation"])
        decoded["ledgers"]["raw_polynomial"]["error_components_C"]["input"] = (Fraction(-1),)*3
        intervals[-1]["observation"] = tag(decoded)
    elif damage == "density":
        row = next(row for row in reversed(rows) if row["kind"] == "voltage_lift_polynomial_domain")
        value = _decode(row["evidence"])
        value["domain_bounds"]["n_m3"] = ((Fraction(-1), Fraction(1)),)*2
        row["evidence"] = tag(value)
    elif damage == "owner":
        row = next(row for row in reversed(rows) if row["kind"] == "voltage_lift_native_observation_packet")
        value = _decode(row["packet"])
        value["owner"] = "foreign-owner"
        row["packet"] = tag(value)
    publish_fixture(tmp_path, data)
    if damage == "trailer":
        path = tmp_path / "NativeHistory.jsonl.gz"
        path.write_bytes(path.read_bytes()[:-5])
    with pytest.raises(HistoryVerificationError):
        read_fixture(tmp_path)


def test_d14_scientific_latch_survives_later_pass_and_event():
    from scripts.benchmarks.coupled_device_prototype import _ScientificProgress
    progress = _ScientificProgress(certificate_failure_policy())
    assert progress.payload()["status"] == "pending"
    progress.note("T1", False, {"interval": 1, "signed": 3})
    first = deepcopy(progress.payload()["first_failure"])
    progress.note("T1", True, {"interval": 2, "signed": -3, "event": "hold"})
    assert progress.payload()["status"] == "failed"
    assert progress.payload()["first_failure"] == first
    assert progress.payload()["passed"] is False


@pytest.mark.parametrize("metrics", [
    {"constraint_potential_bound_V": float("nan")},
    {"constraint_charge_bound_C": float("inf")},
    {"maximum_site_fraction": 1.0},
    {"minimum_f": -.01, "maximum_f": .5},
])
def test_d14_state_safety_is_not_a_scientific_catch(metrics):
    from scripts.benchmarks.coupled_device_prototype import _scientific_sample_safety
    from scripts.benchmarks.contract_prototype import ContractError
    values = {"constraint_potential_bound_V": 0., "constraint_charge_bound_C": 0., "contact_max_relative": 0.}
    values.update(metrics)
    with pytest.raises(ContractError):
        _scientific_sample_safety(values)


@pytest.mark.parametrize("observation", ["online", "sealed-history"])
@pytest.mark.parametrize("fault", [None, "solver", "clock", "initialization"])
def test_d14_actual_producer_loop_continues_science_but_stops_safety(monkeypatch, fault, observation, window=False):
    """Execute the maintained loop with invented buffers; IDA/F/J are absent."""
    from types import SimpleNamespace as NS
    import sys
    import numpy as np
    from scripts.benchmarks import coupled_device_prototype as d, native_observation as o, interval_observation as io
    from scripts.benchmarks.runtime_timing import RuntimeTiming
    request, rows, _ = fixture(lambda r, p: (r.update(certificate_failure_policy=certificate_failure_policy(observation=observation)) or p))
    request["budgets"].update(wall_s=60., rss_bytes=8*1024**3)
    request["interval_observation"]["arithmetic"] = dict(bits=128, evaluations=32, depth=4)
    if window:
        from scripts.benchmarks.physical_tolerance import run_window_policy
        request["execution_window"] = run_window_policy(request, "a")
    segments = tuple(d.ProtocolSegment(**row) for row in request["segments"])
    samples = {(row["segment_id"], float.fromhex(row["history"]["time_hex"]), row["history"]["origin"]): row
               for row in rows if row["kind"] == "voltage_lift_rate_pair"}
    packets = [_decode(row["packet"]) for row in rows if row["kind"] == "voltage_lift_native_observation_packet"]
    domains = [_decode(row["evidence"]) for row in rows if row["kind"] == "voltage_lift_polynomial_domain"]
    intervals = [row for row in rows if row["kind"] == "voltage_lift_interval_charge"]
    proofs = {row["segment_id"]: row["proof"] for row in rows if row["kind"] == "voltage_lift_initialization_input"}
    certificates = {row["sample_record_sha256"]: _decode(row["evidence"]) for row in rows
                    if row["kind"] == "voltage_lift_current_certificate"}
    model = NS(layout=NS(size=6), reference=NS(identity=request["voltage_lift_map"]["physical_reference"]),
               graph=NS(edges=lambda: np.array([[i, i] for i in range(6)]), shape=(6, 6)),
               mass=NS(tocoo=lambda: NS(col=np.arange(4))))
    mapping = NS(model=model, identity=request["map_identity"], frame=None)
    context = NS(request_sha256=d.digest(request), segments=segments, request_copy=lambda: deepcopy(request))
    reference = next(row for row in rows if row["kind"] == "voltage_lift_reference")
    monkeypatch.setattr(d, "validate_voltage_lift_native_request", lambda *a: None)
    monkeypatch.setattr(d, "AffineSamplingContext", lambda *a: context)
    monkeypatch.setattr(d, "VoltageLiftHistory", lambda *a, **k: NS(reference_record=reference["reference"], reference_digest=reference["reference_digest"]))
    monkeypatch.setattr(o, "validate_interval_observation_admission", lambda *a: request["interval_observation"])
    monkeypatch.setattr(o, "require_loaded_observation_backend", lambda *a: None)
    monkeypatch.setattr(o, "prepare_charge_accumulator", lambda *a: None)
    monkeypatch.setitem(sys.modules, "sksundae.ida", NS(IDA=object))
    monkeypatch.setattr(io, "BallIntegrator", lambda **k: NS(calls=[]))
    monkeypatch.setattr(d, "_voltage_lift_guard_applied", lambda *a: None)
    position, current_calls = [-1], [0]

    class Solver:
        def __init__(self, segment): self.segment = segment; self.steps = 0
        def init_step(self, t, z, zdot):
            return NS(t=t, y=z, yp=zdot, success=fault != "initialization", status=0, message="manufactured")
        def step(self, t, **kw):
            assert kw["method"] == "onestep"
            position[0] += 1; self.steps += 1
            end = packets[position[0]]["endpoint"]
            when = end["internal_t"]
            if fault == "clock" and position[0] == 1: when = -.1
            return NS(t=when, y=np.frombuffer(end["raw_y"], dtype="<f8"), yp=np.frombuffer(end["raw_yp"], dtype="<f8"),
                      success=not (fault == "solver" and position[0] == 1), status=end["native_status"], message="manufactured")
        def last_step_snapshot(self): return packets[position[0]]

    monkeypatch.setattr(d, "_voltage_lift_segment_solver", lambda IDA, r, segment, *a, **k: (Solver(segment), None))
    def statistics(solver, segment, ordinal, phase, *a, **kw):
        return {"kind": "native_statistics", "raw_statistics": {"num_steps": solver.steps},
                "work_since_before": {"num_steps": int(phase == "after_onestep")}}
    monkeypatch.setattr(d, "ida_statistics_snapshot", statistics)
    monkeypatch.setattr(d, "voltage_lift_segment_initialization", lambda m, c, s, ordinal, z, previous:
                        (NS(adapter=NS(mapping=m), segment=s), z, np.ones(6), proofs[s.id], None))
    def sample(binding, history, native, previous, *, origin):
        row = deepcopy(samples[(binding.segment.id, native["time"], origin)])
        return NS(identity=row["history"]["point_identity"], time=native["time"]), None, NS(origin=origin), row
    monkeypatch.setattr(d, "voltage_lift_native_sample", sample)

    class Observer:
        def __init__(self, binding, frame, *, arithmetic, previous):
            self.frame, self.binding, self.arithmetic = frame, binding, arithmetic
            self.index = position[0]
            self.prepared = NS(path=NS(identity=domains[self.index]["path_identity"], clock=NS(
                tn=Fraction(frame.snapshot["endpoint"]["internal_t"]), predecessor=Fraction(frame.snapshot["predecessor"]["internal_t"]),
                payload=lambda: _decode(intervals[self.index]["observation"])["clock"])))
        def require_endpoint(self, native, *, predecessor=False):
            o.AcceptedIntervalObserver.require_endpoint_original(self, native, predecessor=predecessor)
        def path_domain_evidence(self, budgets): return domains[self.index]
        def point_evidence(self, pair, native, budget):
            current_calls[0] += 1
            original = samples[(self.binding.segment.id, pair[0].time, pair[2].origin)]
            evidence = deepcopy(certificates[original["record_sha256"]])
            if current_calls[0] == 1:
                evidence.update(passed=False, combined_current_bound_A=(Fraction(2),)*2)
            return evidence
        def reconstruction(self, when):
            return {"time": when, "z": np.full(6, when), "zdot": np.ones(6), "success": True, "status": 0,
                    "coefficient_frame_identity": self.frame.identity, "path_identity": self.prepared.path.identity}
        def charge_evidence(self, left, right, prefixes, **kw):
            assert observation == "online", "deferred producer must not integrate charge"
            evidence = _decode(intervals[self.index]["observation"])
            fresh = {}
            for key, ledger in evidence["ledgers"].items():
                fresh[key] = NS(absolute_defects=ledger["prefix_absolute_defect_C"], reference_errors=ledger["prefix_reference_error_C"])
                for name in ("signed_integral_C",):
                    ledger[name] = tuple(io.Enclosure(Fraction(*v["center"]), Fraction(*v["absolute_error_bound"])) for v in ledger[name])
            self.arithmetic.calls[:] = intervals[self.index]["arithmetic_calls"]
            return fresh, evidence

    Observer.require_endpoint_original = o.AcceptedIntervalObserver.require_endpoint
    monkeypatch.setattr(o, "AcceptedIntervalObserver", Observer)
    paths = [Path(d.__file__), Path(sys.modules[RuntimeTiming.__module__].__file__)]
    admission = {"request_sha256": d.digest(request), "map_identity": mapping.identity,
                 "voltage_lift_native_authorized": True, "coordinator_message": "msg_manufactured_test_only",
                 "source_sha256": {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    saved = []
    result = d.run_voltage_lift_native_pilot(mapping, segments, request, admission, saved.append)
    if fault is None:
        assert result["complete_protocol"] is (not window), result.get("reason")
        assert result["counts"]["native_steps"] == (2 if window else 3)
        assert result["counts"]["initializations"] == (1 if window else 2)
        assert sum(row["kind"] == "requested_sample" for row in saved) == (3 if window else 6)
        assert result["last_numerical"]["time_s"] == (1. if window else 2.)
        if window:
            assert result["complete_run_window"] and result["execution_window"] == request["execution_window"]
        if observation == "sealed-history":
            assert result["charge_observation"] == {"status": "pending", "deferred_intervals": 2 if window else 3}
            assert result["cumulative_absolute_charge_bounds_C"] is None
    else:
        assert result["complete_protocol"] is False
        assert result["counts"]["native_steps"] < 3
        assert result["reason"].startswith({"solver": "voltage_lift_solver_failure", "clock": "voltage_lift_interval_not_monotone",
                                            "initialization": "voltage_lift_initialization_changed"}[fault])
    assert result["last_physically_accepted"] is None
    if fault != "initialization":
        assert result["scientific_status"] == "failed" and result["first_failure"]["gate"] == "current"
        certificates_saved = [row for row in saved if row["kind"] == "voltage_lift_current_certificate"]
        assert result["first_failure"]["record_sha256"] == d.digest(certificates_saved[0])


def test_d25_deferred_reader_preserves_pending_and_consumes_all_intervals(tmp_path):
    data = continued_fixture(observation="sealed-history")
    publish_fixture(tmp_path, data)
    actual = read_fixture(tmp_path)
    assert actual["complete_readback"] and actual["history"]["crc_verified"]
    assert actual["scientific_status"] == "failed"
    assert actual["coverage"]["intervals"] == 3
    seen = []
    class Collector:
        def consume_interval(self, row, left, right, packet, mapping, segment, handoff, science):
            assert row["left"]["record_sha256"] == left["record_sha256"]
            assert row["right"]["record_sha256"] == right["record_sha256"]
            assert packet["kind"] == "voltage_lift_native_observation_packet"
            assert science.summary()["first_failure"] == data[2]["first_failure"]
            seen.append(row["charge_observation"]["deferred_intervals"])
        def summary(self): return {"manufactured_collector": True, "intervals": len(seen)}
    actual = read_fixture(tmp_path, charge_observer=Collector())
    assert seen == [1, 2, 3] and actual["physical_observation"]["intervals"] == 3


@pytest.mark.parametrize("damage", ["zero_error_pass", "count_reset", "missing_suffix"])
def test_d25_deferred_records_never_invent_charge_proof(tmp_path, damage):
    data = continued_fixture(observation="sealed-history")
    intervals = [row for row in data[1] if row["kind"] == "voltage_lift_interval_deferred"]
    if damage == "zero_error_pass": intervals[-1]["observation"]["passed"] = True
    if damage == "count_reset": intervals[-1]["charge_observation"]["deferred_intervals"] = 1
    if damage == "missing_suffix": data[1].remove(intervals[-1])
    publish_fixture(tmp_path, data)
    with pytest.raises(HistoryVerificationError):
        read_fixture(tmp_path)


def test_d25_actual_sealed_observer_matches_online_arb_and_exact_charge_words(monkeypatch):
    """Invented N4 static polynomial; no declared physical case or IDA run."""
    pytest.importorskip("flint")
    import numpy as np
    from dataclasses import asdict
    from scripts.benchmarks import coupled_device_prototype as d, native_observation as o
    from scripts.benchmarks.interval_observation import BallIntegrator, ChargePrefix
    from scripts.benchmarks.native_readback import _ScientificCheck
    # Import only the existing manufactured packet factory; its plan is unused.
    monkeypatch.setenv("REAL_DEVICE_PLAN", str(Path(__file__)))
    from test_native_observation import packet, HEADER
    # This selects the existing static-slab formula tag with invented SI data,
    # a test-source identity and N4 geometry, never a frozen S0 case or run.
    definition = d.SlabDefinition("S0NeutralPublicDeviceV1", str(Path(__file__)), hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), False,
        2., 3., 300., 5., 1., 1., 0., 0., 10., 1., 1., 1., 2., 2., 4., 2., 2., 2., .5, 0., 1.)
    model = d.AffineCoupledSlab(definition, 4)
    mapping = d.AffineVoltageMap(model)
    segment = d.ProtocolSegment("manufactured", 0., 1., (0., 0.), (0., 0.))
    size = model.layout.size
    delta = np.zeros(size); delta[model.layout.offsets["phi_V"].start+1] = float.fromhex("0x1.123456789abcdp-60")
    raw = packet(size=size, previous=0., tn=1., previous_y=np.zeros(size), endpoint_y=delta, previous_rate=delta)
    frame = o.read_native_basis_packet(raw, expected_binding_identity=raw["binding"]["identity"],
                                      expected_header_sha256=HEADER, size=size)
    request = {"schema": "solarlab.voltage-lift-native-request.v1", "case_id": definition.id, "numeric_packet": model.numeric_packet(),
        "voltage_lift_map": mapping.payload(), "map_identity": mapping.identity,
        "z0": [0.]*size, "zdot0": delta.tolist(), "segments": [asdict(segment)],
        "controls": {"nthreads": 1, "max_step": 1.}, "budgets": {"charge_C": 1., "native_steps": 10},
        "observation_times": {segment.id: [0., 1.]}, "quadrature": {str(n): {
            "nodes": [-1+2*(i+1)/(n+1) for i in range(n)], "weights": [2/n]*n} for n in (8, 16, 32)},
        "certificate_failure_policy": certificate_failure_policy(observation="sealed-history")}
    request["interval_observation"] = o.prepare_interval_observation_policy(request,
        binding_identity=raw["binding"]["identity"], header_sha256=HEADER, backend_modules={})
    context = d.AffineSamplingContext(model, request)
    binding = d.VoltageLiftSegmentAdapter(d.VoltageLiftAdapter(mapping), context, segment)
    # The production registration guard admits only declared N8/N16 cases.
    # Substitute that guard for this explicitly invented N4 path, retaining
    # the real map, full-word restore, Arb flux and independent ledger code.
    def manufactured_scope(value):
        assert value.adapter.mapping is mapping and model.intervals == 4 and model.definition.source_path == str(Path(__file__))
        value.context.segment_digest(model, value.segment)
        return mapping
    monkeypatch.setattr(o, "_require_map", manufactured_scope)
    history = d.VoltageLiftHistory(mapping)
    left = history.build_sample(np.zeros(size), 0., [0., 0.], model.reference, delta, [0., 0.], origin="segment_initial")
    right = history.build_sample(delta, 1., [0., 0.], left[0], delta, [0., 0.], origin="stop_output", event_side="left")
    def row(pair):
        return {"record_sha256": digest(pair[3]), "segment_id": segment.id, "history": pair[3]}
    lrow, rrow = row(left), row(right)
    def pointer(record):
        return {"record_sha256": record["record_sha256"], "point_identity": record["history"]["point_identity"],
                "time_s": float.fromhex(record["history"]["time_hex"]), "origin": record["history"]["origin"], "segment_id": segment.id}
    arithmetic = BallIntegrator(**{k: request["interval_observation"]["arithmetic"][k] for k in ("bits", "evaluations", "depth")})
    online = o.AcceptedIntervalObserver(binding, frame, arithmetic=arithmetic)
    prefixes = {k: ChargePrefix.start((1.,)*3) for k in o.CHARGE_LEDGERS}
    _, expected = online.charge_evidence(left, right, prefixes, absolute_error=Fraction(1, 384), charge_budget=1.)
    science = _ScientificCheck(request)
    deferred = {"kind": "voltage_lift_interval_deferred", "segment_id": segment.id,
        "left": pointer(lrow), "right": pointer(rrow), "coefficient_frame_identity": frame.identity,
        "observation": o.observation_record({"frame_identity": frame.identity, "path_identity": online.prepared.path.identity,
                                            "clock": online.prepared.path.clock.payload(), "charge_observation": "pending"}),
        "arithmetic_calls": [], "endpoint_restore": None, "normal_query_performed": False,
        "states_projected": False, "reference_estimates_are_continuum_certificates": False,
        "scientific_progress": science.summary(), "charge_observation": {"status": "pending", "deferred_intervals": 1}}
    emitted = []
    offline = o.SealedChargeObserver(mapping, request, emitted.append, lambda: None)
    offline.consume_interval(deferred, lrow, rrow, {"packet": o.observation_record(raw)}, mapping.payload(), asdict(segment), None, science)
    assert emitted[0]["interval"]["observation"] == o.observation_record(expected)
    assert emitted[0]["interval"]["arithmetic_calls"] == arithmetic.calls
    assert offline.summary()["independent_saved_word_and_ledger_check"]
    assert offline.summary()["scientific_pass"] is False
    # Independent exact affine charge action, with every mapped low word.
    values = [sum((Fraction(float.fromhex(w[i])) for w in right[3]["physical_cumulative_words_hex"]), Fraction())
              for i in range(size)]
    phi = values[model.layout.offsets["phi_V"]]
    truth_left = Fraction(definition.epsilon)*Fraction(definition.area)*(phi[0]-phi[1])/Fraction(float(model.dx[0]))
    bound = expected["ledgers"]["same_state_affine_tangent"]["saved_finite_charge_change_C"][1]
    assert abs(bound.center-truth_left) <= bound.radius
    assert any(float.fromhex(v) for w in right[3]["physical_cumulative_words_hex"][1:] for v in w)


def test_probe_producer_stops_on_declared_side_without_next_initialization(monkeypatch):
    test_d14_actual_producer_loop_continues_science_but_stops_safety(monkeypatch, None, "sealed-history", window=True)


def plan_fixture(folder):
    from scripts.benchmarks.physical_tolerance import PROFILES
    def pin(name, value):
        data = json.dumps(value).encode()
        path = folder / name
        path.write_bytes(data)
        return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    segments = [{"id": key, "start": a, "end": b, "voltage": v, "photons": [0, 0]}
                for key, a, b, v in (("dark_equilibrium_hold", 0., .1, [0, 0]),
                                     ("voltage_ramp", .1, .2, [0, .1]), ("slow_state_hold", .2, 9.2, [.1, .1]))]
    times = {s["id"]: [float(s["start"]+(s["end"]-s["start"])*i/n) for i in range(n+1)]
             for s, n in zip(segments, (32, 32, 64))}
    labels = [{"segment": s["id"], "index": i, "time_hex": value.hex(),
               "event_side": "right" if i == 0 else "left" if i == len(times[s["id"]])-1 else "interior",
               "protocol_initial": ordinal == i == 0,
               "protocol_terminal": ordinal == 2 and i == len(times[s["id"]])-1}
              for ordinal, s in enumerate(segments) for i, value in enumerate(times[s["id"]])]
    cases = []
    for mesh in (8, 16):
        for name, pair in zip(("Coarse", "Main", "Strict"), PROFILES):
            if mesh == 16 and name != "Strict": continue
            profile = dict(zip(("r", "alpha"), pair))
            ancestor = pin(f"N{mesh}{name}.json", {"numeric_packet": {"nodes": mesh+1},
                "physical_tolerance_policy": {"profile": profile}, "segments": segments, "observation_times": times})
            for stage in (("probe", "full") if mesh == 8 else ("full",)):
                cases.append({"id": f"B_N{mesh}_{name}_{stage.title()}", "stage": stage,
                    "mesh_intervals": mesh, "nodes": mesh+1, "coordinates": 5*(mesh+1), "profile": profile,
                    "horizon_s": [0, .2 if stage == "probe" else 9.2], "sample_records": 66 if stage == "probe" else 131,
                    "native_execution_count": 1, "input_ancestor": ancestor})
                cases[-1]["reference_refresh"] = {"enabled": stage == "full", "max_refreshes": 0 if stage == "probe" else 1,
                    "record_schema": "solarlab.reference-state-transfer.v2", "schedule_ref": "/protocol/reference_refresh_schedule"}
    return {"schema": "solarlab.g2-route.run-plan.v3", "version": 3, "status": "design_only_not_admitted",
            "authority": {"scientific_execution_authorized": False}, "implementation": {"available_now": False},
            "source_freeze": {k: None for k in ("future_execution_commit", "environment_lock", "review_record", "frozen_gate_spec")},
            "definitions": {"plan": pin("definition.json", {}), "gate_spec": pin("gate.json", {
                "schema": "solarlab.g2-route.t1-gatespec.v3", "status": "proposed_not_frozen"})},
            "resources": {"WS1_channels": 4, "threads_per_case": 1, "rss_bytes_per_case": 4*1024**3,
                          "WS1_max_declared_rss_bytes": 16*1024**3, "combined_persisted_case_bytes": 4*1024**3,
                          "reader_logical_byte_cap": 32*1024**3},
            "protocol": {"requested_sample_records": labels, "segments": segments, "max_step_s": .02,
                "reference_refresh_schedule": {"time_hex": "0x1.7a4c1c85a8b06p-2", "segment_id": "slow_state_hold",
                    "physical_segment_ordinal": 3, "max_refreshes": 1, "native_api": "init_step",
                    "record_schema": "solarlab.reference-state-transfer.v2"}},
            "cases": cases, "stages": []}


@pytest.mark.parametrize("action", ["run", "observe", "compare"])
def test_route_refuses_candidate_before_runtime_or_namespace(tmp_path, monkeypatch, action):
    from scripts.benchmarks import g2_route
    plan = plan_fixture(tmp_path)
    path = tmp_path / "RunPlan.json"; path.write_text(json.dumps(plan))
    def forbidden(*args): raise AssertionError("unadmitted plan entered runtime")
    monkeypatch.setattr(g2_route, "_runtime", forbidden)
    result = g2_route.validate(path)
    assert result["ready"] is False and "unresolved_environment_lock" in result["blockers"]
    assert sorted(result["sample_records"].values()) == [66]*3+[131]*4
    with pytest.raises(ValueError, match="run_plan_not_admitted"):
        getattr(g2_route, action)(path, "B_N8_Main_Probe") if action != "compare" else g2_route.compare(path)


@pytest.mark.parametrize("damage", ["channel", "labels", "input_pin", "gate_pin", "mesh", "profile"])
def test_route_rejects_wrong_scope_and_references(tmp_path, damage):
    from scripts.benchmarks.g2_route import validate
    plan = plan_fixture(tmp_path)
    if damage == "channel": plan["resources"]["WS1_channels"] = 5
    if damage == "labels": plan["protocol"]["requested_sample_records"].pop()
    if damage == "input_pin": plan["cases"][0]["input_ancestor"]["sha256"] = "0"*64
    if damage == "gate_pin": plan["definitions"]["gate_spec"]["sha256"] = "0"*64
    if damage == "mesh": plan["cases"][-1]["nodes"] = 9
    if damage == "profile": plan["cases"][0]["profile"]["r"] = 1e-7
    path = tmp_path / "RunPlan.json"; path.write_text(json.dumps(plan))
    with pytest.raises(ValueError): validate(path, executable=True)


@pytest.mark.parametrize("damage", ["index", "time", "event_side", "order", "event_pair", "index_bool", "request_samples"])
def test_v3_route_binds_every_ordered_request_sample(tmp_path, damage):
    from scripts.benchmarks.g2_route import validate
    plan = plan_fixture(tmp_path)
    labels = plan["protocol"]["requested_sample_records"]
    if damage == "index": labels[4]["index"] = 999
    elif damage == "time": labels[4]["time_hex"] = float(42).hex()
    elif damage == "event_side": labels[4]["event_side"] = "left"
    elif damage == "order": labels[4], labels[5] = labels[5], labels[4]
    elif damage == "event_pair": labels[32]["event_side"] = labels[33]["event_side"]
    elif damage == "index_bool": labels[0]["index"] = False
    else:
        pin = plan["cases"][0]["input_ancestor"]
        path = Path(pin["path"]); request = json.loads(path.read_text())
        request["observation_times"]["dark_equilibrium_hold"][4] += .0001
        raw = json.dumps(request).encode(); path.write_bytes(raw)
        pin.update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    path = tmp_path / "RunPlan.json"; path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="ordered_sample_tags"):
        validate(path)


@pytest.mark.parametrize("damage", ["unknown_stage", "swapped_ids", "swapped_profile", "float_mesh", "V1", "V2"])
def test_v3_route_binds_id_stage_mesh_profile_and_rejects_old_versions(tmp_path, damage):
    from scripts.benchmarks.g2_route import validate
    plan = plan_fixture(tmp_path)
    if damage == "unknown_stage": plan["cases"][-1]["stage"] = "unrecognized-full-like"
    elif damage == "swapped_ids":
        plan["cases"][0]["id"], plan["cases"][1]["id"] = plan["cases"][1]["id"], plan["cases"][0]["id"]
    elif damage == "swapped_profile": plan["cases"][0]["profile"] = dict(r=1e-6, alpha=1e-10)
    elif damage == "float_mesh": plan["cases"][0]["mesh_intervals"] = 8.0
    elif damage == "V1": plan.update(schema="solarlab.g2-route.run-plan.v1", version=1)
    else: plan.update(schema="solarlab.g2-route.run-plan.v2", version=2)
    path = tmp_path / "RunPlan.json"; path.write_text(json.dumps(plan))
    with pytest.raises(ValueError): validate(path)


def supervision_fixture(tmp_path):
    return {"poll_s": .1, "term_grace_s": 1., "kill_wait_s": 1., "receipt_reserve_s": 1.,
            "rss_bytes": 1024**3, "artifact_bytes": 1024**3,
            "closure_path": str((tmp_path / "Closure.json").resolve()),
            "cancel_path": str((tmp_path / "Cancel").resolve())}


@pytest.mark.parametrize("damage", [None, "negative", "nan", "rss", "artifact", "reserve", "path", "same_path", "missing"])
def test_supervision_fields_checked_during_metadata_validation(tmp_path, damage):
    from scripts.benchmarks.g2_route import validate
    plan = plan_fixture(tmp_path)
    execution = {"wall_s": 10., "supervision": supervision_fixture(tmp_path)}
    plan["cases"][0]["execution"] = execution
    value = execution["supervision"]
    if damage == "negative": value["poll_s"] = -1.
    elif damage == "nan": value["kill_wait_s"] = float("nan")
    elif damage == "rss": value["rss_bytes"] = 5*1024**3
    elif damage == "artifact": value["artifact_bytes"] = 5*1024**3
    elif damage == "reserve": value["receipt_reserve_s"] = 8.
    elif damage == "path": value["closure_path"] = "Closure.json"
    elif damage == "same_path": value["closure_path"] = value["cancel_path"]
    elif damage == "missing": del value["poll_s"]
    path = tmp_path / "RunPlan.json"; path.write_text(json.dumps(plan))
    if damage is None:
        assert not validate(path)["ready"]
    else:
        with pytest.raises(ValueError, match="supervision"):
            validate(path)


def test_supervised_channel_requires_inherited_fd(tmp_path, monkeypatch):
    from scripts.benchmarks.g2_route import _channel
    monkeypatch.delenv("SOLARLAB_G2_ROUTE_CHANNEL_FD", raising=False)
    plan = {"artifacts": {"channel_directory": str(tmp_path)}}
    execution = {"channel": 0, "supervision": supervision_fixture(tmp_path)}
    with pytest.raises(ValueError, match="supervised_channel_fd_required"):
        with _channel(plan, execution):
            pytest.fail("supervised call entered without an inherited lock")
    assert not (tmp_path / "0.lock").exists()


def test_inherited_channel_rejects_other_slot_without_opening(tmp_path, monkeypatch):
    import os
    from scripts.benchmarks.g2_route import _channel
    plan = {"artifacts": {"channel_directory": str(tmp_path)}}
    (tmp_path / "1.lock").touch()
    with (tmp_path / "0.lock").open("a+b") as stream:
        monkeypatch.setenv("SOLARLAB_G2_ROUTE_CHANNEL_FD", str(stream.fileno()))
        def forbidden(*args, **kwargs):
            pytest.fail("inherited branch reopened a file")
        monkeypatch.setattr(Path, "open", forbidden)
        with pytest.raises(ValueError, match="inherited_channel_identity"):
            with _channel(plan, {"channel": 1}):
                pytest.fail("wrong slot accepted")
        assert os.fstat(stream.fileno()).st_ino


def test_child_channel_does_not_unlock_launchers_open_description(tmp_path):
    import fcntl
    import os
    import subprocess
    import sys
    plan = {"status": "admitted", "artifacts": {"channel_directory": str(tmp_path)}}
    execution = {"channel": 0, "supervision": supervision_fixture(tmp_path)}
    with (tmp_path / "0.lock").open("a+b") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        env = dict(os.environ, SOLARLAB_G2_ROUTE_CHANNEL_FD=str(stream.fileno()), PYTHONDONTWRITEBYTECODE="1")
        code = "import json,sys\nfrom scripts.benchmarks.g2_route import _channel\nwith _channel(json.loads(sys.argv[1]),json.loads(sys.argv[2])):\n pass\n"
        child = subprocess.run([sys.executable, "-c", code, json.dumps(plan), json.dumps(execution)],
                               env=env, pass_fds=(stream.fileno(),), capture_output=True, text=True, timeout=10)
        assert child.returncode == 0, child.stderr
        with (tmp_path / "0.lock").open("a+b") as contender:
            with pytest.raises(BlockingIOError):
                fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(stream, fcntl.LOCK_UN)


def test_pending_T1_never_uses_a_loose_L1_upper_bound_as_a_denominator(tmp_path):
    from scripts.benchmarks import g2_route
    # Pure counterexample: a wide upper bound would permit a false pass.
    bound, lower, upper, epsilon = Fraction(2), Fraction(1), Fraction(100), Fraction(1, 10)
    assert bound <= epsilon*upper and bound > epsilon*lower
    plan = plan_fixture(tmp_path)
    plan.update(status="admitted")
    plan["authority"]["scientific_execution_authorized"] = True
    plan["implementation"]["available_now"] = True
    path = tmp_path / "RunPlan.json"; path.write_text(json.dumps(plan))
    result = g2_route.validate(path)
    assert not result["ready"] and "definition_not_frozen_gate_spec" in result["blockers"]
    assert "evaluation_not_admitted" in result["evaluation_blockers"]
    with pytest.raises(ValueError): g2_route.run(path, "B_N8_Main_Probe")


def admitted_metadata_plan(tmp_path,monkeypatch):
    """Complete local metadata fixtures; no real authority or native work."""
    from test_g2_route_bindings import generation_ancestor,frozen_definition,write_json
    from scripts.benchmarks import g2_route as g,g2_route_bindings as b
    plan=plan_fixture(tmp_path)
    plan.update(status="admitted");plan["authority"]["scientific_execution_authorized"]=True
    plan["implementation"]["available_now"]=True
    gate=frozen_definition(tmp_path,"Gate","solarlab.g2-route.t1-gatespec.v3",3)
    method=frozen_definition(tmp_path,"Method","solarlab.g2-reference-method.v1",1)
    plan["definitions"].update(gate_spec=gate,evidence_method_contract=method)
    plan["protocol"]["reference_refresh_schedule"].update(same_owner_generation_increment=1,
        preserve_original_segment_inputs=True,discard_native_multistep_history=True,restored_seed_prefix_credit=False)
    plan["resources"].update(probe_wall_cap_s=3600,full_wall_cap_max_s=43200)
    sources={str(Path(g.__file__).resolve()):hashlib.sha256(Path(g.__file__).read_bytes()).hexdigest()}
    freeze=plan["source_freeze"]
    freeze.update(future_execution_commit="d"*40,generation_plan_sha256="e"*64,frozen_gate_spec=gate,
        frozen_evidence_method_contract=method,source_sha256=sources,method_versions=g.METHODS,
        environment_lock=write_json(tmp_path/"FixtureEnvironment.json",{"fixture_only":True}))
    request_hashes={}
    for case in plan["cases"]:
        ancestor=generation_ancestor(case["mesh_intervals"],case["id"].split("_")[2],monkeypatch)
        case["input_ancestor"]=write_json(tmp_path/(case["id"]+"Ancestor.json"),ancestor)
        request=b.build_effective_request(ancestor,case,source_commit="d"*40,generation_plan_sha256="e"*64,gate_spec=gate)
        request_pin=write_json(tmp_path/(case["id"]+"Request.json"),request)
        request_hashes[case["id"]]=b.digest(request)
        case["execution"]={"request":request_pin,"effective_input_digest":b.digest(request),"channel":0,
            "directory":str(tmp_path/case["id"]),"wall_s":100 if case["stage"] == "probe" else 300,
            "supervision":supervision_fixture(tmp_path),"forecast":{"feasible":True,"estimated_s":100,"slow_hold_model":"fixture"},
            "native_admission":{"fixture_only":True},"max_record_bytes":65536,"integer_io_policy":None}
    review={"fixture_only":True,"accepted":True,"execution_commit":"d"*40,"method_versions":g.METHODS,
        "gate_spec_sha256":gate["sha256"],"evidence_method_contract_sha256":method["sha256"],
        "source_sha256":sources,"request_sha256s":request_hashes}
    freeze["review_record"]=write_json(tmp_path/"FixtureSourceReview.json",review)
    path=tmp_path/"FixtureRunPlan.json";write_json(path,plan)
    return path,plan


def test_run_admission_does_not_require_unreviewed_evaluator(tmp_path,monkeypatch):
    from scripts.benchmarks import g2_route as g
    path,plan=admitted_metadata_plan(tmp_path,monkeypatch)
    value=g.validate(path,executable=True,case_id="B_N8_Main_Probe")
    assert value["run_ready"] and not value["evaluation_ready"]
    with pytest.raises(ValueError,match="evaluation_not_admitted"):
        g.validate(path,executable=True,purpose="evaluate")
    monkeypatch.setattr(g,"_runtime",lambda *args:(_ for _ in ()).throw(AssertionError("unreviewed evaluation reached runtime")))
    with pytest.raises(ValueError,match="evaluation_not_admitted"):
        g.observe(path,"B_N8_Main_Probe")


@pytest.mark.parametrize("damage",[None,"source","review_bool"])
def test_evaluation_requires_its_own_exact_source_review(tmp_path,monkeypatch,damage):
    from scripts.benchmarks import g2_route as g
    from test_g2_route_bindings import write_json
    path,plan=admitted_metadata_plan(tmp_path,monkeypatch)
    root=Path(g.__file__).resolve().parent
    sources={str(root/name):hashlib.sha256((root/name).read_bytes()).hexdigest() for name in g.EVALUATOR_FILES}
    if damage == "source": sources[str(root/"t1_gate.py")]="0"*64
    review={"fixture_only":True,"accepted":1 if damage == "review_bool" else True,"evaluation_source_review":True,
        "source_sha256":sources,"gate_spec_sha256":plan["definitions"]["gate_spec"]["sha256"],
        "evidence_method_contract_sha256":plan["definitions"]["evidence_method_contract"]["sha256"]}
    plan["evaluation_freeze"]={"status":"admitted","source_sha256":sources,
        "review_record":write_json(tmp_path/"FixtureEvaluatorReview.json",review)}
    write_json(path,plan)
    assert g.validate(path,executable=True)["run_ready"]
    if damage:
        with pytest.raises(ValueError,match="evaluation_review_binding|evaluator_source_changed"):
            g.validate(path,executable=True,purpose="evaluate")
    else:
        assert g.validate(path,executable=True,purpose="evaluate")["evaluation_ready"]


def test_run_finishes_with_sealed_producer_output_without_observe(tmp_path,monkeypatch):
    from contextlib import nullcontext
    from scripts.benchmarks import g2_route as g,native_runner
    path,plan=admitted_metadata_plan(tmp_path,monkeypatch)
    context=g.validate(path,executable=True,case_id="B_N8_Main_Probe")
    case=next(c for c in plan["cases"] if c["id"] == "B_N8_Main_Probe")
    request=g._pinned(case["execution"]["request"])
    monkeypatch.setattr(g,"_runtime",lambda *args:(case,request,None,(),lambda:None))
    monkeypatch.setattr(g,"_channel",lambda *args:nullcontext())
    monkeypatch.setattr(g,"_observe",lambda *args:(_ for _ in ()).throw(AssertionError("producer started evaluation")))
    pending={"complete_run_window":True,"complete_protocol":False,"scientific_pass":False,"charge_observation":{"status":"pending"}}
    monkeypatch.setattr(native_runner,"run_recorded",lambda *args,**kwargs:dict(pending))
    assert g.run(path,case["id"]) == pending


@pytest.mark.parametrize("damage",["body","bytes","review_hash","version_alias"])
def test_frozen_definition_is_only_a_status_and_freeze_copy(tmp_path,damage):
    from test_g2_route_bindings import frozen_definition
    from scripts.benchmarks import g2_route as g
    ref=frozen_definition(tmp_path,"Definition",g.T1_SPEC_SCHEMA,3)
    value=g._pinned(ref)
    if damage == "body": value["fixture_only"]=False
    elif damage == "bytes": value["freeze"]["reviewed_bytes"]+=1
    elif damage == "review_hash": value["freeze"]["review_records"][0]["sha256"]="0"*64
    else: value["version"]=3.0
    with pytest.raises(ValueError):g._frozen_definition(value,g.T1_SPEC_SCHEMA,3)


@pytest.mark.parametrize("notes,rule",[("reviewed explanation","prospective only"),
    (["reviewed explanation"],["prospective only"]),("explanation",["rule"]),([],"")])
def test_freeze_explanations_preserve_exact_reviewed_body(tmp_path,notes,rule):
    from scripts.benchmarks import g2_route
    from test_g2_route_bindings import frozen_definition
    ref=frozen_definition(tmp_path,"Definition",g2_route.T1_SPEC_SCHEMA,3)
    value=g2_route._pinned(ref)
    value["freeze"].update(notes=notes,plan_rule=rule)
    assert g2_route._frozen_definition(value,g2_route.T1_SPEC_SCHEMA,3) is value
    value["fixture_only"]=False
    with pytest.raises(ValueError,match="definition_reviewed_body_changed"):
        g2_route._frozen_definition(value,g2_route.T1_SPEC_SCHEMA,3)


@pytest.mark.parametrize("field,bad",[("notes",None),("notes",True),("notes",["text",0]),
    ("plan_rule",{}),("plan_rule",1),("unlisted","")])
def test_freeze_rejects_unlisted_keys_and_nontext_explanations(tmp_path,field,bad):
    from scripts.benchmarks import g2_route
    from test_g2_route_bindings import frozen_definition
    ref=frozen_definition(tmp_path,"Definition",g2_route.T1_SPEC_SCHEMA,3)
    value=g2_route._pinned(ref);value["freeze"][field]=bad
    with pytest.raises(ValueError,match="definition_freeze"):
        g2_route._frozen_definition(value,g2_route.T1_SPEC_SCHEMA,3)


@pytest.mark.parametrize("bad_review",[None,[],False,0,"not a review object"])
def test_nondict_evaluator_review_blocks_only_evaluation(tmp_path,monkeypatch,bad_review):
    from scripts.benchmarks import g2_route
    from test_g2_route_bindings import write_json
    path,plan=admitted_metadata_plan(tmp_path,monkeypatch)
    sources={str(Path(g2_route.__file__).resolve().parent/name):
        hashlib.sha256((Path(g2_route.__file__).resolve().parent/name).read_bytes()).hexdigest()
        for name in g2_route.EVALUATOR_FILES}
    raw=(json.dumps(bad_review)+"\n").encode();review_path=tmp_path/"InvalidReview.json";review_path.write_bytes(raw)
    plan["evaluation_freeze"]={"status":"admitted","source_sha256":sources,
        "review_record":{"path":str(review_path),"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest()}}
    write_json(path,plan)
    value=g2_route.validate(path,executable=True)
    assert value["run_ready"] is True and value["evaluation_ready"] is False
    assert value["evaluation_blockers"] == ["g2_route_evaluation_review_root"]
    monkeypatch.setattr(g2_route,"_runtime",lambda *args:(_ for _ in ()).throw(AssertionError("invalid review reached runtime")))
    with pytest.raises(ValueError,match="evaluation_review_root"):
        g2_route.observe(path,"B_N8_Main_Probe")
