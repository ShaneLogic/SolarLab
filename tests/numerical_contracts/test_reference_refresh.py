"""Opt-in reference refresh contracts using manufactured physical metadata only."""
from dataclasses import asdict
from copy import deepcopy
from fractions import Fraction
import json
from types import SimpleNamespace

import numpy as np
import pytest

from .test_native_transfer import diagnostic_transfer_case
from scripts.benchmarks.contract_prototype import ContractError
from scripts.benchmarks.precision_prototype import PrimitiveExpansion, decode_point, encode_point


@pytest.fixture
def refresh_case(diagnostic_transfer_case):
    c = diagnostic_transfer_case
    primitive = lambda rows: PrimitiveExpansion(tuple(np.array([float.fromhex(x) for x in row]) for row in rows))
    frame = c.d.SegmentAffineFrame(c.mapping.identity, c.d.digest(asdict(c.segment)), 1,
        c.prior.identity, "b"*64, float(c.segment.start),
        primitive(c.seed["frame"]["q0_words_hex"]), primitive(c.seed["frame"]["v0_words_hex"]))
    mapping = c.d.AffineVoltageMap(c.model, frame, c.mapping.mapped_input_profile)
    time = 0.625
    inputs, input_rate = c.segment.inputs(time)
    z, zdot = np.full(10, 2.**-10), np.full(10, -2.**-12)
    point, _ = mapping.trial(z, time, inputs, predecessor=c.saved)
    return SimpleNamespace(**vars(c), old_mapping=mapping, point=point, z=z, zdot=zdot,
                           input_rate=input_rate)


def test_refresh_preserves_full_point_rate_and_inputs(refresh_case, monkeypatch):
    c = refresh_case
    before = encode_point(c.point)
    def forbidden(*args, **kwargs):
        raise AssertionError("refresh must not create a trial or evaluate a physical equation")
    monkeypatch.setattr(c.d.AffineVoltageMap, "trial", forbidden)
    fresh, z, zdot, record = c.d._voltage_lift_frame_handoff(
        c.old_mapping, c.point, c.z, c.zdot, c.input_rate, "c"*64)
    assert record["Point"] == before == encode_point(c.point)
    assert c.point.identity == record["point_identity"]
    assert before["payload"]["authority"]["transition"]["previous_point"] == c.saved.identity
    assert record["input_rates_hex"] == [float(v).hex() for v in c.input_rate]
    assert np.any(c.input_rate) and np.any(c.old_mapping.lift)
    assert record["inputs_hex"] == [float(v).hex() for v in c.point.inputs]
    assert np.all(z == 0) and np.all(zdot == 0)
    assert fresh.model is c.model and fresh.frame.v0.words[1].any()
    assert not record["native_initialization_performed"]
    assert fresh.bind_rate(c.point, z, zdot, c.input_rate).point is c.point


def test_refresh_repeated_low_words_and_signed_zero(refresh_case):
    c = refresh_case
    rates = c.input_rate.copy(); rates[1] = -0.0
    first, z, zdot, one = c.d._voltage_lift_frame_handoff(
        c.old_mapping, c.point, c.z, c.zdot, rates, "c"*64)
    second, _, _, two = c.d._voltage_lift_frame_handoff(first, c.point, z, zdot, rates, "d"*64)
    assert one["Point"] == two["Point"]
    assert one["physical_state_words_hex"] == two["physical_state_words_hex"]
    assert one["physical_rate_words_hex"] == two["physical_rate_words_hex"]
    assert one["input_rates_hex"][1] == two["input_rates_hex"][1] == "-0x0.0p+0"
    assert second.frame.logical_initialization_index == first.frame.logical_initialization_index+1
    assert any(float.fromhex(v) for row in two["physical_state_words_hex"][1:] for v in row)


def test_refresh_rejects_raw_point_mismatch(refresh_case):
    c = refresh_case
    wrong = c.z.copy(); wrong[0] += 2.**-15
    with pytest.raises(ContractError, match="voltage_lift_point_raw_coordinate_mismatch"):
        c.d._voltage_lift_frame_handoff(c.old_mapping, c.point, wrong, c.zdot, c.input_rate, "c"*64)


def test_refresh_four_word_capacity_is_not_truncated(refresh_case):
    c = refresh_case
    tiny = np.nextafter(0., 1.)
    raw = PrimitiveExpansion(tuple(np.full(10, x) for x in (1., 2.**-60, 2.**-120, 2.**-180)))
    with pytest.raises(ContractError, match="primitive_expansion_capacity_exceeded"):
        raw.add(np.full(10, 2.**-240))
    with pytest.raises((ArithmeticError, ContractError)):
        c.d._frame_scale(np.full(10, tiny), np.full(10, 0.5))
    assert sum((Fraction(float(w[0])) for w in raw.words), Fraction()) == sum(
        map(Fraction, (1., 2.**-60, 2.**-120, 2.**-180)))


@pytest.fixture
def admitted_refresh(refresh_case, monkeypatch):
    """Real transfer/policy code with explicit manufactured acceptance receipts."""
    from scripts.benchmarks.native_observation import observation_record
    c = refresh_case
    request = c.request
    request.pop("diagnostic_interval")
    request.update(segments=[asdict(c.segment)], numeric_packet={
        "layout_identity": c.model.layout.identity, "initial_reference": encode_point(c.model.reference),
        "variable_offsets": {name: [value.start, value.stop] for name, value in c.model.layout.offsets.items()}},
        controls={"calc_initcond": None, "rtol": 1e-6, "atol": [1e-10]*10, "first_step": 0.0,
                  "nonlin_guard": "first-correction-wrms-v1", "nonlin_trace_capacity": 8},
        budgets={"current_A": 1e-9, "constraint_potential_V": 1e-8, "constraint_charge_C": 1e-15,
                 "contact_relative": 1e-8, "ion_inventory_relative": 1e-8,
                 "maximum_site_fraction": 1.}, observation_times={c.segment.id: [.5, .75]}, quadrature={})
    request["reference_refresh_policy"] = c.d._reference_refresh_policy(request, c.point.time)
    c.context.request_sha256 = c.d.digest(request)
    binding = c.d.VoltageLiftSegmentAdapter(c.d.VoltageLiftAdapter(c.old_mapping), c.context, c.segment)
    monkeypatch.setattr(c.d, "voltage_lift_rate_projection", lambda *args: {"manufactured_no_projection": True})
    history = c.d.VoltageLiftHistory(c.old_mapping,
        parent_reference=c.d.VoltageLiftHistory(c.mapping).reference_record)
    point, _, _, h = history.build_sample(c.z, c.point.time, c.point.inputs, c.saved, c.zdot, c.input_rate, origin="native")
    assert point.identity == c.point.identity
    sample = {"kind": "voltage_lift_rate_pair", "request_sha256": c.context.request_sha256,
        "map_identity": c.old_mapping.identity, "history": h, "segment_id": c.segment.id,
        "segment_sha256": c.d.digest(asdict(c.segment)),
        "state_checks": {"passed": True, "state_identity": point.identity, "state_changed": False,
                         "constraint_potential_bound_V": 0., "constraint_charge_bound_C": 0., "contact_max_relative": 0.,
                         "ion_inventory_relative": 0., "maximum_site_fraction": .5, "minimum_f": .25, "maximum_f": .25}}
    current = {"high_hex": [0.0.hex()]*2, "low_hex": [0.0.hex()]*2}
    sample.update(raw={"physical_rate": {"identity": c.old_mapping.bind_rate(point, c.z, c.zdot, c.input_rate).identity},
                       "total_inward": current}, physical_tangent={"total_inward": deepcopy(current)})
    sample["record_sha256"] = c.d.digest(sample)
    frame_id, path_id = "e"*64, "f"*64
    interval = {"kind": "voltage_lift_interval_charge", "segment_id": c.segment.id,
        "right": {"record_sha256": sample["record_sha256"], "point_identity": point.identity,
                  "time_s": point.time, "origin": "native", "segment_id": c.segment.id},
        "coefficient_frame_identity": frame_id, "normal_query_performed": False, "states_projected": False,
        "observation": observation_record({"passed": True, "path_identity": path_id})}
    domain = {"kind": "voltage_lift_polynomial_domain",
        "evidence": observation_record({"passed": True, "path_identity": path_id, "checks": {"manufactured_inventory": True}})}
    certificate = {"kind": "voltage_lift_current_certificate", "sample_record_sha256": sample["record_sha256"],
        "evidence": observation_record({"passed": True, "path_identity": path_id, "point_identity": point.identity,
            "native_rate_or_state_changed": False, "combined_current_bound_A": (Fraction(0), Fraction(0))})}
    ledger = observation_record({"prefixes": {"raw_polynomial": {"absolute_defects": (Fraction(1, 10),)*3,
        "reference_errors": (Fraction(1, 100),)*3}}, "sample_cursor": 1,
        "upper_accumulator": {"index": 7, "upper_numerators": (11, 13), "rounded_terms": (2, 3)},
        "last_accepted": interval["right"], "physical_reference_identity": c.model.reference.identity, "signed_prefix": None})
    endpoint = {"internal_t": point.time,
                "identity": ("manufactured-native-owner", 1, 7, point.time.hex(), point.time.hex(), "a"*64),
                "nsteps": 7,
                "raw_y": c.z.tobytes(), "raw_yp": c.zdot.tobytes()}
    frame = SimpleNamespace(identity=frame_id, owner="manufactured-native-owner", generation=1,
        binding_identity="b"*64, snapshot={"endpoint": endpoint, "native_before": {"nsteps": 7}})
    return SimpleNamespace(**vars(c), binding=binding, sample=sample, interval=interval, domain=domain,
                           certificate=certificate, ledger=ledger, frame=frame)


def promote_reference_transfer_v2(c, *, failed=True, observation="online"):
    from scripts.benchmarks.physical_tolerance import certificate_failure_policy
    c.request["certificate_failure_policy"] = certificate_failure_policy(observation=observation)
    c.request["reference_refresh_policy"] = c.d._reference_refresh_policy(c.request, c.point.time)
    c.context.request_sha256 = c.d.digest(c.request)
    c.binding = c.d.VoltageLiftSegmentAdapter(c.d.VoltageLiftAdapter(c.old_mapping), c.context, c.segment)
    c.sample["request_sha256"] = c.context.request_sha256
    if failed:
        c.sample["state_checks"].update(passed=False, contact_max_relative=2e-8)
    c.sample["record_sha256"] = c.d.digest({k: v for k, v in c.sample.items() if k != "record_sha256"})
    c.interval["right"]["record_sha256"] = c.sample["record_sha256"]
    c.interval["kind"] = "voltage_lift_interval_certificate"
    c.certificate["sample_record_sha256"] = c.sample["record_sha256"]
    progress = c.d._ScientificProgress(c.request["certificate_failure_policy"])
    progress.note("state_accuracy", not failed, c.sample)
    c.interval["scientific_progress"] = progress.payload()
    c.ledger.update(scientific_progress=progress.payload(), last_accepted=None)
    if observation == "sealed-history":
        c.interval.update(kind="voltage_lift_interval_deferred",
                          observation={"path_identity": "f"*64, "charge_observation": "pending"},
                          charge_observation={"status": "pending", "deferred_intervals": 7})
        c.ledger.update(prefixes=None, physical_charge=None, upper_accumulator=None, signed_prefix=None,
                        charge_observation={"status": "pending", "deferred_intervals": 7})
    return c


def prepare(c):
    return c.d.prepare_voltage_lift_reference_refresh(c.binding, c.point, c.z, c.zdot,
        sample=c.sample, interval=c.interval, domain=c.domain, certificate=c.certificate,
        ledger=c.ledger, native_frame=c.frame)


class NativeSpy:
    """Only the public statistics/init_step interface; no native backend."""
    def __init__(self, c, mapping, installed, damage=None):
        self.c, self.mapping, self.installed, self.damage = c, mapping, installed, damage
        self.calls = 0

    def statistics(self):
        c = self.c
        frame = c.old_mapping.frame if not self.calls else self.mapping.frame
        gen = 1 if not self.calls or self.damage == "generation" else 2
        owner = "foreign" if not self.calls and self.damage == "old_owner" else c.frame.owner
        state = {"schema": "sksundae.ida.parent-affine-weight-state.v1", "owner": owner, "generation": gen,
            "frame_identity": ("x"*64 if self.damage == ("old_frame" if not self.calls else "frame") else frame.identity),
            "size": 10, "installed": True, "setter": "IDAWFtolerances", "setter_status": 0,
            "t0_hex": frame.t0.hex(), "q0_words": tuple(w.tobytes() for w in frame.q0.words),
            "v0_words": tuple(w.tobytes() for w in frame.v0.words), "callback_calls": 0 if self.calls else 1,
            "callback_status": None if self.calls else 0, "rtol_hex": c.request["controls"]["rtol"].hex(),
            "atol": np.array(c.request["controls"]["atol"]).tobytes(), "raw_tolerance_getter": False,
            "weight_rounding": frame.payload()["weight_rounding"], "basis_time_hex": None, "basis_u": None,
            "rounded_parent": None, "computed_weights": None, "failure_component": None}
        guard = {"installed_at_capture": True, "applied_policy": "first-correction-wrms-v1",
            "requested_policy": "first-correction-wrms-v1", "delegate_data_verified": True,
            "newton_ops_verified": True, "complete": True, "capacity": 8, "owner": owner,
            "generation": gen-1 if self.calls and self.damage == "guard_generation" else gen, "build_identity": "f"*64}
        return {**{k: 0 for k in c.d._IDA_COUNTER_FIELDS}, "num_steps": 0 if self.calls else 7,
            "current_time": c.point.time, "observation_owner": owner, "observation_generation": gen,
            "parent_weight_state": state, "nonlinear_guard": guard, "nonlinear_control_state": {"build_identity": "f"*64}}

    def init_step(self, time, z, zdot, *, parent_weight_frame):
        assert self.installed[-1].adapter.mapping is self.mapping
        assert parent_weight_frame == self.mapping.frame.weight_option()
        assert time == self.c.point.time and not np.any(z) and not np.any(zdot)
        self.calls += 1
        values = z.copy()
        if self.damage == "state": values[0] = 2.**-20
        return SimpleNamespace(t=time, y=values, yp=zdot.copy(), success=True, status=0, message="manufactured init")


def initialized(c, damage=None):
    fresh, zero, _, prepared = prepare(c)
    installed = []
    solver = NativeSpy(c, fresh, installed, damage)
    result = c.d.apply_voltage_lift_reference_refresh(solver, c.binding, fresh, zero, prepared, installed.append)
    return fresh, prepared, solver, installed, result


def independent_reader(c):
    from scripts.benchmarks.native_readback import SegmentFrameCheck, reference_refresh_policy
    # Existing parent-policy checks have separate fixtures; this component
    # starts from an already verified old map and uses the real new policy.
    reader = object.__new__(SegmentFrameCheck)
    reader.request, reader.request_id = c.request, c.context.request_sha256
    reader.enabled, reader.size, reader.input_words = True, 10, 12
    reader.base = c.mapping.payload()
    reader.active, reader.current_map = c.old_mapping.frame.payload(), c.old_mapping.payload()
    reader.segment = asdict(c.segment)
    reader.refresh_policy, reader.refreshes, reader.frames = reference_refresh_policy(c.request), [], []
    return reader


@pytest.mark.parametrize("failed", [False, True])
@pytest.mark.parametrize("observation", ["online", "sealed-history"])
def test_v2_transfer_preserves_failed_or_pending_science_and_exact_words(admitted_refresh, failed, observation):
    from scripts.benchmarks.native_readback import _decode
    c = promote_reference_transfer_v2(admitted_refresh, failed=failed, observation=observation)
    old_point, old_ledger = encode_point(c.point), deepcopy(c.ledger)
    fresh, prepared, solver, installed, result = initialized(c)
    binding, returned, statistics, complete = result
    reader = independent_reader(c)
    actual = reader.prepare_refresh(prepared, asdict(c.segment), 1, c.sample, c.interval,
                                    c.domain, c.certificate, _decode(c.ledger))
    reader.complete_refresh(prepared, complete, statistics)
    assert actual == fresh.payload()
    assert complete["schema"] == "solarlab.reference-state-transfer.v2"
    assert "acceptance" not in complete and "acceptance_sha256" not in complete["transfer"]
    assert complete["state_transfer"]["schema"] == "solarlab.numeric-reference-transfer-evidence.v2"
    assert complete["state_transfer"]["scientific_progress"]["status"] == ("failed" if failed else "pending")
    assert complete["ledger"] == old_ledger and complete["transfer"]["Point"] == old_point == encode_point(c.point)
    assert complete["transfer"]["physical_state_words_hex"] == c.sample["history"]["physical_cumulative_words_hex"]
    assert complete["transfer"]["physical_rate_words_hex"] == c.sample["history"]["physical_rate_words_hex"]
    assert solver.calls == 1 and complete["new_epoch"]["generation"] == 2
    assert complete["native_multistep_history_discarded"] is True


@pytest.mark.parametrize("observation", ["online", "sealed-history"])
@pytest.mark.parametrize("target", ["interval", "state_transfer", "ledger"])
@pytest.mark.parametrize("alias", [0, 0.0], ids=["int-zero", "float-zero"])
def test_v2_reader_rejects_rehashed_progress_alias(admitted_refresh, observation, target, alias):
    from scripts.benchmarks.native_readback import HistoryVerificationError, _decode
    c = promote_reference_transfer_v2(admitted_refresh, observation=observation)
    if target == "interval":
        c.interval["scientific_progress"]["passed"] = alias
    _, _, _, prepared = prepare(c)
    if target in ("state_transfer", "ledger"):
        prepared[target]["scientific_progress"]["passed"] = alias
        if target == "ledger":
            prepared["ledger_sha256"] = c.d.digest(prepared["ledger"])
        prepared["record_sha256"] = c.d.digest({k: v for k, v in prepared.items() if k != "record_sha256"})
    reader = independent_reader(c)
    with pytest.raises(HistoryVerificationError, match="scientific_progress|scientific_acceptance"):
        reader.prepare_refresh(prepared, asdict(c.segment), 1, c.sample, c.interval,
                               c.domain, c.certificate, _decode(c.ledger))


@pytest.mark.parametrize("mode", ["absent", "null", "false", "valid"])
def test_refresh_certificate_policy_mode_is_validated_on_both_ends(admitted_refresh, mode):
    from scripts.benchmarks.native_readback import reference_refresh_policy
    from scripts.benchmarks.physical_tolerance import certificate_failure_policy
    c = admitted_refresh
    if mode != "absent":
        c.request["certificate_failure_policy"] = {"null": None, "false": False,
                                                  "valid": certificate_failure_policy()}[mode]
    if mode in ("null", "false"):
        with pytest.raises(ValueError, match="certificate_policy_binding"):
            c.d._reference_refresh_policy(c.request, c.point.time)
        with pytest.raises(ValueError, match="certificate_policy_binding"):
            reference_refresh_policy(c.request)
    else:
        policy = c.d._reference_refresh_policy(c.request, c.point.time)
        c.request["reference_refresh_policy"] = policy
        assert reference_refresh_policy(c.request) == policy
        assert policy["schema"] == ("solarlab.reference-state-transfer-policy.v2" if mode == "valid"
                                    else "solarlab.accepted-reference-refresh-policy.v1")


@pytest.mark.parametrize("target", ["state_transfer", "ledger"])
@pytest.mark.parametrize("value", [None, False])
def test_v2_reader_rejects_malformed_progress_container(admitted_refresh, target, value):
    from scripts.benchmarks.native_readback import HistoryVerificationError, _decode
    c = promote_reference_transfer_v2(admitted_refresh)
    _, _, _, prepared = prepare(c)
    prepared[target] = value
    if target == "ledger":
        prepared["ledger_sha256"] = c.d.digest(value)
    prepared["record_sha256"] = c.d.digest({k: v for k, v in prepared.items() if k != "record_sha256"})
    with pytest.raises(HistoryVerificationError, match="scientific_progress_type"):
        independent_reader(c).prepare_refresh(prepared, asdict(c.segment), 1, c.sample, c.interval,
                                              c.domain, c.certificate, _decode(c.ledger))


@pytest.mark.parametrize("damage", ["state", "generation", "old_owner", "guard_generation"])
def test_v2_scientific_failure_does_not_hide_native_transfer_failure(admitted_refresh, damage):
    c = promote_reference_transfer_v2(admitted_refresh)
    with pytest.raises(ContractError):
        initialized(c, damage=damage)


def test_v2_refresh_policy_binds_each_actual_request_not_shared_N16_ancestry(admitted_refresh):
    from scripts.benchmarks.native_readback import reference_refresh_policy, HistoryVerificationError
    c = promote_reference_transfer_v2(admitted_refresh)
    original = c.request["reference_refresh_policy"]
    for key, replacement in (("map_identity", "7"*64), ("controls", {**c.request["controls"], "rtol": 1e-5}),
                             ("budgets", {**c.request["budgets"], "current_A": 2e-9})):
        request = deepcopy(c.request); request[key] = replacement
        with pytest.raises(HistoryVerificationError): reference_refresh_policy(request)
        request["reference_refresh_policy"] = c.d._reference_refresh_policy(request, c.point.time)
        current = reference_refresh_policy(request)
        assert current != original and current["time_hex"] == original["time_hex"]
        assert current["trigger"] == "exact_domain_safe_native_endpoint"


def test_refresh_callback_and_weight_epoch_are_coherent(admitted_refresh):
    c = admitted_refresh
    before_point, before_ledger = encode_point(c.point), deepcopy(c.ledger)
    fresh, prepared, solver, installed, result = initialized(c)
    binding, returned, statistics, complete = result
    assert solver.calls == 1 and len(installed) == 2 and installed[-1] is binding
    assert complete["new_epoch"] == {"owner": c.frame.owner, "generation": 2, "nsteps": 0}
    assert c.binding.numeric_handoff_json is None and binding.numeric_handoff_json is not None
    assert complete["ledger"] == c.ledger == before_ledger
    assert encode_point(c.point) == before_point and not np.any(returned["z"])
    assert fresh.frame.v0.words[1].any()


@pytest.mark.parametrize("damage", ["native_only", "pending", "failed", "point", "budget", "current", "rate"])
def test_refresh_requires_complete_acceptance(admitted_refresh, damage):
    c = admitted_refresh
    if damage == "native_only": c.interval.clear(); c.interval["passed"] = True
    if damage == "pending": c.interval["observation"]["passed"] = False
    if damage == "failed": c.domain["evidence"]["passed"] = False
    if damage == "point": c.sample["state_checks"]["state_identity"] = "d"*64
    if damage == "budget": c.sample["state_checks"]["constraint_charge_bound_C"] = 1.
    if damage == "current": c.certificate["sample_record_sha256"] = "d"*64
    if damage == "rate": c.sample["history"]["raw_solver_zdot_hex"][0] = (2.**-20).hex()
    c.sample["record_sha256"] = c.d.digest({k: v for k, v in c.sample.items() if k != "record_sha256"})
    with pytest.raises((ContractError, KeyError)):
        prepare(c)


@pytest.mark.parametrize("damage", ["old_owner", "old_frame", "generation", "frame", "state", "guard_generation"])
def test_refresh_rejects_stale_native_context(admitted_refresh, damage):
    c = admitted_refresh
    fresh, zero, _, prepared = prepare(c)
    installed = []; solver = NativeSpy(c, fresh, installed, damage)
    with pytest.raises(ContractError) as caught:
        c.d.apply_voltage_lift_reference_refresh(solver, c.binding, fresh, zero, prepared, installed.append)
    assert hasattr(caught.value, "reference_refresh_evidence")
    assert solver.calls == (0 if damage.startswith("old_") else 1)


def test_refresh_independent_full_word_and_ledger_check(admitted_refresh):
    from scripts.benchmarks.native_readback import _decode
    c = admitted_refresh
    fresh, prepared, _, _, result = initialized(c)
    _, _, stats, complete = result
    reader = independent_reader(c)
    assert reader.prepare_refresh(prepared, asdict(c.segment), 1, c.sample, c.interval, c.domain,
        c.certificate, _decode(c.ledger)) == fresh.payload()
    reader.weight_state(stats["raw_statistics"]["parent_weight_state"], owner=c.frame.owner, generation=2, initial=True)
    reader.complete_refresh(prepared, complete, stats)
    assert len(reader.refreshes) == 1 and reader.frames[0]["physical_segment_ordinal"] == 1


@pytest.mark.parametrize("damage", ["erased_prefix", "doubled_prefix", "cursor", "rate_word", "Point", "frame", "acceptance"])
def test_refresh_reader_rejects_corruption(admitted_refresh, damage):
    from scripts.benchmarks.native_readback import _decode, HistoryVerificationError
    c = admitted_refresh
    _, _, _, record = prepare(c)
    if damage in {"erased_prefix", "doubled_prefix"}:
        record["ledger"]["prefixes"]["raw_polynomial"]["absolute_defects"] = {"tuple": [{"rational": [0 if damage=="erased_prefix" else 1, 5]}]*3}
        record["ledger_sha256"] = c.d.digest(record["ledger"])
    if damage == "cursor": record["ledger"]["sample_cursor"] += 1; record["ledger_sha256"] = c.d.digest(record["ledger"])
    if damage == "rate_word": record["transfer"]["physical_rate_words_hex"][1][0] = (2.**-80).hex()
    if damage == "Point": record["transfer"]["Point"]["payload"]["identity"] = "c"*64
    if damage == "frame": record["transfer"]["new_frame"]["q0_words_hex"][0][0] = 0.0.hex()
    if damage == "acceptance": record["acceptance"] = {"passed": True}
    record["record_sha256"] = c.d.digest({k: v for k, v in record.items() if k != "record_sha256"})
    with pytest.raises((HistoryVerificationError, KeyError)):
        independent_reader(c).prepare_refresh(record, asdict(c.segment), 1, c.sample, c.interval, c.domain, c.certificate, _decode(c.ledger))


@pytest.mark.parametrize("serialized", [False, True], ids=["memory", "json"])
def test_refresh_clock_and_native_epoch_guards(admitted_refresh, serialized):
    from scripts.benchmarks.interval_observation import AcceptedClock
    from scripts.benchmarks.native_observation import require_frame_successor
    c = admitted_refresh
    fresh, _, _, _, result = initialized(c)
    complete = result[-1]
    if serialized:
        complete = json.loads(json.dumps(complete))
    before = AcceptedClock(.5, .625, .125, 1, 7, 2, .5, .75, c.segment.id, owner=c.frame.owner)
    after = AcceptedClock(.625, .6875, .0625, 2, 1, 1, .5, .75, c.segment.id, owner=c.frame.owner)
    before.require_successor(after, numeric_handoff=complete)
    with pytest.raises(ContractError): before.require_successor(after)
    for start in (.624, .626):
        with pytest.raises(ContractError, match="gap_or_overlap"):
            before.require_successor(AcceptedClock(start, .6875, .0625, 2, 1, 1, .5, .75, c.segment.id, owner=c.frame.owner), numeric_handoff=complete)
    zero = np.zeros(10)
    current = SimpleNamespace(identity="c"*64, owner=c.frame.owner, generation=2, binding_identity=c.frame.binding_identity,
        snapshot={"predecessor": {"internal_t": c.point.time, "nsteps": 0, "raw_y": zero.tobytes(), "raw_yp": zero.tobytes()},
                  "native_before": {"nsteps": 1}})
    require_frame_successor(c.frame, current, previous_segment=c.segment, current_segment=c.segment,
        previous_mapping=c.old_mapping, current_mapping=fresh, numeric_handoff=complete)
    with pytest.raises(ContractError):
        require_frame_successor(c.frame, current, previous_segment=c.segment, current_segment=c.segment,
            previous_mapping=c.old_mapping, current_mapping=fresh)
    damaged = deepcopy(complete); damaged["transfer"]["physical_rate_words_hex"][1][0] = (2.**-80).hex()
    damaged["record_sha256"] = c.d.digest({k: v for k, v in damaged.items() if k != "record_sha256"})
    with pytest.raises(ContractError):
        require_frame_successor(c.frame, current, previous_segment=c.segment, current_segment=c.segment,
            previous_mapping=c.old_mapping, current_mapping=fresh, numeric_handoff=damaged)


@pytest.mark.parametrize("damage", [None, "v1_disguise", "no_homomorphism", "science_erased", "words"])
def test_v2_clock_and_frame_require_distinct_numerical_evidence(admitted_refresh, damage):
    from scripts.benchmarks.interval_observation import AcceptedClock
    from scripts.benchmarks.native_observation import require_frame_successor
    c = promote_reference_transfer_v2(admitted_refresh)
    fresh, _, _, _, (_, _, _, complete) = initialized(c)
    before = AcceptedClock(.5, .625, .125, 1, 7, 2, .5, .75, c.segment.id, owner=c.frame.owner)
    after = AcceptedClock(.625, .6875, .0625, 2, 1, 1, .5, .75, c.segment.id, owner=c.frame.owner)
    zero = np.zeros(10)
    current = SimpleNamespace(identity="c"*64, owner=c.frame.owner, generation=2, binding_identity=c.frame.binding_identity,
        snapshot={"predecessor": {"internal_t": c.point.time, "nsteps": 0, "raw_y": zero.tobytes(), "raw_yp": zero.tobytes()},
                  "native_before": {"nsteps": 1}})
    if damage == "v1_disguise": complete["schema"] = "solarlab.accepted-reference-handoff.v1"
    if damage == "no_homomorphism": complete["transfer"].pop("state_transfer_sha256")
    if damage == "science_erased": complete["state_transfer"]["scientific_progress"] = {"status": "passed"}
    if damage == "words": complete["transfer"]["physical_rate_words_hex"][1][0] = (2.**-80).hex()
    complete["record_sha256"] = c.d.digest({k: v for k, v in complete.items() if k != "record_sha256"})
    if damage in {None, "words"}:
        before.require_successor(after, numeric_handoff=complete)
    else:
        with pytest.raises((ContractError, KeyError)):
            before.require_successor(after, numeric_handoff=complete)
    def frame_check():
        require_frame_successor(c.frame, current, previous_segment=c.segment, current_segment=c.segment,
            previous_mapping=c.old_mapping, current_mapping=fresh, numeric_handoff=complete)
    if damage is None:
        frame_check()
    else:
        with pytest.raises((ContractError, KeyError)):
            frame_check()


def test_refresh_legacy_policy_absence_and_single_cap(admitted_refresh):
    from scripts.benchmarks.native_readback import reference_refresh_policy, HistoryVerificationError
    c = admitted_refresh
    legacy = dict(c.request); legacy.pop("reference_refresh_policy")
    assert reference_refresh_policy(legacy) is None
    assert reference_refresh_policy(c.request) == c.request["reference_refresh_policy"]
    for key, value in (("max_refreshes", 2), ("time_hex", c.segment.start.hex()), ("controls_sha256", "a"*64)):
        bad = deepcopy(c.request); bad["reference_refresh_policy"][key] = value
        with pytest.raises(HistoryVerificationError): reference_refresh_policy(bad)


class Rows:
    def __init__(self, values): self.values = list(values)
    def take(self, kind=None):
        from scripts.benchmarks.native_readback import _require
        _require(bool(self.values), "manufactured_stream_EOF")
        value = self.values.pop(0)
        _require(kind is None or value["kind"] == kind, "manufactured_unexpected_record")
        return value


def protocol_for_refresh(c, records):
    from scripts.benchmarks.native_readback import _Protocol, _decode
    p = object.__new__(_Protocol)
    p.request, p.request_id, p.rows = c.request, c.context.request_sha256, Rows(records)
    p.segment, p.ordinal, p.numeric_ordinal = asdict(c.segment), 1, 1
    p.segment_frame, p.mapping, p.map_id = independent_reader(c), c.old_mapping.payload(), c.old_mapping.identity
    ledger = _decode(c.ledger)
    p.prefix = {k: (v["absolute_defects"], v["reference_errors"]) for k, v in ledger["prefixes"].items()}
    p.charge_refinement = SimpleNamespace(summary=lambda: ledger["upper_accumulator"])
    p.physical_charge = None
    p.signed_integrals = None
    p.frame_id, p.size, p.jacobians, p.diagnostic = c.frame.identity, 10, 0, None
    p.refresh_anchor, p.refresh_binding_identity = None, None
    return p


def rehash_embedded_point(c, prepared, complete, point):
    """Keep every enclosing checksum coherent for the malformed-codec tests."""
    point["sha256"] = c.d.digest(point["payload"])
    for record in (prepared, complete):
        record["transfer"]["Point"] = deepcopy(point)
    prepared["record_sha256"] = c.d.digest({k: v for k, v in prepared.items() if k != "record_sha256"})
    complete["prepared_sha256"] = prepared["record_sha256"]
    complete["record_sha256"] = c.d.digest({k: v for k, v in complete.items() if k != "record_sha256"})


@pytest.mark.parametrize("damage", ["ancestry", "field_projection"])
def test_reader_rejects_rehashed_invalid_complete_point(admitted_refresh, damage):
    """Promoted independent R1 failures; all original history and gates remain."""
    from scripts.benchmarks.native_readback import HistoryVerificationError
    c = admitted_refresh
    _, prepared, _, _, (_, _, statistics, complete) = initialized(c)
    bad = deepcopy(prepared["transfer"]["Point"])
    if damage == "ancestry":
        bad["payload"]["authority"]["transition"]["previous_point"] = "0"*64
    else:
        field = bad["payload"]["fields"]["n_m3"]
        field["high"][0] = (float.fromhex(field["high"][0])+1.).hex()
    rehash_embedded_point(c, prepared, complete, bad)
    assert bad["payload"]["identity"] == c.sample["history"]["point_identity"]
    with pytest.raises(ContractError):
        decode_point(bad, c.model.layout)
    protocol = protocol_for_refresh(c, [prepared, statistics, complete])
    before = deepcopy(protocol.prefix), deepcopy(protocol.segment_frame.active)
    frame = dict(c.frame.snapshot, owner=c.frame.owner, generation=c.frame.generation)
    with pytest.raises(HistoryVerificationError):
        protocol._consume_reference_refresh(c.sample, c.interval, c.domain, c.certificate, 1, frame,
                                            complete["before_statistics"]["raw_statistics"])
    assert (protocol.prefix, protocol.segment_frame.active) == before
    assert protocol.numeric_ordinal == 1 and protocol.refresh_anchor is None


@pytest.mark.parametrize("damage", ["outer_member", "schema", "low_word", "field_shape",
                                    "noncanonical_zero", "paired_previous", "reference_zero_sign",
                                    "payload_type", "rehash_authority"])
def test_refresh_reader_rejects_rehashed_malformed_codec(admitted_refresh, damage):
    from scripts.benchmarks.native_readback import HistoryVerificationError
    c = admitted_refresh
    _, prepared, _, _, (_, _, statistics, complete) = initialized(c)
    bad = deepcopy(prepared["transfer"]["Point"])
    p = bad["payload"]
    if damage == "outer_member": bad["unbound"] = True
    if damage == "schema": p["schema"] = "solarlab.precision-point.v1"
    if damage == "low_word":
        field = p["fields"]["n_m3"]
        field["low"][0] = (float.fromhex(field["low"][0])+2.**-70).hex()
    if damage == "field_shape": p["fields"]["n_m3"]["shape"] = [1, 2]
    if damage == "noncanonical_zero":
        p["authority"]["primitives"]["n_m3"]["words"][-1][0] = "0x0p+0"
    if damage == "paired_previous":
        p["authority"]["transition"]["local_primitives"]["n_m3"]["previous"]["words"][0][0] = 1.0.hex()
    if damage == "reference_zero_sign":
        reference = p["authority"]["coordinate_contract"]["reference"]
        reference["payload"]["fields"]["phi_V"]["low"][0] = (-0.).hex()
        reference["sha256"] = c.d.digest(reference["payload"])
    if damage == "payload_type": bad["payload"] = []
    if damage == "rehash_authority":
        p["authority"]["transition"]["previous_authority"] = "0"*64
        p["authority_identity"] = c.d.digest(p["authority"])
    rehash_embedded_point(c, prepared, complete, bad)
    protocol = protocol_for_refresh(c, [prepared, statistics, complete])
    frame = dict(c.frame.snapshot, owner=c.frame.owner, generation=c.frame.generation)
    with pytest.raises(HistoryVerificationError):
        protocol._consume_reference_refresh(c.sample, c.interval, c.domain, c.certificate, 1, frame,
                                            complete["before_statistics"]["raw_statistics"])
    assert protocol.numeric_ordinal == 1 and not protocol.segment_frame.refreshes


@pytest.mark.parametrize("damage", ["predecessor", "physical_word"])
def test_refresh_complete_point_requires_matching_history(admitted_refresh, damage):
    from scripts.benchmarks.native_readback import _reference_refresh_point, HistoryVerificationError
    c = admitted_refresh
    point = encode_point(c.point)
    assert decode_point(point, c.model.layout).identity == c.point.identity
    history = deepcopy(c.sample["history"])
    if damage == "predecessor": history["predecessor_identity"] = c.prior.identity
    if damage == "physical_word": history["physical_cumulative_words_hex"][0][0] = 1.0.hex()
    with pytest.raises(HistoryVerificationError):
        _reference_refresh_point(point, history, c.request)


@pytest.mark.parametrize("damage", [None, "missing", "overlapping", "wrong_generation"])
@pytest.mark.parametrize("serialized", [False, True], ids=["memory", "json"])
def test_refresh_metadata_stream_requires_exact_handoff(admitted_refresh, damage, serialized):
    from scripts.benchmarks.native_readback import HistoryVerificationError
    c = admitted_refresh
    _, prepared, _, _, (_, _, stats, complete) = initialized(c)
    records = [prepared, stats, complete]
    if damage == "missing": records.pop(0)
    if damage == "overlapping": records.insert(1, deepcopy(prepared))
    if damage == "wrong_generation":
        complete["new_epoch"]["generation"] += 1
        complete["record_sha256"] = c.d.digest({k: v for k, v in complete.items() if k != "record_sha256"})
    if serialized:
        records = json.loads(json.dumps(records))
    protocol = protocol_for_refresh(c, records)
    before = deepcopy(protocol.prefix)
    frame = dict(c.frame.snapshot, owner=c.frame.owner, generation=c.frame.generation)
    def consume():
        return protocol._consume_reference_refresh(c.sample, c.interval, c.domain, c.certificate, 1, frame,
                                                   complete["before_statistics"]["raw_statistics"])
    if damage is None:
        raw = consume()
        assert raw["num_steps"] == 0 and protocol.numeric_ordinal == 2 and protocol.ordinal == 1
        assert not protocol.rows.values and protocol.refresh_anchor["transfer"]["Point"] == encode_point(c.point)
    else:
        with pytest.raises((HistoryVerificationError, KeyError)): consume()
    assert protocol.prefix == before


@pytest.mark.parametrize("damage", ["boolean_generation", "float_generation", "endpoint_generation",
                                    "endpoint_time", "endpoint_digest", "extra_member"])
def test_refresh_wire_rejects_rehashed_old_epoch(admitted_refresh, damage):
    from scripts.benchmarks.native_observation import require_frame_successor
    from scripts.benchmarks.native_readback import HistoryVerificationError
    c = admitted_refresh
    fresh, prepared, _, _, (_, _, statistics, complete) = initialized(c)
    prepared, statistics, complete = json.loads(json.dumps([prepared, statistics, complete]))
    for record in (prepared, complete):
        epoch = record["old_epoch"]
        if damage == "boolean_generation": epoch["generation"] = True
        if damage == "float_generation": epoch["generation"] = float(epoch["generation"])
        if damage == "endpoint_generation": epoch["endpoint_identity"][1] = True
        if damage == "endpoint_time": epoch["endpoint_identity"][3] = .5.hex()
        if damage == "endpoint_digest": epoch["endpoint_identity"][-1] = "0"*64
        if damage == "extra_member": epoch["unbound"] = 1
    prepared["record_sha256"] = c.d.digest({k: v for k, v in prepared.items() if k != "record_sha256"})
    complete["prepared_sha256"] = prepared["record_sha256"]
    complete["record_sha256"] = c.d.digest({k: v for k, v in complete.items() if k != "record_sha256"})
    protocol = protocol_for_refresh(c, [prepared, statistics, complete])
    frame = dict(c.frame.snapshot, owner=c.frame.owner, generation=c.frame.generation)
    with pytest.raises(HistoryVerificationError, match="reference_refresh_old_native_epoch"):
        protocol._consume_reference_refresh(c.sample, c.interval, c.domain, c.certificate, 1, frame,
                                            complete["before_statistics"]["raw_statistics"])
    zero = np.zeros(10)
    current = SimpleNamespace(identity="c"*64, owner=c.frame.owner, generation=2,
        binding_identity=c.frame.binding_identity,
        snapshot={"predecessor": {"internal_t": c.point.time, "nsteps": 0,
                                  "raw_y": zero.tobytes(), "raw_yp": zero.tobytes()},
                  "native_before": {"nsteps": 1}})
    with pytest.raises(ContractError, match="native_observation_invalid_numeric_handoff"):
        require_frame_successor(c.frame, current, previous_segment=c.segment, current_segment=c.segment,
            previous_mapping=c.old_mapping, current_mapping=fresh, numeric_handoff=complete)


@pytest.mark.parametrize("damage", ["generation_float", "steps_bool", "preserved_generation_bool",
                                    "before_steps_float", "success_int", "weight_size_float", "ordinal_float"])
def test_refresh_wire_rejects_rehashed_completion_types(admitted_refresh, damage):
    from scripts.benchmarks.native_observation import require_frame_successor
    from scripts.benchmarks.native_readback import HistoryVerificationError
    c = admitted_refresh
    fresh, prepared, _, _, (_, _, statistics, complete) = initialized(c)
    prepared, statistics, complete = json.loads(json.dumps([prepared, statistics, complete]))
    old_statistics = deepcopy(complete["before_statistics"]["raw_statistics"])
    if damage == "generation_float": complete["new_epoch"]["generation"] = 2.0
    if damage == "steps_bool": complete["new_epoch"]["nsteps"] = False
    if damage == "preserved_generation_bool": complete["old_epoch"]["generation"] = True
    if damage == "before_steps_float": complete["before_statistics"]["raw_statistics"]["num_steps"] = 7.0
    if damage == "success_int": complete["initialization_return"]["success"] = 1
    if damage == "weight_size_float": complete["parent_weight_frame"]["size"] = 10.0
    if damage == "ordinal_float": complete["transfer"]["new_numeric_initialization_ordinal"] = 2.0
    complete["record_sha256"] = c.d.digest({k: v for k, v in complete.items() if k != "record_sha256"})
    protocol = protocol_for_refresh(c, [prepared, statistics, complete])
    frame = dict(c.frame.snapshot, owner=c.frame.owner, generation=c.frame.generation)
    with pytest.raises(HistoryVerificationError):
        protocol._consume_reference_refresh(c.sample, c.interval, c.domain, c.certificate, 1, frame,
                                            old_statistics)
    if damage in {"generation_float", "steps_bool", "preserved_generation_bool"}:
        zero = np.zeros(10)
        current = SimpleNamespace(identity="c"*64, owner=c.frame.owner, generation=2,
            binding_identity=c.frame.binding_identity,
            snapshot={"predecessor": {"internal_t": c.point.time, "nsteps": 0,
                                      "raw_y": zero.tobytes(), "raw_yp": zero.tobytes()},
                      "native_before": {"nsteps": 1}})
        with pytest.raises(ContractError, match="native_observation_invalid_numeric_handoff"):
            require_frame_successor(c.frame, current, previous_segment=c.segment, current_segment=c.segment,
                previous_mapping=c.old_mapping, current_mapping=fresh, numeric_handoff=complete)


def test_refresh_anchor_recertification_retains_original_history(admitted_refresh):
    """Exercise the reader bridge; independent frame algebra has its own test."""
    from scripts.benchmarks.native_observation import observation_record
    c = admitted_refresh
    _, prepared, _, _, (_, _, stats, complete) = initialized(c)
    p = protocol_for_refresh(c, [prepared, stats, complete])
    frame = dict(c.frame.snapshot, owner=c.frame.owner, generation=c.frame.generation)
    p._consume_reference_refresh(c.sample, c.interval, c.domain, c.certificate, 1, frame,
                                 complete["before_statistics"]["raw_statistics"])
    h = c.sample["history"]; original = deepcopy(h)
    zero = (Fraction(0),)*10; ports = (Fraction(0),)*2
    enclosure = {"center": [0, 1], "absolute_error_bound": [0, 1]}
    bytes_of = lambda values: np.array([float.fromhex(v) for v in values], dtype='<f8').tobytes()
    p.path_id = "d"*64
    rb = {"point_identity": c.point.identity, "rate_identity": complete["transfer"]["new_rate_identity"],
        "map_identity": p.map_id, "path_identity": p.path_id, "binding_identity": p.refresh_binding_identity,
        "time_hex": h["time_hex"], "event_side": "continuous", "state_changed": False, "native_source_provenance_claimed": False,
        "raw_z_words": np.zeros(10).tobytes(), "raw_zdot_words": np.zeros(10).tobytes(),
        "input_words": bytes_of(h["inputs_hex"]), "input_rate_words": bytes_of(h["input_rates_hex"]),
        "physical_rate_words": tuple(bytes_of(row) for row in h["physical_rate_words_hex"]),
        "physical_state_absolute_error": zero, "physical_rate_absolute_error": zero, "state_action_error_bounds": zero,
        "physical_state_enclosures": tuple(deepcopy(enclosure) for _ in range(10)),
        "state_error_includes_actual_input_evaluation": True, "do_not_add_same_input_error_twice": True}
    evidence = {"point_identity": c.point.identity, "path_identity": p.path_id, "passed": True,
        "native_rate_or_state_changed": False, "DAE_accuracy_certified": False, "readback": rb,
        "input_error_included_once": True, "reported_currents_A": (ports, ports),
        "raw_reference_A": (enclosure, enclosure), "same_state_exact_tangent_reference_A": (enclosure, enclosure),
        "source_and_readback_error_A": (ports, ports), "combined_current_bound_A": ports,
        "original_allocation_A": Fraction(float(c.request["budgets"]["current_A"])/3)}
    record = {"kind": "voltage_lift_current_certificate", "sample_record_sha256": c.sample["record_sha256"],
              "evidence": observation_record(evidence)}
    p.rows = Rows([record])
    seen = []
    p.segment_frame.readback = lambda value, history: seen.append(history)
    p._certificate(c.sample)
    assert h == original and seen == [h]


@pytest.mark.parametrize("same_owner", [False, True])
def test_refresh_preserves_ordinary_event_rate_kinks(same_owner):
    from scripts.benchmarks.coupled_device_prototype import ProtocolSegment
    from scripts.benchmarks.interval_observation import AcceptedClock
    from scripts.benchmarks.native_observation import require_frame_successor
    left_segment = ProtocolSegment("left", 0., 1., (0., .1), (0., 0.))
    right_segment = ProtocolSegment("right", 1., 2., (.1, .1), (0., 0.))
    owner, generation = ("A", 5) if same_owner else ("B", 1)
    zero, rate = np.zeros(1).tobytes(), np.ones(1).tobytes()
    a = SimpleNamespace(owner="A", generation=4, binding_identity="backend",
        snapshot={"endpoint": {"internal_t": 1., "raw_y": zero, "raw_yp": zero, "identity": "old"}})
    b = SimpleNamespace(owner=owner, generation=generation, binding_identity="backend",
        snapshot={"predecessor": {"internal_t": 1., "raw_y": zero, "raw_yp": rate, "nsteps": 0},
                  "native_before": {"nsteps": 1}})
    require_frame_successor(a, b, previous_segment=left_segment, current_segment=right_segment)
    AcceptedClock(.75, 1., .25, 4, 8, 2, 0., 1., "left", "A").require_successor(
        AcceptedClock(1., 1.25, .25, generation, 1, 1, 1., 2., "right", owner))
