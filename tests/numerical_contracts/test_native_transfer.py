"""Saved and manufactured native-frame handoffs, independent of device plans."""
from dataclasses import asdict
from fractions import Fraction
from types import SimpleNamespace
import json

import numpy as np
import pytest

from scripts.benchmarks.contract_prototype import ContractError
from scripts.benchmarks.coupled_device_prototype import digest


def test_frame_input_saved_fifth_word_and_product_range_rejections():
    """Exact component38 words from the closed B; no model is evaluated."""
    from scripts.benchmarks import coupled_device_prototype as d
    from scripts.benchmarks.precision_prototype import PrimitiveExpansion

    source = ("0x1.1580b973611f8p-59", "0x1.77b1fecbeee8cp-126", "0x1.99870a7229194p-180", "0x0.0p+0")
    raw = PrimitiveExpansion(tuple(np.array([float.fromhex(w)]) for w in source))
    scale = np.array([float.fromhex("0x1.a78f25679cb4ep-6")])
    expected = ("0x1.cb22dde128652p-65", "-0x1.870ea2fa3cf36p-119", "0x1.7fa957bd3a0a3p-176",
                "0x1.f056101dcecb7p-230", "0x1.8000000000000p-286")
    lift = np.array([[1., 0.]])
    mapped = d._frame_mapped_input(raw, scale, lift, [0., 0.], [0., 0.])
    assert tuple(float(w[0]).hex() for w in mapped.words[:5]) == expected
    assert all(not np.any(w) for w in mapped.words[5:])
    exact = sum((Fraction(float(w[0])) for w in raw.words), Fraction())*Fraction(float(scale[0]))
    assert sum((Fraction(float(w[0])) for w in mapped.words), Fraction()) == exact
    with pytest.raises(ContractError, match="primitive_expansion_capacity_exceeded"):
        d._frame_scale(raw, scale)
    for bad in (np.zeros((1, 2)), np.array([[1., 1.]])):
        with pytest.raises(ContractError, match="frame_input_map_bound"):
            d._frame_mapped_input(raw, scale, bad, [0., 0.])
    with pytest.raises(ArithmeticError, match="supported precision range"):
        d._frame_mapped_input([np.nextafter(0., 1.)], [0.5], lift, [0., 0.])
    assert tuple(float(w[0]).hex() for w in raw.words) == source

def test_frame_input_unframed_seed_final_and_restored_handoffs(monkeypatch):
    """Manufactured affine metadata/initializer only; physical F/J are absent."""
    from scripts.benchmarks import coupled_device_prototype as d
    from scripts.benchmarks.contract_prototype import Layout, Support, VariableSpec, StateView, VOLT
    from scripts.benchmarks.precision_prototype import DoubleArray, DoubleArithmetic, RelativeCoordinates

    class Model:
        def __init__(self):
            self.layout = Layout((Support("nodes", "cell", (2,)),),
                (VariableSpec("phi_V", "potential", "nodes", (2,), VOLT),), ())
            self.coordinates = RelativeCoordinates(self.layout, {"phi_V": "linear"})
            self.reference = self.coordinates.initial(StateView(self.layout, [("phi_V", DoubleArray([0., 0.]))]), inputs=[0., 0.])
            self.S, self.Drow, self.x = np.array([0.25, 0.25]), np.ones(2), np.array([0., 1.])
            self.definition = SimpleNamespace(length=1., n_eq=0., p_eq=0., ion_initial=0., f_eq=0.)
            self.source_identity, self.arithmetic = "a"*64, DoubleArithmetic()

        def trial(self, value, time, inputs, **kwargs):
            return self.coordinates.trial(self.reference, value, time, inputs, **kwargs)

        def validate(self, point):
            assert point.state.layout.identity == self.layout.identity

        def public_problem(self, **kwargs):
            return SimpleNamespace()  # no residual/Jacobian/model evaluator

    monkeypatch.setattr(d, "AffineCoupledSlab", Model)
    model = Model(); profile = "frame-input-expansion12-v1"
    base, enabled = d.AffineVoltageMap(model), d.AffineVoltageMap(model, mapped_input_profile=profile)
    assert len(base.physical_primitive([0., 0.], [0., 0.]).words) == 4
    assert len(enabled.physical_primitive([0., 0.], [0., 0.]).words) == 12
    assert {k: v for k, v in enabled.payload().items() if k not in ("mapped_input_profile", "mapped_input_words")} == base.payload()
    with pytest.raises(ContractError):
        d.AffineVoltageMap(model, mapped_input_profile="unknown")
    segments = tuple(d.ProtocolSegment(name, float(i), float(i+1), (0., 0.), (0., 0.))
                     for i, name in enumerate(("dark_equilibrium_hold", "voltage_ramp", "slow_state_hold")))
    request = {"voltage_lift_map": enabled.payload(), "segment_frame_policy": {}, "frame_input_policy": {"profile": profile}}
    context = SimpleNamespace(request_copy=lambda: json.loads(json.dumps(request)), request_sha256=digest(request),
                              segments=segments, segment_digest=lambda model, segment: digest(asdict(segment)))
    # Parent-policy validation is independently tested against exact saved files.
    monkeypatch.setattr(d, "_validate_segment_frame_policy", lambda request: None)
    seen = []

    def initial_input(binding, raw, predecessor):
        mapping, segment = binding.adapter.mapping, binding.segment
        seen.append(mapping.mapped_input_profile)
        inputs, rates = segment.inputs(segment.start)
        point, increment = mapping.trial(raw, segment.start, inputs, predecessor=predecessor)
        supplied = np.array([0.125, 0.25])
        proof = {"raw_zdot_hex": [float(v).hex() for v in supplied],
                 "mapped_physical_rate_words_hex": [[float(v).hex() for v in w] for w in mapping.physical_rate(supplied, rates).words]}
        proof["record_sha256"] = digest(proof)
        return point, increment, supplied, proof

    monkeypatch.setattr(d, "voltage_lift_initial_input", initial_input)
    monkeypatch.setattr(d, "voltage_lift_rate_projection", lambda model, value: {
        "manufactured_history_metadata_only": True, "input_identity": value.identity_bytes().hex()})
    previous, raw = model.reference, np.zeros(2)
    parent_history = d.VoltageLiftHistory(enabled)
    for ordinal, segment in enumerate(segments, 1):
        binding, u, udot, proof, record = d.voltage_lift_segment_initialization(enabled, context, segment, ordinal, raw, previous)
        mapping = binding.adapter.mapping
        assert mapping.mapped_input_profile == profile and len(mapping.frame.q0.words) == len(mapping.frame.v0.words) == 4
        assert len(record["physical_handoff_words_hex"]) == len(record["physical_rate_handoff_words_hex"]) == 12
        history = d.VoltageLiftHistory(mapping, parent_reference=parent_history.reference_record)
        end, _, _, sample = history.build_sample(u, segment.end, [0., 0.], previous, udot, [0., 0.], event_side="left")
        restored, _, _, _ = history.restore(history.reference_record, sample, previous)
        assert restored.identity == end.identity and sample["mapped_input_profile"] == profile
        assert len(sample["physical_cumulative_words_hex"]) == 12
        damaged = dict(sample, mapped_input_words=4)
        with pytest.raises(ContractError):
            history.restore(history.reference_record, damaged, previous)
        raw, previous = mapping.frame.parent_state(segment.end, u), end
    assert seen == [profile]*3

@pytest.fixture
def diagnostic_transfer_case(monkeypatch):
    """Manufactured five-field Point; no device constructor or F/J/native call."""
    from scripts.benchmarks import coupled_device_prototype as d
    from scripts.benchmarks.contract_prototype import Layout, Support, VariableSpec, StateView, VOLT, ONE, PARTICLE, VOLUME
    from scripts.benchmarks.precision_prototype import DoubleArray, DoubleArithmetic, RelativeCoordinates, PrimitiveExpansion, encode_point

    class Model:
        def __init__(self):
            roots = {"n_m3": 8., "p_m3": 4., "phi_V": 0., "c_m3": 2., "f": 0.25}
            self.layout = Layout((Support("nodes", "cell", (2,)),), tuple(
                VariableSpec(name, name, "nodes", (2,), VOLT if name == "phi_V" else
                             ONE if name == "f" else PARTICLE/VOLUME) for name in roots), ())
            self.coordinates = RelativeCoordinates(self.layout, {name: "linear" for name in roots})
            self.reference = self.coordinates.initial(StateView(self.layout, [
                (name, DoubleArray([value, value])) for name, value in roots.items()]), inputs=[0., 0.])
            self.S, self.Drow, self.x = np.full(10, 0.25), np.ones(10), np.array([0., 1.])
            self.definition = SimpleNamespace(length=1., n_eq=8., p_eq=4., ion_initial=2., f_eq=0.25)
            self.source_identity, self.arithmetic = "a"*64, DoubleArithmetic()

        def trial(self, value, time, inputs, **kwargs):
            return self.coordinates.trial(self.reference, value, time, inputs, **kwargs)

        def validate(self, point):
            if point.state.layout.identity != self.layout.identity or point.inputs.shape != (2,):
                raise ContractError("manufactured_reference_mismatch")
            for variable in self.layout.variables:
                self.reference.state.field(variable.id).difference(point.state.field(variable.id))

        def public_problem(self, **kwargs):
            return SimpleNamespace()  # No residual, tangent, Poisson or Jacobian method.

    monkeypatch.setattr(d, "AffineCoupledSlab", Model)
    model = Model()
    mapping = d.AffineVoltageMap(model, mapped_input_profile="frame-input-expansion12-v1")
    segment = d.ProtocolSegment("saved_hold", 0.5, 0.75, (0.125, 0.25), (3., 4.))
    inputs, input_rate = segment.inputs(segment.start)
    q0 = PrimitiveExpansion(tuple(np.full(10, x) for x in (0.25, 2.**-60, -2.**-120, 2.**-180)))
    v0 = PrimitiveExpansion(tuple(np.full(10, x) for x in (0.125, -2.**-62, 2.**-123, -2.**-185)))
    prior, _ = mapping.trial(np.zeros(10), 0.25, [0., 0.], predecessor=model.reference)
    saved, _ = mapping.trial(q0, segment.start, inputs, predecessor=prior)
    words = lambda value: [[float(x).hex() for x in word] for word in value.words]
    seed = {"frame": {"q0_words_hex": words(q0), "v0_words_hex": words(v0)},
            "physical_state_words_hex": words(mapping.physical_primitive(q0, inputs)),
            "physical_rate_words_hex": words(mapping.physical_rate(v0, input_rate)),
            "inputs_hex": [float(x).hex() for x in inputs],
            "input_rates_hex": [float(x).hex() for x in input_rate]}
    restore = {"Point": encode_point(saved)}
    request = {"voltage_lift_map": mapping.payload(), "map_identity": mapping.identity,
        "segment_frame_policy": {}, "frame_input_policy": {"profile": "frame-input-expansion12-v1"},
        "z0": [0.]*10, "zdot0": [0.]*10, "diagnostic_interval": {"saved_restore": {"manufactured": True}}}
    context = SimpleNamespace(request_copy=lambda: json.loads(json.dumps(request)), request_sha256=digest(request),
        segments=(segment,), segment_digest=lambda model, segment: digest(asdict(segment)))
    # Pin/schema metadata validation has separate tests; this fixture exercises
    # the real frame, map, Point codec, full-word relation and rate binding.
    monkeypatch.setattr(d, "diagnostic_interval_context", lambda request: {"accepted": seed, "restore": restore})
    monkeypatch.setattr(d, "_validate_segment_frame_policy", lambda request: None)
    return SimpleNamespace(d=d, model=model, mapping=mapping, segment=segment, context=context,
                           saved=saved, prior=prior, seed=seed, request=request, restore=restore)

def test_diagnostic_initialization_preserves_saved_transition_and_full_words(diagnostic_transfer_case, monkeypatch):
    from scripts.benchmarks.precision_prototype import encode_point
    c = diagnostic_transfer_case
    original = encode_point(c.saved)
    # The previous implementation necessarily built another transition here.
    def forbidden_trial(*args, **kwargs):
        raise AssertionError("initialization must bind the complete saved Point, not construct a new trial")
    monkeypatch.setattr(c.d.AffineVoltageMap, "trial", forbidden_trial)
    binding, z, zdot, proof, record = c.d.voltage_lift_segment_initialization(
        c.mapping, c.context, c.segment, 1, np.zeros(10), c.saved)
    assert proof["Point"] == original == encode_point(c.saved)
    assert original["payload"]["authority"]["transition"]["previous_point"] == c.prior.identity
    assert c.prior.identity != c.saved.identity == proof["point_identity"]
    assert all(proof["transfer_checks"].values())
    assert record["physical_handoff_words_hex"] == c.seed["physical_state_words_hex"]
    assert record["physical_rate_handoff_words_hex"] == c.seed["physical_rate_words_hex"]
    assert len(record["physical_handoff_words_hex"]) == len(record["physical_rate_handoff_words_hex"]) == 12
    assert proof["inputs_hex"] == c.seed["inputs_hex"] and proof["input_rates_hex"] == c.seed["input_rates_hex"]
    rate = binding.adapter.mapping.bind_rate(c.saved, z, zdot, c.segment.inputs(c.segment.start)[1])
    assert rate.point.identity == c.saved.identity and rate.mapping_identity == binding.adapter.mapping.identity
    assert not proof["native_initialization_performed"] and not proof["original_native_authority_restored"]

@pytest.mark.parametrize("damage", ["raw", "state_word", "rate_word", "slope", "point", "source"])
def test_diagnostic_initialization_rejects_changed_transfer(diagnostic_transfer_case, damage):
    c = diagnostic_transfer_case
    if damage == "raw": c.request["z0"][0] = 2.**-40
    if damage == "state_word": c.seed["physical_state_words_hex"][3][0] = (2.**-170).hex()
    if damage == "rate_word": c.seed["physical_rate_words_hex"][3][0] = (2.**-170).hex()
    if damage == "slope": c.seed["input_rates_hex"][0] = 0.0.hex()
    if damage == "point": c.restore["Point"] = __import__("copy").deepcopy(c.restore["Point"]); c.restore["Point"]["sha256"] = "b"*64
    if damage == "source": c.model.source_identity = "b"*64
    with pytest.raises(ContractError) as caught:
        c.d.voltage_lift_segment_initialization(c.mapping, c.context, c.segment, 1, np.zeros(10), c.saved)
    if damage in {"raw", "state_word", "rate_word", "slope"}:
        evidence = caught.value.diagnostic_transfer_evidence
        assert evidence["Point"] == c.restore["Point"]
        assert not all(evidence["checks"].values())
        assert evidence["mapped_physical_state_words_hex"]
    if damage == "raw":
        assert caught.value.reason == "voltage_lift_point_raw_coordinate_mismatch"
        assert caught.value.diagnostic_transfer_evidence["checks"]["raw_Point_relation"] is False
    elif damage in {"state_word", "rate_word", "slope"}:
        assert caught.value.reason == "diagnostic_complete_physical_transfer"
