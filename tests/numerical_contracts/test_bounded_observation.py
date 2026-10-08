"""Pure exact-oracle tests; no model, solver, old stream or native fixture."""
from dataclasses import replace
from fractions import Fraction
import hashlib
import json

import pytest

from scripts.benchmarks.bounded_observation import (
    BoundChannel, ExactBoundTerm, MAX_BITS, UpperAccumulator, UpperSumError,
    UpperSumPolicy, verify_upper_prefix,
)

SOURCE = hashlib.sha256(b"independent-synthetic-interval-source-v1").hexdigest()
RIGHT = BoundChannel("raw_polynomial", "right_metal", "reference")


def policy(*, channels=(RIGHT,), bits=32, count=16, **changes):
    fields = dict(source_identity=SOURCE, channels=channels, quantum_bits=bits,
        max_terms=count, max_slack_per_channel=Fraction(count, 1 << bits),
        max_increment_bits=256, max_evidence_bytes=1024)
    fields.update(changes)
    return UpperSumPolicy(**fields)


def term(p, values, index=0):
    # Opaque original evidence retains signed words, negative zero and enclosures.
    evidence = json.dumps({"interval": index, "signed_words": ["-0x0.0p+0", "-0x1.0p-10", "0x1.0p-10"],
        "local_enclosure": {"center_hex_ratio": ["-0x1", "0x3"], "radius_hex_ratio": ["0x1", "0x100"]}},
        separators=(",", ":")).encode()
    return ExactBoundTerm(SOURCE, p.channel_ids, hashlib.sha256(evidence).hexdigest(),
                          tuple(values), evidence)


def produce(p, terms):
    accumulator = UpperAccumulator(p)
    return [accumulator.append(t) for t in terms]


def verify(p, certificates, terms):
    retained = []
    result = verify_upper_prefix(p, iter(certificates), iter(terms), retain=retained.append)
    assert retained == certificates
    assert [c.term.signed_evidence for c in retained] == [t.signed_evidence for t in terms]
    assert result.retention_deliveries == len(terms)
    return result


def test_exact_zero_and_grid_values_retain_every_signed_word():
    p = policy(bits=40)
    terms = [term(p, (Fraction(0),), 0), term(p, (Fraction(3, 1 << 40),), 1), term(p, (0,), 2)]
    certificates = produce(p, terms)
    result = verify(p, certificates, terms)
    assert certificates[0].upper_numerators == (0,)
    assert certificates[0].rounded_terms == (0,)
    assert result.upper_bounds == result.lower_bounds == (Fraction(3, 1 << 40),)
    assert result.slack_upper_bounds == (0,)
    assert b'"-0x0.0p+0"' in certificates[-1].term.signed_evidence


def test_independent_exact_oracle_bounds_every_prefix():
    p = policy(bits=12)
    values = [Fraction(1,3), Fraction(1,5), Fraction(0), Fraction(1,7), Fraction(1,1 << 20)]
    terms = [term(p, (x,), i) for i, x in enumerate(values)]
    certificates = produce(p, terms)
    total = Fraction(0)
    for i, (value, certificate) in enumerate(zip(values, certificates), 1):
        total += value
        upper = Fraction(certificate.upper_numerators[0], 1 << p.quantum_bits)
        assert total <= upper < total + i*p.quantum
        assert upper - total <= certificate.slack_upper_bounds[0] <= p.max_slack_per_channel
    result = verify(p, certificates, terms)
    assert result.lower_bounds[0] <= total <= result.upper_bounds[0]


def test_all_twelve_channels_remain_distinct_across_segments_and_failed_interval():
    channels = tuple(BoundChannel(ledger, row, kind)
        for ledger in ("raw_polynomial", "same_state_affine_tangent")
        for row in ("device", "left_metal", "right_metal") for kind in ("total", "reference"))
    p = policy(channels=channels, bits=20)
    terms = [term(p, tuple(Fraction((j//2)+1, 7+i) * (2 if c.kind == "total" else 1)
                          for j,c in enumerate(channels)), i) for i in range(3)]
    certificates = produce(p, terms)
    result = verify(p, certificates, terms)
    assert result.count == 3  # The caller-supplied last/failing interval is retained too.
    assert len(result.upper_bounds) == 12
    assert all(result.upper_bounds[i] >= result.upper_bounds[i+1] for i in range(0,12,2))
    assert [c.index for c in certificates] == [1,2,3]
    with pytest.raises(UpperSumError, match="certificate_order"):
        verify(p, [certificates[0], replace(certificates[1], index=1), certificates[2]], terms)


def test_gate_margins_and_indeterminate_enclosure_do_not_change_limit():
    p = policy(bits=80)
    limit = Fraction(1, 1 << 40)
    for value, expected in ((limit-p.quantum/4, "within"),
                            (limit+p.quantum/4, "indeterminate"),
                            (limit+2*p.quantum, "exceeds")):
        terms = [term(p, (value,))]
        result = verify(p, produce(p, terms), terms)
        assert result.gate_status(RIGHT.identity, limit=limit, unit="C") == expected
    with pytest.raises(UpperSumError, match="gate_channel_or_unit"):
        result.gate_status(RIGHT.identity, limit=limit, unit="A")
    with pytest.raises(UpperSumError, match="must_be_exact"):
        result.gate_status(RIGHT.identity, limit=float(limit), unit="C")


@pytest.mark.parametrize("change", [
    {"schema": "solarlab.nonnegative-upper-sum.v2"}, {"source_identity": "unbound"},
    {"channels": ()}, {"channels": (RIGHT,RIGHT)}, {"quantum_bits": True},
    {"quantum_bits": MAX_BITS}, {"max_terms": 0}, {"max_terms": 200001},
    {"max_slack_per_channel": Fraction(0)}, {"max_increment_bits": 0},
    {"max_evidence_bytes": 0},
])
def test_policy_is_explicit_versioned_and_finite(change):
    with pytest.raises(UpperSumError):
        policy(**change)


@pytest.mark.parametrize("kind", ["signed_charge", "state", "current", "radius", ""])
def test_physical_or_unbound_quantities_are_not_upper_sum_channels(kind):
    with pytest.raises(UpperSumError, match="nonnegative_upper_bound"):
        BoundChannel("raw_polynomial", "right_metal", kind)


@pytest.mark.parametrize("value", [-1, Fraction(-1,3), 0.5, True, "1/3", 1 << 257])
def test_invalid_increment_does_not_mutate_producer(value):
    p = policy()
    accumulator = UpperAccumulator(p)
    with pytest.raises(UpperSumError):
        accumulator.append(term(p, (value,)))
    assert accumulator.count == 0
    certificate = accumulator.append(term(p, (0,), 1))
    assert certificate.index == 1 and certificate.upper_numerators == (0,)


def test_wrong_mapping_units_evidence_and_source_are_rejected():
    p = policy()
    original = term(p, (Fraction(1,3),))
    bad = [replace(original, source_identity="0"*64), replace(original, channel_ids=("left_metal",)),
           replace(original, unit="A"), replace(original, signed_evidence=original.signed_evidence+b" "),
           replace(original, signed_evidence=bytearray(original.signed_evidence)),
           replace(original, schema="solarlab.nonnegative-upper-term.v2")]
    for item in bad:
        with pytest.raises(UpperSumError):
            UpperAccumulator(p).append(item)
    certificate = produce(p, [original])[0]
    changed = term(p, original.increments, 99)
    with pytest.raises(UpperSumError, match="evidence_changed"):
        verify(p, [replace(certificate, term=changed)], [original])


def test_paired_total_cannot_understate_reference_increment():
    p = policy(channels=(BoundChannel("raw_polynomial","right_metal","total"), RIGHT))
    with pytest.raises(UpperSumError, match="total_less_than_reference"):
        UpperAccumulator(p).append(term(p, (Fraction(1,5),Fraction(1,3))))


def test_understated_and_inflated_prefix_slack_or_policy_certificates_fail():
    p = policy(bits=10)
    terms = [term(p, (Fraction(1,3),))]
    c = produce(p, terms)[0]
    bad = [replace(c, upper_numerators=(c.upper_numerators[0]-1,)),
           replace(c, upper_numerators=(c.upper_numerators[0]+1,)),
           replace(c, upper_numerators=(True,)), replace(c, rounded_terms=(0,)),
           replace(c, slack_upper_bounds=(0,)), replace(c, policy_sha256="0"*64),
           replace(c, schema="solarlab.nonnegative-upper-step.v2"),
           replace(c, upper_numerators=(1 << (p.upper_integer_bits+1),))]
    for certificate in bad:
        with pytest.raises(UpperSumError):
            verify(p, [certificate], terms)


def test_term_ceiling_missing_extra_and_retention_failure_are_not_complete():
    p = policy(count=1)
    terms = [term(p, (Fraction(1,3),), i) for i in range(2)]
    accumulator = UpperAccumulator(p)
    certificates = [accumulator.append(terms[0])]
    with pytest.raises(UpperSumError, match="too_many_terms"):
        accumulator.append(terms[1])
    assert accumulator.count == 1
    with pytest.raises(UpperSumError, match="too_many_terms"):
        verify(p, certificates*2, terms)
    with pytest.raises(UpperSumError, match="incomplete_or_extra"):
        verify(p, [], terms[:1])
    with pytest.raises(UpperSumError, match="incomplete_or_extra"):
        verify(p, certificates, [])
    def broken_sink(_certificate):
        raise OSError("retention refused")
    with pytest.raises(OSError, match="retention refused"):
        verify_upper_prefix(p, certificates, terms[:1], retain=broken_sink)


def test_verifier_does_not_call_producer_rounding(monkeypatch):
    p = policy()
    terms = [term(p, (Fraction(1,3),))]
    certificates = produce(p, terms)
    def forbidden(*_args):
        raise AssertionError("producer called by verifier")
    monkeypatch.setattr(UpperAccumulator, "append", forbidden)
    verify(p, certificates, terms)


def test_prime_denominators_keep_aggregate_representation_finite():
    primes = []
    value = 2
    while len(primes) < 512:
        if all(value % p for p in primes if p*p <= value):
            primes.append(value)
        value += 1
    p = policy(bits=160, count=len(primes))
    terms = [term(p, (Fraction(1, prime*(1 << 80)),), i) for i,prime in enumerate(primes)]
    certificates = produce(p, terms)
    result = verify(p, certificates, terms)
    exact_sum = sum((t.increments[0] for t in terms), Fraction(0))
    assert exact_sum.denominator.bit_length() > 4*p.quantum_bits
    assert result.upper_bounds[0].denominator.bit_length() <= p.quantum_bits+1
    assert result.lower_bounds[0] <= exact_sum <= result.upper_bounds[0]
    assert result.upper_bounds[0]-exact_sum < p.max_terms*p.quantum
    assert certificates[-1].upper_numerators[0].bit_length() <= p.upper_integer_bits
