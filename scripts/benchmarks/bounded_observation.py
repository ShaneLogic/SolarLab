"""Opt-in conservative upper sums, separate from the native exact-prefix contract.

Only nonnegative *bound increments* are rounded. Every step certificate carries
the untouched interval/signed-evidence bytes. The verifier consumes independent
expected terms and delivers each checked certificate to a required retention
sink; only its final return certifies complete consumption. The sink still owns
durable storage and its I/O/resource checks. No model, solver or native reader
imports this module, and no physical acceptance threshold is changed here.
"""
from dataclasses import dataclass
from fractions import Fraction
import hashlib
from itertools import zip_longest
import json
import re
from typing import Callable, Iterable

POLICY_SCHEMA = "solarlab.nonnegative-upper-sum.v1"
TERM_SCHEMA = "solarlab.nonnegative-upper-term.v1"
CERTIFICATE_SCHEMA = "solarlab.nonnegative-upper-step.v1"
MAX_BITS = 131072
MAX_TERMS = 200000
MAX_EVIDENCE_BYTES = 16 * 1024 * 1024


class UpperSumError(ValueError):
    """A bound, certificate, provenance or finite resource contract failed."""


def _require(condition, reason):
    if not condition:
        raise UpperSumError(reason)


def _integer(value, low, high, name):
    _require(type(value) is int and low <= value <= high, name)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _identity(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _exact(value, bits, name):
    _require(type(value) in (int, Fraction), name + "_must_be_exact")
    value = Fraction(value)
    _require(value >= 0, name + "_negative")
    _require(max(value.numerator.bit_length(), value.denominator.bit_length()) <= bits, name + "_bits")
    return value


@dataclass(frozen=True)
class BoundChannel:
    ledger: str
    row: str
    kind: str
    unit: str = "C"

    def __post_init__(self):
        _require(self.ledger in ("raw_polynomial", "same_state_affine_tangent"), "unsupported_ledger")
        _require(self.row in ("device", "left_metal", "right_metal"), "unsupported_row")
        _require(self.kind in ("total", "reference"), "not_a_nonnegative_upper_bound_channel")
        _require(self.unit == "C", "channel_unit")

    @property
    def identity(self):
        return f"{self.ledger}.{self.row}.{self.kind}"

    @property
    def source_increment(self):
        return "interval_total_bound_C" if self.kind == "total" else "interval_reference_error_C"


@dataclass(frozen=True)
class UpperSumPolicy:
    source_identity: str
    channels: tuple[BoundChannel, ...]
    quantum_bits: int
    max_terms: int
    max_slack_per_channel: Fraction
    max_increment_bits: int
    max_evidence_bytes: int
    schema: str = POLICY_SCHEMA

    def __post_init__(self):
        _require(self.schema == POLICY_SCHEMA, "unsupported_policy_version")
        _require(_identity(self.source_identity), "source_identity")
        _require(type(self.channels) is tuple and 1 <= len(self.channels) <= 12
                 and all(type(c) is BoundChannel for c in self.channels), "channel_map")
        _require(len(set(self.channel_ids)) == len(self.channels), "duplicate_channel")
        _integer(self.quantum_bits, 0, MAX_BITS - 1, "quantum_bits")
        _integer(self.max_terms, 1, MAX_TERMS, "max_terms")
        _integer(self.max_increment_bits, 1, MAX_BITS, "max_increment_bits")
        _integer(self.max_evidence_bytes, 1, MAX_EVIDENCE_BYTES, "max_evidence_bytes")
        slack = _exact(self.max_slack_per_channel, MAX_BITS, "slack")
        _require(self.max_terms * self.quantum <= slack, "lifetime_slack_budget")
        object.__setattr__(self, "max_slack_per_channel", slack)

    @property
    def channel_ids(self):
        return tuple(c.identity for c in self.channels)

    @property
    def quantum(self):
        return Fraction(1, 1 << self.quantum_bits)

    @property
    def upper_integer_bits(self):
        # Each increment is <2^max_increment_bits; each rounded increment is
        # at most2^(max_increment_bits+quantum_bits), before at most N adds.
        return self.max_increment_bits + self.quantum_bits + self.max_terms.bit_length() + 1

    @property
    def identity(self):
        return _digest({"schema": self.schema, "source_identity": self.source_identity,
            "channels": [{"id": c.identity, "unit": c.unit, "increment": c.source_increment,
                          "role": "nonnegative_upper_bound"} for c in self.channels],
            "quantum_bits": self.quantum_bits, "max_terms": self.max_terms,
            "max_slack_per_channel_hex": [hex(self.max_slack_per_channel.numerator),
                                           hex(self.max_slack_per_channel.denominator)],
            "max_increment_bits": self.max_increment_bits, "max_evidence_bytes": self.max_evidence_bytes})


@dataclass(frozen=True)
class ExactBoundTerm:
    source_identity: str
    channel_ids: tuple[str, ...]
    interval_sha256: str
    increments: tuple[Fraction, ...]
    signed_evidence: bytes
    unit: str = "C"
    schema: str = TERM_SCHEMA


@dataclass(frozen=True)
class UpperStepCertificate:
    policy_sha256: str
    index: int
    term: ExactBoundTerm
    upper_numerators: tuple[int, ...]
    rounded_terms: tuple[int, ...]
    slack_upper_bounds: tuple[Fraction, ...]
    schema: str = CERTIFICATE_SCHEMA


def _validate_term(policy, term):
    """Shared structural checks only; verifier rounding arithmetic is separate."""
    _require(type(term) is ExactBoundTerm and term.schema == TERM_SCHEMA, "unsupported_term_version")
    _require(term.source_identity == policy.source_identity, "term_source_identity")
    _require(type(term.channel_ids) is tuple and term.channel_ids == policy.channel_ids, "term_channel_map")
    _require(term.unit == "C", "term_unit")
    _require(type(term.signed_evidence) is bytes and 0 < len(term.signed_evidence) <= policy.max_evidence_bytes,
             "evidence_bytes")
    _require(_identity(term.interval_sha256)
             and hashlib.sha256(term.signed_evidence).hexdigest() == term.interval_sha256, "interval_evidence_identity")
    _require(type(term.increments) is tuple and len(term.increments) == len(policy.channels), "increment_shape")
    values = tuple(_exact(v, policy.max_increment_bits, "increment") for v in term.increments)
    by_channel = dict(zip(policy.channel_ids, values))
    for channel, value in zip(policy.channels, values):
        if channel.kind == "total":
            reference = f"{channel.ledger}.{channel.row}.reference"
            _require(value >= by_channel.get(reference, Fraction(0)), "total_less_than_reference")
    return values


class UpperAccumulator:
    """Constant-size aggregate; callers retain every emitted step certificate."""

    def __init__(self, policy: UpperSumPolicy):
        _require(type(policy) is UpperSumPolicy, "explicit_policy_required")
        self.policy = policy
        self._policy_id = policy.identity
        self._count = 0
        self._upper = (0,) * len(policy.channels)
        self._rounded = (0,) * len(policy.channels)

    @property
    def count(self):
        return self._count

    def append(self, term: ExactBoundTerm) -> UpperStepCertificate:
        _require(self._count < self.policy.max_terms, "too_many_terms")
        values = _validate_term(self.policy, term)
        upper, rounded = [], []
        for k, events, value in zip(self._upper, self._rounded, values):
            quotient, remainder = divmod(value.numerator << self.policy.quantum_bits, value.denominator)
            upper.append(k + quotient + int(remainder != 0))
            rounded.append(events + int(remainder != 0))
        slack = tuple(n * self.policy.quantum for n in rounded)
        _require(all(v <= self.policy.max_slack_per_channel for v in slack), "slack_budget")
        _require(all(k.bit_length() <= self.policy.upper_integer_bits for k in upper), "upper_integer_bits")
        certificate = UpperStepCertificate(self._policy_id, self._count + 1, term,
                                            tuple(upper), tuple(rounded), slack)
        # Validation and certificate construction precede every state mutation.
        self._count, self._upper, self._rounded = certificate.index, certificate.upper_numerators, certificate.rounded_terms
        return certificate


@dataclass(frozen=True)
class VerifiedUpperPrefix:
    policy_sha256: str
    channel_ids: tuple[str, ...]
    count: int
    upper_bounds: tuple[Fraction, ...]
    slack_upper_bounds: tuple[Fraction, ...]
    retention_deliveries: int

    @property
    def lower_bounds(self):
        return tuple(max(Fraction(0), u-s) for u, s in zip(self.upper_bounds, self.slack_upper_bounds))

    def gate_status(self, channel_id: str, *, limit: Fraction, unit: str):
        """Classify this bound enclosure against an unchanged supplied limit.

        This is not native/scientific acceptance. In particular an enclosure
        straddling the limit is indeterminate and cannot earn a pass.
        """
        _require(unit == "C" and channel_id in self.channel_ids, "gate_channel_or_unit")
        limit = _exact(limit, MAX_BITS, "gate_limit")
        index = self.channel_ids.index(channel_id)
        if self.upper_bounds[index] <= limit:
            return "within"
        if self.lower_bounds[index] > limit:
            return "exceeds"
        return "indeterminate"


def verify_upper_prefix(policy: UpperSumPolicy, certificates: Iterable[UpperStepCertificate],
                        expected_terms: Iterable[ExactBoundTerm], *,
                        retain: Callable[[UpperStepCertificate], None]) -> VerifiedUpperPrefix:
    """Independently check local rational inequalities, provenance and evidence.

    The expected terms must come from an independent, bound input decoder. This
    function does not infer physical values from opaque signed-evidence bytes.
    It never calls the producer or its integer-ceiling routine. The required
    sink receives unchanged certificates after each local check; only a final
    return certifies complete matching streams. A later failure leaves merely
    a checked prefix, not a successful complete verification.
    """
    _require(type(policy) is UpperSumPolicy and callable(retain), "policy_and_retention_sink_required")
    policy_id, q = policy.identity, policy.quantum
    upper = (Fraction(0),) * len(policy.channels)
    rounded = (0,) * len(policy.channels)
    count, missing = 0, object()
    for index, (certificate, expected) in enumerate(zip_longest(certificates, expected_terms, fillvalue=missing), 1):
        _require(index <= policy.max_terms, "too_many_terms")
        _require(certificate is not missing and expected is not missing, "incomplete_or_extra_certificate_stream")
        values = _validate_term(policy, expected)
        _require(type(certificate) is UpperStepCertificate and certificate.schema == CERTIFICATE_SCHEMA,
                 "unsupported_certificate_version")
        _require(certificate.policy_sha256 == policy_id, "certificate_policy_identity")
        _require(type(certificate.index) is int and certificate.index == index, "certificate_order")
        _require(type(certificate.term) is ExactBoundTerm and certificate.term == expected, "certificate_term_or_evidence_changed")
        _validate_term(policy, certificate.term)
        for field in (certificate.upper_numerators, certificate.rounded_terms, certificate.slack_upper_bounds):
            _require(type(field) is tuple and len(field) == len(policy.channels), "certificate_shape")
        next_upper, next_rounded = [], []
        for j, value in enumerate(values):
            k, events = certificate.upper_numerators[j], certificate.rounded_terms[j]
            _require(type(k) is int and k >= 0 and k.bit_length() <= policy.upper_integer_bits, "upper_integer_bound")
            _require(type(events) is int and 0 <= events <= index, "rounding_count")
            received = Fraction(k, 1 << policy.quantum_bits)
            exact_step = upper[j] + value
            # An integer-grid point in this half-open interval is unique.
            _require(exact_step <= received < exact_step + q, "nonminimal_or_understated_upper")
            expected_events = rounded[j] + int(received != exact_step)
            _require(events == expected_events, "rounding_count_mismatch")
            supplied_slack = _exact(certificate.slack_upper_bounds[j], MAX_BITS, "certificate_slack")
            _require(supplied_slack == events*q <= policy.max_slack_per_channel, "slack_certificate_mismatch")
            next_upper.append(received)
            next_rounded.append(events)
        retain(certificate)
        upper, rounded, count = tuple(next_upper), tuple(next_rounded), index
    return VerifiedUpperPrefix(policy_id, policy.channel_ids, count, upper,
                               tuple(n*q for n in rounded), count)
