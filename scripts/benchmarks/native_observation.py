"""Prepare physical observations from the existing affine voltage map.

No native memory is inspected and no trajectory is advanced. A raw polynomial
must be supplied explicitly; until the owner-level basis export is available,
this module does not manufacture its provenance from rounded Dky readbacks.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
import json
from math import prod
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np

from perovskite_sim.constants import Q
from scripts.benchmarks.contract_prototype import (
    ContractError, LinearTerm, PhysicalLinearForm, frozen_array,
)
from scripts.benchmarks.coupled_device_prototype import (
    AffineCoupledSlab, AffineVoltageMap, VoltageLiftSegmentAdapter, digest,
)
from scripts.benchmarks.interval_observation import (
    AcceptedClock, BallIntegrator, ChargePrefix, Enclosure, Polynomial, PolynomialPath, SlabPathObserver,
    endpoint_charge_mismatch, identity, newton_polynomials, public_action_enclosures,
    rational, segment_input_roundoff,
)
from scripts.benchmarks.bounded_observation import (
    BoundChannel, ExactBoundTerm, UpperAccumulator, UpperSumPolicy,
)
from scripts.benchmarks.native_history import encode_integer_values


CHARGE_REFINEMENT = "fixed4-upper120-v1"
CHARGE_LEDGERS = ("raw_polynomial", "same_state_affine_tangent")
CHARGE_CHANNELS = tuple(BoundChannel(ledger, row, kind) for ledger in CHARGE_LEDGERS
    for row in ("device", "left_metal", "right_metal") for kind in ("total", "reference"))
CHARGE_LOCAL_FIELDS = ("saved_finite_charge_change_C", "nominal_change_C", "signed_integral_C",
    "error_components_C", "signed_saved_defect_C", "interval_total_bound_C", "interval_reference_error_C",
    "original_charge_budget_C", "original_reference_share_C")


def charge_upper_policy(source_identity):
    """The single explicit bounded representation; no scientific tolerance."""
    return UpperSumPolicy(source_identity, CHARGE_CHANNELS, 120, 200000,
                          Fraction(200000, 2**120), 65536, 8*1024*1024)


def prepare_charge_accumulator(request, observer_policy):
    """Construct once per whole protocol, only for the matching v2 policy."""
    refinement = observer_policy.get("charge_refinement")
    expected = prepare_interval_observation_policy(request,
        binding_identity=observer_policy["binding_identity"], header_sha256=observer_policy["header_sha256"],
        backend_modules=observer_policy["backend_modules"],
        charge_refinement=refinement.get("name") if isinstance(refinement, Mapping) else None)
    if digest(expected) != digest(observer_policy):
        raise ContractError("native_observation_charge_policy_changed")
    return UpperAccumulator(charge_upper_policy(digest(request))) if refinement is not None else None


def charge_upper_summary(accumulator):
    """Actual producer representation, including conservative rounding cost."""
    p = accumulator.policy
    return {"schema": "solarlab.native-upper-prefix-state.v1", "policy_sha256": p.identity,
            "source_identity": p.source_identity, "channel_ids": p.channel_ids, "index": accumulator.count,
            "upper_numerators": accumulator.upper_numerators, "rounded_terms": accumulator.rounded_terms,
            "slack_upper_bounds": tuple(n*p.quantum for n in accumulator.rounded_terms)}


def charge_upper_term(source_identity, frame_identity, path_identity, clock, ledgers):
    """Bind untouched local signed evidence; omit cumulative/self-hash fields."""
    selected = {"schema": "solarlab.native-charge-local-evidence.v1", "source_identity": source_identity,
                "frame_identity": frame_identity, "path_identity": path_identity, "clock": clock,
                "ledgers": {key: {name: ledgers[key][name] for name in CHARGE_LOCAL_FIELDS}
                            for key in CHARGE_LEDGERS}}
    encoded = encode_integer_values(observation_record(selected), {"max_integer_bits": 65536})
    signed = json.dumps(encoded, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    increments = tuple(ledgers[channel.ledger][channel.source_increment][i]
        for channel in CHARGE_CHANNELS for i in (("device", "left_metal", "right_metal").index(channel.row),))
    return ExactBoundTerm(source_identity, tuple(c.identity for c in CHARGE_CHANNELS),
                          sha256(signed).hexdigest(), increments, signed)


def input_evaluation_bounds(segment, clock):
    """Bound the actual input operations at stored binary64 query times.

    The nominal smooth input uses the actual stored slope. This is not a
    derivative of a quantized/sawtooth input. Clock/readback time differences
    remain separate errors. Exact constant and unit-slope cases avoid a
    fictitious uncertainty for operations that introduce no rounding.
    """
    original = segment_input_roundoff(segment, clock)
    starts, slopes = segment.inputs(segment.start)
    records = []
    for start, slope, row in zip(starts, slopes, original, strict=True):
        start, slope = rational(start), rational(slope)
        row = dict(row)
        proof = "three-operation round-to-nearest bound with subnormal allowance"
        if slope == 0:
            row["interior_arithmetic_bound"] = Fraction(0)
            proof = "zero multiplication and addition of zero preserve the stored input"
        elif (start == 0 and abs(slope) == 1 and segment.start > 0
              and rational(segment.end) <= 2*rational(segment.start)):
            row["interior_arithmetic_bound"] = Fraction(0)
            proof = "Sterbenz subtraction for stored times, unit multiplication, addition of zero"
        row["uniform_evaluation_bound"] = max(row["interior_arithmetic_bound"], row["end_snap_mismatch"])
        row["exactness_or_bound_source"] = proof
        records.append(MappingProxyType(row))
    return tuple(records)


@dataclass(frozen=True)
class MappedPolynomial:
    path: PolynomialPath
    raw_polynomials: tuple[Polynomial, ...]
    input_errors: tuple
    binding_identity: str
    map_identity: str
    request_identity: str
    reference_identity: str
    segment_identity: str
    coefficient_frame_identity: str | None = None


def _require_map(binding):
    if not isinstance(binding, VoltageLiftSegmentAdapter):
        raise ContractError("native_observation_segment_adapter_required")
    mapping, segment = binding.adapter.mapping, binding.segment
    model = mapping.model
    if (type(mapping) is not AffineVoltageMap or type(model) is not AffineCoupledSlab
            or model.intervals != 8 or model.layout.size > 45
            or model.definition.id not in {"S0NeutralPublicDeviceV1", "DynamicAcceptorIonPublicDeviceV1"}
            or digest(mapping.payload()) != mapping.identity
            or np.any(model.reference.y != 0)):
        raise ContractError("native_observation_unsupported_map_or_reference")
    binding.context.segment_digest(model, segment)
    phi = model.layout.offsets["phi_V"]
    outside = np.r_[np.arange(phi.start), np.arange(phi.stop, model.layout.size)]
    if np.any(mapping.lift[outside]) or np.any(mapping.lift[:, 1]):
        raise ContractError("native_observation_requires_voltage_only_affine_lift")
    return mapping


def prepare_physical_polynomial(binding, raw_polynomials, clock: AcceptedClock, *,
                                clock_policy="reject_uncovered", coefficient_frame_identity=None):
    """Compose ref+S*z+L*(a-a_ref), retaining all real coefficients exactly.

    This prepares a named numerical reconstruction. It does not assert that
    its raw coefficients were acquired from a native solver. The eventual
    Engineering packet must bind that separate source/owner/generation fact.
    """
    mapping = _require_map(binding)
    model = mapping.model
    raw = tuple(raw_polynomials)
    if (len(raw) != model.layout.size or any(not isinstance(p, Polynomial)
            or len(p.coefficients) > clock.kused+1 for p in raw)):
        raise ContractError("native_observation_coordinate_shape_or_order")
    errors = input_evaluation_bounds(binding.segment, clock)
    inputs = tuple(Polynomial((row["input_at_tn_exact_line"], row["stored_slope"]*clock.hused)) for row in errors)
    fields = {}
    for variable in model.layout.variables:
        root = model.field(model.reference, variable.id)
        origin = [sum((rational(word.flat[i]) for word in root.words), Fraction(0))
                  for i in range(prod(variable.shape))]
        offset = model.layout.offsets[variable.id]
        fields[variable.id] = tuple(
            Polynomial((reference,))+rational(mapping.columns[row])*raw[row]
            +sum((rational(mapping.lift[row, j])*(inputs[j]-rational(mapping.reference_inputs[j]))
                  for j in range(len(inputs))), Polynomial())
            for reference, row in zip(origin, range(offset.start, offset.stop), strict=True))
    coefficient_id = identity({"role": "explicit_polynomial_data_not_native_attestation",
                               "raw": [p.payload() for p in raw], "map": mapping.identity,
                               "request": binding.context.request_sha256, "segment": binding.segment_sha256,
                               "coefficient_frame": coefficient_frame_identity})
    path = PolynomialPath(clock, fields, inputs, model.source_identity, model.layout.identity,
                          coefficient_id, "declared_reconstruction", clock_policy)
    return MappedPolynomial(path, raw, errors, binding.source_identity, mapping.identity,
                            binding.context.request_sha256, model.reference.identity, binding.segment_sha256,
                            coefficient_frame_identity)


def _input_direction(form, mapping, input_index):
    """Exact directional action of the public factored physical form."""
    if form.source_identity != mapping.model.source_identity:
        raise ContractError("native_observation_form_source")
    rows = [Fraction(0)]*len(form.row_ids)
    for term in form.terms:
        if term.field is None:
            continue  # A fixed background has exactly zero input variation.
        if term.field in mapping.model.layout.offsets:
            row = mapping.model.layout.offsets[term.field].start+term.index
            direction = rational(mapping.lift[row, input_index])
        elif term.field == "applied_contact":
            direction = Fraction(int(term.index == 1 and input_index == 0))
        else:
            raise ContractError("native_observation_uncovered_external_source")
        coefficient = Fraction(term.sign)
        for factor in term.factors:
            coefficient *= rational(factor.value)
        if term.divisor is not None:
            coefficient /= rational(term.divisor.value)
        rows[term.row] += coefficient*direction
    return tuple(rows)


def voltage_current_sensitivity(mapping, observer: SlabPathObserver):
    """Uniform SG sensitivity to the map's one correlated voltage input."""
    if observer.model is not mapping.model:
        raise ContractError("native_observation_sensitivity_model")
    model, material, path = mapping.model, mapping.model.definition, observer.path
    start = path.clock.coordinate(path.clock.predecessor)
    magnitude = lambda p: max(map(abs, p.range(start, 0)))
    lift = tuple(rational(v) for v in mapping.lift[model.layout.offsets["phi_V"], 0])
    q, vt = rational(Q), rational(material.vt)
    carrier, ion = [], []
    for i, dx in enumerate(map(rational, model.dx)):
        delta_lift = abs(lift[i+1]-lift[i])
        carrier.append(q*delta_lift/dx*(
            abs(rational(material.mu_n))*(magnitude(observer.dn[i])+magnitude(path.fields["n_m3"][i]))
            +abs(rational(material.mu_p))*(magnitude(observer.dp[i])+magnitude(path.fields["p_m3"][i+1]))))
        ion.append(abs(rational(material.diffusion_ion))*delta_lift/(dx*vt)*(
            magnitude(observer.dc[i])+magnitude(path.fields["c_m3"][i+1])) if material.dynamic else Fraction(0))
    return tuple(carrier), tuple(ion)


def propagated_input_error(binding, prepared: MappedPolynomial):
    """Potential/input evaluation error propagated to physical observations.

    On the real axis -1 <= B'(x) <= 0: the upper inequality follows from
    (1-x)*exp(x)<=1, and the lower from exp(x)>=1+x; B'(0)=-1/2.
    This global Lipschitz bound controls SG changes without subtracting two
    wide Bernoulli intervals. The correlated face lift difference is formed
    before dividing by dx/VT. No density/rate word is projected here.
    """
    mapping = _require_map(binding)
    if (prepared.binding_identity != binding.source_identity or prepared.map_identity != mapping.identity
            or prepared.request_identity != binding.context.request_sha256
            or prepared.reference_identity != mapping.model.reference.identity
            or prepared.segment_identity != binding.segment_sha256):
        raise ContractError("native_observation_prepared_map_binding")
    if prepared.input_errors != input_evaluation_bounds(binding.segment, prepared.path.clock):
        raise ContractError("native_observation_input_error_source_changed")
    observer = SlabPathObserver(mapping.model, prepared.path)
    model, material, path = mapping.model, mapping.model.definition, prepared.path
    ev, el = (row["uniform_evaluation_bound"] for row in prepared.input_errors)
    q, area = rational(Q), rational(material.area)
    carrier_sensitivity, ion_sensitivity = voltage_current_sensitivity(mapping, observer)
    carrier = tuple(value*ev for value in carrier_sensitivity)
    ion = tuple(value*ev for value in ion_sensitivity)
    raw_current = (area*carrier[0], area*carrier[-1])
    # The two exact same-state tangent terminal currents are +/- the
    # spacing-weighted total face current plus/minus C*stored_adot.
    tangent_current = area*sum((dx/observer.length*(ec+q*ei)
                               for dx, ec, ei in zip(observer.dx, carrier, ion, strict=True)), Fraction(0))
    names = ("storage", "body_charge", "metal_charge", "gauss_defect", "displacement", "constraints")
    linear = {name: tuple(sum((abs(c)*row["uniform_evaluation_bound"]
                              for c, row in zip(coefficients, prepared.input_errors, strict=True)), Fraction(0))
                         for coefficients in zip(*(_input_direction(model.linear_forms["state"][name], mapping, j)
                                                   for j in range(2)), strict=True)) for name in names}
    duration = path.clock.tn-path.clock.predecessor
    return {"path_identity": path.identity, "mapping_identity": mapping.identity,
            "scope": "input evaluation at the same stored time; native readback/time/state errors remain separate",
            "voltage_error_V": ev, "photon_flux_error_m2_s": el,
            "affine_observable_error": linear,
            "carrier_face_current_error_A_m2": tuple(carrier), "ion_face_flux_error_m2_s": tuple(ion),
            "raw_terminal_current_error_A": raw_current,
            "same_state_tangent_current_error_A": (tangent_current, tangent_current),
            "raw_charge_integral_error_C": (duration*sum(raw_current), Fraction(0), Fraction(0)),
            "tangent_charge_integral_error_C": (duration*sum(raw_current),
                                               *(duration*(tangent_current+v) for v in raw_current)),
            "raw_tangent_L1_additional_error_C": tuple(duration*(v+tangent_current) for v in raw_current),
            "individual_carrier_generation_error_particles_s": tuple(abs(rational(v))*el for v in model.absorption),
            "generation_charge_cancellation": "identical electron/hole source only; not zero carrier forcing/time error",
            "rate_error_from_input_evaluation": "zero only with the same verified stored input_rate; no quantizer differentiation",
            "held_fixed": ["same raw coordinate polynomial", "same stored input-rate words"],
            "endpoint_error_still_required": True, "native_readback_error_still_required": True,
            "native_admitted": False}


def compare_physical_readback(binding, prepared: MappedPolynomial, point, raw_z, raw_zdot, *, side):
    """Compare actual immutable Point/rate words to the nominal polynomial.

    The public map performs its existing raw-coordinate/Point check, and the
    public authority-aware action supplies state word/projection bounds. No
    new Point or state history is manufactured by this reader.
    """
    mapping = _require_map(binding)
    if prepared.binding_identity != binding.source_identity:
        raise ContractError("native_observation_readback_binding")
    if any(not isinstance(value, np.ndarray) or value.dtype != np.dtype('float64')
           or value.shape != (mapping.model.layout.size,) for value in (raw_z, raw_zdot)):
        raise ContractError("native_observation_raw_binary64_words_required")
    raw_z, raw_zdot = frozen_array(raw_z), frozen_array(raw_zdot)
    t = rational(point.time)
    if not prepared.path.clock.predecessor <= t <= prepared.path.clock.tn:
        raise ContractError("native_observation_readback_outside_interval")
    expected_side = "right" if point.time == binding.segment.start else "left" if point.time == binding.segment.end else "continuous"
    if side != expected_side:
        raise ContractError("native_observation_event_side")
    inputs, input_rate = binding.segment.inputs(point.time)
    if inputs.tobytes() != point.inputs.tobytes():
        raise ContractError("native_observation_actual_input_words")
    rate = mapping.bind_rate(point, raw_z, raw_zdot, input_rate)
    model = mapping.model
    terms, units = [], []
    for variable in model.layout.variables:
        for i in range(prod(variable.shape)):
            terms.append(LinearTerm(len(terms), variable.id, i))
            units.append(variable.unit)
    form = PhysicalLinearForm(model.layout, tuple(f"physical_state_{i}" for i in range(len(terms))),
                              tuple(units), tuple(terms), model.source_identity, "state", len(terms))
    actual_state = public_action_enclosures(point.state.linear_form(form, arithmetic=model.arithmetic, point=point))
    u = prepared.path.clock.coordinate(t)
    polynomials = tuple(p for variable in model.layout.variables for p in prepared.path.fields[variable.id])
    state_error = tuple((actual-Enclosure(p.at(u))).absolute_upper
                        for actual, p in zip(actual_state, polynomials, strict=True))
    actual_rate = tuple(sum((rational(word[i]) for word in rate.words), Fraction(0))
                        for i in range(model.layout.size))
    rate_error = tuple(abs(actual-p.derivative().at(u)/prepared.path.clock.hused)
                       for actual, p in zip(actual_rate, polynomials, strict=True))
    return {"point_identity": point.identity, "rate_identity": rate.identity,
            "path_identity": prepared.path.identity, "map_identity": mapping.identity,
            "binding_identity": binding.source_identity, "event_side": side,
            "time_hex": point.time.hex(), "raw_z_words": raw_z.tobytes(),
            "raw_zdot_words": raw_zdot.tobytes(), "input_words": inputs.tobytes(),
            "input_rate_words": input_rate.tobytes(),
            "physical_rate_words": tuple(word.tobytes() for word in rate.words),
            "physical_state_absolute_error": state_error, "physical_rate_absolute_error": rate_error,
            "physical_state_enclosures": actual_state,
            "state_action_error_bounds": tuple(value.radius for value in actual_state),
            "state_error_includes_actual_input_evaluation": True,
            "do_not_add_same_input_error_twice": True,
            "state_changed": False, "native_source_provenance_claimed": False}


def _stamp_key(stamp):
    if (not isinstance(stamp, Mapping)
            or any(type(stamp.get(key)) is not int for key in ("nsteps", "kused", "kk"))
            or any(type(stamp.get(key)) is not float for key in ("tn", "hused", "hh"))):
        raise ContractError("native_observation_stamp_types")
    for key in ("tn", "hused", "hh"):
        rational(stamp[key])
    statuses = stamp.get("statuses")
    names = ("IDAGetNumSteps", "IDAGetCurrentTime", "IDAGetLastStep", "IDAGetLastOrder",
             "IDAGetCurrentStep", "IDAGetCurrentOrder")
    if not isinstance(statuses, tuple) or len(statuses) != 6 or any(
            not isinstance(row, tuple) or len(row) != 2 or row[0] != name
            or type(row[1]) is not int or row[1] != 0 for row, name in zip(statuses, names)):
        raise ContractError("native_observation_stamp_getter_status")
    return (stamp["nsteps"], stamp["tn"].hex(), stamp["hused"].hex(), stamp["kused"],
            stamp["hh"].hex(), stamp["kk"], statuses)


def _owned_snapshot(value):
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise ContractError("native_observation_packet_key_type")
        return MappingProxyType({key: _owned_snapshot(item) for key, item in value.items()})
    if isinstance(value, tuple):
        return tuple(_owned_snapshot(item) for item in value)
    if type(value) in {str, bytes, int, float, bool, type(None)}:
        return value
    raise ContractError("native_observation_nonimmutable_packet_type")


def _snapshot_identity(value):
    def tagged(item):
        if isinstance(item, Mapping):
            return ["mapping", [[key, tagged(item[key])] for key in sorted(item)]]
        if isinstance(item, tuple):
            return ["tuple", [tagged(part) for part in item]]
        if isinstance(item, bytes):
            return ["bytes", item.hex()]
        if type(item) is float:
            return ["binary64", item.hex()]
        return [type(item).__name__, item]
    return identity(tagged(value))


def _array_words(buffer, shape):
    if type(buffer) is not bytes or len(buffer) != 8*prod(shape):
        raise ContractError("native_observation_buffer_shape_or_ownership")
    values = np.frombuffer(buffer, dtype='<f8').reshape(shape)
    if not np.all(np.isfinite(values)):
        raise ContractError("native_observation_nonfinite_source_words")
    return values


@dataclass(frozen=True)
class NativeBasisFrame:
    """Owned source-packet data, not a native-run or release admission."""

    snapshot: Mapping
    identity: str
    binding_identity: str
    owner: str
    generation: int
    raw_polynomials: tuple[Polynomial, ...]


def read_native_basis_packet(snapshot, *, expected_binding_identity, expected_header_sha256, size):
    """Read Engineering's agreed IDA7.5 packet without private memory access.

    The expected fingerprints must come from the separately reviewed build
    receipt. Unit fixtures can test this reader's structure; they do not
    establish that such a native build or trajectory was actually executed.
    """
    if (not isinstance(expected_binding_identity, str) or len(expected_binding_identity) != 64
            or not isinstance(expected_header_sha256, str) or len(expected_header_sha256) != 64):
        raise ContractError("native_observation_missing_expected_source_pins")
    snapshot = _owned_snapshot(snapshot)
    if (snapshot.get("schema") != "sksundae.ida.accepted-step-observation.v1"
            or snapshot.get("dtype") != '<f8' or snapshot.get("shape") != (size,)
            or type(size) is not int or not 1 <= size <= 45):
        raise ContractError("native_observation_packet_schema_or_shape")
    binding = snapshot.get("binding", {})
    if (binding.get("identity") != expected_binding_identity
            or not isinstance(binding.get("source"), tuple)
            or sha256(repr(binding["source"]).encode()).hexdigest() != expected_binding_identity):
        raise ContractError("native_observation_binding_pin_mismatch")
    source = binding["source"]
    if (len(source) != 8 or source[0] != snapshot["schema"] or source[1] != "scikit-sundae=1.1.3"
            or source[5] != expected_header_sha256):
        raise ContractError("native_observation_binding_source_schema")
    before, after = snapshot.get("native_before"), snapshot.get("native_after")
    if _stamp_key(before) != _stamp_key(after):
        raise ContractError("native_observation_step_changed")
    owner, generation = snapshot.get("owner"), snapshot.get("generation")
    if not isinstance(owner, str) or not owner or type(generation) is not int or generation < 1:
        raise ContractError("native_observation_owner_or_generation")
    q = before["kused"]
    expected_key = (expected_binding_identity, owner, generation, before["nsteps"],
                    before["tn"].hex(), before["hused"].hex(), q, before["hh"].hex(), before["kk"])
    if (not 1 <= q <= 5 or before["hused"] <= 0 or before["nsteps"] < 1
            or snapshot.get("step_key") != expected_key):
        raise ContractError("native_observation_step_key_or_forward_order")
    previous, endpoint = snapshot.get("predecessor", {}), snapshot.get("endpoint", {})
    for point, nsteps in ((previous, before["nsteps"]-1), (endpoint, before["nsteps"])):
        if (point.get("owner") != owner or point.get("generation") != generation
                or point.get("nsteps") != nsteps or type(point.get("internal_t")) is not float
                or type(point.get("output_t")) is not float
                or point["internal_t"].hex() != point["output_t"].hex()
                or type(point.get("native_status")) is not int or point["native_status"] not in {0, 1}):
            raise ContractError("native_observation_unwitnessed_endpoint")
        rational(point["internal_t"])
        for key in ("raw_y", "raw_yp"):
            _array_words(point.get(key), (size,))
        expected = (owner, generation, nsteps, point["internal_t"].hex(), point["output_t"].hex(),
                    sha256(point["raw_y"]+point["raw_yp"]).hexdigest())
        if point.get("identity") != expected:
            raise ContractError("native_observation_endpoint_identity")
    if (not previous["internal_t"] < endpoint["internal_t"] or endpoint["internal_t"] != before["tn"]
            or snapshot.get("raw_y") != endpoint["raw_y"] or snapshot.get("raw_yp") != endpoint["raw_yp"]
            or snapshot.get("clock_gap_terms") != (before["tn"], -before["hused"], -previous["internal_t"])):
        raise ContractError("native_observation_endpoint_or_clock_binding")
    basis = snapshot.get("basis", {})
    if (basis.get("kind") != "ida75_phi_psi_copy_v1" or basis.get("dtype") != '<f8'
            or basis.get("phi_shape") != (q+1, size) or basis.get("psi_shape") != (q,)
            or type(basis.get("status")) is not int or basis["status"] != 0
            or basis.get("source_header_sha256") != expected_header_sha256
            or basis.get("source_config_sha256") != source[6]
            or basis.get("build_identity") != expected_binding_identity):
        raise ContractError("native_observation_missing_or_unbound_phi_psi")
    if type(basis.get("uround")) is not float or not 0 < rational(basis["uround"]) < 1:
        raise ContractError("native_observation_native_roundoff_metadata")
    phi = _array_words(basis.get("phi"), (q+1, size))
    psi = _array_words(basis.get("psi"), (q,))
    predecessor_dky = snapshot.get("predecessor_dky", {})
    statuses = predecessor_dky.get("statuses")
    if (predecessor_dky.get("query_t") != previous["internal_t"]
            or predecessor_dky.get("query_t_hex") != previous["internal_t"].hex()
            or predecessor_dky.get("orders") != (0, 1)
            or not isinstance(statuses, tuple) or len(statuses) != 2
            or any(type(code) is not int for code in statuses)
            or type(predecessor_dky.get("native_eligible")) is not bool
            or predecessor_dky["native_eligible"] != all(code == 0 for code in statuses)
            or _stamp_key(predecessor_dky.get("step_before")) != _stamp_key(before)
            or _stamp_key(predecessor_dky.get("step_after")) != _stamp_key(before)):
        raise ContractError("native_observation_predecessor_query_binding")
    buffers = predecessor_dky.get("buffers")
    if not isinstance(buffers, tuple) or len(buffers) != 2:
        raise ContractError("native_observation_predecessor_buffers")
    for code, buffer in zip(statuses, buffers, strict=True):
        if code == 0:
            _array_words(buffer, (size,))
        elif buffer is not None:
            raise ContractError("native_observation_failed_getter_has_buffer")
    dky = snapshot.get("dky")
    if not isinstance(dky, tuple) or len(dky) != q+1:
        raise ContractError("native_observation_rounded_readbacks_missing")
    for buffer in (*dky, snapshot.get("error_weights"), snapshot.get("estimated_local_errors")):
        _array_words(buffer, (size,))
    statuses = snapshot.get("vector_statuses")
    names = tuple(f"IDAGetDky[{k}]" for k in range(q+1))+("IDAGetErrWeights", "IDAGetEstLocalErrors")
    if not isinstance(statuses, tuple) or len(statuses) != len(names) or any(
            not isinstance(row, tuple) or len(row) != 2 or row[0] != name
            or type(row[1]) is not int or row[1] != 0 for row, name in zip(statuses, names)):
        raise ContractError("native_observation_vector_getter_status")
    if np.any(_array_words(snapshot["error_weights"], (size,)) <= 0):
        raise ContractError("native_observation_error_weights")
    stamp_id = _snapshot_identity(snapshot)
    return NativeBasisFrame(snapshot, stamp_id, expected_binding_identity, owner, generation,
                            newton_polynomials(phi, psi, before["hused"]))


def prepare_native_frame_map(binding, frame: NativeBasisFrame):
    """Map a validated packet, preserving its distinct source and owner epoch."""
    source, segment = frame.snapshot, binding.segment
    native = source["native_before"]
    clock = AcceptedClock(source["predecessor"]["internal_t"], native["tn"], native["hused"],
                          frame.generation, native["nsteps"], native["kused"],
                          segment.start, segment.end, segment.id, owner=frame.owner)
    policy = "reject_uncovered"
    if clock.strip:
        if not source["predecessor_dky"]["native_eligible"]:
            raise ContractError("native_observation_strip_getter_not_eligible")
        policy = "declared_polynomial_extension"
    return prepare_physical_polynomial(binding, frame.raw_polynomials, clock,
                                        clock_policy=policy, coefficient_frame_identity=frame.identity)


def require_frame_successor(previous: NativeBasisFrame, current: NativeBasisFrame, *,
                            previous_segment, current_segment):
    """Keep native owner/generation distinct from the controller segment index."""
    left, right = previous.snapshot["endpoint"], current.snapshot["predecessor"]
    if previous.binding_identity != current.binding_identity:
        raise ContractError("native_observation_backend_changed_within_history")
    if left["internal_t"].hex() != right["internal_t"].hex():
        raise ContractError("native_observation_frame_time_gap")
    same_epoch = previous.owner == current.owner and previous.generation == current.generation
    if same_epoch:
        if (previous_segment.id != current_segment.id or left["identity"] != right["identity"]):
            raise ContractError("native_observation_frame_predecessor_mismatch")
        return
    if (previous_segment.id == current_segment.id or left["internal_t"] != previous_segment.end
            or right["internal_t"] != current_segment.start or left["raw_y"] != right["raw_y"]
            or current.snapshot["native_before"]["nsteps"] != 1 or right["nsteps"] != 0
            or tuple(previous_segment.voltage)[1] != tuple(current_segment.voltage)[0]
            or tuple(previous_segment.photons)[1] != tuple(current_segment.photons)[0]
            or previous.owner == current.owner and current.generation != previous.generation+1
            or previous.owner != current.owner and current.generation != 1):
        raise ContractError("native_observation_unbound_restart_epoch")
    # Raw rates are deliberately not equated across a protocol derivative
    # kink; both original left/right words remain in their separate packets.


def observation_record(value):
    """Lossless observation JSON, including every packet byte and clock word."""
    if isinstance(value, Fraction):
        return {"rational": [value.numerator, value.denominator]}
    if isinstance(value, bytes):
        return {"bytes_hex": value.hex()}
    if isinstance(value, Mapping):
        return {key: observation_record(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return {"tuple": [observation_record(item) for item in value]}
    if isinstance(value, list):
        return [observation_record(item) for item in value]
    if type(value) is float:
        return {"binary64": value.hex()}
    if hasattr(value, "payload"):
        return observation_record(value.payload())
    if type(value) in {str, int, bool, type(None)}:
        return value
    raise ContractError("native_observation_unserializable_record")


def restore_observation_record(value):
    """Restore exact packet types before rechecking its source authority."""
    if isinstance(value, list):
        return [restore_observation_record(item) for item in value]
    if isinstance(value, dict):
        if set(value) == {"rational"}:
            return Fraction(*value["rational"])
        if set(value) == {"bytes_hex"}:
            return bytes.fromhex(value["bytes_hex"])
        if set(value) == {"binary64"}:
            return float.fromhex(value["binary64"])
        if set(value) == {"tuple"}:
            return tuple(restore_observation_record(item) for item in value["tuple"])
        return {key: restore_observation_record(item) for key, item in value.items()}
    return value


def _affine_rate_form(model, name):
    original = model.linear_forms["rate"][name]
    terms = tuple(term for term in original.terms if term.field in model.layout.offsets or term.field is None)
    return PhysicalLinearForm(original.layout, original.row_ids, original.row_units, terms,
                              original.source_identity, "rate", len(terms))


def readback_current_evidence(binding, prepared, pair, raw_z, raw_zdot, arithmetic, current_budget, *, observer=None):
    """Bound the actual saved currents against the same physical equations.

    No readback rate is replaced. Exact affine rate discrepancies use the
    shared public form; nonlinear state discrepancies use real SG/capture
    sensitivity bounds. The point's input error is already in that discrepancy
    and is not added again. A charge/L1 bound never satisfies this point gate.
    """
    mapping = _require_map(binding)
    model, m, point = mapping.model, mapping.model.definition, pair[0]
    side = "right" if point.time == binding.segment.start else "left" if point.time == binding.segment.end else "continuous"
    comparison = compare_physical_readback(binding, prepared, point, raw_z, raw_zdot, side=side)
    observer = SlabPathObserver(model, prepared.path) if observer is None else observer
    if observer.path is not prepared.path or observer.model is not model:
        raise ContractError("native_observation_foreign_point_observer")
    u = prepared.path.clock.coordinate(point.time)
    values = {name: tuple(p.at(u) for p in row) for name, row in prepared.path.fields.items()}
    errors = {name: comparison["physical_state_absolute_error"][offset] for name, offset in model.layout.offsets.items()}
    q, vt, area = rational(Q), rational(m.vt), rational(m.area)
    carrier, ion = [], []
    for i, dx in enumerate(observer.dx):
        xi = observer.xi[i].at(u)
        exi = (errors["phi_V"][i]+errors["phi_V"][i+1])/vt
        # B is 1-Lipschitz and 0<B(x)<=1+abs(x) on the real axis.
        bmax = 1+abs(xi)+exi
        en, ep = errors["n_m3"], errors["p_m3"]
        error_n = bmax*(en[i]+en[i+1])+exi*abs(observer.dn[i].at(u))
        error_n += (abs(xi)+exi)*en[i]+exi*abs(values["n_m3"][i])
        error_p = bmax*(ep[i]+ep[i+1])+exi*abs(observer.dp[i].at(u))
        error_p += (abs(xi)+exi)*ep[i+1]+exi*abs(values["p_m3"][i+1])
        carrier.append(q*vt/dx*(abs(rational(m.mu_n))*error_n+abs(rational(m.mu_p))*error_p))
        if m.dynamic:
            c, ec, capacity = values["c_m3"], errors["c_m3"], rational(m.ion_capacity)
            vacancy = (capacity-c[i], capacity-c[i+1])
            lower = (vacancy[0]-ec[i], vacancy[1]-ec[i+1])
            if min(*vacancy, *lower) <= 0:
                raise ContractError("native_observation_readback_vacancy_error_domain")
            edrive = exi+ec[i]/lower[0]+ec[i+1]/lower[1]
            drive_bound = abs(xi)+abs(c[i+1]-c[i])/min(vacancy)
            error_i = (1+drive_bound+edrive)*(ec[i]+ec[i+1])+edrive*abs(c[i+1]-c[i])
            error_i += (drive_bound+edrive)*ec[i+1]+edrive*abs(c[i+1])
            ion.append(abs(rational(m.diffusion_ion))/dx*error_i)
        else:
            ion.append(Fraction(0))
    capture = []
    for node in (0, model.count-1):
        error = Fraction(0)
        if m.dynamic:
            cn, cp = rational(m.capture_n), rational(m.capture_p)
            en, ep, ef = (errors[name][node] for name in ("n_m3", "p_m3", "f"))
            f = values["f"][node]
            error = cn*abs(1-f)*en+cp*abs(f)*ep
            error += (cn*(abs(values["n_m3"][node])+rational(m.n1)+en)
                      +cp*(abs(values["p_m3"][node])+rational(m.p1)+ep))*ef
        capture.append(q*observer.volumes[node]*rational(m.trap_density)*error)
    rate = mapping.bind_rate(point, raw_z, raw_zdot, binding.segment.inputs(point.time)[1])
    form = _affine_rate_form(model, "total_current")
    actual_affine = public_action_enclosures(rate.linear_form(form, arithmetic=model.arithmetic, point=point))
    nominal_affine = prepared.path.action(form)
    raw_error = tuple((actual-Enclosure(nominal.at(u))).absolute_upper+area*carrier[face]+capture[side]
                      for side, (actual, nominal, face) in enumerate(zip(actual_affine, nominal_affine,
                                                                        (0, model.count-2), strict=True)))
    tangent_error = area*sum((dx/observer.length*(ec+q*ei)
                             for dx, ec, ei in zip(observer.dx, carrier, ion, strict=True)), Fraction(0))
    with arithmetic.flint.ctx.workprec(arithmetic.bits):
        currents = observer.currents(arithmetic, arithmetic.number(u))
        nominal = {key: tuple(arithmetic.enclosure(value) for value in row) for key, row in currents.items()}
    reported = tuple(tuple(sum((rational(word[i]) for word in reading.total_inward.words), Fraction(0))
                           for i in range(2)) for reading in pair[2:4])
    raw_reference = tuple(value+Enclosure(0, error) for value, error in
                          zip(nominal["raw_polynomial_current"], raw_error, strict=True))
    tangent_reference = tuple(value+Enclosure(0, tangent_error)
                              for value in nominal["same_state_affine_tangent_current"])
    source_errors = tuple(tuple((Enclosure(value)-reference).absolute_upper for value, reference in
                                zip(row, reference_row, strict=True)) for row, reference_row in
                          zip(reported, (raw_reference, tangent_reference), strict=True))
    combined = tuple(abs(reported[0][i]-reported[1][i])+source_errors[0][i]+source_errors[1][i] for i in range(2))
    limit = rational(float(current_budget)/3)
    return {"point_identity": point.identity, "path_identity": prepared.path.identity,
            "readback": comparison, "nominal_currents_A": nominal,
            "reported_currents_A": reported, "raw_reference_A": raw_reference,
            "same_state_exact_tangent_reference_A": tangent_reference,
            "nonlinear_readback_carrier_error_A_m2": tuple(carrier),
            "nonlinear_readback_ion_error_m2_s": tuple(ion),
            "source_and_readback_error_A": source_errors, "combined_current_bound_A": combined,
            "original_allocation_A": limit, "passed": all(value <= limit for value in combined),
            "input_error_included_once": True, "native_rate_or_state_changed": False,
            "tangent_reference_origin": "exact same-state affine-constraint estimate, not a native derivative",
            "DAE_accuracy_certified": False}


class AcceptedIntervalObserver:
    """One owned accepted-step observation, with no time-advancement method."""

    def __init__(self, binding, frame, *, arithmetic, previous=None):
        self.binding, self.frame, self.arithmetic = binding, frame, arithmetic
        self.prepared = prepare_native_frame_map(binding, frame)
        if previous is not None:
            require_frame_successor(previous.frame, frame, previous_segment=previous.binding.segment,
                                    current_segment=binding.segment)
            previous.prepared.path.clock.require_successor(self.prepared.path.clock)
        self.observer = SlabPathObserver(binding.adapter.mapping.model, self.prepared.path)

    def require_endpoint(self, native, *, predecessor=False):
        expected = self.frame.snapshot["predecessor" if predecessor else "endpoint"]
        if (type(native["time"]) is not float or native["time"].hex() != expected["internal_t"].hex()
                or native["z"].dtype != np.dtype('float64') or native["zdot"].dtype != np.dtype('float64')
                or native["z"].tobytes() != expected["raw_y"] or native["zdot"].tobytes() != expected["raw_yp"]):
            raise ContractError("native_observation_controller_endpoint_mismatch")

    def reconstruction(self, time):
        """Rounded output of the declared path, explicitly not a native read."""
        t = rational(time)
        if not self.prepared.path.clock.predecessor <= t <= self.prepared.path.clock.tn:
            raise ContractError("native_observation_query_outside_interval")
        u, h = self.prepared.path.clock.coordinate(t), self.prepared.path.clock.hused
        raw = self.prepared.raw_polynomials
        return {"time": float(time), "z": frozen_array([float(p.at(u)) for p in raw]),
                "zdot": frozen_array([float(p.derivative().at(u)/h) for p in raw]),
                "success": True, "status": 0, "message": "declared polynomial reconstruction; no native getter call",
                "coefficient_frame_identity": self.frame.identity,
                "path_identity": self.prepared.path.identity, "native_output": False}

    def point_evidence(self, pair, native, current_budget):
        return readback_current_evidence(self.binding, self.prepared, pair, native["z"], native["zdot"],
                                         self.arithmetic, current_budget, observer=self.observer)

    def path_domain_evidence(self, budgets):
        """Retain the original site/inventory gates on the whole named path."""
        model, path = self.observer.model, self.prepared.path
        checks, values = {}, {}
        if model.definition.dynamic:
            capacity = rational(model.definition.ion_capacity)
            maximum = max(high/capacity for _, high in self.observer.domain_bounds["c_m3"])
            checks["maximum_site_fraction"] = maximum <= rational(budgets["maximum_site_fraction"])
            values["maximum_site_fraction"] = maximum
            reference = public_action_enclosures(model.linear_action("ion_inventory", model.reference))[0]
            inventory = path.action(model.linear_forms["state"]["ion_inventory"])[0]
            lower, upper = (inventory-reference.center).range(path.clock.coordinate(path.clock.predecessor), 0)
            deviation = max(abs(lower), abs(upper))+reference.radius
            denominator = abs(reference.center)-reference.radius
            if denominator <= 0:
                checks["ion_inventory"] = deviation == 0 and reference.absolute_upper == 0
                values["ion_inventory_absolute_deviation"] = deviation
                values["ion_inventory_zero_branch"] = True
            else:
                ratio = deviation/denominator
                checks["ion_inventory"] = ratio <= rational(budgets["ion_inventory_relative"])
                values["ion_inventory_relative"] = ratio
        return {"path_identity": path.identity, "domain_bounds": self.observer.domain_bounds,
                "metrics": values, "checks": checks, "passed": all(checks.values()),
                "scope": "declared polynomial domain only; actual sample state/affine-constraint gates also remain required"}

    def charge_evidence(self, left_pair, right_pair, prefixes, *, absolute_error, charge_budget,
                        charge_accumulator=None):
        """Each body/metal row carries disjoint positive error debits.

        The nominal-path defect plus an absolute endpoint mismatch bounds the
        actual saved-endpoint defect. The mismatch is included once; the saved
        finite increment remains an independent, full-word diagnostic.
        """
        model, path = self.observer.model, self.prepared.path
        left, right, increment = left_pair[0], right_pair[0], right_pair[1]
        inputs = propagated_input_error(self.binding, self.prepared)
        for name in ("raw_charge_integral_error_C", "tangent_charge_integral_error_C"):
            row = inputs.get(name)
            if (not isinstance(row, tuple) or len(row) != 3 or any(value is None or rational(value) < 0 for value in row)):
                raise ContractError("native_observation_missing_input_error_term:"+name)
        if charge_accumulator is not None:
            expected_policy = charge_upper_policy(self.binding.context.request_sha256)
            if (type(charge_accumulator) is not UpperAccumulator
                    or charge_accumulator.policy.identity != expected_policy.identity):
                raise ContractError("native_observation_upper_source_policy")
            old = tuple(Fraction(v, 2**120) for v in charge_accumulator.upper_numerators)
            for j, key in enumerate(CHARGE_LEDGERS):
                if (prefixes[key].intervals != charge_accumulator.count
                        or prefixes[key].absolute_defects != old[j*6:j*6+6:2]
                        or prefixes[key].reference_errors != old[j*6+1:j*6+6:2]):
                    raise ContractError("native_observation_upper_prefix_reset")
        integral = (self.observer.integrate(self.arithmetic, absolute_error, partition_cells=4)
                    if charge_accumulator is not None else self.observer.integrate(self.arithmetic, absolute_error))
        strip = self.observer.strip_current_debit(self.arithmetic)
        actual = (*public_action_enclosures(model.linear_action("body_charge", right, increment=increment, left=left)),
                  *public_action_enclosures(model.linear_action("metal_charge", right, increment=increment, left=left)))
        nominal = tuple(Enclosure(b-a) for name in ("body_charge", "metal_charge") for a, b in
                        zip(self.observer.affine[name]["left"], self.observer.affine[name]["right"], strict=True))
        endpoint = tuple(max(bound, (observed-reference).absolute_upper) for bound, observed, reference in
                         zip(endpoint_charge_mismatch(model, path, left, right), actual, nominal, strict=True))
        gap = integral["raw_tangent_charge_L1_upper_C"]
        # This is a disclosed departure-to-tangent debit, not the obsolete
        # first-word projection. Input perturbations also reach that bound.
        gap_input = tuple(a+b for a, b in zip(inputs["raw_charge_integral_error_C"],
                                             inputs["tangent_charge_integral_error_C"], strict=True))
        ledgers, successors = {}, {}
        for key, input_key in (("raw_polynomial", "raw_charge_integral_error_C"),
                               ("same_state_affine_tangent", "tangent_charge_integral_error_C")):
            if key not in prefixes or not isinstance(prefixes[key], ChargePrefix):
                raise ContractError("native_observation_missing_prefix")
            if prefixes[key].budgets != (rational(charge_budget),)*3:
                raise ContractError("native_observation_charge_gate_changed")
            departure = tuple(a+b for a, b in zip(gap, gap_input, strict=True)) if key == "raw_polynomial" else (Fraction(0),)*3
            components = {"endpoint": endpoint, "input": inputs[input_key], "clock_strip": strip[key],
                          "raw_tangent_departure": departure,
                          "integration": tuple(value.radius for value in integral[key])}
            additional = tuple(sum((components[name][i] for name in components if name != "integration"), Fraction(0))
                               for i in range(3))
            successor, checks = prefixes[key].append(nominal, integral[key], additional)
            errors = tuple(integral[key][i].radius+additional[i] for i in range(3))
            limit = rational(float(charge_budget)/3)
            checks = tuple(check and error <= limit and total <= limit for check, error, total in
                           zip(checks, errors, successor.reference_errors, strict=True))
            ledgers[key] = {"saved_finite_charge_change_C": actual, "nominal_change_C": nominal,
                            "signed_integral_C": integral[key], "error_components_C": components,
                            "signed_saved_defect_C": tuple(a-b for a, b in zip(actual, integral[key], strict=True)),
                            "interval_total_bound_C": tuple(abs((a-b).center)+error for a, b, error in
                                                            zip(nominal, integral[key], errors, strict=True)),
                            "prefix_absolute_defect_C": successor.absolute_defects,
                            "interval_reference_error_C": errors, "prefix_reference_error_C": successor.reference_errors,
                            "original_charge_budget_C": rational(charge_budget), "original_reference_share_C": limit,
                            "row_checks": checks, "passed": all(checks), "endpoint_debited_once": True}
            successors[key] = successor
        evidence = {"path_identity": path.identity, "frame_identity": self.frame.identity,
                            "clock": path.clock.payload(), "input_error": inputs,
                            "raw_tangent_L1_current_bounds": integral["raw_tangent_L1_upper_bounds"],
                            "ledgers": ledgers, "passed": all(row["passed"] for row in ledgers.values()),
                            "scope": "physical observations of the declared fixed-grid polynomial with saved endpoints",
                            "raw_rate_projection_role": "retained counterfactual, not charged as if still used",
                            "source_arithmetic_scope": "the bound ball integrals evaluate the physical sources with the frozen coefficients; no projected nodal source is an integral input",
                            "DAE_time_accuracy_certified": False, "continuum_space_accuracy_certified": False}
        if charge_accumulator is not None:
            certificate = charge_accumulator.append(charge_upper_term(self.binding.context.request_sha256,
                self.frame.identity, path.identity, path.clock.payload(), ledgers))
            upper = tuple(k*charge_accumulator.policy.quantum for k in certificate.upper_numerators)
            for j, key in enumerate(CHARGE_LEDGERS):
                absolute, reference = upper[j*6:j*6+6:2], upper[j*6+1:j*6+6:2]
                successor = ChargePrefix(prefixes[key].budgets, absolute, reference, certificate.index)
                # Local terms remain exact; only the cumulative upper is rounded.
                # A straddling enclosure cannot earn a pass by using its lower end.
                checks = tuple(check and a <= budget and 3*r <= budget and r <= limit
                    for check, a, r, budget in zip(ledgers[key]["row_checks"], absolute, reference,
                                                   successor.budgets, strict=True))
                ledgers[key].update(prefix_absolute_defect_C=absolute, prefix_reference_error_C=reference,
                                    row_checks=checks, passed=all(checks))
                successors[key] = successor
            evidence.update(terminal_partition=integral["terminal_partition"],
                upper_sum={**charge_upper_summary(charge_accumulator), "schema": certificate.schema,
                           "interval_sha256": certificate.term.interval_sha256},
                passed=all(row["passed"] for row in ledgers.values()))
        return successors, evidence


def prepare_interval_observation_policy(request, *, binding_identity, header_sha256, backend_modules,
                                        charge_refinement=None):
    """A concrete review packet; preparing it gives no native authorization."""
    policy = {"schema": "solarlab.interval-observation-policy.v1",
            "map_identity": request["map_identity"],
            "protocol_sha256": digest(request["segments"]),
            "controls_sha256": digest(request["controls"]),
            "budgets_sha256": digest(request["budgets"]),
            "sampling_sha256": digest((request["observation_times"], request["quadrature"])),
            "binding_identity": binding_identity, "header_sha256": header_sha256,
            "backend_modules": backend_modules,
            "arithmetic": {"python_flint": "0.8.0", "bits": 256, "evaluations": 2048, "depth": 16},
            "quadrature_error_allocation": "charge_C/12 * actual_interval/full_protocol / 32 per integral",
            "sampling": "original requested and 8/16/32 times, explicitly declared polynomial reconstruction",
            "history": "actual accepted predecessor/endpoint plus source-owned immutable phi/psi/Dky and error metadata",
            "qualification": "represented numerical path only; original independent time/state/space refinements remain pending"}
    if charge_refinement is not None:
        if charge_refinement != CHARGE_REFINEMENT or type(request["budgets"].get("native_steps")) is not int or request["budgets"]["native_steps"] != 200000:
            raise ContractError("native_observation_unsupported_charge_refinement")
        p = charge_upper_policy("0"*64)  # Only constants enter the request; runtime source binds its final digest.
        parent = digest(policy)
        policy.update(schema="solarlab.interval-observation-policy.v2", parent_policy_sha256=parent,
            charge_refinement={"name": CHARGE_REFINEMENT, "partition_cells": 4,
                "upper_sum": {"schema": p.schema, "quantum_bits": p.quantum_bits, "max_terms": p.max_terms,
                    "max_increment_bits": p.max_increment_bits, "max_evidence_bytes": p.max_evidence_bytes,
                    "max_slack_per_channel": [p.max_slack_per_channel.numerator, p.max_slack_per_channel.denominator],
                    "channel_ids": list(p.channel_ids)},
                "prefix_semantics": "exact signed local evidence; non-resetting upward dyadic upper bounds; original gates"})
    return policy


def validate_interval_observation_admission(request, admission):
    """Require a matching independent review before importing the backend.

    The historic full-word checkpoint stays unadmitted. A separately frozen
    observation policy, executing sources, backend and review must all match;
    setting that old Boolean cannot open this controller.
    """
    import json

    policy = request.get("interval_observation")
    if (not isinstance(policy, Mapping) or admission.get("interval_observation_authorized") is not True
            or admission.get("interval_observation_policy_sha256") != digest(policy)):
        raise ContractError("full_word_native_observation_policy_unqualified")
    try:
        expected = prepare_interval_observation_policy(
            request, binding_identity=policy["binding_identity"], header_sha256=policy["header_sha256"],
            backend_modules=policy["backend_modules"],
            charge_refinement=policy["charge_refinement"]["name"] if "charge_refinement" in policy else None)
    except (KeyError, TypeError) as exc:
        raise ContractError("native_observation_policy_incomplete") from exc
    if (digest(expected) != digest(policy) or any(not isinstance(policy[k], str) or len(policy[k]) != 64
                                                  for k in ("binding_identity", "header_sha256"))):
        raise ContractError("native_observation_policy_changed")
    source_pins = admission.get("source_sha256", {})
    names = ("native_observation.py", "interval_observation.py", "coupled_device_prototype.py")
    if "charge_refinement" in policy:
        names += ("bounded_observation.py", "native_history.py")
    required = tuple(str(Path(__file__).with_name(name).resolve()) for name in names)
    if any(path not in source_pins or sha256(Path(path).read_bytes()).hexdigest() != source_pins[path]
           for path in required):
        raise ContractError("native_observation_executing_sources_not_bound")
    review_receipt = admission.get("interval_observation_review", {})
    try:
        review_bytes = Path(review_receipt["path"]).read_bytes()
        review = json.loads(review_bytes)
    except (KeyError, OSError, ValueError, TypeError) as exc:
        raise ContractError("native_observation_independent_review_missing") from exc
    if (sha256(review_bytes).hexdigest() != review_receipt.get("sha256")
            or review.get("verdict") != "accepted_for_bounded_full_protocol_observation"
            or review.get("policy_sha256") != digest(policy)
            or any(review.get("source_sha256", {}).get(path) != source_pins[path] for path in required)):
        raise ContractError("native_observation_independent_review_mismatch")
    modules = policy["backend_modules"]
    if (not isinstance(modules, Mapping) or not {"sksundae.ida", "sksundae._cy_ida", "flint"} <= modules.keys()
            or any(not isinstance(row, Mapping) or set(row) != {"path", "sha256"}
                   or sha256(Path(row["path"]).read_bytes()).hexdigest() != row["sha256"]
                   for row in modules.values())):
        raise ContractError("native_observation_backend_files_not_bound")
    return dict(policy)


def require_loaded_observation_backend(policy):
    """Check public module provenance, with no native-memory inspection."""
    import importlib

    for name, pin in policy["backend_modules"].items():
        module = importlib.import_module(name)
        if Path(module.__file__).resolve() != Path(pin["path"]).resolve():
            raise ContractError("native_observation_loaded_backend_mismatch:"+name)
    ida = importlib.import_module("sksundae.ida").IDA
    if not callable(getattr(ida, "last_step_snapshot", None)):
        raise ContractError("native_observation_snapshot_capability_missing")
