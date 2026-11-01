"""T1 V2/V3 algebra over sealed, source-bound observation evidence.

This module does not integrate a model or admit a run.  The ordinary observer
provides signed charge/flux enclosures; physical |F| bounds and fixed-window
pieces are additional evidence, never inferred from raw/tangent defect L1.
A valid lower bound can prove pass without an upper bound; otherwise missing
scale/piece/reference evidence stays unknown. Evaluation requires its own
reviewed-source admission, separate from the producer's frozen definitions.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import resource
import sys
import time


CHANNELS = ("device", "left_metal", "right_metal")
COMPONENTS = ("saved_charge_evaluation", "physical_integrand_and_quadrature", "input_representation", "clock_strip")
SPEC_SCHEMA = "solarlab.g2-route.t1-gatespec.v2"
SPEC_SCHEMA_V3 = "solarlab.g2-route.t1-gatespec.v3"
EVIDENCE_SCHEMA = "solarlab.t1-candidate-evidence.v1"
MAX_BITS = 131072
INTEGER_POLICY = {"schema": "solarlab.native-integer-io.v1", "codec": "decimal-with-bounded-hex-v1",
                  "decimal_digits": 4300, "max_integer_bits": MAX_BITS}


class InvalidT1Data(ValueError):
    pass


def require(value, reason):
    if not value:
        raise InvalidT1Data(reason)


def q(value):
    """Exact rational input; decimal binary64 approximations are not operands."""
    if type(value) is Fraction:
        result = value
    elif type(value) is int:
        result = Fraction(value)
    elif type(value) in (list, tuple) and len(value) == 2:
        require(all(type(v) is int for v in value) and value[1] > 0, "rational_integer_pair")
        result = Fraction(*value)
    elif type(value) is str:
        require(len(value) <= 4300, "rational_text_bound")
        try:
            result = Fraction(value)
        except (ValueError, ZeroDivisionError) as error:
            raise InvalidT1Data("rational_text") from error
    else:
        raise InvalidT1Data("rational_type_not_float_or_bool")
    require(max(abs(result.numerator).bit_length(), result.denominator.bit_length()) <= MAX_BITS, "rational_bit_bound")
    return result


def pair(value):
    value = q(value)
    return [value.numerator, value.denominator]


def digest(value):
    from scripts.benchmarks.native_history import integer_json_dumps
    return hashlib.sha256(integer_json_dumps(value, INTEGER_POLICY, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def enclosure(value):
    require(type(value) is dict and set(value) == {"center", "absolute_error_bound"}, "enclosure_fields")
    center, radius = q(value["center"]), q(value["absolute_error_bound"])
    require(radius >= 0, "negative_radius")
    return center, radius


def abs_bounds(center, radius):
    return max(Fraction(0), abs(center)-radius), abs(center)+radius


@dataclass(frozen=True)
class CandidateSpec:
    sha256: str
    Qref: Fraction
    epsilon_metal: Fraction = Fraction(1, 1000)
    epsilon_device: Fraction = Fraction(1, 1000)
    eta: Fraction = Fraction(1, 100)
    quadrature_share: Fraction = Fraction(1, 3)
    version: int = 2

    @property
    def Q0(self):
        return self.Qref / 10**8

    @classmethod
    def from_document(cls, document, sha256):
        version = document.get("version")
        require(type(version) is int and version in (2,3)
                and document.get("schema") == (SPEC_SCHEMA if version == 2 else SPEC_SCHEMA_V3), "candidate_spec_V2_V3")
        p = document["parameters"]
        expected = {"epsilon_metal": Fraction(1, 1000), "epsilon_device_proposal": Fraction(1, 1000),
                    "device_signal_floor_fraction": Fraction(1, 100), "quadrature_share": Fraction(1, 3),
                    "reference_error_share": Fraction(1, 3), "legacy_absolute_fraction": Fraction(1, 10**8),
                    "resolution_multiplier": Fraction(3)}
        require(all(q(p[k]) == v for k, v in expected.items()), "candidate_parameters_changed")
        require(all(type(p[k]) is int and p[k] == v for k, v in
                    (("denominator_partition_max_depth", 8), ("denominator_partition_max_leaves", 256))),
                "candidate_partition_parameters_changed")
        result = cls(sha256, q(document["physical_context"]["Qref_C_exact"]), version=version)
        require(result.Qref > 0 and result.Q0 == q(document["physical_context"]["legacy_absolute_C_exact"]), "input_Q0_binding")
        return result


def verdict(defect, quadrature, lower, upper, epsilon, share=Fraction(1, 3)):
    """Known numerator, lower denominator for pass, upper for definite failure."""
    defect, quadrature, lower = map(q, (defect, quadrature, lower))
    epsilon, share = q(epsilon), q(share)
    require(defect >= 0 and quadrature >= 0 and lower >= 0, "nonnegative_gate_operand")
    upper = None if upper is None else q(upper)
    require((upper is None or upper >= lower) and epsilon > 0 and 0 < share <= 1, "gate_scale_order")
    passed = defect <= epsilon*lower and quadrature <= epsilon*lower*share
    failed = upper is not None and (defect > epsilon*upper or quadrature > epsilon*upper*share)
    return {"status": "passed" if passed else "failed" if failed else "unknown",
            "reason": "within_lower_scale" if passed else "certificate_budget_exceeded" if failed else "unknown_scale",
            "defect_C": pair(defect), "quadrature_C": pair(quadrature),
            "scale_lower_C": pair(lower), "scale_upper_C": None if upper is None else pair(upper),
            "defect_margin_lower_C": pair(epsilon*lower-defect),
            "quadrature_margin_lower_C": pair(epsilon*lower*share-quadrature)}


def operands(row):
    """Interpret the six GateSpec operands, with inseparable integral error.

    The current observer reports a combined integrand/quadrature radius, so
    this candidate charges all uF to the one-third sub-budget.  A smaller
    separated quadrature estimate requires a future independently checked
    decomposition; it cannot be enabled by a label in the data.
    """
    names = ("delta_Q_enclosure_C", "integral_F_enclosure_C", "uncertainty_components_C", "quadrature_component_upper_C")
    if any(row.get(k) is None for k in names):
        return None
    dc, dr = enclosure(row["delta_Q_enclosure_C"])
    fc, fr = enclosure(row["integral_F_enclosure_C"])
    components = row["uncertainty_components_C"]
    require(type(components) is dict and set(components) == set(COMPONENTS), "four_uncertainty_components")
    if any(components[k] is None for k in COMPONENTS):
        return None
    values = {k: q(components[k]) for k in COMPONENTS}
    require(all(v >= 0 for v in values.values()), "negative_uncertainty")
    require(values[COMPONENTS[0]] >= dr and values[COMPONENTS[1]] >= fr, "enclosure_error_not_covered")
    quad = q(row["quadrature_component_upper_C"])
    require(quad == values[COMPONENTS[1]], "inseparable_quadrature_must_charge_combined_uF")
    lower = row.get("integral_abs_F_lower_C")
    upper = row.get("integral_abs_F_upper_C")
    # The signed-integral lower bound also covers the additional input/clock
    # uncertainty.  Q's own enclosure and Q0 remain valid even if this is zero.
    signed_lower = max(Fraction(0), abs(fc)-values[COMPONENTS[1]]-values[COMPONENTS[2]]-values[COMPONENTS[3]])
    if lower is not None:
        lower = q(lower)
        require(lower >= 0, "negative_L1_lower")
    lower = signed_lower if lower is None else max(lower, signed_lower)
    upper = None if upper is None else q(upper)
    require(lower >= 0 and (upper is None or upper >= lower), "physical_flux_L1_bounds")
    if row.get("integral_abs_F_evidence") is not None:
        _check_L1_proof(row)
    return {"dc": dc, "dr": dr, "fc": fc, "fr": fr, "b": dc-fc,
            "u": sum(values.values(), Fraction(0)), "components": values, "quad": quad,
            "qabs": abs_bounds(dc, dr), "l1_lower": lower, "l1_upper": upper}


def _check_L1_proof(row):
    """Check the retained sign/range algebra, never promote an upper to lower."""
    from scripts.benchmarks.native_observation import restore_observation_record
    proof = restore_observation_record(row["integral_abs_F_evidence"])
    require(proof["schema"] == "solarlab.physical-flux-L1.v1", "L1_proof_schema")
    cells = proof["cells"]
    require(type(proof["leaves"]) is int and proof["leaves"] == len(cells) and 1 <= len(cells) <= 256
            and type(proof["depth"]) is int and type(proof["max_depth"]) is int
            and 0 <= proof["depth"] <= proof["max_depth"] <= 8
            and type(proof["max_leaves"]) is int and len(cells) <= proof["max_leaves"] <= 256, "L1_proof_partition")
    root_left,root_right = q(cells[0]["left"]),q(cells[-1]["right"])
    lower,upper,previous = Fraction(0),Fraction(0),None
    for cell in cells:
        a,b = q(cell["left"]),q(cell["right"])
        require(a < b and (previous is None or a == previous) and type(cell["depth"]) is int
                and 0 <= cell["depth"] <= proof["depth"], "L1_proof_cell_cover")
        require(b-a == (root_right-root_left)/2**cell["depth"] and ((a-root_left)/(b-a)).denominator == 1,
                "L1_proof_dyadic_partition")
        previous = b
        if cell["value_range"] is None:
            lo,hi = Fraction(0),None
            require(cell["signed_integral"] is None and cell["sign"] == "unknown", "L1_nonfinite_cell")
        else:
            rc,rr = enclosure(cell["value_range"])
            range_upper = (b-a)*(abs(rc)+rr)
            if cell["signed_integral"] is None:
                lo,hi = Fraction(0),range_upper
                require(cell["sign"] == "unknown", "L1_missing_signed_integral")
            else:
                c,r = enclosure(cell["signed_integral"])
                sign = "nonnegative" if rc-rr >= 0 else "nonpositive" if rc+rr <= 0 else "unknown"
                require(cell["sign"] == sign, "L1_sign_certificate")
                lo = max(Fraction(0),abs(c)-r)
                hi = min(range_upper,abs(c)+r) if sign != "unknown" else range_upper
        require(q(cell["lower"]) == lo and (cell["upper"] is None if hi is None else q(cell["upper"]) == hi), "L1_cell_bounds")
        lower += lo;upper = None if upper is None or hi is None else upper+hi
    require(q(proof["lower"]) == lower and (proof["upper"] is None if upper is None else q(proof["upper"]) == upper), "L1_sum_bounds")
    extra = q(row["scale_input_clock_widening_C"])
    components = row["uncertainty_components_C"]
    required = q(components["input_representation"])+q(components["clock_strip"])
    require(extra >= required, "L1_input_clock_coverage")
    require(extra >= 0 and q(row["integral_abs_F_lower_C"]) == max(Fraction(0),lower-extra)
            and (row["integral_abs_F_upper_C"] is None if upper is None else q(row["integral_abs_F_upper_C"]) == upper+extra),
            "L1_input_clock_widening")
    scope = row["L1_scope"]
    h,tn = q(scope["hused"]),q(scope["native_tn"])
    require(h > 0 and q(cells[0]["left"])*h+tn == q(scope["t_left"])
            and q(cells[-1]["right"])*h+tn == q(scope["t_right"]), "L1_proof_physical_clock")


def _channels_v3(row):
    a,b = _clock(row)
    from scripts.benchmarks.native_observation import restore_observation_record
    native_clock = restore_observation_record(row["native_clock"])
    h,tn,previous = (q(native_clock[k]) for k in ("hused","tn","predecessor"))
    require(h > 0 and previous <= a < b <= tn,"V3_native_clock_coverage")
    for channel in row["channels"].values():
        proof = channel.get("integral_abs_F_evidence")
        require(proof is not None or channel.get("integral_abs_F_lower_C") is None
                and channel.get("integral_abs_F_upper_C") is None, "V3_unproved_L1_scale")
        if proof is not None:
            scope = channel["L1_scope"]
            require(q(scope["t_left"]) == a and q(scope["t_right"]) == b
                    and q(scope["hused"]) == h and q(scope["native_tn"]) == tn
                    and scope["path_identity"] == row["path_identity"], "V3_L1_interval_scope")
        ownership = channel.get("coverage_ownership")
        require(type(ownership) is dict and set(ownership) == set(COMPONENTS)|{"Qquad"}
                and all(type(x) is str and x for x in ownership.values()), "V3_uncertainty_ownership")


class ChargeLedger:
    """One nonresetting positive debt and one floor, in exact C units."""

    def __init__(self):
        self.count = 0
        self.complete = True
        self.dc = self.dr = self.b = self.u = self.D = self.quad = Fraction(0)
        self.q_lower = self.q_upper = self.f_lower = Fraction(0)
        self.f_upper = Fraction(0)
        self.old_scale_lower = self.old_scale_upper = Fraction(0)

    def append(self, value, floor):
        self.count += 1
        if value is None:
            self.complete = False
            return
        self.dc += value["dc"]; self.dr += value["dr"]
        self.b += value["b"]; self.u += value["u"]
        self.D += abs(value["b"])+value["u"]; self.quad += value["quad"]
        self.q_lower += value["qabs"][0]; self.q_upper += value["qabs"][1]
        self.f_lower += value["l1_lower"]
        self.f_upper = (None if self.f_upper is None or value["l1_upper"] is None else self.f_upper+value["l1_upper"])
        self.old_scale_lower += max(value["qabs"][0], value["l1_lower"], floor)
        self.old_scale_upper = (None if self.old_scale_upper is None or value["l1_upper"] is None else
                                self.old_scale_upper+max(value["qabs"][1], value["l1_upper"], floor))
        for number in (self.dc, self.dr, self.b, self.u, self.D, self.quad, self.q_lower, self.q_upper,
                       self.f_lower, self.f_upper, self.old_scale_lower, self.old_scale_upper):
            if number is not None:
                q(number)  # bounded exact arithmetic, never round away a low word

    def result(self, *, floor, epsilon, share=Fraction(1, 3), cap=None, window=False, window_charge=None):
        if not self.complete or self.count == 0:
            return {"status": "unknown", "reason": "unknown_uncertainty_or_missing_records"}
        lo, hi = (abs_bounds(*enclosure(window_charge)) if window_charge is not None else
                  abs_bounds(self.dc, self.dr) if window else (self.q_lower, self.q_upper))
        lower = max(lo, self.f_lower, floor)
        upper = None if self.f_upper is None else max(hi, self.f_upper, floor)
        if cap is not None:
            lower = min(lower, cap)
            upper = cap if upper is None else min(upper, cap)
        result = verdict(self.D, self.quad, lower, upper, epsilon, share)
        result.update(signed_balance_C=pair(self.b), uncertainty_C=pair(self.u), terms=self.count,
                      floor_occurrences=1, scale_cap_C=None if cap is None else pair(cap),
                      legacy_repeated_floor_scale_lower_C=pair(self.old_scale_lower),
                      legacy_repeated_floor_scale_upper_C=None if self.old_scale_upper is None else pair(self.old_scale_upper))
        return result


def assess_interval(row, *, floor, epsilon, share=Fraction(1, 3)):
    value = operands(row)
    ledger = ChargeLedger(); ledger.append(value, q(floor))
    return ledger.result(floor=q(floor), epsilon=q(epsilon), share=q(share))


def observed_contraction(coarse_main, main_strict):
    """A finite-level diagnostic; no tail/forward-error qualification."""
    cmlo, cmhi = map(q, coarse_main); mslo, mshi = map(q, main_strict)
    require(0 <= cmlo <= cmhi and 0 <= mslo <= mshi, "difference_bounds")
    state = ("exact_zero" if cmhi == mshi == 0 else "certified_observed_contraction" if cmlo > 0 and mshi <= cmlo/2
             else "resolved_no_contraction" if mslo > cmhi/2 else "unknown_contraction")
    return {"status": state, "forward_error_qualified": False, "heuristic_two_dMS": pair(2*mshi)}


def matched_growth(coarse_b, coarse_u, finer_b, finer_u):
    """Signed matched-window trend, separate from the positive prefix debt."""
    if any(value is None for value in (coarse_b, coarse_u, finer_b, finer_u)):
        return {"status": "unknown", "reason": "unknown_matched_window_uncertainty"}
    a, ua, b, ub = map(q, (coarse_b, coarse_u, finer_b, finer_u))
    require(ua >= 0 and ub >= 0, "matched_window_uncertainty")
    growth = max(Fraction(0), abs(b)-abs(a))
    return {"status": "passed" if growth <= ua+ub else "failed", "resolved_growth_C": pair(growth),
            "allowance_C": pair(ua+ub), "convergence_order_proved": False}


def resolution_conditions(Qsig, E_q, E_f, spec):
    """Numerical conditions only; evidence eligibility is a separate check."""
    signal, eq, ef = map(q, (Qsig, E_q, E_f))
    require(signal >= 0 and eq >= 0 and ef >= 0, "reference_nonnegative")
    error = max(eq, ef)
    return signal > 0 and signal > 3*error and error <= spec.epsilon_device*spec.eta*signal/3


def _pin(path, expected, *, limit=8*1024**2):
    path = Path(path)
    require(path.is_file() and path.stat().st_size <= limit, "metadata_file_bound")
    raw = path.read_bytes()
    require(len(raw) == expected["bytes"] and hashlib.sha256(raw).hexdigest() == expected["sha256"], "file_identity")
    from scripts.benchmarks.native_history import decode_integer_values
    return decode_integer_values(json.loads(raw),INTEGER_POLICY)


def physical_problem_identity(request):
    """Content identity shared by profiles, retaining every initial word/rate."""
    packet, frame = request["numeric_packet"], request["initial_segment_frame"]
    mapping = request["voltage_lift_map"]
    return digest({"definition": packet["definition"], "x_m": packet["x_m"],
        "physical_reference_fields": packet["physical_reference_fields"], "segments": request["segments"],
        "initial_state_words": frame["physical_handoff_words_hex"],
        "initial_rate_words": frame["physical_rate_handoff_words_hex"],
        "lift_hex": mapping["lift_hex"], "reference_inputs_hex": mapping["reference_inputs_hex"]})


def reference_status(reference, *, spec, labels, problem_identity):
    """Bind independently reviewed q/f forward-error evidence, never a flag.

    This gate checks the pinned proof/review and norm coverage contract.  The
    mathematical validity of that independently reviewed proof remains an
    explicit external prerequisite, not something inferred from its label.
    """
    if spec.version == 3:
        return reference_status_v3(reference,spec=spec,labels=labels,problem_identity=problem_identity)
    unknown = {"status": "unknown", "reason": "unknown_reference_error", "Qsig_C": None}
    if not reference:
        return unknown
    try:
        require(reference["case_id"] == "B_N8_Strict_Full" and reference["problem_identity"] == problem_identity,
                "reference_problem")
        require(reference["unit"] == "C" and digest(reference["labels"]) == digest(labels) and len(labels) == 131, "reference_coverage")
        kind = reference["proof_kind"]
        require(kind in {"analytic_original_problem", "validated_IVP_enclosure", "defect_stability_bound", "independently_bounded_reference"},
                "reference_method_not_forward_bound")
        proof = _pin(reference["proof"]["path"], reference["proof"])
        review = _pin(reference["independent_review"]["path"], reference["independent_review"])
        require(proof["problem_identity"] == problem_identity and proof["kind"] == kind
                and proof["unit"] == "C" and proof["labels_sha256"] == digest(labels), "reference_proof_binding")
        required = {"evaluation", "initial_and_algebraic", "input_clock_parameter", "trajectory_nonlinear", "stability_reconstruction"}
        require(set(proof["error_coverage"]) == required, "reference_remainder_coverage")
        evidence_digest = digest({k: v for k, v in reference.items() if k != "independent_review"})
        require(review.get("accepted") is True and review.get("independent_forward_error_review") is True
                and review["evidence_sha256"] == evidence_digest and review["proof_sha256"] == reference["proof"]["sha256"]
                and review["gate_spec_sha256"] == spec.sha256, "reference_independent_review_binding")
        rows = reference["responses"]
        require(len(rows) == 131 and all(digest(row["tag"]) == digest(tag) for row, tag in zip(rows, labels, strict=True)), "reference_response_tags")
        require(all(enclosure(rows[0][k]) == (0, 0) for k in ("q_C", "f_C")), "reference_initial_zero")
        for row in rows:
            require(q(row["E_q_C"]) >= 0 and q(row["E_f_C"]) >= 0, "reference_error_bounds")
        signal = min(max(abs_bounds(*enclosure(row[k]))[0] for row in rows) for k in ("q_C", "f_C"))
        eq, ef = (max(q(row[k]) for row in rows) for k in ("E_q_C", "E_f_C"))
        require(proof["bound_table_sha256"] == digest(rows), "reference_bound_table")
        if not resolution_conditions(signal, eq, ef, spec):
            return {"status": "unknown", "reason": "unresolved_reference_signal", "Qsig_C": pair(signal),
                    "E_q_C": pair(eq), "E_f_C": pair(ef)}
        return {"status": "passed", "reason": "independent_reference_contract_and_resolution",
                "Qsig_C": pair(signal), "E_q_C": pair(eq), "E_f_C": pair(ef),
                "external_proof_review": reference["independent_review"], "proof_recomputed_by_this_gate": False,
                "independent_proof_validity_is_external_prerequisite": True}
    except (KeyError, TypeError, ValueError, OSError):
        return unknown


def _acceptance_pin(reference, *, decode=True):
    """A5 FilePin: positive byte count and complete bytes, not a future path."""
    require(type(reference) is dict and {"path", "sha256", "bytes"} <= set(reference)
            and type(reference["path"]) is str and reference["path"]
            and Path(reference["path"]).is_absolute()
            and type(reference["bytes"]) is int and reference["bytes"] > 0
            and type(reference["sha256"]) is str and len(reference["sha256"]) == 64
            and all(c in "0123456789abcdef" for c in reference["sha256"]), "acceptance_file_pin")
    if decode:
        require(Path(reference["path"]).suffix == ".json", "acceptance_JSON_metadata_required")
        value = _pin(reference["path"], reference)
        require(type(value) is dict, "acceptance_JSON_root")
        return value
    path = Path(reference["path"])
    require(path.is_file() and path.stat().st_size <= 8*1024**2, "acceptance_file_bound")
    raw = path.read_bytes()
    require(len(raw) == reference["bytes"] and hashlib.sha256(raw).hexdigest() == reference["sha256"],
            "acceptance_file_identity")
    return raw


def _acceptance_pair(value):
    require(type(value) is list and len(value) == 2 and all(type(x) is int for x in value)
            and value[1] > 0, "acceptance_rational_pair")
    result = q(value)
    require(result >= 0, "acceptance_nonnegative_error")
    return result


def _acceptance_quantity(key):
    require(type(key) is str, "acceptance_quantity_key")
    if key in {"q", "f"}:
        return key, "C"
    if key in {"port/left", "port/right"}:
        return "port_" + key.split("/")[1], "A"
    parts = key.split("/")
    units = {"phi_V": "V", "n_m3": "m^-3", "p_m3": "m^-3", "c_m3": "m^-3", "f": "1"}
    require(len(parts) == 3 and parts[0] == "state" and parts[1] in units
            and parts[2] in {str(i) for i in range(9)}, "acceptance_quantity_key")
    return parts[1], units[parts[1]]


def _acceptance_header(review, schema, required):
    require(type(review) is dict and required <= set(review), "acceptance_required_fields")
    require(review["schema"] == schema and type(review["version"]) is int and review["version"] == 1
            and review["scientific_pass"] is False and review["execution_authorized"] is False,
            "acceptance_version_or_scope_flags")
    require(review["claim_kind"] == "conditional_uncertainty_estimate", "acceptance_claim_upgrade")
    reviewer = review["reviewer"]
    require(type(reviewer) is dict and {"id", "role", "evidence_author", "independent", "review_record"} <= set(reviewer)
            and all(type(reviewer[key]) is str and reviewer[key].strip() for key in ("id", "role", "evidence_author"))
            and reviewer["role"] == "independent_science" and reviewer["independent"] is True
            and reviewer["id"] != reviewer["evidence_author"], "acceptance_independent_reviewer")
    _acceptance_pin(reviewer["review_record"], decode=False)
    keys = review["accepted_quantities"]
    require(type(keys) is list and all(type(key) is str for key in keys)
            and keys == sorted(set(keys)) and len(keys) <= 49
            and type(review["quantity_bindings"]) is dict and set(review["quantity_bindings"]) == set(keys),
            "acceptance_exact_quantity_bindings")
    for key in keys:
        _acceptance_quantity(key)
    require(type(review["verdict"]) is str and review["verdict"] in {"accepted", "rejected", "unknown"}
            and (bool(keys) if review["verdict"] == "accepted" else not keys), "acceptance_verdict_quantities")
    conditions, seen = review["conditions"], set()
    require(type(conditions) is list, "acceptance_conditions")
    for condition in conditions:
        require(type(condition) is dict and {"id", "statement", "status", "evidence"} <= set(condition)
                and type(condition["id"]) is str and condition["id"].strip() and condition["id"] not in seen
                and type(condition["statement"]) is str and condition["statement"].strip()
                and type(condition["status"]) is str
                and condition["status"] in {"satisfied", "supported_under_U1", "open", "violated", "not_applicable"}
                and type(condition["evidence"]) is list, "acceptance_condition_fields")
        seen.add(condition["id"])
        for evidence in condition["evidence"]:
            _acceptance_pin(evidence, decode=False)
        if review["verdict"] == "accepted":
            require(condition["status"] not in {"open", "violated"}, "acceptance_condition_unresolved:" + condition["id"])
    require(review["verdict"] == "accepted", "acceptance_verdict_" + review["verdict"])
    return keys


def _acceptance_history(binding, *, case_id, labels, expected=None):
    """A5 full HistoryBinding, using the existing method-specific receipts."""
    fields = {"case_id", "run_id", "source_identity", "input_sha256", "run_result", "history_manifest",
              "independent_readback", "logical_sha256", "logical_bytes", "records", "encoded_bytes",
              "closed", "container_complete", "labels_sha256"}
    require(type(binding) is dict and fields <= set(binding)
            and binding["case_id"] == case_id and binding["closed"] is True and binding["container_complete"] is True
            and all(type(binding[key]) is str and binding[key] for key in ("run_id", "source_identity", "input_sha256"))
            and all(type(binding[key]) is int and binding[key] > 0 for key in ("logical_bytes", "encoded_bytes", "records"))
            and binding["labels_sha256"] == digest(labels), "acceptance_history_fields")
    for key in ("logical_sha256", "input_sha256", "labels_sha256"):
        require(type(binding[key]) is str and len(binding[key]) == 64
                and all(c in "0123456789abcdef" for c in binding[key]), "acceptance_history_digest")
    if expected is not None:
        require(all(binding.get(key) == value for key, value in expected.items()), "acceptance_history_source_identity")
    result = _acceptance_pin(binding["run_result"])
    descriptor = _acceptance_pin(binding["history_manifest"])
    reader = _acceptance_pin(binding["independent_readback"])
    history = descriptor.get("observed_history", descriptor.get("history", descriptor))
    require(type(history) is dict and history.get("closed") is True and history.get("writer_error") is None,
            "acceptance_history_closed")
    names = ("logical_sha256", "logical_bytes", "encoded_bytes", "records")
    require(all(type(history.get(key)) is type(binding[key]) and history[key] == binding[key] for key in names),
            "acceptance_history_descriptor")
    be = case_id.startswith("BE_")
    if be:
        require(reader.get("schema") == "solarlab.g2-reference-be-history.v1"
                and result.get("schema") == "solarlab.g2-reference-be-history.v1"
                and reader.get("method") == result.get("method") == "conservative-backward-euler"
                and reader.get("structural_pass") is True and reader.get("numerical_complete") is True
                and reader.get("case_id") == case_id and type(reader.get("tagged_observations")) is int
                and reader["tagged_observations"] == len(labels)
                and digest([row["tag"] for row in reader["observations"]]) == digest(labels)
                and result.get("numerical_complete") is True and result.get("status") == "completed",
                "acceptance_BE_semantic_readback")
        actual = reader["observed_history"]
        identity = {key: result[key] for key in ("case_id", "input_sha256", "method_contract_sha256", "plan_sha256",
                    "source_commit", "environment_identity", "schedule_identity", "source_identity",
                    "initial_point_identity", "controls_sha256")}
        require(all(type(value) is str and value for value in identity.values()) and digest(identity) == binding["run_id"]
                and identity.get("case_id") == case_id and identity.get("source_identity") == binding["source_identity"]
                and identity.get("input_sha256") == binding["input_sha256"], "acceptance_BE_run_identity")
    else:
        require(reader.get("schema") == "solarlab.native-history-verification.v1"
                and reader.get("verified") is True and reader.get("complete_protocol") is True
                and reader.get("complete_readback") is True and reader.get("case_id") == case_id
                and reader.get("request_sha256") == binding["input_sha256"]
                and result.get("request_sha256") == binding["input_sha256"]
                and result.get("status") == "completed_bounded_voltage_lift_native_pilot"
                and result.get("complete_protocol") is True, "acceptance_IDA_semantic_readback")
        actual = reader["history"]
        require(actual.get("eof_seen") is True and actual.get("crc_verified") is True
                and actual.get("container_complete") is True and type(actual.get("partial_tail_bytes")) is int
                and actual["partial_tail_bytes"] == 0,
                "acceptance_IDA_container")
        operations = [result.get(name, {}).get("admission_binding") for name in
                      ("execution_wall_policy", "execution_output_policy")]
        require(any(type(item) is dict and item.get("operation_id") == binding["run_id"] for item in operations)
                or result.get("run_id") == binding["run_id"], "acceptance_IDA_run_identity")
    require(type(actual) is dict and all(type(actual.get(key)) is type(binding[key])
            and actual[key] == binding[key] for key in names), "acceptance_history_readback")
    sealed = result["observed_history" if be else "history"]
    require(type(sealed) is dict and sealed.get("closed") is True and sealed.get("writer_error") is None
            and all(type(sealed.get(key)) is type(binding[key]) and sealed[key] == binding[key] for key in names),
            "acceptance_history_run_result")
    for key in ("container_sha256", "encoded_sha256", "encoding"):
        present = [doc[key] for doc in (binding, history, actual, sealed) if key in doc]
        require(not present or all(value == present[0] for value in present), "acceptance_container_identity")
    return binding


def _acceptance_quantity_binding(binding, record, record_pin, *, key, labels, strict=False):
    name, unit = _acceptance_quantity(key)
    require(type(binding) is dict and {"reviewed_record", "coverage", "claim_kind", "value", "radius_accounting"} <= set(binding)
            and digest(binding["reviewed_record"]) == digest(record_pin)
            and binding["claim_kind"] == record["claim_kind"] == "conditional_uncertainty_estimate",
            "acceptance_quantity_binding")
    actual = _acceptance_pin(binding["reviewed_record"])
    require(digest(actual) == digest(record), "acceptance_reviewed_quantity_bytes")
    coverage = binding["coverage"]
    require(type(coverage) is dict and {"kind", "records", "labels_sha256", "unit", "norm"} <= set(coverage)
            and coverage["kind"] == "full_protocol" and type(coverage["records"]) is int
            and coverage["records"] == len(labels) == 131 and coverage["labels_sha256"] == digest(labels)
            and coverage["unit"] == unit and coverage["norm"] == record["norm_and_coverage"]["norm"]
            and record["quantity_and_unit"] == {"quantity": name, "unit": unit}, "acceptance_quantity_coverage")
    require(_acceptance_pair(binding["value"]) == _acceptance_pair(record["conditional_estimate_value"]),
            "acceptance_quantity_value")
    ownership = binding["radius_accounting"]
    require(type(ownership) is dict and {"r_R_included_in_E_R", "r_R_source", "tagwise_ownership_record"} <= set(ownership)
            and ownership["r_R_included_in_E_R"] is True
            and ownership["r_R_source"] == "per_record/<tag>/computed_value/radius",
            "acceptance_radius_accounting")
    proof = _acceptance_pin(ownership["tagwise_ownership_record"])
    require(proof.get("quantity_key") == key and digest(proof.get("per_record")) == digest(record["per_record"]),
            "acceptance_tagwise_ownership_record")
    require(digest([row["tag"] for row in record["per_record"]]) == digest(labels), "acceptance_tagwise_labels")
    if strict:
        require(ownership.get("r_S_included_in_E_S") is True, "acceptance_Strict_radius_ownership")
    return ownership


def _accepted_strict_E2(reference, *, spec, labels, problem_identity):
    require(type(reference) is dict and reference.get("schema") == "solarlab.g2-strict-reference.v1",
            "Strict_reference_schema")
    e2 = _acceptance_pin(reference["e2_file"])
    review = _acceptance_pin(reference["independent_review"])
    require(e2.get("schema") == "solarlab.g2-E2-transfer.v1"
            and e2.get("target_problem_identity") == problem_identity
            and digest(e2.get("labels")) == digest(labels) and len(labels) == 131
            and e2["gate_spec"]["sha256"] == spec.sha256, "Strict_E2_scope")
    required = {"schema", "version", "reviewed_e2_file", "inputs", "accepted_quantities", "quantity_bindings",
                "claim_kind", "reviewer", "verdict", "conditions", "target_problem_identity", "same_problem_evidence",
                "scope", "scientific_pass", "execution_authorized"}
    accepted = _acceptance_header(review, "solarlab.g2-strict-reference-acceptance.v1", required)
    require(digest(review["reviewed_e2_file"]) == digest(reference["e2_file"])
            and review["target_problem_identity"] == problem_identity
            and review["scope"] == {"route":"G2-R", "reference_case_id":"B_N8_Strict_Full", "coverage":"full131, both event sides"}
            and type(e2.get("evidence_author")) is str and e2["evidence_author"]
            and review["reviewer"]["evidence_author"] == e2["evidence_author"], "Strict_E2_review")
    inputs = e2["inputs"]
    require(type(inputs) is dict and set(inputs) == {"E_R_acceptance", "IDA_Strict_sealed_history", "BE_reference_history"}
            and digest(review["inputs"]) == digest(inputs), "Strict_E2_input_binding")
    identity = e2["strict_source_and_run_identity"]
    require(type(identity) is dict and identity.get("case_id") == "B_N8_Strict_Full"
            and all(type(identity.get(k)) is str and identity[k] for k in ("source_id", "request_sha256", "run_id")),
            "Strict_E2_execution_identity")
    strict_history = _acceptance_history(inputs["IDA_Strict_sealed_history"], case_id="B_N8_Strict_Full", labels=labels,
        expected={"source_identity":identity["source_id"], "input_sha256":identity["request_sha256"], "run_id":identity["run_id"]})
    er_review = _acceptance_pin(inputs["E_R_acceptance"])
    from scripts.benchmarks.g2_compare import read_error_records
    er = read_error_records(er_review["reviewed_error_file"], inputs["E_R_acceptance"],
        method_pin=e2["method_contract"], problem_identity=e2["BE_error_problem_identity"], labels=labels)
    require(er["status"] == "accepted_for_E2", "Strict_E2_E_R_acceptance:" + str(er.get("reason")))
    require(digest(inputs["BE_reference_history"]) == digest(er["reference_history"]), "Strict_E2_same_L2_history")
    proof = _acceptance_pin(review["same_problem_evidence"])
    require(proof.get("target_problem_identity") == problem_identity
            and proof.get("reference_error_problem_identity") == e2["BE_error_problem_identity"], "Strict_E2_problem_ID_correspondence")
    strict_request = _acceptance_pin(proof["IDA_Strict_input"])
    reference_request = _acceptance_pin(proof["BE_reference_input"])
    require(digest(strict_request) == strict_history["input_sha256"]
            and digest(reference_request) == er["reference_history"]["input_sha256"]
            and physical_problem_identity(strict_request) == physical_problem_identity(reference_request) == problem_identity,
            "Strict_E2_model_mesh_initial_rates_inputs_protocol")
    require(all(type(request["numeric_packet"].get("intervals")) is int
                and type(request["numeric_packet"].get("nodes")) is int
                and request["numeric_packet"]["intervals"] == 8 and request["numeric_packet"]["nodes"] == 9
                and len(request["numeric_packet"]["x_m"]) == 9
                and all(len(request["initial_segment_frame"][key]) == 12
                        and all(len(row) == 45 for row in request["initial_segment_frame"][key])
                        for key in ("physical_handoff_words_hex", "physical_rate_handoff_words_hex"))
                for request in (strict_request, reference_request)), "Strict_E2_N8_initial_word_shape")
    from scripts.benchmarks.g2_route_bindings import sample_labels
    from scripts.benchmarks.g2_reference_error import target_identity
    source = reference_request["voltage_lift_map"]["physical_model"]
    point = reference_request["initial_preparation"]["point_identity"]
    require(source == strict_request["voltage_lift_map"]["physical_model"] == strict_history["source_identity"]
            and source == er["reference_history"]["source_identity"]
            and point == strict_request["initial_preparation"]["point_identity"]
            and target_identity(source, point, reference_request["segments"]) == e2["BE_error_problem_identity"]
            and digest(sample_labels(strict_request)) == digest(sample_labels(reference_request)) == digest(labels),
            "Strict_E2_source_and_label_correspondence")
    require(type(e2.get("quantity_records")) is dict and set(accepted) <= set(e2["quantity_records"])
            and set(accepted) <= set(e2["quantities"]), "Strict_E2_quantity_descriptors")
    for key in accepted:
        require(key in er["quantities"] and er["quantities"][key]["usable_for_E2"], "Strict_E2_quantity_E_R_not_accepted")
        row, source_record = e2["quantities"][key], er["quantities"][key]["record"]
        require(digest(row["source_BE_acceptance"]) == digest(inputs["E_R_acceptance"])
                and digest(row["source_BE_error"]) == digest(er["quantities"][key]["pin"])
                and digest(row["strict_source_and_run_identity"]) == digest({k:v for k,v in identity.items() if k != "case_id"}),
                "Strict_E2_quantity_source_binding")
        _acceptance_quantity_binding(review["quantity_bindings"][key], row, e2["quantity_records"][key],
                                     key=key, labels=labels, strict=True)
        name, unit = _acceptance_quantity(key)
        strict_E2_values(e2, key, quantity=name, unit=unit)
        if key in {"q", "f"}:
            responses = e2.get("strict_responses")
            require(type(responses) is list and len(responses) == len(labels)
                    and digest([response["tag"] for response in responses]) == digest(labels),
                    "Strict_E2_response_coverage")
        for index, (measured, supporting) in enumerate(zip(row["per_record"], source_record["per_record"], strict=True)):
            require(digest(measured["tag"]) == digest(supporting["tag"]), "Strict_E2_tagwise_support")
            cs, rs = enclosure(measured["strict_value"])
            cr, rr = enclosure(measured["reference_value"])
            owned = measured["ownership"]
            error = _acceptance_pair(supporting["computation"]["E_R"])
            require(cr == q(supporting["computed_value"]["center"])
                    and rr == _acceptance_pair(supporting["computed_value"]["radius"])
                    and owned["E_R_includes_r_R"] is True
                    and _acceptance_pair(owned["E_R"]) == error
                    and _acceptance_pair(owned["r_R_marginal"]) == 0
                    and _acceptance_pair(owned["r_S_in_U_read"]) == rs
                    and _acceptance_pair(owned["D_center_difference"]) == abs(cs-cr), "Strict_E2_tagwise_radius_ownership")
            total = abs(cs-cr)+rs+error+_acceptance_pair(owned["U_map"])+_acceptance_pair(owned["subtraction_arithmetic_bound"])
            require(_acceptance_pair(measured["conditional_estimate_value"]) == total, "Strict_E2_tagwise_arithmetic")
            if key in {"q", "f"}:
                require(enclosure(responses[index][key + "_C"]) == (cs, rs), "Strict_E2_charge_response_binding")
    return {"status":"accepted", "reason":None, "e2":e2, "accepted_quantities":accepted}


def accepted_strict_E2(reference, *, spec, labels, problem_identity):
    """Invalid or absent v1 authority is a reasoned unknown, never a thrown gate."""
    try:
        return _accepted_strict_E2(reference, spec=spec, labels=labels, problem_identity=problem_identity)
    except (KeyError, TypeError, ValueError, OSError, AttributeError, OverflowError) as error:
        return {"status":"unknown_reference_error", "reason":str(error), "e2":None, "accepted_quantities":[]}


def strict_E2_values(e2, key, *, quantity, unit):
    """The accepted claim uses its exact per-tag values and exclusive slot."""
    row = e2["quantities"][key]
    require(row.get("schema") == "solarlab.g2-reference-evidence.v1"
            and row.get("status") == "computed_pending_review" and row.get("quantity_key") == key
            and row.get("claim_kind") == "conditional_uncertainty_estimate"
            and row.get("rigorous_bound_value") is None and row.get("diagnostic_value") is None
            and row.get("target_problem_identity") == e2["target_problem_identity"]
            and row.get("quantity_and_unit") == {"quantity":quantity,"unit":unit}, "Strict_E2_quantity")
    require(type(row["norm_and_coverage"]["records"]) is int
            and row["norm_and_coverage"] == {"norm":"max_abs_tagged131","records":131,"labels_sha256":digest(e2["labels"])}
            and digest([r["tag"] for r in row["per_record"]]) == digest(e2["labels"]), "Strict_E2_coverage")
    values = [q(r["conditional_estimate_value"]) for r in row["per_record"]]
    require(all(v >= 0 for v in values) and all(r.get("status") == "computed_pending_review" for r in row["per_record"])
            and q(row["conditional_estimate_value"]) == max(values)
            and digest(row["estimated_or_enclosed_value"]) == digest({"value_field":"conditional_estimate_value",
                "value":row["conditional_estimate_value"],"unit":unit}), "Strict_E2_values")
    return values


def reference_status_v3(reference, *, spec, labels, problem_identity):
    """Consume reviewed typed q/f evidence; do not compute BE/E1/E2/E3 here."""
    unknown = {"status":"unknown","reason":"unknown_reference_error","Qsig_C":None,"claim_kind":None}
    if not reference:
        return unknown
    try:
        require(type(reference) is dict, "V3_reference_root")
        if reference.get("schema") == "solarlab.g2-strict-reference.v1":
            acceptance = accepted_strict_E2(reference,spec=spec,labels=labels,problem_identity=problem_identity)
            if acceptance["status"] != "accepted":
                return {**unknown, "detail":acceptance["reason"]}
            e2, accepted = acceptance["e2"], acceptance["accepted_quantities"]
            require({"q","f"} <= set(accepted), "Strict_E2_q_f_not_accepted")
            rows = e2["strict_responses"]
            require(digest([r["tag"] for r in rows]) == digest(labels)
                    and all(enclosure(rows[0][k]) == (0,0) for k in ("q_C","f_C")), "Strict_E2_response_coverage")
            signal = min(max(abs_bounds(*enclosure(row[k]))[0] for row in rows) for k in ("q_C","f_C"))
            eq, ef = (max(strict_E2_values(e2,key,quantity=key,unit="C")) for key in ("q","f"))
            return {"status":"eligible","reason":"independently_accepted_Strict_E2","Qsig_C":pair(signal),
                    "E_q_C":pair(eq),"E_f_C":pair(ef),"claim_kind":"conditional_uncertainty_estimate",
                    "method_contract":e2["method_contract"],"independent_review":reference["independent_review"],
                    "strict_acceptance_sha256":reference["independent_review"]["sha256"],"e2_file":reference["e2_file"],
                    "proof_recomputed_by_this_gate":False,"independent_proof_validity_is_external_prerequisite":True}
        require(reference["case_id"] == "B_N8_Strict_Full" and reference["problem_identity"] == problem_identity,
                "V3_reference_problem")
        require(reference["unit"] == "C" and digest(reference["labels"]) == digest(labels) and len(labels) == 131,
                "V3_reference_coverage")
        rows = reference["responses"]
        require(len(rows) == 131 and all(digest(r["tag"]) == digest(t) for r,t in zip(rows,labels,strict=True)), "V3_reference_tags")
        require(all(enclosure(rows[0][k]) == (0,0) for k in ("q_C","f_C")), "V3_reference_initial_zero")
        signal = min(max(abs_bounds(*enclosure(row[k]))[0] for row in rows) for k in ("q_C","f_C"))
        method = _pin(reference["method_contract"]["path"],reference["method_contract"])
        from scripts.benchmarks.g2_route import _frozen_definition,EVIDENCE_METHOD_SCHEMA
        _frozen_definition(method,EVIDENCE_METHOD_SCHEMA,1)
        review = _pin(reference["independent_review"]["path"],reference["independent_review"])
        source_digest = digest({k:v for k,v in reference.items() if k != "independent_review"})
        require(type(review) is dict and review.get("accepted") is True and review.get("independent_reference_error_review") is True
                and review.get("evidence_sha256") == source_digest and review.get("gate_spec_sha256") == spec.sha256
                and review.get("method_contract_sha256") == reference["method_contract"]["sha256"]
                and type(review.get("reviewer")) is str and review["reviewer"]
                and type(review.get("evidence_author")) is str and review["evidence_author"]
                and review["reviewer"] != review["evidence_author"], "V3_reference_review")
        required = {"status","evidence_class","claim_kind","target_problem_identity","quantity_and_unit",
            "norm_and_coverage","reference_source_and_run_identity","error_components_and_ownership",
            "assumptions_and_checked_premises","observed_refinement_and_method_independence","estimated_or_enclosed_value",
            "unresolved_common_bias_and_model_error","reviewer_and_decision_authority"}
        axes = {"evaluation","initial_and_algebraic","input_clock_parameter","trajectory_nonlinear","stability_reconstruction"}
        values,kinds = {},[]
        for field,quantity in (("E_q","q"),("E_f","f")):
            record = reference["error_evidence"][field]
            require(type(record) is dict and required <= set(record)
                    and record["schema"] == "solarlab.g2-reference-evidence.v1" and record["status"] == "accepted",
                    "V3_reference_record")
            kind = record["claim_kind"]
            slots = {"rigorous_bound":"rigorous_bound_value","conditional_uncertainty_estimate":"conditional_estimate_value"}
            require(kind in slots, "V3_diagnostic_is_not_reference_error")
            slot = slots[kind]
            require(all(record.get(other) is None for other in
                        ({"rigorous_bound_value","conditional_estimate_value","diagnostic_value"}-{slot})), "V3_mixed_claim_slots")
            value = q(record[slot]);require(value >= 0,"V3_reference_nonnegative")
            expected = {"value_field":slot,"value":record[slot],"unit":"C"}
            require(digest(record["estimated_or_enclosed_value"]) == digest(expected)
                    and record["target_problem_identity"] == problem_identity
                    and record["quantity_and_unit"] == {"quantity":quantity,"unit":"C"}
                    and record["norm_and_coverage"] == {"norm":"max_abs_tagged131","records":131,"labels_sha256":digest(labels)},
                    "V3_reference_value_norm")
            identity = record["reference_source_and_run_identity"]
            require(identity["case_id"] == "B_N8_Strict_Full" and all(type(identity.get(k)) is str and identity[k]
                    for k in ("run_id","source_identity","request_sha256")), "V3_reference_execution_identity")
            require(record["assumptions_and_checked_premises"] and record["unresolved_common_bias_and_model_error"]
                    and record["reviewer_and_decision_authority"], "V3_reference_premises_disclosure")
            components = record["error_components_and_ownership"]
            require(type(components) is list and components,"V3_reference_components")
            covered,seen,total = set(),set(),Fraction(0)
            for component in components:
                require(type(component["id"]) is str and component["id"] not in seen
                        and component["claim_kind"] in slots
                        and (kind != "rigorous_bound" or component["claim_kind"] == "rigorous_bound"), "V3_component_claim")
                owned = set(component["covered_axes"])
                require(owned and owned <= axes and not (covered & owned), "V3_component_ownership")
                covered |= owned;seen.add(component["id"])
                amount = q(component["value_C"]);require(amount >= 0,"V3_component_value")
                total += amount
                _pin(component["source"]["path"],component["source"])
            require(covered == axes and value >= total, "V3_reference_remainder_coverage")
            support = record["observed_refinement_and_method_independence"]
            if kind == "conditional_uncertainty_estimate":
                require(set(support) == {"E1","E2","E3"}, "V3_engineering_axes")
                for axis,pin in support.items():
                    proof = _pin(pin["path"],pin)
                    require(proof.get("status") == "accepted" and proof.get("axis") == axis
                            and proof.get("method_contract_sha256") == reference["method_contract"]["sha256"]
                            and proof.get("target_problem_identity") == problem_identity
                            and proof.get("checked_premises"), "V3_engineering_premises")
            else:
                proof = _pin(record["proof"]["path"],record["proof"])
                require(proof.get("kind") in {"analytic_original_problem","validated_IVP_enclosure",
                            "defect_stability_bound","independently_bounded_reference"}
                        and proof.get("target_problem_identity") == problem_identity
                        and proof.get("checked_premises"), "V3_rigorous_proof_support")
            values[field]=value;kinds.append(kind)
        return {"status":"eligible","reason":"independently_reviewed_typed_reference","Qsig_C":pair(signal),
                **{k+"_C":pair(v) for k,v in values.items()},
                "claim_kind":"conditional_uncertainty_estimate" if "conditional_uncertainty_estimate" in kinds else "rigorous_bound",
                "method_contract":reference["method_contract"],"independent_review":reference["independent_review"],
                "proof_recomputed_by_this_gate":False,"E1_E2_E3_recomputed_by_this_gate":False,
                "independent_proof_validity_is_external_prerequisite":True}
    except (KeyError,TypeError,ValueError,OSError):
        return unknown


def device_disposition(*, evaluated, reference, non_signal_complete, gate_failure=None, undecided=False):
    """V3 precedence: missing/rejected E can never become a weak signal."""
    if gate_failure is not None:
        return {"status":"failed","reason":"applicable_certificate_failure","first_failure":gate_failure}
    if not evaluated:
        return {"status":"not_evaluated","reason":"no_required_device_data"}
    if reference["status"] != "eligible":
        return {"status":"unknown_reference_error","reason":"unknown_reference_error"}
    signal = q(reference["Qsig_C"]);error = max(q(reference["E_q_C"]),q(reference["E_f_C"]))
    if signal > 3*error and error > Fraction(1,1000)*Fraction(1,100)*signal/3:
        return {"status":"failed","reason":"reference_share_exceeded","first_failure":{"kind":"reference_share"}}
    if not non_signal_complete:
        return {"status":"unknown","reason":"incomplete_non_signal_evidence"}
    if signal == 0 or signal <= 3*error:
        return {"status":"unresolved_signal","reason":"qualified_reference_but_unresolved_signal",
                "device_scaled_certification":"not_evaluated"}
    if undecided:
        return {"status":"unknown","reason":"unknown_scale_or_certificate"}
    return {"status":"passed","reason":"all_required_device_certificates_passed"}


def lu2_eligibility(result, route_evidence=None):
    """Local eligibility plus separately pinned route review and restriction."""
    device = result["channels"]["device"]["status"]
    eligible_signal = device == "unresolved_signal"
    missing = []
    if not eligible_signal:missing.append("device_status_not_unresolved_signal")
    if result["case_id"] != "B_N8_Main_Full":missing.append("not_G2R_Main_target")
    if any(result["channels"][k]["status"] != "passed" for k in CHANNELS[1:]):missing.append("metal_certificates")
    if not all(w["complete_pieces"] for w in result["fixed_windows"]):missing.append("fixed_window_coverage")
    names = {"safety","T2","T3","history_identity","matched_growth","port_reference_comparisons","state_reference_comparisons"}
    scope = digest({"case_id":result["case_id"],"run_id":result["run_id"],"channels":result["channels"],
                    "reference":result["reference"],"fixed_windows":result["fixed_windows"]})
    try:
        require(type(route_evidence) is dict and set(route_evidence["checks"]) == names
                and all(v == "passed" for v in route_evidence["checks"].values()), "L_U2_other_requirements")
        review = _pin(route_evidence["independent_review"]["path"],route_evidence["independent_review"])
        restriction = _pin(route_evidence["commander_restriction"]["path"],route_evidence["commander_restriction"])
        require(review.get("accepted") is True and review.get("independent_route_review") is True
                and review.get("result_scope_sha256") == scope and review.get("checks_sha256") == digest(route_evidence["checks"]),
                "L_U2_independent_review")
        require(restriction.get("accepted") is True and restriction.get("id") == "L-U2"
                and restriction.get("scope") == "G2-R" and restriction.get("result_scope_sha256") == scope
                and restriction.get("transfer") == "G2-Q" and restriction.get("exit_before") == "G9P"
                and restriction.get("user_record"), "L_U2_commander_restriction")
    except (KeyError,TypeError,ValueError,OSError) as error:
        missing.append(str(error) if isinstance(error,InvalidT1Data) else "route_review_or_restriction_missing")
    return {"id":"L-U2","device_signal_eligible":eligible_signal,"eligible":not missing,"missing":missing,
            "result_scope_sha256":scope,"route_status":"passed_with_user_accepted_limitation" if not missing else "not_evaluated",
            "device_scientific_pass":False,"full_scientific_pass":False,"transfer":"G2-Q","exit_before":"G9P"}


def _clock(row):
    values = []
    for key in ("t_left_hex", "t_right_hex"):
        value = row[key]
        require(type(value) is str and math.isfinite(float.fromhex(value)) and float.fromhex(value).hex() == value, "canonical_time")
        values.append(Fraction(float.fromhex(value)))
    require(values[0] < values[1], "positive_interval")
    return tuple(values)


def _endpoint_difference(left, right):
    a,ra = enclosure(left);b,rb = enclosure(right)
    return {"center":pair(b-a),"absolute_error_bound":pair(ra+rb)}


def _compatible_enclosures(left, right, reason):
    a,ra = enclosure(left);b,rb = enclosure(right)
    require(abs(a-b) <= ra+rb,reason)


def _piece_v3(piece, parent, window, case_id):
    require(piece.get("schema") == "solarlab.g2-window-piece.v1" and piece.get("case_id") == case_id,
            "V3_piece_schema_case")
    _channels_v3(piece)
    require(piece.get("input_digest") == parent.get("input_digest") == parent.get("request_sha256")
            and digest(piece.get("owner_generation")) == digest(parent.get("owner_generation"))
            and piece["path_identity"] == parent["path_identity"]
            and digest(piece["native_clock"]) == digest(parent["native_clock"]), "V3_piece_input_epoch")
    require(piece["event_side"] == (window[1]["event_side"] if _clock(piece)[1] == window[3] else "interior"),
            "V3_piece_event_side")
    require(digest(piece["window_key"]) == digest({"left":window[0],"right":window[1]})
            and piece["piece_id"] == digest({k:v for k,v in piece.items() if k != "piece_id"}), "V3_piece_identity")
    for name in CHANNELS:
        value = piece["channels"][name]
        left,right = value.get("left_charge_enclosure_C"),value.get("right_charge_enclosure_C")
        if left is None or right is None:
            continue  # Explicit missing endpoint evidence becomes unknown coverage.
        delta = _endpoint_difference(left,right)
        _compatible_enclosures(delta,value["delta_Q_enclosure_C"],"V3_piece_endpoint_delta")
        ownership = value.get("coverage_ownership")
        require(type(ownership) is dict and set(ownership) == set(COMPONENTS)|{"Qquad"}
                and all(type(x) is str and x for x in ownership.values()), "V3_piece_uncertainty_ownership")


def fixed_windows(labels):
    require(len(labels) == 131 and len({r["time_hex"] for r in labels}) == 129, "full_131_labels")
    windows = []
    names = ("dark_equilibrium_hold", "voltage_ramp", "slow_state_hold")
    for name, count, first, last in zip(names, (33, 33, 65), (0., .1, .2), (.1, .2, 9.2), strict=True):
        rows = [row for row in labels if row["segment"] == name]
        require(len(rows) == count and rows[0]["time_hex"] == first.hex() and rows[-1]["time_hex"] == last.hex(), "full_protocol_tags")
        for i, row in enumerate(rows):
            require(type(row["index"]) is int and row["index"] == i and row["event_side"] ==
                    ("right" if i == 0 else "left" if i == count-1 else "interior"), "tag_index_or_side")
    for left, right in zip(labels, labels[1:]):
        a, b = Fraction(float.fromhex(left["time_hex"])), Fraction(float.fromhex(right["time_hex"]))
        if left["segment"] == right["segment"]:
            require(right["index"] == left["index"]+1 and a < b, "fixed_window_labels")
            windows.append((left, right, a, b))
        else:
            require(a == b and left["event_side"] == "left" and right["event_side"] == "right", "event_sides")
    require(len(windows) == 128, "fixed_window_count")
    return windows


def evaluate(intervals, pieces, *, labels, spec, case_id, run_id, problem_identity, reference=None,
             source_id=None, request_sha256=None, refresh_times=(), checkpoint=lambda: None,
             route_evidence=None, known_device_failure=None):
    """Evaluate every native prefix and all fixed windows; never stop at fail."""
    require(case_id in {"B_N8_Coarse_Full", "B_N8_Main_Full", "B_N8_Strict_Full", "B_N16_Strict_Full"}, "full_case_scope")
    windows = fixed_windows(labels)
    refreshes = tuple(Fraction(float.fromhex(value)) for value in refresh_times)
    ref = reference_status(reference, spec=spec, labels=labels, problem_identity=problem_identity)
    if case_id == "B_N16_Strict_Full":
        ref = {"status": "unknown", "reason": "unknown_reference_error", "Qsig_C": ref.get("Qsig_C"),
               "scope": "N8 proof cannot qualify a single N16 temporal trajectory"}
    signal = q(ref["Qsig_C"]) if ref["status"] in {"passed","eligible"} else None
    scale_applicable = (signal is not None and (spec.version == 2 or
        signal > 3*max(q(ref["E_q_C"]),q(ref["E_f_C"])) and signal > 0))
    ledgers = {k: ChargeLedger() for k in CHANNELS}
    segments = {name: (a, b) for name, a, b in
                (("dark_equilibrium_hold", Fraction(0), Fraction(.1)),
                 ("voltage_ramp", Fraction(.1), Fraction(.2)), ("slow_state_hold", Fraction(.2), Fraction(9.2)))}
    rows, by_id, prior = [], {}, None
    for i, row in enumerate(intervals):
        checkpoint()
        left, right = _clock(row)
        require(row["run_id"] == run_id and set(row["channels"]) == set(CHANNELS), "interval_scope")
        require(row["segment_id"] in segments, "interval_segment")
        start, end = segments[row["segment_id"]]
        require(start <= left < right <= end and row["event_side"] == ("left" if right == end else "interior"), "interval_event_scope")
        require(all(type(row[k]) is str and row[k] for k in ("native_interval_id", "frame_id", "map_id", "source_id")), "interval_identities")
        require(source_id is None or row["source_id"] == source_id, "source_identity")
        require(request_sha256 is None or row.get("request_sha256") == request_sha256, "request_identity")
        require(row["native_interval_id"] not in by_id and (prior is None or left == prior), "interval_gap_overlap_or_duplicate")
        if spec.version == 3:
            epoch = row.get("owner_generation")
            require(row.get("case_id") == case_id and type(epoch) is dict and set(epoch) == {"owner","generation"}
                    and type(epoch["owner"]) is str and epoch["owner"]
                    and type(epoch["generation"]) is int and epoch["generation"] > 0, "V3_native_epoch_case")
            _channels_v3(row)
        require(left >= 0 and right <= windows[-1][3], "interval_protocol_range")
        by_id[row["native_interval_id"]] = row
        output = {"index": i, "native_interval_id": row["native_interval_id"],
                  "t_left_hex": row["t_left_hex"], "t_right_hex": row["t_right_hex"], "channels": {},
                  **{key:row.get(key) for key in ("segment_id","source_id","frame_id","map_id","source_record_sha256")}}
        for name in CHANNELS:
            floor = spec.Q0 if name != "device" else spec.eta*signal if signal is not None else Fraction(0)
            epsilon = spec.epsilon_metal if name != "device" else spec.epsilon_device
            value = operands(row["channels"][name]); ledgers[name].append(value, floor)
            local = assess_interval(row["channels"][name], floor=floor, epsilon=epsilon, share=spec.quadrature_share)
            prefix = ledgers[name].result(floor=floor, epsilon=epsilon, share=spec.quadrature_share,
                                          cap=signal if name == "device" else None)
            if name == "device" and signal is None:
                local = {"status": "unknown", "reason": ref["reason"]}
                prefix = {"status": "unknown", "reason": ref["reason"]}
            elif name == "device" and not scale_applicable:
                local = {"status":"not_evaluated","reason":"qualified_reference_but_unresolved_signal"}
                prefix = dict(local)
            ledger = ledgers[name]
            legacy = (None if value is None or not ledger.complete else {
                "role": "diagnostic_only", "absolute_limit_C": pair(spec.Q0),
                "interval_bound_C": pair(abs(value["b"])+value["u"]),
                "signed_prefix_bound_C": pair(abs(ledger.b)+ledger.u), "positive_prefix_bound_C": pair(ledger.D)})
            output["channels"][name] = {"interval": local, "prefix": prefix, "legacy_absolute": legacy,
                "signed_prefix_trend_input": None if not ledger.complete else {"b_C":pair(ledger.b),"u_C":pair(ledger.u)}}
        rows.append(output); prior = right
    coverage_complete = bool(intervals) and _clock(intervals[0])[0] == 0 and prior == windows[-1][3]
    grouped = {i: [] for i in range(128)}
    parent_pieces = {key: [] for key in by_id}
    for piece in pieces:
        checkpoint()
        index = piece["window_index"]
        require(type(index) is int and index in grouped, "piece_window_index")
        require(piece["native_interval_id"] in by_id and piece["run_id"] == run_id, "piece_parent")
        parent = by_id[piece["native_interval_id"]]
        require(piece["parent_sha256"] == digest(parent), "piece_parent_identity")
        if spec.version == 3:
            _piece_v3(piece,parent,windows[index],case_id)
        require(all(piece[k] == parent[k] for k in
                    ("segment_id", "frame_id", "map_id", "source_id", "request_sha256")), "piece_source_scope")
        a, b = _clock(piece); lo, hi = _clock(parent)
        require(lo <= a < b <= hi, "piece_native_range")
        require(not any(a < when < b for when in refreshes), "piece_crosses_refresh_boundary")
        grouped[index].append(piece)
        parent_pieces[piece["native_interval_id"]].append(piece)
    for key, children in parent_pieces.items():
        checkpoint()
        parent = by_id[key]
        a, b = _clock(parent)
        ordered = sorted(children, key=lambda row: _clock(row)[0])
        if not ordered or _clock(ordered[0])[0] != a or _clock(ordered[-1])[1] != b:
            continue  # incomplete pieces stay unknown in their fixed windows
        if any(_clock(x)[1] != _clock(y)[0] for x, y in zip(ordered, ordered[1:])):
            continue
        for name in CHANNELS:
            original = operands(parent["channels"][name])
            values = [operands(row["channels"][name]) for row in ordered]
            if original is None or any(v is None for v in values):
                continue
            for center, radius in (("dc", "dr"), ("fc", "fr")):
                require(abs(sum(v[center] for v in values)-original[center]) <=
                        sum(v[radius] for v in values)+original[radius], "piece_parent_integral_consistency")
            lower = sum(v["l1_lower"] for v in values)
            upper = None if any(v["l1_upper"] is None for v in values) else sum(v["l1_upper"] for v in values)
            if original["l1_upper"] is not None and upper is not None:
                require(max(lower, original["l1_lower"]) <= min(upper, original["l1_upper"]), "piece_parent_L1_consistency")
    matched = []
    for index, (left_tag, right_tag, first, last) in enumerate(windows):
        checkpoint()
        selected = grouped[index]
        complete = bool(selected)
        cursor = first
        local_ledgers = {k: ChargeLedger() for k in CHANNELS}
        for piece in selected:
            a, b = _clock(piece)
            require(a >= cursor and b <= last and piece["segment_id"] == left_tag["segment"], "piece_window_cover")
            # A missing first/interior piece makes this window unknown, while
            # overlap, reordering and identity damage still reject the data.
            complete = complete and a == cursor
            cursor = b
            for name in CHANNELS:
                local_ledgers[name].append(operands(piece["channels"][name]), spec.Q0 if name != "device" else Fraction(0))
        complete = complete and cursor == last
        result = {"index": index, "left": left_tag, "right": right_tag, "complete_pieces": complete, "channels": {}}
        window_charges = {}
        if spec.version == 3 and complete:
            for name in CHANNELS:
                data = [piece["channels"][name] for piece in selected]
                endpoints_complete = all(row.get("left_charge_enclosure_C") is not None
                                         and row.get("right_charge_enclosure_C") is not None for row in data)
                if not endpoints_complete:
                    complete = False;continue
                for previous,current in zip(data,data[1:]):
                    _compatible_enclosures(previous["right_charge_enclosure_C"],current["left_charge_enclosure_C"],
                                           "V3_window_endpoint_continuity")
                independent = _endpoint_difference(data[0]["left_charge_enclosure_C"],data[-1]["right_charge_enclosure_C"])
                if local_ledgers[name].complete:
                    _compatible_enclosures(independent,{"center":pair(local_ledgers[name].dc),
                        "absolute_error_bound":pair(local_ledgers[name].dr)},"V3_window_telescoping")
                window_charges[name] = independent
            result["complete_pieces"] = complete
            result["independent_window_charge_enclosures_C"] = window_charges
        for name in CHANNELS[1:]:
            result["channels"][name] = (local_ledgers[name].result(floor=spec.Q0, epsilon=spec.epsilon_metal,
                share=spec.quadrature_share, window=True,window_charge=window_charges.get(name)) if complete else
                {"status": "unknown", "reason": "unknown_missing_window_pieces"})
        result["signed_device_trend_input"] = ({"b_C": pair(local_ledgers["device"].b), "u_C": pair(local_ledgers["device"].u)}
            if complete and local_ledgers["device"].complete else None)
        matched.append(result)
    summaries = {}
    for name in CHANNELS:
        checkpoint()
        events = [(Fraction(float.fromhex(row["t_right_hex"])), {"kind": gate, "index": row["index"],
                   "native_interval_id": row["native_interval_id"]}, row["channels"][name][gate])
                  for row in rows for gate in ("interval", "prefix")]
        if name != "device":
            events += [(Fraction(float.fromhex(row["right"]["time_hex"])), {"kind": "fixed_window", "index": row["index"]},
                       row["channels"][name]) for row in matched]
        events.sort(key=lambda item: item[0])
        failed = next((where for _, where, result in events if result["status"] == "failed"), None)
        unknown = next((dict(where, reason=result["reason"]) for _, where, result in events if result["status"] == "unknown"), None)
        summaries[name] = {"status": "failed" if failed else "unknown" if unknown or not coverage_complete else "passed",
                           "first_failure": failed, "first_unknown": unknown,
                           "complete_native_coverage": coverage_complete,
                           "coverage_reason": None if coverage_complete else "unknown_incomplete_native_coverage"}
    if spec.version == 3:
        non_signal_complete = coverage_complete and ledgers["device"].complete and all(
            row["complete_pieces"] and row["signed_device_trend_input"] is not None for row in matched)
        disposition = device_disposition(evaluated=bool(rows),reference=ref,non_signal_complete=non_signal_complete,
            gate_failure=known_device_failure or summaries["device"]["first_failure"],
            undecided=summaries["device"]["first_unknown"] is not None)
        summaries["device"].update(disposition,claim_kind=ref.get("claim_kind"),
            status_scope="this run's interval/prefix certification; cross-run comparison is separately required")
    result = {"schema": "solarlab.t1-candidate-result.v1", "candidate": True, "gate_spec_sha256": spec.sha256,
            "case_id": case_id, "run_id": run_id, "reference": ref, "channels": summaries,
            "native": rows, "fixed_windows": matched, "scientific_qualification": False,
            "execution_authorized": False, "definition_frozen_by_this_code": False,
            "gate_spec_version":spec.version,"cross_run_comparison":"not_evaluated_here",
            "scope": "Candidate certificate arithmetic on supplied source-bound enclosures; external proof review and all other gates remain required."}
    if spec.version == 3:
        result["L_U2"] = lu2_eligibility(result,route_evidence)
    return result


def from_sealed_charge(row, *, run_id, scale_evidence=None):
    """Adapter for the actual SealedChargeObserver output; no guessed fields."""
    from scripts.benchmarks.native_readback import _decode
    require(row.get("kind") == "sealed_voltage_lift_charge", "sealed_charge_kind")
    interval = row["interval"]
    observation = _decode(interval["observation"])
    p = observation["physical_charge"]
    require(p["schema"] == "solarlab.native-physical-charge.v1"
            and p["physical_flux_basis"] == "fixed-uniform-epsilon-Dirichlet-slab-reservoir-carrier-ion-trap-capacitance-v1",
            "independent_physical_flux_basis")
    require(p["request_sha256"] == row["request_sha256"]
            and p["frame_identity"] == interval["coefficient_frame_identity"]
            and p["path_identity"] == observation["path_identity"]
            and digest(p["clock"]) == digest(observation["clock"]), "sealed_charge_scope_binding")
    require(q(p["clock"]["predecessor"]) == F_time(interval["left"]["time_s"])
            and q(p["clock"]["tn"]) == F_time(interval["right"]["time_s"]), "sealed_charge_endpoint_clock")
    def enc(value):
        return {"center": pair(value["center"]), "absolute_error_bound": pair(value["absolute_error_bound"])}
    # _decode leaves enclosure payload pairs intact, and decodes the standalone
    # component Fractions.  These fields are emitted at native_observation.py's
    # physical_charge block, rather than its historical raw/tangent L1 ledger.
    result = {"run_id": run_id, "native_interval_id": row["source_interval_sha256"],
        "request_sha256": row["request_sha256"],
        "source_record_sha256": digest(row), "segment_id": interval["segment_id"],
        "event_side": "left" if interval["right"]["time_s"] == float(q(p["clock"]["segment_end"])) else "interior",
        "t_left_hex": float(interval["left"]["time_s"]).hex(), "t_right_hex": float(interval["right"]["time_s"]).hex(),
        "frame_id": p["frame_identity"], "map_id": p["map_identity"], "source_id": p["source_identity"], "channels": {}}
    if scale_evidence is not None:
        require(scale_evidence["source_record_sha256"] == digest(row), "scale_source_binding")
        partition = scale_evidence["partition"]
        require(type(partition["depth"]) is int and 0 <= partition["depth"] <= 8
                and type(partition["leaves"]) is int and 1 <= partition["leaves"] <= 256,
                "scale_partition_limit")
    for i, name in enumerate(CHANNELS):
        scale = scale_evidence["channels"][name] if scale_evidence is not None else {}
        components = p["error_components_C"]
        result["channels"][name] = {"delta_Q_enclosure_C": enc(p["saved_finite_charge_change_C"][i]),
            "integral_F_enclosure_C": enc(p["physical_flux_integral_C"][i]),
            "integral_abs_F_lower_C": scale.get("lower_C"), "integral_abs_F_upper_C": scale.get("upper_C"),
            "uncertainty_components_C": {new: pair(components[old][i]) for new, old in zip(COMPONENTS,
                ("saved_charge_evaluation", "physical_flux_integration", "input", "clock_strip"), strict=True)},
            "quadrature_component_upper_C": pair(components["physical_flux_integration"][i])}
    return result


def from_sealed_t1(row, *, run_id):
    """The V3 supplement cannot replace the independently checked C operands."""
    base = {k:v for k,v in row.items() if k != "t1_evidence"}
    original = from_sealed_charge(base,run_id=run_id)
    result = row["t1_evidence"]
    require(digest(result["native_clock"]) == digest(base["interval"]["observation"]["physical_charge"]["clock"]),
            "sealed_T1_native_clock")
    for key,value in original.items():
        if key != "channels":
            require(digest(result.get(key)) == digest(value),"sealed_T1_source_identity")
    for name in CHANNELS:
        for key in ("delta_Q_enclosure_C","integral_F_enclosure_C","uncertainty_components_C","quadrature_component_upper_C"):
            require(digest(result["channels"][name].get(key)) == digest(original["channels"][name][key]),
                    "sealed_T1_original_operand_changed")
    return result


def F_time(value):
    require(type(value) in (int, float) and math.isfinite(value), "finite_clock_value")
    return Fraction(value)


def read_sealed_evidence(manifest_path, *, max_record_bytes, max_logical_bytes, max_records, wall_s,
                         max_output_bytes=64*1024**2, rss_bytes=4*1024**3, require_admission=False):
    """One admitted bounded pass, reusable by comparison without another decode."""
    from scripts.benchmarks.native_history import HistoryReader, integer_json_dumps
    from scripts.benchmarks.native_readback import _json
    from scripts.benchmarks.g2_route_bindings import sample_labels
    require(all(type(v) is int and v > 0 for v in (max_record_bytes, max_logical_bytes, max_records)), "reader_limits")
    require(type(wall_s) in (int, float) and math.isfinite(wall_s) and 0 < wall_s <= 43200, "reader_wall_limit")
    require(type(max_output_bytes) is int and 0 < max_output_bytes <= 4*1024**3
            and type(rss_bytes) is int and 0 < rss_bytes <= 4*1024**3, "candidate_resource_limits")
    deadline = time.perf_counter()+wall_s
    def checkpoint():
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        require(time.perf_counter() < deadline and rss <= rss_bytes, "candidate_time_or_RSS_limit")
    checkpoint()
    path = Path(manifest_path)
    require(path.stat().st_size <= 8*1024**2, "manifest_size")
    manifest = json.loads(path.read_bytes())
    require(manifest["schema"] == EVIDENCE_SCHEMA, "evidence_schema")
    spec_doc = _pin(manifest["gate_spec"]["path"], manifest["gate_spec"])
    spec = CandidateSpec.from_document(spec_doc, manifest["gate_spec"]["sha256"])
    request = _pin(manifest["request"]["path"], manifest["request"])
    require(manifest["problem_identity"] == physical_problem_identity(request), "physical_problem_identity")
    if spec.version == 3 or require_admission:
        plan_ref = manifest.get("run_plan")
        require(type(plan_ref) is dict,"V3_evaluation_admission_required")
        _pin(plan_ref["path"],plan_ref)
        from scripts.benchmarks.g2_route import validate
        admission = validate(plan_ref["path"],executable=True,case_id=manifest["case_id"],purpose="evaluate")
        case = next(c for c in admission["plan"]["cases"] if c["id"] == manifest["case_id"])
        require(admission["plan"]["definitions"]["gate_spec"]["sha256"] == manifest["gate_spec"]["sha256"]
                and case["execution"]["effective_input_digest"] == digest(request)
                and case["execution"]["native_admission"]["operation_id"] == manifest["run_id"], "evaluation_manifest_binding")
    labels = sample_labels(request)
    expected = manifest["history"]
    require(expected.get("closed") is True and expected.get("container_complete") is True
            and expected["encoding"] == "gzip", "sealed_complete_gzip_required")
    require(all(type(expected[k]) is int and expected[k] >= 0 for k in ("encoded_bytes", "logical_bytes", "records")), "history_count_types")
    require(Path(expected["path"]).stat().st_size == expected["encoded_bytes"], "encoded_size")
    scales = manifest.get("scales", {})
    intervals, pieces, logical = [], [], hashlib.sha256()
    comparison_context, comparison_samples = None, []
    retained = 0
    with HistoryReader(Path(expected["path"]), encoding="gzip", max_record_bytes=max_record_bytes,
                       max_logical_bytes=max_logical_bytes, integer_io_policy=expected.get("integer_io_policy")) as reader:
        for line in reader:
            checkpoint()
            require(reader.complete_lines <= max_records, "reader_record_limit")
            require(line.endswith(b"\n"), "partial_record")
            logical.update(line)
            row = _json(line, reader.integer_io_policy)
            if row.get("kind") == "sealed_voltage_lift_charge":
                normalized = (from_sealed_t1(row,run_id=manifest["run_id"]) if "t1_evidence" in row else
                              from_sealed_charge(row, run_id=manifest["run_id"], scale_evidence=scales.get(digest(row))))
                intervals.append(normalized)
            elif row.get("kind") == "t1_interval_evidence":
                normalized = row["evidence"]
                intervals.append(normalized)
            elif row.get("kind") == "t1_fixed_window_piece":
                normalized = row["evidence"]
                pieces.append(normalized)
            elif row.get("kind") in {"sealed_g2_context","sealed_g2_sample"}:
                require(row["request_sha256"] == digest(request), "comparison_observation_input")
                normalized = row["evidence"]
                if row["kind"] == "sealed_g2_context":
                    require(comparison_context is None and normalized["case_id"] == manifest["case_id"]
                            and normalized["run_id"] == manifest["run_id"]
                            and normalized["request_sha256"] == digest(request), "comparison_context_identity")
                    comparison_context = normalized
                else:
                    require(comparison_context is not None and len(comparison_samples)<131, "comparison_sample_count")
                    comparison_samples.append(normalized)
            else:
                raise InvalidT1Data("unsupported_T1_evidence_record")
            retained += len(integer_json_dumps(normalized, INTEGER_POLICY).encode())
            require(retained <= max_output_bytes, "candidate_retained_evidence_limit")
        require(reader.eof_seen and reader.container_complete is True and not reader.partial_tail
                and reader.logical_bytes == expected["logical_bytes"] and reader.complete_lines == expected["records"]
                and logical.hexdigest() == expected["logical_sha256"], "complete_history_identity")
        history = {"records": reader.complete_lines, "logical_bytes": reader.logical_bytes,
                   "logical_sha256": logical.hexdigest(), "eof_seen": True, "crc_verified": True}
    checkpoint()
    return {"manifest":manifest,"spec":spec,"request":request,"labels":labels,
            "intervals":intervals,"pieces":pieces,"history":history,"checkpoint":checkpoint,
            "comparison_context":comparison_context,"comparison_samples":comparison_samples,"retained_bytes":retained}


def evaluate_sealed(manifest_path, *, max_record_bytes, max_logical_bytes, max_records, wall_s,
                    max_output_bytes=64*1024**2, rss_bytes=4*1024**3, require_admission=False):
    from scripts.benchmarks.native_history import integer_json_dumps
    data = read_sealed_evidence(manifest_path,max_record_bytes=max_record_bytes,
        max_logical_bytes=max_logical_bytes,max_records=max_records,wall_s=wall_s,
        max_output_bytes=max_output_bytes,rss_bytes=rss_bytes,require_admission=require_admission)
    manifest,spec,request,labels,intervals,pieces,history,checkpoint = (
        data[k] for k in ("manifest","spec","request","labels","intervals","pieces","history","checkpoint"))
    result = evaluate(intervals, pieces, labels=labels, spec=spec, case_id=manifest["case_id"],
                      run_id=manifest["run_id"], problem_identity=manifest["problem_identity"], reference=manifest.get("reference"),
                      source_id=manifest["source_id"], request_sha256=digest(request),
                      refresh_times=manifest.get("actual_refresh_times_hex", ()), checkpoint=checkpoint,
                      route_evidence=manifest.get("route_evidence"),known_device_failure=manifest.get("known_device_failure"))
    result["history"] = history
    result["integer_io_policy"] = INTEGER_POLICY
    # Match the actual CLI rendering, including its final newline.
    require(len(integer_json_dumps(result, INTEGER_POLICY, indent=2, allow_nan=False).encode())+1
            <= max_output_bytes, "candidate_output_limit")
    checkpoint()
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--max-record-bytes", type=int, required=True)
    parser.add_argument("--max-logical-bytes", type=int, required=True)
    parser.add_argument("--max-records", type=int, required=True)
    parser.add_argument("--wall-s", type=float, required=True)
    parser.add_argument("--max-output-bytes", type=int, default=64*1024**2)
    parser.add_argument("--rss-bytes", type=int, default=4*1024**3)
    args = parser.parse_args(argv)
    try:
        result = evaluate_sealed(args.manifest, max_record_bytes=args.max_record_bytes,
            max_logical_bytes=args.max_logical_bytes, max_records=args.max_records, wall_s=args.wall_s,
            max_output_bytes=args.max_output_bytes, rss_bytes=args.rss_bytes,require_admission=True)
    except (ValueError, TypeError, KeyError, OSError, EOFError, OverflowError) as error:
        result = {"candidate": True, "status": "invalid_data", "reason": str(error)[:300],
                  "scientific_qualification": False, "execution_authorized": False}
        print(json.dumps(result, indent=2)); return 2
    from scripts.benchmarks.native_history import integer_json_dumps
    print(integer_json_dumps(result, INTEGER_POLICY, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
