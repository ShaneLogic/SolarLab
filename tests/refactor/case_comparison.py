"""Four prospective P00 adapters over the existing supplied-data comparators.

Each context contains named (reference, candidate, gate) triples. Gates are
materialized and reviewed before a candidate run; this module creates no gates,
reference errors, physical data or qualification. Extra observations carry
explicit uncertainty and source-branch records, never default zero errors.
"""
from decimal import Decimal, localcontext
from fractions import Fraction as F
import math

from physical_comparison import compare_physical, definition_fingerprint
from reference_comparison import Rejected, _number, _require, compare

SCOPES = {
    "twod": "scope-benchmark-d84b2405628c3df6",
    "lockin": "scope-oracle-23a290eced9bdb76",
    "el": "scope-experiment-ddece0c2ae0c1f35",
    "eqe": "scope-experiment-d99a1dd374330649",
}
LOWER = F(float.fromhex("0x1.4484bfeebc2a0p-100"))


def number(value, name="number"):
    return F(_number(value, name))


def validate_case_definition(d):
    _require(d.get("schema") == "solarlab.four_case_definition.v1"
             and d.get("kind") in SCOPES and d.get("scope_id") == SCOPES[d["kind"]],
             "case_definition", "One exact original case is required")
    _require(d.get("status") == "prospective_complete" and bool(d.get("case_id")),
             "case_definition", "Numeric completeness is separate from eligibility")
    _require(d.get("reference_error_bound") is None and d.get("reference_qualified") is False
             and d.get("active_comparison") is None and bool(d.get("pending_observables")),
             "case_eligibility", "Prospective definitions cannot qualify data")
    contexts = d.get("contexts", [])
    _require(bool(contexts) and len({c["id"] for c in contexts}) == len(contexts),
             "case_coverage", "Unique original contexts are required")
    for rule in d["rules"].values():
        _require(bool(rule.get("unit")) and number(rule["atol"]) >= 0
                 and number(rule["rtol"]) >= 0, "case_threshold", "Finite dimensional allocations required")
    if d["kind"] == "twod":
        _require({(c["vertical"], c["horizontal"]) for c in contexts}
                 == {(16, 2), (32, 2), (64, 2), (32, 4), (32, 8)},
                 "case_coverage", "Retain both original independent mesh axes")
    elif d["kind"] == "lockin":
        expected = {(f, 80, kind) for f in (.001, .01, 1.)
                    for kind in ("frequency_domain", "time_resolution")}
        _require({(c["frequency"], c["samples"], c["phase_kind"]) for c in contexts} == expected,
                 "case_coverage", "All original final-level frequency and 40-to-80 comparisons")
        _require(d["protocol"]["points_per_cycle"] == [20, 40, 80],
                 "case_coverage", "The original 20/40/80 protocol remains mandatory")
        _require(d["phase_limits"] == {"frequency_domain": .00017453292519943296,
                                        "time_resolution": 1.7453292519943296e-5},
                 "phase_limit", "Preserve the original strict phase words")
    else:
        _require([c["id"] for c in contexts] == ["fixture"], "case_coverage", "One original optical fixture")
        _require(d["wavelengths_nm"] == ([400, 500, 600, 700, 800, 900]
                                         if d["kind"] == "el" else [400, 500, 600, 700, 800]),
                 "case_coverage", "No optical channel may disappear")
    return True


def _guard(d, context, name, triple):
    _require(isinstance(triple, (list, tuple)) and len(triple) == 3,
             "case_packet", "Use existing reference/candidate/gate triples")
    a, b, g = triple
    for record in (a, b, g):
        _require(record.get("case_id") == d["case_id"] and record.get("case_context") == context["id"]
                 and record.get("case_definition_sha256") == definition_fingerprint(d),
                 "case_binding", name)
    _require(g.get("scope") == d["scope_id"] and g.get("reference_qualified") is True,
             "reference_unqualified", "Actual independently qualified data are required")
    _require(g.get("quantity") == name, "case_binding", "Channel names cannot be exchanged")
    _require(a.get("definition_sha256") == b.get("definition_sha256")
             == g.get("definition_sha256") == definition_fingerprint(g),
             "definition_mismatch", "Bind the complete local gate before the candidate")
    return a, b, g


def _value(table):
    points = table.get("points", [])
    _require(len(points) == 1 and points[0].get("status") == "ok",
             "unresolved_data", "One explicit available scalar is required")
    return number(points[0]["value"])


def _error(table, unit):
    e = table.get("uncertainty")
    _require(isinstance(e, dict) and e.get("bound") is not None and e.get("unit") == unit
             and isinstance(e.get("evidence_sha256"), str) and len(e["evidence_sha256"]) == 64
             and all(c in "0123456789abcdef" for c in e["evidence_sha256"]),
             "unresolved_error", "A bound with its actual evidence and units is required")
    result = number(e["bound"])
    _require(result >= 0, "unresolved_error", "Negative uncertainty")
    return result


def _measure(checks, name, passed, **details):
    checks[name] = {"accepted": bool(passed), **details}


def _scalar(d, context, name, triple, rule, checks):
    a, b, g = _guard(d, context, name, triple)
    _require(g.get("kind") == "scalar" and g.get("rule") == "sum"
             and g.get("scale") == "abs_reference" and g.get("unit") == rule["unit"]
             and g.get("atol") == rule["atol"] and g.get("rtol") == rule["rtol"],
             "case_threshold", name)
    result = compare(a, b, g); checks[name] = result
    if result["status"] == "rejected":
        raise Rejected(result["code"], result["detail"])
    x, y = _value(a), _value(b)
    bound = number(g["reference_error_abs"])
    _require(_error(a, rule["unit"]) == bound
             and a["uncertainty"]["evidence_sha256"] == g["reference_error_evidence"],
             "error_binding", name)
    limit = number(rule["atol"]) + number(rule["rtol"])*abs(x)
    _measure(checks, name+":total_error", 3*bound <= limit and abs(y-x)+bound <= limit)
    return x, y


def _physical(d, context, name, triple, rules, checks, scales=None):
    a, b, g = _guard(d, context, name, triple)
    _require(set(g.get("thresholds", {})) == set(rules), "case_metric_coverage", name)
    for metric, rule in rules.items():
        actual = g["thresholds"][metric]
        _require(all(actual.get(k) == rule[k] for k in ("atol", "rtol", "unit"))
                 and actual.get("scale") == (1 if scales is None else scales[metric]),
                 "case_threshold", name+":"+metric)
    result = compare_physical(a, b, g); checks[name] = result
    if result["status"] in ("rejected", "unresolved"):
        raise Rejected(result.get("code", "unresolved_data"), result.get("detail", name))
    return a, b, g


def _coverage(actual, expected, label):
    _require(set(actual) == set(expected), "case_coverage", label)


def _twod(d, context, block, checks):
    channels = block["channels"]; components = d["components"]
    grids = block.get("context_grid")
    _require(isinstance(grids, dict) and set(grids) == {"reference", "candidate"},
             "geometry", "Actual frozen source grids for both sides are required")
    nx, ny = context["horizontal"]+1, 2*context["vertical"]+1
    boundary = number(d["protocol"]["interface_y_m"])
    width = number(d["protocol"]["width_m"])
    interface_rows = {}
    for source_side, grid in grids.items():
        _require(isinstance(grid, dict) and grid.get("shape") == [ny, nx]
                 and grid.get("node_order") == "carrier_local:j*Nx+i"
                 and len(grid["x_m"]) == nx and len(grid["y_m"]) == ny,
                 "geometry", "The grid must belong to this exact original mesh context")
        x, y = [number(v) for v in grid["x_m"]], [number(v) for v in grid["y_m"]]
        _require(x[0] == y[0] == 0 and x[-1] == width and y[-1] == 2*boundary
                 and all(a < b for a, b in zip(x, x[1:])) and all(a < b for a, b in zip(y, y[1:]))
                 and y[context["vertical"]] == boundary,
                 "geometry", "Retain both layers and the source interface node")
        # mol.build_material_arrays uses argmin(abs(y-interface)); its TE face
        # is idx-1. solver_2d keeps that y-face row and handles columns separately.
        row = min(range(ny), key=lambda j: abs(y[j]-boundary))
        _require(grid.get("interface_y_faces") == [row-1]
                 and type(grid["interface_y_faces"][0]) is int, "interface", "Original TE y-face mapping")
        interface_rows[source_side] = row
    for triple in channels.values():
        _require(isinstance(triple, (list, tuple)) and len(triple) == 3
                 and all(isinstance(record, dict) and record.get("context_grid") == grids for record in triple),
                 "geometry", "Each frozen channel gate and observation must bind the same actual grids")
    traces = [f"trace:{carrier}:{side}:{i}" for carrier in ("n", "p")
              for side in ("left", "right") for i in range(context["horizontal"]+1)]
    _coverage(channels, [f"{kind}:{c}" for c in components for kind in ("cells", "density")]+traces,
              "Every component and lateral interface side is mandatory")
    for component in components:
        carrier = component.split(":")[0]; name = "cells:"+component
        triple = channels[name]; a, b, g = triple
        _require(g.get("comparison_target", {}).get("kind") == "fixed_discrete",
                 "comparison_target", "This migration gate does not certify continuum refinement")
        _require(g.get("geometry_dimension") == 2 and g.get("area_unit") == "m2"
                 and g.get("unit_depth_m") == 1 and g.get("volume_unit") == "m3",
                 "geometry", "XY areas and explicit unit depth are required")
        _require(set(g["reference_components"]) == set(g["candidate_components"]) == {component},
                 "component", "No cross-material/component averaging")
        for table, key in ((a, "reference_volumes"), (b, "candidate_volumes")):
            _require(table.get("areas_m2") == g[key], "geometry", "Unit depth1m must be explicit in data")
        _physical(d, context, name, triple,
                  {"l1": d["rules"][carrier+"_l1"], "inventory:"+component: d["rules"][carrier+"_inventory"]}, checks)
        # The log comparison uses exactly the common overlap subcells, not interpolation.
        pairs = [(i, j, v) for i, row in enumerate(g["overlap"]) for j, v in enumerate(row) if v > 0]
        la, lb, lg = channels["density:"+component]
        _require(la["values"] == [a["values"][i] for i, _, _ in pairs]
                 and lb["values"] == [b["values"][j] for _, j, _ in pairs]
                 and lg["volumes"] == [v for _, _, v in pairs]
                 and set(lg["components"]) == {component} and lg.get("zero_cells") == [],
                 "projection", "Positive-carrier log support must equal the verified overlap")
        _physical(d, context, "density:"+component, channels["density:"+component],
                  {"log_rms": d["rules"]["log"], "log_max": d["rules"]["log"],
                   "zero_density_max": d["rules"]["zero"],
                   "inventory:"+component: d["rules"][carrier+"_inventory"]}, checks)
    for name in traces:
        _, carrier, side, index = name.split(":")
        a, b, g = channels[name]
        support = g.get("interface_support", {})
        _require(support.get("side") == side and support.get("carrier") == carrier
                 and support.get("lateral_index") == int(index)
                 and support.get("material_component") == carrier+":layer_chi"+("4.0" if side == "left" else "3.8")
                 and support.get("representation") in ("source_stencil_side", "bounded_one_sided_trace")
                 and support.get("geometry_verified") is True
                 and len(support.get("evidence_sha256", "")) == 64
                 and a.get("interface_support") == b.get("interface_support") == support,
                 "interface", "Every one-sided source support must be explicitly bound")
        for source_side in ("reference", "candidate"):
            location = support[source_side]
            grid = grids[source_side]
            right_row = interface_rows[source_side]
            row = right_row-1 if side == "left" else right_row
            column = int(index)
            _require(type(location.get("state_index")) is int
                     and location["state_index"] == row*nx+column
                     and type(location.get("electrical_face")) is int
                     and location["electrical_face"] == right_row-1
                     and location["position_m"] == [grid["x_m"][column], grid["y_m"][row]],
                     "interface", "Trace node, coordinate and y-face must match the actual context/source side")
        _scalar(d, context, name, channels[name], d["rules"][name.split(":")[1]+"_trace"], checks)
    _coverage(block["zero_ions"], ("reference", "candidate"), "Both zero-ion states are required")
    for zero in block["zero_ions"].values():
        _coverage(zero, ("input_population", "input_background", "state_population"), "Input and observed zeros")
        _require(all(number(x) == 0 for x in zero.values()),
                 "zero_species", "Absent ions/background cannot be inferred from a missing value")


def strict_phase(angle, left_error, right_error, theta):
    return (left_error >= 0 and right_error >= 0 and left_error <= theta/6
            and right_error <= theta/6 and abs(angle)+left_error+right_error < theta)


def _magnitude_lower(pair):
    """Materialize a lower binary64 scale; do not enlarge a mixed allowance."""
    x, y = map(number, pair)
    value = math.hypot(float(x), float(y))
    _require(math.isfinite(value), "invalid_number", "Unrepresentable complex scale")
    while F(value)**2 > x*x+y*y:
        value = math.nextafter(value, 0.0)
    return value


def _lockin(d, context, block, checks):
    channels = block["channels"]; _coverage(channels, ("I", "Z"), "Independent I and Z gates are mandatory")
    _require(block.get("resolution_levels") == [20, 40, 80],
             "case_coverage", "Retain every original extraction level")
    theta = number(d["phase_limits"][context["phase_kind"]]); triples = {}
    for name in ("I", "Z"):
        triple = channels[name]; a, b, g = triple; rule = d["rules"][name]
        _require(g.get("kind") == "complex" and g.get("unit") == rule["unit"]
                 and g.get("zero_indices") == [] and g.get("coordinates") == [context["frequency"]]
                 and g.get("phase_floor") == d["floors"][name],
                 "phase_floor", "Per-frequency declared floor and exact coordinate")
        x = a["values"][0]
        _physical(d, context, name, triple,
                  {"real": rule, "imag": rule, "difference": rule,
                   "phase": {"atol": float(theta), "rtol": 0, "unit": "rad"}}, checks,
                  {"real": abs(x[0]), "imag": abs(x[1]), "difference": _magnitude_lower(x), "phase": 1})
        errors = []
        for side, table in enumerate((a, b)):
            radius = _error(table, rule["unit"]); magnitude = number(_magnitude_lower(table["values"][0]))
            if side == 0:
                _require(radius == number(g["reference_errors"]["difference"]["bound"]),
                         "error_binding", "One physical complex-error denominator")
            phase = _error(table["phase_error"], "rad")
            rounding = _error(table["phase_rounding"], "rad")
            _require(magnitude > number(d["floors"][name]) and magnitude > radius
                     and phase >= radius/(magnitude-radius)+rounding and phase <= theta/6,
                     "unresolved_phase", "Unresolved amplitude/phase error, including evaluation rounding")
            if side == 0:
                _require(phase == number(g["reference_errors"]["phase"]["bound"]),
                         "error_binding", "The strict phase gate uses the same reference error")
            errors.append(phase)
        angle = number(checks[name]["metrics"]["phase"]["value"])
        _measure(checks, name+":strict_phase", strict_phase(angle, *errors, theta))
        triples[name] = triple
    for side in (0, 1):
        current, impedance = triples["I"][side], triples["Z"][side]
        i = number(_magnitude_lower(current["values"][0])); bi = _error(current, "A m-2")
        bv = _error(block["voltage_error"][side], "V")
        rounding = _error(block["division_error"][side], "ohm m2")
        required = (i*bv+number(d["delta_V"])*bi)/(i*(i-bi))+rounding
        _require(_error(impedance, "ohm m2") >= required,
                 "unresolved_error", "Z uncertainty must include voltage/current division error")
        re, im = map(number, current["values"][0]); zr, zi = map(number, impedance["values"][0])
        voltage = number(d["delta_V"]); square = re*re+im*im
        _measure(checks, "Z_division:"+str(side),
                 (zr-voltage*re/square)**2+(zi+voltage*im/square)**2 <= rounding**2)


def _eqe(d, context, block, checks):
    channels = block["channels"]; wavelengths = d["wavelengths_nm"]
    _coverage(channels, [f"{k}:{w}" for w in wavelengths for k in ("EQE", "photo_current")], "All signed EQE wavelengths")
    scale = F(d["current_scale_exact"])
    for w in wavelengths:
        eqe = _scalar(d, context, "EQE:"+str(w), channels["EQE:"+str(w)], d["rules"]["EQE"], checks)
        current = _scalar(d, context, "photo_current:"+str(w), channels["photo_current:"+str(w)], d["rules"]["current"], checks)
        for side in (0, 1):
            raw = block["subtraction"][side]; light = raw["light"][str(w)]; dark = raw["dark"]
            bl, bd = _error(light, "A m-2"), _error(dark, "A m-2")
            rounding = _error(raw["subtraction_error"][str(w)], "A m-2")
            division = _error(raw["division_error"][str(w)], "1")
            exact = number(light["value"])-number(dark["value"])
            _measure(checks, f"signed_subtraction:{w}:{side}", abs(current[side]-exact) <= rounding)
            _measure(checks, f"EQE_units:{w}:{side}", abs(eqe[side]-current[side]/scale) <= division)
            total = bl+bd+rounding
            ec = _error(channels["photo_current:"+str(w)][side], "A m-2")
            ee = _error(channels["EQE:"+str(w)][side], "1")
            _require(ec >= total and ee >= total/scale+division,
                     "unresolved_error", "Both signed light/dark errors must reach EQE")
            if side == 0:
                _require(ec == number(channels["photo_current:"+str(w)][2]["reference_error_abs"])
                         and ee == number(channels["EQE:"+str(w)][2]["reference_error_abs"]),
                         "error_binding", "Subtraction and gate reference errors differ")
            checks[f"sign_resolution:{w}:{side}"] = {"accepted": True,
                "resolved": abs(current[side]) > total, "signed_value": float(current[side])}


def _log_interval(value):
    if value == 1:
        return F(0), F(0)
    with localcontext() as ctx:
        ctx.prec = 80
        x = Decimal(value.numerator)/Decimal(value.denominator)
        return F(x.next_minus().ln().next_minus()), F(x.next_plus().ln().next_plus())


def el_log_domain(emission, signed, emission_error, injection_error, ratio_rounding):
    """Complete raw ratio interval; the source floor is a branch, not an error bound."""
    magnitude = abs(signed)
    if signed >= 0 or magnitude-injection_error < LOWER or emission-emission_error <= 0:
        return None
    low = (emission-emission_error)/(magnitude+injection_error)-ratio_rounding
    high = (emission+emission_error)/(magnitude-injection_error)+ratio_rounding
    return (low, high) if LOWER <= low <= high <= 1 else None


def _el(d, context, block, checks):
    channels = block["channels"]; wavelengths = d["wavelengths_nm"]
    names = [f"{k}:{w}" for w in wavelengths for k in ("spectrum", "absorptance")]
    names += ["J_em_rad", "J_inj", "EQE_EL", "delta_V_nr"]
    _coverage(channels, names, "Six spectra plus independent currents/ratio/log gates")
    values = {}
    for name in names:
        if name in ("EQE_EL", "delta_V_nr"):
            a, b, _ = _guard(d, context, name, channels[name])
            values[name] = _value(a), _value(b)
            continue
        key = name if name.startswith("spectrum:") else name.split(":")[0]
        values[name] = _scalar(d, context, name, channels[name], d["rules"][key], checks)
    q, vt = F(d["charge_exact_C"]), F(d["thermal_voltage_exact_V"])
    unresolved = []
    for side in (0, 1):
        raw = block["raw"][side]
        _coverage(raw["spectrum_per_m"], map(str, wavelengths), "All per-m source channels")
        _coverage(raw["raw_absorptance"], map(str, wavelengths), "All unclipped absorptance channels")
        integral = F(0); required_e = _error(raw["integration_error"], "A m-2")
        for w, weight in zip(wavelengths, d["weights_nm"]):
            value = values["spectrum:"+str(w)][side]
            _require(value >= 0, "spectrum_domain", "Negative emitted photons")
            conversion = _error(raw["conversion_error"][str(w)], "photons m-2 s-1 nm-1")
            _measure(checks, f"nm_m:{w}:{side}", abs(value-number(raw["spectrum_per_m"][str(w)])/10**9) <= conversion)
            ar = number(raw["raw_absorptance"][str(w)])
            _measure(checks, f"absorptance_clip:{w}:{side}", values["absorptance:"+str(w)][side] == min(max(ar, F(0)), F(1)))
            integral += q*weight*value
            required_e += q*weight*(_error(channels["spectrum:"+str(w)][side], "photons m-2 s-1 nm-1")+conversion)
        emission, signed = values["J_em_rad"][side], values["J_inj"][side]
        _require(emission >= 0, "spectrum_domain", "Negative emitted current")
        be, bi = _error(channels["J_em_rad"][side], "A m-2"), _error(channels["J_inj"][side], "A m-2")
        _measure(checks, f"emission_integral:{side}", abs(emission-integral) <= _error(raw["integration_error"], "A m-2"))
        _require(be >= required_e, "unresolved_error", "Independent emission error includes conversion/integration rounding")
        ratio = values["EQE_EL"][side]; magnitude = abs(signed)
        sentinel = magnitude < LOWER
        lower_clip, upper_clip = not sentinel and ratio < LOWER, not sentinel and ratio > 1
        _require(all(type(raw[k]) is bool for k in ("sentinel", "lower_clip", "upper_clip"))
                 and (raw["sentinel"], raw["lower_clip"], raw["upper_clip"]) == (sentinel, lower_clip, upper_clip),
                 "source_branch", "Actual raw ratio/clamp/sentinel flags must agree")
        ratio_round = _error(raw["ratio_error"], "1"); log_round = _error(raw["log_error"], "V")
        conversion_v = _error(raw["loss_conversion_error"], "V")
        loss = values["delta_V_nr"][side]
        _measure(checks, f"loss_mV_V:{side}", abs(loss-number(raw["loss_mV"])/1000) <= conversion_v)
        if sentinel:
            _require(ratio == 0 and loss == 0 and raw["clamped_ratio"] is None,
                     "source_branch", "Preserve the actual zero sentinel outputs")
        else:
            _measure(checks, f"raw_ratio:{side}", abs(ratio-emission/magnitude) <= ratio_round)
            clamped = min(max(ratio, LOWER), F(1))
            _require(number(raw["clamped_ratio"]) == clamped, "source_branch", "Preserve source clamp exactly")
            lo, hi = _log_interval(clamped)
            _measure(checks, f"legacy_log:{side}", -vt*hi-log_round-conversion_v <= loss <= -vt*lo+log_round+conversion_v)
        # Both sides must lie wholly in the source's NON-sentinel, UNCLIPPED domain.
        interval = el_log_domain(emission, signed, be, bi, ratio_round)
        if interval is None or sentinel or lower_clip or upper_clip:
            unresolved.append(side); continue
        rlo, rhi = interval
        br = _error(channels["EQE_EL"][side], "1")
        _require(br >= max(abs(ratio-rlo), abs(rhi-ratio)), "unresolved_error", "Complete ratio uncertainty")
        llo = -vt*_log_interval(rhi)[1]; lhi = -vt*_log_interval(rlo)[0]
        bl = _error(channels["delta_V_nr"][side], "V")
        _require(bl >= max(abs(loss-llo), abs(lhi-loss))+log_round+conversion_v,
                 "unresolved_error", "Complete logarithmic uncertainty")
        if side == 0:
            for name, bound in (("J_em_rad", be), ("J_inj", bi), ("EQE_EL", br), ("delta_V_nr", bl)):
                _require(bound == number(channels[name][2]["reference_error_abs"]), "error_binding", name)
    if unresolved:
        checks["physical_unclipped_log"] = {"accepted": False, "status": "unresolved",
                                            "sides": unresolved}
        raise Rejected("unresolved_EL_log_domain", "Unclipped log unavailable on sides "+str(unresolved))
    for name in ("EQE_EL", "delta_V_nr"):
        _scalar(d, context, name, channels[name], d["rules"][name], checks)


def compare_case(definition, supplied):
    """Compare complete supplied contexts; actual identity/error/review gates remain mandatory."""
    checks = {}
    try:
        validate_case_definition(definition)
        _require(supplied.get("protocol") == definition["protocol"], "case_protocol", "Original protocol unchanged")
        _coverage(supplied["contexts"], (c["id"] for c in definition["contexts"]), "All original contexts required")
        operation = {"twod": _twod, "lockin": _lockin, "eqe": _eqe, "el": _el}[definition["kind"]]
        for context in definition["contexts"]:
            local = {}; checks[context["id"]] = local
            operation(definition, context, supplied["contexts"][context["id"]], local)
        accepted = all(item["accepted"] for group in checks.values() for item in group.values())
        return {"accepted": accepted, "status": "passed" if accepted else "not_passed", "checks": checks,
                "scientific_qualification_granted": False}
    except (Rejected, KeyError, IndexError, TypeError, ValueError, OverflowError, ZeroDivisionError) as error:
        code = error.code if isinstance(error, Rejected) else "malformed_case"
        return {"accepted": False, "status": "unresolved" if code.startswith("unresolved") else "rejected",
                "code": code, "detail": str(error), "checks": checks,
                "raw_observations": supplied, "scientific_qualification_granted": False}
