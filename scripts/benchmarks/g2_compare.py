"""G2-R saved-data comparisons; all physical runs and admissions are external.

The pure reductions below also accept explicitly manufactured test evidence.
The plan entry admits and independently reads real inputs before using them.
No comparison, E2 transfer or successful process exit grants a route review.
"""
from __future__ import annotations

from bisect import bisect_right
from copy import deepcopy
from fractions import Fraction as F
import hashlib
import json
import math
from pathlib import Path
import resource
import sys
import time

from scripts.benchmarks import t1_gate as t
from scripts.benchmarks import qualification_comparison as qc

SCHEMA = "solarlab.g2-comparison.v1"
REQUEST_SCHEMA = "solarlab.g2-compare-request.v1"
FIELDS = ("phi_V", "n_m3", "p_m3", "c_m3", "f")
UNITS = dict(qc.STATE_UNITS)
FULL_CASES = ("B_N8_Coarse_Full", "B_N8_Main_Full", "B_N8_Strict_Full", "B_N16_Strict_Full")
CLAIMS = {"rigorous_bound", "conditional_uncertainty_estimate"}


class ComparisonError(ValueError):
    pass


def require(value, reason):
    if not value:
        raise ComparisonError(reason)


def number(value):
    """Only frozen metadata may contain binary64; scientific pairs stay exact."""
    if type(value) is float:
        require(math.isfinite(value), "nonfinite_metadata")
        return F(value)
    return t.q(value)


def span(value):
    c, r = t.enclosure(value)
    return qc.Span(c-r, c+r)


def enc(center, radius=0):
    center, radius = t.q(center), t.q(radius)
    require(radius >= 0, "negative_error")
    return {"center": t.pair(center), "absolute_error_bound": t.pair(radius)}


def enclose(value):
    return enc((value.lo+value.hi)/2, (value.hi-value.lo)/2)


def absolute(value):
    return (F(0) if value.lo <= 0 <= value.hi else min(abs(value.lo), abs(value.hi)), value.absolute_upper)


def summary(rows):
    failure = next((r for r in rows if r["status"] == "failed"), None)
    unknown = next((r for r in rows if r["status"] not in {"passed", "failed"}), None)
    return {"status": "failed" if failure else "unknown" if unknown or not rows else "passed",
            "first_failure": None if failure is None else failure.get("key"),
            "first_unknown": None if unknown is None else {"key": unknown.get("key"), "reason": unknown.get("reason")},
            "records": len(rows)}


def check_samples(case, labels):
    require(case["case_id"] in FULL_CASES and type(case["samples"]) is list, "case_scope")
    t.fixed_windows(labels)
    actual = [row["tag"] for row in case["samples"]]
    require(t.digest(actual) == t.digest(labels), "all_131_ordered_tags_and_event_sides")
    count = case["mesh_intervals"]+1
    require(case["mesh_intervals"] == (16 if case["case_id"] == FULL_CASES[-1] else 8)
            and len(case["mesh_m"]) == count and len(case["volumes_m3"]) == count
            and len(case["cell_edges_m"]) == count+1, "case_geometry")
    mesh = qc.check_mesh(tuple(map(t.q, case["mesh_m"])))
    edges = tuple(map(t.q, case["cell_edges_m"]))
    require(edges[0] == mesh[0] and edges[-1] == mesh[-1]
            and all(a < b for a,b in zip(edges, edges[1:]))
            and all(t.q(v) > 0 for v in case["volumes_m3"]), "physical_control_volumes")
    support = case["material_defect_support"]
    require(len(support) == count and all(type(x) is str and x for x in support)
            and len(set(support)) == 1, "QC01_continuous_material_defect_support")
    for row in case["samples"]:
        require(set(row["states"]) == set(FIELDS) and set(row["ports"]) == {"left", "right"}, "observed_quantities")
        for name in FIELDS:
            require(len(row["states"][name]) == count, "state_shape")
            for v in row["states"][name]:
                if v is not None:
                    c,_ = t.enclosure(v)
                    require(name not in {"n_m3","p_m3"} or c > 0, "positive_carrier_center")
                    require(name != "c_m3" or c >= 0, "nonnegative_ion_center")
                    require(name != "f" or 0 <= c <= 1, "QC01_no_occupancy_clipping")
        for v in row["ports"].values():
            if v is not None:
                t.enclosure(v)


def _sg_box(a, b, xi, s, carrier):
    """Enclose the existing SG reconstruction without losing endpoint words.

    w(xi)=int_0^s exp(xi*u)du/int_0^1 exp(xi*u)du decreases with xi:
    its derivative is a conditional mean below s minus the full mean.
    Thus the two endpoint xi bounds enclose the weight, including xi=0.
    """
    if s == 0:
        return a
    if s == 1:
        return b
    if carrier == "hole":
        xi = -xi
    lo = qc.sg_interpolate(1, 2, xi.hi, s, carrier="electron")-1
    hi = qc.sg_interpolate(1, 2, xi.lo, s, carrier="electron")-1
    w = qc.Span(max(F(0), lo.lo), min(F(1), hi.hi))
    # Positive weights let the endpoint bounds be combined without cancelling
    # independently uncertain a in a+(b-a)*w.
    lower = min((1-w.lo)*a.lo+w.lo*b.lo, (1-w.hi)*a.lo+w.hi*b.lo)
    upper = max((1-w.lo)*a.hi+w.lo*b.hi, (1-w.hi)*a.hi+w.hi*b.hi)
    return qc.Span(lower, upper)


def reconstruct(case, row, *, node_errors=None, center_only=False):
    """The existing 19 physical locations and QC-01 trap rule, with bounds."""
    mesh = tuple(map(t.q, case["mesh_m"]))
    vt = t.q(case["thermal_voltage_V"])
    require(vt > 0, "thermal_voltage")
    values = {}
    for field in FIELDS:
        values[field] = []
        for i, value in enumerate(row["states"][field]):
            if value is None:
                values[field] = None;break
            c, r = t.enclosure(value)
            if node_errors is not None:
                if node_errors[field][i] is None:
                    values[field] = None;break
                r = t.q(node_errors[field][i])
            elif center_only:
                r = F(0)
            values[field].append(qc.Span(c-r, c+r))
    result = []
    for x in qc.POSITIONS:
        physical = x*mesh[-1]
        i = min(bisect_right(mesh, physical)-1, len(mesh)-2)
        s = (physical-mesh[i])/(mesh[i+1]-mesh[i])
        xi = None if values["phi_V"] is None else (values["phi_V"][i+1]-values["phi_V"][i])/vt
        found = {}
        for field, data in values.items():
            if data is None or field in {"n_m3","p_m3"} and xi is None:
                found[field] = None;continue
            found[field] = (_sg_box(data[i],data[i+1],xi,s,"electron" if field == "n_m3" else "hole")
                            if field in {"n_m3","p_m3"} else (1-s)*data[i]+s*data[i+1])
        result.append(found)
    return result


def error_at(errors, quantity, index):
    record = errors.get(quantity)
    if (type(record) is not dict or record.get("status") != "accepted"
            or record.get("claim_kind") not in CLAIMS):
        return None
    values = record.get("values")
    require(type(values) is list and len(values) == 131, "reference_error_131_values")
    value = t.q(values[index])
    require(value >= 0, "reference_error_positive")
    return value


def assess(candidate, reference, error, allowance, *, key, claim_kind=None):
    """One-third reference share; no zero substitution for missing evidence."""
    result = {"key": key, "claim_kind": claim_kind, "status": "unknown", "reason": "unknown_reference_error"}
    if candidate is None or reference is None:
        result["reason"] = "unknown_observation_or_readback"
        return result
    lo, hi = map(t.q, allowance)
    require(0 <= lo <= hi, "allowance_bounds")
    center = (reference.lo+reference.hi)/2
    observed = (candidate-center).absolute_upper
    result.update(observed_difference_upper=t.pair(observed), allowance_lower=t.pair(lo), allowance_upper=t.pair(hi))
    if error is None:
        return result
    error = t.q(error)
    require(error >= 0, "reference_error_nonnegative")
    # E_ref is relative to this reference center and already includes its own
    # read/map/roundoff terms; adding the reference radius again is forbidden.
    total = observed+error
    passed = total <= lo and error <= lo/3
    failed = total > hi or error > hi/3
    result.update(status="passed" if passed else "failed" if failed else "unknown",
                  reason="within_band_and_reference_share" if passed else "comparison_certificate_exceeded" if failed else "unknown_scale",
                  reference_error=t.pair(error), reference_share_lower=t.pair(lo/3),
                  total_upper=t.pair(total), reference_radius_added_again=False)
    return result


def compare_time(candidate, strict, errors, *, labels, scales, checkpoint=lambda: None):
    """Coarse/Main raw ports and all 19-location states against one Strict."""
    check_samples(candidate,labels);check_samples(strict,labels)
    require(candidate["case_id"] in FULL_CASES[:2] and strict["case_id"] == FULL_CASES[2]
            and candidate["problem_identity"] == strict["problem_identity"]
            and t.digest(candidate["mesh_m"]) == t.digest(strict["mesh_m"])
            and candidate["fixed_electrode_gauge"] == strict["fixed_electrode_gauge"]
            and candidate["material_defect_support"] == strict["material_defect_support"], "time_comparison_same_problem")
    signals = {}
    for side in ("left","right"):
        values = [r["ports"][side] for r in strict["samples"]]
        if any(v is None for v in values):
            signals[side] = None
            continue
        initial = span(values[0])
        changes = [absolute(span(v)-initial) for v in values]
        signals[side] = (max(x[0] for x in changes), max(x[1] for x in changes))
    ports, states, initial_checks = [], [], []
    for index,(a,b) in enumerate(zip(candidate["samples"],strict["samples"],strict=True)):
        checkpoint()
        for side in ("left","right"):
            key = {"tag":a["tag"],"port":side}
            c,r = a["ports"][side],b["ports"][side]
            sig = signals[side]
            error = error_at(errors,"port/"+side,index)
            if c is None or r is None or sig is None:
                result = {"key":key,"status":"unknown","reason":"missing_raw_port_observation"}
            else:
                local = absolute(span(r)-span(strict["samples"][0]["ports"][side]))
                band = (max(local[0],sig[0]/100)/1000,max(local[1],sig[1]/100)/1000)
                result = assess(span(c),qc.Span.of(t.enclosure(r)[0]),error,band,key=key,
                                claim_kind=errors.get("port/"+side,{}).get("claim_kind"))
                if error is not None and sig[0] <= 0:
                    result.update(status="unknown",reason="unresolved_current_signal")
                result.update(unit="A",I_sig_lower_A=t.pair(sig[0]),I_sig_upper_A=t.pair(sig[1]),
                              raw_currents_compared=True,both_initials_subtracted=False)
                if index == 0:
                    initial = assess(span(c),qc.Span.of(t.enclosure(r)[0]),error,(sig[0]/100000,sig[1]/100000),
                        key=key,claim_kind=result["claim_kind"])
                    if error is not None and sig[0] <= 0:
                        initial.update(status="unknown",reason="unresolved_current_signal")
                    initial.update(unit="A",rule="initial-current 1e-5 I_sig")
                    initial_checks.append(initial)
            ports.append(result)
        a_values = reconstruct(candidate,a)
        b_values = reconstruct(strict,b,center_only=True)
        forward = {field:[error_at(errors,f"state/{field}/{j}",index) for j in range(9)] for field in FIELDS}
        target = reconstruct(strict,b,node_errors=forward)
        for j,x in enumerate(qc.POSITIONS):
            for field in FIELDS:
                key = {"tag":a["tag"],"field":field,"position_x_over_L":t.pair(x)}
                av = None if a_values is None else a_values[j][field]
                bv = None if b_values is None else b_values[j][field]
                if bv is None:
                    band = (F(0),F(0))
                elif field in {"phi_V","f"}:
                    band = (F(1,10000),)*2
                elif field == "c_m3":
                    band = (t.q(scales["c_m3"])/1000,)*2
                else:
                    lower,upper = absolute(bv)
                    floor = t.q(scales[field])/100
                    band = (max(lower,floor)/1000,max(upper,floor)/1000)
                bound = None if target[j][field] is None or bv is None else (target[j][field]-(bv.lo+bv.hi)/2).absolute_upper
                required_fields = (field,"phi_V") if field in {"n_m3","p_m3"} else (field,)
                kinds = {errors.get(f"state/{f}/{k}",{}).get("claim_kind") for f in required_fields for k in range(9)}
                kind = "conditional_uncertainty_estimate" if "conditional_uncertainty_estimate" in kinds else "rigorous_bound" if kinds == {"rigorous_bound"} else None
                result = assess(av,bv,bound,band,key=key,claim_kind=kind)
                result.update(unit=UNITS[field],reconstruction="SG carriers; physical-coordinate linear phi/c; QC-01 occupancy")
                states.append(result)
    combined = ports+states+initial_checks
    label_order={t.digest(tag):i for i,tag in enumerate(labels)}
    combined.sort(key=lambda r:label_order[t.digest(r["key"]["tag"])])
    return {"status":summary(combined)["status"],"candidate":candidate["case_id"],"reference":strict["case_id"],
            "role":"mandatory Main time comparison" if candidate["case_id"]==FULL_CASES[1] else
                   "complete control response; final applicability belongs to the reviewed RunPlan",
            "port_tables":ports,"state_tables":states,"initial_bias_checks":initial_checks,
            "summary":summary(combined),"covered_tags":131,"state_locations":19,
            "scientific_qualification":False,"reference_center_radius_counted_once":True}


def e2_transfer(strict_value, reference_value, reference_error, *, includes_reference_radius, map_error=0):
    """Exact E_S <= |c_S-c_R| + r_S + E_R + U_map (+ r_R if unowned)."""
    require(type(includes_reference_radius) is bool, "E2_radius_ownership_missing")
    cs,rs = t.enclosure(strict_value);cr,rr = t.enclosure(reference_value)
    er,um = t.q(reference_error),t.q(map_error)
    require(er >= 0 and um >= 0, "E2_negative_error")
    delta = abs(cs-cr)  # exact rational subtraction; zero arithmetic remainder is proved here
    marginal = F(0) if includes_reference_radius else rr
    total = delta+rs+er+um+marginal
    return {"status":"computed_pending_review","claim_kind":"conditional_uncertainty_estimate",
        "conditional_estimate_value":t.pair(total),"rigorous_bound_value":None,
        "ownership":{"D_center_difference":t.pair(delta),"subtraction_arithmetic_bound":t.pair(0),
            "r_S_in_U_read":t.pair(rs),"E_R":t.pair(er),"E_R_includes_r_R":includes_reference_radius,
            "r_R_marginal":t.pair(marginal),"U_map":t.pair(um)},
        "scientific_pass":False,"independent_acceptance_awarded":False}


def spatial_projection(fine, coarse):
    """Inventory-preserving CV overlap, using actual saved physical volumes."""
    x,y = tuple(map(t.q,fine["cell_edges_m"])),tuple(map(t.q,coarse["cell_edges_m"]))
    vf,vc = tuple(map(t.q,fine["volumes_m3"])),tuple(map(t.q,coarse["volumes_m3"]))
    require(x[0] == y[0] and x[-1] == y[-1]
            and set(fine["material_defect_support"]) == set(coarse["material_defect_support"])
            and len(set(fine["material_defect_support"])) == 1, "QC01_projection_support")
    weights = [[max(F(0),min(b,d)-max(a,c))/(d-c)*vf[j]/vc[i]
                for j,(c,d) in enumerate(zip(x,x[1:]))] for i,(a,b) in enumerate(zip(y,y[1:]))]
    require(all(sum(vc[i]*weights[i][j] for i in range(len(vc))) == vf[j] for j in range(len(vf))),
            "CV_projection_inventory")
    return weights


def compare_space(coarse, fine, *, labels, checkpoint=lambda:None):
    check_samples(coarse,labels);check_samples(fine,labels)
    require(coarse["case_id"] == FULL_CASES[2] and fine["case_id"] == FULL_CASES[3]
            and type(coarse["physical_input_identity"]) is str
            and coarse["physical_input_identity"] == fine["physical_input_identity"]
            and coarse["fixed_electrode_gauge"] == fine["fixed_electrode_gauge"], "space_same_physics_and_initial_history")
    matrix = spatial_projection(fine,coarse)
    xc,xf = tuple(map(t.q,coarse["mesh_m"])),tuple(map(t.q,fine["mesh_m"]))
    vc,vf = tuple(map(t.q,coarse["volumes_m3"])),tuple(map(t.q,fine["volumes_m3"]))
    tables = [];ports=[]
    for a,b in zip(coarse["samples"],fine["samples"],strict=True):
        checkpoint()
        for field in FIELDS:
            if any(v is None for v in a["states"][field]+b["states"][field]):
                tables.append({"key":{"tag":a["tag"],"field":field},"status":"unknown","reason":"missing_spatial_state"});continue
            low,high = tuple(map(span,a["states"][field])),tuple(map(span,b["states"][field]))
            if field in {"n_m3","p_m3","c_m3"}:
                projected = [sum((w*v for w,v in zip(row,high)),qc.Span.of(0)) for row in matrix]
            else:
                projected=[]
                for x in xc:
                    j=min(bisect_right(xf,x)-1,len(xf)-2);s=(x-xf[j])/(xf[j+1]-xf[j])
                    projected.append((1-s)*high[j]+s*high[j+1])
            difference=[x-y for x,y in zip(low,projected)]
            squared=sum(v*d.absolute_upper**2 for v,d in zip(vc,difference))/sum(vc)
            from scripts.benchmarks.interval_observation import rational_sqrt_upper
            rms=rational_sqrt_upper(squared,F(1,2**160))
            inventory_low=sum((v*x for v,x in zip(vc,low)),qc.Span.of(0))
            inventory_fine=sum((v*x for v,x in zip(vf,high)),qc.Span.of(0))
            tables.append({"key":{"tag":a["tag"],"field":field},"status":"reported", "unit":UNITS[field],
                "maximum_difference_upper":t.pair(max(x.absolute_upper for x in difference)),
                "weighted_RMS_upper":t.pair(rms),"weighted_mean_square_upper":t.pair(squared),
                "inventory_difference":enclose(inventory_low-inventory_fine) if field in {"n_m3","p_m3","c_m3"} else None,
                "boundary_traces":{"left":enclose(low[0]-high[0]),"right":enclose(low[-1]-high[-1])},
                "projection":"conservative_saved_volume_overlap" if field in {"n_m3","p_m3","c_m3"} else "fixed-gauge_linear" if field == "phi_V" else "QC-01_linear_nodal_occupancy"})
        for side in ("left","right"):
            ports.append({"key":{"tag":a["tag"],"port":side},"unit":"A",
                "status":"reported" if a["ports"][side] is not None and b["ports"][side] is not None else "unknown",
                "difference":None if a["ports"][side] is None or b["ports"][side] is None else enclose(span(a["ports"][side])-span(b["ports"][side]))})
    return {"status":"reported_spatial_difference","projection_record":{"density_weights":[[t.pair(x) for x in r] for r in matrix],
            "inventory_preserved_exactly":True,"source_mesh":"N16","target_mesh":"N8","QC01_support":"matched_continuous_material_and_defect"},
        "role":"spatial differences only; no two-grid order claim",
        "port_tables":ports,"state_tables":tables,"summary":{"covered_tags":131,"spatial_order":None,
            "convergence_order_estimated":False,"G2Q_qualification":False}}


def matched_windows(results, *, checkpoint=lambda:None):
    """Compare the same 128 windows and carried signed prefixes, never failure endpoints."""
    comparisons=[];hypothesis=[]
    anchor=next((results[key]["fixed_windows"] for key in FULL_CASES if key in results),None)
    if anchor is not None:
        for result in results.values():
            require(len(result["fixed_windows"])==len(anchor)==128,"matched_128_windows")
            require(all(t.digest(a["left"])==t.digest(b["left"]) and t.digest(a["right"])==t.digest(b["right"])
                        for a,b in zip(anchor,result["fixed_windows"],strict=True)), "matched_time_window_tags")
    for coarse,finer in zip(FULL_CASES[:2],FULL_CASES[1:3]):
        if coarse not in results or finer not in results:
            comparisons.append({"pair":[coarse,finer],"status":"unknown","reason":"missing_T1_result"});continue
        left,right=results[coarse]["fixed_windows"],results[finer]["fixed_windows"]
        require(len(left)==len(right)==128,"matched_128_windows")
        prefix=[F(0),F(0),F(0),F(0)];complete=True
        for index,(a,b) in enumerate(zip(left,right,strict=True)):
            checkpoint()
            require(t.digest(a["left"])==t.digest(b["left"]) and t.digest(a["right"])==t.digest(b["right"]),"matched_time_window_tags")
            x,y=a.get("signed_device_trend_input"),b.get("signed_device_trend_input")
            if not a["complete_pieces"] or not b["complete_pieces"] or x is None or y is None:
                complete=False
                comparisons.append({"pair":[coarse,finer],"key":{"kind":"window","index":index},"status":"unknown","reason":"missing_signed_window"});continue
            bc,uc,bf,uf=map(t.q,(x["b_C"],x["u_C"],y["b_C"],y["u_C"]))
            one=t.matched_growth(bc,uc,bf,uf)
            comparisons.append({"pair":[coarse,finer],"key":{"kind":"window","index":index},**one})
            if complete:
                prefix=[old+new for old,new in zip(prefix,(bc,uc,bf,uf))]
                comparisons.append({"pair":[coarse,finer],"key":{"kind":"prefix","index":index},**t.matched_growth(*prefix)})
    for case in FULL_CASES:
        if case not in results:continue
        for window in results[case]["fixed_windows"]:
            for channel in ("left_metal","right_metal"):
                entry=window["channels"][channel]
                scale=window.get("independent_window_charge_enclosures_C",{}).get(channel)
                if scale is None or entry.get("signed_balance_C") is None:
                    hypothesis.append({"case_id":case,"window_index":window["index"],"channel":channel,"status":"unknown","reason":"missing_same_window_charge_or_balance"});continue
                lo,hi=absolute(span(scale));b=t.q(entry["signed_balance_C"]);u=t.q(entry["uncertainty_C"])
                lower=max(F(0),abs(b)-u)/hi if hi>0 else None
                upper=(abs(b)+u)/lo if lo>0 else None
                h=F(15,10**6)
                hypothesis.append({"case_id":case,"window_index":window["index"],"channel":channel,
                    "left":window["left"],"right":window["right"],
                    "status":"unknown" if lower is None or upper is None else "computed",
                    "ratio_lower":None if lower is None else t.pair(lower),"ratio_upper":None if upper is None else t.pair(upper),
                    "hypothesis_value":t.pair(h),"hypothesis_compatible":None if lower is None or upper is None else lower<=h<=upper,
                    "systematic_floor_proved":False,"convergence_order_estimated":False})
    return {"status":summary(comparisons)["status"],"device_growth":comparisons,
            "metal_ratio_hypothesis":hypothesis,"scope":"matched signed trends and their arithmetic uncertainty; no new numerical threshold"}


def case_metadata(request, case_id):
    """The fixed B packet and complete initial words, without constructing B."""
    from perovskite_sim.constants import K_B,Q
    packet=request["numeric_packet"]
    definition=packet["definition"]
    require(definition["id"] == "DynamicAcceptorIonPublicDeviceV1", "fixed_B_scope")
    source=request["voltage_lift_map"]["physical_model"]
    from scripts.benchmarks.g2_reference_error import target_identity
    offsets=packet["variable_offsets"]
    initial=request["initial_segment_frame"]
    def fields(key):
        words=initial[key]
        require(len(words) == 12 and all(len(row) == len(request["z0"]) for row in words), "initial_twelve_word_shape")
        values=[sum((F(float.fromhex(row[i])) for row in words),F(0)) for i in range(len(request["z0"]))]
        return {name:values[start:stop] for name,(start,stop) in offsets.items()}
    state,rate=fields("physical_handoff_words_hex"),fields("physical_rate_handoff_words_hex")
    require(set(state) == set(FIELDS), "B_state_layout")
    uniform=all(len(set(v)) == 1 for v in state.values())
    physical_id=(t.digest({"definition":definition,"segments":request["segments"],
        "physical_initial":{k:t.pair(v[0]) for k,v in state.items()}}) if uniform else None)
    return {"case_id":case_id,"mesh_intervals":packet["intervals"],
        "mesh_m":[t.pair(number(x)) for x in packet["x_m"]],
        "cell_edges_m":[t.pair(number(x)) for x in packet["cell_edges_m"]],
        "volumes_m3":[t.pair(number(x)) for x in packet["volumes_m3"]],
        "thermal_voltage_V":t.pair(F(K_B*definition["temperature"]/Q)),
        "source_id":source,"request_sha256":t.digest(request),"problem_identity":t.physical_problem_identity(request),
        "reference_error_problem_identity":target_identity(source,request["initial_preparation"]["point_identity"],request["segments"]),
        "initial_point_identity":request["initial_preparation"]["point_identity"],
        "physical_input_identity":physical_id,"fixed_electrode_gauge":"fixed B Dirichlet electrode gauge",
        "material_defect_support":[t.digest(definition)]*packet["nodes"],
        "initial_state_words_sha256":t.digest(initial["physical_handoff_words_hex"]),
        "initial_rate_words_sha256":t.digest(initial["physical_rate_handoff_words_hex"]),
        "initial_state":{k:list(map(t.pair,v)) for k,v in state.items()},
        "initial_rate":{k:list(map(t.pair,v)) for k,v in rate.items()},
        "raw_words":request["frame_input_policy"]["raw_parent_words"],
        "mapped_words":request["frame_input_policy"]["mapped_input_words"],"samples":[]}


def normalize_ida_sample(request, tag, row, certificate):
    from scripts.benchmarks.native_readback import _decode,_word_pair
    fields,_=qc.full_state(request,row["history"])
    evidence=_decode(certificate["evidence"])
    require(certificate["sample_record_sha256"] == row["record_sha256"]
            and evidence["point_identity"] == row["history"]["point_identity"], "sample_certificate_identity")
    radii=evidence["readback"]["physical_state_absolute_error"]
    states={}
    for field,(start,stop) in request["numeric_packet"]["variable_offsets"].items():
        states[field]=[enc(x,r) for x,r in zip(fields[field],radii[start:stop],strict=True)]
    raw=_word_pair(row["raw"]["total_inward"],2)
    errors=evidence["source_and_readback_error_A"][0]
    require(evidence["input_error_included_once"] is True
            and evidence["readback"]["do_not_add_same_input_error_twice"] is True, "sample_error_ownership")
    return {"tag":dict(tag),"states":states,"ports":{side:enc(v,r) for side,v,r in zip(("left","right"),raw,errors,strict=True)},
        "q":None,"f":None,"source_record_sha256":row["record_sha256"],
        "current_certificate_sha256":t.digest(certificate),"point_identity":row["history"]["point_identity"],
        "map_identity":row["history"]["map_identity"],"input_error_ownership":"included in readback radius once",
        "state_words_consumed":"original physical reference plus all mapped cumulative words",
        "current_origin":"raw full mapped RateView; no initial subtraction"}


def case_from_evidence(plan_case, data, readback_pin, producer_plan_pin):
    """Use the already checked observer stream; never reopen native history."""
    request=data["request"]
    require(t.digest(request) == plan_case["execution"]["effective_input_digest"], "case_effective_input")
    expected=case_metadata(request,plan_case["id"])
    expected["run_id"]=plan_case["execution"]["native_admission"]["operation_id"]
    require(data["comparison_context"] is not None, "missing_sealed_comparison_samples")
    require(t.digest(expected) == t.digest(data["comparison_context"]), "sealed_comparison_context")
    producer=t._pin(producer_plan_pin["path"],producer_plan_pin)
    original=next(c for c in producer["cases"] if c["id"] == plan_case["id"])
    require(original["execution"]["effective_input_digest"] == t.digest(request)
            and original["execution"]["native_admission"]["operation_id"] == expected["run_id"], "producer_plan_case_binding")
    receipt=t._pin(readback_pin["path"],readback_pin)
    require(receipt["schema"] == "solarlab.native-history-verification.v1" and receipt["verified"] is True
            and receipt["complete_protocol"] is True and receipt["complete_readback"] is True
            and receipt["case_id"] == plan_case["id"] and receipt["request_sha256"] == t.digest(request)
            and receipt["plan_sha256"] == producer_plan_pin["sha256"]
            and receipt["history"]["eof_seen"] is True and receipt["history"]["crc_verified"] is True,
            "complete_independent_case_readback")
    observed=receipt["observed_history"]
    require(observed["closed"] is True and observed["container_complete"] is True
            and all(observed[k] == data["history"][k] for k in ("logical_sha256","logical_bytes","records")),
            "observer_stream_readback_binding")
    expected["samples"]=deepcopy(data["comparison_samples"])
    expected["readback"]=receipt
    expected["independent_readback_pin"]=dict(readback_pin)
    expected["sealed_history_binding"]=dict(receipt["history"],
        path=str((Path(readback_pin["path"]).parent/receipt["history"]["path"]).resolve()))
    check_samples(expected,data["labels"])
    return expected


def attach_charge_series(case, data, checked):
    """Use checked window pieces once for q and the independent physical f."""
    pieces={i:[] for i in range(128)}
    for row in data["pieces"]:pieces[row["window_index"]].append(row)
    samples={t.digest(s["tag"]):s for s in case["samples"]}
    for sample in samples.values():
        sample["q"]=sample["f"]=None
    first_charge=None;fc,fr=F(0),F(0);complete=True
    previous=None;previous_index=None
    for index,(left_tag,right_tag,_,_) in enumerate(t.fixed_windows(data["labels"])):
        rows=pieces[index]
        ok=checked["fixed_windows"][index]["complete_pieces"] and bool(rows)
        if not ok:
            complete=False;continue
        a=rows[0]["channels"]["device"].get("left_charge_enclosure_C")
        b=rows[-1]["channels"]["device"].get("right_charge_enclosure_C")
        if a is None or b is None:
            complete=False;continue
        if previous is not None and previous_index == index-1:
            t._compatible_enclosures(previous,a,"charge_series_event_continuity")
        if index == 0:
            first_charge=a
        left,right=samples[t.digest(left_tag)],samples[t.digest(right_tag)]
        if first_charge is not None:
            left["q"]=enc(0) if index == 0 else enclose(span(a)-span(first_charge))
            right["q"]=enclose(span(b)-span(first_charge))
        if complete:left["f"]=enc(fc,fr)
        for piece in rows:
            operands=t.operands(piece["channels"]["device"])
            if operands is None:
                complete=False;break
            fc+=operands["fc"]
            fr+=sum(operands["components"][k] for k in t.COMPONENTS[1:])
        if complete:right["f"]=enc(fc,fr)
        previous=b;previous_index=index


def quantity_values(case, quantity):
    if quantity in {"q","f"}:
        return [r.get(quantity) for r in case["samples"]]
    parts=quantity.split("/")
    if len(parts)==2 and parts[0]=="port" and parts[1] in {"left","right"}:
        return [r["ports"][parts[1]] for r in case["samples"]]
    require(len(parts)==3 and parts[0]=="state" and parts[1] in FIELDS and parts[2].isdigit(), "quantity_key")
    index=int(parts[2]);require(0 <= index <= case["mesh_intervals"], "quantity_index")
    return [r["states"][parts[1]][index] for r in case["samples"]]


def quantity_unit(quantity):
    return "C" if quantity in {"q","f"} else "A" if quantity.startswith("port/") else UNITS[quantity.split("/")[1]]


def quantity_name(quantity):
    return quantity if quantity in {"q","f"} else "port_"+quantity.split("/")[1] if quantity.startswith("port/") else quantity.split("/")[1]


def read_error_records(result_pin, acceptance_pin, *, method_pin, problem_identity, labels, checkpoint=lambda:None):
    """Consume A5's structured v1 acceptance, retaining exact per-quantity failures."""
    records = {}
    response = {"result":result_pin,"acceptance":acceptance_pin,"quantities":records,
                "source_claim_status_unchanged":True,"status":"unknown_reference_error","reason":None}
    try:
        payload = t._acceptance_pin(result_pin)
        require(payload["schema"] == "solarlab.g2-reference-error.v1"
                and payload["target_problem_identity"] == problem_identity
                and t.digest(payload["method_contract"]) == t.digest(method_pin), "reference_error_envelope")
        descriptors = payload["quantity_records"]
        require(type(descriptors) is dict and 0 < len(descriptors) <= 49, "reference_error_quantity_descriptors")
        for key, ref in descriptors.items():
            t._acceptance_quantity(key)
            records[key] = {"record":None,"pin":ref,"usable_for_E2":False,"acceptance":None,
                            "status":"unknown_reference_error","reason":"quantity_not_explicitly_accepted"}
        require(acceptance_pin is not None, "reference_error_acceptance_missing")
        review = t._acceptance_pin(acceptance_pin)
        required = {"schema","version","reviewed_error_file","accepted_quantities","quantity_bindings",
            "claim_kind","reviewer","verdict","conditions","method_contract","parameters","lineage",
            "target_problem_identity","reference_history","scope","scientific_pass","execution_authorized"}
        accepted = t._acceptance_header(review, "solarlab.g2-reference-error-acceptance.v1", required)
        require(t.digest(review["reviewed_error_file"]) == t.digest(result_pin)
                and t.digest(review["method_contract"]) == t.digest(method_pin)
                and review["target_problem_identity"] == problem_identity
                and type(review["scope"]) is dict and review["scope"].get("route") == "G2-R"
                and type(review["scope"].get("meaning")) is str and review["scope"]["meaning"]
                and set(accepted) <= set(descriptors), "reference_error_review_binding")
        method = t._acceptance_pin(method_pin)
        from scripts.benchmarks.g2_route import _frozen_definition, EVIDENCE_METHOD_SCHEMA
        _frozen_definition(method, EVIDENCE_METHOD_SCHEMA, 1)
        parameters = t._acceptance_pin(review["parameters"])
        lineage = t._acceptance_pin(review["lineage"])
        require(parameters.get("schema") == "solarlab.g2-reference-error-parameters.v1"
                and parameters.get("target_problem_identity") == problem_identity
                and parameters.get("method_contract_sha256") == method_pin["sha256"]
                and lineage.get("schema") == "solarlab.g2-reference-error.v1"
                and lineage.get("kind") == "reference_error_lineage"
                and t.digest(lineage["method_contract"]) == t.digest(method_pin)
                and t.digest(lineage["parameters"]) == t.digest(review["parameters"]), "reference_error_parameters_lineage")
        levels = lineage["levels"]
        require(type(levels) is list and len(levels) == 3, "reference_error_three_level_lineage")
        for ordinal, level in enumerate(levels):
            identity = level["identity"]
            require(type(identity) is dict and identity.get("case_id") == f"BE_N8_L{ordinal}_Full"
                    and identity.get("method_contract_sha256") == method_pin["sha256"]
                    and all(type(identity.get(key)) is str and identity[key] for key in
                        ("input_sha256","plan_sha256","source_commit","environment_identity","schedule_identity",
                         "source_identity","initial_point_identity","controls_sha256")), "reference_error_level_identity")
        finest = levels[2]
        history = t._acceptance_history(review["reference_history"], case_id="BE_N8_L2_Full", labels=labels,
            expected={"source_identity":finest["identity"]["source_identity"],
                      "run_id":t.digest(finest["identity"]),"input_sha256":finest["identity"]["input_sha256"]})
        require(all(finest["observed_history"][key] == history[key] for key in
                    ("logical_sha256","logical_bytes","encoded_bytes","records")), "reference_error_L2_history")
        response["reference_history"] = history
        response["review"] = review
        for key in accepted:
            checkpoint()
            entry = records[key]
            try:
                r = t._acceptance_pin(entry["pin"]); entry["record"] = r
                require(r["schema"] == "solarlab.g2-reference-evidence.v1" and r["quantity_key"] == key
                        and r["quantity_and_unit"] == {"quantity":quantity_name(key),"unit":quantity_unit(key)}
                        and r["target_problem_identity"] == problem_identity
                        and r["status"] == "computed_pending_review" and r.get("method_status", "passed") == "passed"
                        and r["claim_kind"] == "conditional_uncertainty_estimate"
                        and r["rigorous_bound_value"] is None and r.get("diagnostic_value") is None,
                        "quantity_record_binding")
                require(type(r["norm_and_coverage"]["records"]) is int
                        and r["norm_and_coverage"] == {"norm":"max_abs_tagged131","records":131,"labels_sha256":t.digest(labels)}
                        and len(labels) == 131 and t.digest([row["tag"] for row in r["per_record"]]) == t.digest(labels),
                        "reference_error_coverage")
                identity = r["reference_source_and_run_identity"]
                require(identity == {"case_id":"BE_N8_L2_Full","run_id":history["run_id"],
                    "source_identity":history["source_identity"],"request_sha256":history["input_sha256"],
                    "history_sha256":history["logical_sha256"]}, "reference_error_quantity_history")
                authority = r["reviewer_and_decision_authority"]
                require(authority["evidence_author"] == review["reviewer"]["evidence_author"]
                        and t.digest(authority["method_contract"]) == t.digest(method_pin), "reference_error_review_independence")
                require(t.digest(r["assumptions_and_checked_premises"]["E3_parameters"]) == t.digest(review["parameters"])
                        and all(t.digest(component["source"]) == t.digest(review["lineage"])
                                for component in r["error_components_and_ownership"])
                        and bool(r["error_components_and_ownership"]), "reference_error_quantity_lineage")
                values = []
                for row in r["per_record"]:
                    computation = row["computation"]
                    require(computation["status"] == "computed_pending_review"
                            and computation.get("method_status", "passed") == "passed", "reference_error_method_status")
                    error = t._acceptance_pair(computation["E_R"])
                    radius = t._acceptance_pair(row["computed_value"]["radius"])
                    t.q(row["computed_value"]["center"])
                    require(error >= radius, "reference_error_tagwise_radius_not_owned")
                    values.append(error)
                require(t._acceptance_pair(r["conditional_estimate_value"]) == max(values)
                        and t.digest(r["estimated_or_enclosed_value"]) == t.digest({"value_field":"conditional_estimate_value",
                            "value":r["conditional_estimate_value"],"unit":quantity_unit(key)})
                        and r.get("reference_value_radius_ownership") == "reported observation radii are included in E_R; coalesce them in E2",
                        "reference_error_value_slot_and_radius")
                t._acceptance_quantity_binding(review["quantity_bindings"][key], r, entry["pin"], key=key, labels=labels)
                entry.update(usable_for_E2=True, acceptance=acceptance_pin, status="accepted_for_E2", reason=None)
            except (KeyError, TypeError, ValueError, OSError, AttributeError, OverflowError) as error:
                entry["reason"] = str(error)
        active = [key for key, entry in records.items() if entry["usable_for_E2"]]
        response.update(status="accepted_for_E2" if active else "unknown_reference_error",
                        reason=None if active else "no_accepted_reference_error_quantity", accepted_quantities=active)
    except (KeyError, TypeError, ValueError, OSError, AttributeError, OverflowError) as error:
        response["reason"] = str(error)
        for entry in records.values():
            entry.update(usable_for_E2=False, acceptance=None, status="unknown_reference_error", reason=str(error))
    return response


def transfer_E2(strict, reference, records, *, labels, checkpoint=lambda:None, context=None):
    require(strict["problem_identity"] == reference["problem_identity"]
            and t.digest(strict["mesh_m"]) == t.digest(reference["mesh_m"])
            and t.digest([r["tag"] for r in strict["samples"]]) == t.digest(labels)
            and t.digest([r["tag"] for r in reference["samples"]]) == t.digest(labels), "E2_same_problem_mesh_tags")
    require(strict["reference_error_problem_identity"] == reference["reference_error_problem_identity"],
            "E2_original_point_and_operator_identity")
    output={}
    for key,owned in records["quantities"].items():
        checkpoint();record=owned["record"]
        base={"schema":"solarlab.g2-reference-evidence.v1","quantity_key":key,
            "quantity_and_unit":{"quantity":quantity_name(key),"unit":quantity_unit(key)},
            "target_problem_identity":strict["problem_identity"],
            "norm_and_coverage":{"norm":"max_abs_tagged131","records":131,"labels_sha256":t.digest(labels)},
            "claim_kind":"conditional_uncertainty_estimate","rigorous_bound_value":None,
            "source_BE_error":owned["pin"],"source_BE_acceptance":owned["acceptance"],"scientific_pass":False}
        if not owned["usable_for_E2"]:
            output[key]={**base,"status":"unknown_reference_error","reason":owned.get("reason"),
                         "conditional_estimate_value":None,"per_record":[]};continue
        require(record["target_problem_identity"] == reference["reference_error_problem_identity"],
                "E2_BE_error_problem_identity")
        identity=record["reference_source_and_run_identity"]
        require(identity["source_identity"] == reference["source_id"]
                and identity["request_sha256"] == reference["request_sha256"]
                and identity["run_id"] == reference["run_id"], "E2_reference_source_and_run")
        ownership=record.get("reference_value_radius_ownership")
        included=(True if ownership == "reported observation radii are included in E_R; coalesce them in E2" else
                  False if ownership == "reported observation radii are not included in E_R; add them in E2" else None)
        a,b=quantity_values(strict,key),quantity_values(reference,key);rows=[]
        for tag,x,y,source in zip(labels,a,b,record["per_record"],strict=True):
            if x is None or y is None or included is None or source.get("computed_value") is None:
                rows.append({"tag":tag,"status":"unknown_reference_error","conditional_estimate_value":None});continue
            recorded=source["computed_value"]
            value=enc(t.q(recorded["center"]),t.q(recorded["radius"]))
            t._compatible_enclosures(value,y,"E2_recorded_BE_observation")
            require(not included or t.q(source["computation"]["E_R"]) >= t.enclosure(value)[1],
                    "E2_E_R_does_not_cover_declared_radius")
            result=e2_transfer(x,value,source["computation"]["E_R"],includes_reference_radius=included,map_error=0)
            rows.append({"tag":tag,**result,"strict_value":x,"reference_value":value,
                "U_map_reason":"identical physical nodes/ports and exact event-side tags; no interpolation in E2"})
        complete=all(r["status"] == "computed_pending_review" for r in rows)
        value=max(t.q(r["conditional_estimate_value"]) for r in rows) if complete else None
        output[key]={**base,"status":"computed_pending_review" if complete else "unknown_reference_error",
            "conditional_estimate_value":None if value is None else t.pair(value),"per_record":rows,
            "estimated_or_enclosed_value":{"value_field":"conditional_estimate_value","value":None if value is None else t.pair(value),"unit":quantity_unit(key)},
            "strict_source_and_run_identity":{k:strict[k] for k in ("source_id","request_sha256","run_id")},
            "independent_Strict_reference_review":"pending"}
    required={"q","f","port/left","port/right"}|{f"state/{name}/{i}" for name in FIELDS for i in range(9)}
    missing=sorted(required-set(output))
    return {"schema":"solarlab.g2-E2-transfer.v1",**(context or {}),
            "status":"computed_pending_review" if not missing and all(v["status"]=="computed_pending_review" for v in output.values()) else "unknown_reference_error",
            "target_problem_identity":strict["problem_identity"],
            "BE_error_problem_identity":reference["reference_error_problem_identity"],
            "strict_source_and_run_identity":{"case_id":strict["case_id"],**{k:strict[k] for k in ("source_id","request_sha256","run_id")}},
            "labels":labels,"strict_responses":[{"tag":r["tag"],"q_C":r.get("q"),"f_C":r.get("f")} for r in strict["samples"]],
            "quantities":output,"missing_quantities":missing,"scientific_pass":False,"E_R_accepted_does_not_accept_E_S":True}


def normalize_be_readback(request, receipt, identity, labels):
    """Decode the BE reader's actual physical words and observed enclosures."""
    require(receipt["structural_pass"] is True and receipt["numerical_complete"] is True
            and receipt["scientific_status"] != "failed"
            and t.digest([r["tag"] for r in receipt["observations"]]) == t.digest(labels), "BE_complete_reference_coverage")
    metadata=case_metadata(request,identity.case_id)
    require(metadata["source_id"] == identity.source_identity
            and metadata["initial_point_identity"] == identity.initial_point_identity, "BE_actual_source_and_initial_point")
    metadata.update(source_id=identity.source_identity,run_id=t.digest(identity.payload()),readback=receipt)
    require(metadata["request_sha256"] == identity.input_sha256, "BE_request_identity")
    rows=receipt["observations"]
    def observed(value):
        return enc(t.q(value["center"]),t.q(value["radius"]))
    first=observed(rows[0]["charges"]["device"])
    for index,row in enumerate(rows):
        states={}
        for field,data in row["physical_state_words"].items():
            require(field in FIELDS and data["shape"] == [9], "BE_state_shape")
            words=data["words_hex"]
            require(words and all(len(word)==9 for word in words), "BE_physical_word_layers")
            states[field]=[enc(sum((F(float.fromhex(word[i])) for word in words),F(0))) for i in range(9)]
        require(set(states)==set(FIELDS), "BE_state_quantities")
        charge=observed(row["charges"]["device"])
        metadata["samples"].append({"tag":row["tag"],"states":states,
            "ports":{side:observed(row["ports"][side]) for side in ("left","right")},
            "q":enc(0) if index==0 else enclose(span(charge)-span(first)),
            "f":observed(row["cumulative_physical_flux"][0]),
            "source_record_sha256":row["record_sha256"],"point_identity":row["point_identity"],
            "state_radius_scope":"zero for exact complete-word decoding only; forward error is external E_R"})
    return metadata


def read_be_case(descriptor, labels, limits, checkpoint, layout=None):
    from scripts.benchmarks import g2_reference_be as be
    identity=be.RunIdentity(**descriptor["identity"]);identity.payload()
    require(identity.case_id == "BE_N8_L2_Full", "fixed_finest_BE_reference")
    request=t._pin(descriptor["request"]["path"],descriptor["request"])
    require(t.digest(request)==identity.input_sha256, "BE_input_identity_before_layout")
    if layout is None:
        layout=be.reference_layout(request)
    require(request["numeric_packet"]["layout_identity"]==layout.identity, "BE_layout_and_input")
    receipt=be.read_reference_history(Path(descriptor["directory"]),layout,identity,
        max_record_bytes=min(limits["max_record_bytes"],1048576),max_logical_bytes=limits["max_logical_bytes"],checkpoint=checkpoint)
    result=normalize_be_readback(request,receipt,identity,labels)
    result["sealed_history_binding"]=dict(receipt["observed_history"],
        path=str((Path(descriptor["directory"])/receipt["observed_history"]["path"]).resolve()))
    return result


def comparison_reference_errors(reference, *, spec, method_pin, labels, strict):
    """Return explicit acceptance status/reasons and only the qualified keys."""
    result={"status":"unknown_reference_error","reason":"Strict_reference_acceptance_missing",
            "quantities":{},"quantity_failures":{}}
    if reference is None:
        return result
    try:
        acceptance = t.accepted_strict_E2(reference,spec=spec,labels=labels,problem_identity=strict["problem_identity"])
        if acceptance["status"] != "accepted":
            result["reason"] = acceptance["reason"]
            return result
        e2, accepted = acceptance["e2"], acceptance["accepted_quantities"]
        require(t.digest(e2["method_contract"]) == t.digest(method_pin), "Strict_frozen_method")
        identity=e2["strict_source_and_run_identity"]
        declared=e2["inputs"]["IDA_Strict_sealed_history"]
        actual=strict["sealed_history_binding"]
        require(all(identity[k] == strict[k] for k in ("case_id","source_id","request_sha256","run_id"))
                and all(declared[k] == actual[k] for k in ("logical_sha256","logical_bytes","encoded_bytes","records")),
                "Strict_actual_source_and_history")
        actual_readback = strict.get("independent_readback_pin", actual.get("independent_readback"))
        require(actual_readback is not None and t.digest(declared["independent_readback"]) == t.digest(actual_readback),
                "Strict_actual_independent_readback")
    except (KeyError,TypeError,ValueError,OSError,AttributeError,OverflowError) as error:
        result["reason"]=str(error)
        return result
    qualified=result["quantities"]
    for key in accepted:
        try:
            values=t.strict_E2_values(e2,key,quantity=quantity_name(key),unit=quantity_unit(key))
            current=quantity_values(strict,key)
            record=e2["quantities"][key]
            require(all(value is not None and t.digest(value) == t.digest(row["strict_value"])
                        for value,row in zip(current,record["per_record"],strict=True)), "Strict_exact_saved_operands")
            if key in {"q","f"}:
                require(t.digest([r[key+"_C"] for r in e2["strict_responses"]]) == t.digest(current),
                        "Strict_charge_response_binding")
            qualified[key]={"status":"accepted","source_status":record["status"],
                "claim_kind":"conditional_uncertainty_estimate","values":list(map(t.pair,values)),
                "qualification_source":reference["independent_review"],"e2_file":reference["e2_file"],
                "target_problem_identity":strict["problem_identity"],"quantity_and_unit":record["quantity_and_unit"],
                "norm_and_coverage":record["norm_and_coverage"],"strict_source_and_run_identity":identity}
        except (KeyError,TypeError,ValueError,AttributeError,OverflowError) as error:
            result["quantity_failures"][key]=str(error)
    result.update(status="accepted_reference_error" if qualified else "unknown_reference_error",
                  reason=None if qualified else "no_qualified_Strict_reference_quantity")
    return result

def evaluate_T1(data, reference=None, known_failure=None):
    manifest=data["manifest"]
    return t.evaluate(data["intervals"],data["pieces"],labels=data["labels"],spec=data["spec"],
        case_id=manifest["case_id"],run_id=manifest["run_id"],problem_identity=manifest["problem_identity"],
        reference=reference,source_id=manifest["source_id"],request_sha256=t.digest(data["request"]),
        refresh_times=manifest.get("actual_refresh_times_hex",()),checkpoint=data["checkpoint"],known_device_failure=known_failure)


def decision_payload(cases, results, comparisons, spatial, transfers, growth, *, bindings, failures, errors):
    """Direct payload for G2RDecisionDraftV1 comparison_ingest.normalized_evidence."""
    reference_groups={"q":"device_q","port/left":"left_current","port/right":"right_current",
        "phi_V":"phi","n_m3":"n","p_m3":"p","c_m3":"c","f":"f_occupancy"}
    reference={name:{"status":"unknown_reference_error","claim_kind":None,"quantities":{},
        **{field:None for field in ("evidence_class","target_problem_identity","units_and_norm",
            "sample_or_interval_coverage","reference_source_and_run","error_components","input_contribution",
            "reference_contribution","allowance","one_third_check","applicable_one_twelfth_axes",
            "assumptions_and_common_bias","review")}} for name in
        ("device_q","device_integrated_f","left_current","right_current","phi","n","p","c","f_occupancy")}
    for key,row in transfers.get("quantities",{}).items():
        group=("device_integrated_f" if key=="f" else reference_groups[key] if key in {"q","port/left","port/right"}
               else reference_groups[key.split("/")[1]])
        reference[group]["quantities"][key]=row
        reference[group].update(evidence_class="G2-R_BE_to_Strict_E2",target_problem_identity=row["target_problem_identity"],
            reference_source_and_run=row.get("strict_source_and_run_identity"),
            assumptions_and_common_bias="Conditional on accepted E_R and E2 transfer; common model bias is not excluded.")
        reference[group]["units_and_norm"]=dict(reference[group]["units_and_norm"] or {},
            **{key:{"quantity_and_unit":row["quantity_and_unit"],"norm_and_coverage":row["norm_and_coverage"]}})
        reference[group]["sample_or_interval_coverage"]=row["norm_and_coverage"]
        reference[group]["error_components"]={"location":"quantities/<quantity>/per_record/ownership",
            "formula":"abs(c_S-c_R)+r_S+E_R+U_map; r_R only when not already in E_R"}
        reference[group]["reference_contribution"]={"location":"quantities/<quantity>/source_BE_error"}
        reference[group]["claim_kind"]=row["claim_kind"]
        reference[group]["status"]="computed_pending_review" if all(v["status"]=="computed_pending_review" for v in reference[group]["quantities"].values()) else "unknown_reference_error"
    for name,entry in reference.items():
        required=9 if name in {"phi","n","p","c","f_occupancy"} else 1
        entry["missing_quantity_count"]=required-len(entry["quantities"])
        if entry["missing_quantity_count"]:
            entry["status"]="unknown_reference_error"
    for key,row in errors.items():
        group=("device_integrated_f" if key=="f" else reference_groups[key] if key in {"q","port/left","port/right"}
               else reference_groups[key.split("/")[1]])
        reference[group].setdefault("accepted_Strict_quantities",{})[key]=row
        reference[group]["review"]=row["qualification_source"]
        reference[group].update(evidence_class="G2-R_BE_to_Strict_E2",target_problem_identity=row["target_problem_identity"],
            reference_source_and_run=row["strict_source_and_run_identity"],sample_or_interval_coverage=row["norm_and_coverage"])
        reference[group]["units_and_norm"]=dict(reference[group]["units_and_norm"] or {},
            **{key:{"quantity_and_unit":row["quantity_and_unit"],"norm_and_coverage":row["norm_and_coverage"]}})
        reference[group]["error_components"]={"e2_file":row["e2_file"],"location":"quantities/<quantity>/per_record/ownership"}
    for name,entry in reference.items():
        required=9 if name in {"phi","n","p","c","f_occupancy"} else 1
        if len(entry.get("accepted_Strict_quantities",{}))==required:
            entry.update(status="accepted_reference_error",claim_kind="conditional_uncertainty_estimate")
        entry["allowance"]={"source":"frozen RunPlanV3 and T1GateSpecV3; exact operands are in the gate tables"}
        entry["one_third_check"]={"required_fraction":[1,3],
            "source":"point_comparisons tables and matched_window_T1.device reference_share disposition"}
        entry["applicable_one_twelfth_axes"]={"status":"not_inferred","reason":"E_R/E2 joint ownership does not create independent 1/12 credits"}
    safety={}
    for key,case in cases.items():
        receipt=case["readback"]
        progress=receipt.get("scientific_progress",{})
        safety[key]={"complete_protocol":receipt["complete_protocol"],"complete_readback":receipt["complete_readback"],
            "reader_verified":receipt["verified"],"failed_gates":progress.get("failed_gates",[]),
            "first_failure":progress.get("first_failure"),"scientific_pass_awarded":False}
    unknown={"status":"unknown","reason":"missing_required_case_data","port_tables":[],"state_tables":[],
             "initial_bias_checks":[],"projection_record":None,"summary":None}
    normalized={"status":"computed_pending_review" if not failures else "unknown",
        "binding_check":bindings,"record_keys_source":"RunPlanV3#/protocol/requested_sample_records",
        "required_counts":{"full_case_tags":131,"unique_times":129,"positive_matched_windows":128,"probe_tags":66},
        "missing_records":failures,"extra_records":[],"duplicate_or_mismatched_sides":[],
        "point_comparisons":{"N8_Main_vs_Strict":comparisons.get(FULL_CASES[1],deepcopy(unknown)),
            "N8_Coarse_vs_Strict":comparisons.get(FULL_CASES[0],deepcopy(unknown)),
            "N8_Strict_vs_N16_Strict":spatial or deepcopy(unknown)},
        "reference_evidence":reference,
        "matched_window_T1":{"status":growth["status"],
            "device":{key:results[key]["channels"]["device"] if key in results else
                {"status":"not_evaluated","reason":"missing_or_invalid_required_case_data"} for key in FULL_CASES},
            "metal_left":{key:results[key]["channels"]["left_metal"] if key in results else
                {"status":"unknown","reason":"missing_or_invalid_required_case_data"} for key in FULL_CASES},
            "metal_right":{key:results[key]["channels"]["right_metal"] if key in results else
                {"status":"unknown","reason":"missing_or_invalid_required_case_data"} for key in FULL_CASES},
            "all_non_cancelling_prefixes":{key:value["native"] for key,value in results.items()},
            "fixed_window_certificates":{key:value["fixed_windows"] for key,value in results.items()},
            "device_growth":growth["device_growth"],"metal_ratio_hypothesis":growth["metal_ratio_hypothesis"],
            "legacy_absolute_diagnostic":"retained in each native channel record",
            "three_ledger_diagnostic":{"status":"not_evaluated_here","source":"original sealed observation records; not a hard comparison gate"}},
        "full_path_safety_T2_T3":safety,"frozen_gate_and_policy_match":bindings,
        "independent_comparison_review":None}
    main=results.get(FULL_CASES[1])
    known_failure=(any(v["status"]=="failed" for v in comparisons.values())
        or growth["status"]=="failed" or any(c["status"]=="failed" for r in results.values() for c in r["channels"].values())
        or any(v["failed_gates"] for v in safety.values()))
    return {"schema":SCHEMA,"scientific_status":"failed" if known_failure else "unknown",
        "scientific_status_scope":"retained observed failures; hard-gate applicability and route acceptance remain external",
        "scientific_pass":False,"route_decision":"pending_independent_review",
        "normalized_evidence":normalized,"E2_transfer":transfers,"L_U2":{
            "eligible":False,"reason":"Main_T1_not_evaluated","scientific_pass":False} if main is None else main.get("L_U2"),
        "scope":"finite 131-tag time comparison and diagnostic two-mesh differences; no continuous-time or G2-Q qualification"}


def compare_loaded(cases, data, *, spec, method_pin, labels, reference=None, BE=None, BE_errors=None,
                   route_evidence=None, bindings=None, failures=None, checkpoint=lambda:None):
    """Shared pure reduction after the caller has admitted and checked files."""
    failures=[] if failures is None else list(failures)
    results={};comparisons={};spatial=None
    for key,value in data.items():
        checkpoint()
        try:
            results[key]=evaluate_T1(value)
            if key in cases:attach_charge_series(cases[key],value,results[key])
        except (ValueError,KeyError,TypeError) as error:
            failures.append({"case":key,"reason":"invalid_T1_evidence: "+str(error)[:400]})
    strict=cases.get(FULL_CASES[2])
    transfers={"status":"unknown_reference_error","quantities":{}}
    if strict is not None and BE is not None and BE_errors is not None:
        e2_context={"gate_spec":bindings["gate_spec"],"method_contract":method_pin,
            "evidence_author":"WS1 Main g2_compare", "inputs":{
                "E_R_acceptance":BE_errors["acceptance"],"IDA_Strict_sealed_history":strict["sealed_history_binding"],
                "BE_reference_history":BE["sealed_history_binding"]},
            "computation_provenance":{k:bindings[k] for k in ("plan_sha256","manifest","evaluation_source_review")}}
        transfers=transfer_E2(strict,BE,BE_errors,labels=labels,checkpoint=checkpoint,context=e2_context)
    reference_acceptance=({"status":"unknown_reference_error","reason":"Strict_case_missing","quantities":{}}
        if strict is None else comparison_reference_errors(reference,spec=spec,method_pin=method_pin,labels=labels,strict=strict))
    errors=reference_acceptance["quantities"]
    t1_reference=reference if {"q","f"} <= set(errors) else None
    scales={field:number(spec_context) for field,spec_context in
        (("n_m3",bindings["physical_context"]["n_ref_m3"]),("p_m3",bindings["physical_context"]["p_ref_m3"]),
         ("c_m3",bindings["physical_context"]["c_ref_m3"]))}
    if strict is not None:
        for key in FULL_CASES[:2]:
            if key in cases:comparisons[key]=compare_time(cases[key],strict,errors,labels=labels,scales=scales,checkpoint=checkpoint)
        if FULL_CASES[3] in cases:
            try:spatial=compare_space(strict,cases[FULL_CASES[3]],labels=labels,checkpoint=checkpoint)
            except (ValueError,KeyError,TypeError) as error:failures.append({"case":FULL_CASES[3],"reason":str(error)})
    if t1_reference is not None:
        for key in list(results):results[key]=evaluate_T1(data[key],t1_reference)
    growth=matched_windows(results,checkpoint=checkpoint)
    failed_growth=next((r for r in growth["device_growth"] if r["status"]=="failed"),None)
    if failed_growth is not None and FULL_CASES[1] in results:
        results[FULL_CASES[1]]=evaluate_T1(data[FULL_CASES[1]],t1_reference,{"kind":"matched_growth",**failed_growth})
    elif growth["status"] != "passed" and FULL_CASES[1] in results:
        main_result=results[FULL_CASES[1]]
        device=main_result["channels"]["device"]
        if device["status"] in {"passed","unresolved_signal"}:
            device.update(local_certificate_status=device["status"],status="unknown",reason="unknown_matched_window_growth")
            main_result["L_U2"]=t.lu2_eligibility(main_result)
    if FULL_CASES[1] in results and route_evidence is not None:
        main=cases.get(FULL_CASES[1]);readback={} if main is None else main["readback"]
        recorded=readback.get("scientific_progress",{}).get("failed_gates")
        checked=(readback.get("verified") is True and readback.get("complete_protocol") is True and recorded is not None)
        expected={"history_identity":"passed" if checked else "unknown",
            "safety":"passed" if checked and not recorded else "failed" if recorded else "unknown",
            "T2":"failed" if recorded and "state_accuracy" in recorded else "passed" if checked else "unknown",
            "T3":"passed" if checked else "unknown","matched_growth":growth["status"],
            "port_reference_comparisons":summary(comparisons.get(FULL_CASES[1],{}).get("port_tables",[])
                +comparisons.get(FULL_CASES[1],{}).get("initial_bias_checks",[]))["status"],
            "state_reference_comparisons":summary(comparisons.get(FULL_CASES[1],{}).get("state_tables",[]))["status"]}
        if route_evidence.get("checks") == expected:
            results[FULL_CASES[1]]["L_U2"]=t.lu2_eligibility(results[FULL_CASES[1]],route_evidence)
    result=decision_payload(cases,results,comparisons,spatial,transfers,growth,
                            bindings=bindings,failures=failures,errors=errors)
    result["Strict_reference_acceptance"]=reference_acceptance
    return result


def compare_plan(context, *, be_layout=None):
    """Admitted evaluation entry; reuse the existing channel/watchdog contract."""
    from scripts.benchmarks import g2_route as route
    from scripts.benchmarks.native_history import integer_json_dumps
    plan=context["plan"]
    execution=plan.get("comparison_execution")
    require(type(execution) is dict, "comparison_execution_not_bound")
    route._validate_supervision(plan,execution)
    resource_caps=plan["resources"]
    names=("max_record_bytes","max_logical_bytes","max_records","max_retained_bytes","max_output_bytes","rss_bytes")
    require(all(type(execution.get(k)) is int and execution[k]>0 for k in names), "comparison_resource_fields")
    require(execution["max_logical_bytes"]<=resource_caps["reader_logical_byte_cap"]
            and execution["max_output_bytes"]<=resource_caps["combined_persisted_case_bytes"]
            and execution["max_retained_bytes"]<=execution["rss_bytes"]<=resource_caps["rss_bytes_per_case"]
            and type(execution["wall_s"]) in (int,float) and math.isfinite(execution["wall_s"])
            and 0<execution["wall_s"]<=resource_caps["full_wall_cap_max_s"], "comparison_resource_caps")
    deadline=time.perf_counter()+execution["wall_s"]
    def checkpoint():
        rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=="darwin" else 1024)
        require(time.perf_counter()<deadline and rss<=execution["rss_bytes"], "comparison_time_or_RSS_cap")
    checkpoint()
    manifest=t._pin(execution["manifest"]["path"],execution["manifest"])
    require(manifest["schema"]==REQUEST_SCHEMA and set(manifest["cases"])<=set(FULL_CASES), "comparison_manifest_schema")
    gate_pin=plan["source_freeze"]["frozen_gate_spec"]
    method_pin=plan["source_freeze"]["frozen_evidence_method_contract"]
    gate=t._pin(gate_pin["path"],gate_pin)
    spec=t.CandidateSpec.from_document(gate,gate_pin["sha256"])
    require(spec.version==3, "comparison_frozen_V3")
    labels=plan["protocol"]["requested_sample_records"]
    t.fixed_windows(labels)
    cases={};data={};failures=[];logical=retained=records=0
    with route._channel(plan,execution):
        for key in FULL_CASES:
            checkpoint()
            if key not in manifest["cases"]:
                failures.append({"case":key,"reason":"missing_comparison_case_descriptor"});continue
            item=manifest["cases"][key]
            try:
                t._pin(item["evidence"]["path"],item["evidence"])
                loaded=t.read_sealed_evidence(item["evidence"]["path"],max_record_bytes=execution["max_record_bytes"],
                    max_logical_bytes=execution["max_logical_bytes"]-logical,max_records=execution["max_records"]-records,
                    wall_s=deadline-time.perf_counter(),max_output_bytes=execution["max_retained_bytes"]-retained,
                    rss_bytes=execution["rss_bytes"],require_admission=True)
                require(loaded["spec"].sha256==spec.sha256 and loaded["manifest"]["case_id"]==key
                        and t.digest(loaded["labels"])==t.digest(labels), "comparison_case_gate_and_labels")
                logical+=loaded["history"]["logical_bytes"];retained+=loaded["retained_bytes"]
                records+=loaded["history"]["records"]
                configured=next(c for c in plan["cases"] if c["id"]==key)
                one=case_from_evidence(configured,loaded,item["readback"],manifest["producer_run_plan"])
                cases[key],data[key]=one,loaded
            except (ValueError,KeyError,TypeError,OSError,EOFError) as error:
                failures.append({"case":key,"reason":str(error)[:500]})
        checkpoint()
        reference=None
        if manifest.get("strict_reference") is not None:
            try:reference=t._pin(manifest["strict_reference"]["path"],manifest["strict_reference"])
            except (ValueError,KeyError,TypeError,OSError) as error:
                failures.append({"case":"Strict_reference_acceptance","reason":str(error)[:500]})
        BE=BE_errors=None
        if manifest.get("BE") is not None:
            try:
                require(manifest["BE"]["identity"]["method_contract_sha256"]==method_pin["sha256"], "BE_frozen_method")
                BE=read_be_case(manifest["BE"],labels,dict(execution,max_logical_bytes=execution["max_logical_bytes"]-logical),checkpoint,be_layout)
                logical+=BE["readback"]["observed_history"]["logical_bytes"]
                records+=BE["readback"]["records"]
                retained+=len(integer_json_dumps(BE,t.INTEGER_POLICY).encode())
                require(records<=execution["max_records"] and retained<=execution["max_retained_bytes"],
                        "comparison_aggregate_records_or_retained_cap")
                descriptor=manifest["BE_error"]
                BE_errors=read_error_records(descriptor["result"],descriptor.get("acceptance"),
                    method_pin=method_pin,problem_identity=BE["reference_error_problem_identity"],labels=labels,checkpoint=checkpoint)
            except (ValueError,KeyError,TypeError,OSError,EOFError) as error:
                failures.append({"case":"BE_N8_L2_Full","reason":str(error)[:500]})
        else:
            failures.append({"case":"BE_N8_L2_Full","reason":"missing_BE_reference"})
        bindings={"status":"checked_source_bound_inputs","plan_sha256":context["plan_sha256"],
            "manifest":execution["manifest"],"producer_run_plan":manifest["producer_run_plan"],
            "gate_spec":gate_pin,"method_contract":method_pin,"physical_context":gate["physical_context"],
            "evaluation_source_review":plan.get("evaluation_freeze",{}).get("review_record"),
            "case_inputs":{key:{k:v for k,v in one.items() if k not in {"samples","readback"}} for key,one in cases.items()},
            "readbacks":{key:manifest["cases"][key]["readback"] for key in cases},
            "original_native_histories_opened_by_compare":False,"logical_observation_bytes":logical,
            "observation_records":records,"retained_evidence_bytes":retained}
        result=compare_loaded(cases,data,spec=spec,method_pin=method_pin,labels=labels,reference=reference,
            BE=BE,BE_errors=BE_errors,route_evidence=manifest.get("route_evidence"),bindings=bindings,
            failures=failures,checkpoint=checkpoint)
        checkpoint()
        require(len(integer_json_dumps(result,t.INTEGER_POLICY,indent=2,allow_nan=False).encode())+1<=execution["max_output_bytes"],
                "comparison_output_cap")
        return result
