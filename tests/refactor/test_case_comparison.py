"""Synthetic supplied-data checks only; none of these records qualifies a device."""
import copy
from fractions import Fraction as F
import json
import math
from pathlib import Path
import unittest

from case_comparison import (
    LOWER, compare_case, el_log_domain, strict_phase, validate_case_definition,
)
from physical_comparison import definition_fingerprint
from test_physical_comparison import fixture
from test_reference_comparison import example


def definitions():
    root = Path(__file__).resolve().parents[2]
    data = json.loads((root / "reproducibility/RefactorComparisonScopesV1.json").read_text())
    return {p["case_definition"]["kind"]: copy.deepcopy(p["case_definition"])
            for row in data["records"] for p in row.get("parameter_bindings", [])
            if "case_definition" in p}


def error(bound, unit):
    return {"uncertainty": {"bound": bound, "unit": unit, "evidence_sha256": "a"*64}}


def seal(triple):
    a, b, g = triple
    digest = definition_fingerprint(g)
    for record in triple:
        record["definition_sha256"] = digest
    return triple


def bind(triple, d, context):
    for item in triple:
        item.update(case_id=d["case_id"], case_context=context["id"],
                    case_definition_sha256=definition_fingerprint(d))
    return seal(triple)


def scalar(d, c, name, value, rule, bound=0):
    a, b, g = example(values=[value])
    for obj in (a, b, g):
        obj.update(scope=d["scope_id"], quantity=name, unit=rule["unit"])
    for obj in (a, b):
        obj.update(error(bound, rule["unit"]))
    g.update(atol=rule["atol"], rtol=rule["rtol"], reference_qualified=True,
             reference_error_abs=bound, reference_error_evidence="a"*64)
    return bind((a, b, g), d, c)


def physical(d, c, name, kind, values, rules, scales=None, **kw):
    thresholds = {k: {**v, "scale": 1 if scales is None else scales[k]} for k, v in rules.items()}
    return bind(fixture(kind, values, thresholds, scope=d["scope_id"], quantity=name,
                        comparison_target={"kind": "fixed_discrete", "id": d["case_id"]}, **kw), d, c)


def twod_fixture(d):
    supplied = {"protocol": d["protocol"], "contexts": {}}
    for c in d["contexts"]:
        channels = {}
        nx, ny = c["horizontal"]+1, 2*c["vertical"]+1
        # Manufactured supplied coordinates, not a generated native mesh.
        grid = {"shape": [ny,nx], "node_order": "carrier_local:j*Nx+i",
                "x_m": [i/c["horizontal"]*3e-7 for i in range(nx)],
                "y_m": [j/(2*c["vertical"])*2e-7 for j in range(ny)],
                "interface_y_faces": [c["vertical"]-1]}
        grids = {side: copy.deepcopy(grid) for side in ("reference", "candidate")}
        for component in d["components"]:
            carrier = component.split(":")[0]
            value = d["rules"][carrier+"_trace"]["input_scale"]
            volume = 3e-14
            a, b, g = physical(d, c, "cells:"+component, "cells", [value],
                {"l1": d["rules"][carrier+"_l1"], "inventory:"+component: d["rules"][carrier+"_inventory"]},
                unit="m-3", mask="all_cells_and_both_interface_sides", zero_policy="absolute_integrals_no_floor",
                geometry_dimension=2, area_unit="m2", unit_depth_m=1, volume_unit="m3", integral_unit="1",
                reference_volumes=[volume], candidate_volumes=[volume/2]*2,
                reference_components=[component], candidate_components=[component]*2,
                overlap=[[volume/2]*2], interface_ids=[],
                projection_evidence={"status": "approved", "sha256": "a"*64, "geometry_verified": True})
            for table, values, volumes in ((a, [value], [volume]), (b, [value]*2, [volume/2]*2)):
                table.update(values=values, volumes=volumes, areas_m2=volumes, components=[component]*len(values),
                             representation="cell_average", traces={})
            channels["cells:"+component] = (a, b, g)
            channels["density:"+component] = physical(d, c, "density:"+component, "density", [value]*2,
                {"log_rms": d["rules"]["log"], "log_max": d["rules"]["log"],
                 "zero_density_max": d["rules"]["zero"], "inventory:"+component: d["rules"][carrier+"_inventory"]},
                unit="m-3", mask=[True]*2, zero_policy="explicit_zero_cells_no_floor",
                volumes=[volume/2]*2, volume_unit="m3", inventory_unit="1", components=[component]*2, zero_cells=[])
        for carrier in ("n", "p"):
            for side in ("left", "right"):
                for i in range(c["horizontal"]+1):
                    name = f"trace:{carrier}:{side}:{i}"
                    triple = scalar(d, c, name, d["rules"][carrier+"_trace"]["input_scale"], d["rules"][carrier+"_trace"])
                    support = {"carrier": carrier, "side": side, "lateral_index": i,
                               "material_component": carrier+":layer_chi"+("4.0" if side == "left" else "3.8"),
                               "representation": "source_stencil_side",
                               "geometry_verified": True, "evidence_sha256": "a"*64,
                               "meaning": "synthetic distinct one-sided support"}
                    for key in ("reference", "candidate"):
                        row = c["vertical"]-1 if side == "left" else c["vertical"]
                        support[key] = {"state_index": row*nx+i, "electrical_face": c["vertical"]-1,
                                        "position_m": [grid["x_m"][i], grid["y_m"][row]]}
                    for record in triple:
                        record["interface_support"] = copy.deepcopy(support)
                    channels[name] = seal(triple)
        for triple in channels.values():
            for record in triple:
                record["context_grid"] = copy.deepcopy(grids)
            seal(triple)
        supplied["contexts"][c["id"]] = {"channels": channels, "context_grid": grids, "zero_ions": {
            side: {"input_population": 0, "input_background": 0, "state_population": 0}
            for side in ("reference", "candidate")}}
    return supplied


def lockin_fixture(d):
    supplied = {"protocol": d["protocol"], "contexts": {}}
    for c in d["contexts"]:
        channels = {}
        for name, value in (("I", 1.0), ("Z", d["delta_V"])):
            rule = d["rules"][name]
            triple = physical(d, c, name, "complex", [[value, 0.0]],
                {"real": rule, "imag": rule, "difference": rule,
                 "phase": {"atol": d["phase_limits"][c["phase_kind"]], "rtol": 0, "unit": "rad", "basis": "original"}},
                {"real": value, "imag": 0, "difference": value, "phase": 1},
                unit=rule["unit"], mask="all_frequencies", coordinates=[c["frequency"]],
                zero_policy="explicit_zero_else_resolved_phase", zero_indices=[], phase_floor=d["floors"][name],
                phase_floor_evidence={"value": d["floors"][name], "sha256": "a"*64})
            for table in triple[:2]:
                table.update(error(0, rule["unit"]), phase_error=error(0, "rad"), phase_rounding=error(0, "rad"))
            channels[name] = triple
        supplied["contexts"][c["id"]] = {"channels": channels, "resolution_levels": [20, 40, 80],
            "voltage_error": [error(0, "V") for _ in range(2)],
            "division_error": [error(0, "ohm m2") for _ in range(2)]}
    return supplied


def eqe_fixture(d):
    c = d["contexts"][0]; channels = {}; raws = [{"light": {}, "dark": {"value": .25, **error(1e-10, "A m-2")},
              "subtraction_error": {}, "division_error": {}} for _ in range(2)]
    for w, current in zip(d["wavelengths_nm"], [-1., 0., 1., 2., 3.]):
        channels[f"EQE:{w}"] = scalar(d, c, f"EQE:{w}", float(F(current)/F(d["current_scale_exact"])), d["rules"]["EQE"], 1e-12)
        channels[f"photo_current:{w}"] = scalar(d, c, f"photo_current:{w}", current, d["rules"]["current"], 3e-10)
        for raw in raws:
            raw["light"][str(w)] = {"value": current+.25, **error(1e-10, "A m-2")}
            raw["subtraction_error"][str(w)] = error(0, "A m-2")
            raw["division_error"][str(w)] = error(1e-17, "1")
    return {"protocol": d["protocol"], "contexts": {"fixture": {"channels": channels, "subtraction": raws}}}


def el_fixture(d):
    c = d["contexts"][0]; channels = {}; spectrum = {}
    for w in d["wavelengths_nm"]:
        rule = d["rules"][f"spectrum:{w}"]; spectrum[w] = rule["input_scale"]*.25
        channels[f"spectrum:{w}"] = scalar(d, c, f"spectrum:{w}", spectrum[w], rule)
        channels[f"absorptance:{w}"] = scalar(d, c, f"absorptance:{w}", .5, d["rules"]["absorptance"])
    emission = float(F(d["charge_exact_C"])*sum(F(w)*F(spectrum[l]) for l, w in zip(d["wavelengths_nm"], d["weights_nm"])))
    loss = -float(F(d["thermal_voltage_exact_V"]))*math.log(.5)
    for name, value, bound in (("J_em_rad", emission, 1e-12), ("J_inj", -2*emission, 1e-12),
                               ("EQE_EL", .5, 1e-11), ("delta_V_nr", loss, 1e-10)):
        channels[name] = scalar(d, c, name, value, d["rules"][name], bound)
    raw = {"spectrum_per_m": {str(w): spectrum[w]*1e9 for w in spectrum},
           "raw_absorptance": {str(w): .5 for w in spectrum},
           "conversion_error": {str(w): error(2*math.ulp(spectrum[w]), "photons m-2 s-1 nm-1") for w in spectrum},
           "integration_error": error(1e-14, "A m-2"), "sentinel": False, "lower_clip": False,
           "upper_clip": False, "clamped_ratio": .5, "ratio_error": error(1e-16, "1"),
           "log_error": error(1e-16, "V"), "loss_conversion_error": error(1e-16, "V"), "loss_mV": loss*1000}
    return {"protocol": d["protocol"], "contexts": {"fixture": {"channels": channels, "raw": [copy.deepcopy(raw), copy.deepcopy(raw)]}}}


def set_el_branch(d, supplied, side, ratio=.5, signed=None):
    block = supplied["contexts"]["fixture"]; channels = block["channels"]; raw = block["raw"][side]
    emission = channels["J_em_rad"][side]["points"][0]["value"]
    signed = -emission/ratio if signed is None else signed
    sentinel = abs(F(signed)) < LOWER
    ratio = 0.0 if sentinel else emission/abs(signed)
    clamped = None if sentinel else min(max(F(ratio), LOWER), F(1))
    loss = 0.0 if sentinel else -float(F(d["thermal_voltage_exact_V"]))*math.log(float(clamped))
    for name, value in (("J_inj", signed), ("EQE_EL", ratio), ("delta_V_nr", loss)):
        channels[name][side]["points"][0]["value"] = value
    raw.update(sentinel=sentinel, lower_clip=not sentinel and F(ratio)<LOWER,
               upper_clip=not sentinel and ratio>1, clamped_ratio=None if sentinel else float(clamped),
               loss_mV=loss*1000, ratio_error=error(abs(ratio)*1e-14, "1"))


class CaseComparisonTests(unittest.TestCase):
    def setUp(self):
        self.defs = definitions()

    def test_complete_synthetic_packets_pass_without_qualification(self):
        for kind, make in (("twod", twod_fixture), ("lockin", lockin_fixture), ("el", el_fixture), ("eqe", eqe_fixture)):
            with self.subTest(kind=kind):
                d = self.defs[kind]; self.assertTrue(validate_case_definition(d))
                data = make(d); before = copy.deepcopy(data); result = compare_case(d, data)
                self.assertTrue(result["accepted"], result)
                self.assertFalse(result["scientific_qualification_granted"])
                self.assertEqual(data, before)

    def test_missing_component_interface_side_frequency_and_wavelength_fail(self):
        for kind, make, missing in (("twod", twod_fixture, "cells:n:layer_chi3.8"),
                ("twod", twod_fixture, "trace:p:right:0"), ("lockin", lockin_fixture, "I"),
                ("eqe", eqe_fixture, "EQE:800"), ("el", el_fixture, "spectrum:900")):
            with self.subTest(kind=kind, missing=missing):
                d = self.defs[kind]; data = make(d)
                del data["contexts"][d["contexts"][0]["id"]]["channels"][missing]
                self.assertEqual(compare_case(d, data)["code"], "case_coverage")
        d = self.defs["lockin"]; data = lockin_fixture(d)
        del data["contexts"][d["contexts"][-1]["id"]]
        self.assertEqual(compare_case(d, data)["code"], "case_coverage")

    def test_empty_complex_vectors_reject_and_retain_failed_observations(self):
        d = self.defs["lockin"]
        for channel in ("I", "Z"):
            for side in (0, 1):
                with self.subTest(channel=channel, side=side):
                    data = lockin_fixture(d)
                    first = data["contexts"][d["contexts"][0]["id"]]
                    first["channels"][channel][side]["values"] = []
                    original = copy.deepcopy(data)
                    result = compare_case(d, data)
                    self.assertFalse(result["accepted"])
                    self.assertIn(result["status"], ("rejected", "unresolved"))
                    self.assertFalse(result["scientific_qualification_granted"])
                    self.assertEqual(result["raw_observations"], original)
                    self.assertEqual(data, original)

    def test_channels_cannot_be_swapped_or_unreviewed_or_missing_error(self):
        d = self.defs["eqe"]
        for fault in ("swap", "identity", "review", "error", "qualified", "posthoc"):
            data = eqe_fixture(d); channels = data["contexts"]["fixture"]["channels"]
            a, b, g = channels["EQE:400"]
            if fault == "swap": channels["EQE:400"] = channels["EQE:500"]
            elif fault == "identity": b["identity"]["input_sha256"] = "b"*64
            elif fault == "review": g["independent_review"]["status"] = "pending"
            elif fault == "error": g["reference_error_abs"] = None
            elif fault == "qualified": g["reference_qualified"] = False
            else: g["frozen_at"] = "2026-10-04T00:00:00Z"
            seal((a, b, g))
            self.assertFalse(compare_case(d, data)["accepted"], fault)

    def test_twod_component_integrals_and_one_sided_support_are_not_averaged(self):
        d = self.defs["twod"]; data = twod_fixture(d); c = d["contexts"][0]["id"]
        triple = data["contexts"][c]["channels"]["cells:n:layer_chi4.0"]
        triple[1]["values"] = [triple[0]["values"][0]*1.01, triple[0]["values"][0]*.99]
        result = compare_case(d, data)
        self.assertFalse(result["checks"][c]["cells:n:layer_chi4.0"]["metrics"]["l1"]["passed"])
        data = twod_fixture(d); channels = data["contexts"][c]["channels"]
        channels["trace:n:left:0"][1]["interface_support"]["side"] = "right"
        self.assertEqual(compare_case(d, data)["code"], "interface")
        data = twod_fixture(d)
        data["contexts"][c]["zero_ions"]["candidate"]["state_population"] = 1e-300
        self.assertEqual(compare_case(d, data)["code"], "zero_species")

    def test_twod_trace_indices_faces_and_coordinates_match_actual_context(self):
        d = self.defs["twod"]
        for source_side in ("reference", "candidate"):
            for fault in ("huge_indices", "wrong_in_range_node", "wrong_face", "wrong_coordinate", "wrong_column"):
                with self.subTest(source_side=source_side, fault=fault):
                    data = twod_fixture(d)
                    c = d["contexts"][0]
                    triple = data["contexts"][c["id"]]["channels"]["trace:n:left:0"]
                    for record in triple:
                        location = record["interface_support"][source_side]
                        if fault == "huge_indices":
                            location.update(state_index=10**9, electrical_face=10**9)
                        elif fault == "wrong_in_range_node": location["state_index"] -= c["horizontal"]+1
                        elif fault == "wrong_face": location["electrical_face"] -= 1
                        elif fault == "wrong_column": location["state_index"] += 1
                        else: location["position_m"][1] *= .99
                    seal(triple)
                    original = copy.deepcopy(data)
                    result = compare_case(d, data)
                    self.assertFalse(result["accepted"])
                    self.assertEqual(result["code"], "interface")
                    self.assertEqual(result["raw_observations"], original)
                    self.assertFalse(result["scientific_qualification_granted"])
        data = twod_fixture(d); block = data["contexts"][d["contexts"][0]["id"]]
        block["context_grid"]["reference"]["shape"] = [65, 3]
        self.assertEqual(compare_case(d, data)["code"], "geometry")

    def test_strict_phase_equality_and_both_side_errors(self):
        for value in self.defs["lockin"]["phase_limits"].values():
            theta = F(value)
            self.assertFalse(strict_phase(theta, F(0), F(0), theta))
            self.assertFalse(strict_phase(theta*2/3, theta/6, theta/6, theta))
            self.assertTrue(strict_phase(F(math.nextafter(value, 0)), F(0), F(0), theta))
        d = self.defs["lockin"]
        for side in (0, 1):
            data = lockin_fixture(d); block = data["contexts"][d["contexts"][0]["id"]]
            block["channels"]["Z"][side]["phase_error"]["uncertainty"]["bound"] = None
            self.assertEqual(compare_case(d, data)["status"], "unresolved")
        data = lockin_fixture(d); block = data["contexts"][d["contexts"][0]["id"]]
        block["channels"]["Z"][1]["phase_rounding"]["uncertainty"]["bound"] = 1e-5
        self.assertEqual(compare_case(d, data)["code"], "unresolved_phase")

    def test_eqe_negative_zero_and_both_light_dark_errors_are_retained(self):
        d = self.defs["eqe"]; data = eqe_fixture(d)
        result = compare_case(d, data); self.assertTrue(result["accepted"], result)
        checks = result["checks"]["fixture"]
        self.assertLess(checks["sign_resolution:400:1"]["signed_value"], 0)
        self.assertFalse(checks["sign_resolution:500:1"]["resolved"])
        for side in (0, 1):
            for name in ("light", "dark"):
                data = eqe_fixture(d); raw = data["contexts"]["fixture"]["subtraction"][side]
                item = raw["light"]["400"] if name == "light" else raw["dark"]
                item["uncertainty"]["bound"] = None
                self.assertEqual(compare_case(d, data)["code"], "unresolved_error")
        data = eqe_fixture(d); data["contexts"]["fixture"]["subtraction"][1]["dark"]["uncertainty"]["bound"] = 1e-4
        self.assertEqual(compare_case(d, data)["code"], "unresolved_error")
        data = eqe_fixture(d); data["contexts"]["fixture"]["channels"]["EQE:400"][1]["points"][0]["value"] *= -1
        self.assertFalse(compare_case(d, data)["checks"]["fixture"]["EQE_units:400:1"]["accepted"])

    def test_el_nm_m_and_independent_emission_gate(self):
        d = self.defs["el"]; data = el_fixture(d)
        data["contexts"]["fixture"]["raw"][1]["spectrum_per_m"]["900"] /= 1e9
        self.assertFalse(compare_case(d, data)["checks"]["fixture"]["nm_m:900:1"]["accepted"])
        data = el_fixture(d); data["contexts"]["fixture"]["channels"]["J_em_rad"][1]["points"][0]["value"] *= 1.001
        result = compare_case(d, data)
        self.assertFalse(result["checks"]["fixture"]["J_em_rad"]["accepted"])
        self.assertTrue(all(result["checks"]["fixture"][f"spectrum:{w}"]["accepted"] for w in d["wavelengths_nm"]))
        slack = F(d["rules"]["J_em_rad"]["atol"])-F(d["charge_exact_C"])*sum(F(w)*F(d["rules"][f"spectrum:{l}"]["atol"]) for l,w in zip(d["wavelengths_nm"],d["weights_nm"]))
        self.assertEqual(slack, F(d["rounded_spectral_sum_slack_A_m2_exact"]))
        self.assertLess(slack, 0)

    def test_el_entire_below_floor_crossing_clamps_and_exact_boundaries(self):
        for emission, signed, be, bi in ((LOWER/100, -F(1), LOWER/1000, F(0)),
                (LOWER, -F(1), LOWER/10, F(0)), (F(1), -F(1), F(1, 100), F(0)),
                (F(1), -LOWER/2, F(0), F(0)), (F(1, 2), F(1), F(0), F(0))):
            self.assertIsNone(el_log_domain(emission, signed, be, bi, F(0)))
        self.assertEqual(el_log_domain(LOWER, -F(1), F(0), F(0), F(0)), (LOWER, LOWER))
        self.assertEqual(el_log_domain(F(1), -F(1), F(0), F(0), F(0)), (F(1), F(1)))
        d = self.defs["el"]
        for side in (0, 1):
            for branch in ("below", "lower_cross", "upper_cross", "sentinel", "non_injection"):
                with self.subTest(side=side, branch=branch):
                    data = el_fixture(d)
                    if branch == "below": set_el_branch(d, data, side, ratio=float(LOWER)/100)
                    elif branch == "lower_cross": set_el_branch(d, data, side, ratio=float(LOWER)*(1+1e-13))
                    elif branch == "upper_cross": set_el_branch(d, data, side, ratio=1-1e-13)
                    elif branch == "sentinel": set_el_branch(d, data, side, signed=-float(LOWER)/2)
                    else: set_el_branch(d, data, side, signed=1.0)
                    before = copy.deepcopy(data); result = compare_case(d, data)
                    self.assertEqual(result.get("code"), "unresolved_EL_log_domain", result)
                    self.assertIn(side, result["checks"]["fixture"]["physical_unclipped_log"]["sides"])
                    self.assertEqual(result["raw_observations"], before)
                    self.assertEqual(data, before)

    def test_el_flags_and_missing_arithmetic_errors_do_not_pass(self):
        d = self.defs["el"]; data = el_fixture(d)
        data["contexts"]["fixture"]["raw"][0]["lower_clip"] = True
        self.assertEqual(compare_case(d, data)["code"], "source_branch")
        for key in ("integration_error", "ratio_error", "log_error", "loss_conversion_error"):
            data = el_fixture(d); data["contexts"]["fixture"]["raw"][1][key]["uncertainty"]["bound"] = None
            self.assertEqual(compare_case(d, data)["code"], "unresolved_error")


if __name__ == "__main__":
    unittest.main()
