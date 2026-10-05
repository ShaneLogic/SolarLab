"""SCAPS case coverage and exact input targeting, without a physics solve."""
from copy import deepcopy
from decimal import Decimal
import json
from pathlib import Path

import pytest
import yaml

from scripts.benchmarks import scaps_reference as sr


REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def inventory():
    manifest = json.loads((REPO / sr.MANIFEST).read_text())
    contract = json.loads((REPO / sr.CONTRACT).read_text())
    assert sr.file_digest(REPO / sr.CONTRACT) == sr.CONTRACT_SHA256
    assert sr.digest(manifest["scaps"]["groups"]) == contract["reference_set"]["manifest_groups_payload_sha256"]
    return manifest["scaps"]["groups"]


@pytest.fixture(scope="module")
def template():
    return yaml.safe_load((REPO / sr.TEMPLATE).read_text())


def named(items, name, key="name"):
    return next(item for item in items if item[key] == name)


def differences(before, after, path=()):
    if isinstance(before, dict) and isinstance(after, dict):
        assert before.keys() == after.keys()
        return [change for key in before for change in differences(before[key], after[key], (*path, key))]
    if isinstance(before, list) and isinstance(after, list):
        assert len(before) == len(after)
        return [change for i, (left, right) in enumerate(zip(before, after, strict=True))
                for change in differences(left, right, (*path, i))]
    return [] if type(before) is type(after) and before == after else [path]


def test_every_reported_coordinate_and_missing_row_is_retained(template, inventory):
    original = sr.digest(template)
    cases = sr.build_cases(template, inventory)
    assert len(cases) == 262
    assert len({case["id"] for case in cases}) == 262
    assert sum(case["reference_status"] == "ok" for case in cases) == 247
    assert sum(case["reference_status"] == "reference_missing" for case in cases) == 14
    assert sum(case["group_id"] == "scaps_pvk_donor" for case in cases) == 6
    assert sum(case["group_id"] == "scaps_nt_et" for case in cases) == 56
    assert sum(case["group_id"] == "scaps_nt_cbo" for case in cases) == 112
    for group in inventory:
        expected = {tuple(Decimal(str(v)) for v in point["coordinates"]) for point in group["points"]}
        actual = {tuple(Decimal(str(v)) for v in case["coordinates"])
                  for case in cases if case["group_id"] == group["id"]}
        assert actual == expected
    assert sr.digest(template) == original


@pytest.mark.parametrize("group,target,field,value", [
    ("scaps_nt_pvk_cb", "Perovskite-CB", "N_t_cm3", 1e9),
    ("scaps_nt_pvk_vb", "Perovskite-VB", "N_t_cm3", 1e15),
    ("scaps_et_pvk_cb", "Perovskite-CB", "E_t_eV_below_cb", 0.05),
    ("scaps_et_pvk_vb", "Perovskite-VB", "E_t_eV_above_vb", 0.6),
])
def test_bulk_scans_change_one_named_defect_only(template, group, target, field, value):
    result, edits = sr.case_document(template, group, [value])
    defects = named(result["layers"], "SCAPS_PVK")["bulk_defects"]
    assert named(defects, target)[field] == value
    assert len(defects) == 2
    assert len(differences(template, result)) == 1
    assert len(edits) == 1 and edits[0]["selector"] == target
    assert edits[0]["field"] == field
    other = "Perovskite-VB" if target == "Perovskite-CB" else "Perovskite-CB"
    assert named(defects, other) == named(named(template["layers"], "SCAPS_PVK")["bulk_defects"], other)


@pytest.mark.parametrize("group,target", [("scaps_etl_donor", "SCAPS_ETL"),
                                         ("scaps_pvk_donor", "SCAPS_PVK")])
def test_donor_scans_preserve_other_doping_and_layers(template, group, target):
    result, edits = sr.case_document(template, group, [1e10])
    assert named(result["layers"], target)["N_D_cm3"] == 1e10
    assert named(result["layers"], target)["N_A_cm3"] == named(template["layers"], target)["N_A_cm3"]
    assert len(differences(template, result)) == 1
    assert edits[0]["effective_unit"] == "cm^-3"


@pytest.mark.parametrize("group,target,field,value", [
    ("scaps_nt_htl_pvk", "HTL/PVK", "N_t_cm2", 1e9),
    ("scaps_nt_pvk_etl", "PVK/ETL", "N_t_cm2", 1e15),
    ("scaps_et_htl_pvk", "HTL/PVK", "E_t_eV_below_cb", 0.05),
    ("scaps_et_pvk_etl", "PVK/ETL", "E_t_eV_below_cb", 0.4),
])
def test_interface_scans_preserve_all_other_parameters(template, group, target, field, value):
    result, edits = sr.case_document(template, group, [value])
    assert named(result["interfaces"], target, "target")[field] == value
    assert len(differences(template, result)) == 1
    assert edits[0]["container"] == "interface"
    assert result["device"] == template["device"]


@pytest.mark.parametrize("cbo", [-1, -0.2, -0.16, 0, 0.5, 0.7])
def test_cbo_uses_reported_coordinate_and_pvk_affinity(template, cbo):
    result, edits = sr.case_document(template, "scaps_cbo", [cbo])
    expected = float(Decimal("3.94") - Decimal(str(cbo)))
    assert named(result["layers"], "SCAPS_ETL")["chi_eV"] == expected
    assert named(result["layers"], "SCAPS_PVK") == named(template["layers"], "SCAPS_PVK")
    assert edits[0]["axis"] == "CBO" and edits[0]["value"] == expected


@pytest.mark.parametrize("group,coordinates", [("scaps_nt_et", [1e9, 0.1]),
                                               ("scaps_nt_cbo", [1e9, -0.2])])
def test_pair_scans_compose_two_targeted_changes(template, group, coordinates):
    result, edits = sr.case_document(template, group, coordinates)
    assert len(differences(template, result)) == 2
    assert len(edits) == 2
    assert named(result["interfaces"], "PVK/ETL", "target")["N_t_cm2"] == 1e9


def test_case_roundtrip_keeps_reference_answers_out_of_solver_inputs(template, inventory):
    plan = {"template": template, "template_document_sha256": sr.digest(template),
            "cases": sr.build_cases(template, inventory)}
    reviewed_identity = sr.digest(plan)
    plan["identity_sha256"] = reviewed_identity
    restored = json.loads(json.dumps(plan))
    for case in restored["cases"]:
        document = sr.materialize_case(restored, case["id"], expected_plan_identity=reviewed_identity)
        assert sr.digest(document) == case["input_document_sha256"]
        assert "values" not in case
        assert all(metric not in json.dumps(case) for metric in
                   ("Voc_V", "Jsc_mA_cm2", "FF_percent", "PCE_percent"))
    # Full execution identity has not been established, so aliases stay present.
    assert len({case["input_document_sha256"] for case in restored["cases"]}) < len(restored["cases"])


def test_effective_areal_hypothesis_is_explicit(template, inventory):
    cases = sr.build_cases(template, inventory)
    for case in cases:
        for edit in case["updates"]:
            if edit["container"] == "interface" and edit["axis"] == "Nt":
                assert edit["reported_coordinate_unit"] == "cm^-3"
                assert edit["effective_unit"] == "cm^-2"
                assert "no thickness conversion" in edit["unit_interpretation"]


@pytest.mark.parametrize("bad", [True, "1e9", float("nan"), float("inf"), -1])
def test_invalid_donor_value_is_rejected(template, bad):
    with pytest.raises(sr.ReferenceError):
        sr.case_document(template, "scaps_pvk_donor", [bad])


@pytest.mark.parametrize("mutation", ["missing_group", "duplicate_group", "duplicate_coordinate", "axis_order", "missing_point", "unknown_status"])
def test_incomplete_or_ambiguous_reference_coverage_is_rejected(template, inventory, mutation):
    groups = deepcopy(inventory)
    if mutation == "missing_group":
        groups.pop()
    elif mutation == "duplicate_group":
        groups.append(deepcopy(groups[0]))
    elif mutation == "duplicate_coordinate":
        groups[0]["points"][1]["coordinates"] = groups[0]["points"][0]["coordinates"]
    elif mutation == "axis_order":
        groups[-1]["axes"].reverse()
    elif mutation == "missing_point":
        groups[0]["points"].pop()
    else:
        groups[0]["points"][0]["status"] = "filled_from_pixels"
    with pytest.raises(sr.ReferenceError):
        sr.build_cases(template, groups)


@pytest.mark.parametrize("mutation", ["duplicate_target", "missing_field", "renamed_defect"])
def test_template_drift_cannot_retarget_a_defect(template, mutation):
    changed = deepcopy(template)
    defects = named(changed["layers"], "SCAPS_PVK")["bulk_defects"]
    if mutation == "duplicate_target":
        defects.append(deepcopy(defects[0]))
    elif mutation == "missing_field":
        del defects[0]["N_t_cm3"]
    else:
        defects[0]["name"] = "unknown"
    with pytest.raises(sr.ReferenceError):
        sr.case_document(changed, "scaps_nt_pvk_cb", [1e9])


@pytest.mark.parametrize("mutation", ["template", "coordinate", "annotation", "input_hash"])
def test_materialization_checks_template_coordinate_and_display_binding(template, inventory, mutation):
    plan = {"template": deepcopy(template), "template_document_sha256": sr.digest(template),
            "cases": sr.build_cases(template, inventory)}
    reviewed_identity = sr.digest(plan)
    plan["identity_sha256"] = reviewed_identity
    case = next(case for case in plan["cases"] if case["group_id"] == "scaps_pvk_donor")
    if mutation == "template":
        plan["template"]["device"]["V_bi"] = 2.0
    elif mutation == "coordinate":
        case["coordinates"][0] = 3e9
    elif mutation == "annotation":
        case["updates"][0]["selector"] = "SCAPS_ETL"
    else:
        case["input_document_sha256"] = "0" * 64
    with pytest.raises(sr.ReferenceError):
        sr.materialize_case(plan, case["id"], expected_plan_identity=reviewed_identity)


@pytest.mark.parametrize("mutation", ["reported_unit", "interpretation", "axis_unit", "source_row", "status"])
@pytest.mark.parametrize("reseal", [False, True])
def test_materialization_binds_complete_provenance_to_reviewed_plan(template, inventory, mutation, reseal):
    plan = {"template": template, "template_document_sha256": sr.digest(template),
            "cases": sr.build_cases(template, inventory)}
    reviewed_identity = sr.digest(plan)
    plan["identity_sha256"] = reviewed_identity
    case = next(case for case in plan["cases"] if case["id"] == "scaps_nt_pvk_etl:row2")
    if mutation == "reported_unit":
        case["updates"][0]["reported_coordinate_unit"] = "m^-3"
    elif mutation == "interpretation":
        case["updates"][0]["unit_interpretation"] = "converted using an unreported thickness"
    elif mutation == "axis_unit":
        case["axes"][0]["unit_as_reported"] = "m^-3"
    elif mutation == "source_row":
        case["source"]["worksheet_row"] = 999999
    else:
        case["reference_status"] = "reference_missing"
    if reseal:
        plan["identity_sha256"] = sr.digest({key: value for key, value in plan.items()
                                            if key != "identity_sha256"})
    with pytest.raises(sr.ReferenceError, match="independently reviewed identity"):
        sr.materialize_case(plan, case["id"], expected_plan_identity=reviewed_identity)
