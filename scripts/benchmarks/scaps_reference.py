"""Prepare all reported SCAPS coordinates as explicit, source-bound inputs.

This preparation does not run a solver. It retains the historical mirror
template's declared parameters and targets each named defect separately.
The older full-scan script's combined CB/VB update is therefore not replayed.
An actual driver, effective physics, protocol and numerical qualification must
be reviewed before these inputs can be used for scientific acceptance.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any


CONTRACT = "reproducibility/ScapsComparisonContractV1.json"
CONTRACT_SHA256 = "5beaf9096866cd98896d0300425e107aa9323dfc3e7a1fd8d8ca5d1ceed2e3b5"
MANIFEST = "reproducibility/RefactorReferenceManifestV1.json"
TEMPLATE = "perovskite-sim/configs/scaps_mirror_v2.yaml"
CHANNEL = "historical_mirror_inputs_distinct_defect_sweeps_v1"

# axis, container, exact template selector, field, effective input unit.
# A reported interface numeral is used in the explicitly declared areal
# hypothesis, with no invented conversion thickness.
RULES = {
    "scaps_cbo": (("CBO", "layer", "SCAPS_ETL", "chi_eV", "eV"),),
    "scaps_etl_donor": (("N_D", "layer", "SCAPS_ETL", "N_D_cm3", "cm^-3"),),
    "scaps_pvk_donor": (("N_D", "layer", "SCAPS_PVK", "N_D_cm3", "cm^-3"),),
    "scaps_nt_htl_pvk": (("Nt", "interface", "HTL/PVK", "N_t_cm2", "cm^-2"),),
    "scaps_nt_pvk_cb": (("Nt", "bulk_defect", "Perovskite-CB", "N_t_cm3", "cm^-3"),),
    "scaps_nt_pvk_vb": (("Nt", "bulk_defect", "Perovskite-VB", "N_t_cm3", "cm^-3"),),
    "scaps_nt_pvk_etl": (("Nt", "interface", "PVK/ETL", "N_t_cm2", "cm^-2"),),
    "scaps_et_htl_pvk": (("Et", "interface", "HTL/PVK", "E_t_eV_below_cb", "eV"),),
    "scaps_et_pvk_cb": (("Et", "bulk_defect", "Perovskite-CB", "E_t_eV_below_cb", "eV"),),
    "scaps_et_pvk_vb": (("Et", "bulk_defect", "Perovskite-VB", "E_t_eV_above_vb", "eV"),),
    "scaps_et_pvk_etl": (("Et", "interface", "PVK/ETL", "E_t_eV_below_cb", "eV"),),
    "scaps_nt_et": (
        ("Nt", "interface", "PVK/ETL", "N_t_cm2", "cm^-2"),
        ("Et", "interface", "PVK/ETL", "E_t_eV_below_cb", "eV"),
    ),
    "scaps_nt_cbo": (
        ("Nt", "interface", "PVK/ETL", "N_t_cm2", "cm^-2"),
        ("CBO", "layer", "SCAPS_ETL", "chi_eV", "eV"),
    ),
}


class ReferenceError(ValueError):
    """The selected reference, input or coordinate is inconsistent."""


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _number(value: Any) -> int | float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ReferenceError("coordinate must be a finite number, not a boolean or string")
    return value


def _one(rows: Any, key: str, selector: str) -> dict:
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ReferenceError(f"invalid collection for {selector}")
    matches = [row for row in rows if row.get(key) == selector]
    if len(matches) != 1:
        raise ReferenceError(f"expected exactly one {key}={selector!r}")
    return matches[0]


def _target(document: dict, kind: str, selector: str) -> dict:
    if kind == "layer":
        return _one(document.get("layers"), "name", selector)
    if kind == "interface":
        return _one(document.get("interfaces"), "target", selector)
    if kind == "bulk_defect":
        absorber = _one(document.get("layers"), "name", "SCAPS_PVK")
        return _one(absorber.get("bulk_defects"), "name", selector)
    raise ReferenceError(f"unknown input container {kind!r}")


def case_document(template: dict, group_id: str, coordinates: list) -> tuple[dict, list[dict]]:
    """Build a fresh input document; never update a combined SRH lifetime."""
    if group_id == "scaps_base":
        if coordinates:
            raise ReferenceError("the base case has no scan coordinates")
        return deepcopy(template), []
    if group_id not in RULES:
        raise ReferenceError(f"unsupported SCAPS group {group_id!r}")
    rules = RULES[group_id]
    if not isinstance(coordinates, list) or len(coordinates) != len(rules):
        raise ReferenceError("coordinate dimension differs from the named scan")
    document, updates = deepcopy(template), []
    for raw, (axis, kind, selector, field, unit) in zip(coordinates, rules, strict=True):
        value = _number(raw)
        if axis in {"N_D", "Nt", "Et"} and value < 0:
            raise ReferenceError(f"negative {axis} is outside the declared scan")
        transform = "identity"
        if axis == "CBO":
            affinity = _number(_target(template, "layer", "SCAPS_PVK")["chi_eV"])
            value = float(Decimal(str(affinity)) - Decimal(str(value)))
            _number(value)
            transform = "chi_PVK_eV - CBO_eV; decimal input subtraction"
        target = _target(document, kind, selector)
        if field not in target:
            raise ReferenceError(f"missing declared field {selector}.{field}")
        old = target[field]
        target[field] = value
        updates.append({"axis": axis, "container": kind, "selector": selector,
                        "field": field, "old_value": old, "value": value,
                        "effective_unit": unit, "transform": transform})
    return document, updates


def build_cases(template: dict, groups: list[dict]) -> list[dict]:
    """Keep every source row, including missing references and input aliases."""
    group_ids = [group["id"] for group in groups]
    if len(group_ids) != len(set(group_ids)) or set(group_ids) != set(RULES):
        raise ReferenceError("the complete set of thirteen distinct scans is required")
    base, _ = case_document(template, "scaps_base", [])
    cases = [{"id": "scaps_base", "group_id": "scaps_base", "coordinates": [],
              "source": {"source_id": "scaps_pdf", "page": 2},
              "reference_status": "reported_rounded_base", "updates": [],
              "input_document_sha256": digest(base)}]
    for group in groups:
        group_id, points, axes = group["id"], group["points"], group["axes"]
        rules = RULES[group_id]
        if [axis["name"] for axis in axes] != [rule[0] for rule in rules]:
            raise ReferenceError(f"axis order differs for {group_id}")
        if len(points) != group["expected_points"]:
            raise ReferenceError(f"missing source coordinates in {group_id}")
        if "expected_shape" in group and math.prod(group["expected_shape"]) != len(points):
            raise ReferenceError(f"grid shape differs for {group_id}")
        seen_coordinates, seen_rows = set(), set()
        numeric = 0
        for point in points:
            coordinates = point["coordinates"]
            if not isinstance(coordinates, list):
                raise ReferenceError("coordinates must be a named-axis list")
            coordinate_key = tuple(Decimal(str(_number(v))) for v in coordinates)
            row = point["worksheet_row"]
            if type(row) is not int or row < 2 or row in seen_rows or coordinate_key in seen_coordinates:
                raise ReferenceError(f"duplicate or invalid source row in {group_id}")
            seen_rows.add(row)
            seen_coordinates.add(coordinate_key)
            status = point["status"]
            if status not in {"ok", "reference_missing"}:
                raise ReferenceError(f"unknown reference status {status!r}")
            numeric += status == "ok"
            document, updates = case_document(template, group_id, coordinates)
            for update, axis in zip(updates, axes, strict=True):
                update["reported_coordinate_unit"] = axis["unit_as_reported"]
                update["unit_interpretation"] = (
                    "declared historical areal-input hypothesis; no thickness conversion"
                    if update["effective_unit"] != axis["unit_as_reported"] else "same dimension")
            cases.append({"id": f"{group_id}:row{row}", "group_id": group_id,
                          "coordinates": deepcopy(coordinates), "axes": deepcopy(axes),
                          "source": {"source_id": group["source_id"], "worksheet": group["worksheet"],
                                     "worksheet_row": row, "report_page": group["pdf_page"]},
                          "reference_status": status, "updates": updates,
                          "input_document_sha256": digest(document)})
        if numeric != group["numeric_points"]:
            raise ReferenceError(f"numeric/missing reference count differs for {group_id}")
    return cases


def materialize_case(plan: dict, case_id: str, *, expected_plan_identity: str) -> dict:
    """Regenerate an input from a separately reviewed plan identity.

    The caller supplies the identity from its approved execution record, not
    from an unverified incoming plan. The complete payload binds provenance,
    units, display annotations and missing-reference status as well as inputs.
    """
    actual_identity = digest({key: value for key, value in plan.items()
                              if key != "identity_sha256"})
    if (actual_identity != plan.get("identity_sha256")
            or actual_identity != expected_plan_identity):
        raise ReferenceError("case plan differs from the independently reviewed identity")
    if digest(plan["template"]) != plan["template_document_sha256"]:
        raise ReferenceError("template content changed")
    case = _one(plan["cases"], "id", case_id)
    document, updates = case_document(plan["template"], case["group_id"], case["coordinates"])
    if digest(document) != case["input_document_sha256"]:
        raise ReferenceError("case input content changed")
    # Display annotations cannot silently disagree with the executed update.
    planned = [{key: update[key] for key in actual} for update, actual in
               zip(case["updates"], updates, strict=True)]
    if planned != updates:
        raise ReferenceError("case update annotations changed")
    return document


def build_plan(repo: Path, archive: Path) -> dict:
    import yaml

    def git(*args):
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()

    if git("branch", "--show-current") != "devel":
        raise ReferenceError("reference preparation requires devel")
    contract_path, manifest_path = repo / CONTRACT, repo / MANIFEST
    if file_digest(contract_path) != CONTRACT_SHA256:
        raise ReferenceError("comparison contract differs from the reviewed version")
    contract = json.loads(contract_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    groups = manifest["scaps"]["groups"]
    if digest(groups) != contract["reference_set"]["manifest_groups_payload_sha256"]:
        raise ReferenceError("reference coordinates differ from the reviewed source")
    if contract["independent_review"]["status"] != "approved" or not contract["frozen_at"]:
        raise ReferenceError("comparison standard is not frozen and independently reviewed")
    roots, bindings = {"repository": repo, "archive": archive}, []
    for source in contract["source_bindings"]:
        path = roots[source["root"]] / source["path"]
        actual = file_digest(path)
        if actual != source["sha256"]:
            raise ReferenceError(f"source changed: {source['path']}")
        bindings.append({"path": str(path), "sha256": actual, "role": source["role"]})
    template = yaml.safe_load((repo / TEMPLATE).read_text())
    cases = build_cases(template, groups)
    plan = {
        "schema": "solarlab.scaps-case-plan.v1", "channel": CHANNEL,
        "status": "prepared_pending_effective_input_and_execution_review",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "repository": {"branch": "devel", "head": git("rev-parse", "HEAD")},
        "source_bindings": bindings + [
            {"path": str(contract_path), "sha256": CONTRACT_SHA256, "role": "frozen_comparison_standard"},
            {"path": str(manifest_path), "sha256": file_digest(manifest_path), "role": "complete_reference_inventory"},
            {"path": str(Path(__file__).resolve()), "sha256": file_digest(Path(__file__)), "role": "case_preparation"}],
        "template": template, "template_document_sha256": digest(template), "cases": cases,
        "coverage": {"groups": len(groups), "scan_cases": len(cases) - 1, "base_cases": 1,
                     "numeric_reference_rows": sum(case["reference_status"] == "ok" for case in cases),
                     "missing_reference_rows": sum(case["reference_status"] == "reference_missing" for case in cases)},
        "interpretation": {
            "template": "Declared historical mirror_v2 input values retained; no new fit.",
            "bulk_defects": "Each CB/VB scan changes only its named raw defect field before the actual loader; both defects remain present.",
            "interface_density": "Reported cm^-3 labels and effective cm^-2 fields are both shown; the historical areal hypothesis is explicit and unconfirmed by native inputs.",
            "energy_reference": "Existing template references retained; actual driver meaning still needs capture. This is not the proposed lowest-neighbor-CB reconstruction.",
            "gaussian": "Keep existing total, width and reported peak fields. Historical loader behavior and the inconsistent redundant peak must be disclosed; no new Gaussian normalization.",
            "optics": "Keep the declared glass/optical stack; capture any declared-versus-effective loader differences before execution.",
            "identity": "An input-document hash alone does not authorize run reuse; full physics, protocol, source and environment identity are required.",
            "reference_values": "Metric values are absent from solver case recipes; the independent comparator reads the pinned reference separately.",
        },
        "execution_prerequisites": [
            "Select and freeze the actual driver, effective fields, contacts, statistics, optics, ions, environment and every fallback.",
            "Freeze preparation, bias path, output/metric definitions and independent mesh/bias/nonlinear error checks.",
            "Independently review this input channel and its disclosed differences from native SCAPS and the old combined-defect scan.",
            "Admit a bounded source-bound execution with explicit resource and failure limits.",
        ],
        "execution_authorized": False, "scientific_qualification_granted": False,
    }
    plan["identity_sha256"] = digest(plan)
    return plan


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("plan",))
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    plan = build_plan(args.repo.resolve(), args.archive.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(plan, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "identity": plan["identity_sha256"],
                      "coverage": plan["coverage"], "execution_authorized": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
