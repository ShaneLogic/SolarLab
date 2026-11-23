"""Native zero-fit SCAPS-1D 262-case benchmark driver (PVK-xy.scaps + ScapsAm15g1Sun.spe).

Evaluates the 262-case SCAPS-1D comparison matrix (1 base + 13 single/two-parameter
sweeps = 248 numeric reference points + 14 reference_missing points) using the
packaged ``scaps_pvk_xy.yaml`` zero-fit preset paired with the native SCAPS
120-bin AM1.5G spectrum (``reproducibility/ScapsAm15g1Sun.spe``) and SCAPS's
analytical ``sqrt(hv - Eg)`` layer absorption laws.

Engineering alignment and formal scientific qualification remain explicitly
separated (``scientific_qualification = False``).
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from datetime import UTC, datetime
from decimal import Decimal
import hashlib
import json
import math
import multiprocessing
from pathlib import Path
import statistics
import sys
import time
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.constants import c as SPEED_OF_LIGHT_M_S, e as ELEM_CHARGE_C, h as PLANCK_J_S
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT / "src"), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scripts.benchmarks.p09_candidate_validation import (  # noqa: E402
    MANIFEST_REL,
    SCAPS_CONTRACT_REL,
    SCAPS_CONTRACT_SHA256,
    _decimal_ulp,
    _file_sha256,
    _host_environment_record,
    _load_packaged_catalog_and_resources,
    _loaded_forbidden_legacy_modules,
    _prepare_scaps_yaml_device,
    _run_scaps_steady_jv_aligned,
)
from solarlab.device.compile_1d import compile_device_1d  # noqa: E402
from solarlab.materials import load_packaged_preset_sources  # noqa: E402
from solarlab.materials.resources import ResourceLibrary, ResourceTable  # noqa: E402
from solarlab.materials.source import SourceDocument  # noqa: E402
from solarlab.mesh import PHYSICAL_BOUNDARY_TWO_SIDED_V1  # noqa: E402
from solarlab.physics.interface_transport import (  # noqa: E402
    SCAPS_EFFECTIVE_VELOCITY_THERMIONIC_V1,
)
from solarlab.physics.recombination import (  # noqa: E402
    SCAPS_GAUSSIAN_INTERFACE_QUADRATURE,
    SCAPS_INTERFACE_EXTREMA_ANCHOR,
)
from solarlab.solvers.nonlinear.steady_1d import SteadyOptions  # noqa: E402

SCAPS_SPE_REL = "reproducibility/ScapsAm15g1Sun.spe"
SCAPS_SPE_SHA256 = "e357f1237fe82182e51b94426fe90bda2f248f7f1246084d42c46aad78cf03f2"

# Scan coordinate mapping faithful to PVK-xy.scaps (Reference=3: above highest EV)
PVK_XY_RULES: dict[str, tuple[tuple[str, str, str, str, str], ...]] = {
    "scaps_cbo": (("CBO", "layer", "SCAPS_ETL", "chi_eV", "eV"),),
    "scaps_etl_donor": (("N_D", "layer", "SCAPS_ETL", "N_D_cm3", "cm^-3"),),
    "scaps_pvk_donor": (("N_D", "layer", "SCAPS_PVK", "N_D_cm3", "cm^-3"),),
    "scaps_nt_htl_pvk": (("Nt", "interface", "HTL/PVK", "N_t_cm2", "cm^-2"),),
    "scaps_nt_pvk_cb": (("Nt", "bulk_defect", "Perovskite-CB", "N_t_cm3", "cm^-3"),),
    "scaps_nt_pvk_vb": (("Nt", "bulk_defect", "Perovskite-VB", "N_t_cm3", "cm^-3"),),
    "scaps_nt_pvk_etl": (("Nt", "interface", "PVK/ETL", "N_t_cm2", "cm^-2"),),
    "scaps_et_htl_pvk": (("Et", "interface", "HTL/PVK", "E_t_eV_above_vb", "eV"),),
    "scaps_et_pvk_cb": (("Et", "bulk_defect", "Perovskite-CB", "E_t_eV_below_cb", "eV"),),
    "scaps_et_pvk_vb": (("Et", "bulk_defect", "Perovskite-VB", "E_t_eV_above_vb", "eV"),),
    "scaps_et_pvk_etl": (("Et", "interface", "PVK/ETL", "E_t_eV_above_vb", "eV"),),
    "scaps_nt_et": (
        ("Nt", "interface", "PVK/ETL", "N_t_cm2", "cm^-2"),
        ("Et", "interface", "PVK/ETL", "E_t_eV_above_vb", "eV"),
    ),
    "scaps_nt_cbo": (
        ("Nt", "interface", "PVK/ETL", "N_t_cm2", "cm^-2"),
        ("CBO", "layer", "SCAPS_ETL", "chi_eV", "eV"),
    ),
}

_NATIVE_OPTICAL_BY_LAYER: dict[str, str] = {
    "SCAPS_HTL": "scaps_sam_sqrt",
    "SCAPS_PVK": "scaps_pvk_sqrt",
    "SCAPS_ETL": "scaps_c60_sqrt",
}


def load_native_pvk_xy_template() -> dict[str, Any]:
    """Load packaged ``scaps_pvk_xy.yaml`` and bind native SCAPS sqrt(hv-Eg) optical keys."""
    src = next(
        doc for doc in load_packaged_preset_sources() if doc.id == "scaps_pvk_xy.yaml"
    )
    doc = yaml.safe_load(src.content.decode("utf-8"))
    if not isinstance(doc, dict):
        raise ValueError("scaps_pvk_xy.yaml must decode to a mapping")
    for layer in doc["layers"]:
        name = str(layer["name"])
        if name in _NATIVE_OPTICAL_BY_LAYER:
            layer["optical_material"] = _NATIVE_OPTICAL_BY_LAYER[name]
    return doc


def build_native_scaps_resources(
    base_resources: ResourceLibrary, spe_path: Path
) -> ResourceLibrary:
    """Build ResourceLibrary from ScapsAm15g1Sun.spe + SCAPS sqrt(hv - Eg) absorption."""
    if _file_sha256(spe_path) != SCAPS_SPE_SHA256:
        raise RuntimeError(f"{spe_path} SHA-256 mismatch against pinned SCAPS .spe digest")
    spe_lines = spe_path.read_text(encoding="latin-1").splitlines()
    wls_nm_list: list[float] = []
    p_bin_list: list[float] = []
    for line in spe_lines:
        stripped = line.strip()
        if stripped and not stripped.startswith(">"):
            wl_s, pw_s = stripped.split()
            wls_nm_list.append(float(wl_s))
            p_bin_list.append(float(pw_s))
    wl_nm: NDArray[np.float64] = np.asarray(wls_nm_list, dtype=np.float64)
    wl_m: NDArray[np.float64] = wl_nm * 1.0e-9
    p_w_m2: NDArray[np.float64] = np.asarray(p_bin_list, dtype=np.float64)

    # Exact dual trapezoid bin widths matching SCAPS .spe interval definition
    w_m = np.empty_like(wl_m)
    w_m[0] = 0.5 * (wl_m[1] - wl_m[0])
    w_m[1:-1] = 0.5 * (wl_m[2:] - wl_m[:-2])
    w_m[-1] = 0.5 * (wl_m[-1] - wl_m[-2])
    phi_per_m = (p_w_m2 * wl_m / (PLANCK_J_S * SPEED_OF_LIGHT_M_S)) / w_m

    spec_rows = ["wavelength_nm,spectral_flux_m-2_s-1_m-1"] + [
        f"{w:.6f},{fl:.14e}" for w, fl in zip(wl_nm, phi_per_m, strict=True)
    ]
    am15_spe_table = ResourceTable(
        "am15g",
        "spectrum",
        SourceDocument("scaps://ScapsAm15g1Sun.spe", ("\n".join(spec_rows) + "\n").encode("utf-8")),
    )

    hv_eV = (PLANCK_J_S * SPEED_OF_LIGHT_M_S) / (ELEM_CHARGE_C * wl_m)

    def _nk(name: str, eg: float) -> ResourceTable:
        alpha = np.where(
            hv_eV > eg,
            (1.0e7 / math.sqrt(eg)) * np.sqrt(np.maximum(hv_eV - eg, 0.0)),
            0.0,
        )
        k_arr = alpha * wl_m / (4.0 * math.pi)
        rows = ["wavelength_nm,n,k"] + [
            f"{w:.6f},1.0,{kv:.14e}" for w, kv in zip(wl_nm, k_arr, strict=True)
        ]
        return ResourceTable(
            name,
            "nk",
            SourceDocument(f"scaps://{name}.csv", ("\n".join(rows) + "\n").encode("utf-8")),
        )

    tables = tuple(
        am15_spe_table if (t.name == "am15g" and t.kind == "spectrum") else t
        for t in base_resources.tables
    ) + (
        _nk("scaps_sam_sqrt", 3.25),
        _nk("scaps_pvk_sqrt", 1.53),
        _nk("scaps_c60_sqrt", 1.90),
    )
    return ResourceLibrary(tables)


def apply_case_updates(
    template_doc: dict[str, Any], group_id: str, coordinates: list[Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    doc = deepcopy(template_doc)
    if group_id == "scaps_base":
        return doc, []
    rules = PVK_XY_RULES[group_id]
    pvk = next(lay for lay in doc["layers"] if lay["name"] == "SCAPS_PVK")
    chi_pvk = Decimal(str(pvk["chi_eV"]))
    updates: list[dict[str, Any]] = []
    for (axis, kind, target_name, field_name, unit), raw_val in zip(
        rules, coordinates, strict=True
    ):
        val = float(Decimal(str(chi_pvk - Decimal(str(raw_val)))) if axis == "CBO" else Decimal(str(raw_val)))
        if kind == "layer":
            layer = next(lay for lay in doc["layers"] if lay["name"] == target_name)
            layer[field_name] = val
        elif kind == "bulk_defect":
            defect = next(
                d
                for lay in doc["layers"]
                for d in lay.get("bulk_defects", ())
                if d.get("name") == target_name
            )
            defect.pop("E_t_eV_below_cb", None) if field_name.startswith("E_t_") else None
            defect.pop("E_t_eV_above_vb", None) if field_name.startswith("E_t_") else None
            defect[field_name] = val
        elif kind == "interface":
            iface = next(it for it in doc["interfaces"] if it["target"] == target_name)
            if field_name.startswith("E_t_"):
                iface.pop("E_t_eV_below_cb", None)
                iface.pop("E_t_eV_above_vb", None)
            iface[field_name] = val
            if field_name == "N_t_cm2" and iface.get("distribution") == "gaussian":
                e_char = float(iface.get("E_char_eV", 0.10))
                iface["N_peak_cm3"] = float(val) / (math.sqrt(math.pi) * e_char)
        updates.append(
            {"axis": axis, "target": target_name, "field": field_name, "value": val, "unit": unit}
        )
    return doc, updates


_WORKER_CTX: dict[str, Any] = {}


def _init_worker(spe_path_str: str) -> None:
    defaults, base_res = _load_packaged_catalog_and_resources()
    _WORKER_CTX["defaults"] = defaults
    _WORKER_CTX["resources"] = build_native_scaps_resources(base_res, Path(spe_path_str))


def _eval_case(task: dict[str, Any]) -> dict[str, Any]:
    if not _WORKER_CTX:
        _init_worker(str(REPO_ROOT / SCAPS_SPE_REL))
    defaults = _WORKER_CTX["defaults"]
    resources = _WORKER_CTX["resources"]
    cid = str(task["id"])
    gid = str(task["group_id"])
    coords = list(task["coordinates"])
    ref_status = str(task["reference_status"])
    ref_values = task["reference_values"]
    gates_spec = task["pointwise_gates"]
    res_floors = task["resolution_floor"]

    t0 = time.perf_counter()
    doc, updates = apply_case_updates(task["template_doc"], gid, coords)
    yb = yaml.safe_dump(doc, sort_keys=False).encode("utf-8")
    eff_sha = hashlib.sha256(yb).hexdigest()
    prepared = _prepare_scaps_yaml_device(
        yb,
        source_id=f"scaps_pvk_xy://{cid}",
        device_id=cid.replace(":", "_"),
        defaults=defaults,
        resources=resources,
    )
    opts = SteadyOptions(
        continuity_tolerance_A_m2=1.0e-4,
        current_spread_tolerance_A_m2=1.0e-4,
    )
    is_high_cbo = (
        gid in ("scaps_cbo", "scaps_nt_cbo")
        and len(coords) >= 1
        and float(coords[-1]) >= 0.39
    )
    ladder = (
        (15, 10, 8)
        if (is_high_cbo and float(coords[-1]) <= 0.41)
        else ((10, 8, 6) if is_high_cbo else (25, 15, 10))
    )

    cand_metrics: dict[str, float | None] | None = None
    solver_error: str | None = None
    evaluated_points = 0
    compiled_id: str | None = None
    route = "none"
    for n_int in ladder:
        try:
            comp = compile_device_1d(
                prepared,
                intervals_per_layer=n_int,
                interface_topology="two_sided_trace",
                geometry_name=PHYSICAL_BOUNDARY_TWO_SIDED_V1,
                thermionic_mode=SCAPS_EFFECTIVE_VELOCITY_THERMIONIC_V1,
                interface_band_reference_mode=SCAPS_INTERFACE_EXTREMA_ANCHOR,
                interface_defect_distribution_mode=SCAPS_GAUSSIAN_INTERFACE_QUADRATURE,
                optical_model="beer_lambert_spectral",
                allow_legacy_benchmark_deck=True,
            )
            m, evaluated_points = _run_scaps_steady_jv_aligned(
                comp, np.linspace(0.0, 1.45, 37), opts
            )
            cand_metrics = {
                "Voc_V": m.open_circuit_voltage_V,
                "Jsc_mA_cm2": float(m.short_circuit_current_A_m2) * 0.1,
                "FF_percent": float(m.fill_factor) * 100.0,
                "PCE_percent": float(m.efficiency) * 100.0,
            }
            compiled_id = comp.identity
            route = f"n{n_int}"
            solver_error = None
            break
        except Exception as exc:
            solver_error = f"{type(exc).__name__}: {exc}"

    elapsed_s = time.perf_counter() - t0
    metric_comparisons: dict[str, Any] = {}
    if ref_status == "reference_missing" or ref_values is None:
        disposition = "reference_missing"
    elif solver_error is not None or cand_metrics is None:
        disposition = "solver_failed"
    else:
        all_passed = True
        for q_name, g_rule in gates_spec.items():
            ref_v = float(ref_values[q_name])
            cand_raw = cand_metrics[q_name]
            assert cand_raw is not None
            cand_v = float(cand_raw)
            u_rep = _decimal_ulp(ref_v, float(res_floors.get(q_name, 1.0e-6)))
            allowance = float(g_rule["atol"]) + float(g_rule["rtol"]) * abs(ref_v) + u_rep
            diff = abs(cand_v - ref_v)
            passed = bool(diff <= allowance)
            all_passed = all_passed and passed
            metric_comparisons[q_name] = {
                "candidate": cand_v,
                "reference": ref_v,
                "signed_diff": cand_v - ref_v,
                "abs_diff": diff,
                "u_report": u_rep,
                "allowance": allowance,
                "passed": passed,
            }
        disposition = (
            "passed_external_gate" if all_passed else "exceeded_external_tolerance"
        )

    return {
        "id": cid,
        "group_id": gid,
        "coordinates": coords,
        "reference_status": ref_status,
        "disposition": disposition,
        "continuation_route": route,
        "effective_yaml_sha256": eff_sha,
        "compiled_identity": compiled_id,
        "evaluated_bias_points": evaluated_points,
        "elapsed_s": elapsed_s,
        "updates": updates,
        "candidate_metrics": cand_metrics,
        "reference_metrics": ref_values,
        "metric_comparisons": metric_comparisons,
        "error": solver_error,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "outputs" / "scaps_native_pvk_xy_262.json",
        help="Destination JSON path for full per-case scorecard.",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-cases", type=int, default=None)
    args = parser.parse_args(argv)

    contract_path = REPO_ROOT / SCAPS_CONTRACT_REL
    if _file_sha256(contract_path) != SCAPS_CONTRACT_SHA256:
        raise RuntimeError("ScapsComparisonContractV1.json SHA-256 mismatch")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    manifest_path = REPO_ROOT / MANIFEST_REL
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    spe_path = REPO_ROOT / SCAPS_SPE_REL

    template_doc = load_native_pvk_xy_template()
    floors_by_group = {
        row["group_id"]: dict(row["reported_column_resolution_floor"])
        for row in contract["reported_precision"]["per_group"]
    }
    floors_by_group["scaps_base"] = {
        "Voc_V": 1.0e-4,
        "Jsc_mA_cm2": 1.0e-6,
        "FF_percent": 1.0e-2,
        "PCE_percent": 1.0e-2,
    }

    tasks: list[dict[str, Any]] = [
        {
            "id": "scaps_base",
            "group_id": "scaps_base",
            "coordinates": [],
            "reference_status": "reported_rounded_base",
            "reference_values": dict(manifest["scaps"]["base"]["values"]),
            "template_doc": template_doc,
            "pointwise_gates": contract["pointwise_gates"],
            "resolution_floor": floors_by_group["scaps_base"],
        }
    ]
    for grp in manifest["scaps"]["groups"]:
        gid = str(grp["id"])
        for pt in grp["points"]:
            tasks.append(
                {
                    "id": f"{gid}:row{pt['worksheet_row']}",
                    "group_id": gid,
                    "coordinates": list(pt["coordinates"]),
                    "reference_status": pt["status"],
                    "reference_values": pt.get("values"),
                    "template_doc": template_doc,
                    "pointwise_gates": contract["pointwise_gates"],
                    "resolution_floor": floors_by_group[gid],
                }
            )
    if args.max_cases is not None:
        tasks = tasks[: args.max_cases]

    t0 = time.perf_counter()
    if args.workers <= 1:
        _init_worker(str(spe_path))
        rows = [_eval_case(t) for t in tasks]
    else:
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=ctx,
            initializer=_init_worker,
            initargs=(str(spe_path),),
        ) as pool:
            rows = list(pool.map(_eval_case, tasks))
    wall_s = time.perf_counter() - t0

    if _loaded_forbidden_legacy_modules():
        raise RuntimeError("Forbidden legacy module imported")

    numeric_rows = [
        r
        for r in rows
        if r["reference_status"] != "reference_missing"
        and r["disposition"] != "solver_failed"
    ]
    voc_diffs_mV = [
        1000.0 * r["metric_comparisons"]["Voc_V"]["abs_diff"] for r in numeric_rows
    ]
    jsc_diffs = [
        r["metric_comparisons"]["Jsc_mA_cm2"]["abs_diff"] for r in numeric_rows
    ]
    ff_diffs = [r["metric_comparisons"]["FF_percent"]["abs_diff"] for r in numeric_rows]
    pce_diffs = [
        r["metric_comparisons"]["PCE_percent"]["abs_diff"] for r in numeric_rows
    ]

    def _pct(vals: list[float], q: float) -> float:
        if not vals:
            return 0.0
        s = sorted(vals)
        idx = min(len(s) - 1, max(0, math.ceil(q * len(s)) - 1))
        return float(s[idx])

    counts = {
        "total_executed": len(rows),
        "numeric_evaluated": len(numeric_rows),
        "passed_external_gate": sum(
            1 for r in rows if r["disposition"] == "passed_external_gate"
        ),
        "exceeded_external_tolerance": sum(
            1 for r in rows if r["disposition"] == "exceeded_external_tolerance"
        ),
        "reference_missing": sum(
            1 for r in rows if r["disposition"] == "reference_missing"
        ),
        "solver_failed": sum(1 for r in rows if r["disposition"] == "solver_failed"),
        "per_metric_passed": {
            q: sum(1 for r in numeric_rows if r["metric_comparisons"][q]["passed"])
            for q in ("Voc_V", "Jsc_mA_cm2", "FF_percent", "PCE_percent")
        },
        "voc_error_buckets": {
            "le_1mV": sum(1 for v in voc_diffs_mV if v <= 1.0),
            "le_2mV": sum(1 for v in voc_diffs_mV if v <= 2.0),
            "le_5mV": sum(1 for v in voc_diffs_mV if v <= 5.0),
            "le_10mV": sum(1 for v in voc_diffs_mV if v <= 10.0),
            "le_20mV": sum(1 for v in voc_diffs_mV if v <= 20.0),
            "le_30mV": sum(1 for v in voc_diffs_mV if v <= 30.0),
        },
        "error_quantiles": {
            "Voc_mV": {
                "median": statistics.median(voc_diffs_mV) if voc_diffs_mV else 0.0,
                "p90": _pct(voc_diffs_mV, 0.90),
                "max": max(voc_diffs_mV, default=0.0),
            },
            "Jsc_mA_cm2": {
                "median": statistics.median(jsc_diffs) if jsc_diffs else 0.0,
                "p90": _pct(jsc_diffs, 0.90),
                "max": max(jsc_diffs, default=0.0),
            },
            "FF_pct": {
                "median": statistics.median(ff_diffs) if ff_diffs else 0.0,
                "p90": _pct(ff_diffs, 0.90),
                "max": max(ff_diffs, default=0.0),
            },
            "PCE_pct": {
                "median": statistics.median(pce_diffs) if pce_diffs else 0.0,
                "p90": _pct(pce_diffs, 0.90),
                "max": max(pce_diffs, default=0.0),
            },
        },
    }

    payload = {
        "schema": "solarlab.scaps_native_pvk_xy_alignment.v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "run_role": "native_zero_fit_scaps_alignment",
        "scientific_qualification": False,
        "physical_certificate_status": "not_evaluated",
        "source_provenance": {
            "preset_id": "scaps_pvk_xy.yaml",
            "am15g_spe_rel": SCAPS_SPE_REL,
            "am15g_spe_sha256": _file_sha256(spe_path),
            "scaps_contract_sha256": SCAPS_CONTRACT_SHA256,
            "manifest_sha256": _file_sha256(manifest_path),
            "zero_empirical_calibration_factors": True,
        },
        "wall_elapsed_s": wall_s,
        "environment": _host_environment_record(False),
        "summary_counts": counts,
        "cases": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary_counts": counts, "wall_s": round(wall_s, 2)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
