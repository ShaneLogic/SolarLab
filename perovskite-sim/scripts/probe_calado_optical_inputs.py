#!/usr/bin/env python3
"""Replay optical controls through the real backend worker from a captured request.

This is backend execution evidence. It does not claim new browser interaction
or quantitative reproduction of Calado 2016.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from threadpoolctl import threadpool_limits

from backend.main import JobRequest, _JOB_REGISTRY, start_job
from backend.jobs import JobStatus, _DRAIN_TIMEOUT
from summarize_ion_frontend_matrix import audit_record, render_plot, validate_matrix


def optical_cases(original, *, atol_m3=None):
    if atol_m3 is not None and (not np.isfinite(atol_m3) or atol_m3 <= 0):
        raise ValueError("atol_m3 must be finite and positive")
    if original["kind"] != "jv" or original["params"]["waveform"]["uniform_generation_rate_m3_s"] is not None:
        raise ValueError("Expected a captured J-V request using device optics")
    layers = original["device"]["layers"]
    if any(layer.get(name, 0) for layer in layers for name in ("P0", "P0_neg", "D_ion", "D_ion_neg")):
        raise ValueError("Optical isolation requires an ion-free source request")
    absorber = [i for i, layer in enumerate(layers) if layer["role"] == "absorber"]
    if len(absorber) != 1 or any(layer.get("optical_material") for layer in layers):
        raise ValueError("Expected one absorber and Beer-Lambert optics")
    cases = {}
    for name, alpha_scale, flux_scale, uniform in (
        ("baseline", 1, 1, None), ("alpha_zero", 0, 1, None),
        ("alpha_double", 2, 1, None), ("flux_zero", 1, 0, None),
        ("flux_half", 1, 0.5, None), ("uniform", 1, 1, 2.5e27),
        ("uniform_alpha_zero", 0, 1, 2.5e27),
    ):
        request = copy.deepcopy(original)
        request["device"]["layers"][absorber[0]]["alpha"] *= alpha_scale
        request["device"]["device"]["Phi"] *= flux_scale
        request["params"]["waveform"]["uniform_generation_rate_m3_s"] = uniform
        if atol_m3 is not None:
            request["params"].setdefault("waveform_controls", {})["atol_m3"] = atol_m3
        cases[name] = request
    return cases


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "outputs/calado-reproduction/ui-protocol/waveform-optics-baseline-jv-frontend.json")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--atol-m3", type=float,
                        help="Explicit common numerical override; leaves physical inputs unchanged")
    args = parser.parse_args()
    raw = args.source.read_bytes()
    source = json.loads(raw)
    cases = optical_cases(source["request"], atol_m3=args.atol_m3)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    provenance = {"evidence_origin": "backend_job_replay", "source": str(args.source.resolve()),
        "numerical_control_overrides": {} if args.atol_m3 is None else {"atol_m3": args.atol_m3},
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "scope": "Real backend workers from a captured frontend request; no new browser action or paper reproduction claim",
        "code_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in (
            Path(__file__), ROOT / "backend/main.py", ROOT / "perovskite_sim/experiments/waveform_jv.py",
            ROOT / "perovskite_sim/experiments/waveform_jacobian.py",
            ROOT / "perovskite_sim/experiments/jv_sweep.py", ROOT / "perovskite_sim/solver/mol.py",
            ROOT / "perovskite_sim/physics/generation.py")}}
    write_json(args.out_dir / "provenance.json", provenance)
    records, rows, failures = {}, {}, {}
    with threadpool_limits(limits=1, user_api="blas"):
        for name, request in cases.items():
            started = time.monotonic()
            job_id = start_job(JobRequest(**request))["job_id"]
            record = {"evidence_origin": "backend_job_replay", "request": request, "job_id": job_id, "status": "running"}
            filename = args.out_dir / f"{name}.json"
            write_json(filename, record)
            print(f"{name}: submitted {job_id}", flush=True)
            while True:
                event = _JOB_REGISTRY.next_event(job_id, timeout=1.0)
                if event is None:
                    break
                if event is not _DRAIN_TIMEOUT and (event.current == event.total or event.current % 25 == 0):
                    print(f"{name}: {event.stage} {event.current}/{event.total}", flush=True)
            state, data, error = _JOB_REGISTRY.status(job_id)
            record.update(status=state.value, elapsed_s=time.monotonic() - started)
            if state != JobStatus.DONE:
                record["error"] = error
                write_json(filename, record)
                failures[name] = {"job_id": job_id, "status": state.value, "error": error}
                print(f"{name}: failed; details saved in {filename}", flush=True)
                continue
            record["result"] = {"kind": "jv", "data": data, "device": request["device"]}
            write_json(filename, record)
            rows[name], records[name] = audit_record(filename)
            print(f"{name}: Jsc={rows[name]['J_sc_fwd_A_m2']:.8g}, Gbudget={rows[name]['generation_budget_A_m2']:.8g}", flush=True)
    if failures:
        write_json(args.out_dir / "checks.json", {
            "status": "failed", "scope": provenance["scope"],
            "requested_cases": list(cases), "completed_cases": rows, "failed_cases": failures,
            "matrix_checks": "not_run: complete paired results required",
        })
        raise RuntimeError(f"{len(failures)}/{len(cases)} optical cases failed; all cases were attempted")
    checks = {}
    for left, right in (("alpha_zero", "flux_zero"), ("uniform", "uniform_alpha_zero")):
        checks[f"{left}_vs_{right}"] = {}
        for key in ("J_fwd", "J_rev"):
            first, second = records[left]["result"]["data"][key], records[right]["result"]["data"][key]
            np.testing.assert_array_equal(first, second)
            checks[f"{left}_vs_{right}"][key + "_max_difference_A_m2"] = float(np.max(np.abs(np.asarray(first) - second)))
    np.testing.assert_allclose(rows["flux_half"]["generation_budget_A_m2"], 0.5 * rows["baseline"]["generation_budget_A_m2"], rtol=1e-12)
    assert rows["alpha_zero"]["generation_budget_A_m2"] == rows["flux_zero"]["generation_budget_A_m2"] == 0
    assert rows["alpha_double"]["J_sc_fwd_A_m2"] > rows["baseline"]["J_sc_fwd_A_m2"] > rows["flux_half"]["J_sc_fwd_A_m2"] > 0
    for group, names in (("device_optics", list(cases)[:5]), ("uniform_override", list(cases)[5:])):
        subset = {name: records[name] for name in names}
        validate_matrix(subset, "optics")
        report = {"scope": provenance["scope"], "varied_axis": "optics", "cases": {name: rows[name] for name in names}}
        write_json(args.out_dir / f"{group}.json", report)
        render_plot(report, args.out_dir / f"{group}.png")
    write_json(args.out_dir / "checks.json", {"status": "passed", "checks": checks, "cases": rows})


if __name__ == "__main__":
    main()
