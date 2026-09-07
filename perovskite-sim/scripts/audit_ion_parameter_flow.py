#!/usr/bin/env python3
"""Audit configured ions, compiled material arrays and the actual J-V driver."""
from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import math
import os
from pathlib import Path
import sys
import time
import warnings

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from threadpoolctl import threadpool_limits

from backend.main import _coerce_numbers, _config_dict_from_path, stack_from_dict
from perovskite_sim.experiments.jv_sweep import build_electrical_grid, run_jv_sweep
from perovskite_sim.physics.generation import dual_cell_widths
from perovskite_sim.reproducibility import semantic_sha256
from perovskite_sim.solver.mol import build_material_arrays


def jsonable(value):
    if dataclasses.is_dataclass(value):
        return jsonable(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("no_ions", "frozen_ions", "nominal"), required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/calado2016_fig1f.yaml")
    parser.add_argument("--n-grid", type=int, default=60)
    parser.add_argument("--n-points", type=int, default=31)
    parser.add_argument("--scan-rate", type=float, default=1.0)
    parser.add_argument("--v-max", type=float, default=1.2)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload = copy.deepcopy(_coerce_numbers(_config_dict_from_path(str(args.config))))
    if args.case != "nominal":
        for layer in payload["layers"]:
            layer["D_ion"] = layer["D_ion_neg"] = 0.0
            if args.case == "no_ions":
                layer["P0"] = layer["P0_neg"] = 0.0
    stack = stack_from_dict(payload)
    x = build_electrical_grid(stack, args.n_grid)
    mat = build_material_arrays(x, stack)
    widths = dual_cell_widths(x)
    summary = {
        "case": args.case,
        "config_path": str(args.config),
        "device_payload": payload,
        "stack_semantic_sha256": semantic_sha256(stack),
        "driver": "run_jv_sweep; ordinary frontend compatibility protocol",
        "parameters": {
            "N_grid": args.n_grid, "n_points": args.n_points,
            "v_rate": args.scan_rate, "V_max": args.v_max,
            "rtol": args.rtol, "atol": args.atol,
        },
        "effective_ions": {
            "has_dual_ions": mat.has_dual_ions,
            "positive_inventory_m2": float(np.sum(mat.P_ion0 * widths)),
            "D_ion_node": mat.D_ion_node,
            "D_ion_face": mat.D_ion_face,
            "P_ion0": mat.P_ion0,
            "P_ion0_neg": mat.P_ion0_neg,
            "D_ion_neg_node": mat.D_ion_neg_node,
        },
    }
    started = time.monotonic()

    def progress(stage, current, total, message):
        if current == 0 or current % 5 == 0 or current == total:
            print(f"{args.case} {stage} {current}/{total} {message}", flush=True)

    exit_code = 0
    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1, user_api="blas"):
        warnings.simplefilter("always")
        try:
            result = run_jv_sweep(
                stack, **summary["parameters"], save_snapshots=True,
                decompose_currents=True, progress=progress,
            )
            forward_power = float(np.max(result.V_fwd * result.J_fwd))
            reverse_power = float(np.max(result.V_rev * result.J_rev))
            summary.update({
                "status": "completed",
                "metrics_fwd": result.metrics_fwd,
                "metrics_rev": result.metrics_rev,
                "HI_solarlab": result.hysteresis_index,
                "HI_paper": reverse_power / forward_power - 1.0 if forward_power > 0 else None,
                "point_status_fwd": result.status_fwd,
                "point_status_rev": result.status_rev,
                "current_decomp_fwd": result.decomp_fwd,
                "current_decomp_rev": result.decomp_rev,
            })
            arrays = {"x": x, "V_fwd": result.V_fwd, "J_fwd": result.J_fwd,
                      "V_rev": result.V_rev, "J_rev": result.J_rev}
            for branch, snapshots in (("fwd", result.snapshots_fwd), ("rev", result.snapshots_rev)):
                if snapshots:
                    arrays[f"P_{branch}"] = np.stack([snapshot.P for snapshot in snapshots])
                    arrays[f"phi_{branch}"] = np.stack([snapshot.phi for snapshot in snapshots])
                    summary[f"positive_inventory_{branch}_m2"] = arrays[f"P_{branch}"] @ widths
            np.savez_compressed(args.out_dir / "curves.npz", **arrays)
        except Exception as exc:
            summary.update({"status": "failed", "error_type": type(exc).__name__, "error": str(exc)})
            exit_code = 1
        summary["warnings"] = [str(item.message) for item in caught]
    summary["elapsed_s"] = time.monotonic() - started
    with (args.out_dir / "audit.json").open("w") as stream:
        json.dump(jsonable(summary), stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(jsonable({key: summary[key] for key in ("case", "status", "elapsed_s", "HI_paper", "error") if key in summary})), flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
