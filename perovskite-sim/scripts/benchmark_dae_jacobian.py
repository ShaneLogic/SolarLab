#!/usr/bin/env python3
"""Measure dense-central and sparse-analytic DAE Newton work."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from perovskite_sim.discretization.grid import Layer, multilayer_grid  # noqa: E402
from perovskite_sim.models.config_loader import (  # noqa: E402
    load_device_from_yaml,
)
from perovskite_sim.solver.dae import (  # noqa: E402
    build_consistent_initial_condition,
    build_no_ion_no_interface_dae,
)
from perovskite_sim.solver.dae_integrator import (  # noqa: E402
    run_backward_euler_reference,
)
from perovskite_sim.solver.dae_jacobian import (  # noqa: E402
    build_structured_state_jacobian,
)
from perovskite_sim.solver.newton import solve_equilibrium  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--intervals",
        type=int,
        nargs="+",
        default=[8, 16, 32, 64],
    )
    parser.add_argument("--repeats", type=int, default=5)
    return parser


def _problem(intervals: int):
    source = load_device_from_yaml(
        ROOT / "configs/csi_vannijen2025_pn_cv.yaml"
    )
    source_layer = source.layers[1]
    if source_layer.params is None:
        raise RuntimeError("c-Si benchmark layer has no material parameters")
    layer = replace(
        source_layer,
        params=replace(source_layer.params, alpha=2.0e4),
    )
    stack = replace(
        source,
        layers=(layer,),
        V_bi=0.0,
        built_in_potential_mode="legacy_manual",
        Phi=1.0e17,
        interfaces=(),
        interface_defects=(),
        grid_interval_weights=(),
        grid_alphas=(),
    )
    grid = multilayer_grid([Layer(layer.thickness, intervals)], alpha=1.0)
    reference = solve_equilibrium(grid, stack)
    model = build_no_ion_no_interface_dae(
        grid,
        stack,
        reference,
        illuminated=True,
        reference_time_s=1.0e-9,
    )
    return model, build_consistent_initial_condition(model)


def _measure(model, initial, mode: str, repeats: int) -> dict[str, object]:
    time = np.array([0.0, 2.5e-10])
    run_backward_euler_reference(
        model,
        time,
        initial=initial,
        jacobian_mode=mode,
    )
    timings = []
    result = None
    for _ in range(repeats):
        start = perf_counter()
        result = run_backward_euler_reference(
            model,
            time,
            initial=initial,
            jacobian_mode=mode,
        )
        timings.append(perf_counter() - start)
    if result is None:
        raise AssertionError("positive repeat count produced no result")
    return {
        "jacobian_evaluations": result.total_jacobian_evaluations,
        "median_wall_time_s": float(np.median(timings)),
        "newton_iterations": result.total_nonlinear_iterations,
        "residual_evaluations": result.total_residual_evaluations,
        "wall_time_samples_s": timings,
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.repeats < 1 or any(value < 3 for value in args.intervals):
        raise SystemExit("repeats must be positive and intervals must be >= 3")
    rows = []
    for intervals in args.intervals:
        model, initial = _problem(intervals)
        dense = _measure(model, initial, "dense_central", args.repeats)
        structured = _measure(
            model,
            initial,
            "structured_analytic",
            args.repeats,
        )
        tangent = build_structured_state_jacobian(
            model,
            initial.coordinate,
            initial.derivative,
        )
        rows.append(
            {
                "dense": dense,
                "intervals": intervals,
                "nodes": model.layout.node_count,
                "speedup": dense["median_wall_time_s"]
                / structured["median_wall_time_s"],
                "state_size": model.layout.size,
                "structured": structured,
                "structured_nnz": tangent.nonzero_count,
            }
        )
    payload = {
        "environment": {
            name: os.environ.get(name)
            for name in (
                "OPENBLAS_NUM_THREADS",
                "OMP_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            )
        },
        "repeats": args.repeats,
        "rows": rows,
        "schema_version": "dae-jacobian-benchmark-v1",
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
