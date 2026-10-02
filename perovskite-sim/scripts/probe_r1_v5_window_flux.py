"""Split saved ion-flux discrepancies into evaluation and input differences."""
from __future__ import annotations

import argparse
from decimal import Decimal, localcontext
import hashlib
import json
import os
from pathlib import Path

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
import scipy.linalg
from threadpoolctl import threadpool_info, threadpool_limits
from perovskite_sim.constants import Q
from perovskite_sim.experiments.one_dimensional_mechanism_r1 import build_r1_material
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.physics.ion_migration import ion_face_flux
from scripts.diagnose_one_dimensional_mechanism_r1_weak_flux import decimal_flux


def number(value):
    return Decimal.from_float(float(value))


def maximum(values):
    return float(max(map(abs, values), default=Decimal(0)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnosis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    project = Path(__file__).resolve().parents[1]
    config = project / "tests/fixtures/configs/dynamic_interface_defect_ion_transient_absorber_only.yaml"
    stack = load_device_from_yaml(config)
    diagnosis = json.loads((args.diagnosis / "DiagnosisV1.json").read_text())
    selected = json.loads((args.diagnosis / "SelectedRowsV1.json").read_text())
    if diagnosis["request"]["control"] != "D":
        raise ValueError("probe requires the unscaled coupled D control")
    grid, material = build_r1_material(stack, diagnosis["request"]["intervals"])
    if not material.ion_steric_diffusion_only or material.has_dual_ions:
        raise ValueError("probe requires the declared single-ion diffusion-only law")
    dx = np.diff(grid)
    limits = np.broadcast_to(material.P_lim_node, grid.shape)
    results = []
    with threadpool_limits(limits=1, user_api="blas"), localcontext() as ctx:
        ctx.prec = 90
        for entry in selected:
            row = entry["row"]
            parts = row["physics_reconstruction"]["eliminated_operator"]
            population = np.asarray(row["state"]["positive_m3"])
            if np.any(population <= 0) or np.any(population / limits >= .999):
                raise ValueError("oracle excludes invalid or clipped populations")
            groups = {}
            for side in ("direct", "eliminated"):
                potential = np.asarray(parts["potential"][side])
                actual = ion_face_flux(potential, population, dx, material.D_ion_face,
                    material.V_T_device, material.P_lim_face, steric_diffusion_only=True,
                    P_lim_node=material.P_lim_node)
                if not np.array_equal(actual, parts["positive_ion_flux"][side]):
                    raise ValueError("saved flux does not match its stated represented inputs")
                args50 = (potential, population, dx, material.D_ion_face, material.V_T_device, limits)
                fifty = decimal_flux(*args50, precision=50)
                eighty = decimal_flux(*args50, precision=80)
                evaluation = [number(a) - b for a, b in zip(actual, eighty)]
                # This probes one algebraically identical, cancellation-aware
                # evaluation. It is not installed as a production replacement.
                crowding = -np.log1p(-population / limits)
                xi = np.diff(potential) / material.V_T_device + np.diff(crowding)
                eta = np.log1p((population[1:] - population[:-1]) / population[:-1]) + xi
                from perovskite_sim.discretization.fe_operators import bernoulli
                balanced = material.D_ion_face / dx * bernoulli(xi) * population[:-1] * (-np.expm1(eta))
                groups[side] = {"float_flux_m2_s": actual.tolist(),
                    "decimal_flux_m2_s": [str(value) for value in eighty],
                    "evaluation_error_m2_s": [str(value) for value in evaluation],
                    "max_evaluation_error_m2_s": maximum(evaluation),
                    "max_evaluation_error_A_m2": float(number(Q) * max(map(abs, evaluation))),
                    "oracle_50_to_80_m2_s": maximum(a-b for a,b in zip(fifty,eighty)),
                    "balanced_evaluation_error_m2_s": maximum(number(a)-b for a,b in zip(balanced,eighty))}
            direct = [Decimal(v) for v in groups["direct"]["decimal_flux_m2_s"]]
            eliminated = [Decimal(v) for v in groups["eliminated"]["decimal_flux_m2_s"]]
            input_effect = [a-b for a,b in zip(direct,eliminated)]
            total = [number(a)-number(b) for a,b in zip(groups["direct"]["float_flux_m2_s"], groups["eliminated"]["float_flux_m2_s"])]
            cancellation = [Decimal(a)-Decimal(b) for a,b in zip(groups["direct"]["evaluation_error_m2_s"],groups["eliminated"]["evaluation_error_m2_s"])]
            discrepancy = maximum(t-(i+e) for t,i,e in zip(total,input_effect,cancellation))
            if discrepancy > 1e-60:
                raise ValueError("discrepancy decomposition failed at oracle precision")
            results.append({"row":entry["record_index"],"time_s":row["time_s"],"substeps":row["substeps"],
                "sides":groups,"saved_float_difference_m2_s":maximum(total),
                "represented_potential_input_effect_m2_s":maximum(input_effect),
                "represented_potential_input_effect_A_m2":float(number(Q)*max(map(abs,input_effect))),
                "evaluation_difference_m2_s":maximum(cancellation),
                "decomposition_residual_m2_s":discrepancy})
        runtime = threadpool_info()
    report = {"schema":"R1V5WindowFluxOracleV1","scope":"fixed_represented_inputs_no_new_physical_solves",
        "source_diagnosis_sha256":hashlib.sha256((args.diagnosis/"DiagnosisV1.json").read_bytes()).hexdigest(),
        "source_rows_sha256":hashlib.sha256((args.diagnosis/"SelectedRowsV1.json").read_bytes()).hexdigest(),
        "source_hashes":{str(p.relative_to(project)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__),config,project/"scripts/diagnose_one_dimensional_mechanism_r1_weak_flux.py",
             project/"perovskite_sim/physics/ion_migration.py",project/"perovskite_sim/discretization/fe_operators.py",
             project/"perovskite_sim/experiments/one_dimensional_mechanism_r1.py"]},
        "material_inputs":{"grid_m":grid.tolist(),"diffusion_m2_s":material.D_ion_face.tolist(),
            "thermal_voltage_V":material.V_T_device,"site_limit_m3":limits.tolist()},
        "runtime_threadpools":runtime,"cases":results,"production_changed":False,
        "terminal_response_error_budget":"unknown",
        "interpretation":"Splits arithmetic error from two represented potential inputs; neither is a full coupled-trajectory uncertainty bound."}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n")
    for r in results:
        print(r["row"],r["saved_float_difference_m2_s"],r["represented_potential_input_effect_m2_s"],
              r["sides"]["direct"]["max_evaluation_error_m2_s"],r["sides"]["direct"]["balanced_evaluation_error_m2_s"],flush=True)


if __name__ == "__main__":
    main()
