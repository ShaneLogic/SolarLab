"""Propagate two saved high-precision field differences to current components.

This is a fixed-QF, fixed-ion, fixed-occupancy sensitivity estimate. It does
not solve the local carrier equations or certify a coupled trajectory error.
"""
from decimal import Decimal, localcontext
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from perovskite_sim.experiments.one_dimensional_mechanism_r1 import build_r1_material
from perovskite_sim.models.config_loader import load_device_from_yaml


def dec(x):
    return Decimal(x) if isinstance(x, str) else Decimal.from_float(float(x))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def serial(x):
    if isinstance(x, Decimal):
        return str(x)
    if isinstance(x, dict):
        return {key: serial(value) for key, value in x.items()}
    if isinstance(x, (list, tuple)):
        return [serial(value) for value in x]
    return x


def currents(sample, mat, parameters):
    phi, n, p = [list(map(dec, sample[key])) for key in ("phi_V", "pb_n_m3", "pb_p_m3")]
    vt, q = dec(parameters["thermal_voltage_V"]), dec(parameters["elementary_charge_C"])
    chi, gap = list(map(dec, mat.chi)), list(map(dec, mat.Eg))
    def bernoulli(x):
        return Decimal(1) if x == 0 else x / (x.exp() - 1)
    electron, hole = [], []
    for i, spacing in enumerate(parameters["dx_m"]):
        xn = (phi[i+1] + chi[i+1] - phi[i] - chi[i]) / vt
        xp = (phi[i+1] + chi[i+1] + gap[i+1] - phi[i] - chi[i] - gap[i]) / vt
        electron.append(q * dec(mat.D_n_face[i]) / dec(spacing) * (bernoulli(xn)*n[i+1] - bernoulli(-xn)*n[i]))
        hole.append(q * dec(mat.D_p_face[i]) / dec(spacing) * (bernoulli(xp)*p[i] - bernoulli(-xp)*p[i+1]))
    return electron, hole


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    inputs_path, references_path = args.reference / "SharedReferenceInputsV1.json", args.reference / "SharedReferenceResultV1.json"
    inputs, references = json.loads(inputs_path.read_text()), json.loads(references_path.read_text())
    if sha(inputs_path) != references["diagnostic_inputs_sha256"]:
        raise ValueError("high-precision reference no longer binds the recorded inputs")
    parameters = inputs["parameters"]
    stack = load_device_from_yaml(PROJECT / "tests/fixtures/configs/dynamic_interface_defect_ion_transient_absorber_only.yaml")
    grid, mat = build_r1_material(stack, 16)
    if list(grid) != parameters["grid_m"]:
        raise ValueError("current material grid differs from the fixed field diagnostic")
    report = {"schema": "R1SavedFieldCurrentPropagationV4", "scope": "two_fixed_state_pairs_conditional_component_sensitivity",
        "script_sha256": sha(__file__), "inputs_sha256": {p.name: sha(p) for p in (inputs_path, references_path)},
        "reference_source_commit": references["source_commit"], "trajectory_source_commit": references["trajectory_source_commit"],
        "coupled_response_error_budget_status": "unknown",
        "known_terms": ["saved_50_and_80_digit_field_and_population_differences", "ordinary_face_SG_current_changes", "ion_charge_flux_changes", "finite_step_contact_displacement_change_with_previous_state_fixed"],
        "unresolved_terms": ["coupled_carrier_and_interface_resolve", "previous_state_error_and_correlations", "time_discretization", "spatial_discretization", "constitutive_approximation_uncertainty"],
        "acceptance_thresholds_changed": False, "cases": []}
    excluded = int(parameters["interface_left_node"])
    for data, reference in zip(inputs["states"], references["states"]):
        if data["row_index_zero_based"] != reference["row_index_zero_based"]:
            raise ValueError("reference state does not match the requested pair")
        case = {"rows": [data["previous_row_index_zero_based"], data["row_index_zero_based"]],
                "time_s": data["time_s"], "dt_s": data["dt_s"], "by_precision": {}}
        for digits in (50, 80):
            with localcontext() as ctx:
                ctx.prec = digits
                samples = reference["by_precision"][str(digits)]["samples"]
                base = samples["saved_direct"]
                jn0, jp0 = currents(base, mat, parameters)
                values = {}
                for key in ("direct_unrounded_coordinate", "independent_eliminated_shared_reference", "independent_eliminated_absolute"):
                    sample = samples[key]
                    jn, jp = currents(sample, mat, parameters)
                    delta_n, delta_p = [[a-b for a, b in zip(x, y)] for x, y in ((jn, jn0), (jp, jp0))]
                    # The two-sided interface constitutive current is excluded
                    # explicitly; the ordinary SG law is not its replacement.
                    ordinary = [i for i in range(len(jn)) if i != excluded]
                    dphi = [dec(a)-dec(b) for a, b in zip(sample["phi_V"], base["phi_V"])]
                    cap, dt = list(map(dec, parameters["face_capacitance_F_m2"])), dec(data["dt_s"])
                    displacement = [-cap[0]*(dphi[1]-dphi[0])/dt, -cap[-1]*(dphi[-1]-dphi[-2])/dt]
                    conduction = [delta_n[i]+delta_p[i] for i in (0, len(jn)-1)]
                    total = [a+b for a, b in zip(conduction, displacement)]
                    ion = [dec(parameters["elementary_charge_C"])*(dec(a)-dec(b))
                           for a, b in zip(sample["ion_flux_m2_s"], base["ion_flux_m2_s"])]
                    values[key] = {"maximum_field_difference_V": max(map(abs, dphi)),
                        "maximum_ordinary_electron_current_change_A_m2": max(abs(delta_n[i]) for i in ordinary),
                        "maximum_ordinary_hole_current_change_A_m2": max(abs(delta_p[i]) for i in ordinary),
                        "maximum_ion_current_change_A_m2": max(map(abs, ion)),
                        "contact_conduction_change_A_m2": conduction,
                        "contact_displacement_change_A_m2": displacement,
                        "contact_total_change_A_m2": total,
                        "interface_carrier_current_change_A_m2": None,
                        "interpretation": "conditional_re_evaluation_not_an_accepted_corrected_state_or_total_error_bound"}
                case["by_precision"][str(digits)] = values
        report["cases"].append(case)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(serial(report), indent=2, allow_nan=False) + "\n")
    for case in report["cases"]:
        print(case["rows"], {key: serial(value["contact_total_change_A_m2"])
                               for key, value in case["by_precision"]["80"].items()})


if __name__ == "__main__":
    main()
