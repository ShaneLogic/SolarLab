#!/usr/bin/env python3
"""Audit captured frontend spatial/J-V runs without fitting or running a solver."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.main import stack_from_dict
from perovskite_sim.constants import Q
from perovskite_sim.experiments.waveform_jv import JVWaveform, build_waveform_protocol, voltage_samples
from perovskite_sim.experiments.jv_sweep import build_electrical_grid
from perovskite_sim.physics.generation import dual_cell_widths
from perovskite_sim.models.device import electrical_layers
from perovskite_sim.solver.mol import build_material_arrays


def snapshots(record):
    data = record["result"]["data"]
    frames = []
    for branch in ("fwd", "rev"):
        voltage = data[f"V_{branch}"]
        branch_frames = data[f"snapshots_{branch}"]
        if len(voltage) != len(branch_frames) or not branch_frames:
            raise ValueError(f"Missing {branch} snapshots")
        np.testing.assert_allclose([frame["V_app"] for frame in branch_frames], voltage, rtol=0, atol=1e-12)
        frames.extend(branch_frames)
    arrays = {name: np.asarray([frame[name] for frame in frames], dtype=float)
              for name in ("x", "phi", "E", "n", "p", "P", "rho")}
    negative_present = ["P_neg" in frame for frame in frames]
    if any(negative_present):
        if not all(negative_present):
            raise ValueError("Negative-ion snapshots are incomplete")
        arrays["P_neg"] = np.asarray([frame["P_neg"] for frame in frames], dtype=float)
    if any(not np.all(np.isfinite(value)) for value in arrays.values()):
        raise ValueError("Nonfinite spatial result")
    return arrays


def population_summary(population, reference, diffusivity, widths):
    inventory = population @ widths
    reference_inventory = float(reference @ widths)
    if not np.any(diffusivity):
        np.testing.assert_array_equal(population, np.broadcast_to(reference, population.shape))
    if reference_inventory == 0:
        np.testing.assert_array_equal(population, np.zeros_like(population))
    drift = float(np.max(np.abs(inventory - reference_inventory))) / max(abs(reference_inventory), 1.0)
    if drift > 1e-6:
        raise ValueError("Saved ionic inventory is not conserved")
    return {"reference_max_m3": float(np.max(reference)),
        "diffusivity_max_m2_s": float(np.max(diffusivity)),
        "minimum_m3": float(np.min(population)), "maximum_m3": float(np.max(population)),
        "redistribution_Linf_over_reference": float(np.max(np.abs(population - reference))) / max(float(np.max(reference)), 1.0),
        "reference_inventory_m2": reference_inventory, "inventory_relative_drift": drift}


def expected_generation_budget(stack, waveform, mat, x, illuminated):
    if not illuminated:
        return 0.0
    if waveform.uniform_generation_rate_m3_s is not None:
        thickness = sum(layer.thickness for layer in electrical_layers(stack) if layer.role == "absorber")
        return Q * waveform.uniform_generation_rate_m3_s * thickness
    if mat.G_optical is not None:
        return float(Q * (mat.G_optical @ dual_cell_widths(x)))
    # Independent closed-form photon balance for the compiled optical depth.
    depth = float(np.sum(0.5 * (mat.alpha[:-1] + mat.alpha[1:]) * np.diff(x)))
    return float(Q * stack.Phi * -np.expm1(-depth))


def current_summary(data):
    result = {}
    for branch in ("fwd", "rev"):
        voltage, current = np.asarray(data[f"V_{branch}"]), np.asarray(data[f"J_{branch}"])
        if voltage.ndim != 1 or voltage.shape != current.shape or not np.all(np.isfinite(current)):
            raise ValueError("Invalid J-V current samples")
        order = np.argsort(voltage)
        if not (voltage.min() <= 0 <= voltage.max()):
            raise ValueError("Jsc is outside the saved voltage range")
        result[f"J_sc_{branch}_A_m2"] = float(np.interp(0, voltage[order], current[order]))
        positive = voltage >= 0
        result[f"P_max_{branch}_W_m2"] = float(np.max(voltage[positive] * current[positive]))
    if data["hysteresis_index_paper"] is not None:
        if result["P_max_fwd_W_m2"] <= 0:
            raise ValueError("A finite HI requires a positive forward maximum power")
        expected_hi = result["P_max_rev_W_m2"] / result["P_max_fwd_W_m2"] - 1
        np.testing.assert_allclose(data["hysteresis_index_paper"], expected_hi, rtol=1e-10, atol=1e-12)
    return result


def audit_record(filename):
    raw = Path(filename).read_bytes()
    record = json.loads(raw)
    kind = record["request"]["kind"]
    if not record.get("job_id") or kind not in ("spatial", "jv"):
        raise ValueError("Expected a captured spatial or J-V job with its job ID")
    if record["result"]["kind"] != kind or record["result"]["device"] != record["request"]["device"]:
        raise ValueError("Submitted device does not match the completed run snapshot")
    params, data = record["request"]["params"], record["result"]["data"]
    stack = stack_from_dict(record["request"]["device"])
    waveform = JVWaveform.from_dict(params["waveform"])
    protocol = build_waveform_protocol(stack, waveform, n_points=params["n_points"],
        v_rate=params["v_rate"], V_max=params["V_max"], illuminated=params["illuminated"])
    if data["waveform_protocol_sha256"] != protocol.protocol_hash:
        raise ValueError("Result protocol differs from the submitted waveform")
    if data["waveform"] != params["waveform"]:
        raise ValueError("Result waveform differs from the submitted waveform")
    expected_v = voltage_samples(waveform, params["n_points"], params["v_rate"], params["V_max"])
    np.testing.assert_allclose(data["V_fwd"], expected_v, rtol=0, atol=1e-12)
    np.testing.assert_allclose(data["V_rev"], expected_v[::-1], rtol=0, atol=1e-12)
    x = build_electrical_grid(stack, params["N_grid"])
    expected_controls = {"rtol": params["waveform_controls"]["rtol"],
        "atol_m3": params["waveform_controls"]["atol_m3"],
        "requested_N_grid": params["N_grid"], "actual_N_grid": len(x)}
    if data["numerical_controls"] != expected_controls:
        raise ValueError("Result numerical controls differ from the submitted request")
    mat = build_material_arrays(x, stack)
    expected_budget = expected_generation_budget(stack, waveform, mat, x, params["illuminated"])
    np.testing.assert_allclose(data["generation_budget_A_m2"], expected_budget, rtol=1e-12, atol=1e-12)
    if stack.interface_charge_closure != "off":
        raise ValueError("This audit does not cover charged interface states")
    row = {
        "file": str(Path(filename).resolve()), "sha256": hashlib.sha256(raw).hexdigest(),
        "evidence_origin": record.get("evidence_origin", "unspecified_in_record"),
        "job_id": record["job_id"], "mode": stack.mode,
        "HI_paper": data["hysteresis_index_paper"],
        "scan_rate_V_s": params["v_rate"],
        "D_c_max_m2_s": float(np.max(mat.D_ion_node)),
        "c0_max_m3": float(np.max(mat.P_ion0)),
        "negative_state_active": bool(mat.has_dual_ions),
        "a0_configured_max_m3": max(layer.params.P0_neg for layer in electrical_layers(stack)),
        "photon_flux_m2_s": stack.Phi,
        "alpha_max_m1": float(np.max(mat.alpha)),
        "generation_source": "device_optics" if waveform.uniform_generation_rate_m3_s is None else "uniform_absorber",
        "driver_inventory_relative_drift": data["inventory_relative_drift"],
        "generation_budget_A_m2": data["generation_budget_A_m2"],
        "expected_generation_budget_A_m2": expected_budget,
        "protocol_sha256": protocol.protocol_hash, "numerical_controls": data["numerical_controls"],
        "numerical_scope": data["numerical_scope"], "samples_per_branch": len(data["V_fwd"]),
        "jacobian_evaluator": data.get("jacobian_evaluator", "unspecified_in_record"),
        "jacobian_fallback_reason": data.get("jacobian_fallback_reason"),
    }
    if kind == "spatial":
        arrays, widths = snapshots(record), dual_cell_widths(x)
        expected_x = np.broadcast_to(x * 1e9, arrays["x"].shape)
        np.testing.assert_allclose(arrays["x"], expected_x, rtol=1e-12, atol=1e-12)
        if mat.has_dual_ions != ("P_neg" in arrays):
            raise ValueError("Negative-ion state activation does not match the saved snapshots")
        positive = population_summary(arrays["P"], mat.P_ion0, mat.D_ion_node, widths)
        row.update(minimum_c_m3=positive["minimum_m3"], maximum_c_m3=positive["maximum_m3"],
            redistribution_Linf_over_c0=positive["redistribution_Linf_over_reference"],
            reference_inventory_m2=positive["reference_inventory_m2"],
            snapshot_inventory_relative_drift=positive["inventory_relative_drift"])
        if mat.has_dual_ions:
            row["negative_species"] = population_summary(arrays["P_neg"], mat.P_ion0_neg, mat.D_ion_neg_node, widths)
        computed_rho = Q * (arrays["p"] - arrays["n"] + (arrays["P"] - mat.P_ion0) - mat.N_A + mat.N_D)
        if mat.has_dual_ions:
            computed_rho -= Q * (arrays["P_neg"] - mat.P_ion0_neg)
        np.testing.assert_allclose(arrays["rho"], computed_rho, rtol=1e-12, atol=1e-8)
        row["charge_closure_error_C_m3"] = float(np.max(np.abs(arrays["rho"] - computed_rho)))
        row["branch_memory_at_0p4V"] = branch_memory(record, stack, x, widths)
    else:
        row.update(current_summary(data))
    return row, record


def branch_memory(record, stack, x, widths, voltage=0.4):
    edges = np.r_[x[0], x[0] + np.cumsum([layer.thickness for layer in electrical_layers(stack)])]
    active = [index for index, layer in enumerate(electrical_layers(stack)) if layer.role == "absorber"]
    if len(active) != 1:
        return None
    left, right = edges[active[0]:active[0] + 2]
    result = {"V_app": voltage}
    data = record["result"]["data"]
    for branch in ("fwd", "rev"):
        bias = np.asarray(data[f"V_{branch}"])
        index = int(np.argmin(np.abs(bias - voltage)))
        if abs(bias[index] - voltage) > 1e-9:
            return None
        frame = data[f"snapshots_{branch}"][index]
        potential_drop = np.interp(right, x, frame["phi"]) - np.interp(left, x, frame["phi"])
        population = np.asarray(frame["P"])
        inventory = float(population @ widths)
        result[branch] = {
            "absorber_mean_field_V_m": float(-potential_drop / (right - left)),
            "ion_centroid_nm": float((population * x) @ widths / inventory * 1e9) if inventory > 0 else None,
        }
        if "P_neg" in frame:
            negative = np.asarray(frame["P_neg"])
            negative_inventory = float(negative @ widths)
            result[branch]["negative_ion_centroid_nm"] = float((negative * x) @ widths / negative_inventory * 1e9) if negative_inventory > 0 else None
    result["reverse_minus_forward_mean_field_V_m"] = result["rev"]["absorber_mean_field_V_m"] - result["fwd"]["absorber_mean_field_V_m"]
    return result


def compare_neutral_controls(frozen, empty):
    if frozen["request"]["params"] != empty["request"]["params"]:
        raise ValueError("Control protocols or numerical settings differ")
    left, right = snapshots(frozen), snapshots(empty)
    controls = frozen["request"]["params"].get("waveform_controls", {})
    density_rtol, density_atol = controls.get("rtol", 1e-4), controls.get("atol_m3", 1.0)
    limits = {"x": (0, 0), "phi": (0, 1e-10), "E": (1e-10, 1e-3),
              "n": (density_rtol, density_atol), "p": (density_rtol, density_atol),
              "rho": (1e-10, 1e-8)}
    delta = {}
    within_tolerance = {}
    for name in ("x", "phi", "E", "n", "p", "rho"):
        delta[name] = float(np.max(np.abs(left[name] - right[name])))
        relative, absolute = limits[name]
        within_tolerance[name] = bool(np.allclose(left[name], right[name], rtol=relative, atol=absolute))
    hi_delta = abs(frozen["result"]["data"]["hysteresis_index_paper"] - empty["result"]["data"]["hysteresis_index_paper"])
    if hi_delta != 0:
        raise ValueError("Frozen neutral and empty controls have different HI")
    return {"maximum_absolute_state_difference": delta, "HI_absolute_difference": hi_delta,
            "within_diagnostic_tolerance": within_tolerance,
            "diagnostic_tolerances_rtol_atol": limits,
            "scope": "Control comparison only; not a spatial or time convergence certificate"}


def validate_matrix(records, vary):
    if vary not in ("ions", "scan_rate", "optics"):
        raise ValueError("Unsupported matrix axis")
    baseline = None
    for record in records.values():
        request = copy.deepcopy(record["request"])
        if vary == "ions":
            for layer in request["device"]["layers"]:
                for name in ("P0", "P0_neg", "D_ion", "D_ion_neg"):
                    layer.pop(name, None)
        elif vary == "scan_rate":
            request["params"].pop("v_rate")
        else:
            request["device"]["device"].pop("Phi", None)
            for layer in request["device"]["layers"]:
                layer.pop("alpha", None)
        if baseline is None:
            baseline = request
        elif request != baseline:
            raise ValueError(f"Matrix inputs differ outside the declared {vary} axis")


def render_plot(summary, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = summary["cases"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    names = list(rows)
    if summary["varied_axis"] == "optics":
        budget = [rows[name]["generation_budget_A_m2"] for name in names]
        bars = axes[0].bar(names, budget, color="#247a89")
        axes[0].bar_label(bars, fmt="%.3g", padding=3)
        axes[0].set(ylabel="q integral(G dx) (A/m2)", title="Absorbed-photon budget")
        axes[0].margins(y=0.15)
        axes[0].tick_params(axis="x", labelrotation=35)
        for branch, color in (("fwd", "#247a89"), ("rev", "#a64d63")):
            axes[1].scatter(budget, [rows[name][f"J_sc_{branch}_A_m2"] for name in names], label=branch, color=color)
        limit = max(budget) * 1.05
        axes[1].plot([0, limit], [0, limit], color="#555555", linestyle="--", label="unit collection")
        axes[1].set(xlabel="Absorbed-photon budget (A/m2)", ylabel="Jsc (A/m2)", title="Terminal photocurrent response")
        axes[1].legend()
        for axis in axes:
            axis.grid(axis="y", alpha=0.2)
        fig.savefig(destination, dpi=180)
        plt.close(fig)
        return
    bars = axes[0].bar(names, [rows[name]["HI_paper"] for name in names], color="#247a89")
    axes[0].bar_label(bars, fmt="%.3g", padding=3)
    axes[0].set(ylabel="HI = Pmax,rev / Pmax,fwd - 1", title=(
        "Same scan history" if summary["varied_axis"] == "ions" else "Scan rate varies\nPreparation and holds fixed"))
    axes[0].margins(y=0.15)
    if all(rows[name].get("branch_memory_at_0p4V") is not None for name in names):
        field_difference = [abs(rows[name]["branch_memory_at_0p4V"]["reverse_minus_forward_mean_field_V_m"]) * 1e-6 for name in names]
        axes[1].bar(names, field_difference, color="#a64d63")
        axes[1].set(ylabel="|E_rev - E_fwd| (MV/m)", title="Absorber mean field difference at 0.4 V")
    else:
        axes[1].bar(names, [rows[name]["redistribution_Linf_over_c0"] for name in names], color="#a64d63")
        axes[1].set(ylabel="max |c - c0| / max(c0)", title="Positive-ion redistribution")
    for axis in axes:
        axis.tick_params(axis="x", labelrotation=35)
        axis.grid(axis="y", alpha=0.2)
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", nargs=2, action="append", required=True, metavar=("NAME", "JSON"))
    parser.add_argument("--vary", choices=("ions", "scan_rate", "optics"), default="ions")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    rows, records = {}, {}
    for name, filename in args.case:
        if name in rows:
            parser.error(f"Duplicate case: {name}")
        rows[name], records[name] = audit_record(filename)
    validate_matrix(records, args.vary)
    if args.vary != "scan_rate" and len({row["protocol_sha256"] for row in rows.values()}) != 1:
        raise ValueError("Matrix cases must use the same physical scan history and voltage samples")
    if len({json.dumps(row["numerical_controls"], sort_keys=True) for row in rows.values()}) != 1:
        raise ValueError("Matrix cases must use the same numerical controls")
    summary = {"scope": "Frontend parameter-flow diagnosis, not a completed paper reproduction or convergence certificate", "varied_axis": args.vary, "cases": rows}
    if "frozen" in records and "empty" in records:
        summary["neutral_control_comparison"] = compare_neutral_controls(records["frozen"], records["empty"])
    args.out_dir.mkdir(parents=True, exist_ok=False)
    (args.out_dir / "matrix.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    render_plot(summary, args.out_dir / "matrix.png")
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
