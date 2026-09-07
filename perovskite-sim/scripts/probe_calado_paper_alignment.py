#!/usr/bin/env python3
"""Diagnose the paper's preparation and contact boundaries without fitting."""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
import time
import warnings

from audit_ion_parameter_flow import ROOT, jsonable
import numpy as np
from threadpoolctl import threadpool_limits

from perovskite_sim.experiments import jv_sweep as jv
from perovskite_sim.experiments.waveform_jacobian import build_waveform_density_jacobian
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.physics.generation import dual_cell_widths
from perovskite_sim.constants import EPS_0, Q
from plot_calado_fig1f import branch_metrics, control_stack, uniform_generation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contacts", choices=("existing", "paper"), default="paper")
    parser.add_argument("--no-contact-srh", action="store_true")
    parser.add_argument("--n-grid", type=int, default=60)
    parser.add_argument("--uniform-nodes", type=int)
    parser.add_argument("--voltage-ramp", action="store_true")
    parser.add_argument("--analytic-jacobian", action="store_true")
    parser.add_argument("--seed-dark0-seconds", type=float, default=0.0)
    parser.add_argument("--prepare-ramp-seconds", type=float, default=0.0)
    parser.add_argument("--ion-diffusivity-scale", type=float, default=1.0)
    parser.add_argument("--ion-density-scale", type=float, default=1.0)
    parser.add_argument("--author-parameter-precision", action="store_true")
    parser.add_argument("--neutral-contact-ion-background", action="store_true")
    parser.add_argument("--light-on-initial-span", type=float, default=0.0)
    parser.add_argument("--light-on-dwell-seconds", type=float)
    parser.add_argument("--dv", type=float, default=0.02)
    parser.add_argument("--scan-rate", type=float, default=0.04)
    parser.add_argument("--scan-start-voltage", type=float, default=-1.0,
                        help="Diagnostic scan start, independent of dark preparation bias")
    parser.add_argument("--scan-end-voltage", type=float, default=1.2,
                        help="Diagnostic turnaround bias; does not change the production preset")
    parser.add_argument("--prepare-voltage", type=float, default=-1.0)
    parser.add_argument("--prepare-seconds", type=float, default=30.0)
    parser.add_argument("--hold-seconds", type=float, default=3.0)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.dv <= 0 or args.scan_rate <= 0 or args.prepare_seconds <= 0:
        parser.error("dv, scan rate and preparation time must be positive")
    if not np.all(np.isfinite([args.scan_start_voltage, args.scan_end_voltage])):
        parser.error("scan-start-voltage and scan-end-voltage must be finite")
    if args.scan_end_voltage <= max(0.0, args.scan_start_voltage):
        parser.error("scan-end-voltage must be positive and above scan-start-voltage")
    if round((args.scan_end_voltage - args.scan_start_voltage) / args.dv) < 1:
        parser.error("dv must resolve at least one scan interval")
    if args.uniform_nodes is not None and args.uniform_nodes < 3:
        parser.error("uniform-nodes must be at least 3")
    if args.light_on_dwell_seconds is not None and args.light_on_dwell_seconds <= 0:
        parser.error("light-on-dwell-seconds must be positive")
    if min(args.prepare_ramp_seconds, args.seed_dark0_seconds,
           args.ion_diffusivity_scale, args.ion_density_scale, args.light_on_initial_span) < 0:
        parser.error("history times and ion scales must be nonnegative")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stack = load_device_from_yaml(str(ROOT / "configs/calado2016_fig1f.yaml"))
    if args.author_parameter_precision:
        # c46f54b/pinParams.m equations, with the paper's ionic mobility.
        # The old charge used only for plotting currents is NOT adopted.
        thermal = 8.6173324e-5 * 300.0
        dos = 1e26
        intrinsic = dos * np.exp(-0.8 / thermal)
        doping = dos * np.exp(-0.15 / thermal)
        relative_eps = 20 * (552434 * Q * 100) / EPS_0
        exact_layers = []
        for layer, trap in zip(stack.layers, (-1.4, -0.8, -0.2), strict=True):
            params = layer.params
            exact_layers.append(dataclasses.replace(layer, params=dataclasses.replace(
                params, ni=intrinsic, n1=dos * np.exp(trap / thermal),
                p1=dos * np.exp((-1.6 - trap) / thermal), eps_r=relative_eps,
                N_A=doping if params.N_A else 0.0,
                N_D=doping if params.N_D else 0.0,
                D_ion=1e-16 * thermal if params.D_ion else 0.0,
                tau_n=1e100 if layer.role == "absorber" else params.tau_n,
                tau_p=1e100 if layer.role == "absorber" else params.tau_p,
            )))
        stack = dataclasses.replace(stack, layers=tuple(exact_layers))
    if args.neutral_contact_ion_background:
        population = next(layer.params.P0 for layer in stack.layers if layer.role == "absorber")
        stack = dataclasses.replace(stack, layers=tuple(
            dataclasses.replace(layer, params=dataclasses.replace(layer.params, P0=population))
            if layer.role != "absorber" else layer for layer in stack.layers
        ))
    stack = dataclasses.replace(stack, layers=tuple(
        dataclasses.replace(layer, params=dataclasses.replace(
            layer.params, D_ion=layer.params.D_ion * args.ion_diffusivity_scale,
            P0=layer.params.P0 * args.ion_density_scale,
        )) if layer.params is not None else layer for layer in stack.layers
    ))
    if args.no_contact_srh:
        stack = control_stack(stack)
    if args.contacts == "paper":
        # Full enables the existing mixed Robin/Dirichlet contact path.
        stack = dataclasses.replace(stack, mode="full", S_n_left=0.0, S_p_right=0.0)
    x = (
        jv.build_electrical_grid(stack, args.n_grid)
        if args.uniform_nodes is None
        else np.linspace(0.0, sum(layer.thickness for layer in stack.layers), args.uniform_nodes)
    )
    widths = dual_cell_widths(x)
    mat = dataclasses.replace(jv.build_material_arrays(x, stack), G_optical=uniform_generation(x, stack))
    summary = {
        "grade": "diagnostic; no spatial/time convergence certificate yet",
        "settings": vars(args),
        "mode": stack.mode,
        "actual_grid_nodes": len(x),
        "device_stack": jsonable(stack),
        "contacts": {key: getattr(mat, key) for key in ("S_n_L", "S_p_L", "S_n_R", "S_p_R")},
        "generation_budget_A_m2": float(Q * np.sum(mat.G_optical * widths)),
        "preparation": "fixed dark bias, then light on at scan start; dark high-bias hold",
        "sampling": (
            "continuous voltage ramps between samples; first point includes one light-on dwell"
            if args.voltage_ramp
            else "fixed-voltage staircase; first point includes one dwell after light-on"
        ),
        "jacobian_evaluator": "analytic_single_ion_density" if args.analytic_jacobian else "finite_difference",
        "code_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in (
            Path(__file__), ROOT / "perovskite_sim/experiments/waveform_jacobian.py",
            ROOT / "perovskite_sim/experiments/jv_sweep.py", ROOT / "perovskite_sim/solver/mol.py")},
    }
    arrays = {"x": x}
    started = time.monotonic()
    y = jv.solve_equilibrium(x, stack)
    active_bias = 0.0
    jacobian = (build_waveform_density_jacobian(x, stack, mat,
        lambda t: active_bias(t) if callable(active_bias) else active_bias)
        if args.analytic_jacobian else None)
    jacobian_kwargs = {"jacobian": jacobian} if jacobian is not None else {}

    def advance(voltage, seconds, illuminated):
        nonlocal y, active_bias
        active_bias = voltage
        summary["last_attempt"] = {
            "start_bias_V": float(voltage(0.0) if callable(voltage) else voltage),
            "end_bias_V": float(voltage(seconds) if callable(voltage) else voltage),
            "seconds": seconds, "illuminated": illuminated,
        }
        if callable(voltage):
            solved = jv.run_transient(
                x=x, y0=y, stack=stack, t_span=(0.0, seconds),
                t_eval=np.asarray([seconds]),
                V_app=voltage, mat=mat, illuminated=illuminated,
                rtol=args.rtol, atol=args.atol, max_step=seconds / 20,
                max_nfev=100_000,
                **jacobian_kwargs,
            )
            if not solved.success:
                raise RuntimeError(f"voltage-ramp step failed: {solved.message}")
            y = solved.y[:, -1]
            return
        y = jv._integrate_step(
            x, y, stack, mat, voltage, 0.0, seconds, args.rtol, args.atol,
            illuminated=illuminated,
            **jacobian_kwargs,
        )

    def sweep(name, start, end):
        voltage = np.linspace(start, end, int(round(abs(end - start) / args.dv)) + 1)
        dt = abs(voltage[1] - voltage[0]) / args.scan_rate
        current = np.empty_like(voltage)
        profiles = {key: [] for key in ("P", "n", "p", "phi")}
        previous_voltage = start
        for index, value in enumerate(voltage):
            summary["stage"] = f"{name} sample {index}"
            duration = args.light_on_dwell_seconds if index == 0 and args.light_on_dwell_seconds is not None else dt
            previous_state = y.copy()
            bias = (
                (lambda t, left=previous_voltage, right=float(value): left + (right-left) * t/dt)
                if args.voltage_ramp and value != previous_voltage
                else float(value)
            )
            if index == 0 and 0 < args.light_on_initial_span < duration:
                # Same instantaneous light-on and total physical dwell;
                # only partition the integration span around fast carriers.
                advance(bias, args.light_on_initial_span, True)
                advance(bias, float(duration - args.light_on_initial_span), True)
            else:
                advance(bias, float(duration), True)
            current[index] = jv._compute_current(
                x, y, stack, float(value), y_prev=previous_state, dt=duration,
                mat=mat, V_app_prev=previous_voltage,
            )
            if not np.isfinite(current[index]):
                raise RuntimeError(f"nonfinite {name} current at {value}")
            snapshot = jv.extract_spatial_snapshot(x, y, stack, float(value), mat=mat)
            for key in profiles:
                profiles[key].append(getattr(snapshot, key).copy())
            arrays[f"V_{name}"] = voltage[:index + 1].copy()
            arrays[f"J_{name}"] = current[:index + 1].copy()
            for key, values in profiles.items():
                arrays[f"{key}_{name}"] = np.stack(values)
            previous_voltage = float(value)
            if index % 10 == 0:
                print(f"{name} {index + 1}/{len(voltage)} V={value:.3f} J={current[index]:.6g}", flush=True)
        arrays[f"V_{name}"] = voltage
        arrays[f"J_{name}"] = current
        for key, values in profiles.items():
            arrays[f"{key}_{name}"] = np.stack(values)
        return branch_metrics(voltage, current)

    exit_code = 0
    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1, user_api="blas"):
        warnings.simplefilter("always")
        try:
            if args.seed_dark0_seconds > 0:
                summary["stage"] = "dark0 seed"
                print(f"initial dark 0 V history: {args.seed_dark0_seconds} s", flush=True)
                advance(0.0, args.seed_dark0_seconds, False)
            if args.prepare_ramp_seconds > 0:
                summary["stage"] = "dark preparation voltage ramp"
                print(f"dark preparation bias ramp: {args.prepare_ramp_seconds} s", flush=True)
                advance(lambda t: args.prepare_voltage * t / args.prepare_ramp_seconds,
                        args.prepare_ramp_seconds, False)
            print(f"dark preparation: {args.prepare_voltage} V, {args.prepare_seconds} s", flush=True)
            summary["stage"] = "dark preparation hold"
            advance(args.prepare_voltage, args.prepare_seconds, False)
            arrays["y_prepared"] = y.copy()
            arrays["P_prepared"] = jv.StateVec.unpack(y, len(x)).P.copy()
            summary["prepared_inventory_m2"] = float(np.sum(arrays["P_prepared"] * widths))
            forward = sweep("fwd", args.scan_start_voltage, args.scan_end_voltage)
            if args.hold_seconds > 0:
                summary["stage"] = "dark high-bias hold"
                advance(args.scan_end_voltage, args.hold_seconds, False)
            reverse = sweep("rev", args.scan_end_voltage, args.scan_start_voltage)
            summary.update({
                "status": "completed", "stage": "completed", "forward": forward, "reverse": reverse,
                "HI_paper": reverse.P_max / forward.P_max - 1.0,
            })
        except Exception as exc:
            summary.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
            exit_code = 1
        summary["warnings"] = [str(item.message) for item in caught]
    summary["settings"] = {**vars(args), "out_dir": str(args.out_dir)}
    summary["elapsed_s"] = time.monotonic() - started
    np.savez_compressed(args.out_dir / "curves.npz", **arrays)
    with (args.out_dir / "alignment.json").open("w") as stream:
        json.dump(jsonable(summary), stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(jsonable({key: summary[key] for key in ("status", "HI_paper", "elapsed_s", "error") if key in summary})), flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
