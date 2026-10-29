"""One E37 split-stage attempt, then I37 only after numerical E completion.

Explicit invocation only. The original windows, 4/8 grids, drives and controls
remain fixed. A numerical completion does not qualify a reference or turn a
refinement difference into an error estimate. Every result stays on disk.

Integration conditions: optional, unqualified reference infrastructure for
restricted declared E/I inputs and analytic unit fixtures only. Independence
is static-only: no BDF stepping, history, interpolation or accounting is used.
Retained failed attempts stay failures and qualification flags stay false.
Any future nonzero initial uncertainty must be expressed in the selected
weighted log norm; a numerical trajectory alone does not qualify a reference.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from fractions import Fraction
from hashlib import sha256
import json
import math
from pathlib import Path
import runpy
from typing import Any

from flint import arb, ctx, fmpq
import numpy as np

from examples.research_1d.ionic_multilayer import prepare_confined_ion_device
from examples.research_1d.ionic_steady import controls as ionic_controls
from examples.research_1d.steady import load_device
from solarlab.device.compile_1d import compile_device_1d
from solarlab.device.compile_ionic_1d import compile_ionic_device_1d
from solarlab.equations.assembly.classical_1d import AssembledDC1DProblem
from solarlab.equations.assembly.ionic_1d import AssembledIonic1DProblem
from solarlab.solvers.nonlinear.ionic_1d import solve_ionic_steady_state
from solarlab.solvers.time.radau_enclosure import NonlinearInventoryEnclosure, PWLDrive
from solarlab.solvers.time.radau_reference import RadauControls, assembled_inventory_dae
from solarlab.solvers.time.radau_split import (
    SplitRadauFailure,
    SplitRadauModel,
    SplitRadauStep,
    SplitRadauTrajectory,
    integrate_split_radau,
)

# The already tested local budget/JSON helpers contain no temporal algorithm.
_helpers = runpy.run_path(str(Path(__file__).with_name("radau_short_windows.py")))
Budget, serializable = _helpers["Budget"], _helpers["serializable"]


def pin(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
    }


def step_arrays(steps: tuple[SplitRadauStep, ...]) -> dict[str, Any]:
    return {
        "step_times": np.array([(s.left_s, s.right_s) for s in steps]),
        "coefficients_high": np.array([s.coefficients_high for s in steps]),
        "coefficients_low": np.array([s.coefficients_low for s in steps]),
        "coefficient_encoding_error_upper": np.array(
            [s.coefficient_encoding_error_upper for s in steps]
        ),
        "stage_departures": np.array([s.stage_departures for s in steps]),
        "scaled_stage_residual": np.array([s.scaled_stage_residual for s in steps]),
        "scaled_stage_residual_upper": np.array(
            [s.scaled_stage_residual_upper for s in steps]
        ),
        "stage_u": np.array([[p.u for p in s.stages] for s in steps]),
        "stage_inventory_high": np.array(
            [[p.inventory.high for p in s.stages] for s in steps]
        ),
        "stage_inventory_low": np.array(
            [[p.inventory.low for p in s.stages] for s in steps]
        ),
        "stage_inventory_encoding_error_upper": np.array(
            [[p.inventory_encoding_error_upper for p in s.stages] for s in steps]
        ),
        "stage_static_residual_upper": np.array(
            [[p.scaled_reconstruction_residual for p in s.stages] for s in steps]
        ),
        "endpoint_u": np.array([s.endpoint.u for s in steps]),
        "endpoint_inventory_high": np.array([s.endpoint.inventory.high for s in steps]),
        "endpoint_inventory_low": np.array([s.endpoint.inventory.low for s in steps]),
        "endpoint_static_residual_upper": np.array(
            [s.endpoint.scaled_reconstruction_residual for s in steps]
        ),
    }


def save_trajectory(path: Path, trajectory: SplitRadauTrajectory) -> dict[str, Any]:
    np.savez_compressed(
        path,
        initial_u=trajectory.initial.u,
        initial_inventory_high=trajectory.initial.inventory.high,
        initial_inventory_low=trajectory.initial.inventory.low,
        component_scale=trajectory.model.scale,
        **step_arrays(trajectory.steps),
    )
    return {
        **pin(path),
        "steps": len(trajectory.steps),
        "counters": asdict(trajectory.counters),
        "max_scaled_stage_residual": max(
            s.scaled_stage_residual_upper for s in trajectory.steps
        ),
        "max_scaled_reconstruction_residual": max(
            p.scaled_reconstruction_residual
            for s in trajectory.steps
            for p in (*s.stages, s.endpoint)
        ),
        "qualified": False,
    }


def run_case(
    scope: str, old: dict[str, Any], output: Path, budget: Any
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "scope": scope,
        "electrical_nodes": 37,
        "status": "reference_blocked",
        "numerical_short_window_completed": False,
        "tube_proved": False,
        "error_bound": None,
        "observed_refinement_delta": None,
        "histories": [],
        "qualification": False,
        "admission_run": False,
    }
    try:
        if scope == "E_L":
            e = compile_device_1d(load_device(), intervals_per_layer=(12, 12, 12))
            problem: Any = AssembledDC1DProblem(e, 0.0, 1.0)
            if (
                old["device_identity"] != e.identity
                or old["initial"]["accepted"] is not True
            ):
                raise ValueError("saved_E_initial_identity_or_acceptance_mismatch")
            u = np.asarray(old["initial"]["newton"]["state"], dtype=float)
            result["initial"] = {
                "accepted": True,
                "reuse": "exact accepted Short01 E37 packet",
                "u": u.tolist(),
            }
        else:
            ion = compile_ionic_device_1d(
                prepare_confined_ion_device(),
                intervals_per_layer=(12, 12, 12),
                ion_domain_layer="absorber",
                ion_domain_endpoints=("blocking", "blocking"),
            )
            problem = AssembledIonic1DProblem(ion, 0.002, 0.0)
            if old["device_identity"] != ion.identity:
                raise ValueError("saved_I_device_identity_mismatch")
            prepared = budget.call(
                "I_V:initial",
                lambda: solve_ionic_steady_state(
                    ion,
                    voltage_V=0.002,
                    generation_scale=0.0,
                    options=ionic_controls(ion.ion_node_count),
                ),
            )
            result["initial"] = {
                "accepted": prepared.accepted,
                "termination": prepared.termination,
                "checks": serializable(prepared.checks),
                "error": prepared.error,
                "reuse": "new original-control preparation; Short01 I initial coordinates were not retained",
            }
            if not prepared.accepted or prepared.state is None:
                result["reason"] = "I_initial_preparation_failed"
                return result
            u = prepared.state.u
            result["initial"]["u"] = u.tolist()
            result["ion_nodes"] = ion.ion_node_count
        duration = 1e-8 if scope == "E_L" else 1e-4
        result["short_window_s"] = [0.0, duration]
        drive = PWLDrive(**{key: tuple(value) for key, value in old["drive"].items()})
        result["drive"] = asdict(drive)
        result["drive_hex"] = {
            key: [value.hex() for value in values]
            for key, values in asdict(drive).items()
        }
        result["device_identity"] = problem.device.identity
        mask = problem.binding.layout.constraint_mask()
        mask[problem.binding.layout.rows("potential")] = True
        z0 = problem.storage(u)[~mask]
        scale = np.maximum(np.abs(z0), 1.0)
        if scope == "E_L" and not np.array_equal(
            scale, np.asarray(old["component_scale"])
        ):
            raise ValueError("saved_E_scale_mismatch")
        result["component_scale"] = scale.tolist()
        result["initial_inventory_high"] = z0.tolist()
        result["initial_inventory_low"] = np.zeros_like(z0).tolist()
        result["initial_error_semantics"] = (
            "exact encoded initial-inventory IVP only; departure from a true steady state is not enclosed"
        )
        static = assembled_inventory_dae(
            problem,
            drive,
            inventory_scale=scale,
            algebraic_scale=np.ones(np.count_nonzero(mask)),
        )
        model = SplitRadauModel(
            static, NonlinearInventoryEnclosure(problem, drive, scale)
        )
        controls = RadauControls()
        result["controls"] = asdict(controls)
        histories = []
        for count in (4, 8):
            try:
                history = budget.call(
                    scope + f":split_radau_{count}",
                    lambda: integrate_split_radau(
                        model,
                        u,
                        np.linspace(0.0, duration, count + 1),
                        controls=controls,
                    ),
                )
            except SplitRadauFailure as error:
                path = output / (scope + f"_{count}_failed.npz")
                np.savez_compressed(
                    path,
                    initial_u=u,
                    initial_inventory_high=z0,
                    component_scale=scale,
                    failed_start_high=error.step_start.high,
                    failed_start_low=error.step_start.low,
                    failed_trial_departures=(
                        np.empty((0, scale.size))
                        if error.trial_departures is None
                        else error.trial_departures
                    ),
                    **step_arrays(error.completed_steps),
                )
                result["failure"] = {
                    "code": error.code,
                    "cause": str(error.__cause__),
                    "failed_step": error.failed_step,
                    "completed_steps": len(error.completed_steps),
                    "time_s": error.time_s,
                    "scaled_residual_upper": error.scaled_residual_upper,
                    "trial_available": error.trial_departures is not None,
                    "counters": asdict(error.counters)
                    if error.counters is not None
                    else None,
                    "artifact": pin(path),
                }
                result["reason"] = "split_Radau_numerical_prefix_failed"
                return result
            histories.append(history)
            result["histories"].append(
                save_trajectory(output / (scope + f"_{count}.npz"), history)
            )

        def observations() -> dict[str, Any]:
            times = np.linspace(0.0, duration, 9)
            difference = 0.0
            output_norm = 0.0
            saved: dict[str, Any] = {"times_s": times}
            with ctx.workprec(192):
                for count, history in zip((4, 8), histories):
                    points = [history.sample(float(t)) for t in times]
                    saved[f"u_{count}"] = np.array([p.u for p in points])
                    saved[f"inventory_high_{count}"] = np.array(
                        [p.inventory.high for p in points]
                    )
                    saved[f"inventory_low_{count}"] = np.array(
                        [p.inventory.low for p in points]
                    )
                    saved[f"inventory_encoding_error_upper_{count}"] = np.array(
                        [p.inventory_encoding_error_upper for p in points]
                    )
                    output_norm = max(
                        output_norm, *(p.scaled_reconstruction_residual for p in points)
                    )
                for t in times:
                    steps = [
                        next(s for s in history.steps if s.left_s <= t <= s.right_s)
                        for history in histories
                    ]
                    left, right = (step.polynomial(float(t)) for step in steps)
                    for a, b, s in zip(left, right, scale):
                        fraction = Fraction(float(s))
                        value = abs(a - b) / arb(
                            fmpq(fraction.numerator, fraction.denominator)
                        )
                        upper = float(value.upper())
                        difference = max(
                            difference,
                            0.0 if value.is_zero() else math.nextafter(upper, math.inf),
                        )
            path = output / (scope + "_outputs.npz")
            np.savez_compressed(path, **saved)
            return {
                "artifact": pin(path),
                "refinement_delta_upper_at_9_clocks": difference,
                "max_scaled_output_reconstruction_residual": output_norm,
                "meaning": "upper bound on the two encoded polynomials' sampled difference; not error relative to the unknown exact device solution",
            }

        observed = budget.call(scope + ":outputs", observations)
        result["observations"] = observed
        result["observed_refinement_delta"] = observed[
            "refinement_delta_upper_at_9_clocks"
        ]
        result["numerical_short_window_completed"] = True
        proof = budget.call(
            scope + ":fine_tube",
            lambda: histories[-1].bound(
                initial_error=0.0, tube_radius=1e-6, subdivisions=2
            ),
        )
        result["bound_attempt"] = asdict(proof)
        result["tube_proved"], result["error_bound"] = (
            proof.tube_proved,
            proof.error_upper,
        )
        result["status"], result["reason"] = proof.status, proof.reason
        return result
    except Exception as error:
        result["reason"] = type(error).__name__ + ": " + str(error)
        return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, required=True)
    parser.add_argument("--previous-result", type=Path, required=True)
    parser.add_argument("--initial-packet", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise FileExistsError("preserve every prior attempt")
    if (
        sha256(arguments.previous_result.read_bytes()).hexdigest()
        != "5cf571ac4abb0956249dd64da1666472460fa68f8d545544df5ce40bc6ac03d6"
    ):
        raise ValueError("unrecognized_previous_stage2_result")
    previous = json.loads(arguments.previous_result.read_text())
    expected = next(
        row["packet"]["sha256"]
        for row in previous["short_window_table"]
        if row["attempt"] == "Short01"
    )
    if sha256(arguments.initial_packet.read_bytes()).hexdigest() != expected:
        raise ValueError("unrecognized_saved_initial_packet")
    packet = json.loads(arguments.initial_packet.read_text())
    root = Path(__file__).resolve().parents[4]
    sources = [
        "src/solarlab/solvers/time/radau_reference.py",
        "src/solarlab/solvers/time/radau_enclosure.py",
        "src/solarlab/solvers/time/radau_split.py",
        "tests/unit/solvers/time/radau_short_windows.py",
        "tests/unit/solvers/time/radau_split_short_windows.py",
    ]
    pins = {p: sha256((root / p).read_bytes()).hexdigest() for p in sources}
    if pins[sources[0]] != packet["source_pins"][sources[0]]:
        raise ValueError("original_Radau_source_changed")
    if (
        not 0
        < arguments.seconds
        <= min(240.0, 300.0 - previous["budget"]["rounded_charge_s"])
    ):
        raise ValueError("requested_budget_exceeds_remaining_allowance")
    arguments.output.mkdir(parents=True)
    ctx.threads = 1
    budget = Budget(arguments.seconds)
    report: dict[str, Any] = {
        "schema": "solarlab.radau_split_short_windows.v1",
        "source_pins": pins,
        "previous_result": pin(arguments.previous_result),
        "initial_packet": pin(arguments.initial_packet),
        "solver_wall_cap_s": budget.limit,
        "previous_conservative_charge_s": previous["budget"]["rounded_charge_s"],
        "controls": asdict(RadauControls()),
        "qualification": False,
        "admission_runs": 0,
        "cases": [],
        "conditional_I_rule": "I37 only after both E37 grids and all sampled static/output checks complete; tube failure does not masquerade as qualification",
        "planned": [
            {"scope": "E_L", "electrical_nodes": 37, "end_s": 1e-8, "steps": [4, 8]},
            {
                "scope": "I_V",
                "electrical_nodes": 37,
                "ion_nodes": 13,
                "end_s": 1e-4,
                "steps": [4, 8],
                "conditional": True,
            },
        ],
        "tube": {
            "radius": 1e-6,
            "subdivisions": 2,
            "initial_error": 0.0,
            "initial_error_scope": "encoded unit IVP",
        },
    }
    (arguments.output / "Plan.json").write_text(json.dumps(report, indent=2) + "\n")
    for scope in ("E_L", "I_V"):
        if (
            scope == "I_V"
            and not report["cases"][0]["numerical_short_window_completed"]
        ):
            report["cases"].append(
                {
                    "scope": "I_V",
                    "status": "reference_blocked",
                    "reason": "not_started_E37_numerical_condition_not_met",
                    "qualification": False,
                }
            )
            break
        old = next(
            c
            for c in packet["cases"]
            if c["scope"] == scope and c["electrical_nodes"] == 37
        )
        result = run_case(scope, old, arguments.output, budget)
        report["cases"].append(result)
        report["solver_wall_s"], report["operations"] = budget.used, budget.calls
        (arguments.output / "ShortWindows.json").write_text(
            json.dumps(serializable(report), indent=2, allow_nan=False) + "\n"
        )
        print(
            json.dumps(
                {
                    k: v
                    for k, v in result.items()
                    if k
                    in {
                        "scope",
                        "status",
                        "reason",
                        "numerical_short_window_completed",
                        "tube_proved",
                        "error_bound",
                        "observed_refinement_delta",
                        "bound_attempt",
                    }
                },
                indent=2,
            ),
            flush=True,
        )
    report["solver_wall_s"], report["operations"] = budget.used, budget.calls
    (arguments.output / "ShortWindows.json").write_text(
        json.dumps(serializable(report), indent=2, allow_nan=False) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
