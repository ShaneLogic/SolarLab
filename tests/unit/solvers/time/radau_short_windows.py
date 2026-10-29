"""Explicitly invoked short E_L/I_V probe; never part of routine pytest discovery.

One 4/8-step pair per 37/73-node case, shared finite solver/enclosure budget.
This is not an admission run. A refinement difference is only an observation,
never a substitute for a proved error bound. Every stopped case is retained.

Integration conditions: optional, unqualified reference infrastructure for
restricted declared E/I inputs and analytic unit fixtures only. Independence
is static-only: no BDF stepping, history, interpolation or accounting is used.
Retained failed attempts stay failures and qualification flags stay false.
Any future nonzero initial uncertainty must be expressed in the selected
weighted log norm; a numerical trajectory alone does not qualify a reference.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, fields, is_dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import signal
import time
from typing import Any, Callable, TypeVar

import numpy as np
from flint import ctx

from examples.research_1d.ionic_multilayer import prepare_confined_ion_device
from examples.research_1d.ionic_steady import controls as ionic_controls
from examples.research_1d.steady import load_device
from solarlab.device.compile_1d import compile_device_1d
from solarlab.device.compile_ionic_1d import compile_ionic_device_1d
from solarlab.equations.assembly.classical_1d import AssembledDC1DProblem
from solarlab.equations.assembly.ionic_1d import AssembledIonic1DProblem
from solarlab.solvers.nonlinear.ionic_1d import solve_ionic_steady_state
from solarlab.solvers.nonlinear.steady_1d import solve_steady_state
from solarlab.solvers.time.radau_enclosure import (
    NonlinearInventoryEnclosure,
    PWLDrive,
    enclosed_inventory_dae,
    nonlinear_tube_bound,
)
from solarlab.solvers.time.radau_reference import (
    RadauControls,
    RadauFailure,
    RadauTrajectory,
    assembled_inventory_dae,
    integrate_radau,
)

T = TypeVar("T")


def serializable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: serializable(getattr(value, item.name)) for item in fields(value)
        }
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [serializable(x) for x in value]
    if isinstance(value, np.generic):
        return serializable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite": repr(value)}
    return value


class Budget:
    def __init__(self, seconds: float):
        if not 0 < seconds <= 240:
            raise ValueError(
                "one probe is capped at 240 s, within the task's shared 300 s cap"
            )
        self.limit = seconds
        self.used = 0.0
        self.calls: list[dict[str, Any]] = []

    def call(self, name: str, function: Callable[[], T]) -> T:
        remaining = self.limit - self.used
        if remaining <= 0:
            raise TimeoutError("shared_solver_budget_exhausted")

        def expired(signum: int, frame: Any) -> None:
            raise TimeoutError("shared_solver_budget_exhausted")

        before = signal.signal(signal.SIGALRM, expired)
        started = time.perf_counter()
        signal.setitimer(signal.ITIMER_REAL, remaining)
        record: dict[str, Any] = {"operation": name, "completed": False}
        try:
            value = function()
            record["completed"] = True
            return value
        except Exception as error:
            record["error_type"], record["error"] = type(error).__name__, str(error)
            raise
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            signal.signal(signal.SIGALRM, before)
            elapsed = time.perf_counter() - started
            self.used += elapsed
            record["wall_s"] = elapsed
            self.calls.append(record)


def save_trajectory(path: Path, value: RadauTrajectory) -> dict[str, Any]:
    np.savez_compressed(
        path,
        initial_u=value.initial.u,
        initial_inventory=value.initial.inventory,
        times=np.array([value.initial.time_s, *(s.right_s for s in value.steps)]),
        coefficients=np.array([s.coefficients for s in value.steps]),
        endpoint_u=np.array([s.endpoint.u for s in value.steps]),
        endpoint_inventory=np.array([s.endpoint.inventory for s in value.steps]),
        stage_u=np.array([[p.u for p in s.stages] for s in value.steps]),
        stage_inventory=np.array(
            [[p.inventory for p in s.stages] for s in value.steps]
        ),
        stage_rhs=np.array([[p.rhs for p in s.stages] for s in value.steps]),
        stage_residual=np.array([s.stage_residual for s in value.steps]),
    )
    return {
        "path": str(path),
        "sha256": sha256(path.read_bytes()).hexdigest(),
        "steps": len(value.steps),
        "counters": asdict(value.counters),
        "qualified": False,
    }


def run_case(
    scope: str,
    intervals: int,
    output: Path,
    budget: Budget,
    *,
    enclosed: bool = False,
    initial_case: dict[str, Any] | None = None,
) -> dict[str, Any]:
    label = f"{scope}_{3 * intervals + 1}"
    result: dict[str, Any] = {
        "scope": scope,
        "electrical_nodes": 3 * intervals + 1,
        "status": "reference_blocked",
        "tube_proved": False,
        "error_bound": None,
        "observed_refinement_delta": None,
        "histories": [],
        "admission_run": False,
    }
    result["point_evaluator"] = (
        "verified_enclosure_midpoint" if enclosed else "floating_projection"
    )
    ionic = scope == "I_V"
    duration = 1e-4 if ionic else 1e-8
    drive = (
        PWLDrive((0.0, 1.0), (0.002, 0.004), (0.0, 0.0))
        if ionic
        else PWLDrive(
            tuple(4e-7 * j / 16 for j in range(17)),
            (0.0,) * 17,
            tuple(1 + 0.01 * (j / 16) ** 2 for j in range(17)),
        )
    )
    result["short_window_s"] = [0.0, duration]
    result["drive"] = asdict(drive)
    result["drive_hex"] = {
        key: [x.hex() for x in value] for key, value in asdict(drive).items()
    }
    try:
        if ionic:
            device = compile_ionic_device_1d(
                prepare_confined_ion_device(),
                intervals_per_layer=(intervals,) * 3,
                ion_domain_layer="absorber",
                ion_domain_endpoints=("blocking", "blocking"),
            )
            problem: Any = AssembledIonic1DProblem(device, 0.002, 0.0)
            initial = budget.call(
                label + ":initial",
                lambda: solve_ionic_steady_state(
                    device,
                    voltage_V=0.002,
                    generation_scale=0.0,
                    options=ionic_controls(device.ion_node_count),
                ),
            )
            result["initial"] = {
                "accepted": initial.accepted,
                "termination": initial.termination,
                "checks": serializable(initial.checks),
                "error": initial.error,
            }
            if not initial.accepted or initial.state is None:
                result["reason"] = "initial_preparation_failed"
                return result
            u = initial.state.u
            result["ion_nodes"] = device.ion_node_count
        else:
            electronic = compile_device_1d(
                load_device(), intervals_per_layer=(intervals,) * 3
            )
            problem = AssembledDC1DProblem(electronic, 0.0, 1.0)
            if initial_case is not None:
                if (
                    initial_case["device_identity"] != electronic.identity
                    or initial_case["initial"]["accepted"] is not True
                ):
                    raise ValueError("saved_initial_identity_or_acceptance_mismatch")
                u = np.asarray(initial_case["initial"]["newton"]["state"], dtype=float)
                result["initial"] = initial_case["initial"]
                result["initial_reused"] = True
            else:
                prepared = budget.call(
                    label + ":initial",
                    lambda: solve_steady_state(
                        electronic, voltage_V=0.0, generation_scale=1.0
                    ),
                )
                u = prepared.state.u
                result["initial"] = {
                    "accepted": True,
                    "newton": serializable(prepared.newton),
                }
        result["device_identity"] = problem.device.identity
        mask = problem.binding.layout.constraint_mask()
        mask[problem.binding.layout.rows("potential")] = True
        initial_z = problem.storage(u)[~mask]
        scale = np.maximum(np.abs(initial_z), 1.0)
        result["component_scale"] = scale.tolist()
        result["initial_error_semantics"] = (
            "unit IVP defined by exact stored binary64 inventories; no bound on departure from true steady initial state"
        )
        result["initial"]["u"] = u.tolist()
        factory = enclosed_inventory_dae if enclosed else assembled_inventory_dae
        model = factory(
            problem,
            drive,
            inventory_scale=scale,
            algebraic_scale=np.ones(np.count_nonzero(mask)),
        )
        options = RadauControls()
        trajectories = []
        for count in (4, 8):
            try:
                trajectory = budget.call(
                    label + f":radau_{count}",
                    lambda: integrate_radau(
                        model,
                        u,
                        np.linspace(0.0, duration, count + 1),
                        controls=options,
                    ),
                )
            except RadauFailure as failure:
                result["failure"] = {
                    "code": failure.code,
                    "cause": str(failure.__cause__),
                    "failed_step": failure.failed_step,
                    "completed_steps": len(failure.completed_steps),
                    "time_s": failure.time_s,
                    "trial": failure.trial.tolist(),
                }
                if failure.completed_steps:
                    np.savez_compressed(
                        output / (label + f"_{count}_failed_prefix.npz"),
                        coefficients=np.array(
                            [s.coefficients for s in failure.completed_steps]
                        ),
                        endpoint_u=np.array(
                            [s.endpoint.u for s in failure.completed_steps]
                        ),
                        times=np.array([s.right_s for s in failure.completed_steps]),
                    )
                result["reason"] = "radau_numerical_prefix_failed"
                return result
            trajectories.append(trajectory)
            result["histories"].append(
                save_trajectory(output / (label + f"_{count}.npz"), trajectory)
            )

        def observations() -> float:
            return max(
                float(
                    np.max(
                        np.abs(
                            trajectories[0].sample(float(t)).inventory
                            - trajectories[1].sample(float(t)).inventory
                        )
                        / scale
                    )
                )
                for t in np.linspace(0.0, duration, 9)
            )

        result["observed_refinement_delta"] = budget.call(
            label + ":outputs", observations
        )
        result["observation_kind"] = (
            "scaled 4-versus-8-step inventory difference; not an exact error or certificate"
        )

        def bound() -> Any:
            encloser = NonlinearInventoryEnclosure(problem, drive, scale)
            return nonlinear_tube_bound(
                encloser,
                trajectories[-1],
                initial_error=0.0,
                tube_radius=1e-6,
                subdivisions=2,
            )

        proof = budget.call(label + ":enclosure", bound)
        result["bound_attempt"] = asdict(proof)
        result["tube_proved"] = proof.tube_proved
        result["error_bound"] = proof.error_upper
        result["status"] = proof.status
        result["reason"] = proof.reason
        result["qualification"] = False
        return result
    except Exception as error:
        result["reason"] = type(error).__name__ + ": " + str(error)
        return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=240.0)
    parser.add_argument("--enclosed", action="store_true")
    parser.add_argument("--case", choices=("E_L_37", "I_V_37", "E_L_73", "I_V_73"))
    parser.add_argument("--initial-packet", type=Path)
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise FileExistsError("preserve the existing short-window packet")
    arguments.output.mkdir(parents=True)
    ctx.threads = 1
    budget = Budget(arguments.seconds)
    cases: list[dict[str, Any]] = []
    root = Path(__file__).resolve().parents[4]
    files = [
        "src/solarlab/solvers/time/radau_reference.py",
        "src/solarlab/solvers/time/radau_enclosure.py",
        "tests/unit/solvers/time/radau_short_windows.py",
    ]
    report: dict[str, Any] = {
        "source_pins": {
            path: sha256((root / path).read_bytes()).hexdigest() for path in files
        },
        "admission_runs": 0,
        "qualification": False,
        "cases": cases,
        "solver_wall_cap_s": budget.limit,
        "purpose": "short prefix unit validation; separate prospective review and authorization required for admission",
    }
    initial_cases: dict[tuple[str, int], dict[str, Any]] = {}
    if arguments.initial_packet is not None:
        packet = json.loads(arguments.initial_packet.read_text())
        source = "src/solarlab/solvers/time/radau_reference.py"
        if packet["source_pins"][source] != report["source_pins"][source]:
            raise ValueError("saved_initial_packet_has_a_different_Radau_source")
        initial_cases = {
            (c["scope"], c["electrical_nodes"]): c for c in packet["cases"]
        }
        report["initial_packet"] = {
            "path": str(arguments.initial_packet),
            "sha256": sha256(arguments.initial_packet.read_bytes()).hexdigest(),
        }
    report["point_evaluator"] = (
        "verified_enclosure_midpoint" if arguments.enclosed else "floating_projection"
    )
    (arguments.output / "Plan.json").write_text(
        json.dumps(
            {
                **report,
                "planned_cases": [
                    {
                        "scope": scope,
                        "nodes": 3 * m + 1,
                        "end_s": 1e-8 if scope == "E_L" else 1e-4,
                        "step_counts": [4, 8],
                        "tube_radius": 1e-6,
                        "subdivisions": 2,
                    }
                    for m in (12, 24)
                    for scope in ("E_L", "I_V")
                    if arguments.case is None
                    or arguments.case == f"{scope}_{3 * m + 1}"
                ],
            },
            indent=2,
        )
        + "\n"
    )
    for intervals in (12, 24):
        for scope in ("E_L", "I_V"):
            if (
                arguments.case is not None
                and arguments.case != f"{scope}_{3 * intervals + 1}"
            ):
                continue
            result = run_case(
                scope,
                intervals,
                arguments.output,
                budget,
                enclosed=arguments.enclosed,
                initial_case=initial_cases.get((scope, 3 * intervals + 1)),
            )
            cases.append(result)
            report["solver_wall_s"] = budget.used
            report["operations"] = budget.calls
            (arguments.output / "ShortWindows.json").write_text(
                json.dumps(serializable(report), indent=2, allow_nan=False) + "\n"
            )
            print(
                json.dumps(
                    {
                        key: value
                        for key, value in result.items()
                        if key
                        in {
                            "scope",
                            "electrical_nodes",
                            "status",
                            "reason",
                            "tube_proved",
                            "error_bound",
                            "observed_refinement_delta",
                            "short_window_s",
                            "bound_attempt",
                        }
                    },
                    indent=2,
                ),
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
