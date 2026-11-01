"""Test-only BE activation: no production constructor/driver file is edited."""

from contextlib import contextmanager
import hashlib
from typing import Any

import numpy as np

from solarlab.equations.dc_1d import CompiledDevice1D
from solarlab.experiments import transient
from solarlab.solvers.nonlinear.problem import CallableProblem
from solarlab.solvers.time.bdf_1d import BDFDrive
from solarlab.solvers.time.carrier_seed import DensitySeedContext, recover_density_seed, stagnation_basin_signature


def _closure(function: Any) -> dict[str, Any]:
    return dict(zip(function.__code__.co_freevars, (cell.cell_contents for cell in function.__closure__ or ())))


@contextmanager
def injected_BE_density_recovery(request, drive: BDFDrive, ledger: list[dict[str, Any]], *, enabled: bool):
    """Bind the exact original step/deadline through test-only closure inspection.

    Production hooks must pass that context explicitly; this shim is evidence
    only. It runs the original seed first and adds accounting only on recovery.
    """
    old_step, old_newton, old_accounting = transient.BackwardEuler1DStep, transient.solve_newton, transient.step_accounting
    current = None
    index = 0
    records: dict[int, list[dict[str, Any]]] = {}

    class BoundStep(old_step):
        def __post_init__(self):
            nonlocal current, index
            super().__post_init__()
            current = self
            index += 1

    def perform(problem, guess, settings, **kwargs):
        original = old_newton(problem, guess, settings, **kwargs)
        if not enabled or not stagnation_basin_signature(original, settings.tolerance):
            return original
        step = current
        if step is None or index >= len(request.times_s):
            raise ValueError("test-only BE context has no matching physical step")
        end = float(request.times_s[index])
        exact = np.flatnonzero(drive.times_s == end)
        if exact.size != 1:
            return original
        left, right = drive.left_values[int(exact[0])], drive.right_values[int(exact[0])]
        if (left[1] != 0.0 or right[1] <= 0.0 or left[0] != right[0]
                or request.generation_scale[index - 1] != 0.0
                or step.endpoint.generation_scale != right[1]
                or step.previous.voltage_V != right[0] or step.endpoint.voltage_V != right[0]):
            return original
        if not isinstance(problem, CallableProblem):
            raise ValueError("BE recovery requires the original counted callable target")
        enclosing = _closure(problem.residual_function)
        if enclosing.get("step") is not step:
            raise ValueError("BE recovery target differs from the original complete step")
        checkpoint = _closure(enclosing["evaluate"])["checkpoint"]
        device = step.endpoint.device

        def validate():
            if (type(device) is not CompiledDevice1D or step.previous.device_sha256 != device.identity
                    or step.previous.u.shape != (3 * device.node_count,)
                    or not {"ion_free", "classical_1d_dc", "maxwell_boltzmann"} <= set(device.capabilities)
                    or step.dt_s != end - float(request.times_s[index - 1])):
                raise ValueError("BE recovery device/reference/full-step binding differs")

        def fingerprint():
            digest = hashlib.sha256()
            for values in (step.previous.u, step.previous.potential_V,
                           step.previous.electron_density_m3, step.previous.hole_density_m3):
                digest.update(values.tobytes())
            digest.update(step.previous.device_sha256.encode())
            digest.update(float(step.previous.voltage_V).hex().encode())
            return digest.hexdigest()

        context = DensitySeedContext(device, step.endpoint.voltage_V,
            {"policy": "dark_generation_stagnation_density_v1", "device_identity": device.identity,
             "coordinate_form": "electrode_quasi_fermi_1d_v1", "source_knot_s": end,
             "left_source": left.copy(), "right_source": right.copy(),
             "complete_target": "original BackwardEuler1DStep; frozen right-endpoint source convention"},
            validate, fingerprint, checkpoint,
            deadline_exceptions=(TimeoutError, transient._WallBudgetExceeded))
        record = {"stage": "BE_main", "physical_attempt_index": index - 1,
                  "start_s": float(request.times_s[index - 1]), "end_s": end,
                  "order": 1, "coefficients": np.array([1.0, -1.0])}
        ledger.append(record)
        records.setdefault(id(step), []).append(record)
        return recover_density_seed(problem, guess, original, settings, context, record,
                                    convergence_norm=kwargs.get("convergence_norm"))

    def accounting(step, u):
        data = old_accounting(step, u)
        attached = records.get(id(step))
        if attached:
            data["density_seed_recoveries"] = tuple(attached)
            data["density_seed_work_totals"] = {key: sum(r["seed_work"][key] for r in attached)
                                               for key in attached[0]["seed_work"]}
        return data

    transient.BackwardEuler1DStep, transient.solve_newton, transient.step_accounting = BoundStep, perform, accounting
    try:
        yield
    finally:
        transient.BackwardEuler1DStep, transient.solve_newton, transient.step_accounting = old_step, old_newton, old_accounting
