"""Steady J-V experiments executed by the new 1D core.

Each requested voltage is independently residual/current certified by the
steady solver, with the preceding state used only as an initial guess. This
protocol has no scan-rate or hysteresis interpretation. Extracted metrics use
piecewise-linear interpolation and do not certify voltage-grid convergence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

if TYPE_CHECKING:
    from solarlab.equations import CompiledDevice1D
    from solarlab.io.artifacts import ArtifactPayload
    from solarlab.io.run_store import AttemptToken, RunStore
    from solarlab.solvers.nonlinear.steady_1d import SteadyOptions, SteadyState1D


def _vector(value: ArrayLike) -> NDArray[np.float64]:
    result = np.array(value, dtype=float, copy=True)
    if result.ndim != 1 or result.size < 2 or not np.all(np.isfinite(result)):
        raise ValueError("J-V coordinates require at least two finite values")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class SteadyJVMetrics:
    """Missing brackets remain None; FF and efficiency are fractions."""

    short_circuit_current_A_m2: float | None
    open_circuit_voltage_V: float | None
    maximum_power_W_m2: float | None
    fill_factor: float | None
    efficiency: float | None
    limitation: str


def steady_jv_metrics(
    voltage_V: ArrayLike, output_current_A_m2: ArrayLike,
    *, incident_power_W_m2: float | None = None,
) -> SteadyJVMetrics:
    """Metrics for output-positive current, without inventing a missing V_oc."""
    voltage, current = _vector(voltage_V), _vector(output_current_A_m2)
    if current.shape != voltage.shape or np.unique(voltage).size != voltage.size:
        raise ValueError("J-V arrays must have matching shapes and unique voltages")
    if incident_power_W_m2 is not None and (
        not np.isfinite(incident_power_W_m2) or incident_power_W_m2 <= 0.0
    ):
        raise ValueError("incident power must be finite and positive")
    order = np.argsort(voltage)
    voltage, current = voltage[order], current[order]
    if not voltage[0] <= 0.0 <= voltage[-1]:
        return SteadyJVMetrics(None, None, None, None, None, "zero_voltage_not_bracketed")
    jsc = float(np.interp(0.0, voltage, current))
    if jsc <= 0.0:
        return SteadyJVMetrics(jsc, None, None, None, None, "no_positive_short_circuit_output")
    positive = voltage > 0.0
    v = np.r_[0.0, voltage[positive]]
    j = np.r_[jsc, current[positive]]
    crossings = np.flatnonzero(j <= 0.0)
    if crossings.size == 0:
        return SteadyJVMetrics(jsc, None, None, None, None, "open_circuit_not_bracketed")
    end = int(crossings[0])
    voc = float(v[end - 1] + j[end - 1] * (v[end] - v[end - 1]) / (j[end - 1] - j[end]))
    v, j = np.r_[v[:end], voc], np.r_[j[:end], 0.0]
    maximum = float(np.max(v * j))
    for left in range(v.size - 1):
        slope = (j[left + 1] - j[left]) / (v[left + 1] - v[left])
        intercept = j[left] - slope * v[left]
        if slope < 0.0:
            peak = -intercept / (2.0 * slope)
            if v[left] < peak < v[left + 1]:
                maximum = max(maximum, float(peak * (intercept + slope * peak)))
    return SteadyJVMetrics(jsc, voc, maximum, maximum / (jsc * voc),
                           None if incident_power_W_m2 is None else maximum / incident_power_W_m2,
                           "piecewise_linear_voltage_grid; mesh_and_grid_convergence_not_certified")


@dataclass(frozen=True)
class SteadyJVResult:
    voltage_V: NDArray[np.float64]
    current_density_A_m2: NDArray[np.float64]
    output_current_A_m2: NDArray[np.float64]
    solutions: tuple[SteadyState1D, ...]
    metrics: SteadyJVMetrics
    generation_scale: float
    incident_power_W_m2: float | None
    voltage_polarity: float
    compiled_identity: str
    nodes_m: NDArray[np.float64]
    solver_controls: dict[str, Any]


def _prepare_legacy_benchmark_initial_state(
    compiled: CompiledDevice1D,
    *,
    first_voltage_V: float,
    generation_scale: float,
    controls: SteadyOptions,
) -> SteadyState1D | None:
    """Prepare an accepted illuminated seed for legacy manual-V_bi benchmark decks.

    Manual ``V_bi`` decks do not satisfy ``semiconductor_work_function`` dark
    thermodynamic identity ``V_bi == W_L - W_R`` at ``(V=0, G=0)``, so dark
    equilibrium certification fails on contact span even when Poisson and both
    continuity equations converge to machine precision. Stepping light from
    ``1e-12`` to ``generation_scale`` through ``solve_steady_state`` with an
    explicit ``State1D`` certifies every stage under full Poisson/continuity/
    all-face-current gates without relaxing any tolerance.
    """
    if "legacy_benchmark_deck" not in compiled.capabilities or generation_scale <= 0.0:
        return None
    from scipy.sparse import diags

    from solarlab.contracts.units import ELEMENTARY_CHARGE_C
    from solarlab.equations.assembly.classical_1d import AssembledDC1DProblem
    from solarlab.equations.dc_1d import DC1DProblem
    from solarlab.equations.implementation import use_assembled
    from solarlab.solvers.nonlinear.newton import NewtonSettings, solve_newton
    from solarlab.solvers.nonlinear.problem import CallableProblem, ResidualScaling, StepControl
    from solarlab.solvers.nonlinear.production import (
        poisson_row_weights,
        production_convergence_norm,
        production_coordinates,
        production_residual_scaling,
    )
    from solarlab.solvers.nonlinear.steady_1d import (
        SteadyStateError,
        density_basin_seed,
        physical_gates,
        solve_steady_state,
    )

    count = compiled.node_count
    q_charge = ELEMENTARY_CHARGE_C

    def pb_densities(phi: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        _, n, p = compiled.fields(np.r_[phi, np.zeros(2 * count)])
        return n, p

    def pb_residual(phi: NDArray[np.float64]) -> NDArray[np.float64]:
        n, p = pb_densities(phi)
        if not np.all(np.isfinite(n)) or not np.all(np.isfinite(p)):
            return np.full(count, np.nan)
        rho = compiled.charge_ledger(n, p).space_charge_C_m3()
        return compiled.poisson.residual(phi, rho, *compiled.contact_potentials(0.0))

    def pb_jacobian(phi: NDArray[np.float64]) -> object:
        n, p = pb_densities(phi)
        slope = np.zeros(count)
        slope[1:-1] = (
            -q_charge
            * compiled.poisson.interior_widths_m
            * (n[1:-1] + p[1:-1])
            / compiled.thermal_voltage_V
        )
        return compiled.poisson.jacobian() + diags(slope)

    pb_settings = NewtonSettings(
        ResidualScaling(poisson_row_weights(compiled) / controls.poisson_tolerance),
        1.0,
        StepControl(np.full(count, controls.potential_step_V)),
        controls.budget(),
        (),
        False,
    )
    pb = solve_newton(CallableProblem(pb_residual, pb_jacobian), compiled.reference_potential_V, pb_settings)
    if not pb.accepted:
        return None
    u = np.r_[pb.state, np.zeros(2 * count)]
    dark_problem = AssembledDC1DProblem(compiled, 0.0, 0.0) if use_assembled() else DC1DProblem(compiled, 0.0, 0.0)
    dark_settings = NewtonSettings(
        production_residual_scaling(
            compiled,
            poisson_tolerance=controls.poisson_tolerance,
            continuity_tolerance_A_m2=controls.continuity_tolerance_A_m2,
            contact_log_tolerance=controls.contact_log_tolerance,
        ),
        1.0,
        production_coordinates(
            compiled,
            potential_step_V=controls.potential_step_V,
            quasi_fermi_step=controls.quasi_fermi_step,
        ),
        controls.budget(),
        physical_gates(dark_problem, controls, equilibrium=False),
        False,
    )
    dark_newton = solve_newton(
        dark_problem,
        u,
        dark_settings,
        convergence_norm=production_convergence_norm(compiled, convergence_policy=controls.convergence_policy),
    )
    if not dark_newton.accepted:
        return None
    u = dark_newton.state
    last_steady: SteadyState1D | None = None
    for frac in (1.0e-8, 1.0e-6, 1.0e-4, 1.0e-2, 1.0):
        g_val = frac * generation_scale
        lit_prob = AssembledDC1DProblem(compiled, 0.0, g_val) if use_assembled() else DC1DProblem(compiled, 0.0, g_val)
        try:
            u_seed = density_basin_seed(lit_prob, u)
            last_steady = solve_steady_state(
                compiled,
                initial_state=compiled.state(u_seed, voltage_V=0.0),
                voltage_V=0.0,
                generation_scale=g_val,
                options=controls,
            )
        except (ValueError, SteadyStateError):
            return None
        u = last_steady.state.u
    assert last_steady is not None
    if first_voltage_V != 0.0:
        bias_count = max(1, int(np.ceil(abs(first_voltage_V) / controls.max_bias_step_V)))
        cur_state = last_steady.state
        for v_step in np.linspace(0.0, first_voltage_V, bias_count + 1)[1:]:
            last_steady = solve_steady_state(
                compiled,
                initial_state=cur_state,
                voltage_V=float(v_step),
                generation_scale=generation_scale,
                options=controls,
            )
            cur_state = last_steady.state
    return last_steady


def run_steady_jv(
    compiled: CompiledDevice1D, voltage_V: ArrayLike, *,
    generation_scale: float = 1.0, incident_power_W_m2: float | None = None,
    options: SteadyOptions | None = None,
    stop_after_voc: bool = False,
    max_bias_bisect: int = 0,
) -> SteadyJVResult:
    """Run the actual new solver; the first failed point raises its solver error."""
    from solarlab.solvers.nonlinear.steady_1d import (
        SteadyOptions,
        SteadyStateError,
        solve_steady_state,
    )

    if not isinstance(stop_after_voc, bool):
        raise TypeError("stop_after_voc must be a bool")
    if type(max_bias_bisect) is not int or max_bias_bisect < 0:
        raise ValueError("max_bias_bisect must be a nonnegative integer")
    voltages = _vector(voltage_V)
    differences = np.diff(voltages)
    if not (np.all(differences > 0.0) or np.all(differences < 0.0)):
        raise ValueError("steady J-V voltages must be strictly monotonic")
    if not np.isfinite(generation_scale) or generation_scale < 0.0:
        raise ValueError("generation scale must be finite and nonnegative")
    if incident_power_W_m2 is not None and (
        not np.isfinite(incident_power_W_m2) or incident_power_W_m2 <= 0.0
    ):
        raise ValueError("incident power must be finite and positive")
    polarity = float(compiled.voltage_polarity)
    if polarity not in (-1.0, 1.0):
        raise ValueError("compiled device must declare its terminal voltage polarity")
    state = None
    solutions: list[SteadyState1D] = []
    accepted_voltages: list[float] = []
    controls = SteadyOptions() if options is None else options
    legacy_seed = _prepare_legacy_benchmark_initial_state(
        compiled,
        first_voltage_V=float(voltages[0]),
        generation_scale=generation_scale,
        controls=controls,
    )
    if legacy_seed is not None:
        state = legacy_seed.state
    saw_positive_pv = False
    stopped_early = False

    def _record_accepted(point: SteadyState1D, v_val: float) -> bool:
        nonlocal state, saw_positive_pv, stopped_early
        if not point.converged:
            raise RuntimeError(f"uncertified steady-state result at {v_val} V")
        solutions.append(point)
        accepted_voltages.append(v_val)
        state = point.state
        if stop_after_voc:
            j_out = -polarity * float(point.current_density_A_m2[0])
            if v_val >= 0.0 and j_out > 0.0:
                saw_positive_pv = True
            elif saw_positive_pv and v_val > 0.0 and j_out <= 0.0 and len(solutions) >= 2:
                stopped_early = True
                return True
        return False

    def _advance_with_bisection(v_from: float, v_to: float, depth: int) -> None:
        if stopped_early:
            return
        try:
            res_pt = solve_steady_state(
                compiled,
                initial_state=state,
                voltage_V=v_to,
                generation_scale=generation_scale,
                options=controls,
            )
        except SteadyStateError:
            if depth >= max_bias_bisect or state is None:
                raise
            v_mid = 0.5 * (v_from + v_to)
            if v_mid == v_from or v_mid == v_to:
                raise
            _advance_with_bisection(v_from, v_mid, depth + 1)
            if not stopped_early:
                _advance_with_bisection(accepted_voltages[-1], v_to, depth + 1)
            return
        _record_accepted(res_pt, v_to)

    for voltage in voltages:
        v_target = float(voltage)
        if max_bias_bisect == 0 or not accepted_voltages:
            result = solve_steady_state(
                compiled,
                initial_state=state,
                voltage_V=v_target,
                generation_scale=generation_scale,
                options=controls,
            )
            if _record_accepted(result, v_target):
                break
        else:
            _advance_with_bisection(accepted_voltages[-1], v_target, 0)
            if stopped_early:
                break

    voltages_used = (
        _vector(accepted_voltages)
        if (stop_after_voc or max_bias_bisect > 0)
        else voltages
    )
    currents = np.array([result.current_density_A_m2 for result in solutions], dtype=float)
    if currents.ndim != 2 or not np.all(np.isfinite(currents)):
        raise ValueError("steady solutions must contain finite all-face currents")
    # Keep every face, and report the left terminal rather than averaging away
    # a conservation error. The solver's all-face gate must already have passed.
    output = -polarity * currents[:, 0]
    currents.setflags(write=False)
    output.setflags(write=False)
    controls_dict = asdict(controls)
    if stop_after_voc:
        controls_dict["stop_after_voc"] = True
    if max_bias_bisect > 0:
        controls_dict["max_bias_bisect"] = max_bias_bisect
    return SteadyJVResult(voltages_used, currents, output, tuple(solutions),
                          steady_jv_metrics(voltages_used, output, incident_power_W_m2=incident_power_W_m2),
                          generation_scale, incident_power_W_m2, polarity,
                          compiled.identity, _vector(compiled.nodes_m), controls_dict)


def steady_jv_payload(result: SteadyJVResult) -> tuple[dict[str, Any], tuple[tuple[str, ArtifactPayload], ...]]:
    """Encode the completed curve without claiming or inventing an attempt.

    A supervisor may transfer these named payloads before it publishes them.
    Actual artifact IDs and attempt fields belong to the publishing owner.
    """
    from solarlab.io.artifacts import array_artifact

    if len(result.solutions) != result.voltage_V.size or any(
        not point.converged or point.state.device_sha256 != result.compiled_identity
        for point in result.solutions
    ):
        raise ValueError("every J-V point must be an accepted state of the same compiled device")
    arrays = {
        "nodes_m": (result.nodes_m, "m"),
        "voltage_V": (result.voltage_V, "V"),
        "current_density_A_m2": (result.current_density_A_m2, "A/m^2"),
        "contact_current_density_A_m2": (np.array([point.contact_current_density_A_m2 for point in result.solutions]), "A/m^2"),
        "output_current_A_m2": (result.output_current_A_m2, "A/m^2"),
    }
    for name, unit in (("potential_V", "V"), ("electron_density_m3", "m^-3"), ("hole_density_m3", "m^-3")):
        arrays[name] = (np.array([getattr(point.state, name) for point in result.solutions]), unit)
    payloads = tuple((name, array_artifact(array, unit=unit,
                 metadata={"schema": "solarlab.steady-jv.v1", "engine": "solarlab",
                           "compiled_identity": result.compiled_identity}))
               for name, (array, unit) in arrays.items())
    document = {
        "schema": "solarlab.steady-jv.v1", "engine": "solarlab", "outcome": "completed",
        "protocol": "steady_voltage_continuation", "voltage_polarity": result.voltage_polarity,
        "generation_scale": result.generation_scale, "incident_power_W_m2": result.incident_power_W_m2,
        "compiled_identity": result.compiled_identity, "solver_controls": result.solver_controls,
        "metrics": asdict(result.metrics),
        "solver_evidence": [{"voltage_V": point.voltage_V, "accepted": point.newton.accepted,
                             "termination": point.newton.termination.value,
                             "scaled_residual": point.newton.scaled_residual,
                             "iterations": point.newton.iteration_count,
                             "preparation_iterations": [step.iteration_count for step in point.preparation],
                             "density_basin_initializations": point.density_basin_initializations,
                             "density_basin_stages": [
                                 {"voltage_V": voltage, "generation_scale": generation}
                                 for voltage, generation in point.density_basin_stages
                             ],
                             "gates": [asdict(gate) for gate in point.newton.gates]}
                            for point in result.solutions],
        "qualification": "run_completed; final_scientific_acceptance_not_established",
    }
    return document, payloads


def publish_steady_jv(store: RunStore, token: AttemptToken, result: SteadyJVResult) -> dict[str, Any]:
    """Publish the same lossless payload to an already claimed RunStore attempt."""
    document, payloads = steady_jv_payload(result)
    records = {name: store.publish(token, name, payload) for name, payload in payloads}
    return {**document, "run_id": token.run_id, "attempt_id": token.attempt_id,
            "generation": token.generation,
            "arrays": {name: record["artifact_id"] for name, record in records.items()},
            "artifact_ids": [record["artifact_id"] for record in records.values()]}


def read_steady_jv(store: RunStore, document: dict[str, Any]) -> dict[str, NDArray[np.float64]]:
    """Read through RunStore integrity verification, with no solver invocation."""
    if (document.get("schema") != "solarlab.steady-jv.v1" or document.get("engine") != "solarlab"
            or document.get("outcome") != "completed"):
        raise ValueError("expected a new-core steady J-V result")
    units = {"nodes_m": "m", "voltage_V": "V", "current_density_A_m2": "A/m^2", "output_current_A_m2": "A/m^2",
             "contact_current_density_A_m2": "A/m^2",
             "potential_V": "V", "electron_density_m3": "m^-3", "hole_density_m3": "m^-3"}
    if set(document.get("arrays", {})) != set(units):
        raise ValueError("incomplete steady J-V array set")
    values = {}
    for name, unit in units.items():
        identifier = document["arrays"][name]
        record = store.artifact(identifier)
        if record["name"] != name or any(record[key] != document.get(key)
                                          for key in ("run_id", "attempt_id", "generation")):
            raise ValueError("artifact role or originating attempt does not match the result")
        if record["metadata"].get("unit") != unit:
            raise ValueError(f"unexpected unit for {name}")
        if record["metadata"].get("compiled_identity") != document.get("compiled_identity"):
            raise ValueError("array belongs to a different compiled device")
        values[name] = store.read_array(identifier)
    count = values["voltage_V"].size
    if values["voltage_V"].shape != (count,) or values["output_current_A_m2"].shape != (count,):
        raise ValueError("voltage and output current must each be a vector")
    if any(value.shape[0] != count for name, value in values.items() if name != "nodes_m"):
        raise ValueError("steady J-V arrays have inconsistent point counts")
    nodes = values["nodes_m"].size
    if (values["nodes_m"].shape != (nodes,) or values["contact_current_density_A_m2"].shape != (count, 2)
            or values["current_density_A_m2"].shape != (count, nodes - 1)) or any(
        values[name].shape != (count, nodes) for name in ("potential_V", "electron_density_m3", "hole_density_m3")
    ):
        raise ValueError("steady J-V fields do not share the saved node grid")
    return values
