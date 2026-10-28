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


def run_steady_jv(
    compiled: CompiledDevice1D, voltage_V: ArrayLike, *,
    generation_scale: float = 1.0, incident_power_W_m2: float | None = None,
    options: SteadyOptions | None = None,
) -> SteadyJVResult:
    """Run the actual new solver; the first failed point raises its solver error."""
    from solarlab.solvers.nonlinear.steady_1d import SteadyOptions, solve_steady_state

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
    solutions = []
    controls = SteadyOptions() if options is None else options
    for voltage in voltages:
        result = solve_steady_state(compiled, initial_state=state, voltage_V=float(voltage),
                                    generation_scale=generation_scale, options=controls)
        if not result.converged:
            raise RuntimeError(f"uncertified steady-state result at {voltage} V")
        solutions.append(result)
        state = result.state
    currents = np.array([result.current_density_A_m2 for result in solutions], dtype=float)
    if currents.ndim != 2 or not np.all(np.isfinite(currents)):
        raise ValueError("steady solutions must contain finite all-face currents")
    # Keep every face, and report the left terminal rather than averaging away
    # a conservation error. The solver's all-face gate must already have passed.
    output = -polarity * currents[:, 0]
    currents.setflags(write=False)
    output.setflags(write=False)
    return SteadyJVResult(voltages, currents, output, tuple(solutions),
                          steady_jv_metrics(voltages, output, incident_power_W_m2=incident_power_W_m2),
                          generation_scale, incident_power_W_m2, polarity,
                          compiled.identity, _vector(compiled.nodes_m), asdict(controls))


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
