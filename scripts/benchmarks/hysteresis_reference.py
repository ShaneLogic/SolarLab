"""Isolated, unqualified reference driver for the frozen historical HI protocol.

``plan`` reads the saved request and constructs controls without a device solve.
``run`` additionally requires a separately admitted, exactly bound run packet.
The public density engine supplies accepted knots (t_eval=None), not dense
output. All 111/221/441 observations are exact members of the fixed 440-interval
control grid. Other observations are deliberately unsupported.

No production entrypoint is replaced. NC06/device derivative qualification,
continuum error bounds and reference acceptance are NOT provided by this file.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass, is_dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any


CONTRACT = "reproducibility/HysteresisReferenceContractV1.json"
CONTRACT_SHA256 = "bec9fff7f5194a81492b9697f354d460413293a84aa66667c97cf5371adf8d67"
ANALYTIC = "tests/numerical_contracts/AnalyticGatesV1.json"
ANALYTIC_SHA256 = "43c9ba4ab1b6a95b3e278efcd8eed9a67dc4d5e258295c8e6961c53f7ee57201"
PHASE_IDS = ("dark_seed", "dark_prebias", "forward_light_dwell", "forward_ramp",
             "dark_turnaround", "reverse_light_dwell", "reverse_ramp")
OBSERVATION_COUNTS = (111, 221, 441)
THREAD_KEYS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS")
EXACT_DENSE_FACTOR_POLICY = "exact_dense_two_entry_v1"


class ReferenceError(ValueError):
    """An identity, protocol, capability or admission is absent/incompatible."""


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_json(path: Path, expected: str) -> dict:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ReferenceError(f"source mismatch: {path}")
    return json.loads(raw)


def rational(value: Any) -> Fraction:
    if isinstance(value, bool) or not math.isfinite(float(value)):
        raise ReferenceError("time/voltage must be a finite number, not boolean")
    return Fraction(str(value))


@dataclass(frozen=True)
class Control:
    phase_id: str
    index: int
    start: Fraction
    stop: Fraction
    v_start: Fraction
    v_stop: Fraction
    illuminated: bool
    max_step_s: float
    max_rhs_calls: int

    @property
    def duration(self) -> float:
        return float(self.stop - self.start)

    def voltage(self, local_time: float) -> float:
        if not math.isfinite(local_time) or not 0 <= local_time <= self.duration:
            raise ReferenceError("local solver time outside fixed control interval")
        if local_time == 0:
            return float(self.v_start)
        if local_time == self.duration:
            return float(self.v_stop)
        return float(self.v_start) + float(self.v_stop-self.v_start)*local_time/self.duration

    def binding(self) -> dict:
        return {"phase": self.phase_id, "index": self.index,
                "start_s": str(self.start), "stop_s": str(self.stop),
                "start_voltage_V": str(self.v_start), "stop_voltage_V": str(self.v_stop),
                "illuminated": self.illuminated, "max_step_s": self.max_step_s,
                "max_rhs_evaluations": self.max_rhs_calls, "local_time_origin_s": 0}


@dataclass(frozen=True)
class Phase:
    id: str
    controls: tuple[Control, ...]

    @property
    def start(self) -> Fraction:
        return self.controls[0].start

    @property
    def stop(self) -> Fraction:
        return self.controls[-1].stop

    @property
    def voltage_rate(self) -> float:
        return float((self.controls[-1].v_stop-self.controls[0].v_start)/(self.stop-self.start))

    def voltage(self, physical_time: Fraction) -> float:
        if not self.start <= physical_time <= self.stop:
            raise ReferenceError("physical time outside named phase")
        fraction = (physical_time-self.start)/(self.stop-self.start)
        return float(self.controls[0].v_start + fraction*(self.controls[-1].v_stop-self.controls[0].v_start))


def make_timeline(contract: dict, request: dict, step_level: int) -> tuple[Phase, ...]:
    """No observation argument exists at this integration boundary."""
    if type(step_level) is not int or step_level not in (0, 1, 2):
        raise ReferenceError("step level must be 0, 1 or 2")
    p = request["params"]
    profiles = contract["refinement_plan"]["future_decoupled_reference_driver"]["phase_profiles"]
    matches = [x for x in profiles if rational(x["rate_V_s"]) == rational(p["v_rate"])]
    if len(matches) != 1:
        raise ReferenceError("scan rate is not in the frozen phase profiles")
    wf = p["waveform"]
    expected = {"dark_seed_s": 120, "dark_prep_s": 30, "branch_dwell_s": .5,
                "turnaround_s": 3, "start_voltage_V": -1,
                "uniform_generation_rate_m3_s": 2.5e27, "schema_version": 1,
                "turnaround_dark": True}
    if wf != expected or p["V_max"] != 1.2 or p["illuminated"] is not True:
        raise ReferenceError("physical waveform differs from the frozen saved request")
    endpoints = ((0, 0), (-1, -1), (-1, -1), (-1, 1.2),
                 (1.2, 1.2), (1.2, 1.2), (1.2, -1))
    lights = (False, False, True, True, False, True, True)
    phases = []
    if len(matches[0]["phases"]) != 7:
        raise ReferenceError("exactly seven physical phases are required")
    for index, row in enumerate(matches[0]["phases"]):
        if row["id"] != PHASE_IDS[index]:
            raise ReferenceError("physical phase order changed")
        start, stop = rational(row["start_s"]), rational(row["end_s"])
        count = 440 if index in (3, 6) else 1
        if row["control_intervals"] != count or stop-start != rational(row["duration_s"]):
            raise ReferenceError("fixed control grid/duration mismatch")
        if phases and start != phases[-1].stop:
            raise ReferenceError("physical timeline has a gap or overlap")
        v0, v1 = map(rational, endpoints[index])
        controls = tuple(Control(row["id"], j, start+(stop-start)*j/count,
                                 start+(stop-start)*(j+1)/count,
                                 v0+(v1-v0)*j/count, v0+(v1-v0)*(j+1)/count,
                                 lights[index], row["max_step_s_levels"][step_level],
                                 row["max_rhs_evaluations_per_control_interval"])
                         for j in range(count))
        phases.append(Phase(row["id"], controls))
    if len(phases) != 7:
        raise ReferenceError("exactly seven physical phases are required")
    return tuple(phases)


def observation_indices(count: int) -> tuple[int, ...]:
    if type(count) is not int or count not in OBSERVATION_COUNTS:
        raise ReferenceError("only exact 111/221/441 endpoint observations are supported; no dense output")
    return tuple(range(0, 441, 440//(count-1)))


def load_inputs(repo: Path, archive: Path, case_id: str) -> dict:
    contract = checked_json(repo/CONTRACT, CONTRACT_SHA256)
    checked_json(repo/ANALYTIC, ANALYTIC_SHA256)
    manifest = json.loads((repo/contract["reference_set"]["manifest_path"]).read_text())
    cases = manifest["hi"]["cases"]
    if digest(cases) != contract["reference_set"]["cases_payload_sha256"]:
        raise ReferenceError("historical case ledger identity changed")
    if [x["id"] for x in cases] != contract["reference_set"]["required_case_ids"]:
        raise ReferenceError("45-record coverage/ordering changed")
    selected = [x for x in cases if x["id"] == case_id]
    if len(selected) != 1:
        raise ReferenceError("case must identify one of the 45 original records")
    case = selected[0]
    source = next(x for x in manifest["sources"] if x["id"] == case["source_id"])
    if source["root"] != "archive":
        raise ReferenceError("saved request must come from its bound archive artifact")
    raw_path = archive/source["path"]
    raw = checked_json(raw_path, source["sha256"])
    request = raw["request"]
    if digest(request) != case["request_sha256"] or digest(request["device"]) != case["device_input_sha256"]:
        raise ReferenceError("saved request/device identity mismatch")
    nominal = manifest["hi"]["nominal_request"]
    if digest(nominal) != contract["identity"]["nominal_request_sha256"]:
        raise ReferenceError("nominal request identity mismatch")
    # Compare complete captured dictionaries; never fill absent fields from today's defaults.
    expected = json.loads(json.dumps(nominal))
    absorber = next(layer for layer in expected["device"]["layers"] if layer["role"] == "absorber")
    absorber["D_ion"], absorber["P0"] = case["D_ion_m2_s"], case["c0_m3"]
    expected["params"]["v_rate"] = case["scan_rate_V_s"]
    if request != expected:
        raise ReferenceError("record changes inputs outside its registered D/c0/rate coordinate")
    if request["params"]["N_grid"] != 60 or request["params"]["n_points"] != 111:
        raise ReferenceError("historical configured grid/sampling identity changed")
    return {"contract": contract, "case": case, "request": request,
            "raw_input": {"path": str(raw_path), "sha256": source["sha256"]},
            "coverage": {"records": len(cases), "request_identities": len({x["request_sha256"] for x in cases}),
                         "all_case_ids": [x["id"] for x in cases],
                         "execution_reuse": "none;43 equal requests are only potential full-identity reuse, not45-to43 certification"}}


def environment_binding() -> dict:
    distributions = {}
    for name in ("numpy", "scipy", "PyYAML", "fastapi", "pydantic", "threadpoolctl"):
        try:
            dist = importlib.metadata.distribution(name)
            distributions[name] = {"version": dist.version,
                                   "record_sha256": hashlib.sha256((dist.read_text("RECORD") or "").encode()).hexdigest()}
        except importlib.metadata.PackageNotFoundError:
            distributions[name] = None
    return {"executable": sys.executable, "executable_sha256": file_digest(Path(sys.executable)),
            "prefix": sys.prefix, "python": sys.version, "machine": platform.machine(),
            "platform": platform.platform(), "dependencies": distributions,
            "model_environment": {k: v for k, v in sorted(os.environ.items())
                                  if k in THREAD_KEYS or k.startswith(("SOLARLAB_", "PEROVSKITE_"))},
            "qualification": "identity capture only;actual loaded library/thread and dependency bytes need separate qualification"}


def source_binding(repo: Path) -> dict:
    def git(*args):
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()
    if git("branch", "--show-current") != "devel":
        raise ReferenceError("reference source must be on devel")
    paths = git("ls-files", "perovskite-sim/perovskite_sim", "perovskite-sim/backend").splitlines()
    pins = {p: file_digest(repo/p) for p in paths if p.endswith(".py")}
    pins["scripts/benchmarks/hysteresis_reference.py"] = file_digest(Path(__file__))
    return {"head": git("rev-parse", "HEAD"), "branch": "devel", "files": pins,
            "files_sha256": digest(pins), "scope": "actual tracked Python production/backend tree plus this isolated harness;no Git mutation"}


def prebias_first_step(value, control: Control) -> float | None:
    """Validate the optional startup step without changing any physical control."""
    if value is None:
        return None
    if (type(value) not in (int, float) or not math.isfinite(value)
            or not 0 < value <= min(control.duration, control.max_step_s)):
        raise ReferenceError("prebias first_step must be finite, positive and within the control cap")
    return float(value)


def checked_factor_cache_policy(value):
    if value is not None and (type(value) is not str or value != EXACT_DENSE_FACTOR_POLICY):
        raise ReferenceError("unsupported factor cache policy")
    return value


def build_plan(repo: Path, archive: Path, case_id: str, grid: int, step_level: int,
               solver_level: int, *, prebias_first_step_s=None, factor_cache_policy=None) -> dict:
    policy = checked_factor_cache_policy(factor_cache_policy)
    bundle = load_inputs(repo, archive, case_id)
    c = bundle["contract"]
    if type(grid) is not int or grid not in c["refinement_plan"]["phase1_spatial"]["requested_N"]:
        raise ReferenceError("grid request is outside the frozen ladder")
    if type(solver_level) is not int or solver_level not in (0, 1, 2):
        raise ReferenceError("solver level must be0,1,2")
    phases = make_timeline(c, bundle["request"], step_level)
    controls = [x.binding() for p in phases for x in p.controls]
    solver = c["refinement_plan"]["phase1_solver"]
    plan = {"schema": "solarlab.hysteresis_reference_run_plan.v1",
            "case_id": case_id, "request": bundle["request"], "raw_input": bundle["raw_input"],
            "coverage": bundle["coverage"], "contract_sha256": CONTRACT_SHA256,
            "analytic_gates_sha256": ANALYTIC_SHA256, "controls": controls,
            "control_sha256": digest(controls), "source": source_binding(repo),
            "environment": environment_binding(),
            "numerics": {"requested_N_grid": grid, "expected_actual_nodes": grid+1,
                         "max_step_level": step_level, "solver_level": solver_level,
                         "rtol": solver["rtol_values"][solver_level],
                         "atol_m3": solver["density_atol_m3_values"][solver_level],
                         "method": "Radau", "jacobian": "analytic_single_ion_density;no silent fallback",
                         "initial_state": "current-source solve_equilibrium quasi-neutral seed, once before120s dark phase",
                         "state_coordinates": "density", "hold_fallbacks": "none;failure does not change frozen control grid"},
            "resources": {"wall_s": 600, "RSS_bytes": 8*1024**3, "artifact_bytes": 4*1024**3,
                          "scientific_processes": 1, "BLAS_threads": 1},
            "source_channel": "current_source_reference_rebaseline_unqualified;not an exact historical binary replay",
            "accepted_history": "public Radau.step observer;initial input plus accepted states;streamed native prefixes",
            "accepted_stream_encoding": "concatenated NumPy arrays [local_time_s, full_packed_density_state];control index selects frozen RunPlan controls",
            "observation_capability": "exact control endpoints only;no dense output",
            "current_qualification": "pending_NC06_and_independent_device_derivative_and_refinement_errors",
            "execution_authorized": False}
    first_step = prebias_first_step(prebias_first_step_s, phases[1].controls[0])
    if first_step is not None:
        plan["numerics"]["prebias_first_step_s"] = first_step
    if policy is not None:
        plan["numerics"]["factor_cache_policy"] = policy
    plan["identity_sha256"] = digest(plan)
    return plan


@dataclass(frozen=True)
class Segment:
    local_times: tuple[float, ...]
    states: tuple[tuple[float, ...], ...]
    success: bool
    diagnostics: dict


def collect_history(adapter, phases: tuple[Phase, ...], save_segment, save_event) -> dict:
    """Carry the complete state. Output observations cannot influence this loop."""
    state = tuple(adapter.initial_state())
    history = {"phases": {}, "events": [], "initial_state": state}
    previous_point = None
    for phase in phases:
        start_point = adapter.observe(state, phase, phase.start, "right")
        if previous_point is not None:
            event = event_record(previous_point, start_point)
            history["events"].append(event)
            save_event(event)
        else:
            save_event({"kind": "initial_state", "point": start_point})
        knots = [state]
        for control in phase.controls:
            segment = adapter.advance(state, control)
            save_segment(control, segment)
            if not segment.success:
                raise ReferenceError(f"integration failed at{phase.id}[{control.index}]:{segment.diagnostics}")
            if (len(segment.local_times) < 2 or len(segment.states) != len(segment.local_times)
                    or segment.local_times[0] != 0 or segment.local_times[-1] != control.duration
                    or any(b <= a for a, b in zip(segment.local_times, segment.local_times[1:]))):
                raise ReferenceError("accepted raw trajectory does not span the exact control interval")
            if segment.states[0] != state:
                raise ReferenceError("adapter reset/lost state at a control restart")
            if any(len(row) != len(state) or not all(map(math.isfinite, row)) for row in segment.states):
                raise ReferenceError("raw trajectory changed state layout or contains nonfinite values")
            state = segment.states[-1]
            knots.append(state)
        end_point = adapter.observe(state, phase, phase.stop, "left")
        save_event({"kind": "phase_end", "point": end_point})
        history["phases"][phase.id] = {"phase": phase, "states": tuple(knots),
                                      "start": start_point, "end": end_point}
        previous_point = end_point
    return history


def event_record(before: dict, after: dict) -> dict:
    if before["time_s"] != after["time_s"] or before["state"] != after["state"]:
        raise ReferenceError("event must carry the same full state at the same physical time")
    delta = [b-a for a, b in zip(before["D_C_m2"], after["D_C_m2"], strict=True)]
    return {"time_s": after["time_s"], "before": before, "after": after,
            "delta_D_C_m2": delta,
            "electrode_impulse_charge_C_m2": {"left": delta[0], "right": -delta[-1]},
            "impulse_current_A_m2": None,
            "rule": "event charge is retained;no finite current replaces a voltage impulse"}


def current_value(point: dict, *, previous: dict | None = None) -> dict:
    polarity = point["junction_polarity"]
    if polarity not in (-1, 1):
        raise ReferenceError("explicit junction orientation is required")
    if previous is None:
        displacement = [-polarity*x for x in point["Ddot_A_m2"]]
        name = "instantaneous_ramp_side_v1"
    else:
        dt = point["time_s"]-previous["time_s"]
        if dt <= 0 or previous["phase_id"] != point["phase_id"]:
            raise ReferenceError("interval current needs positive time within one physical phase")
        displacement = [-polarity*(b-a)/dt for a, b in zip(previous["D_C_m2"], point["D_C_m2"], strict=True)]
        name = "legacy_interval_endpoint"
    total = [a+b for a, b in zip(point["J_cond_A_m2"], displacement, strict=True)]
    return {"observable": name, "phase_id": point["phase_id"], "time_s": point["time_s"],
            "voltage_V": point["voltage_V"], "event_side": point["event_side"],
            "J_n_A_m2": point["J_n_A_m2"], "J_p_A_m2": point["J_p_A_m2"],
            "J_ion_A_m2": point["J_ion_A_m2"], "J_disp_A_m2": displacement,
            "J_total_A_m2": total, "left_solar_current_A_m2": total[0],
            "uncertainty_A_m2": None, "qualification": "unqualified;NC06/device/error bounds pending"}


def observe_branches(adapter, history: dict, count: int) -> dict:
    indices = observation_indices(count)
    result = {"samples_per_branch": count, "instantaneous_ramp_side_v1": {},
              "legacy_interval_endpoint": {}, "legacy_samples_per_branch": 111,
              "historical_first_point_omissions": 2, "new_first_points_retained": True}
    for direction in ("forward", "reverse"):
        saved = history["phases"][direction+"_ramp"]
        phase = saved["phase"]
        def point(index):
            t = phase.start + (phase.stop-phase.start)*index/440
            return adapter.observe(saved["states"][index], phase, t, "left" if index == 440 else "right")
        result["instantaneous_ramp_side_v1"][direction] = [current_value(point(i)) for i in indices]
        # Historical first sample is a within-light-dwell interval, NOT a ramp derivative.
        dwell = history["phases"][direction+"_light_dwell"]
        first = current_value(dwell["end"], previous=dwell["start"])
        first["historical_reference"] = "missing;original stored-current reconstruction omitted this point"
        first["first_point_role"] = "dwell_interval_endpoint;not ramp-start instantaneous current"
        legacy = [first]
        for index in range(4, 441, 4):
            legacy.append(current_value(point(index), previous=point(index-4)))
        result["legacy_interval_endpoint"][direction] = legacy
    return result


def power_summary(branches: dict, observable: str) -> dict:
    powers = {}
    for name in ("forward", "reverse"):
        rows = branches[observable][name]
        powers[name] = max(r["voltage_V"]*r["left_solar_current_A_m2"] for r in rows if r["voltage_V"] >= 0)
    f, r = powers["forward"], powers["reverse"]
    return {"observable": observable, "sampled_Pmax_W_m2": powers,
            "HI_P": None if f == 0 else r/f-1,
            "HI_normalized": None if r == 0 else (r-f)/r,
            "power_uncertainty_W_m2": None, "reference_power_precondition": "unknown",
            "scope": "sample maxima only;no PCHIP, continuum optimum, Voc/FF/PCE or error certification"}


def differentiated_poisson(factor, rho_dot, right_voltage_rate: float):
    """Reuse the fixed production Poisson operator; return physical Ddot/residual.

    This is a derivative calculation, not its independent error certificate.
    No variable-permittivity, moving-grid or extra trap charge term is assumed.
    """
    import numpy as np
    from perovskite_sim.physics.poisson import solve_poisson_prefactored
    phi_dot = solve_poisson_prefactored(factor, rho_dot, 0., right_voltage_rate)
    Ddot = -factor.C*np.diff(phi_dot)
    residual = np.diff(Ddot)-rho_dot[1:-1]*factor.h_cell
    return phi_dot, Ddot, residual


class ExactDenseLUCache:
    """Two owned factors for one Radau instance's unchanged dense ``lu``.

    Keys are complete dtype/shape/C-order byte strings, captured before the
    backend may overwrite its argument. One real and one complex entry are
    retained; no digest or approximate comparison decides a hit. Only ordinary
    writable, contiguous float64/complex128 square finite arrays are eligible.
    Everything else reaches the original callable, with its original errors.

    The inspected SciPy dense backend warns on a zero U diagonal. Such results,
    and nonfinite factors, are never retained. No warning filter/handler is
    changed. Owned read-only copies also isolate factors from caller aliases;
    the consumer is the unchanged, separately verified Radau ``solve_lu``.
    """

    def __init__(self, original_lu):
        import numpy as np
        self._np = np
        self._original_lu = original_lu
        self._entries = {}
        self.statistics = {"policy": EXACT_DENSE_FACTOR_POLICY,
                           "requests": 0, "hits": 0, "misses": 0, "bypasses": 0,
                           "uncacheable_results": 0, "retained_entries": 0,
                           "retained_bytes": 0, "peak_retained_bytes": 0}

    def __call__(self, matrix):
        np, stats = self._np, self.statistics
        stats["requests"] += 1
        eligible = (type(matrix) is np.ndarray and matrix.ndim == 2
                    and matrix.shape[0] == matrix.shape[1] and matrix.size > 0
                    and matrix.dtype in (np.dtype("float64"), np.dtype("complex128"))
                    and matrix.dtype.metadata is None and matrix.flags.writeable
                    and (matrix.flags.c_contiguous or matrix.flags.f_contiguous)
                    and np.isfinite(matrix).all())
        if not eligible:
            stats["misses"] += 1
            stats["bypasses"] += 1
            return self._original_lu(matrix)
        key = (matrix.dtype.str, matrix.shape, matrix.tobytes(order="C"))
        slot = matrix.dtype.kind
        entry = self._entries.get(slot)
        if entry is not None and entry[0] == key:
            stats["hits"] += 1
            return entry[1]
        stats["misses"] += 1  # Only the unchanged callable increments Radau.nlu.
        factors = self._original_lu(matrix)
        lu, piv = factors
        if (lu.dtype != matrix.dtype or lu.shape != key[1]
                or piv.shape != (key[1][0],) or piv.dtype.kind not in "iu"
                or not np.isfinite(lu).all() or np.any(lu.diagonal() == 0)):
            stats["uncacheable_results"] += 1
            return factors
        # A Fortran-contiguous caller array can alias the returned LU. Neither
        # it nor the returned miss factor may alias the retained entry.
        owned = (lu.copy(order="K"), piv.copy())
        for array in owned:
            array.flags.writeable = False
        self._entries[slot] = (key, owned)
        stats["retained_entries"] = len(self._entries)
        stats["retained_bytes"] = sum(len(k[2])+f[0].nbytes+f[1].nbytes
                                      for k, f in self._entries.values())
        stats["peak_retained_bytes"] = max(stats["peak_retained_bytes"], stats["retained_bytes"])
        return factors


def radau_history(initial_state, observer=None, *, first_step=None, factor_cache_policy=None):
    """Observe public successful steps without changing the Radau algorithm.

    The first record is the input initial state. Later records come only from
    successful ``Radau.step`` calls; failed trials are never called accepted.
    The independent recorder survives the legacy wrapper's empty failure result.
    """
    from scipy.integrate import Radau

    policy = checked_factor_cache_policy(factor_cache_policy)
    times = [0.]
    states = [tuple(map(float, initial_state))]
    cache_reports = []  # Counter dictionaries only; never retain solver/factor objects here.

    class ObservedRadau(Radau):
        factor_cache_reports = cache_reports

        def __init__(self, *args, **kwargs):
            if first_step is not None:
                if kwargs.get("first_step") not in (None, first_step):
                    raise ReferenceError("conflicting requested Radau first_step")
                kwargs["first_step"] = first_step
            super().__init__(*args, **kwargs)
            if policy is not None:
                self.lu = ExactDenseLUCache(self.lu)
                cache_reports.append(self.lu.statistics)

        def step(self):
            previous = self.t
            message = super().step()
            if self.status != "failed" and self.t > previous:
                time_value = float(self.t)
                state = tuple(map(float, self.y))
                if time_value <= times[-1]:
                    raise ReferenceError("accepted Radau time did not advance")
                times.append(time_value)
                states.append(state)
                if observer is not None:
                    observer(time_value, state)
            return message

    return ObservedRadau, times, states


class ProductionBridge:
    """Thin public-engine bridge, restricted to the exact original simple HI deck."""

    def __init__(self, request: dict, numerics: dict):
        import numpy as np
        from backend.main import stack_from_dict
        from perovskite_sim.experiments import jv_sweep as jv
        from perovskite_sim.experiments.waveform_jv import uniform_absorber_generation
        from perovskite_sim.experiments.waveform_jacobian import build_waveform_density_jacobian
        from dataclasses import replace
        self.np, self.jv, self.numerics = np, jv, numerics
        self.stack = stack_from_dict(request["device"])
        jv.require_jv_driver_capability(self.stack, requested_driver="transient")
        self.x = jv.build_electrical_grid(self.stack, numerics["requested_N_grid"])
        jv.require_thick_layer_interface_resolution(self.x, self.stack,
            N_grid=numerics["requested_N_grid"], allow_underresolved_grid=False)
        self.mat = jv.build_material_arrays(self.x, self.stack)
        self.mat = replace(self.mat, G_optical=uniform_absorber_generation(
            self.x, self.stack, request["params"]["waveform"]["uniform_generation_rate_m3_s"]))
        if (len(self.x) != numerics["expected_actual_nodes"] or self.mat.has_dual_ions
                or self.mat.N_iface_state or self.stack.interface_charge_closure != "off"
                or self.mat.bulk_trap_charge_closure != "off"):
            raise ReferenceError("original HI state layout/charge model unsupported;no blocks may be dropped")
        self.active_control = None
        self.jacobian = build_waveform_density_jacobian(
            self.x, self.stack, self.mat, lambda t: self.active_control.voltage(t))
        self.initial_inventory = None
        self.accepted_observer = None

    def initial_state(self):
        from perovskite_sim.physics.generation import dual_cell_widths
        state = self.jv.solve_equilibrium(self.x, self.stack)
        sv = self.jv.StateVec.unpack(state, len(self.x), self.mat.N_iface_state)
        if not self.np.array_equal(sv.P, self.mat.P_ion0):
            raise ReferenceError("initial mobile population/background coupling changed")
        self.widths = dual_cell_widths(self.x)
        self.initial_inventory = float(sv.P @ self.widths)
        return tuple(map(float, state))

    def advance(self, state, control: Control) -> Segment:
        self.active_control = control
        sink = getattr(self, "accepted_observer", None)
        observer = None if sink is None else lambda time_value, values: sink(control, time_value, values)
        requested_first_step = (prebias_first_step(self.numerics.get("prebias_first_step_s"), control)
                                if control.phase_id == "dark_prebias" else None)
        startup = {} if requested_first_step is None else {"first_step": requested_first_step}
        policy = checked_factor_cache_policy(self.numerics.get("factor_cache_policy"))
        if policy is not None:
            startup["factor_cache_policy"] = policy
        method, accepted_times, accepted_states = radau_history(state, observer, **startup)
        def cache_diagnostics():
            return {} if policy is None else {"factor_cache": {
                "policy": policy, "solvers": [dict(row) for row in method.factor_cache_reports]}}
        try:
            result = self.jv.run_transient(
                x=self.x, y0=self.np.asarray(state, dtype=float), stack=self.stack,
                t_span=(0., control.duration), t_eval=None, V_app=control.voltage,
                illuminated=control.illuminated, mat=self.mat, method=method,
                max_step=control.max_step_s, max_nfev=control.max_rhs_calls,
                rtol=self.numerics["rtol"], atol=self.numerics["atol_m3"],
                jacobian=self.jacobian, state_coordinates="density")
            report = getattr(result, "numerical_diagnostics", None)
            diag = {"message": str(result.message), "nfev": getattr(result, "nfev", None),
                    "njev": getattr(result, "njev", None), "nlu": getattr(result, "nlu", None),
                    "numerical_diagnostics": asdict(report) if is_dataclass(report) else None,
                    "dense_output": False, "accepted_mesh": "solve_ivp t_eval=None",
                    "accepted_history_source": "public Radau.step;initial input separately retained",
                    "accepted_steps": len(accepted_times)-1,
                    "last_accepted_local_time_s": accepted_times[-1] if len(accepted_times) > 1 else None,
                    "solver_method": "Radau with observational step subclass"}
            diag.update(cache_diagnostics())
            if requested_first_step is not None:
                diag["requested_first_step_s"] = requested_first_step
            values = self.np.asarray(result.y)
            success = bool(result.success)
            returned_times = self.np.asarray(result.t)
            if returned_times.size:
                if (not self.np.array_equal(returned_times, accepted_times)
                        or not self.np.array_equal(values.T, accepted_states)):
                    raise ReferenceError("solver output differs from the recorded accepted mesh")
            elif success:
                raise ReferenceError("successful solve returned no accepted trajectory")
            # On budget/nonfinite failure the legacy wrapper returns empty t/y.
            # Retain the observed prefix, including its explicitly named seed.
            values = self.np.asarray(accepted_states).T
            if values.ndim == 2 and values.shape[1]:
                n = len(self.x)
                population = values[2*n:3*n]
                inventories = self.widths @ population
                if self.initial_inventory == 0:
                    drift = 0. if not self.np.any(population != 0) else None
                    inventory_valid = drift == 0. and not self.np.any(self.mat.P_ion0 != 0)
                else:
                    drift = float(self.np.max(self.np.abs(inventories-self.initial_inventory))/abs(self.initial_inventory))
                    inventory_valid = math.isfinite(drift) and drift <= 1e-6
                diag["max_inventory_relative_drift_over_accepted_knots"] = drift
                diag["inventory_valid_at_all_accepted_knots"] = bool(inventory_valid)
                if not inventory_valid:
                    success = False
                    diag["reference_failure"] = "original ion-inventory guard failed within this raw trajectory"
            rows = tuple(tuple(map(float, row)) for row in values.T) if values.ndim == 2 else ()
            return Segment(tuple(accepted_times), rows, success, diag)
        except Exception as error:
            return Segment(tuple(accepted_times), tuple(accepted_states), False,
                           {"exception": repr(error), "traceback": traceback.format_exc(),
                            "accepted_steps": len(accepted_times)-1,
                            "last_accepted_local_time_s": accepted_times[-1] if len(accepted_times) > 1 else None,
                            "within_interval_raw_history": "public Radau.step prefix;first record is input seed",
                            **cache_diagnostics()})

    def observe(self, state, phase: Phase, physical_time: Fraction, side: str) -> dict:
        from perovskite_sim.constants import Q
        from perovskite_sim.solver.mol import assemble_rhs
        np = self.np
        y = np.asarray(state, dtype=float)
        voltage = phase.voltage(physical_time)
        snapshot = self.jv.extract_spatial_snapshot(self.x, y, self.stack, voltage, mat=self.mat)
        cond = self.jv.compute_current_components(self.x, y, self.stack, voltage, mat=self.mat)
        # This stateless original RHS has no explicit aging/time dependence.
        ydot = assemble_rhs(float(physical_time), y, self.x, self.stack, self.mat,
                            phase.controls[0].illuminated, voltage)
        rates = self.jv.StateVec.unpack(ydot, len(self.x), self.mat.N_iface_state)
        rho_dot = Q*(rates.p-rates.n+rates.P)
        polarity = int(self.mat.junction_polarity)
        phi_dot, Ddot, residual = differentiated_poisson(
            self.mat.poisson_factor, rho_dot, -polarity*phase.voltage_rate)
        D = -self.mat.poisson_factor.C*np.diff(snapshot.phi)
        inventory = float(snapshot.P @ self.widths)
        initial = self.initial_inventory
        if initial == 0:
            if np.any(snapshot.P != 0) or np.any(self.mat.P_ion0 != 0):
                raise ReferenceError("c0=0 developed mobile inventory or background")
            drift = 0.
        else:
            drift = abs(inventory-initial)/abs(initial)
            if drift > 1e-6:
                raise ReferenceError("ion inventory exceeded original1e-6 diagnostic bound")
        if not all(np.all(np.isfinite(a)) for a in (y, ydot, snapshot.phi, D, Ddot, cond.J_total)):
            raise ReferenceError("nonfinite raw state/derivative/current")
        return {"phase_id": phase.id, "time_s": float(physical_time), "event_side": side,
                "voltage_V": voltage, "voltage_rate_V_s": phase.voltage_rate,
                "illuminated": phase.controls[0].illuminated, "junction_polarity": polarity,
                "state": list(state), "n_m3": snapshot.n.tolist(), "p_m3": snapshot.p.tolist(),
                "positive_ions_m3": snapshot.P.tolist(), "negative_ions_m3": None,
                "trap_occupancy": None, "trap_model": "captured static SRH lifetimes;no dynamic trap state",
                "phi_V": snapshot.phi.tolist(), "rho_C_m3": snapshot.rho.tolist(),
                "D_C_m2": D.tolist(), "ydot_m3_s": ydot.tolist(),
                "rho_dot_C_m3_s": rho_dot.tolist(), "phi_dot_V_s": phi_dot.tolist(),
                "Ddot_A_m2": Ddot.tolist(), "differentiated_Poisson_residual_A_m2": residual.tolist(),
                "J_n_A_m2": cond.J_n.tolist(), "J_p_A_m2": cond.J_p.tolist(),
                "J_ion_A_m2": cond.J_ion.tolist(), "J_cond_A_m2": cond.J_total.tolist(),
                "ion_inventory_m2": inventory, "ion_inventory_relative_drift": drift,
                "derivative_error_bound": None, "NC06_device_derivative_qualification": "pending"}


def verify_admission(plan: dict, admission: dict) -> None:
    if (admission.get("schema") != "solarlab.hysteresis_reference_admission.v1"
            or admission.get("execution_authorized") is not True
            or admission.get("identity_sha256") != plan["identity_sha256"]
            or admission.get("resources") != plan["resources"]
            or not admission.get("root_authorization_message")
            or not admission.get("bound_analytic_test_receipt_sha256")
            or admission.get("purpose") != "unqualified_reference_diagnostic"):
        raise ReferenceError("separate exact source/input/environment/protocol/resource admission is required")
    if any(plan["environment"]["model_environment"].get(k) != "1" for k in THREAD_KEYS):
        raise ReferenceError("all six thread environment values must be1 before imports")


def json_write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def execute(plan: dict, bundle: dict, output: Path) -> None:
    """Called only inside the bounded, separately admitted child process."""
    import pickle
    import numpy as np
    phases = make_timeline(bundle["contract"], bundle["request"], plan["numerics"]["max_step_level"])
    bridge = ProductionBridge(bundle["request"], plan["numerics"])
    (output/"Materials.pkl").write_bytes(pickle.dumps({"stack": bridge.stack, "material": bridge.mat,
                                                    "x_m": bridge.x}, protocol=5))
    raw_dir = output/"Raw"
    raw_dir.mkdir()
    journal = (output/"ControlHistory.jsonl").open("x")
    events = (output/"PhysicalEvents.jsonl").open("x")
    counter = 0
    active_stream = None
    active_stream_path = None
    accepted_count = 0
    def save_accepted(control, local_time, state):
        nonlocal active_stream, active_stream_path, accepted_count
        if active_stream is None:
            active_stream_path = raw_dir/f"Interval{counter:04d}Accepted.npy"
            active_stream = active_stream_path.open("xb")
        np.save(active_stream, np.asarray((local_time, *state), dtype=float), allow_pickle=False)
        active_stream.flush()
        accepted_count += 1
    bridge.accepted_observer = save_accepted
    def save_segment(control, segment):
        nonlocal counter, active_stream, active_stream_path, accepted_count
        if active_stream is not None:
            active_stream.close()
            active_stream = None
        accepted_stream = None if active_stream_path is None else {
            "path": str(active_stream_path.relative_to(output)),
            "sha256": file_digest(active_stream_path), "accepted_steps": accepted_count,
            "encoding": "concatenated npy records:local_time_s then full packed state;input seed in interval NPZ",
        }
        path = raw_dir/f"Interval{counter:04d}.npz"
        np.savez(path, local_time_s=np.asarray(segment.local_times),
                 physical_time_s=np.asarray(segment.local_times)+float(control.start),
                 state=np.asarray(segment.states, dtype=float).T)
        journal.write(json.dumps({"control": control.binding(), "success": segment.success,
                                 "raw_path": str(path.relative_to(output)), "raw_sha256": file_digest(path),
                                 "accepted_stream": accepted_stream,
                                 "diagnostics": segment.diagnostics}, allow_nan=False)+"\n")
        journal.flush()
        counter += 1
        active_stream_path, accepted_count = None, 0
    def save_event(event):
        events.write(json.dumps(event, allow_nan=False)+"\n")
        events.flush()
    try:
        history = collect_history(bridge, phases, save_segment, save_event)
        json_write(output/"PhysicalStates.json", {"initial_state": history["initial_state"],
            "phase_limits": {k: {"start": v["start"], "end": v["end"]} for k, v in history["phases"].items()}})
        for count in OBSERVATION_COUNTS:
            branches = observe_branches(bridge, history, count)
            branches["metrics"] = [power_summary(branches, name) for name in
                                   ("legacy_interval_endpoint", "instantaneous_ramp_side_v1")]
            json_write(output/f"Observations{count}.json", branches)
        json_write(output/"Result.json", {"status": "complete_unqualified", "controls_completed": counter,
                   "identity_sha256": plan["identity_sha256"], "current_qualification": "pending",
                   "full_state_layout": "n,p,positive ions;potential algebraic;no dynamic trap/negative-ion block in exact captured deck",
                   "raw_initial_population_equals_background": True,
                   "reference_power_eligibility": "unknown_no_independent_uncertainties",
                   "dense_output": False, "candidate_or_reference_acceptance": False})
    finally:
        if active_stream is not None:
            active_stream.close()
        journal.close()
        events.close()


def supervise(args, plan: dict) -> int:
    """One scientific child; watchdog retains partial files and bounded failures."""
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    json_write(output/"RunPlan.json", plan)
    argv = [sys.executable, "-B", str(Path(__file__).resolve()), "_execute",
            "--repo", str(args.repo.resolve()), "--archive", str(args.archive.resolve()),
            "--case", args.case, "--grid", str(args.grid), "--step-level", str(args.step_level),
            "--solver-level", str(args.solver_level), "--admission", str(args.admission.resolve()),
            "--output", str(output)]
    if args.prebias_first_step_s is not None:
        argv.extend(("--prebias-first-step-s", repr(args.prebias_first_step_s)))
    if getattr(args, "factor_cache_policy", None) is not None:
        argv.extend(("--factor-cache-policy", args.factor_cache_policy))
    started = time.monotonic()
    stop_reason = None
    with (output/"Stdout.txt").open("x") as stdout, (output/"Stderr.txt").open("x") as stderr:
        process = subprocess.Popen(argv, stdout=stdout, stderr=stderr)
        while process.poll() is None:
            elapsed = time.monotonic()-started
            measured = subprocess.run(["ps", "-o", "rss=", "-p", str(process.pid)],
                                      text=True, capture_output=True, check=False)
            if measured.returncode and process.poll() is not None:
                break  # Normal exit between poll and ps, not a fabricated RSS reading.
            rss_text = measured.stdout.strip()
            if measured.returncode or not rss_text:
                stop_reason = "RSS_measurement_unavailable"
            rss = int(rss_text)*1024 if rss_text else None
            size = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
            if elapsed > plan["resources"]["wall_s"]:
                stop_reason = "wall_limit"
            elif rss is not None and rss > plan["resources"]["RSS_bytes"]:
                stop_reason = "RSS_limit"
            elif size > plan["resources"]["artifact_bytes"]:
                stop_reason = "artifact_limit"
            if stop_reason:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                break
            time.sleep(.25)
    json_write(output/"Supervisor.json", {"returncode": process.returncode, "stop_reason": stop_reason,
        "elapsed_wall_s": time.monotonic()-started, "poll_interval_s": .25, "termination_grace_s": 5,
        "resource_enforcement": "wall/RSS/artifact watchdog;bounded polling/grace overshoot retained",
        "qualification": "none"})
    return process.returncode or (1 if stop_reason else 0)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("plan", "run", "_execute"))
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--grid", type=int, default=60)
    parser.add_argument("--step-level", type=int, default=0)
    parser.add_argument("--solver-level", type=int, default=0)
    parser.add_argument("--prebias-first-step-s", type=float,
                        help="optional first Radau step for dark_prebias only; original default unchanged")
    parser.add_argument("--factor-cache-policy", choices=(EXACT_DENSE_FACTOR_POLICY,),
                        help="opt-in exact dense factor reuse; absent retains original Radau factor calls")
    parser.add_argument("--admission", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    plan = build_plan(args.repo, args.archive, args.case, args.grid, args.step_level, args.solver_level,
                      prebias_first_step_s=args.prebias_first_step_s,
                      factor_cache_policy=args.factor_cache_policy)
    if args.operation == "plan":
        if args.output.exists():
            raise ReferenceError("preserve existing plan;choose a new evidence attempt path")
        json_write(args.output, plan)
        return 0
    if args.admission is None:
        raise ReferenceError("run needs a separately issued admission file")
    verify_admission(plan, json.loads(args.admission.read_text()))
    if args.operation == "run":
        return supervise(args, plan)
    saved = json.loads((args.output/"RunPlan.json").read_text())
    if saved != plan:
        raise ReferenceError("source/input/environment changed between parent admission and child")
    sys.path.insert(0, str((args.repo/"perovskite-sim").resolve()))
    try:
        execute(plan, load_inputs(args.repo, args.archive, args.case), args.output)
    except BaseException as error:
        json_write(args.output/"Failure.json", {"error": repr(error), "traceback": traceback.format_exc(),
                   "status": "failed_unqualified", "preserved_controls": "see immutable raw interval files/journal"})
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
