"""Arbitrary piecewise-linear / triangular / hold V(t) waveform J-V recipes.

Integrates arbitrary piecewise-linear voltage and illumination waveforms using
the conservative Backward Euler 1D transient solvers (``solve_transient`` for
ion-free devices and ``solve_ionic_transient`` for coupled ionic-electronic
devices; Backward Euler remains the default time integrator).

Key numerical contracts (P05-02b):

1. **Exact segment/corner event boundaries**: Every segment endpoint is placed
   explicitly in ``TransientProtocol.times_s`` (``corner_indices``); no time
   step ever straddles a slope or illumination discontinuity.
2. **Displacement-inclusive total terminal current with explicit sign names**:
   - ``pv_output_current_A_m2``: photovoltaic output convention (``> 0`` in the
     PV power quadrant, ``-C * dV/dt`` capacitive shift on ascending ramps).
   - ``voltage_conjugate_current_A_m2``: passive port input convention (``> 0``
     under forward injection and ``+C * dV/dt`` capacitive charging on
     ascending ramps).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields, is_dataclass, replace
import hashlib
import json
import math
from typing import Any, Literal

import numpy as np

from solarlab.equations.dc_1d import CompiledDevice1D
from solarlab.equations.ionic_1d import CompiledIonicDevice1D, Ionic1DProblem
from solarlab.experiments.ionic_transient import (
    IonicTransientAttempt,
    IonicTransientInitial,
    IonicTransientOptions,
    IonicTransientResult,
    solve_ionic_transient,
)
from solarlab.experiments.jv.steady import SteadyJVMetrics, steady_jv_metrics
from solarlab.experiments.suns_voc import (
    default_ionic_steady_options,
    solve_ionic_dc_operating_point,
)
from solarlab.experiments.transient import (
    TransientAttempt,
    TransientOptions,
    TransientProtocol,
    TransientResult1D,
    solve_transient,
)
from solarlab.reproducibility.job_provenance import current_provenance
from solarlab.solvers.nonlinear.ionic_1d import (
    IonicSteadyOptions,
    IonicSteadyResult,
)
from solarlab.solvers.nonlinear.problem import SolveBudget
from solarlab.solvers.nonlinear.production import POISSON_CONTACT_BALANCE_V1
from solarlab.solvers.nonlinear.steady_1d import (
    SteadyOptions,
    SteadyState1D,
    solve_steady_state,
)

__all__ = [
    "JVWaveform",
    "PiecewiseWaveform",
    "WaveformJVOptions",
    "WaveformJVResult",
    "WaveformSegment",
    "build_piecewise_waveform",
    "build_triangular_jv_waveform",
    "package_ionic_steady_initial",
    "run_waveform_jv",
]


def _ro_float(value: Any) -> np.ndarray:
    arr = np.array(value, dtype=np.float64, copy=True)
    arr.setflags(write=False)
    return arr


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating)):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class JVWaveform:
    """Explicit triangular hysteresis protocol specification."""

    start_voltage_V: float = 0.0
    dark_seed_s: float = 0.0
    dark_prep_s: float = 0.0
    branch_dwell_s: float = 0.0
    turnaround_s: float = 0.0
    turnaround_dark: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "start_voltage_V", _finite_float(self.start_voltage_V, "start_voltage_V")
        )
        for name in ("dark_seed_s", "dark_prep_s", "branch_dwell_s", "turnaround_s"):
            val = _finite_float(getattr(self, name), name)
            if val < 0.0:
                raise ValueError(f"{name} must be >= 0")
            object.__setattr__(self, name, val)
        if not isinstance(self.turnaround_dark, bool):
            raise TypeError("turnaround_dark must be a bool")

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> JVWaveform:
        if not isinstance(value, Mapping):
            raise TypeError("waveform must be a mapping")
        expected = {f.name for f in fields(cls)}
        if set(value.keys()) != expected:
            raise ValueError(f"JVWaveform requires exact keys {sorted(expected)}")
        return cls(**value)


@dataclass(frozen=True)
class WaveformSegment:
    """One linear voltage ramp or hold interval with exact endpoint boundaries."""

    name: str
    duration_s: float
    start_voltage_V: float
    stop_voltage_V: float
    generation_scale: float = 1.0
    n_steps: int = 10
    role: Literal["preconditioning", "hold", "forward_ramp", "reverse_ramp", "custom"] = (
        "custom"
    )

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("segment name must be a non-empty string")
        dur = _finite_float(self.duration_s, "duration_s")
        if dur <= 0.0:
            raise ValueError("segment duration_s must be finite and positive")
        v0 = _finite_float(self.start_voltage_V, "start_voltage_V")
        v1 = _finite_float(self.stop_voltage_V, "stop_voltage_V")
        gen = _finite_float(self.generation_scale, "generation_scale")
        if gen < 0.0:
            raise ValueError("segment generation_scale must be >= 0")
        if type(self.n_steps) is not int or self.n_steps < 1:
            raise ValueError("segment n_steps must be an integer >= 1")
        if self.role not in {
            "preconditioning",
            "hold",
            "forward_ramp",
            "reverse_ramp",
            "custom",
        }:
            raise ValueError(f"unsupported segment role: {self.role!r}")
        object.__setattr__(self, "duration_s", dur)
        object.__setattr__(self, "start_voltage_V", v0)
        object.__setattr__(self, "stop_voltage_V", v1)
        object.__setattr__(self, "generation_scale", gen)


@dataclass(frozen=True)
class PiecewiseWaveform:
    """Discretized piecewise-linear V(t) protocol with exact corner indices."""

    segments: tuple[WaveformSegment, ...]
    initial_voltage_V: float
    initial_generation_scale: float
    protocol: TransientProtocol
    corner_times_s: np.ndarray
    corner_indices: tuple[int, ...]
    segment_step_slices: tuple[slice, ...]
    step_dV_dt_V_s: np.ndarray
    protocol_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "corner_times_s", _ro_float(self.corner_times_s))
        object.__setattr__(self, "step_dV_dt_V_s", _ro_float(self.step_dV_dt_V_s))


@dataclass(frozen=True)
class WaveformJVOptions:
    """Numerical controls for conservative Backward Euler waveform J-V."""

    steady: SteadyOptions = field(default_factory=SteadyOptions)
    ionic_steady: IonicSteadyOptions | None = None
    transient: TransientOptions | None = None
    ionic_transient: IonicTransientOptions | None = None
    method: Literal["backward_euler"] | None = None
    incident_power_W_m2: float | None = None
    require_completed: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.steady, SteadyOptions):
            raise ValueError("steady must be an explicit SteadyOptions instance")
        if self.ionic_steady is not None and not isinstance(
            self.ionic_steady, IonicSteadyOptions
        ):
            raise ValueError(
                "ionic_steady must be None or an explicit IonicSteadyOptions"
            )
        if self.method not in (None, "backward_euler"):
            raise ValueError(
                "WaveformJVOptions supports only Backward Euler (None or 'backward_euler')"
            )
        if self.incident_power_W_m2 is not None and (
            not math.isfinite(self.incident_power_W_m2)
            or self.incident_power_W_m2 <= 0.0
        ):
            raise ValueError("incident_power_W_m2 must be finite and positive")


@dataclass(frozen=True)
class WaveformJVResult:
    """Full step-resolved and branch-resolved waveform J-V result."""

    waveform: PiecewiseWaveform
    completed: bool
    accepted_steps: int
    requested_steps: int
    step_times_s: np.ndarray
    step_dt_s: np.ndarray
    step_voltage_V: np.ndarray
    step_dV_dt_V_s: np.ndarray
    step_generation_scale: np.ndarray
    pv_output_current_A_m2: np.ndarray
    pv_conduction_current_A_m2: np.ndarray
    pv_displacement_current_A_m2: np.ndarray
    voltage_conjugate_current_A_m2: np.ndarray
    voltage_conjugate_conduction_A_m2: np.ndarray
    voltage_conjugate_displacement_A_m2: np.ndarray
    V_fwd: np.ndarray | None
    J_fwd: np.ndarray | None
    V_rev: np.ndarray | None
    J_rev: np.ndarray | None
    metrics_fwd: SteadyJVMetrics | None
    metrics_rev: SteadyJVMetrics | None
    hysteresis_index_rev_norm: float | None
    hysteresis_index_fwd_norm: float | None
    inventory_relative_drift: tuple[float, ...]
    voltage_polarity: float
    compiled_identity: str
    transient_result: TransientResult1D | IonicTransientResult
    failure_reason: str | None = None
    job_provenance: dict[str, Any] | None = field(default_factory=current_provenance)

    def __post_init__(self) -> None:
        for name in (
            "step_times_s",
            "step_dt_s",
            "step_voltage_V",
            "step_dV_dt_V_s",
            "step_generation_scale",
            "pv_output_current_A_m2",
            "pv_conduction_current_A_m2",
            "pv_displacement_current_A_m2",
            "voltage_conjugate_current_A_m2",
            "voltage_conjugate_conduction_A_m2",
            "voltage_conjugate_displacement_A_m2",
        ):
            object.__setattr__(self, name, _ro_float(getattr(self, name)))
        for opt_name in ("V_fwd", "J_fwd", "V_rev", "J_rev"):
            arr = getattr(self, opt_name)
            if arr is not None:
                object.__setattr__(self, opt_name, _ro_float(arr))

    @property
    def hysteresis_index(self) -> float | None:
        return self.hysteresis_index_rev_norm

    @property
    def hysteresis_index_paper(self) -> float | None:
        return self.hysteresis_index_fwd_norm


def build_piecewise_waveform(
    segments: Sequence[WaveformSegment],
    *,
    initial_voltage_V: float | None = None,
    initial_generation_scale: float | None = None,
    continuity_tol_V: float = 1e-12,
) -> PiecewiseWaveform:
    """Construct a ``PiecewiseWaveform`` with exact segment corner nodes."""
    seg_tuple = tuple(segments)
    if not seg_tuple:
        raise ValueError("segments must contain at least one WaveformSegment")
    for idx in range(len(seg_tuple) - 1):
        left, right = seg_tuple[idx], seg_tuple[idx + 1]
        if abs(right.start_voltage_V - left.stop_voltage_V) > continuity_tol_V:
            raise ValueError(
                f"discontinuous waveform voltage between segment {left.name!r} "
                f"({left.stop_voltage_V} V) and {right.name!r} ({right.start_voltage_V} V); "
                f"insert an explicit transition segment instead of a step jump"
            )

    v_init = (
        seg_tuple[0].start_voltage_V
        if initial_voltage_V is None
        else _finite_float(initial_voltage_V, "initial_voltage_V")
    )
    if abs(v_init - seg_tuple[0].start_voltage_V) > continuity_tol_V:
        raise ValueError(
            "initial_voltage_V must match the first segment's start_voltage_V"
        )
    g_init = (
        seg_tuple[0].generation_scale
        if initial_generation_scale is None
        else _finite_float(initial_generation_scale, "initial_generation_scale")
    )
    if g_init < 0.0:
        raise ValueError("initial_generation_scale must be >= 0")

    times_list: list[float] = [0.0]
    volt_list: list[float] = [v_init]
    gen_list: list[float] = [g_init]
    corner_times: list[float] = [0.0]
    corner_indices: list[int] = [0]
    step_slices: list[slice] = []
    dv_dt_list: list[float] = []

    t_cursor = 0.0
    step_cursor = 0
    for seg in seg_tuple:
        n_s = seg.n_steps
        slope = (seg.stop_voltage_V - seg.start_voltage_V) / seg.duration_s
        for k in range(1, n_s + 1):
            frac = k / n_s
            t_k = t_cursor + seg.duration_s * frac
            v_k = seg.start_voltage_V + (seg.stop_voltage_V - seg.start_voltage_V) * frac
            times_list.append(t_k)
            volt_list.append(v_k)
            gen_list.append(seg.generation_scale)
            dv_dt_list.append(slope)
        t_cursor += seg.duration_s
        corner_times.append(t_cursor)
        corner_indices.append(len(times_list) - 1)
        step_slices.append(slice(step_cursor, step_cursor + n_s))
        step_cursor += n_s

    protocol = TransientProtocol(
        times_s=np.asarray(times_list, dtype=float),
        voltage_V=np.asarray(volt_list, dtype=float),
        generation_scale=np.asarray(gen_list, dtype=float),
    )
    canonical = json.dumps(
        {
            "initial_voltage_V": v_init,
            "initial_generation_scale": g_init,
            "segments": [asdict(s) for s in seg_tuple],
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return PiecewiseWaveform(
        segments=seg_tuple,
        initial_voltage_V=v_init,
        initial_generation_scale=g_init,
        protocol=protocol,
        corner_times_s=np.asarray(corner_times, dtype=float),
        corner_indices=tuple(corner_indices),
        segment_step_slices=tuple(step_slices),
        step_dV_dt_V_s=np.asarray(dv_dt_list, dtype=float),
        protocol_sha256=digest,
    )


def build_triangular_jv_waveform(
    waveform: JVWaveform | None = None,
    *,
    V_max: float = 1.2,
    v_rate: float = 0.04,
    n_points: int = 61,
    generation_scale: float = 1.0,
) -> PiecewiseWaveform:
    """Build a triangular forward/reverse J-V waveform with exact corner events."""
    wf = JVWaveform() if waveform is None else waveform
    v_stop = _finite_float(V_max, "V_max")
    rate = _finite_float(v_rate, "v_rate")
    gen = _finite_float(generation_scale, "generation_scale")
    if rate <= 0.0:
        raise ValueError("v_rate must be positive")
    if gen < 0.0:
        raise ValueError("generation_scale must be nonnegative")
    if type(n_points) is not int or n_points < 2:
        raise ValueError("n_points must be an integer >= 2")
    if v_stop <= wf.start_voltage_V:
        raise ValueError("V_max must exceed waveform.start_voltage_V")

    ramp_steps = n_points - 1
    ramp_duration = (v_stop - wf.start_voltage_V) / rate
    segments: list[WaveformSegment] = []

    if wf.dark_seed_s > 0.0 and wf.start_voltage_V == 0.0:
        segments.append(
            WaveformSegment(
                "dark_seed_0V",
                wf.dark_seed_s,
                0.0,
                0.0,
                0.0,
                n_steps=max(2, min(10, ramp_steps)),
                role="preconditioning",
            )
        )
    if wf.dark_prep_s > 0.0:
        segments.append(
            WaveformSegment(
                "dark_prebias",
                wf.dark_prep_s,
                wf.start_voltage_V,
                wf.start_voltage_V,
                0.0,
                n_steps=max(2, min(10, ramp_steps)),
                role="preconditioning",
            )
        )
    if wf.branch_dwell_s > 0.0:
        segments.append(
            WaveformSegment(
                "forward_dwell",
                wf.branch_dwell_s,
                wf.start_voltage_V,
                wf.start_voltage_V,
                gen,
                n_steps=max(2, min(10, ramp_steps)),
                role="hold",
            )
        )

    segments.append(
        WaveformSegment(
            "forward_ramp",
            ramp_duration,
            wf.start_voltage_V,
            v_stop,
            gen,
            n_steps=ramp_steps,
            role="forward_ramp",
        )
    )
    if wf.turnaround_s > 0.0:
        segments.append(
            WaveformSegment(
                "turnaround_hold",
                wf.turnaround_s,
                v_stop,
                v_stop,
                0.0 if wf.turnaround_dark else gen,
                n_steps=max(2, min(10, ramp_steps)),
                role="hold",
            )
        )
    if wf.branch_dwell_s > 0.0:
        segments.append(
            WaveformSegment(
                "reverse_dwell",
                wf.branch_dwell_s,
                v_stop,
                v_stop,
                gen,
                n_steps=max(2, min(10, ramp_steps)),
                role="hold",
            )
        )
    segments.append(
        WaveformSegment(
            "reverse_ramp",
            ramp_duration,
            v_stop,
            wf.start_voltage_V,
            gen,
            n_steps=ramp_steps,
            role="reverse_ramp",
        )
    )

    init_gen = 0.0 if (wf.dark_seed_s > 0.0 or wf.dark_prep_s > 0.0) else gen
    return build_piecewise_waveform(
        segments,
        initial_voltage_V=wf.start_voltage_V,
        initial_generation_scale=init_gen,
    )


def _json_strict(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: _json_strict(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, np.ndarray):
        return _json_strict(value.tolist())
    if isinstance(value, np.generic):
        return _json_strict(value.item())
    if isinstance(value, dict):
        return {str(k): _json_strict(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_strict(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("accepted steady state cannot contain nonfinite floats")
    return value


def package_ionic_steady_initial(
    compiled: CompiledIonicDevice1D,
    point: IonicSteadyResult,
    *,
    voltage_V: float,
    generation_scale: float,
    controls: IonicSteadyOptions,
    source_label: str = "in_memory://waveform_jv_initial_steady.json",
) -> IonicTransientInitial:
    """Package an accepted ``IonicSteadyResult`` into an admitted ``IonicTransientInitial``."""
    if not point.accepted or point.state is None:
        reason = point.error or point.termination
        raise RuntimeError(
            f"cannot initialize ionic transient from unaccepted steady result ({reason})"
        )
    summary = {
        "accepted": True,
        "compiled_identity": compiled.identity,
        "voltage_V": float(voltage_V),
        "generation_scale": float(generation_scale),
    }
    evidence_json = (
        json.dumps(
            _json_strict({"summary": summary, "result": point}),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    )
    digest = hashlib.sha256(evidence_json.encode("utf-8")).hexdigest()
    return IonicTransientInitial(
        state=point.state,
        generation_scale=float(generation_scale),
        steady_controls=controls,
        source_path=source_label,
        source_sha256=digest,
        evidence_json=evidence_json,
    )


def _default_transient_options(max_steps: int) -> TransientOptions:
    return TransientOptions(
        poisson_tolerance=1e-11,
        continuity_tolerance_A_m2=1e-8,
        contact_log_tolerance=1e-11,
        potential_step_V=0.1,
        quasi_fermi_step=2.0,
        newton_budget=SolveBudget(50, 1000, 50, 24),
        max_steps=max(max_steps, 1),
        wall_s=120.0,
        predictor="previous",
        convergence_policy=POISSON_CONTACT_BALANCE_V1,
    )


def _unpack_accepted_attempts(
    attempts: Sequence[TransientAttempt | IonicTransientAttempt],
    polarity: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    accepted = [a for a in attempts if a.accepted and a.accounting is not None]
    n_acc = len(accepted)
    t_end: np.ndarray = np.empty(n_acc, dtype=float)
    dt_arr: np.ndarray = np.empty(n_acc, dtype=float)
    v_arr: np.ndarray = np.empty(n_acc, dtype=float)
    gen_arr: np.ndarray = np.empty(n_acc, dtype=float)
    pv_tot: np.ndarray = np.empty(n_acc, dtype=float)
    pv_cond: np.ndarray = np.empty(n_acc, dtype=float)
    pv_disp: np.ndarray = np.empty(n_acc, dtype=float)
    conj_tot: np.ndarray = np.empty(n_acc, dtype=float)
    conj_cond: np.ndarray = np.empty(n_acc, dtype=float)
    conj_disp: np.ndarray = np.empty(n_acc, dtype=float)

    conj_idx = 0 if polarity > 0.0 else -1
    for idx, att in enumerate(accepted):
        assert att.accounting is not None
        acc = att.accounting
        term_tot = np.asarray(
            acc["terminal_total_current_density_A_m2"], dtype=float
        )
        if "terminal_conduction_current_density_A_m2" in acc:
            term_cond = np.asarray(
                acc["terminal_conduction_current_density_A_m2"], dtype=float
            )
        else:
            term_cond = np.asarray(
                acc["conduction_current_density_A_m2"], dtype=float
            )[[0, -1]]
        if "terminal_displacement_current_density_A_m2" in acc:
            term_disp = np.asarray(
                acc["terminal_displacement_current_density_A_m2"], dtype=float
            )
        else:
            term_disp = np.asarray(
                acc["displacement_current_density_A_m2"], dtype=float
            )[[0, -1]]
        t_end[idx] = float(att.end_s)
        dt_arr[idx] = float(att.end_s - att.start_s)
        v_arr[idx] = float(att.voltage_V)
        gen_arr[idx] = float(att.generation_scale)

        pv_tot[idx] = float(-polarity * term_tot[0])
        pv_cond[idx] = float(-polarity * term_cond[0])
        pv_disp[idx] = float(-polarity * term_disp[0])

        conj_tot[idx] = float(polarity * term_tot[conj_idx])
        conj_cond[idx] = float(polarity * term_cond[conj_idx])
        conj_disp[idx] = float(polarity * term_disp[conj_idx])

    return (
        t_end,
        dt_arr,
        v_arr,
        gen_arr,
        pv_tot,
        pv_cond,
        pv_disp,
        conj_tot,
        conj_cond,
        conj_disp,
    )


def _compute_ionic_inventory_drift(
    compiled: CompiledIonicDevice1D,
    result: IonicTransientResult,
) -> tuple[float, ...]:
    if not result.states:
        return ()
    init_state = result.states[0]
    measures = compiled.ion_geometry.cell_measures_m
    drifts: list[float] = []
    for species in compiled.ion_species_names:
        attr = f"{species}_ion_density_m3"
        n0 = float(np.dot(measures, getattr(init_state, attr)))
        scale = max(abs(n0), 1e-30)
        max_rel = max(
            abs(float(np.dot(measures, getattr(st, attr))) - n0) / scale
            for st in result.states
        )
        drifts.append(float(max_rel))
    return tuple(drifts)


def _extract_ramp_branch(
    seg: WaveformSegment,
    slc: slice,
    v_arr: np.ndarray,
    gen_arr: np.ndarray,
    pv_tot: np.ndarray,
    *,
    init_voltage_V: float,
    init_generation_scale: float,
    init_pv_A_m2: float,
) -> tuple[np.ndarray, np.ndarray]:
    v_seg: np.ndarray = v_arr[slc]
    j_seg: np.ndarray = pv_tot[slc]
    if slc.start > 0:
        prev_idx = slc.start - 1
        if abs(float(gen_arr[prev_idx]) - seg.generation_scale) <= 1e-12 and (
            v_seg.size == 0 or abs(float(v_arr[prev_idx]) - float(v_seg[0])) > 1e-14
        ):
            v_seg = np.r_[float(v_arr[prev_idx]), v_seg]
            j_seg = np.r_[float(pv_tot[prev_idx]), j_seg]
    elif abs(init_generation_scale - seg.generation_scale) <= 1e-12 and (
        v_seg.size == 0 or abs(init_voltage_V - float(v_seg[0])) > 1e-14
    ):
        v_seg = np.r_[init_voltage_V, v_seg]
        j_seg = np.r_[init_pv_A_m2, j_seg]
    return v_seg, j_seg


def run_waveform_jv(
    compiled: CompiledDevice1D | CompiledIonicDevice1D,
    waveform: PiecewiseWaveform | JVWaveform | Sequence[WaveformSegment],
    *,
    V_max: float = 1.2,
    v_rate: float = 0.04,
    n_points: int = 61,
    generation_scale: float = 1.0,
    initial: SteadyState1D | IonicTransientInitial | None = None,
    options: WaveformJVOptions | None = None,
) -> WaveformJVResult:
    """Execute a piecewise-linear or triangular V(t) waveform on Backward Euler."""
    opts = WaveformJVOptions() if options is None else options
    if isinstance(waveform, PiecewiseWaveform):
        pw = waveform
    elif isinstance(waveform, JVWaveform):
        pw = build_triangular_jv_waveform(
            waveform,
            V_max=V_max,
            v_rate=v_rate,
            n_points=n_points,
            generation_scale=generation_scale,
        )
    else:
        pw = build_piecewise_waveform(waveform)

    n_req = int(pw.protocol.times_s.size - 1)
    base_trans_opts = (
        _default_transient_options(n_req)
        if opts.transient is None
        else (
            replace(opts.transient, max_steps=max(opts.transient.max_steps, n_req))
            if opts.transient.max_steps < n_req
            else opts.transient
        )
    )

    if isinstance(compiled, CompiledDevice1D):
        polarity = float(compiled.voltage_polarity)
        if initial is None:
            dc_init = solve_steady_state(
                compiled,
                voltage_V=pw.initial_voltage_V,
                generation_scale=pw.initial_generation_scale,
                options=opts.steady,
            )
            if not dc_init.converged:
                raise RuntimeError("waveform initial DC solve failed to converge")
        elif isinstance(initial, SteadyState1D):
            dc_init = initial
        else:
            raise TypeError(
                "electronic waveform J-V requires initial=None or SteadyState1D"
            )

        init_pv_A_m2 = float(
            -polarity * np.asarray(dc_init.current_density_A_m2, dtype=float)[0]
        )
        trans_res: TransientResult1D | IonicTransientResult = solve_transient(
            compiled,
            dc_init,
            pw.protocol,
            options=base_trans_opts,
            initial_options=opts.steady,
            method="backward_euler",
        )
        inv_drift: tuple[float, ...] = ()

    elif isinstance(compiled, CompiledIonicDevice1D):
        polarity = float(compiled.electronic_component.voltage_polarity)
        ionic_steady_opts = (
            default_ionic_steady_options(
                compiled.ion_node_count, electronic=opts.steady
            )
            if opts.ionic_steady is None
            else opts.ionic_steady
        )
        if initial is None:
            ion_dc = solve_ionic_dc_operating_point(
                compiled,
                voltage_V=pw.initial_voltage_V,
                generation_scale=pw.initial_generation_scale,
                options=ionic_steady_opts,
            )
            ion_init = package_ionic_steady_initial(
                compiled,
                ion_dc,
                voltage_V=pw.initial_voltage_V,
                generation_scale=pw.initial_generation_scale,
                controls=ionic_steady_opts,
            )
        elif isinstance(initial, IonicTransientInitial):
            ion_init = initial
        else:
            raise TypeError(
                "ionic waveform J-V requires initial=None or IonicTransientInitial"
            )

        elec_prob = Ionic1DProblem(
            compiled, pw.initial_voltage_V, pw.initial_generation_scale
        ).electronic_problem
        init_contacts = np.asarray(
            elec_prob.contact_current_density(
                ion_init.state.u[: 3 * compiled.node_count]
            ),
            dtype=float,
        )
        init_pv_A_m2 = float(-polarity * init_contacts[0])

        ion_trans_opts = (
            IonicTransientOptions(
                electronic=base_trans_opts,
                ion_balance_tolerance_m2_s=1e6,
                ion_logit_step=1.0,
            )
            if opts.ionic_transient is None
            else (
                replace(
                    opts.ionic_transient,
                    electronic=replace(
                        opts.ionic_transient.electronic,
                        max_steps=max(
                            opts.ionic_transient.electronic.max_steps, n_req
                        ),
                    ),
                )
                if opts.ionic_transient.electronic.max_steps < n_req
                else opts.ionic_transient
            )
        )
        trans_res = solve_ionic_transient(
            compiled,
            ion_init,
            pw.protocol,
            options=ion_trans_opts,
            method="backward_euler",
        )
        assert isinstance(trans_res, IonicTransientResult)
        inv_drift = _compute_ionic_inventory_drift(compiled, trans_res)
    else:
        raise TypeError(
            "compiled must be a CompiledDevice1D or CompiledIonicDevice1D"
        )

    assert isinstance(trans_res, (TransientResult1D, IonicTransientResult))
    completed = bool(trans_res.completed)
    fail_reason: str | None = None
    if not completed:
        failed = trans_res.failed_attempts
        if failed:
            first = failed[0]
            fail_reason = (
                f"step_failed_at_{first.end_s:.6g}s:{first.error or first.termination}"
            )
        else:
            fail_reason = "incomplete_transient_history"
        if opts.require_completed:
            raise RuntimeError(f"Waveform J-V failed to complete: {fail_reason}")

    (
        t_end,
        dt_arr,
        v_arr,
        gen_arr,
        pv_tot,
        pv_cond,
        pv_disp,
        conj_tot,
        conj_cond,
        conj_disp,
    ) = _unpack_accepted_attempts(trans_res.attempts, polarity)
    n_acc = int(t_end.size)
    dv_dt_acc = pw.step_dV_dt_V_s[:n_acc]

    v_fwd: np.ndarray | None = None
    j_fwd: np.ndarray | None = None
    v_rev: np.ndarray | None = None
    j_rev: np.ndarray | None = None
    metrics_fwd: SteadyJVMetrics | None = None
    metrics_rev: SteadyJVMetrics | None = None

    for seg, slc in zip(pw.segments, pw.segment_step_slices, strict=True):
        if slc.stop > n_acc:
            break
        if seg.role == "forward_ramp" and v_fwd is None:
            v_fwd, j_fwd = _extract_ramp_branch(
                seg,
                slc,
                v_arr,
                gen_arr,
                pv_tot,
                init_voltage_V=pw.initial_voltage_V,
                init_generation_scale=pw.initial_generation_scale,
                init_pv_A_m2=init_pv_A_m2,
            )
            if v_fwd.size >= 2 and np.unique(v_fwd).size == v_fwd.size:
                metrics_fwd = steady_jv_metrics(
                    v_fwd, j_fwd, incident_power_W_m2=opts.incident_power_W_m2
                )
        elif seg.role == "reverse_ramp" and v_rev is None:
            v_rev, j_rev = _extract_ramp_branch(
                seg,
                slc,
                v_arr,
                gen_arr,
                pv_tot,
                init_voltage_V=pw.initial_voltage_V,
                init_generation_scale=pw.initial_generation_scale,
                init_pv_A_m2=init_pv_A_m2,
            )
            if v_rev.size >= 2 and np.unique(v_rev).size == v_rev.size:
                metrics_rev = steady_jv_metrics(
                    v_rev, j_rev, incident_power_W_m2=opts.incident_power_W_m2
                )

    hi_rev: float | None = None
    hi_fwd: float | None = None
    if (
        metrics_fwd is not None
        and metrics_rev is not None
        and metrics_fwd.maximum_power_W_m2 is not None
        and metrics_rev.maximum_power_W_m2 is not None
    ):
        p_fwd = metrics_fwd.maximum_power_W_m2
        p_rev = metrics_rev.maximum_power_W_m2
        if p_rev > 0.0:
            hi_rev = float((p_rev - p_fwd) / p_rev)
        if p_fwd > 0.0:
            hi_fwd = float(p_rev / p_fwd - 1.0)

    return WaveformJVResult(
        waveform=pw,
        completed=completed,
        accepted_steps=n_acc,
        requested_steps=n_req,
        step_times_s=t_end,
        step_dt_s=dt_arr,
        step_voltage_V=v_arr,
        step_dV_dt_V_s=dv_dt_acc,
        step_generation_scale=gen_arr,
        pv_output_current_A_m2=pv_tot,
        pv_conduction_current_A_m2=pv_cond,
        pv_displacement_current_A_m2=pv_disp,
        voltage_conjugate_current_A_m2=conj_tot,
        voltage_conjugate_conduction_A_m2=conj_cond,
        voltage_conjugate_displacement_A_m2=conj_disp,
        V_fwd=v_fwd,
        J_fwd=j_fwd,
        V_rev=v_rev,
        J_rev=j_rev,
        metrics_fwd=metrics_fwd,
        metrics_rev=metrics_rev,
        hysteresis_index_rev_norm=hi_rev,
        hysteresis_index_fwd_norm=hi_fwd,
        inventory_relative_drift=inv_drift,
        voltage_polarity=polarity,
        compiled_identity=compiled.identity,
        transient_result=trans_res,
        failure_reason=fail_reason,
    )
