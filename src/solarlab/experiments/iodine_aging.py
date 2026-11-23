"""Multi-stage iodine stress/recovery aging protocols, J-V probes, and persistence.

Orchestrates coupled 1D ``(phi, n, p, I^-, I^0, I_2)`` reaction-diffusion-aging
trajectories across arbitrary sequences of illumination/bias stress and dark
recovery stages, non-destructive frozen-chemistry vs dynamic-scan ``J-V``
checkpoint probes, checksum-bound bundle I/O, and SQLite :class:`RunStore`
recording. Every result explicitly carries ``scientific_qualification = False``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from solarlab.contracts import ContractError, Vector, frozen_array
from solarlab.equations.assembly.iodine_1d import AssembledIodine1DProblem, iodine_system
from solarlab.equations.iodine_1d import CompiledIodineDevice1D, IodineState1D
from solarlab.experiments.jv.steady import SteadyJVMetrics, steady_jv_metrics
from solarlab.io.artifacts import array_artifact, bytes_artifact
from solarlab.io.run_store import RunStore
from solarlab.solvers.nonlinear.iodine_1d import (
    IodineStepAccounting,
    solve_frozen_chemistry_electronic_1d,
    solve_iodine_bdf_transient,
    solve_iodine_steady_state,
)
from solarlab.solvers.time.bdf import BDFOptions

__all__ = [
    "DynamicJVScanProbeResult",
    "FrozenChemistryJVProbeResult",
    "IODINE_AGING_SCHEMA",
    "IodineAgingError",
    "IodineAgingProtocol",
    "IodineAgingResult",
    "IodineCheckpointSnapshot",
    "IodineProbeMode",
    "IodineProbeSpec",
    "StressRecoveryStage",
    "load_iodine_aging_bundle",
    "measure_dynamic_jv_scan",
    "measure_frozen_chemistry_jv",
    "record_iodine_aging_in_run_store",
    "run_iodine_aging_protocol",
    "save_iodine_aging_bundle",
]

IODINE_AGING_SCHEMA = "solarlab.iodine_aging.v1"
IodineProbeMode = Literal["none", "frozen_chemistry_jv", "dynamic_jv_scan", "both"]


class IodineAgingError(RuntimeError):
    """Raised when an aging trajectory stage or probe fails physical/solver gates."""


def _default_aging_bdf_options() -> BDFOptions:
    return BDFOptions(
        initial_step_s=1.0e-6,
        min_step_s=1.0e-14,
        max_step_s=200.0,
        relative_tolerance=1.0e-3,
        absolute_tolerance=1.0e11,
        max_steps=800,
    )


def _default_probe_voltages() -> tuple[float, ...]:
    return tuple(float(v) for v in np.linspace(0.0, 1.25, 26))


@dataclass(frozen=True, slots=True)
class StressRecoveryStage:
    """One constant-bias, constant-illumination stress or recovery stage."""

    name: str
    duration_s: float
    voltage_V: float
    generation_scale: float
    max_step_s: float = 200.0

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ContractError("stage_name_must_be_nonempty")
        for attr in ("duration_s", "max_step_s"):
            val = float(getattr(self, attr))
            if not math.isfinite(val) or val <= 0.0:
                raise ContractError(f"{attr}_must_be_finite_positive")
            object.__setattr__(self, attr, val)
        v = float(self.voltage_V)
        g = float(self.generation_scale)
        if not math.isfinite(v):
            raise ContractError("voltage_V_must_be_finite")
        if not math.isfinite(g) or g < 0.0:
            raise ContractError("generation_scale_must_be_finite_nonnegative")
        object.__setattr__(self, "voltage_V", v)
        object.__setattr__(self, "generation_scale", g)


@dataclass(frozen=True, slots=True)
class IodineProbeSpec:
    """Non-destructive J-V characterization specification at aging checkpoints."""

    mode: IodineProbeMode = "frozen_chemistry_jv"
    voltages_V: tuple[float, ...] = field(default_factory=_default_probe_voltages)
    scan_rate_V_s: float = 1.0
    generation_scale: float = 1.0
    incident_power_W_m2: float = 1000.0

    def __post_init__(self) -> None:
        if self.mode not in ("none", "frozen_chemistry_jv", "dynamic_jv_scan", "both"):
            raise ContractError("invalid_iodine_probe_mode")
        v_tuple = tuple(float(v) for v in self.voltages_V)
        if self.mode != "none":
            if len(v_tuple) < 2 or not all(math.isfinite(v) for v in v_tuple):
                raise ContractError("probe_voltages_V_requires_at_least_two_finite_points")
            if len(set(v_tuple)) != len(v_tuple):
                raise ContractError("probe_voltages_V_must_be_distinct")
        rate = float(self.scan_rate_V_s)
        g = float(self.generation_scale)
        p_in = float(self.incident_power_W_m2)
        if not math.isfinite(rate) or rate <= 0.0:
            raise ContractError("scan_rate_V_s_must_be_finite_positive")
        if not math.isfinite(g) or g < 0.0:
            raise ContractError("probe_generation_scale_must_be_finite_nonnegative")
        if not math.isfinite(p_in) or p_in <= 0.0:
            raise ContractError("incident_power_W_m2_must_be_finite_positive")
        object.__setattr__(self, "voltages_V", tuple(sorted(v_tuple)))
        object.__setattr__(self, "scan_rate_V_s", rate)
        object.__setattr__(self, "generation_scale", g)
        object.__setattr__(self, "incident_power_W_m2", p_in)


@dataclass(frozen=True, slots=True)
class IodineAgingProtocol:
    """Multi-stage iodine stress/recovery schedule and checkpoint probe spec."""

    stages: tuple[StressRecoveryStage, ...]
    checkpoint_times_s: tuple[float, ...] = ()
    probe: IodineProbeSpec = field(default_factory=IodineProbeSpec)
    bdf_options: BDFOptions = field(default_factory=_default_aging_bdf_options)

    def __post_init__(self) -> None:
        st = tuple(self.stages)
        if not st:
            raise ContractError("aging_protocol_requires_at_least_one_stage")
        total_dur = sum(s.duration_s for s in st)
        ckpts = tuple(sorted({float(t) for t in self.checkpoint_times_s}))
        for t in ckpts:
            if not math.isfinite(t) or t < 0.0 or t > total_dur * (1.0 + 1.0e-12):
                raise ContractError("checkpoint_time_s_outside_protocol_horizon")
        object.__setattr__(self, "stages", st)
        object.__setattr__(self, "checkpoint_times_s", ckpts)

    @property
    def total_duration_s(self) -> float:
        return float(sum(s.duration_s for s in self.stages))

    @property
    def sha256(self) -> str:
        doc = {
            "schema": IODINE_AGING_SCHEMA,
            "stages": [
                {
                    "name": s.name,
                    "duration_s": s.duration_s,
                    "voltage_V": s.voltage_V,
                    "generation_scale": s.generation_scale,
                    "max_step_s": s.max_step_s,
                }
                for s in self.stages
            ],
            "checkpoint_times_s": list(self.checkpoint_times_s),
            "probe": {
                "mode": self.probe.mode,
                "voltages_V": list(self.probe.voltages_V),
                "scan_rate_V_s": self.probe.scan_rate_V_s,
                "generation_scale": self.probe.generation_scale,
                "incident_power_W_m2": self.probe.incident_power_W_m2,
            },
        }
        encoded = json.dumps(doc, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class FrozenChemistryJVProbeResult:
    """Stationary electronic J-V curve with frozen spatial iodine profiles."""

    voltages_V: Vector
    output_current_A_m2: Vector
    metrics: SteadyJVMetrics


@dataclass(frozen=True, slots=True)
class DynamicJVScanProbeResult:
    """Transient reverse (V_max -> V_min) and forward (V_min -> V_max) J-V scan."""

    voltages_V: Vector
    scan_rate_V_s: float
    reverse_output_current_A_m2: Vector
    forward_output_current_A_m2: Vector
    reverse_metrics: SteadyJVMetrics
    forward_metrics: SteadyJVMetrics
    hysteresis_index: float | None


@dataclass(frozen=True, slots=True)
class IodineCheckpointSnapshot:
    """Recorded physical state, conservation accounting, and probes at ``time_s``."""

    time_s: float
    stage_name: str
    state: IodineState1D
    in_device_iodine_m2: float
    cumulative_escaped_iodine_atoms_m2: float
    cumulative_immobilized_iodine_atoms_m2: float
    conservation_residual_m2: float
    frozen_jv: FrozenChemistryJVProbeResult | None = None
    dynamic_jv: DynamicJVScanProbeResult | None = None


@dataclass(frozen=True, slots=True)
class IodineAgingResult:
    """Complete trajectory, checkpoint probes, and conservation ledger."""

    protocol_sha256: str
    device_sha256: str
    reference_iodine_inventory_m2: float
    checkpoints: tuple[IodineCheckpointSnapshot, ...]
    step_ledger: tuple[IodineStepAccounting, ...]
    max_relative_conservation_error: float
    completed: bool
    wall_s: float
    scientific_qualification: bool = False


def measure_frozen_chemistry_jv(
    device: CompiledIodineDevice1D,
    state: IodineState1D,
    voltages_V: ArrayLike,
    *,
    generation_scale: float = 1.0,
    incident_power_W_m2: float = 1000.0,
) -> FrozenChemistryJVProbeResult:
    """Sweep stationary electronic J-V at exact 0-ULP frozen chemical species."""
    if state.device_sha256 != device.identity:
        raise ContractError("state_device_mismatch")
    v_grid = np.sort(np.asarray(voltages_V, dtype=np.float64))
    if v_grid.ndim != 1 or v_grid.size < 2 or np.unique(v_grid).size != v_grid.size:
        raise ContractError("voltages_V_must_be_1d_distinct_sorted_array")

    binding = iodine_system(device, constrained_steady=False)
    n_elec = 3 * device.node_count
    chem_ref = state.u[n_elec:].copy()
    polarity = float(device.electronic_component.voltage_polarity)

    u_cur = state.u
    prev_v = state.electronic_state.voltage_V
    j_out = np.empty(v_grid.size, dtype=np.float64)
    for idx, v_val in enumerate(v_grid):
        u_cur, j_mean = solve_frozen_chemistry_electronic_1d(
            device,
            u_cur,
            voltage_V=float(v_val),
            generation_scale=generation_scale,
            prev_voltage_V=prev_v,
            binding=binding,
        )
        if not np.array_equal(u_cur[n_elec:], chem_ref):
            raise IodineAgingError("frozen-chemistry J-V mutated chemical coordinates")
        j_out[idx] = -polarity * j_mean
        prev_v = float(v_val)

    metrics = steady_jv_metrics(
        v_grid, j_out, incident_power_W_m2=incident_power_W_m2
    )
    return FrozenChemistryJVProbeResult(
        voltages_V=frozen_array(v_grid),
        output_current_A_m2=frozen_array(j_out),
        metrics=metrics,
    )


def _instantaneous_output_current_A_m2(
    problem: AssembledIodine1DProblem,
    u: Vector,
    u_prev: Vector,
    dt_s: float,
) -> float:
    """Evaluate total Maxwell output current ``-polarity * mean(J_cond + dD/dt)``."""
    comp = problem.device.electronic_component
    n_c = problem.device.node_count
    j_cond = problem.face_conduction_current_A_m2(u)
    cap = comp.poisson.face_capacitance_F_m2
    dd_dt = -cap * (np.diff(u[:n_c]) - np.diff(u_prev[:n_c])) / max(dt_s, 1.0e-30)
    j_tot = j_cond + dd_dt
    dx = np.diff(comp.nodes_m)
    j_mean = float(math.fsum((dx * j_tot).tolist()) / math.fsum(dx.tolist()))
    return float(-comp.voltage_polarity * j_mean)


def measure_dynamic_jv_scan(
    device: CompiledIodineDevice1D,
    state: IodineState1D,
    voltages_V: ArrayLike,
    *,
    scan_rate_V_s: float = 1.0,
    generation_scale: float = 1.0,
    incident_power_W_m2: float = 1000.0,
    bdf_options: BDFOptions | None = None,
) -> DynamicJVScanProbeResult:
    """Execute non-destructive coupled reverse + forward transient J-V scan."""
    if state.device_sha256 != device.identity:
        raise ContractError("state_device_mismatch")
    v_sorted = np.sort(np.asarray(voltages_V, dtype=np.float64))
    if v_sorted.ndim != 1 or v_sorted.size < 2:
        raise ContractError("voltages_V_requires_at_least_two_points")
    rate = float(scan_rate_V_s)
    v_min = float(v_sorted[0])
    v_max = float(v_sorted[-1])
    span_v = v_max - v_min
    leg_dur_s = span_v / rate

    binding = iodine_system(device, constrained_steady=False)
    # Pre-position electronics cleanly at v_max without altering chemical inventory
    u_start_rev, _ = solve_frozen_chemistry_electronic_1d(
        device,
        state.u,
        voltage_V=v_max,
        generation_scale=generation_scale,
        prev_voltage_V=state.electronic_state.voltage_V,
        binding=binding,
    )

    rev_v_desc = v_sorted[::-1]
    rev_times = (v_max - rev_v_desc) / rate
    fwd_times = (v_sorted - v_min) / rate
    base_opts = _default_aging_bdf_options() if bdf_options is None else bdf_options
    scan_opts = replace(
        base_opts,
        initial_step_s=min(base_opts.initial_step_s, max(leg_dur_s * 1.0e-3, 1.0e-9)),
        max_step_s=min(base_opts.max_step_s, max(leg_dur_s / 25.0, 1.0e-6)),
    )

    rev_bdf, _ = solve_iodine_bdf_transient(
        device,
        u_start_rev,
        (0.0, leg_dur_s),
        voltage_fn=lambda t: v_max - rate * t,
        generation_fn=lambda _t: generation_scale,
        options=scan_opts,
        output_times_s=rev_times,
    )
    if not rev_bdf.completed:
        raise IodineAgingError(
            f"Dynamic reverse J-V scan failed: {rev_bdf.termination}"
        )

    u_turn = rev_bdf.accepted_coordinates[-1]
    fwd_bdf, _ = solve_iodine_bdf_transient(
        device,
        u_turn,
        (0.0, leg_dur_s),
        voltage_fn=lambda t: v_min + rate * t,
        generation_fn=lambda _t: generation_scale,
        options=scan_opts,
        output_times_s=fwd_times,
    )
    if not fwd_bdf.completed:
        raise IodineAgingError(
            f"Dynamic forward J-V scan failed: {fwd_bdf.termination}"
        )

    # Read output current at each sample point (stored in ascending v_sorted order)
    rev_j_desc = np.empty(v_sorted.size, dtype=np.float64)
    for k, samp in enumerate(rev_bdf.samples):
        v_k = float(rev_v_desc[k])
        prob_k = AssembledIodine1DProblem(
            device=device,
            voltage_V=v_k,
            generation_scale=generation_scale,
            binding=binding,
        )
        if k == 0:
            j_mean = prob_k.steady_current_density_A_m2(samp.u)
            rev_j_desc[k] = float(-device.electronic_component.voltage_polarity * j_mean)
        else:
            dt_k = max(samp.time_s - rev_bdf.samples[k - 1].time_s, 1.0e-12)
            rev_j_desc[k] = _instantaneous_output_current_A_m2(
                prob_k, samp.u, rev_bdf.samples[k - 1].u, dt_k
            )
    rev_j_asc = rev_j_desc[::-1]

    fwd_j_asc = np.empty(v_sorted.size, dtype=np.float64)
    for k, samp in enumerate(fwd_bdf.samples):
        v_k = float(v_sorted[k])
        prob_k = AssembledIodine1DProblem(
            device=device,
            voltage_V=v_k,
            generation_scale=generation_scale,
            binding=binding,
        )
        if k == 0:
            j_mean = prob_k.steady_current_density_A_m2(samp.u)
            fwd_j_asc[k] = float(-device.electronic_component.voltage_polarity * j_mean)
        else:
            dt_k = max(samp.time_s - fwd_bdf.samples[k - 1].time_s, 1.0e-12)
            fwd_j_asc[k] = _instantaneous_output_current_A_m2(
                prob_k, samp.u, fwd_bdf.samples[k - 1].u, dt_k
            )

    rev_metrics = steady_jv_metrics(
        v_sorted, rev_j_asc, incident_power_W_m2=incident_power_W_m2
    )
    fwd_metrics = steady_jv_metrics(
        v_sorted, fwd_j_asc, incident_power_W_m2=incident_power_W_m2
    )
    hi: float | None = None
    if (
        rev_metrics.maximum_power_W_m2 is not None
        and fwd_metrics.maximum_power_W_m2 is not None
        and rev_metrics.maximum_power_W_m2 > 0.0
    ):
        hi = float(
            (rev_metrics.maximum_power_W_m2 - fwd_metrics.maximum_power_W_m2)
            / rev_metrics.maximum_power_W_m2
        )

    return DynamicJVScanProbeResult(
        voltages_V=frozen_array(v_sorted),
        scan_rate_V_s=rate,
        reverse_output_current_A_m2=frozen_array(rev_j_asc),
        forward_output_current_A_m2=frozen_array(fwd_j_asc),
        reverse_metrics=rev_metrics,
        forward_metrics=fwd_metrics,
        hysteresis_index=hi,
    )


def _evaluate_checkpoint_probes(
    device: CompiledIodineDevice1D,
    state: IodineState1D,
    probe: IodineProbeSpec,
    bdf_options: BDFOptions,
) -> tuple[FrozenChemistryJVProbeResult | None, DynamicJVScanProbeResult | None]:
    frozen_res: FrozenChemistryJVProbeResult | None = None
    dyn_res: DynamicJVScanProbeResult | None = None
    if probe.mode in ("frozen_chemistry_jv", "both"):
        frozen_res = measure_frozen_chemistry_jv(
            device,
            state,
            probe.voltages_V,
            generation_scale=probe.generation_scale,
            incident_power_W_m2=probe.incident_power_W_m2,
        )
    if probe.mode in ("dynamic_jv_scan", "both"):
        dyn_res = measure_dynamic_jv_scan(
            device,
            state,
            probe.voltages_V,
            scan_rate_V_s=probe.scan_rate_V_s,
            generation_scale=probe.generation_scale,
            incident_power_W_m2=probe.incident_power_W_m2,
            bdf_options=bdf_options,
        )
    return frozen_res, dyn_res


def run_iodine_aging_protocol(
    device: CompiledIodineDevice1D,
    protocol: IodineAgingProtocol,
    *,
    initial_u: Vector | None = None,
) -> IodineAgingResult:
    """Execute a multi-stage stress/recovery protocol with conservation auditing."""
    t_wall_start = time.monotonic()
    if initial_u is None:
        if not device.has_open_iodine_loss:
            eq = solve_iodine_steady_state(device, voltage_V=0.0, generation_scale=0.0)
            if not eq.converged or eq.state is None:
                raise IodineAgingError(
                    f"Initial dark equilibrium failed: {eq.termination}"
                )
            u_curr = eq.state.u
            v_curr = 0.0
        else:
            u_base = device.initial_coordinates(voltage_V=0.0)
            u_curr, _ = solve_frozen_chemistry_electronic_1d(
                device, u_base, voltage_V=0.0, generation_scale=0.0, prev_voltage_V=0.0
            )
            v_curr = 0.0
    else:
        u_curr = frozen_array(initial_u)
        v_curr = 0.0

    state_0 = device.state(u_curr, voltage_V=v_curr)
    inv_0 = state_0.total_iodine_inventory_m2
    inv_scale = max(inv_0, 1.0)

    # Collect stage-end times + user checkpoint times
    stage_bounds: list[tuple[float, float, StressRecoveryStage]] = []
    t_cursor = 0.0
    for stg in protocol.stages:
        stage_bounds.append((t_cursor, t_cursor + stg.duration_s, stg))
        t_cursor += stg.duration_s

    all_ckpts = sorted(
        {0.0, *(b[1] for b in stage_bounds), *protocol.checkpoint_times_s}
    )

    checkpoints: list[IodineCheckpointSnapshot] = []
    all_steps: list[IodineStepAccounting] = []
    cum_escaped = 0.0
    cum_immob = 0.0
    max_rel_err = 0.0

    if 0.0 in all_ckpts:
        f_jv0, d_jv0 = _evaluate_checkpoint_probes(
            device, state_0, protocol.probe, protocol.bdf_options
        )
        checkpoints.append(
            IodineCheckpointSnapshot(
                time_s=0.0,
                stage_name="initial",
                state=state_0,
                in_device_iodine_m2=inv_0,
                cumulative_escaped_iodine_atoms_m2=0.0,
                cumulative_immobilized_iodine_atoms_m2=0.0,
                conservation_residual_m2=0.0,
                frozen_jv=f_jv0,
                dynamic_jv=d_jv0,
            )
        )

    for t_start, t_end, stg in stage_bounds:
        # Sub-intervals broken at any requested checkpoint inside (t_start, t_end]
        sub_targets = [
            t for t in all_ckpts if t_start + 1.0e-14 < t <= t_end + 1.0e-14
        ]
        if not sub_targets or abs(sub_targets[-1] - t_end) > 1.0e-12:
            sub_targets.append(t_end)

        sub_a = t_start
        for sub_b in sub_targets:
            dt_span = sub_b - sub_a
            stg_opts = replace(
                protocol.bdf_options,
                initial_step_s=min(protocol.bdf_options.initial_step_s, dt_span * 0.5),
                max_step_s=min(protocol.bdf_options.max_step_s, stg.max_step_s, dt_span),
            )
            v_const = float(stg.voltage_V)
            g_const = float(stg.generation_scale)

            def _stg_v(_t: float, _v: float = v_const) -> float:
                return _v

            def _stg_g(_t: float, _g: float = g_const) -> float:
                return _g

            bdf_res, sub_ledger = solve_iodine_bdf_transient(
                device,
                u_curr,
                (sub_a, sub_b),
                voltage_fn=_stg_v,
                generation_fn=_stg_g,
                options=stg_opts,
            )
            if not bdf_res.completed:
                raise IodineAgingError(
                    f"Aging stage {stg.name!r} failed on [{sub_a:.3g}, {sub_b:.3g}] s: "
                    f"{bdf_res.termination}"
                )
            for acc in sub_ledger:
                cum_escaped += acc.step_escaped_iodine_atoms_m2
                cum_immob += acc.step_immobilized_iodine_atoms_m2
                tot_res = (acc.in_device_iodine_m2 + cum_escaped + cum_immob) - inv_0
                max_rel_err = max(
                    max_rel_err,
                    abs(tot_res) / inv_scale,
                    abs(acc.step_conservation_residual_m2) / inv_scale,
                )
                all_steps.append(acc)

            u_curr = bdf_res.accepted_coordinates[-1]
            st_ckpt = device.state(u_curr, voltage_V=stg.voltage_V)
            cons_res = (st_ckpt.total_iodine_inventory_m2 + cum_escaped + cum_immob) - inv_0
            max_rel_err = max(max_rel_err, abs(cons_res) / inv_scale)

            f_jv, d_jv = _evaluate_checkpoint_probes(
                device, st_ckpt, protocol.probe, protocol.bdf_options
            )
            checkpoints.append(
                IodineCheckpointSnapshot(
                    time_s=float(sub_b),
                    stage_name=stg.name,
                    state=st_ckpt,
                    in_device_iodine_m2=st_ckpt.total_iodine_inventory_m2,
                    cumulative_escaped_iodine_atoms_m2=cum_escaped,
                    cumulative_immobilized_iodine_atoms_m2=cum_immob,
                    conservation_residual_m2=cons_res,
                    frozen_jv=f_jv,
                    dynamic_jv=d_jv,
                )
            )
            sub_a = sub_b

    return IodineAgingResult(
        protocol_sha256=protocol.sha256,
        device_sha256=device.identity,
        reference_iodine_inventory_m2=inv_0,
        checkpoints=tuple(checkpoints),
        step_ledger=tuple(all_steps),
        max_relative_conservation_error=max_rel_err,
        completed=True,
        wall_s=time.monotonic() - t_wall_start,
        scientific_qualification=False,
    )


def _metrics_dict(m: SteadyJVMetrics) -> dict[str, Any]:
    return {
        "short_circuit_current_A_m2": m.short_circuit_current_A_m2,
        "open_circuit_voltage_V": m.open_circuit_voltage_V,
        "maximum_power_W_m2": m.maximum_power_W_m2,
        "fill_factor": m.fill_factor,
        "efficiency": m.efficiency,
        "limitation": m.limitation,
    }


def save_iodine_aging_bundle(
    directory: Path,
    result: IodineAgingResult,
) -> dict[str, Any]:
    """Persist checksum-verified ``manifest.json`` and ``arrays.npz`` without overwrite."""
    target = Path(directory)
    manifest_path = target / "manifest.json"
    npz_path = target / "arrays.npz"
    if manifest_path.exists() or npz_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing aging bundle at {target}")
    target.mkdir(parents=True, exist_ok=True)

    npz_dict: dict[str, NDArray[np.float64]] = {
        "checkpoint_times_s": np.array(
            [c.time_s for c in result.checkpoints], dtype=np.float64
        ),
        "checkpoint_in_device_iodine_m2": np.array(
            [c.in_device_iodine_m2 for c in result.checkpoints], dtype=np.float64
        ),
        "checkpoint_escaped_iodine_m2": np.array(
            [c.cumulative_escaped_iodine_atoms_m2 for c in result.checkpoints],
            dtype=np.float64,
        ),
        "checkpoint_immobilized_iodine_m2": np.array(
            [c.cumulative_immobilized_iodine_atoms_m2 for c in result.checkpoints],
            dtype=np.float64,
        ),
        "checkpoint_coordinates": np.stack(
            [c.state.u for c in result.checkpoints], axis=0
        ),
        "step_end_s": np.array([s.end_s for s in result.step_ledger], dtype=np.float64),
        "step_conservation_residual_m2": np.array(
            [s.step_conservation_residual_m2 for s in result.step_ledger],
            dtype=np.float64,
        ),
    }
    for idx, ck in enumerate(result.checkpoints):
        if ck.frozen_jv is not None:
            npz_dict[f"ckpt_{idx}_frozen_v"] = ck.frozen_jv.voltages_V
            npz_dict[f"ckpt_{idx}_frozen_j"] = ck.frozen_jv.output_current_A_m2
        if ck.dynamic_jv is not None:
            npz_dict[f"ckpt_{idx}_dyn_v"] = ck.dynamic_jv.voltages_V
            npz_dict[f"ckpt_{idx}_dyn_j_rev"] = ck.dynamic_jv.reverse_output_current_A_m2
            npz_dict[f"ckpt_{idx}_dyn_j_fwd"] = ck.dynamic_jv.forward_output_current_A_m2

    with npz_path.open("wb") as fh:
        np.savez_compressed(fh, allow_pickle=False, **npz_dict)
    npz_bytes = npz_path.read_bytes()
    npz_sha256 = hashlib.sha256(npz_bytes).hexdigest()

    manifest: dict[str, Any] = {
        "schema": IODINE_AGING_SCHEMA,
        "scientific_qualification": False,
        "completed": result.completed,
        "protocol_sha256": result.protocol_sha256,
        "device_sha256": result.device_sha256,
        "reference_iodine_inventory_m2": result.reference_iodine_inventory_m2,
        "max_relative_conservation_error": result.max_relative_conservation_error,
        "step_count": len(result.step_ledger),
        "wall_s": result.wall_s,
        "arrays_npz_sha256": npz_sha256,
        "checkpoints": [
            {
                "index": i,
                "time_s": c.time_s,
                "stage_name": c.stage_name,
                "in_device_iodine_m2": c.in_device_iodine_m2,
                "cumulative_escaped_iodine_atoms_m2": c.cumulative_escaped_iodine_atoms_m2,
                "cumulative_immobilized_iodine_atoms_m2": c.cumulative_immobilized_iodine_atoms_m2,
                "conservation_residual_m2": c.conservation_residual_m2,
                "species_inventories_m2": dict(c.state.species_inventories_m2),
                "frozen_jv_metrics": (
                    None if c.frozen_jv is None else _metrics_dict(c.frozen_jv.metrics)
                ),
                "dynamic_jv": (
                    None
                    if c.dynamic_jv is None
                    else {
                        "scan_rate_V_s": c.dynamic_jv.scan_rate_V_s,
                        "hysteresis_index": c.dynamic_jv.hysteresis_index,
                        "reverse_metrics": _metrics_dict(c.dynamic_jv.reverse_metrics),
                        "forward_metrics": _metrics_dict(c.dynamic_jv.forward_metrics),
                    }
                ),
            }
            for i, c in enumerate(result.checkpoints)
        ],
    }
    manifest_text = json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False)
    manifest_path.write_text(manifest_text + "\n", encoding="utf-8")
    return manifest


def load_iodine_aging_bundle(directory: Path) -> dict[str, Any]:
    """Load and checksum-verify a saved iodine aging bundle."""
    target = Path(directory)
    manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != IODINE_AGING_SCHEMA:
        raise ContractError("unexpected_iodine_aging_bundle_schema")
    npz_path = target / "arrays.npz"
    actual_sha = hashlib.sha256(npz_path.read_bytes()).hexdigest()
    if actual_sha != manifest.get("arrays_npz_sha256"):
        raise ContractError("iodine_aging_bundle_arrays_sha256_mismatch")
    with np.load(npz_path, allow_pickle=False) as data:
        arrays = {k: frozen_array(data[k]) for k in data.files}
    return {"manifest": manifest, "arrays": arrays}


def record_iodine_aging_in_run_store(
    store: RunStore,
    run_id: str,
    device: CompiledIodineDevice1D,
    protocol: IodineAgingProtocol,
    result: IodineAgingResult,
    *,
    owner_id: str = "iodine_aging_worker",
) -> dict[str, Any]:
    """Commit an executed :class:`IodineAgingResult` into SolarLab's SQLite RunStore."""
    queued = store.enqueue(
        run_id,
        opaque_identity={
            "H_input": device.original_prepared_sha256,
            "H_physics": device.identity,
            "H_execution": protocol.sha256,
        },
        input_metadata={
            "schema": IODINE_AGING_SCHEMA,
            "scientific_qualification": False,
            "total_duration_s": protocol.total_duration_s,
            "stage_count": len(protocol.stages),
        },
    )
    token = store.claim(
        run_id,
        expected_attempt_id=queued["current_attempt_id"],
        expected_generation=queued["generation"],
        owner_id=owner_id,
    )
    coords_stack = np.stack([c.state.u for c in result.checkpoints], axis=0)
    coords_art = store.publish(
        token,
        "checkpoint_coordinates",
        array_artifact(
            coords_stack,
            unit="1",
            metadata={"description": "checkpoint_state_coordinates"},
        ),
    )
    summary_doc = {
        "schema": IODINE_AGING_SCHEMA,
        "scientific_qualification": False,
        "protocol_sha256": result.protocol_sha256,
        "device_sha256": result.device_sha256,
        "reference_iodine_inventory_m2": result.reference_iodine_inventory_m2,
        "max_relative_conservation_error": result.max_relative_conservation_error,
        "checkpoint_times_s": [c.time_s for c in result.checkpoints],
        "in_device_iodine_m2": [c.in_device_iodine_m2 for c in result.checkpoints],
        "escaped_iodine_m2": [
            c.cumulative_escaped_iodine_atoms_m2 for c in result.checkpoints
        ],
    }
    summary_bytes = json.dumps(
        summary_doc, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    summary_art = store.publish(
        token,
        "aging_summary",
        bytes_artifact(
            summary_bytes,
            metadata={"content_type": "application/json", "schema": IODINE_AGING_SCHEMA},
        ),
    )
    store.complete(
        token,
        state="succeeded" if result.completed else "failed",
        result={
            "scientific_qualification": False,
            "completed": result.completed,
            "max_relative_conservation_error": result.max_relative_conservation_error,
            "checkpoint_count": len(result.checkpoints),
        },
        artifacts=(coords_art["artifact_id"], summary_art["artifact_id"]),
    )
    return store.get_run(run_id)
