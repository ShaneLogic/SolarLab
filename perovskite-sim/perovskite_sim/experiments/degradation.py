from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Callable, Optional
import numpy as np

ProgressCallback = Callable[[str, int, int, str], None]
"""Callable protocol: fn(stage, current, total, message) -> None."""
from perovskite_sim.models.device import DeviceStack, electrical_layers
from perovskite_sim.discretization.grid import multilayer_grid, Layer
from perovskite_sim.solver.illuminated_ss import solve_illuminated_ss
from perovskite_sim.solver.mol import StateVec, run_transient, split_step, build_material_arrays
from perovskite_sim.experiments.jv_sweep import (
    _compute_current,
    compute_metrics,
)


def _freeze_ions(stack: DeviceStack) -> DeviceStack:
    """Return a copy of the stack with D_ion = 0 in every layer.

    Used for snapshot J-V measurement: carriers relax to quasi-equilibrium
    at each probe voltage while the ion distribution stays pinned at the
    degradation-snapshot value. This is the numerical analogue of a fast
    laboratory sweep (sweep time ≪ τ_ion).
    """
    new_layers = []
    for layer in stack.layers:
        new_layers.append(
            replace(layer, params=replace(layer.params, D_ion=0.0))
        )
    return replace(stack, layers=tuple(new_layers))


_DAMAGE_CAP = 0.95


@dataclass(frozen=True)
class DegradationResult:
    t: np.ndarray
    PCE: np.ndarray
    V_oc: np.ndarray
    J_sc: np.ndarray
    ion_profiles: Optional[np.ndarray]   # shape (len(t), N)


def _measure_snapshot_metrics(
    x: np.ndarray,
    y_ref: np.ndarray,
    stack: DeviceStack,
    voltages: np.ndarray,
    settle_time: float,
    rtol: float,
    atol: float,
):
    """Snapshot J-V with the snapshot ion distribution pinned.

    The coupled MOL system is solved at each probe voltage with D_ion=0 in
    every layer, so the ion profile inherited from `y_ref` stays fixed while
    carriers relax to quasi-steady-state under the new V_app. The terminal
    current is read from the same coupled solve at the settled state — no
    displacement term, because the state is stationary.

    This is the numerical analogue of a laboratory sweep at a rate fast
    enough (≪ τ_ion ~ seconds for MAPbI3) that the ion lattice does not
    rearrange within the measurement, but slow enough that carriers follow
    quasi-statically. Because the ions are literally frozen here we avoid
    the startup displacement transient that would otherwise contaminate the
    first few samples of a state-carrying sweep that jumps from y_ref to
    voltages[0].
    """
    frozen_stack = _freeze_ions(stack)
    mat_frozen = build_material_arrays(x, frozen_stack)
    mat_stack = build_material_arrays(x, stack)
    voltages = np.asarray(voltages, dtype=float)
    J_arr = np.zeros_like(voltages)
    # Tighter tolerances for the settled-state measurement: integration
    # tolerance controls step size, not distance from true steady state,
    # so we tighten both to keep residual drift below the current scale.
    snap_rtol = min(rtol, 1e-5)
    snap_atol = min(atol, 1e-8)
    # Cap Radau's internal step size. Near V_bi the Jacobian is nearly
    # singular and Radau's own error estimator can underreport the local
    # truncation error, letting it accept giant steps that land on the
    # wrong branch. This guarantees the transient is resolved with
    # O(20) sub-steps regardless of settle_time.
    snap_max_step = settle_time / 20.0
    for k, V_k in enumerate(voltages):
        sol = run_transient(
            x, y_ref, (0.0, settle_time), np.array([settle_time]),
            frozen_stack, illuminated=True, V_app=float(V_k),
            rtol=snap_rtol, atol=snap_atol, max_step=snap_max_step,
            mat=mat_frozen,
        )
        if not sol.success:
            raise RuntimeError(
                f"snapshot J-V solver failed at V={V_k:.4f} V"
            )
        y_v = sol.y[:, -1]
        J_arr[k] = _compute_current(x, y_v, stack, float(V_k), mat=mat_stack)
    return compute_metrics(voltages, J_arr)


def _absorber_region(
    x: np.ndarray,
    stack: DeviceStack,
):
    """Return the absorber layer and a mask covering its interior nodes."""
    offset = 0.0
    for layer in electrical_layers(stack):
        hi = offset + layer.thickness
        if layer.role == "absorber":
            strict_mask = (x > offset + 1e-15) & (x < hi - 1e-15)
            if np.count_nonzero(strict_mask) >= 2:
                return layer, strict_mask
            inclusive_mask = (x >= offset - 1e-15) & (x <= hi + 1e-15)
            return layer, inclusive_mask
        offset = hi
    raise ValueError("stack must include an absorber layer")


def _advance_damage(
    damage: float,
    P_prev: np.ndarray,
    P_now: np.ndarray,
    P0: float,
    dt: float,
    motion_gain: float,
    stress_rate: float,
) -> float:
    """Update irreversible damage from ion motion and sustained segregation.

    Both terms are integrated as rate × dt so the accumulated damage is
    independent of the integration step size in the smooth-evolution limit:

      motion_rate      = ‖ΔP/scale‖_RMS / dt      [1/s]  (ion velocity proxy)
      segregation_rate = ‖(P − P0)/scale‖_RMS     [–]    (sustained stress)
      increment        = (motion_gain·motion_rate + stress_rate·segregation_rate)·dt
    """
    if dt <= 0.0:
        return damage
    scale = max(P0, 1e-30)
    motion_rate = np.sqrt(np.mean(((P_now - P_prev) / scale) ** 2)) / dt
    segregation = np.sqrt(np.mean(((P_now - P0) / scale) ** 2))
    increment = (motion_gain * motion_rate + stress_rate * segregation) * dt
    if increment <= 0.0:
        return damage
    return min(damage + (1.0 - damage) * increment, _DAMAGE_CAP)


def _apply_absorber_damage(
    stack: DeviceStack,
    damage: float,
    lifetime_strength: float,
    min_tau_factor: float,
) -> DeviceStack:
    """Map the scalar damage state onto shorter absorber carrier lifetimes."""
    if damage <= 0.0:
        return stack

    tau_factor = max(min_tau_factor, 1.0 / (1.0 + lifetime_strength * damage))
    degraded_layers = []
    for layer in stack.layers:
        if layer.role != "absorber":
            degraded_layers.append(layer)
            continue
        params = layer.params
        degraded_params = replace(
            params,
            tau_n=params.tau_n * tau_factor,
            tau_p=params.tau_p * tau_factor,
        )
        degraded_layers.append(replace(layer, params=degraded_params))
    return replace(stack, layers=tuple(degraded_layers))


def run_degradation(
    stack: DeviceStack,
    t_end: float = 1e5,       # seconds
    n_snapshots: int = 20,
    V_bias: float = 0.9,
    N_grid: int = 60,
    dt_max: float = 1.0,      # max internal time step [s]
    metric_n_points: int = 40,
    metric_V_max: float | None = None,
    metric_settle_time: float = 1e-2,  # per-voltage carrier settling time [s]
    rtol: float = 1e-4,
    atol: float = 1e-6,
    store_ion_profiles: bool = True,
    damage_motion_gain: float = 1.2,
    damage_stress_rate: float = 1e-3,
    damage_lifetime_strength: float = 4.0,
    min_tau_factor: float = 0.2,
    progress: ProgressCallback | None = None,
) -> DegradationResult:
    """Run constant-bias, ion-coupled degradation simulation.

    Ion drift alone is largely reversible and often looks like light-soaking.
    To make the long-time response physically degradation-like, this experiment
    accumulates an irreversible absorber damage state from two proxies:
    cumulative ion motion and sustained ion segregation. The damage feeds back
    as shorter absorber SRH lifetimes, which lowers V_oc and PCE under stress.
    """
    if t_end <= 0:
        raise ValueError(f"t_end must be positive, got {t_end}")
    if N_grid < 3:
        raise ValueError(f"N_grid must be >= 3, got {N_grid}")
    if n_snapshots < 1:
        raise ValueError(f"n_snapshots must be >= 1, got {n_snapshots}")
    if dt_max <= 0:
        raise ValueError(f"dt_max must be positive, got {dt_max}")
    if metric_n_points < 3:
        raise ValueError(f"metric_n_points must be >= 3, got {metric_n_points}")
    if metric_settle_time <= 0:
        raise ValueError(f"metric_settle_time must be positive, got {metric_settle_time}")
    if damage_motion_gain < 0:
        raise ValueError(
            f"damage_motion_gain must be non-negative, got {damage_motion_gain}"
        )
    if damage_stress_rate < 0:
        raise ValueError(
            f"damage_stress_rate must be non-negative, got {damage_stress_rate}"
        )
    if damage_lifetime_strength < 0:
        raise ValueError(
            "damage_lifetime_strength must be non-negative, "
            f"got {damage_lifetime_strength}"
        )
    if not (0.0 < min_tau_factor <= 1.0):
        raise ValueError(
            f"min_tau_factor must be in (0, 1], got {min_tau_factor}"
        )

    # Grid construction uses electrical layers only; substrate is optical-only.
    elec = electrical_layers(stack)
    layers_grid = [Layer(l.thickness, N_grid // len(elec)) for l in elec]
    x = multilayer_grid(layers_grid)
    N = len(x)
    # V_oc can exceed V_bi when heterojunction band offsets are present, so
    # sweep beyond V_bi; the caller may override. Default gives ~30 % headroom.
    v_upper = metric_V_max if metric_V_max is not None else max(stack.compute_V_bi() * 1.3, 1.4)
    metric_voltages = np.linspace(0.0, v_upper, metric_n_points)
    absorber_layer, absorber_mask = _absorber_region(x, stack)
    P0_abs = absorber_layer.params.P0

    # Emit a kickoff progress event so the user sees motion before the first
    # snapshot completes. The initial steady-state + first metric sweep can
    # take 10–20 s of silence otherwise.
    if progress is not None:
        progress("degradation_transient", 0, max(1, int(t_end * 1000)),
                 "solving illuminated steady state")

    # Degradation loop starts from V_bias-equilibrated state so that the very
    # first time chunk does not have to transition SC→V_bias carriers (expensive).
    y = solve_illuminated_ss(x, stack, V_app=V_bias, rtol=rtol, atol=atol)
    active_stack = stack
    mat_active = build_material_arrays(x, active_stack)
    damage = 0.0
    damage_cached = 0.0
    if n_snapshots == 1:
        t_eval = np.array([t_end])
    else:
        t_min = max(min(t_end * 1e-6, t_end), 1e-12)
        t_eval = np.concatenate([[0.0], np.geomspace(t_min, t_end, n_snapshots - 1)])

    PCE_arr = np.zeros(n_snapshots)
    V_oc_arr = np.zeros(n_snapshots)
    J_sc_arr = np.zeros(n_snapshots)
    ion_arr = np.zeros((n_snapshots, N)) if store_ion_profiles else None

    t_prev = 0.0
    for k, t_k in enumerate(t_eval):
        # March from t_prev to t_k in chunks of at most dt_max.
        # This bounds the fallback sub-step count to ceil(dt_max / 0.05)
        # regardless of how large the snapshot interval is.
        t_cur = t_prev
        while t_cur < t_k - 1e-12:
            dt_chunk = min(dt_max, t_k - t_cur)
            P_prev = StateVec.unpack(y, N).P[absorber_mask].copy()
            sol = run_transient(x, y, (t_cur, t_cur + dt_chunk),
                                np.array([t_cur + dt_chunk]),
                                active_stack, illuminated=True, V_app=V_bias,
                                rtol=rtol, atol=atol, mat=mat_active)
            if sol.success:
                y = sol.y[:, -1]
                P_now = StateVec.unpack(y, N).P[absorber_mask]
                damage = _advance_damage(
                    damage, P_prev, P_now, P0_abs, dt_chunk,
                    motion_gain=damage_motion_gain,
                    stress_rate=damage_stress_rate,
                )
                if damage != damage_cached:
                    active_stack = _apply_absorber_damage(
                        stack,
                        damage,
                        lifetime_strength=damage_lifetime_strength,
                        min_tau_factor=min_tau_factor,
                    )
                    mat_active = build_material_arrays(x, active_stack)
                    damage_cached = damage
            else:
                # Coupled solver stalled — operator splitting fallback.
                # dt_chunk ≤ dt_max, so sub-steps ≤ ceil(dt_max / 0.05).
                n_sub = max(1, int(np.ceil(dt_chunk / 0.05)))
                dt_sub = dt_chunk / n_sub
                for _ in range(n_sub):
                    P_prev = StateVec.unpack(y, N).P[absorber_mask].copy()
                    y_new, ok = split_step(x, y, dt_sub, active_stack, V_bias,
                                           rtol=rtol, atol=atol, mat=mat_active)
                    if not ok:
                        raise RuntimeError(
                            "degradation split_step failed to advance the state "
                            f"at t={t_cur:.6g} s"
                        )
                    y = y_new
                    P_now = StateVec.unpack(y, N).P[absorber_mask]
                    damage = _advance_damage(
                        damage, P_prev, P_now, P0_abs, dt_sub,
                        motion_gain=damage_motion_gain,
                        stress_rate=damage_stress_rate,
                    )
                    if damage != damage_cached:
                        active_stack = _apply_absorber_damage(
                            stack,
                            damage,
                            lifetime_strength=damage_lifetime_strength,
                            min_tau_factor=min_tau_factor,
                        )
                        mat_active = build_material_arrays(x, active_stack)
                        damage_cached = damage
            t_cur += dt_chunk
            # Sub-snapshot progress: emit on every chunk so the bar advances
            # continuously across the long final intervals (otherwise the user
            # sees no motion for minutes at a time near t_end).
            if progress is not None:
                progress(
                    "degradation_transient",
                    int(t_cur * 1000),
                    max(1, int(t_end * 1000)),
                    f"snap {k + 1}/{n_snapshots}  t={t_cur:.2e}/{t_end:.2e} s",
                )
        t_prev = t_k

        sv = StateVec.unpack(y, N)
        metrics = _measure_snapshot_metrics(
            x, y, active_stack, metric_voltages, metric_settle_time, rtol=rtol, atol=atol
        )
        PCE_arr[k] = metrics.PCE
        V_oc_arr[k] = metrics.V_oc
        J_sc_arr[k] = metrics.J_sc
        if store_ion_profiles:
            ion_arr[k] = sv.P

        if progress is not None:
            progress("degradation", k + 1, n_snapshots, f"t={float(t_k):.2e} s")

    return DegradationResult(t=t_eval, PCE=PCE_arr, V_oc=V_oc_arr,
                             J_sc=J_sc_arr, ion_profiles=ion_arr)
