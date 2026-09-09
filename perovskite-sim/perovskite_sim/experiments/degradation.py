from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Callable, Optional
import numpy as np

from perovskite_sim.models.device import DeviceStack, electrical_layers
from perovskite_sim.solver.illuminated_ss import solve_illuminated_ss
from perovskite_sim.solver.mol import StateVec, run_transient, split_step, build_material_arrays
from perovskite_sim.solver.tolerances import AbsoluteTolerance, ComponentwiseAtol
from perovskite_sim.experiments.jv_sweep import (
    JVMetrics,
    build_electrical_grid,
    compute_current_components,
    compute_metrics,
)
from perovskite_sim.experiments.steady_state import solve_steady_state


ProgressCallback = Callable[[str, int, int, str], None]


def _freeze_ions(stack: DeviceStack) -> DeviceStack:
    """Freeze both diffusivities without changing ion inventories or capacities.

    Used for snapshot J-V measurement: carriers relax to quasi-equilibrium
    at each probe voltage while the ion distribution stays pinned at the
    degradation-snapshot value. This is the numerical analogue of a fast
    laboratory sweep (sweep time ≪ τ_ion).
    """
    new_layers = []
    for layer in stack.layers:
        if layer.params is None:
            new_layers.append(layer)
            continue
        new_layers.append(
            replace(layer, params=replace(layer.params, D_ion=0.0, D_ion_neg=0.0))
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
    ion_profiles_neg: np.ndarray | None = None
    snapshot_carrier_current_bound_A_m2: np.ndarray | None = None
    snapshot_current_spread_A_m2: np.ndarray | None = None


@dataclass(frozen=True)
class FrozenIonSnapshot:
    """Electronically stationary J-V states at one fixed ionic configuration."""

    V: np.ndarray
    J: np.ndarray
    states: np.ndarray
    carrier_residual_s_inv: np.ndarray
    carrier_current_bound_A_m2: np.ndarray
    current_spread_A_m2: np.ndarray
    metrics: JVMetrics


def measure_frozen_ion_snapshot(
    x: np.ndarray,
    y_ref: np.ndarray,
    stack: DeviceStack,
    voltages: np.ndarray,
    settle_time: float,
    rtol: float,
    atol: AbsoluteTolerance,
    *,
    max_step: float | None = None,
    max_current_error_A_m2: float = 0.05,
) -> FrozenIonSnapshot:
    """Measure a stationary electronic curve with both ion profiles frozen.

    Every probe starts from the same snapshot. The declared transient dwell
    supplies a seed for the existing carrier steady-state solve, which owns
    acceptance. Both solves and current extraction use the frozen material.
    This separates electronic relaxation from the much slower ionic history;
    it does not establish ionic equilibrium or a chemical degradation model.
    """
    if not np.isfinite(settle_time) or settle_time <= 0.0:
        raise ValueError("settle_time must be finite and positive")
    if not np.isfinite(rtol) or rtol <= 0.0:
        raise ValueError("rtol must be finite and positive")
    if not isinstance(atol, ComponentwiseAtol) and (not np.isfinite(atol) or atol <= 0.0):
        raise ValueError("atol must be finite and positive")
    if not np.isfinite(max_current_error_A_m2) or max_current_error_A_m2 <= 0.0:
        raise ValueError("max_current_error_A_m2 must be finite and positive")
    if max_step is None:
        max_step = settle_time / 20.0
    if not np.isfinite(max_step) or max_step <= 0.0:
        raise ValueError("max_step must be finite and positive")
    x = np.asarray(x, dtype=float)
    y_ref = np.asarray(y_ref, dtype=float).copy()
    frozen_stack = _freeze_ions(stack)
    mat_frozen = build_material_arrays(x, frozen_stack)
    # Preserve a declared but empty negative block in an inherited state too.
    if not mat_frozen.has_dual_ions and y_ref.size == 4 * x.size + 4 * mat_frozen.N_iface_state:
        original = build_material_arrays(x, stack)
        if original.has_dual_ions:
            mat_frozen = replace(
                mat_frozen, has_dual_ions=True,
                D_ion_neg_node=np.zeros_like(original.D_ion_neg_node),
                D_ion_neg_face=np.zeros_like(original.D_ion_neg_face),
                P_ion0_neg=original.P_ion0_neg.copy(),
                P_lim_neg_node=original.P_lim_neg_node.copy(),
                P_lim_neg_face=original.P_lim_neg_face.copy(),
            )
    voltages = np.asarray(voltages, dtype=float)
    if voltages.ndim != 1 or voltages.size == 0 or not np.all(np.isfinite(voltages)):
        raise ValueError("voltages must be a non-empty finite vector")
    expected_size = (4 if mat_frozen.has_dual_ions else 3) * x.size + 4 * mat_frozen.N_iface_state
    if y_ref.shape != (expected_size,):
        raise ValueError("snapshot state does not match the frozen material layout")
    reference = StateVec.unpack(y_ref, x.size, N_iface_state=mat_frozen.N_iface_state)

    def require_frozen_state(state):
        if state.shape != y_ref.shape or not np.all(np.isfinite(state)) or np.any(state < 0.0):
            raise RuntimeError("snapshot has a non-finite, negative, or malformed density state")
        blocks = StateVec.unpack(state, x.size, N_iface_state=mat_frozen.N_iface_state)
        for name, values, original, capacity in (
            ("positive", blocks.P, reference.P, mat_frozen.P_lim_node),
            ("negative", blocks.P_neg, reference.P_neg, mat_frozen.P_lim_neg_node),
        ):
            if values is None:
                continue
            if not np.array_equal(values, original):
                raise RuntimeError(f"frozen {name} ion distribution changed during the snapshot")
            if capacity is not None and np.any(values > capacity):
                raise RuntimeError(f"frozen {name} ions exceed their site capacity")
        if (mat_frozen.has_dual_ions and mat_frozen.ion_steric_diffusion_only
                and mat_frozen.ion_steric_shared_site
                and np.any(blocks.P + blocks.P_neg > mat_frozen.P_lim_node)):
            raise RuntimeError("frozen ions exceed their shared site capacity")

    require_frozen_state(y_ref)
    J_arr = np.zeros_like(voltages)
    states = np.empty((voltages.size, y_ref.size))
    residuals = np.empty_like(voltages)
    current_bounds = np.empty_like(voltages)
    current_spreads = np.empty_like(voltages)
    snap_rtol = min(rtol, 1e-5)
    if isinstance(atol, ComponentwiseAtol):
        effective_floor = atol.minimum_atol * atol.refinement_factor
        snap_atol = atol.refined(min(1.0, 1.0e-8 / effective_floor))
    else:
        snap_atol = min(atol, 1e-8)
    for k, V_k in enumerate(voltages):
        sol = run_transient(
            x, y_ref, (0.0, settle_time), np.array([settle_time]),
            frozen_stack, illuminated=True, V_app=float(V_k),
            rtol=snap_rtol, atol=snap_atol, max_step=max_step,
            mat=mat_frozen,
            max_nfev=100_000,
        )
        if not sol.success or np.asarray(sol.y).shape != (y_ref.size, 1):
            raise RuntimeError(
                f"snapshot J-V solver failed at V={V_k:.4f} V"
            )
        require_frozen_state(sol.y[:, -1])
        stationary = solve_steady_state(
            x, frozen_stack, float(V_k), mat=mat_frozen, y0=sol.y[:, -1],
            max_continuity_current_error=max_current_error_A_m2,
        )
        if (not stationary.converged or not np.isfinite(stationary.residual)
                or stationary.residual < 0.0
                or not np.isfinite(stationary.continuity_current_bound)
                or not 0.0 <= stationary.continuity_current_bound <= max_current_error_A_m2):
            raise RuntimeError(f"snapshot lacks an electronic steady-state certificate at V={V_k:.6g} V")
        y_v = stationary.y
        require_frozen_state(y_v)
        currents = compute_current_components(x, y_v, frozen_stack, float(V_k), mat=mat_frozen)
        if not np.all(np.isfinite(currents.J_total)) or np.any(currents.J_ion != 0.0):
            raise RuntimeError(f"snapshot has a non-finite current or nonzero frozen-ion current at V={V_k:.6g} V")
        spread = float(np.ptp(currents.J_total))
        if spread > max_current_error_A_m2:
            raise RuntimeError(f"snapshot current is not spatially continuous at V={V_k:.6g} V: {spread:.6g} A/m2")
        states[k] = y_v
        residuals[k] = stationary.residual
        current_bounds[k] = stationary.continuity_current_bound
        current_spreads[k] = spread
        J_arr[k] = currents.J_total[0]
    return FrozenIonSnapshot(
        V=voltages.copy(), J=J_arr, states=states, carrier_residual_s_inv=residuals,
        carrier_current_bound_A_m2=current_bounds, current_spread_A_m2=current_spreads,
        metrics=compute_metrics(voltages, J_arr),
    )


def _measure_snapshot_metrics(x, y_ref, stack, voltages, settle_time, rtol, atol):
    return measure_frozen_ion_snapshot(x, y_ref, stack, voltages, settle_time, rtol, atol).metrics


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
    atol: AbsoluteTolerance = 1e-6,
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
    x = build_electrical_grid(stack, N_grid)
    N = len(x)
    # V_oc can exceed V_bi when heterojunction band offsets are present, so
    # sweep beyond V_bi; the caller may override. Default gives ~30 % headroom.
    v_upper = (
        metric_V_max
        if metric_V_max is not None
        else max(abs(stack.operating_built_in_potential()) * 1.3, 1.4)
    )
    metric_voltages = np.linspace(0.0, v_upper, metric_n_points)
    absorber_layer, absorber_mask = _absorber_region(x, stack)
    P0_abs = absorber_layer.params.P0

    # Emit a kickoff progress event so the user sees motion before the first
    # snapshot completes. The initial steady-state + first metric sweep can
    # take 10–20 s of silence otherwise.
    if progress is not None:
        progress("degradation_transient", 0, max(1, int(t_end * 1000)),
                 "applying illuminated preconditioning")

    # Degradation loop starts from V_bias-equilibrated state so that the very
    # first time chunk does not have to transition SC→V_bias carriers (expensive).
    y = solve_illuminated_ss(x, stack, V_app=V_bias, rtol=rtol, atol=atol)
    active_stack = stack
    mat_active = build_material_arrays(x, active_stack)
    damage = 0.0
    damage_cached = 0.0
    if n_snapshots == 1:
        t_eval = np.array([t_end])
    elif n_snapshots == 2:
        t_eval = np.array([0.0, t_end])
    else:
        t_min = max(min(t_end * 1e-6, t_end), 1e-12)
        t_eval = np.concatenate([[0.0], np.geomspace(t_min, t_end, n_snapshots - 1)])

    PCE_arr = np.zeros(n_snapshots)
    V_oc_arr = np.zeros(n_snapshots)
    J_sc_arr = np.zeros(n_snapshots)
    ion_arr = np.zeros((n_snapshots, N)) if store_ion_profiles else None
    ion_neg_arr = (np.zeros((n_snapshots, N))
                   if store_ion_profiles and mat_active.has_dual_ions else None)
    snapshot_bounds = np.zeros(n_snapshots)
    snapshot_spreads = np.zeros(n_snapshots)

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
        snapshot = measure_frozen_ion_snapshot(
            x, y, active_stack, metric_voltages, metric_settle_time, rtol=rtol, atol=atol
        )
        metrics = snapshot.metrics
        snapshot_bounds[k] = np.max(snapshot.carrier_current_bound_A_m2)
        snapshot_spreads[k] = np.max(snapshot.current_spread_A_m2)
        PCE_arr[k] = metrics.PCE
        V_oc_arr[k] = metrics.V_oc
        J_sc_arr[k] = metrics.J_sc
        if store_ion_profiles:
            ion_arr[k] = sv.P
            if ion_neg_arr is not None:
                ion_neg_arr[k] = sv.P_neg

        if progress is not None:
            progress("degradation", k + 1, n_snapshots, f"t={float(t_k):.2e} s")

    return DegradationResult(
        t=t_eval, PCE=PCE_arr, V_oc=V_oc_arr, J_sc=J_sc_arr, ion_profiles=ion_arr,
        ion_profiles_neg=ion_neg_arr, snapshot_carrier_current_bound_A_m2=snapshot_bounds,
        snapshot_current_spread_A_m2=snapshot_spreads,
    )
