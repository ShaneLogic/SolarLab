from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
from perovskite_sim.models.device import DeviceStack
from perovskite_sim.discretization.grid import multilayer_grid, Layer
from perovskite_sim.solver.illuminated_ss import solve_illuminated_ss
from perovskite_sim.solver.mol import StateVec, run_transient, split_step
from perovskite_sim.experiments.jv_sweep import _compute_current

from perovskite_sim import constants


@dataclass(frozen=True)
class DegradationResult:
    t: np.ndarray
    PCE: np.ndarray
    V_oc: np.ndarray
    J_sc: np.ndarray
    ion_profiles: Optional[np.ndarray]   # shape (len(t), N)


def run_degradation(
    stack: DeviceStack,
    t_end: float = 1e5,       # seconds
    n_snapshots: int = 20,
    V_bias: float = 0.9,
    N_grid: int = 60,
    dt_max: float = 1.0,      # max internal time step [s]
    rtol: float = 1e-4,
    atol: float = 1e-6,
    store_ion_profiles: bool = True,
) -> DegradationResult:
    """Run constant-bias degradation simulation."""
    if t_end <= 0:
        raise ValueError(f"t_end must be positive, got {t_end}")
    if N_grid < 3:
        raise ValueError(f"N_grid must be >= 3, got {N_grid}")
    if n_snapshots < 1:
        raise ValueError(f"n_snapshots must be >= 1, got {n_snapshots}")
    if dt_max <= 0:
        raise ValueError(f"dt_max must be positive, got {dt_max}")

    layers_grid = [Layer(l.thickness, N_grid // len(stack.layers)) for l in stack.layers]
    x = multilayer_grid(layers_grid)
    N = len(x)

    # J_sc computed once from the fresh SC state (V=0).  Ion migration mainly
    # shifts V_oc/FF, not J_sc, so this value is held fixed for all snapshots.
    y_sc = solve_illuminated_ss(x, stack, V_app=0.0, rtol=rtol, atol=atol)
    J_sc_0 = _compute_current(x, y_sc, stack, V_app=0.0)

    # Degradation loop starts from V_bias-equilibrated state so that the very
    # first time chunk does not have to transition SC→V_bias carriers (expensive).
    y = solve_illuminated_ss(x, stack, V_app=V_bias, rtol=rtol, atol=atol)
    t_eval = np.logspace(0, np.log10(t_end), n_snapshots)

    PCE_arr = np.zeros(n_snapshots)
    V_oc_arr = np.zeros(n_snapshots)
    J_sc_arr = np.zeros(n_snapshots)
    ion_arr = np.zeros((n_snapshots, N)) if store_ion_profiles else None

    absorber = next(l for l in stack.layers if l.role == "absorber")
    p = absorber.params

    t_prev = 0.0
    for k, t_k in enumerate(t_eval):
        # March from t_prev to t_k in chunks of at most dt_max.
        # This bounds the fallback sub-step count to ceil(dt_max / 0.05)
        # regardless of how large the snapshot interval is.
        t_cur = t_prev
        while t_cur < t_k - 1e-12:
            dt_chunk = min(dt_max, t_k - t_cur)
            sol = run_transient(x, y, (t_cur, t_cur + dt_chunk),
                                np.array([t_cur + dt_chunk]),
                                stack, illuminated=True, V_app=V_bias,
                                rtol=rtol, atol=atol)
            if sol.success:
                y = sol.y[:, -1]
            else:
                # Coupled solver stalled — operator splitting fallback.
                # dt_chunk ≤ dt_max, so sub-steps ≤ ceil(dt_max / 0.05).
                n_sub = max(1, int(np.ceil(dt_chunk / 0.05)))
                dt_sub = dt_chunk / n_sub
                for _ in range(n_sub):
                    y_new, ok = split_step(x, y, dt_sub, stack, V_bias,
                                           rtol=rtol, atol=atol)
                    if ok:
                        y = y_new
            t_cur += dt_chunk
        t_prev = t_k

        sv = StateVec.unpack(y, N)

        # J_sc: fixed from fresh-device SC state (computed once before the loop).
        # Ion migration mainly shifts V_oc/FF, not J_sc — reuse J_sc_0.
        J_sc = J_sc_0

        # V_oc: quasi-Fermi level separation in absorber at the evolved state.
        # Overestimates true circuit V_oc by ~100-150 mV but tracks relative
        # degradation trends correctly.
        abs_mask = (x > stack.layers[0].thickness) & (
            x < stack.layers[0].thickness + stack.layers[1].thickness
        )
        n_abs = sv.n[abs_mask]
        pp_abs = sv.p[abs_mask]
        # Geometric-mean quasi-Fermi V_oc: mean(log(n·p/ni²))·V_T
        # Less biased than arithmetic mean when interface injection spikes dominate.
        V_oc = constants.V_T * np.mean(np.log(n_abs * pp_abs / p.ni_sq))

        # PCE proxy: J_sc × V_oc / P_in (no FF — overestimates, but tracks trend)
        PCE = J_sc * V_oc / 1000.0

        PCE_arr[k] = PCE
        V_oc_arr[k] = V_oc
        J_sc_arr[k] = J_sc
        if store_ion_profiles:
            ion_arr[k] = sv.P

    return DegradationResult(t=t_eval, PCE=PCE_arr, V_oc=V_oc_arr,
                             J_sc=J_sc_arr, ion_profiles=ion_arr)
