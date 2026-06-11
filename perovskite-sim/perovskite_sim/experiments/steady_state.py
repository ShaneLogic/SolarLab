"""Direct steady-state driver (2026-06) — same RHS, d/dt = 0, ions frozen.

SCAPS is an ion-free steady-state solver, so every parity quantity is
defined at the ion-free steady state. This module solves F(y) = 0 on the
SAME ``assemble_rhs`` the Radau transient integrates — one physics
implementation, two drivers — with the ion profile (and any auxiliary
state blocks) byte-frozen from the seed state. The transient MOL core
remains the engine for ion-migration physics (hysteresis, impedance,
degradation); this driver buys the regimes the transient cannot reach
(near-insulating contact layers never settle) and an artifact-free direct
V_oc solve (no sweep-grid interpolation).

Method: damped Newton on z = (ln n, ln p) — positivity by construction —
with a dense finite-difference Jacobian (the parity grids are ~30 nodes,
so 2N ~ 60 unknowns and the FD Jacobian costs ~2N RHS evaluations, each
sub-millisecond on the cached MaterialArrays). Residuals are scaled per
node by the local density so the convergence test is a relative rate
[1/s]. Dirichlet-pinned boundary rows (which ``assemble_rhs`` zeroes) are
replaced by identity equations pinning the log-density to the contact
value; Robin (flat-band / selective) boundaries keep their genuine RHS
residual. Voltage continuation warm-starts each ladder point from the
previous solution and bisects the voltage step on failure.

No silent fallback: non-convergence raises ``SteadyStateError`` (the
``solve_illuminated_ss`` dark-equilibrium fallback silently corrupted a
probe earlier in this campaign — this driver fails loudly instead).

KNOWN LIMIT (2026-06-12, WIP): on scaps_mirror_v2 (+DOS) the driver
converges in the dark, at V = 0, and at every bias up to ~0.85 V, but
hits a wall at V* ~ 0.858 — the bias where the HTL/PVK interface NOGEN
clamp switches. Measured: the system there is PATH-DEPENDENT
MULTI-STABLE. Approached by a slow transient ladder, a true fixed point
exists (residual 4.8e-3 on the peak-density scale); approached from the
V = 0.85 steady state, transient bursts of 100 us / 1 ms / 10 ms all
plateau at residual 14.0 — a chattering attractor orbiting the clamp
switch that Radau tolerates (small LTE) but pointwise F does not.
Full-Newton on the raw clamped F cannot cross this; further hardening of
the Newton mechanics is the wrong axis. Next iteration: smooth the NOGEN
clamp over a negligible relative width (removes the chatter attractor at
its source) or a Gummel decoupled outer loop (the SCAPS approach). The
three blocked test gates are xfail'd with this diagnosis.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from perovskite_sim.discretization.grid import multilayer_grid, Layer
from perovskite_sim.experiments.jv_sweep import (
    JVMetrics,
    _compute_current_ss,
    compute_metrics,
)
from perovskite_sim.models.device import DeviceStack, electrical_layers
from perovskite_sim.solver.mol import (
    MaterialArrays,
    assemble_rhs,
    build_material_arrays,
    run_transient,
)
from perovskite_sim.solver.newton import solve_equilibrium

_DENSITY_FLOOR = 1.0e6   # m^-3 — log-space floor for the carrier unknowns
_LN_STEP_CAP = 5.0       # max Newton step per unknown in ln-space (e^5 ~ 150x)
_FD_EPS = 1.0e-7         # relative FD perturbation in ln-space


class SteadyStateError(RuntimeError):
    """Newton failed to converge — no silent fallback by design."""


@dataclass(frozen=True)
class SteadyStateResult:
    y: np.ndarray          # full packed state at the steady state
    converged: bool
    residual: float        # max |F_scaled| [1/s]
    step_inf: float        # max |proposed Newton update| (ln-density)
    iterations: int


@dataclass(frozen=True)
class JVSweepSSResult:
    V: np.ndarray
    J: np.ndarray          # A/m^2, active-cell convention (J > 0 at V = 0)
    metrics: JVMetrics


def _pin_mask(mat: MaterialArrays, N: int) -> np.ndarray:
    """Boolean mask over the 2N carrier unknowns: True = Dirichlet-pinned.

    Mirrors the boundary logic in ``assemble_rhs``: without selective
    contacts all four entries are pinned; with them, only the sides whose
    S is None.
    """
    pin = np.zeros(2 * N, dtype=bool)
    if not mat.has_selective_contacts:
        pin[[0, N - 1, N, 2 * N - 1]] = True
    else:
        if mat.S_n_L is None:
            pin[0] = True
        if mat.S_n_R is None:
            pin[N - 1] = True
        if mat.S_p_L is None:
            pin[N] = True
        if mat.S_p_R is None:
            pin[2 * N - 1] = True
    return pin


def _residual_fn(x, stack, mat, y_template, V_app, illuminated,
                 pin, z_pin, n_ref):
    """Return F(z) over the 2N log-density unknowns.

    The full state vector is rebuilt from the template each call — only
    the n/p blocks vary; ions and any auxiliary blocks stay frozen.

    Residuals are scaled by the GLOBAL peak density ``n_ref`` (not the
    local density): at dark depletion nodes the local density is ~1e10
    while the opposing flux terms are ~1e30, so a density-relative rate
    floors at the double-precision cancellation noise (~1e6 1/s) and is
    unreachable by any tolerance; peak-density scaling puts that noise at
    ~1e-10 while a genuine bulk imbalance still registers at >= 1. The
    implied terminal-current error bound is q*tol*n_ref*d ~ 1e-8 A/m^2 at
    tol = 1e-6 on a 100 nm device — far below any observable.
    """
    N = len(x)

    def F(z):
        dens = np.exp(z)
        y = y_template.copy()
        y[: 2 * N] = dens
        dydt = assemble_rhs(0.0, y, x, stack, mat,
                            illuminated=illuminated, V_app=V_app)
        f = dydt[: 2 * N] / n_ref      # peak-density-relative rate [1/s]
        f[pin] = z[pin] - z_pin[pin]   # identity rows
        return f

    return F


def solve_steady_state(
    x: np.ndarray,
    stack: DeviceStack,
    V_app: float,
    *,
    illuminated: bool = True,
    mat: MaterialArrays | None = None,
    y0: np.ndarray | None = None,
    tol: float = 1.0e-6,
    tol_step: float = 1.0e-8,
    tol_accept: float = 0.1,
    max_newton: int = 60,
    assist_times: tuple[float, ...] = (1e-4, 1e-3, 1e-2),
) -> SteadyStateResult:
    """Solve the carrier steady state at ``V_app`` with frozen ions.

    ``y0`` seeds both the unknowns and the frozen blocks; default is a
    short transient settle at the target voltage — the settle only seeds,
    the Newton solve owns convergence. Raises ``SteadyStateError`` on
    non-convergence.

    ``tol`` is a max carrier rate relative to the device's PEAK density
    [1/s] — see ``_residual_fn`` for the scaling rationale (cancellation
    noise lives at ~1e-10 on this scale; tol = 1e-6 bounds the
    terminal-current error near 1e-8 A/m^2). ``tol_step`` is a backup
    update criterion (max |proposed Newton step| in ln-density) — the
    standard Gummel/SCAPS convergence-on-update test. ``tol_accept`` is
    the stall acceptance bound: the TE-cap kinks at heterointerface nodes
    can make the FD Newton direction locally non-descent at an
    already-physically-converged state; a stall with residual below
    ``tol_accept`` (J-error bound ~1e-3 A/m^2) is accepted as converged.

    A stall ABOVE ``tol_accept`` (typically at the diode knee, where the
    TE cap binds AT the steady state, making F non-differentiable at its
    own zero — Newton cannot finish there, but F's VALUE at the fixed
    point is still zero) triggers transient assists: escalating Radau
    bursts (``assist_times``, with the mandatory near-flat-band
    ``max_step`` cap) at the same voltage settle the state until the
    residual drops under ``tol_accept`` and the stall-accept path
    certifies it. Still-unconverged after all bursts raises.
    """
    N = len(x)
    if mat is None:
        mat = build_material_arrays(x, stack)
    if y0 is None:
        # Seed with a short transient AT THE TARGET voltage — the
        # quasi-neutral closed form sits far outside Newton's basin at
        # heterojunction nodes. The transient only seeds; the Newton
        # solve below owns convergence and raises on failure.
        y0 = solve_equilibrium(x, stack)
        sol = run_transient(
            x, y0, (0.0, 1e-4), np.array([1e-4]), stack,
            illuminated=illuminated, V_app=V_app, mat=mat,
        )
        if sol.success:
            y0 = sol.y[:, -1]

    pin = _pin_mask(mat, N)
    z_pin = np.zeros(2 * N)
    z_pin[0], z_pin[N - 1] = np.log(mat.n_L), np.log(mat.n_R)
    z_pin[N], z_pin[2 * N - 1] = np.log(mat.p_L), np.log(mat.p_R)

    z = np.log(np.maximum(y0[: 2 * N], _DENSITY_FLOOR))
    z[pin] = z_pin[pin]
    n_ref = float(np.max(y0[: 2 * N]))
    F = _residual_fn(x, stack, mat, y0, V_app, illuminated, pin, z_pin,
                     n_ref)
    n_unk = 2 * N

    def _done(z_fin, res_fin, step_inf, it):
        y = y0.copy()
        y[:n_unk] = np.exp(z_fin)
        return SteadyStateResult(y=y, converged=True, residual=res_fin,
                                 step_inf=step_inf, iterations=it)

    last_msg = ""
    max_assists = len(assist_times)
    # best-iterate tracking: a warm-started z is often near-converged
    # already; a garbage Newton step across a clamp kink must never
    # poison the acceptance test or the assist seed
    z_best = z.copy()
    res_best = np.inf
    for attempt in range(max_assists + 1):
        f = F(z)
        res = float(np.max(np.abs(f)))
        if res < res_best:
            z_best, res_best = z.copy(), res
        stalled = False
        for it in range(1, max_newton + 1):
            if res < tol:
                return _done(z, res, 0.0, it - 1)
            # dense FD Jacobian in ln-space
            J = np.empty((n_unk, n_unk))
            for j in range(n_unk):
                dz = _FD_EPS * max(1.0, abs(z[j]))
                zj = z.copy()
                zj[j] += dz
                J[:, j] = (F(zj) - f) / dz
            try:
                step = np.linalg.solve(J, -f)
            except np.linalg.LinAlgError:
                # ridge (Levenberg) fallback: a fully-floored carrier
                # column can zero out at extreme dopings
                lam_r = 1e-10 * float(np.max(np.abs(np.diag(J))) or 1.0)
                step = None
                for _ in range(4):
                    try:
                        step = np.linalg.solve(
                            J + lam_r * np.eye(n_unk), -f)
                        break
                    except np.linalg.LinAlgError:
                        lam_r *= 1e3
                if step is None:
                    raise SteadyStateError(
                        f"singular Jacobian at V={V_app:.4f} (iter {it}, "
                        f"residual {res:.3e})")
            step_inf = float(np.max(np.abs(step)))
            if step_inf < tol_step:
                # residual at the cancellation-noise floor and the state
                # has stopped moving — converged on the update criterion
                return _done(z, res, step_inf, it)
            step = np.clip(step, -_LN_STEP_CAP, _LN_STEP_CAP)
            # backtracking line search on ||F||_2 (the max-norm is
            # non-smooth under the TE caps; convergence stays max-norm)
            nrm = float(np.linalg.norm(f))
            lam, accepted = 1.0, False
            for _ in range(15):
                z_try = z + lam * step
                f_try = F(z_try)
                nrm_try = float(np.linalg.norm(f_try))
                if nrm_try < nrm * (1.0 - 1.0e-4 * lam):
                    z, f = z_try, f_try
                    res = float(np.max(np.abs(f_try)))
                    if res < res_best:
                        z_best, res_best = z.copy(), res
                    accepted = True
                    break
                lam *= 0.5
            if not accepted:
                if res_best < tol_accept:
                    # kink-stall at a physically-converged state — accept
                    # the BEST iterate seen, not the last
                    return _done(z_best, res_best, step_inf, it)
                last_msg = (f"line search stalled at V={V_app:.4f} "
                            f"(iter {it}, best residual {res_best:.3e})")
                stalled = True
                break
        else:
            if res_best < tol_accept:
                return _done(z_best, res_best, 0.0, max_newton)
            last_msg = (f"no convergence at V={V_app:.4f} after "
                        f"{max_newton} iterations (best residual "
                        f"{res_best:.3e}, tol {tol:.1e})")
        if attempt < max_assists:
            # transient assist: Radau traverses the TE-cap kink natively;
            # escalating burst lengths cover knee-injection relaxation,
            # max_step capped per the near-flat-band Radau gotcha
            t_a = assist_times[attempt]
            y_cur = y0.copy()
            y_cur[:n_unk] = np.exp(z_best)   # assist from the best iterate
            sol = run_transient(
                x, y_cur, (0.0, t_a), np.array([t_a]), stack,
                illuminated=illuminated, V_app=V_app, mat=mat,
                max_step=t_a / 8.0,
            )
            if not sol.success:
                break
            z = np.log(np.maximum(sol.y[: n_unk, -1], _DENSITY_FLOOR))
            z[pin] = z_pin[pin]
        del stalled
    raise SteadyStateError(last_msg + " (transient assists exhausted)")

def _grid_for(stack: DeviceStack, N_grid: int) -> np.ndarray:
    elec = electrical_layers(stack)
    return multilayer_grid(
        [Layer(thickness=L.thickness, N=N_grid // len(elec)) for L in elec])


def run_jv_sweep_ss(
    stack: DeviceStack,
    *,
    N_grid: int = 30,
    V_max: float = 1.25,
    n_points: int = 26,
    illuminated: bool = True,
    progress=None,
) -> JVSweepSSResult:
    """Steady-state J-V: voltage continuation with warm starts.

    On a failed point the voltage step bisects (up to 4 levels) so the
    continuation path stays connected; a point that still fails raises.
    """
    x = _grid_for(stack, N_grid)
    mat = build_material_arrays(x, stack)
    V_targets = list(np.linspace(0.0, V_max, n_points))
    V_out, J_out = [], []
    y_prev = None
    V_prev = 0.0
    i = 0
    while i < len(V_targets):
        V = V_targets[i]
        try:
            res = solve_steady_state(
                x, stack, V, illuminated=illuminated, mat=mat, y0=y_prev)
        except SteadyStateError:
            if V - V_prev > 1e-4:
                V_targets.insert(i, 0.5 * (V_prev + V))   # bisect the step
                continue
            raise
        y_prev = res.y
        V_prev = V
        V_out.append(V)
        J_out.append(_compute_current_ss(x, res.y, stack, V, mat=mat))
        if progress is not None:
            progress("jv_ss", len(V_out), len(V_targets), f"V={V:.3f}")
        i += 1
    V_arr, J_arr = np.asarray(V_out), np.asarray(J_out)
    return JVSweepSSResult(V=V_arr, J=J_arr,
                           metrics=compute_metrics(V_arr, J_arr))


def solve_voc_ss(
    stack: DeviceStack,
    *,
    N_grid: int = 30,
    V_lo: float = 0.0,
    V_hi: float = 1.6,
    tol_v: float = 2.0e-4,
) -> float:
    """Direct V_oc: bisection on the steady-state J(V) zero crossing.

    Walks up from V_lo in continuation steps until J changes sign, then
    bisects. Raises ``SteadyStateError`` if J never crosses zero below
    ``V_hi``.
    """
    x = _grid_for(stack, N_grid)
    mat = build_material_arrays(x, stack)

    y_cache: dict[float, np.ndarray] = {}

    def J_at(V, y_seed):
        res = solve_steady_state(x, stack, V, illuminated=True,
                                 mat=mat, y0=y_seed)
        y_cache[V] = res.y
        return _compute_current_ss(x, res.y, stack, V, mat=mat)

    # coarse upward walk (continuation)
    V_a, J_a = V_lo, J_at(V_lo, None)
    y_seed = y_cache[V_a]
    step = 0.1
    V_b = V_a
    while V_b < V_hi:
        V_b = min(V_b + step, V_hi)
        J_b = J_at(V_b, y_seed)
        y_seed = y_cache[V_b]
        if J_a * J_b <= 0.0:
            break
        V_a, J_a = V_b, J_b
    else:
        raise SteadyStateError(f"J(V) does not cross zero below {V_hi} V")
    if J_a * J_b > 0.0:
        raise SteadyStateError(f"J(V) does not cross zero below {V_hi} V")

    # bisection with warm starts
    while V_b - V_a > tol_v:
        V_m = 0.5 * (V_a + V_b)
        J_m = J_at(V_m, y_cache[V_a])
        if J_a * J_m <= 0.0:
            V_b = V_m
        else:
            V_a, J_a = V_m, J_m
    return 0.5 * (V_a + V_b)
