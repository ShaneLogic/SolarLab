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

RESOLVED (2026-06-12): the V* ~ 0.858 wall on scaps_mirror_v2 (+DOS) —
the bias where the HTL/PVK interface switch makes F non-smooth and the
system path-dependent multi-stable (slow approach: true fixed point,
residual 4.8e-3; approach from the adjacent steady state: chattering
attractor, residual pinned at 14 across 100 us - 10 ms bursts). The
resolution, each step measured: (1) smoothed TE cap
(MaterialArrays.te_softness, SS-driver-scoped — reproduces the
cap-removal improvement exactly, 15x); (2) physical stall-acceptance
bound (J-error based, rejects the attractor by 30x); (3) certified
transient point-fallback (_transient_point_fallback): where Newton still
cannot finish, escalating 6.4-100 ms Radau settles compute the point,
residual-guarded — clean states keep the continuation un-poisoned (the
stall-accepted states near the wall were corrupting subsequent warm
starts). Measured outcome: SS J-V matches the frozen-ion transient
within 5 mV V_oc / 1% J_sc end-to-end; direct V_oc agrees with the sweep
within 2 mV.

KNOWN LIMIT: the near-insulating contact regime (e.g. Nd_ETL = 1e10)
still fails honestly — J(V) finds no zero crossing below 1.6 V; the
transient fallback cannot settle that regime by definition and the
Newton path cannot yet converge it. Needs the Gummel decoupled
iteration; the gate is xfail'd with this diagnosis.
"""
from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass

import numpy as np
from scipy.linalg import lu_factor, lu_solve

from perovskite_sim.discretization.grid import multilayer_grid, Layer
from perovskite_sim.experiments.jv_sweep import (
    JVMetrics,
    _compute_current_ss,
    compute_metrics,
)
from perovskite_sim.models.device import DeviceStack, electrical_layers
from perovskite_sim.constants import Q
from perovskite_sim.physics.poisson import solve_poisson_prefactored
from perovskite_sim.solver.mol import (
    MaterialArrays,
    _charge_density,
    _compute_iface_state_dark_eq,
    assemble_rhs,
    build_material_arrays,
    run_transient,
)
from perovskite_sim.solver.newton import solve_equilibrium

_DENSITY_FLOOR = 1.0e6   # m^-3 — log-space floor for the carrier unknowns
# Relative TE-cap smoothing width for the SS driver's own MaterialArrays
# (see MaterialArrays.te_softness): rounds the hard magnitude-min kink at
# heterointerface faces — the measured dominant non-smoothness at the
# V*~0.858 wall — while shifting fluxes only within ~2% of the crossover
# region. Transient experiments never see this.
_TE_SOFTNESS = 0.02
# Full thermal velocity for the interface-plane state block [m/s] in the
# steady-state Newton (P1 of scaps_mode, 2026-06). The transient's 1e-2
# throttle exists only because full velocity makes the ODE block too
# stiff for Radau; an algebraic solve has no such constraint.
_IFACE_STATE_V_TH = 1.0e5
# Modified-Newton Jacobian/LU reuse in solve_steady_state (2026-06). On =
# chord iterations reuse the cached LU factor (~2-4x fewer assemble_rhs
# evals on warm-started solves); a freshly built step stays bit-identical
# to the original full-Newton path. Set SOLARLAB_SS_JAC_REUSE=0 to force a
# fresh Jacobian every iteration (the legacy behaviour / kill-switch).
_SS_JAC_REUSE = os.environ.get("SOLARLAB_SS_JAC_REUSE", "1") != "0"


def _enable_iface_states(mat: MaterialArrays) -> MaterialArrays:
    """Activate the SCAPS-style interface-plane state block on a built
    mat: densities AT each heterointerface plane become true unknowns,
    TE-coupled to the adjacent nodes, with the two-sided P-V
    recombination evaluated on them (assemble_rhs's Phase-E3 block; the
    bulk-sampled interface recombination disables automatically)."""
    n_av = len(mat.interface_V_partition_2)
    if n_av == 0:
        return mat
    repl = dict(N_iface_state=n_av,
                iface_state_v_th=_IFACE_STATE_V_TH,
                iface_state_live_proj=True,
                iface_state_shared_occ=True)
    # SS-only interface-channel calibration: fold the per-interface
    # iface_state_calibration into the cf the state-SRH rate already reads
    # (compute_interface_srh_on_state is reached only on this SS path, so
    # the transient bulk-node interface path is untouched). Default 1.0 =
    # bit-identical.
    ss_cal = getattr(mat, "iface_state_calibration", ())
    if ss_cal and any(float(c) != 1.0 for c in ss_cal):
        base_cf = (mat.interface_calibration_factor
                   or tuple(1.0 for _ in range(len(ss_cal))))
        folded = tuple(
            float(base_cf[k]) * float(ss_cal[k]) if k < len(ss_cal)
            else float(base_cf[k])
            for k in range(len(base_cf))
        )
        repl["interface_calibration_factor"] = folded
    return dataclasses.replace(mat, **repl)


def _ensure_iface_block(y: np.ndarray, mat: MaterialArrays) -> np.ndarray:
    """Append the dark-equilibrium interface-state block when the seed
    predates activation (e.g. solve_equilibrium built its own mat)."""
    k = 4 * mat.N_iface_state
    if k == 0:
        return y
    # StateVec layout: [n, p, P, (P_neg), iface(4K)] — block at the end.
    # Without the block len(y) is 3N (single ion) or 4N (dual); with it,
    # that plus 4K. Anything else is a caller bug — fail loudly.
    N = mat.poisson_factor.N
    base = 4 * N if getattr(mat, "has_dual_ions", False) else 3 * N
    if len(y) == base + k:
        return y
    if len(y) == base:
        return np.concatenate([y, _compute_iface_state_dark_eq(mat)])
    raise SteadyStateError(
        f"state length {len(y)} matches neither {base} nor {base + k}")
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
                 pin, z_pin, n_ref, unk_idx):
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
        y[unk_idx] = dens
        dydt = assemble_rhs(0.0, y, x, stack, mat,
                            illuminated=illuminated, V_app=V_app)
        f = dydt[unk_idx] / n_ref      # peak-density-relative rate [1/s]
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
    tol_accept: float = 0.5,
    max_newton: int = 60,
    assist_times: tuple[float, ...] = (1e-4, 1e-3, 1e-2),
    iface_states: bool = False,
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
    ``tol_accept`` is accepted as converged. The default 0.5 bounds the
    implied terminal-current error near 8e-3 A/m^2 (~30 ppm of J_sc) while
    rejecting the V* chattering-attractor state (residual ~14) by 30x —
    both margins physical, not tuned. With the smoothed TE cap
    (``_TE_SOFTNESS``) the V*~0.858 wall stalls at residual 0.103: the
    smoothing reproduces the cap-removal improvement exactly (15x) and
    this acceptance bound then clears the wall.

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
        mat = dataclasses.replace(
            build_material_arrays(x, stack), te_softness=_TE_SOFTNESS)
        if iface_states:
            mat = _enable_iface_states(mat)
    if y0 is None:
        # Seed with a short transient AT THE TARGET voltage — the
        # quasi-neutral closed form sits far outside Newton's basin at
        # heterojunction nodes. The transient only seeds; the Newton
        # solve below owns convergence and raises on failure.
        y0 = _ensure_iface_block(solve_equilibrium(x, stack), mat)
        sol = run_transient(
            x, y0, (0.0, 1e-4), np.array([1e-4]), stack,
            illuminated=illuminated, V_app=V_app, mat=mat,
        )
        if sol.success:
            y0 = sol.y[:, -1]
    else:
        y0 = _ensure_iface_block(y0, mat)
    # Gummel phi-step: kill the dielectric-relaxation mode analytically
    # before Newton sees the state (decisive in near-insulating layers)
    y0 = _qfl_poisson_relax(x, mat, y0, V_app)

    # unknowns: carriers (first 2N) + interface-plane block (trailing 4K)
    K = 4 * mat.N_iface_state
    unk_idx = (np.r_[0: 2 * N, len(y0) - K: len(y0)]
               if K else np.arange(2 * N))
    n_unk = len(unk_idx)
    pin = np.zeros(n_unk, dtype=bool)
    pin[: 2 * N] = _pin_mask(mat, N)
    z_pin = np.zeros(n_unk)
    z_pin[0], z_pin[N - 1] = np.log(mat.n_L), np.log(mat.n_R)
    z_pin[N], z_pin[2 * N - 1] = np.log(mat.p_L), np.log(mat.p_R)

    z = np.log(np.maximum(y0[unk_idx], _DENSITY_FLOOR))
    z[pin] = z_pin[pin]
    n_ref = float(np.max(y0[: 2 * N]))
    F = _residual_fn(x, stack, mat, y0, V_app, illuminated, pin, z_pin,
                     n_ref, unk_idx)

    def _done(z_fin, res_fin, step_inf, it):
        y = y0.copy()
        y[unk_idx] = np.exp(z_fin)
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
        # Modified-Newton Jacobian reuse (2026-06): build+factor the dense FD
        # Jacobian only when no usable factor exists (first iter, or after a
        # stale factor loses contraction). Chord iterations reuse the cached
        # LU factor at ~1 F-eval vs n_unk full assemble_rhs evals — the
        # dominant cost. A freshly built+solved step is bit-identical to the
        # original full-Newton np.linalg.solve path; only reused (chord)
        # iterations differ, and they converge to the same residual tol.
        lu = None          # cached LU factor of the current Jacobian; None => rebuild
        jac_fresh = False  # True when this iter's step came from a fresh factor
        for it in range(1, max_newton + 1):
            if res < tol:
                return _done(z, res, 0.0, it - 1)
            if lu is None or not _SS_JAC_REUSE:
                # dense FD Jacobian in ln-space (rebuilt only when needed)
                J = np.empty((n_unk, n_unk))
                for j in range(n_unk):
                    dz = _FD_EPS * max(1.0, abs(z[j]))
                    zj = z.copy()
                    zj[j] += dz
                    J[:, j] = (F(zj) - f) / dz
                try:
                    step = np.linalg.solve(J, -f)
                    lu = lu_factor(J)   # cache for subsequent chord steps
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
                    lu = None   # ridge-perturbed singular J — do not reuse it
                jac_fresh = True
            else:
                # chord step: reuse the cached factor (no Jacobian rebuild)
                step = lu_solve(lu, -f)
                if not np.all(np.isfinite(step)):
                    lu = None
                    continue
                jac_fresh = False
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
            if accepted:
                # a reused factor that needed real backtracking is losing
                # contraction — refresh it next iteration
                if not jac_fresh and lam < 0.5:
                    lu = None
            else:
                if not jac_fresh:
                    # the rejected step used a STALE factor — rebuild and
                    # retry this iteration with a fresh Jacobian before
                    # declaring a stall (preserves the original
                    # fresh-Jacobian stall semantics)
                    lu = None
                    continue
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
            y_cur[unk_idx] = np.exp(z_best)  # assist from the best iterate
            sol = run_transient(
                x, y_cur, (0.0, t_a), np.array([t_a]), stack,
                illuminated=illuminated, V_app=V_app, mat=mat,
                max_step=t_a / 8.0,
            )
            if not sol.success:
                break
            y_rel = y0.copy()
            y_rel[unk_idx] = sol.y[unk_idx, -1]
            y_rel = _qfl_poisson_relax(x, mat, y_rel, V_app)
            z = np.log(np.maximum(y_rel[unk_idx], _DENSITY_FLOOR))
            z[pin] = z_pin[pin]
        del stalled
    raise SteadyStateError(last_msg + " (transient assists exhausted)")

def _qfl_poisson_relax(x, mat, y, V_app, *, tol_phi=1e-10, max_iter=60):
    """Gummel phi-step: nonlinear Poisson at frozen quasi-Fermi levels.

    In a near-insulating layer the dielectric-relaxation mode is
    seconds-slow for the transient and produces the enormous phi-mediated
    Jacobian coupling that defeats the coupled Newton (measured: singular
    Jacobians / ridge fallback in the Nd_ETL=1e10 regime). This step does
    that relaxation analytically: solve Poisson with the Boltzmann carrier
    response n(phi) = n_k*exp((phi-phi_k)/V_T), p(phi) = n_k*exp(-...)
    — quasi-Fermi levels exactly preserved — then update the densities to
    the converged phi. The phi-Newton is unconditionally well-posed (the
    operator and the response term are both negative-definite). Ions and
    boundary phi stay fixed; boundary densities are untouched (delta-phi
    is zero at the Dirichlet phi BCs).
    """
    from scipy.linalg import solve_banded

    N = len(x)
    n = np.maximum(y[:N].astype(float), 0.0)
    p = np.maximum(y[N: 2 * N].astype(float), 0.0)
    P = y[2 * N: 3 * N]
    P_neg = y[3 * N: 4 * N] if getattr(mat, "has_dual_ions", False) else None
    P_neg0 = mat.P_ion0_neg if P_neg is not None else None
    fac = mat.poisson_factor
    V_T = mat.V_T_device
    rho0 = _charge_density(p, n, P, mat.P_ion0, mat.N_A, mat.N_D,
                           P_neg=P_neg, P_neg0=P_neg0)
    phi_k = solve_poisson_prefactored(fac, rho0, 0.0, mat.V_bi_bc - V_app)
    rho_static = rho0 + Q * (n - p)          # phi-independent part
    C, h_cell = fac.C, fac.h_cell

    phi = phi_k.copy()
    for _ in range(max_iter):
        dlt = np.clip((phi - phi_k) / V_T, -60.0, 60.0)
        n_phi = n * np.exp(dlt)
        p_phi = p * np.exp(-dlt)
        rho = rho_static + Q * (p_phi - n_phi)
        F = (C[:-1] * (phi[:-2] - phi[1:-1])
             + C[1:] * (phi[2:] - phi[1:-1])
             + rho[1:-1] * h_cell)
        ab = np.zeros((3, N - 2))
        ab[0, 1:] = C[1:-1]                              # super
        ab[1, :] = -(C[:-1] + C[1:]) - Q * (
            n_phi[1:-1] + p_phi[1:-1]) / V_T * h_cell    # main
        ab[2, :-1] = C[1:-1]                             # sub
        dphi = solve_banded((1, 1), ab, -F)
        dphi = np.clip(dphi, -0.5, 0.5)
        phi[1:-1] += dphi
        if float(np.max(np.abs(dphi))) < tol_phi:
            break
    dlt = np.clip((phi - phi_k) / V_T, -60.0, 60.0)
    out = y.copy()
    # Update only strictly-positive densities: a transient-overshoot
    # negative (known Radau artifact at heterojunction nodes) must pass
    # through untouched — flooring it to a hard zero makes the RHS
    # evaluate catastrophically (measured res 2e-3 -> 5.5e+02), while the
    # tiny negative itself is tolerated by assemble_rhs.
    n_raw = y[:N]
    p_raw = y[N: 2 * N]
    out[:N] = np.where(n_raw > 0.0, n_raw * np.exp(dlt), n_raw)
    out[N: 2 * N] = np.where(p_raw > 0.0, p_raw * np.exp(-dlt), p_raw)
    return out


def _transient_point_fallback(x, stack, mat, V_app, y_seed, *,
                              illuminated=True,
                              t_settles=(6.4e-3, 2.5e-2, 1.0e-1),
                              res_guard=1.0):
    """Compute one steady-state point by transient settling (certified).

    Used where Newton cannot finish: near V* the interface-switch region
    makes stall-accepted Newton states poison subsequent warm starts,
    while a ~6 ms Radau settle from the walked continuation ladder
    reaches the true state (measured residual ~1.5e-2 on the peak-density
    scale at the V*~0.858 wall — better than the stall-accepted states).
    The result is certified against ``res_guard`` (rejects the chattering
    attractor, residual ~14) and raises on failure — no silent fallback.
    """
    N = len(x)
    y = _ensure_iface_block(y_seed, mat)
    res = float("inf")
    # escalate: near V_oc the slowest carrier mode needs 10-100 ms
    for t_settle in t_settles:
        sol = run_transient(
            x, y, (0.0, t_settle), np.array([t_settle]), stack,
            illuminated=illuminated, V_app=V_app, mat=mat,
            max_step=t_settle / 8.0,
        )
        if not sol.success:
            raise SteadyStateError(
                f"transient point-fallback failed at V={V_app:.4f}")
        y = _qfl_poisson_relax(x, mat, sol.y[:, -1], V_app)
        dydt = assemble_rhs(0.0, y, x, stack, mat,
                            illuminated=illuminated, V_app=V_app)
        n_ref = float(np.max(y[: 2 * N]))
        f = np.abs(dydt[: 2 * N]) / n_ref
        f[[0, N - 1, N, 2 * N - 1]] = 0.0
        res = float(np.max(f))
        if res <= res_guard:
            return y, res
    raise SteadyStateError(
        f"transient point-fallback uncertified at V={V_app:.4f} "
        f"(residual {res:.3e} > guard {res_guard:.1e})")


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
    iface_states: bool = False,
    stop_after_voc: bool = False,
    progress=None,
) -> JVSweepSSResult:
    """Steady-state J-V: voltage continuation with warm starts.

    On a failed point the voltage step bisects (up to 4 levels) so the
    continuation path stays connected; a point that still fails raises.

    ``stop_after_voc`` (default False = legacy: sweep the full ``V_max``
    range): stop the continuation as soon as J crosses zero (V_oc is
    bracketed). Continuing past V_oc into deep forward injection
    (V >> V_oc) reaches non-convergent points whose certified transient
    fallback grinds for minutes — so when only the figures of merit are
    needed, set this True to keep the sweep fast and robust for any
    ``V_max``. All four metrics are fully determined by the 0->V_oc arc.
    """
    x = _grid_for(stack, N_grid)
    mat = dataclasses.replace(
        build_material_arrays(x, stack), te_softness=_TE_SOFTNESS)
    if iface_states:
        mat = _enable_iface_states(mat)
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
            y_new = res.y
        except SteadyStateError:
            if V - V_prev > 0.011:
                V_targets.insert(i, 0.5 * (V_prev + V))   # bisect the step
                continue
            # Newton cannot finish in the interface-switch region —
            # certified transient settle computes the point instead
            y_new, _ = _transient_point_fallback(
                x, stack, mat, V, y_prev if y_prev is not None
                else solve_equilibrium(x, stack), illuminated=illuminated)
        y_prev = y_new
        V_prev = V
        V_out.append(V)
        J_out.append(_compute_current_ss(x, y_new, stack, V, mat=mat))
        if progress is not None:
            progress("jv_ss", len(V_out), len(V_targets), f"V={V:.3f}")
        if (stop_after_voc and len(J_out) >= 2
                and J_out[-2] > 0.0 >= J_out[-1]):
            # J just crossed zero -> V_oc bracketed; the remaining
            # V > V_oc points are deep-injection and only grind the
            # transient fallback. The 0->V_oc arc fully determines the FOM.
            break
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
    iface_states: bool = False,
) -> float:
    """Direct V_oc: bisection on the steady-state J(V) zero crossing.

    Walks up from V_lo in continuation steps until J changes sign, then
    bisects. Raises ``SteadyStateError`` if J never crosses zero below
    ``V_hi``.
    """
    x = _grid_for(stack, N_grid)
    mat = dataclasses.replace(
        build_material_arrays(x, stack), te_softness=_TE_SOFTNESS)
    if iface_states:
        mat = _enable_iface_states(mat)

    y_cache: dict[float, np.ndarray] = {}

    def J_at(V, y_seed):
        try:
            res = solve_steady_state(x, stack, V, illuminated=True,
                                     mat=mat, y0=y_seed)
            y = res.y
        except SteadyStateError:
            y, _ = _transient_point_fallback(
                x, stack, mat, V,
                y_seed if y_seed is not None
                else solve_equilibrium(x, stack))
        y_cache[V] = y
        return _compute_current_ss(x, y, stack, V, mat=mat)

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
