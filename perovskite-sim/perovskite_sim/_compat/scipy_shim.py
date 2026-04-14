"""
Lightweight scipy shim: replaces scipy.sparse and scipy.integrate.solve_ivp
for the perovskite-sim drift-diffusion solver.

Uses Newton-based backward Euler with numerical Jacobian for stiff systems.
"""
import numpy as np
from dataclasses import dataclass
from typing import Optional, Callable
import warnings


# ---------- sparse replacement ----------
def diags(diagonals, offsets, shape=None, format=None):
    """Build a dense matrix from diagonal bands."""
    if shape is None:
        n = len(diagonals[0]) + abs(offsets[0])
        shape = (n, n)
    A = np.zeros(shape, dtype=float)
    for diag, off in zip(diagonals, offsets):
        d = np.asarray(diag)
        if off >= 0:
            for i in range(len(d)):
                A[i, i + off] = d[i]
        else:
            for i in range(len(d)):
                A[i - off, i] = d[i]
    return A


def spsolve(A, b):
    """Solve Ax = b using Thomas algorithm (tridiagonal) or LU."""
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    n = len(b)

    # Extract diagonals for Thomas algorithm
    a_sub = np.zeros(n)   # sub-diagonal
    a_main = np.zeros(n)  # main diagonal
    a_sup = np.zeros(n)   # super-diagonal

    is_tridiag = True
    for i in range(n):
        a_main[i] = A[i, i]
        if i > 0:
            a_sub[i] = A[i, i-1]
        if i < n-1:
            a_sup[i] = A[i, i+1]
        # Check off-tridiagonal elements
        for j in range(n):
            if abs(i - j) > 1 and abs(A[i, j]) > 1e-30:
                is_tridiag = False
                break
        if not is_tridiag:
            break

    if is_tridiag and n > 2:
        d = b.copy()
        bb = a_main.copy()
        cc = a_sup.copy()
        for i in range(1, n):
            if abs(bb[i-1]) < 1e-30:
                return np.linalg.solve(A, b)
            m = a_sub[i] / bb[i-1]
            bb[i] -= m * cc[i-1]
            d[i] -= m * d[i-1]
        x = np.zeros(n)
        if abs(bb[-1]) < 1e-30:
            return np.linalg.solve(A, b)
        x[-1] = d[-1] / bb[-1]
        for i in range(n-2, -1, -1):
            x[i] = (d[i] - cc[i] * x[i+1]) / bb[i]
        return x
    else:
        return np.linalg.solve(A, b)


# ---------- ODE solver ----------
@dataclass
class OdeResult:
    t: np.ndarray
    y: np.ndarray
    success: bool
    message: str = ""


def _compute_jacobian(fun, t, y, f0, n):
    """Compute Jacobian df/dy via finite differences.
    Uses relative perturbation for numerical stability with wide-range state variables.
    """
    J = np.empty((n, n))
    eps_base = 1e-8
    for j in range(n):
        # Relative perturbation: max(eps * |y_j|, eps)
        pert = eps_base * max(abs(y[j]), 1.0)
        y_pert = y.copy()
        y_pert[j] += pert
        f_pert = fun(t, y_pert)
        J[:, j] = (f_pert - f0) / pert
    return J


def solve_ivp(
    fun: Callable,
    t_span: tuple,
    y0: np.ndarray,
    method: str = "Radau",
    t_eval: Optional[np.ndarray] = None,
    rtol: float = 1e-3,
    atol: float = 1e-6,
    dense_output: bool = False,
    max_step: float = np.inf,
    **kwargs,
) -> OdeResult:
    """Newton-based backward Euler with adaptive step-size.

    Solves y_{n+1} = y_n + dt * f(t_{n+1}, y_{n+1}) using Newton iteration
    with numerically computed Jacobian. A-stable for stiff systems.
    """
    t0, tf = t_span
    y = np.asarray(y0, dtype=float).copy()
    n = len(y)

    # Initial step size
    f0 = fun(t0, y)
    if np.all(np.isfinite(f0)):
        # Estimate step from ||f0|| / ||y||
        sc = atol + rtol * np.abs(y)
        d0 = np.sqrt(np.mean((y / sc)**2))
        d1 = np.sqrt(np.mean((f0 / sc)**2))
        if d0 < 1e-5 or d1 < 1e-5:
            dt0 = 1e-6
        else:
            dt0 = 0.01 * d0 / d1
        dt = min(max_step, abs(tf - t0), dt0)
    else:
        dt = min(max_step, abs(tf - t0) * 1e-6)
    dt = max(dt, 1e-15)

    t = t0
    max_newton = 10
    safety = 0.8
    max_steps = 500000
    min_dt = 1e-18 * max(1.0, abs(tf))
    jac_age = 0
    jac_max_age = 5  # recompute Jacobian every N steps
    J_cached = None

    if t_eval is not None:
        t_eval = np.sort(t_eval)
        results = np.zeros((n, len(t_eval)))
        eval_idx = 0

    steps = 0
    failed = False
    consecutive_fails = 0

    while t < tf - 1e-14 * abs(tf):
        if steps > max_steps:
            failed = True
            break

        dt = min(dt, tf - t, max_step)
        if dt < min_dt:
            t = tf
            break

        t_new = t + dt

        # Newton iteration for: G(y_new) = y_new - y - dt * f(t_new, y_new) = 0
        # Jacobian of G: J_G = I - dt * J_f
        y_new = y + dt * fun(t, y)  # predictor: explicit Euler

        # Clip negative carrier/ion densities for stability
        y_new = np.maximum(y_new, 0.0)

        # Compute or reuse Jacobian
        if J_cached is None or jac_age >= jac_max_age:
            f_at_y = fun(t_new, y_new)
            if np.all(np.isfinite(f_at_y)):
                J_f = _compute_jacobian(fun, t_new, y_new, f_at_y, n)
                J_cached = J_f
                jac_age = 0
            else:
                # f is not finite at predictor; shrink step
                dt *= 0.1
                consecutive_fails += 1
                if consecutive_fails > 30:
                    failed = True
                    break
                continue
        else:
            J_f = J_cached

        J_G = np.eye(n) - dt * J_f

        converged = False
        for newton_it in range(max_newton):
            f_new = fun(t_new, y_new)
            if not np.all(np.isfinite(f_new)):
                break
            G = y_new - y - dt * f_new
            G_norm = np.sqrt(np.mean((G / (atol + rtol * np.abs(y_new)))**2))

            if G_norm < 0.1:
                converged = True
                break

            # Newton step: delta = -J_G^{-1} * G
            try:
                delta = np.linalg.solve(J_G, -G)
            except np.linalg.LinAlgError:
                break

            if not np.all(np.isfinite(delta)):
                break

            # Damped Newton: limit step size
            max_delta = np.max(np.abs(delta / (atol + rtol * np.abs(y_new))))
            if max_delta > 10.0:
                delta *= 10.0 / max_delta

            y_new = y_new + delta
            y_new = np.maximum(y_new, 0.0)  # clip negatives

        if not converged:
            dt *= 0.25
            jac_age = jac_max_age  # force Jacobian recomputation
            consecutive_fails += 1
            if dt < min_dt or consecutive_fails > 50:
                failed = True
                break
            continue

        # Error estimation via step doubling (2 half-steps)
        dt_half = 0.5 * dt
        y_half = y.copy()
        half_ok = True

        for half in range(2):
            t_h = t + half * dt_half
            t_h_new = t_h + dt_half
            f_h = fun(t_h, y_half)
            y_h_pred = y_half + dt_half * f_h
            y_h_pred = np.maximum(y_h_pred, 0.0)

            # Simplified Newton for half-steps (reuse cached Jacobian)
            J_G_half = np.eye(n) - dt_half * J_f
            conv_h = False
            y_h_new = y_h_pred
            for nit in range(max_newton):
                f_h_new = fun(t_h_new, y_h_new)
                if not np.all(np.isfinite(f_h_new)):
                    half_ok = False
                    break
                G_h = y_h_new - y_half - dt_half * f_h_new
                g_norm = np.sqrt(np.mean((G_h / (atol + rtol * np.abs(y_h_new)))**2))
                if g_norm < 0.1:
                    conv_h = True
                    break
                try:
                    dh = np.linalg.solve(J_G_half, -G_h)
                except np.linalg.LinAlgError:
                    half_ok = False
                    break
                if not np.all(np.isfinite(dh)):
                    half_ok = False
                    break
                max_dh = np.max(np.abs(dh / (atol + rtol * np.abs(y_h_new))))
                if max_dh > 10.0:
                    dh *= 10.0 / max_dh
                y_h_new = y_h_new + dh
                y_h_new = np.maximum(y_h_new, 0.0)
            if not conv_h or not half_ok:
                half_ok = False
                break
            y_half = y_h_new

        if half_ok:
            # Richardson error estimate
            err_vec = y_new - y_half
            sc = atol + rtol * np.maximum(np.abs(y), np.abs(y_half))
            err = np.sqrt(np.mean((err_vec / sc)**2))

            if err <= 1.0:
                # Accept with Richardson extrapolation
                y_richardson = 2.0 * y_half - y_new
                if np.all(np.isfinite(y_richardson)):
                    y = np.maximum(y_richardson, 0.0)
                else:
                    y = np.maximum(y_half, 0.0)
                t = t_new
                steps += 1
                jac_age += 1
                consecutive_fails = 0

                if t_eval is not None:
                    while eval_idx < len(t_eval) and t_eval[eval_idx] <= t + 1e-12 * abs(tf):
                        results[:, eval_idx] = y
                        eval_idx += 1

                # Grow step
                if err > 1e-10:
                    factor = min(safety * (1.0 / err)**0.5, 4.0)
                else:
                    factor = 4.0
                dt *= factor
            else:
                # Reject, shrink
                factor = max(safety * (1.0 / err)**0.5, 0.1)
                dt *= factor
                consecutive_fails += 1
        else:
            # Half-step failed; accept full step result
            y = np.maximum(y_new, 0.0)
            t = t_new
            steps += 1
            jac_age += 1
            consecutive_fails = 0

            if t_eval is not None:
                while eval_idx < len(t_eval) and t_eval[eval_idx] <= t + 1e-12 * abs(tf):
                    results[:, eval_idx] = y
                    eval_idx += 1
            # Don't grow step

    if t_eval is not None:
        while eval_idx < len(t_eval):
            results[:, eval_idx] = y
            eval_idx += 1
        return OdeResult(t=t_eval, y=results, success=not failed)
    else:
        return OdeResult(t=np.array([t0, t]), y=np.column_stack([y0, y]), success=not failed)
