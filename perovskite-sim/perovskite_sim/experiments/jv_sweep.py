from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from perovskite_sim.discretization.fe_operators import bernoulli
from perovskite_sim.discretization.grid import multilayer_grid, Layer
from perovskite_sim.physics.poisson import solve_poisson_prefactored
from perovskite_sim.solver.illuminated_ss import solve_illuminated_ss
from perovskite_sim.solver.mol import (
    StateVec, run_transient,
    MaterialArrays, build_material_arrays,
    _charge_density,
    _harmonic_face_average,
)
from perovskite_sim.models.device import DeviceStack
from perovskite_sim.constants import EPS_0, Q, V_T


@dataclass(frozen=True)
class JVMetrics:
    V_oc: float
    J_sc: float
    FF: float
    PCE: float


@dataclass(frozen=True)
class JVResult:
    V_fwd: np.ndarray
    J_fwd: np.ndarray
    V_rev: np.ndarray
    J_rev: np.ndarray
    metrics_fwd: JVMetrics
    metrics_rev: JVMetrics
    hysteresis_index: float


def compute_metrics(V: np.ndarray, J: np.ndarray) -> JVMetrics:
    """Compute V_oc, J_sc, FF, PCE from a J-V array (J in A/m²).

    Reports the metrics directly from the simulated J(V) samples — it does
    NOT clamp or smooth. The caller is responsible for providing a properly
    converged, physically monotone curve (use a fine V grid and a quasi-static
    sweep). V_oc is taken at the first positive→non-positive zero crossing,
    and P_mpp is the maximum of V·J over the operating quadrant 0 ≤ V ≤ V_oc.
    """
    V = np.asarray(V, dtype=float)
    J = np.asarray(J, dtype=float)
    order = np.argsort(V)
    V_s = V[order]
    J_s = J[order]

    J_sc = float(np.interp(0.0, V_s, J_s))
    signs = np.sign(J_s)
    crossings = np.where((signs[:-1] > 0) & (signs[1:] <= 0))[0]
    if len(crossings) == 0:
        return JVMetrics(V_oc=0.0, J_sc=J_sc, FF=0.0, PCE=0.0)
    idx = int(crossings[0])
    dV = V_s[idx + 1] - V_s[idx]
    dJ = J_s[idx + 1] - J_s[idx]
    V_oc = float(V_s[idx] - J_s[idx] * dV / dJ) if dJ != 0.0 else float(V_s[idx])

    mask = (V_s >= 0.0) & (V_s <= V_oc)
    P_mpp = float(np.max(V_s[mask] * J_s[mask])) if mask.any() else 0.0
    FF = P_mpp / (V_oc * J_sc) if (V_oc * J_sc) > 0 else 0.0
    PCE = P_mpp / 1000.0
    return JVMetrics(V_oc=V_oc, J_sc=J_sc, FF=FF, PCE=PCE)


def hysteresis_index(
    V_fwd: np.ndarray, J_fwd: np.ndarray,
    V_rev: np.ndarray, J_rev: np.ndarray,
) -> float:
    m_fwd = compute_metrics(V_fwd, J_fwd)
    m_rev = compute_metrics(V_rev, J_rev)
    if m_rev.PCE == 0:
        return 0.0
    return (m_rev.PCE - m_fwd.PCE) / m_rev.PCE


def _compute_current(
    x: np.ndarray,
    y: np.ndarray,
    stack: DeviceStack,
    V_app: float,
    y_prev: np.ndarray | None = None,
    dt: float | None = None,
    mat: MaterialArrays | None = None,
) -> float:
    """Extract terminal current density J [A/m²] at the contact-adjacent face.

    The terminal current combines carrier conduction and, when a previous state
    is available, the displacement current over the last time step.

    Convention: J > 0 when the device delivers power (J_sc > 0 at V=0).
    """
    if mat is None:
        mat = build_material_arrays(x, stack)

    def state_fields(y_state: np.ndarray):
        N = len(x)
        sv = StateVec.unpack(y_state, N)
        n = sv.n.copy(); n[0] = mat.n_L; n[-1] = mat.n_R
        p = sv.p.copy(); p[0] = mat.p_L; p[-1] = mat.p_R
        rho = _charge_density(p, n, sv.P, mat.P_ion0, mat.N_A, mat.N_D)
        phi = solve_poisson_prefactored(
            mat.poisson_factor, rho, phi_left=0.0, phi_right=stack.V_bi - V_app,
        )
        return n, p, phi

    dx = np.diff(x)
    n, p, phi = state_fields(y)

    D_n_face = mat.D_n_face   # (N-1,)
    D_p_face = mat.D_p_face   # (N-1,)
    chi = mat.chi             # (N,)
    Eg  = mat.Eg              # (N,)

    # Band-corrected potentials — consistent with continuity.py so that
    # the extracted current matches the internal SG flux exactly.
    phi_n = phi + chi
    phi_p = phi + chi + Eg

    xi_n = (phi_n[1:] - phi_n[:-1]) / V_T
    xi_p = (phi_p[1:] - phi_p[:-1]) / V_T
    B_pos_n = bernoulli(xi_n); B_neg_n = bernoulli(-xi_n)
    B_pos_p = bernoulli(xi_p); B_neg_p = bernoulli(-xi_p)

    J_n = Q * D_n_face / dx * (B_pos_n * n[1:] - B_neg_n * n[:-1])
    J_p = Q * D_p_face / dx * (B_pos_p * p[:-1] - B_neg_p * p[1:])
    J_total_internal = J_n[0] + J_p[0]

    if y_prev is not None and dt is not None and dt > 0.0:
        _, _, phi_prev = state_fields(y_prev)
        eps_face = _harmonic_face_average(mat.eps_r)
        E_prev = -(phi_prev[1:] - phi_prev[:-1]) / dx
        E_now = -(phi[1:] - phi[:-1]) / dx
        J_disp = EPS_0 * eps_face * (E_now - E_prev) / dt
        J_total_internal += J_disp[0]

    # Negate for external
    # circuit convention: J_sc > 0.
    return -float(J_total_internal)


def _integrate_step(
    x: np.ndarray,
    y: np.ndarray,
    stack: DeviceStack,
    mat: MaterialArrays,
    V_app: float,
    t_lo: float,
    t_hi: float,
    rtol: float,
    atol: float,
    max_bisect: int = 4,
) -> np.ndarray:
    """Advance the coupled MOL state from t_lo to t_hi at fixed V_app.

    Radau is adaptive but its error estimator can underreport truncation
    error near V_bi, where the Jacobian becomes nearly singular (flat-band
    region). Without an explicit `max_step` cap it sometimes takes a single
    huge step across the whole [t_lo, t_hi] interval and accepts a state
    that landed on the wrong (carrier-injection) branch of the implicit
    system — producing isolated non-physical spikes in the J-V curve. We
    cap max_step to (t_hi - t_lo)/20 so the solver must resolve the
    transient with at least ~20 internal steps, which removes the spikes
    without materially slowing down well-conditioned regions.

    If the implicit solver fails to converge on the full step, subdivide
    the interval (halving up to max_bisect levels) and chain sub-steps.
    Raises RuntimeError if bisection is exhausted.
    """
    dt = t_hi - t_lo
    sol = run_transient(
        x, y, (t_lo, t_hi), np.array([t_hi]),
        stack, illuminated=True, V_app=V_app, rtol=rtol, atol=atol,
        max_step=dt / 20.0 if dt > 0.0 else np.inf,
        mat=mat,
    )
    if sol.success:
        return sol.y[:, -1]
    if max_bisect == 0:
        raise RuntimeError(
            f"JV sweep: coupled solver failed to converge on [{t_lo:.3e},{t_hi:.3e}] "
            f"at V_app={V_app:.4f} V after bisection"
        )
    t_mid = 0.5 * (t_lo + t_hi)
    y_mid = _integrate_step(x, y, stack, mat, V_app, t_lo, t_mid, rtol, atol, max_bisect - 1)
    return _integrate_step(x, y_mid, stack, mat, V_app, t_mid, t_hi, rtol, atol, max_bisect - 1)


def quasi_static_sweep(
    x: np.ndarray,
    y_init: np.ndarray,
    stack: DeviceStack,
    voltages: np.ndarray,
    sweep_time: float,
    rtol: float = 1e-4,
    atol: float = 1e-6,
    mat: MaterialArrays | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Quasi-static illuminated J-V from an existing state, carrying state forward.

    Integrates the full coupled MOL system at piecewise-constant V_app, stepping
    through `voltages` over a total wall time of `sweep_time` seconds. Each step
    advances the carrier+ion state to quasi-steady for the applied voltage, and
    the terminal current (conduction + displacement) is read from the same
    coupled solve. Used for snapshot J-V measurements inside degradation where
    we want ions effectively frozen (sweep_time ≪ τ_ion) but carriers relaxed.
    """
    n = len(voltages)
    if n < 2:
        raise ValueError(f"voltages must have at least 2 points, got {n}")
    if mat is None:
        mat = build_material_arrays(x, stack)
    dt = sweep_time / (n - 1)
    J_arr = np.zeros(n, dtype=float)
    y = y_init.copy()
    t = 0.0
    for k in range(n):
        V_k = float(voltages[k])
        y_prev = y.copy()
        y = _integrate_step(x, y, stack, mat, V_k, t, t + dt, rtol, atol)
        J_arr[k] = _compute_current(x, y, stack, V_k, y_prev=y_prev, dt=dt, mat=mat)
        t += dt
    return np.asarray(voltages, dtype=float), J_arr


def run_jv_sweep(
    stack: DeviceStack,
    N_grid: int = 100,
    v_rate: float = 0.1,      # V/s
    n_points: int = 50,
    rtol: float = 1e-4,
    atol: float = 1e-6,
    V_max: float | None = None,
) -> JVResult:
    """Run forward and reverse J-V sweeps.

    V_max : upper voltage limit. If None, defaults to stack.V_bi. With
      heterojunction band offsets, V_oc can exceed V_bi, so pass a larger
      value (e.g. 1.4 V for MAPbI3) to capture the full forward curve.
    """
    if N_grid < 3:
        raise ValueError(f"N_grid must be >= 3, got {N_grid}")
    if n_points < 2:
        raise ValueError(f"n_points must be >= 2, got {n_points}")
    if v_rate <= 0:
        raise ValueError(f"v_rate must be positive, got {v_rate}")
    for i, layer in enumerate(stack.layers):
        if layer.thickness <= 0:
            raise ValueError(
                f"layer {i} ({layer.name!r}) has non-positive thickness {layer.thickness}"
            )

    layers_grid = [Layer(l.thickness, N_grid // len(stack.layers)) for l in stack.layers]
    x = multilayer_grid(layers_grid)
    N = len(x)
    L = stack.total_thickness

    # Build the material cache once — shared across forward and reverse sweeps
    # and every RHS call inside them. See solver/mol.py:MaterialArrays.
    mat = build_material_arrays(x, stack)

    # Start from illuminated SC state: carriers equilibrated, ions at initial profile
    y_eq = solve_illuminated_ss(x, stack, V_app=0.0, rtol=rtol, atol=atol)

    def _sweep(V_start: float, V_end: float, y_init: np.ndarray):
        """Sweep from V_start to V_end, starting from carrier state y_init.

        Returns (V_arr, J_arr, y_final) so sweeps can be chained: the reverse
        sweep starts from the light-soaked state at the end of the forward sweep,
        matching how hysteresis is measured in the laboratory. If the coupled
        integrator fails on a step, we bisect that step up to a few times rather
        than silently recording a stale/divergent state.
        """
        V_arr = np.linspace(V_start, V_end, n_points)
        dt = abs(V_end - V_start) / (v_rate * (n_points - 1))
        t_points = np.arange(n_points) * dt
        J_arr = np.zeros(n_points)
        y = y_init.copy()
        for k, V_k in enumerate(V_arr):
            y_prev = y.copy()
            t_lo = t_points[k]
            t_hi = t_lo + dt
            y = _integrate_step(x, y, stack, mat, V_k, t_lo, t_hi, rtol, atol)
            J_arr[k] = _compute_current(x, y, stack, V_k, y_prev=y_prev, dt=dt, mat=mat)
        return V_arr, J_arr, y

    V_upper = stack.V_bi if V_max is None else V_max
    # Forward sweep: dark equilibrium → short circuit → open circuit
    V_fwd, J_fwd, y_oc = _sweep(0.0, V_upper, y_eq)
    # Reverse sweep: continue from light-soaked OC state → short circuit
    V_rev, J_rev, _ = _sweep(V_upper, 0.0, y_oc)

    m_fwd = compute_metrics(V_fwd, J_fwd)
    m_rev = compute_metrics(V_rev[::-1], J_rev[::-1])
    HI = hysteresis_index(V_fwd, J_fwd, V_rev[::-1], J_rev[::-1])

    return JVResult(V_fwd=V_fwd, J_fwd=J_fwd, V_rev=V_rev, J_rev=J_rev,
                    metrics_fwd=m_fwd, metrics_rev=m_rev, hysteresis_index=HI)
