from __future__ import annotations
import dataclasses
from dataclasses import dataclass
from typing import Callable, Optional
import numpy as np

ProgressCallback = Callable[[str, int, int, str], None]
"""Callable protocol: fn(stage, current, total, message) -> None."""
from perovskite_sim.discretization.fe_operators import bernoulli
from perovskite_sim.discretization.grid import multilayer_grid, Layer
from perovskite_sim.physics.poisson import solve_poisson_prefactored
from perovskite_sim.solver.illuminated_ss import solve_illuminated_ss
from perovskite_sim.solver.newton import solve_equilibrium
from perovskite_sim.solver.mol import (
    StateVec, run_transient,
    MaterialArrays, build_material_arrays,
    _charge_density,
    _harmonic_face_average,
)
from perovskite_sim.models.device import DeviceStack, electrical_layers
from perovskite_sim.constants import EPS_0, Q


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


def _total_current_faces(
    x: np.ndarray,
    y: np.ndarray,
    stack: DeviceStack,
    V_app: float,
    y_prev: np.ndarray | None = None,
    dt: float | None = None,
    mat: MaterialArrays | None = None,
    V_app_prev: float | None = None,
) -> np.ndarray:
    """Return total (conduction + displacement) current density at every face,
    in external/solar sign convention (positive when device delivers power).

    Shape is (N-1,), one value per mesh face. Callers that want a scalar
    terminal current (e.g. J-V sweep) should take face 0; callers that want
    an AC-sensitive reading (e.g. impedance) should average spatially — by
    the 1D Ramo–Shockley argument the small-signal total current is very
    nearly space-uniform, and the boundary faces are the noisiest.
    """
    if mat is None:
        mat = build_material_arrays(x, stack)

    def state_fields(y_state: np.ndarray, V_bc: float):
        N = len(x)
        sv = StateVec.unpack(y_state, N)
        n = sv.n.copy(); n[0] = mat.n_L; n[-1] = mat.n_R
        p = sv.p.copy(); p[0] = mat.p_L; p[-1] = mat.p_R
        rho = _charge_density(
            p, n, sv.P, mat.P_ion0, mat.N_A, mat.N_D,
            P_neg=sv.P_neg, P_neg0=mat.P_ion0_neg,
        )
        phi = solve_poisson_prefactored(
            mat.poisson_factor, rho, phi_left=0.0, phi_right=stack.V_bi - V_bc,
        )
        return n, p, phi

    dx = np.diff(x)
    n, p, phi = state_fields(y, V_app)
    D_n_face = mat.D_n_face
    D_p_face = mat.D_p_face
    chi = mat.chi
    Eg = mat.Eg
    phi_n = phi + chi
    phi_p = phi + chi + Eg
    V_T_dev = mat.V_T_device
    xi_n = (phi_n[1:] - phi_n[:-1]) / V_T_dev
    xi_p = (phi_p[1:] - phi_p[:-1]) / V_T_dev
    B_pos_n = bernoulli(xi_n); B_neg_n = bernoulli(-xi_n)
    B_pos_p = bernoulli(xi_p); B_neg_p = bernoulli(-xi_p)
    J_n = Q * D_n_face / dx * (B_pos_n * n[1:] - B_neg_n * n[:-1])
    J_p = Q * D_p_face / dx * (B_pos_p * p[:-1] - B_neg_p * p[1:])
    J_tot = J_n + J_p  # (N-1,)
    if y_prev is not None and dt is not None and dt > 0.0:
        V_prev_bc = V_app_prev if V_app_prev is not None else V_app
        _, _, phi_prev = state_fields(y_prev, V_prev_bc)
        eps_face = _harmonic_face_average(mat.eps_r)
        E_prev = -(phi_prev[1:] - phi_prev[:-1]) / dx
        E_now = -(phi[1:] - phi[:-1]) / dx
        J_tot = J_tot + EPS_0 * eps_face * (E_now - E_prev) / dt
    return -J_tot


def _compute_current(
    x: np.ndarray,
    y: np.ndarray,
    stack: DeviceStack,
    V_app: float,
    y_prev: np.ndarray | None = None,
    dt: float | None = None,
    mat: MaterialArrays | None = None,
    V_app_prev: float | None = None,
) -> float:
    """Extract terminal current density J [A/m²] at the contact-adjacent face.

    The terminal current combines carrier conduction and, when a previous state
    is available, the displacement current over the last time step. When the
    applied voltage changed between `y_prev` and `y`, pass `V_app_prev` so the
    Poisson solve for `y_prev` uses the right Dirichlet condition; otherwise
    the ∂V_boundary/∂t contribution to the displacement current is lost.

    Convention: J > 0 when the device delivers power (J_sc > 0 at V=0).
    """
    J_faces = _total_current_faces(
        x, y, stack, V_app, y_prev=y_prev, dt=dt, mat=mat, V_app_prev=V_app_prev,
    )
    return float(J_faces[0])


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
    V_prev = float(voltages[0])
    t = 0.0
    for k in range(n):
        V_k = float(voltages[k])
        y_prev = y.copy()
        y = _integrate_step(x, y, stack, mat, V_k, t, t + dt, rtol, atol)
        J_arr[k] = _compute_current(x, y, stack, V_k, y_prev=y_prev, dt=dt,
                                     mat=mat, V_app_prev=V_prev)
        V_prev = V_k
        t += dt
    return np.asarray(voltages, dtype=float), J_arr


def _grid_node_count(stack: DeviceStack, N_grid: int) -> int:
    """Return the number of electrical grid nodes run_jv_sweep will build.

    This is the single source of truth for the electrical-grid sizing formula.
    Both run_jv_sweep (internally) and tandem callers (to pre-size generation
    profiles) must use this helper so the two sites can never silently diverge.

    Formula derivation:
    - Only electrical layers (non-substrate) are gridded.
    - Each layer receives  n_per = N_grid // n_elec  intervals.
    - multilayer_grid deduplicates shared boundary points, giving
      N = 1 + n_elec * n_per  total nodes.
    """
    elec = electrical_layers(stack)
    n_elec = len(elec)
    n_per = N_grid // n_elec
    return 1 + n_elec * n_per


def run_jv_sweep(
    stack: DeviceStack,
    N_grid: int = 100,
    v_rate: float = 0.1,      # V/s
    n_points: int = 50,
    rtol: float = 1e-4,
    atol: float = 1e-6,
    V_max: float | None = None,
    progress: ProgressCallback | None = None,
    fixed_generation: np.ndarray | None = None,
) -> JVResult:
    """Run forward and reverse J-V sweeps.

    V_max : upper voltage limit. If None, defaults to max(V_bi_eff*1.3, 1.4). With
      heterojunction band offsets, V_oc can exceed V_bi, so pass a larger
      value (e.g. 1.4 V for MAPbI3) to capture the full forward curve.

    fixed_generation : optional pre-computed generation profile G(x) [m⁻³ s⁻¹].
      Must be a 1-D array of shape (N,) where N is the number of electrical-grid
      nodes (determined by N_grid and the number of electrical layers). When
      provided, this profile is used verbatim in place of Beer-Lambert or TMM
      optics for both the initial illuminated steady-state and every subsequent
      solver call. When None (default), the existing single-junction optics path
      is used unchanged.
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

    # Grid construction uses electrical layers only — substrate layers are
    # optical-only and have no drift-diffusion counterpart, so allocating
    # grid nodes inside them would desync MaterialArrays masks with the
    # solver state vector. TMM/optics paths still see the full stack.
    elec = electrical_layers(stack)
    layers_grid = [Layer(l.thickness, N_grid // len(elec)) for l in elec]
    x = multilayer_grid(layers_grid)
    N = _grid_node_count(stack, N_grid)
    assert N == len(x), "grid node count mismatch — _grid_node_count is out of sync"
    L = sum(l.thickness for l in elec)

    # Build the material cache once — shared across forward and reverse sweeps
    # and every RHS call inside them. See solver/mol.py:MaterialArrays.
    mat = build_material_arrays(x, stack)

    # Optional generation override: tandem callers inject a pre-computed G(x)
    # profile (e.g. from combined-TMM) instead of letting the sweep recompute
    # optics internally. Validation is done here so the error surfaces early
    # before any time-consuming ODE work begins.
    if fixed_generation is not None:
        expected_shape = (N,)
        if np.asarray(fixed_generation).shape != expected_shape:
            raise ValueError(
                f"fixed_generation shape {np.asarray(fixed_generation).shape} "
                f"!= expected {expected_shape} "
                f"(N={N} electrical-grid nodes for N_grid={N_grid} "
                f"and {len(elec)} electrical layers)"
            )
        mat = dataclasses.replace(
            mat, G_optical=np.asarray(fixed_generation, dtype=float).copy()
        )

    # Start from illuminated SC state: carriers equilibrated, ions at initial profile.
    # When fixed_generation is provided we bypass solve_illuminated_ss (which builds
    # its own MaterialArrays internally) and replicate the same logic inline,
    # threading our overridden mat so the initial state also sees zero generation.
    if fixed_generation is not None:
        from perovskite_sim.solver.mol import run_transient as _run_transient
        _t_settle = 1e-3
        y_dark = solve_equilibrium(x, stack)
        sol = _run_transient(
            x, y_dark, (0.0, _t_settle), np.array([_t_settle]),
            stack, illuminated=True, V_app=0.0, rtol=rtol, atol=atol,
            mat=mat,
        )
        y_eq = sol.y[:, -1] if sol.success else y_dark
    else:
        y_eq = solve_illuminated_ss(x, stack, V_app=0.0, rtol=rtol, atol=atol)

    def _sweep(V_start: float, V_end: float, y_init: np.ndarray, stage: str):
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
        V_prev = float(V_arr[0])
        for k, V_k in enumerate(V_arr):
            y_prev = y.copy()
            t_lo = t_points[k]
            t_hi = t_lo + dt
            y = _integrate_step(x, y, stack, mat, V_k, t_lo, t_hi, rtol, atol)
            J_arr[k] = _compute_current(x, y, stack, V_k, y_prev=y_prev, dt=dt,
                                         mat=mat, V_app_prev=V_prev)
            V_prev = float(V_k)
            if progress is not None:
                progress(stage, k + 1, n_points, "")
        return V_arr, J_arr, y

    V_bi_eff = stack.compute_V_bi()
    V_upper = max(V_bi_eff * 1.3, 1.4) if V_max is None else V_max
    # Forward sweep: dark equilibrium → short circuit → open circuit
    V_fwd, J_fwd, y_oc = _sweep(0.0, V_upper, y_eq, "jv_forward")
    # Reverse sweep: continue from light-soaked OC state → short circuit
    V_rev, J_rev, _ = _sweep(V_upper, 0.0, y_oc, "jv_reverse")

    m_fwd = compute_metrics(V_fwd, J_fwd)
    m_rev = compute_metrics(V_rev[::-1], J_rev[::-1])
    HI = hysteresis_index(V_fwd, J_fwd, V_rev[::-1], J_rev[::-1])

    return JVResult(V_fwd=V_fwd, J_fwd=J_fwd, V_rev=V_rev, J_rev=J_rev,
                    metrics_fwd=m_fwd, metrics_rev=m_rev, hysteresis_index=HI)
