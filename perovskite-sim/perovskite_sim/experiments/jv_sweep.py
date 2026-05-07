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
from perovskite_sim.models.current import CurrentComponents
from perovskite_sim.models.spatial import SpatialSnapshot
from perovskite_sim.constants import EPS_0, Q


@dataclass(frozen=True)
class JVMetrics:
    V_oc: float
    J_sc: float
    FF: float
    PCE: float
    voc_bracketed: bool = True
    """``True`` iff the J(V) curve crossed zero inside the sampled voltage
    range. ``False`` means V_max stopped short of V_oc — the returned V_oc /
    FF / PCE are sentinel zeros and the caller should warn the user to
    expand the sweep range. ``J_sc`` is still meaningful (interpolated at
    V=0) and is returned even when the bracket fails."""


@dataclass(frozen=True)
class JVCurrentDecomp:
    """Per-voltage-point current decomposition (contact-face values, A/m²)."""
    J_n: np.ndarray
    J_p: np.ndarray
    J_ion: np.ndarray
    J_disp: np.ndarray
    J_total: np.ndarray


@dataclass(frozen=True)
class JVResult:
    V_fwd: np.ndarray
    J_fwd: np.ndarray
    V_rev: np.ndarray
    J_rev: np.ndarray
    metrics_fwd: JVMetrics
    metrics_rev: JVMetrics
    hysteresis_index: float
    snapshots_fwd: tuple[SpatialSnapshot, ...] | None = None
    snapshots_rev: tuple[SpatialSnapshot, ...] | None = None
    decomp_fwd: JVCurrentDecomp | None = None
    decomp_rev: JVCurrentDecomp | None = None


def compute_metrics(
    V: np.ndarray,
    J: np.ndarray,
    *,
    assume_jsc_positive: bool = True,
) -> JVMetrics:
    """Compute V_oc, J_sc, FF, PCE from a J-V array (J in A/m²).

    Reports the metrics directly from the simulated J(V) samples — it does
    NOT clamp or smooth. The caller is responsible for providing a properly
    converged, physically monotone curve (use a fine V grid and a quasi-static
    sweep). V_oc is taken at the first positive→non-positive zero crossing,
    and P_mpp is the maximum of V·J over the operating quadrant 0 ≤ V ≤ V_oc.

    Sign convention. The 1D solver follows the IonMonger / DriftFusion
    convention where J(V=0) > 0 (the photocurrent flows out of the device
    and powers an external load). The 2D solver currently emits the
    opposite sign — J(V=0) < 0 — so 2D callers must pass
    ``assume_jsc_positive=False`` to flip J internally before extraction.
    The returned :class:`JVMetrics` is always reported in the
    "J_sc positive" convention, regardless of which sign the input used.

    Bracketing. If the supplied J(V) does not cross zero in the sampled
    range — i.e. V_max stopped short of V_oc — the returned metrics carry
    ``voc_bracketed=False`` and ``V_oc / FF / PCE`` are sentinel zeros.
    ``J_sc`` is still meaningful (interpolated at V=0) and is returned
    even when the bracket fails. Callers should surface the flag to the
    user as a "increase V_max" warning rather than reading 0 V as a
    physical V_oc.
    """
    V = np.asarray(V, dtype=float)
    J = np.asarray(J, dtype=float)
    if not assume_jsc_positive:
        J = -J
    order = np.argsort(V)
    V_s = V[order]
    J_s = J[order]

    J_sc = float(np.interp(0.0, V_s, J_s))
    signs = np.sign(J_s)
    crossings = np.where((signs[:-1] > 0) & (signs[1:] <= 0))[0]
    if len(crossings) == 0:
        return JVMetrics(
            V_oc=0.0, J_sc=J_sc, FF=0.0, PCE=0.0, voc_bracketed=False,
        )
    idx = int(crossings[0])
    dV = V_s[idx + 1] - V_s[idx]
    dJ = J_s[idx + 1] - J_s[idx]
    V_oc = float(V_s[idx] - J_s[idx] * dV / dJ) if dJ != 0.0 else float(V_s[idx])

    mask = (V_s >= 0.0) & (V_s <= V_oc)
    P_mpp = float(np.max(V_s[mask] * J_s[mask])) if mask.any() else 0.0
    FF = P_mpp / (V_oc * J_sc) if (V_oc * J_sc) > 0 else 0.0
    PCE = P_mpp / 1000.0
    return JVMetrics(
        V_oc=V_oc, J_sc=J_sc, FF=FF, PCE=PCE, voc_bracketed=True,
    )


def hysteresis_index(
    V_fwd: np.ndarray, J_fwd: np.ndarray,
    V_rev: np.ndarray, J_rev: np.ndarray,
) -> float:
    m_fwd = compute_metrics(V_fwd, J_fwd)
    m_rev = compute_metrics(V_rev, J_rev)
    if m_rev.PCE == 0:
        return 0.0
    return (m_rev.PCE - m_fwd.PCE) / m_rev.PCE


def _state_fields(
    x: np.ndarray,
    y_state: np.ndarray,
    stack: DeviceStack,
    V_bc: float,
    mat: MaterialArrays,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, StateVec]:
    """Unpack state vector, apply BCs, solve Poisson. Returns (n, p, phi, sv).

    With selective / Schottky contacts active (Phase 3.3) the Robin sides
    are left free — boundary densities come straight from the state vector
    — while the Dirichlet sides are still pinned so that post-processing
    matches what the solver saw.
    """
    N = len(x)
    sv = StateVec.unpack(y_state, N)
    n = sv.n.copy()
    p = sv.p.copy()
    if not mat.has_selective_contacts:
        n[0] = mat.n_L; n[-1] = mat.n_R
        p[0] = mat.p_L; p[-1] = mat.p_R
    else:
        if mat.S_n_L is None:
            n[0] = mat.n_L
        if mat.S_n_R is None:
            n[-1] = mat.n_R
        if mat.S_p_L is None:
            p[0] = mat.p_L
        if mat.S_p_R is None:
            p[-1] = mat.p_R
    rho = _charge_density(
        p, n, sv.P, mat.P_ion0, mat.N_A, mat.N_D,
        P_neg=sv.P_neg, P_neg0=mat.P_ion0_neg,
    )
    phi = solve_poisson_prefactored(
        mat.poisson_factor, rho, phi_left=0.0, phi_right=stack.V_bi - V_bc,
    )
    return n, p, phi, sv


def extract_spatial_snapshot(
    x: np.ndarray,
    y: np.ndarray,
    stack: DeviceStack,
    V_app: float,
    mat: MaterialArrays | None = None,
) -> SpatialSnapshot:
    """Extract spatial profiles from a state vector at a given voltage.

    Returns a SpatialSnapshot with all node/face quantities in SI units.
    """
    if mat is None:
        mat = build_material_arrays(x, stack)
    n, p, phi, sv = _state_fields(x, y, stack, V_app, mat)
    dx = np.diff(x)
    E = -(phi[1:] - phi[:-1]) / dx
    rho = _charge_density(
        p, n, sv.P, mat.P_ion0, mat.N_A, mat.N_D,
        P_neg=sv.P_neg, P_neg0=mat.P_ion0_neg,
    )
    return SpatialSnapshot(
        x=x.copy(), phi=phi, E=E, n=n, p=p, P=sv.P.copy(), rho=rho, V_app=V_app,
    )


def compute_current_components(
    x: np.ndarray,
    y: np.ndarray,
    stack: DeviceStack,
    V_app: float,
    y_prev: np.ndarray | None = None,
    dt: float | None = None,
    mat: MaterialArrays | None = None,
    V_app_prev: float | None = None,
) -> CurrentComponents:
    """Decompose the total current into electron, hole, ion, and displacement.

    All arrays have shape (N-1,). Sign convention: positive when the device
    delivers power (solar convention, consistent with _compute_current).
    """
    if mat is None:
        mat = build_material_arrays(x, stack)

    dx = np.diff(x)
    n, p, phi, sv = _state_fields(x, y, stack, V_app, mat)
    V_T_dev = mat.V_T_device

    # Electron and hole conduction currents (SG fluxes). When the device
    # enables field-dependent mobility, the terminal J must use the same
    # field-corrected face diffusivities that assemble_rhs saw — otherwise
    # the reported current would be inconsistent with the state the solver
    # integrated, which would break charge conservation at the contact.
    phi_n = phi + mat.chi
    phi_p = phi + mat.chi + mat.Eg
    xi_n = (phi_n[1:] - phi_n[:-1]) / V_T_dev
    xi_p = (phi_p[1:] - phi_p[:-1]) / V_T_dev
    B_pos_n = bernoulli(xi_n); B_neg_n = bernoulli(-xi_n)
    B_pos_p = bernoulli(xi_p); B_neg_p = bernoulli(-xi_p)
    if mat.has_field_mobility:
        from perovskite_sim.physics.field_mobility import apply_field_mobility
        E_face = -(phi[1:] - phi[:-1]) / dx
        mu_n_face_base = mat.D_n_face / V_T_dev
        mu_p_face_base = mat.D_p_face / V_T_dev
        D_n_face_eff = apply_field_mobility(
            mu_n_face_base, E_face,
            mat.v_sat_n_face, mat.ct_beta_n_face, mat.pf_gamma_n_face,
        ) * V_T_dev
        D_p_face_eff = apply_field_mobility(
            mu_p_face_base, E_face,
            mat.v_sat_p_face, mat.ct_beta_p_face, mat.pf_gamma_p_face,
        ) * V_T_dev
    else:
        D_n_face_eff = mat.D_n_face
        D_p_face_eff = mat.D_p_face
    J_n = Q * D_n_face_eff / dx * (B_pos_n * n[1:] - B_neg_n * n[:-1])
    J_p = Q * D_p_face_eff / dx * (B_pos_p * p[:-1] - B_neg_p * p[1:])

    # Ion current: Q * F_ion at each face (positive species)
    xi_ion = (phi[1:] - phi[:-1]) / V_T_dev
    D_ion_face = np.broadcast_to(
        np.asarray(mat.D_ion_face, dtype=float), dx.shape,
    )
    P_lim_face = np.broadcast_to(
        np.asarray(mat.P_lim_face, dtype=float), dx.shape,
    )
    P_avg = 0.5 * (sv.P[:-1] + sv.P[1:])
    steric = 1.0 / np.maximum(
        1.0 - np.clip(P_avg / P_lim_face, 0.0, 0.999999), 1e-6,
    )
    D_eff = D_ion_face * steric
    F_ion = D_eff / dx * (bernoulli(xi_ion) * sv.P[:-1] - bernoulli(-xi_ion) * sv.P[1:])
    J_ion = Q * F_ion

    # Negative ion species contribution (reversed drift)
    if mat.has_dual_ions and sv.P_neg is not None:
        D_neg_face = np.broadcast_to(
            np.asarray(mat.D_ion_neg_face, dtype=float), dx.shape,
        )
        P_lim_neg_face = np.broadcast_to(
            np.asarray(mat.P_lim_neg_face, dtype=float), dx.shape,
        )
        P_neg_avg = 0.5 * (sv.P_neg[:-1] + sv.P_neg[1:])
        steric_neg = 1.0 / np.maximum(
            1.0 - np.clip(P_neg_avg / P_lim_neg_face, 0.0, 0.999999), 1e-6,
        )
        xi_neg = -(phi[1:] - phi[:-1]) / V_T_dev  # reversed sign for negative charge
        D_eff_neg = D_neg_face * steric_neg
        F_neg = D_eff_neg / dx * (
            bernoulli(xi_neg) * sv.P_neg[:-1] - bernoulli(-xi_neg) * sv.P_neg[1:]
        )
        J_ion = J_ion - Q * F_neg  # negative charge: subtract

    # Displacement current
    J_disp = np.zeros_like(J_n)
    if y_prev is not None and dt is not None and dt > 0.0:
        V_prev_bc = V_app_prev if V_app_prev is not None else V_app
        _, _, phi_prev, _ = _state_fields(x, y_prev, stack, V_prev_bc, mat)
        eps_face = _harmonic_face_average(mat.eps_r)
        E_prev = -(phi_prev[1:] - phi_prev[:-1]) / dx
        E_now = -(phi[1:] - phi[:-1]) / dx
        J_disp = EPS_0 * eps_face * (E_now - E_prev) / dt

    J_total = J_n + J_p + J_ion + J_disp
    return CurrentComponents(
        J_n=-J_n, J_p=-J_p, J_ion=-J_ion, J_disp=-J_disp, J_total=-J_total,
    )


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
    """Return total current density at every face (N-1,), solar sign convention.

    Thin wrapper around compute_current_components for backward compatibility.
    """
    return compute_current_components(
        x, y, stack, V_app, y_prev=y_prev, dt=dt, mat=mat, V_app_prev=V_app_prev,
    ).J_total


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


# Upper bound on RHS evaluations per run_transient call inside a JV sweep.
# Calibration: well-conditioned sub-intervals at N_grid=60 complete in a few
# hundred nfev; degenerate calls (reverse sweep reheating from a high-V_app
# carrier-injection state on ionmonger_benchmark) spin indefinitely without
# a bound. Well-conditioned TMM presets at moderate forward bias can legit
# consume ~30-40k nfev on a single sub-interval when the corrected generation
# profile drives high carrier densities, so the cap must stay well above
# that or the bisection fallback wastes budget on already-good calls.
# 100k aborts long before a wall-time user would notice (~10-15 s) and
# leaves headroom for the bisection fallback to retry on halved intervals.
_JV_RADAU_MAX_NFEV = 100_000


def _bake_radiative_reabsorption_step(
    y: np.ndarray, x: np.ndarray, mat: MaterialArrays, illuminated: bool,
) -> MaterialArrays:
    """Freeze the Phase 3.1b G_rad source for one ``_integrate_step`` call.

    Phase 3.1b's per-RHS hook recomputes ``R_tot = ∫ B·n·p dx`` inside every
    Radau Newton iteration, which couples every absorber node to every other
    through a non-local integral. At low forward bias (V ≈ 0.21 V on TMM
    presets) the diode-injection knee makes ``d(n·p)/dV`` steep and the
    Newton iteration cannot contract on the resulting low-rank dense block —
    bisection-and-retry runs out and ``_integrate_step`` raises.

    Fix: evaluate ``R_tot`` once at the entry state ``y`` of each voltage
    step, fold ``G_rad`` into a step-local ``G_optical`` copy, and clear
    ``has_radiative_reabsorption`` on the returned ``mat``. Inside the call,
    the SG flux sees a static G and Newton converges. Across voltage steps
    the warm-start chain refreshes ``R_tot`` from the freshly-settled state,
    so the only error is bounded by how much ``n·p`` drifts inside one
    settle interval — sub-percent for the typical ``v_rate=1 V/s`` sweep,
    well below the ~5 mV equivalence window the 3.1b regression tests use
    for V_oc parity with Phase 3.1.

    Beer-Lambert / non-TMM stacks have ``has_radiative_reabsorption=False``
    and skip this path entirely (returned mat is the original).
    """
    if not (mat.has_radiative_reabsorption and illuminated and mat.absorber_masks):
        return mat
    sv = StateVec.unpack(y, len(x))
    G_base = mat.G_optical if mat.G_optical is not None else np.zeros_like(x)
    G_with_rad = G_base.copy()
    for mask, P_esc, thickness in zip(
        mat.absorber_masks, mat.absorber_p_esc, mat.absorber_thicknesses
    ):
        if thickness <= 0.0 or P_esc >= 1.0:
            continue
        emission = mat.B_rad[mask] * sv.n[mask] * sv.p[mask]
        x_abs = x[mask]
        if x_abs.size < 2:
            continue
        R_tot = float(np.trapezoid(emission, x_abs))
        if R_tot <= 0.0:
            continue
        G_with_rad[mask] = G_with_rad[mask] + R_tot * (1.0 - P_esc) / thickness
    return dataclasses.replace(
        mat,
        G_optical=G_with_rad,
        has_radiative_reabsorption=False,
        absorber_masks=(),
        absorber_p_esc=(),
        absorber_thicknesses=(),
    )


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
    max_bisect: int = 6,
    illuminated: bool = True,
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

    Phase 3.1b fallback: when the device has ``has_radiative_reabsorption``
    on, the per-RHS ``G_rad`` source can prevent the Radau Newton iteration
    from contracting on TMM presets at the diode-injection knee
    (V ≈ 0.21 V — see saved memory `project_tmm_jv_regression_021.md`). If
    the standard call fails, retry once with ``G_rad`` frozen at the entry
    state — the warm-started chain refreshes it on the next voltage step,
    so the lag stays sub-percent on the typical ``v_rate=1 V/s`` sweep.
    Steps where the per-RHS hook converges (the vast majority) keep the
    fully self-consistent semantics; only the pathological steps fall back.
    """
    dt = t_hi - t_lo
    sol = run_transient(
        x, y, (t_lo, t_hi), np.array([t_hi]),
        stack, illuminated=illuminated, V_app=V_app, rtol=rtol, atol=atol,
        max_step=dt / 20.0 if dt > 0.0 else np.inf,
        mat=mat,
        max_nfev=_JV_RADAU_MAX_NFEV,
    )
    if sol.success:
        return sol.y[:, -1]
    if mat.has_radiative_reabsorption and illuminated:
        mat_step = _bake_radiative_reabsorption_step(y, x, mat, illuminated)
        sol = run_transient(
            x, y, (t_lo, t_hi), np.array([t_hi]),
            stack, illuminated=illuminated, V_app=V_app, rtol=rtol, atol=atol,
            max_step=dt / 20.0 if dt > 0.0 else np.inf,
            mat=mat_step,
            max_nfev=_JV_RADAU_MAX_NFEV,
        )
        if sol.success:
            return sol.y[:, -1]
    if max_bisect == 0:
        raise RuntimeError(
            f"JV sweep: coupled solver failed to converge on [{t_lo:.3e},{t_hi:.3e}] "
            f"at V_app={V_app:.4f} V after bisection"
        )
    t_mid = 0.5 * (t_lo + t_hi)
    y_mid = _integrate_step(x, y, stack, mat, V_app, t_lo, t_mid, rtol, atol,
                             max_bisect - 1, illuminated)
    return _integrate_step(x, y_mid, stack, mat, V_app, t_mid, t_hi, rtol, atol,
                            max_bisect - 1, illuminated)


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


def _default_V_max(stack: DeviceStack) -> float:
    """Default upper voltage for a J-V sweep when the caller passes V_max=None.

    Rationale
    ---------
    V_oc on heterostacks is bounded above by the band-offset-aware built-in
    potential V_bi_eff (``stack.compute_V_bi()``), which can exceed the manual
    ``stack.V_bi`` field configured in legacy YAMLs. If we opened the sweep
    only to the manual V_bi, forward sweeps on high-V_oc stacks (MAPbI3 etc.)
    would never cross J = 0 and ``compute_metrics`` would return V_oc = V_max.

    Formula:
        V_upper = max(V_bi_eff * 1.3, 1.4)

    The 1.3 headroom captures the minority-quasi-Fermi-level rise beyond V_bi
    under strong illumination; the 1.4 V floor is a backstop for legacy configs
    where chi/Eg are not set (so compute_V_bi falls back to the manual V_bi,
    which for a MAPbI3-like stack can be ~1.05 V — 1.3× that is only 1.37 V,
    uncomfortably close to the observed 1.05-1.15 V V_oc range).

    This is the single source of truth for the default V_max and is unit-tested
    directly so the formula can be audited without running a full sweep.
    """
    V_bi_eff = stack.compute_V_bi()
    return max(V_bi_eff * 1.3, 1.4)


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
    illuminated: bool = True,
    save_snapshots: bool = False,
    decompose_currents: bool = False,
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

    illuminated : when False, run a dark J-V (G=0 everywhere). The initial
      state is dark equilibrium instead of illuminated steady-state. Cannot
      be combined with fixed_generation.
    """
    if N_grid < 3:
        raise ValueError(f"N_grid must be >= 3, got {N_grid}")
    if n_points < 2:
        raise ValueError(f"n_points must be >= 2, got {n_points}")
    if v_rate <= 0:
        raise ValueError(f"v_rate must be positive, got {v_rate}")
    if not illuminated and fixed_generation is not None:
        raise ValueError("Cannot combine illuminated=False with fixed_generation")
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

    # Start from the appropriate equilibrium state:
    # - Dark mode: use dark equilibrium directly (no illumination settle)
    # - Fixed generation: inline the illuminated-SS logic with overridden mat
    # - Default: use the standard illuminated steady-state solver
    if not illuminated:
        y_eq = solve_equilibrium(x, stack)
    elif fixed_generation is not None:
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

        Returns (V_arr, J_arr, y_final, snapshots, decomp) so sweeps can be
        chained. snapshots and decomp are populated only when the corresponding
        flags (save_snapshots, decompose_currents) are True.
        """
        V_arr = np.linspace(V_start, V_end, n_points)
        dt = abs(V_end - V_start) / (v_rate * (n_points - 1))
        t_points = np.arange(n_points) * dt
        J_arr = np.zeros(n_points)
        snaps: list[SpatialSnapshot] = []
        d_Jn: list[float] = []
        d_Jp: list[float] = []
        d_Jion: list[float] = []
        d_Jdisp: list[float] = []
        d_Jtot: list[float] = []
        y = y_init.copy()
        V_prev = float(V_arr[0])
        for k, V_k in enumerate(V_arr):
            y_prev = y.copy()
            t_lo = t_points[k]
            t_hi = t_lo + dt
            y = _integrate_step(x, y, stack, mat, V_k, t_lo, t_hi, rtol, atol,
                               illuminated=illuminated)
            J_arr[k] = _compute_current(x, y, stack, V_k, y_prev=y_prev, dt=dt,
                                         mat=mat, V_app_prev=V_prev)
            if save_snapshots:
                snaps.append(extract_spatial_snapshot(x, y, stack, float(V_k), mat=mat))
            if decompose_currents:
                cc = compute_current_components(
                    x, y, stack, float(V_k),
                    y_prev=y_prev, dt=dt, mat=mat, V_app_prev=V_prev,
                )
                d_Jn.append(float(cc.J_n[0]))
                d_Jp.append(float(cc.J_p[0]))
                d_Jion.append(float(cc.J_ion[0]))
                d_Jdisp.append(float(cc.J_disp[0]))
                d_Jtot.append(float(cc.J_total[0]))
            V_prev = float(V_k)
            if progress is not None:
                progress(stage, k + 1, n_points, "")
        decomp = None
        if decompose_currents:
            decomp = JVCurrentDecomp(
                J_n=np.array(d_Jn), J_p=np.array(d_Jp),
                J_ion=np.array(d_Jion), J_disp=np.array(d_Jdisp),
                J_total=np.array(d_Jtot),
            )
        return V_arr, J_arr, y, snaps, decomp

    V_upper = _default_V_max(stack) if V_max is None else V_max
    # Forward sweep: dark equilibrium → short circuit → open circuit
    V_fwd, J_fwd, y_oc, snaps_fwd, decomp_fwd = _sweep(0.0, V_upper, y_eq, "jv_forward")
    # Reverse sweep: continue from light-soaked OC state → short circuit
    V_rev, J_rev, _, snaps_rev, decomp_rev = _sweep(V_upper, 0.0, y_oc, "jv_reverse")

    m_fwd = compute_metrics(V_fwd, J_fwd)
    m_rev = compute_metrics(V_rev[::-1], J_rev[::-1])
    HI = hysteresis_index(V_fwd, J_fwd, V_rev[::-1], J_rev[::-1])

    return JVResult(
        V_fwd=V_fwd, J_fwd=J_fwd, V_rev=V_rev, J_rev=J_rev,
        metrics_fwd=m_fwd, metrics_rev=m_rev, hysteresis_index=HI,
        snapshots_fwd=tuple(snaps_fwd) if save_snapshots else None,
        snapshots_rev=tuple(snaps_rev) if save_snapshots else None,
        decomp_fwd=decomp_fwd,
        decomp_rev=decomp_rev,
    )
