from __future__ import annotations
import dataclasses
import warnings
from dataclasses import dataclass
from typing import Callable, Optional
import numpy as np

from perovskite_sim._compat.numpy_compat import trapezoid

ProgressCallback = Callable[[str, int, int, str], None]
"""Callable protocol: fn(stage, current, total, message) -> None."""
from perovskite_sim.discretization.fe_operators import bernoulli
from perovskite_sim.discretization.grid import multilayer_grid, Layer
from perovskite_sim.physics.ion_migration import ion_face_flux
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
from perovskite_sim.physics.grading import has_grading_params
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
    """``True`` iff an open-circuit point was resolved inside the sampled
    voltage range, in which case V_oc / FF / PCE are physical. ``False``
    carries sentinel zeros for those three and has exactly two causes: the
    current never reached zero in the window (V_max stopped short), or the
    extracted V_oc exceeded the ``V_oc_max`` ceiling. ``J_sc`` is still
    meaningful (interpolated at V=0) and is returned in both cases. See
    ``compute_metrics`` for the full extraction contract — in particular
    that "reached zero" means reached the current-resolution floor, so a
    device whose photocurrent collapses before the diode turns on is
    ``True``, not ``False``."""


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


# --- V_oc extraction: terminal-current resolution floor --------------------
#
# ``_J_ZERO_FRACTION_OF_JSC`` is the fraction of |J_sc| below which the
# terminal current is treated as having REACHED zero, instead of waiting for
# a sign change.  Waiting for the sign change is what let a numerical-noise
# flip volts past the true open-circuit point be reported as V_oc on devices
# whose photocurrent collapses before the diode turns on (see the module
# docstring of ``tests/unit/experiments/test_voc_collapsed_current.py``).
#
# Provenance — metrology and diode physics, NOT any observed residual:
#   * Lower bound, from what a real J-V trace can resolve.  A 0.1 cm² lab cell
#     at J_sc ≈ 200-250 A/m² carries I_sc ≈ 2 mA.  A Keithley 2400/2450-class
#     SMU on its 10 mA range has an accuracy floor of ~0.5 µA + 0.05 % of
#     reading ≈ 1.5 µA ≈ 7e-4 of I_sc, and a class-AAA solar simulator's
#     temporal irradiance instability is ±0.5-1 %.  A current under ~1e-3 of
#     J_sc is therefore not a current any real measurement can distinguish
#     from zero, so a simulator must not either.
#   * Upper bound, from what the truncation costs at a GENUINE V_oc.  There
#     the curve is diode-limited, |dJ/dV| ≈ J_sc / (m·V_T), so stopping the
#     search at |J| = ε·J_sc displaces V_oc by at most ε·m·V_T — 52 µV for
#     ε=1e-3, m=2, V_T=25.85 mV.  Keeping that displacement an order of
#     magnitude under the 1 mV at which V_oc is quoted requires
#     ε ≪ 1e-3/(10·m·V_T) ≈ 2e-3.
# 1e-3 sits at the metrology floor and just under the cost bound.
#
# The cost bound holds only where the diode slope does.  On a COLLAPSED
# curve there is no such slope — the device this guard exists for drops
# 148.8 → 0.127 A/m² across one 37 mV sample (measured, scaps_mirror +
# Robin contacts, N_D_ETL = 1e18 m^-3) — so V_oc is then resolved only to
# the sample spacing and is reported at the bracket edge.  That is the same
# resolution the sign-change rule has; the difference is that this bracket
# is anchored at the collapse instead of at a noise flip 1 V further on.
_J_ZERO_FRACTION_OF_JSC = 1e-3


def thermodynamic_voc_ceiling(stack: DeviceStack) -> float | None:
    """Hard upper bound on any physical V_oc for ``stack``, in volts.

    The open-circuit voltage of a single-junction cell cannot exceed the
    smallest band gap the carriers must traverse: q·V_oc is the splitting of
    the quasi-Fermi levels, and that splitting is bounded by the band gap
    (the radiative / detailed-balance limit is well below it).  Returns
    ``min(Eg)/q`` over the ELECTRICAL layers — optical-only substrates are
    excluded, since they carry no carriers.

    Returns ``None`` when no electrical layer declares a band gap (the legacy
    ``chi = Eg = 0`` presets such as ``nip_MAPbI3``/``pin_MAPbI3``).  There is
    no information from which to build a ceiling in that case, so the caller
    must not apply one — a ``min()`` over all-zero gaps would reject every
    V_oc.

    Why ``min`` over the electrical layers and not the absorber's gap.  The
    bound that is actually a theorem is ``V_oc ≤ Eg_absorber/q`` (the
    splitting is generated in the absorber), and ``min`` is only equal to it
    while no transport layer is narrower than the absorber.  On every shipped
    preset it is — measured 2026-07-27 over all of ``configs/*.yaml`` +
    ``configs/twod/*.yaml``: the ``role: absorber`` layer has the narrowest
    electrical gap in every config that declares gaps at all (the closest
    call is ``cSi_homojunction``, where the n-emitter ties it at 1.12 eV).
    ``min`` is preferred because it needs no ``role`` tag, so a config that
    omits or mistags roles still gets a ceiling.  A future stack with a
    genuinely narrow-gap transport layer would make this ceiling TIGHTER
    than the thermodynamic bound and could refuse a valid V_oc — the
    equality is therefore pinned by
    ``tests/unit/experiments/test_voc_collapsed_current.py::
    test_ceiling_equals_the_absorber_gap_on_every_shipped_preset``, which is
    the signal to revisit this choice rather than to relax the test.

    Known plumbing gap (2026-07-27): only :func:`run_jv_sweep` passes this
    ceiling into :func:`compute_metrics`.  The other call sites that hold a
    ``DeviceStack`` — ``degradation._measure_snapshot_metrics``,
    ``steady_state.run_jv_sweep_ss``, ``voc_t``,
    ``twod.experiments.jv_sweep_2d`` and ``twod.experiments.voc_grain_sweep``
    — still call it with the default ``V_oc_max=None``, so they keep the
    resolution floor but not the ceiling.
    """
    gaps = [
        float(layer.params.Eg)
        for layer in electrical_layers(stack)
        if float(layer.params.Eg) > 0.0
    ]
    return min(gaps) if gaps else None


def compute_metrics(
    V: np.ndarray,
    J: np.ndarray,
    *,
    assume_jsc_positive: bool = True,
    P_in: float = 1000.0,
    V_oc_max: float | None = None,
) -> JVMetrics:
    """Compute V_oc, J_sc, FF, PCE from a J-V array (J in A/m²).

    Reports the metrics directly from the simulated J(V) samples — it does
    NOT clamp or smooth. The caller is responsible for providing a properly
    converged, physically monotone curve (use a fine V grid and a quasi-static
    sweep). P_mpp is the maximum of V·J over the operating quadrant
    0 ≤ V ≤ V_oc.

    Sign convention. The 1D solver follows the IonMonger / DriftFusion
    convention where J(V=0) > 0 (the photocurrent flows out of the device
    and powers an external load). The 2D solver currently emits the
    opposite sign — J(V=0) < 0 — so 2D callers must pass
    ``assume_jsc_positive=False`` to flip J internally before extraction.
    The returned :class:`JVMetrics` is always reported in the
    "J_sc positive" convention, regardless of which sign the input used.

    V_oc extraction
    ---------------
    Open circuit is the first voltage at which the terminal current REACHES
    zero — not the first voltage at which it changes sign.  The two coincide
    on a well-resolved curve, but they diverge badly on a device whose
    photocurrent collapses before the diode turns on: J then sits on a
    residual plateau whose SIGN is numerical noise, and the first sign flip
    can land volts past the true open-circuit point.  Past flat band the
    J-V curve is in the near-singular-Jacobian region the solver notes
    already document, so its sign there carries no information.  Measured
    on scaps_mirror + Robin contacts at N_D_ETL = 1e18 m^-3 (Eg = 1.53 eV,
    J_sc = 219 A/m²), forward branch at dV = 75 mV: J falls to 2.1e-2 A/m²
    at V = 1.425 and then plateaus at ~1.5e-3 (7e-6 of J_sc) all the way to
    2.700, with an isolated negative excursion at 2.475 — which the
    sign-change rule reports as V_oc = 2.4056 V, above the band gap.

    The bracket is therefore the first sample pair with
    ``J[i] > J_tol`` and ``J[i+1] <= J_tol``, where
    ``J_tol = 1e-3 · |J_sc|`` is the current-resolution floor documented on
    ``_J_ZERO_FRACTION_OF_JSC`` above.  V_oc is linearly interpolated to
    J = 0 inside that pair and clamped to it.  When ``J_sc = 0`` (dark
    sweeps) ``J_tol`` is zero and this reduces exactly to the sign-change
    rule.

    Cost on healthy curves: none measured.  On a curve with a genuine
    crossing no sample lands in ``(0, J_tol]`` before it, so the bracket,
    the interpolation and the clamp all reproduce the sign-change rule
    exactly.  Verified 2026-07-27 by extracting metrics TWICE from one
    sweep per config (N_grid=60, n_points=40, v_rate=0.5) — once with
    ``_J_ZERO_FRACTION_OF_JSC`` monkeypatched to 0.0 (which is the
    sign-change rule identically) and once as shipped: V_oc, J_sc, FF and
    PCE compared exactly equal as floats on ionmonger_benchmark,
    nip_MAPbI3, pin_MAPbI3, driftfusion_benchmark and scaps_mirror_v2,
    forward and reverse (10/10 branches).

    Cost on collapsed curves: V_oc is resolved only to the sample spacing,
    because the interpolation inside the collapse bracket is meaningless and
    the clamp returns its right edge.  On the fixture above at dV = 37 mV
    this reports 1.3756 V where the (noise-signed) crossing sits at 1.4373 V
    — 62 mV lower, and the current across that gap is ≤ 5.8e-4 of J_sc
    (1.3 µA on a 0.1 cm² cell), i.e. inside the measurement floor the
    threshold is derived from.  Both numbers are honest; neither is
    resolvable.  What the guard buys is that the answer stops depending on
    where the sweep happened to stop.

    ``V_oc_max`` is an optional hard ceiling in volts.  A V_oc above it is
    impossible regardless of the numerics, so it is refused rather than
    reported: the result carries ``voc_bracketed=False`` and sentinel zeros.
    ``jv_sweep`` callers get it from :func:`thermodynamic_voc_ceiling`
    (``min(Eg)/q`` over the electrical layers).  ``None`` (the default)
    disables the check, which is what a stack with no declared band gaps
    must do.

    Bracketing flag
    ---------------
    ``voc_bracketed`` means exactly one thing: **an open-circuit point was
    resolved inside the sampled window, and V_oc / FF / PCE below are
    physical.**  It is ``False``, with sentinel zeros for V_oc / FF / PCE,
    when the current never reached zero in the window (V_max stopped short)
    and when the extracted V_oc was refused by ``V_oc_max``.  It is ``True``
    for a collapsed-current device: a cell delivering less than the
    measurement resolution HAS reached open circuit, and reporting the
    voltage at which it got there is the honest answer — the alternative,
    calling it "unbracketed" and inviting the caller to widen V_max, walks
    further into the region whose sign is noise.  ``J_sc`` is interpolated
    at V=0 and stays meaningful in every case, including the two ``False``
    ones.  Callers should surface ``False`` as an "increase V_max" warning
    rather than reading 0 V as a physical V_oc.
    """
    V = np.asarray(V, dtype=float)
    J = np.asarray(J, dtype=float)
    if not assume_jsc_positive:
        J = -J
    order = np.argsort(V)
    V_s = V[order]
    J_s = J[order]

    J_sc = float(np.interp(0.0, V_s, J_s))
    # Current-resolution floor. Anchored on |J_sc| so it scales with the
    # device; zero for a dark sweep, where it degenerates to the sign rule.
    J_tol = _J_ZERO_FRACTION_OF_JSC * abs(J_sc)
    crossings = np.where((J_s[:-1] > J_tol) & (J_s[1:] <= J_tol))[0]
    if len(crossings) == 0:
        return JVMetrics(
            V_oc=0.0, J_sc=J_sc, FF=0.0, PCE=0.0, voc_bracketed=False,
        )
    idx = int(crossings[0])
    dV = V_s[idx + 1] - V_s[idx]
    dJ = J_s[idx + 1] - J_s[idx]
    V_oc = float(V_s[idx] - J_s[idx] * dV / dJ) if dJ != 0.0 else float(V_s[idx])
    # When J[idx+1] is a small POSITIVE residual rather than a sign flip the
    # interpolation extrapolates past the bracket, and the overshoot is
    # unbounded as J[idx] approaches J_tol from above. Clamp to the bracket.
    # No-op whenever J[idx+1] <= 0, i.e. for every genuine sign crossing.
    V_oc = min(max(V_oc, float(V_s[idx])), float(V_s[idx + 1]))

    if V_oc_max is not None and V_oc > V_oc_max:
        # Above the thermodynamic ceiling: impossible whatever the numerics
        # produced, so refuse it rather than report it.
        return JVMetrics(
            V_oc=0.0, J_sc=J_sc, FF=0.0, PCE=0.0, voc_bracketed=False,
        )

    mask = (V_s >= 0.0) & (V_s <= V_oc)
    P_mpp = float(np.max(V_s[mask] * J_s[mask])) if mask.any() else 0.0
    FF = P_mpp / (V_oc * J_sc) if (V_oc * J_sc) > 0 else 0.0
    # PCE is defined against the incident optical power density P_in
    # [W/m^2]. The default 1000 W/m^2 is the AM1.5G 1-sun convention;
    # callers sweeping illumination intensity or spectra must pass the
    # actual P_in or the reported efficiency is wrong (2026-07 review F18).
    PCE = P_mpp / P_in if P_in > 0.0 else 0.0
    return JVMetrics(
        V_oc=V_oc, J_sc=J_sc, FF=FF, PCE=PCE, voc_bracketed=True,
    )


def hysteresis_index(
    V_fwd: np.ndarray, J_fwd: np.ndarray,
    V_rev: np.ndarray, J_rev: np.ndarray,
    *,
    V_oc_max: float | None = None,
) -> float:
    """Forward/reverse PCE asymmetry. ``V_oc_max`` is forwarded to
    :func:`compute_metrics` on both branches (see its docstring); ``None``
    keeps the pre-existing no-ceiling behaviour."""
    m_fwd = compute_metrics(V_fwd, J_fwd, V_oc_max=V_oc_max)
    m_rev = compute_metrics(V_rev, J_rev, V_oc_max=V_oc_max)
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
        mat.poisson_factor, rho, phi_left=0.0, phi_right=mat.V_bi_bc - V_bc,
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
    # Delegate to the SAME face-flux helper the RHS integrates (2026-07
    # review): this block used to carry its own hard-coded copy of the
    # legacy whole-flux steric form and never consulted
    # ``ion_steric_diffusion_only``, so with that flag on the reported
    # terminal current was not the current the device carried.
    _shared = (
        mat.ion_steric_diffusion_only and mat.ion_steric_shared_site
        and mat.has_dual_ions and sv.P_neg is not None
    )
    F_ion = ion_face_flux(
        phi, sv.P, dx, D_ion_face, V_T_dev, P_lim_face,
        steric_diffusion_only=mat.ion_steric_diffusion_only,
        P_lim_node=mat.P_lim_node,
        P_other_node=(sv.P_neg if _shared else None),
        drift_sign=+1.0,
    )
    J_ion = Q * F_ion

    # Negative ion species contribution (reversed drift)
    if mat.has_dual_ions and sv.P_neg is not None:
        D_neg_face = np.broadcast_to(
            np.asarray(mat.D_ion_neg_face, dtype=float), dx.shape,
        )
        P_lim_neg_face = np.broadcast_to(
            np.asarray(mat.P_lim_neg_face, dtype=float), dx.shape,
        )
        F_neg = ion_face_flux(
            phi, sv.P_neg, dx, D_neg_face, V_T_dev, P_lim_neg_face,
            steric_diffusion_only=mat.ion_steric_diffusion_only,
            P_lim_node=mat.P_lim_neg_node,
            P_other_node=(sv.P if _shared else None),
            drift_sign=-1.0,
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


def _compute_current_ss(
    x: np.ndarray,
    y: np.ndarray,
    stack: DeviceStack,
    V_app: float,
    mat: MaterialArrays | None = None,
) -> float:
    """Extract J [A/m²] using the **median** across all interior faces.

    Use this when the caller has settled the device with a finite ``t_settle``
    and is reading J at fixed V — Suns-Voc, EQE, and similar V=0 SS probes.
    At true steady state every face carries the same total current (charge
    conservation), so any face is correct. In practice on ionic-rich presets
    the contact-adjacent faces still see residual ionic / displacement
    transients that haven't fully damped, while the interior faces agree on
    the photo-current to many digits. Median is the robust mid-summary that
    is insensitive to those boundary outliers — verified on
    ``ionmonger_benchmark_tmm`` where ``J[0]`` swings ±1700 A/m² across
    t_settle but median stays pinned at the linear-with-suns photo value.

    Use ``_compute_current`` (face[0]) on transient J-V sweeps where the
    displacement current at the contact is the physical terminal-current
    quantity flowing through the external circuit.

    Convention: J > 0 when the device delivers power (J_sc > 0 at V=0).
    """
    return _compute_current_ss_with_spread(x, y, stack, V_app, mat=mat)[0]


def _compute_current_ss_with_spread(
    x: np.ndarray,
    y: np.ndarray,
    stack: DeviceStack,
    V_app: float,
    mat: MaterialArrays | None = None,
) -> tuple[float, float]:
    """Return ``(J_median, dJ_spread)`` — the SS current and its certificate.

    Charge conservation makes ``J(x)`` uniform at true steady state, so the
    face-to-face disagreement is a direct measure of how far the state is
    from that limit. Taking the median (see ``_compute_current_ss``) makes
    the reported number robust to that disagreement, but robustness is not
    convergence: without the spread alongside it, a median can look clean
    while the underlying state still carries a continuity residual. This
    helper returns both so callers can gate on the certificate rather than
    trusting the summary (2026-07 review finding F-13).

    ``dJ_spread`` is ``max - min`` over the **interior** faces only. The two
    contact-adjacent faces are excluded on purpose: there the displacement
    term is part of the physical external-circuit current and, on ionic-rich
    presets, legitimately swings by ~1e3 A/m2 while the interior faces agree
    to many digits — including them would report a physical boundary
    transient as a conservation error. Grids with fewer than three faces
    fall back to the full set.

    IMPORTANT — report it, do NOT gate on it. The review that requested
    this certificate also proposed refusing to extract V_oc / J_sc / FF
    until the spread falls below a fixed tolerance. Measured on the TMM
    presets at V=0, that gate is not defensible, because the spread is
    neither monotone in settling time nor stable under mesh refinement
    while the reported median is both:

        preset            N/layer  t_settle   median J   interior spread
        nip_MAPbI3_tmm      20      1e-3       213.02      7.99e+01
        nip_MAPbI3_tmm      20      1e-1       213.02      1.47e+01
        ionmonger_bm_tmm    20      1e-3       212.99      1.63e+02
        ionmonger_bm_tmm    20      1e-1       213.00      7.06e+03   (!)
        ionmonger_bm_tmm    12      1e-1       218.65      8.80e+01

    The median moves by <0.02 % across settling times while the spread
    grows 43x with a *longer* settle and swings ~80x with the mesh, so a
    threshold would fire on converged runs and its natural remedy
    ("settle longer") is backwards. Normalising by the largest current
    component does not rescue it either (0.33 at N=12 vs ~26 at N=20).
    The cause is the documented terminal-current flux cancellation: J is
    a sum of large, nearly-cancelling electron, hole, ionic, and
    displacement contributions, so a small relative error in the parts
    becomes a large absolute error in their sum — which is exactly the
    condition under which the median is the right summary and the raw
    spread is a poor convergence proxy. Treat ``dJ_spread`` as a
    magnitude diagnostic to report, and use mesh convergence of the
    extracted metrics for the convergence question it cannot answer.
    """
    J_faces = _total_current_faces(x, y, stack, V_app, mat=mat)
    J = float(np.median(J_faces))
    interior = J_faces[1:-1] if J_faces.size > 2 else J_faces
    return J, float(np.max(interior) - np.min(interior))


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

# Wrong-branch detector for the forward power quadrant (see
# _integrate_step's docstring for why step control alone cannot fix this).
#
# The bound is physical, not fitted: under illumination at V > 0 the
# terminal current is J(V) = J_sc - J_dark(V) with J_dark >= 0 for forward
# bias, so J can never RISE above its own short-circuit value. A step that
# reports otherwise has landed on the carrier-injection branch of the
# implicit system.
#
# The margin absorbs the one legitimate way J(V) can exceed J(0) on this
# code path — a fast scan over a mobile-ion stack, where the ionic
# configuration lags the bias and the early forward points carry a small
# transient excess. Measured across the shipped presets, the largest
# genuine excess is ~1e-4 of J_sc, four orders below this bound, while the
# defect it must catch is 2.3x (509.8 vs 221.9) and the smallest wrong
# branch seen while sweeping step control is 1.15x (255.9). 5 % therefore
# sits ~500x above the noise and ~3x below the smallest real defect.
_J_BRANCH_EXCESS = 0.05

# Leg counts tried, in order, when the bound above rejects a step. Every
# one of these recovered the identical physical value at the measured
# failure (2/4/8 legs and BDF all give 161.986 vs the single-leg 509.796),
# so the ladder is about robustness on unmeasured configs, not tuning.
_J_BRANCH_RETRY_LEGS = (2, 4, 8)


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
    if not (mat.has_radiative_reabsorption and mat.absorber_masks):
        return mat
    sv = StateVec.unpack(y, len(x))
    # Under dark forward injection (LED-mode), ``mat.G_optical`` still holds
    # the solar absorption profile that ``build_material_arrays`` cached, but
    # the runtime branch in ``assemble_rhs`` zeros G when ``illuminated=False``.
    # The bake mirrors that decision by zeroing G_base here, so the returned
    # mat carries only the lagged R_rad source. The companion
    # ``force_use_g_optical=True`` flag tells ``assemble_rhs`` to skip the
    # dark zeroing on this baked mat.
    if illuminated:
        G_base = mat.G_optical if mat.G_optical is not None else np.zeros_like(x)
    else:
        G_base = np.zeros_like(x)
    G_with_rad = G_base.copy()
    for mask, P_esc, thickness in zip(
        mat.absorber_masks, mat.absorber_p_esc, mat.absorber_thicknesses
    ):
        if thickness <= 0.0 or P_esc >= 1.0:
            continue
        # NET emission B(np - ni^2) — must match the per-RHS form in
        # assemble_rhs so the baked step preserves detailed balance (F08).
        emission = mat.B_rad[mask] * (sv.n[mask] * sv.p[mask] - mat.ni_sq[mask])
        x_abs = x[mask]
        if x_abs.size < 2:
            continue
        R_tot = float(trapezoid(emission, x_abs))
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
        force_use_g_optical=True,
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
    max_bisect: int = 10,
    illuminated: bool = True,
    n_legs: int = 1,
) -> np.ndarray:
    """Advance the coupled MOL state from t_lo to t_hi at fixed V_app.

    Radau is adaptive but its error estimator can underreport truncation
    error near V_bi, where the Jacobian becomes nearly singular (flat-band
    region). Without an explicit `max_step` cap it sometimes takes a single
    huge step across the whole [t_lo, t_hi] interval and accepts a state
    that landed on the wrong (carrier-injection) branch of the implicit
    system — producing isolated non-physical spikes in the J-V curve. We
    cap max_step to (t_hi - t_lo)/20 so the solver must resolve the
    transient with at least ~20 internal steps, which suppresses most of
    those spikes without materially slowing well-conditioned regions.

    **The cap is necessary but NOT sufficient, and it cannot be made
    sufficient by tightening it** (measured 2026-07-28 on
    ``ionmonger_benchmark`` N_grid=40/n_points=20, whose V=1.10526 sample
    sits essentially exactly on ``V_bi`` = 1.1). Sweeping the divisor gives
    J = 509.8 / 162.0 / 255.9 / 162.0 / 162.0 at dt/20, /100, /200, /1000,
    /2000 — non-monotone, because a global change to max_step perturbs
    EVERY step's trajectory and merely relocates which one lands on the
    wrong branch. Same lesson as the E9.3 clamp-shape variants; do not
    re-attempt a divisor fix.

    What does work is re-integrating **only the offending step** with
    forced subdivision, which is what ``n_legs`` is for: the caller detects
    a wrong-branch landing on a physical bound and re-runs that one step as
    ``n_legs`` chained sub-intervals. Measured at the failing step, every
    recovery agrees to the digit — 2, 4 and 8 legs and a BDF single leg all
    return J = 161.986 against the single-leg 509.796, and 161.986 is also
    what four independent mesh/sampling refinements converge to. Because
    the retry only fires on a violation, healthy sweeps are bit-identical.

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
    if n_legs > 1 and dt > 0.0:
        # Forced subdivision: the caller rejected the single-leg result on a
        # physical bound (see _integrate_step_on_physical_branch). Chain
        # `n_legs` equal sub-intervals, each with its own max_step cap.
        edges = np.linspace(t_lo, t_hi, n_legs + 1)
        y_k = y
        for t_a, t_b in zip(edges[:-1], edges[1:]):
            y_k = _integrate_step(
                x, y_k, stack, mat, V_app, float(t_a), float(t_b),
                rtol, atol, max_bisect, illuminated,
            )
        return y_k
    sol = run_transient(
        x, y, (t_lo, t_hi), np.array([t_hi]),
        stack, illuminated=illuminated, V_app=V_app, rtol=rtol, atol=atol,
        max_step=dt / 20.0 if dt > 0.0 else np.inf,
        mat=mat,
        max_nfev=_JV_RADAU_MAX_NFEV,
    )
    if sol.success:
        return sol.y[:, -1]
    if mat.has_radiative_reabsorption:
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
        # Last-chance BDF fallback. When Radau bisection is exhausted on
        # a near-flat-band / deep-injection state, scipy's BDF (variable-
        # order BDF) sometimes converges where Radau cannot. Cost is one
        # extra solve attempt per pathological step; healthy steps never
        # reach this branch.
        sol_bdf = run_transient(
            x, y, (t_lo, t_hi), np.array([t_hi]),
            stack, illuminated=illuminated, V_app=V_app, rtol=rtol, atol=atol,
            max_step=(t_hi - t_lo) / 20.0 if t_hi > t_lo else np.inf,
            mat=mat,
            max_nfev=_JV_RADAU_MAX_NFEV,
            method="BDF",
        )
        if sol_bdf.success:
            return sol_bdf.y[:, -1]
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


def _layer_node_counts(stack: DeviceStack, N_grid: int) -> list[int]:
    """Per-electrical-layer interval count for the multilayer grid.

    Base is ``n_per = N_grid // n_elec`` per layer. A graded layer (when
    ``band_grading`` is on and it declares back endpoints) is refined by its
    ``grading_N_mult`` so a steep notch's band step is resolved over enough
    cells instead of landing on a single over-injecting face. With no graded
    layer every multiplier is 1, recovering the legacy sizing exactly.
    """
    elec = electrical_layers(stack)
    n_per = N_grid // len(elec)
    band_grading = bool(getattr(stack, "band_grading", False))
    counts: list[int] = []
    for l in elec:
        mult = 1
        if band_grading and l.params is not None and has_grading_params(l.params):
            mult = max(1, int(getattr(l.params, "grading_N_mult", 1)))
        counts.append(n_per * mult)
    return counts


def _grid_node_count(stack: DeviceStack, N_grid: int) -> int:
    """Return the number of electrical grid nodes run_jv_sweep will build.

    This is the single source of truth for the electrical-grid sizing formula.
    Both run_jv_sweep (internally) and tandem callers (to pre-size generation
    profiles) must use this helper so the two sites can never silently diverge.

    Formula: ``1 + sum(_layer_node_counts(stack, N_grid))``. multilayer_grid
    deduplicates shared boundary points, so the total node count is one more
    than the sum of per-layer intervals. With no graded layer this reduces to
    ``1 + n_elec * (N_grid // n_elec)`` — byte-identical to the legacy formula.
    """
    return 1 + sum(_layer_node_counts(stack, N_grid))


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
    # compute_V_bi is the SIGNED phi(right)-phi(left) (negative for
    # n-contact-left devices); the sweep range needs its magnitude.
    V_bi_eff = abs(stack.compute_V_bi())
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
    v_max_max_attempts: int = 1,
) -> JVResult:
    """Run forward and reverse J-V sweeps.

    V_max : upper voltage limit. If None, defaults to max(V_bi_eff*1.3, 1.4). With
      heterojunction band offsets, V_oc can exceed V_bi, so pass a larger
      value (e.g. 1.4 V for MAPbI3) to capture the full forward curve.
      That default is derived from the CONTACTS and knows nothing about the
      point being swept, so it misses in both directions. Too low is loud —
      ``voc_bracketed=False``, and ``v_max_max_attempts`` below recovers it.
      Too high is silent and expensive: a device whose V_oc has collapsed
      (deep cliff/spike offsets, short lifetime, strong interface
      recombination) still gets swept to 1.4 V, grinding the deep
      forward-injection region above its own V_oc. Measured on two
      spike-side screening points: 1567 s and 630 s at the default versus
      23 s and 16 s starting low and climbing, agreeing on V_oc to 1.6 mV
      and 0.1 mV. For a sweep over devices whose V_oc varies, prefer a low
      ``V_max`` plus ``v_max_max_attempts`` over one range that fits all.

    v_max_max_attempts : Phase E1.9 — opt-in adaptive V_max bump when the
      first sweep fails to bracket V_oc (forward J never crosses zero).
      Default 1 = legacy no-retry behaviour, bit-identical to pre-E1.9.
      Values > 1 trigger up to ``v_max_max_attempts - 1`` retries with
      V_max bumped by 0.5 V per attempt (cap at V_initial + 2.0 V). The
      cost is bounded — at worst, ``v_max_max_attempts`` full forward/
      reverse sweeps, and fewer once V_max saturates at the cap, since a
      further rung would repeat that sweep verbatim. Each rung holds the
      VOLTAGE STEP of the first attempt and grows ``n_points`` to match, so
      a retry never answers more coarsely than the attempt it is correcting
      (V_oc is interpolated between the bracketing samples, so its error
      scales with that step). Useful for stacks where V_oc may exceed the
      default V_max (e.g. Robin contacts at low ETL doping; SCAPS
      validation script's extreme-N_D sweep points), and — passed together
      with a deliberately low ``V_max`` — as the cheap way to sweep devices
      whose V_oc is unknown or collapses: start below every candidate and
      let the ladder climb only where it must. On exhaustion the
      result returns ``voc_bracketed=False`` and the standard sentinel
      zeros for V_oc/FF/PCE — no exception is raised, so callers can
      inspect the flag and decide how to handle the residual gap.

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
    if v_max_max_attempts < 1:
        raise ValueError(
            f"v_max_max_attempts must be >= 1, got {v_max_max_attempts}"
        )
    if n_points < 2:
        # Checked before the ladder below, which divides by ``n_points - 1``.
        raise ValueError(f"n_points must be >= 2, got {n_points}")
    if v_max_max_attempts > 1:
        # Adaptive V_max bump: run the legacy single-attempt path first;
        # if it brackets V_oc successfully, return that result unchanged
        # (so attempts > 1 with successful first try is bit-identical to
        # attempts = 1). On failed bracket, retry with V_max += 0.5 per
        # attempt until success or attempts exhaust.
        V_max_initial = (
            V_max
            if V_max is not None
            else max(abs(stack.compute_V_bi()) * 1.3, 1.4)
        )
        V_max_cap = V_max_initial + 2.0
        V_max_attempt = V_max_initial
        # Hold the VOLTAGE STEP fixed and let ``n_points`` grow with the rung,
        # instead of stretching a fixed sample count over a wider range. V_oc
        # is interpolated between the two samples that bracket it, so its
        # error scales with this step; passing ``n_points`` through unchanged
        # made every retry answer more coarsely than the attempt it was
        # correcting, and did so systematically in V_max.
        dV = V_max_initial / (n_points - 1)
        attempts_used = 0
        last_result: JVResult | None = None
        while True:
            last_result = run_jv_sweep(
                stack=stack, N_grid=N_grid, v_rate=v_rate,
                n_points=int(round(V_max_attempt / dV)) + 1,
                rtol=rtol, atol=atol,
                V_max=V_max_attempt, progress=progress,
                fixed_generation=fixed_generation,
                illuminated=illuminated, save_snapshots=save_snapshots,
                decompose_currents=decompose_currents,
                v_max_max_attempts=1,  # disable recursion on inner call
            )
            if last_result.metrics_fwd.voc_bracketed:
                return last_result
            attempts_used += 1
            V_max_next = min(V_max_attempt + 0.5, V_max_cap)
            if attempts_used >= v_max_max_attempts or V_max_next <= V_max_attempt:
                # Budget spent, or V_max has saturated at the cap so a further
                # rung would repeat this exact sweep. (The old bail-out here
                # compared a float with a JVResult and could never fire, so an
                # exhausted ladder re-ran the capped sweep to the last attempt.)
                break
            V_max_attempt = V_max_next
        # ``last_result`` is set: the loop body always runs at least once.
        assert last_result is not None
        return last_result
    if N_grid < 3:
        raise ValueError(f"N_grid must be >= 3, got {N_grid}")
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
    layers_grid = [
        Layer(l.thickness, n)
        for l, n in zip(elec, _layer_node_counts(stack, N_grid))
    ]
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

    # Kick off progress BEFORE the initial equilibration solve. That solve
    # (solve_illuminated_ss / dark settle) can be slow on stiff, high-mobility
    # stacks and emits nothing itself, so without this frame the UI sits at a
    # frozen "Idle 0%" for the whole settle. current=0 marks the indeterminate
    # "equilibrating" phase; the sweep loop then reports "jv_forward" from 1.
    if progress is not None:
        progress("jv_init", 0, n_points, "equilibrating")

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

    def _sweep(V_start: float, V_end: float, y_init: np.ndarray, stage: str,
               J_sc_ref: float | None = None):
        """Sweep from V_start to V_end, starting from carrier state y_init.

        Returns (V_arr, J_arr, y_final, snapshots, decomp) so sweeps can be
        chained. snapshots and decomp are populated only when the corresponding
        flags (save_snapshots, decompose_currents) are True.

        ``J_sc_ref`` is the short-circuit current used by the wrong-branch
        detector (``_J_BRANCH_EXCESS``). The forward sweep starts at V=0 and
        therefore learns it from its own first point; the reverse sweep ends
        there, so ``run_jv_sweep`` hands it the forward value.
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
        J_ref = J_sc_ref
        for k, V_k in enumerate(V_arr):
            y_prev = y.copy()
            t_lo = t_points[k]
            t_hi = t_lo + dt
            y = _integrate_step(x, y, stack, mat, V_k, t_lo, t_hi, rtol, atol,
                               illuminated=illuminated)
            J_k = _compute_current(x, y, stack, V_k, y_prev=y_prev, dt=dt,
                                    mat=mat, V_app_prev=V_prev)

            # Wrong-branch rejection. Only fires in the illuminated forward
            # power quadrant, where J(V) <= J_sc is a hard physical bound;
            # healthy steps never enter this block, so results elsewhere are
            # bit-identical.
            if (illuminated and J_ref is not None and J_ref > 0.0
                    and float(V_k) > 0.0
                    and J_k > J_ref * (1.0 + _J_BRANCH_EXCESS)):
                for n_legs in _J_BRANCH_RETRY_LEGS:
                    y_retry = _integrate_step(
                        x, y_prev, stack, mat, V_k, t_lo, t_hi, rtol, atol,
                        illuminated=illuminated, n_legs=n_legs,
                    )
                    J_retry = _compute_current(
                        x, y_retry, stack, V_k, y_prev=y_prev, dt=dt,
                        mat=mat, V_app_prev=V_prev,
                    )
                    if J_retry <= J_ref * (1.0 + _J_BRANCH_EXCESS):
                        y, J_k = y_retry, J_retry
                        break
                else:
                    # Never silently keep a value known to break the bound.
                    warnings.warn(
                        f"J-V sweep: terminal current at V={float(V_k):.4f} V "
                        f"is {J_k:.3f} A/m^2, above the physical ceiling "
                        f"J_sc*(1+{_J_BRANCH_EXCESS:g}) = "
                        f"{J_ref * (1.0 + _J_BRANCH_EXCESS):.3f}; forced "
                        f"subdivision into {_J_BRANCH_RETRY_LEGS} legs did not "
                        "recover the physical branch. Treat this point as "
                        "unconverged.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
            J_arr[k] = J_k
            if J_ref is None and float(V_k) == 0.0:
                J_ref = J_k
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
    # The reverse sweep reaches V=0 only at its LAST point, so it cannot learn
    # J_sc from itself in time to police its own forward-bias points — hand it
    # the forward sweep's value (same device, same illumination).
    J_sc_fwd = float(J_fwd[0]) if len(J_fwd) else None
    V_rev, J_rev, _, snaps_rev, decomp_rev = _sweep(
        V_upper, 0.0, y_oc, "jv_reverse", J_sc_ref=J_sc_fwd,
    )

    # Refuse a V_oc above min(Eg)/q over the electrical layers — impossible
    # regardless of what the numerics produced past flat band. None when the
    # stack declares no band gaps (legacy chi=Eg=0 presets), which disables
    # the check rather than rejecting everything.
    V_oc_ceiling = thermodynamic_voc_ceiling(stack)
    m_fwd = compute_metrics(V_fwd, J_fwd, V_oc_max=V_oc_ceiling)
    m_rev = compute_metrics(V_rev[::-1], J_rev[::-1], V_oc_max=V_oc_ceiling)
    HI = hysteresis_index(V_fwd, J_fwd, V_rev[::-1], J_rev[::-1],
                          V_oc_max=V_oc_ceiling)

    return JVResult(
        V_fwd=V_fwd, J_fwd=J_fwd, V_rev=V_rev, J_rev=J_rev,
        metrics_fwd=m_fwd, metrics_rev=m_rev, hysteresis_index=HI,
        snapshots_fwd=tuple(snaps_fwd) if save_snapshots else None,
        snapshots_rev=tuple(snaps_rev) if save_snapshots else None,
        decomp_fwd=decomp_fwd,
        decomp_rev=decomp_rev,
    )
