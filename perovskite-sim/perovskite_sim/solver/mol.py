from __future__ import annotations
from dataclasses import dataclass
import warnings
import numpy as np
try:
    from scipy.integrate import solve_ivp
except ImportError:
    from scipy_shim import solve_ivp

from perovskite_sim.discretization.grid import multilayer_grid, Layer
from perovskite_sim.physics.poisson import solve_poisson
from perovskite_sim.physics.continuity import carrier_continuity_rhs
from perovskite_sim.physics.ion_migration import ion_continuity_rhs
from perovskite_sim.physics.generation import beer_lambert_generation
from perovskite_sim.models.device import DeviceStack

from perovskite_sim.physics.recombination import interface_recombination
from perovskite_sim.constants import Q, V_T


@dataclass
class StateVec:
    n: np.ndarray
    p: np.ndarray
    P: np.ndarray

    @staticmethod
    def pack(n, p, P) -> np.ndarray:
        return np.concatenate([n, p, P])

    @staticmethod
    def unpack(y: np.ndarray, N: int) -> "StateVec":
        return StateVec(n=y[:N], p=y[N:2*N], P=y[2*N:3*N])


def _build_layerwise_arrays(x: np.ndarray, stack: DeviceStack):
    """Return per-node material arrays over the full device grid."""
    N = len(x)
    eps_r = np.ones(N)
    D_ion = np.zeros(N)
    P_lim = 1e30 * np.ones(N)
    P_ion0 = np.zeros(N)
    N_A   = np.zeros(N)
    N_D   = np.zeros(N)
    alpha = np.zeros(N)
    chi   = np.zeros(N)
    Eg    = np.zeros(N)
    offset = 0.0
    for layer in stack.layers:
        mask = (x >= offset - 1e-12) & (x <= offset + layer.thickness + 1e-12)
        p = layer.params
        eps_r[mask] = p.eps_r
        D_ion[mask] = p.D_ion
        P_lim[mask] = p.P_lim
        P_ion0[mask] = p.P0
        N_A[mask]   = p.N_A
        N_D[mask]   = p.N_D
        alpha[mask] = p.alpha
        chi[mask]   = p.chi
        Eg[mask]    = p.Eg
        offset += layer.thickness
    return eps_r, D_ion, P_lim, P_ion0, N_A, N_D, alpha, chi, Eg


def _harmonic_face_average(values: np.ndarray) -> np.ndarray:
    """Harmonic mean on inter-node faces, with zero conductivity preserved."""
    numer = 2.0 * values[:-1] * values[1:]
    denom = values[:-1] + values[1:]
    return np.divide(numer, denom, out=np.zeros_like(numer), where=denom > 0.0)


def _build_ion_face_params(
    D_ion_node: np.ndarray,
    P_lim_node: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build per-face ion transport coefficients.

    Harmonic averaging makes the interfacial face coefficient exactly zero when
    one side is ion-blocking, preventing artificial leakage into transport
    layers where `D_ion = 0`.
    """
    D_ion_face = _harmonic_face_average(D_ion_node)
    P_lim_face = 0.5 * (P_lim_node[:-1] + P_lim_node[1:])
    return D_ion_face, P_lim_face


def _build_carrier_params(x: np.ndarray, stack: DeviceStack) -> dict:
    """Build per-face diffusion and per-node recombination arrays.

    Returns a params dict compatible with carrier_continuity_rhs:
      D_n, D_p  : (N-1,) arrays – diffusion coefficients at inter-node faces
      ni_sq, tau_n, tau_p, n1, p1, B_rad, C_n, C_p : (N,) per-node arrays

    Interface nodes use the last-matching layer, so transport-layer nodes get
    their own ni, tau values rather than the absorber's. Inter-layer faces use
    the harmonic mean of adjacent nodal D values (series-resistance model),
    consistent with solve_poisson's harmonic-mean treatment of eps_r.
    """
    N = len(x)
    D_n_node = np.empty(N); D_p_node = np.empty(N)
    ni_sq = np.empty(N); tau_n = np.empty(N); tau_p = np.empty(N)
    n1    = np.empty(N); p1    = np.empty(N)
    B_rad = np.empty(N); C_n   = np.empty(N); C_p   = np.empty(N)
    chi   = np.zeros(N); Eg    = np.zeros(N)

    offset = 0.0
    for layer in stack.layers:
        mask = (x >= offset - 1e-12) & (x <= offset + layer.thickness + 1e-12)
        p = layer.params
        D_n_node[mask] = p.D_n;  D_p_node[mask] = p.D_p
        ni_sq[mask]    = p.ni_sq; tau_n[mask] = p.tau_n; tau_p[mask] = p.tau_p
        n1[mask]       = p.n1;    p1[mask]    = p.p1
        B_rad[mask]    = p.B_rad; C_n[mask]   = p.C_n;   C_p[mask]   = p.C_p
        chi[mask]      = p.chi;   Eg[mask]    = p.Eg
        offset += layer.thickness

    # Per-face D via harmonic mean of adjacent nodal values.
    # Matches solve_poisson's harmonic-mean treatment of eps_r at interfaces:
    # both correspond to the series-resistance result for a sharp discontinuity.
    D_n_face = 2.0 * D_n_node[:-1] * D_n_node[1:] / (D_n_node[:-1] + D_n_node[1:])
    D_p_face = 2.0 * D_p_node[:-1] * D_p_node[1:] / (D_p_node[:-1] + D_p_node[1:])

    return dict(D_n=D_n_face, D_p=D_p_face, V_T=V_T,
                ni_sq=ni_sq, tau_n=tau_n, tau_p=tau_p,
                n1=n1, p1=p1, B_rad=B_rad, C_n=C_n, C_p=C_p,
                chi=chi, Eg=Eg)


def _equilibrium_bc(stack: DeviceStack, x: np.ndarray):
    """Ohmic contact carrier densities from doping."""
    def equilibrium_np(N_D, N_A, ni):
        net = 0.5 * (N_D - N_A)
        disc = np.sqrt(net**2 + ni**2)
        if net >= 0:          # n-type or intrinsic: compute n first (large)
            n = net + disc
            p = ni**2 / n
        else:                 # p-type: compute p first (large), avoid cancellation
            p = -net + disc
            n = ni**2 / p
        return n, p

    first_layer = stack.layers[0]
    last_layer  = stack.layers[-1]
    n_L, p_L = equilibrium_np(
        first_layer.params.N_D, first_layer.params.N_A, first_layer.params.ni
    )
    n_R, p_R = equilibrium_np(
        last_layer.params.N_D, last_layer.params.N_A, last_layer.params.ni
    )
    return n_L, p_L, n_R, p_R


def _charge_density(
    p: np.ndarray,
    n: np.ndarray,
    P_ion: np.ndarray,
    P_ion0: np.ndarray,
    N_A: np.ndarray,
    N_D: np.ndarray,
) -> np.ndarray:
    """Space-charge density with ionic charge measured relative to neutral background."""
    return Q * (p - n + (P_ion - P_ion0) - N_A + N_D)


def _find_interface_nodes(x: np.ndarray, stack: DeviceStack) -> list[int]:
    """Return grid indices closest to each internal interface."""
    indices = []
    offset = 0.0
    for layer in stack.layers[:-1]:
        offset += layer.thickness
        idx = int(np.argmin(np.abs(x - offset)))
        indices.append(idx)
    return indices


def _apply_interface_recombination(
    dn: np.ndarray,
    dp: np.ndarray,
    x: np.ndarray,
    n: np.ndarray,
    p: np.ndarray,
    stack: DeviceStack,
    ni_sq: np.ndarray,
    n1: np.ndarray,
    p1: np.ndarray,
) -> None:
    """Subtract interface recombination from dn, dp at interface nodes (in-place).

    Interface recombination is a surface rate [m⁻² s⁻¹] converted to
    volumetric [m⁻³ s⁻¹] by dividing by the dual-grid cell width.
    """
    if not stack.interfaces:
        return
    dx = np.diff(x)
    dx_cell = np.empty(len(x))
    dx_cell[0] = dx[0]
    dx_cell[-1] = dx[-1]
    dx_cell[1:-1] = 0.5 * (dx[:-1] + dx[1:])

    iface_nodes = _find_interface_nodes(x, stack)
    for k, idx in enumerate(iface_nodes):
        if k >= len(stack.interfaces):
            break
        v_n, v_p = stack.interfaces[k]
        if v_n == 0.0 and v_p == 0.0:
            continue
        R_s = interface_recombination(
            n[idx], p[idx], float(ni_sq[idx]),
            float(n1[idx]), float(p1[idx]),
            v_n, v_p,
        )
        R_vol = R_s / dx_cell[idx]
        dn[idx] -= R_vol
        dp[idx] -= R_vol


def assemble_rhs(
    t: float,
    y: np.ndarray,
    x: np.ndarray,
    stack: DeviceStack,
    illuminated: bool = True,
    V_app: float = 0.0,
) -> np.ndarray:
    """Method of Lines RHS: dy/dt = f(t, y)."""
    N = len(x)
    sv = StateVec.unpack(y, N)

    eps_r, D_ion_node, P_lim_node, P_ion0, N_A, N_D, alpha_arr, _, _ = (
        _build_layerwise_arrays(x, stack)
    )

    # Boundary conditions
    n_L, p_L, n_R, p_R = _equilibrium_bc(stack, x)
    n = sv.n.copy(); n[0] = n_L; n[-1] = n_R
    p = sv.p.copy(); p[0] = p_L; p[-1] = p_R

    # Solve Poisson
    # phi_right = V_bi - V_app: forward bias (V_app > 0) reduces the
    # built-in field; V_app = 0 → short circuit, V_app ≈ V_oc → open circuit.
    rho = _charge_density(p, n, sv.P, P_ion0, N_A, N_D)
    phi = solve_poisson(x, eps_r, rho, phi_left=0.0, phi_right=stack.V_bi - V_app)

    # Generation (layered Beer-Lambert with correct cumulative optical depth)
    if illuminated:
        G = beer_lambert_generation(x, alpha_arr, stack.Phi)
    else:
        G = np.zeros(N)

    # Carrier continuity with per-layer D and per-node recombination params
    params = _build_carrier_params(x, stack)
    dn, dp = carrier_continuity_rhs(x, phi, n, p, G, params)

    # Interface recombination (surface SRH at heterointerfaces)
    _apply_interface_recombination(
        dn, dp, x, n, p, stack,
        params["ni_sq"], params["n1"], params["p1"],
    )

    # Ion continuity using per-face transport coefficients so ions remain
    # confined to ion-conducting layers.
    D_ion_face, P_lim_face = _build_ion_face_params(D_ion_node, P_lim_node)
    dP = ion_continuity_rhs(x, phi, sv.P, D_ion_face, V_T, P_lim_face)

    # Enforce Dirichlet BCs: hold boundary nodes fixed
    dn[0] = dn[-1] = 0.0
    dp[0] = dp[-1] = 0.0

    return StateVec.pack(dn, dp, dP)


def run_transient(
    x: np.ndarray,
    y0: np.ndarray,
    t_span: tuple[float, float],
    t_eval: np.ndarray,
    stack: DeviceStack,
    illuminated: bool = True,
    V_app: float = 0.0,
    rtol: float = 1e-4,
    atol: float = 1e-6,
    max_step: float = np.inf,
):
    """Integrate MOL system from t_span[0] to t_span[1]."""
    N = len(x)

    def rhs(t, y):
        return assemble_rhs(t, y, x, stack, illuminated, V_app)

    return solve_ivp(rhs, t_span, y0, t_eval=t_eval,
                     method="Radau", rtol=rtol, atol=atol,
                     dense_output=False, max_step=max_step)


def split_step(
    x: np.ndarray,
    y: np.ndarray,
    dt: float,
    stack: DeviceStack,
    V_app: float = 0.0,
    rtol: float = 1e-4,
    atol: float = 1e-6,
) -> tuple[np.ndarray, bool]:
    """Operator-split step for long-time ion–carrier evolution.

    Decouples the ion (slow, seconds) and carrier (fast, nanoseconds)
    timescales that make the fully-coupled Radau solver fail when ions
    have piled up at interfaces:

    1. **Ion advance**: freeze n, p at current values; advance the ion
       distribution P by dt using only the ion continuity equation.
       This sub-system is much less stiff (no carrier feedback).

    2. **Carrier re-equilibration**: run a short transient (1 µs, one
       carrier lifetime) from the updated ion state so that n and p
       reach their new quasi-steady state.

    Parameters
    ----------
    x, y    : grid positions and current state vector
    dt      : ion time step [s]
    stack   : device stack
    V_app   : applied voltage [V]
    rtol, atol : ODE solver tolerances

    Returns
    -------
    (y_new, success) : updated state and whether the ion advance succeeded.
    If the ion advance fails, returns (y, False) unchanged.
    If carrier re-equilibration fails, returns the ion-advanced state
    with previous carrier values (partial success, still True).
    """
    N = len(x)
    sv = StateVec.unpack(y, N)

    # Apply ohmic contact BCs to frozen carrier arrays
    n_L, p_L, n_R, p_R = _equilibrium_bc(stack, x)
    n_frozen = sv.n.copy(); n_frozen[0] = n_L; n_frozen[-1] = n_R
    p_frozen = sv.p.copy(); p_frozen[0] = p_L; p_frozen[-1] = p_R

    eps_r, D_ion_node, P_lim_node, P_ion0, N_A, N_D, _, _, _ = _build_layerwise_arrays(x, stack)
    D_ion_face, P_lim_face = _build_ion_face_params(D_ion_node, P_lim_node)

    initial_neg = np.any(sv.P < -1e-30)
    _clip_count = [1 if initial_neg else 0]  # mutable closure counter

    def _ion_rhs(t, P):
        # Clip to non-negative: ions cannot have negative density.
        # Track significant clips (|P| > 1e-30) so we can warn the caller.
        neg_mask = P < -1e-30
        if np.any(neg_mask):
            _clip_count[0] += 1
        P_nn = np.maximum(P, 0.0)
        rho = _charge_density(p_frozen, n_frozen, P_nn, P_ion0, N_A, N_D)
        phi = solve_poisson(x, eps_r, rho,
                            phi_left=0.0, phi_right=stack.V_bi - V_app)
        return ion_continuity_rhs(x, phi, P_nn, D_ion_face, V_T, P_lim_face)

    sol_ion = solve_ivp(
        _ion_rhs, (0.0, dt), np.maximum(sv.P, 0.0), t_eval=[dt],
        method="Radau", rtol=rtol, atol=atol,
    )
    if _clip_count[0] > 0:
        warnings.warn(
            f"Ion density clipped to zero in {_clip_count[0]} RHS evaluation(s) "
            "during split_step. This indicates numerical instability — consider "
            "reducing atol for ion states or shortening dt.",
            RuntimeWarning,
            stacklevel=2,
        )
    if not sol_ion.success:
        return y, False

    P_new = np.clip(sol_ion.y[:, -1], 0.0, P_lim_node)
    y_ions_advanced = StateVec.pack(sv.n, sv.p, P_new)

    # Re-equilibrate carriers for one carrier lifetime (~1 µs)
    t_eq = 1e-6
    sol_eq = run_transient(
        x, y_ions_advanced, (0.0, t_eq), np.array([t_eq]),
        stack, illuminated=True, V_app=V_app, rtol=rtol, atol=atol,
    )
    if not sol_eq.success:
        # Ions advanced; keep previous carrier values (conservative fallback)
        return y_ions_advanced, True
    return sol_eq.y[:, -1], True
