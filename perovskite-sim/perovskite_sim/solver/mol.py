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


@dataclass(frozen=True)
class MaterialArrays:
    """Pre-computed per-node and per-face material arrays for one device.

    These quantities depend only on the device geometry and layer stack, not
    on time or state, so they can be built once per experiment and reused on
    every RHS evaluation. Building them inside `assemble_rhs` (the original
    behavior) allocated ~20 numpy arrays per Radau RHS call, which dominated
    the runtime of all three experiments.

    Build one with `build_material_arrays(x, stack)`.
    """
    # Per-node arrays (length N)
    eps_r: np.ndarray
    D_ion_node: np.ndarray
    P_lim_node: np.ndarray
    P_ion0: np.ndarray
    N_A: np.ndarray
    N_D: np.ndarray
    alpha: np.ndarray
    chi: np.ndarray
    Eg: np.ndarray
    ni_sq: np.ndarray
    tau_n: np.ndarray
    tau_p: np.ndarray
    n1: np.ndarray
    p1: np.ndarray
    B_rad: np.ndarray
    C_n: np.ndarray
    C_p: np.ndarray
    # Per-face arrays (length N-1)
    D_n_face: np.ndarray
    D_p_face: np.ndarray
    D_ion_face: np.ndarray
    P_lim_face: np.ndarray
    # Dual-grid cell widths for interface recombination volume conversion
    dx_cell: np.ndarray
    # Interface node indices (length = len(stack.layers) - 1)
    interface_nodes: tuple[int, ...]
    # Ohmic contact carrier densities from doping
    n_L: float
    p_L: float
    n_R: float
    p_R: float

    @property
    def carrier_params(self) -> dict:
        """Dict shape expected by `carrier_continuity_rhs` — legacy key names."""
        return dict(
            D_n=self.D_n_face, D_p=self.D_p_face, V_T=V_T,
            ni_sq=self.ni_sq, tau_n=self.tau_n, tau_p=self.tau_p,
            n1=self.n1, p1=self.p1, B_rad=self.B_rad,
            C_n=self.C_n, C_p=self.C_p,
            chi=self.chi, Eg=self.Eg,
        )


def build_material_arrays(x: np.ndarray, stack: DeviceStack) -> MaterialArrays:
    """Construct the immutable per-experiment material array bundle.

    Consolidates what used to be four separate per-RHS helpers:
    `_build_layerwise_arrays`, `_build_carrier_params`, `_equilibrium_bc`,
    and `_find_interface_nodes`, plus the dx_cell prep from
    `_apply_interface_recombination`. The output is numerically identical
    to the legacy path — this is a caching refactor, not an algorithmic
    change.
    """
    N = len(x)

    eps_r = np.ones(N)
    D_ion_node = np.zeros(N)
    P_lim_node = 1e30 * np.ones(N)
    P_ion0 = np.zeros(N)
    N_A = np.zeros(N)
    N_D = np.zeros(N)
    alpha = np.zeros(N)
    chi = np.zeros(N)
    Eg = np.zeros(N)

    D_n_node = np.empty(N)
    D_p_node = np.empty(N)
    ni_sq = np.empty(N)
    tau_n = np.empty(N)
    tau_p = np.empty(N)
    n1 = np.empty(N)
    p1 = np.empty(N)
    B_rad = np.empty(N)
    C_n = np.empty(N)
    C_p = np.empty(N)

    offset = 0.0
    for layer in stack.layers:
        mask = (x >= offset - 1e-12) & (x <= offset + layer.thickness + 1e-12)
        p = layer.params
        eps_r[mask] = p.eps_r
        D_ion_node[mask] = p.D_ion
        P_lim_node[mask] = p.P_lim
        P_ion0[mask] = p.P0
        N_A[mask] = p.N_A
        N_D[mask] = p.N_D
        alpha[mask] = p.alpha
        chi[mask] = p.chi
        Eg[mask] = p.Eg
        D_n_node[mask] = p.D_n
        D_p_node[mask] = p.D_p
        ni_sq[mask] = p.ni_sq
        tau_n[mask] = p.tau_n
        tau_p[mask] = p.tau_p
        n1[mask] = p.n1
        p1[mask] = p.p1
        B_rad[mask] = p.B_rad
        C_n[mask] = p.C_n
        C_p[mask] = p.C_p
        offset += layer.thickness

    # Per-face diffusion via harmonic mean of adjacent nodal values.
    # Matches solve_poisson's eps_r treatment and the legacy
    # _build_carrier_params / _build_ion_face_params outputs exactly.
    D_n_face = 2.0 * D_n_node[:-1] * D_n_node[1:] / (D_n_node[:-1] + D_n_node[1:])
    D_p_face = 2.0 * D_p_node[:-1] * D_p_node[1:] / (D_p_node[:-1] + D_p_node[1:])
    D_ion_face = _harmonic_face_average(D_ion_node)
    P_lim_face = 0.5 * (P_lim_node[:-1] + P_lim_node[1:])

    # Dual-grid cell widths for surface→volumetric conversion at interfaces.
    dx = np.diff(x)
    dx_cell = np.empty(N)
    dx_cell[0] = dx[0]
    dx_cell[-1] = dx[-1]
    dx_cell[1:-1] = 0.5 * (dx[:-1] + dx[1:])

    # Interface nodes: grid index closest to each internal interface.
    iface_list: list[int] = []
    offset = 0.0
    for layer in stack.layers[:-1]:
        offset += layer.thickness
        iface_list.append(int(np.argmin(np.abs(x - offset))))

    # Ohmic-contact equilibrium carrier densities from doping.
    def _equilibrium_np(N_D_val: float, N_A_val: float, ni_val: float) -> tuple[float, float]:
        net = 0.5 * (N_D_val - N_A_val)
        disc = np.sqrt(net ** 2 + ni_val ** 2)
        if net >= 0:
            n_val = net + disc
            p_val = ni_val ** 2 / n_val
        else:
            p_val = -net + disc
            n_val = ni_val ** 2 / p_val
        return float(n_val), float(p_val)

    first = stack.layers[0].params
    last = stack.layers[-1].params
    n_L, p_L = _equilibrium_np(first.N_D, first.N_A, first.ni)
    n_R, p_R = _equilibrium_np(last.N_D, last.N_A, last.ni)

    return MaterialArrays(
        eps_r=eps_r,
        D_ion_node=D_ion_node,
        P_lim_node=P_lim_node,
        P_ion0=P_ion0,
        N_A=N_A,
        N_D=N_D,
        alpha=alpha,
        chi=chi,
        Eg=Eg,
        ni_sq=ni_sq,
        tau_n=tau_n,
        tau_p=tau_p,
        n1=n1,
        p1=p1,
        B_rad=B_rad,
        C_n=C_n,
        C_p=C_p,
        D_n_face=D_n_face,
        D_p_face=D_p_face,
        D_ion_face=D_ion_face,
        P_lim_face=P_lim_face,
        dx_cell=dx_cell,
        interface_nodes=tuple(iface_list),
        n_L=n_L,
        p_L=p_L,
        n_R=n_R,
        p_R=p_R,
    )


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
    n: np.ndarray,
    p: np.ndarray,
    stack: DeviceStack,
    mat: MaterialArrays,
) -> None:
    """Subtract interface recombination from dn, dp at interface nodes (in-place).

    Interface recombination is a surface rate [m⁻² s⁻¹] converted to
    volumetric [m⁻³ s⁻¹] by dividing by the dual-grid cell width.
    """
    if not stack.interfaces:
        return
    for k, idx in enumerate(mat.interface_nodes):
        if k >= len(stack.interfaces):
            break
        v_n, v_p = stack.interfaces[k]
        if v_n == 0.0 and v_p == 0.0:
            continue
        R_s = interface_recombination(
            n[idx], p[idx], float(mat.ni_sq[idx]),
            float(mat.n1[idx]), float(mat.p1[idx]),
            v_n, v_p,
        )
        R_vol = R_s / mat.dx_cell[idx]
        dn[idx] -= R_vol
        dp[idx] -= R_vol


def assemble_rhs(
    t: float,
    y: np.ndarray,
    x: np.ndarray,
    stack: DeviceStack,
    mat: MaterialArrays,
    illuminated: bool = True,
    V_app: float = 0.0,
) -> np.ndarray:
    """Method of Lines RHS: dy/dt = f(t, y).

    `mat` is the pre-built per-experiment material cache. Building it here
    would allocate ~20 numpy arrays per Radau RHS call and dominated runtime
    of the caching refactor's target experiments.
    """
    N = len(x)
    sv = StateVec.unpack(y, N)

    # Boundary conditions from cached equilibrium densities
    n = sv.n.copy(); n[0] = mat.n_L; n[-1] = mat.n_R
    p = sv.p.copy(); p[0] = mat.p_L; p[-1] = mat.p_R

    # Solve Poisson
    # phi_right = V_bi - V_app: forward bias (V_app > 0) reduces the
    # built-in field; V_app = 0 → short circuit, V_app ≈ V_oc → open circuit.
    rho = _charge_density(p, n, sv.P, mat.P_ion0, mat.N_A, mat.N_D)
    phi = solve_poisson(x, mat.eps_r, rho, phi_left=0.0, phi_right=stack.V_bi - V_app)

    # Generation (layered Beer-Lambert with correct cumulative optical depth)
    if illuminated:
        G = beer_lambert_generation(x, mat.alpha, stack.Phi)
    else:
        G = np.zeros(N)

    # Carrier continuity with per-layer D and per-node recombination params
    dn, dp = carrier_continuity_rhs(x, phi, n, p, G, mat.carrier_params)

    # Interface recombination (surface SRH at heterointerfaces)
    _apply_interface_recombination(dn, dp, n, p, stack, mat)

    # Ion continuity using per-face transport coefficients so ions remain
    # confined to ion-conducting layers.
    dP = ion_continuity_rhs(x, phi, sv.P, mat.D_ion_face, V_T, mat.P_lim_face)

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
    mat: MaterialArrays | None = None,
):
    """Integrate MOL system from t_span[0] to t_span[1].

    If `mat` is None, the material cache is built locally — convenient for
    one-off calls but wasteful when this function is invoked many times
    with the same stack (J-V sweeps, impedance frequency loops). Callers
    that loop should build `mat` once and pass it in.
    """
    if mat is None:
        mat = build_material_arrays(x, stack)

    def rhs(t, y):
        return assemble_rhs(t, y, x, stack, mat, illuminated, V_app)

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
    mat: MaterialArrays | None = None,
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
    if mat is None:
        mat = build_material_arrays(x, stack)

    N = len(x)
    sv = StateVec.unpack(y, N)

    # Apply ohmic contact BCs to frozen carrier arrays
    n_frozen = sv.n.copy(); n_frozen[0] = mat.n_L; n_frozen[-1] = mat.n_R
    p_frozen = sv.p.copy(); p_frozen[0] = mat.p_L; p_frozen[-1] = mat.p_R

    initial_neg = np.any(sv.P < -1e-30)
    _clip_count = [1 if initial_neg else 0]  # mutable closure counter

    def _ion_rhs(t, P):
        # Clip to non-negative: ions cannot have negative density.
        # Track significant clips (|P| > 1e-30) so we can warn the caller.
        neg_mask = P < -1e-30
        if np.any(neg_mask):
            _clip_count[0] += 1
        P_nn = np.maximum(P, 0.0)
        rho = _charge_density(p_frozen, n_frozen, P_nn, mat.P_ion0, mat.N_A, mat.N_D)
        phi = solve_poisson(x, mat.eps_r, rho,
                            phi_left=0.0, phi_right=stack.V_bi - V_app)
        return ion_continuity_rhs(x, phi, P_nn, mat.D_ion_face, V_T, mat.P_lim_face)

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

    P_new = np.clip(sol_ion.y[:, -1], 0.0, mat.P_lim_node)
    y_ions_advanced = StateVec.pack(sv.n, sv.p, P_new)

    # Re-equilibrate carriers for one carrier lifetime (~1 µs)
    t_eq = 1e-6
    sol_eq = run_transient(
        x, y_ions_advanced, (0.0, t_eq), np.array([t_eq]),
        stack, illuminated=True, V_app=V_app, rtol=rtol, atol=atol,
        mat=mat,
    )
    if not sol_eq.success:
        # Ions advanced; keep previous carrier values (conservative fallback)
        return y_ions_advanced, True
    return sol_eq.y[:, -1], True
