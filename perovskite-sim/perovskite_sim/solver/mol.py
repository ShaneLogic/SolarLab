from __future__ import annotations
from dataclasses import dataclass
import warnings
import numpy as np
try:
    from scipy.integrate import solve_ivp
except ImportError:
    from scipy_shim import solve_ivp

from perovskite_sim.discretization.grid import multilayer_grid, Layer
from perovskite_sim.physics.poisson import (
    solve_poisson,
    solve_poisson_prefactored,
    factor_poisson,
    PoissonFactor,
)
from perovskite_sim.physics.continuity import carrier_continuity_rhs
from perovskite_sim.physics.ion_migration import ion_continuity_rhs
from perovskite_sim.physics.generation import beer_lambert_generation
from perovskite_sim.models.device import DeviceStack, electrical_layers

from perovskite_sim.physics.recombination import interface_recombination
from perovskite_sim.physics.temperature import (
    thermal_voltage, ni_at_T, mu_at_T, D_ion_at_T,
)
from perovskite_sim.models.mode import resolve_mode, SimulationMode
from perovskite_sim.constants import Q, V_T as _V_T_300


@dataclass
class StateVec:
    n: np.ndarray
    p: np.ndarray
    P: np.ndarray
    P_neg: np.ndarray | None = None

    @staticmethod
    def pack(n, p, P, P_neg=None) -> np.ndarray:
        parts = [n, p, P]
        if P_neg is not None:
            parts.append(P_neg)
        return np.concatenate(parts)

    @staticmethod
    def unpack(y: np.ndarray, N: int) -> "StateVec":
        if len(y) == 4 * N:
            return StateVec(
                n=y[:N], p=y[N:2*N], P=y[2*N:3*N], P_neg=y[3*N:4*N],
            )
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
    # Precomputed LAPACK LU of the Poisson tridiagonal operator; reused in
    # every RHS call in place of scipy.sparse spsolve, which is the largest
    # single contributor to assemble_rhs runtime at the default grid sizes.
    poisson_factor: PoissonFactor | None = None
    # Effective built-in potential computed from band offsets (or manual V_bi fallback)
    V_bi_eff: float = 1.1
    # Per-node Richardson constants for thermionic emission capping
    A_star_n: np.ndarray | None = None
    A_star_p: np.ndarray | None = None
    # Face indices where thermionic emission capping applies (band offset > threshold)
    interface_faces: tuple[int, ...] = ()
    # TMM-computed generation profile G(x) [m^-3 s^-1]; None = use Beer-Lambert
    G_optical: np.ndarray | None = None
    # Negative ion species arrays (dual-species mode when has_dual_ions is True)
    D_ion_neg_face: np.ndarray | None = None
    P_lim_neg_face: np.ndarray | None = None
    P_ion0_neg: np.ndarray | None = None
    P_lim_neg_node: np.ndarray | None = None
    has_dual_ions: bool = False
    # Temperature and thermal voltage (for T != 300 K)
    T_device: float = 300.0
    V_T_device: float = 0.025852  # kT/q at device temperature

    @property
    def carrier_params(self) -> dict:
        """Dict shape expected by `carrier_continuity_rhs` — legacy key names."""
        d = dict(
            D_n=self.D_n_face, D_p=self.D_p_face, V_T=self.V_T_device,
            ni_sq=self.ni_sq, tau_n=self.tau_n, tau_p=self.tau_p,
            n1=self.n1, p1=self.p1, B_rad=self.B_rad,
            C_n=self.C_n, C_p=self.C_p,
            chi=self.chi, Eg=self.Eg,
        )
        if self.interface_faces:
            d["interface_faces"] = list(self.interface_faces)
            d["A_star_n"] = self.A_star_n
            d["A_star_p"] = self.A_star_p
            d["T"] = self.T_device
        return d


def _compute_tmm_generation(
    x: np.ndarray,
    stack: DeviceStack,
    n_wavelengths: int = 200,
    lam_min: float = 300.0,
    lam_max: float = 1000.0,
) -> np.ndarray | None:
    """Compute TMM generation profile if any layer has optical material data.

    Returns G(x) [m^-3 s^-1] or None if no layers have optical data.
    The computation runs once during build_material_arrays and the result
    is cached in MaterialArrays.G_optical.

    TMM sees the *full* stack (including role=="substrate" layers) so the
    Fresnel chain and thin-film interference are correct. The electrical
    grid ``x`` only covers the non-substrate layers, so we shift it by the
    cumulative substrate thickness before querying ``tmm_generation``;
    ``tmm_absorption_profile`` routes each query to the right layer via
    ``layer_boundaries``.
    """
    has_optical = any(
        layer.params is not None and layer.params.optical_material is not None
        for layer in stack.layers
    )
    if not has_optical:
        return None

    from perovskite_sim.physics.optics import TMMLayer, tmm_generation
    from perovskite_sim.data import load_nk, load_am15g

    wavelengths_nm = np.linspace(lam_min, lam_max, n_wavelengths)
    wavelengths_m = wavelengths_nm * 1e-9

    # Load AM1.5G spectrum
    _, spectral_flux = load_am15g(wavelengths_nm)

    # Build TMM layers from the FULL stack (substrate included).
    tmm_layers: list[TMMLayer] = []
    for layer in stack.layers:
        p = layer.params
        if p.optical_material is not None:
            _, n_arr, k_arr = load_nk(p.optical_material, wavelengths_nm)
        elif p.n_optical is not None:
            # Constant refractive index, compute k from scalar alpha
            n_arr = np.full(n_wavelengths, p.n_optical)
            # k = alpha * lambda / (4 * pi)
            k_arr = p.alpha * wavelengths_m / (4.0 * np.pi)
        else:
            # Fallback: estimate n from eps_r, k from alpha
            n_arr = np.full(n_wavelengths, np.sqrt(p.eps_r))
            k_arr = p.alpha * wavelengths_m / (4.0 * np.pi)
        tmm_layers.append(
            TMMLayer(
                d=layer.thickness,
                n=n_arr,
                k=k_arr,
                incoherent=bool(getattr(p, "incoherent", False)),
            )
        )

    # Layer boundaries for spatial mapping (full stack).
    boundaries = np.zeros(len(stack.layers) + 1)
    for i, layer in enumerate(stack.layers):
        boundaries[i + 1] = boundaries[i] + layer.thickness

    # Shift electrical x by the cumulative substrate thickness so the
    # queries land inside the post-substrate thin films.
    substrate_offset = sum(
        l.thickness for l in stack.layers if l.role == "substrate"
    )
    x_tmm = x + substrate_offset

    G = tmm_generation(
        tmm_layers, wavelengths_m, spectral_flux, x_tmm, boundaries,
    )
    return G


def build_material_arrays(x: np.ndarray, stack: DeviceStack) -> MaterialArrays:
    """Construct the immutable per-experiment material array bundle.

    Single source of truth for per-node / per-face material arrays,
    contact equilibrium densities, dual-grid cell widths, and interface
    node indices. Built once per experiment and threaded through the hot
    path (assemble_rhs, _compute_current, interface recombination).
    """
    N = len(x)

    eps_r = np.ones(N)
    D_ion_node = np.zeros(N)
    P_lim_node = 1e30 * np.ones(N)
    P_ion0 = np.zeros(N)
    D_ion_neg_node = np.zeros(N)
    P_lim_neg_node = 1e30 * np.ones(N)
    P_ion0_neg = np.zeros(N)
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
    A_star_n_node = np.empty(N)
    A_star_p_node = np.empty(N)

    sim_mode = resolve_mode(getattr(stack, "mode", "full"))
    # Temperature scaling is only applied in modes that request it; other
    # modes see a fixed 300 K so that benchmarks and legacy configs stay
    # unaffected by an incidental ``T`` field in the YAML.
    if sim_mode.use_temperature_scaling:
        T_dev = stack.T
        V_T_dev = thermal_voltage(T_dev)
    else:
        T_dev = 300.0
        V_T_dev = _V_T_300

    elec_layers = electrical_layers(stack)

    offset = 0.0
    for layer in elec_layers:
        mask = (x >= offset - 1e-12) & (x <= offset + layer.thickness + 1e-12)
        p = layer.params
        eps_r[mask] = p.eps_r

        # Temperature-scaled ion diffusion
        D_ion_node[mask] = D_ion_at_T(p.D_ion, T_dev, p.E_a_ion)
        P_lim_node[mask] = p.P_lim
        P_ion0[mask] = p.P0
        if sim_mode.use_dual_ions:
            D_ion_neg_node[mask] = D_ion_at_T(p.D_ion_neg, T_dev, p.E_a_ion)
            P_lim_neg_node[mask] = p.P_lim_neg
            P_ion0_neg[mask] = p.P0_neg

        N_A[mask] = p.N_A
        N_D[mask] = p.N_D
        alpha[mask] = p.alpha
        chi[mask] = p.chi
        Eg[mask] = p.Eg

        # Temperature-scaled mobility → diffusion (Einstein: D = mu * V_T)
        mu_n_T = mu_at_T(p.mu_n, T_dev, p.mu_T_gamma)
        mu_p_T = mu_at_T(p.mu_p, T_dev, p.mu_T_gamma)
        D_n_node[mask] = mu_n_T * V_T_dev
        D_p_node[mask] = mu_p_T * V_T_dev

        # Temperature-scaled intrinsic density
        ni_T = ni_at_T(p.ni, p.Eg, T_dev, p.Nc300, p.Nv300)
        ni_sq[mask] = ni_T ** 2

        tau_n[mask] = p.tau_n
        tau_p[mask] = p.tau_p
        n1[mask] = p.n1
        p1[mask] = p.p1
        B_rad[mask] = p.B_rad
        C_n[mask] = p.C_n
        C_p[mask] = p.C_p
        A_star_n_node[mask] = p.A_star_n
        A_star_p_node[mask] = p.A_star_p

        # Spatially varying trap profile: tau(x) = tau_bulk * N_t_bulk / N_t(x)
        if (sim_mode.use_trap_profile
                and p.trap_N_t_interface is not None
                and p.trap_N_t_bulk is not None
                and p.trap_decay_length is not None):
            x_local = x[mask] - offset
            d_left = x_local
            d_right = layer.thickness - x_local
            L_d = p.trap_decay_length
            N_t_x = (p.trap_N_t_bulk
                      + (p.trap_N_t_interface - p.trap_N_t_bulk)
                      * (np.exp(-d_left / L_d) + np.exp(-d_right / L_d)))
            # Scale tau inversely with trap density
            ratio = p.trap_N_t_bulk / np.maximum(N_t_x, 1.0)
            tau_n[mask] *= ratio
            tau_p[mask] *= ratio

        offset += layer.thickness

    # Per-face diffusion via harmonic mean of adjacent nodal values.
    # Matches solve_poisson's eps_r treatment and the legacy
    # _build_carrier_params / _build_ion_face_params outputs exactly.
    D_n_face = 2.0 * D_n_node[:-1] * D_n_node[1:] / (D_n_node[:-1] + D_n_node[1:])
    D_p_face = 2.0 * D_p_node[:-1] * D_p_node[1:] / (D_p_node[:-1] + D_p_node[1:])
    D_ion_face = _harmonic_face_average(D_ion_node)
    P_lim_face = 0.5 * (P_lim_node[:-1] + P_lim_node[1:])
    D_ion_neg_face = _harmonic_face_average(D_ion_neg_node)
    P_lim_neg_face = 0.5 * (P_lim_neg_node[:-1] + P_lim_neg_node[1:])
    _has_dual_ions = np.any(D_ion_neg_node > 0.0)

    # Dual-grid cell widths for surface→volumetric conversion at interfaces.
    dx = np.diff(x)
    dx_cell = np.empty(N)
    dx_cell[0] = dx[0]
    dx_cell[-1] = dx[-1]
    dx_cell[1:-1] = 0.5 * (dx[:-1] + dx[1:])

    # Interface nodes: grid index closest to each internal interface.
    iface_list: list[int] = []
    offset = 0.0
    for layer in elec_layers[:-1]:
        offset += layer.thickness
        iface_list.append(int(np.argmin(np.abs(x - offset))))

    # Interface face indices where band offset exceeds the TE threshold.
    # A face index f corresponds to the interval between nodes f and f+1.
    # Legacy/fast modes skip this so the SG flux is never capped.
    TE_THRESHOLD = 0.05  # eV
    interface_face_list: list[int] = []
    if sim_mode.use_thermionic_emission:
        for idx in iface_list:
            if idx > 0 and idx < N - 1:
                delta_Ec = abs(chi[idx] - chi[idx - 1])
                delta_Ev = abs((chi[idx - 1] + Eg[idx - 1]) - (chi[idx] + Eg[idx]))
                if delta_Ec > TE_THRESHOLD or delta_Ev > TE_THRESHOLD:
                    interface_face_list.append(idx - 1)

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

    first = elec_layers[0].params
    last = elec_layers[-1].params
    n_L, p_L = _equilibrium_np(first.N_D, first.N_A, first.ni)
    n_R, p_R = _equilibrium_np(last.N_D, last.N_A, last.ni)

    # LAPACK LU of the Poisson tridiagonal — constant across the experiment,
    # so we pay the factor cost exactly once and each RHS call becomes a
    # single dgttrs back-substitution.
    poisson_factor = factor_poisson(x, eps_r)

    V_bi_eff = stack.compute_V_bi()

    # TMM optical generation: computed when any layer has optical data and
    # the active mode enables TMM. Legacy/fast fall back to Beer-Lambert.
    if sim_mode.use_tmm_optics:
        G_optical = _compute_tmm_generation(x, stack)
    else:
        G_optical = None

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
        poisson_factor=poisson_factor,
        V_bi_eff=V_bi_eff,
        A_star_n=A_star_n_node,
        A_star_p=A_star_p_node,
        interface_faces=tuple(interface_face_list),
        G_optical=G_optical,
        D_ion_neg_face=D_ion_neg_face if _has_dual_ions else None,
        P_lim_neg_face=P_lim_neg_face if _has_dual_ions else None,
        P_ion0_neg=P_ion0_neg if _has_dual_ions else None,
        P_lim_neg_node=P_lim_neg_node if _has_dual_ions else None,
        has_dual_ions=bool(_has_dual_ions),
        T_device=T_dev,
        V_T_device=V_T_dev,
    )


def _harmonic_face_average(values: np.ndarray) -> np.ndarray:
    """Harmonic mean on inter-node faces, with zero conductivity preserved."""
    numer = 2.0 * values[:-1] * values[1:]
    denom = values[:-1] + values[1:]
    return np.divide(numer, denom, out=np.zeros_like(numer), where=denom > 0.0)


def _charge_density(
    p: np.ndarray,
    n: np.ndarray,
    P_ion: np.ndarray,
    P_ion0: np.ndarray,
    N_A: np.ndarray,
    N_D: np.ndarray,
    P_neg: np.ndarray | None = None,
    P_neg0: np.ndarray | None = None,
) -> np.ndarray:
    """Space-charge density with ionic charge measured relative to neutral background.

    Positive species contribute +(P_ion - P_ion0), negative species contribute
    -(P_neg - P_neg0). When P_neg is None, single-species mode.
    """
    rho = Q * (p - n + (P_ion - P_ion0) - N_A + N_D)
    if P_neg is not None and P_neg0 is not None:
        rho -= Q * (P_neg - P_neg0)
    return rho


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
    rho = _charge_density(
        p, n, sv.P, mat.P_ion0, mat.N_A, mat.N_D,
        P_neg=sv.P_neg, P_neg0=mat.P_ion0_neg,
    )
    phi = solve_poisson_prefactored(
        mat.poisson_factor, rho, phi_left=0.0, phi_right=stack.V_bi - V_app,
    )

    # Generation: TMM-computed profile if available, else Beer-Lambert fallback
    if illuminated:
        if mat.G_optical is not None:
            G = mat.G_optical
        else:
            G = beer_lambert_generation(x, mat.alpha, stack.Phi)
    else:
        G = np.zeros(N)

    # Carrier continuity with per-layer D and per-node recombination params
    dn, dp = carrier_continuity_rhs(x, phi, n, p, G, mat.carrier_params)

    # Interface recombination (surface SRH at heterointerfaces)
    _apply_interface_recombination(dn, dp, n, p, stack, mat)

    # Ion continuity using per-face transport coefficients so ions remain
    # confined to ion-conducting layers.
    dP = ion_continuity_rhs(x, phi, sv.P, mat.D_ion_face, mat.V_T_device, mat.P_lim_face)

    # Negative ion continuity (dual-species mode)
    dP_neg = None
    if mat.has_dual_ions and sv.P_neg is not None:
        from perovskite_sim.physics.ion_migration import ion_continuity_rhs_neg
        dP_neg = ion_continuity_rhs_neg(
            x, phi, sv.P_neg, mat.D_ion_neg_face, mat.V_T_device, mat.P_lim_neg_face,
        )

    # Enforce Dirichlet BCs: hold boundary nodes fixed
    dn[0] = dn[-1] = 0.0
    dp[0] = dp[-1] = 0.0

    return StateVec.pack(dn, dp, dP, dP_neg)


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

    _clip_count = [0]

    dual = mat.has_dual_ions and sv.P_neg is not None

    if dual:
        from perovskite_sim.physics.ion_migration import ion_continuity_rhs_neg

        P_pos_init = np.maximum(sv.P, 0.0)
        P_neg_init = np.maximum(sv.P_neg, 0.0)
        y_ion0 = np.concatenate([P_pos_init, P_neg_init])

        def _ion_rhs(t, y_ion):
            P_pos = np.maximum(y_ion[:N], 0.0)
            P_neg_v = np.maximum(y_ion[N:], 0.0)
            if np.any(y_ion < -1e-30):
                _clip_count[0] += 1
            rho = _charge_density(
                p_frozen, n_frozen, P_pos, mat.P_ion0, mat.N_A, mat.N_D,
                P_neg=P_neg_v, P_neg0=mat.P_ion0_neg,
            )
            phi = solve_poisson_prefactored(
                mat.poisson_factor, rho,
                phi_left=0.0, phi_right=stack.V_bi - V_app,
            )
            dP_pos = ion_continuity_rhs(
                x, phi, P_pos, mat.D_ion_face, mat.V_T_device, mat.P_lim_face,
            )
            dP_neg = ion_continuity_rhs_neg(
                x, phi, P_neg_v, mat.D_ion_neg_face, mat.V_T_device, mat.P_lim_neg_face,
            )
            return np.concatenate([dP_pos, dP_neg])

        sol_ion = solve_ivp(
            _ion_rhs, (0.0, dt), y_ion0, t_eval=[dt],
            method="Radau", rtol=rtol, atol=atol,
        )
    else:
        P_init = np.maximum(sv.P, 0.0)
        if np.any(sv.P < -1e-30):
            _clip_count[0] = 1

        def _ion_rhs(t, P):
            if np.any(P < -1e-30):
                _clip_count[0] += 1
            P_nn = np.maximum(P, 0.0)
            rho = _charge_density(
                p_frozen, n_frozen, P_nn, mat.P_ion0, mat.N_A, mat.N_D,
            )
            phi = solve_poisson_prefactored(
                mat.poisson_factor, rho,
                phi_left=0.0, phi_right=stack.V_bi - V_app,
            )
            return ion_continuity_rhs(
                x, phi, P_nn, mat.D_ion_face, mat.V_T_device, mat.P_lim_face,
            )

        sol_ion = solve_ivp(
            _ion_rhs, (0.0, dt), P_init, t_eval=[dt],
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

    if dual:
        P_new = np.clip(sol_ion.y[:N, -1], 0.0, mat.P_lim_node)
        P_neg_new = np.clip(sol_ion.y[N:, -1], 0.0, mat.P_lim_neg_node)
        y_ions_advanced = StateVec.pack(sv.n, sv.p, P_new, P_neg_new)
    else:
        P_new = np.clip(sol_ion.y[:, -1], 0.0, mat.P_lim_node)
        P_neg_carry = sv.P_neg  # None in single-species mode
        y_ions_advanced = StateVec.pack(sv.n, sv.p, P_new, P_neg_carry)

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
