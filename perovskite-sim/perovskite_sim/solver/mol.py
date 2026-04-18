from __future__ import annotations
from dataclasses import dataclass
import warnings
import numpy as np
try:
    from scipy.integrate import solve_ivp
except ImportError:
    from perovskite_sim._compat.scipy_shim import solve_ivp

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
from perovskite_sim.models.device import (
    DeviceStack, electrical_layers, electrical_interfaces,
)

from perovskite_sim.physics.recombination import interface_recombination
from perovskite_sim.physics.temperature import (
    thermal_voltage, ni_at_T, mu_at_T, D_ion_at_T, B_rad_at_T, eg_at_T,
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
    # Field-dependent mobility (Phase 3.2 — Apr 2026). Per-face arrays
    # derived from the per-node v_sat / β / γ values via simple averaging.
    # ``has_field_mobility`` is True iff at least one layer sets
    # a nonzero v_sat_{n,p} or pf_gamma_{n,p}; otherwise the field-mobility
    # hook is skipped in assemble_rhs and the cached D_n_face / D_p_face
    # are used as-is (bit-identical to pre-3.2).
    v_sat_n_face: np.ndarray | None = None
    v_sat_p_face: np.ndarray | None = None
    ct_beta_n_face: np.ndarray | None = None
    ct_beta_p_face: np.ndarray | None = None
    pf_gamma_n_face: np.ndarray | None = None
    pf_gamma_p_face: np.ndarray | None = None
    has_field_mobility: bool = False
    # Selective / Schottky outer-contact surface recombination velocities
    # (Phase 3.3 — Apr 2026). ``has_selective_contacts`` is True iff the
    # active mode enables it AND the stack config supplies at least one
    # finite ``S_*``. When False the Dirichlet pin remains in force and
    # the boundary node is overwritten to ``n_L`` / ``n_R`` in
    # ``assemble_rhs`` — i.e. the pre-3.3 behaviour, bit-identical.
    # When True the four S values below are used as Robin coefficients
    # in :func:`physics.contacts.apply_selective_contacts` and the
    # boundary node is allowed to evolve freely.
    S_n_L: float | None = None
    S_p_L: float | None = None
    S_n_R: float | None = None
    S_p_R: float | None = None
    has_selective_contacts: bool = False
    # Self-consistent radiative reabsorption (Phase 3.1b — Apr 2026).
    # When ``has_radiative_reabsorption`` is True, ``build_material_arrays``
    # skips the Phase 3.1 build-time ``B_rad *= P_esc`` scaling and instead
    # stashes per-absorber (mask, P_esc, thickness) entries here. At each
    # RHS call :func:`assemble_rhs` integrates the full bulk emission rate
    # ``B_rad · n · p`` across each absorber and adds the reabsorbed
    # fraction ``(1 − P_esc)`` back as a uniform G_rad source on that
    # absorber's nodes. This closes the photon-recycling loop: the bulk
    # ``B_rad`` on ``MaterialArrays`` remains at its intrinsic value and
    # only the net radiative loss (what actually escapes) appears in
    # steady-state. In the spatially-uniform n·p limit this is exactly
    # equivalent to the Phase 3.1 build-time scaling; in the non-uniform
    # regime (e.g. under injection or near contacts) it is physically more
    # accurate because it lets emission near a high-n·p region feed
    # generation near a low-n·p region.
    absorber_masks: tuple[np.ndarray, ...] = ()
    absorber_p_esc: tuple[float, ...] = ()
    absorber_thicknesses: tuple[float, ...] = ()
    has_radiative_reabsorption: bool = False
    # Spatially resolved trap density N_t(x) [m⁻³] (Phase 4a — Apr 2026).
    # Populated from the per-layer (trap_N_t_interface, trap_N_t_bulk,
    # trap_decay_length, trap_profile_shape) parameters when
    # ``SimulationMode.use_trap_profile`` is on. Layers that do not opt
    # in keep their bulk ``trap_N_t_bulk`` value (or 0.0 if unset).
    # Stored for diagnostics and so downstream physics (interface SRH,
    # band-tail absorption, mobility degradation) can be layered on top
    # of the same per-node trap density without re-computing the
    # profile.
    N_t_node: np.ndarray | None = None
    has_trap_profile: bool = False

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
    return_absorbance: bool = False,
):
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

    When ``return_absorbance`` is True the absorption profile ``A(x, λ)``
    and the per-layer refractive-index arrays are returned alongside
    ``G`` so that callers (e.g. the photon-recycling layer) can compute
    per-layer escape probabilities without rebuilding the TMM stack.
    """
    has_optical = any(
        layer.params is not None and layer.params.optical_material is not None
        for layer in stack.layers
    )
    if not has_optical:
        if return_absorbance:
            return None, None
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
                incoherent=bool(p.incoherent),
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

    if return_absorbance:
        G, A_xl = tmm_generation(
            tmm_layers, wavelengths_m, spectral_flux, x_tmm, boundaries,
            return_absorbance=True,
        )
        tmm_info = {
            "wavelengths_m": wavelengths_m,
            "tmm_layers": tmm_layers,
            "boundaries": boundaries,
            "A_xl": A_xl,
            "substrate_offset": substrate_offset,
        }
        return G, tmm_info
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
    # Field-dependent mobility parameters per node (Phase 3.2).
    v_sat_n_node = np.zeros(N)
    v_sat_p_node = np.zeros(N)
    ct_beta_n_node = np.full(N, 2.0)
    ct_beta_p_node = np.full(N, 2.0)
    pf_gamma_n_node = np.zeros(N)
    pf_gamma_p_node = np.zeros(N)
    # Trap profile (Phase 4a). Per-node trap density [m⁻³]; zeros outside
    # any layer that opts in. Filled as layer params are iterated below.
    N_t_node_arr = np.zeros(N)
    _has_trap_profile = False

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
    if len(elec_layers) == 0:
        raise ValueError(
            "stack has no electrical layers (all layers have role:substrate)"
        )

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

        # Phase 4b: temperature-shifted bandgap via Varshni, feeding both
        # the chi/Eg arrays (so downstream thermionic-emission, photon-
        # recycling, and band-offset consumers see the shifted edges) and
        # the ni computation (so ni² ∝ exp(-Eg(T)/kT) is self-consistent).
        # When ``use_temperature_scaling`` is off or varshni_alpha = 0,
        # Eg_T ≡ p.Eg and behaviour is bit-identical to pre-Phase-4b.
        if sim_mode.use_temperature_scaling:
            Eg_T = eg_at_T(p.Eg, T_dev, p.varshni_alpha, p.varshni_beta)
        else:
            Eg_T = p.Eg
        chi[mask] = p.chi
        Eg[mask] = Eg_T

        # Temperature-scaled mobility → diffusion (Einstein: D = mu * V_T)
        mu_n_T = mu_at_T(p.mu_n, T_dev, p.mu_T_gamma)
        mu_p_T = mu_at_T(p.mu_p, T_dev, p.mu_T_gamma)
        D_n_node[mask] = mu_n_T * V_T_dev
        D_p_node[mask] = mu_p_T * V_T_dev

        # Temperature-scaled intrinsic density. Uses the Varshni-shifted
        # Eg_T so ni(T) is self-consistent with the shifted bandgap when
        # the user opts in; when varshni_alpha=0, Eg_T == p.Eg and the
        # result is identical to the Phase 4 path.
        ni_T = ni_at_T(p.ni, Eg_T, T_dev, p.Nc300, p.Nv300)
        ni_sq[mask] = ni_T ** 2

        tau_n[mask] = p.tau_n
        tau_p[mask] = p.tau_p
        n1[mask] = p.n1
        p1[mask] = p.p1
        # Phase 4b: temperature-scaled radiative coefficient. gamma=0
        # (the default) short-circuits to B_300 so pre-Phase-4b configs
        # are unaffected.
        if sim_mode.use_temperature_scaling:
            B_rad[mask] = B_rad_at_T(p.B_rad, T_dev, p.B_rad_T_gamma)
        else:
            B_rad[mask] = p.B_rad
        C_n[mask] = p.C_n
        C_p[mask] = p.C_p
        A_star_n_node[mask] = p.A_star_n
        A_star_p_node[mask] = p.A_star_p

        # Field-dependent mobility parameters (Phase 3.2). Copied verbatim
        # from the layer so per-node arrays can average to face values.
        v_sat_n_node[mask] = p.v_sat_n
        v_sat_p_node[mask] = p.v_sat_p
        ct_beta_n_node[mask] = p.ct_beta_n
        ct_beta_p_node[mask] = p.ct_beta_p
        pf_gamma_n_node[mask] = p.pf_gamma_n
        pf_gamma_p_node[mask] = p.pf_gamma_p

        # Spatially varying trap profile (Phase 4a). The shape and
        # primitives live in physics/traps.py so that the same N_t(x)
        # construction can be reused by future hooks (interface SRH,
        # band-tail absorption, mobility degradation). Two profile
        # shapes are accepted: "exponential" (default — the original
        # Phase 4 form) and "gaussian" (faster decay into the bulk for
        # well-defined defect slabs).
        from perovskite_sim.physics.traps import (
            exponential_edge_profile,
            gaussian_edge_profile,
            tau_from_trap_density,
            has_trap_profile_params,
        )
        if sim_mode.use_trap_profile and has_trap_profile_params(p):
            x_local = x[mask] - offset
            shape = getattr(p, "trap_profile_shape", "exponential") or "exponential"
            if str(shape).lower() == "gaussian":
                N_t_x = gaussian_edge_profile(
                    x_local,
                    layer.thickness,
                    float(p.trap_N_t_interface),
                    float(p.trap_N_t_bulk),
                    float(p.trap_decay_length),
                )
            else:
                N_t_x = exponential_edge_profile(
                    x_local,
                    layer.thickness,
                    float(p.trap_N_t_interface),
                    float(p.trap_N_t_bulk),
                    float(p.trap_decay_length),
                )
            # Cache N_t(x) on the per-node array for diagnostics and
            # downstream physics; map tau via the SRH inverse-density
            # rule so deep bulk recovers the layer's clean tau exactly.
            N_t_node_arr[mask] = N_t_x
            tau_n[mask] = tau_from_trap_density(
                tau_n[mask], N_t_x, float(p.trap_N_t_bulk),
            )
            tau_p[mask] = tau_from_trap_density(
                tau_p[mask], N_t_x, float(p.trap_N_t_bulk),
            )
            _has_trap_profile = True
        elif p.trap_N_t_bulk is not None:
            # Layer specifies a bulk density but no profile — record the
            # uniform value so the diagnostics array still reflects it.
            N_t_node_arr[mask] = float(p.trap_N_t_bulk)

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

    # Field-dependent mobility: linear face averages of the per-node
    # parameters. The CT / PF models handle v_sat = 0 and γ = 0 as "off"
    # at the face level, so heterointerfaces with one side opted-in and
    # the other opted-out end up with a half-strength face parameter —
    # still physically reasonable for a transition region.
    v_sat_n_face = 0.5 * (v_sat_n_node[:-1] + v_sat_n_node[1:])
    v_sat_p_face = 0.5 * (v_sat_p_node[:-1] + v_sat_p_node[1:])
    ct_beta_n_face = 0.5 * (ct_beta_n_node[:-1] + ct_beta_n_node[1:])
    ct_beta_p_face = 0.5 * (ct_beta_p_node[:-1] + ct_beta_p_node[1:])
    pf_gamma_n_face = 0.5 * (pf_gamma_n_node[:-1] + pf_gamma_n_node[1:])
    pf_gamma_p_face = 0.5 * (pf_gamma_p_node[:-1] + pf_gamma_p_node[1:])
    _has_field_mobility = bool(
        sim_mode.use_field_dependent_mobility
        and (
            np.any(v_sat_n_face > 0.0)
            or np.any(v_sat_p_face > 0.0)
            or np.any(pf_gamma_n_face > 0.0)
            or np.any(pf_gamma_p_face > 0.0)
        )
    )

    # Selective / Schottky outer contact Robin BCs (Phase 3.3). Gated by the
    # active mode AND the stack supplying at least one finite S_* value.
    # When inactive the flag stays False and the Dirichlet pin remains the
    # boundary treatment in assemble_rhs — bit-identical to pre-3.3.
    _has_selective_contacts = bool(
        sim_mode.use_selective_contacts
        and (
            stack.S_n_left is not None
            or stack.S_p_left is not None
            or stack.S_n_right is not None
            or stack.S_p_right is not None
        )
    )

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
    # When photon recycling is also active we capture the spectral
    # absorbance A(x, λ) returned by TMM so we can derive a per-absorber
    # escape probability and scale B_rad by (1 − P_esc) in place.
    if sim_mode.use_tmm_optics:
        if sim_mode.use_photon_recycling:
            G_optical, tmm_info = _compute_tmm_generation(
                x, stack, return_absorbance=True,
            )
        else:
            G_optical = _compute_tmm_generation(x, stack)
            tmm_info = None
    else:
        G_optical = None
        tmm_info = None

    # Photon recycling — two branches that share the per-absorber
    # (mask, P_esc, thickness) derivation from TMM:
    #
    # Phase 3.1 (use_photon_recycling ON, use_radiative_reabsorption OFF):
    #   scale B_rad(x) by P_esc on each absorber at build time. This
    #   collapses the reabsorption source directly into a reduced radiative
    #   loss coefficient — cheap and correct in the spatially-uniform n·p
    #   limit (which is where "V_oc boost" regressions are measured).
    #
    # Phase 3.1b (use_radiative_reabsorption ON, which also requires
    # use_photon_recycling ON):
    #   leave B_rad at its intrinsic bulk value and instead cache the
    #   per-absorber (mask, P_esc, thickness) tuples. assemble_rhs then
    #   computes the per-RHS G_rad = (1 − P_esc) · <B · n · p>_abs /
    #   thickness and adds it uniformly onto the absorber nodes of G,
    #   closing the photon-recycling loop self-consistently.
    absorber_masks_list: list[np.ndarray] = []
    absorber_p_esc_list: list[float] = []
    absorber_thicknesses_list: list[float] = []
    _has_radiative_reabsorption = False
    if (
        sim_mode.use_photon_recycling
        and tmm_info is not None
        and G_optical is not None
    ):
        from perovskite_sim.physics.photon_recycling import (
            compute_p_esc,
            wavelength_at_gap,
        )
        wavelengths_m = tmm_info["wavelengths_m"]
        tmm_layers_full = tmm_info["tmm_layers"]
        # Absorber layers live inside the electrical stack; TMM indexes
        # them against the *full* stack, so shift by the substrate prefix
        # count to get the TMM layer index.
        substrate_prefix = 0
        for lyr in stack.layers:
            if lyr.role == "substrate":
                substrate_prefix += 1
            else:
                break

        offset_pr = 0.0
        for i_elec, layer in enumerate(elec_layers):
            if layer.role == "absorber":
                p = layer.params
                Eg_eV = float(p.Eg) if p is not None else 0.0
                if Eg_eV > 0.0:
                    mask_abs = (
                        (x >= offset_pr - 1e-12)
                        & (x <= offset_pr + layer.thickness + 1e-12)
                    )
                    if np.any(mask_abs):
                        lam_gap = wavelength_at_gap(Eg_eV)
                        tmm_idx = i_elec + substrate_prefix
                        n_lambda = tmm_layers_full[tmm_idx].n
                        k_lambda = tmm_layers_full[tmm_idx].k
                        n_at_gap = float(
                            np.interp(lam_gap, wavelengths_m, n_lambda)
                        )
                        k_at_gap = float(
                            np.interp(lam_gap, wavelengths_m, k_lambda)
                        )
                        # α = 4π k / λ  [m⁻¹]
                        alpha_gap = 4.0 * np.pi * k_at_gap / lam_gap
                        P_esc = compute_p_esc(
                            alpha_gap=alpha_gap,
                            thickness=float(layer.thickness),
                            n_at_gap=n_at_gap,
                        )
                        if sim_mode.use_radiative_reabsorption:
                            # Phase 3.1b: keep full B_rad, cache the
                            # absorber geometry for the per-RHS source.
                            absorber_masks_list.append(mask_abs)
                            absorber_p_esc_list.append(float(P_esc))
                            absorber_thicknesses_list.append(
                                float(layer.thickness)
                            )
                            _has_radiative_reabsorption = True
                        else:
                            # Phase 3.1: collapse reabsorption into a
                            # reduced net radiative loss coefficient.
                            # Only the fraction P_esc of emitted photons
                            # leaves the device; the rest is reabsorbed
                            # and regenerates carriers, so the net
                            # radiative loss rate scales by P_esc.
                            # (Yablonovitch 1982; see
                            # physics/photon_recycling.py for details.)
                            B_rad[mask_abs] = B_rad[mask_abs] * P_esc
            offset_pr += layer.thickness

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
        v_sat_n_face=v_sat_n_face if _has_field_mobility else None,
        v_sat_p_face=v_sat_p_face if _has_field_mobility else None,
        ct_beta_n_face=ct_beta_n_face if _has_field_mobility else None,
        ct_beta_p_face=ct_beta_p_face if _has_field_mobility else None,
        pf_gamma_n_face=pf_gamma_n_face if _has_field_mobility else None,
        pf_gamma_p_face=pf_gamma_p_face if _has_field_mobility else None,
        has_field_mobility=_has_field_mobility,
        S_n_L=stack.S_n_left if _has_selective_contacts else None,
        S_p_L=stack.S_p_left if _has_selective_contacts else None,
        S_n_R=stack.S_n_right if _has_selective_contacts else None,
        S_p_R=stack.S_p_right if _has_selective_contacts else None,
        has_selective_contacts=_has_selective_contacts,
        absorber_masks=tuple(absorber_masks_list) if _has_radiative_reabsorption else (),
        absorber_p_esc=tuple(absorber_p_esc_list) if _has_radiative_reabsorption else (),
        absorber_thicknesses=tuple(absorber_thicknesses_list) if _has_radiative_reabsorption else (),
        has_radiative_reabsorption=_has_radiative_reabsorption,
        N_t_node=N_t_node_arr,
        has_trap_profile=_has_trap_profile,
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
    ifaces = electrical_interfaces(stack)
    if not ifaces:
        return
    for k, idx in enumerate(mat.interface_nodes):
        if k >= len(ifaces):
            break
        v_n, v_p = ifaces[k]
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

    # Boundary conditions from cached equilibrium densities. For selective /
    # Schottky contacts (Phase 3.3) the carriers/sides with a finite S are
    # allowed to evolve freely; only the Dirichlet sides still get pinned
    # to the equilibrium value. When has_selective_contacts is False this
    # reduces exactly to the pre-3.3 pin of all four boundary entries.
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

    # Self-consistent radiative reabsorption (Phase 3.1b). Per absorber,
    # integrate the bulk radiative emission rate R_rad = B·n·p across its
    # nodes and add (1 − P_esc) · <R_rad>_abs back as a uniform G_rad on
    # those nodes. This closes the photon-recycling loop without breaking
    # any previous invariant: in the uniform-n·p limit the time-averaged
    # effect is bit-equivalent to the Phase 3.1 B_rad *= P_esc scaling,
    # which is what the regression uses to check monotonicity. We always
    # copy G here (even in the dark V_oc sweep) so we never mutate the
    # cached mat.G_optical — that array is shared across every RHS call in
    # the experiment and must stay read-only.
    if mat.has_radiative_reabsorption and mat.absorber_masks:
        G = G.copy() if G.size else G
        for mask, P_esc, thickness in zip(
            mat.absorber_masks, mat.absorber_p_esc, mat.absorber_thicknesses
        ):
            if thickness <= 0.0 or P_esc >= 1.0:
                continue
            # Integrate B_rad · n · p across the absorber. Trapezoidal on
            # the electrical grid is plenty — the absorber span is always
            # resolved with ≥ 20 nodes in production configs.
            emission = mat.B_rad[mask] * n[mask] * p[mask]
            x_abs = x[mask]
            if x_abs.size < 2:
                continue
            R_tot = float(np.trapezoid(emission, x_abs))
            if R_tot <= 0.0:
                continue
            # Reabsorbed fraction is redistributed uniformly across the
            # absorber. Spatial redistribution of G matters only when n·p
            # is non-uniform (e.g. under strong injection).
            G_rad = R_tot * (1.0 - P_esc) / thickness
            G[mask] = G[mask] + G_rad

    # Carrier continuity with per-layer D and per-node recombination params.
    # If any layer enables field-dependent mobility (Caughey-Thomas or
    # Poole-Frenkel), the baseline D_n_face / D_p_face are overridden on a
    # per-RHS basis from the Poisson-computed face field. This is the one
    # path that intentionally breaks the "build once, reuse" invariant of
    # MaterialArrays — it is unavoidable because μ(E) depends on the state.
    if mat.has_field_mobility:
        from perovskite_sim.physics.field_mobility import apply_field_mobility
        dx_loc = np.diff(x)
        E_face = -(phi[1:] - phi[:-1]) / dx_loc
        mu_n_face_base = mat.D_n_face / mat.V_T_device
        mu_p_face_base = mat.D_p_face / mat.V_T_device
        mu_n_face_eff = apply_field_mobility(
            mu_n_face_base,
            E_face,
            mat.v_sat_n_face,
            mat.ct_beta_n_face,
            mat.pf_gamma_n_face,
        )
        mu_p_face_eff = apply_field_mobility(
            mu_p_face_base,
            E_face,
            mat.v_sat_p_face,
            mat.ct_beta_p_face,
            mat.pf_gamma_p_face,
        )
        carrier_params = dict(mat.carrier_params)
        carrier_params["D_n"] = mu_n_face_eff * mat.V_T_device
        carrier_params["D_p"] = mu_p_face_eff * mat.V_T_device
    else:
        carrier_params = mat.carrier_params

    # Selective / Schottky outer contact Robin BCs (Phase 3.3). Compute one
    # Robin flux per side/carrier that has a finite S. Carriers/sides left
    # as None remain Dirichlet-pinned by carrier_continuity_rhs. When the
    # flag is off this block is skipped entirely and the padded zero flux
    # survives — bit-identical to pre-3.3.
    if mat.has_selective_contacts:
        from perovskite_sim.physics.contacts import selective_contact_flux
        if carrier_params is mat.carrier_params:
            carrier_params = dict(mat.carrier_params)
        if mat.S_n_L is not None:
            carrier_params["J_n_L"] = selective_contact_flux(
                float(n[0]), mat.n_L, mat.S_n_L, carrier="n", side="left",
            )
        if mat.S_n_R is not None:
            carrier_params["J_n_R"] = selective_contact_flux(
                float(n[-1]), mat.n_R, mat.S_n_R, carrier="n", side="right",
            )
        if mat.S_p_L is not None:
            carrier_params["J_p_L"] = selective_contact_flux(
                float(p[0]), mat.p_L, mat.S_p_L, carrier="p", side="left",
            )
        if mat.S_p_R is not None:
            carrier_params["J_p_R"] = selective_contact_flux(
                float(p[-1]), mat.p_R, mat.S_p_R, carrier="p", side="right",
            )
    dn, dp = carrier_continuity_rhs(x, phi, n, p, G, carrier_params)

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

    # Enforce Dirichlet BCs: hold boundary nodes fixed. With selective /
    # Schottky contacts (Phase 3.3) the carrier_continuity_rhs helper does
    # the pinning itself on a per-side / per-carrier basis (Robin sides
    # are deliberately left free to evolve), so only apply the legacy
    # blanket pin when every contact is ohmic.
    if not mat.has_selective_contacts:
        dn[0] = dn[-1] = 0.0
        dp[0] = dp[-1] = 0.0
    else:
        if mat.S_n_L is None:
            dn[0] = 0.0
        if mat.S_n_R is None:
            dn[-1] = 0.0
        if mat.S_p_L is None:
            dp[0] = 0.0
        if mat.S_p_R is None:
            dp[-1] = 0.0

    if _RHS_FINITE_CHECK:
        _assert_finite_rhs(dn, dp, dP, dP_neg, V_app)

    return StateVec.pack(dn, dp, dP, dP_neg)


# Debug guard: set PEROVSKITE_RHS_FINITE_CHECK=1 to raise _RhsNonFinite when
# any RHS component contains NaN/Inf. Off by default so the hot path is
# untouched in production; used by regression tests to catch silent state
# corruption (e.g. the substrate-stack Radau hang would have surfaced in
# seconds instead of 12 minutes under this guard).
import os as _os
_RHS_FINITE_CHECK = _os.environ.get("PEROVSKITE_RHS_FINITE_CHECK", "0") == "1"


class _RhsNonFinite(Exception):
    """Raised by assemble_rhs when its output contains NaN or Inf."""


def _assert_finite_rhs(dn, dp, dP, dP_neg, V_app: float) -> None:
    for name, arr in (("dn", dn), ("dp", dp), ("dP", dP), ("dP_neg", dP_neg)):
        if arr is None:
            continue
        if not np.all(np.isfinite(arr)):
            raise _RhsNonFinite(
                f"non-finite {name} at V_app={V_app:.4f}: "
                f"nan={int(np.sum(np.isnan(arr)))}, inf={int(np.sum(np.isinf(arr)))}"
            )


class _NfevExceeded(Exception):
    """Raised by the RHS wrapper when max_nfev is reached."""


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
    max_nfev: int | None = None,
):
    """Integrate MOL system from t_span[0] to t_span[1].

    If `mat` is None, the material cache is built locally — convenient for
    one-off calls but wasteful when this function is invoked many times
    with the same stack (J-V sweeps, impedance frequency loops). Callers
    that loop should build `mat` once and pass it in.

    If `max_nfev` is set, the RHS is wrapped with a call counter and aborts
    once that many evaluations have been performed. On abort the returned
    sol object has ``success=False`` so callers that handle non-convergence
    (e.g. the bisection in ``jv_sweep._integrate_step``) can take over.
    Without this guard, Radau can spin in its implicit iteration on nearly
    singular Jacobians without any wall-time bound — the canonical failure
    is the reverse leg of a JV sweep on ionmonger_benchmark at N_grid=60
    under single-threaded BLAS (commit history: substrate-stack regression).
    """
    if mat is None:
        mat = build_material_arrays(x, stack)

    if max_nfev is None and not _RHS_FINITE_CHECK:
        def rhs(t, y):
            return assemble_rhs(t, y, x, stack, mat, illuminated, V_app)

        return solve_ivp(rhs, t_span, y0, t_eval=t_eval,
                         method="Radau", rtol=rtol, atol=atol,
                         dense_output=False, max_step=max_step)

    counter = [0]

    def rhs(t, y):
        counter[0] += 1
        if max_nfev is not None and counter[0] > max_nfev:
            raise _NfevExceeded
        return assemble_rhs(t, y, x, stack, mat, illuminated, V_app)

    try:
        return solve_ivp(rhs, t_span, y0, t_eval=t_eval,
                         method="Radau", rtol=rtol, atol=atol,
                         dense_output=False, max_step=max_step)
    except _NfevExceeded:
        from types import SimpleNamespace
        return SimpleNamespace(
            success=False,
            y=np.empty((y0.size, 0)),
            t=np.empty(0),
            message=f"max_nfev={max_nfev} exceeded (actual nfev={counter[0]})",
            nfev=counter[0],
            status=-1,
        )
    except _RhsNonFinite as e:
        from types import SimpleNamespace
        return SimpleNamespace(
            success=False,
            y=np.empty((y0.size, 0)),
            t=np.empty(0),
            message=f"RHS returned non-finite values: {e}",
            nfev=counter[0],
            status=-2,
        )


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

    # Apply ohmic contact BCs to frozen carrier arrays. Selective / Schottky
    # contacts (Phase 3.3) leave the Robin sides free — the current state
    # is whatever the main RHS integration produced — so only the ohmic
    # sides are pinned here.
    n_frozen = sv.n.copy()
    p_frozen = sv.p.copy()
    if not mat.has_selective_contacts:
        n_frozen[0] = mat.n_L; n_frozen[-1] = mat.n_R
        p_frozen[0] = mat.p_L; p_frozen[-1] = mat.p_R
    else:
        if mat.S_n_L is None:
            n_frozen[0] = mat.n_L
        if mat.S_n_R is None:
            n_frozen[-1] = mat.n_R
        if mat.S_p_L is None:
            p_frozen[0] = mat.p_L
        if mat.S_p_R is None:
            p_frozen[-1] = mat.p_R

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
