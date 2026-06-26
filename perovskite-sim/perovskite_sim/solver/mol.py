from __future__ import annotations
from dataclasses import dataclass
import math
import os
import warnings
import numpy as np
try:
    from scipy.integrate import solve_ivp
except ImportError:
    from perovskite_sim._compat.scipy_shim import solve_ivp

from perovskite_sim._compat.numpy_compat import trapezoid
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
    electrical_interface_defects,
)

from perovskite_sim.physics.recombination import interface_recombination
from perovskite_sim.physics.interface_plane import (
    build_plane_params,
    solve_plane_densities,
)
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
    # Phase E3 — optional interface-plane state block. Shape (4*N_iface,)
    # carrying (n_1s, p_1s, n_2s, p_2s) per heterointerface k. Block at
    # END of packed vector so legacy bulk-state slicing stays bit-
    # identical. None / size 0 = legacy MoL behaviour.
    iface_state: np.ndarray | None = None

    @staticmethod
    def pack(n, p, P, P_neg=None, iface_state=None) -> np.ndarray:
        parts = [n, p, P]
        if P_neg is not None:
            parts.append(P_neg)
        if iface_state is not None and np.asarray(iface_state).size > 0:
            parts.append(np.asarray(iface_state))
        return np.concatenate(parts)

    @staticmethod
    def unpack(
        y: np.ndarray, N: int, N_iface_state: int = 0,
    ) -> "StateVec":
        # Interface-plane block at the end (size = 4 * N_iface_state).
        iface_block_size = 4 * max(0, int(N_iface_state))
        bulk_end = len(y) - iface_block_size
        iface_state = (
            y[bulk_end:] if iface_block_size > 0 else None
        )
        # Detect dual-ion presence: bulk segment length 4N → P_neg active.
        if bulk_end == 4 * N:
            return StateVec(
                n=y[:N], p=y[N:2*N], P=y[2*N:3*N], P_neg=y[3*N:4*N],
                iface_state=iface_state,
            )
        return StateVec(
            n=y[:N], p=y[N:2*N], P=y[2*N:3*N],
            iface_state=iface_state,
        )


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
    # Phase E1 — per-interface SRH (n1, p1) used by interface_recombination.
    # Tuples aligned with ``interface_nodes``. Legacy fill = per-node bulk
    # ``n1[idx]`` / ``p1[idx]`` of the layer that owns the interface node
    # (bit-identical to the pre-E1 path). When ``stack.interface_defects``
    # provides an ``InterfaceDefect`` at index k, the entry is replaced with
    # ``srh_n1_p1_from_trap_depth(ni_ref, Eg_ref, defect.E_t_eV,
    # reference="below_cb")`` on the reference side (absorber if exactly one
    # adjacent layer is absorber, else lower-Eg side).
    interface_n1: tuple[float, ...] = ()
    interface_p1: tuple[float, ...] = ()
    # Phase E1.5 — Pauwels-Vanhoutte cross-carrier sampling at heterojunction
    # interface SRH. Two per-interface node indices: one where n is sampled
    # (transport-side interior, ``idx + 1`` — electrons accumulate there
    # under cliff) and one where p is sampled (absorber-side interior,
    # ``idx − 1`` — holes accumulate there under cliff). Both populations
    # rise at cliff → np rises → R explodes → V_oc tanks; spike suppresses
    # both → R small → V_oc preserved. This is the SCAPS cliff/spike
    # direction. Legacy default: both eval nodes equal ``idx`` so the
    # no-defect / pre-E1 path stays bit-identical.
    # ``interface_ni_sq_eff`` is the np_eq reference used in the SRH numerator
    # ``(n · p − ni_eff²)``; defaults to ``ni_sq[idx]`` for legacy, set to
    # the reference-side ``ni²`` (lower-Eg / absorber side, matching the
    # n1·p1 = ni_ref² identity from srh_n1_p1_from_trap_depth) when a defect
    # is populated.
    # Phase E1.6 (Option B-2) — per-interface attenuation factor on (v_n, v_p).
    # Default 1.0 = legacy bit-identical with pre-E1.6 behaviour. When an
    # ``InterfaceDefect`` declares ``calibration_factor`` it is forwarded
    # here at build time and multiplied into the SG-flux-derived surface
    # velocities in ``_apply_interface_recombination``. Lets SCAPS direct
    # N_t values plug into ``configs/scaps_mirror.yaml`` (e.g. ``N_t_cm2:
    # 1e13`` with ``calibration_factor: 1e-5`` instead of the empirical
    # ``N_t_cm2: 1e8``). See Phase A probe data + RFC at
    # ``docs/superpowers/specs/2026-05-26-e1.6-sg-face-density-spec.md``.
    interface_calibration_factor: tuple[float, ...] = ()
    # 2026-06 — per-interface SS interface-plane-state calibration. Folded
    # into ``interface_calibration_factor`` by ``_enable_iface_states`` on
    # the steady-state mat ONLY, so the SS state-channel rate is attenuated
    # without touching the transient bulk-node interface path. Empty/1.0 =
    # bit-identical.
    iface_state_calibration: tuple[float, ...] = ()
    interface_eval_node_n: tuple[int, ...] = ()
    interface_eval_node_p: tuple[int, ...] = ()
    interface_ni_sq_eff: tuple[float, ...] = ()
    # Phase E3 — charge-balance band-bending partition per interface k.
    # Fraction of V_bi_eff absorbed on the LEFT side. Heavy-doped right
    # (e.g. ETL N_D=1e18) ≪ light-doped left (PVK N_A=1e14) → 0.99+.
    interface_V_partition_2: tuple[float, ...] = ()
    # Per-interface equilibrium bulk densities on each side. Used by the
    # Phase E3 _compute_iface_state_dark_eq() helper to initialise the
    # 4·N_iface interface-plane state-vector block.
    interface_n_L_eq: tuple[float, ...] = ()
    interface_p_L_eq: tuple[float, ...] = ()
    interface_n_R_eq: tuple[float, ...] = ()
    interface_p_R_eq: tuple[float, ...] = ()
    # Phase E3 — count of interface-plane state blocks in the packed
    # state vector y. Zero = legacy MoL (iface_state absent). Set to
    # len(interface_V_partition_2) when SOLARLAB_INTERFACE_PLANE_STATE=1
    # AND at least one interface has charge-balance partition cached.
    N_iface_state: int = 0
    # TE coupling velocity for the interface-plane state block [m/s].
    # The 1e-2 default is the Sprint-7 throttle that keeps the block's
    # ODE timescale integrable by Radau (full thermal velocity makes the
    # transient Jacobian too stiff at the diode knee). The steady-state
    # driver overrides this to the FULL thermal velocity (1e5 m/s) on its
    # own mats — in an algebraic Newton solve the stiffness constraint
    # does not exist, which is what finally makes the SCAPS-style
    # interface-plane states feasible (P1 of scaps_mode, 2026-06).
    iface_state_v_th: float = 1.0e-2
    # Live-projection fill fluxes for the state block (P1 of scaps_mode):
    # targets from the LIVE adjacent node densities phi-projected to the
    # plane instead of the E3 equilibrium-cache projections (which pin
    # the states at equilibrium scale under illumination — measured).
    # Set by the steady-state driver together with iface_state_v_th.
    iface_state_live_proj: bool = False
    # Shared-occupancy single-trap SRH on the state block (P1): the
    # cross-pair form pairs majority sides and kills the E_t response;
    # the shared-occupancy form on PLANE densities is SCAPS's actual
    # formulation on the right substrate. Set by the SS driver.
    iface_state_shared_occ: bool = False
    # Occupancy-derived interface trapped charge in Poisson (P1): the
    # surviving E_t mechanism after three rate-algebra falsifications.
    # 0.0 = off (bit-identical); -1.0 = acceptor-like (charge -q*N_t*f),
    # +1.0 = donor-like. Set by the SS driver.
    iface_state_charge: float = 0.0
    # Heterointerface bulk-recombination de-spike fraction (SCAPS-emulation,
    # default 0.0 = off). See DeviceStack.het_recomb_despike.
    het_recomb_despike: float = 0.0
    # Heterointerface node indices (where a band offset produces the spike).
    het_recomb_nodes: tuple[int, ...] = ()
    # V-partition x live-QFL merge for the state targets (P1): sub-grid
    # interface band bending applied to the live node densities. The
    # uniform-suppression scan falsified every scalar model — the
    # bending's per-side/per-carrier/bias-dependent structure is the
    # surviving form. Set by the SS driver.
    iface_state_partition: bool = False
    # Phase E3 Day 4-6 — band offsets per interface for cross-flux χ
    # step coupling (paper eq 15). ΔE_c = chi_R − chi_L (eV) is
    # positive when electron sees a barrier going right→left; the
    # exp(−ΔE_c/V_T) factor in the cross-flux restores CBO sensitivity.
    interface_chi_step: tuple[float, ...] = ()  # chi_R − chi_L, eV
    interface_Eg_step:  tuple[float, ...] = ()  # Eg_R  − Eg_L,  eV
    # Precomputed LAPACK LU of the Poisson tridiagonal operator; reused in
    # every RHS call in place of scipy.sparse spsolve, which is the largest
    # single contributor to assemble_rhs runtime at the default grid sizes.
    poisson_factor: PoissonFactor | None = None
    # Effective built-in potential computed from band offsets (or manual V_bi fallback)
    V_bi_eff: float = 1.1
    # Built-in potential actually applied in the Poisson Dirichlet BC
    # (phi_right = V_bi_bc - V_app). Equals stack.V_bi by default (IonMonger
    # convention); under flat_band_contacts it is the flat-band
    # work-function difference compute_V_bi() (SCAPS convention).
    V_bi_bc: float = 1.1
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
    # Interface-plane projection toggle (2026-06). When True,
    # ``_apply_interface_recombination`` Boltzmann-projects the cross-carrier
    # eval densities onto the interface plane (SCAPS Pauwels-Vanhoutte
    # sampling). Computed at build from ``stack.interface_plane_projection``
    # OR the ``SOLARLAB_IFACE_PROJ=1`` env var. Default False = bit-identical
    # to the bulk-interior (E1.5) path.
    iface_plane_projection: bool = False
    # Two-sided P-V toggle (2026-06): adds the mirror cross-carrier pair in
    # ``_apply_interface_recombination``. From stack.interface_two_sided OR
    # env SOLARLAB_IFACE_TWOSIDED=1. Default False = one-sided E1.5 path.
    iface_two_sided: bool = False
    # Shared-occupancy P-V toggle (2026-06) + per-side trap-level densities
    # n1_i/p1_i [m^-3], each referenced to that side's own band edge and
    # effective DOS (depth_i = E_t + (chi_ref - chi_i)). Zero placeholders
    # at interfaces without an InterfaceDefect.
    iface_shared_occ: bool = False
    interface_n1_L: tuple[float, ...] = ()
    interface_p1_L: tuple[float, ...] = ()
    interface_n1_R: tuple[float, ...] = ()
    interface_p1_R: tuple[float, ...] = ()
    # QSS interface-plane closure (2026-06): per-interface build-time
    # constants (None at interfaces without a defect or without the parity
    # configuration's DOS data). See physics/interface_plane.py.
    iface_plane_closure: bool = False
    interface_plane_prm: tuple = ()
    # Smooth TE-cap blend width (relative). 0.0 = exact hard cap
    # (bit-identical legacy). Set ONLY by the steady-state driver
    # (experiments/steady_state.py): the hard magnitude-min kink at
    # heterointerface faces is the dominant non-smoothness blocking
    # Newton there (measured 15x stall-residual reduction with the cap
    # off at the V*~0.858 wall on scaps_mirror_v2). Transient
    # experiments never set this — their physics is unchanged.
    te_softness: float = 0.0
    # Override the dark/illuminated branch in ``assemble_rhs``: when True,
    # ``G_optical`` is used verbatim regardless of the ``illuminated`` kwarg.
    # Set by the lagged-G_rad fallback in ``_bake_radiative_reabsorption_step``
    # so a baked dark-plus-LED-emission step keeps its R_rad source instead
    # of being zeroed on the dark branch. Default False keeps the historical
    # dark behaviour (G zeroed) bit-identical.
    force_use_g_optical: bool = False
    # Phase E4 — per-node carrier diffusion coefficients for the
    # split-interface-flux helper. Half-flux at a heterointerface face
    # uses single-layer D_L = D_n_node[idx], D_R = D_n_node[idx+1] (not
    # the harmonic-mean face D). Always populated by build_material_arrays
    # because the harmonic-mean face derivation needs them anyway.
    D_n_node: np.ndarray | None = None
    D_p_node: np.ndarray | None = None

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
            if self.te_softness > 0.0:
                d["te_softness"] = self.te_softness
        if self.het_recomb_despike > 0.0 and self.het_recomb_nodes:
            d["het_recomb_despike"] = self.het_recomb_despike
            d["het_recomb_nodes"] = self.het_recomb_nodes
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


_INTERFACE_PLANE_STATE_ENV = "SOLARLAB_INTERFACE_PLANE_STATE"


def _interface_plane_state_active() -> bool:
    """True if Phase E3 interface-plane state path gated on via env var.

    Read at every call so monkeypatched tests can toggle mid-session.
    """
    return os.environ.get(_INTERFACE_PLANE_STATE_ENV) == "1"


def _compute_iface_state_dark_eq(mat: "MaterialArrays") -> np.ndarray:
    """Phase E3 — dark-equilibrium interface-plane state initial condition.

    Returns a 1-D array of shape ``(4 * N_iface,)`` carrying
    ``(n_1s, p_1s, n_2s, p_2s)`` per heterointerface k in that order.

    Boltzmann projection from cached equilibrium bulk densities via the
    charge-balance band-bending partition:
      V_total       = mat.V_bi_eff
      V_2 (PVK side) = partition_left * V_total
      V_1 (ETL side) = (1 - partition_left) * V_total
      n_1s = n_R_eq * exp(-V_1 / V_T)   # ETL e — depleted
      p_1s = p_R_eq * exp(+V_1 / V_T)   # ETL h — accumulated
      n_2s = n_L_eq * exp(+V_2 / V_T)   # PVK e — accumulated
      p_2s = p_L_eq * exp(-V_2 / V_T)   # PVK h — depleted

    Exponent capped at ±30 to avoid overflow at V_2 ≈ V_bi (where the
    naive exponent can reach ±42 for V_bi = 1.1 V at 300 K).
    """
    n_iface = len(mat.interface_V_partition_2)
    if n_iface == 0:
        return np.zeros(0, dtype=float)
    V_T_local = mat.V_T_device if hasattr(mat, "V_T_device") else _V_T_300
    V_total = float(mat.V_bi_eff)
    EXP_CAP = 30.0
    out = np.zeros(4 * n_iface, dtype=float)
    for k in range(n_iface):
        partition_left = float(mat.interface_V_partition_2[k])
        V_2 = partition_left * V_total
        V_1 = (1.0 - partition_left) * V_total
        v1_norm = max(-EXP_CAP, min(EXP_CAP, V_1 / V_T_local))
        v2_norm = max(-EXP_CAP, min(EXP_CAP, V_2 / V_T_local))
        n_R = float(mat.interface_n_R_eq[k])
        p_R = float(mat.interface_p_R_eq[k])
        n_L = float(mat.interface_n_L_eq[k])
        p_L = float(mat.interface_p_L_eq[k])
        # Phase E3 Day 4-6 — χ-step-consistent dark-eq init.
        # Anchor 1s side via Boltzmann from R bulk with V_1 band-bending,
        # derive 2s side via the χ step (paper eq 15: n_1s/n_2s =
        # exp(ΔE_c/V_T) at flat E_F). Keeps cross-flux ~ 0 at dark eq.
        n_1s = n_R * math.exp(-v1_norm)
        p_1s = p_R * math.exp(+v1_norm)
        if mat.interface_chi_step and len(mat.interface_chi_step) > k:
            dE_c = float(mat.interface_chi_step[k])
            dE_g = float(mat.interface_Eg_step[k])
            dE_v = dE_c - dE_g
            ec_norm = max(-EXP_CAP, min(EXP_CAP, dE_c / V_T_local))
            ev_norm = max(-EXP_CAP, min(EXP_CAP, dE_v / V_T_local))
            n_2s = n_1s * math.exp(-ec_norm)
            p_2s = p_1s * math.exp(-ev_norm)
        else:
            n_2s = n_L * math.exp(+v2_norm)
            p_2s = p_L * math.exp(-v2_norm)
        out[4 * k + 0] = n_1s
        out[4 * k + 1] = p_1s
        out[4 * k + 2] = n_2s
        out[4 * k + 3] = p_2s
    return out


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

    # Continuous bandgap grading (2026-06). When enabled (and not LEGACY),
    # a graded layer interpolates its per-node chi/Eg — and the Eg-derived
    # ni²/n1/p1 — from the front scalar (chi/Eg) to the back endpoint
    # (chi_back/Eg_back) via the SCAPS material law, replacing the uniform
    # scalar broadcast below. This is a static (build-time) coefficient
    # transform threaded through the immutable MaterialArrays — never on the
    # per-RHS path, so it carries no Newton-contraction risk. Layers without
    # back endpoints are untouched (has_grading_params False), so legacy and
    # ungraded configs are bit-identical even with the flag on. LEGACY tier
    # forces it off (mirrors dos_band_potentials). See physics/grading.py.
    from perovskite_sim.physics.grading import (
        has_grading_params,
        grading_coordinate,
        band_gap_profile,
        affinity_profile,
        grade_ni_sq,
        grade_n1_p1,
    )
    _band_grading = bool(
        getattr(stack, "band_grading", False)
        or os.environ.get("SOLARLAB_BAND_GRADING") == "1"
    ) and sim_mode.name != "legacy"

    offset = 0.0
    for layer in elec_layers:
        mask = (x >= offset - 1e-12) & (x <= offset + layer.thickness + 1e-12)
        p = layer.params
        # Local coordinate from the layer's front face — used by both the
        # grading profile and the trap profile below (hoisted to avoid
        # recomputation).
        x_local = x[mask] - offset
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
        if _band_grading and has_grading_params(p):
            # Graded layer: front endpoints are the scalar chi / Eg_T; the
            # back endpoints are chi_back / Eg_back (Varshni-shifted to match
            # the front when T-scaling is on). The SCAPS material law fills
            # the per-node transport gap; the DOS fold (below) composes on top.
            y_grade = grading_coordinate(
                x_local, layer.thickness, p.grading_profile,
                p.grading_char_length, p.grading_direction,
            )
            Eg_back = p.Eg_back if p.Eg_back is not None else p.Eg
            if sim_mode.use_temperature_scaling:
                Eg_back_T = eg_at_T(Eg_back, T_dev, p.varshni_alpha, p.varshni_beta)
            else:
                Eg_back_T = Eg_back
            chi_back = p.chi_back if p.chi_back is not None else p.chi
            chi[mask] = affinity_profile(y_grade, p.chi, chi_back)
            Eg[mask] = band_gap_profile(y_grade, Eg_T, Eg_back_T, p.grading_bowing)
        else:
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
        if _band_grading and has_grading_params(p):
            # Front-anchored DOS law: ni²(x) = ni_front²·exp(-(Eg(x)-Eg_front)/V_T).
            # Eg[mask] holds the graded transport gap (pre-DOS-fold), so ni
            # references the true statistical gap — the existing invariant.
            ni_sq[mask] = grade_ni_sq(ni_T ** 2, Eg[mask], Eg_T, V_T_dev)
        else:
            ni_sq[mask] = ni_T ** 2

        tau_n[mask] = p.tau_n
        tau_p[mask] = p.tau_p
        if _band_grading and has_grading_params(p):
            # Trap level fixed relative to midgap → n1·p1 = ni²(x) per node.
            n1[mask], p1[mask] = grade_n1_p1(p.n1, p.p1, Eg[mask], Eg_T, V_T_dev)
        else:
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
            shape = getattr(p, "trap_profile_shape", "exponential") or "exponential"
            edge_target = str(getattr(p, "trap_edge", "both") or "both").lower()
            if str(shape).lower() == "gaussian":
                N_t_x = gaussian_edge_profile(
                    x_local,
                    layer.thickness,
                    float(p.trap_N_t_interface),
                    float(p.trap_N_t_bulk),
                    float(p.trap_decay_length),
                    edge=edge_target,
                )
            else:
                N_t_x = exponential_edge_profile(
                    x_local,
                    layer.thickness,
                    float(p.trap_N_t_interface),
                    float(p.trap_N_t_bulk),
                    float(p.trap_decay_length),
                    edge=edge_target,
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

    # Effective-DOS band potentials (2026-06). With Boltzmann statistics the
    # heterostructure drift-diffusion potentials are phi + chi + V_T·ln(N_C)
    # (electrons) and phi + chi + Eg − V_T·ln(N_V) (holes); the SG flux in
    # continuity.py carries only phi+chi / phi+chi+Eg, so layers with
    # different effective DOS acquire a spurious kT·ln(DOS-ratio) QFL step
    # at the junction. Fold the corrections into the cached chi/Eg arrays
    # (transport + TE see them) with the absorber as reference — only the
    # cross-junction differences matter. ni / n1 / p1 / boundary densities
    # are deliberately untouched (they are statistics, not transport).
    # Default-ON (2026-06): the fold is correct heterojunction transport
    # physics. Layers without Nc300/Nv300 are skipped, so non-DOS configs are
    # bit-identical regardless. LEGACY tier always disables it — LEGACY is
    # contractually bit-identical to IonMonger, which has no DOS-folded
    # transport. ``SOLARLAB_DOS_BAND=1`` is a legacy force-ON (now redundant);
    # set ``dos_band_potentials=False`` on the stack to force the pre-fix path.
    _dos_band = bool(
        getattr(stack, "dos_band_potentials", True)
        or os.environ.get("SOLARLAB_DOS_BAND") == "1"
    ) and sim_mode.name != "legacy"
    if _dos_band:
        _ref = next(
            (L.params for L in elec_layers if L.role == "absorber"),
            elec_layers[0].params,
        )
        if _ref.Nc300 and _ref.Nv300:
            _off = 0.0
            for layer in elec_layers:
                _m = (x >= _off - 1e-12) & (x <= _off + layer.thickness + 1e-12)
                _q = layer.params
                if _q.Nc300 and _q.Nv300:
                    _dC = V_T_dev * math.log(_q.Nc300 / _ref.Nc300)
                    _dV = V_T_dev * math.log(_q.Nv300 / _ref.Nv300)
                    chi[_m] = chi[_m] + _dC
                    Eg[_m] = Eg[_m] - _dC - _dV  # (chi+Eg) shifts by −dV
                _off += layer.thickness

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

    # Interface-plane projection (SCAPS Pauwels-Vanhoutte sampling). Enabled
    # by the stack-level config flag OR the legacy ``SOLARLAB_IFACE_PROJ=1``
    # env var; tier-independent (works on FAST, the SCAPS-parity tier).
    # Default off → ``_apply_interface_recombination`` keeps the bulk-interior
    # (E1.5) path, bit-identical.
    _iface_plane_projection = bool(
        getattr(stack, "interface_plane_projection", False)
        or os.environ.get("SOLARLAB_IFACE_PROJ") == "1"
    )

    _iface_two_sided = bool(
        getattr(stack, "interface_two_sided", False)
        or os.environ.get("SOLARLAB_IFACE_TWOSIDED") == "1"
    )

    _iface_shared_occ = bool(
        getattr(stack, "interface_shared_occupancy", False)
        or os.environ.get("SOLARLAB_IFACE_SHARED_OCC") == "1"
    )

    _iface_plane_closure = bool(
        getattr(stack, "interface_plane_closure", False)
        or os.environ.get("SOLARLAB_IFACE_PLANE") == "1"
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

    # SCAPS-style flat-band contacts (2026-06). Device-level opt-in that
    # activates the Robin path on ALL FOUR carrier/side channels regardless
    # of tier (the SCAPS contact model is finite-S, default 1e7 cm/s), with
    # the existing doping-derived boundary equilibria as the flat-band
    # references, and routes the flat-band work-function difference
    # compute_V_bi() into the Poisson BC via V_bi_bc below. Default False =
    # ideal-ohmic pins + frozen stack.V_bi, bit-identical.
    _flat_band = bool(getattr(stack, "flat_band_contacts", False))
    if _flat_band:
        _has_selective_contacts = True
    _S_FLAT_BAND = 1.0e5  # SCAPS contact default: 1e7 cm/s

    def _s_contact(v):
        if v is not None:
            return v
        return _S_FLAT_BAND if _flat_band else None

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

    # Phase E1 — per-interface (n1, p1). Default = per-node bulk values
    # (bit-identical legacy). Replaced with E_t-derived values when the
    # stack supplies an ``InterfaceDefect`` at the matching index.
    from perovskite_sim.sweeps.device_parameter_sweep import (
        srh_n1_p1_from_trap_depth,
    )
    interface_n1_list: list[float] = []
    interface_p1_list: list[float] = []
    interface_eval_n_list: list[int] = []
    interface_eval_p_list: list[int] = []
    interface_ni_sq_eff_list: list[float] = []
    # Phase E1.6 — per-interface attenuation factor multiplied into v_n, v_p.
    interface_calibration_factor_list: list[float] = []
    # 2026-06 — per-interface SS interface-plane-state calibration (folded
    # into interface_calibration_factor on the SS mat only).
    iface_state_calibration_list: list[float] = []
    # Shared-occupancy P-V (2026-06): per-side trap-level densities.
    interface_n1_L_list: list[float] = []
    interface_p1_L_list: list[float] = []
    interface_n1_R_list: list[float] = []
    interface_p1_R_list: list[float] = []
    interface_plane_prm_list: list = []
    # Phase E3 — charge-balance partition + equilibrium bulk densities
    # per interface. Populated for EVERY interface (defect or not) so
    # the interface-plane state initial condition can build a sensible
    # equilibrium block even on stacks without explicit defects.
    interface_V_partition_2_list: list[float] = []
    interface_n_L_eq_list: list[float] = []
    interface_p_L_eq_list: list[float] = []
    interface_n_R_eq_list: list[float] = []
    interface_p_R_eq_list: list[float] = []
    interface_chi_step_list: list[float] = []
    interface_Eg_step_list: list[float] = []
    # Substrate-offset-aligned defects (2026-06 fix): stack.interface_defects
    # is full-layer-aligned; this loop indexes by the ELECTRICAL interface
    # number, so apply the same offset electrical_interfaces uses. Without it
    # a substrate-prefixed stack silently shifted every defect by one slot
    # (E10.1 glass regression: HTL/PVK fell back to the legacy path).
    defects = electrical_interface_defects(stack)
    N_grid = len(x)

    def _eq_n_p_layer(p_):
        """Equilibrium (n, p) for a layer params object (Phase E3 helper)."""
        net = 0.5 * (p_.N_D - p_.N_A)
        disc = float(np.sqrt(net * net + p_.ni * p_.ni))
        if net >= 0.0:
            n_eq_ = net + disc
            p_eq_ = p_.ni * p_.ni / n_eq_
        else:
            p_eq_ = -net + disc
            n_eq_ = p_.ni * p_.ni / p_eq_
        return n_eq_, p_eq_

    for k, idx in enumerate(iface_list):
        # Phase E3 — populate partition + equilibrium-bulk-density caches
        # for EVERY interface (defect or not). Needed by interface-plane
        # state initial condition + RHS, which can be active even on
        # legacy-style stacks if the user toggles SOLARLAB_INTERFACE_PLANE_STATE.
        left_e3 = elec_layers[k].params
        right_e3 = elec_layers[k + 1].params
        N_L_eff = max(float(left_e3.N_A), float(left_e3.N_D), 1.0e10)
        N_R_eff = max(float(right_e3.N_A), float(right_e3.N_D), 1.0e10)
        denom_qb = (
            N_L_eff * float(left_e3.eps_r)
            + N_R_eff * float(right_e3.eps_r)
        )
        if denom_qb > 0.0:
            partition_left = (
                N_R_eff * float(right_e3.eps_r) / denom_qb
            )
        else:
            partition_left = 0.5
        interface_V_partition_2_list.append(partition_left)
        n_L_eq_e3, p_L_eq_e3 = _eq_n_p_layer(left_e3)
        n_R_eq_e3, p_R_eq_e3 = _eq_n_p_layer(right_e3)
        interface_n_L_eq_list.append(float(n_L_eq_e3))
        interface_p_L_eq_list.append(float(p_L_eq_e3))
        interface_n_R_eq_list.append(float(n_R_eq_e3))
        interface_p_R_eq_list.append(float(p_R_eq_e3))
        # Phase E3 Day 4-6 — band offsets for cross-flux χ-step coupling.
        # ΔE_c = chi_R − chi_L (eV); >0 means CB cliff for electrons
        # going R→L (e.g. ETL chi=4.0 → PVK chi=3.84 → ΔE_c=+0.16 eV).
        # ΔE_v = Eg_R − Eg_L − ΔE_c = (Eg_R − Eg_L) − (chi_R − chi_L).
        # Cache the (Eg_R − Eg_L) term — interface_plane.py derives ΔE_v.
        interface_chi_step_list.append(
            float(right_e3.chi) - float(left_e3.chi)
        )
        interface_Eg_step_list.append(
            float(right_e3.Eg) - float(left_e3.Eg)
        )

        defect = defects[k] if k < len(defects) else None
        if defect is None:
            # Legacy bit-identical: per-node bulk n1/p1, sample at idx,
            # ni² ref from per-node ni_sq[idx] (current SRH numerator),
            # calibration_factor = 1.0 (no attenuation).
            interface_n1_list.append(float(n1[idx]))
            interface_p1_list.append(float(p1[idx]))
            interface_eval_n_list.append(idx)
            interface_eval_p_list.append(idx)
            interface_ni_sq_eff_list.append(float(ni_sq[idx]))
            interface_calibration_factor_list.append(1.0)
            iface_state_calibration_list.append(1.0)
            interface_n1_L_list.append(0.0)
            interface_p1_L_list.append(0.0)
            interface_n1_R_list.append(0.0)
            interface_p1_R_list.append(0.0)
            interface_plane_prm_list.append(None)
            continue
        left = elec_layers[k].params
        right = elec_layers[k + 1].params
        # Reference side: absorber if exactly one side is absorber, else
        # lower-Eg side (smaller bandgap dominates the SRH kinetics).
        left_is_abs = elec_layers[k].role == "absorber"
        right_is_abs = elec_layers[k + 1].role == "absorber"
        if left_is_abs and not right_is_abs:
            ref = left
        elif right_is_abs and not left_is_abs:
            ref = right
        else:
            ref = left if left.Eg <= right.Eg else right
        n1_k, p1_k = srh_n1_p1_from_trap_depth(
            ref.ni, ref.Eg, float(defect.E_t_eV),
            reference="below_cb", thermal_voltage=V_T_dev,
        )
        # Cross-carrier Pauwels-Vanhoutte sampling: electrons from the
        # transport (right interior) side where they pool under cliff,
        # holes from the absorber (left interior) side where they pool
        # under cliff. Bounds-clamped so a single-node-layer pathological
        # grid still resolves to a valid index.
        eval_n_idx = min(idx + 1, N_grid - 1)
        eval_p_idx = max(idx - 1, 0)
        interface_n1_list.append(n1_k)
        interface_p1_list.append(p1_k)
        interface_eval_n_list.append(eval_n_idx)
        interface_eval_p_list.append(eval_p_idx)
        # ni_eff² for cross-carrier detailed balance: at thermal
        # equilibrium n_R_eq · p_L_eq must equal ni_eff² so the SRH
        # numerator (n · p − ni_eff²) vanishes (zero generation/
        # recombination at equilibrium). With heavily-doped contact
        # layers (e.g. ETL N_D=1e24 m⁻³) the equilibrium n_R · p_L can
        # exceed ni_L · ni_R by many orders of magnitude — using a
        # naïve ni_L · ni_R would make R non-zero at dark equilibrium
        # and tank V_oc spuriously. Compute the equilibrium densities
        # of each adjacent layer from its doping, then use the cross
        # product as the np_eq reference.
        def _eq_n_p(p_):
            net = 0.5 * (p_.N_D - p_.N_A)
            disc = float(np.sqrt(net * net + p_.ni * p_.ni))
            if net >= 0.0:
                n_eq = net + disc
                p_eq = p_.ni * p_.ni / n_eq
            else:
                p_eq = -net + disc
                n_eq = p_.ni * p_.ni / p_eq
            return n_eq, p_eq
        _, pL_eq = _eq_n_p(left)
        nR_eq, _ = _eq_n_p(right)
        interface_ni_sq_eff_list.append(float(nR_eq * pL_eq))
        # Phase E1.6 — per-interface attenuation factor from the
        # InterfaceDefect dataclass. Multiplied into v_n, v_p downstream
        # in ``_apply_interface_recombination`` (mathematically equivalent
        # to scaling N_t areal density, so partner can declare SCAPS
        # direct N_t values + an explicit attenuation rather than the
        # empirical N_t calibration that hid the gap inside the
        # validation script).
        interface_calibration_factor_list.append(
            float(defect.calibration_factor)
        )
        iface_state_calibration_list.append(
            float(getattr(defect, "iface_state_calibration_factor", 1.0))
        )
        # Shared-occupancy P-V: per-side trap-level densities. The trap sits
        # at a fixed absolute energy E_t below the REFERENCE side's CB, so
        # each side sees depth_i = E_t + (chi_ref - chi_i) below its own CB.
        # With per-layer effective DOS available (SCAPS loader populates
        # Nc300/Nv300), n1_i = N_C,i*exp(-depth_i/V_T) and
        # p1_i = N_V,i*exp(-(Eg_i - depth_i)/V_T); otherwise fall back to the
        # ni/Eg midgap-symmetric form.
        def _side_n1p1(p_, depth):
            if p_.Nc300 and p_.Nv300:
                return (
                    p_.Nc300 * math.exp(-depth / V_T_dev),
                    p_.Nv300 * math.exp(-(p_.Eg - depth) / V_T_dev),
                )
            return srh_n1_p1_from_trap_depth(
                p_.ni, p_.Eg, depth,
                reference="below_cb", thermal_voltage=V_T_dev,
            )
        _depth_L = float(defect.E_t_eV) + (float(ref.chi) - float(left.chi))
        _depth_R = float(defect.E_t_eV) + (float(ref.chi) - float(right.chi))
        _n1_L, _p1_L = _side_n1p1(left, _depth_L)
        _n1_R, _p1_R = _side_n1p1(right, _depth_R)
        interface_n1_L_list.append(float(_n1_L))
        interface_p1_L_list.append(float(_p1_L))
        interface_n1_R_list.append(float(_n1_R))
        interface_p1_R_list.append(float(_p1_R))
        # QSS plane-closure constants from the FOLDED chi/Eg node values
        # (the DOS fold ran before this loop; its shift is zero on the
        # reference layer, so E_t below the raw reference CB needs no
        # correction). Requires the parity configuration: dos_band_potentials
        # + reference-layer DOS data — otherwise None (closure inactive).
        if _dos_band and ref.Nc300 and ref.Nv300:
            _iL, _iR = max(idx - 1, 0), min(idx + 1, len(x) - 1)
            interface_plane_prm_list.append(build_plane_params(
                float(chi[_iL]), float(chi[_iR]),
                float(chi[_iL]) + float(Eg[_iL]),
                float(chi[_iR]) + float(Eg[_iR]),
                float(ref.Nc300), float(ref.Nv300), float(ref.chi),
                float(defect.E_t_eV), V_T_dev,
            ))
        else:
            interface_plane_prm_list.append(None)

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

    # Intra-band thermionic-field-emission (TFE) tunnelling (2026-06,
    # DeviceStack.interface_tunneling, default OFF). Fold a static, symmetric
    # Padovani-Stratton enhancement Gamma >= 1 into the per-face Richardson
    # constants A* at the TE-capped faces (A*_eff = Gamma·A*). Symmetric ⇒
    # equilibrium J_TE = 0 preserved exactly; the continuity TE cap keeps the
    # SG flux as the ceiling (apply side is byte-identical — A* is read only at
    # these faces). Static ⇒ no per-RHS state ⇒ no Newton risk. Uses the SAME
    # (graded + DOS-folded) chi/Eg the TE cap sees. Requires TE on; LEGACY
    # disables TE so this is off by construction.
    _iface_tunnel = bool(
        getattr(stack, "interface_tunneling", False)
        or os.environ.get("SOLARLAB_IFACE_TUNNEL") == "1"
    ) and sim_mode.use_thermionic_emission and len(interface_face_list) > 0
    if _iface_tunnel:
        from perovskite_sim.physics.tunneling import tfe_gamma
        m_eff = float(getattr(stack, "tunnel_mass_eff", 0.2))
        _capped = set(interface_face_list)
        for k, idx in enumerate(iface_list):
            f = idx - 1
            if f not in _capped or k + 1 >= len(elec_layers):
                continue
            pl = elec_layers[k].params
            pr = elec_layers[k + 1].params
            net_l = abs(pl.N_D - pl.N_A)
            net_r = abs(pr.N_D - pr.N_A)
            # Depletion sits on the lighter-doped side — its doping + eps set
            # the field-emission characteristic energy E_00.
            if net_l <= net_r:
                N_iface, eps_iface = net_l, pl.eps_r
            else:
                N_iface, eps_iface = net_r, pr.eps_r
            dEc = abs(chi[f] - chi[f + 1])
            dEv = abs((chi[f] + Eg[f]) - (chi[f + 1] + Eg[f + 1]))
            A_star_n_node[f] *= tfe_gamma(dEc, N_iface, m_eff, eps_iface, V_T_dev)
            A_star_p_node[f] *= tfe_gamma(dEv, N_iface, m_eff, eps_iface, V_T_dev)

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
                if _band_grading and has_grading_params(p):
                    # Graded absorber: photon recycling escapes at the
                    # dominant (narrowest) emission edge — the smaller of the
                    # front/back endpoint gaps. Uses the true (un-DOS-folded)
                    # endpoint gaps, not the folded transport array.
                    Eg_back = p.Eg_back if p.Eg_back is not None else p.Eg
                    Eg_eV = float(min(p.Eg, Eg_back))
                else:
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

    arrays = MaterialArrays(
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
        D_n_node=D_n_node.copy(),
        D_p_node=D_p_node.copy(),
        D_ion_face=D_ion_face,
        P_lim_face=P_lim_face,
        dx_cell=dx_cell,
        interface_nodes=tuple(iface_list),
        interface_n1=tuple(interface_n1_list),
        interface_p1=tuple(interface_p1_list),
        interface_calibration_factor=tuple(interface_calibration_factor_list),
        iface_state_calibration=tuple(iface_state_calibration_list),
        interface_eval_node_n=tuple(interface_eval_n_list),
        interface_eval_node_p=tuple(interface_eval_p_list),
        interface_ni_sq_eff=tuple(interface_ni_sq_eff_list),
        interface_n1_L=tuple(interface_n1_L_list),
        interface_p1_L=tuple(interface_p1_L_list),
        interface_n1_R=tuple(interface_n1_R_list),
        interface_p1_R=tuple(interface_p1_R_list),
        iface_plane_closure=_iface_plane_closure,
        interface_plane_prm=tuple(interface_plane_prm_list),
        het_recomb_despike=float(getattr(stack, "het_recomb_despike", 0.0)),
        het_recomb_nodes=(
            tuple(iface_list)
            if float(getattr(stack, "het_recomb_despike", 0.0)) > 0.0
            else ()
        ),
        interface_V_partition_2=tuple(interface_V_partition_2_list),
        interface_n_L_eq=tuple(interface_n_L_eq_list),
        interface_p_L_eq=tuple(interface_p_L_eq_list),
        interface_n_R_eq=tuple(interface_n_R_eq_list),
        interface_p_R_eq=tuple(interface_p_R_eq_list),
        interface_chi_step=tuple(interface_chi_step_list),
        interface_Eg_step=tuple(interface_Eg_step_list),
        N_iface_state=(
            len(interface_V_partition_2_list)
            if _interface_plane_state_active()
            and len(interface_V_partition_2_list) > 0
            else 0
        ),
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
        iface_plane_projection=_iface_plane_projection,
        iface_two_sided=_iface_two_sided,
        iface_shared_occ=_iface_shared_occ,
        S_n_L=_s_contact(stack.S_n_left) if _has_selective_contacts else None,
        S_p_L=_s_contact(stack.S_p_left) if _has_selective_contacts else None,
        S_n_R=_s_contact(stack.S_n_right) if _has_selective_contacts else None,
        S_p_R=_s_contact(stack.S_p_right) if _has_selective_contacts else None,
        has_selective_contacts=_has_selective_contacts,
        V_bi_bc=(stack.compute_V_bi() if _flat_band else stack.V_bi),
        absorber_masks=tuple(absorber_masks_list) if _has_radiative_reabsorption else (),
        absorber_p_esc=tuple(absorber_p_esc_list) if _has_radiative_reabsorption else (),
        absorber_thicknesses=tuple(absorber_thicknesses_list) if _has_radiative_reabsorption else (),
        has_radiative_reabsorption=_has_radiative_reabsorption,
        N_t_node=N_t_node_arr,
        has_trap_profile=_has_trap_profile,
    )

    if getattr(stack, "autoloop_generated_lever", False) or os.environ.get("SOLARLAB_AUTOLOOP_GEN") == "1":
        from perovskite_sim.autoloop.generated.lever import adjust_material_arrays
        from perovskite_sim.autoloop.generated._ctx import _LeverContext
        arrays = adjust_material_arrays(arrays, _LeverContext(x=x, stack=stack))
    return arrays


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


_IFACE_PROJ_EXP_CAP = 40.0  # cap |Δφ/V_T| before exp() to avoid overflow

# Phase E11 — TE equilibration velocity for the QSS interface-plane balance.
# Large → interface-plane state ≈ projected bulk for weak SRH (R = SRH(proj));
# self-limiting (depletion) for strong SRH. Env-overridable for tuning probes.
_QSS_V_TH_MS = float(os.environ.get("SOLARLAB_QSS_VTH", "1.0e5"))


def _qss_interface_R(proj_n, proj_p, ni_sq, n1, p1, v_n, v_p, v_th):
    """QSS interface SRH rate [m⁻² s⁻¹] via 1-D bounded root-find.

    Solves the quasi-steady-state balance v_th·δ = SRH(proj_n−δ, proj_p−δ) for
    the interface-plane depletion δ ≥ 0 and returns R = v_th·δ. Intrinsically
    R ≥ 0 (np ≤ ni² → δ=0 → no spurious generation, so no clamp is needed) and
    self-limiting (δ bounded by min(proj_n, proj_p)). No ODE DOF / Newton
    feedback — the stability de-risk vs the dormant interface-plane-state path.
    Validated offline against analytical limits (scripts/probes/e11_qss_math.py).
    """
    if v_n == 0.0 and v_p == 0.0:
        return 0.0
    R0 = interface_recombination(proj_n, proj_p, ni_sq, n1, p1, v_n, v_p)
    if R0 <= 0.0:
        return 0.0  # at/below equilibrium → no recombination (no generation)
    hi = min(proj_n, proj_p) * (1.0 - 1e-9)
    if hi <= 0.0:
        return 0.0

    def f(d):
        return v_th * d - interface_recombination(
            proj_n - d, proj_p - d, ni_sq, n1, p1, v_n, v_p
        )

    if f(hi) < 0.0:
        return v_th * hi  # transport(TE)-limited recombination
    lo, h = 0.0, hi
    for _ in range(40):
        mid = 0.5 * (lo + h)
        if f(mid) > 0.0:
            h = mid
        else:
            lo = mid
    return v_th * 0.5 * (lo + h)


def _apply_interface_recombination(
    dn: np.ndarray,
    dp: np.ndarray,
    n: np.ndarray,
    p: np.ndarray,
    stack: DeviceStack,
    mat: MaterialArrays,
    phi: np.ndarray | None = None,
) -> None:
    """Subtract interface recombination from dn, dp at interface nodes (in-place).

    Interface recombination is a surface rate [m⁻² s⁻¹] converted to
    volumetric [m⁻³ s⁻¹] by dividing by the dual-grid cell width.

    Phase E8 — band-bending interface-plane projection (env-gated by
    ``SOLARLAB_IFACE_PROJ=1``). The E1.5 cross-carrier path samples the
    bulk-interior densities ``n[idx+1]`` / ``p[idx-1]`` directly, which
    over-counts interface SRH because SCAPS reads the depletion-suppressed
    *interface-plane* densities. When the gate is on and ``phi`` is
    supplied, each eval density is Boltzmann-projected onto the interface
    node, and the ``ni_eff²`` reference is projected by the *same* combined
    factor so the SRH numerator (n·p − ni²) stays exactly zero at dark
    equilibrium — the detailed-balance term the failed
    ``failed-prototype/e2-bbd-face-density`` attempt dropped. Default off:
    bit-identical to the E1.5 path.
    """
    ifaces = electrical_interfaces(stack)
    if not ifaces:
        return
    proj = phi is not None and mat.iface_plane_projection
    # Phase E9.3 — forbid net interface SRH *generation* (R_s < 0), DEFAULT ON.
    # The cross-carrier ni_sq_eff = nR_eq·pL_eq is a bulk-asymptotic product; at
    # a thin transport interface (HTL/PVK) it is orders too high (1e44), so
    # under illumination np < ni_sq_eff and the SRH rate flips to spurious
    # generation — measured −82 A/m² at short circuit, which inflated J_sc
    # above the photogeneration (and the SQ limit, 33.3 vs 26.3 mA/cm²) and made
    # the HTL/PVK N_t sweep rise the wrong way. A passive recombination centre
    # cannot be a net carrier *source* at illuminated short circuit; clamp
    # R_s ≥ 0 so HTL/PVK goes inert (matching SCAPS, which holds V_oc flat
    # across that sweep). Escape hatch ``SOLARLAB_IFACE_ALLOW_GEN=1`` restores
    # the legacy (unphysical-generation) branch for back-comparison only.
    nogen = os.environ.get("SOLARLAB_IFACE_ALLOW_GEN", "") != "1"
    # Phase E11 — QSS interface-plane SRH (env-gated, default OFF). Replaces the
    # bulk-interior cross-carrier sampling + clamp with the physically-correct
    # interface-plane carrier model: Boltzmann-project the live bulk densities
    # onto the interface plane, then solve the quasi-steady-state balance
    # v_th·δ = SRH(proj−δ) (1-D bounded root-find, R = v_th·δ). Intrinsically
    # R ≥ 0 (no spurious generation → no clamp needed) and depletion-aware.
    qss = phi is not None and os.environ.get("SOLARLAB_IFACE_QSS", "") == "1"
    V_T_dev = mat.V_T_device if hasattr(mat, "V_T_device") else _V_T_300
    for k, idx in enumerate(mat.interface_nodes):
        if k >= len(ifaces):
            break
        v_n, v_p = ifaces[k]
        # Phase E1.6 — per-interface attenuation factor (Anderson v_eff
        # calibration). When the InterfaceDefect declared a non-default
        # calibration_factor, the effective surface velocities are
        # scaled before the SRH rate computation. Mathematically
        # equivalent to scaling the SCAPS-direct N_t areal density,
        # so partner can write ``N_t_cm2: 1e13 + calibration_factor:
        # 1e-5`` instead of the empirical ``N_t_cm2: 1e8`` that hid
        # the gap in the script. Default 1.0 = legacy bit-identical.
        if mat.interface_calibration_factor:
            cf = mat.interface_calibration_factor[k]
            v_n = v_n * cf
            v_p = v_p * cf
        if v_n == 0.0 and v_p == 0.0:
            continue
        # Cross-carrier (Pauwels-Vanhoutte) sample pair: n from the
        # transport-side eval node, p from the absorber-side eval node.
        # Legacy path (no defect): both fall back to idx so the rate
        # collapses to the pre-E1 single-side form.
        eval_n_idx = mat.interface_eval_node_n[k] if mat.interface_eval_node_n else idx
        eval_p_idx = mat.interface_eval_node_p[k] if mat.interface_eval_node_p else idx
        ni_sq_eff = (
            mat.interface_ni_sq_eff[k]
            if mat.interface_ni_sq_eff
            else float(mat.ni_sq[idx])
        )
        n_eval = float(n[eval_n_idx])
        p_eval = float(p[eval_p_idx])
        # QSS interface-plane closure (2026-06): evaluate the P-V rate on
        # TRUE plane densities from a local implicit 2x2 flux balance —
        # supply-limited, reduced-interface-gap, trap-level-visible (see
        # physics/interface_plane.py). Node densities are Boltzmann-
        # projected to the plane potential here; band-offset penalties and
        # plane-gap constants are cached in mat.interface_plane_prm.
        # Takes precedence over every other interface formulation.
        if (
            mat.iface_plane_closure
            and eval_n_idx != idx
            and mat.interface_plane_prm
            and mat.interface_plane_prm[k] is not None
        ):
            prm_k = mat.interface_plane_prm[k]
            eL = (float(phi[idx]) - float(phi[idx - 1])) / V_T_dev
            eR = (float(phi[idx]) - float(phi[idx + 1])) / V_T_dev
            eL = max(-_IFACE_PROJ_EXP_CAP, min(_IFACE_PROJ_EXP_CAP, eL))
            eR = max(-_IFACE_PROJ_EXP_CAP, min(_IFACE_PROJ_EXP_CAP, eR))
            fL = math.exp(eL)
            fR = math.exp(eR)
            _n_s, _p_s, R_s = solve_plane_densities(
                max(float(n[idx - 1]), 0.0) * fL,
                max(float(n[idx + 1]), 0.0) * fR,
                max(float(p[idx - 1]), 0.0) / fL,
                max(float(p[idx + 1]), 0.0) / fR,
                prm_k, v_n, v_p,
            )
            R_vol = R_s / mat.dx_cell[idx]
            dn[idx] -= R_vol
            dp[idx] -= R_vol
            continue
        # Shared-occupancy P-V (2026-06): ONE trap occupancy fed by both
        # layers — the coupled closed form with per-side trap-level
        # densities in the denominator (the occupancy mechanism) and the
        # discrete-equilibrium-consistent numerator reference (R = 0
        # exactly when the sampled nodes sit at their cached dark-
        # equilibrium values; the textbook n1S*p1S reference assumes
        # interface-plane sampling). Replaces the one-sided pair at defect
        # interfaces; not composed with proj / QSS / two-sided. Densities
        # floored at zero (SG minority overshoots destabilise Radau).
        if mat.iface_shared_occ and eval_n_idx != idx and mat.interface_n1_L:
            nS = max(float(n[eval_p_idx]), 0.0) + max(float(n[eval_n_idx]), 0.0)
            pS = max(float(p[eval_p_idx]), 0.0) + max(float(p[eval_n_idx]), 0.0)
            refS = (
                (mat.interface_n_L_eq[k] + mat.interface_n_R_eq[k])
                * (mat.interface_p_L_eq[k] + mat.interface_p_R_eq[k])
            )
            n1S = mat.interface_n1_L[k] + mat.interface_n1_R[k]
            p1S = mat.interface_p1_L[k] + mat.interface_p1_R[k]
            R_s = interface_recombination(nS, pS, refS, n1S, p1S, v_n, v_p)
            if nogen and R_s < 0.0:
                R_s = 0.0
            R_vol = R_s / mat.dx_cell[idx]
            dn[idx] -= R_vol
            dp[idx] -= R_vol
            continue
        if qss:
            # Interface-plane projection (live bulk → plane via local band
            # bending) then QSS 1-D root-find. Co-project ni² so detailed
            # balance holds (R=0 at equilibrium).
            en = (float(phi[idx]) - float(phi[eval_n_idx])) / V_T_dev
            ep = (float(phi[idx]) - float(phi[eval_p_idx])) / V_T_dev
            en = max(-_IFACE_PROJ_EXP_CAP, min(_IFACE_PROJ_EXP_CAP, en))
            ep = max(-_IFACE_PROJ_EXP_CAP, min(_IFACE_PROJ_EXP_CAP, ep))
            fac_n = math.exp(en); fac_p = math.exp(-ep)
            R_s = _qss_interface_R(
                n_eval * fac_n, p_eval * fac_p, ni_sq_eff * fac_n * fac_p,
                mat.interface_n1[k], mat.interface_p1[k], v_n, v_p, _QSS_V_TH_MS,
            )
            R_vol = R_s / mat.dx_cell[idx]
            dn[idx] -= R_vol
            dp[idx] -= R_vol
            continue
        if proj:
            # Boltzmann-project the bulk-interior eval densities onto the
            # interface plane: n[idx] = n[eval_n]·exp((φ[idx]−φ[eval_n])/V_T),
            # p[idx] = p[eval_p]·exp(−(φ[idx]−φ[eval_p])/V_T). Co-project
            # ni_eff² by the same combined factor (fac_n·fac_p) so the
            # numerator is exactly fac·(n·p − ni²) → zero at equilibrium.
            en = (float(phi[idx]) - float(phi[eval_n_idx])) / V_T_dev
            ep = (float(phi[idx]) - float(phi[eval_p_idx])) / V_T_dev
            en = max(-_IFACE_PROJ_EXP_CAP, min(_IFACE_PROJ_EXP_CAP, en))
            ep = max(-_IFACE_PROJ_EXP_CAP, min(_IFACE_PROJ_EXP_CAP, ep))
            fac_n = math.exp(en)
            fac_p = math.exp(-ep)
            n_eval *= fac_n
            p_eval *= fac_p
            ni_sq_eff = ni_sq_eff * fac_n * fac_p
        R_s = interface_recombination(
            n_eval, p_eval, ni_sq_eff,
            mat.interface_n1[k], mat.interface_p1[k],
            v_n, v_p,
        )
        if nogen and R_s < 0.0:
            R_s = 0.0
        # Volumetric loss conversion uses the interface-node dual cell.
        # The depletion is applied at the interface node itself —
        # diffusion / drift then re-feeds the eval nodes from the bulk
        # state, preserving carrier conservation across the dual cell.
        R_vol = R_s / mat.dx_cell[idx]
        dn[idx] -= R_vol
        dp[idx] -= R_vol
        # Two-sided P-V (2026-06): mirror pair B — electrons from the LEFT
        # slab (n[eval_p_idx]) with holes from the RIGHT slab (p[eval_n_idx])
        # against its own detailed-balance reference n_L_eq*p_R_eq (Phase-E3
        # caches), so R_B vanishes exactly at dark equilibrium. Active only
        # where an InterfaceDefect set cross-carrier eval nodes; legacy
        # single-node interfaces (eval == idx) are untouched. Clamped
        # non-negative independently of the pair-A NOGEN clamp.
        if mat.iface_two_sided and eval_n_idx != idx and mat.interface_n_L_eq:
            # Floor at zero: pair B samples MINORITY densities at
            # heterojunction-adjacent nodes, where SG transients can
            # overshoot negative; a sign-flipping R_B destabilises Radau.
            nB = max(float(n[eval_p_idx]), 0.0)
            pB = max(float(p[eval_n_idx]), 0.0)
            ni_sq_B = mat.interface_n_L_eq[k] * mat.interface_p_R_eq[k]
            if proj:
                enB = (float(phi[idx]) - float(phi[eval_p_idx])) / V_T_dev
                epB = (float(phi[idx]) - float(phi[eval_n_idx])) / V_T_dev
                enB = max(-_IFACE_PROJ_EXP_CAP, min(_IFACE_PROJ_EXP_CAP, enB))
                epB = max(-_IFACE_PROJ_EXP_CAP, min(_IFACE_PROJ_EXP_CAP, epB))
                fnB = math.exp(enB)
                fpB = math.exp(-epB)
                nB *= fnB
                pB *= fpB
                ni_sq_B = ni_sq_B * fnB * fpB
            R_B = interface_recombination(
                nB, pB, ni_sq_B,
                mat.interface_n1[k], mat.interface_p1[k],
                v_n, v_p,
            )
            if R_B > 0.0:
                R_vol_B = R_B / mat.dx_cell[idx]
                dn[idx] -= R_vol_B
                dp[idx] -= R_vol_B


def assemble_rhs(
    t: float,
    y: np.ndarray,
    x: np.ndarray,
    stack: DeviceStack,
    mat: MaterialArrays,
    illuminated: bool = True,
    V_app: float = 0.0,
    phi_frozen: np.ndarray | None = None,
) -> np.ndarray:
    """Method of Lines RHS: dy/dt = f(t, y).

    `mat` is the pre-built per-experiment material cache. Building it here
    would allocate ~20 numpy arrays per Radau RHS call and dominated runtime
    of the caching refactor's target experiments.
    """
    N = len(x)
    sv = StateVec.unpack(y, N, N_iface_state=mat.N_iface_state)

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

    # Solve Poisson — UNLESS a frozen phi is supplied (the Gummel carrier
    # half of experiments/steady_state.py evaluates the carrier-continuity
    # residual at a held electrostatic potential, decoupling the dense
    # phi-mediated Jacobian tail). phi_frozen is None on every default path,
    # so this is bit-identical when unused.
    # phi_right = V_bi - V_app: forward bias (V_app > 0) reduces the
    # built-in field; V_app = 0 → short circuit, V_app ≈ V_oc → open circuit.
    if phi_frozen is not None:
        phi = phi_frozen
    else:
        rho = _charge_density(
            p, n, sv.P, mat.P_ion0, mat.N_A, mat.N_D,
            P_neg=sv.P_neg, P_neg0=mat.P_ion0_neg,
        )
        # Occupancy-derived interface trapped charge (P1 of scaps_mode):
        # areal q*N_t*(f - f_eq) at each defect interface, signed by the
        # acceptor/donor convention, as a volumetric source at the node.
        if (
            mat.N_iface_state > 0
            and mat.iface_state_charge != 0.0
            and sv.iface_state is not None
        ):
            from perovskite_sim.physics.interface_plane import (
                compute_interface_trap_charge,
            )
            _dQ = compute_interface_trap_charge(sv.iface_state, stack, mat)
            rho = rho.copy()
            for _k in range(min(mat.N_iface_state, len(mat.interface_nodes))):
                _ix = mat.interface_nodes[_k]
                rho[_ix] += mat.iface_state_charge * _dQ[_k] / mat.dx_cell[_ix]
        phi = solve_poisson_prefactored(
            mat.poisson_factor, rho, phi_left=0.0,
            phi_right=mat.V_bi_bc - V_app,
        )

    # Generation: TMM-computed profile if available, else Beer-Lambert fallback.
    # ``force_use_g_optical`` lets the lagged-G_rad bake step keep its
    # baked R_rad source under the dark branch (LED-mode forward injection
    # produces radiative emission even with no solar input — the bake step
    # has already pre-zeroed the solar contribution and folded R_rad in).
    if illuminated or mat.force_use_g_optical:
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
            R_tot = float(trapezoid(emission, x_abs))
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
    # Phase E4 — split-flux scaffold landed in physics/continuity.py
    # (split_interface_flux + split_interface_flux_p helpers + divergence
    # override gated by params["interface_split_data"]). Sprint 8 Day 4-7
    # ships the plumbing groundwork; the bulk-side wire-through requires
    # Sprint 9 TE BC at the heterointerface face to conserve carrier mass
    # across the iface plane. Until then, assemble_rhs does NOT inject
    # interface_split_data so carrier_continuity_rhs uses the legacy SG
    # flux divergence (Phase E3 Sprint 7 bulk drain stays active).
    dn, dp = carrier_continuity_rhs(x, phi, n, p, G, carrier_params)

    # Interface recombination (surface SRH at heterointerfaces).
    # Phase E3 path replaces this with TE flux + SRH on state vec; see
    # below near the iface_state RHS block. Legacy E1.5 cross-carrier
    # SRH applies only when N_iface_state == 0.
    if mat.N_iface_state == 0:
        _apply_interface_recombination(dn, dp, n, p, stack, mat, phi)

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

    # Phase E3 -- interface-plane state RHS (TE flux in + SRH sink).
    # Also drains bulk-side carriers via the TE flux: positive TE means
    # carrier leaves bulk and enters interface-plane state. This is the
    # missing coupling that lets the new path actually affect bulk V_oc.
    diface_state = None
    if mat.N_iface_state > 0 and sv.iface_state is not None:
        from perovskite_sim.physics.interface_plane import (
            compute_interface_srh_on_state,
            compute_interface_srh_shared_on_state,
            compute_interface_te_fluxes,
            compute_interface_te_fluxes_live,
        )
        # Phase E3 Day 4-6 — pass v_cross_eff=v_th_eff to activate the
        # χ-step cross-interface TE flux (paper eq 15). Restores CBO
        # sensitivity by coupling n_1s ↔ n_2s and p_1s ↔ p_2s across
        # the χ step.
        if mat.iface_state_live_proj:
            te_fluxes = compute_interface_te_fluxes_live(
                mat, sv.iface_state, n, p, phi,
                v_th_eff=mat.iface_state_v_th,
                v_cross_eff=mat.iface_state_v_th,
                V_app=V_app,
            )
        else:
            te_fluxes = compute_interface_te_fluxes(
                mat, sv.iface_state, V_app=V_app,
                v_th_eff=mat.iface_state_v_th,
                v_cross_eff=mat.iface_state_v_th,
            )
        if mat.iface_state_shared_occ:
            srh_sinks = compute_interface_srh_shared_on_state(
                sv.iface_state, stack, mat,
            )
        else:
            srh_sinks = compute_interface_srh_on_state(
                sv.iface_state, stack, mat,
            )
        diface_state = np.zeros_like(sv.iface_state)
        for k in range(mat.N_iface_state):
            if k >= len(mat.interface_nodes):
                break
            idx = mat.interface_nodes[k]
            dx_iface = float(mat.dx_cell[idx])
            base = 4 * k
            # iface_state evolution: TE flux fills + SRH drains.
            for j in range(4):
                diface_state[base + j] = (
                    te_fluxes[base + j] + srh_sinks[base + j]
                ) / dx_iface
            # Bulk drain: TE flux removes carriers from the bulk-side
            # eval node. Positive TE -> carrier leaves bulk into state.
            # eval_n holds bulk-side electron node (ETL interior idx+1);
            # eval_p holds bulk-side hole node (PVK interior idx-1).
            if mat.interface_eval_node_n and mat.interface_eval_node_p:
                eval_n = mat.interface_eval_node_n[k]
                eval_p = mat.interface_eval_node_p[k]
                # n_1s (block 0) drains ETL electron at eval_n.
                # p_1s (block 1) drains ETL hole   at eval_n.
                # n_2s (block 2) drains PVK electron at eval_p.
                # p_2s (block 3) drains PVK hole   at eval_p.
                dn[eval_n] -= te_fluxes[base + 0] / dx_iface
                dp[eval_n] -= te_fluxes[base + 1] / dx_iface
                dn[eval_p] -= te_fluxes[base + 2] / dx_iface
                dp[eval_p] -= te_fluxes[base + 3] / dx_iface

    if _RHS_FINITE_CHECK:
        _assert_finite_rhs(dn, dp, dP, dP_neg, V_app)

    return StateVec.pack(dn, dp, dP, dP_neg, iface_state=diface_state)


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
    method: str = "Radau",
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
                         method=method, rtol=rtol, atol=atol,
                         dense_output=False, max_step=max_step)

    counter = [0]

    def rhs(t, y):
        counter[0] += 1
        if max_nfev is not None and counter[0] > max_nfev:
            raise _NfevExceeded
        return assemble_rhs(t, y, x, stack, mat, illuminated, V_app)

    try:
        return solve_ivp(rhs, t_span, y0, t_eval=t_eval,
                         method=method, rtol=rtol, atol=atol,
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
    sv = StateVec.unpack(y, N, N_iface_state=mat.N_iface_state)

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
                phi_left=0.0, phi_right=mat.V_bi_bc - V_app,
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
                phi_left=0.0, phi_right=mat.V_bi_bc - V_app,
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
        y_ions_advanced = StateVec.pack(
            sv.n, sv.p, P_new, P_neg_new, iface_state=sv.iface_state,
        )
    else:
        P_new = np.clip(sol_ion.y[:, -1], 0.0, mat.P_lim_node)
        P_neg_carry = sv.P_neg  # None in single-species mode
        y_ions_advanced = StateVec.pack(
            sv.n, sv.p, P_new, P_neg_carry, iface_state=sv.iface_state,
        )

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
