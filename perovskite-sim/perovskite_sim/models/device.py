from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Optional
from perovskite_sim.constants import V_T
from perovskite_sim.models.parameters import MaterialParams
from perovskite_sim.twod.microstructure import Microstructure


BUILT_IN_POTENTIAL_MODES = (
    "legacy_manual",
    "semiconductor_work_function",
    "metal_work_function",
)

INTERFACE_CHARGE_CLOSURES = (
    "off",
    "equilibrium_referenced",
)

INTERFACE_CHARGE_UNLOCK_REQUIREMENTS = (
    "a certified contact-consistent dark reference",
    "a certified interface-recombination charge-off refinement matrix",
    "a certified two-sided Gauss jump on discontinuous permittivity",
    "a self-consistent outer-Poisson/interface-state Jacobian",
)


class InterfaceChargeClosureParkedError(ValueError):
    """A production path attempted to activate the parked charge closure."""


@dataclass(frozen=True)
class LayerSpec:
    name: str
    thickness: float         # metres
    params: Optional[MaterialParams]
    role: str               # "ETL", "absorber", "HTL"


@dataclass(frozen=True)
class InterfaceDefect:
    """Single-level SRH defect localised at a heterointerface.

    ``E_t_eV`` is the trap depth below the conduction band of the reference
    side (SCAPS convention). Reference side is selected in
    ``build_material_arrays``: absorber if exactly one adjacent layer is an
    absorber, else the lower-Eg side. The resulting ``n1`` / ``p1`` use
    ``srh_n1_p1_from_trap_depth(ni_ref, Eg_ref, E_t_eV, reference="below_cb")``.

    Phase E1.6 (Option B-2, Anderson v_eff calibration) — ``calibration_factor``
    multiplies ``v_n, v_p`` from ``DeviceStack.interfaces[k]`` before the
    cross-carrier SRH rate computation in
    ``solver/mol.py:_apply_interface_recombination``. Default 1.0 is
    legacy bit-identical with pre-E1.6 behaviour. Used to absorb the
    SCAPS-vs-SolarLab face-density discretization gap (Phase A probe data:
    cross-carrier bulk-interior sampling over-counts the interface SRH rate
    by ~5 orders vs SCAPS interface-plane carrier evaluation). Setting
    ``N_t_cm2: 1e13`` (SCAPS direct) + ``calibration_factor: 1e-5`` produces
    the same effective SRV as the empirical ``N_t_cm2: 1e8`` + default
    factor, so partner sees the calibration explicitly in the YAML rather
    than hidden in a validation-script constant.
    """
    E_t_eV: float
    calibration_factor: float = 1.0
    # 2026-06 — SS interface-plane-state calibration. Multiplies (v_n, v_p)
    # for this interface ONLY inside the steady-state interface-plane-state
    # recombination channel (``_enable_iface_states`` folds it into
    # ``interface_calibration_factor`` on the SS mat). The transient
    # bulk-node interface path never reads it, so the transient parity
    # config is untouched. Default 1.0 = bit-identical. Calibrated values
    # absorb the over-strong interface-plane channel (base −61 mV → −0.1 mV,
    # Nd_ETL 2× → 0.84×, HTL/PVK N_t 12× → 1.0×): scaps_mirror_v2 uses
    # 0.02 (HTL/PVK) and 0.10 (PVK/ETL).
    iface_state_calibration_factor: float = 1.0
    # Phase E9 — SCAPS-declared areal trap density [cm^-2] this defect was
    # built from. The loader derives the base SRV in ``DeviceStack.interfaces``
    # as σ·v_th·N_t; storing N_t here lets a sweep over interface N_t scale the
    # base SRV by the N_t ratio (σ-consistent) instead of re-deriving with a
    # hardcoded σ. 0.0 = not set (sweep falls back to legacy σ=1e-15 path).
    N_t_cm2: float = 0.0


@dataclass(frozen=True)
class DeviceStack:
    layers: tuple[LayerSpec, ...]
    phi_left: float = 0.0   # V
    # Legacy/manual built-in-potential magnitude [V]. New physical configs
    # should select an explicit ``built_in_potential_mode`` and omit this from
    # their YAML. The field remains for benchmark compatibility and existing
    # programmatic callers.
    V_bi: float = 1.1
    # None preserves the pre-mode compatibility rule: a normal contact uses
    # the manual V_bi magnitude, while flat_band_contacts uses the historical
    # endpoint-Fermi estimate. Explicit modes decouple the Poisson-potential
    # source from the carrier contact kinetics.
    built_in_potential_mode: str | None = None
    # Positive work functions referenced below vacuum [eV]. Both are required
    # by ``metal_work_function``; their difference is numerically volts.
    work_function_left_eV: float | None = None
    work_function_right_eV: float | None = None
    Phi: float = 2.5e21     # photon flux [m⁻² s⁻¹] (AM1.5G)
    # Interface recombination: (v_n, v_p) per internal interface [m/s].
    # interfaces[0] = interface between layers[0] and layers[1], etc.
    # Empty tuple means no interface recombination.
    interfaces: tuple[tuple[float, float], ...] = ()
    # Phase E1 — per-interface SRH defect (optional, aligned with ``interfaces``).
    # ``None`` entries (or an empty tuple) fall back to the per-node bulk
    # ``n1`` / ``p1`` of the layer that owns the interface node, which is
    # bit-identical to the pre-E1 solver path. Populating an
    # ``InterfaceDefect`` activates the E_t-aware n1/p1 derivation at that
    # interface so the SCAPS cliff/spike direction at heterointerfaces with
    # a defect-rich face becomes physically accessible.
    interface_defects: tuple[Optional[InterfaceDefect], ...] = ()
    # Interface-plane projection for SCAPS-parity interface SRH (2026-06).
    # When True (or env ``SOLARLAB_IFACE_PROJ=1``), the cross-carrier
    # interface recombination samples the band-bending-suppressed
    # *interface-plane* carrier densities (Boltzmann-projected from the
    # bulk-interior eval nodes, with ni_eff² co-projected) instead of the
    # bulk-interior densities — matching SCAPS's Pauwels-Vanhoutte interface
    # model. Default False = bit-identical to the pre-projection (E1.5
    # bulk-interior) path.
    interface_plane_projection: bool = False
    # Effective-DOS band potentials for heterojunction transport (2026-06).
    # When True (or env ``SOLARLAB_DOS_BAND=1``), build_material_arrays folds
    # V_T·ln(N_C/N_C_ref) and V_T·ln(N_V/N_V_ref) into the cached chi/Eg
    # arrays used by the SG flux and TE capping, removing the spurious
    # kT·ln(DOS-ratio) quasi-Fermi-level step at DOS-contrast heterojunctions
    # (measured 137 mV on scaps_mirror_v2 — the SolarLab-vs-SCAPS V_oc root
    # cause). Requires per-layer Nc300/Nv300 (populated by the SCAPS loader);
    # layers without DOS data are left untouched, so legacy configs are
    # bit-identical under the flag. Default True (2026-06): the fold is real
    # heterojunction transport physics (omitting it is the bug); set False to
    # force the pre-fix transport. LEGACY tier always disables it regardless
    # (IonMonger bit-identity contract — see build_material_arrays).
    dos_band_potentials: bool = True
    # Physical thermionic-emission normalization (2026-07, review F02).
    # Default False = the legacy density-weighted TE cap (dimensionally
    # A/m^5; empirical, near-inert because |J_TE| is ~1e28-1e35 so the
    # magnitude-min cap almost never binds). When True AND the adjacent
    # layers carry Nc300/Nv300, the TE flux is divided by the band-edge DOS
    # at each capped face, giving the dimensionally correct emission-velocity
    # current J = q v_R (...) with v_R = A*T^2/(q N_dos); the cap then binds
    # at real interface densities. A single face DOS scales both legs equally,
    # so equilibrium J=0 is preserved. Configs without Nc300/Nv300 (e.g.
    # ionmonger_benchmark) are bit-identical even with the flag on. LEGACY
    # tier forces it off. Opt-in while the shift to pinned baselines is
    # characterized (review F02 re-baselining campaign).
    te_physical_norm: bool = False
    # Physical diffusion-only steric ion flux (2026-07, review F05). Default
    # False = the legacy whole-flux steric factor (drift + diffusion scaled
    # equally). When True, the crowding chemical potential is folded into the
    # SG drift argument so steric acts on diffusion only (the dimensionally
    # faithful modified-PNP form), with the Bernoulli structure preserved.
    # DEFAULT SINCE 2026-07-28 (review F-03). Two reasons, both measured on
    # ionmonger_benchmark with P_lim lowered to sweep the initial occupancy
    # theta0 = P0/P_lim at fixed P0:
    #
    #   theta0   legacy FF   legacy J_sc   PNP FF    PNP J_sc
    #    0.01     0.77668      221.573     0.77662    221.571
    #    0.50     0.79294      221.761     0.77658    221.571
    #    0.80     0.82237      222.099     0.77648    221.571
    #    0.95     0.82097      223.346     0.77616    221.571
    #    0.99     0.78163      228.586     0.77580    221.569
    #
    # (1) The legacy whole-flux factor s = 1/(1 - theta) multiplies DRIFT as
    # well as diffusion, so crowding ACCELERATES ion transport instead of
    # impeding it. The consequences are visibly unphysical: fill factor
    # climbing from 0.777 to 0.822 and J_sc gaining 3.2 % as the lattice
    # fills, plus a hysteresis index swinging to -0.066 and a 2.3 -> 5.2 s
    # stiffness cost. The modified-PNP form enhances only the diffusive
    # part, which is the term the crowding chemical potential actually
    # steepens, and is flat across the whole range.
    #
    # (2) It is free where the shipped presets live: at their theta ~ 0.011
    # the two forms differ by 0.04 mV in V_oc and 0.002 A/m^2 in J_sc.
    #
    # LEGACY tier still forces it OFF, so IonMonger reproduction is
    # unaffected by this default. See physics/ion_migration.py.
    ion_steric_diffusion_only: bool = True
    # Dual-ion site-sharing model for the diffusion-only steric flux (F05).
    # Only relevant when ion_steric_diffusion_only is on AND a negative
    # species is configured. True (default) = the two species share one
    # finite-site reservoir, so the crowding potential uses the TOTAL
    # occupancy (P_+ + P_-)/P_lim (standard multi-species finite-size PNP;
    # reduces to the single-species form when one density is zero). False =
    # distinct sublattices, each species crowds only against itself with its
    # own P_lim — declare this only when the defects genuinely occupy
    # different sites. No effect on single-species runs.
    ion_steric_shared_site: bool = True
    # Autoloop Stage 5.3 codegen lever (2026-06). When True (or env
    # ``SOLARLAB_AUTOLOOP_GEN=1``), build_material_arrays calls the sandboxed
    # ``autoloop.generated.lever.adjust_material_arrays`` once on the assembled
    # MaterialArrays. Default False = the generated module is never imported →
    # bit-identical. The autoloop writes the lever body; a human merges the branch.
    autoloop_generated_lever: bool = False
    # SCAPS-style finite-rate carrier contacts (2026-06). When True, the
    # Phase-3.3 Robin path is activated on all four carrier/side channels
    # (S = 1e5 m/s, the SCAPS 1e7 cm/s default, unless explicit ``S_*`` fields
    # are set) relative to the existing doping-derived equilibria. The Poisson
    # potential source is independent when ``built_in_potential_mode`` is
    # explicit. With that mode omitted, the historical implication
    # flat_band_contacts -> compute_V_bi() is retained for compatibility.
    flat_band_contacts: bool = False
    # Flat-band METAL contact reservoir (2026-07). When True (LEGACY forces
    # off), the contact carrier reservoir is set by the metal work function
    # rather than the semiconductor doping: build_material_arrays FLOORS each
    # contact's MAJORITY-carrier ohmic-equilibrium density at the flat-band
    # metal value (the majority carrier is chosen by the contact-layer doping
    # sign, so nip and pin stacks are both correct):
    #   n-type contact: n = max(N_D-eq, N_C·exp(-contact_phi_B_eV / V_T))
    #   p-type contact: p = max(N_A-eq, N_V·exp(-contact_phi_B_eV / V_T))
    # (the paired minority is set to ni²/majority). A heavily-doped contact
    # keeps the ideal-ohmic value (max picks doping →
    # bit-identical); only a weakly-doped contact gets the doping-independent
    # metal supply. This fixes the low-N_D contact starvation where the default
    # ideal pin leaves V_oc unbracketed / on an unphysical super-bandgap branch
    # (the ETL-donor-doping trend no de-spike fraction can fix). The floor is
    # DORMANT at normal doping, so every sweep that holds the contacts at their
    # base doping (CBO, Nt, Et, absorber doping, base point) is bit-identical —
    # only weakly-doped-contact sweeps are affected. Requires the contact layer
    # to carry Nc300/Nv300; a layer without them is skipped (bit-identical).
    # ``contact_phi_B_eV`` is the metal work-function offset (electron barrier
    # at the cathode / hole barrier at the anode); a calibration knob — ~0.42
    # eV lands scaps_mirror_v2's Nd_ETL sweep within ~40 mV RMS of SCAPS.
    # Default False = doping-derived reservoir, bit-identical.
    flat_band_metal_contacts: bool = False
    contact_phi_B_eV: float = 0.0
    # Two-sided Pauwels-Vanhoutte interface capture (2026-06). When True (or
    # env ``SOLARLAB_IFACE_TWOSIDED=1``), interface recombination adds the
    # mirror cross-carrier pair (electrons from the left slab x holes from
    # the right slab) with its own detailed-balance reference n_L_eq*p_R_eq,
    # approximating SCAPS's two-sided trap coupling (jn1/jn2 + jp1/jp2).
    # Only interfaces with an InterfaceDefect are affected; default False is
    # bit-identical to the one-sided E1.5 formulation.
    interface_two_sided: bool = False
    # Shared-occupancy Pauwels-Vanhoutte interface recombination (2026-06).
    # When True (or env ``SOLARLAB_IFACE_SHARED_OCC=1``), defect interfaces
    # use the coupled single-occupancy closed form: both layers feed one
    # trap level, with per-side n1/p1 referenced to each side's own band
    # edge and effective DOS in the denominator, and the discrete-
    # equilibrium-consistent numerator reference. Replaces the one-sided
    # cross-pair rate at those interfaces; default False is bit-identical.
    interface_shared_occupancy: bool = False
    # QSS interface-plane closure (2026-06). When True (or env
    # ``SOLARLAB_IFACE_PLANE=1``), defect-interface recombination is
    # evaluated on true interface-plane densities solved from a local
    # 2x2 flux balance (supply-limited, reduced-interface-gap, trap-level-
    # visible — see physics/interface_plane.py). Activates only with
    # ``dos_band_potentials`` + reference-layer DOS data; takes precedence
    # over the other interface formulations. Default False bit-identical.
    interface_plane_closure: bool = False
    # Let the plane closure report NET GENERATION (review F-04, 2026-07-28).
    # The historical clamp returns 0 whenever n_s*p_s < ni_s^2, which erases
    # physical depletion-region generation at reverse bias — measured
    # -8.36 A/m^2 at the HTL/PVK interface of scaps_mirror_v2 at -0.5 V.
    # Sound HERE and not at the bulk cross-carrier sites because the plane
    # reference ni_s^2 = N_C N_V exp(-Eg_s/V_T) is a true detailed-balance
    # product; the signed rate is bounded by -ni_s^2/(n1_s/v_p + p1_s/v_n),
    # the textbook depletion-generation limit. Requires
    # ``interface_plane_closure``. Default False = bit-identical.
    interface_plane_generation: bool = False
    # Heterointerface bulk-recombination de-spike (2026-06, SCAPS-emulation,
    # default 0.0 = OFF / more physically faithful). The VB/CB band offset
    # produces a Boltzmann carrier spike (exp(dE/kT)) at the junction NODE
    # that feeds BULK Auger/radiative using the absorber's coefficients,
    # while the SAME interface loss is counted by interface SRH — a partial
    # double-count, amplified by the (correct) effective-DOS fold. SCAPS,
    # lacking the DOS fold + with its own interface treatment, under-counts
    # this. Setting 0<f<=1 blends the heterointerface-node recomb density
    # toward the geometric mean of neighbours by fraction f, emulating
    # SCAPS's lower interface recombination. Default 0.0 keeps SolarLab's
    # more-faithful (higher-Auger) base. Calibrated value for scaps_mirror
    # parity ~0.56. See project_scaps_root_cause_reanalysis memory.
    het_recomb_despike: float = 0.0
    # Continuous bandgap grading (2026-06). When True (or env
    # ``SOLARLAB_BAND_GRADING=1``), build_material_arrays interpolates each
    # graded layer's per-node chi/Eg (and the Eg-derived ni²/n1/p1) from the
    # front scalar (chi/Eg) to the back endpoint (chi_back/Eg_back) via the
    # SCAPS material law, instead of the uniform scalar broadcast. Requires a
    # layer to set Eg_back/chi_back (has_grading_params); layers without them
    # are untouched, so legacy configs are bit-identical even with the flag on.
    # Default False = bit-identical; LEGACY tier always forces it off (mirrors
    # dos_band_potentials). See physics/grading.py.
    band_grading: bool = False
    # Intra-band thermionic-field-emission (TFE) tunnelling at heterointerfaces
    # (2026-06). When True (or env ``SOLARLAB_IFACE_TUNNEL=1``),
    # build_material_arrays folds a static Padovani-Stratton enhancement
    # Gamma >= 1 into the per-face Richardson constants A* at the TE-capped
    # interface faces (A*_eff = Gamma·A*), modelling carriers that tunnel
    # through a CB/VB spike rather than only crossing over it (the channel
    # SCAPS's intra-band-tunnelling option exposes). Gamma is symmetric (both
    # legs) so equilibrium J_TE = 0 is preserved exactly, and the existing TE
    # cap keeps the SG flux as the ceiling. Static build-time term → no
    # per-RHS state → no Newton-contraction risk. Requires thermionic emission
    # active (the cap face set is only built then); LEGACY tier disables TE so
    # tunnelling is off by construction. Default False = bit-identical.
    # ``tunnel_mass_eff`` is the tunnelling effective mass (relative to the
    # free-electron mass) in the characteristic energy E_00; only used when
    # ``interface_tunneling`` is on. See physics/tunneling.py.
    interface_tunneling: bool = False
    tunnel_mass_eff: float = 0.2
    # Device temperature [K]. Default 300 K (isothermal).
    T: float = 300.0
    # Simulation mode name; resolved to a SimulationMode by resolve_mode().
    # "full" (default) enables every physics upgrade the config supports;
    # "legacy" reproduces pre-upgrade behaviour for benchmarking.
    mode: str = "full"
    # Selective / Schottky outer contact surface recombination velocities
    # (Phase 3.3 — Apr 2026). When all four are None the contacts are
    # ohmic Dirichlet (current behaviour, bit-identical numerics). When
    # any is set the corresponding boundary uses a Robin-type flux
    # ``J = ±q · S · (n − n_eq)`` and the carrier density at the
    # boundary node is allowed to evolve. Units: m/s. ``S = 0`` is a
    # perfectly blocking contact; ``S → ∞`` recovers the ohmic limit.
    S_n_left: Optional[float] = None
    S_p_left: Optional[float] = None
    S_n_right: Optional[float] = None
    S_p_right: Optional[float] = None
    # Lateral microstructure (2D Stage B — Apr 2026). Carries grain-boundary
    # bands with reduced SRH lifetimes that ``build_material_arrays_2d`` can
    # paint onto the (Ny, Nx) τ field. 1D solver paths and lateral-uniform 2D
    # paths ignore this field, so back-compat is bit-identical when empty.
    microstructure: Microstructure = field(default_factory=Microstructure)
    # Optional electrical-grid protocol, aligned with electrical_layers(self).
    # Appended after all historical fields to preserve positional-constructor
    # compatibility. Empty tuples retain the legacy grid exactly.
    grid_interval_weights: tuple[float, ...] = ()
    grid_alphas: tuple[float, ...] = ()
    # Production J-V driver capability required by this stack. ``general``
    # permits the transient and algebraic drivers. The QF-required policy is
    # reserved for cancellation-sensitive stacks whose physical regression
    # has only been certified in quasi-Fermi variables.
    jv_solver_policy: str = "general"
    # Interface traps remain recombination-only on all production paths.
    # ``equilibrium_referenced`` is a recognized research intent, not an
    # enabled solver mode: material assembly rejects it until the Phase-3
    # entry certificates and self-consistent two-sided Poisson coupling exist.
    interface_charge_closure: str = "off"
    # Opening the research lane invalidates the historical SCAPS calibration.
    # The acknowledgement is required in the config before readiness can even
    # be assessed; it does not bypass the parked production capability gate.
    interface_charge_rebaseline_acknowledged: bool = False

    def __post_init__(self):
        object.__setattr__(self, "layers", tuple(self.layers))
        object.__setattr__(
            self, "grid_interval_weights", tuple(self.grid_interval_weights)
        )
        object.__setattr__(self, "grid_alphas", tuple(self.grid_alphas))
        if self.interface_charge_closure not in INTERFACE_CHARGE_CLOSURES:
            raise ValueError(
                "interface_charge_closure must be one of "
                f"{INTERFACE_CHARGE_CLOSURES}, got "
                f"{self.interface_charge_closure!r}"
            )
        if not isinstance(self.interface_charge_rebaseline_acknowledged, bool):
            raise ValueError(
                "interface_charge_rebaseline_acknowledged must be boolean"
            )
        if (
            self.interface_charge_closure == "equilibrium_referenced"
            and not self.interface_charge_rebaseline_acknowledged
        ):
            raise ValueError(
                "equilibrium_referenced interface charge requires explicit "
                "interface_charge_rebaseline_acknowledged=true because the "
                "historical SCAPS calibration is no longer valid"
            )
        if self.jv_solver_policy not in (
            "general",
            "cancellation_safe_qf_required",
        ):
            raise ValueError(
                "jv_solver_policy must be 'general' or "
                "'cancellation_safe_qf_required', got "
                f"{self.jv_solver_policy!r}"
            )
        mode = self.built_in_potential_mode
        if mode is not None and mode not in BUILT_IN_POTENTIAL_MODES:
            raise ValueError(
                "built_in_potential_mode must be one of "
                f"{BUILT_IN_POTENTIAL_MODES}, got {mode!r}"
            )
        fermi_dirac_layers = tuple(
            layer.name
            for layer in self.layers
            if layer.role != "substrate"
            and layer.params is not None
            and layer.params.carrier_statistics == "fermi_dirac"
        )
        if fermi_dirac_layers and mode != "semiconductor_work_function":
            raise ValueError(
                "fermi_dirac carrier statistics require explicit "
                "built_in_potential_mode='semiconductor_work_function'; "
                "activated layers: "
                + ", ".join(fermi_dirac_layers)
            )
        if not math.isfinite(float(self.V_bi)):
            raise ValueError("V_bi must be finite")
        if mode == "legacy_manual" and float(self.V_bi) < 0.0:
            raise ValueError(
                "legacy_manual V_bi is a non-negative magnitude; contact "
                "orientation is derived separately"
            )
        work_functions = (
            self.work_function_left_eV,
            self.work_function_right_eV,
        )
        if mode == "metal_work_function":
            if any(value is None for value in work_functions):
                raise ValueError(
                    "metal_work_function requires work_function_left_eV and "
                    "work_function_right_eV"
                )
            for name, value in zip(
                ("work_function_left_eV", "work_function_right_eV"),
                work_functions,
            ):
                if not math.isfinite(float(value)) or float(value) <= 0.0:
                    raise ValueError(f"{name} must be finite and positive")
        elif any(value is not None for value in work_functions):
            raise ValueError(
                "contact work functions are only valid with "
                "built_in_potential_mode='metal_work_function'"
            )

    def require_interface_charge_off(self, *, consumer: str) -> None:
        """Fail closed until the Phase-3 electrostatic research lane unlocks."""
        if self.interface_charge_closure == "off":
            return
        requirements = "; ".join(INTERFACE_CHARGE_UNLOCK_REQUIREMENTS)
        raise InterfaceChargeClosureParkedError(
            f"{consumer} cannot activate interface_charge_closure="
            f"{self.interface_charge_closure!r}: interface trap electrostatics "
            f"is PARKED pending {requirements}"
        )

    @property
    def total_thickness(self) -> float:
        return sum(layer.thickness for layer in self.layers)

    @property
    def phi_right(self) -> float:
        return self.phi_left + self.poisson_built_in_potential()

    def resolved_built_in_potential_mode(self) -> str:
        """Return the active public mode, including legacy flag inference."""
        if self.built_in_potential_mode is not None:
            return self.built_in_potential_mode
        if self.flat_band_contacts:
            return "semiconductor_work_function"
        return "legacy_manual"

    def compute_semiconductor_V_bi(self) -> float:
        """Return the signed endpoint-semiconductor work-function difference.

        Unlike the historical :meth:`compute_V_bi`, this physical path is
        fail-closed: both electrical contact layers must provide positive
        ``chi``, ``Eg``, ``Nc300`` and ``Nv300``. The configured carrier
        statistics are evaluated at the temperature used by the active
        physics tier and include the configured Varshni shift and contact-face
        grading/doping profile.
        """
        from perovskite_sim.models.mode import resolve_mode

        elec = electrical_layers(self)
        if not elec:
            raise ValueError(
                "semiconductor_work_function requires electrical layers"
            )
        sim_mode = resolve_mode(self.mode)
        temperature = float(self.T) if sim_mode.use_temperature_scaling else 300.0
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("device temperature must be finite and positive")
        use_grading = bool(self.band_grading) and sim_mode.name != "legacy"
        left = _edge_params(elec[0], "front", use_grading)
        right = _edge_params(elec[-1], "back", use_grading)
        W_left = _semiconductor_work_function(
            left,
            temperature,
            sim_mode.use_temperature_scaling,
        )
        W_right = _semiconductor_work_function(
            right,
            temperature,
            sim_mode.use_temperature_scaling,
        )
        return W_left - W_right

    def compute_metal_V_bi(self) -> float:
        """Return signed ``W_left - W_right`` from explicit metal contacts."""
        if (
            self.work_function_left_eV is None
            or self.work_function_right_eV is None
        ):
            raise ValueError(
                "metal_work_function requires both contact work functions"
            )
        return float(self.work_function_left_eV - self.work_function_right_eV)

    def poisson_built_in_potential(self) -> float:
        """Resolve the signed built-in potential used by the Poisson BC.

        ``built_in_potential_mode=None`` is the compatibility sentinel. It
        preserves the historical coupling where ``flat_band_contacts`` selects
        ``compute_V_bi()`` and the ordinary contact path uses the configured
        magnitude with orientation inferred from the endpoint Fermi levels.
        Explicit modes are independent of the carrier-contact model.
        """
        mode = self.built_in_potential_mode
        if mode is None:
            if self.flat_band_contacts:
                return self.compute_V_bi()
            orientation = -1.0 if self.compute_V_bi() < 0.0 else 1.0
            return orientation * abs(float(self.V_bi))
        if mode == "legacy_manual":
            orientation = -1.0 if self.compute_V_bi() < 0.0 else 1.0
            return orientation * abs(float(self.V_bi))
        if mode == "semiconductor_work_function":
            return self.compute_semiconductor_V_bi()
        if mode == "metal_work_function":
            return self.compute_metal_V_bi()
        raise AssertionError(f"unvalidated built-in-potential mode {mode!r}")

    def operating_built_in_potential(self) -> float:
        """Return the signed potential used by physical operating defaults.

        Compatibility stacks retain the historical band-derived ``V_bi_eff``
        even when their Poisson boundary uses a manual magnitude. An explicit
        mode instead uses its selected contact potential consistently.
        """
        if self.built_in_potential_mode is None:
            return self.compute_V_bi()
        return self.poisson_built_in_potential()

    def compute_V_bi(self) -> float:
        """Derive the built-in potential from the Fermi-level difference
        across the heterostack.

        Uses the numerically stable two-branch formula for majority carrier
        density: net = 0.5*(N_D - N_A), disc = sqrt(net**2 + ni**2), then
        n or p = net + disc  (or  -net + disc for minority branch).

        Returns the SIGNED contact potential phi(right) - phi(left) =
        W_left - W_right at equilibrium (the value the flat-band Poisson
        boundary condition consumes directly). This is positive for the
        usual p-contact-left orientation and NEGATIVE for n-contact-left
        devices (e.g. ZnO/CdS/CIGS, n+/p c-Si) -- the negative sign is
        physically correct, not a bug. Consumers that need the built-in
        potential MAGNITUDE instead (J-V sweep V_max, V_oc-search brackets)
        must take ``abs()`` of this value; the default Poisson BC instead
        uses the positive-magnitude manual ``self.V_bi`` (IonMonger
        convention) -- see the "Band-offset contact BCs" note in CLAUDE.md.

        Falls back to the manual ``self.V_bi`` field when all layers have
        chi = Eg = 0 (backward compatibility with legacy configs).
        """
        # Compute V_bi from the electrical-only contacts. Substrate layers
        # have no Fermi level for the drift-diffusion problem and must be
        # excluded, or a glass layer at index 0 (chi=Eg=0) would drag the
        # left contact potential to zero.
        elec = tuple(l for l in self.layers if l.role != "substrate")
        if not elec:
            return self.V_bi
        all_zero = all(
            layer.params.chi == 0.0 and layer.params.Eg == 0.0
            for layer in elec
            if layer.params is not None
        )
        if all_zero:
            return self.V_bi

        # Contacts use their actual layer-face band and doping parameters. The
        # left contact is the first layer's front face and the right contact is
        # the last layer's back face. Uniform, ungraded stacks return the
        # identical params objects and therefore the identical float.
        band_grading = getattr(self, "band_grading", False)
        left = _edge_params(elec[0], "front", band_grading)
        right = _edge_params(elec[-1], "back", band_grading)

        e_f_left = _fermi_level(left)
        e_f_right = _fermi_level(right)
        return e_f_left - e_f_right


def electrical_layers(stack: "DeviceStack") -> tuple["LayerSpec", ...]:
    """Return layers that participate in the drift-diffusion solve.

    Layers with role == "substrate" are optical-only and skipped. The TMM
    optical path still walks stack.layers (full list); only the electrical
    path uses this filtered view.

    Substrate layers must form a contiguous prefix of stack.layers (or be
    entirely absent). Any substrate layer after a non-substrate layer is
    unsupported and raises ValueError, because the grid/interface indexing
    below assumes the post-filter layer order is a prefix of the full list.
    """
    seen_non_substrate = False
    for layer in stack.layers:
        if layer.role == "substrate":
            if seen_non_substrate:
                raise ValueError(
                    "substrate layers must form a contiguous prefix of "
                    "stack.layers (mid-stack or trailing substrate layers "
                    "are not supported)"
                )
        else:
            seen_non_substrate = True
    return tuple(l for l in stack.layers if l.role != "substrate")


def electrical_interfaces(
    stack: "DeviceStack",
) -> tuple[tuple[float, float], ...]:
    """Return interfaces aligned to electrical_layers (substrate excluded).

    ``stack.interfaces`` still has length ``len(stack.layers) - 1`` and is
    indexed against the *full* layer list. After filtering out a substrate
    prefix, ``electrical_layers`` drops the first ``substrate_prefix``
    layers; the first ``substrate_prefix`` interfaces therefore describe
    substrate↔substrate or substrate↔first-electrical-layer boundaries and
    have no electrical counterpart, so they must be dropped as well. All
    subsequent interfaces are kept in order.

    Assumes the contiguous-substrate-at-edge layout enforced by
    ``electrical_layers``; multi-substrate-prefix is fine, mid-stack or
    trailing substrate is rejected upstream.
    """
    # Count how many contiguous leading layers are substrate.
    substrate_prefix = 0
    for layer in stack.layers:
        if layer.role == "substrate":
            substrate_prefix += 1
        else:
            break
    elec_n = sum(1 for l in stack.layers if l.role != "substrate")
    desired = max(0, elec_n - 1)
    start = substrate_prefix
    return tuple(stack.interfaces[start : start + desired])


def electrical_interface_defects(
    stack: "DeviceStack",
) -> tuple[Optional[InterfaceDefect], ...]:
    """Return interface defects aligned to electrical_layers.

    ``stack.interface_defects`` is parallel to ``stack.interfaces`` (full
    layer list); apply the same substrate-prefix offset as
    ``electrical_interfaces`` so consumers that index by the electrical
    interface number get the right defect. Pads with ``None`` when the
    stack tuple is shorter than the electrical interface count (legacy
    configs may omit it entirely).
    """
    defects = tuple(getattr(stack, "interface_defects", ()) or ())
    substrate_prefix = 0
    for layer in stack.layers:
        if layer.role == "substrate":
            substrate_prefix += 1
        else:
            break
    elec_n = sum(1 for l in stack.layers if l.role != "substrate")
    desired = max(0, elec_n - 1)
    out = list(defects[substrate_prefix : substrate_prefix + desired])
    out.extend([None] * (desired - len(out)))
    return tuple(out)


def _edge_params(layer: "LayerSpec", side: str, band_grading: bool) -> "MaterialParams":
    """Contact-face MaterialParams for ``compute_V_bi``.

    Uniform, ungraded layers return ``p`` unchanged so legacy stacks are
    bit-identical. A graded back contact sees Eg_back/chi_back and the same
    mid-gap intrinsic-density law used by ``grade_ni_sq``. A spatially doped
    contact sees the profile value at that physical face.
    """
    import dataclasses
    from perovskite_sim.physics.grading import has_grading_params
    from perovskite_sim.physics.doping import doping_at_position

    p = layer.params
    if p is None:
        return p
    updates = {}
    if band_grading and side == "back" and has_grading_params(p):
        Eg_back = p.Eg_back if p.Eg_back is not None else p.Eg
        chi_back = p.chi_back if p.chi_back is not None else p.chi
        ni_back = p.ni * math.exp(-(Eg_back - p.Eg) / (2.0 * V_T))
        updates.update(chi=chi_back, Eg=Eg_back, ni=ni_back)
    position = 0.0 if side == "front" else float(layer.thickness)
    N_A_edge, N_D_edge = doping_at_position(
        p, position, float(layer.thickness)
    )
    if N_A_edge != p.N_A or N_D_edge != p.N_D:
        updates.update(N_A=N_A_edge, N_D=N_D_edge)
    return dataclasses.replace(p, **updates) if updates else p


def _fermi_level(p: MaterialParams) -> float:
    """Compute the Fermi level (in eV, referenced to vacuum) for a layer.

    Convention: E_F is measured as a positive energy below the vacuum level,
    so a deeper Fermi level has a larger numerical value.

    The intrinsic level sits at E_i = chi + Eg/2. Then:
    - n-type (N_D > N_A):  E_F = E_i - V_T * ln(n / ni)  (moves toward Ec)
    - p-type (N_A > N_D):  E_F = E_i + V_T * ln(p / ni)  (moves toward Ev)
    - intrinsic:           E_F = E_i

    The majority carrier density uses the numerically stable two-branch
    formula: net = 0.5*(N_D - N_A), disc = sqrt(net**2 + ni**2).
    """
    ni = p.ni
    e_i = p.chi + p.Eg / 2.0
    net = 0.5 * (p.N_D - p.N_A)
    disc = math.sqrt(net * net + ni * ni)

    if p.N_D > p.N_A:
        # n-type: majority electrons; n = net + disc
        n = net + disc
        return e_i - V_T * math.log(n / ni)
    elif p.N_A > p.N_D:
        # p-type: majority holes; p = -net + disc
        hole = -net + disc
        return e_i + V_T * math.log(hole / ni)
    else:
        # intrinsic
        return e_i


def _semiconductor_work_function(
    p: MaterialParams,
    temperature: float,
    use_temperature_scaling: bool,
) -> float:
    """Semiconductor work function below vacuum [eV]."""
    from perovskite_sim.physics.temperature import eg_at_T, thermal_voltage

    if p is None:
        raise ValueError(
            "semiconductor_work_function requires material parameters on both "
            "electrical contact layers"
        )
    if p.carrier_statistics == "fermi_dirac":
        from perovskite_sim.physics.contacts import (
            build_semiconductor_contact_state,
        )

        return build_semiconductor_contact_state(
            p,
            temperature_K=temperature,
            use_temperature_scaling=use_temperature_scaling,
        ).work_function_eV
    required = {
        "chi": p.chi,
        "Eg": p.Eg,
        "Nc300": p.Nc300,
        "Nv300": p.Nv300,
    }
    invalid = [
        name
        for name, value in required.items()
        if value is None or not math.isfinite(float(value)) or float(value) <= 0.0
    ]
    if invalid:
        raise ValueError(
            "semiconductor_work_function requires positive contact-layer "
            f"{', '.join(invalid)}"
        )
    if not all(
        math.isfinite(float(value)) and float(value) >= 0.0
        for value in (p.N_A, p.N_D)
    ):
        raise ValueError(
            "semiconductor_work_function requires finite, non-negative contact "
            "doping densities"
        )

    T = float(temperature)
    V_T_contact = thermal_voltage(T)
    Eg = (
        eg_at_T(p.Eg, T, p.varshni_alpha, p.varshni_beta)
        if use_temperature_scaling
        else float(p.Eg)
    )
    dos_scale = (T / 300.0) ** 1.5 if use_temperature_scaling else 1.0
    Nc = float(p.Nc300) * dos_scale
    Nv = float(p.Nv300) * dos_scale
    net_doping = float(p.N_D - p.N_A)

    if net_doping == 0.0:
        band_distance = 0.5 * Eg + 0.5 * V_T_contact * math.log(Nc / Nv)
        return float(p.chi + band_distance)

    log_ni = 0.5 * (math.log(Nc) + math.log(Nv)) - Eg / (2.0 * V_T_contact)
    ni = math.exp(log_ni) if log_ni > -745.0 else 0.0
    half_net = 0.5 * net_doping
    disc = math.hypot(half_net, ni)
    if net_doping > 0.0:
        n_majority = half_net + disc
        return float(p.chi + V_T_contact * math.log(Nc / n_majority))
    p_majority = -half_net + disc
    return float(p.chi + Eg - V_T_contact * math.log(Nv / p_majority))
