from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Optional
from perovskite_sim.constants import V_T
from perovskite_sim.models.parameters import MaterialParams
from perovskite_sim.twod.microstructure import Microstructure


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
    V_bi: float = 1.1       # built-in voltage [V]
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
    # bulk-interior) path. See docs/reference/SCAPSIfaceSRH.md.
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
    # Autoloop Stage 5.3 codegen lever (2026-06). When True (or env
    # ``SOLARLAB_AUTOLOOP_GEN=1``), build_material_arrays calls the sandboxed
    # ``autoloop.generated.lever.adjust_material_arrays`` once on the assembled
    # MaterialArrays. Default False = the generated module is never imported →
    # bit-identical. The autoloop writes the lever body; a human merges the branch.
    autoloop_generated_lever: bool = False
    # SCAPS-style flat-band contacts (2026-06). When True, both contacts are
    # treated as flat-band metals with finite surface-recombination kinetics
    # (the SCAPS contact model): the Phase-3.3 Robin path is activated on all
    # four carrier/side channels (S = 1e5 m/s, the SCAPS 1e7 cm/s default,
    # unless explicit ``S_*`` fields are set) referenced to the existing
    # doping-derived boundary equilibria, and the Poisson BC uses the
    # flat-band work-function difference ``compute_V_bi()`` instead of the
    # frozen ``V_bi`` field (via ``MaterialArrays.V_bi_bc``). Keeps the
    # contact well-posed when a contact layer is weakly doped — the regime
    # where the default ideal-ohmic pin degenerates. Default False =
    # IonMonger-convention pins, bit-identical.
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

    def __post_init__(self):
        object.__setattr__(self, "layers", tuple(self.layers))

    @property
    def total_thickness(self) -> float:
        return sum(layer.thickness for layer in self.layers)

    @property
    def phi_right(self) -> float:
        return self.phi_left + self.V_bi

    def compute_V_bi(self) -> float:
        """Derive the built-in potential from the Fermi-level difference
        across the heterostack.

        Uses the numerically stable two-branch formula for majority carrier
        density: net = 0.5*(N_D - N_A), disc = sqrt(net**2 + ni**2), then
        n or p = net + disc  (or  -net + disc for minority branch).

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

        # When the outer layers are graded, the contacts sit at their face
        # endpoints: the left contact at the front (= the scalar params,
        # unchanged) and the right contact at the back endpoints. Non-graded
        # stacks return the identical params object → identical float.
        band_grading = getattr(self, "band_grading", False)
        left = _edge_params(elec[0].params, "front", band_grading)
        right = _edge_params(elec[-1].params, "back", band_grading)

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


def _edge_params(p: "MaterialParams", side: str, band_grading: bool) -> "MaterialParams":
    """Contact-face MaterialParams for ``compute_V_bi``.

    For an ungraded layer (or ``band_grading`` off) returns ``p`` unchanged so
    the Fermi level — and hence V_bi — is bit-identical. For a graded layer's
    BACK contact, returns a ``replace``-d copy whose chi/Eg/ni are the back
    endpoints (ni scaled by the same front-anchored DOS law as the solver:
    ni_back = ni·exp(-(Eg_back - Eg_front)/2V_T)). The front side never changes.
    """
    import dataclasses
    from perovskite_sim.physics.grading import has_grading_params
    if not band_grading or p is None or side != "back" or not has_grading_params(p):
        return p
    Eg_back = p.Eg_back if p.Eg_back is not None else p.Eg
    chi_back = p.chi_back if p.chi_back is not None else p.chi
    ni_back = p.ni * math.exp(-(Eg_back - p.Eg) / (2.0 * V_T))
    return dataclasses.replace(p, chi=chi_back, Eg=Eg_back, ni=ni_back)


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
