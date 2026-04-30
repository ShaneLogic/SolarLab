from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from perovskite_sim.constants import Q
from perovskite_sim.models.device import DeviceStack, electrical_layers
from perovskite_sim.physics.contacts import selective_contact_flux
from perovskite_sim.physics.recombination import total_recombination
from perovskite_sim.solver.mol import build_material_arrays as build_material_arrays_1d
from perovskite_sim.twod.continuity_2d import continuity_rhs_2d
from perovskite_sim.twod.field_mobility_2d import (
    arith_mean_face_x, arith_mean_face_y, arith_mean_face_wrap,
    recompute_d_eff_2d,
)
from perovskite_sim.twod.grid_2d import Grid2D
from perovskite_sim.twod.microstructure import Microstructure, build_tau_field
from perovskite_sim.twod.poisson_2d import (
    Poisson2DFactor, build_poisson_2d_factor, solve_poisson_2d,
)


@dataclass(frozen=True)
class MaterialArrays2D:
    """2D analogue of the 1D MaterialArrays cache.

    All per-node fields are shape (Ny, Nx). For Stage A every field is a
    uniform extrusion of the 1D MaterialArrays along x; Stage B will
    override τ_n and τ_p inside grain-boundary bands.

    Field-name notes vs. 1D MaterialArrays:
    - ``D_n`` / ``D_p``: per-node (Ny, Nx) diffusion coefficients. The 1D
      cache stores only per-face values (``D_n_face``, ``D_p_face``); here
      we re-derive the per-node values from layer mu * V_T so that the 2D
      Scharfetter–Gummel scheme can form its own face averages in any
      direction.
    - ``ni``: per-node intrinsic carrier density. The 1D cache stores
      ``ni_sq`` (squared); we take the square root here.
    - ``G_optical``: always a (Ny, Nx) ndarray. When the 1D stack uses
      Beer-Lambert (``MaterialArrays.G_optical is None``), we fill zeros
      and expect the 2D RHS to generate G from Beer-Lambert itself.
    """
    grid: Grid2D
    stack: DeviceStack
    ustruct: Microstructure
    eps_r: np.ndarray
    D_n: np.ndarray
    D_p: np.ndarray
    tau_n: np.ndarray
    tau_p: np.ndarray
    N_A: np.ndarray
    N_D: np.ndarray
    ni: np.ndarray
    G_optical: np.ndarray
    chi: np.ndarray               # (Ny, Nx) electron affinity [eV] — for heterostack SG
    Eg: np.ndarray                # (Ny, Nx) bandgap [eV] — for heterostack SG
    A_star_n: np.ndarray          # (Ny, Nx) Richardson constant for electrons [A/(m²·K²)]
    A_star_p: np.ndarray          # (Ny, Nx) Richardson constant for holes
    interface_y_faces: tuple[int, ...]  # y-face indices where TE capping applies (Stage A)
    T_device: float               # device temperature [K] — for TE flux
    # Stage A: ions are absent from the 2D state vector but contribute a frozen
    # background charge density. P_ion_static is the equilibrated ion profile
    # (defaults to P_ion0 — uniform initial). The Poisson rho gets an extra
    # Q*(P_ion_static - P_ion0_2d) term, matching the 1D charge-balance equation
    # so that 1D-converged states extrude to lateral-uniform 2D equilibria.
    P_ion0_2d: np.ndarray         # (Ny, Nx) initial uniform ion profile
    P_ion_static: np.ndarray      # (Ny, Nx) frozen ion profile (= P_ion0_2d unless overridden)
    # Full 1D recombination parameters extruded to 2D so total_recombination
    # produces the same R(n, p) as the 1D solver. Zeros disable individual
    # channels (Stage A defaults match 1D defaults from layer YAML).
    n1: np.ndarray                # (Ny, Nx) SRH electron trap-level density
    p1: np.ndarray                # (Ny, Nx) SRH hole trap-level density
    B_rad: np.ndarray             # (Ny, Nx) radiative recombination coefficient
    C_n: np.ndarray               # (Ny, Nx) Auger electron coefficient
    C_p: np.ndarray               # (Ny, Nx) Auger hole coefficient
    n_eq_left: np.ndarray         # (Nx,)  top contact (y=0, HTL); value = mat1d.n_L
    p_eq_left: np.ndarray         # (Nx,)  top contact (y=0, HTL); value = mat1d.p_L
    n_eq_right: np.ndarray        # (Nx,)  bottom contact (y=Ny-1, ETL); value = mat1d.n_R
    p_eq_right: np.ndarray        # (Nx,)  bottom contact (y=Ny-1, ETL); value = mat1d.p_R
    V_bi: float
    V_T: float
    poisson_factor: Poisson2DFactor
    layer_role_per_y: tuple[str, ...]
    # --- Stage B(c.1): Robin / selective contacts --------------------------------
    # has_selective_contacts is True iff any of the four S values is not None on
    # the originating DeviceStack.  S values default to 0.0 (zero = Neumann /
    # blocking; Dirichlet ohmic behaviour is restored via the assemble_rhs_2d
    # guard when has_selective_contacts is False).
    #
    # "left/right" are 1D transport-axis names inherited from DeviceStack.
    # In 2D the transport axis is y, so left→top (y=0, HTL) and right→bottom
    # (y=Ny-1, ETL).  The DeviceStack field names are intentionally unchanged.
    has_selective_contacts: bool = False
    S_n_top: float = 0.0   # electron SRV at y=0  (HTL); from DeviceStack.S_n_left
    S_p_top: float = 0.0   # hole    SRV at y=0  (HTL); from DeviceStack.S_p_left
    S_n_bot: float = 0.0   # electron SRV at y=Ny-1 (ETL); from DeviceStack.S_n_right
    S_p_bot: float = 0.0   # hole    SRV at y=Ny-1 (ETL); from DeviceStack.S_p_right
    # --- Stage B(c.2): Field-dependent mobility μ(E) ----------------------------
    # Face-normal formulation: x-faces use only |E_x_face|, y-faces use only
    # |E_y_face|.  See docs/superpowers/specs/2026-04-29-2d-stage-b-c2-field-
    # mobility-design.md.  All eighteen face arrays are None by default; they
    # are populated by build_material_arrays_2d when sim_mode.
    # use_field_dependent_mobility AND any layer sets v_sat>0 or pf_gamma>0.
    # The disabled path (has_field_mobility=False) is bit-identical to the
    # current Stage B(c.1) constant-mobility code path.
    has_field_mobility: bool = False
    # x-face arrays — (Ny, Nx-1)
    v_sat_n_x_face:    np.ndarray | None = None
    v_sat_p_x_face:    np.ndarray | None = None
    ct_beta_n_x_face:  np.ndarray | None = None
    ct_beta_p_x_face:  np.ndarray | None = None
    pf_gamma_n_x_face: np.ndarray | None = None
    pf_gamma_p_x_face: np.ndarray | None = None
    # y-face arrays — (Ny-1, Nx)
    v_sat_n_y_face:    np.ndarray | None = None
    v_sat_p_y_face:    np.ndarray | None = None
    ct_beta_n_y_face:  np.ndarray | None = None
    ct_beta_p_y_face:  np.ndarray | None = None
    pf_gamma_n_y_face: np.ndarray | None = None
    pf_gamma_p_y_face: np.ndarray | None = None
    # Periodic-x wrap face arrays — (Ny,); None unless lateral_bc=="periodic"
    v_sat_n_wrap:    np.ndarray | None = None
    v_sat_p_wrap:    np.ndarray | None = None
    ct_beta_n_wrap:  np.ndarray | None = None
    ct_beta_p_wrap:  np.ndarray | None = None
    pf_gamma_n_wrap: np.ndarray | None = None
    pf_gamma_p_wrap: np.ndarray | None = None
    # --- Stage B(c.3): Self-consistent radiative reabsorption ------------------
    # When mat.has_radiative_reabsorption_2d is True, assemble_rhs_2d augments
    # G_optical per RHS call by summing R_tot_2D = ∬ B·n·p dy dx per absorber
    # and adding the non-escaping fraction back as a uniform G_rad over the
    # absorber 2D area. See docs/superpowers/specs/2026-04-30-2d-stage-b-c3-
    # radiative-reabsorption-design.md. The disabled path (flag=False) is
    # bit-identical to current Stage B(c.2).
    has_radiative_reabsorption_2d:  bool                               = False
    absorber_y_ranges_2d:           tuple[tuple[int, int], ...]        = ()
    absorber_p_esc_2d:              tuple[float, ...]                  = ()
    absorber_thicknesses_2d:        tuple[float, ...]                  = ()
    absorber_areas_2d:              tuple[float, ...]                  = ()


def build_material_arrays_2d(
    grid: Grid2D,
    stack: DeviceStack,
    ustruct: Microstructure,
    *,
    lateral_bc: str = "periodic",
    P_ion_static_1d: np.ndarray | None = None,
) -> MaterialArrays2D:
    """Assemble the 2D MaterialArrays from a stack and a microstructure.

    Strategy: build the 1D MaterialArrays with the existing solver, then
    extrude every per-node field along x (Stage A has no x-features).
    τ_n, τ_p go through ``build_tau_field``, which respects GBs in Stage B
    but identity-extrudes when the microstructure is empty (Stage A).

    Field-name adaptations from the prescribed spec:
    - ``D_n`` / ``D_p`` are reconstructed per-node from layer mu * V_T,
      because the 1D ``MaterialArrays`` only caches per-face values
      (``D_n_face`` / ``D_p_face``).
    - ``ni`` is derived from ``sqrt(mat1d.ni_sq)``.
    - ``G_optical`` is zeroed when ``mat1d.G_optical is None`` (Beer-Lambert
      configs); the 2D RHS will supply Beer-Lambert generation instead.
    - ``V_T`` is read from ``mat1d.V_T_device`` (actual field name on 1D cache).
    """
    mat1d = build_material_arrays_1d(grid.y, stack)
    Nx, Ny = grid.Nx, grid.Ny

    def extrude(v_1d: np.ndarray) -> np.ndarray:
        """Broadcast a 1D per-y array to (Ny, Nx), returning a writeable copy."""
        return np.broadcast_to(v_1d[:, None], (Ny, Nx)).copy()

    eps_r = extrude(mat1d.eps_r)
    N_A = extrude(mat1d.N_A)
    N_D = extrude(mat1d.N_D)
    ni = extrude(np.sqrt(mat1d.ni_sq))
    chi = extrude(mat1d.chi)
    Eg = extrude(mat1d.Eg)

    # Richardson constants for thermionic-emission capping at heterointerfaces.
    # 1D MaterialArrays may leave A_star_n / A_star_p as None when TE is off.
    if mat1d.A_star_n is not None:
        A_star_n = extrude(mat1d.A_star_n)
    else:
        A_star_n = np.zeros((Ny, Nx), dtype=float)
    if mat1d.A_star_p is not None:
        A_star_p = extrude(mat1d.A_star_p)
    else:
        A_star_p = np.zeros((Ny, Nx), dtype=float)
    interface_y_faces = tuple(mat1d.interface_faces)
    T_device = float(mat1d.T_device)

    # Full recombination params from the 1D layer config — extruded uniformly
    # in x for Stage A so total_recombination matches 1D R(n, p) at every
    # node. Zeros disable individual channels (Stage A defaults follow YAML).
    n1 = extrude(mat1d.n1)
    p1 = extrude(mat1d.p1)
    B_rad = extrude(mat1d.B_rad)
    C_n_2d = extrude(mat1d.C_n)
    C_p_2d = extrude(mat1d.C_p)

    # Frozen ion background — required for 2D-1D parity because the 1D
    # Poisson rho includes Q*(P - P_ion0). Default to P_ion0 (uniform initial
    # state) so the contribution is zero on a cold start; pass an equilibrated
    # P_1d profile from solve_illuminated_ss when bootstrapping a warm-start.
    P_ion0_2d = extrude(mat1d.P_ion0)
    if P_ion_static_1d is None:
        P_ion_static = P_ion0_2d.copy()
    else:
        P_ion_static = extrude(np.asarray(P_ion_static_1d, dtype=float))

    # G_optical: 1D returns None for Beer-Lambert stacks (it computes BL at
    # runtime per voltage step). Stage A 2D pre-computes BL at build time
    # via the same helper used by 1D's runtime path; the result is shape
    # (Ny,) extruded to (Ny, Nx). For TMM stacks, mat1d.G_optical is already
    # populated and we just extrude.
    if mat1d.G_optical is not None:
        G_optical = extrude(mat1d.G_optical)
    elif mat1d.alpha is not None and float(stack.Phi) > 0.0:
        from perovskite_sim.physics.generation import beer_lambert_generation
        G_1d = beer_lambert_generation(grid.y, mat1d.alpha, stack.Phi)
        G_optical = extrude(G_1d)
    else:
        G_optical = np.zeros((Ny, Nx), dtype=float)

    # D_n / D_p per-node: reconstruct from layer mu * V_T because the 1D
    # MaterialArrays only stores per-face harmonic means (D_n_face, D_p_face).
    # We compute per-node values here so the 2D SG flux can form its own
    # directional face averages.
    V_T = float(mat1d.V_T_device)
    D_n_node_1d, D_p_node_1d = _diffusion_per_node(grid.y, stack, V_T)
    D_n = extrude(D_n_node_1d)
    D_p = extrude(D_p_node_1d)

    # tau: may be a 1D array or a scalar — normalise to (Ny,) then pass to
    # build_tau_field so Stage B grain-boundary overrides work correctly.
    tau_n_1d = np.atleast_1d(mat1d.tau_n)
    tau_p_1d = np.atleast_1d(mat1d.tau_p)
    if tau_n_1d.size == 1:
        tau_n_1d = np.full(Ny, float(tau_n_1d[0]))
    if tau_p_1d.size == 1:
        tau_p_1d = np.full(Ny, float(tau_p_1d[0]))

    layer_role_per_y = tuple(_layer_role_at_each_y(grid.y, stack))
    tau_n, tau_p = build_tau_field(
        grid, ustruct,
        tau_n_bulk_per_y=tau_n_1d,
        tau_p_bulk_per_y=tau_p_1d,
        layer_role_per_y=layer_role_per_y,
    )

    # Boundary equilibrium concentrations — scalars on the 1D side.
    # Broadcast to length Nx for x-uniform contacts.
    n_eq_left = np.full((Nx,), float(mat1d.n_L))
    p_eq_left = np.full((Nx,), float(mat1d.p_L))
    n_eq_right = np.full((Nx,), float(mat1d.n_R))
    p_eq_right = np.full((Nx,), float(mat1d.p_R))

    poisson_factor = build_poisson_2d_factor(grid, eps_r, lateral_bc=lateral_bc)

    # V_bi: must use stack.V_bi (the manual value the 1D Poisson BC uses) — not
    # V_bi_eff (the band-offset-derived value). The CLAUDE.md guidance pins this:
    # substituting V_bi_eff into the Poisson boundary breaks parity with the 1D
    # solver because IonMonger / 1D treats V_bi as a free parameter rather than
    # as the Fermi-level difference of the contacts.
    V_bi = float(stack.V_bi)

    # Selective contacts: mirror the 1D mol.py:516–524 gating exactly. The
    # tier flag is a ceiling — when sim_mode.use_selective_contacts is False
    # (LEGACY tier) the Robin path stays off even if the stack supplies S
    # values. This preserves the "tier as ceiling" invariant pinned by the
    # 1D test_tier_regression suite.
    from perovskite_sim.models.mode import resolve_mode
    sim_mode = resolve_mode(getattr(stack, "mode", "full"))
    _has_sc = bool(
        sim_mode.use_selective_contacts
        and (
            stack.S_n_left  is not None
            or stack.S_p_left  is not None
            or stack.S_n_right is not None
            or stack.S_p_right is not None
        )
    )
    S_n_top = float(stack.S_n_left)  if stack.S_n_left  is not None else 0.0
    S_p_top = float(stack.S_p_left)  if stack.S_p_left  is not None else 0.0
    S_n_bot = float(stack.S_n_right) if stack.S_n_right is not None else 0.0
    S_p_bot = float(stack.S_p_right) if stack.S_p_right is not None else 0.0

    # --- Stage B(c.2): Field-dependent mobility μ(E) ----------------------------
    # Build per-node v_sat / ct_beta / pf_gamma arrays from layer params via the
    # same y-mask construction used for D_n / D_p above.
    Ny_b, Nx_b = grid.Ny, grid.Nx
    v_sat_n_node    = np.zeros((Ny_b, Nx_b))
    v_sat_p_node    = np.zeros((Ny_b, Nx_b))
    ct_beta_n_node  = np.zeros((Ny_b, Nx_b))
    ct_beta_p_node  = np.zeros((Ny_b, Nx_b))
    pf_gamma_n_node = np.zeros((Ny_b, Nx_b))
    pf_gamma_p_node = np.zeros((Ny_b, Nx_b))
    _offset_fm = 0.0
    for _layer_fm in electrical_layers(stack):
        _mask_fm = (grid.y >= _offset_fm - 1e-12) & (
            grid.y <= _offset_fm + _layer_fm.thickness + 1e-12
        )
        _p_fm = _layer_fm.params
        v_sat_n_node[_mask_fm, :]    = _p_fm.v_sat_n
        v_sat_p_node[_mask_fm, :]    = _p_fm.v_sat_p
        ct_beta_n_node[_mask_fm, :]  = _p_fm.ct_beta_n
        ct_beta_p_node[_mask_fm, :]  = _p_fm.ct_beta_p
        pf_gamma_n_node[_mask_fm, :] = _p_fm.pf_gamma_n
        pf_gamma_p_node[_mask_fm, :] = _p_fm.pf_gamma_p
        _offset_fm += _layer_fm.thickness

    # Tier-as-ceiling activation gate (mirror 1D mol.py:502–509 and B(c.1) Robin gate).
    from perovskite_sim.models.mode import resolve_mode as _resolve_mode_fm
    _sim_mode_fm = _resolve_mode_fm(getattr(stack, "mode", "full"))
    _has_field_mobility = bool(
        _sim_mode_fm.use_field_dependent_mobility
        and (
            np.any(v_sat_n_node    > 0.0) or np.any(v_sat_p_node    > 0.0)
            or np.any(pf_gamma_n_node > 0.0) or np.any(pf_gamma_p_node > 0.0)
        )
    )

    if _has_field_mobility:
        # Arithmetic mean to faces (NOT harmonic — see field_mobility_2d.py docstring).
        v_sat_n_x_face    = arith_mean_face_x(v_sat_n_node)
        v_sat_n_y_face    = arith_mean_face_y(v_sat_n_node)
        v_sat_p_x_face    = arith_mean_face_x(v_sat_p_node)
        v_sat_p_y_face    = arith_mean_face_y(v_sat_p_node)
        ct_beta_n_x_face  = arith_mean_face_x(ct_beta_n_node)
        ct_beta_n_y_face  = arith_mean_face_y(ct_beta_n_node)
        ct_beta_p_x_face  = arith_mean_face_x(ct_beta_p_node)
        ct_beta_p_y_face  = arith_mean_face_y(ct_beta_p_node)
        pf_gamma_n_x_face = arith_mean_face_x(pf_gamma_n_node)
        pf_gamma_n_y_face = arith_mean_face_y(pf_gamma_n_node)
        pf_gamma_p_x_face = arith_mean_face_x(pf_gamma_p_node)
        pf_gamma_p_y_face = arith_mean_face_y(pf_gamma_p_node)
        if lateral_bc == "periodic":
            v_sat_n_wrap    = arith_mean_face_wrap(v_sat_n_node)
            v_sat_p_wrap    = arith_mean_face_wrap(v_sat_p_node)
            ct_beta_n_wrap  = arith_mean_face_wrap(ct_beta_n_node)
            ct_beta_p_wrap  = arith_mean_face_wrap(ct_beta_p_node)
            pf_gamma_n_wrap = arith_mean_face_wrap(pf_gamma_n_node)
            pf_gamma_p_wrap = arith_mean_face_wrap(pf_gamma_p_node)
        else:
            v_sat_n_wrap = v_sat_p_wrap = None
            ct_beta_n_wrap = ct_beta_p_wrap = None
            pf_gamma_n_wrap = pf_gamma_p_wrap = None
    else:
        v_sat_n_x_face = v_sat_n_y_face = None
        v_sat_p_x_face = v_sat_p_y_face = None
        ct_beta_n_x_face = ct_beta_n_y_face = None
        ct_beta_p_x_face = ct_beta_p_y_face = None
        pf_gamma_n_x_face = pf_gamma_n_y_face = None
        pf_gamma_p_x_face = pf_gamma_p_y_face = None
        v_sat_n_wrap = v_sat_p_wrap = None
        ct_beta_n_wrap = ct_beta_p_wrap = None
        pf_gamma_n_wrap = pf_gamma_p_wrap = None

    # --- Stage B(c.3): Radiative reabsorption ----------------------------------
    # Translate 1D mat1d.absorber_* tuples to 2D y-range form. Activation gate
    # mirrors 1D mol.py exactly: requires both tier flags AND the 1D-side
    # build path to have produced the per-absorber tuples.
    #
    # Index convention: the 2D y-range is derived from ``layer_role_per_y``
    # (the canonical 2D notion of "absorber rows", upstream-tagged at layer
    # boundaries) rather than the 1D ``mat1d.absorber_masks`` (which is the
    # inclusive ``offset <= y <= offset + thickness`` mask used by the 1D
    # solver hot path). The 1D mask includes the boundary node at the
    # HTL/absorber interface that ``_layer_role_at_each_y`` tags as "HTL"
    # (upstream); using ``layer_role_per_y`` here keeps the 2D y-ranges
    # internally consistent with the role-tag-driven microstructure painting
    # in build_tau_field. The 1D ``mat1d.absorber_p_esc`` /
    # ``absorber_thicknesses`` tuples (one entry per absorber, in stack
    # order) are zipped with the role-derived ranges in the same order.
    from perovskite_sim.models.mode import resolve_mode as _resolve_mode_rr
    _sim_mode_rr = _resolve_mode_rr(getattr(stack, "mode", "full"))
    absorber_y_ranges_list:    list[tuple[int, int]] = []
    absorber_p_esc_list:       list[float]            = []
    absorber_thicknesses_list: list[float]            = []
    absorber_areas_list:       list[float]            = []
    _has_rr_2d = False

    if (_sim_mode_rr.use_radiative_reabsorption
            and _sim_mode_rr.use_photon_recycling
            and getattr(mat1d, "has_radiative_reabsorption", False)):
        _lateral_length = float(grid.x[-1] - grid.x[0])
        # Derive contiguous absorber y-ranges from layer_role_per_y. Each run
        # of consecutive "absorber" indices is one entry; multi-absorber stacks
        # produce multiple ranges in stack order, matching the order of the
        # 1D mat1d.absorber_* tuples.
        _role_ranges: list[tuple[int, int]] = []
        _j = 0
        _Ny_total = len(layer_role_per_y)
        while _j < _Ny_total:
            if layer_role_per_y[_j] == "absorber":
                _y_lo = _j
                while _j < _Ny_total and layer_role_per_y[_j] == "absorber":
                    _j += 1
                _y_hi = _j  # half-open
                _role_ranges.append((_y_lo, _y_hi))
            else:
                _j += 1

        # Zip role-derived ranges with the 1D per-absorber p_esc/thickness
        # tuples in stack order. If the 1D side produced N absorber entries
        # but the role-tag derivation found M ranges, we pair the first
        # min(N, M) — they should always match in practice (one absorber
        # layer ⇒ one mask ⇒ one role run).
        for (_y_lo, _y_hi), _p_esc, _thickness in zip(
            _role_ranges, mat1d.absorber_p_esc, mat1d.absorber_thicknesses
        ):
            if _y_hi - _y_lo < 2:
                continue
            if _p_esc >= 1.0 or _thickness <= 0.0:
                continue
            absorber_y_ranges_list.append((_y_lo, _y_hi))
            absorber_p_esc_list.append(float(_p_esc))
            absorber_thicknesses_list.append(float(_thickness))
            absorber_areas_list.append(float(_thickness) * _lateral_length)
        _has_rr_2d = len(absorber_y_ranges_list) > 0

    absorber_y_ranges_2d   = tuple(absorber_y_ranges_list)
    absorber_p_esc_2d      = tuple(absorber_p_esc_list)
    absorber_thicknesses_2d = tuple(absorber_thicknesses_list)
    absorber_areas_2d      = tuple(absorber_areas_list)

    return MaterialArrays2D(
        grid=grid, stack=stack, ustruct=ustruct,
        eps_r=eps_r, D_n=D_n, D_p=D_p,
        tau_n=tau_n, tau_p=tau_p,
        N_A=N_A, N_D=N_D, ni=ni, G_optical=G_optical,
        chi=chi, Eg=Eg,
        A_star_n=A_star_n, A_star_p=A_star_p,
        interface_y_faces=interface_y_faces, T_device=T_device,
        P_ion0_2d=P_ion0_2d, P_ion_static=P_ion_static,
        n1=n1, p1=p1, B_rad=B_rad, C_n=C_n_2d, C_p=C_p_2d,
        n_eq_left=n_eq_left, p_eq_left=p_eq_left,
        n_eq_right=n_eq_right, p_eq_right=p_eq_right,
        V_bi=V_bi, V_T=V_T,
        poisson_factor=poisson_factor,
        layer_role_per_y=layer_role_per_y,
        has_selective_contacts=_has_sc,
        S_n_top=S_n_top, S_p_top=S_p_top,
        S_n_bot=S_n_bot, S_p_bot=S_p_bot,
        has_field_mobility=_has_field_mobility,
        v_sat_n_x_face=v_sat_n_x_face, v_sat_p_x_face=v_sat_p_x_face,
        ct_beta_n_x_face=ct_beta_n_x_face, ct_beta_p_x_face=ct_beta_p_x_face,
        pf_gamma_n_x_face=pf_gamma_n_x_face, pf_gamma_p_x_face=pf_gamma_p_x_face,
        v_sat_n_y_face=v_sat_n_y_face, v_sat_p_y_face=v_sat_p_y_face,
        ct_beta_n_y_face=ct_beta_n_y_face, ct_beta_p_y_face=ct_beta_p_y_face,
        pf_gamma_n_y_face=pf_gamma_n_y_face, pf_gamma_p_y_face=pf_gamma_p_y_face,
        v_sat_n_wrap=v_sat_n_wrap, v_sat_p_wrap=v_sat_p_wrap,
        ct_beta_n_wrap=ct_beta_n_wrap, ct_beta_p_wrap=ct_beta_p_wrap,
        pf_gamma_n_wrap=pf_gamma_n_wrap, pf_gamma_p_wrap=pf_gamma_p_wrap,
        has_radiative_reabsorption_2d=_has_rr_2d,
        absorber_y_ranges_2d=absorber_y_ranges_2d,
        absorber_p_esc_2d=absorber_p_esc_2d,
        absorber_thicknesses_2d=absorber_thicknesses_2d,
        absorber_areas_2d=absorber_areas_2d,
    )


def _diffusion_per_node(
    y: np.ndarray,
    stack: DeviceStack,
    V_T: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Rebuild per-node D_n, D_p from layer mobility × thermal voltage.

    The 1D MaterialArrays only stores per-face harmonic-mean diffusion
    coefficients (D_n_face, D_p_face, length Ny-1). The 2D solver needs
    per-node values (length Ny) to form face averages in both x and y.
    We replicate the same mu * V_T construction used inside
    build_material_arrays, without temperature scaling for now (Stage A).
    """
    from perovskite_sim.physics.temperature import mu_at_T, thermal_voltage
    from perovskite_sim.models.mode import resolve_mode

    sim_mode = resolve_mode(getattr(stack, "mode", "full"))
    if sim_mode.use_temperature_scaling:
        T_dev = stack.T
        V_T_actual = thermal_voltage(T_dev)
    else:
        T_dev = 300.0
        V_T_actual = V_T  # already computed by caller from mat1d.V_T_device

    Ny = len(y)
    D_n_node = np.empty(Ny)
    D_p_node = np.empty(Ny)

    elec = electrical_layers(stack)
    offset = 0.0
    for layer in elec:
        mask = (y >= offset - 1e-12) & (y <= offset + layer.thickness + 1e-12)
        p = layer.params
        mu_n = mu_at_T(p.mu_n, T_dev, p.mu_T_gamma)
        mu_p = mu_at_T(p.mu_p, T_dev, p.mu_T_gamma)
        D_n_node[mask] = mu_n * V_T_actual
        D_p_node[mask] = mu_p * V_T_actual
        offset += layer.thickness

    return D_n_node, D_p_node


def _charge_density_2d(n: np.ndarray, p: np.ndarray, mat: MaterialArrays2D) -> np.ndarray:
    """Space-charge density rho = q*(p - n + N_D - N_A + (P_ion - P_ion0)).

    Stage A holds the ion profile fixed at ``mat.P_ion_static``; defaults to
    ``mat.P_ion0_2d`` (uniform initial), which makes the ion term identically
    zero on a cold start. Passing an equilibrated 1D ion profile via
    ``P_ion_static_1d=`` to ``build_material_arrays_2d`` adds the matching
    1D background charge so 2D-1D parity holds on lateral-uniform states.
    """
    return Q * (p - n + mat.N_D - mat.N_A + (mat.P_ion_static - mat.P_ion0_2d))


def _apply_robin_contacts_2d(
    dn: np.ndarray,
    dp: np.ndarray,
    n: np.ndarray,
    p: np.ndarray,
    mat: "MaterialArrays2D",
) -> tuple[np.ndarray, np.ndarray]:
    """Replace Neumann wall flux with Robin contact flux at y=0 and y=Ny-1.

    ``continuity_rhs_2d`` already included a one-sided Neumann y-divergence
    at the boundary rows (zero wall flux assumed). This helper subtracts the
    implicit zero-flux assumption and adds the actual Robin flux
    J = ±q·S·(density − density_eq) instead. The four Dirichlet pins in
    ``assemble_rhs_2d`` are skipped when this helper is called.

    Outward-normal and side conventions:
      y=0    (top / HTL / 1D-left)   outward normal = −y  →  side="left"
      y=Ny−1 (bot / ETL / 1D-right)  outward normal = +y  →  side="right"

    Sign table (verified from first principles):
      dn[0,  :] −= J_n_top / (Q·hy_top)   electrons top:    dn = +div_n/Q
      dp[0,  :] += J_p_top / (Q·hy_top)   holes top:        dp = −div_p/Q → opposite sign
      dn[−1, :] += J_n_bot / (Q·hy_bot)   electrons bottom: J_n_bot < 0 when n > n_eq
      dp[−1, :] −= J_p_bot / (Q·hy_bot)   holes bottom:     J_p_bot > 0 when p > p_eq

    Returns NEW arrays (the inputs are not mutated). The caller is expected to
    rebind via ``dn, dp = _apply_robin_contacts_2d(...)``.
    """
    # Copy inputs to honour the non-mutating contract advertised in the
    # docstring.  Cost is one O(Ny·Nx) allocation per RHS call — negligible
    # vs the Radau LU on the same array, and immune to a future caller that
    # passes in arrays that must not be mutated (e.g. diagnostic harnesses).
    dn = dn.copy()
    dp = dp.copy()

    # Half-cell control-volume thickness at each contact boundary
    hy_top = (mat.grid.y[1]  - mat.grid.y[0])  / 2.0
    hy_bot = (mat.grid.y[-1] - mat.grid.y[-2]) / 2.0

    # --- top contact (y=0, HTL, side="left") --------------------------------
    # selective_contact_flux(carrier="n", side="left") = +Q·S·(n − n_eq)
    # selective_contact_flux(carrier="p", side="left") = −Q·S·(p − p_eq)
    J_n_top = selective_contact_flux(
        n[0, :], mat.n_eq_left, mat.S_n_top, carrier="n", side="left",
    )
    J_p_top = selective_contact_flux(
        p[0, :], mat.p_eq_left, mat.S_p_top, carrier="p", side="left",
    )
    dn[0, :] -= J_n_top / (Q * hy_top)   # subtract: dn = +div_n/Q
    dp[0, :] += J_p_top / (Q * hy_top)   # add:      dp = −div_p/Q (opposite sign)

    # --- bottom contact (y=Ny−1, ETL, side="right") -------------------------
    # selective_contact_flux(carrier="n", side="right") = −Q·S·(n − n_eq)
    # selective_contact_flux(carrier="p", side="right") = +Q·S·(p − p_eq)
    J_n_bot = selective_contact_flux(
        n[-1, :], mat.n_eq_right, mat.S_n_bot, carrier="n", side="right",
    )
    J_p_bot = selective_contact_flux(
        p[-1, :], mat.p_eq_right, mat.S_p_bot, carrier="p", side="right",
    )
    dn[-1, :] += J_n_bot / (Q * hy_bot)   # add:      J_n_bot < 0 when n > n_eq
    dp[-1, :] -= J_p_bot / (Q * hy_bot)   # subtract: J_p_bot > 0 when p > p_eq

    return dn, dp


def assemble_rhs_2d(
    t: float,
    y_state: np.ndarray,
    mat: MaterialArrays2D,
    V_app: float,
) -> np.ndarray:
    """Time-derivative of the flattened state (n, p) on the (Ny, Nx) grid.

    Flatten convention: C-order over (j, i) — y-major. Row j of the (Ny, Nx)
    array sits in y_state[j*Nx : (j+1)*Nx].

    Stage A scope:
      - Poisson via cached splu factor; Dirichlet at y=0 (phi=0) and
        y=Ly (phi = V_bi - V_app)
      - SG drift-diffusion fluxes on horizontal and vertical edges via
        continuity_rhs_2d (which calls sg_fluxes_2d_{n,p})
      - SRH recombination via total_recombination; B_rad=C_n=C_p=n1=p1=0
        (Stage A uses only SRH; no radiative/Auger, no trap-level offsets)
      - Optical generation from mat.G_optical (zero array for Beer-Lambert configs)
      - Lateral BC: periodic or Neumann per mat.poisson_factor.lateral_bc
      - Dirichlet contacts in y: dn[0,:]=dn[-1,:]=dp[0,:]=dp[-1,:]=0

    DONE_WITH_CONCERNS — total_recombination signature adaptation:
      total_recombination(n, p, ni_sq, tau_n, tau_p, n1, p1, B_rad, C_n, C_p)
      expects scalar or broadcastable per-node values. MaterialArrays2D stores
      ni (not ni_sq), tau_n, tau_p as 2D arrays but does NOT carry n1, p1,
      B_rad, C_n, C_p. Those are passed as zero scalars here so that only SRH
      with mid-gap traps (n1=p1=0 limit: R_SRH = (np - ni^2)/(tau_p*n + tau_n*p))
      contributes. The radiative and Auger channels are intentionally absent in
      Stage A; they will be added in Stage B by extending MaterialArrays2D with
      the missing fields (mirroring the 1D MaterialArrays layout).
    """
    g = mat.grid
    Nn = g.n_nodes
    n = y_state[:Nn].reshape((g.Ny, g.Nx))
    p = y_state[Nn:].reshape((g.Ny, g.Nx))

    # --- Poisson -----------------------------------------------------------
    rho = _charge_density_2d(n, p, mat)
    phi = solve_poisson_2d(
        mat.poisson_factor, rho,
        phi_bottom=0.0,
        phi_top=mat.V_bi - V_app,
    )

    # --- Recombination -----------------------------------------------------
    # total_recombination operates element-wise on flat arrays. All channels
    # (SRH + radiative + Auger) come from the extruded 1D layer config so
    # R(n, p) matches the 1D solver pointwise — required for V_oc / FF
    # parity in the validation gate.
    R = total_recombination(
        n=n.flatten(),
        p=p.flatten(),
        ni_sq=(mat.ni ** 2).flatten(),
        tau_n=mat.tau_n.flatten(),
        tau_p=mat.tau_p.flatten(),
        n1=mat.n1.flatten(),
        p1=mat.p1.flatten(),
        B_rad=mat.B_rad.flatten(),
        C_n=mat.C_n.flatten(),
        C_p=mat.C_p.flatten(),
    ).reshape((g.Ny, g.Nx))

    # --- Band-offset quasi-Fermi potentials --------------------------------
    # Use chi/Eg from MaterialArrays2D so the SG fluxes correctly account for
    # heterointerface band offsets.  When chi is uniform (homojunction), this
    # is bit-identical to the old chi=None path because a constant chi offset
    # cancels in the Bernoulli argument (phi_n[j+1] - phi_n[j] = phi[j+1] -
    # phi[j] when chi[j+1] == chi[j]).  For a heterostack (HTL/absorber/ETL)
    # omitting chi causes the SG flux to be wrong by exp(Δχ/V_T) at every
    # interface, making dark equilibrium impossible to reach from any start.
    chi_2d = mat.chi
    Eg_2d = mat.Eg

    # --- Continuity --------------------------------------------------------
    if mat.has_field_mobility:
        # Stage B(c.2): face-normal mu(E) recompute. x-faces use |E_x_face|,
        # y-faces use |E_y_face|. apply_field_mobility takes np.abs(E).
        d_eff = recompute_d_eff_2d(
            phi=phi, x=g.x, y=g.y,
            D_n=mat.D_n, D_p=mat.D_p, V_T=mat.V_T,
            v_sat_n_x_face=mat.v_sat_n_x_face,
            v_sat_n_y_face=mat.v_sat_n_y_face,
            ct_beta_n_x_face=mat.ct_beta_n_x_face,
            ct_beta_n_y_face=mat.ct_beta_n_y_face,
            pf_gamma_n_x_face=mat.pf_gamma_n_x_face,
            pf_gamma_n_y_face=mat.pf_gamma_n_y_face,
            v_sat_p_x_face=mat.v_sat_p_x_face,
            v_sat_p_y_face=mat.v_sat_p_y_face,
            ct_beta_p_x_face=mat.ct_beta_p_x_face,
            ct_beta_p_y_face=mat.ct_beta_p_y_face,
            pf_gamma_p_x_face=mat.pf_gamma_p_x_face,
            pf_gamma_p_y_face=mat.pf_gamma_p_y_face,
            lateral_bc=mat.poisson_factor.lateral_bc,
            v_sat_n_wrap=mat.v_sat_n_wrap,
            v_sat_p_wrap=mat.v_sat_p_wrap,
            ct_beta_n_wrap=mat.ct_beta_n_wrap,
            ct_beta_p_wrap=mat.ct_beta_p_wrap,
            pf_gamma_n_wrap=mat.pf_gamma_n_wrap,
            pf_gamma_p_wrap=mat.pf_gamma_p_wrap,
        )
        dn, dp = continuity_rhs_2d(
            g.x, g.y, phi, n, p,
            mat.G_optical, R,
            mat.D_n, mat.D_p,
            mat.V_T,
            chi=chi_2d,
            Eg=Eg_2d,
            lateral_bc=mat.poisson_factor.lateral_bc,
            interface_y_faces=mat.interface_y_faces,
            A_star_n=mat.A_star_n,
            A_star_p=mat.A_star_p,
            T=mat.T_device,
            D_n_x_face=d_eff.D_n_x, D_n_y_face=d_eff.D_n_y,
            D_p_x_face=d_eff.D_p_x, D_p_y_face=d_eff.D_p_y,
            D_n_wrap=d_eff.D_n_wrap, D_p_wrap=d_eff.D_p_wrap,
        )
    else:
        dn, dp = continuity_rhs_2d(
            g.x, g.y, phi, n, p,
            mat.G_optical, R,
            mat.D_n, mat.D_p,
            mat.V_T,
            chi=chi_2d,
            Eg=Eg_2d,
            lateral_bc=mat.poisson_factor.lateral_bc,
            interface_y_faces=mat.interface_y_faces,
            A_star_n=mat.A_star_n,
            A_star_p=mat.A_star_p,
            T=mat.T_device,
        )

    # --- Contact boundary conditions ---------------------------------------
    # Dirichlet (ohmic) path: pin all four boundary rows to zero (unchanged
    # from Stage A).  Robin path: apply surface-recombination flux correction
    # at each boundary row; the four pins are skipped entirely.
    if mat.has_selective_contacts:
        dn, dp = _apply_robin_contacts_2d(dn, dp, n, p, mat)
    else:
        dn[0, :] = 0.0
        dn[-1, :] = 0.0
        dp[0, :] = 0.0
        dp[-1, :] = 0.0

    return np.concatenate([dn.flatten(), dp.flatten()])


def _layer_role_at_each_y(y: np.ndarray, stack: DeviceStack) -> list[str]:
    """Return the layer role string at each y-node.

    Uses ``electrical_layers()`` (substrate filtered) so the cumulative
    thickness boundaries match the drift-diffusion grid built by
    ``multilayer_grid``.
    """
    layers = electrical_layers(stack)
    boundaries = [0.0]
    for L in layers:
        boundaries.append(boundaries[-1] + L.thickness)

    roles: list[str] = []
    for y_node in y:
        for k, L in enumerate(layers):
            if y_node <= boundaries[k + 1] + 1e-15:
                roles.append(getattr(L, "role", "absorber"))
                break
        else:
            roles.append(getattr(layers[-1], "role", "absorber"))
    return roles


from scipy.integrate import solve_ivp


class _RhsNonFinite2D(Exception):
    pass


def _assert_finite_2d(dydt: np.ndarray, V_app: float) -> None:
    if not np.all(np.isfinite(dydt)):
        raise _RhsNonFinite2D(f"non-finite dy/dt at V_app={V_app:.4f} V")


def run_transient_2d(
    y0: np.ndarray,
    mat: MaterialArrays2D,
    *,
    V_app: float,
    t_end: float,
    max_step: float | None = None,
    rtol: float = 1e-6,
    atol: float = 1e-8,
) -> np.ndarray:
    """Integrate dy/dt = assemble_rhs_2d(...) on [0, t_end] with Radau.

    Returns the state vector at t_end. Raises `_RhsNonFinite2D` if any
    RHS evaluation produces NaN/Inf.
    """
    def rhs(t: float, y_state: np.ndarray) -> np.ndarray:
        dydt = assemble_rhs_2d(t, y_state, mat, V_app)
        _assert_finite_2d(dydt, V_app)
        return dydt

    sol = solve_ivp(
        rhs, (0.0, t_end), y0,
        method="Radau",
        rtol=rtol, atol=atol,
        max_step=max_step if max_step is not None else np.inf,
        dense_output=False,
    )
    if not sol.success:
        raise RuntimeError(f"Radau failed at V_app={V_app:.4f} V: {sol.message}")
    return sol.y[:, -1]


from perovskite_sim.twod.snapshot import SpatialSnapshot2D
from perovskite_sim.twod.flux_2d import sg_fluxes_2d_n, sg_fluxes_2d_p


def extract_snapshot_2d(
    y_state: np.ndarray, mat: MaterialArrays2D, V_app: float,
) -> SpatialSnapshot2D:
    """Re-solve Poisson at the given (n, p) state and compute SG fluxes for
    a complete spatial snapshot. Used after run_transient_2d settles."""
    g = mat.grid
    Nn = g.n_nodes
    n = y_state[:Nn].reshape((g.Ny, g.Nx))
    p = y_state[Nn:].reshape((g.Ny, g.Nx))

    rho = _charge_density_2d(n, p, mat)
    phi = solve_poisson_2d(
        mat.poisson_factor, rho,
        phi_bottom=0.0,
        phi_top=mat.V_bi - V_app,
    )

    phi_n = phi + mat.chi
    phi_p = phi + mat.chi + mat.Eg
    Jx_n, Jy_n = sg_fluxes_2d_n(phi_n, n, g.x, g.y, mat.D_n, mat.V_T)
    Jx_p, Jy_p = sg_fluxes_2d_p(phi_p, p, g.x, g.y, mat.D_p, mat.V_T)

    return SpatialSnapshot2D(
        V=float(V_app),
        x=g.x.copy(), y=g.y.copy(),
        phi=phi, n=n.copy(), p=p.copy(),
        Jx_n=Jx_n, Jy_n=Jy_n, Jx_p=Jx_p, Jy_p=Jy_p,
    )


def compute_terminal_current_2d(snap: SpatialSnapshot2D) -> float:
    """Lateral-average of total current density (electrons + holes) at the
    top contact (y = Ly), units A/m².

    J_y is defined on edges between grid rows j and j+1; the top-most edge
    sits between j=Ny-2 and j=Ny-1 (i.e. row index -1 of Jy arrays).
    Trapezoidal integration over x handles non-uniform x spacing.
    """
    Jy_top_n = snap.Jy_n[-1, :]      # (Nx,)
    Jy_top_p = snap.Jy_p[-1, :]      # (Nx,)
    dx = np.diff(snap.x)             # (Nx-1,)
    L_x = float(snap.x[-1] - snap.x[0])
    avg_n = float(np.sum((Jy_top_n[:-1] + Jy_top_n[1:]) / 2.0 * dx) / L_x)
    avg_p = float(np.sum((Jy_top_p[:-1] + Jy_top_p[1:]) / 2.0 * dx) / L_x)
    return avg_n + avg_p
