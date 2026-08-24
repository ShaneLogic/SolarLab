from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Literal
import dataclasses
import numpy as np

from perovskite_sim.discretization.grid import Layer
from perovskite_sim.experiments.protocol import ImplicitProtocolError, ProtocolMode
from perovskite_sim.experiments.jv_sweep import JVMetrics, compute_metrics
from perovskite_sim.models.device import DeviceStack, electrical_layers
from perovskite_sim.models.mode import resolve_mode
from perovskite_sim.solver.tolerances import AbsoluteTolerance
from perovskite_sim.twod.experiments.jv_protocol_2d import (
    JV2DProtocol,
    build_jv_2d_protocol,
    resolve_jv_2d_protocol,
)
from perovskite_sim.twod.grid_2d import build_grid_2d
from perovskite_sim.twod.ion_migration_2d import (
    MobileIonDiagnostics2D,
    assess_mobile_ion_terminal_2d,
)
from perovskite_sim.twod.interface_recombination_2d import (
    TwoSidedInterfaceSRHReport2D,
    evaluate_two_sided_interface_srh_2d,
)
from perovskite_sim.twod.mobile_ion_current_2d import (
    MobileIonCurrentComponents2D,
)
from perovskite_sim.twod.microstructure import Microstructure
from perovskite_sim.twod.radiative_reabsorption_2d import recompute_g_with_rad_2d
from perovskite_sim.twod.solver_2d import (
    build_material_arrays_2d,
    compute_mobile_ion_current_components_2d,
    run_transient_2d,
    extract_snapshot_2d,
    compute_terminal_current_2d,
)
from perovskite_sim.twod.snapshot import SpatialSnapshot2D


ProgressCallback = Callable[[str, int, int, str], None]
"""Callable protocol: fn(stage, current, total, message) -> None."""


_TWOD_RADAU_MAX_NFEV = 200_000
"""Per-step Newton-iteration budget. 2× the 1D ``_JV_RADAU_MAX_NFEV``
because the 2D Jacobian is larger and Newton contracts more slowly per
iteration; borderline TMM-preset cases (e.g. ``test_jv_sweep_2d_uniform``
at V=0.5 V) cleanly need ~100-150k RHS calls, and fluctuations in
multi-threaded BLAS scheduling can push them just over a 100k cap.

Without this cap the 2D Radau implicit step can spin indefinitely on
nearly singular Jacobians — the canonical case is the diode-injection
knee at V ≈ 0.21 V on TMM presets when Phase 3.1b reabsorption is on,
where the non-local ``R_tot = ∬ B·n·p dy dx`` source destroys Newton
contraction. The cap converts the hang into a fast ``RuntimeError``
(at this budget, ~30-60 s wall time) so the lagged-fallback retry
below can take over.
"""


@dataclass(frozen=True)
class JV2DResult:
    """Result of a forward illuminated 2D J-V sweep.

    V : applied voltages, shape (n_points,), V
    J : terminal current density at each voltage, shape (n_points,), A/m²
    snapshots : per-voltage SpatialSnapshot2D (empty tuple when save_snapshots=False)
    grid_x : lateral grid coordinates, shape (Nx,), m
    grid_y : vertical grid coordinates, shape (Ny,), m
    lateral_bc : lateral boundary condition used in the sweep ("periodic" | "neumann")
    metrics : JVMetrics extracted from (V, J) via the centralized
        ``compute_metrics`` (sign-normalized to the J_sc-positive
        convention via ``assume_jsc_positive=False``). Carries
        ``voc_bracketed=False`` when V_max stopped short of V_oc; the
        backend SSE handler surfaces this to the frontend so the user
        can be told to expand the sweep range. Raw V/J arrays are not
        modified by this extraction.
    """
    V: np.ndarray
    J: np.ndarray
    snapshots: tuple[SpatialSnapshot2D, ...]
    grid_x: np.ndarray
    grid_y: np.ndarray
    lateral_bc: str
    metrics: JVMetrics = dataclasses.field(
        default_factory=lambda: JVMetrics(0.0, 0.0, 0.0, 0.0, voc_bracketed=False)
    )
    protocol: JV2DProtocol | None = None
    current_components: tuple[MobileIonCurrentComponents2D, ...] = ()
    ion_diagnostics: tuple[MobileIonDiagnostics2D, ...] = ()
    interface_srh_diagnostics: tuple[TwoSidedInterfaceSRHReport2D, ...] = ()


def _bake_radiative_reabsorption_step_2d(
    y_state: np.ndarray, mat, illuminated: bool,
):
    """Freeze the Stage B(c.3) G_rad source for one ``run_transient_2d`` call.

    Stage B(c.3)'s per-RHS hook recomputes ``R_tot_2D = ∬ B·n·p dy dx`` inside
    every Radau Newton iteration, which couples every absorber cell to every
    other through a non-local integral. At low forward bias on TMM presets
    (V≈0.21V — see project memory `tmm_jv_regression_021.md`), the diode-
    injection knee can prevent Newton convergence on the dense absorber block.

    Fix (mirrors 1D ``_bake_radiative_reabsorption_step`` in jv_sweep.py): on
    ``run_transient_2d`` failure, evaluate ``R_tot_2D`` once at the entry state
    ``y_state``, fold ``G_rad`` into a step-local ``G_optical`` copy, and clear
    ``has_radiative_reabsorption_2d`` on the returned ``mat``. Across voltage
    steps the warm-start chain refreshes ``R_tot`` from the freshly-settled
    state, so the lag is bounded by ``n·p`` drift inside one settle interval —
    sub-percent on the typical ``v_rate=1 V/s`` sweep, well below the 5 mV V_oc
    parity window.

    No-op when ``has_radiative_reabsorption_2d=False``, ``illuminated=False``,
    or there are no absorbers — returns the original ``mat``.
    """
    if not (
        mat.has_radiative_reabsorption_2d
        and illuminated
        and mat.absorber_y_ranges_2d
    ):
        return mat
    g = mat.grid
    N = g.Ny * g.Nx
    n0 = y_state[:N].reshape((g.Ny, g.Nx))
    p0 = y_state[N : 2 * N].reshape((g.Ny, g.Nx))
    G_with_rad = recompute_g_with_rad_2d(
        G_optical=mat.G_optical, n=n0, p=p0, B_rad=mat.B_rad,
        ni_sq=mat.ni ** 2,
        x=g.x, y=g.y,
        absorber_y_ranges=mat.absorber_y_ranges_2d,
        absorber_p_esc=mat.absorber_p_esc_2d,
        absorber_areas=mat.absorber_areas_2d,
    )
    return dataclasses.replace(
        mat,
        G_optical=G_with_rad,
        has_radiative_reabsorption_2d=False,
        absorber_y_ranges_2d=(),
        absorber_p_esc_2d=(),
        absorber_thicknesses_2d=(),
        absorber_areas_2d=(),
    )


def _integrate_step_2d(
    y_state: np.ndarray,
    mat,
    *,
    V_app: float,
    settle_t: float,
    illuminated: bool,
    max_bisect: int = 6,
    rtol: float = 1.0e-6,
    atol: AbsoluteTolerance = 1.0e-8,
    max_nfev_per_solve: int = _TWOD_RADAU_MAX_NFEV,
    max_step_divisor: int = 50,
    ion_inventory_rtol: float = 1.0e-9,
) -> np.ndarray:
    """Settle the 2D state at fixed V_app from t=0 to t=settle_t. Mirror of
    1D ``_integrate_step`` (perovskite_sim/experiments/jv_sweep.py:381+).

    Strategy
    --------
    1. Try ``run_transient_2d`` on the full ``settle_t`` interval. Success
       returns immediately — the typical hot path.
    2. On ``RuntimeError`` (max_nfev exceeded, RHS non-finite, or Radau
       reporting non-success), if Stage B(c.3) reabsorption is on, bake
       ``R_tot`` once at the entry state and retry with the per-RHS hook
       disabled. Per `project_tmm_jv_regression_021.md`, this fixes the
       1D V≈0.21 V regression and is mirrored here for symmetry.
    3. On further failure, if ``max_bisect == 0``: raise — bisection
       budget exhausted.
    4. Otherwise: split ``settle_t`` into halves and recurse on each
       half (warm-chained from the midpoint state). Each recursive call
       gets ``max_bisect - 1`` so the worst-case fan-out is bounded at
       ``2 ** max_bisect`` sub-intervals.

    Why bisection helps
    -------------------
    The 2D Stage A solver cannot Newton-contract through the V≈0.2 V
    diode-injection knee on TMM presets when given the full ``settle_t``
    interval (`project_stage_a_2d_tmm_newton.md`). Halving the time
    interval shrinks the per-step state change, which in turn relaxes
    the Newton contraction requirement. At the deepest bisection level
    (``max_bisect=6``) the interval is ``settle_t / 64``, typically
    ~1.5e-5 s — well inside the linear-response regime where Newton is
    not stress-tested.

    Parameters
    ----------
    y_state : np.ndarray
        Packed ``(n, p[, P])`` 2D state vector.
    mat : MaterialArrays2D
    V_app : float
        Applied voltage [V].
    settle_t : float
        Length of the integration interval [s].
    illuminated : bool
        Required so the lagged-fallback knows whether the bake-R_tot
        path is admissible — re-using the 1D semantics.
    max_bisect : int, default 6
        Recursion depth budget. 6 → up to 64 sub-intervals.

    Returns
    -------
    np.ndarray : settled 2D state at t=settle_t.

    Raises
    ------
    RuntimeError if the bisection budget is exhausted.
    """
    try:
        return run_transient_2d(
            y_state, mat,
            V_app=V_app, t_end=settle_t,
            max_step=settle_t / max_step_divisor,
            rtol=rtol,
            atol=atol,
            max_nfev=max_nfev_per_solve,
            ion_inventory_rtol=ion_inventory_rtol,
        )
    except RuntimeError:
        pass

    if mat.has_radiative_reabsorption_2d and illuminated:
        mat_step = _bake_radiative_reabsorption_step_2d(
            y_state, mat, illuminated=illuminated,
        )
        try:
            return run_transient_2d(
                y_state, mat_step,
                V_app=V_app, t_end=settle_t,
                max_step=settle_t / max_step_divisor,
                rtol=rtol,
                atol=atol,
                max_nfev=max_nfev_per_solve,
                ion_inventory_rtol=ion_inventory_rtol,
            )
        except RuntimeError:
            pass

    if max_bisect == 0:
        raise RuntimeError(
            f"2D JV sweep: coupled solver failed to converge at "
            f"V_app={V_app:.4f} V on settle_t={settle_t:.3e} after "
            f"bisection budget exhausted"
        )

    t_half = 0.5 * settle_t
    y_mid = _integrate_step_2d(
        y_state, mat,
        V_app=V_app, settle_t=t_half,
        illuminated=illuminated, max_bisect=max_bisect - 1,
        rtol=rtol,
        atol=atol,
        max_nfev_per_solve=max_nfev_per_solve,
        max_step_divisor=max_step_divisor,
        ion_inventory_rtol=ion_inventory_rtol,
    )
    return _integrate_step_2d(
        y_mid, mat,
        V_app=V_app, settle_t=t_half,
        illuminated=illuminated, max_bisect=max_bisect - 1,
        rtol=rtol,
        atol=atol,
        max_nfev_per_solve=max_nfev_per_solve,
        max_step_divisor=max_step_divisor,
        ion_inventory_rtol=ion_inventory_rtol,
    )


def _resolve_microstructure(
    stack: DeviceStack,
    microstructure: Microstructure | None,
) -> Microstructure:
    if microstructure is not None:
        if not isinstance(microstructure, Microstructure):
            raise TypeError("microstructure must be a Microstructure")
        return microstructure
    resolved = getattr(stack, "microstructure", None) or Microstructure()
    if not isinstance(resolved, Microstructure):
        raise TypeError("stack.microstructure must be a Microstructure")
    return resolved


def _jv_2d_voltage_values(V_max: float, V_step: float) -> np.ndarray:
    maximum = float(V_max)
    step = float(V_step)
    if not np.isfinite(maximum) or maximum < 0.0:
        raise ValueError("V_max must be finite and non-negative")
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("V_step must be finite and positive")
    return np.arange(0.0, maximum + step / 2.0, step, dtype=float)


def _jv_2d_grid(
    stack: DeviceStack,
    *,
    lateral_length: float,
    Nx: int,
    Ny_per_layer: int,
):
    elec = electrical_layers(stack)
    layers = [Layer(layer.thickness, Ny_per_layer) for layer in elec]
    return build_grid_2d(
        layers,
        lateral_length=lateral_length,
        Nx=Nx,
        lateral_uniform=True,
    )


def _carrier_boundary_condition(
    stack: DeviceStack,
) -> Literal["ohmic", "selective_robin"]:
    mode = resolve_mode(getattr(stack, "mode", "full"))
    has_selective = mode.use_selective_contacts and any(
        value is not None
        for value in (
            stack.S_n_left,
            stack.S_p_left,
            stack.S_n_right,
            stack.S_p_right,
        )
    )
    return "selective_robin" if has_selective else "ohmic"


def build_jv_2d_execution_protocol(
    stack: DeviceStack,
    microstructure: Microstructure | None = None,
    *,
    lateral_length: float,
    Nx: int,
    V_max: float,
    V_step: float,
    illuminated: bool = True,
    lateral_bc: Literal["periodic", "neumann"] = "periodic",
    Ny_per_layer: int = 20,
    settle_t: float = 1.0e-7,
    save_snapshots: bool = True,
    ion_dynamics: Literal["frozen", "single_mobile"] = "frozen",
    interface_srh: Literal["off", "two_sided_cross_node"] = "off",
    rtol: float = 1.0e-6,
    atol: AbsoluteTolerance = 1.0e-8,
    max_nfev_per_solve: int = _TWOD_RADAU_MAX_NFEV,
    max_bisect: int = 6,
    ion_inventory_rtol: float = 1.0e-9,
    initial_state_settle_s: float = 1.0e-3,
    implicit_legacy_protocol: bool = False,
) -> JV2DProtocol:
    """Build the exact protocol expected by :func:`run_jv_sweep_2d`."""
    resolved_microstructure = _resolve_microstructure(stack, microstructure)
    grid = _jv_2d_grid(
        stack,
        lateral_length=lateral_length,
        Nx=Nx,
        Ny_per_layer=Ny_per_layer,
    )
    voltages = _jv_2d_voltage_values(V_max, V_step)
    return build_jv_2d_protocol(
        temperature_K=stack.T,
        illuminated=illuminated,
        grid=grid,
        microstructure=resolved_microstructure,
        voltages_V=voltages,
        dwell_time_per_voltage_s=settle_t,
        ion_dynamics=ion_dynamics,
        carrier_boundary_condition=_carrier_boundary_condition(stack),
        interface_srh=interface_srh,
        lateral_bc=lateral_bc,
        solver_rtol=rtol,
        solver_atol=atol,
        max_nfev_per_solve=max_nfev_per_solve,
        max_bisect=max_bisect,
        ion_inventory_rtol=ion_inventory_rtol,
        save_snapshots=save_snapshots,
        initial_state_settle_s=initial_state_settle_s,
        implicit_legacy_protocol=implicit_legacy_protocol,
    )


def run_jv_sweep_2d(
    stack: DeviceStack,
    microstructure: Microstructure | None = None,
    *,
    lateral_length: float,
    Nx: int,
    V_max: float,
    V_step: float,
    illuminated: bool = True,
    lateral_bc: Literal["periodic", "neumann"] = "periodic",
    Ny_per_layer: int = 20,
    settle_t: float = 1.0e-7,
    progress: ProgressCallback | None = None,
    save_snapshots: bool = True,
    ion_dynamics: Literal["frozen", "single_mobile"] = "frozen",
    interface_srh: Literal["off", "two_sided_cross_node"] = "off",
    rtol: float = 1.0e-6,
    atol: AbsoluteTolerance = 1.0e-8,
    max_nfev_per_solve: int = _TWOD_RADAU_MAX_NFEV,
    max_bisect: int = 6,
    ion_inventory_rtol: float = 1.0e-9,
    initial_state_settle_s: float = 1.0e-3,
    jv_2d_protocol: JV2DProtocol | None = None,
    protocol_mode: ProtocolMode = "compatibility",
) -> JV2DResult:
    """Forward J-V sweep on a 2D grid.

    Each voltage point performs a complete fixed-voltage dwell from the
    previous settled state. The default remains the historical frozen-ion
    ``(n, p)`` lane with an implicit compatibility protocol. The explicit
    ``single_mobile`` and/or ``two_sided_cross_node`` research topology
    requires a matching :class:`JV2DProtocol` in ``research_strict`` mode.

    The voltage grid walks from 0 → V_max in steps of V_step using
    ``np.arange(0.0, V_max + V_step/2, V_step)``, matching the 1D convention.

    Mobile-ion current is sampled instantaneously at the fixed-voltage dwell
    endpoint with explicit ``dV/dt=0`` and contains carrier conduction,
    positive-ion, and differentiated-Poisson displacement current. The result
    retains per-point current, ion-inventory, and interface-SRH evidence.

    Parameters
    ----------
    stack : DeviceStack
        Device configuration.  Use a TMM-enabled preset so G_optical is
        populated at build time.
    microstructure : Microstructure
        Grain-boundary microstructure descriptor.  Pass ``Microstructure()``
        (empty) for the uniform Stage A case.
    lateral_length : float
        Width of the simulation domain in metres.
    Nx : int
        Number of lateral grid *intervals*; the lateral grid has Nx+1 nodes
        (``build_grid_2d`` convention).
    V_max : float
        Upper voltage limit for the sweep (V).
    V_step : float
        Voltage increment between sweep points (V).
    illuminated : bool
        When False, zero out G_optical for a dark J-V.
    lateral_bc : str
        Lateral boundary condition: ``"periodic"`` (default) or ``"neumann"``.
    Ny_per_layer : int
        Number of vertical grid intervals per electrical layer.
    settle_t : float
        Transient integration time at each voltage step (s).  Should be long
        enough for carriers to relax but short relative to ion drift time.
    progress : ProgressCallback | None
        Optional progress callback ``fn(stage, current, total, message)``.
    save_snapshots : bool
        When True, collect a ``SpatialSnapshot2D`` at every voltage point.
    ion_dynamics : {"frozen", "single_mobile"}
        State topology. Mobile ions require Neumann-x and ohmic contacts.
    interface_srh : {"off", "two_sided_cross_node"}
        Optional conservative cross-node interface sheet sink.
    jv_2d_protocol : JV2DProtocol | None
        Exact execution declaration. Required for either research topology.
    protocol_mode : {"compatibility", "research_strict"}
        Strict mode rejects generated implicit history.

    Returns
    -------
    JV2DResult
        Sweep results including V array, J array, optional snapshots, and grid
        coordinates.
    """
    microstructure = _resolve_microstructure(stack, microstructure)
    grid = _jv_2d_grid(
        stack,
        lateral_length=lateral_length,
        Nx=Nx,
        Ny_per_layer=Ny_per_layer,
    )
    voltages = _jv_2d_voltage_values(V_max, V_step)
    expected_protocol = build_jv_2d_protocol(
        temperature_K=stack.T,
        illuminated=illuminated,
        grid=grid,
        microstructure=microstructure,
        voltages_V=voltages,
        dwell_time_per_voltage_s=settle_t,
        ion_dynamics=ion_dynamics,
        carrier_boundary_condition=_carrier_boundary_condition(stack),
        interface_srh=interface_srh,
        lateral_bc=lateral_bc,
        solver_rtol=rtol,
        solver_atol=atol,
        max_nfev_per_solve=max_nfev_per_solve,
        max_bisect=max_bisect,
        ion_inventory_rtol=ion_inventory_rtol,
        save_snapshots=save_snapshots,
        initial_state_settle_s=initial_state_settle_s,
        implicit_legacy_protocol=True,
    )
    extended_topology = ion_dynamics != "frozen" or interface_srh != "off"
    if extended_topology and protocol_mode != "research_strict":
        raise ImplicitProtocolError(
            "mobile-ion or interface-SRH 2D J-V requires protocol_mode="
            "'research_strict' and an explicit matching jv_2d_protocol"
        )
    resolved_protocol = resolve_jv_2d_protocol(
        jv_2d_protocol,
        expected_protocol,
        mode=protocol_mode,
    )
    mobile_ions = resolved_protocol.state_topology == (
        "single_positive_mobile_ion"
    )
    resolved_ion_dynamics: Literal["frozen", "single_mobile"] = (
        "single_mobile" if mobile_ions else "frozen"
    )
    resolved_atol = resolved_protocol.solver_atol.to_absolute_tolerance()

    # Capability-check research topology before the finite-time 1D
    # preconditioner. The mobile material can be reused; a frozen-ion material
    # is rebuilt below with the actual preconditioned P profile.
    mat = None
    if extended_topology:
        mat = build_material_arrays_2d(
            grid,
            stack,
            microstructure,
            lateral_bc=resolved_protocol.lateral_bc,
            ion_dynamics=resolved_ion_dynamics,
            interface_srh=resolved_protocol.interface_srh,
        )

    # --- Warm-start via the 1D solver --------------------------------------
    # Stage A holds ions frozen and the (n, p) state laterally uniform, so a
    # 1D-equilibrated state is the correct 2D dark/illuminated equilibrium.
    # We call the 1D ``solve_illuminated_ss`` (or ``solve_equilibrium`` for
    # dark sweeps) on the same y-grid, freeze the resulting ion profile P
    # into the 2D Poisson background, and broadcast (n, p) across x.
    #
    # Why this matters: the 1D Poisson rho includes Q*(P − P_ion0). Without
    # passing the equilibrated P as P_ion_static_1d, the 2D Poisson sees a
    # different rho than 1D, the 2D phi diverges from 1D phi, and the SG
    # fluxes at the heterointerfaces blow up by ~30 orders of magnitude.
    from perovskite_sim.solver.illuminated_ss import solve_illuminated_ss
    from perovskite_sim.solver.newton import solve_equilibrium
    from perovskite_sim.solver.mol import StateVec

    if resolved_protocol.illuminated:
        assert resolved_protocol.initial_state_settle_s is not None
        y_1d = solve_illuminated_ss(
            grid.y,
            stack,
            V_app=resolved_protocol.initial_state_voltage_V,
            t_settle=resolved_protocol.initial_state_settle_s,
        )
    else:
        y_1d = solve_equilibrium(grid.y, stack)

    sv1d = StateVec.unpack(y_1d, len(grid.y))
    n_1d, p_1d, P_1d = sv1d.n, sv1d.p, sv1d.P

    if mat is None or not mobile_ions:
        mat = build_material_arrays_2d(
            grid,
            stack,
            microstructure,
            lateral_bc=resolved_protocol.lateral_bc,
            P_ion_static_1d=P_1d,
            ion_dynamics=resolved_ion_dynamics,
            interface_srh=resolved_protocol.interface_srh,
        )

    if not resolved_protocol.illuminated:
        mat = dataclasses.replace(mat, G_optical=np.zeros_like(mat.G_optical))

    Ny, Nx_nodes = grid.Ny, grid.Nx
    n_2d_init = np.broadcast_to(n_1d[:, None], (Ny, Nx_nodes)).copy()
    p_2d_init = np.broadcast_to(p_1d[:, None], (Ny, Nx_nodes)).copy()
    state_blocks = [n_2d_init.ravel(), p_2d_init.ravel()]
    if mobile_ions:
        P_2d_init = np.broadcast_to(P_1d[:, None], (Ny, Nx_nodes)).copy()
        state_blocks.append(P_2d_init.ravel())
    y_state = np.concatenate(state_blocks)

    # --- Voltage sweep -------------------------------------------------------
    voltages = np.asarray(resolved_protocol.voltage_values_V, dtype=float)
    J_list: list[float] = []
    snap_list: list[SpatialSnapshot2D] = []
    current_reports: list[MobileIonCurrentComponents2D] = []
    ion_reports: list[MobileIonDiagnostics2D] = []
    interface_reports: list[TwoSidedInterfaceSRHReport2D] = []
    block_size = Ny * Nx_nodes

    for k, V in enumerate(voltages):
        # Bisection-in-time recovery (mirrors 1D _integrate_step): primary
        # attempt → Stage B(c.3) lagged-fallback retry (when rr is on) →
        # halve settle_t and recurse, up to 2**6 = 64 sub-intervals.
        initial_point_state = y_state.copy()
        y_state = _integrate_step_2d(
            y_state, mat,
            V_app=float(V),
            settle_t=resolved_protocol.dwell_time_per_voltage_s,
            illuminated=resolved_protocol.illuminated,
            max_bisect=resolved_protocol.max_bisect,
            rtol=resolved_protocol.solver_rtol,
            atol=resolved_atol,
            max_nfev_per_solve=resolved_protocol.max_nfev_per_solve,
            max_step_divisor=resolved_protocol.solver_max_step_divisor,
            ion_inventory_rtol=resolved_protocol.ion_inventory_rtol,
        )
        snap = extract_snapshot_2d(y_state, mat, V_app=float(V))
        if mat.interface_srh_couplings:
            interface_reports.append(
                evaluate_two_sided_interface_srh_2d(
                    snap.n,
                    snap.p,
                    grid.y,
                    mat.interface_srh_couplings,
                )
            )
        if mobile_ions:
            current = compute_mobile_ion_current_components_2d(
                y_state,
                mat,
                float(V),
                applied_voltage_rate_V_s=(
                    resolved_protocol.applied_voltage_rate_at_sampling_V_s
                ),
            )
            J_list.append(current.terminal_total_A_m2)
            current_reports.append(current)
            previous_P = initial_point_state[2 * block_size :].reshape(Ny, Nx_nodes)
            terminal_n = y_state[:block_size].reshape(Ny, Nx_nodes)
            terminal_p = y_state[block_size : 2 * block_size].reshape(Ny, Nx_nodes)
            terminal_P = y_state[2 * block_size :].reshape(Ny, Nx_nodes)
            assert mat.P_lim_2d is not None
            diagnostics = assess_mobile_ion_terminal_2d(
                grid.x,
                grid.y,
                previous_P,
                terminal_P,
                mat.P_lim_2d,
                terminal_electron_density=terminal_n,
                terminal_hole_density=terminal_p,
                inventory_rtol=resolved_protocol.ion_inventory_rtol,
            )
            if not diagnostics.passed:
                raise RuntimeError(
                    "2D J-V point failed aggregate mobile-ion diagnostics: "
                    + ", ".join(diagnostics.violations)
                )
            ion_reports.append(diagnostics)
        else:
            J_list.append(compute_terminal_current_2d(snap))
        if resolved_protocol.save_snapshots:
            snap_list.append(snap)
        if progress is not None:
            progress("jv_2d", k + 1, len(voltages), f"V = {V:.3f} V")

    J_arr = np.array(J_list, dtype=float)
    # Centralised metrics extraction (Layer 1 of Phase 6 acceptance follow-up).
    # 2D backend emits J(V=0) < 0 (opposite to 1D); pass
    # ``assume_jsc_positive=False`` so ``compute_metrics`` flips the sign
    # internally and reports JVMetrics in the J_sc-positive convention.
    # ``voc_bracketed=False`` when the sweep stopped short of V_oc; the
    # frontend surfaces this as an "increase V_max" warning. Raw V/J
    # arrays are NOT modified.
    metrics = compute_metrics(voltages, J_arr, assume_jsc_positive=False)
    return JV2DResult(
        V=voltages,
        J=J_arr,
        snapshots=tuple(snap_list),
        grid_x=grid.x.copy(),
        grid_y=grid.y.copy(),
        lateral_bc=resolved_protocol.lateral_bc,
        metrics=metrics,
        protocol=resolved_protocol,
        current_components=tuple(current_reports),
        ion_diagnostics=tuple(ion_reports),
        interface_srh_diagnostics=tuple(interface_reports),
    )
