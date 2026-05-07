from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import dataclasses
import numpy as np

from perovskite_sim.discretization.grid import Layer
from perovskite_sim.models.device import DeviceStack, electrical_layers
from perovskite_sim.twod.grid_2d import build_grid_2d
from perovskite_sim.twod.microstructure import Microstructure
from perovskite_sim.twod.radiative_reabsorption_2d import recompute_g_with_rad_2d
from perovskite_sim.twod.solver_2d import (
    build_material_arrays_2d,
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
    """
    V: np.ndarray
    J: np.ndarray
    snapshots: tuple[SpatialSnapshot2D, ...]
    grid_x: np.ndarray
    grid_y: np.ndarray
    lateral_bc: str


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
    p0 = y_state[N:].reshape((g.Ny, g.Nx))
    G_with_rad = recompute_g_with_rad_2d(
        G_optical=mat.G_optical, n=n0, p=p0, B_rad=mat.B_rad,
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


def run_jv_sweep_2d(
    stack: DeviceStack,
    microstructure: Microstructure | None = None,
    *,
    lateral_length: float,
    Nx: int,
    V_max: float,
    V_step: float,
    illuminated: bool = True,
    lateral_bc: str = "periodic",
    Ny_per_layer: int = 20,
    settle_t: float = 1e-7,
    progress: ProgressCallback | None = None,
    save_snapshots: bool = True,
) -> JV2DResult:
    """Forward illuminated J-V sweep on a 2D grid.

    Mirrors the 1D run_jv_sweep warm-start semantics: each voltage step
    starts from the previous step's settled state so ionic memory (Stage B)
    or carrier redistribution effects are preserved across the sweep.

    The voltage grid walks from 0 → V_max in steps of V_step using
    ``np.arange(0.0, V_max + V_step/2, V_step)``, matching the 1D convention.

    Stage A notes
    -------------
    - Ions are absent from the 2D state vector (Stage A has no D_ion in 2D),
      so there is no ionic memory across voltage steps.  The warm-start still
      matters for carrier relaxation within each settle interval.
    - Optical generation comes from ``mat.G_optical`` which is populated at
      build time when the stack uses TMM (``nip_MAPbI3_tmm.yaml`` etc.).
      Beer-Lambert stacks produce zero G_optical in Stage A because the
      Beer-Lambert runtime path is not reimplemented here.
    - Recombination uses only SRH (mid-gap, n1=p1=0); radiative and Auger
      channels are Stage B additions (see solver_2d.assemble_rhs_2d).

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

    Returns
    -------
    JV2DResult
        Sweep results including V array, J array, optional snapshots, and grid
        coordinates.
    """
    # Resolve microstructure: explicit arg wins, else pick up from
    # stack.microstructure (set by load_device_from_yaml when the YAML has a
    # microstructure block), else fall back to an empty descriptor (Stage A
    # lateral-uniform case).
    if microstructure is None:
        microstructure = getattr(stack, "microstructure", None) or Microstructure()

    elec = electrical_layers(stack)
    layers = [Layer(L.thickness, Ny_per_layer) for L in elec]
    grid = build_grid_2d(
        layers,
        lateral_length=lateral_length,
        Nx=Nx,
        lateral_uniform=True,
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

    if illuminated:
        y_1d = solve_illuminated_ss(grid.y, stack, V_app=0.0, t_settle=1e-3)
    else:
        y_1d = solve_equilibrium(grid.y, stack)

    sv1d = StateVec.unpack(y_1d, len(grid.y))
    n_1d, p_1d, P_1d = sv1d.n, sv1d.p, sv1d.P

    mat = build_material_arrays_2d(
        grid, stack, microstructure, lateral_bc=lateral_bc,
        P_ion_static_1d=P_1d,
    )

    if not illuminated:
        mat = dataclasses.replace(mat, G_optical=np.zeros_like(mat.G_optical))

    Ny, Nx_nodes = grid.Ny, grid.Nx
    n_2d_init = np.broadcast_to(n_1d[:, None], (Ny, Nx_nodes)).copy()
    p_2d_init = np.broadcast_to(p_1d[:, None], (Ny, Nx_nodes)).copy()
    y_state = np.concatenate([n_2d_init.flatten(), p_2d_init.flatten()])

    # --- Voltage sweep -------------------------------------------------------
    voltages = np.arange(0.0, V_max + V_step / 2.0, V_step)
    J_list: list[float] = []
    snap_list: list[SpatialSnapshot2D] = []

    for k, V in enumerate(voltages):
        try:
            y_state = run_transient_2d(
                y_state, mat,
                V_app=float(V),
                t_end=settle_t,
                max_step=settle_t / 50.0,
                max_nfev=_TWOD_RADAU_MAX_NFEV,
            )
        except RuntimeError:
            # Stage B(c.3) lagged fallback: bake R_tot once at the entry state,
            # retry with the disabled flag. Mirrors 1D jv_sweep.py:328+.
            if not (mat.has_radiative_reabsorption_2d and illuminated):
                raise                                      # nothing to fall back on
            mat_step = _bake_radiative_reabsorption_step_2d(
                y_state, mat, illuminated=illuminated,
            )
            y_state = run_transient_2d(
                y_state, mat_step,
                V_app=float(V),
                t_end=settle_t,
                max_step=settle_t / 50.0,
                max_nfev=_TWOD_RADAU_MAX_NFEV,
            )
        snap = extract_snapshot_2d(y_state, mat, V_app=float(V))
        J_list.append(compute_terminal_current_2d(snap))
        if save_snapshots:
            snap_list.append(snap)
        if progress is not None:
            progress("jv_2d", k + 1, len(voltages), f"V = {V:.3f} V")

    return JV2DResult(
        V=voltages,
        J=np.array(J_list, dtype=float),
        snapshots=tuple(snap_list),
        grid_x=grid.x.copy(),
        grid_y=grid.y.copy(),
        lateral_bc=lateral_bc,
    )
