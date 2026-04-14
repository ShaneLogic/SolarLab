"""Series current-matching J-V driver for 2T monolithic tandem cells.

Combines:
    1. A single combined-TMM optical solve over the full tandem stack
       (top_cell + junction + bottom_cell) to partition generation G(x).
    2. Independent sub-cell drift-diffusion J-V sweeps with the pre-computed
       G(x) injected via ``fixed_generation``.
    3. Series voltage addition at a common current grid (current-matching).

Public API
----------
series_match_jv   — pure function: combines two sub-cell J-V curves.
run_tandem_jv     — end-to-end driver returning TandemJVResult.
TandemJVResult    — frozen dataclass holding the tandem J-V and sub-results.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from perovskite_sim.discretization.grid import Layer, multilayer_grid
from perovskite_sim.experiments.jv_sweep import (
    JVMetrics,
    JVResult,
    compute_metrics,
    run_jv_sweep,
)
from perovskite_sim.models.device import DeviceStack, electrical_layers
from perovskite_sim.models.tandem_config import TandemConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _electrical_grid_length(stack: DeviceStack, N_grid: int) -> int:
    """Compute the number of electrical grid nodes run_jv_sweep will use.

    Replicates the grid construction at the top of run_jv_sweep so that callers
    can pre-compute the exact N before calling compute_tandem_generation, which
    requires N_top / N_bot to match the electrical grid lengths exactly.

    The formula follows from multilayer_grid's deduplication logic:
    - Each layer is given  n_per = N_grid // len(elec)  intervals.
    - tanh_grid(n_per, L) returns n_per + 1 points.
    - multilayer_grid drops the leading point of every layer except the first,
      so total points = 1 + len(elec) * n_per.

    Args:
        stack:  DeviceStack whose electrical layers will be gridded.
        N_grid: N_grid kwarg that will be passed to run_jv_sweep.

    Returns:
        Integer count of electrical grid nodes N = len(x).
    """
    elec = electrical_layers(stack)
    n_elec = len(elec)
    n_per = N_grid // n_elec          # intervals per layer (integer division)
    return 1 + n_elec * n_per         # matches len(multilayer_grid(...))


# ---------------------------------------------------------------------------
# Core series-matching logic
# ---------------------------------------------------------------------------

def series_match_jv(
    top_J: np.ndarray,
    top_V: np.ndarray,
    bot_J: np.ndarray,
    bot_V: np.ndarray,
    V_junction: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Series-add two sub-cell J-V curves at a common current grid.

    In a 2T series-connected tandem the same current flows through both
    sub-cells. This function interpolates both V(J) curves onto a common,
    shared J axis that spans only the overlapping current range, then sums
    the voltages to obtain the tandem V(J).

    The returned J_common is sorted ascending (most negative first), and
    V_tandem is monotonically non-increasing (as expected for a solar-cell
    J-V).

    Args:
        top_J: current density [A/m²] for the top sub-cell, any order.
        top_V: voltage [V] for the top sub-cell, paired with top_J.
        bot_J: current density [A/m²] for the bottom sub-cell, any order.
        bot_V: voltage [V] for the bottom sub-cell, paired with bot_J.
        V_junction: recombination-junction voltage offset [V]; default 0.

    Returns:
        (J_common, V_top_m, V_bot_m, V_tandem) where each array has the
        same length determined by max(len(top_J), len(bot_J)).

    Raises:
        ValueError: if the J ranges of the two sub-cells do not overlap.
    """
    top_order = np.argsort(top_J)
    bot_order = np.argsort(bot_J)
    tJ, tV = top_J[top_order], top_V[top_order]
    bJ, bV = bot_J[bot_order], bot_V[bot_order]

    j_lo = max(tJ[0], bJ[0])
    j_hi = min(tJ[-1], bJ[-1])
    if j_lo >= j_hi:
        raise ValueError(
            f"Sub-cell J ranges do not overlap: "
            f"top=[{tJ[0]}, {tJ[-1]}], bottom=[{bJ[0]}, {bJ[-1]}]"
        )

    n = max(len(tJ), len(bJ))
    J_common = np.linspace(j_lo, j_hi, n)
    V_top_m = np.interp(J_common, tJ, tV)
    V_bot_m = np.interp(J_common, bJ, bV)
    V_tandem = V_top_m + V_bot_m + V_junction
    return J_common, V_top_m, V_bot_m, V_tandem


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TandemJVResult:
    """Immutable result bundle for a series-connected tandem J-V sweep.

    Attributes:
        V:          tandem voltage array [V], sorted ascending.
        J:          common current density array [A/m²], sorted ascending.
        V_top:      top sub-cell voltage at each J_common point [V].
        V_bot:      bottom sub-cell voltage at each J_common point [V].
        metrics:    tandem JV metrics (V_oc, J_sc, FF, PCE) from the
                    series-matched curve.
        top_result: full JVResult from the top sub-cell sweep.
        bot_result: full JVResult from the bottom sub-cell sweep.
    """
    V: np.ndarray
    J: np.ndarray
    V_top: np.ndarray
    V_bot: np.ndarray
    metrics: JVMetrics
    top_result: JVResult
    bot_result: JVResult


# ---------------------------------------------------------------------------
# End-to-end tandem driver
# ---------------------------------------------------------------------------

def run_tandem_jv(
    cfg: TandemConfig,
    wavelengths_m: np.ndarray,
    spectral_flux: np.ndarray,
    wavelengths_nm: np.ndarray,
    N_grid: int = 100,
    n_points: int = 50,
) -> TandemJVResult:
    """Run a series-connected 2T tandem J-V sweep.

    Performs three steps:
      1. Compute the actual electrical-grid length for each sub-cell
         (replicating run_jv_sweep's grid construction) so that the generation
         profiles passed via fixed_generation have the correct shape.
      2. Run one combined-TMM optical solve over the full tandem stack to
         produce per-sub-cell generation profiles G_top(x) and G_bot(x).
      3. Run independent sub-cell J-V sweeps with the pre-computed profiles
         injected, series-match the forward curves, and extract tandem metrics.

    Args:
        cfg:           TandemConfig specifying top/bottom sub-cells and junction.
        wavelengths_m: wavelength array in metres [m], shape (n_wl,).
        spectral_flux: spectral photon flux [m⁻² s⁻¹ m⁻¹], shape (n_wl,).
        wavelengths_nm: same wavelengths in nanometres — passed to load_nk.
        N_grid:        number of grid intervals to request per sub-cell sweep
                       (the actual node count depends on the layer count; see
                       _electrical_grid_length).
        n_points:      number of voltage points in each sweep direction.

    Returns:
        TandemJVResult with the series-matched J-V and per-sub-cell results.

    Raises:
        ValueError: propagated from run_jv_sweep if fixed_generation shape
                    mismatches the electrical grid (should not happen if
                    _electrical_grid_length is correct).
        ValueError: from series_match_jv if sub-cell J ranges don't overlap.
    """
    from perovskite_sim.physics.tandem_optics import compute_tandem_generation

    # Step 1: Determine electrical grid node counts for each sub-cell.
    # run_jv_sweep validates fixed_generation.shape == (N,) where N is the
    # electrical node count — not N_grid itself. We replicate that computation
    # here so the shapes agree before any expensive ODE work starts.
    N_top = _electrical_grid_length(cfg.top_cell, N_grid)
    N_bot = _electrical_grid_length(cfg.bottom_cell, N_grid)

    # Step 2: One combined-TMM generation solve over the full tandem stack.
    gen = compute_tandem_generation(
        cfg, wavelengths_m, spectral_flux, wavelengths_nm,
        N_top=N_top, N_bot=N_bot,
    )

    # Step 3: Independent sub-cell sweeps with pre-computed G(x).
    top_result = run_jv_sweep(
        cfg.top_cell,
        N_grid=N_grid,
        n_points=n_points,
        fixed_generation=gen.G_top,
    )
    bot_result = run_jv_sweep(
        cfg.bottom_cell,
        N_grid=N_grid,
        n_points=n_points,
        fixed_generation=gen.G_bot,
    )

    # Step 4: Series-match the forward sweeps.
    J_common, V_top_m, V_bot_m, V_tandem = series_match_jv(
        top_result.J_fwd, top_result.V_fwd,
        bot_result.J_fwd, bot_result.V_fwd,
        V_junction=0.0,
    )

    # Step 5: Extract tandem metrics from the series-matched curve.
    # compute_metrics is already public in jv_sweep and handles the same
    # sign convention (J > 0 at short circuit). V and J must be paired; the
    # natural pairing here is V_tandem sorted by J_common (already ascending).
    metrics = compute_metrics(V_tandem, J_common)

    return TandemJVResult(
        V=V_tandem,
        J=J_common,
        V_top=V_top_m,
        V_bot=V_bot_m,
        metrics=metrics,
        top_result=top_result,
        bot_result=bot_result,
    )
