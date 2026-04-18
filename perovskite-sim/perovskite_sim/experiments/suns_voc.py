"""Suns–V_oc characterisation (pseudo J-V free of series-resistance losses).

Physics
-------
Under open-circuit, all photogenerated carriers recombine locally — no
terminal current flows, so R_s drops out of the terminal voltage. Scan
the illumination intensity X (suns) and record V_oc(X); since the total
recombination current at V_oc equals the generation current X·J_sc,
plotting (V_oc(X), X·J_sc_1sun) traces the intrinsic diode J-V without
the I·R_s drop a standard J-V sweep incurs. The area under this
"pseudo-JV" curve yields the pseudo-FF — an upper bound on the real FF
whose gap to the measured FF quantifies R_s loss.

Implementation
--------------
For each suns level X we:

1. Build a material cache with ``G_optical`` scaled by X (works for both
   TMM-cached profiles and Beer-Lambert). Scaling the generation profile
   is equivalent to scaling the spectrum uniformly — the appropriate
   "suns" operation for this experiment.
2. Settle the device to illuminated steady-state at V_app = 0 with the
   scaled material cache.
3. Bisect to V_oc via the same ``_find_voc`` helper TPV uses (20-pt
   coarse scan to ~1 mV), warm-started from the previous suns level's
   V_oc so the bracket tightens monotonically as we walk up the curve.
4. Record J_sc(X) as J(V_app=0) under the scaled generation so the
   pseudo-JV uses the *actual* (not linearly extrapolated) short-circuit
   current at each intensity.

Pseudo-JV convention
--------------------
At V_oc(X) the net recombination current equals the generation current
at intensity X. Referenced to 1-sun, the hypothetical device current at
V = V_oc(X) is

    J_pseudo(V_oc(X)) = J_sc_ref − J_sc(X)

where ``J_sc_ref = J_sc(1 sun)``. This places:
- X < 1: J_pseudo > 0, V < V_oc_ref (power quadrant)
- X = 1: J_pseudo = 0 at V = V_oc_ref (anchor at 1-sun V_oc)
- X > 1: J_pseudo < 0, V > V_oc_ref (injection quadrant, beyond 1-sun V_oc)

So the pseudo-JV traces the ideal (series-resistance-free) diode curve
that the device would follow at 1-sun illumination. Area in the power
quadrant gives the pseudo-FF — an upper bound on the measured FF.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from perovskite_sim.models.device import DeviceStack, electrical_layers
from perovskite_sim.discretization.grid import multilayer_grid, Layer
from perovskite_sim.physics.generation import beer_lambert_generation
from perovskite_sim.solver.mol import (
    MaterialArrays,
    build_material_arrays,
    run_transient,
)
from perovskite_sim.solver.newton import solve_equilibrium
from perovskite_sim.experiments.jv_sweep import (
    _compute_current,
    _integrate_step,
    _grid_node_count,
)
from perovskite_sim.experiments.tpv import _find_voc

ProgressCallback = Callable[[str, int, int, str], None]


DEFAULT_SUNS = (0.01, 0.1, 1.0, 5.0, 10.0)


@dataclass(frozen=True)
class SunsVocResult:
    """Suns–V_oc sweep with derived pseudo J-V curve.

    Attributes
    ----------
    suns : np.ndarray
        Illumination intensity levels, in units of the stack's baseline
        photon flux (1.0 = AM1.5G). Sorted ascending.
    V_oc : np.ndarray
        Open-circuit voltage [V] at each suns level, measured by
        bisection under the scaled generation profile.
    J_sc : np.ndarray
        Short-circuit current density [A/m²] at each suns level. Linear
        scaling J_sc(X) ≈ X·J_sc(1) is expected; deviations reveal
        non-linear recombination or transport.
    J_pseudo_V, J_pseudo_J : np.ndarray
        The pseudo J-V curve as (V, J) pairs: ``V = V_oc(X)``,
        ``J = J_sc_ref − J_sc(X)`` with ``J_sc_ref`` taken at the suns
        level nearest X = 1. Sorted by V (ascending). Points with
        X < 1 sit in the power quadrant (V < V_oc_ref, J > 0); the
        X = 1 point anchors at (V_oc_ref, 0); points with X > 1 lie
        above V_oc_ref with J < 0 (injection direction).
    pseudo_FF : float
        Pseudo fill factor = max(V·J) on the pseudo-JV power quadrant
        (clipped to V ∈ [0, V_oc_ref], J ∈ [0, J_sc_ref], with (0,
        J_sc_ref) and (V_oc_ref, 0) anchors added for interpolation),
        normalised by V_oc_ref · J_sc_ref. NaN if the reference point
        can't be established.
    """
    suns: np.ndarray
    V_oc: np.ndarray
    J_sc: np.ndarray
    J_pseudo_V: np.ndarray
    J_pseudo_J: np.ndarray
    pseudo_FF: float


def _materialise_G_optical(
    x: np.ndarray, stack: DeviceStack, mat: MaterialArrays
) -> MaterialArrays:
    """Ensure ``mat.G_optical`` is explicit (not None).

    Beer-Lambert stacks have G_optical=None in the cache because the RHS
    computes it on the fly from stack.Phi. To scale by suns without
    mutating the DeviceStack, we materialise the baseline profile here
    so subsequent calls can use ``dataclasses.replace(mat, G_optical=...)``.
    """
    if mat.G_optical is not None:
        return mat
    G_baseline = beer_lambert_generation(x, mat.alpha, stack.Phi)
    return dataclasses.replace(mat, G_optical=G_baseline)


def _solve_illuminated_ss_with_mat(
    x: np.ndarray,
    stack: DeviceStack,
    mat: MaterialArrays,
    V_app: float = 0.0,
    t_settle: float = 1e-3,
    rtol: float = 1e-4,
    atol: float = 1e-6,
) -> np.ndarray:
    """Illuminated steady state at V_app using a caller-supplied mat cache.

    Mirrors ``solver.illuminated_ss.solve_illuminated_ss`` but threads a
    pre-built MaterialArrays through so we can modulate G_optical per
    suns level without rebuilding the whole cache each time.
    """
    y_dark = solve_equilibrium(x, stack)
    sol = run_transient(
        x, y_dark, (0.0, t_settle), np.array([t_settle]),
        stack, illuminated=True, V_app=V_app,
        rtol=rtol, atol=atol, mat=mat,
    )
    if not sol.success:
        return y_dark
    return sol.y[:, -1]


def _compute_pseudo_ff(
    V: np.ndarray, J: np.ndarray, V_oc_ref: float, J_sc_ref: float
) -> float:
    """Pseudo-FF on the power-quadrant of the pseudo-JV curve.

    The pseudo-JV has points scattered across all four quadrants (see
    convention in the module docstring). For the FF we restrict to the
    power quadrant V ∈ [0, V_oc_ref], J ∈ [0, J_sc_ref] and add the two
    axis anchors (short-circuit (0, J_sc_ref) and open-circuit
    (V_oc_ref, 0)) so linear interpolation has a sensible envelope even
    with only 2-3 measured points.

    Returns NaN if the reference product is non-positive.
    """
    if V_oc_ref <= 0.0 or J_sc_ref <= 0.0:
        return float("nan")

    # Keep only points strictly inside the power quadrant — boundaries
    # are re-added below as anchors to guarantee the interpolation span
    # and avoid duplicates when a measured point lies exactly on an
    # axis (e.g. (V_oc_ref, 0) at X=1 sun).
    mask = (V > 0.0) & (V < V_oc_ref) & (J > 0.0) & (J < J_sc_ref)
    V_q = np.concatenate(([0.0], V[mask], [V_oc_ref]))
    J_q = np.concatenate(([J_sc_ref], J[mask], [0.0]))

    order = np.argsort(V_q)
    V_s = V_q[order]
    J_s = J_q[order]

    V_grid = np.linspace(0.0, V_oc_ref, 500)
    J_grid = np.interp(V_grid, V_s, J_s)
    P_max = float(np.max(V_grid * J_grid))
    return P_max / (V_oc_ref * J_sc_ref)


def run_suns_voc(
    stack: DeviceStack,
    suns_levels: Sequence[float] = DEFAULT_SUNS,
    N_grid: int = 60,
    t_settle: float = 1e-3,
    rtol: float = 1e-4,
    atol: float = 1e-6,
    progress: ProgressCallback | None = None,
) -> SunsVocResult:
    """Run a Suns–V_oc sweep and build the pseudo J-V curve.

    Parameters
    ----------
    stack : DeviceStack
    suns_levels : sequence of float, default (0.01, 0.1, 1, 5, 10)
        Illumination intensities in units of the stack's baseline photon
        flux. Sorted ascending internally; duplicates removed.
    N_grid : int, default 60
        Total drift-diffusion grid nodes across the electrical layers.
    t_settle : float, default 1e-3
        Settling time [s] for each suns level's illuminated steady-state
        solve. 1 ms is well beyond carrier recombination lifetimes but
        short compared to ion redistribution, which is appropriate: the
        ion profile at V=0 is what we want held fixed while carriers
        track the new illumination.
    rtol, atol : float
        Scipy solver tolerances forwarded to ``run_transient`` and
        ``_find_voc``.
    progress : ProgressCallback | None
        Called as ``progress("suns_voc", k, n_total, msg)`` after each
        suns level completes.

    Returns
    -------
    SunsVocResult
    """
    suns_sorted = np.array(sorted(set(float(s) for s in suns_levels)), dtype=float)
    if suns_sorted.size == 0:
        raise ValueError("suns_levels must be non-empty")
    if np.any(suns_sorted <= 0.0):
        raise ValueError(f"suns_levels must be positive, got {suns_sorted}")

    # Build the electrical grid once — same shape for every suns level.
    elec = electrical_layers(stack)
    n_per = max(N_grid // len(elec), 2)
    layers_grid = [Layer(l.thickness, n_per) for l in elec]
    x = multilayer_grid(layers_grid)

    # Baseline material cache and materialised G_optical so we can scale.
    mat_baseline = build_material_arrays(x, stack)
    mat_baseline = _materialise_G_optical(x, stack, mat_baseline)
    G_unit = mat_baseline.G_optical.copy()  # 1-sun generation profile

    V_oc_arr = np.zeros_like(suns_sorted)
    J_sc_arr = np.zeros_like(suns_sorted)
    V_guess = stack.compute_V_bi()  # initial bracket upper bound

    for k, suns in enumerate(suns_sorted):
        mat_k = dataclasses.replace(mat_baseline, G_optical=G_unit * suns)

        # Illuminated steady-state at V=0 → gives J_sc(suns) directly.
        y_ss = _solve_illuminated_ss_with_mat(
            x, stack, mat_k, V_app=0.0,
            t_settle=t_settle, rtol=rtol, atol=atol,
        )
        J_sc_arr[k] = _compute_current(x, y_ss, stack, 0.0, mat=mat_k)

        # V_oc: bisect starting from the previous level's V_oc (warm
        # start). V_guess is the upper end of the coarse bracket in
        # _find_voc (scan to V_guess*1.5), so we inflate slightly for
        # the first (lowest-suns) step where V_oc can undershoot V_bi.
        V_oc_k, _y_at_voc = _find_voc(
            x, y_ss, stack, mat_k, V_guess=max(V_guess, 0.3),
            rtol=rtol, atol=atol,
        )
        V_oc_arr[k] = V_oc_k
        # Warm-start: next suns level's V_oc is monotonically higher
        # (more light → higher V_oc), so carry the larger guess forward.
        V_guess = max(V_guess, V_oc_k)

        if progress is not None:
            progress(
                "suns_voc", k + 1, len(suns_sorted),
                f"X={suns:.3g}, V_oc={V_oc_k:.4f} V, J_sc={J_sc_arr[k]:.2f} A/m²",
            )

    # Reference 1-sun if present; otherwise the point nearest X=1.
    idx_ref = int(np.argmin(np.abs(suns_sorted - 1.0)))
    V_oc_ref = float(V_oc_arr[idx_ref])
    J_sc_ref = float(J_sc_arr[idx_ref])

    # Pseudo-JV: standard Sinton convention, referenced to J_sc at the
    # 1-sun anchor. J_pseudo(V_oc(X)) = J_sc_ref - J_sc(X).  At X=1 this
    # gives J_pseudo = 0 — the (V_oc_ref, 0) anchor point on the diode
    # curve. V_pseudo sorted ascending by construction (V_oc(X) monotone
    # in X).
    V_pseudo = V_oc_arr.copy()
    J_pseudo = J_sc_ref - J_sc_arr
    pff = _compute_pseudo_ff(V_pseudo, J_pseudo, V_oc_ref, J_sc_ref)

    return SunsVocResult(
        suns=suns_sorted,
        V_oc=V_oc_arr,
        J_sc=J_sc_arr,
        J_pseudo_V=V_pseudo,
        J_pseudo_J=J_pseudo,
        pseudo_FF=pff,
    )
