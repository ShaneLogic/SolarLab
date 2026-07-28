"""Photon conservation on the transfer-matrix optical path.

Companion to the Beer-Lambert conservation contract documented in
``perovskite_sim/physics/generation.py``.  The solver integrates generation
as the node-centred rectangle rule ``sum_i G[i] * dx_cell[i]``; a G built by
point-sampling the volumetric absorption rate at the nodes is not
photon-conserving on a mesh that is coarse relative to the absorption
length, and over-counts the sharply-peaked front-of-absorber generation.

Measured on ``configs/ionmonger_benchmark_tmm.yaml`` before the fix (exact
absorbed-photon budget 225.00 A/m^2 equivalent, hard optical ceiling
q*Phi*(1-R-T) = 226.67):

    N_grid    10      15      30      60     100     200
    q*sum     353.35  262.85  238.97  228.18  226.25  225.37

i.e. converging to its own budget from ABOVE and, at N_grid = 10, claiming
57 % more photocurrent than the stack can physically absorb.  These tests
pin that the budget is now mesh-INDEPENDENT and bounded, so they fail if
anyone reverts to a point-sampled quadrature.

Cheap by construction — pure optics, no drift-diffusion solve.
"""
from __future__ import annotations

import numpy as np
import pytest

from perovskite_sim.constants import Q
from perovskite_sim._compat.numpy_compat import trapezoid
from perovskite_sim.data import load_am15g
from perovskite_sim.discretization.grid import Layer, multilayer_grid
from perovskite_sim.experiments.jv_sweep import _layer_node_counts
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.models.device import electrical_layers
from perovskite_sim.physics.generation import dual_cell_faces, dual_cell_widths
from perovskite_sim.physics.optics import (
    TMMLayer,
    _incoherent_front_factors,
    _transfer_matrix_stack,
    tmm_absorption_profile,
    tmm_cumulative_absorptance,
    tmm_reflectance,
)
from perovskite_sim.solver.mol import _compute_tmm_generation

TMM_PRESETS = [
    "configs/ionmonger_benchmark_tmm.yaml",
    "configs/nip_MAPbI3_tmm.yaml",
    "configs/pin_MAPbI3_tmm.yaml",
]

# The wavelength window ``_compute_tmm_generation`` models.
LAM_MIN_NM, LAM_MAX_NM, N_WL = 300.0, 1000.0, 200


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _grid(stack, N_grid: int) -> np.ndarray:
    """The electrical grid ``run_jv_sweep`` would build at this N_grid."""
    elec = electrical_layers(stack)
    return multilayer_grid([
        Layer(lyr.thickness, n)
        for lyr, n in zip(elec, _layer_node_counts(stack, N_grid))
    ])


def _tmm_stack(stack):
    """Rebuild the TMM layer stack exactly as ``_compute_tmm_generation`` does."""
    from perovskite_sim.data import load_nk

    wl_nm = np.linspace(LAM_MIN_NM, LAM_MAX_NM, N_WL)
    wl_m = wl_nm * 1e-9
    _, flux = load_am15g(wl_nm)
    layers = []
    for lyr in stack.layers:
        p = lyr.params
        if p.optical_material is not None:
            _, n_arr, k_arr = load_nk(p.optical_material, wl_nm)
        elif p.n_optical is not None:
            n_arr = np.full(N_WL, p.n_optical)
            k_arr = p.alpha * wl_m / (4.0 * np.pi)
        else:
            n_arr = np.full(N_WL, np.sqrt(p.eps_r))
            k_arr = p.alpha * wl_m / (4.0 * np.pi)
        layers.append(TMMLayer(d=lyr.thickness, n=n_arr, k=k_arr,
                               incoherent=bool(p.incoherent)))
    boundaries = np.zeros(len(stack.layers) + 1)
    for i, lyr in enumerate(stack.layers):
        boundaries[i + 1] = boundaries[i] + lyr.thickness
    offset = sum(l.thickness for l in stack.layers if l.role == "substrate")
    return layers, boundaries, wl_m, flux, offset


def _optical_ceiling(stack) -> tuple[float, float]:
    """(incident, incident*(1-R-T)) as A/m^2-equivalent photon flux.

    ``1 - R - T`` is the hard ceiling on what the WHOLE stack can absorb,
    and the electrical layers are a subset of it, so the generation budget
    must sit strictly below.  ``T`` carries the ``n_sub / n_amb`` Poynting
    factor: with an incoherent front slab the sub-stack's amplitude
    transmission is referenced to the slab index, not to air.
    """
    layers, boundaries, wl_m, flux, _ = _tmm_stack(stack)
    R = tmm_reflectance(layers, wl_m)
    if layers[0].incoherent:
        R_front, T_bulk, n_real = _incoherent_front_factors(layers[0], wl_m, 1.0)
        S_sub, _ = _transfer_matrix_stack(
            layers[1:], wl_m, n_ambient=n_real, n_substrate=1.0,
        )
        T = (1.0 - R_front) * T_bulk * np.abs(1.0 / S_sub[:, 0, 0]) ** 2 / n_real
    else:
        S_tot, _ = _transfer_matrix_stack(layers, wl_m)
        T = np.abs(1.0 / S_tot[:, 0, 0]) ** 2
    incident = Q * float(trapezoid(flux, wl_m))
    absorbable = Q * float(trapezoid(flux * (1.0 - R - T), wl_m))
    return incident, absorbable


def _budget(stack, N_grid: int) -> float:
    """q * sum(G * dx_cell) — exactly what the RHS integrates."""
    x = _grid(stack, N_grid)
    G = _compute_tmm_generation(x, stack)
    assert G is not None, "preset is not TMM-enabled"
    return Q * float(np.sum(G * dual_cell_widths(x)))


# ---------------------------------------------------------------------------
# the conservation contract
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cfg", TMM_PRESETS)
def test_tmm_photon_budget_is_mesh_independent(cfg):
    """q*sum(G*dx_cell) must not depend on the mesh — coarse meshes included.

    The per-cell absorbed-photon counts telescope, so in exact arithmetic
    the budget equals ``C(x[-1]) - C(x[0])`` on ANY mesh.  The only error is
    float64 cancellation while summing at most ~200 differences, bounded by
    ``N * eps ~ 200 * 2.2e-16 ~ 4e-14`` relative; the 1e-12 tolerance is
    that round-off bound with ~25x headroom.  It is NOT reverse-engineered
    from an observed spread — the pre-fix spread was 57 %, fourteen orders
    of magnitude larger.

    N_grid = 10 / 15 / 30 are deliberately coarser than anything the shipped
    experiments use: asymptotic convergence is not the property under test.
    """
    stack = load_device_from_yaml(cfg)
    budgets = {n: _budget(stack, n) for n in (10, 15, 30, 60, 100, 200)}
    ref = budgets[100]
    assert ref > 0.0
    worst_n, worst = max(
        budgets.items(), key=lambda kv: abs(kv[1] - ref) / ref
    )
    rel = abs(worst - ref) / ref
    assert rel < 1e-12, (
        f"{cfg}: TMM photon budget is mesh-dependent — N_grid={worst_n} gives "
        f"{worst:.6f} A/m^2 vs {ref:.6f} at N_grid=100 (rel {rel:.3e}). "
        f"All budgets: {  {k: round(v, 6) for k, v in budgets.items()} }"
    )


@pytest.mark.parametrize("cfg", TMM_PRESETS)
def test_tmm_photon_budget_within_optical_ceiling(cfg):
    """Generation can never exceed what the stack absorbs.

    Two nested physical bounds, neither of them a fitted tolerance:
      1. the incident photon flux over the modelled 300-1000 nm window;
      2. the tighter ``Phi * (1 - R - T)`` — the photons that are neither
         reflected nor transmitted, i.e. the entire stack's absorption,
         of which the electrical layers get a subset.
    Only float64 round-off slack (1e-12 relative) is allowed on top.

    The pre-fix code violated BOTH at N_grid = 10 (353.35 vs a 226.67
    ceiling and a 381.88 incident flux).
    """
    stack = load_device_from_yaml(cfg)
    incident, absorbable = _optical_ceiling(stack)
    for n in (10, 15, 30, 60, 100):
        got = _budget(stack, n)
        assert got <= incident * (1.0 + 1e-12), (
            f"{cfg} N_grid={n}: generation budget {got:.4f} A/m^2 exceeds the "
            f"incident photon flux {incident:.4f} A/m^2"
        )
        assert got <= absorbable * (1.0 + 1e-12), (
            f"{cfg} N_grid={n}: generation budget {got:.4f} A/m^2 exceeds the "
            f"optical ceiling q*Phi*(1-R-T) = {absorbable:.4f} A/m^2"
        )


@pytest.mark.parametrize("cfg", TMM_PRESETS)
def test_tmm_photon_budget_equals_electrical_stack_absorption(cfg):
    """The budget must equal the photons the ELECTRICAL layers absorb.

    Reference = the analytic Poynting-flux drop between the first and last
    electrical node, which is the physically correct target: those are the
    only photons that can make a collectable pair.  Conservation without
    this would be satisfiable by simply generating too little.
    """
    stack = load_device_from_yaml(cfg)
    layers, boundaries, wl_m, flux, offset = _tmm_stack(stack)
    x = _grid(stack, 100)
    faces = np.array([x[0], x[-1]]) + offset
    C = tmm_cumulative_absorptance(layers, wl_m, faces, boundaries)
    ref = Q * float(trapezoid((C[1] - C[0]) * flux, wl_m))
    got = _budget(stack, 100)
    assert abs(got - ref) <= 1e-12 * ref, (
        f"{cfg}: budget {got:.9f} != electrical-stack absorption {ref:.9f}"
    )


# ---------------------------------------------------------------------------
# the optics primitive the conservation rests on
# ---------------------------------------------------------------------------

def test_cumulative_absorptance_is_the_exact_antiderivative():
    """``C`` must be the antiderivative of ``tmm_absorption_profile``.

    Compared strictly INSIDE one absorber (an inset of 1e-6 of the
    thickness) so the numerical reference never samples the discontinuity
    of ``A`` at a layer interface.

    Tolerance provenance — measured second-order convergence of the
    reference toward the analytic value on this exact stack:
        n_pts     2001     8001    32001   128001
        max err  9.1e-7   5.7e-8   3.5e-9   2.2e-10
    a clean factor-16 per 4x refinement, i.e. the residual is the
    trapezoid's own O(h^2) error and the analytic form is the limit.  The
    bounds below are the values that trend predicts at 4001 / 16001 points
    (~2.3e-7 / ~1.4e-8) rounded up ~4x; the ratio assertion is what
    actually certifies exactness, since a wrong closed form would leave a
    constant offset that does NOT shrink with refinement.
    """
    stack = load_device_from_yaml("configs/ionmonger_benchmark_tmm.yaml")
    layers, boundaries, wl_m, _flux, offset = _tmm_stack(stack)
    elec = electrical_layers(stack)
    absorber = next(l for l in elec if l.role == "absorber")
    front = offset + sum(
        l.thickness for l in elec[: elec.index(absorber)]
    )
    inset = absorber.thickness * 1e-6
    xa, xb = front + inset, front + absorber.thickness - inset

    C = tmm_cumulative_absorptance(layers, wl_m, np.array([xa, xb]), boundaries)
    analytic = C[1] - C[0]

    errs = {}
    for npts in (4001, 16001):
        xs = np.linspace(xa, xb, npts)
        num = np.trapezoid(
            tmm_absorption_profile(layers, wl_m, xs, boundaries), xs, axis=0,
        )
        errs[npts] = float(np.max(np.abs(analytic - num)))

    assert errs[4001] < 1e-6, f"analytic vs 4001-pt trapezoid: {errs[4001]:.2e}"
    assert errs[16001] < 1e-7, f"analytic vs 16001-pt trapezoid: {errs[16001]:.2e}"
    assert errs[16001] < errs[4001] / 3.0, (
        "reference does not converge toward the analytic value — the closed "
        f"form is wrong, not the quadrature ({errs[4001]:.2e} -> "
        f"{errs[16001]:.2e}, expected ~16x)"
    )


def test_cumulative_absorptance_closes_energy_balance_exactly():
    """R + T + A = 1 to machine precision on a coherent stack.

    Same three-layer stack as ``tests/unit/physics/test_optics.py::
    TestTMMReflectance::test_rta_equals_one``, which can only assert
    ``atol=0.03`` because it integrates ``A`` with a discrete trapezoid.
    With the analytic antiderivative there is no spatial quadrature left,
    so the residual is pure float64 round-off — measured 1.0e-15, pinned
    here at 1e-12 (three orders of headroom).
    """
    n_wl = 60
    wl_nm = np.linspace(350.0, 780.0, n_wl)
    wl_m = wl_nm * 1e-9
    layers = [
        TMMLayer(d=50e-9, n=np.full(n_wl, 2.4),
                 k=np.where(wl_nm < 380, 0.05, 0.001)),
        TMMLayer(d=400e-9, n=np.full(n_wl, 2.5),
                 k=0.4 * np.exp(-(wl_nm - 400) / 150)),
        TMMLayer(d=150e-9, n=np.full(n_wl, 1.8),
                 k=np.where(wl_nm < 420, 0.02, 0.001)),
    ]
    boundaries = np.array([0.0, 50e-9, 450e-9, 600e-9])

    C = tmm_cumulative_absorptance(layers, wl_m, boundaries, boundaries)
    A = C[-1] - C[0]
    R = tmm_reflectance(layers, wl_m)
    S_total, _ = _transfer_matrix_stack(layers, wl_m)
    T = np.abs(1.0 / S_total[:, 0, 0]) ** 2

    np.testing.assert_allclose(R + T + A, 1.0, atol=1e-12)
    assert np.all(np.diff(C, axis=0) >= 0.0), "C must be non-decreasing in x"


def test_cumulative_absorptance_is_exactly_zero_for_a_transparent_stack():
    """k = 0 everywhere must give an EXACT floating-point zero.

    The ``coeff / alpha == n / n_ambient`` collapse removes the 0/0 the
    naive term-by-term integral would hit, and the ``a * u < 300`` branch
    keeps ``exp(0) == 1.0`` bit-exact so the decay terms cancel identically.
    A near-zero would silently leak generation into non-absorbing transport
    layers on every mesh.
    """
    n_wl = 40
    wl_m = np.linspace(400e-9, 700e-9, n_wl)
    layers = [
        TMMLayer(d=100e-9, n=np.full(n_wl, 2.0), k=np.zeros(n_wl)),
        TMMLayer(d=300e-9, n=np.full(n_wl, 2.5), k=np.zeros(n_wl)),
    ]
    boundaries = np.array([0.0, 100e-9, 400e-9])
    C = tmm_cumulative_absorptance(
        layers, wl_m, np.linspace(0.0, 400e-9, 37), boundaries,
    )
    assert np.all(C == 0.0), f"transparent stack absorbed {np.abs(C).max():.3e}"


def test_cumulative_absorptance_raises_instead_of_emitting_inf():
    """A layer past the float64 window must fail loudly, not return +inf.

    The backward-wave power grows as ``e^{+alpha*u}`` while its front-face
    amplitude shrinks as ``e^{-alpha*d/2}``; past a measured ``alpha*d`` of
    ~800 neither factor is representable and there is no honest value to
    return (bisected on a single n=3 / 20 um slab at 500 nm: ``alpha*d`` up
    to 750 returns finite, 800 and above raises).  The slab below spans
    ``alpha*d`` 718 (700 nm) to 1257 (400 nm), so the short-wavelength end
    is what trips the guard.  An un-guarded +inf would land straight in
    ``MaterialArrays.G_optical``, where it is built once and never
    re-checked by the RHS finite guard.

    (The SEPARATE, pre-existing accuracy limit at ``alpha*d ~ 70`` — where
    the front-referenced backward amplitude is round-off noise — is shared
    with ``tmm_absorption_profile``, which goes unphysical at the same
    ``alpha*d``; it is documented in ``optics.py``, not tested here.)
    """
    n_wl = 20
    wl_m = np.linspace(400e-9, 700e-9, n_wl)
    layers = [TMMLayer(d=20e-6, n=np.full(n_wl, 3.0), k=np.full(n_wl, 2.0))]
    with pytest.raises(ValueError, match="not representable in float64"):
        tmm_cumulative_absorptance(
            layers, wl_m, np.linspace(0.0, 20e-6, 51), np.array([0.0, 20e-6]),
        )


def test_dual_cell_faces_bound_the_dual_cells():
    """Faces and widths must describe the same dual mesh.

    Interior widths are exactly the face spacing; the two boundary cells
    are the documented deliberate mismatch — geometric extent ``dx/2``,
    RHS weight the full ``dx`` — which is precisely why the generation
    quadrature divides by ``dx_cell`` and not by the face spacing.
    """
    x = np.array([0.0, 1.0, 3.0, 6.0, 10.0])
    faces = dual_cell_faces(x)
    widths = dual_cell_widths(x)
    assert faces[0] == x[0] and faces[-1] == x[-1]
    assert faces.shape[0] == x.shape[0] + 1
    np.testing.assert_allclose(np.diff(faces)[1:-1], widths[1:-1], rtol=0, atol=0)
    assert np.diff(faces)[0] == pytest.approx(widths[0] / 2.0)
    assert np.diff(faces)[-1] == pytest.approx(widths[-1] / 2.0)
    with pytest.raises(ValueError):
        dual_cell_faces(np.array([1.0]))
