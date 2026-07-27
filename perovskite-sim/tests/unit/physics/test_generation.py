"""Photon conservation of the Beer-Lambert generation quadrature.

What is asserted, and why it needs no physics tolerance
-------------------------------------------------------
``carrier_continuity_rhs`` adds ``G[i]`` to ``dn[i]`` alongside a flux
divergence scaled by ``1/(Q*dx_cell[i])``.  Multiply the balance by
``Q*dx_cell[i]``, sum over the grid, and the flux term telescopes to the two
terminal currents -- so the areal generation the device receives is exactly
``q * sum(G * dx_cell)``, a node-centred RECTANGLE rule.

``beer_lambert_generation`` therefore returns, per node, the exact absorbed
photon count of that node's dual cell divided by ``dx_cell``.  The numerators
telescope, so ``sum(G*dx_cell) == Phi*(1 - exp(-tau_total))`` **identically**,
for any mesh.  That is an algebraic identity, not a convergence statement, so
the only tolerance involved is floating-point round-off:

* ``RTOL_FP = 1e-14`` -- IEEE-754 double eps is 2.22e-16; the sum runs over at
  most ~500 cells under numpy's pairwise summation and each term makes one
  divide/multiply round trip through ``dx_cell``, so a few tens of eps is the
  honest bound.  Provenance: floating-point arithmetic, not this code's
  output.  (Measured worst case across the four presets and 11 meshes:
  2.1e-16, i.e. 1 ulp.)

The hard physical bound -- the device cannot create more electron-hole pairs
per second than there are incident photons, ``q*sum(G*dx_cell) <= q*Phi`` --
carries NO tolerance at all and is asserted as a strict inequality.

The pre-fix point-sampled quadrature violated that bound by +5.881 % at
N_grid = 30, +1.111 % at N_grid = 60 and +0.064 % at N_grid = 100 on
``configs/ionmonger_benchmark.yaml``; N_grid = 60 is what the shipped slow
regression runs and N_grid = 100 is the ``run_jv_sweep`` default, so the
coarse meshes below are not strawmen.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from perovskite_sim.constants import Q
from perovskite_sim.discretization.grid import Layer, multilayer_grid
from perovskite_sim.experiments.jv_sweep import _layer_node_counts
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.models.device import electrical_layers
from perovskite_sim.physics.generation import (
    beer_lambert_generation,
    dual_cell_widths,
)
from perovskite_sim.solver.mol import build_material_arrays

#: Round-off bound only -- see the module docstring for its provenance.
RTOL_FP = 1e-14

#: Every shipped Beer-Lambert preset.  All four put alpha != 0 only in the
#: absorber, which is exactly the sharply-peaked profile the old rectangle
#: rule over-counted (1/alpha = 76.9 nm against 400 nm of absorber).
CONFIGS = (
    "configs/ionmonger_benchmark.yaml",
    "configs/nip_MAPbI3.yaml",
    "configs/pin_MAPbI3.yaml",
    "configs/driftfusion_benchmark.yaml",
)

#: Deliberately spans well below the meshes anyone would ship: 10 and 15 are
#: unusably coarse, 30/60 are the regression meshes, 100 is the sweep default.
MESHES = (10, 15, 30, 40, 60, 100, 120)


def _grid(stack, N_grid: int) -> np.ndarray:
    elec = electrical_layers(stack)
    return multilayer_grid([
        Layer(l.thickness, n)
        for l, n in zip(elec, _layer_node_counts(stack, N_grid))
    ])


def _discrete_optical_depth(x: np.ndarray, alpha: np.ndarray) -> float:
    """tau_total under the same mid-face trapezoid the generation uses."""
    dx = np.diff(x)
    return float(np.sum(0.5 * (alpha[:-1] + alpha[1:]) * dx))


@pytest.fixture(scope="module")
def stacks():
    return {c: load_device_from_yaml(c) for c in CONFIGS}


@pytest.mark.parametrize("config", CONFIGS)
@pytest.mark.parametrize("N_grid", MESHES)
def test_absorbed_budget_telescopes_exactly(stacks, config, N_grid):
    """sum(G*dx_cell) == Phi*(1 - exp(-tau)) to round-off, on ANY mesh."""
    stack = stacks[config]
    x = _grid(stack, N_grid)
    mat = build_material_arrays(x, stack)
    G = beer_lambert_generation(x, mat.alpha, stack.Phi)
    absorbed = float(np.sum(G * mat.dx_cell))
    tau = _discrete_optical_depth(x, mat.alpha)
    expected = stack.Phi * (1.0 - math.exp(-tau))
    assert absorbed == pytest.approx(expected, rel=RTOL_FP), (
        f"{config} N_grid={N_grid}: the discrete absorbed-photon budget "
        f"{absorbed:.12e} does not telescope to Phi*(1-exp(-tau)) = "
        f"{expected:.12e}; the per-cell quadrature is no longer exact."
    )


@pytest.mark.parametrize("config", CONFIGS)
@pytest.mark.parametrize("N_grid", MESHES)
def test_never_exceeds_incident_photon_flux(stacks, config, N_grid):
    """HARD BOUND: no tolerance, no refinement argument, ever."""
    stack = stacks[config]
    x = _grid(stack, N_grid)
    mat = build_material_arrays(x, stack)
    G = beer_lambert_generation(x, mat.alpha, stack.Phi)
    absorbed = Q * float(np.sum(G * mat.dx_cell))
    ceiling = Q * stack.Phi
    assert absorbed <= ceiling, (
        f"{config} N_grid={N_grid}: q*sum(G*dx_cell) = {absorbed:.6f} A/m2 "
        f"exceeds the incident photon current q*Phi = {ceiling:.6f} A/m2 — "
        "the solver is creating carriers that have no photons to come from."
    )


@pytest.mark.parametrize("config", CONFIGS)
@pytest.mark.parametrize("N_grid", MESHES)
def test_generation_is_nonnegative(stacks, config, N_grid):
    stack = stacks[config]
    x = _grid(stack, N_grid)
    mat = build_material_arrays(x, stack)
    G = beer_lambert_generation(x, mat.alpha, stack.Phi)
    assert np.all(G >= 0.0), f"{config} N_grid={N_grid}: negative generation"


@pytest.mark.parametrize("config", CONFIGS)
def test_non_absorbing_layers_generate_exactly_zero(stacks, config):
    """Not "small" -- bit-exact zero, so a transport layer never photo-dopes.

    Both faces of a cell inside an alpha = 0 layer carry the identical
    floating-point optical depth, so the difference of exponentials is an
    exact zero rather than a cancellation residue.
    """
    stack = stacks[config]
    x = _grid(stack, 60)
    mat = build_material_arrays(x, stack)
    G = beer_lambert_generation(x, mat.alpha, stack.Phi)
    dark = mat.alpha == 0.0
    # Cells straddling an absorber boundary legitimately absorb, so only test
    # nodes whose whole dual cell sits in a dark layer (both neighbours dark).
    interior_dark = dark.copy()
    interior_dark[1:] &= dark[:-1]
    interior_dark[:-1] &= dark[1:]
    assert interior_dark.any(), "preset has no fully non-absorbing node"
    assert np.all(G[interior_dark] == 0.0), (
        f"{config}: generation leaked into a non-absorbing layer: "
        f"max = {float(np.max(G[interior_dark])):.6e}"
    )


def test_dual_cell_widths_match_the_solver(stacks):
    """The quadrature is only exact if it divides by the RHS's own weights."""
    for config, stack in stacks.items():
        for N_grid in MESHES:
            x = _grid(stack, N_grid)
            mat = build_material_arrays(x, stack)
            assert np.array_equal(dual_cell_widths(x), mat.dx_cell), (
                f"{config} N_grid={N_grid}: generation.dual_cell_widths has "
                "drifted from MaterialArrays.dx_cell — the photon-conserving "
                "quadrature is invalidated."
            )


@pytest.mark.parametrize("N", (5, 10, 50, 200, 1000))
def test_scalar_alpha_reproduces_the_closed_form(N):
    """Uniform single-layer slab: the budget is analytic, so this is exact.

    Phi*(1 - exp(-alpha*L)) with no discretisation error at ANY N -- the case
    where the old point-sampling was worst (a 5-node grid over 5 absorption
    lengths).
    """
    L, alpha, Phi = 400e-9, 1e7, 2.5e21
    x = np.linspace(0.0, L, N)
    G = beer_lambert_generation(x, alpha, Phi)
    absorbed = float(np.sum(G * dual_cell_widths(x)))
    expected = Phi * (1.0 - math.exp(-alpha * L))
    assert absorbed == pytest.approx(expected, rel=RTOL_FP)
    assert absorbed <= Phi


def test_absorbed_budget_converges_to_the_analytic_value(stacks):
    """The residual mesh error is the O(h) optical-depth trapezoid, not the
    quadrature, and it must contract under refinement.

    ``tau`` is accumulated with a mid-face trapezoid on the per-node ``alpha``
    array.  At a layer boundary the shared interface node carries one layer's
    alpha, so the two faces adjacent to it are averaged across the material
    step and ``tau_total`` picks up an O(h) error.  Nothing here asserts a
    magnitude -- only that successive refinements reduce |error|, which is the
    statement that the discretisation converges.
    """
    ladder = (30, 60, 120, 240)
    for config, stack in stacks.items():
        elec = electrical_layers(stack)
        tau_exact = sum(
            float(l.params.alpha) * float(l.thickness)
            for l in elec if l.params is not None
        )
        analytic = stack.Phi * (1.0 - math.exp(-tau_exact))
        errs = []
        for N_grid in ladder:
            x = _grid(stack, N_grid)
            mat = build_material_arrays(x, stack)
            G = beer_lambert_generation(x, mat.alpha, stack.Phi)
            errs.append(abs(float(np.sum(G * mat.dx_cell)) - analytic))
        assert all(b < a for a, b in zip(errs, errs[1:])), (
            f"{config}: absorbed-photon budget does not converge to the "
            f"closed-form value along N_grid={ladder}: "
            f"|err| = {[f'{e:.4e}' for e in errs]}"
        )
