"""The 1D and 2D solvers must integrate the SAME absorbed-photon spectrum.

They weight their BOUNDARY cells differently, and only one convention can
be right for a shared ``G``:

    1D  physics/generation.dual_cell_widths   w[0] = dy[0]      (FULL)
    2D  twod/continuity_2d.hy_cell            h[0] = dy[0] / 2  (half)

``physics/generation.py`` builds ``G`` so that ``G[i] * w[i]`` is the exact
absorbed-photon count of cell ``i``. Handing that array to the 2D RHS
unchanged gives the two boundary rows half the photons they absorbed.
``solver_2d._to_2d_dual_cell`` rescales them so the count survives the
change of weights.

Measured before the fix (absorbed-photon current, 1D vs 2D):

    ionmonger_benchmark   223.0581 vs 223.0581    0.000 %
    cigs_baseline         400.5442 vs 400.5442   -1e-7 %
    cSi_homojunction      432.5877 vs 432.4894   -0.023 %

Exact on every perovskite preset because ``alpha = 0`` in the outer layers
makes ``G[0] = G[-1] = 0`` — which is why the 2D parity gates never saw it
— and small but real on the stacks whose absorber touches a contact.
"""
from __future__ import annotations

import numpy as np
import pytest

from perovskite_sim.constants import Q
from perovskite_sim.discretization.grid import Layer, multilayer_grid
from perovskite_sim.experiments.jv_sweep import _layer_node_counts
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.models.device import electrical_layers
from perovskite_sim.physics.generation import (
    beer_lambert_generation, dual_cell_widths,
)
from perovskite_sim.solver.mol import build_material_arrays
from perovskite_sim.twod.solver_2d import _to_2d_dual_cell

# Two of these have an absorber touching a contact, so G is non-zero at a
# boundary node; ionmonger is the control where the defect is invisible.
_CONFIGS = (
    "configs/cigs_baseline.yaml",
    "configs/cSi_homojunction.yaml",
    "configs/ionmonger_benchmark.yaml",
)


def _hy_cell(y: np.ndarray) -> np.ndarray:
    """The 2D convention, mirroring twod/continuity_2d.py."""
    dy = np.diff(y)
    h = np.empty(y.shape[0], dtype=float)
    h[0] = dy[0] / 2.0
    h[-1] = dy[-1] / 2.0
    h[1:-1] = 0.5 * (dy[:-1] + dy[1:])
    return h


def _profile(cfg):
    stack = load_device_from_yaml(cfg)
    y = multilayer_grid([
        Layer(l.thickness, n) for l, n in
        zip(electrical_layers(stack), _layer_node_counts(stack, 30))
    ])
    mat = build_material_arrays(y, stack)
    if mat.alpha is None or float(stack.Phi) <= 0.0:
        pytest.skip(f"{cfg} has no Beer-Lambert generation")
    return y, beer_lambert_generation(y, mat.alpha, stack.Phi)


@pytest.mark.parametrize("cfg", _CONFIGS)
def test_2d_weights_recover_the_1d_photon_budget(cfg):
    """The invariant: same photons under either set of weights."""
    y, G = _profile(cfg)
    j_1d = Q * float(np.sum(G * dual_cell_widths(y)))
    j_2d = Q * float(np.sum(_to_2d_dual_cell(G) * _hy_cell(y)))
    assert j_1d > 0.0
    assert j_2d == pytest.approx(j_1d, rel=1e-12), (
        f"{cfg}: 2D integrates {j_2d:.6f} A/m^2 against 1D {j_1d:.6f}"
    )


@pytest.mark.parametrize("cfg", _CONFIGS)
def test_interior_rows_are_untouched(cfg):
    """Only the two boundary rows may be rescaled.

    Guards against a fix that quietly renormalises the whole profile and so
    hides a different error inside the same total.
    """
    y, G = _profile(cfg)
    np.testing.assert_array_equal(_to_2d_dual_cell(G)[1:-1], G[1:-1])


def test_the_defect_was_real_on_an_absorbing_boundary():
    """Without the rescale, cSi loses measurable photons.

    If this stops failing, either the preset changed or the conventions
    converged — in both cases the rationale above needs re-deriving rather
    than inheriting.
    """
    y, G = _profile("configs/cSi_homojunction.yaml")
    assert G[0] > 0.0, "cSi no longer absorbs at its first node"
    j_1d = Q * float(np.sum(G * dual_cell_widths(y)))
    j_2d_unfixed = Q * float(np.sum(G * _hy_cell(y)))
    rel = abs(j_2d_unfixed - j_1d) / j_1d
    assert rel > 1e-4, (
        f"unscaled 2D now matches 1D to {rel:.2e} — the mismatch this guards "
        "against is gone, re-derive the convention note"
    )


def test_degenerate_inputs_pass_through():
    """A 0- or 1-node profile has no dual cell; return it unchanged rather
    than indexing off the end."""
    for arr in (np.zeros(0), np.array([3.0])):
        np.testing.assert_array_equal(_to_2d_dual_cell(arr), arr)
